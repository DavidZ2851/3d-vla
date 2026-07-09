"""Train the standalone 2D MotionExpert on calvin_2d_motion_zarr.

Compact iteration-based loop (DDP-aware, bf16 autocast). Mirrors the config
layout of train.py (model/train/data sections) but drops the 3D-only
normalization machinery.
"""
import argparse
import datetime
import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

try:
    import wandb
except Exception:
    wandb = None

from .train import _build_motion_expert_config
from .zarr_dataset import MotionExpertQwenTokenizer
from .zarr_dataset_2d import (
    MotionExpert2DCollator,
    MotionExpert2DZarrDataset,
    compute_delta_uv_scale,
)
from .pipeline_2d import MotionExpert2DPipeline


def _load_config(path: str):
    with open(path, "r") as h:
        return json.load(h) if path.endswith(".json") else yaml.safe_load(h)


def _is_dist():
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _is_main():
    return int(os.environ.get("RANK", "0")) == 0


def _lr_at(step, base_lr, min_lr, warmup, total):
    if step < warmup:
        return base_lr * float(step + 1) / float(max(1, warmup))
    if total <= warmup:
        return base_lr
    p = (step - warmup) / float(max(1, total - warmup))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * min(1.0, p)))


def _infinite(loader, sampler):
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


@torch.no_grad()
def _eval_loss(module, loader, device, max_batches):
    # Use the UNWRAPPED module: running the DDP wrapper on rank-0 only would
    # trigger a per-forward buffer broadcast and deadlock the other ranks.
    module.eval()
    total, n = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = batch.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = module(batch)
        total += float(out["loss"].item())
        n += 1
    module.train()
    return total / max(1, n)


def main():
    parser = argparse.ArgumentParser(description="Train 2D MotionExpert.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    raw = _load_config(args.config)
    config = _build_motion_expert_config(raw["model"])
    train_cfg, data_cfg = raw["train"], raw["data"]
    if config.florence.model_name_or_path:
        os.environ["FLORENCE2_MODEL_PATH"] = config.florence.model_name_or_path

    distributed = _is_dist()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", init_method="env://", timeout=datetime.timedelta(hours=2))
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_main = _is_main()

    image_size = (config.input_height, config.input_width)
    train_set = MotionExpert2DZarrDataset(
        root=data_cfg["train_zarr"],
        instructions_path=data_cfg.get("train_instructions_path", data_cfg["instructions_path"]),
        image_size=image_size, future_steps=config.future_steps,
        image_pad_value=config.image_pad_value, max_samples=data_cfg.get("max_train_samples"),
    )
    val_set = None
    if data_cfg.get("val_zarr"):
        val_set = MotionExpert2DZarrDataset(
            root=data_cfg["val_zarr"],
            instructions_path=data_cfg.get("val_instructions_path",
                                          data_cfg.get("train_instructions_path", data_cfg["instructions_path"])),
            image_size=image_size, future_steps=config.future_steps,
            image_pad_value=config.image_pad_value, max_samples=data_cfg.get("max_val_samples"),
        )

    tokenizer = MotionExpertQwenTokenizer(
        backbone_type="florence2", model_name_or_path=config.florence.model_name_or_path,
        max_length=config.text_max_length,
    )
    collator = MotionExpert2DCollator(tokenizer)

    batch_size = int(train_cfg.get("batch_size", 32))
    num_workers = int(train_cfg.get("num_workers", 4))
    train_sampler = DistributedSampler(train_set, shuffle=True, drop_last=True) if distributed else None
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=train_sampler is None, sampler=train_sampler,
        num_workers=num_workers, collate_fn=collator, drop_last=True,
        pin_memory=bool(train_cfg.get("pin_memory", True)),
        persistent_workers=num_workers > 0,
    )
    val_loader = None
    if val_set is not None and is_main:
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, collate_fn=collator,
                                pin_memory=bool(train_cfg.get("pin_memory", True)))

    pipeline = MotionExpert2DPipeline.from_backbone(config)

    # UV-delta flow-target scale: from config, else estimated from data (main rank, broadcast).
    scale_cfg = raw["model"].get("traj_delta_uv_scale")
    if scale_cfg is not None:
        scale = torch.tensor([float(x) for x in scale_cfg], dtype=torch.float32)
    else:
        scale = torch.ones(2, dtype=torch.float32)
        if is_main:
            scale = compute_delta_uv_scale(train_set, max_samples=2000)
            print(f"[2d] estimated traj_delta_uv_scale = {scale.tolist()}", flush=True)
    if distributed:
        scale = scale.to(device)
        dist.broadcast(scale, src=0)
        scale = scale.cpu()
    pipeline.model.set_traj_delta_scale(scale)

    pipeline = pipeline.to(device)
    if distributed:
        pipeline = torch.nn.parallel.DistributedDataParallel(
            pipeline, device_ids=[local_rank], find_unused_parameters=True,
            # buffers (traj_delta_uv_scale, frozen florence) are identical across
            # ranks and never change in training; skip the per-forward broadcast.
            broadcast_buffers=False,
        )
    module = pipeline.module if hasattr(pipeline, "module") else pipeline

    params = [p for p in pipeline.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params, lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 5e-3)), betas=(0.9, 0.95),
    )
    base_lr = float(train_cfg.get("lr", 1e-4))
    min_lr = float(train_cfg.get("min_lr", 1e-5))
    warmup = int(train_cfg.get("lr_warmup_steps", 1000))
    total_iters = int(train_cfg.get("train_iters", 20000))
    clip = float(train_cfg.get("clip_grad_norm", 1.0))
    log_every = int(train_cfg.get("train_log_every", 100))
    val_every = int(train_cfg.get("val_every_iters", 2000))
    save_every = int(train_cfg.get("save_every_iters", 2000))
    val_batches = int(train_cfg.get("val_loss_batches", 100))
    out_dir = Path(train_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    run = None
    if is_main and wandb is not None:
        wcfg = train_cfg.get("wandb", {})
        if bool(wcfg.get("enabled", True)) and str(wcfg.get("mode", os.environ.get("WANDB_MODE", "online"))).lower() != "disabled":
            run = wandb.init(project=wcfg.get("project", "motion-expert"),
                             name=wcfg.get("name", out_dir.name), dir=str(out_dir),
                             mode=str(wcfg.get("mode", os.environ.get("WANDB_MODE", "online"))).lower(),
                             config=raw, resume="allow")

    # Slim checkpoints: persist only trainable params + buffers (the frozen
    # florence2 backbone reloads from HF in from_backbone()).
    def _slim_state_dict():
        keep = {n for n, p in module.named_parameters() if p.requires_grad}
        keep |= {n for n, _ in module.named_buffers()}
        return {k: v for k, v in module.state_dict().items() if k in keep}

    start_step = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        module.load_state_dict(ckpt["model_state_dict"], strict=False)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt.get("step", 0))
        if is_main:
            print(f"[2d] resumed from {args.resume} at step {start_step}", flush=True)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if is_main:
        n_train = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"[2d] world_size={world_size} trainable params={n_train:,} "
              f"train_items={len(train_set)} val_items={len(val_set) if val_set else 0} "
              f"batch_per_gpu={batch_size} global_batch={batch_size * world_size} "
              f"iters={total_iters}", flush=True)

    pipeline.train()
    data_iter = _infinite(train_loader, train_sampler)
    for step in range(start_step, total_iters):
        lr = _lr_at(step, base_lr, min_lr, warmup, total_iters)
        for g in optimizer.param_groups:
            g["lr"] = lr
        batch = next(data_iter).to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = pipeline(batch)
            loss = out["loss"]
        loss.backward()
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(params, clip)
        optimizer.step()

        if is_main and (step % log_every == 0 or step == total_iters - 1):
            msg = (f"[2d] step {step}/{total_iters} loss={float(loss.item()):.5f} "
                   f"flow={float(out.get('flow_loss', loss).item()):.5f} lr={lr:.2e}")
            print(msg, flush=True)
            if run is not None:
                wandb.log({"train/loss": float(loss.item()),
                           "train/flow_loss": float(out.get("flow_loss", loss).item()),
                           "train/lr": lr}, step=step)

        if is_main and val_loader is not None and (step + 1) % val_every == 0:
            vl = _eval_loss(module, val_loader, device, val_batches)
            print(f"[2d] step {step} val_loss={vl:.5f}", flush=True)
            if run is not None:
                wandb.log({"val/loss": vl}, step=step)

        if is_main and (step + 1) % save_every == 0:
            slim = _slim_state_dict()
            ckpt_path = out_dir / f"checkpoint_step_{step + 1}.pt"
            torch.save({"model_state_dict": slim, "optimizer": optimizer.state_dict(),
                        "step": step + 1, "config": raw}, ckpt_path)
            torch.save({"model_state_dict": slim, "step": step + 1, "config": raw},
                       out_dir / "last.pt")
            print(f"[2d] saved {ckpt_path}", flush=True)
        # No per-step barrier: ranks stay in lockstep via the backward all-reduce;
        # rank-0's val/save are backward-free extra work the others wait through at
        # the next all-reduce.

    if distributed and dist.is_initialized():
        dist.barrier(device_ids=[torch.cuda.current_device()])
    if is_main:
        torch.save({"model_state_dict": _slim_state_dict(), "step": total_iters, "config": raw},
                   out_dir / "last.pt")
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
