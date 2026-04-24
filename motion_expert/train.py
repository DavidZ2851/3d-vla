import argparse
import json
import os
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from .config import (
    FlorenceFeatureExtractorConfig,
    MotionExpertConfig,
    QwenFeatureExtractorConfig,
    TraceGenDecoderConfig,
)
from .pipeline import MotionExpertPipeline
from .trainer import MotionExpertTrainer
from .zarr_dataset import (
    MotionExpertCollator,
    MotionExpertQwenTokenizer,
    MotionExpertZarrDataset,
    compute_query_xyz_statistics,
)


def _load_config(path: str):
    with open(path, "r") as handle:
        if path.endswith(".json"):
            return json.load(handle)
        return yaml.safe_load(handle)


def _build_motion_expert_config(raw_cfg: dict) -> MotionExpertConfig:
    qwen_cfg = raw_cfg.get("qwen", {})
    florence_cfg = raw_cfg.get("florence", {})
    decoder_cfg = raw_cfg.get("decoder", {})
    return MotionExpertConfig(
        backbone_type=raw_cfg.get("backbone_type", "qwen"),
        grid_size=raw_cfg.get("grid_size", 20),
        future_steps=raw_cfg.get("future_steps", 16),
        input_height=raw_cfg.get("input_height", 384),
        input_width=raw_cfg.get("input_width", 640),
        text_max_length=raw_cfg.get("text_max_length", 128),
        image_pad_value=raw_cfg.get("image_pad_value", 0.5),
        query_uv_is_normalized=raw_cfg.get("query_uv_is_normalized", False),
        dropout=raw_cfg.get("dropout", 0.1),
        query_xyz_center=tuple(raw_cfg.get("query_xyz_center", [0.0, 0.0, 0.0])),
        query_xyz_scale=tuple(raw_cfg.get("query_xyz_scale", [1.0, 1.0, 1.0])),
        delta_range_xyz=tuple(raw_cfg.get("delta_range_xyz", [0.05, 0.05, 0.04])),
        qwen=QwenFeatureExtractorConfig(**qwen_cfg),
        florence=FlorenceFeatureExtractorConfig(**florence_cfg),
        decoder=TraceGenDecoderConfig(**decoder_cfg),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train MotionExpert.")
    parser.add_argument("--config", type=str, required=True, help="YAML or JSON config.")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path.")
    return parser.parse_args()


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    raw_cfg = _load_config(args.config)
    model_cfg = _build_motion_expert_config(raw_cfg["model"])
    train_cfg = raw_cfg["train"]
    data_cfg = raw_cfg["data"]

    if model_cfg.qwen.model_name_or_path is not None:
        os.environ["QWEN3VL_MODEL_PATH"] = model_cfg.qwen.model_name_or_path
    if model_cfg.florence.model_name_or_path is not None:
        os.environ["FLORENCE2_MODEL_PATH"] = model_cfg.florence.model_name_or_path

    device = torch.device(train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    backbone_type = str(model_cfg.backbone_type).lower()
    tokenizer_model_path = (
        model_cfg.qwen.model_name_or_path
        if backbone_type == "qwen"
        else model_cfg.florence.model_name_or_path
    )
    tokenizer = MotionExpertQwenTokenizer(
        backbone_type=backbone_type,
        model_name_or_path=tokenizer_model_path,
        max_length=model_cfg.text_max_length,
    )
    collator = MotionExpertCollator(tokenizer)

    train_dataset = MotionExpertZarrDataset(
        root=data_cfg["train_zarr"],
        instructions_path=data_cfg["instructions_path"],
        image_size=(model_cfg.input_height, model_cfg.input_width),
        instruction_id_key=data_cfg.get("instruction_id_key"),
        query_uv_is_normalized=model_cfg.query_uv_is_normalized,
        image_pad_value=model_cfg.image_pad_value,
        mem_limit_gb=float(data_cfg.get("mem_limit_gb", 16.0)),
    )
    val_dataset = None
    if data_cfg.get("val_zarr"):
        val_dataset = MotionExpertZarrDataset(
            root=data_cfg["val_zarr"],
            instructions_path=data_cfg["instructions_path"],
            image_size=(model_cfg.input_height, model_cfg.input_width),
            instruction_id_key=data_cfg.get("instruction_id_key"),
            query_uv_is_normalized=model_cfg.query_uv_is_normalized,
            image_pad_value=model_cfg.image_pad_value,
            mem_limit_gb=float(data_cfg.get("mem_limit_gb", 16.0)),
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg.get("batch_size", 2)),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 4)),
        pin_memory=bool(train_cfg.get("pin_memory", False)),
        collate_fn=collator,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(train_cfg.get("batch_size", 2)),
            shuffle=False,
            num_workers=int(train_cfg.get("num_workers", 4)),
            pin_memory=bool(train_cfg.get("pin_memory", False)),
            collate_fn=collator,
        )

    pipeline = MotionExpertPipeline.from_backbone(model_cfg)

    stats_path = Path(train_cfg.get("query_stats_path", Path(train_cfg["output_dir"]) / "query_xyz_stats.pt"))
    if stats_path.exists():
        stats = torch.load(stats_path, map_location="cpu")
        query_center = stats["center"]
        query_scale = stats["scale"]
    else:
        query_center, query_scale = compute_query_xyz_statistics(
            [data_cfg["train_zarr"]],
            mem_limit_gb=float(data_cfg.get("mem_limit_gb", 16.0)),
        )
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"center": query_center, "scale": query_scale}, stats_path)
    pipeline.set_query_xyz_statistics(query_center, query_scale)

    trainer = MotionExpertTrainer(
        pipeline=pipeline,
        config=model_cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=train_cfg["output_dir"],
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 5e-3)),
        clip_grad_norm=float(train_cfg.get("clip_grad_norm", 1.0)),
        mixed_precision=bool(train_cfg.get("mixed_precision", False)),
    )

    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)

    epochs = int(train_cfg.get("epochs", 1))
    save_every = int(train_cfg.get("save_every", 1))
    val_every = int(train_cfg.get("val_every", 1))
    inference_steps = int(train_cfg.get("val_num_inference_steps", 20))
    guidance_scale = float(train_cfg.get("guidance_scale", 1.0))
    val_gen_batches = int(train_cfg.get("val_generation_batches", 1))

    for epoch in range(start_epoch, epochs):
        train_metrics = trainer.train_epoch()
        print(f"[train] epoch={epoch} loss={train_metrics['loss']:.6f}")

        if val_loader is not None and ((epoch + 1) % val_every == 0 or epoch == epochs - 1):
            val_loss = trainer.evaluate_loss()
            val_gen = trainer.evaluate_generation(
                num_inference_steps=inference_steps,
                guidance_scale=guidance_scale,
                max_batches=val_gen_batches,
            )
            val_loss_value = val_loss.get("loss", float("inf"))
            print(f"[val] epoch={epoch} loss={val_loss_value:.6f} metrics={val_gen}")
            is_best = val_loss_value < trainer.best_val_loss
            if is_best:
                trainer.best_val_loss = val_loss_value
            if (epoch + 1) % save_every == 0 or is_best:
                trainer.save_checkpoint(epoch, is_best=is_best)
        elif (epoch + 1) % save_every == 0:
            trainer.save_checkpoint(epoch, is_best=False)


if __name__ == "__main__":
    main()
