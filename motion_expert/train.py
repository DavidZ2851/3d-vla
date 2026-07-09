import argparse
import datetime
import json
import os
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils.common_utils import count_parameters

try:
    import wandb
except Exception:
    wandb = None

from .compute_calvin_normalization_stats import (
    compute_calvin_normalization_stats,
    extract_calvin_normalization_tensors,
    load_calvin_normalization_stats,
    save_calvin_normalization_stats,
)
from .config import (
    FlorenceFeatureExtractorConfig,
    MotionExpertConfig,
    SparseMotionDecoderConfig,
    QwenFeatureExtractorConfig,
    TraceGenDecoderConfig,
)
from .pipeline import MotionExpertPipeline
from .trainer import MotionExpertTrainer
from .zarr_dataset import (
    MotionExpertCollator,
    MotionExpertQwenTokenizer,
    MotionExpertZarrDataset,
    compute_calvin_zarr_normalization_stats,
    compute_query_xyz_statistics,
    compute_query_xyz_statistics_from_dataset,
)

DEFAULT_CALVIN_NORMALIZATION_STATS_PATH = Path(
    "/data/user_data/ziyinx/analogical_manipulation_clean/datasets/calvin_motion_normalization_stats.json"
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
    sparse_decoder_cfg = raw_cfg.get("sparse_decoder", {})
    if "query_uv_is_normalized" in raw_cfg:
        raise ValueError(
            "`query_uv_is_normalized` is deprecated and ambiguous. "
            "Use `query_uv_format` with one of: source_pixels, source_unit, model_grid."
        )
    return MotionExpertConfig(
        backbone_type=raw_cfg.get("backbone_type", "qwen"),
        grid_size=raw_cfg.get("grid_size", 20),
        future_steps=raw_cfg.get("future_steps", 15),
        decoder_type=raw_cfg.get("decoder_type", "sparse_factorized"),
        knn_k=raw_cfg.get("knn_k", 16),
        flow_timesteps=raw_cfg.get("flow_timesteps", 100),
        use_motion_weighted_loss=raw_cfg.get("use_motion_weighted_loss", True),
        motion_weight_tau=raw_cfg.get("motion_weight_tau", 0.005),
        motion_weight_temp=raw_cfg.get("motion_weight_temp", 5.0),
        motion_weight_gamma=raw_cfg.get("motion_weight_gamma", 1.0),
        motion_weight_static_floor=raw_cfg.get("motion_weight_static_floor", 0.05),
        input_height=raw_cfg.get("input_height", 384),
        input_width=raw_cfg.get("input_width", 640),
        text_max_length=raw_cfg.get("text_max_length", 16),
        image_pad_value=raw_cfg.get("image_pad_value", 0.5),
        query_uv_format=raw_cfg.get("query_uv_format", "source_pixels"),
        checkpoint_activations=raw_cfg.get("checkpoint_activations", True),
        dropout=raw_cfg.get("dropout", 0.1),
        use_calvin_percentile_normalization=raw_cfg.get(
            "use_calvin_percentile_normalization",
            False,
        ),
        query_xyz_center=tuple(raw_cfg.get("query_xyz_center", [0.0, 0.0, 0.0])),
        query_xyz_scale=tuple(raw_cfg.get("query_xyz_scale", [1.0, 1.0, 1.0])),
        delta_range_xyz=tuple(raw_cfg.get("delta_range_xyz", [0.05, 0.05, 0.04])),
        query_centered_xyz_center=tuple(raw_cfg.get("query_centered_xyz_center", [0.0, 0.0, 0.0])),
        query_centered_xyz_scale=tuple(raw_cfg.get("query_centered_xyz_scale", [1.0, 1.0, 1.0])),
        traj_delta_xyz_center=tuple(raw_cfg.get("traj_delta_xyz_center", [0.0, 0.0, 0.0])),
        traj_delta_xyz_scale=tuple(raw_cfg.get("traj_delta_xyz_scale", [1.0, 1.0, 1.0])),
        sparse_decoder=SparseMotionDecoderConfig(**sparse_decoder_cfg),
        qwen=QwenFeatureExtractorConfig(**qwen_cfg),
        florence=FlorenceFeatureExtractorConfig(**florence_cfg),
        decoder=TraceGenDecoderConfig(**decoder_cfg),
    )


def _looks_like_calvin_source(*values: object) -> bool:
    joined = " ".join(str(v) for v in values if v is not None).lower()
    return "calvin" in joined


def parse_args():
    parser = argparse.ArgumentParser(description="Train MotionExpert.")
    parser.add_argument("--config", type=str, required=True, help="YAML or JSON config.")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path.")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Load a checkpoint and run validation only.",
    )
    return parser.parse_args()


def _init_wandb(
    raw_cfg: dict,
    train_cfg: dict,
    validate_only: bool = False,
):
    wandb_cfg = train_cfg.get("wandb", {})
    enabled = bool(wandb_cfg.get("enabled", True))
    mode = str(wandb_cfg.get("mode", os.environ.get("WANDB_MODE", "online"))).lower()
    if not enabled or mode == "disabled" or wandb is None:
        return None

    output_dir = Path(train_cfg["output_dir"])
    run_name = wandb_cfg.get("name") or output_dir.name
    if validate_only:
        run_name = f"{run_name}-val"

    return wandb.init(
        project=wandb_cfg.get("project", "motion-expert"),
        name=run_name,
        dir=str(wandb_cfg.get("dir", output_dir)),
        mode=mode,
        config=raw_cfg,
        group=wandb_cfg.get("group"),
        tags=wandb_cfg.get("tags"),
        resume="allow",
    )


def _log_wandb_metrics(metrics: Dict[str, float], step: int) -> None:
    if wandb is None or wandb.run is None or not metrics:
        return
    wandb.log(metrics, step=step)


def _prefixed(metrics: Dict[str, float], prefix: str) -> Dict[str, float]:
    return {f"{prefix}/{key}": value for key, value in metrics.items()}


def _update_running_metrics(
    totals: Dict[str, float],
    metrics: Dict[str, float],
) -> None:
    for key, value in metrics.items():
        totals[key] = totals.get(key, 0.0) + float(value)


def _average_running_metrics(
    totals: Dict[str, float],
    count: int,
) -> Dict[str, float]:
    if count <= 0:
        return {}
    return {key: value / float(count) for key, value in totals.items()}


def _count_params(module: torch.nn.Module, trainable_only: bool = False) -> int:
    return sum(
        param.numel()
        for param in module.parameters()
        if (param.requires_grad or not trainable_only)
    )


def _format_params(num_params: int) -> str:
    return f"{num_params:,}"


def _print_param_row(name: str, module: Optional[torch.nn.Module]) -> None:
    if module is None:
        return
    total = _count_params(module)
    trainable = _count_params(module, trainable_only=True)
    frozen = total - trainable
    print(
        f"[params] {name}: total={_format_params(total)} "
        f"trainable={_format_params(trainable)} frozen={_format_params(frozen)}"
    )


def _chunk_loader_batch_size(raw_batch_size: int, chunk_size: int) -> int:
    return max(1, int(raw_batch_size) // max(1, int(chunk_size)))


def _log_motion_expert_summary(
    pipeline: MotionExpertPipeline,
    model_cfg: MotionExpertConfig,
    train_cfg: dict,
    train_dataset: MotionExpertZarrDataset,
    val_dataset: Optional[MotionExpertZarrDataset],
) -> None:
    print("[params] detailed MotionExpert parameter summary")
    _print_param_row("pipeline", pipeline)
    _print_param_row("feature_extractor", pipeline.feature_extractor)
    _print_param_row("model", pipeline.model)

    feature_extractor = pipeline.feature_extractor
    if feature_extractor is not None:
        _print_param_row("feature_extractor.backbone", getattr(feature_extractor, "backbone", None))
        backbone = getattr(feature_extractor, "backbone", None)
        vlm = getattr(backbone, "vlm", None)
        _print_param_row("feature_extractor.backbone.vlm", vlm)
        _print_param_row("feature_extractor.backbone.vlm.visual", getattr(vlm, "visual", None))
        language_model = getattr(vlm, "language_model", None)
        _print_param_row("feature_extractor.backbone.vlm.language_model", language_model)
        layers = getattr(language_model, "layers", None)
        if layers is not None:
            print(f"[params] feature_extractor.backbone.vlm.language_model.layers={len(layers)}")
        _print_param_row("feature_extractor.text_norm", getattr(feature_extractor, "text_norm", None))

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    raw_batch_size = int(train_cfg.get("batch_size", 2))
    chunk_size = int(getattr(train_dataset, "chunk_size", 1))
    loader_batch_size = _chunk_loader_batch_size(raw_batch_size, chunk_size)
    raw_per_gpu = loader_batch_size * chunk_size
    raw_global = raw_per_gpu * max(world_size, 1)
    motion_tokens_per_sample = int(model_cfg.num_query_tokens * model_cfg.future_steps)
    print(
        f"[data] train_chunk_items={len(train_dataset)} "
        f"train_raw_samples={getattr(train_dataset, '_effective_raw_length', 'unknown')} "
        f"chunk_size={chunk_size} configured_raw_batch_size={raw_batch_size} "
        f"loader_batch_size_chunks={loader_batch_size} "
        f"raw_samples_per_gpu={raw_per_gpu} raw_samples_global={raw_global} "
        f"world_size={world_size}"
    )
    if val_dataset is not None:
        print(
            f"[data] val_chunk_items={len(val_dataset)} "
            f"val_raw_samples={getattr(val_dataset, '_effective_raw_length', 'unknown')}"
        )
    print(
        f"[data] motion_tokens_per_sample={motion_tokens_per_sample} "
        f"nominal_motion_tokens_per_gpu={raw_per_gpu * motion_tokens_per_sample}"
    )
    print(f"[data] checkpoint_activations={bool(model_cfg.checkpoint_activations)}")


def _is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _setup_distributed() -> tuple[torch.device, int, bool]:
    distributed = _is_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=datetime.timedelta(hours=2),
        )
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device, local_rank, distributed


def _barrier_if_distributed(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.barrier(device_ids=[torch.cuda.current_device()])


def _load_or_compute_motion_normalization_stats(
    stats_path: Path,
    train_cfg: dict,
    data_cfg: dict,
    *,
    is_main_process: bool,
    distributed: bool,
) -> Dict[str, object]:
    if stats_path.exists():
        return load_calvin_normalization_stats(stats_path)

    if is_main_process:
        num_bins = int(train_cfg.get("normalization_num_bins", 131072))
        source_root_cfg = train_cfg.get("normalization_source_root")
        if source_root_cfg is not None:
            stats = compute_calvin_normalization_stats(
                source_root=Path(source_root_cfg),
                num_bins=num_bins,
            )
        else:
            stats = compute_calvin_zarr_normalization_stats(
                [data_cfg["train_zarr"]],
                mem_limit_gb=float(data_cfg.get("mem_limit_gb", 16.0)),
                chunk_size=int(train_cfg.get("normalization_chunk_size", 512)),
                max_samples=train_cfg.get("normalization_max_samples"),
            )
        save_calvin_normalization_stats(stats, stats_path)

    _barrier_if_distributed(distributed)
    if not stats_path.exists():
        raise FileNotFoundError(
            f"Normalization stats were not created by the main process: {stats_path}"
        )
    return load_calvin_normalization_stats(stats_path)


def _load_or_compute_query_xyz_stats(
    stats_path: Path,
    train_dataset: MotionExpertZarrDataset,
    data_cfg: dict,
    *,
    is_main_process: bool,
    distributed: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if stats_path.exists():
        stats = torch.load(stats_path, map_location="cpu")
        return stats["center"], stats["scale"]

    if is_main_process:
        if "query_world_xyz" in train_dataset.annos:
            query_center, query_scale = compute_query_xyz_statistics(
                [data_cfg["train_zarr"]],
                mem_limit_gb=float(data_cfg.get("mem_limit_gb", 16.0)),
            )
        else:
            query_center, query_scale = compute_query_xyz_statistics_from_dataset(
                train_dataset
            )
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"center": query_center, "scale": query_scale}, stats_path)

    _barrier_if_distributed(distributed)
    if not stats_path.exists():
        raise FileNotFoundError(
            f"Query XYZ stats were not created by the main process: {stats_path}"
        )
    stats = torch.load(stats_path, map_location="cpu")
    return stats["center"], stats["scale"]


def _build_spaghetti_viz_kwargs(
    train_cfg: dict,
    *,
    output_dir: str,
    prefix: str,
) -> Dict[str, object]:
    max_samples = int(train_cfg.get("spaghetti_viz_max_samples", 0))
    if max_samples <= 0:
        return {}
    viz_dir = Path(
        train_cfg.get(
            "spaghetti_viz_dir",
            Path(output_dir) / "spaghetti_viz",
        )
    )
    viz_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir = Path(
        train_cfg.get(
            "spaghetti_rgb_viz_dir",
            viz_dir / "rgb",
        )
    )
    rgb_dir.mkdir(parents=True, exist_ok=True)
    return {
        "spaghetti_viz_output_dir": str(viz_dir),
        "spaghetti_viz_prefix": str(prefix),
        "spaghetti_viz_max_samples": max_samples,
        "spaghetti_motion_threshold": float(train_cfg.get("spaghetti_motion_threshold", 0.005)),
        "spaghetti_max_points_per_sample": int(train_cfg.get("spaghetti_max_points_per_sample", 200)),
        "spaghetti_rgb_output_dir": str(rgb_dir),
    }


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

    device, local_rank, distributed = _setup_distributed()
    is_main_process = _is_main_process()
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
        instructions_path=data_cfg.get("train_instructions_path", data_cfg["instructions_path"]),
        image_size=(model_cfg.input_height, model_cfg.input_width),
        instruction_id_key=data_cfg.get("instruction_id_key"),
        query_uv_format=model_cfg.query_uv_format,
        image_pad_value=model_cfg.image_pad_value,
        mem_limit_gb=float(data_cfg.get("mem_limit_gb", 16.0)),
        traj_3d_xyz_is_delta=data_cfg.get("traj_3d_xyz_is_delta"),
        max_samples=data_cfg.get("max_train_samples"),
        chunk_size=int(data_cfg.get("chunk_size", 1)),
    )
    val_dataset = None
    if data_cfg.get("val_zarr"):
        val_dataset = MotionExpertZarrDataset(
            root=data_cfg["val_zarr"],
            instructions_path=data_cfg.get(
                "val_instructions_path",
                data_cfg.get("train_instructions_path", data_cfg["instructions_path"]),
            ),
            image_size=(model_cfg.input_height, model_cfg.input_width),
            instruction_id_key=data_cfg.get("instruction_id_key"),
            query_uv_format=model_cfg.query_uv_format,
            image_pad_value=model_cfg.image_pad_value,
            mem_limit_gb=float(data_cfg.get("mem_limit_gb", 16.0)),
            traj_3d_xyz_is_delta=data_cfg.get("traj_3d_xyz_is_delta"),
            max_samples=data_cfg.get("max_val_samples"),
            chunk_size=int(data_cfg.get("chunk_size", 1)),
        )

    train_sampler = None
    if distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=_chunk_loader_batch_size(
            int(train_cfg.get("batch_size", 2)),
            int(data_cfg.get("chunk_size", 1)),
        ),
        shuffle=train_sampler is None,
        num_workers=int(train_cfg.get("num_workers", 4)),
        pin_memory=bool(train_cfg.get("pin_memory", False)),
        collate_fn=collator,
        sampler=train_sampler,
        persistent_workers=bool(train_cfg.get("persistent_workers", False))
        and int(train_cfg.get("num_workers", 4)) > 0,
    )
    train_eval_loader = None
    if is_main_process:
        train_eval_loader = DataLoader(
            train_dataset,
            batch_size=_chunk_loader_batch_size(
                int(train_cfg.get("batch_size", 2)),
                int(data_cfg.get("chunk_size", 1)),
            ),
            shuffle=False,
            num_workers=int(train_cfg.get("num_workers", 4)),
            pin_memory=bool(train_cfg.get("pin_memory", False)),
            collate_fn=collator,
            persistent_workers=bool(train_cfg.get("persistent_workers", False))
            and int(train_cfg.get("num_workers", 4)) > 0,
        )
    val_loader = None
    if val_dataset is not None and is_main_process:
        val_loader = DataLoader(
            val_dataset,
            batch_size=_chunk_loader_batch_size(
                int(train_cfg.get("batch_size", 2)),
                int(data_cfg.get("chunk_size", 1)),
            ),
            shuffle=False,
            num_workers=int(train_cfg.get("num_workers", 4)),
            pin_memory=bool(train_cfg.get("pin_memory", False)),
            collate_fn=collator,
            persistent_workers=bool(train_cfg.get("persistent_workers", False))
            and int(train_cfg.get("num_workers", 4)) > 0,
        )

    pipeline = MotionExpertPipeline.from_backbone(model_cfg)
    if is_main_process:
        count_parameters(pipeline)
        _log_motion_expert_summary(
            pipeline=pipeline,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        )

    use_calvin_normalization = bool(
        train_cfg.get(
            "use_calvin_normalization",
            _looks_like_calvin_source(
                data_cfg.get("train_zarr"),
                data_cfg.get("train_instructions_path", data_cfg.get("instructions_path")),
            ),
        )
    )
    if use_calvin_normalization:
        default_stats_path = (
            DEFAULT_CALVIN_NORMALIZATION_STATS_PATH
            if _looks_like_calvin_source(
                data_cfg.get("train_zarr"),
                data_cfg.get("train_instructions_path", data_cfg.get("instructions_path")),
            )
            else Path(train_cfg["output_dir"]) / "calvin_normalization_stats.json"
        )
        stats_path = Path(
            train_cfg.get(
                "normalization_stats_path",
                default_stats_path,
            )
        )
        stats = _load_or_compute_motion_normalization_stats(
            stats_path,
            train_cfg,
            data_cfg,
            is_main_process=is_main_process,
            distributed=distributed,
        )
        query_center, query_scale, traj_center, traj_scale = extract_calvin_normalization_tensors(stats)
        pipeline.set_calvin_normalization_statistics(
            torch.from_numpy(query_center),
            torch.from_numpy(query_scale),
            torch.from_numpy(traj_center),
            torch.from_numpy(traj_scale),
        )
    else:
        stats_path = Path(
            train_cfg.get(
                "query_stats_path",
                Path(train_cfg["output_dir"]) / "query_xyz_stats.pt",
            )
        )
        query_center, query_scale = _load_or_compute_query_xyz_stats(
            stats_path,
            train_dataset,
            data_cfg,
            is_main_process=is_main_process,
            distributed=distributed,
        )
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
        local_rank=local_rank,
        distributed=distributed,
        find_unused_parameters=bool(train_cfg.get("find_unused_parameters", False)),
        train_sampler=train_sampler,
    )
    train_iters = int(
        train_cfg.get(
            "train_iters",
            int(train_cfg.get("epochs", 1)) * max(len(train_loader), 1),
        )
    )
    scheduler_type = str(train_cfg.get("lr_scheduler", "cosine_with_warmup")).lower()
    warmup_steps = int(train_cfg.get("lr_warmup_steps", max(500, int(0.03 * train_iters))))
    min_lr = float(train_cfg.get("min_lr", max(1e-6, 0.1 * float(train_cfg.get("lr", 1e-4)))))
    trainer.configure_lr_scheduler(
        total_steps=train_iters,
        scheduler_type=scheduler_type,
        warmup_steps=warmup_steps,
        min_lr=min_lr,
    )

    start_step = 0
    if args.resume:
        start_step = trainer.load_checkpoint(args.resume)
    _barrier_if_distributed(distributed)

    wandb_run = _init_wandb(raw_cfg, train_cfg, validate_only=args.validate_only) if is_main_process else None

    if args.validate_only:
        if not args.resume:
            raise ValueError("`--validate-only` requires `--resume` to evaluate a trained checkpoint.")
        if is_main_process and val_loader is None:
            raise ValueError("Validation-only mode requires `data.val_zarr`.")
        if is_main_process:
            val_loss_batches = int(train_cfg.get("val_loss_batches", 0))
            val_loss = trainer.evaluate_loss(
                max_batches=val_loss_batches if val_loss_batches >= 0 else None,
            )
            val_viz_kwargs = _build_spaghetti_viz_kwargs(
                train_cfg,
                output_dir=train_cfg["output_dir"],
                prefix=f"val_only_step_{start_step}",
            )
            val_gen = trainer.evaluate_generation(
                num_inference_steps=int(train_cfg.get("val_num_inference_steps", 20)),
                guidance_scale=float(train_cfg.get("guidance_scale", 1.0)),
                max_batches=int(train_cfg.get("val_generation_batches", 1)),
                **val_viz_kwargs,
            )
            metrics = {}
            metrics.update(_prefixed(val_loss, "val"))
            metrics.update(_prefixed(val_gen, "val"))
            print(f"[val-only] checkpoint={args.resume} step={start_step} metrics={metrics}")
            _log_wandb_metrics(metrics, step=start_step)
            if wandb_run is not None:
                wandb_run.summary.update(metrics)
                wandb.finish()
        _barrier_if_distributed(distributed)
        if distributed and dist.is_initialized():
            dist.destroy_process_group()
        return

    train_log_every = int(train_cfg.get("train_log_every", 100))
    train_eval_every = int(train_cfg.get("train_eval_every", train_cfg.get("val_freq", 0)))
    train_eval_loss_batches = int(train_cfg.get("train_eval_loss_batches", 10))
    train_eval_generation_batches = int(train_cfg.get("train_eval_generation_batches", 2))
    val_every = int(train_cfg.get("val_every_iters", 0))
    val_loss_batches = int(train_cfg.get("val_loss_batches", 500))
    save_every = int(
        train_cfg.get(
            "save_every_iters",
            val_every if val_every > 0 else train_eval_every,
        )
    )
    archive_ckpt_start_step = int(train_cfg.get("archive_ckpt_start_step", 10000))
    inference_steps = int(train_cfg.get("val_num_inference_steps", 20))
    guidance_scale = float(train_cfg.get("guidance_scale", 1.0))
    val_gen_batches = int(train_cfg.get("val_generation_batches", 1))
    running_train: Dict[str, float] = {}
    running_train_count = 0

    for step in range(start_step, train_iters):
        completed_step = step + 1
        train_metrics = trainer.train_step()
        _update_running_metrics(running_train, train_metrics)
        running_train_count += 1

        if train_log_every > 0 and (
            completed_step % train_log_every == 0 or completed_step == train_iters
        ):
            if is_main_process:
                avg_train = _average_running_metrics(running_train, running_train_count)
                print(
                    f"[train] step={completed_step}/{train_iters} "
                    f"loss={avg_train.get('loss', 0.0):.6f}"
                )
                log_payload = {
                    "step": float(completed_step),
                    "train/lr": float(trainer.optimizer.param_groups[0]["lr"]),
                    **_prefixed(avg_train, "train"),
                }
                _log_wandb_metrics(log_payload, step=completed_step)
            running_train = {}
            running_train_count = 0

        if is_main_process and train_eval_every > 0 and completed_step % train_eval_every == 0:
            if train_eval_loader is None:
                raise RuntimeError("train_eval_loader is unavailable on main process.")
            train_eval_loss = trainer.evaluate_loss(
                loader=train_eval_loader,
                max_batches=train_eval_loss_batches,
            )
            train_eval_viz_kwargs = _build_spaghetti_viz_kwargs(
                train_cfg,
                output_dir=train_cfg["output_dir"],
                prefix=f"train_eval_step_{completed_step}",
            )
            train_eval_gen = trainer.evaluate_generation(
                loader=train_eval_loader,
                num_inference_steps=inference_steps,
                guidance_scale=guidance_scale,
                max_batches=train_eval_generation_batches,
                **train_eval_viz_kwargs,
            )
            train_eval_metrics = {}
            train_eval_metrics.update(_prefixed(train_eval_loss, "train_eval"))
            train_eval_metrics.update(_prefixed(train_eval_gen, "train_eval"))
            print(
                f"[train-eval] step={completed_step} "
                f"loss={train_eval_loss.get('loss', 0.0):.6f} "
                f"endpoint_acc_001={train_eval_gen.get('endpoint_acc_001', 0.0):.4f}"
            )
            _log_wandb_metrics(train_eval_metrics, step=completed_step)

        ran_val = False
        if (
            is_main_process
            and val_loader is not None
            and val_every > 0
            and completed_step % val_every == 0
        ):
            should_archive_step = completed_step >= archive_ckpt_start_step
            val_loss = trainer.evaluate_loss(
                loader=val_loader,
                max_batches=val_loss_batches,
            )
            val_viz_kwargs = _build_spaghetti_viz_kwargs(
                train_cfg,
                output_dir=train_cfg["output_dir"],
                prefix=f"val_step_{completed_step}",
            )
            val_gen = trainer.evaluate_generation(
                loader=val_loader,
                num_inference_steps=inference_steps,
                guidance_scale=guidance_scale,
                max_batches=val_gen_batches,
                **val_viz_kwargs,
            )
            val_loss_value = val_loss.get("loss", float("inf"))
            log_payload = {}
            log_payload.update(_prefixed(val_loss, "val"))
            log_payload.update(_prefixed(val_gen, "val"))
            print(
                f"[val] step={completed_step} loss={val_loss_value:.6f} "
                f"endpoint_acc_001={val_gen.get('endpoint_acc_001', 0.0):.4f}"
            )
            is_best = val_loss_value < trainer.best_val_loss
            if is_best:
                trainer.best_val_loss = val_loss_value
                log_payload["val/best_loss"] = float(trainer.best_val_loss)
                trainer.save_checkpoint(
                    completed_step,
                    is_best=True,
                    save_latest=True,
                    archive_step=should_archive_step,
                )
            _log_wandb_metrics(log_payload, step=completed_step)
            ran_val = True

        if is_main_process and not ran_val and save_every > 0 and completed_step % save_every == 0:
            should_archive_step = completed_step >= archive_ckpt_start_step
            trainer.save_checkpoint(
                completed_step,
                is_best=False,
                save_latest=True,
                archive_step=should_archive_step,
            )

        _barrier_if_distributed(distributed)

    if is_main_process and (save_every <= 0 or train_iters % save_every != 0):
        should_archive_step = train_iters >= archive_ckpt_start_step
        trainer.save_checkpoint(
            train_iters,
            is_best=False,
            save_latest=True,
            archive_step=should_archive_step,
        )

    if wandb_run is not None:
        wandb.finish()
    _barrier_if_distributed(distributed)
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
