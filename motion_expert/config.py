import json
from dataclasses import dataclass, field
from typing import Optional, Tuple

import yaml

QUERY_UV_FORMAT_SOURCE_PIXELS = "source_pixels"
QUERY_UV_FORMAT_SOURCE_UNIT = "source_unit"
QUERY_UV_FORMAT_MODEL_GRID = "model_grid"
QUERY_UV_FORMATS = (
    QUERY_UV_FORMAT_SOURCE_PIXELS,
    QUERY_UV_FORMAT_SOURCE_UNIT,
    QUERY_UV_FORMAT_MODEL_GRID,
)


@dataclass
class QwenFeatureExtractorConfig:
    model_name_or_path: Optional[str] = None
    freeze_backbone: bool = True
    gradient_checkpointing: bool = False
    vision_lora_rank: int = 0
    vision_lora_alpha: int = 16
    vision_lora_dropout: float = 0.0
    vision_lora_num_layers: int = 0
    vision_lora_target_modules: str = "attn.qkv,attn.proj"
    llm_lora_rank: int = 0
    llm_lora_alpha: int = 16
    llm_lora_dropout: float = 0.0
    llm_lora_num_layers: int = 0
    llm_lora_target_modules: str = (
        "self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj"
    )


@dataclass
class FlorenceFeatureExtractorConfig:
    model_name_or_path: Optional[str] = None
    freeze_backbone: bool = True
    gradient_checkpointing: bool = False
    lora_rank: int = 0
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: str = ""


@dataclass
class TraceGenDecoderConfig:
    latent_dim: int = 1024
    num_attention_heads: int = 16
    attention_head_dim: int = 64
    in_channels: int = 3
    out_channels: int = 3
    num_layers: int = 12
    patch_size: int = 2
    patch_size_t: int = 1
    max_text_seq_length: int = 2048
    text_embed_dim: int = 1024
    use_rotary_positional_embeddings: bool = False
    enable_encoder_hidden_states_grad: bool = True
    torch_dtype: str = "float32"
    scale_factor: float = 0.7
    scale_factor_spatial: int = 1
    scale_factor_temporal: int = 1
    put_frames_in_channels: int = 2


@dataclass
class SparseMotionDecoderConfig:
    hidden_dim: int = 256
    point_encoder_layers: int = 4
    decoder_layers: int = 6
    num_attention_heads: int = 8
    mlp_ratio: int = 4
    visual_global_tokens: int = 16
    decoder_spatial_attention: str = "knn"
    decoder_serialized_patch_size: int = 32
    decoder_serialized_orders: Tuple[str, ...] = ("z", "z-trans")
    decoder_serialized_depth: int = 16
    decoder_serialized_shift_windows: bool = True


@dataclass
class MotionExpertConfig:
    backbone_type: str = "qwen"
    grid_size: int = 20
    future_steps: int = 15
    decoder_type: str = "sparse_factorized"
    knn_k: int = 16
    flow_timesteps: int = 100
    use_motion_weighted_loss: bool = True
    motion_weight_tau: float = 0.005
    motion_weight_temp: float = 5.0
    motion_weight_gamma: float = 1.0
    motion_weight_static_floor: float = 0.05
    input_height: int = 384
    input_width: int = 640
    text_max_length: int = 16
    image_pad_value: float = 0.5
    query_uv_format: str = QUERY_UV_FORMAT_SOURCE_PIXELS
    checkpoint_activations: bool = True
    dropout: float = 0.1
    use_calvin_percentile_normalization: bool = False
    query_xyz_center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    query_xyz_scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    delta_range_xyz: Tuple[float, float, float] = (0.05, 0.05, 0.04)
    query_centered_xyz_center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    query_centered_xyz_scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    traj_delta_xyz_center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    traj_delta_xyz_scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    sparse_decoder: SparseMotionDecoderConfig = field(
        default_factory=SparseMotionDecoderConfig
    )
    decoder: TraceGenDecoderConfig = field(default_factory=TraceGenDecoderConfig)
    qwen: QwenFeatureExtractorConfig = field(
        default_factory=QwenFeatureExtractorConfig
    )
    florence: FlorenceFeatureExtractorConfig = field(
        default_factory=FlorenceFeatureExtractorConfig
    )

    @property
    def num_query_tokens(self) -> int:
        return self.grid_size * self.grid_size

    def __post_init__(self) -> None:
        self.query_uv_format = str(self.query_uv_format).lower()
        if self.query_uv_format not in QUERY_UV_FORMATS:
            raise ValueError(
                "query_uv_format must be one of "
                f"{QUERY_UV_FORMATS}, got `{self.query_uv_format}`."
            )


def build_motion_expert_config(raw_cfg: dict) -> MotionExpertConfig:
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
        query_centered_xyz_center=tuple(
            raw_cfg.get("query_centered_xyz_center", [0.0, 0.0, 0.0])
        ),
        query_centered_xyz_scale=tuple(
            raw_cfg.get("query_centered_xyz_scale", [1.0, 1.0, 1.0])
        ),
        traj_delta_xyz_center=tuple(raw_cfg.get("traj_delta_xyz_center", [0.0, 0.0, 0.0])),
        traj_delta_xyz_scale=tuple(raw_cfg.get("traj_delta_xyz_scale", [1.0, 1.0, 1.0])),
        sparse_decoder=SparseMotionDecoderConfig(**sparse_decoder_cfg),
        qwen=QwenFeatureExtractorConfig(**qwen_cfg),
        florence=FlorenceFeatureExtractorConfig(**florence_cfg),
        decoder=TraceGenDecoderConfig(**decoder_cfg),
    )


def load_motion_expert_config(path: str) -> MotionExpertConfig:
    with open(path, "r") as handle:
        if str(path).endswith(".json"):
            raw = json.load(handle)
        else:
            raw = yaml.safe_load(handle)
    return build_motion_expert_config(raw.get("model", raw))
