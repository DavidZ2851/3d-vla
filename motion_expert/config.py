from dataclasses import dataclass, field
from typing import Optional, Tuple


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
    visual_global_tokens: int = 1
    decoder_spatial_attention: str = "knn"
    decoder_serialized_patch_size: int = 32
    decoder_serialized_orders: Tuple[str, ...] = ("z", "z-trans")
    decoder_serialized_depth: int = 16


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
    text_max_length: int = 128
    image_pad_value: float = 0.5
    query_uv_is_normalized: bool = False
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
