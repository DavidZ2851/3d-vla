import math
import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
from einops import rearrange
from timm.models.vision_transformer import RmsNorm
from torch import nn

from modeling.encoder.vision.florence2 import CLIPTransform, Florence2
from modeling.encoder.vision.qwen3vl import Qwen3VLBackbone, Qwen3VLNormalize
from modeling.encoder.vision.vit_fpn import ViTFPN

from .config import FlorenceFeatureExtractorConfig, QwenFeatureExtractorConfig


def _unwrap_peft_base(module: nn.Module) -> nn.Module:
    inner = module
    while hasattr(inner, "get_base_model"):
        next_inner = inner.get_base_model()
        if next_inner is inner:
            break
        inner = next_inner
    return inner


def _is_peft_wrapped(module: nn.Module) -> bool:
    return getattr(module, "peft_config", None) is not None


def _set_trainable_lora_only(module: nn.Module) -> None:
    for name, param in module.named_parameters():
        param.requires_grad = "lora_" in name


def _resolve_model_path(
    model_name_or_path: Optional[str],
    env_var: str,
    default: Optional[str] = None,
) -> str:
    if model_name_or_path is not None:
        return model_name_or_path
    env_path = os.environ.get(env_var)
    if env_path:
        return env_path
    if default is not None:
        return default
    raise ValueError(
        f"No model path provided. Set {env_var} or pass model_name_or_path explicitly."
    )


def _resolve_linear_target_modules(
    module: nn.Module,
    requested_targets: str,
) -> List[str]:
    if requested_targets.strip():
        return [name.strip() for name in requested_targets.split(",") if name.strip()]

    targets: List[str] = []
    for name, child in module.named_modules():
        if not name:
            continue
        if isinstance(child, nn.Linear):
            targets.append(name)
    if not targets:
        raise RuntimeError("Could not find any linear layers for Florence LoRA.")
    return targets


def _apply_florence_lora(
    module: nn.Module,
    config: FlorenceFeatureExtractorConfig,
) -> Tuple[nn.Module, Sequence[str]]:
    if config.lora_rank <= 0:
        return module, []

    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "PEFT is required for Florence backbone LoRA. Install with: pip install peft"
        ) from exc

    target_modules = _resolve_linear_target_modules(module, config.lora_target_modules)
    lora_cfg = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    wrapped = get_peft_model(module, lora_cfg)
    return wrapped, target_modules


@dataclass
class MotionExpertFeatureOutput:
    visual_grid: torch.Tensor
    visual_tokens: torch.Tensor
    text_tokens: torch.Tensor


QwenFeatureOutput = MotionExpertFeatureOutput


class QwenTraceFeatureExtractor(nn.Module):
    """
    Qwen feature wrapper for MotionExpert.

    Outputs:
      - pre-merge Qwen vision patches
      - text tokens from the Qwen LLM final hidden state
    """

    def __init__(self, config: Optional[QwenFeatureExtractorConfig] = None):
        super().__init__()
        self.config = config or QwenFeatureExtractorConfig()
        self.normalize = Qwen3VLNormalize(self.config.model_name_or_path)
        self.backbone = Qwen3VLBackbone(
            model_name_or_path=self.config.model_name_or_path,
            vision_lora_rank=self.config.vision_lora_rank,
            vision_lora_alpha=self.config.vision_lora_alpha,
            vision_lora_dropout=self.config.vision_lora_dropout,
            vision_lora_num_layers=self.config.vision_lora_num_layers,
            vision_lora_target_modules=self.config.vision_lora_target_modules,
            llm_lora_rank=self.config.llm_lora_rank,
            llm_lora_alpha=self.config.llm_lora_alpha,
            llm_lora_dropout=self.config.llm_lora_dropout,
            llm_lora_num_layers=self.config.llm_lora_num_layers,
            llm_lora_target_modules=self.config.llm_lora_target_modules,
            gradient_checkpointing=self.config.gradient_checkpointing,
            use_multiscale=False,
        )

    @property
    def visual_token_dim(self) -> int:
        return int(self.backbone.vision_hidden_size)

    @property
    def text_token_dim(self) -> int:
        return int(self.backbone.llm_hidden_size)

    def estimate_visual_token_count(self, input_height: int, input_width: int) -> int:
        patch_size = int(self.backbone.patch_size)
        return (input_height // patch_size) * (input_width // patch_size)

    def _visual_dtype(self) -> torch.dtype:
        visual_inner = _unwrap_peft_base(self.backbone.vlm.visual)
        for param in visual_inner.parameters():
            return param.dtype
        return torch.float32

    def forward(
        self,
        images: torch.Tensor,
        text_token_ids: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor] = None,
        text_valid_lens: Optional[torch.Tensor] = None,
    ) -> MotionExpertFeatureOutput:
        if images.dim() == 4:
            images = images.unsqueeze(1)
        if images.dim() != 5:
            raise ValueError(
                "images must be [B, 3, H, W] or [B, ncam, 3, H, W], got "
                f"{tuple(images.shape)}."
            )

        if text_valid_lens is not None and isinstance(text_valid_lens, torch.Tensor):
            text_valid_lens = tuple(int(x) for x in text_valid_lens.tolist())

        images = self.normalize(images)
        visual_dtype = self._visual_dtype()
        if images.dtype != visual_dtype:
            images = images.to(dtype=visual_dtype)

        grad_ctx = nullcontext() if not self.config.freeze_backbone else torch.no_grad()
        with grad_ctx:
            pre_merge_visual, text_tokens, _ = self.backbone(
                images,
                text_token_ids,
                text_attention_mask=text_attention_mask,
                text_valid_lens=text_valid_lens,
            )

        visual_tokens = pre_merge_visual.reshape(
            pre_merge_visual.shape[0],
            -1,
            pre_merge_visual.shape[-1],
        )
        return MotionExpertFeatureOutput(
            visual_grid=pre_merge_visual,
            visual_tokens=visual_tokens,
            text_tokens=text_tokens,
        )


class FlorenceTraceFeatureExtractor(nn.Module):
    """
    Florence2 feature wrapper for MotionExpert.

    It mirrors the Florence path used in the Flower stack:
      - Florence image+text encoder
      - DPT/FPN-style visual neck built from hidden states
      - instruction tokens from the Florence encoder output

    Training modes:
      - frozen backbone: `freeze_backbone=True`, `lora_rank=0`
      - full finetune: `freeze_backbone=False`, `lora_rank=0`
      - whole-backbone LoRA: `lora_rank>0`
    """

    def __init__(self, config: Optional[FlorenceFeatureExtractorConfig] = None):
        super().__init__()
        self.config = config or FlorenceFeatureExtractorConfig()
        model_path = _resolve_model_path(
            self.config.model_name_or_path,
            env_var="FLORENCE2_MODEL_PATH",
            default="microsoft/Florence-2-large",
        )
        self.normalize = CLIPTransform()
        self.backbone = Florence2(model_name=model_path)
        if self.config.gradient_checkpointing and hasattr(
            self.backbone.vlm, "gradient_checkpointing_enable"
        ):
            self.backbone.vlm.gradient_checkpointing_enable()

        self.hidden_dim = int(self.backbone.vlm.config.text_config.d_model)
        self.visual_neck = ViTFPN(
            out_res="res3",
            out_dim=self.hidden_dim,
            features=[128, 256, 512, 1024],
            vit_feat_dim=self.hidden_dim,
        )
        self.text_norm = RmsNorm(self.hidden_dim)
        self.resolved_lora_target_modules: Sequence[str] = []

        if self.config.lora_rank > 0:
            self.backbone.vlm, self.resolved_lora_target_modules = _apply_florence_lora(
                self.backbone.vlm,
                self.config,
            )
            _set_trainable_lora_only(self.backbone.vlm)
        else:
            for param in self.backbone.parameters():
                param.requires_grad = not self.config.freeze_backbone

    @property
    def visual_token_dim(self) -> int:
        return self.hidden_dim

    @property
    def text_token_dim(self) -> int:
        return self.hidden_dim

    def estimate_visual_token_count(self, input_height: int, input_width: int) -> int:
        h_feat = 2 * math.ceil(int(input_height) / 32.0)
        w_feat = 2 * math.ceil(int(input_width) / 32.0)
        return h_feat * w_feat

    def _backbone_dtype(self) -> torch.dtype:
        for param in self.backbone.vlm.parameters():
            return param.dtype
        return torch.float32

    def _backbone_needs_grad(self) -> bool:
        if self.config.lora_rank > 0:
            return True
        return not self.config.freeze_backbone

    def forward(
        self,
        images: torch.Tensor,
        text_token_ids: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor] = None,
        text_valid_lens: Optional[torch.Tensor] = None,
    ) -> MotionExpertFeatureOutput:
        del text_valid_lens  # Florence does not need valid lengths explicitly.

        if images.dim() == 4:
            images = images.unsqueeze(1)
        if images.dim() != 5:
            raise ValueError(
                "images must be [B, 3, H, W] or [B, ncam, 3, H, W], got "
                f"{tuple(images.shape)}."
            )

        batch_size, num_cameras, _, height, width = images.shape
        images = self.normalize(images)
        backbone_dtype = self._backbone_dtype()
        if images.dtype != backbone_dtype:
            images = images.to(dtype=backbone_dtype)

        grad_ctx = nullcontext() if self._backbone_needs_grad() else torch.no_grad()
        with grad_ctx:
            hidden_states, text_tokens = self.backbone(
                images,
                text_token_ids,
                output_hidden_states=True,
            )

        h_feat_rgb = math.ceil(height / 32.0)
        w_feat_rgb = math.ceil(width / 32.0)
        visual_grid = self.visual_neck(
            {
                "res5": rearrange(
                    hidden_states[:, 3],
                    "b (nc h w) c -> (b nc) c h w",
                    nc=num_cameras,
                    h=h_feat_rgb,
                    w=w_feat_rgb,
                ),
                "res4": rearrange(
                    hidden_states[:, 2],
                    "b (nc h w) c -> (b nc) c h w",
                    nc=num_cameras,
                    h=h_feat_rgb,
                    w=w_feat_rgb,
                ),
                "res3": rearrange(
                    hidden_states[:, 1],
                    "b (nc h w) c -> (b nc) c h w",
                    nc=num_cameras,
                    h=h_feat_rgb,
                    w=w_feat_rgb,
                ),
                "res2": rearrange(
                    hidden_states[:, 0],
                    "b (nc h w) c -> (b nc) c h w",
                    nc=num_cameras,
                    h=h_feat_rgb,
                    w=w_feat_rgb,
                ),
            }
        )
        visual_grid = rearrange(
            visual_grid,
            "(b nc) c h w -> b nc h w c",
            b=batch_size,
            nc=num_cameras,
        )
        visual_tokens = visual_grid.reshape(batch_size, -1, visual_grid.shape[-1])

        text_tokens = self.text_norm(text_tokens)
        if text_attention_mask is not None:
            text_tokens = text_tokens * text_attention_mask.to(
                device=text_tokens.device,
                dtype=text_tokens.dtype,
            ).unsqueeze(-1)

        return MotionExpertFeatureOutput(
            visual_grid=visual_grid,
            visual_tokens=visual_tokens,
            text_tokens=text_tokens,
        )
