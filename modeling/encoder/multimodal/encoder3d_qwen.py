import einops
from contextlib import nullcontext
import torch
from torch import nn
from torch.nn import functional as F

from ...policy.flower.transformers import RmsNorm
from ...utils.position_encodings import RotaryPositionEncoding3D
from ...utils.layers import AttentionModule
from .multiscale import QwenMultiScaleFusion
from ..vision.qwen3vl import Qwen3VLBackbone, Qwen3VLNormalize  # noqa: direct import avoids __init__.py side-effects


@torch.no_grad()
def density_based_sampler(features, subsample_factor, k=8):
    B, N, C = features.shape
    dists = torch.cdist(features, features, p=2)
    knn_dists, _ = dists.topk(k=k, dim=-1, largest=False)
    density = knn_dists.mean(dim=-1)
    M = N // subsample_factor
    sampled_inds = density.topk(M, dim=-1, largest=True).indices
    return sampled_inds


def _is_peft_wrapped(module: nn.Module) -> bool:
    return getattr(module, "peft_config", None) is not None


def _set_trainable_lora_only(module: nn.Module) -> None:
    """Enable gradients only on PEFT adapter parameters (name contains ``lora_``)."""
    for n, p in module.named_parameters():
        p.requires_grad = "lora_" in n


def _enable_lora_params(module: nn.Module) -> None:
    for n, p in module.named_parameters():
        if "lora_" in n:
            p.requires_grad = True


def _unwrap_peft_base(module: nn.Module) -> nn.Module:
    inner = module
    while hasattr(inner, "get_base_model"):
        next_inner = inner.get_base_model()
        if next_inner is inner:
            break
        inner = next_inner
    return inner


class Qwen3VLEncoder3D(nn.Module):
    """
    Uses Qwen3-VL vision features for scene tokens and fused LLM hidden states for language tokens.
    Both modalities use ``Linear(RmsNorm(x))`` into ``embedding_dim`` (bias-free linear).

    When ``use_post_merge=False`` (default): uses **pre-merge** features (stride 16,
    dim=``vision_hidden_size``), captured via a forward-pre-hook before PatchMerger.

    When ``use_post_merge=True``: uses **post-merge** features (stride 32,
    dim=``llm_hidden_size``), captured via a forward-hook on PatchMerger output.
    PCD is bilinearly downsampled to the stride-32 grid in this mode.

    **Trainable layout** (typical Calvin / MiBot script):
    - ``visual`` full weights update for the last ``num_finetune_vision_layers``
      blocks plus merger modules, or the whole VLM when ``finetune_backbone``.
    - ``language_model`` full weights update when ``finetune_text_encoder``.
    - Otherwise ``language_model`` base weights stay frozen, and only **PEFT LoRA**
      adapters train when ``llm_lora_rank > 0`` and ``llm_lora_num_layers > 0``.
    """
    def __init__(
        self,
        backbone="qwen3vl",
        embedding_dim=60,
        nhist=1,
        num_attn_heads=9,
        fps_subsampling_factor=5,
        finetune_backbone=False,
        num_finetune_vision_layers=0,
        finetune_text_encoder=False,
        backbone_module=None,
        normalize_module=None,
        vision_lora_rank=0,
        vision_lora_alpha=16,
        vision_lora_dropout=0.0,
        vision_lora_num_layers=0,
        vision_lora_target_modules="attn.qkv,attn.proj",
        llm_lora_rank=0,
        llm_lora_alpha=16,
        llm_lora_dropout=0.0,
        llm_lora_num_layers=0,
        llm_lora_target_modules="self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj",
        gradient_checkpointing=False,
        use_multiscale=False,
        use_post_merge=False,
        **kwargs,
    ):
        super().__init__()
        self.subsampling_factor = fps_subsampling_factor
        self.use_multiscale = bool(use_multiscale)
        self.use_post_merge = bool(use_post_merge)

        self._use_llm_lora = bool(llm_lora_rank > 0 and llm_lora_num_layers > 0)
        self._use_vision_lora = bool(vision_lora_rank > 0 and vision_lora_num_layers > 0)

        # ---------- Qwen3-VL backbone ----------
        self.backbone = backbone_module or Qwen3VLBackbone(
            vision_lora_rank=vision_lora_rank,
            vision_lora_alpha=vision_lora_alpha,
            vision_lora_dropout=vision_lora_dropout,
            vision_lora_num_layers=vision_lora_num_layers,
            vision_lora_target_modules=vision_lora_target_modules,
            llm_lora_rank=llm_lora_rank,
            llm_lora_alpha=llm_lora_alpha,
            llm_lora_dropout=llm_lora_dropout,
            llm_lora_num_layers=llm_lora_num_layers,
            llm_lora_target_modules=llm_lora_target_modules,
            gradient_checkpointing=gradient_checkpointing,
            use_multiscale=use_multiscale,
        )
        self.normalize = normalize_module or Qwen3VLNormalize()
        self.finetune_backbone = finetune_backbone
        self.finetune_text_encoder = finetune_text_encoder
        self.num_finetune_vision_layers = num_finetune_vision_layers
        self.full_visual_finetune = (
            finetune_backbone or num_finetune_vision_layers > 0
        )

        self._configure_backbone_requires_grad()

        # ---------- Bridge layers (trained from scratch) ----------
        if self.use_multiscale:
            self.multiscale_fuser = QwenMultiScaleFusion(
                self.backbone.vision_hidden_size,
                embedding_dim,
                self.backbone.multiscale_layers,
            )
        else:
            # post-merge: dim = llm_hidden_size (merger projects to LLM input space)
            vis_in_dim = (
                self.backbone.llm_hidden_size
                if use_post_merge
                else self.backbone.vision_hidden_size
            )
            self.vis_norm = RmsNorm(vis_in_dim)
            self.vis_proj = nn.Linear(vis_in_dim, embedding_dim, bias=False)

        # Register post-merge hook when use_post_merge=True
        if use_post_merge and not self.use_multiscale:
            self._post_merge_features = None
            visual_inner = _unwrap_peft_base(self.backbone.vlm.visual)
            visual_inner.merger.register_forward_hook(self._post_merge_hook)

        # Text: cond_linear(cond_norm(lang_feats)); LLM dim is Qwen hidden size.
        self.cond_norm = RmsNorm(self.backbone.llm_hidden_size)
        self.cond_linear = nn.Linear(
            self.backbone.llm_hidden_size, embedding_dim, bias=False
        )

        # ---------- 3D proprioception encoder (reused from Encoder3D) ----------
        self.relative_pe_layer = RotaryPositionEncoding3D(embedding_dim)
        self.curr_gripper_embed = nn.Embedding(nhist, embedding_dim)
        self.gripper_context_head = AttentionModule(
            num_layers=3,
            d_model=embedding_dim,
            dim_fw=embedding_dim,
            n_heads=num_attn_heads,
            rotary_pe=True,
            use_adaln=False,
            pre_norm=False,
        )

    def _post_merge_hook(self, module, args, output):
        """Capture PatchMerger output (post-merge, stride-32 features)."""
        self._post_merge_features = output

    def _backbone_visual_dtype(self) -> torch.dtype:
        visual_inner = _unwrap_peft_base(self.backbone.vlm.visual)
        for param in visual_inner.parameters():
            return param.dtype
        return torch.float32

    def _configure_backbone_requires_grad(self) -> None:
        """
        Freeze / unfreeze ``vlm`` submodules so that:
        - Full finetune: entire ``vlm`` trains.
        - Otherwise: last-N ``visual`` blocks + mergers train (and/or vision LoRA);
          ``language_model`` trains fully when ``finetune_text_encoder`` or
          LoRA-only when enabled.
        """
        vlm = self.backbone.vlm
        for p in vlm.parameters():
            p.requires_grad = False

        if self.finetune_backbone:
            for p in vlm.parameters():
                p.requires_grad = True
            return

        visual = vlm.visual
        visual_inner = _unwrap_peft_base(visual)

        # Vision tower (selected ViT blocks + mergers, plus optional vision LoRA)
        if self.num_finetune_vision_layers > 0:
            n_blocks = len(visual_inner.blocks)
            if self.num_finetune_vision_layers >= n_blocks:
                for p in visual_inner.parameters():
                    p.requires_grad = True
            else:
                start = max(0, n_blocks - self.num_finetune_vision_layers)
                for block in visual_inner.blocks[start:]:
                    for p in block.parameters():
                        p.requires_grad = True
                for p in visual_inner.merger.parameters():
                    p.requires_grad = True
                for merger in getattr(visual_inner, "deepstack_merger_list", []):
                    for p in merger.parameters():
                        p.requires_grad = True

        if self._use_vision_lora:
            if self.num_finetune_vision_layers == 0:
                # Vision LoRA-only: keep base vision frozen, adapters only.
                _set_trainable_lora_only(visual)
            else:
                _enable_lora_params(visual)

        # Language model: full finetune, LoRA adapters only, or nothing.
        lm = vlm.language_model
        if self.finetune_text_encoder:
            for p in lm.parameters():
                p.requires_grad = True
        elif _is_peft_wrapped(lm):
            _set_trainable_lora_only(lm)

    def forward(
        self,
        rgb3d,
        rgb2d,
        pcd,
        instruction,
        proprio,
        instr_mask=None,
        instr_lens=None,
    ):
        """
        Args:
            rgb3d:       (B, ncam, 3, H, W) in [0, 1]
            rgb2d:       ignored
            pcd:         (B, ncam, 3, H, W)
            instruction: (B, L) tokenised text
            proprio:     (B, nhist*nhand, D)
            instr_mask:  (B, L) optional; 1 = real token, 0 = pad (from Qwen tokenizer)
            instr_lens:  optional per-sample valid token counts from CPU-side tokenization

        Returns:
            Same 9-tuple as ``Encoder3D.forward``:
            (rgb3d_feats, pcd, rgb2d_feats, rgb2d_pos,
             instr_feats, instr_pos, proprio_feats,
             fps_scene_feats, fps_scene_pos)
        """
        rgb3d_normed = self.normalize(rgb3d)
        visual_dtype = self._backbone_visual_dtype()
        if rgb3d_normed.dtype != visual_dtype:
            rgb3d_normed = rgb3d_normed.to(dtype=visual_dtype)

        needs_grad = (
            self.finetune_backbone
            or self.finetune_text_encoder
            or self.num_finetune_vision_layers > 0
            or self._use_vision_lora
            or self._use_llm_lora
        )
        grad_ctx = nullcontext() if needs_grad else torch.no_grad()
        with grad_ctx:
            pre_merge, text_hidden, multiscale_pre_merge = self.backbone(
                rgb3d_normed,
                instruction,
                text_attention_mask=instr_mask,
                text_valid_lens=instr_lens,
            )
        # pre_merge:   (B, ncam, h16, w16, C_vis)
        # text_hidden: (B, L, C_llm)

        B, ncam, h16, w16, _ = pre_merge.shape
        ms = self.backbone.spatial_merge_size  # typically 2
        h32, w32 = h16 // ms, w16 // ms

        # ---- Vision: RmsNorm on C_vis then Linear → embedding_dim ----
        if self.use_multiscale:
            rgb3d_feats = self.multiscale_fuser(multiscale_pre_merge)
        elif self.use_post_merge:
            # post_merge_features: (B * ncam * h32 * w32, llm_hidden_size)
            post_merge = self._post_merge_features
            post_merge = post_merge.reshape(B, ncam * h32 * w32, -1)
            rgb3d_feats = self.vis_proj(self.vis_norm(post_merge.to(self.vis_proj.weight.dtype)))
        else:
            rgb3d_feats = pre_merge.reshape(B, ncam * h16 * w16, -1)
            rgb3d_feats = self.vis_proj(self.vis_norm(rgb3d_feats))

        # ---- Language: same pattern as Florence2 (RmsNorm + bias-free Linear) ----
        instr_feats = self.cond_linear(self.cond_norm(text_hidden))

        # ---- Align PCD to vision feature grid (stride-16 or stride-32) ----
        num_cameras = pcd.shape[1]
        grid_h = h32 if self.use_post_merge else h16
        grid_w = w32 if self.use_post_merge else w16
        pcd_down = F.interpolate(
            einops.rearrange(pcd, "b nc c h w -> (b nc) c h w"),
            (grid_h, grid_w),
            mode="bilinear",
        )
        pcd_flat = einops.rearrange(
            pcd_down, "(b nc) c h w -> b (nc h w) c", nc=num_cameras
        )

        # ---- Language "position" = last gripper xyz ----
        instr_pos = proprio[:, -1:, :3].repeat(1, instr_feats.size(1), 1)

        # ---- Proprioception encoding ----
        proprio_feats = self.encode_proprio(proprio, rgb3d_feats, pcd_flat)

        # ---- FPS subsampling ----
        fps_feats, fps_pos = self.run_fps(rgb3d_feats, pcd_flat)

        return (
            rgb3d_feats, pcd_flat,
            None, None,
            instr_feats, instr_pos,
            proprio_feats,
            fps_feats, fps_pos,
        )

    def encode_proprio(self, proprio, context_feats, context_pos):
        proprio_feats = self.curr_gripper_embed.weight.unsqueeze(0).expand(
            len(proprio), -1, -1
        )
        proprio_pos = self.relative_pe_layer(proprio[..., :3])
        ctx_pos = self.relative_pe_layer(context_pos)
        proprio_feats = self.gripper_context_head(
            proprio_feats, context_feats,
            seq1_pos=proprio_pos, seq2_pos=ctx_pos,
        )[-1]
        return proprio_feats

    def run_fps(self, features, pos):
        if self.subsampling_factor == 1:
            return features, pos
        bs, npts, ch = features.shape
        inds = density_based_sampler(features, self.subsampling_factor)
        exp = inds.unsqueeze(-1).expand(-1, -1, ch)
        sampled_feats = torch.gather(features, 1, exp)
        if pos is None:
            return sampled_feats, None
        exp3 = inds.unsqueeze(-1).expand(-1, -1, 3)
        sampled_pos = torch.gather(pos, 1, exp3)
        return sampled_feats, sampled_pos
