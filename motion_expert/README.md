# Motion Expert

`motion_expert` is a single-view, TraceGen-style 3D delta generator for a
regular `20 x 20` point grid. It conditions on:

- Whole-image backbone visual tokens.
- Backbone text-token hidden states.
- Per-point `query_uv` and `query_world_xyz`.

It predicts stepwise future world-frame `delta_xyz` and keeps TraceGen's
CogVideoX flow decoder pattern: query, visual, and text tokens are concatenated
and passed as `encoder_hidden_states`.

Two backbone families are supported:

- `qwen`: pre-merge Qwen visual patches plus final Qwen text tokens
- `florence2`: Florence2 visual/text features with the Flower-style ViT-FPN / DPT-like neck

## Tensor Contract

`MotionExpertBatch` accepts either raw Qwen inputs or precomputed Qwen tokens:

- `rgb`: `[B, 3, H, W]`
- `text_token_ids`: `[B, L]`
- `visual_tokens`: `[B, Nv, Cvis]`
- `text_tokens`: `[B, L, Ctxt]`
- `query_uv`: `[B, 400, 2]` or `[B, 20, 20, 2]`, normalized to `[0, 1]` in
  the padded model-input canvas
- `query_world_xyz`: `[B, 400, 3]`
- `future_delta_xyz`: `[B, 400, T, 3]`
- `traj_mask`: `[B, 400, T]`

The data loader computes stepwise deltas from absolute future positions:

- `future_delta_xyz[:, 0] = traj_3d_xyz[:, 0] - query_world_xyz`
- `future_delta_xyz[:, t] = traj_3d_xyz[:, t] - traj_3d_xyz[:, t - 1]` for `t > 0`

Invalid future tails are filled with the last valid point for tensor stability,
and `traj_mask` is still used during training loss.

## Model Layout

1. Extract or receive backbone visual/text tokens online.
2. Build one query token per grid point from normalized `query_uv` and normalized `query_world_xyz`.
3. Concatenate query, visual, and text tokens into one conditioning sequence.
4. Feed that sequence into the same CogVideoX flow decoder family used by TraceGen.
5. Train on normalized stepwise delta XYZ with fixed symmetric `delta_range_xyz`.

Absolute query positions use frozen train-split stats:

- `query_world_xyz_norm = (query_world_xyz - query_xyz_center) / query_xyz_scale`

Delta targets use fixed predefined ranges shared between training and inference:

- `future_delta_xyz_norm = clamp(future_delta_xyz / delta_range_xyz, -1, 1)`

## Example

```python
import torch

from motion_expert import (
    MotionExpertBatch,
    MotionExpertConfig,
    MotionExpertPipeline,
    TraceGenDecoderConfig,
)

config = MotionExpertConfig(
    future_steps=16,
    input_height=384,
    input_width=640,
    decoder=TraceGenDecoderConfig(
        latent_dim=256,
        text_embed_dim=256,
        num_attention_heads=4,
        attention_head_dim=64,
        num_layers=2,
        max_text_seq_length=400 + 960 + 128,
        put_frames_in_channels=2,
    ),
)
pipeline = MotionExpertPipeline(config)

batch = MotionExpertBatch(
    visual_tokens=torch.randn(2, 960, 256),
    text_tokens=torch.randn(2, 128, 256),
    query_uv=torch.rand(2, 400, 2),
    query_world_xyz=torch.randn(2, 400, 3),
    future_delta_xyz=torch.randn(2, 400, 16, 3),
    traj_mask=torch.ones(2, 400, 16, dtype=torch.bool),
)

outputs = pipeline(batch)
print(outputs["loss"])
```

## Using Raw Qwen Inputs

If you want this package to run the selected backbone directly instead of
consuming precomputed tokens, use:

```python
pipeline = MotionExpertPipeline.from_backbone(MotionExpertConfig(backbone_type="qwen"))
pipeline = MotionExpertPipeline.from_backbone(MotionExpertConfig(backbone_type="florence2"))
```

The zarr training path assumes:

- `rgb`: `[3, H, W]`
- `query_uv`: `[400, 2]`
- `query_world_xyz`: `[400, 3]`
- `traj_3d_xyz`: `[400, T, 3]`
- `traj_mask`: `[400, T]`
- an instruction ID key such as `instr_id`, resolved through a side-table JSON

The dataset uses aspect-preserving resize + padding instead of squash resize so
the whole-image patch tokens stay geometrically consistent with the transformed
UV grid.

For Florence2, the extractor can be used in three modes:

- frozen backbone: `freeze_backbone=True`
- full finetune: `freeze_backbone=False`
- whole-backbone LoRA: set `lora_rank > 0`
