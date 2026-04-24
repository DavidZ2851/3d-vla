# MotionExpert V1 Plan

## Summary

- Single-view MotionExpert for a `20 x 20` point grid.
- Condition on:
  - whole-image backbone visual tokens
  - backbone text-token hidden states
  - per-point `query_uv` and `query_world_xyz`
- Predict stepwise world-frame `delta_xyz` only.
- Run the backbone online; do not cache features.
- Keep TraceGen-style CogVideoX flow conditioning and decoder attention.

Supported backbones:

- `qwen`: pre-merge Qwen visual patches + final text tokens
- `florence2`: Florence2 encoder + Flower-style ViT-FPN / DPT-like visual neck

## Dataset Contract

Each zarr sample stores:

- `rgb`: `[3, H, W]`
- `query_uv`: `[400, 2]`
- `query_world_xyz`: `[400, 3]`
- `traj_3d_xyz`: `[400, T, 3]`
- `traj_mask`: `[400, T]`
- an instruction ID key such as `instr_id`, `instruction_id`, or `task_id`

There is no `point_mask`.

## Preprocessing

- Use aspect-preserving resize + padding to a fixed `(input_h, input_w)` canvas.
- Apply the same resize + pad transform to `query_uv`.
- Normalize final `query_uv` into `[0, 1]` in padded-image coordinates.
- Fill invalid future tails with the last valid absolute point for tensor stability.

## Normalization

Absolute query positions use frozen train-split bounds:

- `query_world_xyz_norm = (query_world_xyz - query_xyz_center) / query_xyz_scale`

Delta targets use fixed predefined symmetric ranges:

- `future_delta_xyz[:, 0] = traj_3d_xyz[:, 0] - query_world_xyz`
- `future_delta_xyz[:, t] = traj_3d_xyz[:, t] - traj_3d_xyz[:, t - 1]` for `t > 0`
- `future_delta_xyz_norm = clamp(future_delta_xyz / delta_range_xyz, -1, 1)`

`query_xyz_center` and `query_xyz_scale` are computed once from the training-set
`query_world_xyz` values and then frozen. `delta_range_xyz` stays predefined so
training and unseen-domain inference share the same target scaling.

## Model

1. Reuse the repo's Qwen encoder path for:
   - pre-merge visual tokens
   - final text-token hidden states
2. Or reuse the repo's Florence2 path for:
   - Florence image/text encoder
   - Flower-style ViT-FPN / DPT-like visual feature neck
   - Florence instruction-token features
3. Build one query token per point from `query_uv` and normalized `query_world_xyz`.
4. Concatenate `query_tokens + visual_tokens + text_tokens`.
5. Feed the concatenated sequence into the TraceGen CogVideoX flow decoder as `encoder_hidden_states`.

## Training And Metrics

- Train on normalized delta XYZ.
- Use `traj_mask` in the diffusion loss.
- Evaluate by unnormalizing predictions and comparing delta trajectories directly in world units.
- Absolute future XYZ reconstruction is optional for visualization only.

Backbone training modes:

- frozen backbone
- full backbone finetuning
- whole-backbone LoRA
