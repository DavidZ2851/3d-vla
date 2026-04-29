# Sparse MotionExpert Implementation

## Summary

- Standalone sparse scene-flow expert for a `20 x 20` query grid.
- Predicts exactly 15 future stepwise world-frame `delta_xyz` values.
- Uses query-local visual features, kNN point encoding, flow matching, and a sparse spatial-temporal decoder.
- Keeps action-expert fusion out of this first implementation.

## Interfaces

- `MotionExpertConfig.future_steps = 15`
- `decoder_type = "sparse_factorized"`
- `knn_k = 16`
- `SparseMotionDecoderConfig` controls hidden size, encoder depth, decoder depth, heads, and MLP ratio.
- `MotionExpertBatch` supports optional `depth`, `intrinsics`, and `extrinsics`.
- No `valid_depth`, robot, or gripper fields are added.

## Data And Conditioning

- Prefer stored `query_world_xyz`.
- If absent, reconstruct query XYZ from normalized padded `query_uv`, depth, intrinsics, and extrinsics.
- Use only `traj_mask` as the training supervision mask.
- Qwen3-VL features are pre-merge patch-16 visual features.
- Florence2 features use the DPT/FPN path and patch-8 visual grid.
- Per-query visual features are sampled from `visual_grid`; global visual mean and text tokens condition the decoder.

## Model

- Build point features from `xyz`, normalized `xyz`, `uv`, depth, and sampled visual features.
- Build the kNN graph once from current-frame observed XYZ.
- Encode points with local geometry attention and relative XYZ/distance bias.
- Decode flow-matching velocity over `[B, 400, 15, 3]` with:
  - local spatial attention
  - temporal attention
  - global visual/text cross-attention
  - FFN

## Loss And Metrics

- Normalize future deltas once before flow matching.
- Train with `x_tau = (1 - tau) * noise + tau * target` and `v_target = target - noise`.
- Optional moving-point weighting is enabled by default:
  - `motion_score = sigmoid((||future_delta_xyz|| - tau) * temp / tau)`
  - `loss_weight = traj_mask * max(motion_score^gamma, static_floor)`
  - default `tau=0.005`, `temp=5.0`, `gamma=1.0`, `static_floor=0.05`
- Metrics report delta L1/MSE, endpoint L1, moved-point L1, and static-point L1 in meters.
