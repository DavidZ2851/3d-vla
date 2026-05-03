# Motion Expert

`motion_expert` is a standalone sparse scene-flow predictor for a regular
`20 x 20` point grid. It predicts 15 future stepwise world-frame `delta_xyz`
values with a sparse flow-matching decoder.

The model combines:

- query-local VLM features sampled at each point UV
- pooled visual summary tokens for decoder cross-attention
- PointWorld-style kNN point encoding over current-frame XYZ
- TraceGen-style flow matching in normalized delta space
- PhysCtrl-style spatial then temporal sparse trajectory decoding

Two backbone families are supported:

- `qwen`: pre-merge Qwen3-VL visual patches, patch size 16
- `florence2`: native Florence2 visual tokens, patch size 32

## Tensor Contract

`MotionExpertBatch` accepts raw backbone inputs or precomputed tokens:

- `rgb`: `[B, 3, H, W]`
- `text_token_ids`: `[B, L]`
- `visual_tokens`: `[B, Nv, Cvis]`
- `text_tokens`: `[B, L, Ctxt]`
- `query_uv`: `[B, 400, 2]` or `[B, 20, 20, 2]`, normalized to `[-1, 1]` in the padded model-input canvas
- `query_world_xyz`: optional `[B, 400, 3]`; dataset and pipeline center it by the per-sample query-point mean
- `depth`, `intrinsics`, `extrinsics`: optional fallback when `query_world_xyz` is absent
- `future_delta_xyz`: `[B, 400, 15, 3]`
- `traj_mask`: `[B, 400, 15]`

No `valid_depth`, robot, or gripper fields are part of the batch API.

The zarr training loader accepts raw `query_uv` in one explicit format:

- `query_uv_format: source_pixels`: source RGB pixel coordinates `[u, v]`
- `query_uv_format: source_unit`: source RGB unit coordinates in `[0, 1]`
- `query_uv_format: model_grid`: already-normalized padded model-canvas coordinates in `[-1, 1]`

CALVIN motion zarrs store `query_uv` as source-image pixel coordinates, so the
CALVIN configs use `query_uv_format: source_pixels`. The model and pipeline
always consume `MotionExpertBatch.query_uv` in `model_grid` form.

## Model Layout

1. Extract or receive visual/text tokens.
2. Sample visual-grid features at each normalized query UV.
3. Build point features from normalized centered `xyz`, `uv`, and sampled visual features.
4. Build a fixed kNN graph from centered current observed XYZ.
5. Encode point context with local geometry attention.
6. Denoise `[B, 400, 15, 3]` normalized deltas with sparse decoder blocks:
   local spatial attention, temporal attention, pooled visual/text conditioning, and FFN.

`sparse_decoder.visual_global_tokens` controls how many pooled visual summary
tokens are exposed to decoder cross-attention.

Training uses one supervision mask: `traj_mask`.

## Normalization And Loss

Query positions are centered per sample before normalization. Frozen train-split
stats should be computed on these centered points:

- `centered_query_world_xyz = query_world_xyz - mean(query_world_xyz, dim=points)`
- `query_world_xyz_norm = clamp((centered_query_world_xyz - query_xyz_center) / query_xyz_scale, -2, 2)`

Delta targets use fixed predefined ranges unless CALVIN zarr mean/std
normalization is enabled:

- `future_delta_xyz_norm = clamp(future_delta_xyz / delta_range_xyz, -2, 2)`
- CALVIN: `future_delta_xyz_norm = clamp((future_delta_xyz - traj_delta_xyz_center) / traj_delta_xyz_scale, -2, 2)`

The flow target is:

- `x_tau = (1 - tau) * noise + tau * target`
- `v_target = target - noise`

PointWorld-style motion weighting is optional and enabled by default. It derives
soft weights from GT motion magnitude and uses them only inside the flow loss;
static points keep a small floor weight so the model still learns zero motion.

## Example

```python
import torch

from motion_expert import (
    MotionExpertBatch,
    MotionExpertConfig,
    MotionExpertPipeline,
    SparseMotionDecoderConfig,
)

config = MotionExpertConfig(
    input_height=384,
    input_width=640,
    sparse_decoder=SparseMotionDecoderConfig(hidden_dim=256),
)
pipeline = MotionExpertPipeline(config)

batch = MotionExpertBatch(
    visual_tokens=torch.randn(2, 960, 256),
    text_tokens=torch.randn(2, 128, 256),
    query_uv=torch.rand(2, 400, 2) * 2.0 - 1.0,
    query_world_xyz=torch.randn(2, 400, 3),
    future_delta_xyz=torch.randn(2, 400, 15, 3) * 0.01,
    traj_mask=torch.ones(2, 400, 15, dtype=torch.bool),
)

outputs = pipeline(batch)
print(outputs["loss"])
```

## Using Raw VLM Inputs

```python
pipeline = MotionExpertPipeline.from_backbone(MotionExpertConfig(backbone_type="qwen"))
pipeline = MotionExpertPipeline.from_backbone(MotionExpertConfig(backbone_type="florence2"))
```

The zarr training path accepts `query_world_xyz` directly, or `depth` +
`intrinsics` + `extrinsics` as a fallback for reconstructing query XYZ.

## Validation Spaghetti Visualization

Validation generation can now save 3D spaghetti plots for moving points only:

- GT trajectory: pink
- Pred trajectory: green
- History is preserved (anchor + all valid future waypoints)

Configure in your train config under `train`:

```yaml
train:
  spaghetti_viz_max_samples: 8
  spaghetti_viz_dir: /path/to/output/spaghetti_viz  # optional
  spaghetti_motion_threshold: 0.005                 # optional (meters)
  spaghetti_max_points_per_sample: 200              # optional
```
