Motion Expert architecture

Goal: predict 20×20 sparse scene point motion in robot/world-frame xyz.

Input:
    RGB-D observation
    instruction
    camera intrinsics/extrinsics
    20×20 query UVs in the padded image canvas, normalized to [-1, 1]

Output:
    Δxyz[400, 15, 3]
1. VLM encoder

Use Florence2 or Qwen3-VL as RGB/text encoder.

RGB + instruction
    → visual feature map
    → text tokens / fused VLM tokens

For each query point, sample the visual feature at its [-1, 1] uv. This gives a per-point semantic feature.

2. Point construction

For each of the 400 query pixels:

If query_world_xyz is absent:

uv_i + depth_i + camera params → xyz_i in robot/world frame

When loading data, center query_world_xyz by the per-sample query-point mean.
The centered xyz is what the kNN graph, relative xyz bias, and local spatial
attention receive. This preserves relative geometry while removing absolute
translation from the point inputs.

Point feature:

[normalized_centered_xyz_i, uv_i, sampled_visual_feature_i]
3. Point spatial encoder

Use a lightweight kNN PointTransformer over current-frame xyz.

400 point features
    → kNN graph in centered xyz space, k=16
    → 4–6 PointTransformer layers
    → point_context[400, D]

This follows PointWorld’s 3D point-flow idea, but we only use it as a spatial encoder, not as an action-conditioned MPC world model.

4. Flow-matching motion decoder

Target:

X1 = Δxyz[400, 15, 3]

Normalize both centered query xyz and target Δxyz as (value - mean) / std, then
clamp to [-2, 2]. For CALVIN zarr training, these mean/std values are computed
from the train zarr; query stats are accumulated after per-sample centering.

Flow matching:

X0 ~ N(0, I)
τ ~ Uniform(0, 1)
Xτ = (1 - τ)X0 + τX1
v_target = X1 - X0

Decoder predicts:

v_pred = vθ(Xτ, τ, point_context, text/VLM tokens)

Loss:

||v_pred - v_target||²

This follows TraceGen-style flow matching, but our output is robot/world-frame xyz displacement, not camera/screen trace.

5. Decoder block

Use PhysCtrl-style spatial→temporal factorization, adapted for sparse scene points.

Each decoder block:

1. Local spatial attention
   For each future timestep:
       each point attends to fixed kNN neighbors from current xyz graph
       use relative xyz / distance / uv bias

2. Temporal attention
   For each point:
       attend over 15 future timesteps

3. Global visual/text conditioning
   Cross-attention from Florence2/Qwen text tokens and global visual token(s)

4. FFN

Important: the spatial graph is built from current observed xyz, not noisy future points.

6. Moving-point weighted loss

Borrow PointWorld’s moving-point emphasis as loss weighting only.

motion_score = sigmoid((||Δxyz|| - τ) * temp / τ)
loss_weight = traj_mask * max(motion_score^γ, static_floor)

The only supervision mask is traj_mask. The motion score is not an input token and not an additional mask.

7. Action expert fusion, later stage

Use LaMP-style gated cross-attention.

motion hidden states z_m
    → projector
    → K,V for action expert cross-attention

action_tokens ← action_tokens + sigmoid(g) * CrossAttn(action_tokens, motion_tokens)

Initialize gate near zero.

Final one-line summary

Our standalone motion expert = PointWorld-style 3D point encoding + TraceGen-style flow matching + PhysCtrl-style local spatial/temporal trajectory decoder. LaMP-style gated fusion can be added after standalone flow metrics are stable.
