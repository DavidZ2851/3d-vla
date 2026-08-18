"""Camera-frame-anchored CALVIN policies.

Additive variant of ``FlowerActor3DMotionLatent`` in which every geometric
quantity the policy reasons over -- the point clouds, proprioception, the
decoder's trajectory position queries, and the motion latent's token positions
-- lives in CALVIN's fixed static-camera frame instead of the world frame.

CALVIN's front camera never moves (one pose for the whole dataset, identical
between the train and val env configs), so this is a single constant rigid
transform; see ``utils/calvin_camera_frame``. The observation/action side of
the change is entirely in the data path (``CALVINCamDepth2Cloud`` /
``CALVINCamDataPreprocessor`` / ``CALVINCamTrainTester``) -- this module exists
only to keep the **motion expert** working, because the expert was trained on
world-frame 3D tracks.

The expert stays world-internal: its query is converted camera -> world before
the call (so the trained checkpoint and its world-frame normalization statistics
apply unchanged, with no retraining and no rewrite of ``calvin_motion_zarr``),
and the motion tokens it returns are then positioned at their CAMERA-frame
coordinates for the RoPE-3D scene/action fusion. This is the mirror image of
``FlowerActor3DMotionLatentUVD``, which keeps world-frame token positions around
a UVD-internal expert.

Nothing here modifies the world-frame classes in ``flower_actor.py``.
"""

import numpy as np
import torch

from utils.calvin_camera_frame import get_frame_transforms

from .flower_actor import FlowerActor3DMotionLatent


class FlowerActor3DMotionLatentCam(FlowerActor3DMotionLatent):
    """``flower_actor_3d_motion_latent`` anchored to CALVIN's static camera frame.

    Identical architecture and parameter count to the parent -- same Florence2
    encoder, same world-XYZ ``MotionExpertModel``, same gated ``CrossAttention3D``
    scene fusion and ``MotionLatentTransformerHead``. The only difference is the
    coordinate frame the geometry arrives in, plus the cam<->world hop around the
    expert.
    """

    #: Which fixed frame this policy operates in (see utils/calvin_camera_frame).
    #: Subclasses override it to reuse this class for another fixed frame.
    FRAME_NAME = "calvin_static"

    # NOTE: the parameter list is spelled out rather than forwarded through
    # **kwargs because both `utils/trainers/base.py::get_model` and
    # `online_evaluation_calvin/model_wrapper.py::get_policy` decide which
    # optional kwargs to inject by inspecting this signature. A **kwargs-only
    # ctor would hide every one of them and silently drop --motion_expert_*.
    def __init__(self,
                 # Encoder arguments
                 backbone="florence2",
                 output_level="res3",
                 upsample=False,
                 finetune_backbone=False,
                 num_finetune_vision_layers=0,
                 finetune_text_encoder=False,
                 num_vis_instr_attn_layers=2,
                 fps_subsampling_factor=5,
                 # Encoder and decoder arguments
                 embedding_dim=60,
                 num_attn_heads=9,
                 nhist=3,
                 nhand=1,
                 # Decoder arguments
                 num_shared_attn_layers=4,
                 relative=False,
                 rotation_format='quat_xyzw',
                 # Denoising arguments
                 denoise_timesteps=100,
                 denoise_model="ddpm",
                 # Motion latent arguments
                 motion_expert_config=None,
                 motion_expert_checkpoint=None,
                 motion_solver_steps=10,
                 motion_steps_to_run=1,
                 motion_finetune=True,
                 motion_scene_fusion=True,
                 motion_action_fusion=True,
                 # Geometry frame handed to the motion expert. "world" (default)
                 # converts the camera-frame query back to world so an existing
                 # world-trained expert checkpoint works as-is. "camera" feeds the
                 # camera-frame query straight through, which requires an expert
                 # trained on camera-frame tracks (see the `camera_frame` option in
                 # motion_expert/zarr_dataset.py).
                 motion_expert_frame="world",
                 # Training arguments
                 proprio_nhand=None,
                 lv2_batch_size=1):
        super().__init__(
            backbone=backbone,
            output_level=output_level,
            upsample=upsample,
            finetune_backbone=finetune_backbone,
            num_finetune_vision_layers=num_finetune_vision_layers,
            finetune_text_encoder=finetune_text_encoder,
            num_vis_instr_attn_layers=num_vis_instr_attn_layers,
            fps_subsampling_factor=fps_subsampling_factor,
            embedding_dim=embedding_dim,
            num_attn_heads=num_attn_heads,
            nhist=nhist,
            nhand=nhand,
            num_shared_attn_layers=num_shared_attn_layers,
            relative=relative,
            rotation_format=rotation_format,
            denoise_timesteps=denoise_timesteps,
            denoise_model=denoise_model,
            motion_expert_config=motion_expert_config,
            motion_expert_checkpoint=motion_expert_checkpoint,
            motion_solver_steps=motion_solver_steps,
            motion_steps_to_run=motion_steps_to_run,
            motion_finetune=motion_finetune,
            motion_scene_fusion=motion_scene_fusion,
            motion_action_fusion=motion_action_fusion,
            proprio_nhand=proprio_nhand,
            lv2_batch_size=lv2_batch_size,
        )
        if motion_expert_frame not in ("world", "camera"):
            raise ValueError(
                "motion_expert_frame must be 'world' or 'camera', got "
                f"{motion_expert_frame!r}."
            )
        self.motion_expert_frame = motion_expert_frame
        # frame -> world (4x4). Non-persistent: follows .to()/.cuda() moves but
        # is never written to or read from checkpoints, so state dicts stay
        # interchangeable with the parent's.
        #
        # Resolved from FRAME_NAME so sibling fixed-frame variants (see
        # flower_actor_yaw45.py) reuse this class verbatim. The default keeps
        # this class exactly what it was: the static-camera frame.
        _, frame2world = get_frame_transforms(self.FRAME_NAME)
        self.register_buffer(
            "cam2world",
            torch.from_numpy(np.ascontiguousarray(frame2world)).float(),
            persistent=False,
        )

    def _cam_xyz_to_world(self, query_xyz):
        """Camera-frame query points (B, P, 3) -> world frame."""
        c2w = self.cam2world.to(query_xyz.dtype)
        return query_xyz @ c2w[:3, :3].t() + c2w[:3, 3]

    def _build_motion_tokens(self, query_uv, query_xyz, instr_mask=None, cam=None):
        """Run the expert in its own frame, return tokens at CAMERA positions.

        ``query_xyz`` arrives in the camera frame (it is grid-sampled from the
        camera-frame point cloud). The parent returns ``motion["motion_pos"]``,
        which for the world expert is just the sanitized query it was given --
        so we substitute the camera-frame query instead, keeping the whole
        scene/action fusion in one frame.
        """
        expert_xyz = query_xyz
        if self.motion_expert_frame == "world":
            expert_xyz = self._cam_xyz_to_world(query_xyz.float())
        motion_tokens, _motion_pos_expert = super()._build_motion_tokens(
            query_uv, expert_xyz, instr_mask=instr_mask, cam=cam,
        )
        motion_pos_cam = torch.nan_to_num(
            query_xyz.float(), nan=0.0, posinf=0.0, neginf=0.0
        )
        return motion_tokens, motion_pos_cam.to(motion_tokens.dtype)
