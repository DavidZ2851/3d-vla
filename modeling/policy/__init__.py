from .denoise_actor_3d import DenoiseActor as DenoiseActor3D
from .denoise_actor_2d import DenoiseActor as DenoiseActor2D
from .denoise_actor_3d_2dwrist import DenoiseActor as DenoiseActor2Dwrist
from .flower.model import FLOWERVLA
from .flower.model_peract2 import FLOWERVLA as FLOWERVLARLB
from .flower.model_peract2_3d import FLOWERVLA3D
from .flower_actor import (
    FlowerActor3D,
    FlowerActor3DGroundingCond,
    FlowerActor3DMotionLatent,
    FlowerActor3DMotionLatentUVD,
    FlowerActor3DMotionLatentJoint,
    FlowerActor3DMotionPartialLatent,
    FlowerActor3DMotionEncoderLatent,
    FlowerActor3DMotionTraj,
    FlowerActor3DMotionCleanLatent,
    FlowerActor3DMotion2D,
    FlowerActor2DFA,
    FlowerActor2DJoint,
    FlowerActor2DMotion2D,
    FlowerActor2DCLIP,
)

def fetch_model_class(model_type):
    if model_type == 'denoise3d':  # standard 3DFA
        return DenoiseActor3D
    if model_type == 'denoise2d':  # standard 2DFA
        return DenoiseActor2D
    if model_type == 'denoise3d_2dwrist':  # 3D front + 2D wrist
        return DenoiseActor2Dwrist
    if model_type == 'flower':
        return FLOWERVLA
    if model_type == 'flower_rlbench':
        return FLOWERVLARLB
    if model_type == 'flower_rlbench_3d':
        return FLOWERVLA3D
    if model_type == 'flower_actor_3d':
        return FlowerActor3D
    if model_type == 'flower_actor_3d_grounding_cond':
        return FlowerActor3DGroundingCond
    if model_type == 'flower_actor_3d_motion_latent':
        return FlowerActor3DMotionLatent
    if model_type == 'flower_actor_3d_motion_latent_uvd':  # UVD (u,v,depth) motion expert
        return FlowerActor3DMotionLatentUVD
    if model_type == 'flower_actor_3d_motion_latent_joint':  # joint-angle decoder on the 3D CoT
        return FlowerActor3DMotionLatentJoint
    if model_type == 'flower_actor_3d_motion_partial_latent':  # partially-denoised motion latent ablation
        return FlowerActor3DMotionPartialLatent
    if model_type == 'flower_actor_3d_motion_encoder_latent':  # encoder-only motion latent ablation
        return FlowerActor3DMotionEncoderLatent
    if model_type == 'flower_actor_3d_motion_traj':  # one-step denoise -> explicit trajectory
        return FlowerActor3DMotionTraj
    if model_type == 'flower_actor_3d_motion_clean_latent':  # one-step denoise -> clean-traj latent
        return FlowerActor3DMotionCleanLatent
    if model_type == 'flower_actor_3d_motion_2d':
        return FlowerActor3DMotion2D
    if model_type == 'flower_actor_2d':  # 2D ablation of flower_actor_3d (2dfavla)
        return FlowerActor2DFA
    if model_type == 'flower_actor_2d_joint':  # 2D visual-language joint-action baseline
        return FlowerActor2DJoint
    if model_type == 'flower_actor_2d_motion_2d':
        return FlowerActor2DMotion2D
    if model_type == 'flower_actor_2d_clip':  # legacy CLIP-only 2D actor
        return FlowerActor2DCLIP
    return None
