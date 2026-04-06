"""
Convert CALVIN zarr dataloader samples (datasets.calvin.CALVINDataset) to MiBoT
inputs: center-cropped PIL views and 32-D state, matching XiaomiModel.preprocess/step.
"""
import numpy as np
import torch

from online_evaluation_calvin.xiaomi_model_wrapper import XiaomiModel


def calvin_sample_to_mibot_obs(
    sample: dict,
    frame_idx: int = 0,
    crop_ratio: float = 0.95,
    state_pad_dim: int = 32,
):
    """
    Map one timestep from a CALVINDataset __getitem__ dict to the observation dict
    consumed by XiaomiModel.step (after preprocess-style fields).

    Dataloader layout (see datasets/calvin.py):
      - rgb: (chunk_size, 1, 3, H, W) — static cam
      - rgb2d: (chunk_size, 1, 3, h, w) — wrist cam
      - proprioception: (chunk_size, 1, 1, 7) — zarr stores pos(3)+euler(3) and a
        binarized gripper command (last index), not the continuous opening width
        used in live env robot_obs[6]. MiBoT was fed env-style poses online; if
        you need a perfect match to sim proprio, prefer env observations.

    Args:
        sample: Batch dict from CALVINDataset (single __getitem__).
        frame_idx: Which frame along the chunk axis (0 .. chunk_size-1).
        crop_ratio: Same as XiaomiModel center-crop resize.
        state_pad_dim: Total state length; first 7 come from proprio, rest zeros.

    Returns:
        dict with keys "base", "wrist_left" (PIL.Image), "state" (np.ndarray (32,)).
    """
    if frame_idx < 0 or frame_idx >= sample["rgb"].shape[0]:
        raise IndexError(
            f"frame_idx={frame_idx} out of range for chunk size {sample['rgb'].shape[0]}"
        )

    rgb_t = sample["rgb"][frame_idx, 0]  # (3, H, W)
    wrist_t = sample["rgb2d"][frame_idx, 0]
    prop_t = sample["proprioception"][frame_idx, 0, 0]

    if isinstance(prop_t, torch.Tensor):
        prop_np = prop_t.detach().cpu().float().numpy()
    else:
        prop_np = np.asarray(prop_t, dtype=np.float32)

    prop_np = np.asarray(prop_np).reshape(-1).astype(np.float64)
    if prop_np.size != 7:
        raise ValueError(f"Expected proprio dim 7, got shape {prop_np.shape}")

    pad = state_pad_dim - prop_np.size
    if pad < 0:
        raise ValueError(f"state_pad_dim {state_pad_dim} < proprio size {prop_np.size}")
    state = np.concatenate([prop_np, np.zeros(pad, dtype=np.float64)], axis=-1)

    rgb_static = XiaomiModel._to_pil(rgb_t)
    rgb_gripper = XiaomiModel._to_pil(wrist_t)
    rgb_static, rgb_gripper = XiaomiModel._center_crop_resize(
        [rgb_static, rgb_gripper], crop_ratio
    )

    return {"base": rgb_static, "wrist_left": rgb_gripper, "state": state}


def calvin_instruction_at(sample: dict, frame_idx: int = 0) -> str:
    """Language string for the given frame in the chunk."""
    instr = sample["instr"][frame_idx]
    if not isinstance(instr, str):
        instr = str(instr)
    return instr
