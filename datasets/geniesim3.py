from .base import BaseDataset
import torch
from torch.nn import functional as F
from utils.pytorch3d_transforms import *
from scipy.spatial.transform import Rotation as R

ORIGINAL_HEIGHT = 400
ORIGINAL_WIDTH = 640
LOAD_IMG_SIZE = 256

def resize_intrinsics(K, orig_w, orig_h, new_size):
    new_w = new_size
    new_h = new_size
    scale_x = new_w / orig_w
    scale_y = new_h / orig_h
    K_new = K.clone()
    K_new[..., 0, 0] *= scale_x  # fx
    K_new[..., 1, 1] *= scale_y  # fy
    K_new[..., 0, 2] *= scale_x  # cx
    K_new[..., 1, 2] *= scale_y  # cy
    return K_new

def _resize_rgb_to_load_size(rgb, target_size):
    """Resize rgb (chunk_size, n_cam, 3, H, W) to (..., target_size, target_size)."""
    *batch_dims, C, H, W = rgb.shape
    if H == target_size and W == target_size:
        return rgb
    rgb_flat = rgb.reshape(-1, C, H, W)
    rgb_resized = F.interpolate(
        rgb_flat, (target_size, target_size),
        mode='bilinear', antialias=True, align_corners=False
    )
    return rgb_resized.reshape(*batch_dims, C, target_size, target_size)

def _resize_depth_to_load_size(depth, target_size):
    """Resize depth (chunk_size, n_cam, H, W) to (..., target_size, target_size)."""
    *batch_dims, H, W = depth.shape
    if H == target_size and W == target_size:
        return depth
    depth_flat = depth.reshape(-1, 1, H, W)
    depth_resized = F.interpolate(
        depth_flat, (target_size, target_size),
        mode='bilinear', align_corners=False
    )
    return depth_resized.squeeze(1).reshape(*batch_dims, target_size, target_size)

class GenieSim3Dataset(BaseDataset):
    """GenieSim3.0 dataset for bimanual manipulation."""

    cameras = ("head_front", "hand_left", "hand_right")
    train_copies = 1
    quat_format = 'wxyz'
    camera_inds = [0]
    max_samples = None

    def __init__( 
        self,
        root,
        instructions,
        copies=None,
        relative_action=False,
        mem_limit=8,
        actions_only=False,
        chunk_size=8,
    ):
        super().__init__(
            root=root,
            instructions=instructions,
            copies=copies,
            relative_action=relative_action,
            mem_limit=mem_limit,
            actions_only=actions_only,
            chunk_size=chunk_size
        )
        if self.max_samples is not None:
            total_samples = len(self.annos['action']) // self.chunk_size
            if self.max_samples < total_samples:
                print(f"Limiting dataset to first {self.max_samples} datapoints (out of {total_samples} total)")
            else:
                print(f"max_samples ({self.max_samples}) >= total samples ({total_samples}), using all datapoints")
                self.max_samples = None
        R_EE_TOOL = torch.tensor(
            R.from_euler('y', -90, degrees=True).as_matrix()
            @ R.from_euler('x', -90, degrees=True).as_matrix()
        )
        R_EE_TOOL_INV = R_EE_TOOL.T
        R_roll_fix = torch.tensor([[0, 1, 0], [-1,0, 0], [0, 0, 1]], dtype=R_EE_TOOL.dtype, device=R_EE_TOOL.device)   # Rz(-90)
        R_EE_TOOL_INV = R_EE_TOOL_INV @ R_roll_fix
        self.R_EE_TOOL_INV_QUAT = matrix_to_quaternion(R_EE_TOOL_INV)

    def _get_task(self, idx):
        return ['geniesim'] * self.chunk_size

    def _get_instr(self, idx):
        # Use instr_id from zarr to map to instructions list
        if 'instr_id' in self.annos:
            instr_ids = self.annos['instr_id'][idx:idx + self.chunk_size]
            if isinstance(self._instructions, list):
                return [
                    self._instructions[int(instr_id)] if int(instr_id) < len(self._instructions)
                    else self._instructions[0]
                    for instr_id in instr_ids
                ]
        # Fallback: use chunk index
        if isinstance(self._instructions, list):
            chunk_start_idx = idx // self.chunk_size if self.chunk_size > 1 else idx
            instr_idx = chunk_start_idx % len(self._instructions)
            return [self._instructions[instr_idx]] * self.chunk_size
        return ['instruction'] * self.chunk_size

    def _get_extrinsics(self, idx):
        extrinsics = self._get_attr_by_idx(idx, 'extrinsics', True).clone()
        # Check the 3rd row, 3rd column for the Z-axis direction
        wrong_mask = extrinsics[..., 2, 2] > 0 
        if wrong_mask.any():
            # Matrix to rotate 180 degrees around the local X-axis
            fix_matrix = torch.tensor([
                [1.0,  0.0,  0.0, 0.0],
                [0.0, -1.0,  0.0, 0.0],
                [0.0,  0.0, -1.0, 0.0],
                [0.0,  0.0,  0.0, 1.0]
            ], dtype=extrinsics.dtype, device=extrinsics.device)
            # Right-multiply to apply the rotation in the local camera frame
            extrinsics[wrong_mask] = extrinsics[wrong_mask] @ fix_matrix
        return extrinsics

    def _get_rgb(self, idx, key='rgb'):
        rgb = self._get_attr_by_idx(idx, key, True)
        return rgb  #_resize_rgb_to_load_size(rgb, LOAD_IMG_SIZE)

    def _get_depth(self, idx, key='depth'):
        depth = self._get_attr_by_idx(idx, key, True)
        return depth  #_resize_depth_to_load_size(depth, LOAD_IMG_SIZE)

    def _get_intrinsics(self, idx):
        K = self._get_attr_by_idx(idx, 'intrinsics', True)
        return K  #resize_intrinsics(K, ORIGINAL_WIDTH, ORIGINAL_HEIGHT, LOAD_IMG_SIZE)
    
    def _get_proprioception(self, idx):
        """
        From zarr: proprioception is (N, 3, NHand+1, 8).
        Returns: (chunk_size, 3, NHand, 8) (hands only, base excluded).
        """
        prop = self._get_attr_by_idx(idx, 'proprioception', False)
        hand_prop = prop[:, :, :2, :]
        quat = hand_prop[..., 3:7]  # wxyz
        quat_fixed = quaternion_multiply(hand_prop[..., 3:7], self.R_EE_TOOL_INV_QUAT.to(quat.device))
        hand_prop[..., 3:7] = quat_fixed
        return hand_prop

    def _get_action(self, idx):
        """
        From zarr: action is (N, T, NHand+1, 8).
        Returns: (chunk_size, T, NHand, 8) (hands only, base excluded).
        """
        if self._relative_action and 'rel_action' in self.annos:
            rel_action = self._get_attr_by_idx(idx, 'rel_action', False)
            hand_action = rel_action[:, :, :-1, :]
            return hand_action

        action = self._get_attr_by_idx(idx, 'action', False)
        hand_action = action[:, :, :2, :]  # the first two dims are definitely hands
        quat = hand_action[..., 3:7]  # wxyz
        quat_fixed = quaternion_multiply(hand_action[..., 3:7], self.R_EE_TOOL_INV_QUAT.to(quat.device))
        hand_action[..., 3:7] = quat_fixed
        if self._relative_action:
            prop = self._get_proprioception(idx)
            anchor_action = prop[:, -1, :, :].unsqueeze(-2)  # (chunk_size, nhand, 1, 8)
            hand_action = hand_action.transpose(1, 2)  # (chunk_size, nhand, T, 8)
            from .utils import to_relative_action
            hand_action = to_relative_action(hand_action, anchor_action, 'wxyz')
            hand_action = hand_action.transpose(1, 2)  # (chunk_size, T, nhand, 8)
        return hand_action

    def __len__(self):
        total_samples = len(self.annos['action']) // self.chunk_size
        if self.max_samples is not None:
            return self.copies * min(self.max_samples, total_samples)
        return self.copies * total_samples
    
    def _get_rgb2d(self, idx):
        return None

    def __getitem__(self, idx):
        total_samples = len(self.annos['action']) // self.chunk_size
        effective_size = min(self.max_samples, total_samples) if self.max_samples is not None else total_samples
        idx = idx % effective_size
        idx = idx * self.chunk_size

        if self._actions_only:
            return {"action": self._get_action(idx)}
        return {
            "task": self._get_task(idx),
            "instr": self._get_instr(idx),
            "rgb": self._get_rgb(idx),
            "rgb2d": self._get_rgb2d(idx), 
            "depth": self._get_depth(idx),
            "proprioception": self._get_proprioception(idx),
            "action": self._get_action(idx),
            "extrinsics": self._get_extrinsics(idx),
            "intrinsics": self._get_intrinsics(idx)
        }

