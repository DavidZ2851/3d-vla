from .base import BaseDataset
import torch
from utils.pytorch3d_transforms import quaternion_to_axis_angle

class LIBERODataset(BaseDataset):
    """LIBERO dataset for single-arm manipulation."""
    cameras = ("agent", "wrist")
    train_copies = 1
    quat_format = 'xyzw'
    camera_inds = None
    max_samples = None
    gripper_open_threshold = 0.02

    def __init__( 
        self,
        root,
        instructions,
        copies=None,
        relative_action=False,
        rotation_format="axisangle",
        mem_limit=8,
        actions_only=False,
        chunk_size=8,
        max_samples=None,
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
        self.max_samples = max_samples
        if self._relative_action and rotation_format != "axisangle":
            raise ValueError(
                "LIBERO relative_action=True returns OSC delta xyz + axisangle + openness; "
                f"got rotation_format={rotation_format!r}."
            )
        if self.max_samples is not None:
            total_samples = self._total_samples() // self.chunk_size
            if self.max_samples < total_samples:
                print(f"Limiting dataset to first {self.max_samples} datapoints (out of {total_samples} total)")
            else:
                print(f"max_samples ({self.max_samples}) >= total samples ({total_samples}), using all datapoints")
                self.max_samples = None

    @staticmethod
    def _quat_xyzw_to_axisangle_xyz(pose):
        """Convert pose [..., xyz + quat_xyzw + tail] to [..., xyz + axisangle + tail]."""
        pos = pose[..., :3]
        quat_xyzw = pose[..., 3:7]
        tail = pose[..., 7:]
        quat_wxyz = quat_xyzw[..., [3, 0, 1, 2]]
        axisangle = quaternion_to_axis_angle(quat_wxyz)
        return torch.cat([pos, axisangle, tail], dim=-1)

    @classmethod
    def _binarize_gripper_width(cls, pose):
        """Convert continuous Franka gripper opening width to binary open state."""
        out = pose.clone()
        out[..., -1:] = (out[..., -1:] > cls.gripper_open_threshold).to(out.dtype)
        return out

    @staticmethod
    def _binarize_gripper_action(action):
        """Keep rel_action gripper commands as signed binary values (-1 / +1)."""
        out = action.clone()
        out[..., -1:] = torch.where(
            out[..., -1:] > 0,
            torch.ones_like(out[..., -1:]),
            -torch.ones_like(out[..., -1:]),
        )
        return out

    def _total_samples(self):
        if self._relative_action and 'rel_action' in self.annos:
            return len(self.annos['rel_action'])
        return len(self.annos['abs_action'])

    def _get_task(self, idx):
        return ['libero'] * self.chunk_size

    def _get_instr(self, idx):
        if 'instr_id' in self.annos:
            instr_ids = self.annos['instr_id'][idx:idx + self.chunk_size]
            if isinstance(self._instructions, list):
                return [
                    self._instructions[int(instr_id)] if int(instr_id) < len(self._instructions)
                    else self._instructions[0]
                    for instr_id in instr_ids
                ]
        return ['instruction'] * self.chunk_size
    
    def _get_proprioception(self, idx):
        """
        Relative mode returns xyz + axisangle + binary openness.
        Stored zarr proprioception is xyz + quat_xyzw + openness.
        """
        proprio = self._get_attr_by_idx(idx, 'proprioception', False)
        if self._relative_action:
            proprio = self._quat_xyzw_to_axisangle_xyz(proprio)
        return self._binarize_gripper_width(proprio)

    def _get_action(self, idx):
        """
        Relative mode returns the stored unnormalized OSC delta action:
        xyz delta + axisangle delta + signed binary gripper action.
        """
        if self._relative_action:
            if 'rel_action' not in self.annos:
                raise KeyError("LIBERO relative_action=True requires zarr key 'rel_action'")
            action = self._get_attr_by_idx(idx, 'rel_action', False)
            return self._binarize_gripper_action(action)
        else:
            action = self._get_attr_by_idx(idx, 'abs_action', False)
        return self._binarize_gripper_width(action)

    def __len__(self):
        total_samples = self._total_samples() // self.chunk_size
        if self.max_samples is not None:
            return self.copies * min(self.max_samples, total_samples)
        return self.copies * total_samples
    
    def _get_rgb2d(self, idx):
        return None

    def __getitem__(self, idx):
        total_samples = self._total_samples() // self.chunk_size
        effective_size = min(self.max_samples, total_samples) if self.max_samples is not None else total_samples
        idx = idx % effective_size
        idx = idx * self.chunk_size

        if self._actions_only:
            return {"action": self._get_action(idx)}
        return {
            "task": self._get_task(idx),
            "instr": self._get_instr(idx),
            "rgb": self._get_attr_by_idx(idx, 'rgb', True),
            "rgb2d": self._get_rgb2d(idx), 
            "depth": self._get_attr_by_idx(idx, 'depth', True),
            "proprioception": self._get_proprioception(idx),
            "action": self._get_action(idx),
            "extrinsics": self._get_attr_by_idx(idx, 'extrinsics', True),
            "intrinsics": self._get_attr_by_idx(idx, 'intrinsics', True)
        }
