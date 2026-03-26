from ast import Return
from .base import BaseDataset
import torch
from torch.nn import functional as F
from utils.pytorch3d_transforms import *
from scipy.spatial.transform import Rotation as R

class LIBERODataset(BaseDataset):
    """LIBERO dataset for single-arm manipulation."""
    cameras = ("agent", "wrist")
    train_copies = 1
    quat_format = 'axisangle'
    camera_inds = None
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
        if self._relative_action:
            self.quat_format = 'axisangle'
        else:
            self.quat_format = 'rotmatrix'
        if self.max_samples is not None:
            total_samples = len(self.annos['abs_action']) // self.chunk_size
            if self.max_samples < total_samples:
                print(f"Limiting dataset to first {self.max_samples} datapoints (out of {total_samples} total)")
            else:
                print(f"max_samples ({self.max_samples}) >= total samples ({total_samples}), using all datapoints")
                self.max_samples = None

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
        Use relative action: raw_proprioception is (N, 3, NHand=1, xyz+axisAngle+openness=7).
        Use absolute action: proprioception is (N, 3, NHand=1, xyz+3x3rotMatrix+openness=13).
        """
        if self._relative_action and 'raw_proprioception' in self.annos:
            return self._get_attr_by_idx(idx, 'raw_proprioception', False)
        return self._get_attr_by_idx(idx, 'proprioception', False)

    def _get_action(self, idx):
        """
        Use relative action: control_action is (N, 3, NHand=1, xyz+axisAngle+openness=7).
        Use absolute action: abs_action is (N, T=10, NHand=1, xyz+3x3rotMatrix+openness=13).
        """
        if self._relative_action and 'control_action' in self.annos:
            return self._get_attr_by_idx(idx, 'control_action', False)
        return self._get_attr_by_idx(idx, 'abs_action', False)

    def __len__(self):
        total_samples = len(self.annos['abs_action']) // self.chunk_size
        if self.max_samples is not None:
            return self.copies * min(self.max_samples, total_samples)
        return self.copies * total_samples
    
    def _get_rgb2d(self, idx):
        return None

    def __getitem__(self, idx):
        total_samples = len(self.annos['abs_action']) // self.chunk_size
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

