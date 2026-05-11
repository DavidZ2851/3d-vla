import torch

from .base import BaseDataset
from .utils import quat_wxyz_to_euler_xyz

class RoboTwin2Dataset(BaseDataset):
    """RoboTwin2 dataset."""
    cameras = ("head", "left", "right")
    train_copies = 1
    quat_format = 'wxyz'
    camera_inds = None
    max_samples = None  # max number of chunks
    required_keys = (
        "rgb_head", "rgb_left", "rgb_right",
        "depth_head", "depth_left", "depth_right",
        "extrinsics", "proprioception", "action", "rel_action", "instr_id",
    )

    def __init__(
        self,
        root,
        instructions,
        copies=None,
        relative_action=False,
        mem_limit=8,
        actions_only=False,
        chunk_size=8,
        max_samples=None,
        rotation_format="quat_wxyz",
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
        if rotation_format not in ("quat_wxyz", "euler"):
            raise ValueError(
                "RoboTwin2 data stores wxyz quaternions and supports "
                f"rotation_format='quat_wxyz' or 'euler', got {rotation_format!r}."
            )
        self.rotation_format = rotation_format
        missing = [key for key in self.required_keys if key not in self.annos]
        if missing:
            raise KeyError(
                "RoboTwin2 zarr is missing required keys: "
                f"{missing}. Expected schema with precomputed rel_action and "
                "proprioception."
            )
    
    def __len__(self):
        total_samples = len(self.annos['rel_action']) // self.chunk_size
        if self.max_samples is not None:
            return self.copies * min(self.max_samples, total_samples)
        return self.copies * total_samples
    
    def _get_task(self, idx):
        return ['robotwin2'] * self.chunk_size

    def _get_instr(self, idx):
        if not hasattr(self, "_instruction_texts"):
            instructions = self._instructions
            if isinstance(instructions, dict):
                instructions = instructions.get("instructions", [])
            if isinstance(instructions, list):
                self._instruction_texts = [
                    item["text"] if isinstance(item, dict) else item
                    for item in instructions
                ]
            else:
                self._instruction_texts = None

        if 'instr_id' in self.annos and self._instruction_texts:
            instr_ids = self.annos['instr_id'][idx:idx + self.chunk_size]
            texts = self._instruction_texts
            fallback = texts[0]
            out = []
            text_count = len(texts)
            for instr_id in instr_ids:
                instr_idx = int(instr_id[0]) if hasattr(instr_id, "__len__") else int(instr_id)
                out.append(texts[instr_idx] if 0 <= instr_idx < text_count else fallback)
            return out
        return ['instruction'] * self.chunk_size

    def _get_proprioception(self, idx):
        proprio = self._get_attr_by_idx(idx, 'proprioception', False)  # (B, T_his=1, nhand=2, dim=8)
        return proprio

    def _get_action(self, idx):
        if self._relative_action:
            rel_action = self._get_attr_by_idx(idx, 'rel_action', False)  # （B, T_pred=10, nhand=2, 8）
            return self._convert_action_rotation(rel_action)
        action = self._get_attr_by_idx(idx, 'action', False)  # (B, T_pred, nhand=2, 8)
        return self._convert_action_rotation(action)

    def _convert_action_rotation(self, action):
        if self.rotation_format != "euler":
            return action
        euler = quat_wxyz_to_euler_xyz(action[..., 3:7])
        return torch.cat([action[..., :3], euler, action[..., 7:]], dim=-1)
    
    def __getitem__(self, idx):
        """
        C is the chunk size.
        Returns:
            - task: ['robotwin2'] * C
            - instr: [str], len C
            - rgb_head: (C, 1, 3, 224, 224)
            - rgb_left: (C, 1, 3, 80, 80)
            - rgb_right: (C, 1, 3, 80, 80)
            - depth_head: (C, 1, 224, 224)
            - depth_left: (C, 1, 80, 80)
            - depth_right: (C, 1, 80, 80)
            - extrinsics: (C, 3, 4, 4)
            - proprioception: (C, 1, 2, 3+4+1)
            - action: (C, T=10, 2, 3+4+1)
        """
        # First detect which copy we fall into
        total_samples = len(self.annos['rel_action']) // self.chunk_size
        effective_size = min(self.max_samples, total_samples) if self.max_samples is not None else total_samples
        idx = idx % effective_size
        # and then which chunk
        idx = idx * self.chunk_size
        if self._actions_only:
            return {"action": self._get_action(idx)}
        return {
            "task": self._get_task(idx),
            "instr": self._get_instr(idx),
            "rgb_head": self._get_attr_by_idx(idx, 'rgb_head'),
            "rgb_left": self._get_attr_by_idx(idx, 'rgb_left'),
            "rgb_right": self._get_attr_by_idx(idx, 'rgb_right'),
            "depth_head": self._get_attr_by_idx(idx, 'depth_head'),
            "depth_left": self._get_attr_by_idx(idx, 'depth_left'),
            "depth_right": self._get_attr_by_idx(idx, 'depth_right'),
            "extrinsics": self._get_attr_by_idx(idx, 'extrinsics'),
            "proprioception": self._get_attr_by_idx(idx, 'proprioception'),
            "action": self._get_action(idx),
        }
