import torch

from .base import BaseDataset

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
        "extrinsics", "proprioception", "rel_action", "instr_id",
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
        if not relative_action:
            raise ValueError(
                "RoboTwin2 zarr only stores rel_action from robotwin2_to_zarr.py; "
                "set relative_action=True."
            )
        if rotation_format != "quat_wxyz":
            raise ValueError(
                "RoboTwin2 rel_action stores anchor-relative xyz, absolute "
                "quat_wxyz, and binary 0/1 openness. Use "
                f"rotation_format='quat_wxyz', got {rotation_format!r}."
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
        # robotwin2_to_zarr.py action15_stride2 stores:
        #   xyz      = each future_xyz - current_proprio_xyz
        #   rotation = absolute future quat_wxyz
        #   openness = binary future openness, 0=closed and 1=open
        rel_action = self._get_attr_by_idx(idx, 'rel_action', False)  # (C, T_action, 2, 8)
        return rel_action

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
            - action: (C, T=15, 2, 3+4+1)
        """
        # First detect which copy we fall into
        total_samples = len(self.annos['rel_action']) // self.chunk_size
        effective_size = min(self.max_samples, total_samples) if self.max_samples is not None else total_samples
        idx = idx % effective_size
        # and then which chunk
        idx = idx * self.chunk_size
        proprio = self._get_attr_by_idx(idx, 'proprioception')  # (C, history_len, 2, 8)
        action = self._get_action(idx)
        if self._actions_only:
            return {"action": action}
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
            "proprioception": proprio,
            "action": action,
        }


class RoboTwin2JointDataset(RoboTwin2Dataset):
    """RoboTwin2 dataset for the joint-action models.

    The joint zarr (``robotwin2_zarr_joint``) keeps the EEF fields
    (``rel_action`` / ``proprioception``, used for the 3D model's EEF
    chain-of-thought) and adds the joint targets/state:

      - ``rel_joint_action``: ``(N, H=16, nhand=2, 7)`` -- per arm, dims ``0..5``
        are anchor-relative arm-joint deltas (6 joints) and dim ``6`` is the
        binary gripper command (0=close, 1=open).
      - ``proprioception_joint``: ``(N, history=2, nhand=2, 7)`` -- 6 arm joints
        plus the binary gripper command per arm.

    Everything else (rgb/depth/proprio/instr and EEF ``action`` = ``rel_action``)
    is inherited. Requires ``relative_action=True`` and ``rotation_format
    ='quat_wxyz'`` like the parent. Dataset name: ``RoboTwin2Joint``.
    """

    required_keys = RoboTwin2Dataset.required_keys + (
        "rel_joint_action",
        "proprioception_joint",
    )

    def _get_joint_action(self, idx):
        # (C, H, nhand=2, 6 arm deltas + 1 binary gripper command)
        return self._get_attr_by_idx(idx, 'rel_joint_action', False)

    def _get_joint_proprio(self, idx):
        # (C, history, nhand=2, 6 arm joints + 1 binary gripper command)
        return self._get_attr_by_idx(idx, 'proprioception_joint', False)

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        # super() already converted idx to the raw zarr index and returned the
        # EEF sample; replicate the same index arithmetic for the joint keys.
        total_samples = len(self.annos['rel_action']) // self.chunk_size
        effective_size = (
            min(self.max_samples, total_samples)
            if self.max_samples is not None
            else total_samples
        )
        raw_idx = (idx % effective_size) * self.chunk_size
        if self._actions_only:
            sample["joint_action"] = self._get_joint_action(raw_idx)
            return sample
        sample["joint_action"] = self._get_joint_action(raw_idx)
        sample["joint_proprio"] = self._get_joint_proprio(raw_idx)
        return sample
