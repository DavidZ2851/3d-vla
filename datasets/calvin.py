from .base import BaseDataset
from .utils import to_relative_action, to_tensor
import torchvision.transforms as T, torch

class CALVINDataset(BaseDataset):
    """CALVIN dataset."""
    cameras = ("front", "wrist")
    train_copies = 1
    quat_format= 'euler'
    camera_inds = None
    max_samples=None  # max number of chunks

    def __init__(
        self,
        root,
        instructions,
        copies=None,
        relative_action=False,
        mem_limit=8,
        actions_only=False,
        chunk_size=8,
        max_samples=None
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
    
    def __len__(self):
        total_samples = len(self.annos['action']) // self.chunk_size
        if self.max_samples is not None:
            return self.copies * min(self.max_samples, total_samples)
        return self.copies * total_samples
    
    def _get_task(self, idx):
        return ['calvin'] * self.chunk_size

    def _get_instr(self, idx):
        return [
            self._instructions[int(t_)]
            for t_ in self.annos['instr_id'][idx:idx + self.chunk_size]
        ]

    def _get_proprioception(self, idx):
        original_proprio = self._get_attr_by_idx(idx, 'proprioception', False)  # (B, T_his=1, nhand=1, dim=7)
        return original_proprio
        # NOTE: if use padding to two hands
        # zero_pad = torch.zeros_like(original_proprio)  # euler default (0,0,0) means no rotation
        # return torch.cat([original_proprio, zero_pad], dim=2)  # (B, T_his=1, nhand=2, 7)

    def _get_action(self, idx):
        if self._relative_action and 'rel_action' in self.annos:
            rel_action = self._get_attr_by_idx(idx, 'rel_action', False)  # （B, T_pred=10, nhand=1, 7）
            return rel_action
            # NOTE: if use padding to two hands
            # zero_pad = torch.zeros_like(rel_action)  # euler default (0,0,0) means no rotation
            # action_padded = torch.cat([rel_action, zero_pad], dim=2)  # (B, T_pred=10, nhand=2, 7)
            # return action_padded
        original_action = self._get_attr_by_idx(idx, 'action', False)  # (B, T_pred, nhand=1, 7)
        # NOTE: if use padding to two hands
        # zero_pad = torch.zeros_like(original_action)
        # action_padded = torch.cat([original_action, zero_pad], dim=2)  # (B, T_pred, 2, 7)
        if self._relative_action:
            return ValueError("No implementation for euler relative calculation")
        return original_action
        # return action_padded
    
    def __getitem__(self, idx):
        """
        C is the chunk size.
        Returns:
            - task: ['calvin'] * C
            - instr: [str], len C
            - rgb: (C, 1, 3, 200, 200)
            - depth: (C, 1, 200, 200)
            - rgb2d: (C, 1, 3, 84, 84)
            - wrist_depth: (C, 1, 84, 84)
            - extrinsics_wrist: (C, 4, 4)
            - proprioception: (C, 1, 1, 3+3+1)
            - action: (C, T=10, 1, 3+3+1)
        """
        # First detect which copy we fall into
        total_samples = len(self.annos['action']) // self.chunk_size
        effective_size = min(self.max_samples, total_samples) if self.max_samples is not None else total_samples
        idx = idx % effective_size
        # and then which chunk
        idx = idx * self.chunk_size
        if self._actions_only:
            return {"action": self._get_action(idx)}
        return {
            "task": self._get_task(idx),
            "instr": self._get_instr(idx),
            "rgb": self._get_attr_by_idx(idx, 'rgb_front'),
            "depth": self._get_attr_by_idx(idx, 'depth_front'),
            "rgb2d": self._get_attr_by_idx(idx, 'rgb_wrist'),
            "wrist_depth": self._get_attr_by_idx(idx, 'depth_wrist'),
            "extrinsics_wrist": self._get_attr_by_idx(idx, 'extrinsics_wrist'),
            "proprioception": self._get_attr_by_idx(idx, 'proprioception'),
            "action": self._get_action(idx),
        }


class CALVINContactDataset(CALVINDataset):
    """CALVIN dataset extended with 3D contact point supervision.

    Adds a ``3d_contact_point`` key to every batch item when the zarr archive
    contains a ``3d_contact_point`` array of shape ``(N, 3)`` (world-frame xyz,
    one entry per frame, same value repeated within a chunk since the contact
    point is the task goal for the whole chunk).

    When the array is absent the key is still present but set to ``None``; the
    trainer and model both handle ``None`` gracefully (auxiliary loss is
    skipped, conditioning is a no-op).

    Dataset names: ``CalvinContact`` / ``CalvinContact10th``.
    """

    _CONTACT_KEY = '3d_contact_point'

    def _get_contact_point(self, idx):
        """Return (chunk_size, 3) float32 tensor, or None if key not in zarr.

        The zarr array is expected to have shape ``(N, 3)`` where N matches the
        total number of frames.  Slicing ``[idx : idx + chunk_size]`` yields
        ``(chunk_size, 3)`` so that ``torch.cat`` in the collate function
        produces ``(batch_size, 3)`` -- matching the shape the model expects.
        """
        if self._CONTACT_KEY not in self.annos:
            return None
        return to_tensor(
            self.annos[self._CONTACT_KEY][idx : idx + self.chunk_size]
        ).float()  # (chunk_size, 3)

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        if not self._actions_only:
            # idx was already converted to the raw zarr index inside super();
            # replicate the same index arithmetic to read the contact point.
            total_samples = len(self.annos['action']) // self.chunk_size
            effective_size = (
                min(self.max_samples, total_samples)
                if self.max_samples is not None
                else total_samples
            )
            raw_idx = (idx % effective_size) * self.chunk_size
            sample['3d_contact_point'] = self._get_contact_point(raw_idx)
        return sample
