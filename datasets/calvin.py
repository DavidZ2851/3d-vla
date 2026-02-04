from .base import BaseDataset
from .utils import to_relative_action
import torchvision.transforms as T

class CALVINDataset(BaseDataset):
    """CALVIN dataset."""
    cameras = ("front", "wrist")
    train_copies = 1
    quat_format= 'wxyz'
    camera_inds = None
    max_samples=13  # TODO: overfit on 104 datapoints

    def __init__(
        self,
        root,
        instructions,
        copies=None,
        relative_action=False,
        mem_limit=8,
        actions_only=False,
        chunk_size=4
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

    def _get_action(self, idx):
        if self._relative_action:
            if 'rel_action' in self.annos:
                return self._get_attr_by_idx(idx, 'rel_action', False)
            else:
                action = self._get_attr_by_idx(idx, 'action', False)
                prop = self._get_proprioception(idx)[:,-1:,:]
                action = to_relative_action(action, prop, self.quat_format)
        else:
            action = self._get_attr_by_idx(idx, 'action', False)
        return action
    
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
        # idx = idx % (len(self.annos['action']) // self.chunk_size)
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
            "action": self._get_action(idx)
        }
