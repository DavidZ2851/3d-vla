from .base import BaseDataset


class CALVINDataset(BaseDataset):
    """CALVIN dataset."""
    cameras = ("front", "wrist")
    train_copies = 1
    quat_format= 'wxyz'
    camera_inds = None

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

    def _get_task(self, idx):
        return ['calvin'] * self.chunk_size

    def _get_instr(self, idx):
        return [
            self._instructions[int(t_)]
            for t_ in self.annos['instr_id'][idx:idx + self.chunk_size]
        ]

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
        idx = idx % (len(self.annos['action']) // self.chunk_size)
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
