import json

from torch.utils.data import Dataset

from .utils import to_tensor, read_zarr_with_cache, to_relative_action, convert_quaternion_format


class BaseDataset(Dataset):
    """Base dataset."""

    def __init__(
        self,
        root,  # the directory path of the dataset
        instructions,  # path to instruction file
        copies=None,  # copy the dataset for less loader restarts
        relative_action=False,  # whether to return relative actions
        mem_limit=8,  # cache limit per dataset class in GigaBytes
        actions_only=False,  # return actions without observations
        chunk_size=4,  # chunk size for zarr
        input_quat_format=None  # quaternion format in input data: 'wxyz' or 'xyzw'. If None, uses self.quat_format
    ):
        super().__init__()
        self.copies = self.train_copies if copies is None else copies
        self._relative_action = relative_action
        self._actions_only = actions_only
        self.chunk_size = chunk_size
        if input_quat_format is None:
            input_quat_format = getattr(self, 'quat_format', 'wxyz')
        self._input_quat_format = input_quat_format
        self._output_quat_format = getattr(self, 'output_quat_format', 'wxyz')
        self._needs_quat_conversion = (self._input_quat_format != self._output_quat_format)
        if self._needs_quat_conversion:
            print(f"Dataset will convert quaternions from {self._input_quat_format} to {self._output_quat_format}")

        # Load instructions
        self._instructions = self._load_instructions(instructions)

        # Load all annotations lazily
        self.annos = read_zarr_with_cache(root, mem_gb=mem_limit)
        # Sanity check
        len_ = len(self.annos['action'])
        for key in self.annos:
            assert len(self.annos[key]) == len_, f'length mismatch in {key}'
        print(f"Found {len(self.annos['action'])} samples")

    def _load_instructions(self, instruction_file):
        return json.load(open(instruction_file))

    def _get_attr_by_idx(self, idx, attr, filter_cam=False):
        t = to_tensor(self.annos[attr][idx:idx + self.chunk_size])
        if filter_cam and self.camera_inds is not None:
            t = t[:, self.camera_inds]
        return t

    def _get_task(self, idx):
        return ["task"] * self.chunk_size

    def _get_instr(self, idx):
        return ["instruction"] * self.chunk_size

    def _get_rgb(self, idx, key='rgb'):
        return self._get_attr_by_idx(idx, key, True)

    def _get_depth(self, idx, key='depth'):
        return self._get_attr_by_idx(idx, key, True)

    def _get_proprioception(self, idx):
        prop = self._get_attr_by_idx(idx, 'proprioception', False)
        return prop

    def _get_action(self, idx):
        if self._relative_action:
            if 'rel_action' in self.annos:
                action = self._get_attr_by_idx(idx, 'rel_action', False)
                if self._needs_quat_conversion:
                    action = convert_quaternion_format(action, self._input_quat_format, self._output_quat_format)
                return action
            else:
                action = self._get_attr_by_idx(idx, 'action', False)
                if self._needs_quat_conversion:
                    action = convert_quaternion_format(action, self._input_quat_format, self._output_quat_format)
                prop = self._get_proprioception(idx)  # already converted if needed
                # For single-hand: prop is (chunk_size, nhist, 8) -> last history step is (chunk_size, 1, 8)
                # For multi-hand: prop is (chunk_size, nhist, nhand, 8) -> last history is (chunk_size, nhand, 1, 8)
                if len(prop.shape) == 3:
                    anchor_action = prop[:, -1:, :]  # (chunk_size, 1, 8)
                elif len(prop.shape) == 4:
                    anchor_action = prop[:, -1, :, :].unsqueeze(-2)  # (chunk_size, nhand, 1, 8)
                    if len(action.shape) == 4:
                        action = action.transpose(1, 2)  # (chunk_size, nhand, T, 8)
                else:
                    raise ValueError(f"Unexpected proprioception shape: {prop.shape}")
                action = to_relative_action(action, anchor_action, self._output_quat_format)
                if len(prop.shape) == 4 and len(action.shape) == 4:
                    action = action.transpose(1, 2)  # (chunk_size, T, nhand, 8)
        else:
            action = self._get_attr_by_idx(idx, 'action', False)
            if self._needs_quat_conversion:
                action = convert_quaternion_format(action, self._input_quat_format, self._output_quat_format)
        return action

    def __getitem__(self, idx):
        """
        self.annos: {
            action: (N, T, 8) float
            depth: (N, n_cam, H, W) float16
            proprioception: (N, nhist, 8) float
            rgb: (N, n_cam, 3, H, W) uint8
        }
        In addition self.annos may contain fields for task/instruction ids
        """
        # First detect which copy we fall into
        idx = idx % (len(self.annos['action']) // self.chunk_size)
        # and then which chunk
        idx = idx * self.chunk_size
        if self._actions_only:
            return {"action": self._get_action(idx)}
        return {
            "task": self._get_task(idx),
            "instr": self._get_instr(idx),  # [str]
            "rgb": self._get_rgb(idx),  # tensor(n_cam, 3, H, W)
            "depth": self._get_depth(idx),  # tensor(n_cam, H, W)
            "proprioception": self._get_proprioception(idx),  # tensor(1, 8)
            "action": self._get_action(idx)  # tensor(T, 8)
        }

    def __len__(self):
        return self.copies * (len(self.annos['action']) // self.chunk_size)
