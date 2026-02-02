from .base import BaseDataset
from .utils import convert_quaternion_format


class GenieSim3Dataset(BaseDataset):
    """GenieSim3.0 dataset for bimanual manipulation."""

    cameras = ("head_front", "hand_left", "hand_right")
    train_copies = 1
    quat_format = 'wxyz'
    output_quat_format = 'wxyz'  # keep wxyz end-to-end for GenieSim3.0 training
    camera_inds = None  # Use all cameras by default
    camera_inds2d = None  # No 2D cameras for now

    def __init__(
        self,
        root,
        instructions,
        copies=None,
        relative_action=False,
        mem_limit=8,
        actions_only=False,
        chunk_size=4,
        use_robot_base=False,
        use_head_front=True,
        input_quat_format=None,  # quaternion format in input data: 'wxyz' or 'xyzw'. If None, uses 'wxyz'
        max_samples=100,
    ):
        super().__init__(
            root=root,
            instructions=instructions,
            copies=copies,
            relative_action=relative_action,
            mem_limit=mem_limit,
            actions_only=actions_only,
            chunk_size=chunk_size,
            input_quat_format=input_quat_format,
        )
        self.use_robot_base = use_robot_base
        self.use_head_front = use_head_front
        if use_head_front:
            self.camera_inds = [0]
        self._max_samples = max_samples
        if self._max_samples is not None:
            total_samples = len(self.annos['action']) // self.chunk_size
            if self._max_samples < total_samples:
                print(f"Limiting dataset to first {self._max_samples} datapoints (out of {total_samples} total)")
            else:
                print(f"max_samples ({self._max_samples}) >= total samples ({total_samples}), using all datapoints")
                self._max_samples = None

    def _get_task(self, idx):
        return ['bimanual_lift_ball'] * self.chunk_size

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
        return self._get_attr_by_idx(idx, 'extrinsics', True)

    def _get_intrinsics(self, idx):
        return self._get_attr_by_idx(idx, 'intrinsics', True)

    def _get_rgb2d(self, idx):
        return None

    def _get_proprioception(self, idx):
        """
        From zarr: proprioception is (N, 3, NHand+1, 8).
        Returns: (chunk_size, 3, NHand, 8) (hands only, base excluded).
        """
        prop = self._get_attr_by_idx(idx, 'proprioception', False)
        hand_prop = prop[:, :, :2, :]
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
        if self._relative_action:
            #print('[DEBUG] read dataset action before relative:', hand_action[0,1,:,:])
            prop = self._get_proprioception(idx)
            anchor_action = prop[:, -1, :, :].unsqueeze(-2)  # (chunk_size, nhand, 1, 8)
            hand_action = hand_action.transpose(1, 2)  # (chunk_size, nhand, T, 8)
            from .utils import to_relative_action
            hand_action = to_relative_action(hand_action, anchor_action, 'wxyz')
            hand_action = hand_action.transpose(1, 2)  # (chunk_size, T, nhand, 8)
            #print('[DEBUG] read dataset action after relative:', hand_action[0,1,:,:])
        return hand_action

    def _get_robot_base_proprioception(self, idx):
        prop = self._get_attr_by_idx(idx, 'proprioception', False)
        base_prop = prop[:, :, -1, :7]  # (chunk_size, 3, 7)
        if self._needs_quat_conversion:
            base_prop = convert_quaternion_format(base_prop, self._input_quat_format, self._output_quat_format)
        return base_prop.unsqueeze(2)  # (chunk_size, 3, 1, 7)

    def _get_robot_base_action(self, idx):
        action = self._get_attr_by_idx(idx, 'action', False)
        base_action = action[:, :, -1, :2]  # (chunk_size, T, 2)
        return base_action.unsqueeze(2)  # (chunk_size, T, 1, 2)

    def __len__(self):
        total_samples = len(self.annos['action']) // self.chunk_size
        if self._max_samples is not None:
            return self.copies * min(self._max_samples, total_samples)
        return self.copies * total_samples

    def __getitem__(self, idx):
        total_samples = len(self.annos['action']) // self.chunk_size
        effective_size = min(self._max_samples, total_samples) if self._max_samples is not None else total_samples
        idx = idx % effective_size
        idx = idx * self.chunk_size

        if self._actions_only:
            result = {"action": self._get_action(idx)}
            if self.use_robot_base:
                result["robot_base_action"] = self._get_robot_base_action(idx)
            return result

        result = {
            "task": self._get_task(idx),
            "instr": self._get_instr(idx),
            "rgb": self._get_rgb(idx),
            "rgb2d": self._get_rgb2d(idx),
            "depth": self._get_depth(idx),
            "proprioception": self._get_proprioception(idx),
            "action": self._get_action(idx),
            "extrinsics": self._get_extrinsics(idx),
            "intrinsics": self._get_intrinsics(idx),
        }
        if self.use_robot_base:
            result["robot_base_proprioception"] = self._get_robot_base_proprioception(idx)
            result["robot_base_action"] = self._get_robot_base_action(idx)
        return result

