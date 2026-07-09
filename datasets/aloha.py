"""
ALOHA real-world dataset for single-arm (right) manipulation.

Source zarr schema (produced by data_processing/aloha_to_zarr.py):

    rgb_front      : (N, 1, 3, H=224, W=398) uint8
    rgb_back       : (N, 1, 3, H=224, W=398) uint8
    rgb_wrist      : (N, 1, 3, H=224, W=398) uint8
    depth_front    : (N, 1, H=224, W=398) float32          (meters)
    depth_back     : (N, 1, H=224, W=398) float32          (meters)
    proprioception : (N, NHAND=1, 10) float32              (rot6d[6] + trans[3] + gripper[1])
    rel_action     : (N, K=15, NHAND=1, 10) float32        (rot6d_delta[6] + trans_delta[3] + gripper_abs[1])
    instr_id       : (N, 1) int32

`rel_action` is built with proper SO(3) deltas:
    rot6d_delta = first_two_cols( R(target[t+k]) @ R(proprio[t]).T )
    trans_delta = target[t+k].trans - proprio[t].trans
    gripper     = target.gripper   (absolute, NOT a delta)
Every horizon step is anchor-relative to the current proprio row, not
step-to-step incremental.

The model is trained with `rotation_format='rot6d'`, which expects each pose to be
laid out as [xyz(3), rot6d(6), gripper(1)] = 10 dims. We reorder the zarr's
[rot6d, trans, grip] -> [xyz, rot6d, grip] here. The preferred ALOHA path now
pre-binarizes gripper openness in the LeRobot dataset before zarr conversion, so
binary zarrs keep gripper as 1=open / 0=closed. Older raw-degree zarrs are still
detected and divided by 100 for compatibility.
"""

import torch

from .base import BaseDataset
from .utils import to_tensor


GRIPPER_NORMALIZER = 100.0
EXPECTED_ACTION_HORIZON = 15


def _reorder_pose10(pose10: torch.Tensor, gripper_scale: float) -> torch.Tensor:
    """Zarr [rot6d(6), trans(3), grip(1)] -> model [xyz(3), rot6d(6), grip_norm(1)]."""
    rot6d = pose10[..., 0:6]
    trans = pose10[..., 6:9]
    grip = pose10[..., 9:10] * gripper_scale
    return torch.cat([trans, rot6d, grip], dim=-1)


class AlohaDataset(BaseDataset):
    """ALOHA real-world single-arm dataset (right arm only).

    Multi-camera: front + back kinects are 3D (rgb + depth), the wrist is 2D
    (rgb only). The trainer builds rgb3d=(front, back), rgb2d=(wrist), and
    world-frame pcd=(front, back) from these keys.
    """

    cameras = ("front", "back", "wrist")
    train_copies = 1
    quat_format = "wxyz"  # unused (rot6d path); kept to satisfy BaseDataset interface.
    camera_inds = None
    max_samples = None

    required_keys = (
        "rgb_front", "rgb_back", "rgb_wrist", "depth_front", "depth_back",
        "proprioception", "rel_action", "instr_id",
    )

    def __init__(
        self,
        root,
        instructions,
        copies=None,
        relative_action=True,
        mem_limit=8,
        actions_only=False,
        chunk_size=8,
        max_samples=None,
        rotation_format="rot6d",
    ):
        super().__init__(
            root=root,
            instructions=instructions,
            copies=copies,
            relative_action=relative_action,
            mem_limit=mem_limit,
            actions_only=actions_only,
            chunk_size=chunk_size,
        )
        self.max_samples = max_samples
        if rotation_format != "rot6d":
            raise ValueError(
                "ALOHA dataset emits rot6d-laid-out poses; only "
                f"rotation_format='rot6d' is supported, got {rotation_format!r}."
            )
        self.rotation_format = rotation_format
        if not relative_action:
            raise ValueError(
                "ALOHA zarr only stores rel_action; set relative_action=True."
            )
        missing = [k for k in self.required_keys if k not in self.annos]
        if missing:
            raise KeyError(
                f"ALOHA zarr is missing required keys: {missing}. "
                f"Found keys: {list(self.annos.keys())}"
            )
        horizon = int(self.annos["rel_action"].shape[1])
        if horizon != EXPECTED_ACTION_HORIZON:
            raise ValueError(
                f"ALOHA rel_action horizon is {horizon}; expected "
                f"{EXPECTED_ACTION_HORIZON}. `chunk_size` controls how many "
                "zarr datapoints are loaded together and must not be used as "
                "the action horizon."
            )
        self.gripper_scale = self._detect_gripper_scale()

    def _detect_gripper_scale(self) -> float:
        sample = to_tensor(self.annos["rel_action"][: min(256, len(self.annos["rel_action"]))])
        grip = sample[..., 9]
        gmin = float(grip.min())
        gmax = float(grip.max())
        if -1e-6 <= gmin and gmax <= 1.0 + 1e-6:
            print("ALOHA gripper labels detected as binary openness (1=open, 0=closed).")
            return 1.0
        print("ALOHA gripper labels detected as raw degrees; scaling by /100.")
        return 1.0 / GRIPPER_NORMALIZER

    def __len__(self):
        total = len(self.annos["rel_action"]) // self.chunk_size
        if self.max_samples is not None:
            return self.copies * min(self.max_samples, total)
        return self.copies * total

    def _get_task(self, idx):
        return ["aloha"] * self.chunk_size

    def _get_instr(self, idx):
        instructions = self._instructions
        if not isinstance(instructions, list):
            instructions = instructions.get("instructions", []) if isinstance(instructions, dict) else []
        texts = [
            item["text"] if isinstance(item, dict) else item
            for item in instructions
        ]
        fallback = texts[0] if texts else "instruction"

        if "instr_id" not in self.annos or not texts:
            return [fallback] * self.chunk_size

        instr_ids = self.annos["instr_id"][idx:idx + self.chunk_size]
        out = []
        for instr_id in instr_ids:
            ii = int(instr_id[0]) if hasattr(instr_id, "__len__") else int(instr_id)
            out.append(texts[ii] if 0 <= ii < len(texts) else fallback)
        return out

    def _get_proprioception(self, idx):
        # zarr: (chunk, NHAND=1, 10) -> add nhist axis -> (chunk, nhist=1, 1, 10)
        proprio10 = self._get_attr_by_idx(idx, "proprioception", False)
        return _reorder_pose10(proprio10, self.gripper_scale).unsqueeze(1)

    def _get_action(self, idx):
        # zarr: (chunk, K, NHAND=1, 10) -> (chunk, K, 1, 10)
        rel10 = self._get_attr_by_idx(idx, "rel_action", False)
        return _reorder_pose10(rel10, self.gripper_scale)

    def __getitem__(self, idx):
        total = len(self.annos["rel_action"]) // self.chunk_size
        effective = min(self.max_samples, total) if self.max_samples is not None else total
        idx = idx % effective
        idx = idx * self.chunk_size

        action = self._get_action(idx)
        if self._actions_only:
            return {"action": action}
        return {
            "task": self._get_task(idx),
            "instr": self._get_instr(idx),
            "rgb_front": self._get_attr_by_idx(idx, "rgb_front"),
            "rgb_back": self._get_attr_by_idx(idx, "rgb_back"),
            "rgb_wrist": self._get_attr_by_idx(idx, "rgb_wrist"),
            "depth_front": self._get_attr_by_idx(idx, "depth_front"),
            "depth_back": self._get_attr_by_idx(idx, "depth_back"),
            "proprioception": self._get_proprioception(idx),
            "action": action,
        }


class AlohaJointDataset(AlohaDataset):
    """ALOHA single-arm dataset for the joint-action model
    (``flower_actor_3d_motion_latent_joint``).

    Keeps the full EEF pipeline (rgb front/back/wrist, depth, proprio, and EEF
    ``action`` = rel_action for the model's EEF chain-of-thought) and adds the
    joint targets/state appended to the zarr by the joint-action converter:

      - ``rel_joint_action``: ``(N, H=15, nhand=1, 7)`` -- dims ``0..5`` are the
        anchor-relative arm-joint deltas (6 joints), dim ``6`` is the binary
        gripper command (1=open, 0=closed).
      - ``proprioception_joint``: ``(N, nhist=1, nhand=1, 7)`` -- 6 arm joints +
        binary gripper command.

    UNITS. The ALOHA zarr stores arm joints in DEGREES (unlike CALVIN/RoboTwin2,
    which store radians). We convert the 6 arm joints (NOT the binary gripper) to
    RADIANS here so the unnormalized joint-proprio embedding in the model stays
    well conditioned and the per-joint normalizer matches the sibling joint
    datasets. The per-joint normalizer in ``AlohaJointTrainer.JOINT_NORM_HI`` is
    computed in radians (see scripts/aloha/compute_joint_rel_action_stats.py).

    The model predicts radian arm-joint deltas relative to the radian proprio
    anchor; deployment must convert the predicted deltas back to degrees (x180/pi)
    before adding them to the current degree-valued joint state.

    Dataset name: ``AlohaJoint``.
    """

    # Convert arm joints (dims 0..5) from degrees to radians at load time.
    DEG_TO_RAD = True
    NUM_ARM_JOINTS = 6

    required_keys = AlohaDataset.required_keys + (
        "rel_joint_action",
        "proprioception_joint",
    )

    @classmethod
    def _arm_to_radians(cls, joints: torch.Tensor) -> torch.Tensor:
        """Convert dims 0..5 (arm joints) from degrees to radians; leave the
        trailing binary gripper command untouched. Operates on (..., 7)."""
        if not cls.DEG_TO_RAD:
            return joints
        out = joints.clone()
        out[..., : cls.NUM_ARM_JOINTS] = torch.deg2rad(
            out[..., : cls.NUM_ARM_JOINTS]
        )
        return out

    def _get_joint_action(self, idx):
        # zarr: (chunk, H, nhand=1, 7) -> arm deltas (rad) + binary gripper.
        rel = self._get_attr_by_idx(idx, "rel_joint_action", False)
        return self._arm_to_radians(rel)

    def _get_joint_proprio(self, idx):
        # zarr: (chunk, nhand=1, 7) -> add nhist axis -> (chunk, nhist=1, nhand=1,
        # 7); arm joints (rad) + binary gripper (mirrors _get_proprioception).
        prop = self._get_attr_by_idx(idx, "proprioception_joint", False)
        return self._arm_to_radians(prop).unsqueeze(1)

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        # super() already converted idx to the raw zarr index and returned the
        # EEF sample; replicate the same index arithmetic for the joint keys.
        total = len(self.annos["rel_action"]) // self.chunk_size
        effective = (
            min(self.max_samples, total) if self.max_samples is not None else total
        )
        raw_idx = (idx % effective) * self.chunk_size
        sample["joint_action"] = self._get_joint_action(raw_idx)
        if not self._actions_only:
            sample["joint_proprio"] = self._get_joint_proprio(raw_idx)
        return sample


class AlohaHeadRGBDDataset(AlohaDataset):
    """ALOHA variant that trains only on the two head RGBD cameras.

    Uses front/back RGBD as 3D observations and ignores any wrist RGB array if it
    is present in the zarr store.
    """

    cameras = ("front", "back")
    required_keys = (
        "rgb_front", "rgb_back", "depth_front", "depth_back",
        "proprioception", "rel_action", "instr_id",
    )

    def __getitem__(self, idx):
        total = len(self.annos["rel_action"]) // self.chunk_size
        effective = min(self.max_samples, total) if self.max_samples is not None else total
        idx = idx % effective
        idx = idx * self.chunk_size

        action = self._get_action(idx)
        if self._actions_only:
            return {"action": action}
        return {
            "task": self._get_task(idx),
            "instr": self._get_instr(idx),
            "rgb_front": self._get_attr_by_idx(idx, "rgb_front"),
            "rgb_back": self._get_attr_by_idx(idx, "rgb_back"),
            "depth_front": self._get_attr_by_idx(idx, "depth_front"),
            "depth_back": self._get_attr_by_idx(idx, "depth_back"),
            "proprioception": self._get_proprioception(idx),
            "action": action,
        }
