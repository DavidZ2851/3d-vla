import torch
from torch import nn
from .base import BaseTrainTester


class LiberoTrainTester(BaseTrainTester):
    """Train/test for LIBERO dataset.
    Normalizer strategy:
      axisangle (relative): ``control_action`` from zarr is raw OSC_POSE in [-1, 1]
      for xyz deltas and axis-angle deltas (6D) + gripper. ``normalize_pos`` already
      maps [min, max] -> [-1, 1] for diffusion, so bounds must be [-1, 1] for all
      six dims (identity on those channels).
      rotmatrix (absolute): dataset-derived min/max for xyz only (3D); rotation
      uses rot6d inside the model, not min-max normalized.
    """

    @torch.no_grad()
    def get_workspace_normalizer(self):
        if self.args.rotation_format == 'axisangle':
            # Raw relative actions are already in [-1, 1] per dimension (OSC_POSE).
            return nn.Parameter(torch.tensor([
                [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                [ 1.0,  1.0,  1.0,  1.0,  1.0,  1.0],
            ]), requires_grad=False)
        # rotmatrix (absolute): compute xyz bounds from training data
        return super().get_workspace_normalizer(ndims=3)

    @torch.no_grad()
    def prepare_batch(self, sample, augment=False):
        sample["action"] = self.preprocessor.process_actions(sample["action"])
        proprio = self.preprocessor.process_proprio(sample["proprioception"])
        rgbs, pcds = self.preprocessor.process_obs(
            sample["rgb"], sample["rgb2d"],
            sample["depth"], sample["extrinsics"], sample["intrinsics"],
            augment=augment
        )
        return (
            sample["action"],
            torch.zeros(sample["action"].shape[:-1], dtype=bool, device='cuda'),
            rgbs,
            None,
            pcds,
            sample["instr"],
            proprio
        )
