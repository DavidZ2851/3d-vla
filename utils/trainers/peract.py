import torch

from .base import BaseTrainTester


class PeractTrainTester(BaseTrainTester):

    @torch.no_grad()
    def prepare_batch(self, sample, augment=False):
        sample["action"] = self.preprocessor.process_actions(sample["action"])
        proprio = self.preprocessor.process_proprio(sample["proprioception"])
        rgbs, pcds = self.preprocessor.process_obs(
            sample["rgb"], sample["depth"], sample["extrinsics"], sample["intrinsics"],
            augment=augment
        )
        mask = torch.zeros(sample["action"].shape[:-1], dtype=bool, device='cuda')  # (B, T, 2)
        mask[...,1] = True  # mask out the second "fake" hand
        return (
            sample["action"],
            mask,
            rgbs,
            None,
            pcds,
            sample["instr"],
            proprio
        )
