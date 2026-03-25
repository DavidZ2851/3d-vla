import torch

from .base import BaseTrainTester


class PeractTrainTester(BaseTrainTester):
    """Peract (single-arm) with second hand padded as fake; mask it in model."""

    @torch.no_grad()
    def get_workspace_normalizer_for_fake_static_hand(self, ndims=3):
        """Use only the first (real) hand's positions; padded second hand is zeros and would corrupt bounds."""
        print("Computing workspace normalizer (Peract: first hand only)...")
        train_dataset = self.dataset_cls(
            root=self.args.train_data_dir,
            instructions=self.args.train_instructions,
            copies=1,
            relative_action=self.args.relative_action,
            mem_limit=0.1,
            actions_only=True,
            chunk_size=self.args.chunk_size
        )
        from torch.utils.data import DataLoader
        from .base import actions_collate_fn
        from tqdm import tqdm
        data_loader = DataLoader(
            train_dataset,
            batch_size=max(self.args.batch_size, 64) // self.args.chunk_size,
            collate_fn=actions_collate_fn,
            shuffle=False,
            num_workers=self.args.num_workers
        )
        min_, max_ = torch.ones(ndims) * 10000, -torch.ones(ndims) * 10000
        for sample in tqdm(data_loader):
            action = sample["action"]  # (B, T, 2, 8)
            # Only first hand (index 0); second is padded fake
            action = action[..., 0, :ndims].reshape(-1, ndims)
            min_ = torch.min(min_, action.min(0).values)
            max_ = torch.max(max_, action.max(0).values)
        min_ = min_ - self.args.workspace_normalizer_buffer
        max_ = max_ + self.args.workspace_normalizer_buffer
        from torch import nn
        return nn.Parameter(torch.stack([min_, max_]), requires_grad=False)

    @torch.no_grad()
    def prepare_batch(self, sample, augment=False):
        sample["action"] = self.preprocessor.process_actions(sample["action"])
        proprio = self.preprocessor.process_proprio(sample["proprioception"])
        rgbs, pcds = self.preprocessor.process_obs(
            sample["rgb"], sample["depth"], sample["extrinsics"], sample["intrinsics"],
            augment=augment
        )
        mask = torch.zeros(sample["action"].shape[:-1], dtype=bool, device='cuda')  # (B, T, 2)
        # mask[...,1] = True  # NOTE: mask out the second "fake" hand if padding
        return (
            sample["action"],
            mask,
            rgbs,
            None,
            pcds,
            sample["instr"],
            proprio
        )
