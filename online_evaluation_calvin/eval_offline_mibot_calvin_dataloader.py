"""
Run MiBoT on batches from datasets.calvin.CALVINDataset (zarr), using the same
tensor layout as training data and calvin_dataloader_mibot conversion to MiBoT.

Run from repo root with PYTHONPATH including this package, e.g.:

  cd analogical_manipulation_clean
  python -m online_evaluation_calvin.eval_offline_mibot_calvin_dataloader \\
    --zarr_root /path/to/val_zarr --instructions /path/to/instructions.json \\
    --model_path /path/to/xiaomi_calvin --num_samples 8
"""
import argparse

import numpy as np
import torch

from datasets.calvin import CALVINDataset
from online_evaluation_calvin.calvin_dataloader_mibot import (
    calvin_instruction_at,
    calvin_sample_to_mibot_obs,
)
from online_evaluation_calvin.xiaomi_model_wrapper import XiaomiModel


def parse_arguments():
    p = argparse.ArgumentParser(description="MiBoT offline eval via CALVIN zarr dataloader")
    p.add_argument("--zarr_root", type=str, required=True,help="Directory of CALVIN zarr (validation or training)")
    p.add_argument(
        "--instructions",
        type=str,
        required=True,
        help="JSON list of language strings (same as CALVIN dataset lang ann store)",
    )
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--split", type=str, default="abc")
    p.add_argument("--crop_ratio", type=float, default=0.95)
    p.add_argument("--pred_len", type=int, default=10)
    p.add_argument("--chunk_size", type=int, default=8, help="Must match zarr chunking / training")
    p.add_argument("--num_samples", type=int, default=16)
    p.add_argument(
        "--frame_idx",
        type=int,
        default=0,
        help="Timestep index within each dataloader chunk (0 .. chunk_size-1)",
    )
    p.add_argument("--mem_limit", type=int, default=8, help="Zarr LRU cache size (GB)")
    p.add_argument("--relative_action", action="store_true", help="Load rel_action from zarr if set")
    p.add_argument("--max_samples", type=int, default=None, help="Cap dataset length (debug)")
    p.add_argument(
        "--compare_gt",
        action="store_true",
        help="Report L2 between predicted and zarr action for the chosen frame",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_arguments()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ds = CALVINDataset(
        root=args.zarr_root,
        instructions=args.instructions,
        relative_action=args.relative_action,
        mem_limit=args.mem_limit,
        chunk_size=args.chunk_size,
    )
    if args.max_samples is not None:
        ds.max_samples = args.max_samples

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = XiaomiModel(
        model_path=args.model_path,
        split=args.split,
        crop_ratio=args.crop_ratio,
        action_chunk=args.pred_len,
        device=str(device),
    )

    n = min(args.num_samples, len(ds))
    errs = []

    for i in range(n):
        sample = ds[i]
        obs = calvin_sample_to_mibot_obs(
            sample, frame_idx=args.frame_idx, crop_ratio=args.crop_ratio
        )
        instr = calvin_instruction_at(sample, frame_idx=args.frame_idx)
        pred = model.step(obs, instr)[0]  # (pred_len, 7)

        if args.compare_gt:
            gt = sample["action"][args.frame_idx, :, 0, :].numpy()
            T = min(pred.shape[0], gt.shape[0])
            d = np.linalg.norm(pred[:T] - gt[:T])
            errs.append(float(d))

        if i < 3:
            print(f"[{i}] instr={instr[:80]!r}...")
            print(f"    pred[0]={np.array2string(pred[0], precision=4)}")

    if args.compare_gt and errs:
        print(f"L2 norm pred vs GT (horizon min(pred,gt)), mean={np.mean(errs):.4f} over {len(errs)} batches")


if __name__ == "__main__":
    main()
