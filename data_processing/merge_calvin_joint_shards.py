import argparse
import os
import glob

import numpy as np
import zarr
from numcodecs import Blosc
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--tgt', type=str,
        default='/data/group_data/katefgroup/VLA/zarr_datasets/calvin_zarr_joint',
        help='Directory containing the per-shard zarrs to read.')
    parser.add_argument(
        '--out', type=str, default=None,
        help='Directory to write merged {split}.zarr to. Defaults to --tgt.')
    parser.add_argument('--splits', nargs='+', default=['train', 'val'])
    parser.add_argument('--num_shards', type=int, default=10)
    # Merge directly into chunk_size-8 chunks along dim 0 (rechunk fused into the
    # merge pass). chunk_size=1 reproduces the per-row chunking of the shards.
    parser.add_argument('--chunk_size', type=int, default=8)
    # Rows read+appended per step. Reading whole shards at once (~7GB for rgb)
    # OOMs and stats all ~58k per-row chunk files before returning anything, which
    # looks frozen. Batching bounds memory and gives incremental progress.
    parser.add_argument('--batch_rows', type=int, default=2048)
    parser.add_argument('--delete_shards', action='store_true')
    return parser.parse_args()


def merge_split(tgt, out_dir, split, num_shards, delete_shards, chunk_size,
                batch_rows):
    out = f"{out_dir}/{split}.zarr"
    if os.path.exists(out):
        print(f"{out} already exists, skipping.")
        return
    shards = sorted(glob.glob(f"{tgt}/{split}_shard_*_of_{num_shards:02d}.zarr"))
    if not shards:
        print(f"No shards for split={split}, skipping.")
        return
    os.makedirs(out_dir, exist_ok=True)
    print(f"[{split}] merging {len(shards)} shards -> {out} (chunk_size={chunk_size})")

    compressor = Blosc(cname='lz4', clevel=1, shuffle=Blosc.SHUFFLE)
    dst = zarr.open_group(out, mode='w')
    created = False
    for shard in shards:
        src = zarr.open_group(shard, mode='r')
        keys = list(src.array_keys())
        if not keys or src[keys[0]].shape[0] == 0:
            continue
        if not created:
            for k in keys:
                a = src[k]
                dst.create_dataset(
                    k, shape=(0,) + a.shape[1:],
                    chunks=(chunk_size,) + a.shape[1:],
                    compressor=compressor, dtype=a.dtype)
            created = True
        # Some shards have a row-count mismatch across keys (e.g. rgb has a few
        # more frames than depth); merge only the rows present in every key.
        nrows = min(src[k].shape[0] for k in keys)
        for start in tqdm(range(0, nrows, batch_rows),
                          desc=os.path.basename(shard), leave=False):
            stop = min(start + batch_rows, nrows)
            for k in keys:
                dst[k].append(src[k][start:stop])

    n = dst[next(iter(dst.array_keys()))].shape[0] if created else 0
    assert all(dst[k].shape[0] == n for k in dst.array_keys())
    print(f"[{split}] merged {n} rows -> {out}")

    if delete_shards:
        import shutil
        for shard in shards:
            shutil.rmtree(shard)
        print(f"[{split}] removed {len(shards)} shard dirs")


if __name__ == '__main__':
    args = parse_arguments()
    out_dir = args.out if args.out is not None else args.tgt
    for split in args.splits:
        merge_split(args.tgt, out_dir, split, args.num_shards, args.delete_shards,
                    args.chunk_size, args.batch_rows)
