#!/usr/bin/env python3
"""Side-by-side figure: left = RGB image, right = trajectory image.

RGB files may be named with a ``_rgb`` suffix (e.g. ``foo_rgb.png``) paired with
``foo.png`` in the trajectory folder.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
RGB_STEM_SUFFIX = "_rgb"


def _rgb_pair_key(stem: str) -> str:
    """Map RGB basename to traj basename: strip trailing ``_rgb`` if present."""
    return stem.removesuffix(RGB_STEM_SUFFIX)


def _load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _resize_to_height(img: Image.Image, height: int) -> Image.Image:
    w, h = img.size
    if h == height:
        return img
    new_w = max(1, int(round(w * height / h)))
    return img.resize((new_w, height), Image.Resampling.LANCZOS)


def combine_horizontal(rgb: Image.Image, traj: Image.Image) -> np.ndarray:
    """Stack RGB (left) and traj (right); match heights (traj scaled to RGB height)."""
    h = rgb.size[1]
    traj_r = _resize_to_height(traj.convert("RGB"), h)
    rgb_arr = np.asarray(rgb)
    traj_arr = np.asarray(traj_r)
    return np.concatenate([rgb_arr, traj_arr], axis=1)


def iter_stem_pairs(rgb_dir: Path, traj_dir: Path) -> list[tuple[str, Path, Path]]:
    rgb_map: dict[str, Path] = {}
    for p in rgb_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            key = _rgb_pair_key(p.stem)
            rgb_map[key] = p

    pairs: list[tuple[str, Path, Path]] = []
    for p in traj_dir.iterdir():
        if not p.is_file() or p.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if p.stem in rgb_map:
            pairs.append((p.stem, rgb_map[p.stem], p))

    pairs.sort(key=lambda x: x[0])
    return pairs


def save_combined_figure(
    rgb_path: Path,
    traj_path: Path,
    out_path: Path | None,
    *,
    dpi: int = 150,
) -> None:
    rgb = _load_rgb(rgb_path)
    traj = Image.open(traj_path)
    combined = combine_horizontal(rgb, traj)

    fig, ax = plt.subplots(figsize=(combined.shape[1] / dpi, combined.shape[0] / dpi), dpi=dpi)
    ax.imshow(combined)
    ax.axis("off")
    fig.tight_layout(pad=0)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rgb_dir", type=Path, help="Directory of RGB images")
    parser.add_argument(
        "traj_dir",
        type=Path,
        help="Directory of trajectory images (basename matches RGB after stripping a trailing _rgb)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="If set, write combined PNGs here (one per pair). Otherwise display with plt.show().",
    )
    parser.add_argument(
        "--stem",
        type=str,
        default=None,
        help="Only this traj basename (without extension); ``foo`` or ``foo_rgb`` both work",
    )
    args = parser.parse_args()

    if not args.rgb_dir.is_dir():
        raise SystemExit(f"Not a directory: {args.rgb_dir}")
    if not args.traj_dir.is_dir():
        raise SystemExit(f"Not a directory: {args.traj_dir}")

    pairs = iter_stem_pairs(args.rgb_dir, args.traj_dir)
    if args.stem:
        want = _rgb_pair_key(args.stem)
        pairs = [t for t in pairs if t[0] == want]
        if not pairs:
            raise SystemExit(f"No pair found for stem {args.stem!r} (traj key {want!r})")

    if not pairs:
        raise SystemExit("No matching pairs (traj stem = RGB stem with optional trailing _rgb stripped).")

    for stem, rgb_p, traj_p in pairs:
        out = (args.output_dir / f"{stem}_rgb_traj.png") if args.output_dir else None
        save_combined_figure(rgb_p, traj_p, out)


if __name__ == "__main__":
    main()
