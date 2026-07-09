from pathlib import Path
import os

MPLCONFIGDIR = Path("/tmp/matplotlib-codex")
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from matplotlib.lines import Line2D
import numpy as np


OUT_DIR = Path(__file__).resolve().parent


COLORS = {
    "ink": "#121417",
    "muted": "#4c5563",
    "panel": "#f5f6fb",
    "blue": "#cfe0f6",
    "blue_edge": "#6e8bad",
    "cream": "#fbf7da",
    "cream_edge": "#b6aa73",
    "green": "#e6f2e1",
    "green_edge": "#77a16c",
    "pink": "#f8d8d7",
    "orange": "#f7d79c",
    "teal": "#caece8",
    "purple": "#d8cef3",
    "gray": "#eef0f5",
    "gray_edge": "#9aa2ad",
}


def add_box(ax, x, y, w, h, text, fc, ec=None, lw=1.2, fs=10, weight="normal",
            radius=0.08, ha="center", va="center", color=None, z=3, ls="-"):
    ec = ec or COLORS["gray_edge"]
    color = color or COLORS["ink"]
    shadow = FancyBboxPatch(
        (x + 0.035, y - 0.035), w, h,
        boxstyle=f"round,pad=0.018,rounding_size={radius}",
        linewidth=0, facecolor="#000000", alpha=0.08, zorder=z - 2,
    )
    ax.add_patch(shadow)
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.018,rounding_size={radius}",
        linewidth=lw, edgecolor=ec, facecolor=fc, linestyle=ls, zorder=z,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha=ha, va=va, fontsize=fs, weight=weight, color=color,
        family="DejaVu Sans", linespacing=1.12, zorder=z + 1,
    )
    return patch


def add_plain_label(ax, x, y, text, fs=9, weight="normal", ha="center",
                    color=None, z=10):
    ax.text(
        x, y, text,
        ha=ha, va="center", fontsize=fs, weight=weight,
        color=color or COLORS["ink"], family="DejaVu Sans",
        linespacing=1.12, zorder=z,
    )


def add_arrow(ax, start, end, color=None, lw=1.5, ls="-", rad=0.0,
              mutation_scale=13, z=8):
    color = color or COLORS["ink"]
    arr = FancyArrowPatch(
        start, end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=lw,
        linestyle=ls,
        color=color,
        shrinkA=3,
        shrinkB=3,
        connectionstyle=f"arc3,rad={rad}",
        zorder=z,
    )
    ax.add_patch(arr)
    return arr


def draw_token_stack(ax, x, y, w=0.33, h=1.2, n=5, colors=None, label=None,
                     ec="#5a6472", fs=8, z=5):
    colors = colors or ["#8db7e0", "#7db7b2", "#e6bd68", "#d77bc1", "#8a6fc8"]
    gap = h * 0.05
    cell_h = (h - gap * (n - 1)) / n
    for i in range(n):
        yy = y + i * (cell_h + gap)
        add_box(
            ax, x, yy, w, cell_h, "", colors[i % len(colors)],
            ec=ec, lw=0.9, radius=0.035, z=z,
        )
    if label:
        add_plain_label(ax, x + w / 2, y - 0.16, label, fs=fs, color=COLORS["muted"])


def draw_observation(ax, x, y, w, h, title="RGB-D obs."):
    add_box(ax, x, y, w, h, "", "#ffffff", ec="#7b8794", radius=0.05, z=3)
    table = Rectangle((x + 0.08, y + 0.09), w - 0.16, h - 0.18,
                      facecolor="#e7c79c", edgecolor="none", zorder=4)
    ax.add_patch(table)
    ax.add_patch(Rectangle((x + 0.12, y + 0.18), 0.38, 0.28,
                           facecolor="#dcebf8", edgecolor="#53687f", lw=1.0, zorder=5))
    ax.add_patch(Rectangle((x + 0.67, y + 0.28), 0.32, 0.22,
                           facecolor="#39485b", edgecolor="#1d2530", lw=1.0, zorder=5))
    ax.add_patch(plt.Circle((x + 0.47, y + 0.24), 0.12,
                            facecolor="#f7f8fb", edgecolor="#8794a3", lw=1.0, zorder=6))
    add_plain_label(ax, x + w / 2, y - 0.18, title, fs=8.5, weight="bold")


def draw_query_grid(ax, x, y, w, h):
    add_box(ax, x, y, w, h, "", "#ffffff", ec="#7b8794", radius=0.04, z=3)
    gx0, gy0 = x + 0.12, y + 0.10
    gw, gh = w - 0.24, h - 0.20
    for i in range(7):
        xx = gx0 + i * gw / 6
        ax.add_line(Line2D([xx, xx], [gy0, gy0 + gh], color="#ccd5df", lw=0.6, zorder=4))
        yy = gy0 + i * gh / 6
        ax.add_line(Line2D([gx0, gx0 + gw], [yy, yy], color="#ccd5df", lw=0.6, zorder=4))
    rng = np.random.default_rng(12)
    for _ in range(30):
        px = gx0 + rng.random() * gw
        py = gy0 + rng.random() * gh
        ax.plot(px, py, "o", ms=2.4, color="#5aa0c8", alpha=0.75, zorder=5)
    add_plain_label(ax, x + w / 2, y - 0.18, "20x20 query grid", fs=8.5, weight="bold")


def draw_noise_patch(ax, x, y, w, h, label):
    rng = np.random.default_rng(0)
    noise = rng.random((20, 28, 3))
    ax.imshow(noise, extent=(x, x + w, y, y + h), zorder=3, interpolation="nearest")
    ax.add_patch(Rectangle((x, y), w, h, facecolor="none", edgecolor="#7b8794", lw=1.0, zorder=4))
    add_plain_label(ax, x + w / 2, y - 0.16, label, fs=8.5, weight="bold")


def draw_point_cloud(ax, cx, cy, sx=1.3, sy=0.8, seed=1, z=4):
    rng = np.random.default_rng(seed)
    pts = []
    cols = []
    for _ in range(170):
        u = rng.normal(0.0, 0.42)
        v = rng.normal(0.0, 0.24)
        if abs(u) + 1.6 * abs(v) < 0.95:
            pts.append((cx + sx * u, cy + sy * v))
            cols.append(rng.uniform(0.15, 0.95))
    pts = np.array(pts)
    ax.scatter(
        pts[:, 0], pts[:, 1],
        s=13, c=cols, cmap="viridis", alpha=0.68,
        edgecolors="none", zorder=z,
    )
    # A few sparse predicted motion tracks.
    colors = ["#d7191c", "#2c7bb6", "#fdae61", "#5aae61"]
    for i, c in enumerate(colors):
        for j in range(3):
            px = cx + sx * rng.uniform(-0.45, 0.45)
            py = cy + sy * rng.uniform(-0.30, 0.30)
            dx = sx * rng.uniform(0.10, 0.22) * (-1 if j % 2 else 1)
            dy = sy * rng.uniform(0.04, 0.14)
            ax.add_line(Line2D([px, px + dx], [py, py + dy], color=c, lw=2.0, alpha=0.9, zorder=z + 2))
            ax.add_line(Line2D([px + dx, px + dx * 0.72], [py + dy, py + dy + 0.10],
                               color=c, lw=2.0, alpha=0.9, zorder=z + 2))
    return pts


def draw_mlp_stack(ax, x, y, labels, color="#ffffff"):
    for i, label in enumerate(labels):
        add_box(ax, x, y + i * 0.42, 0.72, 0.27, label, color, ec="#111111",
                lw=1.0, fs=7.2, radius=0.02, z=6)


def draw_wedge(ax, x, y, w, h, text, fc, ec, fs=10):
    shadow = Polygon(
        [[x + 0.06, y - 0.04], [x + w + 0.06, y + h * 0.18 - 0.04],
         [x + w + 0.06, y + h * 0.82 - 0.04], [x + 0.06, y + h - 0.04]],
        closed=True, facecolor="#000000", edgecolor="none", alpha=0.10, zorder=2,
    )
    ax.add_patch(shadow)
    poly = Polygon(
        [[x, y], [x + w, y + h * 0.18], [x + w, y + h * 0.82], [x, y + h]],
        closed=True, facecolor=fc, edgecolor=ec, lw=1.2, zorder=5,
    )
    ax.add_patch(poly)
    add_plain_label(ax, x + w * 0.54, y + h / 2, text, fs=fs, weight="bold", z=6)


def draw_left_panel(ax):
    add_box(ax, 0.18, 0.50, 5.55, 6.12, "", COLORS["panel"], ec="#e3e6ee",
            lw=0.8, radius=0.12, z=0)

    draw_observation(ax, 0.42, 5.33, 1.18, 0.72)
    add_box(ax, 0.40, 4.68, 1.22, 0.36, "Instruction", "#ffffff", ec="#8c98a7",
            fs=8.5, weight="bold", radius=0.04)
    draw_query_grid(ax, 0.42, 3.76, 1.18, 0.66)

    add_arrow(ax, (1.64, 5.70), (2.08, 5.70))
    add_arrow(ax, (1.64, 4.86), (2.08, 5.38), rad=0.12)
    add_box(ax, 2.10, 5.29, 1.54, 0.82,
            "VLM encoder\nFlorence-2\nQwen3-VL",
            COLORS["blue"], ec=COLORS["blue_edge"], fs=8.9, weight="bold")
    add_plain_label(ax, 2.88, 5.13, "visual tokens + text tokens", fs=7.6, color=COLORS["muted"])

    add_arrow(ax, (3.64, 5.64), (4.08, 5.64))
    draw_token_stack(
        ax, 4.12, 5.12, w=0.32, h=1.05, n=4,
        colors=["#a990d7", "#a6c0ec", "#caa3db", "#879fd0"],
        label="V/T tokens", fs=7.6,
    )
    add_arrow(ax, (4.46, 5.60), (4.98, 4.82), rad=-0.08)

    add_arrow(ax, (1.64, 4.08), (2.08, 4.28))
    add_box(ax, 2.10, 4.02, 1.55, 0.72,
            "Point tokens\n[xyz, uv, visual]",
            "#eaf3fb", ec="#7ea2c7", fs=9.2, weight="bold")
    add_arrow(ax, (2.88, 5.27), (2.88, 4.77))
    add_arrow(ax, (3.66, 4.38), (4.08, 4.38))
    add_box(ax, 4.10, 4.03, 1.36, 0.78,
            "Sparse point\nencoder\nkNN=16, x6",
            COLORS["teal"], ec="#62a9a3", fs=8.7, weight="bold")

    add_arrow(ax, (4.78, 4.02), (4.78, 3.42))
    add_box(ax, 3.78, 2.67, 1.96, 0.70,
            "Flow matching\nX_t=(1-t)X0+tX1\npredict velocity",
            "#ffffff", ec="#a8aeb9", fs=8.0)
    draw_noise_patch(ax, 0.52, 2.58, 0.88, 0.52, "noise X0")
    add_arrow(ax, (1.44, 2.84), (3.76, 3.02))

    add_arrow(ax, (4.75, 2.66), (4.12, 2.18), rad=0.08)
    add_box(ax, 2.00, 1.38, 3.56, 0.86, "",
            COLORS["cream"], ec=COLORS["cream_edge"], fs=10.0, weight="bold")
    add_plain_label(ax, 3.78, 2.04, "Sparse factorized trajectory decoder x8",
                    fs=8.8, weight="bold")
    parts = ["spatial", "temporal", "V/T cross", "FFN"]
    for i, p in enumerate(parts):
        add_box(ax, 2.20 + i * 0.80, 1.54, 0.66, 0.24, p,
                "#fffdf0", ec="#d0c68d", lw=0.8, fs=6.9, radius=0.025, z=7)
        if i < len(parts) - 1:
            add_arrow(ax, (2.86 + i * 0.80, 1.66), (2.98 + i * 0.80, 1.66),
                      lw=0.7, mutation_scale=6)

    add_arrow(ax, (4.60, 1.37), (4.55, 1.08))
    add_box(ax, 4.06, 0.62, 1.45, 0.48,
            "Future motion\nDelta xyz\n[400 x 15 x 3]",
            "#e9f5e5", ec=COLORS["green_edge"], fs=7.8, weight="bold")
    add_arrow(ax, (3.15, 1.37), (3.05, 1.08))
    add_box(ax, 2.28, 0.62, 1.45, 0.48,
            "Decoder hidden\nz_m\n[400 x 15 x 320]",
            "#eee8fb", ec="#9885c5", fs=7.8, weight="bold")

    add_plain_label(ax, 2.95, 0.20, "(a) motion expert", fs=13, weight="bold")


def draw_right_panel(ax):
    add_box(ax, 6.28, 0.50, 11.48, 6.12, "", "#ffffff", ec="#edf0f4",
            lw=0.8, radius=0.12, z=0)

    # Inputs and shared Florence2 encoder.
    draw_observation(ax, 6.55, 5.42, 1.10, 0.66, title="RGB + point cloud")
    add_box(ax, 6.54, 4.72, 1.12, 0.38, "Instruction", "#ffffff",
            ec="#8c98a7", fs=8.2, weight="bold", radius=0.04)
    add_box(ax, 6.54, 4.08, 1.12, 0.38, "Proprio", "#ffffff",
            ec="#8c98a7", fs=8.2, weight="bold", radius=0.04)

    add_arrow(ax, (7.70, 5.73), (8.12, 5.73))
    add_arrow(ax, (7.70, 4.90), (8.12, 5.46), rad=0.12)
    add_arrow(ax, (7.70, 4.27), (8.12, 5.28), rad=0.16)
    add_box(ax, 8.14, 5.16, 1.92, 0.96,
            "Florence-2\n3D VLA encoder",
            COLORS["blue"], ec=COLORS["blue_edge"], fs=10.0, weight="bold")
    add_plain_label(ax, 9.10, 4.95, "caches visual/text features for MotionExpert",
                    fs=7.4, color=COLORS["muted"])

    # Scene tokens and spatial context.
    draw_point_cloud(ax, 9.82, 3.95, sx=1.45, sy=0.92, seed=4, z=4)
    ax.text(
        10.15, 4.05, "3D scene tokens",
        ha="center", va="center", fontsize=9.0, weight="bold",
        color="#2d7d78", family="DejaVu Sans", zorder=9,
        path_effects=[pe.withStroke(linewidth=3.0, foreground="white")],
    )
    add_arrow(ax, (9.08, 5.14), (9.45, 4.80))
    add_arrow(ax, (9.98, 5.15), (11.10, 4.93), rad=-0.05)
    draw_token_stack(
        ax, 11.18, 4.30, w=0.34, h=1.06, n=5,
        colors=["#78b5d9", "#72bcaa", "#e6bd68", "#b783c7", "#77a16c"],
        label="scene", fs=7.7,
    )

    # Motion latent branch.
    add_box(ax, 6.62, 3.11, 1.20, 0.62,
            "front PCD\n20x20 xyz",
            "#eaf3fb", ec="#7ea2c7", fs=8.1, weight="bold")
    add_arrow(ax, (7.82, 3.42), (8.35, 3.12))
    add_arrow(ax, (8.55, 5.15), (8.88, 3.73), rad=0.16, ls="--", color="#6a7482")
    add_box(ax, 8.36, 2.48, 1.55, 0.80,
            "MotionExpert\none ODE step\nat tau=0",
            COLORS["cream"], ec=COLORS["cream_edge"], fs=8.8, weight="bold")
    add_arrow(ax, (9.92, 2.88), (10.32, 2.88))
    draw_token_stack(
        ax, 10.36, 2.35, w=0.30, h=1.04, n=5,
        colors=["#d2c070", "#e2cf8b", "#c4b66d", "#e6d798", "#bba854"],
        label="z_m", fs=7.7,
    )
    add_arrow(ax, (10.68, 2.88), (11.04, 2.88))
    add_box(ax, 11.06, 2.55, 1.08, 0.58,
            "Temporal\npool + proj.",
            "#ffffff", ec="#a8aeb9", fs=8.1)
    add_arrow(ax, (12.14, 2.88), (12.58, 2.88))
    add_box(ax, 12.60, 2.46, 1.18, 0.78,
            "motion tokens\n[400 x C]\n+ 3D pos",
            COLORS["purple"], ec="#9885c5", fs=8.1, weight="bold")

    # Scene fusion.
    add_arrow(ax, (13.16, 3.25), (13.16, 3.88), color="#7a5cad")
    add_arrow(ax, (11.54, 4.84), (12.18, 4.48), rad=-0.08)
    add_box(ax, 12.18, 4.08, 1.72, 0.78,
            "gated scene-motion\nRoPE-3D cross-attn\nalpha_scene",
            COLORS["green"], ec=COLORS["green_edge"], fs=8.0, weight="bold")
    add_arrow(ax, (13.90, 4.47), (14.44, 4.47))
    draw_token_stack(
        ax, 14.50, 3.94, w=0.34, h=1.06, n=5,
        colors=["#77a16c", "#78b5d9", "#72bcaa", "#e6bd68", "#b783c7"],
        label="conditioned\nscene", fs=7.2,
    )

    # Language/proprio token branch.
    draw_mlp_stack(ax, 10.25, 5.28, ["MLP", "MLP", "MLP"], color="#ffffff")
    add_plain_label(ax, 11.70, 5.88, "language tokens", fs=9.0, weight="bold", color="#4f9643")
    add_plain_label(ax, 11.68, 5.44, "proprio tokens", fs=9.0, weight="bold", color="#6b4db3")
    draw_token_stack(ax, 13.34, 5.24, w=0.28, h=0.62, n=3,
                     colors=["#aad18f", "#b7d99f", "#986fcf"], label=None)
    add_arrow(ax, (10.02, 5.65), (10.24, 5.65), lw=1.0)
    add_arrow(ax, (10.02, 5.42), (10.24, 5.42), lw=1.0)

    # Action denoising policy.
    draw_noise_patch(ax, 6.62, 1.38, 0.88, 0.48, "noisy action")
    add_box(ax, 7.90, 1.40, 1.02, 0.48,
            "step i\n+ proprio",
            "#ffffff", ec="#a8aeb9", fs=8.2, weight="bold")
    add_arrow(ax, (7.52, 1.62), (7.88, 1.62))
    add_arrow(ax, (8.92, 1.62), (9.30, 1.62))
    draw_token_stack(
        ax, 9.36, 1.10, w=0.34, h=1.06, n=5,
        colors=["#7db7b2", "#d77bc1", "#e6bd68", "#8db7e0", "#cda37c"],
        label="action\ntokens", fs=7.2,
    )
    add_arrow(ax, (9.72, 1.62), (10.74, 1.62))
    add_wedge_text = "MotionLatent\nTransformer Head\nx16"
    draw_wedge(ax, 10.78, 0.98, 2.30, 1.28, add_wedge_text,
               "#f0f0f0", "#777f89", fs=9.2)
    add_plain_label(ax, 11.90, 0.77, "traj-lang attn | scene cross-attn | self-attn",
                    fs=6.9, color=COLORS["muted"])
    add_arrow(ax, (14.66, 3.93), (12.75, 2.24), rad=0.08, color="#4f9643")
    add_arrow(ax, (13.18, 2.46), (12.86, 2.24), rad=-0.12,
              color="#7a5cad", ls="--")
    add_plain_label(ax, 13.42, 2.02, "optional gated\naction-motion",
                    fs=7.0, color="#7a5cad", ha="left")
    add_arrow(ax, (13.08, 1.63), (13.92, 1.63))
    add_box(ax, 13.96, 1.17, 1.36, 0.82,
            "pose heads\nposition\nrotation\nopenness",
            "#fff9e8", ec="#cfaa52", fs=8.3, weight="bold")
    add_arrow(ax, (15.32, 1.59), (15.90, 1.59))
    add_box(ax, 15.94, 1.20, 1.30, 0.76,
            "3D action\nDelta pose\n+ gripper",
            COLORS["pink"], ec="#bd8583", fs=8.5, weight="bold")
    add_arrow(ax, (16.55, 2.00), (16.55, 2.88), rad=-0.2, ls="--", color="#777f89")
    add_plain_label(ax, 16.63, 2.55, "iterative\ndenoising", fs=7.2,
                    color=COLORS["muted"], ha="left")

    add_plain_label(ax, 12.04, 0.20, "(b) 3d action policy", fs=13, weight="bold")


def build_figure():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig, ax = plt.subplots(figsize=(18, 7.05), dpi=200)
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 7.05)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    draw_left_panel(ax)
    ax.add_line(Line2D([6.03, 6.03], [0.42, 6.70], color="#2c3036",
                       lw=1.4, ls=(0, (4, 4)), zorder=20))
    draw_right_panel(ax)

    # Small top labels make the two training roles legible without acting like a title.
    add_plain_label(ax, 0.42, 6.80, "standalone motion supervision",
                    fs=8.2, color=COLORS["muted"], ha="left")
    add_plain_label(ax, 6.45, 6.80, "motion-conditioned 3D action denoising",
                    fs=8.2, color=COLORS["muted"], ha="left")
    return fig


def main():
    fig = build_figure()
    for suffix in ("svg", "pdf", "png"):
        path = OUT_DIR / f"corl_motion_policy_architecture.{suffix}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"wrote {OUT_DIR / 'corl_motion_policy_architecture.svg'}")
    print(f"wrote {OUT_DIR / 'corl_motion_policy_architecture.pdf'}")
    print(f"wrote {OUT_DIR / 'corl_motion_policy_architecture.png'}")


if __name__ == "__main__":
    main()
