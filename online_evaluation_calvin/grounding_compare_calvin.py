"""Cross-model language-grounding comparison for CALVIN.

Tests the hypothesis: "joint finetuning with the motion expert inserted a more
robust grounding ability into the VLM." This is a *weights-level* effect, so it
cannot be probed by perturbing motion tokens at inference (see motion_attn_probe.py).
The right experiment compares two checkpoints that differ ONLY by the motion expert:

  - model A  (--checkpoint)          : flower_actor_3d_motion_latent  (with motion)
  - model B  (--baseline_checkpoint) : flower_actor_3d                (no motion,
                                        same VLM-finetuning recipe / data)

For each rollout frame we extract the SAME per-patch grounding variable from
`rgb3d_feats`: the cosine alignment between each scene patch's feature and the
instruction tokens (max over valid tokens). If the motion expert sharpened the
VLM's grounding, model A's map should localize the manipulation region more than
model B's, on identical frames.

The rollout is driven by model A; both models re-encode the same observations.
The two models are loaded SEQUENTIALLY (A, then B) to avoid OOM.

Run with the SAME shared model flags as eval, plus the baseline checkpoint, e.g.:

  python -m online_evaluation_calvin.grounding_compare_calvin \
      --baseline_checkpoint /.../flower_actor_3d_calvin/last.pth \
      --probe_out_dir vis_outputs/grounding/seq0 \
      --checkpoint /.../3dfavla_motion/last.pth \
      --model_type flower_actor_3d_motion_latent \
      --backbone florence2 --custom_img_size 224 --embedding_dim 960 ... (eval flags)
"""

import argparse
import copy
import gc
import json
import os
import sys

import numpy as np
import torch

# Reuse the visualization / extraction helpers from the motion-attention probe.
from online_evaluation_calvin.motion_attn_probe import (
    _resize, overlay_heatmap, static_patch_map, map_summary, topk_overlap,
    tokenize_instruction,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_probe_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--baseline_checkpoint", type=str, required=True,
                   help="No-motion flower_actor_3d checkpoint (same recipe).")
    p.add_argument("--baseline_model_type", type=str, default="flower_actor_3d")
    # The baseline may use a different working width than the motion model (e.g.
    # C=1024 vs C=960 chosen to fit memory). Build model B with its own dims.
    p.add_argument("--baseline_embedding_dim", type=int, default=-1,
                   help="embedding_dim for the baseline; -1 = same as motion model.")
    p.add_argument("--baseline_num_attn_heads", type=int, default=-1,
                   help="num_attn_heads for the baseline; -1 = same as motion model.")
    p.add_argument("--probe_out_dir", type=str, default="vis_outputs/grounding")
    p.add_argument("--probe_seq_ind", type=int, default=0)
    p.add_argument("--probe_subtask_ind", type=int, default=0)
    p.add_argument("--probe_num_replans", type=int, default=12)
    p.add_argument("--grounding_reduce", type=str, default="max", choices=["max", "mean"])
    p.add_argument("--probe_alpha", type=float, default=0.55)
    p.add_argument("--upsample", type=str, default="linear", choices=["linear", "nearest"])
    p.add_argument("--cmap", type=str, default="turbo")
    p.add_argument("--stop_on_success", type=lambda s: s.lower() in ("1", "true", "yes"),
                   default=True)
    probe_args, remaining = p.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return probe_args


@torch.no_grad()
def grounding_map(rgb3d_feats, instr_feats, instr_mask, reduce="max"):
    """Per-patch language grounding = cosine(rgb3d_feats_patch, instruction tokens),
    reduced over valid tokens. Returns (B, N)."""
    rf = torch.nn.functional.normalize(rgb3d_feats.float(), dim=-1)    # (B, N, F)
    itf = torch.nn.functional.normalize(instr_feats.float(), dim=-1)   # (B, L, F)
    sim = torch.matmul(rf, itf.transpose(1, 2))                        # (B, N, L)
    if instr_mask is not None:
        m = instr_mask.to(sim.device).bool()[:, None, :]               # (B, 1, L)
        if reduce == "max":
            sim = sim.masked_fill(~m, float("-inf"))
            return sim.max(dim=-1).values
        sim = sim.masked_fill(~m, 0.0)
        return sim.sum(dim=-1) / m.sum(dim=-1).clamp_min(1)
    return sim.max(dim=-1).values if reduce == "max" else sim.mean(dim=-1)


@torch.no_grad()
def encode_grounding(model, obs, instr, instr_mask, reduce):
    """Run a model's encoder on one observation and return (static_grounding_map_hw,
    grid). encode_inputs()[1]=rgb3d_feats, [5]=instr_feats for both model types."""
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model.policy.encode_inputs(
            obs["rgb"], None, obs["pcd"], instr, obs["proprio"], instr_mask=instr_mask
        )
    rgb3d_feats, instr_feats = out[1], out[5]
    grid = getattr(model.policy.encoder, "_last_rgb3d_grid", None)
    if grid is None:
        raise RuntimeError("Encoder did not stash _last_rgb3d_grid (need florence2).")
    g = grounding_map(rgb3d_feats, instr_feats, instr_mask, reduce=reduce)
    return static_patch_map(g, grid), grid


def shape_cosine(a, b):
    """Cosine between unit-sum-normalized maps -> compares spatial SHAPE only
    (magnitude-invariant). ~1 means same localization pattern."""
    a = a.reshape(-1).astype(np.float64); b = b.reshape(-1).astype(np.float64)
    a = a - a.min(); b = b - b.min()  # cosine of non-negative saliency shapes
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float((a @ b) / (na * nb + 1e-8))


def save_panel(path, frame, gA, gB, alpha, cmap, mode, title):
    vmax = max(gA.max(), gB.max())
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.4))
    axes[0].imshow(frame); axes[0].set_title("static frame", fontsize=9)
    axes[1].imshow(overlay_heatmap(frame, gA, vmax, cmap, alpha, mode))
    axes[1].set_title(f"motion model (max={gA.max():.3g})", fontsize=9)
    axes[2].imshow(overlay_heatmap(frame, gB, vmax, cmap, alpha, mode))
    axes[2].set_title(f"baseline (max={gB.max():.3g})", fontsize=9)
    diff = gA - gB
    vlim = float(np.abs(diff).max()) + 1e-8
    H, W = frame.shape[:2]
    axes[3].imshow(frame)
    axes[3].imshow(_resize(diff, (H, W), mode), cmap="bwr", vmin=-vlim, vmax=vlim, alpha=alpha)
    axes[3].set_title("motion - baseline", fontsize=9)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


@torch.no_grad()
def main():
    pa = parse_probe_args()

    from omegaconf import OmegaConf
    import hydra
    from online_evaluation_calvin.evaluate_policy import parse_arguments, NUM_SEQUENCES
    from online_evaluation_calvin.model_wrapper import create_model
    from online_evaluation_calvin.utils_with_calvin import (
        get_env, get_env_state_for_initial_condition,
    )
    from online_evaluation_calvin.multistep_sequences import get_sequences

    args = parse_arguments()
    args.local_rank = 0
    args.device = torch.device("cuda")

    import random
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    os.makedirs(pa.probe_out_dir, exist_ok=True)

    instr_dict = OmegaConf.load(args.ann_config_file)
    task_oracle = hydra.utils.instantiate(OmegaConf.load(args.task_config_file))
    initial_state, eval_sequence = get_sequences(NUM_SEQUENCES)[pa.probe_seq_ind]
    subtask = eval_sequence[pa.probe_subtask_ind]
    lang = instr_dict[subtask][0]
    print(f"[grounding] seq={pa.probe_seq_ind} subtask={subtask} task={lang!r}")

    # ---------- Phase 1: roll out with model A (motion), collect frames + A grounding.
    env = get_env(args.merged_config_file, show_gui=False)
    model_a = create_model(args)
    instr_a, mask_a = tokenize_instruction(model_a, lang)

    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    model_a.reset()
    start_info = env.get_info()

    records = []
    raw_obs = env.get_obs()
    for replan in range(pa.probe_num_replans):
        frame = np.asarray(raw_obs["rgb_obs"]["rgb_static"])
        obs = model_a.preprocess(raw_obs, env)
        gA, grid = encode_grounding(model_a, obs, instr_a, mask_a, pa.grounding_reduce)
        records.append({
            "frame": frame, "grid": grid, "lang": lang,
            "rgb": obs["rgb"].detach().cpu(), "pcd": obs["pcd"].detach().cpu(),
            "proprio": obs["proprio"].detach().cpu(), "gA": gA,
        })
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            trajectory = model_a.step(obs, lang)
        exec_mode = getattr(model_a, "calvin_action_exec", "cartesian_abs")
        solved = False
        for act_ind in range(trajectory.shape[1]):
            if exec_mode == "relative_array":
                step_action = np.asarray(trajectory[0, act_ind], dtype=np.float64).reshape(7)
                step_action[6] = 1 if step_action[6] > 0 else -1
                raw_obs, _, _, current_info = env.step(step_action)
            else:
                raw_obs, _, _, current_info = env.step({
                    "action": [trajectory[0, act_ind, :3], trajectory[0, act_ind, 3:6],
                               trajectory[0, act_ind, [6]]],
                    "type": exec_mode,
                })
            if pa.stop_on_success and len(task_oracle.get_task_info_for_set(
                    start_info, current_info, {subtask})) > 0:
                solved = True; break
        if solved:
            print(f"[grounding] task solved at replan {replan}; stopping.")
            break
    env.close()
    del model_a; gc.collect(); torch.cuda.empty_cache()
    print(f"[grounding] collected {len(records)} frames; loading baseline...")

    # ---------- Phase 2: re-encode the SAME frames with model B (no motion).
    args_b = copy.copy(args)
    args_b.checkpoint = pa.baseline_checkpoint
    args_b.model_type = pa.baseline_model_type
    args_b.motion_expert_config = None
    args_b.motion_expert_checkpoint = None
    if pa.baseline_embedding_dim > 0:
        args_b.embedding_dim = pa.baseline_embedding_dim
    if pa.baseline_num_attn_heads > 0:
        args_b.num_attn_heads = pa.baseline_num_attn_heads
    print(f"[grounding] baseline arch: embedding_dim={args_b.embedding_dim} "
          f"num_attn_heads={args_b.num_attn_heads}")
    model_b = create_model(args_b)
    instr_b, mask_b = tokenize_instruction(model_b, lang)

    metrics = []
    for i, rec in enumerate(records):
        obs = {k: rec[k].to(args.device) for k in ("rgb", "pcd", "proprio")}
        gB, _ = encode_grounding(model_b, obs, instr_b, mask_b, pa.grounding_reduce)
        gA = rec["gA"]; frame = rec["frame"]
        title = (f"step {i} | grounding=cos(rgb3d_feats, instr) [{pa.grounding_reduce}] | "
                 f"grid={rec['grid'][1]}x{rec['grid'][2]} | task={rec['lang']!r}")
        save_panel(os.path.join(pa.probe_out_dir, f"step_{i:03d}.png"),
                   frame, gA, gB, pa.probe_alpha, pa.cmap, pa.upsample, title)
        np.savez_compressed(os.path.join(pa.probe_out_dir, f"step_{i:03d}.npz"),
                            frame=frame, grid=np.asarray(rec["grid"]),
                            grounding_motion=gA, grounding_baseline=gB)
        m = {
            "step": i,
            "motion": map_summary(gA), "baseline": map_summary(gB),
            "shape_cosine_motion_baseline": shape_cosine(gA, gB),
            "top10pct_overlap_motion_baseline": topk_overlap(gA, gB),
        }
        metrics.append(m)
        print(f"[grounding] step {i}: peak(motion)={m['motion']['top5pct_mass_frac']:.3f} "
              f"peak(baseline)={m['baseline']['top5pct_mass_frac']:.3f} "
              f"shape_cos={m['shape_cosine_motion_baseline']:.3f} "
              f"top10overlap={m['top10pct_overlap_motion_baseline']:.3f}")
    del model_b; gc.collect(); torch.cuda.empty_cache()

    with open(os.path.join(pa.probe_out_dir, "metrics.json"), "w") as f:
        json.dump({"task": lang, "subtask": subtask,
                   "checkpoint_motion": args.checkpoint,
                   "checkpoint_baseline": pa.baseline_checkpoint,
                   "steps": metrics}, f, indent=2)

    if metrics:
        def avg(path):
            vals = []
            for m in metrics:
                v = m
                for k in path:
                    v = v[k]
                vals.append(v)
            return float(np.mean(vals))
        print("\n[grounding] ===== rollout averages =====")
        print(f"  peakiness (top5% mass) motion   : {avg(['motion','top5pct_mass_frac']):.3f}")
        print(f"  peakiness (top5% mass) baseline : {avg(['baseline','top5pct_mass_frac']):.3f}")
        print(f"  shape cosine (motion, baseline) : {avg(['shape_cosine_motion_baseline']):.3f}")
        print(f"  top10% overlap (motion,baseline): {avg(['top10pct_overlap_motion_baseline']):.3f}")
        print("  -> motion sharper if its peakiness > baseline; low shape-cosine means "
              "the two models ground DIFFERENT regions.")
    print(f"[grounding] wrote {len(metrics)} steps to {pa.probe_out_dir}")


if __name__ == "__main__":
    main()
