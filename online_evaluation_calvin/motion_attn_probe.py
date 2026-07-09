"""Counterfactual motion-attention probe for CALVIN.

Runs ONE CALVIN rollout with `flower_actor_3d_motion_latent` and, at each replan
step, records the *per-patch motion-update norm* on the static camera for:

  - real     : the real predicted motion latent
  - decoy     : motion latent from a DIFFERENT instruction, scored against the
                exact same scene features (kills the "it's just the backbone"
                confound — same scene query, different plan)
  - shuffle   : motion tokens spatially permuted (positions fixed) — same latent
                magnitudes, scrambled plan->location binding
  - zero      : motion tokens zeroed — floor for the gated residual

The per-patch update norm is ||alpha_scene * scene_motion_attn(rgb3d_feats=query,
motion_tokens=key/value)||. Because patches are the *query*, the softmax is
normalized per-patch over motion tokens and is NOT an importance map; the residual
update magnitude is the well-defined "how much did the motion latent change this
patch" signal (it already folds in value vectors and the learned alpha_scene gate).

The real policy actions are always driven by the REAL motion latent; the
counterfactuals are computed only for visualization and never affect execution.

Run (single GPU, no torchrun) with the SAME model flags you use for eval, e.g.:

  export QWEN3VL_MODEL_PATH=/home/ziyinx/Xiaomi-Robotics-0-Pretrain
  python -m online_evaluation_calvin.motion_attn_probe \
      --probe_out_dir vis_outputs/motion_attn/seq0 \
      --probe_num_replans 12 \
      --checkpoint /.../last.pth \
      --model_type flower_actor_3d_motion_latent \
      --backbone florence2 --custom_img_size 224 \
      --embedding_dim 960 --num_attn_heads 16 --num_vis_instr_attn_layers 3 \
      --num_finetune_vision_layers 24 --llm_lora_rank 64 --llm_lora_alpha 128 \
      --llm_lora_num_layers 36 --fps_subsampling_factor 1 \
      --motion_expert_config /.../calvin_florence2.yaml \
      --motion_expert_checkpoint /.../checkpoint_step_18000.pt \
      --motion_scene_fusion true \
      --merged_config_file online_evaluation_calvin/configs/merged_config_val_abc_d_notactile.yaml
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:  # cmap lookup that works across matplotlib versions
    from matplotlib import colormaps as _mpl_colormaps

    def _get_cmap(name):
        return _mpl_colormaps[name]
except Exception:  # pragma: no cover - old matplotlib
    from matplotlib import cm as _mpl_cm

    def _get_cmap(name):
        return _mpl_cm.get_cmap(name)


def parse_probe_args():
    """Pull probe-only flags out of argv, then let evaluate_policy.parse_arguments
    parse the (identical) model/eval flags from what remains."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--probe_out_dir", type=str, default="vis_outputs/motion_attn")
    p.add_argument("--probe_seq_ind", type=int, default=0,
                   help="Which sampled CALVIN sequence to roll out.")
    p.add_argument("--probe_subtask_ind", type=int, default=0,
                   help="Which subtask in the sequence to probe.")
    p.add_argument("--probe_num_replans", type=int, default=12,
                   help="Number of policy replans (heatmap frames) to record.")
    p.add_argument("--decoy_instruction", type=str, default="",
                   help="Counterfactual instruction. Empty -> auto-pick a "
                        "different subtask's annotation.")
    p.add_argument("--probe_alpha", type=float, default=0.55,
                   help="Heatmap overlay opacity.")
    p.add_argument("--upsample", type=str, default="linear",
                   choices=["linear", "nearest"],
                   help="Patch-grid -> image upsampling for the overlay.")
    p.add_argument("--cmap", type=str, default="turbo")
    p.add_argument("--stop_on_success", type=lambda s: s.lower() in ("1", "true", "yes"),
                   default=True)
    probe_args, remaining = p.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return probe_args


def tokenize_instruction(model, text):
    """Mirror Model.step's instruction handling."""
    if not model.args.pre_tokenize:
        return [text], None
    tok = model.tokenizer([text])
    if isinstance(tok, torch.Tensor):
        return tok.cuda(non_blocking=True), None
    return (tok.input_ids.cuda(non_blocking=True),
            tok.attention_mask.cuda(non_blocking=True))


def static_patch_map(flat_map, grid):
    """(B, ncam*h*w) -> (h, w) numpy for camera 0 (static), batch 0."""
    _, h, w = grid
    arr = flat_map[0, : h * w].detach().cpu().numpy().astype(np.float32)
    return arr.reshape(h, w)


def _resize(map_hw, out_hw, mode):
    """Upsample a small (h, w) map to image size (H, W) without a cv2 dependency."""
    src = torch.from_numpy(map_hw)[None, None].float()
    out = torch.nn.functional.interpolate(
        src, size=out_hw,
        mode=("bilinear" if mode == "linear" else "nearest"),
        align_corners=False if mode == "linear" else None,
    )
    return out[0, 0].numpy()


def overlay_heatmap(frame_rgb, map_hw, vmax, cmap, alpha, mode):
    H, W = frame_rgb.shape[:2]
    big = _resize(map_hw, (H, W), mode)
    norm = np.clip(big / (vmax + 1e-8), 0.0, 1.0)
    heat = _get_cmap(cmap)(norm)[..., :3]  # (H, W, 3) in [0,1]
    base = frame_rgb.astype(np.float32) / 255.0
    blended = (1.0 - alpha) * base + alpha * heat
    return np.clip(blended, 0.0, 1.0)


def map_summary(map_hw):
    m = map_hw.reshape(-1)
    k = max(1, int(0.05 * m.size))
    topk = np.sort(m)[::-1][:k]
    return {
        "mean": float(m.mean()),
        "max": float(m.max()),
        "p95": float(np.percentile(m, 95)),
        "top5pct_mass_frac": float(topk.sum() / (m.sum() + 1e-8)),
    }


def topk_overlap(a, b, frac=0.1):
    a, b = a.reshape(-1), b.reshape(-1)
    k = max(1, int(frac * a.size))
    sa = set(np.argsort(a)[::-1][:k].tolist())
    sb = set(np.argsort(b)[::-1][:k].tolist())
    return len(sa & sb) / float(k)


def rel_l1(real, other):
    return float(np.abs(real - other).sum() / (real.sum() + 1e-8))


def save_panel(path, frame, maps, vmax, alpha, cmap, mode, title):
    order = [k for k in ("real", "decoy", "shuffle") if k in maps]
    ncol = 2 + len(order)
    fig, axes = plt.subplots(1, ncol, figsize=(3.0 * ncol, 3.4))
    axes[0].imshow(frame)
    axes[0].set_title("static frame", fontsize=9)
    for ax, key in zip(axes[1:1 + len(order)], order):
        ax.imshow(overlay_heatmap(frame, maps[key], vmax, cmap, alpha, mode))
        ax.set_title(f"{key}  (max={maps[key].max():.3g})", fontsize=9)
    # signed difference real - decoy (or real - shuffle if no decoy)
    ref = "decoy" if "decoy" in maps else ("shuffle" if "shuffle" in maps else None)
    dax = axes[-1]
    if ref is not None:
        diff = maps["real"] - maps[ref]
        vlim = float(np.abs(diff).max()) + 1e-8
        H, W = frame.shape[:2]
        dbig = _resize(diff, (H, W), mode)
        dax.imshow(frame)
        dax.imshow(dbig, cmap="bwr", vmin=-vlim, vmax=vlim, alpha=alpha)
        dax.set_title(f"real - {ref}", fontsize=9)
    else:
        dax.axis("off")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


@torch.no_grad()
def main():
    probe_args = parse_probe_args()

    # Import after argv is cleaned so parse_arguments only sees model/eval flags.
    from omegaconf import OmegaConf
    import hydra
    from online_evaluation_calvin.evaluate_policy import parse_arguments, NUM_SEQUENCES
    from online_evaluation_calvin.model_wrapper import create_model
    from online_evaluation_calvin.utils_with_calvin import (
        get_env, get_env_state_for_initial_condition,
    )
    from online_evaluation_calvin.multistep_sequences import get_sequences

    args = parse_arguments()
    for k, v in vars(probe_args).items():
        setattr(args, k, v)
    args.local_rank = 0
    args.device = torch.device("cuda")

    # Determinism (mirror evaluate_policy.main)
    import random
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.probe_out_dir, exist_ok=True)

    model = create_model(args)
    if getattr(model.policy, "scene_motion_attn", None) is None:
        raise RuntimeError(
            "This checkpoint has no scene_motion_attn (motion_scene_fusion=False); "
            "there is no rgb3d cross-attention to probe."
        )
    model.policy._capture_motion_attn = True

    env = get_env(args.merged_config_file, show_gui=False)
    instr_dict = OmegaConf.load(args.ann_config_file)
    task_oracle = hydra.utils.instantiate(OmegaConf.load(args.task_config_file))

    eval_sequences = get_sequences(NUM_SEQUENCES)
    initial_state, eval_sequence = eval_sequences[args.probe_seq_ind]
    subtask = eval_sequence[args.probe_subtask_ind]
    lang = instr_dict[subtask][0]

    decoy = args.decoy_instruction.strip()
    if not decoy:
        for key in sorted(instr_dict.keys()):
            if key != subtask:
                decoy = instr_dict[key][0]
                break
    decoy_instr, decoy_mask = tokenize_instruction(model, decoy)

    print(f"[probe] seq={args.probe_seq_ind} subtask={subtask}")
    print(f"[probe] task instruction : {lang!r}")
    print(f"[probe] decoy instruction: {decoy!r}")
    print(f"[probe] alpha_scene = {float(model.policy.alpha_scene.detach().float().cpu()):.5f}")

    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    model.reset()
    model.policy._capture_motion_attn = True  # reset() does not touch this flag
    start_info = env.get_info()

    metrics = []
    raw_obs = env.get_obs()
    for replan in range(args.probe_num_replans):
        frame = np.asarray(raw_obs["rgb_obs"]["rgb_static"])  # (H, W, 3) uint8
        obs = model.preprocess(raw_obs, env)

        model.policy._motion_attn_seed = replan
        model.policy._motion_attn_capture = None
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            trajectory = model.step(obs, lang)  # real motion -> capture populated

        cap = model.policy._motion_attn_capture
        if cap is None or cap.get("grid") is None:
            raise RuntimeError("Capture missing; expected florence2 + scene fusion.")
        grid = cap["grid"]

        maps = {
            "real": static_patch_map(cap["real"], grid),
            "shuffle": static_patch_map(cap["shuffle"], grid),
            "zero": static_patch_map(cap["zero"], grid),
        }
        # Decoy: same scene query, motion tokens from a different instruction.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            decoy_tokens, _ = model.policy.encode_motion_tokens(
                obs["rgb"], None, obs["pcd"], decoy_instr, obs["proprio"],
                instr_mask=decoy_mask,
            )
            decoy_flat = model.policy.scene_motion_norm_for_tokens(decoy_tokens)
        maps["decoy"] = static_patch_map(decoy_flat, grid)

        vmax = max(maps[k].max() for k in ("real", "decoy", "shuffle"))
        title = (f"step {replan} | alpha_scene={cap['alpha_scene']:.4f} | "
                 f"grid={grid[1]}x{grid[2]} | task={lang!r}")
        panel_path = os.path.join(args.probe_out_dir, f"step_{replan:03d}.png")
        save_panel(panel_path, frame, maps, vmax, args.probe_alpha,
                   args.cmap, args.upsample, title)
        np.savez_compressed(
            os.path.join(args.probe_out_dir, f"step_{replan:03d}.npz"),
            frame=frame, grid=np.asarray(grid), alpha_scene=cap["alpha_scene"],
            **{f"map_{k}": v for k, v in maps.items()},
        )

        step_metrics = {
            "step": replan,
            "alpha_scene": cap["alpha_scene"],
            "real": map_summary(maps["real"]),
            "decoy": map_summary(maps["decoy"]),
            "shuffle": map_summary(maps["shuffle"]),
            "zero": map_summary(maps["zero"]),
            "rel_l1_real_vs_decoy": rel_l1(maps["real"], maps["decoy"]),
            "rel_l1_real_vs_shuffle": rel_l1(maps["real"], maps["shuffle"]),
            "top10pct_overlap_real_decoy": topk_overlap(maps["real"], maps["decoy"]),
            "top10pct_overlap_real_shuffle": topk_overlap(maps["real"], maps["shuffle"]),
        }
        metrics.append(step_metrics)
        print(f"[probe] step {replan}: real.max={maps['real'].max():.3g} "
              f"decoy.max={maps['decoy'].max():.3g} "
              f"shuffle.max={maps['shuffle'].max():.3g} "
              f"relL1(real,decoy)={step_metrics['rel_l1_real_vs_decoy']:.3f} "
              f"top10%overlap(real,decoy)={step_metrics['top10pct_overlap_real_decoy']:.3f}")

        # Execute the real predicted action chunk to advance the rollout.
        exec_mode = getattr(model, "calvin_action_exec", "cartesian_abs")
        solved = False
        for act_ind in range(trajectory.shape[1]):
            if exec_mode == "relative_array":
                step_action = np.asarray(trajectory[0, act_ind], dtype=np.float64).reshape(7)
                step_action[6] = 1 if step_action[6] > 0 else -1
                raw_obs, _, _, current_info = env.step(step_action)
            else:
                curr_action = {
                    "action": [trajectory[0, act_ind, :3],
                               trajectory[0, act_ind, 3:6],
                               trajectory[0, act_ind, [6]]],
                    "type": exec_mode,
                }
                raw_obs, _, _, current_info = env.step(curr_action)
            if args.stop_on_success:
                done = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
                if len(done) > 0:
                    solved = True
                    break
        if solved:
            print(f"[probe] task solved at replan {replan}; stopping.")
            break

    with open(os.path.join(args.probe_out_dir, "metrics.json"), "w") as f:
        json.dump({"task": lang, "decoy": decoy, "subtask": subtask,
                   "steps": metrics}, f, indent=2)

    # Rollout-level averages: the headline counterfactual numbers.
    if metrics:
        def avg(key_path):
            vals = []
            for m in metrics:
                v = m
                for k in key_path:
                    v = v[k]
                vals.append(v)
            return float(np.mean(vals))
        print("\n[probe] ===== rollout averages =====")
        print(f"  alpha_scene                       : {metrics[0]['alpha_scene']:.4f}")
        print(f"  real    top5% mass frac           : {avg(['real','top5pct_mass_frac']):.3f}")
        print(f"  decoy   top5% mass frac           : {avg(['decoy','top5pct_mass_frac']):.3f}")
        print(f"  shuffle top5% mass frac           : {avg(['shuffle','top5pct_mass_frac']):.3f}")
        print(f"  rel L1(real, decoy)               : {avg(['rel_l1_real_vs_decoy']):.3f}")
        print(f"  rel L1(real, shuffle)             : {avg(['rel_l1_real_vs_shuffle']):.3f}")
        print(f"  top10% patch overlap (real,decoy) : {avg(['top10pct_overlap_real_decoy']):.3f}")
    print(f"[probe] wrote {len(metrics)} steps to {args.probe_out_dir}")

    env.close()


if __name__ == "__main__":
    main()
