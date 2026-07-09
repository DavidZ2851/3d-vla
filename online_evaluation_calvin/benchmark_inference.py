"""Measure per-action-chunk inference latency for CALVIN policies.

Times a single ``model.step()`` call (one action chunk of ``pred_len`` actions)
under the exact eval-time forward path (bf16 autocast, no_grad), excluding the
CALVIN simulator. Inputs are synthetic but correctly shaped, so the measured
compute is identical to deployment. Results are aggregated over several
"rollouts" (each preceded by ``model.reset()``), discarding warmup.

Usage (run inside the flowervla conda env):
    python -m online_evaluation_calvin.benchmark_inference --preset baseline
    python -m online_evaluation_calvin.benchmark_inference --preset motion
"""
import argparse
import json
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

from online_evaluation_calvin.evaluate_policy import parse_arguments
from online_evaluation_calvin.model_wrapper import create_model
from modeling.policy.base_denoise_actor import DenoiseActor as BaseDenoiseActor
from modeling.policy.flower_actor import FlowerActor3DMotionLatent

REPO = "/data/user_data/ziyinx/analogical_manipulation_clean"

# Architecture/eval args per checkpoint, derived from the saved state dicts and
# the corresponding train/eval scripts. Both checkpoints carry the full Florence2
# backbone with no LoRA adapters.
PRESETS = {
    # flower_actor_3d baseline (train_logs/calvin/3dfavla): embedding_dim 1024,
    # full-finetuned backbone, no motion expert.
    "baseline": [
        "--model_type", "flower_actor_3d",
        "--checkpoint", f"{REPO}/train_logs/calvin/3dfavla/last.pth",
        "--embedding_dim", "1024",
    ],
    # flower_actor_3d_motion_latent, h15 motion expert (calvin_florence2 has
    # future_steps: 15). train_logs/calvin/3dfavla_motion: embedding_dim 960.
    "motion": [
        "--model_type", "flower_actor_3d_motion_latent",
        "--checkpoint", f"{REPO}/train_logs/calvin/3dfavla_motion/last.pth",
        "--embedding_dim", "960",
        "--motion_expert_config", f"{REPO}/motion_expert/configs/calvin_florence2.yaml",
        "--motion_expert_checkpoint",
        f"{REPO}/train_logs/motion_expert/calvin_florence2/checkpoint_step_18000.pt",
        "--motion_solver_steps", "10",
        "--motion_steps_to_run", "1",
        "--motion_finetune", "false",
        "--motion_scene_fusion", "true",
        "--motion_action_fusion", "false",
    ],
}

# Args common to both presets (mirrors eval_calvin.sh / train_flower_actor.sh).
COMMON = [
    "--backbone", "florence2",
    "--pred_len", "10",
    "--pre_tokenize", "true",
    "--custom_img_size", "224",
    "--num_finetune_vision_layers", "24",
    "--vision_lora_rank", "0",
    "--llm_lora_rank", "0",
    "--num_attn_heads", "16",
    "--num_vis_instr_attn_layers", "3",
    "--num_shared_attn_layers", "16",
    "--relative_action", "true",
    "--rotation_format", "euler",
    "--denoise_timesteps", "5",
    "--denoise_model", "rectified_flow",
]


# ---------------------------------------------------------------------------
# Per-stage timing. We wrap the three forward stages of a single model.step():
#   VLM            = BaseDenoiseActor.encode_inputs (Florence2 / Encoder3D). For
#                    the motion model this is reached via super().encode_inputs,
#                    so it captures the VLM only (not the motion fusion).
#   motion expert  = FlowerActor3DMotionLatent._build_motion_tokens (the motion
#                    expert ODE solver, solver_steps iterations).
#   action policy  = BaseDenoiseActor.conditional_sample (denoise_timesteps
#                    iterations of the action prediction head).
# Each stage runs exactly once per step(), so the recorded lists are aligned.
# ---------------------------------------------------------------------------
class StageTimers:
    def __init__(self):
        self.records = {"vlm": [], "motion": [], "action": []}
        self._orig = []

    def _wrap(self, cls, name, key):
        orig = getattr(cls, name)
        rec = self.records[key]

        def wrapped(*a, **kw):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = orig(*a, **kw)
            torch.cuda.synchronize()
            rec.append((time.perf_counter() - t0) * 1000.0)
            return out

        setattr(cls, name, wrapped)
        self._orig.append((cls, name, orig))

    def install(self):
        self._wrap(BaseDenoiseActor, "encode_inputs", "vlm")
        self._wrap(BaseDenoiseActor, "conditional_sample", "action")
        self._wrap(FlowerActor3DMotionLatent, "_build_motion_tokens", "motion")
        return self

    def restore(self):
        for cls, name, orig in self._orig:
            setattr(cls, name, orig)

    def reset(self):
        for v in self.records.values():
            v.clear()


def build_args(preset):
    saved = sys.argv
    sys.argv = ["benchmark_inference.py"] + COMMON + PRESETS[preset]
    try:
        args = parse_arguments()
    finally:
        sys.argv = saved
    args.local_rank = 0
    args.device = torch.device("cuda")
    return args


def make_fake_env():
    cam = SimpleNamespace(view_matrix=list(np.eye(4).reshape(-1)))
    return SimpleNamespace(cameras=[None, cam])


def make_fake_obs(rng):
    return {
        "robot_obs": rng.standard_normal(15).astype(np.float32),
        "rgb_obs": {
            "rgb_static": rng.integers(0, 256, (200, 200, 3), dtype=np.uint8),
            "rgb_gripper": rng.integers(0, 256, (84, 84, 3), dtype=np.uint8),
        },
        "depth_obs": {
            "depth_static": (rng.random((200, 200)).astype(np.float32) * 2.0 + 0.3),
            "depth_gripper": (rng.random((84, 84)).astype(np.float32) * 2.0 + 0.3),
        },
    }


def summarize(latencies_ms):
    a = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "n": int(a.size),
        "mean_ms": float(a.mean()),
        "std_ms": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "median_ms": float(np.median(a)),
        "p5_ms": float(np.percentile(a, 5)),
        "p95_ms": float(np.percentile(a, 95)),
        "min_ms": float(a.min()),
        "max_ms": float(a.max()),
        "hz": float(1000.0 / a.mean()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=list(PRESETS), required=True)
    p.add_argument("--rollouts", type=int, default=5)
    p.add_argument("--steps", type=int, default=36, help="chunks per rollout")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--instruction", default="push the block to the right")
    p.add_argument("--json_out", default=None)
    p.add_argument("--breakdown", action="store_true",
                   help="also report per-stage (VLM / motion / action) timings")
    bench = p.parse_args()

    args = build_args(bench.preset)
    torch.manual_seed(0)
    np.random.seed(0)
    rng = np.random.default_rng(0)

    print(f"[bench] preset={bench.preset} model_type={args.model_type} "
          f"embedding_dim={args.embedding_dim} pred_len={args.pred_len}")
    model = create_model(args)
    model.reset()

    env = make_fake_env()
    # One representative preprocessed observation; forward latency is data-independent.
    obs = model.preprocess(make_fake_obs(rng), env)

    def one_step():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return model.step(obs, bench.instruction)

    # Warmup (triggers cudnn autotune / lazy inits, not measured).
    for _ in range(bench.warmup):
        one_step()
    torch.cuda.synchronize()

    timers = StageTimers().install() if bench.breakdown else None
    per_rollout_means = []
    all_latencies = []
    for r in range(bench.rollouts):
        model.reset()
        roll = []
        for _ in range(bench.steps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            one_step()
            torch.cuda.synchronize()
            roll.append((time.perf_counter() - t0) * 1000.0)
        all_latencies.extend(roll)
        per_rollout_means.append(float(np.mean(roll)))
        print(f"  rollout {r}: mean {np.mean(roll):.1f} ms over {len(roll)} chunks")

    stats = summarize(all_latencies)
    rollout_mean = float(np.mean(per_rollout_means))
    rollout_std = float(np.std(per_rollout_means, ddof=1)) if len(per_rollout_means) > 1 else 0.0

    print("\n==== RESULT ====")
    print(f"model: {bench.preset} ({args.model_type})")
    print(f"chunks measured: {stats['n']} "
          f"({bench.rollouts} rollouts x {bench.steps} chunks, warmup {bench.warmup})")
    print(f"per-chunk latency: {stats['mean_ms']:.1f} +/- {stats['std_ms']:.1f} ms "
          f"(median {stats['median_ms']:.1f}, p5 {stats['p5_ms']:.1f}, p95 {stats['p95_ms']:.1f})")
    print(f"per-rollout mean latency: {rollout_mean:.1f} +/- {rollout_std:.1f} ms")
    print(f"throughput: {stats['hz']:.2f} chunks/s "
          f"({stats['hz'] * args.pred_len:.1f} actions/s at pred_len={args.pred_len})")

    stage_out = {}
    if timers is not None:
        timers.restore()
        print("\n  per-stage breakdown (mean +/- std ms per chunk):")
        order = ["vlm", "motion", "action"]
        labels = {"vlm": "VLM (Florence2)", "motion": "motion expert",
                  "action": "action policy"}
        for key in order:
            vals = timers.records[key]
            if not vals:
                continue
            s = summarize(vals)
            stage_out[key] = s
            print(f"    {labels[key]:<18}: {s['mean_ms']:6.1f} +/- {s['std_ms']:4.1f} "
                  f"(median {s['median_ms']:.1f})  [{100*s['mean_ms']/stats['mean_ms']:.0f}% of chunk]")

    out = {"preset": bench.preset, "model_type": args.model_type,
           "pred_len": args.pred_len, "rollouts": bench.rollouts,
           "steps_per_rollout": bench.steps, "warmup": bench.warmup,
           "per_rollout_mean_ms": rollout_mean, "per_rollout_std_ms": rollout_std,
           "stages": stage_out, **stats}
    if bench.json_out:
        with open(bench.json_out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[bench] wrote {bench.json_out}")


if __name__ == "__main__":
    main()
