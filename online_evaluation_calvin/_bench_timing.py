"""Per-action-chunk inference timing for *live* CALVIN online evaluation.

Activated from ``evaluate_policy.py`` when ``BENCH_TIMING=1``. Wraps the model's
``step()`` (one action chunk) and the three forward stages, recording CUDA-synced
latencies during real rollouts. The first ``BENCH_WARMUP`` chunks are skipped.

Stage boundaries (one call each per chunk):
  VLM            = BaseDenoiseActor.encode_inputs (Florence2 / Encoder3D)
  motion expert  = FlowerActor3DMotionLatent._build_motion_tokens (ODE solver)
  action policy  = BaseDenoiseActor.conditional_sample (denoising loop)
"""
import json
import os
import time

import numpy as np
import torch

from modeling.policy.base_denoise_actor import DenoiseActor as BaseDenoiseActor
from modeling.policy.flower_actor import FlowerActor3DMotionLatent


class RolloutTimer:
    def __init__(self, warmup=10):
        self.warmup = warmup
        self.records = {"total": [], "vlm": [], "motion": [], "action": []}
        self._orig = []
        self._calls = 0          # number of step() calls seen
        self._recording = False  # gates stage + total recording for the active step

    def _wrap_stage(self, cls, name, key):
        orig = getattr(cls, name)
        rec = self.records[key]

        def wrapped(*a, **kw):
            if not self._recording:
                return orig(*a, **kw)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = orig(*a, **kw)
            torch.cuda.synchronize()
            rec.append((time.perf_counter() - t0) * 1000.0)
            return out

        setattr(cls, name, wrapped)
        self._orig.append((cls, name, orig))

    def install(self, model):
        # Per-chunk total: wrap the bound model.step.
        orig_step = model.step

        def step_wrapped(*a, **kw):
            self._calls += 1
            self._recording = self._calls > self.warmup
            if not self._recording:
                return orig_step(*a, **kw)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = orig_step(*a, **kw)
            torch.cuda.synchronize()
            self.records["total"].append((time.perf_counter() - t0) * 1000.0)
            return out

        model.step = step_wrapped
        self._model = model
        self._orig_step = orig_step

        self._wrap_stage(BaseDenoiseActor, "encode_inputs", "vlm")
        self._wrap_stage(BaseDenoiseActor, "conditional_sample", "action")
        self._wrap_stage(FlowerActor3DMotionLatent, "_build_motion_tokens", "motion")
        return self

    def restore(self):
        self._model.step = self._orig_step
        for cls, name, orig in self._orig:
            setattr(cls, name, orig)

    @staticmethod
    def _summ(a):
        a = np.asarray(a, dtype=np.float64)
        if a.size == 0:
            return None
        return {
            "n": int(a.size),
            "mean_ms": float(a.mean()),
            "std_ms": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "median_ms": float(np.median(a)),
            "p5_ms": float(np.percentile(a, 5)),
            "p95_ms": float(np.percentile(a, 95)),
            "min_ms": float(a.min()),
            "max_ms": float(a.max()),
        }

    def report(self, tag="", pred_len=None, json_out=None):
        total = self._summ(self.records["total"])
        print("\n==== ONLINE-EVAL INFERENCE TIMING ====")
        print(f"model: {tag}")
        if total is None:
            print("no chunks recorded (did warmup exceed the rollout length?)")
            return
        mean = total["mean_ms"]
        print(f"chunks measured: {total['n']} (warmup {self.warmup} skipped)")
        print(f"per-chunk latency: {mean:.1f} +/- {total['std_ms']:.1f} ms "
              f"(median {total['median_ms']:.1f}, p5 {total['p5_ms']:.1f}, "
              f"p95 {total['p95_ms']:.1f})")
        print(f"throughput: {1000.0/mean:.2f} chunks/s", end="")
        if pred_len:
            print(f" ({1000.0/mean*pred_len:.1f} actions/s at pred_len={pred_len})")
        else:
            print()
        labels = {"vlm": "VLM (Florence2)", "motion": "motion expert",
                  "action": "action policy"}
        stages = {}
        print("  per-stage breakdown (mean +/- std ms per chunk):")
        for key in ("vlm", "motion", "action"):
            s = self._summ(self.records[key])
            if s is None:
                continue
            stages[key] = s
            print(f"    {labels[key]:<18}: {s['mean_ms']:6.1f} +/- {s['std_ms']:4.1f} "
                  f"(median {s['median_ms']:.1f})  [{100*s['mean_ms']/mean:.0f}% of chunk]")
        if json_out:
            out = {"tag": tag, "pred_len": pred_len, "warmup": self.warmup,
                   "total": total, "stages": stages}
            with open(json_out, "w") as f:
                json.dump(out, f, indent=2)
            print(f"[bench] wrote {json_out}")


def maybe_install(model):
    """Return an installed RolloutTimer if BENCH_TIMING is set, else None."""
    if os.environ.get("BENCH_TIMING", "0") not in ("1", "true", "True"):
        return None
    warmup = int(os.environ.get("BENCH_WARMUP", "10"))
    print(f"[bench] online-eval inference timing ENABLED (warmup={warmup})")
    return RolloutTimer(warmup=warmup).install(model)
