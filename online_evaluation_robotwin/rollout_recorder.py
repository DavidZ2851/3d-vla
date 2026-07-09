"""Opt-in RGBD recorder for RoboTwin2 online eval rollouts.

Activated only when ``ROBOTWIN_ROLLOUT_DUMP_DIR`` is set, so it never affects a
normal eval run. During a rollout it captures the static head camera RGB +
metric depth every sim step and, per episode, writes:

  {task}_ep{N}_head_rgb.mp4         RGB video (motion-viz reads this)
  {task}_ep{N}_head_depth_raw.npz   metric depth (T,H,W) under key "depth"
  {task}_ep{N}_head_summary.json    instruction + head intrinsics/extrinsics_w2c

This is the RoboTwin2 analog of the CALVIN/LIBERO rollout_task_video outputs, so
``motion_expert/visualize_robotwin2_rollout_motion_prediction.py`` can pick a
frame and run the motion-expert spaghetti prediction.

RoboTwin ``extrinsic_cv`` is already world->camera (OpenCV) and ``intrinsic_cv``
is OpenCV intrinsics, with metric depth in meters -- the exact convention
``RoboTwinDepth2Cloud`` / the motion-viz unprojection expect. So the summary
stores the head extrinsic_cv verbatim as ``head_extrinsics_w2c`` (no inversion).

Env vars:
  ROBOTWIN_ROLLOUT_DUMP_DIR     output dir (required to activate)
  ROBOTWIN_ROLLOUT_CAMERA       camera to record (default head_camera)
  ROBOTWIN_ROLLOUT_FPS          rgb/depth video fps (default 20)
  ROBOTWIN_ROLLOUT_MAX_EPISODES stop after this many episodes (default 1)
"""

from __future__ import annotations

import atexit
import json
import os
from pathlib import Path

import numpy as np


def _slugify(text: str) -> str:
    import re

    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text).strip())
    return text.strip("_") or "task"


class RoboTwinRolloutRecorder:
    def __init__(self, dump_dir, task_name, camera="head_camera", fps=20, max_episodes=1):
        self.dump_dir = Path(dump_dir)
        self.task_name = _slugify(task_name or "task")
        self.camera = camera
        self.fps = int(fps)
        self.max_episodes = int(max_episodes)
        self.episode_idx = 0
        self._rgb_frames = []
        self._depth_frames = []
        self._intrinsics = None
        self._extrinsics_w2c = None
        self._instruction = ""
        self._img_hw = None
        self._closed = False
        self.dump_dir.mkdir(parents=True, exist_ok=True)
        atexit.register(self.close)
        print(
            f"[rw2-rollout-recorder] active: dir={self.dump_dir} camera={self.camera} "
            f"fps={self.fps} max_episodes={self.max_episodes}",
            flush=True,
        )

    @classmethod
    def maybe_from_env(cls, task_name):
        dump_dir = os.environ.get("ROBOTWIN_ROLLOUT_DUMP_DIR", "").strip()
        if not dump_dir:
            return None
        return cls(
            dump_dir=dump_dir,
            task_name=task_name,
            camera=os.environ.get("ROBOTWIN_ROLLOUT_CAMERA", "head_camera").strip() or "head_camera",
            fps=int(os.environ.get("ROBOTWIN_ROLLOUT_FPS", "20") or 20),
            max_episodes=int(os.environ.get("ROBOTWIN_ROLLOUT_MAX_EPISODES", "1") or 1),
        )

    def note_instruction(self, instruction):
        if instruction and not self._instruction:
            self._instruction = str(instruction)

    @staticmethod
    def _extrinsic_to_4x4(extrinsic):
        extrinsic = np.asarray(extrinsic, dtype=np.float32)
        if extrinsic.shape == (4, 4):
            return extrinsic
        if extrinsic.shape == (3, 4):
            out = np.eye(4, dtype=np.float32)
            out[:3, :4] = extrinsic
            return out
        raise ValueError(f"head_camera extrinsic_cv must be (3,4) or (4,4), got {extrinsic.shape}.")

    def on_obs(self, observation):
        if self._closed or self.episode_idx >= self.max_episodes:
            return
        cam = observation.get("observation", {}).get(self.camera)
        if cam is None:
            return
        rgb = np.ascontiguousarray(np.asarray(cam["rgb"], dtype=np.uint8))
        depth = np.ascontiguousarray(np.asarray(cam["depth"], dtype=np.float32))
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        self._rgb_frames.append(rgb)
        self._depth_frames.append(depth)
        self._img_hw = (int(depth.shape[0]), int(depth.shape[1]))
        if self._intrinsics is None:
            self._intrinsics = np.asarray(cam["intrinsic_cv"], dtype=np.float32)
            self._extrinsics_w2c = self._extrinsic_to_4x4(cam["extrinsic_cv"])

    def flush_episode(self, success=None):
        if self._closed or not self._rgb_frames or self.episode_idx >= self.max_episodes:
            self._reset_buffers()
            return
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover
            print(f"[rw2-rollout-recorder] OpenCV unavailable, skipping dump: {exc}", flush=True)
            self._reset_buffers()
            return

        stem = f"{self.task_name}_ep{self.episode_idx:02d}_{self.camera}"
        rgb_path = self.dump_dir / f"{stem}_rgb.mp4"
        depth_npz_path = self.dump_dir / f"{stem}_depth_raw.npz"
        summary_path = self.dump_dir / f"{stem}_summary.json"

        h, w = self._rgb_frames[0].shape[:2]
        writer = cv2.VideoWriter(str(rgb_path), cv2.VideoWriter_fourcc(*"mp4v"), float(max(1, self.fps)), (w, h))
        if writer.isOpened():
            for frame in self._rgb_frames:
                writer.write(np.ascontiguousarray(frame)[:, :, ::-1])  # RGB -> BGR
            writer.release()
        else:
            print(f"[rw2-rollout-recorder] failed to open writer {rgb_path}", flush=True)

        depth_stack = np.stack([np.squeeze(d) for d in self._depth_frames]).astype(np.float32)
        np.savez_compressed(depth_npz_path, depth=depth_stack)

        finite = depth_stack[np.isfinite(depth_stack) & (depth_stack > 0)]
        depth_min = float(np.percentile(finite, 1.0)) if finite.size else 0.0
        depth_max = float(np.percentile(finite, 99.0)) if finite.size else 1.0
        summary = {
            "task_name": self.task_name,
            "episode_index": self.episode_idx,
            "instruction": self._instruction,
            "camera": self.camera,
            "success": (None if success is None else bool(success)),
            "num_rgb_frames": len(self._rgb_frames),
            "num_depth_frames": len(self._depth_frames),
            "fps": self.fps,
            "img_h": self._img_hw[0],
            "img_w": self._img_hw[1],
            "rgb_video": str(rgb_path),
            "raw_depth_npz": str(depth_npz_path),
            "depth_min": depth_min,
            "depth_max": depth_max,
            "head_intrinsics": np.asarray(self._intrinsics, dtype=np.float32).tolist(),
            "head_extrinsics_w2c": np.asarray(self._extrinsics_w2c, dtype=np.float32).tolist(),
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(
            f"[rw2-rollout-recorder] saved episode {self.episode_idx}: {rgb_path.name} "
            f"({len(self._rgb_frames)} frames)",
            flush=True,
        )
        self.episode_idx += 1
        self._reset_buffers()

    def _reset_buffers(self):
        self._rgb_frames = []
        self._depth_frames = []
        self._intrinsics = None
        self._extrinsics_w2c = None
        self._instruction = ""

    def close(self):
        if self._closed:
            return
        # Flush a final in-progress episode (no reset_obs fires after the last one).
        if self._rgb_frames and self.episode_idx < self.max_episodes:
            self.flush_episode()
        self._closed = True
