"""Runtime helpers for LIBERO / LIBERO-PRO / LIBERO-plus online evaluation."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
import yaml

_BENCHMARK_MODULES = None
_LIBERO_ROOT: Optional[str] = None

LIBERO_PRO_MAX_STEPS = {
    "libero_goal": 300,
    "libero_spatial": 220,
    "libero_10": 520,
    "libero_object": 280,
}

LIBERO_PRO_PERTURBATION_SUITES = [
    "libero_goal_object",
    "libero_spatial_object",
    "libero_10_object",
    "libero_object_object",
    "libero_goal_swap",
    "libero_spatial_swap",
    "libero_10_swap",
    "libero_object_swap",
    "libero_goal_lan",
    "libero_spatial_lan",
    "libero_10_lan",
    "libero_object_lan",
    "libero_goal_task",
    "libero_spatial_task",
    "libero_10_task",
    "libero_object_task",
    "libero_goal_env",
    "libero_spatial_env",
    "libero_10_env",
    "libero_object_env",
]

LIBERO_PLUS_SUITES = [
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
]


def infer_max_steps(benchmark_name: str, default: int = 600) -> int:
    for prefix, steps in LIBERO_PRO_MAX_STEPS.items():
        if benchmark_name.startswith(prefix):
            return steps
    return default


def prepare_libero_runtime(libero_root: str) -> str:
    """Point LIBERO path config at the requested repo before importing libero."""
    global _BENCHMARK_MODULES, _LIBERO_ROOT

    libero_root = os.path.abspath(libero_root)
    libero_pkg_root = os.path.join(libero_root, "libero", "libero")
    config_dir = os.path.join(libero_root, ".eval_config")
    os.makedirs(config_dir, exist_ok=True)

    config = {
        "benchmark_root": libero_pkg_root,
        "bddl_files": os.path.join(libero_pkg_root, "bddl_files"),
        "init_states": os.path.join(libero_pkg_root, "init_files"),
        "datasets": os.path.join(libero_pkg_root, "..", "datasets"),
        "assets": os.path.join(libero_pkg_root, "assets"),
    }
    config_file = os.path.join(config_dir, "config.yaml")
    with open(config_file, "w", encoding="utf-8") as handle:
        yaml.dump(config, handle)

    os.environ["LIBERO_CONFIG_PATH"] = config_dir
    if libero_root not in sys.path:
        sys.path.insert(0, libero_root)

    if _LIBERO_ROOT != libero_root:
        _BENCHMARK_MODULES = None
        _LIBERO_ROOT = libero_root
        for module_name in list(sys.modules):
            if module_name == "libero" or module_name.startswith("libero."):
                del sys.modules[module_name]

    if _BENCHMARK_MODULES is None:
        from libero.libero import get_libero_path
        from libero.libero.benchmark import get_benchmark
        from libero.libero.envs import DummyVectorEnv, OffScreenRenderEnv

        _BENCHMARK_MODULES = {
            "get_libero_path": get_libero_path,
            "get_benchmark": get_benchmark,
            "DummyVectorEnv": DummyVectorEnv,
            "OffScreenRenderEnv": OffScreenRenderEnv,
        }

    return libero_root


def _require_modules():
    if _BENCHMARK_MODULES is None:
        raise RuntimeError("Call prepare_libero_runtime() before using LIBERO helpers.")
    return _BENCHMARK_MODULES


def build_benchmark(benchmark_name: str, task_order_index: int):
    mods = _require_modules()
    return mods["get_benchmark"](benchmark_name)(task_order_index)


def build_task_bddl_path(task) -> str:
    mods = _require_modules()
    return os.path.join(
        mods["get_libero_path"]("bddl_files"),
        task.problem_folder,
        task.bddl_file,
    )


def load_init_states(task, benchmark=None, task_id: Optional[int] = None) -> torch.Tensor:
    if benchmark is not None and task_id is not None and hasattr(
        benchmark, "get_task_init_states"
    ):
        return benchmark.get_task_init_states(task_id)

    mods = _require_modules()
    init_states_path = os.path.join(
        mods["get_libero_path"]("init_states"),
        task.problem_folder,
        task.init_states_file,
    )
    return torch.load(init_states_path)


def create_vectorized_env(
    task,
    img_h: int,
    img_w: int,
    num_envs: int,
    controller: str = "OSC_POSE",
    controller_config_updates: Dict | None = None,
):
    mods = _require_modules()
    env_args = {
        "bddl_file_name": build_task_bddl_path(task),
        "camera_heights": img_h,
        "camera_widths": img_w,
        "camera_names": ["agentview", "robot0_eye_in_hand"],
        "camera_depths": True,
        "controller": controller,
        "controller_config_updates": controller_config_updates or {},
    }
    return mods["DummyVectorEnv"](
        [lambda: mods["OffScreenRenderEnv"](**env_args) for _ in range(num_envs)]
    )


def select_task_ids(
    benchmark,
    all_tasks: bool,
    task_id: int,
    task_start: int,
    task_end: int,
) -> Sequence[int]:
    if all_tasks or task_id < 0:
        stop = benchmark.n_tasks if task_end < 0 else min(task_end, benchmark.n_tasks)
        start = max(0, task_start)
        if start >= stop:
            raise ValueError(
                f"Invalid task range [{start}, {stop}) for benchmark with "
                f"{benchmark.n_tasks} tasks."
            )
        return list(range(start, stop))
    if task_id < 0 or task_id >= benchmark.n_tasks:
        raise ValueError(
            f"task_id={task_id} is out of range for benchmark with "
            f"{benchmark.n_tasks} tasks."
        )
    return [task_id]


def load_plus_task_metadata(libero_root: str) -> Dict[str, Dict[str, str]]:
    metadata_path = (
        Path(libero_root)
        / "libero"
        / "libero"
        / "benchmark"
        / "task_classification.json"
    )
    if not metadata_path.exists():
        return {}

    with open(metadata_path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)

    task_meta: Dict[str, Dict[str, str]] = {}
    for suite_name, entries in raw.items():
        for entry in entries:
            task_meta[f"{suite_name}/{entry['name']}"] = {
                "category": entry.get("category", "Unknown"),
                "difficulty_level": str(entry.get("difficulty_level", "")),
            }
    return task_meta


def aggregate_plus_categories(
    benchmark_name: str,
    per_task_results: Dict[str, Dict],
    libero_root: str,
) -> Dict[str, float]:
    metadata = load_plus_task_metadata(libero_root)
    if not metadata:
        return {}

    category_success: Dict[str, list] = {}
    for task_key, payload in per_task_results.items():
        task_name = payload.get("task_name")
        lookup_key = f"{benchmark_name}/{task_name}"
        category = metadata.get(lookup_key, {}).get("category", "Unknown")
        category_success.setdefault(category, []).append(float(payload["success_rate"]))

    return {
        category: float(sum(values) / len(values))
        for category, values in sorted(category_success.items())
    }


def save_results(output_file: str, payload: Dict):
    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
