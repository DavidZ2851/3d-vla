import sys
from pathlib import Path

from diffusers import FlowMatchEulerDiscreteScheduler


def _ensure_tracegen_on_path() -> Path:
    tracegen_root = Path(__file__).resolve().parents[2] / "TraceGen"
    if not tracegen_root.exists():
        raise FileNotFoundError(
            f"Expected sibling TraceGen repo at {tracegen_root}, but it was not found."
        )
    tracegen_root_str = str(tracegen_root)
    if tracegen_root_str not in sys.path:
        sys.path.insert(0, tracegen_root_str)
    return tracegen_root


def load_cogvideox_decoder_cls():
    _ensure_tracegen_on_path()
    from models.decoder.cogvideox_flow import CogVideoXDecoder_flow

    return CogVideoXDecoder_flow


def build_default_inference_scheduler(num_train_timesteps: int = 100):
    return FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=int(num_train_timesteps)
    )
