"""Serve a GeomVLA checkpoint (fine-tuned on MolmoSpaces zarrs) over openpi's websocket protocol.

MolmoSpaces' eval client (PI_Policy, action_mode="geomvla") connects with openpi_client's
WebsocketClientPolicy. Each request is one observation in the training layout of
data_processing/molmospaces_to_zarr.py, all in the robot base frame:

    rgb         (ncam, 224, 224, 3) uint8    [exterior, wrist]
    depth       (ncam, 224, 224)    float32  metric z-depth
    extrinsics  (ncam, 4, 4)        float32  camera-to-base, OpenCV axes
    intrinsics  (ncam, 3, 3)        float32  for the 224x224 images
    proprio     (7,)                float32  TCP xyz + axis-angle + binary openness (1 = open)
    prompt      str

and the reply is {"actions": (T, 7)}: per-step delta xyz, delta axis-angle (left-multiplied),
gripper (+1 close, -1 open) -- the raw, unnormalized relative actions the model was trained on.

    python online_evaluation_molmospaces/geomvla_server.py --checkpoint <run>/last.pth --port 8090
Architecture args default to scripts/molmospaces/finetune_libero_ckpt.sh's.
"""

import argparse
import asyncio
import functools
import inspect
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MOTION_EXPERT_USE_FLASH_ATTN_VARLEN", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import msgpack  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import websockets  # noqa: E402
import websockets.asyncio.server  # noqa: E402

from modeling.encoder.text import fetch_tokenizers  # noqa: E402
from modeling.policy import fetch_model_class  # noqa: E402
from utils.data_preprocessors import fetch_data_preprocessor  # noqa: E402
from utils.depth2cloud import fetch_depth2cloud  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
log = logging.getLogger("geomvla_server")


# --- openpi_client.msgpack_numpy wire format (copied: openpi-client pins numpy<2) -------------
def _pack_array(obj):
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


packb = functools.partial(msgpack.packb, default=_pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


# --- model --------------------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model_type", default="flower_actor_3d_motion_latent")
    ap.add_argument("--backbone", default="florence2")
    ap.add_argument("--embedding_dim", type=int, default=912)
    ap.add_argument("--num_attn_heads", type=int, default=16)
    ap.add_argument("--num_vis_instr_attn_layers", type=int, default=3)
    ap.add_argument("--num_shared_attn_layers", type=int, default=16)
    ap.add_argument("--num_history", type=int, default=1)
    ap.add_argument("--fps_subsampling_factor", type=int, default=1)
    ap.add_argument("--num_finetune_vision_layers", type=int, default=0)
    ap.add_argument("--custom_img_size", type=int, default=224)
    ap.add_argument("--rotation_format", default="axisangle")
    ap.add_argument("--denoise_timesteps", type=int, default=5)
    ap.add_argument("--denoise_model", default="rectified_flow")
    ap.add_argument("--prediction_len", type=int, default=10)
    ap.add_argument("--motion_expert_config", default=str(REPO / "motion_expert/configs/libero_florence2.yaml"))
    ap.add_argument("--motion_solver_steps", type=int, default=10)
    ap.add_argument("--motion_steps_to_run", type=int, default=1)
    return ap.parse_args()


class GeomVLAPolicy:
    def __init__(self, args):
        self.args = args
        cls = fetch_model_class(args.model_type)
        kwargs = dict(
            backbone=args.backbone,
            num_finetune_vision_layers=args.num_finetune_vision_layers,
            num_vis_instr_attn_layers=args.num_vis_instr_attn_layers,
            fps_subsampling_factor=args.fps_subsampling_factor,
            embedding_dim=args.embedding_dim,
            num_attn_heads=args.num_attn_heads,
            nhist=args.num_history,
            nhand=1,
            num_shared_attn_layers=args.num_shared_attn_layers,
            relative=True,
            rotation_format=args.rotation_format,
            denoise_timesteps=args.denoise_timesteps,
            denoise_model=args.denoise_model,
        )
        motion = dict(
            motion_expert_config=args.motion_expert_config,
            motion_expert_checkpoint=None,  # its fine-tuned weights are in the policy checkpoint
            motion_solver_steps=args.motion_solver_steps,
            motion_steps_to_run=args.motion_steps_to_run,
            motion_finetune=True,
            motion_scene_fusion=True,
            motion_action_fusion=False,
        )
        params = inspect.signature(cls.__init__).parameters
        kwargs.update({k: v for k, v in motion.items() if k in params})
        self.policy = cls(**kwargs)

        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        weights = state.get("ema_weight") or state.get("weight")
        weights = {k[7:] if k.startswith("module.") else k: v for k, v in weights.items()}
        missing, unexpected = self.policy.load_state_dict(weights, strict=False)
        log.info(f"loaded {args.checkpoint} (iter {state.get('iter')}); missing {len(missing)}, unexpected {len(unexpected)}")
        if missing:
            log.warning(f"missing keys (first 10): {missing[:10]}")
        self.policy._continuous_openness = False  # as LiberoTrainTester / the LIBERO eval wrapper
        self.policy._signed_binary_openness_output = True
        self.policy.eval().to(args.device)

        self.tokenizer = fetch_tokenizers(args.backbone)
        self.preprocessor = fetch_data_preprocessor("libero")(
            keypose_only=False,
            num_history=args.num_history,
            custom_imsize=args.custom_img_size,
            depth2cloud=fetch_depth2cloud("libero"),
        )

    @torch.no_grad()
    def infer(self, obs: dict) -> dict:
        rgb = torch.from_numpy(np.ascontiguousarray(obs["rgb"].transpose(0, 3, 1, 2)))[None]  # (1, ncam, 3, H, W)
        depth = torch.from_numpy(np.asarray(obs["depth"], np.float32))[None]
        extr = torch.from_numpy(np.asarray(obs["extrinsics"], np.float32))[None]
        intr = torch.from_numpy(np.asarray(obs["intrinsics"], np.float32))[None]
        proprio = torch.from_numpy(np.asarray(obs["proprio"], np.float32).reshape(1, 1, 1, -1))
        proprio = proprio.repeat(1, self.args.num_history, 1, 1)
        rgbs, pcds = self.preprocessor.process_obs(rgb, None, depth, extr, intr, augment=False)
        proprio = self.preprocessor.process_proprio(proprio)

        tok = self.tokenizer([str(obs["prompt"])])
        if isinstance(tok, torch.Tensor):
            instr, instr_mask = tok.cuda(), None
        else:
            instr, instr_mask = tok.input_ids.cuda(), tok.attention_mask.cuda()
        traj_mask = torch.zeros((1, self.args.prediction_len, 1), dtype=torch.bool, device=self.args.device)
        out = self.policy(None, traj_mask, rgbs, None, pcds, instr, proprio, run_inference=True, instr_mask=instr_mask)
        actions = out.view(1, self.args.prediction_len, 1, -1)[0, :, 0].float().cpu().numpy()
        return {"actions": actions.astype(np.float32)}


async def serve(policy: GeomVLAPolicy, host: str, port: int):
    async def handler(ws):
        log.info(f"connection from {ws.remote_address}")
        await ws.send(packb({"model": "geomvla"}))
        while True:
            try:
                obs = unpackb(await ws.recv())
                t = time.monotonic()
                out = policy.infer(obs)
                out["server_timing"] = {"infer_ms": (time.monotonic() - t) * 1000}
                await ws.send(packb(out))
            except websockets.ConnectionClosed:
                log.info(f"connection from {ws.remote_address} closed")
                break

    async with websockets.asyncio.server.serve(handler, host, port, compression=None, max_size=None):
        log.info(f"serving on {host}:{port}")
        await asyncio.Future()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    a = parse_args()
    asyncio.run(serve(GeomVLAPolicy(a), a.host, a.port))
