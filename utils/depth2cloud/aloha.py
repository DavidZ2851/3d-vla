"""
ALOHA depth -> world-frame point cloud (front + back Azure Kinects).

Intrinsics and extrinsics are static for the rig and live in the lerobot repo at
    /data/user_data/ziyinx/lerobot/lerobot/scripts/aloha_calibration/
read here via calibration_multiview.json (front + back). The intrinsics are
calibrated at the source resolution (720x1280); we rescale to whatever HxW the
depth map has at inference/train time (typically 224x398).

Extrinsics files are named ``T_world_from_camera_*.txt`` — i.e., they directly map
camera-frame points to world-frame points. world_from_cam = T (no inversion). Both
cameras land in the same world frame, so front+back clouds are directly fusible.

Backward-compat: ``deproject(depth)`` / ``__call__(depth)`` keep operating on the
FRONT camera (used by deploy_aloha_policy.py, which is front-only). The multi-camera
training path calls ``deproject_cam(depth, "front"|"back")``.
"""

from pathlib import Path

import json
import numpy as np
import torch


_DEFAULT_CALIB_JSON = Path(
    "/data/user_data/ziyinx/lerobot/lerobot/scripts/aloha_calibration/calibration_multiview.json"
)
SOURCE_HW = (720, 1280)

# calibration_multiview.json camera name -> short name used by the dataset/preprocessor
_CAM_NAME_MAP = {"cam_azure_kinect_front": "front", "cam_azure_kinect_back": "back"}


class AlohaDepth2Cloud:
    """Unproject ALOHA front/back kinect depth to a shared world frame."""

    def __init__(
        self,
        source_hw=SOURCE_HW,
        calib_json: Path = _DEFAULT_CALIB_JSON,
        front_cam: str = "front",
    ):
        calib_json = Path(calib_json)
        base = calib_json.parent.parent  # paths inside are relative to .../scripts/
        spec = json.loads(calib_json.read_text())

        self.source_hw = tuple(source_hw)
        self.front_cam = front_cam
        # short name -> {"K": (3,3), "T": (4,4) world_from_cam}
        self._calib: dict[str, dict[str, np.ndarray]] = {}
        for cam_name, entry in spec.items():
            short = _CAM_NAME_MAP.get(cam_name, cam_name)
            self._calib[short] = {
                "K": np.loadtxt(base / entry["intrinsics"]).astype(np.float32),
                "T": np.loadtxt(base / entry["extrinsics"]).astype(np.float32),
            }
        if front_cam not in self._calib:
            raise KeyError(f"front camera {front_cam!r} not in calibration {list(self._calib)}")

        self._uv_cache: dict = {}
        self._intrinsics_cache: dict = {}
        self._extr_cache: dict = {}

    @property
    def cameras(self) -> list[str]:
        return list(self._calib)

    @staticmethod
    def _uniform_uv1(h, w):
        u = np.tile(np.arange(w, dtype=np.float32), (h, 1))
        v = np.tile(np.arange(h, dtype=np.float32), (w, 1)).T
        ones = np.ones_like(u, dtype=np.float32)
        return np.stack((u, v, ones), axis=0)  # (3, H, W)

    def _rescaled_intrinsics(self, K_src: np.ndarray, target_hw):
        tgt_h, tgt_w = target_hw
        src_h, src_w = self.source_hw
        sx = float(tgt_w) / float(src_w)
        sy = float(tgt_h) / float(src_h)
        K = K_src.copy()
        K[0, 0] *= sx
        K[1, 1] *= sy
        K[0, 2] *= sx
        K[1, 2] *= sy
        return K

    def _get_uv1(self, h, w, device):
        key = (h, w, str(device))
        if key not in self._uv_cache:
            self._uv_cache[key] = torch.from_numpy(self._uniform_uv1(h, w)).to(device=device)
        return self._uv_cache[key]

    def _get_intrinsics(self, cam, h, w, device, dtype):
        key = (cam, h, w, str(device), str(dtype))
        if key not in self._intrinsics_cache:
            self._intrinsics_cache[key] = torch.from_numpy(
                self._rescaled_intrinsics(self._calib[cam]["K"], (h, w))
            ).to(device=device, dtype=dtype)
        return self._intrinsics_cache[key]

    def _get_world_from_cam(self, cam, device, dtype):
        key = (cam, str(device), str(dtype))
        if key not in self._extr_cache:
            self._extr_cache[key] = torch.from_numpy(self._calib[cam]["T"]).to(
                device=device, dtype=dtype
            )
        return self._extr_cache[key]

    def deproject_cam(self, depth, cam):
        """
        Args:
            depth: (B, H, W) float, meters.
            cam:   "front" or "back".
        Returns:
            (B, 3, H, W) world-frame point cloud.
        """
        depth = depth.float()
        b, h, w = depth.shape
        K = self._get_intrinsics(cam, h, w, depth.device, depth.dtype)
        Kinv = torch.linalg.inv(K)
        uv1 = self._get_uv1(h, w, depth.device).unsqueeze(0).expand(b, -1, -1, -1)  # (B,3,H,W)
        uvz = (uv1 * depth[:, None]).reshape(b, 3, -1)
        cam_xyz = Kinv @ uvz                                          # (B, 3, H*W)
        cam_xyz_h = torch.cat([cam_xyz, torch.ones_like(cam_xyz[:, :1])], dim=1)  # (B, 4, H*W)
        T = self._get_world_from_cam(cam, depth.device, depth.dtype).unsqueeze(0).expand(b, -1, -1)
        world_xyz = (T[:, :3] @ cam_xyz_h)                            # (B, 3, H*W)
        return world_xyz.reshape(b, 3, h, w)

    # ---- front-only backward-compat (deploy_aloha_policy.py) ----
    def deproject(self, depth):
        return self.deproject_cam(depth, self.front_cam)

    def __call__(self, depth_front):
        """depth_front shape (B, H, W) -> (B, 3, H, W) world PCD (front camera)."""
        return self.deproject(depth_front)
