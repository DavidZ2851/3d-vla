import numpy as np
import torch


class CALVINDepth2Cloud:

    def __init__(self, front_shape=(160, 160)):
        self.uvs = {}
        # Static camera (front)
        self.T_world_cam = torch.from_numpy(np.linalg.inv(np.asarray([
            0.7232445478439331, -0.030260592699050903, 0.6899287700653076, 0.0,
            0.5141233801841736, 0.6906105875968933, -0.5086592435836792, 0.0,
            -0.46107974648475647, 0.7225935459136963, 0.5150379538536072, 0.0,
            0.21526622772216797, -0.26317155361175537, -4.399168968200684, 1.0
        ]).reshape((4, 4)).T)).float().cuda(non_blocking=True)
        # Front camera: (height, width, fov) = (200, 200, 10)
        self.uvs['front'] = self._get_uv(
            200, 200,
            front_shape,  # assume center crop if different than (200, 200)
            200 / (2 * np.tan(np.deg2rad(10) / 2))  # h / (2 * tan(fov / 2))
        )

        # Gripper camera (wrist): (height, width, fov) = (84, 84, 75)
        self.uvs['wrist'] = self._get_uv(
            84, 84,
            (84, 84),
            84 / (2 * np.tan(np.deg2rad(75) / 2))
        )

    def _get_uv(self, h, w, img_size, foc):
        v, u = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        # Center crop
        h_offset = (h - img_size[0]) // 2
        w_offset = (w - img_size[1]) // 2
        v = v[h_offset:img_size[0]+h_offset, w_offset:img_size[1]+w_offset]
        u = u[h_offset:img_size[0]+h_offset, w_offset:img_size[1]+w_offset]
        # Move to device
        u, v = u.cuda(non_blocking=True), v.cuda(non_blocking=True)
        # Calculate 'instrinsics' matrix
        uv_mat = torch.stack((
            (u  - w // 2) / foc,
            (-v + h // 2) / foc,
            -torch.ones_like(u)
        ), dim=0)  # (3, H, W)
        return uv_mat.reshape(3, -1)  # (3, H*W)

    def deproject(self, depth, extrinsics, cam):
        """
        Args
            depth: (B, H, W)
            extrinsics: (B, 4, 4)
            cam: str, 'front' or 'wrist'

        Output
            point cloud: (B, 3, H, W)
        """
        b, h, w = depth.shape

        # Camera XYZ
        cam_pos = self.uvs[cam][None] * depth.reshape(b, 1, -1)  # B 3 HW

        # # Unproject to world coordinates
        cam_pos = torch.cat((cam_pos, torch.ones_like(cam_pos[:, :1])), dim=1)
        world_pos = torch.bmm(extrinsics[:, :3], cam_pos)  # B 3 HW

        # Reshape
        return world_pos.reshape(b, 3, h, w)

    def __call__(self, depth_front, depth_wrist, extrinsics_wrist):
        pcd_front = None
        if depth_front is not None:
            pcd_front = self.deproject(
                depth_front,
                self.T_world_cam[None].repeat(len(depth_front), 1, 1),
                'front'
            )
            self.T_world_cam[None].repeat(len(depth_front), 1, 1)  # B 4 4
        pcd_wrist = None
        if depth_wrist is not None:
            pcd_wrist = self.deproject(
                depth_wrist,
                extrinsics_wrist,
                'wrist'
            )
        return pcd_front, pcd_wrist
