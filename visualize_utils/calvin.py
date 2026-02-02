import os

import torch

from training.depth2cloud import fetch_depth2cloud
import utils.pytorch3d_transforms as pytorch3d_transforms
from visualize_utils.base import Visualizer as BaseVisualizer, visualize_actions_and_point_clouds


def relative_to_absolute(action, proprio, qform='wxyz'):
    # action (B, T, 8), proprio (B, 1, 7)
    pos = proprio[..., :3] + action[..., :3].cumsum(1)

    if qform == 'xyzw':
        # pytorch3d takes wxyz quaternion, the input is xyzw
        orn = pytorch3d_transforms.quaternion_multiply(
            action[..., [6, 3, 4, 5]],
            proprio[..., [6,3,4,5]]
        )[..., [1, 2, 3, 0]]
    elif qform == 'wxyz':
        orn = pytorch3d_transforms.quaternion_multiply(
            action[..., 3:7],
            proprio[..., 3:7]
        )
    else:
        assert False

    return torch.cat([pos, orn, action[..., 7:]], -1)


class Visualizer(BaseVisualizer):

    def plot_images_depths(self, t):
        from ipdb import set_trace; set_trace()
        super().plot_images_depths(
            t, cams=[0],
            img='rgb_front', depth='depth_front', save_path='im_depth_front/'
        )
        super().plot_images_depths(
            t, cams=[0],
            img='rgb_wrist', depth='depth_wrist', save_path='im_depth_wrist/'
        )

    def plot_aug_images(self, t):
        super().plot_aug_images(
            t, cams=[0],
            img='rgb_front', save_path='aug_front/'
        )

    def plot_point_cloud_grippers(self, t, cams=[0], img='rgb_front', depth='depth_front',
                                  curr=True, action=True, save_path='pcd/',
                                  dataset_name='calvin'):
        imgs = self.annos[img][t][cams]  # Nc 3 h w
        depths = self.annos[depth][t][cams]  # Nc h w
        # Get point cloud
        pc = self.d2c(
            torch.from_numpy(depths).cuda().float(),
            None, None
        )[0][0]  # Nc 3 h w
        # Grippers
        grippers = []
        if curr:
            prop = torch.from_numpy(self.annos['proprioception'][t][-1])  # (2, 8) or (8,)
            if len(prop.shape) > 1:
                grippers.extend([p for p in prop])
            else:
                grippers.append(prop)
        if action:
            traj = torch.from_numpy(self.annos['action'][t])  # (N, 2, 8) or (N, 8)
            if len(traj.shape) > 2:
                for i in range(traj.shape[1]):
                    grippers.extend([t_ for t_ in traj[:, i]])
            else:
                grippers.extend([t_ for t_ in traj])
        # Visualize
        os.makedirs(save_path, exist_ok=True)
        visualize_actions_and_point_clouds(
            pc[None],
            torch.from_numpy(imgs).float() / 255.0,
            grippers,
            dataset_name=dataset_name,
            savename=save_path + f'pcd_{t}.jpg'
        )

        ############## Repeat for wrist camera
        cams = [0]
        imgs = self.annos['rgb_wrist'][t][cams]  # Nc 3 h w
        depths = self.annos['depth_wrist'][t][cams]  # Nc h w
        # Get point cloud
        pc = self.d2c(
            None,
            torch.from_numpy(depths).cuda().float(),
            torch.from_numpy(self.annos['extrinsics_wrist'][t]).cuda().float()[None]
        )[1][0]  # Nc 3 h w
        # Grippers
        grippers = []
        if curr:
            prop = torch.from_numpy(self.annos['proprioception'][t][-1])  # (2, 8) or (8,)
            if len(prop.shape) > 1:
                grippers.extend([p for p in prop])
            else:
                grippers.append(prop)
        if action:
            traj = torch.from_numpy(self.annos['rel_action'][t])  # (N, 2, 8) or (N, 8)
            prop = torch.from_numpy(self.annos['proprioception'][t][-1])
            traj = relative_to_absolute(traj, prop[None])
            if len(traj.shape) > 2:
                for i in range(traj.shape[1]):
                    grippers.extend([t_ for t_ in traj[:, i]])
            else:
                grippers.extend([t_ for t_ in traj])
        # Visualize
        os.makedirs(save_path, exist_ok=True)
        visualize_actions_and_point_clouds(
            pc[None],
            torch.from_numpy(imgs).float() / 255.0,
            grippers,
            dataset_name=dataset_name,
            savename=save_path + f'wrist_pcd_{t}.jpg'
        )

    def show_language(self, t):
        t_ = int(self.annos['instr_id'][t])
        print(self._instructions[t_])


if __name__ == '__main__':
    zarr_path = '/data/user_data/ngkanats/zarr_datasets/CALVIN_keypose_zarr/val.zarr'
    depth2cloud = fetch_depth2cloud('calvin')
    use_meshcat = False
    im_size = 160
    vis = Visualizer(
        zarr_path, depth2cloud, use_meshcat, im_size,
        instruction_file='instructions/calvin/val_keypose_instructions.json'
    )
    for t in range(20):
        # vis.plot_images_depths(t)
        # vis.plot_aug_images(t)
        vis.plot_point_cloud_grippers(t)
        vis.show_language(t)
