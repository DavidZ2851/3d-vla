import os
import random
from tkinter import NONE
import torch
import torch.distributed as dist
import numpy as np
import open3d as o3d
import argparse
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

from utils.trainers.base import (
    BaseTrainTester,
    actions_collate_fn,
    relative_to_absolute as base_relative_to_absolute
)
from utils.data_preprocessors.libero import LiberoDataPreprocessor
from visualize_geniesim_scene import RobotSceneVisualizer
from utils.depth2cloud.libero import LiberoDepth2Cloud
from utils.pytorch3d_transforms import quaternion_to_matrix, matrix_to_quaternion

def setup_ddp():
    """Initializes a single-process DDP group to satisfy BaseTrainTester requirements."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Use 'gloo' as it works on both CPU and GPU and is easier for simple scripts
    if not dist.is_initialized():
        dist.init_process_group("gloo", rank=0, world_size=1)

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

# Define a mock arguments class to replicate what BaseTrainTester expects
class MockArgs:
    def __init__(self):
        # Default parameters based on typical training setups and snippets
        self.dataset = 'libero'
        self.keypose_only = False
        self.num_history = 1
        self.custom_img_size = None
        self.train_data_dir = '/data/group_data/katefgroup/VLA/zarr_datasets/libero_zarr/train_rechunked8.zarr'
        self.eval_data_dir = '/data/group_data/katefgroup/VLA/zarr_datasets/libero_zarr/val_rechunked8.zarr'
        self.train_instructions = 'instructions/libero/instructions.json'
        self.val_instructions = 'instructions/libero/instructions.json'
        self.relative_action = False
        self.memory_limit = 0 # No limit
        self.chunk_size = 4
        self.batch_size = 1
        self.batch_size_val = 1
        self.num_workers = 0
        self.log_dir = './logs'
        self.run_log_dir = 'debug_run'
        self.eval_only = True # Avoid initializing wandb/tensorboard
        self.backbone = 'florence2'
        self.finetune_backbone = False
        self.finetune_text_encoder = False
        self.num_vis_instr_attn_layers = 2
        self.fps_subsampling_factor = 1
        self.embedding_dim = 128
        self.num_attn_heads = 4
        self.bimanual = True
        self.num_shared_attn_layers = 2
        self.rotation_format = 'rotmatrix'
        self.denoise_timesteps = 5
        self.denoise_model = 'rectified_flow'
        self.lv2_batch_size = 1
        self.workspace_normalizer_buffer = 0

def run_visualization(dataset_cls, root_data_path):
    # 1. Setup
    setup_ddp()
    try:
        args = MockArgs()
        args.train_data_dir = root_data_path
        
        # Initialize the Trainer/Tester to get the standard setup logic
        # We pass a dummy model_cls because we won't instantiate the network, just the data pipeline
        trainer = BaseTrainTester(args, dataset_cls, model_cls=lambda **kwargs: None)

        # 2. Read Dataset
        print("Initializing dataset...")
        # Using the logic from trainer.get_datasets() but manually to get just train set
        dataset = dataset_cls(
            root=args.train_data_dir,
            instructions=args.train_instructions,
            relative_action=args.relative_action,
            mem_limit=args.memory_limit,
            chunk_size=args.chunk_size
        )
        
        print(f"Dataset length: {len(dataset)}")

        # 3. Random Sample
        idx = random.randint(0, len(dataset) - 1)
        print(f"Selecting random sample at index: {idx}")
        
        raw_sample = dataset[idx]
        
        # Manually collate to create Batch Size = 1
        # Convert numpy/others to torch and add batch dimension
        batch = {}
        for k, v in raw_sample.items():
            if isinstance(v, (np.ndarray, list)):
                if isinstance(v, list):
                    # Handle list of strings or similar
                    batch[k] = v # Keep as list
                else:
                    tensor_v = torch.from_numpy(v)
                    batch[k] = tensor_v.unsqueeze(0) # (1, ...)
            elif isinstance(v, torch.Tensor):
                batch[k] = v.unsqueeze(0)
            else:
                batch[k] = [v]

        chunk_dim_size = batch['action'].shape[1] # Should be 4
        print(f"Detected chunk size: {chunk_dim_size}")
        
        # Randomly select one chunk index (e.g., 0 to 3)
        chunk_idx = random.randint(0, chunk_dim_size - 1)
        print(f"Slicing batch at chunk index: {chunk_idx}")

        # Slice tensor fields
        batch_sliced = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                # We assume any tensor with dim 1 size == chunk_dim_size needs slicing
                # This protects against 1D tensors (like flags) that might not match
                if v.dim() >= 2 and v.shape[1] == chunk_dim_size:
                    batch_sliced[k] = v[:, chunk_idx] # Keeps dim 0 (batch), removes dim 1
                else:
                    batch_sliced[k] = v
            else:
                # Lists (like instructions) might or might not need slicing. 
                # If it's a list of length 1 (batch), we keep it.
                batch_sliced[k] = v

        print(f"Sliced Action shape: {batch_sliced['action'].shape}") # Expected: (1, 12, 2, 8)
        print(f"Sliced RGB shape: {batch_sliced['rgb'].shape}")       # Expected: (1, 1, 3, 256, 256)
        print(f"Sliced Depth shape: {batch_sliced['depth'].shape}")  

        # 4. Process Data using RLBench Processor
        print("Processing data...")
        preprocessor = LiberoDataPreprocessor(
            keypose_only=args.keypose_only,
            num_history=args.num_history,
            orig_imsize=128,
            custom_imsize=None,
            depth2cloud=LiberoDepth2Cloud((128, 128))
        )

        # Process Actions
        actions = preprocessor.process_actions(batch_sliced['action'])
        proprio = preprocessor.process_proprio(batch_sliced['proprioception'])
        if args.relative_action:
            # GenieSim stores quaternion in wxyz here.
            abs_actions = base_relative_to_absolute(
                actions, proprio[:, [-1]], qform='wxyz'
            ).cpu()
        else:
            abs_actions = actions.cpu()

        # Process Observations (RGB/PCD)
        # process_obs expects (rgbs, rgb2d, depth, extrinsics, intrinsics, augment)
        print('extrinsics', batch_sliced['extrinsics'])
        print('depth_min', batch_sliced['depth'][0, 0, :30].min(), 'depth_max', batch_sliced['depth'][0, 0, :30].max(), 'depth_min', batch_sliced['depth'][0, 0,-30:].mean(), 'depth_max', batch_sliced['depth'][0, 0, -30:].max())

        rgbs_proc, pcds_proc = preprocessor.process_obs(
            batch_sliced['rgb'],
            None,  # rgb2d is None, batch_sliced.get('rgb2d', None),
            batch_sliced['depth'],
            batch_sliced['extrinsics'],
            batch_sliced['intrinsics'],
            augment=False
        )
        print('pcds x min', pcds_proc[0, 0, 0].min(), 'pcds x max', pcds_proc[0, 0, 0].max(), 'pcds y min', pcds_proc[0, 0, 1].min(), 'pcds y max', pcds_proc[0, 0, 1].max(), 'pcds z min', pcds_proc[0, 0, 2].min(), 'pcds z max', pcds_proc[0, 0, 2].max())

        # 5. Visualize
        print("Generating visualization...")
        vis = RobotSceneVisualizer()

        # --- Visualizing Point Cloud ---
        # Extract RGB colors (normalized 0-1)
        rgb_t = rgbs_proc[0] # (ncam, 3, H, W)
        pcd_t = pcds_proc[0] # (ncam, 3, H, W)

        # Flatten
        ncam = rgb_t.shape[0]
        points_all = []
        colors_all = []

        for cam_i in range(ncam):
            # Permute to (H, W, 3)
            c_img = rgb_t[cam_i].permute(1, 2, 0).cpu().numpy()
            p_img = pcd_t[cam_i].permute(1, 2, 0).cpu().numpy()
            
            # Filter valid points (not 0,0,0) - assuming depth 0 maps to origin
            valid_mask = np.any(p_img != 0, axis=-1)
            
            pts = p_img[valid_mask]
            clrs = c_img[valid_mask]
            
            points_all.append(pts)
            colors_all.append(clrs)

        if points_all:
            points_all = np.concatenate(points_all, axis=0)
            colors_all = np.concatenate(colors_all, axis=0)

            # Add to visualizer manually
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_all.astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(colors_all.astype(np.float64)) # Already 0-1 float from preprocessor
            
            # Downsample if too heavy
            if len(pcd.points) > 100000:
                pcd = pcd.voxel_down_sample(voxel_size=0.01)
                
            vis.world_pcds.append(pcd)

        # --- Visualizing Trajectories ---
        # abs_actions is absolute action pose:
        #   pos(3) + rotmat_flat_rowmajor(9) + gripper(1) = 13
        # Shape here: (T, nhands, 13)
        traj = abs_actions[0].numpy() # (T, nhands, 13)
        T, nhands, _ = traj.shape

        def get_rotmat_from_abs_action(vec_13d: np.ndarray) -> np.ndarray:
            rot_flat = vec_13d[3:12]
            return rot_flat.reshape(3, 3)

        # Color Gradients
        # t=0 -> t=T: Fade colors
        for t in range(10):
            alpha = 1 - 0.7 * (t / float(T)) # Fade out

            if nhands > 0:
                pos = traj[t, 0, :3]
                rot_mat = get_rotmat_from_abs_action(traj[t, 0])
                color = (0.7, 0.0, 0.7 * (t/T)) 
                vis.add_trajectory_gripper('left', pos, rot_mat, color, alpha)

            if nhands > 1:
                pos = traj[t, 1, :3]
                rot_mat = get_rotmat_from_abs_action(traj[t, 1])
                color = (0.0, 0.7 * (t/T), 0.7) 
                vis.add_trajectory_gripper('right', pos, rot_mat, color, alpha)

        # Save visualization
        output_filename = f"vis_sample_{idx}.png"
        vis.visualize(filename=output_filename)
        
        # Save corresponding RGB frame (GenieSim uses camera 0).
        # batch_sliced['rgb']: (B, ncam, 3, H, W), uint8
        rgb_frame = batch_sliced['rgb'][0, 0].detach().cpu().permute(1, 2, 0).numpy()
        if rgb_frame.dtype != np.uint8:
            rgb_frame = np.clip(rgb_frame, 0, 255).astype(np.uint8)
        rgb_output_filename = f"vis_sample_{idx}_rgb.png"
        plt.imsave(rgb_output_filename, rgb_frame)

        print(f"Done! Saved to {output_filename} and {rgb_output_filename}")
    finally:
        cleanup_ddp()

if __name__ == "__main__":
    from datasets.libero import LIBERODataset as DatasetCls
    DATA_PATH ='/data/group_data/katefgroup/VLA/zarr_datasets/libero_zarr/train_rechunked8.zarr'     
    run_visualization(DatasetCls, DATA_PATH)