import os
import random
import torch
import torch.distributed as dist
import numpy as np
import open3d as o3d
import argparse
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

from utils.trainers.base import BaseTrainTester, actions_collate_fn
from utils.data_preprocessors.rlbench import RLBenchDataPreprocessor
from visualize_scene import RobotSceneVisualizer
from utils.depth2cloud.rlbench import RLBenchDepth2Cloud
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
        self.dataset = 'geniesim3.0'
        self.keypose_only = False
        self.num_history = 1
        self.custom_img_size = 256
        self.train_data_dir = '/data/group_data/katefgroup/VLA/zarr_datasets/geniesim3.0_zarr/train_rechunked4.zarr'
        self.eval_data_dir = '/data/group_data/katefgroup/VLA/zarr_datasets/geniesim3.0_zarr/val_rechunked4.zarr'
        self.train_instructions = '/data/group_data/katefgroup/VLA/zarr_datasets/geniesim3.0_zarr/instructions/geniesim3/train_instructions.json'
        self.val_instructions = '/data/group_data/katefgroup/VLA/zarr_datasets/geniesim3.0_zarr/instructions/geniesim3/val_instructions.json'
        self.relative_action = True # TODO: debug relative_to_absolute函数
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
        self.rotation_format = 'quat_wxyz' # Inferred from base_trainer snippet
        self.denoise_timesteps = 10
        self.denoise_model = 'rectified_flow'
        self.lv2_batch_size = 1
        self.workspace_normalizer_buffer = 0

# Helper function from base_trainer.py (recreated here for standalone usage)
def relative_to_absolute(action, proprio):
    """
    Convert relative action to absolute action.
    action: (B, T, nhands, 8)
    proprio: (B, 1, nhands, 8) or (B, 1, 7) depending on structure
    """
    # Assuming action is (B, T, nhand, 8) and proprio is (B, 1, nhand, 8)
    # 8 dims: x, y, z, rot_x, rot_y, rot_z, grip, ignore?
    
    # Check dimensions
    if len(proprio.shape) == 3 and len(action.shape) == 4:
         # Proprio might need unsqueeze if nhands dimension is missing in proprio
         if proprio.shape[-1] != action.shape[-1]:
             # Standard fallback if shapes mismatch drastically
             return action 

    pos = proprio[..., :3] + action[..., :3].cumsum(1)
    
    # Simple Euler addition with wrapping as seen in base_trainer snippet
    orn = proprio[..., 3:6] + action[..., 3:6].cumsum(1)
    orn = (orn + torch.pi) % (2 * torch.pi) - torch.pi
    
    return torch.cat([pos, orn, action[..., 6:]], -1)

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
        preprocessor = RLBenchDataPreprocessor(
            keypose_only=args.keypose_only,
            num_history=args.num_history,
            orig_imsize=256,
            custom_imsize=args.custom_img_size,
            depth2cloud=RLBenchDepth2Cloud((256, 256))
        )

        # Process Actions
        actions = preprocessor.process_actions(batch_sliced['action'])
        proprio = preprocessor.process_proprio(batch_sliced['proprioception'])
        # For now, just visualize absolute actions
        abs_actions = actions.cpu()

        # Process Observations (RGB/PCD)
        # process_obs expects (rgbs, rgb2d, depth, extrinsics, intrinsics, augment)
        rgbs_proc, pcds_proc = preprocessor.process_obs(
            batch_sliced['rgb'],
            None,  # rgb2d is None, batch_sliced.get('rgb2d', None),
            batch_sliced['depth'],
            batch_sliced['extrinsics'],
            batch_sliced['intrinsics'],
            augment=False
        )

        # 5. Visualize
        print("Generating visualization...")
        vis = RobotSceneVisualizer()

        # --- Visualizing Point Cloud ---
        # The preprocessor returns pcds_proc. 
        # Structure of pcds_proc is likely (B, ncam, 3, H, W) or (B, ncam, N_points, 3) 
        # depending on depth2cloud implementation which wasn't fully provided, 
        # but rlbench.py snippet implies it handles reshaping.
        
        # Let's assume pcds_proc is (1, ncam, 3, H, W) based on rlbench snippet `pcd_3d = obs...`
        # We flatten this to a list of points and colors.
        
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
        # abs_actions shape: (1, T, nhands, 8)
        traj = abs_actions[0].numpy() # (T, nhands, 8)
        T, nhands, _ = traj.shape

        def get_matrix_from_wxyz(vec_8d):
            # Indices 3:7 are w,x,y,z
            R_EE_TOOL = (
                R.from_euler('y', -90, degrees=True).as_matrix()
                @ R.from_euler('x', -90, degrees=True).as_matrix()
            )
            quat_wxyz = vec_8d[3:7]
            # Scipy needs x,y,z,w
            quat_xyzw = quat_wxyz[[1, 2, 3, 0]]
            return R.from_quat(quat_xyzw).as_matrix() @ R_EE_TOOL  # visualize_scene apply additional @R_EE_TOOL.T

        # Color Gradients
        # t=0 -> t=T: Fade colors
        for t in range(T):
            alpha = 1 - 0.7 * (t / float(T)) # Fade out

            if nhands > 0:
                pos = traj[t, 0, :3]
                rot_mat = get_matrix_from_wxyz(traj[t, 0])
                color = (0.7, 0.0, 0.7 * (t/T)) 
                vis.add_trajectory_gripper('left', pos, rot_mat, color, alpha)

            if nhands > 1:
                pos = traj[t, 1, :3]
                rot_mat = get_matrix_from_wxyz(traj[t, 1])
                color = (0.0, 0.7 * (t/T), 0.7) 
                vis.add_trajectory_gripper('right', pos, rot_mat, color, alpha)

        # Save visualization
        output_filename = f"vis_sample_{idx}.png"
        vis.visualize(filename=output_filename)
        print(f"Done! Saved to {output_filename}")
    finally:
        cleanup_ddp()

if __name__ == "__main__":
    from datasets.geniesim3 import GenieSim3Dataset as DatasetCls
    DATA_PATH ='/data/group_data/katefgroup/VLA/zarr_datasets/geniesim3.0_zarr/train_rechunked4.zarr'     
    run_visualization(DatasetCls, DATA_PATH)