from copy import deepcopy
import os
import random
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
from torch.utils.data.distributed import DistributedSampler
from torch import nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm
import wandb

from modeling.encoder.text import fetch_tokenizers
from ..common_utils import count_parameters
from ..depth2cloud import fetch_depth2cloud
from ..data_preprocessors import fetch_data_preprocessor
from ..ema import EMA
from ..schedulers import fetch_scheduler
from .utils import compute_metrics
from .. import pytorch3d_transforms as pytorch3d_transforms

class BaseTrainTester:
    """Train/test a trajectory optimization algorithm."""

    def __init__(self, args, dataset_cls, model_cls):
        """Initialize."""
        self.args = args
        self.dataset_cls = dataset_cls
        self.model_cls = model_cls

        self.preprocessor = fetch_data_preprocessor(self.args.dataset)(
            self.args.keypose_only,
            self.args.num_history,
            custom_imsize=self.args.custom_img_size,
            depth2cloud=fetch_depth2cloud(self.args.dataset)
        )

        if dist.get_rank() == 0 and not self.args.eval_only:
            self.writer = SummaryWriter(log_dir=args.log_dir)
            wandb.init(
                project="3d-vla",
                name=str(args.run_log_dir) if hasattr(args, 'run_log_dir') else None,
                dir=str(args.log_dir),
                resume="allow"
            )

    def get_datasets(self):
        """Initialize datasets."""
        # Initialize datasets with arguments
        train_dataset = self.dataset_cls(
            root=self.args.train_data_dir,
            instructions=self.args.train_instructions,
            relative_action=self.args.relative_action,
            mem_limit=self.args.memory_limit,
            chunk_size=self.args.chunk_size
        )
        val_dataset = self.dataset_cls(
            root=self.args.eval_data_dir,
            instructions=self.args.val_instructions,
            copies=1,
            relative_action=self.args.relative_action,
            mem_limit=0.1,
            chunk_size=self.args.chunk_size
        )
        return train_dataset, val_dataset

    def get_loaders(self):
        """Initialize data loaders."""
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        # Datasets
        train_dataset, val_dataset = self.get_datasets()
        collate_fn = base_collate_fn
        if self.args.pre_tokenize and self.args.backbone == "qwen3vl":
            collate_fn = partial(
                qwen_tokenize_collate_fn,
                model_name_or_path=os.environ.get("QWEN3VL_MODEL_PATH"),
            )
        # Samplers and loaders
        g = torch.Generator()
        g.manual_seed(0)
        train_sampler = DistributedSampler(train_dataset, drop_last=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size // self.args.chunk_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            worker_init_fn=seed_worker,
            collate_fn=collate_fn,
            pin_memory=True,
            sampler=train_sampler,
            drop_last=True,
            generator=g,
            prefetch_factor=4,
            persistent_workers=True
        )
        # No sampler for val!
        if dist.get_rank() == 0:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.args.batch_size_val // self.args.chunk_size,
                shuffle=False,
                num_workers=self.args.num_workers,
                collate_fn=collate_fn,
                pin_memory=True,
                sampler=None,
                drop_last=False,
                prefetch_factor=4,
                persistent_workers=True
            )
        else:
            val_loader = None
        return train_loader, val_loader, train_sampler

    def get_model(self):
        """Initialize the model."""
        # Initialize model with arguments
        _model = self.model_cls(
            backbone=self.args.backbone,
            finetune_backbone=self.args.finetune_backbone,
            finetune_text_encoder=self.args.finetune_text_encoder,
            num_vis_instr_attn_layers=self.args.num_vis_instr_attn_layers,
            fps_subsampling_factor=self.args.fps_subsampling_factor,
            embedding_dim=self.args.embedding_dim,
            num_attn_heads=self.args.num_attn_heads,
            nhist=self.args.num_history,
            nhand=2 if self.args.bimanual else 1,
            num_shared_attn_layers=self.args.num_shared_attn_layers,
            relative=self.args.relative_action,
            rotation_format=self.args.rotation_format,
            denoise_timesteps=self.args.denoise_timesteps,
            denoise_model=self.args.denoise_model,
            lv2_batch_size=self.args.lv2_batch_size
        )

        # Print basic modules' parameters
        if dist.get_rank() == 0:
            count_parameters(_model)

        # Useful for some models to ensure parameters are contiguous
        for name, param in _model.named_parameters():
            if param.requires_grad and param.ndim > 1 and not param.is_contiguous():
                print(f"Fixing layout for: {name}")
                param.data = param.contiguous()

        return _model

    @torch.no_grad()
    def get_workspace_normalizer(self, ndims=3):
        print("Computing workspace normalizer...")

        # Initialize datasets with arguments
        train_dataset = self.dataset_cls(
            root=self.args.train_data_dir,
            instructions=self.args.train_instructions,
            copies=1,
            relative_action=self.args.relative_action,
            mem_limit=0.1,
            actions_only=True,
            chunk_size=self.args.chunk_size
        )

        data_loader = DataLoader(
            train_dataset,
            batch_size=max(self.args.batch_size, 64) // self.args.chunk_size,
            collate_fn=actions_collate_fn,
            shuffle=False,
            num_workers=self.args.num_workers
        )

        # Loop and compute action min-max
        min_, max_ = torch.ones(ndims) * 10000, -torch.ones(ndims) * 10000
        for sample in tqdm(data_loader):
            action = sample["action"][..., :ndims].reshape([-1, ndims])
            min_ = torch.min(min_, action.min(0).values)
            max_ = torch.max(max_, action.max(0).values)

        min_ = min_ - self.args.workspace_normalizer_buffer
        max_ = max_ + self.args.workspace_normalizer_buffer

        return nn.Parameter(torch.stack([min_, max_]), requires_grad=False)

    def get_optimizer(self, model):
        """Initialize optimizer."""
        optimizer_grouped_parameters = [
            {"params": [], "weight_decay": 0.0, "lr": self.args.lr},
            {"params": [], "weight_decay": self.args.wd, "lr": self.args.lr}
        ]
        if self.args.finetune_backbone:
            optimizer_grouped_parameters.append({
                "params": [], "weight_decay": self.args.wd,
                "lr": self.args.backbone_lr
            })

        # Collect names of all norm parameters
        norm_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.LayerNorm,
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LocalResponseNorm,
            torch.nn.RMSNorm
        )
        norm_param_names = set()
        for module_name, module in model.named_modules():
            if isinstance(module, norm_types):
                for param_name, _ in module.named_parameters(recurse=False):
                    norm_param_names.add(f"{module_name}.{param_name}")

        # Now split parameters based on name
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name in norm_param_names or name.endswith(".bias"):
                optimizer_grouped_parameters[0]["params"].append(param)
            elif self.args.finetune_backbone and 'backbone' in name:
                optimizer_grouped_parameters[2]["params"].append(param)
            else:
                optimizer_grouped_parameters[1]["params"].append(param)
        optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            betas=(0.9, 0.95)
        )
        return optimizer

    def main(self):
        """Run main training/testing pipeline."""
        # Get loaders
        train_loader, val_loader, train_sampler = self.get_loaders()

        # Get model
        model = self.get_model()
        self.tokenizer = fetch_tokenizers(self.args.backbone)
        if self.args.finetune or not os.path.exists(self.args.checkpoint):  # NOTE: for finetune we need normalization for current dataset
            normalizer = self.get_workspace_normalizer()  # calculate workspace normalizer from loaded dataset
            model.workspace_normalizer.copy_(normalizer)
            dist.barrier(device_ids=[torch.cuda.current_device()])

        # Get optimizer
        optimizer = self.get_optimizer(model)
        lr_scheduler = fetch_scheduler(
            self.args.lr_scheduler, optimizer, self.args.train_iters
        )
        scaler = torch.GradScaler()

        # Move model to devices
        if torch.cuda.is_available():
            model = model.cuda()
        # make sure to compile before DDP!
        if self.args.use_compile:
            model.compute_loss = torch.compile(model.compute_loss, fullgraph=True)
        model = DistributedDataParallel(
            model, device_ids=[self.args.local_rank],
            broadcast_buffers=False, find_unused_parameters=True
        )

        # Initialize EMA copy
        ema_model = deepcopy(model)
        self.ema = EMA()

        # Check for a checkpoint
        start_iter, best_loss = 0, None
        if self.args.checkpoint:
            start_iter, best_loss = self.load_checkpoint(model, ema_model, optimizer)
            if self.args.finetune:  # if finetune, we must use workspace normalizer for current dataset and copy to model
                model.module.workspace_normalizer.copy_(normalizer)
                if self.args.use_ema:
                    ema_model.module.workspace_normalizer.copy_(normalizer)
        print(model.module.workspace_normalizer)

        # Eval only
        if self.args.eval_only:
            if dist.get_rank() == 0:
                print("Test evaluation.......")
                model.eval()
                self.evaluate_nsteps(
                    ema_model if self.args.use_ema else model,
                    val_loader, step_id=-1,
                    val_iters=-1
                )
            dist.barrier(device_ids=[torch.cuda.current_device()])
            return ema_model if self.args.use_ema else model

        # Step the lr scheduler to the current step
        for _ in range(start_iter):
            lr_scheduler.step()

        # Step the sampler to the currect "epoch"
        samples_per_epoch = len(train_loader)
        epoch = start_iter // samples_per_epoch + 1
        train_sampler.set_epoch(epoch)  # ensures new batches are sampled

        # Training loop
        model.train()
        iter_loader = iter(train_loader)
        for step_id in trange(start_iter, self.args.train_iters):
            try:
                sample = next(iter_loader)
            except StopIteration:
                # when the iterator is exhausted, we need to reset it and increment the epoch
                # if dist.get_rank() == 0:  # NOTE: comment epoch evaluation as overfitting experiment too slow
                #     model.eval()
                #     self.evaluate_nsteps(
                #         ema_model if self.args.use_ema else model,
                #         train_loader, step_id, 
                #         val_iters=10,    
                #         split='train_epoch',
                #         epoch=epoch
                #     )
                #     model.train()
                epoch += 1
                train_sampler.set_epoch(epoch)
                iter_loader = iter(train_loader)
                sample = next(iter_loader)

            self.train_one_step(model, optimizer, scaler, lr_scheduler, sample)
            self.ema.step(model, ema_model, self.args.use_ema, step_id)

            if (step_id + 1) % self.args.val_freq == 0 and dist.get_rank() == 0:
                print("Train evaluation.......")
                model.eval()
                self.evaluate_nsteps(
                    ema_model if self.args.use_ema else model,
                    train_loader, step_id,
                    val_iters=10,
                    split='train'
                )
                if (step_id + 1) % 5000 == 0:  # frequently evaluate on val and save ckpt
                    print("Test evaluation.......")
                    new_loss = self.evaluate_nsteps(
                        ema_model.module if self.args.use_ema else model.module,
                        val_loader, step_id,
                        val_iters=500  # was 1250, decrease to prevent timeout on other gpus
                    )
                    # save model
                    best_loss = self.save_checkpoint(
                        model, ema_model, optimizer, step_id,
                        new_loss, best_loss
                    )
                model.train()
            dist.barrier(device_ids=[torch.cuda.current_device()])

        return ema_model if self.args.use_ema else model

    @torch.no_grad()
    def prepare_batch(self, sample, augment=False):
        pass  # implement in children

    def _model_forward(self, model, sample, training=True):
        action, action_mask, rgbs, rgb2d, pcds, instr, prop = self.prepare_batch(
            sample, augment=training
        )
        instr_mask = None
        if self.args.pre_tokenize:
            if "instr_input_ids" in sample and "instr_attention_mask" in sample:
                instr = sample["instr_input_ids"].cuda(non_blocking=True)
                instr_mask = sample["instr_attention_mask"].cuda(non_blocking=True)
            else:
                tok_out = self.tokenizer(instr)
                if isinstance(tok_out, torch.Tensor):
                    instr = tok_out.cuda(non_blocking=True)
                else:
                    instr = tok_out.input_ids.cuda(non_blocking=True)
                    instr_mask = tok_out.attention_mask.cuda(non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(
                action, action_mask, rgbs, rgb2d, pcds, instr, prop,
                run_inference=not training,
                instr_mask=instr_mask,
            )
        return out  # loss if training, else action

    def train_one_step(self, model, optimizer, scaler, lr_scheduler, sample):
        """Run a single training step."""
        optimizer.zero_grad()

        # Forward pass
        loss = self._model_forward(model, sample)

        # Backward pass
        scaler.scale(loss).backward()

        # Clip gradients
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)

        # Update
        scaler.step(optimizer)
        scaler.update()

        # Step the lr scheduler
        lr_scheduler.step()

    @torch.no_grad()
    def _log_geniesim_sample_rgb(self, sample, step_id):
        """Log all batch RGB images (single camera) to wandb per train step."""
        if dist.get_rank() != 0:
            return
        if "rgb" not in sample or sample["rgb"] is None:
            return

        rgb = sample["rgb"]
        if not torch.is_tensor(rgb) or rgb.ndim != 5 or rgb.size(0) == 0:
            return

        # rgb shape: (B, ncam, 3, H, W), GenieSim uses camera 0 only.
        rgb_batch = rgb[:, 0].detach().cpu()
        wandb_images = []
        for i in range(rgb_batch.size(0)):
            rgb_img = rgb_batch[i].permute(1, 2, 0).contiguous()
            if rgb_img.dtype != torch.uint8:
                rgb_img = (rgb_img.clamp(0, 1) * 255).to(torch.uint8)
            wandb_images.append(wandb.Image(rgb_img.numpy(), caption=f"batch_idx_{i}"))

        wandb.log(
            {"train/geniesim_batch_rgb": wandb_images},
            step=step_id
        )

    def _plot_geniesim_action_trajs(
        self, pcd_sample, rgb_sample, abs_pred_action_sample, abs_gt_action_sample
        ):
        """Create one figure with point cloud and GT/pred hand trajectories."""
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        pcd0 = pcd_sample[0].detach().float().cpu().permute(1, 2, 0).reshape(-1, 3).numpy()
        rgb0 = rgb_sample[0].detach().float().cpu().permute(1, 2, 0).reshape(-1, 3).numpy()
        if rgb0.max() > 1.0:
            rgb0 = rgb0 / 255.0
        valid = np.isfinite(pcd0).all(axis=-1)
        valid &= (np.linalg.norm(pcd0, axis=-1) > 1e-8)
        points = pcd0[valid]
        colors = np.clip(rgb0[valid], 0.0, 1.0)
        if points.shape[0] > 0:
            num_vis = min(points.shape[0], 50000)
            idx = np.random.choice(points.shape[0], num_vis, replace=False)
            ax.scatter(
                points[idx, 0], points[idx, 1], points[idx, 2],
                c=colors[idx], s=0.5, alpha=0.35
            )

        pred_traj = abs_pred_action_sample.detach().float().cpu().numpy()
        gt_traj = abs_gt_action_sample.detach().float().cpu().numpy()
        nhands = pred_traj.shape[1]

        gt_left_xyz = gt_traj[:, 0, :3]
        pred_left_xyz = pred_traj[:, 0, :3]
        ax.plot(
            gt_left_xyz[:, 0], gt_left_xyz[:, 1], gt_left_xyz[:, 2],
            color='green', linewidth=2, linestyle=':', label='GT Left'
        )
        ax.plot(
            pred_left_xyz[:, 0], pred_left_xyz[:, 1], pred_left_xyz[:, 2],
            color='red', linewidth=2, linestyle=':', label='Pred Left'
        )
        ax.scatter(gt_left_xyz[0, 0], gt_left_xyz[0, 1], gt_left_xyz[0, 2], color='green', s=80)
        ax.scatter(pred_left_xyz[0, 0], pred_left_xyz[0, 1], pred_left_xyz[0, 2], color='red', s=80)
        if nhands > 1:
            gt_right_xyz = gt_traj[:, 1, :3]
            pred_right_xyz = pred_traj[:, 1, :3]
            ax.plot(
                gt_right_xyz[:, 0], gt_right_xyz[:, 1], gt_right_xyz[:, 2],
                color='green', linewidth=2, linestyle='-', label='GT Right'
            )
            ax.plot(
                pred_right_xyz[:, 0], pred_right_xyz[:, 1], pred_right_xyz[:, 2],
                color='red', linewidth=2, linestyle='-', label='Pred Right'
            )
            ax.scatter(gt_right_xyz[0, 0], gt_right_xyz[0, 1], gt_right_xyz[0, 2], color='green', s=80)
            ax.scatter(pred_right_xyz[0, 0], pred_right_xyz[0, 1], pred_right_xyz[0, 2], color='red', s=80)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('GenieSim inference: point cloud + GT/Pred trajectories')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_box_aspect([1, 1, 1])
        fig.tight_layout()
        return fig

    @torch.no_grad()
    def _log_geniesim_infer_visualizations(self, sample, pred_action, gt_action, step_id, split):
        """
        During inference, log one figure per data sample in the batch:
        point cloud + predicted two-hand trajectories.
        """
        if (
            dist.get_rank() != 0
            or wandb.run is None
            or step_id < 0
        ):
            return
        if "proprioception" not in sample or sample["proprioception"] is None:
            return

        _, _, rgbs, _, pcds, _, _ = self.prepare_batch(sample, augment=False)
        if rgbs is None or pcds is None:
            return

        pred_action_vis = pred_action
        gt_action_vis = gt_action
        if self.args.relative_action:
            proprio = sample["proprioception"].cuda(non_blocking=True)[:, [-1]]
            pred_action_vis = relative_to_absolute(
                pred_action_vis, proprio, 'wxyz'
            )
            gt_action_vis = relative_to_absolute(
                gt_action_vis, proprio, 'wxyz'
            )

        wandb_images = []
        batch_size = pred_action_vis.shape[0]
        for b in range(batch_size):
            fig = self._plot_geniesim_action_trajs(
                pcds[b], rgbs[b], pred_action_vis[b], gt_action_vis[b]
            )
            wandb_images.append(wandb.Image(fig, caption=f"sample_{b}"))
            plt.close(fig)

        if wandb_images:
            wandb.log({f"{split}/geniesim_infer_pcd_action_trajs": wandb_images}, step=step_id)

    @torch.inference_mode()
    def evaluate_nsteps(self, model, loader, step_id, val_iters, split='val', epoch=None):
        """Run a given number of evaluation steps."""
        values = {}
        device = next(model.parameters()).device
        model.eval()

        for i, sample in tqdm(enumerate(loader)):
            if i == val_iters:
                break

            pred_action = self._model_forward(model, sample, training=False)
            gt_action = sample["action"].cuda(non_blocking=True)
            #self._log_geniesim_sample_rgb(sample, step_id)  # for geniesim3.0 debug
            #self._log_geniesim_infer_visualizations(sample, pred_action, gt_action, step_id, split)  # for geniesim3.0 debug
            # NOTE: skip relative to absolute evaluation for speed
            # current_proprio = sample["proprioception"].cuda(non_blocking=True)[:, -1:, :, :]
            # if self.args.relative_action:
            #     if self.args.rotation_format == 'euler':  # calvin dataset
            #         pred_action = relative_to_absolute(
            #             pred_action[:, :, 0],
            #             sample["proprioception"].cuda(non_blocking=True)[:, :, 0],
            #             'euler'
            #         )
            #         gt_action = relative_to_absolute(
            #             gt_action[:, :, 0],
            #             sample["proprioception"].cuda(non_blocking=True)[:, :, 0],
            #             'euler'
            #         )
            #     else:  # geniesim3.0 dataset
            #         pred_action = relative_to_absolute(
            #             pred_action,  # (B, T, nhands, 8)
            #             current_proprio,  # (B, 1, nhands, 8)
            #             'wxyz' if self.args.rotation_format == 'quat_wxyz' else 'xyzw'
            #         )
            #         gt_action = relative_to_absolute(
            #             gt_action,  # (B, T, nhands, 8)
            #             current_proprio,  # (B, 1, nhands, 8)
            #             'wxyz' if self.args.rotation_format == 'quat_wxyz' else 'xyzw'
            #         )
            losses = compute_metrics(pred_action, gt_action, nhand=2 if self.args.bimanual else 1)

            # Gather global statistics
            for n, l in losses.items():
                key = f"{split}-losses/mean/{n}"
                if key not in values:
                    values[key] = torch.Tensor([]).to(device)
                values[key] = torch.cat([values[key], l.unsqueeze(0)])

        # Log all statistics
        values = {k: v.mean().item() for k, v in values.items()}
        if dist.get_rank() == 0:
            if step_id > -1:
                for key, val in values.items():
                    self.writer.add_scalar(key, val, step_id)
                if epoch is not None:
                    values["epoch"] = epoch
                wandb.log(values, step=step_id)

            # Also log to terminal
            print(f"Step {step_id}:")
            for key, value in values.items():
                print(f"{key}: {value:.03f}")

        return values[f'{split}-losses/mean/traj_pos_acc_001']  # NOTE: was negative, seems wrong

    def load_checkpoint(self, model, ema_model, optimizer):
        """Load from checkpoint."""
        print("=> trying checkpoint '{}'".format(self.args.checkpoint))
        if not os.path.exists(self.args.checkpoint):
            print('Warning: checkpoint was not found, starting from scratch')
            print('The main process will compute workspace bounds')
            return 0, None

        model_dict = torch.load(
            self.args.checkpoint,
            map_location="cpu",
            weights_only=True
        )
        # Load weights flexibly
        msn, unxpct = model.load_state_dict(model_dict["weight"], strict=False)
        if msn:
            print(f"Missing keys (not found in checkpoint): {len(msn)}")
            print(msn)
        if unxpct:
            print(f"Unexpected keys (ignored): {len(unxpct)}")
            print(unxpct)
        if not msn and not unxpct:
            print("All keys matched successfully!")
        # EMA weights
        if model_dict.get("ema_weight") is not None:
            ema_model.load_state_dict(model_dict["ema_weight"], strict=True)
        if self.args.finetune:  # NOTE: for finetune we only load weights but reset iter and best_loss
            print("=> [FINETUNE MODE] Resetting iter, optimizer, and best_loss.")
            start_iter = 0
            best_loss = None
        else:  # resume
            start_iter = model_dict.get("iter", 0)
            best_loss = model_dict.get("best_loss", None)
            if 'optimizer' in model_dict and not self.args.eval_only:
                optimizer.load_state_dict(model_dict["optimizer"])

        print("=> loaded successfully '{}' (step {})".format(
            self.args.checkpoint, model_dict.get("iter", 0)
        ))
        del model_dict
        torch.cuda.empty_cache()
        return start_iter, best_loss

    def save_checkpoint(self, model, ema_model, optimizer,
                        step_id, new_loss, best_loss):
        """Save checkpoint if requested."""
        model_state = model.state_dict()
        ema_state = ema_model.state_dict() if self.args.use_ema else None

        # Best checkpoint
        if best_loss is None or new_loss <= best_loss:
            best_loss = new_loss
            torch.save({
                "weight": model_state,
                "ema_weight": ema_state,
                "iter": step_id + 1,
                "best_loss": best_loss
            }, self.args.log_dir / "best.pth")

        # Last checkpoint (always saved)
        torch.save({
            "weight": model_state,
            "ema_weight": ema_state,
            "optimizer": optimizer.state_dict(),
            "iter": step_id + 1,
            "best_loss": best_loss
        }, self.args.log_dir / "last.pth")

        # Save intermediate checkpoints
        if (step_id + 1) % self.args.interm_ckpt_freq == 0:
            torch.save({
                "weight": model_state,
                "ema_weight": ema_state,
                "iter": step_id + 1,
                "best_loss": best_loss
            }, self.args.log_dir / f"interm{step_id + 1}.pth")

        return best_loss


def base_collate_fn(batch):
    """Custom collate_fn, measured to be faster than default."""
    _dict = {}

    # Values for these come as lists
    list_keys = ["task", "instr"]
    for key in list_keys:
        if key not in batch[0].keys():
            continue
        _dict[key] = []
        for item in batch:
            _dict[key].extend(item[key])

    # Treat rest as tensors
    _dict.update({
        k_: (
            torch.cat([item[k_] for item in batch])
            if batch[0][k_] is not None else None
        )
        for k_ in batch[0].keys() if k_ not in list_keys
    })

    return _dict


_QWEN_COLLATE_TOKENIZER = {}

def _get_qwen_collate_tokenizer(model_name_or_path=None):
    key = model_name_or_path or os.environ.get("QWEN3VL_MODEL_PATH")
    if key not in _QWEN_COLLATE_TOKENIZER:
        from modeling.encoder.text.qwen3vl import Qwen3VLTokenizer
        _QWEN_COLLATE_TOKENIZER[key] = Qwen3VLTokenizer(model_name_or_path=key)
    return _QWEN_COLLATE_TOKENIZER[key]

def qwen_tokenize_collate_fn(batch, model_name_or_path=None):
    """Collate + CPU tokenization in dataloader workers for Qwen."""
    sample = base_collate_fn(batch)
    if "instr" not in sample:
        return sample
    tokenizer = _get_qwen_collate_tokenizer(model_name_or_path)
    tok_out = tokenizer(sample["instr"])
    sample["instr_input_ids"] = tok_out.input_ids
    sample["instr_attention_mask"] = tok_out.attention_mask
    return sample

def actions_collate_fn(batch):
    return {"action": torch.cat([item["action"] for item in batch])}

# geniesim3.0 dataset
def relative_to_absolute_rough(action, proprio, qform='wxyz'):
    # action (B, T, nhands, 8), proprio (B, 1, nhands, 8)
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
    elif qform == 'euler':
        orn = proprio[..., 3:6] + action[..., 3:6].cumsum(1)
        orn = (orn + torch.pi) % (2 * torch.pi) - torch.pi
    else:
        assert False
    return torch.cat([pos, orn, action[..., -1:]], -1)

def relative_to_absolute(action, proprio, qform='wxyz'):
    # wxyz/xyzw: action (B, T, nhands, 8), proprio (B, 1, nhands, 8)
    # euler: action (B, T, 8), proprio (B, 1, 7)
    pos = proprio[..., :3] + action[..., :3].cumsum(1)
    if qform in ['xyzw', 'wxyz']:
        rel_orn = action[..., 3:7]  # (B, T, 2, 4)
        curr_orn = proprio[..., 3:7]  # Initial state (B, 1, 2, 4)
        abs_orns = []
        num_steps = rel_orn.shape[1]
        for t in range(num_steps):
            q_rel = rel_orn[:, t:t+1] 
            if qform == 'xyzw':
                q_rel_wxyz = q_rel[..., [3, 0, 1, 2]]
                curr_orn_wxyz = curr_orn[..., [3, 0, 1, 2]]
                # absolute_t = relative_t * absolute_{t-1}
                next_orn_wxyz = pytorch3d_transforms.quaternion_multiply(
                    q_rel_wxyz, 
                    curr_orn_wxyz
                )
                next_orn = next_orn_wxyz[..., [1, 2, 3, 0]]
            elif qform == 'wxyz':
                next_orn = pytorch3d_transforms.quaternion_multiply(q_rel, curr_orn)
            abs_orns.append(next_orn)
            curr_orn = next_orn
        orn = torch.cat(abs_orns, dim=1)
    elif qform == 'euler':  # calvin dataset
        orn = proprio[..., 3:6] + action[..., 3:6].cumsum(1)
        orn = (orn + torch.pi) % (2 * torch.pi) - torch.pi
        return torch.cat([pos, orn, action[..., 6:]], -1)
    else:
        assert False
    return torch.cat([pos, orn, action[..., -1:]], -1)