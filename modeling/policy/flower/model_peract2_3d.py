import einops
import torch
from torch import nn
from torch.nn import functional as F

from ...utils.position_encodings import RotaryPositionEncoding3D

from .model_peract2 import FLOWERVLA, CLIPTransform
from .transformers import FlowBlock3D, stateless_norm


class FLOWERVLA3D(FLOWERVLA):

    def __init__(
        self,
        # Encoder arguments
        backbone="clip",
        output_level="res3",
        upsample=False,
        finetune_backbone=False,
        finetune_text_encoder=False,
        num_vis_instr_attn_layers=2,
        fps_subsampling_factor=5,
        # Encoder and decoder arguments
        embedding_dim=60,
        num_attn_heads=9,
        nhist=3,
        nhand=1,
        # Decoder arguments
        relative=False,
        rotation_format='quat_xyzw',
        # Denoising arguments
        denoise_timesteps=100,
        denoise_model="ddpm",
        lv2_batch_size=1,
        # VLM Configuration
        vlm_path: str = "microsoft/Florence-2-large",
        freeze_florence: bool = True,  # False,
        freeze_vision_tower: bool = False,
        vlm_prompt_style: str = "default",
        token_dropout: float = 0.1,
        
        # Model Structure
        multistep: int = 2,
        num_sampling_steps: int = 4,
        lowdim_obs_dim: int = 10,
        action_dim: int = 10,
        act_window_size: int = 2,
        
        # Model flags
        use_second_view: bool = True,
        second_view_key: str = 'image_wrist',
        action_type_adaln: bool = True,
        use_causal_attention: bool = True,
        use_cross_attn: bool = True,
        use_adaln_cond: bool = False,
        use_readout_token: bool = False,
        use_proprio: bool = False,
        return_act_chunk: bool = False,
        
        # DiT Configuration
        sampling_type: str = 'uniform',
        dit_dim: int = 1024,
        n_heads: int = 16,
        n_layers: int = 12,  # was 18 for calvin
        attn_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
        mlp_pdrop: float = 0.1,
        
        # RoPE Configuration
        use_rope: bool = True,
        use_nope: bool = False,
        query_seq_len: int = 100,
        rope_theta: float = 32.0
    ):
        super().__init__()
        # Initialize model flags and configurations
        self._init_flags(
            use_second_view=use_second_view,
            use_causal_attention=use_causal_attention,
            use_cross_attn=use_cross_attn,
            use_adaln_cond=use_adaln_cond,
            use_readout_token=use_readout_token,
            use_rope=use_rope,
            use_nope=use_nope,
            vlm_prompt_style=vlm_prompt_style,
            token_dropout=token_dropout,
            action_type_adaln=action_type_adaln,
            sampling_type=sampling_type,
            use_proprio=use_proprio,
            return_act_chunk=return_act_chunk,
            second_view_key=second_view_key,
        )
        # Initialize model dimensions
        self._init_dimensions(
            dit_dim=dit_dim,
            n_heads=n_heads,
            lowdim_obs_dim=lowdim_obs_dim,
            action_dim=action_dim,
            act_window_size=act_window_size,
            multistep=multistep,
            num_sampling_steps=num_sampling_steps,
        )

        # Setup VLM and core components
        self._setup_vlm(vlm_path, freeze_vision_tower, freeze_florence)
        hidden_dim = self.vlm.config.text_config.d_model
        self.vlm_latent_dim = hidden_dim
        self.action_type_adaln = action_type_adaln
        self.use_proprio = use_proprio

        # Setup DiT components
        self._setup_dit_components(
            dit_dim=dit_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            lowdim_obs_dim=lowdim_obs_dim,
            action_dim=action_dim,
            act_window_size=act_window_size,
            hidden_dim=hidden_dim,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
            mlp_pdrop=mlp_pdrop,
            use_cross_attn=use_cross_attn,
            use_rope=use_rope,
            use_nope=use_nope,
            query_seq_len=query_seq_len,
            rope_theta=rope_theta,
        )

        # Load pre-training weights
        weights = torch.load(
            '/home/ngkanats/repos/lbs/analogical_manipulation/360000_model_weights.pt',
            map_location="cpu",
            weights_only=True
        )
        _dict = {}
        for key, value in weights.items():
            _key = key.replace('agent.', '')
            # if any(w in key for w in ['dit.', 'frequency_embedder', 't_embedder', 'cond_linear']):
            #     continue
            if _key.startswith('dit') and _key.endswith('.weight') and 'mlp' in _key:
                _key = _key.replace('.c_fc1', '.fc1')
                _key = _key.replace('.c_fc2', '.fc2')
                _key = _key.replace('.c_proj', '.proj')
            _dict[_key] = value
        # Load weights flexibly
        msn, unxpct = self.load_state_dict(_dict, strict=False)
        if msn:
            print(f"Missing keys (not found in checkpoint): {len(msn)}")
            print(msn)
        if unxpct:
            print(f"Unexpected keys (ignored): {len(unxpct)}")
            print(unxpct)
        if not msn and not unxpct:
            print("All keys matched successfully!")

        # Normalization for the 3D space, will be loaded in the main process
        self.workspace_normalizer = nn.Parameter(
            torch.Tensor([[0., 0., 0.],
                          [1., 1., 1.]]),
            requires_grad=False
        )
        self.img_normalizer = CLIPTransform()
        self._quaternion_format = 'xyzw'

        # Relative positional embeddings
        self.relative_pe_layer = RotaryPositionEncoding3D(self.dit_dim)

    def _setup_dit_components(self, **kwargs):
        super()._setup_dit_components(**kwargs)
        # Extract parameters
        dit_dim = kwargs['dit_dim']
        n_heads = kwargs['n_heads']
        n_layers = kwargs['n_layers']
        hidden_dim = kwargs['hidden_dim']
        use_cross_attn = kwargs['use_cross_attn']
        use_rope = kwargs['use_rope']
        use_nope = kwargs['use_nope']

        # DiT blocks
        self.dit = nn.ModuleList([
            FlowBlock3D(
                dit_dim, n_heads,
                attn_pdrop=kwargs['attn_pdrop'],
                resid_pdrop=kwargs['resid_pdrop'],
                mlp_pdrop=kwargs['mlp_pdrop'],
                use_cross_attn=use_cross_attn,
                use_rope=use_rope,
                query_seq_len=kwargs['query_seq_len'],
                rope_theta=kwargs['rope_theta'],

            ) for _ in range(n_layers)
        ])

    def _get_pos(self, features, pcds):
        # interpolate pcds
        pcds = F.interpolate(
            einops.rearrange(pcds, "bt ncam c h w -> (bt ncam) c h w"),
            (7, 7),
            mode='nearest'
        )
        # Merge different cameras
        pcds = einops.rearrange(
            pcds,
            "(bt ncam) c h w -> bt ncam (h w) c", ncam=3
        )
        # features is (B, T, C)
        B, T, C = features.shape
        pos_ = torch.cat((
            torch.zeros(B, 1, 3, device=features.device),
            pcds[:, 0],
            torch.zeros(B, 1, 3, device=features.device),
            pcds[:, 1],
            torch.zeros(B, 1, 3, device=features.device),
            pcds[:, 2],
            torch.zeros(B, T - 150, 3, device=features.device)
        ), 1)
        return self.relative_pe_layer(pos_)

    def compute_loss(self, gt_trajectory, rgb3d, rgb2d, instruction, pcds):
        # Compute loss
        obs_features = self.encode_observations(rgb3d, rgb2d, instruction)
        obs_features['pos'] = self._get_pos(obs_features['features'], pcds)
        return self.rf_loss(obs_features, gt_trajectory)

    @torch.no_grad()
    def compute_trajectory(self, trajectory_mask, rgb3d, rgb2d, instruction, pcds):
        """Lightning validation step"""
        obs_features = self.encode_observations(rgb3d, rgb2d, instruction)
        obs_features['pos'] = self._get_pos(obs_features['features'], pcds)
        
        # Generate noise for sampling
        noise_actions = torch.randn(
            size=tuple(trajectory_mask.shape) + (self.action_dim,),
            device=trajectory_mask.device
        )

        return self.sample_actions(noise_actions, obs_features, inference=True)

    def dit_forward(self, z, t, cond_dict):
        """
        Forward pass through the DiT blocks.
        """
        default_dtype = next(self.parameters()).dtype
        B, t_seq, d = z.shape
        traj_pos = self.relative_pe_layer(self.unnormalize_pos(z[..., :3]))
        
        # Get conditioning information
        cond = cond_dict['features'].to(default_dtype)
        frequency_embeds = cond_dict['frequency_embeds'].squeeze(1).to(default_dtype)
        action_type = cond_dict['action_type'].to(z.device)
        
        # Handle proprioception
        proprio_embeds = torch.zeros_like(frequency_embeds)
        
        # Encode actions
        z, valid_dims = self.encode_actions(z, action_type)
        
        # Add positional encoding if not using ROPE/NOPE
        if not self.use_rope and not self.use_nope:
            z = z + self.positional_encoding
        
        # Process embeddings
        t_emb = stateless_norm(self.t_embedder(t)) + \
                stateless_norm(frequency_embeds).squeeze(1) + \
                stateless_norm(proprio_embeds).squeeze(1)
        
        cond = self.cond_linear(self.cond_norm(cond))
        
        # Set up conditioning
        if self.use_adaln_cond:
            vlm_token = cond[:, 0, :] if self.use_readout_token else cond.mean(dim=1)
            global_cond = vlm_token + t_emb
        else:
            global_cond = t_emb
        
        # Setup context
        cx = z
        context = cond if self.use_cross_attn else None
        
        # Get adaln signals
        if not self.action_type_adaln:
            global_adaln = self.adaln(global_cond)
        else:
            global_adaln = self.action_specific_adaln(global_cond, action_type)
        

        # Process through DiT blocks
        for layer in self.dit:
            cx = layer(
                cx, 
                global_cond, 
                context=context, 
                is_causal=True, 
                global_adaln=global_adaln,
                x_pos=traj_pos,
                context_pos=cond_dict['pos']
            )
            
        # Decode and return
        return self.decode_actions(cx, action_type, valid_dims)

    def forward(
        self,
        gt_trajectory,
        trajectory_mask,
        rgb3d,
        rgb2d,
        pcd,
        instruction,
        proprio,
        run_inference=False
    ):
        """
        Arguments:
            gt_trajectory: (B, trajectory_length, nhand, 3+4+X)
            trajectory_mask: (B, trajectory_length, nhand)
            rgb3d: (B, num_3d_cameras, 3, H, W) in [0, 1]
            rgb2d: (B, num_2d_cameras, 3, H, W) in [0, 1]
            pcd: (B, num_3d_cameras, 3, H, W) in world coordinates
            instruction: tokenized text instruction
            proprio: (B, nhist, nhand, 3+4+X)

        Note:
            The input rotation is ALWAYS expressed as a quaternion.
            The model converts it to 6D internally.

        Returns:
            - loss: scalar, if run_inference is False
            - trajectory: (B, trajectory_length, nhand, 3+4+X), at inference
        
        batch["rgb_obs"]['rgb_static'] (B, 1, C, H, W)
        batch["rgb_obs"]['rgb_gripper'] (B, 1, C, H, W)
        batch["lang_text"]
        dataset_batch["actions"] (B, T, action_dim)
        """
        rgb3d = self.img_normalizer(rgb3d)
        # Inference, don't use gt_trajectory
        if run_inference:
            B, T, H = trajectory_mask.shape
            trajectory = self.compute_trajectory(
                trajectory_mask.view(B, T * H),
                rgb3d[:, :1], rgb3d[:, 1:], instruction, pcd
            )
            trajectory = trajectory.reshape(B, T, H, -1)
            trajectory = self.unconvert_rot(
                trajectory.flatten(1, 2)
            ).unflatten(1, (T, H))
            trajectory[..., :3] = self.unnormalize_pos(trajectory[..., :3])
            trajectory[..., -1] = (trajectory[..., -1] + 1) / 2
            return trajectory

        # Training, use gt_trajectory to compute loss
        gt_trajectory = gt_trajectory.clone()
        B, T, H, _ = gt_trajectory.shape
        gt_trajectory[..., :3] = self.normalize_pos(gt_trajectory[..., :3])
        gt_trajectory = self.convert_rot(
            gt_trajectory.flatten(1, 2)
        ).unflatten(1, (T, H))
        gt_trajectory[..., -1] = 2 * gt_trajectory[..., -1] - 1
        return self.compute_loss(
            gt_trajectory.flatten(1, 2), rgb3d[:, :1], rgb3d[:, 1:], instruction, pcd)