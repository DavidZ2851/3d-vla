import functools

import torch
import torch.nn as nn
from timm.layers.mlp import Mlp
from transformers import AutoModelForCausalLM, AutoProcessor

from .transformers import (
    TimestepEmbedder,
    SharedAdaLNController,
    RmsNorm,
    FreqEmbedder,
    ActionSpaceEmbedderParameter,
    FlowBlock, 
    stateless_norm
)
from .utils import ActionIndex, generate_policy_prompt


class FLOWERVLA(nn.Module):

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
        # VLM Configuration
        vlm_path: str = "microsoft/Florence-2-large",
        freeze_florence: bool = False,
        freeze_vision_tower: bool = False,
        vlm_prompt_style: str = "default",
        token_dropout: float = 0.1,
        
        # Model Structure
        multistep: int = 10,
        num_sampling_steps: int = 4,
        lowdim_obs_dim: int = 7,
        action_dim: int = 7,
        act_window_size: int = 10,
        
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
        n_layers: int = 18,
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
        self.action_space_index = ActionIndex()
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

        # Normalization for the 3D space, will be loaded in the main process
        self.workspace_normalizer = nn.Parameter(
            torch.Tensor([[0., 0., 0., 0., 0., 0.],
                          [1., 1., 1., 1., 1., 1.]]),
            requires_grad=False
        )
        self.img_normalizer = CLIPTransform()

    def normalize_pos(self, pos):
        return torch.cat((
            torch.clamp(pos[..., :3], -0.02, 0.02) / 0.02,
            torch.clamp(pos[..., 3:6], -0.05, 0.05) / 0.05
        ), -1)
        pos_min = self.workspace_normalizer[0].float().to(pos.device)
        pos_max = self.workspace_normalizer[1].float().to(pos.device)
        return (pos - pos_min) / (pos_max - pos_min) * 2.0 - 1.0

    def unnormalize_pos(self, pos):
        return torch.cat((
            pos[..., :3] * 0.02,
            pos[..., 3:6] * 0.05
        ), -1)
        pos_min = self.workspace_normalizer[0].float().to(pos.device)
        pos_max = self.workspace_normalizer[1].float().to(pos.device)
        return (pos + 1.0) / 2.0 * (pos_max - pos_min) + pos_min

    def _init_flags(self, **kwargs):
        """Initialize model flags and configurations"""
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        if self.vlm_prompt_style not in ["default", "feature_focused", "state_oriented"]:
            raise ValueError("Invalid VLM prompt style")
            
        if self.sampling_type not in ['ln', 'pi_zero', 'loglogistic', 'uniform', 'stratified']:
            raise ValueError(f"Invalid sampling type: {self.sampling_type}")
        
        self.format_instruction = functools.partial(
                             generate_policy_prompt,
                             robot_name="Franka Panda",
                             action_space="Delta End-Effector",
                             num_arms="1",
                             prompt_style='minimal')
        
        self.use_adaln_cond = self.use_adaln_cond 
        self.use_readout_token = self.use_readout_token and self.use_adaln_cond
        self.use_proprio = self.use_proprio 
        self.use_second_view = self.use_second_view and self.second_view_key is not None
        self.use_cross_attn = self.use_cross_attn
        self.use_rope = self.use_rope and not self.use_nope
        self.use_nope = self.use_nope and not self.use_rope
        self.vlm_prompt_style = self.vlm_prompt_style
        self.return_act_chunk = False

    def _init_dimensions(self, **kwargs):
        """Initialize model dimensions"""
        for key, value in kwargs.items():
            setattr(self, key, value)
            
        if self.dit_dim % self.n_heads != 0:
            raise ValueError(f"dit_dim ({self.dit_dim}) must be divisible by n_heads ({self.n_heads})")

    def _setup_vlm(self, vlm_path, freeze_vision_tower, freeze_florence):
        """Initialize and configure the Florence-2 VLM"""
        print(f"Loading Florence-2 from {vlm_path}")
        
        self.vlm = AutoModelForCausalLM.from_pretrained(vlm_path, trust_remote_code=True)
        
        # Handle parameter freezing
        if freeze_florence:
            for param in self.vlm.parameters():
                param.requires_grad = False
        elif not freeze_vision_tower:
            for param in self.vlm.vision_tower.parameters():
                param.requires_grad = True

        # Setup processor and tokenizer
        self.processor = AutoProcessor.from_pretrained(vlm_path, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer
        
        # Create prompt embedding
        self.prompt_embeds = self._create_prompt_embed("<Flow>")
        
        # Remove unnecessary components
        del self.vlm.language_model.model.decoder
        del self.vlm.language_model.lm_head
        
        # Setup token dropout
        self.vlm_token_dropout = nn.Dropout(self.token_dropout)

    def _setup_dit_components(self, **kwargs):
        """Setup DiT model components"""
        # Extract parameters
        dit_dim = kwargs['dit_dim']
        n_heads = kwargs['n_heads']
        n_layers = kwargs['n_layers']
        hidden_dim = kwargs['hidden_dim']
        use_cross_attn = kwargs['use_cross_attn']
        use_rope = kwargs['use_rope']
        use_nope = kwargs['use_nope']

        self.action_encoders = nn.ModuleDict()
        self.action_decoders = nn.ModuleDict()
        if self.use_proprio:
            self.proprio_encoders = nn.ModuleDict()
            
        self.adaln = nn.ModuleDict() if self.action_type_adaln else None

        # Core components
        self.cond_linear = nn.Linear(hidden_dim, dit_dim, bias=False)
        self.t_embedder = TimestepEmbedder(dit_dim)
        self.cond_norm = RmsNorm(hidden_dim)
        self.frequency_embedder = FreqEmbedder(dit_dim)
        self.action_space_embedder = ActionSpaceEmbedderParameter(dit_dim, max_actions=len(self.action_space_index.action_spaces))


        # Positional encoding if not using ROPE/NOPE
        if not use_rope and not use_nope:
            self.positional_encoding = nn.Parameter(torch.randn(1, kwargs['act_window_size'], dit_dim) * 0.1)

        # DiT blocks
        self.dit = nn.ModuleList([
            FlowBlock(
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

        # Create components per action space
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            input_dim = self.action_space_index.get_action_dim(action_idx)
            
            # Add encoder/decoder for this action
            self.action_encoders[action_name] =  Mlp(in_features=input_dim, hidden_features=dit_dim, out_features=dit_dim, bias=True)
            self.action_decoders[action_name] = nn.Linear(dit_dim, input_dim)  # .to(self.device)
                
            if self.action_type_adaln:
                self.adaln[action_name] = SharedAdaLNController(dit_dim, global_conddim=dit_dim, use_cross_attn=use_cross_attn)

    def compute_loss(self, gt_trajectory, rgb3d, rgb2d, instruction):
        # Compute loss
        obs_features = self.encode_observations(rgb3d, rgb2d, instruction)
        return self.rf_loss(obs_features, gt_trajectory)

    @torch.no_grad()
    def compute_trajectory(self, trajectory_mask, rgb3d, rgb2d, instruction):
        """Lightning validation step"""
        obs_features = self.encode_observations(rgb3d, rgb2d, instruction)
        
        # Generate noise for sampling
        noise_actions = torch.randn(
            size=tuple(trajectory_mask.shape) + (self.action_dim,),
            device=trajectory_mask.device
        )

        return self.sample_actions(noise_actions, obs_features, inference=True)
            
    def rf_loss(self, cond, actions):
        """
        Compute the rectified flow loss.
        """
        default_dtype = next(self.parameters()).dtype
        
        if len(actions.shape) == 4:
            actions = actions.squeeze(1)
        b = actions.size(0)
        device = actions.device
        actions = actions.to(default_dtype)

        # Sample time based on sampling strategy
        if self.sampling_type == "pi_zero":
            alpha, beta = 1.5, 1.0
            t = torch.distributions.Beta(alpha, beta).sample((b,)).to(device)
            t = t.clamp(max=0.999)
        elif self.sampling_type == "ln":
            t = torch.sigmoid(torch.randn((b,), device=device))
            t = t.clamp(max=0.999).to(default_dtype)
        elif self.sampling_type == "uniform":
            eps = 1e-5
            t = (torch.rand(1, device=device) + torch.arange(b, device=device) / b) % (1 - eps)
            t = t.to(default_dtype)
        else:
            raise NotImplementedError(f"Sampling type {self.sampling_type} not implemented")

        # Interpolate between actions and noise
        texp = t.view([b] + [1] * (actions.dim() - 1))
        z1 = torch.randn_like(actions, device=device).to(default_dtype)

        # Interpolate
        zt = (1 - texp) * actions + texp * z1

        # Forward pass
        vtheta = self.dit_forward(zt, t, cond)
        # Compute loss on valid dimensions only
        diff = (z1 - actions) - vtheta
        valid_diff = diff
        loss = (valid_diff ** 2).mean()

        return loss

    def sample_actions(self, z, cond, inference=False):
        """
        Sample actions using Euler method.
        """
        steps = self.num_sampling_steps if inference else 5
        b = z.size(0)
        device = z.device

        # Integration
        dt = 1.0 / steps
        dt_tensor = torch.tensor([dt] * b, device=device).view([b] + [1]*(z.dim()-1))

        for i in range(steps, 0, -1):
            t_val = i / steps
            t_tensor = torch.full((b,), t_val, device=device)

            # Predict velocity field
            vc = self.dit_forward(z, t_tensor, cond)
            z = z - dt_tensor * vc

        return z.clamp(-1, 1)

    def dit_forward(self, z, t, cond_dict):
        """
        Forward pass through the DiT blocks.
        """
        default_dtype = next(self.parameters()).dtype
        B, t_seq, d = z.shape
        
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
                global_adaln=global_adaln
            )
            
        # Decode and return
        return self.decode_actions(cx, action_type, valid_dims)

    def action_specific_adaln(self, global_cond, action_type):
        """
        Generate action-specific AdaLN signals.
        """
        default_type = next(self.parameters()).dtype
        batch_size = global_cond.shape[0]
        num_chunks = 9 if self.use_cross_attn else 6
        device = global_cond.device
        
        mod_signals = [
            torch.zeros(batch_size, self.dit_dim, device=device, dtype=default_type) 
            for _ in range(num_chunks)
        ]
        
        for action_idx in range(len(self.action_space_index.action_spaces)):
            mask = (action_type == action_idx)
            if mask.any():
                action_name = self.action_space_index.get_action_name(action_idx)
                action_mod = self.adaln[action_name](global_cond)
                for i, signal in enumerate(action_mod):
                    mod_signals[i] = signal
        
        return mod_signals

    def _create_prompt_embed(self, prompt_text):
        """Create embeddings for prompt tokens"""
        # Add special token if not in vocabulary
        self.tokenizer.add_special_tokens({'additional_special_tokens': [prompt_text]})
        self.vlm.resize_token_embeddings(len(self.tokenizer))
        
        # Get token ID and create embedding
        prompt_token_id = self.tokenizer.convert_tokens_to_ids(prompt_text)
        prompt_embed = nn.Parameter(
            self.vlm.get_input_embeddings()(torch.tensor(prompt_token_id)), 
            requires_grad=False
        )
    
        return prompt_embed.unsqueeze(0).unsqueeze(0)

    def encode_observations(self, rgb_static, rgb_gripper, lang_text):
        """
        Encode observations using Florence-2.
        """
        device = rgb_static.device
        default_type = next(self.parameters()).dtype

        embed_tensor = torch.zeros(len(rgb_static), 1, 1)
        action_type_tensor = torch.ones(len(rgb_static), self.act_window_size, 7)
        # Process primary image
        image_tensor = rgb_static
        B, T, C, H, W = image_tensor.shape

        # Extract visual features
        image_features = self.vlm._encode_image(
            image_tensor.view(-1, C, H, W).to(device).to(default_type)
        ).to(default_type)
        image_features = image_features.view(B, T * image_features.shape[1], -1)

        # Process second view if enabled
        if self.use_second_view:
            image2_tensor = rgb_gripper
            image2_features = self.vlm._encode_image(
                image2_tensor.view(-1, C, H, W).to(device).to(default_type)
            ).to(default_type)
            image2_features = image2_features.view(B, T * image2_features.shape[1], -1)
            image_features = torch.cat([image_features, image2_features], dim=1)

        # Get text embeddings once to reuse
        constructed_prompts = self.construct_prompts(lang_text)
        text_embeds = self._get_text_embeddings(constructed_prompts, device)

        # Add task prompt and aggregation tokens
        task_prompt = self.prompt_embeds.expand(B, -1, -1).to(image_features.device)

        # Merge sequence
        merged_embeds = torch.cat([
            image_features,
            task_prompt,
            text_embeds.to(image_features.device)
        ], dim=1)

        # Create attention mask
        attention_mask = torch.ones(merged_embeds.shape[:2], device=merged_embeds.device)

        # Process through encoder
        features = self.vlm.get_encoder()(
            inputs_embeds=merged_embeds,
            attention_mask=attention_mask
        ).last_hidden_state

        # Apply dropout 
        features = self.vlm_token_dropout(features)

        # Prepare frequency and action space embeddings
        frequency_embeds = self.frequency_embedder(
            torch.ones_like(embed_tensor).to(device) * 3
        )

        # Get proprioception if enabled
        proprio = None

        return {
            'features': features,
            'frequency_embeds': frequency_embeds,
            'action_space_embeds': None,
            'action_type': torch.ones_like(action_type_tensor), # actiont ype is always 1
            'proprio': proprio,
            'attention_mask': attention_mask,
        }

    def encode_actions(self, z, action_type):
        """Encode actions using action-specific encoders."""
        default_dtype = next(self.parameters()).dtype
        action_type = action_type.to(z.device)
        batch_size = z.shape[0]
        encoded = torch.zeros(batch_size, z.shape[1], self.dit_dim, device=z.device).to(default_dtype)
        
        # Track valid dimensions per type
        valid_dims = torch.zeros_like(z).to(default_dtype)
        
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = (action_type == action_idx)
            if mask.any():
                encoded = self.action_encoders[action_name](z)
        
        return encoded, valid_dims

    def decode_actions(self, z, action_type, valid_dims):
        """Decode actions using action-specific decoders."""
        default_dtype = next(self.parameters()).dtype
        batch_size = z.shape[0]
        max_action_dim = self.action_dim
        decoded = torch.zeros(batch_size, z.shape[1], max_action_dim, 
                        device=z.device).to(default_dtype)
        
        for action_name, action_idx in self.action_space_index.action_spaces.items():
            mask = (action_type == action_idx)
            if mask.any():
                action_dim = self.action_space_index.get_action_dim(action_idx)
                
                pred = self.action_decoders[action_name](z)
                # Only assign to valid dimensions
                decoded = pred
        return decoded

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
            trajectory = self.compute_trajectory(
                trajectory_mask[:, :, 0], rgb3d[:, :1], rgb3d[:, 1:], instruction
            )
            trajectory = trajectory[:, :, None]
            trajectory[..., :6] = self.unnormalize_pos(trajectory[..., :6])
            trajectory[..., 6] = (trajectory[..., 6] + 1) / 2
            return trajectory

        # Training, use gt_trajectory to compute loss
        gt_trajectory = gt_trajectory[:, :, 0].clone()
        gt_trajectory[..., :6] = self.normalize_pos(gt_trajectory[..., :6])
        gt_trajectory[..., 6] = 2 * gt_trajectory[..., 6] - 1
        return self.compute_loss(
            gt_trajectory, rgb3d[:, :1], rgb3d[:, 1:], instruction)
    
    def construct_prompts(self, language_instruction):
        """
        Constructs prompts for Florence-2's encoder to extract task-relevant visual features.
        
        Args:
            language_instruction: [str]
            
        Returns:
            text_prompts: List of formatted prompts for encoder conditioning
        """
        text_prompts = []
        
        for instruction in language_instruction:
            if self.vlm_prompt_style == "default":
                # Original instruction only
                text_prompts.append(self.format_instruction(instruction))
                
            elif self.vlm_prompt_style == "feature_focused":
                # Focus on extracting visual features relevant for manipulation
                prompt = f"<od>{instruction}</od><grounding>identify objects and spatial relationships for robotic manipulation</grounding>"
                text_prompts.append(prompt)
                
            elif self.vlm_prompt_style == "state_oriented":
                # Focus on extracting state-relevant features
                prompt = f"<od>{instruction}</od><referring_expression_segmentation>locate objects and regions for manipulation</referring_expression_segmentation>"
                text_prompts.append(prompt)
                
            else:
                raise ValueError(f"Unknown prompt style: {self.vlm_prompt_style}")
        
        
        return text_prompts
    
    def _get_text_embeddings(self, text, device):
        """Get text embeddings to use with VLM"""
        text_inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        ).to(device)
        return self.vlm.get_input_embeddings()(text_inputs["input_ids"])


class CLIPTransform(nn.Module):

    def __init__(self):
        super().__init__()
        _mean = [0.48145466, 0.4578275, 0.40821073]
        _std = [0.26862954, 0.26130258, 0.27577711]
        self.register_buffer("mean", torch.tensor(_mean).reshape(1, 1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(_std).reshape(1, 1, -1, 1, 1))

    def forward(self, img):
        return (img - self.mean) / self.std
