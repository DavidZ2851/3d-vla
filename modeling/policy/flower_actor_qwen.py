from ..encoder.multimodal.encoder3d_qwen import Qwen3VLEncoder3D
from .flower_actor import TransformerHead
from .base_denoise_actor import DenoiseActor as BaseDenoiseActor


class QwenFlowerActor3D(BaseDenoiseActor):
    """
    3DFA-VLA policy backed by Qwen3-VL:
      - Encoder:  ``Qwen3VLEncoder3D``  (pre-merge vision + fused LLM text)
      - Decoder:  ``TransformerHead``   (3D RoPE, AdaLN, pos/rot/gripper heads)
      - Denoise:  inherited from ``BaseDenoiseActor`` (RF / DDPM / DDIM)
    """
    def __init__(
        self,
        backbone="qwen3vl",
        output_level="res3",
        upsample=False,
        finetune_backbone=False,
        num_finetune_vision_layers=0,
        finetune_text_encoder=False,
        num_vis_instr_attn_layers=2,
        fps_subsampling_factor=5,
        embedding_dim=60,
        num_attn_heads=9,
        nhist=3,
        nhand=1,
        num_shared_attn_layers=4,
        relative=False,
        rotation_format="quat_xyzw",
        denoise_timesteps=100,
        denoise_model="ddpm",
        lv2_batch_size=1,
        vision_lora_rank=0,
        vision_lora_alpha=16,
        vision_lora_dropout=0.0,
        vision_lora_num_layers=0,
        vision_lora_target_modules="attn.qkv,attn.proj",
        llm_lora_rank=0,
        llm_lora_alpha=16,
        llm_lora_dropout=0.0,
        llm_lora_num_layers=0,
        llm_lora_target_modules="self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj",
        gradient_checkpointing=False,
        use_multiscale=False,
    ):
        super().__init__(
            embedding_dim=embedding_dim,
            num_attn_heads=num_attn_heads,
            nhist=nhist,
            nhand=nhand,
            num_shared_attn_layers=num_shared_attn_layers,
            relative=relative,
            rotation_format=rotation_format,
            denoise_timesteps=denoise_timesteps,
            denoise_model=denoise_model,
            lv2_batch_size=lv2_batch_size,
        )

        # Override encoder with Qwen3-VL backed version
        self.encoder = Qwen3VLEncoder3D(
            backbone=backbone,
            embedding_dim=embedding_dim,
            nhist=nhist * nhand,  # NOTE: use nhist=nhist*2 when aligning action space
            num_attn_heads=num_attn_heads,
            fps_subsampling_factor=fps_subsampling_factor,
            finetune_backbone=finetune_backbone,
            num_finetune_vision_layers=num_finetune_vision_layers,
            finetune_text_encoder=finetune_text_encoder,
            vision_lora_rank=vision_lora_rank,
            vision_lora_alpha=vision_lora_alpha,
            vision_lora_dropout=vision_lora_dropout,
            vision_lora_num_layers=vision_lora_num_layers,
            vision_lora_target_modules=vision_lora_target_modules,
            llm_lora_rank=llm_lora_rank,
            llm_lora_alpha=llm_lora_alpha,
            llm_lora_dropout=llm_lora_dropout,
            llm_lora_num_layers=llm_lora_num_layers,
            llm_lora_target_modules=llm_lora_target_modules,
            gradient_checkpointing=gradient_checkpointing,
            use_multiscale=use_multiscale,
        )

        # Override head with flower_actor's TransformerHead (3D RoPE + AdaLN)
        self.prediction_head = TransformerHead(
            embedding_dim=embedding_dim,
            nhist=nhist * nhand,  # NOTE: use nhist=nhist*2 when aligning action space
            num_attn_heads=num_attn_heads,
            num_shared_attn_layers=num_shared_attn_layers,
            rot_dim=3 if rotation_format in ['euler', 'axisangle'] else 6  # NOTE: could change to always 6d
        )
        if nhand == 1:
            self.prediction_head.hand_embed = None
