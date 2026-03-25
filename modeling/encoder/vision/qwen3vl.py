import gc, os, torch
from torch import nn
from transformers import AutoProcessor

class Qwen3VLNormalize(nn.Module):
    """
    MiBot/Qwen3-VL preprocessing:
      - optional rescale by 1/255 (if inputs are uint8-like),
      - normalize with processor-config mean/std (defaults to 0.5/0.5).
    """
    def __init__(self, model_name_or_path=None):
        super().__init__()
        path = _resolve_model_path(model_name_or_path)
        # Mibot default mean and std
        image_mean = [0.5, 0.5, 0.5]
        image_std = [0.5, 0.5, 0.5]
        do_rescale = True
        do_normalize = True
        rescale_factor = 1.0 / 255.0
        try:
            processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
            image_processor = processor.image_processor
            image_mean = getattr(image_processor, "image_mean", image_mean)
            image_std = getattr(image_processor, "image_std", image_std)
            do_rescale = getattr(image_processor, "do_rescale", do_rescale)
            do_normalize = getattr(image_processor, "do_normalize", do_normalize)
            rescale_factor = float(getattr(image_processor, "rescale_factor", rescale_factor))
        except Exception:
            pass  # Fall back to MiBot defaults when processor loading is unavailable.
        self.do_rescale = bool(do_rescale)
        self.do_normalize = bool(do_normalize)
        self.rescale_factor = float(rescale_factor)
        self.register_buffer("mean", torch.tensor(image_mean, dtype=torch.float32).reshape(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(image_std, dtype=torch.float32).reshape(1, -1, 1, 1))

    def forward(self, img):
        x = img
        if self.do_rescale:
            # 3DFA-VLA pipeline applies pixel/255 during data preprocessing.
            # If values look like raw uint8-range, apply MiBot's rescale factor.
            apply_rescale = True
            if torch.is_floating_point(x):
                max_val = x.detach().amax()
                if torch.isfinite(max_val) and max_val <= 1.0 + 1e-6:
                    apply_rescale = False
            if apply_rescale:
                x = x * self.rescale_factor
        if self.do_normalize:
            x = (x - self.mean) / self.std
        return x


def _resolve_model_path(model_name_or_path):
    if model_name_or_path is not None:
        return model_name_or_path
    path = os.environ.get("QWEN3VL_MODEL_PATH")
    if path is None:
        raise ValueError(
            "No model path provided. Set QWEN3VL_MODEL_PATH env var "
            "or pass model_name_or_path to Qwen3VLBackbone."
        )
    return path


class Qwen3VLBackbone(nn.Module):
    """
    Wraps a Qwen3-VL model and outputs two features:
    1. **Pre-merge** vision features – stride-16, dim ``vision_hidden_size``
       (captured by a forward-pre-hook on the main ``PatchMerger``).
    2. **Fused LLM** text hidden states – dim ``llm_hidden_size``, extracted
       from ``last_hidden_state`` at text-token positions (after full decoder + RMSNorm). 
       The forward pass runs MiBot/HF ``Qwen3VLModel`` with ``pixel_values`` set, which (per ``Qwen3VLModel.forward``) calls ``get_image_features`` → ViT returns ``deepstack_image_embeds``, then ``language_model(..., deepstack_visual_embeds=...)`` applies ``_deepstack_process`` at early decoder layers on visual positions.
       Text positions then attend (causally) to those enriched visuals, so the text slice matches the fused representation used when conditioning on the same VLM in MiBot.

    Prompts follow MiBot/Qwen chat template formatting. We derive template scaffolding from the tokenizer (when available) and then expand image placeholders to match merged visual token counts.

    Supports loading from:
      - A **MiBot** checkpoint  (``model_type="mibot"``): loads
        ``MiBoTForActionGeneration``, extracts ``model.vlm.model`` (``Qwen3VLModel``) — weights match because HuggingFace resolves the full ``vlm.model.visual.*`` / ``vlm.model.language_model.*`` key hierarchy during ``from_pretrained``.
      - A **Qwen3-VL** checkpoint (``model_type="qwen3_vl"``): loads
        ``Qwen3VLForConditionalGeneration`` and extracts ``.model``, or loads ``Qwen3VLModel`` directly.
    """
    def __init__(self, model_name_or_path=None):
        super().__init__()
        path = _resolve_model_path(model_name_or_path)
        vlm, vlm_config = self._load_vlm(path)
        self.vlm = vlm

        self.image_token_id = vlm_config.image_token_id
        self.vision_start_token_id = vlm_config.vision_start_token_id
        self.vision_end_token_id = vlm_config.vision_end_token_id
        self.spatial_merge_size = vlm_config.vision_config.spatial_merge_size
        self.patch_size = vlm_config.vision_config.patch_size
        self.temporal_patch_size = vlm_config.vision_config.temporal_patch_size
        self.vision_hidden_size = vlm_config.vision_config.hidden_size
        self.llm_hidden_size = vlm_config.text_config.hidden_size

        vlm_config.text_config.use_cache = False

        self.token_dropout = nn.Dropout(0.1)
        self._debug_timing_printed = False
        self._rope_cache = {}
        self._rope_cache_max = 2048

        self._pre_merge_features = None
        self.vlm.visual.merger.register_forward_pre_hook(self._pre_merge_hook)

        # Reuse MiBot tokenizer metadata and chat-template scaffolding when available.
        self._processor = None
        self._chat_prefix_ids = []
        self._chat_mid_ids = []
        self._chat_suffix_ids = []
        try:
            self._processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
            tok = self._processor.tokenizer
            self.image_token_id = getattr(tok, "image_token_id", self.image_token_id)
            self.vision_start_token_id = getattr(tok, "vision_start_token_id", self.vision_start_token_id)
            self.vision_end_token_id = getattr(tok, "vision_end_token_id", self.vision_end_token_id)
            self._init_chat_template_parts(tok)
        except Exception:
            self._init_chat_template_parts(None)

    @staticmethod
    def _load_vlm(path):
        """
        Load pretrained weights and return ``(Qwen3VLModel, Qwen3VLConfig)``.
        Handles two checkpoint layouts:
        MiBot
            MiBoTForActionGeneration
            └─ vlm (Qwen3VLForConditionalGeneration)
               └─ model (Qwen3VLModel)  ← what we need
        Qwen3-VL
            Qwen3VLForConditionalGeneration
            └─ model (Qwen3VLModel)  ← what we need
        """
        from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

        config = AutoConfig.from_pretrained(path, trust_remote_code=True)

        if getattr(config, "model_type", None) == "mibot":
            # ---- MiBot checkpoint ----
            full = AutoModel.from_pretrained(
                path, trust_remote_code=True, torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                low_cpu_mem_usage=True,
            )
            # MiBoTForActionGeneration.vlm  → Qwen3VLForConditionalGeneration, then Qwen3VLForConditionalGeneration.model → Qwen3VLModel
            vlm = full.vlm.model
            vlm_config = config.vlm_config

            # Free everything we don't need
            full.vlm.lm_head = None
            for attr in (
                "dit", "state_projector", "action_projector",
                "action_output_layer", "t_embedder", "t_projector",
                "rotary_emb", "sink",
            ):
                if hasattr(full, attr):
                    setattr(full, attr, None)
            del full
        else:
            # ---- Standard Qwen3-VL checkpoint ----
            raw = AutoModelForCausalLM.from_pretrained(
                path, trust_remote_code=True, torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
            if hasattr(raw, "model") and hasattr(raw.model, "visual"):
                vlm = raw.model          # Qwen3VLForConditionalGeneration → base
                raw.lm_head = None
                del raw
            elif hasattr(raw, "visual"):
                vlm = raw                # Already Qwen3VLModel
            else:
                raise ValueError(
                    f"Cannot locate Qwen3VL components in {type(raw).__name__}"
                )
            vlm_config = config

        gc.collect()
        torch.cuda.empty_cache()
        return vlm, vlm_config

    def _pre_merge_hook(self, module, args):
        # Input to the main PatchMerger: last ViT block output, shape (total_patches, vision_hidden_size)
        # total_patches = B * ncam * hp * wp (same concat order as pixel_values)
        self._pre_merge_features = args[0]

    @staticmethod
    def _find_subseq_index(sequence, subsequence):
        if len(subsequence) == 0 or len(sequence) < len(subsequence):
            return -1
        limit = len(sequence) - len(subsequence) + 1
        for i in range(limit):
            if sequence[i : i + len(subsequence)] == subsequence:
                return i
        return -1

    def _init_chat_template_parts(self, tokenizer):
        """
        Build template parts:
          [chat_prefix] [vision blocks] [chat_mid] [instruction] [chat_suffix]
        where each vision block expands to <vision_start> + image_pad*merged_tok + <vision_end>.
        """
        # Fallback: no additional chat wrappers.
        self._chat_prefix_ids = []
        self._chat_mid_ids = []
        self._chat_suffix_ids = []
        if tokenizer is None:
            return

        marker = "__3DFA_INSTRUCTION_MARKER__"
        marker_ids = tokenizer(marker, add_special_tokens=False)["input_ids"]
        if len(marker_ids) == 0:
            return

        try:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": marker},
                ],
            }]
            templ_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
            )
            if not isinstance(templ_ids, list) or len(templ_ids) == 0:
                return
        except Exception:
            return

        marker_pos = self._find_subseq_index(templ_ids, marker_ids)
        if marker_pos < 0:
            return
        prefix_with_vision = templ_ids[:marker_pos]
        suffix = templ_ids[marker_pos + len(marker_ids) :]

        vision_start_pos = -1
        for i, tok_id in enumerate(prefix_with_vision):
            if tok_id == self.vision_start_token_id:
                vision_start_pos = i
                break
        if vision_start_pos < 0:
            return

        vision_end_pos = -1
        for i in range(vision_start_pos + 1, len(prefix_with_vision)):
            if prefix_with_vision[i] == self.vision_end_token_id:
                vision_end_pos = i
                break
        if vision_end_pos < 0:
            return

        # Remove the whole template vision span so we can inject exactly
        # ncam blocks of [vision_start, image_pad*merged_tok, vision_end].
        self._chat_prefix_ids = prefix_with_vision[:vision_start_pos]
        self._chat_mid_ids = prefix_with_vision[vision_end_pos + 1 :]
        self._chat_suffix_ids = suffix

    def images_to_pixel_values(self, images):
        """(N,3,H,W) -> (N*hp*wp, C*tp*ps*ps) flat patches in merge-block order.
        Qwen3-VL ViT expects patches grouped so that each spatial_merge_size x spatial_merge_size block is contiguous (matching fast_pos_embed_interpolate, rot_pos_emb, and PatchMerger in the pretrained weights).  
        The HF Qwen2VLImageProcessor produces exactly this layout; we replicate it here for raw tensor inputs.
        """
        N, C, H, W = images.shape
        tp, ps, ms = self.temporal_patch_size, self.patch_size, self.spatial_merge_size
        hp, wp = H // ps, W // ps
        # Decompose H = (bh * ms * ps) and W = (bw * ms * ps), then reorder to
        # (N, bh, bw, ms, ms, C, ps, ps) so every ms x ms merge block is
        # contiguous when flattened -- matching Qwen2VLImageProcessor._preprocess.
        x = images.reshape(N, C, hp // ms, ms, ps, wp // ms, ms, ps)
        x = x.permute(0, 2, 5, 3, 6, 1, 4, 7).contiguous()
        x = x.reshape(N * hp * wp, C, ps, ps)
        if tp > 1:
            x = x.unsqueeze(2).expand(-1, -1, tp, -1, -1)
        else:
            x = x.unsqueeze(2)
        x = x.reshape(N * hp * wp, C * tp * ps * ps)
        return x

    def build_input_ids(self, text_ids, ncam, merged_tok, device):
        """
        Build chat-formatted multimodal ids:
          [chat_prefix] [vision blocks x ncam] [chat_mid] [instruction] [chat_suffix]
        """
        B, tl = text_ids.shape
        prefix_len = len(self._chat_prefix_ids)
        mid_len = len(self._chat_mid_ids)
        suffix_len = len(self._chat_suffix_ids)
        per_img = 1 + merged_tok + 1
        img_len = ncam * per_img
        total_len = prefix_len + img_len + mid_len + tl + suffix_len
        ids = torch.zeros(B, total_len, dtype=torch.long, device=device)

        cursor = 0
        if prefix_len > 0:
            prefix = torch.tensor(self._chat_prefix_ids, dtype=torch.long, device=device)
            ids[:, cursor : cursor + prefix_len] = prefix[None, :]
            cursor += prefix_len
        for _ in range(ncam):
            ids[:, cursor] = self.vision_start_token_id
            ids[:, cursor + 1 : cursor + 1 + merged_tok] = self.image_token_id
            ids[:, cursor + 1 + merged_tok] = self.vision_end_token_id
            cursor += per_img
        if mid_len > 0:
            mid = torch.tensor(self._chat_mid_ids, dtype=torch.long, device=device)
            ids[:, cursor : cursor + mid_len] = mid[None, :]
            cursor += mid_len

        instr_start = cursor
        ids[:, instr_start : instr_start + tl] = text_ids
        cursor += tl

        if suffix_len > 0:
            suffix = torch.tensor(self._chat_suffix_ids, dtype=torch.long, device=device)
            ids[:, cursor : cursor + suffix_len] = suffix[None, :]

        return ids, instr_start, tl

    def _get_native_position_ids_cached(self, input_ids, grid_thw, attention_mask, text_attention_mask, ncam, hp, wp):
        """
        Use original Qwen3VLModel.get_rope_index(), but cache by layout+mask pattern.
        This preserves native MiBot/Qwen logic while avoiding repeated Python loops.
        """
        if text_attention_mask is None:
            valid_text_lens = tuple([input_ids.shape[1]] * input_ids.shape[0])
        else:
            valid_text_lens = tuple(
                text_attention_mask.sum(dim=1).to(dtype=torch.long).tolist()
            )
        key = (ncam, hp, wp, int(input_ids.shape[1]), valid_text_lens)
        cached = self._rope_cache.get(key)
        if cached is not None:
            return cached

        position_ids, rope_deltas = self.vlm.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=grid_thw,
            attention_mask=attention_mask,
        )
        # Keep cache bounded.
        if len(self._rope_cache) >= self._rope_cache_max:
            self._rope_cache.pop(next(iter(self._rope_cache)))
        self._rope_cache[key] = (position_ids, rope_deltas)
        return position_ids, rope_deltas

    def forward(self, images, text_token_ids, text_attention_mask=None):
        """
        Args:
            images:         (B, ncam, 3, H, W), MiBot-preprocessed
            text_token_ids: (B, L), pre-tokenised instruction
            text_attention_mask: (B, L) optional; 1 = real token, 0 = pad (from tokenizer)
        Returns:
            pre_merge_feats: (B, ncam, h16, w16, vision_hidden_size)
            text_hidden:     (B, L, llm_hidden_size)
        """
        B, ncam, C, H, W = images.shape
        dev = images.device
        hp = H // self.patch_size
        wp = W // self.patch_size
        ms = self.spatial_merge_size
        merged_tok = (hp // ms) * (wp // ms)

        pixel_values = self.images_to_pixel_values(images.reshape(B * ncam, C, H, W))
        grid_thw = torch.tensor(
            [1, hp, wp], dtype=torch.long, device=dev
        ).view(1, 3).expand(B * ncam, -1)
        input_ids, instr_start, text_len = self.build_input_ids(
            text_token_ids, ncam, merged_tok, dev
        )

        if text_attention_mask is not None:
            full_attention_mask = torch.ones(
                B, input_ids.shape[1], dtype=text_attention_mask.dtype, device=dev
            )
            full_attention_mask[:, instr_start : instr_start + text_len] = text_attention_mask.to(device=dev)
        else:
            full_attention_mask = None

        position_ids, rope_deltas = self._get_native_position_ids_cached(
            input_ids, grid_thw, full_attention_mask, text_attention_mask, ncam, hp, wp
        )
        # Keep model-internal rope_deltas behavior consistent with native path.
        self.vlm.rope_deltas = rope_deltas

        self._pre_merge_features = None
        # pixel_values non-None → Qwen3VLModel.forward runs get_image_features (ViT + deepstack mergers) and passes deepstack_visual_embeds into language_model.
        outputs = self.vlm(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=grid_thw,
            attention_mask=full_attention_mask,
            position_ids=position_ids,
        )

        # Feature 1 – pre-merge vision (stride 16, ViT hidden dim)
        # The ViT outputs patches in merge-block order (bh, bw, ms, ms).
        # Undo the reorder to recover row-major (hp, wp) spatial layout so features align with the bilinearly-downsampled PCD grid.
        pre_merge = self._pre_merge_features
        ms = self.spatial_merge_size
        pre_merge = pre_merge.reshape(
            B * ncam, hp // ms, wp // ms, ms, ms, self.vision_hidden_size
        )
        pre_merge = pre_merge.permute(0, 1, 3, 2, 4, 5).reshape(
            B, ncam, hp, wp, self.vision_hidden_size
        )

        # Feature 2 – text positions only from final LLM hidden states (deepstack-fused context via causal attention to visual positions)
        text_hidden = outputs.last_hidden_state[:, instr_start : instr_start + text_len, :]
        if text_attention_mask is not None:
            tm = text_attention_mask.to(device=text_hidden.device, dtype=text_hidden.dtype)
            text_hidden = text_hidden * tm.unsqueeze(-1)
        text_hidden = self.token_dropout(text_hidden)
        return pre_merge, text_hidden


def load_qwen3vl():
    return Qwen3VLBackbone(), Qwen3VLNormalize()
