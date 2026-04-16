import gc, os, torch
from torch import nn
from transformers import AutoProcessor


def _parse_dotted_lora_suffixes(dotted_csv: str) -> list[str]:
    """
    Split CLI dotted paths into suffixes **after** the layer index, e.g.
    ``self_attn.q_proj,mlp.gate_proj`` -> ``["self_attn.q_proj", "mlp.gate_proj"]``.
    These are joined with ``layers.{i}.`` / ``blocks.{i}.`` for PEFT (see below).
    """
    out = []
    for part in dotted_csv.split(","):
        p = part.strip()
        if p and p not in out:
            out.append(p)
    return out


def _resolve_peft_target_modules(
    module: nn.Module,
    stack_name: str,
    n_total: int,
    start_layer: int,
    dotted_targets: list[str],
) -> list[str]:
    """
    Resolve user-facing dotted targets to exact ``named_modules()`` keys.

    This is more robust than inferring a path prefix and constructing strings,
    because MiBot / Qwen variants may place the layer stack at slightly different
    nesting depths while still exposing the same leaf module suffixes.
    """
    if not dotted_targets:
        return []

    named_module_keys = [name for name, _ in module.named_modules() if name]
    resolved = []
    missing = []

    for target in dotted_targets:
        target = target.strip()
        if not target:
            continue

        layer_matches = []
        for layer_idx in range(start_layer, n_total):
            want = f"{stack_name}.{layer_idx}.{target}"
            exact = [name for name in named_module_keys if name == want or name.endswith(f".{want}")]
            layer_matches.extend(exact)

        if not layer_matches:
            missing.append(target)
            continue

        for name in layer_matches:
            if name not in resolved:
                resolved.append(name)

    if missing:
        preview = [name for name in named_module_keys if stack_name in name][:20]
        raise ValueError(
            f"Could not resolve LoRA target modules {missing} under '{stack_name}' "
            f"for layers [{start_layer}, {n_total}). Example module keys: {preview}"
        )

    return resolved


def _unwrap_peft_module(module: nn.Module) -> nn.Module:
    """Return inner base model if ``module`` is a PEFT-wrapped model."""
    try:
        from peft import PeftModel
    except ImportError:
        return module
    m = module
    while isinstance(m, PeftModel):
        m = m.get_base_model()
    return m


def _register_merger_pre_hook(visual: nn.Module, hook_fn) -> None:
    inner = _unwrap_peft_module(visual)
    inner.merger.register_forward_pre_hook(hook_fn)


def _register_block_forward_hook(visual: nn.Module, layer_idx: int, hook_fn):
    inner = _unwrap_peft_module(visual)
    return inner.blocks[layer_idx].register_forward_hook(hook_fn)


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
        if self.do_normalize:
            x = (x - self.mean.to(x.dtype)) / self.std.to(x.dtype)
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

    LLM adaptation uses **PEFT LoRA** on ``language_model`` when requested.
    Vision adaptation uses **PEFT LoRA** on ``visual`` when requested; otherwise the
    encoder may still full-finetune ``visual`` via ``requires_grad`` (see ``Qwen3VLEncoder3D``).
    """
    def __init__(
        self,
        model_name_or_path=None,
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
        multiscale_layers=(5, 11, 17, 23),
    ):
        super().__init__()
        use_vision_lora = vision_lora_num_layers > 0 and vision_lora_rank > 0
        use_llm_lora = llm_lora_num_layers > 0 and llm_lora_rank > 0
        use_any_lora = use_vision_lora or use_llm_lora
        LoraConfig = None
        get_peft_model = None
        if use_any_lora:
            try:
                from peft import LoraConfig, get_peft_model
            except ImportError as e:
                raise ImportError(
                    "PEFT is required for Qwen3-VL LoRA. Install with: pip install peft"
                ) from e

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
        if hasattr(self.vlm, "config"):
            self.vlm.config.use_cache = False
            if hasattr(self.vlm.config, "text_config"):
                self.vlm.config.text_config.use_cache = False
        if hasattr(self.vlm.language_model, "config"):
            self.vlm.language_model.config.use_cache = False

        self.token_dropout = nn.Dropout(0.1)
        self._debug_timing_printed = False
        self._rope_cache = {}
        self._rope_cache_max = 2048
        self._template_tensor_cache = {}

        self._pre_merge_features = None
        self.use_multiscale = bool(use_multiscale)
        self.multiscale_layers = tuple(
            dict.fromkeys(int(layer_idx) for layer_idx in multiscale_layers)
        )
        self._multiscale_features = {}
        self._multiscale_hook_handles = []

        # Store dotted paths for optimizer grouping (substring match on param names)
        self.vision_lora_target_modules = (
            [x.strip() for x in vision_lora_target_modules.split(",") if x.strip()]
            if isinstance(vision_lora_target_modules, str)
            else list(vision_lora_target_modules)
        )
        self.llm_lora_target_modules = (
            [x.strip() for x in llm_lora_target_modules.split(",") if x.strip()]
            if isinstance(llm_lora_target_modules, str)
            else list(llm_lora_target_modules)
        )
        self.resolved_vision_lora_target_modules = []
        self.resolved_llm_lora_target_modules = []

        # ---- PEFT: vision LoRA (optional) ----
        if use_vision_lora:
            vis_inner = _unwrap_peft_module(self.vlm.visual)
            n_blocks = len(vis_inner.blocks)
            start = max(0, n_blocks - vision_lora_num_layers)
            v_suf = _parse_dotted_lora_suffixes(",".join(self.vision_lora_target_modules))
            if not v_suf:
                v_suf = ["attn.qkv", "attn.proj"]
            v_tgt = _resolve_peft_target_modules(
                vis_inner,
                "blocks",
                n_blocks,
                start,
                v_suf,
            )
            self.resolved_vision_lora_target_modules = list(v_tgt)
            v_cfg = LoraConfig(
                r=vision_lora_rank,
                lora_alpha=vision_lora_alpha,
                lora_dropout=vision_lora_dropout,
                target_modules=v_tgt,
                bias="none",
            )
            self.vlm.visual = get_peft_model(self.vlm.visual, v_cfg)

        # ---- PEFT: LLM LoRA (optional) ----
        if use_llm_lora:
            lm = _unwrap_peft_module(self.vlm.language_model)
            n_layers = len(lm.layers)
            start = max(0, n_layers - llm_lora_num_layers)
            l_suf = _parse_dotted_lora_suffixes(",".join(self.llm_lora_target_modules))
            if not l_suf:
                l_suf = [
                    "self_attn.q_proj",
                    "self_attn.k_proj",
                    "self_attn.v_proj",
                    "self_attn.o_proj",
                ]
            l_tgt = _resolve_peft_target_modules(
                lm,
                "layers",
                n_layers,
                start,
                l_suf,
            )
            self.resolved_llm_lora_target_modules = list(l_tgt)
            l_cfg = LoraConfig(
                r=llm_lora_rank,
                lora_alpha=llm_lora_alpha,
                lora_dropout=llm_lora_dropout,
                target_modules=l_tgt,
                bias="none",
            )
            self.vlm.language_model = get_peft_model(self.vlm.language_model, l_cfg)

        _register_merger_pre_hook(self.vlm.visual, self._pre_merge_hook)
        if self.use_multiscale:
            vis_inner = _unwrap_peft_module(self.vlm.visual)
            n_blocks = len(vis_inner.blocks)
            invalid_layers = [
                layer_idx
                for layer_idx in self.multiscale_layers
                if layer_idx < 0 or layer_idx >= n_blocks
            ]
            if invalid_layers:
                raise ValueError(
                    f"Invalid multiscale vision layers {invalid_layers}; "
                    f"Qwen vision tower has {n_blocks} blocks."
                )
            for layer_idx in self.multiscale_layers:
                handle = _register_block_forward_hook(
                    self.vlm.visual,
                    layer_idx,
                    self._make_multiscale_hook(layer_idx),
                )
                self._multiscale_hook_handles.append(handle)

        if gradient_checkpointing:
            # This codebase captures intermediate activations through forward hooks
            # and lightweight forward-time side channels (for example pre/post-merge
            # vision features and selected LLM layer states). Reentrant activation
            # checkpointing runs those wrapped layer forwards under checkpoint
            # recomputation semantics that can make such side-captured tensors poor
            # autograd citizens. Non-reentrant checkpointing preserves the memory
            # savings we want here while keeping those captured tensors attached to
            # the forward graph used by downstream policy code.
            self.vlm.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

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
                attn_implementation="flash_attention_2",
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

    def _make_multiscale_hook(self, layer_idx):
        def hook(module, args, output):
            feats = output[0] if isinstance(output, tuple) else output
            self._multiscale_features[layer_idx] = feats

        return hook

    def _reshape_pre_merge_tokens(self, feats, batch_size, num_cameras, hp, wp):
        ms = self.spatial_merge_size
        feats = feats.reshape(
            batch_size * num_cameras,
            hp // ms,
            wp // ms,
            ms,
            ms,
            self.vision_hidden_size,
        )
        feats = feats.permute(0, 1, 3, 2, 4, 5).reshape(
            batch_size,
            num_cameras,
            hp,
            wp,
            self.vision_hidden_size,
        )
        return feats

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

    def _get_cached_template_ids(self, ids, cache_name: str, device) -> torch.Tensor:
        key = (cache_name, str(device))
        cached = self._template_tensor_cache.get(key)
        if cached is None:
            cached = torch.tensor(ids, dtype=torch.long, device=device)
            self._template_tensor_cache[key] = cached
        return cached

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
            prefix = self._get_cached_template_ids(
                self._chat_prefix_ids, "prefix", device
            )
            ids[:, cursor : cursor + prefix_len] = prefix[None, :]
            cursor += prefix_len
        for _ in range(ncam):
            ids[:, cursor] = self.vision_start_token_id
            ids[:, cursor + 1 : cursor + 1 + merged_tok].fill_(self.image_token_id)
            ids[:, cursor + 1 + merged_tok] = self.vision_end_token_id
            cursor += per_img
        if mid_len > 0:
            mid = self._get_cached_template_ids(
                self._chat_mid_ids, "mid", device
            )
            ids[:, cursor : cursor + mid_len] = mid[None, :]
            cursor += mid_len

        instr_start = cursor
        ids[:, instr_start : instr_start + tl] = text_ids.to(device=device)
        cursor += tl

        if suffix_len > 0:
            suffix = self._get_cached_template_ids(
                self._chat_suffix_ids, "suffix", device
            )
            ids[:, cursor : cursor + suffix_len] = suffix[None, :]

        return ids, instr_start, tl

    def _get_native_position_ids_cached(
        self,
        input_ids,
        grid_thw,
        attention_mask,
        text_attention_mask,
        ncam,
        hp,
        wp,
        text_valid_lens=None,
    ):
        """
        Cache native RoPE indices per sample length instead of per whole batch.
        For our prompt layout, RoPE depends on the visual layout and each sample's
        valid text length, so this avoids frequent batch-cache misses.
        """
        if text_valid_lens is None:
            if text_attention_mask is None:
                text_valid_lens = tuple([int(input_ids.shape[1])] * input_ids.shape[0])
            else:
                text_valid_lens = tuple(
                    int(x)
                    for x in text_attention_mask.sum(dim=1).to(dtype=torch.long).tolist()
                )
        else:
            text_valid_lens = tuple(int(x) for x in text_valid_lens)

        cached_pos = []
        cached_deltas = []
        seq_len = int(input_ids.shape[1])
        for batch_idx, valid_len in enumerate(text_valid_lens):
            key = (ncam, hp, wp, seq_len, int(valid_len))
            cached = self._rope_cache.get(key)
            if cached is None:
                single_input_ids = input_ids[batch_idx : batch_idx + 1]
                single_attention_mask = (
                    attention_mask[batch_idx : batch_idx + 1]
                    if attention_mask is not None
                    else None
                )
                single_grid_thw = grid_thw[
                    batch_idx * ncam : (batch_idx + 1) * ncam
                ]
                cached = self.vlm.get_rope_index(
                    input_ids=single_input_ids,
                    image_grid_thw=single_grid_thw,
                    attention_mask=single_attention_mask,
                )
                if len(self._rope_cache) >= self._rope_cache_max:
                    self._rope_cache.pop(next(iter(self._rope_cache)))
                self._rope_cache[key] = cached
            cached_pos.append(cached[0])
            cached_deltas.append(cached[1])

        position_ids = torch.cat(cached_pos, dim=1)
        rope_deltas = torch.cat(cached_deltas, dim=0)
        return position_ids, rope_deltas

    def forward(
        self,
        images,
        text_token_ids,
        text_attention_mask=None,
        text_valid_lens=None,
        return_post_llm_visual=False,
    ):
        """
        Args:
            images:         (B, ncam, 3, H, W), MiBot-preprocessed
            text_token_ids: (B, L), pre-tokenised instruction
            text_attention_mask: (B, L) optional; 1 = real token, 0 = pad (from Qwen tokenizer)
        Returns:
            pre_merge_feats: (B, ncam, h16, w16, vision_hidden_size)
            text_hidden:     (B, L, llm_hidden_size)
            multiscale_feats: dict[layer_idx -> (B, ncam, h16, w16, vision_hidden_size)] or None
            post_llm_visual: (B, ncam*h32*w32, llm_hidden_size), only when
                ``return_post_llm_visual=True``.
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
            input_ids,
            grid_thw,
            full_attention_mask,
            text_attention_mask,
            ncam,
            hp,
            wp,
            text_valid_lens=text_valid_lens,
        )
        # Keep model-internal rope_deltas behavior consistent with native path.
        self.vlm.rope_deltas = rope_deltas

        self._pre_merge_features = None
        if self.use_multiscale:
            self._multiscale_features = {}
        # pixel_values non-None → Qwen3VLModel.forward runs get_image_features
        # (ViT + deepstack mergers) and passes deepstack_visual_embeds into language_model.
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
        pre_merge_flat = self._pre_merge_features
        pre_merge = self._reshape_pre_merge_tokens(pre_merge_flat, B, ncam, hp, wp)

        multiscale_feats = None
        if self.use_multiscale:
            flat_multiscale = dict(self._multiscale_features)
            if self.multiscale_layers:
                flat_multiscale.setdefault(
                    self.multiscale_layers[-1], pre_merge_flat
                )
            missing_layers = [
                layer_idx
                for layer_idx in self.multiscale_layers
                if layer_idx not in flat_multiscale
            ]
            if missing_layers:
                raise RuntimeError(
                    f"Missing multiscale Qwen features for layers {missing_layers}."
                )
            multiscale_feats = {
                layer_idx: self._reshape_pre_merge_tokens(
                    flat_multiscale[layer_idx], B, ncam, hp, wp
                )
                for layer_idx in self.multiscale_layers
            }

        # Feature 2 – text positions only from final LLM hidden states (deepstack-fused context via causal attention to visual positions)
        text_hidden = outputs.last_hidden_state[:, instr_start : instr_start + text_len, :]
        if text_attention_mask is not None:
            tm = text_attention_mask.to(device=text_hidden.device, dtype=text_hidden.dtype)
            text_hidden = text_hidden * tm.unsqueeze(-1)
        text_hidden = self.token_dropout(text_hidden)
        if return_post_llm_visual:
            image_token_mask = (input_ids == self.image_token_id)
            # Prompt template is shared across the batch, so image token
            # positions are identical for all samples.
            image_positions = image_token_mask[0].nonzero(as_tuple=False).squeeze(-1)
            post_llm_visual = outputs.last_hidden_state[:, image_positions, :]
            return pre_merge, text_hidden, multiscale_feats, post_llm_visual

        return pre_merge, text_hidden, multiscale_feats


def load_qwen3vl():
    return Qwen3VLBackbone(), Qwen3VLNormalize()
