import einops
import math
import torch
from torch import nn

from transformers import AutoModelForCausalLM


class CLIPTransform(nn.Module):

    def __init__(self):
        super().__init__()
        _mean = [0.48145466, 0.4578275, 0.40821073]
        _std = [0.26862954, 0.26130258, 0.27577711]
        self.register_buffer("mean", torch.tensor(_mean).reshape(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(_std).reshape(1, -1, 1, 1))

    def forward(self, img):
        return (img - self.mean) / self.std


class Florence2(nn.Module):

    def __init__(self, model_name="microsoft/Florence-2-large"):
        super().__init__()
        
        self.vlm = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True
        )
        
        # Remove unnecessary components
        del self.vlm.language_model.model.decoder
        del self.vlm.language_model.lm_head
        
        # Setup token dropout
        self.vlm_token_dropout = nn.Dropout(0.1)
        self._last_feature_hw = None

    @staticmethod
    def _infer_feature_hw(num_tokens, height, width):
        expected_h = max(1, math.ceil(height / 32))
        expected_w = max(1, math.ceil(width / 32))
        if expected_h * expected_w == num_tokens:
            return expected_h, expected_w

        aspect = width / float(height)
        candidates = []
        for feat_h in range(1, int(math.sqrt(num_tokens)) + 1):
            if num_tokens % feat_h != 0:
                continue
            feat_w = num_tokens // feat_h
            candidates.append((feat_h, feat_w))
            if feat_h != feat_w:
                candidates.append((feat_w, feat_h))

        if not candidates:
            raise ValueError(
                "Cannot infer Florence2 rectangular feature grid from "
                f"{num_tokens} tokens for input {height}x{width}."
            )

        return min(
            candidates,
            key=lambda hw: abs(math.log((hw[1] / float(hw[0])) / aspect)),
        )

    def _encode_image_allow_rectangular(self, pixel_values):
        """Florence2 image encode with rectangular feature-map support."""
        if len(pixel_values.shape) != 4:
            raise ValueError(f"invalid image shape {pixel_values.shape}")

        batch_size, _, height, width = pixel_values.shape
        if height == width:
            encoded = self.vlm._encode_image(pixel_values)
            grid_tokens = encoded.shape[1] - 1
            grid_side = int(math.sqrt(grid_tokens))
            if grid_side * grid_side == grid_tokens:
                self._last_feature_hw = (grid_side, grid_side)
            else:
                self._last_feature_hw = None
            return encoded

        temporal = 1
        x = self.vlm.vision_tower.forward_features_unpool(pixel_values)
        # Some Florence-2 remote-code builds return (features, input_size) here
        # instead of just the features tensor; keep only the features.
        if isinstance(x, (tuple, list)):
            x = x[0]
        x = x.view(batch_size * temporal, -1, x.shape[-1])
        feat_h, feat_w = self._infer_feature_hw(x.shape[-2], height, width)
        self._last_feature_hw = (feat_h, feat_w)

        if self.vlm.image_pos_embed is not None:
            x = x.view(batch_size * temporal, feat_h, feat_w, x.shape[-1])
            pos_embed = self.vlm.image_pos_embed(x)
            x = x + pos_embed
            x = x.view(batch_size, temporal * feat_h * feat_w, x.shape[-1])
        else:
            x = x.view(batch_size, temporal * feat_h * feat_w, x.shape[-1])

        if self.vlm.visual_temporal_embed is not None:
            temporal_view = x.view(batch_size, temporal, -1, x.shape[-1])
            visual_temporal_embed = self.vlm.visual_temporal_embed(
                temporal_view[:, :, 0]
            )
            x = temporal_view + visual_temporal_embed.view(
                1, temporal, 1, x.shape[-1]
            )

        x_feat_dict = {}
        x_view = x.view(batch_size, temporal, -1, x.shape[-1])
        x_feat_dict["spatial_avg_pool"] = x_view.mean(dim=2)
        x_feat_dict["temporal_avg_pool"] = x_view.mean(dim=1)
        x_feat_dict["last_frame"] = x_view[:, -1]

        new_x = []
        for image_feature_source in self.vlm.image_feature_source:
            if image_feature_source not in x_feat_dict:
                raise ValueError(
                    "invalid image feature source: {}".format(
                        image_feature_source
                    )
                )
            new_x.append(x_feat_dict[image_feature_source])

        x = torch.cat(new_x, dim=1)
        x = x @ self.vlm.image_projection
        x = self.vlm.image_proj_norm(x)
        return x

    def forward(self, images, lang_tokens, output_hidden_states=False):
        """
        Encode observations using Florence-2.
        """
        # Process primary image
        B, ncam, C, H, W = images.shape

        # Extract visual features
        im_feats = self._encode_image_allow_rectangular(
            images.reshape(-1, C, H, W)
        )
        im_feats = einops.rearrange(im_feats, "(B T) N F -> B (T N) (F)", B=B)

        # Get text embeddings once to reuse
        text_embeds = self.vlm.get_input_embeddings()(lang_tokens)

        # Merge sequence
        embs = torch.cat([im_feats, text_embeds], 1)

        # Create attention mask
        attention_mask = torch.ones(embs.shape[:2], device=embs.device)

        # Process through encoder
        outs = self.vlm.get_encoder()(
            inputs_embeds=embs,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states
        )
        features = outs['last_hidden_state']

        # Apply dropout 
        features = self.vlm_token_dropout(features)

        # Throw away the summarization token
        lang_feats = features[:, im_feats.shape[1]:]
        ntok = im_feats.shape[1] // ncam
        features = torch.cat([
            features[:, cam*ntok + 1:(cam + 1) * ntok] for cam in range(ncam)
        ], 1)

        # Handle hidden states
        if output_hidden_states:
            hidden = outs['hidden_states']
            hidden = torch.stack([hidden[i] for i in [3, 6, 9]], 1)
            hidden = torch.cat([
                hidden[:, :, cam*ntok + 1:(cam + 1) * ntok] for cam in range(ncam)
            ], 2)
            features = torch.cat([hidden, features.unsqueeze(1)], 1)

        return features, lang_feats


def load_florence2():
    return Florence2(), CLIPTransform()
