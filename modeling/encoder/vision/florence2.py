import einops
import torch
from torch import nn
import math
import torch.nn.functional as F
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

    def forward_padding(self, images, lang_tokens, output_hidden_states=False):
        B, ncam, C, H, W = images.shape
        target_size = max(H, W)
        pad_h = target_size - H
        pad_w = target_size - W
        flat_images = images.reshape(-1, C, H, W)
        if pad_h > 0 or pad_w > 0:
            flat_images = F.pad(flat_images, (0, pad_w, 0, pad_h), value=0.0)
            
        # im_feats: (B*ncam, Total_Tokens, Dim)
        im_feats = self.vlm._encode_image(flat_images)
        total_tokens = im_feats.shape[1]
        grid_side = int(math.sqrt(total_tokens))
        has_cls = False
        if (grid_side * grid_side) != total_tokens:
            grid_side = int(math.sqrt(total_tokens - 1))
            if (grid_side * grid_side) == (total_tokens - 1):
                has_cls = True
            else:
                pass 

        if has_cls:
            cls_token = im_feats[:, 0:1, :]
            patch_tokens = im_feats[:, 1:, :]
        else:
            cls_token = im_feats[:, 0:1, :] * 0
            patch_tokens = im_feats

        D = patch_tokens.shape[-1]
        patch_tokens = patch_tokens.view(-1, grid_side, grid_side, D)
        
        keep_h_ratio = H / target_size
        keep_w_ratio = W / target_size
        valid_h_tokens = int(math.ceil(grid_side * keep_h_ratio))
        valid_w_tokens = int(math.ceil(grid_side * keep_w_ratio))
        
        patch_tokens = patch_tokens[:, :valid_h_tokens, :valid_w_tokens, :]
        patch_tokens = patch_tokens.flatten(1, 2)
        if has_cls:
            im_feats = torch.cat([cls_token, patch_tokens], dim=1)
        else:
            im_feats = patch_tokens
        
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

    def forward(self, images, lang_tokens, output_hidden_states=False):
        """
        Encode observations using Florence-2.
        """
        # Process primary image
        B, ncam, C, H, W = images.shape

        # Extract visual features
        im_feats = self.vlm._encode_image(images.reshape(-1, C, H, W))
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
