import einops
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
