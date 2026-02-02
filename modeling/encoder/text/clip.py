import transformers
import torch
from torch import nn


class ClipTokenizer:

    def __init__(self):
        super().__init__()
        self.tokenizer = transformers.CLIPTokenizer.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        self.tokenizer.model_max_length = 53

    @torch.inference_mode()
    def __call__(self, instructions):
        return self.tokenizer(
            instructions,
            # padding="longest",
            padding="max_length",
            return_tensors="pt"
        )["input_ids"]


class ClipTextEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = transformers.CLIPTextModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

    def forward(self, tokens):
        return self.model(tokens).last_hidden_state
