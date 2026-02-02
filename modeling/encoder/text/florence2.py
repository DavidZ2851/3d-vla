import torch
from transformers import AutoProcessor


class Florence2Tokenizer:

    def __init__(self, model_name="microsoft/Florence-2-large"):
        super().__init__()
        # Setup processor and tokenizer
        processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.tokenizer = processor.tokenizer

    @torch.inference_mode()
    def __call__(self, instructions):
        return self.tokenizer(
            instructions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        )["input_ids"]
