import os
from collections import namedtuple
import torch
from transformers import AutoProcessor, AutoTokenizer


Qwen3VLTokenizerOutput = namedtuple(
    "Qwen3VLTokenizerOutput", ["input_ids", "attention_mask"]
)

class Qwen3VLTokenizer:
    """
    Tokenise text instructions with the same Qwen2 tokenizer as Qwen3-VL / MiBot.
    Returns input_ids and attention_mask (padding-aware), matching MiBotProcessor behaviour (no CLIP-style max_length=77 cap).
    """
    def __init__(self, model_name_or_path=None):
        if model_name_or_path is None:
            model_name_or_path = os.environ.get("QWEN3VL_MODEL_PATH")
        if model_name_or_path is None:
            raise ValueError(
                "No model path provided. Set QWEN3VL_MODEL_PATH env var "
                "or pass model_name_or_path to Qwen3VLTokenizer."
            )
        self.processor = None
        try:
            self.processor = AutoProcessor.from_pretrained(
                model_name_or_path, trust_remote_code=True
            )
            self.tokenizer = self.processor.tokenizer
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, trust_remote_code=True
            )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.inference_mode()
    def __call__(self, instructions):
        out = self.tokenizer(
            instructions,
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
        )
        return Qwen3VLTokenizerOutput(
            input_ids=out["input_ids"],
            attention_mask=out["attention_mask"],
        )
