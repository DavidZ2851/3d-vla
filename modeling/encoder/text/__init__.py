from .clip import ClipTextEncoder, ClipTokenizer
from .florence2 import Florence2Tokenizer
from .siglip2 import SigLip2TextEncoder
from .qwen3vl import Qwen3VLTokenizer


def fetch_text_encoders(model_name):
    """Return encoder class and latent dimension."""
    if model_name == 'clip':
        return ClipTextEncoder(), 512
    if model_name == 'florence2':
        return None, 1024
    if model_name == 'siglip2_256':
        return SigLip2TextEncoder('hf-hub:timm/ViT-B-16-SigLIP2-256'), 768
    if model_name == 'siglip2_512':
        return SigLip2TextEncoder('hf-hub:timm/ViT-B-16-SigLIP2-512'), 768
    if model_name == 'qwen3vl':  # placeholder, never used
        return None, None
    return None


def fetch_tokenizers(model_name):
    """Return tokenizer class."""
    if model_name == 'clip':
        return ClipTokenizer()
    if model_name == 'florence2':
        return Florence2Tokenizer()
    if model_name == 'siglip2_256':
        return None  # implement this later
    if model_name == 'siglip2_512':
        return None  # implement this later
    if model_name == 'qwen3vl':
        return Qwen3VLTokenizer()  # returns Qwen3VLTokenizerOutput (ids + attention_mask)
    return None
