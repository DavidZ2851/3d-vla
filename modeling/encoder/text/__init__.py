from .clip import ClipTextEncoder, ClipTokenizer
from .florence2 import Florence2Tokenizer
from .siglip2 import SigLip2TextEncoder


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
    return None
