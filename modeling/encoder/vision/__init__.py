from .clip import load_clip
from .florence2 import load_florence2
from .siglip2 import load_siglip2_256, load_siglip2_512


def fetch_visual_encoders(model_name):
    if model_name == "clip":
        return load_clip()
    if model_name == "florence2":
        return load_florence2()
    if model_name == "siglip2_256":
        return load_siglip2_256()
    if model_name == "siglip2_512":
        return load_siglip2_512()
