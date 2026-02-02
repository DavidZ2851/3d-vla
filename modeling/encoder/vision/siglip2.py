from torch import nn
from torch.nn import functional as F
from open_clip import create_model_from_pretrained, get_tokenizer


class SigLip2(nn.Module):

    def __init__(self, model_id='hf-hub:timm/ViT-B-16-SigLIP2-512'):
        super().__init__()
        self.model, self.preprocess = create_model_from_pretrained(model_id)
        self.tokenizer = get_tokenizer(model_id)
        del self.model.text
        self.model.eval()

    def forward(self, image, normalize=False):
        tokens = self.model.visual.trunk.forward_features(image)
        if normalize:
            pooled = self.model.visual.trunk.forward_head(tokens)
            return F.normalize(pooled, dim=-1)
        return tokens


def siglip_transform(img):
    return 2 * img - 1


def load_siglip2_512():
    return SigLip2('hf-hub:timm/ViT-B-16-SigLIP2-512'), siglip_transform


def load_siglip2_256():
    return SigLip2('hf-hub:timm/ViT-B-16-SigLIP2-256'), siglip_transform
