import torch
from torch import nn
from timm.models.vision_transformer import RmsNorm


class QwenMultiScaleFusion(nn.Module):
    """Fuse same-grid Qwen vision block features with learned softmax weights."""

    def __init__(self, input_dim, output_dim, layer_ids):
        super().__init__()
        self.layer_ids = tuple(int(layer_id) for layer_id in layer_ids)
        self.norms = nn.ModuleList(
            [RmsNorm(input_dim) for _ in self.layer_ids]
        )
        self.projections = nn.ModuleList(
            [nn.Linear(input_dim, output_dim, bias=False) for _ in self.layer_ids]
        )
        self.layer_logits = nn.Parameter(
            torch.zeros(len(self.layer_ids), dtype=torch.float32)
        )

    def forward(self, feature_maps):
        projected = []
        for idx, layer_id in enumerate(self.layer_ids):
            if layer_id not in feature_maps:
                raise KeyError(
                    f"Missing Qwen multiscale feature for layer {layer_id}."
                )
            feats = feature_maps[layer_id]
            if feats.ndim != 5:
                raise ValueError(
                    f"Expected 5D feature map for layer {layer_id}, got {feats.shape}."
                )
            batch_size = feats.shape[0]
            feats = feats.reshape(batch_size, -1, feats.shape[-1])
            feats = self.projections[idx](self.norms[idx](feats))
            projected.append(feats)

        weights = torch.softmax(self.layer_logits, dim=0).to(
            dtype=projected[0].dtype,
            device=projected[0].device,
        )
        fused = torch.stack(projected, dim=0)
        fused = (fused * weights.view(-1, 1, 1, 1)).sum(dim=0)
        return fused
