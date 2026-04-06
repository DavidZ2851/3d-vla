import torch
from torch import nn
from timm.models.vision_transformer import RmsNorm


class QwenStagewiseMultiScaleFusion(nn.Module):
    """Project Qwen multiscale maps once, then fuse into static + stage memories."""

    def __init__(
        self,
        input_dim,
        output_dim,
        layer_ids,
        num_stages=3,
    ):
        super().__init__()
        self.layer_ids = tuple(int(layer_id) for layer_id in layer_ids)
        self.num_stages = int(num_stages)

        self.norms = nn.ModuleList([RmsNorm(input_dim) for _ in self.layer_ids])
        self.projections = nn.ModuleList(
            [nn.Linear(input_dim, output_dim, bias=False) for _ in self.layer_ids]
        )

        n_layers = len(self.layer_ids)
        self.static_layer_logits = nn.Parameter(torch.zeros(n_layers, dtype=torch.float32))
        self.stage_layer_logits = nn.Parameter(
            torch.zeros(self.num_stages, n_layers, dtype=torch.float32)
        )
        self._init_stage_logits()

    def _init_stage_logits(self):
        if len(self.layer_ids) == 1:
            with torch.no_grad():
                self.stage_layer_logits.fill_(0.0)
            return
        with torch.no_grad():
            pos = torch.linspace(0.0, 1.0, len(self.layer_ids), dtype=torch.float32)
            for stage_idx in range(self.num_stages):
                center = float(stage_idx) / float(max(1, self.num_stages - 1))
                # Smooth bias: each stage softly prefers its corresponding depth band.
                self.stage_layer_logits[stage_idx].copy_(-4.0 * (pos - center).pow(2))

    def _project_multiscale(self, feature_maps):
        projected = []
        for idx, layer_id in enumerate(self.layer_ids):
            if layer_id not in feature_maps:
                raise KeyError(f"Missing Qwen multiscale feature for layer {layer_id}.")
            feats = feature_maps[layer_id]
            if feats.ndim != 5:
                raise ValueError(
                    f"Expected 5D feature map for layer {layer_id}, got {feats.shape}."
                )
            batch_size = feats.shape[0]
            feats = feats.reshape(batch_size, -1, feats.shape[-1])
            feats = self.projections[idx](self.norms[idx](feats))
            projected.append(feats)
        return projected

    def forward(self, feature_maps):
        projected = self._project_multiscale(feature_maps)
        stacked = torch.stack(projected, dim=0)  # (L, B, N, C)

        static_weights = torch.softmax(self.static_layer_logits, dim=0).to(
            dtype=stacked.dtype, device=stacked.device
        )
        static_fused = (stacked * static_weights.view(-1, 1, 1, 1)).sum(dim=0)

        stage_weights = torch.softmax(self.stage_layer_logits, dim=-1).to(
            dtype=stacked.dtype, device=stacked.device
        )  # (S, L)
        stage_fused = torch.einsum("sl,lbnc->sbnc", stage_weights, stacked)
        stage_fused = [stage_fused[i] for i in range(self.num_stages)]
        return static_fused, stage_fused
