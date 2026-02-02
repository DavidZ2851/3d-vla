from typing import OrderedDict

import torch
import torch.nn.functional as F
from torchvision.ops import FeaturePyramidNetwork


class EfficientFeaturePyramidNetwork(FeaturePyramidNetwork):

    def __init__(
        self,
        in_channels_list,
        out_channels,
        extra_blocks=None,
        norm_layer=None,
        output_level="res3"
    ):
        super().__init__(
            in_channels_list,
            out_channels,
            extra_blocks,
            norm_layer,
        )
        self.output_level = output_level

    def forward(self, x):
        """
        Computes the FPN for a set of feature maps.

        Args:
            x (OrderedDict[Tensor]): feature maps for each feature level.
            level_name: the level name to stop the FPN computation at. If None,
                the entire FPN is computed.

        Returns:
            results (OrderedDict[Tensor]): feature maps after FPN layers.
                They are ordered from the highest resolution first.
        """
        names = list(x.keys())
        x = [v.contiguous(memory_format=torch.channels_last) for v in x.values()]

        last_inner = self.get_result_from_inner_blocks(x[-1], -1)
        results = []
        results.append(self.get_result_from_layer_blocks(last_inner, -1))

        if names[-1] != self.output_level:
            for idx in range(len(x) - 2, -1, -1):
                inner_lateral = self.get_result_from_inner_blocks(x[idx], idx)
                feat_shape = inner_lateral.shape[-2:]
                inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
                last_inner = (inner_lateral + inner_top_down).contiguous()
                results.insert(0, self.get_result_from_layer_blocks(last_inner, idx))

                # Don't go over all levels to save compute
                if names[idx] == self.output_level:
                    names = names[idx:]
                    break
        else:
            names = names[-1:]

        if self.extra_blocks is not None:
            results, names = self.extra_blocks(results, x, names)

        # make it back an OrderedDict
        out = {
            k: v.contiguous(memory_format=torch.contiguous_format)
            for k, v in zip(names, results)
        }

        return out
