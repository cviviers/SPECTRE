from typing import Tuple, Union, Callable, Sequence

import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer as VisionTransformerTimm

from spectre.models.layers import PatchEmbed


class VisionTransformer(VisionTransformerTimm):
    """ Vision Transformer with 3D Patch Embedding
    """
    def __init__(
            self, 
            img_size: Union[int, Tuple[int, int, int]] = (128, 128, 64),
            patch_size: Union[int, Tuple[int, int, int]] = (16, 16, 8),
            in_chans: int = 1,
            embed_layer: Callable = PatchEmbed,
            *args, 
            **kwargs
        ) -> None:
        super().__init__(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=in_chans,
            embed_layer=embed_layer, 
            *args, 
            **kwargs
        )
    
    def get_intermediate_layers(
            self,
            x: torch.Tensor,
            n: Union[int, Sequence] = 1,
            reshape: bool = False,
            return_prefix_tokens: bool = False,
            norm: bool = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        # take last n blocks if n is an int, if in is a sequence, select by matching indices
        outputs = self._intermediate_layers(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        prefix_tokens = [out[:, 0:self.num_prefix_tokens] for out in outputs]
        outputs = [out[:, self.num_prefix_tokens:] for out in outputs]

        if reshape:
            grid_size = self.patch_embed.grid_size
            outputs = [
                out.reshape(x.shape[0], *grid_size, -1).permute(0, 4, 1, 2, 3).contiguous()
                for out in outputs
            ]

        if return_prefix_tokens:
            return tuple(zip(outputs, prefix_tokens))
        return tuple(outputs)
    
    def forward(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x, pre_logits=pre_logits)
        return x


def vit_tiny_patch16_128(*args, **kwargs) -> VisionTransformer:
    return VisionTransformer(
        img_size=(128, 128, 64),
        patch_size=(16, 16, 8),
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        *args,
        **kwargs
    )

def vit_small_patch16_128(*args, **kwargs) -> VisionTransformer:
    return VisionTransformer(
        img_size=(128, 128, 64),
        patch_size=(16, 16, 8),
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        *args,
        **kwargs
    )

def vit_base_patch16_128(*args, **kwargs) -> VisionTransformer:
    return VisionTransformer(
        img_size=(128, 128, 64),
        patch_size=(16, 16, 8),
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        *args,
        **kwargs
    )

def vit_base_patch32_128(*args, **kwargs) -> VisionTransformer:
    return VisionTransformer(
        img_size=(128, 128, 64),
        patch_size=(32, 32, 16),
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        *args,
        **kwargs
    )
