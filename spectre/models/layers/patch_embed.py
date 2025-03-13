import math
from typing import Optional, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectre.utils.utils import to_3tuple


class PatchEmbed(nn.Module):
    """ 3D Image to Patch Embedding
    """
    dynamic_img_pad: torch.jit.Final[bool]

    def __init__(
            self,
            img_size: Optional[Union[int, Tuple[int, int, int]]] = (128, 128, 64),
            patch_size: Union[int, Tuple[int, int, int]] = (16, 16, 8),
            in_chans: int = 1,
            embed_dim: int = 768,
            norm_layer: Optional[Callable] = None,
            flatten: bool = True,
            bias: bool = True,
            strict_img_size: bool = True,
            dynamic_img_pad: bool = False,
    ):
        super().__init__()
        self.patch_size = to_3tuple(patch_size)
        if img_size is not None:
            self.img_size = to_3tuple(img_size)
            self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
            self.num_patches = math.prod(self.grid_size)
        else:
            self.img_size = None
            self.grid_size = None
            self.num_patches = None

        self.flatten = flatten
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, H, W, D = x.shape
        if self.img_size is not None:
            if self.strict_img_size:
                assert H == self.img_size[0], f"Input height ({H}) doesn't match model ({self.img_size[0]})."
                assert W == self.img_size[1], f"Input width ({W}) doesn't match model ({self.img_size[1]})."
                assert D == self.img_size[2], f"Input depth ({D}) doesn't match model ({self.img_size[2]})."
            elif not self.dynamic_img_pad:
                assert H % self.patch_size[0] == 0, \
                    f"Input height ({H}) should be divisible by patch size ({self.patch_size[0]})."
                assert W % self.patch_size[1] == 0, \
                    f"Input width ({W}) should be divisible by patch size ({self.patch_size[1]})."
                assert D % self.patch_size[2] == 0, \
                    f"Input depth ({D}) should be divisible by patch size ({self.patch_size[2]})."
        if self.dynamic_img_pad:
            pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
            pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
            pad_d = (self.patch_size[2] - D % self.patch_size[2]) % self.patch_size[2]
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHWD -> NLC
        x = self.norm(x)
        return x