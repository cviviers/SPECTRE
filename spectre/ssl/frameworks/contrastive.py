"""
Implementation of the DINO framework for Self-Supervised Learning (SSL).

This module provides the necessary components to train the DINO framework an is based on the 
original paper: Caron et al., "Emerging Properties in Self-Supervised Vision Transformers" (2021), 
https://arxiv.org/abs/2104.14294
"""
from copy import deepcopy

import torch
import torch.nn as nn

from spectre.models import VisionTransformer
from spectre.ssl.models import MaskedVisionTransformer
from spectre.ssl.heads import DINOProjectionHead
from spectre.utils import deactivate_requires_grad_and_to_eval


class DINO(nn.Module):
    def __init__(
        self, 
        backbone: nn.Module, 
        input_dim: int,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        output_dim: int = 65536,
        freeze_last_layer: int = 1,
    ):
        super().__init__()

        self.backbone_student = backbone
        self.head_student = DINOProjectionHead(
            input_dim, hidden_dim, bottleneck_dim, output_dim, freeze_last_layer=freeze_last_layer,
        )

        self.backbone_teacher = deepcopy(backbone)
        self.head_teacher = DINOProjectionHead(
            input_dim, hidden_dim, bottleneck_dim, output_dim,
        )
        deactivate_requires_grad_and_to_eval(self.backbone_teacher)
        deactivate_requires_grad_and_to_eval(self.head_teacher)
    
    def forward_student(
        self, 
        global_crops: torch.Tensor, 
        local_crops: torch.Tensor
    ) -> torch.Tensor:
        cls_tokens_global = self.backbone_student(global_crops).flatten(start_dim=1)
        cls_tokens_local = self.backbone_student(local_crops).flatten(start_dim=1)

        cls_tokens_global_after_head = self.head_student(cls_tokens_global)
        cls_tokens_local_after_head = self.head_student(cls_tokens_local)

        return cls_tokens_global_after_head, cls_tokens_local_after_head
    
    @torch.no_grad()
    def forward_teacher(
        self, 
        global_crops: torch.Tensor
    ) -> torch.Tensor:
        cls_tokens = self.backbone_teacher(global_crops).flatten(start_dim=1)
        cls_tokens_after_head = self.head_teacher(cls_tokens)
        return cls_tokens_after_head
    
    def forward(
        self, 
        global_crops: torch.Tensor, 
        local_crops: torch.Tensor,
    ):
        student_global_cls_out, student_local_cls_out = self.forward_student(
            global_crops, local_crops)
        teacher_global_cls_out = self.forward_teacher(global_crops)
        return teacher_global_cls_out, student_global_cls_out, student_local_cls_out



class DINOv2(nn.Module):
    def __init__(
        self, 
        backbone: VisionTransformer, 
        input_dim: int,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        output_dim: int = 65536,
    ):
        super().__init__()

        self.student_backbone = MaskedVisionTransformer(vit=backbone)
        self.student_head_dino = DINOProjectionHead(
            input_dim, hidden_dim, bottleneck_dim, output_dim, freeze_last_layer=1,
        )
        self.student_head_ibot = DINOProjectionHead(
            input_dim, hidden_dim, bottleneck_dim, output_dim, freeze_last_layer=1,
        )

        self.teacher_backbone = deepcopy(backbone)
        self.teacher_head_dino = DINOProjectionHead(
            input_dim, hidden_dim, bottleneck_dim, output_dim,
        )
        self.teacher_head_ibot = DINOProjectionHead(
            input_dim, hidden_dim, bottleneck_dim, output_dim,
        )
        deactivate_requires_grad_and_to_eval(self.teacher_backbone)
        deactivate_requires_grad_and_to_eval(self.teacher_head_dino)
        deactivate_requires_grad_and_to_eval(self.teacher_head_ibot)

    def forward(
        self, 
        global_crops: torch.Tensor, 
        local_crops: torch.Tensor,
        masks: torch.Tensor,
        mask_indices: list, 
        upperbound: int,
    ) -> torch.Tensor:
        x_global = self.student_backbone.encode(global_crops, mask=masks)
        x_local = self.student_backbone.encode(local_crops)

        cls_tokens_global = x_global[:, 0]
        patch_tokens_global = x_global[:, 1:]
        cls_tokens_local = x_local[:, 0]

        buffer_tensor = patch_tokens_global.new_zeros(
            upperbound, patch_tokens_global.shape[-1])
        buffer_tensor[:mask_indices.shape[0]].copy_(torch.index_select(
            patch_tokens_global.flatten(0, 1),
            dim=0,
            index=mask_indices,
        ))

        cls_tokens_global_after_head = self.student_head_dino(cls_tokens_global)
        patch_tokens_global_after_head = self.student_head_ibot(buffer_tensor)[
            :mask_indices.shape[0]
        ]
        cls_tokens_local_after_head = self.student_head_dino(cls_tokens_local)
        
        return cls_tokens_global_after_head, patch_tokens_global_after_head, cls_tokens_local_after_head
    
    def forward_teacher(
            self, 
            global_crops: torch.Tensor, 
            mask_indices: list, 
            upperbound: int,
        ) -> torch.Tensor:
        x = self.teacher_backbone.forward_features(global_crops)
        cls_tokens = x[:, 0]
        patch_tokens = x[:, 1:]

        buffer_tensor = patch_tokens.new_zeros(
            upperbound, patch_tokens.shape[-1])
        torch.index_select(
            patch_tokens.flatten(0, 1),
            dim=0,
            index=mask_indices,
            out=buffer_tensor[:mask_indices.shape[0]],
        )

        cls_tokens_after_head = self.teacher_head_dino(cls_tokens)
        patch_tokens_after_head = self.teacher_head_ibot(buffer_tensor)[
            :mask_indices.shape[0]
        ]
           
        return cls_tokens_after_head, patch_tokens_after_head
