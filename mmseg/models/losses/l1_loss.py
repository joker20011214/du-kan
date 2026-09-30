# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES

_reduction_modes = ['none', 'mean', 'sum']


def l1_loss(pred, target, reduction):
    return F.l1_loss(pred, target, reduction=reduction)


@LOSSES.register_module()
class L1Loss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', loss_name='l1_loss', sample_wise=False):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.sample_wise = sample_wise
        self.loss_name = loss_name

    def forward(self, pred, target, weight=None, **kwargs):
        seg_prob = F.softmax(pred, dim=1)
        seg_pred = seg_prob.argmax(dim=1)
        B, H, W = seg_pred.shape
        seg_pred = seg_pred.view(B, 1, H, W)
        target = target.view(B, 1, H, W)
        target[target == 255] = 0
        loss_map = self.loss_weight * l1_loss(seg_pred, target, reduction=self.reduction)

        error_map = torch.where(target > 0, 50, 1)
        loss = (error_map * loss_map).mean()
        return loss * self.loss_weight
