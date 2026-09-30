"""
References: https://github.com/xuebinqin/BASNet>pytorch_ssimi
"""

import torch
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES


def _iou(pred, target, size_average=True):
    b = pred.shape[0]
    IoU = 0.0
    for i in range(0, b):
        # compute the IoU of the foreground
        Iand1 = torch.sum(target[i, :, :, :] * pred[i, :, :, :])
        Ior1 = torch.sum(target[i, :, :, :]) + torch.sum(pred[i, :, :, :]) - Iand1
        IoU1 = Iand1 / Ior1

        # IoU loss is (1-IoU1)
        IoU = IoU + (1 - IoU1)

    return IoU / b


@LOSSES.register_module()
class IOULoss(nn.Module):
    def __init__(self, loss_weight=1, size_average=True, loss_name='iou_loss'):
        super(IOULoss, self).__init__()
        self.loss_weight = loss_weight
        self.size_average = size_average
        self.loss_name = loss_name

    def forward(self, pred, target, weight=None, **kwargs):
        loss = _iou(pred, target, self.size_average) * self.loss_weight
        return loss


class IOUWithLogitsLoss(IOULoss):
    def forward(self, input: Tensor, target: Tensor, mask=None) -> Tensor:
        input = torch.sigmoid(input)
        return super(IOUWithLogitsLoss, self).forward(input, target, mask)
