# Copyright (c) OpenMMLab. All rights reserved.
import timm
from mmcv.cnn.bricks.registry import NORM_LAYERS
from mmcv.runner import BaseModule
from mmcv.utils.parrots_wrapper import SyncBatchNorm
from ..builder import BACKBONES
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


@BACKBONES.register_module()
class UnetBackbone(BaseModule):
    def __init__(self, in_channels=3, channel_list=[64, 128, 256, 512], context_layer=None, coord_att=False, transformer_block=False, **kwargs):
        super(UnetBackbone, self).__init__(**kwargs)
        self.inc = InConv(in_channels, channel_list[0])
        self.down1 = Down(channel_list[0], channel_list[1], coord_att=coord_att)
        self.down2 = Down(channel_list[1], channel_list[2], coord_att=coord_att)
        self.down3 = Down(channel_list[2], channel_list[3], coord_att=coord_att)
        self.down4 = Down(channel_list[3], channel_list[3], coord_att=coord_att)

        self.context_layer = context_layer
        self.coord_att = coord_att
        self.transformer_block = transformer_block
        if self.context_layer == "kernelselect":
            self.context_layer1_1 = KernelSelectAttention(channel=channel_list[0])
            self.context_layer2_1 = KernelSelectAttention(channel=channel_list[1])
            self.context_layer3_1 = KernelSelectAttention(channel=channel_list[2])
        if self.transformer_block:
            self.aspp4 = TransformerBlock(c1=512, c2=512, num_heads=2, num_layers=4)
            self.aspp5 = TransformerBlock(c1=512, c2=512, num_heads=2, num_layers=4)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        if self.context_layer:
            res_x1 = self.context_layer1_1(x1)
            res_x2 = self.context_layer2_1(x2)
            res_x3 = self.context_layer3_1(x3)
            x1 = x1 + res_x1
            x2 = x2 + res_x2
            x3 = x3 + res_x3
        if self.transformer_block:
            x4 = self.aspp4(x4) + x4
            x5 = self.aspp5(x5) + x5
        return [x1, x2, x3, x4, x5]


class KernelSelectAttention(nn.Module):
    def __init__(self, channel=512, kernels=[3, 5, 7, 9], reduction=16, group=1, L=32):
        super(KernelSelectAttention, self).__init__()
        self.d = max(L, channel // reduction)
        self.convs = nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(channel, channel, kernel_size=k, padding=k // 2, groups=group),
                    SyncBatchNorm(channel),
                    nn.ReLU()
                )
            )
        self.fc = nn.Linear(channel, self.d)
        self.fcs = nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d, channel))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        bs, c, _, _ = x.size()
        conv_outs = []
        ### split
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats = torch.stack(conv_outs, 0)  # k,bs,channel,h,w

        ### fuse
        U = sum(conv_outs)  # bs,c,h,w

        ### reduction channel
        S = U.mean(-1).mean(-1)  # bs,c
        Z = self.fc(S)  # bs,d

        ### calculate attention weight
        weights = []
        for fc in self.fcs:
            weight = fc(Z)
            weights.append(weight.view(bs, c, 1, 1))  # bs,channel
        attention_weughts = torch.stack(weights, 0)  # k,bs,channel,1,1
        attention_weughts = self.softmax(attention_weughts)  # k,bs,channel,1,1

        ### fuse
        V = (attention_weughts * feats).sum(0)
        return V


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, coord_att=False):
        super(Down, self).__init__()
        self.coord_att = coord_att
        self.down_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.down_conv(x)
        return x


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            SyncBatchNorm(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            SyncBatchNorm(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = SyncBatchNorm(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return a_w * a_h



class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding=0, groups=g, dilation=d, bias=False)
        self.bn = SyncBatchNorm(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            # 如果TransformerBlock，即ViT模块输入和输出通道不同，提前通过一个卷积层让通道相同
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)