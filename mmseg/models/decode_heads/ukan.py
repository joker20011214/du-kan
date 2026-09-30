import torch
import torch.nn.functional as F
import math
from functools import partial
from typing import Callable
import torch.utils.checkpoint as checkpoint
from ..builder import HEADS
from .decode_head import BaseDecodeHead

# 在 KANLinear 层中，B样条插值用于对输入特征进行非线性建模，并且这些插值函数是可学习的。这意味着通过训练，
# 模型不仅可以调整传统的线性权重（如全连接层中的权重），还可以通过训练样条函数的控制点，进一步增强模型的非线性表达能力。

# 自定义线性层，通过对输入特征进行基于网格的插值来建模复杂的非线性关系
class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,  # 样条网格的数量，用于样条插值，即分成多少个区间
        spline_order=3, # 样条的阶数，样条基函数的多项式阶数，阶数越高，样条的平滑性越好，但计算复杂度也更高。
        scale_noise=0.1, # 缩放噪声
        scale_base=1.0, # 基础缩放
        scale_spline=1.0, # 分段多项式的缩放
        enable_standalone_scale_spline=True, # 是否启用独立的样条缩放权重
        base_activation=torch.nn.SiLU, # 基础激活函数，默认为SiLU
        grid_eps=0.02, # 网格更新时的平滑系数
        grid_range=[-1, 1], # 网格范围
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # 创建网格
        h = (grid_range[1] - grid_range[0]) / grid_size # 计算每个网格的间隔 h = 0.4
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            #  expand 把一维张量扩展到二维，重复了 in_features 次
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)
        # 基础线性权重，用于处理输入 x 的线性部分
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        # 样条插值的权重，定义每个输出特征如何从 B 样条插值中构造
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    # 通过最小二乘法求解样条插值的系数
    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    # 将输入首先通过线性变换，然后通过 B样条插值进行进一步处理，最终将两者结果结合。
    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output

    @torch.no_grad()
    # 在每次前向传播时，可以通过 update_grid 方法对网格进行自适应更新，使得网格能够更好地拟合输入数据的分布
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    # 计算正则化损失，包含基于样条权重的 L1 正则化和熵损失，旨在限制样条权重的大小，从而避免模型过拟合。
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


# KAN 是一个包含多个 KANLinear 层的网络，每一层都继承了 KANLinear 的特性
# KAN 模型的架构可以看作是一个多层的非线性特征变换模型，其中每一层都是一个 KANLinear 层，该层结合了传统的线性层和样条插值
class KAN(torch.nn.Module):
    def __init__(
        self,
        # layers_hidden 中的每个元素表示对应层的 输入和输出特征数。
        # 假设 layers_hidden 传入的是一个列表 [64, 128, 256]，那么网络的结构可以理解为： KAN网络中会有两个KANLinear层
        # 输入层：输入的特征数为 64。
        # 第一层隐藏层：接收 64 个输入特征，输出 128 个特征。
        # 第二层隐藏层：接收 128 个输入特征，输出 256 个特征。
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        # 通过ModuleList来创建一个容器，专门用于存储和管理神经网络模块（层）
        self.layers = torch.nn.ModuleList()
        # 这一行使用 zip() 函数将 layers_hidden 列表中的元素两两配对。
        # 例如，如果 layers_hidden = [64, 128, 256]，zip(layers_hidden, layers_hidden[1:]) 会生成一个 (64, 128) 和 (128, 256) 的配对。
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]): # layers_hidden 中的每个元素和 layers_hidden[1:] 中对应位置的元素配对。
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )


import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt

import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import types
import math
from abc import ABCMeta, abstractmethod
# from mmcv.cnn import ConvModule
from pdb import set_trace as st
from torch.nn import init
import numpy as np
from timm.models.helpers import named_apply
from .mk_unet import (
    MultiKernelDepthwiseConv, gcd, channel_shuffle, act_layer, _init_weights)


# 论文中仅仅KANLayer的操作，包括三层KAN，以及每层KAN之后的深度可卷积分离
# act_layer激活函数
class KANLayer(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., no_kan=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features

        grid_size = 5
        spline_order = 3
        scale_noise = 0.1
        scale_base = 1.0
        scale_spline = 1.0
        base_activation = torch.nn.SiLU
        grid_eps = 0.02
        grid_range = [-1, 1]

        # 使用三层KANlinear
        if not no_kan:
            self.fc1 = KANLinear(
                in_features,
                hidden_features,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
            )
            self.fc2 = KANLinear(
                hidden_features,
                out_features,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
            )
            self.fc3 = KANLinear(
                hidden_features,
                out_features,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
            )
            # # TODO
            # self.fc4 = KANLinear(
            #             hidden_features,
            #             out_features,
            #             grid_size=grid_size,
            #             spline_order=spline_order,
            #             scale_noise=scale_noise,
            #             scale_base=scale_base,
            #             scale_spline=scale_spline,
            #             base_activation=base_activation,
            #             grid_eps=grid_eps,
            #             grid_range=grid_range,
            #         )

        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.fc3 = nn.Linear(hidden_features, out_features)

        # TODO
        # self.fc1 = nn.Linear(in_features, hidden_features)

        # 进行深度卷积
        self.dwconv_1 = DW_bn_relu(hidden_features)
        self.dwconv_2 = DW_bn_relu(hidden_features)
        self.dwconv_3 = DW_bn_relu(hidden_features)

        # # TODO
        # self.dwconv_4 = DW_bn_relu(hidden_features)

        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)  # 遍历所有子模块，并初始化它们的权重

    # 初始化权重
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):  # 线性层权重
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):  # 卷积层权重
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        # pdb.set_trace()
        B, N, C = x.shape

        x = self.fc1(x.reshape(B * N, C))
        x = x.reshape(B, N, C).contiguous()
        x = self.dwconv_1(x, H, W)
        x = self.fc2(x.reshape(B * N, C))
        x = x.reshape(B, N, C).contiguous()
        x = self.dwconv_2(x, H, W)
        x = self.fc3(x.reshape(B * N, C))
        x = x.reshape(B, N, C).contiguous()
        x = self.dwconv_3(x, H, W)

        # # TODO
        # x = x.reshape(B,N,C).contiguous()
        # x = self.dwconv_4(x, H, W)

        return x


# KANBlock中包含KAN层和深度卷积层，不包含patch_embedding,这个是单独操作的
class KANBlock(nn.Module):
    def __init__(self, dim, drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, no_kan=False):
        super().__init__()
        # drop_path防止过拟合
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim)

        self.layer = KANLayer(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                              no_kan=no_kan)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        # 残差连接
        x = x + self.drop_path(self.layer(self.norm2(x), H, W))

        return x


# 深度卷积
class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


# KANLayer中的深度可卷积分离
class DW_bn_relu(nn.Module):
    def __init__(self, dim=768):
        super(DW_bn_relu, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.flatten(2).transpose(1, 2)

        return x


# patch Embedding
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        # patch_embed3 = PatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        # img_size = 56 ，in_chans = 256 embed_dim = 320
        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]  # h=18  w=18
        self.num_patches = self.H * self.W  # 324
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))  # k = 3 s = 2 p = 1
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        # 送进来的x [4,256,28,28]
        x = self.proj(x)  # [4,256,14,14]
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # 从第二个维度开始展平 [4,196,320]
        x = self.norm(x)

        return x, H, W


# 编码器卷积块， 卷积 + 归一化 + relu
class ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvLayer, self).__init__()
        self.conv = nn.Sequential(
            # k为3，p为1，卷积后HW不变，通道改变，下采样通过最大池化操作实现，上采样通过双线性插值实现
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


# 解码器卷积块
class D_ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(D_ConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class SaELayer(nn.Module):
    def __init__(self, in_channel, reduction=2):
        super(SaELayer, self).__init__()
        assert in_channel >= reduction and in_channel % reduction == 0, 'invalid in_channel in SaElayer'
        self.reduction = reduction
        self.cardinality = 4
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # cardinality 1
        self.fc1 = nn.Sequential(
            nn.Linear(in_channel, in_channel // self.reduction, bias=False),
            nn.ReLU(inplace=True)
        )
        # cardinality 2
        self.fc2 = nn.Sequential(
            nn.Linear(in_channel, in_channel // self.reduction, bias=False),
            nn.ReLU(inplace=True)
        )
        # cardinality 3
        self.fc3 = nn.Sequential(
            nn.Linear(in_channel, in_channel // self.reduction, bias=False),
            nn.ReLU(inplace=True)
        )
        # cardinality 4
        self.fc4 = nn.Sequential(
            nn.Linear(in_channel, in_channel // self.reduction, bias=False),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Sequential(
            nn.Linear(in_channel // self.reduction * self.cardinality, in_channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y1 = self.fc1(y)
        y2 = self.fc2(y)
        y3 = self.fc3(y)
        y4 = self.fc4(y)
        y_concate = torch.cat([y1, y2, y3, y4], dim=1)
        y_ex_dim = self.fc(y_concate).view(b, c, 1, 1)

        return x * y_ex_dim.expand_as(x)


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
    def __init__(self, inp, oup, reduction=2):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        # self.bn1 = SyncBatchNorm(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        # y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return a_w * a_h


class VHeatMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0, channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        linear = partial(nn.Conv2d, kernel_size=1, padding=0) if channels_first else nn.Linear
        self.fc1 = linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class DepthwiseSeparableConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        padding_mode="zeros",
        bias=True,
    ):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            padding_mode=padding_mode,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class FourierUnit(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        fu_kernel=1,
        padding_type="reflect",
        fft_norm="ortho",
        use_only_freq=True,
        norm_layer=nn.BatchNorm2d,
        bias=True,
    ):
        super().__init__()
        self.fft_norm = fft_norm
        self.use_only_freq = use_only_freq

        self.inter_conv = nn.Sequential(
            DepthwiseSeparableConv(
                in_channels * 2,
                out_channels * 2,
                kernel_size=fu_kernel,
                stride=1,
                padding=get_padding(fu_kernel),
                padding_mode=padding_type,
                bias=bias,
            ),
            norm_layer(out_channels * 2),
            nn.ReLU(True),
        )

    def forward(self, x):
        input_dtype = x.dtype
        batch_size, ch, freq_dim, embed_dim = x.size()

        dims_to_fft = (-2,) if self.use_only_freq else (-2, -1)
        recover_length = (freq_dim,) if self.use_only_freq else (freq_dim, embed_dim)

        with torch.autocast(device_type=x.device.type, enabled=False):
            x_fft = x.float()
            self.inter_conv.float()
            fft_representation = torch.fft.rfftn(x_fft, dim=dims_to_fft, norm=self.fft_norm)
            fft_representation = torch.stack((fft_representation.real, fft_representation.imag), dim=2)

            ffted_dims = fft_representation.size()[-2:]
            fft_representation = fft_representation.view((batch_size, ch * 2) + ffted_dims)

            fft_representation = (
                self.inter_conv(fft_representation.float())
                .view((batch_size, ch, 2) + ffted_dims)
                .permute(0, 1, 3, 4, 2)
            )

            fft_representation = torch.complex(fft_representation[..., 0], fft_representation[..., 1])
            reconstructed_x = torch.fft.irfftn(
                fft_representation, dim=dims_to_fft, s=recover_length, norm=self.fft_norm
            )

        return reconstructed_x.to(input_dtype)


class SpectralTransform(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        fu_kernel=1,
        padding_type="reflect",
        fft_norm="ortho",
        use_only_freq=True,
        norm_layer=nn.BatchNorm2d,
        bias=False,
    ):
        super().__init__()
        halved_out_ch = out_channels // 2

        self.conv1 = nn.Sequential(
            DepthwiseSeparableConv(
                in_channels, halved_out_ch, kernel_size=1, stride=1, padding=0, padding_mode=padding_type, bias=bias
            ),
            norm_layer(halved_out_ch),
            nn.ReLU(True),
        )

        self.fu = FourierUnit(
            halved_out_ch,
            halved_out_ch,
            fu_kernel=fu_kernel,
            use_only_freq=use_only_freq,
            fft_norm=fft_norm,
            padding_type=padding_type,
            norm_layer=norm_layer,
        )

        self.conv2 = DepthwiseSeparableConv(
            halved_out_ch, out_channels, kernel_size=1, stride=1, padding=0, padding_mode=padding_type, bias=bias
        )

    def forward(self, x):
        residual = self.conv1(x)
        x = self.fu(residual)
        x = x + residual
        x = self.conv2(x)
        return x


class FastFourierConvolution(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        alpha_in=0.5,
        alpha_out=0.5,
        kernel_size=3,
        padding_type="reflect",
        fu_kernel=1,
        fft_norm="ortho",
        bias=True,
        norm_layer=nn.BatchNorm2d,
        activation=nn.ReLU(True),
        use_only_freq=True,
    ):
        super().__init__()
        self.global_in_channels = int(in_channels * alpha_in)
        self.local_in_channels = in_channels - self.global_in_channels
        self.global_out_channels = int(out_channels * alpha_out)
        self.local_out_channels = out_channels - self.global_out_channels

        padding = get_padding(kernel_size)

        tmp_module = self._get_module_on_true_predicate(
            self.local_in_channels > 0 and self.local_out_channels > 0, DepthwiseSeparableConv, nn.Identity
        )
        self.l2l_layer = tmp_module(
            self.local_in_channels,
            self.local_out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            padding_mode=padding_type,
            bias=bias,
        )

        tmp_module = self._get_module_on_true_predicate(
            self.local_in_channels > 0 and self.global_out_channels > 0, DepthwiseSeparableConv, nn.Identity
        )
        self.l2g_layer = tmp_module(
            self.local_in_channels,
            self.global_out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            padding_mode=padding_type,
            bias=bias,
        )

        tmp_module = self._get_module_on_true_predicate(
            self.global_in_channels > 0 and self.local_out_channels > 0, DepthwiseSeparableConv, nn.Identity
        )
        self.g2l_layer = tmp_module(
            self.global_in_channels,
            self.local_out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            padding_mode=padding_type,
            bias=bias,
        )

        tmp_module = self._get_module_on_true_predicate(
            self.global_in_channels > 0 and self.global_out_channels > 0, SpectralTransform, nn.Identity
        )
        self.g2g_layer = tmp_module(
            self.global_in_channels,
            self.global_out_channels,
            fu_kernel=fu_kernel,
            fft_norm=fft_norm,
            padding_type=padding_type,
            bias=bias,
            norm_layer=norm_layer,
            use_only_freq=use_only_freq,
        )

        self.local_bn_relu = (
            nn.Sequential(norm_layer(self.local_out_channels), activation)
            if self.local_out_channels != 0
            else nn.Identity()
        )
        self.global_bn_relu = (
            nn.Sequential(norm_layer(self.global_out_channels), activation)
            if self.global_out_channels != 0
            else nn.Identity()
        )

    @staticmethod
    def _get_module_on_true_predicate(condition, true_module=nn.Identity, false_module=nn.Identity):
        return true_module if condition else false_module

    def forward(self, x):
        x_l, x_g = x[:, : self.local_in_channels, ...], x[:, self.local_in_channels :, ...]
        x_l = 0 if x_l.size(1) == 0 else x_l
        x_g = 0 if x_g.size(1) == 0 else x_g

        out_local = torch.empty(0, device=x.device)
        out_global = torch.empty(0, device=x.device)

        if self.local_out_channels != 0:
            out_local = self.l2l_layer(x_l) + self.g2l_layer(x_g)
            out_local = self.local_bn_relu(out_local)

        if self.global_out_channels != 0:
            out_global = self.l2g_layer(x_l) + self.g2g_layer(x_g)
            out_global = self.global_bn_relu(out_global)

        return torch.cat((out_local, out_global), dim=1)


class Heat2D(nn.Module):
    def __init__(self, infer_mode=False, res=14, dim=96, hidden_dim=96, **kwargs):
        super().__init__()
        self.res = res
        self.dwconv = nn.Conv2d(dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.ffc = FastFourierConvolution(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            alpha_in=0.5,
            alpha_out=0.5,
            kernel_size=3,
            padding_type="reflect",
            fu_kernel=1,
            fft_norm="ortho",
            bias=True,
            norm_layer=nn.BatchNorm2d,
            activation=nn.ReLU(True),
            use_only_freq=True,
        )
        self.infer_mode = infer_mode
        self.to_k = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
        )

    @staticmethod
    def get_cos_map(N=224, device=torch.device("cpu"), dtype=torch.float):
        weight_x = (torch.linspace(0, N - 1, N, device=device, dtype=dtype).view(1, -1) + 0.5) / N
        weight_n = torch.linspace(0, N - 1, N, device=device, dtype=dtype).view(-1, 1)
        weight = torch.cos(weight_n * weight_x * torch.pi) * math.sqrt(2 / N)
        weight[0, :] = weight[0, :] / math.sqrt(2)
        return weight

    @staticmethod
    def get_decay_map(resolution=(224, 224), device=torch.device("cpu"), dtype=torch.float):
        resh, resw = resolution
        weight_n = torch.linspace(0, torch.pi, resh + 1, device=device, dtype=dtype)[:resh].view(-1, 1)
        weight_m = torch.linspace(0, torch.pi, resw + 1, device=device, dtype=dtype)[:resw].view(1, -1)
        weight = torch.pow(weight_n, 2) + torch.pow(weight_m, 2)
        weight = torch.exp(-weight)
        return weight

    def forward(self, x: torch.Tensor, freq_embed=None):
        B, C, H, W = x.shape
        x = self.dwconv(x)
        x = x + self.ffc(x)

        x = self.linear(x.permute(0, 2, 3, 1).contiguous())
        x, z = x.chunk(chunks=2, dim=-1)

        if ((H, W) == getattr(self, "__RES__", (0, 0))) and (
            getattr(self, "__WEIGHT_COSN__", None) is not None
        ) and (getattr(self, "__WEIGHT_COSN__", None).device == x.device):
            weight_cosn = getattr(self, "__WEIGHT_COSN__", None)
            weight_cosm = getattr(self, "__WEIGHT_COSM__", None)
            weight_exp = getattr(self, "__WEIGHT_EXP__", None)
        else:
            weight_cosn = self.get_cos_map(H, device=x.device).detach_()
            weight_cosm = self.get_cos_map(W, device=x.device).detach_()
            weight_exp = self.get_decay_map((H, W), device=x.device).detach_()
            setattr(self, "__RES__", (H, W))
            setattr(self, "__WEIGHT_COSN__", weight_cosn)
            setattr(self, "__WEIGHT_COSM__", weight_cosm)
            setattr(self, "__WEIGHT_EXP__", weight_exp)

        if freq_embed is None:
            # Fallback for integration into non-vHeat backbones.
            freq_embed = x

        N, M = weight_cosn.shape[0], weight_cosm.shape[0]

        x = F.conv1d(x.contiguous().view(B, H, -1), weight_cosn.contiguous().view(N, H, 1))
        x = F.conv1d(
            x.contiguous().view(-1, W, C),
            weight_cosm.contiguous().view(M, W, 1)
        ).contiguous().view(B, N, M, -1)

        k_map = self.to_k(freq_embed)
        if k_map.dim() == 4:
            # Adaptation path for non-vHeat backbones where freq_embed is BxHxWxC.
            decay = torch.pow(weight_exp[None, :, :, None], k_map)
        else:
            # Original vHeat path where freq_embed is HxWxC.
            decay = torch.pow(weight_exp[:, :, None], k_map)
        x = x * decay

        x = F.conv1d(x.contiguous().view(B, N, -1), weight_cosn.t().contiguous().view(H, N, 1))
        x = F.conv1d(
            x.contiguous().view(-1, M, C),
            weight_cosm.t().contiguous().view(W, M, 1)
        ).contiguous().view(B, H, W, -1)

        x = self.out_norm(x)
        x = x * nn.functional.silu(z)
        x = self.out_linear(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class HeatBlock(nn.Module):
    def __init__(
        self,
        res: int = 14,
        infer_mode=False,
        hidden_dim: int = 0,
        drop_path: float = 0.0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        use_checkpoint: bool = False,
        drop: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        mlp_ratio: float = 4.0,
        post_norm=True,
        layer_scale=None,
        **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = LayerNorm2d(hidden_dim)
        self.op = Heat2D(res=res, dim=hidden_dim, hidden_dim=hidden_dim, infer_mode=infer_mode)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = LayerNorm2d(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = VHeatMlp(
                in_features=hidden_dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=drop,
                channels_first=True
            )
        self.post_norm = post_norm
        self.layer_scale = layer_scale is not None

        if self.layer_scale:
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(hidden_dim), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(hidden_dim), requires_grad=True)

    def _forward(self, x: torch.Tensor, freq_embed=None):
        if not self.layer_scale:
            if self.post_norm:
                x = x + self.drop_path(self.norm1(self.op(x, freq_embed)))
                if self.mlp_branch:
                    x = x + self.drop_path(self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(self.op(self.norm1(x), freq_embed))
                if self.mlp_branch:
                    x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x

        if self.post_norm:
            x = x + self.drop_path(self.gamma1[:, None, None] * self.norm1(self.op(x, freq_embed)))
            if self.mlp_branch:
                x = x + self.drop_path(self.gamma2[:, None, None] * self.norm2(self.mlp(x)))
        else:
            x = x + self.drop_path(self.gamma1[:, None, None] * self.op(self.norm1(x), freq_embed))
            if self.mlp_branch:
                x = x + self.drop_path(self.gamma2[:, None, None] * self.mlp(self.norm2(x)))
        return x

    def forward(self, input: torch.Tensor, freq_embed=None):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input, freq_embed)
        return self._forward(input, freq_embed)


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

Max_L = 3


class ShiftResidual(nn.Module):
    """Shift residual branch adapted from BHViT shift idea."""

    def __init__(self, dim, init_scale=0.0):
        super().__init__()
        self.scale = nn.Parameter(torch.full((1, dim, 1, 1), init_scale))

    def forward(self, x, shift_h=0, shift_w=0, ch_half_shift=False):
        if ch_half_shift:
            c = x.size(1)
            left = torch.roll(x[:, : c // 2], shifts=1, dims=1)
            right = torch.roll(x[:, c // 2 :], shifts=-1, dims=1)
            shifted = torch.cat([left, right], dim=1)
        else:
            shifted = torch.roll(x, shifts=(shift_h, shift_w), dims=(2, 3))
        return shifted * self.scale


class MixStructureBlock(nn.Module):
    """Lightweight MixStructure block for NCHW features."""

    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.InstanceNorm2d(dim, affine=True)
        self.norm2 = nn.InstanceNorm2d(dim, affine=True)

        # Reduced kernel/dilation settings to keep params and FLOPs lower.
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect')
        self.conv3_a = nn.Conv2d(dim, dim, kernel_size=5, padding=4, groups=dim, dilation=2, padding_mode='reflect')
        self.conv3_b = nn.Conv2d(dim, dim, kernel_size=3, padding=2, groups=dim, dilation=2, padding_mode='reflect')
        self.conv3_c = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, dilation=1, padding_mode='reflect')

        self.Wv = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, padding_mode='reflect')
        )
        self.Wg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

        pa_mid = max(1, dim // 16)
        self.pa = nn.Sequential(
            nn.Conv2d(dim, pa_mid, 1, padding=0, bias=True),
            nn.GELU(),
            nn.Conv2d(pa_mid, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

        # Smaller expansion ratio (x2 instead of x4).
        self.mlp = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 2, 1),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, 1)
        )
        self.mlp2 = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 2, 1),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, 1)
        )

    def forward(self, x):
        identity = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.cat([self.conv3_a(x), self.conv3_b(x), self.conv3_c(x)], dim=1)
        x = self.mlp(x)
        x = identity + x

        identity = x
        x = self.norm2(x)
        x = torch.cat([self.Wv(x) * self.Wg(x), self.ca(x) * x, self.pa(x) * x], dim=1)
        x = self.mlp2(x)
        x = identity + x
        return x


class MKCoordAtt(nn.Module):
    """Coordinate attention for MKIR (identity-gated, from 2026-06-01-01)."""

    def __init__(self, inp, reduction=32, oup=None):
        super().__init__()
        oup = oup or inp
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        return identity * self.conv_h(x_h).sigmoid() * self.conv_w(x_w).sigmoid()


def _init_coord_att_identity(att):
    nn.init.zeros_(att.conv_h.weight)
    nn.init.zeros_(att.conv_w.weight)
    nn.init.constant_(att.conv_h.bias, 10.0)
    nn.init.constant_(att.conv_w.bias, 10.0)


class MKMultiKernelInvertedResidualBlock(nn.Module):
    """MK-UNet multi-kernel IRB with optional CoordAtt (from 2026-06-01-01)."""

    def __init__(
        self,
        in_c,
        out_c,
        stride,
        expansion_factor=2,
        dw_parallel=True,
        add=True,
        kernel_sizes=(1, 3, 5),
        activation='relu6',
        mkunet_residual=False,
        coord_att_reduction=None,
    ):
        super().__init__()
        assert stride in [1, 2]
        self.stride = stride
        self.in_c = in_c
        self.out_c = out_c
        self.kernel_sizes = list(kernel_sizes)
        self.add = add
        self.n_scales = len(self.kernel_sizes)
        self.mkunet_residual = mkunet_residual
        self.use_skip_connection = mkunet_residual and (stride == 1)
        self.ex_c = int(self.in_c * expansion_factor)
        self.pconv1 = nn.Sequential(
            nn.Conv2d(self.in_c, self.ex_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ex_c),
            act_layer(activation, inplace=True),
        )
        self.multi_scale_dwconv = MultiKernelDepthwiseConv(
            self.ex_c, self.kernel_sizes, self.stride, activation, dw_parallel=dw_parallel)
        self.combined_channels = self.ex_c if self.add else self.ex_c * self.n_scales
        self.pconv2 = nn.Sequential(
            nn.Conv2d(self.combined_channels, self.out_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.out_c),
        )
        self.conv1x1 = None
        if self.use_skip_connection and self.in_c != self.out_c:
            self.conv1x1 = nn.Conv2d(self.in_c, self.out_c, 1, 1, 0, bias=False)
        self.coord_att = (
            MKCoordAtt(self.combined_channels, reduction=coord_att_reduction)
            if coord_att_reduction is not None else None
        )
        self.init_weights('normal')
        if self.coord_att is not None:
            _init_coord_att_identity(self.coord_att)

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        pout1 = self.pconv1(x)
        dwconv_outs = self.multi_scale_dwconv(pout1)
        if self.add:
            dout = sum(dwconv_outs)
        else:
            dout = torch.cat(dwconv_outs, dim=1)
        dout = channel_shuffle(dout, gcd(self.combined_channels, self.out_c))
        if self.coord_att is not None:
            dout = self.coord_att(dout)
        out = self.pconv2(dout)
        if self.use_skip_connection:
            residual = self.conv1x1(x) if self.conv1x1 is not None else x
            return residual + out
        return out


class MKIR(nn.Module):
    """Stack of MK-UNet multi-kernel IR blocks for merge improvement."""

    def __init__(
        self,
        ch_in,
        ch_out,
        count,
        stage_num,
        att=False,
        expansion_factor=2,
        dw_parallel=True,
        add=True,
        kernel_sizes=(1, 3, 5),
        activation='relu6',
        use_coord_att=True,
        coord_att_reduction=32,
        downsample=True,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        ca_red = coord_att_reduction if use_coord_att else None
        for i in range(count):
            if downsample:
                stride = 2 if i == 0 and stage_num != 2 else 1
            else:
                stride = 1
            in_c = ch_in if i == 0 else ch_out
            self.blocks.append(
                MKMultiKernelInvertedResidualBlock(
                    in_c=in_c,
                    out_c=ch_out,
                    stride=stride,
                    expansion_factor=expansion_factor,
                    dw_parallel=dw_parallel,
                    add=add,
                    kernel_sizes=kernel_sizes,
                    activation=activation,
                    mkunet_residual=True,
                    coord_att_reduction=ca_red,
                )
            )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def _get_wav(in_channels, pool=True):
    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]
    harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
    harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
    harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
    harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H
    filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
    filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
    filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
    filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)
    net = nn.Conv2d if pool else nn.ConvTranspose2d
    filters = []
    for filt in (filter_LL, filter_LH, filter_HL, filter_HH):
        conv = net(in_channels, in_channels, kernel_size=2, stride=2, padding=0,
                   bias=False, groups=in_channels)
        conv.weight.requires_grad = False
        conv.weight.data = filt.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()
        filters.append(conv)
    return filters


class WavePool(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.LL, self.LH, self.HL, self.HH = _get_wav(in_channels)

    def forward(self, x):
        return self.LL(x), self.LH(x), self.HL(x), self.HH(x)


class MSCW(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        self.local_attn = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.global_attn = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        pool = torch.mean(x, dim=1, keepdim=True)
        return self.sigmoid(self.local_attn(x) + self.global_attn(pool))


class _DSConv3x3(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channel, in_channel, 3, stride, dilation, dilation=dilation,
                      groups=in_channel, bias=False),
            nn.BatchNorm2d(in_channel),
        ]
        if relu:
            layers.append(nn.ReLU(inplace=True))
        layers.extend([
            nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channel),
        ])
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class D2TMultiheadAttention(nn.Module):
    """D2T Multihead Attention from WPFormer (D2T_RECT_FIX_v2)."""

    def __init__(self, d_model, h, dropout=0.0):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.norm1 = nn.LayerNorm(d_model)
        self.pool = WavePool(d_model)
        self.self_attn1 = nn.MultiheadAttention(d_model, h, dropout=dropout, batch_first=True)
        self.mscw1 = MSCW(d_model=d_model)
        self.proto_size = 16
        self.conv3x3 = _DSConv3x3(d_model, d_model)
        self.Mheads = nn.Linear(d_model, self.proto_size, bias=False)
        self.mscw2 = MSCW(d_model=d_model)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def _pad_even(x):
        _, _, h, w = x.shape
        ph, pw = h % 2, w % 2
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode='replicate')
        return x, (h, w)

    def forward(self, x):
        b, c, h, w = x.shape
        n1 = h * w
        query = key = x.flatten(2).transpose(1, 2)
        feat, _ = self._pad_even(x)
        LL, HL, LH, HH = self.pool(feat)
        high_fre = HL + LH + HH
        low_fre = LL
        high_fre = high_fre.flatten(2).transpose(1, 2)
        low_fre = low_fre.flatten(2).transpose(1, 2)
        wei = self.mscw1(high_fre + low_fre)
        fre = wei * high_fre + low_fre
        x1 = self.self_attn1(query=query, key=fre, value=fre, attn_mask=None)[0]
        x1 = self.norm1(x1 + query)
        feat = self.conv3x3(x).flatten(2).transpose(1, 2)
        multi_heads_weights = F.softmax(
            self.Mheads(feat).view(b, n1, self.proto_size), dim=1)
        protos = multi_heads_weights.transpose(-1, -2) @ key
        protos_expanded = multi_heads_weights @ protos
        attn = self.mscw2(protos_expanded + query)
        x2 = query * attn + query
        x2 = self.norm2(x2)
        return (x1 + x2).transpose(1, 2).view(b, c, h, w)


class SaELayer(nn.Module):
    """Squeeze-and-Excitation Layer with 4 cardinality branches."""

    def __init__(self, in_channel, reduction=2):
        super().__init__()
        assert in_channel >= reduction and in_channel % reduction == 0
        self.reduction = reduction
        self.cardinality = 4
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        branch_dim = in_channel // self.reduction
        self.fc1 = nn.Sequential(nn.Linear(in_channel, branch_dim, bias=False), nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(nn.Linear(in_channel, branch_dim, bias=False), nn.ReLU(inplace=True))
        self.fc3 = nn.Sequential(nn.Linear(in_channel, branch_dim, bias=False), nn.ReLU(inplace=True))
        self.fc4 = nn.Sequential(nn.Linear(in_channel, branch_dim, bias=False), nn.ReLU(inplace=True))
        self.fc = nn.Sequential(
            nn.Linear(branch_dim * self.cardinality, in_channel, bias=True), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y_concate = torch.cat([self.fc1(y), self.fc2(y), self.fc3(y), self.fc4(y)], dim=1)
        return x * self.fc(y_concate).view(b, c, 1, 1).expand_as(x)


def _init_sae_identity(sae):
    linear = sae.fc[0]
    if isinstance(linear, nn.Linear):
        nn.init.zeros_(linear.weight)
        if linear.bias is not None:
            nn.init.constant_(linear.bias, 10.0)


class D2T(nn.Module):
    """D2T transformer layer from WPFormer (from 2026-06-01-01)."""

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0, act=nn.GELU(),
                 use_sae=False, sae_reduction=2):
        super().__init__()
        self.d2t_attn = D2TMultiheadAttention(c1, num_heads, dropout)
        self.fc1 = nn.Linear(c1, cm)
        self.fc2 = nn.Linear(cm, c1)
        self.norm = nn.LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.act = act
        self.use_sae = use_sae
        if self.use_sae:
            self.sae = SaELayer(c1, reduction=sae_reduction)
            _init_sae_identity(self.sae)

    def forward(self, x):
        c, h, w = x.shape[1:]
        x_attn = self.d2t_attn(x)
        x_attn = x_attn.flatten(2).transpose(1, 2)
        x_ffn = self.fc2(self.dropout(self.act(self.fc1(x_attn))))
        x_out = self.norm(x_attn + self.dropout(x_ffn))
        x_out = x_out.transpose(1, 2).view(x.shape[0], c, h, w).contiguous()
        if self.use_sae:
            x_out = self.sae(x_out)
        return x_out


class BHViTBlock(nn.Module):
    """BHViT-style block: attention + MLP + shift residuals."""

    def __init__(self, dim, num_heads=4, mlp_ratio=4.0, drop_path=0.0, shift3=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

        self.shift3 = shift3
        if self.shift3:
            self.shift_h3 = ShiftResidual(dim)
            self.shift_w3 = ShiftResidual(dim)
            self.shift_c3 = ShiftResidual(dim)
        self.mix = MixStructureBlock(dim)

    def forward(self, x):
        b, c, h, w = x.shape
        tokens = x.permute(0, 2, 3, 1).reshape(b, h * w, c)

        norm_tokens = self.norm1(tokens)
        attn_out, _ = self.attn(norm_tokens, norm_tokens, norm_tokens, need_weights=False)
        tokens = tokens + self.drop_path(attn_out)
        tokens = tokens + self.drop_path(self.mlp(self.norm2(tokens)))

        out = tokens.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        if self.shift3:
            out = out + self.shift_h3(x, shift_h=1, shift_w=0)
            out = out + self.shift_w3(x, shift_h=0, shift_w=1)
            out = out + self.shift_c3(x, ch_half_shift=True)
        out = self.mix(out)
        return out

import timm


class MaxVit(nn.Module):
    def __init__(self, num_classes=8, in_channels=3):  # 这里可以更改为你想要的模型
        super(MaxVit, self).__init__()

        # 创建模型时指定输入通道数，修改模型的第一个卷积层
        self.backbone = timm.create_model('maxvit_tiny_rw_256', features_only=True)

        # 修改第一层卷积层的输入通道数
        self.backbone.stem.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x):
        features = self.backbone(x)  # 提取所有特征层
        return features

@HEADS.register_module()
class UKAN(BaseDecodeHead):
    def __init__(self, img_size=256, patch_size=16, in_chans=3, embed_dims=[256, 320, 512], no_kan=False,
                 drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[1, 1, 1],
                 da=False, merge=False, tr=False, **kwargs):
        super(UKAN, self).__init__(**kwargs)
        self.da = da
        self.merge = merge
        self.tr = tr

        if self.da:
            self.maxvit = MaxVit(in_channels=Max_L)

        if self.da and not self.merge:
            # Dual-encoder fusion by lightweight 1x1 convs.
            self.fuse1 = nn.Conv2d(96, 32, kernel_size=1, stride=1, padding=0)
            self.fuse2 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)
            self.fuse3 = nn.Conv2d(384, 256, kernel_size=1, stride=1, padding=0)

        if self.merge:
            # Lightweight MKIR for skip refinement (no downsample, low memory).
            _light_merge = dict(
                count=1,
                att=False,
                downsample=False,
                expansion_factor=1,
                use_coord_att=False,
                kernel_sizes=(3,),
            )
            merge_hidden = (96, 128, 384) if self.da else (48, 64, 128)
            self.merge1 = MKIR(merge_hidden[0], 32, stage_num=2, **_light_merge)
            self.merge2 = MKIR(merge_hidden[1], 64, stage_num=3, **_light_merge)
            self.merge3 = MKIR(merge_hidden[2], 256, stage_num=4, **_light_merge)
            if not self.da:
                self.merge1_adapter = nn.Conv2d(32, merge_hidden[0], kernel_size=1, stride=1, padding=0)
                self.merge2_adapter = nn.Conv2d(64, merge_hidden[1], kernel_size=1, stride=1, padding=0)
                self.merge3_adapter = nn.Conv2d(256, merge_hidden[2], kernel_size=1, stride=1, padding=0)

        if self.tr:
            self.tr_model1 = D2T(320, cm=512, num_heads=4, use_sae=True, sae_reduction=2)
            self.tr_model2 = D2T(512, cm=512, num_heads=4, use_sae=True, sae_reduction=2)

        kan_input_dim = embed_dims[0]

        self.encoder1 = ConvLayer(Max_L, kan_input_dim // 8)
        self.encoder2 = ConvLayer(kan_input_dim // 8, kan_input_dim // 4)
        self.encoder3 = ConvLayer(kan_input_dim // 4, kan_input_dim)  # 三层卷积之后输出的通道为kan_input_dim 256

        # 编码器的norm
        self.norm3 = norm_layer(embed_dims[1])  # 320
        self.norm4 = norm_layer(embed_dims[2])  # 512

        # 解码器的norm
        self.dnorm3 = norm_layer(embed_dims[1])  # 320
        self.dnorm4 = norm_layer(embed_dims[0])  # 256

        # drop_path_rate: 随机深度（Drop Path）率，这是一种正则化技术，通过随机丢弃某些网络路径来增强模型的泛化能力。
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # 编码器的两个KANBlock，每个block中只含有一个KANBlock
        self.block1 = nn.ModuleList([KANBlock(
            dim=embed_dims[1],
            drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer
        )])

        self.block2 = nn.ModuleList([KANBlock(
            dim=embed_dims[2],
            drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer
        )])

        # 解码器的两个KANBlock
        self.dblock1 = nn.ModuleList([KANBlock(
            dim=embed_dims[1],
            drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer
        )])

        self.dblock2 = nn.ModuleList([KANBlock(
            dim=embed_dims[0],
            drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer
        )])

        # Tok-KAN中的下采样通过patch_embed来实现
        self.patch_embed3 = PatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                       embed_dim=embed_dims[1])
        self.patch_embed4 = PatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                       embed_dim=embed_dims[2])

        self.decoder1 = D_ConvLayer(embed_dims[2], embed_dims[1])  # 这些都是通道数的变化
        self.decoder2 = D_ConvLayer(embed_dims[1], embed_dims[0])
        self.decoder3 = D_ConvLayer(embed_dims[0], embed_dims[0] // 4)
        self.decoder4 = D_ConvLayer(embed_dims[0] // 4, embed_dims[0] // 8)
        self.decoder5 = D_ConvLayer(embed_dims[0] // 8, embed_dims[0] // 8)

    @staticmethod
    def _match_feature_size(src, ref):
        if src.shape[2:] != ref.shape[2:]:
            src = F.interpolate(src, size=ref.shape[2:], mode='bilinear', align_corners=False)
        return src

    def forward(self, x):
        B = x.shape[0]
        ### Encoder
        ### Conv Stage
        ### Stage 1
        # 外面经过一个最大池化层，下采样两倍
        out = F.relu(F.max_pool2d(self.encoder1(x), 2, 2))
        t1 = out
        # torch.Size([4, 32, 112, 112])
        ### Stage 2
        out = F.relu(F.max_pool2d(self.encoder2(out), 2, 2))
        t2 = out
        # torch.Size([4, 64, 56, 56]) 编码器2
        ### Stage 3
        out = F.relu(F.max_pool2d(self.encoder3(out), 2, 2))
        t3 = out
        # torch.Size([4, 256, 28, 28]) 编码器3之后，未进入tok
        if self.da:
            features = self.maxvit(x)
            f1 = self._match_feature_size(features[0], t1)
            f2 = self._match_feature_size(features[1], t2)
            f3 = self._match_feature_size(features[2], t3)
            if self.merge:
                t1 = self.merge1(torch.concat([t1, f1], dim=1))
                t2 = self.merge2(torch.concat([t2, f2], dim=1))
                t3 = self.merge3(torch.concat([t3, f3], dim=1))
            else:
                t1 = self.fuse1(torch.concat([t1, f1], dim=1))
                t2 = self.fuse2(torch.concat([t2, f2], dim=1))
                t3 = self.fuse3(torch.concat([t3, f3], dim=1))
        elif self.merge:
            t1 = self.merge1(self.merge1_adapter(t1))
            t2 = self.merge2(self.merge2_adapter(t2))
            t3 = self.merge3(self.merge3_adapter(t3))
        ### Tokenized KAN Stage
        ### Stage 4

        # 进入标记化的KAN块
        # 编码器第一个Tok——KAN
        out, H, W = self.patch_embed3(out)  # torch.Size([4, 196, 320]) embed3中下采样2倍
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = out
        # torch.Size([4, 320, 14, 14])
        ### Bottleneck
        # 编码器第二个Tok——KAN
        out, H, W = self.patch_embed4(out)  # embed4中下采样2倍  H x W:7x7
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # torch.Size([4, 512, 7, 7])
        if self.tr:
            t4 = self.tr_model1(t4) + t4
            out = self.tr_model2(out) + out
        ### Stage 4
        out = F.relu(
            F.interpolate(self.decoder1(out), scale_factor=(2, 2), mode='bilinear'))  # torch.Size([4, 320, 14, 14])
        out = torch.add(out, t4)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)  # torch.Size([4, 196, 320])
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)  # torch.Size([4, 196, 320])
        ### Stage 3
        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = F.relu(
            F.interpolate(self.decoder2(out), scale_factor=(2, 2), mode='bilinear'))  # torch.Size([4, 256, 28, 28])
        out = torch.add(out, t3)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)

        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)  # torch.Size([4, 784, 256])
        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t2)
        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t1)
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2), mode='bilinear'))

        out = self.cls_seg(out)
        return out


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UKAN(num_classes=1).to(device)
    x = torch.randn((4, 3, 224, 224)).to(device)
    pred = model(x)
    print(pred.shape)

