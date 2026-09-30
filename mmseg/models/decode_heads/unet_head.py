# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.utils.parrots_wrapper import SyncBatchNorm
from ..builder import HEADS
from .decode_head import BaseDecodeHead
import torch.nn.functional as F


@HEADS.register_module()
class UnetHead(BaseDecodeHead):
    def __init__(self, decoder_channel=[1024, 512, 256, 128, 64], se=False, **kwargs):
        super(UnetHead, self).__init__(**kwargs)
        if se:
            self.up1 = Up(decoder_channel[0], int(decoder_channel[0] / 4), se=True)
            self.up2 = Up(decoder_channel[1], int(decoder_channel[1] / 4), se=True)
            self.up3 = Up(decoder_channel[2], int(decoder_channel[2] / 4), se=True)
            self.up4 = Up(decoder_channel[3], decoder_channel[4], se=True)
        else:
            self.up1 = Up(decoder_channel[0], int(decoder_channel[0] / 4))
            self.up2 = Up(decoder_channel[1], int(decoder_channel[1] / 4))
            self.up3 = Up(decoder_channel[2], int(decoder_channel[2] / 4))
            self.up4 = Up(decoder_channel[3], decoder_channel[4])

    def forward(self, inputs):
        out = self.up1(inputs[4], inputs[3])
        out = self.up2(out, inputs[2])
        out = self.up3(out, inputs[1])
        out = self.up4(out, inputs[0])
        output = self.cls_seg(out)
        return output


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True, se=False):
        # 定义了self.up的方法
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)  # // 除以的结果向下取整

        self.se = se
        if self.se:
            self.ca = MecaBlock(in_ch)
        self.conv = DoubleConv(in_ch, out_ch)


    def forward(self, x1, x2):  # x2是左侧的输出，x1是上一大层来的输出
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)  # 将两个tensor拼接在一起 dim=1：在通道数（C）上进行拼接
        if self.se:
            x = self.ca(x)
        x = self.conv(x)

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


class SeBlock(nn.Module):
    def __init__(self, channel, ratio=16):
        super(SeBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // ratio, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


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

class MecaBlock(nn.Module):
    def __init__(self, num_feature, ratio=4):
        super(MecaBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_conv = nn.Linear(num_feature, num_feature, bias=False)
        self.fc = nn.Sequential(
            nn.Linear(num_feature, num_feature // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_feature // ratio, num_feature, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        b, c, _, _ = input.size()
        y_avg_out = self.avg_pool(input).view(b, c)
        y_max_out = self.max_pool(input).view(b, c)
        avg_share_out = self.shared_conv(y_avg_out)
        max_share_out = self.shared_conv(y_max_out)
        out = avg_share_out + max_share_out
        score = self.fc(out).view(b, c, 1, 1)
        return score * input
