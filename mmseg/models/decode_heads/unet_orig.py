# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
import torch
from segmentation_models_pytorch.base import SegmentationHead
from ..builder import HEADS
from .decode_head import BaseDecodeHead


class UnetBackbone(nn.Module):
    def __init__(self, in_channels=3, channel_list=[8, 16, 16, 32, 64], context_layer=False, coord_att=False, tr=False, **kwargs):
        super(UnetBackbone, self).__init__(**kwargs)
        self.inc = InConv(in_channels, channel_list[0])
        self.down1 = Down(channel_list[0], channel_list[1])
        self.down2 = Down(channel_list[1], channel_list[2])
        self.down3 = Down(channel_list[2], channel_list[3])
        self.down4 = Down(channel_list[3], channel_list[4])

        self.context_layer = context_layer
        if self.context_layer:
            self.conv_inflatten = nn.Conv2d(1984, channel_list[0], kernel_size=1, stride=1, padding=0)
            self.context_layer_model = KernelSelectAttention(channel_list[0])

        self.coord_att = coord_att
        if self.coord_att:
            self.coord_att_model1 = CoordAtt(channel_list[1], channel_list[1])
            self.coord_att_model2 = CoordAtt(channel_list[2], channel_list[2])
            self.coord_att_model3 = CoordAtt(channel_list[3], channel_list[3])

        self.tr = tr
        if self.tr:
            self.tr_model = ASPP(channel_list[4], channel_list[4])

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        if self.context_layer:
            x5_cat = F.interpolate(x5, (x1.shape[2], x1.shape[3]), align_corners=True, mode="bilinear")
            x4_cat = F.interpolate(x4, (x1.shape[2], x1.shape[3]), align_corners=True, mode="bilinear")
            x3_cat = F.interpolate(x3, (x1.shape[2], x1.shape[3]), align_corners=True, mode="bilinear")
            x2_cat = F.interpolate(x2, (x1.shape[2], x1.shape[3]), align_corners=True, mode="bilinear")
            all_feature = torch.concat([x1, x2_cat, x3_cat, x4_cat, x5_cat], dim=1)
            inflatten = self.conv_inflatten(all_feature)
            ks_feature = self.context_layer_model(inflatten)
            x1 = ks_feature + x1
        if self.coord_att:
            x2 = self.coord_att_model1(x2) + x2
            x3 = self.coord_att_model2(x3) + x3
            x4 = self.coord_att_model3(x4) + x4

        if self.tr:
            x5 = self.tr_model(x5)
            # merge_feature = torch.concat([x5, tr_feature], dim=1)
            # x5 = self.merge(merge_feature)

        return [x1, x2, x3, x4, x5]


class KernelSelectAttention(nn.Module):
    def __init__(self, channel=512, kernels=[3, 5], reduction=16, group=1, L=32):
        super(KernelSelectAttention, self).__init__()
        self.d = max(L, channel // reduction)
        self.convs = nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(channel, channel, kernel_size=k, padding=k // 2, groups=group),
                    nn.BatchNorm2d(channel),
                    nn.ReLU(),
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
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
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
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
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


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UnetHead(nn.Module):
    def __init__(self, num_classes, decoder_channel=[64, 32, 16, 16, 8], **kwargs):
        super(UnetHead, self).__init__(**kwargs)

        self.up1 = Up(decoder_channel[0], decoder_channel[1])
        self.up2 = Up(decoder_channel[1] * 2, decoder_channel[2])
        self.up3 = Up(decoder_channel[2] * 2, decoder_channel[3])
        self.up4 = Up(decoder_channel[3] * 2, decoder_channel[4])

        self.head = SegmentationHead(decoder_channel[4] * 2, num_classes, kernel_size=1)

    def forward(self, inputs):
        out = self.up1(inputs[4], inputs[3])
        out = self.up2(out, inputs[2])
        out = self.up3(out, inputs[1])
        out = self.up4(out, inputs[0])
        return self.head(out)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        # 定义了self.up的方法
        super(Up, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)  # // 除以的结果向下取整

    def forward(self, x1, x2):  # x2是左侧的输出，x1是上一大层来的输出
        x1 = self.conv(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)  # 将两个tensor拼接在一起 dim=1：在通道数（C）上进行拼接
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
        self.bn1 = nn.BatchNorm2d(mip)
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
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class TransformerBlock(nn.Module):
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


# without BN version
class ASPP(nn.Module):
    def __init__(self, in_channel=512, out_channel=256):
        super(ASPP, self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1))  # (1,1)means ouput_dim
        self.conv = nn.Conv2d(in_channel,out_channel, 1, 1)
        self.atrous_block1 = nn.Conv2d(in_channel, out_channel, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, out_channel, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, out_channel, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, out_channel, 3, 1, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv2d(out_channel * 5, out_channel, 1, 1)

    def forward(self, x):
        size = x.shape[2:]

        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features, size=size, mode='bilinear')

        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)

        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        return net


@HEADS.register_module()
class UnetOrig(BaseDecodeHead):
    def __init__(self, in_channels=3,
                 context_layer=False,
                 coord_att=False,
                 tr=False,
                 out_channels=64,
                 **kwargs):
        super(UnetOrig, self).__init__(**kwargs)
        self.backbone = UnetBackbone(in_channels=in_channels, context_layer=context_layer, coord_att=coord_att, tr=tr)
        self.head = UnetHead(num_classes=out_channels)

    def forward(self, x):
        feature_list = self.backbone(x)
        out = self.head(feature_list)
        out = self.cls_seg(out)
        return out
