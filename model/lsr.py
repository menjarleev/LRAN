
import torch
from typing import List
import torch.nn as nn
from model import ops

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4, padding_mode='reflect'):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        inter_channel = max(1, in_planes // ratio)
        self.fc1 = nn.Conv2d(in_planes, inter_channel, 1, bias=False, padding_mode=padding_mode)
        self.actv = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(inter_channel, in_planes, 1, bias=False, padding_mode=padding_mode)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feat):
        avg_out = self.fc2(self.actv(self.fc1(self.avg_pool(feat))))
        max_out = self.fc2(self.actv(self.fc1(self.max_pool(feat))))
        out = avg_out + max_out
        x = self.sigmoid(out)
        return x * feat


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, padding_mode='reflect'):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.fc1 = nn.Conv2d(2, 1, kernel_size, padding=padding, padding_mode=padding_mode, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feat):
        avg_out = torch.mean(feat, dim=1, keepdim=True)
        max_out, _ = torch.max(feat, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.fc1(x)
        x = self.sigmoid(x)
        return feat * x

class ARB(nn.Module):
    def __init__(self, num_channels, res_scale, kernel_size, attn_type='channel', reduction=16, actv=nn.ReLU, padding_mode='zeros'):
        super(ARB, self).__init__()
        body = []
        padding = kernel_size // 2
        self.scale = res_scale
        body += [nn.Conv2d(num_channels, num_channels, kernel_size=kernel_size, stride=1, padding=padding, padding_mode=padding_mode),
                 actv(inplace=True),
                 nn.Conv2d(num_channels, num_channels, kernel_size=kernel_size, stride=1, padding=padding, padding_mode=padding_mode)]
        if attn_type == 'channel':
            body += [ChannelAttention(num_channels, reduction, padding_mode=padding_mode)]
        elif attn_type == 'spatial':
            body += [SpatialAttention(7, padding_mode=padding_mode)]
        else:
            raise NotImplementedError('attention [%s] is not implemented' % attn_type)
        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.body(x).mul(self.scale)
        return x + res







class ExtractBlock(nn.Module):
    def __init__(self, num_channels, reduction, actv=nn.ReLU, padding_mode='reflect', res_scale=1.0):
        super(ExtractBlock, self).__init__()
        self.conv = ARB(num_channels, res_scale, kernel_size=3, attn_type='spatial', reduction=reduction, actv=actv,
                   padding_mode=padding_mode)
        ex = []
        for i in range(5):
            ex += [ARB(num_channels, res_scale, kernel_size=1, attn_type='channel', reduction=reduction, actv=actv,
                       padding_mode=padding_mode)]
        ex += [nn.Conv2d(num_channels, num_channels, kernel_size=1, stride=1, padding=0, padding_mode=padding_mode)]
        self.ex = nn.Sequential(*ex)

    def forward(self, feat):
        x = self.conv(feat)
        res = self.ex(x)
        x = res + x
        return x


class Group(nn.Module):
    def __init__(self, num_blocks, num_channels, reduction, actv=nn.ReLU, padding_mode='reflect', res_scale=1.0):
        super(Group, self).__init__()
        body = []
        for i in range(num_blocks):
            body += [ExtractBlock(num_channels, reduction, actv, padding_mode, res_scale)]
        body += [nn.Conv2d(num_channels, num_channels, kernel_size=1, stride=1, padding=0, padding_mode=padding_mode)]
        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.body(x)
        x = res + x
        return x


class Net(nn.Module):
    def __init__(self, opt):
        super(Net, self).__init__()
        actv = ops.get_layer('actv', opt.actv_G, opt.slope_G)
        padding_mode = opt.padding_G
        if not opt.normalize:
            self.sub_mean = ops.MeanShift(1, rgb_mean=opt.rgb_mean)
            self.add_mean = ops.MeanShift(1, sign=1, rgb_mean=opt.rgb_mean)
        self.normalize = opt.normalize
        input_nc = 3
        if 'cutblur' in opt.augs:
            head = [ops.DownBlock(opt.scale),
                    nn.Conv2d(3*opt.scale**2, opt.num_channels, 3, 1, 1, padding_mode=padding_mode)]
        else:
            head = [nn.Conv2d(input_nc, opt.num_channels, 3, 1, 1, padding_mode=padding_mode)]
        body = []
        for i in range(opt.num_groups):
            body += [Group(opt.num_blocks, opt.num_channels, opt.reduction, actv,
                           padding_mode=opt.padding_G, res_scale=opt.res_scale)]
            body += [nn.Conv2d(opt.num_channels, opt.num_channels, 1, 1, 0, 1, padding_mode=padding_mode)]
        tail = [
                ops.Upsampler(opt.num_channels, opt.scale),
                nn.Conv2d(opt.num_channels, 3, 3, 1, 1, padding_mode=padding_mode)]
        if opt.normalize:
            tail += [nn.Tanh()]
        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)
        self.opt = opt

    def forward(self, x):
        if not self.normalize:
            x = self.sub_mean(x)
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        if not self.normalize:
            x = self.add_mean(x)
        return x


