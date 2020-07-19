import torch
from typing import List
import torch.nn as nn
from model import ops
from torchvision.models import vgg19

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

    def forward(self, x):
        avg_out = self.fc2(self.actv(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.actv(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        x = self.sigmoid(out)
        return x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, padding_mode='reflect'):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.fc1 = nn.Conv2d(2, 1, kernel_size, padding=padding, padding_mode=padding_mode, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.fc1(x)
        x = self.sigmoid(x)
        return x


class AttentionLayer(nn.Module):
    def __init__(self, num_channels, reduction, kernel_size=7, padding_mode='zeros'):
        super(AttentionLayer, self).__init__()
        self.channel = ChannelAttention(num_channels, reduction, padding_mode)
        self.spatial = SpatialAttention(kernel_size, padding_mode)

    def forward(self, feat):
        channel_attn = self.channel(feat) * feat
        spatial_attn = self.spatial(feat) * feat
        x = channel_attn + spatial_attn
        return x


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size: List[int]=[3, 5], dilation: List[int]=[1,1], padding_mode='zeros'):
        super(InceptionModule, self).__init__()
        self.kernel_size = kernel_size
        self.d_size = dilation
        assert len(kernel_size) == len(dilation)
        group_size = out_channels // len(self.kernel_size)
        for d_size, k_size in zip(dilation, kernel_size):
            padding = (k_size + (k_size - 1) * (d_size - 1)) // 2
            setattr(self, 'kernel' + str(k_size) + '_dilation' + str(d_size), nn.Conv2d(in_channels, group_size,
                                                                                     kernel_size=k_size, stride=1,
                                                                                     dilation=d_size, padding=padding,
                                                                                     padding_mode=padding_mode))

    def forward(self, feat):
        x = None
        for d_size, k_size in zip(self.d_size, self.kernel_size):
            module = getattr(self, 'kernel' + str(k_size) + '_dilation' + str(d_size))
            x_i = module(feat)
            x = x_i if x is None else torch.cat([x, x_i], dim=1)
        return x


class ICAB(nn.Module):
    def __init__(self, in_channels, out_channels, reduction, kernel_size, dilation, actv=nn.ReLU, padding_mode='reflect', attention=True):
        super(ICAB, self).__init__()
        body = []
        body += [InceptionModule(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation, padding_mode=padding_mode)]
        if actv:
            body += [actv(inplace=True)]
        if attention:
            body += [AttentionLayer(out_channels, reduction, kernel_size=7, padding_mode=padding_mode)]
        self.body = nn.Sequential(*body)

    def forward(self, x):
        x = self.body(x)
        return x


class RICAB(nn.Module):
    def __init__(self, num_channels, reduction, kernel_size=[3, 3], dilation=[1, 2], actv=nn.ReLU, padding_mode='reflect'):
        super(RICAB, self).__init__()
        body = []
        body += [ICAB(num_channels, num_channels, reduction, kernel_size=kernel_size, dilation=dilation, actv=actv, padding_mode=padding_mode, attention=False)]
        body += [ICAB(num_channels, num_channels, reduction, kernel_size=kernel_size, dilation=dilation, actv=None, padding_mode=padding_mode, attention=True)]
        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class Group(nn.Module):
    def __init__(self, num_blocks, num_channels, reduction, kernel_size, dilation, actv=nn.ReLU, padding_mode='reflect'):
        super(Group, self).__init__()
        body = []
        for i in range(num_blocks):
            body += [RICAB(num_channels, reduction, kernel_size, dilation, actv, padding_mode)]
        body += [nn.Conv2d(num_channels, num_channels, 3, 1, 1, padding_mode=padding_mode)]
        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res



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
        for i in range(2):
            body += [RICAB(opt.num_channels, opt.reduction, kernel_size=[5, 7],
                           dilation=[1, 1], actv=actv, padding_mode=padding_mode)]
        for i in range(opt.num_groups):
            body += [Group(opt.num_blocks, opt.num_channels, opt.reduction, kernel_size=[3, 3], dilation=[1, 2], actv=actv, padding_mode=padding_mode)]
        body += [nn.Conv2d(opt.num_channels, opt.num_channels, 3, 1, 1, padding_mode=padding_mode)]
        tail = [ops.Upsampler(opt.num_channels, opt.scale),
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


