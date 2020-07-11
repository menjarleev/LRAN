import torch
import torch.nn as nn
from model import ops
from torchvision.models import vgg19

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4, actv=nn.ReLU, padding_mode='reflect'):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        inter_channel = max(1, in_planes // ratio)
        self.fc1 = nn.Conv2d(in_planes, inter_channel, 1, bias=False, padding_mode=padding_mode)
        self.actv = actv()
        self.fc2 = nn.Conv2d(inter_channel, in_planes, 1, bias=False, padding_mode=padding_mode)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.actv(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.actv(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, padding_mode='reflect'):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, padding_mode=padding_mode, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class RALayer(nn.Module):
    def __init__(self, num_channels, num_block, kernel_size=7, reduction=16, padding_mode='reflect', actv=nn.ReLU):
        super(RALayer, self).__init__()
        self.conv1 = nn.Conv2d(num_channels * num_block, num_channels // 4, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode)
        self.actv = actv()
        self.conv2 = nn.Conv2d(num_channels // 4, num_channels, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode)
        self.channel_attention = ChannelAttention(num_channels, ratio=reduction, actv=actv, padding_mode=padding_mode)
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size, padding_mode=padding_mode)

    def forward(self, x):
        x = self.conv2(self.actv(self.conv1(x)))
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        return x


def get_LRAB_group(n_channel, n_block, LRAB_dict, res_scale=1, kernel_size=7, reduction=16, actv=nn.ReLU, padding_mode='reflect'):
    assert type(LRAB_dict) == dict
    if n_block & (n_block - 1) == 0 and n_block != 0:
        class Module(nn.Module):
            def __init__(self, num_channel, num_block, class_dict):
                super(Module, self).__init__()
                Module.__name__ = 'LRABx{}'.format(num_block)
                sub_num_block = num_block // 2
                submodule_name = 'LRABx{}'.format(sub_num_block)
                for i in range(2):
                    if sub_num_block == 1:
                        submodule = ops.ResBlock(num_channel, res_scale, activation=actv, padding_mode=padding_mode, res_out=True)
                    elif submodule_name in class_dict:
                        submodule = class_dict[submodule_name](num_channel, sub_num_block, class_dict)
                    else:
                        submodule = get_LRAB_group(num_channel, sub_num_block, class_dict)
                    setattr(self, 'submodule' + str(i + 1), submodule)
                self.res_module = nn.Sequential(nn.Conv2d(num_channel, num_channel, 3, 1, 1, padding_mode=padding_mode),
                                                actv(),
                                                nn.Conv2d(num_channel, num_channel, 3, 1, 1, padding_mode=padding_mode))
                self.res_attention = RALayer(num_channel, 3, kernel_size=kernel_size, reduction=reduction, padding_mode=padding_mode, actv=actv)

            def forward(self, feat):
                x = feat
                res_cat = None
                for i in range(2):
                    submodule_i = getattr(self, 'submodule' + str(i + 1))
                    x, x_res = submodule_i(x)
                    res_cat = x_res if res_cat is None else torch.cat([res_cat, x_res], dim=1)
                x_res = self.res_module(x)
                res_cat = torch.cat([res_cat, x_res], dim=1)
                res_agg = self.res_attention(res_cat)
                out = feat + res_agg
                return out, res_agg
        LRAB_class = Module
        LRAB_dict['LRABx{}'.format(n_block)] = LRAB_class
        return LRAB_class(n_channel, num_block=n_block, class_dict=LRAB_dict)
    else:
        raise ValueError('num_block must be a factor of 2')

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
        if opt.use_vgg:
            vgg_feat = vgg19(pretrained=True).features
            self.slice1 = nn.Sequential()
            for x in range(2):
                self.slice1.add_module(str(x + 1), vgg_feat[x])
            input_nc += 64
            for param in self.slice1.parameters():
                param.requires_grad = False
        self.use_vgg = opt.use_vgg
        if 'cutblur' in opt.augs:
            head = [ops.DownBlock(opt.scale),
                    nn.Conv2d(3*opt.scale**2, opt.num_channels, 3, 1, 1, padding_mode=padding_mode)]
        else:
            head = [nn.Conv2d(input_nc, opt.num_channels, 3, 1, 1, padding_mode=padding_mode)]
        assert ('cutblur' in opt.augs and opt.use_vgg) is False, 'can choose either vgg feat or cutblur'
        tail = [ops.Upsampler(opt.num_channels, opt.scale),
                nn.Conv2d(opt.num_channels, 3, 3, 1, 1, padding_mode=padding_mode)]
        if opt.normalize:
            tail += [nn.Tanh()]
        self.head = nn.Sequential(*head)
        self.body = get_LRAB_group(n_channel=opt.num_channels, n_block=opt.num_blocks,
                                   LRAB_dict={}, res_scale=opt.res_scale, kernel_size=3,
                                   reduction=opt.reduction, actv=actv, padding_mode=padding_mode)
        self.tail = nn.Sequential(*tail)
        self.opt = opt

    def forward(self, x):
        if not self.normalize:
            x = self.sub_mean(x)
        if self.use_vgg:
            feat1 = self.slice1(x)
            x = torch.cat([x, feat1], dim=1)
        x = self.head(x)
        x, _ = self.body(x)
        x = self.tail(x)
        if not self.normalize:
            x = self.add_mean(x)
        return x


