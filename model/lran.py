import torch
import torch.nn as nn
from model import ops
from torchvision.models import vgg19

class SpatialAttention(nn.Module):
    def __init__(self, num_block, kernel_size=7, padding_mode='reflect'):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        paddding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(num_block*2, num_block, kernel_size=kernel_size, padding=paddding, padding_mode=padding_mode, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.num_block = num_block

    def forward(self, block):
        assert type(block) == list and len(block) == self.num_block, 'input must be list and length %s' % str(self.num_block)
        x = None
        for b_i in block:
            avg_out = torch.mean(b_i, dim=1, keepdim=True)
            max_out, _ = torch.max(b_i, dim=1, keepdim=True)
            x_i = torch.cat([avg_out, max_out], dim=1)
            x = x_i if x is None else torch.cat([x, x_i], dim=1)
        x = self.conv(x)
        x = self.sigmoid(x)
        return torch.chunk(x, self.num_block, dim=1)


class ChannelAttention(nn.Module):
    def __init__(self, num_channel, num_block, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        total_channel = num_channel * num_block
        inter_channel = max(1, total_channel // reduction)
        self.fc1 = nn.Conv2d(total_channel, inter_channel, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(inter_channel, total_channel, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.num_block = num_block

    def forward(self, block):
        assert type(block) == list and len(block) == self.num_block, 'input must be list and length %s' % str(self.num_block)
        x = None
        for b_i in block:
            x = b_i if x is None else torch.cat([x, b_i], dim=1)
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        x = avg_out + max_out
        x = self.sigmoid(x)
        return torch.chunk(x, self.num_block, dim=1)


class RALayer(nn.Module):
    def __init__(self, num_channels, num_block, kernel_size=7, reduction=16, padding_mode='reflect'):
        super(RALayer, self).__init__()
        self.channel_attention = ChannelAttention(num_channels, num_block, reduction)
        self.spatial_attention = SpatialAttention(num_block, kernel_size=kernel_size, padding_mode=padding_mode)

    def forward(self, block):
        channel_map = self.channel_attention(block)
        spatial_map = self.spatial_attention(block)
        x = None
        for b_i, c_i, s_i in zip(block, channel_map, spatial_map):
            x_i = c_i * b_i
            x_i = s_i * x_i
            x = x_i if x is None else torch.cat([x_i, x], dim=1)
        return x


def get_LRAB_group(n_channel, n_block, LRAB_dict, res_scale=1, kernel_size=7, reduction=16,  actv=nn.ReLU,
                   padding_mode='reflect', bn_reduction=4, num_submodule=2):
    assert type(LRAB_dict) == dict
    if n_block & (n_block - 1) == 0 and n_block != 0:
        class Module(nn.Module):
            def __init__(self, num_channel, num_block, class_dict):
                super(Module, self).__init__()
                Module.__name__ = 'LRABx{}'.format(num_block)
                sub_num_block = num_block // 2
                submodule_name = 'LRABx{}'.format(sub_num_block)
                for i in range(num_submodule):
                    if sub_num_block == 1:
                        submodule = ops.ResBlock(num_channel, res_scale, activation=actv, padding_mode=padding_mode, res_out=True)
                    elif submodule_name in class_dict:
                        submodule = class_dict[submodule_name](num_channel, sub_num_block, class_dict)
                    else:
                        submodule = get_LRAB_group(num_channel, sub_num_block, class_dict)
                    setattr(self, 'submodule' + str(i + 1), submodule)
                bottleneck = bn_reduction
                self.res_attention = RALayer(num_channel, num_submodule, kernel_size=kernel_size, reduction=reduction, padding_mode=padding_mode)
                self.fusion = nn.Sequential(nn.Conv2d(num_channel * num_submodule, num_channel // bottleneck, 3, 1, 1, padding_mode=padding_mode),
                                                actv(),
                                                nn.Conv2d(num_channel // bottleneck, num_channel, 3, 1, 1, padding_mode=padding_mode))
                self.num_submodule = num_submodule

            def forward(self, feat):
                res_list = []
                x = feat
                for i in range(self.num_submodule):
                    submodule_i = getattr(self, 'submodule' + str(i + 1))
                    x, x_res = submodule_i(x)
                    res_list += [x_res]
                res_attn = self.res_attention(res_list)
                res = self.fusion(res_attn)
                out = feat + res
                return out, res

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
                                   reduction=opt.reduction, actv=actv, padding_mode=padding_mode,
                                   bn_reduction=opt.bn_reduction, num_submodule=opt.num_submodule)
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


