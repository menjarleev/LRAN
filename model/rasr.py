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
        self.actv = actv(inplace=True)
        self.fc2 = nn.Conv2d(inter_channel, in_planes, 1, bias=False, padding_mode=padding_mode)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feat):
        avg_out = self.fc2(self.actv(self.fc1(self.avg_pool(feat))))
        max_out = self.fc2(self.actv(self.fc1(self.max_pool(feat))))
        out = avg_out + max_out
        x = self.sigmoid(out)
        out = x * feat
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, padding_mode='reflect'):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, padding_mode=padding_mode, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feat):
        avg_out = torch.mean(feat, dim=1, keepdim=True)
        max_out, _ = torch.max(feat, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        x = self.sigmoid(x)
        out = x * feat
        return out



class ChannelAggBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16, actv=nn.ReLU, kernel_size=3, padding_mode='zeros'):
        super(ChannelAggBlock, self).__init__()
        body = [
            ChannelAttention(in_channels, reduction, actv, padding_mode),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, padding_mode=padding_mode),
            actv(inplace=True),
        ]
        self.body = nn.Sequential(*body)

    def forward(self, x):
        x = self.body(x)
        return x



class AttentionWrapper(nn.Module):
    def __init__(self, prev_layer, kernel_size=3, padding_mode='zeros'):
        super(AttentionWrapper, self).__init__()
        self.body = prev_layer
        self.attn = SpatialAttention(kernel_size, padding_mode)

    def forward(self, feat):
        x, res = self.body(feat)
        res_out = self.attn(res)
        return x, res_out



def get_RAB_group(n_channel, n_block, RAB_dict, res_scale=1, kernel_size=7, reduction=16,
                   actv=nn.ReLU, padding_mode='reflect', group_size=5):
    assert type(RAB_dict) == dict
    if n_block % group_size == 0:
        class Module(nn.Module):
            def __init__(self, num_channel, num_block, class_dict):
                super(Module, self).__init__()
                Module.__name__ = 'RABx{}'.format(num_block)
                sub_num_block = num_block // group_size
                submodule_name = 'RABx{}'.format(sub_num_block)
                self.group_size = group_size
                for i in range(self.group_size):
                    if sub_num_block == 1:
                        prev_layer = ops.ResBlock(num_channel, res_scale=res_scale, activation=actv, padding_mode=padding_mode, res_out=True)
                        submodule = AttentionWrapper(prev_layer, kernel_size, padding_mode)
                    elif submodule_name in class_dict:
                        prev_layer = class_dict[submodule_name](num_channel, sub_num_block, class_dict)
                        submodule = AttentionWrapper(prev_layer, kernel_size, padding_mode)
                    else:
                        submodule = get_RAB_group(num_channel, sub_num_block, class_dict, res_scale, kernel_size, reduction, actv, padding_mode, group_size)
                    setattr(self, 'submodule' + str(i + 1), submodule)
                self.agg = ChannelAggBlock(self.group_size * num_channel, num_channel, reduction=reduction, actv=actv, kernel_size=kernel_size, padding_mode=padding_mode)

            def forward(self, feat):
                x = feat
                res_cat = None
                for i in range(self.group_size):
                    submodule_i = getattr(self, 'submodule' + str(i + 1))
                    if i != self.group_size - 1:
                        x, x_res = submodule_i(x)
                    else:
                        _, x_res = submodule_i(x)
                    res_cat = x_res if res_cat is None else torch.cat([res_cat, x_res], dim=1)
                res_agg = self.agg(res_cat)
                out = feat + res_agg
                return out, res_agg
        RAB_class = Module
        RAB_dict['RABx{}'.format(n_block)] = RAB_class
        return RAB_class(n_channel, num_block=n_block, class_dict=RAB_dict)
    else:
        raise ValueError('num_block must be a factor of %s' % str(group_size))

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
        self.body = get_RAB_group(n_channel=opt.num_channels, n_block=opt.num_blocks,
                                   RAB_dict={}, res_scale=opt.res_scale, kernel_size=3,
                                   reduction=opt.reduction, actv=actv,
                                   padding_mode=padding_mode, group_size=opt.group_size)
        self.tail = nn.Sequential(*tail)
        self.opt = opt

    def forward(self, x):
        if not self.normalize:
            x = self.sub_mean(x)
        x = self.head(x)
        x, _ = self.body(x)
        x = self.tail(x)
        if not self.normalize:
            x = self.add_mean(x)
        return x


