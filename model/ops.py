import torch
import torch.nn as nn
import math
import functools

class MeanShift(nn.Conv2d):
    def __init__(
            self,
            rgb_range, sign=-1,
            rgb_mean=(0.4294, 0.4267, 0.4021), rgb_std=(1.0, 1.0, 1.0),
    ):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class ResBlock(nn.Module):
    def __init__(self, num_channels, res_scale=1.0, activation=nn.ReLU, padding_mode='reflect', res_out=False):
        super(ResBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, 3, 1, 1, padding_mode=padding_mode),
            activation(),
            nn.Conv2d(num_channels, num_channels, 3, 1, 1, padding_mode=padding_mode)
        )
        self.res_out = res_out
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        x = x + res
        if self.res_out:
            return x, res
        else:
            return x


class Upsampler(nn.Sequential):
    def __init__(self, num_channels, scale):
        m = list()
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m += [nn.Conv2d(num_channels, 4*num_channels, 3, 1, 1),
                      nn.PixelShuffle(2)]
        elif scale == 3:
            m += [nn.Conv2d(num_channels, 9*num_channels, 3, 1, 1),
                  nn.PixelShuffle(3)]
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


# use together with cutblur
class DownBlock(nn.Module):
    def __init__(self, scale):
        super(DownBlock, self).__init__()

        self.scale = scale

    def forward(self, x):
        n, c, h, w = x.size()
        x = x.view(n, c, h//self.scale, self.scale, w//self.scale, self.scale)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        x = x.view(n, c*(self.scale**2), h//self.scale, w//self.scale)
        return x


def get_layer(block_genre, *args):
    def get_res_block(res_block_type):
        if res_block_type == 'default':
            return ResBlock
        else:
            raise NotImplementedError('%s res block is not implemented' % block_type)

    def get_conv_block(block_type):
        if block_type == 'default':
            return nn.Conv2d
        else:
            raise NotImplementedError('%s conv block is not implemented' % block_type)

    def get_norm_layer(norm_type='instance'):
        # TODO implement SpectralNorm & syncBatchNorm
        if norm_type == 'batch':
            norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        elif norm_type == 'instance':
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=True)
        elif norm_type == 'None':
            norm_layer = None
        else:
            raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
        return norm_layer

    def get_activation(actv='LeakyReLU', slope=0.1, inplace=True):
        if actv == 'LeakyReLU':
            actv_func = functools.partial(nn.LeakyReLU, negative_slope=slope, inplace=inplace)
        elif actv == 'ReLU':
            actv_func = functools.partial(nn.ReLU, inplace=inplace)
        elif actv == 'Sigmoid':
            actv_func = nn.Sigmoid
        elif actv == 'Tanh':
            actv_func = nn.Tanh
        else:
            raise NotImplementedError('activation [%s] is not found' % actv)
        return actv_func


    def get_layer_type(t):
        if t == 'conv':
            return get_conv_block
        elif t == 'res':
            return get_res_block
        elif t == 'norm':
            return get_norm_layer
        elif t == 'actv':
            return get_activation
        else:
            return NotImplementedError('cannot find layer type %s' % t)

    layer_type = get_layer_type(block_genre)
    block_type = layer_type(*args)
    return block_type
