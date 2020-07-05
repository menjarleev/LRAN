import torch
from .ops import *

class Net(torch.nn.Module):
    def __init__(self, opt, ndf):
        super(Net, self).__init__()
        input_nc = 3
        actv = get_layer('actv', opt.actv_D, opt.slope_D)
        conv = get_layer('conv', opt.conv_layer_D)
        norm = get_layer('norm', opt.norm_D)
        self.n_layer = opt.n_layer_D
        padding_mode = opt.padding_D
        if opt.normalize:
            sequence = [[conv(input_nc, ndf, kernel_size=3, stride=2, padding_mode=padding_mode),
                         norm(ndf),
                         actv()]]
        else:
            sequence = [[MeanShift(1),
                         conv(input_nc, ndf, kernel_size=3, stride=2, padding_mode=padding_mode),
                         norm(ndf),
                         actv()]]
        nf = ndf
        for n in range(1, self.n_layer):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                conv(nf_prev, nf, kernel_size=3, stride=2, padding_mode=padding_mode),
                norm(nf),
                actv()
            ]]
        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            conv(nf_prev, nf, kernel_size=3, stride=1, padding_mode=padding_mode),
            norm(nf),
            actv()
        ]]
        sequence += [[conv(nf, 1, kernel_size=3, stride=1, padding_mode=padding_mode)]]

        for n in range(len(sequence)):
            setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))

    def forward(self, input):
        res = [input]
        for n in range(self.n_layer + 2):
            model = getattr(self, 'model' + str(n))
            res.append(model(res[-1]))
        return res[1:]


