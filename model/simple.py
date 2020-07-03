import torch
from .ops import *

class Net(torch.nn.Module):
    def __init__(self, opt, ndf):
        super(Net, self).__init__()
        input_nc = 3
        actv = get_layer('actv', opt.actv_D, opt.slope_D)
        conv = get_layer('conv', opt.conv_layer_D)
        self.n_layer = opt.n_layer_D
        self.get_inter_feat = not opt.no_GAN_feat
        padding_mode = opt.padding_D

        sequence = [[ conv(input_nc, ndf, kernel_size=3, stride=2, padding_mode=padding_mode),
                     actv()]]
        nf = ndf
        for n in range(1, self.n_layer):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                conv(nf_prev, nf, kernel_size=3, stride=2, padding_mode=padding_mode),
                actv()
            ]]
        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            conv(nf_prev, nf, kernel_size=3, stride=1, padding_mode=padding_mode),
            actv()
        ]]
        sequence += [[conv(nf, 1, kernel_size=3, stride=1, padding_mode=padding_mode)]]

        if self.get_inter_feat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.get_inter_feat:
            res = [input]
            for n in range(self.n_layer + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)


