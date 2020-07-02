import torch.nn as nn
from .simple import Net as NLayerDiscriminator

class Net(nn.Module):
    def __init__(self, opt):
        super(Net, self).__init__()
        self.__name__ = 'netD'
        self.num_D = opt.num_D
        self.n_layers = opt.n_layer_D
        self.get_inter_feat = not opt.no_GAN_feat
        ndf_max = 64
        ndf = opt.ndf


        for i in range(self.num_D):
            netD = NLayerDiscriminator(opt, min(ndf_max, ndf * (2 ** (self.num_D - 1 - i))))
            if self.get_inter_feat:
                for j in range(self.n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.get_inter_feat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.get_inter_feat:
                model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
                         range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result
