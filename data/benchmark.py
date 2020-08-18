import os
import glob
import data

class BenchmarkSR(data.BaseDataset):
    def __init__(self, phase, opt):
        root = opt.root
        self.scale = opt.scale
        name = opt.test_name
        degradation = opt.degradation
        self.HQ_paths = sorted(glob.glob(os.path.join(root, 'HR', name, 'x{}'.format(self.scale), '*.png')))
        self.LQ_paths = sorted(glob.glob(os.path.join(root, 'LR', 'LR{}'.format(degradation), name, 'x{}'.format(self.scale), '*.png')))

        super(BenchmarkSR, self).__init__(phase, opt)

class BenchmarkDN(BenchmarkSR):
    def __init__(self, phase, opt):
        self.sigma = opt.sigma

        super(BenchmarkDN, self).__init__(phase, opt)

    def get_subdir(self):
        dir_HQ = '{}'.format(self.sigma)
        dir_LQ = '{}'.format(self.sigma)
        return dir_HQ, dir_LQ

class BenchmarkJPEG(BenchmarkSR):
    def __init__(self, phase, opt):
        self.quality = opt.quality

        super(BenchmarkJPEG, self).__init__(phase, opt)

    def get_subdir(self):
        dir_HQ = '{}'.format(self.quality)
        dir_LQ = '{}'.format(self.quality)
        return dir_HQ, dir_LQ
