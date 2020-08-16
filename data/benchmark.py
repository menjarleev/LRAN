import os
import glob
import data

class BenchmarkSR(data.BaseDataset):
    def __init__(self, phase, opt):
        root = opt.dataset_root
        self.scale = opt.scale
        dir_HQ, dir_LQ = self.get_subdir()
        self.HQ_paths = sorted(glob.glob(os.path.join(root, dir_HQ, '*.png')))
        self.LQ_paths = sorted(glob.glob(os.path.join(root, dir_LQ, '*.png')))

        super(BenchmarkSR, self).__init__(phase, opt)

    def get_subdir(self):
        dir_HQ = 'x{}'.format(self.scale)
        dir_LQ = 'x{}'.format(self.scale)
        return dir_HQ, dir_LQ

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
