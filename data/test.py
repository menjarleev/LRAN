import data
import glob
import os

class TestSR(data.BaseDataset):
    def __init__(self, phase, opt):
        root = opt.dataroot_test
        self.scale = opt.scale
        dir_HQ, dir_LQ = self.get_subdir()
        self.HQ_paths = sorted(glob.glob(os.path.join(root, dir_HQ, '*.png')))
        self.LQ_paths = sorted(glob.glob(os.path.join(root, dir_LQ, '*.png')))

        if opt.test_range != 'inf':
            split = [int(n) for n in opt.test_range.split('-')]
            s = slice(split[0]-1, split[1])
            self.HQ_paths, self.LQ_paths = self.HQ_paths[s], self.LQ_paths[s]

        super(TestSR, self).__init__(phase, opt)

    def get_subdir(self):
        dir_HQ = 'x1'
        dir_LQ = 'x{}'.format(self.scale)
        return dir_HQ, dir_LQ
