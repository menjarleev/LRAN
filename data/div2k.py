import os
import glob
from data import BaseDataset

class DIV2KSR(BaseDataset):
    def __init__(self, phase, opt):
        root = opt.root
        name = opt.train_name
        degradation = opt.degradation

        self.scale = opt.scale
        self.HQ_paths = sorted(glob.glob(os.path.join(root, 'HR', name, 'x{}'.format(self.scale), '*.png')))
        self.LQ_paths = sorted(glob.glob(os.path.join(root, 'LR', 'LR{}'.format(degradation),
                                                      name, 'x{}'.format(self.scale), '*.png')))
        split = [int(n) for n in opt.image_range.replace('/', '-').split('-')]
        if phase == 'train':
            s = slice(split[0]-1, split[1])
        else:
            s = slice(split[2]-1, split[3])
        self.HQ_paths, self.LQ_paths = self.HQ_paths[s], self.LQ_paths[s]
        super(DIV2KSR, self).__init__(phase, opt)

    def get_subdir(self):
        dir_HQ = 'x{}'.format(self.scale)
        dir_LQ = 'x{}'.format(self.scale)
        return dir_HQ, dir_LQ
