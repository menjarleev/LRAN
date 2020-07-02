import os
import glob
import data
import data

class DIV2KSR(data.BaseDataset):
    def __init__(self, phase, opt):
        root = opt.dataset_root

        self.scale = opt.scale
        dir_HQ, dir_LQ = self.get_subdir()
        self.HQ_paths = sorted(glob.glob(os.path.join(root, dir_HQ, '*.png')))
        self.LQ_paths = sorted(glob.glob(os.path.join(root, dir_LQ, '*.png')))
        split = [int(n) for n in opt.div2k_range.replace('/', '-').split('-')]
        if phase == 'train':
            s = slice(split[0]-1, split[1])
        else:
            s = slice(split[2]-1, split[3])
        self.HQ_paths, self.LQ_paths = self.HQ_paths[s], self.LQ_paths[s]
        super(DIV2KSR, self).__init__(phase, opt)

    def get_subdir(self):
        dir_HQ = 'x1'
        dir_LQ = 'x{}'.format(self.scale)
        return dir_HQ, dir_LQ
