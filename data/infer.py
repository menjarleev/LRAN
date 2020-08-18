import glob
import torch
import os
import skimage.io as io
from util.tensor_process import im2tensor, normalize

class InferDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        super(InferDataset, self).__init__()
        root = opt.root
        name = opt.infer_name
        degradation = opt.degradation
        self.scale = opt.scale
        self.LQ_paths = sorted(glob.glob(os.path.join(root, 'LR', 'LR{}'.format(degradation),
                                                      name, 'x{}'.format(self.scale), '*.png')))
        self.opt = opt

    def __getitem__(self, idx):
        path = self.LQ_paths[idx]
        LQ = io.imread(path)
        LQ = im2tensor(LQ)
        if self.opt.normalize:
            LQ = normalize(LQ)
        return [LQ, path]

    def __len__(self):
        return len(self.LQ_paths)
