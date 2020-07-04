import importlib
import numpy as np
import skimage.io as io
import skimage.color as color
import torch
import util
from util.tensor_process import im2tensor, normalize

def generate_loader(phase ,opt):
    cname = opt.dataset.replace('_', '')
    if "DIV2K" in opt.dataset:
        mname = importlib.import_module('data.div2k')
    elif 'AIMSR' in opt.dataset:
        mname = importlib.import_module('data.aim')
    else:
        raise ValueError('Unsupported datasetL {}'.format(opt.dataset))

    kwargs = {
        'batch_size': opt.batch_size if phase == 'train' else 1,
        'num_workers': opt.num_workers if phase == 'train' else 0,
        'shuffle': phase == 'train',
        'drop_last': phase == 'train',
    }
    dataset = getattr(mname, cname)(phase, opt)
    return torch.utils.data.DataLoader(dataset, **kwargs)

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, phase, opt):

        self.phase = phase
        self.opt = opt

    def __getitem__(self, index):
        if self.phase == 'train':
            index = index % len(self.HQ_paths)


        HQ_path, LQ_path = self.HQ_paths[index], self.LQ_paths[index]
        HQ = io.imread(HQ_path)
        LQ = io.imread(LQ_path)
        if len(HQ.shape) < 3:
            HQ = color.gray2rgb(HQ)
        if len(LQ.shape) < 3:
            LQ = color.gray2rgb(LQ)

        if self.phase == 'train':
            inp_scale = HQ.shape[0] // LQ.shape[0]
            HQ, LQ = util.crop(HQ, LQ, self.opt.patch_size, inp_scale)
            HQ, LQ = util.flip_and_rotate(HQ, LQ)
        HQ, LQ = im2tensor([HQ, LQ])
        if self.opt.normalize:
            HQ, LQ = normalize([HQ, LQ])
        return HQ, LQ

    def __len__(self):
        if self.phase == 'train':
            if 'AIMSR' in self.opt.dataset:
                return len(self.HQ_paths) // self.opt.batch_size * self.opt.batch_size
            else:
                return (1000 * self.opt.batch_size) // len(self.HQ_paths) * len(self.HQ_paths)
        return len(self.HQ_paths)

