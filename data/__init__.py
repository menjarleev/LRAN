import importlib
import numpy as np
import skimage.io as io
import skimage.color as color
import torch
import util
from torchvision.transforms import Normalize

def generate_loader(phase ,opt):
    cname = opt.dataset.replace('_', '')
    if "DIV2K" in opt.dataset:
        mname = importlib.import_module('data.div2k')
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

        def im2tensor(im, rgb_range=255):
            if type(im) == list:
                return [im2tensor(i) for i in im]
            else:
                np_t = np.ascontiguousarray(im.transpose((2, 0, 1)))
                tensor = torch.from_numpy(np_t).float()
                tensor = tensor.div_(rgb_range)
                return tensor

        def normalize(tensor, mean=0.5, std=0.5):
            def _normalize(t):
                c, _, _ = t.size()
                if type(mean) == list:
                    _mean = mean
                else:
                    _mean = np.repeat(mean, c)
                if type(std) == list:
                    _std = std
                else:
                    _std = np.repeat(std, c)
                tensor = Normalize(_mean, _std)(t)
                return tensor

            if tensor is None:
                return None
            elif type(tensor) == list or type(tensor) == tuple:
                return [normalize(i) for i in tensor]
            else:
                return _normalize(tensor)

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
        HQ, LQ = im2tensor([HQ, LQ], self.opt.rgb_range)
        HQ, LQ = normalize([HQ, LQ])
        return HQ, LQ

    def __len__(self):
        if self.phase == 'train':
            return (1000 * self.opt.batch_size) // len(self.HQ_paths) * len(self.HQ_paths)
        return len(self.HQ_paths)

