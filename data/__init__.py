import importlib
import numpy as np
import os
import random
import skimage.io as io
import skimage.color as color
import torch
import util
from util.tensor_process import im2tensor, normalize

def generate_loader(opt, phase):
    if phase == 'train' or phase == 'validation':
        dataset_name = opt.train_type
    elif phase == 'test':
        dataset_name = opt.test_type
    elif phase == 'infer':
        dataset_name = opt.infer_type
    else:
        raise NotImplementedError('[%s] is not implemented' % phase)
    cname = dataset_name.replace('_', '')
    if "DIV2K" in dataset_name:
        mname = importlib.import_module('data.div2k')
    elif 'Benchmark' in dataset_name:
        mname = importlib.import_module('data.benchmark')
        if 'SR' in dataset_name:
            cname = 'BenchmarkSR'
        elif 'DN' in dataset_name:
            cname = 'BenchmarkDN'
        elif 'JPEG' in dataset_name:
            cname = 'BenchmarkJPEG'
    elif 'infer' in dataset_name:
        mname = importlib.import_module('data.infer')
        cname = 'InferDataset'
    else:
        mname = importlib.import_module('data.benchmark')
        cname = 'BenchmarkSR'

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
        else:
            h, w = LQ.shape[:-1]
            inp_scale = HQ.shape[0] // LQ.shape[0]
            HQ = HQ[0:inp_scale * h, 0:inp_scale * w]
        HQ, LQ = im2tensor([HQ, LQ])
        if self.opt.normalize:
            HQ, LQ = normalize([HQ, LQ])
        return HQ, LQ, HQ_path, LQ_path

    def __len__(self):
        if self.phase == 'train':
            return (1000 * self.opt.batch_size) // len(self.HQ_paths) * len(self.HQ_paths)
        return len(self.HQ_paths)

