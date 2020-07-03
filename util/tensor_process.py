import torch
import numpy as np
from torchvision.transforms import Normalize
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

