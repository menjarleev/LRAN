import json
import importlib
import torch
from option.option import get_option
from util.visualizer import Visualizer
from solver import Solver
import os

def main():
    opt = get_option()
    torch.manual_seed(opt.seed)
    generator = importlib.import_module("model.{}".format(opt.netG.lower()))
    discriminator = importlib.import_module('model.{}'.format(opt.netD.lower()))
    if not opt.test_only:
        file_name = os.path.join(opt.ckpt_root, opt.name, 'opt.txt')
        args = vars(opt)
        with open(file_name, 'w+') as opt_file:
            opt_file.write('--------------- Options ---------------\n')
            for k, v in args.items():
                opt_file.write('%s: %s\n' % (str(k), str(v)))
                print('%s: %s' % (str(k), str(v)))
            opt_file.write('----------------- End -----------------\n')

    solver = Solver(generator, discriminator, opt)
    if opt.test_only:
        print('Evaluate {} (loaded from {})'.format(opt.netG, opt.pretrain))
        psnr = solver.evaluate()
        print("{:.2f}".format(psnr))
    else:
        solver.fit()

if __name__ == '__main__':
    main()

