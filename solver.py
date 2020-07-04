import os
import time
import skimage.io as io
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import generate_loader
import util
from util import augments, loss_collector
from util.visualizer import Visualizer
from subprocess import call
from tqdm import tqdm, trange
import numpy as np
from util.tensor_process import tensor2im


class Solver:
    def __init__(self, netG, netD, opt):
        self.opt = opt
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.netG = netG.Net(opt).to(self.device)
        self.netD, self.optimG, self.optimD, self.schedulerG, self.schedulerD = [None] * 5
        self.state_object_name = ['netG',
                                  'schedulerG',
                                  'optimG',
                                  'netD',
                                  'schedulerD',
                                  'optimD']
        self.loss_collector = loss_collector.LossCollector(opt)
        Visualizer.log_print(opt, "# params of netG: {}".format(sum(map(lambda x: x.numel(), self.netG.parameters()))))
        self.save_dir = os.path.join(self.opt.ckpt_root, self.opt.name)
        os.makedirs(self.save_dir, exist_ok=True)
        torch.backends.cudnn.benchmark = True


        self.optimG = torch.optim.Adam(
            self.netG.parameters(), opt.lr,
            betas=(0.9, 0.999), eps=1e-8
        )

        self.schedulerG = torch.optim.lr_scheduler.MultiStepLR(
            self.optimG, [1000*int(d) for d in opt.decay.split('-')],
            gamma=opt.gamma,
        )

        if not opt.test_only:
            self.train_loader = generate_loader('train', opt)
            if 'GAN' in opt.loss_terms:
                self.netD = netD.Net(opt).to(self.device)
                Visualizer.log_print(opt, "# params of netD: {}".format(sum(map(lambda x: x.numel(), self.netD.parameters()))))
                self.optimD = torch.optim.Adam(
                    self.netD.parameters(), opt.lr,
                    betas=(0.9, 0.999), eps=1e-8
                )
                self.schedulerD = torch.optim.lr_scheduler.MultiStepLR(
                    self.optimD, [1000*int(d) for d in opt.decay.split('-')],
                    gamma=opt.gamma,
                )
        self.test_loader = generate_loader('test', opt)

        if opt.continue_train or opt.pretrain:
            self.load(opt.pretrain)

        if not opt.test_only and opt.amp != 'O0':
            from apex import amp
            Visualizer.log_print(opt, 'use amp optimization')
            if 'GAN' in opt.loss_terms:
                [self.netG, self.netD], [self.optimG, self.optimD] = amp.initialize([self.netG, self.netD],
                                                                                    [self.optimG, self.optimD],
                                                                                    opt_level=opt.amp, num_losses=2)
            else:
                self.netG, self.optimG = amp.initialize(self.netG, self.optimG, opt_level=opt.amp, num_losses=1)
        self.t1, self.t2 = None, None
        self.best_psnr, self.best_step = 0, 0

    def fit(self):
        opt = self.opt

        self.t1 = time.time()
        start = 0
        if opt.continue_train:
            iter_path = os.path.join(self.save_dir, 'iter_step{}.txt'.format(opt.step_label))
            if os.path.isfile(iter_path):
                start, self.best_psnr, self.best_step = np.loadtxt(iter_path, delimiter=',')
                start, self.best_step = int(start), int(self.best_step)
                Visualizer.log_print(opt, '========== Resuming from iteration %d with best psnr %.2f @ step %d ========'
                                     % (start, self.best_psnr, self.best_step))
            else:
                raise FileNotFoundError('iteration file at %s is not found' % iter_path)

        for step in tqdm(range(start, opt.max_steps), desc='train', leave=False):
            try:
                inputs = next(iters)
            except (UnboundLocalError, StopIteration):
                iters = iter(self.train_loader)
                inputs = next(iters)
            HR = inputs[0].to(self.device)
            LR = inputs[1].to(self.device)

            if 'cutblur' in opt.augs and HR.size() != LR.size():
                scale = HR.size(2) // LR.size(2)
                LR = F.interpolate(LR, scale_factor=scale, mode='nearest')

            HR, LR, mask, aug = augments.apply_augment(
                HR, LR,
                opt.augs, opt.prob, opt.alpha,
                opt.aux_alpha, opt.aux_alpha, opt.mix_p
            )

            SR = self.netG(LR)

            if aug == 'cutout':
                SR, HR = SR*mask, HR*mask

            loss_G, loss_D = [], []

            self.loss_collector.update_L1_weight(step)

            if 'GAN' in opt.loss_terms:
                self.loss_collector.compute_GAN_losses(self.netD, [SR, HR], for_discriminator=False)
            if 'VGG' in opt.loss_terms:
                self.loss_collector.compute_VGG_losses(SR, HR)
            if 'feat' in opt.loss_terms:
                self.loss_collector.compute_feat_losses(self.netD, [SR, HR])
            if 'L1' in opt.loss_terms:
                self.loss_collector.compute_L1_losses(SR, HR)

            self.loss_collector.loss_backward(self.loss_collector.loss_names_G, self.optimG, self.schedulerG, 0)

            if opt.gclip > 0:
                torch.nn.utils.clip_grad_value_(self.netG.parameters(), opt.gclip)

            if 'GAN' in opt.loss_terms:
                self.loss_collector.compute_GAN_losses(self.netD, [SR.detach(), HR], for_discriminator=True)
                self.loss_collector.loss_backward(self.loss_collector.loss_names_D, self.optimD, self.schedulerD, 1)

            loss_dict = {**self.loss_collector.loss_names_G, **self.loss_collector.loss_names_D}

            if (step + 1) % opt.eval_steps == 0 or opt.debug:
                self.summary_and_save(step, loss_dict)

    def summary_and_save(self, step, loss_dict):
        opt = self.opt
        if opt.print_mem:
            call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
        self.t2 = time.time()
        errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
        err_msg = ""
        for k, v in errors.items():
            if v != 0:
                err_msg += '%s: %.3f ' % (k, v)
        curr_lr = self.schedulerG.get_lr()[0]
        step, max_steps = step + 1, self.opt.max_steps
        eta = (self.t2 - self.t1) / opt.eval_steps * (max_steps - step) / 3600
        if opt.debug:
            psnr = 0
        else:
            psnr = self.evaluate()

        if psnr >= self.best_psnr:
           self.best_psnr, self.best_step = psnr, step
           self.save(step, self.best_psnr, self.best_step, 'best')
        message = '[{}K/{}K] psnr: {:.2f} (best psnr: {:.2f} @ {}K step) \n' \
                  '{}\n' \
                  'LR:{}, ETA:{:.1f} hours \n'.format(step // 1000, max_steps // 1000, psnr, self.best_psnr, self.best_step // 1000, err_msg, curr_lr, eta)
        Visualizer.log_print(opt, message)
        self.save(step, self.best_psnr, self.best_step, 'latest')
        self.t1 = time.time()

    @torch.no_grad()
    def evaluate(self):
        opt = self.opt
        self.netG.eval()

        if opt.save_result:
            save_root = os.path.join(opt.ckpt_root, opt.name, 'output', opt.dataset)
            os.makedirs(save_root, exist_ok=True)

        psnr = 0
        tqdm_test = tqdm(self.test_loader, desc='test', leave=False)
        for i, inputs in enumerate(tqdm_test):
            HR = inputs[0].to(self.device)
            LR = inputs[1].to(self.device)

            if 'cutblur' in opt.augs and HR.size() != LR.size():
                scale = HR.size(2) // LR.size(2)
                LR = F.interpolate(LR, scale_factor=scale, mode='nearest')

            SR = self.netG(LR).detach()
            HR, SR = tensor2im([HR, SR], normalize=opt.normalize)

            if opt.save_result:
                save_path = os.path.join(save_root, '{:04}.png'.format(i+1))
                io.imsave(save_path, SR)
            if opt.crop:
                HR = HR[opt.crop:-opt.crop, opt.crop:-opt.crop, :]
                SR = SR[opt.crop:-opt.crop, opt.crop:-opt.crop, :]
            if opt.eval_y_only:
                HR = util.rgb2ycbcr(HR)
                SR = util.rgb2ycbcr(SR)
            psnr += util.calculate_psnr(HR, SR)

        self.netG.train()

        return psnr/len(self.test_loader)

    def save(self, step, best_psnr, best_step, step_label):
        def update_stat_dict(state_object, state_name):
            if state_object is not None:
                if 'net' in state_name:
                    state_dict.update({state_name: state_object.cpu().state_dict()})
                    if torch.cuda.is_available():
                        state_object.to(self.device)
                else:
                    state_dict.update({state_name: state_object.state_dict()})
        opt = self.opt
        state_dict = dict()
        state_objects = [self.netG, self.schedulerG, self.optimG]
        if 'GAN' in opt.loss_terms:
            state_objects += [self.netD, self.schedulerD, self.optimD]
        for obj, name in zip(state_objects, self.state_object_name):
            update_stat_dict(obj, name)
        state_path = os.path.join(self.save_dir, 'state_step{}.pth'.format(step_label))
        torch.save(state_dict, state_path)
        iter_path = os.path.join(self.save_dir, 'iter_step{}.txt'.format(step_label))
        np.savetxt(iter_path, (step, best_psnr, best_step), delimiter=',')
        Visualizer.log_print(opt, 'save %s state at step %d' % (step_label, step))

    def load(self, pretrain):
        def load_network(network, pretrained_dict, name):
            opt = self.opt
            try:
                network.load_state_dict(pretrained_dict)
                Visualizer.log_print(opt, 'network %s loaded' % name)
            except:
                model_dict = network.state_dict()
                try:
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                    network.load_state_dict(pretrained_dict)
                    Visualizer.log_print(opt,
                                         'Pretrained network %s has excessive layers; Only loading layers that are used' % name)
                except:
                    Visualizer.log_print(opt,
                                         'Pretrained network %s has fewer layers; The following are not initialized:' % name)
                    not_initialized = set()
                    for k, v in pretrained_dict.items():
                        if v.size() == model_dict[k].size():
                            model_dict[k] = v

                    for k, v in model_dict.items():
                        if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                            not_initialized.add('.'.join(k.split('.')[:2]))
                    Visualizer.log_print(opt, sorted(not_initialized))
                    network.load_state_dict(model_dict)

        def load_other(state_object, pretrained_dict, name):
            try:
                state_object.load_state_dict(pretrained_dict)
                Visualizer.log_print(opt, '%s loaded' % name)
            except:
                Visualizer.log_print(opt,
                                     '%s parameters does not match, ignore loading' % name)
        opt = self.opt
        if pretrain:
            state_path = pretrain
        else:
            label = 'latest' if opt.continue_train else opt.step_label
            state_path = os.path.join(self.save_dir, 'state_step{}.pth'.format(label))
        if not os.path.isfile(state_path):
            Visualizer.log_print(opt, 'state file store in %s is not found' % state_path)
            return
        state = torch.load(state_path)
        state_objects = [self.netG]
        if not opt.test_only:
            state_objects += [self.schedulerG, self.optimG]
            if 'GAN' in opt.loss_terms:
                state_objects += [self.netD, self.schedulerD, self.optimD]
        for obj, name in zip(state_objects, self.state_object_name):
            if 'net' in name and name in state:
                load_network(obj, state[name], name)
                obj.cuda()
            elif name in state:
                load_other(obj, state[name], name)
            else:
                Visualizer.log_print(opt, '%s is not found in state' % name)



