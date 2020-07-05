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
from skimage.metrics import structural_similarity
import json


class Solver:
    def __init__(self, netG, netD, opt):
        self.opt = opt
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.netG = netG.Net(opt).to(self.device)
        self.netD, self.optimG, self.optimD, self.schedulerG, self.schedulerD = [None] * 5
        self.module_dict = {'netG': self.netG}
        self.loss_collector = loss_collector.LossCollector(opt)
        Visualizer.log_print(opt, "# params of netG: {}".format(sum(map(lambda x: x.numel(), self.netG.parameters()))))
        self.save_dir = os.path.join(self.opt.ckpt_root, self.opt.name)
        os.makedirs(self.save_dir, exist_ok=True)
        torch.backends.cudnn.benchmark = True
        self.data = dict()
        self.data['best'] = {
            'ssim': 0,
            'psnr': 0,
            'score': -10,
            'step': 0,
        }
        self.data['latest'] = {
            'ssim': 0,
            'psnr': 0,
            'score': -10,
            'step': 0,
        }

        if not opt.test_only:
            self.optimG = torch.optim.Adam(
                self.netG.parameters(), opt.lr,
                betas=(0.9, 0.999), eps=1e-8
            )

            self.schedulerG = torch.optim.lr_scheduler.MultiStepLR(
                self.optimG, [1000*int(d) for d in opt.decay.split('-')],
                gamma=opt.gamma,
            )
            self.module_dict.update({'optimG': self.optimG,
                                     'schedulerG': self.schedulerG})

            self.train_loader = generate_loader(opt, 'train', opt.dataset)
            if 'gan' in opt.loss_terms:
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
                self.module_dict.update({'netD': self.netD,
                                         'optimD': self.optimD,
                                         'schedulerD': self.schedulerD})
        self.validation_loader = generate_loader(opt, 'validation', opt.dataset)

        if opt.continue_train or opt.pretrain:
            self.load(opt.pretrain, self.module_dict)

        if not opt.test_only and opt.amp != 'O0':
            from apex import amp
            Visualizer.log_print(opt, 'use amp optimization')
            if 'gan' in opt.loss_terms:
                [self.netG, self.netD], [self.optimG, self.optimD] = amp.initialize([self.netG, self.netD],
                                                                                    [self.optimG, self.optimD],
                                                                                    opt_level=opt.amp, num_losses=2)
            else:
                self.netG, self.optimG = amp.initialize(self.netG, self.optimG, opt_level=opt.amp, num_losses=1)
        self.t1, self.t2 = None, None

    def fit(self):
        opt = self.opt

        self.t1 = time.time()
        start = 0
        if opt.continue_train:
            json_path = os.path.join(self.save_dir, 'status.txt')
            if os.path.isfile(json_path):
                with open(json_path) as json_file:
                    data = json.load(json_file)
                # update self.data
                self.data.update(data)
                best = self.data['best']
                latest = self.data['latest']
                Visualizer.log_print(opt, '========== Resuming from iteration {}K with best psnr {:.2f} ssim {:.4f} score {:.4f} @ step {}K ========'
                                     .format(latest['step'] // 1000, best['psnr'], best['ssim'], best['score'], best['step'] // 1000))
            else:
                raise FileNotFoundError('iteration file at %s is not found' % json_path)

        if not opt.test_only and not opt.no_test_during_train:
            self.test_dir = self.save_dir
            os.makedirs(self.test_dir, exist_ok=True)
            self.test_loader = generate_loader(opt, 'test', opt.dataset_test)
            self.test_dict = {'netG': self.netG}
            try:
                test = self.data[opt.test_name]
                Visualizer.log_print(opt, '========== [{}] current best psnr {:.2f} ssim {:.4f} score {:.2f} @ step {}K '
                                     .format(opt.test_name, test['psnr'], test['ssim'], test['score'], test['step'] // 1000))
            except:
                Visualizer.log_print(opt, 'test result not found')
                self.data[opt.test_name] = {
                    'psnr': 0,
                    'ssim': 0,
                    'score': -10,
                    'step': 0,
                }
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

            GAN_fake, GAN_true = SR, HR
            if opt.dis_res:
                temp_LR = LR
                if HR.size() != LR.size():
                    scale = HR.size(2) // LR.size(2)
                    temp_LR = F.interpolate(LR, scale_factor=scale, mode='nearest')
                GAN_fake = GAN_fake - temp_LR
                GAN_true = GAN_true - temp_LR

            if not 'l1' == opt.loss_terms:
                self.loss_collector.update_L1_weight(step)

            if 'gan' in opt.loss_terms:
                self.loss_collector.compute_GAN_losses(self.netD, [GAN_fake, GAN_true], for_discriminator=False)
            if 'vgg' in opt.loss_terms:
                self.loss_collector.compute_VGG_losses(SR, HR)
            if 'feat' in opt.loss_terms:
                self.loss_collector.compute_feat_losses(self.netD, [GAN_fake, GAN_true])
            if 'l1' in opt.loss_terms:
                self.loss_collector.compute_L1_losses(SR, HR)

            self.loss_collector.loss_backward(self.loss_collector.loss_names_G, self.optimG, self.schedulerG, 0)

            if opt.gclip > 0:
                torch.nn.utils.clip_grad_value_(self.netG.parameters(), opt.gclip)

            if 'gan' in opt.loss_terms:
                self.loss_collector.compute_GAN_losses(self.netD, [GAN_fake.detach(), GAN_true], for_discriminator=True)
                self.loss_collector.loss_backward(self.loss_collector.loss_names_D, self.optimD, self.schedulerD, 1)

            loss_dict = {**self.loss_collector.loss_names_G, **self.loss_collector.loss_names_D}

            if (step + 1) % opt.eval_steps == 0 or opt.debug:
                self.summary_and_save(step, loss_dict)

            if (not opt.no_test_during_train and (step + 1) % opt.test_steps == 0) or opt.debug:
                self.test_and_save(step)

    def test_and_save(self, step):
        opt = self.opt
        step = step + 1
        psnr, ssim = self.evaluate(self.test_loader, 'test', opt.test_name)
        score = util.calculate_score(psnr, ssim)
        test = self.data[opt.test_name]
        current_test = {
            'psnr': psnr,
            'ssim': ssim,
            'score': score,
            'step': step
        }
        if util.evaluate(current_test, test, opt.eval_metric):
            self.data[opt.test_name] = current_test
            self.save(opt.test_name, self.test_dict)
            self.save_log_iter(opt.test_name)
        test = self.data[opt.test_name]
        msg = '[test {}] psnr {:.2f} ssim {:.4f} score {:.2f} (best psnr {:.2f} ssim {:.4f} score {:.4f} @ step {}K)'.format(
            opt.test_name, psnr, ssim, score, test['psnr'], test['ssim'], test['score'], test['step'] // 1000
        )
        Visualizer.log_print(opt, msg)


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

        psnr, ssim = self.evaluate(self.validation_loader, 'validation', opt.dataset)
        score = util.calculate_score(psnr, ssim)
        current_latest = {
            'psnr': psnr,
            'ssim': ssim,
            'score': score,
            'step': step
        }
        best = self.data['best']
        if util.evaluate(current_latest, best, opt.eval_metric):
            self.data['best'] = current_latest
            self.save('best', self.module_dict)
            self.save_log_iter('best')
        best = self.data['best']
        self.data['latest'] = current_latest
        message = '[{}K/{}K] psnr: {:.2f} ssim: {:.4f} score: {:.4f} (best psnr: {:.2f} ssim: {:.4f} score: {:.4f} @ {}K step) \n' \
                  '{}\n' \
                  'LR:{}, ETA:{:.1f} hours'.format(step // 1000, max_steps // 1000, psnr, ssim, score,
                                                   best['psnr'], best['ssim'], best['score'],
                                                   best['step'] // 1000, err_msg, curr_lr, eta)
        Visualizer.log_print(opt, message)
        self.save('latest', self.module_dict)
        self.save_log_iter('latest')
        self.t1 = time.time()


    @torch.no_grad()
    def evaluate(self, data_loader, phase, dataset_name):
        opt = self.opt
        self.netG.eval()

        if opt.save_result:
            save_root = os.path.join(self.save_dir, 'output', dataset_name)
            os.makedirs(save_root, exist_ok=True)

        psnr = 0
        ssim = 0
        tqdm_data_loader = tqdm(data_loader, desc=phase, leave=False)
        for i, inputs in enumerate(tqdm_data_loader):
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
            if phase == 'validation':
                psnr += util.calculate_psnr(HR, SR)
                ssim += structural_similarity(HR, SR, data_range=255, multichannel=True, gaussian_weights=True, K1=0.01, K2=0.03)
            else:
                HR = util.rgb2ycbcr(HR)
                SR = util.rgb2ycbcr(SR)
                psnr += util.calculate_psnr(HR, SR)
                ssim += structural_similarity(HR, SR, data_range=255, multichannel=False, gaussian_weights=True, K1=0.01, K2=0.03)

        self.netG.train()

        return psnr/len(data_loader), ssim/len(data_loader)

    def save_log_iter(self, label):
        json_path = os.path.join(self.save_dir, 'status.txt')
        with open(json_path, 'w') as json_file:
            json.dump(self.data, json_file)
        Visualizer.log_print(self.opt, 'update [{}] for status file'.format(label))

    def save(self, step_label, module_dict):
        def update_state_dict(dict):
            for k, v in dict.items():
                if 'net' in k:
                    state_dict.update({k: v.cpu().state_dict()})
                    if torch.cuda.is_available():
                        v.to(self.device)
                else:
                    state_dict.update({k: v.state_dict()})
        state_dict = dict()
        update_state_dict(module_dict)
        state_path = os.path.join(self.save_dir, 'state_{}.pth'.format(step_label))
        torch.save(state_dict, state_path)

    def load(self, pretrain, module_dict):
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
            label = opt.step_label
            state_path = os.path.join(self.save_dir, 'state_{}.pth'.format(label))
        if not os.path.isfile(state_path):
            Visualizer.log_print(opt, 'state file store in %s is not found' % state_path)
            return
        state = torch.load(state_path)
        for k, v in module_dict.items():
            if 'net' in k:
                load_network(v, state[k], k)
                v.cuda()
            else:
                load_other(v, state[k], k)



