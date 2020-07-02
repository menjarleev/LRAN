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


class Solver:
    def __init__(self, netG, netD, opt):
        self.opt = opt
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.netG = netG.Net(opt).to(self.device)
        self.loss_collector = loss_collector.LossCollector(opt)
        Visualizer.log_print(opt, "# params of netG: {}".format(sum(map(lambda x: x.numel(), self.netG.parameters()))))
        self.save_dir = os.path.join(self.opt.ckpt_root, self.opt.name)
        os.makedirs(self.save_dir, exist_ok=True)


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
            [self.netG, self.netD], [self.optimG, self.optimD] = amp.initialize([self.netG, self.netD],
                                                                                [self.optimG, self.optimD],
                                                                                opt_level=opt.amp, num_losses=2)
        self.t1, self.t2 = None, None
        self.best_psnr, self.best_step = 0, 0

    def fit(self):
        opt = self.opt

        self.t1 = time.time()
        for step in range(opt.max_steps):
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
            torch.autograd.set_detect_anomaly(True)
            loss_G_GAN = self.loss_collector.compute_GAN_losses(self.netD, [SR, HR], for_discriminator=False)
            loss_G_VGG = self.loss_collector.compute_VGG_losses(SR, HR)
            loss_G = loss_G_GAN + [loss_G_VGG]
            self.loss_collector.loss_backward(loss_G_GAN, self.optimG, self.schedulerG, 0)
            if opt.gclip > 0:
                torch.nn.utils.clip_grad_value_(self.netG.parameters(), opt.gclip)
            loss_D = self.loss_collector.compute_GAN_losses(self.netD, [SR.detach(), HR], for_discriminator=True)
            self.loss_collector.loss_backward(loss_D, self.optimD, self.schedulerD, 1)
            loss_dict = dict(zip(self.loss_collector.loss_names, loss_G + loss_D))

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
        step, max_steps = step // 1000, max_steps // 1000
        psnr = self.evaluate()

        if psnr >= self.best_psnr:
           self.best_psnr, self.best_step = psnr, step
           self.save(step)
        message = '[{}K/{}K] PSNR: {:.2f} (Best PSNR: {:.2f} @ {}K step) \n' \
                  '{}\n' \
                  'LR:{}, ETA:{:.1f} hours \n'.format(step, max_steps, psnr, self.best_psnr, self.best_step, err_msg, curr_lr, eta)
        Visualizer.log_print(opt, message)
        self.save('latest')
        self.t1 = time.time()

    @torch.no_grad()
    def evaluate(self):
        opt = self.opt
        self.netG.eval()

        if opt.save_result:
            save_root = os.path.join(opt.ckpt_root, opt.name, opt.output, opt.dataset)
            os.makedirs(save_root, exist_ok=True)

        psnr = 0
        for i, inputs in enumerate(self.test_loader):
            HR = inputs[0].to(self.device)
            LR = inputs[1].to(self.device)

            if 'cutblur' in opt.augs and HR.size() != LR.size():
                scale = HR.size(2) // LR.size(2)
                LR = F.interpolate(LR, scale_factor=scale, mode='nearest')

            SR = self.netG(LR).detach()
            HR = HR[0].mul(opt.rgb_range).clamp(0, 255).round().cpu().byte().permute(1, 2, 0).numpy()
            SR = SR[0].mul(opt.rgb_range).clamp(0, 255).round().cpu().byte().permute(1, 2, 0).numpy()

            if opt.save_result:
                save_path = os.path.join(save_root, '{:04}.png'.format(i+1))
                io.imsave(save_path, SR)
            HR = HR[opt.crop:-opt.crop, opt.crop:-opt.crop, :]
            SR = SR[opt.crop:-opt.crop, opt.crop:-opt.crop, :]
            if opt.eval_y_only:
                HR = util.rgb2ycbcr(HR)
                SR = util.rgb2ycbcr(SR)
            psnr += util.calculate_psnr(HR, SR)

        self.netG.train()

        return psnr/len(self.test_loader)

    def save(self, step):
        def save_network(network, network_label):
            save_path = os.path.join(self.save_dir, 'net' + network_label + '_' + str(step)+'.pth')
            torch.save(network.state_dict(), save_path)

        def save_optimizer(optimizer, optimizer_label):
            save_path = os.path.join(self.save_dir, 'optim' + optimizer_label + '_' + str(step)+'.pth')
            torch.save(optimizer.state_dict(), save_path)
        save_network(self.netG, 'G')
        save_optimizer(self.optimG, 'G')
        if self.netD is not None:
            save_network(self.netD, 'D')
            save_optimizer(self.optimD, 'D')

    def load(self, pretrain):
        def load_network(network, network_label, step_label, save_path=''):
            opt = self.opt
            save_filename = 'net%s_%s.pth' % (network_label, step_label)
            save_path = os.path.join(self.save_dir, save_filename) if not save_path else save_path
            if not os.path.isfile(save_path):
                Visualizer.log_print(opt, '%s not exists yet!' % save_path)
            else:
                try:
                    loaded_weights = torch.load(save_path)
                    network.load_state_dict(loaded_weights)

                    Visualizer.log_print(opt, 'network loaded from %s' % save_path)
                except:
                    pretrained_dict = torch.load(save_path)
                    model_dict = network.state_dict()
                    try:
                        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                        network.load_state_dict(pretrained_dict)
                        Visualizer.log_print(opt,
                                             'Pretrained network %s has excessive layers; Only loading layers that are used' % network_label)
                    except:
                        Visualizer.log_print(opt,
                                             'Pretrained network %s has fewer layers; The following are not initialized:' % network_label)
                        not_initialized = set()
                        for k, v in pretrained_dict.items():
                            if v.size() == model_dict[k].size():
                                model_dict[k] = v

                        for k, v in model_dict.items():
                            if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                                not_initialized.add('.'.join(k.split('.')[:2]))
                        Visualizer.log_print(opt, sorted(not_initialized))
                        network.load_state_dict(model_dict)

        def load_optimizer(optimizer, optimizer_label, step_label, save_path=''):
            opt = self.opt
            file_name = 'optim%s_%s.pth' % (optimizer_label, step_label)
            save_path = os.path.join(self.save_dir, file_name) if not save_path else save_path
            if not os.path.join(save_path):
                Visualizer.log_print(opt, '%s does not exist!' % save_path)
            else:
                try:
                    optimizer.load_state_dict(torch.load(save_path))
                    Visualizer.log_print(opt, 'optimizer loaded from %s' % save_path)
                except:
                    Visualizer.log_print(opt,
                                         'Optimizer parameters does not match, ignore loading optimizer')
        opt = self.opt
        label = 'latest' if opt.continue_train else opt.step_label
        load_network(self.netG, 'G', label, pretrain)
        if not opt.test_only:
            load_optimizer(self.optimG, 'G', label, pretrain)
            load_network(self.netD, 'D', label, pretrain)
            load_optimizer(self.optimD, 'D', label, pretrain)


