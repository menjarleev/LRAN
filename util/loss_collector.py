from model.loss import *
import torch
class LossCollector():
    def __init__(self, opt):
        self.opt = opt
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
        self.criterionGAN = GANLoss(opt.gan_mode, tensor=self.tensor)
        self.criterionFeat = torch.nn.L1Loss()
        self.criterionL1 = torch.nn.L1Loss()
        if not opt.no_vgg:
            self.criterionVGG = VGGLoss(opt)
        if 'L1' in opt.loss_term:
            self.loss_names = ['L1']
        elif 'GAN' in opt.loss_term:
            self.loss_names_G = ['G_GAN', 'G_GAN_Feat',
                                 'G_VGG']

            self.loss_names_D = ['D_real', 'D_fake']
            self.loss_names = self.loss_names_G + self.loss_names_D
        else:
            raise NotImplementedError('%s is not implemented' % opt.loss_term)

    def compute_GAN_losses(self, netD, data_list, for_discriminator):
        fake, gt = data_list
        pred_fake = netD(fake)
        pred_real = netD(gt)
        if for_discriminator:
            loss_D_real = self.criterionGAN(pred_real, True)
            loss_D_fake = self.criterionGAN(pred_fake, False)
            return [loss_D_real, loss_D_fake]

        else:
            loss_G_GAN = self.criterionGAN(pred_fake, True)
            loss_G_GAN_Feat = self.GAN_matching_loss(pred_real, pred_fake, for_discriminator)
            return [loss_G_GAN, loss_G_GAN_Feat]

    def compute_L1_losses(self, fake_image, gt_image):
        opt = self.opt
        loss_L1 = self.criterionL1(fake_image, gt_image)
        return loss_L1 * opt.lambda_L1

    def compute_VGG_losses(self, fake_image, gt_image):
        loss_G_VGG = self.tensor(1).fill_(0)
        opt = self.opt
        if not opt.no_vgg:
            if type(fake_image) == list:
                fake_image = fake_image[-1]
                gt_image = gt_image[-1]
            loss_G_VGG = self.criterionVGG(fake_image, gt_image)
        return loss_G_VGG * opt.lambda_vgg

    def GAN_matching_loss(self, pred_real, pred_fake, for_discriminator=False):
        loss_G_GAN_Feat = self.tensor(1).fill_(0)
        if not for_discriminator and not self.opt.no_GAN_feat:
            num_D = len(pred_fake)
            D_masks = 1.0 / num_D
            for i in range(num_D):
                for j in range(len(pred_fake[i])-1):
                    loss = self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach())
                    loss_G_GAN_Feat = loss_G_GAN_Feat + D_masks * loss
        return loss_G_GAN_Feat * self.opt.lambda_feat

    def loss_backward(self, losses, optimizer, scheduler, loss_id):
        opt = self.opt
        losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]
        loss = sum(losses)
        optimizer.zero_grad()
        if opt.amp != 'O0':
            from apex import amp
            with amp.scale_loss(loss, optimizer, loss_id=loss_id) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        scheduler.step()
        return losses
