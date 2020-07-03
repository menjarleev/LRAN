import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)

    # models
    parser.add_argument('--pretrain', type=str, default='')
    parser.add_argument('--netG', type=str, default='LRAN')
    parser.add_argument('--netD', type=str, default='MULTISCALE')
    parser.add_argument('--actv_G', type=str, default='LeakyReLU')
    parser.add_argument('--slope_G', type=float, default=1e-2)
    parser.add_argument('--padding_G', type=str, default='reflect')
    parser.add_argument('--rgb_range', type=int, default=255)
    parser.add_argument('--actv_D', type=str, default='LeakyReLU')
    parser.add_argument('--slope_D', type=float, default=1e-2)
    parser.add_argument('--conv_layer_D', type=str, default='default')
    parser.add_argument('--padding_D', type=str, default='reflect')
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--n_layer_D', type=int, default=3)
    parser.add_argument('--num_D', type=int, default=2)
    parser.add_argument('--gan_mode', type=str, default='ls', help='[ls|origin|hinge]')

    # augmentations
    parser.add_argument('--use_moa', action='store_true')
    parser.add_argument('--augs', nargs='*', default=["none"])
    parser.add_argument('--prob', nargs='*', default=[1.0])
    parser.add_argument('--mix_p', nargs='*')
    parser.add_argument('--alpha', nargs='*', default=[1.0])
    parser.add_argument('--aux_prob', type=float, default=1.0)
    parser.add_argument('--aux_alpha', type=float, default=1.2)

    # dataset
    parser.add_argument('--dataset_root', type=str, default='./')
    parser.add_argument('--dataset', type=str, default='AIMSR')
    parser.add_argument('--camera', type=str, default='all')  # RealSR
    parser.add_argument('--image_range', type=str, default='1-800/801-810')
    parser.add_argument('--scale', type=int, default=4)  # SR scale
    parser.add_argument('--sigma', type=int, default=10)  # DN
    parser.add_argument('--quality', type=int, default=10)  # DeJPEG
    parser.add_argument('--type', type=int, default=1)  # deblur

    # training setups
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--decay', type=str, default='200-400-600')
    parser.add_argument('--gamma', type=int, default=0.5)
    parser.add_argument('--patch_size', type=int, default=48)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_steps', type=int, default=7e5)
    parser.add_argument('--eval_steps', type=int, default=1000)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--gclip', type=int, default=0)

    # misc
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--save_result', action='store_true')
    parser.add_argument('--ckpt_root', type=str, default='./ckpt')
    parser.add_argument('--no_GAN_feat', action='store_true')
    parser.add_argument('--lambda_feat', type=float, default=10.0)
    parser.add_argument('--lambda_L1', type=float, default=1.0)
    parser.add_argument('--L1_decay', type=str, nargs='*', default=[50000, 50000])
    parser.add_argument('--no_vgg', action='store_true')
    parser.add_argument('--lambda_vgg', type=float, default=10.0)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--name', type=str, default='LRAN')
    parser.add_argument('--amp', type=str, default='O1')
    parser.add_argument('--print_mem', type=bool, default=True)
    parser.add_argument('--step_label', type=str, default='latest')
    parser.add_argument('--continue_train', action='store_true')
    parser.add_argument('--loss_term', type=str, default='GAN')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--dir_HQ', type=str, default='x1')
    parser.add_argument('--dir_LQ', type=str, default='x4')

    return parser.parse_args()

def make_template(opt):
    opt.strict_load = opt.test_only

    # model
    if "EDSR" in opt.netG:
        opt.num_blocks = 32
        opt.num_channels = 256
        opt.res_scale = 0.1

    if 'LRAN'in opt.netG:
        opt.num_blocks = 64
        opt.num_channels = 64
        opt.res_scale = 1.0
        opt.decay = '200-400-600-800'
        opt.max_steps = 1000000
        opt.reduction = 16

    if "RCAN" in opt.netG:
        opt.num_groups = 10
        opt.num_blocks = 20
        opt.num_channels = 64
        opt.reduction = 16
        opt.res_scale = 1.0
        opt.max_steps = 1000000
        opt.decay = "200-400-600-800"
        opt.gclip = 0.5 if opt.pretrain else opt.gclip

    if "CARN" in opt.netG:
        opt.num_groups = 3
        opt.num_blocks = 3
        opt.num_channels = 64
        opt.res_scale = 1.0
        opt.batch_size = 64
        opt.decay = "400"

    # training setup
    if "DN" in opt.dataset or "JPEG" in opt.dataset:
        opt.max_steps = 1000000
        opt.decay = "300-550-800"
    if 'AIM' in opt.dataset:
        opt.image_range = '1-19000/1-10'
    if "RealSR" in opt.dataset:
        opt.patch_size *= opt.scale  # identical (LR, HR) resolution

    # evaluation setup
    opt.crop = 6 if "DIV2K" in opt.dataset else 0
    opt.crop += opt.scale if "SR" in opt.dataset else 4

    # note: we tested on color DN task
    if "DIV2K" in opt.dataset or "DN" in opt.dataset:
        opt.eval_y_only = False
    else:
        opt.eval_y_only = True

    # default augmentation policies
    if opt.use_moa:
        opt.augs = ["blend", "rgb", "mixup", "cutout", "cutmix", "cutmixup", "cutblur"]
        opt.prob = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        opt.alpha = [0.6, 1.0, 1.2, 0.001, 0.7, 0.7, 0.7]
        opt.aux_prob, opt.aux_alpha = 1.0, 1.2
        opt.mix_p = None

        if "RealSR" in opt.dataset:
            opt.mix_p = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4]

        if "DN" in opt.dataset or "JPEG" in opt.dataset:
            opt.prob = [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]
        if "CARN" in opt.netG and not "RealSR" in opt.dataset:
            opt.prob = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]


def get_option():
    opt = parse_args()
    make_template(opt)
    ckpt_dir = os.path.join(opt.ckpt_root, opt.name)
    os.makedirs(ckpt_dir, exist_ok=True)
    return opt

