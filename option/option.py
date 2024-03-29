import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)

    # models
    parser.add_argument('--pretrain', type=str, default='')
    parser.add_argument('--netG', type=str, default='LRAN')
    parser.add_argument('--netD', type=str, default='MULTISCALE')
    parser.add_argument('--actv_G', type=str, default='ReLU')
    parser.add_argument('--slope_G', type=float, default=1e-2)
    parser.add_argument('--padding_G', type=str, default='zeros')
    parser.add_argument('--actv_D', type=str, default='ReLU')
    parser.add_argument('--slope_D', type=float, default=1e-2)
    parser.add_argument('--conv_layer_D', type=str, default='default')
    parser.add_argument('--padding_D', type=str, default='zeros')
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--n_layer_D', type=int, default=3)
    parser.add_argument('--num_D', type=int, default=2)
    parser.add_argument('--gan_mode', type=str, default='ls', help='[ls|origin|hinge]')
    parser.add_argument('--norm_D', type=str, default='instance')
    parser.add_argument('--use_vgg', action='store_true')
    parser.add_argument('--dropout', action='store_true')
    parser.add_argument('--num_channels', type=int, default=64)

    # augmentations
    parser.add_argument('--use_moa', action='store_true')
    parser.add_argument('--augs', nargs='*', default=["none"])
    parser.add_argument('--prob', nargs='*', default=[1.0])
    parser.add_argument('--mix_p', nargs='*')
    parser.add_argument('--alpha', nargs='*', default=[0.7])
    parser.add_argument('--aux_prob', type=float, default=1.0)
    parser.add_argument('--aux_alpha', type=float, default=1.2)

    # dataset
    parser.add_argument('--root', type=str, default='./')
    parser.add_argument('--camera', type=str, default='all')  # RealSR
    parser.add_argument('--image_range', type=str, default='1-800/801-810')
    parser.add_argument('--scale', type=int, default=4)  # SR scale
    parser.add_argument('--sigma', type=int, default=10)  # DN
    parser.add_argument('--quality', type=int, default=10)  # DeJPEG
    parser.add_argument('--type', type=int, default=1)  # deblur

    # training setups
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--decay', type=str, default='150-250-350')
    parser.add_argument('--gamma', type=int, default=0.5)
    parser.add_argument('--patch_size', type=int, default=48)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_steps', type=int, default=7e5)
    parser.add_argument('--eval_steps', type=int, default=1000)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--gclip', type=int, default=0)

    # misc
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--no_validation', action='store_true')
    parser.add_argument('--no_test', action='store_true')
    parser.add_argument('--save_result', action='store_true')
    parser.add_argument('--ckpt_root', type=str, default='./ckpt')
    parser.add_argument('--lambda_feat', type=float, default=10.0)
    parser.add_argument('--lambda_L1', type=float, default=1.0)
    parser.add_argument('--L1_decay', type=str, nargs='*', default=[])
    parser.add_argument('--lambda_vgg', type=float, default=10.0)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--name', type=str, default='LRAN')
    parser.add_argument('--amp', type=str, default='O0')
    parser.add_argument('--print_mem', type=bool, default=True)
    parser.add_argument('--step_label', type=str, default='latest')
    parser.add_argument('--continue_train', action='store_true')
    parser.add_argument('--loss_terms', type=str, default='L1')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--normalize', action='store_true')

    # test
    parser.add_argument('--test_steps', type=int, default=5000)
    parser.add_argument('--eval_metric', type=str, default='psnr')

    # inference
    parser.add_argument('--infer', action='store_true')
    parser.add_argument('--num_blocks', type=int, default=128)
    parser.add_argument('--group_size', type=int, default=2)
    parser.add_argument('--degradation', type=str, default='BI')

    parser.add_argument('--train_name', type=str, default='DIV2K')
    parser.add_argument('--infer_name', type=str, default='Set14')
    parser.add_argument('--test_name', type=str, default='Set14')
    parser.add_argument('--train_type', type=str, default='DIV2KSR')
    parser.add_argument('--test_type', type=str, default='benchmarkSR')
    parser.add_argument('--infer_type', type=str, default='inferSR')



    return parser.parse_args()

def make_template(opt):
    opt.strict_load = opt.test_only
    opt.loss_terms = opt.loss_terms.lower()

    # model
    if "EDSR" in opt.netG:
        opt.num_blocks = 32
        opt.num_channels = 256
        opt.res_scale = 0.1

    if 'LRAN'in opt.netG:
        opt.num_blocks = 64
        opt.num_channels = 64
        opt.res_scale = 1.0
        opt.decay = '150-250-350-450'
        opt.max_steps = 500000
        opt.reduction = 8

    if 'INSR' in opt.netG:
        opt.num_blocks = 20
        opt.num_groups = 10
        opt.num_channels = 64
        opt.padding_G = 'reflect'
        opt.reduction = 8

    if 'SRAN' in opt.netG:
        # for school -> receptive field larger than 100
        opt.num_blocks = [[50, 0], [30, 20], [20, 30], [10, 40]]
        opt.batch_size = 32
        opt.num_channels = 64
        opt.num_groups = 4
        opt.padding_G = 'reflect'
        opt.reduction = 16
        opt.res_scale = 1.0

    if 'LSR' in opt.netG:
        # for school -> receptive field larger than 100
        opt.num_blocks = 5
        # for school -> receptive field equals to 51
        # opt.num_blocks = [1, 4]
        opt.batch_size = 16
        opt.num_channels = 64
        opt.num_groups = 4
        opt.padding_G = 'reflect'
        opt.reduction = 16
        opt.res_scale = 1.0

    if 'LWSR' in opt.netG:
        # for school -> receptive field larger than 100
        opt.num_blocks = [[10, 20], [5, 25], [5, 25], [5, 25]]
        # for school -> receptive field equals to 51
        # opt.num_blocks = [1, 4]
        opt.batch_size = 16
        opt.num_channels = 64
        opt.num_groups = 4
        opt.padding_G = 'reflect'
        opt.reduction = 16
        opt.res_scale = 1.0

    if 'PRAN'in opt.netG:
        opt.res_scale = 1.0
        opt.decay = '150-250-350-450'
        opt.max_steps = 500000
        opt.reduction = 16

    if 'RASR'in opt.netG:
        opt.num_channels = 64
        opt.num_blocks = 128
        opt.group_size = 2
        opt.res_scale = 1.0
        opt.decay = '150-250-350-450'
        opt.max_steps = 500000
        opt.reduction = 4


    if 'SRRCAN' in opt.netG:
        opt.num_groups = 10
        opt.num_blocks = [[6, 14]]
        opt.num_blocks += [[2, 18]] * 9
        opt.num_channels = 64
        opt.reduction = 16
        opt.res_scale = 1.0
        opt.max_steps = 1000000
        opt.decay = "200-400-600-800"
        opt.gclip = 0.5 if opt.pretrain else opt.gclip

    elif "RCAN" in opt.netG:
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

    opt.rgb_mean = (0.4488, 0.4371, 0.4040)

    if "DN" in opt.train_name or "JPEG" in opt.train_name:
        opt.max_steps = 1000000
        opt.decay = "300-550-800"

    if 'DIV2K' in opt.train_name:
        opt.decay = "200-400-600-800"
        opt.max_steps = 1000000

    if "RealSR" in opt.train_name:
        opt.patch_size *= opt.scale  # identical (LR, HR) resolution



    # default augmentation policies
    if opt.use_moa:
        opt.augs = ["blend", "rgb", "mixup", "cutout", "cutmix", "cutmixup", "cutblur"]
        opt.prob = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        opt.alpha = [0.6, 1.0, 1.2, 0.001, 0.7, 0.7, 0.7]
        opt.aux_prob, opt.aux_alpha = 1.0, 1.2
        opt.mix_p = None

        if "RealSR" in opt.train_name:
            opt.mix_p = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4]

        if "DN" in opt.train_name or "JPEG" in opt.train_name:
            opt.prob = [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]
        if "CARN" in opt.netG and not "RealSR" in opt.train_name:
            opt.prob = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]


def get_option():
    opt = parse_args()
    make_template(opt)
    ckpt_dir = os.path.join(opt.ckpt_root, opt.name)
    os.makedirs(ckpt_dir, exist_ok=True)
    return opt

