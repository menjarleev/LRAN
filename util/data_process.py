import os
import glob
import PIL.Image as Image
import PIL


def prepare_dataset(root, dataset, extension, scales, degradation):
    for data in dataset:
        for scale in scales:
            HR_path = os.path.join(root, 'HR', data, 'x{}'.format(scale))
            LR_path = os.path.join(root, 'LR', 'LR{}'.format(degradation), data, 'x{}'.format(scale))
            os.makedirs(HR_path, exist_ok=True)
            os.makedirs(LR_path, exist_ok=True)
        for ext in extension:
            origin_path = sorted(glob.glob(os.path.join(root, 'OriginalDataset', data, '*.{}'.format(ext))))
            for fpath in origin_path:
                img = Image.open(fpath)
                basename = os.path.basename(fpath)
                fname = basename[:-4]
                print('{} {}'.format(data, basename), end=' ')
                for scale in scales:
                    HR_path = os.path.join(root, 'HR', data, 'x{}'.format(scale),
                                           '{}_HR_x{}.{}'.format(fname, scale, ext))
                    LR_path = os.path.join(root, 'LR', 'LR{}'.format(degradation), data, 'x{}'.format(scale),
                                           '{}_LR{}_x{}.{}'.format(fname, degradation, scale, ext))
                    h, w = img.size
                    h = h - h % scale
                    w = w - w % scale
                    img = img.crop((0, 0, h, w))
                    resize_img = img.resize((h//scale, w//scale), resample=Image.BICUBIC)
                    img.save(HR_path)
                    resize_img.save(LR_path)
                    print('x{}'.format(scale), end=' ')
                print()

if __name__ == '__main__':
    root = '/home/lhuo9710/PycharmProjects/dataset/'
    dataset = ['Urban100', 'Managa109', 'BSD100', 'DIV2K', 'Set5', 'Set14']
    extension = ['jpg', 'png', 'bmp']
    scale = [2, 3, 4, 8]
    degradation = 'BI'
    prepare_dataset(root, dataset, extension, scale, degradation)


