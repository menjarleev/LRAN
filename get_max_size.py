import os
import PIL.Image as Image

def get_max_size(folder_name):
    file = os.listdir(folder_name)
    file_path = sorted([os.path.join(folder_name, f) for f in file])
    min_h, min_w = float('inf'), float('inf')
    for path in file_path:
        img = Image.open(path)
        min_h = min(min_h, img.size[0])
        min_w = min(min_w, img.size[1])
    return min_h, min_w


if __name__ == '__main__':
    folder_name = '/home/lhuo9710/PycharmProjects/dataset/DIV2K/train/x1'
    h, w = get_max_size(folder_name)
    print(h, w)

