from PIL import Image, ImageOps
import glob
import os
import numpy as np

from resizeimage import resizeimage

def resize_all(dir_src, dir_dest, replace_extension='.jpg', height=512, width=512):

    if not os.path.exists(dir_dest):
        os.mkdir(dir_dest, 0o666)

    img_paths = [f for f in glob.glob(dir_src)]
    if replace_extension != '.jpg':
        img_names = [os.path.basename(f).replace(replace_extension, ".jpg") for f in glob.glob(dir_src)]
    else:
        img_names = [os.path.basename(f) for f in glob.glob(dir_src)]
    print(img_names)

    for img_path, img_name in zip(img_paths, img_names):
        with open(img_path, 'r+b') as f:
            with Image.open(f) as image:
                cover = resizeimage.resize_cover(image, [height, width])
                cover.save(dir_dest + img_name, image.format)


def resize_n(dir_src, dir_dest, amount, replace_extension='.jpg', height=512, width=512):

    if not os.path.exists(dir_dest):
        os.mkdir(dir_dest, 0o666)

    img_paths = [f for f in glob.glob(dir_src)[:amount]]
    if replace_extension != '.jpg':
        img_names = [os.path.basename(f).replace(replace_extension, ".jpg") for f in glob.glob(dir_src)[:amount]]
    else:
        img_names = [os.path.basename(f) for f in glob.glob(dir_src)[:amount]]
    print(img_names)

    for img_path, img_name in zip(img_paths, img_names):
        with open(img_path, 'r+b') as f:
            with Image.open(f) as image:
                cover = resizeimage.resize_cover(image, [height, width])
                cover.save(dir_dest + img_name, image.format)


def to_npy_greyscale_file_n(dir_src, dir_dest, amount, replace_extension='.jpg', height=512, width=512):
    if not os.path.exists(dir_dest):
        os.mkdir(dir_dest, 0o666)

    img_paths = [f for f in glob.glob(dir_src)[:amount]]
    img_names = [os.path.basename(f).replace(replace_extension, "") for f in glob.glob(dir_src)[:amount]]

    print(img_names)

    for img_path, img_name in zip(img_paths, img_names):
        with open(img_path, 'r+b') as f:
            with Image.open(f) as image:
                image = ImageOps.grayscale(image)
                cover = resizeimage.resize_cover(image, [height, width])

                pix = np.array(cover)
                np.save(dir_dest + img_name, pix)


path = 'D:'
file_path = os.path.join(path, "Carvana_dataset", "train", "*.jpg")
to_npy_greyscale_file_n(file_path, dir_dest='../dataset/imgs_512/train_512/', amount=100)

file_path = os.path.join(path, "Carvana_dataset", "train", "*.jpg")
to_npy_greyscale_file_n(file_path, dir_dest='../dataset/imgs_512/test_512/', amount=10)

file_path = os.path.join(path, "Carvana_dataset", "train", "*.jpg")
to_npy_greyscale_file_n(file_path, dir_dest='../dataset/imgs_512/train_masks_512/', amount=100, replace_extension='.gif')


# file_path = os.path.join(path, "Carvana_dataset", "train", "*.jpg")
# resize_n(file_path, './train_512/', amount=10)

# file_path = os.path.join(path, "Carvana_dataset", "test", "*.jpg")
# resize_n(file_path, './test_512/', amount=10)
#
# file_path = os.path.join(path, "Carvana_dataset", "train_masks", "*.gif")
# resize_n(file_path, './train_masks_512/', amount=10, replace_extension='.gif')
