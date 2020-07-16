from PIL import Image
import glob
import numpy as np
from typing import List
import skimage.transform



def np_from_files(files: list) -> np.ndarray:
    x = []
    for f in files:
        image = Image.open(f)
        image = np.asarray(image)

        x.append(image)

    return np.asarray(x)


def get_training_data(normalize = True) -> (np.ndarray, np.ndarray):
    x_files = sorted(glob.glob('input/training/images/*.png'))
    y_files = sorted(glob.glob('input/training/groundtruth/*.png'))

    x = np_from_files(x_files)
    if normalize:
        x = x.astype(np.float32) / 255.
    
    y = np_from_files(y_files)
    y = (y > 0.5).astype(np.uint8)

    return x, y


def get_test_data(normalize = True) -> (np.ndarray, List[str]) :
    x_files = sorted(glob.glob('input/test_images/*.png'))
    
    x = np_from_files(x_files)
    if normalize:
        x = x.astype(np.float) / 255.

    return x, x_files


def augment_data(x,y):
    x_aug = flip(x)
    y_aug = flip(y)
    
    x_aug = rotate(x_aug)
    y_aug = rotate(y_aug)
    
    return x_aug, y_aug


def flip(x):
    l = []
    for i in range(x.shape[0]):
        l.append(x[i])
        l.append(np.fliplr(x[i]))
        l.append(np.flipud(x[i]))
    return np.asarray(l)


def rotate(x):
    l = []
    for i in range(x.shape[0]):
        l.append(x[i])
        for deg in [90, 180, 270]:
            image_rot = skimage.transform.rotate(x[i], deg)
            l.append(image_rot)
    return np.asarray(l)
