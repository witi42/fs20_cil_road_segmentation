from PIL import Image
import glob
import numpy as np
from typing import List
import skimage.transform
from skimage import filters
from skimage.color import rgb2gray
from skimage.color import gray2rgb
from skimage.color import rgb2hsv
from skimage.color import hsv2rgb



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


def augment_data(x, y):
    x_aug = flip_and_rotate(x)
    y_aug = flip_and_rotate(y)
    
    return x_aug, y_aug


def augment_data_extended(x, y, saturation = None, grayscale = False, blur = None):
    if (saturation != None):
        x_aug = saturate(x, saturation)
        y_aug = duplicate(y, 2) # don't need to desaturate groundtruth, just copy it
    
    if grayscale:
        x_aug = grayscale(x_aug)
        y_aug = duplicate(y, 2)
    
    if blur != None:
        x_aug = blur(x_aug, blur)
        y_aug = duplicate(y, 2)
    
    x_aug = flip_and_rotate(x_aug)
    y_aug = flip_and_rotate(y_aug)
    
    return x_aug, y_aug


def duplicate(x, count):
    """
    Duplicate images in x, such that each image is followed by <count - 1> copies.
    Each image thus is present <count> times in a row.
    """
    l = []
    for i in range(x.shape[0]):
        for j in range(count):
            l.append(x[i])
    return np.asarray(l)

    
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
            image_rot = skimage.transform.rotate(x[i], deg, preserve_range = True)
            l.append(image_rot)
    return np.asarray(l)


# if we flip and rotate together we only need to flip once, the other would be redundant
def flip_and_rotate(x):
    l = []
    for i in range(x.shape[0]):
        image_flip = np.flipud(x[i])
        l.append(x[i])
        l.append(image_flip)
        for deg in [90, 180, 270]:
            l.append(skimage.transform.rotate(x[i], deg, preserve_range = True))
            l.append(skimage.transform.rotate(image_flip, deg, preserve_range = True))
    return np.asarray(l)

def grayscale(x):
    l = []
    for i in range(x.shape[0]):
        l.append(x[i])
        l.append(gray2rgb(rgb2gray(x[i])))
    return np.asarray(l)


def blur(x, sigma):
    l = []
    for i in range(x.shape[0]):
        l.append(x[i])
        l.append(filters.gaussian(x[i], sigma = sigma, multichannel = True, preserve_range = True))
    return np.asarray(l)


def saturate(x, factor):
    l = []
    for i in range(x.shape[0]):
        l.append(x[i])
        image_sat = rgb2hsv(x[i])
        image_sat[:, :, 1] = (1 - (1 - image_sat[:, :, 1]) ** factor)
        l.append(hsv2rgb(image_sat))
    return np.asarray(l)
