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
import math
import random



def np_from_files(files: list) -> np.ndarray:
    x = []
    for f in files:
        image = Image.open(f)
        image = np.asarray(image)

        #print(f, image.shape)
    
        x.append(image)
    
    return np.asarray(x)


def np_from_files_grid_cut(files: list) -> np.ndarray:
    size = (400,400)

    x = []
    for f in files:
        image = Image.open(f)
        image = np.asarray(image)

        #print(f, image.shape)

        for x_start in range(0, image.shape[0]-400+1, size[0]):
            for y_start in range(0, image.shape[1]-400+1, size[1]):
                cutout = image[x_start:x_start+400, y_start:y_start+400]

                if (cutout.shape[0:2] != size):
                    print("ERROR: cutout has shape ", cutout.shape)
                else:
                    x.append(cutout)

    return np.asarray(x)


def make_groundtruth_binary(y, threshold = 0.5):
    """
    Ensures that the groundtruth data is stored as {0, 1} np.uint8
    """
    if y.max() > 1:         # normalize it first
        if y.max() > 255:   # should never be the case, but still...
            print("get_data.make_groundtruth_binary() encountered weird data range!")
        y = y / 255.
    return (y > threshold).astype(np.uint8)


# get training data as ndarrays from original dataset
def get_training_data(normalize = True) -> (np.ndarray, np.ndarray):
    x_files = sorted(glob.glob('input/training/images/*.png'))
    y_files = sorted(glob.glob('input/training/groundtruth/*.png'))

    return get_training_data_path(x_files, y_files, normalize)


# get training data as ndarrays from additional dataset
def get_training_data2(normalize=True) -> (np.ndarray, np.ndarray):
    x_files = sorted(glob.glob('input/training/images_aug/*.png'))
    y_files = sorted(glob.glob('input/training/groundtruth_aug/*.png'))

    return get_training_data_path(x_files, y_files, normalize)


def get_training_data_path(x_files, y_files, normalize=True, image_to_np=np_from_files) -> (np.ndarray, np.ndarray):
    x = image_to_np(x_files)
    y = image_to_np(y_files)
    if normalize:
        x = x.astype(np.float32) / 255.
        y = y.astype(np.float32) / 255.

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


def augment_data_extended(x, y, saturation = None, use_grayscale = False, blur_amount = None, num_random_rotations = 3):
    x = random_rotate(x, num_rotations = num_random_rotations)
    y = random_rotate(y, num_rotations = num_random_rotations)
    
    if (saturation != None):
        x = saturate(x, saturation)
        y = duplicate(y, 2) # don't need to desaturate groundtruth, just copy it
    
    if use_grayscale:
        x = grayscale(x)
        y = duplicate(y, 2)
    
    if blur_amount != None:
        x = blur(x, blur_amount)
        y = duplicate(y, 2)
    
    x = flip_and_rotate(x)
    y = flip_and_rotate(y)
    
    y = make_groundtruth_binary(y)
    return x, y


def resize_images(x, size):
    x_shape = list(x.shape)
    x_shape[1:3] = list(size)

    x_out = np.ndarray(shape=x_shape, dtype=x.dtype)

    for i, image in enumerate(x):
        out_image = skimage.transform.resize(image, size, anti_aliasing=True)
        x_out[i] = out_image
    return x_out


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


def random_rotate(x, num_rotations = 3, seed = '4815162342', min_deg = 20, max_deg = 70):
    """
    When specifying a seed, make sure the same seed is used for the groundtruth as well!
    """
    random.seed(seed)
    l = []
    original_size = (x.shape[1], x.shape[2])
    w = original_size[0]
    c_1 = int(w / 2 - w / math.sqrt(8))
    c_2 = int(w - c_1)
    for i in range(x.shape[0]):
        l.append(x[i])
        for j in range(num_rotations):
            deg = random.randrange(min_deg, max_deg)
            img_rot = skimage.transform.rotate(x[i], deg, preserve_range = True)                # rotate
            img_rot = img_rot[c_1:c_2, c_1:c_2]                                                 # crop black area away
            img_rot = skimage.transform.resize(img_rot, original_size, preserve_range = True)   # resize to original size
            l.append(img_rot)
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
