from PIL import Image
import glob
import numpy as np
from typing import List
import skimage.transform



def np_from_files(files: list, rotate = False, flip = False, size = None) -> np.ndarray:
    x = []
    for f in files:
        image = Image.open(f)
        image = np.asarray(image)

        if size != None:
            image = skimage.transform.resize(image, size)

        x.append(image)

        if rotate:
          for deg in [90, 180, 270]:
            image_rot = skimage.transform.rotate(image, deg)
            x.append(image_rot)
        
        if flip:
          x.append(np.fliplr(image))
          x.append(np.flipud(image))

    return np.asarray(x)


def one_or_zero(v):
    return np.venp.vectorize(lambda x: 1.0 if x > 10 else 0.0)


def get_training_data(rotate = False, flip = False, size = None) -> (np.ndarray, np.ndarray):
    x_files = sorted(glob.glob('input/training/images/*.png'))
    y_files = sorted(glob.glob('input/training/groundtruth/*.png'))

    x = np_from_files(x_files, rotate, flip, size)
    y = np_from_files(y_files, rotate, flip, size)
    y = (y > 0.5).astype(np.uint8)

    return x, y


def get_training_validation_data(split=0.2, rotate=False, resize=None, parent_folder='input', normalise_x=False) \
    -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Splits the data into a train and validation set, taking care not to mix the rotated ones.

    :param split: the percentage of samples to use in the test set. Should be > 1/nb_samples
    :param rotate:
    :param resize:
    :param parent_folder:
    :param normalise_x: boolean: whether or not to transform the x values into [0,1]
    :return: x_train, x_val, y_train, y_val
    """
    x_files = np.asarray(glob.glob(parent_folder + '/training/images/*.png'))
    y_files = np.asarray(glob.glob(parent_folder + '/training/groundtruth/*.png'))
    x_files.sort()
    y_files.sort()

    n = len(x_files)
    rand = np.random.choice(n, n, replace=False)

    k = int(split*n)
    x_val = np_from_files(x_files[rand[:k]], rotate=rotate, resize=resize)
    x_train = np_from_files(x_files[rand[k:]], rotate=rotate, resize=resize)
    y_val = (np_from_files(y_files[rand[:k]], rotate=rotate, resize=resize) > 42).astype(np.uint8)
    y_train = (np_from_files(y_files[rand[k:]], rotate=rotate, resize=resize) > 42).astype(np.uint8)

    if normalise_x:
        x_train = x_train.astype(np.float64) / 255.
        x_val = x_val.astype(np.float64) / 255.

    return x_train, x_val, y_train, y_val



def get_augmented_training_data(rotate = False, flip = False, size = None) -> (np.ndarray, np.ndarray):
  x_files = sorted(glob.glob('input/training/augmented/images/*.png'))
  y_files = sorted(glob.glob('input/training/augmented/groundtruth/*.png'))

  x = np_from_files(x_files, rotate, flip, size)
  y = np_from_files(y_files, rotate, flip, size)
  y = (y > 0.5).astype(np.uint8)

  return x, y




def get_test_data(size = None, normalise_x = False) -> (np.ndarray, List[str]) :
    x_files = sorted(glob.glob('input/test_images/*.png'))
    
    x = np_from_files(x_files, rotate = False, flip = False, size = size)
    if normalise_x:
        x = x.astype(np.float64) / 255.

    return x, x_files

