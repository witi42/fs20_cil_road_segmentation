from PIL import Image
import glob
import numpy as np


def np_from_files(files, rotate=False, resize=None) -> np.ndarray:
    x = []
    for f in files:
        image = Image.open(f)

        if resize is not None:
            image.thumbnail(resize, Image.ANTIALIAS)

        x.append(np.asarray(image))

        if rotate:
            for deg in range(90, 271, 90):
                new_image = image.rotate(deg)
                x.append(np.asarray(new_image))

    return np.asarray(x)


def one_or_zero(v):
    return np.venp.vectorize(lambda x: 1.0 if x > 10 else 0.0)


def get_training_data(rotate=False, resize=None, parent_folder='input', normalise_x=False) -> (np.ndarray, np.ndarray):
    x_files = glob.glob(parent_folder + '/training/images/*.png')
    y_files = glob.glob(parent_folder + '/training/groundtruth/*.png')
    x_files.sort()
    y_files.sort()

    x = np_from_files(x_files, rotate=rotate, resize=resize)
    y = np_from_files(y_files, rotate=rotate, resize=resize)
    y = (y > 42).astype(np.uint8)
    if normalise_x:
        x = x.astype(np.float64) / 255.

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





def get_test_data(resize=None, normalise_x=False) -> np.ndarray:
    x_files = glob.glob('input/test_images/*.png')
    x_files.sort()

    x = np_from_files(x_files, rotate=False, resize=resize)
    if normalise_x:
        x = x.astype(np.float64) / 255.

    return x, x_files
