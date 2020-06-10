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


def get_training_data(rotate=False, resize=None, parent_folder='input') -> (np.ndarray, np.ndarray):
    x_files = glob.glob(parent_folder + '/training/images/*.png')
    y_files = glob.glob(parent_folder + '/training/groundtruth/*.png')

    x = np_from_files(x_files, rotate=rotate, resize=resize)
    y = np_from_files(y_files, rotate=rotate, resize=resize)
    y = (y > 42).astype(np.uint8)

    return x, y


def get_test_data(resize=None) -> np.ndarray:
    x_files = glob.glob('input/test_images/*.png')

    x = np_from_files(x_files, rotate=False, resize=resize)

    return x, x_files
