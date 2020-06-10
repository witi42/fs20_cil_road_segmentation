#!/usr/bin/env python3

import os
import glob
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import re
import matplotlib.cm as cm

foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch

# assign a label to a patch
def patch_to_label(patch):
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0


def mask_to_submission_strings(image_filename):
    """Reads a single image and outputs the strings that should go into the submission file"""
    img_number = int(re.search(r"\d+", image_filename).group(0))
    im = mpimg.imread(image_filename)
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))


def masks_to_submission(submission_filename, *image_filenames):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames[0:]:
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn))


def save_predictions(predictions, test_names, submission_name, path='', allow_overwrite=False):
    """
    Saves all the predictions from an array as images and creates the submission file.

    IMPORTANT!! path name should not contain numbers, otherwise the other functions down the line get
    confused with getting the image ids.

    :param predictions: ndarray the predicted image results (\in [0,1])
    :param test_names: the names of the predictions, in the same order as the predictions
    :param submission_name: the name of the folder to save the resulting images and submission file in
    :param path: The path to put the folder in. Default: Current Working Directory
    :param allow_overwrite: bool: If true, the data in the folder may be overridden, otherwise, it mayn't.
    :return: nothing
    """
    if len(path) > 0 and path[-1] != "/":
        path = path + "/"

    if any(char.isdigit() for char in   path + submission_name):
        raise ValueError("Filepath may not contain numbers")

    os.makedirs(path + submission_name, exist_ok=allow_overwrite)
    n = len(test_names)
    for i in range(n):
        name = test_names[i][18:]
        print("\r{}/{}, {}".format(i, n, name), end='')

        pred = predictions[i].reshape(608, 608)
        pred = (pred > 0.5).astype(np.uint8)

        plt.imsave(path + submission_name + "/" + name, pred, cmap=cm.gray)

    submission_filename = path + submission_name + "/" + submission_name + '.csv'
    image_filenames = glob.glob(path + submission_name + "/*.png")
    masks_to_submission(submission_filename, *image_filenames)
