import matplotlib.pyplot as plt
import numpy as np


def show_image_single(image, label_image = None):
    fig=plt.figure(figsize=(8, 8))

    plt_1 = fig.add_subplot(1,1,1)
    plt.imshow(image)

    if label_image != None:
        plt_1.set_title(label_image)
    
    plt.show()


def show_image(image, mask, label_image = None, label_mask = None):
    fig=plt.figure(figsize=(16, 16))

    plt_1 = fig.add_subplot(1,2,1)
    plt.imshow(image)
    plt_2 = fig.add_subplot(1,2,2)
    plt.imshow(mask)

    if label_image != None:
        plt_1.set_title(label_image)
    if label_mask != None:
        plt_2.set_title(label_mask)
    
    plt.show()


def show_image_pred(image, mask, pred, label_image = None, label_mask = None, label_pred = None):
    fig=plt.figure(figsize=(16, 16))

    plt_1 = fig.add_subplot(1,3,1)
    plt.imshow(image)
    plt_2 = fig.add_subplot(1,3,2)
    plt.imshow(mask)
    plt_3 = fig.add_subplot(1,3,3)
    plt.imshow(pred)

    if label_image != None:
        plt_1.set_title(label_image)
    if label_mask != None:
        plt_2.set_title(label_mask)
    if label_pred != None:
        plt_3.set_title(label_mask)
    
    plt.show()



def blend_image(image, mask, ratio = 0.7) -> np.ndarray:
    if len(image.shape) == 3 and image.shape[2] == 3:
        mask = duplicate_channels(mask, 3)
    return image * mask * ratio + (1 - ratio) * image



def duplicate_channels(img, target_channels):
    """
    Duplicate a single channel 2D image (or a list images) to the specified number of channels
    """
    def _duplicate_channels(img, target_channels):
        x = np.empty((img.shape[0], img.shape[1], target_channels))
        for i in range(target_channels):
            x[:, :, i] = img
        return x
    
    if len(img.shape) == 3: # list of images
        x = np.empty(img.shape + (target_channels,))
        for i in range(img.shape[0]):
            x[i] = _duplicate_channels(img[i], target_channels)
    elif len(img.shape) == 2: # single image
        return _duplicate_channels(img, target_channels)
    else:
        raise Exception('Invalid image dimension!')