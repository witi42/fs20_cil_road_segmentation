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
import skimage.io
import skimage.util
import matplotlib.cm as cm
import os



def np_from_files(files: list, rescale_factor = None, labels = False) -> List[np.ndarray]:
    x = []
    for f in files:
        image = Image.open(f)
        image = np.asarray(image)
        
        print("opening: " + f)
        
        if rescale_factor != None:
            image = skimage.transform.rescale(image, rescale_factor, preserve_range = True, multichannel = True)
        
        if labels:
            img_r = np.squeeze(image[:, :, 0])
            x.append(255. - img_r)
        else:
            x.append(image)
            
    
    return x


def split_data(img, size = (400, 400)):
    splits_x = int(img.shape[0] / size[0])
    splits_y = int(img.shape[1] / size[1])
    
    splitted = []
    for i in range(splits_x):
        for j in range(splits_y):
           splitted_img = img[i * size[0] : (i + 1) * size[0], j * size[1] : (j + 1) * size[1]]
           splitted.append(splitted_img)
    
    return np.asarray(splitted)


def save_images(images, folder, base_index):
    for i in range(images.shape[0]):
        name = folder + str(base_index + i) + ".png"
        print("saving: " + name + "    " + str((images[i].min(), images[i].max())))
        img = Image.fromarray(images[i].astype(np.uint8))
        img.save(name)
        


def split_images(input_path = "chicago/", output_path = "split/images/"):
    os.makedirs(output_path, exist_ok = True)
    x_files = sorted(glob.glob(input_path + "*_image.png"))
    
    # rescale factor of 0.35 to make the proportions similar to the training images
    x = np_from_files(x_files, rescale_factor = 0.35)
    
    crt_base_index = 0
    for i in range(len(x)):
        splitted_x = split_data(x[i])
        save_images(splitted_x, output_path + "chicago_", crt_base_index)
        crt_base_index += splitted_x.shape[0]
        


def split_labels(inpt_path = "chicago/", output_path = "split/labels/"):
    os.makedirs(output_path, exist_ok = True)
    y_files = sorted(glob.glob('chicago/*_labels.png'))
    
    # rescale factor of 0.35 to make the proportions similar to the training images
    y = np_from_files(y_files, rescale_factor = 0.35, labels = True)
    
    crt_base_index = 0
    for i in range(len(y)):
        splitted_y = split_data(y[i])
        save_images(splitted_y, output_path + "chicago_", crt_base_index)
        crt_base_index += splitted_y.shape[0]







def main():
    input_path = "generate_data/chicago/"
    output_path = "input/chicago/"
    split_images(input_path, output_path = output_path)
    split_labels(input_path, output_path = output_path)
    

if __name__ == "__main__":
    main()
