from PIL import Image
import glob
import numpy as np
import random
import os
import preproc.get_data as data
from sklearn.model_selection import train_test_split

import multiprocessing
from joblib import Parallel, delayed



def save_image(image, name):
        image = image * 255
        print("saving: " + name + "    " + str((image.min(), image.max())))
        img = Image.fromarray(image.astype(np.uint8))
        img.save(name)


def save_image_label_list(images, labels, tag, path):
    if images.shape[0] != labels.shape[0]:
        raise "image and label count needs to match!"
    
    for i in range(images.shape[0]):
        save_image(images[i], path[0] + tag + '_' + str(i) + '.png')
        save_image(labels[i], path[1] + tag + '_' + str(i) + '.png')


def save_list(l, tag, path):
    for i in range(l.shape[0]):
        save_image(l[i], path + tag + '_' + str(i) + '.png')




def augment_image(image, base_tag, path):
        image = np.asarray([image])
        
        
        flip_rot = data.flip_and_rotate(image, discard_original = True)
        save_list(flip_rot, base_tag + '_flip_rotate', path)
        
        desat = data.saturate(image, 0.5, discard_original = True)
        save_list(desat, base_tag + '_desat', path)
        
        flip_rot_desat = data.saturate(flip_rot, 0.5, discard_original = True)
        save_list(flip_rot_desat, base_tag + '_flip_rotate_desat', path)
        
        
        rand_rot = data.random_rotate(image, num_rotations = 3, seed = None, discard_original = True, sharpen = True)
        save_list(rand_rot, base_tag + '_rand_rot', path)
        
        flip_rot_rand_rot = data.random_rotate(flip_rot, num_rotations = 3, seed = None, discard_original = True, sharpen = True)
        save_list(flip_rot_rand_rot, base_tag + '_flip_rot_rand_rot', path)
        
        desat_rand_rot = data.random_rotate(desat, num_rotations = 3, seed = None, discard_original = True, sharpen = True)
        save_list(desat_rand_rot, base_tag + '_desat_rand_rot', path)
        
        flip_rot_desat_rand_rot = data.random_rotate(flip_rot_desat, num_rotations = 3, seed = None, discard_original = True, sharpen = True)
        save_list(flip_rot_desat_rand_rot, base_tag + '_flip_rot_desat_rand_rot', path)



def augment_label(label, base_tag, path):
    label = np.asarray([label])
    
    
    flip_rot_label = data.flip_and_rotate(label, discard_original = True)
    save_list(flip_rot_label, base_tag + '_flip_rotate', path)
    
    desat_label = label
    save_list(desat_label, base_tag + '_desat', path)
    
    flip_rot_desat_label = flip_rot_label
    save_list(flip_rot_desat_label, base_tag + '_flip_rotate_desat', path)
    
    
    rand_rot_label = data.random_rotate(label, num_rotations = 3, seed = None, discard_original = True, sharpen = True)
    save_list(rand_rot_label, base_tag + '_rand_rot', path)
    
    flip_rot_rand_rot_label = data.random_rotate(flip_rot_label, num_rotations = 3, seed = None, discard_original = True, sharpen = True)
    save_list(flip_rot_rand_rot_label, base_tag + '_flip_rot_rand_rot', path)
    
    desat_rand_rot_label = data.random_rotate(desat_label, num_rotations = 3, seed = None, discard_original = True, sharpen = True)
    save_list(desat_rand_rot_label, base_tag + '_desat_rand_rot', path)
    
    flip_rot_desat_rand_rot_label = data.random_rotate(flip_rot_desat_label, num_rotations = 3, seed = None, discard_original = True, sharpen = True)
    save_list(flip_rot_desat_rand_rot_label, base_tag + '_flip_rot_desat_rand_rot', path)





def main():
    path_train = ('input_full/train/image/', 'input_full/train/label/')
    path_val = ('input_full/val/image/', 'input_full/val/label/')
    #path_temp = ('input_full/temp/image/', 'input_full/temp/label/')
    
    os.makedirs(path_train[0], exist_ok = True)
    os.makedirs(path_train[1], exist_ok = True)
    os.makedirs(path_val[0], exist_ok = True)
    os.makedirs(path_val[1], exist_ok = True)
    
    original, original_label = data.get_training_data()
    original_train, original_val, original_train_label, original_val_label = train_test_split(original, original_label, test_size=0.3, random_state=42424242)
    
    save_image_label_list(original_train, original_train_label, 'original', path_train)
    save_image_label_list(original_val, original_val_label, 'original', path_val)
    
    random.seed(42424242)
    for i in range(original_train.shape[0]):
        augment_image(original_train[i], 'original_' + str(i), path_train[0])
    
    
    random.seed(42424242)
    for i in range(original_train_label.shape[0]):
        augment_label(original_train_label[i], 'original_' + str(i), path_train[1])
    
    
    chicago_image_files = sorted(glob.glob('chicago_data/split_quarter/images/*.png'))
    chicago_label_files = sorted(glob.glob('chicago_data/split_quarter/labels/*.png'))
    
    random.seed(42424242)
    for i in range(len(chicago_image_files)):
        image = Image.open(chicago_image_files[i])
        image = np.asarray(image) / 255.
        save_image(image, path_train[0] + 'chicago_' + str(i) + '.png')
        augment_image(image, 'chicago_' + str(i), path_train[0])
    
    random.seed(42424242)
    for i in range(len(chicago_image_files)):
        label = Image.open(chicago_label_files[i])
        label = np.asarray(label) / 255.
        save_image(label, path_train[1] + 'chicago_' + str(i) + '.png')
        augment_label(label, 'chicago_' + str(i), path_train[1])
        
    

if __name__ == "__main__":
    main()
