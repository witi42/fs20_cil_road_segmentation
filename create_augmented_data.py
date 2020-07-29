from PIL import Image
import glob
import numpy as np
import random
import os
import preproc.get_data as data
from sklearn.model_selection import train_test_split
from distutils.dir_util import copy_tree
import shutil


from sklearn.model_selection import train_test_split



path_src = ('input/ds/train/image/', 'input/ds/train/label/')
path_small = ('input/ds_aug_small/train/image/', 'input/ds_aug_small/train/label/')
path_large = ('input/ds_aug_large/train/image/', 'input/ds_aug_large/train/label/')
path_intermediate = ('input/ds_aug_intermediate/train/image/', 'input/ds_aug_intermediate/train/label/')

path_src_val = 'input/ds/val/'
path_small_val = 'input/ds_aug_small/val/'
path_large_val = 'input/ds_aug_large/val/'
path_intermediate_val = 'input/ds_aug_intermediate/val/'




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




def augment_image(image, base_tag):
    image = np.asarray([image])
    
    
    flip_rot = data.flip_and_rotate(image, discard_original = True)
    save_list(flip_rot, base_tag + '_flip_rotate', path_small[0])                   # flip + 90-rotate for all datasets
    save_list(flip_rot, base_tag + '_flip_rotate', path_large[0])
    save_list(flip_rot, base_tag + '_flip_rotate', path_intermediate[0])
    
    desat = data.saturate(image, 0.5, discard_original = True)
    save_list(desat, base_tag + '_desat', path_large[0])                            # desaturated for large and intermediate datasets
    save_list(desat, base_tag + '_desat', path_intermediate[0])
    
    flip_rot_desat = data.saturate(flip_rot, 0.5, discard_original = True)
    save_list(flip_rot_desat, base_tag + '_flip_rotate_desat', path_large[0])       # flip + 90-rotate + desaturated for large and intermediate datasets
    save_list(flip_rot_desat, base_tag + '_flip_rotate_desat', path_intermediate[0])
    
    
    rand_rot = data.random_rotate(image, num_rotations = 3, seed = None, discard_original = True, sharpen = True)
    rand_rot_intermediate = data.random_rotate(image, num_rotations = 1, seed = None, discard_original = True, sharpen = True)
    save_list(rand_rot, base_tag + '_rand_rot', path_large[0])                      # 3 rotations for large dataset only
    save_list(rand_rot_intermediate, base_tag + '_rand_rot', path_intermediate[0])  # 1 roation for intermediate dataset only
    
    flip_rot_rand_rot = data.random_rotate(flip_rot, num_rotations = 3, seed = None, discard_original = True, sharpen = True)
    flip_rot_rand_rot_intermediate = data.random_rotate(flip_rot, num_rotations = 1, seed = None, discard_original = True, sharpen = True)
    save_list(flip_rot_rand_rot, base_tag + '_flip_rot_rand_rot', path_large[0])
    save_list(flip_rot_rand_rot_intermediate, base_tag + '_flip_rot_rand_rot', path_intermediate[0])
    
    desat_rand_rot = data.random_rotate(desat, num_rotations = 3, seed = None, discard_original = True, sharpen = True)
    desat_rand_rot_intermediate = data.random_rotate(desat, num_rotations = 1, seed = None, discard_original = True, sharpen = True)
    save_list(desat_rand_rot, base_tag + '_desat_rand_rot', path_large[0])
    save_list(desat_rand_rot_intermediate, base_tag + '_desat_rand_rot', path_intermediate[0])
    
    flip_rot_desat_rand_rot = data.random_rotate(flip_rot_desat, num_rotations = 3, seed = None, discard_original = True, sharpen = True)
    flip_rot_desat_rand_rot_intermediate = data.random_rotate(flip_rot_desat, num_rotations = 1, seed = None, discard_original = True, sharpen = True)
    save_list(flip_rot_desat_rand_rot, base_tag + '_flip_rot_desat_rand_rot', path_large[0])
    save_list(flip_rot_desat_rand_rot_intermediate, base_tag + '_flip_rot_desat_rand_rot', path_intermediate[0])



def augment_label(label, base_tag):
    label = np.asarray([label])
    
    
    flip_rot_label = data.flip_and_rotate(label, discard_original = True)
    save_list(flip_rot_label, base_tag + '_flip_rotate', path_small[1])
    save_list(flip_rot_label, base_tag + '_flip_rotate', path_large[1])
    save_list(flip_rot_label, base_tag + '_flip_rotate', path_intermediate[1])
    
    desat_label = label
    save_list(desat_label, base_tag + '_desat', path_large[1])
    save_list(desat_label, base_tag + '_desat', path_intermediate[1])
    
    flip_rot_desat_label = flip_rot_label
    save_list(flip_rot_desat_label, base_tag + '_flip_rotate_desat', path_large[1])
    save_list(flip_rot_desat_label, base_tag + '_flip_rotate_desat', path_intermediate[1])
    
    
    rand_rot_label = data.random_rotate(label, num_rotations = 3, seed = None, discard_original = True, sharpen = True)
    rand_rot_label_intermediate = data.random_rotate(label, num_rotations = 1, seed = None, discard_original = True, sharpen = True)
    save_list(rand_rot_label, base_tag + '_rand_rot', path_large[1])
    save_list(rand_rot_label_intermediate, base_tag + '_rand_rot', path_intermediate[1])
    
    flip_rot_rand_rot_label = data.random_rotate(flip_rot_label, num_rotations = 3, seed = None, discard_original = True, sharpen = True)
    flip_rot_rand_rot_label_intermediate = data.random_rotate(flip_rot_label, num_rotations = 1, seed = None, discard_original = True, sharpen = True)
    save_list(flip_rot_rand_rot_label, base_tag + '_flip_rot_rand_rot', path_large[1])
    save_list(flip_rot_rand_rot_label_intermediate, base_tag + '_flip_rot_rand_rot', path_intermediate[1])
    
    desat_rand_rot_label = data.random_rotate(desat_label, num_rotations = 3, seed = None, discard_original = True, sharpen = True)
    desat_rand_rot_label_intermediate = data.random_rotate(desat_label, num_rotations = 1, seed = None, discard_original = True, sharpen = True)
    save_list(desat_rand_rot_label, base_tag + '_desat_rand_rot', path_large[1])
    save_list(desat_rand_rot_label_intermediate, base_tag + '_desat_rand_rot', path_intermediate[1])
    
    flip_rot_desat_rand_rot_label = data.random_rotate(flip_rot_desat_label, num_rotations = 3, seed = None, discard_original = True, sharpen = True)
    flip_rot_desat_rand_rot_label_intermediate = data.random_rotate(flip_rot_desat_label, num_rotations = 1, seed = None, discard_original = True, sharpen = True)
    save_list(flip_rot_desat_rand_rot_label, base_tag + '_flip_rot_desat_rand_rot', path_large[1])
    save_list(flip_rot_desat_rand_rot_label_intermediate, base_tag + '_flip_rot_desat_rand_rot', path_intermediate[1])




def create_split_ds():
    src = ('input/ds/train/image/', 'input/ds/train/label/')
    src_val = ('input/ds/val/image/', 'input/ds/val/label/')



    #create folders
    os.makedirs(src_val[0], exist_ok = True)
    os.makedirs(src_val[1], exist_ok = True)
    os.makedirs(src[0], exist_ok = True)
    os.makedirs(src[1], exist_ok = True)


    original_i = glob.glob('input/original/image/*.png')
    original_l = glob.glob('input/original/label/*.png')

    o_train_i, val_i = train_test_split(original_i, test_size=0.3, random_state=42424242)
    o_train_l, val_l = train_test_split(original_l, test_size=0.3, random_state=42424242)

    print('i',o_train_i)
    print('l',o_train_l)

    chicago_i = glob.glob('input/chicago/image/*.png')
    chicago_l = glob.glob('input/chicago/label/*.png')

    train_i = o_train_i + chicago_i
    train_l = o_train_l + chicago_l

    #copy files
    for f in train_i:
        shutil.copy(f, src[0])
    for f in train_l:
        shutil.copy(f, src[1])
    for f in val_i:
        shutil.copy(f, src_val[0])
    for f in val_l:
        shutil.copy(f, src_val[1])



def main():
    #create ds
    #create_split_ds()



    # handle folders
    os.makedirs(path_small[0], exist_ok = True)
    os.makedirs(path_small[1], exist_ok = True)
    os.makedirs(path_large[0], exist_ok = True)
    os.makedirs(path_large[1], exist_ok = True)
    os.makedirs(path_intermediate[0], exist_ok = True)
    os.makedirs(path_intermediate[1], exist_ok = True)
    
    os.makedirs(path_small_val, exist_ok = True)
    os.makedirs(path_large_val, exist_ok = True)
    os.makedirs(path_intermediate_val, exist_ok = True)
    
    # copy validation set
    copy_tree(path_src_val, path_small_val)
    copy_tree(path_src_val, path_large_val)
    copy_tree(path_src_val, path_intermediate_val)
    
    
    # getting original training files
    image_files = sorted(glob.glob(path_src[0] + '*.png'))
    label_files = sorted(glob.glob(path_src[1] + '*.png'))

    
    random.seed(42424242)
    for i in range(len(image_files)):
    #for i in range(1):
        image = Image.open(image_files[i])
        image = np.asarray(image) / 255.
        base_tag = os.path.basename(label_files[i])[:-4]
        save_image(image, path_small[0] + base_tag + '.png')
        save_image(image, path_large[0] + base_tag + '.png')
        save_image(image, path_intermediate[0] + base_tag + '.png')
        augment_image(image, base_tag)
    
    random.seed(42424242)
    for i in range(len(label_files)):
    #for i in range(1):
        label = Image.open(label_files[i])
        label = np.asarray(label) / 255.
        base_tag = os.path.basename(label_files[i])[:-4]
        save_image(label, path_small[1] + base_tag + '.png')
        save_image(label, path_large[1] + base_tag + '.png')
        save_image(label, path_intermediate[1] + base_tag + '.png')
        augment_label(label, base_tag)
        
    

if __name__ == "__main__":
    main()
