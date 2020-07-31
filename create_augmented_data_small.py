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

path_src_val = 'input/ds/val/'
path_small_val = 'input/ds_aug_small/val/'




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



def augment_label(label, base_tag):
    label = np.asarray([label])
    
    flip_rot_label = data.flip_and_rotate(label, discard_original = True)
    save_list(flip_rot_label, base_tag + '_flip_rotate', path_small[1])




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

def create_all_ds():

    src = ('input/ds/train/image/', 'input/ds/train/label/')
    src_val = ('input/ds/val/image/', 'input/ds/val/label/')


    #create folders
    os.makedirs(src_val[0], exist_ok = True)
    os.makedirs(src_val[1], exist_ok = True)
    os.makedirs(src[0], exist_ok = True)
    os.makedirs(src[1], exist_ok = True)


    original_i = glob.glob('input/original/image/*.png')
    original_l = glob.glob('input/original/label/*.png')

    chicago_i = glob.glob('input/chicago/image/*.png')
    chicago_l = glob.glob('input/chicago/label/*.png')

    train_i = original_i + chicago_i
    train_l = original_l + chicago_l

    val_i = original_i
    val_l = original_l

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
    #create ds with split for validation
    create_split_ds()
    # #no split
    # create_all_ds()



    # handle folders
    os.makedirs(path_small[0], exist_ok = True)
    os.makedirs(path_small[1], exist_ok = True)
    
    os.makedirs(path_small_val, exist_ok = True)
    
    # copy validation set
    copy_tree(path_src_val, path_small_val)
    
    
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
        augment_image(image, base_tag)
    
    random.seed(42424242)
    for i in range(len(label_files)):
    #for i in range(1):
        label = Image.open(label_files[i])
        label = np.asarray(label) / 255.
        base_tag = os.path.basename(label_files[i])[:-4]
        save_image(label, path_small[1] + base_tag + '.png')
        augment_label(label, base_tag)
        
    

if __name__ == "__main__":
    main()
