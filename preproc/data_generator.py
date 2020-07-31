# inspired by https://mahmoudyusof.github.io/facial-keypoint-detection/data-generator/

import glob
import math
import numpy as np
from PIL import Image
from tensorflow.keras.utils import Sequence


# iterate over dataset provided as images stored in folders
class DataGenerator(Sequence):

    def __init__(self, images, labels, batch_size=8):

        self.images = sorted(glob.glob(images))
        self.labels = sorted(glob.glob(labels))
        self.num_images = len(self.images)
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(self.num_images / self.batch_size)

    def __getitem__(self, idx):
        start = idx*self.batch_size
        end = min(self.num_images, (idx+1)*self.batch_size)
        
        x = self.np_from_files(self.images[start:end])
        y = self.np_from_files(self.labels[start:end])
        
        x = x.astype(np.float32) / 255.
        y = y.astype(np.float32) / 255.
        
        y = (y > 0.5).astype(np.uint8)

        #y = y.reshape(list(y.shape) + [1])
        
        return x,y

    def np_from_files(self, files: list) -> np.ndarray:
        x = []
        for f in files:
            image = Image.open(f)
            image = np.asarray(image)            
            x.append(image)
        
        return np.asarray(x)

def get_train_iter(images='input/large_dataset/train/image/*.png', labels='input/large_dataset/train/label/*.png', batch_size=8):
    return DataGenerator(images, labels, batch_size)

def get_val_iter(images='input/large_dataset/val/image/*.png', labels='input/large_dataset/val/label/*.png', batch_size=8):
    return DataGenerator(images, labels, batch_size)

#returns train and validation iterator
#aug= choose augmentation level: None, 'small', 'medium', 'large'
def get_train_val_iterators(aug=None, base_dir='input/ds/', batch_size=8):
    if aug == 'small':
        base_dir = 'input/ds_aug_small/'
    if aug == 'medium':
        base_dir = 'input/ds_aug_intermediate/'
    if aug == 'large':
        base_dir = 'input/ds_aug_large/'
    
    train_it = get_train_iter(images= base_dir + 'train/image/*.png', labels= base_dir + 'train/label/*.png', batch_size=batch_size)
    val_it = get_val_iter(images= base_dir + 'val/image/*.png', labels= base_dir + 'val/label/*.png', batch_size=batch_size)

    return train_it, val_it


class DataGenerator224(Sequence):

    def __init__(self, images, labels, batch_size=8):

        self.images = sorted(glob.glob(images))
        self.labels = sorted(glob.glob(labels))
        self.num_images = len(self.images)
        self.batch_size = math.ceil(batch_size/4)

    def __len__(self):
        return math.ceil(self.num_images / self.batch_size)

    def __getitem__(self, idx):
        start = idx*self.batch_size
        end = min(self.num_images, (idx+1)*self.batch_size)
        
        x = self.np_from_files(self.images[start:end])
        y = self.np_from_files(self.labels[start:end])
        
        x = x.astype(np.float32) / 255.
        y = y.astype(np.float32) / 255.
        
        y = (y > 0.5).astype(np.uint8)

        y = y.reshape(list(y.shape) + [1])
        
        return x,y

    def np_from_files(self, files: list) -> np.ndarray:
        x = []
        for f in files:
            image = Image.open(f)
            image = np.asarray(image)
            shape = image.shape
            x.append(image[0:224, 0:224])
            x.append(image[0:224, shape[1]-224:shape[1]])
            x.append(image[shape[0]-224:shape[0], 0:224])
            x.append(image[shape[0]-224:shape[0], shape[1]-224:shape[1]])

        return np.asarray(x)


def get_train_iter224(images='input/ds/train/image/*.png', labels='input/ds/train/label/*.png', batch_size=8):
    return DataGenerator224(images, labels, batch_size)

def get_val_iter224(images='input/ds/val/image/*.png', labels='input/ds/val/label/*.png', batch_size=8):
    return DataGenerator224(images, labels, batch_size)



class DataGeneratorSpc(Sequence):

    def __init__(self, images, labels, batch_size=8):

        self.images = sorted(glob.glob(images))
        self.labels = sorted(glob.glob(labels))
        self.num_images = len(self.images)
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(self.num_images / self.batch_size)

    def __getitem__(self, idx):
        start = idx*self.batch_size
        end = min(self.num_images, (idx+1)*self.batch_size)
        
        x = self.np_from_files(self.images[start:end])
        y = self.np_from_files(self.labels[start:end])
        
        x = x.astype(np.float32) / 255.
        y = y.astype(np.float32) / 255.
        
        y = (y > 0.5).astype(np.uint8)

        y = self.transform_y(y)
        
        return x,y

    def np_from_files(self, files: list) -> np.ndarray:
        x = []
        for f in files:
            image = Image.open(f)
            image = np.asarray(image)            
            x.append(image)
        
        return np.asarray(x)
    
    def patch_to_label(self, patch):
        df = np.mean(patch)
        if df > 0.25:
            return 1
        else:
            return 0
    
    def transform_y(self, y):
        l = []
        num_patches = (int(y.shape[1] / 16), int(y.shape[2] / 16))
        for img_idx in range(y.shape[0]):
            img = np.empty(num_patches)
            for i in range(num_patches[0]):
                for j in range(num_patches[1]):
                    patch = y[img_idx, i*16 : (i+1)*16, j*16 : (j+1)*16]
                    img[i, j] = self.patch_to_label(patch)
            l.append(img)
        return np.asarray(l)


def get_train_iter_spc(images='input/ds/train/image/*.png', labels='input/ds/train/label/*.png', batch_size=8):
    return DataGeneratorSpc(images, labels, batch_size)

def get_val_iter_spc(images='input/ds/val/image/*.png', labels='input/ds/val/label/*.png', batch_size=8):
    return DataGeneratorSpc(images, labels, batch_size)

#returns train and validation iterator
#aug= choose augmentation level: None, 'small', 'medium', 'large'
def get_train_val_iterators_spc(aug=None, base_dir='input/ds/', batch_size=8):
    if aug == 'small':
        base_dir = 'input/ds_aug_small/'
    if aug == 'medium':
        base_dir = 'input/ds_aug_intermediate/'
    if aug == 'large':
        base_dir = 'input/ds_aug_large/'
    
    train_it = get_train_iter_spc(images= base_dir + 'train/image/*.png', labels= base_dir + 'train/label/*.png', batch_size=batch_size)
    val_it = get_val_iter_spc(images= base_dir + 'val/image/*.png', labels= base_dir + 'val/label/*.png', batch_size=batch_size)

    return train_it, val_it