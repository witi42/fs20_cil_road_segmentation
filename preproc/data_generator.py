# inspired by https://mahmoudyusof.github.io/facial-keypoint-detection/data-generator/

import glob
import math
import numpy as np
from PIL import Image
from tensorflow.keras.utils import Sequence


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

        y = y.reshape(list(y.shape) + [1])
        
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


def get_train_iter224(images='input/large_dataset/train/image/*.png', labels='input/large_dataset/train/label/*.png', batch_size=8):
    return DataGenerator224(images, labels, batch_size)

def get_val_iter224(images='input/large_dataset/val/image/*.png', labels='input/large_dataset/val/label/*.png', batch_size=8):
    return DataGenerator224(images, labels, batch_size)
