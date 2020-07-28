import preproc.get_data as data
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import glob
import datetime

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

from models import unet
from submission import model_to_submission as submission
from preproc import data_generator



def main():
    x1, y1 = data.get_training_data()
    x2, y2 = data.get_training_data2()

    print('x1, y1',x1.shape, y1.shape)
    x1_train, x_test, y1_train, y_test = train_test_split(x1, y1, test_size=0.3, random_state=42424242)


    print('x1_train, x2', x1_train.shape, x2.shape, x1_train.dtype, x2.dtype)
    x = np.concatenate((x1_train, x2), axis=0)
    print(y1_train.shape, y2.shape)
    y = np.concatenate((y1_train, y2), axis=0)

    from models import tf_unet

    model = tf_unet.get_model()
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    train = data_generator.get_train_iter224()
    val = data_generator.get_val_iter224()

    checkpointer = ModelCheckpoint('checkpoints/tf_unet.h5', verbose=1, save_best_only=True)

    model.fit(train, validation_data=val, epochs=100, callbacks=[checkpointer])

    model_pred = tf_unet.tf_unet(checkpoint='checkpoints/tf_unet.h5')

    submission.create_from_split(model_pred, 'tf_unet_sparsecrossentropy')


if __name__ == "__main__":
    main()
