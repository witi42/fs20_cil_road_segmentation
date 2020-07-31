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

    submission.create_with_split(model_pred, 'tf_unet_sparsecrossentropy')


if __name__ == "__main__":
    main()
