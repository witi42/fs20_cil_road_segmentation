from preproc import data_generator as data_g
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import glob
import datetime
from  metrics.f1 import f1

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

from models import unet2 as unet
from submission import model_to_submission as submission



def train_sub(model, model_name, train, val, epochs=100, batch_size=8, verbose=2):
    tf.compat.v1.reset_default_graph()
    tf.random.set_seed(42424242)
    tf.compat.v1.set_random_seed(42424242)

    print('\n\n\nMODEL:' + model_name)
    earlystopper = EarlyStopping(patience=8, verbose=2)
    model_path_name = 'checkpoints/ckp_{}.h5'.format(model_name)
    checkpointer = ModelCheckpoint(model_path_name, verbose=1, save_best_only=True)

    os.makedirs("tf_logs", exist_ok = True)
    log_dir = "tf_logs/" + model_name + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit(train, validation_data=val,epochs=epochs,
                        callbacks=[earlystopper, checkpointer, tensorboard_callback],
                        verbose=verbose)

    model.load_weights(model_path_name)
    submission.create(model, model_name)



def main():

    # no augmentation
    
    train, val = data_g.get_train_val_iterators(aug=None)

    model = unet.get_model(None, None, 3, do_compile=False)
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy', f1, tf.keras.metrics.MeanIoU(num_classes=2)])
    model_name = 'unet2_crossentropy_augmentation_none'
    train_sub(model, model_name, train, val, epochs=100, verbose=1)

    submission.create(model, model_name)

        
        
    # small augmentation

    train, val = data_g.get_train_val_iterators(aug='small')

    model = unet.get_model(None, None, 3, do_compile=False)
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy', f1, tf.keras.metrics.MeanIoU(num_classes=2)])
    model_name = 'unet2_crossentropy_augmentation_small'
    train_sub(model, model_name, train, val, epochs=100, verbose=1)

    submission.create(model, model_name)




    # medium augmentation

    train, val = data_g.get_train_val_iterators(aug='medium')

    model = unet.get_model(None, None, 3, do_compile=False)
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy', f1, tf.keras.metrics.MeanIoU(num_classes=2)])
    model_name = 'unet2_crossentropy_augmentation_medium'
    train_sub(model, model_name, train, val, epochs=100, verbose=1)

    submission.create(model, model_name)



    # large augmentation

    train, val = data_g.get_train_val_iterators(aug='large')

    model = unet.get_model(None, None, 3, do_compile=False)
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy', f1, tf.keras.metrics.MeanIoU(num_classes=2)])
    model_name = 'unet2_crossentropy_augmentation_large'
    train_sub(model, model_name, train, val, epochs=100, verbose=1)

    submission.create(model, model_name)



if __name__ == "__main__":
    main()
