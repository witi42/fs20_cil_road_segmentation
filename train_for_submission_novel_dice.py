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
    earlystopper = EarlyStopping(patience=20, verbose=2)
    os.makedirs("checkpoints", exist_ok = True)
    model_path_name = 'checkpoints/ckp_{}.h5'.format(model_name)
    checkpointer = ModelCheckpoint(model_path_name, verbose=1, save_best_only=True)

    os.makedirs("tf_logs", exist_ok = True)
    log_dir = "tf_logs/" + model_name + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit(train, validation_data=val,epochs=epochs,
                        callbacks=[earlystopper, checkpointer, tensorboard_callback], verbose=verbose)

    model.load_weights(model_path_name)

    submission.create(model, model_name)



def main():

    from models.sdf_model import get_CNN_SDFt, get_flat_tanh_CNN_SDFt
    from losses.sdf_adapter import convert_to_sdft_loss
    from losses.dice import dice_loss
    
    train, val = data_g.get_train_val_iterators(aug=None)


    msebce = lambda x, y: tf.keras.losses.MSE(x, y) + convert_to_sdft_loss(tf.keras.losses.binary_crossentropy)(x, y)
    msedice = lambda x, y: tf.keras.losses.MSE(x, y) + convert_to_sdft_loss(dice_loss)(x, y)

    # model = get_CNN_SDFt(loss=msedice)
    # model_name = 'SDF-tanh_msedice_loss'

    # train_sub(model, model_name, train, val, epochs=100, verbose=1)


    model = get_flat_tanh_CNN_SDFt(loss=msedice)
    model_name = 'SDF-flat_tanh_msedice_loss'

    train_sub(model, model_name, train, val, epochs=100, verbose=1)


    # for _aug in ['small', 'medium', 'large']:
    #     train, val = data_g.get_train_val_iterators(aug=_aug)
    #     model = get_flat_tanh_CNN_SDFt()
    #     model_name = 'SDF_cnn_scaled_tanh_EXTDATA_aug-' + _aug

    #     train_sub(model, model_name, train, val, epochs=100, verbose=1)
    #     submission.create(model, model_name)

    #     # sdf normal tanh

    #     model = get_CNN_SDFt()
    #     model_name = 'SDF_cnn_tanh_EXTDATA_aug-' + _aug

    #     train_sub(model, model_name, train, val, epochs=100, verbose=1)
    #     submission.create(model, model_name)




if __name__ == "__main__":
    main()
