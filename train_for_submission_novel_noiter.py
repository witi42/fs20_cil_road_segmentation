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

from  metrics.f1 import f1
from models import cnn
from models import unet
from submission import model_to_submission as submission



def train_sub(model, model_name, x, y, validation_data = None, epochs=100, batch_size=8, verbose=2):
    tf.compat.v1.reset_default_graph()
    tf.random.set_seed(42424242)
    tf.compat.v1.set_random_seed(42424242)

    print('\n\n\nMODEL:' + model_name)
    earlystopper = EarlyStopping(patience=8, verbose=2)
    model_path_name = 'checkpoints/ckp_{}.h5'.format(model_name)
    checkpointer = ModelCheckpoint(model_path_name, verbose=1, save_best_only=False)

    os.makedirs("tf_logs", exist_ok = True)
    log_dir = "tf_logs/" + model_name + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit(x, y, validation_data=validation_data,
                        batch_size=batch_size, epochs=epochs,
                        callbacks=[checkpointer, tensorboard_callback, earlystopper],
                        verbose=verbose)

    model.load_weights(model_path_name)
    submission.create(model, model_name + '_608')
    submission.create_with_split(model, model_name + '_400split')



def main():
    x1, y1 = data.get_training_data()
    x2, y2 = data.get_training_data2()


    x = np.concatenate((x1, x2), axis=0)
    y = np.concatenate((y1, y2), axis=0)

             
    from models.sdf_model import get_CNN_SDFt, get_flat_tanh_CNN_SDFt

    # # sdf flat tanh
    # train, val = data_g.get_train_val_iterators(aug=None)

    # model = get_flat_tanh_CNN_SDFt()
    # model_name = 'SDF_cnn_scaled_tanh_EXTDATA_noaug'

    # train_sub(model, model_name, x, y, epochs=100)
    # submission.create(model, model_name)


    # sdf normal tanh

    model = get_CNN_SDFt()
    model_name = 'SDF_cnn_tanh_EXTDATA_noaug'

    train_sub(model, model_name, x, y, epochs=100)
    submission.create(model, model_name)
    



if __name__ == "__main__":
    main()
