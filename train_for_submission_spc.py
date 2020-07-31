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

from models import simple_patch_conv as spc
from submission import model_to_submission as submission



def train_sub(model, model_name, train, val, epochs=100, batch_size=8, verbose=2, transform_pred=None):
    tf.compat.v1.reset_default_graph()
    tf.random.set_seed(42424242)
    tf.compat.v1.set_random_seed(42424242)

    print('\n\n\nMODEL:' + model_name)
    earlystopper = EarlyStopping(patience=8, verbose=2)
    os.makedirs("checkpoints", exist_ok = True)
    model_path_name = 'checkpoints/ckp_{}.h5'.format(model_name)
    checkpointer = ModelCheckpoint(model_path_name, verbose=1, save_best_only=True)

    os.makedirs("tf_logs", exist_ok = True)
    log_dir = "tf_logs/" + model_name + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit(train, validation_data=val,epochs=epochs,
                        #callbacks=[earlystopper, checkpointer, tensorboard_callback],
                        callbacks=[earlystopper, checkpointer],
                        verbose=verbose)

    model.load_weights(model_path_name)

    submission.create(model, model_name, transform_pred)




def transform_pred(pred):
    pred = np.squeeze(pred)
    pred = np.kron(pred, np.ones((16, 16), dtype=int))
    return pred




def main():

    # no augmentation
    
    train, val = data_g.get_train_val_iterators_spc(aug=None, batch_size=64)

    model = spc.get_model(None, None, 3, 4096, do_compile=False)
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy', f1, tf.keras.metrics.MeanIoU(num_classes=2)])
    model_name = 'spc4096'
    train_sub(model, model_name, train, val, epochs=100, verbose=1, transform_pred=transform_pred)
    



if __name__ == "__main__":
    main()
