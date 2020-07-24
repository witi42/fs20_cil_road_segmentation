import preproc.get_data as data
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import glob

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

from models import unet
from submission import model_to_submission as submission



def train_sub(model, model_name, x, y, validation_data, epochs=100, batch_size=8, verbose=2):
    tf.compat.v1.reset_default_graph()
    tf.random.set_seed(42424242)
    tf.compat.v1.set_random_seed(42424242)

    print('\n\n\nMODEL:' + model_name)
    earlystopper = EarlyStopping(patience=10, verbose=2)
    model_path_name = 'checkpoints/ckp_{}.h5'.format(model_name)
    checkpointer = ModelCheckpoint(model_path_name, verbose=1, save_best_only=True)

    history = model.fit(x, y, validation_data=validation_data,
                        batch_size=batch_size, epochs=epochs,
                        callbacks=[earlystopper, checkpointer],
                        verbose=verbose)

    model.load_weights(model_path_name)
    submission.create(model, model_name)



def main():
    x1, y1 = data.get_training_data()
    x2, y2 = data.get_training_data2()

    x1_train, x_test, y1_train, y_test = train_test_split(x1, y1, test_size=0.3, random_state=42424242)

    x = np.concatenate(x1_train, x2)
    y = np.concatenate(y1_train, y2)

    #x, y = data.augment_data(x, y)

    # x, y = data.augment_data(x,y)


    # # crossentropy
    # model = unet.get_model(None, None, 3, do_compile=False)
    # model.compile(optimizer='adam', loss='binary_crossentropy',
    #               metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2)])
    # model_name = 'u_net_cross_entropy_test'
    # train_sub(model, model_name, x, y, epochs=1)


    # # focal
    # from losses import focal
    # loss = focal.focal_loss
    # model_name = 'u_net_focal_loss'
    # model = unet.get_model(None, None, 3, do_compile=False)
    # model.compile(optimizer='adam', loss=loss,
    #               metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2)])
    # train_sub(model, model_name, x, y, epochs=1)


    # # dice
    # from losses import dice
    # loss = dice.dice_loss
    # model_name = 'u_net_dice'
    # model = unet.get_model(None, None, 3, do_compile=False)
    # model.compile(optimizer='adam', loss=loss,
    #               metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2)])
    # train_sub(model, model_name, x, y, epochs=1)



    # lovasz
    from losses import lovasz
    loss = lovasz.lovasz_hinge
    model_name = 'u_net_lovasz_EXT_DATA'
    model = unet.get_model(None, None, 3, do_compile=False)
    model.compile(optimizer='adam', loss=loss,
                  metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2)])
    train_sub(model, model_name, x, y, (x_test, y_test), epochs=100)


if __name__ == "__main__":
    main()
