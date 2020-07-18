import preproc.get_data as data
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import glob

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

from models import unet
from submission import mask_to_submission


def train_sub(model, model_name, x, y, epochs=100, batch_size=8, verbose=2):
    print('\n\n\nMODEL:' + model_name)
    earlystopper = EarlyStopping(patience=15, verbose=2)
    model_path_name = 'checkpoints/ckp_{}.h5'.format(model_name)
    checkpointer = ModelCheckpoint(model_path_name, verbose=1, save_best_only=True)

    history = model.fit(x, y, validation_split=0.1,
                        batch_size=batch_size, epochs=epochs,
                        callbacks=[earlystopper, checkpointer],
                        verbose=verbose)

    model.load_weights(model_path_name)
    create_sub(model, model_name)


def create_sub(model, name):
    x, x_names = data.get_test_data()

    if not os.path.exists('output'):
        os.makedirs('output')

    for i in range(len(x_names)):
        name = x_names[i][18:]
        print(name)

        pred = model.predict(x[i:i + 1])
        pred = pred.reshape(608, 608)
        pred = (pred > 0.5).astype(np.uint8)

        plt.imsave("output/" + name, pred, cmap=cm.gray)

    if not os.path.exists('submission_csv'):
        os.makedirs('submission_csv')
    submission_filename = 'submission_csv/' + name + '.csv'
    image_filenames = glob.glob('output/*.png')

    mask_to_submission.masks_to_submission(submission_filename, *image_filenames)


def main():
    x, y = data.get_training_data()
    x, y = data.augment_data(x,y)
    x = x[0:2]
    y = y[0:2]


    # crossentropy
    model = unet.get_model(None, None, 3, do_compile=False)
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2)])
    model_name = 'u_net_cross_entropy_test'
    train_sub(model, model_name, x, y, epochs=1)


    # focal
    from losses import focal
    loss = focal.focal_loss
    model_name = 'u_net_focal_loss'
    model = unet.get_model(None, None, 3, do_compile=False)
    model.compile(optimizer='adam', loss=loss,
                  metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2)])
    train_sub(model, model_name, x, y, epochs=1)


    # dice
    from losses import dice
    loss = dice.dice_loss
    model_name = 'u_net_dice'
    model = unet.get_model(None, None, 3, do_compile=False)
    model.compile(optimizer='adam', loss=loss,
                  metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2)])
    train_sub(model, model_name, x, y, epochs=1)



    # lovasz
    from losses import lovasz
    loss = lovasz.lovasz_hinge
    model_name = 'u_net_lovasz'
    model = unet.get_model(None, None, 3, do_compile=False)
    model.compile(optimizer='adam', loss=loss,
                  metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2)])
    train_sub(model, model_name, x, y, epochs=1)


if __name__ == "__main__":
    main()
