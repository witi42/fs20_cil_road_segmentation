from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout, Lambda
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten



def get_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, do_compile=False):

    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    conv1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (inputs)
    conv1 = BatchNormalization() (conv1)
    conv1 = Dropout(0.1) (conv1)
    conv1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (conv1)
    conv1 = BatchNormalization() (conv1)
    pooling1 = MaxPooling2D((2, 2)) (conv1)

    conv2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (pooling1)
    conv2 = BatchNormalization() (conv2)
    conv2 = Dropout(0.1) (conv2)
    conv2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (conv2)
    conv2 = BatchNormalization() (conv2)
    pooling2 = MaxPooling2D((2, 2)) (conv2)

    conv3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (pooling2)
    conv3 = BatchNormalization() (conv3)
    conv3 = Dropout(0.2) (conv3)
    conv3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (conv3)
    conv3 = BatchNormalization() (conv3)
    pooling3 = MaxPooling2D((2, 2)) (conv3)

    conv4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (pooling3)
    conv4 = BatchNormalization() (conv4)
    conv4 = Dropout(0.2) (conv4)
    conv4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (conv4)
    conv4 = BatchNormalization() (conv4)
    pooling4 = MaxPooling2D(pool_size=(2, 2)) (conv4)

    conv5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (pooling4)
    conv5 = BatchNormalization() (conv5)
    conv5 = Dropout(0.3) (conv5)
    conv5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (conv5)
    conv5 = BatchNormalization() (conv5)
    pooling5 = MaxPooling2D(pool_size=(2, 2)) (conv5)

    conv6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pooling5)
    conv6 = BatchNormalization()(conv6)
    conv6 = Dropout(0.3)(conv6)
    conv6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    pooling6 = MaxPooling2D(pool_size=(2, 2)) (conv6)

    conv7 = Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pooling6)
    conv7 = BatchNormalization()(conv7)
    conv7 = Dropout(0.3)(conv7)
    conv7 = Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)

    upsample8 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same') (conv7)
    upsample8 = concatenate([upsample8, conv6])
    conv8 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (upsample8)
    conv8 = BatchNormalization() (conv8)
    conv8 = Dropout(0.2) (conv8)
    conv8 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (conv8)
    conv8 = BatchNormalization() (conv8)

    upsample9 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same') (conv8)
    upsample9 = concatenate([upsample9, conv5])
    conv9 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (upsample9)
    conv9 = BatchNormalization() (conv9)
    conv9 = Dropout(0.2) (conv9)
    conv9 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (conv9)
    conv9 = BatchNormalization() (conv9)
    
    upsample10 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (conv5)
    upsample10 = concatenate([upsample10, conv4])
    conv10 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (upsample10)
    conv10 = BatchNormalization() (conv10)
    conv10 = Dropout(0.2) (conv10)
    conv10 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (conv10)
    conv10 = BatchNormalization() (conv10)

    upsample11 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (conv10)
    upsample11 = concatenate([upsample11, conv3])
    conv11 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (upsample11)
    conv11 = BatchNormalization() (conv11)
    conv11 = Dropout(0.2) (conv11)
    conv11 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (conv11)
    conv11 = BatchNormalization() (conv11)

    upsample12 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (conv11)
    upsample12 = concatenate([upsample12, conv2])
    conv12 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (upsample12)
    conv12 = BatchNormalization() (conv12)
    conv12 = Dropout(0.1) (conv12)
    conv12 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (conv12)
    conv12 = BatchNormalization() (conv12)

    upsample13 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (conv12)
    upsample13 = concatenate([upsample13, conv1], axis=3)
    conv13 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (upsample13)
    conv13 = BatchNormalization() (conv13)
    conv13 = Dropout(0.1) (conv13)
    conv13 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same') (conv13)
    conv13 = BatchNormalization() (conv13)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (conv13)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model


def fit(X_train, Y_train, model, epochs=100, validation_split=0.1, validation_data=None, class_weight=None,
        checkpoint_datetime=False,checkpoint_suffix="",batch_size=8):
    """

    :param X_train: The training data
    :param Y_train: The training labels
    :param model: The tf keras model to train
    :param epochs: the number of epochs to train
    :param validation_split: percentage of data to use as validation
    :param validation_data: tuple of (X_val, y_val) validation data to use
    :param class_weight: weights for the classes in the loss function
    :param checkpoint_datetime: whether or not to append the current date and time to the checkpoints
    :param checkpoint_suffix: a optional string to append to the checkpoints
    :return: the results object
    """
    earlystopper = EarlyStopping(patience=20, verbose=1)
    suffix = checkpoint_suffix
    if checkpoint_datetime:
        suffix += str(datetime.datetime.now())
    checkpointer = ModelCheckpoint('checkpoints/unet{}.h5'.format(suffix), verbose=1, save_best_only=True)
    results = model.fit(X_train, Y_train, validation_split=validation_split, batch_size=batch_size, epochs=epochs,
                        validation_data=validation_data,
                        callbacks=[earlystopper, checkpointer], class_weight=class_weight)
    return results


def load(filename='checkpoints/unet.h5'):
    return load_model(filename)

    # # Fit model
    # earlystopper = EarlyStopping(patience=15, verbose=1)
    # checkpointer = ModelCheckpoint('model_unet_checkpoint.h5', verbose=1, save_best_only=True)
    # results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=100,
    #                     callbacks=[earlystopper, checkpointer])
    #
    # # Predict on train, val and test
    # model = load_model('model_unet_checkpoint.h5')
    # preds_train = model.predict(X_train[:int(X_train.shape[0] * 0.9)], verbose=1)
    # preds_val = model.predict(X_train[int(X_train.shape[0] * 0.9):], verbose=1)
    # preds_test = model.predict(X_test, verbose=1)
    #
    # # Threshold predictions
    # preds_train_t = (preds_train > 0.5).astype(np.uint8)
    # preds_val_t = (preds_val > 0.5).astype(np.uint8)
    # preds_test_t = (preds_test > 0.5).astype(np.uint8)
    #
    # # Create list of upsampled test masks
    # preds_test_upsampled = []
    # for i in range(len(preds_test_t)):
    #     preds_test_upsampled.append(resize(np.squeeze(preds_test_t[i]),
    #                                        (sizes_test[i][0], sizes_test[i][1]),
    #                                        mode='constant', preserve_range=True))
