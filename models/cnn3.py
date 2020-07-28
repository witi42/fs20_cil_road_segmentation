import keras

from keras.layers import Input


def get_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, do_compile=False):

    inputs = (400, 400, IMG_CHANNELS)

    # Define the model
    model = keras.models.Sequential()

    # Define the first wave of layers
    
    model.add(keras.layers.Convolution2D(filters=64,
                                                kernel_size=(3, 3),
                                                padding="same",
                                                input_shape=inputs))
    model.add(keras.layers.LeakyReLU(alpha=0.1))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2),
                                                padding="same"))
    model.add(keras.layers.Dropout(rate=0.25))

    # Define the second wave of layers
    model.add(keras.layers.Convolution2D(filters=128,
                                                kernel_size=(3, 3),
                                                padding="same"))
    model.add(keras.layers.LeakyReLU(alpha=0.1))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2),
                                                padding="same"))
    model.add(keras.layers.Dropout(rate=0.25))

    # Define the third wave of layers
    model.add(keras.layers.Convolution2D(filters=256,
                                                kernel_size=(3, 3),
                                                padding="same"))
    model.add(keras.layers.LeakyReLU(alpha=0.1))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2),
                                                padding="same"))
    model.add(keras.layers.Dropout(rate=0.25))

    # Define the fourth wave of layers
    model.add(keras.layers.Convolution2D(filters=256,
                                                kernel_size=(3, 3),
                                                padding="same"))
    model.add(keras.layers.LeakyReLU(alpha=0.1))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2),
                                                padding="same"))
    model.add(keras.layers.Dropout(rate=0.25))

    # Define the fifth wave of layers
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units=128,
                                        kernel_regularizer=keras.regularizers.l2(1e-6)))
    model.add(keras.layers.LeakyReLU(alpha=0.1))
    model.add(keras.layers.Dropout(rate=0.5))

    model.add(keras.layers.Dense(units=2,
                                        kernel_regularizer=keras.regularizers.l2(1e-6),
                                        activation="softmax"))

    '''
    print("Compiling model ...")

    optimiser = keras.optimizers.Adam()
    model.compile(loss=keras.losses.categorical_crossentropy,
                        optimizer=optimiser,
                        metrics=["accuracy"])
                    '''
    print(model.summary())
    print("Done")
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