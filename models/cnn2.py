from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from keras import backend as K
from keras.regularizers import l2
from keras.layers import LeakyReLU
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

from keras.layers import Input


def get_model_cnn(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, do_compile=False):
    """
    :param IMG_HEIGHT:
    :param IMG_WIDTH:
    :param IMG_CHANNELS:
    :param do_compile: whether or not to compile the model yet
    :return:
    """
    run = 2 #0-default, 1-mine-mod, 2-mine
    #inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    
    #s = Lambda(lambda x: x / 255)(inputs)

    patch_size = 16
    window_size = 72
    padding = (window_size - patch_size) // 2
    input_shape = (3, window_size, window_size)
    nb_classes = 2
    reg = 1e-6 # L2 regularization factor (used on weights, but not biases)

    # Size of pooling area for max pooling
    pool_size = (2, 2)

    if run==0:

        '''
        inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

        c = Conv2D(16, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.1), kernel_initializer='he_normal', padding='same')(inputs)
        c = BatchNormalization()(c)
        c = Dropout(0.25)(c)
        c = Conv2D(16, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.1), kernel_initializer='he_normal', padding='same')(c)
        c = BatchNormalization()(c)
        p = MaxPooling2D(pool_size)(c)

        c0 = Conv2D(32, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.1), kernel_initializer='he_normal', padding='same')(p)
        c0 = BatchNormalization()(c0)
        c0 = Dropout(0.25)(c0)
        c0 = Conv2D(32, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.1), kernel_initializer='he_normal', padding='same')(c0)
        c0 = BatchNormalization()(c0)
        p0 = MaxPooling2D(pool_size)(c0)

        c1 = Conv2D(64, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.1), kernel_initializer='he_normal', padding='same')(p0)
        c1 = BatchNormalization()(c1)
        c1 = Dropout(0.25)(c1)
        c1 = Conv2D(64, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.1), kernel_initializer='he_normal', padding='same')(c1)
        c1 = BatchNormalization()(c1)
        p1 = MaxPooling2D(pool_size)(c1)

        c2 = Conv2D(128, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.1), kernel_initializer='he_normal', padding='same')(p1)
        c2 = BatchNormalization()(c2)
        c2 = Dropout(0.25)(c2)
        c2 = Conv2D(128, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.1), kernel_initializer='he_normal', padding='same')(c2)
        c2 = BatchNormalization()(c2)
        p2 = MaxPooling2D(pool_size)(c2)

        c3 = Conv2D(256, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.1), kernel_initializer='he_normal', padding='same')(p2)
        c3 = BatchNormalization()(c3)
        c3 = Dropout(0.25)(c3)
        c3 = Conv2D(256, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.1), kernel_initializer='he_normal', padding='same')(c3)
        c3 = BatchNormalization()(c3)
        p3 = MaxPooling2D(pool_size)(c3)

        c4 = Conv2D(512, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.1), kernel_initializer='he_normal', padding='same')(p3)
        c4 = BatchNormalization()(c4)
        c4 = Dropout(0.25)(c4)
        c4 = Conv2D(512, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.1), kernel_initializer='he_normal', padding='same')(c4)
        c4 = BatchNormalization()(c4)
        p4 = MaxPooling2D(pool_size)(c4)

        c5 = Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
        c5 = BatchNormalization()(c5)
        c5 = Dropout(0.25)(c5)
        c5 = Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
        c5 = BatchNormalization()(c5)
        '''

        '''
        u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = concatenate([u6, c4])
        c6 = Conv2D(512, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.1), kernel_initializer='he_normal', padding='same')(u6)
        c6 = BatchNormalization()(c6)
        c6 = Dropout(0.25)(c6)
        c6 = Conv2D(512, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.1), kernel_initializer='he_normal', padding='same')(c6)
        c6 = BatchNormalization()(c6)

        c7 = Conv2D(256, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.1), kernel_initializer='he_normal', padding='same')(c6)
        c7 = BatchNormalization()(c7)
        c7 = Dropout(0.25)(c7)
        c7 = Conv2D(256, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.1), kernel_initializer='he_normal', padding='same')(c7)
        c7 = BatchNormalization()(c7)

        c8 = Conv2D(128, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.1), kernel_initializer='he_normal', padding='same')(c7)
        c8 = BatchNormalization()(c8)
        c8 = Dropout(0.25)(c8)
        c8 = Conv2D(128, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.1), kernel_initializer='he_normal', padding='same')(c8)
        c8 = BatchNormalization()(c8)

        c9 = Conv2D(64, (5, 5), activation=tf.keras.layers.LeakyReLU(alpha=0.1), kernel_initializer='he_normal', padding='same')(c8)
        c9 = BatchNormalization()(c9)
        c9 = Dropout(0.25)(c9)
        c9 = Conv2D(64, (5, 5), activation=tf.keras.layers.LeakyReLU(alpha=0.1), kernel_initializer='he_normal', padding='same')(c9)
        c9 = BatchNormalization()(c9)

        c10 = Conv2D(32, (5, 5), activation=tf.keras.layers.LeakyReLU(alpha=0.1), kernel_initializer='he_normal', padding='same')(c9)
        c10 = BatchNormalization()(c10)
        c10 = Dropout(0.25)(c10)
        c10 = Conv2D(64, (5, 5), activation=tf.keras.layers.LeakyReLU(alpha=0.1), kernel_initializer='he_normal', padding='same')(c10)
        c10 = BatchNormalization()(c10)

        c11 = Conv2D(32, (5, 5), activation=tf.keras.layers.LeakyReLU(alpha=0.1), kernel_initializer='he_normal', padding='same')(c10)
        c11 = BatchNormalization()(c11)
        c11 = Dropout(0.25)(c11)
        c11 = Conv2D(64, (5, 5), activation=tf.keras.layers.LeakyReLU(alpha=0.1), kernel_initializer='he_normal', padding='same')(c11)
        c11 = BatchNormalization()(c11)

        -------------

        u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = concatenate([u6, c4])
        c6 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
        c6 = BatchNormalization()(c6)
        c6 = Dropout(0.2)(c6)
        c6 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
        c6 = BatchNormalization()(c6)

        u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = concatenate([u7, c3])
        c7 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
        c7 = BatchNormalization()(c7)
        c7 = Dropout(0.2)(c7)
        c7 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
        c7 = BatchNormalization()(c7)

        u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = concatenate([u8, c2])
        c8 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
        c8 = BatchNormalization()(c8)
        c8 = Dropout(0.1)(c8)
        c8 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
        c8 = BatchNormalization()(c8)

        u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = concatenate([u9, c1], axis=3)
        c9 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
        c9 = BatchNormalization()(c9)
        c9 = Dropout(0.1)(c9)
        c9 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
        c9 = BatchNormalization()(c9)
        '''

        #outputs = Conv2D(1, (1, 1), activation='softmax')(c5)

        #model = Model(inputs=[inputs], outputs=[outputs])

    elif run==1:

        inputs = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

        model = Sequential()

        model.add(Conv2D(64,(3,3),activation='relu',input_shape=inputs, padding='same'))
        model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
        model.add(MaxPool2D((2,2),strides=(2,2)))

        model.add(Conv2D(128,(3,3),activation='relu',padding='same'))
        model.add(Conv2D(128,(3,3),activation='relu',padding='same'))
        model.add(MaxPool2D((2,2),strides=(2,2)))

        model.add(Conv2D(256,(3,3),activation='relu',padding='same'))
        model.add(Conv2D(256,(3,3),activation='relu',padding='same'))
        model.add(Conv2D(256,(3,3),activation='relu',padding='same'))
        model.add(MaxPool2D((2,2),strides=(2,2)))

        model.add(Conv2D(512,(3,3),activation='relu',padding='same'))
        model.add(Conv2D(512,(3,3),activation='relu',padding='same'))
        model.add(Conv2D(512,(3,3),activation='relu',padding='same'))
        model.add(MaxPool2D((2,2),strides=(2,2)))

        model.add(Flatten())
        model.add(Dense(nb_classes,activation='relu'))
        model.add(Dense(nb_classes,activation='relu'))
        model.add(Dense(nb_classes,activation='softmax'))

        

    else:

        inputs = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

        model = Sequential()

        model.add(Convolution2D(64, 5, 5, # 64 5x5 filters
                                padding='same',
                                input_shape=inputs
                                ))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=pool_size, padding='same'))
        model.add(Dropout(0.25))


        model.add(Convolution2D(128, 3, 3, # 128 3x3 filters
                                padding='same'
                                ))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=pool_size, padding='same'))
        model.add(Dropout(0.25))


        model.add(Convolution2D(256, 3, 3, # 256 3x3 filters
                                padding='same'
                                ))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=pool_size, padding='same'))
        model.add(Dropout(0.25))


        model.add(Convolution2D(256, 3, 3, # 256 3x3 filters
                                padding='same'
                                ))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=pool_size, padding='same'))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes, activation='softmax'))

        '''
        model.add(Flatten())
        model.add(Dense(128, activation=l2(reg))) # Fully connected layer (128 neurons)
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dropout(0.5))

        model.add(Dense(nb_classes, activation=l2(reg)))

        #model.add(Dense(nb_classes, activation="softmax"))

        #model.add(Convolution2D(1, 1, 1,
        #                        padding='same'
        #                        ))
        #model.add(Dense(nb_classes, activation='softmax'))
        '''
        

    model.summary()

    opt = Adam(lr=0.001) # Adam optimizer with default initial learning rate
    if do_compile:
        model.compile(loss='binary_crossentropy',
                        optimizer=opt,
                        metrics=['accuracy'])

    #np.random.seed(3) # Ensure determinism

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