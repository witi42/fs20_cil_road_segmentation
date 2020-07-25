from tensorflow.keras.layers import Input, Dense, Conv2D, UpSampling2D, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.models import Model


def get_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, num_filters = 1024, do_compile = True, do_upsampling = False):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))                                               # whole image as input
    x = Conv2D(num_filters,                                                                             # apply 1 convolution with <num_filters> to each 16x16 patch
               kernel_size=(16, 16),
               strides=(16, 16),
               activation='relu',
               padding='valid',
               kernel_initializer='he_normal')(inputs)
    x = Dropout(0.1)(x)
    x = BatchNormalization()(x)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(x)                                                # classify on each convoluted patch
    if do_upsampling:
        outputs = UpSampling2D(size = (16, 16), interpolation = 'nearest')(outputs)                     # upsample to original image size
    
    model = Model(inputs=[inputs], outputs=[outputs])
    if compile:
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.summary()
        
    return model


