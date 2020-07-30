import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix
import numpy as np


def get_model():
    OUTPUT_CHANNELS = 2

    base_model = tf.keras.applications.MobileNetV2(input_shape=[224, 224, 3], include_top=False)

    # Use the activations of these layers
    layer_names = [
        'block_1_expand_relu',  # 64x64
        'block_3_expand_relu',  # 32x32
        'block_6_expand_relu',  # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',  # 4x4
    ]
    layers = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

    down_stack.trainable = False

    up_stack = [
        pix2pix.upsample(512, 3),  # 4x4 -> 8x8
        pix2pix.upsample(256, 3),  # 8x8 -> 16x16
        pix2pix.upsample(128, 3),  # 16x16 -> 32x32
        pix2pix.upsample(64, 3),  # 32x32 -> 64x64
    ]

    def unet_model(output_channels):
        inputs = tf.keras.layers.Input(shape=[224, 224, 3])
        x = inputs

        # Downsampling through the model
        skips = down_stack(x)
        x = skips[-1]
        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            concat = tf.keras.layers.Concatenate()
            x = concat([x, skip])

        # This is the last layer of the model
        last = tf.keras.layers.Conv2DTranspose(
            output_channels, 3, strides=2,
            padding='same')  # 64x64 -> 128x128

        x = last(x)

        return tf.keras.Model(inputs=inputs, outputs=x)

    model = unet_model(OUTPUT_CHANNELS)

    return model




class tf_unet():
    def __init__(self, checkpoint = None, model = None):
        if checkpoint != None:
            self.model = get_model()
            self.model.load_weights(checkpoint)
        else:
            self.model = model

    def load_weights(path):
        self.model.load_weights(path)
  

    def predict(self, x):
        out_all = []

        shape = x.shape
        for i in range(shape[0]):
            image = x[i]

            small_s = 224
            large_s = 400

            aa = image[0:small_s, 0:small_s]
            ab = image [0:small_s, large_s-small_s: large_s]
            ba = image [large_s-small_s: large_s, 0:small_s]
            bb = image [large_s-small_s: large_s, large_s-small_s: large_s]

            x_splits = np.array([aa, ab, ba, bb])

            pred = self.model.predict(x_splits)
            pred = pred_image = np.argmax(pred, -1)

            aa_ = pred[0, 0:large_s // 2, 0:large_s // 2]
            ab_ = pred[1, 0:large_s // 2, small_s - large_s // 2:small_s]
            ba_ = pred[2, small_s - large_s // 2:small_s, 0:large_s // 2]
            bb_ = pred[3, small_s - large_s // 2:small_s, small_s - large_s // 2:small_s]

            a = np.concatenate((aa_, ab_), axis=1)
            b = np.concatenate((ba_, bb_), axis=1)
            out = np.concatenate((a, b), axis=0)

            out_all.append(out)

        return np.array(out_all)