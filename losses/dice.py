import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/
def dice_loss(y_true, y_pred):
    #y_pred = tf.cast(tf.math.argmax(y_pred, -1), tf.float32)
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=-1)
    denominator = tf.reduce_sum(y_true + y_pred, axis=-1)
    return 1 - (numerator + 1) / (denominator + 1)
