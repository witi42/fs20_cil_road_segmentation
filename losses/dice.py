import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/
def dice_loss(y_true, y_pred):
    y_true_ = tf.cast(y_true, tf.float32)
    y_pred_ = tf.cast(y_pred, tf.float32)
    numerator = 2 * tf.reduce_sum(y_true_ * y_pred_, axis=-1)
    denominator = tf.reduce_sum(y_true_ + y_pred_, axis=-1)

    return 1 - (numerator + 1) / (denominator + 1)
