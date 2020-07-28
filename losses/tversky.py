import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/
def get_loss(beta):
    """
    :param beta: weight for false positives
    """
    def loss(y_true, y_pred):
        numerator = tf.reduce_sum(y_true * y_pred, axis = -1)
        denominator = y_true * y_pred + beta * (1 - y_true) * y_pred + (1 - beta) * y_true * (1 - y_pred)
        return 1 - (numerator + 1) / (tf.reduce_sum(denominator, axis = -1) + 1)
    return loss
