import tensorflow as tf
from  metrics.f1 import f1


def sdf_accuracy(x, y):
    """
    Takes an SDF GT and an SDF Prediction and computes the accuracy between them.

    Note: SDF <= 0 corresponds to the original GT = 1
    :param x:
    :param y:
    :return:
    """
    return tf.metrics.binary_accuracy(tf.cast(x <= 0, tf.float32), tf.cast(y <= 0, tf.float32))


def sdf_f1(x, y):
    return f1(tf.cast(x <= 0, tf.float32), tf.cast(y <= 0, tf.float32))


class SDFMeanIOU(tf.keras.metrics.MeanIoU):
    def __init__(self, num_classes=2):
        super().__init__(num_classes=num_classes)

    def update_state(self, x, y, sample_weight=None):
        return super().update_state(tf.cast(x <= 0, tf.float32), tf.cast(y <= 0, tf.float32), sample_weight=sample_weight)


