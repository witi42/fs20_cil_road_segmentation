import tensorflow as tf

def sdf_accuracy(x, y):
    """
    Takes an SDF GT and an SDF Prediction and computes the accuracy between them.

    Note: SDF <= 0 corresponds to the original GT = 1
    :param x:
    :param y:
    :return:
    """
    return tf.metrics.binary_accuracy(tf.cast(x <= 0, tf.float32), tf.cast(y <= 0, tf.float32))