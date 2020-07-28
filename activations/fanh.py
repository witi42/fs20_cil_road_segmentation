import tensorflow as tf


def get_scaled_tanh(alpha):
    """
    Returns the activation function $x \mapsto \tanh(\alpha x)$.
    :param alpha: the stretching factor of the argument
    :return: said function
    """
    def tanh_alpha(x):
        return tf.keras.activations.tanh(alpha * x)

    return tanh_alpha
