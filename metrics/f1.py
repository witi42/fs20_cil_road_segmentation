import tensorflow as tf

K = tf.keras.backend

def f1(y_true, y_pred):
    y_true_ = tf.cast(y_true, tf.float32)
    y_pred_ = tf.cast(y_pred, tf.float32)
    y_true_ = K.flatten(y_true_)
    y_pred_ = K.flatten(y_pred_)
    return 2 * (K.sum(y_true_ * y_pred_)+ K.epsilon()) / (K.sum(y_true_) + K.sum(y_pred_) + K.epsilon())

def f1_binary(y_true, y_pred):
    y_true_ = tf.cast(y_true, tf.float32)
    y_pred_ = tf.cast(y_pred, tf.float32)
    y_true_ = K.flatten(y_true_)
    y_pred_ = tf.round(K.flatten(y_pred_))
    return 2 * (K.sum(y_true_ * y_pred_)+ K.epsilon()) / (K.sum(y_true_) + K.sum(y_pred_) + K.epsilon())