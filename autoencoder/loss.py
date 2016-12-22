import tensorflow as tf
from keras.objectives import mean_squared_error


def impute_loss(y_true, y_pred, threshold=0.01):
    mask = y_true > threshold
    mask = tf.cast(mask, tf.float32)
    return mean_squared_error(y_true, y_pred*mask)
