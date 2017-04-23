from keras.engine.topology import Layer
from keras.layers import Lambda
from keras import backend as K
import tensorflow as tf


class ConstantDispersionLayer(Layer):
    '''
        An identity layer which allows us to inject extra parameters
        such as dispersion to Keras models
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.theta = self.add_weight(shape=(1, input_shape[1]),
                                     initializer='zeros',
                                     trainable=True,
                                     name='theta')
        self.theta_exp = 1.0/(K.exp(self.theta)+1e-10)
        super().build(input_shape)

    def call(self, x):
        return tf.identity(x)

    def compute_output_shape(self, input_shape):
        return input_shape


class SliceLayer(Layer):
    def __init__(self, index, **kwargs):
        self.index = index
        super().__init__(**kwargs)

    def build(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError('Input should be a list')

        super().build(input_shape)

    def call(self, x):
        return x[self.index]

    def compute_output_shape(self, input_shape):
        return input_shape[self.index]


nan2zeroLayer = Lambda(lambda x: tf.where(tf.is_nan(x), tf.zeros_like(x), x))