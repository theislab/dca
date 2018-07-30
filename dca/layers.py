from keras.engine.topology import Layer
from keras.layers import Lambda, Dense
from keras.engine.base_layer import InputSpec
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
        self.theta_exp = tf.clip_by_value(K.exp(self.theta), 1e-3, 1e4)
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
        assert isinstance(x, list), 'SliceLayer input is not a list'
        return x[self.index]

    def compute_output_shape(self, input_shape):
        return input_shape[self.index]


class ElementwiseDense(Dense):
    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        assert (input_dim == self.units) or (self.units == 1), \
               "Input and output dims are not compatible"

        # shape=(input_units, ) makes this elementwise bcs of broadcasting
        self.kernel = self.add_weight(shape=(self.units,),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        # use * instead of tf.matmul, we need broadcasting here
        output = inputs * self.kernel
        if self.use_bias:
            output = output + self.bias
        if self.activation is not None:
            output = self.activation(output)
        return output


nan2zeroLayer = Lambda(lambda x: tf.where(tf.is_nan(x), tf.zeros_like(x), x))
ColwiseMultLayer = Lambda(lambda l: l[0]*tf.reshape(l[1], (-1,1)))
