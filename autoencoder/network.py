# Copyright 2016 Goekcen Eraslan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.regularizers import l2
from keras.objectives import mean_squared_error
from keras import backend as K
import tensorflow as tf
slim = tf.contrib.slim

from .loss import poisson_loss, NB

def autoencoder(input_size, hidden_size=(256, 64, 256), l2_coef=0.,
                activation='relu', masking=False, aetype=None):
    '''Construct an autoencoder in Keras and return model, encoder and decoder
    parts.

    Args:
        input_size: Size of the input (i.e. number of features)
        hidden_size: Tuple of sizes for each hidden layer.
        l2_coef: L2 regularization coefficient.
        activation: Activation function of hidden layers. relu is default.
        masking: Whether masking will be supported in the model.
        aetype: Type of autoencoder. Available values are 'normal', 'poisson',
            'nb', 'zinb' and 'zinb-meanmix'. 'zinb' refers to zero-inflated
            negative binomial with constant mixture params. 'zinb-meanmix'
            formulates mixture parameters as a function of NB mean.

    Returns:
        A tuple of Keras model, encoder, decoder, loss function and extra
            models. Extra models keep mixture coefficients (i.e. pi) for ZINB.
    '''

    assert aetype in ['normal', 'poisson', 'nb', 'zinb', 'zinb-meanmix'], \
                     'AE type not supported'

    extra_models  = {}

    inp = Input(shape=(input_size,))
    if masking:
        nan = Lambda(lambda x: tf.where(tf.is_nan(x), tf.zeros_like(x), x))(inp)
        last_hidden = nan
    else:
        last_hidden = inp

    for i, hid_size in enumerate(hidden_size):
        last_hidden = Dense(hid_size, activation=activation,
                      kernel_regularizer=l2(l2_coef))(last_hidden)
        if i == int(np.floor(len(hidden_size) / 2.0)):
            middle_layer = last_hidden

    if aetype == 'normal':
        loss = mean_squared_error
        output = Dense(input_size, activation=None,
                       kernel_regularizer=l2(l2_coef))(last_hidden)
    elif aetype == 'poisson':
        output = Dense(input_size, activation=K.exp,
                       kernel_regularizer=l2(l2_coef))(last_hidden)
        loss = poisson_loss
    elif aetype == 'nb':
        nb = NB(theta_init=tf.zeros([1, input_size]), masking=masking)
        output = Dense(input_size, activation=tf.nn.softplus,
                       kernel_regularizer=l2(l2_coef))(last_hidden)
        loss = nb.loss
    elif aetype == 'zinb':
        pi = Dense(input_size, activation='sigmoid',
                       kernel_regularizer=l2(l2_coef))(last_hidden)
        output = Dense(input_size, activation=tf.nn.softplus,
                       kernel_regularizer=l2(l2_coef))(last_hidden)
        zinb = ZINB(pi, theta_init=tf.zeros([1, num_out]), masking=masking)
        loss = zinb.loss
        extra_models['pi'] = Model(inputs=inp, outputs=pi)
    elif aetype == 'zinb-meanmix':
        #TODO: Add another output module and make pi a function of mean
        raise NotImplemented

    autoencoder = Model(inputs=inp, outputs=output)
    encoder = Model(inputs=inp, outputs=middle_layer)
    decoder = Model(inputs=middle_layer, outputs=output)

    #Ugly hack to inject NB dispersion parameters
    if aetype == 'nb':
        # add theta as a trainable variable to Keras model
        # otherwise, keras optimizers will not update it
        autoencoder.layers[-1].trainable_weights.append(nb.theta_variable)
    elif aetype == 'zinb':
        autoencoder.layers[-1].trainable_weights.extend([zinb.theta_variable,
                                                         *pi.trainable_weights])

    return autoencoder, encoder, decoder, loss, extra_models


#TODO: take Lambda layer and multiple output layers into account
def get_decoder(model):
    num_dec_layers = int((len(model.layers)-1) / 2)
    dec_layer_num = num_dec_layers + 1
    dec_layer = model.layers[dec_layer_num]
    hidden_size = dec_layer.input_shape

    in_layer = Input(shape=hidden_size)
    out_layer = in_layer
    for layer in model.layers[dec_layer_num:]:
        out_layer = layer(out_layer)

    return Model(input=in_layer, output=out_layer)


def get_encoder(model):
    input_size = model.input_shape[1]
    num_enc_layers = int((len(model.layers)-1) / 2)

    in_layer = Input(shape=((input_size, )))
    out_layer = in_layer
    for layer in model.layers[1: (num_enc_layers+1)]:
        out_layer = layer(out_layer)

    return Model(input=in_layer, output=out_layer)
