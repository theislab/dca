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

def autoencoder(input_size, hidden_size=10, l2_coef=0.,
                aetype=None):

    assert aetype in ['normal', 'poisson', 'nb', 'zinb'], \
                     'AE type not supported'

    if aetype == 'normal':
        output_activation = None
        loss = mean_squared_error
    elif aetype == 'poisson':
        output_activation = K.exp
        loss = poisson_loss
    elif aetype == 'nb':
        output_activation = K.exp
        nb = NB(theta_init=tf.zeros([1, input_size]))
        loss = nb.loss
    elif aetype == 'zinb':
        raise NotImplementedError

    inp = Input(shape=(input_size,))
    nan = Lambda(lambda x: tf.where(tf.is_nan(x), tf.zeros_like(x), x))(inp)

    encoded = Dense(hidden_size, activation='relu',
                    W_regularizer=l2(l2_coef))(nan)
    decoded = Dense(input_size, activation=output_activation,
                    W_regularizer=l2(l2_coef))(encoded)

    autoencoder = Model(input=inp, output=decoded)
    encoder = get_encoder(autoencoder)
    decoder = get_decoder(autoencoder)

    #Ugly hack to inject NB dispersion parameters
    if aetype == 'nb':
        # add theta as a trainable variable to Keras model
        # otherwise, keras optimizers will not update it
        autoencoder.layers[-1].trainable_weights.append(nb.theta_variable)

    return autoencoder, encoder, decoder, loss


def get_decoder(model):
    num_dec_layers = int((len(model.layers)-1) / 2)
    dec_layer_num = num_dec_layers + 1
    dec_layer = model.layers[dec_layer_num]
    hidden_size = dec_layer.input_dim

    in_layer = Input(shape=((hidden_size, )))
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
