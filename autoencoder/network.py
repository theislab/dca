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

from .loss import poisson_loss, NB, ZINB

def mlp(input_size, output_size=None, hidden_size=(256,), l2_coef=0.,
                activation='relu', masking=False, loss_type='normal'):
    '''Construct an MLP (or autoencoder if output_size is not given)
    in Keras and return model (also encoder and decoderin case of AE).

    Args:
        input_size: Size of the input (i.e. number of features)
        output_size: Size of the output layer (AE if not specified)
        hidden_size: Tuple of sizes for each hidden layer.
        l2_coef: L2 regularization coefficient.
        activation: Activation function of hidden layers. relu is default.
        masking: Whether masking will be supported in the model.
        loss_type: Type of loss function. Available values are 'normal', 'poisson',
            'nb', 'zinb' and 'zinb-meanmix'. 'zinb' refers to zero-inflated
            negative binomial with constant mixture params. 'zinb-meanmix'
            formulates mixture parameters as a function of NB mean.

    Returns:
        A dict of Keras model, encoder, decoder, loss function and extra
            models. Extra models keep mixture coefficients (i.e. pi) for ZINB.
    '''

    assert loss_type in ['normal', 'poisson', 'nb', 'zinb', 'zinb-meanmix'], \
                         'loss type not supported'

    ae = True if output_size is None else False
    extra_models  = {}
    if output_size is None:
        output_size = input_size

    inp = Input(shape=(input_size,))
    if masking:
        nan = Lambda(lambda x: tf.where(tf.is_nan(x), tf.zeros_like(x), x))(inp)
        last_hidden = nan
    else:
        last_hidden = inp

    for i, hid_size in enumerate(hidden_size):
        last_hidden = Dense(hid_size, activation=activation,
                      kernel_regularizer=l2(l2_coef), name='hidden_%s'%i)(last_hidden)
        if i == int(np.floor(len(hidden_size) / 2.0)):
            middle_layer = last_hidden

    if loss_type == 'normal':
        loss = mean_squared_error
        output = Dense(output_size, activation=None,
                       kernel_regularizer=l2(l2_coef), name='output')(last_hidden)
    elif loss_type == 'poisson':
        output = Dense(output_size, activation=K.exp,
                       kernel_regularizer=l2(l2_coef), name='output')(last_hidden)
        loss = poisson_loss
    elif loss_type == 'nb':
        nb = NB(theta_init=tf.zeros([1, output_size]), masking=masking)
        output = Dense(output_size, activation=K.exp,
                       kernel_regularizer=l2(l2_coef), name='output')(last_hidden)
        loss = nb.loss
    elif loss_type == 'zinb':
        pi_layer = Dense(output_size, activation='sigmoid',
                       kernel_regularizer=l2(l2_coef), name='pi')
        pi = pi_layer(last_hidden)
        output = Dense(output_size, activation=K.exp,
                       kernel_regularizer=l2(l2_coef), name='output')(last_hidden)
        zinb = ZINB(pi, theta_init=tf.zeros([1, output_size]), masking=masking)
        loss = zinb.loss
        extra_models['pi'] = Model(inputs=inp, outputs=pi)
    elif loss_type == 'zinb-meanmix':
        #TODO: Add another output module and make pi a function of mean
        raise NotImplemented

    ret  = {'model': Model(inputs=inp, outputs=output)}

    if ae:
        ret['encoder'] = Model(inputs=inp, outputs=middle_layer)
        ret['decoder'] = None #Model(inputs=middle_layer, outputs=output)

    #Ugly hack to inject NB dispersion parameters
    if loss_type == 'nb':
        # add theta as a trainable variable to Keras model
        # otherwise, keras optimizers will not update it
        ret['model'].layers[-1].trainable_weights.append(nb.theta_variable)
    elif loss_type == 'zinb':
        ret['model'].layers[-1].trainable_weights.extend([zinb.theta_variable,
                                                         *pi_layer.trainable_weights])
    ret['extra_models'] = extra_models
    ret['loss'] = loss

    return ret


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
