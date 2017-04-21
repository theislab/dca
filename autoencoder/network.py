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
from keras.layers import Input, Dense, Lambda, Dropout, Activation
from keras.models import Model
from keras.regularizers import l2
from keras.objectives import mean_squared_error
from keras import backend as K
import tensorflow as tf

from .loss import poisson_loss, NB, ZINB

def mlp(input_size, output_size=None, hidden_size=(256,), l2_coef=0.,
        hidden_dropout=0.1, activation='relu', init='glorot_uniform',
        masking=False, loss_type='normal'):
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

    if isinstance(hidden_dropout, list):
        assert len(hidden_dropout) == len(hidden_size)
    else:
        hidden_dropout = [hidden_dropout]*len(hidden_size)

    inp = Input(shape=(input_size,))
    if masking:
        nan = Lambda(lambda x: tf.where(tf.is_nan(x), tf.zeros_like(x), x))(inp)
        last_hidden = nan
    else:
        last_hidden = inp

    for i, (hid_size, hid_drop) in enumerate(zip(hidden_size, hidden_dropout)):
        if i == int(np.floor(len(hidden_size) / 2.0)):
            layer_name = 'center'
        else:
            layer_name = 'hidden_%s' % i

        last_hidden = Dense(hid_size, activation=None, kernel_initializer=init,
                      kernel_regularizer=l2(l2_coef), name=layer_name)(last_hidden)

        # Use separate act. layers to give user the option to get pre-activations
        # of layers when requested
        last_hidden = Activation(activation, name='%s_act'%layer_name)(last_hidden)
        last_hidden = Dropout(hid_drop, name='%s_drop'%layer_name)(last_hidden)

    if loss_type == 'normal':
        loss = mean_squared_error
        output = Dense(output_size, activation=None, kernel_initializer=init,
                       kernel_regularizer=l2(l2_coef), name='output')(last_hidden)
    elif loss_type == 'poisson':
        output = Dense(output_size, activation=K.exp, kernel_initializer=init,
                       kernel_regularizer=l2(l2_coef), name='output')(last_hidden)
        loss = poisson_loss
    elif loss_type == 'nb':
        nb = NB(theta_init=tf.zeros([1, output_size]), masking=masking)
        output = Dense(output_size, activation=K.exp, kernel_initializer=init,
                       kernel_regularizer=l2(l2_coef), name='output')(last_hidden)
        loss = nb.loss
        extra_models['dispersion'] = lambda :K.function([], [1/nb.theta])([])[0].squeeze()
    elif loss_type == 'zinb':
        pi_layer = Dense(output_size, activation='sigmoid', kernel_initializer=init,
                       kernel_regularizer=l2(l2_coef), name='pi')
        pi = pi_layer(last_hidden)
        output = Dense(output_size, activation=K.exp, kernel_initializer=init,
                       kernel_regularizer=l2(l2_coef), name='output')(last_hidden)
        zinb = ZINB(pi, theta_init=tf.zeros([1, output_size]), masking=masking)
        loss = zinb.loss
        extra_models['pi'] = Model(inputs=inp, outputs=pi)
        extra_models['dispersion'] = lambda :K.function([], [1/zinb.theta])([])[0].squeeze()
    elif loss_type == 'zinb-meanmix':
        #TODO: Add another output module and make pi a function of mean
        raise NotImplemented

    ret  = {'model': Model(inputs=inp, outputs=output)}

    if ae:
        ret['encoder_linear'] = get_encoder(ret['model'], activation = False)
        ret['encoder'] = get_encoder(ret['model'], activation = True)
        ret['decoder'] = None #get_decoder(ret['model'])

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

def get_decoder(model):
    i = 0
    for l in model.layers:
        if l.name == 'center_drop': break
        i += 1

    return Model(inputs=model.get_layer(index=i+1).input,
                 outputs=model.output)

def get_encoder(model, activation=True):
    if activation:
        ret =  Model(inputs=model.input,
                     outputs=model.get_layer('center_act').output)
    else:
        ret =  Model(inputs=model.input,
                     outputs=model.get_layer('center').output)
    return ret
