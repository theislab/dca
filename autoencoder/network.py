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
from keras.layers import Input, Dense
from keras.models import Model
import tensorflow as tf
slim = tf.contrib.slim


def autoencoder(input_size, hidden_size=100):
    inp = Input(shape=(input_size,))
    encoded = Dense(hidden_size, activation='relu')(inp)
    decoded = Dense(input_size)(encoded)

    autoencoder = Model(input=inp, output=decoded)
    encoder = get_encoder(autoencoder)
    decoder = get_decoder(autoencoder)

    return autoencoder, encoder, decoder


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
