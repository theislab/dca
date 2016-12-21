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
    encoder = Model(input=inp, output=encoded)

    # create a placeholder for an encoded (32-dimensional) input
    encoded_input = Input(shape=(hidden_size,))
    decoder_layer = autoencoder.layers[-1]
    decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

    return autoencoder, encoder, decoder
