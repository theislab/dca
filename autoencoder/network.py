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

import os, pickle

import numpy as np
from keras.layers import Input, Dense, Dropout, Activation, BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from keras.objectives import mean_squared_error
from keras.initializers import Constant
from keras import backend as K
import tensorflow as tf

from .loss import poisson_loss, NB, ZINB
from .layers import nan2zeroLayer, ConstantDispersionLayer, SliceLayer, ColWiseMultLayer
from .io import save_matrix, estimate_size_factors, lognormalize

class MLP(object):
    def __init__(self,
                 input_size,
                 output_size=None,
                 hidden_size=(256,),
                 l2_coef=0.,
                 hidden_dropout=0.,
                 activation='relu',
                 init='glorot_uniform',
                 masking=False,
                 loss_type='zinb',
                 file_path=None):
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

        assert loss_type in ['normal', 'poisson', 'nb', 'zinb', 'zinb-conddisp'], \
                             'loss type not supported'

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.l2_coef = l2_coef
        self.hidden_dropout = hidden_dropout
        self.activation = activation
        self.init = init
        self.masking = masking
        self.loss_type = loss_type
        self.file_path = file_path

        self.ae = True if self.output_size is None else False
        if self.output_size is None:
            self.output_size = input_size

        if isinstance(self.hidden_dropout, list):
            assert len(self.hidden_dropout) == len(self.hidden_size)
        else:
            self.hidden_dropout = [self.hidden_dropout]*len(self.hidden_size)


    def build(self):

        inp = Input(shape=(self.input_size,), name='count')
        size_factor_inp = Input(shape=(1,), name='size_factors')
        if self.masking:
            nan = nan2zeroLayer(inp)
            last_hidden = nan
        else:
            last_hidden = inp

        for i, (hid_size, hid_drop) in enumerate(zip(self.hidden_size, self.hidden_dropout)):
            if i == int(np.floor(len(self.hidden_size) / 2.0)):
                layer_name = 'center'
            else:
                layer_name = 'hidden%s' % i

            last_hidden = Dense(hid_size, activation=None, kernel_initializer=self.init,
                          kernel_regularizer=l2(self.l2_coef), name=layer_name)(last_hidden)
            last_hidden = BatchNormalization()(last_hidden)

            # Use separate act. layers to give user the option to get pre-activations
            # of layers when requested
            last_hidden = Activation(self.activation, name='%s_act'%layer_name)(last_hidden)
            last_hidden = Dropout(hid_drop, name='%s_drop'%layer_name)(last_hidden)

        self.extra_models  = {}

        if self.loss_type == 'normal':
            self.loss = mean_squared_error
            mean = Dense(self.output_size, activation=None, kernel_initializer=self.init,
                         kernel_regularizer=l2(self.l2_coef), name='mean')(last_hidden)
            output = ColWiseMultLayer(name='output')([mean, size_factor_inp])

            # keep unscaled output as an extra model
            self.extra_models['mean_norm'] = Model(inputs=inp, outputs=mean)

        elif self.loss_type == 'poisson':
            mean = Dense(self.output_size, activation=K.exp, kernel_initializer=self.init,
                         kernel_regularizer=l2(self.l2_coef), name='mean')(last_hidden)
            output = ColWiseMultLayer(name='output')([mean, size_factor_inp])
            self.loss = poisson_loss
            self.extra_models['mean_norm'] = Model(inputs=inp, outputs=mean)

        elif self.loss_type == 'nb':

            mean = Dense(self.output_size, activation=K.exp, kernel_initializer=self.init,
                         kernel_regularizer=l2(self.l2_coef), name='mean')(last_hidden)

            # Plug in dispersion parameters via fake dispersion layer
            disp = ConstantDispersionLayer(name='dispersion')
            mean = disp(mean)

            output = ColWiseMultLayer(name='output')([mean, size_factor_inp])

            nb = NB(disp.theta_exp, masking=self.masking)
            self.loss = nb.loss
            self.extra_models['dispersion'] = lambda :K.function([], [nb.theta])([])[0].squeeze()
            self.extra_models['mean_norm'] = Model(inputs=inp, outputs=mean)

        elif self.loss_type == 'zinb':
            pi = Dense(self.output_size, activation='sigmoid', kernel_initializer=self.init,
                       kernel_regularizer=l2(self.l2_coef), name='pi')(last_hidden)

            # NB dispersion layer
            disp = ConstantDispersionLayer(name='dispersion')
            last_hidden = disp(last_hidden)

            mean = Dense(self.output_size, activation=K.exp, kernel_initializer=self.init,
                         kernel_regularizer=l2(self.l2_coef), name='mean')(last_hidden)
            output = ColWiseMultLayer(name='output')([mean, size_factor_inp])

            # Inject pi layer via slicing
            output = SliceLayer(index=0, name='slice')([output, pi])

            zinb = ZINB(pi, theta=disp.theta_exp, masking=self.masking)
            self.loss = zinb.loss
            self.extra_models['pi'] = Model(inputs=inp, outputs=pi)
            self.extra_models['dispersion'] = lambda :K.function([], [zinb.theta])([])[0].squeeze()
            self.extra_models['mean_norm'] = Model(inputs=inp, outputs=mean)

        # ZINB with gene-wise dispersion
        elif self.loss_type == 'zinb-conddisp':
            pi = Dense(self.output_size, activation='sigmoid', kernel_initializer=self.init,
                           kernel_regularizer=l2(self.l2_coef), name='pi')(last_hidden)

            disp = Dense(self.output_size, activation=lambda x:1.0/(K.exp(x)+1e-10),
                               kernel_initializer=self.init,
                               kernel_regularizer=l2(self.l2_coef), name='dispersion')(last_hidden)

            mean = Dense(self.output_size, activation=K.exp, kernel_initializer=self.init,
                           kernel_regularizer=l2(self.l2_coef), name='mean')(last_hidden)
            mulmean = ColWiseMultLayer(name='output')([mean, size_factor_inp])

            # Inject pi and disp layers via slicing
            output = SliceLayer(index=0, name='slice')([mulmean, pi, disp])

            zinb = ZINB(pi, theta=disp, masking=self.masking)
            self.loss = zinb.loss
            self.extra_models['pi'] = Model(inputs=inp, outputs=pi)
            self.extra_models['conddispersion'] = Model(inputs=inp, outputs=disp)
            self.extra_models['mean_norm'] = Model(inputs=inp, outputs=mean)

        self.model = Model(inputs=[inp, size_factor_inp], outputs=output)

        if self.ae:
            self.encoder_linear = self.get_encoder(activation = False)
            self.encoder = self.get_encoder(activation = True)
            self.decoder = None #get_decoder()

    def save(self):
        os.makedirs(self.file_path, exist_ok=True)
        with open(os.path.join(self.file_path, 'model.pickle'), 'wb') as f:
            pickle.dump(self, f)

    def load_weights(self, filename):
        self.model.load_weights(filename)
        if self.ae:
            self.encoder_linear = self.get_encoder(activation = False)
            self.encoder = self.get_encoder(activation = True)
            self.decoder = None #get_decoder()

    def get_decoder(self):
        i = 0
        for l in self.model.layers:
            if l.name == 'center_drop': break
            i += 1

        return Model(inputs=self.model.get_layer(index=i+1).input,
                     outputs=selfodel.output)

    def get_encoder(self, activation=True):
        if activation:
            ret =  Model(inputs=self.model.input,
                         outputs=self.model.get_layer('center_act').output)
        else:
            ret =  Model(inputs=self.model.input,
                         outputs=self.model.get_layer('center').output)
        return ret

    def predict(self, count_matrix, size_factors=True, normalize_input=True,
                dimreduce=True, reconstruct=True):
        res = {}
        if size_factors:
            size_factors = estimate_size_factors(count_matrix)
        else:
            size_factors = np.ones((count_matrix.shape[0],))

        if normalize_input:
            count_matrix = lognormalize(count_matrix, size_factors)

        if 'dispersion' in self.extra_models:
            res['dispersion'] = self.extra_models['dispersion']()

        if 'pi' in self.extra_models:
            res['pi'] = self.extra_models['pi'].predict(count_matrix)

        if 'conddispersion' in self.extra_models:
            res['dispersion'] = self.extra_models['conddispersion'].predict(count_matrix)

        if dimreduce:
            print('Calculating low dimensional representations...')
            res['reduced'] = self.encoder.predict({'count': count_matrix,
                                                   'size_factors': size_factors})
            res['reduced_linear'] = self.encoder_linear.predict({'count': count_matrix,
                                                                 'size_factors': size_factors})

        if reconstruct:
            print('Calculating reconstructions...')
            res['mean'] = self.model.predict({'count': count_matrix,
                                              'size_factors': size_factors})

            res['mean_norm'] = self.extra_models['mean_norm'].predict(count_matrix)

            if 'dispersion' in res:
                m, d = res['mean'], res['dispersion']
                res['mode'] = np.floor(m*((d-1)/d)).astype(np.int)
                res['mode'][res['mode'] < 0] = 0

        if self.file_path:
            print('Saving files...')
            os.makedirs(self.file_path, exist_ok=True)
            if 'reduced' in res:
                save_matrix(res['reduced'], os.path.join(self.file_path,
                                                         'reduced.tsv'))
                save_matrix(res['reduced_linear'], os.path.join(self.file_path,
                                                         'reduced_linear.tsv'))
            if 'dispersion' in res:
                save_matrix(res['dispersion'], os.path.join(self.file_path,
                                                            'dispersion.tsv'))
            if 'pi' in res:
                save_matrix(res['pi'], os.path.join(self.file_path, 'pi.tsv'))
            if 'mean' in res:
                save_matrix(res['mean'],
                            os.path.join(self.file_path, 'mean.tsv'))
            if 'mean_norm' in res:
                save_matrix(res['mean_norm'],
                            os.path.join(self.file_path, 'mean_norm.tsv'))
            if 'mode' in res:
                save_matrix(res['mode'],
                            os.path.join(self.file_path, 'mode.tsv'))

        return res

