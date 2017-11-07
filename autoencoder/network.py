# Copyright (C) 2017 Goekcen Eraslan
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from .utils import *
from .io import write_text_matrix, estimate_size_factors, normalize

import torch
from torch.autograd import Variable

LOSS_TYPES = {'mse': torch.nn.MSELoss, 'nb': NBLoss, 'zinb': ZINBLoss,
              'zinbem': ZINBEMLoss, 'poisson': torch.nn.PoissonNLLLoss}


class Autoencoder(torch.nn.Module):
    def __init__(self,
                 input_size,
                 loss_type, loss_args={},
                 enc_size=(32, 32),
                 dec_size=(),
                 out_size=(32,),
                 enc_dropout=0.,
                 dec_dropout=0.,
                 out_dropout=0.,
                 out_modules={'input': torch.nn.Sequential},
                 batchnorm=True):

        assert loss_type in LOSS_TYPES, 'Undefined loss type'

        super().__init__()

        self.input_size = input_size
        self.loss = LOSS_TYPES[loss_type](**loss_args)

        if isinstance(enc_dropout, list):
            assert len(enc_dropout) == len(enc_size)
        else:
            enc_dropout = [enc_dropout]*len(enc_size)

        if isinstance(dec_dropout, list):
            assert len(dec_dropout) == len(dec_size)
        else:
            dec_dropout = [dec_dropout]*len(dec_size)

        if isinstance(out_dropout, list):
            assert len(out_dropout) == len(out_size)
        else:
            out_dropout = [out_dropout]*len(out_size)


        encoder = torch.nn.Sequential()
        self.add_module('encoder', encoder)
        last_hidden_size = input_size

        for i, (e_size, e_drop) in enumerate(zip(enc_size, enc_dropout)):
            layer_name = 'enc%s' % i

            self.encoder.add_module(layer_name, torch.nn.Linear(last_hidden_size, e_size))
            if batchnorm:
                self.encoder.add_module(layer_name + '_bn', torch.nn.BatchNorm1d(e_size, affine=False))

            self.encoder.add_module(layer_name + '_act', torch.nn.ReLU())
            if e_drop > 0.0:
                self.encoder.add_module(layer_name + '_drop', torch.nn.Dropout(e_drop))

            last_hidden_size = e_size

        decoder = torch.nn.Sequential()
        self.add_module('decoder', decoder)

        for i, (d_size, d_drop) in enumerate(zip(dec_size, dec_dropout)):
            layer_name = 'dec%s' % i

            self.decoder.add_module(layer_name, torch.nn.Linear(last_hidden_size, d_size))
            if batchnorm:
                self.decoder.add_module(layer_name + '_bn', torch.nn.BatchNorm1d(d_size, affine=False))

            self.decoder.add_module(layer_name + '_act', torch.nn.ReLU())
            if d_drop > 0.0:
                self.decoder.add_module(layer_name + '_drop', torch.nn.Dropout(d_drop))

            last_hidden_size = d_size

        self.outputs = {k: torch.nn.Sequential() for k in out_modules}

        for i, (o_size, o_drop) in enumerate(zip(out_size, out_dropout)):
            layer_name = 'out%s' % i

            for out in self.outputs:

                self.outputs[out].add_module(out + '_' + layer_name, torch.nn.Linear(last_hidden_size, o_size))
                if batchnorm:
                    self.outputs[out].add_module(out + '_' + layer_name + '_bn',
                                                 torch.nn.BatchNorm1d(o_size, affine=False))

                self.outputs[out].add_module(out + '_' + layer_name + '_act',
                                             torch.nn.ReLU())
                if o_drop > 0.0:
                    self.outpute[out].add_module(out + '_' + layer_name + '_drop',
                                                 torch.nn.Dropout(o_drop))

            last_hidden_size = o_size

        for out, out_act  in out_modules.items():
            self.outputs[out].add_module(out + '_pre', torch.nn.Linear(last_hidden_size, self.input_size))
            self.outputs[out].add_module(out, out_act())

            self.add_module(out, self.outputs[out])


    def forward(self, input):
        intermediate = self.decoder.forward(self.encoder.forward(input))
        return {k: v.forward(intermediate) for k, v in self.outputs.items()}


    def train(self, X, Y, epochs=300, batch_size=32,
              optimizer='Adadelta', optimizer_args={}):

        optimizer = torch.optim.__dict__[optimizer](list(self.parameters()) +
                                                    list(self.loss.parameters()),
                                                    **optimizer_args)

        return train(model_dict={'input': X},
                     loss_dict={'target': Y},
                     model=self, loss=self.loss, optimizer=optimizer,
                     epochs=epochs, batch_size=batch_size,
                     verbose=1)


class ZINBAutoencoder(Autoencoder):
    def __init__(self, *args, **kwargs):
        out = {'input': ExpModule, 'pi': torch.nn.Sigmoid,
               'theta': ExpModule}

        kwargs['out_modules'] = out
        kwargs['loss_type'] = 'zinb'

        super().__init__(*args, **kwargs)


class ZINBEMAutoencoder(Autoencoder):
    def __init__(self, *args, **kwargs):
        out = {'input': ExpModule, 'pi': torch.nn.Sigmoid,
               'theta': ExpModule}

        kwargs['out_modules'] = out
        kwargs['loss_type'] = 'zinbem'
        super().__init__(*args, **kwargs)

    def train(self, X, Y, epochs=300, m_epochs=1, batch_size=32,
              optimizer='Adadelta', optimizer_args={}):

        optimizer = torch.optim.__dict__[optimizer](list(self.parameters()) +
                                                    list(self.loss.parameters()),
                                                    **optimizer_args)

        return train_em(model_dict={'input': X},
                       loss_dict={'target': Y},
                       model=self, loss=self.loss, optimizer=optimizer,
                       epochs=epochs, batch_size=batch_size,
                       verbose=1, m_epochs=m_epochs)


class NBAutoencoder(Autoencoder):
    def __init__(self, *args, **kwargs):
        out = {'input': ExpModule, 'theta': ExpModule}
        kwargs['out_modules'] = out
        kwargs['loss_type'] = 'nb'

        super().__init__(*args, **kwargs)


class PoissonAutoencoder(Autoencoder):
    def __init__(self, *args, **kwargs):
        out = {'log_input': torch.nn.Sequential}
        kwargs['out_modules'] = out
        kwargs['loss_type'] = 'poisson'
        kwargs['loss_args'] = {'full': True, 'log_input': True}

        super().__init__(*args, **kwargs)


class MSEAutoencoder(Autoencoder):
    def __init__(self, *args, **kwargs):
        out = {'input': torch.nn.Sequential}
        kwargs['out_modules'] = out
        kwargs['loss_type'] = 'mse'

        super().__init__(*args, **kwargs)



AE_TYPES = {'zinb': ZINBAutoencoder, 'zinbem': ZINBEMAutoencoder,
            'nb': NBAutoencoder, 'poisson': PoissonAutoencoder,
            'mse': MSEAutoencoder}

