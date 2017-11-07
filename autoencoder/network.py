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


class Autoencoder(torch.nn.Module):
    def __init__(self,
                 input_size,
                 enc_size=(32, 32),
                 dec_size=(),
                 out_size=(32,),
                 enc_dropout=0.,
                 dec_dropout=0.,
                 out_dropout=0.,
                 out_modules={'mean': torch.nn.Sequential},
                 batchnorm=True):

        super().__init__()
        self.input_size = input_size

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


        self.encoder = torch.nn.Sequential()
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

        self.add_module('encoder', self.encoder)
        self.decoder = torch.nn.Sequential()

        for i, (d_size, d_drop) in enumerate(zip(dec_size, dec_dropout)):
            layer_name = 'dec%s' % i

            self.decoder.add_module(layer_name, torch.nn.Linear(last_hidden_size, d_size))
            if batchnorm:
                self.decoder.add_module(layer_name + '_bn', torch.nn.BatchNorm1d(d_size, affine=False))

            self.decoder.add_module(layer_name + '_act', torch.nn.ReLU())
            if d_drop > 0.0:
                self.decoder.add_module(layer_name + '_drop', torch.nn.Dropout(d_drop))

            last_hidden_size = d_size

        self.add_module('decoder', self.decoder)
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

        if len(self.outputs) == 1:
            k = list(self.outputs.keys())[0]
            return self.outputs[k].forward(intermediate)

        return {k: v.forward(intermediate) for k, v in self.outputs.items()}

