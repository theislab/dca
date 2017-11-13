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

import os
from collections import namedtuple
import numpy as np
from .utils import *
from .io import write_text_matrix, estimate_size_factors

import torch
from torch.autograd import Variable

LOSS_TYPES = {'mse': torch.nn.MSELoss, 'nb': NBLoss, 'zinb': ZINBLoss,
              'zinbem': ZINBEMLoss, 'poisson': torch.nn.PoissonNLLLoss}

# metadata of an output module
# such as human readable name, row/col names (true/false) and activation
OutModule = namedtuple('OutModule', 'hname rname cname act')

class AEModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        intermediate = self.decoder.forward(self.encoder.forward(input))
        return {k: v.forward(intermediate) for k, v in self.outputs.named_children()}


class Autoencoder():
    def __init__(self,
                 input_size,
                 loss_type, loss_args={},
                 output_size = None,
                 enc_size=(64, 64),
                 dec_size=(),
                 out_size=(64,),
                 enc_dropout=0.,
                 dec_dropout=0.,
                 out_dropout=0.,
                 out_modules={'input': OutModule(hname='mean',
                                                 rname=True,
                                                 cname=True,
                                                 act=torch.nn.Sequential)},
                 activation='ReLU',
                 batchnorm=True):

        assert loss_type in LOSS_TYPES, 'Undefined loss type'

        super().__init__()

        self.input_size = input_size
        self.output_size = self.input_size if output_size is None else output_size
        self.loss = LOSS_TYPES[loss_type](**loss_args)
        self.outputs_metadata = out_modules
        self.model = AEModule()

        encoder = torch.nn.Sequential()
        self.model.add_module('encoder', encoder)

        decoder = torch.nn.Sequential()
        self.model.add_module('decoder', decoder)

        outputsdict = {k: torch.nn.Sequential() for k in out_modules}
        outputsmod = torch.nn.Module()
        self.model.add_module('outputs', outputsmod)

        act = torch.nn.__dict__[activation]

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

        last_hidden_size = input_size

        for i, (e_size, e_drop) in enumerate(zip(enc_size, enc_dropout)):
            layer_name = 'enc%s' % i

            encoder.add_module(layer_name, torch.nn.Linear(last_hidden_size, e_size))

            if batchnorm:
                encoder.add_module(layer_name + '_bn', torch.nn.BatchNorm1d(e_size, affine=False))

            # make a soft copy of the encoder to be able to get embeddings w/o
            # activation
            if i == (len(enc_size)-1):
                self.model.add_module('encoder_linear',
                                      torch.nn.Sequential(*list(self.encoder._modules.values())))

            encoder.add_module(layer_name + '_act', act())
            if e_drop > 0.0:
                encoder.add_module(layer_name + '_drop', torch.nn.Dropout(e_drop))

            last_hidden_size = e_size

        for i, (d_size, d_drop) in enumerate(zip(dec_size, dec_dropout)):
            layer_name = 'dec%s' % i

            decoder.add_module(layer_name, torch.nn.Linear(last_hidden_size, d_size))
            if batchnorm:
                decoder.add_module(layer_name + '_bn', torch.nn.BatchNorm1d(d_size, affine=False))

            decoder.add_module(layer_name + '_act', act())
            if d_drop > 0.0:
                decoder.add_module(layer_name + '_drop', torch.nn.Dropout(d_drop))

            last_hidden_size = d_size

        for i, (o_size, o_drop) in enumerate(zip(out_size, out_dropout)):
            layer_name = 'out%s' % i

            for out in outputsdict:

                outputsdict[out].add_module(out + '_' + layer_name, torch.nn.Linear(last_hidden_size, o_size))
                if batchnorm:
                    outputsdict[out].add_module(out + '_' + layer_name + '_bn',
                                                torch.nn.BatchNorm1d(o_size, affine=False))

                outputsdict[out].add_module(out + '_' + layer_name + '_act',
                                            act())
                if o_drop > 0.0:
                    outputedict[out].add_module(out + '_' + layer_name + '_drop',
                                                torch.nn.Dropout(o_drop))

            last_hidden_size = o_size

        for out_name, out in out_modules.items():
            outputsdict[out_name].add_module(out_name + '_pre',
                                             torch.nn.Linear(last_hidden_size, self.output_size))
            outputsdict[out_name].add_module(out_name, out.act())

            outputsmod.add_module(out_name, outputsdict[out_name])


    def __repr__(self):
        return self.model.__repr__()

    @property
    def encoder(self):
        return self.model.encoder

    @property
    def decoder(self):
        return self.model.decoder

    @property
    def outputs(self):
        return self.model.outputs

    def train(self, X, Y, epochs=300, batch_size=32, l2=0.0,
              l2_enc=0.0, l2_out=0.0, optimizer='RMSprop', optimizer_args={},
              val_split=0.1, grad_clip=5.0, gpu=False):

        optimizer = self.setup_optimizer(optimizer, optimizer_args,
                                         l2, l2_enc, l2_out)
        if gpu:
            print('Running on GPU')
        else:
            print('Running on CPU')

        ret = train(model_dict={'input': X},
                    loss_dict={'target': Y},
                    model=self.model, loss=self.loss, optimizer=optimizer,
                    epochs=epochs, batch_size=batch_size,
                    verbose=1, gpu=gpu, val_split=val_split, grad_clip=grad_clip)

        self.model = ret['model']
        return ret

    def setup_optimizer(self, optimizer, optimizer_args, l2, l2_enc, l2_out):
        params = [{'params': self.encoder.parameters()},
                  {'params': self.decoder.parameters()},
                  {'params': self.outputs.parameters()}]

        if l2_enc:
            params[0]['weight_decay'] = l2_enc
        if l2_out:
            params[2]['weight_decay'] = l2_out

        if l2:
            return torch.optim.__dict__[optimizer](params, **optimizer_args,
                                                   weight_decay=l2)
        else:
            return torch.optim.__dict__[optimizer](params, **optimizer_args)


    def predict(self, X, rownames=None, colnames=None, folder='./', gpu=False):

        X = Variable(torch.from_numpy(X))
        if gpu:
            X = X.cuda()
            model = self.model.cuda()
        else:
            X = X.float()
            model = self.model.float()

        preds = model(X)
        preds = {k: v.data.numpy() for k, v in preds.items()}
        os.makedirs(folder, exist_ok=True)

        for name, mat in preds.items():
            hname = self.outputs_metadata[name].hname
            rname = self.outputs_metadata[name].rname
            cname = self.outputs_metadata[name].cname
            filename = os.path.join(folder, hname + '.tsv')
            print("Saving %s file..." % (filename))

            write_text_matrix(mat, filename,
                              rownames=rownames if rname else None,
                              colnames=colnames if cname else None)
        return preds


class ZINBAutoencoder(Autoencoder):
    def __init__(self, *args, **kwargs):
        out = {'mean': OutModule(hname='mean', rname=True, cname=True, act=ExpModule),
               'pi': OutModule(hname='pi', rname=True, cname=True, act=torch.nn.Sigmoid),
               'theta': OutModule(hname='dispersion', rname=True, cname=True, act=ExpModule)}

        kwargs['out_modules'] = out
        kwargs['loss_type'] = 'zinb'

        super().__init__(*args, **kwargs)

    def predict(self, *args, **kwargs):
        preds = super().predict(*args, **kwargs)

        nb_mode = np.floor(preds['mean']*((preds['theta']-1)/preds['theta'])).astype(np.int)
        nb_mode[nb_mode < 0] = 0

        print("Saving mode.tsv file...")

        write_text_matrix(nb_mode, 'mode.tsv',
                          rownames=kwargs['rownames'], colnames=kwargs['colnames'])


class ZINBEMAutoencoder(Autoencoder):
    def __init__(self, *args, **kwargs):
        out = {'mean': OutModule(hname='mean', rname=True, cname=True, act=ExpModule),
               'pi': OutModule(hname='pi', rname=True, cname=True, act=torch.nn.Sigmoid),
               'theta': OutModule(hname='dispersion', rname=True, cname=True, act=ExpModule)}

        kwargs['out_modules'] = out
        kwargs['loss_type'] = 'zinbem'
        super().__init__(*args, **kwargs)

    def train(self, X, Y, epochs=300, m_epochs=1, batch_size=32, l2=0, l2_enc=0, l2_out=0,
              optimizer='RMSprop', optimizer_args={}, gpu=False, val_split=0.1,
              grad_clip=5.0):

        optimizer = self.setup_optimizer(optimizer, optimizer_args,
                                         l2, l2_enc, l2_out)
        if gpu:
            print('Running on GPU')
        else:
            print('Running on CPU')

        ret = train_em(model_dict={'input': X},
                       loss_dict={'target': Y},
                       model=self.model, loss=self.loss, optimizer=optimizer,
                       epochs=epochs, batch_size=batch_size,
                       verbose=1, m_epochs=m_epochs, gpu=gpu,
                       val_split=val_split, grad_clip=grad_clip)
        self.model = ret['model']
        return ret

    def predict(self, *args, **kwargs):
        preds = super().predict(*args, **kwargs)

        nb_mode = np.floor(preds['mean']*((preds['theta']-1)/preds['theta'])).astype(np.int)
        nb_mode[nb_mode < 0] = 0

        # Save the mode file
        print("Saving mode.tsv file...")
        write_text_matrix(nb_mode, 'mode.tsv',
                          rownames=kwargs['rownames'], colnames=kwargs['colnames'])

        # Save membership file
        memberships = self.loss.zero_memberships(preds['mean'], preds['pi'])



class NBAutoencoder(Autoencoder):
    def __init__(self, *args, **kwargs):
        out = {'mean': OutModule(hname='mean', rname=True, cname=True, act=ExpModule),
               'theta': OutModule(hname='dispersion', rname=True, cname=True, act=ExpModule)}

        kwargs['out_modules'] = out
        kwargs['loss_type'] = 'nb'

        super().__init__(*args, **kwargs)

    def predict(self, *args, **kwargs):
        preds = super().predict(*args, **kwargs)

        nb_mode = np.floor(preds['mean']*((preds['theta']-1)/preds['theta'])).astype(np.int)
        nb_mode[nb_mode < 0] = 0

        print("Saving mode.tsv file...")

        write_text_matrix(nb_mode, 'mode.tsv',
                          rownames=kwargs['rownames'], colnames=kwargs['colnames'])



class PoissonAutoencoder(Autoencoder):
    def __init__(self, *args, **kwargs):
        out = {'log_input': OutModule(hname='log_mean', rname=True, cname=True, act=ExpModule)}

        kwargs['out_modules'] = out
        kwargs['loss_type'] = 'poisson'
        kwargs['loss_args'] = {'full': True, 'log_input': True}

        super().__init__(*args, **kwargs)


class MSEAutoencoder(Autoencoder):
    def __init__(self, *args, **kwargs):
        out = {'input': OutModule(hname='mean', rname=True, cname=True, act=torch.nn.Sequential)}
        kwargs['out_modules'] = out
        kwargs['loss_type'] = 'mse'

        super().__init__(*args, **kwargs)



AE_TYPES = {'zinb': ZINBAutoencoder, 'zinbem': ZINBEMAutoencoder,
            'nb': NBAutoencoder, 'poisson': PoissonAutoencoder,
            'mse': MSEAutoencoder}

