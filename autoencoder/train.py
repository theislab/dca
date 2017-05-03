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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from . import io
from .network import MLP

import numpy as np
import keras.optimizers as opt
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import backend as K
from keras.preprocessing.image import Iterator


def train(X, network, output_dir, optimizer='Adam', learning_rate=None, train_on_full=False,
          aetype=None, epochs=200, reduce_lr=20, size_factors=False, early_stop=25, batch_size=32,
          clipvalue=5., **kwargs):

    model = network.model
    loss = network.loss

    if learning_rate is None:
        optimizer = opt.__dict__[optimizer](clipvalue=clipvalue)
    else:
        optimizer = opt.__dict__[optimizer](lr=learning_rate, clipvalue=clipvalue)

    model.compile(loss=loss, optimizer=optimizer)

    # Callbacks
    checkpointer = ModelCheckpoint(filepath="%s/weights.hdf5" % output_dir,
                                   verbose=1,
                                   save_weights_only=True,
                                   save_best_only=True)
    es_cb = EarlyStopping(monitor='val_loss', patience=early_stop)
    es_cb_train = EarlyStopping(monitor='train_loss', patience=early_stop)

    lr_cb = ReduceLROnPlateau(monitor='val_loss', patience=reduce_lr)
    lr_cb_train = ReduceLROnPlateau(monitor='train_loss', patience=reduce_lr)

    callbacks = [checkpointer]
    callbacks_train = [checkpointer]

    if reduce_lr:
        callbacks.append(lr_cb)
        callbacks_train.append(lr_cb_train)
    if early_stop:
        callbacks.append(es_cb)
        callbacks_train.append(es_cb_train)

    os.makedirs(output_dir, exist_ok=True)
    print(model.summary())

    fold_losses = list()
    for i, data in enumerate(X['folds']):
        tb_log_dir = os.path.join(output_dir, 'tb', 'fold%0i' % i)
        tb_cb = TensorBoard(log_dir=tb_log_dir, histogram_freq=1)

        loss = model.fit({'count': data['train']['matrix'],
                          'size_factors': data['train']['size_factors']},
                         data['train']['matrix'],
                         epochs=epochs,
                         batch_size=batch_size,
                         shuffle=True,
                         callbacks=callbacks+[tb_cb],
                         validation_data=(([data['val']['matrix'],
                                           data['val']['size_factors']],
                                           data['val']['matrix'])),
                         **kwargs)
        #model.evaluate(data['test'], data['test'], batch_size=32,
        #               verbose=1, sample_weight=None)
        fold_losses.append(loss.history)

    ret =  {'fold': fold_losses}

    if train_on_full:
        # run final training on full dataset
        full_data = X['full']
        tb_log_dir = os.path.join(output_dir, 'tb', 'full')
        tb_cb = TensorBoard(log_dir=tb_log_dir, histogram_freq=1)

        loss = model.fit({'count': full_data,
                          'size_factors': X['size_factors']},
                         full_data,
                         epochs=epochs,
                         batch_size=batch_size,
                         shuffle=True,
                         callbacks=callbacks_train+[tb_cb],
                         **kwargs)
        ret['full'] = loss.history

    #https://github.com/tensorflow/tensorflow/issues/3388
    #K.clear_session()

    return ret


def train_with_args(args):
    x = io.read_from_file(args.trainingset)

    hidden_size = [int(x) for x in args.hiddensize.split(',')]
    hidden_dropout = [float(x) for x in args.dropoutrate.split(',')]
    if len(hidden_dropout) == 1: hidden_dropout = hidden_dropout[0]

    net = MLP(x['shape'][1],
              file_path = args.outputdir,
              hidden_size=hidden_size,
              l2_coef=args.l2,
              hidden_dropout=hidden_dropout,
              activation=args.activation,
              init=args.init,
              masking=(x['mask'] is not None),
              loss_type=args.type)
    net.save()
    net.build()

    losses = train(x, net, output_dir=args.outputdir,
                   learning_rate=args.learningrate,
                   epochs=args.epochs, batch_size=args.batchsize,
                   early_stop=args.earlystop,
                   reduce_lr=args.reducelr,
                   size_factors=args.sizefactors,
                   optimizer=args.optimizer,
                   clipvalue=args.gradclip)

    net.predict(x['full'],
            dimreduce=args.dimreduce,
            reconstruct=args.reconstruct,
            size_factors=args.sizefactors)
