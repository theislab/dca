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
from .network import AE_types

import numpy as np
import keras.optimizers as opt
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import backend as K
from keras.preprocessing.image import Iterator


def train(ds, network, output_dir, optimizer='Adam', learning_rate=None, train_on_full=False,
          aetype=None, epochs=200, reduce_lr=20, size_factors=True, normalize_input=True,
          logtrans_input=True, early_stop=25, batch_size=32, clip_grad=5., save_weights=True,
          **kwargs):
    model = network.model
    loss = network.loss

    if learning_rate is None:
        optimizer = opt.__dict__[optimizer](clipvalue=clip_grad)
    else:
        optimizer = opt.__dict__[optimizer](lr=learning_rate, clipvalue=clip_grad)

    model.compile(loss=loss, optimizer=optimizer)

    # Callbacks
    checkpointer = ModelCheckpoint(filepath="%s/weights.hdf5" % output_dir,
                                   verbose=1,
                                   save_weights_only=True,
                                   save_best_only=True)
    es_cb = EarlyStopping(monitor='val_loss', patience=early_stop, verbose=1)
    es_cb_train = EarlyStopping(monitor='train_loss', patience=early_stop, verbose=1)

    lr_cb = ReduceLROnPlateau(monitor='val_loss', patience=reduce_lr, verbose=1)
    lr_cb_train = ReduceLROnPlateau(monitor='train_loss', patience=reduce_lr, verbose=1)

    callbacks = []
    callbacks_train = []

    if save_weights:
        callbacks.append(checkpointer)
        callbacks_train.append(checkpointer)
    if reduce_lr:
        callbacks.append(lr_cb)
        callbacks_train.append(lr_cb_train)
    if early_stop:
        callbacks.append(es_cb)
        callbacks_train.append(es_cb_train)

    os.makedirs(output_dir, exist_ok=True)
    model.summary()

    tb_log_dir = os.path.join(output_dir, 'tb')
    tb_cb = TensorBoard(log_dir=tb_log_dir, histogram_freq=1, write_grads=True)

    if size_factors:
        sf_mat = ds.train.size_factors[:]
    else:
        sf_mat = np.ones((ds.train.matrix.shape[0], 1),
                         dtype=np.float32)

    loss = model.fit({'count': io.normalize(ds.train.matrix[:],
                                            sf_mat, logtrans=logtrans_input,
                                            sfnorm=size_factors,
                                            zeromean=normalize_input),
                      'size_factors': sf_mat},
                     ds.train.matrix[:],
                     epochs=epochs,
                     batch_size=batch_size,
                     shuffle=True,
                     callbacks=callbacks + [tb_cb],
                     validation_split=0.1,
                     **kwargs)
    # https://github.com/tensorflow/tensorflow/issues/3388
    # K.clear_session()

    return loss


def train_with_args(args):
    ds = io.create_dataset(args.input,
                           output_file=os.path.join(args.outputdir, 'input.zarr'),
                           transpose=args.transpose,
                           test_split=args.testsplit,
                           size_factors=args.normtype)

    hidden_size = [int(x) for x in args.hiddensize.split(',')]
    hidden_dropout = [float(x) for x in args.dropoutrate.split(',')]
    if len(hidden_dropout) == 1:
        hidden_dropout = hidden_dropout[0]

    assert args.type in AE_types, 'loss type not supported'

    net = AE_types[args.type](input_size=ds.train.shape[1],
            hidden_size=hidden_size,
            l2_coef=args.l2,
            l1_coef=args.l1,
            l2_enc_coef=args.l2enc,
            l1_enc_coef=args.l1enc,
            ridge=args.ridge,
            hidden_dropout=hidden_dropout,
            batchnorm=args.batchnorm,
            activation=args.activation,
            init=args.init,
            file_path=args.outputdir)
    net.save()
    net.build()

    losses = train(ds, net, output_dir=args.outputdir,
                   learning_rate=args.learningrate,
                   epochs=args.epochs, batch_size=args.batchsize,
                   early_stop=args.earlystop,
                   reduce_lr=args.reducelr,
                   size_factors=args.sizefactors,
                   normalize_input=args.norminput,
                   optimizer=args.optimizer,
                   clip_grad=args.gradclip,
                   save_weights=args.saveweights)

    net.predict(ds.full,
                size_factors=args.sizefactors,
                normalize_input=args.norminput,
                logtrans_input=args.loginput)
