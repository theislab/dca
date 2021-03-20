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
import random

from . import io
from .network import AE_types
from .hyper import hyper

import numpy as np
import tensorflow as tf
import keras.optimizers as opt
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import backend as K
from keras.preprocessing.image import Iterator


def train(adata, network, output_dir=None, optimizer='RMSprop', learning_rate=None,
          epochs=300, reduce_lr=10, output_subset=None, use_raw_as_output=True,
          early_stop=15, batch_size=32, clip_grad=5., save_weights=False,
          validation_split=0.1, tensorboard=False, verbose=True, threads=None,
          **kwds):

    tf.compat.v1.keras.backend.set_session(
        tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(
                intra_op_parallelism_threads=threads,
                inter_op_parallelism_threads=threads,
            )
        )
    )
    model = network.model
    loss = network.loss
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    if learning_rate is None:
        optimizer = opt.__dict__[optimizer](clipvalue=clip_grad)
    else:
        optimizer = opt.__dict__[optimizer](lr=learning_rate, clipvalue=clip_grad)

    model.compile(loss=loss, optimizer=optimizer)

    # Callbacks
    callbacks = []

    if save_weights and output_dir is not None:
        checkpointer = ModelCheckpoint(filepath="%s/weights.hdf5" % output_dir,
                                       verbose=verbose,
                                       save_weights_only=True,
                                       save_best_only=True)
        callbacks.append(checkpointer)
    if reduce_lr:
        lr_cb = ReduceLROnPlateau(monitor='val_loss', patience=reduce_lr, verbose=verbose)
        callbacks.append(lr_cb)
    if early_stop:
        es_cb = EarlyStopping(monitor='val_loss', patience=early_stop, verbose=verbose)
        callbacks.append(es_cb)
    if tensorboard:
        tb_log_dir = os.path.join(output_dir, 'tb')
        tb_cb = TensorBoard(log_dir=tb_log_dir, histogram_freq=1, write_grads=True)
        callbacks.append(tb_cb)

    if verbose: model.summary()

    inputs = {'count': adata.X, 'size_factors': adata.obs.size_factors}

    if output_subset:
        gene_idx = [np.where(adata.raw.var_names == x)[0][0] for x in output_subset]
        output = adata.raw.X[:, gene_idx] if use_raw_as_output else adata.X[:, gene_idx]
    else:
        output = adata.raw.X if use_raw_as_output else adata.X

    loss = model.fit(inputs, output,
                     epochs=epochs,
                     batch_size=batch_size,
                     shuffle=True,
                     callbacks=callbacks,
                     validation_split=validation_split,
                     verbose=verbose,
                     **kwds)

    return loss


def train_with_args(args):

    tf.compat.v1.keras.backend.set_session(
        tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(
                intra_op_parallelism_threads=args.threads,
                inter_op_parallelism_threads=args.threads,
            )
        )
    )
    # set seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    os.environ['PYTHONHASHSEED'] = '0'

    # do hyperpar optimization and exit
    if args.hyper:
        hyper(args)
        return

    adata = io.read_dataset(args.input,
                            transpose=(not args.transpose), # assume gene x cell by default
                            check_counts=args.checkcounts,
                            test_split=args.testsplit)

    adata = io.normalize(adata,
                         size_factors=args.sizefactors,
                         logtrans_input=args.loginput,
                         normalize_input=args.norminput)

    if args.denoisesubset:
        genelist = list(set(io.read_genelist(args.denoisesubset)))
        assert len(set(genelist) - set(adata.var_names.values)) == 0, \
               'Gene list is not overlapping with genes from the dataset'
        output_size = len(genelist)
    else:
        genelist = None
        output_size = adata.n_vars

    hidden_size = [int(x) for x in args.hiddensize.split(',')]
    hidden_dropout = [float(x) for x in args.dropoutrate.split(',')]
    if len(hidden_dropout) == 1:
        hidden_dropout = hidden_dropout[0]

    assert args.type in AE_types, 'loss type not supported'
    input_size = adata.n_vars

    from tensorflow.python.framework.ops import disable_eager_execution
    disable_eager_execution()

    net = AE_types[args.type](input_size=input_size,
            output_size=output_size,
            hidden_size=hidden_size,
            l2_coef=args.l2,
            l1_coef=args.l1,
            l2_enc_coef=args.l2enc,
            l1_enc_coef=args.l1enc,
            ridge=args.ridge,
            hidden_dropout=hidden_dropout,
            input_dropout=args.inputdropout,
            batchnorm=args.batchnorm,
            activation=args.activation,
            init=args.init,
            debug=args.debug,
            file_path=args.outputdir)

    net.save()
    net.build()

    losses = train(adata[adata.obs.dca_split == 'train'], net,
                   output_dir=args.outputdir,
                   learning_rate=args.learningrate,
                   epochs=args.epochs, batch_size=args.batchsize,
                   early_stop=args.earlystop,
                   reduce_lr=args.reducelr,
                   output_subset=genelist,
                   optimizer=args.optimizer,
                   clip_grad=args.gradclip,
                   save_weights=args.saveweights,
                   tensorboard=args.tensorboard)

    if genelist:
        predict_columns = adata.var_names[[np.where(adata.var_names==x)[0][0] for x in genelist]]
    else:
        predict_columns = adata.var_names

    net.predict(adata, mode='full', return_info=True)
    net.write(adata, args.outputdir, mode='full', colnames=predict_columns)
