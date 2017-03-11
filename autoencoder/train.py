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

import os, json

from .network import autoencoder
from . import io

import numpy as np
from keras.optimizers import SGD
from keras.callbacks import TensorBoard, ModelCheckpoint


def train(X, hidden_size=32, learning_rate=0.01,
          log_dir='logs', aetype=None, epochs=10,
          batch_size=32, censortype=None, censorthreshold=None,
          hyperpar=None, **kwargs):

    model, _, _, loss = autoencoder(X['shape'][1], hidden_size=hidden_size,
                                    aetype=aetype)

    optimizer = SGD(lr=learning_rate, momentum=0.9, nesterov=True)
    model.compile(loss=loss, optimizer=optimizer)
    checkpointer = ModelCheckpoint(filepath="%s/weights.hdf5" % log_dir,
                                   verbose=1)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    with open('%s/arch.json' % log_dir, 'w') as f:
        json.dump(model.to_json(), f)

    print(model.summary())

    results = list()
    for data in X['folds']:
        model.fit(data['train'], data['train'],
                nb_epoch=epochs,
                batch_size=batch_size,
                shuffle=True,
                callbacks=[TensorBoard(log_dir=log_dir, histogram_freq=1),
                           checkpointer],
                validation_data=(data['val'], data['val']),
                **kwargs)
        #model.evaluate(data['test'], data['test'], batch_size=32,
        #               verbose=1, sample_weight=None)


def train_with_args(args):
    X = io.read_from_file(args.trainingset)

    train(X=X, hidden_size=args.hiddensize,
          learning_rate=args.learningrate,
          log_dir=args.logdir,
          aetype=args.type,
          epochs=args.epochs,
          censorthreshold=args.censorthreshold,
          censortype=args.censortype,
          hyperpar=args.hyperpar)
