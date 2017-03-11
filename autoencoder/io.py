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

import pickle, os

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from keras.models import load_model as keras_load_model


def read_csv(inputfile):
    matrix = pd.read_csv(inputfile, sep=None, engine='python')
    matrix = matrix.as_matrix()
    return matrix


def read_from_file(inputfile):
        return pickle.load(open(inputfile, 'rb'))


def load_model(log_dir):
    return keras_load_model("%s/weights.hdf5" % log_dir)


def preprocess(inputfile, kfold, transpose=False, outputfile=None,
               censorfile=None, censortranspose=False):
    matrix = read_csv(inputfile)
    if transpose:
        matrix = matrix.transpose()

    X = dict()
    X['shape'] = matrix.shape

    if censorfile:
        censor = read_csv(censorfile)
        if censortranspose:
            censor = censor.transpose()

        assert censor.shape == X['shape'], 'Input size of censorfile does not ' \
                                           'match the input file'

    X['k'] = kfold if kfold else -1
    X['folds'] = list()
    if censorfile:
        X['censorfolds'] = list()

    nsample = matrix.shape[0]

    # Prepare indices for k-fold cv and train/valid/test split
    # For example 10-fold generates 8-1-1 equally sized folds for
    # train/val/test tests
    for cv_trainval, cv_test in KFold(kfold if kfold else 10, True, 42).split(range(nsample)):
        cv_train, cv_val = train_test_split(cv_trainval,
                                            test_size=1/((kfold if kfold else 10) - 1),
                                            random_state=42)
        X['folds'].append({'train': matrix[cv_train, :],
                           'val':   matrix[cv_val,   :],
                           'test':  matrix[cv_test,  :]})

        if censorfile:
            X['censorfolds'].append({'train': censor[cv_train, :],
                                     'val':   censor[cv_val,   :],
                                     'test':  censor[cv_test,  :]})
        if not kfold: break #if kfold is not give just split it 8-1-1 once


    if outputfile:
        # add file extension if missing
        outfile, outfile_ext = os.path.splitext(outputfile)
        if not outfile_ext:
            ext = 'ae'
            outputfile = '.'.join([outfile, ext])

        with open(outputfile, 'wb') as out:
            pickle.dump(X, out)

    return X


def preprocess_with_args(args):
    preprocess(args.input, kfold=args.kfold, transpose=args.transpose,
               outputfile=args.output, censorfile=args.censorfile,
               censortranspose=args.censortranspose)
