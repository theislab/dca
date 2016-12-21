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


def write_records(inputfile, outputfile, kfold, transpose=False):
    matrix = pd.read_csv(inputfile, sep=None, engine='python')
    matrix = matrix.as_matrix()
    if transpose:
        matrix = matrix.transpose()

    nsample = matrix.shape[0]

    # add file extension if missing
    outfile, outfile_ext = os.path.splitext(outputfile)
    if not outfile_ext:
        ext = 'npy' if not kfold else 'folds'
        outputfile = '.'.join([outfile, ext])

    if kfold is None:
        np.save(outputfile, matrix)
        return

    X = list()

    # Prepare indices for k-fold cv and train/valid/test split
    for cv_trainval, cv_test in KFold(kfold, True, 42).split(range(nsample)):
        cv_train, cv_val = train_test_split(cv_trainval, test_size=1/(kfold-1))
        X.append((matrix[cv_train, :], matrix[cv_val, :], matrix[cv_test, :]))

    with open(outputfile, 'wb') as out:
        pickle.dump(X, out)


def read_records(inputfile):
    infile, infile_ext = os.path.splitext(inputfile)

    if infile_ext == '.npy':
        return np.load(inputfile)
    elif infile_ext == '.folds':
        return pickle.load(open(inputfile, 'rb'))
    else:
        raise Exception('Undefined extension')


def preprocess(args):
    write_records(args.input, args.output, kfold=args.kfold,
                  transpose=args.transpose)
