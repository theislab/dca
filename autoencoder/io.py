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

import pickle, os, numbers

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import scale
from keras.models import load_model as keras_load_model


def read_text(inputfile, type=np.float, header=None):
    """Reads a csv/tsv file and returns a numpy matrix. No header expected by
    default. Spearator is inferred automatically.

    Args:
        inputfile: Name of the input file
        type: Type of resulting matrix. numpy.float by default.
        header: Whether header should be inferred. None by default does not
            expect a header row. 'infer' automatically infers.

    Returns:
        matrix: Numpy matrix
    """
    matrix = pd.read_csv(inputfile, sep=None, engine='python', header=header)
    matrix = matrix.as_matrix()
    matrix = matrix.astype(type)
    return matrix


def save_matrix(matrix, filename):
    if issubclass(matrix.dtype.type, np.floating):
        np.savetxt(filename, matrix, fmt="%.6e", delimiter="\t")
    elif issubclass(matrix.dtype.type, np.integer):
        np.savetxt(filename, matrix, fmt="%i", delimiter="\t")
    else:
        raise TypeError


def read_from_file(inputfile):
    return pickle.load(open(inputfile, "rb"))


def preprocess(matrix, kfold=None, output_file=None,
               mask=None, test_split=True, size_factors='zheng', log=False):
    '''Accepts the AE input matrix and splits it into k-folds. One fold is
    reserved for val and test set and the rest is for the training. For
    example 10-fold results in a list of 10 folds, 8 for training, 1 for val
    and 1 for test. Pickles everything in a dict, if requested.

    Args:
        matrix: Input matrix for autoencoder.
        kfold: Number of folds to be used in k-fold CV.
        output_file: Name of the output file in pickle format.
        mask: Binary mask matrix of same shape as input denoting the positions
            in the input that will not be used in loss calculations. Elements
            to be masked in the loss is set to NaNs in the input matrix.
        test_split: Use one fold as test set.

    Returns:
        A dictionary with 'shape', 'k', folds' keys.

        'shape' is the initial shape of the input matrix.

        'k' denotes the number of folds.

        'folds' is a list of dicts of length 'k'. Each dict has 'train',
            'val' and 'test' keys (if test_split is True).

    '''
    if matrix.dtype != np.float:
        matrix = matrix.astype(np.float)

    # filter out all zero features
    matrix = matrix[:, matrix.sum(0)>0]

    if log:
        matrix = np.log1p(matrix)

    X = dict()
    X['shape'] = matrix.shape

    X['k'] = kfold if kfold else -1
    X['folds'] = list()
    X['mask'] = mask
    X['full'] = matrix
    size_factors = estimate_size_factors(matrix, normtype=size_factors)
    X['size_factors'] = size_factors

    if mask is not None:
        matrix = matrix.copy()
        matrix[~mask] = np.nan #set elements with zeros to NaN

    nsample = matrix.shape[0]

    # Prepare indices for k-fold cv and train/valid/test split
    # For example 10-fold generates 8-1-1 equally sized folds for
    # train/val/test tests
    for cv_train, cv_val in KFold(kfold if kfold else 10, True, 42).split(range(nsample)):
        if test_split:
            cv_train, cv_test = train_test_split(cv_train,
                                                 test_size=1/((kfold if kfold else 10) - 1),
                                                 random_state=42)

            X['folds'].append({'train': {'matrix': matrix[cv_train, :],
                                         'mask'  : mask[cv_train, :] if mask else None,
                                         'size_factors': size_factors[cv_train]},
                               'val':   {'matrix': matrix[cv_val,   :],
                                         'mask'  : mask[cv_val, :] if mask else None,
                                         'size_factors': size_factors[cv_val]},
                               'test':  {'matrix': matrix[cv_test,  :],
                                         'mask'  : mask[cv_test, :] if mask else None,
                                         'size_factors': size_factors[cv_test]}})
        else:
            X['folds'].append({'train': {'matrix': matrix[cv_train, :],
                                         'mask'  : mask[cv_train, :] if mask else None,
                                         'size_factors': size_factors[cv_train]},
                               'val':   {'matrix': matrix[cv_val,   :],
                                         'mask'  : mask[cv_val, :] if mask else None,
                                         'size_factors': size_factors[cv_val]}})

        if not kfold: break #if kfold is not give just split it 8-1-1 once


    if output_file:
        # add file extension if missing
        outfile, outfile_ext = os.path.splitext(output_file)
        if not outfile_ext:
            ext = 'ae'
            outputfile = '.'.join([outfile, ext])

        with open(output_file, 'wb') as out:
            pickle.dump(X, out)

    return X


def estimate_size_factors(x, normtype='zheng'):
    assert normtype in ['deseq', 'zheng']

    if normtype == 'deseq':
        loggeommeans = np.mean(np.log1p(x), 0)
        return np.exp(np.median(np.log1p(x) - loggeommeans, 1))
    elif normtype == 'zheng':
        x = x[:, np.sum(x, 0)>=1] #filter out all-zero genes
        s = np.sum(x, 1)
        return s/np.median(s, 0)
    else:
        raise NotImplemented


def lognormalize(x, sf):
    assert len(sf.shape) == 1
    ret = x / (sf[:, None]+1e-8) #colwise div
    ret = np.log1p(ret)
    return scale(ret, axis=0, copy=False)


def preprocess_with_args(args):
    matrix = read_text(args.input, header=('infer' if args.header else None))
    if args.transpose:
        matrix = matrix.transpose()

    if args.maskfile:
        mask = read_text(args.maskfile, type=np.bool)
        if args.masktranspose:
            mask = mask.transpose()

        assert mask.shape == matrix.shape, 'Input size of maskfile does not ' \
                                           'match that of the input file'
    else:
        mask = None

    result = preprocess(matrix, kfold=args.kfold,
                        output_file=args.output, mask=mask,
                        test_split=args.testsplit,
                        size_factors=args.normtype, log=args.log)
