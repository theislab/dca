# Copyright 2017 Goekcen Eraslan
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
import zarr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale


class Dataset:

    def __init__(self, filename):
        self.root = zarr.open_group(filename)

    # get attributes through the root zarr group
    # e.g. ds=Dataset('x.zarr'); ds.train.matrix[:]
    def __getattr__(self, item):
        return self.root.__getattr__(item)


def text_to_zarr(input_file, output_file, transpose=False,
                 header='infer', test_split=True, size_factors='zheng'):

    root = zarr.open_group(output_file, 'w')
    matrix, rownames, colnames = read_text_matrix(input_file, header=header, transpose=transpose)

    if matrix.dtype != np.float:
        matrix = matrix.astype(np.float)

    root['shape'] = matrix.shape
    root['rownames'] = rownames
    root['colnames'] = colnames
    root.create_dataset('matrix', data=matrix, chunks=(1024, None))
    root.create_dataset('logmatrix', data=np.log1p(matrix), chunks=(1024, None))
    root['size_factors'] = estimate_size_factors(matrix, normtype=size_factors)

    if not test_split:
        root['train/matrix'] = matrix
        root['train/size_factors'] = root['size_factors'][:]
        root['train/rownames'] = root['rownames'][:]
    else:
        mat_train, mat_test, sf_train, sf_test,\
            rownames_train, rownames_test = train_test_split(matrix,
                                                             root['size_factors'][:],
                                                             test_size=0.1,
                                                             random_state=42)

        root['train/matrix'] = mat_train
        root['test/matrix'] = mat_test
        root['train/size_factors'] = sf_train
        root['test/size_factors'] = sf_test
        root['train/rownames'] = rownames_train
        root['test/rownames'] = rownames_test

    return root


def read_text_matrix(inputfile, type=np.float, header=None, transpose=False):
    df = pd.read_csv(inputfile, sep=None, engine='python', header=header)

    if transpose:
        df = df.transpose()

    # check if colnames exist
    if df.columns.dtype == 'O':
        colnames = df.columns
    else:
        colnames = np.array(['Gene' + str(x) for x in df.columns])

    # check if rownames exist
    if df.index == 'O':
        rownames = df.index
    else:
        rownames = np.array(['Cell' + str(x) for x in df.index])

    # filter out all zero features and samples
    colnames = colnames[df.sum(0) > 0]
    df = df.loc[:, df.sum(0) > 0]

    rownames = rownames[df.sum(1) > 0]
    df = df.loc[df.sum(1) > 0, :]

    matrix = df.as_matrix()
    matrix = matrix.astype(type)
    return matrix, list(rownames), list(colnames)


def write_text_matrix(matrix, filename):
    if issubclass(matrix.dtype.type, np.floating):
        np.savetxt(filename, matrix, fmt="%.6e", delimiter="\t")
    elif issubclass(matrix.dtype.type, np.integer):
        np.savetxt(filename, matrix, fmt="%i", delimiter="\t")
    else:
        raise TypeError


def read_pickle(inputfile):
    return pickle.load(open(inputfile, "rb"))


def estimate_size_factors(x, normtype='zheng'):
    assert normtype in ['deseq', 'zheng']

    if normtype == 'deseq':
        loggeommeans = np.mean(np.log1p(x), 0)
        return np.exp(np.median(np.log1p(x) - loggeommeans, 1))
    elif normtype == 'zheng':
        x = x[:, np.sum(x, 0) >= 1]  # filter out all-zero genes
        s = np.sum(x, 1)
        return s/np.median(s, 0)
    else:
        raise NotImplemented


def lognormalize(x, sf, logtrans=True, sfnorm=True, zscore=True):
    if sfnorm:
        assert len(sf.shape) == 1
        x = x / (sf[:, None]+1e-8)  # colwise div

    if logtrans:
        x = np.log1p(x)

    if zscore:
        x = scale(x, axis=0, copy=False)

    return x


def preprocess_with_args(args):

    ds = Dataset()
    ds.import_from_text(args.input,
                        output_file=args.output,
                        transpose=args.transpose,
                        header=('infer' if args.header else None),
                        test_split=args.testsplit,
                        size_factors=args.normtype)

    return ds
