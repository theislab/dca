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
import anndata
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale


class Matrix:
    def __init__(self, grp):
        self._group = grp

    @property
    def shape(self):
        return self._group.matrix.shape[:]

    @property
    def rownames(self):
        return self._group.rownames[:]

    @property
    def colnames(self):
        return self._group.colnames[:]

    @property
    def size_factors(self):
        return self._group.size_factors[:]

    @property
    def matrix(self):
        return self._group.matrix

    def get_sequence(self, batch_size):
        return ZarrSequence(self.matrix, self.size_factors, batch_size)


class ZarrSequence:
    def __init__(self, matrix, batch_size, sf=None):
        self.matrix = matrix
        if sf is None:
            self.size_factors = np.ones((self.matrix.shape[0], 1),
                                        dtype=np.float32)
        else:
            self.size_factors = sf
        self.batch_size = batch_size

    def __len__(self):
        return len(self.matrix) // self.batch_size

    def __getitem__(self, idx):
        batch = self.matrix[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_sf = self.size_factors[idx*self.batch_size:(idx+1)*self.batch_size]

        # return an (X, Y) pair
        return {'count': batch, 'size_factors': batch_sf}, batch


class Dataset:
    def __init__(self, filename):
        self._root = zarr.open_group(filename, mode='r')
        self.train = Matrix(self._root.train)
        self.full = Matrix(self._root.train)
        if 'test' in self._root:
            self.test = Matrix(self._root.test)


def create_dataset(input_file, output_file, transpose=False, test_split=True, size_factors='zheng'):

    root = zarr.open_group(output_file, 'w')

    _, extension = os.path.splitext(input_file)
    extension = extension.lower()

    if extension == '.h5ad':
        matrix, rownames, colnames = read_anndata(input_file, transpose=transpose)
    elif extension in ('.txt', '.tsv', '.csv'):
        matrix, rownames, colnames = read_text_matrix(input_file, transpose=transpose)
    else:
        raise NotImplementedError

    if matrix.dtype != np.float:
        matrix = matrix.astype(np.float)

    logmatrix = np.log1p(matrix)
    root['full/shape'] = matrix.shape
    root['full/rownames'] = rownames
    root['full/colnames'] = colnames
    root.create_dataset('full/matrix', data=matrix, chunks=(128, None))
    root.create_dataset('full/logmatrix', data=logmatrix, chunks=(128, None))
    sf = estimate_size_factors(matrix, normtype=size_factors)
    root['full/size_factors'] = sf

    if not test_split:
        root['train/matrix'] = matrix
        root['train/size_factors'] = sf
        root['train/rownames'] = rownames
        root['train/colnames'] = colnames
    else:
        mat_train, mat_test, \
            logmat_train, logmat_test, \
            sf_train, sf_test,\
            rownames_train, rownames_test = train_test_split(matrix,
                                                             logmatrix,
                                                             sf,
                                                             rownames,
                                                             test_size=0.1,
                                                             random_state=42)

        root['train/matrix'] = mat_train
        root['test/matrix'] = mat_test
        root['train/logmatrix'] = logmat_train
        root['test/logmatrix'] = logmat_test
        root['train/size_factors'] = sf_train
        root['test/size_factors'] = sf_test
        root['train/rownames'] = rownames_train
        root['test/rownames'] = rownames_test
        root['train/colnames'] = colnames
        root['test/colnames'] = colnames

    return Dataset(output_file)


def read_anndata(inputfile, type=np.float, transpose=False):
    adata = anndata.read(inputfile)
    if transpose: adata = adata.transpose()

    matrix, cellnames, genenames = adata.X, adata.obs_names, adata.var_names
    print('### Autoencoder: Successfully preprocessed {} genes and {} cells.'.format(len(genenames), len(cellnames)))
    return matrix.astype(type), list(cellnames), list(genenames)


def read_genelist(filename):
    genelist = list(set(open(filename, 'rt').read().strip().split('\n')))
    assert len(genelist) > 0, 'No genes detected in genelist file'
    print('### Autoencoder: Subset of {} genes will be denoised.'.format(len(genelist)))

    return genelist


def read_text_matrix(inputfile, type=np.float, transpose=False):
    df = pd.read_csv(inputfile, sep=None, header=0, index_col=0, engine='python')

    assert len(df.index) == len(df.index.unique()), 'Rownames are not unique. Please add unique rownames'
    assert len(df.columns) == len(df.columns.unique()), 'Colnames are not unique. Please add unique colnames'

    if transpose:
        df = df.transpose()

    cellnames = df.columns
    genenames = df.index

    # filter out all zero features and samples
    cellnames = cellnames[df.sum(0) > 0]
    df = df.loc[:, df.sum(0) > 0]

    genenames = genenames[df.sum(1) > 0]
    df = df.loc[df.sum(1) > 0, :]

    matrix = df.as_matrix()

    # input matrix is gene x cell but our internal representation must be cell x gene
    matrix = np.ascontiguousarray(matrix.astype(type).T)

    print('### Autoencoder: Successfully preprocessed {} genes and {} cells.'.format(len(genenames), len(cellnames)))

    return matrix, list(cellnames), list(genenames)


def write_text_matrix(matrix, filename, rownames=None, colnames=None, transpose=False):
    if transpose:
        matrix = matrix.T
        rownames, colnames = colnames, rownames

    pd.DataFrame(matrix, index=rownames, columns=colnames).to_csv(filename,
                                                                  sep='\t',
                                                                  index=(rownames is not None),
                                                                  header=(colnames is not None),
                                                                  float_format='%.6f')

def write_anndata(matrix, filename, rownames=None, colnames=None, transpose=False):
    if transpose:
        matrix = matrix.T
        rownames, colnames = colnames, rownames

    adata = anndata.AnnData(matrix,
                            obs=pd.DataFrame(index=rownames),
                            var=pd.DataFrame(index=colnames))
    adata.write(filename)


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
        raise NotImplementedError


def normalize(x, sf, logtrans=True, sfnorm=True, zeromean=True):
    if sfnorm:
        assert len(sf.shape) == 1
        x = x / (sf[:, None]+1e-8)  # colwise div

    if logtrans:
        x = np.log1p(x)

    if zeromean:
        x = scale(x)

    return x
