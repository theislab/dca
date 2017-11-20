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
from .network import AE_TYPES
import numpy as np
import torch


def denoise_with_args(args):
    torch.manual_seed(42)
    np.random.seed(42)

    ds = io.text_to_zarr(args.inputfile,
                         output_file=os.path.join(args.outputdir, 'input.zarr'),
                         transpose=args.transpose,
                         test_split=args.testsplit,
                         size_factors=args.normtype)

    enc_size = [int(x) for x in args.enchiddensize.split(',') if x]
    enc_dropout = [float(x) for x in args.encdropoutrate.split(',')]
    dec_size = [int(x) for x in args.dechiddensize.split(',') if x]
    dec_dropout = [float(x) for x in args.decdropoutrate.split(',')]
    out_size = [int(x) for x in args.outhiddensize.split(',') if x]
    out_dropout = [float(x) for x in args.outdropoutrate.split(',')]

    if len(enc_dropout) == 1:
        enc_dropout = enc_dropout[0]
    if len(dec_dropout) == 1:
        dec_dropout = dec_dropout[0]
    if len(out_dropout) == 1:
        out_dropout = out_dropout[0]

    assert args.type in AE_TYPES, 'AE type not supported'

    pi_ridge = {'pi_ridge': args.piridge} if args.piridge else {}

    net = AE_TYPES[args.type](input_size=ds.train.matrix.shape[1],
            enc_size=enc_size,
            enc_dropout=enc_dropout,
            dec_size=dec_size,
            dec_dropout=dec_dropout,
            out_size=out_size,
            out_dropout=out_dropout,
            batchnorm=args.batchnorm,
            activation=args.activation,
            **pi_ridge)

    print(net)

    norm_options = io.NormOptions(sizefactors=args.sizefactors,
                                  lognorm=args.loginput,
                                  zeromean=args.zeromean)

    X = io.normalize(ds.train.matrix[:], ds.train.size_factors, norm_options)
    Y = ds.train.matrix[:]
    dtype = 'cuda' if args.gpu else 'float'

    net.train(X=X, Y=Y,
              epochs=args.epochs,
              batch_size=args.batchsize,
              l2=args.l2,
              l2_enc=args.l2enc,
              l2_out=args.l2out,
              optimizer=args.optimizer,
              dtype=dtype)

    X = io.normalize(ds['full'].matrix[:], ds['full'].size_factors, norm_options)
    net.predict(X,
                rownames=ds['full'].rownames,
                colnames=ds['full'].colnames,
                folder=args.outputdir,
                dtype=dtype)

    # predict on test set if available
    if 'test' in ds:
        X = io.normalize(ds.test.matrix[:], ds.test.size_factors, norm_options)
        net.predict(X,
                    rownames=ds.test.rownames,
                    colnames=ds.test.colnames,
                    folder=os.path.join(args.outputdir, 'testset'),
                    dtype=dtype)
