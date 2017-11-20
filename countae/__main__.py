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

import os, sys, argparse
import numpy as np
import six

from . import io, denoise


def parse_args():
    parser = argparse.ArgumentParser(description='Deep count autoencoder')

    parser.add_argument('inputfile', type=str, help='Input in TSV/CSV format (cells x genes). Note that row/col names are mandatory.')
    parser.add_argument('outputdir', type=str, help="The directory where everything will be will be saved")

    # IO options
    parser.add_argument('--normtype', type=str, default='zheng',
            help='Type of size factor estimation. Possible values: deseq, zheng (default: zheng)')
    parser.add_argument('-t', '--transpose', dest='transpose',
            action='store_true', help='Transpose input matrix (default: False)')
    parser.add_argument('--testsplit', dest='testsplit',
            action='store_true', help="Use 10 percent as a test set (default: False)")

    # Training options
    parser.add_argument('--type', type=str, default='zinbem',
            help="Type of autoencoder. Possible values: mse, poisson, nb, "
                 "zinb, zinbem(default)")
    parser.add_argument('-b', '--batchsize', type=int, default=32,
            help="Batch size (default:32)")
    parser.add_argument('--gpu', dest='gpu',
            action='store_true', help="Run on GPU")
    parser.add_argument('--sizefactors', dest='sizefactors',
            action='store_true', help="Normalize means by library size (default: True)")
    parser.add_argument('--nosizefactors', dest='sizefactors',
            action='store_false', help="Do not normalize means by library size")
    parser.add_argument('--norminput', dest='zeromean',
            action='store_true', help="Zero-mean normalize input (default: True)")
    parser.add_argument('--nonorminput', dest='zeromean',
            action='store_false', help="Do not zero-mean normalize inputs")
    parser.add_argument('--loginput', dest='loginput',
            action='store_true', help="Log-transform input (default: True)")
    parser.add_argument('--nologinput', dest='loginput',
            action='store_false', help="Do not log-transform inputs")
    parser.add_argument('--batchnorm', dest='batchnorm', action='store_true',
            help="Batchnorm (default: True)")
    parser.add_argument('--nobatchnorm', dest='batchnorm', action='store_false',
            help="Do not use batchnorm")
    parser.add_argument('--l2', type=float, default=0.0,
            help="L2 regularization coefficient (default: 0.0)")
    parser.add_argument('--l2enc', type=float, default=0.0,
            help="Encoder-specific L2 regularization coefficient (default: 0.0)")
    parser.add_argument('--l2out', type=float, default=0.0,
            help="Output ranch-specific L2 regularization coefficient (default: 0.0)")
    parser.add_argument('--piridge', type=float, default=0.0,
            help="L2 regularization coefficient for dropout probabilities (default: 0.0)")
    parser.add_argument('--activation', type=str, default='ReLU',
            help="Activation function of hidden units (default: ReLU)")
    parser.add_argument('--optimizer', type=str, default='RMSprop',
            help="Optimization method (default: RMSprop)")
    parser.add_argument('-e', '--epochs', type=int, default=300,
            help="Max number of epochs to continue training in case of no "
                 "improvement on validation loss (default: 200)")
    parser.add_argument('--enchiddensize', type=str, default='64,64',
            help="Size of hidden encoder layers (default: 64,64)")
    parser.add_argument('--dechiddensize', type=str, default='',
            help="Size of hidden decoder layers (default: None)")
    parser.add_argument('--outhiddensize', type=str, default='64',
            help="Size of hidden output branch layers (default: 64)")
    parser.add_argument('--encdropoutrate', type=str, default='0.0',
            help="Dropout rate for encoder layers (default: 0)")
    parser.add_argument('--decdropoutrate', type=str, default='0.0',
            help="Dropout rate for decoder layers (default: 0)")
    parser.add_argument('--outdropoutrate', type=str, default='0.0',
            help="Dropout rate for output branch layers (default: 0)")

    parser.set_defaults(func=denoise.denoise_with_args, batchnorm=True)

    return parser.parse_args()


def main():
    args = parse_args()
    args.func(args)
