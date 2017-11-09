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

from . import io, train


def parse_args():
    parser = argparse.ArgumentParser(description='Autoencoder')
    subparsers = parser.add_subparsers(title='subcommands',
            help='sub-command help', description='valid subcommands', dest='cmd')
    subparsers.required = True

    # Preprocess subparser
    parser_preprocess = subparsers.add_parser('preprocess',
            help='Create a training set from CSV/TSV files')
    parser_preprocess.add_argument('input', type=str,
            help='Input in TSV/CSV format')
    parser_preprocess.add_argument('-o', '--output', type=str,
            help='Output file path', required=True)
    parser_preprocess.add_argument('--normtype', type=str, default='zheng',
            help='Type of size factor estimation. Possible values: deseq, zheng.'
                 ' (default: zheng)')
    parser_preprocess.add_argument('-t', '--transpose', dest='transpose',
            action='store_true', help='Transpose input matrix (default: False)')
    parser_preprocess.add_argument('--testsplit', dest='testsplit',
            action='store_true', help="Use one fold as a test set (default: False)")

    parser_preprocess.set_defaults(func=io.preprocess_with_args)

    # denoise subparser
    parser_denoise = subparsers.add_parser('denoise',
            help='denoise an autoencoder using given dataset and make predictions.')
    parser_denoise.add_argument('trainingset', type=str,
            help="File path of the training set ")
    parser_denoise.add_argument('-o', '--outputdir', type=str, required=True,
            help="The directory where everything will be will be saved")
    parser_denoise.add_argument('-t', '--type', type=str, default='zinbem',
            help="Type of autoencoder. Possible values: mse, poisson, nb, "
                 "zinb, zinbem(default)")
    parser_denoise.add_argument('-b', '--batchsize', type=int, default=32,
            help="Batch size (default:32)")
    parser_denoise.add_argument('--gpu', dest='gpu',
            action='store_true', help="Run on GPU")
    parser_denoise.add_argument('--sizefactors', dest='sizefactors',
            action='store_true', help="Normalize means by library size (default: True)")
    parser_denoise.add_argument('--nosizefactors', dest='sizefactors',
            action='store_false', help="Do not normalize means by library size")
    parser_denoise.add_argument('--norminput', dest='norminput',
            action='store_true', help="Zero-mean normalize input (default: True)")
    parser_denoise.add_argument('--nonorminput', dest='norminput',
            action='store_false', help="Do not zero-mean normalize inputs")
    parser_denoise.add_argument('--loginput', dest='loginput',
            action='store_true', help="Log-transform input (default: True)")
    parser_denoise.add_argument('--nologinput', dest='loginput',
            action='store_false', help="Do not log-transform inputs")
    parser_denoise.add_argument('--batchnorm', dest='batchnorm', action='store_true',
            help="Batchnorm (default: True)")
    parser_denoise.add_argument('--nobatchnorm', dest='batchnorm', action='store_false',
            help="Do not use batchnorm")
    parser_denoise.add_argument('--l2', type=float, default=0.0,
            help="L2 regularization coefficient (default: 0.0)")
    parser_denoise.add_argument('--l2enc', type=float, default=0.0,
            help="Encoder-specific L2 regularization coefficient (default: 0.0)")
    parser_denoise.add_argument('--l2out', type=float, default=0.0,
            help="Output ranch-specific L2 regularization coefficient (default: 0.0)")
    parser_denoise.add_argument('--piridge', type=float, default=0.0,
            help="L2 regularization coefficient for dropout probabilities (default: 0.0)")
    parser_denoise.add_argument('--activation', type=str, default='ReLU',
            help="Activation function of hidden units (default: ReLU)")
    parser_denoise.add_argument('--optimizer', type=str, default='RMSprop',
            help="Optimization method (default: rmsprop)")
    parser_denoise.add_argument('-e', '--epochs', type=int, default=300,
            help="Max number of epochs to continue training in case of no "
                 "improvement on validation loss (default: 200)")
    parser_denoise.add_argument('--enchiddensize', type=str, default='64,64',
            help="Size of hidden encoder layers (default: 64,64)")
    parser_denoise.add_argument('--dechiddensize', type=str, default='',
            help="Size of hidden decoder layers (default: None)")
    parser_denoise.add_argument('--outhiddensize', type=str, default='64',
            help="Size of hidden output branch layers (default: 64)")
    parser_denoise.add_argument('--encdropoutrate', type=str, default='0.0',
            help="Dropout rate for encoder layers (default: 0)")
    parser_denoise.add_argument('--decdropoutrate', type=str, default='0.0',
            help="Dropout rate for decoder layers (default: 0)")
    parser_denoise.add_argument('--outdropoutrate', type=str, default='0.0',
            help="Dropout rate for output branch layers (default: 0)")

    parser_denoise.set_defaults(func=denoise.denoise_with_args,
                              batchnorm=True)

    # test subparser
    #parser_test = subparsers.add_parser('test', help='Test autoencoder')
    #parser_test.set_defaults(func=test.test)

    # predict subparser
    #parser_predict = subparsers.add_parser('predict',
            #help='make predictions on a given dataset using a pre-trained model.')
    #parser_predict.add_argument('dataset', type=str,
            #help="File path of the input set. It must be preprocessed using "
                 #"preprocess subcommand")
    #parser_predict.add_argument('modeldir', type=str,
            #help="Path of the folder where model weights and arch are saved")
    #parser_predict.add_argument('-o', '--outputdir', type=str,
            #help="Path of the output", required=True)
    #parser_predict.add_argument('-r', '--reduced', dest='reduced',
            #action='store_true', help="predict input to the hidden size")
    #parser_predict.add_argument('--reconstruct', dest='reconstruct',
            #action='store_true', help="Save mean parameter (default: True)")
    #parser_predict.add_argument('--noreconstruct', dest='reconstruct',
            #action='store_false', help="Do not save mean parameter")
    #parser_predict.add_argument('--reduce', dest='dimreduce',
            #action='store_true', help="Save dim reduced matrix (default: True)")
    #parser_predict.add_argument('--noreduce', dest='dimreduce',
            #action='store_false', help="Do not save dim reduced matrix")

    #parser_predict.set_defaults(func=predict.predict_with_args,
                                #dimreduce=True,
                                #reconstruct=True)

    return parser.parse_args()


def main():
    args = parse_args()
    args.func(args)
