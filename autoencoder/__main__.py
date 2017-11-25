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

import os, sys, argparse
from . import train

def parse_args():
    parser = argparse.ArgumentParser(description='Autoencoder')

    parser.add_argument('input', type=str, help='Input in TSV/CSV format. '
                        'Row/col names are mandatory')
    parser.add_argument('outputdir', type=str, help='The path of the output directory')

    # IO and norm options
    parser.add_argument('--normtype', type=str, default='zheng',
            help='Type of size factor estimation. Possible values: deseq, zheng.'
                 ' (default: zheng)')
    parser.add_argument('-t', '--transpose', dest='transpose',
            action='store_true', help='Transpose input matrix (default: False)')
    parser.add_argument('--testsplit', dest='testsplit',
            action='store_true', help="Use one fold as a test set (default: False)")

    # training options
    parser.add_argument('--type', type=str, default='zinb-conddisp',
            help="Type of autoencoder. Possible values: normal, poisson, nb, "
                 "nb-shared, nb-conddisp, nb-fork, zinb, "
                 "zinb-shared, zinb-conddisp(default) zinb-fork")
    parser.add_argument('-b', '--batchsize', type=int, default=32,
            help="Batch size (default:32)")
    parser.add_argument('--sizefactors', dest='sizefactors',
            action='store_true', help="Normalize means by library size (default: True)")
    parser.add_argument('--nosizefactors', dest='sizefactors',
            action='store_false', help="Do not normalize means by library size")
    parser.add_argument('--norminput', dest='norminput',
            action='store_true', help="Zero-mean normalize input (default: True)")
    parser.add_argument('--nonorminput', dest='norminput',
            action='store_false', help="Do not zero-mean normalize inputs")
    parser.add_argument('--loginput', dest='loginput',
            action='store_true', help="Log-transform input (default: True)")
    parser.add_argument('--nologinput', dest='loginput',
            action='store_false', help="Do not log-transform inputs")
    parser.add_argument('-d', '--dropoutrate', type=str, default='0.0',
            help="Dropout rate (default: 0)")
    parser.add_argument('--batchnorm', dest='batchnorm', action='store_true',
            help="Batchnorm (default: True)")
    parser.add_argument('--nobatchnorm', dest='batchnorm', action='store_false',
            help="Do not use batchnorm")
    parser.add_argument('--l2', type=float, default=0.0,
            help="L2 regularization coefficient (default: 0.0)")
    parser.add_argument('--l1', type=float, default=0.0,
            help="L1 regularization coefficient (default: 0.0)")
    parser.add_argument('--l2enc', type=float, default=0.0,
            help="Encoder-specific L2 regularization coefficient (default: 0.0)")
    parser.add_argument('--l1enc', type=float, default=0.0,
            help="Encoder-specific L1 regularization coefficient (default: 0.0)")
    parser.add_argument('--ridge', type=float, default=0.0,
            help="L2 regularization coefficient for dropout probabilities (default: 0.0)")
    parser.add_argument('--gradclip', type=float, default=5.0,
            help="Clip grad values (default: 5.0)")
    parser.add_argument('--activation', type=str, default='elu',
            help="Activation function of hidden units (default: elu)")
    parser.add_argument('--optimizer', type=str, default='rmsprop',
            help="Optimization method (default: rmsprop)")
    parser.add_argument('--init', type=str, default='glorot_uniform',
            help="Initialization method for weights (default: glorot_uniform)")
    parser.add_argument('-e', '--epochs', type=int, default=300,
            help="Max number of epochs to continue training in case of no "
                 "improvement on validation loss (default: 300)")
    parser.add_argument('--earlystop', type=int, default=15,
            help="Number of epochs to stop training if no improvement in loss "
                 "occurs (default: 15)")
    parser.add_argument('--reducelr', type=int, default=10,
            help="Number of epochs to reduce learning rate if no improvement "
            "in loss occurs (default: 10)")
    parser.add_argument('-s', '--hiddensize', type=str, default='64,32,64',
            help="Size of hidden layers (default: 64,32,64)")
    parser.add_argument('-r', '--learningrate', type=float, default=None,
            help="Learning rate (default: 0.001)")
    parser.add_argument('--saveweights', dest='saveweights',
            action='store_true', help="Save weights (default: False)")
    parser.add_argument('--no-saveweights', dest='saveweights',
            action='store_false', help="Do not save weights")

    parser.set_defaults(func=train.train_with_args,
                              saveweights=False,
                              sizefactors=True,
                              batchnorm=True,
                              norminput=True,
                              loginput=True)

    return parser.parse_args()


def main():
    args = parse_args()
    args.func(args)
