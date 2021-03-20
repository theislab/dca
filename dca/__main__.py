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

def parse_args():
    parser = argparse.ArgumentParser(description='Autoencoder')

    parser.add_argument('input', type=str, help='Input is raw count data in TSV/CSV '
                        'or H5AD (anndata) format. '
                        'Row/col names are mandatory. Note that TSV/CSV files must be in '
                        'gene x cell layout where rows are genes and cols are cells (scRNA-seq '
                        'convention).'
                        'Use the -t/--transpose option if your count matrix in cell x gene layout. '
                        'H5AD files must be in cell x gene format (stats and scanpy convention).')
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
    parser.add_argument('--type', type=str, default='nb-conddisp',
            help="Type of autoencoder. Possible values: normal, poisson, nb, "
                 "nb-shared, nb-conddisp (default), nb-fork, zinb, "
                 "zinb-shared, zinb-conddisp( zinb-fork")
    parser.add_argument('--threads', type=int, default=None,
            help='Number of threads for training (default is all cores)')
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
    parser.add_argument('--activation', type=str, default='relu',
            help="Activation function of hidden units (default: relu)")
    parser.add_argument('--optimizer', type=str, default='RMSprop',
            help="Optimization method (default: RMSprop)")
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
    parser.add_argument('--inputdropout', type=float, default=0.0,
            help="Input layer dropout probability"),
    parser.add_argument('-r', '--learningrate', type=float, default=None,
            help="Learning rate (default: 0.001)")
    parser.add_argument('--saveweights', dest='saveweights',
            action='store_true', help="Save weights (default: False)")
    parser.add_argument('--no-saveweights', dest='saveweights',
            action='store_false', help="Do not save weights")
    parser.add_argument('--hyper', dest='hyper',
            action='store_true', help="Optimizer hyperparameters (default: False)")
    parser.add_argument('--hypern', dest='hypern', type=int, default=1000,
            help="Number of samples drawn from hyperparameter distributions during optimization. "
                 "(default: 1000)")
    parser.add_argument('--hyperepoch', dest='hyperepoch', type=int, default=100,
            help="Number of epochs used in each hyperpar optimization iteration. "
                 "(default: 100)")
    parser.add_argument('--debug', dest='debug',
            action='store_true', help="Enable debugging. Checks whether every term in "
                                      "loss functions is finite. (default: False)")
    parser.add_argument('--tensorboard', dest='tensorboard',
            action='store_true', help="Use tensorboard for saving weight distributions and "
                                      "visualization. (default: False)")
    parser.add_argument('--checkcounts', dest='checkcounts', action='store_true',
            help="Check if the expression matrix has raw (unnormalized) counts (default: True)")
    parser.add_argument('--nocheckcounts', dest='checkcounts', action='store_false',
            help="Do not check if the expression matrix has raw (unnormalized) counts")
    parser.add_argument('--denoisesubset', dest='denoisesubset', type=str,
                        help='Perform denoising only for the subset of genes '
                             'in the given file. Gene names should be line '
                             'separated.')

    parser.set_defaults(transpose=False,
                        testsplit=False,
                        saveweights=False,
                        sizefactors=True,
                        batchnorm=True,
                        checkcounts=True,
                        norminput=True,
                        hyper=False,
                        debug=False,
                        tensorboard=False,
                        loginput=True)

    return parser.parse_args()


def main():
    args = parse_args()

    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError('DCA requires TensorFlow v2+. Please follow instructions'
                          ' at https://www.tensorflow.org/install/ to install'
                          ' it.')

    # import tf and the rest after parse_args() to make argparse help faster
    from . import train

    train.train_with_args(args)
