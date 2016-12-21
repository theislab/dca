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
import tensorflow as tf
slim = tf.contrib.slim

from . import io, train, test, encode


def parse_args():
    parser = argparse.ArgumentParser(description='Autoencoder')
    subparsers = parser.add_subparsers(title='subcommands',
            help='sub-command help', description='valid subcommands', dest='cmd')
    subparsers.required = True

    # Preprocess subparser
    parser_preprocess = subparsers.add_parser('preprocess',
            help='Create a training set from CSV/TSV files')
    parser_preprocess.set_defaults(func=io.preprocess)
    parser_preprocess.add_argument('input', type=str,
            help='Input in TSV/CSV format')
    parser_preprocess.add_argument('-o', '--output', type=str,
            help='Output file path', required=True)
    parser_preprocess.add_argument('-k', '--kfold', type=int,
            help='k-fold CV')
    parser_preprocess.add_argument('-t', '--transpose', dest='transpose',
            action='store_true', help='Transpose input matrix')

    # train subparser
    parser_train = subparsers.add_parser('train',
            help='Train an autoencoder using given training set.')
    parser_train.add_argument('trainingset', type=str,
            help="File path of the training set ")
    parser_train.add_argument('-l', '--logdir', type=str, default='logs',
            help="The directory where training logs will be saved (default=logs)")
    parser_train.add_argument('-b', '--batchsize', type=int, default=128,
            help="Batch size")
    parser_train.add_argument('--dropoutrate', type=float, default=0.0,
            help="Dropout rate")
    parser_train.add_argument('-e', '--epochs', type=int, default=100,
            help="Max number of epochs to continue training in case of no "
                 "improvement on validation loss")
    parser_train.add_argument('--checkpoint_every', type=int, default=100,
            help="How many batches to save checkpoints after")
    parser_train.add_argument('-s', '--hiddensize', type=int, default=100,
            help="Size of hidden layers")
    parser_train.add_argument('-r', '--learningrate', type=float, default=1e-4,
            help="Learning rate")

    parser_train.set_defaults(func=train.train_with_args)

    # test subparser
    parser_test = subparsers.add_parser('test',
            help='Test autoencoder on mnist')
    parser_test.set_defaults(func=test.test)

    # encode subparser
    parser_encode = subparsers.add_parser('encode',
            help='Encode a given dataset using a pre-trained model.')
    parser_encode.add_argument('dataset', type=str,
            help="File path of the dataset ")
    parser_encode.add_argument('logdir', type=str,
            help="File path of the model weights ")
    parser_encode.add_argument('-s', '--hiddensize', type=int, default=100,
            help="Size of hidden layers")
    parser_encode.set_defaults(func=encode.encode_with_args)

    return parser.parse_args()


def main():
    args = parse_args()
    args.func(args)
