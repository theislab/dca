import os
import pickle
import json

import numpy as np
from kopt import CompileFN, test_fn
from hyperopt import fmin, tpe, hp, Trials
import keras.optimizers as opt

from . import io
from .network import AE_types


def hyper(args):
    ds = io.create_dataset(args.input,
                           output_file=os.path.join(args.outputdir, 'input.zarr'),
                           transpose=args.transpose,
                           test_split=args.testsplit,
                           size_factors=args.normtype)

    hyper_params = {
            "data": {
                "norm_input_log": hp.choice('d_norm_log', (True, False)),
                "norm_input_zeromean": hp.choice('d_norm_zeromean', (True, False)),
                "norm_input_sf": hp.choice('d_norm_sf', (True, False)),
                },
            "model": {
                "lr": hp.loguniform("m_lr", np.log(1e-3), np.log(1e-2)),
                "ridge": hp.loguniform("m_ridge", np.log(1e-7), np.log(1e-1)),
                "l1_enc_coef": hp.loguniform("m_l1_enc_coef", np.log(1e-7), np.log(1e-1)),
                "hidden_size": hp.choice("m_hiddensize", ((64,32,64), (32,16,32),
                                                          (64,64), (32,32), (16,16),
                                                          (16,), (32,), (64,), (128,))),
                "activation": hp.choice("m_activation", ('relu', 'selu', 'elu',
                                                         'PReLU', 'linear', 'LeakyReLU')),
                "aetype": hp.choice("m_aetype", ('zinb', 'zinb-conddisp')),
                "batchnorm": hp.choice("m_batchnorm", (True, False)),
                "dropout": hp.uniform("m_do", 0, 0.7),
                "input_dropout": hp.uniform("m_input_do", 0, 0.8),
                },
            "fit": {
                "epochs": 100
                }
    }

    def data_fn(norm_input_log, norm_input_zeromean, norm_input_sf):
        if norm_input_sf:
            sf_mat = ds.train.size_factors[:]
        else:
            sf_mat = np.ones((ds.train.matrix.shape[0], 1),
                             dtype=np.float32)

        x_train = {'count': io.normalize(ds.train.matrix[:],
                                         sf_mat, logtrans=norm_input_log,
                                         sfnorm=norm_input_sf,
                                         zeromean=norm_input_zeromean),
                    'size_factors': sf_mat}
        y_train = ds.train.matrix[:]

        return (x_train, y_train),

    def model_fn(train_data, lr, hidden_size, activation, aetype, batchnorm,
                 dropout, input_dropout, ridge, l1_enc_coef):
        net = AE_types[aetype](train_data[1].shape[1],
                hidden_size=hidden_size,
                l2_coef=0.0,
                l1_coef=0.0,
                l2_enc_coef=0.0,
                l1_enc_coef=l1_enc_coef,
                ridge=ridge,
                hidden_dropout=dropout,
                input_dropout=input_dropout,
                batchnorm=batchnorm,
                activation=activation,
                init='glorot_uniform',
                debug=args.debug)
        net.build()

        optimizer = opt.__dict__['rmsprop'](lr=lr, clipvalue=5.0)
        net.model.compile(loss=net.loss, optimizer=optimizer)

        return net.model

    output_dir = os.path.join(args.outputdir, 'hyperopt_results')
    objective = CompileFN('autoencoder_hyperpar_db', 'myexp1',
                          data_fn=data_fn,
                          model_fn=model_fn,
                          loss_metric='loss',
                          loss_metric_mode='min',
                          valid_split=.2,
                          save_model=None,
                          save_results=True,
                          use_tensorboard=False,
                          save_dir=output_dir)

    test_fn(objective, hyper_params, save_model=None)

    trials = Trials()
    best = fmin(objective,
                hyper_params,
                trials=trials,
                algo=tpe.suggest,
                max_evals=args.hypern,
                catch_eval_exceptions=True)

    with open(os.path.join(output_dir, 'trials.pickle'), 'wb') as f:
        pickle.dump(trials, f)

    #TODO: map indices in "best" back to choice-based hyperpars before saving
    with open(os.path.join(output_dir, 'best.json'), 'wt') as f:
        json.dump(best, f, sort_keys=True, indent=4)

    print(best)

    #TODO: not just save the best conf but also train the model with these params
