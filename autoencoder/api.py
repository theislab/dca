import os

from .io import preprocess, save_matrix
from .train import train
from .network import MLP


def autoencode(count_matrix,
               kfold=None,
               dimreduce=True,
               reconstruct=True,
               mask=None,
               type='normal',
               activation='relu',
               testset=False,
               optimizer='Adam',
               learning_rate=None,
               hidden_size=(256,64,256),
               l2_coef=0.,
               hidden_dropout=0.0,
               epochs=200,
               batch_size=32,
               init='glorot_uniform',
               output_folder=None,
               **kwargs):

    x = preprocess(count_matrix, kfold=kfold, mask=mask, testset=testset)

    net = MLP(x['shape'][1],
              hidden_size=hidden_size,
              l2_coef=l2_coef,
              hidden_dropout=hidden_dropout,
              activation=activation,
              init=init,
              masking=(mask is not None),
              loss_type=type)
    net.build()

    losses = train(x, net,
                   learning_rate=learning_rate,
                   epochs=epochs, batch_size=batch_size,
                   optimizer=optimizer, **kwargs)
    res = net.predict(count_matrix, dimreduce=dimreduce,
                      reconstruct=reconstruct, output_folder=output_folder)

    res['losses'] = losses

    return res
