import os

from .io import preprocess, save_matrix
from .train import train
from .network import mlp


def autoencode(count_matrix,
               kfold=None,
               dimreduce=True,
               reconstruct=True,
               mask=None,
               type='normal',
               activation='relu',
               testset=False,
               learning_rate=1e-2,
               hidden_size=(256,64,256),
               l2_coef=0.,
               hidden_dropout=0.1,
               epochs=200,
               batch_size=32,
               optimizer=None,
               init='glorot_uniform',
               output_folder=None,
               **kwargs):

    x = preprocess(count_matrix, kfold=kfold, mask=mask, testset=testset)

    net = mlp(x['shape'][1],
              hidden_size=hidden_size,
              l2_coef=l2_coef,
              hidden_dropout=hidden_dropout,
              activation=activation,
              init=init,
              masking=(mask is not None),
              loss_type=type)

    losses = train(x, net['model'], net['loss'],
                   learning_rate=learning_rate,
                   epochs=epochs, batch_size=batch_size,
                   optimizer=optimizer, **kwargs)

    net['losses'] = losses

    if dimreduce:
        net['reduced'] = net['encoder'].predict(count_matrix)
        net['reduced_linear'] = net['encoder_linear'].predict(count_matrix)
        if 'dispersion' in net['extra_models']:
            net['dispersion'] = net['extra_models']['dispersion']()
        if 'pi' in net['extra_models']:
            net['pi'] = net['extra_models']['pi'].predict(count_matrix)

        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
            save_matrix(net['reduced'], os.path.join(output_folder,
                                                     'reduced.tsv'))
            save_matrix(net['reduced_linear'], os.path.join(output_folder,
                                                     'reduced_linear.tsv'))
            if 'dispersion' in net:
                save_matrix(net['dispersion'], os.path.join(output_folder,
                                                            'dispersion.tsv'))
            if 'pi' in net:
                save_matrix(net['pi'], os.path.join(output_folder,
                                                    'pi.tsv'))

    if reconstruct:
        net['mean'] = net['model'].predict(count_matrix)
        if output_folder:
            save_matrix(net['mean'], os.path.join(output_folder, 'mean.tsv'))

    return net
