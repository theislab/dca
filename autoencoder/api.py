from .io import preprocess
from .train import train
from .network import mlp


def autoencode(count_matrix, kfold=None, dimreduce=True, reconstruct=True,
               mask=None, type='normal', activation='relu', testset=False,
               learning_rate=1e-2, hidden_size=(256,64,256), l2_coef=0.,
               hidden_dropout=0.1, epochs=200, batch_size=32,
               optimizer=None, **kwargs):

    x = preprocess(count_matrix, kfold=kfold, mask=mask, testset=testset)

    net = mlp(x['shape'][1],
              hidden_size=hidden_size,
              l2_coef=l2_coef,
              hidden_dropout=hidden_dropout,
              activation=activation,
              masking=(mask is not None),
              loss_type=type)

    model, encoder, decoder, loss, extras = net['model'], net['encoder'], \
                                            net['decoder'], net['loss'], \
                                            net['extra_models']

    losses = train(x, model, loss,
                   learning_rate=learning_rate,
                   epochs=epochs, batch_size=batch_size,
                   optimizer=optimizer, **kwargs)

    ret = {'model':   model,
           'encoder': encoder,
           'decoder': decoder,
           'extra_models': extras,
           'losses':  losses}

    if dimreduce:
        ret['reduced'] = encoder.predict(count_matrix)
    if reconstruct:
        ret['reconstructed'] = model.predict(count_matrix)

    return ret
