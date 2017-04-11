from .io import preprocess
from .train import train
from .network import mlp
from .encode import encode


def autoencode(count_matrix, kfold=None, dimreduce=True, reconstruct=True,
               mask=None, type='normal', activation='relu', testset=False,
               learning_rate=1e-2, hidden_size=(256,64,256), l2_coef=0.,
               epochs=200, **kwargs):

    x = preprocess(count_matrix, kfold=kfold, mask=mask, testset=testset)

    net = mlp(x['shape'][1],
              hidden_size=hidden_size,
              l2_coef=l2_coef,
              activation=activation,
              masking=(mask is not None),
              loss_type=type)

    model, encoder, decoder, loss, extras = net['model'], net['encoder'], \
                                            net['decoder'], net['loss'], \
                                            net['extra_models']

    losses = train(x, model, loss,
                   learning_rate=learning_rate,
                   epochs=epochs, **kwargs)

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
