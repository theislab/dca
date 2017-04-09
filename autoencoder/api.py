from .io import preprocess
from .train import train
from .encode import encode


def autoencode(count_matrix, kfold=None, reduced=False,
               mask=None, type='normal',
               learning_rate=1e-2, hidden_size=10, l2_coef=0.,
               epochs=200, **kwargs):

    x = preprocess(count_matrix, kfold=kfold, mask=mask, testset=False)
    model, losses = train(x, hidden_size=hidden_size, l2_coef=l2_coef,
                          learning_rate=learning_rate, aetype=type,
                          epochs=epochs, **kwargs)
    encoded = encode(count_matrix, model, reduced=reduced)

    return {'encoded': encoded,
            'model':   model,
            'losses':  losses}
