from .io import preprocess
from .train import train
from .encode import encode


def autoencode(count_matrix, kfold=None, reduced=False,
               censor_matrix=None, type='normal',
               learning_rate=1e-2,
               hidden_size=10,
               epochs=10):

    x = preprocess(count_matrix, kfold=kfold, censor=censor_matrix)
    model = train(x, hidden_size=hidden_size, learning_rate=learning_rate,
                  aetype=type, epochs=epochs)
    encoded = encode(count_matrix, model, reduced=reduced)

    return encoded
