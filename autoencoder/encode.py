import numpy as np

from .network import autoencoder, get_encoder
from .io import read_csv, load_model


def encode(data, model, output_file=None, reduced=False):
    size = data.shape[1]

    assert model.input_shape[1] == size, \
           'Input size of data and pretrained model must be same'

    if reduced:
        encoder = get_encoder(model)
        predictions = encoder.predict(data)
    else:
        predictions = model.predict(data)

    if output_file:
        np.savetxt(output_file, predictions)

    return predictions


def encode_with_args(args):
    X = read_csv(args.dataset)
    if args.transpose:
        X = X.transpose()
    model = load_model(args.logdir)
    encode(X, model, args.outputfile, reduced=args.reduced)
