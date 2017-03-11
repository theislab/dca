import numpy as np

from .network import autoencoder, get_encoder
from .io import read_csv, load_model


def encode(data, model, output_file):
    size = data.shape[1]

    assert model.input_shape[1] == size, \
           'Input size of data and pretrained model must be same'

    encoder = get_encoder(model)
    predictions = encoder.predict(X)
    np.savetxt(output_file, predictions)


def encode_with_args(args):
    X = read_csv(args.dataset)
    if args.transpose:
        X = X.transpose()
    model = load_model(args.logdir)
    encode(X, model, args.outputfile)
