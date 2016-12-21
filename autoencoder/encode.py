import numpy as np

from .network import autoencoder, get_encoder
from .io import read_records, load_model


def encode(input_file, output_file, log_dir):
    X = read_records(input_file)
    size = X.shape[1]

    model = load_model(log_dir)
    assert model.input_shape[1] == size, \
           'Input size of data and pretrained model must be same'


    encoder = get_encoder(model)
    predictions = encoder.predict(X)
    np.savetxt(output_file, predictions)


def encode_with_args(args):
    encode(input_file = args.dataset,
           output_file = args.outputfile,
           log_dir = args.logdir)
