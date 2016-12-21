from .network import autoencoder
from .io import read_records, load_model


def encode(input_file, log_dir, hiddensize=100):

    X = read_records(input_file)
    size = X.shape[1]

    model = load_model(log_dir)
    return model


def encode_with_args(args):
    encode(input_file = args.dataset,
           log_dir=args.logdir,
           hiddensize=args.hiddensize)
