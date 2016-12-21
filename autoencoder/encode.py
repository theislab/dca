from .network import autoencoder
from .io import read_records


def encode(inputfile, weights, hiddensize=100):

    X = read_records(inputfile)
    size = X.shape[1]

    model, encoder, _ = autoencoder(size, hidden_size=hiddensize)
    model.load_weights(weights, by_name=False)



def encode_with_args(args):
    encode(inputfile=args.dataset, weights=args.weightfile,
            hiddensize=args.hiddensize)
