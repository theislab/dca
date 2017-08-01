import pickle, os
import numpy as np

from .io import Dataset

def predict_with_args(args):

    ds = Dataset(args.dataset)

    modelfile = os.path.join(args.modeldir, 'model.pickle')
    weightfile = os.path.join(args.modeldir, 'weights.hdf5')

    # load serialized model
    net = pickle.load(open(modelfile, 'rb'))
    net.build()
    net.load_weights(weightfile)
    net.file_path = args.outputdir
    net.predict(ds.matrix[:], dimreduce=args.dimreduce,
                reconstruct=args.reconstruct)
