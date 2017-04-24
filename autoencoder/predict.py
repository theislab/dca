import pickle, os
import numpy as np

from .io import read_from_file

def predict_with_args(args):

    x = read_from_file(args.dataset)

    modelfile = os.path.join(args.modeldir, 'model.pickle')
    weightfile = os.path.join(args.modeldir, 'weights.hdf5')

    # load serialized model
    net = pickle.load(open(modelfile, 'rb'))
    net.build()
    net.load_weights(weightfile)
    net.file_path = args.outputdir
    net.predict(x['full'], dimreduce=args.dimreduce,
                reconstruct=args.reconstruct)
