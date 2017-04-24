import pickle, os
import numpy as np

from .io import read_from_file


def predict(data, model, output_file=None, reduced=False):
    size = data.shape[1]

    assert model.input_shape[1] == size, \
           'Input size of data and pretrained model must be same'

    if reduced:
        #FIXME
        encoder = get_encoder(model)
        predictions = encoder.predict(data)
    else:
        predictions = model.predict(data)

    if output_file:
        np.savetxt(output_file, predictions)

    return predictions


def predict_with_args(args):

    x = read_from_file(args.trainingset)

    modelfile = os.path.join(args.modeldir, 'model.pickle')
    weightfile = os.path.join(args.modeldir, 'weights.hdf5')

    # load serialized model
    net = pickle.load(open(modelfile, 'rb'))
    net.build()
    net.model.load_weights(weightfile)
    net.file_path = args.modeldir
    net.predict(x['full'], dimreduce=args.dimreduce,
                reconstruct=args.reconstruct)
