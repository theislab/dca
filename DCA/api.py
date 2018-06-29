import os, tempfile, shutil, random
import anndata
import numpy as np

try:
    import tensorflow as tf
except ImportError:
    raise ImportError('DCA requires tensorflow. Please follow instructions'
                      ' at https://www.tensorflow.org/install/ to install'
                      ' it.')


from .io import read_dataset, normalize
from .train import train
from .network import AE_types


def autoencode(adata,
               mode='denoise',
               ae_type='zinb-conddisp',
               normalize_per_cell=True,
               scale=True,
               log1p=True,
               hidden_size=(64, 32, 64), # network args
               hidden_dropout=0.,
               batchnorm=True,
               activation='relu',
               init='glorot_uniform',
               epochs=300,               # training args
               reduce_lr=10,
               early_stop=15,
               batch_size=32,
               optimizer='rmsprop',
               random_state=0,
               network_kwds={},
               training_kwds={},
               verbose=False,
               return_model=False,
               copy=False
               ):
    """Deep count autoencoder(DCA) API.

    Fits a count autoencoder to the count data given in the anndata object
    in order to denoise the data and capture hidden representation of
    cells in low dimensions. Type of the autoencoder and return values are
    determined by the parameters.

    Parameters
    ----------
    adata : :class:`~scanpy.api.AnnData`
        An anndata file with `.raw` attribute representing raw counts.
    mode : `str`, optional. `denoise`(default), `latent` or `full`.
        `denoise` overwrites `adata.X` with denoised expression values.
        In `latent` mode DCA adds `adata.obsm['X_dca']` to given adata
        object. This matrix represent latent representation of cells via DCA.
        In `full` mode, `adata.X` is overwritten and all additional parameters
        of DCA are stored in `adata.obsm` such as dropout
        probabilities (obsm['X_dca_dropout']) and estimated dispersion values
        (obsm['X_dca_dispersion']), in case that autoencoder is of type
        zinb or zinb-conddisp.
    ae_type : `str`, optional. `zinb-conddisp`(default), `zinb`, `nb-conddisp` or `nb`.
        Type of the autoencoder. Return values and the architecture is
        determined by the type e.g. `nb` does not provide dropout
        probabilities.
    normalize_per_cell : `bool`, optional. Default: `True`.
        If true, library size normalization is performed using
        the `sc.pp.normalize_per_cell` function in Scanpy and saved into adata
        object. Mean layer is re-introduces library size differences by
        scaling the mean value of each cell in the output layer. See the
        manuscript for more details.
    scale : `bool`, optional. Default: `True`.
        If true, the input of the autoencoder is centered using
        `sc.pp.scale` function of Scanpy. Note that the output is kept as raw
        counts as loss functions are designed for the count data.
    log1p : `bool`, optional. Default: `True`.
        If true, the input of the autoencoder is log transformed with a
        pseudocount of one using `sc.pp.log1p` function of Scanpy.
    hidden_size : `tuple` or `list`, optional. Default: (64, 32, 64).
        Width of hidden layers.
    hidden_dropout : `float`, `tuple` or `list`, optional. Default: 0.0.
        Probability of weight dropout in the autoencoder (per layer if list
        or tuple).
    batchnorm : `bool`, optional. Default: `True`.
        If true, batch normalization is performed.
    activation : `str`, optional. Default: `relu`.
        Activation function of hidden layers.
    init : `str`, optional. Default: `glorot_uniform`.
        Initialization method used to initialize weights.
    epochs : `int`, optional. Default: 300.
        Number of total epochs in training.
    reduce_lr : `int`, optional. Default: 10.
        Reduces learning rate if validation loss does not improve in given number of epochs.
    early_stop : `int`, optional. Default: 15.
        Stops training if validation loss does not improve in given number of epochs.
    batch_size : `int`, optional. Default: 32.
        Number of samples in the batch used for SGD.
    optimizer : `str`, optional. Default: "rmsprop".
        Type of optimization method used for training.
    random_state : `int`, optional. Default: 0.
        Seed for python, numpy and tensorflow.
    network_kwds : `dict`, optional.
        Additional keyword arguments for the autoencoder.
    training_kwds : `dict`, optional.
        Additional keyword arguments for the training process.
    verbose : `bool`, optional. Default: `False`.
        If true, prints additional information about training and architecture.
    return_model : `bool`, optional. Default: `False.
        If true, trained autoencoder object is returned.
    copy : `bool`, optional. Default: `False`.
        If true, a copy of anndata is returned.

    Returns
    =======

    If copy is true, AnnData object. In "denoise" mode, adata.X is overwritten with the main
    output of the autoencoder. In "latent" mode, latent low dimensional representation of cells are
    stored in adata.obsm['X_dca'] and adata.X is not modified. In "full" mode, both adata.X is
    overwritten and latent representation is added to adata.obsm['X_dca']. In addition, other
    estimated distribution parameters are stored in adata.obsm such as adata.obsm['X_dca_dropout']
    which represents the mixture coefficient (pi) of the zero component in ZINB hence dropout
    probability. Raw counts are stored as `adata.raw`.

    If return_model is given, trained model is returned. When both copy and return_model are true
    a tuple of anndata and model is returned in that order.

    """

    assert isinstance(adata, anndata.AnnData), 'adata must be an AnnData instance'
    assert mode in ('denoise', 'latent', 'full'), '%s is not a valid mode.' % mode

    # set seed for reproducibility
    random.seed(random_state)
    np.random.seed(random_state)
    tf.set_random_seed(random_state)
    os.environ['PYTHONHASHSEED'] = '0'

    adata = read_dataset(adata,
                         transpose=False,
                         test_split=False,
                         copy=copy)

    adata = normalize(adata,
                      filter_min_counts=False, # no filtering, keep cell and gene idxs same
                      size_factors=normalize_per_cell,
                      normalize_input=scale,
                      logtrans_input=log1p)

    network_kwds = {**network_kwds,
        'hidden_size': hidden_size,
        'hidden_dropout': hidden_dropout,
        'batchnorm': batchnorm,
        'activation': activation,
        'init': init
    }

    input_size = output_size = adata.n_vars
    net = AE_types[ae_type](input_size=input_size,
                            output_size=output_size,
                            **network_kwds)
    net.save()
    net.build()

    training_kwds = {**training_kwds,
        'epochs': epochs,
        'reduce_lr': reduce_lr,
        'early_stop': early_stop,
        'batch_size': batch_size,
        'optimizer': optimizer,
        'verbose': verbose
    }

    losses = train(adata[adata.obs.DCA_split == 'train'], net, **training_kwds)
    res = net.predict(adata)

    #TODO: move this part to net.predict() code
    if mode in ('denoise', 'full'):
        adata.X = res['mean']

    if mode in ('latent', 'full'):
        adata.obsm['X_dca'] = res['reduced']

    if mode == 'full':
        if 'pi' in res:
            if res['pi'].ndim > 1:
                adata.obsm['X_dca_dropout'] = res['pi']
            else: # non-conditional case
                adata.var['X_dca_dropout'] = res['pi']

        if 'dispersion' in res:
            if res['dispersion'].ndim > 1:
                adata.obsm['X_dca_dispersion'] = res['dispersion']
            else:
                adata.var['X_dca_dispersion'] = res['dispersion']

        adata.uns['dca_loss_history'] = losses

    if return_model:
        return (adata, net) if copy else net
    else:
        return adata if copy else None
