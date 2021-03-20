import os, tempfile, shutil, random
import anndata
import numpy as np
import scanpy as sc

try:
    import tensorflow as tf
except ImportError:
    raise ImportError('DCA requires TensorFlow v2+. Please follow instructions'
                      ' at https://www.tensorflow.org/install/ to install'
                      ' it.')


from .io import read_dataset, normalize
from .train import train
from .network import AE_types


def dca(adata,
        mode='denoise',
        ae_type='nb-conddisp',
        normalize_per_cell=True,
        scale=True,
        log1p=True,
        hidden_size=(64, 32, 64), # network args
        hidden_dropout=0.,
        batchnorm=True,
        activation='relu',
        init='glorot_uniform',
        network_kwds={},
        epochs=300,               # training args
        reduce_lr=10,
        early_stop=15,
        batch_size=32,
        optimizer='RMSprop',
        learning_rate=None,
        random_state=0,
        threads=None,
        verbose=False,
        training_kwds={},
        return_model=False,
        return_info=False,
        copy=False,
        check_counts=True,
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
    mode : `str`, optional. `denoise`(default), or `latent`.
        `denoise` overwrites `adata.X` with denoised expression values.
        In `latent` mode DCA adds `adata.obsm['X_dca']` to given adata
        object. This matrix represent latent representation of cells via DCA.
    ae_type : `str`, optional. `nb-conddisp`(default), `zinb`, `nb-conddisp` or `nb`.
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
    network_kwds : `dict`, optional.
        Additional keyword arguments for the autoencoder.
    epochs : `int`, optional. Default: 300.
        Number of total epochs in training.
    reduce_lr : `int`, optional. Default: 10.
        Reduces learning rate if validation loss does not improve in given number of epochs.
    early_stop : `int`, optional. Default: 15.
        Stops training if validation loss does not improve in given number of epochs.
    batch_size : `int`, optional. Default: 32.
        Number of samples in the batch used for SGD.
    learning_rate : `float`, optional. Default: None.
        Learning rate to use in the training.
    optimizer : `str`, optional. Default: "RMSprop".
        Type of optimization method used for training.
    random_state : `int`, optional. Default: 0.
        Seed for python, numpy and tensorflow.
    threads : `int` or None, optional. Default: None
        Number of threads to use in training. All cores are used by default.
    verbose : `bool`, optional. Default: `False`.
        If true, prints additional information about training and architecture.
    training_kwds : `dict`, optional.
        Additional keyword arguments for the training process.
    return_model : `bool`, optional. Default: `False`.
        If true, trained autoencoder object is returned. See "Returns".
    return_info : `bool`, optional. Default: `False`.
        If true, all additional parameters of DCA are stored in `adata.obsm` such as dropout
        probabilities (obsm['X_dca_dropout']) and estimated dispersion values
        (obsm['X_dca_dispersion']), in case that autoencoder is of type
        zinb or zinb-conddisp.
    copy : `bool`, optional. Default: `False`.
        If true, a copy of anndata is returned.
    check_counts : `bool`. Default `True`.
        Check if the counts are unnormalized (raw) counts.

    Returns
    -------
    If `copy` is true and `return_model` is false, AnnData object is returned.

    In "denoise" mode, `adata.X` is overwritten with the denoised values. In "latent" mode, latent
    low dimensional representation of cells are stored in `adata.obsm['X_dca']` and `adata.X`
    is not modified. Note that these values are not corrected for library size effects.

    If `return_info` is true, all estimated distribution parameters are stored in AnnData such as:

    - `.obsm["X_dca_dropout"]` which is the mixture coefficient (pi) of the zero component
    in ZINB, i.e. dropout probability. (Only if ae_type is zinb or zinb-conddisp)

    - `.obsm["X_dca_dispersion"]` which is the dispersion parameter of NB.

    - `.uns["dca_loss_history"]` which stores the loss history of the training.

    Finally, the raw counts are stored as `.raw`.

    If `return_model` is given, trained model is returned. When both `copy` and `return_model`
    are true, a tuple of anndata and model is returned in that order.
    """

    assert isinstance(adata, anndata.AnnData), 'adata must be an AnnData instance'
    assert mode in ('denoise', 'latent'), '%s is not a valid mode.' % mode

    # set seed for reproducibility
    random.seed(random_state)
    np.random.seed(random_state)
    tf.random.set_seed(random_state)
    os.environ['PYTHONHASHSEED'] = '0'

    # this creates adata.raw with raw counts and copies adata if copy==True
    adata = read_dataset(adata,
                         transpose=False,
                         test_split=False,
                         copy=copy,
                         check_counts=check_counts)

    # check for zero genes
    nonzero_genes, _ = sc.pp.filter_genes(adata.X, min_counts=1)
    assert nonzero_genes.all(), 'Please remove all-zero genes before using DCA.'

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
    
    from tensorflow.python.framework.ops import disable_eager_execution
    disable_eager_execution()

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
        'verbose': verbose,
        'threads': threads,
        'learning_rate': learning_rate
    }

    hist = train(adata[adata.obs.dca_split == 'train'], net, **training_kwds)
    res = net.predict(adata, mode, return_info, copy)
    adata = res if copy else adata

    if return_info:
        adata.uns['dca_loss_history'] = hist.history

    if return_model:
        return (adata, net) if copy else net
    else:
        return adata if copy else None
