import os, tempfile, shutil
import anndata
import numpy as np
import tensorflow as tf

from .io import read_dataset, normalize
from .train import train
from .network import AE_types


def autoencode(adata,
               aetype='zinb-conddisp',
               size_factors=True,
               normalize_input=True,
               logtrans_input=True,
               test_split=False,
               net_kwargs={},
               training_kwargs={},
               copy=False):

    assert isinstance(adata, anndata.AnnData), 'adata must be an AnnData instance'

    # set seed for reproducibility
    np.random.seed(42)
    tf.set_random_seed(42)

    adata = read_dataset(adata,
                         transpose=False,
                         test_split=test_split,
                         copy=copy)

    adata = normalize(adata,
                      size_factors=size_factors,
                      normalize_input=normalize_input,
                      logtrans_input=logtrans_input)

    input_size = output_size = adata.n_vars
    net = AE_types[aetype](input_size=input_size,
                           output_size=output_size,
                           **net_kwargs)
    net.save()
    net.build()

    losses = train(adata[adata.obs.DCA_split == 'train'], net, **training_kwargs)
    res = net.predict(adata)

    adata.obsm['X_dca'] = res['mean_norm']
    adata.obsm['X_dca_mean'] = res['mean']
    adata.obsm['X_dca_hidden'] = res['reduced']

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

    adata.uns['DCA_losses'] = losses

    return adata