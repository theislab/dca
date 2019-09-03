import numpy as np
import scanpy as sc

from .api import dca

def test_api():
    adata = sc.datasets.paul15()
    epochs = 1

    # simple tests for denoise
    ret = dca(adata, mode='denoise', copy=True, epochs=epochs, verbose=True)
    assert not np.allclose(ret.X[:10], adata.X[:10])

    ret, _ = dca(adata, mode='denoise', ae_type='nb-conddisp', copy=True, epochs=epochs,
              return_model=True, return_info=True)
    assert not np.allclose(ret.X[:10], adata.X[:10])
    assert 'X_dca_dispersion' in ret.obsm_keys()
    assert _ is not None

    ret = dca(adata, mode='denoise', ae_type='nb', copy=True, epochs=epochs,
              return_model=False, return_info=True)
    assert not np.allclose(ret.X[:10], adata.X[:10])
    assert 'X_dca_dispersion' in ret.var_keys()

    ret = dca(adata, mode='denoise', ae_type='zinb', copy=True, epochs=epochs,
              return_model=False, return_info=True)
    assert not np.allclose(ret.X[:10], adata.X[:10])
    assert 'X_dca_dropout' in ret.obsm_keys()
    assert 'dca_loss_history' in ret.uns_keys()

    ret = dca(adata, mode='denoise', ae_type='zinb-elempi', copy=True, epochs=epochs,
              return_model=False, return_info=True)
    assert not np.allclose(ret.X[:10], adata.X[:10])
    assert 'X_dca_dropout' in ret.obsm_keys()
    assert 'dca_loss_history' in ret.uns_keys()

    ret = dca(adata, mode='denoise', ae_type='zinb-elempi', copy=True, epochs=epochs,
              return_model=False, return_info=True, network_kwds={'sharedpi': True})
    assert not np.allclose(ret.X[:10], adata.X[:10])
    assert 'X_dca_dropout' in ret.obsm_keys()
    assert 'dca_loss_history' in ret.uns_keys()

    # simple tests for latent
    hid_size = (10, 2, 10)
    ret = dca(adata, mode='latent', hidden_size=hid_size, copy=True, epochs=epochs)
    assert 'X_dca' in ret.obsm_keys()
    assert ret.obsm['X_dca'].shape[1] == hid_size[1]

    ret = dca(adata, mode='latent', ae_type='nb-conddisp', hidden_size=hid_size, copy=True, epochs=epochs)
    assert 'X_dca' in ret.obsm_keys()
    assert ret.obsm['X_dca'].shape[1] == hid_size[1]

    ret = dca(adata, mode='latent', ae_type='nb', hidden_size=hid_size, copy=True, epochs=epochs, return_info=True)
    assert 'X_dca' in ret.obsm_keys()
    assert ret.obsm['X_dca'].shape[1] == hid_size[1]

    ret = dca(adata, mode='latent', ae_type='zinb', hidden_size=hid_size, copy=True, epochs=epochs)
    assert 'X_dca' in ret.obsm_keys()
    assert ret.obsm['X_dca'].shape[1] == hid_size[1]
