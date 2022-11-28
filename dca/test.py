import numpy as np
import scanpy as sc

from unittest import TestCase

from .api import dca

class TestAPI(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.adata = sc.datasets.paul15()
        cls.epochs = 1
        cls.hid_size = (10, 2, 10)

    def test_denoise_simple(self):
        ret = dca(self.adata, mode='denoise', copy=True, epochs=self.epochs, verbose=True)
        assert not np.allclose(ret.X[:10], self.adata.X[:10])

    def test_denoise_nb_conddisp(self):
        ret, _ = dca(self.adata, mode='denoise', ae_type='nb-conddisp', copy=True, epochs=self.epochs,
                  return_model=True, return_info=True)
        assert not np.allclose(ret.X[:10], self.adata.X[:10])
        assert 'X_dca_dispersion' in ret.obsm_keys()
        assert _ is not None

    def test_denoise_nb(self):
        ret = dca(self.adata, mode='denoise', ae_type='nb', copy=True, epochs=self.epochs,
                  return_model=False, return_info=True)
        assert not np.allclose(ret.X[:10], self.adata.X[:10])
        assert 'X_dca_dispersion' in ret.var_keys()

    def test_denoise_zinb(self):
        ret = dca(self.adata, mode='denoise', ae_type='zinb', copy=True, epochs=self.epochs,
                  return_model=False, return_info=True)
        assert not np.allclose(ret.X[:10], self.adata.X[:10])
        assert 'X_dca_dropout' in ret.obsm_keys()
        assert 'dca_loss_history' in ret.uns_keys()

    def test_denoise_zinb_elempi(self):
        ret = dca(self.adata, mode='denoise', ae_type='zinb-elempi', copy=True, epochs=self.epochs,
                  return_model=False, return_info=True)
        assert not np.allclose(ret.X[:10], self.adata.X[:10])
        assert 'X_dca_dropout' in ret.obsm_keys()
        assert 'dca_loss_history' in ret.uns_keys()

    def test_denoise_zinb_elempi_sharedpi(self):
        ret = dca(self.adata, mode='denoise', ae_type='zinb-elempi', copy=True, epochs=self.epochs,
                  return_model=False, return_info=True, network_kwds={'sharedpi': True})
        assert not np.allclose(ret.X[:10], self.adata.X[:10])
        assert 'X_dca_dropout' in ret.obsm_keys()
        assert 'dca_loss_history' in ret.uns_keys()

    def test_latent_simple(self):
        # simple tests for latent
        ret = dca(self.adata, mode='latent', hidden_size=self.hid_size, copy=True, epochs=self.epochs)
        assert 'X_dca' in ret.obsm_keys()
        assert ret.obsm['X_dca'].shape[1] == self.hid_size[1]

    def test_latent_nb_conddisp(self):
        ret = dca(self.adata, mode='latent', ae_type='nb-conddisp', hidden_size=self.hid_size, copy=True, epochs=self.epochs)
        assert 'X_dca' in ret.obsm_keys()
        assert ret.obsm['X_dca'].shape[1] == self.hid_size[1]

    def test_latent_nb(self):
        ret = dca(self.adata, mode='latent', ae_type='nb', hidden_size=self.hid_size, copy=True, epochs=self.epochs, return_info=True)
        assert 'X_dca' in ret.obsm_keys()
        assert ret.obsm['X_dca'].shape[1] == self.hid_size[1]

    def test_latent_zinb(self):
        ret = dca(self.adata, mode='latent', ae_type='zinb', hidden_size=self.hid_size, copy=True, epochs=self.epochs)
        assert 'X_dca' in ret.obsm_keys()
        assert ret.obsm['X_dca'].shape[1] == self.hid_size[1]
