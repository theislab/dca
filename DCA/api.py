import os, tempfile, shutil
import anndata

from .io import create_dataset
from .train import train
from .network import AE_types


def autoencode(adata,
               output_dir=None,
               aetype='zinb-conddisp',
               size_factors=True,
               normalize_input=True,
               logtrans_input=True,
               net_kwargs={},
               training_kwargs={}):

    assert isinstance(adata, anndata.AnnData), 'adata must be an AnnData instance'

    temp = False

    if output_dir is None:
        temp = True
        output_dir = tempfile.mkdtemp()

    ds = create_dataset(adata,
                        output_file=os.path.join(output_dir, 'input.zarr'),
                        transpose=False,
                        test_split=False)

    input_size = output_size = ds.train.shape[1]
    net = AE_types[aetype](input_size=input_size,
                         output_size=output_size,
                         **net_kwargs)
    net.save()
    net.build()

    losses = train(ds, net, output_dir=output_dir,
                   size_factors=size_factors,
                   normalize_input=normalize_input,
                   logtrans_input=logtrans_input,
                   **training_kwargs)

    res = net.predict(ds.full.matrix[:],
                      ds.full.rownames,
                      ds.full.colnames,
                      size_factors=size_factors,
                      normalize_input=normalize_input,
                      logtrans_input=logtrans_input)

    res['losses'] = losses
    res['net'] = net

    if temp:
        shutil.rmtree(output_dir, ignore_errors=True)

    return res
