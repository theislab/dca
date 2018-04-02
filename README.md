## Deep count autoencoder for denoising scRNA-seq data

A deep count autoencoder network to denoise scRNA-seq data and remove the dropout effect by taking the count structure, overdispersed nature and sparsity of the data into account using a deep autoencoder with zero-inflated negative binomial (ZINB) loss function.

See the [tutorial](tutorial.ipynb) for more details.

### Installation

`pip install git+https://github.com/gokceneraslan/countae`

Alternatively, if you have `git` installed, you can clone the repo:

`git clone https://github.com/gokceneraslan/countae.git`

and then run

`cd countae; pip install -e .`

to install the count autoencoder and the required packages.

### Usage

You can run the autoencoder from the command line:

`dca matrix.csv results`

where `matrix.csv` is a CSV/TSV-formatted raw count matrix with genes in rows and cells in columns. Cell and gene labels are mandatory. Results folder contains dropout, mean and dispersion matrices in TSV format. `mean.tsv` is the main output of the method.

Use `-h` option to see all available parameters and defaults.

### Hyperparameter optimization

You can run the autoencoder with `--hyper` option do perform hyperparameter search.
