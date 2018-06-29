## Deep count autoencoder for denoising scRNA-seq data

A deep count autoencoder network to denoise scRNA-seq data and remove the dropout effect by taking the count structure, overdispersed nature and sparsity of the data into account using a deep autoencoder with zero-inflated negative binomial (ZINB) loss function.

See our [bioRxiv manuscript](https://www.biorxiv.org/content/early/2018/04/13/300681) and [tutorial](tutorial.ipynb) for more details.

### Installation

Use

```
pip install dca
```

command to install the count autoencoder and the required packages.

### Usage

You can run the autoencoder from the command line:

`dca matrix.csv results`

where `matrix.csv` is a CSV/TSV-formatted raw count matrix with genes in rows and cells in columns. Cell and gene labels are mandatory. 

### Results

Output folder contains the main output file (representing the mean parameter of ZINB distribution) as well as some additional matrices in TSV format:

- `mean.tsv` is the main output of the method which represents the mean parameter of the ZINB distribution. This file has the same dimensions as the input file (except that the zero-expression genes or cells are excluded). It is formatted as a `gene x cell` matrix. Additionally, `mean_norm.tsv` file contains the library size-normalized expressions of each cell and gene. See `normalize_per_cell` function from [Scanpy](https://scanpy.readthedocs.io/en/latest/api/scanpy.api.pp.normalize_per_cell.html#scanpy.api.pp.normalize_per_cell) for the details about the default library size normalization method used in DCA.

- `pi.tsv` and `dispersion.tsv` files represent dropout probabilities and dispersion for each cell and gene. Matrix dimensions are same as `mean.tsv` and the input file.

- `reduced.tsv` file contains the hidden representation of each cell (in a 32-dimensional space by default), which denotes the activations of bottleneck neurons.

Use `-h` option to see all available parameters and defaults.

### Hyperparameter optimization

You can run the autoencoder with `--hyper` option to perform hyperparameter search.
