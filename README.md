## Deep count autoencoder for denoising scRNA-seq data

A deep count autoencoder network to remove the dropout effect from scRNA-seq datasets by taking the count structure, overdispersed nature and sparsity of the data into account using zero-inflated negative binomial distribution (ZINB).

### Installation

pip install git+https://github.com/gokceneraslan/autoencoder

### Hyperparameter optimization

`Hyperopt` (from github master branch) and `kopt` packages are required for hyperparameter optimization:

pip install git+https://github.com/hyperopt/hyperopt

and

pip install git+https://github.com/Avsecz/keras-hyperopt
