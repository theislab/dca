## Deep count autoencoder for denoising scRNA-seq data

A deep count autoencoder network to denoise scRNA-seq data and remove the dropout effect by taking the count structure, overdispersed nature and sparsity of the data into account using a deep autoencoder with zero-inflated negative binomial (ZINB) loss function.

See our [manuscript](https://www.nature.com/articles/s41467-018-07931-2) and [tutorial](https://nbviewer.ipython.org/github/theislab/dca/blob/master/tutorial.ipynb) for more details.

## Getting Started

For hassle-free usage of this package, I recommend using Docker. You can pull the latest Docker image from Docker Hub using the following command:

```bash
docker pull rb16b/dca:latest
```
Once you have the Docker image, you can run a container with Jupyter Notebook inside it using the following command:

```
docker run -p 8888:8888 -v /path/to/your/notebooks:/home/jovyan/work rb16b/dca:latest

```
Replace /path/to/your/notebooks with the path to the directory where you want to store your Jupyter notebooks. This command will start a container with Jupyter Notebook running, and you can access it by opening your web browser and navigating to http://localhost:8888.


Make sure to replace `username/image:latest` with the actual Docker image name and tag. Additionally, provide relevant information about mounting volumes if necessary, like the `-v` option in the Docker run command, which mounts a local directory to a directory inside the container.


### Usage

You can run the autoencoder from the command line:

`dca matrix.csv results`

where `matrix.csv` is a CSV/TSV-formatted raw count matrix with genes in rows and cells in columns. Cell and gene labels are mandatory. 

### Results

Output folder contains the main output file (representing the mean parameter of ZINB distribution) as well as some additional matrices in TSV format:

- `mean.tsv` is the main output of the method which represents the mean parameter of the ZINB distribution. This file has the same dimensions as the input file (except that the zero-expression genes or cells are excluded). It is formatted as a `gene x cell` matrix. Additionally, `mean_norm.tsv` file contains the library size-normalized expressions of each cell and gene. See `normalize_total` function from [Scanpy](https://scanpy.readthedocs.io/en/stable/api/scanpy.pp.normalize_total.html) for the details about the default library size normalization method used in DCA.

- `pi.tsv` and `dispersion.tsv` files represent dropout probabilities and dispersion for each cell and gene. Matrix dimensions are same as `mean.tsv` and the input file.

- `reduced.tsv` file contains the hidden representation of each cell (in a 32-dimensional space by default), which denotes the activations of bottleneck neurons.

Use `-h` option to see all available parameters and defaults.

### Hyperparameter optimization

You can run the autoencoder with `--hyper` option to perform hyperparameter search.
