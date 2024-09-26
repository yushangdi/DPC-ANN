# PECANN: Parallel Efficient Clustering with Approximate Nearest Neighbors

PECANN is an efficient clustering algorithm for large scale, high dimensional density peaks clustering. This library contains PECANN's fast and parallel C++ implementation, as well as easy to use python bindings.


## Building PECANN

To build PECANN, you can build either python bindings or our C++ library/executable file. Below are instructions for both:

### Python Bindings

```bash
python3 -m pip install .
```


### C++ Library, Command Line Executable, and Python .so File

```bash
python3 -m pip install nanobind
mkdir build
cd build
cmake ..
make
```

## Example Usage

Below are simple examples for how to run PECANN from python and from the command line on one of the sample gaussian datasets:

### Python

```python

import dpc_ann
cluster_results = dpc_ann.dpc_numpy(
        center_finder=dpc_ann.ThresholdCenterFinder(dependant_dist_threshold=8.36),
        density_computer=dpc_ann.KthDistanceDensityComputer(),
        data=data
)
# You could also do e.g. center_finder=dpc_ann.ProductCenterFinder(num_clusters=4)
clusters = clustering_result.clusters # 1 X N numpy array of cluster assignments
metadata = cluster_results.metadata # Dictionary of runtimes for various parts of the clustering process
```

Below are some additional arguments this function takes, as well as a short description of each argument.


### Additional arguments

| Argument             | Description                                                                                       |
|----------------------|---------------------------------------------------------------------------------------------------|
| data                 | A 2d float32 numpy array.                                                                        |
| K                    | How many neighbors to use to estimate the density (default is 6).                                  |
| Lbuild               | The beam search size to use during the underlying graph index construction (default is 12).      |
| L                    | The beam search size to use while finding the nearest K neighbors for each point (default is 12).|
| Lnn                  | The initial beam search size to use when finding dependent points (default is 4).                 |
| center_finder        | The center finder to use to prune the DPC tree. Options include ProductCenterFinder or ThresholdCenterFinder, both of which can be directly constructed from the dpc_ann module.|
| density_computer    | The density computer to use to compute the density of each point. Options include KthDistanceDensityComputer, ExpSquaredDensityComputer, and the NormalizedDensityComputer (among others), all of which can be constructed directly from the dpc_ann module. Default is KthDistanceDensityComputer. |
| output_path          | An optional parameter specifying the path to write the cluster assignment to.                      |
| decision_graph_path  | An optional parameter specifying the path to write the decision graph to.          |
| max_degree           | The maximum degree of the underlying graph index (default 16).                                    |
| alpha                | The alpha to use for the underlying graph index, for indices that require it (default 1.2).       |
| num_clusters         | The number of times to independently reconstruct the underlying graph index, for indices that require this (default 4).|
| graph_type           | Which underlying graph index to use (default "Vamana", choices are "Vamana", "HCNNG", "BruteForce", "pyNNDescent").|


### Command Line

```bash

./build/dpc_framework_exe  --query_file ./data/gaussian_example/gaussian_4_1000.data --decision_graph_path ./results/gaussian_4_1000.dg --output_file ./results/gaussian_4_1000.cluster --dist_cutoff 8.36 

```

The command line takes most of the same options as the python bindings, and they can be seen by running the command line script with no arguments.

## Reproducing our Experiments

All of the experiments from our paper can be easily reproduced. 

### Installing Dependencies

Our experiment's dependencies are listed in requirements.txt. You can install them by running
```pip3 install -r requirements.txt```

As baselines for our experiments, we evaluate the K-means implementation from the excellent FAISS library as well as fast-dp, a prior high dimensional DPC algorithm. FAISS is included as a dependency in requirements.txt. To install fastDP, you need to clone their repo and build it:
```
git clone https://github.com/uef-machine-learning/fastdp
cd fastdp
pip3 install .
```

### Datasets

Our experiments expect each dataset to be in a folder named data/<dataset_name> (if you wish to use another location for disk space reasons, you can store the datasets there and add a simlink to the data folder). A dataset consists of two files: a <dataset_name>.npy file that is a 2D float32 numpy array, where each line is a point in the dataset, and a <dataset_name>.gt text file, where the ith line is a number representing the ground truth clustering for the ith point in the dataset.

You can also generate them as follows:
- `bin/download_simple_datasets.sh` downloads s2, s3, and the unbalance datasets from [here](http://cs.joensuu.fi/sipu/datasets/).
- `bin/download_mnist.sh` downloads mnist.
- `python3 data_processors/create_imagenet.py --dataset_path <path to kaggle imagenet download>` creates the ImageNet embedding dataset. [link to kaggle](https://www.kaggle.com/c/imagenet-object-localization-challenge/)
- `python3 data_processors/create_birds.py --dataset_path <path to kaggle birds download train subset folder>` creates the birds embedding dataset. [link to kaggle](https://www.kaggle.com/datasets/gpiosenka/100-bird-species)
- `python3 data_processors/embed_huggingface.py --dataset_name mteb/reddit-clustering --download_dir <path to store downloaded data> --data_dir <path to store embedded data>` creates the reddit dataset.
- `python3 data_processors/embed_huggingface.py mteb/arxiv-clustering-s2s  --download_dir <path to store downloaded data> --data_dir <path to store embedded data>` creates the arxiv dataset.

### Running Experiments

By default, all experiments will run with all threads on your machine except for the thread scaling experiment.

#### Dataset Size Scaling:

The first command below varies the dataset size and saves the ARI results to a log file in results (cluster_analysis_dataset_size_scaling_{timestamp}), while the second command plots the results from that file and saves a resulting pdf of the graph to results/paper. This is a similar pattern to most of our experiments.

```bash
python3 experiments/paper/dataset_size_scaling/dataset_size_scaling.py
python3 experiments/paper/dataset_size_scaling/plot_dataset_size_scaling.py results/cluster_analysis_dataset_size_scaling_{fill_in_run_specific_value}
```

#### Pareto Frontiers and Brute Force Table:

The following commands generate the data for Pareto frontiers for each method. Since they include brute force runs of DPC and many hyperparameter settings, they may take a few days or longer to run, depending on how many cores your machine has.
```bash
python3 experiments/paper/brute_force.py
python3 experiments/paper/pareto/different_graph_methods_on_imagenet.py
python3 experiments/paper/pareto/vamana_pareto.py
python3 experiments/paper/pareto/test_fastdp.sh
```

The following command plots the pareto fronts:
```bash
python3 experiments/paper/pareto/plot_pareto.py results

```

#### Runtime Decompositions and Accuracy vs Time for Various Density Methods

```bash
python3 experiments/paper/varying_k/varying_k.py
python3 experiments/paper/varying_k results
```


#### ARI vs. (# Clusters Passed to Method / # Ground Truth Clusters)

```bash
python3 experiments/paper/varying_num_clusters/varying_num_clusters.py
python3 experiments/paper/varying_num_clusters/plot_varying_num_clusters.py results/cluster_analysis_varying_num_clusters.csv
```

#### Thread Scaling

The thread scaling experiment assumes a 30 core, 60 thread machine. If you have a different size machine, you will need to change line 78 of experiments/paper/thread_scaling/thread_scaling.py accordingly:
```python
    if len(current_threads) in [1, 2, 4, 8, 15, 30, 60]:
```
Also, if your machine has multiple numa nodes, it is a good idea to change line 78 to be the single numa node thread counts and run it, then change it to be just the multiple numa node thread counts and run it prefixed with numactl -i all, and then combine the resulting result csv files.

The commands:
```bash
python3 experiments/paper/thread_scaling/thread_scaling.py
python3 experiments/paper/thread_scaling/plot_thread_scaling.py results --threads
```


```
@software{Yu_PECANN_Parallel_Efficient_2023,
author = {Yu, Shangdi and Engels, Joshua  and Huang, Yihao and Shun, Julian },
month = dec,
title = {{PECANN: Parallel Efficient Clustering with Graph-Based Approximate Nearest Neighbor Search}},
url = {https://github.com/yushangdi/PECANN-DPC/},
version = {1.0.0},
year = {2023}
}
```