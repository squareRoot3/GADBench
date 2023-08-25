# GADBench: Revisiting and Benchmarking Supervised Graph Anomaly Detection

This is the official implementation of the following paper:

> [GADBench: Revisiting and Benchmarking Supervised Graph Anomaly Detection](https://arxiv.org/abs/2306.12251)

Environment Setup
---
Before you begin, ensure that you have Anaconda or Miniconda installed on your system. 
This guide assumes the use of a CUDA-enabled GPU.
```shell
# Create and activate a new Conda environment named 'GADBench'
conda create -n GADBench
conda activate GADBench

# Install PyTorch, torchvision, torchaudio, and DGL with CUDA 11.7 support
# If your CUDA version is different, please refer to the PyTorch and DGL websites for the appropriate versions.
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 -c pytorch -c nvidia
conda install -c dglteam/label/cu117 dgl

# Install additional dependencies
conda install pip
pip install xgboost pyod scikit-learn sympy pandas
```

Dataset Preparation
---
GADBench utilizes 10 different datasets. 
Download these datasets from the provided [google drive link](https://drive.google.com/file/d/1txzXrzwBBAOEATXmfKzMUUKaXh6PJeR1/view?usp=sharing). 
Due to the Copyright of [DGraph-Fin](https://dgraph.xinye.com/introduction) and [Elliptic](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set), you need to download these datasets by yourself. 

The script to preprocess DGraph-Fin and Elliptic can be found in `datasets/preprocess.inpynb`.
After downloading, unzip all the files into a folder named dataset within the GADBench directory.

Additionally, GADBench includes an example dataset named `reddit', which does not require manual downloading.

Benchmarking
---

### With Default Hyperparameters

Benchmark the GCN model on the Reddit dataset with the fully-supervised setting (single trial).
```
python benchmark.py --trial 1 --datasets 0 --models GCN
```
Benchmark GIN and BWGNN on all 10 datasets in the semi-supervised setting (10 trials). 
```
python benchmark.py --trial 10 --models GIN-BWGNN --semi_supervised 1 
```
Benchmark 26 models on all 10 datasets in the fully-supervised setting (10 trials). 
```
python benchmark.py --trial 10
```
Benchmark severl models in the inductive setting
```
python benchmark.py --datasets 5,8 --models GAT-GraphSAGE-XGBGraph
```
Benchmark CAREGNN and GraphConsis on heterogeneous graph datasets 
```
python benchmark.py --datasets 10,11 --models CAREGNN-GraphConsis
```

### Using Optimal Hyperparameters through Random Search


Perform a random search of hyperparameters for the GCN model on the Reddit dataset in the fully-supervised setting (100 trials).
```
python random_search.py --trial 100 --datasets 0 --models GCN
```
Perform a random search of hyperparameters for all 26 models on all 10 datasets in the fully-supervised setting (100 trials). 

```
python random_search.py --trial 100
```
Please refer to the code for more information on additional options and parameters.



