# GADBench: Revisiting and Benchmarking Supervised Graph Anomaly Detection

This is the official implementation of the following paper:

> [GADBench: Revisiting and Benchmarking Supervised Graph Anomaly Detection](https://arxiv.org/abs/2306.12251)
>
> Jianheng Tang, Fengrui Hua, Ziqi Gao, Peilin Zhao, Jia Li
>
> NeurIPS 2023 Datasets and Benchmarks Track

Environment Setup
-----------------

Before you begin, ensure that you have Anaconda or Miniconda installed on your system.
This guide assumes the use of a CUDA-enabled GPU.

```shell
# Create and activate a new Conda environment named 'GADBench'
conda create -n GADBench
conda activate GADBench

# Install Pytorch and DGL with CUDA 11.7 support
# If your use a different CUDA version, please refer to the PyTorch and DGL websites for the appropriate versions.
conda install numpy
conda install pytorch==1.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c dglteam/label/cu117 dgl

# Install additional dependencies
conda install pip
pip install xgboost pyod scikit-learn sympy pandas catboost
```

Dataset Preparation
-------------------

GADBench utilizes 10 different datasets.
Download these datasets from the provided [google drive link](https://drive.google.com/file/d/1txzXrzwBBAOEATXmfKzMUUKaXh6PJeR1/view?usp=sharing).
Due to the Copyright of [DGraph-Fin](https://dgraph.xinye.com/introduction) and [Elliptic](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set), you need to download these datasets by yourself.

The script to preprocess DGraph-Fin and Elliptic can be found in `datasets/preprocess.inpynb`.
After downloading, unzip all the files into a folder named `datasets' within the GADBench directory.

Additionally, GADBench includes an example dataset `reddit', which does not require manual downloading.

Benchmarking
------------

### Use Default Hyperparameters

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
python benchmark.py --datasets 5,8 --models GAT-GraphSAGE-XGBGraph --inductive 1
```

Benchmark CAREGNN and GraphConsis on heterogeneous graph datasets

```
python benchmark.py --datasets 10,11 --models CAREGNN-GraphConsis
```

### Use Optimal Hyperparameters through Random Search

Perform a random search of hyperparameters for the GCN model on the Reddit dataset in the fully-supervised setting (100 trials).

```
python random_search.py --trial 100 --datasets 0 --models GCN
```

Perform a random search of hyperparameters for all 26 models on all 10 datasets in the fully-supervised setting (100 trials).

```
python random_search.py --trial 100
```

Please refer to the code for more information on additional options and parameters.

## Reference

### Dataset Information

In the table below, we summarize all datasets in GADBench including the number of nodes and edges, the node feature dimension, the ratio of anomalous labels, the training ratio in the fully-supervised setting, the concept of relations, and the type of node features. Misc. indicates the node features are a combination of heterogeneous attributes, possibly including categorical, numerical, and temporal information,

| ID | Name                                                                                                        |    #Nodes |     #Edges | #Dim. | Anomaly | Train | Relation Concept     | Feature Type      |
| -- | ----------------------------------------------------------------------------------------------------------- | --------: | ---------: | ----: | ------: | ----: | -------------------- | ----------------- |
| 0  | [Reddit](https://github.com/pygod-team/data)                                                                   |    10,984 |    168,016 |    64 |   3.3\% |  40\% | Under Same Post      | Text Embedding    |
| 1  | [Weibo](https://github.com/pygod-team/data)                                                                    |     8,405 |    407,963 |   400 |  10.3\% |  40\% | Under Same Hashtag   | Text Embedding    |
| 2  | [Amazon](https://docs.dgl.ai/en/latest/generated/dgl.data.FraudAmazonDataset.html#dgl.data.FraudAmazonDataset) |    11,944 |  4,398,392 |    25 |   9.5\% |  70\% | Review Correlation   | Misc. Information |
| 3  | [YelpChi](https://docs.dgl.ai/en/latest/generated/dgl.data.FraudYelpDataset.html#dgl.data.FraudYelpDataset)    |    45,954 |  3,846,979 |    32 |  14.5\% |  70\% | Reviewer Interaction | Misc. Information |
| 4  | [Tolokers](https://docs.dgl.ai/en/latest/generated/dgl.data.TolokersDataset.html)                              |    11,758 |    519,000 |    10 |  21.8\% |  40\% | Work Collaboration   | Misc. Information |
| 5  | [Questions](https://docs.dgl.ai/en/latest/generated/dgl.data.QuestionsDataset.html)                            |    48,921 |    153,540 |   301 |   3.0\% |  52\% | Question Answering   | Text Embedding    |
| 6  | [T-Finance](https://github.com/squareRoot3/Rethinking-Anomaly-Detection)                                       |    39,357 | 21,222,543 |    10 |   4.6\% |  50\% | Transaction Record   | Misc. Information |
| 7  | [Elliptic](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)                                       |   203,769 |    234,355 |   166 |   9.8\% |  50\% | Payment Flow         | Misc. Information |
| 8  | [DGraph-Fin](https://dgraph.xinye.com/)                                                                        | 3,700,550 |  4,300,999 |    17 |   1.3\% |  70\% | Loan Guarantor       | Misc. Information |
| 9  | [T-Social](https://github.com/squareRoot3/Rethinking-Anomaly-Detection)                                        | 5,781,065 | 73,105,508 |    10 |   3.0\% |  40\% | Social Friendship    | Misc. Information |
| 10 | Amazon (Hetero)                                                                                             |    11,944 |  4,398,392 |    25 |   9.5\% |  70\% | Review Correlation   | Misc. Information |
| 11 | YelpChi (Hetero)                                                                                            |    45,954 |  3,846,979 |    32 |  14.5\% |  70\% | Reviewer Interaction | Misc. Information |

### Citation

If you use this package and find it useful, please cite our paper using the following BibTeX. Thanks! :)
