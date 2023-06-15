# GADBench: Revisiting and Benchmarking Supervised Graph Anomaly Detection


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
pip install xgboost pyod scikit-learn simpy
```

Dataset Preparation
---
GADBench utilizes 10 different datasets. 
Download these datasets from the provided [google drive link](https://drive.google.com/drive/folders/1PpNwvZx_YRSCDiHaBUmRIS3x1rZR7fMr?usp=sharing). 
After downloading, unzip all the files into a folder named dataset within the GADBench directory.
Additionally, GADBench includes an example dataset named `reddit', which does not require manual downloading.

Benchmarking
---

### With Default Hyperparameters

Benchmark the GCN model on the Reddit dataset with the fully-supervised setting (single trial).
```
python benchmark.py --trial 1 --datasets 0 --model GCN
```
Benchmark all 23 models on all 10 datasets in the semi-supervised setting (10 trials). 
This reproduces Figure 1 and Table 7 from the paper.
```
python benchmark.py --trial 10 --semi_supervised 1
```
Benchmark all 23 models on all 10 datasets in the fully-supervised setting (10 trials). 
This reproduces Figure 1 and Table 8 from the paper.
```
python benchmark.py --trial 10
```


### Using Optimal Hyperparameters through Random Search


Perform a random search of hyperparameters for the GCN model on the Reddit dataset in the fully-supervised setting (100 trials).
```
python random_search.py --trial 100 --datasets 0 --model GCN
```
Perform a random search of hyperparameters for all 23 models on all 10 datasets in the fully-supervised setting (100 trials). 
This reproduces Tables 4 and 9 from the paper.

```
python random_search.py --trial 100
```
Please refer to the code for more information on additional options and parameters.



