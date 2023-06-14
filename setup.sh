# Take CUDA==11.7 as an example.
# If your CUDA version is less than 11.7, please find appropriate version of dgl and pytorch according to their website
conda create -n GADBench
source activate GADBench
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c dglteam/label/cu117 dgl
conda install pip
pip install xgboost
pip install pyod
pip install sklearn
pip install simpy