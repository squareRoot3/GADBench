import dgl
import torch
import numpy as np
import os
import random
import pandas
from dataset import *

g = Dataset('dgraphfin').graph
g = dgl.remove_self_loop(g)
ind1, ind2 = g.adj().coalesce().indices()[0], g.adj().coalesce().indices()[1]
labels = g.ndata['label']

normal_feat = g.ndata['feature'][labels==0].cuda()
anomaly_feat = g.ndata['feature'][labels==1].cuda()
knt=0
for i in range(15509):
    if torch.cdist(anomaly_feat[1].unsqueeze(0), normal_feat).argmin() <= 0.00001:
        knt += 1
        print(knt, i)