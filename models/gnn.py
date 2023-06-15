import torch
import torch.nn.functional as F
import dgl.function as fn
import sympy
import scipy
import dgl.nn.pytorch.conv as dglnn
from torch import nn
from scipy.special import comb
import math


class PolyConv(nn.Module):
    def __init__(self, theta):
        super(PolyConv, self).__init__()
        self._theta = theta
        self._k = len(self._theta)

    def forward(self, graph, feat):
        def unnLaplacian(feat, D_invsqrt, graph):
            """ Operation Feat * D^-1/2 A D^-1/2 """
            graph.ndata['h'] = feat * D_invsqrt
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            return feat - graph.ndata.pop('h') * D_invsqrt

        with graph.local_scope():
            D_invsqrt = torch.pow(graph.in_degrees().float().clamp(
                min=1), -0.5).unsqueeze(-1).to(feat.device)
            h = self._theta[0]*feat
            for k in range(1, self._k):
                feat = unnLaplacian(feat, D_invsqrt, graph)
                h += self._theta[k]*feat
        return h


class BernConv(nn.Module):
    def __init__(self, orders=2):
        super().__init__()
        self.K = orders
        self.weight = nn.Parameter(torch.ones(k+1))

    def forward(self, graph, feat):
        def unnLaplacian1(feat, D_invsqrt, graph):
            """ \hat{L} X """
            graph.ndata['h'] = feat * D_invsqrt
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            return feat - graph.ndata.pop('h') * D_invsqrt

        def unnLaplacian2(feat, D_invsqrt, graph):
            """ (2I - \hat{L}) X """
            graph.ndata['h'] = feat * D_invsqrt
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            return feat + graph.ndata.pop('h') * D_invsqrt

        with graph.local_scope():
            tmp = [feat]
            weight = nn.functional.relu(self.weight)
            D_invsqrt = torch.pow(graph.in_degrees().float().clamp(
                min=1), -0.5).unsqueeze(-1).to(feat.device)
            for i in range(self.K):
                feat = unnLaplacian2(feat, D_invsqrt, graph)
                tmp.append(feat)

            out_feat = (comb(self.K, 0)/(2**self.K))*weight[0]*tmp[self.K]
            for i in range(self.K):
                x = tmp[self.K-i-1]
                for j in range(i+1):
                    x = unnLaplacian1(feat, D_invsqrt, graph)
                out_feat = out_feat+(comb(self.K, i+1)/(2**self.K))*weight[i+1]*x
        return out_feat


class BernNet(nn.Module):
    def __init__(self, in_feats, h_feats=32, num_classes=2, orders=2, mlp_layers=1, dropout_rate=0, activation='ReLU',
                 **kwargs):
        super().__init__()
        self.bernconv = BernConv(orders=orders)
        self.linear1 = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.mlp = MLP(h_feats, h_feats, num_classes, mlp_layers, dropout_rate=dropout_rate)
        self.act = getattr(nn, activation)()

    def forward(self, graph):
        in_feat = graph.ndata['feature']
        h = self.linear1(in_feat)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)
        h = self.bernconv(graph, h)
        h = self.act(h)
        h = self.mlp(h, False)
        return h


class AMNet(nn.Module):
    def __init__(self, in_feats, h_feats=32, num_classes=2, orders=2, num_layers=3, mlp_layers=1, dropout_rate=0,
                 activation='ReLU', **kwargs):
        super().__init__()
        self.linear1 = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.mlp_layers = nn.Sequential()
        if mlp_layers > 0:
            for i in range(mlp_layers-1):
                self.mlp_layers.append(nn.Linear(h_feats, h_feats))
            self.mlp_layers.append(nn.Linear(h_feats, num_classes))
        self.act = getattr(nn, activation)()
        self.attn = nn.Tanh()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        self.num_layers = num_layers
        self.layers = nn.Sequential()
        for i in range(num_layers):
            self.layers.append(BernConv(orders))
        self.linear_transform_in = nn.Sequential(self.linear1, self.act, self.linear2)
        self.W_f = nn.Sequential(self.linear2, self.attn)
        self.W_x = nn.Sequential(self.linear2, self.attn)
        self.linear_cls_out = nn.Sequential(self.mlp_layers)

    def forward(self, graph):
        in_feat = graph.ndata['feature']
        h = self.linear_transform_in(in_feat)
        h_list = []
        for i, layer in enumerate(self.layers):
            h_ = layer(graph, h)
            h_list.append(h_)
        h_filters = torch.stack(h_list, dim=1)
        h_filters_proj = self.W_f(h_filters)
        x_proj = self.W_x(h).unsqueeze(-1)
        score_logit = torch.bmm(h_filters_proj, x_proj)
        soft_score = F.softmax(score_logit, dim=1)
        score = soft_score
        res = h_filters[:, 0, :] * score[:, 0]
        for i in range(1, self.num_layers):
            res += h_filters[:, i, :] * score[:, i]
        y_hat = self.linear_cls_out(res)
        return y_hat


def calculate_theta(d):
    thetas = []
    x = sympy.symbols('x')
    for i in range(d+1):
        f = sympy.poly((x/2) ** i * (1 - x/2) ** (d-i) / (scipy.special.beta(i+1, d+1-i)))
        coeff = f.all_coeffs()
        inv_coeff = []
        for i in range(d+1):
            inv_coeff.append(float(coeff[d-i]))
        thetas.append(inv_coeff)
    return thetas


class BWGNN(nn.Module):
    def __init__(self, in_feats, h_feats=32, num_classes=2, num_layers=2, mlp_layers=2, dropout_rate=0,
                 activation='ReLU', **kwargs):
        super(BWGNN, self).__init__()
        self.thetas = calculate_theta(d=num_layers)
        self.conv = []
        for i in range(len(self.thetas)):
            self.conv.append(PolyConv(self.thetas[i]))
        self.linear = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.mlp = MLP(h_feats*len(self.conv), h_feats, num_classes, mlp_layers, dropout_rate)
        self.act = getattr(nn, activation)()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, graph):
        in_feat = graph.ndata['feature']
        h = self.linear(in_feat)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)
        h_final = torch.zeros([len(in_feat), 0], device=h.device)

        for conv in self.conv:
            h0 = conv(graph, h)
            h_final = torch.cat([h_final, h0], -1)
        h_final = self.dropout(h_final)
        h = self.mlp(h_final, False)
        return h


class GCN(nn.Module):
    def __init__(self, in_feats, h_feats=32, num_classes=2, num_layers=2, mlp_layers=1, dropout_rate=0,
                 activation='ReLU', **kwargs):
        super().__init__()
        self.h_feats = h_feats
        self.layers = nn.ModuleList()
        self.act = getattr(nn, activation)()
        self.layers.append(dglnn.GraphConv(in_feats, h_feats, activation=self.act))
        for i in range(num_layers-1):
            self.layers.append(dglnn.GraphConv(h_feats, h_feats, activation=self.act))
        self.mlp = MLP(h_feats, h_feats, num_classes, mlp_layers, dropout_rate)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, graph):
        h = graph.ndata['feature']
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(graph, h)
        h = self.mlp(h, False)
        return h


class GIN(nn.Module):
    def __init__(self, in_feats, h_feats=32, num_classes=2, num_layers=2, agg='mean', dropout_rate=0,
                 activation='ReLU', **kwargs):
        super().__init__()
        self.layers = nn.ModuleList()
        self.act = getattr(nn, activation)()
        self.layers.append(dglnn.GINConv(nn.Linear(in_feats, h_feats), activation=self.act, aggregator_type=agg))
        for i in range(1, num_layers-1):
            self.layers.append(dglnn.GINConv(nn.Linear(h_feats, h_feats), activation=self.act, aggregator_type=agg))
        self.layers.append(dglnn.GINConv(nn.Linear(h_feats, num_classes),  activation=None, aggregator_type=agg))
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, graph):
        h = graph.ndata['feature']
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(graph, h)
        return h


class GIN_noparam(nn.Module):
    def __init__(self, num_layers=2, agg='mean', init_eps=-1, **kwargs):
        super().__init__()
        self.gnn = dglnn.GINConv(None, activation=None, init_eps=init_eps,
                                 aggregator_type=agg)
        self.num_layers = num_layers

    def forward(self, graph):
        h = graph.ndata['feature']
        h_final = h.detach().clone()
        for i in range(self.num_layers):
            h = self.gnn(graph, h)
            h_final = torch.cat([h_final, h], -1)
        return h_final


class ChebNet(nn.Module):
    def __init__(self, in_feats, h_feats=32, num_classes=2, num_layers=2, dropout_rate=0, activation='ReLU', **kwargs):
        super().__init__()
        self.input_linear = nn.Linear(in_feats, h_feats)
        self.act = getattr(nn, activation)()
        self.chebconv = dglnn.ChebConv(h_feats, h_feats, num_layers, activation=self.act)
        self.output_linear = nn.Linear(h_feats, num_classes)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, graph):
        h = graph.ndata['feature']
        h = self.input_linear(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.chebconv(graph, h, lambda_max=[2])
        h = self.dropout(h)
        h = self.output_linear(h)
        return h


class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats=32, num_classes=2, num_layers=2, agg='pool', dropout_rate=0,
                 activation='ReLU', **kwargs):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.act = getattr(nn, activation)()
        self.layers.append(dglnn.SAGEConv(in_feats, h_feats, agg, activation=self.act))
        for i in range(num_layers-1):
            self.layers.append(dglnn.SAGEConv(h_feats, h_feats, agg, activation=self.act))
        self.output_linear = nn.Linear(h_feats, num_classes)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, graph):
        h = graph.ndata['feature']
        for layer in self.layers:
            h = self.dropout(h)
            h = layer(graph, h)
        h = self.output_linear(h)
        return h


class MLP(nn.Module):
    def __init__(self, in_feats, h_feats=32, num_classes=2, num_layers=2, dropout_rate=0, activation='ReLU', **kwargs):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.act = getattr(nn, activation)()
        if num_layers == 0:
            return
        if num_layers == 1:
            self.layers.append(nn.Linear(in_feats, num_classes))
        else:
            self.layers.append(nn.Linear(in_feats, h_feats))
            for i in range(1, num_layers-1):
                self.layers.append(nn.Linear(h_feats, h_feats))
            self.layers.append(nn.Linear(h_feats, num_classes))
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, h, is_graph=True):
        if is_graph:
            h = h.ndata['feature']
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(h)
            if i != len(self.layers)-1:
                h = self.act(h)
        return h


class SGC(nn.Module):
    def __init__(self, in_feats, h_feats=32, num_classes=2, k=2, mlp_layers=1, dropout_rate = 0, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList()
        self.mlp = None
        if mlp_layers==1:
            self.sgc = dglnn.SGConv(in_feats, num_classes, k=k)
        else:
            self.sgc = dglnn.SGConv(in_feats, h_feats, k=k)
            self.mlp = MLP(h_feats, num_classes, num_classes, mlp_layers-1, dropout_rate)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, graph):
        h = graph.ndata['feature']
        h = self.dropout(h)
        h = self.sgc(graph,h)
        if self.mlp is not None:
            h = self.mlp(h, False)
        return h


class DGI_GIN(nn.Module):
    def __init__(self, in_feats, h_feats=32, num_layers=2, agg='mean', dropout_rate=0, **kwargs):
        super().__init__()
        self.h_feats = h_feats
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.GINConv(nn.Linear(in_feats, h_feats), activation=F.relu, aggregator_type=agg))
        for i in range(num_layers-1):
            self.layers.append(dglnn.GINConv(nn.Linear(h_feats, h_feats), activation=F.relu, aggregator_type=agg))
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, graph, h=None):
        if h is None:
            h = graph.ndata['feature']
        for i, layer in enumerate(self.layers):
            if i != 0 and self.dropout:
                h = self.dropout(h)
            h = layer(graph, h)
        return h


class DCI_Encoder(nn.Module):
    def __init__(self, in_feats, h_feats=32, num_classes=2, num_layers=2, mlp_layers=2, dropout_rate=0, agg='mean',  **kwargs):
        super().__init__()
        self.conv = DGI_GIN(in_feats, h_feats, num_layers, agg, dropout_rate)
        self.mlp = MLP(h_feats, h_feats, num_classes, mlp_layers, dropout_rate)

    def forward(self, graph, corrupt=False, use_mlp=False):
        h = graph.ndata['feature']
        if corrupt:
            perm = torch.randperm(graph.num_nodes())
            h = h[perm]
        h = self.conv(graph, h)
        if use_mlp:
            h = self.mlp(h, is_graph=False)
        return h


class Discriminator(nn.Module):
    def __init__(self, h_feats, **kwargs):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(h_feats, h_feats))
        self.reset_parameters()

    def uniform(self, size, tensor):
        bound = 1.0 / math.sqrt(size)
        if tensor is not None:
            tensor.data.uniform_(-bound, bound)

    def reset_parameters(self):
        size = self.weight.size(0)
        self.uniform(size, self.weight)

    def forward(self, features, summary):
        features = torch.matmul(features, torch.matmul(self.weight, summary))
        return features



class DCI(nn.Module):
    def __init__(self, in_feats, h_feats=32, num_classes=2, num_layers=2, mlp_layers=1, dropout_rate=0, **kwargs):
        super().__init__()
        self.encoder = DCI_Encoder(in_feats, h_feats, num_classes, num_layers, mlp_layers, dropout_rate)
        self.discriminator = Discriminator(h_feats)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, graph, cluster_info, cluster_num):
        positive = self.encoder(graph, corrupt=False)
        negative = self.encoder(graph, corrupt=True)

        loss = 0

        for i in range(cluster_num):
            node_idx = cluster_info[i]

            positive_block = torch.unsqueeze(positive[node_idx], 0)
            negative_block = torch.unsqueeze(negative[node_idx], 0)
            summary = torch.sigmoid(positive.mean(dim=0))

            negative_block = self.discriminator(negative_block, summary)
            positive_block = self.discriminator(positive_block, summary)

            l1 = self.loss(positive_block, torch.ones_like(positive_block))
            l2 = self.loss(negative_block, torch.zeros_like(negative_block))

            loss_tmp = l1 + l2
            loss += loss_tmp

        return loss / cluster_num

    def get_emb(self, graph):
        h = self.encoder(graph, corrupt=False)
        return h