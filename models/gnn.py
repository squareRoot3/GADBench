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
        self.weight = nn.Parameter(torch.ones(orders+1))

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
        print(h_final)
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
        h = self.sgc(graph, h)
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
    

class PrincipalAggregate(nn.Module):
    def __init__(self, aggregators=['mean', 'max', 'min', 'std'], h_feats=32, act='ReLU', dropout=0.):
        super().__init__()
        self.aggregators = aggregators
        self.agg_funcs = [getattr(self, f"agg_{agg}") for agg in aggregators]
        self.linear = nn.Linear(len(self.agg_funcs)*h_feats, h_feats)
        self.act =getattr(nn, act)()

    def forward(self, mfg, feat):
        h = [agg(mfg, feat) for agg in self.agg_funcs]
        return self.act(self.linear(torch.cat(h, dim=1)))

    def agg_mean(self, mfg, X):
        mfg.srcdata['h'] = X
        mfg.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h'))
        return mfg.dstdata['h']

    def agg_min(self, mfg, X):
        mfg.srcdata['h'] = X
        mfg.update_all(fn.copy_u('h', 'm'), fn.min('m', 'h'))
        return mfg.dstdata['h']

    def agg_max(self, mfg, X):
        mfg.srcdata['h'] = X
        mfg.update_all(fn.copy_u('h', 'm'), fn.max('m', 'h'))
        return mfg.dstdata['h']

    def agg_std(self, mfg, X):
        diff = self.agg_mean(mfg, X ** 2) - self.agg_mean(mfg, X) ** 2
        return torch.sqrt(F.relu(diff) + 1e-5)

    def agg_sum(self, mfg, X):
        mfg.srcdata['h'] = X
        mfg.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
        return mfg.dstdata['h']

    def __repr__(self):
        return f"""PrincipalAggregate(
            aggregators={self.aggregators}
            linear={self.linear}
        )"""


class PNA(nn.Module):
    def __init__(self, in_feats, h_feats=32, num_classes=2, num_layers=2, activation='ReLU', dropout_rate=0, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList()
        # self.layers.append(nn.Linear(in_feats, h_feats))
        self.input_linear = nn.Linear(in_feats, h_feats)

        self.act = getattr(nn, activation)()
        self.output_linear = nn.Linear(h_feats, num_classes)
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

        for i in range(0, num_layers):
            self.layers.append(PrincipalAggregate(h_feats=h_feats, act=activation, dropout=dropout_rate))


    def forward(self, graph):
        h = graph.ndata['feature']
        h = self.input_linear(h)
        for i, layer in enumerate(self.layers):
            if i != 0 and self.dropout:
                h = self.dropout(h)
            h = layer(graph, h)
        h = self.output_linear(h)
        return h


class GATv2(nn.Module):
    def __init__(self, in_feats, h_feats=16, num_classes=2, num_layers=2, num_heads=4, dropout_rate=0, residual=False, activation='ReLU', **kwargs):
        super().__init__()
        self.layers = nn.ModuleList()
        self.input_linear = nn.Linear(in_feats, h_feats)
        self.act = getattr(nn, activation)()
        # self.layers.append(dglnn.GATv2Conv(in_feats, h_feats, num_heads))

        for i in range(0, num_layers):
            self.layers.append(dglnn.GATv2Conv(h_feats, h_feats//num_heads, num_heads, feat_drop=dropout_rate, attn_drop=dropout_rate, residual=residual, activation=self.act))
        self.output_linear = nn.Linear(h_feats, num_classes)

    def forward(self, graph):
        h = graph.ndata['feature']
        h = self.input_linear(h)
        for i, layer in enumerate(self.layers):
            h = layer(graph, h)
            h = h.reshape([h.shape[0], -1])
        h = self.output_linear(h)
        return h


class CAREConv(nn.Module):
    def __init__(self,in_feats, h_feats, num_classes=2, activation=None, step_size=0.02, **kwargs):
        super().__init__()
        self.activation = activation
        self.step_size = step_size
        self.in_feats = in_feats
        self.h_feats = h_feats
        self.num_classes = num_classes
        self.dist = {}

        self.linear = nn.Linear(self.in_feats, self.h_feats)
        self.MLP = nn.Linear(self.in_feats, self.num_classes)

        self.p = {}
        self.last_avg_dist = {}
        self.f = {}
        self.cvg = {}


    def _calc_distance(self, edges):
        # formula 2
        d = torch.norm(torch.tanh(self.MLP(edges.src["h"]))
            - torch.tanh(self.MLP(edges.dst["h"])), 1, 1,)
        return {"d": d}

    def _top_p_sampling(self, graph, p):
        # Compute the number of neighbors to keep for each node
        in_degrees = graph.in_degrees()
        num_neigh = torch.ceil(in_degrees.float() * p).int()

        # Fetch all edges and their distances
        all_edges = graph.edges(form="eid")
        dist = graph.edata["d"]

        # Create a prefix sum array for in-degrees to use for indexing
        prefix_sum = torch.cat([torch.tensor([0]).cuda(), in_degrees.cumsum(0)[:-1]])

        # Get the edges for each node using advanced indexing
        selected_edges = []
        for i, node_deg in enumerate(num_neigh):
            start_idx = prefix_sum[i]
            end_idx = start_idx + node_deg
            sorted_indices = torch.argsort(dist[start_idx:end_idx])[:node_deg]
            selected_edges.append(all_edges[start_idx:end_idx][sorted_indices])
        return torch.cat(selected_edges)

    def forward(self, graph, epoch=0):
        feat = graph.ndata['feature']
        edges = graph.canonical_etypes
        if epoch == 0:
            for etype in edges:
                self.p[etype] = 0.5
                self.last_avg_dist[etype] = 0
                self.f[etype] = []
                self.cvg[etype] = False

        with graph.local_scope():
            graph.ndata["h"] = feat

            hr = {}
            for i, etype in enumerate(edges):
                graph.apply_edges(self._calc_distance, etype=etype)
                self.dist[etype] = graph.edges[etype].data["d"]
                sampled_edges = self._top_p_sampling(graph[etype], self.p[etype])

                # formula 8
                graph.send_and_recv(
                    sampled_edges,
                    fn.copy_u("h", "m"),
                    fn.mean("m", "h_%s" % etype[1]),
                    etype=etype,
                )
                hr[etype] = graph.ndata["h_%s" % etype[1]]
                if self.activation is not None:
                    hr[etype] = self.activation(hr[etype])

            # formula 9 using mean as inter-relation aggregator
            p_tensor = (
                torch.Tensor(list(self.p.values())).view(-1, 1, 1).to(graph.device)
            )
            h_homo = torch.sum(torch.stack(list(hr.values())) * p_tensor, dim=0)
            h_homo += feat
            if self.activation is not None:
                h_homo = self.activation(h_homo)

            return self.linear(h_homo)


class GraphConsis(nn.Module):
    def __init__(self, in_feats, num_classes=2, h_feats=64, edges=None, num_layers=1, activation=None, step_size=0.02, **kwargs):
        super().__init__()
        self.in_feats = in_feats
        self.h_feats = h_feats
        self.num_classes = num_classes
        self.activation = None if activation is None else getattr(nn, activation)()
        self.step_size = step_size
        self.num_layers = num_layers
        self.output_linear = nn.Linear(h_feats, num_classes)
        self.layers = nn.ModuleList()
        self.layers.append(          # Input layer
            CAREConv(self.in_feats, self.num_classes, self.num_classes, activation=self.activation, step_size=self.step_size,))
        for i in range(self.num_layers - 1):  # Hidden layers with n - 2 layers
            self.layers.append(CAREConv(self.h_feats, self.h_feats, self.num_classes, activation=self.activation, step_size=self.step_size,))
            # self.layers.append(   # Output layer
                # CAREConv(self.h_feats, self.num_classes, self.num_classes, activation=self.activation, step_size=self.step_size,))

    def forward(self, graph, epoch=0):            
        for layer in self.layers:
            feat = layer(graph, epoch)
        return feat

    def RLModule(self, graph, epoch, idx):
        for layer in self.layers:
            for etype in graph.canonical_etypes:
                if not layer.cvg[etype]:
                    # formula 5
                    eid = graph.in_edges(idx, form='eid', etype=etype)
                    avg_dist = torch.mean(layer.dist[etype][eid])

                    # formula 6
                    if layer.last_avg_dist[etype] < avg_dist:
                        if layer.p[etype] - self.step_size > 0:
                            layer.p[etype] -=   self.step_size
                        layer.f[etype].append(-1)
                    else:
                        if layer.p[etype] + self.step_size <= 1:
                            layer.p[etype] += self.step_size
                        layer.f[etype].append(+1)
                    layer.last_avg_dist[etype] = avg_dist

                    # formula 7
                    if epoch >= 9 and abs(sum(layer.f[etype][-10:])) <= 2:
                        layer.cvg[etype] = True

class H2FDetector_layer(nn.Module):
    def __init__(self, in_feats, h_feats, head, relation_aware, etype, dropout_rate, if_sum=False):
        super().__init__()
        self.etype = etype
        self.head = head
        self.hd = h_feats
        self.if_sum = if_sum
        self.relation_aware = relation_aware
        self.w_liner = nn.Linear(in_feats, h_feats*head)
        self.atten = nn.Linear(2*self.hd, 1)
        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, graph, h):
        with graph.local_scope():
            graph.ndata['feat'] = h
            graph.apply_edges(self.sign_edges, etype=self.etype)
            h = self.w_liner(h)
            graph.ndata['h'] = h
            graph.update_all(message_func=self.message, reduce_func=self.reduce, etype=self.etype)
            out = graph.ndata['out']
            return out

    def message(self, edges):
        src = edges.src
        src_features = edges.data['sign'].view(-1,1)*src['h']
        src_features = src_features.view(-1, self.head, self.hd)
        z = torch.cat([src_features, edges.dst['h'].view(-1, self.head, self.hd)], dim=-1)
        alpha = self.atten(z)
        alpha = self.leakyrelu(alpha)
        return {'atten':alpha, 'sf':src_features}

    def reduce(self, nodes):
        alpha = nodes.mailbox['atten']
        sf = nodes.mailbox['sf']
        alpha = self.softmax(alpha)
        out = torch.sum(alpha*sf, dim=1)
        if not self.if_sum:
            out = out.view(-1, self.head*self.hd)
        else:
            out = out.sum(dim=-2)
        return {'out':out}

    def sign_edges(self, edges):
        src = edges.src['feat']
        dst = edges.dst['feat']
        score = self.relation_aware(src, dst)
        return {'sign':torch.sign(score)}


class RelationAware(nn.Module):
    def __init__(self, in_feats, h_feats, dropout_rate):
        super().__init__()
        self.d_liner = nn.Linear(in_feats, h_feats)
        self.f_liner = nn.Linear(3*h_feats, 1)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, src, dst):
        src = self.d_liner(src)
        dst = self.d_liner(dst)
        diff = src-dst
        e_feats = torch.cat([src, dst, diff], dim=1)
        if self.dropout is not None:
            e_feats = self.dropout(e_feats)
        score = self.f_liner(e_feats).squeeze()
        score = self.tanh(score)
        return score


def hinge_loss(labels, scores):
    margin = 1
    ls = labels*scores
    
    loss = F.relu(margin-ls)
    loss = loss.mean()
    return loss


class MultiRelationH2FDetectorLayer(nn.Module):
    def __init__(self, in_feats, h_feats, head, dataset, dropout_rate, if_sum=False):
        super().__init__()
        self.relation = copy.deepcopy(dataset.etypes)
        # self.relation.remove('net_upu')
        self.n_relation = len(self.relation)
        if not if_sum:
            self.liner = nn.Linear(self.n_relation*h_feats*head, h_feats*head)
        else:
            self.liner = nn.Linear(self.n_relation*h_feats, h_feats)
        self.relation_aware = RelationAware(in_feats, h_feats*head, dropout_rate)
        self.minelayers = nn.ModuleDict()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        for e in self.relation:
            self.minelayers[e] = H2FDetector_layer(in_feats, h_feats, head, self.relation_aware, e, dropout_rate, if_sum)
    
    
    def forward(self, graph, h):
        hs = []
        for e in self.relation:
            he = self.minelayers[e](graph, h)
            hs.append(he)
        h = torch.cat(hs, dim=1)
        if self.dropout is not None:
            h = self.dropout(h)

        h = self.liner(h)
        return h
    
    def loss(self, graph, h):
        with graph.local_scope():
            graph.ndata['feat'] = h
            agg_h = self.forward(graph, h)
            
            graph.apply_edges(self.score_edges, etype='homo')
            edges_score = graph.edges['homo'].data['score']
            edge_train_mask = graph.edges['homo'].data['train_mask'].bool()
            edge_train_label = graph.edges['homo'].data['label'][edge_train_mask]
            edge_train_pos = edge_train_label == 1
            edge_train_neg = edge_train_label == -1
            edge_train_pos_index = edge_train_pos.nonzero().flatten().detach().cpu().numpy()
            edge_train_neg_index = edge_train_neg.nonzero().flatten().detach().cpu().numpy()
            edge_train_pos_index = np.random.choice(edge_train_pos_index, size=len(edge_train_neg_index))
            index = np.concatenate([edge_train_pos_index, edge_train_neg_index])
            index.sort()
            edge_train_score = edges_score[edge_train_mask]
            # hinge loss
            edge_diff_loss = hinge_loss(edge_train_label[index], edge_train_score[index])

            train_mask = graph.ndata['train_mask'].bool()
            train_h = agg_h[train_mask]
            train_label = graph.ndata['label'][train_mask]
            train_pos = train_label==1
            train_neg = train_label==0
            train_pos_index = train_pos.nonzero().flatten().detach().cpu().numpy()
            train_neg_index = train_neg.nonzero().flatten().detach().cpu().numpy()
            train_neg_index = np.random.choice(train_neg_index, size=len(train_pos_index))
            node_index = np.concatenate([train_neg_index, train_pos_index])
            node_index.sort()
            pos_prototype = torch.mean(train_h[train_pos], dim=0).view(1,-1)
            neg_prototype = torch.mean(train_h[train_neg], dim=0).view(1,-1)
            train_h_loss = train_h[node_index]
            pos_prototypes = pos_prototype.expand(train_h_loss.shape)
            neg_prototypes = neg_prototype.expand(train_h_loss.shape)
            diff_pos = - F.pairwise_distance(train_h_loss, pos_prototypes)
            diff_neg = - F.pairwise_distance(train_h_loss, neg_prototypes)
            diff_pos = diff_pos.view(-1,1)
            diff_neg = diff_neg.view(-1,1)
            diff = torch.cat([diff_neg, diff_pos], dim=1)
            diff_loss = F.cross_entropy(diff, train_label[node_index])

            return agg_h, edge_diff_loss, diff_loss
        
    def score_edges(self, edges):
        src = edges.src['feat']
        dst = edges.dst['feat']
        score = self.relation_aware(src, dst)
        return {'score':score}
        
        
class H2FD(nn.Module):
    def __init__(self, in_feats, graph, n_layer=2, intra_dim=8, n_class=2, gamma1=0.4, gamma2=1.4, head=1, dropout_rate=0, **kwargs):
        super().__init__()
        self.in_feats = in_feats
        self.n_layer = n_layer 
        self.intra_dim = intra_dim 
        self.n_class = n_class
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.head = head
        self.dropout_rate = dropout_rate
        self.mine_layers = nn.ModuleList()
        if n_layer == 1:
            self.mine_layers.append(MultiRelationH2FDetectorLayer(self.in_feats, self.n_class, head, graph, dropout_rate, if_sum=True))
        else:
            self.mine_layers.append(MultiRelationH2FDetectorLayer(self.in_feats, self.intra_dim, head, graph, dropout_rate))
            for _ in range(1, self.n_layer-1):
                self.mine_layers.append(MultiRelationH2FDetectorLayer(self.intra_dim*head, self.intra_dim, head, graph, dropout_rate))
            self.mine_layers.append(MultiRelationH2FDetectorLayer(self.intra_dim*head, self.n_class, head, graph, dropout_rate, if_sum=True))
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.relu = nn.ReLU()
    
    def forward(self, graph):
        feats = graph.ndata['feature'].float()
        train_mask = graph.ndata['train_mask'].bool()
        train_label = graph.ndata['label'][train_mask]
        train_pos = train_label == 1
        train_neg = train_label == 0
        
        pos_index = train_pos.nonzero().flatten().detach().cpu().numpy()
        neg_index = train_neg.nonzero().flatten().detach().cpu().numpy()
        neg_index = np.random.choice(neg_index, size=len(pos_index), replace=False)
        index = np.concatenate([pos_index, neg_index])
        index.sort()
        h, edge_loss, prototype_loss = self.mine_layers[0].loss(graph, feats)
        if self.n_layer > 1:
            h = self.relu(h)
            h = self.dropout(h)
            for i in range(1, len(self.mine_layers)-1):
                h, e_loss, p_loss = self.mine_layers[i].loss(graph, h)
                h = self.relu(h)
                h = self.dropout(h)
                edge_loss += e_loss
                prototype_loss += p_loss
            h, e_loss, p_loss = self.mine_layers[-1].loss(graph, h)
            edge_loss += e_loss
            prototype_loss += p_loss
        model_loss = F.cross_entropy(h[train_mask][index], train_label[index])
        loss = model_loss + self.gamma1*edge_loss + self.gamma2*prototype_loss
        return loss, h
