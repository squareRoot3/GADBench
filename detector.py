import sklearn.svm
import torch
from gnn import *
from attention import *
from dataset import *
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from dgl.nn.pytorch.factory import KNNGraph
from sklearn.cluster import KMeans


class BaseDetector(object):
    def __init__(self, train_config, model_config, data):
        self.model_config = model_config
        self.train_config = train_config
        self.data = data
        model_config['in_feats'] = self.data.graph.ndata['feature'].shape[1]
        graph = self.data.graph.to(self.train_config['device'])
        self.labels = graph.ndata['label']
        self.train_mask = graph.ndata['train_mask'].bool()
        self.val_mask = graph.ndata['val_mask'].bool()
        self.test_mask = graph.ndata['test_mask'].bool()
        self.weight = (1 - self.labels[self.train_mask]).sum().item() / self.labels[self.train_mask].sum().item()
        self.source_graph = graph
        self.best_score = -1
        self.patience_knt = 0

    def train(self):
        pass

    def eval(self, labels, probs):
        score = {}
        with torch.no_grad():
            if torch.is_tensor(labels):
                labels = labels.cpu().numpy()
            if torch.is_tensor(probs):
                probs = probs.cpu().numpy()
            score['AUROC'] = roc_auc_score(labels, probs)
            score['AUPRC'] = average_precision_score(labels, probs)
            labels = np.array(labels)
            k = labels.sum()
        score['RECK'] = sum(labels[probs.argsort()[-k:]]) / sum(labels)
        return score


class BaseGNNDetector(BaseDetector):
    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        gnn = globals()[model_config['model']]
        model_config['in_feats'] = self.data.graph.ndata['feature'].shape[1]
        self.model = gnn(**model_config).to(train_config['device'])

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_config['lr'])
        train_labels, val_labels, test_labels = self.labels[self.train_mask], \
                                                self.labels[self.val_mask], self.labels[self.test_mask]
        for e in range(self.train_config['epochs']):
            self.model.train()
            logits = self.model(self.source_graph)
            loss = F.cross_entropy(logits[self.train_mask], train_labels,
                                   weight=torch.tensor([1., self.weight], device=self.labels.device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if self.model_config['drop_rate'] > 0:
                self.model.eval()
                logits = self.model(self.source_graph)
            probs = logits.softmax(1)[:, 1]
            val_score = self.eval(val_labels, probs[self.val_mask])
            if val_score[self.train_config['metric']] > self.best_score:
                self.patience_knt = 0
                self.best_score = val_score[self.train_config['metric']]
                test_score = self.eval(test_labels, probs[self.test_mask])
                # print('Loss {:.4f}, Val AUC {:.4f}, Pre {:.4f}, RecK {:.4f}, test AUC {:.4f}, Pre {:.4f}, RecK {:.4f}'.format(
                #     loss, val_score['AUROC'], val_score['AUPRC'], val_score['RECK'],
                #     test_score['AUROC'], test_score['AUPRC'], test_score['RECK']))
            else:
                self.patience_knt += 1
                if self.patience_knt > self.train_config['patience']:
                    break
        return test_score


class SVMDetector(BaseDetector):
    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        penalty = 'l2' if 'penalty' not in self.model_config else self.model_config['penalty']
        loss = 'squared_hinge' if 'loss' not in self.model_config else self.model_config['loss']
        C = 1 if 'C' not in self.model_config else self.model_config['C']
        self.model = svm.LinearSVC(penalty=penalty, loss=loss, C=C)

    def train(self):
        train_X = self.source_graph.ndata['feature'][self.train_mask].cpu().numpy()
        train_y = self.source_graph.ndata['label'][self.train_mask].cpu().numpy()
        val_X = self.source_graph.ndata['feature'][self.val_mask].cpu().numpy()
        val_y = self.source_graph.ndata['label'][self.val_mask].cpu().numpy()
        test_X = self.source_graph.ndata['feature'][self.test_mask].cpu().numpy()
        test_y = self.source_graph.ndata['label'][self.test_mask].cpu().numpy()
        self.model.fit(train_X, train_y)
        pred_val_y = self.model.decision_function(val_X)
        pred_y = self.model.decision_function(test_X)
        val_score = self.eval(val_y, pred_val_y)
        self.best_score = val_score[self.train_config['metric']]
        test_score = self.eval(test_y, pred_y)
        return test_score


class KNNDetector(BaseDetector):
    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        k = 5 if 'k' not in self.model_config else self.model_config['k']
        weights = 'uniform' if 'weights' not in self.model_config else self.model_config['weights']
        p = 2 if 'p' not in self.model_config else self.model_config['p']
        self.model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=k, weights=weights, p=p, n_jobs=32)

    def train(self):
        train_X = self.source_graph.ndata['feature'][self.train_mask].cpu().numpy()
        train_y = self.source_graph.ndata['label'][self.train_mask].cpu().numpy()
        val_X = self.source_graph.ndata['feature'][self.val_mask].cpu().numpy()
        val_y = self.source_graph.ndata['label'][self.val_mask].cpu().numpy()
        test_X = self.source_graph.ndata['feature'][self.test_mask].cpu().numpy()
        test_y = self.source_graph.ndata['label'][self.test_mask].cpu().numpy()
        self.model.fit(train_X, train_y)
        pred_val_y = self.model.predict_proba(val_X)[:, 1]
        pred_y = self.model.predict_proba(test_X)[:, 1]
        val_score = self.eval(val_y, pred_val_y)
        self.best_score = val_score[self.train_config['metric']]
        test_score = self.eval(test_y, pred_y)
        return test_score


class XGBODDetector(BaseDetector):
    def __init__(self, train_config, model_config, data):
        from pyod.models.xgbod import XGBOD
        super().__init__(train_config, model_config, data)
        self.model = XGBOD(n_jobs=32, **model_config)

    def train(self):
        # tofix .cpu() should be removed
        train_X = self.source_graph.ndata['feature'][self.train_mask].cpu().numpy()
        train_y = self.source_graph.ndata['label'][self.train_mask].cpu().numpy()
        if self.train_mask.sum() > 100000: # avoid out of time
            train_X = train_X[:2000]
            train_y = train_y[:2000]
        print(train_X.shape, train_y.shape)
        val_X = self.source_graph.ndata['feature'][self.val_mask].cpu().numpy()
        val_y = self.source_graph.ndata['label'][self.val_mask].cpu().numpy()
        test_X = self.source_graph.ndata['feature'][self.test_mask].cpu().numpy()
        test_y = self.source_graph.ndata['label'][self.test_mask].cpu().numpy()
        self.model.fit(train_X, train_y)
        pred_val_y = self.model.decision_function(val_X)
        pred_y = self.model.decision_function(test_X)
        val_score = self.eval(val_y, pred_val_y)
        self.best_score = val_score[self.train_config['metric']]
        test_score = self.eval(test_y, pred_y)
        return test_score


class XGBoostDetector(BaseDetector):
    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        import xgboost as xgb
        eval_metric = roc_auc_score if train_config['metric'] == "AUROC" else average_precision_score
        self.model = xgb.XGBClassifier(tree_method='gpu_hist', eval_metric=eval_metric, **model_config)

    def train(self):
        train_X = self.source_graph.ndata['feature'][self.train_mask].cpu().numpy()
        train_y = self.source_graph.ndata['label'][self.train_mask].cpu().numpy()
        val_X = self.source_graph.ndata['feature'][self.val_mask].cpu().numpy()
        val_y = self.source_graph.ndata['label'][self.val_mask].cpu().numpy()
        test_X = self.source_graph.ndata['feature'][self.test_mask].cpu().numpy()
        test_y = self.source_graph.ndata['label'][self.test_mask].cpu().numpy()
        weights = np.where(train_y == 0, 1, self.weight)

        self.model.fit(train_X, train_y, sample_weight=weights, eval_set=[(val_X, val_y)], verbose=False)
        pred_val_y = self.model.predict_proba(val_X)[:, 1]
        pred_y = self.model.predict_proba(test_X)[:, 1]
        val_score = self.eval(val_y, pred_val_y)
        self.best_score = val_score[self.train_config['metric']]
        test_score = self.eval(test_y, pred_y)
        return test_score


class XGBGraphDetector(BaseDetector):
    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        import xgboost as xgb
        eval_metric = roc_auc_score if train_config['metric'] == "AUROC" else average_precision_score
        self.model = xgb.XGBClassifier(tree_method='gpu_hist', eval_metric=eval_metric, verbose=2, **model_config)
        gnn = GIN_noparam(**model_config).to(self.source_graph.device)
        self.source_graph.ndata['feature'] = gnn(self.source_graph)

    def train(self):
        train_X = self.source_graph.ndata['feature'][self.train_mask].cpu().numpy()
        train_y = self.source_graph.ndata['label'][self.train_mask].cpu().numpy()
        val_X = self.source_graph.ndata['feature'][self.val_mask].cpu().numpy()
        val_y = self.source_graph.ndata['label'][self.val_mask].cpu().numpy()
        test_X = self.source_graph.ndata['feature'][self.test_mask].cpu().numpy()
        test_y = self.source_graph.ndata['label'][self.test_mask].cpu().numpy()
        weights = np.where(train_y == 0, 1, self.weight)

        self.model.fit(train_X, train_y, sample_weight=weights, eval_set=[(val_X, val_y)])  # early_stopping_rounds =20
        pred_val_y = self.model.predict_proba(val_X)[:, 1]
        pred_y = self.model.predict_proba(test_X)[:, 1]
        val_score = self.eval(val_y, pred_val_y)
        self.best_score = val_score[self.train_config['metric']]
        test_score = self.eval(test_y, pred_y)
        return test_score


class RFDetector(BaseDetector):
    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        # eval_metric = roc_auc_score if train_config['metric'] == "AUROC" else average_precision_score
        n_estimators = 100 if 'n_estimators' not in model_config else model_config['n_estimators']
        criterion = 'gini' if 'criterion' not in model_config else model_config['criterion']
        max_samples = None if 'max_samples' not in model_config else model_config['max_samples']
        max_features = 'sqrt' if 'max_features' not in model_config else model_config['max_features']
        self.model = RandomForestClassifier(n_jobs=32, n_estimators=n_estimators, criterion=criterion,
                                            max_samples=max_samples, max_features=max_features)

    def train(self):
        train_X = self.source_graph.ndata['feature'][self.train_mask].cpu().numpy()
        train_y = self.source_graph.ndata['label'][self.train_mask].cpu().numpy()
        val_X = self.source_graph.ndata['feature'][self.val_mask].cpu().numpy()
        val_y = self.source_graph.ndata['label'][self.val_mask].cpu().numpy()
        test_X = self.source_graph.ndata['feature'][self.test_mask].cpu().numpy()
        test_y = self.source_graph.ndata['label'][self.test_mask].cpu().numpy()
        weights = np.where(train_y == 0, 1, self.weight)
        self.model.fit(train_X, train_y, sample_weight=weights)
        pred_val_y = self.model.predict_proba(val_X)[:, 1]
        pred_y = self.model.predict_proba(test_X)[:, 1]
        val_score = self.eval(val_y, pred_val_y)
        self.best_score = val_score[self.train_config['metric']]
        test_score = self.eval(test_y, pred_y)
        return test_score


class RFGraphDetector(BaseDetector):
    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        # eval_metric = roc_auc_score if train_config['metric'] == "AUROC" else average_precision_score
        n_estimators = 100 if 'n_estimators' not in model_config else model_config['n_estimators']
        criterion = 'gini' if 'criterion' not in model_config else model_config['criterion']
        max_samples = None if 'max_samples' not in model_config else model_config['max_samples']
        max_features = 'sqrt' if 'max_features' not in model_config else model_config['max_features']
        self.model = RandomForestClassifier(n_jobs=32, n_estimators=n_estimators, criterion=criterion,
                                            max_samples=max_samples, max_features=max_features)
        gnn = GIN_noparam(**model_config).to(self.source_graph.device)
        self.source_graph.ndata['feature'] = gnn(self.source_graph)

    def train(self):
        train_X = self.source_graph.ndata['feature'][self.train_mask].cpu().numpy()
        train_y = self.source_graph.ndata['label'][self.train_mask].cpu().numpy()
        val_X = self.source_graph.ndata['feature'][self.val_mask].cpu().numpy()
        val_y = self.source_graph.ndata['label'][self.val_mask].cpu().numpy()
        test_X = self.source_graph.ndata['feature'][self.test_mask].cpu().numpy()
        test_y = self.source_graph.ndata['label'][self.test_mask].cpu().numpy()
        weights = np.where(train_y == 0, 1, self.weight)
        self.model.fit(train_X, train_y, sample_weight=weights)
        pred_val_y = self.model.predict_proba(val_X)[:, 1]
        pred_y = self.model.predict_proba(test_X)[:, 1]
        val_score = self.eval(val_y, pred_val_y)
        self.best_score = val_score[self.train_config['metric']]
        test_score = self.eval(test_y, pred_y)
        return test_score


class GASDetector(BaseDetector):
    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        # gnn = globals()[model_config['model']]
        model_config['in_feats'] = self.data.graph.ndata['feature'].shape[1]
        model_config['mlp_layers'] = 0
        self.model = GCN(**model_config).to(train_config['device'])
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.model.h_feats * 2, self.model.h_feats),
            torch.nn.ReLU(),
            torch.nn.Linear(self.model.h_feats, 2)).to(train_config['device'])

    def train(self):
        k = 5 if 'k' not in self.model_config else self.model_config['k']
        dist = 'cosine' if 'dist' not in self.model_config else self.model_config['dist']
        feat = self.data.graph.ndata['feature'].to(self.train_config['device'])
        knn_graph = KNNGraph(k)
        knn_g = knn_graph(feat, algorithm="bruteforce-sharemem", dist=dist)
        knn_g.ndata["feature"] = feat

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_config['lr'])
        train_labels, val_labels, test_labels = self.labels[self.train_mask], \
                                                self.labels[self.val_mask], self.labels[self.test_mask]

        for e in range(self.train_config['epochs']):
            self.model.train()
            h_origin = self.model(self.source_graph)
            h_knn = self.model(knn_g)
            h_all = torch.cat([h_origin, h_knn], -1)
            logits = self.mlp(h_all)
            loss = F.cross_entropy(logits[self.train_mask], train_labels,
                                   weight=torch.tensor([1., self.weight], device=self.labels.device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if self.model_config['drop_rate'] > 0:
                self.model.eval()
                logits = self.model(self.source_graph)
            probs = logits.softmax(1)[:, 1]
            val_score = self.eval(val_labels, probs[self.val_mask])
            if val_score[self.train_config['metric']] > self.best_score:
                self.patience_knt = 0
                self.best_score = val_score[self.train_config['metric']]
                test_score = self.eval(test_labels, probs[self.test_mask])
            else:
                self.patience_knt += 1
                if self.patience_knt > self.train_config['patience']:
                    break
            # print('Loss {:.4f}, Val AUC {:.4f}, Pre {:.4f}, RecK {:.4f}, test AUC {:.4f}, Pre {:.4f}, RecK {:.4f}'.format(
            #     loss, val_score['AUROC'], val_score['AUPRC'], val_score['RECK'],
            #     test_score['AUROC'], test_score['AUPRC'], test_score['RECK']))
        return test_score


class KNNGCNDetector(BaseDetector):
    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        # gnn = globals()[model_config['model']]
        model_config['in_feats'] = self.data.graph.ndata['feature'].shape[1]
        self.model = GCN(**model_config).to(train_config['device'])

    def train(self):
        k = 5 if 'k' not in self.model_config else self.model_config['k']
        dist = 'cosine' if 'dist' not in self.model_config else self.model_config['dist']
        feat = self.data.graph.ndata['feature'].to(self.train_config['device'])
        knn_graph = KNNGraph(k)
        knn_g = knn_graph(feat, algorithm="bruteforce-sharemem", dist=dist)
        new_g = dgl.merge([knn_g, self.source_graph])
        print(new_g)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_config['lr'])
        train_labels, val_labels, test_labels = self.labels[self.train_mask], \
                                                self.labels[self.val_mask], self.labels[self.test_mask]

        for e in range(self.train_config['epochs']):
            self.model.train()
            logits = self.model(new_g)
            loss = F.cross_entropy(logits[self.train_mask], train_labels,
                                   weight=torch.tensor([1., self.weight], device=self.labels.device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if self.model_config['drop_rate'] > 0:
                self.model.eval()
                logits = self.model(self.source_graph)
            if self.model_config['drop_rate'] > 0:
                self.model.eval()
                logits = self.model(self.source_graph)
            probs = logits.softmax(1)[:, 1]
            val_score = self.eval(val_labels, probs[self.val_mask])
            if val_score[self.train_config['metric']] > self.best_score:
                self.patience_knt = 0
                self.best_score = val_score[self.train_config['metric']]
                test_score = self.eval(test_labels, probs[self.test_mask])
            else:
                self.patience_knt += 1
                if self.patience_knt > self.train_config['patience']:
                    break
            # print('Loss {:.4f}, Val AUC {:.4f}, Pre {:.4f}, RecK {:.4f}, test AUC {:.4f}, Pre {:.4f}, RecK {:.4f}'.format(
            #     loss, val_score['AUROC'], val_score['AUPRC'], val_score['RECK'],
            #     test_score['AUROC'], test_score['AUPRC'], test_score['RECK']))
        return test_score


class GHRNDetector(BaseDetector):
    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        model_config['in_feats'] = self.data.graph.ndata['feature'].shape[1]
        self.model = BWGNN(**model_config).to(train_config['device'])

    def random_walk_update(self, delete_ratio):
        graph = self.source_graph
        edge_weight = torch.ones(graph.num_edges()).to(self.train_config['device'])
        norm = dgl.nn.pytorch.conv.EdgeWeightNorm(norm='both')
        graph.edata['w'] = norm(graph, edge_weight)
        aggregate_fn = fn.u_mul_e('h', 'w', 'm')
        reduce_fn = fn.sum(msg='m', out='ay')

        graph.ndata['h'] = graph.ndata['feature']
        graph.update_all(aggregate_fn, reduce_fn)
        graph.ndata['ly'] = graph.ndata['feature'] - graph.ndata['ay']
        graph.apply_edges(self.inner_product_black)
        black = graph.edata['inner_black']
        threshold = int(delete_ratio * graph.num_edges())
        edge_to_move = set(black.sort()[1][:threshold].tolist())
        graph_new = dgl.remove_edges(graph, list(edge_to_move))
        return graph_new

    def inner_product_black(self, edges):
        inner_black = (edges.src['ly'] * edges.dst['ly']).sum(axis=1)
        return {'inner_black': inner_black}

    def train(self):
        del_ratio = 0.015 if 'del_ratio' not in self.model_config else self.model_config['del_ratio']
        adj_type = 'sym' if 'adj_type' not in self.model_config else self.model_config['adj_type']
        if del_ratio != 0.:
            graph = self.random_walk_update(del_ratio)
            graph = dgl.add_self_loop(dgl.remove_self_loop(graph))

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_config['lr'])
        train_labels, val_labels, test_labels = self.labels[self.train_mask], \
                                                self.labels[self.val_mask], self.labels[self.test_mask]
        for e in range(self.train_config['epochs']):
            self.model.train()
            logits = self.model(graph)
            loss = F.cross_entropy(logits[self.train_mask], train_labels,
                                   weight=torch.tensor([1., self.weight], device=self.labels.device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            probs = logits.softmax(1)[:, 1]
            val_score = self.eval(val_labels, probs[self.val_mask])
            if val_score[self.train_config['metric']] > self.best_score:
                self.patience_knt = 0
                self.best_score = val_score[self.train_config['metric']]
                test_score = self.eval(test_labels, probs[self.test_mask])
                print('Loss {:.4f}, Val AUC {:.4f}, Pre {:.4f}, RecK {:.4f}, test AUC {:.4f}, Pre {:.4f}, RecK {:.4f}'.format(
                    loss, val_score['AUROC'], val_score['AUPRC'], val_score['RECK'],
                    test_score['AUROC'], test_score['AUPRC'], test_score['RECK']))
            else:
                self.patience_knt += 1
                if self.patience_knt > self.train_config['patience']:
                    break
        return test_score


class PCGNNDetector(BaseDetector):
    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        model_config['in_feats'] = self.data.graph.ndata['feature'].shape[1]
        self.model = ChebNet(**model_config).to(train_config['device'])
        # self.model = BWGNN(**model_config).to(train_config['device'])

    def preprocess(self, del_ratio=0.7, add_ratio=0.3, k_max=5, dist='cosine', **kwargs):
        graph = self.source_graph.long()
        features = graph.ndata['feature']
        # features = (features - features.min(0)[0])/(features.max(0)[0]-features.min(0)[0])
        # print(features)
        # graph.ndata['feature'] = features
        # the choose step: remove edges
        edges = graph.adj().coalesce()
        edges_num = edges.indices().shape[1]
        dd = torch.zeros([edges_num], device=features.device)
        step = 5000000
        idx = 0
        while idx < edges_num:  # avoid OOM on large datasets
            st = idx
            idx += step
            ed = idx if idx < edges_num else edges_num
            f1 = features[edges.indices()[0, st:ed]]
            f2 = features[edges.indices()[1, st:ed]]
            dd[st:ed] = (f1 - f2).norm(1, dim=1).detach().clone()
        selected_edges = (dd).topk(int(edges_num * del_ratio)).indices.long()
        # print(selected_edges)
        graph = dgl.remove_edges(graph, selected_edges)
        selected_nodes = (graph.ndata['label'] == 1) & (graph.ndata['train_mask'] == 1)

        # the choose step: add edges
        g_id = selected_nodes.nonzero().squeeze(-1)
        ave_degree = graph.in_degrees(g_id).float().mean() * add_ratio
        k = min(int(ave_degree), k_max) + 1

        knn_g = dgl.knn_graph(graph.ndata['feature'][selected_nodes], algorithm="bruteforce-sharemem",
                              k=k, dist=dist)
        u, v = g_id[knn_g.edges()[0]], g_id[knn_g.edges()[1]]
        graph = dgl.add_edges(graph, u.long(), v.long())
        return graph

    def train(self):
        graph = self.preprocess(**self.model_config)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_config['lr'])
        train_labels, val_labels, test_labels = self.labels[self.train_mask], \
                                                self.labels[self.val_mask], self.labels[self.test_mask]
        for e in range(self.train_config['epochs']):
            self.model.train()
            logits = self.model(graph)
            loss = F.cross_entropy(logits[self.train_mask], train_labels,
                                   weight=torch.tensor([1., self.weight], device=self.labels.device), reduce=False)
            degree_weight = graph.in_degrees()
            loss = (loss * degree_weight[self.train_mask]).mean() / degree_weight.max()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if self.model_config['drop_rate'] > 0:
                self.model.eval()
                logits = self.model(self.source_graph)
            probs = logits.softmax(1)[:, 1]
            val_score = self.eval(val_labels, probs[self.val_mask])
            if val_score[self.train_config['metric']] > self.best_score:
                self.patience_knt = 0
                self.best_score = val_score[self.train_config['metric']]
                test_score = self.eval(test_labels, probs[self.test_mask])
                # print(
                #     'Loss {:.4f}, Val AUC {:.4f}, Pre {:.4f}, RecK {:.4f}, test AUC {:.4f}, Pre {:.4f}, RecK {:.4f}'.format(
                #         loss, val_score['AUROC'], val_score['AUPRC'], val_score['RECK'],
                #         test_score['AUROC'], test_score['AUPRC'], test_score['RECK']))
            else:
                self.patience_knt += 1
                if self.patience_knt > self.train_config['patience']:
                    break
        return test_score


class DCIDetector(BaseDetector):
    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        gnn = globals()[model_config['model']]
        print(gnn)
        model_config['in_feats'] = self.data.graph.ndata['feature'].shape[1]
        self.model = gnn(**model_config).to(train_config['device'])
        self.num_cluster = 2 if 'num_cluster' not in model_config else model_config['num_cluster']
        self.pretrain_epochs = 100 if 'pretrain_epochs' not in model_config else model_config['pretrain_epochs']

        self.kmeans = KMeans(n_clusters=self.num_cluster, random_state=0).fit(self.data.graph.ndata['feature'])
        self.ss_label = self.kmeans.labels_
        self.cluster_info = [list(np.where(self.ss_label == i)[0]) for i in range(self.num_cluster)]

    def train(self):
        train_labels, val_labels, test_labels = self.labels[self.train_mask], \
                                                self.labels[self.val_mask], self.labels[self.test_mask]

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        for e in range(1, self.pretrain_epochs):
            self.model.train()
            loss = self.model(self.source_graph, self.cluster_info, self.num_cluster)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                print(loss)
                optimizer.step()
            # re-clustering
            if e % 20 == 0:
                self.model.eval()
                emb = self.model.get_emb(self.source_graph)
                kmeans = KMeans(n_clusters=self.num_cluster, random_state=0).fit(emb.detach().cpu().numpy())
                ss_label = kmeans.labels_
                self.cluster_info = [list(np.where(ss_label == i)[0]) for i in range(self.num_cluster)]
        # print('Pre-training done!')

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_config['lr'])
        for e in range(self.train_config['epochs']):
            self.model.train()
            logits = self.model.encoder(self.source_graph, use_mlp=True)
            loss = F.cross_entropy(logits[self.train_mask], train_labels,
                                   weight=torch.tensor([1., self.weight], device=self.labels.device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if self.model_config['drop_rate'] > 0:
                self.model.eval()
                logits = self.model.encoder(self.source_graph)
            probs = logits.softmax(1)[:, 1]
            val_score = self.eval(val_labels, probs[self.val_mask])

            if val_score[self.train_config['metric']] > self.best_score:
                self.patience_knt = 0
                self.best_score = val_score[self.train_config['metric']]
                test_score = self.eval(test_labels, probs[self.test_mask])
                print(
                    'Loss {:.4f}, Val AUC {:.4f}, Pre {:.4f}, RecK {:.4f}, test AUC {:.4f}, Pre {:.4f}, RecK {:.4f}'.format(
                        loss, val_score['AUROC'], val_score['AUPRC'], val_score['RECK'],
                        test_score['AUROC'], test_score['AUPRC'], test_score['RECK']))
            else:
                self.patience_knt += 1
                if self.patience_knt > self.train_config['patience']:
                    break
        return test_score