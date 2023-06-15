import torch
from torch import nn
from dgl import ops
from dgl.nn.functional import edge_softmax


class ResidualModuleWrapper(nn.Module):
    def __init__(self, module, normalization, dim, **kwargs):
        super().__init__()
        self.normalization = normalization(dim)
        self.module = module(dim=dim, **kwargs)

    def forward(self, graph, h):
        h_res = self.normalization(h)
        h_res = self.module(graph, h_res)
        h = h + h_res
        return h


class FeedForwardModule(nn.Module):
    def __init__(self, dim, hidden_dim_multiplier, drop_rate=0, input_dim_multiplier=1, **kwargs):
        super().__init__()
        input_dim = int(dim * input_dim_multiplier)
        hidden_dim = int(dim * hidden_dim_multiplier)
        self.linear_1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.dropout_1 = nn.Dropout(drop_rate)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(in_features=hidden_dim, out_features=dim)
        self.dropout_2 = nn.Dropout(drop_rate)

    def forward(self, graph, h):
        h = self.linear_1(h)
        h = self.dropout_1(h)
        h = self.act(h)
        h = self.linear_2(h)
        h = self.dropout_2(h)

        return h


class TransformerAttentionModule(nn.Module):
    def __init__(self, dim, num_heads, drop_rate=0, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.attn_query = nn.Linear(in_features=dim, out_features=dim)
        self.attn_key = nn.Linear(in_features=dim, out_features=dim)
        self.attn_value = nn.Linear(in_features=dim, out_features=dim)

        self.output_linear = nn.Linear(in_features=dim, out_features=dim)
        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0 else None

    def forward(self, graph, h):
        queries = self.attn_query(h)
        keys = self.attn_key(h)
        values = self.attn_value(h)

        queries = queries.reshape(-1, self.num_heads, self.head_dim)
        keys = keys.reshape(-1, self.num_heads, self.head_dim)
        values = values.reshape(-1, self.num_heads, self.head_dim)

        attn_scores = ops.u_dot_v(graph, queries, keys) / self.head_dim ** 0.5
        attn_probs = edge_softmax(graph, attn_scores)

        h = ops.u_mul_e_sum(graph, values, attn_probs)
        h = h.reshape(-1, self.dim)

        h = self.output_linear(h)
        return h


class TransformerAttentionSepModule(nn.Module):
    def __init__(self, dim, num_heads, drop_rate=0, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.attn_query = nn.Linear(in_features=dim, out_features=dim)
        self.attn_key = nn.Linear(in_features=dim, out_features=dim)
        self.attn_value = nn.Linear(in_features=dim, out_features=dim)

        self.output_linear = nn.Linear(in_features=dim * 2, out_features=dim)
        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0 else None

    def forward(self, graph, h):
        queries = self.attn_query(h)
        keys = self.attn_key(h)
        values = self.attn_value(h)

        queries = queries.reshape(-1, self.num_heads, self.head_dim)
        keys = keys.reshape(-1, self.num_heads, self.head_dim)
        values = values.reshape(-1, self.num_heads, self.head_dim)

        attn_scores = ops.u_dot_v(graph, queries, keys) / self.head_dim ** 0.5
        attn_probs = edge_softmax(graph, attn_scores)

        message = ops.u_mul_e_sum(graph, values, attn_probs)
        message = message.reshape(-1, self.dim)
        h = torch.cat([h, message], axis=1)

        h = self.output_linear(h)
        # h = self.dropout(h)

        return h


class GATModule(nn.Module):
    def __init__(self, dim, hidden_dim_multiplier, num_heads, drop_rate=0, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.input_linear = nn.Linear(in_features=dim, out_features=dim)
        self.attn_linear_u = nn.Linear(in_features=dim, out_features=num_heads)
        self.attn_linear_v = nn.Linear(in_features=dim, out_features=num_heads, bias=False)
        self.attn_act = nn.LeakyReLU(negative_slope=0.2)

        self.feed_forward_module = FeedForwardModule(dim=dim,
                                                     hidden_dim_multiplier=hidden_dim_multiplier,
                                                     drop_rate=drop_rate)

    def forward(self, graph, h):
        h = self.input_linear(h)

        attn_scores_u = self.attn_linear_u(h)
        attn_scores_v = self.attn_linear_v(h)
        attn_scores = ops.u_add_v(graph, attn_scores_u, attn_scores_v)
        attn_scores = self.attn_act(attn_scores)
        attn_probs = edge_softmax(graph, attn_scores)

        h = h.reshape(-1, self.head_dim, self.num_heads)
        h = ops.u_mul_e_sum(graph, h, attn_probs)
        h = h.reshape(-1, self.dim)

        h = self.feed_forward_module(graph, h)

        return h


class GATSepModule(nn.Module):
    def __init__(self, dim, hidden_dim_multiplier, num_heads, drop_rate=0, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.input_linear = nn.Linear(in_features=dim, out_features=dim)
        self.attn_linear_u = nn.Linear(in_features=dim, out_features=num_heads)
        self.attn_linear_v = nn.Linear(in_features=dim, out_features=num_heads, bias=False)
        self.attn_act = nn.LeakyReLU(negative_slope=0.2)

        self.feed_forward_module = FeedForwardModule(dim=dim, input_dim_multiplier=2,
                                                     hidden_dim_multiplier=hidden_dim_multiplier, drop_rate=drop_rate)

    def forward(self, graph, h):
        h = self.input_linear(h)

        attn_scores_u = self.attn_linear_u(h)
        attn_scores_v = self.attn_linear_v(h)
        attn_scores = ops.u_add_v(graph, attn_scores_u, attn_scores_v)
        attn_scores = self.attn_act(attn_scores)
        attn_probs = edge_softmax(graph, attn_scores)

        h = h.reshape(-1, self.head_dim, self.num_heads)
        message = ops.u_mul_e_sum(graph, h, attn_probs)
        h = h.reshape(-1, self.dim)
        message = message.reshape(-1, self.dim)
        h = torch.cat([h, message], axis=1)

        h = self.feed_forward_module(graph, h)

        return h


class BaseModel(nn.Module):
    def __init__(self, in_feats, h_feats=32, num_layers=2, hidden_dim_multiplier=1, num_heads=4, num_classes=2,
                 drop_rate=0, **kwargs):
        super().__init__()
        self.in_feats = in_feats
        self.hidden_dim = h_feats
        self.num_layers = num_layers
        self.hidden_dim_multiplier = hidden_dim_multiplier
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.normalization = nn.Identity
        self.input_linear = nn.Linear(in_feats, h_feats)
        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0 else None
        self.act = nn.GELU()
        self.output_linear = nn.Linear(h_feats, num_classes)

    def forward(self, graph):
        h = graph.ndata['feature']
        h = self.input_linear(h)
        h = self.act(h)
        for residual_module in self.layers:
            h = residual_module(graph, h)
        if self.dropout is not None:
            h = self.dropout(h)
        h = self.output_linear(h).squeeze(1)
        return h


class GT(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            for module in [TransformerAttentionModule, FeedForwardModule]:
                residual_module = ResidualModuleWrapper(module=module, normalization=self.normalization,
                                                        dim=self.hidden_dim,
                                                        hidden_dim_multiplier=self.hidden_dim_multiplier,
                                                        num_heads=self.num_heads)
                self.layers.append(residual_module)


class GTSep(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            for module in [TransformerAttentionSepModule, FeedForwardModule]:
                residual_module = ResidualModuleWrapper(module=module, normalization=self.normalization,
                                                        dim=self.hidden_dim,
                                                        hidden_dim_multiplier=self.hidden_dim_multiplier,
                                                        num_heads=self.num_heads)
                self.layers.append(residual_module)


class GAT(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            for module in [GATModule]:
                residual_module = ResidualModuleWrapper(module=module, normalization=self.normalization,
                                                        dim=self.hidden_dim,
                                                        hidden_dim_multiplier=self.hidden_dim_multiplier,
                                                        num_heads=self.num_heads)
                self.layers.append(residual_module)


class GATSep(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            for module in [GATSepModule]:
                residual_module = ResidualModuleWrapper(module=module, normalization=self.normalization,
                                                        dim=self.hidden_dim,
                                                        hidden_dim_multiplier=self.hidden_dim_multiplier,
                                                        num_heads=self.num_heads)
                self.layers.append(residual_module)
