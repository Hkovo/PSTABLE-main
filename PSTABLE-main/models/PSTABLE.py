import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GATConv


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def reset_parameters(self):
        for layer in self.project:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

    def forward(self, z):
        # z: (N, num_meta_paths, D)
        w = self.project(z).mean(0)
        beta = torch.softmax(w, dim=0)
        beta = beta.expand((z.shape[0],) + beta.shape)
        return (beta * z).sum(1), beta[:, :, 0]


class HANLayer(nn.Module):
    def __init__(self, meta_paths, in_size, out_size, layer_num_heads, dropout, drop_ratio=0.0):
        super().__init__()
        self.gat_layers = nn.ModuleList()
        self.meta_paths = list(tuple(mp) for mp in meta_paths)
        self._cached_graph = None
        self._cached_coalesced_graph = {}
        self.drop_ratio = drop_ratio

        for _ in self.meta_paths:
            self.gat_layers.append(GATConv(
                in_feats=in_size,
                out_feats=out_size,
                num_heads=layer_num_heads,
                feat_drop=dropout,
                attn_drop=dropout,
                activation=F.elu,
                residual=False
            ))

        self.semantic_attention = SemanticAttention(
            in_size=out_size * layer_num_heads
        )
        self.semantic_weights = None
        self.semantic_embeddings = None

    def reset_parameters(self):
        for gat in self.gat_layers:
            gat.reset_parameters()
        self.semantic_attention.reset_parameters()

    def forward(self, g, h, ntype):
        semantic_embeddings = []

        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                self._cached_coalesced_graph[meta_path] = \
                    dgl.metapath_reachable_graph(g, meta_path)

        for i, meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[meta_path]
            new_g = dgl.add_self_loop(new_g)

            # Structural perturbation during training (robustness-oriented)
            if self.training:
                edge_mask = torch.rand(
                    new_g.num_edges(),
                    device=new_g.device
                ) > self.drop_ratio
                edge_subg = dgl.edge_subgraph(new_g, edge_mask)

                edge_subg = dgl.heterograph(
                    data_dict={
                        etype: edge_subg.edges(etype=etype, form='uv')
                        for etype in edge_subg.canonical_etypes
                    },
                    num_nodes_dict={
                        ntype: new_g.num_nodes(ntype)
                        for ntype in new_g.ntypes
                    }
                )
                new_g = dgl.add_self_loop(edge_subg)

            out = self.gat_layers[i](new_g, h[ntype])
            semantic_embeddings.append(out.flatten(1))

        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)
        semantic_out, beta = self.semantic_attention(semantic_embeddings)

        # Cache semantic sensitivity indicators
        self.semantic_weights = beta.detach()
        self.semantic_embeddings = semantic_embeddings.detach()

        return semantic_out


class PSTABLE(nn.Module):
    def __init__(
        self,
        meta_paths,
        in_size,
        hidden_size,
        out_size,
        num_heads,
        dropout,
        target_type="paper",
        drop_ratio=0.0
    ):
        super().__init__()
        self.target_type = target_type
        self.mod_dict = nn.ModuleDict()

        # Learnable fusion weights for multi-view sensitivity
        self.raw_structural_weight = nn.Parameter(torch.tensor(0.5))
        self.raw_predictive_weight = nn.Parameter(torch.tensor(0.5))

        self.layers = nn.ModuleList()
        self.layers.append(
            HANLayer(
                meta_paths,
                in_size,
                hidden_size,
                num_heads[0],
                dropout,
                drop_ratio=drop_ratio
            )
        )

        for l in range(1, len(num_heads)):
            self.layers.append(
                HANLayer(
                    meta_paths,
                    hidden_size * num_heads[l - 1],
                    hidden_size,
                    num_heads[l],
                    dropout,
                    drop_ratio=drop_ratio
                )
            )

        self.predict = nn.Linear(hidden_size * num_heads[-1], out_size)
        self.reset_parameters()

        self.mod_dict[self.target_type] = nn.ModuleDict({
            'layers': self.layers,
            'predict': self.predict
        })

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        self.predict.reset_parameters()

    def forward(self, h_dict, g):
        h = h_dict[self.target_type]
        for layer in self.mod_dict[self.target_type]['layers']:
            h = layer(g, {self.target_type: h}, self.target_type)
        return self.mod_dict[self.target_type]['predict'](h)

    def register_input_feature(self, h_dict):
        self.raw_node_features = {
            ntype: h_dict[ntype].clone()
            for ntype in h_dict
            if isinstance(h_dict[ntype], torch.Tensor)
        }

    def get_node_semantic_weights(self):
        """
        Return semantic sensitivity weights of nodes
        (from the last HAN layer).
        """
        node_semantic_weights = {}
        for ntype in self.mod_dict:
            last_layer = self.mod_dict[ntype]['layers'][-1]
            if hasattr(last_layer, 'semantic_weights'):
                node_semantic_weights[ntype] = last_layer.semantic_weights
        return node_semantic_weights

    def get_node_meta_path_sensitivity(self, node_indices):
        """
        Retrieve meta-path-level semantic sensitivity for given nodes.

        Args:
            node_indices (List[int])

        Returns:
            Dict[int, List[Tuple[meta_path, sensitivity_weight]]]
        """
        result = {}
        last_layer = self.mod_dict[self.target_type]['layers'][-1]
        beta = last_layer.semantic_weights
        meta_paths = last_layer.meta_paths
        g_dict = last_layer._cached_coalesced_graph

        for idx in node_indices:
            valid_paths = []
            for i, meta_path in enumerate(meta_paths):
                g = g_dict.get(meta_path)
                if g is None:
                    continue
                if idx < g.num_nodes():
                    valid_paths.append(
                        (meta_path, beta[idx, i].item())
                    )
            result[idx] = valid_paths

        return result
