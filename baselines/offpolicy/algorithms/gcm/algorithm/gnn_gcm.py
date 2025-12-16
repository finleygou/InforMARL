import numpy as np
from scipy import sparse
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import torch_geometric.nn as gnn
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, TransformerConv
from torch_geometric.utils import add_self_loops, to_dense_batch

import argparse
from typing import List, Tuple, Union, Optional
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size

from baselines.offpolicy.algorithms.utils.util import init, get_clones

"""GNN modules for GCM"""


class EmbedConv(MessagePassing):
    def __init__(
        self,
        input_dim: int,
        num_embeddings: int,
        embedding_size: int,
        hidden_size: int,
        layer_N: int,
        use_orthogonal: bool,
        use_ReLU: bool,
        use_layerNorm: bool,
        add_self_loop: bool,
        edge_dim: int = 0,
    ):
        """
            EmbedConv Layer which takes in node features, node_type (entity type)
            and the  edge features (if they exist)
            `entity_embedding` is concatenated with `node_features` and
            `edge_features` and are passed through linear layers.
            The `message_passing` is similar to GCN layer

        Args:
            input_dim (int):
                The node feature dimension
            num_embeddings (int):
                The number of embedding classes aka the number of entity types
            embedding_size (int):
                The embedding layer output size
            hidden_size (int):
                Hidden layer size of the linear layers
            layer_N (int):
                Number of linear layers for aggregation
            use_orthogonal (bool):
                Whether to use orthogonal initialization for each layer
            use_ReLU (bool):
                Whether to use reLU for each layer
            use_layerNorm (bool):
                Whether to use layerNorm for each layer
            add_self_loop (bool):
                Whether to add self loops in the graph
            edge_dim (int, optional):
                Edge feature dimension, If zero then edge features are not
                considered. Defaults to 0.
        """
        super(EmbedConv, self).__init__(aggr="add")
        self._layer_N = layer_N
        self._add_self_loops = add_self_loop
        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        layer_norm = [nn.Identity(), nn.LayerNorm(hidden_size)][use_layerNorm]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(["tanh", "relu"][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.entity_embed = nn.Embedding(num_embeddings, embedding_size)
        self.lin1 = nn.Sequential(
            init_(nn.Linear(input_dim + embedding_size + edge_dim, hidden_size)),
            active_func,
            layer_norm,
        )
        self.lin_h = nn.Sequential(
            init_(nn.Linear(hidden_size, hidden_size)), active_func, layer_norm
        )

        self.lin2 = get_clones(self.lin_h, self._layer_N)

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
    ):
        if self._add_self_loops and edge_attr is None:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j: Tensor, edge_attr: OptTensor):
        """
        The node_obs obtained from the environment
        is actually [node_features, node_num, entity_type]
        x_i' = AGG([x_j, EMB(ent_j), e_ij] : j \in \mathcal{N}(i))
        """
        node_feat_j = x_j[:, :-1]
        # dont forget to convert to torch.LongTensor
        entity_type_j = x_j[:, -1].long()
        entity_embed_j = self.entity_embed(entity_type_j)
        if edge_attr is not None:
            node_feat = torch.cat([node_feat_j, entity_embed_j, edge_attr], dim=1)
        else:
            node_feat = torch.cat([node_feat_j, entity_embed_j], dim=1)
        x = self.lin1(node_feat)
        for i in range(self._layer_N):
            x = self.lin2[i](x)
        return x


class SimplifiedAttentionNet(nn.Module):
    """
    Implementation of Communication Enhanced Network (CEN)
    Uses Multi-Head Graph Attention and Soft Attention Gate.
    """
    def __init__(self,
                input_dim: int,
                num_embeddings: int,
                embedding_size: int,
                hidden_size: int,
                layer_N: int,
                use_ReLU: bool,
                graph_aggr: str,
                global_aggr_type: str,
                embed_hidden_size: int,
                embed_layer_N: int,
                embed_use_orthogonal: bool,
                embed_use_ReLU: bool,
                embed_use_layerNorm: bool,
                embed_add_self_loop: bool,
                max_edge_dist: float,
                edge_dim: int = 1,
                num_heads: int = 3):
        super(SimplifiedAttentionNet, self).__init__()
        self.active_func = nn.ReLU() if use_ReLU else nn.Tanh()
        self.edge_dim = edge_dim
        self.max_edge_dist = max_edge_dist
        self.graph_aggr = graph_aggr
        self.global_aggr_type = global_aggr_type
        self.num_heads = num_heads

        # Embedding layer
        self.embed_layer = EmbedConv(
            input_dim=input_dim - 1,  # -1 because node_obs = [node_feat, entity_type]
            num_embeddings=num_embeddings,
            embedding_size=embedding_size,
            hidden_size=embed_hidden_size,
            layer_N=embed_layer_N,
            use_orthogonal=embed_use_orthogonal,
            use_ReLU=embed_use_ReLU,
            use_layerNorm=embed_use_layerNorm,
            add_self_loop=embed_add_self_loop,
            edge_dim=edge_dim,
        )

        # Multi-head attention
        self.attention_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_hidden_size, hidden_size),
                self.active_func,
                nn.Linear(hidden_size, hidden_size)
            ) for _ in range(num_heads)
        ])

        # Soft attention gate
        self.soft_attention_gate = nn.Sequential(
            nn.Linear(embed_hidden_size, hidden_size),
            self.active_func,
            nn.Linear(hidden_size, num_heads)
        )

    def forward(self, batch: Batch, agent_id: Tensor):
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        x = self.embed_layer(x, edge_index, edge_attr)

        # Multi-head attention
        head_outputs = [head(x) for head in self.attention_heads]
        head_outputs = torch.stack(head_outputs, dim=-1)  # [num_nodes, hidden_size, num_heads]

        # Soft attention gate
        alpha = self.soft_attention_gate(x)  # [num_nodes, num_heads]
        alpha = F.softmax(alpha, dim=-1)

        # Aggregate heads
        x = (head_outputs * alpha.unsqueeze(1)).sum(dim=-1)  # [num_nodes, hidden_size]

        # Convert to dense batch if needed (not strictly needed for gathering but good for structure)
        x_dense, mask = to_dense_batch(x, batch.batch)

        if self.graph_aggr == "node":
            return self.gatherNodeFeats(x_dense, agent_id)
        elif self.graph_aggr == "global":
            return self.graphAggr(x_dense)

        raise ValueError(f"Invalid graph_aggr: {self.graph_aggr}")

    @staticmethod
    def process_adj(adj: Tensor, max_edge_dist: float) -> Tuple[Tensor, Tensor]:
        assert adj.dim() >= 2 and adj.dim() <= 3
        assert adj.size(-1) == adj.size(-2)

        connect_mask = ((adj < max_edge_dist) & (adj > 0)).float()
        adj = adj * connect_mask
        if adj.dim() == 3:
            batch_size, num_nodes, _ = adj.shape
            edge_index = adj.nonzero(as_tuple=False)
            edge_attr = adj[edge_index[:, 0], edge_index[:, 1], edge_index[:, 2]]
            batch = edge_index[:, 0] * num_nodes
            edge_index = torch.stack([batch + edge_index[:, 1], batch + edge_index[:, 2]], dim=0)
        else:
            edge_index = adj.nonzero(as_tuple=False).t().contiguous()
            edge_attr = adj[edge_index[0], edge_index[1]]

        edge_attr = edge_attr.unsqueeze(1) if edge_attr.dim() == 1 else edge_attr
        return edge_index, edge_attr

    def gatherNodeFeats(self, x: Tensor, idx: Tensor):
        out = []
        batch_size, num_nodes, num_feats = x.shape
        idx = idx.long()
        for i in range(idx.shape[1]):
            idx_tmp = idx[:, i].unsqueeze(-1)
            idx_tmp = idx_tmp.repeat(1, num_feats).unsqueeze(1)
            gathered_node = x.gather(1, idx_tmp).squeeze(1)
            out.append(gathered_node)
        return torch.cat(out, dim=1)

    def graphAggr(self, x: Tensor):
        if self.global_aggr_type == "mean":
            return x.mean(dim=1)
        elif self.global_aggr_type == "max":
            return x.max(dim=1)[0]
        elif self.global_aggr_type == "add":
            return x.sum(dim=1)
        else:
            raise ValueError(f"Invalid global_aggr_type: {self.global_aggr_type}")


class GCMGNN(nn.Module):
    def __init__(self, args: argparse.Namespace, 
                node_obs_shape: Union[List, Tuple],
                edge_dim: int, graph_aggr: str):
        super(GCMGNN, self).__init__()

        self.args = args
        self.hidden_size = args.gnn_hidden_size
        self.heads = args.gnn_num_heads
        self.concat = args.gnn_concat_heads

        self.gnn = SimplifiedAttentionNet(
            input_dim=node_obs_shape,
            edge_dim=edge_dim,
            num_embeddings=args.num_embeddings,
            embedding_size=args.embedding_size,
            hidden_size=args.gnn_hidden_size,
            layer_N=args.gnn_layer_N,
            use_ReLU=args.gnn_use_ReLU,
            graph_aggr=graph_aggr,
            global_aggr_type=args.global_aggr_type,
            embed_hidden_size=args.embed_hidden_size,
            embed_layer_N=args.embed_layer_N,
            embed_use_orthogonal=args.use_orthogonal,
            embed_use_ReLU=args.embed_use_ReLU,
            embed_use_layerNorm=args.use_feature_normalization,
            embed_add_self_loop=args.embed_add_self_loop,
            max_edge_dist=args.max_edge_dist,
            num_heads=args.gnn_num_heads
        )
        self.out_dim = args.gnn_hidden_size

    def forward(self, node_obs: Tensor, adj: Tensor, agent_id: Tensor):
        batch_size, num_nodes, _ = node_obs.shape
        edge_index, edge_attr = SimplifiedAttentionNet.process_adj(adj, self.gnn.max_edge_dist)
        
        x = node_obs.view(-1, node_obs.size(-1))
        batch = torch.arange(batch_size, device=node_obs.device).repeat_interleave(num_nodes)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
    
        x = self.gnn(data, agent_id)

        if self.gnn.graph_aggr == "node":
            return x.view(batch_size, -1)
        return x
