import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch, add_self_loops
from torch_geometric.nn import MessagePassing
import argparse
from typing import List, Tuple, Union, Optional
from .util import init, get_clones
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size


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
            # print("node_feat_j{}, entity_embed_j{}, edge_att{}".format(node_feat_j.shape, entity_embed_j.shape, edge_attr.shape))
            node_feat = torch.cat([node_feat_j, entity_embed_j, edge_attr], dim=1)
        else:
            node_feat = torch.cat([node_feat_j, entity_embed_j], dim=1)
        x = self.lin1(node_feat)
        for i in range(self._layer_N):
            x = self.lin2[i](x)
        return x



class GNNBase(nn.Module):
    def __init__(
        self,
        args: argparse.Namespace,
        node_obs_shape: Union[List, Tuple],
        edge_dim: int,
        graph_aggr: str,
    ):
        super(GNNBase, self).__init__()
        self.max_edge_dist = args.max_edge_dist
        self.hidden_size = args.gnn_hidden_size # hidden size of the GNN, also its output dim
        self.graph_aggr = graph_aggr
        self.global_aggr_type = args.global_aggr_type
        self.input_dim = node_obs_shape  # input dim of gnn

        self.num_embeddings = args.num_embeddings  # types of agents
        self.embedding_size = args.embedding_size  # size of the entity embedding, dim of feature x
        self.embed_hidden_size = args.embed_hidden_size  # hidden size of the EmbedConv layer, output dim of EmbedConv
        self.embed_layer_N = args.embed_layer_N
        self.embed_add_self_loop = args.embed_add_self_loop
        self.edge_dim = edge_dim

        self.use_ReLU = args.gnn_use_ReLU
        self.use_layerNorm = args.use_feature_normalization
        self.use_orthogonal = args.use_orthogonal
        self.active_func = [nn.Tanh(), nn.ReLU()][self.use_ReLU]
        self.layer_norm = [nn.Identity(), nn.LayerNorm(self.hidden_size)][self.use_layerNorm]
        self.init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self.use_orthogonal]
        self.gain = nn.init.calculate_gain(["tanh", "relu"][self.use_ReLU])

        def init_(m):
            return init(m, self.init_method, lambda x: nn.init.constant_(x, 0), gain=self.gain)

        self.fc1 = EmbedConv(
            input_dim=self.input_dim - 1,
            num_embeddings=self.num_embeddings,
            embedding_size=self.embedding_size,
            hidden_size=self.embed_hidden_size,  # output dim of EmbedConv
            layer_N=self.embed_layer_N,
            use_orthogonal=self.use_orthogonal,
            use_ReLU=self.use_ReLU,
            use_layerNorm=self.use_layerNorm,
            add_self_loop= self.embed_add_self_loop,
            edge_dim=self.edge_dim,
        )

        self.fc2 = nn.Sequential(
            init_(nn.Linear(self.hidden_size, self.hidden_size)),
            self.active_func,
            nn.Linear(self.hidden_size, 1)
        )

        self.fc3 = nn.Sequential(
            init_(nn.Linear(self.hidden_size, self.hidden_size)),
            self.active_func, 
            self.layer_norm
        )
        
        self.fc4 = nn.Sequential(
            init_(nn.Linear(self.hidden_size, self.hidden_size)),
            self.active_func, 
            self.layer_norm
        )

    def forward(self, node_obs: torch.Tensor, adj: torch.Tensor, agent_id: torch.Tensor):
        batch_size = node_obs.shape[0]
        # print("batch_size: ", batch_size)
        datalist = []
        for i in range(batch_size):
            edge_index, edge_attr = self.processAdj(adj[i])
            # if edge_attr is only one dimensional
            if len(edge_attr.shape) == 1:
                edge_attr = edge_attr.unsqueeze(1)
            datalist.append(Data(x=node_obs[i], edge_index=edge_index, edge_attr=edge_attr))
        loader = DataLoader(datalist, shuffle=False, batch_size=batch_size)
        data = next(iter(loader))
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch

        if self.edge_dim is None:
            edge_attr = None

        embeddings = self.fc1(x, edge_index, edge_attr)
        edge_weight = self.attention(embeddings)
        value_embeddings = self.fc3(embeddings)
        weighted_embeddings = edge_weight * value_embeddings
        x = self.fc4(weighted_embeddings)
        
        x, mask = to_dense_batch(x, batch)

        # only pull the node-specific features from output
        if self.graph_aggr == "node":
            x = self.gatherNodeFeats(x, agent_id)  # shape [batch_size, out_channels]
        # perform global pool operation on the node features of the graph
        elif self.graph_aggr == "global":
            x = self.graphAggr(x)
        return x

    def attention(self, x):
        alpha = self.fc2(x)
        alpha = F.softmax(alpha, dim=0)
        return alpha

    def processAdj(self, adj: torch.Tensor):
        connect_mask = ((adj < self.max_edge_dist) * (adj > 0)).float()
        adj = adj * connect_mask

        index = adj.nonzero(as_tuple=True)
        edge_attr = adj[index]
        if len(index) == 3:
            batch = index[0] * adj.size(-1)
            index = (batch + index[1], batch + index[2])

        return torch.stack(index, dim=0), edge_attr

    def gatherNodeFeats(self, x: torch.Tensor, idx: torch.Tensor):
        out = []
        batch_size, num_nodes, num_feats = x.shape
        idx = idx.long()
        for i in range(idx.shape[1]):
            idx_tmp = idx[:, i].unsqueeze(-1)
            idx_tmp = idx_tmp.repeat(1, num_feats)
            idx_tmp = idx_tmp.unsqueeze(1)
            gathered_node = x.gather(1, idx_tmp).squeeze(1)
            out.append(gathered_node)
        out = torch.cat(out, dim=1)
        return out
    
    def graphAggr(self, x: torch.Tensor):
        """
        Aggregate the graph node features by performing global pool


        Args:
            x (Tensor): Tensor of shape [batch_size, num_nodes, num_feats]
            aggr (str): Aggregation method for performing the global pool

        Raises:
            ValueError: If `aggr` is not in ['mean', 'max']

        Returns:
            Tensor: The global aggregated tensor of shape [batch_size, num_feats]
        """
        if self.global_aggr_type == "mean":
            return x.mean(dim=1)
        elif self.global_aggr_type == "max":
            max_feats, idx = x.max(dim=1)
            return max_feats
        elif self.global_aggr_type == "add":
            return x.sum(dim=1)
        else:
            raise ValueError(f"`aggr` should be one of 'mean', 'max', 'add'")
        
    @property
    def out_dim(self):
        return self.hidden_size