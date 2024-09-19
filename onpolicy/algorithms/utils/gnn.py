import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, to_dense_batch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


class GNNBase(nn.Module):
    def __init__(self, in_channels, out_channels, device='cpu'):
        super(GNNBase, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.fc4 = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, out_channels)
        )
        self.to(device)

    def forward(self, node_obs: torch.Tensor, adj: torch.Tensor, agent_id: torch.Tensor):
        """
        node_obs: Tensor shape:(batch_size, num_nodes, node_obs_dim)
        adj: Tensor shape:(batch_size, num_nodes, num_nodes)
        agent_id: Tensor shape:(batch_size)
        """
        batch_size = node_obs.shape[0]
        datalist = []
        for i in range(batch_size):
            edge_index, edge_attr = self.processAdj(adj[i])
            datalist.append(
                Data(x=node_obs[i], edge_index=edge_index, edge_attr=edge_attr)
            )
        loader = DataLoader(datalist, shuffle=False, batch_size=batch_size)
        data = next(iter(loader))
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch

        embeddings = self.fc1(edge_attr)
        edge_weight = self.attention(embeddings)
        value_embeddings = self.fc3(embeddings)
        weighted_embeddings = torch.mm(edge_weight.T, value_embeddings)
        x = self.fc4(weighted_embeddings)
        
        x, mask = to_dense_batch(x, batch)
        x = self.gatherNodeFeats(x, agent_id)
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