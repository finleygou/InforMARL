import torch
import torch.nn as nn
from baselines.offpolicy.utils.util import to_torch
from baselines.offpolicy.algorithms.utils.mlp import MLPBase
from baselines.offpolicy.algorithms.utils.rnn import RNNBase
from baselines.offpolicy.algorithms.utils.act import ACTLayer
from baselines.offpolicy.algorithms.gcm.algorithm.gnn_gcm import GCMGNN
from onpolicy.utils.util import get_shape_from_obs_space

class GCMAgentQ(nn.Module):
    """
    GCM Agent Q-Network using GNN for observation processing.
    """
    def __init__(self, args, obs_space, node_obs_space, edge_obs_space, act_dim, device):
        super(GCMAgentQ, self).__init__()
        self.args = args
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self._use_rnn_layer = args.use_rnn_layer
        self.hidden_size = args.hidden_size
        self._gain = args.gain

        # GNN Setup
        obs_shape = get_shape_from_obs_space(obs_space)
        node_obs_shape = get_shape_from_obs_space(node_obs_space)[1]
        edge_dim = get_shape_from_obs_space(edge_obs_space)[0]

        self.gnn_base = GCMGNN(args, node_obs_shape, edge_dim, args.actor_graph_aggr)
        gnn_out_dim = self.gnn_base.out_dim
        
        # MLP Base input dimension: GNN output + original observation
        mlp_base_in_dim = gnn_out_dim + obs_shape[0]

        if self._use_rnn_layer:
            self.rnn = RNNBase(args, mlp_base_in_dim)
        else:
            self.mlp = MLPBase(args, mlp_base_in_dim)

        self.q = ACTLayer(act_dim, self.hidden_size, args.use_orthogonal, gain=self._gain)
        
        self.to(device)

    def forward(self, obs, rnn_states, node_obs, adj, agent_id):
        """
        Compute q values using GNN-processed observations.
        """
        obs = to_torch(obs).to(**self.tpdv)
        node_obs = to_torch(node_obs).to(**self.tpdv)
        adj = to_torch(adj).to(**self.tpdv)
        agent_id = to_torch(agent_id).to(**self.tpdv)
        rnn_states = to_torch(rnn_states).to(**self.tpdv)

        # GNN Forward
        gnn_out = self.gnn_base(node_obs, adj, agent_id)
        
        # Concatenate GNN output with original observation
        inp = torch.cat([obs, gnn_out], dim=-1)

        no_sequence = False
        if len(inp.shape) == 2:
            no_sequence = True
            inp = inp[None]
        if len(rnn_states.shape) == 2:
            rnn_states = rnn_states[None]

        if self._use_rnn_layer:
            rnn_outs, h_final = self.rnn(inp, rnn_states)
        else:
            rnn_outs = self.mlp(inp)
            h_final = rnn_states[0, :, :]

        q_outs = self.q(rnn_outs, no_sequence)

        return q_outs, h_final
