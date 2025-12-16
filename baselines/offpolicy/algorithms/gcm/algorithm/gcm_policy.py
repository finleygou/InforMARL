import numpy as np
import torch
from baselines.offpolicy.algorithms.gcm.algorithm.gcm_agent_q import GCMAgentQ
from baselines.offpolicy.algorithms.base.recurrent_policy import RecurrentPolicy
from baselines.offpolicy.utils.util import (
    get_dim_from_space,
    is_discrete,
    is_multidiscrete,
    DecayThenFlatSchedule,
    to_torch,
)

class GCMPolicy(RecurrentPolicy):
    def __init__(self, config, policy_config, train=True):
        self.args = config["args"]
        self.device = config["device"]
        self.obs_space = policy_config["obs_space"]
        self.obs_dim = get_dim_from_space(self.obs_space)
        self.act_space = policy_config["act_space"]
        self.act_dim = get_dim_from_space(self.act_space)
        self.output_dim = sum(self.act_dim) if isinstance(self.act_dim, np.ndarray) else self.act_dim
        self.hidden_size = self.args.hidden_size
        self.central_obs_dim = policy_config["cent_obs_dim"]
        self.discrete = is_discrete(self.act_space)
        self.multidiscrete = is_multidiscrete(self.act_space)

        # Graph observation spaces
        self.node_obs_space = policy_config["node_obs_space"]
        self.edge_obs_space = policy_config["edge_obs_space"]

        if self.args.prev_act_inp:
            self.q_network_input_dim = self.obs_dim + self.act_dim
        else:
            self.q_network_input_dim = self.obs_dim

        self.q_network = GCMAgentQ(
            self.args, 
            self.obs_space, 
            self.node_obs_space, 
            self.edge_obs_space, 
            self.act_dim, 
            self.device
        )

        if train:
            self.exploration = DecayThenFlatSchedule(
                self.args.epsilon_start,
                self.args.epsilon_finish,
                self.args.epsilon_anneal_time,
                decay="linear",
            )

    def get_q_values(self, obs_batch, node_obs_batch, adj_batch, agent_id_batch, prev_action_batch, rnn_states, action_batch=None):
        if self.args.prev_act_inp:
            prev_action_batch = to_torch(prev_action_batch)
            input_batch = torch.cat((obs_batch, prev_action_batch), dim=-1)
        else:
            input_batch = obs_batch

        q_batch, new_rnn_states = self.q_network(input_batch, rnn_states, node_obs_batch, adj_batch, agent_id_batch)

        if action_batch is not None:
            action_batch = to_torch(action_batch).to(self.device)
            q_values = self.q_values_from_actions(q_batch, action_batch)
        else:
            q_values = q_batch
        return q_values, new_rnn_states

    def get_actions(self, obs, node_obs, adj, agent_id, prev_actions, rnn_states, available_actions=None, t_env=None, explore=False):
        if self.args.prev_act_inp:
            prev_actions = to_torch(prev_actions)
            input_batch = torch.cat((obs, prev_actions), dim=-1)
        else:
            input_batch = obs

        q_values, new_rnn_states = self.q_network(input_batch, rnn_states, node_obs, adj, agent_id)
        
        if self.multidiscrete:
            # Handle multidiscrete actions (omitted for brevity, similar to QMixPolicy)
            pass
        else:
            if available_actions is not None:
                available_actions = to_torch(available_actions).to(self.device)
                q_values[available_actions == 0] = -1e10

            if explore:
                epsilon = self.exploration.eval(t_env)
                if np.random.random() < epsilon:
                    if available_actions is not None:
                        actions = torch.multinomial(available_actions.float(), 1)
                    else:
                        actions = torch.randint(0, self.act_dim, (obs.shape[0], 1)).to(self.device)
                else:
                    actions = q_values.argmax(dim=-1).unsqueeze(-1)
            else:
                actions = q_values.argmax(dim=-1).unsqueeze(-1)

        return actions, new_rnn_states

    def q_values_from_actions(self, q_batch, action_batch):
        if self.multidiscrete:
            ind = 0
            all_q_values = []
            for i in range(len(self.act_dim)):
                curr_q_batch = q_batch[i]
                curr_action_portion = action_batch[:, :, ind : ind + self.act_dim[i]]
                curr_action_inds = curr_action_portion.max(dim=-1)[1]
                curr_q_values = torch.gather(curr_q_batch, 2, curr_action_inds.unsqueeze(-1))
                all_q_values.append(curr_q_values)
                ind += self.act_dim[i]
            return torch.cat(all_q_values, dim=-1)
        else:
            return torch.gather(q_batch, 2, action_batch.long())
