import numpy as np
from baselines.offpolicy.utils.rec_buffer import RecReplayBuffer, RecPolicyBuffer
from baselines.offpolicy.utils.util import get_dim_from_space

class GraphRecPolicyBuffer(RecPolicyBuffer):
    def __init__(self, buffer_size, episode_length, num_agents, obs_space, share_obs_space, act_space, 
                 node_obs_space, edge_obs_space, use_same_share_obs, use_avail_acts, use_reward_normalization=False):
        super(GraphRecPolicyBuffer, self).__init__(buffer_size, episode_length, num_agents, obs_space, share_obs_space, act_space, use_same_share_obs, use_avail_acts, use_reward_normalization)
        
        # Graph observations
        # node_obs_space is likely (num_nodes, node_feat_dim)
        # edge_obs_space is likely (edge_dim,)
        
        # We need to know num_nodes. It might be dynamic or fixed max.
        # Assuming fixed max for buffer storage.
        # node_obs_space shape from gym space
        
        if isinstance(node_obs_space, list):
             node_obs_shape = node_obs_space
        else:
             node_obs_shape = node_obs_space.shape
             
        # node_obs_shape is (num_nodes, node_feat_dim)
        self.num_nodes = node_obs_shape[0]
        self.node_feat_dim = node_obs_shape[1]
        
        self.node_obs = np.zeros((self.episode_length + 1, self.buffer_size, self.num_agents, self.num_nodes, self.node_feat_dim), dtype=np.float32)
        self.adj = np.zeros((self.episode_length + 1, self.buffer_size, self.num_agents, self.num_nodes, self.num_nodes), dtype=np.float32)
        self.agent_id = np.zeros((self.episode_length + 1, self.buffer_size, self.num_agents, 1), dtype=np.int32) # Assuming agent_id is scalar index

    def insert(self, num_insert_episodes, obs, share_obs, acts, rewards, dones, dones_env, avail_acts, node_obs, adj, agent_id):
        # Insert standard data
        idx_range = super().insert(num_insert_episodes, obs, share_obs, acts, rewards, dones, dones_env, avail_acts)
        
        # Insert graph data
        # We need to handle the indices correctly. super().insert updates self.current_i and self.filled_i
        # But we need the indices *before* update or just use the returned range if we override carefully.
        # Actually super().insert returns idx_range. But it also updates internal state.
        # So we should probably copy the logic or call super and then fill using the returned range.
        
        # The returned idx_range is (start, end)
        start, end = idx_range
        
        step = 0 # We usually insert a whole episode or chunk. 
        # Wait, RecPolicyBuffer.insert takes (num_insert_episodes, ...) and inserts at current_i.
        # The input arrays are expected to be (episode_length, num_insert_episodes, ...)
        
        # We need to match the slicing.
        # The super insert does:
        # self.obs[:, self.current_i : self.current_i + num_insert_episodes] = obs
        
        # So we do the same:
        self.node_obs[:, start:end] = node_obs
        self.adj[:, start:end] = adj
        self.agent_id[:, start:end] = agent_id
        
        return idx_range

    def sample_inds(self, inds):
        obs, share_obs, acts, rewards, dones, dones_env, avail_acts = super().sample_inds(inds)
        
        node_obs = self.node_obs[:, inds]
        adj = self.adj[:, inds]
        agent_id = self.agent_id[:, inds]
        
        return obs, share_obs, acts, rewards, dones, dones_env, avail_acts, node_obs, adj, agent_id

class GraphRecReplayBuffer(RecReplayBuffer):
    def __init__(self, policy_info, policy_agents, buffer_size, episode_length, use_same_share_obs, use_avail_acts, use_reward_normalization=False):
        self.policy_info = policy_info
        self.policy_buffers = {
            p_id: GraphRecPolicyBuffer(
                buffer_size,
                episode_length,
                len(policy_agents[p_id]),
                self.policy_info[p_id]["obs_space"],
                self.policy_info[p_id]["share_obs_space"],
                self.policy_info[p_id]["act_space"],
                self.policy_info[p_id]["node_obs_space"],
                self.policy_info[p_id]["edge_obs_space"],
                use_same_share_obs,
                use_avail_acts,
                use_reward_normalization,
            )
            for p_id in self.policy_info.keys()
        }

    def insert(self, num_insert_episodes, obs, share_obs, acts, rewards, dones, dones_env, avail_acts, node_obs, adj, agent_id):
        for p_id in self.policy_info.keys():
            idx_range = self.policy_buffers[p_id].insert(
                num_insert_episodes,
                np.array(obs[p_id]),
                np.array(share_obs[p_id]),
                np.array(acts[p_id]),
                np.array(rewards[p_id]),
                np.array(dones[p_id]),
                np.array(dones_env[p_id]),
                np.array(avail_acts[p_id]),
                np.array(node_obs[p_id]),
                np.array(adj[p_id]),
                np.array(agent_id[p_id]),
            )
        return idx_range

    def sample(self, batch_size):
        inds = np.random.choice(self.__len__(), batch_size)
        obs, share_obs, acts, rewards, dones, dones_env, avail_acts, node_obs, adj, agent_id = (
            {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
        )
        for p_id in self.policy_info.keys():
            (
                obs[p_id],
                share_obs[p_id],
                acts[p_id],
                rewards[p_id],
                dones[p_id],
                dones_env[p_id],
                avail_acts[p_id],
                node_obs[p_id],
                adj[p_id],
                agent_id[p_id],
            ) = self.policy_buffers[p_id].sample_inds(inds)

        return obs, share_obs, acts, rewards, dones, dones_env, avail_acts, node_obs, adj, agent_id
