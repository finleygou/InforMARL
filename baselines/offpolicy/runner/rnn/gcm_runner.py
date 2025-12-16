import numpy as np
import torch
import wandb
import os
from torch.utils.tensorboard import SummaryWriter

from baselines.offpolicy.runner.rnn.mpe_runner import MPERunner
from baselines.offpolicy.algorithms.gcm.algorithm.gcm_policy import GCMPolicy
from baselines.offpolicy.algorithms.gcm.gcm import GCM
from baselines.offpolicy.algorithms.gcm.gcm_buffer import GraphRecReplayBuffer

class GCMRunner(MPERunner):
    def __init__(self, config):
        # We cannot call super().__init__(config) because RecRunner.__init__ will fail 
        # for unknown algorithm "gcm".
        # So we have to duplicate the initialization logic.
        
        # Base Runner Init Logic
        self.args = config["args"]
        self.device = config["device"]
        self.q_learning = ["qmix", "vdn", "gcm"] # Added gcm

        self.share_policy = self.args.share_policy
        self.algorithm_name = self.args.algorithm_name
        self.env_name = self.args.env_name
        self.num_env_steps = self.args.num_env_steps
        self.use_wandb = self.args.use_wandb
        self.use_reward_normalization = self.args.use_reward_normalization
        self.use_popart = self.args.use_popart
        self.use_per = self.args.use_per
        self.per_alpha = self.args.per_alpha
        self.per_beta_start = self.args.per_beta_start
        self.buffer_size = self.args.buffer_size
        self.batch_size = self.args.batch_size
        self.hidden_size = self.args.hidden_size
        self.use_soft_update = self.args.use_soft_update
        self.hard_update_interval_episode = self.args.hard_update_interval_episode
        self.popart_update_interval_step = self.args.popart_update_interval_step
        self.actor_train_interval_step = self.args.actor_train_interval_step
        self.train_interval_episode = self.args.train_interval_episode
        self.train_interval = self.args.train_interval
        self.use_eval = self.args.use_eval
        self.eval_interval = self.args.eval_interval
        self.save_interval = self.args.save_interval
        self.log_interval = self.args.log_interval

        self.total_env_steps = 0
        self.num_episodes_collected = 0
        self.total_train_steps = 0
        self.last_train_episode = 0
        self.last_eval_T = 0
        self.last_save_T = 0
        self.last_log_T = 0
        self.last_hard_update_episode = 0

        if config.__contains__("take_turn"):
            self.take_turn = config["take_turn"]
        else:
            self.take_turn = False

        if config.__contains__("use_same_share_obs"):
            self.use_same_share_obs = config["use_same_share_obs"]
        else:
            self.use_same_share_obs = False

        if config.__contains__("use_available_actions"):
            self.use_avail_acts = config["use_available_actions"]
        else:
            self.use_avail_acts = False

        if config.__contains__("buffer_length"):
            self.episode_length = config["buffer_length"]
            if self.args.use_naive_recurrent_policy:
                self.data_chunk_length = config["buffer_length"]
            else:
                self.data_chunk_length = self.args.data_chunk_length
        else:
            self.episode_length = self.args.episode_length
            if self.args.use_naive_recurrent_policy:
                self.data_chunk_length = self.args.episode_length
            else:
                self.data_chunk_length = self.args.data_chunk_length

        self.policy_info = config["policy_info"]
        self.policy_ids = sorted(list(self.policy_info.keys()))
        self.policy_mapping_fn = config["policy_mapping_fn"]

        self.num_agents = config["num_agents"]
        self.agent_ids = [i for i in range(self.num_agents)]

        self.env = config["env"]
        self.eval_env = config["eval_env"]
        self.num_envs = 1 # MPE Runner sets this to 1 for recurrent version usually? 
        # Wait, MPERunner in train_mpe.py sets num_envs based on n_rollout_threads?
        # In train_mpe.py:
        # if all_args.n_rollout_threads == 1: return DummyVecEnv...
        # else: return SubprocVecEnv...
        # But RecRunner sets self.num_envs = 1. This seems wrong if n_rollout_threads > 1.
        # Ah, RecRunner says "no parallel envs" comment.
        # But SubprocVecEnv handles parallel envs.
        # Let's check RecRunner again. It sets self.num_envs = 1.
        # And MPERunner uses self.num_envs in shared_collect_rollout:
        # share_obs = obs.reshape(self.num_envs, -1)
        # If self.num_envs is 1, it assumes 1 env.
        # But if we use SubprocVecEnv, we have multiple envs.
        # Maybe RecRunner assumes we pass a VecEnv which acts as one object, but we need to know how many envs are inside.
        # Actually, RecRunner seems to assume n_rollout_threads=1 for recurrent version in train_mpe.py:
        # assert all_args.n_rollout_threads == 1, "only support 1 env in recurrent version."
        # So self.num_envs = 1 is correct for RecRunner usage in train_mpe.py.
        
        self.num_envs = self.args.n_rollout_threads # Should be 1 based on assertion

        # dir
        self.model_dir = self.args.model_dir
        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
        else:
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / "logs")
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir / "models")
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        # GCM Specific Init
        self.policies = {
            p_id: GCMPolicy(config, self.policy_info[p_id]) for p_id in self.policy_ids
        }

        self.trainer = GCM(
            self.args,
            self.num_agents,
            self.policies,
            self.policy_mapping_fn,
            device=self.device,
            episode_length=self.episode_length,
        )

        if self.model_dir is not None:
            self.restore_q()

        self.policy_agents = {
            policy_id: sorted(
                [
                    agent_id
                    for agent_id in self.agent_ids
                    if self.policy_mapping_fn(agent_id) == policy_id
                ]
            )
            for policy_id in self.policies.keys()
        }

        self.saver = self.save_q
        self.restorer = self.restore_q
        self.train = self.batch_train_q

        # Buffer
        self.buffer = GraphRecReplayBuffer(
            self.policy_info,
            self.policy_agents,
            self.buffer_size,
            self.episode_length,
            self.use_same_share_obs,
            self.use_avail_acts,
            self.use_reward_normalization,
        )

        # MPERunner Init Logic
        self.collecter = (
            self.shared_collect_rollout
            if self.share_policy
            else self.separated_collect_rollout
        )
        
        # Warmup
        num_warmup_episodes = max((self.batch_size, self.args.num_random_episodes))
        self.warmup(num_warmup_episodes)
        import time
        self.start = time.time()
        self.log_clear()

    @torch.no_grad()
    def shared_collect_rollout(self, explore=True, training_episode=True, warmup=False):
        env_info = {}
        p_id = "policy_0"
        policy = self.policies[p_id]

        env = self.env if training_episode or warmup else self.eval_env

        # Reset returns 4 values for GraphEnv
        obs, agent_id, node_obs, adj = env.reset()

        rnn_states_batch = np.zeros(
            (self.num_envs * self.num_agents, self.hidden_size), dtype=np.float32
        )
        last_acts_batch = np.zeros(
            (self.num_envs * self.num_agents, policy.output_dim), dtype=np.float32
        )

        episode_obs = {p_id: np.zeros((self.episode_length + 1, self.num_envs, self.num_agents, policy.obs_dim), dtype=np.float32)}
        episode_share_obs = {p_id: np.zeros((self.episode_length + 1, self.num_envs, self.num_agents, policy.central_obs_dim), dtype=np.float32)}
        episode_acts = {p_id: np.zeros((self.episode_length, self.num_envs, self.num_agents, policy.output_dim), dtype=np.float32)}
        episode_rewards = {p_id: np.zeros((self.episode_length, self.num_envs, self.num_agents, 1), dtype=np.float32)}
        episode_dones = {p_id: np.ones((self.episode_length, self.num_envs, self.num_agents, 1), dtype=np.float32)}
        episode_dones_env = {p_id: np.ones((self.episode_length, self.num_envs, 1), dtype=np.float32)}
        episode_avail_acts = {p_id: None}
        
        # Graph specific storage
        # node_obs shape: (num_envs, num_agents, num_nodes, node_feat_dim)
        # adj shape: (num_envs, num_agents, num_nodes, num_nodes)
        # agent_id shape: (num_envs, num_agents, 1)
        
        node_obs_shape = node_obs.shape[2:]
        adj_shape = adj.shape[2:]
        
        episode_node_obs = {p_id: np.zeros((self.episode_length + 1, self.num_envs, self.num_agents, *node_obs_shape), dtype=np.float32)}
        episode_adj = {p_id: np.zeros((self.episode_length + 1, self.num_envs, self.num_agents, *adj_shape), dtype=np.float32)}
        episode_agent_id = {p_id: np.zeros((self.episode_length + 1, self.num_envs, self.num_agents, 1), dtype=np.int32)}

        t = 0
        while t < self.episode_length:
            share_obs = obs.reshape(self.num_envs, -1)
            obs_batch = np.concatenate(obs)
            node_obs_batch = np.concatenate(node_obs)
            adj_batch = np.concatenate(adj)
            agent_id_batch = np.concatenate(agent_id)
            
            if warmup:
                acts_batch = policy.get_random_actions(obs_batch)
                _, rnn_states_batch = policy.get_actions(obs_batch, node_obs_batch, adj_batch, agent_id_batch, last_acts_batch, rnn_states_batch)
            else:
                acts_batch, rnn_states_batch = policy.get_actions(
                    obs_batch, node_obs_batch, adj_batch, agent_id_batch,
                    last_acts_batch, rnn_states_batch,
                    t_env=self.total_env_steps, explore=explore
                )
            
            acts_batch = acts_batch if isinstance(acts_batch, np.ndarray) else acts_batch.cpu().detach().numpy()
            rnn_states_batch = rnn_states_batch if isinstance(rnn_states_batch, np.ndarray) else rnn_states_batch.cpu().detach().numpy()
            last_acts_batch = acts_batch

            env_acts = np.split(acts_batch, self.num_envs)
            
            # Step returns 7 values
            next_obs, next_agent_id, next_node_obs, next_adj, rewards, dones, infos = env.step(env_acts)
            
            if training_episode:
                self.total_env_steps += self.num_envs

            dones_env = np.all(dones, axis=1)
            terminate_episodes = np.any(dones_env) or t == self.episode_length - 1

            episode_obs[p_id][t] = obs
            episode_share_obs[p_id][t] = share_obs
            episode_acts[p_id][t] = np.stack(env_acts)
            episode_rewards[p_id][t] = rewards
            episode_dones[p_id][t] = dones[:, :, np.newaxis]
            episode_dones_env[p_id][t] = dones_env[:, np.newaxis]
            
            episode_node_obs[p_id][t] = node_obs
            episode_adj[p_id][t] = adj
            episode_agent_id[p_id][t] = agent_id

            t += 1
            obs = next_obs
            agent_id = next_agent_id
            node_obs = next_node_obs
            adj = next_adj

            if terminate_episodes:
                break

        episode_obs[p_id][t] = obs
        episode_share_obs[p_id][t] = obs.reshape(self.num_envs, -1)
        episode_node_obs[p_id][t] = node_obs
        episode_adj[p_id][t] = adj
        episode_agent_id[p_id][t] = agent_id

        if explore:
            self.num_episodes_collected += self.num_envs
            self.buffer.insert(
                self.num_envs,
                episode_obs,
                episode_share_obs,
                episode_acts,
                episode_rewards,
                episode_dones,
                episode_dones_env,
                episode_avail_acts,
                episode_node_obs,
                episode_adj,
                episode_agent_id
            )

        env_info["average_episode_rewards"] = np.mean(np.sum(episode_rewards[p_id], axis=0))
        return env_info

    @torch.no_grad()
    def render(self):
        env = self.env
        # Reset returns 4 values for GraphEnv
        obs, agent_id, node_obs, adj = env.reset()
        
        if self.args.save_gifs:
            import imageio
            all_frames = []
            image = env.render("rgb_array")[0][0]
            all_frames.append(image)
        else:
            env.render("human")

        rnn_states_batch = np.zeros(
            (self.num_envs * self.num_agents, self.hidden_size), dtype=np.float32
        )
        last_acts_batch = np.zeros(
            (self.num_envs * self.num_agents, self.policies["policy_0"].output_dim), dtype=np.float32
        )
        
        for t in range(self.episode_length):
            obs_batch = np.concatenate(obs)
            node_obs_batch = np.concatenate(node_obs)
            adj_batch = np.concatenate(adj)
            agent_id_batch = np.concatenate(agent_id)
            
            policy = self.policies["policy_0"]
            acts_batch, rnn_states_batch = policy.get_actions(
                obs_batch, node_obs_batch, adj_batch, agent_id_batch,
                last_acts_batch, rnn_states_batch,
                t_env=self.total_env_steps, explore=False
            )
            
            acts_batch = acts_batch if isinstance(acts_batch, np.ndarray) else acts_batch.cpu().detach().numpy()
            rnn_states_batch = rnn_states_batch if isinstance(rnn_states_batch, np.ndarray) else rnn_states_batch.cpu().detach().numpy()
            last_acts_batch = acts_batch

            env_acts = np.split(acts_batch, self.num_envs)
            
            # Step returns 7 values
            next_obs, next_agent_id, next_node_obs, next_adj, rewards, dones, infos = env.step(env_acts)
            
            if self.args.save_gifs:
                image = env.render("rgb_array")[0][0]
                all_frames.append(image)
            else:
                env.render("human")

            obs = next_obs
            agent_id = next_agent_id
            node_obs = next_node_obs
            adj = next_adj
            
        if self.args.save_gifs:
            imageio.mimsave(str(self.run_dir) + '/render.gif', all_frames, duration=0.1)
