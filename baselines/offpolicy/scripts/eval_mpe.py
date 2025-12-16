import sys
import os
import numpy as np
from pathlib import Path
import torch
import csv
from distutils.util import strtobool

sys.path.append(os.path.abspath(os.getcwd()))

from baselines.offpolicy.config import get_config
from multiagent.MPE_env import MPEEnv, GraphMPEEnv
from baselines.offpolicy.envs.env_wrappers import DummyVecEnv, SubprocVecEnv
from baselines.offpolicy.runner.rnn.gcm_runner import GCMRunner
from baselines.offpolicy.utils.util import get_cent_act_dim, get_dim_from_space

def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "MPE":
                env = MPEEnv(all_args)
            elif all_args.env_name == "GraphMPE":
                env = GraphMPEEnv(all_args)
            else:
                print(f"Can not support the {all_args.env_name} environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])

def parse_args(args, parser):
    parser.add_argument("--scenario_name", type=str, default="simple_spread", help="Which scenario to run on")
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument("--num_agents", type=int, default=3, help="number of agents")
    parser.add_argument("--use_same_share_obs", action="store_false", default=True, help="Whether to use available actions")
    parser.add_argument("--num_obstacles", type=int, default=3, help="Number of obstacles")
    parser.add_argument("--collaborative", type=lambda x: bool(strtobool(x)), default=True, help="Number of agents in the env")
    parser.add_argument("--max_speed", type=float, default=2, help="Max speed for agents. NOTE that if this is None, then max_speed is 2 with discrete action space")
    parser.add_argument("--collision_rew", type=float, default=5, help="The reward to be negated for collisions with other agents and obstacles")
    parser.add_argument("--goal_rew", type=float, default=5, help="The reward to be added if agent reaches the goal")
    parser.add_argument("--min_dist_thresh", type=float, default=0.05, help="The minimum distance threshold to classify whether agent has reached the goal or not")
    parser.add_argument("--use_dones", type=lambda x: bool(strtobool(x)), default=False, help="Whether we want to use the 'done=True' when agent has reached the goal or just return False like the `simple.py` or `simple_spread.py`")
    
    all_args = parser.parse_known_args(args)[0]
    return all_args

def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    
    # Force eval settings
    all_args.use_eval = True
    all_args.use_render = True
    all_args.n_rollout_threads = 1
    all_args.n_training_threads = 1
    all_args.cuda = False # Use CPU for rendering usually safer/easier
    
    if all_args.algorithm_name == "gcm":
        pass
    else:
        raise NotImplementedError("Only GCM supported in this script for now")

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("Choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("Choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # env init
    env = make_eval_env(all_args)
    num_agents = all_args.num_agents

    # Policy info
    policy_info = {
        "policy_0": {
            "cent_obs_dim": get_dim_from_space(env.share_observation_space[0]),
            "cent_act_dim": get_cent_act_dim(env.action_space),
            "obs_space": env.observation_space[0],
            "share_obs_space": env.share_observation_space[0],
            "act_space": env.action_space[0],
        }
    }
    
    if hasattr(env, "get_graph_spaces"):
        node_obs_space, edge_obs_space = env.get_graph_spaces()
        policy_info["policy_0"]["node_obs_space"] = node_obs_space[0]
        policy_info["policy_0"]["edge_obs_space"] = edge_obs_space[0]
    elif hasattr(env, "envs"):
        policy_info["policy_0"]["node_obs_space"] = env.envs[0].node_observation_space[0]
        policy_info["policy_0"]["edge_obs_space"] = env.envs[0].edge_observation_space[0]

    def policy_mapping_fn(id):
        return "policy_0"

    config = {
        "args": all_args,
        "policy_info": policy_info,
        "policy_mapping_fn": policy_mapping_fn,
        "env": env,
        "eval_env": env,
        "num_agents": num_agents,
        "device": device,
        "run_dir": Path(all_args.model_dir).parent.parent if all_args.model_dir else Path("./"),
    }

    runner = GCMRunner(config)
    runner.render()
    
    if all_args.save_data:
        from multiagent.environment import INFO
        print(f"Saving INFO to INFO.csv with {len(INFO)} records")
        with open('INFO.csv', 'w', encoding='utf-8', newline="") as file:
            writer = csv.writer(file)
            for data in INFO:
                # data is likely a list or tuple
                writer.writerow(data)
        print("Saved INFO.csv")

    env.close()

if __name__ == "__main__":
    main(sys.argv[1:])
