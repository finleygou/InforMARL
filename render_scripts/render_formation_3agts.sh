#!/bin/bash
set -e
# Run the script
seed_max=1
n_agents=3
ep_lens=200
use_curriculum="False"

for seed in $(seq ${seed_max});
do
    echo "seed: ${seed}"
    # execute the script with different params
    python ../onpolicy/scripts/eval_mpe.py \
    --use_valuenorm --use_popart \
    --project_name "GP_Graph" \
    --env_name "GraphMPE" \
    --algorithm_name "rmappo" \
    --seed ${seed} \
    --experiment_name "check" \
    --scenario_name "graph_formation_3agts" \
    --max_edge_dist 1.8 \
    --hidden_size 64 \
    --layer_N 1 \
    --use_wandb "False" \
    --save_gifs "False" \
    --use_render "True" \
    --save_data "False" \
    --use_curriculum "False" \
    --use_policy "False" \
    --gp_type "formation" \
    --num_target 0 \
    --num_agents 3 \
    --num_obstacle 4 \
    --num_dynamic_obs 4 \
    --n_rollout_threads 1 \
    --use_lstm "True" \
    --episode_length ${ep_lens} \
    --render_episodes 1 \
    --ppo_epoch 15 --use_ReLU --gain 0.01 \
    --user_name "finleygou" \
    --use_cent_obs "False" \
    --graph_feat_type "relative" \
    --use_att_gnn "False" \
    --model_dir "/data/goufandi_space/Projects/InforMARL/onpolicy/results/GraphMPE/graph_formation_3agts/rmappo/check/wandb/run-20241013_153321-8bf1syw8/files/"
done