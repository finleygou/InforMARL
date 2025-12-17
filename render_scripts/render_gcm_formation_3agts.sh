#!/bin/bash

# Run the script
seed_max=1
n_agents=3
ep_lens=200

for seed in `seq ${seed_max}`;
do
echo "seed: ${seed}"
export PYTHONPATH=../:$PYTHONPATH
# execute the script with different params
CUDA_VISIBLE_DEVICES='0' python ../baselines/offpolicy/scripts/eval_mpe.py \
    --project_name "GCM_Graph" \
    --env_name "GraphMPE" \
    --algorithm_name "gcm" \
    --seed ${seed} \
    --experiment_name "check_gcm" \
    --scenario_name "graph_formation_3agts_test" \
    --model_dir "/home/goufandi/goufandi_space/Projects/InforMARL/baselines/offpolicy/scripts/results/GraphMPE/graph_formation_3agts_test/gcm/check_gcm/run18/models/" \
    --save_gifs "false" \
    --render_episodes 5 \
    --episode_length ${ep_lens} \
    --use_render "True" \
    --max_edge_dist 1.8 \
    --num_target 0 --num_agents 3 --num_obstacle 4 --num_dynamic_obs 4 \
    --use_wandb "False" \
    --use_policy "False" \
    --use_curriculum "False" \
    --n_rollout_threads 1 \
    --user_name "finleygou" \
    --use_cent_obs False \
    --graph_feat_type "relative" \
    --use_att_gnn True
done
