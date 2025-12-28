#!/bin/bash

# Run the script
seed_max=1
n_agents=5
ep_lens=200

for seed in `seq ${seed_max}`;
do
echo "seed: ${seed}"
export PYTHONPATH=../:$PYTHONPATH
# execute the script with different params
CUDA_VISIBLE_DEVICES='0' python ../baselines/offpolicy/scripts/train/train_mpe.py \
--project_name "GCM_Graph" \
--env_name "GraphMPE" \
--algorithm_name "gcm" \
--seed ${seed} \
--log_interval 6400 \
--experiment_name "check_gcm" \
--scenario_name "graph_encirclement_5agts" \
--max_edge_dist 1.8 \
--clip_param 0.15 --gamma 0.985 \
--num_target 1 --num_agents 5 --num_obstacle 4 --num_dynamic_obs 4 \
--use_wandb "False" \
--use_policy "False" \
--use_curriculum "True" \
--gp_type "encirclement" \
--guide_cp 0.6 --cp 0.4 --js_ratio 0.7 \
--save_data "True" \
--reward_file_name "r_enc_3agts_gcm-noGP" \
--MC_file_name "MC_3agts" \
--n_training_threads 16 --n_rollout_threads 1 \
--episode_length ${ep_lens} \
--num_env_steps 6000000 \
--batch_size 128 \
--lr 2e-4 \
--user_name "finleygou" \
--use_cent_obs False \
--graph_feat_type "relative"
done
