#!/bin/bash

# Run the script
seed_max=1
n_agents=5
# graph_feat_types=("global" "global" "relative" "relative")
# cent_obs=("True" "False" "True" "False")
ep_lens=200

for seed in `seq ${seed_max}`;
do
# seed=`expr ${seed} + 1`
echo "seed: ${seed}"
# execute the script with different params
CUDA_VISIBLE_DEVICES='1' python  ../onpolicy/scripts/train_mpe.py \
--use_valuenorm --use_popart \
--project_name "GP_Graph_NT" \
--env_name "GraphMPE" \
--algorithm_name "rmappo" \
--seed ${seed} \
--experiment_name "check" \
--scenario_name "graph_encirclement_5agts" \
--max_edge_dist 1.8 \
--clip_param 0.15 --gamma 0.985 \
--hidden_size 64 --layer_N 1 \
--num_target 1 --num_agents 5 --num_obstacle 4 --num_dynamic_obs 4 \
--gp_type "encirclement" \
--save_data "True" \
--reward_file_name "r_encirclement_5agts-NT-v1" \
--use_policy "False" \
--use_curriculum "True" \
--guide_cp 0.4 --cp 0.4 --js_ratio 0.7 \
--use_wandb "True" \
--n_training_threads 16 --n_rollout_threads 32 \
--use_lstm "True" \
--episode_length ${ep_lens} \
--num_env_steps 10000000 \
--data_chunk_length 20 \
--ppo_epoch 15 --use_ReLU --gain 0.01 --lr 1e-4 --critic_lr 1e-4 \
--user_name "finleygou" \
--use_cent_obs "False" \
--graph_feat_type "relative" \
--use_att_gnn "True" \
--split_batch "True" --max_batch_size 512 \
--auto_mini_batch_size "True" --target_mini_batch_size 512
done