#!/bin/bash

# Run the script
seed_max=1
n_agents=6
# graph_feat_types=("global" "global" "relative" "relative")
# cent_obs=("True" "False" "True" "False")
ep_lens=200
save_gifs="False"
use_curriculum="False"


for seed in `seq ${seed_max}`;
do
# seed=`expr ${seed} + 1`
echo "seed: ${seed}"
# execute the script with different params
python  ../onpolicy/scripts/eval_mpe.py
--project_name "GP_Graph" \
--env_name "GraphMPE" \
--algorithm_name "rmappo" \
--seed ${seed} \
--experiment_name "check" \
--scenario_name "graph_formation_6agts" \
--use_wandb "False" \
--save_gifs ${save_gifs} \
--use_render "True" \
--save_data "False" \
--use_curriculum "False" \
--use_policy "False" \
--num_target 0 \
--num_agents 6 \
--num_obstacle 4 \
--num_dynamic_obs 4 \
--n_rollout_threads 1 \
--use_lstm "False" \
--episode_length ${ep_lens} \
--render_episodes 5 \
--ppo_epoch 15 --use_ReLU --gain 0.01 \
--user_name "finleygou" \
--use_cent_obs "False" \
--graph_feat_type "relative" \
--use_att_gnn "False" \
--model_dir "/data/goufandi_space/Projects/InforMARL/onpolicy/results/GraphMPE/graph_formation_6agts/rmappo/check/wandb/run-20241002_214019-zrys3y0c/files/"
done