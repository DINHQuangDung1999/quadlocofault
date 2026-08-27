#!/usr/bin/env bash
set -euo pipefail

python scripts/quadlocofault_rsl_rl/eval.py \
    --protocol rough \
    --models FTNet EquivGCNMLP \
    --ftnet_checkpoint logs/rsl_rl/unitree_go2_rough_ftnet/2026-08-26_21-36-06_benchmark_v2_ftnet_dreamflexrewards_noclip_hist30_seed1/model_1999.pt \
    --flex_checkpoint logs/rsl_rl/unitree_go2_rough_flex/2026-08-25_15-11-13_benchmark_v1_flex_native_noclip_hist5_seed1/model_1999.pt \
    --equiv_gcn_mlp_checkpoint logs/rsl_rl/unitree_go2_rough_equiv_gcn_mlp/2026-08-27_01-13-57_benchmark_v2_equivgcnmlp_dreamflexrewards_noclip_hist30_seed1/model_1999.pt \
    --flex_history_length 5 \
    --terrain_difficulty_min 1.0 \
    --terrain_difficulty_max 1.0 \
    --stair_step_height_max 0.15 \
    --success_distance 3.75 \
    --success_confirmation_time 0.5 \
    --num_envs 300 \
    --fault_joints \
        FL_hip_joint FR_hip_joint RL_hip_joint RR_hip_joint \
        FL_thigh_joint FR_thigh_joint RL_thigh_joint RR_thigh_joint \
        FL_calf_joint FR_calf_joint RL_calf_joint RR_calf_joint \
    --batch_fault_joints \
    --eval_seeds 0 \
    --output_dir logs/evaluation/benchmark_v2_dreamflexrewards_balanced_faults_max_difficulty \
    --headless
