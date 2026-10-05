#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

python scripts/quadlocofault_rsl_rl/eval.py \
    --protocol rough \
    --models EquivGCN \
    --equivgcn_checkpoint logs/rsl_rl/unitree_go2_rough_equiv_gcn/2026-10-01_07-31-48_benchmark_v8.4_complete_fault_mixed_terrain_4000epochs_equivgcn_hist30_seed1/model_3999.pt \
    --rough_command_vx 0.75 \
    --terrain_difficulty_min 1.0 \
    --terrain_difficulty_max 1.0 \
    --stair_step_height_max 0.10 \
    --success_distance 3.75 \
    --success_confirmation_time 0.5 \
    --num_envs 300 \
    --fault_joints \
        FL_hip_joint FR_hip_joint RL_hip_joint RR_hip_joint \
        FL_thigh_joint FR_thigh_joint RL_thigh_joint RR_thigh_joint \
        FL_calf_joint FR_calf_joint RL_calf_joint RR_calf_joint \
    --batch_fault_joints \
    --eval_seeds 0 1 2 3 4 \
    --output_dir logs/evaluation/benchmark_v8.4_equivgcn_max_difficulty_seeds0to4_vx075 \
    --headless
