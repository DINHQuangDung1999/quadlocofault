#!/usr/bin/env bash
set -euo pipefail

python scripts/quadlocofault_rsl_rl/eval.py \
    --protocol latent \
    --models EquivGCNMLP \
    --equiv_gcn_mlp_checkpoint logs/rsl_rl/unitree_go2_rough_equiv_gcn_mlp/2026-09-01_15-26-52_benchmark_v6_4000epochs_equivgcnmlp_dreamflexrewards_nopowerdistr_leglinkcontact_footplanarvel_noclip_hist30_seed2/model_3999.pt \
    --latent_collect_steps 0 50 100 125 140 149 150 151 160 170 180 190 200 210 225 250 \
    --fault_time 3.0 \
    --fault_coefficients 0.1 \
    --num_envs 300 \
    --fault_joints \
        FL_hip_joint FR_hip_joint RL_hip_joint RR_hip_joint \
        FL_thigh_joint FR_thigh_joint RL_thigh_joint RR_thigh_joint \
        FL_calf_joint FR_calf_joint RL_calf_joint RR_calf_joint \
    --batch_fault_joints \
    --eval_seeds 0 \
    --output_dir logs/evaluation/benchmark_v6_4000epochs_equivgcnmlp_seed2_latents_alpha_0.1_fault_at_3s \
    --headless
