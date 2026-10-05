#!/usr/bin/env bash
set -euo pipefail

python scripts/quadlocofault_rsl_rl/eval.py \
    --protocol flat \
    --models EquivGCNMLP \
    --flex_checkpoint logs/rsl_rl/unitree_go2_rough_flex/2026-09-01_07-04-28_benchmark_v6_4000epochs_flex_dreamflexrewards_nopowerdistr_leglinkcontact_footplanarvel_noclip_hist30_seed2/model_3999.pt \
    --equiv_gcn_mlp_checkpoint logs/rsl_rl/unitree_go2_rough_equiv_gcn_mlp/2026-09-01_15-26-52_benchmark_v6_4000epochs_equivgcnmlp_dreamflexrewards_nopowerdistr_leglinkcontact_footplanarvel_noclip_hist30_seed2/model_3999.pt \
    --flex_history_length 5 \
    --fault_coefficients 0.0 \
    --fault_joint RR_hip_joint \
    --flat_duration 10.0 \
    --fault_time 5.0 \
    --num_envs 1 \
    --eval_seeds 0 \
    --output_dir logs/evaluation/flat_recovery_equivgcnmlp_4000epochs_seed2_rr_hip \
    --no-resume-eval \
    --debug_fault_vis \
    --video \
    --headless
