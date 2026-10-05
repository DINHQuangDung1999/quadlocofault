#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
checkpoint="/home/qdinh/summerschool_ws/quadlocofault/logs/rsl_rl/unitree_go2_rough_equiv_gcn_mlp/2026-09-29_01-12-25_benchmark_v8.3_complete_fault_random_rough_4000epochs_equivgcnmlp_film_hist30_seed1/model_3999.pt"
output_root="logs/v83_complete_fault_videos_15s_marker"
mkdir -p "$output_root"
for joint in FL_hip_joint FR_hip_joint RL_hip_joint RR_hip_joint FL_thigh_joint FR_thigh_joint RL_thigh_joint RR_thigh_joint FL_calf_joint FR_calf_joint RL_calf_joint RR_calf_joint; do
    case "$joint" in
        FL_*|RL_*) viewer_side="left" ;;
        FR_*|RR_*) viewer_side="right" ;;
    esac
    if compgen -G "$output_root/$joint/*.mp4" > /dev/null; then
        echo "[SKIP] $joint"
        continue
    fi
    echo "[RUN] $joint"
    mkdir -p "$output_root/$joint"
    /home/qdinh/miniconda3/envs/env_isaaclab/bin/python scripts/quadlocofault_rsl_rl/play.py \
        --task EquivGCNMLP-Isaac-Velocity-Rough-Unitree-Go2-Play-v0 \
        --num_envs 1 --checkpoint "$checkpoint" --fault_joint "$joint" \
        --fault_coef 0.0 --fault_time 5.0 --terrain_type random_rough --viewer_side "$viewer_side" \
        --video --video_length 750 --video_folder "$output_root/$joint" --headless \
        > "$output_root/$joint/run.log" 2>&1
done
