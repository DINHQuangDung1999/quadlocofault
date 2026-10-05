#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# V8.4 mixed terrain; 30% healthy episodes, otherwise one complete fault at 0-3 s.
python scripts/quadlocofault_rsl_rl/train.py \
    --task EquivGCN-V84-CompleteFault-MixedTerrain-Unitree-Go2-v0 \
    --headless \
    --num_envs 4096 \
    --max_iterations 4000 \
    --seed 1 \
    --run_name benchmark_v8.4_complete_fault_mixed_terrain_4000epochs_equivgcn_hist30_seed1 \
    'env.events.physics_material.params.static_friction_range=[0.2,1.25]' \
    'env.events.physics_material.params.dynamic_friction_range=[0.2,1.25]' \
    env.rewards.feet_slide.weight=-0.1 \
    env.rewards.feet_slide.params.ignore_faulty_legs=false \
    env.observations.history.history_length=30 \
    agent.actor.graph_reflection_equivariant=true \
    agent.actor.fault_encoder_type=tcn \
    "$@"
