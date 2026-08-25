# python scripts/quadlocofault_rsl_rl/eval.py \
#     --protocol rough \
#     --latest_run_name ftnetrewards \
#     --models FTNet FLEX EquivGCNMLP \
#     --fault_joints \
#         FL_hip_joint FR_hip_joint RL_hip_joint RR_hip_joint \
#         FL_thigh_joint FR_thigh_joint RL_thigh_joint RR_thigh_joint \
#         FL_calf_joint FR_calf_joint RL_calf_joint RR_calf_joint \
#     --success_distance 3.75 \
#     --success_confirmation_time 0.5 \
#     --num_envs 100 \
#     --eval_seeds 0 1 2 \
#     --headless

python scripts/quadlocofault_rsl_rl/eval.py \
    --protocol rough \
    --latest_run_name ftnetrewards \
    --models GCN \
    --success_distance 3.75 \
    --success_confirmation_time 0.5 \
    --num_envs 1000 \
    --eval_seeds 0 1 2 \
    --headless


cd /home/dung-admin/quadloco_ws/quadlocofault

python scripts/quadlocofault_rsl_rl/eval.py \
    --protocol rough \
    --models EquivGCN EquivGCNMLP \
    --equivgcn_checkpoint logs/rsl_rl/unitree_go2_rough_equiv_gcn/2026-08-17_13-54-49_gcnrewards_seed1/model_2999.pt \
    --equiv_gcn_mlp_checkpoint logs/rsl_rl/unitree_go2_rough_equiv_gcn_mlp/2026-08-17_20-07-17_gcnrewards_seed1/model_3000.pt \
    --success_distance 3.75 \
    --success_confirmation_time 0.5 \
    --num_envs 1000 \
    --eval_seeds 0 1 2 \
    --output_dir logs/evaluation/gcnrewards_seed1 \
    --headless