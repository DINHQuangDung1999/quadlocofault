# python scripts/quadlocofault_rsl_rl/play.py \
#     --task Base-Isaac-Velocity-Rough-Unitree-Go2-Play-v0 \
#     --num_envs 32 \
#     --checkpoint /home/dung-admin/quadloco_ws/quadlocofault/logs/rsl_rl/unitree_go2_rough_base/wave_no/model_1999.pt
    
# python scripts/quadlocofault_rsl_rl/play.py \
#     --task Oracle-Isaac-Velocity-Rough-Unitree-Go2-Play-v0 \
#     --num_envs 32 \
#     --checkpoint /home/dung-admin/quadloco_ws/quadlocofault/logs/rsl_rl/unitree_go2_rough_oracle/2026-07-24_15-03-49/model_3999.pt
    
# python scripts/quadlocofault_rsl_rl/play.py \
#     --task FTNet-Isaac-Velocity-Rough-Unitree-Go2-Play-v0 \
#     --num_envs 1 \
#     --checkpoint /home/dung-admin/quadloco_ws/quadlocofault/logs/rsl_rl/unitree_go2_rough_ftnet/2026-08-26_06-56-36_benchmark_v1_ftnet_native_clip_hist30_seed1/model_1999.pt \
#     --fault_joint FR_calf_joint

# python scripts/quadlocofault_rsl_rl/play.py \
#     --task GCN-Isaac-Velocity-Rough-Unitree-Go2-Play-v0 \
#     --num_envs 32 \
#     --checkpoint /home/dung-admin/quadloco_ws/quadlocofault/logs/rsl_rl/unitree_go2_rough_gcn/baseline/model_3999.pt

# python scripts/quadlocofault_rsl_rl/play.py \
#     --task EquivGCN-Isaac-Velocity-Rough-Unitree-Go2-Play-v0 \
#     --num_envs 32 \
#     --checkpoint /home/dung-admin/quadloco_ws/quadlocofault/logs/rsl_rl/unitree_go2_rough_equiv_gcn/baseline/model_3999.pt \
#     --fault_tcn_checkpoint /home/dung-admin/quadloco_ws/quadlocofault/datasets/prop_fault/FTNet-Isaac-Velocity-Rough-Unitree-Go2-Play-v0/2026-07-27_11-45-57/fault_tcn_runs/2026-07-27_11-53-18/best.pt

python scripts/quadlocofault_rsl_rl/play.py \
    --task EquivGCNMLP-Isaac-Velocity-Rough-Unitree-Go2-Play-v0 \
    --num_envs 1 \
    --checkpoint /home/dung-admin/quadloco_ws/quadlocofault/logs/rsl_rl/unitree_go2_rough_equiv_gcn_mlp/2026-08-27_01-13-57_benchmark_v2_equivgcnmlp_dreamflexrewards_noclip_hist30_seed1/model_1999.pt \
    --fault_joint RR_hip_joint \
#     --terrain_type random_rough
    # --headless \
    # --export both \
    # --export_only
    # --fault_joint FR_hip_joint

# python scripts/quadlocofault_rsl_rl/play.py\
#     --task FLEX-Isaac-Velocity-Rough-Unitree-Go2-Play-v0 \
#     --num_envs 1 \
#     --checkpoint /home/dung-admin/quadloco_ws/quadlocofault/logs/rsl_rl/unitree_go2_rough_flex/2026-08-26_10-36-33_benchmark_v1_flex_native_clip_hist5_seed1/model_1999.pt \
#     --fault_joint FR_calf_joint
    # --checkpoint /home/dung-admin/quadloco_ws/quadlocofault/logs/rsl_rl/unitree_go2_rough_flex/2026-06-10_23-36-44/model_1999.pt

# python scripts/quadlocofault_rsl_rl/play.py \
#   --task EquivGCNMLP-Isaac-Velocity-Rough-Unitree-Go2-Play-v0 \
#   --num_envs 4000 \
#   --checkpoint /home/dung-admin/quadloco_ws/quadlocofault/logs/rsl_rl/unitree_go2_rough_equiv_gcn_mlp/ftnetrewards/model_3999.pt \
#   --collect_fused_latent \
#   --headless
#   --latent_collect_step 50 \
#   --latent_tsne_perplexity 30 \
#   --latent_tsne_output fused_latent_tsne.pdf \
#   --headless \
#   --checkpoint /home/dung-admin/quadloco_ws/quadlocofault/logs/rsl_rl/unitree_go2_rough_equiv_gcn/2026-07-30_14-34-10/model_2499.pt

