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
#     --checkpoint /home/qdinh/summerschool_ws/quadlocofault/logs/rsl_rl/unitree_go2_rough_ftnet/2026-08-30_15-20-27_benchmark_v6_4000epochs_ftnet_dreamflexrewards_nopowerdistr_leglinkcontact_footplanarvel_noclip_hist30_seed1/model_3999.pt \
#     --fault_joint FR_hip_joint

# python scripts/quadlocofault_rsl_rl/play.py \
#     --task GCN-Isaac-Velocity-Rough-Unitree-Go2-Play-v0 \
#     --num_envs 32 \
#     --checkpoint /home/dung-admin/quadloco_ws/quadlocofault/logs/rsl_rl/unitree_go2_rough_gcn/baseline/model_3999.pt

python scripts/quadlocofault_rsl_rl/play.py \
    --task EquivGCN-Isaac-Velocity-Rough-Unitree-Go2-Play-v0 \
    --num_envs 1 \
    --checkpoint /home/qdinh/summerschool_ws/quadlocofault/logs/rsl_rl/unitree_go2_rough_equiv_gcn/2026-10-01_07-31-48_benchmark_v8.4_complete_fault_mixed_terrain_4000epochs_equivgcn_hist30_seed1/model_3999.pt \
    --fault_joint FR_hip_joint \
    --terrain_type flat

# python scripts/quadlocofault_rsl_rl/play.py \
#     --task EquivGCN13Node-V84-CompleteFault-MixedTerrain-Unitree-Go2-Play-v0 \
#     --num_envs 1 \
#     --checkpoint /home/qdinh/summerschool_ws/quadlocofault/logs/rsl_rl/unitree_go2_rough_equiv_gcn_13_node/2026-10-03_03-20-06_benchmark_v8.4_complete_fault_mixed_terrain_4000epochs_equivgcn_13_node_hist30_seed1/model_3999.pt \
#     --fault_joint RR_hip_joint \
#     --terrain_type random_rough

# python scripts/quadlocofault_rsl_rl/play.py \
#     --task EquivGCNMLP-Isaac-Velocity-Rough-Unitree-Go2-Play-v0 \
#     --num_envs 1 \
#     --checkpoint /home/qdinh/summerschool_ws/quadlocofault/logs/rsl_rl/unitree_go2_rough_equiv_gcn_mlp/2026-09-29_23-24-45_benchmark_v8.4_complete_fault_mixed_terrain_4000epochs_equivgcnmlp_film_hist30_seed1/model_3999.pt\
#     --fault_joint FL_calf_joint \
#     --terrain_type random_rough
    # --headless \
    # --export both \
    # --export_only
    # --video \
    # --video_length 500 \
    # --real-time \
    # --headless
#     --terrain_type random_rough
    # --headless \
    # --export both \
    # --export_only
    # --fault_joint FR_hip_joint

# python scripts/quadlocofault_rsl_rl/play.py\
#     --task FLEX-Isaac-Velocity-Rough-Unitree-Go2-Play-v0 \
#     --num_envs 1 \
#     --checkpoint /home/qdinh/summerschool_ws/quadlocofault/logs/rsl_rl/unitree_go2_rough_flex/2026-08-31_00-01-04_benchmark_v6_4000epochs_flex_dreamflexrewards_nopowerdistr_leglinkcontact_footplanarvel_noclip_hist30_seed1/model_3999.pt \
#     --fault_joint FR_hip_joint
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

