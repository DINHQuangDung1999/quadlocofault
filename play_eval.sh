# python scripts/quadlocofault_rsl_rl/play_eval.py \
#     --num_envs 100 \
#     --task GCN-Isaac-Velocity-Rough-Unitree-Go2-Play-v0 \
#     --checkpoint /home/dung-admin/quadloco_ws/quadlocofault/logs/rsl_rl/unitree_go2_rough_gcn/2026-06-11_03-18-37/model_1999.pt \
#     --headless

# python scripts/quadlocofault_rsl_rl/play_eval.py \
#     --num_envs 100 \
#     --task FTNet-Isaac-Velocity-Rough-Unitree-Go2-Play-v0 \
#     --checkpoint /home/dung-admin/quadloco_ws/quadlocofault/logs/rsl_rl/unitree_go2_rough_ftnet/2026-06-10_19-50-55/model_1999.pt \
#     --headless

# python scripts/quadlocofault_rsl_rl/play_eval.py \
#     --num_envs 100 \
#     --task FLEX-Isaac-Velocity-Rough-Unitree-Go2-Play-v0 \
#     --checkpoint /home/dung-admin/quadloco_ws/quadlocofault/logs/rsl_rl/unitree_go2_rough_flex/2026-06-10_23-36-44/model_1999.pt \
#     --headless

python scripts/quadlocofault_rsl_rl/play_eval.py \
    --num_envs 100 \
    --task Base-Isaac-Velocity-Rough-Unitree-Go2-Play-v0 \
    --checkpoint /home/dung-admin/quadloco_ws/quadlocofault/logs/rsl_rl/unitree_go2_rough_base/2026-06-10_14-22-50/model_1999.pt \
    --headless