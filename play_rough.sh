# python scripts/quadlocofault_rsl_rl/play.py \
#     --task Base-Isaac-Velocity-Rough-Unitree-Go2-Play-v0 \
#     --num_envs 32 \
#     --headless\
#     --checkpoint /home/dung-admin/quadloco_ws/quadlocofault/logs/rsl_rl/unitree_go2_rough_base/vanilla/model_2999.pt
    
# python scripts/quadlocofault_rsl_rl/play.py \
#     --task FTNet-Isaac-Velocity-Rough-Unitree-Go2-Play-v0 \
#     --num_envs 32 \
#     --checkpoint /home/dung-admin/quadloco_ws/quadlocofault/logs/rsl_rl/unitree_go2_rough_ftnet/2026-06-08_17-13-31/model_1999.pt
python scripts/quadlocofault_rsl_rl/play.py \
    --task GCN-Isaac-Velocity-Rough-Unitree-Go2-Play-v0 \
    --num_envs 32 \
    --checkpoint /home/dung-admin/quadloco_ws/quadlocofault/logs/rsl_rl/unitree_go2_rough_gcn/2026-06-11_03-18-37/model_1999.pt