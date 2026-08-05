# python scripts/quadlocofault_rsl_rl/eval.py \
#   --task EquivGCN-Isaac-Velocity-Rough-Unitree-Go2-Play-v0 \
#   --checkpoint /home/dung-admin/quadloco_ws/quadlocofault/logs/rsl_rl/unitree_go2_rough_equiv_gcn/2026-07-27_13-38-42/model_3999.pt \
#   --num_envs 100 \
#   --duration 10 \
#   --num_episodes 2\
#   --command_x 1.0 \
#   --headless 


# python scripts/quadlocofault_rsl_rl/eval.py \
#   --task FLEX-Isaac-Velocity-Rough-Unitree-Go2-Play-v0 \
#   --checkpoint /home/dung-admin/quadloco_ws/quadlocofault/logs/rsl_rl/unitree_go2_rough_flex/baseline/model_3999.pt \
#   --num_envs 50 \
#   --duration 8 \
#   --num_episodes 2\
#   --command_x 1.0 \

python scripts/quadlocofault_rsl_rl/eval.py \
  --task FTNet-Isaac-Velocity-Rough-Unitree-Go2-Play-v0 \
  --checkpoint /home/dung-admin/quadloco_ws/quadlocofault/logs/rsl_rl/unitree_go2_rough_ftnet/baseline/model_3999.pt \
  --num_envs 50 \
  --duration 10 \
  --num_episodes 2\
  --command_x 0.7 \
  --headless 
