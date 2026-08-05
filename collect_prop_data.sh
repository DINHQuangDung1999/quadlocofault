python scripts/quadlocofault_rsl_rl/run_collect_prop_data.py \
    --task FTNet-Isaac-Velocity-Rough-Unitree-Go2-Play-v0 \
    --checkpoint /home/dung-admin/quadloco_ws/quadlocofault/logs/rsl_rl/unitree_go2_rough_ftnet/baseline/model_3999.pt \
    --num_envs 32 \
    --collection_steps 20000 \
    --shard_steps 250 \
    --headless