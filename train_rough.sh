# python scripts/quadlocofault_rsl_rl/train.py \
#     --task Oracle-Isaac-Velocity-Rough-Unitree-Go2-v0 \
#     --headless \
#     --num_envs 4096 \
#     --max_iterations 4000 

# python scripts/quadlocofault_rsl_rl/train.py \
#     --task Base-Isaac-Velocity-Rough-Unitree-Go2-v0 \
#     --headless \
#     --num_envs 4096 \
#     --max_iterations 1200 \
#     --resume \
#     --load_run vanilla2 \
#     --checkpoint model_1000.pt \
#     --max_iterations 2000

# python scripts/quadlocofault_rsl_rl/train.py \
#     --task FTNet-Isaac-Velocity-Rough-Unitree-Go2-v0 \
#     --headless \
#     --num_envs 4096 \
#     --max_iterations 4000 \
    # --resume \
    # --load_run wave_no \
    # --checkpoint model_1999.pt \
    # --max_iterations 1000

python scripts/quadlocofault_rsl_rl/train.py \
    --task FLEX-Isaac-Velocity-Rough-Unitree-Go2-v0 \
    --headless \
    --num_envs 4096 \
    --max_iterations 4000 \
    # --resume \
    # --load_run new_rewards \
    # --checkpoint model_999.pt \
    # --max_iterations 1000

# python scripts/quadlocofault_rsl_rl/train.py \
#     --task GCN-Isaac-Velocity-Rough-Unitree-Go2-v0 \
#     --headless \
#     --num_envs 4096 \
#     --max_iterations 4000 \
#     --resume \
#     --load_run 2026-05-26_11-45-07 \
#     --checkpoint model_400.pt \
#     --max_iterations 1600

# python scripts/quadlocofault_rsl_rl/train.py \
#     --task EquivGCN-Isaac-Velocity-Rough-Unitree-Go2-v0 \
#     --headless \
#     --num_envs 4096 \
#     --max_iterations 4000
    # --resume \
    # --load_run 2026-07-27_00-18-28 \
    # --checkpoint model_1000.pt 