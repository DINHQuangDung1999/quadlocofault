# python scripts/quadlocofault_rsl_rl/train.py \
#     --task Base-Isaac-Velocity-Rough-Unitree-Go2-v0 \
#     --headless \
#     --num_envs 4096 \
#     --max_iterations 2000 \
#     --resume \
#     --load_run vanilla2 \
#     --checkpoint model_1000.pt \
#     --max_iterations 2000

python scripts/quadlocofault_rsl_rl/train.py \
    --task FTNet-Isaac-Velocity-Rough-Unitree-Go2-v0 \
    --headless \
    --num_envs 4096 \
    --max_iterations 2000 \
#     --resume \
#     --load_run new_rewards \
#     --checkpoint model_999.pt \
#     --max_iterations 1000

python scripts/quadlocofault_rsl_rl/train.py \
    --task FLEX-Isaac-Velocity-Rough-Unitree-Go2-v0 \
    --headless \
    --num_envs 4096 \
    --max_iterations 2000 \
    # --resume \
    # --load_run new_rewards \
    # --checkpoint model_999.pt \
    # --max_iterations 1000

# python scripts/quadlocofault_rsl_rl/train.py \
#     --task PINN-Isaac-Velocity-Rough-Unitree-Go2-v0 \
#     --headless \
#     --num_envs 4096 \
#     --max_iterations 2000

python scripts/quadlocofault_rsl_rl/train.py \
    --task GCN-Isaac-Velocity-Rough-Unitree-Go2-v0 \
    --headless \
    --num_envs 4096 \
    --max_iterations 2000 \
#     --resume \
#     --load_run 2026-05-26_11-45-07 \
#     --checkpoint model_400.pt \
#     --max_iterations 1600