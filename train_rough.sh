python scripts/quadlocofault_rsl_rl/train.py \
    --task Base-Isaac-Velocity-Rough-Unitree-Go2-v0 \
    --headless \
    --num_envs 4096 \
    --max_iterations 1000

python scripts/quadlocofault_rsl_rl/train.py \
    --task FTNet-Isaac-Velocity-Rough-Unitree-Go2-v0 \
    --headless \
    --num_envs 4096 \
    --max_iterations 1000
    # --resume \
    # --load_run 2026-05-03_20-43-22 \
    # --checkpoint model_1200.pt \
    # --max_iterations 800

# python scripts/quadlocofault_rsl_rl/train.py \
#     --task FLEX-Isaac-Velocity-Rough-Unitree-Go2-v0 \
#     --headless \
#     --num_envs 4096 \
#     --max_iterations 1000

# python scripts/quadlocofault_rsl_rl/train.py \
#     --task PINN-Isaac-Velocity-Rough-Unitree-Go2-v0 \
#     --headless \
#     --num_envs 4096 \
#     --max_iterations 2000