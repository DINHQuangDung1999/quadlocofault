python scripts/quadlocofault_rsl_rl/train_fault_residual_tcn.py \
    --dataset_dir datasets/prop_fault/FTNet-Isaac-Velocity-Rough-Unitree-Go2-Play-v0/2026-07-31_09-54-34 \
    --epochs 100 \
    --batch_size 512 \
    --min_fault_age_steps 30 \
    --device cuda \
    --amp