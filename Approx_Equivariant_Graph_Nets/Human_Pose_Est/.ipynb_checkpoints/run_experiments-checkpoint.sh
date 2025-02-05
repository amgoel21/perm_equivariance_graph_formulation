#!/bin/bash

module load anaconda/Python-ML-2024b

# Navigate to the directory where the code is located
cd /Approx_Equivariant_Graph_Nets/Human_Pose_Est

# Define commands for each experiment
commands=(
    "python3 main_gcn_aut.py --epochs 30 --hid_dim 128 --checkpoint './checkpoints' > logs/default_semgcn.log 2>&1"
    "python3 main_gcn_aut.py --no_tie --epochs 30 --hid_dim 128 --checkpoint './checkpoints' > logs/relax_s16.log 2>&1"
    "python3 main_gcn_aut.py --aut --epochs 30 --hid_dim 128 --checkpoint './checkpoints' > logs/s2_aut_g.log 2>&1"
    "python3 main_gcn_aut.py --triv --epochs 30 --hid_dim 128 --checkpoint './checkpoints' > logs/trivial.log 2>&1"
    "python3 main_gcn_aut.py --no_gc --no_ew --epochs 30 --hid_dim 128 --checkpoint './checkpoints' > logs/vanilla_gnet.log 2>&1"
    "python3 main_gcn_aut.py --no_ew --epochs 30 --hid_dim 128 --checkpoint './checkpoints' > logs/gnet_gc.log 2>&1"
    "python3 main_gcn_aut.py --no_gc --no_ew --pointwise --epochs 30 --hid_dim 128 --checkpoint './checkpoints' > logs/gnet_pt.log 2>&1"
)

# Create logs directory if it doesn't exist
mkdir -p logs

# Run all commands in parallel and wait for them to finish
for cmd in "${commands[@]}"; do
    echo "Running: $cmd"
    eval $cmd &
done

# Wait for all background processes to finish
wait

echo "All experiments are completed."
