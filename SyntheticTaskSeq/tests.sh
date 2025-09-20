#!/bin/bash
#SBATCH --job-name=inv_run
#SBATCH --output=./logs/inv_run_%j.out
#SBATCH --error=./logs/inv_run_%j.err
#SBATCH --time=10:00:00
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

# Print host and start time
echo "Running on $(hostname)"
echo "Start time: $(date)"

mkdir -p ./logs

# Run your Python script
python3 -u train_inv.py

echo "Job completed at: $(date)"
