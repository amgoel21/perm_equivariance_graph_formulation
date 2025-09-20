#!/bin/bash
#SBATCH --job-name=inv_run
#SBATCH --output=./logs_r/inv_run_%j.out
#SBATCH --error=./logs_r/inv_run_%j.err
#SBATCH --time=11:59:59
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G

# Print host and start time
echo "Running on $(hostname)"
echo "Start time: $(date)"

mkdir -p ./logs_r

# Run your Python script
python3 -u inv_regression.py

echo "Job completed at: $(date)"
