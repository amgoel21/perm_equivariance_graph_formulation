#!/bin/bash
#SBATCH --job-name=equiv_run
#SBATCH --output=./logs/equiv_exp_%j.out
#SBATCH --error=./logs/equiv_exp_%j.err
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
python3 -u equiv_experiments.py

echo "Job completed at: $(date)"
