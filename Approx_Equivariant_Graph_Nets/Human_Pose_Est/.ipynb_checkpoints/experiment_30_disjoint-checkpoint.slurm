#!/bin/bash
#SBATCH --job-name=gine_30_disjoint1     # Job name
#SBATCH --output=./newlogs/gine_30_disjoint1.log    # Log file for output (stdout and stderr)
#SBATCH --error=./newlogs/gine_30_disjoint1.err     # Separate log file for errors
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks=1                    # Number of tasks
#SBATCH --cpus-per-task=4             # Number of CPUs per task
#SBATCH --mem=16G                     # Memory per node
#SBATCH --gres=gpu:volta:1            # Request 1 GPU of type Volta

# Load the Anaconda module
module load anaconda/Python-ML-2024b

# Print debugging info
echo "Running on $(hostname)"
echo "Start time: $(date)"

# Define and run the command
COMMAND="python3 main_gine.py --epochs 50 --batch_size 64 --lr 0.001 --hid_dim 128 --num_layers 5 --dropout 0.1 --perm_type disjoint --checkpoint checkpoints_tests --snapshot 5"

# Create logs directory if it doesn't exist
mkdir -p ./newlogs

echo "Running command: $COMMAND"
$COMMAND 2>&1 | tee ./newlogs/gine_30_disjoint1.log

# Print completion message
echo "Job completed at: $(date)"
