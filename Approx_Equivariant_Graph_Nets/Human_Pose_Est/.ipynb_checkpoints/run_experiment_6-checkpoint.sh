#!/bin/bash

# Load the Anaconda module
module load anaconda/Python-ML-2024b

# Print the hostname and start time for debugging purposes
echo "Running on $(hostname)"
echo "Start time: $(date)"

# Define the command for the sixth experiment (G-Net(gc))
COMMAND="python3 main_gcn_aut.py --no_ew --epochs 30 --hid_dim 128 --checkpoint './checkpoints'"

# Define the log file
LOGFILE="./logs/semgcn_gnet_gc_run.log"

# Create logs directory if it doesn't exist
mkdir -p ./logs

# Run the command and redirect both stdout and stderr to the log file
echo "Running command: $COMMAND"
$COMMAND > "$LOGFILE" 2>&1

# Print completion message
echo "Job completed at: $(date)"
