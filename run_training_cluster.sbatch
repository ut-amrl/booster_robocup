#!/bin/bash

#================================================
# SBATCH Slurm Configuration
#================================================
#SBATCH --job-name=isaac_lab_job      # Name of the job
#SBATCH --account=bewg-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1                     # Total number of nodes to request
#SBATCH --tasks=1                     # Total number of tasks (processes) to run
#SBATCH --gpus-per-node=1             # Number of GPUs per node
#SBATCH --cpus-per-task=16             # Cores per task
#SBATCH --mem=20g                     # Main memory requested
#SBATCH --time=48:00:00               # Duration of job
#SBATCH --output=isaac_job_%j.out     # Standard output log file
#SBATCH --error=isaac_job_%j.err      # Standard error log file

#================================================
# Environment and Job Execution
#================================================
echo "Job started on $(hostname) at $(date)"
echo "Setting environment variables for Singularity..."

# Set environment variables for the container using the SINGULARITYENV_ prefix
# This is the recommended method.
export SINGULARITYENV_WANDB_USERNAME="<your-wandb-username>"

echo "Running Singularity container..."
singularity exec --nv \
  --cleanenv \
  --bind $HOME/booster_robocup:/workspace/booster_robocup \
  --bind $HOME/booster_robocup/isaac-sim-cache:/isaac-sim/kit/cache \
  --bind $HOME/booster_robocup/isaac-sim-data:/isaac-sim/kit/data \
  isaac-lab-ros_w-robocup-packages.sif \
  /workspace/isaaclab/_isaac_sim/python.sh scripts/rsl_rl/train.py --task T1-Baseline-v0 --headless --video

echo "Job finished at $(date)"