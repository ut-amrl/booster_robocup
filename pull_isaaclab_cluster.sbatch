#!/bin/bash
#SBATCH --job-name=singularity_pull_gpu
#SBATCH --output=pull_isaaclab_gpu.out
#SBATCH --error=pull_isaaclab_gpu.err
#SBATCH --partition=gpuA40x4-interactive   # or any GPU partition you have access to
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gpus-per-node=1                   # request 1 GPU
#SBATCH --time=00:30:00                     # 30 minutes should be plenty

# Load singularity (adjust for your system)
module --ignore_cache load "singularity"

# Pull the Docker image as a Singularity image (.sif)
singularity pull --force --dir $HOME/booster_robocup   --name isaac-lab-ros_w-robocup-packages.sif docker://llqqmm/llqqmm/isaac-lab-ros2:w_robocup_packages
