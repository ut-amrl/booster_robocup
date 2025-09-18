#!/bin/bash
#SBATCH --job-name=singularity_pull_gpu
#SBATCH --output=pull_isaaclab_cluster.out
#SBATCH --error=pull_isaaclab_cluster.err
#SBATCH --partition=gpuA40x4   # or any GPU partition you have access to
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gpus-per-node=1                   # request 1 GPU
#SBATCH --time=01:00:00                     # 1 hour should be plenty


# Pull the Docker image as a Singularity image (.sif)
singularity pull --force --dir $HOME/booster_robocup   --name isaac-lab-ros_w-robocup-packages.sif docker://llqqmm/isaac-lab-ros2:w-robocup-packages
