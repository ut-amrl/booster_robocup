#!/bin/bash

#================================================
# SBATCH Slurm Configuration
#================================================
#SBATCH --job-name=isaac_wandb_sweep
#SBATCH --account=bewg-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1          # one GPU per trial
#SBATCH --cpus-per-task=16
#SBATCH --mem=20g
#SBATCH --time=48:00:00
#SBATCH --output=sweeps/logs/wandb_sweep_%j.out     # Standard output log file
#SBATCH --error=sweeps/logs/wandb_sweep_%j.err      # Standard error log file
# Run N trials total with 2 in parallel
#SBATCH --array=0-7%2

#================================================
# Environment and Job Execution
#================================================
set -euo pipefail

echo "Job started on $(hostname) at $(date)"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"

# -------- W&B auth (exported into container) --------
export SINGULARITYENV_WANDB_USERNAME="<your_wandb_username>"
export SINGULARITYENV_WANDB_API_KEY="<your_wandb_api_key>"

# (Optional but recommended) use fast scratch for W&B
export SINGULARITYENV_WANDB_MODE="online"
export SINGULARITYENV_WANDB__SERVICE_WAIT="300"
export SINGULARITYENV_WANDB_DIR="${SCRATCH:-$PWD}/wandb_runs"

# Optional NCCL niceties (harmless if unused)
export SINGULARITYENV_NCCL_DEBUG="INFO"
export SINGULARITYENV_NCCL_IB_DISABLE="0"
export SINGULARITYENV_NCCL_SOCKET_IFNAME="^lo,docker0"

SIF="isaac-lab-ros_w-robocup-packages.sif"
HOST_WS="$HOME/booster_robocup"
HOST_CACHE="$HOME/booster_robocup/isaac-sim-cache"
HOST_DATA="$HOME/booster_robocup/isaac-sim-data"

# -------- Your EXACT sweep command (runs INSIDE the container) --------
# We route via the Isaac python shim so the right env/site-packages are used.
SWEEP_ID="amrl-robocup/T1_WalkInPlace_Sweep/y7wsbm2u"
SWEEP_CMD="/workspace/isaaclab/_isaac_sim/python.sh -m wandb agent --count 1 ${SWEEP_ID}"

echo "Launching W&B agent for one trial..."
singularity exec --nv \
  --cleanenv \
  --bind "${HOST_WS}:/workspace/booster_robocup" \
  --bind "${HOST_CACHE}:/isaac-sim/kit/cache" \
  --bind "${HOST_DATA}:/isaac-sim/kit/data" \
  "${SIF}" \
  bash -lc "${SWEEP_CMD}"

echo "Task ${SLURM_ARRAY_TASK_ID} finished at $(date)"
