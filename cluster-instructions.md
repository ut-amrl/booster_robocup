# Training IsaacLab Policies on the NCSA Delta Cluster
Delta User Documentation: https://docs.ncsa.illinois.edu/systems/delta/en/latest/index.html

## Running Training
This is the workflow for training policies on the Delta Cluster using an existing Docker image with all dependencies needed to run Robocup training. It may be easier to run training on the AMRL servers while debugging, but real training runs should be done on the cluster.

### Make an Account
1. Make an account at http://access-ci.org/
2. Follow all instructions, including setting up Duo
3. Send Joydeep your username

### On the Cluster
1. ssh with the following command:
    ```
    ssh <username>@login.delta.ncsa.illinois.edu
    ```

    Type `1` when prompted and accept the Duo Push verification. You are now in a login node.

2. Clone and `cd` into this repo
3. Run the following command:
    ```
    sbatch --account=<account> pull_isaaclab_cluster.sbatch
    ```
    Your accounts can be found by running the `accounts` command. For example:

    ```
    [lmao1@dt-login02 ~]$ accounts
    Project Summary for User 'lmao1':

    Account                        Balance(Hours)   Deposited(Hours)  Project
    ----------------------------  ----------------  ----------------  ----------------------
    bewg-delta-gpu                            2988              2999  simulator-augmented...
    ```
    where `bewg-delta-gpu` is the account.

    This will launch a job to pull a pre-built docker image with all dependencies needed to run training. Note that this step may take ~1hr. You can see the job output with
    `tail -f pull_isaaclab_cluster.err` or `tail -f pull_isaaclab_cluster.out`.

    Upon successful completion, there should be a singularity image at `isaac-lab-ros_w-robocup-packages.sif`.

4. Run the following commands:
    ```
    mkdir isaac-sim-cache	
    mkdir isaac-sim-data
    ```
    This will create writeable cache directories which we will bind to filepaths in the container which are written to on IsaacLab setup.

5. Change the `SINGULARITYENV_WANDB_USERNAME` to your `wandb` username in `run_training_cluster.sbatch`. Optionally change the SLURM parameters [TODO(luisamao) configure for multiple gpus]

6. Run the following command to launch training
    ```
    sbatch --account=<account> run_training_cluster.sbatch
    ```
    Optionally run the following commands to see output from the training run
    ```
    tail -f isaac_job_<job-id>.err
    tail -f isaac_job_<job-id>.out
    ```

## Useful Commands
Launch an interactive job with a GPU with max time limit 1 hour.
```
srun --account=bewg-delta-gpu \
    --partition=gpuA40x4-interactive \
    --nodes=1 \
    --gpus-per-node=1 \
    --tasks=1 \
    --tasks-per-node=16 \
    --cpus-per-task=1 \
    --mem=20g \
    --time=01:00:00 \
    --pty bash
```
Enter the singularity container in the interactive terminal for debugging
```
singularity exec --nv \
  --cleanenv \
  --bind $HOME/booster_robocup:/workspace/booster_robocup \
  --bind $HOME/booster_robocup/isaac-sim-cache:/isaac-sim/kit/cache \
  --bind $HOME/booster_robocup/isaac-sim-data:/isaac-sim/kit/data \
  isaac-lab-ros_w-robocup-packages.sif \
  /bin/bash
```


Kill a job
```
scancel <job-id>
```
See your jobs
```
squeue -u <username>
```

## Building Docker 
[Todo(luisamao) make this workflow better]

The following is documentation for how the docker container was built. This section can be ignored by most users of this repository.

1. Build the isaaclab-ros2 container
2. run the booster_robocup install script with conda off. make sure the pip packages are all editable mode
3. commit, tag, push
    ```
    docker commit <container-id> container_name
    docker tag container_name:tag llqqmm/container_name:tag
    docker push container_name:tag
    ```
