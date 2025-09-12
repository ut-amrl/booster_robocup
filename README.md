# Humanoid Robots in IsaacLab

## Installation
### Conda Environment
1. Follow the instructions here [https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html] to install IsaacSim and IsaacLab.


2. Install local versions of required packages by running the command below. This will install our copies of IsaacLab packages from `source/`.
```bash
./isaaclab.sh -i
```

### Docker [Todo]
* still figuring this out since robovision seems to be having issues
* running in docker will allow visualization streaming, but this is probably not necessary yet

## Run Training

### Set Up wandb
1. Make an account at https://wandb.ai/site
2. Set the environment variable to enable `wandb` for logging. (**NOTE**: the field listed in User > username is **not** the username. The true username is listed in User > Teams)
```
export WANDB_USERNAME=<user-name>
```
3. Follow the link printed in the terminal to view runs.

### Set Visible Devices Environment Variable

Choose an unused GPU. Please don't take too many at a time.
```
export CUDA_VISIBLE_DEVICES=<gpu id>
```

### Run Training
For baseline funcitonality with `rsl_rl`

```bash
./isaaclab.sh -p scripts/rsl_rl/train.py \
    --task T1-Baseline-v0 \
    --headless \
    --video
```

## Play a Policy
### Using IsaacSim
The following command will take you through the steps of pulling a policy from `wandb` and playing it in a single environment. If running on a lab server, use the `--headless --video` flags.

```
./isaaclab.sh -p scripts/rsl_rl/play.py \
    --task T1-Baseline-Play-v0 \
    --num_envs 1 \
    --wandb \
    --headless \
    --video 
```

### Using Booster-provided Simulation [Todo]
[https://booster.feishu.cn/wiki/XY6Kwrq1bizif4kq7X9c14twnle]

## Committing to This Repo
Changes cannot be made directory to `main` by non-admins. Instead, make changes on a feature branch and make a pull request.

This repo is configured with a format check, which PRs must pass.

Install code formatter and linter: `pip install black ruff`.

Run code formatter on `source` and `script` directories:
```
black source scripts
```

Run linter on `source` and `script` directories:
```
ruff check --fix source scripts
```