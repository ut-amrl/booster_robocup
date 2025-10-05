# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Test an RL agent with RSL-RL.")
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos of tests."
)
parser.add_argument(
    "--video_length",
    type=int,
    default=None,
    help="Length of the recorded video, or None to default to max_length.",
)
parser.add_argument(
    "--max_length",
    type=int,
    default=2000,
    help="Maximum length of each test episode (in steps).",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="Name of the RL agent configuration entry point.",
)
parser.add_argument(
    "--seed", type=int, default=42, help="Seed used for the environment"
)
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--real-time",
    action="store_true",
    default=False,
    help="Run in real-time, if possible.",
)
parser.add_argument(
    "--export_policy",
    action="store_true",
    default=False,
    help="Export the policy as ONNX and JIT.",
)
parser.add_argument(
    "--wandb", action="store_true", default=False, help="Test with model from WandB."
)
parser.add_argument("--wandb_run", type=str, default="", help="Run from WandB.")
parser.add_argument("--wandb_model", type=str, default="", help="Model from WandB.")
parser.add_argument("--wandb_log", action="store_true", default=False, help="Log results to WandB (if --wandb is used)")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
import wandb

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)

import isaaclab_tasks  # noqa: F401
import humanoid_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

from humanoid_utils.wandb_utils import load_wandb_policy
from humanoid_utils.metrics import *

# PLACEHOLDER: Extension template (do not remove this comment)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]  # noqa: F841
    train_task_name = task_name.replace("-Benchmark", "")  # noqa: F841

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.set_num_envs(args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )

    # set episode length
    env_cfg.episode_length_s = args_cli.max_length * env_cfg.sim.dt * env_cfg.decimation

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print(
                "[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task."
            )
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    elif args_cli.wandb:
        # load configuration
        run_path = args_cli.wandb_run
        model_name = args_cli.wandb_model
        resume_path, _ = load_wandb_policy(run_path, model_name, log_root_path)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
        )

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "test"),
            "step_trigger": lambda step: False,
            "video_length": args_cli.video_length if args_cli.video_length is not None else args_cli.max_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during testing.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(
            env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
        )
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(
            env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
        )
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    if args_cli.export_policy:
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        export_policy_as_jit(
            policy_nn,
            normalizer=normalizer,
            path=export_model_dir,
            filename="policy.pt",
        )
        export_policy_as_onnx(
            policy_nn,
            normalizer=normalizer,
            path=export_model_dir,
            filename="policy.onnx",
        )

    dt = env.unwrapped.step_dt

    # record video
    if args_cli.video:
        video_name = "benchmark-video"
        env.env.start_recording(video_name)

    # reset environment
    obs = env.get_observations()

    # keep track of metrics
    alive = torch.ones(env.num_envs, dtype=torch.long)
    metrics = {
        "survival_time": SurvivalTime(env.unwrapped), 
        "movement_error": MovementError(env.unwrapped),
        "energy": Energy(env.unwrapped),
        "smoothness": Smoothness(env.unwrapped),
    }

    # simulate environment
    for timestep in range(1, args_cli.max_length+1):
        if not simulation_app.is_running(): break
        start_time = time.time()

        # can't use inference mode because env.reset modifies tensors
        with torch.no_grad():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, rewards, dones, extras = env.step(actions)

        alive &= ~dones.cpu()
        if not alive.any():
            break

        for metric in metrics.values():
            metric.update(env.unwrapped, alive)

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    if args_cli.video:
        env.env.stop_recording()
        
    computed_metrics = {name: metric.compute(env.unwrapped.cfg.subtask_names) for name, metric in metrics.items()}

    # print metrics
    print("Test results:")
    print_dict(computed_metrics)

    # log to file
    log_file = os.path.join(log_dir, "benchmarks.json")
    with open(log_file, "w") as f:
        log_results = {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_path": resume_path,
            "metrics": computed_metrics,
        }
        json.dump(log_results, f, indent=2)

    # log to wandb
    if args_cli.wandb and args_cli.wandb_log:
        split_path = os.path.relpath(log_dir, log_root_path).split(os.path.sep)
        entity, project, run_id = split_path[0], split_path[1], split_path[-1]

        try:
            run = wandb.init(entity=entity, project=project, id=run_id, resume="allow")
        except wandb.errors.AuthenticationError:
            old_run_name = wandb.Api().run(f"{entity}/{project}/{run_id}").name
            run = wandb.init(
                name=f"{old_run_name}_Benchmark",
                entity=os.environ["WANDB_USERNAME"], 
                project=agent_cfg["wandb_project"]
            )
            print(f"You do not have permission to update runs! Created a new run at {run.url}")

        flat = {f"{m}/{s}": v for m, subs in computed_metrics.items() for s, v in subs.items()}
        wandb.log(flat)

        artifact = wandb.Artifact("benchmark-results", type="evaluation")
        artifact.add_file(log_file)
        if args_cli.video:
            artifact.add_file(os.path.join(log_dir, "videos", "test", f"{video_name}.mp4"))
        run.log_artifact(artifact)

        run.finish()

    # close the env and the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
