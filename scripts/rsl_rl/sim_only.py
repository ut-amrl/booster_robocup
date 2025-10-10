# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during training."
)
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Length of the recorded video (in steps).",
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
    "--seed", type=int, default=None, help="Seed used for the environment"
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

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
)

import isaaclab_tasks  # noqa: F401
import humanoid_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config


# PLACEHOLDER: Extension template (do not remove this comment)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]  # noqa: F841
    train_task_name = task_name.replace("-Play", "")  # noqa: F841

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = (
        args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )

    log_dir = "/home/luisamao/booster_robocup/debug_logs"

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
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    from isaaclab.managers import SceneEntityCfg
    asset_cfg = SceneEntityCfg("robot", joint_names = ["joint_ll1"])
    asset = env.unwrapped.scene[asset_cfg.name]
    joint_poses = []
    log_actions = []
    from isaaclab.sensors import ContactSensor
    sensor_cfg = SceneEntityCfg("contact_forces", #"Trunk", "H1", "H2", "AL", "AR", "Waist", "Hip", "Shank", "Ankle"
                                    body_names = ["trunk", "h1", "h2",
                                                  "al1", "al2", "al3", "al4",
                                                  "ar1", "ar2", "ar3", "ar4",
                                                  "waist",
                                                  "ll1", "ll2", "ll3", "ll4", "ll5",
                                                  "lr1", "lr2", "lr3", "lr4", "lr5",])
    contact_sensor: ContactSensor = env.unwrapped.scene.sensors[sensor_cfg.name]
    body_forces = []
    joint_forces = []
    computed_torques = []
    applied_torques = []
    freq = 1
    t = torch.tensor([0], dtype=torch.float32)
    while simulation_app.is_running():
        start_time = time.time()
        actions = torch.zeros((1, 12), device = env.device)
        # actions[0,0] = torch.sin(freq * torch.fmod(t, 1) * 2*torch.pi)
        t+=dt
        obs, _, _, _ = env.step(actions)
        log_actions.append(actions.cpu().numpy().tolist())
        joint_pos = asset.data.joint_pos[0, asset_cfg.joint_ids]
        joint_poses.append(joint_pos.cpu().numpy().tolist())
        computed_torques.append(asset.data.computed_torque[0, asset_cfg.joint_ids].cpu().numpy().tolist())
        applied_torques.append(asset.data.applied_torque[0, asset_cfg.joint_ids].cpu().numpy().tolist())
        body_forces.append(contact_sensor.data.net_forces_w.cpu().numpy().tolist())
        jf = asset._root_physx_view.get_dof_projected_joint_forces().cpu().numpy().reshape(asset._root_physx_view.count, asset._root_physx_view.max_dofs)
        joint_forces.append(jf.tolist())
        print(asset._root_physx_view.count,  asset._root_physx_view.max_dofs, asset._root_physx_view.get_dof_projected_joint_forces().shape)
        if args_cli.video:
            timestep += 1
            if timestep % 50 == 0:
                print("timestep", timestep)
            # Exit the play loop after recording one video
            if timestep >= args_cli.video_length:
                break

    # close the simulator
    env.close()
    import json
    # fname = "/home/luisamao/booster_robocup/debug_logs/lab_test_actuator.json"
    fname = "/home/luisamao/booster_robocup/debug_logs/lab_fall.json"
    data = {
        "joint_poses": joint_poses,
        "actions": log_actions,
        "computed_torques": computed_torques,
        "applied_torques": applied_torques,
        "body_forces": body_forces,
        "joint_forces": joint_forces
    }
    # data = {
    #     "body_forces": body_forces,
    #     "joint_forces": joint_forces
    # }
    with open(fname, "w") as f:
        json.dump(data, f, indent=4)
    print("dumped to", fname)

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
