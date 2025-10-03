from collections import OrderedDict
from dataclasses import MISSING
from functools import partial
import math
import sys
from typing import Dict, Sequence, Callable

import torch
import humanoid_mdp
import isaaclab.terrains as terrain_gen
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv, ViewerCfg

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import CommandTermCfg as CommandTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp
import humanoid_mdp
from .rsl_rl_cfg import T1BaselineCfg

"""
4 config classes, each define a different benchmark setting
- defines a subterrain
- defines command modifier
- defines events to add

custom terrain generator, just makes 4 subterrains

custom velocitycommand term, takes in 4 config classes
- override command property, concat each subtask command modifier

# custom command term, each of command, update, and re

custom event term, subclass event term but also takes in target cfg class
- in event, do according to idx

in actual cfg:
- set num envs to a number * 4
- in env reset, reset location depending on idx
- in get command, get according to idx
- loop over each subtask, add their events 

IM GOING TO LOSE IT ALDKJFLDSKFJADFDF


commands:

SplitCommand(ActualCommandTerm):
    takes in splitcommandtermcfg in init
    - has class type like VelocityCommandTerm
    - has list of VelocityCommandTermCfg
    
    in init:
    - initalizes each of the sub command terms
    - initializes another VelocityCommandTerm to do other functionality like visualization
    - in 
"""

@configclass
class Subtask:
    name: str = MISSING
    subterrain: terrain_gen.SubTerrainBaseCfg = terrain_gen.MeshPlaneTerrainCfg()
    base_velocity: mdp.UniformVelocityCommandCfg = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(1e9, 1e9),  # effectively no resampling
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.5, 0.5),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(0.0, 0.0),
        ),
        heading_command=True,
    )
    events: Dict[str, EventTerm] = {}

@configclass
class WalkSubtask(Subtask):
    name = "walk"

@configclass
class RunSubtask(Subtask):
    name = "run"
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(1e9, 1e9),  # effectively no resampling
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(2.0, 2.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-math.pi, math.pi),
            heading=(0.0, 0.0),
        ),
        heading_command=True,
    )

@configclass
class UnevenSubtask(Subtask):
    name = "uneven"
    subterrain = terrain_gen.HfRandomUniformTerrainCfg(noise_range=(0.00, 0.03), noise_step=0.01)

@configclass
class PushSubtask(Subtask):
    name = "push"
    events = {
        "push_robot": EventTerm(
            func=humanoid_mdp.push_by_adding_velocity,
            mode="interval",
            interval_range_s=(2.0, 3.0),
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
                "velocity_range": {
                    "x": (-1.5, 1.5),
                    "y": (-1.5, 1.5),
                },
            },
        )
    }

class T1Baseline_BENCHMARK(T1BaselineCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # define subtasks
        self.subtasks = [WalkSubtask(), RunSubtask(), UnevenSubtask(), PushSubtask()]
        self.subtask_names = [subtask.name for subtask in self.subtasks]
        self.num_subtasks = len(self.subtasks)

        self.reset_configs()

        # create terrain
        self.scene.terrain.terrain_type = "generator"
        self.scene.env_spacing = 0.0
        self.scene.terrain.terrain_generator = terrain_gen.TerrainGeneratorCfg(
            size=(50.0, 10.0),
            num_cols=self.num_subtasks,
            sub_terrains=OrderedDict([(subtask.name, subtask.subterrain) for subtask in self.subtasks]),
            curriculum=True # needed to ensure subterrains are created equally
        )


        # make split base velocity command
        self.commands.base_velocity = split_command_cfg(
            self.commands.base_velocity,
            [subtask.base_velocity for subtask in self.subtasks],
        )

        # add in events from each subtask
        for idx, subtask in enumerate(self.subtasks):
            for event_name, event_cfg in subtask.events.items():
                filtered_event_cfg = EventTerm(
                    func=filtered_func,
                    mode=event_cfg.mode,
                    interval_range_s=event_cfg.interval_range_s,
                    is_global_time=event_cfg.is_global_time,
                    min_step_count_between_reset=event_cfg.min_step_count_between_reset,
                    params={
                        "subtask_idx": idx,
                        "func": event_cfg.func,
                        "params": event_cfg.params,
                    },
                )

                setattr(self.events, f"{subtask.name}_{event_name}", filtered_event_cfg)

                # setattr(
                #     self.events, 
                #     f"{subtask.name}_{event_name}", 
                #     filter_event_cfg(f"{subtask.name}_{event_name}", event_cfg, idx, self.num_subtasks)
                # )

    def reset_configs(self):
        # viewer
        self.viewer = ViewerCfg(
            eye=(-20, 0, 25),
            lookat=(10, 0, 0),
            # eye=(-10.0, 0.0, 15.0),
            # lookat=(25.0, 0.0, 0.0),
            origin_type="world",
            env_index=0,
            asset_name="robot",
        )

        # default env numbers
        self.set_num_envs(4)

        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # disable randomization for play
        self.observations.policy.enable_corruption = False

        #remove event randomization
        self.events.physics_material = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="l[lr]6"),
                # "asset_cfg": SceneEntityCfg("robot", body_names=[".*_foot_link"]),
                "static_friction_range": (0.65, 0.65),
                "dynamic_friction_range": (0.55, 0.55),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 1,
            },
        )
        self.events.base_mass = None
        self.events.reset_base = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "pose_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                },
                "velocity_range": {},
            },
        )

        self.events.reset_joints = EventTerm(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["joint_l[lr].*"]),
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
            },
        )
        # self.events.physics_material.params["static_friction_range"] = (0.65, 0.65)
        # self.events.physics_material.params["dynamic_friction_range"] = (0.55, 0.55)
        # self.events.base_mass.params["mass_distribution_params"] = (0.0, 0.0)
        # self.events.reset_base["pose_range"] = {}
        # self.events.reset_base["velocity_range"] = {}
        # self.events.reset_joints["position_range"] = (0.0, 0.0)
        # self.events.reset_joints["velocity_range"] = (0.0, 0.0)

    
    def set_num_envs(self, num_envs: int):
        self.scene.num_envs = num_envs * self.num_subtasks

    def fix_env_origins(self, env: ManagerBasedRLEnv):
        terrain_generator = env.scene.terrain.cfg.terrain_generator
        print(terrain_generator.sub_terrains.values())
        print(env.scene.terrain.terrain_origins)
        print(env.scene.terrain.env_origins)
        breakpoint()
        # n = env.num_envs // self.num_subtasks
        # env.scene.terrain.cfg.terrain_generator.sub_terrains.values()
        # for i in range(self.num_subtasks):
        #     env.scene.env_origins[i*n:(i+1)*n, 0] = i * self.scene.terrain.terrain_generator.size[0]

def split_command_cfg(cfg: CommandTerm, subterm_cfgs: Sequence[CommandTerm]) -> CommandTerm:
    class SplitCommand(cfg.class_type):
        __name__ = f"SplitCommand_{cfg.class_type.__name__}"
        def __init__(self, cfg: CommandTerm, env: ManagerBasedRLEnv) -> None:
            super().__init__(cfg, env)
            self.subterms = [cfg.class_type(subterms, env) for subterms in subterm_cfgs]
            self.num_subtasks = len(self.subterms)
            self.n = env.num_envs // self.num_subtasks

            self.cmd = torch.zeros(super().command.shape, device=self.device)

        @property
        def command(self) -> torch.Tensor:
            for i in range(self.num_subtasks):
                self.cmd[i*self.n:(i+1)*self.n] = self.subterms[i].command[i*self.n:(i+1)*self.n]
            return self.cmd

        def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
            metrics = super().reset(env_ids)
            for subterm in self.subterms:
                subterm.reset(env_ids)
            
            return metrics
        
        def compute(self, dt: float):
            super().compute(dt)
            for subterm in self.subterms:
                subterm.compute(dt)

    this_mod = sys.modules[__name__]
    setattr(this_mod, SplitCommand.__name__, SplitCommand)
    
    cfg.class_type = SplitCommand
    return cfg

def filtered_func(
    env: ManagerBasedEnv, 
    env_ids: torch.Tensor, 
    subtask_idx: str, 
    func: Callable, 
    params: Dict
):
    # idx = next(i for i, subtask in enumerate(env.cfg.subtasks) if subtask.name == subtask_name)
    n = env.num_envs // env.cfg.num_subtasks
    sub_env_ids = env_ids[(subtask_idx*n <= env_ids) & (env_ids < (subtask_idx+1)*n)]
    if len(sub_env_ids) > 0:
        func(env, sub_env_ids, **params)
# class FilteredFunction:
#     def __init__(self, func, idx: int, num_subtasks: int):
#         self.func = func
#         self.idx = idx
#         self.num_subtasks = num_subtasks

#     def __call__(self, env: ManagerBasedEnv, env_ids: torch.Tensor, *args, **kwargs):
#         n = env.num_envs // self.num_subtasks
#         sub_env_ids = env_ids[(self.idx*n <= env_ids) & (env_ids < (self.idx+1)*n)]
#         if len(sub_env_ids) > 0:
#             self.func(env, sub_env_ids, *args, **kwargs)


# def filtered_func(env: ManagerBasedEnv, env_ids: torch.Tensor, idx: int, num_subtasks: int, func, *args, **kwargs):
#     n = env.num_envs // num_subtasks
#     sub_env_ids = env_ids[(idx*n <= env_ids) & (env_ids < (idx+1)*n)]
#     if len(sub_env_ids) > 0:
#         func(env, sub_env_ids, *args, **kwargs)
    
# def filter_event_cfg(cfg: EventTerm, idx: int, num_subtasks: int):
#     cfg.func = lambda env, env_ids, *args, **kwargs: filtered_func(env, env_ids, idx, num_subtasks, cfg.func, *args, **kwargs)
#     return cfg
# def filter_event_cfg(name: str, cfg: EventTerm, idx: int, num_subtasks: int) -> EventTerm:
#     def filtered_func(env: ManagerBasedEnv, env_ids: torch.Tensor, *args, **kwargs):
#         n = env.num_envs // num_subtasks
#         sub_env_ids = env_ids[idx*n <= env_ids < (env+1)*n]
#         if len(sub_env_ids) > 0:
#             cfg.func(env, sub_env_ids, *args, **kwargs)

#     filtered_func.__name__ = f"filtered_{name}"

#     this_mod = sys.modules[__name__]
#     setattr(this_mod, filtered_func.__name__, filtered_func)
    
#     cfg.func = filtered_func
#     return cfg
