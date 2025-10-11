from collections import OrderedDict
from dataclasses import MISSING
import math
from typing import Dict

import humanoid_mdp
import isaaclab.terrains as terrain_gen
from isaaclab.envs import ViewerCfg

from isaaclab.managers import EventTermCfg
from isaaclab.managers import TerminationTermCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp
from .rsl_rl_cfg import T1BaselineCfg
from .benchmark_cfg_utils import (
    split_command_cfg,
    filtered_func,
    reset_root_state_uniform_once,
    subterrain_out_of_bounds,
)


@configclass
class Subtask:
    name: str = MISSING
    subterrain: terrain_gen.SubTerrainBaseCfg = terrain_gen.MeshPlaneTerrainCfg()
    base_velocity: mdp.UniformVelocityCommandCfg = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(1e9, 1e9),  # effectively no resampling
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(1.0, 1.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-math.pi, math.pi),
            heading=(0.0, 0.0),
        ),
        heading_command=True,
    )
    events: Dict[str, EventTermCfg] = {}


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
    subterrain = terrain_gen.HfRandomUniformTerrainCfg(
        noise_range=(0.00, 0.03), noise_step=0.005
    )


@configclass
class PushSubtask(Subtask):
    name = "push"
    events = {
        "push_robot": EventTermCfg(
            func=humanoid_mdp.push_by_adding_velocity,
            mode="interval",
            interval_range_s=(2.0, 3.0),
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
                "velocity_range": {
                    "x": (-1.0, 1.0),
                    "y": (-1.0, 1.0),
                },
            },
        )
    }


class T1Baseline_BENCHMARK(T1BaselineCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()
        self.setup_benchmark_mode()

        # define subtasks
        self.subtasks = [WalkSubtask(), RunSubtask(), UnevenSubtask(), PushSubtask()]
        self.subtask_names = [subtask.name for subtask in self.subtasks]
        self.num_subtasks = len(self.subtasks)

        self.setup_subtasks()
        self.set_num_envs(32)

    def set_num_envs(self, num_envs: int):
        self.scene.num_envs = num_envs * self.num_subtasks

    def setup_subtasks(self):
        # create split terrain
        self.scene.terrain.terrain_type = "generator"
        self.scene.env_spacing = 0.0
        self.scene.terrain.terrain_generator = terrain_gen.TerrainGeneratorCfg(
            size=(100.0, 10.0),
            num_cols=self.num_subtasks,
            border_width=3,
            sub_terrains=OrderedDict(
                [(subtask.name, subtask.subterrain) for subtask in self.subtasks]
            ),
            curriculum=True,  # needed to ensure subterrains are created equally
        )

        # make split base velocity command
        self.commands.base_velocity = split_command_cfg(
            self.commands.base_velocity,
            [subtask.base_velocity for subtask in self.subtasks],
        )

        # add in events from each subtask
        for idx, subtask in enumerate(self.subtasks):
            for event_name, event_cfg in subtask.events.items():
                filtered_event_cfg = EventTermCfg(
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

    def setup_benchmark_mode(self):
        # viewer
        self.viewer = ViewerCfg(
            eye=(-20, 0, 25),
            lookat=(10, 0, 0),
            origin_type="world",
            env_index=0,
            asset_name="robot",
        )

        # disable policy noise
        self.observations.policy.enable_corruption = False

        # remove interval events
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # remove startup randomization
        self.events.base_mass = None
        self.events.physics_material = EventTermCfg(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="l[lr]6"),
                "static_friction_range": (0.65, 0.65),
                "dynamic_friction_range": (0.55, 0.55),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 1,
            },
        )

        # reset events
        self.events.reset_base = EventTermCfg(
            func=reset_root_state_uniform_once,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "pose_range": {},
                "velocity_range": {},
            },
        )
        self.events.reset_joints = EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["joint_l[lr].*"]),
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
            },
        )

        # terminations
        self.terminations.terrain_out_of_bounds = None
        self.terminations.subterrain_out_of_bounds = TerminationTermCfg(
            func=subterrain_out_of_bounds,
            params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 0.5},
            time_out=True,
        )
