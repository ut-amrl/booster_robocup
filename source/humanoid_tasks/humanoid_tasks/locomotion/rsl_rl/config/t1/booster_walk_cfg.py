from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import (
    ManagerBasedRLEnvCfg,
    ManagerBasedRLEnv,
    ViewerCfg,
    VecEnvStepReturn,
)
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import (
    ISAAC_NUCLEUS_DIR,
    NVIDIA_NUCLEUS_DIR,
)
from isaaclab.utils.noise import GaussianNoiseCfg


import isaaclab.envs.mdp as mdp
import humanoid_mdp
from humanoid_assets import T1_CFG2
import torch


##
# Scene definition
##

JOINT_CFG = SceneEntityCfg(
    "robot",
    joint_names=[
        "joint_ll1",
        "joint_ll2",
        "joint_ll3",
        "joint_ll4",
        "joint_ll5",
        "joint_ll6",
        "joint_lr1",
        "joint_lr2",
        "joint_lr3",
        "joint_lr4",
        "joint_lr5",
        "joint_lr6",
    ],
    preserve_order=True,
)
FEET_CFG = SceneEntityCfg("robot", body_names=["ll6", "lr6"])
TRUNK_CFG = SceneEntityCfg(name="robot", body_names="trunk", preserve_order=True)
OTHER_LINKS_CFG = SceneEntityCfg(
    name="robot", body_names="^(?!.*trunk).*", preserve_order=True
)


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    env_spacing: float = 5.0  # spacing between envs

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",  # could also be "plane"
        terrain_generator=None,  # or none
        # max_init_terrain_level=humanoid_mdp.COBBLESTONE_ROAD_CFG.num_rows - 1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Natural/Grass_Countryside.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,  # show origin of each environment
        env_spacing=5.0,
    )
    # robots
    robot: ArticulationCfg = T1_CFG2.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # sensors
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        debug_vis=True,
        update_period=0.002,
    )
    left_foot = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ll6",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/lr6"],
        history_length=3,
        track_air_time=True,
        debug_vis=True,
    )
    right_foot = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/lr6",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/ll6"],
        history_length=3,
        track_air_time=True,
        debug_vis=True,
    )
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.8, 0.8, 0.8), intensity=1000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            color=(1.0, 1.0, 1.0),
            intensity=300.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 12.0),
        rel_standing_envs=0.1,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0, 0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0, 0),
        ),
    )
    gait_cycle = humanoid_mdp.GaitCycleCommandCfg(
        resampling_time_range=(6.0, 8.0),
        range=(1.0, 2.0),
        debug_vis=False,
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    """
    TODO
    - Clip action to joint lims
    - Rewards (foot roll, etc)
    """

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "joint_ll1",
            "joint_ll2",
            "joint_ll3",
            "joint_ll4",
            "joint_ll5",
            "joint_ll6",
            "joint_lr1",
            "joint_lr2",
            "joint_lr3",
            "joint_lr4",
            "joint_lr5",
            "joint_lr6",
        ],
        preserve_order=True,
        scale=1.0,
        use_default_offset=True,
        clip={
            "joint_l[lr][1-6]": (-1.0, 1.0)
        },  # change the clipping to before adding default pose
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        # observation terms (order preserved)
        """Observations for policy group."""
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, noise=GaussianNoiseCfg(mean=0, std=0.01)
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, noise=GaussianNoiseCfg(mean=0, std=0.1)
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        gait_cycle = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "gait_cycle"}
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": JOINT_CFG},
            noise=GaussianNoiseCfg(mean=0, std=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": JOINT_CFG},
            noise=GaussianNoiseCfg(mean=0, std=0.1),
            scale=0.1,
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = None

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for policy group."""

        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, noise=GaussianNoiseCfg(mean=0, std=0.05)
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, noise=GaussianNoiseCfg(mean=0, std=0.1)
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        gait_cycle = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "gait_cycle"}
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": JOINT_CFG},
            noise=GaussianNoiseCfg(mean=0, std=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": JOINT_CFG},
            noise=GaussianNoiseCfg(mean=0, std=0.1),
            scale=0.1,
        )
        actions = ObsTerm(func=mdp.last_action)

        ########### privileged ###########
        # these are different in paper
        body_mass = ObsTerm(
            func=humanoid_mdp.body_mass,
            params={
                "asset_cfg": TRUNK_CFG,
            },
        )
        base_com = ObsTerm(func=humanoid_mdp.base_com, scale=1.0)
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, noise=GaussianNoiseCfg(mean=0, std=0.05)
        )
        base_height = ObsTerm(func=mdp.base_pos_z)
        base_push_force = ObsTerm(
            func=mdp.body_incoming_wrench, params={"asset_cfg": TRUNK_CFG}, scale=0.1
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # interval
    base_external_force_torque = EventTerm(  # how to get these as privileged obs?
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(5.0, 6.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
            "force_range": (-10, 10),
            "torque_range": (-2.0, 2.0),
        },
    )
    base_external_velocity = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(2.0, 2.1),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
            "velocity_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "z": (-0.1, 0.1),
                "roll": (-0.02, 0.02),
                "pitch": (-0.02, 0.02),
                "yaw": (-0.02, 0.02),
            },
        },
    )

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": FEET_CFG,
            "static_friction_range": (0.1, 2.0),
            "dynamic_friction_range": (0.1, 2.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 256,
            "make_consistent": True,
        },
    )
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
            "com_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.1, 0.1)},
        },
    )
    base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
            "mass_distribution_params": [0.8, 1.2],
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    other_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": OTHER_LINKS_CFG,
            "com_range": {
                "x": (-0.005, 0.005),
                "y": (-0.005, 0.005),
                "z": (-0.005, 0.005),
            },
        },
    )
    other_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": OTHER_LINKS_CFG,
            "mass_distribution_params": (0.98, 1.02),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.95, 1.05),
            "damping_distribution_params": (0.95, 1.05),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    joint_parameters = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.0, 2.0),
            "operation": "add",
            "distribution": "uniform",
        },
    )

    # reset
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
        },
    )

    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "position_range": (-0.05, 0.05),
            "velocity_range": (-0.1, 0.1),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    survival = RewTerm(
        func=humanoid_mdp.reward_survival,
        weight=0.25,
        params={},
    )
    tracking_lin_vel_x = RewTerm(
        func=humanoid_mdp.reward_tracking_lin_vel_x,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    tracking_lin_vel_y = RewTerm(
        func=humanoid_mdp.reward_tracking_lin_vel_y,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    tracking_ang_vel_yaw = RewTerm(
        func=humanoid_mdp.reward_tracking_ang_vel_yaw,
        weight=0.5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    base_height = RewTerm(
        func=humanoid_mdp.reward_base_height,
        weight=-20,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    orientation = RewTerm(
        func=humanoid_mdp.reward_orientation,
        weight=-5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    torques = RewTerm(
        func=humanoid_mdp.reward_torques,
        weight=-2.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    torque_tiredness = RewTerm(
        func=humanoid_mdp.reward_torque_tiredness,
        weight=-1.0e-2,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    power = RewTerm(
        func=humanoid_mdp.reward_power,
        weight=-2.0e-3,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    lin_vel_z = RewTerm(
        func=humanoid_mdp.reward_lin_vel_z,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    ang_vel_xy = RewTerm(
        func=humanoid_mdp.reward_ang_vel_xy,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    dof_vel = RewTerm(
        func=humanoid_mdp.reward_dof_vel,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    dof_acc = RewTerm(
        func=humanoid_mdp.reward_dof_acc,
        weight=-1.0e-7,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    root_acc = RewTerm(
        func=humanoid_mdp.reward_root_acc,
        weight=-1.0e-4,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    action_rate = RewTerm(
        func=humanoid_mdp.reward_action_rate,
        weight=-1.0,
    )
    dof_pos_limits = RewTerm(
        func=humanoid_mdp.reward_dof_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    collision = RewTerm(
        func=humanoid_mdp.reward_collision,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",  # "Trunk", "H1", "H2", "AL", "AR", "Waist", "Hip", "Shank", "Ankle"
                body_names=[
                    "trunk",
                    "h1",
                    "h2",
                    "al1",
                    "al2",
                    "al3",
                    "al4",
                    "ar1",
                    "ar2",
                    "ar3",
                    "ar4",
                    "waist",
                    "ll1",
                    "ll2",
                    "ll3",
                    "ll4",
                    "ll5",
                    "lr1",
                    "lr2",
                    "lr3",
                    "lr4",
                    "lr5",
                ],
            )
        },
    )
    feet_slip = RewTerm(
        func=humanoid_mdp.reward_feet_slip, weight=-0.1, params={"asset_cfg": FEET_CFG}
    )
    feet_yaw_diff = RewTerm(
        func=humanoid_mdp.reward_feet_yaw_diff,
        weight=-1.0,
        params={"asset_cfg": FEET_CFG},
    )
    feet_yaw_mean = RewTerm(
        func=humanoid_mdp.reward_feet_yaw_mean,
        weight=-1.0,
        params={"asset_cfg": FEET_CFG},
    )
    feet_roll = RewTerm(
        func=humanoid_mdp.reward_feet_roll, weight=-0.1, params={"asset_cfg": FEET_CFG}
    )
    feet_distance = RewTerm(
        func=humanoid_mdp.reward_feet_distance,
        weight=-1.0,
        params={"asset_cfg": FEET_CFG},
    )
    feet_swing = RewTerm(
        func=humanoid_mdp.reward_feet_swing,
        weight=3.0,  # here
        params={"asset_cfg": FEET_CFG},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    body_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.45, "asset_cfg": SceneEntityCfg("robot")},
    )
    terrain_out_of_bounds = DoneTerm(
        func=humanoid_mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
        time_out=True,
    )


def override_command_range(env, env_ids, old_value, value, num_steps):
    # Override after num_steps
    if env.common_step_counter > num_steps:
        return value
    return mdp.modify_term_cfg.NO_CHANGE


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""


##
# Environment configuration
##


class T1ManagerBasedRLEnv(ManagerBasedRLEnv):
    def __init__(
        self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs
    ):
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        # Call parent step to compute obs, reward, done, info
        obs, rewards, reset_terminated, reset_time_outs, extras = super().step(action)

        # Modify reward buffer in place (clip negatives to 0)
        offset = -self.reward_buf
        self.reward_buf += offset * (self.reward_buf < 0)

        return (
            self.obs_buf,
            self.reward_buf,
            self.reset_terminated,
            self.reset_time_outs,
            self.extras,
        )


@configclass
class T1BaselineCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Viewer
    viewer = ViewerCfg(
        eye=(12.5, 12.5, 10.5),
        #    lookat=(50.0, 0.0, 0.0),
        origin_type="env",
        env_index=0,
        asset_name="robot",
    )  # Scene settings
    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 10  # 50 Hz
        self.episode_length_s = 30.0
        # simulation settings
        self.sim.dt = 0.002  # 200 Hz
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material

        # if self.scene.height_scanner is not None:
        #     self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        self.sim.render.antialiasing_mode = "Off"
        self.sim.render.enable_dlssg = False
        self.sim.render.enable_dl_denoiser = False
        self.sim.render.dlss_mode = 2

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.left_foot.update_period = self.sim.dt
        self.scene.right_foot.update_period = self.sim.dt
        self.observations.critic.enable_corruption = False


class T1Baseline_PLAY(T1BaselineCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None

        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.gait_cycle.range = (1.5, 1.5)

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.base_external_velocity = None
        # self.events.push_robot = None
