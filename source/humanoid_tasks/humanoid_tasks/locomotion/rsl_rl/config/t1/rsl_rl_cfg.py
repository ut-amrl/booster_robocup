from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
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
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise
from isaaclab.sensors import RayCasterCfg, patterns


import isaaclab.envs.mdp as mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as locomotion_mdp
import humanoid_mdp
from humanoid_assets import T1_CFG


##
# Scene definition
##


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
    robot: ArticulationCfg = T1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # sensors
    # height_scanner = RayCasterCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/trunk",
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    #     ray_alignment="yaw",
    #     pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
    #     debug_vis=True,  # visualize the raycast. Turn off for speed
    #     mesh_prim_paths=["/World/ground"],
    # )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        debug_vis=True,
        update_period=0.005,
    )
    left_foot = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ll6",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/lr6"],
        # prim_path="{ENV_REGEX_NS}/Robot/left_foot_link",
        # filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/right_foot_link"],
        history_length=3,
        track_air_time=True,
        debug_vis=True,
    )
    right_foot = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/lr6",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/ll6"],
        # prim_path="{ENV_REGEX_NS}/Robot/right_foot_link",
        # filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/left_foot_link"],
        history_length=3,
        track_air_time=True,
        debug_vis=True,
    )
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.8, 0.8, 0.8), intensity=1000.0),
    )
    # sky_light = AssetBaseCfg(
    #     prim_path="/World/skyLight",
    #     spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1000.0),
    # )

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
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 2.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-1.0, 1.0),
        ),
    )
    frequency = humanoid_mdp.FrequencyCommandCfg(
        sensor_cfg=SceneEntityCfg("contact_forces", body_names="l[lr]6"),
        resampling_time_range=(6.0, 8.0),
        range=(2.0, 4.0),
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
            "joint_al1",
            "joint_al2",
            "joint_al3",
            "joint_al4",
            "joint_ar1",
            "joint_ar2",
            "joint_ar3",
            "joint_ar4",
            "joint_waist",
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
        scale=0.2,
        use_default_offset=True,
        clip=None,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        # observation terms (order preserved)
        """Observations for policy group."""
        # height_scan = ObsTerm(
        #     func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     noise=Unoise(n_min=-0.1, n_max=0.1),
        #     clip=(-1.0, 1.0),
        # )
        # foot_force = ObsTerm(
        #     func=humanoid_mdp.contact_sensor,
        #     params={
        #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names="l[lr]6")
        #     },
        #     # noise=Unoise(n_min=-5, n_max=5),
        # )
        # ground_friction = ObsTerm(
        #     func=humanoid_mdp.contact_friction,
        #     params={"asset_cfg": SceneEntityCfg("robot", body_names="l[lr]6")},
        #     noise=Unoise(n_min=-0.01, n_max=0.01),
        # )
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "joint_al1",
                        "joint_al2",
                        "joint_al3",
                        "joint_al4",
                        "joint_ar1",
                        "joint_ar2",
                        "joint_ar3",
                        "joint_ar4",
                        "joint_waist",
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
            },
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "joint_al1",
                        "joint_al2",
                        "joint_al3",
                        "joint_al4",
                        "joint_ar1",
                        "joint_ar2",
                        "joint_ar3",
                        "joint_ar4",
                        "joint_waist",
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
            },
            noise=Unoise(n_min=-0.5, n_max=0.5),
        )
        actions = ObsTerm(func=mdp.last_action)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        # frequency_command = ObsTerm(
        #     func=mdp.generated_commands, params={"command_name": "frequency"}
        # )

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = None

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        # frequency_command = ObsTerm(
        #     func=mdp.generated_commands, params={"command_name": "frequency"}
        # )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "joint_al1",
                        "joint_al2",
                        "joint_al3",
                        "joint_al4",
                        "joint_ar1",
                        "joint_ar2",
                        "joint_ar3",
                        "joint_ar4",
                        "joint_waist",
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
            },
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "joint_al1",
                        "joint_al2",
                        "joint_al3",
                        "joint_al4",
                        "joint_ar1",
                        "joint_ar2",
                        "joint_ar3",
                        "joint_ar4",
                        "joint_waist",
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
            },
            noise=Unoise(n_min=-0.5, n_max=0.5),
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # NEW USING YAML
    # i'm very confident this does nothing but gonna leave it here just in case
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
            "force_range": (-0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )



    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="l[lr]6"),
            # "asset_cfg": SceneEntityCfg("robot", body_names=[".*_foot_link"]),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # link_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "mass_distribution_params": (0.98, 1.02),
    #         "operation": "scale",
    #         "distribution": "uniform",
    #     },
    # )

    base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
            # "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            "mass_distribution_params": (-2.0, 2.0),
            "operation": "add",
            "distribution": "uniform",
        },
    )

    # actuator_gains = EventTerm(
    #     func=mdp.randomize_actuator_gains,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "stiffness_distribution_params": (0.9, 1.1),
    #         "damping_distribution_params": (1.0, 2.0),
    #         "operation": "scale",
    #         "distribution": "uniform",
    #     },
    # )

    # reset
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"yaw": (-math.pi, math.pi)},
            "velocity_range": {
                "x": (-1.0, 1.0),
                "y": (-0.5, 0.5),
                "yaw": (-1.0, 1.0),
            },
        },
    )

    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint_l[lr].*"]),
            # "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "position_range": (-0.2, 0.2),
            "velocity_range": (-2.5, 2.5),
        },
    )

    # interval
    push_robot = EventTerm(
        func=humanoid_mdp.push_by_adding_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
            # "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                # "yaw": (-0.5, 0.5),
            },
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    base_linear_velocity = RewTerm(
        func=humanoid_mdp.base_linear_velocity_reward,
        weight=5.0,  # 5
        params={
            "std": 1.0,
            "ramp_rate": 0.5,
            "ramp_at_vel": 1.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    base_angular_velocity = RewTerm(
        func=humanoid_mdp.base_angular_velocity_reward,
        weight=5.0,
        params={"std": 1.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    # air_time = RewTerm(
    #     func=humanoid_mdp.air_time_reward_cmd_biped,
    #     weight=5.0,
    #     params={
    #         "sensor_cfg": SceneEntityCfg(
    #             "contact_forces",
    #             body_names="l[lr]6",
    #             # body_names=[".*_foot_link"]
    #         )
    #     },
    # )

    # NEW
    feet_air_time = RewTerm(
        func=locomotion_mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name" : "base_velocity",
            "threshold" : 0.4,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names="l[lr]6",
                preserve_order=False,
            )
        }
    )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="joint_l[lr]6"),
        }
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint_l[lr][1-3]"]),
        }
    )
    joint_lateral_leg_pos = RewTerm(
        func=humanoid_mdp.joint_position_penalty,
        weight=-10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint_l[lr]2"]),
            "stand_still_scale": 10.0,
        }
    )
    joint_lateral_leg_vel = RewTerm(
        func=humanoid_mdp.joint_velocity_penalty,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["joint_l[lr]2"],
            )
        },
    )
    # END OF NEW
    foot_clearance = RewTerm(
        func=humanoid_mdp.foot_clearance_reward,
        weight=0.5,
        params={
            "std": 0.05,
            "tanh_multiplier": 2.0,
            "target_height": 0.15,
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names="l[lr]6",
                # body_names=[".*_foot_link"]
            ),
        },
    )
    gait = RewTerm(
        func=humanoid_mdp.gait_reward_biped,
        weight=10.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names="l[lr]6",
                # body_names=[".*_foot_link"],
                preserve_order=False,
            ),
            "std": 0.1,
            "max_error": 0.2,
        },
    )
    # base_height = RewTerm(
    #     func=humanoid_mdp.target_base_height, weight=1.0, params={"target_height": 0.68}
    # )
    # alive = RewTerm(func=mdp.is_alive, weight=5.0)

    # -- penalties
    angular_motion = RewTerm(
        func=humanoid_mdp.angular_motion_penalty, 
        weight=-0.5,
    )
    base_orientation = RewTerm(
        func=humanoid_mdp.base_orientation_penalty, 
        weight=-1.0
    )
    foot_slip = RewTerm(
        func=humanoid_mdp.foot_slip_penalty,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names="l[lr]6",
                # body_names=[".*_foot_link"]
            ),
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names="l[lr]6",
                # body_names=[".*_foot_link"]
            ),
            "threshold": 1.0,
        },
    )
    action_smoothness = RewTerm(
        func=humanoid_mdp.action_smoothness_penalty, 
        weight=-1.0,
    )
    air_time_variance = RewTerm(
        func=humanoid_mdp.air_time_variance_penalty,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names="l[lr]6",
                # body_names=[".*_foot_link"]
            )
        },
    )
    foot_impact = RewTerm(
        func=humanoid_mdp.foot_impact_penalty,
        weight=-60.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names="l[lr]6",
                # body_names=[".*_foot_link"]
            ),
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names="l[lr]6",
                # body_names=[".*_foot_link"]
            ),
            "cutoff": 10.0,
        },
    )
    joint_arm_pos = RewTerm(
        func=humanoid_mdp.joint_position_penalty,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["joint_a[lr].*", "joint_l[lr]2", "joint_waist"]
            ),
            "stand_still_scale": 10.0,
        },
    )
    joint_leg_pos = RewTerm(
        func=humanoid_mdp.joint_position_penalty,
        weight=-0.001,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint_l[lr].*"]),
            "stand_still_scale": 10.0,
        },
    )
    # TODO make in terms of body wrt to ground, not joint angle
    # ankle_roll_joint_pos_penalty = RewTerm(
    #     func=humanoid_mdp.joint_position_penalty,
    #     weight=-1.0,
    #     params={"asset_cfg": SceneEntityCfg(
    #         "robot",
    #         joint_names=["joint_l[lr]6"]
    #     ), "stand_still_scale": 10.0},
    # )
    joint_vel = RewTerm(
        func=humanoid_mdp.joint_velocity_penalty,
        weight=-1.0e-2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["joint_[al][lr].*", "joint_waist"],
                # joint_names=".*"
            )
        },
    )
    joint_acc = RewTerm(
        func=humanoid_mdp.joint_acceleration_penalty,
        weight=-1.0e-4,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["joint_[al][lr].*", "joint_waist"],
                # joint_names=".*"
            )
        },
    )
    # ankle_joint_acc = RewTerm(
    #     func=humanoid_mdp.joint_acceleration_penalty,
    #     weight=-1.0e-3,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint_l[lr][56]"])},
    # )
    # arm_joint_acc = RewTerm(
    #     func=humanoid_mdp.joint_acceleration_penalty,
    #     weight=-1.0e-3,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint_a[lr].*"])},
    # )
    joint_torques = RewTerm(
        func=humanoid_mdp.joint_torques_penalty,
        weight=-5.0e-4,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["joint_[al][lr].*", "joint_waist"],
                # joint_names=".*"
            )
        },
    )
    # joint_arm_torques = RewTerm(
    #     func=humanoid_mdp.joint_torques_penalty,
    #     weight=-5.0e-3,
    #     params={"asset_cfg": SceneEntityCfg(
    #         "robot",
    #         # joint_names=["joint_a[lr].*", "joint_waist"]
    #         joint_names=[
    #             ".*_Shoulder_Pitch",
    #             ".*_Shoulder_Roll",
    #             ".*_Elbow_Pitch",
    #             ".*_Elbow_Yaw",
    #             "Waist",
    #         ]
    #     )},
    # )
    # foot_orientation = RewTerm(
    #     func=humanoid_mdp.foot_orientation,
    #     weight=-15.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             body_names="l[lr]6",
    #             # body_names=[".*_foot_link"]
    #         )
    #     },
    # )
    # foot_lateral_penalty = RewTerm(
    #     func=humanoid_mdp.foot_lateral_penalty, weight=-15.0,
    #     params={"asset_cfg": SceneEntityCfg(
    #         "robot",
    #         body_names="l[lr]6"
    #         # body_names=[".*_foot_link"]
    #     )}
    # )
    # feet_yaw_diff_penalty = RewTerm(
    #     func=humanoid_mdp.feet_yaw_diff_penalty,
    #     weight=-5.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             body_names="l[lr]6",
    #             # body_names=[".*_foot_link"]
    #         )
    #     },
    # )

    # foot_distance = RewTerm(
    #     func=humanoid_mdp.foot_distance,
    #     weight=-10.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             body_names="l[lr]6",
    #             # body_names=[".*_foot_link"]
    #         ),
    #         "min_dist": 0.3,
    #     },
    # )
    # heel_toe_stepping_penalty = RewTerm(
    #     func=humanoid_mdp.heel_toe_stepping_penalty, weight=-1.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=["l[lr]6"]),
    #         "sensor_cfg": SceneEntityCfg(
    #             "contact_forces",
    #             body_names="l[lr]6"
    #             # body_names=[".*_foot_link"]
    #         ),
    #     }
    # )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    body_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.4, "asset_cfg": SceneEntityCfg("robot")},
    )
    # non_foot_contact = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={"sensor_cfg": SceneEntityCfg(
    #         "contact_forces",
    #         # body_names="l[lr][1-5]"
    #         body_names=[
    #             "Hip_Pitch_.*",
    #             "Hip_Roll_.*",
    #             "Hip_Yaw_.*",
    #             "Shank_.*",
    #             "Ankle_Cross_.*"
    #         ]
    #     ), "threshold": 1.0},
    # )
    # left_foot_contact = DoneTerm(
    #     func=humanoid_mdp.illegal_contact_filtered,
    #     params={
    #         "threshold": 1.0,
    #         "sensor_cfg": SceneEntityCfg(
    #             "left_foot",
    #             body_names="ll6",
    #             # body_names="left_foot_link"
    #         ),
    #     },
    # )
    # right_foot_contact = DoneTerm(
    #     func=humanoid_mdp.illegal_contact_filtered,
    #     params={
    #         "threshold": 1.0,
    #         "sensor_cfg": SceneEntityCfg(
    #             "right_foot",
    #             # body_names="lr6"
    #             body_names="right_foot_link"
    #         ),
    #     }
    # )
    terrain_out_of_bounds = DoneTerm(
        func=humanoid_mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
        time_out=True,
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=humanoid_mdp.terrain_levels_vel)


##
# Environment configuration
##


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
    # curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 4  # 50 Hz
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005  # 200 Hz
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

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        # self.events.base_external_force_torque = None
        # self.events.push_robot = None
