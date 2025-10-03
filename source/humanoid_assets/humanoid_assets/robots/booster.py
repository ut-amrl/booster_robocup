from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

import os

EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
"""Path to the extension source directory."""

##
# Configuration
##

T1_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=False,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=None, damping=None
            )
        ),
        make_instanceable=True,
        asset_path=f"{EXT_DIR}/data/urdf/t1/t1.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.7),
        joint_pos={"joint_h.*": 0.0,
                   "joint_a[lr]1": 0.25,
                   "joint_al2": -1.4,
                   "joint_al3": 0.0,
                   "joint_al4": -0.5,
                   "joint_ar2": 1.4,
                   "joint_ar3": 0.0,
                   "joint_ar4": 0.5,
                   "joint_waist": 0.0,
                   "joint_l[lr]1": -0.4,
                   "joint_l[lr]2": 0.0,
                   "joint_l[lr]3": 0.0,
                   "joint_l[lr]4": 0.6,
                   "joint_l[lr]5": -0.25,
                   "joint_l[lr]6": 0.0,
                   },
    ),
    actuators={
        "head": ImplicitActuatorCfg(
            joint_names_expr=["joint_h.*"],
            stiffness=20,
            damping=0.1,
        ),
        "shoulder_pitch" : ImplicitActuatorCfg(
            joint_names_expr=["joint_a[lr]1"],
            stiffness=20,
            damping=0.5,
        ),
        "shoulder_roll" : ImplicitActuatorCfg(
            joint_names_expr=["joint_a[lr]2"],
            stiffness=20,
            damping=1.5,
        ),
        "elbows": ImplicitActuatorCfg(
            joint_names_expr=["joint_a[lr][3-4]"],
            stiffness=20,
            damping=0.2,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["joint_waist"],
            stiffness=200,
            damping=5.0,
        ),
        "hip_pitch_roll": ImplicitActuatorCfg(
            joint_names_expr=["joint_l[lr][1-2]"],
            stiffness=200,
            damping=7.5,
        ),
        "hip_yaw": ImplicitActuatorCfg(
            joint_names_expr=["joint_l[lr]3"],
            stiffness=200,
            damping=3.0,
        ),
        "knee": ImplicitActuatorCfg(
            joint_names_expr=["joint_l[lr]4"],
            stiffness=200,
            damping=5.5,
        ),
        "ankles": ImplicitActuatorCfg(
            joint_names_expr=["joint_l[lr][5-6]"],
            stiffness=150,
            damping=0.5,
        ),
    },
)
"""Configuration for the Booster T1 Humanoid robot."""
