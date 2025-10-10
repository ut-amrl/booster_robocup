from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg

import os

EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
"""Path to the extension source directory."""

##
# Configuration
##

T1_CFG2 = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UrdfFileCfg(
        fix_base= False, #True,
        merge_fixed_joints=False,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=None, damping=None
            )
        ),
        make_instanceable=True,
        # asset_path=f"{EXT_DIR}/data/urdf/t1/t1.urdf",
        asset_path=f"{EXT_DIR}/data/urdf/t1/T1_locomotion.urdf",
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
        pos=(0.0, 0.0, 1.0), # maybe needs nonzero height
        joint_pos={
                   "joint_l[lr]1": 0.0, # -0.2,
                   "joint_l[lr]2": 0.0,
                   "joint_l[lr]3": 0.0,
                   "joint_l[lr]4": 0.0, # should be .4
                   "joint_l[lr]5": 0.0,
                   "joint_l[lr]6": 0.0,
                   },
        # joint_pos={
        #            "joint_l[lr]1": -0.2, # -0.2,
        #            "joint_l[lr]2": 0.0,
        #            "joint_l[lr]3": 0.0,
        #            "joint_l[lr]4": 0.4, # should be .4
        #            "joint_l[lr]5": -0.25,
        #            "joint_l[lr]6": 0.0,
        #            },
    ),
    actuators={
        "hip_pitch_roll": IdealPDActuatorCfg(
            joint_names_expr=["joint_l[lr][1-2]"],
            stiffness=200,
            damping=5.0,
            friction = 1.0,
            effort_limit_sim = {"joint_ll1": 45, "joint_ll2": 30, "joint_lr1": 45, "joint_lr2": 30,}
        ),
        "hip_yaw": IdealPDActuatorCfg(
            joint_names_expr=["joint_l[lr]3"],
            stiffness=200,
            damping=5.0,
            friction = 1.0,
            effort_limit_sim = {"joint_ll3": 30, "joint_lr3": 30,}
        ),
        "knee": IdealPDActuatorCfg(
            joint_names_expr=["joint_l[lr]4"],
            stiffness=200,
            damping=5.0,
            friction = 1.0,
            effort_limit_sim = {"joint_ll4": 60, "joint_lr4": 60,}
        ),
        "ankles": IdealPDActuatorCfg(
            joint_names_expr=["joint_l[lr][5-6]"],
            stiffness=50,
            damping=1.0,
            friction = 1.0,
            effort_limit_sim = {"joint_ll5": 24, "joint_lr5": 24,"joint_ll6": 15, "joint_lr6": 15,}
        ),
    },
)
"""Configuration for the Booster T1 Humanoid robot."""
