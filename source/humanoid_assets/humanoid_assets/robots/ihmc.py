from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, DelayedPDActuatorCfg, IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg

import os
EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
"""Path to the extension source directory."""

##
# Configuration
##
ActuatorCfg = IdealPDActuatorCfg
STIFF = True

ALEXANDER_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        root_link_name="TORSO_LINK",
        link_density=0.001,
        merge_fixed_joints=True,
        convert_mimic_joints_to_normal_joints=False,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None)
        ),
        collider_type="convex_hull",
        self_collision=True,
        replace_cylinders_with_capsules=False,
        scale=None,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        fixed_tendons_props=None,
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(
            drive_type="force",
        ),
        visual_material_path="material",
        visual_material=None,
        make_instanceable=True,
        asset_path=f"{EXT_DIR}/data/urdf/alexander/alexander_v1.fullBody.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        joint_pos={
            "NECK_Z": 0.0,
            "NECK_Y": 0.0,
            "SPINE_Z": 0.0,
            "LEFT_SHOULDER_[XYZ]": 0.0,
            "RIGHT_SHOULDER_[XYZ]": 0.0,
            "LEFT_ELBOW_Y": 0.0,
            "RIGHT_ELBOW_Y": 0.0,
            "LEFT_WRIST_[XZ]": 0.0,
            "RIGHT_WRIST_[XZ]": 0.0,
            "LEFT_GRIPPER_Z": 0.0,
            "RIGHT_GRIPPER_Z": 0.0,
            "LEFT_HIP_[XYZ]": 0.0,
            "RIGHT_HIP_[XYZ]": 0.0,
            "LEFT_KNEE_Y": 0.0,
            "RIGHT_KNEE_Y": 0.0,
            "LEFT_ANKLE_[XY]": 0.0,
            "RIGHT_ANKLE_[XY]": 0.0,
        },
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "head": ActuatorCfg(
            joint_names_expr=["NECK_Z", "NECK_Y"],
            stiffness=20,
            damping=0.2 if STIFF else 2,
            # effort_limit_sim=7.0,
            # velocity_limit_sim=50.0,
            effort_limit=7.0,
            velocity_limit=50.0,
        ),
        "arms": ActuatorCfg(
            joint_names_expr=["LEFT_SHOULDER_[XYZ]",
                              "RIGHT_SHOULDER_[XYZ]",
                              "LEFT_ELBOW_Y",
                              "RIGHT_ELBOW_Y"],
            stiffness=20,
            damping=0.5 if STIFF else 2,
            # effort_limit_sim=10.0,
            # velocity_limit_sim=50.0,
            effort_limit=10.0,
            velocity_limit=50.0,
        ),
        "hands": ActuatorCfg(
            joint_names_expr=["LEFT_WRIST_[XZ]", 
                    "RIGHT_WRIST_[XZ]",
                    "LEFT_GRIPPER_Z",
                    "RIGHT_GRIPPER_Z"],
            stiffness=20,
            damping=0.5 if STIFF else 2,
            # effort_limit_sim=10.0,
            # velocity_limit_sim=50.0,
            effort_limit=10.0,
            velocity_limit=50.0,
        ),
        "waist": ActuatorCfg(
            joint_names_expr=["SPINE_Z"],
            stiffness=200 if STIFF else 50,
            damping=3,
            # effort_limit_sim=30.0,
            # velocity_limit_sim=50.0,
            effort_limit=30.0,
            velocity_limit=50.0,
        ),
        "legs1": ActuatorCfg(
            joint_names_expr=[".*_HIP_Z"],
            stiffness=200 if STIFF else 30,
            damping=5 if STIFF else 3,
            # effort_limit_sim=60.0,
            # velocity_limit_sim=50.0,
            effort_limit=60.0,
            velocity_limit=50.0,
        ),
        "legs2": ActuatorCfg(
            joint_names_expr=[".*_HIP_X"],
            stiffness=200 if STIFF else 30,
            damping=5 if STIFF else 3,
            # effort_limit_sim=25.0,
            # velocity_limit_sim=50.0,
            effort_limit=25.0,
            velocity_limit=50.0,
        ),
        "legs3": ActuatorCfg(
            joint_names_expr=[".*_HIP_Y"],
            stiffness=200 if STIFF else 30,
            damping=5 if STIFF else 3,
            # effort_limit_sim=30.0,
            # velocity_limit_sim=50.0,
            effort_limit=30.0,
            velocity_limit=50.0,
        ),
        "legs4": ActuatorCfg(
            joint_names_expr=[".*_KNEE_Y"],
            stiffness=200 if STIFF else 30,
            damping=5 if STIFF else 3,
            # effort_limit_sim=60.0,
            # velocity_limit_sim=50.0,
            effort_limit=60.0,
            velocity_limit=50.0,
        ),
        "legs5": ActuatorCfg(
            joint_names_expr=[".*_ANKLE_X"],
            stiffness=50 if STIFF else 10,
            damping=1 if STIFF else 3,
            # effort_limit_sim=24.0,
            # velocity_limit_sim=50.0,
            effort_limit=24.0,
            velocity_limit=50.0,
        ),
        "legs6": ActuatorCfg(
            joint_names_expr=[".*_ANKLE_Y"],
            stiffness=50 if STIFF else 10,
            damping=1 if STIFF else 3,
            # effort_limit_sim=24.0,
            # velocity_limit_sim=50.0,
            effort_limit=24.0,
            velocity_limit=50.0,
        ),
    }, # type: ignore
)
"""Configuration for the Booster T1 Humanoid robot."""

