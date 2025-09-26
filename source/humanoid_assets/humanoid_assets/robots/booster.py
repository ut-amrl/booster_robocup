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
                   "joint_a[lr]1": 0.2,
                   "joint_al2": -1.35,
                   "joint_ar2": 1.35,
                   "joint_a[lr]3": 0.0,
                   "joint_al4": -0.5,
                   "joint_ar4": 0.5,
                   "joint_waist": 0.0,
                   "joint_l[lr]1": -0.2,
                   "joint_l[lr]2": 0.0,
                   "joint_l[lr]3": 0.0,
                   "joint_l[lr]4": 0.4,
                   "joint_l[lr]5": -0.25,
                   "joint_l[lr]6": 0.0,
                   },
    ),
    actuators={
        "head": ImplicitActuatorCfg(
            joint_names_expr=["joint_h.*"],
            stiffness=20,
            damping=0.2,
            effort_limit_sim=7.0,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=["joint_a[lr][1-4]"],
            stiffness=20,
            damping=0.5,
            effort_limit_sim=10.0,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["joint_waist"],
            stiffness=200,
            damping=5.0,
            effort_limit_sim=30.0,
        ),
        "l1": ImplicitActuatorCfg(
            joint_names_expr=["joint_l[lr]1"],
            stiffness=200,
            damping=5.0,
            effort_limit_sim=60.0,
        ),
        "l2": ImplicitActuatorCfg(
            joint_names_expr=["joint_l[lr]2"],
            stiffness=200,
            damping=5.0,
            effort_limit_sim=25.0,
        ),
        "l3": ImplicitActuatorCfg(
            joint_names_expr=["joint_l[lr]3"],
            stiffness=200,
            damping=5.0,
            effort_limit_sim=30.0,
        ),
        "l4": ImplicitActuatorCfg(
            joint_names_expr=["joint_l[lr]4"],
            stiffness=200,
            damping=5.0,
            effort_limit_sim=60.0,
        ),
        "l5": ImplicitActuatorCfg(
            joint_names_expr=["joint_l[lr]5"],
            stiffness=50,
            damping=1.0,
            effort_limit_sim=24.0,
        ),
        "l6": ImplicitActuatorCfg(
            joint_names_expr=["joint_l[lr]6"],
            stiffness=50,
            damping=1.0,
            effort_limit_sim=15.0,
        ),
    },
)
"""Configuration for the Booster T1 Humanoid robot."""

# This is a placholder. The G1 has actual values here.
T1_ACTION_SCALE = {}
for a in T1_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            T1_ACTION_SCALE[n] = 0.25 * e[n] / s[n]