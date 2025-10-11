from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
import isaaclab.utils.math as math_utils
from isaaclab.utils.array import convert_to_torch
from einops import rearrange
from humanoid_mdp import GaitCycleCommand
from humanoid_mdp import foot_orientation, joint_acceleration_penalty


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnvCfg

########## Utils ###########

def get_feet_contact(asset: Articulation, env: ManagerBasedRLEnvCfg, body_ids):
    feet_edge_relative_pos = [[ 0.1215,  0.05, -0.03],
            [ 0.1215, -0.05, -0.03],
            [-0.1015,  0.05, -0.03],
            [-0.1015, -0.05, -0.03]] # x,y,z [m]
    feet_edge_relative_pos = convert_to_torch(feet_edge_relative_pos).unsqueeze(0).unsqueeze(0).expand(env.num_envs, 2, -1, -1).to(device = env.device) # [num_envs, num_feet, num_points, 3]
    feet_pos = asset.data.body_pos_w[:, body_ids, :].unsqueeze(2).expand(-1, -1, feet_edge_relative_pos.shape[2], -1).reshape(-1, 3)
    feet_quat = asset.data.body_quat_w[:, body_ids, :].unsqueeze(2).expand(-1, -1, feet_edge_relative_pos.shape[2], -1).reshape(-1, 4)
    feet_edge_pos = math_utils.quat_apply(feet_quat, feet_edge_relative_pos.reshape(-1, 3)) + feet_pos # [num_envs * num_feet * num_points, 3]
    feet_contact = torch.any(
        (feet_edge_pos[:, 2] - 0 < 0.01).reshape(
            env.num_envs, 2, feet_edge_relative_pos.shape[2]
        ),
        dim=2,
    )
    return feet_contact

########## Booster Gym Rewards ###########
def reward_survival(env: ManagerBasedRLEnvCfg):
    return torch.ones(env.num_envs, dtype=torch.float, device=env.device)

def reward_tracking_lin_vel_x(
    env: ManagerBasedRLEnvCfg,
    asset_cfg: SceneEntityCfg,
    tracking_sigma: float = 0.25
) -> torch.Tensor:
    # Tracking of linear velocity commands (x axes)
    asset = env.scene[asset_cfg.name]
    command_x = env.command_manager.get_command("base_velocity")[:, 0]
    robot_x = asset.data.root_lin_vel_b[:, 0]
    return torch.exp(-torch.square(command_x - robot_x) / tracking_sigma)

def reward_tracking_lin_vel_y(
    env: ManagerBasedRLEnvCfg,
    asset_cfg: SceneEntityCfg,
    tracking_sigma: float = 0.25,
) -> torch.Tensor:
    """Reward tracking of commanded linear velocity along y-axis."""
    asset = env.scene[asset_cfg.name]
    command_y = env.command_manager.get_command("base_velocity")[:, 1]
    robot_y = asset.data.root_lin_vel_b[:, 1]
    return torch.exp(-torch.square(command_y - robot_y) / tracking_sigma)

def reward_tracking_ang_vel_yaw(
    env: ManagerBasedRLEnvCfg,
    asset_cfg: SceneEntityCfg,
    tracking_sigma: float = 0.25,
) -> torch.Tensor:
    """Reward tracking of commanded angular velocity around z-axis (yaw)."""
    asset = env.scene[asset_cfg.name]
    command_yaw = env.command_manager.get_command("base_velocity")[:, 2]
    robot_yaw = asset.data.root_ang_vel_b[:, 2]
    return torch.exp(-torch.square(command_yaw - robot_yaw) / tracking_sigma)

def reward_base_height(
    env: ManagerBasedRLEnvCfg,
    asset_cfg: SceneEntityCfg,
    base_height_target: float = 0.68,
) -> torch.Tensor:
    """Reward tracking of base height above terrain."""
    asset = env.scene[asset_cfg.name]
    
    # robot base position in world frame
    base_pos = asset.data.root_pos_w  # shape: [num_envs, 3]
    
    # terrain height under robot’s xy location
    # terrain_height = env.scene.terrain.height_field(base_pos[:, :2])
    terrain_height = 0 # zero for now since terrain is flat
    
    # compute actual base height above terrain
    base_height = base_pos[:, 2] - terrain_height
    
    # squared error penalty from target height
    return torch.square(base_height - base_height_target)

def reward_lin_vel_z(
    env: ManagerBasedRLEnvCfg,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize z-axis base linear velocity."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def reward_ang_vel_xy(
    env: ManagerBasedRLEnvCfg,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize xy angular velocity of the base."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=-1)


def reward_orientation(
    env: ManagerBasedRLEnvCfg,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize deviation from upright orientation (using projected gravity)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=-1)


def reward_torques(
    env: ManagerBasedRLEnvCfg,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize joint torques."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque), dim=-1)


def reward_dof_vel(
    env: ManagerBasedRLEnvCfg,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize joint velocities."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel), dim=-1)

def reward_dof_acc(
    env: ManagerBasedRLEnvCfg,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize joint accelerations (finite difference)."""
    return joint_acceleration_penalty(env, asset_cfg).square()


def reward_root_acc(
    env: ManagerBasedRLEnvCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), # assume trunk link only
) -> torch.Tensor:
    """Penalize squared base linear and angular acceleration"""
    asset: Articulation = env.scene[asset_cfg.name]
    root_acc = asset.data.body_acc_w # shape (num_instances, 1, 6)
    root_acc = root_acc[:,0,:]
    return torch.sum(torch.square(root_acc), dim = 1)


def reward_action_rate(
    env: ManagerBasedRLEnvCfg
) -> torch.Tensor:
    """Penalize large instantaneous changes in the network action output"""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=-1)

def reward_torque_tiredness(    # torque limits come from urdf
    env: ManagerBasedRLEnvCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    max_torques = asset.data.joint_effort_limits[:, asset_cfg.joint_ids]
    torques = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(torques / max_torques), dim=-1)


def reward_power(
    env: ManagerBasedRLEnvCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    torques = asset.data.applied_torque[:, asset_cfg.joint_ids]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.sum((torques * joint_vel).clip(min = 0), dim=-1)

def reward_dof_pos_limits(env: ManagerBasedRLEnvCfg,
    soft_limit: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    lower_limit = asset.data.joint_pos_limits[:, asset_cfg.joint_ids, 0]
    upper_limit = asset.data.joint_pos_limits[:, asset_cfg.joint_ids, 1]
    margin = 0.5 * (upper_limit - lower_limit)
    soft_lower_limit = lower_limit + (1-soft_limit) * margin
    soft_upper_limit = upper_limit - (1-soft_limit) * margin
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(((joint_pos < soft_lower_limit) | (joint_pos > soft_upper_limit)).float(), dim=-1)

def reward_collision(env: ManagerBasedRLEnvCfg,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0
) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # sum over contacts for each environment
    return torch.sum(is_contact, dim=1)

def reward_feet_swing(
    env: ManagerBasedRLEnvCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    swing_period: float = 0.2,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    # gait_process = env.observation_manager.find_terms("gait_cycle")[0]
    gait_command: GaitCycleCommand = env.command_manager.get_term("gait_cycle")
    gait_process = gait_command.phase 
    left_swing = (torch.abs(gait_process - 0.25) < 0.5 * swing_period).squeeze()
    right_swing = (torch.abs(gait_process - 0.75) < 0.5 * swing_period).squeeze()
    feet_contact = get_feet_contact(asset, env, asset_cfg.body_ids)
    return (left_swing & ~feet_contact[:, 0]).float() + (right_swing & ~feet_contact[:, 1]).float()

def reward_feet_slip( # this seems right?
    env: ManagerBasedRLEnvCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.body_com_vel_w[:, asset_cfg.body_ids] # assume feet only
    feet_contact = get_feet_contact(asset, env, asset_cfg.body_ids)
    return torch.sum(
        torch.square(body_vel).sum(dim=-1) * feet_contact.float(),
        dim=-1,
    )

def reward_feet_yaw(
    env: ManagerBasedRLEnvCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    return foot_orientation(env, asset_cfg)

def reward_feet_yaw_diff(
    env: ManagerBasedRLEnvCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    feet_quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids]
    feet_quat = rearrange(feet_quat, 'b n c -> (b n) c')
    feet_yaw = math_utils.euler_xyz_from_quat(feet_quat)[-1]
    feet_yaw = rearrange(feet_yaw, '(b n)-> b n', n = len(asset_cfg.body_ids))
    return torch.square((feet_yaw[:, 1] - feet_yaw[:, 0] + torch.pi) % (2*torch.pi) - torch.pi)

def reward_feet_yaw_mean(
    env: ManagerBasedRLEnvCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    feet_quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids]
    feet_quat = rearrange(feet_quat, 'b n c -> (b n) c')
    feet_yaw = math_utils.euler_xyz_from_quat(feet_quat)[-1]
    feet_yaw = rearrange(feet_yaw, '(b n)-> b n', n = len(asset_cfg.body_ids))
    feet_yaw_mean = feet_yaw.mean(dim=-1) + torch.pi * (torch.abs(feet_yaw[:, 1] - feet_yaw[:, 0]) > torch.pi)
    base_quat = asset.data.root_link_quat_w
    base_yaw = math_utils.euler_xyz_from_quat(base_quat)[-1]
    return torch.square((base_yaw - feet_yaw_mean + torch.pi) % (2 * torch.pi) - torch.pi)

def reward_feet_roll(
    env: ManagerBasedRLEnvCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    feet_quat = asset.data.body_pose_w[:, asset_cfg.body_ids, 3:]
    feet_quat = rearrange(feet_quat, 'b n c -> (b n) c')
    feet_roll = math_utils.euler_xyz_from_quat(feet_quat)[0]
    feet_roll = rearrange(feet_roll, '(b n)-> b n', n = len(asset_cfg.body_ids))
    return torch.sum(torch.square(feet_roll), dim=-1)

def reward_feet_distance( # feet are two points
    env: ManagerBasedRLEnvCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), feet_distance_ref: float = 0.2
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    feet_xyz = asset.data.body_pose_w[:, asset_cfg.body_ids, :3] # assume body_ids has two entries for the feet
    base_quat = asset.data.root_pose_w[:, 3:]
    base_yaw = math_utils.euler_xyz_from_quat(base_quat)[2]
    feet_distance = torch.abs(
        torch.cos(base_yaw) * (feet_xyz[:, 1, 1] - feet_xyz[:, 0, 1])
        - torch.sin(base_yaw) * (feet_xyz[:, 1, 0] - feet_xyz[:, 0, 0])
    )
    return torch.clip(feet_distance_ref - feet_distance, min=0.0, max=0.1)