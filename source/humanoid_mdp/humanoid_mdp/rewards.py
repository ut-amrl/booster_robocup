from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg, ManagerTermBase, RewardTermCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms
from isaaclab.sensors import ContactSensor, RayCaster
import isaaclab.utils.math as math_utils

from itertools import combinations

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv

def base_linear_velocity_reward(
    env: ManagerBasedRLEnvCfg,
    asset_cfg: SceneEntityCfg,
    std: float,
    ramp_at_vel: float = 1.0,
    ramp_rate: float = 0.5,
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using abs exponential kernel"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the error
    command = env.command_manager.get_command("base_velocity")[:, :2]
    lin_vel_error = torch.linalg.norm((command - asset.data.root_lin_vel_b[:, :2]), dim=1)
    # fixed 1.0 multiple for tracking below the ramp_at_vel value, then scale by the rate above
    vel_cmd_magnitude = torch.linalg.norm(command, dim=1)
    velocity_scaling_multiple = torch.clamp(1.0 + ramp_rate * (vel_cmd_magnitude - ramp_at_vel), min=1.0)
    return torch.exp(-lin_vel_error / std) * velocity_scaling_multiple

def base_angular_velocity_reward(env: ManagerBasedRLEnvCfg, asset_cfg: SceneEntityCfg, std: float) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using abs exponential kernel"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the error
    command = env.command_manager.get_command("base_velocity")[:, 2]
    ang_vel_error = torch.linalg.norm((command - asset.data.root_ang_vel_b[:, 2]).unsqueeze(1), dim=1)
    return torch.exp(-ang_vel_error / std)

def air_time_reward_cmd_biped(
        env: ManagerBasedRLEnvCfg,
        sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward longer feet air and contact time"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # get air and stance times
    current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    current_contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    command = torch.norm(env.command_manager.get_command("base_velocity"), dim=1).unsqueeze(dim=1).expand(-1, 2)
    freq_cmd = env.command_manager.get_command("frequency")
    mode_time = 1/freq_cmd
    t_max = torch.max(current_air_time, current_contact_time)
    t_min = torch.clip(t_max, max=mode_time)
    # when in stance: reward for contact time, penalty for air time
    stance_command_reward = torch.clip(current_contact_time - current_air_time, -mode_time, mode_time)
    # when in motion: reward until mode_time, zero reward after that
    reward = torch.where(command > 0.0, torch.where(t_max < mode_time, t_min, 0), stance_command_reward)
    return torch.sum(reward, dim=1)

def foot_clearance_reward(
    env: ManagerBasedRLEnvCfg,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
    tanh_multiplier: float,
) -> torch.Tensor:
    """Reward tracking of foot clearance using square exponential kernel"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the error
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    # scale reward from zero (stance command) to one with rate determined by tanh scaled by the multiplier
    foot_velocity_tanh = torch.tanh(tanh_multiplier * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)

def gait_reward_biped(
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        std: float,
        max_error: float
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        air_time = contact_sensor.data.current_air_time
        contact_time = contact_sensor.data.current_contact_time

        error = torch.clip(
            torch.square(air_time[:, sensor_cfg.body_ids[0]] - contact_time[:, sensor_cfg.body_ids[1]]),
            max=max_error**2) + torch.clip(
            torch.square(contact_time[:, sensor_cfg.body_ids[0]] - air_time[:, sensor_cfg.body_ids[1]]),
            max=max_error**2)

        command = torch.norm(env.command_manager.get_command("base_velocity"), dim=1)
        return torch.where(command > 0.0, torch.exp(-error / std), 1.0)

def target_base_height(
        env: ManagerBasedRLEnvCfg,
        target_height: float = 0.665,
        std: float = 0.25,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Track desired base z height."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    height_err = asset.data.root_com_pos_w[:, 2] - target_height
    return torch.exp(-height_err.pow(2) / std)

############### Penalty Terms ###############

def angular_motion_penalty(
        env: ManagerBasedRLEnvCfg,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize base roll/pitch velocity"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(asset.data.root_ang_vel_b[:, :2]), dim=1)

def base_orientation_penalty(
        env: ManagerBasedRLEnvCfg,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize non-flat base orientation by penalizing the xy-components of the projected gravity vector"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.projected_gravity_b[:, :2]), dim=1)

def foot_slip_penalty(
    env: ManagerBasedRLEnvCfg,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    threshold: float
) -> torch.Tensor:
    """Penalize foot planar (xy) slip when in contact with the ground"""
    asset: Articulation = env.scene[asset_cfg.name]
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    foot_planar_velocity = torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    # penalty is the norm of the planar velocity when in contact
    reward = is_contact * foot_planar_velocity
    return torch.sum(reward, dim=1)

def action_smoothness_penalty(env: ManagerBasedRLEnvCfg, clamp: float = 1e6) -> torch.Tensor:
    """Penalize large instantaneous changes in the network action output"""
    # print(env.action_manager.action)
    # print(f'action names {env.action_manager.get_term("joint_pos")._joint_names}') # to print joint order
    # print()
    norm = torch.linalg.norm((env.action_manager.action - env.action_manager.prev_action), dim=1)
    return norm.clamp(min=-clamp, max=clamp)

def air_time_variance_penalty(env: ManagerBasedRLEnvCfg, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )

def foot_impact_penalty(env: ManagerBasedRLEnvCfg, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, cutoff: float) -> torch.Tensor:
    """Penalize foot impact when coming into contact with the ground"""
    asset: Articulation = env.scene[asset_cfg.name]
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # get contact state
    is_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    # get velocity at contact
    foot_down_velocity = torch.clamp(asset.data.body_lin_vel_w[:, asset_cfg.body_ids], min=-abs(cutoff), max=0.0).norm(dim=-1)
    # penalty is the velocity at contact squared when in contact
    reward = is_contact * torch.square(foot_down_velocity)
    return torch.sum(reward, dim=1)

def joint_position_penalty(
        env: ManagerBasedRLEnvCfg,
        asset_cfg: SceneEntityCfg,
        stand_still_scale: float,
        velocity_threshold: float = 0.5
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    command = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm((
            asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
        ).view(env.scene.num_envs, len(asset_cfg.joint_ids)), dim=1)
    # when the robot is in stance, scale the reward (force stance in the default pose)
    return torch.where(torch.logical_or(command > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)

def joint_velocity_penalty(env: ManagerBasedRLEnvCfg, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint velocities on the articulation"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, asset_cfg.joint_ids].norm(dim=1)

def joint_acceleration_penalty(env: ManagerBasedRLEnvCfg, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint accelerations on the articulation"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_acc[:, asset_cfg.joint_ids].norm(dim=1)

def joint_torques_penalty(env: ManagerBasedRLEnvCfg, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint torques on the articulation"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.applied_torque[:, asset_cfg.joint_ids].norm(dim=1)

def foot_orientation(
        env: ManagerBasedRLEnvCfg,
        asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize deviation of foot yaw from root yaw."""
    asset: Articulation = env.scene[asset_cfg.name]
    foot_quats = asset.data.body_quat_w[:, asset_cfg.body_ids]
    root_quat = asset.data.root_quat_w.view(env.scene.num_envs, 1, 4)

    # Extract yaw-only quaternions
    root_yaw_quat = math_utils.yaw_quat(root_quat)
    foot_yaw_quats = math_utils.yaw_quat(foot_quats)

    # Compute squared quaternion difference using dot product (cosine distance)
    quat_diff = 1 - (root_yaw_quat * foot_yaw_quats).sum(dim=-1).abs()

    # Penalize deviation from root yaw
    return quat_diff.pow(2).sum(dim=1)

def foot_lateral_penalty(env: ManagerBasedRLEnvCfg, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize feet being too far laterally from the root body position."""
    asset: Articulation = env.scene[asset_cfg.name]
    # world positions
    root_pos = asset.data.root_pos_w      # (num_envs, 3)
    root_quat = asset.data.root_quat_w    # (num_envs, 4)
    foot_pos = asset.data.body_pos_w[:, asset_cfg.body_ids]  # (num_envs, num_feet, 3)

    # import pdb; pdb.set_trace()
    # shift feet positions relative to root
    disp_w = foot_pos - root_pos.unsqueeze(1)  # (num_envs, num_feet, 3)

    # rotate displacement into root frame
    # root_quat_exp = root_quat.expand(-1, disp_w.shape[1], -1)
    disp_root = torch.stack([math_utils.quat_apply_inverse(root_quat, disp_w[:, i, :]) for i in range(disp_w.shape[1])], dim=1)  # (num_envs, num_feet, 3)


    # lateral offset = y-coordinate in root frame
    lateral_offset = disp_root[..., 1]  # (num_envs, num_feet)

    # penalty: squared lateral distance
    min_dist = 0.35
    penalty = torch.where(
        lateral_offset.abs() < min_dist,
        torch.zeros_like(lateral_offset),
        lateral_offset ** 2
    ).sum(dim=1)  # (num_envs,)

    # Sum across feet
    return penalty

def feet_yaw_diff_penalty(env: ManagerBasedRLEnvCfg, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize misalignment in feet yaw."""
    # extract quantifies
    asset: Articulation = env.scene[asset_cfg.name]
    foot_quats = asset.data.body_quat_w[:, asset_cfg.body_ids]
    # root_quat = asset.data.root_quat_w.view(env.scene.num_envs, 1, 4)

    # Extract yaw-only quaternions
    # root_yaw_quat = math_utils.yaw_quat(root_quat)
    foot_yaw_quats = math_utils.yaw_quat(foot_quats)
    left_foot_quats = foot_yaw_quats[:,0,:]
    right_foot_quats = foot_yaw_quats[:,1,:]

    # Compute squared quaternion difference using dot product (cosine distance)
    quat_diff = 1 - (right_foot_quats * left_foot_quats).sum(dim=-1).abs()
    
    # Penalize misalignment in feet
    return quat_diff.pow(2) #.sum(dim=1)


def foot_distance(
        env: ManagerBasedRLEnvCfg,
        asset_cfg: SceneEntityCfg,
        min_dist: float = 0.30,
) -> torch.Tensor:
    """Penalize feet too close together."""
    asset: Articulation = env.scene[asset_cfg.name]
    foot_pos = asset.data.body_pos_w[:, asset_cfg.body_ids]
    foot_dist = (foot_pos[:, 0] - foot_pos[:, 1]).norm(dim=-1)
    return torch.where(foot_dist < min_dist, (min_dist-foot_dist).pow(2), 0)

def heel_toe_stepping_penalty(
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        threshold: float
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # get contact state
    # is_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    is_contact = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids].norm(dim=2) > threshold
    # get foot angle
    foot_angle_fwd = asset.data.joint_pos[:, asset_cfg.joint_ids] > 0
    foot_angle_bwd = asset.data.joint_pos[:, asset_cfg.joint_ids] < 0
    foot_angle = torch.where(
        asset.data.root_lin_vel_b > 0,
        (is_contact & foot_angle_fwd).sum(dim=1),
        (is_contact & foot_angle_bwd).sum(dim=1)
    )
    return foot_angle
