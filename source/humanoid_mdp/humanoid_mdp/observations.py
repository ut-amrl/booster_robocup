"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv, ManagerBasedRLEnvCfg
    from isaaclab.sensors import ContactSensor
import isaaclab.utils.math as math_utils

"""
Sensors.
"""


def height_scan_3d(
    env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.5
) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame.

    Returns HxWx3 tensor: hit positions relative to the sensor position.
    The provided offset (Defaults to 0.5) is subtracted from the z values.
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]

    return sensor.data.pos_w.unsqueeze(1) - sensor.data.ray_hits_w - offset


class contact_friction(ManagerTermBase):
    def __init__(self, env: ManagerBasedRLEnvCfg, cfg: ObservationTermCfg) -> None:
        super().__init__(cfg, env)
        asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        assert cfg.params["asset_cfg"].body_ids != slice(
            None
        ), "Error: must provide body_ids to the 'contact_friction' ObservationTermCfg"
        num_shapes_per_body = []
        for link_path in asset.root_physx_view.link_paths[0]:
            link_physx_view = asset._physics_sim_view.create_rigid_body_view(link_path)  # type: ignore
            num_shapes_per_body.append(link_physx_view.max_shapes)

        materials = asset.root_physx_view.get_material_properties().to(env.device)
        friction = torch.zeros(env.num_envs, 2, device=env.device)
        for body_id in cfg.params["asset_cfg"].body_ids:
            # start index of shape
            start_idx = sum(num_shapes_per_body[:body_id])
            # end index of shape
            end_idx = start_idx + num_shapes_per_body[body_id]
            friction += materials[:, start_idx:end_idx, :2].mean(dim=1)
        self.friction = friction / len(cfg.params["asset_cfg"].body_ids)

    def __call__(
        self, env: ManagerBasedRLEnvCfg, asset_cfg: SceneEntityCfg
    ) -> torch.Tensor:
        return self.friction.clone()


def body_mass(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """access the ground truth mass of the body"""
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    masses = asset.root_physx_view.get_masses()[:, asset_cfg.body_ids].to(asset.device)
    default_masses = asset.data.default_mass[:, asset_cfg.body_ids].to(asset.device)
    return (masses - default_masses) / default_masses


def base_com(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """access the ground truth com of the body"""
    # https://github.com/isaac-sim/IsaacLab/issues/3211
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    com_pos_w = asset.data.root_com_pos_w
    root_pos_w = asset.data.root_link_pos_w
    com_pos_b = math_utils.quat_apply_inverse(
        asset.data.root_link_quat_w, com_pos_w - root_pos_w
    )
    return com_pos_b


def contact_sensor(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame.

    The provided offset (Defaults to 0.5) is subtracted from the returned values.
    """
    # extract the used quantities (to enable type-hinting)
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    return sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].flatten(start_dim=1)
