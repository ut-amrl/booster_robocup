"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.envs import ManagerBasedEnv


def terrain_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(
        asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
    )
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = (
        distance
        < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    )
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())


def curriculum_success_promotion(
    env: ManagerBasedEnv, env_ids: torch.Tensor, command_name: str = "base_velocity"
):
    """
    On reset, check success (time-on-task + tracking tolerances) and promote the curriculum grid
    for the command term registered under `command_name`.
    """
    if env_ids is None or len(env_ids) == 0:
        return

    # fetch our curriculum command term
    cmd_term = env.command_manager.get_term(command_name)
    if cmd_term is None or not hasattr(cmd_term, "grid"):
        # not using our curriculum command; nothing to do
        return

    # timing thresholds
    sim_dt = float(env.cfg.sim.dt)
    decim = int(getattr(env.cfg, "decimation", 1))
    total_steps = int(env.cfg.episode_length_s / (sim_dt * decim))
    thresh = int(total_steps * (1.0 - cmd_term.cfg.episode_length_toler))

    # per-env last episode lengths
    ep_len = env.episode_length_buf[env_ids]

    # velocities: use live base velocities; if you keep filtered buffers, swap them here
    v = cmd_term.robot.data.root_lin_vel_b[env_ids]
    w = cmd_term.robot.data.root_ang_vel_b[env_ids]

    # commanded targets that were active just before reset
    cmd = cmd_term.command[env_ids]  # (n, 3) [vx, vy, yaw]

    # success tests
    ok_time = ep_len > thresh
    ok_vx = torch.abs(v[:, 0] - cmd[:, 0]) < cmd_term.cfg.lin_vel_x_toler
    ok_vy = torch.abs(v[:, 1] - cmd[:, 1]) < cmd_term.cfg.lin_vel_y_toler
    ok_wz = torch.abs(w[:, 2] - cmd[:, 2]) < cmd_term.cfg.ang_vel_yaw_toler

    success_mask = ok_time & ok_vx & ok_vy & ok_wz
    succ_ids = env_ids[success_mask]

    if len(succ_ids) > 0:
        cmd_term.promote(succ_ids)

    return {
        "mean_lin_level": torch.mean(cmd_term.levels[:, 0].float().abs()),
        "mean_ang_level": torch.mean(cmd_term.levels[:, 1].float().abs()),
        "success_rate": len(succ_ids) / len(env_ids),
    }
