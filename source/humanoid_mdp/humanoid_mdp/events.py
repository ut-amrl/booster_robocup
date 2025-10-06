from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
import isaaclab.envs.mdp as mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def push_by_adding_velocity(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Push the asset by setting the root velocity to a random value within the given ranges.

    This creates an effect similar to pushing the asset with a random impulse that changes the asset's velocity.
    It samples the root velocity from the given ranges and sets the velocity into the physics simulation.

    The function takes a dictionary of velocity ranges for each axis and rotation. The keys of the dictionary
    are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form ``(min, max)``.
    If the dictionary does not contain a key, the velocity is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # velocities
    vel_w = asset.data.root_vel_w[env_ids]
    # sample random velocities
    range_list = [
        velocity_range.get(key, (0.0, 0.0))
        for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=asset.device)
    vel_w[:] += math_utils.sample_uniform(
        ranges[:, 0], ranges[:, 1], vel_w.shape, device=asset.device
    )
    # set the velocities into the physics simulation
    asset.write_root_velocity_to_sim(vel_w, env_ids=env_ids)


def curriculum_success_promotion(env: ManagerBasedEnv, env_ids: torch.Tensor, command_name: str = "base_velocity"):
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
    base_lin = mdp.base_lin_vel(env)  # (num_envs, 3) body-frame
    base_ang = mdp.base_ang_vel(env)  # (num_envs, 3)
    v = base_lin[env_ids]
    w = base_ang[env_ids]

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
        levels = cmd_term.levels[succ_ids]  # (n, 2) ints [ℓ, a]
        cmd_term.grid.promote(levels[:, 0], levels[:, 1], update_rate=cmd_term.cfg.update_rate)
        env.extras["curriculum/success_rate"] = float(len(succ_ids)) / float(len(env_ids))
