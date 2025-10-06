from __future__ import annotations

from isaaclab.managers import SceneEntityCfg
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from dataclasses import dataclass, field
from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.sensors import ContactSensor
from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv




class FrequencyCommand(CommandTerm):
    r"""Command generator that generates a frequency command from uniform distribution.

    Mathematically, the footstep period is computed as follows from the frequency command:

    .. math::

        \T = \frac{1}{f}

    """

    cfg: FrequencyCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: FrequencyCommandCfg, env: ManagerBasedEnv) -> None:
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.

        """
        # initialize the base class
        super().__init__(cfg, env)

        # obtain the robot asset
        # -- sensor
        self.sensor: ContactSensor = env.scene.sensors[cfg.sensor_cfg.name]

        # crete buffers to store the command
        # -- command: x vel, y vel, yaw vel, heading
        self.freq = torch.zeros(self.num_envs, 1, device=self.device)
        # -- metrics
        self.metrics["error_step_length"] = torch.zeros(
            self.num_envs, device=self.device
        )

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "FrequencyCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        return self.freq

    """
    Implementation specific functions.
    """

    def _update_metrics(self) -> None:
        first_contact = self.sensor.compute_first_contact(self._env.step_dt)[
            :, self.cfg.sensor_cfg.body_ids
        ]
        last_air_time = self.sensor.data.last_air_time[:, self.cfg.sensor_cfg.body_ids]
        error = torch.mean((last_air_time - 1 / self.freq).abs() * first_contact, dim=1)
        error *= (
            torch.norm(
                self._env.command_manager.get_command("base_velocity")[:, :2], dim=1
            )
            > 0.1
        )
        self.metrics["error_step_length"] += error

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        # sample frequency commands
        r = torch.empty(len(env_ids), device=self.device)
        # -- freq
        self.freq[env_ids, 0] = r.uniform_(*self.cfg.range)

    def _update_command(self) -> None:
        """Post-processes the frequency command."""
        pass


@configclass
class FrequencyCommandCfg(CommandTermCfg):
    class_type: type = FrequencyCommand

    sensor_cfg: SceneEntityCfg = MISSING
    """Name of the sensor in the environment for which the commands are generated."""

    range: tuple[float, float] = MISSING
    """Distribution ranges for the frequency command."""


# ---------- Shared 2D probability grid ----------
class CurriculumGrid:
    def __init__(self, lin_levels: int, ang_levels: int, device: torch.device):
        self.lin_levels = lin_levels
        self.ang_levels = ang_levels
        H = 1 + 2 * lin_levels
        W = 1 + 2 * ang_levels
        self.P = torch.zeros(H, W, device=device)
        # start with all mass at the center (easy)
        self.P[lin_levels, ang_levels] = 1.0

    def sample_levels(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Multinomial sample n cells, return signed levels (ℓ, a)."""
        probs = self.P.flatten().clone()
        if torch.all(probs <= 0):
            probs.fill_(1.0)
        idx = torch.multinomial(probs, num_samples=n, replacement=True)
        H, W = self.P.shape
        j = idx // W  # row (ang)
        i = idx % W   # col (lin)
        ell = i - self.lin_levels
        ang = j - self.ang_levels
        return ell, ang

    def promote(self, ell: torch.Tensor, ang: torch.Tensor, update_rate: float):
        """Add +update_rate to succeeded cells and their 4-neighbors, clamp to 1.0."""
        H, W = self.P.shape
        i = ell + self.lin_levels   # (n,)
        j = ang + self.ang_levels   # (n,)
        di = torch.tensor([0, -1, 1, 0, 0], device=self.P.device)
        dj = torch.tensor([0, 0, 0, -1, 1], device=self.P.device)
        ii = (i[:, None] + di[None, :]).reshape(-1)
        jj = (j[:, None] + dj[None, :]).reshape(-1)
        mask = (ii >= 0) & (ii < W) & (jj >= 0) & (jj < H)
        ii, jj = ii[mask], jj[mask]
        self.P[jj, ii] = (self.P[jj, ii] + update_rate).clamp_(max=1.0)


# ---------- Config + Command term ----------

class CurriculumVelocityCommand(UniformVelocityCommand):
    """
    Emits per-env [vx, vy, yaw] commands and maintains a shared curriculum grid.
    Integrates with Observation term via mdp.generated_commands(command_name=...).
    """
        
    cfg: CurriculumVelocityCommandCfg

    def __init__(self, cfg: CurriculumVelocityCommandCfg, env):
        super().__init__(cfg, env)  # keep parent's command tensor & debug-vis
        # curriculum state
        self.grid = CurriculumGrid(cfg.lin_vel_levels, cfg.ang_vel_levels, self.device)
        self.levels = torch.zeros(self.num_envs, 2, dtype=torch.long, device=self.device)  # (ℓ, a)

    # def _resample_command(self, env_ids):
    #     if env_ids is None or len(env_ids) == 0:
    #         return

    #     # 1) pick grid levels
    #     ell, ang = self.grid.sample_levels(len(env_ids))
    #     self.levels[env_ids, 0] = ell
    #     self.levels[env_ids, 1] = ang

    #     # 2) map to continuous with half-level jitter
    #     ex = torch.empty(len(env_ids), device=self.device).uniform_(-0.5, 0.5)
    #     ez = torch.empty(len(env_ids), device=self.device).uniform_(-0.5, 0.5)

    #     # IMPORTANT: mutate the parent's command tensor IN PLACE (debug-vis reads this)
    #     cmd = self.command  # (num_envs, 3) from the parent class

    #     # vx
    #     cmd[env_ids, 0] = (ell + ex) * self.cfg.lin_vel_x_resolution
    #     # vy scales with |ℓ|
    #     eta = torch.empty(len(env_ids), device=self.device).uniform_(-1.0, 1.0)
    #     cmd[env_ids, 1] = torch.abs(ell).float() * eta * self.cfg.lin_vel_y_resolution
    #     # yaw
    #     cmd[env_ids, 2] = (ang + ez) * self.cfg.ang_vel_resolution

    #     # still subset
    #     k = int(self.cfg.still_proportion * len(env_ids))
    #     if k > 0:
    #         perm = env_ids[torch.randperm(len(env_ids), device=self.device)[:k]]
    #         cmd[perm, :] = 0.0

    def _resample_command(self, env_ids):
        if env_ids is None or len(env_ids) == 0:
            return

        # 1) pick grid levels
        ell, ang = self.grid.sample_levels(len(env_ids))
        self.levels[env_ids, 0] = ell
        self.levels[env_ids, 1] = ang

        # 2) map to continuous with half-level jitter
        ex = torch.empty(len(env_ids), device=self.device).uniform_(-0.5, 0.5)
        ez = torch.empty(len(env_ids), device=self.device).uniform_(-0.5, 0.5)
        eta = torch.empty(len(env_ids), device=self.device).uniform_(-1.0, 1.0)

        # write into the parent's tensor (this is what debug-vis reads)
        cmd = self.command  # (num_envs, 3)

        # vx, vy, yaw
        cmd[env_ids, 0] = (ell + ex) * self.cfg.lin_vel_x_resolution
        cmd[env_ids, 1] = torch.abs(ell).float() * eta * self.cfg.lin_vel_y_resolution
        cmd[env_ids, 2] = (ang + ez) * self.cfg.ang_vel_resolution

        # still subset
        k = int(self.cfg.still_proportion * len(env_ids))
        if k > 0:
            perm = env_ids[torch.randperm(len(env_ids), device=self.device)[:k]]
            cmd[perm, :] = 0.0

        # optional: clamp to parent ranges so arrows and limits match viewer
        rx = self.cfg.ranges.lin_vel_x
        ry = self.cfg.ranges.lin_vel_y
        rz = self.cfg.ranges.ang_vel_z
        cmd[env_ids, 0].clamp_(rx[0], rx[1])
        cmd[env_ids, 1].clamp_(ry[0], ry[1])
        cmd[env_ids, 2].clamp_(rz[0], rz[1])


    def _update_metrics(self) -> None:
        """Publish any extras/metrics for logging."""
        self._env.extras["curriculum/mean_lin_level"] = torch.mean(torch.abs(self.levels[:, 0].float()))
        self._env.extras["curriculum/mean_ang_level"] = torch.mean(torch.abs(self.levels[:, 1].float()))


    # def _resample(self, env_ids: torch.Tensor):
    #     if len(env_ids) == 0:
    #         return

    #     now_step = self._env.episode_length_buf[env_ids]
    #     if self.cfg.curriculum:
    #         # 1) draw difficulty levels from grid
    #         ell, ang = self.grid.sample_levels(len(env_ids))
    #         self.levels[env_ids, 0] = ell
    #         self.levels[env_ids, 1] = ang

    #         # 2) map to continuous commands with half-level jitter
    #         ex = torch.empty(len(env_ids), device=self.device).uniform_(-0.5, 0.5)
    #         ez = torch.empty(len(env_ids), device=self.device).uniform_(-0.5, 0.5)
    #         # vx
    #         self.commands[env_ids, 0] = (ell + ex) * self.cfg.lin_vel_x_resolution
    #         # vy scales with |ℓ|
    #         eta = torch.empty(len(env_ids), device=self.device).uniform_(-1.0, 1.0)
    #         self.commands[env_ids, 1] = torch.abs(ell).float() * eta * self.cfg.lin_vel_y_resolution
    #         # yaw
    #         self.commands[env_ids, 2] = (ang + ez) * self.cfg.ang_vel_resolution
    #     else:
    #         # uniform fallback (no curriculum)
    #         self.commands[env_ids, 0] = torch.empty(len(env_ids), device=self.device).uniform_(-1.0, 2.0)
    #         self.commands[env_ids, 1] = torch.zeros(len(env_ids), device=self.device)  # your current cfg has 0 lateral
    #         self.commands[env_ids, 2] = torch.empty(len(env_ids), device=self.device).uniform_(-1.0, 1.0)

    #     # still subset
    #     k = int(self.cfg.still_proportion * len(env_ids))
    #     if k > 0:
    #         perm = env_ids[torch.randperm(len(env_ids), device=self.device)[:k]]
    #         self.commands[perm, :] = 0.0

    #     rx = self.cfg.ranges.lin_vel_x
    #     ry = self.cfg.ranges.lin_vel_y
    #     rz = self.cfg.ranges.ang_vel_z
    #     self.commands[env_ids, 0].clamp_(rx[0], rx[1])
    #     self.commands[env_ids, 1].clamp_(ry[0], ry[1])
    #     self.commands[env_ids, 2].clamp_(rz[0], rz[1])

    #     # schedule next dwell
    #     self._set_next_resample(env_ids, now_step)

@dataclass
class CurriculumVelocityCommandCfg(UniformVelocityCommandCfg):
    """Velocity-only curriculum commands publishing [vx, vy, yaw]."""
    class_type: type = CurriculumVelocityCommand

    # curriculum on/off
    curriculum: bool = True
    # fraction of resamples that are forced to still/zero commands
    still_proportion: float = 0.10
    # dwell window in seconds before resampling
    resampling_time_range: tuple[float, float] = (8.0, 12.0)

    # grid and mapping
    lin_vel_levels: int = 10
    ang_vel_levels: int = 10
    lin_vel_x_resolution: float = 0.2  # m/s per level
    lin_vel_y_resolution: float = 0.1  # m/s per |level|
    ang_vel_resolution: float = 0.2    # rad/s per level
    update_rate: float = 0.1           # promotion amount

    # success tolerances (used by the Event promotion function)
    lin_vel_x_toler: float = 0.4
    lin_vel_y_toler: float = 0.2
    ang_vel_yaw_toler: float = 0.2
    episode_length_toler: float = 0.1

    debug_vis: bool = True