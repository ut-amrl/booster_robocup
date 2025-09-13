from __future__ import annotations

from isaaclab.managers import SceneEntityCfg
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.sensors import ContactSensor

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
