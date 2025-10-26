from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import RecorderTerm, RecorderTermCfg
from isaaclab.envs import ManagerBasedRLEnv

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

class SweepPerformanceRecorder(RecorderTerm): 
    def record_post_reset(self, env_ids: Sequence[int] | None) -> tuple[str | None, torch.Tensor | dict | None]:
        gait_cycle_return = self._env.extras["log"]["Episode_Reward/feet_swing"]
        survival_return = self._env.extras["log"]["Episode_Reward/survival"]

        self._env.extras["log"]["Sweep/performance"] = gait_cycle_return + survival_return

        return None, None