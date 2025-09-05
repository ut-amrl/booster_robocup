#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

"""Implementation of different RL agents."""

from .distullation import Distillation
from .ppo import PPO

__all__ = ["PPO", "Distillation"]
