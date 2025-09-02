from dataclasses import MISSING

import copy
import torch
import os

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
)

@configclass
class AttentionModuleCfg:
    """Configuration for the PPO actor-critic networks."""

    class_name: str = "AttentionModule"
    """The policy class name. Default is AttentionModule."""

    height_scan_shape: list[int, int, int] = MISSING

    init_noise_std: float = MISSING
    """The initial noise standard deviation for the policy."""

    actor_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the actor network."""

    critic_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the critic network."""

    activation: str = MISSING
    """The activation function for the actor and critic networks."""
