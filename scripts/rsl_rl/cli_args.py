# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
import json
import os
import random
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg


def add_rsl_rl_args(parser: argparse.ArgumentParser):
    """Add RSL-RL arguments to the parser.

    Args:
        parser: The parser to add the arguments to.
    """
    # create a new argument group
    arg_group = parser.add_argument_group(
        "rsl_rl", description="Arguments for RSL-RL agent."
    )
    # -- experiment arguments
    arg_group.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Name of the experiment folder where logs will be stored.",
    )
    arg_group.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Run name suffix to the log directory.",
    )
    # -- load arguments
    arg_group.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Whether to resume from a checkpoint.",
    )
    arg_group.add_argument(
        "--load_run",
        type=str,
        default=None,
        help="Name of the run folder to resume from.",
    )
    arg_group.add_argument(
        "--checkpoint", type=str, default=None, help="Checkpoint file to resume from."
    )
    # -- logger arguments
    arg_group.add_argument(
        "--logger",
        type=str,
        default=None,
        choices={"wandb", "tensorboard", "neptune"},
        help="Logger module to use.",
    )
    arg_group.add_argument(
        "--log_project_name",
        type=str,
        default=None,
        help="Name of the logging project when using wandb or neptune.",
    )


def parse_rsl_rl_cfg(
    task_name: str, args_cli: argparse.Namespace
) -> RslRlOnPolicyRunnerCfg:
    """Parse configuration for RSL-RL agent based on inputs.

    Args:
        task_name: The name of the environment.
        args_cli: The command line arguments.

    Returns:
        The parsed configuration for RSL-RL agent based on inputs.
    """
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    # load the default configuration
    rslrl_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(
        task_name, "rsl_rl_cfg_entry_point"
    )
    rslrl_cfg = update_rsl_rl_cfg(rslrl_cfg, args_cli)
    return rslrl_cfg


def update_rsl_rl_cfg(agent_cfg: RslRlOnPolicyRunnerCfg, args_cli: argparse.Namespace):
    """Update configuration for RSL-RL agent based on inputs.

    Args:
        agent_cfg: The configuration for RSL-RL agent.
        args_cli: The command line arguments.

    Returns:
        The updated configuration for RSL-RL agent based on inputs.
    """
    # override the default configuration with CLI arguments
    if hasattr(args_cli, "seed") and args_cli.seed is not None:
        # randomly sample a seed if seed = -1
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)
        agent_cfg.seed = args_cli.seed
    if args_cli.resume is not None:
        agent_cfg.resume = args_cli.resume
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    # set the project name for wandb and neptune
    if agent_cfg.logger in {"wandb", "neptune"} and args_cli.log_project_name:
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name

    return agent_cfg


def add_json_override_args(parser: argparse.ArgumentParser):
    """Add JSON override argument for Hydra."""
    parser.add_argument(
        "--json_overrides",
        type=str,
        default=None,
        help="Path to a JSON file of Hydra-style overrides.",
    )


def _dict_to_overrides(d: dict[str, Any], prefix: str = "") -> list[str]:
    """Recursively flatten a nested dict into Hydra override strings."""
    out = []
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.extend(_dict_to_overrides(v, key))
        else:
            if isinstance(v, (str, list, dict)):
                rhs = json.dumps(v)
            elif isinstance(v, bool):
                rhs = "true" if v else "false"
            else:
                rhs = str(v)
            out.append(f"{key}={rhs}")
    return out


def load_json_overrides(args_cli: argparse.Namespace) -> list[str]:
    """Load overrides from a JSON file if provided.

    Args:
        args_cli: Parsed CLI args.

    Returns:
        A list of Hydra-style override strings (empty if none).
    """
    if not getattr(args_cli, "json_overrides", None):
        return []

    json_path = args_cli.json_overrides
    if not os.path.exists(json_path):
        print(f"[WARN] JSON overrides file not found: {json_path}")
        return []

    with open(json_path, "r") as f:
        data = json.load(f)
    overrides = _dict_to_overrides(data)
    print(f"[INFO] Loaded {len(overrides)} overrides from {json_path}")
    return overrides