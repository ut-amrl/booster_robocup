import sys
from typing import Callable, Dict, Sequence

import torch
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import (
    CommandTermCfg,
    EventTermCfg,
    ManagerTermBase,
    SceneEntityCfg,
)

import isaaclab.envs.mdp as mdp


def split_command_cfg(
    cfg: CommandTermCfg, subterm_cfgs: Sequence[CommandTermCfg]
) -> CommandTermCfg:
    """Utility function to create a split command config term.

    This function creates a SplitCommand subclass that splits the command generation
    across multiple subterms, each responsible for a subset of the environments. All
    subterms must be of the same type.

    Subclassing is used here to allow other parts of CommandTerm (such as debug vis) to
    work as intended. However, this requires us to register the subclass to avoid issues
    with hydra serialization.
    """

    class SplitCommand(cfg.class_type):
        __name__ = f"SplitCommand_{cfg.class_type.__name__}"

        def __init__(self, cfg: CommandTermCfg, env: ManagerBasedRLEnv) -> None:
            super().__init__(cfg, env)
            self.subterms = [cfg.class_type(subterms, env) for subterms in subterm_cfgs]
            self.num_subtasks = len(self.subterms)
            self.n = env.num_envs // self.num_subtasks

            self.cmd = torch.zeros(super().command.shape, device=self.device)

        @property
        def command(self) -> torch.Tensor:
            for i in range(self.num_subtasks):
                self.cmd[i * self.n : (i + 1) * self.n] = self.subterms[i].command[
                    i * self.n : (i + 1) * self.n
                ]
            return self.cmd

        def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
            metrics = super().reset(env_ids)
            for subterm in self.subterms:
                subterm.reset(env_ids)

            return metrics

        def compute(self, dt: float):
            super().compute(dt)
            for subterm in self.subterms:
                subterm.compute(dt)

    # register the subclass to the module for hydra
    this_mod = sys.modules[__name__]
    setattr(this_mod, SplitCommand.__name__, SplitCommand)

    cfg.class_type = SplitCommand
    return cfg


def filtered_func(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    subtask_idx: str,
    func: Callable,
    params: Dict,
):
    """Wrapper to filter env ids by subtask index before calling a function.

    This function grabs a subset of the env_ids in the range from subtask_id * n
    to subtask_id * (n + 1) and calls the event function on the filtered env_ids.
    """
    n = env.num_envs // env.cfg.num_subtasks
    sub_env_ids = env_ids[
        (subtask_idx * n <= env_ids) & (env_ids < (subtask_idx + 1) * n)
    ]
    if len(sub_env_ids) > 0:
        func(env, sub_env_ids, **params)


class reset_root_state_uniform_once(ManagerTermBase):
    """Reset term that moves the agent underground after its first death for visual purposes

    Internally calls reset_root_state_uniform.
    """
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        pose_range: dict[str, tuple[float, float]],
        velocity_range: dict[str, tuple[float, float]],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ):
        mdp.reset_root_state_uniform(
            env,
            env_ids[self.alive[env_ids]],
            pose_range,
            velocity_range,
            asset_cfg,
        )

        mdp.reset_root_state_uniform(
            env,
            env_ids[~self.alive[env_ids]],
            {"z": (-2.0, -2.0)},
            {},
            asset_cfg,
        )

        self.alive[env_ids] = False


def subterrain_out_of_bounds(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    distance_buffer: float = 3.0,
) -> torch.Tensor:
    """Terminate when the actor moves too close to the edge of the subterrain.

    If the actor moves too close to the edge of the subterrain, the termination is
    activated. The env origin is assumed to be the equal to the terrain origin.
    """
    if env.scene.cfg.terrain.terrain_type == "plane":
        return torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )  # we have infinite terrain because it is a plane
    elif env.scene.cfg.terrain.terrain_type == "generator":
        # obtain the size of the sub-terrains
        terrain_gen_cfg = env.scene.terrain.cfg.terrain_generator
        grid_width, grid_length = terrain_gen_cfg.size

        # extract the used quantities (to enable type-hinting)
        asset: RigidObject = env.scene[asset_cfg.name]

        # relative root position to the environment origin
        rel_root_pos = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]

        # check if the agent is out of bounds
        x_out_of_bounds = (
            torch.abs(rel_root_pos[:, 0]) > grid_width / 2 - distance_buffer
        )
        y_out_of_bounds = (
            torch.abs(rel_root_pos[:, 1]) > grid_length / 2 - distance_buffer
        )
        return torch.logical_or(x_out_of_bounds, y_out_of_bounds)
    else:
        raise ValueError(
            "Received unsupported terrain type, must be either 'plane' or 'generator'."
        )
