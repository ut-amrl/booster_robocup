import gymnasium as gym

from . import agents

gym.register(
    id="Alexander-Baseline-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rsl_rl_cfg:AlexanderBaselineCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AlexanderPPORunnerCfg",
    },
)
gym.register(
    id="Alexander-Baseline-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rsl_rl_cfg:AlexanderBaseline_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AlexanderPPORunnerCfg",
    },
)
