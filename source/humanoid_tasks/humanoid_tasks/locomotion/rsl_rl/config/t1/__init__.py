import gymnasium as gym

from . import agents

gym.register(
    id="T1-Baseline-v1",
    entry_point=f"{__name__}.recreate_booster_cfg:T1ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.recreate_booster_cfg:T1BaselineCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T1BaselinePPORunnerCfg",
    },
)
gym.register(
    id="T1-Baseline-Play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.recreate_booster_cfg:T1Baseline_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T1BaselinePPORunnerCfg",
    },
)

gym.register(
    id="T1-Baseline-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rsl_rl_cfg:T1BaselineCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T1BaselinePPORunnerCfg",
    },
)
gym.register(
    id="T1-Baseline-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rsl_rl_cfg:T1Baseline_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:T1BaselinePPORunnerCfg",
    },
)
