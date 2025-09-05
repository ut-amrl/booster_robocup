import gymnasium as gym

from . import agents

gym.register(
    id="Attention-T1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.attention_cfg:T1AttentionModuleCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.attention_ppo_cfg:AttentionPPORunnerCfg"
    },
)
gym.register(
    id="Attention-T1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.attention_cfg:T1AttentionModule_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.attention_ppo_cfg:AttentionPPORunnerCfg"
    },
)
