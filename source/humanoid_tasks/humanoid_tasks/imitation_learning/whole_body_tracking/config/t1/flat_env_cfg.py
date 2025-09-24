from isaaclab.utils import configclass

from humanoid_assets import T1_CFG, T1_ACTION_SCALE
from humanoid_tasks.imitation_learning.whole_body_tracking.config.t1.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from humanoid_tasks.imitation_learning.whole_body_tracking.tracking_env_cfg import TrackingEnvCfg


@configclass
class T1FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = T1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = T1_ACTION_SCALE
        self.commands.motion.anchor_body_name = "trunk"
        self.commands.motion.body_names = [
            "waist",
            "ll2",
            "ll4",
            "ll6",
            "lr2",
            "lr4",
            "lr6",
            "trunk",
            "al2",
            "al4",
            "ar2",
            "ar4",
        ]


@configclass
class T1FlatWoStateEstimationEnvCfg(T1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class T1FlatLowFreqEnvCfg(T1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE
