import isaaclab.terrains as terrain_gen
import humanoid_mdp
from rsl_rl_cfg import *

class T1Baseline_BENCHMARK(T1BaselineCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # disable randomization for play
        self.observations.policy.enable_corruption = False

        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None

class T1Baseline_BENCHMARK_Walk(T1Baseline_BENCHMARK):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)

class T1Baseline_BENCHMARK_Run(T1Baseline_BENCHMARK):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = (3.0, 3.0)

class T1Baseline_BENCHMARK_Rough(T1Baseline_BENCHMARK):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.02, 0.05), noise_step=0.02, border_width=0.25
        ),

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)

class T1Baseline_BENCHMARK_Push(T1Baseline_BENCHMARK):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.events.push_robot = EventTerm(
            func=humanoid_mdp.push_by_adding_velocity,
            mode="interval",
            interval_range_s=(3.0, 5.0),
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
                "velocity_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                },
            },
        )

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)

