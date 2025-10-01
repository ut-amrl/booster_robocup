import humanoid_mdp
import isaaclab.terrains as terrain_gen

from .rsl_rl_cfg import *

@configclass
class MetricsCfg:
    """Expose metrics through reward terms for benchmarking."""
    pass

class T1Baseline_BENCHMARK(T1BaselineCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # reset terrain
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # set movement command to 0
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        # disable randomization for play
        self.observations.policy.enable_corruption = False

        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        
    def customize_env(self, test_cfg) -> None:
        # set movement command
        speed = test_cfg["speed"]
        self.commands.base_velocity.ranges.lin_vel_x = (speed, speed)

        # set terrain
        if test_cfg["uneven"] == True:
            self.scene.terrain.terrain_type = "generator"
            self.scene.terrain.terrain_generator = terrain_gen.TerrainGeneratorCfg(
                size=(100.0, 100.0),
                border_width=0.0,
                num_rows=1,
                num_cols=1,
                horizontal_scale=0.1,
                vertical_scale=0.005,
                slope_threshold=0.75,
                difficulty_range=(0.0, 1.0),
                use_cache=False,
                sub_terrains={
                    "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                        proportion=0.2, noise_range=(0.02, 0.05), noise_step=0.02
                    ),
                },
            )

        # set pushing event
        if test_cfg["push"] == True:
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