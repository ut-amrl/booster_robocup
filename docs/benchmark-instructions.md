# Benchmark instructions

This document explains how to run benchmark script `scripts/rsl_rl/test.py`, and how to modify the `T1Baseline_BENCHMARK` configuration in
`benchmark_cfg.py` to add new subtasks with custom terrains, events, and commands.

## Running the benchmark/test script

```bash
./isaaclab.sh -p scripts/rsl_rl/test.py \
    --task T1-Baseline-Benchmark-v0 \
    --headless \
    --video \
    --max_length 3000 \
    --wandb \
    --wandb_log
```

Notes on the example above:
- `--max_length` specifies the maximum episode length in steps before truncation.
- `--video` enables recording of environment frames; the length of the video is determined by `--video_length`, defaulting to `--max_length` if not specified
- `--wandb` will take you through the steps to pull the policy from Weights & Biases
- `--wandb_log` will optionally upload the benchmark results to the run where the policy is pulled from when `--wandb` is enabled

Other arguments:
- `--num_envs` (default 32) specifies the number of environments to simulate *per subtask*
- `--seed` (default 42)

## Defining and modifying subtasks

A "subtask" is one column/variant of the benchmark. Each subtask defines the terrain, the command sampling method, and any events/terminations that apply only to that subtask's environments.

To define and modify subtasks in `T1Baseline_BENCHMARK`, edit the `self.subtasks` list in the `setup_subtasks()` method. To define a new subtask, subclass the `Subtask` dataclass. Each subtask is a small dataclass that bundles together:

- a name (used to identify the subtask and to build per-subtask event names),
- a `subterrain` config (a `SubTerrain`-style cfg from `isaaclab.terrains`),
- a command cfg (for example `mdp.UniformVelocityCommandCfg`), and
- an `events` dictionary mapping event names to `EventTermCfg` instances.

```py
class ExampleSubtask(Subtask):
    name: str = "example"

    # flat plane
    subterrain: terrain_gen.SubTerrainBaseCfg = terrain_gen.MeshPlaneTerrainCfg()

    # random direction
    base_velocity: mdp.UniformVelocityCommandCfg = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(1e9, 1e9),
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(1.0, 1.0),
            lin_vel_y=(1.0, 1.0),
            ang_vel_z=(0.0, 0.0),
        ),
        heading_command=False,
    )

    # just one event
    events: Dict[str, EventTermCfg] = {
        "randomize_mass": EventTermCfg(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
                "mass_distribution_params": (-2.0, 2.0),
                "operation": "add",
                "distribution": "uniform",
            },
        )
    }
```

(See `Subtask`, `WalkSubtask`, `RunSubtask`, etc. in `benchmark_cfg.py` for more examples.)

Under the hood, `T1Baseline_BENCHMARK.setup_subtasks()` will:
   - build a `TerrainGeneratorCfg` with one column per subtask, using your subtask's `subterrain` as the column definition,
   - call `split_command_cfg` to make the per-subtask `base_velocity` commands apply only to their respective column/envs,
   - and register per-subtask events (wrapped so they only affect that subtask) with names like `"<subtask>_<event>"`.

## Modifying other parameters

**Terrain size:** The terrain generator is configured in `setup_subtasks()` as a `TerrainGeneratorCfg` with `size=(100.0, 10.0)`. Modify `size` to change the grid cell size, or change `border_width`.

**Disable/enable specific events and randomizations:** `setup_benchmark_mode()` sets or clears several `self.events` entries from T1BaselineCfg to remove event randomization. You can re-enable events by removing the `None` assignment or adjusting `EventTermCfg` parameters.

**Viewer and rendering options:** `self.viewer` sets default camera positions for visual inspection. To change the viewer for debugging non-headless runs, edit the `eye` and `lookat` values.