from __future__ import annotations

import os
import shutil
import pathlib
from dataclasses import asdict
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
except ModuleNotFoundError:
    raise ModuleNotFoundError("Wandb is required to log to Weights and Biases.")

class WandbSummaryWriter(SummaryWriter):
    """Summary writer for Weights and Biases."""

    def __init__(self, log_dir: str, flush_secs: int, cfg):
        super().__init__(log_dir, flush_secs)

        try:
            project = cfg["wandb_project"]
        except KeyError:
            raise KeyError("Please specify wandb_project in the runner config, e.g. legged_gym.")

        try:
            entity = os.environ["WANDB_USERNAME"]
        except KeyError:
            raise KeyError(
                "Wandb username not found. Please run or add to ~/.bashrc: export WANDB_USERNAME=YOUR_USERNAME"
            )

        wandb.init(project=project, entity=entity)

        # Change generated name to project-number format
        wandb.run.name = project + wandb.run.name.split("-")[-1]

        self.name_map = {
            "Train/mean_reward/time": "Train/mean_reward_time",
            "Train/mean_episode_length/time": "Train/mean_episode_length_time",
        }

        run_name = os.path.split(log_dir)[-1]

        wandb.log({"log_dir": run_name})

        self.saved_videos = {}
        self.fps = 50

    def store_config(self, env_cfg, runner_cfg, alg_cfg, policy_cfg):
        wandb.config.update({"runner_cfg": runner_cfg})
        wandb.config.update({"policy_cfg": policy_cfg})
        wandb.config.update({"alg_cfg": alg_cfg})
        wandb.config.update({"env_cfg": asdict(env_cfg)})

    def _map_path(self, path):
        if path in self.name_map:
            return self.name_map[path]
        else:
            return path

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None, new_style=False):
        super().add_scalar(
            tag,
            scalar_value,
            global_step=global_step,
            walltime=walltime,
            new_style=new_style,
        )
        wandb.log({self._map_path(tag): scalar_value}, step=global_step)

    def log_video_files(self, log_name: str = "Video", video_subdir: str | None = "videos"):
        if video_subdir is not None:
            video_dir = pathlib.Path(os.path.join(self.log_dir, video_subdir))
        else:
            video_dir = pathlib.Path(self.log_dir)
        videos = list(video_dir.rglob("*.mp4"))
        for video in videos:
            video_name = str(video)
            video_size_kb = os.stat(video_name).st_size / 1024
            if video_name not in self.saved_videos.keys():
                self.saved_videos[video_name] = {"size": video_size_kb, "recorded": False, "steps": 0}
            else:
                video_info = self.saved_videos[video_name]
                if video_info["recorded"]:
                    continue
                elif video_info["size"] == video_size_kb and video_size_kb > 100:
                    # wait 10 steps after recording has been completed
                    if video_info["steps"] > 10:
                        self.add_video(video_name, fps=self.fps, log_name=log_name)
                        self.saved_videos[video_name]["recorded"] = True
                    else:
                        video_info["steps"] += 1
                else:
                    self.saved_videos[video_name]["size"] = video_size_kb
                    self.saved_videos[video_name]["steps"] = 0    

    def add_video(self, video_path: str, fps: int = 50, log_name: str = "Video"):
        wandb.log({log_name: wandb.Video(video_path, fps=fps)})    

    def callback(self, step):
        self.log_video_files()

    def stop(self):
        wandb.finish()

    def log_config(self, env_cfg, runner_cfg, alg_cfg, policy_cfg):
        self.store_config(env_cfg, runner_cfg, alg_cfg, policy_cfg)
        env_cfg_dict = asdict(env_cfg)
        self.fps = 1 / (env_cfg_dict["decimation"] * env_cfg_dict["sim"]["dt"])

    def save_model(self, model_path, iter):
        wandb.save(model_path, base_path=os.path.dirname(model_path))

    def save_file(self, path, iter=None):
        wandb.save(path, base_path=os.path.dirname(path))

    def set_fps(self, fps):
        self.fps = fps

def pull_policy_from_wandb(save_dir: str, run_path: str, model_name: str) -> tuple[str, dict]:
    import wandb

    # login to wandb
    wandb.login()
    api = wandb.Api()
    wandb_run = api.run(run_path)

    # download models to tmp_models folder
    wandb_folder_path = os.path.join(save_dir, run_path)
    os.makedirs(wandb_folder_path, exist_ok=True)
    model_file = wandb_run.file(model_name)
    print(f"[INFO] Downloading model file to {wandb_folder_path}/{model_name}")
    model_file.download(f"{wandb_folder_path}", replace=True)
    resume_path = os.path.abspath(os.path.join(wandb_folder_path, model_name))
    # pull wandb model config
    print("[INFO] Pulling policy config from wandb")
    env_cfg = wandb_run.config["env_cfg"]
    return resume_path, env_cfg

def load_wandb_policy(run_path: str = "", model_name: str = "", log_root_path: str = "") -> tuple[str, dict]:
    for i in range(3):
        if run_path == "":
            run_path = input(
                "\033[96mEnter the Weights and Biases run path located on the Overview panel; i.e"
                " usr/Spot-Blind/abc123\033[0m\n"
            )
        if model_name == "":
            model_name = input(
                "\n\033[96mEnter the name of the model file to download; i.e model_100.pt \n"
                + "Press Enter again without a file name to quit.\033[0m\n"
            )
        if model_name == "":
            return
        if model_name[:6] != "model_":
            model_name = "model_" + model_name
        if model_name[-3:] != ".pt":
            model_name += ".pt"
        try:
            resume_path, env_cfg = pull_policy_from_wandb(log_root_path, run_path, model_name)
            print(f"\033[92m\n[INFO] added policy to load\033[0m")
            model_file_name = os.path.splitext(os.path.basename(resume_path))[0]
            model_dir = os.path.join(resume_path, run_path.split("/")[-1])
            # os.makedirs(model_dir, exist_ok=True)
            # shutil.copy2(f"{resume_path}", f"{model_dir}/{model_file_name}.pt")
            break
        except Exception:
            print(
                "\n\033[91m[WARN] Unable to download from Weights and Biases, is the path"
                " and filename correct?\033[0m"
            )
            print(f"run_path: {run_path}\nmodel_name: {model_name}")
            run_path = ""
            model_name = ""

    return resume_path, env_cfg
