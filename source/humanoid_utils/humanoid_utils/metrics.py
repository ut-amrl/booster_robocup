from isaaclab.envs import ManagerBasedRLEnv
import torch

class SurvivalTime:
    """Survival time in seconds, averaged across environments."""

    def __init__(self, env: ManagerBasedRLEnv) -> None:
        self.duration = torch.zeros(env.num_envs, dtype=torch.float)
    
    def update(self, env: ManagerBasedRLEnv, alive: torch.LongTensor) -> None:
        self.duration += alive * env.step_dt
    
    def compute(self, subtask_names) -> torch.Tensor:
        n = len(self.duration) // len(subtask_names)
        results = {name: self.duration[i*n:(i+1)*n].mean().item() for i, name in enumerate(subtask_names)}
        return results

class MovementError:
    """Average tracking error in m/s, averaged again across environments."""

    def __init__(self, env: ManagerBasedRLEnv) -> None:
        self.distance = torch.zeros(env.num_envs, dtype=torch.float)
        self.duration = torch.zeros(env.num_envs, dtype=torch.float)
    
    def update(self, env: ManagerBasedRLEnv, alive: torch.LongTensor) -> None:
        target_vel = env.command_manager.get_command("base_velocity")[:, :2]
        actual_vel = env.scene["robot"].data.root_lin_vel_b[:, :2]

        error = torch.linalg.norm((target_vel - actual_vel), dim=1)
        self.distance += alive * error.cpu() * env.step_dt
        self.duration += alive * env.step_dt

    def compute(self, subtask_names) -> torch.Tensor:
        movement_error = self.distance / self.duration
        n = len(movement_error) // len(subtask_names)
        results = {name: movement_error[i*n:(i+1)*n].mean().item() for i, name in enumerate(subtask_names)}
        return results

class Energy:
    """Average total joint energy consumption in J/s, averaged again across environments."""

    def __init__(self, env: ManagerBasedRLEnv) -> None:
        self.work = torch.zeros(env.num_envs, dtype=torch.float)
        self.duration = torch.zeros(env.num_envs, dtype=torch.float)
    
    def update(self, env: ManagerBasedRLEnv, alive: torch.LongTensor) -> None:
        robot = env.scene["robot"]
        power = torch.abs(robot.data.applied_torque * robot.data.joint_vel).sum(dim=1)
        self.work += alive * power.cpu() * env.step_dt
        self.duration += alive * env.step_dt
    
    def compute(self, subtask_names) -> torch.Tensor:
        energy = self.work / self.duration
        n = len(energy) // len(subtask_names)
        results = {name: energy[i*n:(i+1)*n].mean().item() for i, name in enumerate(subtask_names)}
        return results

class Smoothness:
    """Average total joint jerk in m/s^3, averaged again across environments."""

    def __init__(self, env: ManagerBasedRLEnv) -> None:
        self.acc_delta = torch.zeros(env.num_envs, dtype=torch.float)
        self.duration = torch.zeros(env.num_envs, dtype=torch.float)
        self.last_acc = torch.zeros(env.num_envs, env.scene["robot"].num_joints, dtype=torch.float)
    
    def update(self, env: ManagerBasedRLEnv, alive: torch.LongTensor) -> None:
        joint_acc = env.scene["robot"].data.joint_acc.cpu()
        jerk = torch.abs(joint_acc - self.last_acc).sum(dim=1) / env.step_dt
        self.acc_delta += alive * jerk * env.step_dt
        self.duration += alive * env.step_dt

        self.last_acc = joint_acc
    
    def compute(self, subtask_names) -> torch.Tensor:
        smoothness = self.acc_delta / self.duration
        n = len(smoothness) // len(subtask_names)
        results = {name: smoothness[i*n:(i+1)*n].mean().item() for i, name in enumerate(subtask_names)}
        return results