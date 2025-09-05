import os
import torch

def export_RMA_policy_as_onnx(base_policy: object, adaption_module: object, path: str, filename="policy.onnx", verbose=False) -> None:
    """Export policy into a Torch ONNX file.

    Args:
        actor_critic: The actor-critic torch module.
        path: The path to the saving directory.
        filename: The name of exported onnx file. Defaults to "policy.onnx".
        verbose: Whether to print the model summary. Defaults to False.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _RMAOnnxPolicyExporter(base_policy, adaption_module, verbose)
    print(f"Saving {path}/{filename}")
    policy_exporter.export(path, filename)

class _RMAOnnxPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into ONNX file."""

    def __init__(self, base_policy, adaption_module, verbose=False) -> None:
        super().__init__()
        self.verbose = verbose
        self.mlp = copy.deepcopy(adaption_module.encoder)
        self.conv_net = copy.deepcopy(adaption_module.conv_net)
        self.actor = copy.deepcopy(base_policy.actor)

    def forward(self, obs) -> torch.Tensor:
        history = 50
        obs_dict = {
            "lin_vel": 3,
            "ang_vel": 3,
            "proj_g": 3,
            "vel_com": 3,
            "joint_pos": 12,
            "joint_vel": 12,
            "actions": 12
        }

        index = 0
        obs_ordered = torch.empty((obs.shape[0], 0), dtype=torch.float32)
        obs_actor = torch.empty((obs.shape[0], 0), dtype=torch.float32)
        for v in obs_dict.values():
            for i in range(v):
                obs_slice = obs[:, index+i:index+v*history:v]
                obs_ordered = torch.cat([obs_ordered, obs_slice], dim=-1)
                a = obs[:, index+i].unsqueeze(dim=1)
                obs_actor = torch.cat([obs_actor, a], dim=-1)
            index += v*history

        obs_ordered = obs_ordered.view(1, 48, 50)
        obs_ordered = obs_ordered.transpose(2,1)
        z_hat = self.mlp(obs_ordered)
        z_hat = z_hat.transpose(2,1)
        z_hat = self.conv_net(z_hat)
        actor_obs = torch.cat((z_hat,obs_actor), dim=-1)
        acts = self.actor(actor_obs)
        return acts

    def export(self, path, filename) -> None:
        self.to("cpu")
        obs = torch.zeros(1, 2400)
        torch.onnx.export(
            self,
            obs,
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={},
        )
        