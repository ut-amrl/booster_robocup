import os
import torch
import copy


def export_policy_as_jit(
    actor_critic: object,
    normalizer: object | None,
    path: str,
    filename="policy.pt",
    alg: str = "RSL_RL",
):
    """Export policy into a Torch JIT file.

    Args:
        actor_critic: The actor-critic torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported JIT file. Defaults to "policy.pt".
    """
    if alg == "rma":
        policy_exporter = _RMAJitPolicyExporter(actor_critic, normalizer)
    else:
        policy_exporter = _TorchPolicyExporter(actor_critic, normalizer)
    policy_exporter.export(path, filename)


def export_policy_as_onnx(
    actor_critic: object, path: str, filename="policy.onnx", verbose=False
) -> None:
    """Export policy into a Torch ONNX file.

    Args:
        actor_critic: The actor-critic torch module.
        path: The path to the saving directory.
        filename: The name of exported onnx file. Defaults to "policy.onnx".
        verbose: Whether to print the model summary. Defaults to False.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxPolicyExporter(actor_critic, verbose)
    print(f"Saving {filename}/{path}")
    policy_exporter.export(path, filename)


"""
Helper Classes - Private.
"""


class _OnnxPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into ONNX file."""

    def __init__(self, actor_critic, verbose=False) -> None:
        super().__init__()
        self.verbose = verbose
        self.actor = copy.deepcopy(actor_critic.actor)

    def forward(self, x) -> torch.Tensor:
        return self.actor(x)

    def export(self, path, filename) -> None:
        self.to("cpu")
        layer = self.actor[0]
        while type(layer).__name__ == "Sequential":
            layer = layer[0]
        obs = torch.zeros(1, layer.in_features)
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


class _TorchPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into JIT file."""

    def __init__(self, actor_critic, normalizer=None):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        if self.is_recurrent:
            self.rnn = copy.deepcopy(actor_critic.memory_a.rnn)
            self.rnn.cpu()
            self.register_buffer(
                "hidden_state",
                torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size),
            )
            self.register_buffer(
                "cell_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            )
            self.forward = self.forward_lstm
            self.reset = self.reset_memory
        # copy normalizer if exists
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward_lstm(self, x):
        x = self.normalizer(x)
        x, (h, c) = self.rnn(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        x = x.squeeze(0)
        return self.actor(x)

    def forward(self, x):
        return self.actor(self.normalizer(x))

    @torch.jit.export
    def reset(self):
        pass

    def reset_memory(self):
        self.hidden_state[:] = 0.0
        self.cell_state[:] = 0.0

    def export(self, path, filename):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, filename)
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


class _RMAJitPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into JIT file."""

    def __init__(self, actor_critic, normalizer=None):
        super().__init__()
        self.encoder = copy.deepcopy(actor_critic.encoder)
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        if self.is_recurrent:
            self.rnn = copy.deepcopy(actor_critic.memory_a.rnn)
            self.rnn.cpu()
            self.register_buffer(
                "hidden_state",
                torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size),
            )
            self.register_buffer(
                "cell_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            )
            self.forward = self.forward_lstm
            self.reset = self.reset_memory
        # copy normalizer if exists
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward_lstm(self, obs):
        obs = self.normalizer(obs)
        x = obs[:, : self.encoder.in_features]
        fts = self.encoder(x)
        actor_obs = torch.cat((fts, obs[:, self.encoder.in_features :]), dim=-1)
        return self.actor(actor_obs)

    def forward(self, obs):
        obs = self.normalizer(obs)
        image_len = 128 * 128
        x = obs[:, :image_len]
        fts = self.encoder(x)
        actor_obs = torch.cat((fts, obs[:, image_len:]), dim=-1)
        return self.actor(actor_obs)

    @torch.jit.export
    def reset(self):
        pass

    def reset_memory(self):
        self.hidden_state[:] = 0.0
        self.cell_state[:] = 0.0

    def export(self, path, filename):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, filename)
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)
