#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from humanoid_rl.modules import ActorCritic

class AttentionModule(ActorCritic):
    is_recurrent = False

    def __init__(
        self,
        num_obs,
        num_critic_obs,
        num_actions,
        height_scan_shape,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "AttentionModule.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        torch.nn.Module.__init__(self)
        activation = get_activation(activation)
        self.height_scan_shape = height_scan_shape
        self.num_prop_obs = num_obs - self.height_scan_shape[0]*self.height_scan_shape[1]*self.height_scan_shape[2]

        mha_embed_dim = 64
        num_heads = 4

        # Scan CNN
        self.height_scan_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding='same'),
            activation,
            nn.Conv2d(16, mha_embed_dim-3, kernel_size=5, padding='same'),
            activation
        )

        # Prop MLP
        self.prop_encoder = nn.Sequential(
            nn.Linear(self.num_prop_obs, mha_embed_dim),
        )

        # MHA
        self.mha = nn.MultiheadAttention(embed_dim=mha_embed_dim, num_heads=num_heads, batch_first=True)

        # Actor
        actor_layers = []
        actor_layers.append(nn.Linear(mha_embed_dim + self.num_prop_obs, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

         # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(num_critic_obs, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print("Attention Module")
        print(f"Height Scan Encoder: {self.height_scan_encoder}")
        print(f"Prop Encoder: {self.prop_encoder}")
        print(f"MHA: {self.mha}")
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self, observations):
        height_map = observations[:, :-self.num_prop_obs]
        batch_size = height_map.shape[0]
        height_map = height_map.reshape(batch_size, self.height_scan_shape[2], self.height_scan_shape[0], self.height_scan_shape[1])
        z_points = height_map[:, 2:3, :, :]

        map_feats = self.height_scan_encoder(z_points)  # (B, C, H, W)
        map_feats = torch.cat([map_feats, height_map], dim=1)  # (B, C+3, H, W)
        map_feats = map_feats.flatten(2).permute(0, 2, 1)  # (B, H*W, C+3)

        prop_obs = observations[:, -self.num_prop_obs:]
        prop_feats = self.prop_encoder(prop_obs).unsqueeze(dim=1)  # (B, 1, C)

        map_encoding, _ = self.mha(prop_feats, map_feats, map_feats)
        
        actor_input = torch.cat([map_encoding, prop_obs.unsqueeze(1)], dim=-1).squeeze(1)

        actions = self.actor(actor_input)
        return actions

    def update_distribution(self, observations):
        mean = self.forward(observations)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, **kwargs):
        # Supporting Phase 1 and Phase 2 inference
        if 'z' in kwargs.keys():
            obs_actor = observations
            z = kwargs['z']
        else:
            obs_actor = observations[:,self.num_env_obs:]
            z = self.get_latent(observations)
        actor_input = torch.cat([z, obs_actor], dim=-1)
        actions_mean = self.actor(actor_input)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value
    
    def get_latent(self, observations):
        encoder_observations = observations[:, :self.num_env_obs]
        return self.encoder(encoder_observations)
    

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.CReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None