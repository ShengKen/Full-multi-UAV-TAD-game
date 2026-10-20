from __future__ import annotations

import dataclasses
import math
from typing import Callable

import torch
import torch.nn as nn

from skrl.multi_agents.torch import MultiAgentCfg

from ..models import build_mlp
from ..team_utils import aggregate_team_reward


def flat_noise_scale(timestep: int, timesteps: int, *, scheduler: bool, floor: float, halflife: float) -> float:
    if not scheduler or timesteps <= 0:
        return 1.0
    if halflife > 0:
        return floor + (1.0 - floor) * (2.0 ** (-float(timestep) / halflife))
    return max(floor, 1.0 - timestep / timesteps)


class TanhGaussianActor(nn.Module):
    _EPS = 1e-6

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden: list[int],
        activation: str = "elu",
        use_layer_norm: bool = False,
        initial_log_std: float = -1.2,
        max_log_std: float = -0.6,
    ):
        super().__init__()
        self.net = build_mlp(obs_dim, hidden, act_dim, activation, use_layer_norm)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(initial_log_std)))
        self.max_log_std = float(max_log_std)

    def distribution(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu = self.net(obs)
        log_std = self.log_std.clamp(max=self.max_log_std)
        std = log_std.exp().unsqueeze(0).expand_as(mu)
        return mu, std

    @staticmethod
    def log_prob_from_z(z: torch.Tensor, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        logn = (-0.5 * ((z - mu) / std) ** 2 - torch.log(std) - 0.5 * math.log(2.0 * math.pi)).sum(dim=-1)
        jac = torch.log(1.0 - torch.tanh(z) ** 2 + TanhGaussianActor._EPS).sum(dim=-1)
        return logn - jac

    @classmethod
    def log_prob_u(cls, u: torch.Tensor, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        z = torch.atanh(u.clamp(-1.0 + cls._EPS, 1.0 - cls._EPS))
        return cls.log_prob_from_z(z, mu, std)

    def sample(self, obs: torch.Tensor, deterministic: bool = False):
        mu, std = self.distribution(obs)
        if deterministic:
            u = torch.tanh(mu)
            return u, mu, None, (mu, std)
        z = mu + std * torch.randn_like(mu)
        u = torch.tanh(z)
        return u, z, self.log_prob_from_z(z, mu, std), (mu, std)

    def entropy(self, obs: torch.Tensor) -> torch.Tensor:
        log_std = self.log_std.clamp(max=self.max_log_std)
        return (0.5 * (math.log(2.0 * math.pi * math.e) + log_std)).sum().expand(obs.shape[0])


class ValueMLP(nn.Module):
    def __init__(self, state_dim: int, hidden: list[int], activation: str = "elu", use_layer_norm: bool = False):
        super().__init__()
        self.net = build_mlp(state_dim, hidden, 1, activation, use_layer_norm)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


@dataclasses.dataclass(kw_only=True)
class FlatPPO_CFG(MultiAgentCfg):
    rollouts: int = 512

    learning_epochs: int = 3

    mini_batches: int = 8

    discount_factor: float = 0.995

    gae_lambda: float = 0.95

    learning_rate: float | tuple[float, float] = 3.0e-4

    grad_norm_clip: float = 1.0

    learning_starts: int = 0

    random_timesteps: int = 0

    ratio_clip: float = 0.2

    value_loss_scale: float = 0.5

    entropy_loss_scale: float = 0.0

    kl_threshold: float = 0.0

    mixed_precision: bool = False

    rewards_shaper: Callable | None = None
