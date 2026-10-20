from __future__ import annotations

import torch
import torch.nn as nn
from torch.func import functional_call, vmap

from .models import build_mlp


class QLowCritic(nn.Module):
    def __init__(
        self, obs_dim: int, act_dim: int = 4, hidden=(256, 256), activation: str = "elu", use_layer_norm: bool = False
    ):
        super().__init__()
        self.net = build_mlp(int(obs_dim) + int(act_dim), list(hidden), 1, activation, use_layer_norm)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1))


class BatchedQLowCritic(nn.Module):
    def __init__(
        self,
        n_aid: int,
        obs_dim: int,
        act_dim: int = 4,
        hidden=(256, 256),
        activation: str = "elu",
        use_layer_norm: bool = False,
        device=None,
    ):
        super().__init__()
        self.n_aid = int(n_aid)

        self.template = QLowCritic(obs_dim, act_dim, hidden, activation, use_layer_norm)
        if device is not None:
            self.template = self.template.to(device)
        self._names = [name for name, _ in self.template.named_parameters()]
        self._key_for = {name: name.replace(".", "_") for name in self._names}

        inits = [QLowCritic(obs_dim, act_dim, hidden, activation, use_layer_norm) for _ in range(self.n_aid)]
        self.params = nn.ParameterDict()
        for name in self._names:
            stacked = torch.stack([dict(m.named_parameters())[name].detach() for m in inits])
            if device is not None:
                stacked = stacked.to(device)
            self.params[self._key_for[name]] = nn.Parameter(stacked.clone())

    def _param_dict(self) -> dict:
        return {name: self.params[self._key_for[name]] for name in self._names}

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        def one(p_i, o_i, a_i):
            return functional_call(self.template, p_i, (o_i, a_i)).squeeze(-1)

        return vmap(one)(self._param_dict(), obs, action)

    def forward_one(self, i: int, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        p_i = {name: self.params[self._key_for[name]][i] for name in self._names}
        return functional_call(self.template, p_i, (obs, action)).squeeze(-1)

    def load_per_aid(self, state_dicts: list[dict]) -> None:
        assert len(state_dicts) == self.n_aid, f"期望 {self.n_aid} 份 state_dict，收到 {len(state_dicts)}"
        for i, sd in enumerate(state_dicts):
            self.load_one(i, sd)

    def load_one(self, i: int, sd: dict) -> None:
        for name in self._names:
            self.params[self._key_for[name]].data[i] = sd[name]

    def dump_per_aid(self) -> list[dict]:
        return [self.dump_one(i) for i in range(self.n_aid)]

    def dump_one(self, i: int) -> dict:
        return {name: self.params[self._key_for[name]][i].detach().clone() for name in self._names}


class _QLowSliceCheckpoint(nn.Module):
    def __init__(self, batched: "BatchedQLowCritic", i: int):
        super().__init__()
        self._batched = batched
        self._i = int(i)

    def state_dict(self, *args, **kwargs):
        return self._batched.dump_one(self._i)

    def load_state_dict(self, state_dict, strict=True):
        self._batched.load_one(self._i, state_dict)
        return None
