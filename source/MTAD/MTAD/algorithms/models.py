# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.

# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl.memories.torch import RandomMemory
from skrl.utils.model_instantiators.torch import categorical_model, deterministic_model, gaussian_model
from gymnasium.spaces import Box, Discrete


SET_ATTN_TOKEN_DIM = 19


def mlp_backbone_fingerprint_spec() -> dict:
    return {"backbone": "mlp"}


def zero_initialize_module(module: nn.Module) -> None:
    with torch.no_grad():
        for submodule in module.modules():
            if isinstance(submodule, nn.Linear):
                nn.init.constant_(submodule.weight, 0.0)
                if submodule.bias is not None:
                    nn.init.constant_(submodule.bias, 0.0)
            elif isinstance(submodule, nn.LayerNorm):
                if submodule.elementwise_affine:
                    if submodule.weight is not None:
                        nn.init.constant_(submodule.weight, 0.0)
                    if submodule.bias is not None:
                        nn.init.constant_(submodule.bias, 0.0)


def build_mlp(input_dim: int, hidden, output_dim: int, activation: str, use_layer_norm: bool = False) -> nn.Sequential:
    """Build a multilayer network with the requested activation and normalization."""

    act_fn = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}[activation]
    layers = []
    last = input_dim
    for h in hidden:
        layers.append(nn.Linear(last, h))
        if use_layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(act_fn())
        last = h
    layers.append(nn.Linear(last, output_dim))
    return nn.Sequential(*layers)


def _space_size(space) -> int:
    if isinstance(space, Discrete):
        return int(space.n)
    if isinstance(space, Box):
        return int(space.shape[0])
    if hasattr(space, "shape") and space.shape:
        return int(space.shape[0])
    return int(space)


_CTRL_LOG_STD_INIT_DEFAULT = math.log(0.3)
_CTRL_LOG_STD_MAX_DEFAULT = math.log(0.6)


def ctrl_log_std_bounds(rl_cfg: dict) -> tuple[float, float]:
    init = float(rl_cfg.get("ctrl_initial_log_std", _CTRL_LOG_STD_INIT_DEFAULT))
    mx = float(rl_cfg.get("ctrl_max_log_std", _CTRL_LOG_STD_MAX_DEFAULT))
    if mx < init:
        raise ValueError(
            f"ctrl_max_log_std({mx:.4f}) < ctrl_initial_log_std({init:.4f})：σ 初值一出生就被裁剪，检查配置"
        )
    return init, mx


def _make_masked_compute(orig_compute):
    import os as _os

    _mask_debug = _os.environ.get("TAD_MASK_DEBUG", "0") == "1"

    def _compute(inputs, role=""):
        logits, outputs = orig_compute(inputs, role)
        mask = inputs.get("action_mask", None)
        if mask is not None:
            if _mask_debug:
                _mn, _mx = mask.min().item(), mask.max().item()
                if _mn < -1e-6 or _mx > 1.0 + 1e-6:
                    raise ValueError(f"action_mask 含非 0/1 值 [{_mn:.4g},{_mx:.4g}]；疑似误切（如把候选特征当 mask）")
            logits = logits.masked_fill(~mask.bool(), -1e9)
        outputs = dict(outputs)
        outputs["logits"] = logits
        outputs["net_output"] = logits
        return logits, outputs

    return _compute


class _PlainActorShim(nn.Module):
    @property
    def device(self):
        return next(self.parameters()).device

    def enable_training_mode(self, enabled: bool = True) -> None:
        self.train(enabled)

    def broadcast_parameters(self, *, rank: int = 0) -> None:
        state = [self.state_dict()]
        torch.distributed.broadcast_object_list(state, rank)
        self.load_state_dict(state[0])

    def reduce_parameters(self) -> None:
        grads = [p.grad.reshape(-1) for p in self.parameters() if p.grad is not None]
        if not grads:
            return
        flat = torch.cat(grads)
        torch.distributed.all_reduce(flat, op=torch.distributed.ReduceOp.SUM)
        offset = 0
        world = torch.distributed.get_world_size()
        for p in self.parameters():
            if p.grad is None:
                continue
            n = p.numel()
            p.grad.copy_(flat[offset : offset + n].view_as(p.grad) / world)
            offset += n


class MaskedCategoricalActor(_PlainActorShim):
    def __init__(self, obs_dim: int, target_dim: int, hidden, activation: str, use_layer_norm: bool = False):
        super().__init__()
        self.target_dim = int(target_dim)
        self.net = build_mlp(obs_dim, hidden, target_dim, activation, use_layer_norm)

    def _distribution(self, obs, target_mask=None):
        logits = self.net(obs)
        if target_mask is not None:
            logits = logits.masked_fill(~target_mask.bool(), -1e9)
        return torch.distributions.Categorical(logits=logits)

    def sample(self, obs, target_mask=None, deterministic=False) -> dict:
        distribution = self._distribution(obs, target_mask)
        idx = distribution.logits.argmax(dim=-1) if deterministic else distribution.sample()
        return {"idx": idx, "log_prob": distribution.log_prob(idx)}

    def log_prob(self, obs, idx, target_mask=None) -> torch.Tensor:
        return self._distribution(obs, target_mask).log_prob(idx.long())

    def entropy(self, obs, target_mask=None) -> torch.Tensor:
        return self._distribution(obs, target_mask).entropy()


class MaskedGumbelActor(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int, hidden, activation: str, use_layer_norm: bool = False):
        super().__init__()
        self.net = build_mlp(obs_dim, hidden, num_actions, activation, use_layer_norm)
        self.num_actions = num_actions

    def _logits(self, obs, mask):
        l = self.net(obs)
        if mask is not None:
            l = l.masked_fill(~mask.bool(), -1e9)
        return l

    def forward(self, obs, mask=None, tau: float = 1.0, hard: bool = True):
        return F.gumbel_softmax(self._logits(obs, mask), tau=tau, hard=hard, dim=-1)

    def argmax_onehot(self, obs, mask=None):
        l = self._logits(obs, mask)
        oh = torch.zeros_like(l)
        oh.scatter_(-1, l.argmax(dim=-1, keepdim=True), 1.0)
        return oh


def build_mappo_models(
    observation_spaces: dict, action_spaces: dict, state_space, num_envs: int, device, rl_cfg: dict, team_of: dict
) -> tuple[dict, dict]:
    possible_agents = list(observation_spaces.keys())

    critic_hidden = rl_cfg.get("critic_hidden", [256, 256])
    activation = rl_cfg.get("activation", "elu")
    rollouts = int(rl_cfg.get("rollouts", 16))
    use_layer_norm = bool(rl_cfg.get("use_layer_norm", False))
    zero_init = bool(rl_cfg.get("zero_init", False))

    hier = bool(rl_cfg.get("hier", False))

    def _layer_of_uid(u: str) -> str:
        if hier:
            return "dec" if u.endswith("/dec") else "ctrl"
        return "agent"

    team_critics = {}
    team_layers = {(team_of[u], _layer_of_uid(u)) for u in possible_agents}
    for team, layer in sorted(team_layers):
        team_critics[(team, layer)] = deterministic_model(
            state_space=state_space,
            device=device,
            network=[
                {
                    "name": "net",
                    "input": "STATES",
                    "layers": critic_hidden,
                    "activations": activation,
                }
            ],
            output="ONE",
        )

        if use_layer_norm or zero_init:
            state_dim = state_space.shape[0] if hasattr(state_space, "shape") else int(state_space)
            team_critics[(team, layer)].net_container = build_mlp(
                state_dim, critic_hidden, 1, activation, use_layer_norm=use_layer_norm
            ).to(device)
        if zero_init:
            zero_initialize_module(team_critics[(team, layer)].net_container)

    models = {}
    clip_actions = bool(rl_cfg.get("clip_actions", False))

    _ctrl_ls_init, _ctrl_ls_max = ctrl_log_std_bounds(rl_cfg)
    discrete = bool(rl_cfg.get("is_discrete", False))

    actor_team_share = bool((hier or discrete) and rl_cfg.get("actor_mode", "per_agent") == "team")
    shared_actors: dict = {}

    n_pool = int(rl_cfg.get("n_targets_max", 0))
    for uid in possible_agents:
        act_space = action_spaces[uid]

        if actor_team_share:
            if hier:
                _layer = "dec" if uid.endswith("/dec") else "ctrl"
            else:
                _layer = "agent"
            _gk = (uid.split("/")[0].split("_")[0] if hier else uid.split("_")[0], _layer)
        else:
            _gk = None

        if actor_team_share and _gk in shared_actors:
            models[uid] = {"policy": shared_actors[_gk], "value": team_critics[(team_of[uid], _layer)]}
            continue

        actor_hidden_uid = (
            list(rl_cfg.get("upper_actor_hidden", [256, 256]))
            if hier and uid.endswith("/dec")
            else list(rl_cfg.get("lower_actor_hidden", [256, 256]))
            if hier and uid.endswith("/ctrl")
            else [256, 256]
        )

        uid_discrete = isinstance(act_space, Discrete) if hier else discrete
        if uid_discrete:
            actor = categorical_model(
                observation_space=observation_spaces[uid],
                action_space=act_space,
                device=device,
                unnormalized_log_prob=True,
                network=[
                    {"name": "net", "input": "OBSERVATIONS", "layers": actor_hidden_uid, "activations": activation}
                ],
                output="ACTIONS",
            )
            if use_layer_norm or zero_init:
                actor.net_container = build_mlp(
                    _space_size(observation_spaces[uid]),
                    actor_hidden_uid,
                    _space_size(act_space),
                    activation,
                    use_layer_norm=use_layer_norm,
                ).to(device)
            if zero_init:
                zero_initialize_module(actor.net_container)
            actor.compute = _make_masked_compute(actor.compute)
        else:
            if clip_actions and isinstance(act_space, int):
                act_space = Box(low=-1.0, high=1.0, shape=(act_space,))
            actor = gaussian_model(
                observation_space=observation_spaces[uid],
                action_space=act_space,
                device=device,
                clip_actions=clip_actions,
                clip_log_std=True,
                min_log_std=-20.0,
                max_log_std=_ctrl_ls_max,
                initial_log_std=_ctrl_ls_init,
                network=[
                    {
                        "name": "net",
                        "input": "OBSERVATIONS",
                        "layers": actor_hidden_uid,
                        "activations": activation,
                    }
                ],
                output="ACTIONS",
            )

            if use_layer_norm or zero_init:
                obs_dim_u = (
                    observation_spaces[uid].shape[0]
                    if hasattr(observation_spaces[uid], "shape")
                    else int(observation_spaces[uid])
                )
                act_dim_u = act_space.shape[0] if hasattr(act_space, "shape") else int(act_space)
                actor.net_container = build_mlp(
                    obs_dim_u, actor_hidden_uid, act_dim_u, activation, use_layer_norm=use_layer_norm
                ).to(device)
            if zero_init:
                zero_initialize_module(actor.net_container)

        models[uid] = {"policy": actor, "value": team_critics[(team_of[uid], _layer_of_uid(uid))]}
        if actor_team_share:
            shared_actors[_gk] = actor

    memories = {uid: RandomMemory(num_envs=num_envs, memory_size=rollouts, device=device) for uid in possible_agents}

    return models, memories
