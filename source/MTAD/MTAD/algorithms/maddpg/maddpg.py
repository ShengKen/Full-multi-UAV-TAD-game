# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.

# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import contextlib

import gymnasium
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.spaces import Discrete

from skrl.multi_agents.torch import MultiAgent

from ..models import build_mlp, zero_initialize_module, MaskedGumbelActor, mlp_backbone_fingerprint_spec
from ..running_standard_scaler import RunningStandardScaler
from ..resume_checkpoint import register_training_progress, register_architecture_fingerprint
from ..team_utils import (
    masked_mean,
    physical_uid,
    select_dynamic_anchor_indices,
    select_dynamic_team_anchor,
    stage_decisions_rebuild_ctrl,
)
from ..roll_rotation import update_roll_rotation
from ..train_takeover import update_train_takeover
from .maddpg_cfg import MADDPG_CFG


class _IdentityPreprocessor(nn.Module):
    def forward(self, x: torch.Tensor, train: bool = False) -> torch.Tensor:
        return x


class ActorMLP(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: list[int], activation: str, use_layer_norm: bool = False):
        super().__init__()

        self.net = build_mlp(obs_dim, hidden, act_dim, activation, use_layer_norm)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs))


class CriticMLP(nn.Module):
    def __init__(
        self, state_dim: int, total_act_dim: int, hidden: list[int], activation: str, use_layer_norm: bool = False
    ):
        super().__init__()
        self.net = build_mlp(state_dim + total_act_dim, hidden, 1, activation, use_layer_norm)

    def forward(self, state: torch.Tensor, joint_action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, joint_action], dim=-1)
        return self.net(x)


class MADDPG(MultiAgent):
    _high_ppo_mode = False

    def __init__(
        self,
        *,
        possible_agents: list[str],
        observation_spaces: dict[str, gymnasium.Space],
        action_spaces: dict[str, gymnasium.Space],
        state_space: gymnasium.Space,
        num_envs: int,
        device: str | torch.device,
        team_of: dict[str, str],
        cfg: dict,
        team_reward_aggr: str = "sum",
        critic_hidden: list[int] | None = None,
        activation: str = "elu",
    ):
        self._gamma_control = float(cfg.get("discount_factor_control", cfg.get("discount_factor", 0.99)))
        self._gamma = self._gamma_control

        _P_dec = max(1, int(cfg.get("decision_period", 50)))
        _gd_raw = cfg.get("discount_factor_decision", None)
        self._gamma_decision = float(_gd_raw) if _gd_raw is not None else self._gamma_control**_P_dec
        self._rho_high = self._gamma_decision ** (1.0 / _P_dec)

        self._time_limit_bootstrap = bool(cfg.get("time_limit_bootstrap", False))
        self._polyak = float(cfg.get("polyak", 0.995))
        self._grad_norm_clip = float(cfg.get("grad_norm_clip", 1.0))
        self._learning_starts = int(cfg.get("learning_starts", 1000))
        self._cfg = cfg
        self._gradient_steps = int(cfg.get("gradient_steps", 1))

        self._update_every = max(1, int(cfg.get("update_every", 1)))

        self._telemetry_interval = max(1, int(cfg.get("telemetry_interval", cfg.get("write_interval", 50))))
        self._random_timesteps = int(cfg.get("random_timesteps", 1000))
        self._policy_lr = float(cfg.get("policy_lr", 3e-4))
        self._critic_lr = float(cfg.get("critic_lr", 3e-4))
        self._noise_std = float(cfg.get("exploration_noise_std", 0.3))
        self._noise_scheduler = bool(cfg.get("exploration_scheduler", True))
        self._noise_floor = min(1.0, max(0.0, float(cfg.get("exploration_noise_floor", 0.0))))

        self._noise_halflife = float(cfg.get("exploration_noise_halflife", 0.0) or 0.0)
        self._q_low_lr = float(cfg.get("q_low_lr", self._critic_lr))
        self._batch_size = int(cfg.get("batch_size", 256))
        self._replay_size = int(cfg.get("replay_size", 1000000))

        _replay_steps_key = "replay_steps"
        _replay_steps = int(cfg.get(_replay_steps_key, cfg.get("replay_steps", 0)))
        self._num_envs_for_replay = num_envs
        self._replay_steps = _replay_steps

        self._replay_device = str(cfg.get("replay_device", "cpu")).lower()

        self._prefetch = bool(cfg.get("prefetch_replay", False)) and str(device).startswith("cuda")
        self._prefetch_stream = torch.cuda.Stream(device=device) if self._prefetch else None
        self._pending_low: dict = {}
        self._q_low_batched = False
        self._q_low_idx: dict = {}

        self._n_step = max(1, int(cfg.get("n_step", 1)))

        self._replay_terminal_mix = min(max(float(cfg.get("replay_terminal_mix", 0.25)), 0.0), 1.0)
        self._replay_terminal_thresh = float(cfg.get("replay_terminal_thresh", 20.0))
        self._q_low_team: dict = {}
        self._team_of = team_of
        self._team_reward_aggr = team_reward_aggr
        self.frozen_agents: set[str] = set()
        self._tensor_track: dict[str, list[torch.Tensor]] = {}

        self._upper_actor_hidden = list(cfg.get("upper_actor_hidden", [256, 256]))
        self._lower_actor_hidden = list(cfg.get("lower_actor_hidden", [256, 256]))
        critic_hidden = critic_hidden or cfg.get("critic_hidden", [256, 256])

        self._hier = bool(cfg.get("hier", False))

        self._identity_preprocessor = _IdentityPreprocessor()
        self._normalize = bool(cfg.get("normalize", True))
        self._norm_eps = float(cfg.get("norm_epsilon", 1e-8))
        self._norm_clip = float(cfg.get("norm_clip_threshold", 5.0))

        self._use_layer_norm = bool(cfg.get("use_layer_norm", False))
        self._zero_init = bool(cfg.get("zero_init", False))

        self._actor_mode = str(cfg.get("actor_mode", "per_agent"))

        self._batched_fwd = bool(cfg.get("perf_batched_actor", True))

        self._team_members: dict[str, list[str]] = {}
        for uid in possible_agents:
            self._team_members.setdefault(team_of[uid], []).append(uid)

        self._team_drone_count: dict[str, int] = {
            team: len({physical_uid(m) for m in members}) for team, members in self._team_members.items()
        }

        self._team_representative: dict[str, str] = {team: members[0] for team, members in self._team_members.items()}
        self._uid_index = {uid: i for i, uid in enumerate(possible_agents)}
        self._total_timesteps = None

        self._state_dim = state_space.shape[0]
        self._num_agents = len(possible_agents)
        if self._hier:
            self._obs_dim = observation_spaces[possible_agents[0]].shape[0]
            self._obs_dim_per_agent = {uid: observation_spaces[uid].shape[0] for uid in possible_agents}
            self._discrete = False
            self._n_pool = int(cfg.get("n_targets_max", 0))
            self._dec_exec_dim = self._n_pool
            self._dec_repr_dim = self._dec_exec_dim
            self._mask_tail_dim = self._n_pool
            self._nm = self._dec_repr_dim
            self._drone_uids = list(cfg.get("all_uids", []))
            self._drone_index = {u: i for i, u in enumerate(self._drone_uids)}
            self._M = len(self._drone_uids)
            self._block = self._dec_repr_dim + 4
            self._total_act_dim = self._M * self._block
            self._high_act_dim = self._M * self._dec_repr_dim
            self._act_dim = self._block
            self._is_dec = {uid: isinstance(action_spaces[uid], Discrete) for uid in possible_agents}

            self._hier_dual = self._hier and any(self._is_dec.values()) and any(not v for v in self._is_dec.values())
            self._critic_act_dim = self._high_act_dim if self._hier_dual else self._total_act_dim
            self._n_targets = {uid: self._mask_tail_dim for uid in possible_agents if self._is_dec[uid]}
            self._gumbel_tau = float(cfg.get("gumbel_tau", 1.0))
            self._eps_greedy = float(cfg.get("epsilon_greedy", 0.1))
            self._eps_decay = float(cfg.get("eps_decay", 1.0))

            self._agent_slot = {}
            for uid in possible_agents:
                du = uid.split("/")[0]
                di = self._drone_index[du]
                base = di * self._block
                if self._is_dec[uid]:
                    self._agent_slot[uid] = (di, "dec", base, self._nm)
                else:
                    self._agent_slot[uid] = (di, "ctrl", base + self._nm, 4)

        import dataclasses

        known = {f.name for f in dataclasses.fields(MADDPG_CFG)}
        cfg_clean = {k: v for k, v in cfg.items() if k in known}

        state_spaces = {uid: state_space for uid in possible_agents}

        models = {uid: {} for uid in possible_agents}

        super().__init__(
            cfg=MADDPG_CFG(**cfg_clean),
            possible_agents=possible_agents,
            models=models,
            memories=None,
            observation_spaces=observation_spaces,
            state_spaces=state_spaces,
            action_spaces=action_spaces,
            device=device,
        )

        self._init_hier(critic_hidden, activation, num_envs)

    @staticmethod
    def _alive_info(infos, key: str) -> dict[str, torch.Tensor] | None:
        value = infos.get(key) if isinstance(infos, dict) else None
        return value if isinstance(value, dict) else None

    @staticmethod
    def _team_term_sum(term_info, members, reference: torch.Tensor) -> torch.Tensor | None:
        if not isinstance(term_info, dict):
            return None
        picked: dict[str, torch.Tensor] = {}
        for m in members:
            uid = physical_uid(m)
            if uid in picked:
                continue
            value = term_info.get(m, term_info.get(uid))
            if value is not None:
                picked[uid] = value.to(reference.device).view_as(reference)
        if not picked:
            return None
        total = None
        for value in picked.values():
            total = value.clone() if total is None else total + value
        return total

    def _active_per_agent(self, infos, key: str, reference: torch.Tensor) -> dict[str, torch.Tensor]:
        alive = self._alive_info(infos, key)
        n = reference.shape[0]
        out = {}
        for aid in self.possible_agents:
            mask = None if alive is None else alive.get(aid, alive.get(physical_uid(aid)))
            if mask is None:
                mask = torch.ones(n, dtype=torch.bool, device=reference.device)
            out[aid] = mask.to(reference.device).bool().view(-1)
        return out

    def _team_state_anchor(self, states, team: str, alive):
        rep = self._team_representative[team]
        if not isinstance(states, dict) or not isinstance(alive, dict):
            return (states[rep] if isinstance(states, dict) else states), None

        state_members = []
        seen = set()
        for aid in self._team_members[team]:
            uid = physical_uid(aid)
            if uid in seen:
                continue
            seen.add(uid)
            if aid in states:
                state_members.append(aid)
            elif uid in states:
                state_members.append(uid)
        if not state_members:
            return states[rep], None
        fallback = next((m for m in state_members if physical_uid(m) == physical_uid(rep)), state_members[0])
        if getattr(self, "_state_anchor_free", False):
            try:
                idx = select_dynamic_anchor_indices(
                    alive, state_members, fallback_member=fallback, device=states[rep].device
                )
            except (KeyError, ValueError):
                idx = None
            return states[rep], idx
        try:
            return select_dynamic_team_anchor(states, alive, state_members, fallback_member=fallback)
        except (KeyError, ValueError):
            return states[rep], None

    @staticmethod
    def _masked_scalar_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return masked_mean(values, mask.bool(), dim=tuple(range(values.ndim)))

    def _random_onehot(self, mask: torch.Tensor) -> torch.Tensor:
        logits = mask.float().masked_fill(~mask.bool(), -1e9)
        return F.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)

    def _eps_schedule(self, timestep: int, timesteps: int) -> float:
        eps = self._eps_greedy
        if self._eps_decay < 1.0 and timesteps and timesteps > 0:
            frac = min(max(float(timestep) / float(timesteps), 0.0), 1.0)
            eps = eps * (self._eps_decay**frac)
        return eps

    def _dec_exec_idx(self, a: torch.Tensor) -> torch.Tensor:
        return a.argmax(dim=-1).long()

    def _random_dec_action(self, mask: torch.Tensor) -> torch.Tensor:
        return self._random_onehot(mask)

    def _epsilon_greedy_dec(self, a: torch.Tensor, mask: torch.Tensor, timestep: int, timesteps: int) -> torch.Tensor:
        eps = self._eps_schedule(timestep, timesteps)
        rand = self._random_dec_action(mask)
        keep = (torch.rand(a.shape[0], device=a.device) >= eps).float().unsqueeze(-1)
        return keep * a + (1.0 - keep) * rand

    def act(self, observations, states, *, timestep, timesteps):
        return self._hier_act(observations, states, timestep=timestep, timesteps=timesteps)

    def record_transition(
        self,
        *,
        observations,
        states,
        actions,
        rewards,
        next_observations,
        next_states,
        terminated,
        truncated,
        infos,
        timestep,
        timesteps,
    ):
        return self._hier_record_transition(
            observations=observations,
            states=states,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            next_states=next_states,
            terminated=terminated,
            truncated=truncated,
            infos=infos,
            timestep=timestep,
            timesteps=timesteps,
        )

    def pre_interaction(self, *, timestep, timesteps):
        pass

    def _roll_value_frozen(self, team: str) -> bool:
        return team in getattr(self, "_roll_full_frozen_teams", set())

    def post_interaction(self, *, timestep, timesteps):
        update_train_takeover(self, timestep)
        update_roll_rotation(self, timestep)
        if self.training and timestep >= self._learning_starts and (timestep % self._update_every == 0):
            for _ in range(self._gradient_steps):
                self.update(timestep=timestep, timesteps=timesteps)

        super().post_interaction(timestep=timestep, timesteps=timesteps)

    def track_data(self, tag: str, value) -> None:
        if torch.is_tensor(value):
            self._tensor_track.setdefault(tag, []).append(value.detach().reshape(()))
        else:
            super().track_data(tag, value)

    def write_tracking_data(self, *, timestep: int, timesteps: int) -> None:
        if self._tensor_track:
            for tag, vals in self._tensor_track.items():
                for v in torch.stack(vals).cpu().tolist():
                    super().track_data(tag, v)
            self._tensor_track.clear()
        super().write_tracking_data(timestep=timestep, timesteps=timesteps)

    def update(self, *, timestep, timesteps):
        return self._hier_update(timestep=timestep, timesteps=timesteps)

    def _soft_update(self, target: nn.Module, source: nn.Module):
        with torch.no_grad():
            tps = list(target.parameters())
            sps = list(source.parameters())
            torch._foreach_mul_(tps, self._polyak)
            torch._foreach_add_(tps, sps, alpha=1.0 - self._polyak)

    def _init_hier(self, critic_hidden, activation, num_envs):
        from ..replay_buffer import DecisionReplayBuffer, HierReplayBuffer, HierReplayCommonStore

        self._q_low_input = str(self._cfg.get("q_low_input", "none")).lower()
        _rel_dim = int(self._cfg.get("q_low_rel_dim", 7))
        if self._q_low_input == "full":
            self._q_low_extra_dim = (self._M - 1) * (self._dec_repr_dim + 4 + _rel_dim)
        elif self._q_low_input == "selected":
            self._q_low_extra_dim = 4 + _rel_dim
        else:
            self._q_low_extra_dim = 0
        self._q_low_act_dim = 4 + self._q_low_extra_dim

        self._lower_critic_mode = str(self._cfg.get("lower_critic_mode", "per_agent")).lower()
        if self._lower_critic_mode not in ("per_agent", "team"):
            raise ValueError(f"lower_critic_mode={self._lower_critic_mode!r} 非法，需 per_agent|team")
        if self._lower_critic_mode == "team":
            if not self._hier_dual:
                raise ValueError("lower_critic_mode=team 需要 train_layer=both（双 critic 路径）")
            if self._q_low_input != "none":
                print(
                    f"[TAD][WARN] lower_critic_mode=team 忽略 q_low_input={self._q_low_input!r}"
                    f"（Q_low 输入=state+dec/lock ctx+joint ctrl）"
                )
                self._q_low_input = "none"
                self._q_low_extra_dim = 0
                self._q_low_act_dim = 4
            self._q_low_batched = False

        self._actor_group_of: dict = {}
        self._actor_groups: dict = {}
        for aid in self.possible_agents:
            layer = "dec" if self._is_dec[aid] else "ctrl"
            team = aid.split("/")[0].split("_")[0]
            gk = (team, layer) if self._actor_mode == "team" else ("__aid__", aid)
            self._actor_group_of[aid] = gk
            self._actor_groups.setdefault(gk, []).append(aid)
        self._actor_group_rep: dict = {gk: aids[0] for gk, aids in self._actor_groups.items()}

        def _space_dim(sp):
            if sp is None:
                return None
            if isinstance(sp, Discrete):
                return int(sp.n)
            shp = tuple(int(x) for x in sp.shape)
            return shp[0] if len(shp) == 1 else shp

        _spaces_dims = {
            uid: (
                _space_dim(self.observation_spaces[uid]),
                _space_dim(self.state_spaces[uid]),
                _space_dim(self.action_spaces[uid]),
            )
            for uid in self.possible_agents
        }
        self._architecture_fingerprint = {
            "format": "tad_hier_maddpg_central_actor",
            "centralized": False,
            "team_sizes": {
                t: len([a for a in ms if self._is_dec[a]]) or len([a for a in ms if not self._is_dec[a]])
                for t, ms in self._team_members.items()
            },
            "upper": mlp_backbone_fingerprint_spec(),
            "lower": mlp_backbone_fingerprint_spec(),
            "critic_backbone": "mlp",
            "critic_state": str(self._cfg.get("critic_state", "token")),
            "dec_head": "mlp",
            "actor_mode": self._actor_mode,
            "dec_exec_dim": int(getattr(self, "_dec_exec_dim", 0)),
            "dec_repr_dim": int(getattr(self, "_dec_repr_dim", 0)),
            "hier_dual": bool(getattr(self, "_hier_dual", False)),
            "target_dim": int(getattr(self, "_n_pool", self._cfg.get("n_targets_max", 0))),
            "q_low_input": self._q_low_input,
            "q_low_act_dim": self._q_low_act_dim,
            "lower_critic_mode": self._lower_critic_mode,
            "spaces_dims": _spaces_dims,
        }

        self._critic_state = str(self._cfg.get("critic_state", "token")).lower()
        self._critic_token_state = self._critic_state == "token"

        self._state_anchor_free = self._critic_token_state

        self.actors, self.target_actors = {}, {}
        for gk, aids in self._actor_groups.items():
            rep = aids[0]
            obs_dim = self._obs_dim_per_agent[rep]

            if self._is_dec[rep]:
                if self._high_ppo_mode:
                    self._rng_burn_dec_gumbel_pair(obs_dim, activation)
                    continue
                hidden = self._upper_actor_hidden
                in_dim = obs_dim
                out_dim = self._dec_exec_dim
                a = MaskedGumbelActor(in_dim, out_dim, hidden, activation, self._use_layer_norm).to(self.device)
                ta = MaskedGumbelActor(in_dim, out_dim, hidden, activation, self._use_layer_norm).to(self.device)
            else:
                hidden = self._lower_actor_hidden

                ctrl_in = obs_dim
                ctrl_out = 4
                a = ActorMLP(ctrl_in, ctrl_out, hidden, activation, self._use_layer_norm).to(self.device)
                ta = ActorMLP(ctrl_in, ctrl_out, hidden, activation, self._use_layer_norm).to(self.device)
            if self._zero_init:
                zero_initialize_module(a)
            ta.load_state_dict(a.state_dict())
            for p in ta.parameters():
                p.requires_grad = False
            for aid in aids:
                self.actors[aid] = a
                self.target_actors[aid] = ta

            self.checkpoint_modules[rep]["policy"] = a
            self.checkpoint_modules[rep]["target_policy"] = ta

        self.critics, self.target_critics = {}, {}
        if self._high_ppo_mode:
            for _team in self._team_members:
                self._rng_burn_high_critic_pair(critic_hidden, activation)
        else:
            for team in self._team_members:
                c = CriticMLP(
                    self._state_dim, self._critic_act_dim, critic_hidden, activation, self._use_layer_norm
                ).to(self.device)
                if self._zero_init:
                    zero_initialize_module(c)
                tc = CriticMLP(
                    self._state_dim, self._critic_act_dim, critic_hidden, activation, self._use_layer_norm
                ).to(self.device)
                tc.load_state_dict(c.state_dict())
                for p in tc.parameters():
                    p.requires_grad = False
                self.critics[team] = c
                self.target_critics[team] = tc

        self.actor_optimizers = {}
        for gk, aids in self._actor_groups.items():
            rep = aids[0]
            if self._high_ppo_mode and self._is_dec[rep]:
                continue
            opt = torch.optim.Adam(self.actors[rep].parameters(), lr=self._policy_lr)
            for aid in aids:
                self.actor_optimizers[aid] = opt
        self.critic_optimizers = (
            {team: torch.optim.Adam(self.critics[team].parameters(), lr=self._critic_lr) for team in self._team_members}
            if not self._high_ppo_mode
            else {}
        )

        for gk, aids in self._actor_groups.items():
            rep = aids[0]
            if self._high_ppo_mode and self._is_dec[rep]:
                continue
            self.checkpoint_modules[rep]["actor_optimizer"] = self.actor_optimizers[rep]
        if not self._high_ppo_mode:
            for team, members in self._team_members.items():
                rep = members[0]
                self.checkpoint_modules[rep]["critic"] = self.critics[team]
                self.checkpoint_modules[rep]["target_critic"] = self.target_critics[team]
                self.checkpoint_modules[rep]["critic_optimizer"] = self.critic_optimizers[team]
        register_architecture_fingerprint(self, self._architecture_fingerprint)
        register_training_progress(self)

        agent_obs_dims = {aid: self._obs_dim_per_agent[aid] for aid in self.possible_agents}
        _has_dec_layer = any(self._is_dec.values())
        _ctrl_ids = [aid for aid in self.possible_agents if not self._is_dec[aid]]
        _dec_ids_all = [aid for aid in self.possible_agents if self._is_dec[aid]]
        if not self._hier_dual:
            _obs_agent_ids = None
            _next_obs_agent_ids = _dec_ids_all + (_ctrl_ids if self._time_limit_bootstrap else [])

        else:
            _obs_agent_ids = _ctrl_ids
            _nobs = _ctrl_ids if self._time_limit_bootstrap else []
            _next_obs_agent_ids = _nobs

        _store_state = (not self._hier_dual) or (self._lower_critic_mode == "team")

        self.replays = {}

        if self._replay_steps and self._replay_steps > 0:
            _capacity = self._replay_steps * num_envs
        else:
            _capacity = self._replay_size
        self._replay_capacity = _capacity

        _store_device = "cpu" if self._replay_device == "cpu" else self.device
        _replay_dec_kwargs = dict(dec_dim=self._dec_repr_dim, ctrl_dim=4)
        self._replay_common = HierReplayCommonStore(
            _capacity,
            num_envs,
            self._total_act_dim,
            agent_obs_dims,
            self.device,
            store_device=_store_device,
            num_physical_agents=self._M,
            **_replay_dec_kwargs,
            q_low_team_ctx_dim=(self._M * (self._dec_repr_dim + 1 + 4) if self._lower_critic_mode == "team" else 0),
            obs_agent_ids=_obs_agent_ids,
            next_obs_agent_ids=_next_obs_agent_ids,
        )
        for team, members in self._team_members.items():
            low_aids = [aid for aid in members if not self._is_dec[aid]] if self._hier_dual else []

            self.replays[team] = HierReplayBuffer(
                _capacity,
                num_envs,
                self._state_dim,
                self._total_act_dim,
                agent_obs_dims,
                self.device,
                low_reward_agent_ids=low_aids,
                store_device=_store_device,
                common_store=self._replay_common,
                q_low_extra_agent_ids=low_aids,
                q_low_extra_dim=self._q_low_extra_dim,
                n_step=self._n_step,
                n_step_discount=self._gamma_control,
                priority_mix=self._replay_terminal_mix,
                ctrl_next_obs_successor=not self._time_limit_bootstrap,
                store_state=_store_state,
            )

        _total_b = self._replay_common.allocation_bytes() + sum(b.allocation_bytes() for b in self.replays.values())
        print(
            f"[MADDPG][R05-R07] replay schema：obs 列={len(self._replay_common.obs_per_agent)}/"
            f"{len(agent_obs_dims)}、next_obs 列={len(self._replay_common.next_obs_per_agent)}、"
            f"state 列={'on' if _store_state else 'off'}、"
            f"successor={'on' if not self._time_limit_bootstrap else 'off'} | "
            f"控制回放常驻 ≈ {_total_b / 2**30:.2f} GiB"
            f"（{'pinned CPU' if _store_device == 'cpu' else _store_device}）"
        )
        if self._n_step > 1:
            print(
                f"[MADDPG][A2 n-step] n_step={self._n_step}（控制步 TD：Q_low/队critic 窗口折叠，"
                f"γ={self._gamma_control}）"
            )
        print(
            f"[MADDPG][方案B] 高层段折扣 ρ_H=γ_H^(1/P)={self._rho_high:.8f}"
            f"（γ_H={self._gamma_decision:.6f}/决策间隔，视野 1/(1-γ_H)={1.0 / (1.0 - self._gamma_decision):.1f} 个决策间隔）"
            f" | [终局语义] time_limit_bootstrap={self._time_limit_bootstrap}"
            f"（false=自然超时切断 TD bootstrap）"
        )
        if self._replay_terminal_mix > 0.0:
            print(
                f"[MADDPG][终端优先回放] mix={self._replay_terminal_mix} thresh={self._replay_terminal_thresh}"
                f"（|team_reward|>={self._replay_terminal_thresh} 的行按 mix 混入采样批，IS 权重纠偏；"
                f"仅 flat 采样路径，option replay 不参与）"
            )

        if self._normalize:
            self._obs_preprocessor = {
                aid: RunningStandardScaler(
                    self._obs_dim_per_agent[aid],
                    epsilon=self._norm_eps,
                    clip_threshold=self._norm_clip,
                    device=self.device,
                ).to(self.device)
                for aid in self.possible_agents
            }
            self._state_preprocessor = {}
            for team, members in self._team_members.items():
                rep = members[0]

                self._state_preprocessor[rep] = (
                    self._identity_preprocessor
                    if self._critic_token_state
                    else RunningStandardScaler(
                        self._state_dim, epsilon=self._norm_eps, clip_threshold=self._norm_clip, device=self.device
                    ).to(self.device)
                )
            for team, members in self._team_members.items():
                rep = members[0]
                for aid in members:
                    self.checkpoint_modules[aid]["observation_preprocessor"] = self._obs_preprocessor[aid]
                    self.checkpoint_modules[aid]["state_preprocessor"] = self._state_preprocessor[rep]
        else:
            self._obs_preprocessor = {aid: self._identity_preprocessor for aid in self.possible_agents}
            self._state_preprocessor = {
                self._team_representative[t]: self._identity_preprocessor for t in self._team_members
            }
        self._current_onehot: dict = {}
        self._current_action: dict = {}
        training_cfg = self._cfg
        self.high_replays: dict = {}
        self._seg_state0: dict = {}
        self._seg_high_action0: dict = {}
        self._seg_return: dict = {}
        self._seg_discount: dict = {}
        self._seg_len: dict = {}
        self._seg_anchor0: dict = {}
        self._seg_obs0: dict = {}
        self.q_low: dict = {}
        self.target_q_low: dict = {}
        self.q_low_opts: dict = {}
        if self._hier_dual:
            from ..q_low import QLowCritic
            from ..q_low import BatchedQLowCritic, _QLowSliceCheckpoint

            self._q_low_batched = bool(training_cfg.get("q_low_batched", False))
            _low_hidden = list(training_cfg.get("lower_critic_hidden", [256, 256]))
            _low_lr = self._q_low_lr
            _store_device_decision = "cpu" if self._replay_device == "cpu" else self.device

            self._high_update_every = max(1, int(training_cfg.get("high_update_every", 1)))
            self._decision_replay_size = int(training_cfg.get("decision_replay_size", self._replay_size))
            _dec_aids = [a for a in self.possible_agents if self._is_dec[a]]
            self._dec_aids = _dec_aids
            _dec_obs_dims = {aid: self._obs_dim_per_agent[aid] for aid in _dec_aids}
            _high_dec_kwargs = dict(dec_dim=self._dec_repr_dim)

            for team in self._team_members:
                if not self._high_ppo_mode:
                    self.high_replays[team] = DecisionReplayBuffer(
                        self._decision_replay_size,
                        self._state_dim,
                        self._high_act_dim,
                        _dec_obs_dims,
                        self.device,
                        store_device=_store_device_decision,
                        num_physical_agents=self._M,
                        **_high_dec_kwargs,
                    )

            for aid in _dec_aids if not self._high_ppo_mode else []:
                self._seg_state0[aid] = torch.zeros(num_envs, self._state_dim, device=self.device)
                self._seg_high_action0[aid] = torch.zeros(num_envs, self._high_act_dim, device=self.device)
                self._seg_return[aid] = torch.zeros(num_envs, device=self.device)
                self._seg_discount[aid] = torch.ones(num_envs, device=self.device)
                self._seg_len[aid] = torch.zeros(num_envs, dtype=torch.long, device=self.device)

                self._seg_anchor0[aid] = torch.full((num_envs,), -1, dtype=torch.long, device=self.device)
                self._seg_obs0[aid] = {
                    da: torch.zeros(num_envs, self._obs_dim_per_agent[da], device=self.device) for da in _dec_aids
                }

            if self._lower_critic_mode == "team":
                self._q_low_batched = False
                _ctx_w = self._M * (self._dec_repr_dim + 1)
                self.q_low_team_net: dict = {}
                self.target_q_low_team_net: dict = {}
                self.q_low_team_opts: dict = {}
                for _team, _members in self._team_members.items():
                    if not any(not self._is_dec[a] for a in _members):
                        continue
                    q = CriticMLP(
                        self._state_dim + _ctx_w, self._M * 4, _low_hidden, activation, self._use_layer_norm
                    ).to(self.device)
                    tq = CriticMLP(
                        self._state_dim + _ctx_w, self._M * 4, _low_hidden, activation, self._use_layer_norm
                    ).to(self.device)
                    tq.load_state_dict(q.state_dict())
                    for p in tq.parameters():
                        p.requires_grad = False
                    self.q_low_team_net[_team] = q
                    self.target_q_low_team_net[_team] = tq
                    self.q_low_team_opts[_team] = torch.optim.Adam(q.parameters(), lr=_low_lr)
                    rep = _members[0]
                    self.checkpoint_modules[rep]["q_low"] = q
                    self.checkpoint_modules[rep]["target_q_low"] = tq
                    self.checkpoint_modules[rep]["q_low_optimizer"] = self.q_low_team_opts[_team]
            else:
                _team_ctrl = {t: [a for a in m if not self._is_dec[a]] for t, m in self._team_members.items()}
                _all_dims = {self._obs_dim_per_agent[a] for aids in _team_ctrl.values() for a in aids}
                _can_batch = self._q_low_batched and any(_team_ctrl.values()) and len(_all_dims) == 1

                if self._q_low_batched and not _can_batch and any(_team_ctrl.values()):
                    print(f"[TAD][B3b] q_low_batched=true 但 /ctrl obs_dim 不一致 {_all_dims}，回退 per-aid")
                self._q_low_batched = _can_batch
                self.q_low_batch: dict = {}
                self.target_q_low_batch: dict = {}
                self.q_low_opt: dict = {}
                if _can_batch:
                    _od = next(iter(_all_dims))
                    for _team, _ctrl in _team_ctrl.items():
                        if not _ctrl:
                            continue
                        qb = BatchedQLowCritic(
                            len(_ctrl),
                            _od,
                            self._q_low_act_dim,
                            _low_hidden,
                            activation,
                            self._use_layer_norm,
                            device=self.device,
                        )
                        tqb = BatchedQLowCritic(
                            len(_ctrl),
                            _od,
                            self._q_low_act_dim,
                            _low_hidden,
                            activation,
                            self._use_layer_norm,
                            device=self.device,
                        )
                        tqb.load_per_aid(qb.dump_per_aid())
                        for p in tqb.parameters():
                            p.requires_grad = False
                        self.q_low_batch[_team] = qb
                        self.target_q_low_batch[_team] = tqb
                        self.q_low_opt[_team] = torch.optim.Adam(qb.parameters(), lr=_low_lr)
                        for i, aid in enumerate(_ctrl):
                            self._q_low_idx[aid] = i
                            self._q_low_team[aid] = _team

                            self.checkpoint_modules[aid]["q_low"] = _QLowSliceCheckpoint(qb, i)
                            self.checkpoint_modules[aid]["target_q_low"] = _QLowSliceCheckpoint(tqb, i)
                else:
                    for _team, _ctrl in _team_ctrl.items():
                        for aid in _ctrl:
                            ql = QLowCritic(
                                self._obs_dim_per_agent[aid],
                                self._q_low_act_dim,
                                _low_hidden,
                                activation,
                                self._use_layer_norm,
                            ).to(self.device)
                            tql = QLowCritic(
                                self._obs_dim_per_agent[aid],
                                self._q_low_act_dim,
                                _low_hidden,
                                activation,
                                self._use_layer_norm,
                            ).to(self.device)
                            tql.load_state_dict(ql.state_dict())
                            for p in tql.parameters():
                                p.requires_grad = False
                            self.q_low[aid] = ql
                            self.target_q_low[aid] = tql
                            self.q_low_opts[aid] = torch.optim.Adam(ql.parameters(), lr=_low_lr)
                            self.checkpoint_modules[aid]["q_low"] = ql
                            self.checkpoint_modules[aid]["target_q_low"] = tql
                            self.checkpoint_modules[aid]["q_low_optimizer"] = self.q_low_opts[aid]
        self._print_network_architecture()

    def _rng_burn_dec_gumbel_pair(self, in_dim: int, activation: str) -> None:
        hidden = self._upper_actor_hidden
        out_dim = self._dec_exec_dim
        MaskedGumbelActor(in_dim, out_dim, hidden, activation, self._use_layer_norm)
        MaskedGumbelActor(in_dim, out_dim, hidden, activation, self._use_layer_norm)

    def _rng_burn_high_critic_pair(self, critic_hidden, activation: str) -> None:
        CriticMLP(self._state_dim, self._critic_act_dim, critic_hidden, activation, self._use_layer_norm)
        CriticMLP(self._state_dim, self._critic_act_dim, critic_hidden, activation, self._use_layer_norm)

    def _print_network_architecture(self) -> None:
        sep = "=" * 66
        print(f"\n[TAD][net] {sep}")
        print("[TAD][net] 网络架构（每类打印一次；'agent 模块' = 共享该架构的学习 agent / 队）")

        def _emit(tag: str, module, aids, *, in_dim, out_dim, extra: str = "") -> None:
            params = sum(p.numel() for p in module.parameters())
            dim_str = f"input={in_dim}  output={out_dim}  " if in_dim is not None else ""
            print(f"[TAD][net] -- {tag} [{type(module).__name__}] agent 模块={aids} {extra}")
            print(f"[TAD][net]      {dim_str}params={params:,}")
            for line in repr(module).splitlines():
                print(f"[TAD][net]      {line}")

        dec_aids = [aid for aid in self.possible_agents if self._is_dec.get(aid)]
        ctrl_aids = [aid for aid in self.possible_agents if not self._is_dec.get(aid)]
        if dec_aids and not self._high_ppo_mode:
            rep = dec_aids[0]
            _emit(
                "/dec 上层 actor（选目标）",
                self.actors[rep],
                dec_aids,
                in_dim=self._obs_dim_per_agent[rep],
                out_dim=self._nm,
            )
        elif dec_aids:
            print("[TAD][net] -- /dec 上层 actor：高层 PPO 算法（Categorical，由子类构建，见其启动打印）")
        else:
            print("[TAD][net] -- /dec 上层 actor：本 run 无学习 agent（未训上层=最近目标启发式）")
        if ctrl_aids:
            rep = ctrl_aids[0]
            _emit(
                "/ctrl 下层 actor（连续控制 [T,p,q,r]）",
                self.actors[rep],
                ctrl_aids,
                in_dim=self._obs_dim_per_agent[rep],
                out_dim=4,
            )
        else:
            print("[TAD][net] -- /ctrl 下层 actor：本 run 无学习 agent（未训下层=级联脚本控制）")

        if self.critics:
            team = next(iter(self.critics))
            _emit(
                "critic（队 Q，每队一个结构相同）",
                self.critics[team],
                list(self._team_members),
                in_dim=self._state_dim + self._critic_act_dim,
                out_dim=1,
                extra=f"(=state({self._state_dim})+joint({self._critic_act_dim}))",
            )
        elif self._team_members and self._high_ppo_mode:
            print("[TAD][net] -- critic（队 Q）：高层 PPO 算法不构建（V_dec 由子类构建，见其启动打印）")

        if getattr(self, "q_low_team_net", None):
            team = next(iter(self.q_low_team_net))
            _ctx_w = self._M * (self._dec_repr_dim + 1)
            _emit(
                "q_low 低层 critic（队级共享，lower_critic_mode=team）",
                self.q_low_team_net[team],
                list(self.q_low_team_net),
                in_dim=self._state_dim + _ctx_w + self._M * 4,
                out_dim=1,
                extra=f"(=state({self._state_dim})+ctx({_ctx_w})+joint_ctrl({self._M * 4}))",
            )

        print(f"[TAD][net] {sep}\n")

    def _ctrl_actor_forward(self, aid, obs_full_n, *, target=False):
        actor = self.target_actors[aid] if target else self.actors[aid]
        return actor(obs_full_n)

    def _dec_group_actions(self, gk, obs_n_by_aid, mask_by_aid, *, target=False, deterministic=False):
        aids = self._actor_groups[gk]
        masks_ok = mask_by_aid is None or all(mask_by_aid.get(a) is not None for a in aids)
        if self._batched_fwd and len(aids) > 1 and masks_ok:
            actor = self.target_actors[aids[0]] if target else self.actors[aids[0]]
            obs_stk = torch.stack([obs_n_by_aid[a] for a in aids], dim=0)
            mask_stk = None if mask_by_aid is None else torch.stack([mask_by_aid[a] for a in aids], dim=0)
            if deterministic or target:
                oh_stk = actor.argmax_onehot(obs_stk, mask_stk)
            else:
                oh_stk = actor(obs_stk, mask_stk, tau=self._gumbel_tau, hard=True)
            return {a: oh_stk[i] for i, a in enumerate(aids)}
        out = {}
        for a in aids:
            actor = self.target_actors[a] if target else self.actors[a]
            mask = mask_by_aid.get(a) if mask_by_aid is not None else None
            if deterministic or target:
                out[a] = actor.argmax_onehot(obs_n_by_aid[a], mask)
            else:
                out[a] = actor(obs_n_by_aid[a], mask, tau=self._gumbel_tau, hard=True)
        return out

    def _ctrl_group_actions(self, gk, obs_n_by_aid, obs_raw_by_aid=None, *, target=False):
        aids = self._actor_groups[gk]
        route_src = obs_raw_by_aid or obs_n_by_aid
        if self._batched_fwd and len(aids) > 1:
            actor = self.target_actors[aids[0]] if target else self.actors[aids[0]]
            out_stk = actor(torch.stack([obs_n_by_aid[a] for a in aids], dim=0))
            return {a: out_stk[i] for i, a in enumerate(aids)}
        return {a: (self.target_actors[a] if target else self.actors[a])(obs_n_by_aid[a]) for a in aids}

    def _eval_row_mask(self, ref: torch.Tensor) -> torch.Tensor | None:
        m = getattr(self, "_eval_env_mask", None)
        if m is None:
            return None
        m = m.to(device=ref.device, dtype=torch.bool).view(-1)
        return m if m.numel() == ref.shape[0] else None

    def _eval_target_dec(self) -> bool:
        return str(getattr(self, "_test_actor", "target")).lower() == "target"

    def _dec_eval_rows(self, oh_map: dict, obs_n: dict, masks: dict, eval_mask: torch.Tensor) -> dict:
        idx = eval_mask.nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            return oh_map
        want_target = self._eval_target_dec()
        out = {}
        for gk, members in self._actor_groups.items():
            if not self._is_dec[members[0]] or members[0] not in oh_map:
                continue
            for a in members:
                mask_e = None if masks.get(a) is None else masks[a][idx]
                actor = self.target_actors[a] if (want_target and a in self.target_actors) else self.actors[a]
                oh_e = actor.argmax_onehot(obs_n[a][idx], mask_e)
                row = oh_map[a].clone()
                row[idx] = oh_e.view(-1, self._nm)
                out[a] = row
        return out

    def _ctrl_eval_rows(self, a_map: dict, obs_n_by: dict, raw_by: dict, eval_mask: torch.Tensor) -> dict:
        if not self._eval_target_dec():
            return a_map
        idx = eval_mask.nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            return a_map
        out = dict(a_map)
        for gk, members in self._actor_groups.items():
            if self._is_dec[members[0]] or members[0] not in a_map:
                continue
            ev = self._ctrl_group_actions(
                gk, {a: obs_n_by[a][idx] for a in members}, {a: raw_by[a][idx] for a in members}, target=True
            )
            for a in members:
                row = out[a].clone()
                row[idx] = ev[a]
                out[a] = row
        return out

    def _commit_decisions_and_rebuild_ctrl_obs(self, observations, actions) -> dict:
        return stage_decisions_rebuild_ctrl(self, observations, actions)

    def _hier_act(self, observations, states, *, timestep, timesteps):
        self._total_timesteps = timesteps
        actions = {}
        noise_scale = 1.0
        if self.training and self._noise_scheduler and timesteps > 0:
            if self._noise_halflife > 0:
                noise_scale = self._noise_floor + (1.0 - self._noise_floor) * (
                    2.0 ** (-float(timestep) / self._noise_halflife)
                )
            else:
                noise_scale = max(self._noise_floor, 1.0 - timestep / timesteps)
        warmup_steps = getattr(self, "_warmup_steps", 0)
        warmup_env = getattr(self, "_warmup_env", None)
        warmup_mask = None
        if warmup_steps > 0 and warmup_env is not None:
            warmup_mask = warmup_env.episode_length_buf < warmup_steps

        self._current_ctrl_obs_raw: dict = {}
        with torch.no_grad():
            dec_aids = [a for a in self.possible_agents if self._is_dec[a]]
            dec_obs_n = {a: self._obs_preprocessor[a](observations[a]) for a in dec_aids}
            dec_mask = {a: observations[a][:, -self._mask_tail_dim :] for a in dec_aids}
            for gk, members in self._actor_groups.items():
                if not self._is_dec[members[0]]:
                    continue
                if not self.training:
                    oh_map = self._dec_group_actions(gk, dec_obs_n, dec_mask, deterministic=True)
                elif timestep < self._random_timesteps:
                    oh_map = {a: self._random_dec_action(dec_mask[a]) for a in members}
                else:
                    oh_map = self._dec_group_actions(gk, dec_obs_n, dec_mask)
                    oh_map = {a: self._epsilon_greedy_dec(oh_map[a], dec_mask[a], timestep, timesteps) for a in members}

                _ev_m = self._eval_row_mask(dec_obs_n[members[0]])
                if _ev_m is not None:
                    oh_map = self._dec_eval_rows(oh_map, dec_obs_n, dec_mask, _ev_m)
                for a in members:
                    self._current_onehot[a] = oh_map[a]
                    dec_idx = self._dec_exec_idx(oh_map[a])
                    actions[a] = dec_idx.unsqueeze(-1).long()

            ctrl_raw = self._commit_decisions_and_rebuild_ctrl_obs(observations, actions)
            self._current_ctrl_obs_raw = dict(ctrl_raw)
            for gk, members in self._actor_groups.items():
                if self._is_dec[members[0]]:
                    continue
                raw_by = {a: ctrl_raw[a] for a in members}
                obs_n_by = {a: self._obs_preprocessor[a](raw_by[a]) for a in members}
                a_map = self._ctrl_group_actions(gk, obs_n_by, raw_by)

                _ev_m = self._eval_row_mask(obs_n_by[members[0]])
                if _ev_m is not None:
                    a_map = self._ctrl_eval_rows(a_map, obs_n_by, raw_by, _ev_m)
                    _ev_col = _ev_m.unsqueeze(-1)
                else:
                    _ev_col = None
                for aid in members:
                    du = aid.split("/")[0]
                    a = a_map[aid]
                    if self.training and timestep < self._random_timesteps:
                        _rnd = torch.zeros_like(a).uniform_(-1.0, 1.0)
                        a = _rnd if _ev_col is None else torch.where(_ev_col, a, _rnd)
                    elif self.training:
                        _noisy = torch.clamp(a + torch.randn_like(a) * self._noise_std * noise_scale, -1.0, 1.0)
                        a = _noisy if _ev_col is None else torch.where(_ev_col, a, _noisy)
                    if warmup_mask is not None and bool(warmup_mask.any()):
                        ctrl_fn = getattr(warmup_env, "compute_warmup_stabilize_action", None)
                        ctrl = ctrl_fn(du) if ctrl_fn is not None else torch.zeros_like(a)
                        a = torch.where(warmup_mask.unsqueeze(-1), ctrl, a)
                    self._current_action[aid] = a
                    actions[aid] = a
        return actions, {}

    def _high_joint_action_from_env(self, env) -> torch.Tensor:
        cache = "_eff_onehot_dec"
        return torch.cat([getattr(env, cache)[uid] for uid in self._drone_uids], dim=-1).detach()

    def _high_slot(self, aid: str) -> tuple[int, int]:
        du = aid.split("/")[0]
        start = self._drone_index[du] * self._nm
        return start, self._nm

    def _dec_obs_snapshot(self, obs_dict: dict, mask: torch.Tensor | None = None) -> dict:
        out = {}
        for aid in self._dec_aids:
            val = obs_dict[aid]
            out[aid] = val[mask] if mask is not None else val.detach().clone()
        return out

    def _finalize_decision_segment(
        self,
        aid: str,
        team: str,
        rows: torch.Tensor,
        next_state: torch.Tensor,
        next_high_action: torch.Tensor,
        next_dec_obs: dict,
        episode_done: torch.Tensor,
        next_anchor_idx: torch.Tensor | None = None,
        next_alive_per_agent: dict | None = None,
    ):
        if team not in self.high_replays:
            return
        terminated_rows = episode_done[rows].bool()
        if next_anchor_idx is not None:
            frame_switch = self._seg_anchor0[aid][rows] != next_anchor_idx[rows]
            terminated_rows = terminated_rows | (frame_switch & (self._seg_anchor0[aid][rows] >= 0))
        self.high_replays[team].add(
            state=self._seg_state0[aid][rows],
            next_state=next_state[rows],
            high_joint_action=self._seg_high_action0[aid][rows],
            next_high_joint_action=next_high_action[rows],
            reward=self._seg_return[aid][rows],
            discount=self._seg_discount[aid][rows],
            terminated=terminated_rows,
            length=self._seg_len[aid][rows],
            agent_mask=F.one_hot(
                torch.full((int(rows.sum().item()),), self._dec_aids.index(aid), dtype=torch.long, device=self.device),
                num_classes=len(self._dec_aids),
            ).to(self._seg_return[aid].dtype),
            obs_per_agent={k: v[rows] for k, v in self._seg_obs0[aid].items()},
            next_obs_per_agent={k: v[rows] for k, v in next_dec_obs.items()},
            next_alive_per_agent=(
                {k: v[rows].bool().view(-1) for k, v in next_alive_per_agent.items()}
                if next_alive_per_agent is not None
                else None
            ),
        )

    def _step_decision_segment(
        self,
        aid: str,
        team: str,
        boundary: torch.Tensor,
        episode_done: torch.Tensor,
        was_alive: torch.Tensor,
        still_alive: torch.Tensor,
        team_reward: torch.Tensor,
        state_t: torch.Tensor,
        next_state_t: torch.Tensor,
        high_action_t: torch.Tensor,
        observations: dict,
        next_observations: dict,
        anchor_idx_t: torch.Tensor | None = None,
        anchor_idx_t1: torch.Tensor | None = None,
        td_episode_done: torch.Tensor | None = None,
        was_alive_pa: dict | None = None,
        still_alive_pa: dict | None = None,
    ) -> None:
        boundary = boundary.view(-1)
        episode_done = episode_done.view(-1)
        if td_episode_done is None:
            td_episode_done = episode_done
        else:
            td_episode_done = td_episode_done.view(-1)
        was_alive = was_alive.bool().view(-1)
        still_alive = still_alive.bool().view(-1)
        newly_dead = was_alive & ~still_alive

        seg_len = self._seg_len[aid]

        finalize_old = boundary.bool() & (seg_len > 0)
        start_new = was_alive & (boundary.bool() | (seg_len == 0))
        final_done = (episode_done.bool() | newly_dead) & ((seg_len + was_alive.to(seg_len.dtype)) > 0)
        _has_fin_old, _has_start_new, _has_final_done = torch.stack(
            [finalize_old.any(), start_new.any(), final_done.any()]
        ).tolist()
        if _has_fin_old:
            self._finalize_decision_segment(
                aid,
                team,
                finalize_old,
                state_t,
                high_action_t,
                self._dec_obs_snapshot(observations),
                torch.zeros_like(episode_done, dtype=torch.bool),
                next_anchor_idx=anchor_idx_t,
                next_alive_per_agent=was_alive_pa,
            )
        if _has_start_new:
            self._seg_state0[aid][start_new] = state_t[start_new].detach()
            self._seg_high_action0[aid][start_new] = high_action_t[start_new].detach()
            self._seg_return[aid][start_new] = 0.0
            self._seg_discount[aid][start_new] = 1.0
            self._seg_len[aid][start_new] = 0

            self._seg_anchor0[aid][start_new] = (
                anchor_idx_t[start_new]
                if anchor_idx_t is not None
                else torch.full_like(self._seg_anchor0[aid][start_new], -1)
            )
            for da in self._dec_aids:
                self._seg_obs0[aid][da][start_new] = observations[da][start_new].detach()

        self._seg_return[aid][was_alive] += self._seg_discount[aid][was_alive] * team_reward[was_alive]
        self._seg_discount[aid][was_alive] *= self._rho_high
        self._seg_len[aid][was_alive] += 1

        if _has_final_done:
            self._finalize_decision_segment(
                aid,
                team,
                final_done,
                next_state_t,
                high_action_t,
                self._dec_obs_snapshot(next_observations),
                td_episode_done | newly_dead,
                next_anchor_idx=anchor_idx_t1,
                next_alive_per_agent=still_alive_pa,
            )
            self._seg_return[aid][final_done] = 0.0
            self._seg_discount[aid][final_done] = 1.0
            self._seg_len[aid][final_done] = 0

    def _hier_record_transition(
        self,
        *,
        observations,
        states,
        actions,
        rewards,
        next_observations,
        next_states,
        terminated,
        truncated,
        infos,
        timestep,
        timesteps,
    ):
        _term_ref = next(iter(terminated.values())) if isinstance(terminated, dict) else terminated
        _eval_m = self._eval_row_mask(_term_ref) if isinstance(terminated, dict) else None
        if _eval_m is not None:

            def _row_mask(v: torch.Tensor) -> torch.Tensor:
                return _eval_m.view([-1] + [1] * (v.dim() - 1))

            super().record_transition(
                observations=observations,
                states=states,
                actions=actions,
                rewards={k: torch.where(_row_mask(v), torch.zeros_like(v), v) for k, v in rewards.items()},
                next_observations=next_observations,
                next_states=next_states,
                terminated={k: v & ~_row_mask(v) for k, v in terminated.items()},
                truncated={k: v & ~_row_mask(v) for k, v in truncated.items()},
                infos=infos,
                timestep=timestep,
                timesteps=timesteps,
            )
        else:
            super().record_transition(
                observations=observations,
                states=states,
                actions=actions,
                rewards=rewards,
                next_observations=next_observations,
                next_states=next_states,
                terminated=terminated,
                truncated=truncated,
                infos=infos,
                timestep=timestep,
                timesteps=timesteps,
            )
        if not self.training:
            return
        env = getattr(self, "_hier_env", None)
        uid0 = self.possible_agents[0]
        term_t = terminated[uid0] if isinstance(terminated, dict) else terminated
        trunc_t = truncated[uid0] if isinstance(truncated, dict) else truncated
        term = term_t | trunc_t
        term_f = term.float().view(-1)

        td_done_t = (term_t if self._time_limit_bootstrap else term).bool()

        temporal = env.temporal_transition_info()
        joint_action = temporal["executed_joint_action"]
        high_action_t = (
            self._high_joint_action_from_env(env)
            if (self._hier_dual and env is not None and not self._high_ppo_mode)
            else None
        )

        trunc_only = (trunc_t & ~term_t).bool().view(-1)
        fin = temporal.get("truncation_final")
        next_observations_eff = next_observations
        next_states_eff = next_states
        if fin is not None and bool(trunc_only.any()):
            _fin_mask, _fin_obs, _fin_states = fin

            def _blend(cur: dict, fin_dict: dict) -> dict:
                out = {}
                for k, x in cur.items():
                    fx = fin_dict.get(k)
                    if fx is None or fx.shape != x.shape:
                        continue
                    bm = trunc_only.view([-1] + [1] * (x.dim() - 1))
                    out[k] = torch.where(bm, fx.to(dtype=x.dtype, device=x.device), x)
                return out

            next_observations_eff = _blend(next_observations, _fin_obs)
            next_states_eff = _blend(next_states, _fin_states)
        decision_boundaries = {
            aid: (
                temporal["decision_boundary"][aid.split("/")[0]]
                if self._is_dec[aid]
                else torch.zeros_like(term, dtype=torch.bool)
            )
            for aid in self.possible_agents
        }
        obs_pa = {
            aid: (
                self._current_ctrl_obs_raw[aid]
                if aid in getattr(self, "_current_ctrl_obs_raw", {})
                else observations[aid]
            )
            for aid in self.possible_agents
        }
        nobs_pa = {aid: next_observations_eff[aid] for aid in self.possible_agents}
        if self._normalize:
            for aid in self.possible_agents:
                _o = obs_pa[aid]
                self._obs_preprocessor[aid](_o if _eval_m is None else _o[~_eval_m], train=True)
        states_is_dict = isinstance(states, dict)
        next_states_is_dict = isinstance(next_states, dict)
        alive_count = infos.get("team_alive_count") if isinstance(infos, dict) else None

        term_info = infos.get("agent_term_reward") if isinstance(infos, dict) else None
        was_alive_info = self._alive_info(infos, "agent_was_alive")
        still_alive_info = self._alive_info(infos, "agent_still_alive")
        active_per_agent = self._active_per_agent(infos, "agent_was_alive", term)
        still_active_per_agent = self._active_per_agent(infos, "agent_still_alive", term)

        if _eval_m is not None:
            _tr = ~_eval_m
            learn_active_per_agent = {aid: (m & _tr) for aid, m in active_per_agent.items()}
            learn_still_active_per_agent = {aid: (m & _tr) for aid, m in still_active_per_agent.items()}
        else:
            learn_active_per_agent = active_per_agent
            learn_still_active_per_agent = still_active_per_agent

        self._replay_common.add(
            joint_action=joint_action,
            terminated=term_f,
            obs_per_agent=obs_pa,
            next_obs_per_agent=nobs_pa,
            decision_boundary_per_agent=decision_boundaries,
            active_per_agent=learn_active_per_agent,
            boot_terminated=td_done_t.float().view(-1),
            q_low_team_ctx=temporal.get("q_low_team_ctx"),
        )
        for team, members in self._team_members.items():
            team_reward = rewards[members[0]].clone()
            for m in members[1:]:
                team_reward = team_reward + rewards[m]
            if self._team_reward_aggr == "mean":
                if alive_count is not None and team in alive_count:
                    denom = alive_count[team].to(team_reward.device).float().view_as(team_reward).clamp(min=1.0)
                else:
                    denom = self._team_drone_count.get(team, max(len(members), 1))

                team_term = self._team_term_sum(term_info, members, team_reward)
                if team_term is not None:
                    n_nominal = float(self._team_drone_count.get(team, max(len(members), 1)))
                    team_reward = (team_reward - team_term) / denom + team_term / n_nominal
                else:
                    team_reward = team_reward / denom
            team_reward = team_reward.view(-1)
            rep = self._team_representative[team]
            state_t, _anchor_idx_t = self._team_state_anchor(states, team, was_alive_info)

            _need_next_state = self.replays[team].state is not None or (self._hier_dual and not self._high_ppo_mode)
            next_state_t = (
                self._team_state_anchor(next_states_eff, team, still_alive_info)[0] if _need_next_state else None
            )

            team_active = torch.stack([learn_active_per_agent[m] for m in members], dim=0).any(dim=0)

            _anchor_idx_t1 = self._team_state_anchor(next_states, team, still_alive_info)[1]
            anchor_switch_t = (
                torch.zeros_like(team_active, dtype=torch.bool)
                if self._state_anchor_free or _anchor_idx_t is None or _anchor_idx_t1 is None
                else (_anchor_idx_t != _anchor_idx_t1)
            )
            low_reward_per_agent = None
            if self._hier_dual and env is not None and self._high_ppo_mode:
                low_reward_per_agent = {aid: team_reward for aid in members if not self._is_dec.get(aid, False)}
            elif self._hier_dual and env is not None:
                for aid in members:
                    if not self._is_dec.get(aid, False):
                        continue
                    uid = aid.split("/")[0]

                    boundary = env._decision_boundary[uid]

                    self._step_decision_segment(
                        aid,
                        team,
                        boundary,
                        term,
                        learn_active_per_agent[aid],
                        learn_still_active_per_agent[aid],
                        team_reward,
                        state_t,
                        next_state_t,
                        high_action_t,
                        observations,
                        next_observations_eff,
                        anchor_idx_t=_anchor_idx_t,
                        anchor_idx_t1=_anchor_idx_t1,
                        td_episode_done=td_done_t,
                        was_alive_pa=learn_active_per_agent,
                        still_alive_pa=learn_still_active_per_agent,
                    )
                low_reward_per_agent = {}
                for aid in members:
                    if self._is_dec.get(aid, False):
                        continue
                    low_reward_per_agent[aid] = team_reward
            if timestep % self._telemetry_interval == 0:
                self.track_data(
                    f"Reward / shaped (team_{team})",
                    team_reward.float().mean() if _eval_m is None else team_reward.float()[~_eval_m].mean(),
                )
            if self._normalize:
                self._state_preprocessor[rep](state_t if _eval_m is None else state_t[~_eval_m], train=True)

            q_low_extra_pa = None
            if self._q_low_extra_dim > 0 and env is not None:
                _env_extra = getattr(env, "_q_low_extra", {})
                if _env_extra:
                    q_low_extra_pa = {
                        aid: _env_extra[aid.split("/")[0]]
                        for aid in members
                        if (not self._is_dec[aid] and aid.split("/")[0] in _env_extra)
                    }
            self.replays[team].add(
                state=state_t,
                next_state=next_state_t,
                reward=team_reward,
                low_reward_per_agent=low_reward_per_agent,
                team_active=team_active,
                q_low_extra_per_agent=q_low_extra_pa,
                anchor_switch=anchor_switch_t,
                priority=(
                    None
                    if self._replay_terminal_mix <= 0.0
                    else ((team_reward.abs() >= self._replay_terminal_thresh) & team_active)
                ),
            )

    def _update_high_from_decision_replay(self, team: str, members: list[str]) -> None:
        hbuf = self.high_replays.get(team)
        if hbuf is None or len(hbuf) < self._batch_size:
            return
        batch = hbuf.sample(self._batch_size)
        s = batch["state"]
        ns = batch["next_state"]
        ha = batch["high_joint_action"]
        nha = batch["next_high_joint_action"]
        r = batch["reward"]
        discount = batch["discount"]
        done = batch["terminated"]
        obs_pa = batch["obs_per_agent"]
        nobs_pa = batch["next_obs_per_agent"]
        agent_mask = batch.get("agent_mask", None)
        next_alive_pa = batch.get("next_alive_per_agent", {})
        rep = members[0]
        if self._normalize:
            s = self._state_preprocessor[rep](s)
            ns = self._state_preprocessor[rep](ns)
        with torch.no_grad():
            next_ha = nha.clone()

            for gk, gaids in self._actor_groups.items():
                if not self._is_dec[gaids[0]]:
                    continue
                gaids = [a for a in gaids if a in nobs_pa]
                if not gaids:
                    continue
                if self._batched_fwd and len(gaids) > 1:
                    nobs_stk = torch.stack([self._obs_preprocessor[a](nobs_pa[a]) for a in gaids], dim=0)
                    mask_stk = torch.stack([nobs_pa[a][:, -self._mask_tail_dim :] for a in gaids], dim=0)
                    oh_stk = self.target_actors[gaids[0]].argmax_onehot(nobs_stk, mask_stk)
                    for i, aid in enumerate(gaids):
                        start, width = self._high_slot(aid)
                        alive_i = next_alive_pa.get(aid)
                        dec_oh = oh_stk[i]
                        if alive_i is None:
                            next_ha[:, start : start + width] = dec_oh
                        else:
                            next_ha[:, start : start + width] = torch.where(
                                alive_i.bool().view(-1).unsqueeze(-1), dec_oh, next_ha[:, start : start + width]
                            )
                    continue
                for aid in gaids:
                    raw = nobs_pa[aid]
                    nobs_i = self._obs_preprocessor[aid](raw)
                    mask = raw[:, -self._mask_tail_dim :]
                    dec_oh = self.target_actors[aid].argmax_onehot(nobs_i, mask)
                    start, width = self._high_slot(aid)
                    alive_i = next_alive_pa.get(aid)
                    if alive_i is None:
                        next_ha[:, start : start + width] = dec_oh
                    else:
                        next_ha[:, start : start + width] = torch.where(
                            alive_i.bool().view(-1).unsqueeze(-1), dec_oh, next_ha[:, start : start + width]
                        )
            target_q = self.target_critics[team](ns, next_ha).squeeze(-1)
            target = r + discount * (1.0 - done) * target_q
        current_q = self.critics[team](s, ha).squeeze(-1)
        critic_loss = ((current_q - target) ** 2).mean()
        if not self._roll_value_frozen(team):
            self.critic_optimizers[team].zero_grad()
            critic_loss.backward()
            if self._grad_norm_clip > 0:
                nn.utils.clip_grad_norm_(self.critics[team].parameters(), self._grad_norm_clip)
            self.critic_optimizers[team].step()

        frozen_agents = getattr(self, "frozen_agents", set())
        seen_groups: set = set()
        for aid in members:
            if not self._is_dec.get(aid, False):
                continue
            gk = self._actor_group_of[aid]
            if gk in seen_groups:
                continue
            seen_groups.add(gk)
            group_aids = [a for a in self._actor_groups[gk] if self._is_dec.get(a, False)]
            if not group_aids or any(a in frozen_agents for a in group_aids):
                continue
            rep_aid = group_aids[0]
            opt = self.actor_optimizers[rep_aid]
            opt.zero_grad()

            if self._batched_fwd and len(group_aids) > 1:
                m_b = len(group_aids)
                B = s.shape[0]
                obs_stk = torch.stack([self._obs_preprocessor[a](obs_pa[a]) for a in group_aids], dim=0)
                mask_stk = torch.stack([obs_pa[a][:, -self._mask_tail_dim :] for a in group_aids], dim=0)
                acts_stk = self.actors[rep_aid](obs_stk, mask_stk, tau=self._gumbel_tau, hard=True)
                ja_stk = ha.unsqueeze(0).expand(m_b, B, ha.shape[-1]).clone()
                for i, a_id in enumerate(group_aids):
                    start, width = self._high_slot(a_id)
                    ja_stk[i, :, start : start + width] = acts_stk[i]
                s_stk = s.unsqueeze(0).expand(m_b, B, s.shape[-1])
                q_stk = (
                    self.critics[team](s_stk.reshape(-1, s.shape[-1]), ja_stk.reshape(-1, ja_stk.shape[-1]))
                    .squeeze(-1)
                    .view(m_b, B)
                )
                if agent_mask is not None:
                    gate_stk = torch.stack([agent_mask[:, self._dec_aids.index(a)] > 0.5 for a in group_aids], dim=0)
                else:
                    gate_stk = torch.ones_like(q_stk, dtype=torch.bool)
                denom = gate_stk.sum(dim=1).clamp(min=1)
                total_actor_loss = -((q_stk * gate_stk).sum(dim=1) / denom).sum() / m_b
                total_actor_loss.backward()
            else:
                total_actor_loss = 0.0
                for a_id in group_aids:
                    obs_i = self._obs_preprocessor[a_id](obs_pa[a_id])
                    mask = obs_pa[a_id][:, -self._mask_tail_dim :]
                    a_i = self.actors[a_id](obs_i, mask, tau=self._gumbel_tau, hard=True)
                    ja_grad = ha.clone()
                    start, width = self._high_slot(a_id)
                    ja_grad[:, start : start + width] = a_i
                    q = self.critics[team](s, ja_grad).squeeze(-1)
                    gate = (
                        (agent_mask[:, self._dec_aids.index(a_id)] > 0.5)
                        if agent_mask is not None
                        else torch.ones_like(q, dtype=torch.bool)
                    )
                    if not bool(gate.any()):
                        continue
                    actor_loss = -q[gate].mean() / len(group_aids)
                    actor_loss.backward()
                    total_actor_loss = total_actor_loss + actor_loss.detach().double()
            if self._grad_norm_clip > 0:
                nn.utils.clip_grad_norm_(self.actors[rep_aid].parameters(), self._grad_norm_clip)
            opt.step()
            self.track_data(f"Loss / High actor loss (team_{team}_{rep_aid})", total_actor_loss)
        self._soft_update(self.target_critics[team], self.critics[team])
        seen_target_groups: set = set()
        for aid in members:
            if not self._is_dec.get(aid, False) or aid in frozen_agents:
                continue
            gk = self._actor_group_of[aid]
            if gk in seen_target_groups:
                continue
            seen_target_groups.add(gk)
            rep_target = self._actor_group_rep[gk]
            self._soft_update(self.target_actors[rep_target], self.actors[rep_target])
        self.track_data(f"Loss / High critic loss (team_{team})", critic_loss)
        self.track_data(f"Q / high mean (team_{team})", current_q.mean())

    def _update_low_from_control_batch(self, team: str, members: list[str], batch: dict) -> None:
        if self._lower_critic_mode == "team":
            self._update_low_team(team, members, batch)
            return
        ja = batch["joint_action"]
        done = batch.get("boot_terminated", batch["terminated"])
        _g_ns = batch.get("nstep_gamma")
        g_eff = _g_ns if _g_ns is not None else self._gamma_control
        obs_pa = batch["obs_per_agent"]
        nobs_pa = batch["next_obs_per_agent"]
        low_rewards = batch["low_reward_per_agent"]
        active_pa = batch.get("active_per_agent", {})
        ctrl_aids = self._q_low_ctrl_aids(members)
        frozen_agents = getattr(self, "frozen_agents", set())
        extra_pa = batch.get("q_low_extra_per_agent", {})

        extra_pa_next = batch.get("next_q_low_extra_per_agent") or extra_pa

        w_row = batch.get("is_weight")

        if self._q_low_batched and ctrl_aids:
            n = len(ctrl_aids)
            obs_stk = torch.stack([self._obs_preprocessor[a](obs_pa[a]) for a in ctrl_aids])
            nobs_stk = torch.stack([self._obs_preprocessor[a](nobs_pa[a]) for a in ctrl_aids])
            own_act_stk = torch.stack([ja[:, self._agent_slot[a][2] : self._agent_slot[a][2] + 4] for a in ctrl_aids])

            extra_stk = None
            extra_stk_next = None
            if self._q_low_extra_dim > 0:
                extra_stk = torch.stack([self._q_low_extra_for(a, extra_pa) for a in ctrl_aids])
                extra_stk_next = torch.stack([self._q_low_extra_for(a, extra_pa_next) for a in ctrl_aids])
            act_stk = self._q_low_cat_action(own_act_stk, extra_stk)
            rew_stk = torch.stack([low_rewards[a] for a in ctrl_aids])
            with torch.no_grad():
                next_a = torch.stack([self.target_actors[a](nobs_stk[i]) for i, a in enumerate(ctrl_aids)])
                next_act_stk = self._q_low_cat_action(next_a, extra_stk_next)
                target_q = self.target_q_low_batch[team](nobs_stk, next_act_stk)
                target = rew_stk + g_eff * (1.0 - done) * target_q
            current_q = self.q_low_batch[team](obs_stk, act_stk)
            active_stk = torch.stack(
                [
                    active_pa.get(a, torch.ones(done.shape[0], dtype=torch.bool, device=done.device)).bool()
                    for a in ctrl_aids
                ]
            )
            sq_low = (current_q - target) ** 2
            if w_row is not None:
                sq_low = sq_low * w_row.unsqueeze(0)
            loss_per_aid = (sq_low * active_stk).sum(dim=1) / active_stk.sum(dim=1).clamp(min=1)
            if not self._roll_value_frozen(team):
                self.q_low_opt[team].zero_grad()
                loss_per_aid.sum().backward()
                self._clip_q_low_grad_per_aid(team)
                self.q_low_opt[team].step()
            loss_det = loss_per_aid.detach()

            qmean_det = (current_q.detach() * active_stk).sum(dim=1) / active_stk.sum(dim=1).clamp(min=1)
            for i, a in enumerate(ctrl_aids):
                self.track_data(f"Loss / Q_low critic ({a})", loss_det[i])
                self.track_data(f"Q / low mean ({a})", qmean_det[i])
        else:
            nobs_n_by = {a: self._obs_preprocessor[a](nobs_pa[a]) for a in ctrl_aids}
            next_a_by: dict = {}
            _seen = set()
            for aid in ctrl_aids:
                gk = self._actor_group_of[aid]
                if gk in _seen:
                    continue
                _seen.add(gk)
                grp = [a for a in self._actor_groups[gk] if a in ctrl_aids]
                with torch.no_grad():
                    next_a_by.update(
                        self._ctrl_group_actions(
                            gk, {a: nobs_n_by[a] for a in grp}, {a: nobs_pa[a] for a in grp}, target=True
                        )
                    )
            for aid in ctrl_aids:
                obs_i = self._obs_preprocessor[aid](obs_pa[aid])
                nobs_i = nobs_n_by[aid]
                extra_i = self._q_low_extra_for(aid, extra_pa)
                extra_i_next = self._q_low_extra_for(aid, extra_pa_next)
                with torch.no_grad():
                    target_low_q = self.target_q_low[aid](
                        nobs_i, self._q_low_cat_action(next_a_by[aid], extra_i_next)
                    ).squeeze(-1)
                    target_low = low_rewards[aid] + g_eff * (1.0 - done) * target_low_q
                start = self._agent_slot[aid][2]
                current_low_q = self.q_low[aid](
                    obs_i, self._q_low_cat_action(ja[:, start : start + 4], extra_i)
                ).squeeze(-1)
                active_i = active_pa.get(aid, torch.ones_like(done, dtype=torch.bool)).bool()
                if not bool(active_i.any()):
                    continue
                sq_low = (current_low_q - target_low) ** 2
                if w_row is not None:
                    sq_low = sq_low * w_row
                low_critic_loss = sq_low[active_i].mean()
                if not self._roll_value_frozen(self._team_of[aid]):
                    self.q_low_opts[aid].zero_grad()
                    low_critic_loss.backward()
                    if self._grad_norm_clip > 0:
                        nn.utils.clip_grad_norm_(self.q_low[aid].parameters(), self._grad_norm_clip)
                    self.q_low_opts[aid].step()
                self.track_data(f"Loss / Q_low critic ({aid})", low_critic_loss)

                self.track_data(f"Q / low mean ({aid})", current_low_q[active_i].mean())
        seen_low_groups: set = set()
        for aid in ctrl_aids:
            gk = self._actor_group_of[aid]
            if gk in seen_low_groups:
                continue
            seen_low_groups.add(gk)
            group_ctrl = [a for a in self._actor_groups[gk] if a in ctrl_aids]
            if not group_ctrl or any(a in frozen_agents for a in group_ctrl):
                continue
            rep_low = group_ctrl[0]
            low_opt = self.actor_optimizers[rep_low]
            low_opt.zero_grad()
            total_low_actor_loss = 0.0
            obs_n_grp = {a: self._obs_preprocessor[a](obs_pa[a]) for a in group_ctrl}
            a_map = self._ctrl_group_actions(gk, obs_n_grp, {a: obs_pa[a] for a in group_ctrl})
            if self._batched_fwd and self._q_low_batched and True:
                act_stk = torch.stack(
                    [self._q_low_cat_action(a_map[a], self._q_low_extra_for(a, extra_pa)) for a in group_ctrl], dim=0
                )
                active_stk = torch.stack(
                    [active_pa.get(a, torch.ones_like(done, dtype=torch.bool)).bool() for a in group_ctrl], dim=0
                )
                q_stk = self.q_low_batch[team](torch.stack([obs_n_grp[a] for a in group_ctrl], dim=0), act_stk)
                pooled_count = int(active_stk.sum().item())
                if pooled_count == 0:
                    continue
                low_actor_loss = -(q_stk * active_stk.to(q_stk.dtype)).sum() / pooled_count
            else:
                pooled_q_sum = None
                pooled_count = 0
                for a_id in group_ctrl:
                    a_i = a_map[a_id]
                    active_i = active_pa.get(a_id, torch.ones(a_i.shape[0], dtype=torch.bool, device=a_i.device)).bool()

                    act_arg = self._q_low_cat_action(a_i, self._q_low_extra_for(a_id, extra_pa))
                    q_i = self._q_low_forward_eval(a_id, obs_n_grp[a_id], act_arg)
                    q_sum = q_i[active_i].sum()
                    pooled_q_sum = q_sum if pooled_q_sum is None else pooled_q_sum + q_sum
                    pooled_count += int(active_i.sum().item())
                if pooled_count == 0:
                    continue
                low_actor_loss = -pooled_q_sum / pooled_count
            low_actor_loss.backward()
            total_low_actor_loss = low_actor_loss
            if self._grad_norm_clip > 0:
                nn.utils.clip_grad_norm_(self.actors[rep_low].parameters(), self._grad_norm_clip)
            low_opt.step()
            self.track_data(f"Loss / Q_low actor (team_{team}_{rep_low})", total_low_actor_loss)
        seen_target_groups: set = set()
        if self._q_low_batched:
            self._q_low_polyak_batched(team)
        else:
            for aid in ctrl_aids:
                if aid in frozen_agents:
                    continue
                self._soft_update(self.target_q_low[aid], self.q_low[aid])
        for aid in ctrl_aids:
            gk = self._actor_group_of[aid]
            if gk in seen_target_groups:
                continue
            seen_target_groups.add(gk)
            rep_target = self._actor_group_rep[gk]
            self._soft_update(self.target_actors[rep_target], self.actors[rep_target])

    def _update_low_team(self, team: str, members: list[str], batch: dict) -> None:
        ctx = batch.get("q_low_team_ctx")
        if ctx is None:
            raise ValueError("lower_critic_mode=team 但 batch 无 q_low_team_ctx（replay ctx 存储未启用？）")
        nctx = batch.get("next_q_low_team_ctx")
        if nctx is None:
            nctx = ctx
        B = int(ctx.shape[0])
        done = batch.get("boot_terminated", batch["terminated"])
        _g_ns = batch.get("nstep_gamma")
        g_eff = _g_ns if _g_ns is not None else self._gamma_control
        r = batch["reward"]
        w_row = batch.get("is_weight")
        team_active = batch["team_active"].bool()
        anchor_fold = batch["anchor_switch"]
        obs_pa = batch["obs_per_agent"]
        nobs_pa = batch["next_obs_per_agent"]
        active_pa = batch.get("active_per_agent", {})
        ctrl_aids = self._q_low_ctrl_aids(members)
        frozen_agents = getattr(self, "frozen_agents", set())
        rep = members[0]
        _ctx_w = self._M * (self._dec_repr_dim + 1)
        ctx_obs, ctx_ctrl = ctx[:, :_ctx_w], ctx[:, _ctx_w:].view(B, self._M, 4)
        nctx_obs, nctx_ctrl = nctx[:, :_ctx_w], nctx[:, _ctx_w:].view(B, self._M, 4)
        s_n = self._state_preprocessor[rep](batch["state"])
        ns_n = self._state_preprocessor[rep](batch["next_state"])
        done_eff = (done + anchor_fold.to(done.dtype)).clamp(max=1.0)
        ones = torch.ones(B, dtype=torch.bool, device=done.device)

        if ctrl_aids:
            nobs_n_by = {a: self._obs_preprocessor[a](nobs_pa[a]) for a in ctrl_aids}
            with torch.no_grad():
                next_ctrl = nctx_ctrl.clone()
                _seen: set = set()
                for aid in ctrl_aids:
                    gk = self._actor_group_of[aid]
                    if gk in _seen:
                        continue
                    _seen.add(gk)
                    grp = [a for a in self._actor_groups[gk] if a in ctrl_aids]
                    na_map = self._ctrl_group_actions(
                        gk, {a: nobs_n_by[a] for a in grp}, {a: nobs_pa[a] for a in grp}, target=True
                    )
                    for a in grp:
                        di = self._agent_slot[a][0]
                        gate = active_pa.get(a, ones).bool().view(-1, 1)
                        next_ctrl[:, di, :] = torch.where(gate, na_map[a], next_ctrl[:, di, :])
                tq = self.target_q_low_team_net[team](
                    torch.cat([ns_n, nctx_obs], dim=-1), next_ctrl.reshape(B, -1)
                ).squeeze(-1)
                target = r + g_eff * (1.0 - done_eff) * tq
            current_q = self.q_low_team_net[team](torch.cat([s_n, ctx_obs], dim=-1), ctx_ctrl.reshape(B, -1)).squeeze(
                -1
            )
            sq = (current_q - target) ** 2
            if w_row is not None:
                sq = sq * w_row
            n_active = int(team_active.sum().item())
            if n_active > 0:
                loss = (sq * team_active.to(sq.dtype)).sum() / n_active
                if not self._roll_value_frozen(team):
                    self.q_low_team_opts[team].zero_grad()
                    loss.backward()
                    if self._grad_norm_clip > 0:
                        nn.utils.clip_grad_norm_(self.q_low_team_net[team].parameters(), self._grad_norm_clip)
                    self.q_low_team_opts[team].step()

                self.track_data(f"Loss / Q_low critic (team_{team})", loss.detach())
                self.track_data(f"Q / low mean (team_{team})", (current_q.detach() * team_active).sum() / n_active)

        seen_low_groups: set = set()
        for aid in ctrl_aids:
            gk = self._actor_group_of[aid]
            if gk in seen_low_groups:
                continue
            seen_low_groups.add(gk)
            group_ctrl = [a for a in self._actor_groups[gk] if a in ctrl_aids]
            if not group_ctrl or any(a in frozen_agents for a in group_ctrl):
                continue
            rep_low = group_ctrl[0]
            low_opt = self.actor_optimizers[rep_low]
            low_opt.zero_grad()
            obs_n_grp = {a: self._obs_preprocessor[a](obs_pa[a]) for a in group_ctrl}
            a_map = self._ctrl_group_actions(gk, obs_n_grp, {a: obs_pa[a] for a in group_ctrl})
            joint_grad = ctx_ctrl.clone()
            row_gate = None
            for a_id in group_ctrl:
                di = self._agent_slot[a_id][0]
                joint_grad[:, di, :] = a_map[a_id]
                act_i = active_pa.get(a_id, ones).bool()
                row_gate = act_i.clone() if row_gate is None else (row_gate | act_i)
            q = self.q_low_team_net[team](torch.cat([s_n, ctx_obs], dim=-1), joint_grad.reshape(B, -1)).squeeze(-1)
            pooled_count = int(row_gate.sum().item())
            if pooled_count == 0:
                continue
            low_actor_loss = -(q * row_gate.to(q.dtype)).sum() / pooled_count
            low_actor_loss.backward()
            if self._grad_norm_clip > 0:
                nn.utils.clip_grad_norm_(self.actors[rep_low].parameters(), self._grad_norm_clip)
            low_opt.step()
            self.track_data(f"Loss / Q_low actor (team_{team}_{rep_low})", low_actor_loss.detach())

        if not self._roll_value_frozen(team):
            self._soft_update(self.target_q_low_team_net[team], self.q_low_team_net[team])
        seen_target_groups: set = set()
        for aid in ctrl_aids:
            gk = self._actor_group_of[aid]
            if gk in seen_target_groups:
                continue
            seen_target_groups.add(gk)
            rep_target = self._actor_group_rep[gk]
            self._soft_update(self.target_actors[rep_target], self.actors[rep_target])

    def _consume_low_batch(self, team: str, buf) -> dict:
        fut = self._pending_low.pop(team, None)
        if fut is not None:
            return fut.get()
        return buf.sample(self._batch_size)

    def _issue_low_prefetch(self, team: str, buf) -> None:
        if not self._prefetch or len(buf) < self._batch_size:
            return
        self._pending_low[team] = buf.sample_prefetch(self._batch_size, self._prefetch_stream)

    def _q_low_ctrl_aids(self, members: list[str]) -> list[str]:
        if self._lower_critic_mode == "team":
            return [aid for aid in members if not self._is_dec[aid]]
        if self._q_low_batched:
            return [aid for aid in members if not self._is_dec[aid] and aid in self._q_low_idx]
        return [aid for aid in members if not self._is_dec[aid] and aid in self.q_low]

    def _q_low_forward_eval(self, aid: str, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        if self._q_low_batched:
            return self.q_low_batch[self._q_low_team[aid]].forward_one(self._q_low_idx[aid], obs, act)
        return self.q_low[aid](obs, act).squeeze(-1)

    def _q_low_extra_for(self, aid: str, extra_pa: dict | None) -> torch.Tensor | None:
        if self._q_low_extra_dim == 0 or not extra_pa:
            return None
        return extra_pa.get(aid)

    @staticmethod
    def _q_low_cat_action(own_ctrl: torch.Tensor, extra: torch.Tensor | None) -> torch.Tensor:
        if extra is None:
            return own_ctrl
        return torch.cat([own_ctrl, extra], dim=-1)

    def _clip_q_low_grad_per_aid(self, team: str) -> None:
        if self._grad_norm_clip <= 0:
            return
        qb = self.q_low_batch[team]
        keys = list(qb.params.keys())
        for i in range(qb.n_aid):
            sq = None
            for k in keys:
                g = qb.params[k].grad
                if g is None:
                    continue
                term = (g[i] ** 2).sum()
                sq = term if sq is None else sq + term
            if sq is None:
                continue
            norm = sq.sqrt()
            coef = self._grad_norm_clip / (norm + 1e-6)
            if coef < 1.0:
                for k in keys:
                    g = qb.params[k].grad
                    if g is not None:
                        g[i].mul_(coef)

    def _q_low_polyak_batched(self, team: str) -> None:
        if self._q_low_batched:
            self._soft_update(self.target_q_low_batch[team], self.q_low_batch[team])

    def _hier_update(self, *, timestep, timesteps):
        for team, members in self._team_members.items():
            buf = self.replays[team]
            if self._hier_dual:
                if (not self._high_ppo_mode) and timestep % self._high_update_every == 0:
                    self._update_high_from_decision_replay(team, members)
                if len(buf) >= self._batch_size:
                    self._update_low_from_control_batch(team, members, self._consume_low_batch(team, buf))
                if timestep % self._telemetry_interval == 0:
                    hbuf = self.high_replays.get(team)
                    self.track_data(f"Replay / Decision_size (team_{team})", len(hbuf) if hbuf is not None else 0)
                    self.track_data(f"Replay / Control_size (team_{team})", len(buf))
                    if self._replay_terminal_mix > 0.0:
                        self.track_data(
                            f"Replay / terminal_frac (team_{team})", buf.priority[: len(buf)].float().mean()
                        )
                self._issue_low_prefetch(team, buf)
                continue
            if len(buf) < self._batch_size:
                continue
            batch = self._consume_low_batch(team, buf)
            s = batch["state"]
            ns = batch["next_state"]
            ja = batch["joint_action"]
            r = batch["reward"]
            done = batch.get("boot_terminated", batch["terminated"])

            _g_ns = batch.get("nstep_gamma")
            g_eff = _g_ns if _g_ns is not None else self._gamma_control

            _asw = batch.get("anchor_switch")
            if _asw is not None:
                done = (done.bool() | _asw.bool()).to(done.dtype)
            team_active = batch.get("team_active", torch.ones_like(done)).bool()
            active_pa = batch.get("active_per_agent", {})
            obs_pa = batch["obs_per_agent"]
            nobs_pa = batch["next_obs_per_agent"]
            if self._normalize:
                rep = members[0]
                s = self._state_preprocessor[rep](s)
                ns = self._state_preprocessor[rep](ns)

            with torch.no_grad():
                next_ja = ja.clone()

                for aid in self.possible_agents:
                    if not self._is_dec[aid]:
                        continue
                    _, _, start, width = self._agent_slot[aid]
                    raw = nobs_pa[aid]
                    nobs_i = self._obs_preprocessor[aid](raw)
                    mask = raw[:, -self._mask_tail_dim :]
                    dec_oh = self.target_actors[aid].argmax_onehot(nobs_i, mask)
                    next_ja[:, start : start + width] = torch.where(
                        active_pa.get(aid, torch.ones(next_ja.shape[0], dtype=torch.bool, device=next_ja.device))
                        .bool()
                        .unsqueeze(-1),
                        dec_oh,
                        next_ja[:, start : start + width],
                    )
                for aid in self.possible_agents:
                    if self._is_dec[aid]:
                        continue
                    _, _, start, width = self._agent_slot[aid]
                    raw = nobs_pa[aid]
                    du = aid.split("/")[0]
                    nobs_i = self._obs_preprocessor[aid](raw)
                    next_a = self.target_actors[aid](nobs_i)
                    next_ja[:, start : start + width] = torch.where(
                        active_pa.get(aid, torch.ones(next_ja.shape[0], dtype=torch.bool, device=next_ja.device))
                        .bool()
                        .unsqueeze(-1),
                        next_a,
                        next_ja[:, start : start + width],
                    )
                target_q = self.target_critics[team](ns, next_ja).squeeze(-1)
                target = r + g_eff * (1.0 - done) * target_q

            current_q = self.critics[team](s, ja).squeeze(-1)
            _sq = (current_q - target) ** 2
            _w_row = batch.get("is_weight")
            if _w_row is not None:
                _sq = _sq * _w_row
            critic_loss = self._masked_scalar_mean(_sq, team_active)
            if not self._roll_value_frozen(team):
                self.critic_optimizers[team].zero_grad()
                critic_loss.backward()
                if self._grad_norm_clip > 0:
                    nn.utils.clip_grad_norm_(self.critics[team].parameters(), self._grad_norm_clip)
                self.critic_optimizers[team].step()

            frozen_agents = getattr(self, "frozen_agents", set())
            seen_groups: set = set()
            for aid in members:
                gk = self._actor_group_of[aid]
                if gk in seen_groups:
                    continue
                seen_groups.add(gk)
                group_aids = self._actor_groups[gk]
                rep = group_aids[0]

                if any(a in frozen_agents for a in group_aids):
                    continue
                opt = self.actor_optimizers[rep]
                opt.zero_grad()
                total_actor_loss = 0.0

                for a_id in group_aids:
                    _, _, start, width = self._agent_slot[a_id]
                    obs_i = self._obs_preprocessor[a_id](obs_pa[a_id])
                    ja_grad = ja.clone()
                    if self._is_dec[a_id]:
                        mask = obs_pa[a_id][:, -self._mask_tail_dim :]
                        a_i = self.actors[a_id](obs_i, mask, tau=self._gumbel_tau, hard=True)
                        ja_grad[:, start : start + width] = a_i
                    a_i = self.actors[a_id](obs_i)
                    ja_grad[:, start : start + width] = a_i
                    active_i = active_pa.get(a_id, torch.ones(s.shape[0], dtype=torch.bool, device=s.device)).bool()
                    ja_grad[:, start : start + width] = torch.where(
                        active_i.unsqueeze(-1), ja_grad[:, start : start + width], ja[:, start : start + width]
                    )
                    q = self.critics[team](s, ja_grad).squeeze(-1)
                    actor_loss = -self._masked_scalar_mean(q, active_i) / len(group_aids)
                    actor_loss.backward()
                    total_actor_loss = total_actor_loss + actor_loss.detach().double()
                if self._grad_norm_clip > 0:
                    nn.utils.clip_grad_norm_(self.actors[rep].parameters(), self._grad_norm_clip)
                opt.step()
                self.track_data(f"Loss / Actor loss (team_{team}_{rep})", total_actor_loss)

            if not self._roll_value_frozen(team):
                self._soft_update(self.target_critics[team], self.critics[team])
            seen_groups = set()
            for aid in members:
                gk = self._actor_group_of[aid]
                if gk in seen_groups:
                    continue
                seen_groups.add(gk)
                if any(a in frozen_agents for a in self._actor_groups[gk]):
                    continue
                rep = self._actor_group_rep[gk]
                self._soft_update(self.target_actors[rep], self.actors[rep])

            self.track_data(f"Loss / Critic loss (team_{team})", critic_loss)
            self.track_data(f"Q / mean (team_{team})", current_q.mean())
            self.track_data(f"Q / max (team_{team})", current_q.max())
            self._issue_low_prefetch(team, buf)
