# [TAD vendor + 改造] 衍生自 skrl 2.1.0 的 MAPPO（MIT, Copyright skrl contributors）

# 原始来源：skrl/multi_agents/torch/mappo/mappo.py


from __future__ import annotations

from typing import Any

import gymnasium
from packaging import version

import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.spaces import Discrete

from skrl import config, logger
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.multi_agents.torch import MultiAgent
from skrl.resources.schedulers.torch import KLAdaptiveLR

from ..resume_checkpoint import register_training_progress, register_architecture_fingerprint
from ..team_utils import (
    DecisionTrajectoryCollector,
    masked_mean,
    select_dynamic_team_anchor,
    stage_decisions_rebuild_ctrl,
)
from ..roll_rotation import update_roll_rotation
from ..train_takeover import update_train_takeover
from ..models import (
    mlp_backbone_fingerprint_spec,
)
from .mappo_cfg import MAPPO_CFG


def compute_gae(
    *,
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    values: torch.Tensor,
    last_values: torch.Tensor,
    discount_factor: float = 0.99,
    lambda_coefficient: float = 0.95,
) -> torch.Tensor:
    """Compute returns and advantages over a rollout."""

    advantage = 0
    advantages = torch.zeros_like(rewards)

    not_done = (terminated | truncated).logical_not()
    memory_size = rewards.shape[0]

    for i in reversed(range(memory_size)):
        next_values = values[i + 1] if i < memory_size - 1 else last_values
        advantage = (
            rewards[i] - values[i] + discount_factor * not_done[i] * (next_values + lambda_coefficient * advantage)
        )
        advantages[i] = advantage

    returns = advantages + values

    return returns, advantages


def terminated_local_from_active(active: torch.Tensor, terminated: torch.Tensor) -> torch.Tensor:
    a = active.to(torch.bool)
    nd = torch.zeros_like(a)
    nd[:-1] = a[:-1] & ~a[1:]
    return terminated.to(torch.bool) | nd


class MAPPO(MultiAgent):
    def __init__(
        self,
        *,
        possible_agents: list[str],
        models: dict[str, dict[str, Model]],
        memories: dict[str, Memory] | None = None,
        observation_spaces: dict[str, gymnasium.Space] | None = None,
        state_spaces: dict[str, gymnasium.Space] | None = None,
        action_spaces: dict[str, gymnasium.Space] | None = None,
        device: str | torch.device | None = None,
        cfg: MAPPO_CFG | dict = {},
        team_of: dict[str, str] | None = None,
        team_reward_aggr: str = "sum",
        hierarchical_cfg: dict | None = None,
        actor_mode: str = "per_agent",
    ) -> None:
        self.cfg: MAPPO_CFG
        super().__init__(
            possible_agents=possible_agents,
            models=models,
            memories=memories,
            observation_spaces=observation_spaces,
            state_spaces=state_spaces,
            action_spaces=action_spaces,
            device=device,
            cfg=MAPPO_CFG(**cfg) if isinstance(cfg, dict) else cfg,
        )

        self._team_of = team_of if team_of is not None else {uid: uid for uid in self.possible_agents}
        self._team_reward_aggr = team_reward_aggr

        hc = hierarchical_cfg or {}

        self._rollouts_dec_cfg = int(hc.get("rollouts_dec", 0))

        self._expert_curriculum_on = bool(hc.get("expert_curriculum_enable", False))

        self._state_anchor_free = str(hc.get("critic_state", "token")).lower() == "token"
        self._expert_prob0 = float(hc.get("expert_curriculum_prob", 0.0))
        self._expert_anneal_frac = float(hc.get("expert_curriculum_anneal_frac", 0.0))
        self._expert_prob_cur = self._expert_prob0
        self._expert_anneal_done = False
        self._expert_anneal_warned = False
        self._dec_rollouts = 0
        self.frozen_agents: set[str] = set()

        self._team_members: dict[str, list[str]] = {}
        for uid in self.possible_agents:
            self._team_members.setdefault(self._team_of[uid], []).append(uid)

        self._team_drone_count: dict[str, int] = {
            team: len({m.split("/", 1)[0] for m in members}) for team, members in self._team_members.items()
        }

        self._team_representative: dict[str, str] = {team: members[0] for team, members in self._team_members.items()}

        self._state_anchor: dict[str, str] = {
            uid: self._team_representative[self._team_of[uid]] for uid in self.possible_agents
        }

        self._current_anchor_states: dict[str, torch.Tensor] = {}
        self._next_anchor_states: dict[str, torch.Tensor] = {}
        self._last_alive: dict[str, torch.Tensor] = {}

        self._hier_map = any("/" in uid for uid in self.possible_agents)

        def _layer_of(u: str) -> str:
            if self._hier_map:
                return "dec" if u.endswith("/dec") else "ctrl"
            return "agent"

        self._layer_of = _layer_of

        def _value_team_layer(u: str) -> tuple[str, str]:
            return (self._team_of[u], _layer_of(u))

        self._value_team_layer = _value_team_layer
        self._value_representative: dict[tuple[str, str], str] = {}
        for uid in self.possible_agents:
            self._value_representative.setdefault(_value_team_layer(uid), uid)

        self.policies = {uid: self.models[uid].get("policy", None) for uid in self.possible_agents}
        self.values = {uid: self.models[uid].get("value", None) for uid in self.possible_agents}

        for uid in self.possible_agents:
            self.checkpoint_modules[uid]["policy"] = self.policies[uid]
            self.checkpoint_modules[uid]["value"] = self.values[uid]

        self._discrete = any(isinstance(self.action_spaces[uid], Discrete) for uid in self.possible_agents)

        self._hier = any("/" in uid for uid in self.possible_agents)
        self._is_dec = {uid: isinstance(self.action_spaces[uid], Discrete) for uid in self.possible_agents}

        self._actor_mode = str(actor_mode) if (self._hier or self._discrete) else "per_agent"
        self._actor_group_of: dict = {}
        self._actor_groups: dict = {}
        for uid in self.possible_agents:
            if self._hier:
                layer = "dec" if self._is_dec[uid] else "ctrl"
                team = uid.split("/")[0].split("_")[0]
            else:
                layer, team = "agent", uid.split("_")[0]
            gk = (team, layer) if self._actor_mode == "team" else ("__uid__", uid)
            self._actor_group_of[uid] = gk
            self._actor_groups.setdefault(gk, []).append(uid)
        self._actor_group_rep: dict = {gk: uids[0] for gk, uids in self._actor_groups.items()}

        if self._actor_mode == "team":
            for gk, uids in self._actor_groups.items():
                for uid in uids[1:]:
                    self.checkpoint_modules[uid].pop("policy", None)

        for uid in self.possible_agents:
            if config.torch.is_distributed:
                logger.info("Broadcasting models' parameters")
                if self.policies[uid] is not None:
                    self.policies[uid].broadcast_parameters()

                    if self.values[uid] is not None and self.policies[uid] is not self.values[uid]:
                        self.values[uid].broadcast_parameters()

        self._device_type = torch.device(self.device).type
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.amp.GradScaler(device=self._device_type, enabled=self.cfg.mixed_precision)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.mixed_precision)

        self.policy_optimizers = {}
        self.value_optimizers = {}
        self.schedulers = {}
        for gk, uids in self._actor_groups.items():
            rep = uids[0]
            if self.policies[rep] is None:
                for uid in uids:
                    self.schedulers[uid] = None
                continue

            opt = torch.optim.Adam(self.policies[rep].parameters(), lr=self.cfg.learning_rate[rep][0])
            sched_spec = self.cfg.learning_rate_scheduler[rep][0]
            sched = (
                sched_spec(opt, **self.cfg.learning_rate_scheduler_kwargs[rep][0]) if sched_spec is not None else None
            )
            for uid in uids:
                self.policy_optimizers[uid] = opt
                self.schedulers[uid] = sched

            self.checkpoint_modules[rep]["optimizer"] = opt
            if sched is not None:
                self.checkpoint_modules[rep]["scheduler"] = sched

        seen_vtl = set()
        for uid in self.possible_agents:
            vtl = self._value_team_layer(uid)
            if vtl in seen_vtl:
                continue
            seen_vtl.add(vtl)
            if self.values[uid] is not None:
                self.value_optimizers[vtl] = torch.optim.Adam(
                    self.values[uid].parameters(), lr=self.cfg.learning_rate[uid][1]
                )

        self._observation_preprocessor = {}
        self._state_preprocessor = {}
        self._value_preprocessor = {}
        team_state_preprocessors = {}
        vtl_value_preprocessors = {}
        for uid in self.possible_agents:
            if self.cfg.observation_preprocessor[uid]:
                self._observation_preprocessor[uid] = self.cfg.observation_preprocessor[uid](
                    **self.cfg.observation_preprocessor_kwargs[uid]
                )
                self.checkpoint_modules[uid]["observation_preprocessor"] = self._observation_preprocessor[uid]
            else:
                self._observation_preprocessor[uid] = self._empty_preprocessor

            team = self._team_of[uid]
            if team not in team_state_preprocessors:
                if self.cfg.state_preprocessor[uid]:
                    team_state_preprocessors[team] = self.cfg.state_preprocessor[uid](
                        **self.cfg.state_preprocessor_kwargs[uid]
                    )
                else:
                    team_state_preprocessors[team] = self._empty_preprocessor
            vtl = self._value_team_layer(uid)
            if vtl not in vtl_value_preprocessors:
                if self.cfg.value_preprocessor[uid]:
                    vtl_value_preprocessors[vtl] = self.cfg.value_preprocessor[uid](
                        **self.cfg.value_preprocessor_kwargs[uid]
                    )
                else:
                    vtl_value_preprocessors[vtl] = self._empty_preprocessor

            self._state_preprocessor[uid] = team_state_preprocessors[team]
            self._value_preprocessor[uid] = vtl_value_preprocessors[vtl]

            if self.cfg.state_preprocessor[uid]:
                self.checkpoint_modules[uid]["state_preprocessor"] = self._state_preprocessor[uid]
            if self.cfg.value_preprocessor[uid]:
                self.checkpoint_modules[uid]["value_preprocessor"] = self._value_preprocessor[uid]

        for uid in self.possible_agents:
            vtl = self._value_team_layer(uid)
            if uid == self._value_representative[vtl]:
                self.checkpoint_modules[uid]["value_optimizer"] = self.value_optimizers[vtl]
            if self.schedulers.get(uid) is not None and uid == self._actor_group_rep[self._actor_group_of[uid]]:
                self.checkpoint_modules[uid]["scheduler"] = self.schedulers[uid]
            self.checkpoint_modules[uid]["amp_scaler"] = self.scaler

        self._n_pool = int(hc.get("n_pool", 0))
        self._n_dec = self._n_pool
        self._gamma_control = float(hc.get("gamma_control", 0.99))
        self._telemetry_interval = int(hc.get("telemetry_interval", 50))

        self._dec_collectors: dict[str, DecisionTrajectoryCollector] = {}

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
            "format": "tad_hier_mappo",
            "requires_fingerprint": True,
            "upper_backbone": mlp_backbone_fingerprint_spec(),
            "lower_backbone": mlp_backbone_fingerprint_spec(),
            "dual_v": True,
            "actor_mode": self._actor_mode,
            "dec_head": "mlp",
            "critic_backbone": "mlp",
            "critic_state": str(hc.get("critic_state", "token")),
            "spaces_dims": _spaces_dims,
        }
        register_architecture_fingerprint(self, self._architecture_fingerprint)

        register_training_progress(self)

    def _dec_action_mask(self, obs: torch.Tensor) -> torch.Tensor:
        return obs[:, -self._n_dec :] if self._n_dec > 0 else obs[:, -self._n_pool :]

    def init(self, *, trainer_cfg: dict[str, Any] | None = None) -> None:
        super().init(trainer_cfg=trainer_cfg)
        self.enable_models_training_mode(False)

        if self._expert_curriculum_on and self._expert_anneal_frac > 0.0:
            setter = getattr(getattr(self, "_hier_env", None), "set_expert_curriculum_prob", None)
            if setter is not None:
                setter(1.0)
                self._expert_prob_cur = 1.0

        if self.memories:
            for uid in self.possible_agents:
                self.memories[uid].create_tensor(
                    name="observations", size=self.observation_spaces[uid], dtype=torch.float32
                )
                self.memories[uid].create_tensor(name="states", size=self.state_spaces[uid], dtype=torch.float32)
                self.memories[uid].create_tensor(name="actions", size=self.action_spaces[uid], dtype=torch.float32)
                self.memories[uid].create_tensor(name="rewards", size=1, dtype=torch.float32)
                self.memories[uid].create_tensor(name="terminated", size=1, dtype=torch.bool)
                self.memories[uid].create_tensor(name="truncated", size=1, dtype=torch.bool)
                self.memories[uid].create_tensor(name="log_prob", size=1, dtype=torch.float32)
                self.memories[uid].create_tensor(name="values", size=1, dtype=torch.float32)
                self.memories[uid].create_tensor(name="returns", size=1, dtype=torch.float32)
                self.memories[uid].create_tensor(name="advantages", size=1, dtype=torch.float32)
                self.memories[uid].create_tensor(name="active", size=1, dtype=torch.bool)
                self.memories[uid].create_tensor(name="team_active", size=1, dtype=torch.bool)

                self.memories[uid].create_tensor(name="anchor_switch", size=1, dtype=torch.bool)

            self._tensors_names = [
                "observations",
                "states",
                "actions",
                "log_prob",
                "values",
                "returns",
                "advantages",
                "active",
                "team_active",
            ]
            if self._hier:
                _n = self.memories[self.possible_agents[0]].num_envs
                _bound = max(4096, int(self.cfg.rollouts) * _n * 8)
                self._dec_collectors = {
                    uid: DecisionTrajectoryCollector(
                        num_envs=_n, max_completed_samples=_bound, gamma_c=self._gamma_control
                    )
                    for uid in self.possible_agents
                    if self._is_dec.get(uid, False)
                }

                self._dec_rollouts = self._rollouts_dec_cfg if self._rollouts_dec_cfg > 0 else _n

        self._current_next_observations = {}
        self._current_next_states = {}
        self._current_log_prob = {}
        self._current_values = {}
        self._rollout = 0

    def _alive_from_bound_env(self) -> dict[str, torch.Tensor] | None:
        env = getattr(self, "_hier_env", None) or getattr(self, "_warmup_env", None)
        for candidate in (env, getattr(env, "unwrapped", None), getattr(env, "env", None)):
            alive = getattr(candidate, "_alive", None)
            if isinstance(alive, dict):
                return alive
        return self._last_alive or None

    def _members_for_anchor(self, uid: str) -> list[str]:
        layer = self._layer_of(uid)
        return [m for m in self._team_members[self._team_of[uid]] if self._layer_of(m) == layer]

    def _anchor_state(
        self, states: dict[str, torch.Tensor], uid: str, alive: dict[str, torch.Tensor] | None
    ) -> torch.Tensor:
        if not alive:
            return states[self._state_anchor[uid]]
        members = self._members_for_anchor(uid)
        try:
            selected, _ = select_dynamic_team_anchor(states, alive, members, fallback_member=self._state_anchor[uid])
            return selected
        except (KeyError, ValueError):
            return states[self._state_anchor[uid]]

    def _anchor_index(
        self, states: dict[str, torch.Tensor], uid: str, alive: dict[str, torch.Tensor] | None
    ) -> torch.Tensor | None:
        if not alive or not isinstance(states, dict):
            return None
        members = self._members_for_anchor(uid)
        try:
            _, idx = select_dynamic_team_anchor(states, alive, members, fallback_member=self._state_anchor[uid])
            return idx
        except (KeyError, ValueError):
            return None

    @staticmethod
    def _physical_mask(mapping, uid: str, *, default: torch.Tensor) -> torch.Tensor:
        if isinstance(mapping, dict):
            physical = uid.split("/", 1)[0]
            value = mapping.get(uid, mapping.get(physical))
            if value is not None:
                return value.to(default.device).bool().view(-1)
        return default.bool().view(-1)

    @staticmethod
    def _team_term_sum(term_info, members, reference: torch.Tensor) -> torch.Tensor | None:
        if not isinstance(term_info, dict):
            return None
        picked: dict[str, torch.Tensor] = {}
        for m in members:
            uid = m.split("/", 1)[0]
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

    @staticmethod
    def _gae_terminated(memory) -> torch.Tensor:
        terminated = terminated_local_from_active(
            memory.get_tensor_by_name("active"), memory.get_tensor_by_name("terminated")
        )
        try:
            switch = memory.get_tensor_by_name("anchor_switch")
        except (KeyError, ValueError):
            return terminated
        return terminated | switch.to(torch.bool)

    @staticmethod
    def _masked_loss(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return masked_mean(values, mask, dim=0)

    @staticmethod
    def _normalize_active_advantages(advantages: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        mask = active.bool()
        while mask.ndim < advantages.ndim:
            mask = mask.unsqueeze(-1)
        selected = advantages[mask]
        if selected.numel() == 0:
            return torch.zeros_like(advantages)
        std = selected.std(unbiased=False)
        normalized = (advantages - selected.mean()) / (std + 1e-8)
        return torch.where(mask, normalized, torch.zeros_like(normalized))

    def _roll_value_frozen(self, team: str) -> bool:
        return team in getattr(self, "_roll_full_frozen_teams", set())

    def _log_gae(self, uid: str, returns: torch.Tensor, advantages: torch.Tensor, values: torch.Tensor) -> None:
        with torch.no_grad():
            r = returns.float().reshape(-1)
            v = values.float().reshape(-1)
            a = advantages.float().reshape(-1)
            var_r = r.var(unbiased=False)
            ev = (1.0 - (r - v).var(unbiased=False) / (var_r + 1e-8)).clamp(-1.0, 1.0)
            tag = f"team_{self._team_of[uid]}_{self._layer_of(uid)}"
            self.track_data(f"Value / explained_variance ({tag})", float(ev))
            self.track_data(f"Value / advantage_mean ({tag})", float(a.mean()))
            self.track_data(f"Value / advantage_std ({tag})", float(a.std(unbiased=False)))
            self.track_data(f"Value / return_mean ({tag})", float(r.mean()))
            self.track_data(f"Value / return_std ({tag})", float(r.std(unbiased=False)))

    def act(self, observations, states, *, timestep, timesteps):
        actions = {}
        log_prob = {}
        outputs = {}
        current_values = {}

        self._current_gaussian: dict = {}
        current_alive = self._alive_from_bound_env()
        self._current_anchor_states = {
            uid: self._anchor_state(states, uid, current_alive) for uid in self.possible_agents
        }

        def _value_of(uid, inputs):
            if self.training:
                values, _ = self.values[uid].act(inputs, role="value")
                current_values[uid] = self._value_preprocessor[uid](values, inverse=True)

        warmup_steps = getattr(self, "_warmup_steps", 0)
        warmup_env = getattr(self, "_warmup_env", None)
        warmup_mask = None
        if warmup_steps > 0 and warmup_env is not None:
            warmup_mask = warmup_env.episode_length_buf < warmup_steps

        self._current_ctrl_obs_raw: dict = {}
        _dec_uids = [uid for uid in self.possible_agents if self._is_dec.get(uid, self._discrete)]
        _ctrl_uids = [uid for uid in self.possible_agents if not self._is_dec.get(uid, self._discrete)]
        for _pass_uids in (_dec_uids, _ctrl_uids):
            if _pass_uids is _ctrl_uids and _ctrl_uids:
                _rebuilt = stage_decisions_rebuild_ctrl(self, observations, actions)
                self._current_ctrl_obs_raw = dict(_rebuilt)
                observations = {**observations, **_rebuilt}
            for uid in _pass_uids:
                inputs = {
                    "observations": self._observation_preprocessor[uid](observations[uid]),
                    "states": self._state_preprocessor[uid](self._current_anchor_states[uid]),
                }

                if self._is_dec.get(uid, self._discrete):
                    inputs["action_mask"] = self._dec_action_mask(observations[uid])

                if timestep < self.cfg.random_timesteps:
                    policy_out, outputs[uid] = self.policies[uid].random_act(inputs, role="policy")
                    log_prob[uid] = outputs[uid].get(
                        "log_prob", torch.zeros(observations[uid].shape[0], device=self.device)
                    )
                    actions[uid] = policy_out
                    if self.training:
                        values, _ = self.values[uid].act(inputs, role="value")
                        current_values[uid] = self._value_preprocessor[uid](values, inverse=True)
                else:
                    with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                        policy_out, outputs[uid] = self.policies[uid].act(inputs, role="policy")

                        if not self.training:
                            if self._is_dec.get(uid, self._discrete):
                                policy_out = outputs[uid]["logits"].argmax(dim=-1, keepdim=True)
                            else:
                                policy_out = outputs[uid]["mean_actions"].clamp(-1.0, 1.0)
                        log_prob[uid] = outputs[uid]["log_prob"]
                        actions[uid] = policy_out

                        _gd = getattr(self.policies[uid], "_g_distribution", None)
                        if _gd is not None:
                            self._current_gaussian[uid] = (_gd.mean.detach(), _gd.stddev.detach())
                    _value_of(uid, inputs)

                _ev_m = self._eval_row_mask(observations[uid])
                if _ev_m is not None:
                    if timestep < self.cfg.random_timesteps:
                        if not getattr(self, "_eval_random_warned", False):
                            self._eval_random_warned = True
                            print(
                                "[TAD][test-env][WARN] timestep < random_timesteps，test 行确定性覆写暂不生效"
                                "（该窗口内 Test/* 指标不具参考性；random_timesteps 默认 0 不影响正常训练）"
                            )
                    else:
                        _det = self._eval_det_rows(uid, inputs, observations, outputs.get(uid), _ev_m)
                        if _det is not None and _det.shape == actions[uid].shape:
                            actions[uid] = torch.where(
                                _ev_m.view([-1] + [1] * (actions[uid].dim() - 1)),
                                _det.to(actions[uid].dtype),
                                actions[uid],
                            )

                if warmup_mask is not None and not self._is_dec.get(uid, self._discrete) and bool(warmup_mask.any()):
                    m_col = warmup_mask.unsqueeze(-1)

                    ctrl_env = getattr(self, "_warmup_env", None)
                    ctrl_fn = (
                        getattr(ctrl_env, "compute_warmup_stabilize_action", None) if ctrl_env is not None else None
                    )

                    ctrl_uid = uid.split("/")[0] if self._hier else uid
                    ctrl = ctrl_fn(ctrl_uid) if ctrl_fn is not None else torch.zeros_like(actions[uid])
                    actions[uid] = torch.where(m_col, ctrl, actions[uid])

                    rec_gauss = self._current_gaussian.get(uid)
                    if self.training and rec_gauss is not None:
                        rec_w = actions[uid].detach()
                        mean_w, std_w = rec_gauss
                        logp_w = torch.distributions.Normal(mean_w, std_w).log_prob(rec_w).sum(dim=-1, keepdim=True)
                        log_prob[uid] = torch.where(m_col, logp_w, log_prob[uid])

        self._current_log_prob = log_prob
        self._current_values = current_values
        return actions, outputs

    def _eval_row_mask(self, ref: torch.Tensor) -> torch.Tensor | None:
        m = getattr(self, "_eval_env_mask", None)
        if m is None:
            return None
        m = m.to(device=ref.device, dtype=torch.bool).view(-1)
        return m if m.numel() == ref.shape[0] else None

    def _eval_det_rows(self, uid, inputs, observations, out_uid, eval_mask):
        idx = eval_mask.nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            return None
        g = self._current_gaussian.get(uid)
        if g is not None:
            return g[0][idx].clamp(-1.0, 1.0)
        if out_uid is None:
            return None
        if self._is_dec.get(uid, self._discrete):
            logits = out_uid.get("logits")
            return None if logits is None else logits[idx].argmax(dim=-1, keepdim=True)
        mean = out_uid.get("mean_actions")
        return None if mean is None else mean[idx].clamp(-1.0, 1.0)

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
        if getattr(self, "_current_ctrl_obs_raw", None):
            observations = {**observations, **self._current_ctrl_obs_raw}

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

        truncated_lb = truncated
        if _eval_m is not None and isinstance(truncated, dict):
            truncated_lb = {k: v & ~_eval_m.view([-1] + [1] * (v.dim() - 1)) for k, v in truncated.items()}

        if self.training:
            self._current_next_observations = next_observations
            self._current_next_states = next_states
            was_alive_info = infos.get("agent_was_alive") if isinstance(infos, dict) else None
            still_alive_info = infos.get("agent_still_alive") if isinstance(infos, dict) else None
            alive_count_info = infos.get("team_alive_count") if isinstance(infos, dict) else None

            term_info = infos.get("agent_term_reward") if isinstance(infos, dict) else None
            _ref_done = terminated[self.possible_agents[0]].view(-1)
            _ones = torch.ones_like(_ref_done, dtype=torch.bool)
            active = {uid: self._physical_mask(was_alive_info, uid, default=_ones) for uid in self.possible_agents}
            still_alive = {
                uid.split("/", 1)[0]: self._physical_mask(still_alive_info, uid, default=_ones)
                for uid in self.possible_agents
            }
            self._last_alive = still_alive
            self._next_anchor_states = {
                uid: self._anchor_state(next_states, uid, still_alive) for uid in self.possible_agents
            }

            _team_switch: dict[str, torch.Tensor] = {}
            anchor_switch: dict[str, torch.Tensor] = {}
            for uid in self.possible_agents:
                team = self._team_of[uid]
                if self._state_anchor_free:
                    anchor_switch[uid] = torch.zeros_like(_ref_done, dtype=torch.bool)
                    continue
                if team not in _team_switch:
                    idx_t = self._anchor_index(states, uid, active)
                    idx_t1 = self._anchor_index(next_states, uid, still_alive)
                    _team_switch[team] = (
                        (idx_t != idx_t1)
                        if (idx_t is not None and idx_t1 is not None)
                        else torch.zeros_like(_ref_done, dtype=torch.bool)
                    )
                anchor_switch[uid] = _team_switch[team]
            team_active = {}
            for team, members in self._team_members.items():
                if isinstance(alive_count_info, dict) and team in alive_count_info:
                    team_active[team] = alive_count_info[team].to(self.device).view(-1) > 0
                else:
                    team_active[team] = torch.stack([active[m] for m in members], dim=0).any(dim=0)

            if _eval_m is not None:
                _tr = ~_eval_m
                learn_active = {u: (m & _tr) for u, m in active.items()}
                learn_still_alive = {u: (m & _tr) for u, m in still_alive.items()}
                learn_team_active = {t: (m & _tr) for t, m in team_active.items()}
            else:
                learn_active, learn_still_alive, learn_team_active = active, still_alive, team_active

            shaped = {}
            for uid in self.possible_agents:
                r = rewards[uid]
                if self.cfg.rewards_shaper is not None:
                    r = self.cfg.rewards_shaper(r, timestep, timesteps)
                if self.cfg.time_limit_bootstrap[uid] and truncated_lb[uid].any():
                    with torch.no_grad():
                        nobs_tb = next_observations[uid]
                        nst_tb = self._next_anchor_states[uid]
                        env_tb = getattr(self, "_hier_env", None)
                        fin_tb = (
                            env_tb.temporal_transition_info().get("truncation_final") if env_tb is not None else None
                        )
                        if fin_tb is not None:
                            _fin_mask, _fin_obs, _fin_states = fin_tb
                            _tc = truncated_lb[uid].view(-1).bool()
                            _fo = _fin_obs.get(uid)
                            if _fo is not None and _fo.shape == nobs_tb.shape:
                                nobs_tb = torch.where(
                                    _tc.view([-1] + [1] * (nobs_tb.dim() - 1)), _fo.to(nobs_tb.dtype), nobs_tb
                                )
                            _fs = _fin_states.get(uid)
                            if _fs is not None and _fs.shape == nst_tb.shape:
                                nst_tb = torch.where(
                                    _tc.view([-1] + [1] * (nst_tb.dim() - 1)), _fs.to(nst_tb.dtype), nst_tb
                                )
                        inputs = {
                            "observations": self._observation_preprocessor[uid](nobs_tb),
                            "states": self._state_preprocessor[uid](nst_tb),
                        }
                        next_values, _ = self.values[uid].act(inputs, role="value")
                        next_values = self._value_preprocessor[uid](next_values, inverse=True)
                    r = r + self.cfg.discount_factor[uid] * next_values * truncated_lb[uid]
                shaped[uid] = r

            alive_count = alive_count_info
            team_reward = {}
            for team, members in self._team_members.items():
                agg = shaped[members[0]].clone()
                for m in members[1:]:
                    agg = agg + shaped[m]
                if self._team_reward_aggr == "mean":
                    if alive_count is not None and team in alive_count:
                        denom = alive_count[team].to(agg.device).float().view_as(agg).clamp(min=1.0)
                    else:
                        denom = self._team_drone_count.get(team, max(len(members), 1))

                    team_term = self._team_term_sum(term_info, members, agg)
                    if team_term is not None:
                        n_nominal = float(self._team_drone_count.get(team, max(len(members), 1)))
                        agg = (agg - team_term) / denom + team_term / n_nominal
                    else:
                        agg = agg / denom
                team_reward[team] = agg

                self.track_data(
                    f"Reward / shaped (team_{team})",
                    agg.float().mean().item() if _eval_m is None else agg.float()[~_eval_m].mean().item(),
                )

            per_uid_reward = {uid: team_reward[self._team_of[uid]] for uid in self.possible_agents}
            env_h_rec = getattr(self, "_hier_env", None)

            for uid in self.possible_agents:
                team = self._team_of[uid]
                stored_actions = actions[uid]
                stored_log_prob = self._current_log_prob[uid]

                if self._hier and self._is_dec.get(uid, False):
                    env_h = getattr(self, "_hier_env", None)
                    du = uid.split("/")[0]
                    eff_flat = env_h._eff_dec_idx[du].long().view(-1)
                    stored_actions = eff_flat.view(-1, 1)

                    with torch.no_grad():
                        obs_proc_all = self._observation_preprocessor[uid](observations[uid])
                        inputs_lp = {
                            "observations": obs_proc_all,
                            "states": self._state_preprocessor[uid](self._current_anchor_states[uid]),
                            "action_mask": self._dec_action_mask(observations[uid]),
                            "taken_actions": eff_flat.view(-1, 1),
                        }
                        _, oh = self.policies[uid].act(inputs_lp, role="policy")
                        stored_log_prob = oh["log_prob"]

                if self._hier and not self._is_dec.get(uid, False) and env_h_rec is not None:
                    _ex_mask_fn = getattr(env_h_rec, "expert_ctrl_mask", None)
                    if _ex_mask_fn is not None:
                        ex = _ex_mask_fn(uid)
                        if bool(ex.any()):
                            du = uid.split("/")[0]
                            executed = env_h_rec.actions[du].clamp(-1.0, 1.0)
                            stored_actions = torch.where(ex.view(-1, 1), executed, stored_actions)

                            rec_gauss = self._current_gaussian.get(uid)
                            if self.training and rec_gauss is not None:
                                mean_ex, std_ex = rec_gauss
                                act_ex = stored_actions.detach()
                                lp_ex = (
                                    torch.distributions.Normal(mean_ex, std_ex).log_prob(act_ex).sum(-1, keepdim=True)
                                )
                                stored_log_prob = torch.where(ex.view(-1, 1), lp_ex, stored_log_prob)

                if uid in self._dec_collectors and env_h_rec is not None:
                    du = uid.split("/", 1)[0]
                    done_uid = (terminated[uid] | truncated[uid]).view(-1)

                    newly_dead = learn_active[uid].view(-1) & ~learn_still_alive[du].view(-1)
                    collector = self._dec_collectors[uid]
                    collector.step(
                        boundary=env_h_rec._decision_boundary[du],
                        done=done_uid,
                        death=newly_dead,
                        reward=per_uid_reward[uid],
                        active=learn_active[uid],
                        anchor_switch=anchor_switch[uid],
                        sample_fields={
                            "observations": observations[uid],
                            "states": self._current_anchor_states.get(uid, states[self._state_anchor[uid]]),
                            "actions": stored_actions,
                            "log_prob": stored_log_prob,
                            "values": self._current_values[uid],
                            "active": learn_active[uid].view(-1, 1),
                            "team_active": learn_team_active[team].view(-1, 1),
                        },
                    )
                self.memories[uid].add_samples(
                    observations=observations[uid],
                    states=self._current_anchor_states.get(uid, states[self._state_anchor[uid]]),
                    actions=stored_actions,
                    rewards=per_uid_reward[uid],
                    terminated=terminated[uid],
                    truncated=truncated[uid],
                    log_prob=stored_log_prob,
                    values=self._current_values[uid],
                    active=learn_active[uid].view(-1, 1),
                    team_active=learn_team_active[team].view(-1, 1),
                    anchor_switch=anchor_switch[uid].view(-1, 1),
                )

    def _prepare_dec_trajectories(self, uid: str) -> list[dict[str, torch.Tensor]]:
        trajectories = self._dec_collectors[uid].pop_completed()

        _is_dec_rep = self._value_representative[self._value_team_layer(uid)] == uid
        _raw = ([], [], [])
        out = []
        for tr in trajectories:
            rewards = tr["reward"].view(-1, 1)
            values = tr["values"].view(-1, 1)
            zeros = torch.zeros_like(rewards, dtype=torch.bool)
            terminal = zeros.clone()
            terminal[-1] = True

            switch = tr.get("anchor_switch")
            if switch is not None:
                terminal = terminal | switch.view(-1, 1).bool()
            returns, advantages = compute_gae(
                rewards=rewards,
                terminated=terminal,
                truncated=zeros,
                values=values,
                last_values=torch.zeros_like(values[-1]),
                discount_factor=self.cfg.discount_factor[uid],
                lambda_coefficient=self.cfg.gae_lambda[uid],
            )
            if _is_dec_rep:
                _raw[0].append(returns.reshape(-1))
                _raw[1].append(advantages.reshape(-1))
                _raw[2].append(values.reshape(-1))
            tr["values"] = values
            tr["returns"] = returns
            tr["_adv_raw"] = advantages
            out.append(tr)
        if out:
            adv_cat = torch.cat([tr["_adv_raw"].view(-1, 1) for tr in out], dim=0)
            act_cat = torch.cat([tr["active"].view(-1) for tr in out], dim=0).bool()
            norm_cat = self._normalize_active_advantages(adv_cat, act_cat)
            ofs = 0
            for tr in out:
                k = tr["_adv_raw"].numel()
                tr["advantages"] = norm_cat[ofs : ofs + k].view_as(tr["_adv_raw"])
                ofs += k
                del tr["_adv_raw"]
        if _is_dec_rep and _raw[0]:
            self._log_gae(uid, torch.cat(_raw[0]), torch.cat(_raw[1]), torch.cat(_raw[2]))
        return out

    def _update_dec_group(self, group: list[str]) -> None:
        rep = group[0]
        policy = self.policies[rep]
        value = self.values[rep]
        team = self._team_of[rep]
        vtl = self._value_team_layer(rep)
        value_rep = self._value_representative[vtl]
        trajectories = {uid: self._prepare_dec_trajectories(uid) for uid in group}
        if not any(trajectories.values()):
            return
        frozen = any(uid in self.frozen_agents for uid in group)
        epochs = self.cfg.learning_epochs[rep]
        pol_total = val_total = ent_total = 0.0
        updates = 0
        kl_all = []

        policy_count = 0
        value_count = 0
        for uid in group:
            for tr in trajectories[uid]:
                policy_count += int(tr["active"].view(-1).bool().sum().item())
                if uid == value_rep:
                    value_count += int(tr["team_active"].view(-1).bool().sum().item())
        policy_trainable = policy_count > 0 and not frozen
        value_trainable = value_count > 0 and not self._roll_value_frozen(team)
        policy_count = max(policy_count, 1)
        value_count = max(value_count, 1)
        for epoch in range(epochs):
            kl_num = torch.zeros((), device=self.device)
            kl_count = 0
            pol_acc = ent_acc = val_acc = 0.0
            if policy_trainable:
                self.policy_optimizers[rep].zero_grad()
            if value_trainable:
                self.value_optimizers[vtl].zero_grad()
            for uid in group:
                parts = [
                    (tr["active"].view(-1).bool(), tr)
                    for tr in trajectories[uid]
                    if bool(tr["active"].view(-1).bool().any())
                ]
                if parts:
                    obs_raw = torch.cat([tr["observations"][m] for m, tr in parts], dim=0)
                    actions = torch.cat([tr["actions"][m] for m, tr in parts], dim=0)
                    old_lp = torch.cat([tr["log_prob"][m] for m, tr in parts], dim=0).view(-1, 1)
                    adv = torch.cat([tr["advantages"][m] for m, tr in parts], dim=0).view(-1, 1)
                    obs = self._observation_preprocessor[uid](obs_raw, train=(epoch == 0))
                    inputs = {
                        "observations": obs,
                        "taken_actions": actions.long(),
                        "action_mask": self._dec_action_mask(obs_raw),
                    }
                    _, outputs = policy.act(inputs, role="policy")
                    next_lp = outputs["log_prob"].view(-1, 1)
                    entropy = policy.get_entropy(role="policy")
                    log_ratio = next_lp - old_lp
                    ratio = log_ratio.exp()
                    clipped = torch.clip(ratio, 1 - self.cfg.ratio_clip[uid], 1 + self.cfg.ratio_clip[uid])
                    loss_rows = -torch.min(adv * ratio, adv * clipped).view(-1)
                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - log_ratio).view(-1)
                        kl_num = kl_num + approx_kl.sum()
                        kl_count += int(loss_rows.numel())
                    if policy_trainable:
                        slice_loss = loss_rows.sum() / policy_count
                        if self.cfg.entropy_loss_scale[uid]:
                            ent_term = -self.cfg.entropy_loss_scale[uid] * entropy.view(-1).sum() / policy_count
                            slice_loss = slice_loss + ent_term
                            ent_acc += float(ent_term.detach())
                        pol_acc += float((loss_rows.sum() / policy_count).detach())
                        self.scaler.scale(slice_loss).backward()
                if uid == value_rep and value_trainable:
                    vparts = [
                        (tr["team_active"].view(-1).bool(), tr)
                        for _m, tr in parts
                        if bool(tr["team_active"].view(-1).bool().any())
                    ]
                    if vparts:
                        inputs_v = {
                            "observations": self._observation_preprocessor[uid](
                                torch.cat([tr["observations"][m] for m, tr in vparts], dim=0), train=False
                            ),
                            "states": self._state_preprocessor[uid](
                                torch.cat([tr["states"][m] for m, tr in vparts], dim=0), train=(epoch == 0)
                            ),
                        }
                        predicted, _ = value.act(inputs_v, role="value")
                        old_values = self._value_preprocessor[uid](
                            torch.cat([tr["values"][m] for m, tr in vparts], dim=0), train=(epoch == 0)
                        )
                        targets = self._value_preprocessor[uid](
                            torch.cat([tr["returns"][m] for m, tr in vparts], dim=0), train=(epoch == 0)
                        )
                        if self.cfg.value_clip[uid] > 0:
                            predicted = old_values + torch.clip(
                                predicted - old_values, -self.cfg.value_clip[uid], self.cfg.value_clip[uid]
                            )
                        v_slice = self.cfg.value_loss_scale[uid] * (((predicted - targets) ** 2).sum() / value_count)
                        val_acc += float(v_slice.detach())
                        self.scaler.scale(v_slice).backward()
            batch_kl = (kl_num / max(kl_count, 1)) if kl_count > 0 else None

            if batch_kl is not None and self.cfg.kl_threshold[rep] and bool(batch_kl > self.cfg.kl_threshold[rep]):
                if policy_trainable:
                    self.policy_optimizers[rep].zero_grad()
                if value_trainable:
                    self.value_optimizers[vtl].zero_grad()
                break
            if policy_trainable:
                self.scaler.unscale_(self.policy_optimizers[rep])
                nn.utils.clip_grad_norm_(policy.parameters(), self.cfg.grad_norm_clip[rep])
                self.scaler.step(self.policy_optimizers[rep])
            if value_trainable:
                self.scaler.unscale_(self.value_optimizers[vtl])
                nn.utils.clip_grad_norm_(value.parameters(), self.cfg.grad_norm_clip[rep])
                self.scaler.step(self.value_optimizers[vtl])
            self.scaler.update()
            pol_total += pol_acc
            ent_total += ent_acc
            val_total += val_acc
            if batch_kl is not None:
                kl_all.append(batch_kl.detach())
            updates += 1
        if updates:
            self.track_data(f"Loss / Policy loss ({rep})", pol_total / updates)
            self.track_data(f"Loss / Value loss (team_{team}_dec)", val_total / updates)
            if self.cfg.entropy_loss_scale[rep]:
                self.track_data(f"Loss / Entropy loss ({rep})", ent_total / updates)
            if kl_all:
                self.track_data(f"Policy / Approximate KL divergence ({rep})", torch.stack(kl_all).mean().item())

            if self.schedulers.get(rep) is not None:
                if isinstance(self.schedulers[rep], KLAdaptiveLR):
                    if kl_all:
                        self.schedulers[rep].step(torch.stack(kl_all).mean().item())
                else:
                    self.schedulers[rep].step()

    def pre_interaction(self, *, timestep, timesteps):
        pass

    def _maybe_update_dec(self) -> None:
        if not self._dec_collectors:
            return
        self.enable_models_training_mode(True)
        if self._actor_mode == "team":
            for _gk, uids in self._actor_groups.items():
                if not self._is_dec.get(uids[0], False):
                    continue

                if any(u in self.frozen_agents for u in uids):
                    continue
                if self._dec_collectors[uids[0]].completed_trajectories >= self._dec_rollouts:
                    self._update_dec_group(uids)
        else:
            for uid in self.possible_agents:
                if not self._is_dec.get(uid, False):
                    continue
                if any(u in self.frozen_agents for u in (uid,)):
                    continue
                if self._dec_collectors[uid].completed_trajectories >= self._dec_rollouts:
                    self._update_dec_group([uid])
        self.enable_models_training_mode(False)

    def _anneal_expert_prob(self, timestep: int, timesteps: int) -> None:
        if self._expert_anneal_done:
            return
        env = getattr(self, "_hier_env", None)
        setter = getattr(env, "set_expert_curriculum_prob", None) if env is not None else None
        if setter is None:
            if not self._expert_anneal_warned:
                self._expert_anneal_warned = True
                print("[TAD][MAPPO][WARN] env 无 set_expert_curriculum_prob → 专家课程退火未生效（退化为恒 prob）")
            return
        span = max(1.0, self._expert_anneal_frac * float(timesteps))
        prob = max(0.0, 1.0 - float(timestep) / span)
        setter(prob)
        self._expert_prob_cur = prob
        if prob <= 0.0:
            self._expert_anneal_done = True
            print(f"[TAD][MAPPO] 专家课程退火完成（step={timestep} / {int(span)}）→ 接管概率 0，转入纯 on-policy")
        elif timestep % self._telemetry_interval == 0:
            self.track_data("Curriculum/expert_prob", prob)

    def post_interaction(self, *, timestep, timesteps):
        if self.training:
            self._rollout += 1
            update_train_takeover(self, timestep)
            update_roll_rotation(self, timestep)

            if self._expert_curriculum_on and self._expert_anneal_frac > 0.0:
                self._anneal_expert_prob(timestep, timesteps)

            if self._dec_collectors and timestep >= self.cfg.learning_starts:
                self._maybe_update_dec()
            if not self._rollout % self.cfg.rollouts and timestep >= self.cfg.learning_starts:
                self.enable_models_training_mode(True)
                if self._actor_mode == "team":
                    for _gk, uids in self._actor_groups.items():
                        layer = "dec" if self._is_dec.get(uids[0], False) else "ctrl"
                        if layer == "dec":
                            continue
                        self.update_group(group=uids)
                else:
                    for uid in self.possible_agents:
                        if self._is_dec.get(uid, False):
                            continue
                        self.update(timestep=timestep, timesteps=timesteps, uid=uid)
                self.enable_models_training_mode(False)

        super().post_interaction(timestep=timestep, timesteps=timesteps)

    def update(self, *, timestep, timesteps, uid):
        policy = self.policies[uid]
        value = self.values[uid]
        memory = self.memories[uid]
        team = self._team_of[uid]
        vtl = self._value_team_layer(uid)
        is_rep = self._value_representative[vtl] == uid

        with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
            inputs = {
                "observations": self._observation_preprocessor[uid](self._current_next_observations[uid]),
                "states": self._state_preprocessor[uid](
                    self._next_anchor_states.get(uid, self._current_next_states[self._state_anchor[uid]])
                ),
            }
            value.enable_training_mode(False)
            last_values, _ = value.act(inputs, role="value")
            value.enable_training_mode(True)
            last_values = self._value_preprocessor[uid](last_values, inverse=True)

        values = memory.get_tensor_by_name("values")
        returns, advantages = compute_gae(
            rewards=memory.get_tensor_by_name("rewards"),
            terminated=self._gae_terminated(memory),
            truncated=memory.get_tensor_by_name("truncated"),
            values=values,
            last_values=last_values,
            discount_factor=self.cfg.discount_factor[uid],
            lambda_coefficient=self.cfg.gae_lambda[uid],
        )
        if is_rep:
            self._log_gae(uid, returns, advantages, values)
        memory.set_tensor_by_name("values", self._value_preprocessor[uid](values, train=is_rep))
        memory.set_tensor_by_name("returns", self._value_preprocessor[uid](returns, train=is_rep))
        memory.set_tensor_by_name(
            "advantages", self._normalize_active_advantages(advantages, memory.get_tensor_by_name("active"))
        )

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0

        for epoch in range(self.cfg.learning_epochs[uid]):
            kl_divergences = []
            for (
                sampled_observations,
                sampled_states,
                sampled_actions,
                sampled_log_prob,
                sampled_values,
                sampled_returns,
                sampled_advantages,
                sampled_active,
                sampled_team_active,
            ) in memory.sample(
                names=self._tensors_names, batch_size=len(memory), mini_batches=self.cfg.mini_batches[uid]
            ):
                with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                    inputs = {
                        "observations": self._observation_preprocessor[uid](sampled_observations, train=not epoch),
                        "states": self._state_preprocessor[uid](sampled_states, train=not epoch and is_rep),
                    }
                    uid_discrete = self._is_dec.get(uid, self._discrete)
                    if uid_discrete:
                        inputs["action_mask"] = self._dec_action_mask(sampled_observations)
                    ta = sampled_actions.long() if uid_discrete else sampled_actions
                    _, outputs = policy.act({**inputs, "taken_actions": ta}, role="policy")
                    next_log_prob = outputs["log_prob"]

                    with torch.no_grad():
                        ratio = next_log_prob - sampled_log_prob
                        _am = sampled_active.view(-1).bool()
                        kl_divergence = self._masked_loss(((torch.exp(ratio) - 1) - ratio).view(-1), _am)
                        kl_divergences.append(kl_divergence)
                    if self.cfg.kl_threshold[uid] and kl_divergence > self.cfg.kl_threshold[uid]:
                        break

                    if self.cfg.entropy_loss_scale[uid]:
                        entropy_loss = -self.cfg.entropy_loss_scale[uid] * self._masked_loss(
                            policy.get_entropy(role="policy").view(-1), _am
                        )
                    else:
                        entropy_loss = 0

                    ratio = torch.exp(next_log_prob - sampled_log_prob)
                    surrogate = sampled_advantages * ratio
                    surrogate_clipped = sampled_advantages * torch.clip(
                        ratio, 1.0 - self.cfg.ratio_clip[uid], 1.0 + self.cfg.ratio_clip[uid]
                    )
                    policy_loss = -self._masked_loss(torch.min(surrogate, surrogate_clipped).view(-1), _am)

                    value_loss = torch.zeros((), device=self.device)
                    if is_rep:
                        predicted_values, _ = value.act(inputs, role="value")
                        if self.cfg.value_clip[uid] > 0:
                            predicted_values = sampled_values + torch.clip(
                                predicted_values - sampled_values,
                                min=-self.cfg.value_clip[uid],
                                max=self.cfg.value_clip[uid],
                            )
                        value_loss = self.cfg.value_loss_scale[uid] * self._masked_loss(
                            ((sampled_returns - predicted_values) ** 2).view(-1), sampled_team_active.view(-1).bool()
                        )

                frozen_agents = getattr(self, "frozen_agents", set())
                policy_trainable = uid not in frozen_agents
                value_trainable = is_rep and not self._roll_value_frozen(self._team_of[uid])
                if policy_trainable:
                    self.policy_optimizers[uid].zero_grad()
                if value_trainable:
                    self.value_optimizers[vtl].zero_grad()
                if policy_trainable and value_trainable:
                    loss = policy_loss + entropy_loss + value_loss
                elif policy_trainable:
                    loss = policy_loss + entropy_loss
                elif value_trainable:
                    loss = value_loss
                else:
                    loss = None
                if loss is not None:
                    self.scaler.scale(loss).backward()
                if policy_trainable and config.torch.is_distributed:
                    policy.reduce_parameters()
                if policy_trainable:
                    if self.cfg.grad_norm_clip[uid] > 0:
                        self.scaler.unscale_(self.policy_optimizers[uid])
                        nn.utils.clip_grad_norm_(policy.parameters(), self.cfg.grad_norm_clip[uid])
                    self.scaler.step(self.policy_optimizers[uid])
                if value_trainable and self.cfg.grad_norm_clip[uid] > 0:
                    self.scaler.unscale_(self.value_optimizers[vtl])
                    nn.utils.clip_grad_norm_(value.parameters(), self.cfg.grad_norm_clip[uid])
                if value_trainable:
                    self.scaler.step(self.value_optimizers[vtl])
                if loss is not None:
                    self.scaler.update()

                cumulative_policy_loss += policy_loss.item()
                if is_rep:
                    cumulative_value_loss += value_loss.item()
                if self.cfg.entropy_loss_scale[uid]:
                    cumulative_entropy_loss += entropy_loss.item()

            if uid not in getattr(self, "frozen_agents", set()) and self.schedulers[uid]:
                if isinstance(self.schedulers[uid], KLAdaptiveLR):
                    kl = torch.tensor(kl_divergences, device=self.device).mean()
                    if config.torch.is_distributed:
                        torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                        kl /= config.torch.world_size
                    self.schedulers[uid].step(kl.item())
                else:
                    self.schedulers[uid].step()

        denom = self.cfg.learning_epochs[uid] * self.cfg.mini_batches[uid]
        self.track_data(f"Loss / Policy loss ({uid})", cumulative_policy_loss / denom)
        if is_rep:
            self.track_data(f"Loss / Value loss (team_{team}_{self._layer_of(uid)})", cumulative_value_loss / denom)
        if self.cfg.entropy_loss_scale[uid]:
            self.track_data(f"Loss / Entropy loss ({uid})", cumulative_entropy_loss / denom)

        if self._discrete:
            self.track_data(f"Policy / Entropy ({uid})", policy.get_entropy(role="policy").mean().item())
        else:
            self.track_data(
                f"Policy / Standard deviation ({uid})", policy.distribution(role="policy").stddev.mean().item()
            )
        if kl_divergences:
            self.track_data(f"Policy / Approximate KL divergence ({uid})", torch.stack(kl_divergences).mean().item())
        if self.schedulers[uid]:
            self.track_data(f"Learning / Learning rate ({uid})", self.schedulers[uid].get_last_lr()[0])

    def update_group(self, *, group):
        """Update the policies and shared critic for a training group."""

        rep = group[0]
        policy = self.policies[rep]
        team = self._team_of[rep]
        vtl = self._value_team_layer(rep)
        value = self.values[rep]
        team_rep = self._value_representative[vtl]
        has_team_rep = team_rep in group
        is_rep = has_team_rep
        frozen = getattr(self, "frozen_agents", set())
        policy_trainable = not any(u in frozen for u in group)

        n_epochs = self.cfg.learning_epochs[rep]
        n_mb = self.cfg.mini_batches[rep]

        for uid in group:
            memory = self.memories[uid]
            with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                inputs = {
                    "observations": self._observation_preprocessor[uid](self._current_next_observations[uid]),
                    "states": self._state_preprocessor[uid](
                        self._next_anchor_states.get(uid, self._current_next_states[self._state_anchor[uid]])
                    ),
                }
                value.enable_training_mode(False)
                last_values, _ = value.act(inputs, role="value")
                value.enable_training_mode(True)
                last_values = self._value_preprocessor[uid](last_values, inverse=True)
            values = memory.get_tensor_by_name("values")
            returns, advantages = compute_gae(
                rewards=memory.get_tensor_by_name("rewards"),
                terminated=self._gae_terminated(memory),
                truncated=memory.get_tensor_by_name("truncated"),
                values=values,
                last_values=last_values,
                discount_factor=self.cfg.discount_factor[uid],
                lambda_coefficient=self.cfg.gae_lambda[uid],
            )
            if uid == team_rep:
                self._log_gae(uid, returns, advantages, values)
            uid_is_team_rep = uid == team_rep
            memory.set_tensor_by_name("values", self._value_preprocessor[uid](values, train=uid_is_team_rep))
            memory.set_tensor_by_name("returns", self._value_preprocessor[uid](returns, train=uid_is_team_rep))
            memory.set_tensor_by_name(
                "advantages", self._normalize_active_advantages(advantages, memory.get_tensor_by_name("active"))
            )

        cumulative_policy_loss = 0.0
        cumulative_entropy_loss = 0.0
        cumulative_value_loss = 0.0
        kl_divergences_all = []

        for epoch in range(n_epochs):
            kl_divergences_all = []

            mbatches = {
                uid: list(
                    self.memories[uid].sample(
                        names=self._tensors_names, batch_size=len(self.memories[uid]), mini_batches=n_mb
                    )
                )
                for uid in group
            }
            for k in range(n_mb):
                rep_inputs = None
                with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                    policy_loss_sum = torch.zeros((), device=self.device)
                    entropy_loss_sum = torch.zeros((), device=self.device)
                    kl_vals = []
                    if policy_trainable:
                        for uid in group:
                            (so, ss, sa, slp, sv, sr, sadv, sactive, steam_active) = mbatches[uid][k]
                            inputs = {
                                "observations": self._observation_preprocessor[uid](so, train=not epoch),
                                "states": self._state_preprocessor[uid](ss, train=(not epoch) and uid == team_rep),
                            }
                            if uid == team_rep:
                                rep_inputs = inputs
                            uid_discrete = self._is_dec.get(uid, self._discrete)
                            if uid_discrete:
                                inputs["action_mask"] = self._dec_action_mask(so)
                            ta = sa.long() if uid_discrete else sa
                            _, outputs = policy.act({**inputs, "taken_actions": ta}, role="policy")
                            next_log_prob = outputs["log_prob"]
                            _active_rows = sactive.view(-1).bool()
                            with torch.no_grad():
                                ratio = next_log_prob - slp
                                kl_vals.append(
                                    self._masked_loss(((torch.exp(ratio) - 1) - ratio).view(-1), _active_rows)
                                )
                            if self.cfg.entropy_loss_scale[uid]:
                                entropy_loss_sum = entropy_loss_sum - self.cfg.entropy_loss_scale[
                                    uid
                                ] * self._masked_loss(policy.get_entropy(role="policy").view(-1), _active_rows)
                            ratio_e = torch.exp(next_log_prob - slp)
                            surrogate = sadv * ratio_e
                            surrogate_clipped = sadv * torch.clip(
                                ratio_e, 1.0 - self.cfg.ratio_clip[uid], 1.0 + self.cfg.ratio_clip[uid]
                            )
                            policy_loss_sum = policy_loss_sum - self._masked_loss(
                                torch.min(surrogate, surrogate_clipped).view(-1), _active_rows
                            )
                        policy_loss = policy_loss_sum / len(group)
                        entropy_loss = entropy_loss_sum / len(group)
                    else:
                        policy_loss = torch.zeros((), device=self.device)
                        entropy_loss = torch.zeros((), device=self.device)
                    kl_divergence = torch.stack(kl_vals).mean() if kl_vals else torch.zeros((), device=self.device)
                    kl_divergences_all.append(kl_divergence)

                    if policy_trainable and self.cfg.kl_threshold[rep] and kl_divergence > self.cfg.kl_threshold[rep]:
                        break

                    value_loss = torch.zeros((), device=self.device)
                    if has_team_rep:
                        (so, ss, sa, slp, sv, sr, sadv, sactive, steam_active) = mbatches[team_rep][k]
                        if rep_inputs is None:
                            rep_inputs = {
                                "observations": self._observation_preprocessor[team_rep](so, train=not epoch),
                                "states": self._state_preprocessor[team_rep](ss, train=not epoch),
                            }
                        predicted_values, _ = value.act(rep_inputs, role="value")
                        if self.cfg.value_clip[rep] > 0:
                            predicted_values = sv + torch.clip(
                                predicted_values - sv, min=-self.cfg.value_clip[rep], max=self.cfg.value_clip[rep]
                            )
                        value_loss = self.cfg.value_loss_scale[rep] * self._masked_loss(
                            ((sr - predicted_values) ** 2).view(-1), steam_active.view(-1).bool()
                        )

                value_trainable = has_team_rep and not self._roll_value_frozen(team)
                if policy_trainable:
                    self.policy_optimizers[rep].zero_grad()
                if value_trainable:
                    self.value_optimizers[vtl].zero_grad()
                if policy_trainable and value_trainable:
                    loss = policy_loss + entropy_loss + value_loss
                elif policy_trainable:
                    loss = policy_loss + entropy_loss
                elif value_trainable:
                    loss = value_loss
                else:
                    loss = None
                if loss is not None:
                    self.scaler.scale(loss).backward()
                if policy_trainable and config.torch.is_distributed:
                    policy.reduce_parameters()
                if policy_trainable:
                    if self.cfg.grad_norm_clip[rep] > 0:
                        self.scaler.unscale_(self.policy_optimizers[rep])
                        nn.utils.clip_grad_norm_(policy.parameters(), self.cfg.grad_norm_clip[rep])
                    self.scaler.step(self.policy_optimizers[rep])
                if value_trainable and self.cfg.grad_norm_clip[rep] > 0:
                    self.scaler.unscale_(self.value_optimizers[vtl])
                    nn.utils.clip_grad_norm_(value.parameters(), self.cfg.grad_norm_clip[rep])
                if value_trainable:
                    self.scaler.step(self.value_optimizers[vtl])
                if loss is not None:
                    self.scaler.update()

                cumulative_policy_loss += policy_loss.item() if policy_trainable else 0.0
                if has_team_rep:
                    cumulative_value_loss += value_loss.item()
                if self.cfg.entropy_loss_scale[rep] and policy_trainable:
                    cumulative_entropy_loss += entropy_loss.item()

            if policy_trainable and self.schedulers[rep]:
                if isinstance(self.schedulers[rep], KLAdaptiveLR):
                    kl = torch.tensor(kl_divergences_all, device=self.device).mean()
                    if config.torch.is_distributed:
                        torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                        kl /= config.torch.world_size
                    self.schedulers[rep].step(kl.item())
                else:
                    self.schedulers[rep].step()

        denom = n_epochs * n_mb
        self.track_data(f"Loss / Policy loss ({rep})", cumulative_policy_loss / denom)
        if is_rep:
            self.track_data(f"Loss / Value loss (team_{team}_{self._layer_of(rep)})", cumulative_value_loss / denom)
        if self.cfg.entropy_loss_scale[rep]:
            self.track_data(f"Loss / Entropy loss ({rep})", cumulative_entropy_loss / denom)
        if self._is_dec.get(rep, self._discrete):
            self.track_data(f"Policy / Entropy ({rep})", policy.get_entropy(role="policy").mean().item())
        else:
            self.track_data(
                f"Policy / Standard deviation ({rep})", policy.distribution(role="policy").stddev.mean().item()
            )
        if kl_divergences_all:
            self.track_data(
                f"Policy / Approximate KL divergence ({rep})", torch.stack(kl_divergences_all).mean().item()
            )
        if self.schedulers[rep]:
            self.track_data(f"Learning / Learning rate ({rep})", self.schedulers[rep].get_last_lr()[0])
