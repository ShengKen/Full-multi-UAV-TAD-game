from __future__ import annotations

import dataclasses

import gymnasium
import torch
import torch.nn as nn

from skrl.multi_agents.torch import MultiAgent

from ..maddpg.maddpg import ActorMLP, CriticMLP, _IdentityPreprocessor
from ..maddpg.maddpg_cfg import MADDPG_CFG
from ..replay_buffer import HierReplayBuffer, HierReplayCommonStore
from ..resume_checkpoint import register_architecture_fingerprint, register_training_progress
from ..running_standard_scaler import RunningStandardScaler
from ..team_utils import aggregate_team_reward, masked_mean
from .flat_common import flat_noise_scale


class FlatTD3(MultiAgent):
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
        team_reward_aggr: str = "mean",
    ):
        self._cfg = dict(cfg)
        self._gamma = float(cfg.get("discount_factor_control", cfg.get("discount_factor", 0.995)))
        self._time_limit_bootstrap = bool(cfg.get("time_limit_bootstrap", False))
        if self._time_limit_bootstrap:
            raise ValueError("flattd3 仅支持 time_limit_bootstrap=false（自然超时=有限任务真终局）")
        self._polyak = float(cfg.get("polyak", 0.995))
        self._grad_norm_clip = float(cfg.get("grad_norm_clip", 1.0))
        self._learning_starts = int(cfg.get("learning_starts", 1000))
        self._update_every = max(1, int(cfg.get("update_every", 1)))
        self._gradient_steps = int(cfg.get("gradient_steps", 1))
        self._telemetry_interval = max(1, int(cfg.get("telemetry_interval", 50)))
        self._random_timesteps = int(cfg.get("random_timesteps", 0))
        self._policy_lr = float(cfg.get("policy_lr", 3e-4))
        self._critic_lr = float(cfg.get("critic_lr", 3e-4))
        self._noise_std = float(cfg.get("exploration_noise_std", 0.3))
        self._noise_scheduler = bool(cfg.get("exploration_scheduler", True))
        self._noise_floor = min(1.0, max(0.0, float(cfg.get("exploration_noise_floor", 0.0))))
        self._noise_halflife = float(cfg.get("exploration_noise_halflife", 0.0) or 0.0)
        self._batch_size = int(cfg.get("batch_size", 256))
        self._replay_steps = int(cfg.get("replay_steps", 0))
        self._replay_size = int(cfg.get("replay_size", 1000000))
        self._replay_device = str(cfg.get("replay_device", "cpu")).lower()
        self._n_step = max(1, int(cfg.get("n_step", 1)))
        self._replay_terminal_mix = min(max(float(cfg.get("replay_terminal_mix", 0.0)), 0.0), 1.0)
        self._replay_terminal_thresh = float(cfg.get("replay_terminal_thresh", 20.0))

        self._matd3_twin = bool(cfg.get("matd3_twin", True))
        self._matd3_smooth = bool(cfg.get("matd3_smooth", True))
        self._matd3_smooth_std = float(cfg.get("matd3_smooth_std", 0.2))
        self._matd3_smooth_clip = float(cfg.get("matd3_smooth_clip", 0.5))
        self._matd3_delay = bool(cfg.get("matd3_delay", True))
        self._matd3_policy_delay = max(1, int(cfg.get("matd3_policy_delay", 2)))
        self._tick = 0

        self._flat_actor_hidden = list(cfg.get("flat_actor_hidden", [440, 440, 440]))
        self._critic_hidden = list(cfg.get("critic_hidden", [512, 512, 512]))
        self._activation = str(cfg.get("activation", "relu"))
        self._use_layer_norm = bool(cfg.get("use_layer_norm", False))
        self._normalize = bool(cfg.get("normalize", True))
        self._norm_eps = float(cfg.get("norm_epsilon", 1e-8))
        self._norm_clip = float(cfg.get("norm_clip_threshold", 5.0))

        self._team_of = team_of
        self._team_reward_aggr = team_reward_aggr
        self._team_members: dict[str, list[str]] = {}
        for uid in possible_agents:
            self._team_members.setdefault(team_of[uid], []).append(uid)
        self._team_representative = {t: m[0] for t, m in self._team_members.items()}
        self._team_drone_count = {t: len(m) for t, m in self._team_members.items()}
        self._obs_dim_per_agent = {a: observation_spaces[a].shape[0] for a in possible_agents}
        self._state_dim = state_space.shape[0]

        self._drone_uids = list(cfg.get("all_uids", []))
        self._drone_index = {u: i for i, u in enumerate(self._drone_uids)}
        self._M = len(self._drone_uids)
        self._agent_slot = {a: (self._drone_index[a], self._drone_index[a] * 4, 4) for a in possible_agents}
        self._total_act_dim = self._M * 4
        self.frozen_agents: set[str] = set()
        self._tensor_track: dict[str, list[torch.Tensor]] = {}
        self._total_timesteps = None

        known = {f.name for f in dataclasses.fields(MADDPG_CFG)}
        cfg_clean = {k: v for k, v in cfg.items() if k in known}
        state_spaces = {a: state_space for a in possible_agents}
        models = {a: {} for a in possible_agents}
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

        self.actors: dict[str, nn.Module] = {}
        self.target_actors: dict[str, nn.Module] = {}
        self.actor_optimizers: dict[str, torch.optim.Optimizer] = {}
        self.critics: dict[str, nn.Module] = {}
        self.target_critics: dict[str, nn.Module] = {}
        self.critics2: dict[str, nn.Module] = {}
        self.target_critics2: dict[str, nn.Module] = {}
        self.critic_optimizers: dict[str, torch.optim.Optimizer] = {}
        self.critic_optimizers2: dict[str, torch.optim.Optimizer] = {}
        self._identity_preprocessor = _IdentityPreprocessor()
        for team, members in self._team_members.items():
            rep = members[0]
            a = ActorMLP(
                self._obs_dim_per_agent[rep], 4, self._flat_actor_hidden, self._activation, self._use_layer_norm
            ).to(self.device)
            ta = ActorMLP(
                self._obs_dim_per_agent[rep], 4, self._flat_actor_hidden, self._activation, self._use_layer_norm
            ).to(self.device)
            ta.load_state_dict(a.state_dict())
            for p in ta.parameters():
                p.requires_grad = False
            self.actors[team] = a
            self.target_actors[team] = ta
            self.actor_optimizers[team] = torch.optim.Adam(a.parameters(), lr=self._policy_lr)

            self.checkpoint_modules[rep]["policy"] = a
            self.checkpoint_modules[rep]["target_policy"] = ta
            self.checkpoint_modules[rep]["actor_optimizer"] = self.actor_optimizers[team]

            c1 = CriticMLP(
                self._state_dim, self._total_act_dim, self._critic_hidden, self._activation, self._use_layer_norm
            ).to(self.device)
            tc1 = CriticMLP(
                self._state_dim, self._total_act_dim, self._critic_hidden, self._activation, self._use_layer_norm
            ).to(self.device)
            tc1.load_state_dict(c1.state_dict())
            for p in tc1.parameters():
                p.requires_grad = False
            self.critics[team] = c1
            self.target_critics[team] = tc1
            self.critic_optimizers[team] = torch.optim.Adam(c1.parameters(), lr=self._critic_lr)
            self.checkpoint_modules[rep]["critic"] = c1
            self.checkpoint_modules[rep]["target_critic"] = tc1
            self.checkpoint_modules[rep]["critic_optimizer"] = self.critic_optimizers[team]
            if self._matd3_twin:
                c2 = CriticMLP(
                    self._state_dim, self._total_act_dim, self._critic_hidden, self._activation, self._use_layer_norm
                ).to(self.device)
                tc2 = CriticMLP(
                    self._state_dim, self._total_act_dim, self._critic_hidden, self._activation, self._use_layer_norm
                ).to(self.device)
                tc2.load_state_dict(c2.state_dict())
                for p in tc2.parameters():
                    p.requires_grad = False
                self.critics2[team] = c2
                self.target_critics2[team] = tc2
                self.critic_optimizers2[team] = torch.optim.Adam(c2.parameters(), lr=self._critic_lr)
                self.checkpoint_modules[rep]["critic2"] = c2
                self.checkpoint_modules[rep]["target_critic2"] = tc2
                self.checkpoint_modules[rep]["critic2_optimizer"] = self.critic_optimizers2[team]

        if self._normalize:
            self._obs_preprocessor = {
                a: RunningStandardScaler(
                    self._obs_dim_per_agent[a],
                    epsilon=self._norm_eps,
                    clip_threshold=self._norm_clip,
                    device=self.device,
                ).to(self.device)
                for a in possible_agents
            }
            self._state_preprocessor = {
                self._team_representative[t]: self._identity_preprocessor for t in self._team_members
            }
            for a in possible_agents:
                self.checkpoint_modules[a]["observation_preprocessor"] = self._obs_preprocessor[a]
            for t, rep in self._team_representative.items():
                self.checkpoint_modules[rep]["state_preprocessor"] = self._state_preprocessor[rep]
        else:
            self._obs_preprocessor = {a: self._identity_preprocessor for a in possible_agents}
            self._state_preprocessor = {
                self._team_representative[t]: self._identity_preprocessor for t in self._team_members
            }

        agent_obs_dims = {a: self._obs_dim_per_agent[a] for a in possible_agents}
        _capacity = self._replay_steps * num_envs if self._replay_steps > 0 else self._replay_size
        _store_device = "cpu" if self._replay_device == "cpu" else self.device
        self._replay_common = HierReplayCommonStore(
            _capacity, num_envs, self._total_act_dim, agent_obs_dims, self.device, store_device=_store_device
        )
        self.replays: dict[str, HierReplayBuffer] = {}
        for team in self._team_members:
            self.replays[team] = HierReplayBuffer(
                _capacity,
                num_envs,
                self._state_dim,
                self._total_act_dim,
                agent_obs_dims,
                self.device,
                store_device=_store_device,
                common_store=self._replay_common,
                n_step=self._n_step,
                n_step_discount=self._gamma,
                priority_mix=self._replay_terminal_mix,
            )
        self._print_replay_memory_estimate(_capacity, agent_obs_dims, num_envs)

        self._architecture_fingerprint = {
            "format": "tad_flat_td3",
            "requires_fingerprint": True,
            "flat_obs": str(cfg.get("flat_obs", "pool")),
            "flat_actor_hidden": list(self._flat_actor_hidden),
            "critic_state": "token",
            "matd3_twin": self._matd3_twin,
            "matd3_smooth": self._matd3_smooth,
            "matd3_smooth_std": self._matd3_smooth_std,
            "matd3_smooth_clip": self._matd3_smooth_clip,
            "matd3_delay": self._matd3_delay,
            "matd3_policy_delay": self._matd3_policy_delay,
            "gamma_control": self._gamma,
            "n_step": self._n_step,
            "spaces_dims": {
                a: (
                    int(self.observation_spaces[a].shape[0]),
                    int(self.state_spaces[a].shape[0]),
                    int(self.action_spaces[a].shape[0]),
                )
                for a in possible_agents
            },
        }
        register_architecture_fingerprint(self, self._architecture_fingerprint)
        register_training_progress(self)
        print(
            f"[TAD][build] FLAT-TD3：teams={list(self._team_members)}  actor_hidden={self._flat_actor_hidden}  "
            f"twinQ(state {self._state_dim} + joint {self._total_act_dim}={self._state_dim + self._total_act_dim})  "
            f"hidden={self._critic_hidden}  n_step={self._n_step}  γ={self._gamma}  "
            f"twin={self._matd3_twin} smooth={self._matd3_smooth}(σ={self._matd3_smooth_std},"
            f"c={self._matd3_smooth_clip}) delay={self._matd3_delay}(d={self._matd3_policy_delay})"
        )

    def _print_replay_memory_estimate(self, capacity: int, agent_obs_dims: dict, num_envs: int) -> None:
        common_bytes = capacity * (sum(agent_obs_dims.values()) * 2 + self._total_act_dim) * 4 + capacity * (
            2 + 2 * len(agent_obs_dims)
        )
        team_bytes = capacity * (self._state_dim * 2 + 4) * 4 * len(self._team_members)
        total = common_bytes + team_bytes
        print(
            f"[TAD][flat] replay 内存估算：capacity={capacity} 行（replay_steps×num_envs）"
            f"≈{total / 1024**3:.1f} GiB（store={self._replay_device}；"
            f"obs Σdim={sum(agent_obs_dims.values())} state={self._state_dim} joint={self._total_act_dim}）"
        )

    def _eval_row_mask(self, ref: torch.Tensor) -> torch.Tensor | None:
        m = getattr(self, "_eval_env_mask", None)
        if m is None:
            return None
        m = m.to(device=ref.device, dtype=torch.bool).view(-1)
        return m if m.numel() == ref.shape[0] else None

    def _soft_update(self, target: nn.Module, source: nn.Module) -> None:
        with torch.no_grad():
            tps, sps = list(target.parameters()), list(source.parameters())
            torch._foreach_mul_(tps, self._polyak)
            torch._foreach_add_(tps, sps, alpha=1.0 - self._polyak)

    def act(self, observations, states, *, timestep, timesteps):
        self._total_timesteps = timesteps
        actions = {}
        noise_scale = (
            flat_noise_scale(
                timestep,
                timesteps,
                scheduler=self._noise_scheduler,
                floor=self._noise_floor,
                halflife=self._noise_halflife,
            )
            if self.training
            else 1.0
        )
        warmup_steps = getattr(self, "_warmup_steps", 0)
        warmup_env = getattr(self, "_warmup_env", None)
        warmup_mask = None
        if warmup_steps > 0 and warmup_env is not None:
            warmup_mask = warmup_env.episode_length_buf < warmup_steps
        ref = observations[self.possible_agents[0]]
        eval_mask = self._eval_row_mask(ref)
        with torch.no_grad():
            for team, members in self._team_members.items():
                obs_n = {a: self._obs_preprocessor[a](observations[a]) for a in members}
                if len(members) > 1:
                    stk = torch.stack([obs_n[a] for a in members], dim=0)
                    out = self.actors[team](stk)
                    a_map = {a: out[i] for i, a in enumerate(members)}
                else:
                    a_map = {a: self.actors[team](obs_n[a]) for a in members}

                if eval_mask is not None:
                    idx = eval_mask.nonzero(as_tuple=False).view(-1)
                    if idx.numel() > 0:
                        src = (
                            self.target_actors[team]
                            if str(getattr(self, "_test_actor", "target")).lower() == "target"
                            else self.actors[team]
                        )
                        if idx.numel() < obs_n[members[0]].shape[0]:
                            det = src(torch.stack([obs_n[a][idx] for a in members], dim=0))
                            for i, a in enumerate(members):
                                a_map[a] = a_map[a].clone()
                                a_map[a][idx] = det[i]
                        else:
                            det = src(torch.stack([obs_n[a] for a in members], dim=0))
                            a_map = {a: det[i] for i, a in enumerate(members)}
                for a in members:
                    act = a_map[a]
                    if self.training and timestep < self._random_timesteps:
                        act = torch.zeros_like(act).uniform_(-1.0, 1.0)
                    elif self.training:
                        act = torch.clamp(act + torch.randn_like(act) * self._noise_std * noise_scale, -1.0, 1.0)
                    if warmup_mask is not None and bool(warmup_mask.any()):
                        ctrl_fn = getattr(warmup_env, "compute_warmup_stabilize_action", None)
                        ctrl = ctrl_fn(a) if ctrl_fn is not None else torch.zeros_like(act)
                        act = torch.where(warmup_mask.unsqueeze(-1), ctrl, act)
                    actions[a] = act
        return actions, {}

    def pre_interaction(self, *, timestep, timesteps):
        pass

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
        term = (term_t | trunc_t).bool().view(-1)

        td_done = term.clone()
        joint_ctrl = env.temporal_transition_info()["executed_joint_ctrl"] if env is not None else None
        if joint_ctrl is None:
            raise RuntimeError("flattd3 需要 agent._hier_env（train_tad 注入）读 executed_joint_ctrl")

        was_alive = (infos.get("agent_was_alive") if isinstance(infos, dict) else None) or {}
        n = joint_ctrl.shape[0]
        active_pa = {
            a: (
                was_alive[a].to(self.device).bool().view(-1)
                if a in was_alive
                else torch.ones(n, dtype=torch.bool, device=self.device)
            )
            for a in self.possible_agents
        }
        if _eval_m is not None:
            active_pa = {a: (m & ~_eval_m) for a, m in active_pa.items()}

        if self._normalize:
            for a in self.possible_agents:
                _o = observations[a]
                self._obs_preprocessor[a](_o if _eval_m is None else _o[~_eval_m], train=True)

        self._replay_common.add(
            joint_action=joint_ctrl,
            terminated=term.float(),
            obs_per_agent=observations,
            next_obs_per_agent=next_observations,
            decision_boundary_per_agent=None,
            active_per_agent=active_pa,
            boot_terminated=td_done.float(),
        )

        states_is_dict = isinstance(states, dict)
        for team, members in self._team_members.items():
            team_reward = aggregate_team_reward(
                rewards, infos, members, aggr=self._team_reward_aggr, nominal_n=self._team_drone_count[team], team=team
            )
            rep = self._team_representative[team]
            state_t = states[rep] if states_is_dict else states
            next_state_t = next_states[rep] if isinstance(next_states, dict) else next_states
            if self._normalize:
                self._state_preprocessor[rep](state_t if _eval_m is None else state_t[~_eval_m], train=True)

            learn_row = torch.ones(n, dtype=torch.bool, device=self.device) if _eval_m is None else ~_eval_m
            self.replays[team].add(
                state=state_t,
                next_state=next_state_t,
                reward=team_reward,
                team_active=learn_row,
                priority=(
                    None
                    if self._replay_terminal_mix <= 0.0
                    else ((team_reward.abs() >= self._replay_terminal_thresh) & learn_row)
                ),
            )
            if timestep % self._telemetry_interval == 0:
                self.track_data(
                    f"Reward / shaped (team_{team})",
                    team_reward.float().mean() if _eval_m is None else team_reward.float()[~_eval_m].mean(),
                )

    def post_interaction(self, *, timestep, timesteps):
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
        actor_due = (not self._matd3_delay) or (self._tick % self._matd3_policy_delay == 0)
        self._tick += 1
        frozen = getattr(self, "frozen_agents", set())
        for team, members in self._team_members.items():
            buf = self.replays[team]
            if len(buf) < self._batch_size:
                continue
            batch = buf.sample(self._batch_size)
            rep = self._team_representative[team]
            s = self._state_preprocessor[rep](batch["state"])
            ns = self._state_preprocessor[rep](batch["next_state"])
            ja = batch["joint_action"]
            r = batch["reward"]
            done = batch.get("boot_terminated", batch["terminated"]).bool()
            _g = batch.get("nstep_gamma")
            g_eff = _g if _g is not None else self._gamma
            obs_pa = batch["obs_per_agent"]
            nobs_pa = batch["next_obs_per_agent"]
            active_pa = batch.get("active_per_agent", {})
            learn_row = batch.get("team_active", torch.ones_like(done)).bool()
            is_w = batch.get("is_weight")

            with torch.no_grad():
                next_ja = ja.clone()
                for a in self.possible_agents:
                    _, start, _ = self._agent_slot[a]
                    nobs_i = self._obs_preprocessor[a](nobs_pa[a])
                    na = self.target_actors[self._team_of[a]](nobs_i)
                    if self._matd3_smooth:
                        noise = (torch.randn_like(na) * self._matd3_smooth_std).clamp(
                            -self._matd3_smooth_clip, self._matd3_smooth_clip
                        )
                        na = (na + noise).clamp(-1.0, 1.0)
                    act_i = active_pa.get(a, torch.ones_like(done)).bool().unsqueeze(-1)
                    next_ja[:, start : start + 4] = torch.where(act_i, na, next_ja[:, start : start + 4])
                q1t = self.target_critics[team](ns, next_ja).squeeze(-1)
                if self._matd3_twin:
                    q2t = self.target_critics2[team](ns, next_ja).squeeze(-1)
                    q_min = torch.min(q1t, q2t)
                else:
                    q_min = q1t
                target = r + g_eff * (1.0 - done.float()) * q_min

            q1 = self.critics[team](s, ja).squeeze(-1)
            sq = (q1 - target) ** 2
            if is_w is not None:
                sq = sq * is_w
            critic_loss = masked_mean(sq, learn_row, dim=tuple(range(sq.ndim)))
            self.critic_optimizers[team].zero_grad()
            critic_loss.backward()
            if self._grad_norm_clip > 0:
                nn.utils.clip_grad_norm_(self.critics[team].parameters(), self._grad_norm_clip)
            self.critic_optimizers[team].step()
            if self._matd3_twin:
                q2 = self.critics2[team](s, ja).squeeze(-1)
                sq2 = (q2 - target) ** 2
                if is_w is not None:
                    sq2 = sq2 * is_w
                critic2_loss = masked_mean(sq2, learn_row, dim=tuple(range(sq2.ndim)))
                self.critic_optimizers2[team].zero_grad()
                critic2_loss.backward()
                if self._grad_norm_clip > 0:
                    nn.utils.clip_grad_norm_(self.critics2[team].parameters(), self._grad_norm_clip)
                self.critic_optimizers2[team].step()
                self.track_data(f"Critic / twin gap (team_{team})", (q1 - q2).detach().abs().mean())

            team_frozen = any(a in frozen for a in members)
            if actor_due and not team_frozen:
                opt = self.actor_optimizers[team]
                opt.zero_grad()
                total_actor_loss = 0.0
                for a in members:
                    _, start, _ = self._agent_slot[a]
                    obs_i = self._obs_preprocessor[a](obs_pa[a])
                    a_i = self.actors[team](obs_i)
                    act_i = active_pa.get(a, torch.ones_like(done)).bool().unsqueeze(-1)
                    ja_grad = ja.clone()
                    ja_grad[:, start : start + 4] = torch.where(act_i, a_i, ja[:, start : start + 4])
                    q = self.critics[team](s, ja_grad).squeeze(-1)
                    actor_loss = -masked_mean(q, act_i.squeeze(-1), dim=tuple(range(q.ndim))) / len(members)
                    actor_loss.backward()
                    total_actor_loss = total_actor_loss + actor_loss.detach().double()
                if self._grad_norm_clip > 0:
                    nn.utils.clip_grad_norm_(self.actors[team].parameters(), self._grad_norm_clip)
                opt.step()
                self.track_data(f"Loss / Actor loss (team_{team})", total_actor_loss)
                self._soft_update(self.target_actors[team], self.actors[team])

            self._soft_update(self.target_critics[team], self.critics[team])
            if self._matd3_twin:
                self._soft_update(self.target_critics2[team], self.critics2[team])

            if timestep % self._telemetry_interval == 0:
                self.track_data(f"Loss / Critic loss (team_{team})", critic_loss)
                self.track_data(f"Q / mean (team_{team})", q1.mean())
                self.track_data(f"Q / max (team_{team})", q1.max())
                if self._replay_terminal_mix > 0.0:
                    self.track_data(f"Replay / terminal_frac (team_{team})", buf.priority[: len(buf)].float().mean())
