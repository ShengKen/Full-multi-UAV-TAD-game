from __future__ import annotations

import gymnasium
import torch
import torch.nn as nn

from skrl.multi_agents.torch import MultiAgent

from ..maddpg.maddpg import _IdentityPreprocessor
from ..resume_checkpoint import register_architecture_fingerprint, register_training_progress
from ..running_standard_scaler import RunningStandardScaler
from ..team_utils import aggregate_team_reward
from ..mappo.mappo import compute_gae
from .flat_common import FlatPPO_CFG, TanhGaussianActor, ValueMLP


class FlatPPO(MultiAgent):
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
        self._rollouts = max(1, int(cfg.get("rollouts", 512)))
        self._epochs = max(1, int(cfg.get("learning_epochs", 3)))
        self._mini_batches = max(1, int(cfg.get("mini_batches", 8)))
        self._gamma = float(cfg.get("discount_factor_control", cfg.get("discount_factor", 0.995)))
        self._lam = float(cfg.get("gae_lambda", 0.95))
        self._policy_lr = float(cfg.get("policy_lr", 3e-4))
        self._critic_lr = float(cfg.get("critic_lr", 3e-4))
        self._grad_norm_clip = float(cfg.get("grad_norm_clip", 1.0))
        self._learning_starts = int(cfg.get("learning_starts", 0))
        self._random_timesteps = int(cfg.get("random_timesteps", 0))
        if self._random_timesteps != 0:
            raise ValueError("flatppo 固定 random_timesteps=0（探索由 tanh-Gaussian 分布提供）")
        self._ratio_clip = float(cfg.get("ratio_clip", 0.2))
        self._value_loss_scale = float(cfg.get("value_loss_scale", 0.5))

        self._entropy_coef = float(cfg.get("entropy_loss_scale_ctrl", 0.0))
        self._kl_threshold = float(cfg.get("kl_early_stop", 0.0))
        self._telemetry_interval = max(1, int(cfg.get("telemetry_interval", 50)))
        self._time_limit_bootstrap = bool(cfg.get("time_limit_bootstrap", False))
        if self._time_limit_bootstrap:
            raise ValueError("flatppo 仅支持 time_limit_bootstrap=false（自然超时=有限任务真终局）")

        self._flat_actor_hidden = list(cfg.get("flat_actor_hidden", [440, 440, 440]))
        self._critic_hidden = list(cfg.get("critic_hidden", [512, 512, 512]))
        self._activation = str(cfg.get("activation", "relu"))
        self._use_layer_norm = bool(cfg.get("use_layer_norm", False))

        self._init_log_std = float(cfg.get("flat_ppo_initial_log_std", -1.2))
        self._max_log_std = float(cfg.get("flat_ppo_max_log_std", -0.6))
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
        self.frozen_agents: set[str] = set()
        self._tensor_track: dict[str, list[torch.Tensor]] = {}
        self._roll_ptr = 0
        self._roll_count = 0

        state_spaces = {a: state_space for a in possible_agents}
        models = {a: {} for a in possible_agents}
        super().__init__(
            cfg=FlatPPO_CFG(
                **{
                    k: v
                    for k, v in {
                        "rollouts": self._rollouts,
                        "learning_epochs": self._epochs,
                        "mini_batches": self._mini_batches,
                        "discount_factor": self._gamma,
                        "gae_lambda": self._lam,
                        "learning_rate": (self._policy_lr, self._critic_lr),
                        "grad_norm_clip": self._grad_norm_clip,
                        "learning_starts": self._learning_starts,
                        "random_timesteps": 0,
                        "ratio_clip": self._ratio_clip,
                        "value_loss_scale": self._value_loss_scale,
                        "entropy_loss_scale": self._entropy_coef,
                        "kl_threshold": self._kl_threshold,
                        "experiment": cfg.get("experiment"),
                    }.items()
                }
            ),
            possible_agents=possible_agents,
            models=models,
            memories=None,
            observation_spaces=observation_spaces,
            state_spaces=state_spaces,
            action_spaces=action_spaces,
            device=device,
        )

        self.actors: dict[str, TanhGaussianActor] = {}
        self.values: dict[str, ValueMLP] = {}
        self.actor_optimizers: dict[str, torch.optim.Optimizer] = {}
        self.value_optimizers: dict[str, torch.optim.Optimizer] = {}
        self._identity_preprocessor = _IdentityPreprocessor()
        for team, members in self._team_members.items():
            rep = members[0]
            a = TanhGaussianActor(
                self._obs_dim_per_agent[rep],
                4,
                self._flat_actor_hidden,
                self._activation,
                self._use_layer_norm,
                initial_log_std=self._init_log_std,
                max_log_std=self._max_log_std,
            ).to(self.device)
            v = ValueMLP(self._state_dim, self._critic_hidden, self._activation, self._use_layer_norm).to(self.device)
            self.actors[team] = a
            self.values[team] = v
            self.actor_optimizers[team] = torch.optim.Adam(a.parameters(), lr=self._policy_lr)
            self.value_optimizers[team] = torch.optim.Adam(v.parameters(), lr=self._critic_lr)
            self.checkpoint_modules[rep]["policy"] = a
            self.checkpoint_modules[rep]["optimizer"] = self.actor_optimizers[team]
            self.checkpoint_modules[rep]["value"] = v
            self.checkpoint_modules[rep]["value_optimizer"] = self.value_optimizers[team]

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
            for rep in self._team_representative.values():
                self.checkpoint_modules[rep]["state_preprocessor"] = self._state_preprocessor[rep]
        else:
            self._obs_preprocessor = {a: self._identity_preprocessor for a in possible_agents}
            self._state_preprocessor = {
                self._team_representative[t]: self._identity_preprocessor for t in self._team_members
            }

        R, N = self._rollouts, num_envs
        z_dev = self.device
        self._buf_states = torch.zeros(R, N, self._state_dim, device=z_dev)
        self._buf_dones = torch.zeros(R, N, dtype=torch.bool, device=z_dev)
        self._buf_learn = torch.zeros(R, N, dtype=torch.bool, device=z_dev)
        self._buf_rewards = {t: torch.zeros(R, N, device=z_dev) for t in self._team_members}
        self._buf_values = {t: torch.zeros(R, N, device=z_dev) for t in self._team_members}
        self._buf_obs = {a: torch.zeros(R, N, self._obs_dim_per_agent[a], device=z_dev) for a in possible_agents}
        self._buf_u = {a: torch.zeros(R, N, 4, device=z_dev) for a in possible_agents}
        self._buf_z = {a: torch.zeros(R, N, 4, device=z_dev) for a in possible_agents}
        self._buf_logp = {a: torch.zeros(R, N, device=z_dev) for a in possible_agents}
        self._buf_alive = {a: torch.zeros(R, N, dtype=torch.bool, device=z_dev) for a in possible_agents}

        self._buf_script = {a: torch.zeros(R, N, dtype=torch.bool, device=z_dev) for a in possible_agents}
        self._last_next_state = {t: torch.zeros(N, self._state_dim, device=z_dev) for t in self._team_members}

        self._architecture_fingerprint = {
            "format": "tad_flat_ppo",
            "requires_fingerprint": True,
            "flat_obs": str(cfg.get("flat_obs", "pool")),
            "flat_actor_hidden": list(self._flat_actor_hidden),
            "critic_state": "token",
            "flat_ppo_initial_log_std": self._init_log_std,
            "flat_ppo_max_log_std": self._max_log_std,
            "gamma_control": self._gamma,
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
            f"[TAD][build] FLAT-PPO：teams={list(self._team_members)}  "
            f"actor_hidden={self._flat_actor_hidden}  V(state {self._state_dim})  "
            f"rollouts={R}  epochs={self._epochs}  mini_batches={self._mini_batches}  "
            f"γ={self._gamma}  λ={self._lam}  σ_init={self._init_log_std}(log) "
            f"σ_max={self._max_log_std}(log)  entropy={self._entropy_coef}"
        )

    def _eval_row_mask(self, ref: torch.Tensor) -> torch.Tensor | None:
        m = getattr(self, "_eval_env_mask", None)
        if m is None:
            return None
        m = m.to(device=ref.device, dtype=torch.bool).view(-1)
        return m if m.numel() == ref.shape[0] else None

    def pre_interaction(self, *, timestep, timesteps):
        pass

    def act(self, observations, states, *, timestep, timesteps):
        actions = {}
        self._current_logp = {}
        self._current_gaussian = {}
        warmup_steps = getattr(self, "_warmup_steps", 0)
        warmup_env = getattr(self, "_warmup_env", None)
        warmup_mask = None
        if warmup_steps > 0 and warmup_env is not None:
            warmup_mask = warmup_env.episode_length_buf < warmup_steps
        ref = observations[self.possible_agents[0]]
        eval_mask = self._eval_row_mask(ref)
        with torch.no_grad():
            for team, members in self._team_members.items():
                actor = self.actors[team]
                for a in members:
                    obs_n = self._obs_preprocessor[a](observations[a])
                    if not self.training:
                        u, z, logp, (mu, std) = actor.sample(obs_n, deterministic=True)
                    else:
                        u, z, logp, (mu, std) = actor.sample(obs_n)
                        if eval_mask is not None:
                            det = torch.tanh(mu)
                            u = torch.where(eval_mask.unsqueeze(-1), det, u)
                            logp = torch.where(eval_mask, torch.zeros_like(logp), logp)
                    self._current_gaussian[a] = (mu, std)

                    if warmup_mask is not None and bool(warmup_mask.any()):
                        ctrl_fn = getattr(warmup_env, "compute_warmup_stabilize_action", None)
                        ctrl = ctrl_fn(a) if ctrl_fn is not None else torch.zeros_like(u)
                        u = torch.where(warmup_mask.unsqueeze(-1), ctrl, u)
                        if self.training:
                            mu_w, std_w = self._current_gaussian[a]
                            logp_w = TanhGaussianActor.log_prob_u(u, mu_w, std_w)
                            logp = torch.where(warmup_mask, logp_w, logp)
                    actions[a] = u
                    self._current_logp[a] = logp

        self._current_values = {}
        if self.training:
            states_rep = states if not isinstance(states, dict) else None
            for team, rep in self._team_representative.items():
                s_t = states[rep] if isinstance(states, dict) else states_rep
                self._current_values[team] = self.values[team](s_t).squeeze(-1)
        return actions, {}

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
        uid0 = self.possible_agents[0]
        term_t = terminated[uid0] if isinstance(terminated, dict) else terminated
        trunc_t = truncated[uid0] if isinstance(truncated, dict) else truncated
        done = (term_t | trunc_t).bool().view(-1)
        n = done.shape[0]
        learn_row = torch.ones(n, dtype=torch.bool, device=self.device) if _eval_m is None else ~_eval_m

        was_alive = (infos.get("agent_was_alive") if isinstance(infos, dict) else None) or {}

        _env = getattr(self, "_hier_env", None)
        _joint_ctrl = _env.temporal_transition_info()["executed_joint_ctrl"] if _env is not None else None
        if _joint_ctrl is None:
            raise RuntimeError("flatppo 需要 agent._hier_env（train_tad 注入）读 executed_joint_ctrl 检测 script 行")
        script_m = {
            a: (_joint_ctrl[:, _env._drone_index[a] * 4 : (_env._drone_index[a] + 1) * 4] != actions[a]).any(dim=-1)
            for a in self.possible_agents
        }

        if self._normalize:
            for a in self.possible_agents:
                _o = observations[a]
                self._obs_preprocessor[a](_o if _eval_m is None else _o[~_eval_m], train=True)

        i = self._roll_ptr
        rep0 = self._team_representative[next(iter(self._team_members))]
        self._buf_states[i] = (states[rep0] if isinstance(states, dict) else states).detach()
        self._buf_dones[i] = done
        self._buf_learn[i] = learn_row
        for team, members in self._team_members.items():
            self._buf_rewards[team][i] = aggregate_team_reward(
                rewards, infos, members, aggr=self._team_reward_aggr, nominal_n=self._team_drone_count[team], team=team
            ).detach()
            self._buf_values[team][i] = self._current_values.get(team, torch.zeros(n, device=self.device))
            rep = self._team_representative[team]
            self._last_next_state[team] = (next_states[rep] if isinstance(next_states, dict) else next_states).detach()
        for a in self.possible_agents:
            self._buf_obs[a][i] = observations[a].detach()
            self._buf_u[a][i] = actions[a].detach()

            self._buf_logp[a][i] = self._current_logp.get(a, torch.zeros(n, device=self.device)).detach()
            alive_a = (
                was_alive[a].to(self.device).bool().view(-1)
                if a in was_alive
                else torch.ones(n, dtype=torch.bool, device=self.device)
            )
            self._buf_alive[a][i] = alive_a & learn_row
            self._buf_script[a][i] = script_m[a] & alive_a & learn_row

            self._buf_z[a][i] = torch.atanh(actions[a].clamp(-1 + 1e-6, 1 - 1e-6)).detach()

    def post_interaction(self, *, timestep, timesteps):
        if self.training:
            self._roll_ptr += 1
            if self._roll_ptr >= self._rollouts:
                self._roll_ptr = 0
                if timestep >= self._learning_starts:
                    self._ppo_update(timestep=timestep, timesteps=timesteps)
                    self._roll_count += 1
        super().post_interaction(timestep=timestep, timesteps=timesteps)

    def update(self, *, timestep, timesteps, uid=None):
        pass

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

    def _ppo_update(self, *, timestep, timesteps) -> None:
        frozen = getattr(self, "frozen_agents", set())
        R, N = self._rollouts, self._buf_states.shape[1]

        for team, members in self._team_members.items():
            if any(a in frozen for a in members):
                continue
            with torch.no_grad():
                last_values = self.values[team](self._last_next_state[team]).squeeze(-1)

            returns, adv = compute_gae(
                rewards=self._buf_rewards[team][:R],
                terminated=self._buf_dones[:R],
                truncated=torch.zeros_like(self._buf_dones[:R]),
                values=self._buf_values[team][:R],
                last_values=last_values,
                discount_factor=self._gamma,
                lambda_coefficient=self._lam,
            )
            learn = self._buf_learn[:R]

            expert_row = torch.stack([self._buf_script[a][:R] for a in members]).any(dim=0)
            norm_mask = learn & ~expert_row

            adv_mean = adv[norm_mask].mean() if bool(norm_mask.any()) else torch.zeros((), device=adv.device)
            adv_std = adv[norm_mask].std() if int(norm_mask.sum()) > 1 else torch.zeros((), device=adv.device)
            adv_n = torch.where(learn, (adv - adv_mean) / (adv_std + 1e-8), torch.zeros_like(adv))

            valid_idx = learn.reshape(-1).nonzero(as_tuple=False).view(-1)
            if valid_idx.numel() == 0:
                continue
            perm = valid_idx[torch.randperm(valid_idx.numel(), device=self.device)]
            mb_size = max(1, (perm.numel() + self._mini_batches - 1) // self._mini_batches)
            opt_a, opt_v = self.actor_optimizers[team], self.value_optimizers[team]
            actor, value = self.actors[team], self.values[team]
            total_pol_loss = total_val_loss = 0.0
            n_pol_steps = n_val_steps = 0
            stop = False
            for epoch in range(self._epochs):
                if stop:
                    break
                approx_kl = torch.zeros((), device=self.device)
                for mb_start in range(0, perm.numel(), mb_size):
                    rows = perm[mb_start : mb_start + mb_size]
                    r_idx, c_idx = rows // N, rows % N

                    pg_loss = 0.0
                    ent_loss = 0.0
                    n_valid_samples = 0
                    for a in members:
                        alive_m = self._buf_alive[a][:R][r_idx, c_idx] & ~self._buf_script[a][:R][r_idx, c_idx]
                        if not bool(alive_m.any()):
                            continue
                        obs_i = self._obs_preprocessor[a](self._buf_obs[a][:R][r_idx, c_idx], train=(epoch == 0))
                        z_i = self._buf_z[a][:R][r_idx, c_idx]
                        logp_old = self._buf_logp[a][:R][r_idx, c_idx]
                        mu, std = actor.distribution(obs_i)
                        logp_new = TanhGaussianActor.log_prob_from_z(z_i, mu, std)
                        ratio = torch.exp(logp_new - logp_old)
                        adv_m = adv_n[r_idx, c_idx]
                        surr = torch.min(
                            ratio * adv_m, torch.clamp(ratio, 1 - self._ratio_clip, 1 + self._ratio_clip) * adv_m
                        )
                        pg_loss = pg_loss - (surr * alive_m).sum()
                        if self._entropy_coef != 0.0:
                            ent = actor.entropy(obs_i)
                            ent_loss = ent_loss + (ent * alive_m).sum()
                        with torch.no_grad():
                            approx_kl = approx_kl + ((logp_old - logp_new) * alive_m).sum()
                        n_valid_samples = n_valid_samples + int(alive_m.sum())
                    if n_valid_samples > 0:
                        opt_a.zero_grad()
                        actor_loss = pg_loss / n_valid_samples - self._entropy_coef * ent_loss / n_valid_samples
                        actor_loss.backward()
                        if self._grad_norm_clip > 0:
                            nn.utils.clip_grad_norm_(actor.parameters(), self._grad_norm_clip)
                        opt_a.step()
                        approx_kl = approx_kl / n_valid_samples
                        total_pol_loss = total_pol_loss + actor_loss.detach().double()
                        n_pol_steps += 1

                    opt_v.zero_grad()
                    v_pred = value(self._buf_states[:R][r_idx, c_idx]).squeeze(-1)
                    ret_m = returns[r_idx, c_idx]
                    val_loss = self._value_loss_scale * ((v_pred - ret_m) ** 2).mean()
                    val_loss.backward()
                    if self._grad_norm_clip > 0:
                        nn.utils.clip_grad_norm_(value.parameters(), self._grad_norm_clip)
                    opt_v.step()
                    total_val_loss = total_val_loss + val_loss.detach().double()
                    n_val_steps += 1
                if self._kl_threshold > 0 and float(approx_kl) > self._kl_threshold:
                    stop = True
            _tick = timestep % self._telemetry_interval == 0 or self._roll_count == 0
            if n_pol_steps > 0 and _tick:
                self.track_data(f"Loss / Policy loss (team_{team})", total_pol_loss / n_pol_steps)
            if n_val_steps > 0 and _tick:
                self.track_data(f"Loss / Value loss (team_{team})", total_val_loss / n_val_steps)
            if _tick:
                self.track_data(
                    f"Policy / Standard deviation (team_{team})",
                    actor.log_std.clamp(max=actor.max_log_std).exp().mean(),
                )
            if timestep % self._telemetry_interval == 0:
                self.track_data(f"Value / mean (team_{team})", self._buf_values[team][:R].mean())
                self.track_data(f"PPO / script frac (team_{team})", expert_row.float().mean())
