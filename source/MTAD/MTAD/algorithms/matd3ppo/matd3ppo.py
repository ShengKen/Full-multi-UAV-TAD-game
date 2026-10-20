from __future__ import annotations

import torch
import torch.nn.functional as F

from ..matd3.matd3 import MATD3
from ..models import MaskedCategoricalActor, build_mlp
from ..team_utils import DecisionTrajectoryCollector
from ..resume_checkpoint import register_architecture_fingerprint


class MATD3PPO(MATD3):
    _high_ppo_mode = True

    def __init__(
        self,
        *,
        possible_agents: list[str],
        observation_spaces: dict,
        action_spaces: dict,
        state_space,
        num_envs: int,
        device,
        team_of: dict[str, str],
        cfg: dict,
        team_reward_aggr: str = "sum",
        critic_hidden: list[int] | None = None,
        activation: str = "elu",
    ):
        self._ppo_ratio_clip = float(cfg.get("ratio_clip", 0.2))
        self._ppo_lambda = float(cfg.get("gae_lambda", 0.95))
        self._ppo_epochs = int(cfg.get("learning_epochs", 3))
        self._ppo_entropy = float(cfg.get("entropy_loss_scale", 0.0))
        self._ppo_value_scale = float(cfg.get("value_loss_scale", 0.5))
        self._ppo_kl_stop = float(cfg.get("kl_early_stop", 0.05))
        self._ppo_rollouts_dec = int(cfg.get("rollouts_dec", 0))
        super().__init__(
            possible_agents=possible_agents,
            observation_spaces=observation_spaces,
            action_spaces=action_spaces,
            state_space=state_space,
            num_envs=num_envs,
            device=device,
            team_of=team_of,
            cfg=cfg,
            team_reward_aggr=team_reward_aggr,
            critic_hidden=critic_hidden,
            activation=activation,
        )
        self._validate_scope()
        self._ppo_num_envs = int(num_envs)
        if self._ppo_rollouts_dec <= 0:
            self._ppo_rollouts_dec = int(num_envs)
        self._dec_group_list = [(gk, aids) for gk, aids in self._actor_groups.items() if self._is_dec[aids[0]]]

        self._dec_cat_actors: dict[str, MaskedCategoricalActor] = {}
        for gk, aids in self._dec_group_list:
            rep = aids[0]
            act = MaskedCategoricalActor(
                self._obs_dim_per_agent[rep],
                self._n_pool,
                self._upper_actor_hidden,
                self._matd3_activation,
                self._use_layer_norm,
            ).to(self.device)
            self._dec_cat_actors[rep] = act

            for aid in aids:
                self.actors[aid] = act
                self.target_actors[aid] = act
            opt = torch.optim.Adam(act.parameters(), lr=self._policy_lr)
            for aid in aids:
                self.actor_optimizers[aid] = opt
            self.checkpoint_modules[rep]["policy"] = act
            self.checkpoint_modules[rep]["target_policy"] = act
            self.checkpoint_modules[rep]["actor_optimizer"] = opt

        self._dec_values: dict[str, torch.nn.Module] = {}
        self._dec_value_opts: dict[str, torch.optim.Optimizer] = {}
        for team, members in self._team_members.items():
            rep = members[0]
            v = build_mlp(
                self._state_dim, self._matd3_critic_hidden, 1, self._matd3_activation, self._use_layer_norm
            ).to(self.device)
            self._dec_values[team] = v
            self._dec_value_opts[team] = torch.optim.Adam(v.parameters(), lr=self._critic_lr)
            self.checkpoint_modules[rep]["value"] = v
            self.checkpoint_modules[rep]["value_optimizer"] = self._dec_value_opts[team]

        self._dec_collectors: dict[str, DecisionTrajectoryCollector] = {
            aid: DecisionTrajectoryCollector(
                num_envs=int(num_envs), max_completed_samples=65536, gamma_c=self._rho_high
            )
            for aid in self.possible_agents
            if self._is_dec[aid]
        }
        self._ppo_sampled_idx: dict[str, torch.Tensor] = {}

        self._architecture_fingerprint = {
            **self._architecture_fingerprint,
            "format": "tad_hier_matd3ppo",
            "requires_fingerprint": True,
            "dec": "ppo_categorical",
        }
        register_architecture_fingerprint(self, self._architecture_fingerprint)
        print(
            f"[MATD3PPO] 高层=masked Categorical PPO（ρ_H={self._rho_high:.8f} λ_H={self._ppo_lambda} "
            f"ε={self._ppo_ratio_clip} epochs={self._ppo_epochs} β_H={self._ppo_entropy} "
            f"rollouts_dec={self._ppo_rollouts_dec} KL_stop={self._ppo_kl_stop}）| 低层=MATD3 继承"
        )

        _n_params = (
            sum(p.numel() for p in self._dec_cat_actors[self._dec_group_list[0][1][0]].parameters())
            if self._dec_group_list
            else 0
        )
        print(
            f"[MATD3PPO][R01-R04] 高层资源：dec Categorical actor ×{len(self._dec_group_list)} 组"
            f"（每组 {_n_params:,} 参数，target=自身别名）、V_dec ×{len(self._dec_values)}、"
            f"collector ×{len(self._dec_collectors)}；B_H/高层Q/twin/_seg_* = 不构建"
        )

    def _validate_scope(self) -> None:
        if not (getattr(self, "_hier", False) and getattr(self, "_hier_dual", False)):
            raise ValueError("MATD3PPO v1 仅支持 train_layer=both（hier-dual 路径）；单层训练请用 --algo maddpg")
        if self._time_limit_bootstrap:
            raise ValueError(
                "MATD3PPO v1 仅支持 time_limit_bootstrap=false（自然超时=任务终局；"
                "超时 bootstrap 需 pre-reset V(x')，未实现）"
            )

    def _hier_act(self, observations, states, *, timestep, timesteps):
        """Select discrete targets and continuous flight controls."""
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
            for gk, members in self._dec_group_list:
                act = self._dec_cat_actors[members[0]]
                for a in members:
                    smp = act.sample(dec_obs_n[a], dec_mask[a], deterministic=not self.training)

                    _ev_m = self._eval_row_mask(dec_obs_n[a])
                    if self.training and _ev_m is not None:
                        det = act.sample(dec_obs_n[a], dec_mask[a], deterministic=True)
                        smp = {k: torch.where(_ev_m.view(-1), det[k], v) for k, v in smp.items()}
                    idx = smp["idx"]
                    self._current_onehot[a] = F.one_hot(idx, self._dec_exec_dim).to(dec_obs_n[a].dtype)
                    actions[a] = idx.unsqueeze(-1).long()
                    self._ppo_sampled_idx[a] = idx.detach()

            ctrl_raw = self._commit_decisions_and_rebuild_ctrl_obs(observations, actions)
            self._current_ctrl_obs_raw = dict(ctrl_raw)
            for gk, members in self._actor_groups.items():
                if self._is_dec[members[0]]:
                    continue
                raw_by = {a: self._current_ctrl_obs_raw[a] for a in members}
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
        """Record decision trajectories and low-level transitions."""
        env = getattr(self, "_hier_env", None)
        pre = None
        if self.training and env is not None and self._dec_collectors:
            pre = self._ppo_prepare_record_fields(observations, states, terminated, truncated, infos)

        super()._hier_record_transition(
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
        if pre is None:
            return

        for team, members in self._team_members.items():
            buf = self.replays[team]
            base = (buf._ptr - self._ppo_num_envs) % buf.capacity
            team_reward = buf.reward[base : base + self._ppo_num_envs].to(self.device).view(-1)
            anchor_sw = (
                buf.anchor_switch[base : base + self._ppo_num_envs].to(self.device).view(-1).bool()
                if not self._state_anchor_free
                else None
            )
            was_alive_pa = pre["was_alive_pa"]
            still_alive_pa = pre["still_alive_pa"]
            for aid in members:
                if not self._is_dec[aid]:
                    continue
                uid = aid.split("/")[0]
                wa = was_alive_pa[aid].bool().view(-1)
                sa = still_alive_pa[aid].bool().view(-1)
                death = wa & ~sa
                f = pre["fields"][aid]
                self._dec_collectors[aid].step(
                    boundary=env._decision_boundary[uid].view(-1),
                    done=pre["episode_done"],
                    reward=team_reward,
                    active=pre["learn_active"][aid],
                    sample_fields=f,
                    death=death,
                    anchor_switch=anchor_sw,
                )

    def _ppo_prepare_record_fields(self, observations, states, terminated, truncated, infos) -> dict:
        uid0 = self.possible_agents[0]
        term_t = terminated[uid0] if isinstance(terminated, dict) else terminated
        trunc_t = truncated[uid0] if isinstance(truncated, dict) else truncated
        episode_done = (term_t | trunc_t).bool().view(-1)
        _term_ref = next(iter(terminated.values())) if isinstance(terminated, dict) else terminated
        _eval_m = self._eval_row_mask(_term_ref) if isinstance(terminated, dict) else None
        active_pa = self._active_per_agent(infos, "agent_was_alive", term_t)
        still_pa = self._active_per_agent(infos, "agent_still_alive", term_t)
        learn_active = (
            {aid: (m.view(-1) & ~_eval_m) for aid, m in active_pa.items()}
            if _eval_m is not None
            else {aid: m.view(-1) for aid, m in active_pa.items()}
        )
        was_alive_info = self._alive_info(infos, "agent_was_alive")
        fields: dict[str, dict[str, torch.Tensor]] = {}
        for team, members in self._team_members.items():
            state_t, _ = self._team_state_anchor(states, team, was_alive_info)
            rep_state = members[0]
            with torch.no_grad():
                value = self._dec_values[team](self._state_preprocessor[rep_state](state_t)).squeeze(-1)
            for aid in members:
                if not self._is_dec[aid]:
                    continue
                uid = aid.split("/")[0]
                boundary = self._hier_env._decision_boundary[uid].view(-1)
                executed = self._hier_env._eff_dec_idx[uid].long().view(-1).clamp(0, self._dec_exec_dim - 1)
                sampled = self._ppo_sampled_idx.get(aid)
                sampled = executed if sampled is None else sampled.view(-1)
                script = executed != sampled

                logp_old = torch.zeros_like(value)
                bnd = boundary & learn_active[aid].bool()
                if bool(bnd.any()):
                    actor = self._dec_cat_actors[self._actor_group_rep[self._actor_group_of[aid]]]
                    obs_b = observations[aid][bnd]
                    mask_b = obs_b[:, -self._mask_tail_dim :]
                    idx_b = executed[bnd]
                    logp_old[bnd] = actor.log_prob(self._obs_preprocessor[aid](obs_b), idx_b, mask_b)
                obs_raw = observations[aid]
                fields[aid] = {
                    "obs": obs_raw,
                    "mask": obs_raw[:, -self._mask_tail_dim :],
                    "idx": executed,
                    "log_prob": logp_old,
                    "value": value,
                    "state": state_t,
                    "script": script,
                }
        return {
            "episode_done": episode_done,
            "learn_active": learn_active,
            "was_alive_pa": {a: m.view(-1) for a, m in active_pa.items()},
            "still_alive_pa": {a: m.view(-1) for a, m in still_pa.items()},
            "fields": fields,
        }

    def _hier_update(self, *, timestep, timesteps):
        """Coordinate high-level PPO and low-level TD3 updates."""
        if not self._hier_dual:
            return super()._hier_update(timestep=timestep, timesteps=timesteps)
        for team, members in self._team_members.items():
            buf = self.replays[team]
            if len(buf) >= self._batch_size:
                self._update_low_from_control_batch(team, members, self._consume_low_batch(team, buf))
            self._issue_low_prefetch(team, buf)
        total_traj = sum(c.completed_trajectories for c in self._dec_collectors.values())
        if total_traj >= self._ppo_rollouts_dec:
            self._update_dec_ppo(timestep)
        if timestep % self._telemetry_interval == 0:
            self.track_data("PPO / Dec pending trajectories", float(total_traj))

    @staticmethod
    def _ppo_gae_trajectory(tr: dict[str, torch.Tensor], lam: float) -> tuple[torch.Tensor, torch.Tensor]:
        v = tr["value"].view(-1)
        r = tr["reward"].view(-1)
        gam = tr["seg_gamma"].view(-1)
        terminal = torch.zeros_like(v, dtype=torch.bool)
        terminal[-1] = True
        switch = tr.get("anchor_switch")
        if switch is not None:
            terminal = terminal | switch.view(-1).bool()
        b = (~terminal).to(v.dtype)
        v_next = torch.cat([v[1:], torch.zeros_like(v[:1])])
        delta = r + gam * b * v_next - v
        adv = torch.zeros_like(delta)
        adv_next = 0.0
        for k in range(delta.shape[0] - 1, -1, -1):
            adv[k] = delta[k] + gam[k] * lam * b[k] * adv_next
            adv_next = adv[k]
        return adv, v + adv

    @staticmethod
    @torch.no_grad()
    def _value_fit_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        targets = targets.detach().float().reshape(-1)
        errors = predictions.detach().float().reshape(-1) - targets
        denominator = targets.var(unbiased=False) + 1e-8
        nmse = errors.square().mean() / denominator
        ev = 1.0 - errors.var(unbiased=False) / denominator
        return nmse, ev

    def _update_dec_ppo(self, timestep: int) -> None:
        """Optimize discrete target-selection policies using PPO."""
        frozen_agents = getattr(self, "frozen_agents", set())

        per_aid: dict[str, dict[str, torch.Tensor]] = {}
        for aid, collector in self._dec_collectors.items():
            trajs = collector.pop_completed()
            if not trajs:
                continue
            parts = {k: [] for k in ("obs", "mask", "idx", "log_prob", "value", "adv", "ret", "script", "state")}
            for tr in trajs:
                adv, ret = self._ppo_gae_trajectory(tr, self._ppo_lambda)
                for k_src, dst in (
                    ("obs", "obs"),
                    ("mask", "mask"),
                    ("idx", "idx"),
                    ("log_prob", "log_prob"),
                    ("value", "value"),
                    ("script", "script"),
                    ("state", "state"),
                ):
                    parts[dst].append(tr[k_src].view(-1) if k_src in ("idx", "log_prob", "script") else tr[k_src])
                parts["adv"].append(adv)
                parts["ret"].append(ret)
            per_aid[aid] = {k: (torch.cat(v) if v else None) for k, v in parts.items()}
        if not per_aid:
            return

        team_rows: dict[str, list[str]] = {}
        for aid in per_aid:
            team_rows.setdefault(self._team_of[aid], []).append(aid)
        cat: dict[str, dict[str, torch.Tensor]] = {}
        for team, aids in team_rows.items():
            d = {k: torch.cat([per_aid[a][k] for a in aids]) for k in per_aid[aids[0]]}
            valid = ~d["script"].bool()
            adv = d["adv"]
            if bool(valid.any()):
                mu, sd = adv[valid].mean(), adv[valid].std()
                d["a_norm"] = (adv - mu) / (sd + 1e-8)
            else:
                d["a_norm"] = torch.zeros_like(adv)
            d["valid"] = valid
            d["n_valid"] = max(int(valid.sum().item()), 1)
            cat[team] = d

        rep_by_team = {team: members[0] for team, members in self._team_members.items()}
        value_fit_totals = []
        value_fit_samples = 0
        for team, d in cat.items():
            rep = rep_by_team[team]
            if self._roll_value_frozen(team):
                continue
            v_pred = self._dec_values[team](self._state_preprocessor[rep](d["state"])).squeeze(-1)
            v_loss = self._ppo_value_scale * ((v_pred - d["ret"]) ** 2).mean()
            self._dec_value_opts[team].zero_grad()
            v_loss.backward()
            if self._grad_norm_clip > 0:
                torch.nn.utils.clip_grad_norm_(self._dec_values[team].parameters(), self._grad_norm_clip)
            self._dec_value_opts[team].step()
            self.track_data(f"Loss / Dec value loss (team_{team})", v_loss.detach())

            nmse, ev = self._value_fit_metrics(v_pred, d["ret"])
            self.track_data(f"Loss/nmse (team_{team}_dec)", nmse)
            self.track_data(f"Loss/ev (team_{team}_dec)", ev)
            count = d["ret"].numel()
            value_fit_totals.append(torch.stack((nmse, ev)) * count)
            value_fit_samples += count
        if value_fit_samples:
            nmse, ev = torch.stack(value_fit_totals).sum(dim=0) / value_fit_samples
            self.track_data("Loss/nmse", nmse)
            self.track_data("Loss/ev", ev)

        for gk, group in self._dec_group_list:
            rep = group[0]
            team = self._team_of[rep]
            d = cat.get(team)
            if d is None or any(a not in per_aid for a in group):
                continue
            if any(a in frozen_agents for a in group):
                continue
            actor = self._dec_cat_actors[rep]

            rows = torch.zeros(d["idx"].shape[0], dtype=torch.bool, device=self.device)
            ofs = 0
            for a in team_rows[team]:
                n_i = per_aid[a]["idx"].shape[0]
                if a in group:
                    rows[ofs : ofs + n_i] = True
                ofs += n_i
            obs_g = self._obs_preprocessor[rep](d["obs"][rows])
            mask_g = d["mask"][rows]
            idx_g = d["idx"][rows].long()
            logp_g = d["log_prob"][rows]
            a_g = d["a_norm"][rows]
            valid_g = d["valid"][rows]
            n_g = max(int(valid_g.sum().item()), 1)
            stopped = False
            for _ep in range(self._ppo_epochs):
                logp_new = actor.log_prob(obs_g, idx_g, mask_g)
                w = torch.exp(logp_new - logp_g)
                sur = torch.min(w * a_g, torch.clamp(w, 1.0 - self._ppo_ratio_clip, 1.0 + self._ppo_ratio_clip) * a_g)
                ent = actor.entropy(obs_g, mask_g)
                pi_loss = -(sur[valid_g].sum() / n_g + self._ppo_entropy * ent[valid_g].sum() / n_g)
                self.actor_optimizers[rep].zero_grad()
                pi_loss.backward()
                if self._grad_norm_clip > 0:
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), self._grad_norm_clip)
                self.actor_optimizers[rep].step()
                with torch.no_grad():
                    kl = ((w - 1.0) - (logp_new - logp_g))[valid_g].mean()
                self.track_data(f"Policy / Dec KL (team_{team})", kl.detach())
                self.track_data(f"Policy / Dec entropy (team_{team})", ent[valid_g].mean().detach())
                self.track_data(f"Loss / Dec actor loss (team_{team})", pi_loss.detach())
                if self._ppo_kl_stop > 0 and bool((kl > self._ppo_kl_stop).item()):
                    stopped = True
                    break
            if stopped:
                self.track_data("Policy / Dec KL early stop", 1.0)
