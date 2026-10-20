from __future__ import annotations

import torch
import torch.nn as nn

from ..maddpg.maddpg import MADDPG, CriticMLP
from ..models import zero_initialize_module
from ..q_low import QLowCritic, BatchedQLowCritic, _QLowSliceCheckpoint
from ..resume_checkpoint import register_architecture_fingerprint


class MATD3(MADDPG):
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
        self._matd3_twin = bool(cfg.get("matd3_twin", True))
        self._matd3_smooth = bool(cfg.get("matd3_smooth", True))
        self._matd3_smooth_std = float(cfg.get("matd3_smooth_std", 0.2))
        self._matd3_smooth_clip = float(cfg.get("matd3_smooth_clip", 0.5))
        self._matd3_delay = bool(cfg.get("matd3_delay", True))
        self._matd3_policy_delay = max(1, int(cfg.get("matd3_policy_delay", 2)))
        if self._matd3_smooth_std < 0 or self._matd3_smooth_clip < 0:
            raise ValueError(
                f"matd3_smooth_std={self._matd3_smooth_std} / matd3_smooth_clip={self._matd3_smooth_clip} 需 >= 0"
            )
        self._matd3_critic_hidden = list(critic_hidden) if critic_hidden else list(cfg.get("critic_hidden", [256, 256]))
        self._matd3_activation = activation

        self._matd3_high_tick = 0
        self._matd3_low_tick = 0
        self._matd3_actor_due_high = True
        self._matd3_actor_due_low = True
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
        self._matd3_validate_scope()
        if self._matd3_twin:
            self._matd3_build_twins()

        self._architecture_fingerprint = {
            **self._architecture_fingerprint,
            "format": "tad_hier_matd3",
            "requires_fingerprint": True,
            "matd3_twin": self._matd3_twin,
            "matd3_smooth": self._matd3_smooth,
            "matd3_smooth_std": self._matd3_smooth_std,
            "matd3_smooth_clip": self._matd3_smooth_clip,
            "matd3_delay": self._matd3_delay,
            "matd3_policy_delay": self._matd3_policy_delay,
        }
        register_architecture_fingerprint(self, self._architecture_fingerprint)

    def _matd3_validate_scope(self) -> None:
        if not (getattr(self, "_hier", False) and getattr(self, "_hier_dual", False)):
            raise ValueError("MATD3 v1 仅支持 train_layer=both（hier-dual 双 critic 路径）；单层训练请用 --algo maddpg")

    def _matd3_build_twins(self) -> None:
        self.critics2, self.target_critics2 = {}, {}
        if self._high_ppo_mode:
            for _team in self._team_members:
                CriticMLP(
                    self._state_dim,
                    self._critic_act_dim,
                    self._matd3_critic_hidden,
                    self._matd3_activation,
                    self._use_layer_norm,
                )
                CriticMLP(
                    self._state_dim,
                    self._critic_act_dim,
                    self._matd3_critic_hidden,
                    self._matd3_activation,
                    self._use_layer_norm,
                )
            self.critic_optimizers2 = {}
        else:
            for team in self._team_members:
                c2 = CriticMLP(
                    self._state_dim,
                    self._critic_act_dim,
                    self._matd3_critic_hidden,
                    self._matd3_activation,
                    self._use_layer_norm,
                ).to(self.device)
                tc2 = CriticMLP(
                    self._state_dim,
                    self._critic_act_dim,
                    self._matd3_critic_hidden,
                    self._matd3_activation,
                    self._use_layer_norm,
                ).to(self.device)
                if self._zero_init:
                    zero_initialize_module(c2)
                tc2.load_state_dict(c2.state_dict())
                for p in tc2.parameters():
                    p.requires_grad = False
                self.critics2[team] = c2
                self.target_critics2[team] = tc2
            self.critic_optimizers2 = {
                team: torch.optim.Adam(self.critics2[team].parameters(), lr=self._critic_lr)
                for team in self._team_members
            }
            for team, members in self._team_members.items():
                rep = members[0]
                self.checkpoint_modules[rep]["critic2"] = self.critics2[team]
                self.checkpoint_modules[rep]["target_critic2"] = self.target_critics2[team]
                self.checkpoint_modules[rep]["critic2_optimizer"] = self.critic_optimizers2[team]

        _low_hidden = list(self._cfg.get("lower_critic_hidden", [256, 256]))
        _low_lr = self._q_low_lr
        if self._lower_critic_mode == "team":
            self.q_low_team_net2: dict = {}
            self.target_q_low_team_net2: dict = {}
            self.q_low_team_opts2: dict = {}
            _ctx_w = self._M * (self._dec_repr_dim + 1)
            for _team, _members in self._team_members.items():
                if not any(not self._is_dec[a] for a in _members):
                    continue
                q2 = CriticMLP(
                    self._state_dim + _ctx_w, self._M * 4, _low_hidden, self._matd3_activation, self._use_layer_norm
                ).to(self.device)
                tq2 = CriticMLP(
                    self._state_dim + _ctx_w, self._M * 4, _low_hidden, self._matd3_activation, self._use_layer_norm
                ).to(self.device)
                tq2.load_state_dict(q2.state_dict())
                for p in tq2.parameters():
                    p.requires_grad = False
                self.q_low_team_net2[_team] = q2
                self.target_q_low_team_net2[_team] = tq2
                self.q_low_team_opts2[_team] = torch.optim.Adam(q2.parameters(), lr=_low_lr)
                rep = _members[0]
                self.checkpoint_modules[rep]["q_low2"] = q2
                self.checkpoint_modules[rep]["target_q_low2"] = tq2
                self.checkpoint_modules[rep]["q_low2_optimizer"] = self.q_low_team_opts2[_team]
            return
        _team_ctrl = {t: [a for a in m if not self._is_dec[a]] for t, m in self._team_members.items()}
        if self._q_low_batched:
            self.q_low_batch2: dict = {}
            self.target_q_low_batch2: dict = {}
            self.q_low_opt2: dict = {}
            _od = next(iter({self._obs_dim_per_agent[a] for aids in _team_ctrl.values() for a in aids}))
            for _team, _ctrl in _team_ctrl.items():
                if not _ctrl:
                    continue
                qb2 = BatchedQLowCritic(
                    len(_ctrl),
                    _od,
                    self._q_low_act_dim,
                    _low_hidden,
                    self._matd3_activation,
                    self._use_layer_norm,
                    device=self.device,
                )
                tqb2 = BatchedQLowCritic(
                    len(_ctrl),
                    _od,
                    self._q_low_act_dim,
                    _low_hidden,
                    self._matd3_activation,
                    self._use_layer_norm,
                    device=self.device,
                )
                tqb2.load_per_aid(qb2.dump_per_aid())
                for p in tqb2.parameters():
                    p.requires_grad = False
                self.q_low_batch2[_team] = qb2
                self.target_q_low_batch2[_team] = tqb2
                self.q_low_opt2[_team] = torch.optim.Adam(qb2.parameters(), lr=_low_lr)
                for i, aid in enumerate(_ctrl):
                    self.checkpoint_modules[aid]["q_low2"] = _QLowSliceCheckpoint(qb2, i)
                    self.checkpoint_modules[aid]["target_q_low2"] = _QLowSliceCheckpoint(tqb2, i)
        else:
            self.q_low2: dict = {}
            self.target_q_low2: dict = {}
            self.q_low_opts2: dict = {}
            for _team, _ctrl in _team_ctrl.items():
                for aid in _ctrl:
                    ql2 = QLowCritic(
                        self._obs_dim_per_agent[aid],
                        self._q_low_act_dim,
                        _low_hidden,
                        self._matd3_activation,
                        self._use_layer_norm,
                    ).to(self.device)
                    tql2 = QLowCritic(
                        self._obs_dim_per_agent[aid],
                        self._q_low_act_dim,
                        _low_hidden,
                        self._matd3_activation,
                        self._use_layer_norm,
                    ).to(self.device)
                    tql2.load_state_dict(ql2.state_dict())
                    for p in tql2.parameters():
                        p.requires_grad = False
                    self.q_low2[aid] = ql2
                    self.target_q_low2[aid] = tql2
                    self.q_low_opts2[aid] = torch.optim.Adam(ql2.parameters(), lr=_low_lr)
                    self.checkpoint_modules[aid]["q_low2"] = ql2
                    self.checkpoint_modules[aid]["target_q_low2"] = tql2
                    self.checkpoint_modules[aid]["q_low2_optimizer"] = self.q_low_opts2[aid]

    def update(self, *, timestep, timesteps):
        """Update the hierarchical TD3 policies and target networks."""
        self._matd3_actor_due_low = (not self._matd3_delay) or (self._matd3_low_tick % self._matd3_policy_delay == 0)
        self._matd3_low_tick += 1

        if timestep % getattr(self, "_high_update_every", 1) == 0:
            self._matd3_actor_due_high = (not self._matd3_delay) or (
                self._matd3_high_tick % self._matd3_policy_delay == 0
            )
            self._matd3_high_tick += 1
        return super().update(timestep=timestep, timesteps=timesteps)

    def _q_low_polyak_batched(self, team: str) -> None:
        super()._q_low_polyak_batched(team)
        if self._matd3_twin and team in getattr(self, "target_q_low_batch2", {}):
            self._soft_update(self.target_q_low_batch2[team], self.q_low_batch2[team])

    def _clip_q_low_grad_per_aid(self, team: str, q_low_batch=None) -> None:
        if self._grad_norm_clip <= 0:
            return
        qb = self.q_low_batch[team] if q_low_batch is None else q_low_batch
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
            if self._matd3_twin:
                target_q = torch.minimum(target_q, self.target_critics2[team](ns, next_ha).squeeze(-1))
            target = r + discount * (1.0 - done) * target_q
        current_q = self.critics[team](s, ha).squeeze(-1)
        critic_loss = ((current_q - target) ** 2).mean()
        if not self._roll_value_frozen(team):
            self.critic_optimizers[team].zero_grad()
            critic_loss.backward()
            if self._grad_norm_clip > 0:
                nn.utils.clip_grad_norm_(self.critics[team].parameters(), self._grad_norm_clip)
            self.critic_optimizers[team].step()
        current_q2 = None
        critic_loss2 = None
        if self._matd3_twin:
            current_q2 = self.critics2[team](s, ha).squeeze(-1)
            critic_loss2 = ((current_q2 - target) ** 2).mean()
            if not self._roll_value_frozen(team):
                self.critic_optimizers2[team].zero_grad()
                critic_loss2.backward()
                if self._grad_norm_clip > 0:
                    nn.utils.clip_grad_norm_(self.critics2[team].parameters(), self._grad_norm_clip)
                self.critic_optimizers2[team].step()

        if self._matd3_actor_due_high:
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
                        gate_stk = torch.stack(
                            [agent_mask[:, self._dec_aids.index(a)] > 0.5 for a in group_aids], dim=0
                        )
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
            if self._matd3_twin:
                self._soft_update(self.target_critics2[team], self.critics2[team])
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
        if self._matd3_twin:
            self.track_data(f"Loss / High critic loss twin2 (team_{team})", critic_loss2)
            self.track_data(f"Q / high mean twin2 (team_{team})", current_q2.mean())

            self.track_data(f"Critic / twin gap high (team_{team})", (current_q - current_q2).detach().abs().mean())

    def _update_low_from_control_batch(self, team: str, members: list[str], batch: dict) -> None:
        """Optimize low-level actors and critics from replay samples."""

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
                if self._matd3_smooth:
                    next_a = (
                        next_a
                        + (torch.randn_like(next_a) * self._matd3_smooth_std).clamp(
                            -self._matd3_smooth_clip, self._matd3_smooth_clip
                        )
                    ).clamp(-1.0, 1.0)
                next_act_stk = self._q_low_cat_action(next_a, extra_stk_next)
                target_q = self.target_q_low_batch[team](nobs_stk, next_act_stk)
                if self._matd3_twin:
                    target_q = torch.minimum(target_q, self.target_q_low_batch2[team](nobs_stk, next_act_stk))
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
            current_q2 = None
            loss_per_aid2 = None
            if self._matd3_twin:
                current_q2 = self.q_low_batch2[team](obs_stk, act_stk)
                sq_low2 = (current_q2 - target) ** 2
                if w_row is not None:
                    sq_low2 = sq_low2 * w_row.unsqueeze(0)
                loss_per_aid2 = (sq_low2 * active_stk).sum(dim=1) / active_stk.sum(dim=1).clamp(min=1)
                if not self._roll_value_frozen(team):
                    self.q_low_opt2[team].zero_grad()
                    loss_per_aid2.sum().backward()
                    self._clip_q_low_grad_per_aid(team, q_low_batch=self.q_low_batch2[team])
                    self.q_low_opt2[team].step()
            loss_det = loss_per_aid.detach()

            qmean_det = (current_q.detach() * active_stk).sum(dim=1) / active_stk.sum(dim=1).clamp(min=1)
            for i, a in enumerate(ctrl_aids):
                self.track_data(f"Loss / Q_low critic ({a})", loss_det[i])
                self.track_data(f"Q / low mean ({a})", qmean_det[i])
            if self._matd3_twin:
                qmean2_det = (current_q2.detach() * active_stk).sum(dim=1) / active_stk.sum(dim=1).clamp(min=1)
                for i, a in enumerate(ctrl_aids):
                    self.track_data(f"Loss / Q_low critic twin2 ({a})", loss_per_aid2.detach()[i])
                    self.track_data(f"Q / low mean twin2 ({a})", qmean2_det[i])
                self.track_data(
                    f"Critic / twin gap low (team_{team})",
                    ((current_q - current_q2).detach().abs() * active_stk).sum() / active_stk.sum().clamp(min=1),
                )
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
            if self._matd3_smooth:
                for _aid in next_a_by:
                    next_a_by[_aid] = (
                        next_a_by[_aid]
                        + (torch.randn_like(next_a_by[_aid]) * self._matd3_smooth_std).clamp(
                            -self._matd3_smooth_clip, self._matd3_smooth_clip
                        )
                    ).clamp(-1.0, 1.0)
            for aid in ctrl_aids:
                obs_i = self._obs_preprocessor[aid](obs_pa[aid])
                nobs_i = nobs_n_by[aid]
                extra_i = self._q_low_extra_for(aid, extra_pa)
                extra_i_next = self._q_low_extra_for(aid, extra_pa_next)
                with torch.no_grad():
                    target_low_q = self.target_q_low[aid](
                        nobs_i, self._q_low_cat_action(next_a_by[aid], extra_i_next)
                    ).squeeze(-1)
                    if self._matd3_twin:
                        target_low_q = torch.minimum(
                            target_low_q,
                            self.target_q_low2[aid](nobs_i, self._q_low_cat_action(next_a_by[aid], extra_i)).squeeze(
                                -1
                            ),
                        )
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
                if self._matd3_twin:
                    current_low_q2 = self.q_low2[aid](
                        obs_i, self._q_low_cat_action(ja[:, start : start + 4], extra_i)
                    ).squeeze(-1)
                    sq_low2 = (current_low_q2 - target_low) ** 2
                    if w_row is not None:
                        sq_low2 = sq_low2 * w_row
                    low_critic_loss2 = sq_low2[active_i].mean()
                    if not self._roll_value_frozen(self._team_of[aid]):
                        self.q_low_opts2[aid].zero_grad()
                        low_critic_loss2.backward()
                        if self._grad_norm_clip > 0:
                            nn.utils.clip_grad_norm_(self.q_low2[aid].parameters(), self._grad_norm_clip)
                        self.q_low_opts2[aid].step()
                    self.track_data(f"Loss / Q_low critic twin2 ({aid})", low_critic_loss2)
                    self.track_data(f"Q / low mean twin2 ({aid})", current_low_q2[active_i].mean())
                    self.track_data(
                        f"Critic / twin gap low (team_{self._team_of[aid]})",
                        (current_low_q - current_low_q2).detach().abs()[active_i].mean(),
                    )
                self.track_data(f"Loss / Q_low critic ({aid})", low_critic_loss)

                self.track_data(f"Q / low mean ({aid})", current_low_q[active_i].mean())

        if self._matd3_actor_due_low:
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
                        [self._q_low_cat_action(a_map[a], self._q_low_extra_for(a, extra_pa)) for a in group_ctrl],
                        dim=0,
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
                        active_i = active_pa.get(
                            a_id, torch.ones(a_i.shape[0], dtype=torch.bool, device=a_i.device)
                        ).bool()

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
                    if self._matd3_twin:
                        self._soft_update(self.target_q_low2[aid], self.q_low2[aid])
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
                        act_i = na_map[a]
                        if self._matd3_smooth:
                            act_i = (
                                act_i
                                + (torch.randn_like(act_i) * self._matd3_smooth_std).clamp(
                                    -self._matd3_smooth_clip, self._matd3_smooth_clip
                                )
                            ).clamp(-1.0, 1.0)
                        di = self._agent_slot[a][0]
                        gate = active_pa.get(a, ones).bool().view(-1, 1)
                        next_ctrl[:, di, :] = torch.where(gate, act_i, next_ctrl[:, di, :])
                tq = self.target_q_low_team_net[team](
                    torch.cat([ns_n, nctx_obs], dim=-1), next_ctrl.reshape(B, -1)
                ).squeeze(-1)
                if self._matd3_twin:
                    tq = torch.minimum(
                        tq,
                        self.target_q_low_team_net2[team](
                            torch.cat([ns_n, nctx_obs], dim=-1), next_ctrl.reshape(B, -1)
                        ).squeeze(-1),
                    )
                target = r + g_eff * (1.0 - done_eff) * tq
            q_in_cur = torch.cat([s_n, ctx_obs], dim=-1)
            act_cur = ctx_ctrl.reshape(B, -1)
            current_q = self.q_low_team_net[team](q_in_cur, act_cur).squeeze(-1)
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
                if self._matd3_twin:
                    current_q2 = self.q_low_team_net2[team](q_in_cur, act_cur).squeeze(-1)
                    sq2 = (current_q2 - target) ** 2
                    if w_row is not None:
                        sq2 = sq2 * w_row
                    loss2 = (sq2 * team_active.to(sq2.dtype)).sum() / n_active
                    if not self._roll_value_frozen(team):
                        self.q_low_team_opts2[team].zero_grad()
                        loss2.backward()
                        if self._grad_norm_clip > 0:
                            nn.utils.clip_grad_norm_(self.q_low_team_net2[team].parameters(), self._grad_norm_clip)
                        self.q_low_team_opts2[team].step()
                    self.track_data(f"Loss / Q_low critic twin2 (team_{team})", loss2.detach())
                    self.track_data(
                        f"Q / low mean twin2 (team_{team})", (current_q2.detach() * team_active).sum() / n_active
                    )
                    self.track_data(
                        f"Critic / twin gap low (team_{team})",
                        ((current_q - current_q2).detach().abs() * team_active).sum() / n_active,
                    )

        if self._matd3_actor_due_low:
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
                if self._matd3_twin:
                    self._soft_update(self.target_q_low_team_net2[team], self.q_low_team_net2[team])
            seen_target_groups: set = set()
            for aid in ctrl_aids:
                gk = self._actor_group_of[aid]
                if gk in seen_target_groups:
                    continue
                seen_target_groups.add(gk)
                rep_target = self._actor_group_rep[gk]
                self._soft_update(self.target_actors[rep_target], self.actors[rep_target])
