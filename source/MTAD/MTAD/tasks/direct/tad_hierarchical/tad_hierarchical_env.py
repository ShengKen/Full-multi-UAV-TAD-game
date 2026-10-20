# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.

# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import os as _os

import torch
import torch.nn.functional as F

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectMARLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import (
    euler_xyz_from_quat,
    quat_apply,
    quat_from_euler_xyz,
    quat_inv,
)

from ..quadrotor_common import QuadrotorEnvMixin
from ..spawn_utils import sample_in_arena_positions
from ..episode_horizon import (
    episode_horizon_step_bounds,
    sample_episode_horizon_steps,
)
from ..tad.scenario import (
    TARGET_TEAM,
    min_pair_distance,
    nearest_pair_los_vel,
    pair_engagement,
    per_drone_closing,
    per_drone_nearest_team_closing,
    per_drone_nearest_team_closing_batch,
    per_drone_nearest_team_distance_batch,
    per_drone_team_distance_sum_batch,
    greedy_assigned_team_closing,
    min_total_distance_assignment,
)
from .tad_hierarchical_env_cfg import TadHierarchicalEnvCfg


def _gate_oob_pos(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return torch.where(x > 0, x * gate, x)


class TadHierarchicalEnv(QuadrotorEnvMixin, DirectMARLEnv):
    cfg: TadHierarchicalEnvCfg

    def __init__(self, cfg: TadHierarchicalEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize UAV teams, simulation buffers, and agent state."""
        super().__init__(cfg, render_mode, **kwargs)

        self._learning_set = set(self.cfg.possible_agents)

        self._flat = getattr(self.cfg, "agent_layout", "hier") == "flat"
        self._flat_pool_obs = (
            str(getattr(self.cfg, "flat_obs_mode", getattr(self.cfg, "flat_obs", "pool"))).lower() == "pool"
        )
        self._all_uids = list(self.cfg.all_uids)
        self._drone_index = {uid: i for i, uid in enumerate(self._all_uids)}
        self._team_indices = {
            team: [self._drone_index[uid] for uid in members] for team, members in self.cfg.team_members.items()
        }
        self._M = len(self._all_uids)
        self._n_pool = int(self.cfg.n_pool_max)
        self._n_dec = int(self.cfg.n_dec)
        self._dec_repr_dim = int(getattr(self.cfg, "dec_repr_dim", self._n_dec))
        self._block = self._dec_repr_dim + 4

        _team_id = {"T": 0, "A": 1, "D": 2}
        _ppt = int(self.cfg._pool_per_team)
        self._pool_slot_team_id = {}
        for team in ("T", "A", "D"):
            grid = list(self.cfg._selectable_grid[team])
            slot_team = [_team_id[grid[s // _ppt]] for s in range(self._n_pool)]
            self._pool_slot_team_id[team] = torch.tensor(slot_team, dtype=torch.long, device=self.device)
        self._team_id = _team_id

        _id2team = {0: "T", 1: "A", 2: "D"}
        self._pool_slot_uid_index = {}
        for team in ("T", "A", "D"):
            grid = list(self.cfg._selectable_grid[team])
            n_sel_slots = len(self.cfg.selectable_targets[team])
            slot_idx = []
            for s in range(self._n_pool):
                ts, mi = s // _ppt, s % _ppt
                t = grid[ts]
                members = self.cfg.team_members.get(t, [])

                ok_slot = (ts < n_sel_slots) and (mi < len(members))
                slot_idx.append(self._drone_index[members[mi]] if ok_slot else -1)
            self._pool_slot_uid_index[team] = torch.tensor(slot_idx, dtype=torch.long, device=self.device)

        self._q_low_input = str(getattr(self.cfg, "q_low_input", "none")).lower()
        self._q_low_rel_dim = int(getattr(self.cfg, "q_low_rel_dim", 7))
        if self._q_low_input == "none":
            self._q_low_extra_dim = 0
        elif self._q_low_input == "full":
            self._q_low_extra_dim = (self._M - 1) * (self._dec_repr_dim + 4 + self._q_low_rel_dim)
        else:
            self._q_low_extra_dim = 4 + self._q_low_rel_dim
        self._q_low_extra = {}

        self._lower_critic_mode = str(getattr(self.cfg, "lower_critic_mode", "per_agent")).lower()
        self._lower_critic_team = self._lower_critic_mode == "team"
        self._q_low_team_ctx_dim = self._M * (self._dec_repr_dim + 1 + 4) if self._lower_critic_team else 0
        self._q_low_team_ctx = (
            torch.zeros(self.num_envs, self._q_low_team_ctx_dim, device=self.device)
            if self._lower_critic_team
            else None
        )

        self._body_ids = {}
        for uid in self._all_uids:
            bid, _ = self.drones[uid].find_bodies("body")
            self._body_ids[uid] = bid

        first_uid = self._all_uids[0]
        self._mass = self.drones[first_uid].root_physx_view.get_masses()[0].sum()
        gravity_norm = torch.tensor(self.sim.cfg.gravity, device=self.device).norm().item()
        weight = (self._mass * gravity_norm).item()
        self._weight = weight
        self._thrust_max = self.cfg.thrust_to_weight * weight

        self._hover_action = torch.tensor([0.0, 0.0, 0.0, 0.0], device=self.device)

        if self.cfg.action_mode not in ("rates", "wrench"):
            raise ValueError(f"未知 action_mode={self.cfg.action_mode}，需为 'rates' 或 'wrench'")

        self._outcome: dict = {}

        self._alive = {uid: torch.ones(self.num_envs, dtype=torch.bool, device=self.device) for uid in self._all_uids}
        self._graveyard_z = -50.0

        self._locked_pool = {
            uid: torch.zeros(self.num_envs, dtype=torch.long, device=self.device) for uid in self._all_uids
        }
        self._lock_count = {
            uid: torch.zeros(self.num_envs, dtype=torch.long, device=self.device) for uid in self._all_uids
        }
        self._decision_boundary = {
            uid: torch.zeros(self.num_envs, dtype=torch.bool, device=self.device) for uid in self._all_uids
        }

        self._decisions_staged = False
        self._staged: dict | None = None

        self._eff_pool_idx = {
            uid: torch.zeros(self.num_envs, dtype=torch.long, device=self.device) for uid in self._all_uids
        }
        self._eff_dec_idx = {
            uid: torch.zeros(self.num_envs, dtype=torch.long, device=self.device) for uid in self._all_uids
        }
        self._eff_onehot_dec = {
            uid: torch.zeros(self.num_envs, self._n_dec, device=self.device) for uid in self._all_uids
        }

        self._executed_joint_action = torch.zeros(self.num_envs, self._M * self._block, device=self.device)

        self._executed_joint_ctrl = torch.zeros(self.num_envs, self._M * 4, device=self.device) if self._flat else None

        self.actions = {uid: torch.zeros(self.num_envs, 4, device=self.device) for uid in self._all_uids}

        self._episode_sums = {
            "reward_T": torch.zeros(self.num_envs, device=self.device),
            "reward_A": torch.zeros(self.num_envs, device=self.device),
            "reward_D": torch.zeros(self.num_envs, device=self.device),
            "attacker_win": torch.zeros(self.num_envs, device=self.device),
            "defender_win": torch.zeros(self.num_envs, device=self.device),
            "timeout": torch.zeros(self.num_envs, device=self.device),
            "episode_length": torch.zeros(self.num_envs, device=self.device),
        }
        self._cum_episodes: int = 0
        self._cum_attacker_wins: int = 0
        self._cum_defender_wins: int = 0
        self._cum_timeouts: int = 0

        import collections as _collections

        self._win_rate_horizons = [int(h) for h in (self.cfg.win_rate_horizons or [1000, 5000])]

        self._roll = {H: {"dq": _collections.deque(), "a": 0, "d": 0, "t": 0} for H in self._win_rate_horizons}

        self._telemetry_interval = max(1, int(self.cfg.telemetry_interval))
        self._telemetry_step = 0

        lo_h, hi_h = episode_horizon_step_bounds(
            self.cfg.episode_length_s, self.cfg.episode_length_jitter_s, self.step_dt
        )
        self._episode_horizon_lo, self._episode_horizon_hi = lo_h, hi_h
        self.episode_length_limit_buf = torch.full((self.num_envs,), hi_h, dtype=torch.long, device=self.device)

        self._pbrs_prev: dict[str, torch.Tensor] = {
            uid: torch.full((self.num_envs,), float("nan"), device=self.device) for uid in self._all_uids
        }

        self._ep_dist_sum: dict[str, torch.Tensor] = {s: torch.zeros((), device=self.device) for s in ("A_T", "D_A")}
        self._ep_dist_cnt: dict[str, torch.Tensor] = {s: torch.zeros((), device=self.device) for s in ("A_T", "D_A")}

        self._frozen_env: dict[str, torch.Tensor] = {
            team: torch.zeros(self.num_envs, dtype=torch.bool, device=self.device) for team in ("T", "A", "D")
        }

        self._expert_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._exper_teams: set[str] = set()

        self._roll_exper_mask: dict[str, torch.Tensor] = {
            team: torch.ones(self.num_envs, dtype=torch.bool, device=self.device) for team in ("T", "A", "D")
        }

        self._exper_rate: dict[str, float] = {"T": 1.0, "A": 1.0, "D": 1.0}

        _ot = str(getattr(self.cfg, "oob_teams", "all") or "all").strip().upper()
        if _ot == "ALL":
            self._oob_teams = ("T", "A", "D")
        elif _ot in ("T", "A", "D"):
            self._oob_teams = (_ot,)
        else:
            raise ValueError(f"oob_teams={_ot!r} 非法：需 T | A | D | all")
        print(
            f"[TAD][oob] 发放对象={'+'.join(self._oob_teams)} "
            f"dense_mask={self.cfg.oob_mask_dense} rew_oob={float(self.cfg.rew_oob)}"
        )
        self._expert_curriculum_on = bool(
            getattr(self.cfg, "expert_curriculum_enable", False)
            and float(getattr(self.cfg, "expert_curriculum_prob", 0.0)) > 0.0
        )
        print(
            f"[TAD][课程] expert_curriculum on={self._expert_curriculum_on} "
            f"prob={float(getattr(self.cfg, 'expert_curriculum_prob', 0.0)):.2f}"
        )

        self._trunc_final: tuple | None = None

        if self.cfg.curriculum_enable:
            self.cfg.capture_radius = max(
                float(self.cfg.capture_radius), float(self.cfg.curriculum_capture_radius_start)
            )
        self._curriculum_check_counter = 0

        self._test_num_envs = int(getattr(self.cfg, "test_num_envs", 0) or 0)
        if self._test_num_envs > 0:
            self._test_ids = torch.arange(self.num_envs - self._test_num_envs, self.num_envs, device=self.device)
            self._test_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._test_env[self._test_ids] = True
            self._test_every = max(1, int(getattr(self.cfg, "test_every_steps", 2500)))
            self._test_target_eps = max(1, int(getattr(self.cfg, "test_episodes", 16)))

            self._test_phase_active = False
            self._test_locked = False
            self._test_discard = None
            self._test_acc = {
                "episodes": 0,
                "attacker_win": 0,
                "defender_win": 0,
                "timeout": 0,
                "ep_len": 0.0,
                "reward_T": 0.0,
                "reward_A": 0.0,
                "reward_D": 0.0,
            }
            print(
                f"[TAD][test-env] 周期评估开启：{self._test_num_envs} 个 test env（尾部切片），"
                f"每 {self._test_every} control step 一相、每相满 {self._test_target_eps} 回合上报 Test/*"
            )

            self._test_hold: dict[str, torch.Tensor] | None = None

            self._test_row_cat: torch.Tensor | None = None
            self._test_cats: list[str] = []
            self._test_acc_by_cat: dict[str, dict] = {}
            if bool(getattr(self.cfg, "test_exper", False)):
                learn_teams = [
                    t for t in ("T", "A", "D") if any(a.split("/")[0].split("_")[0] == t for a in self._learning_set)
                ]
                if not learn_teams:
                    print("[TAD][test-env][WARN] test_exper=true 但无学习队 → 退化为全模型（旧行为）")
                else:
                    cats = torch.arange(self._test_num_envs, device=self.device) % len(learn_teams)
                    self._test_cats = learn_teams
                    self._test_row_cat = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
                    self._test_row_cat[self._test_ids] = cats
                    self._test_acc_by_cat = {t: self._new_test_acc() for t in learn_teams}
                    self._test_hold = {}
                    for t in ("T", "A", "D"):
                        if not self._team_indices.get(t):
                            continue
                        hold = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                        if t in learn_teams:
                            hold[self._test_ids] = cats != learn_teams.index(t)
                        else:
                            hold[self._test_ids] = True
                        self._test_hold[t] = hold
                    counts = {t: int((cats == i).sum()) for i, t in enumerate(learn_teams)}
                    if self._test_num_envs < len(learn_teams):
                        print(
                            f"[TAD][test-env][WARN] test_exper：test_num_envs={self._test_num_envs} "
                            f"< 类别数 {len(learn_teams)}，存在空类别"
                        )
                    print(
                        "[TAD][test-env] test_exper=true：模型队分布 "
                        + "、".join(f"{t}={counts[t]} env" for t in learn_teams)
                        + "（各类别内其余队=专家/脚本接管）"
                    )
        else:
            self._test_ids = None
            self._test_env = None

    def _setup_scene(self):
        all_uids = self.cfg.all_uids
        self.drones = {}
        for uid in all_uids:
            self.drones[uid] = Articulation(self.cfg.drone_articulation_cfg(uid))
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(color=(0.82, 0.82, 0.78)))
        self._spawn_env_grid_dividers()
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        for uid in all_uids:
            self.scene.articulations[uid] = self.drones[uid]
        dome_cfg = sim_utils.DomeLightCfg(intensity=600.0, color=(0.8, 0.8, 0.85))
        dome_cfg.func("/World/Light", dome_cfg)

    def _pre_physics_step(self, actions: dict) -> None:
        """Resolve target choices and prepare continuous control inputs."""

        if self._decisions_staged:
            self._decisions_staged = False
        else:
            self._resolve_targets(actions)
        self._execute_controls(actions)

    def _resolve_targets(self, actions: dict) -> None:
        nm_pool = self._n_pool
        dev = self.device
        roles = [u.split("_")[0] for u in self._all_uids]
        N, M = self.num_envs, self._M

        team_pos, team_vel, team_alive = self._team_stacks()
        own_pos = torch.stack([self.drones[u].data.root_pos_w for u in self._all_uids], dim=1)
        own_vel = torch.stack([self.drones[u].data.root_lin_vel_w for u in self._all_uids], dim=1)
        own_quat = torch.stack([self.drones[u].data.root_quat_w for u in self._all_uids], dim=1)

        pools = [self._selectable_pool(r, (team_pos, team_vel, team_alive)) for r in roles]
        pos_nm = torch.stack([p[0] for p in pools], dim=1)
        alive_nm = torch.stack([p[2] for p in pools], dim=1)
        rm_per_role = torch.stack([p[3] for p in pools], dim=0)
        real_mask = rm_per_role
        valid = real_mask.unsqueeze(0) & alive_nm

        ne = torch.arange(N, device=dev).view(-1, 1).expand(N, M)
        md = torch.arange(M, device=dev).view(1, -1).expand(N, M)

        locked_pool = torch.stack([self._locked_pool[u] for u in self._all_uids], dim=1)
        lock_count = torch.stack([self._lock_count[u] for u in self._all_uids], dim=1)
        lp = locked_pool.clamp(0, nm_pool - 1)
        locked_alive = valid[ne, md, lp]
        need = (lock_count <= 0) | (~locked_alive)

        dists0 = (pos_nm - own_pos.unsqueeze(2)).norm(dim=-1).masked_fill(~valid, float("inf"))
        _slot_team_id = torch.stack([self._pool_slot_team_id[r] for r in roles], dim=0)
        _primary_id = torch.tensor([self._team_id[TARGET_TEAM[r]] for r in roles], device=dev).view(-1, 1)
        dists0 = dists0.masked_fill((_slot_team_id != _primary_id).unsqueeze(0), float("inf"))
        new_pool = dists0.argmin(-1).long()

        ex_on = self._expert_curriculum_on and bool(self._expert_env.any())

        _fz_ad_list = (
            [self._frozen_env[t] for t in ("A", "D") if self._frozen_env.get(t) is not None]
            if float(self.cfg.frozen_opponent_prob) > 0.0
            else []
        )
        fz_ad = bool(torch.stack(_fz_ad_list).any()) if _fz_ad_list else False

        ex_any = bool(self._exper_teams)
        ex_hold: dict[str, torch.Tensor] = {}
        if ex_any:
            for _t in self._exper_teams:
                _m = self._roll_exper_mask[_t]
                ex_hold[_t] = _m if self._test_env is None else (_m & ~self._test_env)

        _th = getattr(self, "_test_hold", None)
        if _th is not None:
            ex_any = True
        _takeover: dict[str, torch.Tensor | None] = {}
        for _t in ("T", "A", "D"):
            _m = ex_hold.get(_t)
            _mt = _th.get(_t) if _th is not None else None
            if _mt is not None:
                _m = _mt if _m is None else (_m | _mt)
            _takeover[_t] = _m
        if ex_on or ex_any or fz_ad:
            _ppt_ex = int(self.cfg._pool_per_team)
            for team in ("A", "D"):
                idxs = self._team_indices.get(team, [])
                primary = TARGET_TEAM[team]
                grid_ex = list(self.cfg._selectable_grid[team])
                if (not idxs) or primary not in grid_ex or not self.cfg.team_members.get(primary):
                    continue
                mask = self._expert_env if ex_on else None
                _fz = self._frozen_env.get(team)
                if _fz is not None:
                    mask = _fz if mask is None else (mask | _fz)
                _tk = _takeover[team]
                if _tk is not None:
                    mask = _tk if mask is None else (mask | _tk)
                if mask is None:
                    continue
                slot_base = grid_ex.index(primary) * _ppt_ex
                idx_t = torch.tensor(idxs, device=dev)
                assign_t = min_total_distance_assignment(own_pos[:, idx_t], team_pos[primary], team_alive[primary])
                for k, m_idx in enumerate(idxs):
                    new_pool[:, m_idx] = torch.where(mask, assign_t[:, k] + slot_base, new_pool[:, m_idx])
        force_scripted = bool(getattr(self, "_force_scripted_policy", False))
        for m, uid in enumerate(self._all_uids):
            dec_aid = f"{uid}/dec"
            if (not force_scripted) and dec_aid in self._learning_set and dec_aid in actions:
                a_idx = self._parse_dec_action(actions[dec_aid])
                pi = a_idx % nm_pool

                fz_dec = self._frozen_env[uid.split("_")[0]]
                if ex_on and uid.split("_")[0] != "T":
                    fz_dec = fz_dec | self._expert_env
                _tk = _takeover[uid.split("_")[0]]
                if _tk is not None:
                    fz_dec = fz_dec | _tk
                new_pool[:, m] = torch.where(fz_dec, new_pool[:, m], pi)
        eff_pool = torch.where(need, new_pool, locked_pool)
        lock_count_new = torch.where(
            need, torch.full_like(lock_count, max(self.cfg.decision_period - 1, 0)), lock_count - 1
        )
        for m, uid in enumerate(self._all_uids):
            pi_c = eff_pool[:, m].clamp(0, nm_pool - 1)
            self._locked_pool[uid] = pi_c
            self._lock_count[uid] = lock_count_new[:, m]
            self._decision_boundary[uid] = need[:, m]

            self._eff_pool_idx[uid] = pi_c
            self._eff_dec_idx[uid] = pi_c
            self._eff_onehot_dec[uid] = torch.zeros(N, self._n_dec, device=dev)
            self._eff_onehot_dec[uid].scatter_(1, self._eff_dec_idx[uid].unsqueeze(-1), 1.0)

        _fz_teams = list(self._frozen_env)
        _fz_any = torch.stack([self._frozen_env[t] for t in _fz_teams]).any(dim=1).tolist()
        frozen_any_team = dict(zip(_fz_teams, _fz_any))

        for _t in self._exper_teams:
            frozen_any_team[_t] = True

        if _th is not None:
            for _t, _mt in _th.items():
                if _mt is not None:
                    frozen_any_team[_t] = True
        self._staged = {
            "team_stacks": (team_pos, team_vel, team_alive),
            "own_pos": own_pos,
            "own_vel": own_vel,
            "own_quat": own_quat,
            "pools": pools,
            "pos_nm": pos_nm,
            "valid": valid,
            "ne": ne,
            "md": md,
            "takeover": _takeover,
            "frozen_any_team": frozen_any_team,
            "ex_on": ex_on,
            "ex_any": ex_any,
            "fz_ad": fz_ad,
        }
        observer = getattr(self, "_play_decision_observer", None)
        if observer is not None:
            observer(actions, need, lock_count, locked_alive, _takeover, force_scripted)

    def _execute_controls(self, actions: dict) -> None:
        nm_pool = self._n_pool
        dev = self.device
        roles = [u.split("_")[0] for u in self._all_uids]
        N = self.num_envs
        st = self._staged
        team_pos = st["team_stacks"][0]
        own_pos, own_vel, own_quat = st["own_pos"], st["own_vel"], st["own_quat"]
        pools, pos_nm, valid = st["pools"], st["pos_nm"], st["valid"]
        ne, md = st["ne"], st["md"]
        _takeover, frozen_any_team = st["takeover"], st["frozen_any_team"]
        ex_on, ex_any, fz_ad = st["ex_on"], st["ex_any"], st["fz_ad"]
        force_scripted = bool(getattr(self, "_force_scripted_policy", False))

        locked_pool = torch.stack([self._locked_pool[u] for u in self._all_uids], dim=1)
        ep = locked_pool.clamp(0, nm_pool - 1)
        tgt_pos = pos_nm[ne, md, ep]
        tgt_alive = valid[ne, md, ep]
        sign = self._role_sign(roles, dev)

        def _rl_aid(uid: str) -> str:
            return uid if self._flat else f"{uid}/ctrl"

        scripted_uids = [
            uid
            for uid in self._all_uids
            if force_scripted or _rl_aid(uid) not in self._learning_set or frozen_any_team[uid.split("_")[0]]
        ]
        if scripted_uids:
            idx_t = torch.tensor([self._drone_index[u] for u in scripted_uids], device=dev)
            ctrl_sub = self._compute_cascade_target_action_batch(
                scripted_uids,
                own_pos[:, idx_t],
                own_vel[:, idx_t],
                own_quat[:, idx_t],
                tgt_pos[:, idx_t],
                tgt_alive[:, idx_t],
                sign=sign[:, idx_t] if sign is not None else None,
            )
            for k, uid in enumerate(scripted_uids):
                self.actions[uid] = ctrl_sub[:, k]

        for m, uid in enumerate(self._all_uids):
            rl_aid = _rl_aid(uid)
            if (not force_scripted) and rl_aid in self._learning_set and rl_aid in actions:
                rl_a = actions[rl_aid].clamp(-1.0, 1.0)
                fz_ctrl = self._frozen_env[uid.split("_")[0]]

                _tk = _takeover[uid.split("_")[0]]
                if _tk is not None:
                    fz_ctrl = fz_ctrl | _tk
                self.actions[uid] = torch.where(fz_ctrl.unsqueeze(-1), self.actions[uid], rl_a)

        if ex_on or ex_any or fz_ad:
            vel_nm = torch.stack([p[1] for p in pools], dim=1)
            tgt_vel = vel_nm[ne, md, ep]
            ppn_uids = [
                u
                for u in self._all_uids
                if u.split("_")[0] in ("A", "D") and (ex_on or fz_ad or _takeover[u.split("_")[0]] is not None)
            ]
            idx_p = torch.tensor([self._drone_index[u] for u in ppn_uids], device=dev)
            ppn_sub = self._compute_ppn_target_action_batch(
                ppn_uids,
                own_pos[:, idx_p],
                own_vel[:, idx_p],
                own_quat[:, idx_p],
                tgt_pos[:, idx_p],
                tgt_vel[:, idx_p],
                tgt_alive[:, idx_p],
            )
            for k, uid in enumerate(ppn_uids):
                col = self._expert_env if ex_on else None
                _fz_c = self._frozen_env.get(uid.split("_")[0])
                if _fz_c is not None:
                    col = _fz_c if col is None else (col | _fz_c)
                _tk = _takeover[uid.split("_")[0]]
                if _tk is not None:
                    col = _tk if col is None else (col | _tk)
                self.actions[uid] = torch.where(col.unsqueeze(-1), ppn_sub[:, k], self.actions[uid])

        for uid in self._all_uids:
            dead = ~self._alive[uid]
            self.actions[uid] = torch.where(dead.unsqueeze(-1), self._hover_action, self.actions[uid])

        K = self.cfg.hover_warmup_steps
        if K > 0:
            warmup = self.episode_length_buf < K
            if warmup.any():
                for uid in self._all_uids:
                    self.actions[uid][warmup] = self.compute_warmup_stabilize_action(uid)[warmup]

        self._assemble_full_joint_action()
        self._staged = None

    def _assemble_full_joint_action(self) -> None:
        ja = torch.zeros(self.num_envs, self._M * self._block, device=self.device)
        for uid in self._all_uids:
            base = self._drone_index[uid] * self._block
            dec_block = self._eff_onehot_dec[uid]
            ja[:, base : base + self._dec_repr_dim] = dec_block.detach()
            ja[:, base + self._dec_repr_dim : base + self._block] = self.actions[uid].detach()
        self._executed_joint_action = ja

        if self._flat:
            torch.cat([self.actions[uid].detach() for uid in self._all_uids], dim=-1, out=self._executed_joint_ctrl)

        if self._lower_critic_team:
            self._assemble_q_low_team_ctx()
        else:
            self._assemble_q_low_extra()

    def _q_low_rel_state(
        self, own_uid: str, tgt_pos: torch.Tensor, tgt_vel: torch.Tensor, tgt_alive: torch.Tensor
    ) -> torch.Tensor:
        own = self.drones[own_uid]
        own_pos = own.data.root_pos_w
        own_quat = own.data.root_quat_w
        q_inv = quat_inv(own_quat)
        rel_pos_w = tgt_pos - own_pos
        rel_vel_w = tgt_vel - own.data.root_lin_vel_w
        rel_pos_b = quat_apply(q_inv, rel_pos_w)
        rel_vel_b = quat_apply(q_inv, rel_vel_w)
        m = (tgt_alive & self._alive[own_uid]).to(rel_pos_b.dtype).unsqueeze(-1)
        rel_pos_b = rel_pos_b * m
        rel_vel_b = rel_vel_b * m
        alive = m
        return torch.cat([rel_pos_b, rel_vel_b, alive], dim=-1)

    def _assemble_q_low_extra(self) -> None:
        if self._q_low_input == "none":
            self._q_low_extra = {}
            return
        N = self.num_envs
        rel_dim = self._q_low_rel_dim
        dec_cache = "_eff_onehot_dec"

        pos_all = torch.stack([self.drones[u].data.root_pos_w for u in self._all_uids], dim=1)
        vel_all = torch.stack([self.drones[u].data.root_lin_vel_w for u in self._all_uids], dim=1)
        alive_all = torch.stack([self._alive[u] for u in self._all_uids], dim=1)
        ne = torch.arange(N, device=self.device)
        extra_map: dict = {}
        for uid in self._all_uids:
            team = uid.split("_")[0]
            i = self._drone_index[uid]
            own_quat = self.drones[uid].data.root_quat_w
            q_inv = quat_inv(own_quat)
            if self._q_low_input == "full":
                blocks = []
                for j, ouid in enumerate(self._all_uids):
                    if ouid == uid:
                        continue
                    dec_block = getattr(self, dec_cache)[ouid].detach()
                    ctrl_block = self.actions[ouid].detach()

                    rel_pos_w = pos_all[:, j] - pos_all[:, i]
                    rel_vel_w = vel_all[:, j] - vel_all[:, i]
                    rel_pos_b = quat_apply(q_inv, rel_pos_w)
                    rel_vel_b = quat_apply(q_inv, rel_vel_w)
                    m = (alive_all[:, j] & alive_all[:, i]).to(rel_pos_b.dtype).unsqueeze(-1)
                    rel = torch.cat([rel_pos_b * m, rel_vel_b * m, m], dim=-1)

                    blocks.append(torch.cat([dec_block * m, ctrl_block * m, rel], dim=-1))
                extra_map[uid] = torch.cat(blocks, dim=-1)
            else:
                lp = self._locked_pool[uid].clamp(0, self._n_pool - 1)
                slot_idx = self._pool_slot_uid_index[team]
                tgt_idx = slot_idx[lp]
                valid_slot = tgt_idx >= 0
                tgt_idx_safe = tgt_idx.clamp(min=0)
                t_pos = pos_all[ne, tgt_idx_safe]
                t_vel = vel_all[ne, tgt_idx_safe]
                t_alive = alive_all[ne, tgt_idx_safe] & valid_slot
                act_all = torch.stack([self.actions[u].detach() for u in self._all_uids], dim=1)
                ctrl_block = act_all[ne, tgt_idx_safe]

                m_sel = (t_alive & self._alive[uid]).to(ctrl_block.dtype).unsqueeze(-1)
                rel = self._q_low_rel_state(uid, t_pos, t_vel, t_alive)
                extra_map[uid] = torch.cat([ctrl_block * m_sel, rel], dim=-1)
        self._q_low_extra = extra_map

    def _assemble_q_low_team_ctx(self) -> None:
        dec_cache = "_eff_onehot_dec"
        dec_flat = torch.cat([getattr(self, dec_cache)[u].detach() for u in self._all_uids], dim=-1)
        lock_frac = torch.stack([self._lock_count[u].float() for u in self._all_uids], dim=-1) / float(
            max(int(self.cfg.decision_period), 1)
        )
        ctrl_flat = torch.cat([self.actions[u].detach() for u in self._all_uids], dim=-1)
        self._q_low_team_ctx = torch.cat([dec_flat, lock_frac, ctrl_flat], dim=-1)

    def _parse_dec_action(self, a: torch.Tensor) -> torch.Tensor:
        if a.dim() >= 2 and a.shape[-1] > 1:
            return a.argmax(dim=-1).long()
        return a.long().view(-1)

    _ROLE_SIGN = {"T": -1.0, "A": 1.0, "D": 1.0}

    def _role_sign(self, roles: list[str], dev) -> torch.Tensor:
        signs = torch.tensor([self._ROLE_SIGN[r] for r in roles], device=dev)
        return signs.unsqueeze(0).expand(self.num_envs, -1)

    def _apply_action(self) -> None:
        """Apply the prepared forces and moments to the UAVs."""
        for uid in self._all_uids:
            self._enforce_hard_attitude_limit(uid)
            self._enforce_dead_drone_state(uid)
        for uid in self._all_uids:
            self._apply_drone_action(uid, self.actions[uid], active_mask=self._alive[uid])

    def _alive_stack(self, team: str) -> torch.Tensor:
        members = self.cfg.team_members[team]
        if not members:
            return torch.zeros(self.num_envs, 0, dtype=torch.bool, device=self.device)
        return torch.stack([self._alive[u] for u in members], dim=1)

    def _team_stacks(self) -> tuple:
        pos_all = torch.stack([self.drones[uid].data.root_pos_w for uid in self._all_uids], dim=1)
        vel_all = torch.stack([self.drones[uid].data.root_lin_vel_w for uid in self._all_uids], dim=1)
        alive_all = torch.stack([self._alive[uid] for uid in self._all_uids], dim=1)
        pos_teams = {team: pos_all[:, indices] for team, indices in self._team_indices.items()}
        vel_teams = {team: vel_all[:, indices] for team, indices in self._team_indices.items()}
        alive_teams = {team: alive_all[:, indices] for team, indices in self._team_indices.items()}
        return pos_teams, vel_teams, alive_teams

    @staticmethod
    def _seg_closest_point(p: torch.Tensor, q: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        pq = q - p
        denom = (pq * pq).sum(dim=-1).clamp(min=1e-8)
        t = ((x - p) * pq).sum(dim=-1) / denom
        t = t.clamp(0.0, 1.0).unsqueeze(-1)
        return p + t * pq

    @staticmethod
    def _nearest_alive_pos(pos_self: torch.Tensor, pos_team: torch.Tensor, alive_team: torch.Tensor) -> torch.Tensor:
        n = pos_team.shape[1]
        if n == 0:
            return pos_self.clone()
        d = torch.cdist(pos_self.unsqueeze(1), pos_team).squeeze(1)
        d = d.masked_fill(~alive_team.bool(), float("inf"))
        has = d.isfinite().any(dim=-1)
        idx = d.argmin(dim=-1)
        rows = torch.arange(pos_self.shape[0], device=pos_self.device)
        nearest = pos_team[rows, idx]
        return torch.where(has.unsqueeze(-1), nearest, pos_self.clone())

    @staticmethod
    def _pad_team(x: torch.Tensor, nm: int) -> torch.Tensor:
        n = x.shape[1]
        if n >= nm:
            return x[:, :nm]
        pad = torch.zeros(x.shape[0], nm - n, 3, device=x.device, dtype=x.dtype)
        return torch.cat([x, pad], dim=1)

    @staticmethod
    def _pad_alive(x: torch.Tensor, nm: int) -> torch.Tensor:
        n = x.shape[1]
        if n >= nm:
            return x[:, :nm]
        pad = torch.zeros(x.shape[0], nm - n, dtype=torch.bool, device=x.device)
        return torch.cat([x, pad], dim=1)

    def _team_distances(self) -> tuple:
        pos, _, _ = self._team_stacks()
        d_TA = min_pair_distance(pos["T"], pos["A"])
        d_AD = min_pair_distance(pos["A"], pos["D"])
        return d_TA, d_AD

    def _compute_outcome(self) -> dict:
        members = self.cfg.team_members
        was_alive = {uid: self._alive[uid].clone() for uid in self._all_uids}
        pos, vel, alive_teams = self._team_stacks()
        alive_T0 = alive_teams["T"]
        alive_A0 = alive_teams["A"]
        alive_D0 = alive_teams["D"]

        new_T_captured, new_A_capturing = pair_engagement(
            pos["T"], pos["A"], alive_T0, alive_A0, self.cfg.capture_radius
        )
        new_A_intercepted, new_D_intercepting = pair_engagement(
            pos["A"], pos["D"], alive_A0, alive_D0, self.cfg.capture_radius
        )

        new_A_dead_cap = new_A_capturing
        new_A_dead_int = new_A_intercepted & ~new_A_capturing

        _catcher_survives = self.cfg.elimination_mode == "catcher_survives"
        _never_dead = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        new_A_dead = new_A_dead_int if _catcher_survives else (new_A_dead_cap | new_A_dead_int)
        f_T_captured, f_A_capture, f_A_intercepted, f_D_intercept = {}, {}, {}, {}
        newly_dead = {}
        for i, uid in enumerate(members["T"]):
            f_T_captured[uid] = new_T_captured[:, i]
            newly_dead[uid] = new_T_captured[:, i]
        for i, uid in enumerate(members["A"]):
            f_A_capture[uid] = new_A_dead_cap[:, i]
            f_A_intercepted[uid] = new_A_dead_int[:, i]
            newly_dead[uid] = new_A_dead[:, i]
        for i, uid in enumerate(members["D"]):
            f_D_intercept[uid] = new_D_intercepting[:, i]

            newly_dead[uid] = _never_dead if _catcher_survives else new_D_intercepting[:, i]

        for uid in self._all_uids:
            self._alive[uid] = was_alive[uid] & ~newly_dead[uid]
            self._enforce_dead_drone_state(uid)

        alive_T = self._alive_stack("T")
        alive_A = self._alive_stack("A")
        all_T_dead = alive_T.sum(dim=1) == 0
        all_A_dead = alive_A.sum(dim=1) == 0

        terminated = all_T_dead | all_A_dead
        attacker_win = all_T_dead
        defender_win = all_A_dead & ~all_T_dead

        time_out = (self.episode_length_buf >= self.episode_length_limit_buf) & ~terminated

        alive_D = self._alive_stack("D")

        c_los_A_ta, c_los_T = nearest_pair_los_vel(pos["A"], vel["A"], pos["T"], vel["T"], alive_A, alive_T)
        c_los_D, _ = nearest_pair_los_vel(pos["D"], vel["D"], pos["A"], vel["A"], alive_D, alive_A)
        c_los = {}
        for uid in members["A"]:
            c_los[uid] = c_los_A_ta
        for uid in members["T"]:
            c_los[uid] = c_los_T
        for uid in members["D"]:
            c_los[uid] = c_los_D

        still_alive = {uid: self._alive[uid].clone() for uid in self._all_uids}
        team_alive_count = {}
        for team, mem in members.items():
            if mem:
                team_alive_count[team] = torch.stack([was_alive[u] for u in mem], dim=1).sum(dim=1).float()
            else:
                team_alive_count[team] = torch.zeros(self.num_envs, device=self.device)

        c_own = {}
        c_threat = {}
        locked_pos = {}
        locked_vel = {}
        locked_alive = {}
        team_stacks = (pos, vel, alive_teams)

        _pool_nm = {t: self._team_pool_stacks(t, team_stacks) for t in ("T", "A", "D") if self.cfg.team_members.get(t)}
        n_pool = self._n_pool
        ne = torch.arange(self.num_envs, device=self.device)
        for uid in self._all_uids:
            team = uid.split("_")[0]
            mi = int(uid.split("_")[1])

            self_pos = pos[team][:, mi, :]
            self_vel = vel[team][:, mi, :]
            pos_nm, vel_nm, alive_nm = _pool_nm[team]
            lp = self._locked_pool[uid].clamp(0, n_pool - 1)
            tgt_pos = pos_nm[ne, lp]
            tgt_vel = vel_nm[ne, lp]
            tgt_alive = alive_nm[ne, lp]
            c_own[uid] = per_drone_closing(self_pos, self_vel, tgt_pos, tgt_vel, was_alive[uid], tgt_alive)
            locked_pos[uid] = tgt_pos
            locked_vel[uid] = tgt_vel
            locked_alive[uid] = tgt_alive
            c_threat[uid] = torch.zeros(self.num_envs, device=self.device)

        if self.cfg.team_members.get("A") and self.cfg.team_members.get("D"):
            a_uids = self.cfg.team_members["A"]
            a_pos = pos["A"]
            a_vel = vel["A"]
            a_alive = torch.stack([was_alive[u] for u in a_uids], dim=1)
            c_threat_A = per_drone_nearest_team_closing_batch(a_pos, a_vel, pos["D"], vel["D"], a_alive, alive_D)
            for i, u in enumerate(a_uids):
                c_threat[u] = c_threat_A[:, i]

        c_close = {}
        d_close = {}
        ct = self.cfg.closing_target
        if ct == "nearest":
            alive_map = {"T": alive_T, "A": alive_A, "D": alive_D}
            primary_target = {"T": "A", "A": "T", "D": "A"}
            for uid in self._all_uids:
                team = uid.split("_")[0]
                primary = primary_target[team]
                mi = int(uid.split("_")[1])
                self_pos = pos[team][:, mi, :]
                self_vel = vel[team][:, mi, :]
                c_close[uid] = per_drone_nearest_team_closing(
                    self_pos, self_vel, pos[primary], vel[primary], was_alive[uid], alive_map[primary]
                )
        elif ct == "nearest1":
            alive_map = {"T": alive_T, "A": alive_A, "D": alive_D}
            primary_target = {"T": "A", "A": "T", "D": "A"}
            for team in ("T", "A", "D"):
                members = self.cfg.team_members.get(team) or []
                if not members:
                    continue
                primary = primary_target[team]
                if not self.cfg.team_members.get(primary):
                    for uid in members:
                        c_close[uid] = torch.zeros(self.num_envs, device=self.device)
                    continue
                chaser_pos = pos[team]
                chaser_vel = vel[team]
                chaser_alive = torch.stack([was_alive[u] for u in members], dim=1)
                c_team = greedy_assigned_team_closing(
                    chaser_pos, chaser_vel, pos[primary], vel[primary], chaser_alive, alive_map[primary]
                )
                for i, uid in enumerate(members):
                    c_close[uid] = c_team[:, i]
        elif ct == "distance":
            d_close = {uid: torch.zeros(self.num_envs, device=self.device) for uid in self._all_uids}

            pos_T, pos_A, pos_D = pos.get("T"), pos.get("A"), pos.get("D")
            if members["T"] and members["A"]:
                was_T = torch.stack([was_alive[u] for u in self.cfg.team_members["T"]], dim=1)
                d_T = per_drone_nearest_team_distance_batch(pos_T, pos_A, was_T, alive_A)
                for i, u in enumerate(self.cfg.team_members["T"]):
                    d_close[u] = d_T[:, i]
            if members["A"] and members["T"]:
                was_A = torch.stack([was_alive[u] for u in self.cfg.team_members["A"]], dim=1)
                d_A = per_drone_team_distance_sum_batch(pos_A, pos_T, was_A, alive_T)
                for i, u in enumerate(self.cfg.team_members["A"]):
                    d_close[u] = d_A[:, i]
            if members["D"] and members["A"]:
                was_D = torch.stack([was_alive[u] for u in self.cfg.team_members["D"]], dim=1)
                d_D = per_drone_team_distance_sum_batch(pos_D, pos_A, was_D, alive_A)
                for i, u in enumerate(self.cfg.team_members["D"]):
                    d_close[u] = d_D[:, i]

            c_close = c_own
        elif ct == "pbrs":
            d_close = {uid: torch.zeros(self.num_envs, device=self.device) for uid in self._all_uids}

            pos_T, pos_A, pos_D = pos.get("T"), pos.get("A"), pos.get("D")
            if members["T"] and members["A"]:
                was_T = torch.stack([was_alive[u] for u in self.cfg.team_members["T"]], dim=1)
                d_T = per_drone_nearest_team_distance_batch(pos_T, pos_A, was_T, alive_A)
                for i, u in enumerate(self.cfg.team_members["T"]):
                    d_close[u] = d_T[:, i]
            if members["A"] and members["T"]:
                was_A = torch.stack([was_alive[u] for u in self.cfg.team_members["A"]], dim=1)
                d_A = per_drone_nearest_team_distance_batch(pos_A, pos_T, was_A, alive_T)
                for i, u in enumerate(self.cfg.team_members["A"]):
                    d_close[u] = d_A[:, i]
            if members["D"] and members["A"]:
                was_D = torch.stack([was_alive[u] for u in self.cfg.team_members["D"]], dim=1)
                d_D = per_drone_nearest_team_distance_batch(pos_D, pos_A, was_D, alive_A)
                for i, u in enumerate(self.cfg.team_members["D"]):
                    d_close[u] = d_D[:, i]

            if getattr(self.cfg, "pbrs_on_locked", False):
                _pool_cache = {}
                for u in self._all_uids:
                    _team = u.split("_")[0]
                    if _team not in _pool_cache:
                        _pool_cache[_team] = self._team_pool_stacks(_team, (pos, vel, alive_teams))
                    tgt_pos, _, tgt_alive = self._locked_target_state(u, None, _pool_cache[_team])
                    d_lock = (self.drones[u].data.root_pos_w - tgt_pos).norm(dim=-1)
                    d_close[u] = torch.where(tgt_alive & was_alive[u], d_lock, d_close[u])
            c_close = c_own
        else:
            c_close = c_own
        return {
            "c_los": c_los,
            "c_own": c_own,
            "c_close": c_close,
            "d_close": d_close,
            "c_threat": c_threat,
            "locked_pos": locked_pos,
            "locked_vel": locked_vel,
            "locked_alive": locked_alive,
            "team_pos": pos,
            "team_vel": vel,
            "team_alive": alive_teams,
            "terminated": terminated,
            "attacker_win": attacker_win,
            "defender_win": defender_win,
            "time_out": time_out,
            "was_alive": was_alive,
            "still_alive": still_alive,
            "newly_dead": newly_dead,
            "team_alive_count": team_alive_count,
            "f_T_captured": f_T_captured,
            "f_A_capture": f_A_capture,
            "f_A_intercepted": f_A_intercepted,
            "f_D_intercept": f_D_intercept,
        }

    def _get_observations(self) -> dict:
        """Build policy observations and centralized critic inputs."""
        if self._flat:
            return self._get_flat_observations()
        team_stacks = self._team_stacks()
        pos_teams, vel_teams, alive_teams = team_stacks
        full = self.cfg.obs_full_obs

        states_by_aid = self._get_states() if full else None
        pool_nm = {}
        if not full:
            ctrl_teams = {aid.split("/")[0].split("_")[0] for aid in self.cfg.possible_agents if aid.endswith("/ctrl")}
            pool_nm = {team: self._team_pool_stacks(team, team_stacks) for team in ctrl_teams}

        dec_teams = {aid.split("/")[0].split("_")[0] for aid in self.cfg.possible_agents if aid.endswith("/dec")}
        upper_pool = {team: self._selectable_pool(team, team_stacks) for team in dec_teams}
        obs = {}
        for aid in self.cfg.possible_agents:
            uid = aid.split("/")[0]
            if aid.endswith("/dec"):
                obs[aid] = (
                    self._build_full_upper_obs(uid, team_stacks, states_by_aid[aid], pool=upper_pool[uid.split("_")[0]])
                    if full
                    else self._build_upper_obs(
                        uid, pos_teams, vel_teams, alive_teams, pool=upper_pool[uid.split("_")[0]]
                    )
                )
            else:
                obs[aid] = (
                    self._build_full_lower_obs(uid, states_by_aid[aid])
                    if full
                    else self._build_lower_obs(uid, team_stacks, pool_nm[uid.split("_")[0]])
                )
        return obs

    def _build_upper_obs(self, uid, pos_teams, vel_teams, alive_teams, pool: tuple | None = None) -> torch.Tensor:
        n_pool = self._n_pool
        drone = self.drones[uid]
        own_pos = drone.data.root_pos_w
        own_quat = drone.data.root_quat_w
        own_vel = drone.data.root_lin_vel_w
        origins = self.scene.env_origins
        pieces = []
        if self.cfg.obs_include_own:
            pieces.append(drone.data.root_lin_vel_b)
            pieces.append(drone.data.root_ang_vel_b)
            pieces.append(drone.data.projected_gravity_b)
            pieces.append((own_pos[:, 2] - origins[:, 2]).unsqueeze(-1))
            _, _, yaw = euler_xyz_from_quat(own_quat)
            pieces.append(torch.stack([torch.sin(yaw), torch.cos(yaw)], dim=-1))

        team = uid.split("_")[0]
        cpos, cvel, calive, rm_1d = (
            pool if pool is not None else self._selectable_pool(team, (pos_teams, vel_teams, alive_teams))
        )
        real_mask = rm_1d.unsqueeze(0).expand(own_pos.shape[0], -1)
        feats_flat, mask = self._candidate_features(own_pos, own_quat, own_vel, cpos, cvel, calive, real_mask, n_pool)

        if self.cfg.obs_include_lock_count:
            lock_cnt = self._team_lock_count(team, uid) * mask
            feats_flat = torch.cat([feats_flat, lock_cnt], dim=-1)

        ttg = self._time_frac()
        if ttg is not None:
            pieces.append(ttg)
        pieces.append(feats_flat)
        pieces.append(mask)

        obs = torch.cat(pieces, dim=-1)
        alive = self._alive[uid].unsqueeze(-1)
        return torch.where(alive, obs, torch.zeros_like(obs))

    def _time_frac(self) -> torch.Tensor | None:
        if not self.cfg.obs_time_to_go:
            return None
        t = self.episode_length_buf.to(torch.float32)
        horizon = self.episode_length_limit_buf.to(torch.float32).clamp(min=1.0)
        return (t / horizon).clamp(0.0, 1.0).unsqueeze(-1)

    def _get_flat_observations(self) -> dict:
        team_stacks = self._team_stacks()
        teams = {aid.split("_")[0] for aid in self.cfg.possible_agents}
        obs = {}
        if self._flat_pool_obs:
            pool_by_team = {t: self._selectable_pool(t, team_stacks) for t in teams}
            for aid in self.cfg.possible_agents:
                obs[aid] = self._build_flat_obs(aid, pool_by_team[aid.split("_")[0]])
        else:
            peer_by_team = {t: self._flat_peer_stacks(t, team_stacks) for t in teams}
            for aid in self.cfg.possible_agents:
                obs[aid] = self._build_flat_obs_peers(aid, peer_by_team[aid.split("_")[0]])
        return obs

    def _flat_own_block(self, uid: str) -> torch.Tensor:
        drone = self.drones[uid]
        pos_local = drone.data.root_pos_w - self.scene.env_origins
        _, _, yaw = euler_xyz_from_quat(drone.data.root_quat_w)
        ttg = self._time_frac()
        if ttg is None:
            ttg = torch.zeros(self.num_envs, 1, device=self.device)
        obs = torch.cat(
            [
                pos_local,
                drone.data.root_lin_vel_b,
                drone.data.root_ang_vel_b,
                drone.data.projected_gravity_b,
                torch.stack([torch.sin(yaw), torch.cos(yaw)], dim=-1),
                ttg,
            ],
            dim=-1,
        )
        alive = self._alive[uid].unsqueeze(-1)
        return torch.where(alive, obs, torch.zeros_like(obs))

    def _build_flat_obs(self, uid: str, pool: tuple) -> torch.Tensor:
        drone = self.drones[uid]
        own_pos = drone.data.root_pos_w
        own_quat = drone.data.root_quat_w
        own_vel = drone.data.root_lin_vel_w
        cpos, cvel, calive, rm_1d = pool
        real_mask = rm_1d.unsqueeze(0).expand(self.num_envs, -1)
        feats_flat, mask = self._candidate_features(
            own_pos, own_quat, own_vel, cpos, cvel, calive, real_mask, self._n_pool
        )
        obs = torch.cat([self._flat_own_block(uid), feats_flat, mask], dim=-1)
        alive = self._alive[uid].unsqueeze(-1)
        return torch.where(alive, obs, torch.zeros_like(obs))

    def _flat_peer_stacks(self, team: str, team_stacks: tuple):
        layout_all = []
        seen = set()
        for u in self.cfg.team_members[team]:
            for p in getattr(self.cfg, "flat_peer_layout", {}).get(u, []):
                if p not in seen:
                    seen.add(p)
                    layout_all.append(p)
        pos_teams, vel_teams, alive_teams = team_stacks
        pos_parts, vel_parts, alive_parts = [], [], []
        for u in layout_all:
            t, k = u.split("_")[0], int(u.split("_")[1])
            pos_parts.append(pos_teams[t][:, k])
            vel_parts.append(vel_teams[t][:, k])
            alive_parts.append(alive_teams[t][:, k])
        pos = torch.stack(pos_parts, dim=1)
        vel = torch.stack(vel_parts, dim=1)
        alive = torch.stack(alive_parts, dim=1)
        team_oh = torch.zeros(len(layout_all), 3, device=self.device)
        for i, u in enumerate(layout_all):
            team_oh[i, self._team_id[u.split("_")[0]]] = 1.0
        index = {u: i for i, u in enumerate(layout_all)}
        return pos, vel, alive, team_oh, index

    def _build_flat_obs_peers(self, uid: str, peer: tuple) -> torch.Tensor:
        ppos, pvel, palive, team_oh, index = peer
        layout = self.cfg.flat_peer_layout[uid]
        idx = torch.tensor([index[u] for u in layout], device=self.device)
        ppos_i = ppos.index_select(1, idx)
        pvel_i = pvel.index_select(1, idx)
        palive_i = palive.index_select(1, idx)
        oh_i = team_oh.index_select(0, idx)
        drone = self.drones[uid]
        own_pos, own_quat, own_vel = drone.data.root_pos_w, drone.data.root_quat_w, drone.data.root_lin_vel_w
        R = len(layout)
        rel_pos_w = ppos_i - own_pos.unsqueeze(1)
        rel_vel_w = pvel_i - own_vel.unsqueeze(1)
        q_inv = quat_inv(own_quat)
        q_inv_exp = q_inv.unsqueeze(1).expand(self.num_envs, R, 4).reshape(self.num_envs * R, 4)
        rel_pos_b = quat_apply(q_inv_exp, rel_pos_w.reshape(-1, 3)).reshape(self.num_envs, R, 3)
        rel_vel_b = quat_apply(q_inv_exp, rel_vel_w.reshape(-1, 3)).reshape(self.num_envs, R, 3)
        valid = palive_i.bool()
        vm = valid.unsqueeze(-1).to(rel_pos_b.dtype)

        feats = torch.cat([rel_pos_b * vm, rel_vel_b * vm, valid.to(rel_pos_b.dtype).unsqueeze(-1)], dim=-1)
        oh = oh_i.view(1, R, 3).expand(self.num_envs, -1, -1)
        peer_block = torch.cat([oh, feats], dim=-1).reshape(self.num_envs, 10 * R)
        obs = torch.cat([self._flat_own_block(uid), peer_block], dim=-1)
        alive = self._alive[uid].unsqueeze(-1)
        return torch.where(alive, obs, torch.zeros_like(obs))

    def _build_full_upper_obs(self, uid, team_stacks, state, pool: tuple | None = None) -> torch.Tensor:
        n_pool = self._n_pool
        drone = self.drones[uid]
        own_pos = drone.data.root_pos_w
        own_quat = drone.data.root_quat_w
        own_vel = drone.data.root_lin_vel_w
        team = uid.split("_")[0]
        cpos, cvel, calive, rm_1d = pool if pool is not None else self._selectable_pool(team, team_stacks)
        real_mask = rm_1d.unsqueeze(0).expand(own_pos.shape[0], -1)
        feats_flat, mask = self._candidate_features(own_pos, own_quat, own_vel, cpos, cvel, calive, real_mask, n_pool)

        if self.cfg.obs_include_lock_count:
            lock_cnt = self._team_lock_count(team, uid) * mask
            feats_flat = torch.cat([feats_flat, lock_cnt], dim=-1)
        pieces = [state]
        pieces.append(feats_flat)
        pieces.append(mask)
        obs = torch.cat(pieces, dim=-1)
        alive = self._alive[uid].unsqueeze(-1)
        return torch.where(alive, obs, torch.zeros_like(obs))

    def _team_lock_count(self, team: str, exclude_uid: str) -> torch.Tensor:
        P = self._n_pool
        members = self.cfg.team_members[team]
        if not members:
            return torch.zeros(self.num_envs, P, device=self.device)
        alive = self._alive_stack(team)
        lp = torch.stack([self._locked_pool[u] for u in members], dim=1)
        onehot = torch.nn.functional.one_hot(lp.clamp(0, P - 1), P)
        cnt = (onehot.to(alive.dtype) * alive.unsqueeze(-1)).sum(1)
        if exclude_uid in members:
            self_oh = torch.nn.functional.one_hot(self._locked_pool[exclude_uid].clamp(0, P - 1), P).to(cnt.dtype)
            cnt = (cnt - self_oh).clamp(min=0)
        return cnt

    def _candidate_features(self, own_pos, own_quat, own_vel, cand_pos, cand_vel, cand_alive, real_mask, nm):
        N = own_pos.shape[0]
        rel_pos_w = cand_pos - own_pos.unsqueeze(1)
        rel_vel_w = cand_vel - own_vel.unsqueeze(1)
        q_inv = quat_inv(own_quat)
        q_inv_exp = q_inv.unsqueeze(1).expand(N, nm, 4).reshape(N * nm, 4)
        rel_pos_b = quat_apply(q_inv_exp, rel_pos_w.reshape(N * nm, 3)).reshape(N, nm, 3)
        rel_vel_b = quat_apply(q_inv_exp, rel_vel_w.reshape(N * nm, 3)).reshape(N, nm, 3)
        dist = rel_pos_w.norm(dim=-1)
        valid = real_mask & cand_alive
        vm = valid.unsqueeze(-1).to(rel_pos_b.dtype)
        feats = torch.cat([rel_pos_b * vm, rel_vel_b * vm, (dist * valid.to(dist.dtype)).unsqueeze(-1)], dim=-1)
        return feats.reshape(N, 7 * nm), valid.to(rel_pos_b.dtype)

    def expert_ctrl_mask(self, uid: str) -> torch.Tensor:
        if not self._expert_curriculum_on or uid.split("_")[0] == "T":
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return self._expert_env

    def set_expert_curriculum_prob(self, prob: float) -> None:
        self.cfg.expert_curriculum_prob = float(max(0.0, prob))

    def set_expert_curriculum_enable(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled == self._expert_curriculum_on:
            return
        self._expert_curriculum_on = enabled
        print(
            f"[TAD][课程] expert_curriculum 运行期 {'启用' if enabled else '停用'}"
            f"（{'下一回合 reset 起生效；已采样的回合自然跑完' if not enabled else ''}）"
        )

    def set_exper_teams(self, teams, per_episode_prob: dict | None = None) -> None:
        new = {str(t) for t in teams if str(t) in ("T", "A", "D") and self.cfg.team_members.get(str(t))}
        if per_episode_prob is not None:
            for t in ("T", "A", "D"):
                if t not in per_episode_prob:
                    continue
                r = float(per_episode_prob[t])
                if r == self._exper_rate.get(t, 1.0):
                    continue
                self._exper_rate[t] = r
                if r >= 1.0:
                    self._roll_exper_mask[t].fill_(True)
                else:
                    self._roll_exper_mask[t].fill_(False)
            if self._test_env is not None:
                for m in self._roll_exper_mask.values():
                    m[self._test_ids] = False
        if new != self._exper_teams:
            self._exper_teams = new
            _r = {t: self._exper_rate.get(t, 1.0) for t in sorted(new)}
            print(
                f"[TAD][roll exper] 专家/脚本接管队={sorted(new) if new else '无'}"
                f"（A/D=min-dist+PPN，T=nearest+级联；test 行除外）"
                f"{f'  每队回合级概率={_r}' if per_episode_prob is not None else ''}"
            )

    def pre_step_decision_boundary(self, uid: str) -> torch.Tensor:
        return self.pre_step_decision_boundaries([uid])[uid]

    def pre_step_decision_boundaries(self, uids: list[str]) -> dict[str, torch.Tensor]:
        team_stacks = self._team_stacks()
        pool_nm: dict = {}
        out: dict = {}
        for uid in uids:
            team = uid.split("_")[0]
            if team not in pool_nm:
                pool_nm[team] = self._team_pool_stacks(team, team_stacks)
            _, _, alive = self._locked_target_state(uid, None, pool_nm[team])
            out[uid] = (self._lock_count[uid] <= 0) | (~alive)
        return out

    def commit_decisions(self, dec_actions: dict) -> dict[str, torch.Tensor]:
        if self._decisions_staged:
            raise ValueError("commit_decisions 在同一控制步被调用两次（上一次提交未被 env.step 消费？）")
        self._resolve_targets(dec_actions)
        self._decisions_staged = True

        team_stacks = self._staged["team_stacks"]
        full = self.cfg.obs_full_obs
        states_by_aid = self._get_states() if full else None
        pool_nm: dict = {}
        obs: dict[str, torch.Tensor] = {}
        for aid in self.cfg.possible_agents:
            if not aid.endswith("/ctrl"):
                continue
            uid = aid.split("/")[0]
            team = uid.split("_")[0]
            if team not in pool_nm:
                pool_nm[team] = self._team_pool_stacks(team, team_stacks)
            obs[aid] = (
                self._build_full_lower_obs(uid, states_by_aid[aid])
                if full
                else self._build_lower_obs(uid, team_stacks, pool_nm[team])
            )
        return obs

    def temporal_transition_info(self) -> dict:
        return {
            "executed_joint_action": self._executed_joint_action,
            "executed_joint_ctrl": self._executed_joint_ctrl,
            "decision_boundary": {u: self._decision_boundary[u] for u in self._all_uids},
            "effective_dec_idx": {u: self._eff_dec_idx[u] for u in self._all_uids},
            "q_low_extra": {u: self._q_low_extra[u] for u in self._q_low_extra},
            "q_low_team_ctx": self._q_low_team_ctx,
            "truncation_final": self._trunc_final,
        }

    def _selectable_pool(self, team: str, team_stacks: tuple):
        grid = self.cfg._selectable_grid[team]
        n_sel = len(self.cfg.selectable_targets[team])
        ppt = self.cfg._pool_per_team
        n_team = {"T": self.cfg.n_T, "A": self.cfg.n_A, "D": self.cfg.n_D}
        team_pos, team_vel, team_alive = team_stacks
        pos_parts, vel_parts, alive_parts, rc = [], [], [], []
        for ts, t in enumerate(grid):
            pos_parts.append(self._pad_team(team_pos[t], ppt))
            vel_parts.append(self._pad_team(team_vel[t], ppt))
            alive_parts.append(self._pad_alive(team_alive[t], ppt))
            rc.append(n_team[t] if ts < n_sel else 0)
        pos = torch.cat(pos_parts, dim=1)
        vel = torch.cat(vel_parts, dim=1)
        alive = torch.cat(alive_parts, dim=1)
        ar = torch.arange(self._n_pool, device=self.device)
        ts_idx = ar // ppt
        member_idx = ar % ppt
        rc_t = torch.tensor(rc, device=self.device)
        real_mask = member_idx < rc_t[ts_idx]
        return pos, vel, alive, real_mask

    def _team_pool_stacks(self, team: str, team_stacks: tuple | None = None):
        if team_stacks is None:
            team_stacks = self._team_stacks()
        pos, vel, alive, real_mask = self._selectable_pool(team, team_stacks)
        return pos, vel, alive & real_mask.unsqueeze(0)

    def _locked_target_state(self, uid, team_stacks: tuple | None = None, pool_nm: tuple | None = None):
        team = uid.split("_")[0]
        n_pool = self._n_pool
        if pool_nm is None:
            pool_nm = self._team_pool_stacks(team, team_stacks)
        pos_nm, vel_nm, alive_nm = pool_nm
        lp = self._locked_pool[uid].clamp(0, n_pool - 1)
        ne = torch.arange(self.num_envs, device=self.device)
        tgt_pos = pos_nm[ne, lp]
        tgt_vel = vel_nm[ne, lp]
        tgt_alive = alive_nm[ne, lp]
        return tgt_pos, tgt_vel, tgt_alive

    def _build_lower_obs(self, uid, team_stacks: tuple | None = None, pool_nm: tuple | None = None) -> torch.Tensor:
        own = self.drones[uid]
        own_pos = own.data.root_pos_w
        own_quat = own.data.root_quat_w
        origins = self.scene.env_origins
        tgt_pos, tgt_vel, tgt_alive = self._locked_target_state(uid, team_stacks, pool_nm)

        own_pos_local = own_pos - origins
        own_lin_vel_b = own.data.root_lin_vel_b
        grav_b = own.data.projected_gravity_b
        _, _, yaw = euler_xyz_from_quat(own_quat)
        q_inv = quat_inv(own_quat)
        rel_pos_w = tgt_pos - own_pos
        rel_pos_b = quat_apply(q_inv, rel_pos_w)
        tgt_vel_b = quat_apply(q_inv, tgt_vel)
        dist = torch.linalg.norm(rel_pos_w, dim=-1)

        m = (tgt_alive & self._alive[uid]).float().unsqueeze(-1)
        rel_pos_b = rel_pos_b * m
        tgt_vel_b = tgt_vel_b * m
        dist = dist * m.squeeze(-1)

        obs = torch.cat(
            [
                own_pos_local,
                own_lin_vel_b,
                grav_b,
                torch.stack([torch.sin(yaw), torch.cos(yaw)], dim=-1),
                tgt_vel_b,
                rel_pos_b,
                dist.unsqueeze(-1),
            ],
            dim=-1,
        )

        if self.cfg.obs_ang_vel and not self.cfg.obs_full_obs:
            obs = torch.cat([obs, own.data.root_ang_vel_b], dim=-1)
        ttg = self._time_frac()
        if ttg is not None:
            obs = torch.cat([obs, ttg], dim=-1)
        alive = self._alive[uid].unsqueeze(-1)
        return torch.where(alive, obs, torch.zeros_like(obs))

    def _build_full_lower_obs(self, uid, state) -> torch.Tensor:
        obs = state
        alive = self._alive[uid].unsqueeze(-1)
        return torch.where(alive, obs, torch.zeros_like(obs))

    def _get_states(self):
        ttg = self._time_frac()
        origins = self.scene.env_origins

        if getattr(self.cfg, "critic_state", "token") == "token":
            from MTAD.algorithms.models import SET_ATTN_TOKEN_DIM

            uids = self._all_uids
            M = len(uids)
            N = self.num_envs
            pos_local = torch.stack([self.drones[u].data.root_pos_w - origins for u in uids], dim=1)
            lin = torch.stack([self.drones[u].data.root_lin_vel_b for u in uids], dim=1)
            ang = torch.stack([self.drones[u].data.root_ang_vel_b for u in uids], dim=1)
            grav = torch.stack([self.drones[u].data.projected_gravity_b for u in uids], dim=1)
            yaw = torch.stack([euler_xyz_from_quat(self.drones[u].data.root_quat_w)[2] for u in uids], dim=1)
            sincos = torch.stack([torch.sin(yaw), torch.cos(yaw)], dim=-1)
            alive = torch.stack([self._alive[u] for u in uids], dim=1)
            am = alive.unsqueeze(-1).to(pos_local.dtype)
            pos_local, lin, ang, grav, sincos = (pos_local * am, lin * am, ang * am, grav * am, sincos * am)
            if _os.environ.get("TAD_ATTITUDE_DEBUG", "") == "1":
                _live = am.squeeze(-1) > 0
                assert torch.allclose(
                    (sincos[..., 0] ** 2 + sincos[..., 1] ** 2)[_live],
                    torch.ones_like(sincos[..., 0][_live]),
                    atol=1e-5,
                )
                assert torch.allclose(
                    torch.linalg.norm(grav, dim=-1)[_live], torch.ones_like(grav[..., 0][_live]), atol=1e-3
                )
            team_oh = torch.zeros(M, 3, device=self.device)
            for i, u in enumerate(uids):
                team_oh[i, "TAD".index(u.split("_")[0])] = 1.0
            t_global = ttg if ttg is not None else torch.zeros(N, 1, device=self.device)
            tok = torch.cat(
                [
                    team_oh.view(1, M, 3).expand(N, -1, -1),
                    pos_local,
                    lin,
                    ang,
                    grav,
                    sincos,
                    alive.unsqueeze(-1).to(pos_local.dtype),
                    t_global.unsqueeze(1).expand(N, M, 1),
                ],
                dim=-1,
            )
            assert tok.shape[-1] == SET_ATTN_TOKEN_DIM
            flat = tok.reshape(N, M * SET_ATTN_TOKEN_DIM)
            return {aid: flat for aid in self.cfg.possible_agents}
        uids = self._all_uids
        M = len(uids)
        pos_local = torch.stack([self.drones[u].data.root_pos_w - origins for u in uids], dim=1)
        lin_vel_b = torch.stack([self.drones[u].data.root_lin_vel_b for u in uids], dim=1)
        ang_vel_b = torch.stack([self.drones[u].data.root_ang_vel_b for u in uids], dim=1)
        grav_b = torch.stack([self.drones[u].data.projected_gravity_b for u in uids], dim=1)
        yaw_b = torch.stack([euler_xyz_from_quat(self.drones[u].data.root_quat_w)[2] for u in uids], dim=1)
        N = pos_local.shape[0]
        alive_all = torch.stack([self._alive[u] for u in uids], dim=1)
        am1 = alive_all.unsqueeze(-1).to(lin_vel_b.dtype)
        lin_vel_b = lin_vel_b * am1
        ang_vel_b = ang_vel_b * am1
        grav_b = grav_b * am1
        sin_b = torch.sin(yaw_b) * am1.squeeze(-1)
        cos_b = torch.cos(yaw_b) * am1.squeeze(-1)
        lin_flat = lin_vel_b.reshape(N, 3 * M)
        ang_flat = ang_vel_b.reshape(N, 3 * M)
        grav_flat = grav_b.reshape(N, 3 * M)
        sin_flat = sin_b.reshape(N, M)
        cos_flat = cos_b.reshape(N, M)
        anchor_height = pos_local[:, :, 2]
        idx_of = {u: i for i, u in enumerate(uids)}
        states_per_drone = {}
        for uid in self._all_uids:
            i = idx_of[uid]
            rel = pos_local - pos_local[:, i : i + 1, :]
            idx_others = [j for j in range(M) if j != i]
            rel_others = rel[:, idx_others, :]
            alive_i = alive_all[:, i : i + 1].to(pos_local.dtype)
            alive_others = alive_all[:, idx_others].to(rel_others.dtype).unsqueeze(-1)
            rel_others = (rel_others * alive_others * alive_i.unsqueeze(-1)).reshape(N, 3 * (M - 1))
            xy_anchor = pos_local[:, i, :2] * alive_i
            h_anchor = anchor_height[:, i : i + 1] * alive_i
            states_per_drone[uid] = torch.cat(
                [rel_others, lin_flat, ang_flat, grav_flat, sin_flat, cos_flat, xy_anchor, h_anchor], dim=-1
            )
            if ttg is not None:
                states_per_drone[uid] = torch.cat([states_per_drone[uid], ttg], dim=-1)

        return {aid: states_per_drone[aid.split("/")[0]] for aid in self.cfg.possible_agents}

    def _engagement_geom(self, p_self, v_self, p_tgt, team_pos, team_alive, was_alive, tgt_alive, evade: bool = False):
        a_star = self._nearest_alive_pos(p_self, team_pos["A"], team_alive["A"])
        cpt = self._seg_closest_point(a_star, p_tgt, p_self)
        d1 = cpt - p_self
        d1 = d1 / (d1.norm(dim=-1, keepdim=True) + 1e-8)
        d2 = a_star - p_self
        d2 = d2 / (d2.norm(dim=-1, keepdim=True) + 1e-8)
        s2 = -1.0 if evade else 1.0
        geom = (v_self * (d1 + s2 * d2)).sum(dim=-1)
        return geom * was_alive * tgt_alive

    def _pbrs_g(self, d: torch.Tensor) -> torch.Tensor:
        dref = float(self.cfg.pbrs_log_dref)
        if dref <= 0.0:
            return d
        return dref * torch.log1p(d / dref)

    def _get_rewards(self) -> dict:
        """Compute team rewards from captures and flight behavior."""
        oc = self._outcome
        c_own_oc, c_threat_oc = oc["c_own"], oc["c_threat"]
        c_close_oc = oc["c_close"]
        d_close_oc = oc.get("d_close", {})
        terminated = oc["terminated"]
        attacker_win = oc["attacker_win"]
        defender_win = oc["defender_win"]
        was_alive = oc["was_alive"]
        still_alive = oc["still_alive"]
        team_alive_count = oc["team_alive_count"]
        self.extras["team_alive_count"] = team_alive_count
        self.extras["agent_was_alive"] = was_alive
        self.extras["agent_still_alive"] = still_alive
        dt = self.step_dt

        r_drone = {}
        comp = {}
        term_drone = {}

        oob_mask = {}
        for uid in self._all_uids:
            team = uid.split("_")[0]
            was = was_alive[uid].float()
            effort = self.cfg.rew_effort * torch.sum(self.actions[uid] ** 2, dim=-1)
            if self.cfg.closing_target == "selected":
                lp = self._locked_pool[uid].clamp(0, self._n_pool - 1)
                ltid = self._pool_slot_team_id[team][lp]
                if team == "A":
                    dense = torch.where(
                        ltid == self._team_id["T"],
                        -self.cfg.rew_sel_A_lockT * c_own_oc[uid],
                        self.cfg.rew_sel_A_lockD * c_own_oc[uid],
                    )

                    dense = dense + self.cfg.rew_sel_A_threat * torch.clamp(c_threat_oc[uid], min=0.0)
                elif team == "D":
                    _mi = int(uid.split("_")[1])
                    geom = self._engagement_geom(
                        oc["team_pos"][team][:, _mi, :],
                        oc["team_vel"][team][:, _mi, :],
                        oc["locked_pos"][uid],
                        oc["team_pos"],
                        oc["team_alive"],
                        was_alive[uid],
                        oc["locked_alive"][uid],
                    )
                    dense = torch.where(
                        ltid == self._team_id["A"],
                        self.cfg.rew_sel_D_lockA * c_own_oc[uid],
                        self.cfg.rew_sel_D_lockT * geom,
                    )
                else:
                    _mi = int(uid.split("_")[1])
                    geom = self._engagement_geom(
                        oc["team_pos"][team][:, _mi, :],
                        oc["team_vel"][team][:, _mi, :],
                        oc["locked_pos"][uid],
                        oc["team_pos"],
                        oc["team_alive"],
                        was_alive[uid],
                        oc["locked_alive"][uid],
                        evade=True,
                    )
                    dense = torch.where(
                        ltid == self._team_id["A"],
                        self.cfg.rew_sel_T_lockA * c_own_oc[uid],
                        -self.cfg.rew_sel_T_lockD * geom,
                    )
                if team == "T":
                    dense = dense + self.cfg.rew_T_survive
            elif self.cfg.closing_target == "distance":
                if team == "T":
                    dense = self.cfg.rew_dist * d_close_oc[uid] + self.cfg.rew_T_survive
                elif team == "A":
                    dense = -self.cfg.rew_dist * d_close_oc[uid]
                else:
                    dense = -self.cfg.rew_dist * d_close_oc[uid]
            elif self.cfg.closing_target == "pbrs":
                dense = torch.zeros(self.num_envs, device=self.device)
                if team == "T":
                    dense = dense + self.cfg.rew_T_survive
            else:
                if team == "T":
                    dense = self.cfg.rew_T_evade * (-c_close_oc[uid]) + self.cfg.rew_T_survive
                elif team == "A":
                    dense = self.cfg.rew_A_close * c_close_oc[uid] + self.cfg.rew_A_threat * torch.clamp(
                        c_threat_oc[uid], min=0.0
                    )
                else:
                    dense = self.cfg.rew_D_close * c_close_oc[uid]

            if self.cfg.closing_target in ("distance", "pbrs"):
                los_vel = torch.zeros(self.num_envs, device=self.device)
            else:
                los_vel = self.cfg.rew_los_vel * oc["c_los"][uid]

            p = self.drones[uid].data.root_pos_w
            origins_xy = self.scene.env_origins[:, :2]
            horiz = torch.sqrt(((p[:, :2] - origins_xy) ** 2).sum(dim=-1))
            over_horiz = torch.clamp(
                horiz - (self.cfg.arena_radius - self.cfg.oob_horiz_margin), min=0.0, max=self.cfg.oob_horiz_cap
            )

            over_z = torch.clamp(
                (self.cfg.arena_z_min + self.cfg.oob_z_margin) - p[:, 2], min=0.0, max=self.cfg.oob_z_cap
            ) + torch.clamp(p[:, 2] - (self.cfg.arena_z_max - self.cfg.oob_z_margin), min=0.0, max=self.cfg.oob_z_cap)

            if team in self._oob_teams:
                oob_mask[uid] = ((over_horiz > 0) | (over_z > 0)) & still_alive[uid]
                oob_pen = self.cfg.rew_oob * (over_horiz**2 + over_z**2)
                oob_pen = oob_pen * still_alive[uid].float()
            else:
                oob_mask[uid] = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                oob_pen = torch.zeros(self.num_envs, device=self.device)

            dense_gate = (~oob_mask[uid]).to(dense.dtype) if self.cfg.oob_mask_dense else torch.ones_like(dense)

            term = torch.zeros(self.num_envs, device=self.device)
            if team == "T":
                term += oc["f_T_captured"][uid].float() * self.cfg.terminal_T_captured
            elif team == "A":
                cap_gate = torch.ones(self.num_envs, device=self.device)
                if self.cfg.capture_gate_enable:
                    v_close = c_close_oc[uid].clamp(min=0.0)
                    cap_gate = torch.exp(
                        -self.cfg.capture_gate_gain * (v_close - self.cfg.capture_gate_v_safe).clamp(min=0.0)
                    )
                if self.cfg.capture_time_bonus != 0.0:
                    _remain = (
                        (self.episode_length_limit_buf - self.episode_length_buf.to(torch.float32))
                        / self.episode_length_limit_buf.clamp(min=1.0)
                    ).clamp(min=0.0)
                    cap_gate = cap_gate * (1.0 + self.cfg.capture_time_bonus * _remain)
                term += oc["f_A_capture"][uid].float() * self.cfg.terminal_A_capture * cap_gate
                term += oc["f_A_intercepted"][uid].float() * self.cfg.terminal_A_captured_by_D
            else:
                cap_gate = torch.ones(self.num_envs, device=self.device)
                if self.cfg.capture_gate_enable:
                    v_close = c_close_oc[uid].clamp(min=0.0)
                    cap_gate = torch.exp(
                        -self.cfg.capture_gate_gain * (v_close - self.cfg.capture_gate_v_safe).clamp(min=0.0)
                    )
                term += oc["f_D_intercept"][uid].float() * self.cfg.terminal_D_intercept * cap_gate
            survive = still_alive[uid]

            if team == "A":
                term += attacker_win.float() * self.cfg.terminal_A_win
            elif team == "D":
                term += attacker_win.float() * self.cfg.terminal_D_fail
                term += defender_win.float() * self.cfg.terminal_D_win

                if self.cfg.terminal_D_timeout != 0.0:
                    _a_members = self.cfg.team_members.get("A") or []
                    _a_frac = (
                        oc["team_alive_count"]["A"].to(term.dtype) / max(len(_a_members), 1)
                        if _a_members
                        else torch.ones_like(term)
                    )
                    term += oc["time_out"].float() * self.cfg.terminal_D_timeout * _a_frac
            else:
                _t_surv_flag = (defender_win | oc["time_out"]).float()
                if self.cfg.terminal_T_survive_inbounds:
                    _t_surv_flag = _t_surv_flag * (~oob_mask[uid]).to(_t_surv_flag.dtype)
                term += _t_surv_flag * self.cfg.terminal_T_survive

            if team == "A" and self.cfg.terminal_A_timeout != 0.0:
                _t_members = self.cfg.team_members.get("T") or []
                _t_frac = (
                    oc["team_alive_count"]["T"].to(term.dtype) / max(len(_t_members), 1)
                    if _t_members
                    else torch.ones_like(term)
                )
                term += (oc["time_out"] & survive).float() * self.cfg.terminal_A_timeout * _t_frac

            pbrs_term = torch.zeros(self.num_envs, device=self.device)
            if self.cfg.closing_target == "pbrs":
                kappa = {"T": self.cfg.rew_pbrs_T, "A": self.cfg.rew_pbrs_A, "D": self.cfg.rew_pbrs_D}[team]
                prev = self._pbrs_prev[uid]
                d_cur = d_close_oc[uid]
                m = float(self.cfg.pbrs_clamp_m)
                if m > 0.0:
                    d_used = torch.clamp(d_cur, min=prev - m, max=prev + m)
                else:
                    d_used = d_cur
                delta = self._pbrs_g(prev) - self._pbrs_g(d_used)
                delta = torch.where(torch.isnan(prev), torch.zeros_like(delta), delta)
                pbrs_term = kappa * (-delta if team == "T" else delta) * was
                self._pbrs_prev[uid] = torch.where(was > 0, d_cur, prev)

            step_term = (self.cfg.rew_step_cost if team == "T" else -self.cfg.rew_step_cost) * was

            dense_g = _gate_oob_pos(dense, dense_gate)
            los_g = _gate_oob_pos(los_vel, dense_gate)
            pbrs_g = _gate_oob_pos(pbrs_term, dense_gate)
            step_g = _gate_oob_pos(step_term, dense_gate)

            r_dense = (dense_g + los_g) * dt * was + effort * was + oob_pen
            r_drone[uid] = r_dense + pbrs_g + step_g + term

            term_drone[uid] = term
            comp[uid] = {
                "dense": dense_g * dt * was,
                "los_vel": los_g * dt * was,
                "effort": effort * was,
                "oob": oob_pen,
                "term": term,
                "pbrs": pbrs_g,
                "step": step_g,
            }

        for team, key in (("T", "reward_T"), ("A", "reward_A"), ("D", "reward_D")):
            if not self.cfg.team_members[team]:
                continue
            members = self.cfg.team_members[team]
            team_reward = torch.zeros(self.num_envs, device=self.device)
            team_term = torch.zeros(self.num_envs, device=self.device)
            for uid in members:
                team_reward = team_reward + r_drone[uid]
                team_term = team_term + term_drone[uid]
            if self.cfg.team_reward_aggr == "mean":
                team_reward = (team_reward - team_term) / team_alive_count[team].clamp(min=1.0) + team_term / float(
                    max(len(members), 1)
                )
            self._episode_sums[key] += team_reward
        self._episode_sums["attacker_win"] += attacker_win.float() * terminated.float()
        self._episode_sums["defender_win"] += defender_win.float() * terminated.float()
        self._episode_sums["timeout"] += oc["time_out"].float()
        self._episode_sums["episode_length"] += 1.0

        first_uid = self._all_uids[0]
        for team, members in self.cfg.team_members.items():
            if not members:
                continue
            denom = team_alive_count[team].clamp(min=1.0) if self.cfg.team_reward_aggr == "mean" else None
            n_nominal = float(max(len(members), 1))
            for part in comp[first_uid]:
                agg = comp[members[0]][part].clone()
                for m in members[1:]:
                    agg = agg + comp[m][part]
                if denom is not None:
                    agg = agg / (n_nominal if part == "term" else denom)

                self.extras["log"][f"Reward / {part} (team_{team})"] = self._train_rows_mean(agg)

        self._telemetry_step += 1
        if self._telemetry_step % self._telemetry_interval == 0:
            try:
                for team, metric_suffix in (("A", "A_T"), ("D", "D_A")):
                    members = self.cfg.team_members.get(team, [])
                    if not members:
                        continue
                    if self.cfg.closing_target in ("distance", "pbrs"):
                        cnt = self._ep_dist_cnt[metric_suffix]
                        key = f"Control / dist_{metric_suffix}"
                        if float(cnt) > 0:
                            self.extras["log"][key] = self._ep_dist_sum[metric_suffix] / cnt
                        else:
                            self.extras["log"].pop(key, None)
                        self._ep_dist_sum[metric_suffix] = torch.zeros((), device=self.device)
                        self._ep_dist_cnt[metric_suffix] = torch.zeros((), device=self.device)
                        continue
                    alive = torch.stack([was_alive[u] for u in members], dim=1)
                    metric = torch.stack([c_close_oc[u] for u in members], dim=1)
                    denom = alive.sum(dim=1).clamp(min=1).to(metric.dtype)

                    self.extras["log"][f"Control / closing_rate_{metric_suffix}"] = self._train_rows_mean(
                        (metric * alive).sum(dim=1) / denom
                    )
                alive_all = torch.stack([still_alive[u] for u in self._all_uids], dim=1)
                oob_all = torch.stack([oob_mask[u] for u in self._all_uids], dim=1)

                if self._test_env is not None:
                    _keep = ~self._test_env
                    alive_all = alive_all[_keep]
                    oob_all = oob_all[_keep]
                alive_count = alive_all.sum().clamp(min=1).to(torch.float32)
                self.extras["log"]["Control / oob_rate"] = (oob_all & alive_all).sum().to(torch.float32) / alive_count
                actions_all = torch.stack([self.actions[u] for u in self._all_uids], dim=1)
                if self._test_env is not None:
                    actions_all = actions_all[~self._test_env]
                sat = (actions_all.abs() > 0.95) & alive_all.unsqueeze(-1)
                self.extras["log"]["Control / action_saturation"] = sat.sum().to(torch.float32) / (
                    alive_count * actions_all.shape[-1]
                )
            except (KeyError, IndexError, RuntimeError, ValueError) as _e:
                cnt = getattr(self, "_telemetry_fail_count", 0) + 1
                self._telemetry_fail_count = cnt
                if cnt == 1:
                    print(f"[TAD][WARN] Control/* telemetry 计算失败（后续静默计数）：{type(_e).__name__}: {_e}")
                self.extras["log"]["Control / telemetry_fail_count"] = torch.tensor(float(cnt), device=self.device)

        rew = {aid: torch.zeros(self.num_envs, device=self.device) for aid in self.cfg.possible_agents}
        rew_term = {aid: torch.zeros(self.num_envs, device=self.device) for aid in self.cfg.possible_agents}
        for uid in self._all_uids:
            if self._flat:
                if uid in self._learning_set:
                    rew[uid] = r_drone[uid]
                    rew_term[uid] = term_drone[uid]
                continue
            dec_aid = f"{uid}/dec"
            ctrl_aid = f"{uid}/ctrl"
            if dec_aid in self._learning_set:
                rew[dec_aid] = r_drone[uid]
                rew_term[dec_aid] = term_drone[uid]
            elif ctrl_aid in self._learning_set:
                rew[ctrl_aid] = r_drone[uid]
                rew_term[ctrl_aid] = term_drone[uid]

        self.extras["agent_term_reward"] = rew_term
        return rew

    def _get_dones(self):
        """Identify completed and time-limited environments."""
        self.extras["log"] = dict()
        self._update_capture_curriculum()
        oc = self._compute_outcome()
        self._outcome = oc
        terminated = oc["terminated"]
        time_out = oc["time_out"]

        self.extras["frozen_team_env"] = {t: self._frozen_env[t].clone() for t in self._frozen_env}

        if self._expert_curriculum_on and self.common_step_counter % self._telemetry_interval == 0:
            self.extras["log"]["Curriculum/expert_env_frac"] = self._train_rows_mean(self._expert_env.float())

        if bool(time_out.any()):
            self._trunc_final = (time_out.clone(), self._get_observations(), self._get_states())
        else:
            self._trunc_final = None

        forced = self._test_phase_forced_reset()
        self._test_discard = forced
        if forced is not None:
            time_out = time_out | forced
        terminated_dict = {aid: terminated for aid in self.cfg.possible_agents}
        time_outs_dict = {aid: time_out for aid in self.cfg.possible_agents}
        return terminated_dict, time_outs_dict

    def _update_capture_curriculum(self):
        if not self.cfg.curriculum_enable:
            return
        self._curriculum_check_counter += 1
        if self._curriculum_check_counter % self.cfg.curriculum_check_every != 0:
            return
        r_cur = float(self.cfg.capture_radius)
        r_end = float(self.cfg.curriculum_capture_radius_end)
        self.extras["log"]["Curriculum/capture_radius"] = torch.tensor(r_cur, device=self.device)
        if r_cur <= r_end + 1e-9:
            return
        H = max(self._win_rate_horizons) if self._win_rate_horizons else 0
        roll = self._roll.get(H)
        m = len(roll["dq"]) if roll else 0
        aw = int(roll["a"]) if roll else 0
        dw = int(roll["d"]) if roll else 0
        n_dec = aw + dw
        decisive_rate = (n_dec / m) if m > 0 else 0.0

        if str(self.cfg.curriculum_promotion_metric).lower() == "decisive":
            promotion_rate = decisive_rate
            metric_desc = "decisive_rate"
        else:
            promotion_rate = (aw / m) if m > 0 else 0.0
            metric_desc = "attacker_rate"
        self.extras["log"]["Curriculum/decisive_rate"] = torch.tensor(decisive_rate, device=self.device)
        self.extras["log"]["Curriculum/promotion_rate"] = torch.tensor(promotion_rate, device=self.device)
        if m < int(self.cfg.curriculum_min_samples):
            return
        if promotion_rate >= float(self.cfg.curriculum_promotion_winrate):
            r_new = max(r_end, r_cur - float(self.cfg.curriculum_capture_radius_step))
            self.cfg.capture_radius = r_new
            self.extras["log"]["Curriculum/capture_radius"] = torch.tensor(r_new, device=self.device)
            print(
                f"[TAD][curriculum] {metric_desc}={promotion_rate:.3f} (a={aw},d={dw}/{m}) >= "
                f"{self.cfg.curriculum_promotion_winrate} → capture_radius {r_cur:.2f} → {r_new:.2f}"
            )

    def get_roll_decisive_stats(self, horizon: int | None = None) -> dict:
        H = horizon or (max(self._win_rate_horizons) if self._win_rate_horizons else 0)
        r = self._roll.get(H)
        aw = dw = 0
        if r is not None and len(r["dq"]) > 0:
            aw, dw = r["a"], r["d"]
        n_dec = aw + dw
        if n_dec == 0:
            return {
                "n_decisive": 0,
                "a_win_rate_decisive": None,
                "d_win_rate_decisive": None,
                "t_win_rate_decisive": None,
                "team_win_rate": {team: None for team in self.cfg.team_members},
            }
        rate_by = {"A": aw / n_dec, "D": dw / n_dec, "T": dw / n_dec}
        return {
            "n_decisive": n_dec,
            "a_win_rate_decisive": rate_by["A"],
            "d_win_rate_decisive": rate_by["D"],
            "t_win_rate_decisive": rate_by["T"],
            "team_win_rate": {team: rate_by.get(team) for team in self.cfg.team_members},
        }

    def _emit_rolling_windows(self):
        for H in self._win_rate_horizons:
            r = self._roll.get(H)
            if r is None or len(r["dq"]) == 0:
                continue
            m = len(r["dq"])
            aw = r["a"] / m
            dw = r["d"] / m
            to = r["t"] / m
            draw = max(1.0 - aw - dw, 0.0)
            self.extras["log"][f"Episode/roll_attacker_win_{H}"] = torch.tensor(aw, device=self.device)
            self.extras["log"][f"Episode/roll_defender_win_{H}"] = torch.tensor(dw, device=self.device)

            self.extras["log"][f"Episode/roll_target_win_{H}"] = torch.tensor(dw + to, device=self.device)
            self.extras["log"][f"Episode/roll_timeout_{H}"] = torch.tensor(to, device=self.device)
            self.extras["log"][f"Episode/roll_draw_{H}"] = torch.tensor(draw, device=self.device)

    def test_env_mask(self) -> torch.Tensor | None:
        return self._test_env

    def _train_rows_mean(self, x: torch.Tensor) -> torch.Tensor:
        if self._test_env is None:
            return x.mean()
        return x[~self._test_env].mean()

    def _accumulate_ep_dist(self, env_ids) -> None:
        if self.cfg.closing_target not in ("distance", "pbrs"):
            return
        oc = getattr(self, "_outcome", None)
        if not oc:
            return
        pos_all, was_alive = oc.get("team_pos") or {}, oc.get("was_alive") or {}
        if not pos_all or not was_alive:
            return
        alive_by_team = {}
        for team, members in self.cfg.team_members.items():
            if members and team in pos_all:
                alive_by_team[team] = torch.stack([was_alive[u][env_ids] for u in members], dim=1)
        for suffix, (chaser, target) in (("A_T", ("A", "T")), ("D_A", ("D", "A"))):
            if chaser not in pos_all or target not in pos_all:
                continue
            pc, pt = pos_all[chaser][env_ids], pos_all[target][env_ids]
            if pc.shape[1] == 0 or pt.shape[1] == 0:
                continue
            d = torch.cdist(pc, pt)
            both = alive_by_team[chaser][:, :, None] & alive_by_team[target][:, None, :]
            best = d.masked_fill(~both, float("inf")).amin(dim=(1, 2))
            ok = torch.isfinite(best)
            self._ep_dist_sum[suffix] = self._ep_dist_sum[suffix] + best[ok].sum()
            self._ep_dist_cnt[suffix] = self._ep_dist_cnt[suffix] + ok.sum()

    @staticmethod
    def _new_test_acc() -> dict:
        return {
            "episodes": 0,
            "attacker_win": 0,
            "defender_win": 0,
            "timeout": 0,
            "ep_len": 0.0,
            "reward_T": 0.0,
            "reward_A": 0.0,
            "reward_D": 0.0,
        }

    def _test_phase_forced_reset(self) -> torch.Tensor | None:
        if self._test_env is None:
            return None
        if self.common_step_counter % self._test_every != 0:
            return None

        if self._test_phase_active and not self._test_locked and self._test_acc["episodes"] > 0:
            self._emit_test_metrics()
        self._test_acc = self._new_test_acc()
        if self._test_hold is not None:
            self._test_acc_by_cat = {t: self._new_test_acc() for t in self._test_cats}
        self._test_locked = False
        self._test_phase_active = True
        return self._test_env.clone()

    def _accumulate_test_stats(self, test_ids: torch.Tensor) -> None:
        if self._test_locked:
            return
        sums = self._episode_sums

        def _acc_into(acc: dict, ids: torch.Tensor) -> None:
            acc["episodes"] += int(len(ids))
            acc["attacker_win"] += int(sums["attacker_win"][ids].sum().item())
            acc["defender_win"] += int(sums["defender_win"][ids].sum().item())
            acc["timeout"] += int(sums["timeout"][ids].sum().item())
            acc["ep_len"] += float(sums["episode_length"][ids].sum().item())
            for team, key in (("T", "reward_T"), ("A", "reward_A"), ("D", "reward_D")):
                if self.cfg.team_members.get(team):
                    acc[key] += float(sums[key][ids].sum().item())

        _acc_into(self._test_acc, test_ids)
        if self._test_hold is not None and self._test_row_cat is not None:
            row_cat = self._test_row_cat[test_ids]
            for c, team in enumerate(self._test_cats):
                sel = test_ids[row_cat == c]
                if len(sel) > 0:
                    _acc_into(self._test_acc_by_cat[team], sel)
        if self._test_acc["episodes"] >= self._test_target_eps:
            self._emit_test_metrics()

    def _emit_test_metrics(self) -> None:
        def _emit(acc: dict, prefix: str) -> None:
            n = max(int(acc["episodes"]), 1)
            aw, dw, to = acc["attacker_win"], acc["defender_win"], acc["timeout"]
            log = self.extras.setdefault("log", {})
            log[prefix + "attacker_win_rate"] = torch.tensor(aw / n, device=self.device)
            log[prefix + "defender_win_rate"] = torch.tensor(dw / n, device=self.device)
            log[prefix + "timeout_rate"] = torch.tensor(to / n, device=self.device)
            log[prefix + "decisive_rate"] = torch.tensor((aw + dw) / n, device=self.device)
            log[prefix + "mean_episode_length_s"] = torch.tensor(acc["ep_len"] / n * self.step_dt, device=self.device)
            for team in ("T", "A", "D"):
                if self.cfg.team_members.get(team):
                    log[prefix + f"mean_reward_{team}"] = torch.tensor(acc[f"reward_{team}"] / n, device=self.device)
            log[prefix + "episodes"] = torch.tensor(float(acc["episodes"]), device=self.device)

        _emit(self._test_acc, "Test/")
        if self._test_hold is not None:
            for team in self._test_cats:
                _emit(self._test_acc_by_cat[team], f"Test/{team}/")
        self._test_locked = True

    def _reset_idx(self, env_ids):
        """Reset the selected environments and their team state."""
        self.extras.setdefault("log", {})
        for k in list(self.extras["log"].keys()):
            if k.startswith("Episode/"):
                del self.extras["log"][k]

        n_done = len(env_ids)

        if self._test_env is not None:
            env_t = env_ids if torch.is_tensor(env_ids) else torch.as_tensor(env_ids, device=self.device)
            train_ids = env_t[~self._test_env[env_t]]
            test_ids = env_t[self._test_env[env_t]]
        else:
            train_ids = env_ids
            test_ids = None

        if self.common_step_counter > 0 and len(train_ids) > 0:
            self._accumulate_ep_dist(train_ids)

        a_win = self._episode_sums["attacker_win"][train_ids]
        d_win = self._episode_sums["defender_win"][train_ids]
        to = self._episode_sums["timeout"][train_ids]
        if len(train_ids) > 0:
            for key in self._episode_sums.keys():
                val = torch.mean(self._episode_sums[key][train_ids])
                if key == "episode_length":
                    val = val * self.step_dt
                self.extras["log"]["Episode/" + key] = val

        n_train_done = len(train_ids)
        if self.common_step_counter > 0 and n_train_done > 0:
            self._cum_episodes += n_train_done
            self._cum_attacker_wins += int(a_win.sum().item())
            self._cum_defender_wins += int(d_win.sum().item())
            self._cum_timeouts += int(to.sum().item())
            if self._cum_episodes > 0:
                self.extras["log"]["Episode/cum_attacker_win"] = torch.tensor(
                    self._cum_attacker_wins / self._cum_episodes, device=self.device
                )
                self.extras["log"]["Episode/cum_defender_win"] = torch.tensor(
                    self._cum_defender_wins / self._cum_episodes, device=self.device
                )
                self.extras["log"]["Episode/cum_timeout"] = torch.tensor(
                    self._cum_timeouts / self._cum_episodes, device=self.device
                )

        if self.common_step_counter > 0 and n_train_done > 0:
            a_win_i = (a_win > 0.5).long().cpu().tolist()
            d_win_i = (d_win > 0.5).long().cpu().tolist()
            to_i = (to > 0.5).long().cpu().tolist()
            for i in range(n_train_done):
                rec = (a_win_i[i], d_win_i[i], to_i[i])
                for H, r in self._roll.items():
                    dq = r["dq"]
                    dq.append(rec)
                    r["a"] += rec[0]
                    r["d"] += rec[1]
                    r["t"] += rec[2]
                    if len(dq) > H:
                        oa, od, ot = dq.popleft()
                        r["a"] -= oa
                        r["d"] -= od
                        r["t"] -= ot
            self._emit_rolling_windows()

        if test_ids is not None and len(test_ids) > 0 and self._test_phase_active and self._test_discard is None:
            self._accumulate_test_stats(test_ids)

        for key in self._episode_sums.keys():
            self._episode_sums[key][env_ids] = 0.0

        if self.cfg.frozen_opponent_prob > 0.0:
            p_fz = float(self.cfg.frozen_opponent_prob)
            for team, fz in self._frozen_env.items():
                fz[env_ids] = torch.rand(len(env_ids), device=self.device) < p_fz
        else:
            for fz in self._frozen_env.values():
                fz[env_ids] = False

        for team, m in self._roll_exper_mask.items():
            r = float(self._exper_rate.get(team, 1.0))
            if r >= 1.0:
                m[env_ids] = True
            elif r <= 0.0:
                m[env_ids] = False
            else:
                m[env_ids] = torch.rand(len(env_ids), device=self.device) < r

        if self._expert_curriculum_on:
            self._expert_env[env_ids] = torch.rand(len(env_ids), device=self.device) < float(
                self.cfg.expert_curriculum_prob
            )
        else:
            self._expert_env[env_ids] = False
        for fz in self._frozen_env.values():
            fz[env_ids] &= ~self._expert_env[env_ids]

        if self._test_env is not None:
            for fz in self._frozen_env.values():
                fz[self._test_ids] = False
            self._expert_env[self._test_ids] = False

        for uid in self._all_uids:
            self._pbrs_prev[uid][env_ids] = float("nan")

        for uid in self._all_uids:
            self.drones[uid].reset(env_ids)
            self._alive[uid][env_ids] = True
            self._locked_pool[uid][env_ids] = 0
            self._lock_count[uid][env_ids] = 0
            self._decision_boundary[uid][env_ids] = False
            self._eff_pool_idx[uid][env_ids] = 0
            self._eff_dec_idx[uid][env_ids] = 0
            self._eff_onehot_dec[uid][env_ids] = 0.0
            if self._n_dec > 0:
                self._eff_onehot_dec[uid][env_ids, 0] = 1.0
        super()._reset_idx(env_ids)

        self.episode_length_limit_buf[env_ids] = sample_episode_horizon_steps(
            len(env_ids), self._episode_horizon_lo, self._episode_horizon_hi, self.device
        )

        env_origins = self.scene.env_origins[env_ids]
        k = len(env_ids)
        centers = {
            "T": torch.tensor(self.cfg.t_start_pos, device=self.device),
            "A": torch.tensor(self.cfg.a_start_pos, device=self.device),
            "D": torch.tensor(self.cfg.d_start_pos, device=self.device),
        }
        R = self.cfg.team_spawn_radius

        spawn_z_min = self.cfg.arena_z_min + self.cfg.oob_z_margin
        spawn_z_max = self.cfg.arena_z_max - self.cfg.oob_z_margin

        _s = float(getattr(self.cfg, "spawn_z_shrink", 0.0))
        if _s > 0.0:
            _s = min(_s, 0.95)
            _c = 0.5 * (spawn_z_min + spawn_z_max)
            _half = 0.5 * (spawn_z_max - spawn_z_min) * (1.0 - _s)
            spawn_z_min, spawn_z_max = _c - _half, _c + _half
        for team, members in self.cfg.team_members.items():
            for uid in members:
                pos = sample_in_arena_positions(
                    centers[team],
                    R,
                    k,
                    env_origins,
                    arena_radius=self.cfg.arena_radius,
                    z_min=spawn_z_min,
                    z_max=spawn_z_max,
                )
                root = self.drones[uid].data.default_root_state[env_ids].clone()
                root[:, :3] = pos
                if self.cfg.random_init_attitude:
                    rng = self.cfg.init_attitude_range
                    roll = (torch.rand(k, device=self.device) * 2.0 - 1.0) * rng[0]
                    pitch = (torch.rand(k, device=self.device) * 2.0 - 1.0) * rng[1]
                    yaw = (torch.rand(k, device=self.device) * 2.0 - 1.0) * rng[2]
                    root[:, 3:7] = quat_from_euler_xyz(roll, pitch, yaw)
                else:
                    root[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
                root[:, 7:] = 0.0
                self.drones[uid].write_root_pose_to_sim(root[:, :7], env_ids)
                self.drones[uid].write_root_velocity_to_sim(root[:, 7:], env_ids)
                self.drones[uid].write_joint_state_to_sim(
                    self.drones[uid].data.default_joint_pos[env_ids],
                    self.drones[uid].data.default_joint_vel[env_ids],
                    None,
                    env_ids,
                )
