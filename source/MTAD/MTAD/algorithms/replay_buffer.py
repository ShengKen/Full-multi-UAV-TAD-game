# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.

# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn.functional as F


class _PinnedBatchStager:
    def __init__(self, device: str, pinned: bool):
        self.device = device
        self.pinned = bool(pinned and torch.cuda.is_available() and str(device).startswith("cuda"))
        self._slots: dict = {}
        self._next: dict = {}
        self._events: dict = {}

    def take(self, key: str, src: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        flat_idx = indices.reshape(-1)
        out_shape = (*indices.shape, *src.shape[1:])
        if not self.pinned:
            return src.index_select(0, flat_idx).reshape(out_shape).to(self.device)
        flat_shape = (flat_idx.numel(), *src.shape[1:])
        slots = self._slots.get(key)
        if slots is None or slots[0].shape != flat_shape:
            slots = [torch.empty(flat_shape, dtype=src.dtype, device="cpu", pin_memory=True) for _ in range(2)]
            self._slots[key] = slots
            self._events[key] = [None, None]
            self._next[key] = 0
        slot = self._next[key]
        event = self._events[key][slot]
        if event is not None:
            event.synchronize()
        torch.index_select(src, 0, flat_idx, out=slots[slot])
        result = slots[slot].reshape(out_shape).to(self.device, non_blocking=True)
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(device=self.device))
        self._events[key][slot] = event
        self._next[key] = 1 - slot
        return result


def _choose_index_dtype(width: int) -> torch.dtype:
    width = int(width)
    if width <= 256:
        return torch.uint8
    if width <= 32767:
        return torch.int16
    return torch.int32


def _expand_joint_action(dec_idx: torch.Tensor, ctrl_action: torch.Tensor, dec_dim: int) -> torch.Tensor:
    dec_oh = F.one_hot(dec_idx.long(), num_classes=int(dec_dim)).to(ctrl_action.dtype)
    dense = torch.cat([dec_oh, ctrl_action], dim=-1)
    return dense.reshape(*dense.shape[:-2], dense.shape[-2] * dense.shape[-1])


def _expand_high_action(dec_idx: torch.Tensor, dec_dim: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    dense = F.one_hot(dec_idx.long(), num_classes=int(dec_dim)).to(dtype)
    return dense.reshape(*dense.shape[:-2], dense.shape[-2] * dense.shape[-1])


def _allocated_tensor_bytes(obj) -> int:
    total = 0
    for v in vars(obj).values():
        if isinstance(v, torch.Tensor):
            total += v.numel() * v.element_size()
        elif isinstance(v, dict):
            total += sum(x.numel() * x.element_size() for x in v.values() if isinstance(x, torch.Tensor))
    return total


class HierReplayCommonStore:
    def __init__(
        self,
        capacity: int,
        num_envs: int,
        total_act_dim: int,
        agent_obs_dims: dict,
        device: str,
        store_device: str | None = None,
        *,
        num_physical_agents: int | None = None,
        dec_dim: int | None = None,
        ctrl_dim: int = 4,
        compact_actions: bool | None = None,
        q_low_team_ctx_dim: int = 0,
        obs_agent_ids: list | None = None,
        next_obs_agent_ids: list | None = None,
    ):
        self.capacity = int(capacity)
        self.num_envs = int(num_envs)
        self.device = device
        self.store_device = store_device if store_device is not None else device
        self._pinned = str(self.store_device) == "cpu"
        self.agent_ids = list(agent_obs_dims)
        self._ptr = 0
        self._size = 0
        self._total_written = 0
        self._write_event = None
        self._stager = _PinnedBatchStager(device, self._pinned)
        self._total_act_dim = int(total_act_dim)
        self._num_physical_agents = int(num_physical_agents) if num_physical_agents is not None else None
        self._dec_dim = int(dec_dim) if dec_dim is not None else None
        self._ctrl_dim = int(ctrl_dim)
        expected_dim = None
        if self._num_physical_agents is not None and self._dec_dim is not None:
            expected_dim = self._num_physical_agents * (self._dec_dim + self._ctrl_dim)
        if compact_actions is True and expected_dim != self._total_act_dim:
            raise ValueError(
                f"compact joint_action geometry mismatch: total_act_dim={self._total_act_dim}, "
                f"M={self._num_physical_agents}, dec_dim={self._dec_dim}, "
                f"ctrl_dim={self._ctrl_dim}"
            )
        self._compact_actions = (
            bool(expected_dim == self._total_act_dim) if compact_actions is None else bool(compact_actions)
        )
        if self._compact_actions:
            self._block = self._dec_dim + self._ctrl_dim
        else:
            self._block = None

        def alloc(*shape, dtype=torch.float32):
            return torch.zeros(*shape, dtype=dtype, device=self.store_device, pin_memory=self._pinned)

        if self._compact_actions:
            self.joint_action = None
            idx_dtype = _choose_index_dtype(self._dec_dim)
            self.joint_dec_idx = alloc(self.capacity, self._num_physical_agents, dtype=idx_dtype)
            self.joint_ctrl_action = alloc(self.capacity, self._num_physical_agents, self._ctrl_dim)
        else:
            self.joint_action = alloc(self.capacity, total_act_dim)
            self.joint_dec_idx = None
            self.joint_ctrl_action = None
        self.terminated = alloc(self.capacity, dtype=torch.bool)

        self.boot_terminated = alloc(self.capacity, dtype=torch.bool)

        self.obs_ids = list(obs_agent_ids) if obs_agent_ids is not None else list(agent_obs_dims)
        self.next_obs_ids = list(next_obs_agent_ids) if next_obs_agent_ids is not None else list(self.obs_ids)
        self.obs_per_agent = {aid: alloc(self.capacity, agent_obs_dims[aid]) for aid in self.obs_ids}
        self.next_obs_per_agent = {aid: alloc(self.capacity, agent_obs_dims[aid]) for aid in self.next_obs_ids}
        self.active_per_agent = {aid: alloc(self.capacity, dtype=torch.bool) for aid in self.agent_ids}

        self.decision_boundary = {aid: alloc(self.capacity, dtype=torch.bool) for aid in self.agent_ids}

        self.q_low_team_ctx = alloc(self.capacity, int(q_low_team_ctx_dim)) if int(q_low_team_ctx_dim) > 0 else None

    def _scatter_copy(self, dst: torch.Tensor, src: torch.Tensor, base: int, n: int):
        non_blocking = self._pinned and src.device.type == "cuda"
        end = base + n
        if end <= self.capacity:
            dst[base:end].copy_(src, non_blocking=non_blocking)
        else:
            first = self.capacity - base
            dst[base:].copy_(src[:first], non_blocking=non_blocking)
            dst[: n - first].copy_(src[first:], non_blocking=non_blocking)

    def _record_write_event(self):
        if self._pinned and torch.cuda.is_available() and str(self.device).startswith("cuda"):
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(device=self.device))
            self._write_event = event

    def wait_writes(self):
        if self._write_event is not None:
            self._write_event.synchronize()
            self._write_event = None

    def add(
        self,
        joint_action,
        terminated,
        obs_per_agent,
        next_obs_per_agent,
        decision_boundary_per_agent=None,
        active_per_agent=None,
        boot_terminated=None,
        q_low_team_ctx=None,
    ):
        n = joint_action.shape[0]
        if n != self.num_envs:
            raise ValueError("HierReplayCommonStore expects one row per environment")
        base = self._ptr
        if self._compact_actions:
            ja = joint_action.view(n, self._num_physical_agents, self._block)
            dec_idx = ja[..., : self._dec_dim].argmax(dim=-1).to(self.joint_dec_idx.dtype)
            ctrl = ja[..., self._dec_dim : self._dec_dim + self._ctrl_dim]
            self._scatter_copy(self.joint_dec_idx, dec_idx, base, n)
            self._scatter_copy(self.joint_ctrl_action, ctrl, base, n)
        else:
            self._scatter_copy(self.joint_action, joint_action, base, n)
        self._scatter_copy(self.terminated, terminated.bool().view(-1), base, n)

        boot = terminated if boot_terminated is None else boot_terminated
        self._scatter_copy(self.boot_terminated, boot.bool().view(-1), base, n)

        for aid in self.obs_ids:
            self._scatter_copy(self.obs_per_agent[aid], obs_per_agent[aid], base, n)
        for aid in self.next_obs_ids:
            self._scatter_copy(self.next_obs_per_agent[aid], next_obs_per_agent[aid], base, n)
        for aid in self.agent_ids:
            active = None if active_per_agent is None else active_per_agent.get(aid)
            if active is None:
                active = torch.ones(n, dtype=torch.bool, device=joint_action.device)
            self._scatter_copy(self.active_per_agent[aid], active.bool().view(-1), base, n)

        for aid in self.agent_ids:
            boundary = None if decision_boundary_per_agent is None else decision_boundary_per_agent.get(aid)
            if boundary is None:
                boundary = torch.zeros(n, dtype=torch.bool, device=self.store_device)
            self._scatter_copy(self.decision_boundary[aid], boundary.bool().view(-1), base, n)

        if self.q_low_team_ctx is not None:
            if q_low_team_ctx is None:
                raise ValueError("q_low_team_ctx storage enabled but add() received None")
            self._scatter_copy(self.q_low_team_ctx, q_low_team_ctx, base, n)
        self._ptr = (base + n) % self.capacity
        self._size = min(self._size + n, self.capacity)
        self._total_written += n
        self._record_write_event()

    def random_indices(self, batch_size: int, reserve: int = 0) -> torch.Tensor:
        self.wait_writes()
        if reserve <= 0:
            return torch.randint(0, self._size, (batch_size,), device=self.store_device)
        lo = self._total_written - self._size
        hi = self._total_written - reserve * self.num_envs
        if hi <= lo:
            return torch.randint(0, self._size, (batch_size,), device=self.store_device)
        lin = torch.randint(lo, hi, (batch_size,), device=self.store_device)
        return lin % self.capacity

    def window_start_indices(self, batch_size: int, n_step: int, reserve: int = 0) -> tuple[torch.Tensor, int]:
        self.wait_writes()
        K = max(1, int(n_step))
        lo = self._total_written - self._size
        hi_r = self._total_written - (K - 1 + max(0, int(reserve))) * self.num_envs
        hi = min(hi_r, self._total_written - (K - 1) * self.num_envs)
        if hi <= lo:
            hi = self._total_written - (K - 1) * self.num_envs
            if hi <= lo:
                return self.random_indices(batch_size), 1
        lin = torch.randint(lo, hi, (batch_size,), device=self.store_device)
        return lin % self.capacity, K

    def take(self, key: str, src: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return self._stager.take(key, src, indices)

    def take_joint_action(self, key: str, indices: torch.Tensor) -> torch.Tensor:
        if not self._compact_actions:
            return self.take(key, self.joint_action, indices)
        ctrl = self.take(f"{key}:ctrl", self.joint_ctrl_action, indices)
        dec_idx = self.take(f"{key}:dec_idx", self.joint_dec_idx, indices)
        return _expand_joint_action(dec_idx, ctrl, self._dec_dim)

    def __len__(self):
        return self._size

    def allocation_bytes(self) -> int:
        return _allocated_tensor_bytes(self)


class _PrefetchFuture:
    def __init__(self, batch: dict, stream, device):
        self.batch = batch
        self.stream = stream
        self.device = device
        self._consumed = False

    def get(self) -> dict:
        if not self._consumed and self.stream is not None:
            torch.cuda.current_stream(self.device).wait_stream(self.stream)
            self._consumed = True
        return self.batch


class HierReplayBuffer:
    def __init__(
        self,
        capacity: int,
        num_envs: int,
        state_dim: int,
        total_act_dim: int,
        agent_obs_dims: dict,
        device: str,
        low_reward_agent_ids: list[str] | None = None,
        store_device: str | None = None,
        common_store: HierReplayCommonStore | None = None,
        q_low_extra_agent_ids: list[str] | None = None,
        q_low_extra_dim: int = 0,
        n_step: int = 1,
        n_step_discount: float = 0.99,
        priority_mix: float = 0.0,
        ctrl_next_obs_successor: bool = False,
        store_state: bool = True,
    ):
        self.capacity = int(capacity)
        self.num_envs = int(num_envs)
        self.device = device
        self.store_device = store_device if store_device is not None else device
        self._pinned = str(self.store_device) == "cpu"
        self.agent_ids = list(agent_obs_dims)
        self.n_step = max(1, int(n_step))
        self.n_step_discount = float(n_step_discount)

        self.priority_mix = min(max(float(priority_mix), 0.0), 1.0)

        self.ctrl_next_obs_successor = bool(ctrl_next_obs_successor)
        self.common = (
            common_store
            if common_store is not None
            else HierReplayCommonStore(
                capacity, num_envs, total_act_dim, agent_obs_dims, device, store_device=self.store_device
            )
        )
        self._stager = _PinnedBatchStager(device, self._pinned)
        self._write_event = None

        def alloc(*shape, dtype=torch.float32):
            return torch.zeros(*shape, dtype=dtype, device=self.store_device, pin_memory=self._pinned)

        self.store_state = bool(store_state)
        self.state = alloc(self.capacity, state_dim) if self.store_state else None
        self.next_state = alloc(self.capacity, state_dim) if self.store_state else None
        self.reward = alloc(self.capacity)
        self.team_active = alloc(self.capacity, dtype=torch.bool)

        self.anchor_switch = alloc(self.capacity, dtype=torch.bool)

        self.priority = alloc(self.capacity, dtype=torch.bool)
        self.low_reward = {aid: alloc(self.capacity) for aid in (low_reward_agent_ids or [])}

        self.q_low_extra_dim = int(q_low_extra_dim)
        self.q_low_extra = (
            {aid: alloc(self.capacity, self.q_low_extra_dim) for aid in (q_low_extra_agent_ids or [])}
            if self.q_low_extra_dim > 0
            else {}
        )
        self._ptr = 0
        self._size = 0

    def _scatter_copy(self, dst: torch.Tensor, src: torch.Tensor, base: int, n: int):
        non_blocking = self._pinned and src.device.type == "cuda"
        end = base + n
        if end <= self.capacity:
            dst[base:end].copy_(src, non_blocking=non_blocking)
        else:
            first = self.capacity - base
            dst[base:].copy_(src[:first], non_blocking=non_blocking)
            dst[: n - first].copy_(src[first:], non_blocking=non_blocking)

    def _record_write_event(self):
        if self._pinned and torch.cuda.is_available() and str(self.device).startswith("cuda"):
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(device=self.device))
            self._write_event = event

    def wait_writes(self):
        if self._write_event is not None:
            self._write_event.synchronize()
            self._write_event = None
        self.common.wait_writes()

    def add(
        self,
        state,
        next_state,
        reward,
        low_reward_per_agent=None,
        team_active=None,
        q_low_extra_per_agent=None,
        anchor_switch=None,
        priority=None,
    ):
        """Store a hierarchical control transition in replay."""

        n = reward.shape[0]
        if n != self.num_envs:
            raise ValueError("HierReplayBuffer expects one row per environment")
        base = self._ptr
        if self.state is not None:
            self._scatter_copy(self.state, state, base, n)
            self._scatter_copy(self.next_state, next_state, base, n)
        self._scatter_copy(self.reward, reward, base, n)
        if team_active is None:
            team_active = torch.ones(n, dtype=torch.bool, device=reward.device)
        self._scatter_copy(self.team_active, team_active.bool().view(-1), base, n)
        if anchor_switch is None:
            anchor_switch = torch.zeros(n, dtype=torch.bool, device=reward.device)
        self._scatter_copy(self.anchor_switch, anchor_switch.bool().view(-1), base, n)
        if priority is not None:
            self._scatter_copy(self.priority, priority.bool().view(-1), base, n)
        if self.low_reward:
            if low_reward_per_agent is None:
                raise ValueError("Control low_reward storage enabled but add() received None")
            for aid in self.low_reward:
                self._scatter_copy(self.low_reward[aid], low_reward_per_agent[aid].view(-1), base, n)
        if self.q_low_extra:
            if q_low_extra_per_agent is None:
                raise ValueError("Control q_low_extra storage enabled but add() received None")
            for aid in self.q_low_extra:
                self._scatter_copy(self.q_low_extra[aid], q_low_extra_per_agent[aid], base, n)
        self._ptr = (base + n) % self.capacity
        self._size = min(self._size + n, self.capacity)
        self._record_write_event()
        if self._ptr != self.common._ptr or self._size != self.common._size:
            raise RuntimeError(
                f"private/common replay cursors diverged: private=({self._ptr},{self._size}) "
                f"common=({self.common._ptr},{self.common._size})"
            )

    def _take(self, key: str, src: torch.Tensor, indices: torch.Tensor):
        return self._stager.take(key, src, indices)

    def _next_boundary_column(self, base_rows: torch.Tensor) -> dict:
        if not self.common.decision_boundary:
            return {}
        next_rows = (base_rows + self.num_envs) % self.common.capacity
        stale = ((base_rows - (self.common._ptr - self.num_envs)) % self.common.capacity) < self.num_envs
        keep = (~stale).to(self.device)
        return {
            aid: self.common.take(f"nbnd:{aid}", self.common.decision_boundary[aid], next_rows) & keep
            for aid in self.agent_ids
        }

    def _take_successor(self, key: str, src: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        nxt = (rows + self.common.num_envs) % self.common.capacity
        stale = ((rows - (self.common._ptr - self.common.num_envs)) % self.common.capacity) < self.common.num_envs
        val = self._take(key, src, nxt)
        if not bool(stale.any()):
            return val
        own = self._take(f"{key}:own", src, rows)
        keep = (~stale).to(device=val.device)
        return torch.where(keep.unsqueeze(-1) if val.dim() > 1 else keep, val, own)

    def sample(self, batch_size: int) -> dict:
        """Sample control transitions for off-policy updates."""
        self.wait_writes()
        return self._sample_impl(batch_size)

    def _sample_impl(self, batch_size: int) -> dict:
        if self.n_step > 1:
            return self._sample_nstep_impl(batch_size)
        return self._sample_1step_impl(batch_size)

    def _eligible_ring_mask(self, K: int) -> torch.Tensor:
        prio = self.priority
        if K <= 1:
            return prio
        E = self.common.num_envs
        elig = prio
        for k in range(1, K):
            elig = elig | torch.roll(prio, shifts=-k * E)
        return elig

    def _mixed_indices(self, batch_size: int, n_step: int):
        K = max(1, int(n_step))
        common = self.common

        _res = 1 if self.ctrl_next_obs_successor else 0

        def _uniform():
            if K > 1:
                return (*common.window_start_indices(batch_size, K, reserve=_res), None)
            return common.random_indices(batch_size, reserve=_res), 1, None

        if self.priority_mix <= 0.0:
            return _uniform()
        lo = common._total_written - common._size
        hi = common._total_written - (K - 1 + _res) * common.num_envs
        if hi <= lo:
            return _uniform()
        lin_all = torch.arange(lo, hi, device=self.store_device)
        elig = self._eligible_ring_mask(K)[lin_all % common.capacity]
        n_elig = int(elig.sum().item())
        if n_elig <= 0:
            return _uniform()
        B = int(batch_size)
        n_term = min(int(round(B * self.priority_mix)), B, n_elig)
        n_uni = B - n_term
        N = int(lin_all.numel())
        pos = torch.nonzero(elig, as_tuple=False).view(-1)

        term_lin = lo + pos[torch.randint(0, n_elig, (n_term,), device=self.store_device)]
        uni_lin = torch.randint(lo, hi, (n_uni,), device=self.store_device) if n_uni > 0 else term_lin[:0]
        lin = torch.cat([term_lin, uni_lin])

        p_row = torch.where(
            elig[lin - lo],
            torch.full_like(lin, n_term / (B * n_elig), dtype=torch.float32),
            torch.zeros_like(lin, dtype=torch.float32),
        ) + torch.full_like(lin, n_uni / (B * N), dtype=torch.float32)
        w = (1.0 / N) / p_row.clamp(min=1e-12)
        w = (w / w.max()).to(self.device, non_blocking=self._pinned)
        return lin % common.capacity, K, w

    def _ctrl_next_obs_override(self, nobs: dict, rows: torch.Tensor) -> dict:
        if not self.ctrl_next_obs_successor:
            return nobs
        out = dict(nobs)
        for aid in self.agent_ids:
            if aid.endswith("/ctrl"):
                out[aid] = self._take_successor(f"nobs_succ:{aid}", self.common.obs_per_agent[aid], rows)
        return out

    def _sample_1step_impl(self, batch_size: int) -> dict:
        idx, _k, is_w = self._mixed_indices(batch_size, 1)
        nbnd = self._next_boundary_column(idx)

        nobs_pa = self._ctrl_next_obs_override(
            {
                aid: self.common.take(f"nobs:{aid}", self.common.next_obs_per_agent[aid], idx)
                for aid in self.common.next_obs_per_agent
            },
            idx,
        )
        return {
            "state": (self._take("state", self.state, idx) if self.state is not None else None),
            "next_state": (self._take("next_state", self.next_state, idx) if self.next_state is not None else None),
            "joint_action": self.common.take_joint_action("joint_action", idx),
            "reward": self._take("reward", self.reward, idx),
            "terminated": self.common.take("terminated", self.common.terminated, idx).to(torch.float32),
            "boot_terminated": self.common.take("boot_terminated", self.common.boot_terminated, idx).to(torch.float32),
            "obs_per_agent": {
                aid: self.common.take(f"obs:{aid}", self.common.obs_per_agent[aid], idx)
                for aid in self.common.obs_per_agent
            },
            "next_obs_per_agent": nobs_pa,
            "active_per_agent": {
                aid: self.common.take(f"active:{aid}", self.common.active_per_agent[aid], idx) for aid in self.agent_ids
            },
            "team_active": self._take("team_active", self.team_active, idx),
            "anchor_switch": self._take("anchor_switch", self.anchor_switch, idx).bool(),
            "low_reward_per_agent": {
                aid: self._take(f"low:{aid}", self.low_reward[aid], idx) for aid in self.low_reward
            },
            "q_low_extra_per_agent": {
                aid: self._take(f"qle:{aid}", self.q_low_extra[aid], idx) for aid in self.q_low_extra
            },
            "next_q_low_extra_per_agent": {
                aid: self._take_successor(f"nqle:{aid}", self.q_low_extra[aid], idx) for aid in self.q_low_extra
            },
            "q_low_team_ctx": (
                self.common.take("qctx", self.common.q_low_team_ctx, idx)
                if self.common.q_low_team_ctx is not None
                else None
            ),
            "next_q_low_team_ctx": (
                self._take_successor("nqctx", self.common.q_low_team_ctx, idx)
                if self.common.q_low_team_ctx is not None
                else None
            ),
            "is_weight": is_w,
            "decision_boundary_next_per_agent": nbnd,
        }

    def _sample_nstep_impl(self, batch_size: int) -> dict:
        K = self.n_step
        idx, k_eff, _is_w = self._mixed_indices(batch_size, K)
        if k_eff == 1:
            return self._sample_1step_impl(batch_size)
        B = int(idx.shape[0])
        offs = torch.arange(K, device=idx.device) * self.common.num_envs
        widx = (idx.unsqueeze(0) + offs.unsqueeze(1)) % self.common.capacity
        term_w = self.common.take("ns:term", self.common.terminated, widx).bool()
        boot_w = self.common.take("ns:boot", self.common.boot_terminated, widx).bool()

        csum = term_w.cumsum(dim=0)
        prev_csum = torch.cat([torch.zeros(1, B, dtype=csum.dtype, device=csum.device), csum[:-1]], dim=0)
        first_end = csum.eq(1) & prev_csum.eq(0)
        include = csum.eq(0) | first_end
        k_i = include.sum(dim=0).long()
        gam = torch.tensor(
            [self.n_step_discount**k for k in range(K)], dtype=torch.float32, device=csum.device
        ).unsqueeze(1)
        w = include.to(torch.float32) * gam
        pick_last = (k_i - 1).unsqueeze(0)

        last_rows = torch.gather(widx, 0, pick_last.to(device=widx.device, dtype=widx.dtype)).squeeze(0)
        boot_last = torch.gather(boot_w, 0, pick_last.to(device=boot_w.device)).squeeze(0)
        anchor_w = self._take("ns:asw", self.anchor_switch, widx).bool()

        anchor_fold = (anchor_w & include).any(dim=0)

        def _fold(win: torch.Tensor) -> torch.Tensor:
            return (win.to(torch.float32) * w).sum(dim=0)

        nobs_pa = self._ctrl_next_obs_override(
            {
                aid: self.common.take(f"nobs:{aid}", self.common.next_obs_per_agent[aid], last_rows)
                for aid in self.common.next_obs_per_agent
            },
            last_rows,
        )
        return {
            "state": (self._take("state", self.state, idx) if self.state is not None else None),
            "next_state": (
                self._take("next_state", self.next_state, last_rows) if self.next_state is not None else None
            ),
            "joint_action": self.common.take_joint_action("joint_action", idx),
            "reward": _fold(self._take("ns:rew", self.reward, widx)),
            "terminated": term_w.any(dim=0).to(torch.float32),
            "boot_terminated": boot_last.to(torch.float32),
            "obs_per_agent": {
                aid: self.common.take(f"obs:{aid}", self.common.obs_per_agent[aid], idx)
                for aid in self.common.obs_per_agent
            },
            "next_obs_per_agent": nobs_pa,
            "active_per_agent": {
                aid: self.common.take(f"active:{aid}", self.common.active_per_agent[aid], idx) for aid in self.agent_ids
            },
            "team_active": self._take("team_active", self.team_active, idx),
            "anchor_switch": anchor_fold,
            "low_reward_per_agent": {
                aid: _fold(self._take(f"ns:low:{aid}", self.low_reward[aid], widx)) for aid in self.low_reward
            },
            "q_low_extra_per_agent": {
                aid: self._take(f"qle:{aid}", self.q_low_extra[aid], idx) for aid in self.q_low_extra
            },
            "next_q_low_extra_per_agent": {
                aid: self._take_successor(f"nqle:{aid}", self.q_low_extra[aid], last_rows) for aid in self.q_low_extra
            },
            "q_low_team_ctx": (
                self.common.take("qctx", self.common.q_low_team_ctx, idx)
                if self.common.q_low_team_ctx is not None
                else None
            ),
            "next_q_low_team_ctx": (
                self._take_successor("nqctx", self.common.q_low_team_ctx, last_rows)
                if self.common.q_low_team_ctx is not None
                else None
            ),
            "nstep_gamma": torch.pow(
                torch.tensor(self.n_step_discount, dtype=torch.float32, device=csum.device), k_i.to(torch.float32)
            ),
            "is_weight": _is_w,
            "decision_boundary_next_per_agent": self._next_boundary_column(last_rows),
        }

    def sample_prefetch(self, batch_size: int, stream) -> "_PrefetchFuture":
        self.wait_writes()
        with torch.cuda.stream(stream):
            batch = self._sample_impl(batch_size)
        return _PrefetchFuture(batch, stream, self.device)

    def __len__(self):
        return self._size

    def allocation_bytes(self) -> int:
        return _allocated_tensor_bytes(self)


class DecisionReplayBuffer:
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        high_act_dim: int,
        dec_obs_dims: dict,
        device: str,
        store_device: str | None = None,
        *,
        num_physical_agents: int | None = None,
        dec_dim: int | None = None,
        compact_actions: bool | None = None,
    ):
        self.capacity = int(capacity)
        self.device = device
        self.store_device = store_device if store_device is not None else device
        self._pinned = str(self.store_device) == "cpu"
        self.agent_ids = list(dec_obs_dims)
        self._ptr = 0
        self._size = 0
        self._write_event = None
        self._stager = _PinnedBatchStager(device, self._pinned)
        self._high_act_dim = int(high_act_dim)
        self._num_physical_agents = int(num_physical_agents) if num_physical_agents is not None else None
        self._dec_dim = int(dec_dim) if dec_dim is not None else None
        expected_dim = None
        if self._num_physical_agents is not None and self._dec_dim is not None:
            expected_dim = self._num_physical_agents * self._dec_dim
        if compact_actions is True and expected_dim != self._high_act_dim:
            raise ValueError(
                f"compact high_action geometry mismatch: high_act_dim={self._high_act_dim}, "
                f"M={self._num_physical_agents}, dec_dim={self._dec_dim}, "
                "target-only actions"
            )
        self._compact_actions = (
            bool(expected_dim == self._high_act_dim) if compact_actions is None else bool(compact_actions)
        )

        def alloc(*shape, dtype=torch.float32):
            return torch.zeros(*shape, dtype=dtype, device=self.store_device, pin_memory=self._pinned)

        self.state = alloc(self.capacity, state_dim)
        self.next_state = alloc(self.capacity, state_dim)
        if self._compact_actions:
            self.high_joint_action = None
            self.next_high_joint_action = None
            idx_dtype = _choose_index_dtype(self._dec_dim)
            self.high_dec_idx = alloc(self.capacity, self._num_physical_agents, dtype=idx_dtype)
            self.next_high_dec_idx = alloc(self.capacity, self._num_physical_agents, dtype=idx_dtype)
        else:
            self.high_joint_action = alloc(self.capacity, high_act_dim)
            self.next_high_joint_action = alloc(self.capacity, high_act_dim)
            self.high_dec_idx = None
            self.next_high_dec_idx = None
        self.reward = alloc(self.capacity)
        self.discount = alloc(self.capacity)
        self.terminated = alloc(self.capacity, dtype=torch.bool)
        self.length = alloc(self.capacity, dtype=torch.long)
        self.agent_mask = alloc(self.capacity, len(self.agent_ids), dtype=torch.bool)

        self.next_alive = {
            aid: torch.ones(self.capacity, dtype=torch.bool, device=self.store_device, pin_memory=self._pinned)
            for aid in self.agent_ids
        }
        self.obs_per_agent = {aid: alloc(self.capacity, d) for aid, d in dec_obs_dims.items()}
        self.next_obs_per_agent = {aid: alloc(self.capacity, d) for aid, d in dec_obs_dims.items()}

    def _scatter_copy(self, dst: torch.Tensor, src: torch.Tensor, base: int, n: int):
        non_blocking = self._pinned and src.device.type == "cuda"
        end = base + n
        if end <= self.capacity:
            dst[base:end].copy_(src, non_blocking=non_blocking)
        else:
            first = self.capacity - base
            dst[base:].copy_(src[:first], non_blocking=non_blocking)
            dst[: n - first].copy_(src[first:], non_blocking=non_blocking)

    def _record_write_event(self):
        if self._pinned and torch.cuda.is_available() and str(self.device).startswith("cuda"):
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(device=self.device))
            self._write_event = event

    def wait_writes(self):
        if self._write_event is not None:
            self._write_event.synchronize()
            self._write_event = None

    def add(
        self,
        *,
        state,
        next_state,
        high_joint_action,
        next_high_joint_action,
        reward,
        discount,
        terminated,
        obs_per_agent,
        next_obs_per_agent,
        length,
        agent_mask=None,
        next_alive_per_agent=None,
    ):
        """Store a completed target-selection transition."""

        n = state.shape[0]
        if n <= 0:
            return
        base = self._ptr
        self._scatter_copy(self.state, state, base, n)
        self._scatter_copy(self.next_state, next_state, base, n)
        if self._compact_actions:
            high_idx = high_joint_action.view(n, self._num_physical_agents, self._dec_dim).argmax(dim=-1)
            next_high_idx = next_high_joint_action.view(n, self._num_physical_agents, self._dec_dim).argmax(dim=-1)
            self._scatter_copy(self.high_dec_idx, high_idx.to(self.high_dec_idx.dtype), base, n)
            self._scatter_copy(self.next_high_dec_idx, next_high_idx.to(self.next_high_dec_idx.dtype), base, n)
        else:
            self._scatter_copy(self.high_joint_action, high_joint_action, base, n)
            self._scatter_copy(self.next_high_joint_action, next_high_joint_action, base, n)
        self._scatter_copy(self.reward, reward.view(-1), base, n)
        self._scatter_copy(self.discount, discount.view(-1), base, n)
        self._scatter_copy(self.terminated, terminated.bool().view(-1), base, n)
        self._scatter_copy(self.length, length.long().view(-1), base, n)
        if agent_mask is None:
            agent_mask = torch.ones(n, len(self.agent_ids), dtype=torch.bool, device=state.device)
        self._scatter_copy(self.agent_mask, agent_mask.bool(), base, n)
        for aid in self.agent_ids:
            self._scatter_copy(self.obs_per_agent[aid], obs_per_agent[aid], base, n)
            self._scatter_copy(self.next_obs_per_agent[aid], next_obs_per_agent[aid], base, n)
        if next_alive_per_agent is not None:
            for aid in self.agent_ids:
                al = next_alive_per_agent.get(aid)
                if al is None:
                    al = torch.ones(n, dtype=torch.bool, device=state.device)
                self._scatter_copy(self.next_alive[aid], al.bool().view(-1), base, n)
        self._ptr = (base + n) % self.capacity
        self._size = min(self._size + n, self.capacity)
        self._record_write_event()

    def _take(self, key: str, src: torch.Tensor, indices: torch.Tensor):
        return self._stager.take(key, src, indices)

    def _take_high_action(self, key: str, indices: torch.Tensor, next_action: bool = False):
        if not self._compact_actions:
            src = self.next_high_joint_action if next_action else self.high_joint_action
            return self._take(key, src, indices)
        src = self.next_high_dec_idx if next_action else self.high_dec_idx
        dec_idx = self._take(f"{key}:idx", src, indices)
        return _expand_high_action(dec_idx, self._dec_dim, dtype=torch.float32)

    def sample(self, batch_size: int) -> dict:
        """Sample decision transitions for policy updates."""
        self.wait_writes()
        idx = torch.randint(0, self._size, (batch_size,), device=self.store_device)
        return {
            "state": self._take("state", self.state, idx),
            "next_state": self._take("next_state", self.next_state, idx),
            "high_joint_action": self._take_high_action("high_joint_action", idx),
            "next_high_joint_action": self._take_high_action("next_high_joint_action", idx, next_action=True),
            "reward": self._take("reward", self.reward, idx),
            "discount": self._take("discount", self.discount, idx),
            "terminated": self._take("terminated", self.terminated, idx).to(torch.float32),
            "length": self._take("length", self.length, idx),
            "agent_mask": self._take("agent_mask", self.agent_mask, idx).to(torch.float32),
            "obs_per_agent": {aid: self._take(f"obs:{aid}", self.obs_per_agent[aid], idx) for aid in self.agent_ids},
            "next_obs_per_agent": {
                aid: self._take(f"nobs:{aid}", self.next_obs_per_agent[aid], idx) for aid in self.agent_ids
            },
            "next_alive_per_agent": {
                aid: self._take(f"nal:{aid}", self.next_alive[aid], idx) for aid in self.agent_ids
            },
        }

    def __len__(self):
        return self._size
