# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.

# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import itertools

import torch


TARGET_TEAM = {"T": "A", "A": "T", "D": "A"}


def stack_team(positions: dict[str, torch.Tensor], members: list[str]) -> torch.Tensor:
    if not members:
        sample = next(iter(positions.values()), None)
        if sample is None:
            raise ValueError("stack_team 需要至少一个参考张量来推断 num_envs/device")
        return sample.new_empty((sample.shape[0], 0, sample.shape[-1]))

    return torch.stack([positions[uid] for uid in members], dim=1)


def stack_team_vel(velocities: dict[str, torch.Tensor], members: list[str]) -> torch.Tensor:
    return stack_team(velocities, members)


def min_pair_distance(team_a: torch.Tensor, team_b: torch.Tensor) -> torch.Tensor:
    if team_a.shape[1] == 0 or team_b.shape[1] == 0:
        return torch.full((team_a.shape[0],), float("inf"), device=team_a.device, dtype=team_a.dtype)

    d = torch.cdist(team_a, team_b)

    return d.min(dim=-1).values.min(dim=-1).values


def pair_engagement(
    pos_a: torch.Tensor,
    pos_b: torch.Tensor,
    alive_a: torch.Tensor,
    alive_b: torch.Tensor,
    radius: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Detect capture engagements between two UAV teams."""

    N = pos_a.shape[0]
    z_a = torch.zeros(N, pos_a.shape[1], dtype=torch.bool, device=pos_a.device)
    z_b = torch.zeros(N, pos_b.shape[1], dtype=torch.bool, device=pos_b.device)

    if pos_a.shape[1] == 0 or pos_b.shape[1] == 0:
        return z_a, z_b
    d = torch.cdist(pos_a, pos_b)

    within = (d < radius) & alive_a[:, :, None] & alive_b[:, None, :]
    engaged_a = within.any(dim=2)
    engaged_b = within.any(dim=1)
    return engaged_a, engaged_b


def nearest_pair_closing(
    pos_a: torch.Tensor,
    vel_a: torch.Tensor,
    pos_b: torch.Tensor,
    vel_b: torch.Tensor,
    alive_a: torch.Tensor | None = None,
    alive_b: torch.Tensor | None = None,
) -> torch.Tensor:
    N = pos_a.shape[0]

    if pos_a.shape[1] == 0 or pos_b.shape[1] == 0:
        return torch.zeros(N, device=pos_a.device, dtype=pos_a.dtype)
    nb = pos_b.shape[1]
    d = torch.cdist(pos_a, pos_b)

    if alive_a is not None or alive_b is not None:
        valid = torch.ones_like(d, dtype=torch.bool)
        if alive_a is not None:
            valid = valid & alive_a[:, :, None]
        if alive_b is not None:
            valid = valid & alive_b[:, None, :]
        d = d.masked_fill(~valid, float("inf"))

    has_pair = d.view(N, -1).isfinite().any(dim=-1)
    idx = d.view(N, -1).argmin(dim=-1)
    ia = torch.div(idx, nb, rounding_mode="floor")
    ib = idx % nb
    rows = torch.arange(N, device=pos_a.device)
    pa = pos_a[rows, ia]
    pb = pos_b[rows, ib]
    va = vel_a[rows, ia]
    vb = vel_b[rows, ib]
    r = (pb - pa) / (torch.linalg.norm(pb - pa, dim=-1, keepdim=True) + 1e-8)
    v_rel = vb - va
    c = -(r * v_rel).sum(dim=-1)

    return torch.where(has_pair, c, torch.zeros_like(c))


def per_drone_closing(
    pos_self: torch.Tensor,
    vel_self: torch.Tensor,
    pos_tgt: torch.Tensor,
    vel_tgt: torch.Tensor,
    alive_self: torch.Tensor | None = None,
    alive_tgt: torch.Tensor | None = None,
) -> torch.Tensor:
    r = pos_tgt - pos_self
    norm = torch.linalg.norm(r, dim=-1, keepdim=True)
    r_hat = r / (norm + 1e-8)
    v_rel = vel_tgt - vel_self
    c = -(r_hat * v_rel).sum(dim=-1)
    mask = torch.ones_like(c, dtype=torch.bool)
    if alive_self is not None:
        mask = mask & alive_self.bool()
    if alive_tgt is not None:
        mask = mask & alive_tgt.bool()
    return torch.where(mask, c, torch.zeros_like(c))


def per_drone_nearest_team_closing(
    pos_self: torch.Tensor,
    vel_self: torch.Tensor,
    pos_b: torch.Tensor,
    vel_b: torch.Tensor,
    alive_self: torch.Tensor | None = None,
    alive_b: torch.Tensor | None = None,
) -> torch.Tensor:
    N = pos_self.shape[0]
    if pos_b.shape[1] == 0:
        return torch.zeros(N, device=pos_self.device, dtype=pos_self.dtype)
    d = torch.cdist(pos_self.unsqueeze(1), pos_b).squeeze(1)
    if alive_b is not None:
        d = d.masked_fill(~alive_b.bool(), float("inf"))
    has = d.isfinite().any(dim=-1)
    ib = d.argmin(dim=-1)
    rows = torch.arange(N, device=pos_self.device)
    pb = pos_b[rows, ib]
    vb = vel_b[rows, ib]
    r = pb - pos_self
    r_hat = r / (torch.linalg.norm(r, dim=-1, keepdim=True) + 1e-8)
    c = -(r_hat * (vb - vel_self)).sum(dim=-1)
    mask = has
    if alive_self is not None:
        mask = mask & alive_self.bool()
    return torch.where(mask, c, torch.zeros_like(c))


def greedy_assigned_team_closing(
    chaser_pos: torch.Tensor,
    chaser_vel: torch.Tensor,
    tgt_pos: torch.Tensor,
    tgt_vel: torch.Tensor,
    chaser_alive: torch.Tensor | None = None,
    tgt_alive: torch.Tensor | None = None,
) -> torch.Tensor:
    N, n_ch = chaser_pos.shape[0], chaser_pos.shape[1]
    n_tgt = tgt_pos.shape[1]
    device, dtype = chaser_pos.device, chaser_pos.dtype
    out = torch.zeros(N, n_ch, device=device, dtype=dtype)
    if n_tgt == 0 or n_ch == 0:
        return out
    rows = torch.arange(N, device=device)

    tgt_dead = (~tgt_alive.bool()) if tgt_alive is not None else torch.zeros(N, n_tgt, dtype=torch.bool, device=device)
    d_all = torch.cdist(chaser_pos, tgt_pos).masked_fill(tgt_dead.unsqueeze(1).expand(N, n_ch, n_tgt), float("inf"))
    any_tgt = (~tgt_dead).any(dim=-1)
    taken = tgt_dead.clone()
    ch_alive = chaser_alive.bool() if chaser_alive is not None else torch.ones(N, n_ch, dtype=torch.bool, device=device)
    for ci in range(n_ch):
        d_ci = d_all[:, ci].masked_fill(taken, float("inf"))
        has = d_ci.isfinite().any(dim=-1)

        ib = torch.where(has, d_ci.argmin(dim=-1), d_all[:, ci].argmin(dim=-1))
        pb = tgt_pos[rows, ib]
        vb = tgt_vel[rows, ib]
        r = pb - chaser_pos[:, ci]
        r_hat = r / (torch.linalg.norm(r, dim=-1, keepdim=True) + 1e-8)
        c = -(r_hat * (vb - chaser_vel[:, ci])).sum(dim=-1)
        out[:, ci] = torch.where(ch_alive[:, ci] & any_tgt, c, torch.zeros_like(c))

        consume = ch_alive[:, ci] & has
        taken[rows, ib] = taken[rows, ib] | consume
    return out


_ASSIGN_PERM_CACHE: dict[tuple[int, int], torch.Tensor] = {}

_ASSIGN_PERM_LIMIT = 4096


def min_total_distance_assignment(
    chaser_pos: torch.Tensor,
    slot_pos: torch.Tensor,
    slot_valid: torch.Tensor,
) -> torch.Tensor:
    """Assign targets to minimize total engagement distance."""

    N, C = chaser_pos.shape[0], chaser_pos.shape[1]
    S = slot_pos.shape[1]
    device = chaser_pos.device
    if C == 0 or S == 0:
        return torch.zeros(N, C, dtype=torch.long, device=device)

    BIG = 1.0e6
    valid_b = slot_valid.bool()
    cost = torch.cdist(chaser_pos, slot_pos).masked_fill(~valid_b.unsqueeze(1), BIG)
    k = min(C, S)
    cache_key = (C, S)
    perms = _ASSIGN_PERM_CACHE.get(cache_key)
    if perms is None:
        import math

        n_perms = math.perm(S, k)
        if n_perms > _ASSIGN_PERM_LIMIT:
            out = torch.zeros(N, C, dtype=torch.long, device=device)
            taken = ~valid_b
            rows = torch.arange(N, device=device)
            for ci in range(C):
                d_ci = cost[:, ci].masked_fill(taken, BIG)
                has = (d_ci < BIG).any(dim=-1)
                ib = d_ci.argmin(dim=-1)
                out[:, ci] = ib
                consume = torch.zeros_like(taken)
                consume[rows, ib] = has
                taken = taken | consume

            bad = ~valid_b.gather(1, out)
            if bool(bad.any()):
                near_valid = cost.masked_fill(~valid_b.unsqueeze(1), BIG).argmin(dim=-1)
                near_valid = torch.where(valid_b.any(dim=-1, keepdim=True), near_valid, torch.zeros_like(near_valid))
                out = torch.where(bad, near_valid, out)
            return out
        perms = torch.tensor(list(itertools.permutations(range(S), k)), dtype=torch.long, device=device)
        _ASSIGN_PERM_CACHE[cache_key] = perms

    P = perms.shape[0]
    rows_n = torch.arange(N, device=device).view(N, 1, 1)
    c_idx = torch.arange(k, device=device).view(1, k, 1).expand(N, k, P)
    s_idx = perms.t().unsqueeze(0).expand(N, k, P)
    totals = cost[rows_n, c_idx, s_idx].sum(dim=1)
    best = totals.argmin(dim=-1)
    assign = perms[best]
    out = torch.zeros(N, C, dtype=torch.long, device=device)
    out[:, :k] = assign
    if C > S:
        out[:, S:] = cost.argmin(dim=-1)[:, S:]

    rows_c = torch.arange(C, device=device).view(1, -1)
    bad = ~valid_b.gather(1, out)
    if bool(bad.any()):
        near_valid = cost.masked_fill(~valid_b.unsqueeze(1), BIG).argmin(dim=-1)
        near_valid = torch.where(valid_b.any(dim=-1, keepdim=True), near_valid, torch.zeros_like(near_valid))
        out = torch.where(bad, near_valid, out)
    return out


def per_drone_nearest_team_closing_batch(
    pos_self: torch.Tensor,
    vel_self: torch.Tensor,
    pos_b: torch.Tensor,
    vel_b: torch.Tensor,
    alive_self: torch.Tensor | None = None,
    alive_b: torch.Tensor | None = None,
) -> torch.Tensor:
    N, ns = pos_self.shape[0], pos_self.shape[1]
    if pos_b.shape[1] == 0:
        return torch.zeros(N, ns, device=pos_self.device, dtype=pos_self.dtype)

    diff = pos_self[:, :, None, :] - pos_b[:, None, :, :]
    d = diff.norm(dim=-1)
    if alive_b is not None:
        d = d.masked_fill(~alive_b.bool()[:, None, :], float("inf"))
    has = d.isfinite().any(dim=-1)
    ib = d.argmin(dim=-1)
    rows_n = torch.arange(N, device=pos_self.device)[:, None]
    pb = pos_b[rows_n, ib]
    vb = vel_b[rows_n, ib]
    r = pb - pos_self
    r_hat = r / (r.norm(dim=-1, keepdim=True) + 1e-8)
    c = -(r_hat * (vb - vel_self)).sum(dim=-1)
    mask = has
    if alive_self is not None:
        mask = mask & alive_self.bool()
    return torch.where(mask, c, torch.zeros_like(c))


def per_drone_nearest_team_distance_batch(
    pos_self: torch.Tensor,
    pos_b: torch.Tensor,
    alive_self: torch.Tensor | None = None,
    alive_b: torch.Tensor | None = None,
) -> torch.Tensor:
    N, ns = pos_self.shape[0], pos_self.shape[1]
    if pos_b.shape[1] == 0:
        return torch.zeros(N, ns, device=pos_self.device, dtype=pos_self.dtype)
    diff = pos_self[:, :, None, :] - pos_b[:, None, :, :]
    d = diff.norm(dim=-1)
    if alive_b is not None:
        d = d.masked_fill(~alive_b.bool()[:, None, :], float("inf"))
    has = d.isfinite().any(dim=-1)
    ib = d.argmin(dim=-1)
    rows_n = torch.arange(N, device=pos_self.device)[:, None]
    pb = pos_b[rows_n, ib]
    dist = (pb - pos_self).norm(dim=-1)
    if alive_self is not None:
        has = has & alive_self.bool()
    return torch.where(has, dist, torch.zeros_like(dist))


def per_drone_team_distance_sum_batch(
    pos_self: torch.Tensor,
    pos_b: torch.Tensor,
    alive_self: torch.Tensor | None = None,
    alive_b: torch.Tensor | None = None,
) -> torch.Tensor:
    N, ns = pos_self.shape[0], pos_self.shape[1]
    if pos_b.shape[1] == 0:
        return torch.zeros(N, ns, device=pos_self.device, dtype=pos_self.dtype)
    diff = pos_self[:, :, None, :] - pos_b[:, None, :, :]
    d = diff.norm(dim=-1)
    if alive_b is not None:
        d = d.masked_fill(~alive_b.bool()[:, None, :], 0.0)
    s = d.sum(dim=-1)
    if alive_self is not None:
        s = s.masked_fill(~alive_self.bool(), 0.0)
    return s


def nearest_pair_los_vel(
    pos_p: torch.Tensor,
    vel_p: torch.Tensor,
    pos_e: torch.Tensor,
    vel_e: torch.Tensor,
    alive_p: torch.Tensor | None = None,
    alive_e: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    N = pos_p.shape[0]
    z = torch.zeros(N, device=pos_p.device, dtype=pos_p.dtype)
    if pos_p.shape[1] == 0 or pos_e.shape[1] == 0:
        return z, z
    ne = pos_e.shape[1]
    d = torch.cdist(pos_p, pos_e)
    if alive_p is not None or alive_e is not None:
        valid = torch.ones_like(d, dtype=torch.bool)
        if alive_p is not None:
            valid = valid & alive_p[:, :, None]
        if alive_e is not None:
            valid = valid & alive_e[:, None, :]
        d = d.masked_fill(~valid, float("inf"))
    has_pair = d.view(N, -1).isfinite().any(dim=-1)
    idx = d.view(N, -1).argmin(dim=-1)
    ip = torch.div(idx, ne, rounding_mode="floor")
    ie = idx % ne
    rows = torch.arange(N, device=pos_p.device)
    pp = pos_p[rows, ip]
    pe = pos_e[rows, ie]
    vp = vel_p[rows, ip]
    ve = vel_e[rows, ie]
    r = (pe - pp) / (torch.linalg.norm(pe - pp, dim=-1, keepdim=True) + 1e-8)
    c_los_p = -(r * vp).sum(dim=-1)
    c_los_e = -(r * ve).sum(dim=-1)
    return torch.where(has_pair, c_los_p, z), torch.where(has_pair, c_los_e, z)


def nearest_member_body_frame(
    own_pos: torch.Tensor,
    own_quat: torch.Tensor,
    other_team_pos: torch.Tensor,
    other_team_vel: torch.Tensor,
    own_lin_vel: torch.Tensor,
    quat_inv,
    quat_apply,
    exclude_self_idx: int | None = None,
    other_alive: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = own_pos.shape[0]
    n_members = other_team_pos.shape[1]

    if n_members == 0:
        z3 = torch.zeros(n, 3, device=own_pos.device, dtype=own_pos.dtype)
        z1 = torch.zeros(n, device=own_pos.device, dtype=own_pos.dtype)
        return z3, z3, z1

    rel_pos_w = other_team_pos - own_pos.unsqueeze(1)

    dists = torch.linalg.norm(rel_pos_w, dim=-1)

    masked = exclude_self_idx is not None or other_alive is not None
    if masked:
        dists = dists.clone()
        if exclude_self_idx is not None:
            dists[:, exclude_self_idx] = float("inf")
        if other_alive is not None:
            dists = dists.masked_fill(~other_alive, float("inf"))

    if other_alive is not None:
        has_alive = other_alive.any(dim=-1)
    elif exclude_self_idx is not None and n_members <= 1:
        has_alive = torch.zeros(n, dtype=torch.bool, device=own_pos.device)
    else:
        has_alive = torch.ones(n, dtype=torch.bool, device=own_pos.device)
    nearest_idx = torch.argmin(dists, dim=-1)

    idx = nearest_idx.view(n, 1, 1).expand(n, 1, 3)
    rel_pos_w_nearest = rel_pos_w.gather(1, idx).squeeze(1)
    other_vel_nearest = other_team_vel.gather(1, idx).squeeze(1)

    dist_nearest = torch.linalg.norm(rel_pos_w_nearest, dim=-1)

    rel_vel_w = other_vel_nearest - own_lin_vel

    own_q_inv = quat_inv(own_quat)
    rel_pos_b = quat_apply(own_q_inv, rel_pos_w_nearest)
    rel_vel_b = quat_apply(own_q_inv, rel_vel_w)

    keep = has_alive.unsqueeze(-1)
    rel_pos_b = torch.where(keep, rel_pos_b, torch.zeros_like(rel_pos_b))
    rel_vel_b = torch.where(keep, rel_vel_b, torch.zeros_like(rel_vel_b))
    dist_nearest = torch.where(has_alive, dist_nearest, torch.zeros_like(dist_nearest))

    return rel_pos_b, rel_vel_b, dist_nearest
