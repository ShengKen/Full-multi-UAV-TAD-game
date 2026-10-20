# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.

# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch

_MAX_RESAMPLE_ROUNDS = 32


def _draw_spawn_ball(
    center: torch.Tensor, R: float, k: int, env_origins: torch.Tensor, z_min: float, z_max: float
) -> torch.Tensor:
    device = env_origins.device
    dir_vec = torch.randn(k, 3, device=device)
    dir_vec = dir_vec / (dir_vec.norm(dim=-1, keepdim=True) + 1e-6)
    radius = R * torch.rand(k, 1, device=device) ** (1.0 / 3.0)
    pos = center.unsqueeze(0) + dir_vec * radius + env_origins
    pos[:, 2] = torch.clamp(pos[:, 2], z_min + 0.2, z_max - 0.2)
    return pos


def _spawn_in_arena(
    pos: torch.Tensor, env_origins: torch.Tensor, arena_radius: float, z_min: float, z_max: float
) -> torch.Tensor:
    horiz = (pos[:, :2] - env_origins[:, :2]).norm(dim=-1)
    z = pos[:, 2]
    return (horiz <= arena_radius) & (z >= z_min) & (z <= z_max)


def sample_in_arena_positions(
    center: torch.Tensor,
    R: float,
    k: int,
    env_origins: torch.Tensor,
    *,
    arena_radius: float,
    z_min: float,
    z_max: float,
) -> torch.Tensor:
    pos = _draw_spawn_ball(center, R, k, env_origins, z_min, z_max)
    valid = _spawn_in_arena(pos, env_origins, arena_radius, z_min, z_max)
    for _ in range(_MAX_RESAMPLE_ROUNDS):
        if bool(valid.all()):
            break
        bad = ~valid
        pos[bad] = _draw_spawn_ball(center, R, int(bad.sum().item()), env_origins[bad], z_min, z_max)
        valid = _spawn_in_arena(pos, env_origins, arena_radius, z_min, z_max)

    if not bool(valid.all()):
        xy_loc = pos[:, :2] - env_origins[:, :2]
        horiz = xy_loc.norm(dim=-1, keepdim=True)
        over = horiz > arena_radius
        scale = torch.where(over, arena_radius / (horiz + 1e-6), torch.ones_like(horiz))
        pos[:, :2] = xy_loc * scale + env_origins[:, :2]
    return pos
