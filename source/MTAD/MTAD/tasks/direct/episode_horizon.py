# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.

# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import math
import operator
from typing import SupportsFloat

import torch

_FLOAT_TOLERANCE = 1e-9


def _finite_float(name: str, value: SupportsFloat) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _snap_near_integer(value: float) -> float:
    nearest = round(value)
    if math.isclose(value, nearest, rel_tol=0.0, abs_tol=_FLOAT_TOLERANCE):
        return float(nearest)
    return value


def episode_horizon_step_bounds(max_s: float, jitter_s: float, step_dt: float) -> tuple[int, int]:
    max_s = _finite_float("max_s", max_s)
    jitter_s = _finite_float("jitter_s", jitter_s)
    step_dt = _finite_float("step_dt", step_dt)

    if max_s <= 0.0:
        raise ValueError("max_s must be greater than 0")
    if jitter_s < 0.0 or jitter_s > max_s:
        raise ValueError("jitter_s must satisfy 0 <= jitter_s <= max_s")
    if step_dt <= 0.0:
        raise ValueError("step_dt must be greater than 0")

    high_steps = math.floor(_snap_near_integer(max_s / step_dt))
    if jitter_s == 0.0:
        if high_steps < 1:
            raise ValueError("max_s is shorter than one simulation step")
        return high_steps, high_steps

    lower_s = max_s - jitter_s
    low_steps = max(1, math.ceil(_snap_near_integer(lower_s / step_dt)))
    if high_steps < low_steps:
        raise ValueError("episode horizon interval contains no whole simulation step")
    return low_steps, high_steps


def _integer(name: str, value: int) -> int:
    try:
        return operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def sample_episode_horizon_steps(
    num_envs: int, low_steps: int, high_steps: int, device: str | torch.device
) -> torch.Tensor:
    num_envs = _integer("num_envs", num_envs)
    low_steps = _integer("low_steps", low_steps)
    high_steps = _integer("high_steps", high_steps)

    if num_envs < 0:
        raise ValueError("num_envs must be non-negative")
    if low_steps < 1 or high_steps < low_steps:
        raise ValueError("step bounds must satisfy 1 <= low_steps <= high_steps")
    if high_steps >= torch.iinfo(torch.long).max:
        raise ValueError("high_steps is too large for inclusive torch.randint sampling")

    return torch.randint(low_steps, high_steps + 1, (num_envs,), device=device, dtype=torch.long)
