# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class TrainingBudget:
    env_type: str
    source: str
    decision_period: int
    requested_control_steps: int
    additional_decisions: int
    effective_control_steps: int

    @property
    def rounded_control_steps(self) -> int:
        return self.effective_control_steps - self.requested_control_steps

    @property
    def is_decision_env(self) -> bool:
        return self.env_type == "tad_decision"

    def as_dict(self) -> dict:
        data = asdict(self)
        data["rounded_control_steps"] = self.rounded_control_steps
        return data


def resolve_training_budget(
    *,
    env_type: str,
    requested_timesteps: int,
    decision_period: int = 1,
    algo: str,
    rollouts: int = 1,
    max_iterations: int = 0,
    cli_timesteps_explicit: bool = False,
) -> TrainingBudget:
    period = int(decision_period) if env_type == "tad_decision" else 1
    if period <= 0:
        raise ValueError(f"decision_period 必须为正整数，当前={period}")

    requested = int(requested_timesteps)
    max_iter = int(max_iterations or 0)
    rollout_count = int(rollouts)

    use_iterations = algo == "mappo" and max_iter > 0 and not cli_timesteps_explicit
    if use_iterations:
        if rollout_count <= 0:
            raise ValueError(f"MAPPO rollouts 必须为正整数，当前={rollout_count}")
        decisions = max_iter * rollout_count
        controls = decisions * period
        return TrainingBudget(
            env_type=env_type,
            source="max_iterations",
            decision_period=period,
            requested_control_steps=controls,
            additional_decisions=decisions,
            effective_control_steps=controls,
        )

    if requested <= 0:
        raise ValueError(f"timesteps 必须为正整数，当前={requested}")
    decisions = math.ceil(requested / period)
    effective = decisions * period
    return TrainingBudget(
        env_type=env_type,
        source="timesteps",
        decision_period=period,
        requested_control_steps=requested,
        additional_decisions=decisions,
        effective_control_steps=effective,
    )


def resolve_periodic_interval(value, *, budget: TrainingBudget, divisor: int) -> int | str:
    if not budget.is_decision_env:
        return value
    if value == "auto":
        return max(1, math.ceil(budget.additional_decisions / divisor))
    try:
        interval = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"无效写出间隔 {value!r}；需为正整数、0 或 'auto'") from exc
    if interval <= 0:
        return 0
    return max(1, math.ceil(interval / budget.decision_period))


def decision_to_control_step(
    decision_step: int,
    *,
    origin_decisions: int,
    origin_controls: int,
    decision_period: int,
) -> int:
    delta = int(decision_step) - int(origin_decisions)
    return int(origin_controls) + delta * int(decision_period)
