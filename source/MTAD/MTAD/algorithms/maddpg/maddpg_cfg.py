# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.

# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

from typing import Callable

import dataclasses

from skrl.multi_agents.torch import MultiAgentCfg


@dataclasses.dataclass(kw_only=True)
class MADDPG_CFG(MultiAgentCfg):
    gradient_steps: int = 1

    learning_starts: int = 1000

    discount_factor: float = 0.99

    polyak: float = 0.995

    grad_norm_clip: float = 1.0

    random_timesteps: int = 1000

    policy_lr: float = 3.0e-04

    critic_lr: float = 3.0e-04

    batch_size: int = 256

    replay_size: int = 1000000

    n_step: int = 1

    replay_terminal_mix: float = 0.25

    replay_terminal_thresh: float = 20.0

    exploration_noise_std: float = 0.3

    exploration_scheduler: bool = True

    exploration_noise_floor: float = 0.0

    exploration_noise_halflife: float = 0.0

    q_low_lr: float | None = None

    rewards_shaper: Callable | None = None

    mixed_precision: bool = False
