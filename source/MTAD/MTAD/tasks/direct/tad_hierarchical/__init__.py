# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.

# SPDX-License-Identifier: BSD-3-Clause


import gymnasium as gym


gym.register(
    id="Template-TadHierarchical-v0",
    entry_point=f"{__name__}.tad_hierarchical_env:TadHierarchicalEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tad_hierarchical_env_cfg:TadHierarchicalEnvCfg",
    },
)
