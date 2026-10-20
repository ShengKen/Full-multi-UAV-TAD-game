# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.

# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations


_patched = False


def patch_isaaclab_multigent_state() -> None:
    global _patched
    if _patched:
        return
    from skrl.envs.wrappers.torch.isaaclab_envs import IsaacLabMultiAgentWrapper

    _orig_state = IsaacLabMultiAgentWrapper.state

    def state(self):
        if hasattr(self._env, "state"):
            s = self._env.state()
        else:
            s = self._unwrapped.state()
        if s is None:
            return {uid: None for uid in self.possible_agents}
        if isinstance(s, dict):
            return {uid: s[uid] for uid in self.possible_agents}

        return _orig_state(self)

    IsaacLabMultiAgentWrapper.state = state
    _patched = True
