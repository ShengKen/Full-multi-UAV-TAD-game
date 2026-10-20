# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.

# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations


def _wandb_log(payload: dict, step: int) -> None:
    import wandb

    wandb.log(payload, step=step)


class WandbWriter:
    _SKIP_PREFIXES = ("Stats /",)

    _SKIP_TAGS = frozenset(
        {
            "Episode / Total timesteps (max)",
            "Episode / Total timesteps (min)",
            "Episode / Total timesteps (mean)",
        }
    )

    def __init__(self):
        self._buffer = {}
        self._last_step = None

    @staticmethod
    def _skip_tag(tag: str) -> bool:
        if tag.startswith(WandbWriter._SKIP_PREFIXES) or tag in WandbWriter._SKIP_TAGS:
            return True
        compact = tag.replace(" ", "")

        if compact.startswith("Episode/roll_draw_"):
            return True

        if compact.startswith("Episode/roll_") and compact.endswith("_5000"):
            return True
        return False

    def add_scalar(self, tag: str, value, timestep: int):
        if self._skip_tag(tag):
            return

        if self._last_step is not None and timestep != self._last_step and self._buffer:
            _wandb_log(self._buffer, self._last_step)
            self._buffer = {}
        try:
            value = float(value)
        except (TypeError, ValueError):
            pass
        self._buffer[tag] = value
        self._last_step = timestep

    def flush(self):
        if self._buffer:
            step = self._last_step if self._last_step is not None else 0
            _wandb_log(self._buffer, step)
            self._buffer = {}

    def close(self):
        self.flush()
