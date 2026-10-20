from __future__ import annotations

import dataclasses

from ..maddpg.maddpg_cfg import MADDPG_CFG


@dataclasses.dataclass(kw_only=True)
class MATD3_CFG(MADDPG_CFG):
    matd3_twin: bool = True

    matd3_smooth: bool = True

    matd3_smooth_std: float = 0.2

    matd3_smooth_clip: float = 0.5

    matd3_delay: bool = True

    matd3_policy_delay: int = 2
