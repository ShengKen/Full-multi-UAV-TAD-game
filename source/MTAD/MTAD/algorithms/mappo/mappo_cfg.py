# [TAD vendor] 衍生自 skrl 2.1.0 的 MAPPO_CFG（MIT, Copyright skrl contributors）


from __future__ import annotations

from typing import Callable

import dataclasses

from skrl.multi_agents.torch import MultiAgentCfg


@dataclasses.dataclass(kw_only=True)
class MAPPO_CFG(MultiAgentCfg):
    rollouts: int = 16

    learning_epochs: int | dict[str, int] = 8

    mini_batches: int | dict[str, int] = 2

    discount_factor: float | dict[str, float] = 0.99

    gae_lambda: float | dict[str, float] = 0.95

    learning_rate: float | tuple[float, float] | dict[str, float | tuple[float, float]] = 1e-3

    learning_rate_scheduler: (
        type | tuple[type | None, type | None] | dict[str, type | tuple[type | None, type | None]] | None
    ) = None

    learning_rate_scheduler_kwargs: dict | tuple[dict, dict] | dict[str, dict | tuple[dict, dict]] = dataclasses.field(
        default_factory=dict
    )

    observation_preprocessor: type | None | dict[str, type | None] = None

    observation_preprocessor_kwargs: dict | dict[str, dict] = dataclasses.field(default_factory=dict)

    state_preprocessor: type | None | dict[str, type | None] = None

    state_preprocessor_kwargs: dict | dict[str, dict] = dataclasses.field(default_factory=dict)

    value_preprocessor: type | None | dict[str, type | None] = None

    value_preprocessor_kwargs: dict | dict[str, dict] = dataclasses.field(default_factory=dict)

    random_timesteps: int = 0

    learning_starts: int = 0

    grad_norm_clip: float | dict[str, float] = 0.5

    ratio_clip: float | dict[str, float] = 0.2

    value_clip: float | dict[str, float] = 0.2

    entropy_loss_scale: float | dict[str, float] = 0.0

    value_loss_scale: float | dict[str, float] = 2.5

    kl_threshold: float | dict[str, float] = 0.0

    time_limit_bootstrap: bool | dict[str, bool] = False

    rewards_shaper: Callable | None = None

    mixed_precision: bool = False

    def expand(self, *, possible_agents: list[str]) -> None:
        super().expand(
            possible_agents=possible_agents,
            immutable=["learning_starts", "mixed_precision", "random_timesteps", "rewards_shaper", "rollouts"],
        )

        for uid in possible_agents:
            if not isinstance(self.learning_rate[uid], (tuple, list)):
                self.learning_rate[uid] = (self.learning_rate[uid], self.learning_rate[uid])

            if self.learning_rate_scheduler[uid] is None:
                self.learning_rate_scheduler[uid] = (None, None)
            elif not isinstance(self.learning_rate_scheduler[uid], (tuple, list)):
                self.learning_rate_scheduler[uid] = (
                    self.learning_rate_scheduler[uid],
                    self.learning_rate_scheduler[uid],
                )

            if not isinstance(self.learning_rate_scheduler_kwargs[uid], (tuple, list)):
                self.learning_rate_scheduler_kwargs[uid] = (
                    self.learning_rate_scheduler_kwargs[uid],
                    self.learning_rate_scheduler_kwargs[uid],
                )
