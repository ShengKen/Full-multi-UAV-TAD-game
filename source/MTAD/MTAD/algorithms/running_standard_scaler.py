# 衍生自 skrl (MIT, Copyright skrl contributors) 的


from __future__ import annotations

import torch
import torch.nn as nn


def _parse_device(device):
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str):
        return torch.device(device)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RunningStandardScaler(nn.Module):
    def __init__(
        self,
        size: int,
        *,
        epsilon: float = 1e-8,
        clip_threshold: float = 5.0,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()

        self.epsilon = epsilon
        self.clip_threshold = clip_threshold

        self.device = _parse_device(device)

        self.register_buffer("running_mean", torch.zeros(size, dtype=torch.float64, device=self.device))
        self.register_buffer("running_variance", torch.ones(size, dtype=torch.float64, device=self.device))
        self.register_buffer("current_count", torch.ones((), dtype=torch.float64, device=self.device))

    def _parallel_variance(self, input_mean: torch.Tensor, input_var: torch.Tensor, input_count: int) -> None:
        delta = input_mean - self.running_mean
        total_count = self.current_count + input_count
        M2 = (
            (self.running_variance * self.current_count)
            + (input_var * input_count)
            + delta**2 * self.current_count * input_count / total_count
        )

        self.running_mean = self.running_mean + delta * input_count / total_count
        self.running_variance = M2 / total_count
        self.current_count = total_count

    def _compute(self, x: torch.Tensor, *, train: bool = False, inverse: bool = False) -> torch.Tensor:
        if train:
            if x.dim() == 3:
                self._parallel_variance(
                    torch.mean(x, dim=(0, 1)), torch.var(x, dim=(0, 1), correction=0), x.shape[0] * x.shape[1]
                )
            else:
                self._parallel_variance(torch.mean(x, dim=0), torch.var(x, dim=0, correction=0), x.shape[0])

        if inverse:
            return (
                torch.sqrt(self.running_variance.float())
                * torch.clamp(x, min=-self.clip_threshold, max=self.clip_threshold)
                + self.running_mean.float()
            )

        return torch.clamp(
            (x - self.running_mean.float()) / (torch.sqrt(self.running_variance.float()) + self.epsilon),
            min=-self.clip_threshold,
            max=self.clip_threshold,
        )

    def forward(
        self, x: torch.Tensor | None, *, train: bool = False, inverse: bool = False, no_grad: bool = True
    ) -> torch.Tensor | None:
        if x is None:
            return None
        if no_grad:
            with torch.no_grad():
                return self._compute(x, train=train, inverse=inverse)
        return self._compute(x, train=train, inverse=inverse)
