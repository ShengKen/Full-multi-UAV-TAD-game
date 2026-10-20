from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import re

import torch


_MEMBER_INDEX_RE = re.compile(r"_(\d+)$")


def physical_uid(agent_id: str) -> str:
    return agent_id.split("/", 1)[0]


def _member_sort_key(agent_id: str) -> tuple[int, str, str]:
    uid = physical_uid(agent_id)
    match = _MEMBER_INDEX_RE.search(uid)
    if match is None:
        raise ValueError(f"team member key has no numeric uid suffix: {agent_id!r}")
    return int(match.group(1)), uid, agent_id


def _alive_for_member(alive: Mapping[str, torch.Tensor], member: str) -> torch.Tensor:
    if member in alive:
        return alive[member]
    uid = physical_uid(member)
    if uid in alive:
        return alive[uid]
    raise KeyError(f"missing alive mask for {member!r} (physical uid {uid!r})")


def select_dynamic_anchor_indices(
    alive: Mapping[str, torch.Tensor],
    members: Sequence[str],
    *,
    fallback_member: str | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    if not members:
        raise ValueError("members must not be empty")
    ordered = tuple(sorted(members, key=_member_sort_key))
    if fallback_member is None:
        fallback_member = ordered[0]
    if fallback_member not in ordered:
        raise ValueError(f"fallback_member {fallback_member!r} is not in members")
    fallback_index = ordered.index(fallback_member)

    alive_stack = torch.stack([_alive_for_member(alive, member) for member in ordered], dim=1)
    if device is not None:
        alive_stack = alive_stack.to(device=device)
    alive_stack = alive_stack.to(dtype=torch.bool)
    if alive_stack.ndim != 2:
        raise ValueError(f"alive masks must stack to [num_envs, num_members], got {tuple(alive_stack.shape)}")

    selected = alive_stack.to(torch.int64).argmax(dim=1)
    any_alive = alive_stack.any(dim=1)
    return torch.where(any_alive, selected, torch.full_like(selected, fallback_index))


def select_dynamic_team_anchor(
    states: Mapping[str, torch.Tensor],
    alive: Mapping[str, torch.Tensor],
    members: Sequence[str],
    *,
    fallback_member: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not members:
        raise ValueError("members must not be empty")

    ordered = tuple(sorted(members, key=_member_sort_key))
    missing = [member for member in ordered if member not in states]
    if missing:
        raise KeyError(f"missing states for team members: {missing}")

    state_stack = torch.stack([states[member] for member in ordered], dim=1)
    selected = select_dynamic_anchor_indices(alive, ordered, fallback_member=fallback_member, device=state_stack.device)
    if state_stack.shape[0] != selected.shape[0]:
        raise ValueError("state and alive masks have different num_envs")

    row = torch.arange(state_stack.shape[0], device=state_stack.device)
    return state_stack[row, selected], selected


def _broadcast_mask(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim > values.ndim:
        raise ValueError(f"mask ndim {mask.ndim} exceeds values ndim {values.ndim}")
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    try:
        return torch.broadcast_to(mask, values.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} is not broadcastable to values shape {tuple(values.shape)}"
        ) from exc


def masked_sum_count(
    values: torch.Tensor,
    mask: torch.Tensor,
    *,
    dim: int | tuple[int, ...],
    keepdim: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    expanded = _broadcast_mask(values, mask).to(device=values.device)
    weights = expanded.to(dtype=values.dtype)
    return (
        (values * weights).sum(dim=dim, keepdim=keepdim),
        weights.sum(dim=dim, keepdim=keepdim),
    )


def masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    *,
    dim: int | tuple[int, ...],
    keepdim: bool = False,
) -> torch.Tensor:
    total, count = masked_sum_count(values, mask, dim=dim, keepdim=keepdim)
    return total / count.clamp(min=1)


class DecisionTrajectoryCollector:
    def __init__(self, *, num_envs: int, max_completed_samples: int = 65536, gamma_c: float = 1.0) -> None:
        self.num_envs = int(num_envs)
        self.max_completed_samples = int(max_completed_samples)
        self._pending: list[dict | None] = [None] * self.num_envs
        self._current: list[list[dict]] = [[] for _ in range(self.num_envs)]
        self._completed: list[list[dict]] = []
        self._completed_samples = 0

        self._gamma_c = float(gamma_c)

        self._pending_active: torch.Tensor | None = None
        self._reward_acc: torch.Tensor | None = None
        self._seg_disc: torch.Tensor | None = None

        self._switch_acc: torch.Tensor | None = None

    @property
    def completed_samples(self) -> int:
        return self._completed_samples

    @property
    def completed_trajectories(self) -> int:
        return len(self._completed)

    def step(
        self,
        *,
        boundary: torch.Tensor,
        done: torch.Tensor,
        reward: torch.Tensor,
        active: torch.Tensor,
        sample_fields: Mapping[str, torch.Tensor],
        death: torch.Tensor | None = None,
        anchor_switch: torch.Tensor | None = None,
    ) -> None:
        boundary = boundary.bool().view(-1)
        done = done.bool().view(-1)
        active = active.bool().view(-1)
        reward = reward.view(-1)
        if death is not None:
            death = death.bool().view(-1)
        if anchor_switch is not None:
            anchor_switch = anchor_switch.bool().view(-1)
        if any(x.numel() != self.num_envs for x in (boundary, done, active, reward)):
            raise ValueError("collector inputs must have one row per environment")

        if self._pending_active is None:
            self._pending_active = reward.new_zeros(self.num_envs, dtype=torch.bool)
            self._reward_acc = reward.new_zeros(self.num_envs)
            self._seg_disc = reward.new_ones(self.num_envs)
        if anchor_switch is not None and self._switch_acc is None:
            self._switch_acc = reward.new_zeros(self.num_envs, dtype=torch.bool)

        for env_i in boundary.nonzero().flatten().tolist():
            self._close_sample(env_i)
            if bool(active[env_i]):
                self._pending[env_i] = {name: value[env_i].detach().clone() for name, value in sample_fields.items()}
                self._pending[env_i]["reward"] = reward.new_zeros(())
                self._pending_active[env_i] = True
                self._reward_acc[env_i] = 0.0
                self._seg_disc[env_i] = 1.0
                if self._switch_acc is not None:
                    self._switch_acc[env_i] = False

        if self._gamma_c == 1.0:
            self._reward_acc = torch.where(self._pending_active, self._reward_acc + reward.detach(), self._reward_acc)
        else:
            self._reward_acc = torch.where(
                self._pending_active, self._reward_acc + self._seg_disc * reward.detach(), self._reward_acc
            )
            self._seg_disc = torch.where(self._pending_active, self._seg_disc * self._gamma_c, self._seg_disc)

        if self._switch_acc is not None and anchor_switch is not None:
            self._switch_acc = torch.where(self._pending_active, self._switch_acc | anchor_switch, self._switch_acc)

        if death is not None:
            for env_i in death.nonzero().flatten().tolist():
                self._close_sample(env_i)

        for env_i in done.nonzero().flatten().tolist():
            self._close_sample(env_i)
            self._finish_trajectory(env_i)

    def _close_sample(self, env_i: int) -> None:
        pending = self._pending[env_i]
        if pending is not None:
            pending["reward"] = self._reward_acc[env_i].clone().detach()

            pending["seg_gamma"] = self._seg_disc[env_i].clone().detach()
            if self._switch_acc is not None:
                pending["anchor_switch"] = self._switch_acc[env_i].clone().detach()
            self._current[env_i].append(pending)
            self._pending[env_i] = None
            self._pending_active[env_i] = False

    def _finish_trajectory(self, env_i: int) -> None:
        trajectory = self._current[env_i]
        if trajectory:
            self._completed.append(trajectory)
            self._completed_samples += len(trajectory)
            self._current[env_i] = []
            self._trim_completed()

    def _trim_completed(self) -> None:
        while self._completed and self._completed_samples > self.max_completed_samples:
            dropped = self._completed.pop(0)
            self._completed_samples -= len(dropped)

    def pop_completed(self) -> list[dict[str, torch.Tensor]]:
        out: list[dict[str, torch.Tensor]] = []
        for trajectory in self._completed:
            keys = trajectory[0].keys()
            out.append({key: torch.stack([sample[key] for sample in trajectory]) for key in keys})
        self._completed.clear()
        self._completed_samples = 0
        return out


def stage_decisions_rebuild_ctrl(agent, observations, actions) -> dict:
    env_h = getattr(agent, "_hier_env", None)
    commit_fn = getattr(env_h, "commit_decisions", None) if env_h is not None else None
    is_dec = getattr(agent, "_is_dec", {})
    ctrl_aids = [a for a in agent.possible_agents if not is_dec.get(a, False)]
    if commit_fn is None:
        return {aid: observations[aid] for aid in ctrl_aids}
    dec_actions = {a: actions[a] for a in agent.possible_agents if is_dec.get(a, False) and a in actions}
    rebuilt = commit_fn(dec_actions)
    return {aid: rebuilt.get(aid, observations[aid]) for aid in ctrl_aids}


def aggregate_team_reward(rewards, infos, members, *, aggr: str, nominal_n: int, team: str):
    team_reward = rewards[members[0]].clone()
    for m in members[1:]:
        team_reward = team_reward + rewards[m]
    if str(aggr) != "mean":
        return team_reward.view(-1)
    alive_count = infos.get("team_alive_count") if isinstance(infos, dict) else None
    if alive_count is not None and team in alive_count:
        denom = alive_count[team].to(team_reward.device).float().view_as(team_reward).clamp(min=1.0)
    else:
        denom = float(max(nominal_n, 1))

    term_info = infos.get("agent_term_reward") if isinstance(infos, dict) else None
    team_term = None
    if isinstance(term_info, dict):
        for m in members:
            v = term_info.get(m, term_info.get(physical_uid(m)))
            if v is None:
                continue
            v = v.to(team_reward.device).view_as(team_reward)
            team_term = v.clone() if team_term is None else team_term + v
    n_nominal = float(max(nominal_n, 1))
    if team_term is not None:
        team_reward = (team_reward - team_term) / denom + team_term / n_nominal
    else:
        team_reward = team_reward / denom
    return team_reward.view(-1)
