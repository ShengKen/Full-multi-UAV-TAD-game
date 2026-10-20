# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

from dataclasses import dataclass
import os
import re
from types import MethodType

import torch

from MTAD.algorithms.training_units import decision_to_control_step

_PROGRESS_VERSION = 2


class ArchitectureFingerprint:
    VERSION = 2

    def __init__(self, fingerprint: dict):
        self.fingerprint = dict(fingerprint)

    def state_dict(self) -> dict:
        return {"version": self.VERSION, "fingerprint": self.fingerprint}

    def load_state_dict(self, state: dict) -> None:
        return None


def register_architecture_fingerprint(agent, fingerprint: dict) -> ArchitectureFingerprint:
    marker = ArchitectureFingerprint(fingerprint)
    agent.architecture_fingerprint = marker
    for uid in agent.possible_agents:
        agent.checkpoint_modules[uid]["architecture_fingerprint"] = marker
    return marker


def validate_architecture_fingerprint(
    payload: dict, possible_agents: list[str], fingerprint: dict, *, resume_critic: bool = True
) -> None:
    expected = dict(fingerprint)
    is_mlp = all(v.get("backbone", "mlp") == "mlp" for v in expected.values() if isinstance(v, dict))
    found = []
    for uid in possible_agents:
        marker = payload.get(uid, {}).get("architecture_fingerprint")
        if isinstance(marker, dict):
            recorded = marker.get("fingerprint", marker)
            if int(marker.get("version", 1)) < ArchitectureFingerprint.VERSION:
                recorded = {key: value for key, value in recorded.items() if key in expected}
            found.append(recorded)
    if not found:
        requires_fingerprint = bool(expected.get("requires_fingerprint", False))
        if (not is_mlp) or requires_fingerprint:
            reason = "当前架构要求显式指纹" if requires_fingerprint else "不能加载到 GRU actor"
            raise ValueError(f"checkpoint 缺少 architecture_fingerprint，{reason}；请使用同架构 checkpoint")
        return
    if not resume_critic:
        expected = {k: v for k, v in expected.items() if k not in _CRITIC_ONLY_FINGERPRINT_KEYS}
        found = [{k: v for k, v in item.items() if k not in _CRITIC_ONLY_FINGERPRINT_KEYS} for item in found]

    def _fingerprint_match(exp: dict, fnd: dict) -> bool:
        for k in set(exp) | set(fnd):
            ev, fv = exp.get(k), fnd.get(k)
            if k in ("spaces_dims", "team_sizes") and isinstance(ev, dict) and isinstance(fv, dict):
                if any(ev[u] != fv[u] for u in set(ev) & set(fv)):
                    return False
                continue

            if (
                k in _MLP_BACKBONE_FINGERPRINT_KEYS
                and isinstance(ev, dict)
                and isinstance(fv, dict)
                and ev.get("backbone") == "mlp"
                and fv.get("backbone") == "mlp"
            ):
                continue
            if ev != fv:
                return False
        return True

    if any(not _fingerprint_match(expected, item) for item in found):
        raise ValueError(f"checkpoint architecture_fingerprint 与当前配置不匹配：expected={expected}, found={found[0]}")


class TrainingProgress:
    def __init__(self):
        self.format_version = _PROGRESS_VERSION
        self.completed_decision_transitions = 0
        self.completed_control_steps = 0
        self.decision_period = 1

    def set_progress(self, *, decisions: int, controls: int, decision_period: int) -> None:
        self.format_version = _PROGRESS_VERSION
        self.completed_decision_transitions = int(decisions)
        self.completed_control_steps = int(controls)
        self.decision_period = int(decision_period)

    def state_dict(self) -> dict:
        return {
            "format_version": int(self.format_version),
            "completed_decision_transitions": int(self.completed_decision_transitions),
            "completed_control_steps": int(self.completed_control_steps),
            "decision_period": int(self.decision_period),
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state.get("format_version", 0)) >= _PROGRESS_VERSION:
            self.set_progress(
                decisions=int(state.get("completed_decision_transitions", 0)),
                controls=int(state.get("completed_control_steps", 0)),
                decision_period=int(state.get("decision_period", 1)),
            )
            return

        legacy = int(state.get("completed_timesteps", 0))
        self.set_progress(decisions=legacy, controls=legacy, decision_period=1)


@dataclass(frozen=True)
class ResolvedProgress:
    completed_decisions: int
    completed_controls: int
    decision_period: int
    legacy: bool


def register_training_progress(agent) -> TrainingProgress:
    progress = TrainingProgress()
    agent.training_progress = progress
    for uid in agent.possible_agents:
        agent.checkpoint_modules[uid]["training_progress"] = progress
    return progress


def infer_completed_timesteps(path: str) -> int:
    match = re.search(r"agent_(\d+)\.pt$", os.path.basename(path))
    return int(match.group(1)) if match else 0


def resolve_checkpoint_progress(
    path: str,
    payload: dict,
    possible_agents: list[str],
) -> ResolvedProgress:
    states = []
    for uid in possible_agents:
        state = payload[uid].get("training_progress")
        if isinstance(state, dict):
            states.append(state)

    if states:
        first = states[0]
        if any(state != first for state in states[1:]):
            raise ValueError("checkpoint 各 agent 的 training_progress 不一致，拒绝续训")
        if int(first.get("format_version", 0)) >= _PROGRESS_VERSION:
            return ResolvedProgress(
                completed_decisions=int(first.get("completed_decision_transitions", 0)),
                completed_controls=int(first.get("completed_control_steps", 0)),
                decision_period=int(first.get("decision_period", 1)),
                legacy=False,
            )
        legacy_steps = int(first.get("completed_timesteps", infer_completed_timesteps(path)))
    else:
        legacy_steps = infer_completed_timesteps(path)

    return ResolvedProgress(legacy_steps, legacy_steps, 1, legacy=True)


def load_training_checkpoint(agent, payload: dict, *, algo: str, resume_critic: bool) -> dict:
    """Restore a checkpoint and resolve completed training progress."""
    fingerprint = getattr(agent, "_architecture_fingerprint", None)
    if fingerprint is not None:
        validate_architecture_fingerprint(payload, agent.possible_agents, fingerprint, resume_critic=resume_critic)
    _LOWER_CRITIC = {"q_low", "target_q_low", "q_low_optimizer"}

    _matd3_critic = {"critic2", "target_critic2", "critic2_optimizer", "q_low2", "target_q_low2", "q_low2_optimizer"}
    if algo == "mappo":
        critic_names = {"value", "value_optimizer", "state_preprocessor", "value_preprocessor"} | _LOWER_CRITIC
        expected_critic = {"value", "value_optimizer"}
    elif algo == "flatppo":
        critic_names = {"value", "value_optimizer", "state_preprocessor"}
        expected_critic = {"value", "value_optimizer"}
    elif algo == "flattd3":
        critic_names = {"critic", "target_critic", "critic_optimizer", "state_preprocessor"} | _matd3_critic
        expected_critic = {"critic", "target_critic", "critic_optimizer"} | {
            k
            for k in _matd3_critic
            if any(agent.checkpoint_modules[uid].get(k) is not None for uid in agent.possible_agents)
        }
    else:
        critic_names = {"critic", "target_critic", "critic_optimizer", "state_preprocessor"} | _LOWER_CRITIC
        expected_critic = {"critic", "target_critic", "critic_optimizer"}
        if algo == "matd3":
            critic_names = critic_names | _matd3_critic
            expected_critic = expected_critic | {
                k
                for k in _matd3_critic
                if any(agent.checkpoint_modules[uid].get(k) is not None for uid in agent.possible_agents)
            }
        if algo == "matd3ppo":
            critic_names = critic_names | _matd3_critic | {"value", "value_optimizer"}
            expected_critic = {"value", "value_optimizer"} | {
                k
                for k in _matd3_critic
                if any(agent.checkpoint_modules[uid].get(k) is not None for uid in agent.possible_agents)
            }
            if any(agent.checkpoint_modules[uid].get("critic") is not None for uid in agent.possible_agents):
                expected_critic |= {"critic", "target_critic", "critic_optimizer"}

    loaded_names: set[str] = set()
    skipped_names: set[str] = set()
    for uid in agent.possible_agents:
        for name, data in payload[uid].items():
            if name == "training_progress":
                continue
            if not resume_critic and name in critic_names:
                skipped_names.add(name)
                continue
            module = agent.checkpoint_modules[uid].get(name)
            if module is None:
                continue
            module.load_state_dict(data)
            loaded_names.add(name)

    if algo in ("maddpg", "matd3", "matd3ppo", "flattd3"):
        _synced: set[int] = set()
        for uid in agent.possible_agents:
            modules = payload[uid]
            if "policy" in modules and "target_policy" not in modules:
                if id(agent.target_actors[uid]) in _synced:
                    continue
                _synced.add(id(agent.target_actors[uid]))
                agent.target_actors[uid].load_state_dict(agent.actors[uid].state_dict())

    return {
        "loaded": sorted(loaded_names),
        "skipped": sorted(skipped_names),
        "critic_complete": resume_critic and expected_critic.issubset(loaded_names),
    }


_SHAPE_CHECK_SKIP = {
    "actor_optimizer",
    "critic_optimizer",
    "value_optimizer",
    "optimizer",
    "scheduler",
    "amp_scaler",
    "training_progress",
    "q_low_optimizer",
    "critic2_optimizer",
    "q_low2_optimizer",
}


_MLP_BACKBONE_FINGERPRINT_KEYS = frozenset({"upper_backbone", "lower_backbone", "upper", "lower"})


_CRITIC_ONLY_FINGERPRINT_KEYS = frozenset(
    {
        "q_low_input",
        "q_low_act_dim",
        "high_critic_action_dim",
        "replay_semantics",
        "critic_backbone",
        "critic_state",
        "matd3_twin",
        "matd3_smooth",
        "matd3_smooth_std",
        "matd3_smooth_clip",
        "matd3_delay",
        "matd3_policy_delay",
        "lower_critic_mode",
    }
)


_LOWER_CRITIC_NET = frozenset({"q_low", "target_q_low", "q_low2", "target_q_low2"})


def parse_per_uid_cli(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    result: dict[str, str] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise ValueError(f"--per_uid_checkpoint 每条需为 uid=path：{pair!r}")
        uid, path = pair.split("=", 1)
        uid, path = uid.strip(), path.strip()
        if not uid or not path:
            raise ValueError(f"--per_uid_checkpoint uid/path 不能为空：{pair!r}")
        result[uid] = path
    return result


_TEAM_PREFIXES = ("T", "A", "D")


def _agent_team(aid: str) -> str:
    return aid.split("/")[0].split("_")[0]


def expand_team_prefix(per_uid: dict[str, str], possible_agents: list[str]) -> dict[str, str]:
    if not per_uid:
        return {}
    agent_set = set(possible_agents)
    expanded: dict[str, str] = {}
    for key, value in per_uid.items():
        if key in agent_set:
            expanded[key] = value
            continue

        if "/" in key:
            drone_key, layer = key.split("/", 1)
        else:
            drone_key, layer = key, None
        if drone_key not in _TEAM_PREFIXES:
            expanded[key] = value
            continue
        matched = []
        for aid in possible_agents:
            if _agent_team(aid) != drone_key:
                continue
            if layer is not None and (aid.split("/", 1)[1] if "/" in aid else None) != layer:
                continue
            matched.append(aid)
        if not matched:
            raise ValueError(f"队前缀 {key!r} 未匹配任何 agent；合法 agent={possible_agents}")
        for aid in matched:
            expanded[aid] = value
    return expanded


def resolve_resume_sources(
    possible_agents: list[str],
    *,
    cli_per_uid: dict[str, str] | None,
    yaml_per_uid: dict[str, str] | None,
    cli_checkpoint: str | None,
    yaml_resume: str,
) -> dict[str, str | None]:
    cli_per_uid = cli_per_uid or {}
    yaml_per_uid = yaml_per_uid or {}
    yaml_resume = str(yaml_resume or "").strip()
    unknown = (set(cli_per_uid) | set(yaml_per_uid)) - set(possible_agents)
    if unknown:
        raise ValueError(f"per-uid 源含未知 uid {sorted(unknown)}；合法 uid={possible_agents}")
    resolved: dict[str, str | None] = {}
    for uid in possible_agents:
        if uid in cli_per_uid:
            resolved[uid] = cli_per_uid[uid]
        elif uid in yaml_per_uid:
            resolved[uid] = str(yaml_per_uid[uid] or "").strip() or None
        elif cli_checkpoint:
            resolved[uid] = cli_checkpoint
        elif yaml_resume:
            resolved[uid] = yaml_resume
        else:
            resolved[uid] = None
    return resolved


def _check_state_dict_shapes(name: str, source_sd: dict, target_module) -> None:
    target_sd = target_module.state_dict()
    for key, src_val in source_sd.items():
        if not hasattr(src_val, "shape"):
            continue
        if key not in target_sd:
            continue
        src_shape = tuple(src_val.shape)
        tgt_shape = tuple(target_sd[key].shape)
        if src_shape != tgt_shape:
            raise ValueError(
                f"结构不兼容：module={name} key={key} "
                f"source shape={src_shape} vs current shape={tgt_shape}。"
                f"常见原因：critic_mode(team/per_agent)、hierarchical_actor、"
                f"is_discrete、use_layer_norm、hidden 层宽或 obs_dim 变化。"
            )


def load_and_validate_per_uid_sources(
    uid_sources: dict[str, str | None],
    possible_agents: list[str],
    algo: str,
    agent,
    *,
    resume_critic: bool,
    source_uids: dict[str, str] | None = None,
) -> dict[str, dict]:
    """Load compatible policy sources for the requested agents."""

    fingerprint = getattr(agent, "_architecture_fingerprint", None)
    required = {"policy", "value"} if algo == "mappo" else {"policy"}
    if algo == "mappo" and not resume_critic:
        required = {"policy"}

    critic_skip: set[str] = (
        set()
        if resume_critic
        else {
            "value",
            "value_optimizer",
            "value_preprocessor",
            "critic",
            "target_critic",
            "critic_optimizer",
            "critic2",
            "target_critic2",
        }
        | set(_LOWER_CRITIC_NET)
    )
    source_uids = source_uids or {}
    unknown_source_targets = set(source_uids) - set(possible_agents)
    if unknown_source_targets:
        raise ValueError(
            f"per_uid_source 含未知目标 agent {sorted(unknown_source_targets)}；合法 agent={possible_agents}"
        )
    path_cache: dict[str, dict] = {}

    def _load(path: str) -> dict:
        if path not in path_cache:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(payload, dict):
                raise ValueError(f"checkpoint 顶层必须是 dict：{path}")
            if fingerprint is not None:
                validate_architecture_fingerprint(
                    payload, list(payload.keys()), fingerprint, resume_critic=resume_critic
                )
            path_cache[path] = payload
        return path_cache[path]

    uid_payloads: dict[str, dict] = {}
    for uid in possible_agents:
        path = uid_sources.get(uid)
        if not path:
            continue
        payload = _load(path)

        explicit_source = source_uids.get(uid)
        if explicit_source is not None:
            if explicit_source not in payload:
                raise ValueError(f"源 {path} 不含显式指定的 source uid={explicit_source}（target={uid}）")
            sub = payload[explicit_source]
        elif uid in payload:
            sub = payload[uid]
        else:
            bare = uid.split("/")[0]
            if bare in payload:
                sub = payload[bare]
            else:
                team = bare.split("_")[0]
                layer = uid.split("/", 1)[1] if "/" in uid else None
                same_team_sources = [
                    k for k in payload.keys() if isinstance(k, str) and k.split("/")[0].split("_")[0] == team
                ]
                if layer is not None:
                    same_layer = [k for k in same_team_sources if "/" in k and k.split("/", 1)[1] == layer]
                    same_team_sources = same_layer or same_team_sources
                if same_team_sources:
                    source_key = sorted(same_team_sources)[0]
                    sub = payload[source_key]
                    print(f"[TAD] 源 {path} 不含 uid={uid}/bare={bare}，自动使用同队源 {source_key}")
                else:
                    raise ValueError(
                        f"源 {path} 不含 uid={uid}（也无 drone uid={bare} / 同队源）；可用 per_uid_source 显式选源网络"
                    )

        sub = _adapt_hier_sub_payload(sub, uid, algo)
        if not isinstance(sub, dict):
            raise ValueError(f"源 {path}[{uid}] 必须是 module dict")

        required_uid = {m for m in required if agent.checkpoint_modules[uid].get(m) is not None}
        missing = sorted(required_uid - sub.keys())
        if missing:
            raise ValueError(f"源 {path}[{uid}] 缺必要模块 {missing}")

        for name, data in sub.items():
            if name in _SHAPE_CHECK_SKIP or name in critic_skip:
                continue
            module = agent.checkpoint_modules[uid].get(name)
            if module is None:
                continue
            _check_state_dict_shapes(f"{uid}/{name}", data, module)
        uid_payloads[uid] = sub
    return uid_payloads


def _adapt_hier_sub_payload(sub: dict, uid: str, algo: str) -> dict:
    if not isinstance(sub, dict) or "policy" not in sub:
        return sub
    if algo in ("maddpg", "matd3") and uid.endswith("/ctrl"):
        remapped = dict(sub)
        changed = False
        for module_name in ("policy", "target_policy"):
            state_dict = sub.get(module_name)
            if isinstance(state_dict, dict) and any(k.startswith("net_container.") for k in state_dict):
                remapped[module_name] = {
                    (k.replace("net_container.", "net.") if k.startswith("net_container.") else k): v
                    for k, v in state_dict.items()
                }
                changed = True
        if changed:
            sub = remapped
    return sub


def resolve_multi_source_progress(
    uid_sources: dict[str, str | None],
    uid_payloads: dict[str, dict],
    possible_agents: list[str],
) -> ResolvedProgress:
    path_to_uids: dict[str, list[str]] = {}
    for uid in possible_agents:
        path = uid_sources.get(uid)
        if not path or uid not in uid_payloads:
            continue
        path_to_uids.setdefault(path, []).append(uid)
    path_to_progress: dict[str, ResolvedProgress] = {}
    for path, uids in path_to_uids.items():
        sub_payload = {uid: uid_payloads[uid] for uid in uids}
        prog = resolve_checkpoint_progress(path, sub_payload, uids)
        path_to_progress[path] = prog
        print(
            f"[TAD] 源 {path} (uids={uids}): decisions={prog.completed_decisions} "
            f"controls={prog.completed_controls} period={prog.decision_period} "
            f"legacy={prog.legacy}"
        )
    if not path_to_progress:
        return ResolvedProgress(0, 0, 1, legacy=False)
    best_path = max(path_to_progress, key=lambda p: path_to_progress[p].completed_controls)
    best = path_to_progress[best_path]
    if len(path_to_progress) > 1:
        print(f"[TAD] 多源进度不一致，取 max(completed_controls)={best.completed_controls} 作起点（主源={best_path}）")
    return best


class StepMappedWriter:
    def __init__(self, writer, *, origin_decisions: int, origin_controls: int, decision_period: int):
        self._writer = writer
        self._origin_decisions = int(origin_decisions)
        self._origin_controls = int(origin_controls)
        self._decision_period = int(decision_period)

    def add_scalar(self, tag: str, value, timestep: int):
        control_step = decision_to_control_step(
            timestep,
            origin_decisions=self._origin_decisions,
            origin_controls=self._origin_controls,
            decision_period=self._decision_period,
        )
        return self._writer.add_scalar(tag=tag, value=value, timestep=control_step)

    def __getattr__(self, name):
        return getattr(self._writer, name)


def install_timestep_adapter(
    agent,
    *,
    completed_decisions: int,
    completed_controls: int,
    additional_decisions: int,
    decision_period: int,
) -> None:
    origin_decisions = int(completed_decisions)
    origin_controls = int(completed_controls)
    period = int(decision_period)
    total_decisions = origin_decisions + int(additional_decisions)
    target_controls = origin_controls + int(additional_decisions) * period
    agent.training_progress.set_progress(decisions=origin_decisions, controls=origin_controls, decision_period=period)

    def _wrap(name: str):
        original = getattr(agent, name)

        def wrapped(self, *args, **kwargs):
            local_t = int(kwargs["timestep"])
            absolute_decision = local_t + origin_decisions
            kwargs["timestep"] = absolute_decision
            kwargs["timesteps"] = total_decisions
            if name == "post_interaction":
                completed_decision = absolute_decision + 1
                completed_control = decision_to_control_step(
                    completed_decision,
                    origin_decisions=origin_decisions,
                    origin_controls=origin_controls,
                    decision_period=period,
                )
                self.training_progress.set_progress(
                    decisions=completed_decision,
                    controls=completed_control,
                    decision_period=period,
                )
            return original(*args, **kwargs)

        setattr(agent, name, MethodType(wrapped, agent))

    for method_name in ("pre_interaction", "act", "record_transition", "post_interaction"):
        _wrap(method_name)

    original_write_checkpoint = agent.write_checkpoint

    def mapped_write_checkpoint(self, *, timestep: int, timesteps: int):
        control_step = decision_to_control_step(
            timestep,
            origin_decisions=origin_decisions,
            origin_controls=origin_controls,
            decision_period=period,
        )

        os.makedirs(os.path.join(self.experiment_dir, "checkpoints"), exist_ok=True)
        return original_write_checkpoint(timestep=control_step, timesteps=target_controls)

    agent.write_checkpoint = MethodType(mapped_write_checkpoint, agent)
    agent._step_mapping = {
        "origin_decisions": origin_decisions,
        "origin_controls": origin_controls,
        "decision_period": period,
        "target_decisions": total_decisions,
        "target_controls": target_controls,
    }


def wrap_writer_steps(agent, writer=None) -> None:
    base_writer = writer if writer is not None else agent.writer
    mapping = agent._step_mapping
    agent.writer = StepMappedWriter(
        base_writer,
        origin_decisions=mapping["origin_decisions"],
        origin_controls=mapping["origin_controls"],
        decision_period=mapping["decision_period"],
    )
