# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.

# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import os

import yaml


def _parse_normalization(sec) -> dict:
    sec = sec or {}
    return {
        "normalize": bool(sec.get("normalize", True)),
        "value_normalize": bool(sec.get("value_normalize", False)),
        "epsilon": float(sec.get("epsilon", 1e-8)),
        "clip_threshold": float(sec.get("clip_threshold", 5.0)),
    }


def _split_common_sections(d: dict) -> tuple[dict, dict, dict, dict]:
    rl_cfg = d.get("rl", {})

    if isinstance(rl_cfg, dict) and isinstance(rl_cfg.get("value"), dict):
        rl_cfg = rl_cfg["value"]
    train_set_cfg = d.get("train_set", {}) or {}
    wandb_cfg = d.get("wandb", {})
    norm_cfg = _parse_normalization(d.get("normalization", {}))
    return rl_cfg, train_set_cfg, wandb_cfg, norm_cfg


def _migrate_train_set(d: dict) -> bool:
    if d.get("train_set"):
        return False
    migrated: dict = {}
    env_sec = d.get("env", {}) or {}
    if env_sec.get("device"):
        migrated["device"] = str(env_sec["device"])
    scen_sec = d.get("scenario", {}) or {}
    if "ckpt_path" in scen_sec and scen_sec["ckpt_path"]:
        migrated["ckpt_path"] = str(scen_sec["ckpt_path"])
    rl_sec = d.get("rl", {}) or {}
    if isinstance(rl_sec, dict) and isinstance(rl_sec.get("value"), dict):
        rl_sec = rl_sec["value"]
    if "resume_checkpoint" in rl_sec:
        rc = rl_sec["resume_checkpoint"]
        if isinstance(rc, dict):
            migrated["per_uid_checkpoint"] = dict(rc)
            migrated["resume_checkpoint"] = ""
        elif rc:
            migrated["resume_checkpoint"] = str(rc)
    if "resume_critic" in rl_sec:
        migrated["resume_critic"] = bool(rl_sec["resume_critic"])
    if not migrated:
        return False
    d["train_set"] = migrated
    print(f"[TAD] 旧配置兼容：已把训练运行设置迁移到 train_set 节：{sorted(migrated.keys())}")
    return True


def load_env_config(path: str):
    """Load scenario, algorithm, and runtime settings from YAML."""
    path = os.path.abspath(path)
    with open(path, "r") as f:
        d = yaml.safe_load(f) or {}

    task_sec = d.get("task", {}) or {}
    env_type = str(d.get("env_type", task_sec.get("kind", "tad_hierarchical"))).lower()

    _migrate_train_set(d)

    if env_type in ("tad_hierarchical", "tad-hier", "hier"):
        from .tasks.direct.tad_hierarchical.tad_hierarchical_env_cfg import TadHierarchicalEnvCfg

        env_cfg = TadHierarchicalEnvCfg.from_config_dict(d)
    else:
        raise ValueError(f"未知 env_type={env_type!r}，仓库现仅支持 tad_hierarchical")

    rl_cfg, train_set_cfg, wandb_cfg, norm_cfg = _split_common_sections(d)
    return env_cfg, rl_cfg, train_set_cfg, wandb_cfg, norm_cfg


def merge_rl_overrides(rl_cfg: dict, algo: str | None = None, max_iterations: int | None = None) -> dict:
    cfg = dict(rl_cfg)
    resolved_algo = algo if algo is not None else cfg.get("algo")
    if algo is not None:
        cfg["algo"] = algo

    if max_iterations is None:
        max_iterations = cfg.get("max_iterations", 0)
    try:
        max_iterations = int(max_iterations)
    except (TypeError, ValueError):
        max_iterations = 0

    cfg["max_iterations"] = max_iterations if resolved_algo == "mappo" else 0
    return cfg


def save_run_config(
    base_config_path: str,
    log_dir: str,
    env_cfg,
    rl_cfg: dict,
    seed: int,
    argv: list[str],
    device=None,
    wandb_run_id: str | None = None,
    wandb_run_name: str | None = None,
    training_budget: dict | None = None,
    train_set_cfg: dict | None = None,
) -> str:
    d = build_run_config_dict(
        base_config_path,
        env_cfg,
        rl_cfg,
        seed,
        device=device,
        training_budget=training_budget,
        train_set_cfg=train_set_cfg,
    )

    if wandb_run_id is not None or wandb_run_name is not None:
        d.setdefault("wandb", {})
        if wandb_run_id is not None:
            d["wandb"]["run_id"] = wandb_run_id
        if wandb_run_name is not None:
            d["wandb"]["run_name"] = wandb_run_name

    os.makedirs(log_dir, exist_ok=True)
    out_path = os.path.join(log_dir, "run_config.yaml")
    with open(out_path, "w") as f:
        f.write("# 训练实际生效配置（含 CLI 覆盖；由 train_tad 自动生成，供 play_tad 复现）\n")
        yaml.safe_dump(d, f, sort_keys=False, allow_unicode=True, default_flow_style=False)
        f.write("\n# ==== 启动命令（launch command） ====\n# " + " ".join(argv) + "\n")
    return out_path


def build_run_config_dict(
    base_config_path: str,
    env_cfg,
    rl_cfg: dict,
    seed: int,
    device=None,
    training_budget: dict | None = None,
    train_set_cfg: dict | None = None,
) -> dict:
    with open(os.path.abspath(base_config_path), "r") as f:
        d = yaml.safe_load(f) or {}

    d.setdefault("scenario", {})
    d["scenario"]["n_T"] = int(env_cfg.n_T)
    d["scenario"]["n_A"] = int(env_cfg.n_A)
    d["scenario"]["n_D"] = int(env_cfg.n_D)

    d["scenario"]["train_side"] = str(env_cfg.train_side)

    d["scenario"]["roll_period"] = int(env_cfg.roll_period)
    d["scenario"]["roll_order"] = list(env_cfg.roll_order)
    d["scenario"]["roll_freeze_depth"] = str(env_cfg.roll_freeze_depth)

    d["scenario"]["exper_rate"] = [float(r) for r in (getattr(env_cfg, "exper_rate", None) or [0.0, 0.0, 0.0])]

    d["scenario"]["train_takeover"] = str(getattr(env_cfg, "train_takeover", "false"))
    d["scenario"]["train_takeover_step"] = int(getattr(env_cfg, "train_takeover_step", 0) or 0)

    d["scenario"]["train_layer"] = str(env_cfg.train_layer)
    d["scenario"]["decision_period"] = int(env_cfg.decision_period)
    d["scenario"]["selectable_targets"] = {k: list(v) for k, v in env_cfg.selectable_targets.items()}

    d.setdefault("env", {})

    _train_n = int(getattr(env_cfg, "train_num_envs", 0))
    if _train_n <= 0:
        raise ValueError(
            "env_cfg.train_num_envs 未设置（需经 from_config_dict 构造）；run_config 的 "
            "env.num_envs 必须写【训练】env 数（不含 test env）"
        )
    d["env"]["num_envs"] = _train_n
    d["env"]["test_num_envs"] = int(env_cfg.test_num_envs)
    d["env"]["test_every_steps"] = int(env_cfg.test_every_steps)
    d["env"]["test_episodes"] = int(env_cfg.test_episodes)
    d["env"]["test_exper"] = bool(getattr(env_cfg, "test_exper", False))

    d["rl"] = dict(rl_cfg)
    d["rl"]["seed"] = int(seed)
    d.setdefault("train_set", {})
    if train_set_cfg is not None:
        d["train_set"] = {**d["train_set"], **{k: v for k, v in train_set_cfg.items() if v is not None}}
    if device is not None:
        d["train_set"]["device"] = str(device)

    for _stale in ("resume_checkpoint", "resume_critic"):
        d["rl"].pop(_stale, None)
    d.get("scenario", {}).pop("ckpt_path", None)
    d.get("env", {}).pop("device", None)

    for _stale_test in ("test_num_envs", "test_every_steps", "test_episodes", "test_exper"):
        d.get("scenario", {}).pop(_stale_test, None)
    if training_budget is not None:
        d["training_budget"] = dict(training_budget)
    return d


def _has_scenario(path: str) -> bool:
    with open(path, "r") as f:
        d = yaml.safe_load(f) or {}
    return isinstance(d, dict) and "scenario" in d


def find_run_config(checkpoint_path: str) -> str | None:
    p = os.path.dirname(os.path.abspath(checkpoint_path))
    for _ in range(6):
        for name in ("run_config.yaml", "config.yaml"):
            cand = os.path.join(p, name)
            if os.path.isfile(cand) and _has_scenario(cand):
                return cand
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent
    return None


def load_drone_config(path: str) -> dict:
    path = os.path.abspath(path)
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def find_run_drone_config(checkpoint_path: str) -> str | None:
    p = os.path.dirname(os.path.abspath(checkpoint_path))
    for _ in range(6):
        cand = os.path.join(p, "drone_config.yaml")
        if os.path.isfile(cand):
            return cand
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent
    return None


def save_run_drone_config(drone_cfg: dict, log_dir: str) -> str:
    os.makedirs(log_dir, exist_ok=True)
    out_path = os.path.join(log_dir, "drone_config.yaml")
    with open(out_path, "w") as f:
        f.write("# 训练实际生效的无人机配置（由 train_tad 自动生成，供 play_tad 复现）\n")
        yaml.safe_dump(drone_cfg, f, sort_keys=False, allow_unicode=True, default_flow_style=False)
    return out_path
