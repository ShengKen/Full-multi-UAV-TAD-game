# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.

# SPDX-License-Identifier: BSD-3-Clause


import argparse
import os
import sys


sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "source", "MTAD")))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="TAD 多无人机追逃防训练。")
parser.add_argument("--config", type=str, default="configs/tad_hierarchical_config.yaml", help="配置文件路径。")
parser.add_argument(
    "--log_dir",
    type=str,
    default=None,
    help="训练日志/检查点目录（覆盖默认 logs/tad/<时间戳>_<algo>）；run 脚本预建它，使 stdout 日志与训练产物同目录。",
)
parser.add_argument(
    "--drone_config",
    type=str,
    default="configs/drone_config.yaml",
    help="无人机配置文件路径（资产/物理/电机；默认 configs/drone_config.yaml）。",
)
parser.add_argument(
    "--algo",
    type=str,
    default=None,
    choices=["mappo", "maddpg", "matd3", "matd3ppo", "flattd3", "flatppo", "random"],
    help="算法（覆盖 yaml rl.algo）。random=随机策略冒烟。matd3=MADDPG+TD3 三技巧（twin min/target 平滑/延迟 actor，rl 节 matd3_* 键）。matd3ppo=高层离散 MAPPO+低层 MATD3 混合（docs/codex/high_level_discrete_mappo_math.md）。"
    "flattd3/flatppo=[Flat] 直接控制基线（每架机一个连续 agent、无 /dec 决策层；env 布局随 algo 整体切换，其余配置不变；docs/codex/flat_matd3_flat_mappo_engineering_spec.md）。",
)
parser.add_argument(
    "--flat_obs",
    type=str,
    default=None,
    choices=["pool", "peers"],
    help="[Flat] flat 观测模式（覆盖 yaml rl.flat_obs；仅 flattd3/flatppo 读取）："
    "pool=候选池版（15+8P，当前 79 维）| peers=队友+可选目标版（15+10·N_other，当前 T/D 75、A 105）。",
)
parser.add_argument("--num_envs", type=int, default=None, help="并行环境数（覆盖 yaml；=训练 env 数，不含 test）。")
parser.add_argument(
    "--test_num_envs",
    type=int,
    default=None,
    help="[周期评估] test env 数（覆盖 yaml env.test_num_envs；0=off）。"
    "尾部切片为 test，动作纯确定性 actor，每 --test_every 步一相、满 --test_episodes 回合上报 Test/*。",
)
parser.add_argument(
    "--test_every",
    type=int,
    default=None,
    help="[周期评估] 测试相间隔 control step（覆盖 yaml env.test_every_steps）。",
)
parser.add_argument(
    "--test_episodes", type=int, default=None, help="[周期评估] 每相累计回合数（覆盖 yaml env.test_episodes）。"
)
parser.add_argument(
    "--test_actor",
    type=str,
    default=None,
    choices=["target", "online"],
    help="[周期评估] test env 动作源（覆盖 yaml rl.test_actor）：target=polyak target actor | online=当前 actor 无噪声。"
    "仅 MADDPG 生效；MAPPO 恒 online 确定性。",
)
parser.add_argument(
    "--test_exper",
    type=str,
    default=None,
    choices=["true", "false"],
    help="[test_exper] test 行分类评估（覆盖 yaml env.test_exper）：true=test env round-robin 均分给"
    "各学习队，每类一个“模型队”走确定性 actor、其余队专家/脚本接管（A/D=min-dist+PPN、T=nearest+级联）。",
)
parser.add_argument("--num_steps", type=int, default=60, help="random 冒烟的步数。")
parser.add_argument("--max_iterations", type=int, default=None, help="训练迭代数（= max_iterations*rollouts）。")
parser.add_argument(
    "--timesteps",
    type=int,
    default=None,
    help="本次追加训练步数：TAD-Decision=control step，其他环境=env/control step；覆盖 yaml rl.timesteps。",
)
parser.add_argument("--learning_starts", type=int, default=None, help="MADDPG 开始训练前的回放填充步数（覆盖 yaml）。")
parser.add_argument("--n_T", type=int, default=None, help="Target 队无人机数（覆盖 yaml scenario.n_T）。")
parser.add_argument("--n_A", type=int, default=None, help="Attacker 队无人机数（覆盖 yaml scenario.n_A）。")
parser.add_argument("--n_D", type=int, default=None, help="Defender 队无人机数（覆盖 yaml scenario.n_D）。")
parser.add_argument(
    "--train_side",
    type=str,
    default=None,
    choices=["pursuer", "evader", "both", "T", "A", "D", "all", "roll"],
    help="训练哪一方（覆盖 yaml scenario.train_side）。1v1 追逃：pursuer/evader/both；"
    "TAD-Decision：T/A/D（单方）/all（三方同训）；TAD-Hierarchical：T/A/D/all/roll"
    "（roll=三队轮转训练，非激活队按 --exper_rate 回合级采样是否由 env 接管为专家/脚本）。",
)
parser.add_argument(
    "--train_layer",
    type=str,
    default=None,
    choices=["both", "upper", "lower"],
    help="[TAD-Hierarchical v4] 训练哪些层（覆盖 yaml scenario.train_layer）。"
    "both=上下层联合；upper=只学决策(下层=级联)；lower=只学控制(上层=最近启发式)。",
)
parser.add_argument(
    "--train_takeover",
    type=str,
    default=None,
    choices=["false", "T", "A", "D", "all", "roll"],
    help="[两阶段] 训练模式切换（覆盖 yaml scenario.train_takeover）：false=纯 train_side；"
    "T/A/D/all/roll=按 train_side 训练 --train_takeover_step 步后一次性切到该模式"
    "（窄模式阶段非学习队由 env 接管为专家/脚本）。窄 train_side 起步禁用（早拒）。",
)
parser.add_argument(
    "--train_takeover_step",
    type=int,
    default=None,
    help="[两阶段] 切换步数（绝对 control step，含续训进度；覆盖 yaml scenario.train_takeover_step；0=立即切换）。",
)
parser.add_argument(
    "--exper_rate",
    type=str,
    default=None,
    help="[exper_rate] roll 模式下【非激活队】整回合由 env 接管为专家/脚本的概率"
    "（覆盖 yaml scenario.exper_rate）：标量（三队同值，如 0.2）或 T/A/D 三个逗号分隔"
    "值（如 0.1,0.2,0.3）；每 env 每回合按各队概率独立采样，1=恒专家（=旧 exper）。",
)
parser.add_argument("--checkpoint", type=str, default=None, help="断点续训路径（全局兜底，全员同源）。")
parser.add_argument(
    "--per_uid_checkpoint",
    type=str,
    default=None,
    help="per-uid/层续训源，逗号分隔 agent_id=path（优先于 --checkpoint）。例：A_0/dec=upper.pt,A_0/ctrl=pursuit.pt。",
)
parser.add_argument(
    "--per_uid_source",
    type=str,
    default=None,
    help="目标 agent_id 对应 checkpoint 内的源网络 uid，逗号分隔 target=source。"
    "例：A_1/ctrl=A_0（把 Pursuit ckpt 内 A_0 actor 加载到 A_1/ctrl）。",
)
parser.add_argument(
    "--freeze_agents",
    type=str,
    default=None,
    help="冻结指定 agent/layer 的 actor，逗号分隔。例：A_0/ctrl,A_1/ctrl。CLI 覆盖 train_set.freeze_agents。",
)
parser.add_argument(
    "--resume_critic",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="是否续训 critic/value（--resume_critic / --no-resume_critic；覆盖 rl.resume_critic）。",
)
parser.add_argument("--no_wandb", action="store_true", default=False, help="禁用 wandb 记录。")
parser.add_argument("--seed", type=int, default=None, help="随机种子。")

AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()


if "--device" not in sys.argv:
    try:
        import yaml as _yaml

        with open(os.path.abspath(args_cli.config)) as _f:
            _cfg_peek = _yaml.safe_load(_f) or {}
        _dev = (_cfg_peek.get("train_set") or {}).get("device") or (_cfg_peek.get("env") or {}).get("device")
        if _dev:
            args_cli.device = str(_dev)
    except Exception:
        pass


app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import random
import time
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

from isaaclab_rl.skrl import SkrlVecEnvWrapper
from skrl.trainers.torch import SequentialTrainer

import MTAD.tasks
from MTAD.algorithms.wandb_writer import WandbWriter

from MTAD.algorithms.skrl_state_patch import patch_isaaclab_multigent_state
from MTAD.config_loader import (
    load_env_config,
    load_drone_config,
    merge_rl_overrides,
    save_run_config,
    save_run_drone_config,
    build_run_config_dict,
)

from MTAD.algorithms.agent_builder import build_flat_agent, build_hierarchical_agent
from MTAD.algorithms.resume_checkpoint import (
    expand_team_prefix,
    install_timestep_adapter,
    load_and_validate_per_uid_sources,
    load_training_checkpoint,
    parse_per_uid_cli,
    resolve_multi_source_progress,
    resolve_resume_sources,
    wrap_writer_steps,
)
from MTAD.algorithms.training_units import resolve_periodic_interval, resolve_training_budget
from MTAD.algorithms.train_takeover import init_train_mode
from MTAD.tasks.direct.tad_hierarchical.tad_hierarchical_env_cfg import (
    canonical_train_mode,
    validate_exper_rate,
    validate_train_takeover,
)


def _parse_agent_list(raw, *, option_name: str) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        chunks = raw.split(",")
    elif isinstance(raw, (list, tuple, set)):
        chunks = []
        for item in raw:
            if item is None:
                continue
            chunks.extend(str(item).split(","))
    else:
        raise ValueError(f"{option_name} 必须是字符串或列表：{raw!r}")
    result: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        uid = str(chunk).strip()
        if not uid or uid in seen:
            continue
        result.append(uid)
        seen.add(uid)
    return result


def _git_state() -> dict | None:
    import hashlib
    import subprocess

    def _git(*args: str) -> str:
        return subprocess.run(["git", *args], capture_output=True, text=True, timeout=15).stdout.strip()

    try:
        commit = _git("rev-parse", "HEAD")
        if not commit or commit.startswith("fatal"):
            return None
        status_lines = [line for line in _git("status", "--porcelain").splitlines() if line.strip()]
        diff_bytes = subprocess.run(["git", "diff", "HEAD", "--no-color"], capture_output=True, timeout=15).stdout
        return {
            "commit": commit,
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD") or "detached",
            "dirty": bool(status_lines),
            "dirty_files": status_lines[:30],
            "dirty_files_count": len(status_lines),
            "diff_sha256": hashlib.sha256(diff_bytes).hexdigest() if diff_bytes else "",
        }
    except Exception:
        return None


def _network_dimension_config(env_cfg, agent, observation_spaces: dict, action_spaces: dict) -> dict:
    per_agent_obs = {aid: int(space.shape[0]) for aid, space in observation_spaces.items()}
    per_agent_action = {
        aid: (int(space.n) if isinstance(space, gym.spaces.Discrete) else int(space.shape[0]))
        for aid, space in action_spaces.items()
    }
    state_dim = int(getattr(agent, "_state_dim", env_cfg.state_space))
    _algo = str(getattr(env_cfg, "rl_algo", ""))
    if _algo in ("flattd3", "flatppo") or getattr(env_cfg, "agent_layout", "hier") == "flat":
        actor_params = {
            t: sum(p.numel() for p in agent.actors[t].parameters()) for t in getattr(agent, "_team_members", {})
        }
        joint_dim = int(getattr(agent, "_total_act_dim", 0))
        dims = {
            "layout": "flat",
            "flat_obs": str(getattr(env_cfg, "flat_obs_mode", getattr(env_cfg, "flat_obs", "pool"))),
            "actor": {"per_team_obs_dim": dict(per_agent_obs), "action_dim": 4, "per_team_params": actor_params},
            "team_critic": {
                "state_dim": state_dim,
                "joint_ctrl_dim": joint_dim,
                "input_dim": state_dim + joint_dim,
                "twin": bool(getattr(agent, "_matd3_twin", False)),
                "kind": "twin Q (flattd3)" if _algo == "flattd3" else "V (flatppo)",
            },
            "per_agent_obs_dim": per_agent_obs,
            "per_agent_action_dim": per_agent_action,
        }
        return dims
    if not getattr(env_cfg, "hier", False):
        actor_obs = int(next(iter(per_agent_obs.values()))) if per_agent_obs else int(env_cfg.obs_dim)
        actor_action = int(next(iter(per_agent_action.values()))) if per_agent_action else int(env_cfg.act_dim)
        return {
            "actor": {"obs_dim": actor_obs, "action_dim": actor_action},
            "critic": {"state_dim": state_dim},
            "per_agent_obs_dim": per_agent_obs,
            "per_agent_action_dim": per_agent_action,
        }

    dec_aids = [aid for aid in per_agent_obs if aid.endswith("/dec")]
    ctrl_aids = [aid for aid in per_agent_obs if aid.endswith("/ctrl")]
    upper_obs = int(per_agent_obs[dec_aids[0]]) if dec_aids else None
    lower_obs = int(per_agent_obs[ctrl_aids[0]]) if ctrl_aids else None
    upper_action = int(per_agent_action[dec_aids[0]]) if dec_aids else None
    lower_action = int(per_agent_action[ctrl_aids[0]]) if ctrl_aids else None
    critic_action_dim = int(getattr(agent, "_critic_act_dim", getattr(agent, "_total_act_dim", 0)))
    full_joint_action_dim = int(getattr(agent, "_total_act_dim", 0))

    return {
        "upper_actor_dec": {
            "obs_dim": upper_obs,
            "env_action_dim": upper_action,
            "critic_action_repr_dim": int(getattr(agent, "_dec_repr_dim", upper_action or 0))
            if upper_action is not None
            else None,
            "action_representation": "target_onehot",
            "action_type": "Discrete execution index / configurable critic representation",
        },
        "lower_actor_ctrl": {
            "obs_dim": lower_obs,
            "action_dim": lower_action,
            "action_type": "Box",
        },
        "team_critic": {
            "state_dim": state_dim,
            "joint_action_dim": critic_action_dim,
            "input_dim": state_dim + critic_action_dim,
            "note": "decision-only joint action for dual Q; full joint action for single-layer Q",
        },
        "executed_joint_action": {
            "dim": full_joint_action_dim,
            "note": "all drones: target one-hot(P) + executed control(4)",
        },
        "q_low": {
            "enabled": bool(getattr(agent, "_hier_dual", False)),
            "obs_dim": lower_obs,
            "action_dim": int(getattr(agent, "_q_low_act_dim", lower_action)) if lower_action is not None else None,
            "extra_dim": int(getattr(agent, "_q_low_extra_dim", 0)),
            "input_mode": str(getattr(agent, "_q_low_input", "none")),
            "input_dim": (lower_obs + int(getattr(agent, "_q_low_act_dim", lower_action)))
            if lower_obs is not None and lower_action is not None
            else None,
        },
        "per_agent_obs_dim": per_agent_obs,
        "per_agent_action_dim": per_agent_action,
    }


def _apply_frozen_agents(agent, frozen_agents: set[str]) -> None:
    if not frozen_agents:
        return

    def _freeze_module(module) -> None:
        if module is None:
            return
        for param in module.parameters():
            param.requires_grad = False

    for attr in ("actors", "policies"):
        table = getattr(agent, attr, None)
        if not isinstance(table, dict):
            continue
        for uid in sorted(frozen_agents):
            _freeze_module(table.get(uid))
    print(f"[TAD] frozen_agents={sorted(frozen_agents)}（actor 固定；critic/value 仍训练）")


def main():
    """Configure the environment and run multi-agent training."""

    env_cfg, rl_cfg, train_set_cfg, wandb_cfg, norm_cfg = load_env_config(args_cli.config)
    env_type = getattr(env_cfg, "env_type", "tad_hierarchical")
    gym_id = getattr(env_cfg, "gym_id", "Template-MTAD-v0")
    log_subdir = getattr(env_cfg, "log_subdir", "tad")

    drone_cfg = None
    if hasattr(env_cfg, "apply_drone_config"):
        drone_cfg = load_drone_config(args_cli.drone_config)
        env_cfg.apply_drone_config(drone_cfg)
    algo = args_cli.algo or rl_cfg.get("algo", "mappo")

    if args_cli.flat_obs is not None:
        env_cfg.flat_obs = str(args_cli.flat_obs)
    algo = env_cfg.apply_rl_algo(algo)
    rl_cfg = merge_rl_overrides(rl_cfg, algo=algo, max_iterations=args_cli.max_iterations)

    rl_cfg["flat_obs"] = str(getattr(env_cfg, "flat_obs_mode", getattr(env_cfg, "flat_obs", "pool")))

    if algo == "mappo" and float(getattr(env_cfg, "frozen_opponent_prob", 0.0)) > 0.0:
        raise ValueError(
            f"algo=mappo 与 scenario.frozen_opponent_prob={env_cfg.frozen_opponent_prob} 不兼容："
            "frozen 是整回合脚本接管，env 执行的是脚本动作，而 MAPPO memory 记录的是策略采样动作+logp，"
            "记录≠执行会让 advantage / critic 回归目标错配。\n"
            "修复：把 configs 里 scenario.frozen_opponent_prob 设为 0.0（对手多样性改用 "
            "scenario.expert_curriculum_enable/prob），或改用 --algo maddpg。"
        )

    if args_cli.timesteps is not None:
        rl_cfg["timesteps"] = int(args_cli.timesteps)
    if args_cli.learning_starts is not None:
        rl_cfg["learning_starts"] = int(args_cli.learning_starts)

    counts_changed = False
    if args_cli.n_T is not None:
        env_cfg.n_T = int(args_cli.n_T)
        counts_changed = True
    if args_cli.n_A is not None:
        env_cfg.n_A = int(args_cli.n_A)
        counts_changed = True
    if args_cli.n_D is not None:
        env_cfg.n_D = int(args_cli.n_D)
        counts_changed = True
    if counts_changed:
        env_cfg._build_agent_spec()

    if args_cli.train_side is not None:
        env_cfg.train_side = args_cli.train_side
        env_cfg._build_agent_spec()

    if args_cli.train_layer is not None and hasattr(env_cfg, "train_layer"):
        env_cfg.train_layer = args_cli.train_layer
        env_cfg._build_agent_spec()

    if args_cli.train_takeover is not None:
        env_cfg.train_takeover = args_cli.train_takeover
    if args_cli.train_takeover_step is not None:
        env_cfg.train_takeover_step = int(args_cli.train_takeover_step)
    validate_train_takeover(env_cfg)

    if args_cli.exper_rate is not None:
        _parts = [p for p in str(args_cli.exper_rate).replace("[", "").replace("]", "").split(",") if p.strip()]
        env_cfg.exper_rate = float(_parts[0]) if len(_parts) == 1 else [float(p) for p in _parts]
    validate_exper_rate(env_cfg)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.train_num_envs = int(args_cli.num_envs)

    if args_cli.test_num_envs is not None:
        env_cfg.test_num_envs = int(args_cli.test_num_envs)
        if env_cfg.test_num_envs < 0:
            raise ValueError(f"--test_num_envs={args_cli.test_num_envs} 非法，需 >=0")
    if args_cli.test_every is not None:
        env_cfg.test_every_steps = max(1, int(args_cli.test_every))
    if args_cli.test_episodes is not None:
        env_cfg.test_episodes = max(1, int(args_cli.test_episodes))
    if args_cli.test_actor is not None:
        rl_cfg["test_actor"] = args_cli.test_actor
    if args_cli.test_exper is not None:
        env_cfg.test_exper = args_cli.test_exper.lower() == "true"

        if env_cfg.test_exper and env_cfg.test_num_envs <= 0:
            print("[TAD][cfg][WARN] --test_exper=true 但 test_num_envs=0 → test_exper 不生效（忽略）")
            env_cfg.test_exper = False
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    seed = args_cli.seed if args_cli.seed is not None else int(rl_cfg.get("seed", 42))
    env_cfg.seed = seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if getattr(env_cfg, "hier", False):
        _dec_ids = [a for a in env_cfg.possible_agents if a.endswith("/dec")]
        _ctrl_ids = [a for a in env_cfg.possible_agents if a.endswith("/ctrl")]
        _upper_obs = env_cfg.obs_dim_per_agent[_dec_ids[0]] if _dec_ids else None
        _lower_obs = env_cfg.obs_dim_per_agent[_ctrl_ids[0]] if _ctrl_ids else None
        print(
            f"[TAD] env_type={env_type}  env_id={gym_id}  algo={algo}  num_envs={env_cfg.scene.num_envs}  "
            f"agents={env_cfg.possible_agents}  upper_actor_obs={_upper_obs}  "
            f"lower_actor_obs={_lower_obs}  critic_state={env_cfg.state_space}"
        )
    else:
        print(
            f"[TAD] env_type={env_type}  env_id={gym_id}  algo={algo}  num_envs={env_cfg.scene.num_envs}  "
            f"agents={env_cfg.possible_agents}  actor_obs={env_cfg.obs_dim}  critic_state={env_cfg.state_space}"
        )

    budget = None
    if algo != "random":
        resolved_max_iterations = (
            int(args_cli.max_iterations)
            if args_cli.max_iterations is not None
            else int(rl_cfg.get("max_iterations", 0) or 0)
        )
        budget = resolve_training_budget(
            env_type=env_type,
            requested_timesteps=int(rl_cfg.get("timesteps", 48000)),
            decision_period=int(getattr(env_cfg, "decision_period", 1)),
            algo=algo,
            rollouts=int(rl_cfg.get("rollouts", 1)),
            max_iterations=resolved_max_iterations,
            cli_timesteps_explicit=args_cli.timesteps is not None,
        )
        if budget.is_decision_env:
            rounded = (
                f"（向上取整 +{budget.rounded_control_steps} control steps）" if budget.rounded_control_steps else ""
            )
            print(
                f"[TAD] 训练预算：requested={budget.requested_control_steps} control steps → "
                f"{budget.additional_decisions} decisions → effective={budget.effective_control_steps} "
                f"control steps {rounded}"
            )

    patch_isaaclab_multigent_state()

    _test_num = int(getattr(env_cfg, "test_num_envs", 0) or 0)
    if _test_num > 0:
        _train_num = int(env_cfg.train_num_envs)
        if _test_num >= _train_num:
            raise ValueError(f"test_num_envs={_test_num} 应远小于训练 env 数（当前 {_train_num}）")
        env_cfg.scene.num_envs = _train_num + _test_num
        print(f"[TAD][test-env] scene.num_envs = {_train_num}(train) + {_test_num}(test) = {env_cfg.scene.num_envs}")
    env = gym.make(gym_id, cfg=env_cfg)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    possible_agents = env_cfg.possible_agents

    if getattr(env_cfg, "hier", False):
        team_of = {aid: aid.split("/")[0].split("_")[0] for aid in possible_agents}
    else:
        team_of = {uid: uid.split("_")[0] for uid in possible_agents}

    yaml_freeze_agents = _parse_agent_list(train_set_cfg.get("freeze_agents"), option_name="train_set.freeze_agents")
    cli_freeze_agents = _parse_agent_list(args_cli.freeze_agents, option_name="--freeze_agents")
    freeze_agents = cli_freeze_agents if args_cli.freeze_agents is not None else yaml_freeze_agents
    unknown_frozen = set(freeze_agents) - set(possible_agents)
    if unknown_frozen:
        raise ValueError(f"freeze_agents 含未知 agent {sorted(unknown_frozen)}；合法 agent={possible_agents}")

    cli_per_uid = parse_per_uid_cli(args_cli.per_uid_checkpoint)
    cli_per_uid_source = parse_per_uid_cli(args_cli.per_uid_source)
    yaml_per_uid_source = train_set_cfg.get("per_uid_source", {}) or {}
    per_uid_source = {**yaml_per_uid_source, **cli_per_uid_source}

    cli_per_uid = expand_team_prefix(cli_per_uid, possible_agents)
    yaml_per_uid_expanded = expand_team_prefix(train_set_cfg.get("per_uid_checkpoint", {}) or {}, possible_agents)
    per_uid_source = expand_team_prefix(per_uid_source, possible_agents)
    uid_sources_raw = resolve_resume_sources(
        possible_agents,
        cli_per_uid=cli_per_uid,
        yaml_per_uid=yaml_per_uid_expanded,
        cli_checkpoint=args_cli.checkpoint,
        yaml_resume=str(train_set_cfg.get("resume_checkpoint", "") or ""),
    )
    uid_sources: dict[str, str | None] = {}
    for _uid, _raw in uid_sources_raw.items():
        if _raw:
            _cand = os.path.abspath(os.path.expandvars(os.path.expanduser(str(_raw).strip())))
            if os.path.isfile(_cand):
                uid_sources[_uid] = _cand
            else:
                print(f"[TAD] resume checkpoint 不存在，{_uid} 按全新训练处理：{_cand}")
                uid_sources[_uid] = None
        else:
            uid_sources[_uid] = None
    resume_path = next((p for p in uid_sources.values() if p), None)
    resume_critic = (
        args_cli.resume_critic if args_cli.resume_critic is not None else bool(train_set_cfg.get("resume_critic", True))
    )
    has_resume = any(p is not None for p in uid_sources.values())
    if has_resume:
        print(f"[TAD] 续训来源：{ {u: p for u, p in uid_sources.items() if p} } | resume_critic={resume_critic}")

    state_space = gym.spaces.Box(-np.inf, np.inf, (env_cfg.state_space,), dtype=np.float32)

    observation_spaces = {
        aid: gym.spaces.Box(-np.inf, np.inf, (env_cfg.obs_dim_per_agent[aid],), dtype=np.float32)
        for aid in possible_agents
    }
    action_spaces = {}
    for aid in possible_agents:
        if aid.endswith("/dec"):
            action_spaces[aid] = gym.spaces.Discrete(int(env_cfg.n_dec))
        else:
            action_spaces[aid] = gym.spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)

    if args_cli.log_dir:
        log_dir = args_cli.log_dir
    else:
        log_dir = os.path.join("logs", log_subdir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algo}")
    os.makedirs(log_dir, exist_ok=True)
    drone_yaml = None
    if drone_cfg is not None:
        drone_yaml = save_run_drone_config(drone_cfg, log_dir)
        print(f"[TAD] 无人机配置已保存：{drone_yaml}")
    raw_write_interval = wandb_cfg.get("write_interval", "auto")
    write_interval = (
        resolve_periodic_interval(raw_write_interval, budget=budget, divisor=100)
        if budget is not None
        else raw_write_interval
    )
    checkpoint_interval = resolve_periodic_interval("auto", budget=budget, divisor=10) if budget is not None else "auto"
    experiment = {
        "directory": log_dir,
        "experiment_name": rl_cfg.get("experiment_name", "") or "",
        "write_interval": write_interval,
        "checkpoint_interval": checkpoint_interval,
    }
    if budget is not None and budget.is_decision_env:
        print(
            f"[TAD] 写出间隔：requested={raw_write_interval} control steps → "
            f"effective={write_interval} decisions ({write_interval * budget.decision_period} controls)"
        )

    if algo == "random":
        run_yaml = save_run_config(
            args_cli.config,
            log_dir,
            env_cfg,
            rl_cfg,
            seed,
            sys.argv,
            device=env_cfg.sim.device,
            train_set_cfg=train_set_cfg,
        )
        print(f"[TAD] 运行配置已保存：{run_yaml}")
        return run_random_smoke(env, possible_agents, env_cfg, args_cli.num_steps)

    if getattr(env_cfg, "hier", False):
        agent = build_hierarchical_agent(
            algo,
            env,
            rl_cfg,
            env_cfg,
            observation_spaces,
            action_spaces,
            state_space,
            env.num_envs,
            env.device,
            possible_agents,
            team_of,
            experiment,
            norm_cfg,
        )
    elif algo in ("flattd3", "flatppo"):
        agent = build_flat_agent(
            algo,
            env,
            rl_cfg,
            env_cfg,
            observation_spaces,
            action_spaces,
            state_space,
            env.num_envs,
            env.device,
            possible_agents,
            team_of,
            experiment,
            norm_cfg,
        )
    else:
        raise ValueError("仓库现仅支持 tad_hierarchical 环境（需 env_cfg.hier=True 或 algo=flat_*）；非分层环境已移除")
    agent.frozen_agents = set(freeze_agents)

    agent._roll_period = int(getattr(env_cfg, "roll_period", 512))
    agent._roll_order = list(getattr(env_cfg, "roll_order", ["T", "A", "D"]) or ["T", "A", "D"])
    agent._roll_freeze_depth = str(getattr(env_cfg, "roll_freeze_depth", "full")).lower()
    agent._roll_full_frozen_teams: set = set()
    agent._roll_active_team = None
    agent._roll_origin = 0
    agent._roll_static_frozen = set(freeze_agents)

    _er = list(getattr(env_cfg, "exper_rate", None) or [0.0, 0.0, 0.0])
    agent._roll_exper_rate = {t: float(_er[i]) for i, t in enumerate(("T", "A", "D"))}
    agent._roll_exper = any(v > 0.0 for v in agent._roll_exper_rate.values())

    agent._train_side_mode = canonical_train_mode(getattr(env_cfg, "train_side", "all"))
    _tk = canonical_train_mode(getattr(env_cfg, "train_takeover", "false"))
    agent._takeover_mode = None if _tk == "false" else _tk
    agent._takeover_step = int(getattr(env_cfg, "train_takeover_step", 0) or 0)
    agent._takeover_applied = None
    agent._takeover_current_mode = None

    agent._warmup_env = env
    agent._warmup_steps = int(getattr(env_cfg, "hover_warmup_steps", 0))

    if getattr(env_cfg, "hier", False) or algo in ("flattd3", "flatppo"):
        agent._hier_env = env

    agent._eval_env_mask = env.test_env_mask()
    agent._test_actor = str(rl_cfg.get("test_actor", "target")).lower()
    if agent._eval_env_mask is not None:
        print(
            f"[TAD][test-env] 已注入 _eval_env_mask（{int(agent._eval_env_mask.sum())} 行 test）"
            f" test_actor={agent._test_actor}"
        )

    train_set_for_save = dict(train_set_cfg)
    train_set_for_save["resume_checkpoint"] = ""

    train_set_for_save["per_uid_checkpoint"] = {}
    train_set_for_save["per_uid_source"] = {}
    train_set_for_save["freeze_agents"] = list(freeze_agents)
    train_set_for_save["_resume_sources_used"] = {uid: (p or "(fresh)") for uid, p in uid_sources.items()}
    cfg_dict = build_run_config_dict(
        args_cli.config,
        env_cfg,
        rl_cfg,
        seed,
        device=env_cfg.sim.device,
        training_budget=budget.as_dict() if budget else None,
        train_set_cfg=train_set_for_save,
    )

    network_dims = _network_dimension_config(env_cfg, agent, observation_spaces, action_spaces)
    print(f"[TAD] network_dimensions={network_dims}")
    writer = None
    use_wandb = wandb_cfg.get("enabled", True) and (not args_cli.no_wandb)
    wandb_run_id = wandb_run_name = None
    if use_wandb:
        try:
            import wandb

            log_sys = bool(wandb_cfg.get("log_system_metrics", False))
            settings = wandb.Settings(x_disable_stats=not log_sys)

            run_id = os.path.basename(log_dir)

            _dev = str(args_cli.device or env.device)
            try:
                _dev_tag = torch.cuda.get_device_name(0).split()[-1] if _dev.startswith("cuda") else "cpu"
            except Exception:
                _dev_tag = _dev.split(":")[0]
            _mode = str(getattr(env_cfg, "elimination_mode", "paired"))
            _scen_tag = f"{env_cfg.n_T}{env_cfg.n_A}{env_cfg.n_D}" + {
                "paired": "paired",
                "catcher_survives": "survive",
            }.get(_mode, _mode)
            _tags = list(dict.fromkeys(list(wandb_cfg.get("tags", [])) + [_dev_tag, _scen_tag]))
            print(f"[TAD] wandb tags：{_tags}")
            wandb.init(
                project=wandb_cfg.get("project", "MTAD-TAD"),
                entity=wandb_cfg.get("entity"),
                id=run_id,
                name=wandb_cfg.get("run_name") or run_id,
                resume="allow",
                config={
                    "algo": algo,
                    "num_envs": env.num_envs,
                    "n_T": env_cfg.n_T,
                    "n_A": env_cfg.n_A,
                    "n_D": env_cfg.n_D,
                    "network_dimensions": network_dims,
                    "train_side": getattr(env_cfg, "train_side", None),
                    "rl": rl_cfg,
                    "training_budget": budget.as_dict(),
                    "reward": cfg_dict.get("reward") or {},
                    "scenario": cfg_dict.get("scenario") or {},
                    "normalization": cfg_dict.get("normalization") or {},
                    "train_set": cfg_dict.get("train_set") or {},
                    "launch_command": " ".join(sys.argv),
                },
                mode=wandb_cfg.get("mode", "online"),
                tags=_tags,
                dir=os.path.abspath("logs"),
                settings=settings,
            )
            wandb_run_id = wandb.run.id
            wandb_run_name = wandb.run.name
            writer = WandbWriter()

            git_state = _git_state()
            if git_state is not None:
                try:
                    prev = dict(wandb.run.config.get("git") or {})
                    if not prev:
                        wandb.run.config.update({"git": git_state})
                    elif prev.get("commit") == git_state["commit"] and prev.get("dirty") == git_state["dirty"]:
                        pass
                    else:
                        hist = list(wandb.run.config.get("git_resume") or [])
                        hist.append(git_state)
                        wandb.run.config.update({"git_resume": hist})
                        print(
                            f"[TAD] 续训时代码已变（{str(prev.get('commit', '?'))[:7]} → "
                            f"{git_state['commit'][:7]}），原 git 记录保留，变更记入 git_resume"
                        )
                    print(
                        f"[TAD] git 溯源：{git_state['commit'][:12]}"
                        f"（{'dirty：' + str(git_state['dirty_files_count']) + ' 个文件' if git_state['dirty'] else 'clean'}）"
                    )
                except Exception as git_exc:
                    print(f"[TAD] git 溯源写入 wandb 失败（不影响训练）：{git_exc}")
        except Exception as e:
            print(f"[TAD] wandb 初始化失败，跳过 wandb：{e}")
            writer = None

    run_yaml = save_run_config(
        args_cli.config,
        log_dir,
        env_cfg,
        rl_cfg,
        seed,
        sys.argv,
        device=env_cfg.sim.device,
        wandb_run_id=wandb_run_id,
        wandb_run_name=wandb_run_name,
        training_budget=budget.as_dict(),
        train_set_cfg=train_set_for_save,
    )
    print(f"[TAD] 运行配置已保存：{run_yaml}")

    if writer is not None:
        try:
            import wandb

            for f in (run_yaml, drone_yaml):
                if f is not None:
                    wandb.save(os.path.abspath(f), base_path=os.path.dirname(f))
        except Exception as e:
            print(f"[TAD] wandb 上传配置失败（忽略）：{e}")

    trainer_cfg = {
        "timesteps": budget.additional_decisions,
        "environment_info": "log",
        "headless": args_cli.headless,
    }

    rl_cfg["training_budget"] = budget.as_dict()
    trainer = SequentialTrainer(env=env, agents=agent, cfg=trainer_cfg)

    completed_decisions = 0
    completed_controls = 0
    saved_period = budget.decision_period
    if has_resume:
        uid_payloads = load_and_validate_per_uid_sources(
            uid_sources, possible_agents, algo, agent, resume_critic=resume_critic, source_uids=per_uid_source
        )

        progress = resolve_multi_source_progress(uid_sources, uid_payloads, possible_agents)
        completed_decisions = progress.completed_decisions
        completed_controls = progress.completed_controls
        saved_period = progress.decision_period

        merged_payload = {uid: uid_payloads.get(uid, {}) for uid in possible_agents}
        loaded = load_training_checkpoint(agent, merged_payload, algo=algo, resume_critic=resume_critic)
        sources_summary = ", ".join(f"{u}={p}" for u, p in uid_sources.items() if p) or "(none)"
        print(
            f"[TAD] 已加载 checkpoint：{sources_summary} | completed_controls={completed_controls} | "
            f"completed_decisions={completed_decisions} | resume_critic={resume_critic} | legacy={progress.legacy}"
        )
        if saved_period != budget.decision_period:
            print(
                f"[TAD] 续训 decision_period 变化：历史={saved_period}，当前={budget.decision_period}；"
                "历史 controls 保持不变，新 decisions 按当前 period 累加。"
            )
        if resume_critic and not loaded["critic_complete"]:
            print(f"[TAD] checkpoint 缺少完整 critic 训练状态，critic 按新初始化继续：{loaded['loaded']}")
        elif not resume_critic:
            print(
                "[TAD] resume_critic=false：actor 续训；critic/value、对应 optimizer 与 state/value scaler 从头训练。"
            )
        if algo in ("maddpg", "matd3", "flattd3"):
            refill = int(rl_cfg.get("learning_starts", 0))
            agent._learning_starts = completed_decisions + refill
            print(
                f"[TAD] {algo.upper()} replay 不写 checkpoint；将在绝对 decision={agent._learning_starts} 后恢复更新"
                f"（重新填充 {refill} decision transitions）。"
            )
    _apply_frozen_agents(agent, set(freeze_agents))

    init_train_mode(agent, completed_controls)

    install_timestep_adapter(
        agent,
        completed_decisions=completed_decisions,
        completed_controls=completed_controls,
        additional_decisions=budget.additional_decisions,
        decision_period=budget.decision_period,
    )

    selected_writer = writer if writer is not None else getattr(agent, "writer", None)
    if selected_writer is not None:
        wrap_writer_steps(agent, selected_writer)

    env_cfg.print_config_summary(rl_cfg, algo, trainer_cfg, use_wandb, log_dir)
    start = time.time()
    trainer.train()
    total_decisions = completed_decisions + budget.additional_decisions
    total_controls = completed_controls + budget.effective_control_steps

    final_ckpt_dir = os.path.join(agent.experiment_dir, "checkpoints")
    os.makedirs(final_ckpt_dir, exist_ok=True)
    final_ckpt = os.path.join(final_ckpt_dir, f"agent_{total_controls}.pt")
    agent.training_progress.set_progress(
        decisions=total_decisions,
        controls=total_controls,
        decision_period=budget.decision_period,
    )
    agent.save(final_ckpt)
    actual_final_ckpt = final_ckpt
    print(
        f"[TAD] 训练完成，用时 {time.time() - start:.1f}s | "
        f"completed_controls={total_controls} | completed_decisions={total_decisions} | "
        f"final checkpoint={actual_final_ckpt}"
    )

    if writer is not None:
        writer.flush()
    try:
        import wandb

        wandb.finish()
    except Exception:
        pass
    env.close()


def run_random_smoke(env, possible_agents, env_cfg, num_steps):
    """Run a short random-policy environment check."""
    print("[TAD][random] 开始随机策略冒烟测试...")
    obs, infos = env.reset()
    n_envs = env.num_envs
    device = env.device

    for uid in possible_agents:
        print(f"  obs[{uid}] shape = {tuple(obs[uid].shape)}")

    sum_r = {"T": 0.0, "A": 0.0, "D": 0.0}
    attacker_wins = 0.0
    defender_wins = 0.0
    resets = 0
    for step in range(num_steps):
        nm = int(env_cfg.n_dec)
        actions = {}
        for aid in possible_agents:
            if aid.endswith("/dec"):
                actions[aid] = torch.randint(0, max(nm, 1), (n_envs, 1), device=device).long()
            else:
                actions[aid] = torch.zeros(n_envs, 4, device=device).uniform_(-1.0, 1.0)
        next_obs, rewards, terminated, truncated, infos = env.step(actions)

        for team in sum_r:
            for uid in possible_agents:
                if uid.startswith(team + "_"):
                    sum_r[team] += float(rewards[uid].sum())

        log = infos.get("log", {}) if isinstance(infos, dict) else {}
        if "Episode/attacker_win" in log:
            attacker_wins += float(log["Episode/attacker_win"])
            defender_wins += float(log["Episode/defender_win"])
            resets += 1
        obs = next_obs
        if step % 10 == 0 or step == num_steps - 1:
            print(
                f"  step {step}: rew_T={sum_r['T']:.1f} rew_A={sum_r['A']:.1f} rew_D={sum_r['D']:.1f} resets={resets}"
            )
    print(
        f"[TAD][random] 冒烟完成。回合数={resets}，攻击方胜率≈{attacker_wins / max(resets, 1):.3f}，"
        f"防御方胜率≈{defender_wins / max(resets, 1):.3f}"
    )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
