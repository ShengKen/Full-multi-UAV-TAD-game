# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause


import argparse
import math
import os
import sys

sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "source", "MTAD")))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="TAD-Hierarchical v4 分层环境推理/可视化。")
parser.add_argument(
    "--config",
    type=str,
    default="configs/tad_hierarchical_config.yaml",
    help="基础配置；默认优先用 checkpoint 同次训练保存的 run_config.yaml。",
)
parser.add_argument("--no_run_config", action="store_true", default=False)
parser.add_argument("--drone_config", type=str, default="configs/drone_config.yaml")
parser.add_argument(
    "--algo", type=str, default=None, choices=["mappo", "maddpg", "matd3", "matd3ppo", "flattd3", "flatppo"]
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="全局兜底 ckpt（遗留：单分层 ckpt 按 config train_side×train_layer 加载；或按层模式下作为未指定层的回退源）。",
)
parser.add_argument(
    "--dec_checkpoint",
    type=str,
    default=None,
    help="上层 /dec 决策 ckpt：广播到全机所有 /dec；未给的机→最近存活启发式。",
)
parser.add_argument(
    "--ctrl_checkpoint",
    type=str,
    default=None,
    help="下层 /ctrl 控制 ckpt：广播到全机所有 /ctrl；未给的机→Pursuit 级联启发式。",
)
parser.add_argument(
    "--per_uid_checkpoint",
    type=str,
    default=None,
    help='按 uid/层指定源（优先于 dec/ctrl 广播）："A_0/dec=path,A_0/ctrl=path,..."。',
)
parser.add_argument(
    "--per_uid_source",
    type=str,
    default=None,
    help='多 uid ckpt 内选源网络："D_0/ctrl=A_0,..."（如 D 复用 pursuit 追方网络）。',
)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--gpu_max_rigid_patch_count",
    type=int,
    default=None,
    help="仅覆盖本次 play 的 PhysX GPU 接触 patch 缓冲容量；大规模 env 评测时使用。",
)
parser.add_argument("--video_path", type=str, default=None)
parser.add_argument("--video_fps", type=int, default=30)
parser.add_argument("--no_video", action="store_true", default=False)
parser.add_argument(
    "--record_episodes", type=int, default=3, help="play 开始后录制前 N 局；默认 3。--no_video 时忽略。"
)
parser.add_argument("--cam_eye", type=str, default="24.0,-24.0,18.0")
parser.add_argument("--cam_lookat", type=str, default="0.0,0.0,3.0")
parser.add_argument("--no_wandb", action="store_true", default=False)
parser.add_argument("--max_duration", type=float, default=None)
parser.add_argument("--max_steps", type=int, default=0, help="控制步预算；0 表示最多一百万步。")
parser.add_argument("--eval_episodes", type=int, default=0, help="达到此全环境完整回合数后退出；0=不限。")
parser.add_argument("--max_episodes", type=int, default=0, help="达到此 env0 完整回合数后退出；0=不限。")
parser.add_argument("--model_sides", type=str, default=None, help="覆盖 play_set.model_sides：all 或 T/A/D。")
parser.add_argument("--telemetry_dir", type=str, default=None, help="本次评测目录，保存 env0 CSV 和 summary.json。")
parser.add_argument("--eval_group", type=str, default="play")
parser.add_argument("--offset_reservation", type=str, default=None, help="批次预分配的 manifest reservation id。")
parser.add_argument("--print_interval", type=float, default=1.0)
parser.add_argument(
    "--wandb_step_offset",
    type=str,
    default="0",
    help="wandb resume 训练 run 时 play/* 的 step 偏移（多组 play 测试写同一训练 run 时，"
    "给每组一段互不重叠的 x 轴区间，避免同名曲线混叠；仅影响 wandb 的 play_step，"
    "不影响 env 步进/回合统计。0=旧行为；'auto'=接着本 run manifest 已用的最大 "
    "play_step 往后排（间隔 20k），并在 <run>/play_eval/wandb_offsets.json 追加记录",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = not args_cli.no_video
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


from datetime import datetime
import time

import gymnasium as gym
import numpy as np
import torch

from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab_rl.skrl import SkrlVecEnvWrapper

import MTAD.tasks
from MTAD.algorithms.agent_builder import build_flat_agent, build_hierarchical_agent
from MTAD.algorithms.resume_checkpoint import (
    expand_team_prefix,
    load_and_validate_per_uid_sources,
    load_training_checkpoint,
    parse_per_uid_cli,
)
from MTAD.algorithms.skrl_state_patch import patch_isaaclab_multigent_state
from MTAD.config_loader import (
    find_run_config,
    find_run_drone_config,
    load_drone_config,
    load_env_config,
)
from play_common import (
    DroneShadowLines,
    episode_recording_path,
    lighten_recording_ground,
    parse_floats,
    save_episode,
    setup_play_wandb,
)
from play_eval_support import (
    DecisionRecorder,
    atomic_json,
    find_run_root,
    reserve_offset,
    update_offset,
    upload_play_eval_records,
)

_TARGET_TEAM = {"T": "A", "A": "T", "D": "A"}


def _team_of_agent(agent_id: str) -> str:
    return agent_id.split("/")[0].split("_")[0]


def _decode_pool_target_uid(env_cfg, team: str, pool_idx: int):
    ppt = int(env_cfg._pool_per_team)
    if ppt <= 0 or pool_idx < 0:
        return None
    team_slot = pool_idx // ppt
    member_idx = pool_idx % ppt
    selectable = list(env_cfg.selectable_targets[team])
    if team_slot >= len(selectable):
        return None
    target_team = env_cfg._selectable_grid[team][team_slot]
    members = env_cfg.team_members[target_team]
    if member_idx >= len(members):
        return None
    return members[member_idx]


def main():
    """Load trained policies and record evaluation episodes."""
    _pu = parse_per_uid_cli(args_cli.per_uid_checkpoint)
    primary_ckpt = args_cli.checkpoint

    config_path = args_cli.config
    train_cfg_dir = None
    if not args_cli.no_run_config and primary_ckpt:
        run_cfg = find_run_config(primary_ckpt)
        if run_cfg is not None:
            config_path = run_cfg
            train_cfg_dir = os.path.dirname(os.path.abspath(run_cfg))
            print(f"[TAD-HIER][play] 使用训练配置：{run_cfg}")
        else:
            print(f"[TAD-HIER][play] 未找到训练配置，用基础配置：{config_path}")
    else:
        print(f"[TAD-HIER][play] 用基础配置：{config_path}")

    env_cfg, rl_cfg, _train_set, wandb_cfg, norm_cfg = load_env_config(config_path)
    if not getattr(env_cfg, "hier", False) and getattr(env_cfg, "agent_layout", "hier") != "flat":
        raise SystemExit("[TAD-HIER][play] checkpoint/run_config 不是 tad_hierarchical v4。")

    _play_set_raw = {}
    try:
        import yaml as _y_ps

        with open(config_path, "r", encoding="utf-8") as _f_ps:
            _play_set_raw = (_y_ps.safe_load(_f_ps) or {}).get("play_set") or {}
    except Exception as _exc_ps:
        print(f"[TAD-HIER][play][WARN] 读 play_set 失败（按全模型处理）: {_exc_ps}")
    model_sides = sorted({str(s).upper() for s in (_play_set_raw.get("model_sides") or [])})
    if args_cli.model_sides is not None:
        model_sides = [] if args_cli.model_sides.lower() == "all" else [args_cli.model_sides.upper()]
    _bad_sides = [s for s in model_sides if s not in ("T", "A", "D")]
    if _bad_sides:
        raise SystemExit(f"[TAD-HIER][play] play_set.model_sides 非法队名 {_bad_sides}（只支持 T|A|D）")

    if getattr(env_cfg, "curriculum_enable", False):
        env_cfg.curriculum_enable = False
        print(f"[TAD-HIER][play] 课程退火已关闭（play 用 capture_radius={env_cfg.capture_radius}m）")

    if getattr(env_cfg, "expert_curriculum_enable", False):
        env_cfg.expert_curriculum_enable = False
        print(
            f"[TAD-HIER][play] 专家课程已关闭（忽略 run_config 的 "
            f"expert_curriculum_prob={getattr(env_cfg, 'expert_curriculum_prob', 0.0)}）"
        )
    if float(getattr(env_cfg, "frozen_opponent_prob", 0.0) or 0.0) > 0.0:
        env_cfg.frozen_opponent_prob = 0.0
        print("[TAD-HIER][play] frozen 对手接管已关闭（frozen_opponent_prob→0）")

    if getattr(env_cfg, "test_num_envs", 0):
        print(f"[TAD-HIER][play] 训练期周期评估已关闭（忽略 run_config 的 test_num_envs={env_cfg.test_num_envs}）")
        env_cfg.test_num_envs = 0

    if getattr(env_cfg, "test_exper", False):
        print("[TAD-HIER][play] test_exper 已关闭（play 无测试相）")
        env_cfg.test_exper = False
    gym_id = env_cfg.gym_id
    log_subdir = env_cfg.log_subdir

    drone_cfg_path = args_cli.drone_config
    if not args_cli.no_run_config and primary_ckpt:
        run_drone = find_run_drone_config(primary_ckpt)
        if run_drone is not None:
            drone_cfg_path = run_drone
    env_cfg.apply_drone_config(load_drone_config(drone_cfg_path))

    train_set = _train_set or {}

    algo = env_cfg.apply_rl_algo(args_cli.algo or rl_cfg.get("algo", "maddpg"))
    _flat = getattr(env_cfg, "agent_layout", "hier") == "flat"
    if _flat:
        print(
            f"[TAD-HIER][play] flat 布局（algo={algo}，flat_obs={getattr(env_cfg, 'flat_obs_mode', env_cfg.flat_obs)}）"
        )

    cli_per_uid = parse_per_uid_cli(args_cli.per_uid_checkpoint)
    yaml_per_uid = train_set.get("per_uid_checkpoint", {}) or {}
    cli_per_uid_source = parse_per_uid_cli(args_cli.per_uid_source)
    yaml_per_uid_source = train_set.get("per_uid_source", {}) or {}
    per_uid_source = {**yaml_per_uid_source, **cli_per_uid_source}
    dec_ckpt, ctrl_ckpt, global_ckpt = args_cli.dec_checkpoint, args_cli.ctrl_checkpoint, args_cli.checkpoint
    per_layer_mode = bool(dec_ckpt or ctrl_ckpt or cli_per_uid or yaml_per_uid)

    def _norm_ckpt(p):
        if not p:
            return None
        c = os.path.abspath(os.path.expandvars(os.path.expanduser(str(p).strip())))
        return c if os.path.isfile(c) else None

    if per_layer_mode and _flat and (dec_ckpt or ctrl_ckpt):
        raise SystemExit(
            "[TAD-HIER][play] flat 算法无决策层，不支持 --dec_checkpoint/--ctrl_checkpoint"
            '（用 --checkpoint 或 --per_uid_checkpoint "A_0=path,..."）'
        )
    if per_layer_mode and not _flat:
        candidate_agents = [f"{u}/{L}" for u in env_cfg.all_uids for L in ("dec", "ctrl")]
    else:
        candidate_agents = list(env_cfg.possible_agents)

    cli_per_uid = expand_team_prefix(cli_per_uid, candidate_agents)
    yaml_per_uid = expand_team_prefix(yaml_per_uid, candidate_agents)
    per_uid_source = expand_team_prefix(per_uid_source, candidate_agents)
    uid_sources: dict[str, str | None] = {}
    for aid in candidate_agents:
        raw = cli_per_uid.get(aid) or yaml_per_uid.get(aid)
        if raw is None and not _flat:
            layer = aid.split("/")[1]
            raw = dec_ckpt if layer == "dec" else ctrl_ckpt
        if raw is None:
            raw = global_ckpt
        path = _norm_ckpt(raw)
        if raw and not path:
            print(f"[TAD-HIER][play] {aid} ckpt 不存在，该层改用脚本：{raw}")
        uid_sources[aid] = path
    play_agents = [aid for aid in candidate_agents if uid_sources[aid]]
    baseline_mode = not play_agents
    if baseline_mode:
        play_agents = list(env_cfg.all_uids) if _flat else [f"{u}/dec" for u in env_cfg.all_uids]
        algo = "baseline"
        print("[TAD-HIER][play] 未指定 checkpoint，启用全脚本 baseline：上层=最近存活目标，下层=Pursuit 级联控制。")

    if (not baseline_mode) and args_cli.algo is None and not _flat:
        first_path = next(uid_sources[a] for a in play_agents if uid_sources[a])
        low = first_path.lower()
        inferred = (
            "matd3ppo"
            if "matd3ppo" in low
            else "matd3"
            if "matd3" in low
            else "maddpg"
            if "maddpg" in low
            else "mappo"
            if "mappo" in low
            else None
        )
        if inferred is not None:
            algo = inferred

    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.gpu_max_rigid_patch_count is not None:
        if args_cli.gpu_max_rigid_patch_count <= 0:
            raise ValueError("gpu_max_rigid_patch_count must be positive")
        env_cfg.sim.physx.gpu_max_rigid_patch_count = args_cli.gpu_max_rigid_patch_count
        print(f"[TAD-HIER][play] PhysX GPU patch capacity={args_cli.gpu_max_rigid_patch_count}")
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    env_cfg.possible_agents = list(play_agents)
    env_cfg.obs_dim_per_agent = {}
    env_cfg.observation_spaces = {}
    env_cfg.action_spaces = {}
    if _flat:
        for aid in play_agents:
            env_cfg.obs_dim_per_agent[aid] = int(env_cfg.flat_obs_dim_per_team[aid.split("_")[0]])
            env_cfg.action_spaces[aid] = 4
            env_cfg.observation_spaces[aid] = env_cfg.obs_dim_per_agent[aid]
    else:
        for aid in play_agents:
            if aid.endswith("/dec"):
                env_cfg.obs_dim_per_agent[aid] = int(env_cfg.obs_dim)
                env_cfg.action_spaces[aid] = {int(env_cfg.n_dec)}
            else:
                _ttg = 1 if (env_cfg.obs_time_to_go and not env_cfg.obs_full_obs) else 0
                _ang = 3 if (getattr(env_cfg, "obs_ang_vel", False) and not env_cfg.obs_full_obs) else 0
                _ctrl_base = env_cfg.state_space if env_cfg.obs_full_obs else env_cfg.lower_obs_dim + _ang + _ttg
                env_cfg.obs_dim_per_agent[aid] = int(_ctrl_base)
                env_cfg.action_spaces[aid] = 4
            env_cfg.observation_spaces[aid] = env_cfg.obs_dim_per_agent[aid]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args_cli.video_path:
        play_dir = os.path.dirname(args_cli.video_path) or f"logs/{log_subdir}/play"
        stem = os.path.splitext(os.path.basename(args_cli.video_path))[0]
    elif train_cfg_dir is not None:
        play_dir = os.path.join(train_cfg_dir, "play")
        stem = "episode1"
    else:
        play_dir = f"logs/{log_subdir}/play"
        stem = "episode1"
    args_cli.video_path = os.path.join(play_dir, f"{algo}_{stem}_{ts}.mp4")

    viewer = env_cfg.viewer
    viewer.origin_type = "env"
    viewer.env_index = 0
    viewer.resolution = (1920, 1080)
    viewer.eye = parse_floats(args_cli.cam_eye)
    viewer.lookat = parse_floats(args_cli.cam_lookat)

    patch_isaaclab_multigent_state()
    env = gym.make(gym_id, cfg=env_cfg, render_mode=None if args_cli.no_video else "rgb_array")
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    possible_agents = list(env_cfg.possible_agents)
    team_of = {aid: _team_of_agent(aid) for aid in possible_agents}

    if model_sides:
        _actor_teams = set(team_of.values())
        _no_actor = [s for s in model_sides if s not in _actor_teams]
        if _no_actor:
            raise SystemExit(
                f"[TAD-HIER][play] play_set.model_sides 含无 actor 的队 {_no_actor}"
                f"（本次 play 的模型只覆盖 {sorted(_actor_teams)}）；请从 model_sides 移除或换 ckpt"
            )
    state_space = gym.spaces.Box(-np.inf, np.inf, (env_cfg.state_space,), dtype=np.float32)
    observation_spaces = {
        aid: gym.spaces.Box(-np.inf, np.inf, (env_cfg.obs_dim_per_agent[aid],), dtype=np.float32)
        for aid in possible_agents
    }
    action_spaces = {
        aid: (
            gym.spaces.Discrete(int(env_cfg.n_dec))
            if aid.endswith("/dec")
            else gym.spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)
        )
        for aid in possible_agents
    }
    experiment = {
        "directory": f"logs/{log_subdir}/play",
        "experiment_name": "",
        "write_interval": 0,
        "checkpoint_interval": 0,
    }
    agent = None
    if not baseline_mode:
        if _flat:
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
        agent._warmup_env = env
        agent._warmup_steps = int(env_cfg.hover_warmup_steps)
        agent._hier_env = env
        agent.init(trainer_cfg=None)

        if per_layer_mode:
            uid_payloads = load_and_validate_per_uid_sources(
                uid_sources, possible_agents, algo, agent, resume_critic=False, source_uids=per_uid_source
            )
            merged_payload = {aid: uid_payloads.get(aid, {}) for aid in possible_agents}
            load_training_checkpoint(agent, merged_payload, algo=algo, resume_critic=False)
            src_summary = ", ".join(f"{a}={uid_sources[a]}" for a in possible_agents if uid_sources[a])
            scripted = [a for a in candidate_agents if not uid_sources.get(a)]
            print(f"[TAD-HIER][play] 已加载（actor-only，algo={algo}）：{src_summary}")
            print(f"[TAD-HIER][play] 脚本化层（未给 ckpt → dec 追最近 / ctrl 级联）：{scripted}")
        else:
            print(f"[TAD-HIER][play] 加载 checkpoint(legacy): {global_ckpt}（algo={algo}）")

            try:
                _payload = torch.load(global_ckpt, map_location="cpu")
                load_training_checkpoint(agent, _payload, algo=algo, resume_critic=False)
                print(f"[TAD-HIER][play] legacy ckpt 已加载（actor-only，critic 跳过；play 不需要 critic）")
            except (ValueError, RuntimeError, KeyError, FileNotFoundError) as _exc:
                raise RuntimeError(
                    f"[TAD-HIER][play] legacy checkpoint 与当前配置维度不兼容，拒绝加载：{_exc}\n"
                    f"  常见原因：full_obs / include_lock_count / n_T|n_A|n_D / upper|lower_actor_hidden / 姿态表示(gravity+sin/cos yaw) 变更导致 obs/state/act 维度漂移。"
                    f"  请用 run_config.yaml 对齐配置或换匹配的 checkpoint。"
                ) from _exc
        agent.enable_training_mode(False)

    play_run_config: dict = {}
    try:
        import yaml as _yaml

        with open(config_path, "r", encoding="utf-8") as _f:
            play_run_config = _yaml.safe_load(_f) or {}
    except Exception as _exc:
        print(f"[TAD-HIER][play] 读取配置 {config_path} 为 wandb 失败：{_exc}")
    play_run_config["_play"] = {
        "algo": algo,
        "baseline_mode": bool(baseline_mode),
        "num_envs": int(args_cli.num_envs),
        "config_path": config_path,
        "train_cfg_dir": train_cfg_dir,
        "eval_group": args_cli.eval_group,
        "model_sides": model_sides,
        "physx_gpu_max_rigid_patch_count": env_cfg.sim.physx.gpu_max_rigid_patch_count,
    }

    wandb_run = setup_play_wandb(
        enabled=not args_cli.no_wandb,
        wandb_cfg=wandb_cfg,
        train_cfg_dir=train_cfg_dir,
        default_project="MTAD-Hierarchical",
        run_name=f"{algo}_hier_play_{ts}",
        tag="TAD-HIER-play",
        config=play_run_config,
    )

    base_env = env.unwrapped
    if not args_cli.no_video:
        ground_visuals = lighten_recording_ground()
        print(f"[TAD-HIER][play] 淡蓝网格录像背景：{ground_visuals}")
    if baseline_mode:
        base_env._force_scripted_policy = True

    if model_sides and not baseline_mode:
        _all_teams = [t for t in ("T", "A", "D") if base_env.cfg.team_members.get(t)]
        expert_teams = [t for t in _all_teams if t not in model_sides]
        base_env.set_exper_teams(expert_teams, per_episode_prob={team: 1.0 for team in _all_teams})
        print(f"[TAD-HIER][play] 定点评测：模型队={model_sides} 专家/脚本队={expert_teams}")
    elif model_sides and baseline_mode:
        print("[TAD-HIER][play][WARN] baseline 模式全脚本，忽略 play_set.model_sides")
    uid0 = possible_agents[0]
    ctrl_hz = 1.0 / base_env.step_dt
    stride = max(1, round(ctrl_hz / max(args_cli.video_fps, 1)))
    out_fps = max(1, round(ctrl_hz / stride))
    all_uids = list(base_env.cfg.all_uids)
    team_members = base_env.cfg.team_members
    learning_drones = set() if baseline_mode else {aid.split("/")[0] for aid in possible_agents}
    frames = []
    record_episode_limit = max(0, int(args_cli.record_episodes))
    ep_reward = {"T": 0.0, "A": 0.0, "D": 0.0}
    cum_episodes = cum_attacker = cum_defender = 0

    g_episodes = g_A_win = g_D_win = g_T_win = 0

    _n_envs = base_env.num_envs
    _dev = base_env.device
    ep_reward_env = {t: torch.zeros(_n_envs, device=_dev) for t in ("T", "A", "D")}
    recent_ep_rewards = {"T": [], "A": [], "D": []}
    _RECENT_WIN = 200
    last_ep_result = 0

    obs, _ = env.reset()
    states = env.state()
    shadow_lines = None
    if not args_cli.no_video:
        try:
            base_env.render()
            shadow_lines = DroneShadowLines(base_env)
            shadow_lines.update()
        except Exception as exc:
            print(f"[TAD-HIER][play] 渲染/投影线初始化失败：{exc}")

    _offset_root = find_run_root(primary_ckpt)
    _offset_raw = str(args_cli.wandb_step_offset).strip().lower()
    _offset_auto = _offset_raw == "auto"
    _step_budget = args_cli.max_steps or 1_000_000
    _reservation = None
    if _offset_root is not None:
        if args_cli.offset_reservation:
            import json

            with open(_offset_root / "play_eval" / "wandb_offsets.json", encoding="utf-8") as stream:
                _entries = json.load(stream)
            _reservation = next(e for e in _entries if e.get("id") == args_cli.offset_reservation)
            if _reservation["status"] != "reserved":
                raise ValueError("Offset reservation has already been consumed")
            _offset_val = _reservation["offset"]
            if _step_budget > _reservation["reserved_end"] - _offset_val:
                raise ValueError("max_steps exceeds reserved offset interval")
        else:
            _reservation = reserve_offset(
                _offset_root,
                _offset_raw,
                _step_budget,
                {
                    "checkpoint": primary_ckpt,
                    "config": config_path,
                    "model_sides": model_sides,
                    "num_envs": args_cli.num_envs,
                    "group": args_cli.eval_group,
                },
            )
            _offset_val = _reservation["offset"]
        update_offset(_offset_root, _reservation["id"], expected_status="reserved", status="running", pid=os.getpid())
    elif _offset_auto:
        raise ValueError("Auto offset requires a checkpoint with a run_config.yaml ancestor")
    else:
        _offset_val = int(_offset_raw)
    print(f"[TAD-HIER][play] wandb step offset={_offset_val}")
    _telemetry_dir = args_cli.telemetry_dir or os.path.join(play_dir, f"telemetry_{ts}_{os.getpid()}")
    decision_recorder = DecisionRecorder(base_env, _telemetry_dir)
    base_env._play_decision_observer = decision_recorder.capture
    step = ep_step = episode = 0
    play_step = _offset_val
    t0 = time.time()
    last_print = 0.0
    stop_reason = "error"
    if args_cli.no_video:
        print(f"[TAD-HIER][play] 开始统计评测（无录像）；env0 决策记录 → {_telemetry_dir}")
    else:
        print(f"[TAD-HIER][play] 开始推理；前 {record_episode_limit} 局录像 → {args_cli.video_path} @ {out_fps}fps")
    print(f"[TAD-HIER][play] agents={possible_agents}  decision_period={env_cfg.decision_period} controls")

    try:
        while True:
            if args_cli.max_duration is not None and time.time() - t0 > args_cli.max_duration:
                print(f"[TAD-HIER][play] 达 --max_duration {args_cli.max_duration:.0f}s，退出。")
                stop_reason = "max_duration"
                break
            if step >= _step_budget:
                stop_reason = "max_steps"
                break
            if baseline_mode:
                actions = {
                    aid: torch.zeros((env.num_envs, 1), dtype=torch.long, device=env.device) for aid in possible_agents
                }
            else:
                with torch.no_grad():
                    actions, _ = agent.act(obs, states, timestep=step, timesteps=max(1, round(ctrl_hz * 60)))
            obs, rewards, term, trunc, infos = env.step(actions)
            states = env.state()
            step += 1
            play_step += 1
            ep_step += 1
            _env0_done = bool(term[uid0][0]) or bool(trunc[uid0][0])
            decision_metrics = decision_recorder.record(play_step, episode, ep_step, _env0_done)

            if not _env0_done:
                if shadow_lines is not None:
                    shadow_lines.update()
                if episode < record_episode_limit and not args_cli.no_video and step % stride == 0:
                    frame = base_env.render()
                    if frame is not None and frame.size > 0 and frame.max() > 0:
                        frames.append(frame)

            engage = {}
            for uid in all_uids:
                team = uid.split("_")[0]
                pool = decision_recorder.rows[uid]["select_idx"]
                target_uid = _decode_pool_target_uid(base_env.cfg, team, pool)
                distance = (
                    float(
                        torch.norm(
                            base_env.drones[uid].data.root_pos_w[0] - base_env.drones[target_uid].data.root_pos_w[0]
                        )
                    )
                    if target_uid is not None and not _env0_done
                    else float("nan")
                )
                engage[uid] = (target_uid, distance, decision_recorder.rows[uid]["lock_count"], pool)

            for team in ep_reward:
                ep_reward[team] += sum(float(rewards[aid][0]) for aid in possible_agents if _team_of_agent(aid) == team)

            for team in ("T", "A", "D"):
                _team_rew = sum(
                    (rewards[aid].view(-1) for aid in possible_agents if _team_of_agent(aid) == team),
                    torch.zeros(_n_envs, device=_dev),
                )
                ep_reward_env[team] += _team_rew

            _done_n = term[uid0].view(-1).bool() | trunc[uid0].view(-1).bool()
            if bool(_done_n.any()):
                _oc = base_env._outcome
                _term_n = term[uid0].view(-1).bool()
                _trunc_n = trunc[uid0].view(-1).bool()
                _done_idx = _done_n.nonzero(as_tuple=False).squeeze(-1)
                g_episodes += int(_done_n.sum().item())
                g_A_win += int((_oc["attacker_win"].bool() & _term_n).sum().item())
                g_D_win += int((_oc["defender_win"].bool() & _term_n).sum().item())
                g_T_win += int((_trunc_n & ~_term_n).sum().item())

                for team in ("T", "A", "D"):
                    recent_ep_rewards[team].extend(ep_reward_env[team][_done_idx].detach().cpu().tolist())
                    ep_reward_env[team][_done_idx] = 0.0
                    recent_ep_rewards[team] = recent_ep_rewards[team][-_RECENT_WIN:]

            if wandb_run is not None:
                log = {
                    "play/play_step": play_step,
                    "play/ep_step": ep_step,
                    "play/episode": episode,
                    **decision_metrics,
                }
                d_ta, d_ad = base_env._team_distances()
                log["play/dist_TA"] = float(d_ta[0])
                log["play/dist_AD"] = float(d_ad[0])
                for aid in possible_agents:
                    if not aid.endswith("/dec"):
                        for i in range(4):
                            log[f"play/{aid}/policy_action_{i}"] = float(actions[aid][0, i])
                for uid in all_uids:
                    target_uid, distance, lock_count, pool = engage[uid]
                    if target_uid is not None:
                        log[f"play/{uid}/dist_to_target"] = distance
                    data = base_env.drones[uid].data
                    pos = data.root_pos_w[0]
                    vel = data.root_lin_vel_w[0]
                    ang = data.root_ang_vel_b[0]
                    roll, pitch, yaw = euler_xyz_from_quat(data.root_quat_w)
                    for name, value in zip(("pos_x", "pos_y", "pos_z"), pos):
                        log[f"play/{uid}/{name}"] = float(value)
                    for name, value in zip(("lin_vel_x", "lin_vel_y", "lin_vel_z"), vel):
                        log[f"play/{uid}/{name}"] = float(value)
                    log[f"play/{uid}/speed"] = float(vel.norm())
                    log[f"play/{uid}/roll"] = math.degrees(float(roll[0]))
                    log[f"play/{uid}/pitch"] = math.degrees(float(pitch[0]))
                    log[f"play/{uid}/yaw"] = math.degrees(float(yaw[0]))
                    for name, value in zip(("ang_vel_x", "ang_vel_y", "ang_vel_z"), ang):
                        log[f"play/{uid}/{name}"] = math.degrees(float(value))
                    log[f"play/{uid}/ang_speed"] = math.degrees(float(ang.norm()))
                    for i in range(4):
                        log[f"play/{uid}/action_{i}"] = float(base_env.actions[uid][0, i])
                for team in ep_reward:
                    log[f"play/reward_{team}_episode_running"] = ep_reward[team]

                log["play/g_episodes"] = g_episodes
                log["play/g_A_win"] = g_A_win
                log["play/g_D_win"] = g_D_win
                log["play/g_T_win"] = g_T_win
                if g_episodes > 0:
                    log["play/panel/A_win_rate"] = g_A_win / g_episodes
                    log["play/panel/D_win_rate"] = g_D_win / g_episodes
                    log["play/panel/T_win_rate"] = g_T_win / g_episodes

                log["play/panel/episode_result"] = last_ep_result

                for team in ("T", "A", "D"):
                    if recent_ep_rewards[team]:
                        log[f"play/panel/mean_ep_reward_{team}"] = sum(recent_ep_rewards[team]) / len(
                            recent_ep_rewards[team]
                        )
                wandb_run.log(log)

            if args_cli.print_interval > 0 and time.time() - last_print >= args_cli.print_interval:
                parts = []
                for uid in all_uids:
                    target_uid, distance, _, pool = engage[uid]
                    dead = not bool(base_env._alive[uid][0])
                    mark = "†" if dead else ("*" if uid in learning_drones else " ")
                    parts.append(f"{mark}{uid}[pool={pool}]→{target_uid}(d={distance:.1f})")
                print(
                    f"[ep{episode} t={ep_step * base_env.step_dt:5.1f}s] "
                    + "  ".join(parts)
                    + "   (*=至少一层学习  †=阵亡)"
                )
                last_print = time.time()

            if _env0_done:
                if shadow_lines is not None:
                    shadow_lines.reset()
                if episode < record_episode_limit and (not args_cli.no_video) and frames:
                    save_episode(frames, episode_recording_path(args_cli.video_path, episode), out_fps, "TAD-HIER-play")
                    frames = []
                    if episode + 1 >= record_episode_limit:
                        print(f"[TAD-HIER][play] 前 {record_episode_limit} 局录像已保存。")
                outcome = base_env._outcome
                attacker = int(bool(outcome["attacker_win"][0]))
                defender = int(bool(outcome["defender_win"][0]))
                cum_episodes += 1
                cum_attacker += attacker
                cum_defender += defender
                last_ep_result = 1 if attacker else 2 if defender else 3
                events = []
                for uid in all_uids:
                    if bool(outcome["newly_dead"][uid][0]):
                        if bool(outcome.get("f_T_captured", {}).get(uid, torch.zeros(1, dtype=torch.bool))[0]):
                            why = "被A抓"
                        elif bool(outcome.get("f_A_capture", {}).get(uid, torch.zeros(1, dtype=torch.bool))[0]):
                            why = "抓到T"
                        elif bool(outcome.get("f_A_intercepted", {}).get(uid, torch.zeros(1, dtype=torch.bool))[0]):
                            why = "被D拦"
                        elif bool(outcome.get("f_D_intercept", {}).get(uid, torch.zeros(1, dtype=torch.bool))[0]):
                            why = "拦到A"
                        else:
                            why = "淘汰"
                        events.append(f"{uid}({why})")
                result = "攻击方胜" if attacker else "防御方胜" if defender else "超时无胜负"
                print(
                    f"[TAD-HIER][play] 回合 {episode}：{ep_step * base_env.step_dt:.1f}s | {result} | "
                    f"累计 A胜={cum_attacker}/{cum_episodes} D胜={cum_defender}/{cum_episodes} | "
                    f"reward T={ep_reward['T']:.2f} A={ep_reward['A']:.2f} D={ep_reward['D']:.2f}"
                )
                if events:
                    print("                淘汰：" + " ".join(events))
                if wandb_run is not None:
                    ep_log = {
                        "play/play_step": play_step,
                        "play/episode": episode,
                        "play/episode_length_s": ep_step * base_env.step_dt,
                        "play/episode_attacker_win": attacker,
                        "play/episode_defender_win": defender,
                    }
                    for team in ep_reward:
                        ep_log[f"play/episode_reward_{team}"] = ep_reward[team]
                    wandb_run.log(ep_log)
                ep_reward = {"T": 0.0, "A": 0.0, "D": 0.0}
                ep_step = 0
                episode += 1
            if args_cli.eval_episodes and g_episodes >= args_cli.eval_episodes:
                stop_reason = "eval_episodes"
                break
            if args_cli.max_episodes and episode >= args_cli.max_episodes:
                stop_reason = "max_episodes"
                break
    except KeyboardInterrupt:
        stop_reason = "interrupted"
        print("\n[TAD-HIER][play] 收到 Ctrl+C，退出。")
    finally:
        if g_episodes > 0:
            print(f"\n[TAD-HIER][play] ===== 全局统计（num_envs={args_cli.num_envs}）=====")
            print(
                f"[TAD-HIER][play] 回合总数: {g_episodes}"
                f" | A胜 {g_A_win} ({100.0 * g_A_win / g_episodes:.1f}%)"
                f" | D胜 {g_D_win} ({100.0 * g_D_win / g_episodes:.1f}%)"
                f" | T撑到超时 {g_T_win} ({100.0 * g_T_win / g_episodes:.1f}%)"
            )
            for team in ("T", "A", "D"):
                if recent_ep_rewards[team]:
                    mean_r = sum(recent_ep_rewards[team]) / len(recent_ep_rewards[team])
                    print(f"[TAD-HIER][play] 近{len(recent_ep_rewards[team])}回合平均奖励 {team}: {mean_r:.2f}")

        if episode < record_episode_limit and not args_cli.no_video and frames:
            save_episode(frames, episode_recording_path(args_cli.video_path, episode), out_fps, "TAD-HIER-play")
        decision_recorder.close()
        summary = {
            "group": args_cli.eval_group,
            "num_envs": args_cli.num_envs,
            "model_sides": model_sides,
            "checkpoint": primary_ckpt,
            "offset": _offset_val,
            "end": play_step,
            "steps": step,
            "episodes": g_episodes,
            "env0_episodes": episode,
            "wins": {"A": g_A_win, "D": g_D_win, "T": g_T_win},
            "stop_reason": stop_reason,
            "wandb_active": wandb_run is not None,
            "elapsed_s": time.time() - t0,
            "physx_gpu_max_rigid_patch_count": env_cfg.sim.physx.gpu_max_rigid_patch_count,
        }
        atomic_json(os.path.join(_telemetry_dir, "summary.json"), summary)
        if _reservation is not None:
            update_offset(
                _offset_root,
                _reservation["id"],
                end=play_step,
                steps=step,
                status="failed" if stop_reason == "error" else "finished",
                summary=summary,
                num_envs=args_cli.num_envs,
            )
        if wandb_run is not None:
            try:
                uploaded = upload_play_eval_records(wandb_run, _telemetry_dir, _offset_root)
                print(f"[TAD-HIER][play] W&B 评测记录已登记上传（offline 模式待 sync）：{uploaded}")
            except Exception as exc:
                print(f"[TAD-HIER][play][WARN] W&B 评测记录上传失败，本地结果已保留：{exc}")
            wandb_run.finish()
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
