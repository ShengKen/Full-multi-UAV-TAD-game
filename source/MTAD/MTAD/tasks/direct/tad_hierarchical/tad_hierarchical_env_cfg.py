# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.

# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.utils import configclass

from isaaclab_assets import CRAZYFLIE_CFG

import os as _os


_LOCAL_CF2X_USD = _os.path.abspath(
    _os.path.join(_os.path.dirname(__file__), "../../../../../../assets/crazyflie/cf2x.usd")
)

_TEAM_COLORS = {
    "T": (1.0, 0.0, 0.0),
    "A": (0.0, 1.0, 0.0),
    "D": (0.0, 0.0, 1.0),
}


_LOWER_OBS_DIM = 18

_TEAM_ORDER = ("T", "A", "D")


FLAT_ALGOS = ("flattd3", "flatppo")
_FLAT_ALGO_ALIAS = {
    "flat-matd3": "flattd3",
    "flat_matd3": "flattd3",
    "flatmatd3": "flattd3",
    "flat-td3": "flattd3",
    "flat-mappo": "flatppo",
    "flat_mappo": "flatppo",
    "flatmappo": "flatppo",
    "flat-ppo": "flatppo",
    "flat_td3": "flattd3",
    "flat_ppo": "flatppo",
}


def canonical_algo(algo) -> str:
    a = str(algo).strip().lower()
    return _FLAT_ALGO_ALIAS.get(a, a)


def algo_is_flat(algo) -> bool:
    return canonical_algo(algo) in FLAT_ALGOS


def canonical_train_mode(value) -> str:
    if value is None or value is False:
        return "false"
    s = str(value).strip().lower()
    if s in ("", "false", "none"):
        return "false"
    alias = {
        "t": "T",
        "target": "T",
        "evader": "T",
        "a": "A",
        "attacker": "A",
        "pursuer": "A",
        "d": "D",
        "defender": "D",
        "all": "all",
        "both": "all",
        "roll": "roll",
    }
    if s not in alias:
        _hint = "（旧值 exper 已删除：等价写法是 roll + exper_rate=[1,1,1]）" if s == "exper" else ""
        raise ValueError(f"train 模式值 {value!r} 非法：需 false | T | A | D | all | roll{_hint}")
    return alias[s]


def normalize_exper_rate(value) -> list:
    if value is None:
        return [0.0, 0.0, 0.0]
    if isinstance(value, dict):
        raw = [value.get(t, 0.0) for t in _TEAM_ORDER]
    elif isinstance(value, (list, tuple)):
        if len(value) != len(_TEAM_ORDER):
            raise ValueError(f"exper_rate 列表需为 [r_T, r_A, r_D] 三项（按 T/A/D 顺序），收到 {value!r}")
        raw = list(value)
    elif isinstance(value, bool):
        raise ValueError(f"exper_rate={value!r} 非法：需 0~1 的数、[r_T,r_A,r_D] 或 {{T/A/D: 值}}")
    else:
        raw = [value] * len(_TEAM_ORDER)
    out = []
    for t, v in zip(_TEAM_ORDER, raw):
        try:
            r = float(v)
        except (TypeError, ValueError):
            raise ValueError(f"exper_rate[{t}]={v!r} 非法：需 0~1 的数")
        if not (0.0 <= r <= 1.0):
            raise ValueError(f"exper_rate[{t}]={r} 越界：需在 [0,1] 内")
        out.append(r)
    return out


def validate_exper_rate(cfg) -> None:
    cfg.exper_rate = normalize_exper_rate(getattr(cfg, "exper_rate", None))
    if not any(r > 0.0 for r in cfg.exper_rate):
        return
    sides = (
        canonical_train_mode(getattr(cfg, "train_side", "all")),
        canonical_train_mode(getattr(cfg, "train_takeover", "false")),
    )
    if "roll" not in sides:
        print(
            f"[TAD][cfg][WARN] exper_rate={cfg.exper_rate} 但 train_side/train_takeover 均非 roll "
            f"→ 不生效（仅 roll 模式有「非激活队」概念）"
        )


def validate_train_takeover(cfg) -> None:
    mode = canonical_train_mode(getattr(cfg, "train_takeover", "false"))
    raw_step = getattr(cfg, "train_takeover_step", 0)
    if isinstance(raw_step, bool):
        if raw_step:
            raise ValueError("train_takeover_step 需为非负整数（control step），不接受 true")
        raw_step = 0
    step = int(raw_step)
    if step < 0:
        raise ValueError(f"train_takeover_step={step} 需 >=0")
    cfg.train_takeover = mode
    cfg.train_takeover_step = step
    if mode == "false":
        return
    side = canonical_train_mode(getattr(cfg, "train_side", "all"))
    if side not in ("all", "roll"):
        raise ValueError(
            f"train_side={cfg.train_side!r} 为窄模式（仅建该队网络）时 train_takeover 必须为 false："
            f"中途拓宽/换队需重建 agent 与空间，现有单 gym.make 管线不支持。"
            f"要两阶段请用宽模式起步（all/roll）再切 {mode!r}"
        )
    if mode == "roll":
        _valid = {"T", "A", "D"}
        _bad = [t for t in (cfg.roll_order or []) if t not in _valid]
        if _bad or not cfg.roll_order:
            raise ValueError(f"train_takeover={mode} 需要 roll_order={cfg.roll_order!r} 为 T/A/D 的非空子集")
        if cfg.roll_period <= 0:
            raise ValueError(f"train_takeover={mode} 需要 roll_period={cfg.roll_period} >0")


@configclass
class TadHierarchicalEnvCfg(DirectMARLEnvCfg):
    env_type: str = "tad_hierarchical"
    gym_id: str = "Template-TadHierarchical-v0"
    log_subdir: str = "tad_hierarchical"
    is_discrete: bool = False
    hier: bool = True

    rl_algo: str = "matd3ppo"

    flat_obs: str = "pool"

    n_T: int = 2
    n_A: int = 2
    n_D: int = 2

    elimination_mode: str = "paired"

    train_side: str = "all"
    train_layer: str = "both"

    roll_period: int = 512
    roll_order: list = None
    roll_freeze_depth: str = "full"

    exper_rate: list = None

    train_takeover: str = "false"
    train_takeover_step: int = 0
    decision_period: int = 8
    scripted_decision: str = "nearest"

    q_low_input: str = "none"
    q_low_rel_dim: int = 7

    lower_critic_mode: str = "per_agent"

    selectable_targets: dict | None = None

    frozen_opponent_prob: float = 0.0

    expert_curriculum_enable: bool = False
    expert_curriculum_prob: float = 0.25

    expert_curriculum_anneal_frac: float = 0.0

    curriculum_enable: bool = False
    curriculum_capture_radius_start: float = 2.0
    curriculum_capture_radius_end: float = 0.5
    curriculum_capture_radius_step: float = 0.25
    curriculum_promotion_winrate: float = 0.8
    curriculum_promotion_metric: str = "attacker"
    curriculum_min_samples: int = 100
    curriculum_check_every: int = 2500

    test_num_envs: int = 0
    test_every_steps: int = 2500
    test_episodes: int = 16
    test_exper: bool = False

    train_num_envs: int = 0

    ppn_nav_const: float = 3.0
    evader_a_max: float = 6.0
    accel_k_att: float = 6.0
    accel_k_tilt: float = 1.0
    max_lean: float = 0.6
    heur_k_yaw: float = 2.5
    heur_k_pitch: float = 0.6
    heur_k_roll: float = 0.6
    heur_max_lean: float = 0.4
    heur_k_att: float = 6.0
    heur_k_alt: float = 0.4
    heur_k_dz: float = 0.2
    heur_noise_std: float = 0.0

    evader_wall_margin: float = 4.0
    evader_inward_beta: float = 0.5
    evader_tangent_min: float = 0.2

    evader_k_z: float = 1.0
    evader_z_margin: float = 1.0

    possible_agents: list = None
    all_uids: list = None
    action_spaces: dict = None
    observation_spaces: dict = None
    state_space: int = None

    critic_state: str = "token"

    decimation: int = 2
    episode_length_s: float = 15.0
    episode_length_jitter_s: float = 0.0

    win_rate_horizons: list[int] = None

    telemetry_interval: int = 50
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physx=PhysxCfg(
            min_velocity_iteration_count=1,
            enable_external_forces_every_iteration=True,
            gpu_max_rigid_patch_count=2**20,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024,
        env_spacing=6.0,
        replicate_physics=True,
    )

    drone_scale: float = 5.0
    thrust_to_weight: float = 1.9
    moment_max: float = 0.01
    action_mode: str = "rates"
    rate_max: float = 5.0
    rate_kp: float = 0.05
    max_roll: float = 0.0
    max_pitch: float = 0.0
    drag_enable: bool = False

    capture_radius: float = 0.5
    arena_radius: float = 15.0
    arena_z_min: float = 0.3
    arena_z_max: float = 8.0
    t_start_pos: tuple = (-8.0, -8.0, 4.0)
    a_start_pos: tuple = (8.0, -8.0, 4.0)
    d_start_pos: tuple = (0.0, 8.0, 4.0)
    team_spawn_radius: float = 1.5

    spawn_z_shrink: float = 0.0
    hover_warmup_steps: int = 20
    random_init_attitude: bool = False
    init_attitude_range: tuple = (0.3, 0.3, 0.3)
    warmup_k_att: float = 6.0

    obs_include_own: bool = True
    obs_full_obs: bool = False

    obs_include_lock_count: bool = True

    obs_time_to_go: bool = False

    obs_ang_vel: bool = False

    team_reward_aggr: str = "sum"
    rew_effort: float = 0.0
    rew_T_evade: float = 1.0
    rew_T_survive: float = 0.1
    rew_A_close: float = 1.0
    rew_A_threat: float = -0.5
    rew_D_close: float = 1.0
    rew_dist: float = 0.0

    rew_pbrs_A: float = 0.0
    rew_pbrs_D: float = 0.0
    rew_pbrs_T: float = 0.0

    rew_step_cost: float = 0.0

    capture_gate_enable: bool = False
    capture_gate_v_safe: float = 2.0
    capture_gate_gain: float = 3.0

    rew_sel_A_lockT: float = 0.0
    rew_sel_A_lockD: float = 0.0
    rew_sel_A_threat: float = -0.5
    rew_sel_D_lockA: float = 0.0
    rew_sel_D_lockT: float = 0.0
    rew_sel_T_lockA: float = 0.0
    rew_sel_T_lockD: float = 0.0

    closing_target: str = "selected"

    max_lin_vel: float = 3.0
    defender_vel_ratio: float = 1.0
    evader_vel_ratio: float = 1.0
    rew_los_vel: float = 0.0
    terminal_A_capture: float = 50.0
    terminal_A_captured_by_D: float = -50.0
    terminal_T_captured: float = -50.0
    terminal_T_survive: float = 25.0
    terminal_A_win: float = 25.0
    terminal_D_intercept: float = 50.0
    terminal_D_fail: float = -25.0
    terminal_D_win: float = 25.0

    terminal_D_timeout: float = 0.0
    terminal_A_timeout: float = 0.0

    pbrs_log_dref: float = 0.0

    pbrs_clamp_m: float = 0.5

    pbrs_on_locked: bool = False

    capture_time_bonus: float = 0.0
    rew_oob: float = -1.0
    oob_horiz_margin: float = 4.0
    oob_z_margin: float = 1.0
    oob_horiz_cap: float = 8.0
    oob_z_cap: float = 5.0

    oob_mask_dense: bool = False

    oob_teams: str = "all"

    terminal_T_survive_inbounds: bool = False

    enable_cameras: bool = False

    obs_dim: int = 0
    act_dim: int = 0
    obs_dim_per_agent: dict = None
    lower_obs_dim: int = _LOWER_OBS_DIM
    n_targets_per_uid: dict = None
    n_targets_max: int = 0
    dec_repr_dim: int = 0
    upper_target_mask_dim: int = 0

    def __post_init__(self):
        self._build_agent_spec()

    def _build_agent_spec(self):
        team_members = {
            "T": [f"T_{i}" for i in range(self.n_T)],
            "A": [f"A_{i}" for i in range(self.n_A)],
            "D": [f"D_{i}" for i in range(self.n_D)],
        }
        self.team_members = team_members
        self.all_uids = team_members["T"] + team_members["A"] + team_members["D"]

        ts = str(self.train_side).lower()
        if ts in ("all", "both", "roll"):
            learning_drones = list(self.all_uids)
        elif ts in ("t", "target", "evader"):
            learning_drones = list(team_members["T"])
        elif ts in ("a", "attacker", "pursuer"):
            learning_drones = list(team_members["A"])
        elif ts in ("d", "defender"):
            learning_drones = list(team_members["D"])
        else:
            raise ValueError(f"train_side={self.train_side!r} 非法，需为 T/A/D/all/roll")

        _flat = algo_is_flat(self.rl_algo)
        self.agent_layout = "flat" if _flat else "hier"
        self.hier = not _flat
        if _flat:
            self._flat_validate()

        tl = str(self.train_layer).lower()
        if tl not in ("both", "upper", "lower"):
            raise ValueError(f"train_layer={self.train_layer!r} 非法，需为 both/upper/lower")
        if _flat:
            learning_agents = list(learning_drones)
        else:
            learning_agents = []
            for uid in learning_drones:
                if tl in ("both", "upper"):
                    learning_agents.append(f"{uid}/dec")
                if tl in ("both", "lower"):
                    learning_agents.append(f"{uid}/ctrl")
        self.possible_agents = learning_agents

        if int(self.n_T) < 1 or int(self.n_A) < 1 or int(self.n_D) < 0:
            raise ValueError(
                f"TAD-Hierarchical 数量非法：n_T={self.n_T}, n_A={self.n_A}, n_D={self.n_D}；需 n_T/n_A>=1 且 n_D>=0"
            )
        if int(self.decision_period) < 1:
            raise ValueError(f"decision_period={self.decision_period} 非法，需 >=1")
        n_team = {"T": self.n_T, "A": self.n_A, "D": self.n_D}
        _PRIMARY_POOL = {"T": "A", "A": "T", "D": "A"}

        _ALL_ENEMY = {"T": ["A", "D"], "A": ["T", "D"], "D": ["A", "T"]}
        _DEFAULT_ENEMY = {t: [v for v in _ALL_ENEMY[t] if int(n_team[v]) > 0] for t in ("T", "A", "D")}
        _valid_teams = {"T", "A", "D"}
        _sel_raw = self.selectable_targets or {}
        unknown_keys = set(_sel_raw.keys()) - _valid_teams if isinstance(_sel_raw, dict) else set()
        if unknown_keys:
            raise ValueError(f"selectable_targets 含未知队名键 {sorted(unknown_keys)}，仅支持 T/A/D")
        self.selectable_targets = {}
        for t in ("T", "A", "D"):
            raw = _sel_raw.get(t, _DEFAULT_ENEMY[t]) if isinstance(_sel_raw, dict) else _DEFAULT_ENEMY[t]
            vals = list(raw)
            if not vals:
                raise ValueError(f"selectable_targets[{t}] 为空；每队至少需要一个可选目标队")
            bad = [v for v in vals if v not in _valid_teams]
            if bad:
                raise ValueError(f"selectable_targets[{t}] 含未知队名 {bad}，仅支持 T/A/D")
            if t in vals:
                raise ValueError(f"selectable_targets[{t}] 不应包含本队 {t}；当前仅支持跨队选目标")
            zero = [v for v in vals if int(n_team[v]) <= 0]
            if zero:
                raise ValueError(f"selectable_targets[{t}] 选择了无成员队 {zero}；请从可选池移除或增加对应 n_*")
            self.selectable_targets[t] = vals

        _n_team_slots = max(len(v) for v in self.selectable_targets.values())
        _sel_teams_all = {pt for v in self.selectable_targets.values() for pt in v}
        _n_pool_per_team = max((n_team[t] for t in _sel_teams_all), default=0)
        self._pool_team_slots = int(_n_team_slots)
        self._pool_per_team = int(_n_pool_per_team)

        self._selectable_grid = {
            t: (self.selectable_targets[t] + [_PRIMARY_POOL[t]] * (_n_team_slots - len(self.selectable_targets[t])))
            for t in ("T", "A", "D")
        }

        n_pool_max = self._pool_team_slots * self._pool_per_team
        if int(n_pool_max) < 1:
            raise ValueError("拼接可选目标池宽 n_pool_max=0；请检查 selectable_targets 与 n_T/n_A/n_D")
        self.n_pool_max = n_pool_max
        self.n_targets_max = n_pool_max
        self.n_dec = n_pool_max
        self.dec_repr_dim = self.n_dec
        self.upper_target_mask_dim = n_pool_max

        n_total = self.n_T + self.n_A + self.n_D
        _ttg_dim = 1 if self.obs_time_to_go else 0

        _ang_dim = 3 if (self.obs_ang_vel and not self.obs_full_obs) else 0
        self.critic_state = str(getattr(self, "critic_state", "token")).lower()
        if self.critic_state not in ("token", "anchor"):
            raise ValueError(f"critic_state={self.critic_state!r} 非法（critic 状态表示只支持 token|anchor）")

        self.state_space = 19 * n_total if self.critic_state == "token" else 14 * n_total + _ttg_dim

        self.n_targets_per_uid = {uid: n_team[self.selectable_targets[uid.split("_")[0]][0]] for uid in self.all_uids}

        own_dim = 12 if self.obs_include_own else 0

        nm = n_pool_max
        per_cand = 7 + (1 if self.obs_include_lock_count else 0)
        base = self.state_space if self.obs_full_obs else own_dim
        _ttg_upper = 0 if self.obs_full_obs else _ttg_dim
        upper_obs_dim = base + per_cand * nm + _ttg_upper + nm
        self.obs_dim = upper_obs_dim
        self.act_dim = self.n_dec

        self.obs_dim_per_agent = {}
        self.action_spaces = {}
        self.observation_spaces = {}
        if _flat:
            self._build_flat_spaces()
        else:
            for aid in learning_agents:
                if aid.endswith("/dec"):
                    self.obs_dim_per_agent[aid] = upper_obs_dim
                    self.action_spaces[aid] = {self.n_dec}
                else:
                    ctrl_dim = self.state_space if self.obs_full_obs else self.lower_obs_dim + _ang_dim + _ttg_dim
                    self.obs_dim_per_agent[aid] = ctrl_dim
                    self.action_spaces[aid] = 4
                self.observation_spaces[aid] = self.obs_dim_per_agent[aid]

    def _flat_validate(self) -> None:
        if str(self.train_layer).lower() != "both":
            raise ValueError(f"flat 算法无层概念，train_layer 需为 both（当前 {self.train_layer!r}）")
        if str(getattr(self, "critic_state", "token")).lower() != "token":
            raise ValueError("flat 算法仅支持 rl.critic_state=token（19/机 全局 token state）")
        if bool(self.obs_full_obs):
            raise ValueError("flat 算法不支持 observation.full_obs=true（flat 有自己的观测构造，见 _build_flat_obs）")
        if canonical_train_mode(getattr(self, "train_side", "all")) == "roll":
            raise ValueError("flat 算法 v1 不支持 train_side=roll（轮转冻结未实现）；请用 all/T/A/D")
        if canonical_train_mode(getattr(self, "train_takeover", "false")) != "false":
            raise ValueError("flat 算法 v1 不支持 train_takeover（两阶段模式切换未实现）")
        if not getattr(self, "_flat_notice_printed", False):
            self._flat_notice_printed = True
            print(
                "[TAD][flat] 布局=flat（algo=" + self.rl_algo + "）；忽略的 hier 键："
                "observation.include_lock_count / reward.pbrs_on_locked（flat 锁恒为脚本 nearest，与 false 等价）"
                " / observation.time_to_go（15 维自机块固定含 t/T） / observation.ang_vel（角速度固定含于自机块）"
            )

    def _build_flat_spaces(self) -> None:
        own = 15
        mode = str(self.flat_obs).lower()
        self.flat_obs_mode = mode
        self.flat_obs_dim_per_team: dict[str, int] = {}
        self.flat_peer_layout: dict[str, list[str]] | None = None
        if mode == "pool":
            P = int(self.n_pool_max)
            self.flat_obs_dim_per_team = {t: own + 7 * P + P for t in self.team_members}
        elif mode == "peers":
            self.flat_peer_layout = {}
            for team, members in self.team_members.items():
                sel = list(self.selectable_targets[team])
                if len(set(sel)) != len(sel):
                    raise ValueError(f"flat_obs=peers 要求 selectable_targets[{team}] 无重复队名：{sel}")
                tgt_uids: list[str] = []
                for t in sel:
                    tgt_uids.extend(self.team_members.get(t, []))
                for u in members:
                    self.flat_peer_layout[u] = [m for m in members if m != u] + tgt_uids
                self.flat_obs_dim_per_team[team] = own + 10 * ((len(members) - 1) + len(tgt_uids))
        else:
            raise ValueError(f"rl.flat_obs={self.flat_obs!r} 非法，需 pool|peers")
        for aid in self.possible_agents:
            team = aid.split("_")[0]
            self.obs_dim_per_agent[aid] = self.flat_obs_dim_per_team[team]
            self.action_spaces[aid] = 4
            self.observation_spaces[aid] = self.obs_dim_per_agent[aid]
        _first_team = self.possible_agents[0].split("_")[0]
        self.obs_dim = self.flat_obs_dim_per_team[_first_team]
        self.act_dim = 4
        _dims = {
            t: d for t, d in self.flat_obs_dim_per_team.items() if t in {a.split("_")[0] for a in self.possible_agents}
        }
        print(
            f"[TAD][flat] flat_obs={mode}  actor_obs_dim(按队)={_dims}  "
            f"agents={len(self.possible_agents)}×Box(4)  state={self.state_space}(token)"
        )

    def apply_rl_algo(self, algo) -> str:
        a = canonical_algo(algo)
        self.rl_algo = a
        self.flat_obs = str(self.flat_obs).lower()
        self._build_agent_spec()
        return a

    def apply_drone_config(self, drone_cfg: dict):
        """Apply the saved UAV asset and dynamics settings."""
        self.drone_scale = float(drone_cfg.get("scale", self.drone_scale))
        self.thrust_to_weight = float(drone_cfg.get("thrust_to_weight", self.thrust_to_weight))
        self.moment_max = float(drone_cfg.get("moment_max", self.moment_max))
        self.action_mode = str(drone_cfg.get("action_mode", self.action_mode))
        self.rate_max = float(drone_cfg.get("rate_max", self.rate_max))
        self.rate_kp = float(drone_cfg.get("rate_kp", self.rate_kp))
        self.max_roll = float(drone_cfg.get("max_roll", self.max_roll))
        self.max_pitch = float(drone_cfg.get("max_pitch", self.max_pitch))
        self.drag_enable = bool(drone_cfg.get("drag_enable", self.drag_enable))
        asset = drone_cfg.get("asset", {}) or {}
        self._drone_usd_path = asset.get("usd_path", None)
        self._drone_copy_from_source = bool(asset.get("copy_from_source", False))
        self._drone_rigid_props = drone_cfg.get("rigid_props", {}) or {}
        self._drone_articulation_props = drone_cfg.get("articulation_props", {}) or {}
        self._drone_init_state = drone_cfg.get("init_state", {}) or {}
        self._drone_actuators = drone_cfg.get("actuators", {}) or {}

    def drone_articulation_cfg(self, uid: str) -> ArticulationCfg:
        asset = self._drone_init_state
        team = uid.split("_")[0]
        color = _TEAM_COLORS.get(team, (0.5, 0.5, 0.5))
        return CRAZYFLIE_CFG.replace(
            prim_path=f"/World/envs/env_.*/{uid}",
            spawn=sim_utils.UsdFileCfg(
                usd_path=self._drone_usd_path or _LOCAL_CF2X_USD,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(**self._drone_rigid_props),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(**self._drone_articulation_props),
                scale=(self.drone_scale,) * 3,
                copy_from_source=self._drone_copy_from_source,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=tuple(asset.get("pos", [0.0, 0.0, 0.5])),
                joint_vel=asset.get("joint_vel", {}),
            ),
            actuators={
                "dummy": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    stiffness=float(self._drone_actuators.get("stiffness", 0.0)),
                    damping=float(self._drone_actuators.get("damping", 0.0)),
                    effort_limit=self._drone_actuators.get("effort_limit"),
                    velocity_limit=self._drone_actuators.get("velocity_limit"),
                ),
            },
        )

    @classmethod
    def from_config_dict(cls, d: dict) -> "TadHierarchicalEnvCfg":
        """Construct the environment configuration from scenario settings."""
        cfg = cls()

        s = d.get("scenario", {})
        cfg.n_T = int(s.get("n_T", cfg.n_T))
        cfg.n_A = int(s.get("n_A", cfg.n_A))
        cfg.n_D = int(s.get("n_D", cfg.n_D))
        cfg.elimination_mode = str(s.get("elimination_mode", cfg.elimination_mode)).lower()
        if cfg.elimination_mode not in ("paired", "catcher_survives"):
            raise ValueError(f"elimination_mode={cfg.elimination_mode!r} 非法，需 paired|catcher_survives")
        cfg.train_side = str(s.get("train_side", cfg.train_side))
        cfg.train_layer = str(s.get("train_layer", cfg.train_layer))

        cfg.roll_period = int(s.get("roll_period", cfg.roll_period))
        _ro = s.get("roll_order", cfg.roll_order)
        cfg.roll_order = list(_ro) if _ro else ["T", "A", "D"]
        cfg.roll_freeze_depth = str(s.get("roll_freeze_depth", cfg.roll_freeze_depth)).lower()
        if cfg.roll_freeze_depth not in ("actor", "full"):
            raise ValueError(f"roll_freeze_depth={cfg.roll_freeze_depth!r} 非法，需 actor|full")
        if str(cfg.train_side).lower() == "roll":
            _valid_teams = {"T", "A", "D"}
            _bad = [t for t in cfg.roll_order if t not in _valid_teams]
            if _bad or not cfg.roll_order:
                raise ValueError(f"roll_order={cfg.roll_order!r} 非法，需 T/A/D 的非空子集")
            if cfg.roll_period <= 0:
                raise ValueError(f"roll_period={cfg.roll_period} 需 >0")

        cfg.exper_rate = s.get("exper_rate", None)

        cfg.train_takeover = s.get("train_takeover", "false")
        cfg.train_takeover_step = s.get("train_takeover_step", 0)
        validate_train_takeover(cfg)

        validate_exper_rate(cfg)
        cfg.decision_period = int(s.get("decision_period", cfg.decision_period))
        cfg.scripted_decision = str(s.get("scripted_decision", cfg.scripted_decision))
        cfg.q_low_input = str(s.get("q_low_input", cfg.q_low_input)).lower()
        if cfg.q_low_input not in ("none", "full", "selected"):
            raise ValueError(f"q_low_input={cfg.q_low_input!r} 非法，需 none|full|selected")
        cfg.q_low_rel_dim = int(s.get("q_low_rel_dim", cfg.q_low_rel_dim))

        _sel = s.get("selectable_targets")
        if _sel:
            cfg.selectable_targets = {k: list(v) for k, v in _sel.items()}
        cfg.ppn_nav_const = float(s.get("ppn_nav_const", cfg.ppn_nav_const))
        cfg.evader_a_max = float(s.get("evader_a_max", cfg.evader_a_max))
        cfg.accel_k_att = float(s.get("accel_k_att", cfg.accel_k_att))
        cfg.accel_k_tilt = float(s.get("accel_k_tilt", cfg.accel_k_tilt))
        cfg.max_lean = float(s.get("max_lean", cfg.max_lean))
        cfg.heur_k_yaw = float(s.get("heur_k_yaw", cfg.heur_k_yaw))
        cfg.heur_k_pitch = float(s.get("heur_k_pitch", cfg.heur_k_pitch))
        cfg.heur_k_roll = float(s.get("heur_k_roll", cfg.heur_k_roll))
        cfg.heur_max_lean = float(s.get("heur_max_lean", cfg.heur_max_lean))
        cfg.heur_k_att = float(s.get("heur_k_att", cfg.heur_k_att))
        cfg.heur_k_alt = float(s.get("heur_k_alt", cfg.heur_k_alt))
        cfg.heur_k_dz = float(s.get("heur_k_dz", cfg.heur_k_dz))
        cfg.heur_noise_std = float(s.get("heur_noise_std", cfg.heur_noise_std))

        cfg.frozen_opponent_prob = float(s.get("frozen_opponent_prob", cfg.frozen_opponent_prob))

        cfg.expert_curriculum_enable = bool(s.get("expert_curriculum_enable", cfg.expert_curriculum_enable))
        cfg.expert_curriculum_prob = float(s.get("expert_curriculum_prob", cfg.expert_curriculum_prob))

        cfg.expert_curriculum_anneal_frac = float(
            s.get("expert_curriculum_anneal_frac", cfg.expert_curriculum_anneal_frac)
        )

        cfg.curriculum_enable = bool(s.get("curriculum_enable", cfg.curriculum_enable))
        cfg.curriculum_capture_radius_start = float(
            s.get("curriculum_capture_radius_start", cfg.curriculum_capture_radius_start)
        )
        cfg.curriculum_capture_radius_end = float(
            s.get("curriculum_capture_radius_end", cfg.curriculum_capture_radius_end)
        )
        cfg.curriculum_capture_radius_step = float(
            s.get("curriculum_capture_radius_step", cfg.curriculum_capture_radius_step)
        )
        cfg.curriculum_promotion_winrate = float(
            s.get("curriculum_promotion_winrate", cfg.curriculum_promotion_winrate)
        )
        cfg.curriculum_promotion_metric = str(
            s.get("curriculum_promotion_metric", cfg.curriculum_promotion_metric)
        ).lower()
        if cfg.curriculum_promotion_metric not in ("attacker", "decisive"):
            raise ValueError(
                f"curriculum_promotion_metric={cfg.curriculum_promotion_metric!r} 非法，需为 attacker|decisive"
            )
        cfg.curriculum_min_samples = int(s.get("curriculum_min_samples", cfg.curriculum_min_samples))
        cfg.curriculum_check_every = max(1, int(s.get("curriculum_check_every", cfg.curriculum_check_every)))
        cfg.capture_radius = float(s.get("capture_radius", cfg.capture_radius))
        cfg.arena_radius = float(s.get("arena_radius", cfg.arena_radius))
        cfg.arena_z_min = float(s.get("arena_z_min", cfg.arena_z_min))
        cfg.arena_z_max = float(s.get("arena_z_max", cfg.arena_z_max))
        cfg.episode_length_s = float(s.get("episode_length_s", cfg.episode_length_s))
        cfg.episode_length_jitter_s = float(s.get("episode_length_jitter_s", cfg.episode_length_jitter_s))
        _wr = s.get("win_rate_horizons", None)
        cfg.win_rate_horizons = list(_wr) if _wr else ([1000, 5000])
        cfg.telemetry_interval = max(1, int(s.get("telemetry_interval", cfg.telemetry_interval)))
        if "t_start_pos" in s:
            cfg.t_start_pos = tuple(s["t_start_pos"])
        if "a_start_pos" in s:
            cfg.a_start_pos = tuple(s["a_start_pos"])
        if "d_start_pos" in s:
            cfg.d_start_pos = tuple(s["d_start_pos"])
        cfg.team_spawn_radius = float(s.get("team_spawn_radius", cfg.team_spawn_radius))
        cfg.spawn_z_shrink = float(s.get("spawn_z_shrink", cfg.spawn_z_shrink))
        cfg.hover_warmup_steps = int(s.get("hover_warmup_steps", cfg.hover_warmup_steps))
        cfg.random_init_attitude = bool(s.get("random_init_attitude", cfg.random_init_attitude))
        cfg.init_attitude_range = tuple(s.get("init_attitude_range", cfg.init_attitude_range))
        cfg.warmup_k_att = float(s.get("warmup_k_att", cfg.warmup_k_att))

        cfg.max_lin_vel = float(s.get("max_lin_vel", cfg.max_lin_vel))
        cfg.defender_vel_ratio = float(s.get("defender_vel_ratio", cfg.defender_vel_ratio))
        cfg.evader_vel_ratio = float(s.get("evader_vel_ratio", cfg.evader_vel_ratio))
        cfg.oob_horiz_margin = float(s.get("oob_horiz_margin", cfg.oob_horiz_margin))
        cfg.oob_z_margin = float(s.get("oob_z_margin", cfg.oob_z_margin))
        cfg.oob_horiz_cap = float(s.get("oob_horiz_cap", cfg.oob_horiz_cap))
        cfg.oob_z_cap = float(s.get("oob_z_cap", cfg.oob_z_cap))
        cfg.evader_wall_margin = float(s.get("evader_wall_margin", cfg.evader_wall_margin))
        cfg.evader_inward_beta = float(s.get("evader_inward_beta", cfg.evader_inward_beta))
        cfg.evader_tangent_min = float(s.get("evader_tangent_min", cfg.evader_tangent_min))
        cfg.evader_k_z = float(s.get("evader_k_z", cfg.evader_k_z))
        cfg.evader_z_margin = float(s.get("evader_z_margin", cfg.evader_z_margin))

        e = d.get("env", {})
        cfg.scene.num_envs = int(e.get("num_envs", cfg.scene.num_envs))

        cfg.train_num_envs = cfg.scene.num_envs
        cfg.scene.env_spacing = float(e.get("env_spacing", cfg.scene.env_spacing))
        cfg.decimation = int(e.get("decimation", cfg.decimation))
        if "sim_dt" in e:
            cfg.sim.dt = float(e["sim_dt"])
            cfg.sim.render_interval = cfg.decimation

        cfg.test_num_envs = int(e.get("test_num_envs", s.get("test_num_envs", cfg.test_num_envs)))
        cfg.test_every_steps = int(e.get("test_every_steps", s.get("test_every_steps", cfg.test_every_steps)))
        cfg.test_episodes = int(e.get("test_episodes", s.get("test_episodes", cfg.test_episodes)))
        cfg.test_exper = bool(e.get("test_exper", s.get("test_exper", cfg.test_exper)))
        if cfg.test_exper and cfg.test_num_envs <= 0:
            print("[TAD][cfg][WARN] env.test_exper=true 但 test_num_envs=0 → test_exper 不生效（忽略）")
            cfg.test_exper = False
        if cfg.test_num_envs < 0:
            raise ValueError(f"test_num_envs={cfg.test_num_envs} 非法，需 >=0")
        if cfg.test_num_envs > 0:
            if cfg.test_every_steps < 1:
                raise ValueError(f"test_every_steps={cfg.test_every_steps} 非法，需 >=1")
            if cfg.test_episodes < 1:
                raise ValueError(f"test_episodes={cfg.test_episodes} 非法，需 >=1")

        if cfg.test_num_envs > 0:
            _step_dt = cfg.sim.dt * cfg.decimation
            _max_ep_steps = int(-(-cfg.episode_length_s // _step_dt))
            if cfg.test_every_steps < _max_ep_steps:
                print(
                    f"[TAD][test-env][WARN] test_every_steps={cfg.test_every_steps} < 最长回合步数"
                    f"≈{_max_ep_steps}（episode_length_s={cfg.episode_length_s}s / step_dt={_step_dt}s）："
                    f"测试相内回合大量被边界丢弃，建议 ≥{_max_ep_steps}"
                )

        t = d.get("train_set", {}) or {}
        if "device" in t and t["device"]:
            cfg.sim.device = str(t["device"])

        o = d.get("observation", {})
        cfg.obs_include_own = bool(o.get("include_own", cfg.obs_include_own))
        cfg.obs_full_obs = bool(o.get("full_obs", cfg.obs_full_obs))
        cfg.obs_include_lock_count = bool(o.get("include_lock_count", cfg.obs_include_lock_count))
        cfg.obs_time_to_go = bool(o.get("time_to_go", cfg.obs_time_to_go))
        cfg.obs_ang_vel = bool(o.get("ang_vel", cfg.obs_ang_vel))

        cfg.critic_state = str(d.get("rl", {}).get("critic_state", cfg.critic_state)).lower()
        cfg.lower_critic_mode = str(d.get("rl", {}).get("lower_critic_mode", cfg.lower_critic_mode)).lower()
        if cfg.lower_critic_mode not in ("per_agent", "team"):
            raise ValueError(f"rl.lower_critic_mode={cfg.lower_critic_mode!r} 非法，需 per_agent|team")

        cfg.rl_algo = canonical_algo(d.get("rl", {}).get("algo", cfg.rl_algo))
        cfg.flat_obs = str(d.get("rl", {}).get("flat_obs", cfg.flat_obs)).lower()
        if cfg.flat_obs not in ("pool", "peers"):
            raise ValueError(f"rl.flat_obs={cfg.flat_obs!r} 非法，需 pool|peers")

        r = d.get("reward", {})
        cfg.team_reward_aggr = str(r.get("aggr", cfg.team_reward_aggr))
        cfg.closing_target = str(r.get("closing_target", cfg.closing_target))
        for key in [
            "rew_effort",
            "rew_T_evade",
            "rew_T_survive",
            "rew_A_close",
            "rew_A_threat",
            "rew_D_close",
            "rew_dist",
            "rew_pbrs_A",
            "rew_pbrs_D",
            "rew_pbrs_T",
            "rew_step_cost",
            "rew_sel_A_lockT",
            "rew_sel_A_lockD",
            "rew_sel_A_threat",
            "rew_sel_D_lockA",
            "rew_sel_D_lockT",
            "rew_sel_T_lockA",
            "rew_sel_T_lockD",
            "rew_los_vel",
            "terminal_A_capture",
            "terminal_A_captured_by_D",
            "terminal_T_captured",
            "terminal_T_survive",
            "terminal_A_win",
            "terminal_D_intercept",
            "terminal_D_fail",
            "terminal_D_win",
            "terminal_A_timeout",
            "terminal_D_timeout",
            "pbrs_log_dref",
            "pbrs_clamp_m",
            "rew_oob",
            "capture_time_bonus",
        ]:
            if key in r:
                setattr(cfg, key, float(r[key]))
        cfg.pbrs_on_locked = bool(r.get("pbrs_on_locked", cfg.pbrs_on_locked))
        cfg.oob_mask_dense = bool(r.get("oob_mask_dense", cfg.oob_mask_dense))
        _ot = str(r.get("oob_teams", cfg.oob_teams) or "all").strip().upper()
        if _ot not in ("T", "A", "D", "ALL"):
            raise ValueError(f"oob_teams={_ot!r} 非法：需 T | A | D | all")
        cfg.oob_teams = _ot
        cfg.terminal_T_survive_inbounds = bool(r.get("terminal_T_survive_inbounds", cfg.terminal_T_survive_inbounds))

        cg = r.get("capture_gate", {}) or {}
        cfg.capture_gate_enable = bool(cg.get("enable", cfg.capture_gate_enable))
        cfg.capture_gate_v_safe = float(cg.get("v_safe", cfg.capture_gate_v_safe))
        cfg.capture_gate_gain = float(cg.get("gain", cfg.capture_gate_gain))

        cfg._build_agent_spec()
        return cfg

    def print_config_summary(self, rl_cfg, algo, trainer_cfg, use_wandb, log_dir):
        algo_line = ""
        if algo == "mappo":
            algo_line = (
                f"  [MAPPO] 上层=Categorical 下层=Gaussian  rollouts={rl_cfg.get('rollouts')}  "
                f"p_lr={rl_cfg.get('policy_lr')}  c_lr={rl_cfg.get('critic_lr')}  "
                f"epochs={rl_cfg.get('learning_epochs')}  mini_batches={rl_cfg.get('mini_batches')}  "
                f"λ={rl_cfg.get('gae_lambda')}  clip=ε{rl_cfg.get('ratio_clip')}  "
                f"entropy={rl_cfg.get('entropy_loss_scale')}  kl_thr={rl_cfg.get('kl_early_stop')}\n"
            )
        elif algo == "maddpg":
            algo_line = (
                f"  [MADDPG] 上层=Gumbel 下层=DPG  "
                f"B_L_steps={rl_cfg.get('replay_steps', 0)}  "
                f"n_step={rl_cfg.get('n_step', 1)}  "
                f"B_H={rl_cfg.get('decision_replay_size', rl_cfg.get('replay_size', 1e6))}  "
                f"store={rl_cfg.get('replay_device', 'cuda')}  "
                f"batch={rl_cfg.get('batch_size')}  grad_steps={rl_cfg.get('gradient_steps')}  "
                f"lr_start={rl_cfg.get('learning_starts')}  τ={1 - rl_cfg.get('polyak', 0.995):.3f}  "
                f"gumbel_tau={rl_cfg.get('gumbel_tau')}  eps_greedy={rl_cfg.get('epsilon_greedy')}  "
                f"noise_std={rl_cfg.get('exploration_noise_std')}\n"
            )
        elif algo == "matd3":
            algo_line = (
                f"  [MATD3] 上层=Gumbel 下层=DPG(twin/smooth/delay)  "
                f"B_L_steps={rl_cfg.get('replay_steps', 0)}  "
                f"n_step={rl_cfg.get('n_step', 1)}  "
                f"batch={rl_cfg.get('batch_size')}  grad_steps={rl_cfg.get('gradient_steps')}  "
                f"lr_start={rl_cfg.get('learning_starts')}  τ={1 - rl_cfg.get('polyak', 0.995):.3f}  "
                f"twin={rl_cfg.get('matd3_twin', True)}  "
                f"smooth={rl_cfg.get('matd3_smooth', True)}"
                f"(σ={rl_cfg.get('matd3_smooth_std', 0.2)},c={rl_cfg.get('matd3_smooth_clip', 0.5)})  "
                f"delay={rl_cfg.get('matd3_delay', True)}(d={rl_cfg.get('matd3_policy_delay', 2)})  "
                f"noise_std={rl_cfg.get('exploration_noise_std')}\n"
            )
        elif algo in FLAT_ALGOS:
            algo_line = (
                f"  [Flat-{'TD3' if algo == 'flattd3' else 'PPO'}] 直接控制基线（无 /dec 决策层）  "
                f"flat_obs={getattr(self, 'flat_obs_mode', getattr(self, 'flat_obs', 'pool'))}  "
                f"actor_hidden={rl_cfg.get('flat_actor_hidden')}  "
                + (
                    f"batch={rl_cfg.get('batch_size')}  n_step={rl_cfg.get('n_step', 1)}  "
                    f"twin={rl_cfg.get('matd3_twin', True)}  delay={rl_cfg.get('matd3_delay', True)}"
                    f"(d={rl_cfg.get('matd3_policy_delay', 2)})  "
                    f"noise_std={rl_cfg.get('exploration_noise_std')}\n"
                    if algo == "flattd3"
                    else f"rollouts={rl_cfg.get('rollouts')}  epochs={rl_cfg.get('learning_epochs')}  "
                    f"mini_batches={rl_cfg.get('mini_batches')}  λ={rl_cfg.get('gae_lambda')}  "
                    f"clip=ε{rl_cfg.get('ratio_clip')}\n"
                )
            )

        dec_agent_count = sum(1 for a in self.possible_agents if a.endswith("/dec"))
        n_ctrl = sum(1 for a in self.possible_agents if a.endswith("/ctrl"))
        _flat = algo_is_flat(self.rl_algo)
        if _flat:
            obs_action_lines = (
                f"  [观测]  flat_obs={getattr(self, 'flat_obs_mode', self.flat_obs)}  "
                f"actor_obs_dim(按队)={self.flat_obs_dim_per_team}  "
                f"state_dim={self.state_space}(token)\n"
                f"  [动作]  全机 Box(4)（直接控制 [T,p,q,r]；无 /dec 决策层，锁=脚本 nearest 仅内部用）\n"
                f"  [智能体] possible_agents={self.possible_agents}（每架物理机一个连续 agent）\n"
            )
        else:
            obs_action_lines = (
                f"  [观测]  上层 obs_dim={self.obs_dim}  下层 obs_dim={self.state_space if self.obs_full_obs else self.lower_obs_dim}{'+3(ang)' if self.obs_ang_vel and not self.obs_full_obs else ''}{'+' + str(1) if self.obs_time_to_go and not self.obs_full_obs else ''}  state_dim={self.state_space}({self.critic_state})  full_obs={self.obs_full_obs}  time_to_go={self.obs_time_to_go}  ang_vel={self.obs_ang_vel}\n"
                f"  [动作]  上层 Discrete(n_dec={self.n_dec}=P)  dec_repr=target_onehot(repr_dim={self.dec_repr_dim}, target_mask={self.upper_target_mask_dim})  n_pool_slot={self.n_pool_max}  selectable={self.selectable_targets}  下层 Box(4)\n"
                f"  [智能体] possible_agents={self.possible_agents}  (dec×{dec_agent_count}, ctrl×{n_ctrl})\n"
            )
        print(
            f"\n{'=' * 70}\n"
            f"  TAD-Hierarchical (v4.0.0) 训练配置概览\n"
            f"{'=' * 70}\n"
            f"  [场景]  {self.n_T}v{self.n_A}v{self.n_D}  elimination_mode={self.elimination_mode}  "
            f"train_side={self.train_side}  train_layer={self.train_layer}  "
            f"exper_rate={self.exper_rate}  |  "
            f"capture={self.capture_radius}m  arena=R{self.arena_radius}/Z[{self.arena_z_min},{self.arena_z_max}]m  "
            f"ep_len={self.episode_length_s}s  ep_jitter={self.episode_length_jitter_s}s\n"
            f"  [分层]  decision_period={self.decision_period}  共享critic+reward  "
            f"未学下层=级联  k_yaw={self.heur_k_yaw}  k_att={self.heur_k_att}  max_lean={self.heur_max_lean}  "
            f"k_alt={self.heur_k_alt}  k_dz={self.heur_k_dz}  noise={self.heur_noise_std}\n"
            f"  [环境]  num_envs={getattr(self, 'train_num_envs', 0) or self.scene.num_envs}"
            f"{f' + test {self.test_num_envs}' if getattr(self, 'test_num_envs', 0) else ''}"
            f"  device={self.sim.device}  "
            f"dt={self.sim.dt:.4f}s  decimation={self.decimation}  action_mode={self.action_mode}\n"
            + obs_action_lines
            + f"  [算法]  algo={algo}  trainer_steps={trainer_cfg['timesteps']}  "
            f"γ_c={rl_cfg.get('discount_factor_control', rl_cfg.get('discount_factor', 0.99))}  "
            f"γ_d={rl_cfg.get('discount_factor_decision', rl_cfg.get('discount_factor_control', rl_cfg.get('discount_factor', 0.99)))}  "
            f"critic_mode={rl_cfg.get('critic_mode', 'team')}  "
            f"actor_mode={rl_cfg.get('actor_mode', 'per_agent')}  "
            f"\n" + algo_line + f"  [日志]  wandb={'on' if use_wandb else 'off'}  dir={log_dir}\n"
            f"{'=' * 70}\n"
        )
