from MTAD.algorithms.running_standard_scaler import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR

from MTAD.algorithms.mappo import MAPPO
from MTAD.algorithms.maddpg import MADDPG
from MTAD.algorithms.matd3 import MATD3
from MTAD.algorithms.matd3ppo import MATD3PPO
from MTAD.algorithms.models import build_mappo_models
from MTAD.algorithms.flat import FlatTD3, FlatPPO


def build_hierarchical_agent(
    algo,
    env,
    rl_cfg,
    env_cfg,
    observation_spaces,
    action_spaces,
    state_space,
    num_envs,
    device,
    possible_agents,
    team_of,
    experiment,
    norm_cfg=None,
):
    """Build the hierarchical policy and its training configuration."""

    norm_cfg = norm_cfg or {}
    if algo in ("maddpg", "matd3", "matd3ppo"):
        cfg = dict(rl_cfg)
        cfg["experiment"] = experiment
        cfg["hier"] = True
        cfg["all_uids"] = list(env_cfg.all_uids)
        cfg["n_targets_max"] = int(env_cfg.n_targets_max)
        cfg["critic_state"] = str(getattr(env_cfg, "critic_state", "token")).lower()
        cfg["q_low_input"] = str(getattr(env_cfg, "q_low_input", "none")).lower()
        cfg["q_low_rel_dim"] = int(getattr(env_cfg, "q_low_rel_dim", 7))
        cfg["lower_critic_mode"] = str(getattr(env_cfg, "lower_critic_mode", "per_agent")).lower()
        cfg["decision_period"] = int(getattr(env_cfg, "decision_period", 8))
        cfg["gumbel_tau"] = float(rl_cfg.get("gumbel_tau", 1.0))
        cfg["epsilon_greedy"] = float(rl_cfg.get("epsilon_greedy", 0.1))
        cfg["eps_decay"] = float(rl_cfg.get("eps_decay", 1.0))
        cfg["normalize"] = bool(norm_cfg.get("normalize", True))
        cfg["norm_epsilon"] = float(norm_cfg.get("epsilon", 1e-8))
        cfg["norm_clip_threshold"] = float(norm_cfg.get("clip_threshold", 5.0))
        agent = (MADDPG if algo == "maddpg" else MATD3 if algo == "matd3" else MATD3PPO)(
            possible_agents=possible_agents,
            observation_spaces=observation_spaces,
            action_spaces=action_spaces,
            state_space=state_space,
            num_envs=num_envs,
            device=device,
            team_of=team_of,
            cfg=cfg,
            team_reward_aggr=env_cfg.team_reward_aggr,
            critic_hidden=rl_cfg.get("critic_hidden", [256, 256]),
            activation=rl_cfg.get("activation", "elu"),
        )
        dec_agent_count = sum(1 for a in possible_agents if a.endswith("/dec"))
        n_ctrl = sum(1 for a in possible_agents if a.endswith("/ctrl"))
        _pool = cfg["n_targets_max"]
        _n_dec = _pool
        _repr = _pool
        _m = len(cfg["all_uids"])
        print(
            f"[TAD][build] {algo.upper()} hier：dec×{dec_agent_count} ctrl×{n_ctrl} n_pool={_pool} "
            f"exec_dec={_n_dec} repr_dec={_repr} M={_m} "
            f"high_act={_m * _repr} total_act={_m * (_repr + 4)} lower_critic_mode={cfg['lower_critic_mode']}"
        )
        if algo == "matd3":
            print(
                f"[TAD][build] MATD3 tricks：twin={cfg.get('matd3_twin', True)} "
                f"smooth={cfg.get('matd3_smooth', True)}(σ={cfg.get('matd3_smooth_std', 0.2)},"
                f"c={cfg.get('matd3_smooth_clip', 0.5)}) "
                f"delay={cfg.get('matd3_delay', True)}(d={cfg.get('matd3_policy_delay', 2)})"
            )
        return agent
    elif algo == "mappo":
        return _build_hierarchical_mappo_agent(
            env,
            rl_cfg,
            env_cfg,
            observation_spaces,
            action_spaces,
            state_space,
            possible_agents,
            team_of,
            experiment,
            norm_cfg,
        )
    else:
        raise ValueError(f"hier 不支持算法 {algo}")


def build_flat_agent(
    algo,
    env,
    rl_cfg,
    env_cfg,
    observation_spaces,
    action_spaces,
    state_space,
    num_envs,
    device,
    possible_agents,
    team_of,
    experiment,
    norm_cfg=None,
):
    """Build a flat continuous-control baseline."""

    norm_cfg = norm_cfg or {}
    cfg = dict(rl_cfg)
    cfg["experiment"] = experiment
    cfg["all_uids"] = list(env_cfg.all_uids)
    cfg["flat_obs"] = str(getattr(env_cfg, "flat_obs_mode", getattr(env_cfg, "flat_obs", "pool")))
    cfg["normalize"] = bool(norm_cfg.get("normalize", True))
    cfg["norm_epsilon"] = float(norm_cfg.get("epsilon", 1e-8))
    cfg["norm_clip_threshold"] = float(norm_cfg.get("clip_threshold", 5.0))
    cfg["telemetry_interval"] = int(getattr(env_cfg, "telemetry_interval", 50))

    for _k in ("flat_ppo_initial_log_std", "flat_ppo_max_log_std"):
        if rl_cfg.get(_k) is not None:
            cfg[_k] = float(rl_cfg[_k])
    cls = FlatTD3 if algo == "flattd3" else FlatPPO
    agent = cls(
        possible_agents=possible_agents,
        observation_spaces=observation_spaces,
        action_spaces=action_spaces,
        state_space=state_space,
        num_envs=num_envs,
        device=device,
        team_of=team_of,
        cfg=cfg,
        team_reward_aggr=env_cfg.team_reward_aggr,
    )
    print(
        f"[TAD][build] FLAT {algo}：agents={possible_agents}  "
        f"flat_obs={cfg['flat_obs']}  state={state_space.shape[0]}(token)  "
        f"norm={'on' if cfg['normalize'] else 'off'}  aggr={env_cfg.team_reward_aggr}"
    )
    return agent


def _build_hierarchical_mappo_agent(
    env,
    rl_cfg,
    env_cfg,
    observation_spaces,
    action_spaces,
    state_space,
    possible_agents,
    team_of,
    experiment,
    norm_cfg=None,
):
    norm_cfg = norm_cfg or {}

    _critic_state = str(getattr(env_cfg, "critic_state", "token")).lower()
    if int(rl_cfg.get("random_timesteps", 0)) != 0:
        raise ValueError("TAD-Hierarchical MAPPO 固定 random_timesteps=0；探索由 Categorical/Gaussian 策略分布提供")
    norm_enabled = bool(norm_cfg.get("normalize", True))

    value_norm = norm_enabled and bool(norm_cfg.get("value_normalize", False))
    norm_eps = float(norm_cfg.get("epsilon", 1e-8))
    norm_clip = float(norm_cfg.get("clip_threshold", 5.0))
    state_spaces = {aid: state_space for aid in possible_agents}
    _P = int(getattr(env_cfg, "n_targets_max", 0))
    _has_dec = any(a.endswith("/dec") for a in possible_agents)
    _has_ctrl = any(a.endswith("/ctrl") for a in possible_agents)
    gamma_control = float(rl_cfg.get("discount_factor_control", rl_cfg.get("discount_factor", 0.99)))

    gamma_decision = float(rl_cfg.get("discount_factor_decision", rl_cfg.get("discount_factor", gamma_control)))
    hierarchical_cfg = {
        "n_pool": _P,
        "decision_period": int(getattr(env_cfg, "decision_period", 8)),
        "gamma_control": gamma_control,
        "polyak": float(rl_cfg.get("polyak", 0.995)),
        "batch_size": int(rl_cfg.get("batch_size", 512)),
        "telemetry_interval": int(getattr(env_cfg, "telemetry_interval", 50)),
        "use_layer_norm": bool(rl_cfg.get("use_layer_norm", False)),
        "activation": rl_cfg.get("activation", "elu"),
        "rollouts_dec": int(rl_cfg.get("rollouts_dec", 0)),
        "expert_curriculum_enable": bool(getattr(env_cfg, "expert_curriculum_enable", False)),
        "expert_curriculum_prob": float(getattr(env_cfg, "expert_curriculum_prob", 0.0)),
        "expert_curriculum_anneal_frac": float(getattr(env_cfg, "expert_curriculum_anneal_frac", 0.0)),
        "obs_include_own": bool(getattr(env_cfg, "obs_include_own", True)),
        "lower_actor_hidden": list(rl_cfg.get("lower_actor_hidden", [256, 256])),
        "clip_actions": bool(rl_cfg.get("clip_actions", False)),
        "observation_space_per_aid": {aid: observation_spaces[aid] for aid in possible_agents},
        "action_space_per_aid": {aid: action_spaces[aid] for aid in possible_agents},
        "learning_rate": (float(rl_cfg.get("policy_lr", 3e-4)), float(rl_cfg.get("critic_lr", 3e-4))),
    }

    for _k in ("ctrl_initial_log_std", "ctrl_max_log_std"):
        if rl_cfg.get(_k) is not None:
            hierarchical_cfg[_k] = float(rl_cfg[_k])

    hierarchical_cfg["critic_state"] = _critic_state
    rl_cfg_models = dict(rl_cfg)
    rl_cfg_models["hier"] = True
    rl_cfg_models["is_discrete"] = False
    rl_cfg_models["n_targets_max"] = _P
    rl_cfg_models["critic_state"] = _critic_state
    models, memories = build_mappo_models(
        observation_spaces=observation_spaces,
        action_spaces=action_spaces,
        state_space=state_space,
        num_envs=env.num_envs,
        device=env.device,
        rl_cfg=rl_cfg_models,
        team_of=team_of,
    )
    lr_kl_threshold = float(rl_cfg.get("lr_kl_threshold", 0.008))
    per_uid_scheduler_kwargs = {aid: {"kl_threshold": lr_kl_threshold} for aid in possible_agents}
    dev = env.device

    per_uid_obs_kwargs = {
        aid: {"size": observation_spaces[aid].shape[0], "device": dev, "epsilon": norm_eps, "clip_threshold": norm_clip}
        for aid in possible_agents
    }
    state_dim = env_cfg.state_space
    per_uid_state_kwargs = {
        aid: {"size": state_dim, "device": dev, "epsilon": norm_eps, "clip_threshold": norm_clip}
        for aid in possible_agents
    }
    per_uid_value_kwargs = {
        aid: {"size": 1, "device": dev, "epsilon": norm_eps, "clip_threshold": norm_clip} for aid in possible_agents
    }
    mappo_cfg = {
        "rollouts": int(rl_cfg.get("rollouts", 16)),
        "learning_epochs": int(rl_cfg.get("learning_epochs", 8)),
        "mini_batches": int(rl_cfg.get("mini_batches", 1)),
        "discount_factor": {
            uid: (gamma_decision if uid.endswith("/dec") else gamma_control) for uid in possible_agents
        },
        "gae_lambda": float(rl_cfg.get("gae_lambda", 0.95)),
        "learning_rate": (float(rl_cfg.get("policy_lr", 3e-4)), float(rl_cfg.get("critic_lr", 3e-4))),
        "learning_rate_scheduler": KLAdaptiveLR if lr_kl_threshold > 0 else None,
        "learning_rate_scheduler_kwargs": per_uid_scheduler_kwargs,
        "observation_preprocessor": RunningStandardScaler if norm_enabled else None,
        "observation_preprocessor_kwargs": per_uid_obs_kwargs,
        "state_preprocessor": RunningStandardScaler if (norm_enabled and _critic_state != "token") else None,
        "state_preprocessor_kwargs": per_uid_state_kwargs,
        "value_preprocessor": RunningStandardScaler if value_norm else None,
        "value_preprocessor_kwargs": per_uid_value_kwargs,
        "random_timesteps": 0,
        "learning_starts": int(rl_cfg.get("learning_starts", 0)),
        "grad_norm_clip": float(rl_cfg.get("grad_norm_clip", 1.0)),
        "ratio_clip": float(rl_cfg.get("ratio_clip", 0.2)),
        "value_clip": float(rl_cfg.get("value_clip", 0.2)),
        "entropy_loss_scale": {
            uid: (
                float(rl_cfg.get("entropy_loss_scale", 0.0))
                if uid.endswith("/dec")
                else float(rl_cfg.get("entropy_loss_scale_ctrl", 0.0))
            )
            for uid in possible_agents
        },
        "value_loss_scale": float(rl_cfg.get("value_loss_scale", 2.0)),
        "kl_threshold": float(rl_cfg.get("kl_early_stop", 0.0)),
        "time_limit_bootstrap": bool(rl_cfg.get("time_limit_bootstrap", False)),
        "rewards_shaper": None,
        "experiment": experiment,
    }
    agent = MAPPO(
        possible_agents=possible_agents,
        models=models,
        memories=memories,
        observation_spaces=observation_spaces,
        state_spaces=state_spaces,
        action_spaces=action_spaces,
        device=env.device,
        cfg=mappo_cfg,
        team_of=team_of,
        team_reward_aggr=env_cfg.team_reward_aggr,
        hierarchical_cfg=hierarchical_cfg,
        actor_mode=rl_cfg.get("actor_mode", "per_agent"),
    )
    print(
        f"[TAD][build] MAPPO hier agents={possible_agents}  "
        f"norm(obs/state)={'on' if norm_enabled else 'off'} norm(value)={'on' if value_norm else 'off'} "
        f"P={_P} gamma_c={gamma_control} gamma_d={gamma_decision} "
        f"critic_state={_critic_state} "
        f"learning_starts={mappo_cfg['learning_starts']} time_limit_bootstrap={mappo_cfg['time_limit_bootstrap']} "
        f"rollouts_dec={rl_cfg.get('rollouts_dec')} "
        f"expert_curriculum={'on' if hierarchical_cfg['expert_curriculum_enable'] else 'off'} "
        f"prob_fixed={hierarchical_cfg['expert_curriculum_prob']}(仅anneal=0时生效) "
        f"anneal_frac={hierarchical_cfg['expert_curriculum_anneal_frac']}(>0→纯退火1→0)"
    )
    return agent
