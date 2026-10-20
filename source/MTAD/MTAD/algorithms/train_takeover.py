from __future__ import annotations

import os

_ROTATING = ("roll",)


def effective_train_phase(agent, t_next: int) -> tuple[str, int]:
    side = getattr(agent, "_train_side_mode", "all")
    takeover = getattr(agent, "_takeover_mode", None)
    step = int(getattr(agent, "_takeover_step", 0))
    if takeover is None or t_next < step:
        return side, 0
    return takeover, (0 if side in _ROTATING else max(step, 0))


def apply_train_mode(agent, mode: str, rot_origin: int = 0, t_mark: int = 0) -> None:
    prev = getattr(agent, "_takeover_current_mode", None)
    setter = getattr(getattr(agent, "_hier_env", None), "set_exper_teams", None)
    team_of = agent._team_of
    agent_teams = set(team_of.values())
    static = set(getattr(agent, "_roll_static_frozen", set()))
    depth = str(getattr(agent, "_roll_freeze_depth", "full"))

    if mode in ("T", "A", "D"):
        agent._roll_mode = False
        agent._roll_exper = False
        agent.frozen_agents = static | {u for u in agent.possible_agents if team_of.get(u) != mode}
        agent._roll_full_frozen_teams = {t for t in agent_teams if t != mode} if depth == "full" else set()
        if setter is not None:
            setter({t for t in agent_teams if t != mode})
    elif mode == "all":
        agent._roll_mode = False
        agent._roll_exper = False
        agent.frozen_agents = set(static)
        agent._roll_full_frozen_teams = set()
        if setter is not None:
            setter(set())
    else:
        rate = dict(getattr(agent, "_roll_exper_rate", {}) or {})
        agent._roll_mode = True
        agent._roll_exper = any(float(v) > 0.0 for v in rate.values())
        agent._roll_origin = int(rot_origin)
        agent._roll_active_team = None
        period = int(getattr(agent, "_roll_period", 512))
        order = list(getattr(agent, "_roll_order", ["T", "A", "D"]) or ["T", "A", "D"])
        if period > 0 and order and setter is not None:
            active0 = order[(max(int(t_mark) - int(rot_origin), 0) // period) % len(order)]
            if agent._roll_exper:
                setter({t for t in agent_teams if t != active0}, rate)
            else:
                setter(set())

        cur_switch = getattr(getattr(agent, "_hier_env", None), "set_expert_curriculum_enable", None)
        if agent._roll_exper and cur_switch is not None:
            cur_switch(False)

    agent._takeover_current_mode = mode
    if prev is not None and prev != mode:
        print(f"[TAD][takeover] step={int(t_mark)} 切换训练模式：{prev} → {mode}")


def init_train_mode(agent, completed_controls: int) -> None:
    t_next = max(int(completed_controls), 0)
    mode, origin = effective_train_phase(agent, t_next)
    apply_train_mode(agent, mode, origin, t_mark=t_next)
    agent._takeover_applied = (mode, origin)
    takeover = getattr(agent, "_takeover_mode", None)
    if takeover is not None:
        print(
            f"[TAD][takeover] 起始训练模式={mode}"
            f"（step>={int(getattr(agent, '_takeover_step', 0))} 切换到 {takeover}；"
            f"当前 completed_controls={t_next}）"
        )


def _save_takeover_checkpoint(agent, t_mark: int) -> None:
    ckpt_dir = os.path.join(agent.experiment_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"agent_{int(t_mark)}.pt")
    agent.save(path)
    print(f"[TAD][takeover] 已保存切换点 checkpoint（step={int(t_mark)}）：{path}")


def update_train_takeover(agent, timestep: int) -> None:
    if getattr(agent, "_takeover_mode", None) is None:
        return
    t_next = int(timestep) + 1
    mode, origin = effective_train_phase(agent, t_next)
    if (mode, origin) == getattr(agent, "_takeover_applied", None):
        return
    _save_takeover_checkpoint(agent, t_next)
    apply_train_mode(agent, mode, origin, t_mark=t_next)
    agent._takeover_applied = (mode, origin)
    if getattr(agent, "training", False):
        agent.track_data("Takeover / phase", 1.0 if mode == agent._takeover_mode else 0.0)
