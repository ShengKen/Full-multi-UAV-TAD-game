from __future__ import annotations


def update_roll_rotation(agent, timestep: int) -> None:
    """Rotate the active training teams at the configured interval."""

    if not getattr(agent, "_roll_mode", False):
        return
    period = int(getattr(agent, "_roll_period", 512))
    order = list(getattr(agent, "_roll_order", ["T", "A", "D"]) or ["T", "A", "D"])
    depth = str(getattr(agent, "_roll_freeze_depth", "full"))
    if period <= 0 or not order:
        return

    t_rot = max(int(timestep) - int(getattr(agent, "_roll_origin", 0)), 0)
    active = order[(t_rot // period) % len(order)]
    prev = agent._roll_active_team
    agent._roll_active_team = active

    team_of = agent._team_of
    static = set(getattr(agent, "_roll_static_frozen", set()))

    rotating = {uid for uid in agent.possible_agents if team_of.get(uid) != active}
    agent.frozen_agents = static | rotating

    agent._roll_full_frozen_teams = {t for t in set(team_of.values()) if t != active} if depth == "full" else set()

    if prev != active:
        agent.track_data("Roll / active_team_idx", float(order.index(active)))

        if getattr(agent, "_roll_exper", False):
            setter = getattr(getattr(agent, "_hier_env", None), "set_exper_teams", None)
            if setter is not None:
                setter(
                    {t for t in set(team_of.values()) if t != active},
                    dict(getattr(agent, "_roll_exper_rate", {}) or {}),
                )
