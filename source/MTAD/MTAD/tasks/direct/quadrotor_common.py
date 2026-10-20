# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse, quat_from_euler_xyz


def _wrap_to_pi(a: torch.Tensor) -> torch.Tensor:
    return (a + math.pi) % (2 * math.pi) - math.pi


_PPN_SPEED_KV = 2.0


class QuadrotorEnvMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _spawn_env_grid_dividers(self) -> None:
        spacing = float(self.cfg.scene.env_spacing)
        half = spacing / 2.0
        thin = 0.15
        z = 0.05
        material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.9, 0.9), roughness=0.6, metallic=0.0)
        base = "/World/envs/env_0/EnvDivider"
        edges = {
            "north": ((spacing, thin, 0.1), (0.0, half, z)),
            "south": ((spacing, thin, 0.1), (0.0, -half, z)),
            "east": ((thin, spacing, 0.1), (half, 0.0, z)),
            "west": ((thin, spacing, 0.1), (-half, 0.0, z)),
        }
        for side, (size, translation) in edges.items():
            cfg = sim_utils.CuboidCfg(size=size, visual_material=material)
            sim_utils.spawn_cuboid(f"{base}/{side}", cfg, translation=translation)

    def compute_warmup_stabilize_action(self, uid: str) -> torch.Tensor:
        drone = self.drones[uid]
        roll, pitch, _ = euler_xyz_from_quat(drone.data.root_quat_w)
        gain = self.cfg.warmup_k_att
        action = torch.zeros(roll.shape[0], 4, device=self.device)
        action[:, 1] = (-gain * roll / self.cfg.rate_max).clamp(-1.0, 1.0)
        action[:, 2] = (-gain * pitch / self.cfg.rate_max).clamp(-1.0, 1.0)
        return action

    def _enforce_dead_drone_state(self, uid: str) -> torch.Tensor:
        dead = ~self._alive[uid]

        env_ids = dead.nonzero(as_tuple=False).squeeze(-1)
        if env_ids.numel() == 0:
            return dead
        root_pose = torch.zeros(len(env_ids), 7, device=self.device)
        root_pose[:, :2] = self.scene.env_origins[env_ids, :2]
        root_pose[:, 2] = self._graveyard_z
        root_pose[:, 3] = 1.0
        self.drones[uid].write_root_pose_to_sim(root_pose, env_ids=env_ids)
        self.drones[uid].write_root_velocity_to_sim(torch.zeros(len(env_ids), 6, device=self.device), env_ids=env_ids)
        return dead

    def _apply_drone_action(self, uid: str, action: torch.Tensor, active_mask: torch.Tensor | None = None) -> None:
        drone = self.drones[uid]
        num_envs = action.shape[0]

        thrust = self._weight + action[:, 0] * (self._thrust_max - self._weight)
        thrust = torch.clamp(thrust, min=0.0)
        forces = torch.zeros(num_envs, 1, 3, device=self.device)
        forces[:, 0, 2] = thrust

        k_drag = self._drag_k_for_uid(uid)
        if k_drag > 0.0:
            v_w = drone.data.root_lin_vel_w
            v_b = quat_apply_inverse(drone.data.root_quat_w, v_w)
            forces[:, 0, :] -= (self._weight / 9.81) * k_drag * v_w.norm(dim=-1, keepdim=True) * v_b

        if self.cfg.action_mode == "rates":
            desired_rates = action[:, 1:4] * self.cfg.rate_max
            torque = self.cfg.rate_kp * (desired_rates - drone.data.root_ang_vel_b)
            torque = torch.clamp(torque, -self.cfg.moment_max, self.cfg.moment_max)
            torques = torque.unsqueeze(1)
        else:
            torques = (action[:, 1:4] * self.cfg.moment_max).unsqueeze(1)

        if active_mask is not None:
            active = active_mask.bool().view(num_envs, 1, 1)
            forces = torch.where(active, forces, torch.zeros_like(forces))
            torques = torch.where(active, torques, torch.zeros_like(torques))

        drone.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._body_ids[uid], forces=forces, torques=torques
        )

    def _compute_cascade_target_action_batch(
        self,
        uid_order: list[str],
        own_pos: torch.Tensor,
        own_vel: torch.Tensor,
        own_quat: torch.Tensor,
        tgt_pos: torch.Tensor,
        tgt_alive: torch.Tensor,
        sign: torch.Tensor | None = None,
    ) -> torch.Tensor:
        c = self.cfg
        N, M = own_pos.shape[0], own_pos.shape[1]
        dev = own_pos.device
        q = own_quat.reshape(N * M, 4)
        roll, pitch, yaw = euler_xyz_from_quat(q)
        roll = roll.reshape(N, M)
        pitch = pitch.reshape(N, M)
        yaw = yaw.reshape(N, M)

        d = tgt_pos - own_pos
        if sign is None:
            is_T = torch.tensor([u.split("_")[0] == "T" for u in uid_order], device=dev).view(1, M, 1)
            s = torch.where(is_T, -1.0, 1.0).expand(N, M, 1)
        else:
            s = sign.to(dev).view(N, M, 1)
        dir_xy = s * d[:, :, :2]
        dir_xy = dir_xy / (dir_xy.norm(dim=-1, keepdim=True) + 1e-6)
        is_evader = s[:, :, 0] < 0

        origins_xy = self.scene.env_origins[:, :2]
        pos_xy = own_pos[:, :, :2] - origins_xy.unsqueeze(1)
        horiz = pos_xy.norm(dim=-1)
        soft_h = c.arena_radius - c.oob_horiz_margin
        over_h = torch.clamp(horiz - soft_h, min=0.0, max=c.oob_horiz_cap)
        alpha_h = over_h / c.oob_horiz_cap
        dir_inward = -pos_xy / (horiz.unsqueeze(-1) + 1e-6)
        dir_pursuit = (1.0 - alpha_h).unsqueeze(-1) * dir_xy + alpha_h.unsqueeze(-1) * dir_inward
        dir_pursuit = dir_pursuit / (dir_pursuit.norm(dim=-1, keepdim=True) + 1e-6)
        r_hat = -dir_inward
        alpha_b = torch.clamp((horiz - (soft_h - c.evader_wall_margin)) / c.evader_wall_margin, min=0.0, max=1.0)
        out_r = (dir_xy * r_hat).sum(dim=-1, keepdim=True).clamp(0.0, 1.0 - c.evader_tangent_min)
        dir_ev = dir_xy - out_r * r_hat - (c.evader_inward_beta * alpha_b).unsqueeze(-1) * r_hat
        dir_ev = dir_ev / (dir_ev.norm(dim=-1, keepdim=True) + 1e-6)
        dir_xy = torch.where(is_evader.unsqueeze(-1), dir_ev, dir_pursuit)

        psi_des = torch.atan2(dir_xy[:, :, 1], dir_xy[:, :, 0])
        dpsi = _wrap_to_pi(psi_des - yaw)
        a_r = (c.heur_k_yaw * dpsi / c.rate_max).clamp(-1.0, 1.0)

        vlim = torch.tensor([self._vel_limit_for_uid(uid) for uid in uid_order], device=dev).view(1, M, 1)
        v_des = dir_xy * vlim
        speed_xy = own_vel[:, :, :2].norm(dim=-1)
        over_v = (speed_xy - vlim.squeeze(-1)).clamp(min=0.0)
        v_des_decel = (own_vel[:, :, :2] / (speed_xy.unsqueeze(-1) + 1e-6)) * vlim
        v_des = torch.where(over_v.unsqueeze(-1) > 0, v_des_decel, v_des)

        cp, sp = torch.cos(yaw), torch.sin(yaw)

        v_des_bx = cp * v_des[:, :, 0] + sp * v_des[:, :, 1]
        v_des_by = -sp * v_des[:, :, 0] + cp * v_des[:, :, 1]
        vx_b = cp * own_vel[:, :, 0] + sp * own_vel[:, :, 1]
        vy_b = -sp * own_vel[:, :, 0] + cp * own_vel[:, :, 1]
        e_fwd = v_des_bx - vx_b
        e_lat = v_des_by - vy_b
        align = torch.clamp(1.0 - dpsi.abs() / math.pi, 0.0, 1.0)
        theta_des = (c.heur_k_pitch * e_fwd * align).clamp(-c.heur_max_lean, c.heur_max_lean)
        phi_des = (-c.heur_k_roll * e_lat * align).clamp(-c.heur_max_lean, c.heur_max_lean)

        p_des = c.heur_k_att * (phi_des - roll)
        q_des = c.heur_k_att * (theta_des - pitch)
        a_p = (p_des / c.rate_max).clamp(-1.0, 1.0)
        a_q = (q_des / c.rate_max).clamp(-1.0, 1.0)

        soft_zmin = c.arena_z_min + c.oob_z_margin
        soft_zmax = c.arena_z_max - c.oob_z_margin
        z_lo = soft_zmin + c.evader_z_margin
        z_hi = soft_zmax - c.evader_z_margin
        z_ev = own_pos[:, :, 2] + c.evader_k_z * (own_pos[:, :, 2] - tgt_pos[:, :, 2])
        z_des_ev = z_ev.clamp(z_lo, z_hi)
        e_z = torch.where(is_evader, z_des_ev - own_pos[:, :, 2], tgt_pos[:, :, 2] - own_pos[:, :, 2])
        over_zlow = torch.clamp(soft_zmin - own_pos[:, :, 2], min=0.0, max=c.oob_z_cap)
        over_zhigh = torch.clamp(own_pos[:, :, 2] - soft_zmax, min=0.0, max=c.oob_z_cap)
        alpha_z = torch.clamp((over_zlow + over_zhigh) / c.oob_z_cap, min=0.0, max=1.0)
        a_thrust = (
            c.heur_k_alt * e_z * (1.0 - alpha_z)
            - c.heur_k_dz * own_vel[:, :, 2]
            + c.heur_k_alt * (over_zlow - over_zhigh)
        ).clamp(-1.0, 1.0)

        action = torch.stack([a_thrust, a_p, a_q, a_r], dim=-1)
        if c.heur_noise_std > 0.0:
            action = (action + c.heur_noise_std * torch.randn_like(action)).clamp(-1.0, 1.0)

        hover = self._hover_action.view(1, 1, 4)
        return torch.where(tgt_alive.unsqueeze(-1), action, hover)

    def _compute_ppn_target_action_batch(
        self,
        uid_order: list[str],
        own_pos: torch.Tensor,
        own_vel: torch.Tensor,
        own_quat: torch.Tensor,
        tgt_pos: torch.Tensor,
        tgt_vel: torch.Tensor,
        tgt_alive: torch.Tensor,
    ) -> torch.Tensor:
        c = self.cfg
        N, M = own_pos.shape[0], own_pos.shape[1]
        dev = own_pos.device
        q = own_quat.reshape(N * M, 4)
        roll, pitch, yaw = euler_xyz_from_quat(q)
        roll = roll.reshape(N, M)
        pitch = pitch.reshape(N, M)
        yaw = yaw.reshape(N, M)

        r = tgt_pos - own_pos
        rdot = tgt_vel - own_vel
        omega = torch.cross(r, rdot, dim=-1) / (r.pow(2).sum(-1, keepdim=True) + 1e-6)
        a_lat = c.ppn_nav_const * torch.cross(omega, own_vel, dim=-1)
        a_xy = a_lat[:, :, :2]

        vlim = torch.tensor([self._vel_limit_for_uid(uid) for uid in uid_order], device=dev).view(1, M, 1)
        speed_xy = own_vel[:, :, :2].norm(dim=-1, keepdim=True)
        v_hat = own_vel[:, :, :2] / (speed_xy + 1e-6)
        nose_xy = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)
        v_hat = torch.where(speed_xy < 0.5, nose_xy, v_hat)
        a_xy = a_xy + _PPN_SPEED_KV * (vlim - speed_xy) * v_hat

        origins_xy = self.scene.env_origins[:, :2]
        pos_xy = own_pos[:, :, :2] - origins_xy.unsqueeze(1)
        horiz = pos_xy.norm(dim=-1)
        soft_h = c.arena_radius - c.oob_horiz_margin
        over_h = torch.clamp(horiz - soft_h, min=0.0, max=c.oob_horiz_cap)
        alpha_h = over_h / c.oob_horiz_cap
        a_in = -pos_xy / (horiz.unsqueeze(-1) + 1e-6)

        a_xy = (1.0 - alpha_h).unsqueeze(-1) * a_xy + alpha_h.unsqueeze(-1) * a_in

        dir_xy = r[:, :, :2]
        dir_xy = dir_xy / (dir_xy.norm(dim=-1, keepdim=True) + 1e-6)
        psi_des = torch.atan2(dir_xy[:, :, 1], dir_xy[:, :, 0])
        dpsi = _wrap_to_pi(psi_des - yaw)
        a_r = (c.heur_k_yaw * dpsi / c.rate_max).clamp(-1.0, 1.0)

        cp, sp = torch.cos(yaw), torch.sin(yaw)
        a_bx = cp * a_xy[:, :, 0] + sp * a_xy[:, :, 1]
        a_by = -sp * a_xy[:, :, 0] + cp * a_xy[:, :, 1]
        align = torch.clamp(1.0 - dpsi.abs() / math.pi, 0.0, 1.0)
        theta_des = (c.accel_k_tilt * a_bx * align).clamp(-c.max_lean, c.max_lean)
        phi_des = (-c.accel_k_tilt * a_by * align).clamp(-c.max_lean, c.max_lean)

        p_des = c.accel_k_att * (phi_des - roll)
        q_des = c.accel_k_att * (theta_des - pitch)
        a_p = (p_des / c.rate_max).clamp(-1.0, 1.0)
        a_q = (q_des / c.rate_max).clamp(-1.0, 1.0)

        e_z = tgt_pos[:, :, 2] - own_pos[:, :, 2]
        soft_zmin = c.arena_z_min + c.oob_z_margin
        soft_zmax = c.arena_z_max - c.oob_z_margin
        over_zlow = torch.clamp(soft_zmin - own_pos[:, :, 2], min=0.0, max=c.oob_z_cap)
        over_zhigh = torch.clamp(own_pos[:, :, 2] - soft_zmax, min=0.0, max=c.oob_z_cap)
        alpha_z = torch.clamp((over_zlow + over_zhigh) / c.oob_z_cap, min=0.0, max=1.0)
        a_thrust = (
            c.heur_k_alt * e_z * (1.0 - alpha_z)
            - c.heur_k_dz * own_vel[:, :, 2]
            + c.heur_k_alt * (over_zlow - over_zhigh)
        ).clamp(-1.0, 1.0)

        action = torch.stack([a_thrust, a_p, a_q, a_r], dim=-1)
        hover = self._hover_action.view(1, 1, 4)
        return torch.where(tgt_alive.unsqueeze(-1), action, hover)

    def _vel_limit_for_uid(self, uid: str) -> float:
        cfg = self.cfg
        team = uid.split("_")[0]
        if team == "A":
            return cfg.max_lin_vel
        elif team == "D":
            return cfg.max_lin_vel * cfg.defender_vel_ratio
        else:
            return cfg.max_lin_vel * cfg.evader_vel_ratio

    def _drag_k_for_uid(self, uid: str) -> float:
        if not getattr(self.cfg, "drag_enable", False):
            return 0.0
        v_term = self._vel_limit_for_uid(uid)
        if v_term <= 0.0:
            return 0.0
        tilt = max(getattr(self.cfg, "max_roll", 0.0), getattr(self.cfg, "max_pitch", 0.0))
        if tilt <= 0.0:
            tilt = 0.8
        a_push = 9.81 * math.tan(tilt)
        rigid_props = getattr(self.cfg, "_drone_rigid_props", None) or {}
        lam = float(rigid_props.get("linear_damping", 0.0) or 0.0)
        k = a_push - lam * v_term
        if k <= 0.0:
            if not getattr(self, "_drag_k_anchor_warned", False):
                print(
                    f"[TAD][WARN] 阻力锚点不可达：v_term={v_term:.2f} m/s ≥ a_push/λ="
                    f"{a_push / max(lam, 1e-9):.2f} m/s（uid={uid}）。该队速度上限由线性阻尼决定，"
                    f"二次阻力置 0；请调小 scenario.max_lin_vel 或对应 vel_ratio。"
                )
                self._drag_k_anchor_warned = True
            return 0.0
        return k / (v_term * v_term)

    def _enforce_hard_attitude_limit(self, uid: str) -> None:
        max_roll = float(getattr(self.cfg, "max_roll", 0.0))
        max_pitch = float(getattr(self.cfg, "max_pitch", 0.0))
        if max_roll <= 0.0 and max_pitch <= 0.0:
            return
        drone = self.drones[uid]
        roll, pitch, yaw = euler_xyz_from_quat(drone.data.root_quat_w)
        new_roll = roll.clamp(-max_roll, max_roll) if max_roll > 0.0 else roll
        new_pitch = pitch.clamp(-max_pitch, max_pitch) if max_pitch > 0.0 else pitch
        changed = (new_roll != roll) | (new_pitch != pitch)

        env_ids = changed.nonzero(as_tuple=False).squeeze(-1)
        if env_ids.numel() == 0:
            return
        q = quat_from_euler_xyz(new_roll[env_ids], new_pitch[env_ids], yaw[env_ids])
        pos = drone.data.root_pos_w[env_ids]
        pose = torch.cat([pos, q], dim=-1)
        drone.write_root_pose_to_sim(pose, env_ids=env_ids)
