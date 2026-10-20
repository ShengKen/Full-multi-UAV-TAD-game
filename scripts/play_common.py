# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pxr import Usd


_TEAM_COLORS = {"T": (1.0, 0.0, 0.0), "A": (0.0, 1.0, 0.0), "D": (0.0, 0.0, 1.0)}


RECORDING_GROUND_TINT = (0.90, 0.96, 1.0)
RECORDING_GROUND_ALBEDO_ADD = 0.15


def lighten_recording_ground(
    *, prim_path: str = "/World/ground", stage: Usd.Stage | None = None
) -> dict[str, list[float] | float]:
    from pxr import Gf, Sdf, UsdShade

    if stage is None:
        import omni.usd

        stage = omni.usd.get_context().get_stage()
    shader_prim = stage.GetPrimAtPath(f"{prim_path}/Looks/theGrid/Shader")
    if not shader_prim.IsValid():
        raise RuntimeError(f"Isaac grid ground shader not found: {prim_path}/Looks/theGrid/Shader")
    shader = UsdShade.Shader(shader_prim)
    shader.CreateInput("diffuse_tint", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*RECORDING_GROUND_TINT))

    shader.CreateInput("albedo_add", Sdf.ValueTypeNames.Float).Set(RECORDING_GROUND_ALBEDO_ADD)
    return {"diffuse_tint": list(RECORDING_GROUND_TINT), "albedo_add": RECORDING_GROUND_ALBEDO_ADD}


def _team_of_uid(uid: str) -> str:
    return uid.split("/")[0].split("_")[0]


class DroneShadowLines:
    def __init__(
        self,
        base_env,
        *,
        prim_path: str = "/World/Visuals/DroneShadowLines",
        radius: float = 0.02,
        env_index: int = 0,
        trail_every: int = 3,
        trail_max_points: int = 400,
        trail_min_seg: float = 0.05,
    ):
        from collections import deque

        import numpy as np
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        self.base_env = base_env
        self.env_index = int(env_index)
        self._np = np

        self.uids = list(base_env.drones.keys())
        self._teams = [_team_of_uid(u) for u in self.uids]

        teams_present = []
        for t in self._teams:
            if t not in teams_present:
                teams_present.append(t)
        markers_cfg = {
            t: sim_utils.CylinderCfg(
                radius=radius,
                height=1.0,
                axis="Z",
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=_TEAM_COLORS.get(t, (0.5, 0.5, 0.5))),
            )
            for t in teams_present
        }
        self._proto_index = {t: i for i, t in enumerate(teams_present)}
        cfg = VisualizationMarkersCfg(prim_path=prim_path, markers=markers_cfg)
        self._markers = VisualizationMarkers(cfg)
        self._markers.set_visibility(True)

        self._trail_every = max(1, int(trail_every))
        self._trail_max = max(2, int(trail_max_points))
        self._trail_min_seg = float(trail_min_seg)
        self._update_count = 0
        self._trails = {u: deque(maxlen=self._trail_max) for u in self.uids}

    def _alive(self, uid: str) -> bool:
        return bool(self.base_env._alive[uid][self.env_index])

    def _clear_markers(self) -> None:
        np = self._np
        self._markers.visualize(
            translations=np.empty((0, 3), dtype=np.float32),
            orientations=np.empty((0, 4), dtype=np.float32),
            scales=np.empty((0, 3), dtype=np.float32),
            marker_indices=np.empty((0,), dtype=np.int32),
        )

    def reset(self) -> None:
        for buf in self._trails.values():
            buf.clear()
        self._clear_markers()

    def update(self) -> None:
        np = self._np
        ei = self.env_index

        alive_now = {u: self._alive(u) for u in self.uids}
        cur = {}
        for u in self.uids:
            if not alive_now[u]:
                continue
            p = self.base_env.drones[u].data.root_pos_w[ei].detach().cpu().numpy()
            cur[u] = (float(p[0]), float(p[1]), float(p[2]))

        self._update_count += 1
        if self._update_count % self._trail_every == 0:
            for u in self.uids:
                if alive_now[u]:
                    self._trails[u].append(cur[u])

        trans, scales, oris, idx = [], [], [], []
        z_axis = np.array([0.0, 0.0, 1.0])

        for u, team in zip(self.uids, self._teams):
            if not alive_now[u]:
                continue
            x, y, z = cur[u]
            if z <= 0.0:
                continue
            trans.append((x, y, z * 0.5))
            scales.append((1.0, 1.0, z))
            oris.append((1.0, 0.0, 0.0, 0.0))
            idx.append(self._proto_index[team])

        for u, team in zip(self.uids, self._teams):
            buf = self._trails[u]
            if len(buf) < 2:
                continue
            arr = np.asarray(buf, dtype=np.float64)
            p0, p1 = arr[:-1], arr[1:]
            seg = p1 - p0
            length = np.linalg.norm(seg, axis=1)
            keep = length > self._trail_min_seg
            if not np.any(keep):
                continue
            dhat = seg[keep] / length[keep, None]
            mid = ((p0 + p1) * 0.5)[keep]
            seglen = length[keep]

            cos_t = np.clip(dhat[:, 2], -1.0, 1.0)
            cos_half = np.sqrt(np.maximum(0.0, (1.0 + cos_t) * 0.5))
            sin_half = np.sqrt(np.maximum(0.0, (1.0 - cos_t) * 0.5))
            axis = np.cross(z_axis, dhat)
            an = np.linalg.norm(axis, axis=1, keepdims=True)
            an = np.where(an < 1e-9, 1.0, an)
            axis_u = axis / an

            anti = cos_t <= -0.999999
            qx = sin_half[:, None] * axis_u
            qx[anti] = np.array([1.0, 0.0, 0.0])
            qw = cos_half.copy()
            qw[anti] = 0.0
            quat = np.concatenate([qw[:, None], qx], axis=1)
            n = mid.shape[0]
            pi = self._proto_index[team]
            trans.extend(mid.tolist())
            scales.extend([(1.0, 1.0, float(L)) for L in seglen])
            oris.extend(quat.tolist())
            idx.extend([pi] * n)

        if not trans:
            self._clear_markers()
            return
        self._markers.visualize(
            translations=np.asarray(trans, dtype=np.float32),
            orientations=np.asarray(oris, dtype=np.float32),
            scales=np.asarray(scales, dtype=np.float32),
            marker_indices=np.asarray(idx, dtype=np.int32),
        )


def parse_floats(value: str) -> tuple[float, float, float]:
    values = tuple(float(x) for x in str(value).replace(" ", "").split(","))
    if len(values) != 3:
        raise ValueError(f"需要 3 个逗号分隔的数，得到：{value}")
    return values


def episode_recording_path(base_video_path: str, episode_index: int) -> str:
    if episode_index <= 0:
        return base_video_path
    stem, ext = os.path.splitext(base_video_path)
    return f"{stem}_episode{episode_index + 1}{ext or '.mp4'}"


def save_episode(frames, out_path: str, fps: int, tag: str) -> None:
    import imageio

    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(out_path)[0]
    mp4_path = stem + ".mp4"
    height, width = frames[0].shape[:2]

    try:
        imageio.mimsave(
            mp4_path,
            frames,
            format="FFMPEG",
            fps=fps,
            codec="libx264",
            quality=10,
            macro_block_size=1,
            output_params=["-preset", "slow", "-pix_fmt", "yuv420p"],
        )
        print(f"[{tag}] ✔ MP4 已保存：{mp4_path}（{len(frames)} 帧, {fps} fps, {width}x{height}）")
    except Exception as exc:
        print(f"[{tag}] ✗ MP4 写入失败：{exc}")


def setup_play_wandb(
    *,
    enabled: bool,
    wandb_cfg: dict,
    train_cfg_dir: str | None,
    default_project: str,
    run_name: str,
    tag: str,
    config: dict | None = None,
):
    if not enabled:
        return None
    try:
        import wandb

        init_kwargs = {
            "project": wandb_cfg.get("project", default_project),
            "entity": wandb_cfg.get("entity") or None,
            "mode": wandb_cfg.get("mode", "online"),
            "dir": os.path.abspath("logs"),
        }

        if "finish_timeout" in getattr(wandb.Settings, "model_fields", {}):
            init_kwargs["settings"] = wandb.Settings(
                finish_timeout=60,
                finish_timeout_raises=False,
                x_file_stream_max_bytes=1024 * 1024,
                x_file_stream_timeout_seconds=30,
            )
        if config is not None:
            init_kwargs["config"] = config
        resume_id = wandb_cfg.get("run_id") or (os.path.basename(train_cfg_dir) if train_cfg_dir else None)
        if resume_id:
            init_kwargs.update(id=resume_id, name=resume_id, resume="allow")
            print(f"[{tag}] wandb resume 训练 run id={resume_id}")
        else:
            init_kwargs["name"] = run_name
        run = wandb.init(**init_kwargs)
        wandb.define_metric("play/play_step")

        wandb.define_metric("play/*", step_metric="play/play_step")
        return run
    except Exception as exc:
        print(f"[{tag}] wandb 不可用，跳过：{exc}")
        return None
