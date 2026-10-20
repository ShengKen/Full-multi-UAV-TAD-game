# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import csv
import fcntl
import json
import os
import tempfile
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def atomic_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, encoding="utf-8", delete=False) as stream:
        temporary = stream.name
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def find_run_root(path: str | None) -> Path | None:
    if not path:
        return None
    for parent in Path(path).resolve().parents:
        if (parent / "run_config.yaml").is_file():
            return parent
    return None


@contextmanager
def _manifest(run_root: str | Path):
    directory = Path(run_root) / "play_eval"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "wandb_offsets.json"
    with (directory / "wandb_offsets.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        entries = json.loads(path.read_text()) if path.exists() else []
        if not isinstance(entries, list) or any(not isinstance(entry, dict) for entry in entries):
            raise ValueError(f"Invalid offset manifest: {path}")
        for entry in entries:
            for key in ("offset", "end", "reserved_end"):
                if key in entry and (not isinstance(entry[key], int) or entry[key] < 0):
                    raise ValueError(f"Invalid {key} in offset manifest: {path}")
        yield path, entries
        atomic_json(path, entries)


def reserve_offset(run_root: str | Path, requested: str, steps: int, metadata: dict, minimum_offset: int = 0) -> dict:
    if steps <= 0:
        raise ValueError("Offset reservations need a positive step budget")
    with _manifest(run_root) as (_, entries):
        maximum = max((max(e.get("end", -1), e.get("reserved_end", -1)) for e in entries), default=-1)
        offset = max(minimum_offset, maximum + 20_000 if maximum >= 0 else 0) if requested == "auto" else int(requested)
        if offset < 0:
            raise ValueError("wandb_step_offset must be non-negative")
        end = offset + steps
        for entry in entries:
            old_end = max(entry.get("end", -1), entry.get("reserved_end", -1))
            if offset < old_end and end > entry.get("offset", 0):
                raise ValueError(f"Offset interval {offset}:{end} overlaps an existing play segment")
        entry = {
            **metadata,
            "id": uuid.uuid4().hex,
            "offset": offset,
            "reserved_end": end,
            "end": offset,
            "status": "reserved",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        entries.append(entry)
        return entry


def update_offset(run_root: str | Path, reservation_id: str, **fields) -> None:
    with _manifest(run_root) as (_, entries):
        entry = next(e for e in entries if e.get("id") == reservation_id)
        expected = fields.pop("expected_status", None)
        if expected is not None and entry.get("status") != expected:
            raise ValueError("Offset reservation has already been consumed")
        if "end" in fields and fields["end"] > entry["reserved_end"]:
            raise ValueError("Play exceeded its reserved offset interval")
        entry.update(fields)


def upload_play_eval_records(wandb_run, directory: str | Path, run_root: str | Path | None = None) -> list[str]:
    if wandb_run is None:
        return []

    directory = Path(directory).resolve()
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))

    relative = Path("play_eval") / "sessions" / uuid.uuid4().hex
    if run_root is not None:
        run_root = Path(run_root).resolve()
        if directory.is_relative_to(run_root):
            candidate = directory.relative_to(run_root)
            if len(candidate.parts) > 1 and candidate.parts[0] == "play_eval":
                relative = candidate

    records = {relative / "summary.json": summary}
    if run_root is not None:
        manifest = json.loads((run_root / "play_eval/wandb_offsets.json").read_text(encoding="utf-8"))
        records[relative / "wandb_offsets.json"] = manifest
        records[Path("play_eval/wandb_offsets.json")] = manifest

    staging_root = Path(wandb_run.dir).resolve()
    uploaded = []
    for relative_path, contents in records.items():
        destination = staging_root / relative_path
        atomic_json(destination, contents)
        saved = wandb_run.save(str(destination), base_path=str(staging_root), policy="now")
        if not saved:
            raise RuntimeError(f"W&B did not accept evaluation record: {relative_path.as_posix()}")
        uploaded.append(relative_path.as_posix())
    return uploaded


def _scalar(value):
    return value.item() if hasattr(value, "item") else value


def decision_record(
    *,
    uid: str,
    selected: int,
    proposed: int | None,
    boundary: bool,
    old_count: int,
    target_alive: bool,
    own_alive: bool,
    takeover: bool,
    takeover_changed: bool,
    lock_count: int,
) -> dict:
    event = boundary and own_alive
    trigger = (2 if takeover_changed else 1 if old_count > 0 and not target_alive else 0) if event else None
    gap = abs(proposed - selected) if proposed is not None and own_alive else None
    return {
        "uid": uid,
        "env_index": 0,
        "select_idx": selected,
        "proposed_idx": proposed if own_alive else None,
        "proposal_gap": gap,
        "override": int(bool(event and gap)),
        "takeover": int(takeover),
        "decision_source": "expert" if takeover else "model" if proposed is not None else "scripted",
        "boundary": int(event),
        "relock": selected if event else None,
        "relock_trigger": trigger,
        "alive_before": int(own_alive),
        "lock_count": lock_count,
    }


class DecisionRecorder:
    def __init__(self, env, directory: str | Path):
        self.env = env
        self.rows = {}
        self.previous_takeover = {}
        self.files = []
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        fields = [
            "play_step",
            "episode",
            "episode_step",
            "episode_end",
            "uid",
            "env_index",
            "select_idx",
            "proposed_idx",
            "proposal_gap",
            "override",
            "takeover",
            "decision_source",
            "boundary",
            "relock",
            "relock_trigger",
            "alive_before",
            "lock_count",
        ]
        self.writers = []
        for name in ("decisions.csv", "relock_events.csv"):
            stream = (directory / name).open("x", newline="", encoding="utf-8")
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            self.files.append(stream)
            self.writers.append(writer)

    def capture(self, actions, need, old_counts, locked_alive, takeover_masks, force_scripted):
        env = self.env
        rows = {}
        for index, uid in enumerate(env._all_uids):
            team = uid.split("_")[0]
            aid = f"{uid}/dec"
            mask = takeover_masks.get(team)
            takeover = bool(_scalar(env._frozen_env[team][0])) or bool(mask is not None and _scalar(mask[0]))
            takeover |= bool(env._staged["ex_on"] and team != "T" and _scalar(env._expert_env[0]))
            takeover |= force_scripted
            proposed = int(_scalar(actions[aid][0].reshape(-1)[0])) if aid in actions and not force_scripted else None
            rows[uid] = decision_record(
                uid=uid,
                selected=int(_scalar(env._eff_dec_idx[uid][0])),
                proposed=proposed,
                boundary=bool(_scalar(need[0, index])),
                old_count=int(_scalar(old_counts[0, index])),
                target_alive=bool(_scalar(locked_alive[0, index])),
                own_alive=bool(_scalar(env._alive[uid][0])),
                takeover=takeover,
                takeover_changed=uid in self.previous_takeover and self.previous_takeover[uid] != takeover,
                lock_count=int(_scalar(env._lock_count[uid][0])),
            )
            if rows[uid]["boundary"]:
                self.previous_takeover[uid] = takeover
        self.rows = rows

    def record(self, play_step: int, episode: int, episode_step: int, done: bool) -> dict:
        metrics = {}
        for uid, row in self.rows.items():
            full = {
                "play_step": play_step,
                "episode": episode,
                "episode_step": episode_step,
                "episode_end": int(done),
                **row,
            }
            self.writers[0].writerow(full)
            if row["boundary"]:
                self.writers[1].writerow(full)
            for key in (
                "select_idx",
                "proposed_idx",
                "proposal_gap",
                "override",
                "takeover",
                "boundary",
                "relock",
                "relock_trigger",
                "alive_before",
            ):
                if row[key] is not None:
                    metrics[f"play/{uid}/dec/{key}"] = row[key]
            metrics[f"play/{uid}/locked_pool"] = row["select_idx"]
            metrics[f"play/{uid}/lock_count"] = row["lock_count"]
        if done:
            self.previous_takeover.clear()
        if done or play_step % 100 == 0:
            for stream in self.files:
                stream.flush()
        return metrics

    def close(self):
        for stream in self.files:
            stream.close()
