# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import quote

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
BEGIN = "<!-- BEGIN PUBLIC DEMOS -->"
END = "<!-- END PUBLIC DEMOS -->"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def checkpoint_for(run: Path, step: int | None) -> tuple[int, Path] | None:
    candidates = []
    for path in run.glob("*/checkpoints/agent_*.pt"):
        match = re.fullmatch(r"agent_(\d+)\.pt", path.name)
        if match and (step is None or int(match[1]) == step):
            candidates.append((int(match[1]), path))
    if not candidates:
        if step is not None:
            raise ValueError(f"Missing requested agent_{step}.pt in {run}")
        return None
    highest = max(item[0] for item in candidates)
    latest = [item for item in candidates if item[0] == highest]
    if len(latest) != 1:
        raise ValueError(f"Multiple checkpoints at step {highest} in {run}; resolve the ambiguity first")
    return latest[0]


def source_video(run: Path, explicit: str | None = None) -> Path | None:
    if explicit:
        path = run / explicit
        if not path.is_file():
            raise ValueError(f"Missing configured video: {path}")
        return path
    for pattern in ("play_eval/**/video_internal/*.mp4", "play/*.mp4", "play_eval/**/*.mp4"):
        videos = sorted(run.glob(pattern))
        if videos:
            return videos[0]
    return None


def inspect_run(root: Path, spec: dict) -> dict:
    name = spec["name"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError(f"Invalid run directory name: {name!r}")
    run = root / "logs" / "tad_hierarchical" / name
    entry = {"name": name, "showcase": spec.get("showcase", False), "status": "missing", "files": []}
    if not run.is_dir():
        return entry
    config = run / "run_config.yaml"
    if not config.is_file():
        raise ValueError(f"Missing experiment configuration: {config}")
    data = yaml.safe_load(config.read_text())
    scenario = data["scenario"]
    teams = [int(scenario[key]) for key in ("n_T", "n_A", "n_D")]
    if entry["showcase"] and sum(teams) != 30:
        raise ValueError(f"30 UAV showcase requires 30 UAVs, got {teams}: {name}")
    checkpoint = checkpoint_for(run, spec.get("checkpoint_step"))
    video = source_video(run, spec.get("video"))
    files = {config}
    if checkpoint:
        files.add(checkpoint[1])
    drone_config = run / "drone_config.yaml"
    if drone_config.is_file():
        files.add(drone_config)
    for directory in ("play_eval", "play"):
        files.update(path for path in (run / directory).rglob("*") if path.is_file() and path.suffix.lower() != ".mp4")
    if video:
        files.add(video)
    for path in files:
        if not path.resolve().is_relative_to(run.resolve()):
            raise ValueError(f"Artifact escapes run directory: {path}")
        if path.suffix in {".pt", ".mp4"}:
            with path.open("rb") as stream:
                if stream.read(100).startswith(b"version https://git-lfs.github.com/spec/"):
                    raise ValueError(f"Git LFS object not downloaded; run git lfs pull: {path}")
    entry.update(
        status="available",
        algorithm=data.get("rl", {}).get("algo", "unknown"),
        teams=teams,
        elimination_mode=scenario.get("elimination_mode", "unspecified"),
        checkpoint=str(checkpoint[1].relative_to(root)) if checkpoint else None,
        checkpoint_step=checkpoint[0] if checkpoint else None,
        config=str(config.relative_to(root)),
        drone_config=str(drone_config.relative_to(root)) if drone_config.is_file() else None,
        video=str(video.relative_to(root)) if video else None,
        videos=sorted(str(path.relative_to(root)) for path in files if path.suffix == ".mp4"),
        files=sorted(str(path.relative_to(root)) for path in files),
    )
    return entry


def inspect_featured_video(root: Path, spec: dict) -> dict:
    name = spec["name"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError(f"Invalid run directory name: {name!r}")
    preview_id = spec.get("preview_id", name)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", preview_id):
        raise ValueError(f"Invalid preview identifier: {preview_id!r}")
    run = root / "logs" / "tad_hierarchical" / name
    video = source_video(run, spec["video"])
    videos = [video] + [source_video(run, path) for path in spec.get("additional_videos", [])]
    for recording in videos:
        if not recording.resolve().is_relative_to(run.resolve()) or recording.suffix != ".mp4":
            raise ValueError(f"Invalid featured video: {recording}")
        with recording.open("rb") as stream:
            if stream.read(100).startswith(b"version https://git-lfs.github.com/spec/"):
                raise ValueError(f"Git LFS object not downloaded; run git lfs pull: {recording}")
    config = run / "run_config.yaml"
    data = yaml.safe_load(config.read_text())
    scenario = data["scenario"]
    path = str(video.relative_to(root))
    entry = {
        "name": name,
        "showcase": True,
        "preview_id": preview_id,
        "status": "available",
        "algorithm": data.get("rl", {}).get("algo", "unknown"),
        "teams": [int(scenario[key]) for key in ("n_T", "n_A", "n_D")],
        "elimination_mode": scenario.get("elimination_mode", "unspecified"),
        "video": path,
        "config": str(config.relative_to(root)),
        "files": sorted({str(recording.relative_to(root)) for recording in videos} | {str(config.relative_to(root))}),
    }
    if "clip_seconds" in spec:
        seconds = float(spec["clip_seconds"])
        if not 0 < seconds <= 30:
            raise ValueError("Per-video clip_seconds must be in (0, 30]")
        entry["clip_seconds"] = seconds
    return entry


def ffmpeg_binary() -> str:
    binary = shutil.which("ffmpeg")
    if binary:
        return binary
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError("Install ffmpeg or imageio-ffmpeg to prepare showcase previews") from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


def prepare_preview(root: Path, entry: dict, seconds: float) -> None:
    if not entry.get("video"):
        return
    source = root / entry["video"]
    directory = root / "media"
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{sum(entry['teams'])}uav-{entry.get('preview_id', entry['name'])}"
    clip = directory / f"{stem}.mp4"
    preview = directory / f"{stem}.gif"
    provenance = directory / f"{stem}.json"
    metadata = {
        "source": entry["video"],
        "source_sha256": digest(source),
        "start_seconds": 0,
        "maximum_duration_seconds": seconds,
        "playback_speed": 1,
        "clip_encoding": "h264, yuv420p, 960px, CRF 25, no audio",
        "preview_encoding": "first 6 seconds, 640px, 8fps, 96-color GIF",
    }
    cached = json.loads(provenance.read_text()) if provenance.exists() else None
    if cached != metadata or not clip.is_file() or not preview.is_file():
        ffmpeg = ffmpeg_binary()
        subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source),
                "-t",
                str(seconds),
                "-vf",
                "scale=960:-2",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "25",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-an",
                str(clip),
            ],
            check=True,
        )
        subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(clip),
                "-t",
                "6",
                "-filter_complex",
                "fps=8,scale=640:-2:flags=lanczos,split[a][b];"
                "[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=3",
                "-loop",
                "0",
                str(preview),
            ],
            check=True,
        )
        provenance.write_text(json.dumps(metadata, indent=2) + "\n")
    entry["clip"] = str(clip.relative_to(root))
    entry["preview"] = str(preview.relative_to(root))
    entry["files"].extend(str(path.relative_to(root)) for path in (clip, preview, provenance))


def media_url(settings: dict, path: str) -> str:
    return f"https://github.com/{settings['repository']}/blob/{settings['ref']}/{quote(path)}"


def scenario_description(entry: dict) -> str:
    targets, attackers, defenders = entry["teams"]
    text = (
        f"{targets} target UAVs evade {attackers} attacking UAVs, "
        f"while {defenders} defending UAVs intercept the attackers."
    )
    if entry.get("elimination_mode") == "paired":
        text += " A successful capture removes both participating UAVs from the game."
    elif entry.get("elimination_mode") == "catcher_survives":
        text += " A successful capture removes the captured UAV; the capturing UAV remains active."
    return text


def demo_markdown(settings: dict, entries: list[dict]) -> str:
    blocks = []
    for entry in entries:
        if not entry["showcase"]:
            continue
        if entry["status"] == "missing":
            blocks.append(f"`{entry['name']}`: run not available yet; no video link generated.")
        elif entry.get("clip"):
            blocks.append(
                f"**{sum(entry['teams'])}-UAV game · "
                f"T/A/D = {'/'.join(map(str, entry['teams']))}**\n\n"
                f"[![{sum(entry['teams'])} UAV animated preview]({entry['preview']})]"
                f"({media_url(settings, entry['clip'])})\n\n"
                f"{scenario_description(entry)}"
            )
        else:
            blocks.append(f"`{entry['name']}`: run found, but no recording is available yet.")
    return "\n\n".join(blocks) or "No demonstration videos configured yet."


def select_artifacts(
    entries: list[dict], featured: list[dict], runs: list[str] | None, featured_only: bool
) -> list[str]:
    pool = featured if featured_only else entries + featured
    return sorted({path for entry in pool if not runs or entry["name"] in runs for path in entry["files"]})


def write_readme(root: Path, settings: dict, entries: list[dict], featured: list[dict] | None = None) -> list[Path]:
    featured = featured or []
    readme = root / "README.md"
    text = readme.read_text()
    if text.count(BEGIN) != 1 or text.count(END) != 1:
        raise ValueError("README.md must contain exactly one PUBLIC DEMOS marker pair")
    before, rest = text.split(BEGIN)
    _, after = rest.split(END)
    readme.write_text(before + BEGIN + "\n" + demo_markdown(settings, entries + featured) + "\n" + END + after)
    manifest = root / "logs" / "public_artifacts" / "manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "repository": settings["repository"],
                "ref": settings["ref"],
                "runs": entries,
                "featured_videos": featured,
            },
            indent=2,
        )
        + "\n"
    )
    return [readme]


def main() -> None:
    """Prepare selected artifacts and update the homepage gallery."""
    parser = argparse.ArgumentParser(description="Prepare public models and video previews.")
    parser.add_argument("--catalog", type=Path, default=REPO_ROOT / "configs/public_artifacts.json")
    parser.add_argument("--run", action="append", help="Only stage these catalog runs; repeat to select several")
    parser.add_argument(
        "--featured-only", action="store_true", help="Select only individual featured videos and previews"
    )
    parser.add_argument("--stage", action="store_true", help="Stage selected artifacts and the generated README")
    args = parser.parse_args()
    settings = json.loads(args.catalog.read_text())
    if not 0 < settings["clip_seconds"] <= 30:
        raise ValueError("clip_seconds must be in (0, 30]")
    specs = settings["runs"] + settings.get("featured_videos", [])
    unknown = set(args.run or []) - {spec["name"] for spec in specs}
    if unknown:
        raise ValueError(f"Add these run names to the catalog first: {sorted(unknown)}")
    entries = [inspect_run(REPO_ROOT, spec) for spec in settings["runs"]]
    featured = [inspect_featured_video(REPO_ROOT, spec) for spec in settings.get("featured_videos", [])]
    for entry in entries + featured:
        if entry["status"] == "available" and entry["showcase"]:
            prepare_preview(REPO_ROOT, entry, entry.get("clip_seconds", settings["clip_seconds"]))
    selected = select_artifacts(entries, featured, args.run, args.featured_only)
    if args.featured_only:
        tracked = set(subprocess.check_output(["git", "-C", str(REPO_ROOT), "ls-files", "-z"]).decode().split("\0"))
        included = tracked | set(selected)
        for entry in entries:
            entry["files"] = [path for path in entry["files"] if path in included]
            if "videos" in entry:
                entry["videos"] = [path for path in entry["videos"] if path in included]
    generated = write_readme(REPO_ROOT, settings, entries, featured)
    listing = REPO_ROOT / "logs" / "public_artifacts" / "selected_paths.txt"
    listing.parent.mkdir(parents=True, exist_ok=True)
    listing.write_text("\n".join(selected) + "\n")
    if args.stage:
        paths = selected + [str(path.relative_to(REPO_ROOT)) for path in generated]
        subprocess.run(["git", "-C", str(REPO_ROOT), "add", "-f", "--", *paths], check=True)
    print(f"Prepared {len(selected)} selected artifact files; {'staged' if args.stage else 'not staged'}.")
    for entry in entries + featured:
        print(f"{entry['name']}: {entry['status']}")
    print(f"Selected paths: {listing.relative_to(REPO_ROOT)}")
    print("No commit or push was performed.")


if __name__ == "__main__":
    main()
