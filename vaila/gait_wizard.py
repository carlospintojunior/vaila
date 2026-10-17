"""
Project: vailá Multimodal Toolbox
Script: gait_wizard.py - Marcha Fácil (assistente passo a passo de análise de marcha)

Author: Carlos Pinto Jr
Email: carlos@vulcanum.com.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 12 October 2026
Update Date: 17 October 2026
Version: 0.4.9

Description:
A step-by-step wizard (UI in Portuguese) that takes a user with no computer
background from a "Record Cameras" session to gait results, one big button
per screen:

1. Captura       - recording from here is not available yet ("em breve");
                   the user picks an existing session from a list (date,
                   time, cameras, status and a preview of each camera).
2. Corte         - opens "Cut Cameras" with the session already loaded and
                   moves on by itself when the cut finishes; or "use the
                   whole video", or reuse a previous cut.
3. Processamento - the user identifies each camera's role (side-view /
                   "lateral", frontal / "coronal", or not used) and clicks
                   "Analisar": MediaPipe + automatic gait events/phases
                   (gait_auto.py, lateral cameras only) + MP Angles joint
                   angles and the MP Angles skeleton + angles video
                   (mpangles.py) for every lateral camera, plus a MediaPipe
                   + MP Angles skeleton pass (video + stick-figure image,
                   no gait-cycle detection) plus frontal-plane metrics
                   (pelvic obliquity, knee valgus/varus) for the coronal
                   camera when one is marked, with a plain-language
                   checklist and elapsed time.
4. Resultados    - a few big numbers (cadence, stance/swing, stride time,
                   symmetry), joint angles over the gait cycle, plain advice
                   when the recording has problems, a big button that plays
                   the skeleton + angles video, and buttons to open the full
                   report or the results folder.

Progress is saved in <session_dir>/assistente_marcha.json, so closing the
program and coming back resumes where the user stopped.

Output (inside the analysed folder, i.e. the cut folder or the session):
gait_analysis_YYYYMMDD_HHMMSS/
    gait_report.html, gait_events.csv, gait_cycles.csv, gait_summary.csv
    angles/angles_<video>.mp4                            skeleton + angles video (H.264)
    angles/<video>_rel.csv, _abs_180.csv, _abs_360.csv   (MP Angles)
    angles/<video>.csv       landmarks low-passed at 6 Hz (input of MP Angles)
    angles/<video>_frontal.csv  coronal camera only: pelvic obliquity, knee valgus/varus per frame
    joint_angles_cycle.csv   mean/SD per % of the gait cycle
    joint_angles_cycle.png   hip / knee / ankle over the gait cycle

Usage:
GUI: click "Marcha Fácil" in vailá's Video and Image tools, or run
     ``uv run python vaila/gait_wizard.py [sessions_root]``.
"""

from __future__ import annotations

import base64
import contextlib
import json
import os
import platform
import queue
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import cast

import numpy as np
import pandas as pd

try:
    from .filter_utils import butter_filter
    from .gait_auto import (
        DEFAULT_CAMERAS,
        GaitResult,
        _video_size,
        analyze_session_full,
        run_mediapipe,
        select_camera_videos,
        symmetry_index,
    )
    from .gait_report import ReportInfo, write_reports
    from .multicam_cut import discover_session_videos, read_frame_times
    from .multicam_recorder import load_config_value, save_config_value
except ImportError:
    from filter_utils import butter_filter  # ty: ignore[unresolved-import]
    from gait_auto import (  # ty: ignore[unresolved-import]
        DEFAULT_CAMERAS,
        GaitResult,
        _video_size,
        analyze_session_full,
        run_mediapipe,
        select_camera_videos,
        symmetry_index,
    )
    from gait_report import ReportInfo, write_reports  # ty: ignore[unresolved-import]
    from multicam_cut import (  # ty: ignore[unresolved-import]
        discover_session_videos,
        read_frame_times,
    )
    from multicam_recorder import (  # ty: ignore[unresolved-import]
        load_config_value,
        save_config_value,
    )


CONFIG_SECTION = "gait_wizard"
STATE_FILE = "assistente_marcha.json"
STEPS = ("Captura", "Corte", "Processamento", "Resultados")
# MP Angles relative angles are unsigned (0..180°): knee is shown as flexion
# (180 - angle, 0 = straight leg); hip (trunk-thigh) and ankle (shank-foot)
# are shown as computed by MP Angles.
JOINTS = (
    ("hip", "Quadril (ângulo tronco-coxa)"),
    ("knee", "Joelho (flexão)"),
    ("ankle", "Tornozelo (ângulo perna-pé)"),
)
SIDE_COLORS = {"left": "#1f77b4", "right": "#d62728"}
SIDE_NAMES = {"left": "esquerdo", "right": "direito"}
CYCLE_POINTS = 101  # 0..100 % of the gait cycle
MIN_CYCLE_COVERAGE = 0.8  # fraction of a cycle's frames that must have angles
ANGLE_LOWPASS_HZ = 6.0
FEET_CUT_LIMIT = 0.3
FEW_CYCLES = 3
AFFECTED_SIDES = (("Nenhum", ""), ("Direito", "right"), ("Esquerdo", "left"), ("Ambos", "both"))
REPORT_LANGS = (("Português", "pt"), ("English", "en"))
# Camera role in the Processamento step: "lateral" goes through gait-cycle
# detection (gait_auto.py); "coronal" only gets MediaPipe + the MP Angles
# skeleton video/image (no gait events); "nao" is not analysed.
CAMERA_ROLES = (("Lateral", "lateral"), ("Coronal (de frente)", "coronal"), ("Não usar", "nao"))


# ── Sessions & saved progress ────────────────────────────────────────────────


@dataclass
class SessionInfo:
    path: Path
    name: str
    when: datetime
    videos: list[Path]
    state: dict = field(default_factory=dict)

    @property
    def cameras(self) -> list[str]:
        return [camera_name(v) for v in self.videos]

    @property
    def status(self) -> str:
        if self.state.get("analysis_dir") and Path(self.state["analysis_dir"]).is_dir():
            return "Analisada"
        if self.state.get("source_dir") and Path(self.state["source_dir"]).is_dir():
            return "Cortada" if self.state.get("cut_dir") else "Pronta para analisar"
        return "Nova"


def camera_name(video: Path) -> str:
    """``cam2_trial01_frame_1_to_90.mp4`` -> ``cam2``."""
    return video.stem.split("_")[0]


def session_datetime(path: Path) -> datetime:
    """Date/time from a ``<name>_YYYYMMDD_HHMMSS`` folder, else its modification time."""
    m = re.search(r"(\d{8})_(\d{6})$", path.name)
    if m:
        with contextlib.suppress(ValueError):
            return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    return datetime.fromtimestamp(path.stat().st_mtime)


def load_state(session_dir: Path) -> dict:
    try:
        return json.loads((session_dir / STATE_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_state(session_dir: Path, state: dict) -> None:
    state = {**state, "updated": datetime.now().isoformat(timespec="seconds")}
    with contextlib.suppress(OSError):
        (session_dir / STATE_FILE).write_text(
            json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def _session_videos(folder: Path) -> list[Path]:
    try:
        return discover_session_videos(folder)
    except Exception:  # noqa: BLE001 - not a session folder
        return []


def find_sessions(root: Path) -> list[SessionInfo]:
    """Session folders (with camera videos) in ``root`` and ``root`` itself, newest first."""
    root = Path(root)
    if not root.is_dir():
        return []
    candidates = [root, *(p for p in root.iterdir() if p.is_dir())]
    sessions = []
    for folder in candidates:
        if folder.name.startswith(("processed_", "gait_analysis_", "mediapipe")):
            continue
        videos = _session_videos(folder)
        if videos:
            sessions.append(
                SessionInfo(
                    folder, folder.name, session_datetime(folder), videos, load_state(folder)
                )
            )
    return sorted(sessions, key=lambda s: s.when, reverse=True)


def latest_cut_dir(session_dir: Path) -> Path | None:
    cuts = sorted(p for p in session_dir.glob("processed_multicam_cut_*") if _session_videos(p))
    return cuts[-1] if cuts else None


def default_lateral_cameras(available: list[str]) -> list[str]:
    saved = [c for c in load_config_value(CONFIG_SECTION, "lateral_cameras").split(",") if c]
    for preferred in (saved, list(DEFAULT_CAMERAS)):
        chosen = [c for c in preferred if c in available]
        if chosen:
            return chosen
    return list(available)


def default_coronal_cameras(available: list[str], lateral: list[str]) -> list[str]:
    """Remembered coronal camera(s), restricted to cameras not already marked lateral."""
    saved = [c for c in load_config_value(CONFIG_SECTION, "coronal_cameras").split(",") if c]
    return [c for c in saved if c in available and c not in lateral]


# ── Joint angles over the gait cycle (MP Angles) ─────────────────────────────


def _smooth(values: np.ndarray, fs: float) -> np.ndarray:
    """Fill gaps by interpolation and low-pass; frames that had no data stay NaN."""
    bad = ~np.isfinite(values)
    if bad.all() or (~bad).sum() < 15:
        return values
    good = np.flatnonzero(~bad)
    filled = np.interp(np.arange(len(values)), good, values[good])
    out = butter_filter(filled, fs, cutoff=min(ANGLE_LOWPASS_HZ, 0.4 * fs))
    out[bad] = np.nan
    return out


def normalize_cycles(
    angles: pd.DataFrame, times: np.ndarray, cycles: pd.DataFrame, camera: str
) -> list[dict]:
    """Resample each joint angle of each complete cycle to 0..100 % of the cycle.

    ``angles`` has one row per frame (``frame`` column) and ``<side>_<joint>``
    columns; ``cycles`` comes from gait_auto (hs_time_s .. next_hs_time_s).
    Knee values are converted to flexion (0 = straight leg).
    """
    if cycles.empty or angles.empty:
        return []
    frames = angles["frame"].to_numpy(dtype=int)
    keep = (frames >= 0) & (frames < len(times))
    angles, frames = angles[keep], frames[keep]
    t = times[frames]
    fs = 1.0 / float(np.median(np.diff(t))) if len(t) > 1 else 30.0
    pct = np.linspace(0, 100, CYCLE_POINTS)
    rows = []
    for i, cyc in enumerate(cycles[cycles["camera"] == camera].itertuples(index=False)):
        side = str(cyc.side)
        start, end = float(cyc.hs_time_s), float(cyc.next_hs_time_s)
        inside = (t >= start) & (t <= end)
        if inside.sum() < 5:
            continue
        for joint, _label in JOINTS:
            col = f"{side}_{joint}"
            if col not in angles:
                continue
            values = angles[col].to_numpy(dtype=float)
            if joint == "knee":
                values = 180.0 - values
            values = _smooth(values, fs)
            seg_t, seg_v = t[inside], values[inside]
            ok = np.isfinite(seg_v)
            if ok.mean() < MIN_CYCLE_COVERAGE:
                continue
            curve = np.interp(start + pct / 100 * (end - start), seg_t[ok], seg_v[ok])
            rows.append(
                {
                    "camera": camera,
                    "cycle": i,
                    "side": side,
                    "joint": joint,
                    "to_pct": float(cyc.stance_pct),
                    "curve": curve,
                }
            )
    return rows


def smooth_landmarks_csv(src: Path, dst: Path, fs: float) -> None:
    """Low-pass every landmark column of a ``_pixel_vaila.csv``, keeping no-person frames empty.

    MP Angles' own filter (filtfilt over the whole column) turns a column with
    any NaN into all-NaN, and webcam sessions always have frames with nobody
    in view; filtering here keeps the angle video steady instead of jittery.
    """
    df = pd.read_csv(src)
    for col in df.columns[1:]:
        df[col] = _smooth(df[col].to_numpy(dtype=float), fs)
    df.to_csv(dst, index=False, float_format="%.3f")


def to_h264(video: Path, log: Callable[[str], None] = print) -> Path:
    """Re-encode OpenCV's MPEG-4 Part 2 output to H.264 (plays in any player and browser).

    Returns the original file if FFmpeg is unavailable or fails.
    """
    try:
        from .ffmpeg_utils import get_ffmpeg_path
    except ImportError:
        from ffmpeg_utils import get_ffmpeg_path  # ty: ignore[unresolved-import]

    tmp = video.with_name(video.stem + "_h264.mp4")
    cmd = [
        get_ffmpeg_path(),
        "-y",
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(tmp),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=1800)
        tmp.replace(video)
    except Exception as exc:  # noqa: BLE001 - the MPEG-4 file is still usable
        log(f"H.264 re-encode skipped for {video.name}: {exc}")
        with contextlib.suppress(OSError):
            tmp.unlink()
    return video


def compute_cycle_angles(
    gait: GaitResult, log: Callable[[str], None] = print, make_video: bool = True
) -> list[dict]:
    """Run MP Angles on each analysed camera and cut the angles into gait cycles.

    With ``make_video`` it also renders MP Angles' skeleton + angles video
    (``angles/angles_<video>.mp4``) for every camera.
    """
    try:
        from .mpangles import process_angles, process_video_with_visualization
    except ImportError:
        from mpangles import (  # ty: ignore[unresolved-import]
            process_angles,
            process_video_with_visualization,
        )

    angles_dir = gait.out_dir / "angles"
    angles_dir.mkdir(exist_ok=True)
    rows: list[dict] = []
    for video in gait.videos:
        csv = gait.mediapipe_run / video.stem / f"{video.stem}_pixel_vaila.csv"
        if not csv.is_file():
            log(f"MP Angles: {csv.name} not found, skipping {video.name}")
            continue
        w, h, count, fps = _video_size(video)
        times = read_frame_times(video, count, fps)
        fs = 1.0 / float(np.median(np.diff(times))) if len(times) > 1 else fps
        smooth_csv = angles_dir / f"{video.stem}.csv"
        smooth_landmarks_csv(csv, smooth_csv, fs)
        rel_csv = angles_dir / f"{video.stem}_rel.csv"
        if make_video:
            log(f"MP Angles: rendering the angle video for {video.name}")
            with contextlib.suppress(Exception):
                process_video_with_visualization(
                    str(video), str(smooth_csv), str(angles_dir), {"enabled": False}
                )
            rendered = angles_dir / f"angles_{video.name}"
            if rendered.is_file():
                to_h264(rendered, log)
        if not rel_csv.is_file():
            process_angles(str(smooth_csv), str(smooth_csv), {"enabled": False}, (w, h))
        rel = pd.read_csv(rel_csv)
        if gait.cycles.empty:
            continue
        if len(times) < len(rel):
            times = read_frame_times(video, len(rel), fps)
        rows += normalize_cycles(rel, times, gait.cycles, camera_name(video))
    return rows


def frontal_plane_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Per-frame frontal-plane deviations from a coronal camera's pixel landmarks.

    ``pelvic_obliquity_deg``: signed, positive = right hip lower (right pelvic
    drop), negative = left pelvic drop; 0 = level pelvis.
    ``left_knee_deg`` / ``right_knee_deg``: signed, positive = valgus (knee
    deviates toward the body midline), negative = varus (knee deviates away
    from it); 0 = straight leg in the frontal plane. Magnitude is the angle
    between the thigh (hip->knee) and shank (knee->ankle) vectors; the sign
    comes from which side of the straight hip-ankle line the knee falls on,
    relative to the midpoint between the two hips.

    Approximate 2D monocular measures, with no calibration: sensitive to the
    person's rotation relative to the camera. Meaningful only as a screening
    aid when the coronal camera is reasonably square-on to the walkway.
    """
    try:
        from .mpangles import get_vector_landmark
    except ImportError:
        from mpangles import get_vector_landmark  # ty: ignore[unresolved-import]

    n = len(df)
    left_hip = get_vector_landmark(df, "left_hip")
    right_hip = get_vector_landmark(df, "right_hip")
    midline_x = (left_hip[:, 0] + right_hip[:, 0]) / 2.0

    dx = np.abs(right_hip[:, 0] - left_hip[:, 0])
    dy = right_hip[:, 1] - left_hip[:, 1]
    pelvic_obliquity = np.degrees(np.arctan2(dy, dx))

    def knee_deviation(side: str) -> np.ndarray:
        hip = get_vector_landmark(df, f"{side}_hip")
        knee = get_vector_landmark(df, f"{side}_knee")
        ankle = get_vector_landmark(df, f"{side}_ankle")
        thigh = knee - hip
        shank = ankle - knee
        norm_t = np.linalg.norm(thigh, axis=1)
        norm_s = np.linalg.norm(shank, axis=1)
        magnitude = np.full(n, np.nan)
        valid = (norm_t > 0) & (norm_s > 0)
        cos_a = np.sum(thigh[valid] * shank[valid], axis=1) / (norm_t[valid] * norm_s[valid])
        magnitude[valid] = np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))

        with np.errstate(invalid="ignore", divide="ignore"):
            t = (knee[:, 1] - hip[:, 1]) / (ankle[:, 1] - hip[:, 1])
        expected_knee_x = hip[:, 0] + t * (ankle[:, 0] - hip[:, 0])
        offset = knee[:, 0] - expected_knee_x
        medial_sign = np.where(midline_x >= hip[:, 0], 1.0, -1.0)
        signed = offset * medial_sign
        sign = np.where(np.isfinite(signed), np.where(signed >= 0, 1.0, -1.0), np.nan)
        return sign * magnitude

    return pd.DataFrame(
        {
            "frame": df.iloc[:, 0].to_numpy(),
            "pelvic_obliquity_deg": pelvic_obliquity,
            "left_knee_deg": knee_deviation("left"),
            "right_knee_deg": knee_deviation("right"),
        }
    )


def process_coronal_cameras(
    source_dir: Path,
    coronal_cameras: list[str],
    out_dir: Path,
    log: Callable[[str], None] = print,
) -> None:
    """MediaPipe + MP Angles skeleton video/image for the coronal (frontal) camera(s).

    No gait-cycle detection here: ``analyze_session_full`` finds walking passes
    from horizontal displacement across the image, which a frontal camera does
    not show. This renders the skeleton + angles video and the MP Angles
    stick-figure sequence image (``angles/<stem>_stick_sequence.png``), and
    writes frontal-plane metrics (pelvic obliquity, knee valgus/varus) to
    ``angles/<stem>_frontal.csv`` — all picked up by ``gait_report.py``.
    """
    if not coronal_cameras:
        return
    try:
        from .mpangles import process_video_with_visualization
    except ImportError:
        from mpangles import process_video_with_visualization  # ty: ignore[unresolved-import]

    try:
        videos = select_camera_videos(source_dir, coronal_cameras)
    except FileNotFoundError as exc:
        log(f"Coronal camera: {exc}")
        return
    log(f"Coronal camera(s): {', '.join(v.name for v in videos)}")
    run = run_mediapipe(videos, out_dir / "mediapipe", log)
    angles_dir = out_dir / "angles"
    angles_dir.mkdir(exist_ok=True)
    for video in videos:
        csv = run / video.stem / f"{video.stem}_pixel_vaila.csv"
        if not csv.is_file():
            log(f"MP Angles: {csv.name} not found, skipping {video.name}")
            continue
        _w, _h, count, fps = _video_size(video)
        times = read_frame_times(video, count, fps)
        fs = 1.0 / float(np.median(np.diff(times))) if len(times) > 1 else fps
        smooth_csv = angles_dir / f"{video.stem}.csv"
        smooth_landmarks_csv(csv, smooth_csv, fs)
        log(f"Frontal-plane metrics: pelvic obliquity and knee valgus/varus for {video.name}")
        with contextlib.suppress(Exception):
            frontal = frontal_plane_metrics(pd.read_csv(smooth_csv))
            frontal.to_csv(angles_dir / f"{video.stem}_frontal.csv", index=False)
        log(f"MP Angles: rendering the frontal-view skeleton for {video.name}")
        with contextlib.suppress(Exception):
            process_video_with_visualization(
                str(video), str(smooth_csv), str(angles_dir), {"enabled": False}
            )
        rendered = angles_dir / f"angles_{video.name}"
        if rendered.is_file():
            to_h264(rendered, log)


def angle_videos(out_dir: Path, passes: dict[str, int] | None = None) -> list[tuple[str, Path]]:
    """``[(camera, angles_<video>.mp4)]``, cameras with valid walking passes first."""
    found = [
        (camera_name(Path(p.name.removeprefix("angles_"))), p)
        for p in sorted((out_dir / "angles").glob("angles_*.mp4"))
    ]
    passes = passes or {}
    return sorted(found, key=lambda cv: (passes.get(cv[0], 0) == 0, cv[0]))


def cycle_angle_table(rows: list[dict]) -> pd.DataFrame:
    """Mean and SD per joint, side and % of the cycle."""
    out = []
    pct = np.linspace(0, 100, CYCLE_POINTS)
    if not rows:
        return pd.DataFrame()
    for key, group in pd.DataFrame(rows).groupby(["joint", "side"]):
        joint, side = cast(tuple[str, str], key)
        curves = np.vstack(group["curve"].to_list())
        sd = curves.std(axis=0, ddof=1) if len(curves) > 1 else np.full(CYCLE_POINTS, np.nan)
        for k, p in enumerate(pct):
            out.append(
                {
                    "joint": joint,
                    "side": side,
                    "cycle_pct": p,
                    "angle_mean_deg": curves[:, k].mean(),
                    "angle_sd_deg": sd[k],
                    "n_cycles": len(curves),
                }
            )
    return pd.DataFrame(out)


def plot_cycle_angles(rows: list[dict], png: Path) -> bool:
    """Hip / knee / ankle mean (± SD) over the gait cycle, per side. False if no data."""
    if not rows:
        return False
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = pd.DataFrame(rows)
    pct = np.linspace(0, 100, CYCLE_POINTS)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, (joint, label) in zip(axes, JOINTS, strict=True):
        for side, color in SIDE_COLORS.items():
            group = data[(data["joint"] == joint) & (data["side"] == side)]
            if group.empty:
                continue
            curves = np.vstack(group["curve"].to_list())
            mean = curves.mean(axis=0)
            ax.plot(pct, mean, color=color, lw=2, label=f"{SIDE_NAMES[side]} (n={len(curves)})")
            if len(curves) > 1:
                sd = curves.std(axis=0, ddof=1)
                ax.fill_between(pct, mean - sd, mean + sd, color=color, alpha=0.15)
            ax.axvline(group["to_pct"].mean(), color=color, ls="--", lw=1, alpha=0.7)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("% do ciclo da marcha")
        ax.set_xlim(0, 100)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("graus (°)")
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle(
        "Ângulos ao longo do ciclo (linha tracejada = fim do apoio / início do balanço)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(png, dpi=90)
    plt.close(fig)
    return True


def add_angles_to_report(
    report: Path, png: Path | None, videos: list[tuple[str, Path]] | None = None
) -> None:
    """Insert the joint-angle figure and the MP Angles videos into gait_auto's HTML report.

    Videos are linked by relative path (not embedded), so they play while the
    report stays in its analysis folder.
    """
    section = ""
    if png is not None:
        data = base64.b64encode(png.read_bytes()).decode("ascii")
        section += (
            "<h2>Joint angles over the gait cycle (MP Angles)</h2>"
            "<p>Mean ± SD per side; dashed line = toe-off. Knee shown as flexion "
            "(0° = straight leg). In a side view the leg farther from the camera is "
            "partly hidden, so its curve is less reliable.</p>"
            f"<img src='data:image/png;base64,{data}' style='max-width:100%'>\n"
        )
    for camera, video in videos or []:
        rel = video.relative_to(report.parent).as_posix()
        section += (
            f"<h2>{camera}: skeleton and joint angles (MP Angles)</h2>"
            f"<video src='{rel}' controls style='max-width:100%'></video>"
            f"<p><a href='{rel}'>{video.name}</a></p>\n"
        )
    if not section:
        return
    text = report.read_text(encoding="utf-8")
    marker = "<h2>Method notes</h2>"
    text = text.replace(marker, section + marker, 1) if marker in text else text + section
    report.write_text(text, encoding="utf-8")


# ── Full pipeline (no Tk: runs on a worker thread, testable) ─────────────────


def run_pipeline(
    source_dir: Path,
    cameras: list[str],
    participant: str,
    log: Callable[[str], None] = print,
    stage: Callable[[int], None] = lambda _i: None,
    info: ReportInfo | None = None,
    coronal_cameras: list[str] | None = None,
) -> GaitResult:
    """MediaPipe -> gait events/phases -> joint angles -> reports. ``stage(i)`` marks progress.

    ``coronal_cameras`` (frontal/coronal-plane cameras) skip gait-cycle
    detection and only get a MediaPipe + MP Angles skeleton pass, for the
    report's skeleton-overview section.
    """

    def gait_log(msg: str) -> None:
        log(msg)
        if "walking pass" in msg:  # MediaPipe finished, gait events started
            stage(1)

    stage(0)
    gait = analyze_session_full(source_dir, cameras, participant, log=gait_log)
    stage(2)
    rows = compute_cycle_angles(gait, log)
    process_coronal_cameras(source_dir, coronal_cameras or [], gait.out_dir, log)
    stage(3)
    cycle_angle_table(rows).to_csv(gait.out_dir / "joint_angles_cycle.csv", index=False)
    png = gait.out_dir / "joint_angles_cycle.png"
    add_angles_to_report(
        gait.report,
        png if plot_cycle_angles(rows, png) else None,
        angle_videos(gait.out_dir, gait.passes),
    )
    try:
        write_reports(gait.out_dir, info or ReportInfo(participant=participant))
    except Exception as exc:  # noqa: BLE001 - the analysis itself is done; report can be redone
        log(f"Printable report failed: {type(exc).__name__}: {exc}")
    stage(4)
    return gait


def print_report(out_dir: Path, lang: str) -> Path | None:
    """The printable report in ``lang``: the PDF when there is one, else the HTML."""
    try:
        from .gait_report import REPORT_NAMES
    except ImportError:
        from gait_report import REPORT_NAMES  # ty: ignore[unresolved-import]
    for suffix in (".pdf", ".html"):
        path = Path(out_dir) / f"{REPORT_NAMES[lang]}{suffix}"
        if path.is_file():
            return path
    return None


def report_info(state: dict, recording: str = "") -> ReportInfo:
    return ReportInfo(
        participant=state.get("participant", ""),
        age=state.get("age", ""),
        affected_side=state.get("affected_side", ""),
        therapist=state.get("therapist", ""),
        institution=state.get("institution", ""),
        recording=recording,
    )


# ── Plain-language summary of a finished analysis ────────────────────────────


@dataclass
class ResultView:
    n_cycles: int
    cards: list[tuple[str, str, str]]  # (title, value, explanation)
    advice: list[str]
    angles_png: Path | None
    report: Path
    out_dir: Path
    videos: list[tuple[str, Path]] = field(default_factory=list)  # MP Angles videos


def _summary_value(summary: pd.DataFrame, metric: str) -> float:
    try:
        row = summary[(summary["camera"] == "all") & (summary["side"] == "both")].iloc[0]
        return float(row[f"{metric}_mean"])
    except (IndexError, KeyError, TypeError, ValueError):
        return float("nan")


def recording_advice(
    n_cycles: int, passes: dict[str, int], feet_cut: dict[str, float]
) -> list[str]:
    advice = []
    for cam, frac in feet_cut.items():
        if np.isfinite(frac) and frac > FEET_CUT_LIMIT:
            advice.append(
                f"Câmera {cam}: os pés saem pela parte de baixo da imagem em "
                f"{100 * frac:.0f}% do tempo. Na próxima gravação, afaste a câmera ou "
                "incline-a para baixo até aparecer um pouco do chão."
            )
    for cam, n in passes.items():
        if n == 0:
            advice.append(
                f"Câmera {cam}: não encontramos a pessoa atravessando a imagem de lado, "
                "do começo ao fim. Ela deve aparecer inteira (da cabeça aos pés), "
                "caminhando de um lado para o outro da imagem."
            )
    if n_cycles == 0:
        advice.append(
            "Não foi possível medir nenhum ciclo de marcha completo. Grave novamente "
            "seguindo as dicas acima, com várias idas e voltas."
        )
    elif n_cycles < FEW_CYCLES:
        advice.append(
            f"Foram medidos só {n_cycles} ciclo(s) de marcha: os valores são apenas "
            "indicativos. Para um resultado confiável, grave várias idas e voltas "
            "(10 ciclos ou mais)."
        )
    return advice


def build_result_view(
    out_dir: Path, passes: dict[str, int], feet_cut: dict[str, float]
) -> ResultView:
    try:
        summary = pd.read_csv(out_dir / "gait_summary.csv")
    except (OSError, pd.errors.EmptyDataError):
        summary = pd.DataFrame()
    n_cycles = 0
    with contextlib.suppress(Exception):
        n_cycles = int(
            summary[(summary["camera"] == "all") & (summary["side"] == "both")]["n_cycles"].iloc[0]
        )

    def fmt(v: float, digits: int, unit: str) -> str:
        return f"{v:.{digits}f}{unit}".replace(".", ",") if np.isfinite(v) else "–"

    sym = symmetry_index(summary, "stance_pct") if not summary.empty else float("nan")
    cards = [
        (
            "Cadência",
            fmt(_summary_value(summary, "cadence_steps_min"), 0, " passos/min"),
            "Quantos passos por minuto.",
        ),
        (
            "Apoio",
            fmt(_summary_value(summary, "stance_pct"), 0, "%"),
            "Parte do ciclo com o pé no chão (normal ≈ 60%).",
        ),
        (
            "Balanço",
            fmt(_summary_value(summary, "swing_pct"), 0, "%"),
            "Parte do ciclo com o pé no ar (normal ≈ 40%).",
        ),
        (
            "Tempo da passada",
            fmt(_summary_value(summary, "stride_time_s"), 2, " s"),
            "De um toque do calcanhar ao próximo do mesmo pé.",
        ),
        (
            "Simetria E/D",
            fmt(abs(sym), 0, "%") if np.isfinite(sym) else "–",
            "Diferença do apoio entre as pernas (0% = iguais).",
        ),
    ]
    png = out_dir / "joint_angles_cycle.png"
    return ResultView(
        n_cycles,
        cards,
        recording_advice(n_cycles, passes, feet_cut),
        png if png.is_file() else None,
        out_dir / "gait_report.html",
        out_dir,
        angle_videos(out_dir, passes),
    )


def open_path(path: Path) -> None:
    """Open a file or folder with the system's default application."""
    if path.suffix.lower() in (".html", ".htm"):
        webbrowser.open(path.resolve().as_uri())
        return
    system = platform.system()
    with contextlib.suppress(Exception):
        if system == "Windows":
            os.startfile(str(path))  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
        elif system == "Darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])


def short_path(path: Path, limit: int = 80) -> str:
    """``Pasta: …/end/of/a/long/path`` (fits one line of the wizard)."""
    text = str(path)
    if len(text) > limit:
        text = "…" + text[-(limit - 1) :]
    return f"Pasta: {text}"


def grab_thumbnail(video: Path, width: int = 200):
    """A frame from the middle of ``video`` as a PIL image (None if unreadable)."""
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(str(video))
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if count > 2:
            cap.set(cv2.CAP_PROP_POS_FRAMES, count // 2)
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        return None
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image.thumbnail((width, width))
    return image


# ── The wizard window ────────────────────────────────────────────────────────

# Tk maps Helvetica to Arial on Windows; on X11 without Xft it has more symbol
# glyphs (▶ ✔ ✂) than the default font, especially in bold.
FONT = "Helvetica"
BG = "#f7f7f9"
ACCENT = "#1565c0"
DONE = "#2e7d32"
MUTED = "#9e9e9e"


class GaitWizardApp:
    def __init__(self, window: tk.Toplevel, sessions_root: str | None = None) -> None:
        self.window = window
        self.window.title("vailá - Marcha Fácil")
        self.window.geometry("960x720")
        self.window.minsize(820, 600)
        self.window.configure(bg=BG)
        self._setup_styles()

        root = (
            sessions_root
            or load_config_value(CONFIG_SECTION, "sessions_root")
            or load_config_value("multicam_recorder", "last_output_dir")
            or str(Path.home())
        )
        self.sessions_root = Path(root)
        self.sessions: list[SessionInfo] = []
        self.session: SessionInfo | None = None
        self.source_dir: Path | None = None  # cut folder or the session itself
        self.step = 0
        self.processing = False
        self._images: list = []  # keep PhotoImage references alive
        self._cut_app = None  # the open Cut Cameras tool, if any
        self._events: queue.Queue[tuple] = queue.Queue()
        self._log_lines: list[str] = []

        self._build_frame()
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
        self.show_step(0)

    # ── layout ──

    def _setup_styles(self) -> None:
        style = ttk.Style(self.window)
        style.configure("W.TFrame", background=BG)
        style.configure("W.TLabel", background=BG, font=(FONT, 11))
        style.configure("Title.TLabel", background=BG, font=(FONT, 17, "bold"))
        style.configure("Hint.TLabel", background=BG, foreground="#555", font=(FONT, 11))
        style.configure("Big.TButton", font=(FONT, 13, "bold"), padding=(18, 10))
        style.configure("Mid.TButton", font=(FONT, 11), padding=(12, 6))
        style.configure("W.TCheckbutton", background=BG, font=(FONT, 11))
        style.configure("Treeview", font=(FONT, 11), rowheight=28)
        style.configure("Treeview.Heading", font=(FONT, 11, "bold"))

    def _build_frame(self) -> None:
        self.stepper = tk.Frame(self.window, bg=BG)
        self.stepper.pack(fill="x", padx=20, pady=(16, 6))
        ttk.Separator(self.window).pack(fill="x", padx=20)
        self.body = ttk.Frame(self.window, style="W.TFrame", padding=(24, 14))
        self.body.pack(fill="both", expand=True)
        ttk.Separator(self.window).pack(fill="x", padx=20)
        footer = ttk.Frame(self.window, style="W.TFrame", padding=(20, 10))
        footer.pack(fill="x")
        self.back_btn = ttk.Button(
            footer, text="◀  Voltar", style="Mid.TButton", command=self.go_back
        )
        self.back_btn.pack(side="left")
        self.next_btn = ttk.Button(footer, text="Próximo  ▶", style="Big.TButton")
        self.next_btn.pack(side="right")

    def _draw_stepper(self) -> None:
        for w in self.stepper.winfo_children():
            w.destroy()
        for i, name in enumerate(STEPS):
            if i:
                tk.Label(self.stepper, text="──────", bg=BG, fg=MUTED).pack(side="left", padx=4)
            if i < self.step:
                text, color, weight = f"✔ {name}", DONE, "normal"
            elif i == self.step:
                text, color, weight = f"{i + 1}  {name}", ACCENT, "bold"
            else:
                text, color, weight = f"{i + 1}  {name}", MUTED, "normal"
            if i == 0:
                text += " (em breve)" if i != self.step else ""
            tk.Label(self.stepper, text=text, bg=BG, fg=color, font=(FONT, 12, weight)).pack(
                side="left"
            )

    def _clear_body(self) -> None:
        for w in self.body.winfo_children():
            w.destroy()
        self._images = []

    def _title(self, text: str, hint: str = "") -> None:
        ttk.Label(self.body, text=text, style="Title.TLabel").pack(anchor="w")
        if hint:
            ttk.Label(self.body, text=hint, style="Hint.TLabel", wraplength=880).pack(
                anchor="w", pady=(4, 12)
            )

    def show_step(self, step: int) -> None:
        self.step = step
        self._draw_stepper()
        self._clear_body()
        self.back_btn.config(state="normal" if step > 0 else "disabled")
        self.next_btn.pack(side="right")
        [self._step_capture, self._step_cut, self._step_process, self._step_results][step]()

    def go_back(self) -> None:
        if self.step > 0 and not self.processing:
            self.show_step(self.step - 1)

    # ── step 1: choose a recording ──

    def _step_capture(self) -> None:
        self._title(
            "Escolha a gravação",
            "A gravação direto por aqui chegará em breve. Por enquanto, escolha abaixo "
            "uma gravação já feita com o Record Cameras e clique em Próximo.",
        )
        row = ttk.Frame(self.body, style="W.TFrame")
        row.pack(fill="x", pady=(0, 8))
        ttk.Button(row, text="Trocar pasta…", style="Mid.TButton", command=self._pick_root).pack(
            side="right"
        )
        self.root_label = ttk.Label(row, text=short_path(self.sessions_root), style="W.TLabel")
        self.root_label.pack(side="left")

        cols = ("data", "hora", "nome", "cameras", "situacao")
        self.tree = ttk.Treeview(self.body, columns=cols, show="headings", height=8)
        for col, title, width in zip(
            cols,
            ("Data", "Hora", "Gravação", "Câmeras", "Situação"),
            (110, 80, 300, 90, 170),
            strict=True,
        ):
            self.tree.heading(col, text=title)
            self.tree.column(col, width=width, anchor="w")
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<<TreeviewSelect>>", lambda _e: self._on_session_select())
        self.tree.bind("<Double-1>", lambda _e: self._capture_next())

        self.thumbs = ttk.Frame(self.body, style="W.TFrame")
        self.thumbs.pack(fill="x", pady=(10, 0))
        self.next_btn.config(text="Próximo  ▶", command=self._capture_next)
        self._fill_sessions()

    def _pick_root(self) -> None:
        folder = filedialog.askdirectory(
            parent=self.window, title="Pasta com as gravações", initialdir=str(self.sessions_root)
        )
        if folder:
            self.sessions_root = Path(folder)
            save_config_value(CONFIG_SECTION, "sessions_root", folder)
            self.root_label.config(text=short_path(self.sessions_root))
            self._fill_sessions()

    def _fill_sessions(self) -> None:
        self.tree.delete(*self.tree.get_children())
        self.sessions = find_sessions(self.sessions_root)
        for i, s in enumerate(self.sessions):
            self.tree.insert(
                "",
                "end",
                iid=str(i),
                values=(
                    s.when.strftime("%d/%m/%Y"),
                    s.when.strftime("%H:%M"),
                    s.name,
                    len(s.videos),
                    s.status,
                ),
            )
        if not self.sessions:
            ttk.Label(
                self.thumbs,
                text="Nenhuma gravação encontrada nesta pasta. Clique em “Trocar pasta…”.",
                style="Hint.TLabel",
            ).pack(anchor="w")
            return
        current = (
            next((i for i, s in enumerate(self.sessions) if s.path == self.session.path), 0)
            if self.session
            else 0
        )
        self.tree.selection_set(str(current))
        self.tree.see(str(current))

    def _on_session_select(self) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        self.session = self.sessions[int(sel[0])]
        for w in self.thumbs.winfo_children():
            w.destroy()
        self._images = []
        from PIL import ImageTk

        for video in self.session.videos:
            cell = ttk.Frame(self.thumbs, style="W.TFrame")
            cell.pack(side="left", padx=(0, 10))
            image = grab_thumbnail(video, 190)
            if image is not None:
                photo = ImageTk.PhotoImage(image, master=self.window)
                self._images.append(photo)
                ttk.Label(cell, image=photo, style="W.TLabel").pack()
            ttk.Label(cell, text=camera_name(video), style="W.TLabel").pack()

    def _capture_next(self) -> None:
        if self.session is None:
            messagebox.showinfo(
                "Marcha Fácil", "Escolha uma gravação na lista.", parent=self.window
            )
            return
        self.session.state = load_state(self.session.path)
        save_config_value(CONFIG_SECTION, "sessions_root", str(self.sessions_root))
        analysis = self.session.state.get("analysis_dir")
        if (
            analysis
            and Path(analysis).is_dir()
            and messagebox.askyesno(
                "Marcha Fácil",
                "Esta gravação já foi analisada.\n\nQuer ver os resultados?\n"
                "(Escolha “Não” para refazer a análise.)",
                parent=self.window,
            )
        ):
            self.source_dir = Path(self.session.state.get("source_dir", self.session.path))
            self.show_step(3)
            return
        self.show_step(1)

    # ── step 2: cut ──

    def _step_cut(self) -> None:
        assert self.session is not None
        self._title(
            "Corte o trecho da caminhada",
            "Recomendado: deixe só o trecho em que a pessoa está caminhando. A análise "
            "fica mais rápida e mais precisa. Ao terminar o corte, o assistente segue "
            "sozinho para o próximo passo.",
        )
        box = ttk.Frame(self.body, style="W.TFrame")
        box.pack(fill="x", pady=10)
        ttk.Button(
            box, text="✂   Cortar o vídeo", style="Big.TButton", command=self._open_cut
        ).pack(anchor="w", pady=6)
        session_path = self.session.path
        previous = self.session.state.get("cut_dir")
        previous_dir = Path(previous) if previous and Path(previous).is_dir() else None
        previous_dir = previous_dir or latest_cut_dir(session_path)
        if previous_dir is not None:
            ttk.Button(
                box,
                text=f"Usar o corte já feito ({session_datetime(previous_dir):%d/%m %H:%M})",
                style="Mid.TButton",
                command=lambda: self._use_source(previous_dir, cut=True),
            ).pack(anchor="w", pady=6)
        ttk.Button(
            box,
            text="Usar o vídeo inteiro (sem cortar)",
            style="Mid.TButton",
            command=lambda: self._use_source(session_path, cut=False),
        ).pack(anchor="w", pady=6)
        self.cut_status = ttk.Label(self.body, text="", style="Hint.TLabel")
        self.cut_status.pack(anchor="w", pady=10)
        tips = (
            "Como cortar: arraste o marcador verde até o início da caminhada e o "
            "vermelho até o fim. Depois clique em “Cut all cameras”."
        )
        ttk.Label(self.body, text=tips, style="Hint.TLabel", wraplength=880).pack(anchor="w")
        self.next_btn.config(text="Próximo  ▶", command=self._cut_next)
        if self.source_dir is not None and self.source_dir.is_dir():
            self.cut_status.config(text=f"Escolhido: {self.source_dir.name}")

    def _open_cut(self) -> None:
        try:
            from .multicam_cut import run_multicam_cut
        except ImportError:
            from multicam_cut import run_multicam_cut  # ty: ignore[unresolved-import]
        assert self.session is not None
        self.cut_status.config(text="Faça o corte na janela que abriu…")
        self._cut_app = run_multicam_cut(
            parent=self.window,
            session_dir=str(self.session.path),
            on_cut_done=lambda out: self._use_source(out, cut=True),
        )

    def _use_source(self, folder: Path, cut: bool) -> None:
        assert self.session is not None
        self.source_dir = Path(folder)
        state = load_state(self.session.path)
        state.update(source_dir=str(self.source_dir), cut_dir=str(folder) if cut else "")
        state.pop("analysis_dir", None)
        self.session.state = state
        save_state(self.session.path, state)
        self.show_step(2)

    def _cut_next(self) -> None:
        if self.source_dir is None or not self.source_dir.is_dir():
            messagebox.showinfo(
                "Marcha Fácil",
                "Escolha uma opção: cortar o vídeo ou usar o vídeo inteiro.",
                parent=self.window,
            )
            return
        self.show_step(2)

    # ── step 3: processing ──

    def _step_process(self) -> None:
        assert self.session is not None and self.source_dir is not None
        self._title(
            "Analisar a caminhada",
            "Identifique cada câmera (lateral, coronal/de frente, ou não usar) e clique em "
            "“Analisar”. O computador faz o resto; isso pode levar alguns minutos.",
        )
        videos = _session_videos(self.source_dir)
        available = [camera_name(v) for v in videos]
        chosen = self.session.state.get("cameras") or default_lateral_cameras(available)
        chosen_coronal = self.session.state.get("coronal_cameras") or default_coronal_cameras(
            available, chosen
        )

        state = self.session.state
        form = ttk.Frame(self.body, style="W.TFrame")
        form.pack(fill="x", pady=(0, 10))
        self.participant_var = tk.StringVar(value=state.get("participant", ""))
        self.age_var = tk.StringVar(value=state.get("age", ""))
        side_label = {v: k for k, v in AFFECTED_SIDES}.get(state.get("affected_side", ""))
        self.side_var = tk.StringVar(value=side_label or AFFECTED_SIDES[0][0])
        self.therapist_var = tk.StringVar(
            value=state.get("therapist") or load_config_value(CONFIG_SECTION, "therapist")
        )
        self.institution_var = tk.StringVar(
            value=state.get("institution") or load_config_value(CONFIG_SECTION, "institution")
        )
        lang = state.get("report_lang") or load_config_value(CONFIG_SECTION, "report_lang") or "pt"
        self.lang_var = tk.StringVar(value={v: k for k, v in REPORT_LANGS}.get(lang, "Português"))
        self._process_inputs: list[ttk.Entry | ttk.Combobox] = []
        fields = (
            ("Participante (nome ou código)", ttk.Entry, self.participant_var, None, 30),
            ("Idade", ttk.Entry, self.age_var, None, 6),
            ("Lado acometido", ttk.Combobox, self.side_var, [k for k, _ in AFFECTED_SIDES], 12),
            ("Idioma do relatório", ttk.Combobox, self.lang_var, [k for k, _ in REPORT_LANGS], 12),
            ("Terapeuta", ttk.Entry, self.therapist_var, None, 30),
            ("Instituição", ttk.Entry, self.institution_var, None, 30),
        )
        for i, (label, kind, var, values, width) in enumerate(fields):
            row, col = divmod(i, 2)
            ttk.Label(form, text=label, style="W.TLabel").grid(
                row=row, column=2 * col, sticky="w", padx=(0 if col == 0 else 24, 8), pady=3
            )
            if kind is ttk.Combobox:
                widget = ttk.Combobox(
                    form,
                    textvariable=var,
                    values=values or [],
                    width=width,
                    state="readonly",
                    font=(FONT, 11),
                )
            else:
                widget = ttk.Entry(form, textvariable=var, width=width, font=(FONT, 11))
            widget.grid(row=row, column=2 * col + 1, sticky="w", pady=3)
            self._process_inputs.append(widget)

        cams = ttk.Frame(self.body, style="W.TFrame")
        cams.pack(fill="x")
        self.camera_vars: dict[str, tk.StringVar] = {}
        role_to_label = {role: label for label, role in CAMERA_ROLES}
        from PIL import ImageTk

        for video in videos:
            name = camera_name(video)
            cell = ttk.Frame(cams, style="W.TFrame")
            cell.pack(side="left", padx=(0, 12))
            image = grab_thumbnail(video, 170)
            if image is not None:
                photo = ImageTk.PhotoImage(image, master=self.window)
                self._images.append(photo)
                ttk.Label(cell, image=photo, style="W.TLabel").pack()
            ttk.Label(cell, text=name, style="W.TLabel").pack()
            role = "coronal" if name in chosen_coronal else "lateral" if name in chosen else "nao"
            var = tk.StringVar(value=role_to_label[role])
            self.camera_vars[name] = var
            combo = ttk.Combobox(
                cell,
                textvariable=var,
                values=[k for k, _ in CAMERA_ROLES],
                width=18,
                state="readonly",
                font=(FONT, 10),
            )
            combo.pack()
            self._process_inputs.append(combo)

        self.progress_frame = ttk.Frame(self.body, style="W.TFrame")
        self.progress_frame.pack(fill="both", expand=True, pady=(14, 0))
        self.next_btn.config(text="▶  Analisar", command=self._start_processing)

    def _start_processing(self) -> None:
        assert self.session is not None and self.source_dir is not None
        label_to_role = dict(CAMERA_ROLES)
        roles = {c: label_to_role.get(v.get(), "nao") for c, v in self.camera_vars.items()}
        cameras = [c for c, role in roles.items() if role == "lateral"]
        coronal_cameras = [c for c, role in roles.items() if role == "coronal"]
        if not cameras:
            messagebox.showinfo(
                "Marcha Fácil",
                "Marque pelo menos uma câmera como Lateral (de lado).",
                parent=self.window,
            )
            return
        participant = self.participant_var.get().strip()
        save_config_value(CONFIG_SECTION, "lateral_cameras", ",".join(cameras))
        save_config_value(CONFIG_SECTION, "coronal_cameras", ",".join(coronal_cameras))
        fields = {
            "participant": participant,
            "age": self.age_var.get().strip(),
            "affected_side": dict(AFFECTED_SIDES).get(self.side_var.get(), ""),
            "therapist": self.therapist_var.get().strip(),
            "institution": self.institution_var.get().strip(),
            "report_lang": dict(REPORT_LANGS).get(self.lang_var.get(), "pt"),
        }
        for key in ("therapist", "institution", "report_lang"):
            save_config_value(CONFIG_SECTION, key, fields[key])
        state = load_state(self.session.path)
        state.update(
            cameras=cameras,
            coronal_cameras=coronal_cameras,
            source_dir=str(self.source_dir),
            **fields,
        )
        save_state(self.session.path, state)
        self.session.state = state
        info = report_info(state, self.session.name)

        for w in self.progress_frame.winfo_children():
            w.destroy()
        n_cams = len(cameras) + len(coronal_cameras)
        labels = (
            f"Detectar o corpo em {n_cams} câmera(s) — a parte mais demorada",
            "Encontrar os passos e as fases da marcha",
            "Calcular os ângulos e gerar o vídeo com os ângulos (MP Angles)",
            "Montar o relatório para imprimir (PDF)",
        )
        self.stage_labels = []
        for text in labels:
            lbl = tk.Label(self.progress_frame, text=f"○  {text}", bg=BG, fg=MUTED, font=(FONT, 12))
            lbl.pack(anchor="w", pady=2)
            self.stage_labels.append((lbl, text))
        self.bar = ttk.Progressbar(self.progress_frame, mode="indeterminate", length=600)
        self.bar.pack(anchor="w", pady=(10, 4))
        self.bar.start(12)
        self.elapsed_label = ttk.Label(self.progress_frame, text="", style="Hint.TLabel")
        self.elapsed_label.pack(anchor="w")

        self.processing = True
        self._log_lines = []
        self._started = time.monotonic()
        self.next_btn.config(state="disabled")
        self.back_btn.config(state="disabled")
        for widget in self._process_inputs:
            widget.state(["disabled"])

        source = self.source_dir

        def worker() -> None:
            try:
                gait = run_pipeline(
                    source,
                    cameras,
                    participant,
                    log=lambda m: self._events.put(("log", m)),
                    stage=lambda i: self._events.put(("stage", i)),
                    info=info,
                    coronal_cameras=coronal_cameras,
                )
                self._events.put(("done", gait))
            except Exception as exc:  # noqa: BLE001 - shown to the user
                self._events.put(("error", f"{type(exc).__name__}: {exc}"))

        threading.Thread(target=worker, daemon=True).start()
        self.window.after(200, self._pump)

    def _set_stage(self, current: int) -> None:
        for i, (lbl, text) in enumerate(self.stage_labels):
            if i < current:
                lbl.config(text=f"✔  {text}", fg=DONE, font=(FONT, 12))
            elif i == current:
                lbl.config(text=f"➜  {text}…", fg=ACCENT, font=(FONT, 12, "bold"))

    def _pump(self) -> None:
        if not self.window.winfo_exists():
            return
        while True:
            try:
                kind, payload = self._events.get_nowait()
            except queue.Empty:
                break
            if kind == "log":
                print(payload, flush=True)
                self._log_lines.append(str(payload))
            elif kind == "stage":
                self._set_stage(int(payload))
            elif kind == "done":
                self._on_processing_done(cast(GaitResult, payload))
                return
            else:
                self._on_processing_error(str(payload))
                return
        minutes, seconds = divmod(int(time.monotonic() - self._started), 60)
        self.elapsed_label.config(text=f"Tempo decorrido: {minutes}:{seconds:02d}")
        self.window.after(300, self._pump)

    def _on_processing_done(self, gait: GaitResult) -> None:
        self.processing = False
        self.bar.stop()
        self.next_btn.config(state="normal")
        self.back_btn.config(state="normal")
        assert self.session is not None
        state = load_state(self.session.path)
        state.update(
            analysis_dir=str(gait.out_dir),
            passes=gait.passes,
            feet_cut={k: (v if np.isfinite(v) else None) for k, v in gait.feet_cut.items()},
        )
        save_state(self.session.path, state)
        self.session.state = state
        self.show_step(3)

    def _on_processing_error(self, message: str) -> None:
        self.processing = False
        self.bar.stop()
        self.next_btn.config(state="normal")
        self.back_btn.config(state="normal")
        tk.Label(
            self.progress_frame,
            text="Não foi possível concluir a análise. Verifique se a gravação abre "
            "normalmente e tente de novo. Se o problema continuar, envie os detalhes "
            "abaixo para o suporte.",
            bg=BG,
            fg="#c62828",
            font=(FONT, 12, "bold"),
            wraplength=880,
            justify="left",
        ).pack(anchor="w", pady=(10, 4))
        details = tk.Text(self.progress_frame, height=6, font=("TkFixedFont", 9))
        details.insert("end", "\n".join([*self._log_lines[-15:], "", message]))
        details.config(state="disabled")
        details.pack(fill="both", expand=True)

    # ── step 4: results ──

    def _step_results(self) -> None:
        assert self.session is not None
        state = self.session.state
        out_dir = Path(state.get("analysis_dir", ""))
        feet_cut = {
            k: (float(v) if v is not None else float("nan"))
            for k, v in state.get("feet_cut", {}).items()
        }
        view = build_result_view(out_dir, state.get("passes", {}), feet_cut)
        who = f" — {state['participant']}" if state.get("participant") else ""
        self._title(
            f"Resultados{who}",
            f"{view.n_cycles} ciclo(s) de marcha medidos · gravação {self.session.name}",
        )

        cards = ttk.Frame(self.body, style="W.TFrame")
        cards.pack(fill="x", pady=(0, 10))
        for title, value, explanation in view.cards:
            card = tk.Frame(cards, bg="white", highlightbackground="#ddd", highlightthickness=1)
            card.pack(side="left", padx=(0, 10), fill="y")
            tk.Label(card, text=title, bg="white", fg="#555", font=(FONT, 10)).pack(
                anchor="w", padx=10, pady=(8, 0)
            )
            tk.Label(card, text=value, bg="white", fg=ACCENT, font=(FONT, 20, "bold")).pack(
                anchor="w", padx=10
            )
            tk.Label(
                card,
                text=explanation,
                bg="white",
                fg="#777",
                font=(FONT, 9),
                wraplength=150,
                justify="left",
            ).pack(anchor="w", padx=10, pady=(0, 8))

        for tip in view.advice:
            tk.Label(
                self.body,
                text=f"Atenção: {tip}",
                bg="#fff8e1",
                fg="#6d4c00",
                font=(FONT, 11),
                wraplength=880,
                justify="left",
                anchor="w",
            ).pack(fill="x", pady=2, ipady=4, ipadx=6)

        if view.angles_png is not None:
            from PIL import Image, ImageTk

            image = Image.open(view.angles_png)
            image.thumbnail((900, 200))
            photo = ImageTk.PhotoImage(image, master=self.window)
            self._images.append(photo)
            ttk.Label(self.body, image=photo, style="W.TLabel").pack(anchor="w", pady=(8, 0))

        main_row = ttk.Frame(self.body, style="W.TFrame")
        main_row.pack(fill="x", pady=(10, 0))
        more_row = ttk.Frame(self.body, style="W.TFrame")
        more_row.pack(fill="x", pady=(6, 0))
        # The skeleton + angles video is what patients and therapists look at
        # first: the camera that saw complete walking passes gets the big button.
        for k, (camera, video) in enumerate(view.videos):
            ttk.Button(
                main_row if k == 0 else more_row,
                text=f"▶  Ver vídeo com ângulos ({camera})" if k == 0 else f"▶  Vídeo {camera}",
                style="Big.TButton" if k == 0 else "Mid.TButton",
                command=lambda v=video: open_path(v),
            ).pack(side="left", padx=(0, 8))
        lang = state.get("report_lang") or "pt"
        other = "en" if lang == "pt" else "pt"
        ttk.Button(
            main_row,
            text="Relatório para imprimir" if lang == "pt" else "Printable report",
            style="Big.TButton",
            command=lambda: self._open_print_report(out_dir, lang),
        ).pack(side="left", padx=(0, 8))
        ttk.Button(
            more_row,
            text="English version" if other == "en" else "Versão em português",
            style="Mid.TButton",
            command=lambda: self._open_print_report(out_dir, other),
        ).pack(side="left", padx=(0, 8))
        if view.report.is_file():
            ttk.Button(
                more_row,
                text="Relatório técnico (HTML)",
                style="Mid.TButton",
                command=lambda: open_path(view.report),
            ).pack(side="left", padx=(0, 8))
        ttk.Button(
            more_row,
            text="Abrir pasta dos resultados",
            style="Mid.TButton",
            command=lambda: open_path(view.out_dir),
        ).pack(side="left", padx=(0, 8))
        self.next_btn.config(text="Analisar outra gravação", command=self._restart)

    def _open_print_report(self, out_dir: Path, lang: str) -> None:
        """Open the printable report, building it first for analyses made before it existed."""
        path = print_report(out_dir, lang)
        if path is None and self.session is not None:
            self.window.config(cursor="watch")
            self.window.update_idletasks()
            try:
                write_reports(out_dir, report_info(self.session.state, self.session.name))
            except Exception as exc:  # noqa: BLE001 - shown to the user
                messagebox.showerror(
                    "Marcha Fácil",
                    f"Não foi possível montar o relatório:\n{exc}",
                    parent=self.window,
                )
            finally:
                self.window.config(cursor="")
            path = print_report(out_dir, lang)
        if path is not None:
            open_path(path)

    def _restart(self) -> None:
        self.source_dir = None
        self.show_step(0)

    # ── teardown ──

    def on_close(self) -> None:
        if self.processing and not messagebox.askyesno(
            "Marcha Fácil",
            "A análise ainda está em andamento. Fechar mesmo assim?\n"
            "(O progresso desta análise será perdido.)",
            parent=self.window,
        ):
            return
        self.window.destroy()


def run_gait_wizard(
    parent: tk.Tk | tk.Toplevel | None = None, sessions_root: str | None = None
) -> GaitWizardApp:
    """Open the step-by-step gait wizard."""
    print(f"Running script: {Path(__file__).name}")
    print(f"Script directory: {Path(__file__).parent}")
    print("Starting gait_wizard...")

    created_root = False
    default_root = cast("tk.Tk | tk.Toplevel | None", getattr(tk, "_default_root", None))
    root = parent or default_root
    if root is None:
        root = tk.Tk()
        root.withdraw()
        created_root = True

    window = tk.Toplevel(root)
    app = GaitWizardApp(window, sessions_root=sessions_root)

    if created_root:
        window.bind("<Destroy>", lambda e: root.quit() if e.widget is window else None)
        root.mainloop()
    return app


if __name__ == "__main__":
    run_gait_wizard(sessions_root=sys.argv[1] if len(sys.argv) > 1 else None)
