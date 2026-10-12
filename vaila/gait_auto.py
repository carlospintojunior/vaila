"""
Project: vailá Multimodal Toolbox
Script: gait_auto.py - Automatic Gait Analysis (stance / swing)

Author: Carlos Pinto Jr
Email: carlos@vulcanum.com.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 10 October 2026
Update Date: 12 October 2026
Version: 0.4.8

Description:
Turns a "Record Cameras" session of people walking past side-view (sagittal)
cameras into temporal gait parameters, with no user interaction:

1. MediaPipe pose (Markerless 2D, headless batch) on each lateral camera.
2. Frame quality gate: key leg landmarks visible, feet inside the image,
   pelvis moving — drops empty-room false detections and the partial bodies
   of a subject entering/leaving the field of view.
3. Pass segmentation: each continuous, single-direction crossing of the
   camera's field of view becomes one "pass".
4. Gait events per foot (Zeni et al., 2008, relative to the pelvis, in the
   image's horizontal axis): heel-strike (HS) = most anterior heel, toe-off
   (TO) = most posterior toe; peaks are refined to sub-frame precision.
5. Per gait cycle: stance, swing, stride (s and % of cycle), step time,
   double support, cadence; per-side means and a symmetry index.
6. Self-contained HTML report + CSVs written into the session folder.

Times come from the real packet timestamps of each video (webcams do not run
at exactly their nominal rate), so they hold for variable frame rates.

Only temporal parameters are computed: no camera calibration is needed.
Cameras are analysed independently (no synchronization needed); the pooled
summary therefore counts a step seen by two cameras twice.

Output:
<session_dir>/gait_analysis_YYYYMMDD_HHMMSS/
    gait_report.html     visual report (signals with events, tables)
    gait_events.csv      one row per detected HS / TO
    gait_cycles.csv      one row per complete gait cycle
    gait_summary.csv     per camera / side (+ pooled) means and SDs
    mediapipe/           MediaPipe outputs per camera (CSVs, annotated video)

Usage:
    uv run python vaila/gait_auto.py --session <session_dir> [--cameras cam2 cam3]
        [--participant ID] [--mediapipe-dir <existing mediapipe_cli_* dir>]
        [--no-open] [--progress-window]

Reference:
Zeni JA, Richards JG, Higginson JS. Two simple methods for determining gait
events during treadmill and overground walking using kinematic data.
Gait Posture. 2008;27(4):710-714.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import html
import io
import os
import shutil
import subprocess
import sys
import tempfile
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

try:
    from .filter_utils import butter_filter
    from .multicam_cut import natural_sort_key, read_frame_times
except ImportError:
    from filter_utils import butter_filter  # ty: ignore[unresolved-import]
    from multicam_cut import natural_sort_key, read_frame_times  # ty: ignore[unresolved-import]


# ── Tunables ─────────────────────────────────────────────────────────────────

DEFAULT_CAMERAS = ("cam2", "cam3")
MEDIAPIPE_CONFIG = """[mediapipe]
min_detection_confidence = 0.5
min_tracking_confidence = 0.5
model_complexity = 2
export_world_landmarks = true
"""
LEG_LANDMARKS = (
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)
MIN_VISIBILITY = 0.5
EDGE_MARGIN = 0.01  # fraction of width/height a foot must stay inside the image
BOTTOM_TOLERANCE = 0.10  # feet may be extrapolated this fraction of height below the image
MIN_LEG_FRACTION = 0.12  # hip-to-ankle height (px) / image height: smaller = not a person
MAX_JUMP_LEG = 0.5  # heel/hip displacement between frames (x leg length): larger = glitch
MIN_PASS_SECONDS = 1.0  # shorter crossings cannot hold a complete gait cycle
MAX_GAP_FRAMES = 3  # bad frames bridged (interpolated) inside one pass
MIN_PELVIS_SPEED = 0.15  # leg lengths per second: slower = standing/turning
LOWPASS_HZ = 6.0
MIN_EVENT_SPACING_S = 0.35  # same-foot events closer than this are noise
PEAK_PROMINENCE = 0.25  # x leg length (px); smaller peaks = entering/leaving noise
STRIDE_RANGE_S = (0.6, 2.5)  # plausible stride durations for walking


# ── Data containers ──────────────────────────────────────────────────────────


@dataclass
class CameraData:
    name: str
    video: Path
    pixel: pd.DataFrame  # indexed by frame, <landmark>_x/_y columns (px)
    visibility: pd.DataFrame  # indexed by frame, <landmark> columns (0..1)
    times: np.ndarray  # real timestamp (s) of each frame
    width: int
    height: int


@dataclass
class Pass:
    camera: str
    index: int
    start: int  # first frame (inclusive)
    end: int  # last frame (inclusive)
    direction: int  # +1 walking towards +x in the image, -1 towards -x
    signals: pd.DataFrame = field(repr=False)  # filtered Zeni signals per frame
    leg_length_px: float = 0.0


@dataclass
class GaitEvent:
    camera: str
    pass_index: int
    side: str  # "left" | "right"
    kind: str  # "HS" | "TO"
    frame: float  # sub-frame position
    time: float  # seconds


@dataclass
class GaitResult:
    report: Path
    out_dir: Path
    mediapipe_run: Path
    videos: list[Path]
    cycles: pd.DataFrame
    summary: pd.DataFrame
    passes: dict[str, int]  # camera -> valid walking passes
    feet_cut: dict[str, float]  # camera -> fraction of person frames with feet cut off
    warnings: list[str]


# ── MediaPipe stage ──────────────────────────────────────────────────────────


def select_camera_videos(session_dir: Path, cameras: list[str]) -> list[Path]:
    """Return the session videos whose stem starts with one of ``cameras``."""
    videos = sorted(
        (p for p in session_dir.iterdir() if p.suffix.lower() in (".mp4", ".avi", ".mov")),
        key=natural_sort_key,
    )
    chosen = [v for v in videos if any(v.stem.split("_")[0] == cam for cam in cameras)]
    if not chosen:
        raise FileNotFoundError(
            f"No videos for cameras {cameras} in {session_dir} (found: {[v.name for v in videos]})"
        )
    return chosen


def run_mediapipe(videos: list[Path], output_dir: Path, log: Callable[[str], None]) -> Path:
    """Run the Markerless 2D headless batch on ``videos``; return its output dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("markerless_2d_analysis.py")
    with tempfile.TemporaryDirectory(prefix="vaila_gait_") as tmp:
        tmp_dir = Path(tmp)
        input_dir = tmp_dir / "videos"
        input_dir.mkdir()
        for video in videos:
            link = input_dir / video.name
            try:
                link.symlink_to(video.resolve())
            except OSError:  # e.g. Windows without symlink rights
                shutil.copy2(video, link)
        config = tmp_dir / "gait_mediapipe.toml"
        config.write_text(MEDIAPIPE_CONFIG, encoding="utf-8")
        cmd = [
            sys.executable,
            str(script),
            "batch",
            "-i",
            str(input_dir),
            "-o",
            str(output_dir),
            "-c",
            str(config),
            "--no-sleep-between-videos",
        ]
        log(f"MediaPipe: {len(videos)} video(s) — this takes a few minutes on CPU")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            if line.startswith(("Processing video", "Processing complete", "[OK]", "Error")):
                log(f"  {line}")
        if proc.wait() != 0:
            raise RuntimeError(f"MediaPipe batch failed (exit code {proc.returncode})")
    runs = sorted(output_dir.glob("mediapipe_cli_*"))
    if not runs:
        raise RuntimeError(f"MediaPipe produced no output under {output_dir}")
    return runs[-1]


# ── Loading ──────────────────────────────────────────────────────────────────


def _video_size(video: Path) -> tuple[int, int, int, float]:
    import cv2

    cap = cv2.VideoCapture(str(video))
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    finally:
        cap.release()
    return width, height, count, fps


def load_camera(name: str, video: Path, mediapipe_run: Path) -> CameraData:
    """Load one camera's MediaPipe pixel + visibility CSVs and frame times."""
    stem_dir = mediapipe_run / video.stem
    pixel = pd.read_csv(stem_dir / f"{video.stem}_mp_pixel.csv").set_index("frame_index")
    world = pd.read_csv(stem_dir / f"{video.stem}_mp_world.csv")
    frame_col = "frame_index" if "frame_index" in world.columns else world.columns[0]
    world = world.set_index(frame_col)
    visibility = pd.DataFrame(
        {lm: world[f"{lm}_visibility"] for lm in LEG_LANDMARKS if f"{lm}_visibility" in world},
        index=world.index,
    ).reindex(pixel.index)
    # The headless batch currently writes an all-empty *_mp_world.csv; treat
    # visibility as unavailable then (the geometric checks still apply).
    visibility = visibility.loc[:, visibility.notna().any()]
    width, height, count, fps = _video_size(video)
    times = read_frame_times(video, max(count, len(pixel)), fps)
    return CameraData(name, video, pixel, visibility, times, width, height)


# ── Quality gate & pass segmentation ─────────────────────────────────────────


def feet_cut_fraction(cam: CameraData) -> float:
    """Fraction of person-sized detections whose heels fall below the image.

    MediaPipe extrapolates feet that the camera does not see; heel-strike and
    toe-off timing from such frames is guesswork, so the report flags it.
    """
    px = cam.pixel
    hip_y = (px["left_hip_y"] + px["right_hip_y"]) / 2
    ankle_y = (px["left_ankle_y"] + px["right_ankle_y"]) / 2
    person = (ankle_y - hip_y) > MIN_LEG_FRACTION * cam.height
    if not person.any():
        return float("nan")
    heel_y = px.loc[person, ["left_heel_y", "right_heel_y"]].max(axis=1)
    return float((heel_y > cam.height).mean())


def frame_quality(cam: CameraData) -> np.ndarray:
    """Boolean per frame: whole legs detected, plausible, visible and inside the image."""
    px = cam.pixel
    ok = np.ones(len(px), dtype=bool)
    for lm in LEG_LANDMARKS:
        x = px[f"{lm}_x"].to_numpy(dtype=float)
        y = px[f"{lm}_y"].to_numpy(dtype=float)
        ok &= np.isfinite(x) & np.isfinite(y)
        if lm in cam.visibility:
            ok &= cam.visibility[lm].fillna(0).to_numpy() >= MIN_VISIBILITY
    hip_y = (px["left_hip_y"] + px["right_hip_y"]).to_numpy(dtype=float) / 2
    ankle_y = (px["left_ankle_y"] + px["right_ankle_y"]).to_numpy(dtype=float) / 2
    leg = ankle_y - hip_y  # image y grows downwards: ankles below hips
    with np.errstate(invalid="ignore"):
        ok &= leg > MIN_LEG_FRACTION * cam.height
    # Single-frame glitches (a leg snapping to a wrong blob, left/right flips)
    leg_ref = np.nanmedian(np.where(ok, leg, np.nan)) if ok.any() else np.nan
    if np.isfinite(leg_ref):
        for lm in ("left_heel", "right_heel", "left_hip", "right_hip"):
            x = px[f"{lm}_x"].to_numpy(dtype=float)
            y = px[f"{lm}_y"].to_numpy(dtype=float)
            jump = np.hypot(np.diff(x, prepend=np.nan), np.diff(y, prepend=np.nan))
            with np.errstate(invalid="ignore"):
                ok &= ~(jump > MAX_JUMP_LEG * leg_ref)
    mx, my = EDGE_MARGIN * cam.width, EDGE_MARGIN * cam.height
    for lm in ("left_heel", "right_heel", "left_foot_index", "right_foot_index"):
        x = px[f"{lm}_x"].to_numpy(dtype=float)
        y = px[f"{lm}_y"].to_numpy(dtype=float)
        with np.errstate(invalid="ignore"):
            # MediaPipe extrapolates feet slightly cut by the bottom edge reasonably
            # well; left/right edges mean the subject is entering/leaving the view.
            ok &= (x > mx) & (x < cam.width - mx) & (y > my)
            ok &= y < cam.height * (1 + BOTTOM_TOLERANCE)
    return ok


def _runs(mask: np.ndarray, max_gap: int) -> list[tuple[int, int]]:
    """Contiguous True runs (inclusive bounds), bridging gaps of <= max_gap."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    runs, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > max_gap + 1:
            runs.append((int(start), int(prev)))
            start = i
        prev = i
    runs.append((int(start), int(prev)))
    return runs


def _fill(values: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaNs (gaps already bounded by the pass segmentation)."""
    out = values.astype(float).copy()
    bad = ~np.isfinite(out)
    if bad.all():
        return out
    if bad.any():
        good = np.flatnonzero(~bad)
        out[bad] = np.interp(np.flatnonzero(bad), good, out[good])
    return out


def segment_passes(cam: CameraData) -> list[Pass]:
    """Split the quality-gated frames into single-direction walking passes."""
    ok = frame_quality(cam)
    px = cam.pixel
    times = cam.times[: len(px)]
    period = float(np.median(np.diff(times))) if len(times) > 1 else 1 / 30
    fs = 1.0 / period
    min_frames = int(round(MIN_PASS_SECONDS * fs))
    passes: list[Pass] = []
    for a, b in _runs(ok, MAX_GAP_FRAMES):
        if b - a + 1 < min_frames:
            continue
        seg = px.iloc[a : b + 1].copy()
        seg.loc[~ok[a : b + 1]] = np.nan  # bridged bad frames get interpolated
        cols = {}
        for lm in LEG_LANDMARKS:
            for axis in ("x", "y"):
                raw = _fill(seg[f"{lm}_{axis}"].to_numpy())
                cols[f"{lm}_{axis}"] = butter_filter(raw, fs, cutoff=LOWPASS_HZ)
        sig = pd.DataFrame(cols, index=seg.index)
        pelvis_x = (sig["left_hip_x"] + sig["right_hip_x"]) / 2
        leg = float(
            np.nanmedian(
                np.abs(
                    (sig["left_hip_y"] + sig["right_hip_y"]) / 2
                    - (sig["left_ankle_y"] + sig["right_ankle_y"]) / 2
                )
            )
        )
        if not np.isfinite(leg) or leg <= 0:
            continue
        # Split where the pelvis stops or reverses (turns at the end of the walkway).
        velocity = np.gradient(pelvis_x.to_numpy()) * fs / leg
        moving = np.abs(velocity) >= MIN_PELVIS_SPEED
        direction = np.sign(velocity)
        for c, d in _runs(moving, MAX_GAP_FRAMES):
            sub_dir = direction[c : d + 1]
            # keep the dominant direction's longest stretch
            dom = 1 if np.sum(sub_dir > 0) >= np.sum(sub_dir < 0) else -1
            for e, f in _runs(sub_dir == dom, MAX_GAP_FRAMES):
                if f - e + 1 < min_frames:
                    continue
                part = sig.iloc[c + e : c + f + 1].copy()
                pel = (part["left_hip_x"] + part["right_hip_x"]) / 2
                for side in ("left", "right"):
                    part[f"{side}_hs_signal"] = (part[f"{side}_heel_x"] - pel) * dom
                    part[f"{side}_to_signal"] = -(part[f"{side}_foot_index_x"] - pel) * dom
                passes.append(
                    Pass(
                        camera=cam.name,
                        index=len(passes) + 1,
                        start=int(part.index[0]),
                        end=int(part.index[-1]),
                        direction=dom,
                        signals=part,
                        leg_length_px=leg,
                    )
                )
    return passes


# ── Gait events ──────────────────────────────────────────────────────────────


def _refine_peak(y: np.ndarray, i: int) -> float:
    """Sub-sample peak position by fitting a parabola through i-1, i, i+1."""
    if i <= 0 or i >= len(y) - 1:
        return float(i)
    denom = y[i - 1] - 2 * y[i] + y[i + 1]
    if denom == 0:
        return float(i)
    return float(i + 0.5 * (y[i - 1] - y[i + 1]) / denom)


def detect_events(p: Pass, times: np.ndarray) -> list[GaitEvent]:
    """Zeni HS/TO events for both feet inside one pass."""
    frames = p.signals.index.to_numpy(dtype=float)
    seg_times = np.interp(frames, np.arange(len(times)), times)
    fs = 1.0 / float(np.median(np.diff(seg_times))) if len(seg_times) > 1 else 30.0
    distance = max(1, int(MIN_EVENT_SPACING_S * fs))
    prominence = PEAK_PROMINENCE * p.leg_length_px
    events: list[GaitEvent] = []
    for side in ("left", "right"):
        for kind, column in (("HS", f"{side}_hs_signal"), ("TO", f"{side}_to_signal")):
            y = p.signals[column].to_numpy(dtype=float)
            peaks, _ = find_peaks(y, distance=distance, prominence=prominence)
            for i in peaks:
                pos = _refine_peak(y, int(i))
                frame = float(np.interp(pos, np.arange(len(frames)), frames))
                t = float(np.interp(frame, np.arange(len(times)), times))
                events.append(GaitEvent(p.camera, p.index, side, kind, frame, t))
    events.sort(key=lambda e: e.time)
    return events


# ── Gait cycles & parameters ─────────────────────────────────────────────────


def _first_after(events: list[GaitEvent], t: float, side: str, kind: str) -> GaitEvent | None:
    return next((e for e in events if e.time > t and e.side == side and e.kind == kind), None)


def _last_before(events: list[GaitEvent], t: float, side: str, kind: str) -> GaitEvent | None:
    found = [e for e in events if e.time < t and e.side == side and e.kind == kind]
    return found[-1] if found else None


def gait_cycles(events: list[GaitEvent]) -> list[dict]:
    """Complete cycles (HS -> TO -> next HS of the same foot) inside one pass.

    A cycle is kept only if its stride is plausible and exactly one HS of the
    other foot falls inside it (left/right alternation — guards against
    MediaPipe swapping the legs in a side view).
    """
    rows = []
    for hs in (e for e in events if e.kind == "HS"):
        other = "right" if hs.side == "left" else "left"
        to = _first_after(events, hs.time, hs.side, "TO")
        nxt = _first_after(events, hs.time, hs.side, "HS")
        if to is None or nxt is None or not (hs.time < to.time < nxt.time):
            continue
        stride = nxt.time - hs.time
        if not STRIDE_RANGE_S[0] <= stride <= STRIDE_RANGE_S[1]:
            continue
        opp_hs = [
            e for e in events if e.side == other and e.kind == "HS" and hs.time < e.time < nxt.time
        ]
        if len(opp_hs) != 1:
            continue
        stance = to.time - hs.time
        swing = nxt.time - to.time
        row = {
            "camera": hs.camera,
            "pass": hs.pass_index,
            "side": hs.side,
            "hs_time_s": round(hs.time, 4),
            "to_time_s": round(to.time, 4),
            "next_hs_time_s": round(nxt.time, 4),
            "stride_time_s": stride,
            "stance_time_s": stance,
            "swing_time_s": swing,
            "stance_pct": 100 * stance / stride,
            "swing_pct": 100 * swing / stride,
            "step_time_s": opp_hs[0].time - hs.time,
            "cadence_steps_min": 120.0 / stride,
            "double_support_s": np.nan,
            "double_support_pct": np.nan,
        }
        # Initial DS: HS -> other foot's TO; terminal DS: other foot's HS -> own TO.
        opp_to = _first_after(events, hs.time, other, "TO")
        opp_hs_before_to = _last_before(events, to.time, other, "HS")
        if (
            opp_to is not None
            and opp_to.time < to.time
            and opp_hs_before_to is not None
            and opp_hs_before_to.time > hs.time
        ):
            ds = (opp_to.time - hs.time) + (to.time - opp_hs_before_to.time)
            row["double_support_s"] = ds
            row["double_support_pct"] = 100 * ds / stride
        rows.append(row)
    return rows


SUMMARY_METRICS = (
    "stride_time_s",
    "stance_time_s",
    "swing_time_s",
    "stance_pct",
    "swing_pct",
    "step_time_s",
    "double_support_s",
    "double_support_pct",
    "cadence_steps_min",
)


def summarize(cycles: pd.DataFrame) -> pd.DataFrame:
    """Mean/SD per camera and side, plus pooled rows and a symmetry index."""
    if cycles.empty:
        return pd.DataFrame()
    groups = []
    for camera in [*sorted(cycles["camera"].unique(), key=str), "all"]:
        sub_cam = cycles if camera == "all" else cycles[cycles["camera"] == camera]
        for side in ("left", "right", "both"):
            sub = sub_cam if side == "both" else sub_cam[sub_cam["side"] == side]
            row: dict = {"camera": camera, "side": side, "n_cycles": len(sub)}
            for m in SUMMARY_METRICS:
                row[f"{m}_mean"] = sub[m].mean() if len(sub) else np.nan
                row[f"{m}_sd"] = sub[m].std(ddof=1) if len(sub) > 1 else np.nan
            groups.append(row)
    summary = pd.DataFrame(groups)
    return summary


def symmetry_index(summary: pd.DataFrame, metric: str, camera: str = "all") -> float:
    """Robinson symmetry index (%) between left and right means; 0 = symmetric."""
    try:
        rows = summary[summary["camera"] == camera].set_index("side")[f"{metric}_mean"]
        left, right = float(rows["left"]), float(rows["right"])
    except (IndexError, KeyError):
        return float("nan")
    if not (np.isfinite(left) and np.isfinite(right)) or left + right == 0:
        return float("nan")
    return 100 * (left - right) / (0.5 * (left + right))


# ── Report ───────────────────────────────────────────────────────────────────


def _figure_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80, bbox_inches="tight")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def plot_pass(p: Pass, events: list[GaitEvent], times: np.ndarray) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = p.signals.index.to_numpy(dtype=float)
    t = np.interp(frames, np.arange(len(times)), times)
    fig, ax = plt.subplots(figsize=(9, 2.8))
    colors = {"left": "#1f77b4", "right": "#d62728"}
    for side, c in colors.items():
        ax.plot(t, p.signals[f"{side}_hs_signal"], c=c, lw=1.4, label=f"{side} heel (HS signal)")
        ax.plot(t, p.signals[f"{side}_to_signal"], c=c, lw=1, ls="--", label=f"{side} toe (TO)")
    for e in events:
        if e.pass_index != p.index:
            continue
        ax.axvline(e.time, c=colors[e.side], alpha=0.35, lw=1, ls="-" if e.kind == "HS" else ":")
        ax.text(e.time, ax.get_ylim()[1], e.kind, color=colors[e.side], fontsize=7, ha="center")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("px rel. pelvis")
    ax.set_title(
        f"{p.camera} · pass {p.index} · frames {p.start}–{p.end} · "
        f"walking {'→' if p.direction > 0 else '←'}",
        fontsize=9,
    )
    ax.legend(fontsize=6, ncol=4, loc="lower left")
    ax.grid(alpha=0.3)
    data = _figure_base64(fig)
    plt.close(fig)
    return data


def _fmt(mean: float, sd: float, digits: int = 3) -> str:
    if not np.isfinite(mean):
        return "–"
    return f"{mean:.{digits}f}" + (f" ± {sd:.{digits}f}" if np.isfinite(sd) else "")


def build_report(
    session_dir: Path,
    participant: str,
    summary: pd.DataFrame,
    cycles: pd.DataFrame,
    pass_plots: list[str],
    notes: list[str],
    warnings: list[str] | None = None,
) -> str:
    labels = [
        ("Stance (s)", "stance_time_s", 3),
        ("Swing (s)", "swing_time_s", 3),
        ("Stance (% cycle)", "stance_pct", 1),
        ("Swing (% cycle)", "swing_pct", 1),
        ("Stride time (s)", "stride_time_s", 3),
        ("Step time (s)", "step_time_s", 3),
        ("Double support (% cycle)", "double_support_pct", 1),
        ("Cadence (steps/min)", "cadence_steps_min", 1),
    ]

    def table(camera: str) -> str:
        rows = []
        head = "<tr><th>Parameter</th><th>Left</th><th>Right</th><th>Both</th></tr>"
        n = {
            side: int(
                summary[(summary["camera"] == camera) & (summary["side"] == side)]["n_cycles"].iloc[
                    0
                ]
            )
            for side in ("left", "right", "both")
        }
        rows.append(
            f"<tr class='n'><td>Cycles (n)</td><td>{n['left']}</td>"
            f"<td>{n['right']}</td><td>{n['both']}</td></tr>"
        )
        for label, metric, digits in labels:
            cells = []
            for side in ("left", "right", "both"):
                r = summary[(summary["camera"] == camera) & (summary["side"] == side)].iloc[0]
                cells.append(f"<td>{_fmt(r[f'{metric}_mean'], r[f'{metric}_sd'], digits)}</td>")
            rows.append(f"<tr><td>{label}</td>{''.join(cells)}</tr>")
        return f"<table>{head}{''.join(rows)}</table>"

    if summary.empty:
        tables = "<p class='warn'>No complete gait cycle detected.</p>"
    else:
        sym = symmetry_index(summary, "stance_time_s")
        tables = "<h2>All cameras (pooled)</h2>" + table("all")
        tables += (
            f"<p>Stance-time symmetry index (left vs right): "
            f"<b>{sym:.1f} %</b> (0 % = symmetric)</p>"
            if np.isfinite(sym)
            else ""
        )
        for camera in sorted(c for c in summary["camera"].unique() if c != "all"):
            tables += f"<h3>{html.escape(str(camera))}</h3>" + table(camera)

    figures = "".join(f"<img src='data:image/png;base64,{b}' alt='pass'>" for b in pass_plots)
    cycle_table = (
        cycles.round(3).to_html(index=False, classes="cycles", border=0) if len(cycles) else ""
    )
    notes_html = "".join(f"<li>{html.escape(n)}</li>" for n in notes)
    warn_html = "".join(f"<p class='warn'>⚠ {html.escape(w)}</p>" for w in warnings or [])
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gait Analysis Report</title>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 24px; color: #222; max-width: 1000px; }}
 h1 {{ font-size: 1.5em; margin-bottom: 0; }} .meta {{ color: #666; margin-top: 4px; }}
 table {{ border-collapse: collapse; margin: 8px 0 16px; font-size: 0.92em; }}
 th, td {{ border-bottom: 1px solid #ddd; padding: 4px 12px; text-align: right; }}
 th:first-child, td:first-child {{ text-align: left; }}
 tr.n td {{ color: #666; }} .warn {{ color: #b00; font-weight: bold; }}
 img {{ max-width: 100%; display: block; margin: 6px 0; }}
 table.cycles {{ font-size: 0.8em; }} details {{ margin-top: 16px; }}
 ul.notes {{ color: #555; font-size: 0.9em; }}
</style></head><body>
<h1>Gait analysis — stance &amp; swing</h1>
<p class="meta">Participant: <b>{html.escape(participant or "–")}</b> ·
Session: {html.escape(session_dir.name)} ·
Generated {datetime.now():%Y-%m-%d %H:%M} by <i>vailá</i> gait_auto</p>
{warn_html}
{tables}
<h2>Gait events per pass</h2>
<p>Solid lines: heel-strike (HS); dotted: toe-off (TO). Blue = left, red = right.</p>
{figures or "<p class='warn'>No valid walking pass found.</p>"}
<details><summary>All gait cycles ({len(cycles)})</summary>{cycle_table}</details>
<h2>Method notes</h2><ul class="notes">{notes_html}</ul>
</body></html>
"""


# ── Orchestration ────────────────────────────────────────────────────────────


def analyze_session(
    session_dir: Path,
    cameras: list[str] | None = None,
    participant: str = "",
    mediapipe_dir: Path | None = None,
    log: Callable[[str], None] = print,
) -> Path:
    """Run the full pipeline on one session; return the HTML report path."""
    return analyze_session_full(session_dir, cameras, participant, mediapipe_dir, log).report


def analyze_session_full(
    session_dir: Path,
    cameras: list[str] | None = None,
    participant: str = "",
    mediapipe_dir: Path | None = None,
    log: Callable[[str], None] = print,
) -> GaitResult:
    """Run the full pipeline on one session; return the report path and the data."""
    session_dir = Path(session_dir).resolve()
    cameras = list(cameras or DEFAULT_CAMERAS)
    out_dir = session_dir / f"gait_analysis_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    videos = select_camera_videos(session_dir, cameras)
    log(f"Session: {session_dir}")
    log(f"Lateral cameras: {', '.join(v.name for v in videos)}")

    run = (
        Path(mediapipe_dir) if mediapipe_dir else run_mediapipe(videos, out_dir / "mediapipe", log)
    )

    all_events: list[GaitEvent] = []
    all_cycles: list[dict] = []
    plots: list[str] = []
    notes = [
        "Events: Zeni et al. (2008) relative-distance method in the image's horizontal "
        "axis (HS = heel most anterior to the pelvis, TO = toe most posterior), "
        f"{LOWPASS_HZ:g} Hz low-pass, sub-frame peak refinement.",
        "Times from each video's real packet timestamps; cameras analysed independently "
        "(pooled values count a step seen by two cameras twice).",
        f"Frames kept only with all leg landmarks visible (≥ {MIN_VISIBILITY}) and feet "
        f"inside the image; passes shorter than {MIN_PASS_SECONDS:g} s discarded.",
        "Cycles kept only with alternating left/right heel-strikes and a stride of "
        f"{STRIDE_RANGE_S[0]}–{STRIDE_RANGE_S[1]} s.",
    ]
    warnings: list[str] = []
    n_passes: dict[str, int] = {}
    feet_cut: dict[str, float] = {}
    for video in videos:
        name = video.stem.split("_")[0]
        cam = load_camera(name, video, run)
        cut = feet_cut_fraction(cam)
        feet_cut[name] = cut
        if np.isfinite(cut) and cut > 0.3:
            msg = (
                f"{name}: the feet are below the bottom of the image in {100 * cut:.0f} % "
                "of the frames with a person — reframe this camera so the whole feet "
                "(with some floor) are visible; foot events from it are unreliable."
            )
            warnings.append(msg)
            log(f"WARNING {msg}")
        passes = segment_passes(cam)
        n_passes[name] = len(passes)
        fps = 1 / float(np.median(np.diff(cam.times))) if len(cam.times) > 1 else float("nan")
        log(f"{name}: {len(passes)} walking pass(es), {fps:.2f} fps")
        notes.append(f"{name}: {cam.width}x{cam.height} @ {fps:.2f} fps, {len(passes)} pass(es).")
        for p in passes:
            events = detect_events(p, cam.times)
            cycles = gait_cycles(events)
            all_events.extend(events)
            all_cycles.extend(cycles)
            plots.append(plot_pass(p, events, cam.times))
            log(f"  pass {p.index}: {len(events)} events, {len(cycles)} complete cycle(s)")

    events_df = pd.DataFrame([e.__dict__ for e in all_events])
    cycles_df = pd.DataFrame(all_cycles)
    summary = summarize(cycles_df)
    if participant:
        for df in (events_df, cycles_df, summary):
            if not df.empty:
                df.insert(0, "participant", participant)
    events_df.to_csv(out_dir / "gait_events.csv", index=False)
    cycles_df.to_csv(out_dir / "gait_cycles.csv", index=False)
    summary.to_csv(out_dir / "gait_summary.csv", index=False)

    report = out_dir / "gait_report.html"
    report.write_text(
        build_report(
            session_dir,
            participant,
            summary.drop(columns="participant", errors="ignore"),
            cycles_df.drop(columns="participant", errors="ignore"),
            plots,
            notes,
            warnings,
        ),
        encoding="utf-8",
    )
    log(f"{len(cycles_df)} complete gait cycle(s). Report: {report}")
    return GaitResult(
        report, out_dir, run, videos, cycles_df, summary, n_passes, feet_cut, warnings
    )


# ── Progress window (when launched from Record Cameras) ──────────────────────


def run_with_progress_window(args: argparse.Namespace) -> int:
    """Run the analysis on a worker thread while a small Tk window shows progress."""
    import queue
    import threading
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("vailá - Gait Analysis")
    frame = ttk.Frame(root, padding=10)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text=f"Analysing gait: {Path(args.session).name}").pack(anchor="w")
    bar = ttk.Progressbar(frame, mode="indeterminate", length=420)
    bar.pack(fill="x", pady=6)
    bar.start(12)
    text = tk.Text(frame, height=14, width=80, state="disabled", font=("TkFixedFont", 9))
    text.pack(fill="both", expand=True)
    messages: queue.Queue[tuple[str, str]] = queue.Queue()
    result: dict[str, object] = {}

    def log(msg: str) -> None:
        print(msg, flush=True)
        messages.put(("log", msg))

    def worker() -> None:
        try:
            result["report"] = analyze_session(
                Path(args.session),
                args.cameras,
                args.participant,
                Path(args.mediapipe_dir) if args.mediapipe_dir else None,
                log,
            )
            messages.put(("done", ""))
        except Exception as exc:  # noqa: BLE001 - surfaced to the user in the window
            messages.put(("error", f"{type(exc).__name__}: {exc}"))

    def pump() -> None:
        while True:
            try:
                kind, msg = messages.get_nowait()
            except queue.Empty:
                break
            if kind == "log":
                text.config(state="normal")
                text.insert("end", msg + "\n")
                text.see("end")
                text.config(state="disabled")
            elif kind == "done":
                bar.stop()
                if not args.no_open:
                    webbrowser.open(Path(str(result["report"])).as_uri())
                root.after(1500, root.destroy)
                return
            else:
                bar.stop()
                text.config(state="normal")
                text.insert("end", f"\nERROR: {msg}\n")
                text.config(state="disabled")
                ttk.Button(frame, text="Close", command=root.destroy).pack(pady=6)
                result["error"] = msg
                return
        root.after(200, pump)

    threading.Thread(target=worker, daemon=True).start()
    root.after(200, pump)
    root.mainloop()
    return 1 if "error" in result else 0


def launch_detached(
    session_dir: Path, cameras: list[str], participant: str, extra: list[str] | None = None
) -> subprocess.Popen:
    """Start the analysis (with progress window) as an independent process."""
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--session",
        str(session_dir),
        "--progress-window",
        "--cameras",
        *cameras,
    ]
    if participant:
        cmd += ["--participant", participant]
    cmd += extra or []
    env = os.environ.copy()
    with contextlib.suppress(Exception):
        env.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
    return subprocess.Popen(cmd, env=env, start_new_session=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Automatic gait analysis (stance/swing) from lateral camera videos."
    )
    parser.add_argument("--session", required=True, help="Record Cameras session folder")
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=list(DEFAULT_CAMERAS),
        help="Lateral camera prefixes to analyse (default: cam2 cam3)",
    )
    parser.add_argument("--participant", default="", help="Participant ID for the report")
    parser.add_argument(
        "--mediapipe-dir",
        help="Reuse an existing mediapipe_cli_* output folder instead of running MediaPipe",
    )
    parser.add_argument("--no-open", action="store_true", help="Do not open the report")
    parser.add_argument("--progress-window", action="store_true", help="Show a Tk progress window")
    args = parser.parse_args(argv)
    if args.progress_window:
        return run_with_progress_window(args)
    report = analyze_session(
        Path(args.session),
        args.cameras,
        args.participant,
        Path(args.mediapipe_dir) if args.mediapipe_dir else None,
    )
    if not args.no_open:
        webbrowser.open(report.as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
