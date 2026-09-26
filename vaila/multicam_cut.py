"""
Project: vailá Multimodal Toolbox
Script: multicam_cut.py - Cut Cameras

Author: Carlos Pinto Jr
Email: carlos@vulcanum.com.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 26 September 2026
Update Date: 26 September 2026
Version: 0.4.5

Description:
Cuts every video of a multi-camera session (e.g. a folder written by
"Record Cameras") at once, keeping them synchronized. All cameras are shown
side by side in one mosaic window -- the same kind of mosaic used by the
recorder's live preview -- driven by one shared timeline with three handles:
a green Start handle, a red End handle and a blue cursor that shows the
video at that instant.

The timeline is in *time* (seconds), not frames: webcams in one session often
run at different (and variable) frame rates, e.g. a 1280x720 camera limited
by USB bandwidth to ~8 fps next to 640x480 cameras at 30 fps. Each camera
maps timeline time to its own frame through the real packet timestamps read
from the file (ffprobe), so a cut keeps every camera on the same instant.

Cameras recorded by parallel ffmpeg processes never start at exactly the same
instant, so each camera has an *offset* in milliseconds:
camera time = timeline time + offset. Step a camera's offset (one click = one
frame of that camera) until a sharp event (a clap, a flash, a foot strike)
lands on the same instant in every tile, then set Start/End once and cut.

Output:
<session_dir>/processed_multicam_cut_YYYYMMDD_HHMMSS/
    <video_stem>_frame_<a>_to_<b>.mp4   (one per camera, 1-based inclusive)
    multicam_cut.toml                   (times, offsets, per-camera frames)

Usage:
GUI: click "Cut Cameras" in vailá's Video and Image tools, or run
     ``uv run python vaila/multicam_cut.py [session_dir]``.

Timeline (mouse):
    drag green ▲ / red ▲   move Start / End (the mosaic follows the handle)
    drag blue cursor       scrub the video; click anywhere on the bar to jump

Keys (control window or mosaic window):
    Space        play / pause
    a / d        step back / forward one frame (of the fastest camera)
    j / l        step back / forward 10 frames
    i / o        set Start / End at the cursor

Requirements:
- FFmpeg / ffprobe (via vaila.ffmpeg_utils)
- OpenCV
"""

from __future__ import annotations

import contextlib
import functools
import queue
import re
import shlex
import subprocess
import sys
import threading
import time
import tkinter as tk
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import cast

import cv2
import numpy as np

try:
    from .cutvideo import cut_output_filename
    from .ffmpeg_utils import (
        encoder_device_tag,
        encoders_with_cpu_fallback,
        get_ffmpeg_video_encoding_args,
        get_ffprobe_path,
        get_video_encode_ffmpeg_path,
        is_hardware_video_encoder,
    )
    from .multicam_recorder import (
        _build_mosaic,
        _restore_cv2_qt_plugin_env,
        _scrub_cv2_qt_plugin_env,
        load_config_value,
        save_config_value,
    )
    from .syncvid import SyncWorkflowError, discover_video_files
except ImportError:
    from cutvideo import cut_output_filename  # ty: ignore[unresolved-import]
    from ffmpeg_utils import (  # ty: ignore[unresolved-import]
        encoder_device_tag,
        encoders_with_cpu_fallback,
        get_ffmpeg_video_encoding_args,
        get_ffprobe_path,
        get_video_encode_ffmpeg_path,
        is_hardware_video_encoder,
    )
    from multicam_recorder import (  # ty: ignore[unresolved-import]
        _build_mosaic,
        _restore_cv2_qt_plugin_env,
        _scrub_cv2_qt_plugin_env,
        load_config_value,
        save_config_value,
    )
    from syncvid import SyncWorkflowError, discover_video_files  # ty: ignore[unresolved-import]


# ── Session discovery & frame timestamps ─────────────────────────────────────


def natural_sort_key(path: Path) -> list[int | str]:
    """Sort ``cam2`` before ``cam10`` (plain name sort would not)."""
    return [
        int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", path.name)
    ]


def discover_session_videos(session_dir: str | Path) -> list[Path]:
    """Return the session's videos in natural camera order."""
    return sorted(discover_video_files(session_dir), key=natural_sort_key)


def parse_packet_times(ffprobe_csv: str) -> np.ndarray:
    """Parse ``ffprobe -show_entries packet=pts_time -of csv=p=0`` output.

    Returns sorted presentation times relative to the first frame (seconds).
    """
    values = []
    for line in ffprobe_csv.splitlines():
        field = line.strip().split(",")[0]
        with contextlib.suppress(ValueError):
            values.append(float(field))
    if not values:
        return np.array([], dtype=float)
    times = np.sort(np.array(values, dtype=float))
    return times - times[0]


def read_frame_times(path: Path, frame_count: int, fps: float) -> np.ndarray:
    """Real per-frame timestamps via ffprobe; uniform ``i / fps`` if unavailable."""
    try:
        result = subprocess.run(
            [
                get_ffprobe_path(),
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "packet=pts_time",
                "-of",
                "csv=p=0",
                str(path),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
        times = parse_packet_times(result.stdout or "")
    except (OSError, subprocess.TimeoutExpired):
        times = np.array([], dtype=float)
    if len(times) < 2:
        return np.arange(frame_count, dtype=float) / (fps if fps > 0 else 30.0)
    return times


def frame_period(times: np.ndarray) -> float:
    """Typical frame duration (median spacing), used for the last frame and stepping."""
    if len(times) < 2:
        return 1 / 30
    period = float(np.median(np.diff(times)))
    return period if period > 0 else 1 / 30


def camera_duration(times: np.ndarray) -> float:
    """Time span covered by a camera: last frame start + one frame period."""
    return float(times[-1]) + frame_period(times) if len(times) else 0.0


def frame_index_at(times: np.ndarray, t: float) -> int | None:
    """Frame shown at camera time ``t`` (the last frame starting at or before it)."""
    if len(times) == 0 or t < 0 or t >= camera_duration(times):
        return None
    return int(np.searchsorted(times, t + 1e-9, side="right")) - 1


# ── Timeline math (pure) ─────────────────────────────────────────────────────
# Camera i shows camera time ``t + offsets[i]`` at timeline time ``t``, so it
# covers timeline times ``[-offsets[i], durations[i] - offsets[i])``.


def timeline_bounds(durations: Sequence[float], offsets: Sequence[float]) -> tuple[float, float]:
    """Timeline range where *at least one* camera has a frame."""
    lo = min(-off for off in offsets) + 0.0  # + 0.0 turns -0.0 into 0.0
    hi = max(dur - off for dur, off in zip(durations, offsets, strict=True))
    return lo, hi


def common_range(
    durations: Sequence[float], offsets: Sequence[float]
) -> tuple[float, float] | None:
    """Timeline range where *every* camera has a frame, or None if they don't overlap."""
    lo = max(-off for off in offsets) + 0.0
    hi = min(dur - off for dur, off in zip(durations, offsets, strict=True))
    return (lo, hi) if lo < hi else None


def validate_cut(
    start: float | None,
    end: float | None,
    durations: Sequence[float],
    offsets: Sequence[float],
) -> str | None:
    """Return an error message if the cut can't be applied to every camera."""
    if start is None or end is None:
        return "Set both Start and End first."
    if start >= end:
        return "Start must come before End."
    overlap = common_range(durations, offsets)
    if overlap is None:
        return "With the current offsets the cameras don't overlap in time."
    lo, hi = overlap
    eps = 1e-6
    if start < lo - eps or end > hi + eps:
        return (
            f"Cut {start:.3f}-{end:.3f} s falls outside the range every camera "
            f"covers ({lo:.3f}-{hi:.3f} s). Adjust Start/End or the offsets."
        )
    return None


def camera_frame_ranges(
    start: float,
    end: float,
    offsets: Sequence[float],
    times_list: Sequence[np.ndarray],
) -> list[tuple[int, int]]:
    """Per-camera (first, last) frame indices (0-based, inclusive) for a timeline cut."""
    ranges = []
    for off, times in zip(offsets, times_list, strict=True):
        last_t = camera_duration(times) - 1e-9
        first = frame_index_at(times, min(max(start + off, 0.0), last_t))
        last = frame_index_at(times, min(max(end + off, 0.0), last_t))
        if first is None or last is None:
            raise ValueError("Cut falls outside a camera's recording.")
        ranges.append((first, last))
    return ranges


def browse_start_dir(last_session_dir: str, recorder_output_dir: str) -> str | None:
    """Where the session folder chooser opens.

    Sessions are sibling folders written by Record Cameras, so the useful
    starting point is the folder that *contains* the last session (a new
    recording is then one click away); before any cut, fall back to Record
    Cameras' output directory.
    """
    if last_session_dir:
        parent = Path(last_session_dir).parent
        if parent.is_dir():
            return str(parent)
    if recorder_output_dir and Path(recorder_output_dir).is_dir():
        return recorder_output_dir
    return None


def build_output_dir(session_dir: Path, timestamp: str | None = None) -> Path:
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    return session_dir / f"processed_multicam_cut_{ts}"


def build_cut_report(
    videos: Sequence[Path],
    offsets: Sequence[float],
    start: float,
    end: float,
    frame_ranges: Sequence[tuple[int, int]],
    times_list: Sequence[np.ndarray],
) -> dict:
    """TOML-ready record of the cut (frames 1-based, inclusive; times in seconds)."""
    cameras = []
    for video, off, (first, last), times in zip(
        videos, offsets, frame_ranges, times_list, strict=True
    ):
        cameras.append(
            {
                "video": video.name,
                "output": cut_output_filename(video.stem, first, last),
                "offset_s": round(off, 6),
                "start_frame": first + 1,
                "end_frame": last + 1,
                "frame_count": last - first + 1,
                "start_time_s": round(float(times[first]), 6),
                "mean_fps": round(1 / frame_period(times), 3),
            }
        )
    return {
        "multicam_cut": {
            "created": datetime.now().isoformat(timespec="seconds"),
            "timeline_start_s": round(start, 6),
            "timeline_end_s": round(end, 6),
            "duration_s": round(end - start, 6),
        },
        "cameras": cameras,
    }


def time_to_x(t: float, lo: float, hi: float, x0: float, x1: float) -> float:
    return x0 if hi <= lo else x0 + (t - lo) / (hi - lo) * (x1 - x0)


def x_to_time(x: float, lo: float, hi: float, x0: float, x1: float) -> float:
    if x1 <= x0:
        return lo
    frac = min(1.0, max(0.0, (x - x0) / (x1 - x0)))
    return lo + frac * (hi - lo)


# ── Cutting (ffmpeg, seek by the frame's real timestamp) ────────────────────


@functools.cache
def passthrough_timestamp_args(ffmpeg_path: str) -> tuple[str, ...]:
    """Args that keep each frame's source timestamp instead of forcing a constant rate.

    Without them the mp4 muxer re-times frames to the container's *declared*
    frame rate, which webcams often get wrong (e.g. declared 15.9 fps for a
    real ~8 fps stream), making the cut play back too fast. ``-fps_mode``
    exists since FFmpeg 5.1; older builds only know ``-vsync``.
    """
    try:
        result = subprocess.run(
            [ffmpeg_path, "-hide_banner", "-h", "long"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
        help_text = result.stdout or ""
    except (OSError, subprocess.TimeoutExpired):
        help_text = ""
    flag = "-fps_mode" if "-fps_mode" in help_text else "-vsync"
    return (flag, "passthrough")


def build_cut_command(
    ffmpeg_path: str,
    video: Path,
    output: Path,
    seek_time: float,
    n_frames: int,
    encoding_args: Sequence[str],
    timestamp_args: Sequence[str] = ("-fps_mode", "passthrough"),
) -> list[str]:
    """Accurate input seek to ``seek_time`` then keep exactly ``n_frames`` frames.

    No ``-r`` and passthrough timestamps: output keeps each source frame's own
    timestamp, so variable frame-rate webcam footage stays in real time.
    """
    return [
        ffmpeg_path,
        "-y",
        "-nostdin",
        "-loglevel",
        "error",
        "-ss",
        f"{seek_time:.6f}",
        "-i",
        str(video),
        "-map",
        "0:v:0",
        "-frames:v",
        str(n_frames),
        *timestamp_args,
        *encoding_args,
        "-an",
        str(output),
    ]


def seek_time_for(times: np.ndarray, first: int) -> float:
    """Seek slightly before the first frame's timestamp so float rounding can't drop it."""
    if first <= 0:
        return 0.0
    return max(0.0, float(times[first]) - 0.25 * frame_period(times))


def cut_camera(
    video: Path,
    output: Path,
    times: np.ndarray,
    first: int,
    last: int,
    keep_going: Callable[[], bool],
) -> bool:
    """Cut one camera, preferring hardware H.264 with CPU libx264 fallback."""
    for encoder in encoders_with_cpu_fallback():
        ffmpeg_path = get_video_encode_ffmpeg_path(encoder)
        cmd = build_cut_command(
            ffmpeg_path,
            video,
            output,
            seek_time_for(times, first),
            last - first + 1,
            get_ffmpeg_video_encoding_args(encoder),
            passthrough_timestamp_args(ffmpeg_path),
        )
        print(
            f">> vaila/multicam_cut: [{encoder_device_tag(encoder)}] Equivalent CLI:\n"
            f">>   {' '.join(shlex.quote(part) for part in cmd)}",
            flush=True,
        )
        process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        while process.poll() is None:
            if not keep_going():
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                return False
            time.sleep(0.05)
        if process.returncode == 0:
            return True
        err = process.stderr.read().decode(errors="replace") if process.stderr else ""
        print(f">> vaila/multicam_cut: {encoder} failed: {err.strip()[-500:]}", flush=True)
        if not is_hardware_video_encoder(encoder):
            return False
    return False


# ── Video playback (one capture per camera) ─────────────────────────────────


@dataclass
class SessionVideo:
    path: Path
    times: np.ndarray
    width: int
    height: int

    @property
    def frame_count(self) -> int:
        return len(self.times)

    @property
    def duration(self) -> float:
        return camera_duration(self.times)

    @property
    def fps(self) -> float:
        return 1 / frame_period(self.times)


def probe_session_video(path: Path) -> SessionVideo:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise SyncWorkflowError(f"OpenCV could not open video: {path}")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    if frame_count <= 0:
        raise SyncWorkflowError(f"Video reports no frames: {path}")
    return SessionVideo(path, read_frame_times(path, frame_count, fps), width, height)


class _VideoReader:
    """Random-access frame reader that avoids seeking during linear playback."""

    def __init__(self, video: SessionVideo) -> None:
        self.video = video
        self._cap = cv2.VideoCapture(str(video.path))
        self._next_index = 0
        self._last: tuple[int, np.ndarray] | None = None

    def frame(self, index: int | None) -> np.ndarray | None:
        if index is None or not 0 <= index < self.video.frame_count:
            return None
        if self._last is not None and self._last[0] == index:
            return self._last[1]
        if index != self._next_index:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, img = self._cap.read()
        if not ok:
            self._next_index = -1  # force a seek next time
            return None
        self._next_index = index + 1
        self._last = (index, img)
        return img

    def close(self) -> None:
        self._cap.release()


_MOSAIC_WINDOW_TITLE = "vailá - Cut Cameras"
_TILE_WIDTH = 480


def tile_size_for(videos: Sequence[SessionVideo]) -> tuple[int, int]:
    first = videos[0]
    aspect = first.height / first.width if first.width else 9 / 16
    return _TILE_WIDTH, max(1, round(_TILE_WIDTH * aspect))


def render_tile(
    img: np.ndarray | None,
    tile_size: tuple[int, int],
    label: str,
    frame_text: str,
    in_cut: bool,
) -> np.ndarray:
    width, height = tile_size
    if img is None:
        tile = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(
            tile,
            "no frame",
            (width // 2 - 50, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    else:
        tile = cv2.resize(img, tile_size)
    color = (0, 255, 0) if in_cut else (0, 200, 255)  # BGR: green inside Start/End
    for text, y in ((label, 20), (frame_text, 42)):
        cv2.putText(tile, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(tile, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    cv2.rectangle(tile, (0, 0), (width - 1, height - 1), color, 2 if in_cut else 1)
    return tile


# ── Range timeline widget (Start / cursor / End) ────────────────────────────


class RangeTimeline(tk.Canvas):
    """Timeline bar with a green Start handle, a red End handle and a blue cursor.

    ``on_change(kind, t)`` is called while dragging, with ``kind`` one of
    ``"start"``, ``"end"`` or ``"cursor"``; the owner clamps the value and
    calls :meth:`set_state` back, so the widget holds no rules of its own.
    """

    PAD = 16
    TRACK_Y = 30
    HIT_PX = 12

    def __init__(self, parent: tk.Widget, on_change: Callable[[str, float], None]) -> None:
        super().__init__(parent, height=62, highlightthickness=0, background="#fafafa")
        self.on_change = on_change
        self.lo, self.hi = 0.0, 1.0
        self.common: tuple[float, float] | None = None
        self.start: float | None = None
        self.end: float | None = None
        self.cursor = 0.0
        self._drag: str | None = None
        self.bind("<Configure>", lambda _e: self.redraw())
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_motion)
        self.bind("<ButtonRelease-1>", lambda _e: setattr(self, "_drag", None))

    def set_state(
        self,
        lo: float,
        hi: float,
        common: tuple[float, float] | None,
        start: float | None,
        end: float | None,
        cursor: float,
    ) -> None:
        self.lo, self.hi = lo, hi
        self.common, self.start, self.end, self.cursor = common, start, end, cursor
        self.redraw()

    def _x(self, t: float) -> float:
        return time_to_x(t, self.lo, self.hi, self.PAD, self.winfo_width() - self.PAD)

    def _t(self, x: float) -> float:
        return x_to_time(x, self.lo, self.hi, self.PAD, self.winfo_width() - self.PAD)

    def redraw(self) -> None:
        self.delete("all")
        width = self.winfo_width()
        y = self.TRACK_Y
        x0, x1 = self.PAD, width - self.PAD
        self.create_rectangle(x0, y - 4, x1, y + 4, fill="#d0d0d0", outline="")
        if self.common is not None:
            self.create_rectangle(
                self._x(self.common[0]), y - 4, self._x(self.common[1]), y + 4,
                fill="#9db4c8", outline="",
            )  # fmt: skip
        if self.start is not None and self.end is not None:
            self.create_rectangle(
                self._x(self.start), y - 6, self._x(self.end), y + 6,
                fill="#7bc47f", outline="",
            )  # fmt: skip
        for value, color in ((self.start, "#1e8e3e"), (self.end, "#d93025")):
            if value is not None:
                x = self._x(value)
                self.create_line(x, y - 8, x, y + 8, fill=color, width=2)
                self.create_polygon(x, y + 8, x - 8, y + 22, x + 8, y + 22, fill=color)
        xc = self._x(self.cursor)
        self.create_line(xc, 6, xc, y + 8, fill="#1a73e8", width=2)
        self.create_oval(xc - 6, 2, xc + 6, 14, fill="#1a73e8", outline="")
        # second ticks along the bottom
        span = self.hi - self.lo
        if span > 0:
            step = next(s for s in (0.5, 1, 2, 5, 10, 30, 60, 300, 600, 1e9) if span / s <= 12)
            t = step * np.ceil(self.lo / step)
            while t <= self.hi:
                x = self._x(t)
                self.create_line(x, y + 24, x, y + 28, fill="#888")
                self.create_text(x, y + 30, text=f"{t:g}s", anchor="n", fill="#666", font=("", 7))
                t += step

    def _on_press(self, event: tk.Event) -> None:
        candidates: list[tuple[str, float | None]] = [("cursor", self.cursor)]
        if event.y > self.TRACK_Y:  # lower half: handles take priority over the cursor
            candidates = [("start", self.start), ("end", self.end), *candidates]
        else:
            candidates += [("start", self.start), ("end", self.end)]
        best, best_dist = None, float(self.HIT_PX)
        for kind, value in candidates:
            if value is None:
                continue
            dist = abs(self._x(value) - event.x)
            if dist < best_dist:
                best, best_dist = kind, dist
        self._drag = best or "cursor"
        self.on_change(self._drag, self._t(event.x))

    def _on_motion(self, event: tk.Event) -> None:
        if self._drag:
            self.on_change(self._drag, self._t(event.x))


# ── GUI ──────────────────────────────────────────────────────────────────────


class MulticamCutApp:
    """Tk control window + cv2 mosaic of every camera in a session."""

    def __init__(self, window: tk.Toplevel, session_dir: str | None = None) -> None:
        self.window = window
        self.window.title("vailá - Cut Cameras")
        self.session_dir_var = tk.StringVar(
            value=session_dir or load_config_value("multicam_cut", "last_session_dir")
        )
        self.status_var = tk.StringVar(value="Choose a session folder.")
        self.info_var = tk.StringVar(value="")

        self.videos: list[SessionVideo] = []
        self.readers: list[_VideoReader] = []
        self.offset_vars: list[tk.StringVar] = []
        self.current = 0.0
        self.start: float | None = None
        self.end: float | None = None
        self.playing = False
        self._play_clock = 0.0
        self.cutting = False
        self._cancel_event = threading.Event()
        # The cut worker thread never touches Tk: it only posts events here,
        # which _tick() drains on the main thread (Tk isn't thread-safe).
        self._worker_events: queue.Queue[tuple] = queue.Queue()
        self._window_open = False
        self._dirty = True
        self._shown_frames: list[int | None] = []
        self._tile_size = (_TILE_WIDTH, 270)

        self._build_widgets()
        self._bind_keys()
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
        if session_dir:
            self.load_session(session_dir)
        self._tick()

    # ── widgets ──

    def _build_widgets(self) -> None:
        outer = ttk.Frame(self.window, padding=10)
        outer.pack(fill="both", expand=True)

        session_row = ttk.Frame(outer)
        session_row.pack(fill="x")
        ttk.Label(session_row, text="Session folder:").pack(side="left")
        self.session_entry = ttk.Entry(session_row, textvariable=self.session_dir_var, width=48)
        self.session_entry.pack(side="left", padx=6)
        # Enter reloads whatever is typed/pre-filled (the last session by default).
        self.session_entry.bind(
            "<Return>", lambda _e: self.load_session(self.session_dir_var.get().strip())
        )
        self.browse_btn = ttk.Button(session_row, text="Browse...", command=self._choose_session)
        self.browse_btn.pack(side="left")

        ttk.Label(
            outer,
            text=(
                "Offset (ms) shifts a camera in time; each arrow click = 1 frame of that "
                "camera. Step it until a sharp event (clap, flash) lines up in every tile."
            ),
            foreground="gray",
            wraplength=620,
        ).pack(anchor="w", pady=(8, 2))
        self.camera_frame = ttk.Frame(outer)
        self.camera_frame.pack(fill="x", pady=(0, 8))

        self.timeline = RangeTimeline(outer, self._on_timeline_change)
        self.timeline.configure(width=640)
        self.timeline.pack(fill="x")
        ttk.Label(
            outer,
            text="Drag green ▲ = Start · red ▲ = End · blue cursor = video position",
            foreground="gray",
        ).pack(anchor="w")
        ttk.Label(outer, textvariable=self.info_var, font=("TkFixedFont", 9)).pack(
            anchor="w", pady=(2, 0)
        )

        nav = ttk.Frame(outer)
        nav.pack(pady=6)
        buttons = [
            ("|<", 4, lambda: self.seek(self._bounds()[0])),
            ("-10", 4, lambda: self.step(-10)),
            ("-1", 4, lambda: self.step(-1)),
            ("Play/Pause", 10, self.toggle_play),
            ("+1", 4, lambda: self.step(1)),
            ("+10", 4, lambda: self.step(10)),
            (">|", 4, lambda: self.seek(self._bounds()[1])),
        ]
        for text, width, command in buttons:
            ttk.Button(nav, text=text, width=width, command=command).pack(side="left", padx=2)

        marks = ttk.Frame(outer)
        marks.pack(pady=4)
        self.start_btn = ttk.Button(marks, text="Start at cursor (i)", command=self.mark_start)
        self.end_btn = ttk.Button(marks, text="End at cursor (o)", command=self.mark_end)
        self.start_btn.pack(side="left", padx=2)
        self.end_btn.pack(side="left", padx=2)

        self.cut_btn = ttk.Button(outer, text="Cut all cameras", command=self._on_cut)
        self.cut_btn.pack(pady=(8, 2))
        ttk.Label(outer, textvariable=self.status_var, wraplength=620).pack(anchor="w")

    def _bind_keys(self) -> None:
        for key, action in self._key_actions().items():
            self.window.bind(f"<KeyPress-{key}>", lambda _e, a=action: self._key_guard(a))

    def _key_actions(self) -> dict[str, Callable[[], None]]:
        return {
            "space": self.toggle_play,
            "a": lambda: self.step(-1),
            "d": lambda: self.step(1),
            "j": lambda: self.step(-10),
            "l": lambda: self.step(10),
            "i": self.mark_start,
            "o": self.mark_end,
        }

    def _key_guard(self, action: Callable[[], None]) -> None:
        # Don't hijack keystrokes typed into an Entry/Spinbox.
        if isinstance(self.window.focus_get(), (tk.Entry, ttk.Entry, tk.Spinbox, ttk.Spinbox)):
            return
        action()

    # ── session loading ──

    def _choose_session(self) -> None:
        selected = filedialog.askdirectory(
            parent=self.window,
            title="Select multi-camera session folder",
            initialdir=browse_start_dir(
                load_config_value("multicam_cut", "last_session_dir"),
                load_config_value("multicam_recorder", "last_output_dir"),
            ),
        )
        if selected:
            self.load_session(selected)

    def load_session(self, session_dir: str) -> None:
        try:
            paths = discover_session_videos(session_dir)
            if len(paths) < 2:
                raise SyncWorkflowError(
                    f"Found {len(paths)} video(s) in {session_dir}; need at least 2 cameras."
                )
            videos = [probe_session_video(p) for p in paths]
        except SyncWorkflowError as exc:
            messagebox.showerror("vailá", str(exc), parent=self.window)
            return

        self._close_readers()
        self.session_dir_var.set(session_dir)
        save_config_value("multicam_cut", "last_session_dir", session_dir)
        self.videos = videos
        self.readers = [_VideoReader(v) for v in videos]
        self._tile_size = tile_size_for(videos)
        self.playing = False
        self._build_camera_table()
        overlap = common_range(self._durations(), self.offsets())
        self.start, self.end = overlap if overlap else (None, None)
        self.seek(self._bounds()[0])

        msg = f"Loaded {len(videos)} cameras: " + ", ".join(
            f"{v.path.stem} {v.fps:.1f} fps/{v.duration:.2f} s" for v in videos
        )
        self.status_var.set(msg)
        print(f">> vaila/multicam_cut: {msg}", flush=True)

    def _build_camera_table(self) -> None:
        for widget in self.camera_frame.winfo_children():
            widget.destroy()
        self.offset_vars = []
        for col, text in enumerate(("Camera", "Frames", "FPS", "Size", "Offset (ms)")):
            ttk.Label(self.camera_frame, text=text).grid(row=0, column=col, padx=4, sticky="w")
        for row, video in enumerate(self.videos, start=1):
            var = tk.StringVar(value="0")
            self.offset_vars.append(var)
            cells = (video.path.name, video.frame_count, f"{video.fps:.2f}")
            for col, text in enumerate((*cells, f"{video.width}x{video.height}")):
                ttk.Label(self.camera_frame, text=str(text)).grid(
                    row=row, column=col, padx=4, sticky="w"
                )
            ttk.Spinbox(
                self.camera_frame,
                from_=-600000,
                to=600000,
                increment=round(1000 / video.fps, 1),
                width=9,
                textvariable=var,
            ).grid(row=row, column=4, padx=4)
            # Trace only after the initial value is set, then refresh once below.
            var.trace_add("write", lambda *_: self._on_offsets_changed())

    # ── state helpers ──

    def offsets(self) -> list[float]:
        """Per-camera offsets in seconds."""
        values = []
        for var in self.offset_vars:
            try:
                values.append(float(var.get()) / 1000.0)
            except ValueError:
                values.append(0.0)
        return values

    def _durations(self) -> list[float]:
        return [v.duration for v in self.videos]

    def _bounds(self) -> tuple[float, float]:
        if not self.videos:
            return 0.0, 1.0
        return timeline_bounds(self._durations(), self.offsets())

    def _common(self) -> tuple[float, float] | None:
        if not self.videos:
            return None
        return common_range(self._durations(), self.offsets())

    def _step_seconds(self) -> float:
        return min((frame_period(v.times) for v in self.videos), default=1 / 30)

    def _on_offsets_changed(self) -> None:
        common = self._common()
        if common is not None and self.start is not None and self.end is not None:
            self.start = min(max(self.start, common[0]), common[1])
            self.end = min(max(self.end, common[0]), common[1])
        self.seek(self.current)

    def _refresh_timeline(self) -> None:
        lo, hi = self._bounds()
        common = self._common()
        self.timeline.set_state(lo, hi, common, self.start, self.end, self.current)
        s = "-" if self.start is None else f"{self.start:8.3f}"
        e = "-" if self.end is None else f"{self.end:8.3f}"
        dur = (
            f"{self.end - self.start:.3f}"
            if self.start is not None and self.end is not None
            else "-"
        )
        self.info_var.set(
            f"Start {s} s | Cursor {self.current:8.3f} s | End {e} s | Duration {dur} s"
        )

    # ── navigation ──

    def seek(self, t: float) -> None:
        lo, hi = self._bounds()
        self.current = max(lo, min(hi - 1e-6, t))
        self._dirty = True
        self._refresh_timeline()

    def step(self, frames: int) -> None:
        self.playing = False
        self.seek(self.current + frames * self._step_seconds())

    def _on_timeline_change(self, kind: str, t: float) -> None:
        if not self.videos or self.cutting:
            return
        self.playing = False
        common = self._common()
        if kind == "cursor" or common is None:
            self.seek(t)
            return
        lo, hi = common
        if kind == "start":
            limit = self.end if self.end is not None else hi
            self.start = min(max(t, lo), limit)
            self.seek(self.start)  # show the frame under the handle
        else:
            limit = self.start if self.start is not None else lo
            self.end = max(min(t, hi), limit)
            self.seek(self.end)

    def toggle_play(self) -> None:
        if self.videos:
            self.playing = not self.playing
            self._play_clock = time.monotonic()

    def mark_start(self) -> None:
        self._on_timeline_change("start", self.current)

    def mark_end(self) -> None:
        self._on_timeline_change("end", self.current)

    # ── mosaic rendering (main thread, driven by Tk's after()) ──

    def _tick(self) -> None:
        self._drain_worker_events()
        if self.videos:
            if self.playing:
                now = time.monotonic()
                elapsed, self._play_clock = now - self._play_clock, now
                _lo, hi = self._bounds()
                if self.current + elapsed >= hi - 1e-6:
                    self.playing = False
                self.seek(self.current + elapsed)
            if self._dirty:
                self._render()
                self._dirty = False
            self._poll_mosaic_keys()
        self.window.after(15, self._tick)

    def _render(self) -> None:
        offsets = self.offsets()
        frames = [
            frame_index_at(v.times, self.current + off)
            for v, off in zip(self.videos, offsets, strict=True)
        ]
        if frames == self._shown_frames and self._window_open:
            return  # same frames as on screen (e.g. slow camera mid-period)
        self._shown_frames = frames
        in_cut = (
            self.start is not None
            and self.end is not None
            and self.start <= self.current <= self.end
        )
        tiles = []
        for reader, index in zip(self.readers, frames, strict=True):
            video = reader.video
            frame_text = (
                f"frame {index + 1}/{video.frame_count}  {video.fps:.1f} fps"
                if index is not None
                else "outside recording"
            )
            tiles.append(
                render_tile(
                    reader.frame(index), self._tile_size, video.path.stem, frame_text, in_cut
                )
            )
        if not self._window_open:
            _restore_cv2_qt_plugin_env()
            cv2.namedWindow(_MOSAIC_WINDOW_TITLE, cv2.WINDOW_NORMAL)
            self._window_open = True
        cv2.imshow(_MOSAIC_WINDOW_TITLE, _build_mosaic(tiles, self._tile_size))

    def _poll_mosaic_keys(self) -> None:
        if not self._window_open:
            return
        key = cv2.waitKey(1) & 0xFF
        if key == 255:
            return
        action = self._key_actions().get("space" if key == ord(" ") else chr(key))
        if action is not None:
            action()

    # ── cutting ──

    def _on_cut(self) -> None:
        if self.cutting:
            self._cancel_event.set()
            self.status_var.set("Cancelling...")
            return
        if not self.videos:
            messagebox.showerror("vailá", "Load a session folder first.", parent=self.window)
            return
        offsets = self.offsets()
        error = validate_cut(self.start, self.end, self._durations(), offsets)
        if error:
            messagebox.showerror("vailá", error, parent=self.window)
            return
        start, end = cast(float, self.start), cast(float, self.end)
        times_list = [v.times for v in self.videos]
        try:
            ranges = camera_frame_ranges(start, end, offsets, times_list)
        except ValueError as exc:
            messagebox.showerror("vailá", str(exc), parent=self.window)
            return

        self.playing = False
        self.cutting = True
        self._cancel_event.clear()
        self._set_inputs_enabled(False)
        self.cut_btn.config(text="Cancel")

        out_dir = build_output_dir(Path(self.session_dir_var.get()))
        videos = [v.path for v in self.videos]
        threading.Thread(
            target=self._cut_worker,
            args=(videos, times_list, ranges, offsets, start, end, out_dir),
            daemon=True,
        ).start()

    def _cut_worker(
        self,
        videos: list[Path],
        times_list: list[np.ndarray],
        ranges: list[tuple[int, int]],
        offsets: list[float],
        start: float,
        end: float,
        out_dir: Path,
    ) -> None:
        written: list[Path] = []
        ok = True
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            for index, (video, times, (first, last)) in enumerate(
                zip(videos, times_list, ranges, strict=True), start=1
            ):
                self._post_status(
                    f"Cutting {index}/{len(videos)}: {video.name} "
                    f"(frames {first + 1}-{last + 1}) ..."
                )
                output = out_dir / cut_output_filename(video.stem, first, last)
                ok = cut_camera(
                    video, output, times, first, last, lambda: not self._cancel_event.is_set()
                )
                if not ok:
                    break
                written.append(output)
            if ok:
                import toml

                report = build_cut_report(videos, offsets, start, end, ranges, times_list)
                with open(out_dir / "multicam_cut.toml", "w", encoding="utf-8") as fh:
                    toml.dump(report, fh)
        except Exception as exc:  # noqa: BLE001
            ok = False
            self._post_status(f"Error: {exc}")
        self._worker_events.put(("done", ok, written, out_dir))

    def _post_status(self, text: str) -> None:
        print(f">> vaila/multicam_cut: {text}", flush=True)
        self._worker_events.put(("status", text))

    def _drain_worker_events(self) -> None:
        while True:
            try:
                event = self._worker_events.get_nowait()
            except queue.Empty:
                return
            if event[0] == "status":
                self.status_var.set(event[1])
            elif event[0] == "done":
                self._on_cut_done(*event[1:])

    def _on_cut_done(self, ok: bool, written: list[Path], out_dir: Path) -> None:
        self.cutting = False
        self._set_inputs_enabled(True)
        self.cut_btn.config(text="Cut all cameras")
        if ok:
            self.status_var.set(f"Done: {len(written)} videos in {out_dir}")
            summary = "\n".join(p.name for p in written)
            messagebox.showinfo(
                "vailá",
                f"Cut {len(written)} cameras into:\n{out_dir}\n\n{summary}",
                parent=self.window,
            )
        elif self._cancel_event.is_set():
            self.status_var.set(f"Cancelled. Partial output (if any) in {out_dir}")
        else:
            messagebox.showerror(
                "vailá", f"Cutting failed. Partial output (if any) in {out_dir}", parent=self.window
            )

    def _set_inputs_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for widget in (self.session_entry, self.browse_btn, self.start_btn, self.end_btn):
            widget.config(state=state)
        for child in self.camera_frame.winfo_children():
            if isinstance(child, ttk.Spinbox):
                child.config(state=state)

    # ── teardown ──

    def _close_readers(self) -> None:
        for reader in self.readers:
            reader.close()
        self.readers = []
        self._shown_frames = []

    def on_close(self) -> None:
        if self.cutting:
            if not messagebox.askyesno(
                "vailá", "Cutting is in progress. Cancel it and close?", parent=self.window
            ):
                return
            self._cancel_event.set()
        self._close_readers()
        if self._window_open:
            with contextlib.suppress(cv2.error):
                cv2.destroyWindow(_MOSAIC_WINDOW_TITLE)
            self._window_open = False
            _scrub_cv2_qt_plugin_env()
        self.window.destroy()


def run_multicam_cut(
    parent: tk.Tk | tk.Toplevel | None = None, session_dir: str | None = None
) -> None:
    """Open the synchronized multi-camera cut tool."""
    print(f"Running script: {Path(__file__).name}")
    print(f"Script directory: {Path(__file__).parent}")
    print("Starting multicam_cut...")

    created_root = False
    default_root = cast("tk.Tk | tk.Toplevel | None", getattr(tk, "_default_root", None))
    root = parent or default_root
    if root is None:
        root = tk.Tk()
        root.withdraw()
        created_root = True

    window = tk.Toplevel(root)
    MulticamCutApp(window, session_dir=session_dir)

    if created_root:
        window.bind("<Destroy>", lambda e: root.quit() if e.widget is window else None)
        root.mainloop()


if __name__ == "__main__":
    run_multicam_cut(session_dir=sys.argv[1] if len(sys.argv) > 1 else None)
