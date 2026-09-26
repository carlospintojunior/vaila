"""Tests for multicam_cut: time-based timeline math, naming and an end-to-end cut.

The end-to-end test writes two synthetic videos at *different* frame rates
(like a USB-bandwidth-limited 1280x720 webcam next to a 30 fps one) whose
frames encode their own index as brightness, cuts them on a shared time range
with an offset, and checks every output starts on the expected source frame.

Update Date: 26 September 2026
Version: 0.4.5
"""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from vaila import multicam_cut as mc

# ── frame timestamps ─────────────────────────────────────────────────────────


def test_parse_packet_times_sorts_and_rebases():
    csv = "0.200000\n0.100000,\n0.300000\nN/A\n"
    np.testing.assert_allclose(mc.parse_packet_times(csv), [0.0, 0.1, 0.2])


def test_frame_index_at_uses_real_timestamps():
    times = np.array([0.0, 0.124, 0.252, 0.376])  # ~8 fps, slightly irregular
    assert mc.frame_index_at(times, 0.0) == 0
    assert mc.frame_index_at(times, 0.123) == 0
    assert mc.frame_index_at(times, 0.124) == 1
    assert mc.frame_index_at(times, 0.40) == 3
    assert mc.frame_index_at(times, -0.01) is None
    assert mc.frame_index_at(times, 0.376 + 0.2) is None  # past last frame + period


# ── timeline math (seconds) ──────────────────────────────────────────────────


def test_bounds_and_common_range_without_offsets():
    durations, offsets = [6.26, 6.92, 6.90], [0.0, 0.0, 0.0]
    assert mc.timeline_bounds(durations, offsets) == (0.0, 6.92)
    assert mc.common_range(durations, offsets) == (0.0, 6.26)


def test_offsets_shift_camera_coverage():
    # cam2 started 0.5 s earlier -> its time t+0.5 matches cam1 time t
    durations, offsets = [10.0, 10.0], [0.0, 0.5]
    assert mc.common_range(durations, offsets) == (0.0, 9.5)
    assert mc.timeline_bounds(durations, offsets) == (-0.5, 10.0)


def test_common_range_none_when_no_overlap():
    assert mc.common_range([1.0, 1.0], [0.0, 5.0]) is None


@pytest.mark.parametrize(
    ("start", "end", "fragment"),
    [
        (None, 5.0, "Set both"),
        (5.0, 5.0, "before End"),
        (0.0, 9.9, "outside"),
    ],
)
def test_validate_cut_errors(start, end, fragment):
    error = mc.validate_cut(start, end, [10.0, 10.0], [0.0, 0.5])
    assert error is not None and fragment in error


def test_validate_cut_ok():
    assert mc.validate_cut(0.0, 9.5, [10.0, 10.0], [0.0, 0.5]) is None


def test_camera_frame_ranges_mixed_fps():
    slow = np.arange(50) / 8.0  # 8 fps
    fast = np.arange(200) / 30.0  # 30 fps
    ranges = mc.camera_frame_ranges(1.0, 2.0, [0.0, 0.5], [slow, fast])
    assert ranges[0] == (8, 16)  # 1.0 s .. 2.0 s at 8 fps
    assert ranges[1] == (45, 75)  # 1.5 s .. 2.5 s at 30 fps


def test_time_x_roundtrip():
    x = mc.time_to_x(2.5, 0.0, 10.0, 16, 416)
    assert x == pytest.approx(116)
    assert mc.x_to_time(x, 0.0, 10.0, 16, 416) == pytest.approx(2.5)
    assert mc.x_to_time(-50, 0.0, 10.0, 16, 416) == 0.0  # clamped


# ── naming / discovery / report ──────────────────────────────────────────────


def test_discover_session_videos_natural_order(tmp_path: Path):
    for name in ("cam10_t.mp4", "cam2_t.mp4", "cam1_t.mp4", "notes.txt"):
        (tmp_path / name).write_bytes(b"x")
    names = [p.name for p in mc.discover_session_videos(tmp_path)]
    assert names == ["cam1_t.mp4", "cam2_t.mp4", "cam10_t.mp4"]


def test_build_output_dir_is_timestamped(tmp_path: Path):
    out = mc.build_output_dir(tmp_path, "20260926_120000")
    assert out == tmp_path / "processed_multicam_cut_20260926_120000"


def test_build_cut_report_is_one_based():
    times = [np.arange(50) / 8.0, np.arange(200) / 30.0]
    ranges = [(8, 16), (45, 75)]
    report = mc.build_cut_report(
        [Path("cam1_t.mp4"), Path("cam2_t.mp4")], [0.0, 0.5], 1.0, 2.0, ranges, times
    )
    assert report["multicam_cut"]["duration_s"] == 1.0
    cam2 = report["cameras"][1]
    assert (cam2["start_frame"], cam2["end_frame"], cam2["frame_count"]) == (46, 76, 31)
    assert cam2["output"] == "cam2_t_frame_46_to_76.mp4"
    assert cam2["mean_fps"] == 30.0


def test_build_cut_command_seeks_and_counts_frames():
    cmd = mc.build_cut_command(
        "ffmpeg", Path("in.mp4"), Path("out.mp4"), 1.5, 31, ["-c:v", "libx264"]
    )
    assert cmd[cmd.index("-ss") + 1] == "1.500000"
    assert cmd.index("-ss") < cmd.index("-i")  # input (accurate) seek
    assert cmd[cmd.index("-frames:v") + 1] == "31"
    assert "-r" not in cmd  # keep source timestamps (VFR-safe)


# ── end-to-end synchronized cut ──────────────────────────────────────────────


def _write_indexed_video(path: Path, n_frames: int, fps: float) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (64, 48))
    for i in range(n_frames):
        writer.write(np.full((48, 64, 3), i * 4, dtype=np.uint8))
    writer.release()


def _frames_brightness_index(path: Path) -> list[int]:
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out.append(round(float(frame.mean()) / 4))
    cap.release()
    return out


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_cut_keeps_mixed_fps_cameras_in_sync(tmp_path: Path):
    videos = [tmp_path / "cam1_t.avi", tmp_path / "cam2_t.avi"]
    _write_indexed_video(videos[0], 40, 8.0)
    _write_indexed_video(videos[1], 60, 12.0)
    session = [mc.probe_session_video(v) for v in videos]
    times_list = [s.times for s in session]
    offsets, start, end = [0.0, 0.25], 1.0, 2.0

    ranges = mc.camera_frame_ranges(start, end, offsets, times_list)
    assert ranges == [(8, 16), (15, 27)]
    for video, times, (first, last) in zip(videos, times_list, ranges, strict=True):
        out = tmp_path / mc.cut_output_filename(video.stem, first, last)
        assert mc.cut_camera(video, out, times, first, last, lambda: True)
        indices = _frames_brightness_index(out)
        assert len(indices) == last - first + 1
        assert abs(indices[0] - first) <= 1
        assert abs(indices[-1] - last) <= 1


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_cut_keeps_real_time_when_declared_fps_is_wrong(tmp_path: Path):
    """The cut keeps real frame timing (duration = frames / real fps)."""
    import subprocess

    src = tmp_path / "raw.avi"
    _write_indexed_video(src, 40, 8.0)
    video = tmp_path / "cam1_t.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-c", "copy", str(video)],
        check=True,
    )
    times = mc.probe_session_video(video).times
    out = tmp_path / "cut.mp4"
    assert mc.cut_camera(video, out, times, 8, 23, lambda: True)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(out)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert float(probe.stdout) == pytest.approx(16 / 8.0, abs=0.15)


# ── remembered folders ───────────────────────────────────────────────────────


def test_browse_start_dir_prefers_parent_of_last_session(tmp_path: Path):
    session = tmp_path / "recordings" / "trial01_20260926_165533"
    session.mkdir(parents=True)
    assert mc.browse_start_dir(str(session), "") == str(session.parent)


def test_browse_start_dir_falls_back_to_recorder_output(tmp_path: Path):
    assert mc.browse_start_dir("", str(tmp_path)) == str(tmp_path)
    assert mc.browse_start_dir(str(tmp_path / "gone" / "x"), str(tmp_path)) == str(tmp_path)
    assert mc.browse_start_dir("", "") is None


def test_config_value_roundtrip_keeps_other_sections(tmp_path: Path, monkeypatch):
    from vaila import multicam_recorder

    monkeypatch.setattr(multicam_recorder, "_VAILA_CONFIG_PATH", tmp_path / "vaila_config.toml")
    multicam_recorder.save_last_output_dir("/data/rec")
    multicam_recorder.save_config_value("multicam_cut", "last_session_dir", "/data/rec/trial01")
    assert multicam_recorder.load_last_output_dir() == "/data/rec"
    assert multicam_recorder.load_config_value("multicam_cut", "last_session_dir") == (
        "/data/rec/trial01"
    )
    assert multicam_recorder.load_config_value("multicam_cut", "missing") == ""
