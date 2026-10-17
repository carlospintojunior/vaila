"""Tests for vaila/gait_auto.run_mediapipe (per-video processes, retry on a crash)."""

import io
from pathlib import Path

import pytest

from vaila import gait_auto


class _FakeProc:
    def __init__(self, code: int, lines: list[str]):
        self.stdout = io.StringIO("".join(f"{line}\n" for line in lines))
        self._code = code

    def wait(self) -> int:
        return self._code


def _fake_popen(outcomes: list[int], calls: list[str]):
    """Popen stand-in: each call pops an exit code; 0 writes the batch's result layout."""

    def popen(cmd, **kwargs):
        input_dir = Path(cmd[cmd.index("-i") + 1])
        out_dir = Path(cmd[cmd.index("-o") + 1])
        (video,) = list(input_dir.iterdir())
        calls.append(video.name)
        code = outcomes.pop(0)
        stem_dir = out_dir / "mediapipe_cli_20261017_120000" / video.stem
        stem_dir.mkdir(parents=True)
        if code == 0:
            (stem_dir / f"{video.stem}_mp_pixel.csv").write_text("frame_index\n0\n")
        return _FakeProc(code, [f"Processing video: {video}", "Analysis loop"])

    return popen


def _videos(tmp_path: Path) -> list[Path]:
    session = tmp_path / "session"
    session.mkdir()
    videos = [session / "cam2_trial.mp4", session / "cam3_trial.mp4"]
    for v in videos:
        v.write_bytes(b"x")
    return videos


def test_run_mediapipe_retries_a_crashed_video_and_merges_results(tmp_path, monkeypatch):
    calls: list[str] = []
    # cam2 ok; cam3 crashes once (Windows access violation), then succeeds
    monkeypatch.setattr(gait_auto.subprocess, "Popen", _fake_popen([0, 3221225477, 0], calls))
    logs: list[str] = []
    run_dir = gait_auto.run_mediapipe(_videos(tmp_path), tmp_path / "mp", logs.append)

    assert calls == ["cam2_trial.mp4", "cam3_trial.mp4", "cam3_trial.mp4"]
    assert run_dir.name.startswith("mediapipe_cli_")
    for stem in ("cam2_trial", "cam3_trial"):
        assert (run_dir / stem / f"{stem}_mp_pixel.csv").is_file()
    assert any("attempt 1" in line for line in logs)


def test_run_mediapipe_gives_up_after_the_last_attempt(tmp_path, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(gait_auto.subprocess, "Popen", _fake_popen([0, 1, 1], calls))
    with pytest.raises(RuntimeError, match=r"cam3_trial\.mp4 after 2 attempts"):
        gait_auto.run_mediapipe(_videos(tmp_path), tmp_path / "mp", lambda _: None, attempts=2)
