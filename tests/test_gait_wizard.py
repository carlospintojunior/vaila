"""Tests for vaila/gait_wizard.py (non-GUI helpers)."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vaila import gait_wizard as gw


def _touch_session(folder: Path, cameras=("cam1", "cam2", "cam3")) -> Path:
    folder.mkdir(parents=True)
    for cam in cameras:
        (folder / f"{cam}_trial01.mp4").write_bytes(b"")
    return folder


# MediaPipe pose landmark order (0..32), used by mpangles.get_vector_landmark.
_LANDMARK_NAMES = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
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
]


def _landmarks_row(**positions: tuple[float, float]) -> pd.DataFrame:
    """One-frame pixel-landmarks DataFrame (frame, p0_x, p0_y, ...); unset landmarks are (0, 0)."""
    cols: dict[str, list[float]] = {"frame": [0]}
    for i, name in enumerate(_LANDMARK_NAMES):
        x, y = positions.get(name, (0.0, 0.0))
        cols[f"p{i}_x"] = [x]
        cols[f"p{i}_y"] = [y]
    return pd.DataFrame(cols)


def test_session_datetime_from_folder_name(tmp_path):
    folder = tmp_path / "trial01_20261010_094738"
    folder.mkdir()
    assert gw.session_datetime(folder) == datetime(2026, 10, 10, 9, 47, 38)


def test_find_sessions_newest_first_and_skips_outputs(tmp_path):
    _touch_session(tmp_path / "trial01_20261010_083159")
    newer = _touch_session(tmp_path / "trial01_20261010_095143")
    _touch_session(tmp_path / "processed_multicam_cut_20261012_101010")
    (tmp_path / "empty_folder").mkdir()
    sessions = gw.find_sessions(tmp_path)
    assert [s.name for s in sessions] == ["trial01_20261010_095143", "trial01_20261010_083159"]
    assert sessions[0].path == newer
    assert sessions[0].cameras == ["cam1", "cam2", "cam3"]
    assert sessions[0].status == "Nova"


def test_find_sessions_missing_root(tmp_path):
    assert gw.find_sessions(tmp_path / "nope") == []


def test_state_roundtrip_and_status(tmp_path):
    session = _touch_session(tmp_path / "trial01_20261010_094738")
    cut = _touch_session(session / "processed_multicam_cut_20261012_101010")
    gw.save_state(session, {"source_dir": str(cut), "cut_dir": str(cut), "participant": "João"})
    state = gw.load_state(session)
    assert state["participant"] == "João" and "updated" in state
    info = gw.find_sessions(tmp_path)[0]
    assert info.status == "Cortada"
    assert gw.latest_cut_dir(session) == cut
    analysis = session / "gait_analysis_x"
    analysis.mkdir()
    gw.save_state(session, {**state, "analysis_dir": str(analysis)})
    assert gw.find_sessions(tmp_path)[0].status == "Analisada"


def test_load_state_missing_or_corrupt(tmp_path):
    assert gw.load_state(tmp_path) == {}
    (tmp_path / gw.STATE_FILE).write_text("{not json", encoding="utf-8")
    assert gw.load_state(tmp_path) == {}


def test_camera_name_from_cut_file():
    assert gw.camera_name(Path("cam2_trial01_frame_10_to_500.mp4")) == "cam2"


def test_default_coronal_cameras_remembers_saved_value(monkeypatch):
    monkeypatch.setattr(
        gw,
        "load_config_value",
        lambda _section, key: "cam1,cam4" if key == "coronal_cameras" else "",
    )
    assert gw.default_coronal_cameras(["cam1", "cam2", "cam3"], ["cam2"]) == ["cam1"]


def test_default_coronal_cameras_no_saved_value(monkeypatch):
    monkeypatch.setattr(gw, "load_config_value", lambda *_args: "")
    assert gw.default_coronal_cameras(["cam1", "cam2", "cam3"], ["cam2", "cam3"]) == []


def test_process_coronal_cameras_noop_without_coronal(tmp_path):
    gw.process_coronal_cameras(tmp_path, [], tmp_path, log=lambda _m: None)
    assert not (tmp_path / "angles").exists()


def test_frontal_plane_metrics_pelvic_obliquity_sign():
    df = _landmarks_row(left_hip=(100, 200), right_hip=(160, 210))
    out = gw.frontal_plane_metrics(df)
    expected = np.degrees(np.arctan2(10, 60))
    assert out["pelvic_obliquity_deg"].iloc[0] == pytest.approx(expected)


def test_frontal_plane_metrics_knee_valgus_and_varus_signs():
    # Right leg, straight down (hip.x == ankle.x); knee offset decides valgus/varus.
    straight = {"left_hip": (100, 200), "right_hip": (160, 200), "right_ankle": (160, 400)}
    valgus = _landmarks_row(right_knee=(150, 300), **straight)  # knee shifts toward the midline
    varus = _landmarks_row(right_knee=(170, 300), **straight)  # knee shifts away from the midline

    out_valgus = gw.frontal_plane_metrics(valgus)["right_knee_deg"].iloc[0]
    out_varus = gw.frontal_plane_metrics(varus)["right_knee_deg"].iloc[0]
    expected_magnitude = np.degrees(np.arccos(9900 / 10100.0))

    assert out_valgus == pytest.approx(expected_magnitude)
    assert out_varus == pytest.approx(-expected_magnitude)


def test_frontal_plane_metrics_straight_leg_is_zero():
    df = _landmarks_row(
        left_hip=(100, 200),
        right_hip=(160, 200),
        right_knee=(160, 300),
        right_ankle=(160, 400),
    )
    out = gw.frontal_plane_metrics(df)
    assert out["right_knee_deg"].iloc[0] == pytest.approx(0.0)


def test_normalize_cycles_resamples_and_converts_knee():
    fs = 30.0
    n = 120
    times = np.arange(n) / fs
    # Knee included angle 180 - 30*sin: flexion should come back as 30*sin.
    phase = np.sin(2 * np.pi * times / 1.0)
    angles = pd.DataFrame(
        {
            "frame": np.arange(n),
            "left_knee": 180.0 - 30.0 * np.abs(phase),
            "left_hip": np.full(n, 170.0),
            "left_ankle": np.full(n, 90.0),
        }
    )
    cycles = pd.DataFrame(
        [
            {
                "camera": "cam2",
                "side": "left",
                "hs_time_s": 1.0,
                "next_hs_time_s": 2.0,
                "stance_pct": 60.0,
            },
            {
                "camera": "cam3",
                "side": "left",
                "hs_time_s": 1.0,
                "next_hs_time_s": 2.0,
                "stance_pct": 60.0,
            },
        ]
    )
    rows = gw.normalize_cycles(angles, times, cycles, "cam2")
    assert {r["joint"] for r in rows} == {"hip", "knee", "ankle"}
    knee = next(r for r in rows if r["joint"] == "knee")["curve"]
    assert knee.shape == (gw.CYCLE_POINTS,)
    assert knee.min() >= -2 and knee.max() == pytest.approx(30, abs=3)
    hip = next(r for r in rows if r["joint"] == "hip")["curve"]
    assert np.allclose(hip, 170.0, atol=0.5)

    table = gw.cycle_angle_table(rows)
    assert set(table["joint"]) == {"hip", "knee", "ankle"}
    assert (table["n_cycles"] == 1).all()


def test_normalize_cycles_drops_cycle_without_data():
    times = np.arange(60) / 30.0
    angles = pd.DataFrame({"frame": np.arange(60), "right_knee": np.full(60, np.nan)})
    cycles = pd.DataFrame(
        [
            {
                "camera": "cam2",
                "side": "right",
                "hs_time_s": 0.2,
                "next_hs_time_s": 1.5,
                "stance_pct": 60.0,
            }
        ]
    )
    assert gw.normalize_cycles(angles, times, cycles, "cam2") == []


def test_recording_advice():
    advice = gw.recording_advice(0, {"cam2": 1, "cam3": 0}, {"cam2": 0.05, "cam3": 0.89})
    text = "\n".join(advice)
    assert "cam3" in text and "89%" in text
    assert "nenhum ciclo" in text
    assert "cam2" not in text
    assert gw.recording_advice(12, {"cam2": 3}, {"cam2": 0.0}) == []
    assert "indicativos" in gw.recording_advice(2, {"cam2": 1}, {"cam2": float("nan")})[0]


def test_build_result_view_reads_summary(tmp_path):
    summary = pd.DataFrame(
        [
            {"camera": "all", "side": "left", "n_cycles": 2, "stance_pct_mean": 62.0},
            {"camera": "all", "side": "right", "n_cycles": 2, "stance_pct_mean": 58.0},
            {
                "camera": "all",
                "side": "both",
                "n_cycles": 4,
                "cadence_steps_min_mean": 110.4,
                "stance_pct_mean": 60.0,
                "swing_pct_mean": 40.0,
                "stride_time_s_mean": 1.087,
            },
        ]
    )
    summary.to_csv(tmp_path / "gait_summary.csv", index=False)
    view = gw.build_result_view(tmp_path, {"cam2": 2}, {"cam2": 0.0})
    values = {title: value for title, value, _ in view.cards}
    assert view.n_cycles == 4
    assert values["Cadência"] == "110 passos/min"
    assert values["Tempo da passada"] == "1,09 s"
    assert values["Simetria E/D"] == "7%"
    assert view.angles_png is None


def test_build_result_view_without_results(tmp_path):
    view = gw.build_result_view(tmp_path, {}, {})
    assert view.n_cycles == 0
    assert all(value == "–" for _, value, _ in view.cards)


def test_smooth_landmarks_csv_keeps_empty_frames(tmp_path):
    n = 90
    x = 100 + 10 * np.sin(np.arange(n) / 5) + np.random.default_rng(0).normal(0, 2, n)
    x[:20] = np.nan  # nobody in view
    x[50] = np.nan  # one missed detection
    src, dst = tmp_path / "v_pixel_vaila.csv", tmp_path / "v.csv"
    pd.DataFrame({"frame": np.arange(n), "p0_x": x, "p0_y": x}).to_csv(src, index=False)
    gw.smooth_landmarks_csv(src, dst, fs=30.0)
    out = pd.read_csv(dst)
    assert out["p0_x"][:20].isna().all()
    assert out["p0_x"][20:].notna().sum() == n - 21
    assert np.nanstd(np.diff(out["p0_x"][20:])) < np.nanstd(np.diff(x[20:]))


def test_angle_videos_puts_cameras_with_passes_first(tmp_path):
    angles = tmp_path / "angles"
    angles.mkdir()
    for name in ("angles_cam2_trial01.mp4", "angles_cam3_trial01.mp4"):
        (angles / name).write_bytes(b"")
    (angles / "cam2_trial01_rel.csv").write_text("")
    videos = gw.angle_videos(tmp_path, {"cam2": 0, "cam3": 2})
    assert [cam for cam, _ in videos] == ["cam3", "cam2"]
    assert gw.angle_videos(tmp_path / "none") == []


def test_add_angles_to_report_links_videos(tmp_path):
    report = tmp_path / "gait_report.html"
    report.write_text("<html><h2>Method notes</h2></html>", encoding="utf-8")
    video = tmp_path / "angles" / "angles_cam2_trial01.mp4"
    gw.add_angles_to_report(report, None, [("cam2", video)])
    text = report.read_text(encoding="utf-8")
    assert "src='angles/angles_cam2_trial01.mp4'" in text
    assert text.index("cam2: skeleton") < text.index("Method notes")


def test_print_report_prefers_pdf_and_report_info(tmp_path):
    assert gw.print_report(tmp_path, "pt") is None
    (tmp_path / "relatorio_marcha.html").write_text("x")
    assert gw.print_report(tmp_path, "pt").name == "relatorio_marcha.html"
    (tmp_path / "relatorio_marcha.pdf").write_bytes(b"%PDF")
    assert gw.print_report(tmp_path, "pt").name == "relatorio_marcha.pdf"
    assert gw.print_report(tmp_path, "en") is None
    info = gw.report_info({"participant": "P1", "age": "54", "affected_side": "left"}, "trial01")
    assert (info.participant, info.age, info.affected_side, info.recording) == (
        "P1",
        "54",
        "left",
        "trial01",
    )
