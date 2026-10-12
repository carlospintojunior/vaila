"""Tests for vaila/gait_report.py (printable gait report, PT/EN)."""

import numpy as np
import pandas as pd
import pytest

from vaila import gait_report as gr

FS = 30.0
STRIDE = 1.1  # s


def _knee_included(t: np.ndarray) -> np.ndarray:
    """MP Angles knee (included angle): ~5° flexion in stance, ~60° peak in swing."""
    phase = (t % STRIDE) / STRIDE
    flexion = 5 + 55 * np.exp(-(((phase - 0.72) / 0.1) ** 2))
    return 180.0 - flexion


def _walk(n_cycles: int = 5, t0: float = 1.0):
    n = int((t0 + n_cycles * STRIDE + 1.0) * FS)
    times = np.arange(n) / FS
    angles = pd.DataFrame(
        {
            "frame": np.arange(n),
            "neck": 160 + np.sin(times),
            "trunk": np.zeros(n),
            "right_knee": _knee_included(times - t0),
            "left_knee": _knee_included(times - t0 - STRIDE / 2),
            "right_hip": 165 + 10 * np.sin(2 * np.pi * (times - t0) / STRIDE),
        }
    )
    rows = []
    for side, offset in (("right", 0.0), ("left", STRIDE / 2)):
        for k in range(n_cycles - 1):
            hs = t0 + offset + k * STRIDE
            rows.append(
                {
                    "camera": "cam2",
                    "side": side,
                    "hs_time_s": hs,
                    "to_time_s": hs + 0.6 * STRIDE,
                    "next_hs_time_s": hs + STRIDE,
                    "stride_time_s": STRIDE,
                    "stance_pct": 60.0,
                    "swing_pct": 40.0,
                    "double_support_pct": 20.0,
                    "cadence_steps_min": 120 / STRIDE,
                }
            )
    return angles, times, pd.DataFrame(rows)


def test_split_column_and_clinical_angles():
    assert gr.split_column("right_knee") == ("right", "knee")
    assert gr.split_column("neck") == ("center", "neck")
    angles, _times, _cycles = _walk()
    clin = gr.clinical_angles(angles)
    assert "trunk" not in clin.columns  # constant column dropped
    assert clin["right_knee"].min() == pytest.approx(5, abs=0.5)  # flexion, 0 = straight


def test_normal_gait_peaks_are_not_extreme():
    angles, times, cycles = _walk()
    clin = gr.clinical_angles(angles)
    mask = gr.analysis_mask(times, gr.cycle_windows(cycles, "center"))
    extremes = gr.extreme_points(clin, times, mask, cycles)
    assert extremes.empty  # the ~60° knee peak in swing is the normal pattern


def test_departure_from_own_cycle_pattern_is_flagged():
    angles, times, cycles = _walk()
    clin = gr.clinical_angles(angles)
    # 25° extra knee flexion in mid-stance of the 3rd right cycle
    hs = float(cycles[cycles["side"] == "right"]["hs_time_s"].iloc[2])
    bump = (times > hs + 0.25) & (times < hs + 0.40)
    clin.loc[bump, "right_knee"] += 25
    mask = gr.analysis_mask(times, gr.cycle_windows(cycles, "center"))
    extremes = gr.extreme_points(clin, times, mask, cycles)
    pattern = extremes[(extremes["kind"] == "high") & (extremes["column"] == "right_knee")]
    assert len(pattern) == 1
    assert hs + 0.25 < pattern["time_s"].iloc[0] < hs + 0.40
    stats = gr.joint_statistics(clin, times, mask, cycles, extremes)
    knee = stats[stats["column"] == "right_knee"].iloc[0]
    assert knee["n_extreme"] >= 1 and knee["n_cycles"] == 4


def test_abrupt_jump_is_flagged_even_with_few_cycles():
    angles, times, cycles = _walk(n_cycles=2)
    clin = gr.clinical_angles(angles)
    clin.loc[60, "right_hip"] += 30  # one-frame tracking glitch
    mask = np.ones(len(times), dtype=bool)
    extremes = gr.extreme_points(clin, times, mask, cycles)
    jumps = extremes[extremes["kind"] == "jump"]
    assert (jumps["column"] == "right_hip").any()
    assert (extremes["kind"] != "high").all() or len(cycles) >= 3


def test_cycle_pattern_identical_cycles_have_no_variability():
    angles, times, cycles = _walk()
    clin = gr.clinical_angles(angles)
    mean, sd, n = gr.cycle_pattern(
        clin["right_knee"].to_numpy(), times, gr.cycle_windows(cycles, "right")
    )
    assert n == 4 and np.nanmax(sd) < 0.5 and mean.max() == pytest.approx(60, abs=2)


def test_reference_status_and_formatting():
    assert gr.reference_status("stance_pct", 60) == "ok"
    assert gr.reference_status("stance_pct", 67) == "above"
    assert gr.reference_status("cadence_steps_min", 80) == "below"
    assert gr.reference_status("stance_pct", float("nan")) == "na"
    assert gr._fmt(1.14, 2, " s", "pt") == "1,14 s"
    assert gr._fmt(1.14, 2, " s", "en") == "1.14 s"
    assert gr._fmt(-0.2, 0, "°") == "0°"


def _analysis_dir(tmp_path, n_cycles=5):
    angles, _times, cycles = _walk(n_cycles)
    session = tmp_path / "trial01_20261010_095143"
    out = session / "gait_analysis_20261012_000000"
    (out / "angles").mkdir(parents=True)
    angles.to_csv(out / "angles" / "cam2_trial01_rel.csv", index=False)
    cycles.to_csv(out / "gait_cycles.csv", index=False)
    summary = pd.DataFrame(
        [
            {"camera": "all", "side": "left", "n_cycles": 4, "stance_pct_mean": 60.0},
            {"camera": "all", "side": "right", "n_cycles": 4, "stance_pct_mean": 61.0},
            {
                "camera": "all",
                "side": "both",
                "n_cycles": 8,
                "cadence_steps_min_mean": 109.0,
                "stance_pct_mean": 67.0,
                "swing_pct_mean": 33.0,
                "double_support_pct_mean": 20.0,
                "stride_time_s_mean": 1.1,
            },
        ]
    )
    summary.to_csv(out / "gait_summary.csv", index=False)
    return out


@pytest.mark.parametrize("lang", ["pt", "en"])
def test_build_report_html(tmp_path, lang):
    out = _analysis_dir(tmp_path)
    info = gr.ReportInfo("Paciente X", "54", "right", "Dra. Y", "Lab Z")
    text = gr.build_report_html(out, info, lang)
    t = gr.TEXT[lang]
    assert t["title"] in text and "Paciente X" in text and "Lab Z" in text
    assert "trial01_20261010_095143" in text  # session name, not the analysis folder
    assert t["above"] in text  # stance 67 % is flagged
    assert text.count("<h3>") >= 3  # right / left / neck statistics tables
    assert t["affected_tag"] in text
    assert "window.print()" in text


def test_write_reports_without_pdf(tmp_path):
    out = _analysis_dir(tmp_path, n_cycles=2)
    reports = gr.write_reports(out, gr.ReportInfo(), pdf=False)
    assert set(reports) == {"pt", "en"}
    for html_path, pdf_path in reports.values():
        assert html_path.is_file() and pdf_path is None
    assert gr.TEXT["pt"]["pattern_needs_cycles"].split(":")[0] in (
        reports["pt"][0].read_text(encoding="utf-8")
    )
