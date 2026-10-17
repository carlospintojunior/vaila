"""
Project: vailá Multimodal Toolbox
Script: gait_report.py - Printable gait report (Portuguese / English)

Author: Carlos Pinto Jr
Email: carlos@vulcanum.com.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 12 October 2026
Update Date: 17 October 2026
Version: 0.4.9

Description:
Builds a printable report (HTML + PDF) from a Marcha Fácil / gait_auto
analysis folder, in Portuguese or English:

Page 1 (patient and therapist): identification, a plain-language summary,
temporal gait parameters next to literature reference ranges (values outside
the range are highlighted), joint angles over the gait cycle, four frames
of the MP Angles video (heel strike, mid-stance, toe-off, mid-swing), and
the MP Angles skeleton-sequence image for every analysed camera (lateral
and, when present, a coronal/frontal camera).

Page 2 (therapist): range of motion per side against references; MP Angles
statistics per joint split into Right / Left / Neck tables (mean, SD,
minimum and maximum with their instant, range, CV) instead of one wide
table; frontal-plane screening from the coronal camera when present (pelvic
obliquity, knee valgus/varus: mean/SD and signed peak, with a time chart);
the points of extreme oscillation (|z| above the threshold, or an
abrupt frame-to-frame jump), in bold and colour with the instant to check in
the video; gait cycles; method and references.

Screen-only appendix: the frame-by-frame angle tables split by side, with
the extreme values highlighted.

The PDF is printed by a Chromium-based browser in headless mode (Chrome,
Chromium or Edge); without one, the HTML has a "Print / Save as PDF" button.

Output (inside the analysis folder):
    relatorio_marcha.html / relatorio_marcha.pdf    (Portuguese)
    gait_report_print_en.html / gait_report_print_en.pdf   (English)

Usage:
    uv run python vaila/gait_report.py --analysis <gait_analysis_dir>
        [--lang pt en] [--participant ID] [--age 54] [--side right]
        [--therapist NAME] [--institution NAME] [--no-pdf]

References:
Perry J, Burnfield JM. Gait Analysis: Normal and Pathological Function.
2nd ed. Thorofare (NJ): SLACK; 2010.
Robinson RO, Herzog W, Nigg BM. Use of force platform variables to quantify
the effects of chiropractic manipulation on gait symmetry. J Manipulative
Physiol Ther. 1987;10(4):172-176.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import html
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd

try:
    from .gait_auto import _video_size, symmetry_index
    from .multicam_cut import read_frame_times
except ImportError:
    from gait_auto import _video_size, symmetry_index  # ty: ignore[unresolved-import]
    from multicam_cut import read_frame_times  # ty: ignore[unresolved-import]


Z_EXTREME = 2.5  # departure from the own mean cycle, in between-cycle SDs
PATTERN_MIN_DEG = 10.0  # ...and never less than this many degrees
MIN_PATTERN_CYCLES = 3  # cycles of one side needed to define its own pattern
CYCLE_POINTS = 101
JUMP_MIN_DEG = 15.0  # frame-to-frame change above max(this, JUMP_MAD x MAD) = abrupt jump
JUMP_MAD = 6.0
MAX_ATTENTION_ROWS = 20
REPORT_NAMES = {"pt": "relatorio_marcha", "en": "gait_report_print_en"}
SIDES = ("right", "left", "center")
JOINT_ORDER = ("hip", "knee", "ankle", "shoulder", "elbow", "wrist", "neck")
GAIT_JOINTS = ("hip", "knee", "ankle")

# Approximate ranges around the mean values of healthy adults walking at a
# self-selected speed (Perry & Burnfield 2010); symmetry: Robinson et al. 1987.
REFERENCES = {
    "cadence_steps_min": (100.0, 125.0),
    "stance_pct": (57.0, 63.0),
    "swing_pct": (37.0, 43.0),
    "double_support_pct": (16.0, 24.0),
    "stride_time_s": (0.95, 1.25),
    "symmetry_pct": (0.0, 10.0),
    "knee_peak_flexion": (55.0, 65.0),
    "hip_rom": (35.0, 45.0),
    "knee_rom": (55.0, 70.0),
    "ankle_rom": (25.0, 35.0),
}

TEXT = {
    "pt": {
        "title": "Relatório de Marcha",
        "subtitle": "Análise de vídeo (MediaPipe + MP Angles) — vailá",
        "participant": "Participante",
        "age": "Idade",
        "years": "anos",
        "affected": "Lado acometido",
        "therapist": "Terapeuta",
        "institution": "Instituição",
        "recording": "Gravação",
        "date": "Data do relatório",
        "cycles_measured": "Ciclos de marcha medidos",
        "summary": "Resumo",
        "parameters": "Parâmetros da marcha",
        "parameter": "Parâmetro",
        "value": "Valor",
        "reference": "Referência",
        "status": "Situação",
        "ok": "dentro da faixa",
        "above": "acima da faixa",
        "below": "abaixo da faixa",
        "na": "não medido",
        "angles_cycle": "Ângulos ao longo do ciclo da marcha",
        "chart_x": "% do ciclo da marcha",
        "chart_y": "graus (°)",
        "chart_hip": "Quadril (tronco-coxa)",
        "chart_ankle": "Tornozelo (perna-pé)",
        "angles_cycle_note": (
            "Média de cada lado (azul = esquerdo, vermelho = direito); a linha tracejada "
            "marca o fim do apoio (o pé sai do chão)."
        ),
        "key_frames": "Momentos do ciclo (vídeo com ângulos)",
        "kf_hs": "Toque do calcanhar",
        "kf_ms": "Meio do apoio",
        "kf_to": "Saída do pé",
        "kf_sw": "Meio do balanço",
        "skeleton_overview": "Esqueleto e ângulos ao longo da caminhada (MP Angles)",
        "skeleton_overview_note": (
            "Sequência de quadros com o esqueleto detectado e os ângulos das "
            "articulações, uma imagem por câmera analisada."
        ),
        "view_lateral": "vista lateral",
        "view_coronal": "vista frontal (coronal)",
        "frontal_plane": "Plano frontal (câmera coronal): obliquidade pélvica e valgo/varo do joelho",
        "frontal_plane_note": (
            "Estimativa 2D sem calibração, a partir da câmera de frente: sensível ao "
            "alinhamento da câmera e à rotação do corpo — use como triagem, não como "
            "medida absoluta. Positivo = queda do quadril direito / joelho em valgo "
            "(em direção à linha média); negativo = queda do quadril esquerdo / "
            "joelho em varo (afastando-se da linha média); zero = quadril nivelado / "
            "perna reta."
        ),
        "pelvic_obliquity_metric": "Obliquidade pélvica",
        "right_knee_metric": "Joelho direito (valgo/varo)",
        "left_knee_metric": "Joelho esquerdo (valgo/varo)",
        "knee_valgus_varus": "Valgo (+) / varo (−) do joelho",
        "peak": "Pico",
        "chart_pelvic_y": "graus (°) — + queda à direita",
        "chart_knee_y": "graus (°) — + valgo, − varo",
        "chart_frontal_x": "tempo (s)",
        "technical": "Detalhes técnicos (para o terapeuta)",
        "rom": "Amplitude de movimento no ciclo (média dos ciclos)",
        "joint": "Articulação",
        "side": "Lado",
        "stats": "Estatística dos ângulos durante a caminhada (MP Angles)",
        "stats_note": (
            "Calculada nos quadros dentro dos ciclos de marcha medidos (ou em todos os "
            "quadros com a pessoa visível, se não houve ciclo completo). Joelho em "
            "flexão (0° = perna esticada); demais ângulos como calculados pelo MP Angles "
            "(0–180°, sem sinal). Variabilidade entre ciclos: desvio-padrão médio entre os "
            "ciclos no mesmo ponto do ciclo (precisa de 2 ciclos ou mais do lado). Em "
            "vermelho: contagens de oscilação extrema."
        ),
        "mean": "Média",
        "sd": "DP",
        "min": "Mínimo (instante)",
        "max": "Máximo (instante)",
        "range": "Amplitude",
        "cycle_sd": "Variab. entre ciclos",
        "n_extreme": "Pontos extremos",
        "expected": "Esperado / anterior",
        "attention": "Pontos de oscilação extrema",
        "attention_note": (
            "Fora do padrão: o ângulo se afasta da média dos outros ciclos da pessoa, no mesmo "
            "ponto do ciclo, por mais de {z} desvios-padrão entre ciclos (e pelo menos "
            "{deg}°) — exige {n} ciclos ou mais do mesmo lado. Mudança brusca: o ângulo "
            "salta de um quadro para o outro. Podem indicar instabilidade, compensação ou "
            "erro de rastreamento: confira no vídeo no instante indicado."
        ),
        "time": "Instante (s)",
        "frame": "Quadro",
        "angle": "Ângulo",
        "kind": "Tipo",
        "kind_high": "acima do padrão",
        "kind_low": "abaixo do padrão",
        "kind_jump": "mudança brusca",
        "no_attention": "Nenhum ponto de oscilação extrema.",
        "pattern_needs_cycles": (
            "Comparação com o padrão do ciclo não feita: há menos de {n} ciclos de algum "
            "lado. Só mudanças bruscas foram verificadas."
        ),
        "cycles": "Ciclos de marcha",
        "method": "Método",
        "references": "Referências",
        "warnings": "Avisos sobre a gravação",
        "appendix": "Anexo: ângulos quadro a quadro",
        "appendix_note": "Valores extremos destacados. Esta seção não é impressa.",
        "print": "Imprimir / salvar PDF",
        "right": "Direito",
        "left": "Esquerdo",
        "center": "Pescoço",
        "both": "Ambos",
        "none": "Nenhum",
        "affected_tag": "lado acometido",
        "hip": "Quadril",
        "knee": "Joelho (flexão)",
        "ankle": "Tornozelo",
        "shoulder": "Ombro",
        "elbow": "Cotovelo",
        "wrist": "Punho",
        "neck": "Pescoço",
        "knee_peak_flexion": "Flexão máxima do joelho no balanço",
        "hip_rom": "Amplitude do quadril",
        "knee_rom": "Amplitude do joelho",
        "ankle_rom": "Amplitude do tornozelo",
        "cadence_steps_min": "Cadência",
        "stance_pct": "Apoio (% do ciclo)",
        "swing_pct": "Balanço (% do ciclo)",
        "double_support_pct": "Duplo apoio (% do ciclo)",
        "stride_time_s": "Tempo da passada",
        "symmetry_pct": "Diferença entre as pernas (apoio)",
        "unit_cadence": "passos/min",
        "sum_ok": "{name}: {value} — dentro da faixa de referência ({ref}).",
        "sum_above": "{name}: {value} — acima da faixa de referência ({ref}). {why}",
        "sum_below": "{name}: {value} — abaixo da faixa de referência ({ref}). {why}",
        "why_cadence_steps_min_above": "Passos mais curtos e rápidos que o habitual.",
        "why_cadence_steps_min_below": "Caminhada mais lenta, com menos passos por minuto.",
        "why_stance_pct_above": "O pé fica mais tempo no chão (busca de estabilidade).",
        "why_stance_pct_below": "O pé fica menos tempo no chão que o habitual.",
        "why_double_support_pct_above": "Mais tempo com os dois pés no chão (busca de estabilidade).",
        "why_double_support_pct_below": "Pouco tempo com os dois pés no chão.",
        "why_stride_time_s_above": "Cada passada demora mais que o habitual.",
        "why_stride_time_s_below": "Passadas mais rápidas que o habitual.",
        "why_symmetry_pct_above": "As duas pernas não se comportam da mesma forma.",
        "why_knee_peak_flexion_below": "O joelho dobra pouco ao levar a perna à frente.",
        "why_knee_peak_flexion_above": "O joelho dobra mais que o habitual no balanço.",
        "few_cycles": (
            "Foram medidos só {n} ciclo(s): os valores são indicativos. Para um resultado "
            "confiável, grave várias idas e voltas (10 ciclos ou mais)."
        ),
        "no_cycles": "Nenhum ciclo de marcha completo foi medido nesta gravação.",
        "extreme_summary": (
            "{n} ponto(s) de oscilação extrema nos ângulos — veja a página 2 e confira no vídeo."
        ),
        "ref_source": "Faixas aproximadas em torno dos valores médios de adultos saudáveis.",
        "method_text": (
            "Pose 2D por MediaPipe nas câmeras laterais; eventos de marcha pelo método de "
            "Zeni et al. (2008) com os tempos reais de cada quadro; ângulos pelo MP Angles "
            "sobre os pontos suavizados (passa-baixa de 6 Hz). Sem calibração: apenas "
            "tempos e ângulos 2D no plano da câmera. Câmera usada nas estatísticas: {cam}."
        ),
        "disclaimer": (
            "Relatório gerado automaticamente para apoio à avaliação. Não substitui a "
            "avaliação clínica."
        ),
    },
    "en": {
        "title": "Gait Report",
        "subtitle": "Video analysis (MediaPipe + MP Angles) — vailá",
        "participant": "Participant",
        "age": "Age",
        "years": "years",
        "affected": "Affected side",
        "therapist": "Therapist",
        "institution": "Institution",
        "recording": "Recording",
        "date": "Report date",
        "cycles_measured": "Gait cycles measured",
        "summary": "Summary",
        "parameters": "Gait parameters",
        "parameter": "Parameter",
        "value": "Value",
        "reference": "Reference",
        "status": "Status",
        "ok": "within range",
        "above": "above range",
        "below": "below range",
        "na": "not measured",
        "angles_cycle": "Joint angles over the gait cycle",
        "chart_x": "% of gait cycle",
        "chart_y": "degrees (°)",
        "chart_hip": "Hip (trunk-thigh)",
        "chart_ankle": "Ankle (shank-foot)",
        "angles_cycle_note": (
            "Mean of each side (blue = left, red = right); the dashed line marks the end of "
            "stance (the foot leaves the floor)."
        ),
        "key_frames": "Moments of the cycle (angle video)",
        "kf_hs": "Heel strike",
        "kf_ms": "Mid-stance",
        "kf_to": "Toe-off",
        "kf_sw": "Mid-swing",
        "skeleton_overview": "Skeleton and angles across the walk (MP Angles)",
        "skeleton_overview_note": (
            "Sequence of frames with the detected skeleton and joint angles, one "
            "image per analysed camera."
        ),
        "view_lateral": "side view",
        "view_coronal": "frontal view (coronal)",
        "frontal_plane": "Frontal plane (coronal camera): pelvic obliquity and knee valgus/varus",
        "frontal_plane_note": (
            "Uncalibrated 2D estimate from the frontal camera: sensitive to camera "
            "alignment and body rotation — use as a screening aid, not an absolute "
            "measure. Positive = right pelvic drop / valgus (knee toward the "
            "midline); negative = left pelvic drop / varus (knee away from the "
            "midline); zero = level pelvis / straight leg."
        ),
        "pelvic_obliquity_metric": "Pelvic obliquity",
        "right_knee_metric": "Right knee (valgus/varus)",
        "left_knee_metric": "Left knee (valgus/varus)",
        "knee_valgus_varus": "Knee valgus (+) / varus (−)",
        "peak": "Peak",
        "chart_pelvic_y": "degrees (°) — + right drop",
        "chart_knee_y": "degrees (°) — + valgus, − varus",
        "chart_frontal_x": "time (s)",
        "technical": "Technical details (for the therapist)",
        "rom": "Range of motion in the cycle (mean of cycles)",
        "joint": "Joint",
        "side": "Side",
        "stats": "Joint angle statistics while walking (MP Angles)",
        "stats_note": (
            "Computed on the frames inside the measured gait cycles (or on every frame with "
            "the person visible when no complete cycle was found). Knee as flexion "
            "(0° = straight leg); other angles as computed by MP Angles (0–180°, unsigned). "
            "Between-cycle variability: mean SD across cycles at the same point of the cycle "
            "(needs 2 or more cycles of that side). In red: extreme-oscillation counts."
        ),
        "mean": "Mean",
        "sd": "SD",
        "min": "Minimum (time)",
        "max": "Maximum (time)",
        "range": "Range",
        "cycle_sd": "Between-cycle var.",
        "n_extreme": "Extreme points",
        "expected": "Expected / previous",
        "attention": "Points of extreme oscillation",
        "attention_note": (
            "Off pattern: the angle departs from the mean of the person's other cycles, at the same "
            "point of the cycle, by more than {z} between-cycle SDs (and at least {deg}°) "
            "— needs {n} or more cycles of that side. Abrupt change: the angle jumps between "
            "consecutive frames. They may indicate instability, compensation or a tracking "
            "error: check the video at that time."
        ),
        "time": "Time (s)",
        "frame": "Frame",
        "angle": "Angle",
        "kind": "Type",
        "kind_high": "above pattern",
        "kind_low": "below pattern",
        "kind_jump": "abrupt change",
        "no_attention": "No point of extreme oscillation.",
        "pattern_needs_cycles": (
            "Comparison with the cycle pattern not done: fewer than {n} cycles on some side. "
            "Only abrupt changes were checked."
        ),
        "cycles": "Gait cycles",
        "method": "Method",
        "references": "References",
        "warnings": "Recording warnings",
        "appendix": "Appendix: frame-by-frame angles",
        "appendix_note": "Extreme values highlighted. This section is not printed.",
        "print": "Print / save PDF",
        "right": "Right",
        "left": "Left",
        "center": "Neck",
        "both": "Both",
        "none": "None",
        "affected_tag": "affected side",
        "hip": "Hip",
        "knee": "Knee (flexion)",
        "ankle": "Ankle",
        "shoulder": "Shoulder",
        "elbow": "Elbow",
        "wrist": "Wrist",
        "neck": "Neck",
        "knee_peak_flexion": "Peak knee flexion in swing",
        "hip_rom": "Hip range of motion",
        "knee_rom": "Knee range of motion",
        "ankle_rom": "Ankle range of motion",
        "cadence_steps_min": "Cadence",
        "stance_pct": "Stance (% of cycle)",
        "swing_pct": "Swing (% of cycle)",
        "double_support_pct": "Double support (% of cycle)",
        "stride_time_s": "Stride time",
        "symmetry_pct": "Difference between legs (stance)",
        "unit_cadence": "steps/min",
        "sum_ok": "{name}: {value} — within the reference range ({ref}).",
        "sum_above": "{name}: {value} — above the reference range ({ref}). {why}",
        "sum_below": "{name}: {value} — below the reference range ({ref}). {why}",
        "why_cadence_steps_min_above": "Shorter, quicker steps than usual.",
        "why_cadence_steps_min_below": "Slower walking, fewer steps per minute.",
        "why_stance_pct_above": "The foot stays on the floor longer (seeking stability).",
        "why_stance_pct_below": "The foot stays on the floor for less time than usual.",
        "why_double_support_pct_above": "More time with both feet on the floor (seeking stability).",
        "why_double_support_pct_below": "Little time with both feet on the floor.",
        "why_stride_time_s_above": "Each stride takes longer than usual.",
        "why_stride_time_s_below": "Faster strides than usual.",
        "why_symmetry_pct_above": "The two legs do not behave the same way.",
        "why_knee_peak_flexion_below": "The knee bends little when the leg swings forward.",
        "why_knee_peak_flexion_above": "The knee bends more than usual in swing.",
        "few_cycles": (
            "Only {n} cycle(s) were measured: values are indicative. For a reliable result, "
            "record several back-and-forth walks (10 cycles or more)."
        ),
        "no_cycles": "No complete gait cycle was measured in this recording.",
        "extreme_summary": (
            "{n} point(s) of extreme oscillation in the joint angles — see page 2 and check "
            "the video."
        ),
        "ref_source": "Approximate ranges around the mean values of healthy adults.",
        "method_text": (
            "2D pose by MediaPipe on the side-view cameras; gait events by the method of "
            "Zeni et al. (2008) using each frame's real timestamp; angles by MP Angles on "
            "the smoothed landmarks (6 Hz low-pass). No calibration: only times and 2D "
            "angles in the camera plane. Camera used for the statistics: {cam}."
        ),
        "disclaimer": (
            "Automatically generated report to support assessment. It does not replace "
            "clinical assessment."
        ),
    },
}

REFERENCE_TEXT = [
    "Perry J, Burnfield JM. Gait Analysis: Normal and Pathological Function. 2nd ed. "
    "Thorofare (NJ): SLACK; 2010.",
    "Robinson RO, Herzog W, Nigg BM. Use of force platform variables to quantify the effects "
    "of chiropractic manipulation on gait symmetry. J Manipulative Physiol Ther. "
    "1987;10(4):172-176.",
    "Zeni JA, Richards JG, Higginson JS. Two simple methods for determining gait events "
    "during treadmill and overground walking using kinematic data. Gait Posture. "
    "2008;27(4):710-714.",
]


@dataclass
class ReportInfo:
    participant: str = ""
    age: str = ""
    affected_side: str = ""  # "", "right", "left", "both"
    therapist: str = ""
    institution: str = ""
    recording: str = ""


# ── Data helpers ─────────────────────────────────────────────────────────────


def _records(df: pd.DataFrame) -> list[Any]:
    """Rows as attribute objects (``r.side``), for building tables."""
    return [SimpleNamespace(**rec) for rec in df.to_dict("records")]


def split_column(col: str) -> tuple[str, str]:
    """``right_knee`` -> (``right``, ``knee``); ``neck`` -> (``center``, ``neck``)."""
    for side in ("right", "left"):
        if col.startswith(side + "_"):
            return side, col[len(side) + 1 :]
    return "center", col


def clinical_angles(rel: pd.DataFrame) -> pd.DataFrame:
    """MP Angles relative angles with the knee as flexion; constant columns dropped."""
    out = rel.copy()
    for col in out.columns:
        if col.endswith("_knee"):
            out[col] = 180.0 - out[col]
    keep = ["frame"] + [
        c
        for c in out.columns
        if c != "frame" and out[c].notna().any() and np.nanstd(out[c].to_numpy(dtype=float)) > 0
    ]
    return out[keep]


def analysis_mask(times: np.ndarray, windows: list[tuple[float, float]]) -> np.ndarray:
    """Frames inside any (start, end) time window; all frames when there is no window."""
    if not windows:
        return np.ones(len(times), dtype=bool)
    mask = np.zeros(len(times), dtype=bool)
    for start, end in windows:
        mask |= (times >= start) & (times <= end)
    return mask


def cycle_windows(cycles: pd.DataFrame, side: str) -> list[tuple[float, float]]:
    """(heel strike, next heel strike) of the cycles of ``side`` (all cycles for ``center``)."""
    if cycles.empty:
        return []
    sub = cycles if side == "center" else cycles[cycles["side"] == side]
    return [
        (float(a), float(b)) for a, b in zip(sub["hs_time_s"], sub["next_hs_time_s"], strict=True)
    ]


def cycle_curves(
    values: np.ndarray, times: np.ndarray, windows: list[tuple[float, float]]
) -> list[tuple[tuple[float, float], np.ndarray]]:
    """``[(window, values resampled at 0..100 % of the cycle)]`` for cycles with enough data."""
    pct = np.linspace(0, 100, CYCLE_POINTS)
    out = []
    for start, end in windows:
        in_window = (times >= start) & (times <= end)
        inside = in_window & np.isfinite(values)
        if inside.sum() < 5 or inside.sum() < 0.8 * in_window.sum():
            continue
        cp = 100 * (times[inside] - start) / (end - start)
        out.append(((start, end), np.interp(pct, cp, values[inside])))
    return out


def cycle_pattern(
    values: np.ndarray, times: np.ndarray, windows: list[tuple[float, float]]
) -> tuple[np.ndarray, np.ndarray, int]:
    """Mean and SD (across cycles) of ``values`` at each % (0..100) of the gait cycle."""
    curves = [c for _w, c in cycle_curves(values, times, windows)]
    if not curves:
        return np.full(CYCLE_POINTS, np.nan), np.full(CYCLE_POINTS, np.nan), 0
    stack = np.vstack(curves)
    sd = stack.std(axis=0, ddof=1) if len(stack) > 1 else np.full(CYCLE_POINTS, np.nan)
    return stack.mean(axis=0), sd, len(stack)


def joint_statistics(
    angles: pd.DataFrame,
    times: np.ndarray,
    mask: np.ndarray,
    cycles: pd.DataFrame,
    extremes: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Mean, SD, min/max with their time, range, between-cycle variability and extreme count."""
    if extremes is None:
        extremes = extreme_points(angles, times, mask, cycles)
    counts = extremes.groupby("column").size() if not extremes.empty else pd.Series(dtype=int)
    rows = []
    for col in angles.columns:
        if col == "frame":
            continue
        values = angles[col].to_numpy(dtype=float)
        sel = mask & np.isfinite(values)
        if sel.sum() < 3:
            continue
        v, t = values[sel], times[sel]
        side, joint = split_column(col)
        _mean_curve, sd_curve, n_cyc = cycle_pattern(values, times, cycle_windows(cycles, side))
        rows.append(
            {
                "column": col,
                "side": side,
                "joint": joint,
                "n": int(sel.sum()),
                "mean": float(v.mean()),
                "sd": float(v.std(ddof=1)),
                "min": float(v.min()),
                "t_min": float(t[v.argmin()]),
                "max": float(v.max()),
                "t_max": float(t[v.argmax()]),
                "range": float(v.max() - v.min()),
                "n_cycles": n_cyc,
                "cycle_sd": float(np.nanmean(sd_curve)) if n_cyc > 1 else float("nan"),
                "n_extreme": int(counts.get(col, 0)),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    order = {j: i for i, j in enumerate(JOINT_ORDER)}
    side_order = {s: i for i, s in enumerate(SIDES)}
    out["_s"] = out["side"].map(side_order)
    out["_o"] = out["joint"].map(lambda j: order.get(j, 99))
    return out.sort_values(["_s", "_o"]).drop(columns=["_s", "_o"]).reset_index(drop=True)


def extreme_points(
    angles: pd.DataFrame,
    times: np.ndarray,
    mask: np.ndarray,
    cycles: pd.DataFrame,
    z: float = Z_EXTREME,
) -> pd.DataFrame:
    """Points of extreme oscillation, gait-aware.

    * ``pattern``: the angle departs from the person's own mean cycle at the same
      % of the cycle by more than ``z`` between-cycle SDs (at least
      PATTERN_MIN_DEG); needs MIN_PATTERN_CYCLES cycles of that side. A z-score
      over the whole walk would flag normal gait peaks (e.g. ~60° knee flexion in
      swing), so it is not used.
    * ``jump``: abrupt frame-to-frame change (tremor, loss of balance or a
      tracking error), anywhere in the analysed frames.
    One row per episode (consecutive flagged frames), at its largest deviation.
    """
    frames = angles["frame"].to_numpy(dtype=int)
    pct_axis = np.linspace(0, 100, CYCLE_POINTS)
    rows = []
    for col in angles.columns:
        if col == "frame":
            continue
        values = angles[col].to_numpy(dtype=float)
        sel = mask & np.isfinite(values)
        if sel.sum() < 10:
            continue
        side, _joint = split_column(col)
        windows = cycle_windows(cycles, side)
        curves = cycle_curves(values, times, windows)
        if len(curves) >= MIN_PATTERN_CYCLES:
            zs = np.zeros(len(values))
            expected = np.full(len(values), np.nan)
            stack = np.vstack([c for _w, c in curves])
            for i_cyc, ((start, end), _curve) in enumerate(curves):
                # Leave this cycle out: an odd cycle would otherwise pull the mean
                # and inflate the SD it is compared with, and hide itself.
                others = np.delete(stack, i_cyc, axis=0)
                mean_curve = others.mean(axis=0)
                sd_curve = others.std(axis=0, ddof=1)
                inside = (times >= start) & (times <= end) & np.isfinite(values)
                cp = 100 * (times[inside] - start) / (end - start)
                exp = np.interp(cp, pct_axis, mean_curve)
                sd = np.maximum(np.interp(cp, pct_axis, sd_curve), PATTERN_MIN_DEG / z)
                expected[inside] = exp
                zs[inside] = (values[inside] - exp) / sd
            flagged = np.abs(zs) > z
            i = 0
            while i < len(values):
                if not flagged[i]:
                    i += 1
                    continue
                j = i
                while (
                    j + 1 < len(values) and flagged[j + 1] and np.sign(zs[j + 1]) == np.sign(zs[i])
                ):
                    j += 1
                k = i + int(np.argmax(np.abs(zs[i : j + 1])))
                rows.append(_point(col, frames[k], times[k], values[k], expected[k], zs[k]))
                i = j + 1
        diffs = np.abs(np.diff(values))
        ok = sel[1:] & sel[:-1] & np.isfinite(diffs)
        if ok.sum() >= 5:
            mad = float(np.median(diffs[ok])) or 1e-6
            limit = max(JUMP_MIN_DEG, JUMP_MAD * mad)
            for k in np.flatnonzero(ok & (diffs > limit)) + 1:
                rows.append(
                    _point(
                        col,
                        frames[k],
                        times[k],
                        values[k],
                        values[k - 1],
                        float("nan"),
                        diffs[k - 1],
                    )
                )
    cols = [
        "column",
        "side",
        "joint",
        "frame",
        "time_s",
        "value",
        "expected",
        "z",
        "kind",
        "jump_deg",
    ]
    out = pd.DataFrame.from_records(rows, columns=cols)
    if out.empty:
        return out
    out["_k"] = out["z"].abs().fillna(0) + out["jump_deg"].fillna(0) / 10
    return out.sort_values("_k", ascending=False).drop(columns="_k").reset_index(drop=True)


def _point(col, frame, time, value, expected, z, jump: float | None = None) -> dict:
    side, joint = split_column(col)
    kind = "jump" if jump is not None else ("high" if z > 0 else "low")
    return {
        "column": col,
        "side": side,
        "joint": joint,
        "frame": int(frame),
        "time_s": float(time),
        "value": float(value),
        "expected": float(expected),
        "z": float(z),
        "kind": kind,
        "jump_deg": float(jump) if jump is not None else float("nan"),
    }


def cycle_rom(curves: pd.DataFrame) -> pd.DataFrame:
    """Peak, minimum and range of the mean cycle curve per joint and side."""
    rows = []
    if curves.empty:
        return pd.DataFrame(rows)
    for key, g in curves.groupby(["joint", "side"]):
        joint, side = cast(tuple[str, str], key)
        mean = g.sort_values("cycle_pct")["angle_mean_deg"].to_numpy(dtype=float)
        pct = g.sort_values("cycle_pct")["cycle_pct"].to_numpy(dtype=float)
        rows.append(
            {
                "joint": joint,
                "side": side,
                "n_cycles": int(g["n_cycles"].iloc[0]),
                "peak": float(np.nanmax(mean)),
                "peak_pct": float(pct[np.nanargmax(mean)]),
                "min": float(np.nanmin(mean)),
                "rom": float(np.nanmax(mean) - np.nanmin(mean)),
            }
        )
    return pd.DataFrame(rows)


def cycle_chart(curves: pd.DataFrame, cycles: pd.DataFrame, lang: str) -> str | None:
    """Hip / knee / ankle mean (± SD) over the gait cycle per side, labelled in ``lang``.

    Returns a base64 PNG, or None without curves.
    """
    if curves.empty:
        return None
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = TEXT[lang]
    titles = {"hip": t["chart_hip"], "knee": t["knee"], "ankle": t["chart_ankle"]}
    colors = {"left": "#1f77b4", "right": "#d62728"}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.0))
    for ax, joint in zip(axes, GAIT_JOINTS, strict=True):
        for side, color in colors.items():
            g = curves[(curves["joint"] == joint) & (curves["side"] == side)].sort_values(
                "cycle_pct"
            )
            if g.empty:
                continue
            pct = g["cycle_pct"].to_numpy(dtype=float)
            mean = g["angle_mean_deg"].to_numpy(dtype=float)
            sd = g["angle_sd_deg"].to_numpy(dtype=float)
            n = int(g["n_cycles"].iloc[0])
            ax.plot(pct, mean, color=color, lw=2, label=f"{t[side]} (n={n})")
            if np.isfinite(sd).any():
                ax.fill_between(pct, mean - sd, mean + sd, color=color, alpha=0.15)
            if not cycles.empty and (cycles["side"] == side).any():
                to_pct = float(cycles[cycles["side"] == side]["stance_pct"].mean())
                ax.axvline(to_pct, color=color, ls="--", lw=1, alpha=0.7)
        ax.set_title(titles[joint], fontsize=11)
        ax.set_xlabel(t["chart_x"])
        ax.set_xlim(0, 100)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(t["chart_y"])
    axes[0].legend(fontsize=8, loc="best")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def reference_status(metric: str, value: float) -> str:
    """``ok`` / ``above`` / ``below`` against REFERENCES, ``na`` when not measured."""
    if metric not in REFERENCES or not np.isfinite(value):
        return "na"
    low, high = REFERENCES[metric]
    if value < low:
        return "below"
    if value > high:
        return "above"
    return "ok"


def _summary_mean(summary: pd.DataFrame, metric: str) -> float:
    try:
        row = summary[(summary["camera"] == "all") & (summary["side"] == "both")].iloc[0]
        return float(row[f"{metric}_mean"])
    except (IndexError, KeyError, TypeError, ValueError):
        return float("nan")


def best_camera(cycles: pd.DataFrame, rel_files: dict[str, Path]) -> str | None:
    """Camera with the most complete cycles (else the first one with angles)."""
    if not cycles.empty:
        counts = cycles["camera"].value_counts()
        for cam in counts.index:
            if cam in rel_files:
                return str(cam)
    return next(iter(rel_files), None)


def key_frame_times(cycle: pd.Series) -> list[tuple[str, float]]:
    hs, to, nxt = (
        float(cycle["hs_time_s"]),
        float(cycle["to_time_s"]),
        float(cycle["next_hs_time_s"]),
    )
    return [("kf_hs", hs), ("kf_ms", (hs + to) / 2), ("kf_to", to), ("kf_sw", (to + nxt) / 2)]


def grab_frames(video: Path, times: np.ndarray, wanted: list[tuple[str, float]], width: int = 420):
    """``[(label, base64 JPEG)]`` of the frames of ``video`` closest to each wanted time."""
    import cv2

    out = []
    cap = cv2.VideoCapture(str(video))
    try:
        for label, t in wanted:
            idx = int(np.argmin(np.abs(times - t)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (width, int(h * width / w)))
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if ok:
                out.append((label, base64.b64encode(buf.tobytes()).decode("ascii")))
    finally:
        cap.release()
    return out


def stick_sequence_images(angles_dir: Path, lateral_cams: set[str]) -> list[tuple[str, bool, str]]:
    """``[(camera, is_lateral, base64 PNG)]`` for every MP Angles stick-figure sequence found.

    One image per analysed camera (``mpangles.process_video_with_visualization``
    always renders ``<stem>_stick_sequence.png`` into ``angles_dir``); cameras not
    in ``lateral_cams`` (e.g. a coronal/frontal camera) are flagged as such.
    """
    if not angles_dir.is_dir():
        return []
    out = []
    for path in sorted(angles_dir.glob("*_stick_sequence.png")):
        camera = path.name.split("_")[0]
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        out.append((camera, camera in lateral_cams, data))
    return out


FRONTAL_METRICS = ("pelvic_obliquity_deg", "left_knee_deg", "right_knee_deg")


def frontal_plane_files(angles_dir: Path) -> list[tuple[str, Path]]:
    """``[(camera, csv_path)]`` for every coronal camera's frontal-plane metrics CSV."""
    if not angles_dir.is_dir():
        return []
    return [(p.name.split("_")[0], p) for p in sorted(angles_dir.glob("*_frontal.csv"))]


def frontal_plane_stats(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Mean, SD and signed peak (furthest from 0) per frontal-plane metric column."""
    out = {}
    for col in FRONTAL_METRICS:
        values = df[col].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            out[col] = {"mean": float("nan"), "sd": float("nan"), "peak": float("nan")}
            continue
        out[col] = {
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
            "peak": float(values[np.argmax(np.abs(values))]),
        }
    return out


def frontal_plane_times(angles_dir: Path, stem: str, n: int) -> np.ndarray:
    """Real per-frame times (s) for a coronal camera's video, else an approximation."""
    video = angles_dir.parent.parent / f"{stem}.mp4"
    if video.is_file():
        _w, _h, count, fps = _video_size(video)
        return read_frame_times(video, max(count, n), fps)[:n]
    return np.arange(n, dtype=float) / 30.0


def frontal_plane_chart(df: pd.DataFrame, times: np.ndarray, lang: str) -> str | None:
    """Pelvic obliquity and knee valgus/varus over time, as a base64 PNG (or ``None``)."""
    try:
        import io

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    t = TEXT[lang]
    colors = {"left": "#1f77b4", "right": "#d62728"}
    time_s = times[: len(df)]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.0))
    axes[0].plot(time_s, df["pelvic_obliquity_deg"], color="#6b7280", lw=1.4)
    axes[0].axhline(0, color="#999", lw=1, ls="--")
    axes[0].set_title(t["pelvic_obliquity_metric"], fontsize=11)
    axes[0].set_ylabel(t["chart_pelvic_y"])
    axes[1].plot(time_s, df["right_knee_deg"], color=colors["right"], lw=1.4, label=t["right"])
    axes[1].plot(time_s, df["left_knee_deg"], color=colors["left"], lw=1.4, label=t["left"])
    axes[1].axhline(0, color="#999", lw=1, ls="--")
    axes[1].set_title(t["knee_valgus_varus"], fontsize=11)
    axes[1].set_ylabel(t["chart_knee_y"])
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel(t["chart_frontal_x"])
        ax.grid(alpha=0.3)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ── HTML ─────────────────────────────────────────────────────────────────────

CSS = """
:root { --ink:#1f2933; --muted:#616e7c; --line:#d9e2ec; --accent:#1565c0;
        --ok:#2e7d32; --warn:#b26a00; --bad:#c62828; --bg:#ffffff; --soft:#f5f7fa; }
* { box-sizing: border-box; }
body { font-family: 'Segoe UI', Arial, Helvetica, sans-serif; color: var(--ink);
       background: #eef1f5; margin: 0; padding: 16px; font-size: 13px; line-height: 1.45; }
.page { background: var(--bg); max-width: 860px; margin: 0 auto 16px; padding: 28px 32px;
        box-shadow: 0 1px 6px rgba(0,0,0,.08); }
header { display: flex; justify-content: space-between; align-items: flex-end;
         border-bottom: 3px solid var(--accent); padding-bottom: 8px; margin-bottom: 14px; }
h1 { margin: 0; font-size: 24px; color: var(--accent); }
.sub { color: var(--muted); font-size: 12px; }
.inst { text-align: right; font-weight: 600; color: var(--muted); }
h2 { font-size: 16px; margin: 18px 0 6px; color: var(--accent);
     border-left: 4px solid var(--accent); padding-left: 8px; }
h3 { font-size: 14px; margin: 12px 0 4px; }
.id { display: grid; grid-template-columns: repeat(3, 1fr); gap: 4px 16px;
      background: var(--soft); border: 1px solid var(--line); padding: 10px 12px; border-radius: 6px; }
.id div span { color: var(--muted); display: block; font-size: 11px; }
ul.summary { margin: 4px 0; padding-left: 0; list-style: none; }
ul.summary li { margin: 3px 0; }
table { border-collapse: collapse; width: 100%; margin: 6px 0 10px; font-size: 12px; }
th, td { border: 1px solid var(--line); padding: 4px 6px; text-align: right; }
th { background: var(--soft); color: var(--ink); text-align: center; font-weight: 600; }
td:first-child, th:first-child, td.l { text-align: left; }
.ok { color: var(--ok); }
.above, .below, .warn { color: var(--warn); font-weight: 700; }
.bad { color: var(--bad); font-weight: 700; }
.tag { font-size: 11px; color: var(--bad); font-weight: 600; }
.note { color: var(--muted); font-size: 11.5px; }
.frames { display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; }
.frames figure { margin: 0; text-align: center; font-size: 11px; color: var(--muted); }
.frames img { width: 100%; border-radius: 4px; border: 1px solid var(--line); }
.skeletons { display: flex; flex-direction: column; gap: 10px; margin: 8px 0; }
.skeletons figure { margin: 0; text-align: center; font-size: 11.5px; color: var(--muted); }
.skeletons img { width: 100%; border-radius: 4px; border: 1px solid var(--line); }
img.chart { width: 100%; }
.alert { background: #fff8e1; border-left: 4px solid var(--warn); padding: 6px 10px; margin: 4px 0; }
.two { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.disclaimer { color: var(--muted); font-size: 11px; margin-top: 14px; border-top: 1px solid var(--line);
              padding-top: 6px; }
.toolbar { max-width: 860px; margin: 0 auto 10px; text-align: right; }
.toolbar button { font-size: 14px; padding: 8px 16px; background: var(--accent); color: #fff;
                  border: 0; border-radius: 6px; cursor: pointer; }
details { margin-top: 8px; }
.scroll { max-height: 420px; overflow: auto; }
@media (max-width: 700px) { .id, .two { grid-template-columns: 1fr; } .frames { grid-template-columns: 1fr 1fr; } }
@media print {
  @page { size: A4; margin: 12mm; }
  body { background: #fff; padding: 0; font-size: 10.5px; line-height: 1.35; }
  header { margin-bottom: 8px; padding-bottom: 4px; }
  h1 { font-size: 20px; }
  h2 { font-size: 13.5px; margin: 10px 0 4px; }
  th, td { padding: 2px 5px; }
  table { margin: 4px 0 6px; font-size: 10.5px; }
  .id { padding: 6px 10px; }
  img.chart { max-height: 58mm; object-fit: contain; }
  .frames img { max-height: 30mm; object-fit: cover; }
  .skeletons img { max-height: 70mm; object-fit: contain; }
  .page { box-shadow: none; margin: 0; padding: 0; max-width: none; page-break-after: always; }
  .page:last-of-type { page-break-after: auto; }
  .toolbar, .screen-only { display: none !important; }
  table, figure, .alert { page-break-inside: avoid; }
}
"""


def _esc(text: object) -> str:
    return html.escape(str(text))


def _fmt(value: float, digits: int = 1, unit: str = "", lang: str = "pt") -> str:
    if value is None or not np.isfinite(value):
        return "–"
    if abs(value) < 0.5 * 10 ** (-digits):
        value = 0.0
    text = f"{value:.{digits}f}"
    if lang == "pt":
        text = text.replace(".", ",")
    return f"{text}{unit}"


def _ref_text(metric: str, unit: str, digits: int, lang: str) -> str:
    low, high = REFERENCES[metric]
    if unit == "s":
        unit = " s"
    if metric == "symmetry_pct":
        return f"≤ {_fmt(high, 0, unit, lang)}"
    return f"{_fmt(low, digits, '', lang)}–{_fmt(high, digits, unit, lang)}"


def _side_name(side: str, t: dict, affected: str) -> str:
    name = t.get(side, side)
    if affected and (affected == side or (affected == "both" and side in ("right", "left"))):
        name += f" <span class='tag'>({t['affected_tag']})</span>"
    return name


def build_report_html(analysis_dir: Path, info: ReportInfo, lang: str = "pt") -> str:
    """The printable report of one analysis folder, in ``lang`` ("pt" or "en")."""
    t = TEXT[lang]
    analysis_dir = Path(analysis_dir)
    summary = _read_csv(analysis_dir / "gait_summary.csv")
    cycles = _read_csv(analysis_dir / "gait_cycles.csv")
    curves = _read_csv(analysis_dir / "joint_angles_cycle.csv")
    angles_dir = analysis_dir / "angles"
    rel_files = {
        p.name.split("_")[0]: p
        for p in sorted(angles_dir.glob("*_rel.csv"))
        if not p.name.startswith("angles_")
    }
    camera = best_camera(cycles, rel_files)
    skeletons = stick_sequence_images(angles_dir, set(rel_files))
    frontal_cams = [
        (cam, path.name[: -len("_frontal.csv")], pd.read_csv(path))
        for cam, path in frontal_plane_files(angles_dir)
    ]
    if not info.recording:
        parent = analysis_dir.parent
        info.recording = parent.parent.name if parent.name.startswith("processed_") else parent.name
    n_cycles = len(cycles)

    # Temporal parameters
    sym = symmetry_index(summary, "stance_pct") if not summary.empty else float("nan")
    params = [
        (
            "cadence_steps_min",
            _summary_mean(summary, "cadence_steps_min"),
            0,
            " " + t["unit_cadence"],
        ),
        ("stance_pct", _summary_mean(summary, "stance_pct"), 0, "%"),
        ("swing_pct", _summary_mean(summary, "swing_pct"), 0, "%"),
        ("double_support_pct", _summary_mean(summary, "double_support_pct"), 0, "%"),
        ("stride_time_s", _summary_mean(summary, "stride_time_s"), 2, " s"),
        ("symmetry_pct", abs(sym) if np.isfinite(sym) else float("nan"), 0, "%"),
    ]
    rom = cycle_rom(curves)
    knee_peak = float("nan")
    if not rom.empty and (rom["joint"] == "knee").any():
        knee_peak = float(rom[rom["joint"] == "knee"]["peak"].mean())
    params.append(("knee_peak_flexion", knee_peak, 0, "°"))

    # Angle statistics on the main camera
    stats = pd.DataFrame()
    extremes = pd.DataFrame()
    cycles_ok = False
    frames_b64: list[tuple[str, str]] = []
    coronal_frames: list[tuple[str, list[tuple[str, str]]]] = []
    angles = pd.DataFrame()
    times = np.array([])
    if camera is not None:
        rel_csv = rel_files[camera]
        stem = rel_csv.name[: -len("_rel.csv")]
        video = analysis_dir.parent / f"{stem}.mp4"
        angles = clinical_angles(pd.read_csv(rel_csv))
        if video.is_file():
            _w, _h, count, fps = _video_size(video)
            times = read_frame_times(video, max(count, len(angles)), fps)
        else:
            times = angles["frame"].to_numpy(dtype=float) / 30.0
        frame_idx = angles["frame"].to_numpy(dtype=int).clip(0, len(times) - 1)
        times_f = times[frame_idx]
        cam_cycles = cycles[cycles["camera"] == camera] if not cycles.empty else cycles
        mask = analysis_mask(times_f, cycle_windows(cam_cycles, "center"))
        extremes = extreme_points(angles, times_f, mask, cam_cycles)
        cycles_ok = not cam_cycles.empty and all(
            (cam_cycles["side"] == side).sum() >= MIN_PATTERN_CYCLES for side in ("right", "left")
        )
        stats = joint_statistics(angles, times_f, mask, cam_cycles, extremes)
        angle_video = angles_dir / f"angles_{stem}.mp4"
        if angle_video.is_file() and not cam_cycles.empty:
            frames_b64 = grab_frames(angle_video, times, key_frame_times(cam_cycles.iloc[0]))
            wanted = key_frame_times(cam_cycles.iloc[0])
            for coronal_cam, coronal_stem, coronal_df in frontal_cams:
                coronal_video = angles_dir / f"angles_{coronal_stem}.mp4"
                if not coronal_video.is_file():
                    continue
                coronal_times = frontal_plane_times(angles_dir, coronal_stem, len(coronal_df))
                frames = grab_frames(coronal_video, coronal_times, wanted)
                if frames:
                    coronal_frames.append((coronal_cam, frames))
        times = times_f

    # Plain-language summary
    bullets = []
    if n_cycles == 0:
        bullets.append(f"<li class='warn'>{_esc(t['no_cycles'])}</li>")
    for metric, value, digits, unit in params:
        if metric in ("swing_pct",):
            continue  # mirrors stance
        status = reference_status(metric, value)
        if status == "na":
            continue
        ref = _ref_text(
            metric, unit.strip() if unit.strip() != t["unit_cadence"] else "", digits, lang
        )
        text = t[f"sum_{status}"].format(
            name=t[metric],
            value=_fmt(value, digits, unit, lang),
            ref=ref,
            why=t.get(f"why_{metric}_{status}", ""),
        )
        cls = "ok" if status == "ok" else "warn"
        bullets.append(f"<li><span class='{cls}'>●</span> {_esc(text)}</li>")
    if 0 < n_cycles < 3:
        bullets.append(f"<li class='warn'>{_esc(t['few_cycles'].format(n=n_cycles))}</li>")
    gait_extremes = (
        extremes[extremes["joint"].isin(GAIT_JOINTS)] if not extremes.empty else extremes
    )
    if len(gait_extremes):
        bullets.append(f"<li>{_esc(t['extreme_summary'].format(n=len(gait_extremes)))}</li>")

    # Parameters table
    param_rows = []
    for metric, value, digits, unit in params:
        status = reference_status(metric, value)
        ref_unit = unit.strip() if unit.strip() != t["unit_cadence"] else ""
        param_rows.append(
            f"<tr><td>{_esc(t[metric])}</td>"
            f"<td class='{'' if status in ('ok', 'na') else status}'>{_esc(_fmt(value, digits, unit, lang))}</td>"
            f"<td>{_esc(_ref_text(metric, ref_unit, digits, lang))}</td>"
            f"<td class='{status if status != 'na' else ''}'>{_esc(t[status])}</td></tr>"
        )

    affected = info.affected_side
    id_items = [
        (t["participant"], info.participant or "–"),
        (t["age"], f"{info.age} {t['years']}" if info.age else "–"),
        (t["affected"], t.get(affected, "–") if affected else t["none"]),
        (t["therapist"], info.therapist or "–"),
        (t["recording"], info.recording or analysis_dir.parent.name),
        (t["date"], datetime.now().strftime("%d/%m/%Y" if lang == "pt" else "%Y-%m-%d")),
    ]
    id_html = "".join(f"<div><span>{_esc(k)}</span>{_esc(v)}</div>" for k, v in id_items)

    chart = ""
    data = cycle_chart(
        curves, cycles[cycles["camera"] == camera] if camera and not cycles.empty else cycles, lang
    )
    if data:
        chart = (
            f"<h2>{_esc(t['angles_cycle'])}</h2><img class='chart' src='data:image/png;base64,{data}'>"
            f"<div class='note'>{_esc(t['angles_cycle_note'])}</div>"
        )
    frames_html = ""
    if frames_b64:
        figs = "".join(
            f"<figure><img src='data:image/jpeg;base64,{b}'><figcaption>{_esc(t[label])}</figcaption></figure>"
            for label, b in frames_b64
        )
        frames_html = (
            f"<h2>{_esc(t['key_frames'])} — {_esc(camera)}</h2><div class='frames'>{figs}</div>"
        )
    for coronal_cam, frames in coronal_frames:
        figs = "".join(
            f"<figure><img src='data:image/jpeg;base64,{b}'><figcaption>{_esc(t[label])}</figcaption></figure>"
            for label, b in frames
        )
        frames_html += (
            f"<h2>{_esc(t['key_frames'])} — {_esc(coronal_cam)} "
            f"({_esc(t['view_coronal'])})</h2><div class='frames'>{figs}</div>"
        )

    skeleton_html = ""
    if skeletons:
        figs = "".join(
            f"<figure><img src='data:image/png;base64,{b}'>"
            f"<figcaption>{_esc(cam)} — "
            f"{_esc(t['view_lateral'] if lat else t['view_coronal'])}</figcaption></figure>"
            for cam, lat, b in skeletons
        )
        skeleton_html = (
            f"<h2>{_esc(t['skeleton_overview'])}</h2>"
            f"<div class='note'>{_esc(t['skeleton_overview_note'])}</div>"
            f"<div class='skeletons'>{figs}</div>"
        )

    page1 = f"""
<section class="page">
  <header><div><h1>{_esc(t["title"])}</h1><div class="sub">{_esc(t["subtitle"])}</div></div>
  <div class="inst">{_esc(info.institution)}</div></header>
  <div class="id">{id_html}</div>
  <h2>{_esc(t["summary"])}</h2>
  <div class="sub">{_esc(t["cycles_measured"])}: <b>{n_cycles}</b></div>
  <ul class="summary">{"".join(bullets)}</ul>
  <h2>{_esc(t["parameters"])}</h2>
  <table><thead><tr><th>{_esc(t["parameter"])}</th><th>{_esc(t["value"])}</th>
  <th>{_esc(t["reference"])}</th><th>{_esc(t["status"])}</th></tr></thead>
  <tbody>{"".join(param_rows)}</tbody></table>
  <div class="note">{_esc(t["ref_source"])}</div>
  {chart}
  {frames_html}
  {skeleton_html}
  <div class="disclaimer">{_esc(t["disclaimer"])}</div>
</section>"""

    # Page 2: technical
    rom_rows = []
    for joint in GAIT_JOINTS:
        for side in ("right", "left"):
            r = rom[(rom["joint"] == joint) & (rom["side"] == side)] if not rom.empty else rom
            if r.empty:
                continue
            r0 = r.iloc[0]
            metric = f"{joint}_rom"
            status = reference_status(metric, r0["rom"])
            rom_rows.append(
                f"<tr><td class='l'>{_esc(t[joint])}</td><td class='l'>{_side_name(side, t, affected)}</td>"
                f"<td>{_esc(_fmt(r0['min'], 0, '°', lang))}</td><td>{_esc(_fmt(r0['peak'], 0, '°', lang))}</td>"
                f"<td class='{'' if status in ('ok', 'na') else status}'>{_esc(_fmt(r0['rom'], 0, '°', lang))}</td>"
                f"<td>{_esc(_ref_text(metric, '°', 0, lang))}</td><td>{int(r0['n_cycles'])}</td></tr>"
            )
    rom_html = ""
    if rom_rows:
        rom_html = (
            f"<h3>{_esc(t['rom'])}</h3><table><thead><tr><th>{_esc(t['joint'])}</th>"
            f"<th>{_esc(t['side'])}</th><th>{_esc(t['min'].split(' ')[0])}</th>"
            f"<th>{_esc(t['max'].split(' ')[0])}</th><th>{_esc(t['range'])}</th>"
            f"<th>{_esc(t['reference'])}</th><th>n</th></tr></thead><tbody>{''.join(rom_rows)}</tbody></table>"
        )

    stats_html = ""
    if not stats.empty:
        parts = []
        for side in SIDES:
            sub = stats[stats["side"] == side]
            if sub.empty:
                continue
            rows = []
            for r in _records(sub):
                gait = r.joint in GAIT_JOINTS
                name = (
                    f"<b>{_esc(t.get(r.joint, r.joint))}</b>"
                    if gait
                    else _esc(t.get(r.joint, r.joint))
                )
                ext = f"<td class='bad'>{r.n_extreme}</td>" if r.n_extreme else "<td>0</td>"
                rows.append(
                    f"<tr><td>{name}</td><td>{_esc(_fmt(r.mean, 1, '°', lang))}</td>"
                    f"<td>{_esc(_fmt(r.sd, 1, '°', lang))}</td>"
                    f"<td>{_esc(_fmt(r.min, 0, '°', lang))} ({_esc(_fmt(r.t_min, 2, ' s', lang))})</td>"
                    f"<td>{_esc(_fmt(r.max, 0, '°', lang))} ({_esc(_fmt(r.t_max, 2, ' s', lang))})</td>"
                    f"<td>{_esc(_fmt(r.range, 0, '°', lang))}</td>"
                    f"<td>{_esc(_fmt(r.cycle_sd, 1, '°', lang))}</td>{ext}</tr>"
                )
            parts.append(
                f"<h3>{_side_name(side, t, affected)}</h3><table><thead><tr>"
                f"<th>{_esc(t['joint'])}</th><th>{_esc(t['mean'])}</th><th>{_esc(t['sd'])}</th>"
                f"<th>{_esc(t['min'])}</th><th>{_esc(t['max'])}</th><th>{_esc(t['range'])}</th>"
                f"<th>{_esc(t['cycle_sd'])}</th><th>{_esc(t['n_extreme'])}</th></tr></thead>"
                f"<tbody>{''.join(rows)}</tbody></table>"
            )
        stats_html = (
            f"<h2>{_esc(t['stats'])}</h2><div class='note'>{_esc(t['stats_note'])}</div>"
            + "".join(parts)
        )

    frontal_html = ""
    for cam, stem, fdf in frontal_cams:
        stats = frontal_plane_stats(fdf)
        times = frontal_plane_times(angles_dir, stem, len(fdf))
        chart_b64 = frontal_plane_chart(fdf, times, lang)
        rows = "".join(
            f"<tr><td class='l'>{_esc(t[label])}</td>"
            f"<td>{_esc(_fmt(stats[col]['mean'], 1, '°', lang))} ± "
            f"{_esc(_fmt(stats[col]['sd'], 1, '°', lang))}</td>"
            f"<td>{_esc(_fmt(stats[col]['peak'], 1, '°', lang))}</td></tr>"
            for col, label in (
                ("pelvic_obliquity_deg", "pelvic_obliquity_metric"),
                ("right_knee_deg", "right_knee_metric"),
                ("left_knee_deg", "left_knee_metric"),
            )
        )
        chart_html = (
            f"<img class='chart' src='data:image/png;base64,{chart_b64}'>" if chart_b64 else ""
        )
        frontal_html += (
            f"<h2>{_esc(t['frontal_plane'])} — {_esc(cam)}</h2>"
            f"<div class='note'>{_esc(t['frontal_plane_note'])}</div>"
            f"{chart_html}"
            f"<table><thead><tr><th>{_esc(t['parameter'])}</th><th>{_esc(t['mean'])} ± "
            f"{_esc(t['sd'])}</th><th>{_esc(t['peak'])}</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
        )

    attention_html = f"<h2>{_esc(t['attention'])}</h2>"
    attention_html += (
        "<div class='note'>"
        + _esc(
            t["attention_note"].format(
                z=_fmt(Z_EXTREME, 1, "", lang),
                deg=_fmt(PATTERN_MIN_DEG, 0, "", lang),
                n=MIN_PATTERN_CYCLES,
            )
        )
        + "</div>"
    )
    if not cycles_ok:
        attention_html += f"<div class='note warn'>{_esc(t['pattern_needs_cycles'].format(n=MIN_PATTERN_CYCLES))}</div>"
    if extremes.empty:
        attention_html += f"<p>{_esc(t['no_attention'])}</p>"
    else:
        ordered = pd.concat(
            [
                extremes[extremes["joint"].isin(GAIT_JOINTS)],
                extremes[~extremes["joint"].isin(GAIT_JOINTS)],
            ]
        ).head(MAX_ATTENTION_ROWS)
        rows = []
        for r in _records(ordered):
            kind = t[f"kind_{r.kind}"]
            zs = (
                _fmt(r.z, 1, "", lang)
                if r.kind != "jump"
                else f"Δ {_fmt(r.jump_deg, 0, '°', lang)}"
            )
            rows.append(
                f"<tr><td>{_esc(_fmt(r.time_s, 2, '', lang))}</td><td>{r.frame}</td>"
                f"<td class='l'>{_side_name(r.side, t, affected)}</td><td class='l'>{_esc(t.get(r.joint, r.joint))}</td>"
                f"<td class='bad'>{_esc(_fmt(r.value, 0, '°', lang))}</td>"
                f"<td>{_esc(_fmt(r.expected, 0, '°', lang))}</td><td>{_esc(zs)}</td><td class='l'>{_esc(kind)}</td></tr>"
            )
        attention_html += (
            f"<table><thead><tr><th>{_esc(t['time'])}</th><th>{_esc(t['frame'])}</th>"
            f"<th>{_esc(t['side'])}</th><th>{_esc(t['joint'])}</th><th>{_esc(t['angle'])}</th>"
            f"<th>{_esc(t['expected'])}</th><th>z</th><th>{_esc(t['kind'])}</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )

    cycles_html = ""
    if not cycles.empty:
        rows = []
        for r in _records(cycles):
            rows.append(
                f"<tr><td>{_esc(r.camera)}</td><td class='l'>{_side_name(str(r.side), t, affected)}</td>"
                f"<td>{_esc(_fmt(r.hs_time_s, 2, '', lang))}</td>"
                f"<td>{_esc(_fmt(r.stride_time_s, 2, ' s', lang))}</td>"
                f"<td>{_esc(_fmt(r.stance_pct, 0, '%', lang))}</td>"
                f"<td>{_esc(_fmt(r.swing_pct, 0, '%', lang))}</td>"
                f"<td>{_esc(_fmt(r.double_support_pct, 0, '%', lang))}</td>"
                f"<td>{_esc(_fmt(r.cadence_steps_min, 0, '', lang))}</td></tr>"
            )
        cycles_html = (
            f"<h2>{_esc(t['cycles'])}</h2><table><thead><tr><th>cam</th><th>{_esc(t['side'])}</th>"
            f"<th>{_esc(t['time'])}</th><th>{_esc(t['stride_time_s'])}</th>"
            f"<th>{_esc(t['stance_pct'])}</th><th>{_esc(t['swing_pct'])}</th>"
            f"<th>{_esc(t['double_support_pct'])}</th><th>{_esc(t['cadence_steps_min'])}</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )

    refs = "".join(f"<li>{_esc(r)}</li>" for r in REFERENCE_TEXT)
    page2 = f"""
<section class="page">
  <header><div><h1>{_esc(t["technical"])}</h1><div class="sub">{_esc(info.participant)} · {_esc(info.recording or analysis_dir.parent.name)}</div></div>
  <div class="inst">{_esc(info.institution)}</div></header>
  {rom_html}
  {stats_html}
  {frontal_html}
  {attention_html}
  {cycles_html}
  <h2>{_esc(t["method"])}</h2><p class="note">{_esc(t["method_text"].format(cam=camera or "–"))}</p>
  <h2>{_esc(t["references"])}</h2><ol class="note">{refs}</ol>
  <div class="disclaimer">{_esc(t["disclaimer"])}</div>
</section>"""

    appendix = (
        _appendix_html(angles, times, extremes, t, affected, lang) if not angles.empty else ""
    )
    return f"""<!doctype html>
<html lang="{"pt-BR" if lang == "pt" else "en"}"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(t["title"])} — {_esc(info.participant or analysis_dir.parent.name)}</title>
<style>{CSS}</style></head><body>
<div class="toolbar"><button onclick="window.print()">{_esc(t["print"])}</button></div>
{page1}
{page2}
{appendix}
</body></html>
"""


def _appendix_html(angles, times, extremes, t, affected, lang) -> str:
    """Frame-by-frame tables per side (screen only), extreme values highlighted."""
    flagged = set()
    if not extremes.empty:
        flagged = {(r.column, r.frame) for r in _records(extremes)}
    parts = []
    for side in SIDES:
        cols = [c for c in angles.columns if c != "frame" and split_column(c)[0] == side]
        if not cols:
            continue
        head = "".join(f"<th>{_esc(t.get(split_column(c)[1], c))}</th>" for c in cols)
        rows = []
        for i, r in enumerate(_records(angles)):
            frame = int(r.frame)
            cells = []
            any_value = False
            for c in cols:
                v = getattr(r, c)
                if v is None or not np.isfinite(v):
                    cells.append("<td></td>")
                    continue
                any_value = True
                cls = " class='bad'" if (c, frame) in flagged else ""
                cells.append(f"<td{cls}>{_esc(_fmt(v, 1, '', lang))}</td>")
            if any_value:
                tt = _fmt(times[i], 2, "", lang) if i < len(times) else ""
                rows.append(f"<tr><td>{frame}</td><td>{_esc(tt)}</td>{''.join(cells)}</tr>")
        parts.append(
            f"<details><summary><b>{_side_name(side, t, affected)}</b></summary><div class='scroll'>"
            f"<table><thead><tr><th>{_esc(t['frame'])}</th><th>{_esc(t['time'])}</th>{head}</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div></details>"
        )
    return (
        f"<section class='page screen-only'><h2>{_esc(t['appendix'])}</h2>"
        f"<div class='note'>{_esc(t['appendix_note'])}</div>{''.join(parts)}</section>"
    )


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError):
        return pd.DataFrame()


# ── PDF ──────────────────────────────────────────────────────────────────────


def find_browser() -> str | None:
    """A Chromium-based browser able to print to PDF headless (Chrome, Chromium, Edge)."""
    names = ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "msedge"]
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    candidates = []
    if platform.system() == "Windows":
        for base in (os.environ.get("PROGRAMFILES", ""), os.environ.get("PROGRAMFILES(X86)", "")):
            candidates += [
                Path(base) / "Google/Chrome/Application/chrome.exe",
                Path(base) / "Microsoft/Edge/Application/msedge.exe",
            ]
    elif platform.system() == "Darwin":
        candidates += [
            Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
            Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
            Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
        ]
    return next((str(c) for c in candidates if c.is_file()), None)


def html_to_pdf(html_path: Path, pdf_path: Path, timeout: int = 120) -> bool:
    """Print ``html_path`` to ``pdf_path`` with a headless browser; False if not possible."""
    browser = find_browser()
    if browser is None:
        return False
    with tempfile.TemporaryDirectory(prefix="vaila_pdf_") as profile:
        cmd = [
            browser,
            "--headless",
            "--disable-gpu",
            "--no-first-run",
            f"--user-data-dir={profile}",
            "--no-pdf-header-footer",
            "--print-to-pdf-no-header",
            f"--print-to-pdf={pdf_path}",
            Path(html_path).resolve().as_uri(),
        ]
        with contextlib.suppress(Exception):
            subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)
    return pdf_path.is_file() and pdf_path.stat().st_size > 0


def write_reports(
    analysis_dir: Path,
    info: ReportInfo,
    langs: tuple[str, ...] = ("pt", "en"),
    pdf: bool = True,
) -> dict[str, tuple[Path, Path | None]]:
    """Write the report in each language; ``{lang: (html, pdf or None)}``."""
    analysis_dir = Path(analysis_dir)
    out: dict[str, tuple[Path, Path | None]] = {}
    for lang in langs:
        html_path = analysis_dir / f"{REPORT_NAMES[lang]}.html"
        html_path.write_text(build_report_html(analysis_dir, info, lang), encoding="utf-8")
        pdf_path: Path | None = analysis_dir / f"{REPORT_NAMES[lang]}.pdf"
        if not (pdf and html_to_pdf(html_path, pdf_path)):
            pdf_path = None
        out[lang] = (html_path, pdf_path)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Printable gait report (PT/EN).")
    parser.add_argument("--analysis", required=True, help="gait_analysis_* folder")
    parser.add_argument("--lang", nargs="+", default=["pt", "en"], choices=["pt", "en"])
    parser.add_argument("--participant", default="")
    parser.add_argument("--age", default="")
    parser.add_argument("--side", default="", choices=["", "right", "left", "both"])
    parser.add_argument("--therapist", default="")
    parser.add_argument("--institution", default="")
    parser.add_argument("--no-pdf", action="store_true")
    args = parser.parse_args(argv)
    info = ReportInfo(args.participant, args.age, args.side, args.therapist, args.institution)
    for lang, (html_path, pdf_path) in write_reports(
        Path(args.analysis), info, tuple(args.lang), not args.no_pdf
    ).items():
        print(f"{lang}: {html_path}" + (f" | {pdf_path}" if pdf_path else " (no PDF)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
