# gait_report

## 📋 Module Information

- **Category:** Tools
- **File:** `vaila/gait_report.py`
- **Version:** 0.4.8
- **GUI Interface:** ✅ Yes (built automatically by **Marcha Fácil**; button "Relatório para imprimir" on its results screen)
- **CLI Interface:** ✅ Yes (`uv run python vaila/gait_report.py --analysis <gait_analysis_dir>`)
- **Languages:** Portuguese and English

## 📖 Description

Builds a **printable gait report** (HTML + PDF) from a Marcha Fácil / `gait_auto`
analysis folder.

**Page 1, for the patient and the therapist**
- Identification: participant, age, affected side, therapist, institution, recording.
- A plain-language summary. Each parameter is compared with its reference range,
  with a short explanation when it is outside the range (e.g. "the foot stays on
  the floor longer (seeking stability)").
- Temporal parameters: cadence, stance, swing, double support, stride time,
  left/right difference and peak knee flexion in swing. Values outside the
  reference range are shown in bold and colour.
- Hip, knee and ankle angles over the gait cycle (mean ± SD per side).
- Four frames of the MP Angles video: heel strike, mid-stance, toe-off and
  mid-swing.

**Page 2, technical (for the therapist)**
- Range of motion per joint and side, compared with references.
- **MP Angles statistics split into Right / Left / Neck tables** instead of one
  15-column table. Each table gives the mean, SD, minimum and maximum with the
  time they happen, range, between-cycle variability and the number of extreme
  points. The affected side is tagged.
- **Points of extreme oscillation**, in bold red, with the time to check in the
  video:
  - *off pattern*: the angle departs from the mean of the person's **other**
    cycles, at the same % of the cycle, by more than 2.5 between-cycle SDs
    (and at least 10°). This needs at least 3 cycles of that side. The cycle
    under test is left out of its own reference, so an odd cycle cannot hide
    itself. A z-score over the whole walk is **not** used, because it flags the
    normal ~60° knee flexion of swing.
  - *abrupt change*: a frame-to-frame jump larger than max(15°, 6 × the median
    jump). This can be instability, compensation or a tracking error.
- Gait cycles, method and references.

**Screen-only appendix:** the frame-by-frame angle tables, split by side, with
the extreme values highlighted. This section is not printed.

The PDF is printed by a Chromium-based browser in headless mode (Chrome,
Chromium or Edge). Without one, the HTML has a "Print / save PDF" button.

## 📁 Output (inside the analysis folder)

```
relatorio_marcha.html / .pdf          Portuguese
gait_report_print_en.html / .pdf      English
```

## 📐 Reference ranges

These are approximate ranges around the mean values of healthy adults walking
at a self-selected speed (Perry & Burnfield 2010; symmetry index: Robinson et
al. 1987):

| Parameter | Range |
| --- | --- |
| Cadence | 100–125 steps/min |
| Stance / swing | 57–63 % / 37–43 % of the cycle |
| Double support | 16–24 % of the cycle |
| Stride time | 0.95–1.25 s |
| Left/right difference (stance) | ≤ 10 % |
| Peak knee flexion in swing | 55–65° |
| Range of motion: hip / knee / ankle | 35–45° / 55–70° / 25–35° |

Hip and ankle come from MP Angles' unsigned 2D angles (trunk–thigh,
shank–foot), so their range of motion is an approximation.

## 💻 CLI

```bash
uv run python vaila/gait_report.py --analysis <gait_analysis_dir> \
    [--lang pt en] [--participant ID] [--age 54] [--side right|left|both] \
    [--therapist NAME] [--institution NAME] [--no-pdf]
```

## ⚠️ Limitations

- The report is generated to support an assessment. It does not replace
  clinical assessment.
- With fewer than 3 cycles per side, only abrupt changes are checked, and the
  report says so.
- 2D angles in the camera plane, with no calibration.

---

📅 **Updated:** 12/10/2026
🔗 **Part of _vailá_ - Multimodal Toolbox**
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
