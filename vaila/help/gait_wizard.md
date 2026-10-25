# gait_wizard

## 📋 Module Information

- **Category:** Tools
- **File:** `vaila/gait_wizard.py`
- **Version:** 0.4.9
- **GUI Interface:** ✅ Yes (button **Marcha Fácil**, `C_B_r6_c3`)
- **CLI Interface:** ⚠️ Partial (`uv run python vaila/gait_wizard.py [sessions_root]` opens the GUI on that folder)
- **Language:** the wizard's screens are in **Portuguese** (first version)

## 📖 Description

**Marcha Fácil** is a step-by-step wizard for people with no computer
background. It takes a **Record Cameras** session to gait results with one big
button per screen and no file dialogs or settings to understand:

1. **Captura** — **Gravar agora** opens Record Cameras with the output folder already set to
   the recordings folder, and the list refreshes when it closes. **Nova
   pasta…** creates a new folder inside the recordings folder (the native Linux
   folder picker has no such button). Or the user picks an existing session from a list showing date, time,
   number of cameras, status (Nova / Cortada / Analisada) and a preview image
   of every camera.
2. **Corte** — **Cut Cameras** opens with the session already loaded; when the
   cut finishes, the wizard moves on by itself. The user can also reuse a
   previous cut or analyse the whole video.
3. **Processamento** — for each camera the user picks a role (remembered for
   next time): **Lateral** (sees the person from the side — goes through
   gait-event/cycle detection), **Coronal (de frente)** (frontal camera — only
   gets a skeleton + angles pass, no gait events: a front view has no
   sideways walking pass to detect — plus pelvic obliquity and knee
   valgus/varus, estimated from the frontal-view landmarks), or **Não usar**.
   The user also types a participant name/code and clicks **Analisar**. A
   checklist shows each stage in plain language with the elapsed time:
   MediaPipe pose → gait events and phases (`vaila/gait_auto.py` logic:
   Zeni heel-strike / toe-off, lateral cameras only) → joint angles and the
   **MP Angles video** (skeleton + angles drawn over the person, every
   analysed camera including the coronal one) → report.
4. **Resultados** — big numbers (cadence, stance %, swing %, stride time,
   left/right symmetry) with one-line explanations, hip / knee / ankle angles
   over the gait cycle (mean ± SD per side, toe-off marked), and plain advice
   when the recording has problems (e.g. "the feet leave the bottom of the
   image in 89 % of the time — move the camera back or tilt it down"). A big
   **"▶ Ver vídeo com ângulos"** button plays the skeleton + angles video of
   the camera that measured the walking (other cameras get smaller buttons);
   other buttons open the full HTML report (which also embeds the videos) or
   the results folder.

Progress is saved in `<session>/assistente_marcha.json`: closing the program
and opening the same session again offers its results directly.

## 🖨 Printable report (PT/EN)

The processing step also asks for age, affected side, therapist, institution
(both remembered) and the report language. At the end it writes a printable
report (HTML + PDF, Portuguese and English) with a plain-language summary,
parameters against reference ranges, angle statistics per side and the points
of extreme oscillation. The results screen opens it with **"Relatório para
imprimir"** (the other language is one click away). Details:
[gait_report](gait_report.md).

## 📁 Output

Inside the analysed folder (the cut folder, or the session if not cut):

```
gait_analysis_YYYYMMDD_HHMMSS/
    gait_report.html          full technical report (+ joint-angle section)
    relatorio_marcha.html/.pdf, gait_report_print_en.html/.pdf   printable report (PT / EN)
    gait_events.csv           every heel-strike / toe-off
    gait_cycles.csv           one row per complete gait cycle
    gait_summary.csv          means / SD per camera and side
    joint_angles_cycle.csv    angle mean / SD per % of the gait cycle
    joint_angles_cycle.png    hip / knee / ankle over the gait cycle
    angles/angles_<video>.mp4 skeleton + angles video per camera (H.264)
    angles/                   MP Angles CSVs per camera (_rel, _abs_180, _abs_360)
    angles/<video>_frontal.csv  coronal camera only: pelvic obliquity, knee valgus/varus per frame
    mediapipe/                MediaPipe outputs (CSVs, annotated video)
```

## 📐 Angles

Before MP Angles, the landmarks are low-passed at 6 Hz (gaps filled, frames
with nobody in view left empty), so the video and the angles are steady.
MP Angles' own filter is not used: it turns a whole column into NaN when the
person leaves the image even once.

MP Angles relative angles are unsigned (0–180°). The knee is shown as
**flexion** (180° − angle, 0° = straight leg); hip (trunk–thigh) and ankle
(shank–foot) are shown as computed by MP Angles. In a side view the leg
farther from the camera is partly hidden, so its curve is less reliable.

## 🧪 Test record

Screenshots and results of the first real-data test: [`docs/gait_wizard/README.md`](../../docs/gait_wizard/README.md).

## ⚠️ Known Limitations (v1)

- Only temporal parameters and 2D sagittal angles; no calibration, no
  distances or speeds.
- Results need lateral cameras that show the **whole body including the feet**
  and several crossings (aim for ≥ 10 gait cycles).
- MediaPipe runs on the CPU: a few minutes per short session on a laptop.
- Closing the wizard during processing abandons that analysis.
- Screens are in Portuguese only; the full HTML report is still in English.

---

📅 **Updated:** 17/10/2026
🔗 **Part of _vailá_ - Multimodal Toolbox**
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
