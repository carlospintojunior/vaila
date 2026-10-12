# gait_wizard

## 📋 Module Information

- **Category:** Tools
- **File:** `vaila/gait_wizard.py`
- **Version:** 0.4.5
- **GUI Interface:** ✅ Yes (button **Marcha Fácil**, `C_B_r6_c3`)
- **CLI Interface:** ⚠️ Partial (`uv run python vaila/gait_wizard.py [sessions_root]` opens the GUI on that folder)
- **Language:** the wizard's screens are in **Portuguese** (first version)

## 📖 Description

**Marcha Fácil** is a step-by-step wizard for people with no computer
background. It takes a **Record Cameras** session to gait results with one big
button per screen and no file dialogs or settings to understand:

1. **Captura** — recording from inside the wizard is planned ("em breve"). For
   now the user picks an existing session from a list showing date, time,
   number of cameras, status (Nova / Cortada / Analisada) and a preview image
   of every camera.
2. **Corte** — **Cut Cameras** opens with the session already loaded; when the
   cut finishes, the wizard moves on by itself. The user can also reuse a
   previous cut or analyse the whole video.
3. **Processamento** — the user ticks the cameras that see the person **from
   the side** (remembered for next time), optionally types a participant
   name/code and clicks **Analisar**. A checklist shows each stage in plain
   language with the elapsed time:
   MediaPipe pose → gait events and phases (`vaila/gait_auto.py` logic:
   Zeni heel-strike / toe-off) → joint angles (**MP Angles**) → report.
4. **Resultados** — big numbers (cadence, stance %, swing %, stride time,
   left/right symmetry) with one-line explanations, hip / knee / ankle angles
   over the gait cycle (mean ± SD per side, toe-off marked), and plain advice
   when the recording has problems (e.g. "the feet leave the bottom of the
   image in 89 % of the time — move the camera back or tilt it down"). Buttons
   open the full HTML report or the results folder.

Progress is saved in `<session>/assistente_marcha.json`: closing the program
and opening the same session again offers its results directly.

## 📁 Output

Inside the analysed folder (the cut folder, or the session if not cut):

```
gait_analysis_YYYYMMDD_HHMMSS/
    gait_report.html          full report (+ joint-angle section)
    gait_events.csv           every heel-strike / toe-off
    gait_cycles.csv           one row per complete gait cycle
    gait_summary.csv          means / SD per camera and side
    joint_angles_cycle.csv    angle mean / SD per % of the gait cycle
    joint_angles_cycle.png    hip / knee / ankle over the gait cycle
    angles/                   MP Angles CSVs per camera (_rel, _abs_180, _abs_360)
    mediapipe/                MediaPipe outputs (CSVs, annotated video)
```

## 📐 Angles

MP Angles relative angles are unsigned (0–180°). The knee is shown as
**flexion** (180° − angle, 0° = straight leg); hip (trunk–thigh) and ankle
(shank–foot) are shown as computed by MP Angles. In a side view the leg
farther from the camera is partly hidden, so its curve is less reliable.

## ⚠️ Known Limitations (v1)

- Only temporal parameters and 2D sagittal angles; no calibration, no
  distances or speeds.
- Results need lateral cameras that show the **whole body including the feet**
  and several crossings (aim for ≥ 10 gait cycles).
- MediaPipe runs on the CPU: a few minutes per short session on a laptop.
- Closing the wizard during processing abandons that analysis.
- Screens are in Portuguese only; the full HTML report is still in English.

---

📅 **Updated:** 12/10/2026
🔗 **Part of _vailá_ - Multimodal Toolbox**
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
