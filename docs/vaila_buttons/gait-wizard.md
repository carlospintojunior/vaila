# Marcha Fácil (Gait Wizard)

**Button:** `C_B_r6_c3` · **Method:** `gait_wizard` · **Module:** `vaila/gait_wizard.py`

Step-by-step wizard (screens in Portuguese) for users with no computer
background: **Captura** (*Gravar agora* opens [Record Cameras](record-cameras.md) on the
recordings folder, or pick an existing session from a list with previews;
*Nova pasta…* creates a folder inside the recordings folder) → **Corte** ([Cut Cameras](cut-cameras.md)
opens with the session loaded and the wizard continues by itself) →
**Processamento** (tick the side-view cameras, click *Analisar*: MediaPipe +
gait events/phases + MP Angles) → **Resultados** (cadence, stance/swing,
stride time, symmetry, joint angles over the gait cycle, plain advice on
recording problems, full report).

Full help: [`vaila/help/gait_wizard.md`](../../vaila/help/gait_wizard.md).
