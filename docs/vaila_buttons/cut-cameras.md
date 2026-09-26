# Cut Cameras

**Button:** `C_B_r6_c2` · **Method:** `cut_cameras` · **Module:** `vaila/multicam_cut.py`

Cuts every video of a multi-camera session (e.g. a folder written by
[Record Cameras](record-cameras.md)) at once, keeping them synchronized. All
cameras are shown in one mosaic driven by a time-based timeline (green Start,
red End, blue cursor); per-camera offsets (ms) align them, and one Start/End
pair cuts them all — even when cameras run at different frame rates.

Full help: [`vaila/help/multicam_cut.md`](../../vaila/help/multicam_cut.md).
