# multicam_cut

## 📋 Module Information

- **Category:** Tools
- **File:** `vaila/multicam_cut.py`
- **Version:** 0.4.5
- **GUI Interface:** ✅ Yes
- **CLI Interface:** ⚠️ Partial (`uv run python vaila/multicam_cut.py [session_dir]` opens the GUI on that folder)

## 📖 Description

Cuts every video of a multi-camera session **at once, keeping them synchronized**.
All cameras are shown side by side in one mosaic window (the same kind of mosaic
as the **Record Cameras** live preview), driven by one shared timeline with three
handles:

- **green ▲** — Start of the cut
- **red ▲** — End of the cut
- **blue cursor** — the instant shown in the mosaic

The timeline is in **time (seconds), not frames**. Webcams in one session often
run at different — and variable — frame rates (e.g. a 1280x720 camera limited by
USB bandwidth to ~8 fps next to 640x480 cameras at 30 fps). Each camera maps
timeline time to its own frame using the **real packet timestamps** read from the
file (ffprobe), so a cut keeps every camera on the same instant, each with its
own number of frames.

Cameras recorded by parallel `ffmpeg` processes never start at exactly the same
instant, so each camera has an **offset in milliseconds**:

```
camera time = timeline time + offset
```

Each arrow click on a camera's offset moves it by **one frame of that camera**.
Step it until a sharp event (a clap, a flash, a foot strike) lands on the same
instant in every tile.

### Key Features

- **Mosaic view** of every camera, each tile showing its own frame number and fps; tiles turn green inside Start/End
- **Range timeline**: drag Start/End handles (the mosaic follows the handle you drag) and the cursor; click anywhere to jump
- **Common range highlight**: the part of the bar where every camera has video is shaded blue; Start/End stay inside it
- **Mixed / variable frame rates** handled via real frame timestamps
- **Frame-accurate cut** (hardware H.264 when available, CPU `libx264` fallback) that **keeps each frame's real timestamp**, so a camera whose file declares a wrong frame rate still plays back in real time
- **Background cutting** with Cancel; prints the equivalent `ffmpeg` command per camera (GUI→CLI mirror)
- **Reproducible output**: `multicam_cut.toml` records times, offsets and each camera's frame range
- **Remembers the last session** (`~/.vaila/vaila_config.toml`): the folder field comes pre-filled (press Enter to reload it) and **Browse...** opens in the folder that holds your sessions (Record Cameras' output directory before the first cut)

## 🚀 Usage

### GUI Mode (from *vailá*)

Select **Cut Cameras** in the *vailá* toolbox (Video and Image tools, next to **Record Cameras**).

1. **Browse...** to a session folder (e.g. `trial01_20260926_120000/` written by Record Cameras). It needs at least 2 videos.
2. If needed, align cameras: find a sharp event and step each camera's **Offset (ms)** until it lines up in every tile.
3. Drag the **green ▲** to where the cut starts and the **red ▲** to where it ends (or put the cursor there and press **i** / **o**).
4. Click **Cut all cameras**.

### Keys (control window or mosaic window)

| Key | Action |
| --- | --- |
| `Space` | Play / pause (real time) |
| `a` / `d` | Step back / forward one frame (of the fastest camera) |
| `j` / `l` | Step back / forward 10 frames |
| `i` / `o` | Set Start / End at the cursor |

### Standalone

```bash
uv run python vaila/multicam_cut.py /path/to/session_dir
```

## 📤 Output

```
<session_dir>/processed_multicam_cut_YYYYMMDD_HHMMSS/
    cam1_trial01_frame_<a>_to_<b>.mp4
    cam2_trial01_frame_<a>_to_<b>.mp4
    ...
    multicam_cut.toml
```

Frame numbers in filenames and the TOML are **1-based, inclusive** (same
convention as Cut Video and Make Sync file). All outputs cover the same time
span; cameras at different frame rates have different frame counts.

## 📋 Requirements

- **FFmpeg + ffprobe** (resolved via `vaila.ffmpeg_utils`)
- **OpenCV**
- Python 3.12 with Tkinter

## ⚠️ Known limitations (v1)

- Alignment is manual (no automatic clap/flash detection yet).
- Output is re-encoded (H.264) for frame accuracy, not stream-copied.
- A camera recording at a much lower frame rate than the others (e.g. ~8 fps)
  limits the timing precision of that view; lower its resolution in Record
  Cameras if it's USB-bandwidth-limited.

---

📅 **Updated:** 26/09/2026
🔗 **Part of *vailá* - Multimodal Toolbox**
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
