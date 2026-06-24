# gaze-mouse

Move the macOS mouse cursor with your **head pose** and click by **dwelling**.
Webcam → MediaPipe FaceLandmarker → head yaw/pitch → One-Euro-smoothed cursor →
Quartz cursor warp + synthetic click.

Two tracking modes, switchable at runtime (`m`) or at launch (`--mode`):

- **head** (default) — head yaw/pitch. Smooth frame-to-frame, no per-user
  calibration grid; just a one-key neutral-pose recenter.
- **eye** — real iris gaze (appearance-based iris offset inside each eye). This
  is genuine eye tracking, but it is **jittery** and **low-resolution** (the
  webcam iris is only a handful of pixels), so the cursor wanders more and it
  may need a proper calibration routine later. Expect it to feel rougher than
  head mode — that is inherent to webcam iris tracking, not a bug.

A thin **face bounding box** is always drawn in the preview. In eye mode the
preview also marks the two iris centers and the four eye corners so you can see
exactly what is being tracked.

## Permissions (macOS)

Grant both to the terminal/Python that runs this, in
**System Settings → Privacy & Security**:

- **Camera** — to read the webcam.
- **Accessibility** — cursor warping and synthetic clicks require Accessibility
  to be granted to the app launching Python (Terminal, iTerm, VS Code, etc.).
  Without it the preview works but the cursor will not move.

## Run

```sh
.venv/bin/python gaze_mouse.py            # preview window, control starts OFF
.venv/bin/python gaze_mouse.py --debug    # DEBUG logging
.venv/bin/python gaze_mouse.py --no-window # headless, no preview window
.venv/bin/python gaze_mouse.py --record   # record annotated preview to debug/*.mp4
.venv/bin/python gaze_mouse.py --trace    # per-frame JSONL trace to debug/*.jsonl
.venv/bin/python gaze_mouse.py --mode eye # start in iris-gaze mode (default: head)
```

On first run it downloads `face_landmarker.task` into the project dir.
Logs go to stdout and the rotating file `gaze_mouse.log`.

## Keys (preview window)

- `q` — quit
- `c` — recenter / recalibrate the neutral for the **active mode**
- `space` — toggle cursor control on/off (starts **OFF** for safety)
- `r` — toggle webcam recording on/off (writes to `debug/`, local file only)
- `m` — toggle tracking mode (head ↔ eye)

Press `space` only once you see the HUD tracking correctly. After switching
mode with `m`, press `c` to recenter the neutral for that mode.

## Debug artifacts

Debug captures land in `debug/` (created on demand):

- `session-<timestamp>.mp4` — the annotated preview (camera + HUD + face dot +
  screen minimap). Start with `--record` or the `r` key; works in `--no-window`
  too. **Privacy:** this webcam video is a **local file only** — nothing is
  uploaded anywhere.
- `trace-<timestamp>.jsonl` — one JSON object per frame (`--trace`) with the
  active `mode`, pose (`yaw`/`pitch`), raw iris gaze (`gaze_x`/`gaze_y`, present
  in both modes for comparison), target/cursor coords, dwell progress, control
  state, and click events.

The on-frame **screen minimap** (top-right) maps the full display, showing the
raw target position (hollow circle) and the smoothed cursor (filled dot with a
dwell ring).

Smoke-test the pure helpers without a camera:

```sh
.venv/bin/python tests/smoke_debug.py   # recorder / tracer / minimap
.venv/bin/python tests/smoke_eye.py     # eye-gaze + face-bbox math
```

## Tuning knobs (top of `gaze_mouse.py`)

- `GAIN_X`, `GAIN_Y` — how far a given head turn/tilt moves the cursor.
- `EYE_GAIN_X`, `EYE_GAIN_Y` — same, for iris-gaze mode (larger, since the iris
  offset range is small). Raise to reach screen edges with less eye travel.
- `DWELL_TIME`, `DWELL_RADIUS` — how long/still you must hold to fire a click.
- `MIN_CUTOFF`, `BETA` — One-Euro filter: lower jitter vs. lower lag.
- `CAMERA_INDEX` — webcam to use.
