# gaze-mouse

Move the macOS mouse cursor with your **head pose** and click by **dwelling**.
Webcam → MediaPipe FaceLandmarker → head yaw/pitch → One-Euro-smoothed cursor →
Quartz cursor warp + synthetic click.

It uses **head pose, not eye gaze**: head pose is far smoother frame-to-frame and
needs no per-user calibration grid — just a one-key neutral-pose recenter. Eye
gaze from a webcam is jittery and would demand a calibration routine.

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
```

On first run it downloads `face_landmarker.task` into the project dir.
Logs go to stdout and the rotating file `gaze_mouse.log`.

## Keys (preview window)

- `q` — quit
- `c` — recenter / recalibrate the neutral head pose
- `space` — toggle cursor control on/off (starts **OFF** for safety)
- `r` — toggle webcam recording on/off (writes to `debug/`, local file only)

Press `space` only once you see the HUD tracking your head correctly.

## Debug artifacts

Debug captures land in `debug/` (created on demand):

- `session-<timestamp>.mp4` — the annotated preview (camera + HUD + face dot +
  screen minimap). Start with `--record` or the `r` key; works in `--no-window`
  too. **Privacy:** this webcam video is a **local file only** — nothing is
  uploaded anywhere.
- `trace-<timestamp>.jsonl` — one JSON object per frame (`--trace`) with pose,
  target/cursor coords, dwell progress, control state, and click events.

The on-frame **screen minimap** (top-right) maps the full display, showing the
raw target position (hollow circle) and the smoothed cursor (filled dot with a
dwell ring).

Smoke-test the debug helpers without a camera:

```sh
.venv/bin/python tests/smoke_debug.py
```

## Tuning knobs (top of `gaze_mouse.py`)

- `GAIN_X`, `GAIN_Y` — how far a given head turn/tilt moves the cursor.
- `DWELL_TIME`, `DWELL_RADIUS` — how long/still you must hold to fire a click.
- `MIN_CUTOFF`, `BETA` — One-Euro filter: lower jitter vs. lower lag.
- `CAMERA_INDEX` — webcam to use.
