# gaze-mouse

Move the macOS mouse cursor with your **head pose** and click by **dwelling**.
Webcam → MediaPipe FaceLandmarker → head yaw/pitch → One-Euro-smoothed cursor →
Quartz cursor warp + synthetic click.

Two tracking modes, switchable at runtime (`m`) or at launch (`--mode`):

- **head** (default) — head yaw/pitch. Smooth frame-to-frame, no per-user
  calibration grid; just a one-key neutral-pose recenter.
- **eye** — real iris gaze (appearance-based iris offset inside each eye). This
  is genuine eye tracking, but it is **jittery** and **low-resolution** (the
  webcam iris is only a handful of pixels), so the cursor wanders more. Expect
  it to feel rougher than head mode — that is inherent to webcam iris tracking,
  not a bug. **Calibration is what makes eye mode actually land on targets**:
  the raw iris offset is fitted to your screen with a per-user model (see
  below), replacing the flat gain mapping.

## Eye-gaze calibration

The raw iris offset alone does not reach screen corners reliably. A one-time
**9-point calibration** fits a per-user iris→pixel model that eye mode then uses
instead of the flat gain:

```sh
.venv/bin/python gaze_mouse.py --calibrate   # calibrate first, then run
.venv/bin/python gaze_mouse.py --no-calib    # ignore the saved profile (use gain)
```

Press `k` in the preview window to (re)calibrate at any time. A fullscreen
window shows nine dots (a 3×3 grid at ~10% margins); look at each red dot while
a green ring fills, `space` skips a bad point, `ESC` aborts (keeping the
previous mapping). The fitted profile is saved to
**`debug/eye_calibration.json`** and loaded automatically on the next run; the
HUD shows `calib on/off` in eye mode and the startup log line reports `calib=`.

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
.venv/bin/python gaze_mouse.py               # preview window, control starts OFF
.venv/bin/python gaze_mouse.py --debug       # DEBUG logging
.venv/bin/python gaze_mouse.py --no-window   # headless, no preview window
.venv/bin/python gaze_mouse.py --record      # also record recording.mp4 in the session
.venv/bin/python gaze_mouse.py --no-snapshots # disable periodic + on-click snapshots
.venv/bin/python gaze_mouse.py --snap-interval 5 # seconds between periodic snapshots
.venv/bin/python gaze_mouse.py --mode eye    # start in iris-gaze mode (default: head)
.venv/bin/python gaze_mouse.py --calibrate   # run 9-point eye calibration first
.venv/bin/python gaze_mouse.py --no-calib    # ignore saved eye calibration profile
```

On first run it downloads `face_landmarker.task` into the project dir.
Logs go to stdout and to this run's `debug/session-<timestamp>/session.log`.
`--trace` is a deprecated no-op: the per-frame trace is always written now.

## Keys (preview window)

- `q` — quit
- `c` — recenter / recalibrate the neutral for the **active mode**
- `space` — toggle cursor control on/off (starts **OFF** for safety)
- `r` — toggle webcam recording on/off (writes to `debug/`, local file only)
- `m` — toggle tracking mode (head ↔ eye)
- `k` — run the 9-point eye-gaze calibration (recalibrate anytime)

Press `space` only once you see the HUD tracking correctly. After switching
mode with `m`, press `c` to recenter the neutral for that mode (head mode) or
`k` to calibrate (eye mode).

## Debug artifacts

Every run gets **one folder per session** under `debug/`. Logs, the trace and
snapshots are **per-session and on by default**; `--record` now nests inside the
same folder:

```
debug/session-<YYYYMMDD-HHMMSS>/
  session.log      this run's full log, isolated (no shared rotating log)
  trace.jsonl      per-frame JSONL trace — ALWAYS ON
  recording.mp4    annotated preview video — only with --record / key 'r'
  frames/          annotated camera snapshots (iris dots + minimap visible)
    frame-<seq>-t<elapsed>s.png   periodic, every --snap-interval seconds
    click-<seq>-t<elapsed>s.png   one per dwell-click
  meta.json        written on close (graceful shutdown or Ctrl-C)
```

- `session.log` — the full log for this run only. The old shared rotating
  `gaze_mouse.log` is gone; nothing is written outside the session folder.
- `trace.jsonl` — one JSON object per frame (always written) with the active
  `mode`, pose (`yaw`/`pitch`), raw iris gaze (`gaze_x`/`gaze_y`, present in both
  modes for comparison), target/cursor coords, dwell progress, control state,
  and click events.
- `frames/` — annotated camera snapshots (same overlays as the preview: iris
  dots, face box, HUD, screen minimap). Saved **periodically** every
  `--snap-interval` seconds (default `3.0`) **plus one immediately on every
  dwell-click** (`click-` prefix). On by default; disable with `--no-snapshots`.
- `recording.mp4` — the annotated preview video. Start with `--record` or the
  `r` key; works in `--no-window` too.
- `meta.json` — written on graceful shutdown and on Ctrl-C: start/end
  timestamps, duration, total frames, mean fps, screen size, modes used,
  calibration active + point count, dwell-click count, snapshot count, and the
  flags used.

**Privacy:** the webcam video and snapshots are **local files only** — nothing
is uploaded anywhere.

The calibration profile is **not** per-session — it stays shared at the
`debug/` root:

- `debug/eye_calibration.json` — the fitted per-user iris→pixel model (two
  polynomial coefficient vectors, feature order, screen size, point count,
  timestamp). Written by calibration (`k` / `--calibrate`) and auto-loaded on
  the next run unless `--no-calib` is passed.

The on-frame **screen minimap** (top-right) maps the full display, showing the
raw target position (hollow circle) and the smoothed cursor (filled dot with a
dwell ring).

Smoke-test the pure helpers without a camera:

```sh
.venv/bin/python tests/smoke_debug.py    # recorder / tracer / minimap
.venv/bin/python tests/smoke_eye.py      # eye-gaze + face-bbox math
.venv/bin/python tests/smoke_calib.py    # calibration fit/predict math
.venv/bin/python tests/smoke_session.py  # per-session folder + snapshots + meta
```

## Tuning knobs (top of `gaze_mouse.py`)

- `GAIN_X`, `GAIN_Y` — how far a given head turn/tilt moves the cursor.
- `EYE_GAIN_X`, `EYE_GAIN_Y` — same, for iris-gaze mode **when uncalibrated**
  (larger, since the iris offset range is small). Raise to reach screen edges
  with less eye travel. Once you calibrate (`k` / `--calibrate`) the fitted
  model is used instead and these gains no longer apply.
- `CALIB_MARGIN`, `CALIB_SAMPLES`, `CALIB_MIN_POINTS` — calibration grid inset,
  samples collected per target, and minimum good points to accept a fit.
- `DWELL_TIME`, `DWELL_RADIUS` — how long/still you must hold to fire a click.
- `MIN_CUTOFF`, `BETA` — One-Euro filter: lower jitter vs. lower lag.
- `CAMERA_INDEX` — webcam to use.
