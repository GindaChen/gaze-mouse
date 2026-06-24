#!/usr/bin/env python3
"""Move the macOS mouse cursor with head pose, with dwell-to-click.

Pipeline:
    webcam (OpenCV)
      -> MediaPipe FaceLandmarker (VIDEO mode, facial transformation matrix)
      -> head yaw / pitch from the 4x4 transform's rotation part
      -> map to screen coordinates (gain + calibrated neutral pose)
      -> One-Euro filter smoothing
      -> Quartz.CGWarpMouseCursorPosition to move the cursor
      -> dwell-to-click via CGEventCreateMouseEvent / CGEventPost

Supports two tracking modes (switch at runtime with 'm' or via --mode):
    head  HEAD POSE -- smoother frame-to-frame, no per-user calibration grid,
          just a neutral-pose recenter.
    eye   IRIS GAZE -- real appearance-based gaze from the iris offset inside
          each eye; jitterier and lower-resolution, may need calibration.

Two gaze ENGINES (select at launch with --engine):
    builtin     (default) the MediaPipe FaceLandmarker pipeline above. MIT.
    eyegestures the third-party eyeGestures appearance-based tracker; its
                EyeGestures_v3.step() supplies the per-frame screen point,
                fed through the same smoothing / dwell / recording downstream.
                NOTE: eyeGestures is GPLv3; selecting this engine pulls in a
                GPLv3 dependency. It is lazy-imported only when requested so
                the builtin (MIT) path runs without it installed.

Keys (in the preview window):
    q      quit
    c      recenter / recalibrate the neutral for the active mode
    space  toggle cursor control on/off (starts OFF for safety)
    r      toggle webcam recording on/off (local file in debug/)
    m      toggle tracking mode (head <-> eye)
    k      run the 9-point eye-gaze calibration (recalibrate anytime)

Debug artifacts (written to debug/session-<timestamp>/):
    session.log      this run's full log, isolated to the session folder
    trace.jsonl      per-frame JSONL trace (always on)
    recording.mp4    annotated preview video (only with --record / key 'r')
    frames/          annotated camera snapshots (periodic + on dwell-click)
    meta.json        written on close (graceful shutdown or SIGINT)
The recorded webcam video and snapshots are local files only; nothing is
uploaded. The eye calibration profile stays shared at debug/eye_calibration.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import signal
import sys
import time

import cv2
import numpy as np

import Quartz

# --------------------------------------------------------------------------- #
# Config constants
# --------------------------------------------------------------------------- #
CAMERA_INDEX = 0

# Gaze engines. "builtin" is the MediaPipe FaceLandmarker pipeline (MIT).
# "eyegestures" delegates the per-frame screen point to the third-party
# EyeGestures_v3 tracker (GPLv3, lazy-imported only when selected).
ENGINE_BUILTIN = "builtin"
ENGINE_EYEGESTURES = "eyegestures"
ENGINES = (ENGINE_BUILTIN, ENGINE_EYEGESTURES)

# eyeGestures engine knobs. calibration_radius drives how tight the lib's
# acceptance ring gets; the NxM grid of normalized [0,1] points is what the
# tracker walks through during calibration (one fixation target at a time).
EYEGESTURES_CALIB_RADIUS = 1000
EYEGESTURES_GRID = (5, 5)   # columns x rows of normalized calibration targets

GAIN_X = 4.0          # screen-x sensitivity to yaw (radians -> fraction of width)
GAIN_Y = 4.0          # screen-y sensitivity to pitch (radians -> fraction of height)

# Eye-gaze sensitivity. The iris offset signal lives in a small ~[-1, 1] range
# with a much narrower usable span than head pose, so gains are larger. Tuning
# knobs: raise to reach the screen edges with less eye travel, lower to calm
# jitter (eye mode is inherently jitterier than head mode).
EYE_GAIN_X = 6.0      # screen-x sensitivity to horizontal iris offset
EYE_GAIN_Y = 6.0      # screen-y sensitivity to vertical iris offset

# Iris/eye landmark indices in the 478-point FaceLandmarker mesh.
# (outer_corner, inner_corner, upper_lid, lower_lid, iris_center)
LEFT_EYE_IDX = (33, 133, 159, 145, 468)
RIGHT_EYE_IDX = (263, 362, 386, 374, 473)

BBOX_MARGIN = 0.04    # face bounding box margin as a fraction of its size

DWELL_TIME = 0.8      # seconds the cursor must stay still to fire a click
DWELL_RADIUS = 30     # px radius that counts as "still"

# One-Euro filter defaults (tuned for ~30 fps head-pose signal)
MIN_CUTOFF = 1.0
BETA = 0.02
D_CUTOFF = 1.0

PERIODIC_LOG_INTERVAL = 2.0   # seconds between throttled debug stat lines

MODEL_FILENAME = "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# Debug recording / minimap overlay
DEBUG_DIR_NAME = "debug"
DEFAULT_FPS = 20.0            # fallback recording fps before fps is measured
TRACE_FLUSH_INTERVAL = 1.0    # seconds between trace file flushes

# Per-session debug output. Each run owns one debug/session-<ts>/ folder holding
# its log, trace, optional recording, snapshot frames and a meta.json summary.
SESSION_PREFIX = "session-"
SESSION_TS_FORMAT = "%Y%m%d-%H%M%S"
SESSION_LOG_NAME = "session.log"
TRACE_NAME = "trace.jsonl"
RECORDING_NAME = "recording.mp4"
FRAMES_DIR_NAME = "frames"
META_NAME = "meta.json"
DEFAULT_SNAP_INTERVAL = 3.0   # seconds between periodic camera snapshots

MINIMAP_W = 240              # px, on-frame screen minimap width
MINIMAP_H = 150             # px, on-frame screen minimap height
MINIMAP_MARGIN = 12         # px, gap from the frame edge

# Eye-gaze calibration: a fitted iris->pixel model replaces neutral+gain in eye
# mode. Targets are a 3x3 grid at CALIB_MARGIN from each screen edge.
CALIB_FILENAME = "eye_calibration.json"
CALIB_MARGIN = 0.10          # grid target inset from each screen edge (fraction)
CALIB_SETTLE = 0.8           # seconds to ignore at each target before sampling
CALIB_SAMPLES = 25           # raw gaze samples collected per target
CALIB_MIN_POINTS = 6         # minimum good targets required to fit a model
CALIB_FEATURE_ORDER = ["1", "gx", "gy", "gx2", "gy2", "gxgy"]

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_DIR, MODEL_FILENAME)
DEBUG_DIR = os.path.join(PROJECT_DIR, DEBUG_DIR_NAME)
CALIB_PATH = os.path.join(DEBUG_DIR, CALIB_FILENAME)

log = logging.getLogger("gaze_mouse")


# --------------------------------------------------------------------------- #
# Session: one debug folder per run
# --------------------------------------------------------------------------- #
class Session:
    """Owns one debug/session-<ts>/ folder and the paths inside it.

    Created at startup. The folder and its frames/ subdirectory are made
    eagerly so every artifact (log, trace, recording, snapshots, meta) lands
    in the same isolated place. The shared eye_calibration.json deliberately
    stays at the debug/ root and is NOT part of a session.
    """

    def __init__(self, debug_dir: str = DEBUG_DIR, stamp: str | None = None) -> None:
        self.stamp = stamp or time.strftime(SESSION_TS_FORMAT)
        self.dir = os.path.join(debug_dir, f"{SESSION_PREFIX}{self.stamp}")
        self.frames_dir = os.path.join(self.dir, FRAMES_DIR_NAME)
        os.makedirs(self.frames_dir, exist_ok=True)

        self.log_path = os.path.join(self.dir, SESSION_LOG_NAME)
        self.trace_path = os.path.join(self.dir, TRACE_NAME)
        self.recording_path = os.path.join(self.dir, RECORDING_NAME)
        self.meta_path = os.path.join(self.dir, META_NAME)


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def setup_logging(debug: bool, log_path: str) -> None:
    """Configure logging to both stdout and this session's log file."""
    level = logging.DEBUG if debug else logging.INFO
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S"
    )

    log.setLevel(level)
    log.handlers.clear()

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    stream.setLevel(level)
    log.addHandler(stream)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    file_handler.setLevel(level)
    log.addHandler(file_handler)


# --------------------------------------------------------------------------- #
# One-Euro filter
# --------------------------------------------------------------------------- #
class OneEuroFilter:
    """Low-latency low-jitter filter for a noisy 1-D signal.

    See Casiez et al., "1 Euro Filter" (CHI 2012). One instance per axis.
    """

    def __init__(
        self,
        min_cutoff: float = MIN_CUTOFF,
        beta: float = BETA,
        d_cutoff: float = D_CUTOFF,
    ) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_prev: float | None = None
        self._dx_prev = 0.0
        self._t_prev: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float, t: float) -> float:
        if self._t_prev is None or self._x_prev is None:
            self._t_prev = t
            self._x_prev = x
            return x

        dt = t - self._t_prev
        if dt <= 0.0:
            dt = 1e-3

        dx = (x - self._x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = t
        return x_hat


# --------------------------------------------------------------------------- #
# Quartz helpers
# --------------------------------------------------------------------------- #
def get_screen_size() -> tuple[float, float]:
    """Return (width, height) of the main display in points."""
    bounds = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())
    return float(bounds.size.width), float(bounds.size.height)


def warp_cursor(x: float, y: float) -> None:
    """Instantly move the cursor to (x, y) in global display coordinates."""
    Quartz.CGWarpMouseCursorPosition(Quartz.CGPointMake(x, y))


def left_click(x: float, y: float) -> None:
    """Synthesize a left mouse down + up at the given point."""
    point = Quartz.CGPointMake(x, y)
    down = Quartz.CGEventCreateMouseEvent(
        None, Quartz.kCGEventLeftMouseDown, point, Quartz.kCGMouseButtonLeft
    )
    up = Quartz.CGEventCreateMouseEvent(
        None, Quartz.kCGEventLeftMouseUp, point, Quartz.kCGMouseButtonLeft
    )
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, down)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, up)


# --------------------------------------------------------------------------- #
# Debug: minimap overlay
# --------------------------------------------------------------------------- #
def draw_minimap(
    frame: np.ndarray,
    screen_w: float,
    screen_h: float,
    target_xy: tuple[float, float] | None,
    cursor_xy: tuple[float, float] | None,
    dwell_progress: float,
) -> np.ndarray:
    """Draw a small screen map (top-right) showing cursor positions.

    The rectangle represents the full display, aspect-ratio preserved inside
    a MINIMAP_W x MINIMAP_H box. The raw TARGET position (where head pose
    points) is a hollow circle; the SMOOTHED/actual cursor is a filled dot
    with a dwell ring scaled/filled by dwell progress. Mutates and returns
    the frame so it is captured by the recorder.
    """
    h, w = frame.shape[:2]

    # Fit the display aspect ratio inside the minimap box.
    if screen_w <= 0 or screen_h <= 0:
        return frame
    scale = min(MINIMAP_W / screen_w, MINIMAP_H / screen_h)
    map_w = max(1, int(screen_w * scale))
    map_h = max(1, int(screen_h * scale))

    x0 = w - map_w - MINIMAP_MARGIN
    y0 = MINIMAP_MARGIN
    x1 = x0 + map_w
    y1 = y0 + map_h

    # Dim backing panel so markers stay readable over the camera image.
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (180, 180, 180), 1, cv2.LINE_AA)

    def to_map(pt: tuple[float, float]) -> tuple[int, int]:
        mx = x0 + int(clamp(pt[0] / screen_w, 0.0, 1.0) * (map_w - 1))
        my = y0 + int(clamp(pt[1] / screen_h, 0.0, 1.0) * (map_h - 1))
        return mx, my

    if target_xy is not None:
        tx, ty = to_map(target_xy)
        cv2.circle(frame, (tx, ty), 4, (0, 180, 255), 1, cv2.LINE_AA)

    if cursor_xy is not None:
        cx, cy = to_map(cursor_xy)
        # dwell ring: radius scaled by progress, filled arc shows progress.
        ring_r = 6 + int(dwell_progress * 10)
        cv2.circle(frame, (cx, cy), ring_r, (0, 200, 0), 1, cv2.LINE_AA)
        if dwell_progress > 0.0:
            cv2.ellipse(
                frame, (cx, cy), (ring_r, ring_r), -90, 0,
                int(360 * clamp(dwell_progress, 0.0, 1.0)),
                (0, 255, 0), 2, cv2.LINE_AA,
            )
        cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1, cv2.LINE_AA)
        coord = f"({int(cursor_xy[0])},{int(cursor_xy[1])})"
        cv2.putText(
            frame, coord, (x0 + 2, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
            (220, 220, 220), 1, cv2.LINE_AA,
        )

    cv2.putText(
        frame, "screen", (x0 + 2, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
        (180, 180, 180), 1, cv2.LINE_AA,
    )
    return frame


# --------------------------------------------------------------------------- #
# Debug: webcam recorder
# --------------------------------------------------------------------------- #
class Recorder:
    """Lazily-initialized cv2.VideoWriter for annotated preview frames.

    The writer is created on the first written frame so its size is known.
    Records to the session folder's recording.mp4 with the mp4v fourcc. The
    video is a LOCAL FILE ONLY; nothing leaves the machine.
    """

    def __init__(self, path: str, fps: float | None = None) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self._fps = fps
        self._writer: cv2.VideoWriter | None = None
        self._size: tuple[int, int] | None = None
        self.frame_count = 0
        self._start_time: float | None = None
        log.info(
            "Started webcam recording (local file only): %s", self.path
        )

    def write(self, frame: np.ndarray) -> None:
        if self._writer is None:
            h, w = frame.shape[:2]
            self._size = (w, h)
            fps = self._fps if self._fps and self._fps > 0 else DEFAULT_FPS
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(self.path, fourcc, float(fps), (w, h))
            self._start_time = time.time()
            log.info(
                "Recording writer ready: %dx%d @ %.1f fps", w, h, fps
            )
        self._writer.write(frame)
        self.frame_count += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        duration = (
            time.time() - self._start_time if self._start_time else 0.0
        )
        log.info(
            "Stopped webcam recording: %s (%d frames, %.1fs)",
            self.path, self.frame_count, duration,
        )


# --------------------------------------------------------------------------- #
# Debug: per-frame JSONL tracer
# --------------------------------------------------------------------------- #
class Tracer:
    """Newline-delimited JSON trace, one object per processed frame.

    Writes to the session folder's trace.jsonl and flushes periodically.
    """

    def __init__(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self._fh = open(self.path, "w", encoding="utf-8")
        self.line_count = 0
        self._last_flush = time.time()
        log.info("Started JSONL trace: %s", self.path)

    def write(self, record: dict) -> None:
        self._fh.write(json.dumps(record) + "\n")
        self.line_count += 1
        now = time.time()
        if now - self._last_flush >= TRACE_FLUSH_INTERVAL:
            self._fh.flush()
            self._last_flush = now

    def close(self) -> None:
        try:
            self._fh.flush()
            self._fh.close()
        finally:
            log.info(
                "Stopped JSONL trace: %s (%d lines)",
                self.path, self.line_count,
            )


# --------------------------------------------------------------------------- #
# Debug: annotated camera snapshots
# --------------------------------------------------------------------------- #
class Snapshotter:
    """Saves annotated BGR frames as PNGs into the session frames/ folder.

    Two triggers: periodic (every `interval` seconds) and on-click (one per
    dwell-click, filename prefixed `click-`). Filenames embed a monotonically
    increasing sequence and the elapsed seconds since the snapshotter started.
    The interval is the only throttle, which keeps the periodic stream bounded.
    """

    def __init__(self, frames_dir: str, interval: float = DEFAULT_SNAP_INTERVAL) -> None:
        os.makedirs(frames_dir, exist_ok=True)
        self.frames_dir = frames_dir
        self.interval = float(interval)
        self.count = 0
        self._seq = 0
        self._start = time.time()
        self._last_periodic = 0.0  # force a snapshot on the first frame
        self._logged_first = False

    def _save(self, frame: np.ndarray, now: float, prefix: str) -> str:
        self._seq += 1
        elapsed = now - self._start
        name = f"{prefix}-{self._seq:05d}-t{elapsed:.1f}s.png"
        path = os.path.join(self.frames_dir, name)
        cv2.imwrite(path, frame)
        self.count += 1
        if not self._logged_first:
            self._logged_first = True
            log.info("First snapshot saved: %s", path)
        return path

    def maybe_periodic(self, frame: np.ndarray, now: float) -> None:
        """Save a periodic frame if the interval has elapsed."""
        if now - self._last_periodic >= self.interval:
            self._last_periodic = now
            self._save(frame, now, "frame")

    def on_click(self, frame: np.ndarray, now: float) -> str:
        """Save a snapshot for a dwell-click; does not affect periodic timing."""
        return self._save(frame, now, "click")

    def close(self) -> None:
        log.info("Snapshots saved: %d (%s)", self.count, self.frames_dir)


# --------------------------------------------------------------------------- #
# Head-pose math
# --------------------------------------------------------------------------- #
def yaw_pitch_from_matrix(matrix: np.ndarray) -> tuple[float, float]:
    """Extract (yaw, pitch) in radians from a 4x4 facial transform matrix.

    The upper-left 3x3 block is the rotation. We read Euler angles in a
    yaw (Y) / pitch (X) convention sufficient for steering a cursor.
    """
    r = np.asarray(matrix, dtype=np.float64)[:3, :3]

    # pitch about X, yaw about Y, using the standard ZYX decomposition.
    sy = math.sqrt(r[0, 0] * r[0, 0] + r[1, 0] * r[1, 0])
    if sy > 1e-6:
        pitch = math.atan2(r[2, 1], r[2, 2])
        yaw = math.atan2(-r[2, 0], sy)
    else:  # gimbal-lock fallback
        pitch = math.atan2(-r[1, 2], r[1, 1])
        yaw = math.atan2(-r[2, 0], sy)
    return yaw, pitch


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


# --------------------------------------------------------------------------- #
# Eye-gaze math
# --------------------------------------------------------------------------- #
_eye_gaze_warn_last = 0.0


def _eye_ratio(landmarks, idx: tuple) -> tuple[float, float] | None:
    """Normalized (horizontal, vertical) iris offset for one eye, or None.

    Horizontal: where the iris center sits between inner and outer corner,
    re-centered so ~0 means looking straight ahead. Vertical: where the iris
    center sits between the upper and lower eyelid, re-centered to ~0.
    Returns None if a landmark is missing or the eye span is degenerate.
    """
    outer_i, inner_i, upper_i, lower_i, iris_i = idx
    try:
        outer = landmarks[outer_i]
        inner = landmarks[inner_i]
        upper = landmarks[upper_i]
        lower = landmarks[lower_i]
        iris = landmarks[iris_i]
    except (IndexError, TypeError, KeyError):
        return None

    width = outer.x - inner.x
    height = lower.y - upper.y
    if abs(width) < 1e-6 or abs(height) < 1e-6:
        return None

    # Fraction of the eye span (0 at inner, 1 at outer), re-centered to ~0.
    h_ratio = (iris.x - inner.x) / width - 0.5
    v_ratio = (iris.y - upper.y) / height - 0.5
    # Scale [-0.5, 0.5] span to roughly [-1, 1].
    return h_ratio * 2.0, v_ratio * 2.0


def eye_gaze_from_landmarks(landmarks) -> tuple[float, float]:
    """Appearance-based gaze as a normalized iris offset, averaged over eyes.

    `landmarks` is the normalized landmark list for face 0 (each item exposing
    .x/.y in [0, 1]). Returns (gaze_x, gaze_y) in roughly [-1, 1] each, where
    (0, 0) is looking center. Degrades gracefully (returns the available eye,
    or (0.0, 0.0) if neither eye is usable) and logs a throttled warning.
    """
    left = _eye_ratio(landmarks, LEFT_EYE_IDX)
    right = _eye_ratio(landmarks, RIGHT_EYE_IDX)

    usable = [e for e in (left, right) if e is not None]
    if not usable:
        global _eye_gaze_warn_last
        now = time.time()
        if now - _eye_gaze_warn_last >= PERIODIC_LOG_INTERVAL:
            _eye_gaze_warn_last = now
            log.warning("Eye-gaze landmarks unavailable; gaze falling back to 0")
        return 0.0, 0.0

    gaze_x = sum(e[0] for e in usable) / len(usable)
    gaze_y = sum(e[1] for e in usable) / len(usable)
    return gaze_x, gaze_y


# --------------------------------------------------------------------------- #
# Eye-gaze calibration (pure, unit-testable)
# --------------------------------------------------------------------------- #
def poly_features(gx: float, gy: float) -> list[float]:
    """2nd-order feature vector for the iris->pixel regression.

    Order matches CALIB_FEATURE_ORDER: [1, gx, gy, gx^2, gy^2, gx*gy].
    """
    return [1.0, gx, gy, gx * gx, gy * gy, gx * gy]


def fit_calibration(samples) -> dict:
    """Fit two least-squares polynomials mapping gaze -> screen x and y.

    `samples` is a list of ((gx, gy), (sx, sy)) correspondences. Returns a
    JSON-serializable model dict holding the two coefficient vectors, the
    feature order, the screen size, the point count and a timestamp.
    """
    if not samples:
        raise ValueError("fit_calibration needs at least one sample")

    a = np.array([poly_features(gx, gy) for (gx, gy), _ in samples], dtype=np.float64)
    bx = np.array([sx for _, (sx, _sy) in samples], dtype=np.float64)
    by = np.array([sy for _, (_sx, sy) in samples], dtype=np.float64)

    coef_x, *_ = np.linalg.lstsq(a, bx, rcond=None)
    coef_y, *_ = np.linalg.lstsq(a, by, rcond=None)

    return {
        "feature_order": list(CALIB_FEATURE_ORDER),
        "coef_x": [float(c) for c in coef_x],
        "coef_y": [float(c) for c in coef_y],
        "points": len(samples),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def predict_calibration(model: dict, gx: float, gy: float) -> tuple[float, float]:
    """Predict (screen_x, screen_y) from a fitted model and a gaze sample."""
    feats = poly_features(gx, gy)
    coef_x = model["coef_x"]
    coef_y = model["coef_y"]
    x = sum(f * c for f, c in zip(feats, coef_x))
    y = sum(f * c for f, c in zip(feats, coef_y))
    return x, y


def face_bbox_from_landmarks(
    landmarks, frame_w: int, frame_h: int
) -> tuple[int, int, int, int]:
    """Pixel-space face bounding box (x, y, w, h) over all landmarks.

    Min/max over normalized landmark x/y, expanded by BBOX_MARGIN and clamped
    to the frame. Returns (0, 0, 0, 0) if there are no landmarks.
    """
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    if not xs or not ys:
        return 0, 0, 0, 0

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    mx = (max_x - min_x) * BBOX_MARGIN
    my = (max_y - min_y) * BBOX_MARGIN

    x0 = clamp((min_x - mx) * frame_w, 0, frame_w - 1)
    y0 = clamp((min_y - my) * frame_h, 0, frame_h - 1)
    x1 = clamp((max_x + mx) * frame_w, 0, frame_w - 1)
    y1 = clamp((max_y + my) * frame_h, 0, frame_h - 1)

    x, y = int(x0), int(y0)
    w = max(0, int(x1) - x)
    h = max(0, int(y1) - y)
    return x, y, w, h


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def ensure_model() -> str:
    """Return the model path, downloading it on first run if missing."""
    if os.path.exists(MODEL_PATH):
        return MODEL_PATH
    log.info("Model not found, downloading from %s", MODEL_URL)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    log.info("Model downloaded to %s", MODEL_PATH)
    return MODEL_PATH


def save_calibration(model: dict, screen_w: float, screen_h: float) -> str:
    """Persist a fitted calibration model to debug/eye_calibration.json."""
    os.makedirs(DEBUG_DIR, exist_ok=True)
    record = dict(model)
    record["screen_w"] = float(screen_w)
    record["screen_h"] = float(screen_h)
    with open(CALIB_PATH, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
    log.info(
        "Saved eye calibration: %s (%d points, screen %.0fx%.0f)",
        CALIB_PATH, record.get("points", 0), screen_w, screen_h,
    )
    return CALIB_PATH


def load_calibration() -> dict | None:
    """Load a saved calibration model, or None if absent / unreadable."""
    if not os.path.exists(CALIB_PATH):
        return None
    try:
        with open(CALIB_PATH, encoding="utf-8") as fh:
            model = json.load(fh)
    except (OSError, ValueError) as exc:
        log.warning("Could not read calibration %s: %s", CALIB_PATH, exc)
        return None
    if "coef_x" not in model or "coef_y" not in model:
        log.warning("Calibration %s missing coefficients; ignoring", CALIB_PATH)
        return None
    return model


def build_eyegestures():
    """Construct an EyeGestures_v3 tracker, importing the library lazily.

    eyeGestures is GPLv3 and optional: it is imported here, only when the
    eyegestures engine is selected, so the MIT builtin path runs even if the
    library (and its sklearn dependency) is not installed. On import failure we
    log a clear, actionable error and re-raise so the caller can exit cleanly.
    """
    try:
        from eyeGestures import EyeGestures_v3
    except ImportError as exc:
        log.error(
            "eyeGestures engine requested but the library failed to import: %s. "
            "Install it (and scikit-learn) into the venv, or use "
            "--engine builtin.",
            exc,
        )
        raise
    g = EyeGestures_v3(calibration_radius=EYEGESTURES_CALIB_RADIUS)
    return g


def eyegestures_grid(cols: int, rows: int) -> list[list[float]]:
    """Normalized NxM calibration grid of [0,1] points the tracker walks.

    Row-major list of [x, y] in [0, 1]. Uploaded via uploadCalibrationMap; the
    tracker advances one target at a time as the user fixates each in turn.
    """
    if cols < 2 or rows < 2:
        raise ValueError("calibration grid needs at least 2x2 points")
    grid = []
    for r in range(rows):
        fy = r / (rows - 1)
        for c in range(cols):
            fx = c / (cols - 1)
            grid.append([fx, fy])
    return grid


def build_landmarker():
    """Create a MediaPipe FaceLandmarker in VIDEO mode with transform output."""
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    base_options = mp_python.BaseOptions(model_asset_path=ensure_model())
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        output_facial_transformation_matrixes=True,
        output_face_blendshapes=False,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #
class GazeMouse:
    """Owns the capture loop, head-pose mapping, smoothing and clicking."""

    def __init__(
        self,
        session: Session,
        show_window: bool,
        record: bool = False,
        mode: str = "head",
        use_calib: bool = True,
        calibrate_on_start: bool = False,
        snapshots: bool = True,
        snap_interval: float = DEFAULT_SNAP_INTERVAL,
        engine: str = ENGINE_BUILTIN,
    ) -> None:
        self.session = session
        self.show_window = show_window
        self.engine = engine  # "builtin" or "eyegestures"
        self.mode = mode  # "head" or "eye"
        self.screen_w, self.screen_h = get_screen_size()
        self.start_wall = time.time()

        # Eye-gaze calibration model (fitted iris->pixel map). When present it
        # replaces the neutral+gain mapping for eye mode.
        self.use_calib = use_calib
        self.calibrate_on_start = calibrate_on_start
        self.calib_model: dict | None = None
        if use_calib:
            self.calib_model = load_calibration()
            self._log_calib_state()
        elif load_calibration() is not None:
            log.info("Eye calibration present but ignored (--no-calib)")

        # debug recording / tracing / snapshots
        self.record_requested = record
        self.snapshots_enabled = snapshots
        self.snap_interval = snap_interval
        self.recorder: Recorder | None = None
        self.tracer: Tracer | None = None
        self.snapshotter: Snapshotter | None = None
        self.frame_idx = 0
        self.click_count = 0
        self.modes_used: set[str] = {mode}

        # Flags recorded into meta.json. Set by main() before run().
        self.flags: dict = {
            "window": show_window,
            "engine": engine,
            "record": record,
            "mode": mode,
            "use_calib": use_calib,
            "calibrate_on_start": calibrate_on_start,
            "snapshots": snapshots,
            "snap_interval": snap_interval,
        }

        self.filter_x = OneEuroFilter()
        self.filter_y = OneEuroFilter()

        self.neutral_yaw = 0.0
        self.neutral_pitch = 0.0
        self.neutral_gaze_x = 0.0
        self.neutral_gaze_y = 0.0
        self.calibrated = False

        self.control_enabled = False  # OFF for safety until the user opts in

        # dwell state
        self.dwell_anchor: tuple[float, float] | None = None
        self.dwell_start = 0.0
        self.dwell_armed = True  # must leave radius before re-firing

        # throttled logging / fps
        self._last_periodic_log = 0.0
        self._last_frame_time = time.time()
        self._fps = 0.0

        self.running = True

    # -- calibration ------------------------------------------------------- #
    def recenter(
        self, yaw: float, pitch: float, gaze_x: float, gaze_y: float
    ) -> None:
        """Set the neutral for the currently active mode.

        Both signals are stored so switching modes preserves each neutral, but
        only the active mode's neutral is updated from this frame's reading.
        """
        if self.mode == "eye":
            self.neutral_gaze_x = gaze_x
            self.neutral_gaze_y = gaze_y
            log.info(
                "Recentered neutral gaze: gaze_x=%.3f gaze_y=%.3f", gaze_x, gaze_y
            )
        else:
            self.neutral_yaw = yaw
            self.neutral_pitch = pitch
            log.info(
                "Recentered neutral pose: yaw=%.3f pitch=%.3f rad", yaw, pitch
            )
        self.calibrated = True
        self.filter_x = OneEuroFilter()
        self.filter_y = OneEuroFilter()

    def toggle_mode(self) -> None:
        self.mode = "eye" if self.mode == "head" else "head"
        self.modes_used.add(self.mode)
        self.filter_x = OneEuroFilter()
        self.filter_y = OneEuroFilter()
        log.info("Tracking mode switched to %s", self.mode.upper())

    def toggle_control(self) -> None:
        self.control_enabled = not self.control_enabled
        log.info("Cursor control %s", "ON" if self.control_enabled else "OFF")

    # -- debug recording --------------------------------------------------- #
    def toggle_recording(self) -> None:
        if self.recorder is None:
            fps = self._fps if self._fps > 0 else None
            self.recorder = Recorder(self.session.recording_path, fps=fps)
        else:
            self.recorder.close()
            self.recorder = None

    # -- mapping ----------------------------------------------------------- #
    def pose_to_screen(self, yaw: float, pitch: float) -> tuple[float, float]:
        """Map a head pose to a clamped screen point."""
        dyaw = yaw - self.neutral_yaw
        dpitch = pitch - self.neutral_pitch

        # yaw: turning head left (negative) should move cursor left.
        nx = 0.5 + dyaw * GAIN_X
        # pitch: looking down (positive) should move cursor down.
        ny = 0.5 + dpitch * GAIN_Y

        x = clamp(nx, 0.0, 1.0) * (self.screen_w - 1)
        y = clamp(ny, 0.0, 1.0) * (self.screen_h - 1)
        return x, y

    def calib_active(self) -> bool:
        """True when a fitted calibration model is loaded and enabled."""
        return self.use_calib and self.calib_model is not None

    def _log_calib_state(self) -> None:
        """Log whether eye mapping is the fitted model or the gain fallback."""
        if self.calib_model is not None:
            log.info(
                "Eye calibration ACTIVE: %d points, fitted %s (%s)",
                self.calib_model.get("points", 0),
                self.calib_model.get("timestamp", "?"),
                CALIB_PATH,
            )
        else:
            log.info("Eye calibration not loaded; eye mode falls back to gain")

    def gaze_to_screen(self, gaze_x: float, gaze_y: float) -> tuple[float, float]:
        """Map an iris-gaze offset to a clamped screen point.

        Uses the fitted calibration model when one is active; otherwise the
        neutral + gain + clamp pipeline (same shape as head pose).
        """
        if self.calib_active():
            px, py = predict_calibration(self.calib_model, gaze_x, gaze_y)
            x = clamp(px, 0.0, self.screen_w - 1)
            y = clamp(py, 0.0, self.screen_h - 1)
            return x, y

        dx = gaze_x - self.neutral_gaze_x
        dy = gaze_y - self.neutral_gaze_y

        nx = 0.5 + dx * EYE_GAIN_X
        ny = 0.5 + dy * EYE_GAIN_Y

        x = clamp(nx, 0.0, 1.0) * (self.screen_w - 1)
        y = clamp(ny, 0.0, 1.0) * (self.screen_h - 1)
        return x, y

    # -- dwell ------------------------------------------------------------- #
    def update_dwell(self, x: float, y: float, now: float) -> tuple[float, bool]:
        """Track dwell state and fire a click when satisfied.

        Returns (dwell progress in [0, 1], clicked-this-frame) for the HUD
        and trace.
        """
        if self.dwell_anchor is None:
            self.dwell_anchor = (x, y)
            self.dwell_start = now
            return 0.0, False

        ax, ay = self.dwell_anchor
        dist = math.hypot(x - ax, y - ay)

        if dist > DWELL_RADIUS:
            # moved away: reset anchor and re-arm the click.
            self.dwell_anchor = (x, y)
            self.dwell_start = now
            self.dwell_armed = True
            return 0.0, False

        elapsed = now - self.dwell_start
        progress = clamp(elapsed / DWELL_TIME, 0.0, 1.0)

        clicked = False
        if progress >= 1.0 and self.dwell_armed and self.control_enabled:
            left_click(x, y)
            self.dwell_armed = False  # require leaving radius before re-firing
            self.dwell_start = now
            clicked = True
            self.click_count += 1
            log.info("Dwell-click at (%.0f, %.0f)", x, y)
        return progress, clicked

    # -- logging ----------------------------------------------------------- #
    def periodic_log(
        self, now: float, yaw: float, pitch: float, x: float, y: float
    ) -> None:
        if now - self._last_periodic_log >= PERIODIC_LOG_INTERVAL:
            self._last_periodic_log = now
            log.debug(
                "fps=%.1f yaw=%.3f pitch=%.3f cursor=(%.0f, %.0f) control=%s",
                self._fps, yaw, pitch, x, y, self.control_enabled,
            )

    def update_fps(self, now: float) -> None:
        dt = now - self._last_frame_time
        self._last_frame_time = now
        if dt > 0:
            inst = 1.0 / dt
            self._fps = 0.9 * self._fps + 0.1 * inst if self._fps else inst

    # -- HUD --------------------------------------------------------------- #
    def draw_hud(
        self,
        frame: np.ndarray,
        yaw: float,
        pitch: float,
        cursor: tuple[float, float] | None,
        progress: float,
        face_xy: tuple[int, int] | None,
    ) -> None:
        h, w = frame.shape[:2]
        if face_xy is not None:
            cv2.circle(frame, face_xy, 5, (0, 255, 0), -1)

        line = (
            f"fps {self._fps:4.1f} | eng {self.engine[:3].upper()} | "
            f"mode {self.mode.upper():4s} | "
            f"yaw {math.degrees(yaw):+5.1f} "
            f"pitch {math.degrees(pitch):+5.1f} | "
            f"ctrl {'ON' if self.control_enabled else 'OFF'} | "
            f"dwell {int(progress * 100):3d}%"
        )
        if self.engine == ENGINE_BUILTIN and self.mode == "eye":
            line += f" | calib {'on' if self.calib_active() else 'off'}"
        line += f" | snap {'on' if self.snapshots_enabled else 'off'}"
        cv2.putText(
            frame, line, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0, 255, 255), 2, cv2.LINE_AA,
        )
        hint = "q quit  c recenter  space toggle  r record  m mode  k calibrate"
        cv2.putText(
            frame, hint, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (200, 200, 200), 1, cv2.LINE_AA,
        )
        # dwell progress bar
        bar_w = int(progress * (w - 20))
        cv2.rectangle(frame, (10, h - 40), (10 + bar_w, h - 30),
                      (0, 200, 0), -1)

    def draw_face_overlay(
        self, frame: np.ndarray, landmarks, cam_w: int, cam_h: int
    ) -> None:
        """Draw the face bounding box, plus eye markers in eye mode."""
        x, y, bw, bh = face_bbox_from_landmarks(landmarks, cam_w, cam_h)
        if bw > 0 and bh > 0:
            cv2.rectangle(
                frame, (x, y), (x + bw, y + bh), (255, 200, 0), 1, cv2.LINE_AA
            )

        if self.mode != "eye":
            return

        def px(i: int) -> tuple[int, int] | None:
            try:
                lm = landmarks[i]
            except (IndexError, TypeError, KeyError):
                return None
            return int(lm.x * cam_w), int(lm.y * cam_h)

        # Eye corners (upper/lower lid + corners) as small markers.
        for idx in LEFT_EYE_IDX[:4] + RIGHT_EYE_IDX[:4]:
            p = px(idx)
            if p is not None:
                cv2.circle(frame, p, 2, (0, 255, 255), 1, cv2.LINE_AA)

        # Iris centers as filled dots.
        for idx in (LEFT_EYE_IDX[4], RIGHT_EYE_IDX[4]):
            p = px(idx)
            if p is not None:
                cv2.circle(frame, p, 3, (0, 0, 255), -1, cv2.LINE_AA)

    # -- eye calibration --------------------------------------------------- #
    def _calib_targets(self) -> list[tuple[float, float]]:
        """3x3 grid of screen pixel targets at CALIB_MARGIN from each edge."""
        fracs = (CALIB_MARGIN, 0.5, 1.0 - CALIB_MARGIN)
        targets = []
        for fy in fracs:
            for fx in fracs:
                targets.append((fx * (self.screen_w - 1), fy * (self.screen_h - 1)))
        return targets

    def run_calibration(self, cap, landmarker) -> bool:
        """Run the fullscreen 9-point eye-gaze calibration.

        Presents each target, settles, then collects median raw gaze samples,
        fits a model and (on success) installs it for immediate use. Returns
        True if a new model was fitted, False if aborted or too few points
        (in which case the existing mapping is kept).
        """
        from mediapipe import Image, ImageFormat

        win = "gaze-calibration"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        sw, sh = int(self.screen_w), int(self.screen_h)
        targets = self._calib_targets()
        samples: list[tuple[tuple[float, float], tuple[float, float]]] = []
        aborted = False
        log.info("Eye calibration started: %d targets", len(targets))

        def read_gaze() -> tuple[float, float] | None:
            ok, frm = cap.read()
            if not ok:
                return None
            frm = cv2.flip(frm, 1)
            rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
            mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
            res = landmarker.detect_for_video(mp_image, int(time.time() * 1000))
            if not res.face_landmarks:
                return None
            return eye_gaze_from_landmarks(res.face_landmarks[0])

        def draw_target(tx, ty, ring_frac, label):
            canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
            cx, cy = int(tx), int(ty)
            cv2.circle(canvas, (cx, cy), 30, (40, 40, 40), 2, cv2.LINE_AA)
            if ring_frac > 0.0:
                cv2.ellipse(
                    canvas, (cx, cy), (30, 30), -90, 0,
                    int(360 * clamp(ring_frac, 0.0, 1.0)), (0, 220, 0), 4,
                    cv2.LINE_AA,
                )
            cv2.circle(canvas, (cx, cy), 10, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(
                canvas, label, (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (200, 200, 200), 2, cv2.LINE_AA,
            )
            cv2.putText(
                canvas, "look at the red dot   ESC abort   space skip point",
                (40, sh - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (150, 150, 150), 1, cv2.LINE_AA,
            )
            cv2.imshow(win, canvas)

        for i, (tx, ty) in enumerate(targets):
            label = f"point {i + 1}/{len(targets)}"
            # Settle: ignore samples, show static dot.
            settle_end = time.time() + CALIB_SETTLE
            skip = False
            while time.time() < settle_end:
                read_gaze()
                draw_target(tx, ty, 0.0, label + " (settle)")
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    aborted = True
                    break
                if key == ord(" "):
                    skip = True
                    break
            if aborted:
                break
            if skip:
                log.info("Calibration point %d skipped", i + 1)
                continue

            # Collect samples, filling a progress ring.
            gxs: list[float] = []
            gys: list[float] = []
            while len(gxs) < CALIB_SAMPLES:
                g = read_gaze()
                if g is not None:
                    gxs.append(g[0])
                    gys.append(g[1])
                draw_target(tx, ty, len(gxs) / CALIB_SAMPLES, label)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    aborted = True
                    break
                if key == ord(" "):
                    skip = True
                    break
            if aborted:
                break
            if skip or not gxs:
                log.info("Calibration point %d skipped (no samples)", i + 1)
                continue

            med_gx = float(np.median(gxs))
            med_gy = float(np.median(gys))
            samples.append(((med_gx, med_gy), (tx, ty)))
            log.info(
                "Calib point %d: gaze=(%.3f, %.3f) -> screen=(%.0f, %.0f) "
                "from %d samples",
                i + 1, med_gx, med_gy, tx, ty, len(gxs),
            )

        cv2.destroyWindow(win)

        if aborted:
            log.warning("Calibration aborted (ESC); keeping previous mapping")
            return False
        if len(samples) < CALIB_MIN_POINTS:
            log.warning(
                "Calibration got %d/%d good points (<%d); keeping previous "
                "mapping", len(samples), len(targets), CALIB_MIN_POINTS,
            )
            return False

        model = fit_calibration(samples)
        save_calibration(model, self.screen_w, self.screen_h)
        self.calib_model = model
        self.use_calib = True
        self.filter_x = OneEuroFilter()
        self.filter_y = OneEuroFilter()
        log.info(
            "Calibration complete: fitted model from %d points, now ACTIVE",
            len(samples),
        )
        return True

    # -- main loop --------------------------------------------------------- #
    def run(self) -> None:
        """Dispatch to the selected engine's capture loop."""
        if self.engine == ENGINE_EYEGESTURES:
            self.run_eyegestures()
        else:
            self.run_builtin()

    def run_builtin(self) -> None:
        landmarker = build_landmarker()
        from mediapipe import Image, ImageFormat

        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            log.error("Failed to open camera index %d", CAMERA_INDEX)
            return

        cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log.info("Camera opened: %dx%d", cam_w, cam_h)
        log.info("Screen size: %.0fx%.0f", self.screen_w, self.screen_h)
        log.info("Session folder: %s", self.session.dir)
        log.info("Cursor control starts OFF; press space to enable.")

        self.tracer = Tracer(self.session.trace_path)
        if self.record_requested:
            self.recorder = Recorder(self.session.recording_path, fps=None)
        if self.snapshots_enabled:
            self.snapshotter = Snapshotter(
                self.session.frames_dir, interval=self.snap_interval
            )

        if self.calibrate_on_start and self.show_window:
            self.run_calibration(cap, landmarker)
        elif self.calibrate_on_start:
            log.warning("--calibrate needs the preview window; skipping")

        try:
            while self.running:
                ok, frame = cap.read()
                if not ok:
                    log.warning("Dropped frame from camera")
                    continue

                frame = cv2.flip(frame, 1)  # mirror for natural movement
                now = time.time()
                self.update_fps(now)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
                ts_ms = int(now * 1000)
                result = landmarker.detect_for_video(mp_image, ts_ms)

                self.frame_idx += 1
                yaw = pitch = 0.0
                gaze_x = gaze_y = 0.0
                have_gaze = False
                cursor: tuple[float, float] | None = None
                target: tuple[float, float] | None = None
                progress = 0.0
                clicked = False
                face_xy: tuple[int, int] | None = None
                landmarks = (
                    result.face_landmarks[0] if result.face_landmarks else None
                )

                if landmarks is not None:
                    gaze_x, gaze_y = eye_gaze_from_landmarks(landmarks)
                    have_gaze = True

                if result.facial_transformation_matrixes:
                    matrix = np.array(
                        result.facial_transformation_matrixes[0]
                    ).reshape(4, 4)
                    yaw, pitch = yaw_pitch_from_matrix(matrix)

                    if not self.calibrated:
                        self.recenter(yaw, pitch, gaze_x, gaze_y)

                    if landmarks is not None:
                        nose = landmarks[1]
                        face_xy = (int(nose.x * cam_w), int(nose.y * cam_h))

                    if self.mode == "eye":
                        raw_x, raw_y = self.gaze_to_screen(gaze_x, gaze_y)
                    else:
                        raw_x, raw_y = self.pose_to_screen(yaw, pitch)
                    target = (raw_x, raw_y)
                    sx = self.filter_x(raw_x, now)
                    sy = self.filter_y(raw_y, now)
                    cursor = (sx, sy)

                    if self.control_enabled:
                        warp_cursor(sx, sy)
                    progress, clicked = self.update_dwell(sx, sy, now)

                    self.periodic_log(now, yaw, pitch, sx, sy)

                # Build the annotated frame (HUD + face dot + minimap). This is
                # the same BGR frame shown in the window, fed to the recorder and
                # saved as snapshots, so it is built even in --no-window mode
                # whenever any consumer needs it.
                annotate = (
                    self.show_window
                    or self.recorder is not None
                    or self.snapshotter is not None
                )
                if annotate:
                    if landmarks is not None:
                        self.draw_face_overlay(frame, landmarks, cam_w, cam_h)
                    self.draw_hud(frame, yaw, pitch, cursor, progress, face_xy)
                    draw_minimap(
                        frame, self.screen_w, self.screen_h,
                        target, cursor, progress,
                    )

                if self.recorder is not None:
                    self.recorder.write(frame)

                if self.snapshotter is not None:
                    if clicked:
                        self.snapshotter.on_click(frame, now)
                    self.snapshotter.maybe_periodic(frame, now)

                if self.tracer is not None:
                    self.tracer.write({
                        "t": now,
                        "frame": self.frame_idx,
                        "fps": round(self._fps, 2),
                        "engine": self.engine,
                        "mode": self.mode,
                        "yaw": round(yaw, 5),
                        "pitch": round(pitch, 5),
                        "gaze_x": round(gaze_x, 5) if have_gaze else None,
                        "gaze_y": round(gaze_y, 5) if have_gaze else None,
                        "target_x": round(target[0], 1) if target else None,
                        "target_y": round(target[1], 1) if target else None,
                        "cursor_x": round(cursor[0], 1) if cursor else None,
                        "cursor_y": round(cursor[1], 1) if cursor else None,
                        "dwell": round(progress, 3),
                        "control_on": self.control_enabled,
                        "click": clicked,
                    })

                if self.show_window:
                    cv2.imshow("gaze-mouse", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        log.info("Quit requested (q)")
                        break
                    if key == ord("c") and result.facial_transformation_matrixes:
                        self.recenter(yaw, pitch, gaze_x, gaze_y)
                    if key == ord(" "):
                        self.toggle_control()
                    if key == ord("r"):
                        self.toggle_recording()
                    if key == ord("m"):
                        self.toggle_mode()
                    if key == ord("k"):
                        self.run_calibration(cap, landmarker)
        finally:
            cap.release()
            if self.show_window:
                cv2.destroyAllWindows()
            if self.recorder is not None:
                self.recorder.close()
                self.recorder = None
            if self.tracer is not None:
                self.tracer.close()
                self.tracer = None
            if self.snapshotter is not None:
                self.snapshotter.close()
            self.write_meta()
            landmarker.close()
            log.info("Shut down cleanly")

    # -- eyeGestures engine ------------------------------------------------ #
    def _eyegestures_target(self, event) -> tuple[float, float]:
        """Clamp an EyeGestures step event's screen point to the display."""
        px = float(event.point[0])
        py = float(event.point[1])
        x = clamp(px, 0.0, self.screen_w - 1)
        y = clamp(py, 0.0, self.screen_h - 1)
        return x, y

    def run_eyegestures(self) -> None:
        """Capture loop backed by the third-party eyeGestures tracker (GPLv3).

        The per-frame screen point comes from EyeGestures_v3.step() instead of
        the FaceLandmarker pipeline, then flows through the SAME downstream as
        the builtin engine: One-Euro smoothing, screen clamp, cursor warp,
        dwell-click, minimap/HUD overlays, the JSONL trace, snapshots, the
        recorder and meta.json.
        """
        try:
            tracker = build_eyegestures()
        except ImportError:
            return  # build_eyegestures already logged an actionable error

        grid = eyegestures_grid(*EYEGESTURES_GRID)
        tracker.uploadCalibrationMap(grid)
        self._eg_grid_len = len(grid)

        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            log.error("Failed to open camera index %d", CAMERA_INDEX)
            return

        cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log.info("Camera opened: %dx%d", cam_w, cam_h)
        log.info("Screen size: %.0fx%.0f", self.screen_w, self.screen_h)
        log.info("Session folder: %s", self.session.dir)
        log.info("Engine: eyeGestures (GPLv3, grid %dx%d)", *EYEGESTURES_GRID)
        log.info("Cursor control starts OFF; press space to enable.")

        self.tracer = Tracer(self.session.trace_path)
        if self.record_requested:
            self.recorder = Recorder(self.session.recording_path, fps=None)
        if self.snapshots_enabled:
            self.snapshotter = Snapshotter(
                self.session.frames_dir, interval=self.snap_interval
            )

        # When --calibrate is passed, start in calibration; the grid is walked
        # in-loop so all recording features stay live during calibration too.
        self._eg_calibrating = bool(self.calibrate_on_start)
        self._eg_calib_seen = 0
        self._eg_last_calib_pt: tuple[int, int] | None = None
        if self._eg_calibrating:
            log.info(
                "eyeGestures calibration started: walking %d targets", len(grid)
            )

        try:
            while self.running:
                ok, frame = cap.read()
                if not ok:
                    log.warning("Dropped frame from camera")
                    continue

                frame = cv2.flip(frame, 1)  # mirror for natural movement
                now = time.time()
                self.update_fps(now)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frame_idx += 1

                target: tuple[float, float] | None = None
                cursor: tuple[float, float] | None = None
                calib_target: tuple[float, float] | None = None
                calib_radius = 0.0
                progress = 0.0
                clicked = False

                try:
                    event, calibration = tracker.step(
                        rgb, self._eg_calibrating,
                        int(self.screen_w), int(self.screen_h),
                    )
                except Exception:  # noqa: BLE001 - a bad frame must not kill the loop
                    log.exception("eyeGestures step failed on this frame")
                    event, calibration = None, None

                if calibration is not None:
                    try:
                        calib_target = (
                            float(calibration.point[0]),
                            float(calibration.point[1]),
                        )
                        calib_radius = float(calibration.acceptance_radius)
                    except (TypeError, IndexError, AttributeError):
                        calib_target = None

                if self._eg_calibrating and calib_target is not None:
                    # Count distinct calibration targets as the tracker advances
                    # through the grid; finish once all have been visited.
                    pt_key = (int(calib_target[0]), int(calib_target[1]))
                    if pt_key != self._eg_last_calib_pt:
                        self._eg_last_calib_pt = pt_key
                        self._eg_calib_seen += 1
                        log.info(
                            "eyeGestures calibration target %d/%d at (%d, %d)",
                            self._eg_calib_seen, self._eg_grid_len,
                            pt_key[0], pt_key[1],
                        )
                    if self._eg_calib_seen > self._eg_grid_len:
                        self._eg_calibrating = False
                        log.info(
                            "eyeGestures calibration finished; live tracking on"
                        )

                if event is not None and not self._eg_calibrating:
                    raw_x, raw_y = self._eyegestures_target(event)
                    target = (raw_x, raw_y)
                    sx = self.filter_x(raw_x, now)
                    sy = self.filter_y(raw_y, now)
                    cursor = (sx, sy)

                    if self.control_enabled:
                        warp_cursor(sx, sy)
                    progress, clicked = self.update_dwell(sx, sy, now)
                    self.periodic_log(now, 0.0, 0.0, sx, sy)

                # Annotate (HUD + minimap + calibration target). Face bbox / iris
                # dots are skipped here: the lib does not expose a matching
                # landmark list, so we simply do not draw them.
                annotate = (
                    self.show_window
                    or self.recorder is not None
                    or self.snapshotter is not None
                )
                if annotate:
                    self.draw_hud(frame, 0.0, 0.0, cursor, progress, None)
                    if self._eg_calibrating and calib_target is not None:
                        self._draw_eg_calib_target(
                            frame, cam_w, cam_h, calib_target, calib_radius
                        )
                    draw_minimap(
                        frame, self.screen_w, self.screen_h,
                        target, cursor, progress,
                    )

                if self.recorder is not None:
                    self.recorder.write(frame)

                if self.snapshotter is not None:
                    if clicked:
                        self.snapshotter.on_click(frame, now)
                    self.snapshotter.maybe_periodic(frame, now)

                if self.tracer is not None:
                    self.tracer.write({
                        "t": now,
                        "frame": self.frame_idx,
                        "fps": round(self._fps, 2),
                        "engine": self.engine,
                        "mode": self.mode,
                        "yaw": 0.0,
                        "pitch": 0.0,
                        "gaze_x": None,
                        "gaze_y": None,
                        "target_x": round(target[0], 1) if target else None,
                        "target_y": round(target[1], 1) if target else None,
                        "cursor_x": round(cursor[0], 1) if cursor else None,
                        "cursor_y": round(cursor[1], 1) if cursor else None,
                        "calibrating": self._eg_calibrating,
                        "dwell": round(progress, 3),
                        "control_on": self.control_enabled,
                        "click": clicked,
                    })

                if self.show_window:
                    cv2.imshow("gaze-mouse", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        log.info("Quit requested (q)")
                        break
                    if key == ord(" "):
                        self.toggle_control()
                    if key == ord("r"):
                        self.toggle_recording()
                    if key == ord("k"):
                        # Restart the lib's built-in calibration from the grid.
                        tracker.reset()
                        self._eg_calibrating = True
                        self._eg_calib_seen = 0
                        self._eg_last_calib_pt = None
                        self.filter_x = OneEuroFilter()
                        self.filter_y = OneEuroFilter()
                        log.info(
                            "eyeGestures calibration started: walking %d targets",
                            self._eg_grid_len,
                        )
        finally:
            cap.release()
            if self.show_window:
                cv2.destroyAllWindows()
            if self.recorder is not None:
                self.recorder.close()
                self.recorder = None
            if self.tracer is not None:
                self.tracer.close()
                self.tracer = None
            if self.snapshotter is not None:
                self.snapshotter.close()
            self.write_meta()
            log.info("Shut down cleanly")

    def _draw_eg_calib_target(
        self,
        frame: np.ndarray,
        cam_w: int,
        cam_h: int,
        calib_target: tuple[float, float],
        calib_radius: float,
    ) -> None:
        """Draw eyeGestures' active calibration point onto the preview frame.

        The target is in screen pixels; map it into the camera frame so it is
        visible in the preview window (and recorded / snapshotted). The
        acceptance radius is scaled by the same factor.
        """
        if self.screen_w <= 0 or self.screen_h <= 0:
            return
        sx = cam_w / self.screen_w
        sy = cam_h / self.screen_h
        cx = int(clamp(calib_target[0] * sx, 0, cam_w - 1))
        cy = int(clamp(calib_target[1] * sy, 0, cam_h - 1))
        ring = max(6, int(calib_radius * min(sx, sy)))
        cv2.circle(frame, (cx, cy), ring, (0, 220, 0), 1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(
            frame, "calibrating: look at the red dot", (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2, cv2.LINE_AA,
        )

    # -- meta -------------------------------------------------------------- #
    def write_meta(self) -> str:
        """Write meta.json summarizing the session. Idempotent and crash-safe."""
        end = time.time()
        duration = end - self.start_wall
        snap_count = self.snapshotter.count if self.snapshotter else 0
        mean_fps = self.frame_idx / duration if duration > 0 else 0.0
        meta = {
            "start": time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.localtime(self.start_wall)
            ),
            "end": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(end)),
            "duration_s": round(duration, 2),
            "frames": self.frame_idx,
            "mean_fps": round(mean_fps, 2),
            "screen_w": int(self.screen_w),
            "screen_h": int(self.screen_h),
            "engine": self.engine,
            "modes_used": sorted(self.modes_used),
            "calibration_active": self.calib_active(),
            "calibration_points": (
                self.calib_model.get("points", 0) if self.calib_model else 0
            ),
            "dwell_clicks": self.click_count,
            "snapshots": snap_count,
            "flags": dict(self.flags),
        }
        try:
            with open(self.session.meta_path, "w", encoding="utf-8") as fh:
                json.dump(meta, fh, indent=2)
            log.info("Wrote session meta: %s", self.session.meta_path)
        except OSError as exc:
            log.warning("Could not write meta.json: %s", exc)
        return self.session.meta_path


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move the macOS cursor with head pose (dwell-to-click)."
    )
    parser.add_argument(
        "--debug", action="store_true", help="enable DEBUG-level logging"
    )
    parser.add_argument(
        "--no-window", action="store_true",
        help="run headless without the preview window",
    )
    parser.add_argument(
        "--record", action="store_true",
        help="record the annotated preview to the session's recording.mp4",
    )
    parser.add_argument(
        "--trace", action="store_true",
        help="deprecated no-op; the per-frame trace.jsonl is always written",
    )
    parser.add_argument(
        "--no-snapshots", action="store_true",
        help="disable periodic + on-click camera snapshots (default: enabled)",
    )
    parser.add_argument(
        "--snap-interval", type=float, default=DEFAULT_SNAP_INTERVAL,
        help="seconds between periodic camera snapshots (default: %(default)s)",
    )
    parser.add_argument(
        "--engine", choices=ENGINES, default=ENGINE_BUILTIN,
        help=(
            "gaze engine: builtin MediaPipe pipeline (default, MIT) or "
            "eyegestures (GPLv3, lazy-imported only when selected)"
        ),
    )
    parser.add_argument(
        "--mode", choices=("head", "eye"), default="head",
        help="tracking source: head pose (default) or iris gaze",
    )
    parser.add_argument(
        "--calibrate", action="store_true",
        help="run the 9-point eye-gaze calibration first, then continue",
    )
    parser.add_argument(
        "--no-calib", action="store_true",
        help="ignore any saved debug/eye_calibration.json (use gain mapping)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    session = Session()
    setup_logging(args.debug, session.log_path)

    if args.trace:
        log.info("--trace is deprecated; trace.jsonl is always written now")

    snapshots = not args.no_snapshots
    use_calib = not args.no_calib
    calib_state = "off" if args.no_calib else (
        "fit" if args.calibrate else "auto"
    )
    log.info(
        "Startup config: engine=%s mode=%s gain=(%.1f, %.1f) "
        "eye_gain=(%.1f, %.1f) "
        "dwell=%.2fs/%.0fpx one_euro=(min_cutoff=%.2f, beta=%.3f) "
        "camera=%d window=%s debug=%s record=%s calib=%s "
        "snapshots=%s snap_interval=%.1fs session=%s",
        args.engine, args.mode, GAIN_X, GAIN_Y, EYE_GAIN_X, EYE_GAIN_Y,
        DWELL_TIME, DWELL_RADIUS, MIN_CUTOFF, BETA,
        CAMERA_INDEX, not args.no_window, args.debug,
        args.record, calib_state,
        snapshots, args.snap_interval, session.dir,
    )

    app = GazeMouse(
        session=session,
        show_window=not args.no_window,
        record=args.record,
        mode=args.mode,
        use_calib=use_calib,
        calibrate_on_start=args.calibrate,
        snapshots=snapshots,
        snap_interval=args.snap_interval,
        engine=args.engine,
    )

    def handle_sigint(_signum, _frame):
        log.info("Ctrl-C received, stopping")
        app.running = False

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        app.run()
    except Exception:  # noqa: BLE001 - log full trace before exiting
        log.exception("Fatal error in capture loop")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
