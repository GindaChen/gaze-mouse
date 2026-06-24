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

Keys (in the preview window):
    q      quit
    c      recenter / recalibrate the neutral for the active mode
    space  toggle cursor control on/off (starts OFF for safety)
    r      toggle webcam recording on/off (local file in debug/)
    m      toggle tracking mode (head <-> eye)

Debug artifacts (written to debug/):
    --record         start webcam recording immediately (also key 'r')
    --trace          write a per-frame JSONL trace
The recorded webcam video is a local file only; nothing is uploaded.
"""

from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import math
import os
import signal
import sys
import time
import urllib.request
from datetime import datetime

import cv2
import numpy as np

import Quartz

# --------------------------------------------------------------------------- #
# Config constants
# --------------------------------------------------------------------------- #
CAMERA_INDEX = 0

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

MINIMAP_W = 240              # px, on-frame screen minimap width
MINIMAP_H = 150             # px, on-frame screen minimap height
MINIMAP_MARGIN = 12         # px, gap from the frame edge

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(PROJECT_DIR, "gaze_mouse.log")
MODEL_PATH = os.path.join(PROJECT_DIR, MODEL_FILENAME)
DEBUG_DIR = os.path.join(PROJECT_DIR, DEBUG_DIR_NAME)

log = logging.getLogger("gaze_mouse")


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def setup_logging(debug: bool) -> None:
    """Configure logging to both stdout and a rotating file with timestamps."""
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

    file_handler = logging.handlers.RotatingFileHandler(
        LOG_PATH, maxBytes=1_000_000, backupCount=3
    )
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
    Records to debug/session-<timestamp>.mp4 with the mp4v fourcc. The video
    is a LOCAL FILE ONLY; nothing leaves the machine.
    """

    def __init__(self, debug_dir: str = DEBUG_DIR, fps: float | None = None) -> None:
        os.makedirs(debug_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.path = os.path.join(debug_dir, f"session-{stamp}.mp4")
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

    Writes to debug/trace-<timestamp>.jsonl and flushes periodically.
    """

    def __init__(self, debug_dir: str = DEBUG_DIR) -> None:
        os.makedirs(debug_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.path = os.path.join(debug_dir, f"trace-{stamp}.jsonl")
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
        show_window: bool,
        record: bool = False,
        trace: bool = False,
        mode: str = "head",
    ) -> None:
        self.show_window = show_window
        self.mode = mode  # "head" or "eye"
        self.screen_w, self.screen_h = get_screen_size()

        # debug recording / tracing
        self.record_requested = record
        self.trace_enabled = trace
        self.recorder: Recorder | None = None
        self.tracer: Tracer | None = None
        self.frame_idx = 0

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
            self.recorder = Recorder(fps=fps)
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

    def gaze_to_screen(self, gaze_x: float, gaze_y: float) -> tuple[float, float]:
        """Map an iris-gaze offset to a clamped screen point.

        Same neutral + gain + clamp pipeline as head pose, just a different
        source signal and gain constants.
        """
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
            f"fps {self._fps:4.1f} | mode {self.mode.upper():4s} | "
            f"yaw {math.degrees(yaw):+5.1f} "
            f"pitch {math.degrees(pitch):+5.1f} | "
            f"ctrl {'ON' if self.control_enabled else 'OFF'} | "
            f"dwell {int(progress * 100):3d}%"
        )
        cv2.putText(
            frame, line, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0, 255, 255), 2, cv2.LINE_AA,
        )
        hint = "q quit  c recenter  space toggle  r record  m mode"
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

    # -- main loop --------------------------------------------------------- #
    def run(self) -> None:
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
        log.info("Cursor control starts OFF; press space to enable.")

        if self.trace_enabled:
            self.tracer = Tracer()
        if self.record_requested:
            self.recorder = Recorder(fps=None)

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
                # the same BGR frame shown in the window and fed to the recorder,
                # so it is built even in --no-window mode when recording.
                annotate = self.show_window or self.recorder is not None
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

                if self.tracer is not None:
                    self.tracer.write({
                        "t": now,
                        "frame": self.frame_idx,
                        "fps": round(self._fps, 2),
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
            landmarker.close()
            log.info("Shut down cleanly")


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
        help="record the annotated preview to debug/session-*.mp4 (local file)",
    )
    parser.add_argument(
        "--trace", action="store_true",
        help="write a per-frame JSONL trace to debug/trace-*.jsonl",
    )
    parser.add_argument(
        "--mode", choices=("head", "eye"), default="head",
        help="tracking source: head pose (default) or iris gaze",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.debug)

    log.info(
        "Startup config: mode=%s gain=(%.1f, %.1f) eye_gain=(%.1f, %.1f) "
        "dwell=%.2fs/%.0fpx one_euro=(min_cutoff=%.2f, beta=%.3f) "
        "camera=%d window=%s debug=%s record=%s trace=%s",
        args.mode, GAIN_X, GAIN_Y, EYE_GAIN_X, EYE_GAIN_Y,
        DWELL_TIME, DWELL_RADIUS, MIN_CUTOFF, BETA,
        CAMERA_INDEX, not args.no_window, args.debug,
        args.record, args.trace,
    )

    app = GazeMouse(
        show_window=not args.no_window,
        record=args.record,
        trace=args.trace,
        mode=args.mode,
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
