#!/usr/bin/env python3
"""Camera-free unit test for the pure eye-gaze + face-bbox helpers.

Builds SYNTHETIC normalized landmark lists and asserts:
  - eye_gaze_from_landmarks returns the expected gaze_x sign when the iris is
    deliberately offset left vs right, and ~0 when centered.
  - face_bbox_from_landmarks returns a plausible in-frame box.

Run:
    .venv/bin/python tests/smoke_eye.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gaze_mouse import (  # noqa: E402
    LEFT_EYE_IDX,
    RIGHT_EYE_IDX,
    eye_gaze_from_landmarks,
    face_bbox_from_landmarks,
)


@dataclass
class LM:
    x: float
    y: float
    z: float = 0.0


def build_landmarks(iris_frac: float, vert_frac: float = 0.5):
    """Synthetic 478-landmark list with iris placed at a chosen eye fraction.

    iris_frac/vert_frac are positions inside the eye span (0 = inner/upper
    corner, 1 = outer/lower). 0.5 means centered. The same fraction is used for
    both eyes so the averaged gaze is unambiguous.
    """
    n = 478
    pts = [LM(0.5, 0.5) for _ in range(n)]

    # Each eye: (outer_corner, inner_corner, upper_lid, lower_lid, iris_center).
    for idx, (cx, cy) in ((LEFT_EYE_IDX, (0.35, 0.45)), (RIGHT_EYE_IDX, (0.65, 0.45))):
        outer_i, inner_i, upper_i, lower_i, iris_i = idx
        half_w, half_h = 0.05, 0.02
        # inner corner left of outer for the layout below (width = outer - inner).
        inner_x, outer_x = cx - half_w, cx + half_w
        upper_y, lower_y = cy - half_h, cy + half_h
        pts[outer_i] = LM(outer_x, cy)
        pts[inner_i] = LM(inner_x, cy)
        pts[upper_i] = LM(cx, upper_y)
        pts[lower_i] = LM(cx, lower_y)
        iris_x = inner_x + iris_frac * (outer_x - inner_x)
        iris_y = upper_y + vert_frac * (lower_y - upper_y)
        pts[iris_i] = LM(iris_x, iris_y)

    # Spread a few extra points so the bbox spans a realistic face region.
    pts[10] = LM(0.5, 0.10)   # forehead top
    pts[152] = LM(0.5, 0.90)  # chin bottom
    pts[234] = LM(0.20, 0.50)  # left cheek
    pts[454] = LM(0.80, 0.50)  # right cheek
    return pts


def approx_zero(v: float, tol: float = 0.05) -> bool:
    return abs(v) <= tol


def main() -> int:
    # Centered iris -> gaze ~0.
    gx_c, gy_c = eye_gaze_from_landmarks(build_landmarks(0.5, 0.5))
    assert approx_zero(gx_c), f"centered gaze_x should be ~0, got {gx_c}"
    assert approx_zero(gy_c), f"centered gaze_y should be ~0, got {gy_c}"

    # Iris toward the inner corner (frac < 0.5) -> negative gaze_x.
    gx_left, _ = eye_gaze_from_landmarks(build_landmarks(0.1, 0.5))
    assert gx_left < -0.1, f"inner-offset gaze_x should be negative, got {gx_left}"

    # Iris toward the outer corner (frac > 0.5) -> positive gaze_x.
    gx_right, _ = eye_gaze_from_landmarks(build_landmarks(0.9, 0.5))
    assert gx_right > 0.1, f"outer-offset gaze_x should be positive, got {gx_right}"

    # Vertical: iris toward lower lid -> positive gaze_y.
    _, gy_down = eye_gaze_from_landmarks(build_landmarks(0.5, 0.9))
    assert gy_down > 0.1, f"lower-offset gaze_y should be positive, got {gy_down}"

    # Degenerate / empty landmarks fall back to (0, 0) without raising.
    gx_empty, gy_empty = eye_gaze_from_landmarks([])
    assert (gx_empty, gy_empty) == (0.0, 0.0), "empty landmarks should yield (0, 0)"

    # Face bbox: plausible, in-frame, positive size.
    frame_w, frame_h = 640, 480
    x, y, w, h = face_bbox_from_landmarks(build_landmarks(0.5), frame_w, frame_h)
    assert x >= 0 and y >= 0, f"bbox origin must be non-negative, got ({x}, {y})"
    assert w > 0 and h > 0, f"bbox size must be positive, got ({w}, {h})"
    assert x + w <= frame_w, f"bbox exceeds frame width: {x + w} > {frame_w}"
    assert y + h <= frame_h, f"bbox exceeds frame height: {y + h} > {frame_h}"

    # Empty landmarks -> zero box, no crash.
    assert face_bbox_from_landmarks([], frame_w, frame_h) == (0, 0, 0, 0)

    print("SMOKE PASS")
    print(f"  centered gaze: ({gx_c:+.3f}, {gy_c:+.3f})")
    print(f"  inner/outer gaze_x: {gx_left:+.3f} / {gx_right:+.3f}")
    print(f"  lower-lid gaze_y: {gy_down:+.3f}")
    print(f"  bbox: x={x} y={y} w={w} h={h} (frame {frame_w}x{frame_h})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
