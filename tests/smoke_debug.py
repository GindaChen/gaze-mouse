#!/usr/bin/env python3
"""Camera-free smoke test for the debug recording/trace/minimap helpers.

Exercises draw_minimap, Recorder and Tracer against synthetic BGR frames,
then asserts the produced .mp4 and .jsonl files exist and are non-empty.

Run:
    .venv/bin/python tests/smoke_debug.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gaze_mouse import Recorder, Tracer, draw_minimap  # noqa: E402


def main() -> int:
    screen_w, screen_h = 2560.0, 1600.0
    n_frames = 8

    tmp = tempfile.mkdtemp(prefix="smoke_debug_")
    rec = Recorder(os.path.join(tmp, "recording.mp4"), fps=20.0)
    tracer = Tracer(os.path.join(tmp, "trace.jsonl"))

    for i in range(n_frames):
        # Synthetic 640x480 BGR frame with a moving gradient so it is non-trivial.
        frame = np.full((480, 640, 3), i * 20 % 256, dtype=np.uint8)
        frame[:, :, 1] = (i * 30) % 256

        target = (screen_w * (i / n_frames), screen_h * 0.5)
        cursor = (screen_w * (i / n_frames) - 40, screen_h * 0.5 + 10)
        progress = i / (n_frames - 1)

        out = draw_minimap(frame, screen_w, screen_h, target, cursor, progress)
        assert out is frame, "draw_minimap should mutate and return the frame"
        assert out.shape == (480, 640, 3)

        rec.write(out)
        tracer.write({
            "t": float(i),
            "frame": i,
            "fps": 20.0,
            "yaw": 0.01 * i,
            "pitch": -0.01 * i,
            "target_x": target[0],
            "target_y": target[1],
            "cursor_x": cursor[0],
            "cursor_y": cursor[1],
            "dwell": progress,
            "control_on": False,
            "click": i == n_frames - 1,
        })

    mp4_path = rec.path
    jsonl_path = tracer.path
    rec.close()
    tracer.close()

    assert os.path.exists(mp4_path), f"missing mp4: {mp4_path}"
    assert os.path.getsize(mp4_path) > 0, f"empty mp4: {mp4_path}"
    assert os.path.exists(jsonl_path), f"missing jsonl: {jsonl_path}"
    assert os.path.getsize(jsonl_path) > 0, f"empty jsonl: {jsonl_path}"

    with open(jsonl_path, encoding="utf-8") as fh:
        lines = [ln for ln in fh if ln.strip()]
    assert len(lines) == n_frames, f"expected {n_frames} trace lines, got {len(lines)}"

    print("SMOKE PASS")
    print(f"  mp4:   {mp4_path} ({os.path.getsize(mp4_path)} bytes, {rec.frame_count} frames)")
    print(f"  jsonl: {jsonl_path} ({os.path.getsize(jsonl_path)} bytes, {len(lines)} lines)")

    shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
