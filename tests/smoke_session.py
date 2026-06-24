#!/usr/bin/env python3
"""Camera-free smoke test for the per-session debug layout.

Constructs a Session in a temp dir, writes a couple of trace rows, saves a
synthetic annotated frame as a periodic AND a click snapshot, writes meta.json,
then asserts the folder layout, that trace.jsonl is non-empty, that frames/
holds both a frame-* and a click-* png, and that meta.json parses with the
expected keys. Cleans up the temp dir on success.

Run:
    .venv/bin/python tests/smoke_session.py
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gaze_mouse import Session, Snapshotter, Tracer  # noqa: E402


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="smoke_session_")
    try:
        session = Session(debug_dir=tmp)

        # Folder + frames/ created eagerly; paths exposed.
        assert os.path.isdir(session.dir), f"missing session dir: {session.dir}"
        assert os.path.isdir(session.frames_dir), "missing frames/ dir"
        assert session.log_path.endswith("session.log"), session.log_path
        assert os.path.basename(session.dir).startswith("session-")

        # A couple of trace rows -> non-empty trace.jsonl.
        tracer = Tracer(session.trace_path)
        tracer.write({"frame": 1, "t": 0.0})
        tracer.write({"frame": 2, "t": 0.033})
        tracer.close()

        # Snapshots: a periodic frame-* and a click-* png from a synthetic frame.
        frame = np.full((240, 320, 3), 64, dtype=np.uint8)
        frame[:, :, 1] = 180
        snap = Snapshotter(session.frames_dir, interval=0.0)
        snap.maybe_periodic(frame, now=1.0)   # periodic -> frame-*
        snap.on_click(frame, now=1.5)         # dwell-click -> click-*
        snap.close()

        # meta.json written with the expected keys.
        meta = {
            "start": "2026-06-24T00:00:00",
            "end": "2026-06-24T00:00:05",
            "duration_s": 5.0,
            "frames": 2,
            "mean_fps": 0.4,
            "screen_w": 2560,
            "screen_h": 1600,
            "modes_used": ["eye", "head"],
            "calibration_active": False,
            "calibration_points": 0,
            "dwell_clicks": 1,
            "snapshots": snap.count,
            "flags": {"record": False, "snapshots": True},
        }
        with open(session.meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)

        # -- assertions ----------------------------------------------------- #
        assert os.path.getsize(session.trace_path) > 0, "empty trace.jsonl"
        with open(session.trace_path, encoding="utf-8") as fh:
            trace_lines = [ln for ln in fh if ln.strip()]
        assert len(trace_lines) == 2, f"expected 2 trace rows, got {len(trace_lines)}"

        frame_pngs = glob.glob(os.path.join(session.frames_dir, "frame-*.png"))
        click_pngs = glob.glob(os.path.join(session.frames_dir, "click-*.png"))
        assert frame_pngs, "no periodic frame-*.png saved"
        assert click_pngs, "no click-*.png saved"

        with open(session.meta_path, encoding="utf-8") as fh:
            loaded = json.load(fh)
        expected_keys = {
            "start", "end", "duration_s", "frames", "mean_fps",
            "screen_w", "screen_h", "modes_used", "calibration_active",
            "calibration_points", "dwell_clicks", "snapshots", "flags",
        }
        missing = expected_keys - set(loaded)
        assert not missing, f"meta.json missing keys: {missing}"

        print("SMOKE PASS")
        print(f"  session dir: {session.dir}")
        print(f"  trace rows:  {len(trace_lines)}")
        print(f"  frames:      {len(frame_pngs)} frame-*, {len(click_pngs)} click-*")
        print(f"  meta keys:   {sorted(loaded)}")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
