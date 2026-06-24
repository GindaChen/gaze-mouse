#!/usr/bin/env python3
"""Camera-free smoke test for the --engine flag and the eyeGestures wiring.

Asserts, without a camera:
  - `gaze_mouse.py --help` advertises `--engine {builtin,eyegestures}`.
  - importing gaze_mouse does NOT import eyeGestures (the GPLv3 dependency is
    lazy: the builtin/MIT path must run even if eyeGestures is absent).
  - the normalized calibration grid helper produces in-[0,1] points.
  - EyeGestures_v3 can be constructed from the installed package (a cheap
    no-camera smoke of the third-party tracker).

Run:
    .venv/bin/python tests/smoke_engine.py
"""

from __future__ import annotations

import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def main() -> int:
    # --help advertises the engine flag with both choices.
    help_out = subprocess.run(
        [sys.executable, os.path.join(PROJECT_ROOT, "gaze_mouse.py"), "--help"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert "--engine {builtin,eyegestures}" in help_out, (
        f"--help missing engine flag; got:\n{help_out}"
    )

    # Importing gaze_mouse must NOT pull in eyeGestures (lazy import).
    probe = (
        "import sys; import gaze_mouse; "
        "sys.exit(0 if 'eyeGestures' not in sys.modules else 1)"
    )
    rc = subprocess.run(
        [sys.executable, "-c", probe], cwd=PROJECT_ROOT,
    ).returncode
    assert rc == 0, "importing gaze_mouse eagerly imported eyeGestures (not lazy)"

    # The calibration grid helper yields normalized [0,1] points.
    import gaze_mouse  # noqa: E402

    grid = gaze_mouse.eyegestures_grid(5, 5)
    assert len(grid) == 25, f"expected 25 grid points, got {len(grid)}"
    for x, y in grid:
        assert 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0, f"grid point out of range: {(x, y)}"
    corners = {(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)}
    grid_set = {(x, y) for x, y in grid}
    assert corners <= grid_set, f"grid missing corners: {corners - grid_set}"

    # EyeGestures_v3 constructs from the installed (GPLv3) package, no camera.
    from eyeGestures import EyeGestures_v3  # noqa: E402

    tracker = EyeGestures_v3()
    assert tracker is not None
    # The builder used by the app should also succeed.
    built = gaze_mouse.build_eyegestures()
    assert built is not None

    print("SMOKE PASS")
    print("  --help lists --engine {builtin,eyegestures}")
    print("  gaze_mouse import is lazy (eyeGestures not imported at module load)")
    print(f"  calibration grid: {len(grid)} normalized points, corners present")
    print("  EyeGestures_v3 constructed from the installed package")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
