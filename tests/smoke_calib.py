#!/usr/bin/env python3
"""Camera-free unit test for the eye-gaze calibration fit/predict math.

Builds SYNTHETIC gaze->screen correspondences from a KNOWN quadratic ground
truth, fits a model with fit_calibration, and asserts predict_calibration
recovers held-out points within a small tolerance. Also checks poly_features
length and ordering.

Run:
    .venv/bin/python tests/smoke_calib.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gaze_mouse import (  # noqa: E402
    CALIB_FEATURE_ORDER,
    fit_calibration,
    poly_features,
    predict_calibration,
)


# Known quadratic ground-truth maps gaze (gx, gy) -> screen (sx, sy). The model
# fits exactly this feature space, so a correct fit recovers it to numerical
# precision.
COEF_X = [120.0, 900.0, 30.0, 40.0, -15.0, 60.0]   # for screen_x
COEF_Y = [80.0, 25.0, 700.0, -20.0, 35.0, -50.0]   # for screen_y


def ground_truth(gx: float, gy: float) -> tuple[float, float]:
    feats = poly_features(gx, gy)
    sx = sum(f * c for f, c in zip(feats, COEF_X))
    sy = sum(f * c for f, c in zip(feats, COEF_Y))
    return sx, sy


def main() -> int:
    # poly_features ordering / length.
    feats = poly_features(0.3, -0.2)
    assert len(feats) == 6, f"poly_features must be length 6, got {len(feats)}"
    assert len(CALIB_FEATURE_ORDER) == 6, "feature order constant must be length 6"
    gx, gy = 0.3, -0.2
    expected = [1.0, gx, gy, gx * gx, gy * gy, gx * gy]
    assert feats == expected, f"feature ordering mismatch: {feats} != {expected}"

    # Build a gaze grid (training points) avoiding the held-out probe points.
    train = []
    grid = [-0.4, -0.2, 0.0, 0.2, 0.4]
    for ix in grid:
        for iy in grid:
            train.append(((ix, iy), ground_truth(ix, iy)))

    model = fit_calibration(train)
    assert model["feature_order"] == CALIB_FEATURE_ORDER
    assert model["points"] == len(train)
    assert "timestamp" in model and isinstance(model["timestamp"], str)

    # Held-out probe points (not on the training grid).
    held_out = [(-0.3, 0.1), (0.15, -0.35), (0.05, 0.27), (-0.1, -0.05)]
    max_err = 0.0
    for gx, gy in held_out:
        px, py = predict_calibration(model, gx, gy)
        sx, sy = ground_truth(gx, gy)
        err = max(abs(px - sx), abs(py - sy))
        max_err = max(max_err, err)
        assert err < 1e-3, (
            f"prediction off at ({gx}, {gy}): got ({px:.3f}, {py:.3f}) "
            f"expected ({sx:.3f}, {sy:.3f}), err={err:.4f}"
        )

    print("SMOKE PASS")
    print(f"  trained on {len(train)} points, feature order {model['feature_order']}")
    print(f"  held-out max abs px error: {max_err:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
