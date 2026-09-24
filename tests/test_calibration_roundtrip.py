import time

import pytest

from threshold_engine import build_calibration_curves, calibration_gate_decision


def _row(alert_key: str, conf_pct: float, win: bool) -> dict:
    return {
        "alert_key": alert_key,
        "conf_pct": conf_pct,
        "win": win,
    }


def test_calibration_curve_build_and_gate_roundtrip():
    rows = []

    for i in range(20):
        rows.append(
            _row(
                "test_buy",
                70.0 + (i % 5),
                i < 10,
            )
        )

    result = build_calibration_curves(
        rows,
        bucket_pct=20.0,
        min_sample=5,
    )

    assert result["curves"]
    assert result["ece_mean"] is not None
    assert result["ece_mean_label"] == "mean_per_alert_ece"
    assert result["built_at"] <= int(time.time())

    curve = result["curves"]["test_buy"]

    ok, calibrated_wr, reason = calibration_gate_decision(
        curve,
        72.0,
        target_wr=0.55,
        min_sample=5,
        slack=0.05,
    )

    assert calibrated_wr is not None
    assert reason in {
        "ok",
        "thin_bucket_fail_open",
        "no_curve",
    }
    assert isinstance(ok, bool)