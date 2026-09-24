import time

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

def test_calibration_gate_out_of_range_fails_open():
    """A conf_pct outside the curve's covered range must fail open,
    not be silently assigned the nearest bucket.

    build_calibration_curves() always emits boundaries spanning the full
    [0, 100] range by construction (first bucket lo=0.0, last bucket
    hi=100.0), so a curve built through it can never actually go
    out-of-range for a conf_pct in [0, 100). The guard exists for the
    scenario the code comments describe: a persisted curve that is
    malformed or whose covered range has gone stale relative to the
    live conf_pct. We construct that curve by hand to exercise it.
    """
    narrow_curve = {
        "buckets": [
            {
                "lo": 60.0, "hi": 80.0,
                "predicted": 0.70, "observed": 0.70,
                "n": 20, "trusted": True,
                "wilson_lo": 0.50, "wilson_hi": 0.85,
            },
        ],
        "ece": 0.0,
        "n": 20,
    }

    # 5.0 is below the curve's only bucket.
    ok, cal_wr, reason = calibration_gate_decision(
        narrow_curve, 5.0, target_wr=0.55, min_sample=5, slack=0.05,
    )
    assert ok is True
    assert reason == "out_of_range_fail_open"
    assert cal_wr is None

    # 99.9 is above the curve's only bucket.
    ok, cal_wr, reason = calibration_gate_decision(
        narrow_curve, 99.9, target_wr=0.55, min_sample=5, slack=0.05,
    )
    assert ok is True
    assert reason == "out_of_range_fail_open"
    assert cal_wr is None

def test_calibration_persistence_roundtrip():
    """End-to-end: build → JSON-serialize → deserialize → gate lookup.

    This exercises the actual shape the live gate consumes
    (payload['curves']), not the raw dict returned by build_calibration_curves().
    It does not touch Redis; it verifies that whatever brain.py writes to
    Redis and whatever macd_unified.py reads back is what
    calibration_gate_decision() sees at dispatch time.
    """
    import json

    rows = [_row("test_buy", 70.0 + (i % 5), i < 10) for i in range(20)]
    built = build_calibration_curves(rows, bucket_pct=20.0, min_sample=5)

    # brain.py persists the entire `built` dict via json_dumps (orjson).
    # Emulate the exact serialize/deserialize path.
    serialized = json.dumps(built)
    loaded_payload = json.loads(serialized)

    # macd_unified.py does: calibration_curves = payload.get("curves", {}) or {}
    loaded_curves = loaded_payload.get("curves", {}) or {}

    assert "test_buy" in loaded_curves
    assert loaded_payload.get("ece_mean_label") == "mean_per_alert_ece"

    ok, calibrated_wr, reason = calibration_gate_decision(
        loaded_curves["test_buy"],
        72.0,
        target_wr=0.55,
        min_sample=5,
        slack=0.05,
    )

    assert isinstance(ok, bool)
    assert reason in {
        "ok",
        "thin_bucket_fail_open",
        "out_of_range_fail_open",
        "no_curve",
    } 