import time
from threshold_engine import (
    KillSwitch, portfolio_heat_check, build_calibration_curves,
    calibration_gate_decision, fill_reconciliation,
)


def _row(win, pct=0.5, ts=None, ak="ppo_signal_up", conf=75.0,
         pair="ETHUSD", direction="buy", **kw):
    base = {
        "pair": pair, "alert_key": ak, "direction": direction,
        "score": conf * 0.3, "total": 30.0, "conf_pct": conf,
        "win": win, "pct_move": pct if win else -pct,
        "entry_ts": ts or time.time(),
    }
    base.update(kw)
    return base


def test_kill_switch_streak():
    now = time.time()
    rows = [_row(True, ts=now - 3600 * 5)] + [
        _row(False, ts=now - 600 * i) for i in range(5, -1, -1)
    ]
    ks = KillSwitch(max_consecutive_losses=6, max_drawdown_pct=99.0)
    state = ks.evaluate(rows, now_ts=now)
    assert state["tripped"] and "consecutive" in state["reason"]


def test_kill_switch_drawdown():
    now = time.time()
    rows = [_row(False, pct=1.0, ts=now - 600 * i) for i in range(4)]
    ks = KillSwitch(max_consecutive_losses=99, max_drawdown_pct=3.0)
    assert ks.evaluate(rows, now_ts=now)["tripped"]


def test_kill_switch_neutral_on_wins():
    now = time.time()
    rows = [_row(True, ts=now - 600 * i) for i in range(6)]
    assert not KillSwitch().evaluate(rows, now_ts=now)["tripped"]


def test_portfolio_heat_net_cap():
    opens = [{"pair": f"P{i}", "direction": "buy"} for i in range(3)]
    v = portfolio_heat_check(opens, "P9", "buy", max_concurrent=10, max_net_directional=3)
    assert v["blocked"]
    # a sell reduces the net — must pass
    assert not portfolio_heat_check(opens, "P9", "sell", max_concurrent=10, max_net_directional=3)["blocked"]


def test_portfolio_heat_concurrent_cap_and_dup():
    opens = [{"pair": f"P{i}", "direction": "buy"} for i in range(6)]
    assert portfolio_heat_check(opens, "PX", "sell", max_concurrent=6)["blocked"]
    assert portfolio_heat_check(opens[:2], "P0", "sell", max_concurrent=6)["blocked"]


def test_calibration_gate_blocks_miscalibrated():
    now = time.time()
    # claims ~75% confluence, actually wins ~15% (2/13 in the 75-80 bucket)
    rows = [_row(i < 4, conf=74.0 + (i % 3), ts=now - i * 60) for i in range(20)]
    calib = build_calibration_curves(rows, bucket_pct=5.0, min_sample=10)
    curve = calib["curves"]["ppo_signal_up"]
    ok, cal_wr, _ = calibration_gate_decision(curve, 75.0, target_wr=0.55, min_sample=10)
    assert not ok and cal_wr < 0.55


def test_calibration_gate_fail_open_on_thin():
    ok, _, reason = calibration_gate_decision({"buckets": []}, 75.0, 0.55)
    assert ok and reason == "no_curve"


def test_fill_reconciliation_estimated_tier():
    now = time.time()
    # wins only move 0.6% but target implies 2.0 × 0.5 = 1.0% → leakage
    rows = [_row(True, pct=0.6, ts=now - i * 60) for i in range(15)]
    fr = fill_reconciliation(rows, rr_target=2.0, stop_pct=0.5, min_sample=10)
    assert fr["valid"] and not fr["measured"]
    assert fr["realized_slippage_per_side"] > 0.0003
    assert fr["ev_overstated_pct_per_trade"] > 0


def test_fill_reconciliation_measured_tier():
    now = time.time()
    rows = [
        _row(True, ts=now - i * 60, direction="buy",
             signal_price=100.0, fill_price=100.08)  # 8bps of buy slippage
        for i in range(12)
    ]
    fr = fill_reconciliation(rows, min_sample=10)
    assert fr["valid"] and fr["measured"]
    assert abs(fr["realized_slippage_per_side"] - 0.0008) < 1e-6