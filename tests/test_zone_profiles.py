"""Validated TP/SL zones: replay, OOS validation, safety rails, streak promotion."""
import asyncio
import random

import pytest

import alert_advisor as adv
import brain as brain_mod
import threshold_engine as engine


@pytest.fixture(autouse=True)
def _fixed_bracket(monkeypatch):
    # fixed bracket: SL 2%, TP 4%
    monkeypatch.setattr(engine.cfg, "OUTCOME_MAE_LOSS_PCT", 2.0, raising=False)
    monkeypatch.setattr(engine.cfg, "OUTCOME_RR_TARGET", 2.0, raising=False)
    engine.clear_ev_first_cache()


def _timeout_rows(n=100, mfe_rng=None, mae_rng=(0.4, 1.6), seed=3, pair="BTCUSD",
                  alert="ppo_cross_up", d="buy", adx=25.0):
    """Trades that never reach the fixed 2%/4% bracket (exit at horizon close).
    By default favourable excursion is inversely related to adverse excursion
    (shallow pullback -> big run), as in real winning setups; pass mfe_rng for
    an independent uniform MFE instead."""
    rng = random.Random(seed)
    rows, ts = [], 1_700_000_000
    for _ in range(n):
        pm = rng.uniform(-0.6, 1.0)
        mae = rng.uniform(*mae_rng)
        mfe = rng.uniform(*mfe_rng) if mfe_rng else max(0.3, 3.5 - 1.3 * (mae - 0.4) + rng.gauss(0, 0.15))
        rows.append({
            "pair": pair, "alert_key": alert, "direction": d, "adx_val": adx, "entry_ts": ts,
            "mae": mae / 100.0, "mfe": mfe / 100.0,
            "pct_move": pm, "net_pnl_pct": pm - 0.18, "win": pm > 0.18,
            "outcome_reason": "no_hit", "tp_first": None,
        })
        ts += 900
    return rows


def test_replay_semantics():
    cost = 0.18
    r = {"direction": "buy", "mae": 0.005, "mfe": 0.03, "pct_move": 0.4}
    assert engine.zone_replay_pnl(r, 1.0, 2.5, cost) == pytest.approx(2.5 - cost)      # target only
    assert engine.zone_replay_pnl(dict(r, mae=0.012), 1.0, 2.5, cost) == pytest.approx(-1.0 - cost)  # both -> stop
    assert engine.zone_replay_pnl(dict(r, mfe=0.01), 1.0, 2.5, cost) == pytest.approx(0.4 - cost)    # timeout, buy
    assert engine.zone_replay_pnl(dict(r, mfe=0.01, direction="sell"), 1.0, 2.5, cost) == pytest.approx(-0.4 - cost)
    assert engine.zone_replay_pnl({"direction": "buy", "mae": None, "mfe": 0.01}, 1.0, 2.5, cost) is None


def test_better_zone_passes_oos_validation():
    out = engine.zone_candidates(_timeout_rows(150))
    c = out["candidates"]["alertdir:ppo_cross_up|buy"]
    assert c["passed"], c["reasons"]
    assert c["delta_ev"] >= 0.05 and c["p_better"] >= 0.80
    assert 1.0 <= c["sl_pct"] <= 4.0 and 2.0 <= c["tp1_pct"] <= 8.0


def test_no_edge_is_not_promoted():
    """Independent MAE/MFE: replayed zone does not beat the fixed bracket."""
    rows = _timeout_rows(100, mfe_rng=(1.6, 3.6))
    c = engine.zone_candidates(rows)["candidates"]["alertdir:ppo_cross_up|buy"]
    assert not c["passed"] and c["delta_ev"] < 0.05


def test_borderline_evidence_waits_for_a_bigger_holdout():
    """At n=100 the 32-row holdout cannot reach the confidence bar, so the gate waits."""
    c = engine.zone_candidates(_timeout_rows(100))["candidates"]["alertdir:ppo_cross_up|buy"]
    assert not c["passed"] and c["reasons"] == ["not_significantly_better"]


def test_thin_samples_make_no_candidates():
    assert engine.zone_candidates(_timeout_rows(40))["candidates"] == {}


def test_safety_rail_rejects_a_too_tight_stop():
    rows = _timeout_rows(100, mfe_rng=(4.1, 4.5), mae_rng=(0.16, 0.25))
    c = engine.zone_candidates(rows)["candidates"]["alertdir:ppo_cross_up|buy"]
    assert not c["passed"] and "sl_outside_deviation_rail" in c["reasons"]


def test_promotion_needs_consecutive_passes_and_failure_demotes():
    cand = {"k": {"bucket": "k", "passed": True, "sl_pct": 1.4, "tp1_pct": 2.8, "tp2_pct": 3.4, "n": 100}}
    promoted, streaks = engine.zone_promote(cand, {}, 2)
    assert promoted == {} and streaks == {"k": 1}
    promoted, streaks = engine.zone_promote(cand, streaks, 2)
    assert "k" in promoted and promoted["k"]["streak"] == 2
    failing = {"k": dict(cand["k"], passed=False)}
    promoted, streaks = engine.zone_promote(failing, streaks, 2)
    assert promoted == {} and streaks == {}                       # demoted, streak reset
    promoted, _ = engine.zone_promote({}, {"k": 5}, 2)           # vanished bucket
    assert promoted == {}


def test_lookup_prefers_most_specific_and_ignores_unknown_regime():
    zones = {
        "alertdir:ppo_cross_up|buy": {"sl_pct": 1.0, "tp1_pct": 2.0, "tp2_pct": 3.0, "n": 100},
        "alertreg:ppo_cross_up|buy|trending": {"sl_pct": 1.1, "tp1_pct": 2.2, "tp2_pct": 3.3, "n": 80},
        "leafreg:BTCUSD|ppo_cross_up|buy|trending": {"sl_pct": 1.2, "tp1_pct": 2.4, "tp2_pct": 3.6, "n": 60},
    }
    blob = {"median_adx": 20.0, "zones": zones}
    z = engine.zone_lookup(blob, "BTCUSD", "ppo_cross_up", "buy", 30.0)
    assert z["bucket"].startswith("leafreg:")
    assert engine.zone_lookup(blob, "ETHUSD", "ppo_cross_up", "buy", 30.0)["bucket"].startswith("alertreg:")
    assert engine.zone_lookup(blob, "ETHUSD", "ppo_cross_up", "buy", 10.0)["bucket"].startswith("alertdir:")
    assert engine.zone_lookup(blob, "BTCUSD", "ppo_cross_up", "buy", None)["bucket"].startswith("alertdir:")
    assert engine.zone_lookup(blob, "BTCUSD", "rsi_up", "buy", 30.0) is None
    assert engine.zone_lookup({}, "BTCUSD", "ppo_cross_up", "buy", 30.0) is None


def test_advisor_prefers_validated_zone_and_labels_it():
    tq = {"trade_plan": {"sl_suggested_pct": 1.0, "tp1_suggested_pct": 2.0, "tp2_suggested_pct": 3.0, "n": 500},
          "trade_zone": {"sl_pct": 1.4, "tp1_pct": 2.8, "tp2_pct": 3.4, "n": 80}}
    plan = adv._plan([tq])
    assert plan["validated"] and plan["sl"] == 1.4
    assert "validated zone" in adv._plan_line(plan, 1.0, 2.0)
    plain = adv._plan([{"trade_plan": tq["trade_plan"]}])
    assert not plain["validated"] and "validated" not in adv._plan_line(plain, 1.0, 2.0)


def _engine_with(blob):
    eng = brain_mod.BrainEngine.__new__(brain_mod.BrainEngine)

    async def load_quality():
        return blob

    async def none_async(*a, **k):
        return None

    eng._load_quality_inputs = load_quality
    eng._load_calibration_curve = none_async
    eng._load_market_state_model = none_async
    eng._load_ml_calibration_curve = none_async
    return eng


def _blob():
    ev = {"p_ev_positive": 0.95, "net_ev": 0.6, "ev_p5": 0.1, "n": 300, "oos_validated": True}
    zone = {"median_adx": 20.0, "zones": {
        "alertdir:ppo_cross_up|buy": {"sl_pct": 1.4, "tp1_pct": 2.8, "tp2_pct": 3.4, "n": 100}}}
    return {"ev_by_alert": {"ppo_cross_up": ev}, "regime_info": None, "hierarchical_leaves": {},
            "hierarchical_median_adx": None, "zone_profiles": zone}


def _tq(eng):
    return asyncio.run(eng.get_trade_quality("BTCUSD", "ppo_cross_up", "buy", 80.0, adx_val=30.0))


def test_dispatch_attaches_zone_only_in_live_mode(monkeypatch):
    eng = _engine_with(_blob())
    for mode, expect in (("live", True), ("advisory", False), ("off", False)):
        monkeypatch.setattr(brain_mod.cfg, "ZONE_MODE", mode, raising=False)
        assert ("trade_zone" in _tq(eng)) is expect
