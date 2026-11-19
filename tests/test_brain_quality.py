from __future__ import annotations
import threshold_engine as engine
import random
import asyncio
import brain as brain_mod
import pytest
import alert_advisor as adv
import time
from threshold_engine import build_calibration_curves, calibration_gate_decision
from types import SimpleNamespace
import state as state_mod
import json
from pathlib import Path

# ======================================================================
# from test_trade_quality_evidence.py
# ======================================================================
"""trade_quality_score: evidence cap, drift flag, ensemble Bayesian input."""


def _ev(n, oos=False, drop=None, hier=None, recent=None):
    d = {"valid": True, "n": n, "p_ev_positive": 0.95, "net_ev": 0.8,
         "ev_p5": 0.1, "wr": 0.7, "oos_validated": oos}
    if drop is not None:
        d["wr_drop_recent"] = drop
    if hier is not None:
        d["hierarchical_wr"] = hier
    if recent is not None:
        d["recent_wr"] = recent
    return d


def _tq__trade_quality_evidence(ev):
    row = {"pair": "BTCUSD", "alert_key": "vwap_buy", "direction": "buy",
           "conf_pct": 80.0, "context": {}}
    return engine.trade_quality_score(row, ev, None, None)


def test_insufficient_evidence_caps_to_low():
    r = _tq__trade_quality_evidence(_ev(10))
    assert r["evidence_state"] == "INSUFFICIENT"
    assert r["verdict"] == "LOW"


def test_shadow_evidence_never_high():
    r = _tq__trade_quality_evidence(_ev(30))
    assert r["evidence_state"] == "SHADOW"
    assert r["verdict"] != "HIGH"


def test_oos_validated_reaches_actionable():
    r = _tq__trade_quality_evidence(_ev(250, oos=True))
    assert r["evidence_state"] == "ACTIONABLE"
    assert r["oos_validated"] is True


def test_drift_warning_flag():
    assert _tq__trade_quality_evidence(_ev(250, drop=0.15))["drift_warning"] is True
    assert _tq__trade_quality_evidence(_ev(250, drop=0.02))["drift_warning"] is False


def test_ensemble_uses_bayesian_and_recent():
    r = _tq__trade_quality_evidence(_ev(250, hier=0.6, recent=0.5))
    comps = r["ensemble_components"]
    assert comps["bayesian"]["value"] == 0.6 and comps["recent"]["value"] == 0.5


# ======================================================================
# from test_regime_gate.py
# ======================================================================
"""Regime-aware live quality gate: sample-gated, OOS-confirmed, restrict-only."""



def _row__regime_gate(key, d, adx, ts, win):
    return {"alert_key": key, "direction": d, "pair": "BTCUSD", "adx_val": adx, "entry_ts": ts,
            "win": win, "outcome_reason": "target_hit" if win else "stop_hit", "tp_first": win}


def _rows__regime_gate(n_bad_trend=120, n_good_range=120, key="ppo_cross_up", d="buy"):
    """Trending (adx 30): ~15% wins. Ranging (adx 10): ~85% wins. Seeded random,
    15-min spaced (uniform outcomes would give zero bootstrap variance)."""
    rng = random.Random(7)
    rows, ts = [], 1_700_000_000
    for i in range(max(n_bad_trend, n_good_range)):
        if i < n_bad_trend:
            rows.append(_row__regime_gate(key, d, 30.0, ts, rng.random() < 0.15))
        if i < n_good_range:
            rows.append(_row__regime_gate(key, d, 10.0, ts + 450, rng.random() < 0.85))
        ts += 900
    return rows


def test_regime_specific_loser_is_blocked_and_winner_untouched():
    out = engine.regime_gate_analysis(_rows__regime_gate())
    assert out["valid"] and out["median_adx"] is not None
    bad = out["segments"]["ppo_cross_up|buy|trending"]
    good = out["segments"]["ppo_cross_up|buy|ranging"]
    assert bad["action"] == "BLOCK" and bad["oos_confirmed_negative"]
    assert good["action"] == "NONE"


def test_thin_samples_are_inert():
    out = engine.regime_gate_analysis(_rows__regime_gate(30, 30))      # below min_n_downgrade=50
    assert out["segments"] == {}
    mid = engine.regime_gate_analysis(_rows__regime_gate(70, 70))      # >=50 but <100: no BLOCK possible
    assert all(s["action"] != "BLOCK" for s in mid["segments"].values())


def test_alert_bad_everywhere_is_not_a_regime_finding():
    rows = _rows__regime_gate(120, 0) + [dict(r, adx_val=10.0, entry_ts=r["entry_ts"] + 450) for r in _rows__regime_gate(120, 0)]
    out = engine.regime_gate_analysis(rows)
    assert all(s["action"] == "NONE" for s in out["segments"].values())


def test_lookup_uses_current_regime_and_ignores_unknown():
    blob = engine.regime_gate_analysis(_rows__regime_gate())
    assert engine.regime_gate_lookup(blob, "ppo_cross_up", "buy", 30.0)["action"] == "BLOCK"
    assert engine.regime_gate_lookup(blob, "ppo_cross_up", "buy", 10.0) is None
    assert engine.regime_gate_lookup(blob, "ppo_cross_up", "buy", None) is None
    assert engine.regime_gate_lookup({}, "ppo_cross_up", "buy", 30.0) is None


def test_apply_is_restrict_only_and_mode_aware():
    seg = {"action": "BLOCK", "regime": "trending", "net_ev": -0.4, "baseline_net_ev": 0.1,
           "p_ev_positive": 0.05, "n": 120}
    r = {"verdict": "HIGH"}
    engine.apply_regime_gate(r, seg, "shadow")
    assert r["verdict"] == "HIGH" and r["regime_gate_shadow"]["action"] == "BLOCK"
    r = {"verdict": "HIGH"}
    engine.apply_regime_gate(r, seg, "off")
    assert r == {"verdict": "HIGH"}
    r = {"verdict": "HIGH"}
    engine.apply_regime_gate(r, seg, "live")
    assert r["verdict"] == "BLOCKED" and r["verdict_before_regime_gate"] == "HIGH"
    assert r["reason"].startswith("regime_gate:")
    down = dict(seg, action="DOWNGRADE")
    for before, after in (("HIGH", "LOW"), ("MEDIUM", "LOW"), ("LOW", "LOW"), ("BLOCKED", "BLOCKED")):
        r = {"verdict": before}
        engine.apply_regime_gate(r, down, "live")
        assert r["verdict"] == after            # never raised, never changed when already lower


def test_trade_quality_score_wires_the_gate():
    ev = {"p_ev_positive": 0.95, "net_ev": 0.6, "ev_p5": 0.1, "n": 300, "oos_validated": True}
    row = {"pair": "BTCUSD", "alert_key": "ppo_cross_up", "direction": "buy",
           "conf_pct": 80.0, "context": {"adx_val": 30.0}}
    base = engine.trade_quality_score(row, ev, None, None)
    assert base["verdict"] in ("HIGH", "MEDIUM")
    seg = {"action": "BLOCK", "regime": "trending", "net_ev": -0.4, "baseline_net_ev": 0.1,
           "p_ev_positive": 0.05, "n": 120}
    gated = engine.trade_quality_score(row, ev, None, None, regime_gate=seg, regime_gate_mode="live")
    assert gated["verdict"] == "BLOCKED" and gated["regime_gate"]["applied"]
    shadow = engine.trade_quality_score(row, ev, None, None, regime_gate=seg, regime_gate_mode="shadow")
    assert shadow["verdict"] == base["verdict"] and "regime_gate_shadow" in shadow


# ── EV cache must be keyed on content, not just (n, first_ts, last_ts) ──

def test_ev_cache_does_not_collide_for_same_size_same_span_row_sets():
    engine.clear_ev_first_cache()
    good = [_row__regime_gate("a", "buy", 20.0, 1_700_000_000 + 900 * i, i % 5 != 0) for i in range(80)]
    bad = [_row__regime_gate("b", "buy", 20.0, 1_700_000_000 + 900 * i, i % 5 == 0) for i in range(80)]
    ev_good = engine.ev_first_objective(good, min_sample=15)
    ev_bad = engine.ev_first_objective(bad, min_sample=15)
    assert ev_good["net_ev"] > 0 > ev_bad["net_ev"]


# ── End to end through BrainEngine.get_trade_quality ──

def _engine_with__regime_gate(blob):
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


def _blob__regime_gate(gate):
    ev = {"p_ev_positive": 0.95, "net_ev": 0.6, "ev_p5": 0.1, "n": 300, "oos_validated": True}
    return {"ev_by_alert": {"ppo_cross_up": ev}, "regime_info": None,
            "hierarchical_leaves": {}, "hierarchical_median_adx": None, "regime_gate": gate}


def _tq__regime_gate(eng, adx):
    return asyncio.run(eng.get_trade_quality(
        "BTCUSD", "ppo_cross_up", "buy", 80.0, adx_val=adx,
    ))


def test_live_verdict_is_blocked_only_in_the_bad_regime(monkeypatch):
    gate = engine.regime_gate_analysis(_rows__regime_gate())
    gate["segments"] = {k: dict(v, action="BLOCK" if v["regime"] == "trending" else "NONE")
                        for k, v in gate["segments"].items()}
    eng = _engine_with__regime_gate(_blob__regime_gate(gate))
    monkeypatch.setattr(brain_mod.cfg, "REGIME_GATE_MODE", "live", raising=False)
    assert _tq__regime_gate(eng, 30.0)["verdict"] == "BLOCKED"
    assert _tq__regime_gate(eng, 30.0)["reason"].startswith("regime_gate:")
    assert _tq__regime_gate(eng, 10.0)["verdict"] in ("HIGH", "MEDIUM")
    assert _tq__regime_gate(eng, None)["verdict"] in ("HIGH", "MEDIUM")      # unknown ADX: gate abstains


def test_shadow_and_off_never_change_the_verdict(monkeypatch):
    gate = engine.regime_gate_analysis(_rows__regime_gate())
    gate["segments"] = {k: dict(v, action="BLOCK") for k, v in gate["segments"].items()}
    eng = _engine_with__regime_gate(_blob__regime_gate(gate))
    monkeypatch.setattr(brain_mod.cfg, "REGIME_GATE_MODE", "shadow", raising=False)
    r = _tq__regime_gate(eng, 30.0)
    assert r["verdict"] in ("HIGH", "MEDIUM") and "regime_gate_shadow" in r
    monkeypatch.setattr(brain_mod.cfg, "REGIME_GATE_MODE", "off", raising=False)
    r = _tq__regime_gate(eng, 30.0)
    assert r["verdict"] in ("HIGH", "MEDIUM") and "regime_gate_shadow" not in r


# ======================================================================
# from test_zone_profiles.py
# ======================================================================
"""Validated TP/SL zones: replay, OOS validation, safety rails, streak promotion."""




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


def _engine_with__zone_profiles(blob):
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


def _blob__zone_profiles():
    ev = {"p_ev_positive": 0.95, "net_ev": 0.6, "ev_p5": 0.1, "n": 300, "oos_validated": True}
    zone = {"median_adx": 20.0, "zones": {
        "alertdir:ppo_cross_up|buy": {"sl_pct": 1.4, "tp1_pct": 2.8, "tp2_pct": 3.4, "n": 100}}}
    return {"ev_by_alert": {"ppo_cross_up": ev}, "regime_info": None, "hierarchical_leaves": {},
            "hierarchical_median_adx": None, "zone_profiles": zone}


def _tq__zone_profiles(eng):
    return asyncio.run(eng.get_trade_quality("BTCUSD", "ppo_cross_up", "buy", 80.0, adx_val=30.0))


def test_dispatch_attaches_zone_only_in_live_mode(monkeypatch):
    eng = _engine_with__zone_profiles(_blob__zone_profiles())
    for mode, expect in (("live", True), ("advisory", False), ("off", False)):
        monkeypatch.setattr(brain_mod.cfg, "ZONE_MODE", mode, raising=False)
        assert ("trade_zone" in _tq__zone_profiles(eng)) is expect


# ======================================================================
# from test_calibration_roundtrip.py
# ======================================================================


def _row__calibration_roundtrip(alert_key: str, conf_pct: float, win: bool) -> dict:
    return {
        "alert_key": alert_key,
        "conf_pct": conf_pct,
        "win": win,
    }


def test_calibration_curve_build_and_gate_roundtrip():
    rows = []

    for i in range(20):
        rows.append(
            _row__calibration_roundtrip(
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
    """... Uses bot_config.json_dumps/json_loads — the same orjson-backed
    helpers brain.py and macd_unified.py actually call — rather than stdlib
    json, so this test keeps matching production if the curve schema ever
    grows a NumPy/datetime field instead of silently drifting from it."""
    from bot_config import json_dumps, json_loads

    rows = [_row__calibration_roundtrip("test_buy", 70.0 + (i % 5), i < 10) for i in range(20)]
    built = build_calibration_curves(rows, bucket_pct=20.0, min_sample=5)

    # brain.py persists the entire `built` dict via json_dumps.
    serialized = json_dumps(built)
    loaded_payload = json_loads(serialized)

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


# ======================================================================
# from test_calibration_online_and_ablation.py
# ======================================================================
"""Regression tests: online calibration keeps history + is cursor-safe,
ablation separates informative votes from noise, reasoning chain reads real data."""




def _rows__calibration_online_and_ablation(n, seed, p_win=0.55, ak="ppo_cross_buy"):
    rnd = random.Random(seed)
    return [
        {"alert_key": ak, "conf_pct": rnd.uniform(50, 90), "win": rnd.random() < p_win}
        for _ in range(n)
    ]


# ── online calibration (pure function) ────────────────────────────────
def test_fold_keeps_history_and_adds_rows():
    base = engine.build_calibration_curves(_rows__calibration_online_and_ablation(300, 1), bucket_pct=20, min_sample=15)
    n_before = base["curves"]["ppo_cross_buy"]["n"]
    n_buckets = len(base["curves"]["ppo_cross_buy"]["buckets"])
    built_at = base["built_at"]

    folded = engine.fold_outcomes_into_calibration(base, _rows__calibration_online_and_ablation(20, 2), min_sample=15)

    c = base["curves"]["ppo_cross_buy"]
    assert folded == 20
    assert c["n"] == n_before + 20                    # history retained
    assert len(c["buckets"]) == n_buckets             # boundaries untouched
    assert sum(b["n"] for b in c["buckets"]) == n_before + 20
    assert base["built_at"] == built_at               # age-based rebuild still fires


def test_fold_equals_full_rebuild_on_observed_rate():
    old, new = _rows__calibration_online_and_ablation(300, 1), _rows__calibration_online_and_ablation(40, 2)
    base = engine.build_calibration_curves(old, bucket_pct=20, min_sample=15)
    engine.fold_outcomes_into_calibration(base, new, min_sample=15)
    folded_wins = sum(round(b["observed"] * b["n"]) for b in base["curves"]["ppo_cross_buy"]["buckets"])
    assert abs(folded_wins - sum(r["win"] for r in old + new)) <= len(base["curves"]["ppo_cross_buy"]["buckets"])


def test_fold_ignores_unknown_alert_key_and_out_of_range():
    base = engine.build_calibration_curves(_rows__calibration_online_and_ablation(200, 1), bucket_pct=20, min_sample=15)
    assert engine.fold_outcomes_into_calibration(base, _rows__calibration_online_and_ablation(5, 3, ak="other_key"), 15) == 0


# ── online calibration (cursor flow against a fake Redis) ─────────────
class _FakeRedis:
    def __init__(self):
        self.streams = {"outcome_log_stream": [], "shadow_log_stream": []}
        self.kv = {}
        self._seq = 0

    def add(self, stream, fields):
        self._seq += 1
        self.streams[stream].append((f"{1000 + self._seq}-0", fields))

    async def xrevrange(self, key, count=1):
        return list(reversed(self.streams[key]))[:count]

    async def xrange(self, key, min="-", max="+", count=None):
        def key_of(i):
            ms, seq = i.split("-")
            return (int(ms), int(seq))
        out = [e for e in self.streams[key] if key_of(e[0]) >= key_of(min)]
        return out[:count] if count else out

    async def get(self, k):
        return self.kv.get(k)

    async def set(self, k, v, ex=None):
        self.kv[k] = v
        return True


class _FakeSDB:
    degraded = False

    def __init__(self, r):
        self._redis = r

    async def _safe_redis_op(self, fn, timeout, label):
        return await fn()


def _outcome(conf, win, ts):
    return {"pair": "BTCUSD", "alert_key": "ppo_cross_buy", "direction": "buy",
            "score": str(conf), "total": "100", "win": "1" if win else "0",
            "pct_move": "0.5", "entry_ts": str(ts)}


def test_cursor_flow_no_double_count(monkeypatch):
    import brain
    from bot_config import cfg, json_loads
    monkeypatch.setattr(brain, "_rc", lambda r: r)
    monkeypatch.setattr(cfg, "ENABLE_CALIBRATION_GATE", True, raising=False)

    r = _FakeRedis()
    now = int(time.time())
    rnd = random.Random(5)
    for i in range(120):
        r.add("outcome_log_stream", _outcome(rnd.uniform(50, 90), rnd.random() < 0.55, now - 100000 + i))

    eng = brain.BaseBrainEngine.__new__(brain.BaseBrainEngine) if hasattr(brain, "BaseBrainEngine") else None
    cls = [c for n, c in vars(brain).items() if isinstance(c, type) and hasattr(c, "maybe_refresh_calibration")][0]
    eng = cls.__new__(cls)
    eng.sdb = _FakeSDB(r)
    eng._calib_cache = None
    eng._calib_cache_ts = 0.0
    import logging
    log = logging.getLogger("t")

    async def scenario():
        await eng.maybe_refresh_calibration(log)                       # full rebuild
        first = json_loads(r.kv[brain.CALIBRATION_CURVES_KEY])
        n0 = first["curves"]["ppo_cross_buy"]["n"]
        assert "stream_cursors" in first

        for i in range(15):                                            # 15 new outcomes
            r.add("outcome_log_stream", _outcome(rnd.uniform(50, 90), rnd.random() < 0.55, now + i))
        await eng.maybe_refresh_calibration(log)                       # fold
        second = json_loads(r.kv[brain.CALIBRATION_CURVES_KEY])
        assert second["curves"]["ppo_cross_buy"]["n"] == n0 + 15
        assert second["built_at"] == first["built_at"]

        await eng.maybe_refresh_calibration(log)                       # nothing new
        third = json_loads(r.kv[brain.CALIBRATION_CURVES_KEY])
        assert third["curves"]["ppo_cross_buy"]["n"] == n0 + 15        # idempotent

    asyncio.run(scenario())


# ── ablation ───────────────────────────────────────────────────────────
def _vote_rows(seed):
    rnd = random.Random(seed)
    rows = []
    for _ in range(400):
        a, b = rnd.random() < 0.5, rnd.random() < 0.5
        rows.append({"win": rnd.random() < (0.75 if a else 0.30),
                     "votes": {"informative": a, "noise": b}})
    return rows


def test_ablation_never_cuts_informative_vote():
    for seed in range(15):
        res = {a["vote"]: a for a in engine.actionable_condition_ablation(
            _vote_rows(seed), min_sample=30, n_permutations=15)}
        assert res["informative"]["action"] == "keep"


def test_ablation_flags_noise_vote_often():
    cuts = sum(
        {a["vote"]: a for a in engine.actionable_condition_ablation(
            _vote_rows(s), min_sample=30, n_permutations=15)}["noise"]["action"] == "reduce_weight"
        for s in range(30)
    )
    assert cuts >= 10


def test_ablation_skips_unmeasurable_votes():
    rows = _vote_rows(1)
    for r in rows:
        r["votes"]["always_on"] = True
    names = {a["vote"] for a in engine.actionable_condition_ablation(rows, min_sample=30)}
    assert "always_on" not in names


# ======================================================================
# from test_weight_path_enforcement.py
# ======================================================================
"""Live CONFLUENCE_WEIGHTS may change only via promotion or rollback."""



class _FakeRedisStateStore:
    _WEIGHT_WRITE_SOURCES = state_mod.RedisStateStore._WEIGHT_WRITE_SOURCES
    degraded = False
    _redis = object()

    def __init__(self):
        self.written = {}

    async def set_metadata(self, key, value, ttl=None):
        self.written[key] = value
        return True

def _call(**kw):
    db = _FakeRedisStateStore()
    ok = asyncio.run(state_mod.RedisStateStore.set_dynamic_weights(db, {"x": 1.0}, **kw))
    return ok, db.written


def test_direct_write_refused_when_enforced(monkeypatch):
    monkeypatch.setattr(state_mod, "cfg", SimpleNamespace(ENFORCE_SINGLE_WEIGHT_PATH=True))
    ok, written = _call()
    assert ok is False and not written


def test_promotion_and_rollback_allowed(monkeypatch):
    monkeypatch.setattr(state_mod, "cfg", SimpleNamespace(ENFORCE_SINGLE_WEIGHT_PATH=True))
    for src in ("promotion", "rollback"):
        ok, written = _call(source=src)
        assert ok is True and "dynamic_weights" in written


def test_direct_write_allowed_when_not_enforced(monkeypatch):
    monkeypatch.setattr(state_mod, "cfg", SimpleNamespace(ENFORCE_SINGLE_WEIGHT_PATH=False))
    ok, _ = _call()
    assert ok is True


# ======================================================================
# from test_e2e_alert_to_brain.py
# ======================================================================
"""Smoke: resolved outcome → JSONL → trade_quality / calibration / ablation."""


def test_alert_pending_resolve_jsonl_brain_quality(tmp_path, monkeypatch):
    """Smoke: resolved outcome → JSONL → trade_quality / calibration / ablation."""
    from outcome_storage import append_outcome_batch, OUTCOME_SCHEMA_VERSION
    import threshold_engine as engine
    import outcome_storage as osmod

    # append_outcome_batch writes under the private module global _OUTCOME_DIR
    # (set at import from cfg). Point it at tmp_path and create subdirs.
    root = str(tmp_path)
    monkeypatch.setattr(osmod, "_OUTCOME_DIR", root)
    for sub in ("outcomes", "shadow", "reports"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)

    row = {
        "schema_version": OUTCOME_SCHEMA_VERSION,
        "pair": "BTCUSD",
        "alert_key": "ppo_cross_buy",
        "direction": "buy",
        "entry_ts": int(time.time()) - 3600,
        "entry_price": 50000.0,
        "win": True,
        "conf_pct": 72.0,
        "confluence_score": 22.0,
        "confluence_total": 30.0,
        "mfe": 0.018,
        "mae": 0.004,
        "net_pnl_pct": 0.012,
        "outcome_reason": "target_hit",
        "votes": {"base_trend": True, "ppo_cross": True, "adx": True},
        "adx_val": 28.0,
        "session": "london",
    }
    append_outcome_batch([row], shadow=False)

    files = list(Path(root).rglob("*.jsonl"))
    assert files, "resolved outcome was not written to JSONL"
    written = json.loads(files[0].read_text(encoding="utf-8").strip().splitlines()[0])
    assert written["alert_key"] == "ppo_cross_buy"
    assert written["win"] is True

    tq = engine.trade_quality_score(
        {
            "pair": "BTCUSD",
            "alert_key": "ppo_cross_buy",
            "direction": "buy",
            "conf_pct": 72.0,
        },
        {
            "p_profit": 0.62,
            "expected_return": 0.008,
            "net_ev": 0.006,
        },
        None,
        None,
    )
    assert "verdict" in tq

    calib = engine.build_calibration_curves([row], bucket_pct=20.0, min_sample=1)
    assert "ppo_cross_buy" in calib["curves"]
    assert calib["curves"]["ppo_cross_buy"]["n"] == 1

    # Online update must ADD to the curve built above, never replace it.
    folded = engine.fold_outcomes_into_calibration(calib, [row], min_sample=1)
    assert folded == 1
    assert calib["curves"]["ppo_cross_buy"]["n"] == 2
    assert calib["curves"]["ppo_cross_buy"]["buckets"][0]["observed"] == 1.0

# ======================================================================
# win-rate posterior, break-even target, override proof, playbook age,
# baseline proof for the live ML-EV gate
# ======================================================================
"""Statistical helpers and live-gate preconditions added in the Brain audit."""


def test_prob_wr_at_least_matches_beta_tail_and_edges():
    f = engine.prob_wr_at_least
    # values cross-checked against numerical integration of the Beta(1+w, 1+l) upper tail
    assert abs(f(6, 14, 0.3933) - 0.21824) < 1e-4
    assert abs(f(9, 11, 0.55) - 0.1841) < 1e-4
    assert abs(f(10, 10, 0.55) - 0.3210) < 1e-4
    assert abs(f(5, 5, 0.5) - 0.5) < 1e-9
    assert f(0, 0, 0.55) == 1.0 and f(3, 3, 0.0) == 1.0 and f(3, 3, 1.0) == 0.0   # "no evidence" never blocks
    assert f(-4, -1, 0.5) == 1.0                                                   # negative counts are clamped
    assert f(10, 5, 0.5) > f(9, 6, 0.5) > f(8, 7, 0.5)                             # monotone in wins
    assert f(8, 7, 0.4) > f(8, 7, 0.5) > f(8, 7, 0.6)                              # and in the target
    assert 0.0 <= f(5000, 5000, 0.5) <= 1.0                                        # no overflow at large n


def test_breakeven_and_gate_target(monkeypatch):
    for k, v in dict(OUTCOME_MAE_LOSS_PCT=1.0, OUTCOME_RR_TARGET=2.0, BRAIN_FEE_PCT=0.0006,
                     BRAIN_SLIPPAGE_PCT=0.0003, MIN_WIN_RATE=0.55, GATE_BREAKEVEN_MARGIN=0.05).items():
        monkeypatch.setattr(brain_mod.cfg, k, v, raising=False)
    assert abs(brain_mod.cfg.breakeven_wr() - 0.3933) < 1e-3          # (1 + 0.18R) / (1 + 2R)
    monkeypatch.setattr(brain_mod.cfg, "GATE_TARGET_MODE", "fixed", raising=False)
    assert brain_mod.cfg.gate_target_wr() == 0.55
    monkeypatch.setattr(brain_mod.cfg, "GATE_TARGET_MODE", "breakeven", raising=False)
    assert abs(brain_mod.cfg.gate_target_wr() - (0.3933 + 0.05)) < 1e-3
    monkeypatch.setattr(brain_mod.cfg, "MIN_WIN_RATE", 0.40, raising=False)       # never above the configured floor
    assert brain_mod.cfg.gate_target_wr() == 0.40
    monkeypatch.setattr(brain_mod.cfg, "OUTCOME_RR_TARGET", 3.0, raising=False)   # a wider target lowers break-even
    assert brain_mod.cfg.breakeven_wr() < 0.3933


class _OverrideSdb:
    degraded = False
    _redis = object()

    def __init__(self, wins, losses, acquire=True):
        self.h, self.acquire = {"wins": str(wins), "losses": str(losses)}, acquire

    async def _safe_redis_op(self, fn, timeout, label):
        return self.h if label.startswith("brain_hiconf") else self.acquire


def _override(monkeypatch, wins, losses, proven=True):
    for k, v in dict(BRAIN_REWARDABLE_MIN_CONFLUENCE_PCT=80.0, BRAIN_REWARDABLE_MIN_SHADOW_SAMPLE=10,
                     BRAIN_REWARDABLE_MIN_SHADOW_WR=0.60, BRAIN_REWARDABLE_REQUIRE_PROVEN_EDGE=proven,
                     OUTCOME_MAE_LOSS_PCT=1.0, OUTCOME_RR_TARGET=2.0, BRAIN_FEE_PCT=0.0006,
                     BRAIN_SLIPPAGE_PCT=0.0003).items():
        monkeypatch.setattr(brain_mod.cfg, k, v, raising=False)
    b = object.__new__(brain_mod.BrainCore)
    b.sdb = _OverrideSdb(wins, losses)
    return asyncio.run(b.check_rewardable_override("vwap_buy", 85.0, 100.0))


def test_rewardable_override_needs_a_proven_edge_not_a_point_estimate(monkeypatch):
    assert _override(monkeypatch, 6, 4) is None                  # 60% of 10: Wilson lower bound 31% < 39.3% break-even
    assert _override(monkeypatch, 7, 3)                          # 70% of 10: lower bound 39.7% just clears 39.3%
    out = _override(monkeypatch, 13, 8)                          # 62% of 21: lower bound 40.9% >= 39.3%
    assert out and "shadow WR 62%" in out and "21 tracked" in out
    assert _override(monkeypatch, 6, 4, proven=False)            # switch off: the old point-estimate rule


def test_override_cooldown_and_thin_samples_still_block(monkeypatch):
    assert _override(monkeypatch, 5, 3) is None                  # below the minimum shadow sample
    for k, v in dict(BRAIN_REWARDABLE_MIN_CONFLUENCE_PCT=80.0, BRAIN_REWARDABLE_MIN_SHADOW_SAMPLE=10,
                     BRAIN_REWARDABLE_MIN_SHADOW_WR=0.60, BRAIN_REWARDABLE_REQUIRE_PROVEN_EDGE=True).items():
        monkeypatch.setattr(brain_mod.cfg, k, v, raising=False)
    b = object.__new__(brain_mod.BrainCore)
    b.sdb = _OverrideSdb(13, 8, acquire=False)                   # an override already fired inside the cooldown
    assert asyncio.run(b.check_rewardable_override("vwap_buy", 85.0, 100.0)) is None


def test_playbook_age_and_loader(monkeypatch):
    monkeypatch.setattr(brain_mod.cfg, "PLAYBOOK_MODE", "veto", raising=False)
    b = object.__new__(brain_mod.BrainCore)
    b._playbook_cache = None
    assert b.playbook_age_hours() is None                        # never loaded this run
    b._playbook_cache = {}
    assert b.playbook_age_hours() == float("inf")                # loaded, but no playbook written yet
    b._playbook_cache = {"generated_ts": time.time() - 7200}
    assert 1.99 < b.playbook_age_hours() < 2.01
    monkeypatch.setattr(brain_mod.cfg, "PLAYBOOK_MODE", "off", raising=False)
    assert b.playbook_age_hours() is None                        # playbook not in use: nothing to warn about

    calls = []

    async def fake_load():
        calls.append(1)
        b._playbook_cache = {"generated_ts": time.time() - 3600}
    b._playbook_cache = None
    b._load_playbook_blob = fake_load
    monkeypatch.setattr(brain_mod.cfg, "PLAYBOOK_MODE", "veto", raising=False)
    asyncio.run(b.ensure_playbook_loaded())                      # a quiet run: no alert has loaded it yet
    assert calls == [1] and 0.99 < b.playbook_age_hours() < 1.01
    monkeypatch.setattr(brain_mod.cfg, "PLAYBOOK_MODE", "off", raising=False)
    asyncio.run(b.ensure_playbook_loaded())
    assert calls == [1]                                          # off: no Redis read

    async def boom():
        raise RuntimeError("redis down")
    b._load_playbook_blob = boom
    monkeypatch.setattr(brain_mod.cfg, "PLAYBOOK_MODE", "veto", raising=False)
    asyncio.run(b.ensure_playbook_loaded())                      # never raises


def test_baseline_proof_gates_the_live_ml_model(monkeypatch):
    from brain_helpers import baseline_proof_ok
    cfg_ = brain_mod.cfg
    monkeypatch.setattr(cfg_, "ML_BASELINE_PROOF_MAX_AGE_HOURS", 72.0, raising=False)
    monkeypatch.setattr(cfg_, "ML_REQUIRE_BASELINE_PROOF", True, raising=False)
    now = time.time()
    ok = {"ev_baseline_proof": {"status": "validated", "at": now - 3600}}
    assert baseline_proof_ok(ok) is True
    assert baseline_proof_ok({"ev_baseline_proof": {"status": "not_better", "at": now}}) is False
    assert baseline_proof_ok({"ev_baseline_proof": {"status": "validated", "at": now - 80 * 3600}}) is False   # stale
    assert baseline_proof_ok({"ev_baseline_proof": {"status": "validated", "at": now + 3600}}) is False        # from the future
    assert baseline_proof_ok({"ev_baseline_proof": {"status": "validated"}}) is False                            # no timestamp
    assert baseline_proof_ok({}) is False and baseline_proof_ok(None) is False
    assert baseline_proof_ok({"ev_baseline_proof": {"status": "validated", "at": "garbage"}}) is False
    monkeypatch.setattr(cfg_, "ML_REQUIRE_BASELINE_PROOF", False, raising=False)
    assert baseline_proof_ok(None) is True                      # switch off: the old behaviour

# ======================================================================
# statistical core: action gate, walk-forward purge/embargo, CUSUM, online challenger
# ======================================================================
def _sc_cfg(monkeypatch, **extra):
    """Fixed bracket (1% stop, RR 2, 0.18% round-trip cost) so every figure is hand-checkable."""
    vals = dict(OUTCOME_MAE_LOSS_PCT=1.0, OUTCOME_RR_TARGET=2.0, BRAIN_FEE_PCT=0.0006,
                BRAIN_SLIPPAGE_PCT=0.0003, OUTCOME_LOOKAHEAD_CANDLES=12, OUTCOME_FILL_DELAY_CANDLES=1,
                BRAIN_EV_GATE_P_THRESHOLD=0.85, BRAIN_EV_GATE_P5_FLOOR=-0.10,
                BRAIN_CUSUM_TARGET_WR=0.40, BRAIN_CUSUM_DRIFT_DELTA=0.15, BRAIN_CUSUM_THRESHOLD=8.0,
                CONFLUENCE_MIN_PCT=60.0, CONFLUENCE_MIN_ABS_SCORE=18.0)
    vals.update(extra)
    for k, v in vals.items():
        monkeypatch.setattr(brain_mod.cfg, k, v, raising=False)


def _sc_row(ts, win, score=5.0, pair="BTCUSD", key="vwap_buy", **kw):
    r = {"pair": pair, "alert_key": key, "direction": "buy", "entry_ts": int(ts), "score": score,
         "win": bool(win), "outcome_reason": "target_hit" if win else "stop_hit",
         "pct_move": 2.0 if win else -1.0, "votes": {"a": True}}
    r.update(kw)
    return r


def _sc_book(n=600, p_hi=0.65, p_lo=0.20, step=600, end_ago=10 * 86400, seed=3):
    """n resolved trades ending `end_ago` seconds ago; high scores (>=5) win more often."""
    rnd = random.Random(seed)
    t0 = int(time.time()) - end_ago - n * step
    rows = []
    for i in range(n):
        sc = rnd.uniform(0, 10)
        rows.append(_sc_row(t0 + i * step, rnd.random() < (p_hi if sc >= 5 else p_lo), score=sc))
    return rows


# ── rolling_walk_forward ───────────────────────────────────────────────
def test_walk_forward_needs_enough_rows_and_valid_folds(monkeypatch):
    _sc_cfg(monkeypatch)
    short = engine.rolling_walk_forward(_sc_book(n=50))
    assert short == {"valid": False, "error": "insufficient_data", "n": 50}
    # Every train row's outcome window spills past the cut -> nothing trainable -> no valid fold.
    none = engine.rolling_walk_forward(_sc_book(), lookahead_sec=10 ** 9)
    assert none["valid"] is False and none["error"] == "too_few_valid_folds" and none["n_folds"] == 0


def test_walk_forward_purges_train_and_embargoes_test(monkeypatch):
    import threshold_validation as tv
    _sc_cfg(monkeypatch)
    rows = _sc_book()
    ordered = sorted(rows, key=lambda r: r["entry_ts"])
    fold = len(ordered) // 6
    cuts = [ordered[fold * (i + 1)]["entry_ts"] for i in range(5)]
    look, emb = (1 + 12 + 1) * 900, 900
    seen_train, seen_test = [], []
    real_rec, real_ev = tv.recommend_threshold, tv.ev_and_kelly_for

    def spy_rec(train, **kw):
        seen_train.append(train)
        return real_rec(train, **kw)

    def spy_ev(test, *a, **kw):
        seen_test.append(test)
        return real_ev(test, *a, **kw)

    monkeypatch.setattr(tv, "recommend_threshold", spy_rec)
    monkeypatch.setattr(tv, "ev_and_kelly_for", spy_ev)
    out = tv.rolling_walk_forward(rows, n_folds=5)
    assert out["valid"] and seen_train and seen_test
    for train in seen_train:
        top = max(r["entry_ts"] for r in train)
        cut = min(c for c in cuts if c > top)                    # the fold this train set belongs to
        assert top + look < cut - emb                            # purge: no label reaches into the test period
        assert len(train) < fold * (cuts.index(cut) + 1)         # ...and some rows really were dropped
    for test in seen_test:
        low = min(r["entry_ts"] for r in test)
        cut = max(c for c in cuts if c <= low)
        assert low >= cut + emb                                  # embargo: first 15 minutes of the fold skipped


def test_walk_forward_pass_rate_is_the_share_of_positive_folds(monkeypatch):
    _sc_cfg(monkeypatch)
    good = engine.rolling_walk_forward(_sc_book(), n_folds=5)
    assert good["valid"] and good["oos_ev_mean"] > 0
    assert good["p_ev_positive"] == good["positive_fold_rate"] == 1.0
    # A book that loses after costs (WR 20% < 39.3% break-even) can never reach the 0.70 gate.
    bad = engine.rolling_walk_forward(_sc_book(p_hi=0.2, p_lo=0.2, seed=4), n_folds=5)
    assert (not bad["valid"]) or bad["p_ev_positive"] < 0.70


# ── _action_gate_check ────────────────────────────────────────────────
class _AuditOK:
    def can_run(self, name):
        return True, ""


def _gate(monkeypatch, rows=None, oos=0.90, ev=(0.95, 0.20), **kw):
    """Run the gate with the OOS and EV layers pinned, so each test isolates one layer."""
    import brain_enhanced as be
    _sc_cfg(monkeypatch)
    monkeypatch.setattr(be, "get_audit", lambda: _AuditOK())
    monkeypatch.setattr(engine, "rolling_walk_forward",
                        lambda rows, **k: {"valid": oos is not None, "p_ev_positive": oos})
    monkeypatch.setattr(engine, "ev_first_objective",
                        lambda rows, **k: {"valid": True, "p_ev_positive": ev[0], "ev_p5": ev[1]}
                        if ev else {"valid": False})
    return be.BrainEngineV2._action_gate_check(rows if rows is not None else _sc_book(n=150), **kw)


def test_action_gate_all_layers_pass_on_a_profitable_stable_book(monkeypatch):
    import brain_enhanced as be
    _sc_cfg(monkeypatch)
    monkeypatch.setattr(be, "get_audit", lambda: _AuditOK())
    g = be.BrainEngineV2._action_gate_check(_sc_book(), active_drift_keys=[])   # real OOS + EV, nothing pinned
    layers = ("data_quality", "oos_prediction", "profitability", "stability", "risk", "execution")
    assert {k: g[k] for k in layers} == dict.fromkeys(layers, True)
    assert g["actionable"] is True


def test_action_gate_data_and_oos_layers(monkeypatch):
    assert _gate(monkeypatch)["actionable"] is True                              # baseline for the cases below
    thin = _gate(monkeypatch, rows=_sc_book(n=99))
    assert thin["data_quality"] is False and thin["actionable"] is False         # 99 rows < 100
    assert _gate(monkeypatch, rows=_sc_book(n=100))["data_quality"] is True      # 100 is enough
    assert _gate(monkeypatch, oos=0.70)["oos_prediction"] is True                # boundary is >= 0.70
    low = _gate(monkeypatch, oos=0.69)
    assert low["oos_prediction"] is False and low["actionable"] is False
    assert _gate(monkeypatch, oos=None)["oos_prediction"] is False               # no valid walk-forward = no pass


def test_action_gate_profitability_thresholds(monkeypatch):
    assert _gate(monkeypatch, ev=(0.85, -0.09))["profitability"] is True         # p >= 0.85 and p5 > -0.10
    assert _gate(monkeypatch, ev=(0.84, 0.20))["profitability"] is False
    assert _gate(monkeypatch, ev=(0.95, -0.10))["profitability"] is False        # p5 must be strictly above the floor
    g = _gate(monkeypatch, ev=None)
    assert g["profitability"] is False and g["actionable"] is False              # no valid EV = not profitable


def test_action_gate_stability_prefers_the_persisted_alarm_set(monkeypatch):
    drift = [{"type": "cusum_drift", "alert": "vwap_buy"}]
    assert _gate(monkeypatch, active_drift_keys=["vwap_buy"])["stability"] is False
    # Authoritative: an empty persisted set means "no alarm" even if this report's recs mention drift.
    assert _gate(monkeypatch, active_drift_keys=[], recommendations=drift)["stability"] is True
    # Without the persisted set, fall back to this cycle's recommendation list.
    assert _gate(monkeypatch, recommendations=drift)["stability"] is False
    assert _gate(monkeypatch, recommendations=[{"type": "other"}])["stability"] is True
    assert _gate(monkeypatch)["stability"] is True


def test_action_gate_execution_needs_nonzero_costs(monkeypatch):
    assert _gate(monkeypatch)["execution"] is True
    monkeypatch.setattr(brain_mod.cfg, "BRAIN_FEE_PCT", 0.0, raising=False)
    import brain_enhanced as be
    g = be.BrainEngineV2._action_gate_check(_sc_book(n=150))
    assert g["execution"] is False and g["actionable"] is False


def test_action_gate_risk_trips_on_recent_streak_and_fails_closed_on_error(monkeypatch):
    now = int(time.time())
    book = _sc_book(n=150) + [_sc_row(now - 3600 + i * 60, False) for i in range(6)]   # 6 straight losses, last hour
    g = _gate(monkeypatch, rows=book)
    assert g["risk"] is False and g["risk_consecutive_losses"] == 6 and g["actionable"] is False
    assert _gate(monkeypatch)["risk"] is True                                          # old book: nothing in the window

    class _Boom:
        def __init__(self, **kw):
            pass

        def evaluate(self, rows):
            raise RuntimeError("redis down")

    monkeypatch.setattr(engine, "KillSwitch", _Boom)
    import brain_enhanced as be
    g = be.BrainEngineV2._action_gate_check(_sc_book(n=150))
    assert g["risk"] is False and "fail-closed" in g["risk_reason"] and g["actionable"] is False


# ── _shadow_weight_check ──────────────────────────────────────────────
def _shadow(n_a, n_b, win_a, win_b, move_a=2.0, move_b=2.0):
    rows = [{"entry_ts": 1000 + i, "votes": {"a": True, "b": False}, "win": win_a, "pct_move": move_a}
            for i in range(n_a)]
    rows += [{"entry_ts": 2000 + i, "votes": {"a": False, "b": True}, "win": win_b, "pct_move": move_b}
             for i in range(n_b)]
    return rows


def test_shadow_weight_check_vetoes_worse_weights_and_passes_thin_or_equal(monkeypatch):
    import brain_enhanced as be
    _sc_cfg(monkeypatch)
    check = be.BrainEngineV2._shadow_weight_check
    cur, new = {"a": 20.0, "b": 1.0}, {"a": 1.0, "b": 20.0}      # current keeps the 'a' rows, proposal keeps 'b'
    ok, why = check(_shadow(20, 20, True, False), cur, new)
    assert ok is False and "shadow WR 100%→0%" in why
    ok, why = check(_shadow(20, 20, True, True, move_a=2.0, move_b=0.5), cur, new)   # same WR, much lower EV
    assert ok is False and "shadow EV" in why
    ok, why = check(_shadow(20, 20, True, False), cur, cur)
    assert ok is True and "stable" in why
    ok, why = check(_shadow(10, 10, True, False), cur, new)      # fewer than 15 kept rows: cannot veto
    assert ok is True and "too thin" in why


# ── CUSUMDetector ─────────────────────────────────────────────────────
def test_cusum_detects_edge_decay_and_round_trips_its_state():
    from threshold_validation import CUSUMDetector
    det = CUSUMDetector(target_wr=0.40, drift_delta=0.15, threshold=8.0)
    fired = [det.update(False) for _ in range(30)]
    assert fired.index(True) == 24 and not any(fired[:24])        # 25th straight loss: 25 x 0.325 = 8.125 > 8
    ok = CUSUMDetector(target_wr=0.40, drift_delta=0.15, threshold=8.0)
    assert not any(ok.update(i % 2 == 0) for i in range(400))     # a 50% book sits above the 40% reference
    assert ok.status()["s_neg"] < 8.0 and ok.s_pos > 0            # only the decay side can alarm
    clone = CUSUMDetector.from_dict(det.to_dict())
    assert clone.to_dict() == det.to_dict() and clone.status()["drift_detected"] is True
    legacy = CUSUMDetector.from_dict({"s_neg": 1.0, "n": 4})      # old state without settings: old defaults
    assert (legacy.mu, legacy.delta, legacy.h) == (0.55, 0.10, 2.0)


# ── _check_cusum_drift ────────────────────────────────────────────────
class _CusumSdb:
    """Persists (watermark, state) per key the way load/save_cusum_bulk do, minus Redis."""
    def __init__(self, fail_load=False):
        self.store, self.fail_load, self.saves = {}, fail_load, 0

    async def load_cusum_bulk(self, keys):
        if self.fail_load:
            return None
        return {k: self.store.get(k, (0, None)) for k in keys}

    async def save_cusum_bulk(self, items):
        self.saves += 1
        for k, st, wm in items:
            self.store[k] = (wm, json.loads(json.dumps(st)))
        return True


def _cusum_engine(sdb):
    b = object.__new__(brain_mod.BrainCore)
    b.sdb, b._cusum_detectors = sdb, {}
    return b


def _cusum_rows(n, key="vwap_buy", pair="BTCUSD", t0=100000, win=False):
    return [{"pair": pair, "alert_key": key, "entry_ts": t0 + i * 900, "win": win} for i in range(n)]


def test_cusum_feeds_each_outcome_once_across_replays_and_restarts(monkeypatch):
    _sc_cfg(monkeypatch)
    sdb = _CusumSdb()
    rows = _cusum_rows(10)
    eng = _cusum_engine(sdb)
    assert asyncio.run(eng._check_cusum_drift(rows)) == []
    wm, st = sdb.store["vwap_buy"]
    assert st["n"] == 10 and wm == rows[-1]["entry_ts"]
    assert st["seen"] == sorted(f"BTCUSD|{r['entry_ts']}" for r in rows)
    asyncio.run(eng._check_cusum_drift(rows))                                    # same window, same process
    assert sdb.store["vwap_buy"][1]["n"] == 10
    asyncio.run(_cusum_engine(sdb)._check_cusum_drift(rows))                     # same window, fresh process
    assert sdb.store["vwap_buy"][1]["n"] == 10
    asyncio.run(_cusum_engine(sdb)._check_cusum_drift(rows + _cusum_rows(3, t0=200000)))
    assert sdb.store["vwap_buy"][1]["n"] == 13                                   # only the 3 new ones


def test_cusum_late_resolving_trade_on_another_pair_is_not_dropped(monkeypatch):
    _sc_cfg(monkeypatch)
    sdb = _CusumSdb()
    a, b = _cusum_rows(1, pair="BTCUSD"), _cusum_rows(1, pair="ETHUSD")          # same entry_ts, different pair
    assert a[0]["entry_ts"] == b[0]["entry_ts"]
    asyncio.run(_cusum_engine(sdb)._check_cusum_drift(a))                         # ETH has not resolved yet
    asyncio.run(_cusum_engine(sdb)._check_cusum_drift(a + b))                     # it resolves a run later
    assert sdb.store["vwap_buy"][1]["n"] == 2                                     # a timestamp-only watermark would give 1


def test_cusum_alarm_fires_once_threshold_crossed_and_stops_consuming(monkeypatch):
    _sc_cfg(monkeypatch)
    sdb = _CusumSdb()
    alerts = asyncio.run(_cusum_engine(sdb)._check_cusum_drift(_cusum_rows(30)))
    assert [a["type"] for a in alerts] == ["cusum_drift"] and alerts[0]["n"] == 25
    st = sdb.store["vwap_buy"][1]
    assert st["n"] == 25 and len(st["seen"]) == 25                                # break: rows 26-30 stay unconsumed
    assert st["s_neg"] > st["h"] == 8.0


def test_cusum_legacy_watermark_state_and_setting_rebase(monkeypatch):
    _sc_cfg(monkeypatch)
    rows = _cusum_rows(6)                                                         # ts 100000 .. 104500
    # Legacy state (no seen-list): everything at or below the watermark counts as consumed.
    sdb = _CusumSdb()
    sdb.store["vwap_buy"] = (rows[2]["entry_ts"], {"mu": 0.40, "delta": 0.15, "h": 8.0, "s_pos": 0.0,
                                                   "s_neg": 1.0, "n": 3})
    asyncio.run(_cusum_engine(sdb)._check_cusum_drift(rows))
    assert sdb.store["vwap_buy"][1]["n"] == 3 + 3                                 # rows 3..5 only
    # the row sitting exactly on the watermark counts as consumed and is remembered, so it is never re-fed
    assert sdb.store["vwap_buy"][1]["seen"] == sorted(f"BTCUSD|{r['entry_ts']}" for r in rows)
    # State built under the old 55% / h=2 settings is re-based once and its alarm cleared.
    sdb2 = _CusumSdb()
    sdb2.store["vwap_buy"] = (rows[0]["entry_ts"], {"mu": 0.55, "delta": 0.10, "h": 2.0, "s_pos": 0.0,
                                                    "s_neg": 5.0, "n": 50, "seen": [f"BTCUSD|{rows[0]['entry_ts']}"]})
    asyncio.run(_cusum_engine(sdb2)._check_cusum_drift(rows[:2]))                 # one new loss
    st = sdb2.store["vwap_buy"][1]
    assert (st["mu"], st["delta"], st["h"]) == (0.40, 0.15, 8.0)
    assert st["n"] == 51 and st["s_neg"] == pytest.approx(0.325)                  # cleared, then +0.325


def test_cusum_skips_the_cycle_when_redis_read_fails(monkeypatch):
    _sc_cfg(monkeypatch)
    sdb = _CusumSdb(fail_load=True)
    assert asyncio.run(_cusum_engine(sdb)._check_cusum_drift(_cusum_rows(30))) == []
    assert sdb.saves == 0 and sdb.store == {}                                     # no fake-zero watermark replay


# ── online challenger ─────────────────────────────────────────────────
def _champ(trained_at=1000, beta=None):
    return {"valid": True, "trained_at": trained_at, "feature_names": ["vote:a"], "scaler": None,
            "beta": beta if beta is not None else [0.0] * 7}      # intercept + vote:a + 4 sessions + direction


def _orow(ts, win):
    return {"entry_ts": int(ts), "win": bool(win), "votes": {"a": True}, "session": "ny", "direction": "buy"}


def test_online_challenger_requires_a_valid_champion():
    from threshold_models import online_challenger_step
    out = online_challenger_step(None, None, [_orow(5000, True)], 100000)
    assert out == {"state": None, "summary": {"status": "no_champion"}}
    out = online_challenger_step(None, {"valid": False, "beta": [0.0] * 7, "feature_names": ["vote:a"]},
                                 [_orow(5000, True)], 100000)
    assert out["summary"]["status"] == "no_champion"


def test_online_challenger_only_learns_from_resolved_trades_and_never_twice():
    from threshold_models import online_challenger_step
    now, lag = 100000, 14400                                      # cutoff 85600
    rows = [_orow(500, True),                                     # before the champion was trained: ignored
            _orow(85600, True), _orow(85601, True)]               # exactly at the cutoff: used; 1s later: too fresh
    out = online_challenger_step(None, _champ(), rows, now, lag_sec=lag)
    s = out["summary"]
    assert s["n_new"] == 1 and s["n_updates"] == 1 and out["state"]["last_ts"] == 85600
    again = online_challenger_step(out["state"], _champ(), rows, now, lag_sec=lag)   # same rows, next run
    assert again["summary"]["n_new"] == 0 and again["summary"]["n_updates"] == 1
    later = online_challenger_step(again["state"], _champ(), rows, now + 3600, lag_sec=lag)
    assert later["summary"]["n_new"] == 1 and later["summary"]["n_updates"] == 2     # the 85601 row has now aged in


def test_online_challenger_verdict_needs_min_eval_and_a_clear_gap():
    from threshold_models import online_challenger_step
    from plan_replay import BLOCK_SEC
    now = 10 ** 7

    def rows(n, f):
        return [_orow(5000 + i * BLOCK_SEC, f(i)) for i in range(n)]

    few = online_challenger_step(None, _champ(), rows(59, lambda i: True), now, min_eval=60)["summary"]
    assert few["status"] == "insufficient" and few["n_eval"] == 59 and "mean_brier_gain" not in few
    exact = online_challenger_step(None, _champ(), rows(60, lambda i: True), now, min_eval=60)["summary"]
    assert exact["n_eval"] == 60 and exact["status"] != "insufficient"            # the 60th pair is enough
    # Champion says ~12%, trades keep winning: the online copy adapts and is measurably better.
    low = _champ(beta=[-2.0] + [0.0] * 6)
    better = online_challenger_step(None, low, rows(80, lambda i: True), now, min_eval=60)["summary"]
    assert better["status"] == "online_better" and better["lo"] > 0 and better["n_eval"] == 80
    # Champion is calibrated at 50% and outcomes alternate: chasing the last result only hurts.
    worse = online_challenger_step(None, _champ(), rows(80, lambda i: i % 2 == 0), now, min_eval=60)["summary"]
    assert worse["status"] == "batch_better" and worse["hi"] < 0


def test_online_challenger_drift_cap_restart_and_retrained_champion():
    from threshold_models import online_challenger_step
    now = 10 ** 6
    rows = [_orow(5000 + i * 900, True) for i in range(40)]
    big = online_challenger_step(None, _champ(beta=[-2.0] + [0.0] * 6), rows, now, max_drift=1e-6)["summary"]
    assert big["max_coef_drift"] > 0 and big["drift_ok"] is False
    first = online_challenger_step(None, _champ(), rows, now)
    assert first["summary"]["drift_ok"] is True
    # Past max_age the online copy is thrown away and restarted from the champion.
    restart = online_challenger_step(first["state"], _champ(), rows, now + 15 * 86400, max_age_sec=14 * 86400)
    assert restart["state"]["started_at"] == now + 15 * 86400
    # A retrained champion is appended to history and reported as the one being served.
    cont = online_challenger_step(first["state"], _champ(trained_at=9000), rows, now + 3600)
    assert len(cont["state"]["history"]) == 2 and cont["summary"]["served_trained_at"] == 9000
    assert cont["summary"]["base_trained_at"] == 1000


def test_online_model_is_served_only_after_promotion():
    from threshold_models import online_challenger_step, online_model_for_serving
    out = online_challenger_step(None, _champ(), [_orow(5000 + i * 900, i % 3 == 0) for i in range(30)], 10 ** 6)
    st = out["state"]
    assert online_model_for_serving(st) is None and online_model_for_serving(None) is None
    st = dict(st, promoted=True, promoted_at=777)
    m = online_model_for_serving(st)
    assert m["provenance"] == "online" and m["trained_at"] == 777 and m["base_trained_at"] == 1000
    assert m["beta"] == st["beta"] and m["beta"] is not st["beta"]
    assert online_model_for_serving(dict(st, beta=[0.0, 1.0])) is None            # length mismatch: refuse

# ======================================================================
# win-rate prior, drift scope, skipped-signal study (Part 5)
# ======================================================================
def _beta_sf(a, b, t, n=40000):
    """P(p >= t) for Beta(a, b), by Simpson integration of the density (independent of the code under test)."""
    import math
    lg = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)

    def f(x):
        return 0.0 if x <= 0 or x >= 1 else math.exp(lg + (a - 1) * math.log(x) + (b - 1) * math.log1p(-x))

    h = (1 - t) / n
    s = f(t) + f(1 - 1e-12) + sum((4 if i % 2 else 2) * f(t + i * h) for i in range(1, n))
    return s * h / 3


def test_shrunk_posterior_matches_numerical_integration_and_pulls_toward_the_pool():
    t = 0.3933
    flat = engine.shrunk_prob_wr_at_least(7, 13, t)
    assert flat == engine.prob_wr_at_least(7, 13, t) == pytest.approx(_beta_sf(8, 14, t), abs=1e-6)
    assert engine.shrunk_prob_wr_at_least(7, 13, t, 0.26, 0.0) == flat      # zero strength: no borrowing
    assert engine.shrunk_prob_wr_at_least(7, 13, t, None, 10.0) == flat     # no pooled rate: no borrowing
    bad = engine.shrunk_prob_wr_at_least(7, 13, t, 0.26, 10.0)              # 3 wins + 7 losses of pseudo-trades
    good = engine.shrunk_prob_wr_at_least(7, 13, t, 0.56, 10.0)             # 6 wins + 4 losses
    assert bad == pytest.approx(_beta_sf(10, 20, t), abs=1e-6) and good == pytest.approx(_beta_sf(13, 17, t), abs=1e-6)
    assert bad < flat < good
    # A key that almost never wins (3%): the win pseudo-count rounds to 0 but is floored at 1 (the flat Beta(1, 1)
    # minimum), so 3 wins + 7 losses become Beta(4, 17). A zero floor would drop one real win from the count.
    assert engine.shrunk_prob_wr_at_least(3, 7, t, 0.03, 10.0) == pytest.approx(_beta_sf(4, 17, t), abs=1e-6)
    # The live decision: 8 wins in 20 (40%) passes the 20% bar alone, but not for a key that wins 14% overall.
    assert engine.shrunk_prob_wr_at_least(8, 12, t) >= 0.20 > engine.shrunk_prob_wr_at_least(8, 12, t, 0.14, 10.0)


def test_curve_pooled_wr_reads_the_whole_key_not_one_bucket():
    assert engine.curve_pooled_wr({"buckets": [{"n": 40, "observed": 0.5}, {"n": 60, "observed": 0.3}]}) == (
        pytest.approx(0.38), 100)
    for bad in (None, {}, {"buckets": []}, {"buckets": [{"n": 0, "observed": 0.5}]}, {"buckets": [{"n": "x"}]}):
        assert engine.curve_pooled_wr(bad) == (None, 0)


def test_action_gate_lists_failed_layers_and_per_key_scope_for_reenables(monkeypatch):
    from brain_recommend_full import drift_scope_allows_reenable as ok
    assert _gate(monkeypatch)["failed_layers"] == []
    assert _gate(monkeypatch, active_drift_keys=["a"])["failed_layers"] == ["stability"]
    assert _gate(monkeypatch, oos=0.5, active_drift_keys=["a"])["failed_layers"] == ["oos_prediction", "stability"]
    gate = {"failed_layers": ["stability"]}
    rec = {"pending_action": "enable", "alert": "vwap_buy"}
    assert ok(rec, gate, ["other_key"], False, "per_key") is True            # drift elsewhere only
    assert ok(rec, gate, ["vwap_buy"], False, "per_key") is False            # its own alarm still blocks
    assert ok(rec, gate, ["other_key"], False, "global") is False            # old behaviour
    assert ok(rec, {"failed_layers": ["stability", "risk"]}, [], False, "per_key") is False   # another layer failed
    assert ok(rec, {"failed_layers": ["oos_prediction"]}, [], False, "per_key") is False
    assert ok(rec, gate, ["other_key"], True, "per_key") is False            # CUSUM state unreadable: stay closed
    assert ok(dict(rec, pending_action="disable"), gate, [], False, "per_key") is False
    assert ok({"alert": "vwap_buy"}, gate, [], False, "per_key") is False         # no pending action at all
    assert ok(rec, dict(gate, disabled=True), [], False, "per_key") is False
    assert ok(rec, gate, None, False, "per_key") is True


def _sig(win, reason=None, rej=None):
    r = {"win": win, "outcome_reason": "target_hit" if win else "stop_hit", "pct_move": 2.0 if win else -1.0}
    if reason:
        r["reason"] = reason
    if rej:
        r["rejection_reason"] = rej
    return r


def test_skipped_signal_study_precision_opportunity_loss_and_followable_gap(monkeypatch):
    from threshold_quality import skipped_signal_study
    _sc_cfg(monkeypatch)                                          # 1% stop, RR 2, 0.18% costs: +1.82 / -1.18
    real = [_sig(True)] * 3 + [_sig(False)] * 7
    shadow = [_sig(True, rej="hypothesis_veto")] * 2 + [_sig(False, rej="hypothesis_veto")] * 8
    cand = [_sig(True, "ignored_cooldown"), _sig(False, "ignored_open_trade"), _sig(True, "skipped_active")]
    st = skipped_signal_study(real, shadow, cand, min_n=20)
    sup = st["suppression"]["hypothesis_veto"]
    assert sup["n"] == 10 and sup["precision"] == pytest.approx(0.8) and sup["opportunity_loss"] == pytest.approx(0.2)
    assert sup["mean_net"] == pytest.approx((2 * 1.82 - 8 * 1.18) / 10)
    assert st["suppression_total"]["n"] == 10
    assert st["recorded"]["net_ev"] == pytest.approx((3 * 1.82 - 7 * 1.18) / 10)
    fo = st["followable"]                                         # skipped_active is NOT something a follower saw
    assert fo["n"] == 12 and fo["extra_n"] == 2 and fo["wins"] == 4
    assert fo["gap_ev"] == pytest.approx((4 * 1.82 - 8 * 1.18) / 12 - (3 * 1.82 - 7 * 1.18) / 10)
    assert st["candidates"]["skipped_active"]["n"] == 1 and st["candidates"]["ignored_cooldown"]["wins"] == 1
    assert skipped_signal_study(real, [], [], min_n=20)["valid"] is False
    assert "followable" not in skipped_signal_study([], [], cand)         # nothing Recorded: no gap to report
    ungraded = skipped_signal_study(real, [], [{"reason": "ignored_cooldown", "win": None}])
    assert ungraded["candidates"] == {} and ungraded["valid"] is False   # an ungraded row is not a result


def test_skipped_signal_report_lines_flag_thin_samples_and_stay_quiet_without_data(monkeypatch):
    from brain_report import _skipped_signal_lines
    from threshold_quality import skipped_signal_study
    _sc_cfg(monkeypatch)
    assert _skipped_signal_lines(None) == [] and _skipped_signal_lines({"valid": False}) == []
    real = [_sig(True)] * 30 + [_sig(False)] * 30
    shadow = [_sig(False, rej="calibration_gate")] * 25 + [_sig(True, rej="run_cap")] * 3
    cand = [_sig(True, "ignored_open_trade")] * 4
    lines = _skipped_signal_lines(skipped_signal_study(real, shadow, cand, min_n=20))
    text = "\n".join(lines)
    assert "Recorded trades: n=60" in text and "Followable book" in text and "overstated" in text
    assert "calibration gate: n=25, would have lost 100%, would have won 0%" in text
    assert "run cap: n=3" in text and "(thin: under 20)" in text.split("run cap: n=3")[1].splitlines()[0]
    assert "ignored open trade: n=4" in text and "(thin: under 20)" in text.split("ignored open trade")[1].splitlines()[0]
    assert "a negative net means the suppression saved money" in text


def test_win_rate_filter_verdict_leans_on_the_alert_keys_overall_rate(monkeypatch):
    import alerts as alerts_mod
    for k, v in dict(WIN_RATE_PRIOR_STRENGTH=10.0, WIN_RATE_PRIOR_MIN_POOLED_N=50,
                     WIN_RATE_FILTER_MIN_PROB=0.20).items():
        monkeypatch.setattr(brain_mod.cfg, k, v, raising=False)
    verdict, t = alerts_mod._posterior_win_rate_verdict, 0.3933

    def curve(wr, n=100):
        return {"buckets": [{"n": n, "observed": wr}]}

    assert verdict(0.40, 20, t, None) is None                                # 8/20, no pool: passes (P 0.55)
    blocked = verdict(0.40, 20, t, curve(0.14))                              # same 8/20, key wins 14% overall
    assert blocked[0] == 0.40 and "pulled toward this alert's overall 14% (n=100)" in blocked[1]
    assert "only 13% chance" in blocked[1] and "need >= 20%" in blocked[1]
    assert verdict(0.40, 20, t, curve(0.14, n=40)) is None                   # curve rests on < 50 outcomes: ignored
    assert verdict(0.25, 20, t, None) is not None                            # 5/20 alone: blocked (P 0.11)
    assert verdict(0.25, 20, t, curve(0.56)) is None                         # ...but not for a key that wins 56% overall
    assert verdict(0.40, 20, t, curve(0.14, n=50)) is not None                # exactly 50 outcomes: the pool counts
    p = engine.shrunk_prob_wr_at_least(8, 12, t, 0.14, 10.0)
    monkeypatch.setattr(brain_mod.cfg, "WIN_RATE_FILTER_MIN_PROB", p, raising=False)
    assert verdict(0.40, 20, t, curve(0.14)) is None                         # exactly on the bar passes (>=)
    monkeypatch.setattr(brain_mod.cfg, "WIN_RATE_FILTER_MIN_PROB", 0.20, raising=False)
    monkeypatch.setattr(brain_mod.cfg, "WIN_RATE_PRIOR_STRENGTH", 0.0, raising=False)
    assert verdict(0.25, 20, t, curve(0.56))[1].count("pulled toward") == 0  # no borrowing, and the note says none
    assert verdict(0.40, 20, t, curve(0.14)) is None                         # strength 0: the old flat prior
    assert verdict(0.25, 20, t, curve(0.56)) is not None



