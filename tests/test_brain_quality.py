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
