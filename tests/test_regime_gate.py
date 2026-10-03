"""Regime-aware live quality gate: sample-gated, OOS-confirmed, restrict-only."""
import random
import asyncio
import brain as brain_mod

import threshold_engine as engine


def _row(key, d, adx, ts, win):
    return {"alert_key": key, "direction": d, "pair": "BTCUSD", "adx_val": adx, "entry_ts": ts,
            "win": win, "outcome_reason": "target_hit" if win else "stop_hit", "tp_first": win}


def _rows(n_bad_trend=120, n_good_range=120, key="ppo_cross_up", d="buy"):
    """Trending (adx 30): ~15% wins. Ranging (adx 10): ~85% wins. Seeded random,
    15-min spaced (uniform outcomes would give zero bootstrap variance)."""
    rng = random.Random(7)
    rows, ts = [], 1_700_000_000
    for i in range(max(n_bad_trend, n_good_range)):
        if i < n_bad_trend:
            rows.append(_row(key, d, 30.0, ts, rng.random() < 0.15))
        if i < n_good_range:
            rows.append(_row(key, d, 10.0, ts + 450, rng.random() < 0.85))
        ts += 900
    return rows


def test_regime_specific_loser_is_blocked_and_winner_untouched():
    out = engine.regime_gate_analysis(_rows())
    assert out["valid"] and out["median_adx"] is not None
    bad = out["segments"]["ppo_cross_up|buy|trending"]
    good = out["segments"]["ppo_cross_up|buy|ranging"]
    assert bad["action"] == "BLOCK" and bad["oos_confirmed_negative"]
    assert good["action"] == "NONE"


def test_thin_samples_are_inert():
    out = engine.regime_gate_analysis(_rows(30, 30))      # below min_n_downgrade=50
    assert out["segments"] == {}
    mid = engine.regime_gate_analysis(_rows(70, 70))      # >=50 but <100: no BLOCK possible
    assert all(s["action"] != "BLOCK" for s in mid["segments"].values())


def test_alert_bad_everywhere_is_not_a_regime_finding():
    rows = _rows(120, 0) + [dict(r, adx_val=10.0, entry_ts=r["entry_ts"] + 450) for r in _rows(120, 0)]
    out = engine.regime_gate_analysis(rows)
    assert all(s["action"] == "NONE" for s in out["segments"].values())


def test_lookup_uses_current_regime_and_ignores_unknown():
    blob = engine.regime_gate_analysis(_rows())
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
    good = [_row("a", "buy", 20.0, 1_700_000_000 + 900 * i, i % 5 != 0) for i in range(80)]
    bad = [_row("b", "buy", 20.0, 1_700_000_000 + 900 * i, i % 5 == 0) for i in range(80)]
    ev_good = engine.ev_first_objective(good, min_sample=15)
    ev_bad = engine.ev_first_objective(bad, min_sample=15)
    assert ev_good["net_ev"] > 0 > ev_bad["net_ev"]


# ── End to end through BrainEngine.get_trade_quality ──

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


def _blob(gate):
    ev = {"p_ev_positive": 0.95, "net_ev": 0.6, "ev_p5": 0.1, "n": 300, "oos_validated": True}
    return {"ev_by_alert": {"ppo_cross_up": ev}, "regime_info": None,
            "hierarchical_leaves": {}, "hierarchical_median_adx": None, "regime_gate": gate}


def _tq(eng, adx):
    return asyncio.run(eng.get_trade_quality(
        "BTCUSD", "ppo_cross_up", "buy", 80.0, adx_val=adx,
    ))


def test_live_verdict_is_blocked_only_in_the_bad_regime(monkeypatch):
    gate = engine.regime_gate_analysis(_rows())
    gate["segments"] = {k: dict(v, action="BLOCK" if v["regime"] == "trending" else "NONE")
                        for k, v in gate["segments"].items()}
    eng = _engine_with(_blob(gate))
    monkeypatch.setattr(brain_mod.cfg, "REGIME_GATE_MODE", "live", raising=False)
    assert _tq(eng, 30.0)["verdict"] == "BLOCKED"
    assert _tq(eng, 30.0)["reason"].startswith("regime_gate:")
    assert _tq(eng, 10.0)["verdict"] in ("HIGH", "MEDIUM")
    assert _tq(eng, None)["verdict"] in ("HIGH", "MEDIUM")      # unknown ADX: gate abstains


def test_shadow_and_off_never_change_the_verdict(monkeypatch):
    gate = engine.regime_gate_analysis(_rows())
    gate["segments"] = {k: dict(v, action="BLOCK") for k, v in gate["segments"].items()}
    eng = _engine_with(_blob(gate))
    monkeypatch.setattr(brain_mod.cfg, "REGIME_GATE_MODE", "shadow", raising=False)
    r = _tq(eng, 30.0)
    assert r["verdict"] in ("HIGH", "MEDIUM") and "regime_gate_shadow" in r
    monkeypatch.setattr(brain_mod.cfg, "REGIME_GATE_MODE", "off", raising=False)
    r = _tq(eng, 30.0)
    assert r["verdict"] in ("HIGH", "MEDIUM") and "regime_gate_shadow" not in r
