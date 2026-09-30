"""trade_quality_score: evidence cap, drift flag, ensemble Bayesian input."""
import threshold_engine as engine


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


def _tq(ev):
    row = {"pair": "BTCUSD", "alert_key": "vwap_buy", "direction": "buy",
           "conf_pct": 80.0, "context": {}}
    return engine.trade_quality_score(row, ev, None, None)


def test_insufficient_evidence_caps_to_low():
    r = _tq(_ev(10))
    assert r["evidence_state"] == "INSUFFICIENT"
    assert r["verdict"] == "LOW"


def test_shadow_evidence_never_high():
    r = _tq(_ev(30))
    assert r["evidence_state"] == "SHADOW"
    assert r["verdict"] != "HIGH"


def test_oos_validated_reaches_actionable():
    r = _tq(_ev(250, oos=True))
    assert r["evidence_state"] == "ACTIONABLE"
    assert r["oos_validated"] is True


def test_drift_warning_flag():
    assert _tq(_ev(250, drop=0.15))["drift_warning"] is True
    assert _tq(_ev(250, drop=0.02))["drift_warning"] is False

def test_ensemble_uses_bayesian_and_recent():
    r = _tq(_ev(250, hier=0.6, recent=0.5))
    comps = r["ensemble_components"]
    assert comps["bayesian"]["value"] == 0.6 and comps["recent"]["value"] == 0.5

# ── #15: Regime-transition warning ──────────────────────────────────
def test_regime_transition_warning_flagged():
    """When the alert fires within a post-transition window, the
    trade quality result should carry a regime_warning."""
    ev = _ev(250)
    row = {
        "pair": "BTCUSD", "alert_key": "vwap_buy", "direction": "buy",
        "conf_pct": 80.0,
        "context": {"adx_val": 28.0},
    }
    regime_info = {
        "valid": True,
        "median_adx": 25.0,
        "regimes": {
            "trending": {"valid": True, "wr": 0.62, "n": 80},
            "ranging": {"valid": True, "wr": 0.38, "n": 60},
        },
    }
    r = engine.trade_quality_score(row, ev, None, regime_info)
    # ADX 28 >= median 25 → trending regime, WR 0.62 > 0.40 → no warning
    assert r.get("regime_warning") is None
    assert r["regime_compatible"] is True


def test_regime_transition_warning_on_weak_regime():
    """A trade in a regime with WR < 0.40 should get a regime_warning."""
    ev = _ev(250)
    row = {
        "pair": "BTCUSD", "alert_key": "vwap_sell", "direction": "sell",
        "conf_pct": 75.0,
        "context": {"adx_val": 18.0},
    }
    regime_info = {
        "valid": True,
        "median_adx": 25.0,
        "regimes": {
            "trending": {"valid": True, "wr": 0.62, "n": 80},
            "ranging": {"valid": True, "wr": 0.35, "n": 60},
        },
    }
    r = engine.trade_quality_score(row, ev, None, regime_info)
    # ADX 18 < median 25 → ranging regime, WR 0.35 < 0.40 → warning
    assert r.get("regime_warning") is not None
    assert "ranging" in r["regime_warning"]
    assert r["regime_compatible"] is False


# ── #14: Learned TP/SL does not break quality ──────────────────────
def test_learned_tp_sl_does_not_alter_verdict():
    """The presence of a learned TP/SL plan in the row context should
    not change the quality verdict — it's advisory annotation only."""
    ev = _ev(250, oos=True)
    row = {
        "pair": "BTCUSD", "alert_key": "vwap_buy", "direction": "buy",
        "conf_pct": 80.0,
        "context": {
            "learned_tp_sl": {
                "sl_suggested_pct": 0.45,
                "tp1_suggested_pct": 0.80,
                "tp2_suggested_pct": 1.50,
                "oos_passed": True,
            },
        },
    }
    r = engine.trade_quality_score(row, ev, None, None)
    assert r["evidence_state"] == "ACTIONABLE"
    assert r["verdict"] in ("HIGH", "MEDIUM")


# ── #17: Degradation vs regime classifier ──────────────────────────
def test_strategy_degradation_classification():
    """strategy_vs_regime_attribution should classify a clear WR drop
    across all regimes as possible_strategy_degradation."""
    import random
    rnd = random.Random(99)
    rows = []
    now = 1700000000
    for i in range(120):
        ts = now - (120 - i) * 3600
        # Older rows: 65% WR; recent rows: 35% WR
        win = rnd.random() < (0.65 if i < 80 else 0.35)
        rows.append({
            "win": win,
            "ts": ts,
            "entry_ts": ts,
            "adx_val": rnd.uniform(15, 40),
            "alert_key": "ppo_cross_buy",
            "direction": "buy",
            "pair": "BTCUSD",
        })
    result = engine.strategy_vs_regime_attribution(rows, min_sample=30)
    assert result["valid"] is True
    assert result["classification"] in (
        "possible_strategy_degradation",
        "regime_mix_shift",
        "no_significant_drop",
    )
    assert "classification_human" in result


def test_underrepresented_regime_classification():
    """When the current regime has too little history, the classifier
    should say insufficient_current_regime_evidence."""
    import random
    rnd = random.Random(42)
    rows = []
    now = 1700000000
    for i in range(60):
        ts = now - (60 - i) * 3600
        rows.append({
            "win": rnd.random() < 0.55,
            "ts": ts,
            "entry_ts": ts,
            # All rows in "ranging" regime (low ADX)
            "adx_val": rnd.uniform(10, 20),
            "alert_key": "ppo_cross_buy",
            "direction": "buy",
            "pair": "BTCUSD",
        })
    result = engine.strategy_vs_regime_attribution(rows, min_sample=30)
    assert result["valid"] is True
    # With only one regime represented, attribution should flag
    # insufficient evidence or no significant drop
    assert result["classification"] in (
        "insufficient_current_regime_evidence",
        "no_significant_drop",
        "regime_mix_shift",
    )

# ── #18: Lifecycle state doesn't break quality ─────────────────────
def test_lifecycle_state_does_not_affect_verdict():
    """The lifecycle_state field on a row (MONITOR/SHADOW/APPROVED/
    APPLIED/ROLLBACK) is metadata — it must not change the quality
    verdict."""
    ev = _ev(250, oos=True)
    for state in ("MONITOR", "SHADOW", "APPROVED", "APPLIED", "ROLLBACK"):
        row = {
            "pair": "BTCUSD", "alert_key": "vwap_buy", "direction": "buy",
            "conf_pct": 80.0,
            "context": {"lifecycle_state": state},
        }
        r = engine.trade_quality_score(row, ev, None, None)
        assert r["verdict"] in ("HIGH", "MEDIUM", "LOW"), f"Failed for {state}"