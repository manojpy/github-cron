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
