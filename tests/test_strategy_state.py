"""#17: strategy broken vs current regime underrepresented vs regime-mix shift."""
from __future__ import annotations

import random
import time

import threshold_engine as engine

DAY = 86400


def _rows(n, wr, adx_lo, adx_hi, days_ago_lo, days_ago_hi, seed):
    rnd = random.Random(seed)
    now = int(time.time())
    return [{
        "win": rnd.random() < wr,
        "adx_val": rnd.uniform(adx_lo, adx_hi),
        "entry_ts": now - int(rnd.uniform(days_ago_lo, days_ago_hi) * DAY),
    } for _ in range(n)]


def _hist(seed=1):
    # Older history: half trending (ADX 30-45, WR 72%), half ranging (ADX 10-22, WR 38%).
    return (_rows(150, 0.72, 30, 45, 20, 60, seed) + _rows(150, 0.38, 10, 22, 20, 60, seed + 1))


def test_insufficient_data():
    assert engine.classify_strategy_state(_hist()[:30])["state"] == "INSUFFICIENT_DATA"


def test_stable_when_no_drop():
    recent = _rows(60, 0.72, 30, 45, 1, 12, 5) + _rows(60, 0.38, 10, 22, 1, 12, 6)
    assert engine.classify_strategy_state(_hist() + recent)["state"] == "STABLE"


def test_strategy_degraded_same_regime_mix_lower_wr():
    # Same 50/50 mix as history, but WR collapsed in BOTH regimes.
    recent = _rows(60, 0.40, 30, 45, 1, 12, 5) + _rows(60, 0.25, 10, 22, 1, 12, 6)
    res = engine.classify_strategy_state(_hist() + recent)
    assert res["state"] == "STRATEGY_DEGRADED", res
    assert res["mix_adjusted_drop"] > 0.06


def test_regime_shift_when_mix_alone_explains_drop():
    # WR per regime unchanged, but recent trades are almost all in the weak ranging regime.
    recent = _rows(10, 0.72, 30, 45, 1, 12, 5) + _rows(110, 0.38, 10, 22, 1, 12, 6)
    res = engine.classify_strategy_state(_hist() + recent)
    assert res["state"] == "REGIME_SHIFT", res


def test_regime_underrepresented_when_history_has_little_of_it():
    # History is almost entirely ranging; recent trades are mostly high-ADX.
    hist = _rows(8, 0.72, 30, 45, 20, 60, 1) + _rows(292, 0.60, 10, 22, 20, 60, 2)
    recent = _rows(90, 0.35, 30, 45, 1, 12, 5) + _rows(30, 0.60, 10, 22, 1, 12, 6)
    res = engine.classify_strategy_state(hist + recent)
    assert res["state"] == "REGIME_UNDERREPRESENTED", res
    assert "trending" in res["underrepresented"]


def test_regime_unknown_without_adx():
    rows = _hist() + _rows(120, 0.30, 0, 0, 1, 12, 5)
    for r in rows:
        r["adx_val"] = None
    assert engine.classify_strategy_state(rows)["state"] == "DEGRADED_REGIME_UNKNOWN"


def test_reasoning_chain_shows_strategy_state():
    import brain_enhanced as be
    from bot_config import cfg
    F = {
        "n": 120, "wr": 0.5, "net_ev": 0.1, "days": 30, "conf": "MODERATE",
        "gate_ok": False, "rows": [], "sessions": [],
        "gate": {"oos_prediction": True, "stability": False},
        "ai": {"strategy_state": {"state": "REGIME_UNDERREPRESENTED"}},
    }
    text = "\n".join(str(p) for p in be._sec_reasoning_chain(F, cfg))
    assert "CUSUM drift active" in text
    assert "underrepresented" in text and "not proof of decay" in text
