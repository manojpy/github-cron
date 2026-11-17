"""Candidate-ledger + statistical-discipline regression tests (audit fixes).

Covers: candidate slot exclusion, mining-population labels, purge/embargo,
hierarchical EV shrinkage, runtime EV decisions, CUSUM persistence/recovery,
and the suppression scorecard's lower-bound / opportunity-cost accounting.
"""
import asyncio
import json

import pytest

import state
import scoreboard
import threshold_engine as engine
from feature_store import chronological_split
from hypothesis_gen import shadow_pool_rows
from threshold_stats import pair_key_ev_table


# ── candidate slot exclusion (state.get_active_shadow_trade) ──────────────

class _FakeRedis:
    def __init__(self, rows):
        self._rows = rows

    async def mget(self, keys):
        return [self._rows.get(k) for k in keys]


class _StubStore:
    """Just enough of RedisStateStore for get_active_shadow_trade()."""
    degraded = False
    _fetch_pending_keys = state.RedisStateStore._fetch_pending_keys
    get_active_shadow_trade = state.RedisStateStore.get_active_shadow_trade

    def __init__(self, rows):
        self._redis = _FakeRedis(rows)
        self._shadow_pending_outcome_keys_by_pair = {"BTCUSD": list(rows)}


def _pending_key(pair, key, ts):
    return f"{state.RedisKeyPrefix.SHADOW_PENDING}{pair}:{key}:{ts}"


def _pending_payload(ts, candidate):
    return json.dumps({"entry_ts": ts, "direction": "buy",
                       "context": {"candidate_only": candidate}})


def test_candidate_never_occupies_single_active_shadow_slot():
    rep = _pending_key("BTCUSD", "ppo_cross_up", 1_700_000_000)
    cand = _pending_key("BTCUSD", "rsi_cross_up", 1_700_000_900)  # newer candle
    sdb = _StubStore({rep: _pending_payload(1_700_000_000, False),
                      cand: _pending_payload(1_700_000_900, True)})
    active = asyncio.run(sdb.get_active_shadow_trade("BTCUSD"))
    assert active is not None
    assert active["alert_key"] == "ppo_cross_up"   # representative, not the newer candidate


def test_only_candidate_rows_means_no_active_shadow_trade():
    cand = _pending_key("BTCUSD", "rsi_cross_up", 1_700_000_900)
    sdb = _StubStore({cand: _pending_payload(1_700_000_900, True)})
    assert asyncio.run(sdb.get_active_shadow_trade("BTCUSD")) is None


# ── mining population labels ──────────────────────────────────────────────

def test_shadow_pool_takes_candidates_and_run_cap_not_gate_rows():
    rows = [
        {"candidate_only": True, "rejection_reason": "delivered_unrecorded_cooldown"},
        {"candidate_only": True},                                  # demoted, no reason
        {"rejection_reason": "run_cap"},
        {"rejection_reason": "confluence_gate"},                   # feature-selecting: excluded
        {"rejection_reason": "verdict_firewall"},                  # outcome-aware: excluded
        {},
    ]
    out = shadow_pool_rows(rows)
    assert len(out) == 3
    assert all(r.get("candidate_only") or r.get("rejection_reason") == "run_cap" for r in out)


# ── purge / embargo ───────────────────────────────────────────────────────

def test_chronological_split_purges_labels_that_reach_into_test():
    step = 900
    rows = [{"ts": 1_700_000_000 + i * step} for i in range(40)]
    train, test = chronological_split(rows, test_frac=0.3, max_horizon=12,
                                      candle_sec=step, embargo_candles=2)
    assert train and test
    first_test = test[0]["ts"]
    reach = (12 + 2) * step
    assert all(r["ts"] + reach <= first_test for r in train)
    assert len(train) + len(test) < len(rows)          # overlapping rows purged, not moved
    assert test == sorted(rows, key=lambda r: r["ts"])[28:]


# ── hierarchical EV table ─────────────────────────────────────────────────

def _ev_row(pair, key, pnl, ts):
    return {"pair": pair, "alert_key": key, "net_pnl_pct": pnl, "entry_ts": ts, "win": pnl > 0}


def test_pair_key_ev_table_shrinks_and_drops_thin_cells():
    ts0 = 1_700_000_000
    rows = [_ev_row("BTCUSD", "ppo_cross_up", -1.0, ts0 + i * 900) for i in range(30)]
    rows += [_ev_row("ETHUSD", "ppo_cross_up", 0.5, ts0 + i * 900) for i in range(60)]
    rows += [_ev_row("SOLUSD", "ppo_cross_up", -2.0, ts0 + i * 900) for i in range(5)]
    table = pair_key_ev_table(rows, cost_pct=0.18, min_pair_n=25, shrink_k=30.0)
    cell = table["ppo_cross_up"]["BTCUSD"]
    assert cell["n"] == 30
    assert -1.0 < cell["ev"] < 0.5                     # shrunk toward the key pool
    assert cell["lo"] < cell["ev"] < cell["hi"]
    assert "SOLUSD" not in table["ppo_cross_up"]       # below min_pair_n


# ── runtime EV decision ───────────────────────────────────────────────────

class _EvStore:
    def __init__(self, table):
        self._table = table

    async def get_pair_key_ev(self):
        return self._table


def _ev_decision(table, tq, want):
    from alerts import _ev_runtime_decision
    return asyncio.run(_ev_runtime_decision(_EvStore(table), "BTCUSD", "ppo_cross_up", tq, want))


def test_ev_decision_blocks_on_deep_negative_pair_cell():
    table = {"ppo_cross_up": {"BTCUSD": {"ev": -0.5, "lo": -0.9, "hi": -0.1, "n": 40}}}
    dec = _ev_decision(table, None, "block")
    assert dec is not None and dec[2] == "pair"


def test_ev_decision_deep_inconclusive_cell_does_not_fall_back_to_pool():
    table = {"ppo_cross_up": {"BTCUSD": {"ev": -0.2, "lo": -0.5, "hi": 0.2, "n": 40}}}
    tq = {"n_oos": 100, "p_ev_positive": 0.05, "net_ev": -0.4}   # pooled says block
    assert _ev_decision(table, tq, "block") is None


def test_ev_decision_keeps_on_positive_pooled_when_pair_thin():
    tq = {"n_oos": 60, "p_ev_positive": 0.9, "net_ev": 0.3}
    dec = _ev_decision({}, tq, "keep")
    assert dec is not None and dec[2] == "pooled"


# ── CUSUM persistence / recovery / dedup rationale ────────────────────────

def test_cusum_state_roundtrip_continues_identically():
    a = engine.CUSUMDetector(target_wr=0.40, drift_delta=0.15, threshold=8.0)
    outcomes = [False, False, True, False, False, False, True, False]
    for w in outcomes[:5]:
        a.update(w)
    b = engine.CUSUMDetector.from_dict(a.to_dict())
    tail = outcomes[5:]
    assert [a.update(w) for w in tail] == [b.update(w) for w in tail]
    assert a.to_dict() == b.to_dict()


def test_cusum_repeated_outcome_would_inflate_without_dedup():
    # Documents WHY the seen-list in brain._check_cusum_drift must exist:
    # the detector has no row identity — feeding one outcome twice counts twice.
    d = engine.CUSUMDetector(target_wr=0.40, drift_delta=0.15, threshold=8.0)
    d.update(False)
    n1, s1 = d.n, d.s_neg
    d.update(False)
    assert d.n == n1 + 1 and d.s_neg > s1


# ── suppression scorecard discipline ──────────────────────────────────────

def _sc_row(src, pnl, ts, reason=None, candidate=False):
    r = {"source": src, "entry_ts": ts, "pnl": pnl, "candidate_only": candidate}
    if reason:
        r["rejection_reason"] = reason
    return r


def test_scorecard_s2_uses_lower_bound_and_reports_realized_cost(monkeypatch):
    monkeypatch.setattr(scoreboard, "replay_plan", lambda r, *a, **k: (0, float(r["pnl"]), 1.0))
    ts0 = 1_700_000_000
    step = 3 * 3600
    rows = [_sc_row("LIVE", 0.5, ts0 + i * step) for i in range(10)]
    sup_pnls = [-0.4] * 6 + [0.6] * 4     # point precision 60% >= 55% target …
    rows += [_sc_row("SHADOW", p, ts0 + (10 + i) * step, reason="win_rate_filter")
             for i, p in enumerate(sup_pnls)]
    rows.append(_sc_row("SHADOW", 9.9, ts0 + 99 * step, reason="win_rate_filter", candidate=True))
    sc = scoreboard.build_suppression_scorecard(
        rows, cost_pct=0.18, fixed={"sl": 1.0, "tp": 2.0, "h": 12}, now_ts=ts0 + 100 * step,
    )
    s2 = next(c for c in sc["criteria_checks"] if c["id"] == "S2")
    assert sc["precision"] == pytest.approx(0.6)
    assert sc["precision_lo"] < 0.55      # … but the Wilson lower bound is not
    assert s2["pass"] is False
    assert sc["opportunity_cost_pct"] == pytest.approx(sum(p for p in sup_pnls if p > 0))
    assert sc["suppressed"]["n"] == 10    # candidate-only row excluded