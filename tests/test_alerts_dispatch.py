"""Consolidated alert / dispatch / brain-filter / state tests.

Original file was assembled from several standalone test modules
(test_alert_advisor, test_alert_dispatch_format, test_dispatch_outcome_ordering,
test_resolution_archive_ordering, test_survival_and_brain_filter_msgs,
test_new_gates) plus extra DLQ-replay and one-trade-per-pair cases.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from types import SimpleNamespace as NS
import numpy as np
import pytest
import alerts as A
import feedback
import outcome_storage
import state
from alert_advisor import AVOID, TAKE, WATCH, advise_pair
from alerts import AlertPayload, dispatch_combined_alerts
from bot_config import BiasContext, cfg
from fetcher import PriceData
from state import RedisStateStore
from threshold_engine import (
    KillSwitch,
    build_calibration_curves,
    calibration_gate_decision,
    fill_reconciliation,
    portfolio_heat_check,
)

LOG = logging.getLogger("t")


# ======================================================================
# from test_alert_advisor.py
# ======================================================================
"""Conviction / verdict logic and the per-pair Telegram body."""


DOWN = NS(up_pct=0.20, down_pct=0.47)
UP = NS(up_pct=0.64, down_pct=0.21)
GOOD_VOTES = {
    "base_trend": True, "dynamic_flow_ribbon": True, "gaussian_trend": True,
    "oi_funding": True, "adx": False,
}


def tq(ev=0.04, p=0.5, shadow=True, regime=("trending", 0.40),
       plan=(1.18, 1.76, 2.67, 0.5, 23), **kw):
    t = {"verdict": "LOW", "p_ev_positive": p, "p_ev_positive_ensemble": p,
         "net_ev": ev, "evidence_state": "SHADOW" if shadow else "ACTIONABLE"}
    if regime:
        t["regime_name"], t["regime_wr"] = regime
    if plan:
        t["trade_plan"] = {
            "sl_suggested_pct": plan[0], "tp1_suggested_pct": plan[1],
            "tp2_suggested_pct": plan[2], "tp_first_rate": plan[3], "n": plan[4],
        }
    t.update(kw)
    return t


def adv(tqs, bias="with", votes=GOOD_VOTES, score=24, total=27, **kw):
    return advise_pair(direction="buy", score=score, total=total, required=21,
                       votes=votes, tqs=tqs, bias=bias, **kw)


def test_negative_ev_is_avoid_even_with_weak_evidence():
    a = adv([tq(ev=-0.07)])
    assert a.verdict == AVOID and a.conviction < 50
    assert "negative" in a.why_line


def test_shadow_evidence_can_never_take():
    a = adv([tq(ev=0.9, p=0.9, regime=("trending", 0.8), plan=(1, 2, 3, 0.9, 50))])
    assert a.verdict != TAKE and a.conviction <= 55


def test_proven_edge_with_bias_is_take():
    a = adv([tq(ev=0.9, p=0.82, shadow=False, regime=("trending", 0.64),
                plan=(1.1, 1.65, 2.5, 0.64, 120), size_hint=1.0)])
    assert a.verdict == TAKE and a.conviction >= 70 and a.confidence == "HIGH"
    assert "Size 1.00" in a.size_line


def test_conflicting_signals_downgrade_take_to_watch():
    kw = dict(tqs=[tq(ev=0.9, p=0.82, shadow=False,
                      regime=("trending", 0.64), plan=(1.1, 1.65, 2.5, 0.64, 120))])
    assert adv(**kw).verdict == TAKE
    assert adv(conflicting=True, **kw).verdict == WATCH


def test_blocked_verdict_is_avoid():
    assert adv([tq(ev=0.5, verdict="BLOCKED")]).verdict == AVOID


def test_against_bias_and_oi_failed_is_avoid():
    v = dict(GOOD_VOTES, oi_funding=False)
    a = adv([tq(ev=0.5, shadow=False, p=0.7)], bias="against", votes=v)
    assert a.verdict == AVOID


def test_no_data_never_invents_a_conviction():
    clean = {k: True for k in GOOD_VOTES}
    a = adv([None], bias="with", votes=clean)
    assert a.conviction is None and a.verdict == WATCH and a.risk_label == "Missing"
    b = adv([None], bias="against")
    assert b.conviction is None and b.verdict == AVOID
    c = adv([None], bias="with", score=17, total=27)          # weak technicals
    assert c.verdict == AVOID and not c.risk_line.startswith(";")


def test_default_plan_when_no_history_and_r_multiples_with_history():
    assert "default plan" in adv([None]).plan_line
    line = adv([tq()]).plan_line
    assert "SL -1.18%" in line and "TP1 +1.76% (1.5R)" in line and "TP2 +2.67% (2.3R)" in line


def test_no_votes_list_in_any_line():
    a = adv([tq()])
    blob = " ".join([a.brain_line, a.edge_line, a.why_line, a.risk_line, a.plan_line])
    assert "Votes" not in blob and "✓" not in blob and "✗" not in blob


def test_bias_alignment():
    assert A._bias_alignment("buy", UP) == "with" and A._bias_alignment("sell", UP) == "against"
    assert A._bias_alignment("sell", DOWN) == "with"
    assert A._bias_alignment("buy", NS(up_pct=0.3, down_pct=0.3)) == "neutral"
    assert A._bias_alignment("buy", None) == "neutral"


def test_price_precision_for_low_priced_pairs():
    # Rules: >=10 → 2dp, >=1 → 3dp, <1 → 4dp
    assert A._format_price(0.0462) == "$0.0462"
    assert A._format_price(0.00731) == "$0.0073"
    assert A._format_price(1.1934) == "$1.193"
    assert A._format_price(10.5) == "$10.50"
    assert A._format_price(108420) == "$108,420.00"


def _msg(**over):
    kw = dict(
        pair="LABUSD", direction="sell", price=0.0462, ts=1790826300, score=24, total=27,
        items=[("🟣▼ VWAP Cross", ""), ("🌊🔴 Dynamic Flow Cross SELL", "")],
        keys=["vwap_down", "dynamic_flow_cross_sell"],
        tq_by_key={"vwap_down": tq(ev=0.04),
                   "dynamic_flow_cross_sell": tq(ev=-0.16, plan=(1.18, 1.27, 2.34, 0.0, 42))},
        votes=GOOD_VOTES, required=21, bias_context=DOWN, logger_pair=LOG,
    )
    kw.update(over)
    return A.build_pair_msg_safe(**kw).replace("\\", "")


def test_message_shape_header_conviction_and_no_votes():
    m = _msg().split("\n")
    assert m[0] == "🟣▼ LABUSD — SELL | $0.0462 - (24/27) 89%"
    assert m[1].startswith("🎯 Conviction ") and m[1].endswith("| ⛔ AVOID")
    assert m[2] == "VWAP Cross, Dynamic Flow Cross SELL"
    assert [x.split(" ")[0] for x in m[3:8]] == ["🧠", "📊", "💡", "⚠️", "🛡"]
    assert "Votes" not in "\n".join(m)


def test_mixed_direction_pair_flags_conflict_and_only_lists_matching_side():
    m = _msg(direction="buy",
             items=[("🟢🔄 Strong Reversal BUY", ""), ("🔴🌀 Fib Pivot Reversal SELL", "")],
             keys=["strong_reversal_buy", "fib_reversal_sell"], tq_by_key={})
    assert "Fib Pivot" not in m.split("\n")[2] and "opposing signal" in m


def test_bad_tq_degrades_instead_of_raising():
    m = _msg(tq_by_key={"vwap_down": {"net_ev": "abc", "trade_plan": {"n": 3}}})
    assert m.startswith("🟣▼ LABUSD") and "🎯 Conviction n/a" in m


def test_formatter_failure_falls_back_to_plain_body(monkeypatch):
    monkeypatch.setattr(A, "advise_pair",
                        lambda **k: (_ for _ in ()).throw(RuntimeError("boom")))
    m = _msg()
    assert "LABUSD" in m and "VWAP Cross" in m and "Conviction" not in m


def test_markdown_v2_special_chars_are_escaped():
    raw = A.build_pair_msg_safe(**{**dict(
        pair="LABUSD", direction="sell", price=0.0462, ts=1790826300, score=24, total=27,
        items=[("🟣▼ VWAP Cross", "")], keys=["vwap_down"], tq_by_key={"vwap_down": tq()},
        votes=GOOD_VOTES, required=21, bias_context=DOWN, logger_pair=LOG)})
    for ch in "().-+|":
        assert raw.count(ch) == raw.count("\\" + ch), ch


# ── bias alignment: follows the dominant bucket and needs a real lead ───────

def _b(up, down, neutral):
    return NS(up_pct=up, down_pct=down, neutral_pct=neutral)


def test_neutral_dominant_bias_is_never_counter_trend():
    # the AAVE 09:30 case: Neutral 37%, up 30%, down 33%
    assert A._bias_alignment("buy", _b(0.30, 0.33, 0.37)) == "neutral"
    assert A._bias_alignment("sell", _b(0.30, 0.33, 0.37)) == "neutral"


def test_thin_lead_is_neutral_and_clear_lead_counts():
    assert A._bias_alignment("buy", _b(0.40, 0.37, 0.23)) == "neutral"      # 3-pt lead
    assert A._bias_alignment("buy", _b(0.53, 0.17, 0.30)) == "with"
    assert A._bias_alignment("sell", _b(0.53, 0.17, 0.30)) == "against"
    assert A._bias_alignment("buy", _b(0.40, 0.37, 0.23), min_edge=0.02) == "with"


# ── unproven-but-strong tier ────────────────────────────────────────────────

CLEAN = {"base_trend": True, "dynamic_flow_ribbon": True, "gaussian_trend": True,
         "oi_funding": True, "adx_strength": False}


def un(bias="neutral", votes=CLEAN, score=26, total=27, **kw):
    return advise_pair(direction="buy", score=score, total=total, required=21,
                       votes=votes, tqs=[None], bias=bias,
                       unproven_take_min_pct=90.0, **kw)


def test_strong_unproven_setup_reads_take_small_size_without_fake_conviction():
    a = un()
    assert a.verdict == TAKE and a.conviction is None
    assert a.label == "✅ TAKE (small size)"
    assert "Size 0.25" in a.size_line and "unproven" in a.size_line
    assert "Risk" == a.risk_label and "weak: trend strength" in a.risk_line


def test_unproven_take_needs_every_condition():
    assert un(score=24, total=27).verdict == WATCH                         # 89% < 90% bar
    assert un(votes=dict(CLEAN, oi_funding=False)).verdict == WATCH        # OI/funding failed
    assert un(conflicting=True).verdict == WATCH                           # opposing signal
    assert un(bias="against", votes=dict(CLEAN, oi_funding=False)).verdict == AVOID
    assert un(bias="against").verdict == WATCH                             # counter-trend, 96%: watch only
    assert un(bias="against", score=24, total=27).verdict == AVOID         # counter-trend, 89%


def test_unproven_take_is_off_by_default_and_with_flag_none():
    a = advise_pair(direction="buy", score=26, total=27, required=21, votes=CLEAN,
                    tqs=[None], bias="with")
    assert a.verdict == WATCH and a.label == "🟡 WATCH"


def test_why_line_says_all_high_weight_checks_pass_instead_of_repeating_top3():
    w = {"base_trend": 3.0, "gaussian_trend": 2.0, "adx_strength": 1.0}
    a = un(votes={"base_trend": True, "gaussian_trend": True, "adx_strength": False},
           vote_weights=w)
    assert "All high-weight checks pass" in a.why_line
    b = un(votes={"base_trend": True, "gaussian_trend": False, "adx_strength": True},
           vote_weights=w)
    assert "Supported by" in b.why_line and "All high-weight" not in b.why_line


# ======================================================================
# from test_alert_dispatch_format.py
# ======================================================================
"""Dispatcher: ordering by confluence %, same-underlying note, bias footer."""


def _bias(up, down, neutral):
    n = 30
    return BiasContext(up_count=round(up * n), down_count=round(down * n),
                       neutral_count=round(neutral * n), total_pairs=n,
                       up_pct=up, down_pct=down, neutral_pct=neutral)


# NOTE: renamed from UP/DOWN to avoid shadowing the SimpleNamespace
# UP/DOWN defined in the advisor section above. See file header.
BIAS_UP = _bias(0.6, 0.2, 0.2)
BIAS_DOWN = _bias(0.2, 0.57, 0.23)

@pytest.fixture(autouse=True)
def _no_run_cap(monkeypatch):
    """The dispatch tests assume no per-run cap (config_macd.json sets one);
    the cap itself is covered by test_run_cap_keeps_strongest_and_shadow_tracks_the_rest."""
    monkeypatch.setattr(cfg, "ALERT_MAX_PAIRS_PER_RUN", 0, raising=False)
    monkeypatch.setattr(cfg, "ALERT_MAX_PER_DIRECTION", 0, raising=False)

class _Tg:
    def __init__(self):
        self.sent = []

    async def send(self, msg):
        self.sent.append(msg)
        return True


class _Sdb:
    async def atomic_batch_update(self, c):
        return True

    async def set_last_processed_candle_ts(self, *a):
        return None

    async def release_recent_alert(self, *a):
        return None


def _p(pair, direction, score, total, verdict=None):
    return AlertPayload(pair_name=pair, direction=direction, score=score, total=total,
                        msg_body=f"BODY-{pair}", dedup_keys=["k"], state_changes=[],
                        budget_count=1, ts=1_790_826_300, alert_keys=["vwap_up"],
                        verdict=verdict)


def _send(payloads, bias):
    tg = _Tg()
    asyncio.run(dispatch_combined_alerts(payloads, tg, bias, _Sdb(), [0], asyncio.Lock(), 50,
                                         logging.getLogger("t")))
    return "\n".join(tg.sent)


def _order(text, names):
    return [n for n in sorted(names, key=lambda n: text.index(f"BODY-{n}"))]


def test_orders_by_confluence_percent_not_raw_score():
    # raw score says A(26/30=87%) > B(25/27=93%); percent must put B first
    ps = [_p("AAAUSD", "buy", 26, 30), _p("BBBUSD", "buy", 25, 27),
          _p("CCCUSD", "sell", 28, 29)]
    text = _send(ps, BIAS_UP)
    assert _order(text, ["AAAUSD", "BBBUSD", "CCCUSD"]) == ["BBBUSD", "AAAUSD", "CCCUSD"]


def test_negative_bias_puts_sells_first():
    ps = [_p("AAAUSD", "buy", 26, 30), _p("CCCUSD", "sell", 20, 29)]
    text = _send(ps, BIAS_DOWN)
    assert _order(text, ["AAAUSD", "CCCUSD"]) == ["CCCUSD", "AAAUSD"]


def test_same_underlying_note_only_on_later_same_direction_member():
    ps = [_p("PAXGUSD", "sell", 24, 27), _p("XAUTUSD", "sell", 22, 27),
          _p("ETHUSD", "sell", 20, 27)]
    text = _send(ps, BIAS_DOWN)
    assert text.count("Same underlying as PAXGUSD") == 1
    assert text.index("Same underlying") > text.index("BODY-XAUTUSD")
    assert text.index("Same underlying") < text.index("BODY-ETHUSD")


def test_opposite_directions_of_a_group_are_not_flagged():
    ps = [_p("PAXGUSD", "sell", 24, 27), _p("XAUTUSD", "buy", 22, 27)]
    assert "Same underlying" not in _send(ps, BIAS_DOWN)


def test_footer_has_candle_open_time_and_bias():
    text = _send([_p("AAAUSD", "buy", 24, 27)], BIAS_DOWN)
    assert "⏰" in text and "📆" in text and "▼57% 🟥" in text

def test_basket_note_when_three_same_direction_takes():
    ps = [_p(n, "buy", 26, 27, "TAKE") for n in ("AAAUSD", "BBBUSD", "CCCUSD")] \
         + [_p("DDDUSD", "buy", 20, 27, "WATCH")]
    text = _send(ps, BIAS_UP)
    assert text.count("size them as ONE basket") == 1 and "3 BUY TAKEs" in text

def test_run_cap_keeps_strongest_and_shadow_tracks_the_rest(monkeypatch):
    monkeypatch.setattr(cfg, "ALERT_MAX_PAIRS_PER_RUN", 3, raising=False)
    monkeypatch.setattr(cfg, "ALERT_MAX_PER_DIRECTION", 2, raising=False)
    seen = []

    def _shadow(name):
        async def _rec(reason):
            seen.append((name, reason))
        return _rec

    ps = [_p("AAAUSD", "buy", 26, 27), _p("BBBUSD", "buy", 25, 27), _p("CCCUSD", "buy", 24, 27),
          _p("DDDUSD", "sell", 20, 27), _p("EEEUSD", "sell", 19, 27)]
    for p in ps:
        p.record_shadow = _shadow(p.pair_name)
    text = _send(ps, BIAS_UP)
    names = ("AAAUSD", "BBBUSD", "CCCUSD", "DDDUSD", "EEEUSD")
    assert [n for n in names if f"BODY-{n}" in text] == ["AAAUSD", "BBBUSD", "DDDUSD"]
    assert sorted(seen) == [("CCCUSD", "run_cap"), ("EEEUSD", "run_cap")]

def _veto_call(alert_key, pb, calls, **over):
    async def _rec(sdb, pair, alerts, ts, px, **kw):
        calls.append((pair, [a[2] for a in alerts], kw.get("block_reason")))
    A._record_counterfactual_block, saved = _rec, A._record_counterfactual_block
    try:
        kw = dict(sdb=None, pair_name="ETHUSD", alert=("t", "e", alert_key), pb=pb,
                  session="asian", conf_pct=70.0, adx=18.0, ts_curr=1_790_826_300,
                  close_curr=100.0, confluence_scores=None, gr=None, context=None,
                  logger_pair=LOG)
        kw.update(over)
        return asyncio.run(A._playbook_veto_drops(**kw))
    finally:
        A._record_counterfactual_block = saved


def test_playbook_veto_modes(monkeypatch):
    avoid = {"restrict": "AVOID"}
    calls = []
    monkeypatch.setattr(cfg, "ALERT_VETO_MODE", "off", raising=False)
    assert _veto_call("pivot_up_R1", avoid, calls) is False and calls == []
    monkeypatch.setattr(cfg, "ALERT_VETO_MODE", "shadow", raising=False)
    assert _veto_call("pivot_up_R1", avoid, calls) is False and calls == []
    monkeypatch.setattr(cfg, "ALERT_VETO_MODE", "live", raising=False)
    assert _veto_call("pivot_up_R1", avoid, calls) is True
    assert calls == [("ETHUSD", ["pivot_up_R1"], "playbook_veto")]


def test_playbook_veto_fails_open(monkeypatch):
    from playbook import PLAYBOOK_SCHEMA, playbook_lookup
    monkeypatch.setattr(cfg, "ALERT_VETO_MODE", "live", raising=False)
    calls = []
    assert _veto_call("pivot_up_R1", None, calls) is False
    assert _veto_call("pivot_up_R1", {"restrict": "WATCH"}, calls) is False
    stale = {"schema": PLAYBOOK_SCHEMA, "generated_ts": 1_000,
             "entries": {"pivot_up_R1|buy": {"restrict": "AVOID"}}}
    entry = playbook_lookup(stale, "pivot_up_R1", "buy", now_ts=1_000 + 40 * 3600, max_age_sec=36 * 3600)
    assert entry is None and _veto_call("pivot_up_R1", entry, calls) is False
    fresh = playbook_lookup(stale, "pivot_up_R1", "buy", now_ts=1_000 + 3600, max_age_sec=36 * 3600)
    assert _veto_call("pivot_up_R1", fresh, calls) is True
    assert calls == [("ETHUSD", ["pivot_up_R1"], "playbook_veto")]


def test_playbook_veto_rule_hit_needs_its_flag(monkeypatch):
    monkeypatch.setattr(cfg, "ALERT_VETO_MODE", "live", raising=False)
    pb = {"avoid_rules": [{"rule": [["session", "eq", "asian"]], "text": "session is asian"}]}
    calls = []
    monkeypatch.setattr(cfg, "ALERT_VETO_ON_RULES", False, raising=False)
    assert _veto_call("dynamic_flow_buy", pb, calls) is False
    monkeypatch.setattr(cfg, "ALERT_VETO_ON_RULES", True, raising=False)
    assert _veto_call("dynamic_flow_buy", pb, calls) is True
    assert _veto_call("dynamic_flow_buy", pb, calls, session="ny") is False
    assert calls == [("ETHUSD", ["dynamic_flow_buy"], "playbook_veto")]


def test_playbook_veto_mixed_keys_on_one_pair(monkeypatch):
    monkeypatch.setattr(cfg, "ALERT_VETO_MODE", "live", raising=False)
    calls = []
    kept = [k for k, pb in (("pivot_up_R1", {"restrict": "AVOID"}),
                            ("vwap_up", None),
                            ("choch_buy", {"restrict": "AVOID"}))
            if not _veto_call(k, pb, calls)]
    assert kept == ["vwap_up"]
    assert [c[1] for c in calls] == [["pivot_up_R1"], ["choch_buy"]]

def test_no_basket_note_below_threshold_or_for_watch():
    two = [_p("AAAUSD", "buy", 26, 27, "TAKE"), _p("BBBUSD", "buy", 25, 27, "TAKE")]
    assert "basket" not in _send(two, BIAS_UP)
    watch = [_p(n, "buy", 26, 27, "WATCH") for n in ("AAAUSD", "BBBUSD", "CCCUSD")]
    assert "basket" not in _send(watch, BIAS_UP)


# ======================================================================
# from test_dispatch_outcome_ordering.py
# ======================================================================
"""Behavioural tests: outcome + ACTIVE state must follow Telegram delivery.

A failed send must NOT leave a recorded trade or an ACTIVE alert state behind
(phantom trade), and a failing post-send hook must not skip the
candle-processed marker (which would allow a duplicate alert next run).
"""


class FakeTelegram:
    def __init__(self, results):
        self.results = list(results)   # one bool per send() call
        self.sent = []

    async def send(self, msg):
        self.sent.append(msg)
        return self.results.pop(0) if self.results else False


class FakeSdb:
    def __init__(self):
        self.state_batches = []
        self.processed = []
        self.released = []

    async def atomic_batch_update(self, changes):
        self.state_batches.append(list(changes))
        return True

    async def set_last_processed_candle_ts(self, pair, ts):
        self.processed.append((pair, ts))

    async def release_recent_alert(self, pair, key):
        self.released.append((pair, key))


def _payload(pair, calls, *, record_raises=False):
    async def record():
        calls.append(pair)
        if record_raises:
            raise RuntimeError("redis down")

    return AlertPayload(
        pair_name=pair, direction="buy", score=5.0, total=10.0,
        msg_body=f"body {pair}", dedup_keys=["k"],
        state_changes=[(f"{pair}:state", "ACTIVE", None)],
        budget_count=1, ts=1_700_000_000,
        alert_keys=["ppo_cross_up"],
        record_win_rate=record, mark_candle_processed=True,
    )


def _run(payloads, telegram, sdb):
    return asyncio.run(dispatch_combined_alerts(
        payloads, telegram, None, sdb, [0], asyncio.Lock(), 50,
        logging.getLogger("test"),
    ))


def test_send_success_records_and_activates_once():
    calls, sdb = [], FakeSdb()
    n = _run([_payload("ETHUSD", calls)], FakeTelegram([True]), sdb)
    assert n == 1
    assert calls == ["ETHUSD"]
    assert sdb.state_batches == [[("ETHUSD:state", "ACTIVE", None)]]
    assert sdb.processed == [("ETHUSD", 1_700_000_000)]


def test_send_failure_creates_no_phantom_trade():
    calls, sdb = [], FakeSdb()
    # combined send fails, individual fallback send fails too
    n = _run([_payload("ETHUSD", calls)], FakeTelegram([False, False]), sdb)
    assert n == 0
    assert calls == []                 # no outcome recorded
    assert sdb.state_batches == []     # alert state NOT activated
    assert sdb.processed == []


def test_post_send_hook_error_still_marks_candle_processed():
    calls, sdb = [], FakeSdb()
    n = _run([_payload("ETHUSD", calls, record_raises=True)], FakeTelegram([True]), sdb)
    assert n == 1
    assert calls == ["ETHUSD"]
    assert sdb.processed == [("ETHUSD", 1_700_000_000)]


# ── Correction 1: pipeline result inspection ──

def test_atomic_batch_update_rejects_per_command_error():
    import state as S
    sdb = object.__new__(S.RedisStateStore)
    sdb.degraded = False
    sdb.expiry_seconds = 0
    sdb.state_prefix = "pair_state:"

    class _Pipe:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        def hset(self, *a, **k): return None
        async def execute(self):
            return [1, RuntimeError("WRONGTYPE")]

    class _R:
        def pipeline(self): return _Pipe()

    sdb._redis = _R()
    ok = asyncio.run(sdb.atomic_batch_update(
        updates=[("ETHUSD:a", "ACTIVE", None), ("ETHUSD:b", "ACTIVE", None)],
    ))
    assert ok is False


def test_atomic_batch_update_groups_hdel_count():
    import state as S
    sdb = object.__new__(S.RedisStateStore)
    sdb.degraded = False
    sdb.expiry_seconds = 0
    sdb.state_prefix = "pair_state:"

    class _Pipe:
        def __init__(self): self.queued = []
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        def __getattr__(self, name):
            def _rec(*a, **k): self.queued.append((name, a))
            return _rec
        async def execute(self):
            return [1] * len(self.queued)

    class _R:
        def __init__(self, p): self._p = p
        def pipeline(self): return self._p

    pipe = _Pipe()
    sdb._redis = _R(pipe)
    ok = asyncio.run(sdb.atomic_batch_update(
        updates=[], deletes=["ETHUSD:a", "ETHUSD:b", "ETHUSD:c"],
    ))
    assert ok is True
    assert sum(1 for n, _ in pipe.queued if n == "hdel") == 1

# ======================================================================
# from test_resolution_archive_ordering.py
# ======================================================================
"""Resolved outcomes must reach the file archive BEFORE the Redis pending key
is deleted. If the archive write fails, the pending key must survive."""


class FakePipe:
    def __init__(self, store, log):
        self.store, self.log, self.ops = store, log, []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def get(self, key):
        self.ops.append(("get", key))

    def __getattr__(self, name):          # hincrby / expire / xadd / delete ...
        def _rec(*args, **kw):
            self.ops.append((name, args))
        return _rec

    async def execute(self):
        self.log.append("redis_execute")
        if any(op[0] == "get" for op in self.ops):
            return ["raw"] * sum(1 for op in self.ops if op[0] == "get")
        if any(op[0] == "delete" for op in self.ops):
            self.store["deleted"] = True
        return []


class FakeRedis:
    def __init__(self, log):
        self.log, self.store = log, {"deleted": False}

    def pipeline(self):
        return FakePipe(self.store, self.log)

    async def mget(self, keys):
        self.log.append("redis_execute")
        return ["raw"] * len(keys)

def _make_store(monkeypatch, log):
    sdb = object.__new__(RedisStateStore)
    sdb.degraded = False
    sdb._redis = FakeRedis(log)
    sdb._run_resolved_total = 0
    sdb._run_archived_total = 0

    async def fake_keys(*a, **k):
        return ["pending:ETHUSD:ppo_cross_up:1700000000"]

    result = {
        "alert_key": "ppo_cross_up", "direction": "buy", "entry_ts": 1_700_000_000,
        "pct_move": 1.5, "win": True, "mae": 0.1, "mfe": 2.0,
        "conf_score": 5.0, "conf_total": 10.0, "conf_votes": {}, "adx_val": 20.0,
    }
    monkeypatch.setattr(sdb, "_fetch_pending_keys", fake_keys)
    monkeypatch.setattr(sdb, "_parse_pending_outcome_row", lambda *a, **k: (result, ""))
    monkeypatch.setattr(cfg, "ENABLE_WIN_RATE_FILTER", True, raising=False)
    monkeypatch.setattr(cfg, "BRAIN_USE_FILE_STORAGE", True, raising=False)
    return sdb


def _resolve(sdb):
    asyncio.run(sdb.resolve_pending_outcomes("ETHUSD", None, 0, logging.getLogger("t")))


def test_archive_written_before_redis_delete(monkeypatch):
    log = []
    sdb = _make_store(monkeypatch, log)
    written = []

    def fake_append(rows, shadow=False):
        log.append("archive_write")
        written.extend(rows)

    monkeypatch.setattr(outcome_storage, "append_outcome_batch", fake_append)
    _resolve(sdb)
    assert log.index("archive_write") < len(log) - 1          # before the write pipeline
    assert log[-1] == "redis_execute" and sdb._redis.store["deleted"] is True
    assert written[0]["_stream_id"] == "ETHUSD:ppo_cross_up:1700000000"


def test_archive_failure_keeps_pending_key(monkeypatch):
    log = []
    sdb = _make_store(monkeypatch, log)

    def boom(rows, shadow=False):
        raise OSError("disk full")

    monkeypatch.setattr(outcome_storage, "append_outcome_batch", boom)
    _resolve(sdb)
    assert sdb._redis.store["deleted"] is False               # pending outcome NOT deleted
    assert sdb._run_resolved_total == 0


# ======================================================================
# from test_survival_and_brain_filter_msgs.py
# ======================================================================
"""#25 survival checklist + #26 BRAIN FILTER Telegram notice
(build_* / _notify_brain_filter)."""


class _Q:
    def __init__(self, ok=True):
        self.sent, self.ok = [], ok

    async def send(self, msg, priority="normal"):
        self.sent.append(msg)
        return self.ok


class _Redis:
    def __init__(self):
        self.keys = set()

    def set(self, key, val, nx=False, ex=None):
        if nx and key in self.keys:
            return None
        self.keys.add(key)
        return True


class _SDB:
    degraded = False

    def __init__(self):
        self._redis = _Redis()

    async def _safe_redis_op(self, fn, timeout, label):
        return fn()


def _notify(q, sdb, key="ppo_cross_buy", gate="calibration gate", tq=None):
    return asyncio.run(A._notify_brain_filter(
        q, sdb, pair_name="BTCUSD", alert_key=key, gate=gate,
        lines=["calibrated WR 38% below 55% floor"], tq=tq, score=22.0, total=30.0,
    ))


def _on(monkeypatch, cap=3):
    monkeypatch.setattr(cfg, "ENABLE_BRAIN_FILTER_TELEGRAM", True, raising=False)
    monkeypatch.setattr(cfg, "BRAIN_FILTER_TELEGRAM_MAX_PER_RUN", cap, raising=False)
    monkeypatch.setattr(A, "_rc", lambda r: r)


TQ = {"verdict": "BLOCKED", "p_ev_positive": 0.41, "net_ev": -0.12,
      "evidence_state": "SHADOW", "n_oos": 38, "drift_warning": True,
      "regime_warning": "ranging regime WR=35%"}

def test_message_has_fields_and_is_markdown_safe():
    msg = A.build_brain_filter_message(
        pair="BTC_USD", alert_key="ppo_cross_buy", gate="calibration gate",
        lines=["x (y)"], tq=TQ, score=22.0, total=30.0,
    )
    for needle in ("BRAIN FILTER", "Signal gates: PASS", "BLOCKED", "evidence SHADOW",
                   "recent WR drift", "ranging regime"):
        assert needle in msg
    assert "P\\(profit\\) 41%" in msg and "netEV \\-0\\.12%" in msg
    assert "BTC\\_USD" in msg and "x \\(y\\)" in msg


def test_off_by_default(monkeypatch):
    monkeypatch.setattr(cfg, "ENABLE_BRAIN_FILTER_TELEGRAM", False, raising=False)
    q = _Q()
    assert _notify(q, _SDB()) is False and q.sent == []


def test_cooldown_dedupes_same_pair_alert_gate(monkeypatch):
    _on(monkeypatch)
    q, sdb = _Q(), _SDB()
    assert _notify(q, sdb) is True
    assert _notify(q, sdb) is False                      # same key inside cooldown
    assert _notify(q, sdb, gate="quality hard block") is True   # different gate
    assert len(q.sent) == 2


def test_per_run_cap(monkeypatch):
    _on(monkeypatch, cap=2)
    q, sdb = _Q(), _SDB()
    results = [_notify(q, sdb, key=f"k{i}") for i in range(4)]
    assert results == [True, True, False, False]


def test_degraded_redis_sends_nothing(monkeypatch):
    _on(monkeypatch)
    q, sdb = _Q(), _SDB()
    sdb.degraded = True
    assert _notify(q, sdb) is False and q.sent == []


def test_never_raises_on_send_failure(monkeypatch):
    _on(monkeypatch)

    class _Boom(_Q):
        async def send(self, msg, priority="normal"):
            raise RuntimeError("telegram down")

    assert _notify(_Boom(), _SDB()) is False


# ======================================================================
# from test_new_gates.py
# ======================================================================


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
    assert not portfolio_heat_check(
        opens, "P9", "sell", max_concurrent=10, max_net_directional=3
    )["blocked"]


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


# ======================================================================
# DLQ replay must restore the side effects of a normal send, and the
# button-free heat gate must see delivered TAKE alerts as open positions.
# ======================================================================

def test_dlq_replay_applies_outcomes_and_state_after_delivery(monkeypatch):
    monkeypatch.setattr(cfg, "ENABLE_TELEGRAM_DLQ", True, raising=False)
    monkeypatch.setattr(cfg, "DRY_RUN_MODE", False, raising=False)
    monkeypatch.setattr(cfg, "TELEGRAM_DLQ_MAX_ITEMS_PER_RUN", 5, raising=False)

    class _DlqSdb(FakeSdb):
        degraded = False

        def __init__(self):
            super().__init__()
            self.parked, self.outcomes, self.deleted = {}, [], []

        async def dlq_push(self, pair, message, ts, *, dedup_keys=None, source="",
                           state_changes=None, outcomes=None):
            self.parked[f"k:{pair}"] = {
                "pair": pair, "ts": int(ts), "message": message,
                "dedup_keys": list(dedup_keys or []),
                "state_changes": [list(c) for c in (state_changes or [])],
                "outcomes": list(outcomes or []), "attempts": 0,
            }
            return True

        async def dlq_list(self, limit=10):
            return list(self.parked.items())

        async def dlq_delete(self, key):
            self.deleted.append(key)
            self.parked.pop(key, None)

        async def dlq_save_entry(self, key, entry):
            if key in self.parked:
                self.parked[key] = entry
            return True

        async def record_pending_outcome(self, **kw):
            self.outcomes.append(kw)
            return True

    sdb = _DlqSdb()
    ts = int(time.time())

    async def record(sink=None):
        row = dict(pair="ETHUSD", alert_key="ppo_cross_up", direction="buy",
                   entry_ts=ts, entry_price=100.0)
        if sink is not None:
            sink.append(row)
        else:
            await sdb.record_pending_outcome(**row)

    p = AlertPayload(
        pair_name="ETHUSD", direction="buy", score=5.0, total=10.0, msg_body="body",
        dedup_keys=["k"], state_changes=[("ETHUSD:state", "ACTIVE", None)],
        budget_count=1, ts=ts, alert_keys=["ppo_cross_up"],
        record_win_rate=record, mark_candle_processed=True,
    )
    # Telegram down for the combined send and the individual fallback: alert is parked.
    n = _run([p], FakeTelegram([False, False]), sdb)
    assert n == 0 and sdb.outcomes == [] and sdb.state_batches == []
    assert len(sdb.parked) == 1

    # Telegram back: the replay delivers AND applies the owed outcome + ACTIVE state.
    asyncio.run(A.replay_telegram_dlq(
        sdb, FakeTelegram([True]), logging.getLogger("test")))
    assert [o["pair"] for o in sdb.outcomes] == ["ETHUSD"]
    assert sdb.state_batches == [[("ETHUSD:state", "ACTIVE", None)]]
    assert sdb.parked == {}


# ── Correction 2: NEEDS_REPAIR save failure stops the pass ──

def test_replay_stops_when_repair_marker_write_fails(monkeypatch, caplog):
    import alerts as A
    monkeypatch.setattr(cfg, "ENABLE_TELEGRAM_DLQ", True, raising=False)
    monkeypatch.setattr(cfg, "DRY_RUN_MODE", False, raising=False)
    monkeypatch.setattr(cfg, "TELEGRAM_DLQ_MAX_ITEMS_PER_RUN", 10, raising=False)
    monkeypatch.setattr(cfg, "TELEGRAM_DLQ_MAX_ATTEMPTS", 1, raising=False)

    class _Sdb:
        degraded = False
        def __init__(self):
            self.parked = {
                "k:AAAUSD": {
                    "pair": "AAAUSD", "ts": 0, "message": "m", "delivered": True,
                    "state_changes": [["AAAUSD:s", "ACTIVE", None]],
                    "outcomes": [], "attempts": 1, "side_tries": 1,
                },
                "k:BBBUSD": {
                    "pair": "BBBUSD", "ts": 0, "message": "m", "delivered": True,
                    "state_changes": [["BBBUSD:s", "ACTIVE", None]],
                    "outcomes": [], "attempts": 1, "side_tries": 1,
                },
            }
            self.sent = []
        async def dlq_list(self, limit=10): return list(self.parked.items())
        async def dlq_save_entry(self, key, entry): return False
        async def dlq_delete(self, key): return True
        async def record_pending_outcome(self, **kw): return True
        async def atomic_batch_update(self, *a, **k): return False

    class _Tg:
        async def send(self, msg): return True

    sdb = _Sdb()
    asyncio.run(A.replay_telegram_dlq(sdb, _Tg(), logging.getLogger("t")))
    # Second entry must not have been processed at all.
    assert "BBBUSD" not in "\n".join(r.message for r in caplog.records)
    assert sdb.parked["k:BBBUSD"]["side_tries"] == 1   # untouched

def test_auto_open_positions_count_only_delivered_takes():
    class _Meta:
        def __init__(self):
            self.d = {}

        async def get_metadata(self, k, timeout=2.0):
            return self.d.get(k)

        async def read_metadata_checked(self, k, timeout=2.0, durable=False):
            return True, self.d.get(k)

        async def set_metadata(self, k, v, timeout=2.0, ttl=None):
            self.d[k] = v

    async def go():
        m, now = _Meta(), time.time()
        ts = int(now) - 900
        await feedback.record_auto_positions(
            m,
            [("BTCUSD", "sell", ts, "TAKE"),
             ("ETHUSD", "sell", ts, "AVOID"),
             ("SOLUSD", "buy", ts, None)],
            verdicts=["TAKE"], max_age_sec=10800, now=now)
        live = await feedback.load_open_positions(m, 10800, now, source="alerts")
        stale = await feedback.load_open_positions(m, 10800, now + 20000, source="alerts")
        taps = await feedback.load_open_positions(m, 10800, now, source="taps")
        return live, stale, taps

    live, stale, taps = asyncio.run(go())
    assert [(p["pair"], p["direction"]) for p in live] == [("BTCUSD", "sell")]
    assert stale == [] and taps == []


# ======================================================================
# one recorded trade per pair until stop/target
# ======================================================================

def _one_trade_price_data(highs, lows):
    n = len(highs)
    ts = np.arange(n, dtype=np.int64) * 900 + 1_000_000
    flat = np.full(n, 100.0)
    return PriceData(ts=ts, open=flat.copy(), high=np.array(highs, float),
                     low=np.array(lows, float), close=flat.copy(), volume=flat.copy())


def _one_trade_row(direction="buy", entry_idx=0):
    return json.dumps({"entry_ts": 1_000_000 + entry_idx * 900, "entry_price": 100.0,
                       "direction": direction})


def _one_trade_store(rows, degraded=False):
    class _P:
        def __init__(self):
            self.keys = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def get(self, k):
            self.keys.append(k)

        async def execute(self):
            return [rows.get(k) for k in self.keys]

    class _R:
        def pipeline(self, transaction=False):
            return _P()

        async def mget(self, keys):
            return [rows.get(k) for k in keys]

    s = object.__new__(state.RedisStateStore)
    s.degraded = degraded
    s._redis = _R()

    async def _keys(*a, **k):
        return list(rows)

    s._fetch_pending_keys = _keys
    return s


def test_early_close_marks_target_stop_and_leaves_open_trades_alone():
    store = object.__new__(state.RedisStateStore)
    f = store._early_close_payload
    # fill at open[1]=100; target 102, stop 99 (defaults 1% risk, 2R)
    quiet = _one_trade_price_data([100, 100.5, 100.5, 100.5], [100, 99.5, 99.5, 99.5])
    assert f(_one_trade_row(), quiet, 3) is None
    up = _one_trade_price_data([100, 100.5, 102.5, 100.5], [100, 99.5, 99.5, 99.5])
    out = json.loads(f(_one_trade_row(), up, 3))
    assert out["closed_reason"] == "target" and out["closed_ts"]
    dn = _one_trade_price_data([100, 100.5, 100.5, 100.5], [100, 99.5, 98.9, 99.5])
    assert json.loads(f(_one_trade_row(), dn, 3))["closed_reason"] == "stop"
    # a sell trade reads the same path the other way round
    deep = _one_trade_price_data([100, 100.5, 100.5, 100.5], [100, 99.5, 97.9, 99.5])
    assert json.loads(f(_one_trade_row("sell"), deep, 3))["closed_reason"] == "target"
    # already closed rows are not rewritten
    assert f(out and json.dumps(out), up, 3) is None
    # the fill candle itself is not closed yet
    assert f(_one_trade_row(), up, 0) is None


def test_get_active_trade_ignores_closed_rows_and_picks_newest():
    base = "outcome_pending:BTCUSD:"
    open_old = _one_trade_row(entry_idx=0)
    open_new = _one_trade_row(entry_idx=2)
    closed = json.dumps({**json.loads(_one_trade_row(entry_idx=5)), "closed_ts": 1})
    rows = {base + "a_key:1000000": open_old, base + "b_key:1001800": open_new,
            base + "c_key:1004500": closed}
    got = asyncio.run(_one_trade_store(rows).get_active_trade("BTCUSD"))
    assert got["alert_key"] == "b_key" and got["entry_ts"] == 1_001_800
    only_closed = {base + "c_key:1004500": closed}
    assert asyncio.run(_one_trade_store(only_closed).get_active_trade("BTCUSD")) is None
    assert asyncio.run(
        _one_trade_store(rows, degraded=True).get_active_trade("BTCUSD")
    ) is None

def test_record_status_line_text():
    line = A._record_status_line(
        "IGNORED", {"pair": "BTCUSD", "direction": "buy", "entry_ts": 1_000_000}
    )
    assert line.startswith("⏭ Ignored") and "BTCUSD buy" in line
    assert A._record_status_line("RECORDED", None).startswith("📝 Recorded")

# ======================================================================
# pullback-continuation detector
# ======================================================================
def _pb_cfg(**over):
    base = dict(
        PULLBACK_LOOKBACK_CANDLES=24, PULLBACK_IMPULSE_LOOKBACK_CANDLES=24,
        PULLBACK_MIN_IMPULSE_ATR=2.0, PULLBACK_MIN_DEPTH_ATR=0.8,
        PULLBACK_MIN_RETRACE=0.2, PULLBACK_MAX_RETRACE=0.786, PULLBACK_MIN_BARS=2,
        PULLBACK_MAX_AGE_BARS=8, PULLBACK_EXHAUSTION_MIN_VOTES=2,
        PULLBACK_REJECTION_WICK_RATIO=0.4, PULLBACK_TRIGGER_BUFFER_ATR=0.0,
        PULLBACK_CONFIRM_WINDOW_BARS=1,
    )
    base.update(over)
    return NS(**base)

def _pb_series(extra_closes=()):
    closes = list(np.linspace(100, 120, 21)) + [118, 116, 114.5, 113.5, 113.0, 113.8, 114.6, 115.4]
    closes += list(extra_closes)
    c = np.array(closes, float)
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) + 0.1
    low = np.minimum(o, c) - 0.1
    n = len(c)
    n_tail = n - 26
    ph = np.r_[np.linspace(0, 2, 21), np.linspace(1.8, -1, 5), np.linspace(-0.8, 0.5, n_tail)]
    rsi = np.r_[np.linspace(50, 75, 21), np.linspace(70, 45, 5), np.linspace(47, 55, n_tail)]
    return o, h, low, c, np.arange(n) * 900, np.full(n, 1.0), ph, rsi

def test_pullback_detector_sequence_and_buy_sell_symmetry():
    o, h, low, c, ts, atr, ph, rsi = _pb_series()
    cfg_o = _pb_cfg()
    buy = [A._pullback_continuation_state(o, h, low, c, ts, atr, ph, rsi, i, True, True, cfg_o)[0]
           for i in range(24, 29)]
    sell = [A._pullback_continuation_state(-o, -low, -h, -c, ts, atr, -ph, -rsi, i, False, True, cfg_o)[0]
            for i in range(24, 29)]
    assert buy == ["pullback", "pullback", "exhaustion", "confirmed", "confirmed"]
    assert sell == buy
    assert A._pullback_continuation_state(
        o, h, low, c, ts, atr, ph, rsi, 27, True, False, cfg_o)[0] == "none"

def test_pullback_confirm_window_and_invalidation():
    o, h, low, c, ts, atr, ph, rsi = _pb_series(extra_closes=[115.0])
    strict, wide = _pb_cfg(), _pb_cfg(PULLBACK_CONFIRM_WINDOW_BARS=2)
    i = 29   # bar 29 closes below bar 28's close, so it is not a trigger itself
    ph2, rsi2 = ph.copy(), rsi.copy()
    ph2[i], rsi2[i] = ph[i - 1] + 0.1, rsi[i - 1] + 1.0
    st_strict = A._pullback_continuation_state(o, h, low, c, ts, atr, ph2, rsi2, i, True, True, strict)[0]
    st_wide = A._pullback_continuation_state(o, h, low, c, ts, atr, ph2, rsi2, i, True, True, wide)[0]
    assert st_strict == "exhaustion" and st_wide == "confirmed"
    c2 = c.copy()
    c2[24:27] = [99.0, 98.0, 98.5]   # pullback breaks the impulse start
    o2 = np.r_[c2[0], c2[:-1]]
    h2, l2 = np.maximum(o2, c2) + 0.1, np.minimum(o2, c2) - 0.1

    assert A._pullback_continuation_state(o2, h2, l2, c2, ts, atr, ph, rsi, 26, True, True, strict)[0] == "invalid"

# ======================================================================
# late-trend exhaustion: scorer, helpers, verdict overlay, study
# ======================================================================
import autopsy
from indicators import _exhaustion_risk

# Every threshold the scorer reads is pinned here, so tests never depend on config defaults.
_EX_PIN = dict(
    EXHAUSTION_LOOKBACK_CANDLES=96, EXHAUSTION_RUN_BARS=24, EXHAUSTION_BASELINE_EMA=50,
    EXHAUSTION_SWING_BARS=2, EXHAUSTION_EXT_ATR=3.5, EXHAUSTION_MIN_LEGS=4,
    EXHAUSTION_DIV_MAX_AGE_BARS=30, EXHAUSTION_RSI_DIV_MIN=1.0, EXHAUSTION_MIN_ROOM_R=0.75,
    EXHAUSTION_TOP_WICK_RATIO=0.5, EXHAUSTION_TOP_MAX_AGE_BARS=12, EXHAUSTION_MOM_FADE_BARS=12,
    EXHAUSTION_PROVISIONAL_MOMENTUM=False, EXHAUSTION_VETO_SCORE=4, EXHAUSTION_MIN_CATEGORIES=2,
    OUTCOME_MAE_LOSS_PCT=1.0,
)

def _ex_cfg(**over):
    return cfg.model_copy(update={**_EX_PIN, **over})

def _ex_series(n=140):
    """Staircase up-trend (6 up bars, 6 down bars) with fading momentum."""
    c = [100.0]
    for k in range(1, n):
        c.append(c[-1] + (0.35 if (k // 6) % 2 == 0 else -0.25))
    c = np.array(c)
    o = np.r_[c[0], c[:-1]]
    h, low = np.maximum(o, c) + 0.1, np.minimum(o, c) - 0.1
    atr = np.full(n, 0.3)
    ph = np.array([max(0.5 - 0.02 * k, -1) if (k // 6) % 2 == 0 else -0.2 for k in range(n)])
    rsi = np.array([70 - 0.1 * k for k in range(n)], dtype=float)
    return o, h, low, c, atr, ph, rsi


def _ex_buy(i=119, cfg_o=None, series=None):
    o, h, low, c, atr, ph, rsi = series or _ex_series()
    return _exhaustion_risk(o, h, low, c, atr, ph, rsi, i, True, cfg_o or _ex_cfg())

def test_exhaustion_scorer_votes_and_boundaries():
    score, info = _ex_buy()
    assert info["status"] == "ok" and score == info["score"] > 0
    assert info["ext_atr"] >= 3.5 and info["legs"] >= 4          # fixture property, thresholds pinned above
    assert "EXT_ATR" in info["reasons"] and "LEGS" in info["reasons"]
    # extension vote flips exactly around its threshold
    assert "EXT_ATR" in _ex_buy(cfg_o=_ex_cfg(EXHAUSTION_EXT_ATR=max(1.0, info["ext_atr"] - 0.05)))[1]["reasons"]
    assert "EXT_ATR" not in _ex_buy(cfg_o=_ex_cfg(EXHAUSTION_EXT_ATR=info["ext_atr"] + 0.05))[1]["reasons"]
    # leg-count vote flips at its threshold
    assert "LEGS" in _ex_buy(cfg_o=_ex_cfg(EXHAUSTION_MIN_LEGS=info["legs"]))[1]["reasons"]
    assert "LEGS" not in _ex_buy(cfg_o=_ex_cfg(EXHAUSTION_MIN_LEGS=info["legs"] + 1))[1]["reasons"]
    # veto needs BOTH the score and enough independent categories
    at = _ex_cfg(EXHAUSTION_VETO_SCORE=info["score"], EXHAUSTION_MIN_CATEGORIES=info["cats"])
    assert _ex_buy(cfg_o=at)[1]["veto"] is True
    assert _ex_buy(cfg_o=_ex_cfg(EXHAUSTION_VETO_SCORE=info["score"] + 1,
                                  EXHAUSTION_MIN_CATEGORIES=1))[1]["veto"] is False       # score one short
    assert _ex_buy(cfg_o=_ex_cfg(EXHAUSTION_VETO_SCORE=1,
                                  EXHAUSTION_MIN_CATEGORIES=info["cats"] + 1))[1]["veto"] is False  # one category short
    assert info["score"] <= 6 and info["mom_pts"] <= 2


def test_exhaustion_scorer_buy_sell_symmetry():
    o, h, low, c, atr, ph, rsi = _ex_series()
    for i in (60, 100, 119, 125):
        b = _exhaustion_risk(o, h, low, c, atr, ph, rsi, i, True, _ex_cfg())
        s = _exhaustion_risk(-o, -low, -h, -c, atr, -ph, -rsi, i, False, _ex_cfg())
        assert b == s

def test_exhaustion_scorer_missing_data_fails_open():
    o, h, low, c, atr, ph, rsi = _ex_series()
    assert _exhaustion_risk(o, h, low, c, atr, ph, rsi, 10, True, _ex_cfg())[1]["status"] == "unknown"      # short history
    c_nan = c.copy()
    c_nan[100] = np.nan
    s, info = _exhaustion_risk(o, h, low, c_nan, atr, ph, rsi, 119, True, _ex_cfg())
    assert (s, info["status"], info["veto"]) == (0, "unknown", False)
    atr0 = atr.copy()
    atr0[119] = 0.0
    assert _exhaustion_risk(o, h, low, c, atr0, ph, rsi, 119, True, _ex_cfg())[1]["status"] == "unknown"
    ph_nan = ph.copy()
    ph_nan[:] = np.nan               
    s, info = _exhaustion_risk(o, h, low, c, atr, ph_nan, rsi, 119, True, _ex_cfg())
    assert info["status"] == "ok" and info["ppo_div"] == 0

def test_exhaustion_scorer_never_looks_ahead():
    o, h, low, c, atr, ph, rsi = _ex_series()
    full = _exhaustion_risk(o, h, low, c, atr, ph, rsi, 119, True, _ex_cfg())
    cut = _exhaustion_risk(o[:120], h[:120], low[:120], c[:120], atr[:120], ph[:120], rsi[:120], 119, True, _ex_cfg())
    assert full == cut

def test_exhaustion_rejection_freshness_and_provisional_fade():
    o, h, low, c, atr, ph, rsi = _ex_series()
    _, base = _ex_buy()
    age = base["top_age"]
    assert age >= 1, "series must end after its run extreme (the signal bar cannot be the extreme)"
    h2 = h.copy()
    h2[119 - age] += 1.5                                      # huge upper wick on the run-extreme candle
    fresh = _exhaustion_risk(o, h2, low, c, atr, ph, rsi, 119, True, _ex_cfg())[1]
    assert fresh["rej_wick"] == 1 and "TOP_REJECT" in fresh["reasons"]
    stale = _exhaustion_risk(o, h2, low, c, atr, ph, rsi, 119, True, _ex_cfg(EXHAUSTION_TOP_MAX_AGE_BARS=age - 1))[1]
    assert stale["rej_wick"] == 0
    # fresh run high whose histogram / RSI are already below their own recent peak
    n = 140
    c2 = c.copy()
    c2[130:] = c2[129] + np.arange(1, n - 129) * 0.4
    o2 = np.r_[c2[0], c2[:-1]]
    h3, l3 = np.maximum(o2, c2) + 0.1, np.minimum(o2, c2) - 0.1
    ph3, rsi3 = ph.copy(), rsi.copy()
    ph3[120:128], ph3[130:] = 2.0, 0.3
    rsi3[120:128], rsi3[130:] = 80.0, 70.0
    off = _exhaustion_risk(o2, h3, l3, c2, atr, ph3, rsi3, 139, True, _ex_cfg(EXHAUSTION_PROVISIONAL_MOMENTUM=False))
    on = _exhaustion_risk(o2, h3, l3, c2, atr, ph3, rsi3, 139, True, _ex_cfg(EXHAUSTION_PROVISIONAL_MOMENTUM=True))
    assert off[1]["mom_fade"] == 1 and "MOM_FADE" not in off[1]["reasons"]      # measured, not scored
    assert on[1]["mom_fade"] == 1 and "MOM_FADE" in on[1]["reasons"] and on[0] == off[0] + 1


def _ex_ctx(score=4, cats=2, status="ok"):
    info = {"status": status, "score": score, "cats": cats, "veto": score >= 4 and cats >= 2,
            "reasons": ["EXT_ATR", "LEGS"], "ext_atr": 5.1, "legs": 5, "ppo_div": 0, "rsi_div": 1,
            "room_r": None, "top_wick": 0.2, "top_age": 3, "ext_pts": 2, "mom_pts": 1, "fail_pts": 1,
            "v_ext": 1, "v_legs": 1, "v_room": 0, "rej_wick": 1, "rej_break": 0, "mom_fade": 0}
    return {"exhaustion_info_buy": info, "exhaustion_info_sell": {}}


def test_exhaustion_scope_helpers_and_toggles(monkeypatch):
    monkeypatch.setattr(cfg, "EXHAUSTION_SCOPE", "all", raising=False)
    monkeypatch.setattr(cfg, "EXHAUSTION_EXCLUDE_KEYS", ["ppo_zero_up"], raising=False)
    assert A._exhaustion_applies("pivot_up_r1") and A._exhaustion_applies("strong_reversal_buy")
    assert not A._exhaustion_applies("ppo_zero_up")
    monkeypatch.setattr(cfg, "EXHAUSTION_SCOPE", "hosts", raising=False)
    monkeypatch.setattr(cfg, "EXHAUSTION_EXCLUDE_KEYS", [], raising=False)
    assert A._exhaustion_applies("strong_reversal_buy") and not A._exhaustion_applies("pivot_up_r1")
    monkeypatch.setattr(cfg, "EXHAUSTION_SCOPE", "all", raising=False)
    ctx = _ex_ctx()
    # snapshot stored with trades: flat, numeric, empty when unknown
    f = A.exhaustion_trade_fields(ctx, "buy")
    assert f["exh_score"] == 4 and f["exh_cats"] == 2 and "exh_room_r" not in f and f["exh_rej_wick"] == 1
    assert A.exhaustion_trade_fields(_ex_ctx(status="unknown"), "buy") == {}
    assert A.exhaustion_trade_fields(ctx, "sell") == {}
    # warning line: only at/above the show threshold, never when the filter is off
    monkeypatch.setattr(cfg, "EXHAUSTION_MODE", "shadow", raising=False)
    monkeypatch.setattr(cfg, "EXHAUSTION_SHOW_MIN_SCORE", 3, raising=False)
    assert "Late-trend risk 4/6" in A.exhaustion_message_line(ctx, "buy", ["vwap_up"])
    assert A.exhaustion_message_line(_ex_ctx(score=2), "buy", ["vwap_up"]) is None
    monkeypatch.setattr(cfg, "EXHAUSTION_MODE", "off", raising=False)
    assert A.exhaustion_message_line(ctx, "buy", ["vwap_up"]) is None
    # info handed to the verdict advisor follows both toggles
    monkeypatch.setattr(cfg, "EXHAUSTION_MODE", "shadow", raising=False)
    monkeypatch.setattr(cfg, "EXHAUSTION_VERDICT_MODE", "live", raising=False)
    assert A.exhaustion_info_for_verdict(ctx, "buy", ["vwap_up"]) is ctx["exhaustion_info_buy"]
    monkeypatch.setattr(cfg, "EXHAUSTION_VERDICT_MODE", "off", raising=False)
    assert A.exhaustion_info_for_verdict(ctx, "buy", ["vwap_up"]) is None
    monkeypatch.setattr(cfg, "EXHAUSTION_VERDICT_MODE", "live", raising=False)
    monkeypatch.setattr(cfg, "EXHAUSTION_MODE", "off", raising=False)
    assert A.exhaustion_info_for_verdict(ctx, "buy", ["vwap_up"]) is None


def test_exhaustion_verdict_overlay_live_shadow_off():
    strong = {"net_ev": 0.30, "verdict": "ACTIONABLE", "evidence_state": "ACTIONABLE",
              "regime_wr": 0.7, "p_ev_positive": 0.8}
    base = dict(direction="buy", score=26, total=27, required=21, votes={}, tqs=[strong],
                bias="with", take_min=70, watch_min=50)
    prm = dict(watch_score=3, avoid_score=5, min_cats=2, penalty_per_point=5.0)

    def run(score, cats, mode, status="ok", **over):
        info = {"status": status, "score": score, "cats": cats, "reasons": ["EXT_ATR"]}
        return advise_pair(exhaustion=info, exhaustion_mode=mode, exhaustion_params=prm, **{**base, **over})

    plain = advise_pair(**base)
    assert plain.verdict == TAKE and plain.exh_adj is None
    assert run(2, 2, "live").verdict == TAKE                       # below the watch score
    assert run(3, 1, "live").verdict == TAKE                       # one category is never enough
    w = run(3, 2, "live")
    assert w.verdict == WATCH and w.conviction < 70 and "late-trend exhaustion 3/6" in w.risk_line
    a = run(5, 2, "live")
    assert a.verdict == AVOID and a.conviction < 50 and a.size_line == ""
    s = run(5, 2, "shadow")                                        # shadow: recorded, not applied
    assert (s.verdict, s.conviction) == (plain.verdict, plain.conviction)
    assert s.exh_adj["verdict_to"] == AVOID and s.exh_adj["applied"] is False
    assert run(5, 3, "off").exh_adj is None
    assert run(5, 3, "live", status="unknown").verdict == TAKE     # missing data fails open
    # restrict-only: a verdict that is already AVOID is never raised
    weak = dict(strong, net_ev=-0.2)
    assert run(5, 2, "live", tqs=[weak]).verdict == AVOID
    # no brain history: hard rule still applies, no conviction is invented
    nodata = run(5, 2, "live", tqs=[None])
    assert nodata.verdict == AVOID and nodata.conviction is None

def test_exhaustion_study_population_early_stop_and_configured_cut():
    def row(score, cats, early, net, ts):
        return {"context": {"exh_score": score, "exh_cats": cats}, "entry_ts": ts, "pair": "X",
                "outcome_class": "sl" if net < 0 else "tp",
                "outcome_reason": "stop_hit" if net < 0 else "target_hit",
                "sl_candle": (0 if early else 5) if net < 0 else None, "win": net > 0,
                "realized_cost_pct": 0.1}
    rows = ([row(5, 3, True, -1, k) for k in range(10)] + [row(5, 3, False, 1, k) for k in range(10, 12)]
            + [row(1, 0, False, 1, k) for k in range(12, 40)]
            + [{"context": {}, "entry_ts": 99, "pair": "X", "outcome_reason": "stop_hit"}])   # no snapshot: ignored
    st = autopsy.exhaustion_study(rows, cost_pct=0.1, early_bars=2, min_n=30, veto_score=5)
    assert st["n"] == 40 and st["status"] == "OK" and st["early_rate"] == round(10 / 40, 3)
    cut = next(c for c in st["cuts"] if c["configured"])
    assert cut["score"] == 5 and cut["blocked"] == 12 and cut["early_caught"] == 1.0
    assert cut["winners_lost"] == round(2 / 30, 3) and cut["net_after"] > cut["net_before"]
    # stop after the early window is not an early stop
    late = autopsy.exhaustion_study([row(5, 3, False, -1, 1)] * 3, cost_pct=0.1, early_bars=2, min_n=1)
    assert late["early_rate"] == 0.0

def test_exhaustion_veto_decision_and_shadow_recording(monkeypatch):
    ctx = {"exhaustion_veto_buy": True, "exhaustion_veto_sell": False, **_ex_ctx(score=5, cats=3)}
    monkeypatch.setattr(cfg, "EXHAUSTION_SCOPE", "all", raising=False)
    monkeypatch.setattr(cfg, "EXHAUSTION_EXCLUDE_KEYS", ["tl_break_buy"], raising=False)
    monkeypatch.setattr(cfg, "EXHAUSTION_MODE", "shadow", raising=False)
    assert A.exhaustion_vetoes(ctx, "vwap_up") is False                 # shadow never vetoes
    monkeypatch.setattr(cfg, "EXHAUSTION_MODE", "live", raising=False)
    assert A.exhaustion_vetoes(ctx, "vwap_up") is True                  # live, covered, BUY side vetoed
    assert A.exhaustion_vetoes(ctx, "vwap_down") is False               # SELL side has no veto
    assert A.exhaustion_vetoes(ctx, "tl_break_buy") is False            # excluded key
    monkeypatch.setattr(cfg, "EXHAUSTION_SCOPE", "hosts", raising=False)
    assert A.exhaustion_vetoes(ctx, "vwap_up") is False                 # not a host alert
    assert A.exhaustion_vetoes(ctx, "strong_reversal_buy") is True
    # the vetoed alert is shadow-tracked with its reason and the exhaustion snapshot
    seen = []

    class _Sdb:
        degraded = False

        async def record_shadow_pending_outcome(self, pair, key, direction, ts, price, **kw):
            seen.append((pair, key, direction, kw["context"]))
            return "created"

    monkeypatch.setattr(cfg, "ENABLE_BRAIN", True, raising=False)
    monkeypatch.setattr(cfg, "BRAIN_SHADOW_MODE", True, raising=False)
    asyncio.run(A._record_counterfactual_block(
        _Sdb(), "ETHUSD", [("Strong Reversal", "", "strong_reversal_buy")], 1000, 100.0,
        block_reason="exhaustion_veto", context=ctx, logger_pair=LOG))
    assert len(seen) == 1 and seen[0][1] == "strong_reversal_buy" and seen[0][2] == "buy"
    assert seen[0][3]["rejection_reason"] == "exhaustion_veto" and seen[0][3]["exh_score"] == 5

# ======================================================================
# one recorded trade per pair across Elliott + MACD, candidate-only rows,
# shadow flags, pullback WATCH rules, co-key credit reader
# ======================================================================
"""In-memory Redis stand-in (no fakeredis dependency) for the cross-source tests."""
import fnmatch
from datetime import datetime, timezone

from elliott_wave import ComponentScore, WaveSetup
import elliott_alerts as EA

T_OPEN = int(datetime(2026, 11, 18, 0, 0, tzinfo=timezone.utc).timestamp())   # 05:30 IST


class _MiniPipe:
    def __init__(self, r):
        self.r, self.ops = r, []

    def __getattr__(self, name):
        def rec(*a, **k):
            self.ops.append((name, a, k))
        return rec

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def execute(self):
        return [self.r._op(n, *a, **k) for n, a, k in self.ops]


class _MiniRedis:
    def __init__(self):
        self.d, self.ttl = {}, {}

    def _op(self, name, *a, **k):
        d = self.d
        if name == "get":
            return d.get(a[0])
        if name == "set":
            key, val = a
            if (k.get("nx") and key in d) or (k.get("xx") and key not in d):
                return None
            d[key] = val
            if k.get("ex"):
                self.ttl[key] = k["ex"]
            elif not k.get("keepttl"):
                self.ttl.pop(key, None)
            return True
        if name == "mget":
            return [d.get(x) for x in a[0]]
        if name == "exists":
            return int(a[0] in d)
        if name == "lpush":
            d.setdefault(a[0], []).insert(0, a[1])
            return len(d[a[0]])
        if name == "ltrim":
            d[a[0]] = d.get(a[0], [])[a[1]:a[2] + 1]
            return True
        if name == "lrange":
            return d.get(a[0], [])[a[1]:(None if a[2] == -1 else a[2] + 1)]
        if name == "llen":
            return len(d.get(a[0], []))
        if name == "expire":
            self.ttl[a[0]] = a[1]
            return True
        if name == "hincrby":
            h = d.setdefault(a[0], {})
            h[a[1]] = str(int(h.get(a[1], 0)) + int(a[2]))
            return int(h[a[1]])
        if name == "hgetall":
            return dict(d.get(a[0], {}))
        if name == "hset":
            d.setdefault(a[0], {})[a[1]] = a[2]
            return 1
        if name == "hdel":
            return sum(d.get(a[0], {}).pop(f, None) is not None for f in a[1:])
        if name == "hkeys":
            return list(d.get(a[0], {}))
        if name == "hlen":
            return len(d.get(a[0], {}))
        raise NotImplementedError(name)

    def __getattr__(self, name):
        async def call(*a, **k):
            return self._op(name, *a, **k)
        return call

    def pipeline(self, transaction=False):
        return _MiniPipe(self)

    async def scan_iter(self, match="*", count=0):
        for k in list(self.d):
            if fnmatch.fnmatchcase(k, match):
                yield k


def _mini_store():
    s = object.__new__(state.RedisStateStore)
    s.degraded = False
    s._redis = _MiniRedis()
    s._pending_outcome_keys_by_pair = None
    s._shadow_pending_outcome_keys_by_pair = None
    s._shadow_registered_this_run = {}
    s._recorded_this_run = {}
    s._claim_token = "tok"
    s._close_event_ids = set()
    s.trade_close_events = []
    return s


def _one_trade_cfg(monkeypatch):
    for k, v in dict(
        ENABLE_WIN_RATE_FILTER=True, ENABLE_BRAIN=True, ENABLE_SINGLE_ACTIVE_TRADE=True,
        ENABLE_TRADE_CLOSE_NOTICE=True, ONE_TRADE_ACROSS_SOURCES=True, TRADE_CLOSE_COOLDOWN_CANDLES=3,
        DRY_RUN_MODE=False, CANDIDATE_LOG_MAX_PER_PAIR=300, ENABLE_PULLBACK_CONTINUATION=True,
        PULLBACK_SEND_WATCH=True, TRADE_CLOSE_COOLDOWN_ON_STOP=False,
    ).items():
        monkeypatch.setattr(cfg, k, v, raising=False)


def _wave(pair="XAUTUSD", direction="sell", ts=T_OPEN):
    return WaveSetup(
        family="wave3", direction=direction, status="eligible", reason="", score=93.0, components={},
        labels={}, event_id=f"ev-{pair}-{ts}", structure_id=f"st-{pair}", signal_idx=10, signal_ts=ts,
        entry_price=3833.9, trigger_level=3841.15, entry_distance_atr=1.36, invalidation=3862.83,
        stop=3851.49, targets=(3803.98, 3775.44), rr=1.7, atr=10.0, age_bars=0, htf_bias=1,
        detail={"structure": ComponentScore("structure", 25, 25, "", {})}, features={}, pair=pair)


def _put(sdb, prefix, pair, key, ts, direction, **extra):
    row = {"direction": direction, "entry_ts": ts, "entry_price": 100.0, **extra}
    sdb._redis.d[f"{prefix}{pair}:{key}:{ts}"] = json.dumps(row)


def _candidates(sdb, pair):
    return [json.loads(x) for x in sdb._redis.d.get(f"{state.RedisKeyPrefix.CANDIDATE_EVENTS}{pair}", [])]


class _TgCapture:
    def __init__(self):
        self.sent = []

    async def send(self, m):
        self.sent.append(m)
        return True


def test_elliott_footer_shows_candle_open_time():
    msg = EA.format_elliott_message([_wave()])
    assert "05:30 IST" in msg and "05:45" not in msg


def test_active_trade_spans_sources_and_the_switch_restores_isolation(monkeypatch):
    _one_trade_cfg(monkeypatch)
    sdb = _mini_store()
    _put(sdb, state.RedisKeyPrefix.OUTCOME_PENDING, "XAUTUSD", "elliott_wave3_sell", T_OPEN, "sell")
    got = asyncio.run(sdb.get_active_trade("XAUTUSD"))
    assert got and got["alert_key"] == "elliott_wave3_sell"          # a MACD alert now sees the Elliott trade
    assert asyncio.run(sdb.get_active_trade("XAUTUSD", elliott=False)) is not None   # default semantics
    monkeypatch.setattr(cfg, "ONE_TRADE_ACROSS_SOURCES", False, raising=False)
    assert asyncio.run(sdb.get_active_trade("XAUTUSD")) is None      # old isolation
    assert asyncio.run(sdb.get_active_trade("XAUTUSD", elliott=True)) is not None
    assert asyncio.run(sdb.get_active_trade("XAUTUSD", elliott=None)) is not None


def test_pair_record_state_open_closed_and_cooldown(monkeypatch):
    _one_trade_cfg(monkeypatch)
    sdb = _mini_store()
    key = f"{state.RedisKeyPrefix.OUTCOME_PENDING}XAUTUSD:elliott_wave3_sell:{T_OPEN}"
    _put(sdb, state.RedisKeyPrefix.OUTCOME_PENDING, "XAUTUSD", "elliott_wave3_sell", T_OPEN, "sell")
    st, det = asyncio.run(sdb.pair_record_state("XAUTUSD", T_OPEN + 900, "sell"))
    assert st == "IGNORED" and det["direction"] == "sell"
    assert asyncio.run(sdb.pair_record_state("XAUTUSD", T_OPEN + 900, "buy"))[0] == "IGNORED"   # either direction
    sdb._redis.d[key] = json.dumps({"direction": "sell", "entry_ts": T_OPEN, "closed_ts": T_OPEN + 1800})
    assert asyncio.run(sdb.pair_record_state("XAUTUSD", T_OPEN + 2700, "sell"))[0] == "RECORDED"
    asyncio.run(sdb._start_trade_cooldown("XAUTUSD", T_OPEN + 1800, LOG, "Recorded", "sell"))
    st, det = asyncio.run(sdb.pair_record_state("XAUTUSD", T_OPEN + 2700, "sell"))
    assert st == "COOLDOWN" and det["until_ts"] > det["hit_ts"]
    assert asyncio.run(sdb.pair_record_state("XAUTUSD", T_OPEN + 2700, "buy"))[0] == "RECORDED"


def test_trade_recorded_this_run_blocks_the_other_source_despite_stale_prescan(monkeypatch):
    _one_trade_cfg(monkeypatch)
    sdb = _mini_store()
    sdb._pending_outcome_keys_by_pair = {}                        # the run-start scan: nothing open
    mkey = sorted(A.BUY_ALERT_KEYS)[0]
    assert asyncio.run(sdb.record_pending_outcome(
        "BTCUSD", mkey, "buy", T_OPEN, 100.0, confluence_score=5.0, confluence_total=10.0)) is True
    st, det = asyncio.run(sdb.pair_record_state("BTCUSD", T_OPEN, "buy"))
    assert st == "IGNORED" and det["alert_key"] == mkey
    assert asyncio.run(sdb.pair_record_state("ETHUSD", T_OPEN, "buy"))[0] == "RECORDED"


def test_elliott_cooldown_honoured_by_recording_and_started_by_an_elliott_target(monkeypatch):
    _one_trade_cfg(monkeypatch)
    sdb = _mini_store()
    row = {"direction": "sell", "entry_ts": T_OPEN, "closed_reason": "target", "closed_candle_ts": T_OPEN + 1800,
           "context": {"elliott_wave_family": "wave3"}}
    asyncio.run(sdb._publish_close_events("XAUTUSD", [row], "Recorded", LOG))
    assert asyncio.run(sdb.in_trade_cooldown("XAUTUSD", T_OPEN + 2700, direction="sell")) == T_OPEN + 1800
    mkey = sorted(A.SELL_ALERT_KEYS)[0]
    # the cooldown now also stops a MACD recording, and an Elliott one
    assert asyncio.run(sdb.record_pending_outcome("XAUTUSD", mkey, "sell", T_OPEN + 2700, 100.0, 5.0, 10.0)) is True
    assert not any(k.startswith(state.RedisKeyPrefix.OUTCOME_PENDING) for k in sdb._redis.d)
    assert asyncio.run(sdb.record_pending_outcome("XAUTUSD", "elliott_wave3_sell", "sell", T_OPEN + 2700, 100.0, 93.0, 100.0)) is True
    assert not any(k.startswith(state.RedisKeyPrefix.OUTCOME_PENDING) for k in sdb._redis.d)
    monkeypatch.setattr(cfg, "ONE_TRADE_ACROSS_SOURCES", False, raising=False)     # old isolation: no cooldown from Elliott
    sdb2 = _mini_store()
    asyncio.run(sdb2._publish_close_events("XAUTUSD", [row], "Recorded", LOG))
    assert asyncio.run(sdb2.in_trade_cooldown("XAUTUSD", T_OPEN + 2700, direction="sell")) is None


def test_elliott_delivers_a_blocked_alert_labelled_and_does_not_record_it(monkeypatch):
    _one_trade_cfg(monkeypatch)
    sdb, tg = _mini_store(), _TgCapture()
    _put(sdb, state.RedisKeyPrefix.OUTCOME_PENDING, "XAUTUSD", "pivot_x", T_OPEN - 900, "sell")
    sel = [("XAUTUSD", _wave("XAUTUSD", "sell")), ("ETHUSD", _wave("ETHUSD", "buy"))]
    verdicts = asyncio.run(EA._one_trade_gate(sdb, sel, LOG))
    stats = {"errors": 0, "delivered": 0}
    got = asyncio.run(EA._deliver(sel, T_OPEN + 900, sdb, tg, LOG, stats, {}, rec_state=verdicts))
    assert len(got) == 2 and stats["delivered"] == 2                       # both are SENT
    assert "Ignored" in tg.sent[0] and "XAUTUSD sell trade open since" in tg.sent[0]
    pend = [k for k in sdb._redis.d if k.startswith(state.RedisKeyPrefix.OUTCOME_PENDING)]
    assert not any("XAUTUSD:elliott" in k for k in pend)                    # blocked pair: not recorded
    assert any("ETHUSD:elliott_wave3_buy" in k for k in pend)               # free pair: recorded
    rows = _candidates(sdb, "XAUTUSD")
    assert rows and rows[0]["reason"] == "ignored_open_trade" and rows[0]["open_key"] == "pivot_x"
    assert rows[0]["entry_price"] == 3833.9 and rows[0]["score"] == 93.0 and rows[0]["total"] == 100.0   # gradable later
    assert _candidates(sdb, "ETHUSD") == []


def test_elliott_one_trade_gate_is_off_when_the_switch_is_off(monkeypatch):
    _one_trade_cfg(monkeypatch)
    monkeypatch.setattr(cfg, "ONE_TRADE_ACROSS_SOURCES", False, raising=False)
    sdb = _mini_store()
    _put(sdb, state.RedisKeyPrefix.OUTCOME_PENDING, "XAUTUSD", "pivot_x", T_OPEN - 900, "sell")
    assert asyncio.run(EA._one_trade_gate(sdb, [("XAUTUSD", _wave())], LOG)) == {}


def test_shadow_overlap_recorded_active_and_skipped_active_rows(monkeypatch):
    _one_trade_cfg(monkeypatch)
    mkey = sorted(A.BUY_ALERT_KEYS)[0]
    sdb = _mini_store()
    sk = f"{state.RedisKeyPrefix.SHADOW_PENDING}SOLUSD:{mkey}:{T_OPEN - 900}"
    sdb._redis.d[sk] = json.dumps({"direction": "buy", "entry_ts": T_OPEN - 900, "entry_price": 100.0, "context": {}})
    sdb._redis.ttl[sk] = 777
    asyncio.run(sdb.record_pending_outcome("SOLUSD", mkey, "buy", T_OPEN, 100.0, 5.0, 10.0))
    assert json.loads(sdb._redis.d[sk])["context"]["overlapped_by_recorded"] is True
    assert sdb._redis.ttl[sk] == 777                                         # TTL untouched (keepttl)
    # a shadow registered while a Recorded trade is open carries recorded_active
    sdb = _mini_store()
    asyncio.run(sdb.record_pending_outcome("ADAUSD", mkey, "buy", T_OPEN, 1.0, 5.0, 10.0))
    res = asyncio.run(sdb.record_shadow_pending_outcome("ADAUSD", mkey, "sell", T_OPEN + 900, 1.0, 5.0, 10.0, None, {}))
    srow = json.loads(sdb._redis.d[f"{state.RedisKeyPrefix.SHADOW_PENDING}ADAUSD:{mkey}:{T_OPEN + 900}"])
    assert res == "created" and srow["context"]["recorded_active"] is True
    # a later shadow while that one is open is skipped, and the skip is logged, not dropped silently
    res2 = asyncio.run(sdb.record_shadow_pending_outcome("ADAUSD", mkey, "sell", T_OPEN + 1800, 1.0, 5.0, 10.0, None, {}))
    ev = _candidates(sdb, "ADAUSD")
    assert res2 == "skipped_active" and ev and ev[0]["reason"] == "skipped_active"
    assert ev[0]["entry_price"] == 1.0 and ev[0]["score"] == 5.0 and ev[0]["total"] == 10.0 and ev[0]["votes_on"] == []
    assert not any(k.endswith(str(T_OPEN + 1800)) for k in sdb._redis.d if k.startswith(state.RedisKeyPrefix.SHADOW_PENDING))
    # candidate rows are capped and can be switched off
    monkeypatch.setattr(cfg, "CANDIDATE_LOG_MAX_PER_PAIR", 2, raising=False)
    for i in range(5):
        asyncio.run(sdb.log_candidate_event("ADAUSD", mkey, "buy", T_OPEN + i, "ignored_open_trade"))
    assert len(_candidates(sdb, "ADAUSD")) == 2
    monkeypatch.setattr(cfg, "CANDIDATE_LOG_MAX_PER_PAIR", 0, raising=False)
    asyncio.run(sdb.log_candidate_event("XRPUSD", mkey, "buy", T_OPEN, "ignored_open_trade"))
    assert _candidates(sdb, "XRPUSD") == []


def test_watch_levels_render_with_price_precision_and_fail_closed():
    assert A._fmt_level(0.0382) == "0.0382" and A._fmt_level(3811.85) == "3,811.85" and A._fmt_level(2.5) == "2.500"
    info = {"depth_atr": 8.2, "retrace": 0.6, "trigger": 0.0371, "invalidation": 0.0392}
    txt = A._pullback_levels_text(info)
    assert "trigger 0.0371" in txt and "invalid beyond 0.0392" in txt and "trigger 0.0 " not in txt
    ok = {"trigger": 0.0371, "invalidation": 0.0392}
    assert A._watch_levels_problem(ok, 0.0379, "sell") is None
    for bad, why in (({"trigger": None, "invalidation": 0.0392}, "trigger missing"),
                     ({"trigger": 0.0371, "invalidation": None}, "invalidation missing"),
                     ({"trigger": 0.0, "invalidation": 0.0}, "positive price"),
                     ({"trigger": 0.0379, "invalidation": 0.0392}, "not below price"),
                     ({"trigger": 0.0371, "invalidation": 0.0370}, "not above price"),
                     ({"trigger": 0.0001, "invalidation": 0.0392}, "50% away")):
        assert why in (A._watch_levels_problem(bad, 0.0379, "sell") or "")
    assert A._watch_levels_problem({"trigger": 0.0382, "invalidation": 0.0370}, 0.0379, "buy") is None
    assert "price unavailable" in A._watch_levels_problem(ok, None, "sell")


def _watch_ctx(trig=0.0371, inv=0.0392, side="sell"):
    return {f"pullback_state_{side}": "exhaustion",
            f"pullback_info_{side}": {"low_ts": T_OPEN, "votes": ["rejection", "rsi_turn"], "trigger": trig,
                                      "invalidation": inv, "depth_atr": 8.2, "retrace": 0.6}}


def _watch_gr(ts=T_OPEN):
    return NS(pair_name="SKYAIUSD", ts_curr=ts, close_curr=0.0379)


def test_pullback_watch_rules_by_pair_state(monkeypatch):
    _one_trade_cfg(monkeypatch)
    mkey = sorted(A.BUY_ALERT_KEYS)[0]

    async def claim(*a, **k):
        return True
    # recorded trade open, OPPOSITE direction (Elliott): suppressed, and the log keeps both directions
    A._PULLBACK_WATCH_QUEUE.clear()
    sdb = _mini_store(); sdb.claim_recent_alert = claim
    _put(sdb, state.RedisKeyPrefix.OUTCOME_PENDING, "SKYAIUSD", "elliott_wave3_buy", T_OPEN - 900, "buy")
    assert asyncio.run(A.send_pullback_watch(_watch_ctx(), _watch_gr(), sdb, None, LOG)) == 0
    ev = _candidates(sdb, "SKYAIUSD")
    assert ev[0]["reason"] == "watch_suppressed_open_trade" and ev[0]["same_direction"] is False \
        and ev[0]["open_direction"] == "buy" and not A._PULLBACK_WATCH_QUEUE
    # shadow open: sent
    sdb = _mini_store(); sdb.claim_recent_alert = claim
    _put(sdb, state.RedisKeyPrefix.SHADOW_PENDING, "SKYAIUSD", mkey, T_OPEN - 900, "sell")
    assert asyncio.run(A.send_pullback_watch(_watch_ctx(), _watch_gr(), sdb, None, LOG)) == 1
    block = A._PULLBACK_WATCH_QUEUE[0][2].replace("\\", "")
    assert "trigger 0.0371" in block and "invalid beyond 0.0392" in block and "Cooldown" not in block
    # cooldown: sent, with the cooldown line
    A._PULLBACK_WATCH_QUEUE.clear()
    sdb = _mini_store(); sdb.claim_recent_alert = claim
    asyncio.run(sdb._start_trade_cooldown("SKYAIUSD", T_OPEN - 900, LOG, "Recorded", "sell"))
    assert asyncio.run(A.send_pullback_watch(_watch_ctx(), _watch_gr(), sdb, None, LOG)) == 1
    assert "Cooldown" in A._PULLBACK_WATCH_QUEUE[0][2]
    # nothing open: sent
    A._PULLBACK_WATCH_QUEUE.clear()
    assert asyncio.run(A.send_pullback_watch(_watch_ctx(), _watch_gr(), _wd(claim), None, LOG)) == 1
    A._PULLBACK_WATCH_QUEUE.clear()
    # bad levels: not sent, no claim taken, logged
    taken = []

    async def claim2(*a, **k):
        taken.append(a)
        return True
    sdb = _mini_store(); sdb.claim_recent_alert = claim2
    assert asyncio.run(A.send_pullback_watch(_watch_ctx(None, 0.0392), _watch_gr(), sdb, None, LOG)) == 0
    assert not taken and _candidates(sdb, "SKYAIUSD")[0]["reason"] == "watch_bad_levels"
    A._PULLBACK_WATCH_QUEUE.clear()


def _wd(claim):
    s = _mini_store()
    s.claim_recent_alert = claim
    return s


def test_pullback_watch_flush_rechecks_a_trade_opened_after_queueing(monkeypatch):
    _one_trade_cfg(monkeypatch)
    mkey = sorted(A.BUY_ALERT_KEYS)[0]
    released, tg = [], _TgCapture()

    async def claim(*a, **k):
        return True

    async def release(pair, key):
        released.append((pair, key))
        return True
    A._PULLBACK_WATCH_QUEUE.clear()
    sdb = _mini_store(); sdb.claim_recent_alert = claim; sdb.release_recent_alert = release
    assert asyncio.run(A.send_pullback_watch(_watch_ctx(), _watch_gr(), sdb, None, LOG)) == 1
    asyncio.run(sdb.record_pending_outcome("SKYAIUSD", mkey, "buy", T_OPEN, 0.0379, 5.0, 10.0))   # a trade opens later in the run
    assert asyncio.run(A.flush_pullback_watch(sdb, tg, LOG)) == 0 and not tg.sent
    assert released == [("SKYAIUSD", f"pbwatch_sell_{T_OPEN}")]            # claim given back: it can be sent after the close
    assert _candidates(sdb, "SKYAIUSD")[-1]["reason"] == "watch_suppressed_open_trade"
    # with no open trade the same queue is sent as before
    A._PULLBACK_WATCH_QUEUE.clear()
    sdb = _mini_store(); sdb.claim_recent_alert = claim; sdb.release_recent_alert = release
    asyncio.run(A.send_pullback_watch(_watch_ctx(), _watch_gr(), sdb, None, LOG))
    assert asyncio.run(A.flush_pullback_watch(sdb, tg, LOG)) == 1 and len(tg.sent) == 1


def test_admits_shadow_leaves_out_rows_that_could_not_have_been_followed():
    import threshold_repair as tr
    policy = {"relax_gates": ["confluence_gate"]}
    row = {"alert_key": "x", "context": {"rejection_reason": "confluence_gate"}}
    assert tr._admits_shadow(row, policy, {}) is True                       # an ordinary blocked opportunity
    for flag in ("recorded_active", "cooldown_active"):
        flagged = {**row, "context": {**row["context"], flag: True}}
        assert tr._admits_shadow(flagged, policy, {}) is False              # could not have been followed


def test_cokey_credit_keys_and_reader_merge(monkeypatch):
    _one_trade_cfg(monkeypatch)
    buy = sorted(A.BUY_ALERT_KEYS)
    sell = sorted(A.SELL_ALERT_KEYS)
    primary, b1, b2, s1 = buy[0], buy[1], buy[2], sell[0]
    f = state.RedisStateStore._cokey_credit_keys
    ctx = {"co_fired_keys": [b1, s1, b2, primary, b1, "elliott_wave3_buy", "not_a_key"]}
    assert f(primary, "buy", ctx) == [b1, b2]                       # same direction, deduplicated, no primary/Elliott/unknown
    assert f(primary, "buy", None) == [] and f(primary, "buy", {"co_fired_keys": None}) == []
    monkeypatch.setattr(cfg, "ENABLE_COKEY_STATS_CREDIT", True, raising=False)
    monkeypatch.setattr(cfg, "MIN_WIN_RATE_SAMPLE", 20, raising=False)
    sdb = _mini_store()
    sdb._redis.d[f"alert_stats:ETHUSD:{b1}"] = {"wins": "5", "losses": "14"}
    sdb._redis.d[f"alert_stats_co:ETHUSD:{b1}"] = {"wins": "1", "losses": "0"}
    out = asyncio.run(sdb.batch_get_alert_win_rates("ETHUSD", [b1, b2]))
    assert out[b1] == (6 / 20, 20) and out[b2] == (None, 0)          # 5+1 wins of 20: the credit reaches the minimum sample
    monkeypatch.setattr(cfg, "ENABLE_COKEY_STATS_CREDIT", False, raising=False)
    assert asyncio.run(sdb.batch_get_alert_win_rates("ETHUSD", [b1]))[b1] == (None, 19)   # switch off: own samples only


def test_dedup_summary_reports_delivered_but_unscored_alerts():
    base = {"claims_taken": 1}
    assert "not scored" not in A.format_dedup_summary(dict(base))
    line = A.format_dedup_summary({**base, "unscored_open_trade": 2, "unscored_cooldown": 1})
    assert "delivered but not scored: trade open 2, cooldown 1" in line

# ======================================================================
# graded candidate rows: signals that were sent or seen but never recorded
# ======================================================================
def _cand_cfg(monkeypatch):
    for k, v in dict(ENABLE_BRAIN=True, CANDIDATE_LOG_MAX_PER_PAIR=300, CANDIDATE_RESOLVE_OUTCOMES=True,
                     OUTCOME_MAE_LOSS_PCT=1.0, OUTCOME_RR_TARGET=2.0, OUTCOME_LOOKAHEAD_CANDLES=12,
                     OUTCOME_FILL_DELAY_CANDLES=1, BRAIN_FEE_PCT=0.0006, BRAIN_SLIPPAGE_PCT=0.0003,
                     OUTCOME_COST_MODEL="flat").items():
        monkeypatch.setattr(cfg, k, v, raising=False)


def _cand_candles(n=40, hit_at=8):
    """Flat market at 100; candle `hit_at` spikes to 102.5 (a buy's +2% target, a sell's +1% stop)."""
    t0 = int(time.time()) - 3 * 86400
    ts = np.array([t0 + i * 900 for i in range(n)], dtype=np.int64)
    high = np.full(n, 100.3)
    high[hit_at] = 102.5
    data = PriceData(ts=ts, open=np.full(n, 100.0), high=high, low=np.full(n, 99.8),
                     close=np.full(n, 100.0), volume=np.ones(n))
    return data, int(ts[5])


def _cand_log(sdb, pair, key, direction, ts, reason, **extra):
    asyncio.run(sdb.log_candidate_event(pair, key, direction, ts, reason, extra))


def _cand_grade(sdb, pair, data, i15):
    asyncio.run(sdb.resolve_candidate_outcomes(pair, data, i15, LOG))


def _cand_store(monkeypatch):
    _cand_cfg(monkeypatch)
    data, ets = _cand_candles()
    sdb = _mini_store()
    px = dict(entry_price=100.0, score=20.0, total=30.0, votes_on=["a", "b"])
    _cand_log(sdb, "XAUTUSD", "vwap_buy", "buy", ets, "ignored_open_trade", **px)
    _cand_log(sdb, "XAUTUSD", "vwap_buy", "buy", ets, "ignored_cooldown", **px)       # same alert logged twice
    _cand_log(sdb, "XAUTUSD", "ppo_sell", "sell", ets, "ignored_cooldown", **px)
    _cand_log(sdb, "XAUTUSD", "pullback_watch", "buy", ets, "watch_suppressed_open_trade")   # no entry price
    return sdb, data, ets


def test_candidate_rows_are_graded_once_and_only_after_the_horizon(monkeypatch):
    sdb, data, ets = _cand_store(monkeypatch)
    done = f"{state.RedisKeyPrefix.CANDIDATE_RESOLVED}XAUTUSD"
    _cand_grade(sdb, "XAUTUSD", data, 17)                       # fill candle 6 + 12 = 18: not ready yet
    assert done not in sdb._redis.d
    _cand_grade(sdb, "XAUTUSD", data, 18)
    h = sdb._redis.d[done]
    assert sorted(h) == [f"ppo_sell|{ets}", f"vwap_buy|{ets}"]  # the duplicate and the WATCH row are not graded
    buy, sell = json.loads(h[f"vwap_buy|{ets}"]), json.loads(h[f"ppo_sell|{ets}"])
    assert buy["win"] is True and buy["outcome_reason"] == "target_hit" and buy["net_pnl_pct"] == pytest.approx(1.82)
    assert sell["win"] is False and sell["outcome_reason"] == "stop_hit" and sell["net_pnl_pct"] == pytest.approx(-1.18)
    assert buy["reason"] == "ignored_cooldown" and buy["votes_on"] == ["a", "b"] and buy["score"] == 20.0
    assert sdb._redis.ttl[done] == 30 * 86400
    h[f"vwap_buy|{ets}"] = "MARKER"                              # a re-grade would overwrite this
    before = dict(h)
    _cand_grade(sdb, "XAUTUSD", data, 18)                       # a second pass grades nothing again
    assert sdb._redis.d[done] == before and sdb._redis.d[done][f"vwap_buy|{ets}"] == "MARKER"


def test_candidate_grading_touches_nothing_else_and_can_be_switched_off(monkeypatch):
    sdb, data, ets = _cand_store(monkeypatch)
    _cand_log(sdb, "XAUTUSD", "vwap_buy", "buy", ets + 1, "ignored_open_trade", entry_price=100.0)   # not a candle time
    _cand_grade(sdb, "XAUTUSD", data, 18)
    assert set(sdb._redis.d) == {f"{state.RedisKeyPrefix.CANDIDATE_EVENTS}XAUTUSD",
                                 f"{state.RedisKeyPrefix.CANDIDATE_RESOLVED}XAUTUSD"}   # no stream, totals or stats
    done = f"{state.RedisKeyPrefix.CANDIDATE_RESOLVED}XAUTUSD"
    assert f"vwap_buy|{ets + 1}" not in sdb._redis.d[done]
    old = int(time.time()) - 40 * 86400
    sdb._redis.d[done][f"old_key|{old}"] = "{}"
    _cand_grade(sdb, "XAUTUSD", data, 18)
    assert f"old_key|{old}" not in sdb._redis.d[done]            # ids older than 30 days are pruned
    rows = asyncio.run(sdb.load_candidate_resolved())
    assert sorted(r["alert_key"] for r in rows) == ["ppo_sell", "vwap_buy"]
    _cand_log(sdb, "XAUTUSD", "late_buy", "buy", ets + 900, "ignored_cooldown", entry_price=100.0)
    _cand_grade(sdb, "XAUTUSD", data, 19)                        # one candle later: this one is ready too
    assert [r["alert_key"] for r in asyncio.run(sdb.load_candidate_resolved(limit=1))] == ["late_buy"]   # newest first
    sdb.degraded = True
    assert asyncio.run(sdb.load_candidate_resolved()) is None    # a failed read is never "no data"
    sdb.degraded = False

    async def _boom(*a, **k):
        raise RuntimeError("redis down")
        yield                                                    # pragma: no cover
    sdb._redis.scan_iter = _boom
    assert asyncio.run(sdb.load_candidate_resolved()) is None    # an error mid-read is not "no data" either
    off, data2, _e = _cand_store(monkeypatch)
    monkeypatch.setattr(cfg, "CANDIDATE_RESOLVE_OUTCOMES", False, raising=False)
    _cand_grade(off, "XAUTUSD", data2, 18)
    assert f"{state.RedisKeyPrefix.CANDIDATE_RESOLVED}XAUTUSD" not in off._redis.d


def test_candidate_grading_is_capped_per_pass_and_skips_unpriced_rows(monkeypatch):
    sdb, data, ets = _cand_store(monkeypatch)
    for i in range(60):
        _cand_log(sdb, "XAUTUSD", f"k{i}", "buy", ets, "ignored_cooldown", entry_price=100.0)
    _cand_log(sdb, "XAUTUSD", "free_lunch", "buy", ets, "ignored_cooldown", entry_price=0.0)
    done = f"{state.RedisKeyPrefix.CANDIDATE_RESOLVED}XAUTUSD"
    _cand_grade(sdb, "XAUTUSD", data, 18)
    assert len(sdb._redis.d[done]) == 50                         # at most 50 rows per pass
    _cand_grade(sdb, "XAUTUSD", data, 18)
    assert len(sdb._redis.d[done]) == 62                         # the rest (2 + 60 priced rows) follow on the next pass
    assert f"free_lunch|{ets}" not in sdb._redis.d[done]         # a zero entry price is never graded


def test_outcome_resolution_pass_also_grades_candidates(monkeypatch):
    import gates
    calls = []

    class _Sdb:
        async def resolve_pending_outcomes(self, *a):
            calls.append("real")

        async def resolve_shadow_pending_outcomes(self, *a):
            calls.append("shadow")

        async def resolve_candidate_outcomes(self, *a):
            calls.append("candidate")

    for k, v in dict(ENABLE_WIN_RATE_FILTER=True, ENABLE_BRAIN=True, BRAIN_SHADOW_MODE=True).items():
        monkeypatch.setattr(cfg, k, v, raising=False)
    asyncio.run(gates._resolve_pair_outcomes("X", None, 5, _Sdb(), LOG))
    assert calls == ["real", "shadow", "candidate"]
    calls.clear()
    monkeypatch.setattr(cfg, "BRAIN_SHADOW_MODE", False, raising=False)
    asyncio.run(gates._resolve_pair_outcomes("X", None, 5, _Sdb(), LOG))
    assert calls == ["real", "candidate"]
    calls.clear()
    monkeypatch.setattr(cfg, "ENABLE_WIN_RATE_FILTER", False, raising=False)
    asyncio.run(gates._resolve_pair_outcomes("X", None, 5, _Sdb(), LOG))
    assert calls == []
