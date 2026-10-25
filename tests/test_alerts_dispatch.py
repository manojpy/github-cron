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
    assert "⏰" in text and "📆" in text and "Bias" in text


def test_basket_note_when_three_same_direction_takes():
    ps = [_p(n, "buy", 26, 27, "TAKE") for n in ("AAAUSD", "BBBUSD", "CCCUSD")] \
         + [_p("DDDUSD", "buy", 20, 27, "WATCH")]
    text = _send(ps, BIAS_UP)
    assert text.count("size them as ONE basket") == 1 and "3 BUY TAKEs" in text


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


