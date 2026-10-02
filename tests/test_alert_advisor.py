"""Conviction / verdict logic and the per-pair Telegram body."""
import logging
from types import SimpleNamespace as NS

import alerts as A
from alert_advisor import advise_pair, AVOID, TAKE, WATCH

LOG = logging.getLogger("t")
DOWN = NS(up_pct=0.20, down_pct=0.47)
UP = NS(up_pct=0.64, down_pct=0.21)
GOOD_VOTES = {"base_trend": True, "dynamic_flow_ribbon": True, "ichimoku_cloud": True,
              "oi_funding": True, "adx": False}


def tq(ev=0.04, p=0.5, shadow=True, regime=("trending", 0.40), plan=(1.18, 1.76, 2.67, 0.5, 23), **kw):
    t = {"verdict": "LOW", "p_ev_positive": p, "p_ev_positive_ensemble": p, "net_ev": ev,
         "evidence_state": "SHADOW" if shadow else "ACTIONABLE"}
    if regime:
        t["regime_name"], t["regime_wr"] = regime
    if plan:
        t["trade_plan"] = {"sl_suggested_pct": plan[0], "tp1_suggested_pct": plan[1],
                           "tp2_suggested_pct": plan[2], "tp_first_rate": plan[3], "n": plan[4]}
    t.update(kw)
    return t


def adv(tqs, bias="with", votes=GOOD_VOTES, score=24, total=27, **kw):
    return advise_pair(direction="buy", score=score, total=total, required=21, votes=votes,
                       tqs=tqs, bias=bias, **kw)


def test_negative_ev_is_avoid_even_with_weak_evidence():
    a = adv([tq(ev=-0.07)])
    assert a.verdict == AVOID and a.conviction < 50
    assert "negative" in a.why_line


def test_shadow_evidence_can_never_take():
    a = adv([tq(ev=0.9, p=0.9, regime=("trending", 0.8), plan=(1, 2, 3, 0.9, 50))])
    assert a.verdict != TAKE and a.conviction <= 55


def test_proven_edge_with_bias_is_take():
    a = adv([tq(ev=0.9, p=0.82, shadow=False, regime=("trending", 0.64), plan=(1.1, 1.65, 2.5, 0.64, 120),
                size_hint=1.0)])
    assert a.verdict == TAKE and a.conviction >= 70 and a.confidence == "HIGH"
    assert "Size 1.00" in a.size_line


def test_conflicting_signals_downgrade_take_to_watch():
    kw = dict(tqs=[tq(ev=0.9, p=0.82, shadow=False, regime=("trending", 0.64), plan=(1.1, 1.65, 2.5, 0.64, 120))])
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
    assert A._format_price(0.0462) == "$0.0462"
    assert A._format_price(0.00731) == "$0.007310"
    assert A._format_price(1.1934) == "$1.19" and A._format_price(108420) == "$108,420.00"


def _msg(**over):
    kw = dict(pair="LABUSD", direction="sell", price=0.0462, ts=1790826300, score=24, total=27,
              items=[("🟣▼ VWAP Cross", ""), ("🌊🔴 Dynamic Flow Cross SELL", "")],
              keys=["vwap_down", "dynamic_flow_cross_sell"],
              tq_by_key={"vwap_down": tq(ev=0.04), "dynamic_flow_cross_sell": tq(ev=-0.16, plan=(1.18, 1.27, 2.34, 0.0, 42))},
              votes=GOOD_VOTES, required=21, bias_context=DOWN, logger_pair=LOG)
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
    m = _msg(direction="buy", items=[("🟢🔄 Strong Reversal BUY", ""), ("🔴🌀 Fib Pivot Reversal SELL", "")],
             keys=["strong_reversal_buy", "fib_reversal_sell"], tq_by_key={})
    assert "Fib Pivot" not in m.split("\n")[2] and "opposing signal" in m


def test_bad_tq_degrades_instead_of_raising():
    m = _msg(tq_by_key={"vwap_down": {"net_ev": "abc", "trade_plan": {"n": 3}}})
    assert m.startswith("🟣▼ LABUSD") and "🎯 Conviction n/a" in m


def test_formatter_failure_falls_back_to_plain_body(monkeypatch):
    monkeypatch.setattr(A, "advise_pair", lambda **k: (_ for _ in ()).throw(RuntimeError("boom")))
    m = _msg()
    assert "LABUSD" in m and "VWAP Cross" in m and "Conviction" not in m


def test_markdown_v2_special_chars_are_escaped():
    raw = A.build_pair_msg_safe(**{**dict(
        pair="LABUSD", direction="sell", price=0.0462, ts=1790826300, score=24, total=27,
        items=[("🟣▼ VWAP Cross", "")], keys=["vwap_down"], tq_by_key={"vwap_down": tq()},
        votes=GOOD_VOTES, required=21, bias_context=DOWN, logger_pair=LOG)})
    for ch in "().-+|":
        assert raw.count(ch) == raw.count("\\" + ch), ch
