"""Slim 8-section Brain report: structure, honesty caps, Telegram safety."""

import asyncio
import random
import re
import time

import brain_enhanced as be
from archive_reader import _parse_jsonl_row
from bot_config import cfg
from brain_audit import HealthStatus, get_audit
from outcome_storage import OUTCOME_SCHEMA_VERSION

_NOW = int(time.time())
_SPECIAL = set("_*[]()~>#+-=|{}.!")


def _rows(spec, days=2.5, seed=3, positive=()):
    rnd = random.Random(seed)
    raws, i = [], 0
    total = sum(c for _, _, c in spec)
    for ak, d, cnt in spec:
        for _ in range(cnt):
            i += 1
            if ak in positive:
                reason, win, mfe, mae, mv = "target_hit", True, 0.045, 0.005, rnd.uniform(1.0, 2.0)
            else:
                r = rnd.random()
                if r < 0.05:
                    reason, win, mfe, mae, mv = "target_hit", True, 0.045, 0.005, 4.5
                elif r < 0.65:
                    reason, win, mfe, mae, mv = "stop_hit", False, 0.006, 0.021, rnd.uniform(-2.5, -1.0)
                else:
                    reason, win, mfe, mae, mv = "no_hit", False, 0.007, 0.012, rnd.uniform(-1, 1)
            raws.append({
                "pair": rnd.choice(["ETHUSD", "BTCUSD", "SOLUSD", "BNBUSD"]),
                "alert_key": ak, "direction": d,
                "entry_ts": _NOW - int(days * 86400) + int(i * days * 86400 / total),
                "score": rnd.uniform(20, 40), "total": 50.0, "win": win, "pct_move": mv,
                "mfe": mfe, "mae": mae, "outcome_reason": reason, "tp_first": win,
                "schema_version": OUTCOME_SCHEMA_VERSION,
                "session": rnd.choice(["asian", "ny", "london"]),
            })
    return [x for x in (_parse_jsonl_row(r) for r in raws) if x]


_SPEC = [("choch_sell", "sell", 40), ("dynamic_flow_cross_buy", "buy", 25),
         ("vwap_down", "sell", 16), ("fib_reversal_buy", "buy", 9)]
_BLOCKED = {"actionable": False, "data_quality": True, "oos_prediction": False,
            "profitability": False, "stability": False, "risk": False, "execution": True}
_OPEN = {k: True for k in _BLOCKED}

def _report(rows, gate=_BLOCKED, net_ev=-0.38, days_override=None, patch=None, ai_extra=None):
    a = get_audit()
    a.begin_cycle()
    a.set_history_coverage(rows, requested_days=180)
    if days_override is not None:
        a._history.actual_days = days_override
    a.set_reconciliation(pending_count=45, resolved_this_run=0, archived_this_run=0,
                         total_archived=len(rows), loaded_by_brain=len(rows), shadow_loaded=0)
    a.record_analysis("monte_carlo", HealthStatus.INSUFFICIENT_DATA, detail="need 21 days")
    ai_metrics = {"net_ev": net_ev, "action_gate": gate, **(ai_extra or {})}
    recs = {"_real_rows": rows, "_shadow_rows": [], "config_patch": patch or [],
            "ai_metrics": ai_metrics, "_archive_stats": {}}
    return be.build_brain_report(recs, cfg)

def _plain(msgs):
    return "\n".join(re.sub(r"\\(.)", r"\1", m) for m in msgs)

def test_confidence_is_capped_by_history_span():
    a = get_audit()
    a.begin_cycle()
    rows = _rows(_SPEC, days=2.5)
    a.set_history_coverage(rows * 4, requested_days=180)      # 4x rows: n>=300, still 2.5 days
    assert a.statistical_confidence_label() == "VERY LOW"
    a._history.actual_days = 20
    assert a.statistical_confidence_label() == "MODERATE"
    a._history.actual_days = 45
    assert a.statistical_confidence_label() == "HIGH"


def test_all_eight_sections_in_order_and_fit_telegram():
    msgs = _report(_rows(_SPEC))
    text = _plain(msgs)
    positions = [text.index(f"{i:02d} │") for i in range(1, 9)]
    assert positions == sorted(positions)
    assert "BRAIN REPORT" in msgs[0] and "END OF BRAIN REPORT" in _plain(msgs[-1:])
    assert all(len(m) <= 4096 for m in msgs)
    assert text.index("05 │") < text.index("CONTEXT (sessions") < text.index("06 │")
  
    # Dropped technical sections must not appear
    for gone in ("09 │", "10 │", "11 │", "12 │", "13 │", "14 │", "15 │", "16 │"):
        assert gone not in text

def test_counterfactual_candidates_beat_control_shown_in_what_to_do_now():
    scenarios = [
        {"label": "Threshold +1", "ev": 0.81, "delta_ev": 0.39,
         "n": 284, "shadow_validated": True, "shadow_n": 137},
        {"label": "RSI cap -3", "ev": 0.55, "delta_ev": 0.13,
         "n": 198, "shadow_validated": True, "shadow_n": 8},
        {"label": "Combined", "ev": 0.91, "delta_ev": 0.49,
         "n": 142, "shadow_validated": False, "shadow_n": 22},
        {"label": "Threshold +2 (worse than control)", "ev": 0.30, "delta_ev": -0.12,
         "n": 90, "shadow_validated": None, "shadow_n": None},
        {"label": "Session filter", "ev": 0.50, "delta_ev": 0.08,
         "n": 300, "shadow_validated": True, "shadow_n": 15},
        {"label": "4th best — should be cut by the top-3 cap", "ev": 0.45, "delta_ev": 0.03,
         "n": 60, "shadow_validated": True, "shadow_n": 20},
    ]
    text = _plain(_report(_rows(_SPEC), ai_extra={"counterfactual_scenarios": scenarios}))
    sec2 = text[text.index("02 │"):text.index("03 │")]
    assert "CANDIDATE CONFIGS" in sec2
    # Ranked best (Combined, ev=0.91) to 3rd (RSI cap -3, ev=0.55); Session
    # filter (0.50) and the 4th-best (0.45) are positive but cut by the cap.
    assert sec2.index("Combined") < sec2.index("Threshold +1") < sec2.index("RSI cap -3")
    assert "Threshold +2 (worse than control)" not in sec2   # delta_ev < 0, never a candidate
    assert "Session filter" not in sec2                       # positive but outside top 3
    assert "4th best — should be cut by the top-3 cap" not in sec2
    assert "PROMOTION-ELIGIBLE" in sec2                       # Threshold +1: validated, n=137 >= 15
    assert "REJECTED" in sec2                                 # Combined: shadow disagrees
    assert "NOT YET (need 15 shadow, have 8)" in sec2         # RSI cap -3: validated but thin

def test_low_history_says_diagnose_and_never_advises_disabling():
    text = _plain(_report(_rows(_SPEC)))
    assert "NEEDS ATTENTION" in text and "DIAGNOSE + COLLECT DATA" in text
    assert "NOT yet \"disable\" candidates" in text
    assert "Do not disable alerts solely from this report." in text
    assert "DO NOT CHANGE YET" in text
    assert "APPROVED TO APPLY" not in text
    assert "APPROVED CHANGES" not in text

def test_win_rule_explained_with_break_even():
    text = _plain(_report(_rows(_SPEC)))
    assert "WHY THE WIN RATE LOOKS LOW" in text
    assert "before it moves -2.0% against you, within 3 hours" in text
    assert "Break-even needs roughly 33% wins" in text


def test_scorecard_uses_friendly_names_and_summarises_thin():
    text = _plain(_report(_rows(_SPEC)))
    score = text.split("06 │")[1].split("07 │")[0]
    assert "choch_sell" not in score
    assert "INSUFFICIENT DATA" in score
    assert "see archive report if needed" in score
    # At least one pretty name appears somewhere in the report
    assert "CHoCH SELL" in text or "Dynamic Flow BUY" in text

def test_telegram_markdownv2_is_valid():
    for m in _report(_rows(_SPEC)):
        parts = m.split("```")
        assert len(parts) % 2 == 1, "unbalanced code fence"
        for idx in range(0, len(parts), 2):             # prose parts only
            prose = parts[idx]
            for i, ch in enumerate(prose):
                if ch in _SPECIAL:
                    assert i > 0 and prose[i - 1] == "\\", f"unescaped {ch!r} near {prose[max(0, i-20):i+5]!r}"

def test_validated_mode_shows_verdict_and_approved_block():
    spec = [("ppo_signal_up", "buy", 80), ("vwap_down", "sell", 40), ("choch_sell", "sell", 40)]
    rows = _rows(spec, days=45, positive=("ppo_signal_up",))
    patch = [{"path": "CONFLUENCE_MIN_ABS_SCORE", "current": 21, "suggested": 22, "reason": "x"}]
    text = _plain(_report(rows, gate=_OPEN, net_ev=0.6, days_override=45, patch=patch))
    assert "BRAIN VERDICT" in text and "CURRENTLY VALIDATED EDGE" in text
    assert "PPO Signal UP" in text and "Action Gate: APPROVED" in text
    # Slim report: approved patches live in section 02, not a separate "DO NOT APPLY" block
    assert "APPROVED CHANGES" in text or "CONFLUENCE_MIN_ABS_SCORE" in text
    assert "DO NOT APPLY" not in text
    assert "DO NOT CHANGE YET" not in text

def test_empty_data_does_not_crash():
    msgs = _report([])
    assert msgs and "01 │" in _plain(msgs)


def _engine(monkeypatch, sent):
    class Q:
        async def send(self, m):
            sent.append(m)
            return True

    eng = object.__new__(be.BrainEngineV2)
    rows = _rows(_SPEC)
    a = get_audit()
    a.begin_cycle()
    a.set_history_coverage(rows, requested_days=180)

    async def fake_recs():
        return {"_real_rows": rows, "_shadow_rows": [], "config_patch": [], "_archive_stats": {},
                "ai_metrics": {"net_ev": -0.38, "action_gate": _BLOCKED}}

    async def fake_store(recs):
        return None

    monkeypatch.setattr(eng, "generate_recommendations", fake_recs)
    monkeypatch.setattr(eng, "_store_pending_plan", fake_store)
    return eng, Q()

def test_report_is_archived_as_markdown_and_still_sent(monkeypatch):
    import logging
    import outcome_storage
    saved, sent = [], []
    monkeypatch.setattr(outcome_storage, "save_report", lambda md: saved.append(md) or "reports/x.md")
    eng, q = _engine(monkeypatch, sent)
    assert asyncio.run(eng.generate_report([], q, logging.getLogger("t"))) is True
    assert len(saved) == 1 and len(sent) >= 1
    md = saved[0]
    assert md.startswith("# 🧠 Brain Report")
    # Slim trader report: 8 sections
    assert len(re.findall(r"^## \d\d │", md, re.M)) == 8
    assert md.count("```") % 2 == 0
    assert not re.search(r"\\[.\-()!+=|]", md)           # no Telegram MarkdownV2 escaping in the file
    assert "CONTEXT (sessions" in _plain(sent) or "Action Gate" in _plain(sent) or "08 │" in md

def test_archive_failure_is_not_fatal_and_sends_once(monkeypatch):
    import logging
    import outcome_storage
    sent = []

    def boom(md):
        raise OSError("disk full")

    monkeypatch.setattr(outcome_storage, "save_report", boom)
    eng, q = _engine(monkeypatch, sent)
    assert asyncio.run(eng.generate_report([], q, logging.getLogger("t"))) is True
    assert "END OF BRAIN REPORT" in _plain(sent[-1:])
    assert not any("classic" in m for m in sent)               # no fallback report was sent


def test_generate_report_falls_back_to_base_report(monkeypatch):
    import logging
    sent, called = [], []
    eng, q = _engine(monkeypatch, sent)
    monkeypatch.setattr(be, "build_brain_report_sections",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))

    async def base_report(pairs, telegram_queue, logger_run):
        called.append(True)
        return True

    monkeypatch.setattr(eng, "_generate_and_send", base_report)
    assert asyncio.run(eng.generate_report([], q, logging.getLogger("t"))) is True
    assert called == [True] and sent == []


def test_build_profit_action_plan_is_gone():
    assert not hasattr(be, "build_profit_action_plan")