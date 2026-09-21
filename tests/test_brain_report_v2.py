"""Layered 16-section Brain report: structure, honesty caps, Telegram safety."""
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


def _report(rows, gate=_BLOCKED, net_ev=-0.38, days_override=None, patch=None):
    a = get_audit()
    a.begin_cycle()
    a.set_history_coverage(rows, requested_days=180)
    if days_override is not None:
        a._history.actual_days = days_override
    a.set_reconciliation(pending_count=45, resolved_this_run=0, archived_this_run=0,
                         total_archived=len(rows), loaded_by_brain=len(rows), shadow_loaded=0)
    a.record_analysis("monte_carlo", HealthStatus.INSUFFICIENT_DATA, detail="need 21 days")
    recs = {"_real_rows": rows, "_shadow_rows": [], "config_patch": patch or [],
            "ai_metrics": {"net_ev": net_ev, "action_gate": gate}, "_archive_stats": {}}
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


def test_all_sixteen_sections_in_order_and_fit_telegram():
    msgs = _report(_rows(_SPEC))
    text = _plain(msgs)
    positions = [text.index(f"{i:02d} │") for i in range(1, 17)]
    assert positions == sorted(positions)
    assert "BRAIN REPORT" in msgs[0] and "END OF BRAIN REPORT" in _plain(msgs[-1:])
    assert all(len(m) <= 4096 for m in msgs)
    assert text.index("05 │") < text.index("THE EVIDENCE BEHIND") < text.index("06 │")


def test_low_history_says_diagnose_and_never_advises_disabling():
    text = _plain(_report(_rows(_SPEC)))
    assert "NEEDS ATTENTION" in text and "DIAGNOSE + COLLECT DATA" in text
    assert "NOT yet \"disable\" candidates" in text
    assert "Do not disable alerts solely from this report." in text
    assert "DO NOT APPLY" in text and "APPROVED TO APPLY" not in text


def test_win_rule_explained_with_break_even():
    text = _plain(_report(_rows(_SPEC)))
    assert "WHY THE WIN RATE LOOKS LOW" in text
    assert "before it moves -2.0% against you, within 3 hours" in text
    assert "Break-even needs roughly 33% wins" in text


def test_scorecard_lists_every_alert_and_names_are_friendly():
    text = _plain(_report(_rows(_SPEC)))
    score = text.split("06 │")[1].split("07 │")[0]
    for name in ("CHoCH SELL", "Dynamic Flow BUY", "VWAP DOWN", "Fib Reversal BUY"):
        assert name in score, name
    assert "choch_sell" not in score


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
    assert "APPROVED TO APPLY" in text and "DO NOT APPLY" not in text


def test_empty_data_does_not_crash():
    msgs = _report([])
    assert msgs and "01 │" in _plain(msgs)


def test_generate_report_falls_back_to_classic_plan(monkeypatch):
    sent = []

    class Q:
        async def send(self, m):
            sent.append(m)
            return True

    eng = object.__new__(be.BrainEngineV2)

    async def fake_recs():
        return {"_real_rows": [], "ai_metrics": {}}

    async def fake_store(recs):
        return None

    monkeypatch.setattr(eng, "generate_recommendations", fake_recs)
    monkeypatch.setattr(eng, "_store_pending_plan", fake_store)
    monkeypatch.setattr(be, "build_brain_report", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(be, "build_profit_action_plan", lambda *a, **k: ["classic"])
    import logging
    ok = asyncio.run(eng.generate_report([], Q(), logging.getLogger("t")))
    assert ok is True and sent == ["classic"]