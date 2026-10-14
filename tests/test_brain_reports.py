from __future__ import annotations
from bot_config import cfg, json_dumps, json_loads
import asyncio
import random
import re
import time
import brain_enhanced as be
from archive_reader import _parse_jsonl_row
from brain_audit import HealthStatus, get_audit
from outcome_storage import OUTCOME_SCHEMA_VERSION
import json
import confluence_tier_report as ctr
from datetime import datetime, timedelta, timezone
from pathlib import Path
import cleanup_outcomes as co
import alerts
import outcome_storage
from brain_enhanced import BrainEngineV2, PLAN_HISTORY_KEY
import logging
import threshold_engine as engine

# ======================================================================
# from test_brain_report_v2.py
# ======================================================================
"""Slim 9-section Brain report: structure, honesty caps, Telegram safety."""



_NOW = int(time.time())
_SPECIAL = set("_*[]()~>#+-=|{}.!")


def _rows__brain_report_v2(spec, days=2.5, seed=3, positive=()):
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
    rows = _rows__brain_report_v2(_SPEC, days=2.5)
    a.set_history_coverage(rows * 4, requested_days=180)      # 4x rows: n>=300, still 2.5 days
    assert a.statistical_confidence_label() == "VERY LOW"
    a._history.actual_days = 20
    assert a.statistical_confidence_label() == "MODERATE"
    a._history.actual_days = 45
    assert a.statistical_confidence_label() == "HIGH"

def test_all_nine_sections_in_order_and_fit_telegram():
    msgs = _report(_rows__brain_report_v2(_SPEC))
    text = _plain(msgs)
    positions = [text.index(f"{i:02d} │") for i in range(1, 10)]  # 01..09
    assert positions == sorted(positions)
    assert "BRAIN REPORT" in msgs[0] and "END OF BRAIN REPORT" in _plain(msgs[-1:])
    assert all(len(m) <= 4096 for m in msgs)
    # Human pack ends after section 06; context starts at 07
    assert text.index("06 │") < text.index("CONTEXT (sessions") < text.index("07 │")

    for gone in ("10 │", "11 │", "12 │", "13 │", "14 │", "15 │", "16 │"):
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
    text = _plain(_report(_rows__brain_report_v2(_SPEC), ai_extra={"counterfactual_scenarios": scenarios}))
    # WHAT TO DO NOW is section 03 after BRAIN DECISION was inserted as 01
    sec3 = text[text.index("03 │"):text.index("04 │")]
    assert "CANDIDATE CONFIGS" in sec3
    # Ranked best (Combined, ev=0.91) to 3rd (RSI cap -3, ev=0.55); Session
    # filter (0.50) and the 4th-best (0.45) are positive but cut by the cap.
    assert sec3.index("Combined") < sec3.index("Threshold +1") < sec3.index("RSI cap -3")
    assert "Threshold +2 (worse than control)" not in sec3   # delta_ev < 0, never a candidate
    assert "Session filter" not in sec3                       # positive but outside top 3
    assert "4th best — should be cut by the top-3 cap" not in sec3
    assert "PROMOTION-ELIGIBLE" in sec3                       # Threshold +1: validated, n=137 >= 15
    assert "REJECTED" in sec3                                 # Combined: shadow disagrees
    assert "NOT YET (need 15 shadow, have 8)" in sec3         # RSI cap -3: validated but thin

def test_confidence_breakdown_shows_four_independent_tiers():
    gate = {**_OPEN, "profit_p_ev_positive": 0.91, "ev_p5": 0.12,
            "oos_p_ev_positive": 0.61, "stability": True}
    text = _plain(_report(_rows__brain_report_v2(_SPEC), gate=gate, ai_extra={"brier_score": 0.03}))

    # ACTION GATE is section 09
    sec9 = text[text.index("09 │"):]
    assert "CONFIDENCE BREAKDOWN" in sec9
    assert "🟢 MODEL       HIGH" in sec9 and "Brier 0.03" in sec9

    # _SPEC totals 90 trades over the default 2.5-day window: n>=60 alone
    # would rank HIGH, but the short span caps it to MEDIUM.
    assert "🔴 DATA        NOT READY" in sec9 and "90 trades" in sec9
    assert "🟢 CHANGE      HIGH" in sec9 and "P(EV>0) 91%" in sec9 and "EV p5 +0.12%" in sec9
    assert "🟡 DEPLOYMENT  MEDIUM" in sec9 and "OOS P(EV>0) 61%" in sec9

def test_confidence_breakdown_not_ready_on_active_drift_regardless_of_oos_score():
    # A near-perfect OOS score must still show NOT READY once stability
    # has failed — deploying into active drift is never "ready", whatever
    # the historical walk-forward number says.
    gate = {**_OPEN, "profit_p_ev_positive": 0.91, "ev_p5": 0.12,
            "oos_p_ev_positive": 0.95, "stability": False}
    text = _plain(_report(_rows__brain_report_v2(_SPEC), gate=gate, ai_extra={"brier_score": None}))
    # ACTION GATE is section 09
    sec9 = text[text.index("09 │"):]

    assert "⚪ MODEL       N/A" in sec9 and "no calibration data yet" in sec9
    assert "🔴 DEPLOYMENT  NOT READY" in sec9 and "active drift" in sec9

def test_low_history_says_diagnose_and_never_advises_disabling():
    text = _plain(_report(_rows__brain_report_v2(_SPEC)))
    assert "NEEDS ATTENTION" in text and "DIAGNOSE + COLLECT DATA" in text
    assert "NOT yet \"disable\" candidates" in text
    assert "Do not disable alerts solely from this report." in text
    assert "DO NOT CHANGE YET" in text
    assert "APPROVED TO APPLY" not in text
    assert "APPROVED CHANGES" not in text

def test_win_rule_explained_with_break_even():
    text = _plain(_report(_rows__brain_report_v2(_SPEC)))
    assert "WHY THE WIN RATE LOOKS LOW" in text
    assert "before it moves -1.0% against you, within 3 hours" in text
    assert "Break-even needs roughly 33% wins" in text

def test_scorecard_uses_friendly_names_and_summarises_thin():
    text = _plain(_report(_rows__brain_report_v2(_SPEC)))
    # ALERT SCORECARD is section 07; SESSION ANALYSIS is 08
    score = text.split("07 │")[1].split("08 │")[0]
    assert "choch_sell" not in score
    assert "INSUFFICIENT DATA" in score
    assert "see archive report if needed" in score
    # At least one pretty name appears somewhere in the report
    assert "CHoCH SELL" in text or "Dynamic Flow BUY" in text

def test_telegram_markdownv2_is_valid():
    for m in _report(_rows__brain_report_v2(_SPEC)):
        parts = m.split("```")
        assert len(parts) % 2 == 1, "unbalanced code fence"
        for idx in range(0, len(parts), 2):             # prose parts only
            prose = parts[idx]
            for i, ch in enumerate(prose):
                if ch in _SPECIAL:
                    assert i > 0 and prose[i - 1] == "\\", f"unescaped {ch!r} near {prose[max(0, i-20):i+5]!r}"

def test_validated_mode_shows_verdict_and_approved_block():
    spec = [("ppo_signal_up", "buy", 80), ("vwap_down", "sell", 40), ("choch_sell", "sell", 40)]
    rows = _rows__brain_report_v2(spec, days=45, positive=("ppo_signal_up",))
    patch = [{"path": "CONFLUENCE_MIN_ABS_SCORE", "current": 21, "suggested": 22, "reason": "x"}]
    text = _plain(_report(rows, gate=_OPEN, net_ev=0.6, days_override=45, patch=patch))
    assert "BRAIN VERDICT" in text and "CURRENTLY VALIDATED EDGE" in text
    assert "PPO Signal UP" in text and "Action Gate: APPROVED" in text
    # Slim report: approved patches live in WHAT TO DO NOW (03), not a separate "DO NOT APPLY" block
    assert "APPROVED CHANGES" in text or "CONFLUENCE_MIN_ABS_SCORE" in text
    assert "DO NOT APPLY" not in text
    assert "DO NOT CHANGE YET" not in text

def test_empty_data_does_not_crash():
    msgs = _report([])
    assert msgs and "01 │" in _plain(msgs)


def _engine__brain_report_v2(monkeypatch, sent):
    class Q:
        async def send(self, m):
            sent.append(m)
            return True

    eng = object.__new__(be.BrainEngineV2)
    rows = _rows__brain_report_v2(_SPEC)
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
    eng, q = _engine__brain_report_v2(monkeypatch, sent)
    assert asyncio.run(eng.generate_report([], q, logging.getLogger("t"))) is True
    assert len(saved) == 1 and len(sent) >= 1
    md = saved[0]
    assert md.startswith("# 🧠 Brain Report")
    # Slim trader report: 9 sections (includes BRAIN DECISION)
    assert len(re.findall(r"^## \d\d │", md, re.M)) == 9
    assert md.count("```") % 2 == 0
    assert not re.search(r"\\[.\-()!+=|]", md)           # no Telegram MarkdownV2 escaping in the file
    assert (
        "CONTEXT (sessions" in _plain(sent)
        or "Action Gate" in _plain(sent)
        or "09 │" in md
    )

def test_archive_failure_is_not_fatal_and_sends_once(monkeypatch):
    import logging
    import outcome_storage
    sent = []

    def boom(md):
        raise OSError("disk full")

    monkeypatch.setattr(outcome_storage, "save_report", boom)
    eng, q = _engine__brain_report_v2(monkeypatch, sent)
    assert asyncio.run(eng.generate_report([], q, logging.getLogger("t"))) is True
    assert "END OF BRAIN REPORT" in _plain(sent[-1:])
    assert not any("classic" in m for m in sent)               # no fallback report was sent


def test_generate_report_falls_back_to_base_report(monkeypatch):
    import logging
    sent, called = [], []
    eng, q = _engine__brain_report_v2(monkeypatch, sent)
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


# ======================================================================
# from test_confluence_tier_report.py
# ======================================================================
"""Tier report: grouping maths, verdict wording, and an end-to-end archive read."""



def _row(conf, win, ev, direction="buy"):
    return {"conf_pct": conf, "win": win, "net_pnl_pct": ev, "direction": direction}


def test_groups_rows_into_tiers_and_directions():
    rows = [_row(96, True, 1.0), _row(92, False, -0.5), _row(92, True, 0.8, "sell"),
            _row(86, True, 0.2), _row(70, False, -1.0)]
    rep = ctr.tier_report(rows, bar=90)
    assert rep["tiers"]["95-100%"]["n"] == 1 and rep["tiers"]["90-95%"]["n"] == 2
    assert rep["tiers"]["<80%"]["n"] == 1
    assert rep["by_direction"]["sell"]["90-95%"]["n"] == 1
    assert rep["at_or_above_bar"]["n"] == 3
    assert abs(rep["tiers"]["90-95%"]["net_ev"] - 0.15) < 1e-9


def test_verdict_needs_enough_data_then_judges_edge():
    few = ctr.tier_report([_row(95, True, 1.0)] * 5)
    assert few["verdict"].startswith("NOT ENOUGH DATA")
    good = ctr.tier_report([_row(95, True, 0.9)] * 20 + [_row(95, False, -0.4)] * 15)
    assert good["verdict"].startswith("SUPPORTED")
    bad = ctr.tier_report([_row(95, True, 0.2)] * 10 + [_row(95, False, -1.0)] * 25)
    assert bad["verdict"].startswith("NOT SUPPORTED")


def test_empty_input_does_not_crash():
    rep = ctr.tier_report([])
    assert rep["n_total"] == 0 and "NOT ENOUGH DATA" in rep["verdict"]
    assert "Confluence tier report" in ctr.render(rep)


def test_reads_the_real_archive_format(tmp_path, capsys):
    import time
    day = time.strftime("%Y-%m-%d", time.gmtime())
    d = tmp_path / "outcomes"
    d.mkdir()
    base = dict(pair="AAVEUSD", alert_key="vwap_up", direction="buy", entry_ts=int(time.time()) - 3600,
                score=26, total=27, pct_move=1.2, win=True, session="asia", votes={}, context={},
                mae=0.2, mfe=1.5, close_win=True, mfe_win=True, mae_loss=False, tp_first=True,
                outcome_reason="tp", bonus_win=False, rr_achieved=1.5, win_weight=1.0,
                net_pnl_pct=1.0, realized_cost_pct=0.1, adx_val=30.0, schema_version=OUTCOME_SCHEMA_VERSION)
    with open(d / f"{day}.jsonl", "w") as f:
        for i in range(3):
            f.write(json.dumps({**base, "outcome_id": f"id{i}", "sid": f"s{i}"}) + "\n")
    assert ctr.main(["--data-dir", str(tmp_path), "--days", "5", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["n_total"] == 3 and out["tiers"]["95-100%"]["n"] == 3 and out["tiers"]["90-95%"]["n"] == 0


# ======================================================================
# from test_report_retention.py
# ======================================================================
"""Reports are kept 7 days (exact, from the file name); outcomes keep their own limit."""



def _mk(dirpath: Path, name: str) -> Path:
    dirpath.mkdir(parents=True, exist_ok=True)
    f = dirpath / name
    f.write_text("x")
    return f


def _report_name(age: timedelta) -> str:
    return (datetime.now(timezone.utc) - age).strftime("%Y-%m-%d_%H-%M") + ".md"


def test_reports_older_than_7_days_removed_newer_kept(tmp_path):
    new = _mk(tmp_path / "reports", _report_name(timedelta(hours=1)))
    six_d = _mk(tmp_path / "reports", _report_name(timedelta(days=6, hours=23)))
    eight_d = _mk(tmp_path / "reports", _report_name(timedelta(days=7, hours=1)))
    keep = _mk(tmp_path / "reports", ".gitkeep")
    removed = co.cleanup_by_age(tmp_path, max_age_days=185, reports_max_age_days=7)
    assert removed == 1
    assert new.exists() and six_d.exists() and keep.exists()
    assert not eight_d.exists()


def test_outcomes_keep_their_own_185_day_limit(tmp_path):
    day = lambda n: (datetime.now(timezone.utc) - timedelta(days=n)).strftime("%Y-%m-%d")
    old_ok = _mk(tmp_path / "outcomes", f"{day(100)}.jsonl")
    too_old = _mk(tmp_path / "outcomes", f"{day(200)}.jsonl")
    co.cleanup_by_age(tmp_path, max_age_days=185, reports_max_age_days=7)
    assert old_ok.exists() and not too_old.exists()


def test_dry_run_removes_nothing(tmp_path):
    old = _mk(tmp_path / "reports", _report_name(timedelta(days=30)))
    assert co.cleanup_by_age(tmp_path, 185, dry_run=True, reports_max_age_days=7) == 1
    assert old.exists()


def test_report_age_uses_time_in_file_name():
    ref = co.file_age_reference(Path("2026-03-04_12-30.md"))
    assert ref == datetime(2026, 3, 4, 12, 30, tzinfo=timezone.utc)
    assert co.file_age_reference(Path("2026-03-04.jsonl")).hour == 23


def test_workflows_use_12_hour_cadence_and_stage_reports():
    root = Path(__file__).resolve().parent.parent / ".github" / "workflows"
    run = (root / "run-bot.yml").read_text(encoding="utf-8")

    assert "(NOW / 900) % 48 ))" in run
    assert "(NOW / 900) % 24 ))" not in run
    assert "% 16" not in run

    assert "git add --sparse -f reports" in run

    assert "--reports-max-age-days 7" in (
        root / "cleanup-outcomes.yml"
    ).read_text(encoding="utf-8")


# ======================================================================
# from test_run_health_and_plan_history.py
# ======================================================================
"""Run-health counters, Brain plan audit trail, and the JSONL round trip."""



class FakeSdb:
    def __init__(self):
        self.meta = {}

    async def get_metadata(self, key):
        return self.meta.get(key)

    async def set_metadata(self, key, value, ttl=None):
        self.meta[key] = value
        return True


def _engine__run_health_and_plan_history():
    eng = object.__new__(BrainEngineV2)
    eng.sdb = FakeSdb()
    return eng


def test_plan_history_appends_and_caps():
    eng = _engine__run_health_and_plan_history()
    asyncio.run(eng._record_plan_event("PLAN-001", "pending", "gate passed"))
    asyncio.run(eng._record_plan_event("PLAN-001", "applied"))
    hist = json.loads(eng.sdb.meta[PLAN_HISTORY_KEY])
    assert [h["status"] for h in hist] == ["pending", "applied"]
    for i in range(150):
        asyncio.run(eng._record_plan_event(f"PLAN-{i}", "blocked"))
    assert len(json.loads(eng.sdb.meta[PLAN_HISTORY_KEY])) == 100


def test_plan_history_ignores_missing_id():
    eng = _engine__run_health_and_plan_history()
    asyncio.run(eng._record_plan_event(None, "applied"))
    assert PLAN_HISTORY_KEY not in eng.sdb.meta


def test_dedup_stats_reset():
    alerts.DEDUP_STATS["released"] = 3
    alerts.DEDUP_STATS["kept_repaint"] = 2
    alerts.reset_dedup_stats()
    assert all(v == 0 for v in alerts.DEDUP_STATS.values())


def test_telegram_counters(monkeypatch):
    q = object.__new__(alerts.TelegramQueue)
    q.sent_ok = 0
    q.sent_failed = 0
    results = iter([True, False])

    async def fake_impl(self, message):
        return next(results)

    monkeypatch.setattr(alerts.TelegramQueue, "_send_impl", fake_impl)
    assert asyncio.run(q.send("a")) is True
    assert asyncio.run(q.send("b")) is False
    assert (q.sent_ok, q.sent_failed) == (1, 1)


def test_jsonl_round_trip_filters_signal_only_rows(tmp_path, monkeypatch):
    for sub in ("outcomes", "shadow"):
        (tmp_path / sub).mkdir()
    monkeypatch.setattr(outcome_storage, "_OUTCOME_DIR", str(tmp_path))
    now = 1_900_000_000
    monkeypatch.setattr(outcome_storage.time, "time", lambda: now)
    # pre-resolution shadow row (no `win`) + resolved outcome row
    outcome_storage.append_outcome({"pair": "BTCUSD", "alert_key": "vwap_buy", "entry_ts": now}, shadow=True)
    outcome_storage.append_outcome_batch(
        [{"pair": "BTCUSD", "alert_key": "vwap_buy", "entry_ts": now, "win": True, "pct_move": 1.2}],
        shadow=True,
    )
    rows = outcome_storage.load_recent_outcomes(days=1, shadow=True)
    assert len(rows) == 1 and rows[0]["win"] is True
    assert rows[0]["schema_version"] == outcome_storage.OUTCOME_SCHEMA_VERSION


def test_redis_key_inventory_counts_prefixes_and_no_ttl():
    import macd_unified

    class Pipe:
        def __init__(self):
            self.keys = []

        def ttl(self, key):
            self.keys.append(key)

        async def execute(self):
            return [-1 if k.startswith("metadata:") else 300 for k in self.keys]

    class R:
        def __init__(self, keys):
            self._keys = keys

        async def scan_iter(self, match="*", count=500):
            for k in self._keys:
                yield k

        def pipeline(self):
            return Pipe()

    class S:
        pass

    s = S()
    s._redis = R(["pending:a:1", "pending:b:2", "metadata:x", "dedup:c"])
    inv = asyncio.run(macd_unified._redis_key_inventory(s))
    assert inv["pending"] == {"keys": 2, "no_ttl": 0}
    assert inv["metadata"] == {"keys": 1, "no_ttl": 1}
    assert inv["dedup"]["keys"] == 1


# ======================================================================
# from test_auto_rollback.py
# ======================================================================
"""#6: snapshot at apply + objective post-apply harm monitor + safe revert."""



HOUR = 3600
LOG = logging.getLogger("t")


class FakeSDB:
    degraded = False

    def __init__(self):
        self.meta = {}
        self.override = {}
        self.disabled = set()
        self.weights = None

    async def get_metadata(self, k):
        return self.meta.get(k)

    async def set_metadata(self, k, v, ttl=None):
        self.meta[k] = v
        return True

    async def get_config_override(self):
        return dict(self.override)

    async def write_config_override(self, f, v):
        self.override[f] = v
        return True

    async def remove_config_override_field(self, f):
        self.override.pop(f, None)
        return True

    async def get_disabled_alert_keys(self):
        return set(self.disabled)

    async def set_alert_key_disabled(self, ak, flag):
        (self.disabled.add if flag else self.disabled.discard)(ak)
        return True

    async def get_dynamic_weights(self):
        return None if self.weights is None else dict(self.weights)

    async def set_dynamic_weights(self, w, ttl=None, **kw):
        self.weights = dict(w)
        return True

    async def clear_dynamic_weights(self):
        self.weights = None
        return True


class FakeQ:
    def __init__(self):
        self.sent = []

    async def send(self, msg, priority="normal"):
        self.sent.append(msg)
        return True


def _engine__auto_rollback(sdb):
    cls = be.BrainEngineV2
    e = cls.__new__(cls)
    e.sdb = sdb
    return e


def _rows__auto_rollback(n, wr, t0, t1, seed, adx=True):
    rnd = random.Random(seed)
    return [{
        "win": rnd.random() < wr, "entry_ts": int(rnd.uniform(t0, t1)),
        "adx_val": rnd.uniform(10, 45) if adx else None,
        "net_pnl_pct": 0.5, "pct_move": 0.5,
    } for _ in range(n)]


def _seed_snapshot(sdb, applied_ago_h=72, **kw):
    now = int(time.time())
    snap = {
        "plan_id": "plan-1", "applied_at": now - int(applied_ago_h * HOUR), "status": "active",
        "overrides": kw.get("overrides", {}), "disabled": kw.get("disabled", {}),
        "weights": kw.get("weights"),
    }
    sdb.meta[be.APPLY_SNAPSHOT_KEY] = json_dumps([snap])
    return snap, now


def _status(sdb):
    return json_loads(sdb.meta[be.APPLY_SNAPSHOT_KEY])[0]["status"]


def test_apply_captures_prior_state():
    sdb = FakeSDB()
    sdb.override = {"CONFLUENCE_MIN_PCT": 70.0}
    sdb.weights = {"a": 1.0}
    sdb.disabled = {"old_key"}
    plan = {
        "plan_id": "p9", "generated_at": int(time.time()), "_action_gate_passed": True,
        "config_patch": [
            {"path": "CONFLUENCE_MIN_PCT", "suggested": 75.0},
            {"path": "CONFLUENCE_MIN_ABS_SCORE", "suggested": 20.0},
        ],
        "disable_alerts": ["bad_key"], "weight_adjustments": [],
    }
    sdb.meta["brain_pending_plan"] = json_dumps(plan)
    eng = _engine__auto_rollback(sdb)
    ok = asyncio.run(eng.apply_pending_plan(FakeQ(), LOG))
    assert ok
    snap = json_loads(sdb.meta[be.APPLY_SNAPSHOT_KEY])[0]
    assert snap["overrides"]["CONFLUENCE_MIN_PCT"] == {"prev": 70.0, "new": 75.0}
    assert snap["overrides"]["CONFLUENCE_MIN_ABS_SCORE"] == {"prev": None, "new": 20.0}
    assert snap["disabled"]["bad_key"] == {"prev": False, "new": True}
    assert snap["status"] == "active"


def test_harm_reverts_overrides_to_previous_values_and_absence():
    sdb = FakeSDB()
    sdb.override = {"CONFLUENCE_MIN_PCT": 75.0, "CONFLUENCE_MIN_ABS_SCORE": 20.0}
    sdb.disabled = {"bad_key"}
    snap, now = _seed_snapshot(
        sdb,
        overrides={"CONFLUENCE_MIN_PCT": {"prev": 70.0, "new": 75.0},
                   "CONFLUENCE_MIN_ABS_SCORE": {"prev": None, "new": 20.0}},
        disabled={"bad_key": {"prev": False, "new": True}},
    )
    at = snap["applied_at"]
    rows = _rows__auto_rollback(80, 0.62, at - 72 * HOUR, at - 1, 1) + _rows__auto_rollback(80, 0.35, at + 1, now, 2)
    ev = asyncio.run(_engine__auto_rollback(sdb).monitor_applied_plans(rows, LOG))
    assert [e["status"] for e in ev] == ["rolled_back"], ev
    assert sdb.override == {"CONFLUENCE_MIN_PCT": 70.0}       # restored / removed
    assert "bad_key" not in sdb.disabled                      # re-enabled
    assert _status(sdb) == "rolled_back"
    assert ev[0]["needs_restart"] is True
    hist = json_loads(sdb.meta[be.PLAN_HISTORY_KEY])
    assert hist[-1]["status"] == "rolled_back"


def test_harm_reverts_weights_to_none_or_prev():
    for prev in (None, {"a": 1.0, "b": 2.0}):
        sdb = FakeSDB()
        sdb.weights = {"a": 0.5, "b": 2.0}
        snap, now = _seed_snapshot(sdb, weights={"prev": prev, "new": {"a": 0.5, "b": 2.0}})
        at = snap["applied_at"]
        rows = _rows__auto_rollback(80, 0.62, at - 72 * HOUR, at - 1, 1) + _rows__auto_rollback(80, 0.35, at + 1, now, 2)
        asyncio.run(_engine__auto_rollback(sdb).monitor_applied_plans(rows, LOG))
        assert sdb.weights == prev


def test_no_revert_when_value_changed_since():
    sdb = FakeSDB()
    sdb.override = {"CONFLUENCE_MIN_PCT": 80.0}               # a LATER plan set 80, ours wrote 75
    snap, now = _seed_snapshot(
        sdb, overrides={"CONFLUENCE_MIN_PCT": {"prev": 70.0, "new": 75.0}})
    at = snap["applied_at"]
    rows = _rows__auto_rollback(80, 0.62, at - 72 * HOUR, at - 1, 1) + _rows__auto_rollback(80, 0.35, at + 1, now, 2)
    ev = asyncio.run(_engine__auto_rollback(sdb).monitor_applied_plans(rows, LOG))
    assert sdb.override == {"CONFLUENCE_MIN_PCT": 80.0}       # untouched
    assert "CONFLUENCE_MIN_PCT (changed since)" in ev[0]["result"]["skipped"]


def test_no_revert_on_noise_or_small_sample():
    sdb = FakeSDB()
    sdb.override = {"CONFLUENCE_MIN_PCT": 75.0}
    snap, now = _seed_snapshot(
        sdb, overrides={"CONFLUENCE_MIN_PCT": {"prev": 70.0, "new": 75.0}})
    at = snap["applied_at"]
    # same WR before/after
    rows = _rows__auto_rollback(80, 0.55, at - 72 * HOUR, at - 1, 1) + _rows__auto_rollback(80, 0.55, at + 1, now, 2)
    assert asyncio.run(_engine__auto_rollback(sdb).monitor_applied_plans(rows, LOG)) == []
    # big drop but only 12 post outcomes
    rows = _rows__auto_rollback(80, 0.62, at - 72 * HOUR, at - 1, 1) + _rows__auto_rollback(12, 0.20, at + 1, now, 2)
    assert asyncio.run(_engine__auto_rollback(sdb).monitor_applied_plans(rows, LOG)) == []
    assert sdb.override == {"CONFLUENCE_MIN_PCT": 75.0} and _status(sdb) == "active"


def test_too_early_is_not_judged():
    sdb = FakeSDB()
    sdb.override = {"CONFLUENCE_MIN_PCT": 75.0}
    snap, now = _seed_snapshot(
        sdb, applied_ago_h=3, overrides={"CONFLUENCE_MIN_PCT": {"prev": 70.0, "new": 75.0}})
    at = snap["applied_at"]
    rows = _rows__auto_rollback(80, 0.62, at - 3 * HOUR, at - 1, 1) + _rows__auto_rollback(80, 0.30, at + 1, now, 2)
    assert asyncio.run(_engine__auto_rollback(sdb).monitor_applied_plans(rows, LOG)) == []


def test_regime_attributed_drop_is_withheld():
    sdb = FakeSDB()
    sdb.override = {"CONFLUENCE_MIN_PCT": 75.0}
    snap, now = _seed_snapshot(
        sdb, overrides={"CONFLUENCE_MIN_PCT": {"prev": 70.0, "new": 75.0}})
    at = snap["applied_at"]
    rows = _rows__auto_rollback(80, 0.62, at - 72 * HOUR, at - 1, 1, adx=False) + _rows__auto_rollback(80, 0.35, at + 1, now, 2, adx=False)
    ev = asyncio.run(_engine__auto_rollback(sdb).monitor_applied_plans(rows, LOG))
    assert ev and ev[0]["status"] == "rollback_withheld"      # no ADX → cannot rule out regime
    assert sdb.override == {"CONFLUENCE_MIN_PCT": 75.0}


def test_daily_cap_and_clear_after_monitor_window():
    sdb = FakeSDB()
    now = int(time.time())
    old = {"plan_id": "old", "applied_at": now - 20 * 86400, "status": "active",
           "overrides": {}, "disabled": {}, "weights": None}
    done = {"plan_id": "done", "applied_at": now - 5 * 86400, "status": "rolled_back",
            "rolled_back_at": now - HOUR, "overrides": {}, "disabled": {}, "weights": None}
    sdb.meta[be.APPLY_SNAPSHOT_KEY] = json_dumps([done, old])
    rows = _rows__auto_rollback(80, 0.55, now - 40 * 86400, now - 20 * 86400 - 1, 1) + _rows__auto_rollback(80, 0.55, now - 20 * 86400 + 1, now, 2)
    ev = asyncio.run(_engine__auto_rollback(sdb).monitor_applied_plans(rows, LOG))
    assert [e["status"] for e in ev] == ["cleared"]


def test_disabled_by_flag(monkeypatch):
    monkeypatch.setattr(cfg, "BRAIN_AUTO_ROLLBACK_HURT", False, raising=False)
    sdb = FakeSDB()
    snap, now = _seed_snapshot(sdb, overrides={"X": {"prev": 1, "new": 2}})
    assert asyncio.run(_engine__auto_rollback(sdb).monitor_applied_plans([], LOG)) == []


def test_regime_shift_explains_drop_is_withheld():
    # Per-regime WR unchanged (trending 72%, ranging 38%), but post-apply trades are
    # almost all ranging -> overall WR drops purely from the mix.
    sdb = FakeSDB()
    sdb.override = {"CONFLUENCE_MIN_PCT": 75.0}
    snap, now = _seed_snapshot(
        sdb, overrides={"CONFLUENCE_MIN_PCT": {"prev": 70.0, "new": 75.0}})
    at = snap["applied_at"]

    def mk(n, wr, lo, hi, adx_lo, adx_hi, seed):
        rnd = random.Random(seed)
        return [{"win": rnd.random() < wr, "entry_ts": int(rnd.uniform(lo, hi)),
                 "adx_val": rnd.uniform(adx_lo, adx_hi), "net_pnl_pct": 0.5} for _ in range(n)]

    pre = mk(70, 0.72, at - 72 * HOUR, at - 1, 30, 45, 1) + mk(70, 0.38, at - 72 * HOUR, at - 1, 10, 22, 2)
    post = mk(10, 0.72, at + 1, now, 30, 45, 3) + mk(110, 0.38, at + 1, now, 10, 22, 4)
    ev = asyncio.run(_engine__auto_rollback(sdb).monitor_applied_plans(pre + post, LOG))
    assert ev and ev[0]["status"] == "rollback_withheld", ev
    assert ev[0]["reason"].startswith("drop attributed to regime")
    assert sdb.override == {"CONFLUENCE_MIN_PCT": 75.0}


# ======================================================================
# from test_strategy_state.py
# ======================================================================
"""#17: strategy broken vs current regime underrepresented vs regime-mix shift."""



DAY = 86400


def _rows__strategy_state(n, wr, adx_lo, adx_hi, days_ago_lo, days_ago_hi, seed):
    rnd = random.Random(seed)
    now = int(time.time())
    return [{
        "win": rnd.random() < wr,
        "adx_val": rnd.uniform(adx_lo, adx_hi),
        "entry_ts": now - int(rnd.uniform(days_ago_lo, days_ago_hi) * DAY),
    } for _ in range(n)]


def _hist(seed=1):
    # Older history: half trending (ADX 30-45, WR 72%), half ranging (ADX 10-22, WR 38%).
    return (_rows__strategy_state(150, 0.72, 30, 45, 20, 60, seed) + _rows__strategy_state(150, 0.38, 10, 22, 20, 60, seed + 1))


def test_insufficient_data():
    assert engine.classify_strategy_state(_hist()[:30])["state"] == "INSUFFICIENT_DATA"


def test_stable_when_no_drop():
    recent = _rows__strategy_state(60, 0.72, 30, 45, 1, 12, 5) + _rows__strategy_state(60, 0.38, 10, 22, 1, 12, 6)
    assert engine.classify_strategy_state(_hist() + recent)["state"] == "STABLE"


def test_strategy_degraded_same_regime_mix_lower_wr():
    # Same 50/50 mix as history, but WR collapsed in BOTH regimes.
    recent = _rows__strategy_state(60, 0.40, 30, 45, 1, 12, 5) + _rows__strategy_state(60, 0.25, 10, 22, 1, 12, 6)
    res = engine.classify_strategy_state(_hist() + recent)
    assert res["state"] == "STRATEGY_DEGRADED", res
    assert res["mix_adjusted_drop"] > 0.06


def test_regime_shift_when_mix_alone_explains_drop():
    # WR per regime unchanged, but recent trades are almost all in the weak ranging regime.
    recent = _rows__strategy_state(10, 0.72, 30, 45, 1, 12, 5) + _rows__strategy_state(110, 0.38, 10, 22, 1, 12, 6)
    res = engine.classify_strategy_state(_hist() + recent)
    assert res["state"] == "REGIME_SHIFT", res


def test_regime_underrepresented_when_history_has_little_of_it():
    # History is almost entirely ranging; recent trades are mostly high-ADX.
    hist = _rows__strategy_state(8, 0.72, 30, 45, 20, 60, 1) + _rows__strategy_state(292, 0.60, 10, 22, 20, 60, 2)
    recent = _rows__strategy_state(90, 0.35, 30, 45, 1, 12, 5) + _rows__strategy_state(30, 0.60, 10, 22, 1, 12, 6)
    res = engine.classify_strategy_state(hist + recent)
    assert res["state"] == "REGIME_UNDERREPRESENTED", res
    assert "trending" in res["underrepresented"]


def test_regime_unknown_without_adx():
    rows = _hist() + _rows__strategy_state(120, 0.30, 0, 0, 1, 12, 5)
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


# ======================================================================
# Redis pre-loads that overlap the candle fetch
# ======================================================================
def test_preloads_overlap_collect_reraise_and_fall_back_inline():
    import asyncio
    import time
    import macd_unified

    async def scenario():
        P = macd_unified._Preloads()
        ran = []

        async def slow(name):
            ran.append(name)
            await asyncio.sleep(0.2)
            return name

        async def boom():
            raise RuntimeError("redis down")

        t0 = time.monotonic()
        P.start("a", lambda: slow("a"))
        P.start("b", lambda: slow("b"))
        P.start("bad", boom)
        # three 0.2s reads started together finish in about 0.2s, not 0.6s
        assert await P.take("a", lambda: slow("inline-a")) == "a"
        assert await P.take("b", lambda: slow("inline-b")) == "b"
        assert time.monotonic() - t0 < 0.45
        # a failed read re-raises where it is collected (the caller's own try/except)
        try:
            await P.take("bad", lambda: slow("x"))
            raise AssertionError("expected RuntimeError")
        except RuntimeError as e:
            assert "redis down" in str(e)
        # never started -> runs inline, so behaviour without a pre-load is unchanged
        assert await P.take("missing", lambda: slow("inline")) == "inline"
        # leftovers are cancelled, not leaked
        P.start("late", lambda: slow("late"))
        P.cancel_all()
        assert await P.take("late", lambda: slow("inline-late")) == "inline-late"

    asyncio.run(scenario())
