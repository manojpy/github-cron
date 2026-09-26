#!/usr/bin/env python3
"""brain_enhanced.py — Prescriptive Brain (Roadmap Phases 1.5-6)"""

from __future__ import annotations
import asyncio
import json
import logging
import random
import time
import unicodedata
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple
import os
from pathlib import Path
from datetime import datetime, timedelta, timezone

from brain_audit import (
    DataCoverage, HealthStatus, RecommendationTier, get_audit, reset_audit,
    ACTION_GATE_MIN_ROWS,
)

from archive_reader import load_archived_outcomes
from bot_config import cfg, CONFLUENCE_WEIGHTS, CONFIG_OVERRIDE_ALLOWED_FIELDS, json_dumps, json_loads

from state import RedisKeyPrefix, RedisStateStore
from brain import BrainEngine as BaseBrainEngine, _extract_p_value_for_fdr
import threshold_engine as engine

from threshold_engine import (
    optimize_vote_weights, conditional_performance,
    interaction_miner, simulate_config_change, regime_profile_optimizer,
    hash_config_state, learned_actionability, compare_config_versions,
)
from repair_ledger import (
    record_repair_issued, mark_plan_applied,
    evaluate_pending_repairs, repair_success_rates, ledger_stats,
    load_ledger_entries,
)

from alerts import escape_markdown_v2

_PHASE_MIN_SAMPLES = {
    "weight_optimizer": 100,
    "parameter_autopsy": 30,
    "conditional_gating": 15,
    "vote_interactions": 20,
    "counterfactual": 10,
    "regime_profiles": 25,
    "config_regression": 20,
}

def _report_section_failed(failed: List[str], name: str, exc: Exception) -> None:
    """A report section raised. Never let that vanish: record it in the audit
    layer (logs at WARNING with the exception type) and queue the section
    name so the report itself says it is unavailable."""
    try:
        get_audit().record_analysis_exception(f"report:{name}", exc)
    except Exception:
        logging.getLogger("macd_bot").warning(
            f"Brain report: {name} section failed: {type(exc).__name__}: {exc}"
        )
    failed.append(name)

# ══════════════════════════════════════════════════════════════════════
#  BRAIN REPORT v2 — layered 16-section layout
#  Sections 01-05 = the "human Brain" (read these first).
#  Sections 06-16 = the evidence behind it.
#  Every number comes from data the Brain already computes; nothing is
#  invented. build_profit_action_plan() is kept as the fallback.
# ══════════════════════════════════════════════════════════════════════

_RULE = "━" * 30
_IST = timezone(timedelta(hours=5, minutes=30))
_LADDER = ["⚪", "🟡", "🟠", "🔵", "🟢"]          # Observation → Validated
_LADDER_NAMES = ["Observation", "Early evidence", "Meaningful evidence",
                 "Strong evidence", "Validated evidence"]
_MSG_LIMIT = 3800                                   # rendered chars per Telegram message
_HUMAN_SECTIONS = 5                                 # sections 1-5 are packed on their own

_TOKEN_NAMES = {
    "choch": "CHoCH", "ppo": "PPO", "vwap": "VWAP", "rsi": "RSI", "tk": "TK",
    "rma": "RMA", "hist": "Hist", "ppohist": "PPO Hist", "adx": "ADX",
    "macd": "MACD", "fib": "Fib", "up": "UP", "down": "DOWN", "buy": "BUY",
    "sell": "SELL", "s1": "S1", "s2": "S2", "s3": "S3", "r1": "R1", "r2": "R2",
    "r3": "R3", "p": "P", "ob": "OB", "fvg": "FVG", "bos": "BOS", "atr": "ATR",
    "ema": "EMA", "sma": "SMA", "bb": "BB",
}
_DIRECTION_TOKENS = {"buy", "sell", "up", "down", "cross", "s1", "s2", "s3", "r1", "r2", "r3", "p"}


def _pretty_alert(key: str) -> str:
    """strong_reversal_buy -> 'Strong Reversal BUY'."""
    toks = [t for t in str(key).split("_") if t and t.lower() != "cross"]
    return " ".join(_TOKEN_NAMES.get(t.lower(), t.capitalize()) for t in toks) or _pretty_alert(key)

def _alert_family(key: str) -> str:
    """pivot_down_S1 -> 'Pivot'; dynamic_flow_cross_sell -> 'Dynamic Flow'."""
    toks = [t for t in str(key).split("_") if t and t.lower() not in _DIRECTION_TOKENS]
    return " ".join(_TOKEN_NAMES.get(t.lower(), t.capitalize()) for t in toks) or _pretty_alert(key)

class _Piece(str):
    """A rendered Telegram fragment that remembers its plain source text and
    kind ('p' prose, 'c' code, 'h' section header), so the same report can
    also be written out as Markdown."""
    kind: str
    raw: str

    def __new__(cls, rendered: str, kind: str, raw: str) -> "_Piece":
        obj = super().__new__(cls, rendered)
        obj.kind = kind
        obj.raw = raw
        return obj


def _p(text: str) -> "_Piece":
    """Prose piece, MarkdownV2-escaped."""
    return _Piece(escape_markdown_v2(text), "p", text)


def _c(text: str) -> "_Piece":
    """Code-block piece. Inside ``` only ` and \\ need escaping."""
    return _Piece("```\n" + text.replace("\\", "\\\\").replace("`", "\\`") + "\n```", "c", text)

def _c_split(lines: List[str], limit: int = 3000) -> List[_Piece]:
    """Fenced blocks of at most `limit` chars, split on line boundaries."""
    out: List[_Piece] = []          # was: List[str]
    cur: List[str] = []
    size = 0
    for ln in lines:
        if cur and size + len(ln) + 1 > limit:
            out.append(_c("\n".join(cur)))
            cur, size = [], 0
        cur.append(ln)
        size += len(ln) + 1
    if cur:
        out.append(_c("\n".join(cur)))
    return out

def _hdr(num: int, title: str) -> "_Piece":
    return _Piece(escape_markdown_v2(f"{_RULE}\n{num:02d} │ {title}\n{_RULE}"), "h", f"{num:02d} │ {title}")

def _evidence_rank(n: int, days: Optional[float], validated: bool = False) -> int:
    """0 ⚪ Observation … 4 🟢 Validated. Capped by history span so a short
    history can never look 'strong' however many trades it holds."""
    if validated:
        return 4
    r = 0 if n < 15 else 1 if n < 30 else 2 if n < 60 else 3
    if days is not None:
        r = min(r, 1 if days < 14 else 2 if days < 30 else 3)
    return r

def _wrap_names(names: List[str], width: int = 34, indent: str = "   ") -> List[str]:
    """Bullet-free wrapped list: 'A · B · C' broken into short lines."""
    lines: List[str] = []
    cur = ""
    for nm in names:
        add = nm if not cur else f" · {nm}"
        if cur and len(cur) + len(add) > width:
            lines.append(indent + cur)
            cur = nm
        else:
            cur += add
    if cur:
        lines.append(indent + cur)
    return lines

def _collect_facts(recs: Dict[str, Any], cfg) -> Dict[str, Any]:
    """Everything the sections need, computed once."""
    audit = get_audit()
    rows = recs.get("_real_rows", []) or []
    ai = recs.get("ai_metrics", {}) or {}
    n = len(rows)
    wins = sum(1 for r in rows if r["win"])
    F: Dict[str, Any] = {
        "audit": audit, "rows": rows, "ai": ai, "n": n,
        "wr": wins / n if n else 0.0,
        "net_ev": ai.get("net_ev", 0.0) or 0.0,
        "gate": ai.get("action_gate", {}) or {},
        "cfg_patch": recs.get("config_patch", []) or [],
        "shadow_rows": recs.get("_shadow_rows", []) or [],
        "archive_stats": recs.get("_archive_stats", {}) or {},
        "conf": audit.statistical_confidence_label(),
        "recon": audit.reconciliation_snapshot(),
        "coverage": audit.history_coverage(),
    }
    F["gate_ok"] = bool(F["gate"].get("actionable"))
    span = audit.history_span()
    F["days"] = span[0] if span else None
    F["req_days"] = span[1] if span else int(getattr(cfg, "BRAIN_LONG_WINDOW_DAYS", 180))
    F["low_trust"] = F["conf"] in ("VERY LOW", "LOW")

    # Portfolio-level EV object (drawdown, P(EV>0)) — cached by the engine.
    ev_obj = None
    try:
        ev_obj = engine.ev_first_objective(rows, min_sample=10) if rows else None
    except Exception:
        ev_obj = None
    F["ev_obj"] = ev_obj if ev_obj and ev_obj.get("valid") else None
    F["p_ev"] = (F["ev_obj"] or {}).get("p_ev_positive")
    F["dd"] = (F["ev_obj"] or {}).get("max_drawdown_pct")
    F["dd_budget"] = float(getattr(cfg, "KILL_SWITCH_MAX_DRAWDOWN_PCT", 3.0))

    # Per-alert table (EV only where the sample is big enough to compute one).
    by_alert: Dict[str, list] = defaultdict(list)
    for r in rows:
        by_alert[r["alert_key"]].append(r)
    alerts: List[Dict[str, Any]] = []
    for ak, arows in by_alert.items():
        cnt = len(arows)
        awr = sum(1 for r in arows if r["win"]) / cnt
        ev = None
        if cnt >= 10:
            try:
                e = engine.ev_first_objective(arows, min_sample=10)
                ev = e if e and e.get("valid") else None
            except Exception:
                ev = None
        net = ev.get("net_ev", 0.0) if ev else None
        alerts.append({
            "key": ak, "name": _pretty_alert(ak), "family": _alert_family(ak),
            "n": cnt, "wr": awr, "ev": net, "p": ev.get("p_ev_positive", 0.0) if ev else None,
            "damage": (net * cnt) if net is not None else 0.0,
        })
    for a in alerts:
        validated = (
            F["gate_ok"] and a["ev"] is not None and a["ev"] > 0
            and (a["p"] or 0) >= 0.85 and a["n"] >= 60 and (F["days"] or 0) >= 30
        )
        a["rank"] = _evidence_rank(a["n"], F["days"], validated)
    F["alerts"] = alerts
    F["weak"] = sorted([a for a in alerts if a["ev"] is not None and a["ev"] < 0],
                       key=lambda a: a["damage"])
    F["good"] = sorted([a for a in alerts if a["ev"] is not None and a["ev"] > 0],
                       key=lambda a: -a["ev"])
    F["thin"] = [a for a in alerts if a["ev"] is None]

    # Direction, coin and session splits.
    try:
        F["dir"] = engine.direction_split(rows)
    except Exception:
        F["dir"] = (None, 0, None, 0)
    try:
        F["pairs"] = engine.per_pair_breakdown(rows, min_sample=5)          # worst-first
    except Exception:
        F["pairs"] = []
    try:
        F["sessions"] = engine.session_breakdown(rows, min_sample=1)        # worst-first
    except Exception:
        F["sessions"] = []

    # Entry-bar simulation.
    F["target_wr"] = getattr(cfg, "MIN_WIN_RATE", 0.55)
    try:
        F["rec_thr"] = engine.recommend_threshold(
            rows, target_winrate=F["target_wr"],
            min_sample=getattr(cfg, "MIN_WIN_RATE_SAMPLE", 20),
        )
    except Exception:
        F["rec_thr"] = {"valid": False}
    F["rec_thr_ok"] = bool(
        F["rec_thr"].get("valid") and F["rec_thr"].get("recommended")
        and F["rec_thr"]["recommended"] > cfg.CONFLUENCE_MIN_ABS_SCORE
    )
    F["thr_tier"] = audit.max_recommendation_tier("threshold_recommendation")
    F["anatomy_text"], F["anatomy"] = _outcome_anatomy(rows, cfg)
    F["shadow_on"] = bool(getattr(cfg, "BRAIN_SHADOW_MODE", False))
    return F

def _outcome_anatomy(rows: List[Dict[str, Any]], cfg) -> Tuple[Optional[str], Dict[str, Any]]:
    """Plain-English 'how a win is judged' text plus how trades really ended.
    Uses only fields already on every outcome row (outcome_reason, mfe, mae;
    stored as fractions, 0.02 = 2%). Returns (text, facts)."""
    risk = float(getattr(cfg, "OUTCOME_MAE_LOSS_PCT", 0.0) or 0.0)
    rr = float(getattr(cfg, "OUTCOME_RR_TARGET", 0.0) or 0.0)
    target = risk * rr
    hours = float(getattr(cfg, "OUTCOME_LOOKAHEAD_CANDLES", 0) or 0) * 15.0 / 60.0   # 15m candles
    if not rows or risk <= 0 or target <= 0 or hours <= 0:
        return None, {}
    buckets = {"target_hit": "target", "target_hit_ever": "target",
               "stop_hit": "stop", "stop_hit_ever": "stop",
               "both_hit": "both", "ambiguous_same_candle": "both", "no_hit": "neither"}
    counts = {"target": 0, "stop": 0, "both": 0, "neither": 0}
    for r in rows:
        reason = r.get("outcome_reason")
        b = buckets.get(reason) if isinstance(reason, str) else None
        if b:
            counts[b] += 1
    known = sum(counts.values())

    def _med(vals: List[float]) -> Optional[float]:
        if not vals:
            return None
        v = sorted(vals)
        m = len(v) // 2
        return v[m] if len(v) % 2 else (v[m - 1] + v[m]) / 2.0

    med_mfe = _med([r["mfe"] * 100.0 for r in rows if r.get("mfe") is not None])
    med_mae = _med([r["mae"] * 100.0 for r in rows if r.get("mae") is not None])
    lines = [
        f"A trade counts as a WIN only if price moves +{target:.1f}% your way "
        f"before it moves -{risk:.1f}% against you, within {hours:g} hours.",
        f"Break-even needs roughly {1.0 / (1.0 + rr):.0%} wins (before fees).",
    ]
    if known:
        lines.append(f"How {known} trades actually ended:")
        lines.append(f"  • {counts['target'] / known:.0%} reached the +{target:.1f}% target")
        lines.append(f"  • {counts['stop'] / known:.0%} hit the -{risk:.1f}% stop first")
        lines.append(f"  • {counts['neither'] / known:.0%} did neither in {hours:g}h")
        if counts["both"]:
            lines.append(f"  • {counts['both'] / known:.0%} touched both (order unclear)")
    if med_mfe is not None:
        lines.append(f"Typical best move for you: {med_mfe:.1f}% (target {target:.1f}%)")
    if med_mae is not None:
        lines.append(f"Typical worst move against you: {med_mae:.1f}% (stop {risk:.1f}%)")
    strict = med_mfe is not None and 0 < med_mfe < 0.5 * target
    if strict and med_mfe is not None:
        lines.append(
            f"⚠️ The target is {target / med_mfe:.0f}x your typical best move, so this rule "
            f"is hard for ANY 15-minute signal to meet. It lowers every alert's win rate. "
            f"Compare alerts with each other, not with the {_goal_wr(cfg):.0%} goal."
        )
    return "\n".join(lines), {"target": target, "risk": risk, "hours": hours, "rr": rr,
                              "med_mfe": med_mfe, "med_mae": med_mae, "strict": strict,
                              "counts": counts, "known": known}


def _goal_wr(cfg) -> float:
    return float(getattr(cfg, "MIN_WIN_RATE", 0.55))


# ── status helpers ────────────────────────────────────────────────────

def _profit_status(net_ev: float, n: int) -> str:
    if n == 0:
        return "⚪ NO DATA"
    return "🔴 POOR" if net_ev <= -0.05 else "🟡 FLAT" if net_ev < 0.05 else "🟢 POSITIVE"

def _data_status(F: Dict[str, Any]) -> str:
    cov_obj = F.get("coverage")
    cov = getattr(cov_obj, "coverage", None) if cov_obj else None
    if not isinstance(cov, DataCoverage):
        return "⚪ UNKNOWN"
    return {
        DataCoverage.FULL: "🟢 GOOD",
        DataCoverage.PARTIAL: "🟡 PARTIAL HISTORY",
        DataCoverage.SEVERELY_LIMITED: "🟠 LIMITED HISTORY",
        DataCoverage.CRITICAL: "🔴 INSUFFICIENT HISTORY",
    }.get(cov, "⚪ UNKNOWN")

def _recording_status(F: Dict[str, Any]) -> str:
    r = F["recon"]
    if r is None:
        return "⚪ UNKNOWN"
    label = "HEALTHY" if r.status == HealthStatus.OK else r.status.value
    return f"{r.status.icon} {label}"


def _conf_status(conf: str) -> str:
    return {"VERY LOW": "🔴 LOW", "LOW": "🔴 LOW", "MODERATE": "🟡 MODERATE", "HIGH": "🟢 HIGH"}[conf]


def _icon(ok: Optional[bool]) -> str:
    return "🟢" if ok else "🔴"


def _overall(F: Dict[str, Any]) -> str:
    prof = _profit_status(F["net_ev"], F["n"])
    rec = _recording_status(F)
    if prof.startswith("🔴") or rec.startswith("🔴"):
        return "🔴 NEEDS ATTENTION"
    others = [prof, _data_status(F), rec, _conf_status(F["conf"])]
    if any(not o.startswith("🟢") for o in others) or not F["gate_ok"]:
        return "🟡 MONITOR"
    return "🟢 HEALTHY"


def _fmt_days(d: Optional[float]) -> str:
    return "n/a" if d is None else f"{d:.1f} days"

def _fmt_span(d: Optional[float]) -> str:
    """Adjective form: '2.5-day' (for 'the current 2.5-day sample')."""
    return "very short" if d is None else f"{d:.1f}-day"


_WIDE_EMOJI_RANGES = (
    (0x1F300, 0x1FAFF), (0x2600, 0x27BF), (0x2B00, 0x2BFF), (0x1F000, 0x1F02F),
)

def _vwidth(s: str) -> int:
    """Visual width of `s` in monospace columns. Plain str length
    undercounts emoji and other wide glyphs — they render ~2 columns
    wide in virtually every renderer that shows this report (Telegram,
    Gemini, a terminal, Notepad's default monospace font) — so padding
    computed from len() alone silently drifts by one column per emoji.
    This is the actual reason earlier alignment attempts looked fine in
    the source but scattered on screen."""
    w = 0
    for ch in s:
        cp = ord(ch)
        if unicodedata.east_asian_width(ch) in ("W", "F"):
            w += 2
        elif any(lo <= cp <= hi for lo, hi in _WIDE_EMOJI_RANGES):
            w += 2
        elif unicodedata.combining(ch):
            pass
        else:
            w += 1
    return w

def _ljust(s: str, width: int) -> str:
    return s + " " * max(0, width - _vwidth(s))

def _rjust(s: str, width: int) -> str:
    return " " * max(0, width - _vwidth(s)) + s

def _table(rows: Sequence[Tuple[str, ...]], aligns: str, gap: int = 2) -> List[str]:
    """Render `rows` (equal-length tuples of cell text, header included if
    any) as monospace lines whose columns are all aligned to the widest
    VISUAL width in that column across every row — so a dot/emoji in row 3
    can't push row 3's own values out of line with rows 1, 2, 4... `aligns`
    is one 'l' or 'r' per column. The last column is never padded (no
    point padding text nothing follows)."""
    ncol = len(aligns)
    widths = [max((_vwidth(r[i]) for r in rows), default=0) for i in range(ncol)]
    out = []

    for r in rows:
        cells = []
        for i, cell in enumerate(r):
            if i == ncol - 1 and aligns[i] != "r":
                cells.append(cell)
            else:
                cells.append((_ljust if aligns[i] == "l" else _rjust)(cell, widths[i]))
        out.append((" " * gap).join(cells))
    return out

def _split_leading_emoji(text: str) -> Tuple[str, str]:
    """Split off a leading emoji (plus any variation selector / ZWJ
    continuation) from `text`. Returns (emoji, rest). No leading emoji
    → ('', text).

    '🟡 FLAT'    → ('🟡', 'FLAT')
    '⚠️ WARNING'  → ('⚠️', 'WARNING')
    'FLAT'       → ('',  'FLAT')

    Uses the same ranges _vwidth() uses, so anything _vwidth counts as
    a wide glyph is treated as an emoji here too — the two stay in sync.
    """
    i, n = 0, len(text)
    while i < n:
        cp = ord(text[i])
        if any(lo <= cp <= hi for lo, hi in _WIDE_EMOJI_RANGES) or cp in (0xFE0F, 0x200D):
            i += 1
            continue
        break
    if i == 0:
        return "", text
    return text[:i], text[i:].lstrip()

def _kv_table(pairs: Sequence[Tuple[str, str]]) -> List[str]:
    """Convenience for the common 'Label: Value' block.

    Emoji in a value is lifted to column 0 so every row reads
        🟡 Observed profitability:   FLAT
        🔴 Data quality:             LOW
    with one aligned value column, instead of the emoji drifting with
    the value and the labels looking ragged. Rows whose values carry no
    emoji at all (e.g. 'Current: 18') are left in the plain
    'Label: Value' form — no phantom emoji column.
    """
    rows = []
    for label, value in pairs:
        emoji, rest = _split_leading_emoji(value)
        if emoji:
            rows.append((f"{emoji} {label}:", rest))
        else:
            rows.append((f"{label}:", value))
    return _table(rows, "ll")

def _active_blockers(F: Dict[str, Any]) -> List[str]:
    g = F["gate"]
    out: List[str] = []
    if F["days"] is not None and F["days"] < 14:
        out.append(f"Only {F['days']:.1f} days history")
    if g and g.get("data_quality") is False:
        out.append("Fewer than 100 trades")
    if g and g.get("oos_prediction") is False:
        out.append("OOS validation unavailable")
    if g and g.get("profitability") is False:
        out.append("Net EV not confidently positive")
    if g and g.get("stability") is False:
        out.append("CUSUM drift active")
    if g and g.get("risk") is False:
        out.append("Drawdown outside current budget")
    if g and g.get("execution") is False:
        out.append("Fee/slippage assumptions missing")
    return out

# ── the sections ──────────────────────────────────────────────────────

def _sec_summary(F: Dict[str, Any], cfg) -> List[_Piece]:
    n, wr, net_ev, days = F["n"], F["wr"], F["net_ev"], F["days"]
    validated_mode = F["gate_ok"] and net_ev > 0 and F["conf"] in ("MODERATE", "HIGH")
    if validated_mode:
        return _sec_verdict(F, cfg)
    prof = _profit_status(net_ev, n)
    out = [_hdr(1, 'EXECUTIVE SUMMARY — "WHAT DO I NEED TO KNOW?"')]
    out.append(_p(f"OVERALL SYSTEM STATUS\n{_overall(F)}"))
    out.append(_c("\n".join(_kv_table([
        ("Observed profitability", prof),
        ("Data quality", _data_status(F)),
        ("Outcome recording", _recording_status(F)),
        ("Statistical confidence", _conf_status(F['conf'])),
        ("Brain action gate", '🟢 PASSED' if F['gate_ok'] else '🔴 BLOCKED'),
    ]))))
    out.append(_c("\n".join(
        [f"📊 {n} resolved trades"] + _table([
            ("📈 Win Rate:", f"{wr:.0%}"),
            ("💰 Net EV/trade:", f"{net_ev:+.2f}%"),
            ("📅 History available:", _fmt_days(days)),
            ("📅 History requested:", f"{F['req_days']} days"),
        ], "lr")
    )))
    losing = net_ev <= -0.05
    if losing and F["low_trust"]:
        text = (
            "The system is currently producing poor results in the available sample.\n\n"
            f"However, only {_fmt_days(days)} of history are available, so the Brain does NOT yet "
            "have enough evidence to say whether this is a persistent strategy problem.\n\n"
            "➡️ Current priority: DIAGNOSE + COLLECT DATA\n"
            "➡️ Not yet: AGGRESSIVE OPTIMISATION"
        )
    elif losing:
        text = (
            "The system is producing poor results and the history is long enough for this to be "
            "taken seriously.\n\n"
            "➡️ Current priority: FIX THE WEAKEST PARTS (see sections 04 and 12)\n"
            + ("➡️ The action gate is still blocked: changes need evidence first."
               if not F["gate_ok"] else "➡️ The action gate is open for validated changes.")
        )
    elif F["low_trust"]:
        text = (
            "Results look positive so far, but the history is too short to trust them.\n\n"
            "➡️ Current priority: COLLECT DATA\n➡️ Not yet: INCREASE RISK OR WEIGHTS"
        )
    else:
        text = (
            "Results are around break-even to positive with usable history.\n\n"
            "➡️ Current priority: VALIDATE, THEN OPTIMISE CAREFULLY"
        )
    out.append(_p("🧠 BRAIN'S SIMPLE INTERPRETATION\n\n" + text))
    return out


def _sec_verdict(F: Dict[str, Any], cfg) -> List[_Piece]:
    """End-state Section 1, shown only when the action gate passes."""
    g = F["gate"]
    out = [_hdr(1, "🧠 BRAIN VERDICT")]
    out.append(_c("\n".join(_kv_table([
        ("System health", _recording_status(F).split(' ')[0]),
        ("Profitability", _profit_status(F['net_ev'], F['n']).split(' ')[0]),
        ("Evidence quality", _conf_status(F['conf']).split(' ')[0]),
        ("OOS validation", _icon(g.get('oos_prediction'))),
        ("Drift", _icon(g.get('stability'))),
        ("Drawdown", _icon(g.get('risk'))),
    ]))))
    validated = [a for a in F["alerts"] if a["rank"] == 4]
    validated.sort(key=lambda a: -(a["ev"] or 0))
    edge = [f"• {a['name']}  (EV {a['ev']:+.2f}%, WR {a['wr']:.0%}, n={a['n']})" for a in validated[:3]]
    good_sess = [s for s in F["sessions"] if s[2] >= 30]
    good_pair = [p for p in F["pairs"] if p[2] >= 30]
    if good_sess:
        s = good_sess[-1]
        edge.append(f"• Session: {s[0].upper()} ({s[1]:.0%} WR, n={s[2]})")
    if good_pair:
        p_ = good_pair[-1]
        edge.append(f"• Coin: {p_[0]} ({p_[1]:.0%} WR, n={p_[2]})")
    pev = f"P(EV > 0): {F['p_ev']:.0%}" if F["p_ev"] is not None else "P(EV > 0): n/a"
    out.append(_p(
        "🎯 CURRENTLY VALIDATED EDGE\n"
        + ("\n".join(edge) if edge else "No single alert has reached 'validated' evidence yet.")
        + f"\n\nEvidence: {F['n']} trades | {_fmt_days(F['days'])} | Net EV {F['net_ev']:+.2f}%\n{pev}"
    ))
    weak_ready = [a for a in F["weak"] if a["n"] >= 30]
    worst = (weak_ready or F["weak"] or [None])[0]
    if worst:
        out.append(_p(
            f"🎯 CURRENT WEAKNESS\n{worst['name']}\n\n"
            f"Evidence: {worst['n']} trades | EV {worst['ev']:+.2f}% | WR {worst['wr']:.0%}"
        ))
    approved = [p for p in F["cfg_patch"] if not p.get("_blocked_by_action_gate")]
    if approved:
        first = approved[0]
        action = f"{first.get('path')}: {first.get('current')} → {first.get('suggested')}"
    else:
        action = "No parameter change is supported by the evidence. Keep current settings."
    out.append(_p(
        f"🤖 RECOMMENDED ACTION\n{action}\n\n"
        f"Confidence: {F['conf']}\nAction Gate: APPROVED"
    ))
    return out

def _sec_do_now(F: Dict[str, Any], cfg) -> List[_Piece]:
    n, wr = F["n"], F["wr"]
    out = [_hdr(2, "🚦 WHAT SHOULD I DO NOW?")]
    do_now: List[str] = []
    recon_ok = F["recon"] is not None and F["recon"].status == HealthStatus.OK
    if wr < 0.10 or not recon_ok:
        do_now.append("Verify alert → trade → outcome recording is correct.")
    if F["coverage"] is None or F["coverage"].coverage != DataCoverage.FULL:
        do_now.append("Continue collecting clean outcome data.")
    do_now.append("Keep Shadow Mode enabled." if F["shadow_on"]
                  else "Turn Shadow Mode ON (BRAIN_SHADOW_MODE=true) so rejected trades can be judged.")
    if wr < 0.10 and n:
        do_now.append(f"Investigate the very low {wr:.0%} win rate (see section 03).")
    if F["gate"].get("stability") is False or F["gate"].get("risk") is False:
        do_now.append("Monitor active drift/drawdown warnings.")
    out.append(_p("🔴 DO NOW\n\n" + "\n".join(f"• {x}" for x in do_now)))

    watch: List[str] = []
    seen = set()
    for a in F["weak"]:
        if a["family"] not in seen:
            seen.add(a["family"])
            watch.append(a["family"])
        if len(watch) >= 5:
            break
    b_wr, b_n, s_wr, s_n = F["dir"]
    if b_wr is not None and s_wr is not None and b_n >= 5 and s_n >= 5 and abs(b_wr - s_wr) >= 0.15:
        watch.append("BUY vs SELL imbalance")
    if len(F["sessions"]) >= 2 and F["sessions"][-1][1] - F["sessions"][0][1] >= 0.05:
        watch.append(f"{F['sessions'][0][0].title()} vs {F['sessions'][-1][0].title()} session")
    out.append(_p("🟡 WATCH / INVESTIGATE\n\n" + ("\n".join(f"• {x}" for x in watch) if watch
                                                 else "• Nothing flagged right now.")))

    if F["gate_ok"] and not F["low_trust"]:
        approved = [p for p in F["cfg_patch"] if not p.get("_blocked_by_action_gate")]
        out.append(_p("✅ APPROVED CHANGES\n\n" + (
            "\n".join(f"• {p.get('path')}: {p.get('current')} → {p.get('suggested')}" for p in approved[:6])
            if approved else "• No change is supported by the evidence right now.")))
    else:
        dont = []
        if F["low_trust"] or not F["gate_ok"]:
            dont.append("Do not disable alerts solely from this report.")
            dont.append("Do not change confluence weights.")
            dont.append("Do not change adaptive thresholds.")
        if F["rec_thr_ok"]:
            dont.append("Do not apply the simulated entry-bar change.")
        if F["low_trust"]:
            dont.append(f"Do not optimise from the current {_fmt_span(F['days'])} sample.")
        out.append(_p("🚫 DO NOT CHANGE YET\n\n" + "\n".join(f"• {x}" for x in dont)))
    return out

def _sec_profit(F: Dict[str, Any], cfg) -> List[_Piece]:
    n, wr, net_ev, days = F["n"], F["wr"], F["net_ev"], F["days"]
    an = F["anatomy"] or {}
    be = 1.0 / (1.0 + an["rr"]) if an.get("rr") else None
    out = [_hdr(3, '📊 PROFITABILITY — "ARE WE ACTUALLY MAKING MONEY?"')]
    wr_verdict = ("🔴 Very poor" if be and wr < be * 0.5 else "🔴 Below break-even" if be and wr < be
                  else "🟢 At/above break-even" if be else "⚪")
    dd, bud = F["dd"], F["dd_budget"]

    # Emoji lifted out of VERDICT and placed at column 0; column order is
    # now emoji | metric | result (right-aligned) | verdict text.
    entries = [
        ("Trades", str(n),
         "🟢 Good sample" if n >= 100 else "🟡 Small sample" if n >= 30 else "🔴 Tiny sample"),
        ("Win Rate", f"{wr:.0%}", wr_verdict),
        ("Net EV", f"{net_ev:+.2f}%",
         "🟢 Positive" if net_ev > 0.05 else "🟡 Flat" if net_ev > -0.05 else "🔴 Negative"),
        ("History", "n/a" if days is None else f"{days:.1f}d",
         "🟢 Long enough" if (days or 0) >= 30 else "🟡 Short" if (days or 0) >= 14 else "🔴 Too short"),
        ("After costs?", "YES" if F["gate"].get("execution", True) else "NO",
         "🟢 Costs accounted" if F["gate"].get("execution", True) else "🔴 Missing"),
        ("Drawdown", "n/a" if dd is None else f"{dd:.1f}%",
         "⚪ unknown" if dd is None else f"{'🟢 Within' if dd <= bud else '🔴 Over'} {bud:.1f}% budget"),
        ("CUSUM drift", "ACTIVE" if F["gate"].get("stability") is False else "NONE",
         "🔴 Drifting" if F["gate"].get("stability") is False else "🟢 Stable"),
    ]
    rows = [("", "METRIC", "RESULT", "VERDICT")]
    for metric, value, verdict in entries:
        emoji, rest = _split_leading_emoji(verdict)
        rows.append((emoji or "⚪", metric, value, rest))
    out.extend(_c_split(_table(rows, "llrl")))

    if net_ev < 0:
        meaning = ("The observed trades are losing money after costs.\n\n"
                   + ("BUT the Brain does not yet know whether this will persist across different "
                      "market conditions." if F["low_trust"]
                      else "The history is long enough that this should be treated as real."))
    else:
        meaning = "The observed trades are not losing money after costs in this sample."
    recon_ok = F["recon"] is not None and F["recon"].status == HealthStatus.OK
    out.append(_p(f"🧠 What this means:\n\n{meaning}"))
    out.append(_c("\n".join(_kv_table([
        ("Confidence in the RESULT", _conf_status(F['conf'])),
        ("Confidence in the DATA PIPELINE", '🟢 GOOD' if recon_ok else '🟡 CHECK SECTION 14'),
    ]))))
    if F["anatomy_text"]:
        out.append(_p("🎲 WHY THE WIN RATE LOOKS LOW\n\n" + F["anatomy_text"]))
    return out


_EVIDENCE_LEGEND = "Ev = evidence: ⚪ observation · 🟡 early · 🟠 meaningful · 🔵 strong · 🟢 validated"

def _alert_table(items: List[Dict[str, Any]], limit: int) -> List[_Piece]:
    rows = [("Alert", "EV%", "WR", "N")]
    for a in items[:limit]:
        rows.append((
            f"{_LADDER[a['rank']]} {a['name']}",
            f"{a['ev']:+.2f}", f"{a['wr']:.0%}", str(a['n']),
        ))
    return _c_split(_table(rows, "lrrr")) + [_p(_EVIDENCE_LEGEND)]

def _sec_loss(F: Dict[str, Any], cfg) -> List[_Piece]:
    out = [_hdr(4, '🔎 LOSS DIAGNOSIS — "WHERE ARE WE FALTERING?"')]
    weak = F["weak"]
    if not weak:
        out.append(_p("No alert with at least 10 trades has negative EV in this sample."))
        return out
    out.append(_p("🔴 CURRENTLY WEAK ALERT FAMILIES (worst total loss first)"))
    out.extend(_alert_table(weak, 8))
    if len(weak) > 8:
        out.append(_p(f"…and {len(weak) - 8} more (full list in section 16)."))
    ready = [a for a in weak if a["n"] >= 30 and a["rank"] >= 2]
    if ready and not F["low_trust"]:
        text = ("Alerts with enough evidence to consider disabling: "
                + ", ".join(a["name"] for a in ready[:5]) + ".\nOthers are still 'investigate' only.")
    else:
        text = ("These alerts are currently contributing negative results.\n\n"
                "However, most have small samples.\n\n"
                "➡️ They are \"investigate\" candidates.\n➡️ They are NOT yet \"disable\" candidates.")
    out.append(_p("🧠 INTERPRETATION\n\n" + text))
    return out

def _sec_positive(F: Dict[str, Any], cfg) -> List[_Piece]:
    out = [_hdr(5, '🟢 POSITIVE SIGNS — "WHERE ARE WE DOING BETTER?"')]
    good = F["good"]
    if not good:
        out.append(_p("No alert with at least 10 trades has positive EV in this sample yet."))
        return out
    out.extend(_alert_table(good, 6))
    validated = [a for a in good if a["rank"] == 4]
    out.append(_p(
        "🧠 INTERPRETATION\n\n"
        + ("These alerts have produced positive EV in the current sample, but evidence is still weak.\n\n"
           "➡️ Monitor.\n➡️ Do NOT increase their weight yet." if not validated
           else "Validated: " + ", ".join(a["name"] for a in validated[:5])
           + ".\nThe others remain 'monitor'.")
    ))
    return out

def _sec_scorecard(F: Dict[str, Any], cfg) -> List[_Piece]:
    out = [_hdr(6, '🚦 ALERT SCORECARD — "WHAT SHOULD I TRUST?"')]
    validated = [a for a in F["good"] if a["rank"] == 4]
    promising = [a for a in F["good"] if a["rank"] < 4]

    def _group(title: str, names: List[str], empty: str = "None currently.") -> None:
        body = "\n".join(_wrap_names(names)) if names else "   " + empty
        out.append(_p(f"{title}\n{body}"))

    _group("🟢 VALIDATED / ACTIONABLE", [a["name"] for a in validated])
    _group("🟡 PROMISING — NEED MORE EVIDENCE", [a["name"] for a in promising])
    _group("🔴 UNDERPERFORMING — INVESTIGATE", [a["name"] for a in F["weak"]])
    thin = sorted(F["thin"], key=lambda a: -a["n"])
    _group(f"⚪ INSUFFICIENT DATA ({len(thin)} alert types, fewer than 10 trades)",
           [a["name"] for a in thin])
    zero = len(F["alerts"]) - len(validated) - len(promising) - len(F["weak"]) - len(thin)
    if zero > 0:
        out.append(_p(f"➖ {zero} alert(s) with exactly zero net EV are not listed above."))
    return out

def _sec_coins(F: Dict[str, Any], cfg) -> List[_Piece]:
    out = [_hdr(7, '🪙 COIN ANALYSIS — "WHERE ARE WE WINNING/LOSING?"')]
    pairs = F["pairs"]
    if len(pairs) < 2:
        out.append(_p("Not enough trades per coin yet (need at least 5 on two coins)."))
    else:
        worst, best = pairs[0], pairs[-1]

        def _ev_txt(n_: int) -> str:
            r = _evidence_rank(n_, F["days"])
            return f"{_LADDER[r]} {'TOO SMALL' if r == 0 else 'INVESTIGATE' if r == 1 else 'MEANINGFUL'}"

        out.append(_p(f"🟢 POSITIVE OBSERVATION\n\n{best[0]}\nWR: {best[1]:.0%} | Trades: {best[2]}\n"
                      f"Evidence: {_ev_txt(best[2])}"))
        out.append(_p(f"🔴 NEGATIVE OBSERVATION\n\n{worst[0]}\nWR: {worst[1]:.0%} | Trades: {worst[2]}\n"
                      f"Evidence: {_ev_txt(worst[2])}"))
        out.append(_p(f"{len(pairs)} coins have at least 5 trades."))
    out.append(_p("🧠 RULE:\n\nNo coin should be recommended for removal or increased weight unless "
                  "minimum sample + statistical validation requirements are satisfied."))
    return out


def _sec_sessions(F: Dict[str, Any], cfg) -> List[_Piece]:
    out = [_hdr(8, "⏰ SESSION / TIME ANALYSIS")]
    sess = F["sessions"]                                     # worst-first
    if not sess:
        out.append(_p("No session data yet."))
        return out

    best, worst = sess[-1][0], sess[0][0]
    rows = [("", "SESSION", "WR", "TRADES", "STATUS")]
    for name, swr, sn in sorted(sess, key=lambda t: -t[1]):
        if name == best and len(sess) > 1:
            emoji, tag = "🟡", "Best observed"
        elif name == worst and len(sess) > 1:
            emoji, tag = "🔴", "Weakest observed"
        else:
            emoji, tag = "⚪", ""
        rows.append((emoji, name.upper(), f"{swr:.0%}", str(sn), tag))
    out.extend(_c_split(_table(rows, "llrrl")))
    if len(sess) > 1:
        out.append(_p(f"🧠 Interpretation:\n\n{best.upper()} has performed better in this sample.\n\n"
                      f"{worst.upper()} has performed worse."))
    if F["low_trust"]:
        out.append(_p(f"⚠️ Only {_fmt_days(F['days'])} are available, therefore session effects are "
                      f"observations rather than confirmed conclusions."))
    return out


def _sec_filters(F: Dict[str, Any], cfg) -> List[_Piece]:
    out = [_hdr(9, "🧩 FILTER / CONFLUENCE ANALYSIS")]
    groups: Dict[str, list] = defaultdict(list)
    for r in F["shadow_rows"]:
        groups[r.get("rejection_reason") or "win_rate_filter"].append(r)
    for must in ("calibration_gate", "confluence_gate", "win_rate_filter"):
        groups.setdefault(must, [])
    min_s = getattr(cfg, "MIN_WIN_RATE_SAMPLE", 20)

    rows = [("", "FILTER", "TRADES", "EV", "VERDICT")]
    for reason in sorted(groups):
        g = groups[reason]
        label = reason.replace("_gate", "").replace("_", " ").capitalize()
        if not g:
            rows.append(("⚪", label, "0", "n/a", ""))
            continue
        try:
            ev = engine.ev_first_objective(g, min_sample=min_s)
        except Exception:
            ev = {"valid": False}
        if not ev.get("valid"):
            rows.append(("⚪", label, str(len(g)), "n/a", f"need {min_s}"))
        else:
            ne, pe = ev.get("net_ev", 0.0), ev.get("p_ev_positive", 0.0)
            if ne > 0 and pe >= 0.65:
                emoji, verdict = "🟡", "may over-block"
            else:
                emoji, verdict = "🟢", "filters losers"
            rows.append((emoji, label, str(len(g)), f"{ne:+.2f}%", verdict))
    out.extend(_c_split(_table(rows, "llrrl")))
    out.append(_p(
        "🧠 QUESTION THE BRAIN IS TRYING TO ANSWER:\n\n"
        "\"Are my filters removing bad trades or accidentally removing profitable trades?\"\n\n"
        f"Shadow Mode: {'🟢 ON' if F['shadow_on'] else '🔴 OFF'}"
        + (f" ({len(F['shadow_rows'])} rejected trades tracked)" if F["shadow_rows"] else "")
        + "\n\nRejected trades → eventual outcome → \"Was this rejection actually beneficial?\""
    ))
    return out

def _sec_investigation(F: Dict[str, Any], cfg) -> List[_Piece]:
    out = [_hdr(10, '🔬 BRAIN INVESTIGATION — "WHAT ARE WE TESTING?"')]
    topics = ["Entry quality", "BUY vs SELL performance", "Alert-family performance",
              "Coin performance", "Session performance", "Confluence score", "Volatility regime",
              "Trend/range regime", "Filter effectiveness", "Fees/slippage impact",
              "Alert combinations", "Entry timing"]
    nums = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫"
    out.append(_p("The Brain is currently investigating:\n\n"
                  + "\n".join(f"{nums[i]} {t}" for i, t in enumerate(topics))))
    blockers = _active_blockers(F)
    supp = F["audit"].suppressed_analysis_names()
    if supp:
        blockers.append(f"{len(supp)} advanced analyses waiting for history")
    out.append(_p("CURRENT BLOCKERS:\n\n" + ("\n".join(f"🔴 {b}" for b in blockers) if blockers
                                            else "🟢 None")))
    return out

def _config_json_block(F: Dict[str, Any], cfg) -> Optional[str]:
    changes: Dict[str, Any] = {}
    for p in F["cfg_patch"]:
        if p.get("_blocked_by_action_gate"):
            continue
        if p.get("path") == "CONFLUENCE_WEIGHTS":
            changes["CONFLUENCE_WEIGHTS"] = p.get("suggested", {})
        elif p.get("current") is not None and p.get("suggested") is not None:
            changes[p["path"]] = p["suggested"]
    if F["rec_thr_ok"] and F["thr_tier"] == RecommendationTier.ACTIONABLE:
        changes["CONFLUENCE_MIN_ABS_SCORE"] = round(F["rec_thr"]["recommended"], 1)
    return json.dumps(changes, indent=1) if changes else None

def _cf_verdict(scenario: Dict[str, Any], cfg) -> Tuple[str, str]:
    """Shadow status + promotion verdict for one counterfactual scenario.

    Returns (shadow_status, verdict), both short emoji-prefixed strings for
    display. Promotion requires shadow_validated is True AND the shadow
    sample clears BRAIN_COUNTERFACTUAL_PROMOTE_MIN_SHADOW — mirrors how
    _shadow_weight_check already gates weight-change promotion, applied
    here to threshold/gate candidates.
    """
    sv = scenario.get("shadow_validated")
    shadow_n = scenario.get("shadow_n") or 0
    min_shadow = getattr(cfg, "BRAIN_COUNTERFACTUAL_PROMOTE_MIN_SHADOW", 15)
    if sv is None:
        return "🟡 too thin to validate", "🕒 NOT YET (shadow inconclusive)"
    if sv is False:
        return f"🔴 disagrees (n={shadow_n})", "🚫 REJECTED (curve-fit risk)"
    if shadow_n >= min_shadow:
        return f"🟢 confirmed (n={shadow_n})", "✅ PROMOTION-ELIGIBLE"
    return (f"🟢 confirmed (n={shadow_n})",
            f"🕒 NOT YET (need {min_shadow} shadow, have {shadow_n})")

def _sec_sims(F: Dict[str, Any], cfg) -> List[_Piece]:
    out = [_hdr(11, "🧪 SIMULATIONS / WHAT-IF ANALYSIS")]
    gate = F["gate"]
    if F["rec_thr_ok"]:
        rt = F["rec_thr"]
        validated = F["thr_tier"] == RecommendationTier.ACTIONABLE
        out.append(_p("CONFLUENCE MIN SCORE"))
        out.append(_c("\n".join(_kv_table([
            ("Current", f"{cfg.CONFLUENCE_MIN_ABS_SCORE:.0f}"),
            ("Simulation", f"{rt['recommended']:.0f}"),
        ]))))
        out.append(_p((
            f"Historical simulated WR: {rt.get('rec_wr', 0):.0%}\n\n"
            f"Status:\n{'✅ VALIDATED' if validated else '🚫 NOT VALIDATED'}\n\n"
            + ("" if validated else "Reason:\nIn-sample simulation only.")
        ).rstrip()))
        checks = [
            ("Minimum sample reached", gate.get("data_quality")),
            ("Walk-forward validation passes", gate.get("oos_prediction")),
            ("Out-of-sample EV positive", gate.get("oos_prediction")),
            ("Drawdown acceptable", gate.get("risk")),
            ("No active drift problem", gate.get("stability")),
        ]
        out.append(_p("The Brain will NOT recommend implementation until:\n\n"
                      + "\n".join(f"{'✅' if ok else '❌'} {label}" for label, ok in checks)))
    else:
        out.append(_p("No entry-bar simulation is available yet."))
    cand: List[str] = []
    for p in F["cfg_patch"]:
        blocked = bool(p.get("_blocked_by_action_gate"))
        if p.get("path") == "CONFLUENCE_WEIGHTS":
            cur, sug = p.get("current", {}) or {}, p.get("suggested", {}) or {}
            for vote, new_w in sug.items():
                old_w = cur.get(vote, CONFLUENCE_WEIGHTS.get(vote, 0.0))
                if abs(new_w - old_w) >= 0.05:
                    cand.append(f"{'⬆️' if new_w > old_w else '⬇️'} weight {vote}: {old_w:.1f} → {new_w:.1f}"
                                f" — {'🚫 blocked' if blocked else '✅ approved'}")
            if blocked and not sug:
                cand.append("⚖️ CONFLUENCE_WEIGHTS: candidate changes — 🚫 blocked")
        elif p.get("current") is not None and p.get("suggested") is not None:
            cand.append(f"🔬 {p['path']}: {p['current']} → {p['suggested']}"
                        f" — {'🚫 blocked' if blocked else '✅ approved'}")

    if cand:
        out.append(_p("🔬 CANDIDATE SETTINGS (what the Brain would change):\n\n" + "\n".join(cand)))

    # ── Candidate vs Control (counterfactual simulator, best 5 by EV) ──
    # Full numeric detail (gross EV, delta_n, shadow WR) lives in the
    # Technical Appendix (section 16); this stays short for Telegram.
    cf_scenarios = sorted(
        F["ai"].get("counterfactual_scenarios") or [],
        key=lambda s: s.get("ev", float("-inf")),
        reverse=True,
    )[:5]
    if cf_scenarios:
        lines = []
        for i, s in enumerate(cf_scenarios, 1):
            shadow_status, verdict = _cf_verdict(s, cfg)
            lines.append(
                f"{i}. {s.get('label', '?')}\n"
                f"   EV: {F['net_ev']:+.2f}% → {s.get('ev', 0.0):+.2f}% "
                f"(Δ{s.get('delta_ev', 0.0):+.2f}%, n={s.get('n', 0)})\n"
                f"   Shadow: {shadow_status} → {verdict}"
            )
        out.append(_p(
            "🥇 CANDIDATE vs CONTROL (top 5 by EV, control = current live config):\n\n"
            + "\n\n".join(lines)
        ))
    return out

def _sec_recs(F: Dict[str, Any], cfg) -> List[_Piece]:
    out = [_hdr(12, "🤖 BRAIN RECOMMENDATIONS")]
    act: List[str] = []
    recon_ok = F["recon"] is not None and F["recon"].status == HealthStatus.OK
    if F["wr"] < 0.10 or not recon_ok:
        act.append("Verify outcome recording.")
    act.append("Keep collecting data.")
    act.append("Keep Shadow Mode enabled." if F["shadow_on"] else "Turn Shadow Mode ON.")
    if F["wr"] < 0.10 and F["n"]:
        act.append(f"Investigate the {F['wr']:.0%} WR.")
    if F["gate"].get("stability") is False or F["gate"].get("risk") is False:
        act.append("Investigate current drift/drawdown.")
    out.append(_p("### 🔴 ACT NOW\n\n" + "\n".join(f"{i}. {x}" for i, x in enumerate(act, 1))))
    watch = []
    for a in F["weak"]:
        if a["name"] not in watch:
            watch.append(a["name"])
        if len(watch) >= 6:
            break
    out.append(_p("### 🟡 WATCH\n\n" + ("\n".join(f"{i}. {x}" for i, x in enumerate(watch, 1))
                                       if watch else "Nothing flagged.")))
    cands = [a["name"] for a in F["good"][:3]]
    out.append(_p("### 🟢 POTENTIAL FUTURE CANDIDATES\n\n"
                  + ("\n".join(f"{i}. {x}" for i, x in enumerate(cands, 1)) if cands else "None yet.")
                  + ("\n\n⚠️ None is currently validated enough to increase weighting."
                     if not any(a["rank"] == 4 for a in F["good"]) else "")))
    if F["gate_ok"] and not F["low_trust"]:
        blk = _config_json_block(F, cfg)
        out.append(_p("### ✅ APPROVED TO APPLY\n\n" + ("Copy into config_macd.json:" if blk
                                                       else "No change is supported by the evidence.")))
        if blk:
            out.append(_c(blk))
    else:
        out.append(_p("### 🚫 DO NOT APPLY\n\n• Alert disabling\n• Weight changes\n• Threshold changes\n"
                      "• Confluence score change\n• Automatic Brain plan"))
    return out


def _sec_gate(F: Dict[str, Any], cfg) -> List[_Piece]:
    g = F["gate"]
    out = [_hdr(13, '🛡️ ACTION GATE — "CAN THE BRAIN SAFELY CHANGE ANYTHING?"')]

    labels = [("data_quality", "Minimum trades"), ("oos_prediction", "OOS EV"),
              ("profitability", "Net EV confidence"), ("stability", "CUSUM drift"),
              ("risk", "Drawdown budget"), ("execution", "Cost assumptions")]
    # Emoji at column 0, label, then PASS/FAIL — same shape as _kv_table
    # without needing to wrap because the pairs are built inline here.
    rows = [(f"{'🟢' if g.get(k) else '🔴'} {lab}:", "PASS" if g.get(k) else "FAIL")
            for k, lab in labels]
    out.extend(_c_split(_table(rows, "ll")))
    out.append(_p(
        "OVERALL:\n\n"
        + ("🟢 BRAIN ACTION GATE = PASSED\n\nMeaning:\n\n\"The Brain's evidence is strong enough to "
           "recommend specific changes to the live strategy.\""
           if F["gate_ok"] else
           "🔴 BRAIN ACTION GATE = BLOCKED\n\nMeaning:\n\n\"The Brain may analyse and recommend what "
           "to investigate, but it is not sufficiently confident to modify the live strategy.\"")
    ))
    return out


def _sec_quality(F: Dict[str, Any], cfg) -> List[_Piece]:
    out = [_hdr(14, "📊 DATA QUALITY & RECONCILIATION")]
    cov, rec, st = F["coverage"], F["recon"], F["archive_stats"]

    def _n(v: Any) -> str:
        return "n/a" if v is None else str(v)

    rows = []
    if cov:
        rows += [("Requested history", f"{cov.requested_days} days"),
                 ("Available", f"{cov.actual_days:.1f} days"),
                 ("Coverage", f"{cov.coverage_ratio:.0%}"),
                 ("Resolved trades", str(cov.n_rows))]
    if rec:
        rows += [("Pending at start", _n(rec.pending_count)),
                 ("Resolved this run", _n(rec.resolved_this_run)),
                 ("Archived this run", _n(rec.archived_this_run)),
                 ("Archive lines read", _n(rec.total_archived)),
                 ("Loaded by Brain", str(rec.loaded_by_brain)),
                 ("Shadow loaded", str(rec.shadow_loaded))]
    if rows:
        out.extend(_c_split(_table(rows, "lr")))
    status_rows = []
    if rec:
        status_rows.append(("Outcome reconciliation", f"{rec.status.icon} {rec.status.value}"))
    else:
        status_rows.append(("Outcome reconciliation", "⚪ no data"))
    migrated, dropped = st.get("migrated_forward", 0), st.get("dropped_unmigratable", 0)
    status_rows.append(("Schema migration", (
        "🟠 " + f"{dropped} rows could not be migrated" if dropped
        else f"🟡 {migrated} old rows upgraded (some fields empty)" if migrated else "🟢 OK")))
    out.append(_c("\n".join(_kv_table(status_rows))))
    lines: List[str] = []
    if rec:
        lines += [f"⚠️ {note}" for note in rec.notes]
        for label, val in (("Stale-schema rows excluded", rec.archive_rejects_stale_schema),
                           ("Signal-only rows excluded", rec.archive_rejects_signal_only),
                           ("Malformed rows excluded", rec.archive_rejects_malformed)):
            if val:
                lines.append(f"{label}: {val}")
    if cov and cov.warnings:
        lines += [f"⚠️ {w}" for w in cov.warnings]
    if lines:
        out.append(_p("\n".join(lines)))
    supp = F["audit"].suppressed_analysis_names()
    if supp:
        out.append(_p("Advanced analyses blocked (not enough history yet):\n\n"
                      + "\n".join(f"• {s.replace('_', ' ').title()}" for s in supp)))
    return out


def _sec_evidence(F: Dict[str, Any], cfg) -> List[_Piece]:
    out = [_hdr(15, "📈 EVIDENCE / CONFIDENCE")]
    days, n, g = F["days"], F["n"], F["gate"]
    recon_ok = F["recon"] is not None and F["recon"].status == HealthStatus.OK
    out.extend(_c_split(_kv_table([
        ("Data integrity", '🟢 GOOD' if recon_ok else '🟡 CHECK'),
        ("Trade count", '🟢 GOOD' if n >= 100 else '🟡 OK' if n >= 30 else '🔴 POOR'),
        ("History depth", '🟢 GOOD' if (days or 0) >= 30 else '🟡 FAIR' if (days or 0) >= 14 else '🔴 POOR'),
        ("Market regime coverage", '🟢 GOOD' if (days or 0) >= 30 else '🟡 FAIR' if (days or 0) >= 14 else '🔴 POOR'),
        ("OOS validation", '🟢 PASSED' if g.get('oos_prediction') else '🔴 BLOCKED'),
        ("Statistical confidence", _conf_status(F['conf'])),
        ("Strategy-change confidence", '🟢 HIGH' if F['gate_ok'] else '🔴 LOW'),
    ])))
    used = sorted({a["rank"] for a in F["alerts"]})
    ladder = "\n".join(f"{_LADDER[i]} {_LADDER_NAMES[i]}" for i in range(5))
    now = " / ".join(_LADDER[i] for i in used) if used else "⚪"
    out.append(_p(f"CONFIDENCE LADDER\n\n{ladder}\n\nMost current findings:\n{now}\n\n"
                  + ("Therefore:\nNo aggressive optimisation yet." if F["low_trust"] or not F["gate_ok"]
                     else "Therefore:\nValidated findings may be acted on, one change at a time.")))
    return out


def _sec_appendix(F: Dict[str, Any], cfg) -> List[_Piece]:
    out = [_hdr(16, "📚 TECHNICAL APPENDIX")]
    out.append(_p("Detailed data for advanced users."))
    rows = [("Alert", "N", "WR", "EV", "P>0")]
    for a in sorted(F["alerts"], key=lambda a: -a["n"]):
        ev = f"{a['ev']:+.2f}" if a["ev"] is not None else "n/a"
        pv = f"{a['p']:.0%}" if a["p"] is not None else "n/a"
        rows.append((f"{_LADDER[a['rank']]} {a['key']}",
                     str(a['n']), f"{a['wr']:.0%}", ev, pv))
    out.append(_p("• Full alert-by-alert statistics (EV = net % per trade, P>0 = P(EV>0))"))
    out.extend(_c_split(_table(rows, "lrrrr")))
    ev_obj = F["ev_obj"] or {}
    if ev_obj:
        out.append(_p("• Portfolio EV: net " + f"{ev_obj.get('net_ev', 0):+.2f}%"
                      + (f", P5 {ev_obj['ev_p5']:+.2f}%" if ev_obj.get("ev_p5") is not None else "")
                      + (f", P(EV>0) {ev_obj['p_ev_positive']:.0%}" if ev_obj.get("p_ev_positive") is not None else "")
                      + (f", max drawdown {ev_obj['max_drawdown_pct']:.1f}%" if ev_obj.get("max_drawdown_pct") is not None else "")))
    health = F["audit"].analysis_health_entries()
    if health:
        out.append(_p("• Analysis health\n" + "\n".join(e.report_line.strip() for e in health)))
    g = F["gate"]
    if g:
        out.append(_p("• Gate diagnostics\n" + "\n".join(
            f"{'✅' if v else '❌'} {k}" for k, v in g.items() if isinstance(v, bool))))
    st = F["archive_stats"]
    if st:
        out.append(_p("• Archive reconciliation\n" + "\n".join(
            f"{k}: {v}" for k, v in st.items() if isinstance(v, (int, float)))))
    ledger = (F["ai"] or {}).get("repair_ledger")
    if ledger:
        out.append(_p("• Repair ledger\n" + "\n".join(f"{k}: {v}" for k, v in list(ledger.items())[:10])))
    if F["cfg_patch"]:
        out.append(_p("• Full Brain candidate table\n" + "\n".join(
            f"{p.get('path')}: {p.get('current')} → {p.get('suggested')}"
            f"{' [blocked]' if p.get('_blocked_by_action_gate') else ''}"
            for p in F["cfg_patch"][:20])))
    cf_all = F["ai"].get("counterfactual_scenarios") or []
    if cf_all:
        cf_rows = [("Scenario", "N", "WR", "EV", "GrossEV", "Δn", "ShadowN", "ShadowΔEV", "ShadowWR", "Valid")]
        for s in sorted(cf_all, key=lambda s: s.get("ev", float("-inf")), reverse=True):
            sv = s.get("shadow_validated")
            cf_rows.append((
                str(s.get("label", "?")), str(s.get("n", 0)), f"{s.get('wr', 0.0):.0%}",
                f"{s.get('ev', 0.0):+.2f}", f"{s.get('gross_ev', 0.0):+.2f}",
                f"{s.get('delta_n', 0):+d}",
                str(s.get("shadow_n")) if s.get("shadow_n") is not None else "n/a",
                f"{s['shadow_delta_ev']:+.2f}" if s.get("shadow_delta_ev") is not None else "n/a",
                f"{s['shadow_wr']:.0%}" if s.get("shadow_wr") is not None else "n/a",
                "yes" if sv is True else "no" if sv is False else "n/a",
            ))
        out.append(_p("• Full counterfactual scenario table (all evaluated, unranked cap)"))
        out.extend(_c_split(_table(cf_rows, "lrrrrrrrrr")))
    out.append(_p("Also computed but not shown here: Monte Carlo, CUSUM, walk-forward, calibration, "
                  "permutation importance, hierarchical analysis and weight optimisation "
                  "(each appears in 'Analysis health' above when it ran)."))
    return out

_REPORT_SECTIONS = (
    ("EXECUTIVE SUMMARY", _sec_summary), ("WHAT TO DO NOW", _sec_do_now),
    ("PROFITABILITY", _sec_profit), ("LOSS DIAGNOSIS", _sec_loss),
    ("POSITIVE SIGNS", _sec_positive), ("ALERT SCORECARD", _sec_scorecard),
    ("COIN ANALYSIS", _sec_coins), ("SESSION ANALYSIS", _sec_sessions),
    ("FILTER ANALYSIS", _sec_filters), ("INVESTIGATION", _sec_investigation),
    ("SIMULATIONS", _sec_sims), ("RECOMMENDATIONS", _sec_recs),
    ("ACTION GATE", _sec_gate), ("DATA QUALITY", _sec_quality),
    ("EVIDENCE", _sec_evidence), ("TECHNICAL APPENDIX", _sec_appendix),
)

def build_brain_report_sections(recs: Dict[str, Any], cfg) -> Tuple[List[List["_Piece"]], str]:
    """Compute the 16 report sections once. Returns (sections, stamp); each
    section is a list of rendered pieces. An individual failing section is
    replaced by a notice; the call raises only if the shared facts cannot
    be computed."""
    F = _collect_facts(recs, cfg)
    stamp = datetime.now(_IST).strftime("%d %b %Y | %H:%M IST").upper()
    sections: List[List[_Piece]] = []
    failed: List[str] = []
    for i, (name, fn) in enumerate(_REPORT_SECTIONS, 1):
        try:
            sections.append(fn(F, cfg))
        except Exception as e:
            _report_section_failed(failed, name, e)
            sections.append([_hdr(i, name), _p("⚠️ This section is unavailable (analysis error — see logs).")])
    return sections, stamp

def render_report_messages(sections: List[List[_Piece]], stamp: str) -> List[str]:
    """Pack the sections into Telegram-ready MarkdownV2 messages..."""
    msgs: List[str] = []
    
    # Explicitly type as str to allow f-string reassignments later
    cur: str = _p(f"{'═' * 30}\n🧠 BRAIN REPORT\n{stamp}\n{'═' * 30}")
    
    def _flush() -> None:
        nonlocal cur
        if cur:
            msgs.append(cur)
            cur = ""
            
    for idx, pieces in enumerate(sections, 1):
        if len(pieces) > 1:                            # never strand a header from its body
            # Combine header and first body piece.
            # FIX 1: Use \n\n for proper Telegram paragraph spacing.
            # FIX 2: Set kind="p" so render_report_markdown() doesn't 
            # accidentally format the body text as a Markdown heading (##).
            combined_rendered = f"{pieces[0]}\n\n{pieces[1]}"
            combined_raw = f"{pieces[0].raw}\n\n{pieces[1].raw}"
            combined = _Piece(combined_rendered, "p", combined_raw)
            pieces = [combined] + pieces[2:]
            
        # RESTORED: Original optimization to keep small sections intact
        whole = "\n\n".join(pieces)
        if cur and len(cur) + len(whole) + 2 > _MSG_LIMIT and len(whole) <= _MSG_LIMIT:
            _flush()                                   # small section: start a fresh message
            
        for piece in pieces:
            if cur and len(cur) + len(piece) + 2 > _MSG_LIMIT:
                _flush()
            cur = f"{cur}\n\n{piece}" if cur else piece
            
        if idx == _HUMAN_SECTIONS:
            divider = _p("▼ SECTIONS 06–16: THE EVIDENCE BEHIND THE ABOVE ▼")
            if cur and len(cur) + len(divider) + 2 > _MSG_LIMIT:
                _flush()
            cur = f"{cur}\n\n{divider}" if cur else divider
            
    tail = _p(f"{'═' * 30}\nEND OF BRAIN REPORT\n{'═' * 30}")
    if cur and len(cur) + len(tail) + 2 <= _MSG_LIMIT:
        cur = f"{cur}\n\n{tail}"
    else:
        _flush()
        cur = tail
    _flush()
    return msgs

def _md_prose(raw: str) -> str:
    """Plain prose -> Markdown: keep line breaks, neutralise * _ ` and \\."""
    text = raw.replace("\\", "\\\\")
    for ch in ("*", "_", "`"):
        text = text.replace(ch, "\\" + ch)
    paragraphs = [p.replace("\n", "  \n") for p in text.split("\n\n")]
    return "\n\n".join(paragraphs)


def render_report_markdown(sections: List[List[_Piece]], stamp: str) -> str:
    """The same report as a Markdown document (for the reports/ archive):
    no Telegram escaping, real headings, tables kept as code blocks."""
    out: List[str] = [f"# 🧠 Brain Report — {stamp}"]
    for idx, pieces in enumerate(sections, 1):
        for piece in pieces:
            kind = getattr(piece, "kind", "p")
            raw = getattr(piece, "raw", str(piece))
            if kind == "h":
                out.append("## " + raw)
            elif kind == "c":
                out.append("```\n" + raw + "\n```")
            else:
                out.append(_md_prose(raw))
        if idx == _HUMAN_SECTIONS:
            out.append("---\n\n*Sections 06–16: the evidence behind the above.*")
    out.append("---\n\n*End of Brain report.*")
    return "\n\n".join(out) + "\n"


def build_brain_report(recs: Dict[str, Any], cfg) -> List[str]:
    """Convenience wrapper: sections -> Telegram messages."""
    sections, stamp = build_brain_report_sections(recs, cfg)
    return render_report_messages(sections, stamp)

class BrainEngineV2(BaseBrainEngine):
    """Drop-in replacement for BrainEngine. Inherits the original and adds
    prescriptive phases 1.5-6 plus actionability scoring."""

    def __init__(self, sdb: RedisStateStore):
        super().__init__(sdb)
        self._phase_samples = _PHASE_MIN_SAMPLES
        self._recs_cache: Optional[Dict[str, Any]] = None
        self._recs_cache_ts: float = 0.0
        self._repair_success_rates: Dict[str, Dict[str, float]] = {}
        self._ledger_stats: Dict[str, Any] = {}
        self._repair_help_preds: Dict[str, float] = {}
        self._rows_cache: Optional[Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]] = None
        self._recent_rows_cache: Optional[List[Dict[str, Any]]] = None
        self._layered_rows_cache: Optional[Tuple[
            List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]
        ]] = None

    @staticmethod
    def _shadow_weight_check(
        shadow_rows, current_weights, suggested_weights,
        min_n=15, max_wr_drop=0.05,
    ):
        """Out-of-sample veto on proposed weight changes.

        FIX (Priority 4): Reproduces the ACTUAL live confluence gate:
            required = max(CONFLUENCE_MIN_ABS_SCORE,
                           CONFLUENCE_MIN_PCT / 100 * total)

        Shadow rows are alerts the live system REJECTED — an independent
        sample from the same window. Score them under current vs suggested
        weights and veto if the proposal materially degrades WR or net EV.

        FIX (Issue 5 cleanup): dropped the unused `threshold` parameter —
        abs_floor is read directly from cfg inside the function.
        """
        min_pct = getattr(cfg, "CONFLUENCE_MIN_PCT", 60.0)
        abs_floor = getattr(cfg, "CONFLUENCE_MIN_ABS_SCORE", 18.0)

        def _wr_and_ev_at(rows, weights):
            kept = []
            for r in rows:
                votes = r.get("votes")
                if not votes:
                    continue
                # Compute score and total the same way the live path does
                score = sum(w for vn, w in weights.items() if votes.get(vn))
                total = sum(w for vn, w in weights.items() if vn in votes)
                if total <= 0:
                    continue
                # Reproduce the actual gate: max(abs_floor, min_pct% of total)
                required = max(abs_floor, total * (min_pct / 100.0))
                if score >= required:
                    kept.append(r)
            if len(kept) < min_n:
                return None, None, len(kept)
            wr = sum(r["win"] for r in kept) / len(kept)
            ev, _hk, _wr = engine.ev_and_kelly_for(kept)
            return wr, ev, len(kept)

        cur_wr, cur_ev, cur_n = _wr_and_ev_at(shadow_rows, current_weights)
        new_wr, new_ev, new_n = _wr_and_ev_at(shadow_rows, suggested_weights)

        if cur_wr is None or new_wr is None:
            return True, f"shadow too thin to veto (cur n={cur_n}, new n={new_n})"

        if new_wr < cur_wr - max_wr_drop:
            return False, f"suggested weights degrade shadow WR {cur_wr:.0%}→{new_wr:.0%} (n={new_n})"

        if new_ev is not None and cur_ev is not None and new_ev < cur_ev - 0.02:
            return False, f"suggested weights degrade shadow EV {cur_ev:+.3f}%→{new_ev:+.3f}% (n={new_n})"

        return True, f"shadow WR stable {cur_wr:.0%}→{new_wr:.0%}, EV {cur_ev:+.3f}%→{new_ev:+.3f}% (n={new_n})"

    @staticmethod
    def _action_gate_check(
        real_rows: List[Dict[str, Any]],
        min_sample: int = 20,
        recommendations: Optional[List[Dict[str, Any]]] = None,
        active_drift_keys: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Six-layer confirmation gate. Returns which patches are actionable.

        `active_drift_keys` is the persisted CUSUM alarm set (s_neg > h),
        supplied by the caller. When present, it is authoritative for the
        stability layer: a detector whose alarm persists in Redis keeps
        the gate closed even if no new cusum_drift recommendation fires
        this cycle. Without it, the check falls back to this report's rec
        list — the old behaviour, which flickers run-to-run.
        """
        gate: Dict[str, bool] = {
            "data_quality": len(real_rows) >= ACTION_GATE_MIN_ROWS,
            "oos_prediction": False,
            "profitability": False,
            "stability": True,
            "risk": True,
            "execution": True,
        }
        # OOS prediction: rolling walk-forward must pass
        _rwc_allowed = True
        try:
            _rwc_allowed, _ = get_audit().can_run("walk_forward")
        except Exception as _e:
            logging.getLogger("macd_bot").debug(
                f"Audit unavailable for walk_forward gate, running check: {_e}"
            )  # fail-open: run the real check if the audit is unavailable

        if _rwc_allowed:
            rwc = engine.rolling_walk_forward(real_rows, n_folds=5)
            if rwc.get("valid"):
                gate["oos_prediction"] = rwc["p_ev_positive"] >= 0.70

        # Profitability: net EV must be positive with high confidence
        ev_obj = engine.ev_first_objective(real_rows, min_sample=min_sample)

        if ev_obj.get("valid"):
            gate["profitability"] = (
                ev_obj["p_ev_positive"] >= getattr(cfg, "BRAIN_EV_GATE_P_THRESHOLD", 0.85)
                and ev_obj["ev_p5"] > getattr(cfg, "BRAIN_EV_GATE_P5_FLOOR", -0.10)
            )
    
        # ── Stability: no active CUSUM edge-decay alarm. Prefer the
        if active_drift_keys:
            gate["stability"] = False
        elif recommendations is not None:
            gate["stability"] = not any(
                r.get("type") == "cusum_drift" for r in recommendations
            )
        # ── Risk: realized max drawdown must stay inside the same budget
        # the live kill switch enforces. ev_obj is already computed above,
        # so this reuses it instead of a second pass over the rows. ──
        if ev_obj.get("valid"):
            dd_budget = getattr(cfg, "KILL_SWITCH_MAX_DRAWDOWN_PCT", 3.0)
            gate["risk"] = ev_obj["max_drawdown_pct"] <= dd_budget
        else:
            gate["risk"] = False

        # ── Execution: cost assumptions must be non-trivial, or every EV
        # figure above is silently optimistic. ──
        fee_pct = getattr(cfg, "BRAIN_FEE_PCT", 0.0006)
        slip_pct = getattr(cfg, "BRAIN_SLIPPAGE_PCT", 0.0003)
        gate["execution"] = fee_pct > 0 and slip_pct > 0

        gate["actionable"] = all(gate.values())
        return gate

    @staticmethod
    def _match_shadow_regime(rpo_shadow, rng):
        """Shadow regime overlapping a real regime's ADX range (≥50% span overlap)."""
        if not rpo_shadow or not rpo_shadow.get("valid"):
            return None
        lo, hi = rng
        for sreg in rpo_shadow.get("regimes", []):
            slo, shi = sreg["range"]
            inter = min(hi, shi) - max(lo, slo)
            span = min(hi - lo, shi - slo)
            if span > 0 and inter >= 0.5 * span:
                return sreg
        return None

    async def _get_rows(self) -> tuple:
        """Override base class: read from file archive first, fall back to Redis."""
        return await self._load_rows()

    async def _get_layered_window_rows(self) -> tuple:
        """Override base class: read the recent/medium/long layered
        windows from the file archive first (mirrors _load_rows()'s
        file-storage branching), falling back to the base class's
        Redis-stream path only when no archive directory is configured."""
        cached = getattr(self, "_layered_rows_cache", None)
        if cached is not None:
            return cached

        recent_days = getattr(cfg, "BRAIN_ANALYSIS_WINDOW_DAYS", 30)
        medium_days = getattr(cfg, "BRAIN_MEDIUM_WINDOW_DAYS", 90)
        long_days = getattr(cfg, "BRAIN_LONG_WINDOW_DAYS", 180)

        data_dir = getattr(cfg, "OUTCOME_DATA_DIR", None) or os.environ.get("OUTCOME_DATA_DIR")

        if data_dir and Path(data_dir).exists():
            cached_recent = getattr(self, "_recent_rows_cache", None)
            recent_rows = cached_recent if cached_recent is not None else load_archived_outcomes(
                data_dir, window_days=recent_days, shadow=False
            )
            medium_rows = load_archived_outcomes(data_dir, window_days=medium_days, shadow=False)
            long_rows = load_archived_outcomes(data_dir, window_days=long_days, shadow=False)

            result = (recent_rows, medium_rows, long_rows)
            self._layered_rows_cache = result
            return result

        result = await super()._get_layered_window_rows()
        self._layered_rows_cache = result
        return result

    async def _load_rows(self) -> tuple:
        """Shared row loader — reads from archived files if available,
        otherwise falls back to Redis streams. Memoized per report cycle."""
        cached = getattr(self, "_rows_cache", None)
        if cached is not None:
            # Keep the layered-window recent cache in sync on the hit path
            # so a future partial invalidation cannot serve stale rows.
            if self._recent_rows_cache is None:
                self._recent_rows_cache = cached[0]
            return cached
        window_days = getattr(cfg, "BRAIN_ANALYSIS_WINDOW_DAYS", 30)
        
        data_dir = getattr(cfg, "OUTCOME_DATA_DIR", None) or os.environ.get("OUTCOME_DATA_DIR")
        
        if data_dir and Path(data_dir).exists():
            real_rows, real_stats = load_archived_outcomes(
                data_dir, window_days=window_days, shadow=False,
                return_stats=True,
            )
            shadow_rows, shadow_stats = load_archived_outcomes(
                data_dir, window_days=window_days, shadow=True,
                return_stats=True,
            )
            if real_rows or shadow_rows:
                logger = logging.getLogger("macd_bot")
                logger.info(
                    f"🗄️ Brain using file archive: {len(real_rows)} real, "
                    f"{len(shadow_rows)} shadow rows | "
                    f"Archive stats: kept={real_stats['kept']}, "
                    f"migrated={real_stats['migrated_forward']}, "
                    f"unmigratable={real_stats['dropped_unmigratable']}, "
                    f"signal_only={real_stats['dropped_missing_win']}, "
                    f"malformed={real_stats['lines_malformed']}"
                )
                audit = get_audit()
                # Lifecycle counters exist only when this process ran the
                # pending-outcome pre-scan + resolution (not --brain-only).
                _pend_map = getattr(self.sdb, "_pending_outcome_keys_by_pair", None)
                _ran_resolution = _pend_map is not None
                # Extract to a variable so mypy can properly narrow _pend_map to non-None 
                # and correctly infer the generator's item type (int) for sum().
                pending_count = sum(len(v) for v in _pend_map.values()) if _pend_map is not None else None
                audit.set_reconciliation(
                    pending_count=pending_count,
                    resolved_this_run=(
                        getattr(self.sdb, "_run_resolved_total", None)
                        if _ran_resolution else None
                    ),
                    archived_this_run=(
                        getattr(self.sdb, "_run_archived_total", None)
                        if _ran_resolution
                        and getattr(cfg, "BRAIN_USE_FILE_STORAGE", False)
                        else None
                    ),
                    loaded_by_brain=len(real_rows),
                    shadow_loaded=len(shadow_rows),
                    archive_stats=real_stats,
                )
                self._rows_cache = (real_rows, shadow_rows)
                self._recent_rows_cache = real_rows
                return real_rows, shadow_rows

        sample_size = getattr(cfg, "BRAIN_REPORT_STREAM_SAMPLE", 5000)
        long_sample = getattr(cfg, "BRAIN_LONG_WINDOW_STREAM_SAMPLE", 15000)
        fetch_size = max(sample_size, long_sample)
        real_raw, shadow_raw = await asyncio.gather(
            self._read_stream(RedisKeyPrefix.OUTCOME_LOG_STREAM, fetch_size),
            self._read_stream(RedisKeyPrefix.SHADOW_LOG_STREAM, sample_size),
        )
        # Keep the base class's layered-window fast path fed: base
        # _get_layered_window_rows() re-filters _cached_real_raw instead of
        # hitting Redis a second time.
        self._cached_real_raw = real_raw
        real_rows = self._parse_rows(real_raw, window_days=window_days)
        shadow_rows = self._parse_rows(shadow_raw, window_days=window_days)
        self._rows_cache = (real_rows, shadow_rows)
        return real_rows, shadow_rows

    async def generate_recommendations(self) -> Dict[str, Any]:
        """Caching wrapper so the action plan + technical report share one compute."""
        now = time.time()
        if self._recs_cache is not None and (now - self._recs_cache_ts) < 120:
            return self._recs_cache
        # Invalidate the row-level caches so a fresh report re-reads the
        # archive, but calls within the same report cycle reuse them.
        self._rows_cache = None
        self._recent_rows_cache = None
        self._layered_rows_cache = None
        result = await self._generate_recommendations_full()
        self._recs_cache = result
        self._recs_cache_ts = now
        return result

    async def _generate_recommendations_full(self) -> Dict[str, Any]:
        # ── Phase timer — one INFO line per phase so a slow report can be
        # diagnosed from the workflow log without a profiler. Overhead is
        # one time.time() call per mark; negligible against the phases. ──
        _phase_t0 = time.time()
        def _phase_mark(_label: str) -> None:
            nonlocal _phase_t0
            _now = time.time()
            logger.info(f"⏱️ Brain phase '{_label}': {_now - _phase_t0:.2f}s")
            _phase_t0 = _now

        # ── 0. Baseline (original brain logic) ───────────────────────────
        base_recs = await self._generate_baseline_recommendations()
        logger = logging.getLogger("macd_bot")
        _phase_mark("baseline")
        real_rows = base_recs.get("_real_rows", [])
        shadow_rows = base_recs.get("_shadow_rows", [])
        # ════════════════════════════════════════���═════════════════════════
        #  BRAIN AUDIT LAYER — initialize and validate data population
        # ══════════════════════���════════════════���══════════════════════════  
        audit = get_audit()  # keep coverage/reconciliation set during baseline

        # History coverage was already set in _generate_baseline_recommendations
        long_days = getattr(cfg, "BRAIN_LONG_WINDOW_DAYS", 180)
        history = audit._history

        # Log coverage warning (structured, replaces the old ad-hoc warning)
        if history and history.coverage in (DataCoverage.SEVERELY_LIMITED, DataCoverage.CRITICAL):
            logger.warning(
                f"⚠️ Brain audit: DATA COVERAGE = {history.coverage.value}. "
                f"Requested {long_days}d, have {history.actual_days:.1f}d. "
                f"Multi-window analyses will be suppressed or degraded."
            )
        recommendations: List[Dict[str, Any]] = list(base_recs.get("recommendations", []))
        config_patch: List[Dict[str, Any]] = list(base_recs.get("config_patch", []))
        ai_metrics: Dict[str, Any] = dict(base_recs.get("ai_metrics", {}))

        min_sample = getattr(cfg, "MIN_WIN_RATE_SAMPLE", 20)
        disable_wr = getattr(cfg, "BRAIN_ALERT_DISABLE_THRESHOLD_WR", 0.40)
        star_wr = getattr(cfg, "BRAIN_STAR_ALERT_WR", 0.70)
        max_weight_delta = getattr(cfg, "BRAIN_WEIGHT_OPTIMIZER_MAX_DELTA", 2.0)
        wf_weight_opt = getattr(cfg, "BRAIN_WEIGHT_OPTIMIZER_WALK_FORWARD", True)

        # ── Repair Ledger: close the loop on past repairs ────────────────
        try:
            fresh = await evaluate_pending_repairs(
                self.sdb, real_rows, horizon_hours=48, min_outcomes=30,
            )
            if fresh:
                logger.info(f"📒 Repair ledger: evaluated {len(fresh)} pending repair(s)")
            self._repair_success_rates = await repair_success_rates(self.sdb)
            self._ledger_stats = await ledger_stats(self.sdb)
            ai_metrics["repair_ledger"] = dict(self._ledger_stats)
        except Exception as e:
            audit.record_analysis_exception("repair_ledger", e)
            self._repair_success_rates = {}
            self._ledger_stats = {}
        _phase_mark("repair_ledger")

        # ── ML: contextual repair-effectiveness model ───────────────────
        # Learns P(repair helps | system state) from resolved ledger entries,
        # then annotates each new repair with that probability below.
        try:
            ledger_entries = await load_ledger_entries(self.sdb)
            current_state = {
                "overall_wr": (
                    sum(1 for r in real_rows if r["win"]) / len(real_rows)
                ) if real_rows else None,
                "n": len(real_rows),
                "net_ev": ai_metrics.get("net_ev"),
                "brier": ai_metrics.get("brier_score"),
            }
            rem = engine.learn_repair_effectiveness(
                ledger_entries, current_state, min_records=50,
            )
            if rem.get("valid"):
                self._repair_help_preds = rem["p_help_by_category"]
                ai_metrics["repair_effectiveness_model"] = rem
        except Exception as e:
            audit.record_analysis_exception("repair_effectiveness_model", e)
            self._repair_help_preds = {}

        # ── REPAIR SHOP (runs first — highest priority) ──────────────────
        drift_alerts = [r for r in recommendations if r.get("type") == "cusum_drift"]
        repairs = engine.repair_shop_diagnosis(
            real_rows, drift_alerts,
            config={
                "CONFLUENCE_MIN_ABS_SCORE": cfg.CONFLUENCE_MIN_ABS_SCORE,
                "CONFLUENCE_MIN_PCT": cfg.CONFLUENCE_MIN_PCT,
            },
            target_wr=cfg.MIN_WIN_RATE,
            disable_wr=disable_wr,
            min_sample=min_sample,
        )
        for repair in repairs:
            wrapped = {
                "type": "repair_shop",
                "severity": repair["severity"],
                "category": repair["category"],
                "message": (
                    f"🔧 [{repair['category'].upper()}] {repair['diagnosis']}\n"
                    f"   → {repair['action']}\n"
                    f"   Impact: {repair['expected_impact']}"
                ),
                # ── FDR: carry the p-value through for the BH pass ──
                "p_value": repair.get("p_value"),
                "posterior": repair.get("posterior"),
                # ── Scope: the subset of trades this repair can affect.
                "scope": repair.get("scope"), 
                "version_before": repair.get("version_before"),
                "version_after": repair.get("version_after"),
                "delta_wr": repair.get("delta_wr"),
                # Wiring #4: mechanical config-patch fields, when present.
                "config_field": repair.get("config_field"),
                "config_current": repair.get("config_current"),
                "config_suggested": repair.get("config_suggested"),
            }
            # ── ML: annotate with learned P(helps) for this category ──
            cat = repair.get("category")
            if cat in self._repair_help_preds:
                wrapped["p_helps_learned"] = self._repair_help_preds[cat]

            # ── Ledger: record the issue with a pre-repair snapshot.
            # real_rows lets the ledger compute scope_wr/scope_n so the
            # verdict later compares like-for-like on the affected subset
            # rather than the whole book. ──
            try:
                snapshot = {
                    "overall_wr": (sum(1 for r in real_rows if r["win"]) / len(real_rows))
                                   if real_rows else None,
                    "n": len(real_rows),
                    "net_ev": ai_metrics.get("net_ev"),
                    "brier": ai_metrics.get("brier_score"),
                }
                rid = await record_repair_issued(
                    self.sdb, wrapped, snapshot, real_rows=real_rows,
                )
                if rid:
                    wrapped["_repair_id"] = rid
            except Exception as e:
                audit.record_analysis_exception("repair_ledger_write", e)          
            recommendations.append(wrapped)

        _phase_mark("repair_shop")

        # ── Wiring #3: change-point regression → concrete revert patch ──
        for repair in repairs:
            if repair.get("category") != "config_regression_pinpoint":
                continue
            if (repair.get("delta_wr") or 0) >= 0:
                continue  # only regressions warrant a revert
            version_before = repair.get("version_before")
            version_after = repair.get("version_after")
            if not version_before or version_before == version_after:
                continue
            prior = await self._lookup_config_version(version_before)
            if not prior:
                continue
            for field in CONFIG_OVERRIDE_ALLOWED_FIELDS:
                prior_val = prior.get(field)
                current_val = getattr(cfg, field, None)
                if prior_val is None or current_val is None:
                    continue
                if prior_val == current_val:
                    continue
                config_patch.append({
                    "path": field,
                    "current": current_val,
                    "suggested": prior_val,
                    "reason": (
                        f"Revert to config version {version_before}: WR fell "
                        f"{repair.get('delta_wr', 0):+.0%} at the change point."
                    ),
                    "_source_category": "config_regression_pinpoint",
                })

        _phase_mark("wiring_and_config_regression")

        # ── Phase 1.5: Vote Weight Optimizer (FIXED) ─────────────────────
        _wopt_ok, _wopt_why = audit.can_run("weight_optimizer")
        if not _wopt_ok:
            audit.record_analysis(
                "weight_optimizer", HealthStatus.INSUFFICIENT_DATA,
                detail=_wopt_why,
            )
        if _wopt_ok and len(real_rows) >= self._phase_samples["weight_optimizer"]:
            wopt = optimize_vote_weights(
                real_rows, CONFLUENCE_WEIGHTS,
                min_sample=self._phase_samples["weight_optimizer"],
                walk_forward=wf_weight_opt,
                max_weight_delta=max_weight_delta,
            )

            if wopt.get("valid"):
                changed = wopt.get("changed_votes", [])
                wf_status = "✅ WF-validated" if wopt.get("walk_forward_passed") else "⚠️ No WF data"
                conf_label = wopt.get("confidence_label", "LOW")
                conf_score = wopt.get("confidence", 0.0)

                # FIX: read the configured floor instead of hardcoding 0.4
                min_conf = getattr(cfg, "BRAIN_WEIGHT_OPTIMIZER_MIN_CONFIDENCE", 0.4)

                # ── FIX (Priority 4): OOS veto through the SAME effective
                oos_weight_ok = True
                oos_weight_note = ""
                oos_test_ran = False
                if wopt.get("walk_forward_passed") and len(real_rows) >= 200:
                    train_rows_wf, holdout_rows_wf = engine.walk_forward_split(real_rows)
                    if len(holdout_rows_wf) >= 20:
                        min_pct = getattr(cfg, "CONFLUENCE_MIN_PCT", 60.0)
                        abs_floor = getattr(cfg, "CONFLUENCE_MIN_ABS_SCORE", 18.0)

                        def _kept_at(rows, weights):
                            kept = []
                            for r in rows:
                                votes = r.get("votes")
                                if not votes:
                                    continue
                                score = sum(w for vn, w in weights.items() if votes.get(vn))
                                total = sum(w for vn, w in weights.items() if vn in votes)
                                if total <= 0:
                                    continue
                                required = max(abs_floor, total * (min_pct / 100.0))
                                if score >= required:
                                    kept.append(r)
                            return kept

                        cur_kept = _kept_at(holdout_rows_wf, CONFLUENCE_WEIGHTS)
                        sug_kept = _kept_at(holdout_rows_wf, wopt["suggested_weights"])
                        if len(cur_kept) >= 10 and len(sug_kept) >= 10:
                            oos_test_ran = True
                            cur_ev_oos, _, _ = engine.ev_and_kelly_for(cur_kept)
                            sug_ev_oos, _, _ = engine.ev_and_kelly_for(sug_kept)
                            if sug_ev_oos < cur_ev_oos - 0.01:
                                oos_weight_ok = False
                                oos_weight_note = (
                                    f"OOS veto: suggested EV {sug_ev_oos:+.3f}% < "
                                    f"current {cur_ev_oos:+.3f}%"
                                )
                            else:
                                oos_weight_note = (
                                    f"OOS pass: suggested EV {sug_ev_oos:+.3f}% vs "
                                    f"current {cur_ev_oos:+.3f}%"
                                )
                        else:
                            oos_weight_note = (
                                f"OOS EV test skipped: gate-empty arms "
                                f"(cur={len(cur_kept)}, sug={len(sug_kept)})"
                            )
                    else:
                        oos_weight_note = (
                            f"OOS EV test skipped: holdout too thin "
                            f"({len(holdout_rows_wf)} rows after split)"
                        )

                # ── Shadow out-of-sample veto ─────────�������────────────
                shadow_weight_ok, shadow_weight_note = True, ""
                if (wopt.get("walk_forward_passed") and conf_score >= min_conf
                and len(shadow_rows) >= 15 and oos_weight_ok):
                    shadow_weight_ok, shadow_weight_note = self._shadow_weight_check(
                        shadow_rows, CONFLUENCE_WEIGHTS,
                        wopt["suggested_weights"],
                    )        

                # ── Emit recommendation (NOW oos_weight_note is defined) ──
                if changed:
                    change_strs = [f"{k}: {old:.1f}→{new:.1f}" for k, old, new in changed[:6]]
                    extra = f" (+{len(changed)-6} more)" if len(changed) > 6 else ""
                    oos_note = f"\n{oos_weight_note}" if oos_weight_note else ""
                    recommendations.append({
                        "type": "weight_optimizer",
                        "severity": "high" if conf_score > 0.6 else "medium",
                        "message": (
                            f"🧮 Weight Optimizer (n={wopt['n_samples']}, {wf_status}, "
                            f"confidence {conf_label} {conf_score:.0%}):\n"
                            f"   Changes: {', '.join(change_strs)}{extra}\n"
                            f"   Max delta/cycle: ±{max_weight_delta}"
                            f"{oos_note}"
                        ),
                        "delta_ev": 0.0,
                        "wilson_lo": max(0.0, 0.5 - conf_score * 0.2),
                        "wilson_hi": min(1.0, 0.5 + conf_score * 0.2),
                    })

                oos_test_ran_and_passed = oos_test_ran and oos_weight_ok

                if not changed:
                    recommendations.append({
                        "type": "weight_optimizer",
                        "severity": "low",
                        "message": f"🧮 Weight Optimizer: no significant changes detected (n={wopt['n_samples']}).",
                    })
                elif (wopt.get("walk_forward_passed")
                    and oos_test_ran_and_passed
                    and conf_score >= min_conf):

                    # Shadow is now subordinate: if it vetoes but OOS EV strictly improved,
                    # we still approve but flag the divergence for human review.
                    if shadow_weight_ok:
                        shadow_note = f"Shadow✅ {shadow_weight_note}"
                    else:
                        shadow_note = f"Shadow⚠️ vetoed ({shadow_weight_note}) but OOS EV strictly improved, so approving."

                    config_patch.append({
                        "path": "CONFLUENCE_WEIGHTS",
                        "current": dict(CONFLUENCE_WEIGHTS),
                        "suggested": wopt["suggested_weights"],
                        "reason": (
                            f"Logistic-regression optimal ({wf_status}, conf={conf_score:.2f}). "
                            f"{oos_weight_note}. {shadow_note}"
                        ),
                    })
                else:
                    if not wopt.get("walk_forward_passed"):
                        reason = "walk-forward FAILED"
                    elif len(real_rows) < 200:
                        reason = (
                            f"deployed-population OOS veto requires ≥200 rows "
                            f"(have {len(real_rows)}). Weight changes need more "
                            f"trade history before the Brain will move them."
                        )
                    elif not oos_test_ran_and_passed:
                        reason = (
                            f"OOS EV test did not pass. {oos_weight_note}"
                        )
                    elif conf_score < min_conf:
                        reason = f"confidence too low ({conf_score:.2f} < {min_conf:.2f})"
                    else:
                        reason = "unknown — all gates passed but patch not emitted"
                    recommendations.append({
                        "type": "weight_optimizer_blocked",
                        "severity": "low",
                        "message": (
                            f"🛡️ Weight changes BLOCKED: {reason}. "
                            f"Keeping current weights. "
                            f"Accumulate more data or reduce max_weight_delta."
                        ),
                    })
                if wopt.get("negative_votes"):
                    recommendations.append({
                        "type": "negative_votes",
                        "severity": "medium",
                        "message": (
                            f"⚠️ Harmful votes (negative logistic coefficients): "
                            f"{', '.join(f'{v}({c:+.3f})' for v, c in wopt['negative_votes'][:4])}. "
                            f"Consider disabling or reducing their weights."
                        ),
                    })

            elif wopt.get("error") == "walk_forward_degraded":
                recommendations.append({
                    "type": "weight_optimizer_blocked",
                    "severity": "medium",
                    "message": (
                        f"🛡️ Weight Optimizer REJECTED by walk-forward: "
                        f"holdout WR {wopt.get('holdout_wr', 0):.0%} < "
                        f"baseline {wopt.get('baseline_holdout_wr', 0):.0%}. "
                        f"Current weights are better. No changes applied."
                    ),
                })
        _phase_mark("weight_optimizer")

        # ─ Per-alert breakdown ──────────────────────────────────────────
        alert_stats = engine.per_alert_breakdown(real_rows, min_sample=min_sample)
        if alert_stats:
            display = alert_stats if len(alert_stats) <= 10 else alert_stats[:5] + alert_stats[-5:]
            msg_parts = []
            for idx, (ak, wr, cnt, avg_s) in enumerate(display):
                if len(alert_stats) > 10 and idx == 5:
                    msg_parts.append(f"... ({len(alert_stats) - 10} more) ...")
                flag = " 🔴" if wr < disable_wr else (" 🟢" if wr >= star_wr else "")
                msg_parts.append(f"{ak}: {wr:.0%} WR (n={cnt}, avg score {avg_s:.1f}){flag}")
            recommendations.append({
                "type": "per_alert_breakdown", "severity": "low",
                "data": alert_stats,
                "message": "Per-alert breakdown:\n" + "\n".join(msg_parts),
            })

        _phase_mark("per_alert_breakdown")

        # ── Phase 2: Parameter Autopsy ─────────────────────────────────
        if real_rows and any("context" in r for r in real_rows):
            PARAM_ALERT_MAP = {
                "ppo_adaptive_threshold": ["ppo_adaptive_up", "ppo_adaptive_down"],
                "rsi_adaptive_buy": ["rsi_ema5_up", "rsi_cross_adaptive_up"],
                "rsi_adaptive_sell": ["rsi_ema5_down", "rsi_cross_adaptive_down"],
                "buy_wick_ratio": ["strong_reversal_buy", "hist_rma_buy", "ppohist_buy", "tk_conversion_up", "kijun_cross_up"],
                "sell_wick_ratio": ["strong_reversal_sell", "hist_rma_sell", "ppohist_sell", "tk_conversion_down", "kijun_cross_down"],
            }
            params_higher_worse = {
                "rsi_adaptive_buy": True,
                "rsi_adaptive_sell": False,
                "ppo_adaptive_threshold": True,
                "buy_wick_ratio": True,
                "sell_wick_ratio": True,
            }
            for param, higher_is_worse in params_higher_worse.items():
                if len(real_rows) < self._phase_samples["parameter_autopsy"]:
                    break
                autopsy = engine.parameter_autopsy(
                    real_rows, param,
                    min_sample=self._phase_samples["parameter_autopsy"],
                    higher_is_worse=higher_is_worse,
                )
                if not autopsy.get("valid") or autopsy.get("optimal_cutoff") is None:
                    continue
                last_bucket = autopsy["buckets"][-1]
                if last_bucket["wilson_hi"] < cfg.MIN_WIN_RATE:
                    affected = PARAM_ALERT_MAP.get(param, [])
                    alert_hint = f" (affects: {', '.join(affected[:3])})" if affected else ""
                    recommendations.append({
                        "type": "parameter_autopsy",
                        "severity": "high",
                        "param": param,
                        "message": (
                            f"🎚️ {param}{alert_hint}: trades above {autopsy['optimal_cutoff']:.2f} "
                            f"show {last_bucket['wr']:.0%} WR (n={last_bucket['n']}). "
                            f"Consider tightening to ≤{autopsy['optimal_cutoff']:.2f}."
                        ),
                        "delta_ev": max(0.0, cfg.MIN_WIN_RATE - last_bucket["wr"]),
                        "wilson_lo": last_bucket["wilson_lo"],
                        "wilson_hi": last_bucket["wilson_hi"],
                        # ── FDR: one-sample test against MIN_WIN_RATE ──
                        "n": last_bucket["n"],
                        "wr": last_bucket["wr"],
                    })
                    config_path = None
                    if param == "ppo_adaptive_threshold":
                        config_path = "PPO_ADAPTIVE_VOLATILE" if higher_is_worse else "PPO_ADAPTIVE_CALM"
                    elif param == "rsi_adaptive_buy":
                        config_path = "RSI_ADAPTIVE_BUY_VOLATILE"
                    elif param == "rsi_adaptive_sell":
                        config_path = "RSI_ADAPTIVE_SELL_VOLATILE"
                    if config_path and config_path not in {p["path"] for p in config_patch}:
                        config_patch.append({
                            "path": config_path,
                            "current": getattr(cfg, config_path, None),
                            "suggested": round(autopsy["optimal_cutoff"], 3),
                            "reason": f"Parameter autopsy: WR drops above {autopsy['optimal_cutoff']:.2f}",
                        })

        _phase_mark("parameter_autopsy")

        # ── Phase 3: Conditional Alert Gating ────────────────────────────
        if real_rows and len(real_rows) >= self._phase_samples["conditional_gating"]:
            ak_counts: Dict[str, int] = defaultdict(int)
            for r in real_rows:
                ak_counts[r["alert_key"]] += 1
            top_aks = sorted(ak_counts, key=lambda k: -ak_counts[k])[:5]
            conditions = [("adx_val", 25.0), ("rsi_curr", 50.0), ("buy_wick_ratio", 0.3)]
            for ak in top_aks:
                for cond_field, cond_thr in conditions:
                    cp = conditional_performance(real_rows, ak, cond_field, cond_thr,
                                                 min_sample=self._phase_samples["conditional_gating"])
                    if cp.get("valid") and cp["recommendation"] != "neutral":
                        recommendations.append({
                            "type": "conditional_gating",
                            "severity": "medium",
                            "message": (
                                f"🔀 {ak} under {cond_field}: "
                                f"{cp['above']['wr']:.0%} when >{cond_thr} vs "
                                f"{cp['below']['wr']:.0%} when ≤{cond_thr}. "
                                f"→ {cp['recommendation']}."
                            ),
                            "delta_ev": abs(cp["gap"]),
                            "wilson_lo": min(cp["above"]["wilson_lo"], cp["below"]["wilson_lo"]),
                            "wilson_hi": max(cp["above"]["wilson_hi"], cp["below"]["wilson_hi"]),
                            # ── FDR: two-proportion test, above vs below ──
                            "above_n": cp["above"]["n"],
                            "above_wr": cp["above"]["wr"],
                            "below_n": cp["below"]["n"],
                            "below_wr": cp["below"]["wr"],
                        })

        _phase_mark("conditional_gating")

        # ── Phase 4: Vote Interaction Miner ──────────────────────────────
        if len(real_rows) >= self._phase_samples["vote_interactions"]:
            interactions = interaction_miner(real_rows, min_sample=self._phase_samples["vote_interactions"])
            for inter in interactions[:5]:
                v1, v2 = inter["pair"]
                if inter["type"] == "synergy":
                    # wr_only_v2 may be absent when the v2-alone arm was too
                    # thin to clear min_sample — the corrected miner drops
                    # the key entirely rather than fabricating a 0.0. Format
                    # the message defensively so a missing key doesn't crash
                    # the report.
                    _v2_alone_str = (
                        f", {v2}={inter['wr_only_v2']:.0%}"
                        if "wr_only_v2" in inter else ""
                    )
                    recommendations.append({
                        "type": "vote_interaction", "kind": "synergy", "severity": "low",
                        "message": (
                            f"🔗 Synergy: {v1}+{v2} = {inter['wr_both']:.0%} WR "
                            f"(n={inter['n_both']}). Alone: {v1}={inter['wr_only_v1']:.0%}"
                            f"{_v2_alone_str}."
                        ),
                        "delta_ev": abs(inter["delta"]),
                        # ── FDR: p_value stamped by the miner directly ──
                        "p_value": inter.get("p_value"),
                    })
                else:
                    poisoner, victim = inter["poisoner"], inter["victim"]
                    wr_victim_alone = inter["wr_only_v1"] if victim == v1 else inter["wr_only_v2"]
                    recommendations.append({
                        "type": "vote_interaction", "kind": "poison", "severity": "medium",
                        "message": (
                            f"Poison: {poisoner} kills {victim}. "
                            f"Together={inter['wr_both']:.0%}, {victim} alone={wr_victim_alone:.0%}."
                        ),
                        "delta_ev": abs(inter["delta"]),
                        # ── FDR: p_value stamped by the miner directly ──
                        "p_value": inter.get("p_value"),
                    })

        _phase_mark("vote_interaction_miner")

        # ─ Phase 5: Counterfactual Simulator (shadow-validated) ──────
        baseline_ev = ai_metrics.get("net_ev") or 0.0
        min_cf = self._phase_samples["counterfactual"]
        if real_rows and len(real_rows) >= min_cf:
            shadow_usable = len(shadow_rows) >= min_cf
            shadow_baseline_ev = 0.0
            if shadow_usable:
                shadow_baseline_ev, _hk, _swr = engine.ev_and_kelly_for(shadow_rows)

            rsi_cap_ok = any(
                "context" in r and r["context"].get("rsi_adaptive_buy") for r in real_rows
            )
            rsi_cap = getattr(cfg, "RSI_ADAPTIVE_BUY_VOLATILE", 70.0) - 3
            specs: List[Dict[str, Any]] = [
                {"label": f"Threshold +1 ({cfg.CONFLUENCE_MIN_ABS_SCORE + 1.0})",
                 "new_threshold": cfg.CONFLUENCE_MIN_ABS_SCORE + 1.0, "new_params": None},
            ]
            if rsi_cap_ok:
                specs.append({"label": "RSI buy cap -3",
                              "new_threshold": None, "new_params": {"rsi_curr": rsi_cap}})
                specs.append({"label": "Threshold +1 + RSI cap -3",
                              "new_threshold": cfg.CONFLUENCE_MIN_ABS_SCORE + 1.0,
                              "new_params": {"rsi_curr": rsi_cap}})

            scenarios: List[Dict[str, Any]] = []
            for spec in specs:
                real_sim = simulate_config_change(
                    real_rows, baseline_ev,
                    new_threshold=spec["new_threshold"], new_params=spec["new_params"],
                )
                if not real_sim:
                    continue
                scenario = {"label": spec["label"], **real_sim}
                # ── Shadow out-of-sample check: same change, second sample ──
                if shadow_usable:
                    shadow_sim = simulate_config_change(
                        shadow_rows, shadow_baseline_ev,
                        new_threshold=spec["new_threshold"], new_params=spec["new_params"],
                    )
                    if shadow_sim and shadow_sim["n"] >= 5:
                        scenario["shadow_n"] = shadow_sim["n"]
                        scenario["shadow_delta_ev"] = shadow_sim["delta_ev"]
                        scenario["shadow_wr"] = shadow_sim["wr"]
                        # Agreement test: two samples from the same window
                        # must point the same way, or the "improvement" is
                        # one split, one distribution, one luck draw.
                        scenario["shadow_validated"] = (
                            (real_sim["delta_ev"] >= 0) == (shadow_sim["delta_ev"] >= 0)
                        )
                    else:
                        scenario["shadow_validated"] = None
                else:
                    scenario["shadow_validated"] = None
                scenarios.append(scenario)

            if scenarios:
                best = max(scenarios, key=lambda x: x["ev"])
                sv = best.get("shadow_validated")
                if sv is True:
                    shadow_note = (
                        f"\n   Shadow-confirmed: Δ{best['shadow_delta_ev']:+.3f}% "
                        f"on {best['shadow_n']} rejected-path samples."
                    )
                elif sv is False:
                    shadow_note = (
                        f"\n   ⚠️ Shadow DISAGREES: Δ{best['shadow_delta_ev']:+.3f}% "
                        f"on {best['shadow_n']} samples — treat as curve-fit."
                    )
                else:
                    shadow_note = "\n   Shadow sample too thin to validate."
                recommendations.append({
                    "type": "counterfactual",
                    # "high" now REQUIRES the out-of-sample shadow check to
                    # agree — real-data-only wins stay medium/low.
                    "severity": (
                        "high" if best["delta_ev"] > 0.05 and sv is True
                        else "medium" if best["delta_ev"] > 0.05 and sv is None
                        else "low"
                    ),
                    "shadow_validated": sv,
                    "message": (
                        f"🔮 Best scenario: '{best['label']}' → "
                        f"EV {best['ev']:+.3f}%/trade (Δ{best['delta_ev']:+.3f}%), "
                        f"WR {best['wr']:.0%}, n={best['n']}.{shadow_note}"
                    ),
                    "delta_ev": best["delta_ev"],
                })
                ai_metrics["counterfactual_scenarios"] = scenarios

        _phase_mark("counterfactual")

        # ── Phase 6: Regime Profiles (shadow-validated) ───────────────
        if len(real_rows) >= self._phase_samples["regime_profiles"]:
            rpo = regime_profile_optimizer(
                real_rows, regime_field="adx_val",
                min_sample=self._phase_samples["regime_profiles"],
            )
            rpo_shadow = None
            if len(shadow_rows) >= self._phase_samples["regime_profiles"]:
                rpo_shadow = regime_profile_optimizer(
                    shadow_rows, regime_field="adx_val",
                    min_sample=self._phase_samples["regime_profiles"],
                )
            if rpo.get("valid") and len(rpo.get("regimes", [])) >= 2:
                lines = []
                for reg in rpo["regimes"]:
                    sreg = self._match_shadow_regime(rpo_shadow, reg["range"])
                    if sreg is not None:
                        gap = abs(sreg["recommended_threshold"] - reg["recommended_threshold"])
                        tag = (
                            f"shadow✅ thr={sreg['recommended_threshold']:.1f}"
                            if gap <= 3.0 else
                            f"shadow⚠️ thr diverges {gap:.1f}pts"
                        ) + f" (n={sreg['n']})"
                    elif rpo_shadow is None:
                        tag = "shadow: insufficient data"
                    else:
                        tag = "shadow: no overlapping regime"
                    lines.append(
                        f"  Regime {reg['regime_id']} (ADX {reg['range'][0]}-{reg['range'][1]}): "
                        f"thr={reg['recommended_threshold']:.1f}, WR={reg['wr']:.0%} | {tag}"
                    )
                recommendations.append({
                    "type": "dynamic_regime_profile", "severity": "low",
                    "message": "📊 Regime thresholds:\n" + "\n".join(lines),
                })

        _phase_mark("regime_profiles")

        # ── Config Version Regression ────────────────────────────────────
        version_comparisons = compare_config_versions(
            real_rows, min_sample=self._phase_samples["config_regression"]
        )
        for comp in version_comparisons:
            _comp_shared = {
                "prev_version": comp["prev_version"],
                "cur_version": comp["cur_version"],
                "prev_n": comp["prev_n"],
                "cur_n": comp["cur_n"],
                "prev_wr": comp["prev_wr"],
                "cur_wr": comp["cur_wr"],
            }
            if comp["regression"]:
                rec_entry = {
                    "type": "config_regression", "severity": "high",
                    "message": (
                        f"🚨 Config regression: WR {comp['prev_wr']:.0%}→{comp['cur_wr']:.0%} "
                        f"({comp['prev_version']}→{comp['cur_version']}). Consider reverting."
                    ),
                    "delta_ev": abs(comp["delta_wr"]),
                }
                rec_entry.update(_comp_shared)
                recommendations.append(rec_entry)
            elif comp["improvement"]:
                rec_entry = {
                    "type": "config_improvement", "severity": "low",
                    "message": (
                        f"✅ Config improved WR: {comp['prev_wr']:.0%}→{comp['cur_wr']:.0%} "
                        f"({comp['prev_version']}→{comp['cur_version']})."
                    ),
                }
                rec_entry.update(_comp_shared)
                recommendations.append(rec_entry)

        ai_metrics["config_comparisons"] = version_comparisons

        _phase_mark("config_version_regression")

        # ── AI/ML: OOS Permutation Importance (EV-based, walk-forward) ────
        # FIX: honor cfg.BRAIN_PERMUTATION_IMPORTANCE — previously this
        # ran whenever sample size was sufficient, regardless of the flag.
        _perm_enabled = (
            getattr(cfg, "BRAIN_PERMUTATION_IMPORTANCE", True)
            and len(real_rows) >= min_sample * 3
        )
        _perm_ok, _perm_why = audit.can_run("permutation_importance")
        if _perm_enabled and not _perm_ok:
            audit.record_analysis(
                "permutation_importance", HealthStatus.INSUFFICIENT_DATA,
                detail=_perm_why,
            )
        if _perm_enabled and _perm_ok:
            _perm_n = 15
            perm_imp = engine.oos_permutation_importance(
                real_rows, min_sample=min_sample, n_permutations=_perm_n
            )
            if perm_imp:
                top_positive = [p for p in perm_imp if p["direction"] == "positive"][:3]
                top_negative = [p for p in perm_imp if p["direction"] == "negative"][:3]
                parts = []
                if top_positive:
                    parts.append("most impactful: " + ", ".join(
                        f"{p['feature']}({p['importance_ev']:+.4f})" for p in top_positive))
                if top_negative:
                    parts.append("harmful: " + ", ".join(
                        f"{p['feature']}({p['importance_ev']:+.4f})" for p in top_negative))

                # FDR tests the single strongest signal. If the top signal
                _top = (top_positive + top_negative)[:1]
                _top_rec = _top[0] if _top else None

                _rec: Dict[str, Any] = {
                    "type": "permutation_importance", "severity": "low",
                    "message": f"🤖 OOS permutation importance (net EV) — {'; '.join(parts)}",
                }
                if _top_rec is not None:
                    _rec["top_vote"] = _top_rec["feature"]
                    _rec["top_importance"] = _top_rec["importance_ev"]
                    _rec["top_std"] = _top_rec.get("std", 0.0)
                    _rec["n_permutations"] = _perm_n
                recommendations.append(_rec)

        _phase_mark("permutation_importance")

# ── Benjamini-Hochberg FDR correction ────────────────────────
        _fdr_t0 = time.time()
        p_val_indices: List[int] = []
        p_vals: List[float] = []
        for idx, r in enumerate(recommendations):
            p = _extract_p_value_for_fdr(r)
            if p is not None:
                p_val_indices.append(idx)
                p_vals.append(p)  
        if p_vals:
            keep_mask = engine.benjamini_hochberg(p_vals, alpha=0.10)
            n_survived = sum(keep_mask)
            n_tested = len(p_vals)
            for fdr_flag, idx in zip(keep_mask, p_val_indices):
                recommendations[idx]["fdr_passed"] = bool(fdr_flag)
            for fdr_flag, idx in zip(keep_mask, p_val_indices):
                if not fdr_flag and recommendations[idx]["severity"] in ("high", "medium"):
                    recommendations[idx]["severity"] = "low"
                    recommendations[idx]["message"] = (
                        f"{recommendations[idx]['message']}\n"
                        f"[FDR: not significant after BH correction across "
                        f"{n_tested} tests at α=0.10]"
                    )
            if n_tested > 3:
                recommendations.append({
                    "type": "fdr_summary",
                    "severity": "low",
                    "message": (
                        f"🔬 FDR (Benjamini-Hochberg, α=0.10): {n_survived}/{n_tested} "
                        f"statistical claims survived correction across the report. "
                        f"Surviving claims are marked `fdr_passed=True`; demoted "
                        f"claims are downgraded to low severity."
                    ),
                })
        logger.info(f"⏱️   └ fdr_block: {time.time() - _fdr_t0:.2f}s")

        # ─ Actionability scoring (blended with empirical repair outcomes) ──
        _act_t0 = time.time()
        for rec in recommendations:
            rec["actionability_score"] = round(
                learned_actionability(rec, self._repair_success_rates), 3
            )
            if rec.get("_repair_id"):
                cat = rec.get("category") or rec.get("type")
                if cat:
                    stats = (self._repair_success_rates or {}).get(cat, {})
                    if stats.get("n", 0) >= 8:
                        rec["_empirical_help_rate"] = round(stats["help_rate"], 3)

        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        recommendations.sort(key=lambda x: (
            severity_order.get(x.get("severity", ""), 4),
            -x.get("actionability_score", 0),
        ))
    
        logger.info(f"⏱️   └ actionability_block: {time.time() - _act_t0:.2f}s")

        # ── Config version hash ──────────────��───────────────────────────
        _hash_t0 = time.time()
        ai_metrics["config_version"] = hash_config_state(
            CONFLUENCE_WEIGHTS, cfg.CONFLUENCE_MIN_ABS_SCORE, cfg.CONFLUENCE_MIN_PCT
        )
        await self._remember_config_version(ai_metrics["config_version"])
        logger.info(f"⏱️   └ hash_and_remember: {time.time() - _hash_t0:.2f}s")

        # ── Bonus-aware metrics ───────────────────────────────�����──────────
        if real_rows:
            bonus_count = sum(1 for r in real_rows if r.get("bonus_win"))
            total_wins = sum(1 for r in real_rows if r["win"])
            rr_vals = [r.get("rr_achieved", 0) for r in real_rows if r.get("rr_achieved", 0) > 0]
            ai_metrics["bonus_wins"] = bonus_count
            ai_metrics["bonus_rate_of_wins"] = bonus_count / max(total_wins, 1)
            ai_metrics["avg_rr_achieved"] = round(sum(rr_vals) / len(rr_vals), 2) if rr_vals else 0.0
            total_win_weight = sum(r.get("win_weight", 1.0 if r["win"] else 0.0) for r in real_rows)
            ai_metrics["weighted_wr"] = round(min(total_win_weight / len(real_rows), 1.0), 4) if real_rows else 0.0

        if shadow_rows:
            ai_metrics["shadow_win_rate"] = round(
                sum(1 for r in shadow_rows if r["win"]) / len(shadow_rows), 4
            )

        # ─ Action gate: suppress config patches unless evidence is strong ──      
        _cusum_read_failed = False
        _active_drift_keys: List[str] = []
        _below_floor_drift: List[Tuple[str, int]] = []

        if getattr(cfg, "BRAIN_ACTION_GATE_ENABLED", True):
            _cusum_min_n = int(getattr(cfg, "BRAIN_CUSUM_MIN_SAMPLE", 30))
            try:
                _seen_aks = {r["alert_key"] for r in real_rows}
                # Detectors were loaded + updated by _check_cusum_drift, so
                # reuse them; bulk-read (one round-trip) only what is missing.
                _states: Dict[str, Dict[str, Any]] = {
                    ak: self._cusum_detectors[ak].to_dict()
                    for ak in _seen_aks if ak in self._cusum_detectors
                }
                _missing = [ak for ak in _seen_aks if ak not in _states]
                if _missing:
                    for _ak, (_wm, _st) in ((await self.sdb.load_cusum_bulk(_missing)) or {}).items():
                        if _st:
                            _states[_ak] = _st
                for _ak, _cusum_state in _states.items():
                    if _cusum_state.get("s_neg", 0.0) <= _cusum_state.get("h", 2.0):

                        continue
                    _n_seen = int(_cusum_state.get("n", 0))
                    if _n_seen >= _cusum_min_n:
                        _active_drift_keys.append(_ak)
                    else:
                        _below_floor_drift.append((_ak, _n_seen))
                if _active_drift_keys:
                    logger.info(
                        f"Action gate: {len(_active_drift_keys)} persisted CUSUM "
                        f"alarm(s) active with n>={_cusum_min_n} — stability layer "
                        f"forced False "
                        f"({_active_drift_keys[:5]}{'…' if len(_active_drift_keys) > 5 else ''})"
                    )
                if _below_floor_drift:
                    logger.info(
                        f"Action gate: {len(_below_floor_drift)} CUSUM alarm(s) "
                        f"seen but below BRAIN_CUSUM_MIN_SAMPLE={_cusum_min_n} — "
                        f"not vetoing tuning "
                        f"({[f'{ak}(n={n})' for ak, n in _below_floor_drift[:5]]}"
                        f"{'' if len(_below_floor_drift) > 5 else ''})"
                    )

            except Exception as e:
                _cusum_read_failed = True
                audit.record_analysis_exception("cusum_gate_read", e)

        if getattr(cfg, "BRAIN_ACTION_GATE_ENABLED", True):
            action_gate = self._action_gate_check(
                real_rows, min_sample=min_sample,
                recommendations=recommendations,
                active_drift_keys=_active_drift_keys or None,
            )
        else:
            action_gate = {"actionable": True, "disabled": True}
        if _cusum_read_failed and not action_gate.get("disabled"):
            action_gate["stability"] = False
            action_gate["actionable"] = False
        ai_metrics["action_gate"] = action_gate
        if not action_gate.get("actionable", False):
            # Downgrade all config patches to informational
            for patch in config_patch:
                patch["_blocked_by_action_gate"] = True
                patch["reason"] = (
                    f"[GATE BLOCKED] {patch.get('reason', '')} "
                    f"— OOS EV evidence insufficient"
                )
            # Only RISK-INCREASING auto-actions (re-enable) need this gate.
            # A disable is protective and stands on its own per-alert evidence
            # (upper CI bound below the disable threshold, net EV negative,
            # minimum sample). Gating it on portfolio-wide EV/drawdown/drift
            # would stop the Brain from switching off a losing alert exactly
            # when the system as a whole is doing worst.
            for rec in recommendations:
                if rec.get("pending_auto_action") and rec.get("pending_action") != "disable":
                    rec["pending_auto_action"] = False
                    rec["message"] += " [BLOCKED by action gate]"
            logger.info(
                f"🚫 Action gate BLOCKED config patches and re-enables "
                f"(protective disables still proceed): {action_gate}"
            )

        # ── Execute deferred auto-actions. 'disable' always gets here; 'enable'
        # only survives the block above when the gate passed. ──
        for rec in recommendations:
            if not rec.get("pending_auto_action"):
                continue
            ak = rec.get("alert")
            action = rec.get("pending_action")
            if not ak or not action:
                continue
            try:
                if action == "disable":
                    ok = await self.sdb.set_alert_key_disabled(ak, True)
                    if ok:
                        rec["message"] = rec["message"].replace(
                            "[Pending action gate]", "[APPLIED]"
                        )
                        logger.info(f"🔒 Post-gate auto-disabled: {ak}")
                elif action == "enable":
                    ok = await self.sdb.set_alert_key_disabled(ak, False)
                    if ok:
                        rec["message"] = rec["message"].replace(
                            "[Pending action gate]", "[APPLIED]"
                        )
                        logger.info(f"🔓 Post-gate auto-re-enabled: {ak}")
            except Exception as e:
                logger.warning(f"Post-gate auto-action failed for {ak}: {e}")
            # Mark as consumed regardless of success
            rec["pending_auto_action"] = False

        _phase_mark("fdr_and_actionability")

        # ── Attach audit to ai_metrics for persistence ──
        ai_metrics["brain_audit"] = audit.to_dict()
        ai_metrics["data_quality_header"] = audit.build_data_quality_header()

        # ── Attach archive stats for the profit action plan ──
        result = dict(base_recs)
        result["recommendations"] = recommendations
        result["recommendation_count"] = len(recommendations)
        result["config_patch"] = config_patch
        result["ai_metrics"] = ai_metrics
        result["_archive_stats"] = audit._archive_stats or {}
        _phase_mark("audit_attach")
        return result

    # ── Baseline wrapper that also exposes raw rows ─────────────────────

    async def _generate_baseline_recommendations(self) -> Dict[str, Any]:
        # Fresh audit BEFORE any row loading, so reconciliation data and
        # analysis-health records made during the baseline survive to the report.
        audit = reset_audit()
        # Load rows ONCE, attach them, and hand them to the baseline via the
        # subclass hook so the parent doesn't read the archive a second time.
        real_rows, shadow_rows = await self._get_rows()
        # Coverage is measured on the LONG window (not the 30-day analysis
        # window) and must be set BEFORE the baseline runs, so audit.can_run()
        # sees real row counts.
        long_days = getattr(cfg, "BRAIN_LONG_WINDOW_DAYS", 180)
        try:
            _recent, _medium, long_rows = await self._get_layered_window_rows()
        except Exception as e:
            audit.record_analysis_exception("history_coverage", e)
            long_rows = real_rows
        audit.set_history_coverage(
            long_rows or real_rows,
            requested_days=long_days,
            analysis_rows=real_rows,
        )
        audit.set_shadow_count(len(shadow_rows))
        base = await super().generate_recommendations()
        base["_real_rows"] = real_rows
        base["_shadow_rows"] = shadow_rows
        return base

    async def _store_pending_plan(self, recs: Dict[str, Any]) -> None:
        """Store the current recommendations for later application."""
        try:
            # Extract actionable items
            config_patches = []
            disable_alerts = []
            reinstate_alerts = []
            weight_adjustments = []

            # FIX (Priority 3): the action gate is authoritative. `_blocked_by_action_gate`
            # is only ever set on config_patch dicts, never on recommendations — so the
            # old check here was a no-op. Read the gate verdict directly.
            action_gate_passed = bool(
                recs.get("ai_metrics", {})
                    .get("action_gate", {})
                    .get("actionable", False)
            )

            for rec in recs.get("recommendations", []):
                rec_type = rec.get("type")
                
                # FIX: Skip any auto-actions that were already handled (applied or blocked) 
                # by the post-gate executor in _generate_recommendations_full.
                # The executor modifies the message to include [APPLIED] or [BLOCKED].
                msg = rec.get("message", "")
                if "[APPLIED]" in msg or "[BLOCKED" in msg:
                    continue

                if rec_type == "disable_alert":
                    if not action_gate_passed:
                        continue
                    ak = rec.get("alert") or rec.get("alert_key")
                    if ak:
                        disable_alerts.append(ak)

                elif rec_type in ("reinstate_alert", "recovered_alert", "auto_reenabled"):
                    if not action_gate_passed:
                        continue
                    ak = rec.get("alert") or rec.get("alert_key")
                    if ak:
                        reinstate_alerts.append(ak)

                elif rec_type == "repair_shop" and rec.get("category") == "root_cause":
                    # FIX (Issue 4): Root-cause weight adjustments must also pass 
                    # the global action gate, exactly like disable/reinstate alerts.
                    if not action_gate_passed:
                        continue
                        
                    # Wiring #1: turn the root-cause segment into an
                    # actionable weight reduction instead of leaving it prose.
                    adj = self._root_cause_to_weight_adjustment(rec)
                    if adj:
                        weight_adjustments.append(adj)
            
                elif rec_type == "repair_shop" and rec.get("category") == "threshold_too_low":
                    if not action_gate_passed:
                        continue
                    field = rec.get("config_field")
                    suggested = rec.get("config_suggested")
                    if field in CONFIG_OVERRIDE_ALLOWED_FIELDS and suggested is not None:
                        config_patches.append({
                            "path": field,
                            "current": rec.get("config_current"),
                            "suggested": suggested,
                            "reason": rec.get("message", ""),
                            "_source_category": "threshold_too_low",
                        })

            # Only store safe config patches
            for patch in recs.get("config_patch", []):
                # Action gate is authoritative — a patch marked blocked must
                # not enter the pending plan, regardless of field safelisting.
                if patch.get("_blocked_by_action_gate"):
                    continue
                field = patch.get("path")
                suggested = patch.get("suggested")
                if field in CONFIG_OVERRIDE_ALLOWED_FIELDS and suggested is not None:
                    config_patches.append({
                        "path": field,
                        "current": patch.get("current"),
                        "suggested": suggested,
                        "reason": patch.get("reason", ""),
                        # Wiring #2: tag with a ledger category so selection
                        # can sample that category's learned help-rate.
                        "_source_category": (
                            patch.get("_source_category")
                            or self._infer_patch_category(field)
                        ),
                    })

            # ── FIX: dedupe by field. Both repair_shop_diagnosis()
            def _delta(p: Dict[str, Any]) -> float:
                cur, sug = p.get("current"), p.get("suggested")
                if isinstance(cur, (int, float)) and isinstance(sug, (int, float)):
                    return abs(float(sug) - float(cur))
                return 0.0

            def _pref(p: Dict[str, Any]) -> tuple:
                # (has_source_category, |delta|) — lexicographic, higher wins
                return (1 if p.get("_source_category") else 0, _delta(p))

            deduped: Dict[str, Dict[str, Any]] = {}
            for p in config_patches:
                existing = deduped.get(p["path"])
                if existing is None or _pref(p) > _pref(existing):
                    deduped[p["path"]] = p
            config_patches = list(deduped.values())

            # ── Optional budget: cap total entries per plan ──
            budget = getattr(cfg, "BRAIN_MAX_PLAN_ENTRIES", 0)
            if budget > 0:
                total = (len(config_patches) + len(disable_alerts)
                         + len(reinstate_alerts) + len(weight_adjustments))
                if total > budget:
                    # Wiring #2: Thompson-sample the top-`budget`, but seed
                    # each draw with the category's learned P(helps) instead
                    # of an uninformed uniform prior.
                    rng = random.Random(int(time.time() // 3600))
                    candidates = (
                        [("config", p, p.get("_source_category")) for p in config_patches]
                        + [("weight", w, w.get("category")) for w in weight_adjustments]
                        + [("disable", ak, None) for ak in disable_alerts]
                        + [("reinstate", ak, None) for ak in reinstate_alerts]
                    )
                    scored = []
                    for kind, item, category in candidates:
                        a, b = self._category_beta_prior(category)
                        sampled = rng.betavariate(a, b)
                        scored.append((sampled, kind, item))
                    scored.sort(key=lambda t: -t[0])
                    keep = scored[:budget]
                    config_patches = [item for _, k, item in keep if k == "config"]
                    weight_adjustments = [item for _, k, item in keep if k == "weight"]
                    disable_alerts = [item for _, k, item in keep if k == "disable"]
                    reinstate_alerts = [item for _, k, item in keep if k == "reinstate"]

            plan_data = {
                "generated_at": int(time.time()),
                "_action_gate_passed": action_gate_passed,
                "config_patch": config_patches,
                "disable_alerts": disable_alerts,
                "reinstate_alerts": reinstate_alerts,
                "weight_adjustments": weight_adjustments,
            }
            await self.sdb.set_metadata(
                "brain_pending_plan",
                json_dumps(plan_data),
                ttl=7 * 86400  # 7 days
            )
        except Exception as e:
            logging.getLogger("macd_bot").warning(f"Failed to store pending plan: {e}")

    @staticmethod
    def _infer_patch_category(field: str) -> Optional[str]:
        """Map a config patch to the repair-ledger category that would have
        produced it, so selection can use that category's track record.
        Returns None when there is no clean ledger category — those keep a
        uniform prior rather than borrowing an unrelated one."""
        if field in ("CONFLUENCE_MIN_ABS_SCORE", "CONFLUENCE_MIN_PCT"):
            return "threshold_too_low"
        return None

    def _category_beta_prior(self, category: Optional[str]) -> tuple:
        """Beta(a, b) prior for Thompson selection, centred on the learned
        P(repair helps) for this category. Falls back to the empirical
        ledger help-rate, then to an uninformative Beta(1,1) for anything
        new — a never-tried action is never penalised."""
        if not category:
            return 1.0, 1.0
        # Prefer the contextual learned model, fall back to the ledger rate.
        p_help = self._repair_help_preds.get(category)
        if p_help is None:
            stats = self._repair_success_rates.get(category)
            if stats and stats.get("n", 0) >= 3:
                p_help = stats.get("help_rate")
        if p_help is None:
            return 1.0, 1.0
        p_help = max(0.0, min(1.0, p_help))      
        # Reduce strength until we have enough repair history
        stats = self._repair_success_rates.get(category)
        n_resolved = stats.get("n", 0) if stats else 0
        strength = 2.0 if n_resolved >= 100 else 0.5
        return 1.0 + strength * p_help, 1.0 + strength * (1.0 - p_help)

    def _root_cause_to_weight_adjustment(self, rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert a root_cause segment into a concrete weight reduction.
        Only actionable when the toxic condition is a vote being ON — the
        one case addressable through the existing dynamic-weights mechanism.
        Context-threshold segments stay prose until a matching override
        exists."""
        scope = rec.get("scope") or {}
        if scope.get("kind") != "segment":
            return None
        val = scope.get("value") or {}
        feature = val.get("feature") or ""
        op = val.get("op")
        threshold = val.get("threshold")
        if not feature.startswith("vote:") or threshold is None:
            return None
        if op != ">" or not (0.0 <= threshold < 1.0):
            return None

        vote = feature.split(":", 1)[1]
        current_w = CONFLUENCE_WEIGHTS.get(vote)
        if current_w is None or current_w <= 0:
            return None
        return {
            "vote": vote,
            "current": current_w,
            "suggested": round(current_w * 0.5, 2),  # halve, never zero
            "category": "root_cause",
            "reason": f"root_cause segment {feature} {op} {threshold}",
        }

    async def _remember_config_version(self, version_hash: str) -> None:
        """Persist the overridable config values under this version hash, so
        a later change-point regression can be reverted to real values — not
        just pointed at a hash."""
        try:
            raw = await self.sdb.get_metadata("brain_config_version_snapshots")
            snapshots = {}
            if raw:
                try:
                    snapshots = json_loads(raw)
                except Exception:
                    snapshots = {}
            if version_hash in snapshots:
                return
            snapshot = {
                field: getattr(cfg, field, None)
                for field in CONFIG_OVERRIDE_ALLOWED_FIELDS
            }
            snapshot["_seen_at"] = int(time.time())
            snapshots[version_hash] = snapshot
            # keep the map bounded
            if len(snapshots) > 50:
                ordered = sorted(
                    snapshots.items(),
                    key=lambda kv: kv[1].get("_seen_at", 0),
                    reverse=True,
                )[:50]
                snapshots = dict(ordered)
            await self.sdb.set_metadata(
                "brain_config_version_snapshots",
                json_dumps(snapshots),
                ttl=90 * 86400,
            )
        except Exception as e:
            logging.getLogger("macd_bot").debug(
                f"Config snapshot store failed (non-fatal): {e}"
            )

    async def _lookup_config_version(self, version_hash: Optional[str]) -> Optional[Dict[str, Any]]:
        if not version_hash:
            return None
        try:
            raw = await self.sdb.get_metadata("brain_config_version_snapshots")
            if not raw:
                return None
            snapshots = json_loads(raw)
            return snapshots.get(version_hash)
        except Exception:
            return None

    async def apply_pending_plan(self, telegram_queue, logger_run) -> bool:
        """Apply the last generated action plan. Returns True if any changes were applied."""
        try:
            raw = await self.sdb.get_metadata("brain_pending_plan")
            if not raw:
                await telegram_queue.send(escape_markdown_v2(
                    "⚠️ No pending brain plan found\\.\n"
                    "Run a report first with `BRAIN_REPORT_ON_DEMAND=true`\\."
                ))
                return False
            
            plan = json_loads(raw)
            applied = []
            applied_config = False   # config overrides — loaded only at startup
            applied_live = False     # dynamic weights / alert flags — read live

            plan_gate_passed = bool(plan.get("_action_gate_passed", False))

            # Apply config changes
            for patch in plan.get("config_patch", []):
                if not plan_gate_passed:
                    logger_run.warning(
                        f"Skipping config patch (plan did not pass action gate): {patch.get('path')}"
                    )
                    continue

                # Defense in depth: even if a blocked patch somehow reached
                # the stored plan (older plan, mid-upgrade race), never apply it.
                if patch.get("_blocked_by_action_gate"):
                    logger_run.warning(
                        f"Skipping blocked patch (action gate): {patch.get('path')}"
                    )
                    continue
                field = patch.get("path")
                value = patch.get("suggested")
                if field in CONFIG_OVERRIDE_ALLOWED_FIELDS:
                    ok = await self.sdb.write_config_override(field, value)
                    if ok:
                        applied.append(f"✅ {field}: {value}")
                        applied_config = True
                        logger_run.info(f"Applied brain config: {field} = {value}")

            if plan_gate_passed:
                for ak in plan.get("disable_alerts", []):
                    ok = await self.sdb.set_alert_key_disabled(ak, True)
                    if ok:
                        applied.append(f"🔴 Disabled: {ak}")
                        applied_live = True
                        logger_run.info(f"Applied brain disable: {ak}")
                for ak in plan.get("reinstate_alerts", []):
                    ok = await self.sdb.set_alert_key_disabled(ak, False)
                    if ok:
                        applied.append(f"🟢 Reinstated: {ak}")
                        applied_live = True
                        logger_run.info(f"Applied brain reinstate: {ak}")
            else:
                logger_run.warning(
                    "Skipping disable/reinstate entries — plan did not pass action gate"
                )

            # Apply root-cause weight adjustments via dynamic weights
            weight_adj = plan.get("weight_adjustments", [])
            if weight_adj and plan_gate_passed:
                weights = dict(CONFLUENCE_WEIGHTS)
                for adj in weight_adj:
                    vote = adj.get("vote")
                    suggested = adj.get("suggested")
                    if vote in weights and suggested is not None:
                        weights[vote] = suggested
                if await self.sdb.set_dynamic_weights(weights):
                    applied_live = True
                    for adj in weight_adj:
                        applied.append(
                            f"⚖️ {adj.get('vote')}: "
                            f"{adj.get('current')} → {adj.get('suggested')}"
                        )
                    logger_run.info(
                        "Applied brain root-cause weight cuts: "
                        f"{[a.get('vote') for a in weight_adj]}"
                    )
            elif weight_adj and not plan_gate_passed:
                logger_run.warning(
                    "Skipping root-cause weight adjustments — plan did not pass action gate"
                )         

            if applied:
                # ── Ledger: mark all repairs in this plan as applied ──
                plan_ts = plan.get("generated_at", int(time.time()))
                try:
                    marked = await mark_plan_applied(self.sdb, plan_ts)
                    if marked:
                        logger_run.info(f"📒 Repair ledger: marked {marked} repair(s) applied")
                except Exception as e:
                    logger_run.debug(f"Repair ledger apply-mark failed (non-fatal): {e}")

                trailer = ""
                if applied_config:
                    trailer += "\nConfig overrides require a restart\\."
                if applied_live:
                    trailer += "\nWeight/alert changes take effect on the next dispatch\\."

                msg = (
                    f"✅ APPLIED BRAIN PLAN\n"
                    f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    + "\n".join(applied)
                    + trailer
                )
                await telegram_queue.send(escape_markdown_v2(msg))
                # Clear the pending plan
                await self.sdb.set_metadata("brain_pending_plan", "{}", ttl=60)
                return True
            else:
                await telegram_queue.send(escape_markdown_v2(
                    "⚠️ No applicable changes in the plan\\.\n"
                    "All recommended changes may already be applied\\."
                ))
                return False
                
        except Exception as e:
            logger_run.error(f"Apply plan failed: {e}")
            await telegram_queue.send(escape_markdown_v2(
                f"❌ Failed to apply brain plan: {str(e)[:100]}"
            ))
            return False

    @staticmethod
    def _archive_report(sections: List[List[_Piece]], stamp: str, logger_run: logging.Logger) -> None:
        """Save this report as Markdown in <OUTCOME_DATA_DIR>/reports/
        (YYYY-MM-DD_HH-MM.md, UTC). Never fatal: a failed archive must not
        stop the Telegram report or trigger the fallback report."""
        try:
            from outcome_storage import save_report
            path = save_report(render_report_markdown(sections, stamp))
            logger_run.info(f"Brain report archived: {path}")
        except Exception as e:
            logger_run.warning(f"Brain report archive failed (non-fatal): {e}")

    async def generate_report(self, pairs, telegram_queue, logger_run) -> bool:
        """Override: send ONLY the plain-English action plan (no jargon), then store for application."""
        try:
            recs = await self.generate_recommendations()

            # Build and send the plain-English action plan
            # Layered 16-section report. If building it raises, the outer
            # handler below falls back to the base technical report.
            sections, stamp = build_brain_report_sections(recs, cfg)
            plan_messages = render_report_messages(sections, stamp)   # already MarkdownV2-escaped
            self._archive_report(sections, stamp, logger_run)
            sent_ok = True
            for msg in plan_messages:
                if not await telegram_queue.send(msg):
                    sent_ok = False

            # Store the plan for later application (feature kept intact)
            await self._store_pending_plan(recs)

            if sent_ok:
                logger_run.info(f"Brain report sent ({len(plan_messages)} messages) and stored for application")
            else:
                logger_run.error(f"Brain report FAILED to send ({len(plan_messages)} messages attempted) — plan still stored")
            return sent_ok
        except Exception as e:
            logger_run.warning(f"Report generation failed: {e}")
            # Fall back to the old technical report if the new one fails
            try:
                return await self._generate_and_send(pairs, telegram_queue, logger_run)
            except Exception as fallback_e:
                logger_run.error(f"Fallback report also failed: {fallback_e}")
                return False

    async def _deliver_report(self, pairs: List[str], telegram_queue: Any, logger_run: logging.Logger) -> bool:
        """Override: route through generate_report() (Profit Action Plan)
        instead of BrainEngine._generate_and_send() (old jargon report).
        maybe_generate_report()/send_report_now() and all their guards are
        inherited unchanged from the base class."""
        return await self.generate_report(pairs, telegram_queue, logger_run)