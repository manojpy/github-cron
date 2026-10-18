"""brain_report — Brain report layer: storage-key constants, fact collection and the 16-section Telegram/Markdown renderers. Pure functions over the recommendation dict; no engine state, no Redis.

Split out of the original module without logic changes; the original
module remains as a facade re-exporting every name."""
from __future__ import annotations
import logging
import unicodedata
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from datetime import datetime, timedelta, timezone
from brain_audit import DataCoverage, HealthStatus, get_audit
import threshold_engine as engine
from plan_replay import build_plan_lab
from alerts import escape_markdown_v2
from state import format_outcome_totals
from alert_registry import ALERT_KEYS, alert_family_of as _registry_family_of, pretty_alert as _pretty_alert
from brain_helpers import multiple_testing_note, kill_switch_mode

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

_RULE = "━" * 30

PLAN_HISTORY_KEY = "brain_plan_history"

PLAN_HISTORY_MAX = 100

# Current lifecycle state of the most recent Brain plan (operator-visible).
PLAN_STATE_KEY = "brain_plan_state"

PLAN_LIFECYCLE_STATES = (
    "blocked", "pending", "superseded", "applied", "rolled_back", "monitor_cleared",
)

# prev state -> states it may move to. None = first event recorded for a plan id
# (repair:<id> entries start at rolled_back).
PLAN_LIFECYCLE_TRANSITIONS: Dict[Optional[str], Set[str]] = {
    None: {"blocked", "pending", "rolled_back"},
    "blocked": {"superseded", "applied"},
    "pending": {"applied", "superseded", "blocked"},
    "applied": {"rolled_back", "monitor_cleared"},
    "superseded": set(),
    "rolled_back": set(),
    "monitor_cleared": set(),
}

def plan_transition_allowed(prev: Optional[str], new: str) -> bool:
    """True when prev -> new is a legal Brain plan lifecycle step (repeating the
    same state is allowed)."""
    return prev == new or new in PLAN_LIFECYCLE_TRANSITIONS.get(prev, set())

APPLY_SNAPSHOT_KEY = "brain_apply_snapshots"

APPLY_SNAPSHOT_MAX = 20

CHALLENGER_STREAK_KEY = "challenger_promo_streak"

_IST = timezone(timedelta(hours=5, minutes=30))

_LADDER = ["⚪", "🟡", "🟠", "🔵", "🟢"]          # Observation → Validated

_LADDER_NAMES = ["Observation", "Early evidence", "Meaningful evidence",
                 "Strong evidence", "Validated evidence"]

_MSG_LIMIT = 3800                                   # rendered chars per Telegram message

_HUMAN_SECTIONS = 6                                 # sections 1-5 are packed on their own

def _alert_family(key: str) -> str:
    """Canonical family label — delegates to the alert registry so reports and
    hierarchical family analysis share one taxonomy."""
    return _registry_family_of(key)

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
        "net_ev_known": ai.get("net_ev") is not None,
        "gate": ai.get("action_gate", {}) or {},
        "cfg_patch": recs.get("config_patch", []) or [],
        "shadow_rows": recs.get("_shadow_rows", []) or [],
        "outcome_totals": recs.get("_outcome_totals") or {},
        "archive_stats": recs.get("_archive_stats", {}) or {},
        "conf": audit.statistical_confidence_label(),
        "recon": audit.reconciliation_snapshot(),
        "coverage": audit.history_coverage(),
        "plan_lifecycle": recs.get("_plan_lifecycle") or {},
        "recs_list": recs.get("recommendations", []) or [],
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

    F["dd"] = F["gate"].get("risk_drawdown_pct")
    F["dd_budget"] = float(
        F["gate"].get("risk_budget_pct")
        or getattr(cfg, "KILL_SWITCH_MAX_DRAWDOWN_PCT", 3.0)
    )

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
    try:
        _fee = float(getattr(cfg, "BRAIN_FEE_PCT", 0.0006))
        _slip = float(getattr(cfg, "BRAIN_SLIPPAGE_PCT", 0.0003))
        _sl = float(getattr(cfg, "OUTCOME_MAE_LOSS_PCT", 1.0))
        F["plan_text"], F["plan_lab"] = build_plan_lab(
            rows, sl_pct=_sl, tp_pct=_sl * float(getattr(cfg, "OUTCOME_RR_TARGET", 2.0)),
            horizon=int(getattr(cfg, "OUTCOME_LOOKAHEAD_CANDLES", 12)),
            cost_pct=(_fee * 2 + _slip * 2) * 100.0, name_fn=_pretty_alert,
        )
    except Exception as e:
        logging.getLogger("macd_bot").warning(f"Brain report: trade-plan lab failed: {type(e).__name__}: {e}")
        F["plan_text"], F["plan_lab"] = None, {}
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

def _ev_text(F: Dict[str, Any]) -> str:
    """Net EV as text; 'n/a' when the Brain has not computed one yet (too few
    trades) instead of showing a misleading +0.00%."""
    return f"{F['net_ev']:+.2f}%" if F.get("net_ev_known") else "n/a"

def _shadow_line(F: Dict[str, Any]) -> Optional[str]:
    """One line for the rejected (shadowed) alerts and how they ended."""
    rows = F.get("shadow_rows") or []
    if not rows:
        return None
    n = len(rows)
    wins = sum(1 for r in rows if r.get("win"))
    swr = wins / n
    text = (f"👻 SHADOWED (rejected alerts, not sent): {n} trades | "
            f"{wins} wins | {n - wins} losses | WR {swr:.0%}")
    if n >= 10 and F.get("n", 0) >= 10 and swr - F["wr"] >= 0.10:
        text += "\nRejected alerts did better than the alerts sent: the filters may be too strict."
    return text

def _outcome_totals_lines(F: Dict[str, Any]) -> Optional[str]:
    """All-time Recorded / Shadowed totals, in the same format as the per-run log."""
    cum = F.get("outcome_totals") or {}
    if not cum:
        return None
    return "\n".join(
        format_outcome_totals(source, cum.get(source), icon)
        for source, icon in (("Recorded", "📝"), ("Shadowed", "👁"))
    )

def _profit_status(net_ev: float, n: int, known: bool = True) -> str:
    if n == 0:
        return "⚪ NO DATA"
    if not known:
        return "⚪ NOT ENOUGH DATA"
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
    prof = _profit_status(F["net_ev"], F["n"], F.get("net_ev_known", True))
    rec = _recording_status(F)
    if prof.startswith("🔴") or rec.startswith("🔴"):
        return "🔴 NEEDS ATTENTION"
    others = [prof, _data_status(F), rec, _conf_status(F["conf"])]
    if any(not o.startswith("🟢") for o in others) or not F["gate_ok"]:
        return "🟡 MONITOR"
    return "🟢 HEALTHY"

def _fmt_days(d: Optional[float]) -> str:
    return "n/a" if d is None else f"{d:.1f} days"

def _fmt_age(ts: Any) -> str:
    """'14h ago' style age of a unix timestamp ('n/a' when unusable)."""
    try:
        secs = max(0.0, datetime.now(timezone.utc).timestamp() - float(ts))
    except (TypeError, ValueError):
        return "n/a"
    if secs < 3600:
        return f"{int(secs // 60)}m ago"
    if secs < 86400:
        return f"{int(secs // 3600)}h ago"
    return f"{int(secs // 86400)}d ago"

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
        _r = g.get("risk_reason") or "outside current budget"
        out.append(f"Risk check failed: {_r}")
    if g and g.get("execution") is False:
        out.append("Fee/slippage assumptions missing")
    return out

def _sec_summary(F: Dict[str, Any], cfg) -> List[_Piece]:
    n, wr, net_ev, days = F["n"], F["wr"], F["net_ev"], F["days"]
    validated_mode = F["gate_ok"] and net_ev > 0 and F["conf"] in ("MODERATE", "HIGH")
    if validated_mode:
        return _sec_verdict(F, cfg)
    prof = _profit_status(net_ev, n, F["net_ev_known"])
    out = [_hdr(2, 'EXECUTIVE SUMMARY — "WHAT DO I NEED TO KNOW?"')]
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
            ("💰 Net EV/trade:", _ev_text(F)),
            ("📅 History available:", _fmt_days(days)),
            ("📅 History requested:", f"{F['req_days']} days"),
        ], "lr")
    )))
    _shadow = _shadow_line(F)
    if _shadow:
        out.append(_p(_shadow))
    _totals = _outcome_totals_lines(F)
    if _totals:
        out.append(_p(_totals))
    losing = F["net_ev_known"] and net_ev <= -0.05
    if losing and F["low_trust"]:
        text = (
            "The system is currently producing poor results in the available sample.\n\n"
            f"However, only {_fmt_days(days)} of history are available, so the Brain does NOT yet "
            "have enough evidence to say whether this is a persistent strategy problem.\n\n"
            "🎯 Current priority: DIAGNOSE + COLLECT DATA\n"
            "➡️ Not yet: AGGRESSIVE OPTIMISATION"
        )
    elif losing:
        text = (
            "The system is producing poor results and the history is long enough for this to be "
            "taken seriously.\n\n"
            "➡️ Current priority: FIX THE WEAKEST PARTS (see sections 04 and 05)\n"
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
    out = [_hdr(2, "🧠 BRAIN VERDICT")]
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
    _shadow = _shadow_line(F)
    if _shadow:
        out.append(_p(_shadow))
    _totals = _outcome_totals_lines(F)
    if _totals:
        out.append(_p(_totals))
    return out

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

def _replay_text(s: Dict[str, Any]) -> str:
    """One short line of same-opportunity replay evidence for a scenario ('' if none)."""
    rp = s.get("replay") or {}
    dl = rp.get("delta")
    if not rp.get("valid") or not dl:
        return ""
    label = {
        "supported": "🟢 change holds up after allowing for clustering",
        "likely_harmful": "🔴 replay says it would have lost money",
        "not_distinguishable": "🟡 change not distinguishable from zero",
        "insufficient": "🟡 too few changed trades to judge",
    }.get(str(rp.get("verdict") or ""), "")
    ci = "" if dl.get("lcb") is None else f" (90% range {dl['lcb']:+.1f} to {dl['ucb']:+.1f})"
    return (f"\n   Replay: {dl['total_pnl_pct']:+.1f} pts total P&L{ci}, "
            f"trades -{rp.get('trades_removed', 0)}/+{rp.get('trades_added', 0)}, "
            f"max drawdown {dl['max_dd_change_pct']:+.1f} pts — {label}")

def _sec_do_now(F: Dict[str, Any], cfg) -> List[_Piece]:
    n, wr = F["n"], F["wr"]
    out = [_hdr(3, "🚦 WHAT SHOULD I DO NOW?")]
    do_now: List[str] = []
    recon_ok = F["recon"] is not None and F["recon"].status == HealthStatus.OK
    if wr < 0.10 or not recon_ok:
        do_now.append("Verify alert → trade → outcome recording is correct.")
    if F["coverage"] is None or F["coverage"].coverage != DataCoverage.FULL:
        do_now.append("Continue collecting clean outcome data.")
    do_now.append("Keep Shadow Mode enabled." if F["shadow_on"]
                  else "Turn Shadow Mode ON (BRAIN_SHADOW_MODE=true) so rejected trades can be judged.")
    if wr < 0.10 and n:
        do_now.append(f"Investigate the very low {wr:.0%} win rate (see section 04).")
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

    # ── Candidate vs Control (counterfactual simulator) ──
    # Only scenarios that actually beat the live config, top 3 — this report
    # is meant to be scannable, not a full simulation log. Silent when
    # nothing currently beats control.
    beats_control = sorted(
        (s for s in (F["ai"].get("counterfactual_scenarios") or []) if s.get("delta_ev", 0.0) > 0),
        key=lambda s: s.get("ev", float("-inf")),
        reverse=True,
    )[:3]
    if beats_control:
        lines = []
        for s in beats_control:
            _, verdict = _cf_verdict(s, cfg)
            if "PROMOTION-ELIGIBLE" in verdict and s.get("ev", 0.0) <= 0:
                verdict = "🕒 HOLD (still negative EV)"
            lines.append(
                f"• {s.get('label', '?')}: {_ev_text(F)} → {s.get('ev', 0.0):+.2f}% "
                f"(n={s.get('n', 0)}) — {verdict}" + _replay_text(s)
            )
        out.append(_p("🥇 CANDIDATE CONFIGS (beat current live config)\n\n" + "\n".join(lines)))
    return out

def _sec_profit(F: Dict[str, Any], cfg) -> List[_Piece]:
    n, wr, net_ev, days = F["n"], F["wr"], F["net_ev"], F["days"]
    an = F["anatomy"] or {}
    be = 1.0 / (1.0 + an["rr"]) if an.get("rr") else None
    out = [_hdr(4, '📊 PROFITABILITY — "ARE WE ACTUALLY MAKING MONEY?"')]
    wr_verdict = ("🔴 Very poor" if be and wr < be * 0.5 else "🔴 Below break-even" if be and wr < be
                  else "🟢 At/above break-even" if be else "⚪")
    dd, bud = F["dd"], F["dd_budget"]
    dd_win = F["gate"].get("risk_window_hours", 24)

    # Emoji lifted out of VERDICT and placed at column 0; column order is
    # now emoji | metric | result (right-aligned) | verdict text.
    entries = [
        ("Trades", str(n),
         "🟢 Good sample" if n >= 100 else "🟡 Small sample" if n >= 30 else "🔴 Tiny sample"),
        ("Win Rate", f"{wr:.0%}", wr_verdict),
        ("Net EV", _ev_text(F),
         "⚪ Not computed yet" if not F["net_ev_known"]
         else "🟢 Positive" if net_ev > 0.05 else "🟡 Flat" if net_ev > -0.05 else "🔴 Negative"),
        ("History", "n/a" if days is None else f"{days:.1f}d",
         "🟢 Long enough" if (days or 0) >= 30 else "🟡 Short" if (days or 0) >= 14 else "🔴 Too short"),
        ("After costs?", "YES" if F["gate"].get("execution", True) else "NO",
         "🟢 Costs accounted" if F["gate"].get("execution", True) else "🔴 Missing"),
        (f"Drawdown ({dd_win}h)", "n/a" if dd is None else f"{dd:.1f}%",
         "⚪ unknown" if dd is None else f"{'🟢 Within' if dd <= bud else '🔴 Over'} {bud:.1f}% budget"),
        ("CUSUM drift", "ACTIVE" if F["gate"].get("stability") is False else "NONE",
         "🔴 Drifting" if F["gate"].get("stability") is False else "🟢 Stable"),
    ]
    rows = [("", "METRIC", "RESULT", "VERDICT")]
    for metric, value, verdict in entries:
        emoji, rest = _split_leading_emoji(verdict)
        rows.append((emoji or "⚪", metric, value, rest))
    out.extend(_c_split(_table(rows, "llrl")))

    if not F["net_ev_known"]:
        meaning = ("Net EV is not computed yet (too few trades), so profit cannot be judged. "
                   "The win rate and the trade outcomes below are the only guide for now.")
    elif net_ev < 0:
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
        ("Confidence in the DATA PIPELINE", '🟢 GOOD' if recon_ok else '🟡 CHECK OUTCOME RECORDING (section 02)'),
    ]))))
    if F["anatomy_text"]:
        _anat_title = ("🎲 WHY THE WIN RATE LOOKS LOW" if (be is not None and wr < be)
                       else "🎲 HOW THE TRADES ENDED")
        out.append(_p(f"{_anat_title}\n\n" + F["anatomy_text"]))

    if F.get("plan_text"):
        out.append(_p("🧪 TRADE-PLAN LAB — \"WOULD A DIFFERENT TARGET/STOP HAVE WORKED?\"\n\n" + F["plan_text"]))
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
    out = [_hdr(5, '🔎 LOSS DIAGNOSIS — "WHERE ARE WE FALTERING?"')]
    weak = F["weak"]
    if not weak:
        out.append(_p("No alert with at least 10 trades has negative EV in this sample."))
        return out
    out.append(_p("🔴 CURRENTLY WEAK ALERT FAMILIES (worst total loss first)"))
    out.extend(_alert_table(weak, 8))
    if len(weak) > 8:
        out.append(_p(f"…and {len(weak) - 8} more (full list in the archived report)."))
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
    out = [_hdr(6, '🟢 POSITIVE SIGNS — "WHERE ARE WE DOING BETTER?"')]
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
    out = [_hdr(7, '🚦 ALERT SCORECARD — "WHAT SHOULD I TRUST?"')]
    validated = [a for a in F["good"] if a["rank"] == 4]
    promising = [a for a in F["good"] if a["rank"] < 4]

    def _group(title: str, names: List[str], empty: str = "None currently.") -> None:
        body = "\n".join(_wrap_names(names)) if names else "   " + empty
        out.append(_p(f"{title}\n{body}"))

    _group("🟢 VALIDATED / ACTIONABLE", [a["name"] for a in validated])
    _group("🟡 PROMISING — NEED MORE EVIDENCE", [a["name"] for a in promising])
    _group("🔴 UNDERPERFORMING — INVESTIGATE", [a["name"] for a in F["weak"]])
    thin = sorted(F["thin"], key=lambda a: -a["n"])
    # Names omitted on purpose: 20–30 thin alerts bloat Telegram without aiding decisions.
    _group(
        f"⚪ INSUFFICIENT DATA ({len(thin)} alert types, <10 trades)",
        [],
        empty=f"{len(thin)} types — see archive report if needed.",
    )
    zero = len(F["alerts"]) - len(validated) - len(promising) - len(F["weak"]) - len(thin)
    if zero > 0:
        out.append(_p(f"➖ {zero} alert(s) with exactly zero net EV are not listed above."))
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

def _confidence_tier(value: Optional[float], high: float, medium: float) -> str:
    """Map a 0-1 confidence-like score to a HIGH/MEDIUM/NOT READY tier.
    None (score not computable yet, e.g. insufficient data) reports N/A
    rather than a false NOT READY — those are different situations."""
    if value is None:
        return "⚪ N/A"
    if value >= high:
        return "🟢 HIGH"
    if value >= medium:
        return "🟡 MEDIUM"
    return "🔴 NOT READY"

def _confidence_breakdown(F: Dict[str, Any], cfg) -> List[Tuple[str, str, str]]:
    """(axis, tier, detail) for the four independent confidence axes —
    MODEL (classifier calibration), DATA (sample size), CHANGE (does a
    candidate beat control with confidence), DEPLOYMENT (survived OOS and
    currently stable). Kept separate rather than blended into one score:
    a change can be HIGH-confidence on thin evidence, or well-evidenced but
    not yet deployment-ready — one number hides exactly that distinction.
    """
    gate = F["gate"]
    rows: List[Tuple[str, str, str]] = []

    brier = F["ai"].get("brier_score")
    if brier is None:
        rows.append(("MODEL", "⚪ N/A", "no calibration data yet"))
    else:
        _wr = float(F.get("wr") or 0.0)
        _ref = _wr * (1.0 - _wr)      # Brier of always predicting the base rate
        if F.get("n") and brier > _ref + 0.005:
            rows.append(("MODEL", "🔴 NOT READY", f"Brier {brier:.2f} is worse than base rate {_ref:.2f}"))
        else:
            tier = "🟢 HIGH" if brier < 0.20 else "🟡 MEDIUM" if brier < 0.25 else "🔴 NOT READY"
            rows.append(("MODEL", tier, f"Brier {brier:.2f}"))
    rank = _evidence_rank(F["n"], F["days"])
    days = F["days"]
    if rank == 0 or (days is not None and days < 7):
        data_tier = "🔴 NOT READY"
    elif rank == 1:
        data_tier = "🟡 MEDIUM"
    else:
        data_tier = "🟢 HIGH"
    rows.append(("DATA", data_tier, f"{F['n']} trades, {_fmt_days(F['days'])}"))

    p_thr = getattr(cfg, "BRAIN_EV_GATE_P_THRESHOLD", 0.85)
    change_p = gate.get("profit_p_ev_positive")
    ev_p5 = gate.get("ev_p5")
    change_tier = _confidence_tier(change_p, high=p_thr, medium=max(0.0, p_thr - 0.15))
    change_detail = (f"P(EV>0) {change_p:.0%}, EV p5 {ev_p5:+.2f}%"
                      if change_p is not None and ev_p5 is not None else "insufficient data")
    rows.append(("CHANGE", change_tier, change_detail))

    oos_p = gate.get("oos_p_ev_positive")
    deploy_tier = _confidence_tier(oos_p, high=0.70, medium=0.55)
    deploy_detail = f"OOS P(EV>0) {oos_p:.0%}" if oos_p is not None else "walk-forward not run"
    if gate.get("stability") is False:
        deploy_tier = "🔴 NOT READY"
        deploy_detail += ", active drift"
    rows.append(("DEPLOYMENT", deploy_tier, deploy_detail))
    return rows

_PLAN_STATE_LABEL = {
    "pending": ("🟡", "PENDING — waiting for you to apply"),
    "blocked": ("🔴", "BLOCKED — action gate not passed"),
    "applied": ("🟢", "APPLIED — under post-apply monitoring"),
    "monitor_cleared": ("✅", "CLEARED — monitoring found no harm"),
    "rolled_back": ("↩️", "ROLLED BACK — change reverted"),
    "superseded": ("⚪", "SUPERSEDED — replaced by a newer plan"),
}

def _fmt_plan_ts(ts: Any) -> str:
    try:
        return datetime.fromtimestamp(int(ts), _IST).strftime("%d %b %H:%M")
    except Exception:
        return "?"

def _plan_lifecycle_text(L: Dict[str, Any]) -> Optional[str]:
    """Plain-text lifecycle block for the action-gate section: the last recorded
    plan state with its reason, plus the most recent transitions. None when no
    plan has been recorded yet."""
    cur = (L or {}).get("current") or {}
    state = cur.get("state")
    if not state:
        return None
    icon, label = _PLAN_STATE_LABEL.get(str(state), ("⚪", str(state).upper()))
    lines = [
        "PLAN LIFECYCLE (last recorded)",
        "",
        f"{icon} {cur.get('plan_id', '?')}: {label}",
        f"Since {_fmt_plan_ts(cur.get('ts'))} IST",
    ]
    if cur.get("reason"):
        lines.append(f"Reason: {str(cur['reason'])[:160]}")
    if cur.get("unexpected"):
        lines.append("⚠️ Unexpected state change — see logs")
    recent = [e for e in ((L or {}).get("recent") or []) if isinstance(e, dict)][-5:]
    if recent:
        lines += ["", "Recent steps (newest first):"]
        for ev in reversed(recent):
            lines.append(
                f"{_fmt_plan_ts(ev.get('ts'))}  {ev.get('plan_id')}  "
                f"{ev.get('from') or 'new'} → {ev.get('status')}"
            )
    return "\n".join(lines)

def _sec_gate(F: Dict[str, Any], cfg) -> List[_Piece]:
    g = F["gate"]
    out = [_hdr(9, '🛡️ ACTION GATE — "CAN THE BRAIN SAFELY CHANGE ANYTHING?"')]
    # Risk label shows the WINDOW and measured values so a FAIL is
    _dd_pct = g.get("risk_drawdown_pct")
    _dd_win = g.get("risk_window_hours", 24)
    _dd_bud = g.get("risk_budget_pct", 3.0)
    if _dd_pct is None:
        _risk_label = f"Drawdown ({_dd_win}h: n/a)"
    else:
        _risk_label = f"Drawdown ({_dd_win}h: {_dd_pct:.1f}% / {_dd_bud:.1f}%)"

    labels = [("data_quality", "Minimum trades"), ("oos_prediction", "OOS EV"),
              ("profitability", "Net EV confidence"), ("stability", "CUSUM drift"),
              ("risk", _risk_label), ("execution", "Cost assumptions")]
    rows = []
    for k, lab in labels:
        if k == "oos_prediction" and g.get("oos_p_ev_positive") is None and not g.get(k):
            rows.append((f"⚪ {lab}:", "N/A (not run)"))
            continue
        rows.append((f"{'🟢' if g.get(k) else '🔴'} {lab}:", "PASS" if g.get(k) else "FAIL"))
    out.extend(_c_split(_table(rows, "ll")))
    out.append(_p("CONFIDENCE BREAKDOWN"))
    _cb_rows = []
    for axis, tier, detail in _confidence_breakdown(F, cfg):
        _emoji, _tier_txt = _split_leading_emoji(tier)
        _cb_rows.append((f"{_emoji or '⚪'} {axis}", f"{_tier_txt}  ({detail})"))
    out.extend(_c_split(_table(_cb_rows, "ll")))
    out.append(_p(
        "OVERALL:\n\n"

        + ("🟢 BRAIN ACTION GATE = PASSED\n\nMeaning:\n\n\"The Brain's evidence is strong enough to "
           "recommend specific changes to the live strategy.\""
           if F["gate_ok"] else
           "🔴 BRAIN ACTION GATE = BLOCKED\n\nMeaning:\n\n\"The Brain may analyse and recommend what "
           "to investigate, but it is not sufficiently confident to modify the live strategy.\"")
    ))
    _lifecycle = _plan_lifecycle_text(F.get("plan_lifecycle") or {})
    if _lifecycle:
        out.append(_p(_lifecycle))
    _ptr_lines = _prediction_track_lines((F.get("ai") or {}).get("prediction_track_record"))
    if _ptr_lines:
        out.append(_p("PREDICTION TRACK RECORD — did the Brain's stated odds match reality?"))
        out.extend(_c_split(_ptr_lines))
    _acct_lines = _accountability_lines(
        (F.get("ai") or {}).get("family_accountability"),
        str(getattr(cfg, "FAMILY_ACCOUNTABILITY_MODE", "shadow")),
        (F.get("ai") or {}).get("control_summary"),
    )
    if _acct_lines:
        out.append(_p("RECOMMENDATION ACCOUNTABILITY — did past changes help?"))
        out.extend(_c_split(_acct_lines))
    _df_lines = _data_foundation_lines((F.get("ai") or {}).get("data_foundation"))
    if _df_lines:
        out.append(_p("DATA FOUNDATION — is the evidence base consistent?"))
        out.extend(_c_split(_df_lines))
    _ly_lines = _layer_lines((F.get("ai") or {}).get("layer_audit"), str(getattr(cfg, "LAYER_BOUNDARY_MODE", "report")))
    if _ly_lines:
        out.append(_p("ARCHITECTURE BOUNDARY — research, validation, policy"))
        out.extend(_c_split(_ly_lines))
    _hg_lines = _hypothesis_gen_lines((F.get("ai") or {}).get("hypothesis_pool_summary"))
    if _hg_lines:
        out.append(_p("SELF-GENERATED RULES — proposed, tested, followed forward"))
        out.extend(_c_split(_hg_lines))
    _vh_lines = _vol_horizon_lines((F.get("ai") or {}).get("vol_horizon_study"))
    if _vh_lines:
        out.append(_p("VOLATILITY AND TRADE PLANS — do regimes need different stops and horizons?"))
        out.extend(_c_split(_vh_lines))
    _om_lines = _online_model_lines((F.get("ai") or {}).get("online_model"))
    if _om_lines:
        out.append(_p("ONLINE MODEL vs BATCH MODEL — does continuous learning help?"))
        out.extend(_c_split(_om_lines))
    _fe_lines = _future_edge_lines((F.get("ai") or {}).get("future_edge_record"))
    if _fe_lines:
        out.append(_p("FUTURE EDGE SCORE — noise vs ignorance"))
        out.extend(_c_split(_fe_lines))
    _nev_lines = _net_ev_audit_lines((F.get("ai") or {}).get("net_ev_audit"))

    if _nev_lines:
        out.append(_p("NET-EV MODEL vs SIMPLE RULES — does it pick better trades?"))
        out.extend(_c_split(_nev_lines))
    _hyp_lines = _hypothesis_lines((F.get("ai") or {}).get("hypothesis_ledger"))
    if _hyp_lines:
        out.append(_p("CONCLUSIONS THAT CHANGED — what the Brain used to believe"))
        out.extend(_c_split(_hyp_lines))
    _exp_lines = _experiment_lines((F.get("ai") or {}).get("experiment_ledger"))
    if _exp_lines:
        out.append(_p("EXPERIMENT LEDGER — how much was tried to find these results?"))
        out.extend(_c_split(_exp_lines))
    return out

ACTION_PLAN_SCHEMA = "brain_action_plan/1"
ACTION_PLAN_TTL_HOURS = 7 * 24          # matches the 7-day life of the stored pending plan
ACTION_MONITOR_HORIZON_HOURS = 48       # same defaults evaluate_pending_repairs uses
ACTION_MONITOR_MIN_OUTCOMES = 30
_GATE_LAYERS = ("data_quality", "oos_prediction", "profitability", "stability", "risk", "execution")

def to_action_plan(plan_data: Dict[str, Any], recs: Dict[str, Any], cfg, now: Optional[int] = None, impacts: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Machine-readable twin of the human report: one object per proposed change.

    Pure: reads the stored pending plan and the recommendation dict, returns JSON-safe
    data, changes nothing. `validation_status` is one of insufficient / exploratory /
    validated. (shadow- and live-confirmed need a repair-ledger verdict, which exists
    only after the change has been applied, so they are not assigned here.)
    `gate_cleared` says the six-layer action gate passed; whether a change is actually
    applied automatically is decided elsewhere, never by this function.
    `impacts` (optional) maps 'kind:target' to a counterfactual replay result; a change
    whose replay is 'likely_harmful' is marked not cleared. Replay is descriptive, not causal."""
    import time as _time
    now = int(now if now is not None else _time.time())
    ai = recs.get("ai_metrics") or {}
    gate = ai.get("action_gate") or {}
    gate_passed = bool(plan_data.get("_action_gate_passed"))
    failed = [k for k in _GATE_LAYERS if not gate.get(k)]
    if not gate.get("data_quality"):
        status = "insufficient"
    elif gate_passed:
        status = "validated"
    else:
        status = "exploratory"
    generated = int(plan_data.get("generated_at") or now)
    expires = generated + ACTION_PLAN_TTL_HOURS * 3600
    plan_id = plan_data.get("plan_id")
    acct = ai.get("family_accountability") or {}
    disable_recs = {
        str(r.get("alert") or r.get("alert_key")): r
        for r in (recs.get("recommendations") or []) if r.get("type") == "disable_alert"
    }

    actions: List[Dict[str, Any]] = []

    def _add(kind: str, target: str, old: Any, new: Any, rationale: str,
             applies_via: str, category: Optional[str], impact: Dict[str, Any],
             auto_enabled: Optional[bool]) -> None:
        rp = (impacts or {}).get(f"{kind}:{target}")
        harmful = False
        if rp and rp.get("replayable") and rp.get("delta"):
            dl = rp["delta"]
            impact = {
                "basis": "historical_replay", "causal": False,
                "net_pnl_delta_pct_points": dl.get("total_pnl_pct"),
                "ci90": [dl.get("lcb"), dl.get("ucb")],
                "net_ev_delta_pct": dl.get("ev_per_trade_change"),
                "trades_added": rp.get("trades_added"), "trades_removed": rp.get("trades_removed"),
                "max_dd_change_pct_points": dl.get("max_dd_change_pct"),
                "uses_shadow_rows": rp.get("uses_shadow_rows"),
                "replay_verdict": rp.get("verdict"),
                "n_opportunities": rp.get("n_opportunities"),
            }
            harmful = rp.get("verdict") == "likely_harmful"
        elif rp and not rp.get("replayable", True):
            impact = {"basis": "not_replayable", "reason": rp.get("reason"), "net_ev_delta_pct": None}
        actions.append({
            "id": f"{plan_id or 'PLAN'}#{len(actions) + 1}",
            "kind": kind, "target": target, "old": old, "new": new,
            "rationale": rationale, "source_category": category,
            "family": (
                {k: acct[category].get(k) for k in ("status", "n", "helped", "hurt", "ev_lift_mean", "control_effect_mean", "lift_tag")}
                if category in acct else None
            ),
            "validation_status": status,
            "gate_cleared": gate_passed and not harmful,
            "blocked_by": ([] if gate_passed else list(failed)) + (["counterfactual_replay"] if harmful else []),
            "applies_via": applies_via,
            "auto_apply_enabled": auto_enabled,
            "expected_impact": impact,
            "monitoring": {
                "metric": "scoped net EV and win rate after application",
                "horizon_hours": ACTION_MONITOR_HORIZON_HOURS,
                "min_outcomes": ACTION_MONITOR_MIN_OUTCOMES,
                "baseline_net_ev_pct": ai.get("net_ev"),
                "ledger_key": "brain_repair_ledger",
            },
            "rollback": {
                "trigger": "repair-ledger verdict 'hurt' (net EV falls by more than 0.02 with win rate not rising)",
                "restore": old,
            },
            "expires_at": expires,
        })

    _unknown = {"basis": "not_estimated", "net_ev_delta_pct": None}
    for p in plan_data.get("config_patch") or []:
        _add("config_patch", str(p.get("path")), p.get("current"), p.get("suggested"),
             str(p.get("reason") or ""), "apply_config_override", p.get("_source_category"), _unknown, None)
    for w in plan_data.get("weight_adjustments") or []:
        _add("weight", f"weight:{w.get('vote')}", w.get("current"), w.get("suggested"),
             str(w.get("reason") or ""), "dynamic_weights", w.get("category"), _unknown,
             bool(getattr(cfg, "BRAIN_AUTO_APPLY_DYNAMIC_WEIGHTS", False)))
    for ak in plan_data.get("disable_alerts") or []:
        r = disable_recs.get(str(ak)) or {}
        _add("disable_alert", str(ak), "enabled", "disabled", str(r.get("message") or "brain_disable"),
             "alert_toggle", "disable_alert",
             {"basis": "observed", "win_rate": r.get("win_rate"), "n": r.get("sample_size"), "net_ev_delta_pct": None}, None)
    for ak in plan_data.get("reinstate_alerts") or []:
        _add("reinstate_alert", str(ak), "disabled", "enabled", "brain_reinstate",
             "alert_toggle", "reinstate_alert", _unknown, None)

    ptr = ai.get("prediction_track_record") or {}
    ledger = ai.get("experiment_ledger") or {}
    n_tests = int((ledger.get("last") or {}).get("zone_candidates", 0)) + int((ledger.get("last") or {}).get("regime_segments", 0))
    return {
        "schema": ACTION_PLAN_SCHEMA,
        "plan_id": plan_id,
        "generated_at": generated,
        "expires_at": expires,
        "status": status,
        "evidence": {
            "n_real": (plan_data.get("sample_size") or {}).get("real"),
            "n_shadow": (plan_data.get("sample_size") or {}).get("shadow"),
            "action_gate": {k: bool(gate.get(k)) for k in _GATE_LAYERS},
            "oos_p_ev_positive": gate.get("oos_p_ev_positive"),
            "profit_p_ev_positive": gate.get("profit_p_ev_positive"),
            "ev_p5": gate.get("ev_p5"),
            "net_ev_pct": ai.get("net_ev"),
            "prediction_status": {
                f: p.get("status") for f, p in (ptr.get("predictors") or {}).items()
            },
            "multiple_testing": multiple_testing_note(n_tests),
        },
        "actions": actions,
        "accountability_mode": str(getattr(cfg, "FAMILY_ACCOUNTABILITY_MODE", "shadow")),
        "suppressed_families": plan_data.get("suppressed_families") or [],
        "would_suppress_families": plan_data.get("would_suppress_families") or [],
        "note": "Historical evidence on fired alerts; not a forecast of future profit.",
    }

def _accountability_lines(acct: Optional[Dict[str, Any]], mode: str, control_sum: Optional[Dict[str, Any]] = None) -> List[str]:
    """Plain-text lines: which recommendation families helped or hurt when acted on."""
    if not acct:
        return []
    icon = {"eligible": "🟢", "insufficient": "⚪", "probation": "🟡", "suspended": "🔴"}
    rows: List[Tuple[str, ...]] = []
    for cat, f in sorted(acct.items(), key=lambda kv: (kv[1].get("eligible", True), -kv[1].get("n", 0))):
        _lift = f.get("ev_lift_mean")
        _ctl = f.get("control_effect_mean")
        rows.append((f"{icon.get(f['status'], '⚪')} {cat}", f"n={f['n']}",
                     f"helped {f['helped']} / hurt {f['hurt']} / neutral {f['neutral']}", f["status"],
                     "lift " + (f"{_lift:+.3f}" if _lift is not None else "n/a")
                     + " vs control " + (f"{_ctl:+.3f}" if _ctl is not None else "n/a")
                     + f" ({str(f.get('lift_tag') or 'insufficient').replace('_', ' ')})"))
    lines = list(_table(rows, "lllll"))
    sus = [c for c, f in acct.items() if not f.get("eligible", True)]
    if sus:
        lines.append("")
        lines.append(
            ("Suspended (changes from these are left out of the plan): " if mode == "live"
             else "Would be suspended (shadow mode, nothing is blocked yet): ") + ", ".join(sorted(sus))
        )
    lines.append("")
    lines.append("Counts only changes that were applied and later measured; families with no measured changes do not appear.")
    cs = control_sum or {}
    if cs.get("checked"):
        lines.append(
            f"Comparison-group check: of {cs['checked']} measured verdicts, {cs['confirmed']} held up against trades "
            f"the change did not touch, {cs['drift']} looked like the whole market moving, {cs['contradicted']} pointed the other way."
        )
    return lines

def _hypothesis_lines(led: Optional[Dict[str, Any]]) -> List[str]:
    """Plain-text lines: conclusions that were later reversed, and unstable ones."""
    if not led or not led.get("subjects"):
        return []
    ev = led.get("events") or []
    lines = [f"Tracking {len(led['subjects'])} standing conclusions; {sum(1 for e in ev if e.get('kind') == 'overturned')} reversed so far."]
    for e in ev[:5]:
        held = e.get("held_sec") or 0
        lines.append(
            f"{'Reversed' if e['kind'] == 'overturned' else 'No longer supported'}: {e.get('claim')} "
            f"({e.get('from')} -> {e.get('to')}, stood {held / 86400:.1f} days)"
        )
    shaky = [sid for sid, v in led["subjects"].items() if int(v.get("flips") or 0) >= 2]
    if shaky:
        lines.append("Keeps flipping (treat as unproven): " + ", ".join(sorted(shaky)))
    return lines

def _online_model_lines(rec: Optional[Dict[str, Any]]) -> List[str]:
    """Plain-text lines: has the online-updated model beaten the batch model it would replace?"""
    if not rec or not rec.get("mode"):
        return []

    names = {
        "online_better": "the online copy predicts better", "batch_better": "the batch model predicts better",
        "indistinguishable": "no clear difference yet", "insufficient": "not enough newly resolved trades yet",
    }
    status = rec.get("status")
    verdict = names.get(status, status) if isinstance(status, str) else status
    lines = [
        f"Mode: {rec['mode']}. Trades scored by both models: {rec.get('n_eval', 0)}; "
        f"the online copy has learned from {rec.get('n_updates', 0)}.",
        f"Verdict: {verdict}.",
    ]
    if rec.get("mean_brier_gain") is not None:
        lines.append(
            f"Average Brier gain per trade {rec['mean_brier_gain']:+.4f} (90% range {rec['lo']:+.4f} to {rec['hi']:+.4f}); "
            "positive means the online copy was closer to what happened."
        )
    if rec.get("max_coef_drift") is not None:
        lines.append(f"Largest coefficient change from the starting model: {rec['max_coef_drift']} ({'within' if rec.get('drift_ok') else 'OVER'} the limit).")
    if rec.get("promoted"):
        lines.append("Being served in place of the batch model.")
    elif rec.get("would_promote"):
        lines.append("Would be promoted (shadow mode: nothing changed).")
    else:
        lines.append(f"Promotion needs {rec.get('min_passes', 3)} consecutive passing evaluations; passed so far: {rec.get('passes', 0)}.")
    if rec.get("demoted"):
        lines.append("Demoted this cycle.")
    return lines

def _future_edge_lines(rec: Optional[Dict[str, Any]]) -> List[str]:
    """Plain-text lines: noise vs ignorance, and whether the Future Edge Score has predicted anything."""
    if not rec or not rec.get("n"):
        return []
    g = rec.get("groups") or {}
    names = {"positive": "edge survives the uncertainty", "uncertain": "average is positive but not after uncertainty", "none": "no positive average"}
    lines = []
    for k in ("positive", "uncertain", "none"):
        if k in g:
            lines.append(f"{names[k]}: {g[k]['n']} alerts, average {g[k]['mean_net']:+.3f}% per trade")
    if rec.get("mean_epistemic_share") is not None:
        lines.append(
            f"On average {rec['mean_epistemic_share']:.1%} of the uncertainty about a single trade is ignorance "
            f"(reducible with more data); the rest is market noise that no model can remove."
        )
    st = rec.get("status")
    if st == "predictive":
        lines.append(f"Verdict: PREDICTIVE — positive-score alerts beat the rest by {rec['positive_minus_rest']:+.3f}% per trade (lower bound {rec['lo']}).")
    elif st == "not_predictive":
        lines.append(f"Verdict: NOT PREDICTIVE yet — difference {rec['positive_minus_rest']:+.3f}% per trade, lower bound {rec['lo']}.")
    else:
        lines.append("Verdict: too few alerts in each group to judge yet.")
    return lines

def _vol_horizon_lines(st: Optional[Dict[str, Any]]) -> List[str]:
    """Plain-text lines: do volatility regimes call for different stops, targets or horizons?"""
    if not st or not st.get("buckets"):
        return []
    if st.get("status") == "insufficient":
        return [f"Not judged yet: {st.get('n', 0)} trades with a stored path and volatility reading."]
    lines = []
    for name in ("low", "mid", "high"):
        b = (st.get("buckets") or {}).get(name) or {}
        if not b.get("n"):
            continue
        p = b.get("plan") or {}
        plan = f"best stop {p['sl']:g}% / target {p['tp']:g}% / {p['h']} candles" if p else "no plan chosen"
        lines.append(
            f"{name.capitalize()} volatility (n={b['n']}): typical best move {b.get('median_best_move')}%, "
            f"typical worst {b.get('median_worst_move')}%, best move comes around candle {b.get('median_candle_of_best')}; {plan}"
        )
    sp = st.get("single_plan") or {}
    if sp:
        lines.append(f"One tuned plan for everything: stop {sp['sl']:g}% / target {sp['tp']:g}% / {sp['h']} candles.")
    if st.get("overall_delta") is not None:
        lines.append(
            f"On {st.get('n_hold_pairs')} newer trades, per-regime plans vs that single plan: {st['overall_delta']:+.3f}% per trade "
            f"(lower bound {st.get('overall_lo')}); single tuned plan vs the current fixed plan: {st.get('single_over_fixed'):+.3f}%."
        )
    verdict = {"supported": "SUPPORTED — volatility-specific plans beat one tuned plan on unseen trades.",
               "worse": "WORSE — per-regime plans did worse than one tuned plan.",
               "not_distinguishable": "NOT DISTINGUISHABLE — no evidence that volatility regimes need different plans."}.get(str(st.get("status") or ""), "")

    if verdict:
        lines.append("Verdict: " + verdict + " Nothing is applied; the outcome label stays at the fixed 12 candles.")
    return lines

def _hypothesis_gen_lines(sm: Optional[Dict[str, Any]]) -> List[str]:
    """Plain-text lines: self-generated skip-rules, what is being followed and what held up."""
    if not sm or sm.get("status") == "insufficient_data" and not (sm.get("shadow") or sm.get("confirmed") or sm.get("events")):
        return []
    lines = []
    if sm.get("status") == "ran":
        lines.append(
            f"Proposed {sm['n_tested']} new candidate rules this cycle ({sm['tried_total']} ever tried); "
            f"a rule needed a {sm['z_required']}-sigma loss on newer trades to survive. {sm.get('survivors', 0)} survived."
        )
    for c in sm.get("confirmed") or []:
        f = c.get("fwd") or {}
        lines.append(f"CONFIRMED: skip when {c['text']} — {f.get('n')} later trades averaged {f.get('mean')}% each")
    for c in sm.get("shadow") or []:
        f = c.get("fwd") or {}
        lines.append(
            f"Watching: skip when {c['text']} — found on older data (avg {c['hold']['mean']}% on {c['hold']['n']} newer trades); "
            f"forward so far {f.get('n', 0)} trades" + (f", avg {f['mean']}%" if f.get("mean") is not None else "")
        )
    for e in sm.get("events") or []:
        if e.get("status") in ("rejected", "revoked", "expired"):
            lines.append(f"{e['status'].capitalize()}: {e['text']}")
    if lines:
        lines.append("Nothing here is applied. A confirmed rule is a candidate to review before adding it to the 'when not to take it' list.")
    return lines

def _layer_lines(a: Optional[Dict[str, Any]], mode: str) -> List[str]:
    """Plain-text lines: does anything outside the policy layer change live behaviour?"""
    if not a:
        return []
    lay = a.get("layers") or {}
    lines = [
        f"{sum(lay.values())} modules classified ({lay.get('research', 0)} research, {lay.get('validation', 0)} validation, "
        f"{lay.get('policy', 0)} policy, {lay.get('infra', 0)} plumbing)."
    ]
    for v in a.get("violations") or []:
        lines.append(f"VIOLATION: {v['module']} (line {v['line']}) — {v['kind'].replace('_', ' ')}: {v['detail']}")
    for e in a.get("exceptions") or []:
        lines.append(f"Known: {e['module']} applies '{e['writer']}' itself — {e['reason']}")
    for m in a.get("unclassified") or []:
        lines.append(f"Unclassified module: {m} (add it to brain_layers.LAYER_OF)")
    for s in a.get("stale_exceptions") or []:
        lines.append(f"Documented exception no longer in the code: {s}")
    if a.get("ok"):
        lines.append("Boundary holds: nothing outside the policy layer changes live behaviour except the documented cases above.")
    elif mode == "block":
        lines.append("Plans are NOT applied until this is fixed (LAYER_BOUNDARY_MODE=block).")
    else:
        lines.append("Reported only; set LAYER_BOUNDARY_MODE=block to stop plan application while this stands.")
    return lines

def _data_foundation_lines(cov: Optional[Dict[str, Any]]) -> List[str]:
    """Plain-text lines for the point-in-time dataset coverage (empty when none)."""
    if not cov or not cov.get("n"):
        return []
    lines = [
        f"{cov['n']} resolved alerts, {cov.get('n_features', 0)} features "
        f"(feature set {cov.get('current_version')}: {cov.get('current_version_share', 0):.0%} of rows)"
    ]
    hz = cov.get("horizons") or {}
    parts = []
    for h in sorted(hz):
        m = hz[h].get("mean_net")
        parts.append(f"{h} candles: {hz[h]['frac']:.0%} labelled" + (f", avg {m:+.2f}%" if m is not None else ""))
    if parts:
        lines.append("Outcome horizons — " + "; ".join(parts))
    if cov.get("mixed_versions"):
        lines.append("Warning: rows from different feature-set versions are mixed; compare models only within one version.")
    if cov.get("leaked_keys"):
        lines.append(f"Warning: {cov['leaked_keys']} post-outcome fields were found among the features and removed.")
    ch = cov.get("cache") or {}
    if ch.get("enabled"):
        if ch.get("verified") is False:
            lines.append("Warning: the stored dataset failed its spot-check and was rebuilt from the archive.")
        else:
            lines.append(
                f"Dataset cache: {ch.get('hit', 0)} rows reused, {ch.get('built', 0)} built this run"
                + (" (spot-check passed)" if ch.get("verified") else "")
                + (f"; not stored ({ch['skipped'].replace('_', ' ')})" if ch.get("skipped") else "") + "."
            )
    weak = cov.get("mostly_empty_features") or []
    if weak:
        lines.append("Mostly-empty features: " + ", ".join(f"{n} ({z:.0%})" for n, z in weak))
    return lines

def _net_ev_audit_lines(a: Optional[Dict[str, Any]]) -> List[str]:
    """Plain-text lines for the net-EV baseline audit (empty when not run)."""
    if not a or not a.get("status"):
        return []
    if a["status"] == "insufficient":
        why = "the model would take too few trades to judge" if a.get("reason") == "model_takes_too_few" else "not enough older/newer trades yet"
        return [f"Not judged: {why} (trained on {a.get('n_train', 0)}, tested on {a.get('n_test', 0)})."]
    m = a.get("mean_net_per_test_trade") or {}
    lines = [
        f"Tested on the newest {a['n_test']} trades after training on {a['n_train']} older ones "
        f"(outcome at {a['horizon']} candles, after costs).",
        "Average result per test trade — model {model:+.3f}%, take-all {take_all:+.3f}%, "
        "key rule {key_rule:+.3f}%, trailing rule {trailing:+.3f}%".format(**{k: m.get(k, 0.0) for k in ("model", "take_all", "key_rule", "trailing")}),
        f"Model takes {a.get('n_taken')} of {a['n_test']}; the trades it takes average {a.get('taken_mean_net')}%.",
        f"Against the best simple rule ({a.get('best_baseline')}): change {a.get('delta_vs_best'):+.3f}% per trade, "
        f"one-sided lower bound {a.get('delta_lo')}.",
    ]
    if a.get("auc") is not None:
        lines.append(f"Ranking quality (AUC) {a['auc']:.2f}; probability skill vs base rate {a.get('brier_skill')}.")

    lines.append(
        "Verdict: VALIDATED — beats every simple rule on unseen newer trades." if a["status"] == "validated"
        else "Verdict: NOT BETTER than a simple rule yet — keep the ML-EV gate off."
    )
    on = a.get("online") or {}
    if on.get("mean_net"):
        mn, vs = on["mean_net"], on.get("online_vs_static") or {}
        rf, orf = on.get("refit_vs_static") or {}, on.get("online_vs_refit") or {}
        lines.append(
            f"Updating as results arrive (newest trades replayed one by one): fixed model {mn['static']:+.3f}%, "
            f"updated after every resolved trade {mn['online']:+.3f}%, retrained every {on.get('refit_every')} trades {mn['refit']:+.3f}% per trade."
        )
        lines.append(
            f"Updated vs fixed: {vs.get('delta'):+.3f}% (lower bound {vs.get('lo')}, {vs.get('status')}); "
            f"retrained vs fixed: {rf.get('delta'):+.3f}% ({rf.get('status')}); updated vs retrained: {orf.get('delta'):+.3f}% ({orf.get('status')})."
        )
        lines.append(
            "Read: if updating beats the fixed model but not the periodic retrain, the existing refresh cycle is enough; "
            "nothing here changes the live model."
        )
    return lines

def _experiment_lines(led: Optional[Dict[str, Any]]) -> List[str]:
    """Plain-text lines for the experiment ledger (empty when nothing recorded)."""
    if not led or not led.get("last"):
        return []
    last = led["last"]
    n = int(last.get("zone_candidates", 0)) + int(last.get("regime_segments", 0))
    note = multiple_testing_note(n)
    lines = [
        f"Candidates examined this cycle: {last.get('zone_candidates', 0)} TP/SL zones, "
        f"{last.get('regime_segments', 0)} regime segments, {last.get('recommendations', 0)} recommendations",
    ]
    if last.get("generated_rules"):
        lines.append(
            f"Plus {last['generated_rules']} self-generated skip-rules proposed this cycle; the evidence bar rises with every rule ever tried."
        )
    if last.get("plan_variants"):
        lines.append(
            f"Plus {last['plan_variants']} stop/target/horizon variants tried per volatility regime "
            f"(chosen on older trades and re-checked on newer ones, so not counted in the chance-hit estimate below)."
        )
    if n:
        lines.append(
            f"With {n} segment/zone tests, about {note['expected_chance_hits']:.1f} would look "
            f"significant at the 5% level by chance alone."
        )
    cum = led.get("cum") or {}
    lines.append(f"Looks since tracking began: {sum(int(v) for v in cum.values())} over {led.get('cycles', 0)} cycles "
                 f"(same data re-examined, not independent tests).")
    if led.get("learner_cum_hypotheses") is not None:
        lines.append(f"Learner playbook hypotheses tried to date: {led['learner_cum_hypotheses']}")
    return lines

def _prediction_track_lines(ptr: Optional[Dict[str, Any]]) -> List[str]:
    """Plain-text lines for the prediction track record (empty when there is nothing to show)."""
    if not ptr or not ptr.get("valid") or not ptr.get("n_with_prediction"):
        return []
    labels = {"ensemble_p": "Ensemble", "market_state_p_win": "ML model", "p_ev_positive": "EV model"}
    status_text = {
        "insufficient": "🟡 too few trades to judge",
        "no_skill": "🔴 no better than the average win rate",
        "skilful_but_miscalibrated": "🟡 beats the average, odds are off",
        "skilful": "🟢 beats the average win rate",
    }
    lines = [f"Alerts with a stored prediction: {ptr['n_with_prediction']} of {ptr['n_rows']}"]
    rows: List[Tuple[str, ...]] = []
    for f, p in (ptr.get("predictors") or {}).items():
        skill = p.get("brier_skill")
        rows.append((
            labels.get(f, f), f"n={p['n']}",
            "skill n/a" if skill is None else f"skill {skill:+.2f}",
            f"calib err {p['ece']:.0%}", status_text.get(p.get("status"), str(p.get("status"))),
        ))
    if rows:
        lines.extend(_table(rows, "lllll"))
    bv = ptr.get("by_verdict") or {}
    if bv:
        lines.append("")
        lines.append("Realised results by verdict at alert time:")
        vrows: List[Tuple[str, ...]] = []
        for v in ("HIGH", "MEDIUM", "LOW"):
            if v in bv:
                e = bv[v]
                vrows.append((v, f"n={e['n']}", f"WR {e['wr']:.0%}",
                              "EV n/a" if e.get("net_ev") is None else f"EV {e['net_ev']:+.2f}%"))
        if vrows:
            lines.extend(_table(vrows, "llll"))
    lines.append("")
    lines.append("Covers alerts that fired only; past accuracy is not a forecast.")
    return lines

def _sec_reasoning_chain(F: Dict[str, Any], cfg) -> List[_Piece]:
    """Roadmap #19 — explicit Brain decision narrative. Trade counts, session,
    calibration, OOS and gate detail are shown in sections 02, 04, 08 and 09, so
    this section carries only what they do not: the verdict and why, the
    recent-vs-earlier win-rate comparison, and the strategy-state reading."""
    out = [_hdr(1, 'BRAIN DECISION — "WHY THIS VERDICT?"')]
    try:
        n = int(F.get("n") or 0)
        net_ev = float(F.get("net_ev") or 0.0)
        gate = F.get("gate") or {}
        ai = F.get("ai") or {}
        conf = str(F.get("conf") or "LOW")

        if F.get("gate_ok") and net_ev > 0 and conf in ("MODERATE", "HIGH"):
            decision = "APPROVED — evidence supports limited parameter change (still via shadow/plan)"
        elif F.get("gate_ok"):
            decision = "MONITOR — gate open but edge not strong enough to change live rules"
        elif n < int(getattr(cfg, "BRAIN_AUDIT_MIN_ROWS_FOR_RECOMMENDATION", 100)):
            decision = "BLOCKED — insufficient sample for any live change"
        elif not gate.get("stability"):
            decision = "BLOCKED — drift detected; freeze parameter changes"
        else:
            decision = "BLOCKED — action gate not satisfied (see ACTION GATE)"
        lines = [f"Decision:  {decision}"]

        recent_wr, older_wr, recent_n = engine.detect_temporal_drift(F.get("rows") or [])
        if recent_wr is not None and recent_n:
            lines.append(
                f"Recent:    WR={float(recent_wr):.0%} (n={recent_n}) "
                f"vs earlier {float(older_wr):.0%}"
            )
        _ss = (ai.get("strategy_state") or {}).get("state")
        _ss_text = {
            "STRATEGY_DEGRADED": "strategy degraded (drop persists within the same regimes)",
            "REGIME_UNDERREPRESENTED": "current regime underrepresented in history — not proof of decay",
            "REGIME_SHIFT": "regime mix shifted to a weaker regime — expected dip",
            "DEGRADED_REGIME_UNKNOWN": "WR fell; ADX data too thin to attribute",
        }.get(_ss or "")
        _drift = "CUSUM drift active" if not gate.get("stability") else ""
        if _drift and _ss_text:
            _drift = f"{_drift} | {_ss_text}"
        elif _ss_text:
            _drift = _ss_text
        if _drift:
            lines.append(f"Drift:     {_drift}")
        out.append(_c("\n".join(lines)))
    except Exception as e:
        out.append(_p(f"Reasoning chain unavailable: {type(e).__name__}"))
    return out

_TRUSTED_MIN_REAL = 30      # real trades an alert key needs before its numbers are "trusted"
_TRUSTED_MIN_SHADOW = 15    # plus this many shadow (counterfactual) trades

def _health_rows(F: Dict[str, Any], cfg) -> List[Tuple[str, str, str]]:
    """(icon, label, detail) rows describing the Brain's own learning
    machinery (data, calibration, models, safety switches), not trading
    results. Reads only what the report already holds. Each check is
    isolated: one failing check shows as unavailable and never hides the rest."""
    ai = F.get("ai") or {}
    rows: List[Tuple[str, str, str]] = []

    def add(icon: str, label: str, detail: str) -> None:
        rows.append((icon, label, detail))

    def guard(label: str, fn) -> None:
        try:
            fn()
        except Exception as e:
            logging.getLogger("macd_bot").warning(f"Brain health: '{label}' check failed: {type(e).__name__}: {e}")
            add("⚪", label, "unavailable (check failed, see logs)")

    def evidence() -> None:
        n, n_sh = int(F.get("n", 0)), len(F.get("shadow_rows") or [])
        days, req = F.get("days"), F.get("req_days")
        # Thin evidence is "still learning", not a fault, so it is amber, never red.
        icon = {"HIGH": "🟢", "MODERATE": "🟡"}.get(str(F.get("conf")), "🟠")
        span = f"{days:.1f} of {req} days" if days is not None and req else "span unknown"
        add(icon, "Evidence base", f"{n} real + {n_sh} shadow outcomes over {span}; confidence {F.get('conf')}")

    def reconciliation() -> None:
        rec = F.get("recon")
        if rec is None:
            add("⚪", "Outcome bookkeeping", "no reconciliation this run")
            return
        def _n(v: Any) -> str:
            return "n/a" if v is None else str(v)
        icon = rec.status.icon
        detail = (
            f"{_n(rec.loaded_by_brain)} loaded by Brain, {_n(rec.pending_count)} pending; "
            f"this run resolved {_n(rec.resolved_this_run)}, archived {_n(rec.archived_this_run)}"
        )
        if rec.resolved_this_run is not None and rec.archived_this_run is not None \
                and rec.resolved_this_run != rec.archived_this_run:
            icon = "🟡" if icon == "🟢" else icon
            detail += f" (gap of {abs(rec.resolved_this_run - rec.archived_this_run)})"
        add(icon, "Outcome bookkeeping", detail)

    def features() -> None:
        df = ai.get("data_foundation")
        if not df or not df.get("n"):
            add("⚪", "Feature data", "dataset not built this run")
            return
        h = int(getattr(cfg, "OUTCOME_LOOKAHEAD_CANDLES", 12))
        hz = df.get("horizons") or {}
        entry = hz.get(h) or hz.get(str(h)) or {}
        frac = entry.get("frac")
        share = float(df.get("current_version_share") or 0.0)
        notes = [f"{df.get('current_version')} on {share:.0%} of {df['n']} rows"]
        if frac is not None:
            notes.append(f"{h}-candle labels {frac:.0%}")
        ok = not df.get("mixed_versions") and not df.get("leaked_keys") and (frac is None or frac >= 0.9)
        if df.get("mixed_versions"):
            notes.append("MIXED feature versions")
        if df.get("leaked_keys"):
            notes.append(f"{df['leaked_keys']} post-outcome fields removed")
        if (df.get("cache") or {}).get("verified") is False:
            ok = False
            notes.append("cache failed spot-check, rebuilt")
        add("🟢" if ok else "🟡", "Feature data", "; ".join(notes))

    def calibration() -> None:
        ece = ai.get("calibration_ece_mean")
        saved = ai.get("calibration_persistence")
        gate = "ON" if getattr(cfg, "ENABLE_CALIBRATION_GATE", False) else "OFF"
        if ece is None:
            add("⚪", "Calibration", f"no curve measured yet; gate {gate}")
            return
        icon = "🟢" if ece <= 0.10 else "🟡" if ece <= 0.20 else "🔴"
        if saved == "FAILED":
            icon = "🔴"
        add(icon, "Calibration", f"error (ECE) {ece:.2f}, lower is better; curve saved: {saved or 'n/a'}; gate {gate}")

    def online() -> None:
        om = ai.get("online_model")
        if not om or not om.get("mode"):
            add("⚪", "Online model", "off or not run this report")
            return
        icon = {"online_better": "🟢", "indistinguishable": "🟡", "batch_better": "🟡"}.get(om.get("status"), "🟠")
        parts = [f"{om['mode']} mode", f"{om.get('n_eval', 0)} paired trades"]
        if om.get("mean_brier_gain") is not None:
            parts.append(f"Brier gain vs batch {om['mean_brier_gain']:+.4f}")
        parts.append(f"passes {om.get('passes', 0)}/{om.get('min_passes', 3)}")
        parts.append("serving" if om.get("promoted") else "not serving")
        parts.append(f"last promoted {_fmt_age(om['last_promoted_at'])}" if om.get("last_promoted_at") else "never promoted")
        if om.get("last_demoted_at"):
            parts.append(f"last demoted {_fmt_age(om['last_demoted_at'])}")
        if om.get("demoted"):
            icon = "🟡"
            parts.append("DEMOTED this run")
        add(icon, "Online model", "; ".join(parts))

    def challenger() -> None:
        if not getattr(cfg, "ENABLE_CHAMPION_CHALLENGER", False):
            add("⚪", "Weight challenger", "off")
            return
        cp = ai.get("challenger_promotion")
        if not cp:
            add("⚪", "Weight challenger", "no challenger waiting")
        elif cp.get("promoted"):
            add("🟢", "Weight challenger", "promoted to champion this run")
        else:
            add("🟡", "Weight challenger", f"not promoted: {str(cp.get('reason', 'n/a')).replace('_', ' ')}")

    def kill_switch() -> None:
        mode = ai.get("kill_switch_mode") or kill_switch_mode(cfg)
        if mode == "off":
            add("⚪", "Kill switch", "off (alerts are not paused by a loss streak or drawdown)")
            return
        ks = ai.get("kill_switch")
        if mode == "shadow":
            log = ai.get("kill_switch_shadow_log") or []
            history = (
                f"; would have fired {len(log)} time(s), last {_fmt_age(log[0].get('ts'))}"
                if log else "; has never fired"
            )
            if not ks:
                add("⚪", "Kill switch", "shadow mode, not evaluated this run" + history)
            elif ks.get("tripped"):
                add("🟠", "Kill switch", f"shadow: WOULD be tripped now ({ks.get('reason', 'n/a')}), alerts still sent" + history)
            else:
                add("🟢", "Kill switch", f"shadow mode, not tripped (streak {ks.get('consecutive_losses', 'n/a')}, drawdown {ks.get('drawdown_pct', 'n/a')}%)" + history)
            return
        if not ks:
            add("⚪", "Kill switch", "armed, not evaluated this run")
        elif ks.get("tripped"):
            add("🔴", "Kill switch", f"TRIPPED: {ks.get('reason', 'n/a')}")
        else:
            add("🟢", "Kill switch", f"armed, not tripped (streak {ks.get('consecutive_losses', 'n/a')}, drawdown {ks.get('drawdown_pct', 'n/a')}%)")

    def drift() -> None:
        if not F.get("n"):
            add("⚪", "Edge-decay alarms", "no outcomes to monitor")
            return
        keys = sorted({r.get("alert") for r in (F.get("recs_list") or []) if r.get("type") == "cusum_drift" and r.get("alert")})
        if keys or F.get("gate", {}).get("stability") is False:
            shown = ", ".join(_pretty_alert(k) for k in keys[:4]) if keys else "see action gate"
            add("🔴", "Edge-decay alarms", f"{len(keys)} alert key(s) flagged: {shown}")
        else:
            add("🟢", "Edge-decay alarms", "none active")

    def disabled() -> None:
        dk = ai.get("disabled_alert_keys")
        if dk is None:
            add("⚪", "Disabled alert keys", "not reported this run")
            return
        cap = int(getattr(cfg, "BRAIN_MAX_DISABLED_KEYS", 6))
        icon = "🟢" if not dk else "🟡" if len(dk) < cap else "🔴"
        names = (": " + ", ".join(_pretty_alert(k) for k in list(dk)[:5])) if dk else ""
        add(icon, "Disabled alert keys", f"{len(dk)} of cap {cap}{names}")

    def trust() -> None:
        alerts = F.get("alerts") or []
        total = len(ALERT_KEYS)
        act_min = int(getattr(cfg, "BRAIN_AUTO_DISABLE_MIN_SAMPLE", 200))
        shadow_n: Dict[str, int] = defaultdict(int)
        for r in (F.get("shadow_rows") or []):
            shadow_n[r.get("alert_key")] += 1
        n_ev = sum(1 for a in alerts if a.get("ev") is not None)
        n_trusted = sum(
            1 for a in alerts
            if a.get("n", 0) >= _TRUSTED_MIN_REAL and shadow_n.get(a.get("key"), 0) >= _TRUSTED_MIN_SHADOW
        )
        n_act = sum(1 for a in alerts if a.get("n", 0) >= act_min)
        icon = "🟠" if n_trusted == 0 else "🟡" if n_trusted * 2 < max(len(alerts), 1) else "🟢"
        add(icon, "Alert keys with evidence",
            f"{len(alerts)} of {total} seen; {n_ev} usable EV (10+ real); "
            f"{n_trusted} trusted ({_TRUSTED_MIN_REAL}+ real and {_TRUSTED_MIN_SHADOW}+ shadow); "
            f"{n_act} ready to auto-act on ({act_min}+ real)")

    for label, fn in (
        ("Evidence base", evidence), ("Outcome bookkeeping", reconciliation), ("Feature data", features),
        ("Calibration", calibration), ("Online model", online), ("Weight challenger", challenger),
        ("Kill switch", kill_switch), ("Edge-decay alarms", drift), ("Disabled alert keys", disabled),
        ("Alert keys with evidence", trust),
    ):
        guard(label, fn)
    return rows

def _health_overall(rows: Sequence[Tuple[str, str, str]]) -> str:
    icons = {r[0] for r in rows}
    if "🔴" in icons:
        return "🔴 NEEDS ATTENTION — at least one item is failing, tripped or alarming (see the red line)."
    if icons & {"🟡", "🟠"}:
        return "🟡 LEARNING — working, but some parts still need more data or a look."
    return "🟢 HEALTHY — every checked part of the learning machinery is working."

def _sec_health(F: Dict[str, Any], cfg) -> List[_Piece]:
    import textwrap
    out = [_hdr(10, '🩺 BRAIN HEALTH — "IS THE LEARNING MACHINERY WORKING?"')]
    out.append(_p("This checks the Brain's own machinery (data, calibration, models, safety switches), not trading results."))
    rows = _health_rows(F, cfg)
    lines: List[str] = []
    for icon, label, detail in rows:
        lines.append(f"{icon} {label}")
        lines.extend(textwrap.wrap(detail, 38, initial_indent="   ", subsequent_indent="   "))
    out.extend(_c_split(lines))
    out.append(_p("OVERALL: " + _health_overall(rows)))
    return out

_REPORT_SECTIONS = (
    ("BRAIN DECISION", _sec_reasoning_chain),
    ("EXECUTIVE SUMMARY", _sec_summary),
    ("WHAT TO DO NOW", _sec_do_now),
    ("PROFITABILITY", _sec_profit),
    ("LOSS DIAGNOSIS", _sec_loss),
    ("POSITIVE SIGNS", _sec_positive),
    ("ALERT SCORECARD", _sec_scorecard),
    ("SESSION ANALYSIS", _sec_sessions),
    ("ACTION GATE", _sec_gate),
    ("BRAIN HEALTH", _sec_health),
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
            divider = _p("▼ CONTEXT (sessions · gate) ▼")
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
            out.append("---\n\n*Context: sessions and action gate.*")
    out.append("---\n\n*End of Brain report.*")
    return "\n\n".join(out) + "\n"

def build_brain_report(recs: Dict[str, Any], cfg) -> List[str]:
    """Convenience wrapper: sections -> Telegram messages."""
    sections, stamp = build_brain_report_sections(recs, cfg)
    return render_report_messages(sections, stamp)
