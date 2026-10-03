"""brain_report — Brain report layer: storage-key constants, fact collection and the 16-section Telegram/Markdown renderers. Pure functions over the recommendation dict; no engine state, no Redis.

Split out of the original module without logic changes; the original
module remains as a facade re-exporting every name."""
from __future__ import annotations
import logging
import unicodedata
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple
from datetime import datetime, timedelta, timezone
from brain_audit import DataCoverage, HealthStatus, get_audit
import threshold_engine as engine
from alerts import escape_markdown_v2
from alert_registry import alert_family_of as _registry_family_of, pretty_alert as _pretty_alert

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
    prof = _profit_status(net_ev, n)
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
            lines.append(
                f"• {s.get('label', '?')}: {F['net_ev']:+.2f}% → {s.get('ev', 0.0):+.2f}% "
                f"(n={s.get('n', 0)}) — {verdict}"
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
        ("Net EV", f"{net_ev:+.2f}%",
         "🟢 Positive" if net_ev > 0.05 else "🟡 Flat" if net_ev > -0.05 else "🔴 Negative"),
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
    out = [_hdr(5, '🔎 LOSS DIAGNOSIS — "WHERE ARE WE FALTERING?"')]
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
    rows = [(f"{'🟢' if g.get(k) else '🔴'} {lab}:", "PASS" if g.get(k) else "FAIL")
            for k, lab in labels]

    out.extend(_c_split(_table(rows, "ll")))
    out.append(_p("CONFIDENCE BREAKDOWN\n\n" + "\n".join(
        f"{axis.ljust(12)}{tier}  ({detail})" for axis, tier, detail in _confidence_breakdown(F, cfg)
    )))
    out.append(_p(
        "OVERALL:\n\n"

        + ("🟢 BRAIN ACTION GATE = PASSED\n\nMeaning:\n\n\"The Brain's evidence is strong enough to "
           "recommend specific changes to the live strategy.\""
           if F["gate_ok"] else
           "🔴 BRAIN ACTION GATE = BLOCKED\n\nMeaning:\n\n\"The Brain may analyse and recommend what "
           "to investigate, but it is not sufficiently confident to modify the live strategy.\"")
    ))
    return out

def _sec_reasoning_chain(F: Dict[str, Any], cfg) -> List[_Piece]:
    """Roadmap #19 — explicit Brain decision narrative."""
    out = [_hdr(1, 'BRAIN DECISION — "WHY THIS VERDICT?"')]
    try:
        n = int(F.get("n") or 0)
        wr = float(F.get("wr") or 0.0)
        net_ev = float(F.get("net_ev") or 0.0)
        days = F.get("days")
        gate = F.get("gate") or {}
        ai = F.get("ai") or {}
        conf = str(F.get("conf") or "LOW")

        market_line = "regime tags limited in this window"
        sessions = F.get("sessions") or []
        # session_breakdown() returns (session, wr, n) tuples, worst first.
        if sessions and isinstance(sessions[0], (tuple, list)) and len(sessions[0]) >= 3:
            w_name, w_wr, w_n = sessions[0][0], float(sessions[0][1]), int(sessions[0][2])
            market_line = f"weakest session={w_name} (WR={w_wr:.0%}, n={w_n})"
        hist_line = (
            f"n={n} trades over {_fmt_days(days)} | "
            f"WR={wr:.0%} | Net EV/trade={net_ev:+.2f}%"
        )

        recent_wr, older_wr, recent_n = engine.detect_temporal_drift(F.get("rows") or [])
        if recent_wr is not None and recent_n:
            recent_line = (
                f"Recent WR={float(recent_wr):.0%} (n={recent_n}) "
                f"vs earlier {float(older_wr):.0%}"
            )
        else:
            recent_line = "recent window not separately scored this report"
        ece = ai.get("calibration_ece_mean")
        if ece is None:
            calib_line = "no calibration curve yet"
        else:
            ece_f = float(ece)
            tag = "GOOD" if ece_f < 0.08 else "WATCH" if ece_f < 0.15 else "POOR"
            calib_line = f"mean-per-alert ECE={ece_f:.3f} — {tag}"

        oos_line = "PASS" if gate.get("oos_prediction") else "FAIL / unavailable"
        drift_line = "NONE detected" if gate.get("stability") else "CUSUM drift active"
        _ss = (ai.get("strategy_state") or {}).get("state")
        _ss_text = {
            "STRATEGY_DEGRADED": "strategy degraded (drop persists within the same regimes)",
            "REGIME_UNDERREPRESENTED": "current regime underrepresented in history — not proof of decay",
            "REGIME_SHIFT": "regime mix shifted to a weaker regime — expected dip",
            "DEGRADED_REGIME_UNKNOWN": "WR fell; ADX data too thin to attribute",
            "STABLE": "no significant WR drop",
        }.get(_ss or "")
        if _ss_text:
            drift_line = f"{drift_line} | {_ss_text}"

        if F.get("gate_ok") and net_ev > 0 and conf in ("MODERATE", "HIGH"):
            decision = "APPROVED — evidence supports limited parameter change (still via shadow/plan)"
        elif F.get("gate_ok"):
            decision = "MONITOR — gate open but edge not strong enough to change live rules"
        elif n < int(getattr(cfg, "ACTION_GATE_MIN_ROWS", 100)):
            decision = "BLOCKED — insufficient sample for any live change"
        elif not gate.get("stability"):
            decision = "BLOCKED — drift detected; freeze parameter changes"
        else:
            decision = "BLOCKED — action gate not satisfied (see ACTION GATE)"

        lines = [
            f"Market:      {market_line}",
            f"Historical:  {hist_line}",
            f"Recent:      {recent_line}",
            f"Calibration: {calib_line}",
            f"OOS:         {oos_line}",
            f"Drift:       {drift_line}",
            f"Decision:    {decision}",
        ]
        out.append(_c("\n".join(lines)))
        out.append(_p(
            "Hard signal rules are unchanged. This block only states the Brain's "
            "quality assessment and whether any plan is allowed to proceed."
        ))
    except Exception as e:
        out.append(_p(f"Reasoning chain unavailable: {type(e).__name__}"))
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
