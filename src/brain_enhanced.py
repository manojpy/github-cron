#!/usr/bin/env python3
"""brain_enhanced.py — Prescriptive Brain (Roadmap Phases 1.5-6)"""

from __future__ import annotations
import asyncio
import json
import logging
import random
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
import os
from pathlib import Path

from brain_audit import (
    DataCoverage, HealthStatus, RecommendationTier, get_audit, reset_audit,
)

from archive_reader import load_archived_outcomes
from bot_config import cfg, CONFLUENCE_WEIGHTS, CONFIG_OVERRIDE_ALLOWED_FIELDS, json_dumps, json_loads
from state import RedisKeyPrefix, RedisStateStore
from brain import BrainEngine as BaseBrainEngine, _extract_p_value_for_fdr
import threshold_engine as engine
from outcome_storage import OUTCOME_SCHEMA_VERSION

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
#  PLAIN-ENGLISH PROFIT ACTION PLAN (layman-friendly report layer)
# ══════════════════════════════════════════════════════════════════════

def build_profit_action_plan(recs: Dict[str, Any], cfg) -> List[str]:
    """Translate Brain findings into a plain-English, copy-paste action plan.
    Returns a list of Telegram-ready messages (each within the 4096-char limit)."""
    rows = recs.get("_real_rows", []) or []
    cfg_patch = recs.get("config_patch", []) or []
    ai = recs.get("ai_metrics", {}) or {}
    sections: List[str] = []
    failed_sections: List[str] = []

    # ══════════════════════════════════════════════════════════════════
    #  DATA QUALITY HEADER (always first, before any analysis)
    # ══════════════════════════════════════════════════════════════════
    audit = get_audit()
    header_lines = audit.build_data_quality_header()

    # Action gate summary
    action_gate = ai.get("action_gate", {}) or {}
    if action_gate:
        header_lines.append("")
        header_lines.extend(audit.build_action_gate_summary(action_gate))

    # Schema migration advisory
    archive_stats = recs.get("_archive_stats", {})
    if archive_stats:
        advisory = audit.schema_migration_advisory(
            current_version=OUTCOME_SCHEMA_VERSION,
            stale_count=(
                archive_stats.get("dropped_unmigratable", 0)
                + archive_stats.get("migrated_forward", 0)
            ),
            migrated_count=archive_stats.get("migrated_forward", 0),
            unmigratable_count=archive_stats.get("dropped_unmigratable", 0),
            total_archive_rows=archive_stats.get("lines_total", 0),
        )
        if advisory:
            header_lines.append(f"   {advisory}")

    sections.append("\n".join(header_lines))

    _GATE_CHECK_LABELS = {
        "data_quality": "fewer than 100 trades logged",
        "oos_prediction": "didn't hold up in out-of-sample testing",
        "profitability": "not confident enough of net profit (P>0 or EV floor)",
        "stability": "a drift alarm is currently active on one of your alerts",
        "risk": "recent drawdown exceeds the kill-switch budget",
        "execution": "fee/slippage assumptions aren't configured",
    }
    action_gate = ai.get("action_gate", {}) or {}
    _failing_checks = [
        label for key, label in _GATE_CHECK_LABELS.items()
        if action_gate.get(key) is False
    ]
    gate_block_reason = "; ".join(_failing_checks) if _failing_checks else "insufficient evidence"

    n = len(rows)
    wins = sum(1 for r in rows if r["win"])
    wr = wins / n if n else 0.0
    target = getattr(cfg, "MIN_WIN_RATE", 0.55)
    kill_thr = getattr(cfg, "BRAIN_ALERT_DISABLE_THRESHOLD_WR", 0.40)
    net_ev = ai.get("net_ev", 0.0) or 0.0

    # Computed once here (was previously computed a second time, later,
    # in the ENTRY GATE THRESHOLD section) so BOTTOM LINE can show
    # current-vs-expected WR without a duplicate call.
    try:
        rec_thr = engine.recommend_threshold(
            rows, target_winrate=target, min_sample=getattr(cfg, "MIN_WIN_RATE_SAMPLE", 20)
        )
    
    except Exception as e:
        _report_section_failed(failed_sections, "ENTRY BAR RECOMMENDATION", e)
        rec_thr = {"valid": False}
    rec_thr_available = (
        rec_thr.get("valid") and rec_thr.get("recommended")
        and rec_thr["recommended"] > cfg.CONFLUENCE_MIN_ABS_SCORE
    )
    # ── BOTTOM LINE ────────────────────────────────────────────────────
    try:
        buy_wr, buy_n, sell_wr, sell_n = engine.direction_split(rows)    
        ev_obj = engine.ev_first_objective(rows, min_sample=10) if rows else None    
        verdict = audit.qualify_verdict(
            net_ev=net_ev, wr=wr, n=n,
            p_ev_positive=ev_obj.get("p_ev_positive", 0) if ev_obj else 0.0,
        )
        dir_note = ""
        if buy_wr is not None and sell_wr is not None and buy_n >= 5 and sell_n >= 5:
            if sell_wr < buy_wr - 0.15:
                dir_note = f"Your SELL alerts win only {sell_wr:.0%} vs BUY {buy_wr:.0%} — the sell side is dragging you down."
            elif buy_wr < sell_wr - 0.15:
                dir_note = f"Your BUY alerts win only {buy_wr:.0%} vs SELL {sell_wr:.0%} — the buy side is dragging you down."
        
        ev_note = f"Net per trade: {net_ev:+.2f}% after fees" + (" — negative ❌." if net_ev < 0 else " — positive ✅.")
        
        wr_note = ""
        if (
            rec_thr_available
            and audit.max_recommendation_tier("threshold_recommendation")
            == RecommendationTier.ACTIONABLE
        ):
            rec_wr_val = rec_thr.get("rec_wr", 0)
            wr_note = (
                f"\nHistorical filtered WR at the higher bar: {rec_wr_val:.0%} "
                f"(in-sample estimate, not a forecast; see 🚪 below)."
            )
        low_data = ""  # verdict text already carries the confidence caveat
        sections.append(
            f"🧠 PROFIT ACTION PLAN\n📊 Based on {n} trades\n\n🎯 BOTTOM LINE\n{verdict}"
            + (f"\n{dir_note}" if dir_note else "")
            + f"\n{ev_note}{wr_note}{low_data}"
        )
    except Exception as e:
        _report_section_failed(failed_sections, "BOTTOM LINE", e)
        
    # ── PER-ALERT HEALTH ───────────────────────────────────────────────
    try:
        stats = engine.per_alert_breakdown(rows, min_sample=1)  # (ak, wr, n, avg_score)
        groups: Dict[str, List[str]] = {"🔴": [], "🟡": [], "🟢": [], "⚪": []}

        # Group rows by alert_key for EV computation
        rows_by_alert: Dict[str, list] = defaultdict(list)
        for r in rows:
            rows_by_alert[r["alert_key"]].append(r)

        needs_data: List[Tuple[str, float, int]] = []
        for ak, awr, cnt, _avg in stats:
            if cnt < 10:
                needs_data.append((ak, awr, cnt))
                continue

            ak_rows = rows_by_alert.get(ak, [])
            ak_ev = engine.ev_first_objective(ak_rows, min_sample=10) if ak_rows else None
            ak_net_ev = ak_ev.get("net_ev", 0) if ak_ev and ak_ev.get("valid") else 0
            ak_p_ev = ak_ev.get("p_ev_positive", 0) if ak_ev and ak_ev.get("valid") else 0

            if ak_net_ev > 0 and ak_p_ev >= 0.85:
                groups["🟢"].append(
                    f"🟢 {ak}: EV {ak_net_ev:+.2f}% (P>0: {ak_p_ev:.0%}), "
                    f"WR {awr:.0%} (n={cnt}) — profitable, keep it"
                )
            elif ak_net_ev > 0:
                groups["🟡"].append(
                    f"🟡 {ak}: EV {ak_net_ev:+.2f}% but P(EV>0)={ak_p_ev:.0%}, "
                    f"WR {awr:.0%} (n={cnt}) — thin evidence, monitor"
                )
            elif awr >= kill_thr:
                groups["🟡"].append(
                    f"🟡 {ak}: EV {ak_net_ev:+.2f}%, WR {awr:.0%} (n={cnt}) "
                    f"— below target; tighten its filter"
                )
            else:
                _lo, _hi, _ = engine.wilson_ci(int(awr * cnt), cnt)
                _conf = engine.confidence_label(cnt, _lo, _hi)
                groups["🔴"].append(
                    f"💀 {ak}: EV {ak_net_ev:+.2f}%, WR {awr:.0%} (n={cnt}, "
                    f"confidence: {_conf}) — negative EV, consider disabling"
                )
        titles = {
            "🔴": "NEGATIVE EV — REVIEW FOR DISABLE",
            "🟡": "MIXED / MONITOR",
            "🟢": "POSITIVE EV — KEEP",
        }
        block = "🚦 YOUR ALERTS — EVIDENCE BY BUCKET"
        for bucket in ("🔴", "🟡", "🟢"):
            if groups[bucket]:
                block += f"\n\n{titles[bucket]}:\n" + "\n".join(groups[bucket])

        if needs_data:
            needs_data.sort(key=lambda t: t[2], reverse=True)
            shown = needs_data[:5]
            nd_lines = [f"⚪ {ak}: {awr:.0%} WR (n={cnt})" for ak, awr, cnt in shown]
            remainder = len(needs_data) - len(shown)
            nd_block = f"NEED MORE DATA ({len(needs_data)} alerts; closest to a verdict shown):\n" + "\n".join(nd_lines)
            if remainder > 0:
                nd_block += f"\n…and {remainder} more with too few trades to list."
            block += f"\n\n{nd_block}"
        sections.append(block)
    except Exception as e:
        _report_section_failed(failed_sections, "PER-ALERT HEALTH", e)
        
    # ── BLOCKED BY WIN-RATE FILTER (shadow, not dispatched) ─────────────
    try:

        # Shadow rows also come from the confluence, OOD, calibration,
        # portfolio-heat and brain-disabled paths (tagged rejection_reason);
        # only the win-rate filter's own rejections belong in this section.
        shadow_rows = [
            r for r in (recs.get("_shadow_rows", []) or [])
            if r.get("rejection_reason") in (None, "win_rate_filter")
        ]
        if not shadow_rows:
            sections.append(
                "🚫 BLOCKED BY WIN-RATE FILTER\n"
                "   Shadow mode is off — no data on what the filter is rejecting.\n"
                "   Set BRAIN_SHADOW_MODE=true to see whether rejections would have won."
            )
        else:
            target = getattr(cfg, "MIN_WIN_RATE", 0.55)
            shadow_stats = engine.per_alert_breakdown(shadow_rows, min_sample=1)
            dropped_lines = []
            over_blocking = 0
            for ak, awr, cnt, _avg in shadow_stats:
                if cnt < 5:
                    continue
                if awr >= target:
                    over_blocking += 1
                    dropped_lines.append(
                        f"  ⚠️ {ak}: {cnt} blocked, shadow WR {awr:.0%} "
                        f"(ABOVE {target:.0%} target — filter may be over-blocking)"
                    )
                elif awr < target * 0.75:
                    dropped_lines.append(
                        f"  ✅ {ak}: {cnt} blocked, shadow WR {awr:.0%} "
                        f"(filter correctly rejected)"
                    )
                else:
                    dropped_lines.append(
                        f"  👻 {ak}: {cnt} blocked, shadow WR {awr:.0%}"
                    )
            if dropped_lines:
                header = "🚫 BLOCKED BY WIN-RATE FILTER (sent to shadow, not dispatched)"
                if over_blocking:
                    header += (
                        f"\n   ⚠️ {over_blocking} alert(s) blocked at or above target — "
                        f"review MIN_WIN_RATE / MIN_WIN_RATE_SAMPLE for those keys."
                    )          
                sections.append(header + "\n" + "\n".join(dropped_lines[:10]))
    except Exception as e:
        _report_section_failed(failed_sections, "BLOCKED BY WIN-RATE FILTER", e)
        
    # ── GATE IMPACT (shadow, per-gate counterfactual EV) ────────────────
    try:
        shadow_rows = recs.get("_shadow_rows", []) or []
        gate_groups: Dict[str, list] = defaultdict(list)
        for r in shadow_rows:
            reason = r.get("rejection_reason")
            if reason:
                gate_groups[reason].append(r)

        min_sample = getattr(cfg, "MIN_WIN_RATE_SAMPLE", 20)
        gate_lines = []

        for reason, grows in sorted(gate_groups.items()):
            ev = engine.ev_first_objective(grows, min_sample=min_sample)

            if not ev.get("valid"):
                gate_lines.append(
                    f"⚪ {reason}: n={len(grows)} — below min sample ({min_sample}), no verdict yet"
                )
                continue

            net_ev = ev.get("net_ev", 0.0)
            p_ev = ev.get("p_ev_positive", 0.0)

            verdict = (
                "would likely have HELPED"
                if net_ev > 0 and p_ev >= 0.65
                else "correctly filtering losers"
            )

            gate_lines.append(
                f"{'🟡' if net_ev > 0 else '🟢'} {reason}: rejected {len(grows)} signals, "
                f"their hypothetical net EV was {net_ev:+.2f}% (P>0: {p_ev:.0%}) — {verdict}"
            )

        if gate_lines:
            sections.append(
                "🚧 GATE IMPACT — WHAT EACH FILTER IS COSTING/SAVING YOU\n"
                + "\n".join(gate_lines)
            )
    except Exception as e:
        _report_section_failed(failed_sections, "GATE IMPACT", e)
        
    # ── CONFLUENCE WEIGHT CHANGES ──────────────────────────────────────
    blocked_lines: List[str] = []
    try:
        weight_lines: List[str] = []
        for p in cfg_patch:
            if p.get("path") != "CONFLUENCE_WEIGHTS":
                continue
            if p.get("_blocked_by_action_gate"):
                blocked_lines.append(f"🔬 CONFLUENCE_WEIGHTS: candidate changes — blocked: {gate_block_reason}")
                continue
            cur = p.get("current", {}) or {}
            sug = p.get("suggested", {}) or {}
            for vote, new_w in sug.items():
                old_w = cur.get(vote, CONFLUENCE_WEIGHTS.get(vote, 0.0))
                if abs(new_w - old_w) < 0.05:
                    continue
                if new_w > old_w:
                    weight_lines.append(f"⬆️ {vote}: {old_w:.1f} → {new_w:.1f}  (this vote predicts wins — give it more power)")
                else:
                    weight_lines.append(f"⬇️ {vote}: {old_w:.1f} → {new_w:.1f}  (this vote hurts accuracy — reduce its power)")
        if weight_lines:
            sections.append(
                "⚖️ CONFLUENCE WEIGHTS — CHANGE THESE\n"
                "(How much each signal counts toward the entry gate)\n\n" + "\n".join(weight_lines)
            )
        else:
            sections.append("⚖️ CONFLUENCE WEIGHTS\nNo safe weight changes yet — need more trade history before the Brain will move them.")
    except Exception as e:
        _report_section_failed(failed_sections, "CONFLUENCE WEIGHT CHANGES", e)

    # ── INDICATOR SETTING CHANGES ──────────────────────────────────────
    try:
        setting_lines: List[str] = []
        for p in cfg_patch:
            if p.get("path") == "CONFLUENCE_WEIGHTS":
                continue
            cur, sug = p.get("current"), p.get("suggested")
            if cur is None or sug is None:
                continue
            if p.get("_blocked_by_action_gate"):
                # Not vetted by the action gate — don't present it as a
                # ready change or let it into the copy-paste block below.
                blocked_lines.append(f"🔬 {p['path']}: candidate {sug} — blocked: {gate_block_reason}")
                continue
            setting_lines.append(f"🔧 {p['path']}: {cur} → {sug}\n   Why: {p.get('reason', 'data-driven optimum')}")
        if setting_lines:
            sections.append("🎚️ INDICATOR SETTINGS — CHANGE THESE\n\n" + "\n".join(setting_lines))
        if blocked_lines:    
            sections.append("🔬 UNDER REVIEW — not confident enough to apply yet\n" + "\n".join(blocked_lines))
    except Exception as e:
        _report_section_failed(failed_sections, "INDICATOR SETTING CHANGES", e)

    # ── ENTRY GATE THRESHOLD ───────────────────────────────────────────
    gate_rec = None
    try:
        if rec_thr_available:
            tier = audit.max_recommendation_tier("threshold_recommendation")
            rec_wr_val = rec_thr.get("rec_wr", 0)

            if tier == RecommendationTier.CANDIDATE:
                sections.append(
                    f"🧪 ENTRY BAR — SIMULATION ONLY (NOT VALIDATED)\n"
                    f"CONFLUENCE_MIN_ABS_SCORE: {cfg.CONFLUENCE_MIN_ABS_SCORE:.1f} → "
                    f"{rec_thr['recommended']:.1f}\n"
                    f"   ⚠️ Historical simulation on {n} trades, NOT a validated forecast.\n"
                    f"   Historical filtered WR: {rec_wr_val:.0%} (in-sample estimate)\n"
                    f"   Do not apply until sample ≥100 and walk-forward passes."
                )
            elif tier == RecommendationTier.ACTIONABLE:
                gate_rec = rec_thr
                if rec_wr_val > wr:
                    outcome_line = f"lifts expected WR to ~{rec_wr_val:.0%} (from {wr:.0%})"
                else:
                    outcome_line = (
                        f"expected WR is ~{rec_wr_val:.0%} — *below* your current {wr:.0%}. "
                        f"This trades hit-rate for a better EV/R:R profile"
                    )
                sections.append(
                    f"🚪 ENTRY BAR — RAISE IT\n"
                    f"CONFLUENCE_MIN_ABS_SCORE: {cfg.CONFLUENCE_MIN_ABS_SCORE:.1f} → "
                    f"{rec_thr['recommended']:.1f}\n"
                    f"   This alone filters out {rec_thr.get('dropped', 0)} weak trades "
                    f"({rec_thr.get('dropped_pct', 0):.0%}) and {outcome_line}."
                )
            else:
                # DESCRIPTIVE or STATISTICAL tier — just mention it
                sections.append(
                    f"📊 ENTRY BAR (observation only)\n"
                    f"Current: {cfg.CONFLUENCE_MIN_ABS_SCORE:.1f} | "
                    f"Historical simulation suggests: {rec_thr['recommended']:.1f}\n"
                    f"   Insufficient data for a recommendation. Accumulate more outcomes."
                )
    except Exception as e:
        _report_section_failed(failed_sections, "ENTRY GATE THRESHOLD", e)

    # ── COPY-PASTE CONFIG BLOCK ────────────────────────────────────────
    # Kept OUT of `sections` so it can be emitted as a real Telegram code
    # block. If we ran it through escape_markdown_v2 like every other
    # section, `CONFLUENCE_WEIGHTS` becomes `CONFLUENCE\_WEIGHTS`, every
    # `{` becomes `\{` etc., and the user cannot paste the result into a
    # JSON file. Inside a MarkdownV2 code block, only ` and \ need escaping.
    json_block: Optional[str] = None
    try:   
        json_changes: Dict[str, Any] = {}
        for p in cfg_patch:
            if p.get("_blocked_by_action_gate"):
                continue  # not vetted — never let it into the paste-ready block
            if p.get("path") == "CONFLUENCE_WEIGHTS":
                json_changes["CONFLUENCE_WEIGHTS"] = p.get("suggested", {})
            elif p.get("current") is not None and p.get("suggested") is not None:
                json_changes[p["path"]] = p["suggested"]
        if gate_rec is not None:
            json_changes["CONFLUENCE_MIN_ABS_SCORE"] = round(gate_rec["recommended"], 1)
        if json_changes:
            raw_json = json.dumps(json_changes, indent=1)
            code_safe = raw_json.replace("\\", "\\\\").replace("`", "\\`")
            
            json_block = "```json\n" + code_safe + "\n```"
    except Exception as e:
        _report_section_failed(failed_sections, "COPY-PASTE CONFIG BLOCK", e)

    # ── BEST / WORST CONDITIONS ────────────────────────────────────
    try:
        pair_stats = engine.per_pair_breakdown(rows, min_sample=5)  # worst-first
        if len(pair_stats) >= 2:
            worst, best = pair_stats[0], pair_stats[-1]
            # FIX (Priority 8): Qualify with confidence label instead of
            # raw WR ranking. A 70% WR on n=5 is NOT stronger evidence
            # than 60% WR on n=150.
            best_lo, best_hi, _ = engine.wilson_ci(
                int(best[1] * best[2]), best[2]
            )
            worst_lo, worst_hi, _ = engine.wilson_ci(
                int(worst[1] * worst[2]), worst[2]
            )
            best_conf = engine.confidence_label(best[2], best_lo, best_hi)
            worst_conf = engine.confidence_label(worst[2], worst_lo, worst_hi)
            line = (f"🌍 WHERE YOU WIN & LOSE\n"
                    f"🏆 Best: {best[0]} at {best[1]:.0%} WR "
                    f"(n={best[2]}, confidence: {best_conf})\n"
                    f"💀 Worst: {worst[0]} at {worst[1]:.0%} WR "
                    f"(n={worst[2]}, confidence: {worst_conf})")
            if worst_conf in ("LOW", "MEDIUM"):
                line += " — insufficient evidence to remove; monitor"
            else:
                line += " — consider removing this pair"
            sess = engine.session_breakdown(rows, min_sample=5)
            if len(sess) >= 2:
                line += (f"\n⏰ Best session: {sess[-1][0]} ({sess[-1][1]:.0%}, n={sess[-1][2]}) "
                         f"| Worst: {sess[0][0]} ({sess[0][1]:.0%}, n={sess[0][2]})")
            
            sections.append(line)
    except Exception as e:
        _report_section_failed(failed_sections, "BEST/WORST CONDITIONS", e)

    if failed_sections:
        sections.append(
            "⚠️ REPORT SECTIONS UNAVAILABLE (analysis error — see logs):\n   "
            + ", ".join(failed_sections)
        )

    if not sections and json_block is None:
        return []

    # ── Pack sections into <4096-char messages ─────────────────────────
    msgs: List[str] = []
    cur = ""
    for s in sections:
        if len(cur) + len(s) + 2 > 3500:
            if cur:
                msgs.append(cur)
            cur = s
        else:
            cur = (cur + "\n\n" + s) if cur else s
    if cur:
        msgs.append(cur)

    escaped_msgs = [escape_markdown_v2(m) for m in msgs]

    if json_block is not None:
        header = escape_markdown_v2("📋 COPY-PASTE INTO config_macd.json")
        candidate = header + "\n" + json_block
        if escaped_msgs and len(escaped_msgs[-1]) + len(candidate) + 2 <= 3900:
            escaped_msgs[-1] = escaped_msgs[-1] + "\n\n" + candidate
        else:
            escaped_msgs.append(candidate)

    return escaped_msgs

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
            "data_quality": len(real_rows) >= 100,
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
        # ══════════════════════════════════════════════════════════════════
        #  BRAIN AUDIT LAYER — initialize and validate data population
        # ═══════════════════════════════════════���══════════════════════════  
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

                # ── Shadow out-of-sample veto ──────────────────────
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

        # ── Config version hash ──────────────────────────────────────────
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
                        f"{'…' if len(_below_floor_drift) > 5 else ''})"
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

    async def generate_report(self, pairs, telegram_queue, logger_run) -> bool:
        """Override: send ONLY the plain-English action plan (no jargon), then store for application."""
        try:
            recs = await self.generate_recommendations()

            # Build and send the plain-English action plan
            plan_messages = build_profit_action_plan(recs, cfg)  # already MarkdownV2-escaped
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
