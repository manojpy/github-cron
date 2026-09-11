#!/usr/bin/env python3
"""brain_enhanced.py — Prescriptive Brain (Roadmap Phases 1.5-6)"""

from __future__ import annotations
import asyncio
import json
import logging
import time
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional
import os
from pathlib import Path

from archive_reader import load_archived_outcomes
from bot_config import cfg, CONFLUENCE_WEIGHTS, CONFIG_OVERRIDE_ALLOWED_FIELDS, json_dumps, json_loads
from state import RedisKeyPrefix, RedisStateStore
from brain import BrainEngine as BaseBrainEngine, _extract_p_value_for_fdr
import threshold_engine as engine
from threshold_engine import (
    optimize_vote_weights, conditional_performance,
    interaction_miner, simulate_config_change, regime_profile_optimizer,
    hash_config_state, score_actionability, compare_config_versions,
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

# ══════════════════════════════════════════════════════════════════════
#  PLAIN-ENGLISH PROFIT ACTION PLAN (layman-friendly report layer)
# ══════════════════════════════════════════════════════════════════════
_TG_ESCAPE = re.compile(r'[_*\[\]()~`>#+\-=|{}.!]')


def _tg(x: Any) -> str:
    """MarkdownV2-escape so TelegramQueue.send() never rejects the message."""
    return _TG_ESCAPE.sub(r'\\\g<0>', str(x))

def build_profit_action_plan(recs: Dict[str, Any], cfg) -> List[str]:
    """Translate Brain findings into a plain-English, copy-paste action plan.
    Returns a list of Telegram-ready messages (each within the 4096-char limit)."""
    rows = recs.get("_real_rows", []) or []
    cfg_patch = recs.get("config_patch", []) or []
    ai = recs.get("ai_metrics", {}) or {}
    sections: List[str] = []

    n = len(rows)
    wins = sum(1 for r in rows if r["win"])
    wr = wins / n if n else 0.0
    target = getattr(cfg, "MIN_WIN_RATE", 0.55)
    star = getattr(cfg, "BRAIN_STAR_ALERT_WR", 0.70)
    kill_thr = getattr(cfg, "BRAIN_ALERT_DISABLE_THRESHOLD_WR", 0.40)
    net_ev = ai.get("net_ev", 0.0) or 0.0

    # ── BOTTOM LINE ────────────────────────────────────────────────────
    try:
        buy_wr, buy_n, sell_wr, sell_n = engine.direction_split(rows)
        if wr >= target:
            verdict = f"✅ You're WINNING at {wr:.0%} (target {target:.0%}). Keep what works; tweaks below push it higher."
        else:
            verdict = f"⚠️ You're LOSING at {wr:.0%} (target {target:.0%})."
        dir_note = ""
        if None not in (buy_wr, sell_wr) and buy_n >= 5 and sell_n >= 5:
            if sell_wr < buy_wr - 0.15:
                dir_note = f"Your SELL alerts win only {sell_wr:.0%} vs BUY {buy_wr:.0%} — the sell side is dragging you down."
            elif buy_wr < sell_wr - 0.15:
                dir_note = f"Your BUY alerts win only {buy_wr:.0%} vs SELL {sell_wr:.0%} — the buy side is dragging you down."
        ev_note = f"Net per trade: {net_ev:+.2f}% after fees" + (" — negative ❌." if net_ev < 0 else " — positive ✅.")
        low_data = f"\nℹ️ Only {n} trades so far — treat these as strong hints, not certainties." if n < 100 else ""
        sections.append(
            f"🧠 PROFIT ACTION PLAN\n📊 Based on {n} trades\n\n🎯 BOTTOM LINE\n{verdict}"
            + (f"\n{dir_note}" if dir_note else "")
            + f"\n{ev_note}{low_data}"
        )
    except Exception:
        pass

    # ── PER-ALERT HEALTH ───────────────────────────────────────────────
    try:
        stats = engine.per_alert_breakdown(rows, min_sample=1)  # (ak, wr, n, avg_score)
        groups: Dict[str, List[str]] = {"🔴": [], "🟡": [], "🟢": [], "⚪": []}
        for ak, awr, cnt, _avg in stats:
            if cnt < 10:
                groups["⚪"].append(f"⚪ {ak}: {awr:.0%} WR (n={cnt}) — not enough trades yet to judge")
            elif awr >= star:
                groups["🟢"].append(f"🟢 {ak}: {awr:.0%} WR (n={cnt}) — star performer, keep it")
            elif awr >= target:
                groups["🟢"].append(f"🟢 {ak}: {awr:.0%} WR (n={cnt}) — profitable, keep it")
            elif awr >= kill_thr:
                groups["🟡"].append(f"🟡 {ak}: {awr:.0%} WR (n={cnt}) — below {target:.0%} target; raise its required score or tighten its filter")
            else:
                groups["🔴"].append(f"🔴 {ak}: {awr:.0%} WR (n={cnt}) — losing money, disable it now")
        titles = {"🔴": "DISABLE THESE NOW", "🟡": "IMPROVE THESE", "🟢": "KEEP THESE", "⚪": "NEED MORE DATA"}
        block = "🚦 YOUR ALERTS — WHAT TO DO WITH EACH"
        for e in ("🔴", "🟡", "🟢", "⚪"):
            if groups[e]:
                block += f"\n\n{titles[e]}:\n" + "\n".join(groups[e])
        sections.append(block)
    except Exception:
        pass

    # ── CONFLUENCE WEIGHT CHANGES ──────────────────────────────────────
    try:
        weight_lines: List[str] = []
        for p in cfg_patch:
            if p.get("path") != "CONFLUENCE_WEIGHTS":
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
    except Exception:
        pass

    # ── INDICATOR SETTING CHANGES ──────────────────────────────────────
    try:
        setting_lines: List[str] = []
        for p in cfg_patch:
            if p.get("path") == "CONFLUENCE_WEIGHTS":
                continue
            cur, sug = p.get("current"), p.get("suggested")
            if cur is None or sug is None:
                continue
            setting_lines.append(f"🔧 {p['path']}: {cur} → {sug}\n   Why: {p.get('reason', 'data-driven optimum')}")
        if setting_lines:
            sections.append("🎚️ INDICATOR SETTINGS — CHANGE THESE\n\n" + "\n".join(setting_lines))
    except Exception:
        pass

    # ── ENTRY GATE THRESHOLD ───────────────────────────────────────────
    gate_rec = None
    try:
        rec_thr = engine.recommend_threshold(rows, target_winrate=target, min_sample=getattr(cfg, "MIN_WIN_RATE_SAMPLE", 20))
        if rec_thr.get("valid") and rec_thr.get("recommended") and rec_thr["recommended"] > cfg.CONFLUENCE_MIN_ABS_SCORE:
            gate_rec = rec_thr
            sections.append(
                f"🚪 ENTRY BAR — RAISE IT\n"
                f"CONFLUENCE_MIN_ABS_SCORE: {cfg.CONFLUENCE_MIN_ABS_SCORE:.1f} → {rec_thr['recommended']:.1f}\n"
                f"   This alone filters out {rec_thr.get('dropped', 0)} weak trades "
                f"({rec_thr.get('dropped_pct', 0):.0%}) and lifts expected WR to ~{rec_thr.get('rec_wr', 0):.0%}."
            )
    except Exception:
        pass

    # ── COPY-PASTE CONFIG BLOCK ────────────────────────────────────────
    try:
        json_changes: Dict[str, Any] = {}
        for p in cfg_patch:
            if p.get("path") == "CONFLUENCE_WEIGHTS":
                json_changes["CONFLUENCE_WEIGHTS"] = p.get("suggested", {})
            elif p.get("current") is not None and p.get("suggested") is not None:
                json_changes[p["path"]] = p["suggested"]
        if gate_rec is not None:
            json_changes["CONFLUENCE_MIN_ABS_SCORE"] = round(gate_rec["recommended"], 1)
        if json_changes:
            sections.append("📋 COPY-PASTE INTO config_macd.json\n" + json.dumps(json_changes, indent=1))
    except Exception:
        pass

    # ── BEST / WORST CONDITIONS ────────────────────────────────────────
    try:
        pair_stats = engine.per_pair_breakdown(rows, min_sample=5)  # worst-first
        if len(pair_stats) >= 2:
            worst, best = pair_stats[0], pair_stats[-1]
            line = (f"🌍 WHERE YOU WIN & LOSE\n"
                    f"🏆 Best: {best[0]} at {best[1]:.0%} WR (n={best[2]})\n"
                    f"💀 Worst: {worst[0]} at {worst[1]:.0%} WR (n={worst[2]}) — consider removing this pair")
            sess = engine.per_pair_session_breakdown(rows, min_sample=5)
            if len(sess) >= 2:
                line += f"\n⏰ Best session: {sess[-1][1]} ({sess[-1][2]:.0%}) | Worst: {sess[0][1]} ({sess[0][2]:.0%})"
            sections.append(line)
    except Exception:
        pass

    if not sections:
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
    return [escape_markdown_v2(m) for m in msgs]

class BrainEngineV2(BaseBrainEngine):
    """Drop-in replacement for BrainEngine. Inherits the original and adds
    prescriptive phases 1.5-6 plus actionability scoring."""

    def __init__(self, sdb: RedisStateStore):
        super().__init__(sdb)
        self._phase_samples = _PHASE_MIN_SAMPLES
        self._recs_cache: Optional[Dict[str, Any]] = None
        self._recs_cache_ts: float = 0.0

    @staticmethod
    def _shadow_weight_check(
        shadow_rows, current_weights, suggested_weights,
        threshold, min_n=15, max_wr_drop=0.05,
    ):
        """Out-of-sample veto on proposed weight changes. Shadow rows are
        alerts the live system REJECTED — an independent sample from the
        same window. Score them under current vs suggested weights (same
        scoring rule both times, so the comparison is apples-to-apples
        even though this reconstruction omits gate-level score
        components) and veto if the proposal materially degrades WR."""
        def _wr_at(rows, weights):
            kept = [
                r for r in rows
                if r.get("votes") and
                sum(w for vn, w in weights.items() if r["votes"].get(vn)) >= threshold
            ]
            if len(kept) < min_n:
                return None, len(kept)
            return sum(r["win"] for r in kept) / len(kept), len(kept)

        cur_wr, cur_n = _wr_at(shadow_rows, current_weights)
        new_wr, new_n = _wr_at(shadow_rows, suggested_weights)
        if cur_wr is None or new_wr is None:
            return True, f"shadow too thin to veto (cur n={cur_n}, new n={new_n})"
        if new_wr < cur_wr - max_wr_drop:
            return False, f"suggested weights degrade shadow WR {cur_wr:.0%}→{new_wr:.0%} (n={new_n})"
        return True, f"shadow WR stable {cur_wr:.0%}→{new_wr:.0%} (n={new_n})"

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

    async def _load_rows(self) -> tuple:
        """Shared row loader — reads from archived files if available,
        otherwise falls back to Redis streams."""
        window_days = getattr(cfg, "BRAIN_ANALYSIS_WINDOW_DAYS", 30)
        
        # Try file archive first (set in config or env)
        data_dir = getattr(cfg, "OUTCOME_DATA_DIR", None) or os.environ.get("OUTCOME_DATA_DIR")
        if data_dir and Path(data_dir).exists():
            real_rows = load_archived_outcomes(data_dir, window_days=window_days, shadow=False)
            shadow_rows = load_archived_outcomes(data_dir, window_days=window_days, shadow=True)
            if real_rows or shadow_rows:
                logger = logging.getLogger("macd_bot")
                logger.info(
                    f"🗄️ Brain using file archive: {len(real_rows)} real, "
                    f"{len(shadow_rows)} shadow rows from {data_dir}"
                )
                return real_rows, shadow_rows
        
        # Fallback to Redis streams
        sample_size = getattr(cfg, "BRAIN_REPORT_STREAM_SAMPLE", 5000)
        real_raw, shadow_raw = await asyncio.gather(
            self._read_stream(RedisKeyPrefix.OUTCOME_LOG_STREAM, sample_size),
            self._read_stream(RedisKeyPrefix.SHADOW_LOG_STREAM, sample_size),
        )
        real_rows = self._parse_rows(real_raw, window_days=window_days)
        shadow_rows = self._parse_rows(shadow_raw, window_days=window_days)
        return real_rows, shadow_rows

    async def generate_recommendations(self) -> Dict[str, Any]:
        """Caching wrapper so the action plan + technical report share one compute."""
        now = time.time()
        if self._recs_cache is not None and (now - self._recs_cache_ts) < 120:
            return self._recs_cache
        result = await self._generate_recommendations_full()
        self._recs_cache = result
        self._recs_cache_ts = now
        return result

    async def _generate_recommendations_full(self) -> Dict[str, Any]:
        # ── 0. Baseline (original brain logic) ───────────────────────────
        base_recs = await self._generate_baseline_recommendations()
        real_rows = base_recs.get("_real_rows", [])
        shadow_rows = base_recs.get("_shadow_rows", [])
        recommendations: List[Dict[str, Any]] = list(base_recs.get("recommendations", []))
        config_patch: List[Dict[str, Any]] = list(base_recs.get("config_patch", []))
        ai_metrics: Dict[str, Any] = dict(base_recs.get("ai_metrics", {}))

        min_sample = getattr(cfg, "MIN_WIN_RATE_SAMPLE", 20)
        disable_wr = getattr(cfg, "BRAIN_ALERT_DISABLE_THRESHOLD_WR", 0.40)
        star_wr = getattr(cfg, "BRAIN_STAR_ALERT_WR", 0.70)
        max_weight_delta = getattr(cfg, "BRAIN_WEIGHT_OPTIMIZER_MAX_DELTA", 2.0)
        wf_weight_opt = getattr(cfg, "BRAIN_WEIGHT_OPTIMIZER_WALK_FORWARD", True)

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
            recommendations.append({
                "type": "repair_shop",
                "severity": repair["severity"],
                "category": repair["category"],
                "message": (
                    f"🔧 [{repair['category'].upper()}] {repair['diagnosis']}\n"
                    f"   → {repair['action']}\n"
                    f"   Impact: {repair['expected_impact']}"
                ),
            })

        # ── Phase 1.5: Vote Weight Optimizer (FIXED) ─────────────────────
        if len(real_rows) >= self._phase_samples["weight_optimizer"]:
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

                if changed:
                    change_strs = [f"{k}: {old:.1f}→{new:.1f}" for k, old, new in changed[:6]]
                    extra = f" (+{len(changed)-6} more)" if len(changed) > 6 else ""

                    recommendations.append({
                        "type": "weight_optimizer",
                        "severity": "high" if conf_score > 0.6 else "medium",
                        "message": (
                            f"🧮 Weight Optimizer (n={wopt['n_samples']}, {wf_status}, "
                            f"confidence {conf_label} {conf_score:.0%}):\n"
                            f"   Changes: {', '.join(change_strs)}{extra}\n"
                            f"   Max delta/cycle: ±{max_weight_delta}"
                        ),
                        "delta_ev": 0.0,  # Will be computed by counterfactual
                        "wilson_lo": max(0.0, 0.5 - conf_score * 0.2),
                        "wilson_hi": min(1.0, 0.5 + conf_score * 0.2),
                    })

                    # ── Shadow out-of-sample veto ──────────────────────
                    shadow_weight_ok, shadow_weight_note = True, ""
                    if (wopt.get("walk_forward_passed") and conf_score >= 0.4
                            and len(shadow_rows) >= 15):
                        shadow_weight_ok, shadow_weight_note = self._shadow_weight_check(
                            shadow_rows, CONFLUENCE_WEIGHTS,
                            wopt["suggested_weights"], cfg.CONFLUENCE_MIN_ABS_SCORE,
                        )

                    # Only emit config patch if walk-forward passed AND
                    # confidence is decent AND shadow sample doesn't veto.
                    if (wopt.get("walk_forward_passed") and conf_score >= 0.4
                            and shadow_weight_ok):
                        config_patch.append({
                            "path": "CONFLUENCE_WEIGHTS",
                            "current": dict(CONFLUENCE_WEIGHTS),
                            "suggested": wopt["suggested_weights"],
                            "reason": (
                                f"Logistic-regression optimal weights ({wf_status}, "
                                f"conf={conf_score:.2f}, {shadow_weight_note})"
                            ),
                        })
                        if getattr(cfg, "BRAIN_AUTO_APPLY_DYNAMIC_WEIGHTS", False):
                            saved = await self.sdb.set_dynamic_weights(wopt["suggested_weights"])
                            if saved:
                                recommendations.append({
                                    "type": "dynamic_weights_applied",
                                    "severity": "medium",
                                    "message": f"💾 Dynamic weights persisted ({len(changed)} votes updated).",
                                })
                    else:
                        if not wopt.get("walk_forward_passed"):
                            reason = "walk-forward FAILED"
                        elif conf_score < 0.4:
                            reason = f"confidence too low ({conf_score:.2f})"
                        else:
                            reason = f"shadow-sample veto — {shadow_weight_note}"
                        recommendations.append({
                            "type": "weight_optimizer_blocked",
                            "severity": "low",
                            "message": (
                                f"🛡️ Weight changes BLOCKED: {reason}. "
                                f"Keeping current weights. "
                                f"Accumulate more data or reduce max_weight_delta."
                            ),
                        })                
                else:
                    recommendations.append({
                        "type": "weight_optimizer",
                        "severity": "low",
                        "message": f"🧮 Weight Optimizer: no significant changes detected (n={wopt['n_samples']}).",
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

        # ── Per-alert breakdown ──────────────────────────────────────────
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

        # ── Phase 2: Parameter Autopsy ───────────────────────────────────
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
                            f"☠️ Poison: {poisoner} kills {victim}. "
                            f"Together={inter['wr_both']:.0%}, {victim} alone={wr_victim_alone:.0%}."
                        ),
                        "delta_ev": abs(inter["delta"]),
                        # ── FDR: p_value stamped by the miner directly ──
                        "p_value": inter.get("p_value"),
                    })

        # ── Phase 5: Counterfactual Simulator (shadow-validated) ──────
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

        # ── AI/ML: Permutation Vote Importance ───────────────────────────
        if len(real_rows) >= 30:
            _perm_n = 15
            perm_imp = engine.permutation_vote_importance(
                real_rows, min_sample=30, n_permutations=_perm_n
            )
            if perm_imp:
                top_positive = [p for p in perm_imp if p["direction"] == "positive"][:3]
                top_negative = [p for p in perm_imp if p["direction"] == "negative"][:3]
                parts = []
                if top_positive:
                    parts.append("most impactful: " + ", ".join(
                        f"{p['vote']}({p['importance']:+.3f})" for p in top_positive))
                if top_negative:
                    parts.append("harmful: " + ", ".join(
                        f"{p['vote']}({p['importance']:+.3f})" for p in top_negative))

                # FDR tests the single strongest signal. If the top signal
                _top = (top_positive + top_negative)[:1]
                _top_rec = _top[0] if _top else None

                _rec = {
                    "type": "permutation_importance", "severity": "low",
                    "message": f"🤖 Permutation importance — {'; '.join(parts)}",
                }
                if _top_rec is not None:
                    _rec["top_vote"] = _top_rec["vote"]
                    _rec["top_importance"] = _top_rec["importance"]
                    _rec["top_std"] = _top_rec.get("std", 0.0)
                    _rec["n_permutations"] = _perm_n
                recommendations.append(_rec)

        # ── Benjamini-Hochberg FDR correction ────────────────────────────
       
        p_val_indices: List[int] = []
        p_vals: List[float] = []
        for idx, r in enumerate(recommendations):
            p = _extract_p_value_for_fdr(r, real_rows, min_sample)
            if p is not None:
                p_val_indices.append(idx)
                p_vals.append(p)
        if p_vals:
            keep_mask = engine.benjamini_hochberg(p_vals, alpha=0.10)
            n_survived = sum(keep_mask)
            n_tested = len(p_vals)
            for flag, idx in zip(keep_mask, p_val_indices):
                recommendations[idx]["fdr_passed"] = bool(flag)
            for flag, idx in zip(keep_mask, p_val_indices):
                if not flag and recommendations[idx]["severity"] in ("high", "medium"):
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
                        f"📊 FDR (Benjamini-Hochberg, α=0.10): {n_survived}/{n_tested} "
                        f"statistical claims survived correction across the report. "
                        f"Surviving claims are marked `fdr_passed=True`; demoted "
                        f"claims are downgraded to low severity."
                    ),
                })

        # ── Actionability scoring (FIXED confidence) ─────────────────────
        for rec in recommendations:
            rec["actionability_score"] = round(score_actionability(rec), 3)

        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        recommendations.sort(key=lambda x: (
            severity_order.get(x.get("severity"), 4),
            -x.get("actionability_score", 0),
        ))

        # ── Config version hash ──────────────────────────────────────────
        ai_metrics["config_version"] = hash_config_state(
            CONFLUENCE_WEIGHTS, cfg.CONFLUENCE_MIN_ABS_SCORE, cfg.CONFLUENCE_MIN_PCT
        )

        # ── Bonus-aware metrics ──────────────────────────────────────────
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

        # ── Re-assemble ──────────────────────────────────────────────────
        result = dict(base_recs)
        result["recommendations"] = recommendations
        result["recommendation_count"] = len(recommendations)
        result["config_patch"] = config_patch
        result["ai_metrics"] = ai_metrics
        return result

    # ── Baseline wrapper that also exposes raw rows ──────────────────────
    async def _generate_baseline_recommendations(self) -> Dict[str, Any]:
        base = await super().generate_recommendations()
        real_rows, shadow_rows = await self._get_rows()
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
            
            for rec in recs.get("recommendations", []):
                rec_type = rec.get("type")
                if rec_type == "disable_alert":
                    ak = rec.get("alert_key")
                    if ak:
                        disable_alerts.append(ak)
                elif rec_type == "reinstate_alert":
                    ak = rec.get("alert_key")
                    if ak:
                        reinstate_alerts.append(ak)
            
            # Only store safe config patches
            for patch in recs.get("config_patch", []):
                field = patch.get("path")
                suggested = patch.get("suggested")
                if field in CONFIG_OVERRIDE_ALLOWED_FIELDS and suggested is not None:
                    config_patches.append({
                        "path": field,
                        "current": patch.get("current"),
                        "suggested": suggested,
                        "reason": patch.get("reason", "")
                    })
            
            plan_data = {
                "generated_at": int(time.time()),
                "config_patch": config_patches,
                "disable_alerts": disable_alerts,
                "reinstate_alerts": reinstate_alerts,
            }
            await self.sdb.set_metadata(
                "brain_pending_plan",
                json_dumps(plan_data),
                ttl=7 * 86400  # 7 days
            )
        except Exception as e:
            logging.getLogger("macd_bot").warning(f"Failed to store pending plan: {e}")

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
            
            # Apply config changes
            for patch in plan.get("config_patch", []):
                field = patch.get("path")
                value = patch.get("suggested")
                if field in CONFIG_OVERRIDE_ALLOWED_FIELDS:
                    ok = await self.sdb.write_config_override(field, value)
                    if ok:
                        applied.append(f"✅ {field}: {value}")
                        logger_run.info(f"Applied brain config: {field} = {value}")
            
            # Apply alert disables
            for ak in plan.get("disable_alerts", []):
                ok = await self.sdb.set_alert_key_disabled(ak, True)
                if ok:
                    applied.append(f"🔴 Disabled: {ak}")
                    logger_run.info(f"Applied brain disable: {ak}")
            
            # Apply alert reinstates
            for ak in plan.get("reinstate_alerts", []):
                ok = await self.sdb.set_alert_key_disabled(ak, False)
                if ok:
                    applied.append(f"🟢 Reinstated: {ak}")
                    logger_run.info(f"Applied brain reinstate: {ak}")
            
            if applied:
                msg = (
                    f"✅ APPLIED BRAIN PLAN\n"
                    f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                    + "\n".join(applied)
                    + "\n\nRestart the bot for changes to take effect\\."
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

    async def generate_report(self, pairs, telegram_queue, logger_run):
        """Override: send ONLY the plain-English action plan (no jargon), then store for application."""
        try:
            recs = await self.generate_recommendations()
            
            # Build and send the plain-English action plan
            plan_messages = build_profit_action_plan(recs, cfg)
            for msg in plan_messages:
                await telegram_queue.send(msg)
            
            # Add a footer explaining how to apply
            apply_hint = (
                "━━━━━━━━━━━━━━━━━━━━━━\n"
                "📝 TO APPLY THESE CHANGES:\n"
                "Run: `python macd_unified.py --apply-brain`\n"
                "Or manually edit config_macd.json with the values above\\."
            )
            await telegram_queue.send(escape_markdown_v2(apply_hint))
            
            # Store the plan for later application
            await self._store_pending_plan(recs)
            
            logger_run.info(f"Brain report sent ({len(plan_messages)} messages) and stored for application")
            return recs
            
        except Exception as e:
            logger_run.warning(f"Report generation failed: {e}")
            # Fall back to the old technical report if the new one fails
            try:
                return await self._generate_and_send(pairs, telegram_queue, logger_run)
            except Exception as fallback_e:
                logger_run.error(f"Fallback report also failed: {fallback_e}")
                return None

    async def maybe_generate_report(
        self,
        pairs: List[str],
        telegram_queue: Any,
        logger_run: logging.Logger,
    ) -> None:
        """Override: same run-count/interval gating as the base class, but
        route the actual send through generate_report() (Profit Action Plan)
        instead of BrainEngine._generate_and_send() (old jargon report)."""
        interval = getattr(cfg, "BRAIN_REPORT_INTERVAL_RUNS", 48)
        if interval <= 0:
            logger_run.warning("BRAIN_REPORT_INTERVAL_RUNS is <= 0, disabling brain reports.")
            return
        if not getattr(cfg, "ENABLE_BRAIN", True):
            return
        if not cfg.ENABLE_WIN_RATE_FILTER:
            logger_run.warning(
                "ENABLE_BRAIN is on but ENABLE_WIN_RATE_FILTER is off — brain has no data source, skipping report."
            )
            return
        if getattr(cfg, "DRY_RUN_MODE", False):
            logger_run.info("DRY_RUN_MODE is on — skipping brain report (outcome data would be synthetic).")
            return

        run_count = await self._next_run_count()
        if run_count is None or run_count % interval != 0:
            return

        try:
            await self.generate_report(pairs, telegram_queue, logger_run)
        except Exception:
            await self._rollback_run_count()
            raise

    async def send_report_now(self, pairs: List[str], telegram_queue: Any, logger_run: logging.Logger) -> bool:
        """Override: on-demand path also goes through generate_report() (Profit
        Action Plan) instead of BrainEngine._generate_and_send()."""
        if not getattr(cfg, "ENABLE_BRAIN", True):
            logger_run.warning("ENABLE_BRAIN is off — skipping on-demand brain report.")
            return True
        if not cfg.ENABLE_WIN_RATE_FILTER:
            logger_run.warning(
                "ENABLE_BRAIN is on but ENABLE_WIN_RATE_FILTER is off — "
                "brain has no data source, skipping report."
            )
            return True
        if getattr(cfg, "DRY_RUN_MODE", False):
            logger_run.info("DRY_RUN_MODE is on — skipping brain report (outcome data would be synthetic).")
            return True

        result = await self.generate_report(pairs, telegram_queue, logger_run)
        return result is not None
