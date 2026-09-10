#!/usr/bin/env python3
"""brain_enhanced.py — Prescriptive Brain (Roadmap Phases 1.5-6)"""

from __future__ import annotations
import asyncio 
import logging
import time
from collections import defaultdict
from typing import Any, Dict, List
import os
from pathlib import Path
from archive_reader import load_archived_outcomes

from bot_config import cfg, CONFLUENCE_WEIGHTS
from state import RedisKeyPrefix, RedisStateStore
from brain import BrainEngine as BaseBrainEngine
import threshold_engine as engine

from threshold_engine import (
    optimize_vote_weights, conditional_performance,
    interaction_miner, simulate_config_change, regime_profile_optimizer,
    hash_config_state, score_actionability, compare_config_versions,
)

_PHASE_MIN_SAMPLES = {
    "weight_optimizer": 100,
    "parameter_autopsy": 30,
    "conditional_gating": 15,
    "vote_interactions": 20,
    "counterfactual": 10,
    "regime_profiles": 25,
    "config_regression": 20,
}

class BrainEngineV2(BaseBrainEngine):
    """Drop-in replacement for BrainEngine. Inherits the original and adds
    prescriptive phases 1.5-6 plus actionability scoring."""

    def __init__(self, sdb: RedisStateStore):
        super().__init__(sdb)
        self._phase_samples = _PHASE_MIN_SAMPLES

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

                    # Only emit config patch if walk-forward passed AND confidence is decent
                    if wopt.get("walk_forward_passed") and conf_score >= 0.4:
                        config_patch.append({
                            "path": "CONFLUENCE_WEIGHTS",
                            "current": dict(CONFLUENCE_WEIGHTS),
                            "suggested": wopt["suggested_weights"],
                            "reason": f"Logistic-regression optimal weights ({wf_status}, conf={conf_score:.2f})",
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
                        reason = "walk-forward FAILED" if not wopt.get("walk_forward_passed") else f"confidence too low ({conf_score:.2f})"
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
                        })

        # ── Phase 4: Vote Interaction Miner ──────────────────────────────
        if len(real_rows) >= self._phase_samples["vote_interactions"]:
            interactions = interaction_miner(real_rows, min_sample=self._phase_samples["vote_interactions"])
            for inter in interactions[:5]:
                v1, v2 = inter["pair"]
                if inter["type"] == "synergy":
                    recommendations.append({
                        "type": "vote_interaction", "kind": "synergy", "severity": "low",
                        "message": (
                            f"🔗 Synergy: {v1}+{v2} = {inter['wr_both']:.0%} WR "
                            f"(n={inter['n_both']}). Alone: {v1}={inter['wr_only_v1']:.0%}, "
                            f"{v2}={inter['wr_only_v2']:.0%}."
                        ),
                        "delta_ev": abs(inter["delta"]),
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
                    })

        # ── Phase 5: Counterfactual Simulator ────────────────────────────
        baseline_ev = ai_metrics.get("net_ev") or 0.0
        if real_rows and len(real_rows) >= self._phase_samples["counterfactual"]:
            scenarios: List[Dict[str, Any]] = []
            s1 = simulate_config_change(real_rows, baseline_ev,
                                        new_threshold=cfg.CONFLUENCE_MIN_ABS_SCORE + 1.0)
            if s1:
                scenarios.append({"label": f"Threshold +1 ({cfg.CONFLUENCE_MIN_ABS_SCORE+1.0})", **s1})
            if any("context" in r and r["context"].get("rsi_adaptive_buy") for r in real_rows):
                rsi_buy_cap = getattr(cfg, "RSI_ADAPTIVE_BUY_VOLATILE", 70.0)
                s2 = simulate_config_change(real_rows, baseline_ev,
                                            new_params={"rsi_curr": rsi_buy_cap - 3})
                if s2:
                    scenarios.append({"label": "RSI buy cap -3", **s2})
            if s1 and any("context" in r for r in real_rows):
                s3 = simulate_config_change(real_rows, baseline_ev,
                                            new_threshold=cfg.CONFLUENCE_MIN_ABS_SCORE + 1.0,
                                            new_params={"rsi_curr": getattr(cfg, "RSI_ADAPTIVE_BUY_VOLATILE", 70.0) - 3})
                if s3:
                    scenarios.append({"label": "Threshold +1 + RSI cap -3", **s3})
            if scenarios:
                best = max(scenarios, key=lambda x: x["ev"])
                recommendations.append({
                    "type": "counterfactual",
                    "severity": "high" if best["delta_ev"] > 0.05 else "low",
                    "message": (
                        f"🔮 Best scenario: '{best['label']}' → "
                        f"EV {best['ev']:+.3f}%/trade (Δ{best['delta_ev']:+.3f}%), "
                        f"WR {best['wr']:.0%}, n={best['n']}."
                    ),
                    "delta_ev": best["delta_ev"],
                })
                ai_metrics["counterfactual_scenarios"] = scenarios

        # ── Phase 6: Regime Profiles ─────────────────────────────────────
        if len(real_rows) >= self._phase_samples["regime_profiles"]:
            rpo = regime_profile_optimizer(real_rows, regime_field="adx_val",
                                           min_sample=self._phase_samples["regime_profiles"])
            if rpo.get("valid") and len(rpo.get("regimes", [])) >= 2:
                lines = [
                    f"  Regime {reg['regime_id']} (ADX {reg['range'][0]}-{reg['range'][1]}): "
                    f"thr={reg['recommended_threshold']:.1f}, WR={reg['wr']:.0%}"
                    for reg in rpo["regimes"]
                ]
                recommendations.append({
                    "type": "dynamic_regime_profile", "severity": "low",
                    "message": f"📊 Regime thresholds:\n" + "\n".join(lines),
                })

        # ── Config Version Regression ────────────────────────────────────
        version_comparisons = compare_config_versions(
            real_rows, min_sample=self._phase_samples["config_regression"]
        )
        for comp in version_comparisons:
            if comp["regression"]:
                recommendations.append({
                    "type": "config_regression", "severity": "high",
                    "message": (
                        f"🚨 Config regression: WR {comp['prev_wr']:.0%}→{comp['cur_wr']:.0%} "
                        f"({comp['prev_version']}→{comp['cur_version']}). Consider reverting."
                    ),
                    "delta_ev": abs(comp["delta_wr"]),
                })
            elif comp["improvement"]:
                recommendations.append({
                    "type": "config_improvement", "severity": "low",
                    "message": (
                        f"✅ Config improved WR: {comp['prev_wr']:.0%}→{comp['cur_wr']:.0%} "
                        f"({comp['prev_version']}→{comp['cur_version']})."
                    ),
                })
        ai_metrics["config_comparisons"] = version_comparisons

        # ── AI/ML: Permutation Vote Importance ───────────────────────────
        if len(real_rows) >= 30:
            perm_imp = engine.permutation_vote_importance(real_rows, min_sample=30, n_permutations=15)
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
                recommendations.append({
                    "type": "permutation_importance", "severity": "low",
                    "message": f"🤖 Permutation importance — {'; '.join(parts)}",
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

        # ── Re-assemble ──────────────────────────────────────────────────
        result = dict(base_recs)
        result["recommendations"] = recommendations
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
    
    async def generate_report(self, pairs, telegram_queue, logger_run):
        return await self._generate_and_send(pairs, telegram_queue, logger_run)