
#!/usr/bin/env python3
"""brain_enhanced.py — Prescriptive Brain (Roadmap Phases 1.5-6)"""
from __future__ import annotations
import asyncio
import json
import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

from alerts import escape_markdown_v2

from bot_config import cfg, json_dumps, format_ist_time, CONFLUENCE_WEIGHTS
from state import RedisKeyPrefix, RedisStateStore
from brain import BrainEngine as BaseBrainEngine
import threshold_engine as engine

from threshold_engine import (
    optimize_vote_weights, parameter_autopsy, conditional_performance,
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

    async def _load_rows(self) -> tuple:
        """Shared row loader used by original + new phases."""
        sample_size = getattr(cfg, "BRAIN_REPORT_STREAM_SAMPLE", 5000)
        window_days = getattr(cfg, "BRAIN_ANALYSIS_WINDOW_DAYS", 30)
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

        # ── Phase 1.5: Vote Weight Optimizer ─────────────────────────────
        if len(real_rows) >= self._phase_samples["weight_optimizer"]:
            wopt = optimize_vote_weights(real_rows, CONFLUENCE_WEIGHTS,
                                         min_sample=self._phase_samples["weight_optimizer"])
            if wopt["valid"]:
                # Build diff
                changed = []
                for k, new_v in wopt["suggested_weights"].items():
                    old_v = CONFLUENCE_WEIGHTS.get(k)
                    if old_v is not None and abs(new_v - old_v) > 0.3:
                        changed.append(f"{k}: {old_v}→{new_v}")
                if changed:
                    recommendations.append({
                        "type": "weight_optimizer",
                        "severity": "high",
                        "message": (
                            f"🧮 Vote Weight Optimizer (n={wopt['n_samples']}): "
                            f"data-driven weights differ from current. Top changes: "
                            f"{', '.join(changed[:5])}."
                        ),

                        "delta_ev": max(abs(new_v - CONFLUENCE_WEIGHTS.get(k, 0))
                            for k, new_v in wopt["suggested_weights"].items()) / 100.0,
                        "wilson_lo": 0.0,
                        "wilson_hi": 0.0,
                    })
                    config_patch.append({
                        "path": "CONFLUENCE_WEIGHTS",
                        "current": dict(CONFLUENCE_WEIGHTS),
                        "suggested": wopt["suggested_weights"],
                        "reason": "Logistic-regression optimal weights",
                    })
                if wopt["negative_votes"]:
                    recommendations.append({
                        "type": "negative_votes",
                        "severity": "medium",
                        "message": (
                            f"⚠️ Votes with negative coefficients (harmful): "
                            f"{', '.join(f'{v}({c})' for v,c in wopt['negative_votes'][:3])}. "
                            f"Consider disabling these votes."
                        ),
                    })

        # ── NEW: Per-alert breakdown (worst + best) ──────────────────────
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
                "message": "Per-alert breakdown:\n" + "\n".join(msg_parts),
            })

        # ── Phase 2: Parameter Autopsy (ALERT-AWARE) ─────────────────────
        if real_rows and any("context" in r for r in real_rows):
            # Map context parameters → the alert keys they gate
            PARAM_ALERT_MAP = {
                "ppo_adaptive_threshold": [
                    "ppo_adaptive_up", "ppo_adaptive_down"
                ],
                "rsi_adaptive_buy": [
                    "rsi_ema5_up", "rsi_cross_adaptive_up"
                ],
                "rsi_adaptive_sell": [
                    "rsi_ema5_down", "rsi_cross_adaptive_down"
                ],
                "buy_wick_ratio": [
                    "strong_reversal_buy", "hist_rma_buy", "ppohist_buy",
                    "tk_conversion_up", "kijun_cross_up",
                ],
                "sell_wick_ratio": [
                    "strong_reversal_sell", "hist_rma_sell", "ppohist_sell",
                    "tk_conversion_down", "kijun_cross_down",
                ],
            }

            params_higher_worse = {
                "rsi_adaptive_buy": True,    # higher buy cap = worse
                "rsi_adaptive_sell": False,  # higher sell cap = better (lower is worse)
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
                    # Suggest a config patch for manual review (not auto-applied)
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
            # Check top 5 most frequent alert keys
            ak_counts: Dict[str, int] = defaultdict(int)
            for r in real_rows:
                ak_counts[r["alert_key"]] += 1
            top_aks = sorted(ak_counts, key=lambda k: -ak_counts[k])[:5]
            conditions = [
                ("adx_val", 25.0),
                ("rsi_curr", 50.0),
                ("buy_wick_ratio", 0.3),
            ]
            for ak in top_aks:
                for cond_field, cond_thr in conditions:
                    cp = conditional_performance(real_rows, ak, cond_field, cond_thr,
                                                 min_sample=self._phase_samples["conditional_gating"])
                    if cp.get("valid") and cp["recommendation"] != "neutral":
                        recommendations.append({
                            "type": "conditional_gating",
                            "severity": "medium",
                            "message": (
                                f"🔀 {ak} performs differently under {cond_field}: "
                                f"{cp['above']['wr']:.0%} WR when >{cond_thr} "
                                f"vs {cp['below']['wr']:.0%} when ≤{cond_thr}. "
                                f"Recommendation: {cp['recommendation']}."
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
                        "type": "vote_interaction",
                        "severity": "low",
                        "message": (
                            f"🔗 Synergy: {v1} + {v2} together = {inter['wr_both']:.0%} WR "
                            f"(n={inter['n_both']}). Alone: {v1}={inter['wr_only_v1']:.0%}, "
                            f"{v2}={inter['wr_only_v2']:.0%}. Stack these votes."
                        ),
                        "delta_ev": abs(inter["delta"]), 
                    })
                else:
                    recommendations.append({
                        "type": "vote_interaction",
                        "severity": "medium",
                        "message": (
                            f"☠️ Poison: {v2} kills {v1}. Together={inter['wr_both']:.0%} WR, "
                            f"{v1} alone={inter['wr_only_v1']:.0%}. Avoid this combo."
                        ),
                        "delta_ev": abs(inter["delta"]),
                    })

        # ── Phase 5: Counterfactual Simulator ────────────────────────────
        baseline_ev = ai_metrics.get("net_ev") or 0.0
        if real_rows and len(real_rows) >= self._phase_samples["counterfactual"]:
            scenarios: List[Dict[str, Any]] = []
            # Scenario A: current threshold +1
            s1 = simulate_config_change(real_rows, baseline_ev,
                                        new_threshold=cfg.CONFLUENCE_MIN_ABS_SCORE + 1.0)
            if s1:
                scenarios.append({"label": f"Threshold +1 ({cfg.CONFLUENCE_MIN_ABS_SCORE+1.0})", **s1})
            # Scenario B: tighten RSI buy cap by 3
            if any("context" in r and r["context"].get("rsi_adaptive_buy") for r in real_rows):

                rsi_buy_cap = getattr(cfg, "RSI_ADAPTIVE_BUY", 70.0)  # or whatever your default is
                s2 = simulate_config_change(real_rows, baseline_ev,
                                            new_params={"rsi_curr": rsi_buy_cap - 3})

                if s2:
                    scenarios.append({"label": "RSI buy cap -3", **s2})
            # Scenario C: combined
            if s1 and any("context" in r for r in real_rows):
                s3 = simulate_config_change(real_rows, baseline_ev,
                                            new_threshold=cfg.CONFLUENCE_MIN_ABS_SCORE + 1.0,
                                            new_params={"rsi_curr": cfg.RSI_ADAPTIVE_BUY_VOLATILE - 3})
                if s3:
                    scenarios.append({"label": "Threshold +1 + RSI cap -3", **s3})

            if scenarios:
                best = max(scenarios, key=lambda x: x["ev"])
                recommendations.append({
                    "type": "counterfactual",
                    "severity": "high" if best["delta_ev"] > 0.05 else "low",
                    "message": (
                        f"🔮 Counterfactual: best simulated scenario is '{best['label']}' → "
                        f"EV {best['ev']:+.3f}%/trade (Δ{best['delta_ev']:+.3f}%), "
                        f"WR {best['wr']:.0%}, n={best['n']}."
                    ),
                    "delta_ev": best["delta_ev"],
                })
                ai_metrics["counterfactual_scenarios"] = scenarios

        # ── Phase 6: Dynamic Regime Profiles ─────────────────────────────
        if len(real_rows) >= self._phase_samples["regime_profiles"]:
            rpo = regime_profile_optimizer(real_rows, regime_field="adx_val",
                                           min_sample=self._phase_samples["regime_profiles"])
            if rpo.get("valid") and len(rpo.get("regimes", [])) >= 2:
                lines = []
                for reg in rpo["regimes"]:
                    lines.append(
                        f"  Regime {reg['regime_id']} (ADX {reg['range'][0]}-{reg['range'][1]}): "
                        f"threshold={reg['recommended_threshold']:.1f}, WR={reg['wr']:.0%}"
                    )
                recommendations.append({
                    "type": "dynamic_regime_profile",
                    "severity": "low",
                    "message": (
                        f"📊 Regime-specific thresholds (by ADX):\n"
                        + "\n".join(lines)
                        + "\n(Diagnostic only — apply manually if you add regime branching.)"
                    ),
                })

        # ── Risk Flag: Config Version Regression Check ──────────────────
        version_comparisons = compare_config_versions(
            real_rows, min_sample=self._phase_samples["config_regression"]
        )
        for comp in version_comparisons:
            if comp["regression"]:
                recommendations.append({
                    "type": "config_regression",
                    "severity": "high",
                    "message": (
                        f"🚨 Config regression: WR fell {comp['prev_wr']:.0%}→{comp['cur_wr']:.0%} "
                        f"after the last config change (config {comp['prev_version']}→{comp['cur_version']}, "
                        f"n={comp['prev_n']}/{comp['cur_n']}). Consider reverting."
                    ),
                    "delta_ev": abs(comp["delta_wr"]),
                })
            elif comp["improvement"]:
                recommendations.append({
                    "type": "config_improvement",
                    "severity": "low",
                    "message": (
                        f"✅ Config change improved WR: {comp['prev_wr']:.0%}→{comp['cur_wr']:.0%} "
                        f"(config {comp['prev_version']}→{comp['cur_version']})."
                    ),
                })
        ai_metrics["config_comparisons"] = version_comparisons

        # ── Risk Flag: Actionability Scoring ─────────────────────────────
        for rec in recommendations:
            rec["actionability_score"] = round(score_actionability(rec), 3)

        # Sort by severity first, then actionability within same severity
        severity_order = {"high": 0, "medium": 1, "low": 2}
        recommendations.sort(key=lambda x: (
            severity_order.get(x.get("severity"), 3),
            -x.get("actionability_score", 0),
        ))

        # ── Risk Flag: Config Version ────────────────────────────────────
        ai_metrics["config_version"] = hash_config_state(
            CONFLUENCE_WEIGHTS, cfg.CONFLUENCE_MIN_ABS_SCORE, cfg.CONFLUENCE_MIN_PCT
        )

        # ── Re-assemble final report ─────────────────────────────────────
        result = dict(base_recs)
        result["recommendations"] = recommendations
        result["config_patch"] = config_patch
        result["ai_metrics"] = ai_metrics
        return result

    # ── Baseline wrapper that also exposes raw rows ──────────────────────
    async def _generate_baseline_recommendations(self) -> Dict[str, Any]:
        base = await super().generate_recommendations()  # BaseBrainEngine always has it — drop the hasattr guard entirely
        real_rows, shadow_rows = await self._load_rows()
        base["_real_rows"] = real_rows
        base["_shadow_rows"] = shadow_rows
        return base

    async def _minimal_baseline(self) -> Dict[str, Any]:
        """Fallback baseline if original BrainEngine is unavailable."""
        real_rows, shadow_rows = await self._load_rows()
        return {
            "generated_at": int(time.time()),
            "real_sample_size": len(real_rows),
            "shadow_sample_size": len(shadow_rows),
            "recommendation_count": 0,
            "recommendations": [],
            "shadow_summary": {},
            "config_patch": [],
            "current_config": {
                "CONFLUENCE_MIN_ABS_SCORE": cfg.CONFLUENCE_MIN_ABS_SCORE,
                "CONFLUENCE_MIN_PCT": cfg.CONFLUENCE_MIN_PCT,
            },
            "ai_metrics": {
                "brier_score": 0.5,
                "brier_status": "Unknown",
                "net_ev": None,
                "half_kelly": None,
                "cusum_drifts": 0,
                "threshold_history": await self.sdb.load_threshold_history(),
                "ood_status": "Unknown",
            },
            "_real_rows": real_rows,
            "_shadow_rows": shadow_rows,
        }
