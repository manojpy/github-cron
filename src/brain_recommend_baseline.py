"""Baseline recommendation builder for the Brain core (moved verbatim out of BrainEngine.generate_recommendations; the method now delegates here)."""
from __future__ import annotations
import logging
import statistics
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
from bot_config import cfg, json_dumps, json_loads, CONFLUENCE_WEIGHTS
from state import _rc
import threshold_engine as engine
from brain_audit import get_audit, HealthStatus
from brain_helpers import KILL_SWITCH_KEY, _resolve_config_path

async def build_baseline_recommendations(self) -> Dict[str, Any]:
    """Build the full recommendation set: per-alert verdicts, a confluence
        threshold suggestion, shadow-mode insight, and a machine-readable
        config patch."""
    engine.clear_ev_first_cache()
    real_rows, shadow_rows = await self._get_rows()
    audit = get_audit() 
    recommendations: List[Dict[str, Any]] = []
    config_patch: List[Dict[str, Any]] = []
    ai_metrics: Dict[str, Any] = {}
    seen_paths = set()
    min_sample = getattr(cfg, "MIN_WIN_RATE_SAMPLE", 20)
    target_wr = cfg.MIN_WIN_RATE
    disable_wr = getattr(cfg, "BRAIN_ALERT_DISABLE_THRESHOLD_WR", 0.40)

    # ── Per-alert win rate (pooled across pairs), Wilson-bound verdicts ──
    alert_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"rows": [], "pairs": set()})
    for r in real_rows:
        s = alert_stats[r["alert_key"]]
        s["rows"].append(r)
        s["pairs"].add(r["pair"])

    # ── CUSUM drift detection (moved up so path_to_keys below can check
    drift_alerts = await self._check_cusum_drift(real_rows)
    if drift_alerts:
        recommendations.extend(drift_alerts)

    current_disabled_keys = await self.sdb.get_disabled_alert_keys()
    auto_disable_min = getattr(cfg, "BRAIN_AUTO_DISABLE_MIN_SAMPLE", 500)
    auto_disable_on = getattr(cfg, "BRAIN_AUTO_DISABLE_ENABLED", False)
    recency_on = getattr(cfg, "ENABLE_RECENCY_WEIGHTING", False)
    recency_decay_days = getattr(cfg, "RECENCY_DECAY_DAYS", 7.0)

    alert_verdicts: Dict[str, str] = {}  # alert_key -> "disable" | "star" | "monitor"
    key_history = await self.sdb.get_alert_key_history()
    now_ts = time.time()
    probation_s = float(getattr(cfg, "BRAIN_REENABLE_PROBATION_DAYS", 30)) * 86400.0

    for alert_key, s in alert_stats.items():
        # Probation after a re-enable: judge only outcomes recorded since,
        # so the losses that got it disabled cannot re-disable it at once.
        _reen_ts = key_history["reenabled_at"].get(alert_key)
        if _reen_ts is not None and probation_s > 0 and (now_ts - _reen_ts) < probation_s:
            s = {"rows": [r for r in s["rows"] if r["entry_ts"] > _reen_ts], "pairs": s["pairs"]}
        total = len(s["rows"])
        if total < min_sample:
            continue
        wins = sum(1 for r in s["rows"] if r["win"])

        if recency_on:
            wr, n_eff, lo, hi = engine.weighted_win_rate_with_bonus(
                s["rows"], decay_days=recency_decay_days
            )
            assert wr is not None
            sample_label = (
                f"{total} samples (n_eff={n_eff:.0f} "
                f"recency+bonus-weighted, {recency_decay_days:.0f}d decay)"
            )
        else:
            wr = wins / total
            lo, hi, _ = engine.wilson_ci(wins, total)
            sample_label = f"{total} samples"

        auto_eligible = auto_disable_on and total >= auto_disable_min

        if hi < disable_wr:
            # ── EV-gated disable: only disable if EV is ALSO negative ──
            _ev_cache_key = (alert_key, len(s["rows"]))
            if not hasattr(self, "_ev_obj_cache"):
                self._ev_obj_cache: Dict[Tuple[str, int], Dict[str, Any]] = {}
            if _ev_cache_key in self._ev_obj_cache:
                ev_obj = self._ev_obj_cache[_ev_cache_key]
            else:
                ev_obj = engine.ev_first_objective(s["rows"], min_sample=min_sample)
                self._ev_obj_cache[_ev_cache_key] = ev_obj
            ev_negative = ev_obj.get("valid") and ev_obj["net_ev"] <= 0
            if ev_negative:
                alert_verdicts[alert_key] = "disable"
            else:
                alert_verdicts[alert_key] = "monitor"

            if ev_negative:
                recommendations.append({
                    "type": "disable_alert", "severity": "high", "alert": alert_key,
                    "win_rate": round(wr, 3), "sample_size": total,
                    "pairs_affected": len(s["pairs"]),
                    "message": (
                        f"LOW WR / NEGATIVE EV {alert_key}: {wr:.0%} WR over "
                        f"{sample_label} across {len(s['pairs'])} pairs "
                        f"(95% CI upper bound {hi:.0%} < {disable_wr:.0%}); "
                        f"net EV {ev_obj.get('net_ev', 0):+.3f}%/trade. "
                        "Evidence supports disabling — review before applying."
                    ),
                })
            else:
                recommendations.append({
                    "type": "low_wr_but_ev_positive", "severity": "medium",
                    "alert": alert_key,
                    "win_rate": round(wr, 3), "sample_size": total,
                    "message": (
                        f"⚠️ LOW WR / EV-POSITIVE {alert_key}: {wr:.0%} WR over "
                        f"{sample_label} (95% CI upper bound {hi:.0%} < "
                        f"{disable_wr:.0%}), but net EV "
                        f"{ev_obj.get('net_ev', 0):+.3f}%/trade is positive. "
                        "Keeping ENABLED — low WR alone is not a disable condition."
                    ),
                })

            # Auto-disable ONLY on the EV-negative branch.
            if auto_eligible and ev_negative and alert_key not in current_disabled_keys:
                # FIX (Priority 3): Do NOT mutate here. Tag the
                # recommendation so brain_enhanced can execute it
                # after the action gate passes.
                recommendations.append({
                    "type": "auto_disabled", "severity": "high", "alert": alert_key,
                    "pending_auto_action": True,
                    "pending_action": "disable",
                    "message": (
                        f"🔒 Auto-disable CANDIDATE {alert_key}: {wr:.0%} WR over "
                        f"{sample_label}, net EV "
                        f"{ev_obj.get('net_ev', 0):+.3f}%/trade, "
                        f"(≥{auto_disable_min} required). "
                        "[Pending action gate]"
                    ),
                })
        elif lo >= cfg.MIN_WIN_RATE:
            alert_verdicts[alert_key] = "recovered"
            recommendations.append({
                "type": "recovered_alert", "severity": "low", "alert": alert_key,
                "win_rate": round(wr, 3), "sample_size": total,
                "message": (
                    f"RECOVERED: {alert_key} at {wr:.0%} WR over {sample_label} "
                    f"(95% CI lower bound {lo:.0%} ≥ target {cfg.MIN_WIN_RATE:.0%})."
                ),
            })
        else:
            alert_verdicts[alert_key] = "monitor"
            recommendations.append({
                "type": "monitor", "severity": "medium", "alert": alert_key,
                "win_rate": round(wr, 3), "sample_size": total,
                "message": f"{alert_key} viable ({wr:.0%} WR, {sample_label}).",
            })

    # ── Re-enable pass (hysteresis) ──────────────────────────────────
    # Independent of the pre-disable win rate above: a key only comes back
    # on fresh post-disable evidence (see _reenable_evidence).
    if auto_disable_on:
        for dk in sorted(current_disabled_keys):
            ok, ev_detail = self._reenable_evidence(
                dk, shadow_rows, key_history["disabled_at"].get(dk), now_ts,
            )
            if ok:
                recommendations.append({
                    "type": "auto_reenabled", "severity": "medium", "alert": dk,
                    "pending_auto_action": True,
                    "pending_action": "enable",
                    "message": (
                        f"🔓 Re-enable CANDIDATE {dk}: {ev_detail['wr']:.0%} WR over "
                        f"{ev_detail['n']} independent post-disable outcomes "
                        f"(lower bound {ev_detail['lo']:.0%}), net EV "
                        f"{ev_detail['ev']:+.3f}%/trade. [Pending action gate]"
                    ),
                })
            else:
                recommendations.append({
                    "type": "reenable_waiting", "severity": "low", "alert": dk,
                    "message": f"{dk} stays disabled — {ev_detail['reason']}.",
                })
    path_to_keys: Dict[str, List[str]] = defaultdict(list)
    for alert_key in alert_stats:
        path = _resolve_config_path(alert_key)
        if path:
            path_to_keys[path].append(alert_key)

    for path, keys in path_to_keys.items():
        frozen_keys = [k for k in keys if self._is_alert_frozen(k, drift_alerts)]
        active_keys = [k for k in keys if k not in frozen_keys]
        if frozen_keys:
            recommendations.append({
                "type": "config_patch_frozen", "severity": "medium",
                "message": (
                    f"{path}: config patch suppressed for {', '.join(frozen_keys)} — "
                    f"CUSUM drift detected, awaiting manual review."
                ),
            })
        verdicts = {k: alert_verdicts.get(k) for k in active_keys if k in alert_verdicts}
        if not verdicts:
            continue
        if all(v == "disable" for v in verdicts.values()) and len(verdicts) == len([k for k in active_keys if k in alert_stats]):
            if path not in seen_paths:
                seen_paths.add(path)
                config_patch.append({
                    "path": path, "current": True, "suggested": False,
                    "reason": f"All alert types on this config path are underperforming: {', '.join(active_keys)}",
                })
        elif "disable" in verdicts.values() and not all(v == "disable" for v in verdicts.values()):
            bad = [k for k, v in verdicts.items() if v == "disable"]
            good = [k for k, v in verdicts.items() if v != "disable"]
            recommendations.append({
                "type": "investigate", "severity": "medium",
                "message": (
                    f"{path} is shared by {', '.join(active_keys)} — {', '.join(bad)} underperforming but "
                    f"{', '.join(good)} is not. Disabling {path} would also kill the good direction; "
                    f"needs a per-direction config key or manual review."
                ),
            })

    # Warn on any disable-worthy alert with no config path at all (exact or prefix)
    for alert_key, verdict in alert_verdicts.items():
        if verdict == "disable" and not _resolve_config_path(alert_key):
            recommendations.append({
                "type": "unmapped_disable", "severity": "medium",
                "message": (
                    f"{alert_key} is recommended for disable but has no entry in "
                    f"ALERT_CONFIG_MAP — no config_patch was emitted. Add a mapping or disable manually."
                ),
            })
    threshold_rec: Dict[str, Any] = {}
    net_ev = half_kelly = kelly_wr = None
    target_floor: Optional[float] = None
    rec = engine.recommend_threshold(
        real_rows, target_winrate=target_wr, min_sample=min_sample,
    ) if real_rows else {"valid": False}

    # ── Brier Score / Calibration ─────────────────────────────���──────
    brier, cal_curve = engine.brier_score_and_calibration(real_rows)
    cal_alerts = engine.calibration_alert(real_rows)
    has_calibration_data = bool(cal_curve or cal_alerts)

    brier_status: Optional[str] = None
    if has_calibration_data:
        _ev_check = engine.ev_first_objective(real_rows, min_sample=min_sample)
        ev_positive = bool(
            _ev_check.get("valid") and _ev_check.get("net_ev", 0.0) > 0
        )
        brier_status = (
            "Healthy" if (brier < 0.20 and ev_positive)
            else "MISALIBRATED" if brier >= 0.20
            else "CALIBRATED-BUT-EV-NEGATIVE"
        )
        recommendations.append({
            "type": "calibration",
            "severity": "medium" if brier >= 0.20 or cal_alerts else "low",
            "brier_score": round(brier, 4),
            "brier_status": brier_status,
            "message": (
                f"Model Calibration (Brier): {brier:.3f} ({brier_status})"
                + (
                    f" | {len(cal_alerts)} bucket(s) show predicted-vs-observed "
                    f"divergence >10%"
                    if cal_alerts else ""
                )
            ),
        })
    if cal_alerts:
        for ca in cal_alerts[:3]:
            recommendations.append({
                "type": "calibration_divergence",
                "severity": "medium",
                "message": (
                    f"Calibration gap at score {ca['score_floor']:.0f}: "
                    f"predicted {ca['predicted']:.0%} vs observed "
                    f"{ca['observed']:.0%} (n={ca['n']})"
                ),
                # ── FDR: one-sample test, observed rate vs fixed
                # predicted reference from the train split ──
                "n": ca["n"],
                "predicted": ca["predicted"],
                "observed": ca["observed"],
            })

    if rec.get("valid") and abs(rec["recommended"] - cfg.CONFLUENCE_MIN_ABS_SCORE) >= 0.5:
        target_floor = rec["recommended"]
        rec_n = rec["rec_n"]
        rec_wr = rec["rec_wr"]
        ev, rr = rec["rec_ev"], rec["rec_rr"]
        buy_wr, buy_n, sell_wr, sell_n = rec["buy_wr"], rec["buy_n"], rec["sell_wr"], rec["sell_n"]
        direction_note = ""
        if buy_wr is not None and sell_wr is not None:
            direction_note = f" | Buy WR {buy_wr:.0%} ({buy_n}), Sell WR {sell_wr:.0%} ({sell_n})"

        wf = engine.validate_threshold_walk_forward(
            real_rows, target_winrate=target_wr, min_sample=min_sample,
        )
        if wf["valid"] and wf.get("passed") is False:
            wf_note = (
                f"⚠️ NOT applied — failed walk-forward validation: held up on "
                f"{wf['train_n']} older samples but degraded to {wf['holdout_wr']:.0%} WR "
                f"on {wf['holdout_n_at_threshold']} newer, unseen ones "
                f"({wf['degraded_pct']:+.1%} vs train). Likely curve-fit to this window."
            )
            emit_patch = False
        elif wf["valid"] and wf.get("passed") is True:
            wf_note = (
                f"✅ Walk-forward validated: held at {wf['holdout_wr']:.0%} WR on "
                f"{wf['holdout_n_at_threshold']} newer samples it wasn't fit on."
            )
            emit_patch = True
        else:
            wf_note = "ℹ️ Not enough data yet for walk-forward validation — treat as provisional."
            emit_patch = True

        threshold_rec = {
            "type": "confluence_threshold", "severity": "high" if emit_patch else "medium",
            "current_abs_score": cfg.CONFLUENCE_MIN_ABS_SCORE,
            "suggested_abs_score": target_floor,
            "supporting_samples": rec_n, "resulting_wr": round(rec_wr, 3),
            "ev": round(ev, 4), "rr": rr, "walk_forward_passed": wf.get("passed"),
            "confidence": rec.get("confidence"),
            "message": (
                f"Set CONFLUENCE_MIN_ABS_SCORE to {target_floor:.1f} "
                f"(currently {cfg.CONFLUENCE_MIN_ABS_SCORE:.1f}) for {rec_wr:.0%} WR "
                f"[{rec.get('rec_wilson_lo', 0):.0%}-{rec.get('rec_wilson_hi', 0):.0%}], "
                f"confidence {rec.get('confidence', 'N/A')}, "
                f"EV {ev:+.3f}%/trade, R:R {engine.format_rr(rr)} across {rec_n} trades.{direction_note}\n"
                f"Alert frequency: {rec['alerts_per_week_before']:.1f}/wk -> "
                f"{rec['alerts_per_week_after']:.1f}/wk "
                f"(dropping {rec['dropped']}, {rec['dropped_pct']:.0%}).\n"
                f"{wf_note}"
            ),
        }

        # ── Stability Gate check on threshold recommendation ─────────────
        if emit_patch and target_floor is not None:
            history = await self.sdb.load_threshold_history()
            gate_ok, gate_reason = self.stability_gate.approve(
                target_floor, history,
            )
            if not gate_ok:
                threshold_rec["severity"] = "medium"
                threshold_rec["stability_blocked"] = True
                threshold_rec["message"] += (
                    f"⚠️ STABILITY GATE BLOCKED: {gate_reason}. "
                    f"Patch suppressed to prevent oscillation."
                )
                emit_patch = False
            else:
                await self.sdb.save_threshold_value(target_floor)

        # ─ Net EV + Kelly sizing at recommended threshold ───────────────
        rec_subset_kelly = [
            r for r in real_rows if r["score"] >= target_floor
        ] if target_floor else []
        if rec_subset_kelly:
            net_ev, half_kelly, kelly_wr = engine.ev_and_kelly_for(rec_subset_kelly)
            kelly_maes = [r["mae"] for r in rec_subset_kelly if r.get("mae") is not None]
            mae_note = f" | Mean MAE: {statistics.mean(kelly_maes):.2%}" if kelly_maes else ""
            recommendations.append({
                "type": "kelly_sizing",
                "severity": "low",
                "message": (
                    f"Net EV (after fees/slippage): {net_ev:+.3f}%/trade | "
                    f"Half-Kelly position size: {half_kelly:.1%} | "
                    f"WR: {kelly_wr:.0%}{mae_note}"
                ),
            })

        recommendations.append(threshold_rec)
        if emit_patch and target_floor is not None:
            config_patch.append({
                "path": "CONFLUENCE_MIN_ABS_SCORE",
                "current": cfg.CONFLUENCE_MIN_ABS_SCORE,
                "suggested": target_floor,
                "supporting_samples": rec_n,
            })
            rec_subset = [r for r in real_rows if r["score"] >= target_floor]
            avg_total = sum(r["total"] for r in rec_subset) / rec_n if rec_n else 0.0
            suggested_pct = min(100.0, (target_floor / avg_total) * 100.0) if avg_total else cfg.CONFLUENCE_MIN_PCT
            config_patch.append({
                "path": "CONFLUENCE_MIN_PCT", "current": cfg.CONFLUENCE_MIN_PCT,
                "suggested": round(suggested_pct, 1), "supporting_samples": rec_n,
                "note": "Derived from suggested abs score / avg total this window — informational, "
                "the abs score patch above is the one that reliably binds.",
            })
    if cfg.BRAIN_MC_SIMULATIONS > 0:
        # ── Audit gate: MC is O(BRAIN_MC_SIMULATIONS × caps) bootstraps
        # against the same rows. On a 2-day archive it burns ~15s to
        # produce a statistic the audit layer already knows is
        # unreliable. Consult can_run() first; if it says the sample
        # or history is insufficient, record the skip and move on. ──
        _mc_allowed, _mc_reason = audit.can_run("monte_carlo")
        if _mc_allowed:
            _mc_seed = int(
                engine.hash_config_state(
                    CONFLUENCE_WEIGHTS,
                    cfg.CONFLUENCE_MIN_ABS_SCORE,
                    cfg.CONFLUENCE_MIN_PCT,
                ),
                16,
            ) & 0xFFFFFFFF
            mc = engine.monte_carlo_walk_forward(
                real_rows, n_simulations=cfg.BRAIN_MC_SIMULATIONS,
                min_sample=min_sample, target_winrate=target_wr,
                seed=_mc_seed,
            )

            if mc["valid"]:
                robust_icon = "✅ ROBUST" if mc["robustness_score"] > 2.0 else "⚠️ FRAGILE"
                recommendations.append({
                    "type": "monte_carlo_robustness", "severity": "low",
                    "message": (
                        f"Monte Carlo ({mc['n_simulations']} block-bootstrap sims): "
                        f"OOS WR mean {mc['oos_wr_mean']:.0%} ±{mc['oos_wr_std']:.0%}, "
                        f"worst-case (5th pct) {mc['oos_wr_p5']:.0%}. "
                        f"Robustness {mc['robustness_score']:.2f} — {robust_icon}\n"
                        f"Diagnostic only — does not change the config patch above."
                    ),
                })
        else:
            audit.record_analysis(
                "monte_carlo", HealthStatus.INSUFFICIENT_DATA,
                detail=_mc_reason,
            )

    rb = engine.regime_breakdown(real_rows, min_sample=min_sample)
    if rb["valid"] and "wr_gap" in rb:
        trending, ranging = rb["regimes"]["trending"], rb["regimes"]["ranging"]
        gap = rb["wr_gap"]
        gap_note = (
            "NOT regime-neutral — worth tracking separately"
            if abs(gap) > 0.10 else "roughly regime-neutral so far"
        )
        recommendations.append({
            "type": "regime_breakdown", "severity": "low",
            "message": (
                f"Regime split (median ADX {rb['median_adx']:.1f} this window): "
                f"trending WR {trending['wr']:.0%} (n={trending['n']}, {trending['confidence']}) "
                f"vs ranging WR {ranging['wr']:.0%} (n={ranging['n']}, {ranging['confidence']}). "
                f"Gap {gap:+.1%} — {gap_note}.\n"
                f"Regime-specific action: only through the regime gate "
                f"(sample- and OOS-gated, lowers the quality verdict only)."
            ),
        })
    # ── Layered recent/medium/long-history comparison ──
    if getattr(cfg, "ENABLE_LAYERED_WINDOW_ANALYSIS", True):
        recent_days = getattr(cfg, "BRAIN_ANALYSIS_WINDOW_DAYS", 30)
        long_days = getattr(cfg, "BRAIN_LONG_WINDOW_DAYS", 180)      
        can_run_lw, lw_reason = audit.can_run("layered_window")
        if can_run_lw:
            _t0_lw = time.time()
            try:
                recent_lw, medium_lw, long_lw = await self._get_layered_window_rows()
                lwa = engine.layered_window_analysis(
                    recent_lw, medium_lw, long_lw, min_sample=min_sample,
                )
                _elapsed_lw = (time.time() - _t0_lw) * 1000
                if lwa.get("valid"):
                    audit.record_analysis(
                        "layered_window", HealthStatus.OK,
                        detail=f"{len(lwa.get('per_alert', {}))} alerts analyzed",
                        duration_ms=_elapsed_lw,
                    )
                else:
                    audit.record_analysis(
                        "layered_window", HealthStatus.DEGRADED,
                        detail="Analysis returned valid=False",
                        duration_ms=_elapsed_lw,
                    )
            except Exception as e:
                audit.record_analysis_exception("layered_window", e)
                lwa = {"valid": False}
                logging.getLogger("macd_bot").debug(f"Brain: layered window analysis failed: {e}")
        else:
            audit.record_analysis(
                "layered_window", HealthStatus.INSUFFICIENT_DATA,
                detail=lw_reason,
            )
            lwa = {"valid": False}
        if lwa.get("valid"):
            ai_metrics["layered_window_analysis"] = lwa["per_alert"]
            weak_now = [
                (ak, v) for ak, v in lwa["per_alert"].items()
                if v["verdict"] == "historically_good_currently_weak"
            ]
            emerging = [
                (ak, v) for ak, v in lwa["per_alert"].items()
                if v["verdict"] == "recently_emerging_edge"
            ]
            always_weak = [
                ak for ak, v in lwa["per_alert"].items() if v["verdict"] == "always_weak"
            ]
            if weak_now or emerging:
                lines = []
                for ak, v in sorted(weak_now, key=lambda t: t[1]["recent_vs_long_ev_gap"])[:3]:
                    lines.append(
                        f"  {ak}: recent netEV {v['recent']['net_ev']:+.2f}% (n={v['recent']['n']}) "
                        f"vs {long_days}d {v['long']['net_ev']:+.2f}% (n={v['long']['n']}) — "
                        "historically good, currently weak"
                    )
                for ak, v in sorted(emerging, key=lambda t: -t[1]["recent_vs_long_ev_gap"])[:3]:
                    lines.append(
                        f"  {ak}: recent netEV {v['recent']['net_ev']:+.2f}% (n={v['recent']['n']}) "
                        f"vs {long_days}d {v['long']['net_ev']:+.2f}% (n={v['long']['n']}) — "
                        "newly emerging, not yet in the long-history baseline"
                    )
                always_weak_note = (
                    f"\n{len(always_weak)} alert(s) confirmed weak in both windows — not a temporary dip."
                    if always_weak else ""
                )
                recommendations.append({
                    "type": "layered_window_analysis", "severity": "medium",
                    "message": (
                        f"🕰️ Recent ({recent_days}d) vs long-history ({long_days}d) divergence "
                        f"on {len(weak_now) + len(emerging)} alert(s):\n" + "\n".join(lines)
                        + always_weak_note +
                        "\nDiagnostic only — no threshold change applied yet."
                    ),
                })

    # ── Hierarchical pair+direction+alert+regime analysis ──
    if getattr(cfg, "ENABLE_HIERARCHICAL_COMBINATION_ANALYSIS", True):
        can_run_hca, hca_reason = audit.can_run("hierarchical")
        if can_run_hca:
            _t0_hca = time.time()
            try:
                hca = engine.hierarchical_combination_analysis(
                    real_rows,
                    min_leaf_sample=getattr(cfg, "HIERARCHICAL_MIN_LEAF_SAMPLE", 15),
                    shrinkage_k=getattr(cfg, "HIERARCHICAL_SHRINKAGE_K", 20.0),
                )
                _elapsed_hca = (time.time() - _t0_hca) * 1000
                if hca.get("valid"):
                    audit.record_analysis(
                        "hierarchical", HealthStatus.OK,
                        detail=f"{len(hca.get('leaves', {}))} leaves",
                        duration_ms=_elapsed_hca,
                    )
                else:
                    audit.record_analysis(
                        "hierarchical", HealthStatus.DEGRADED,
                        detail="No valid leaves",
                        duration_ms=_elapsed_hca,
                    )
            except Exception as e:
                audit.record_analysis_exception("hierarchical", e)
                hca = {"valid": False}
        else:
            audit.record_analysis(
                "hierarchical", HealthStatus.INSUFFICIENT_DATA,
                detail=hca_reason,
            )
            hca = {"valid": False}
        if hca.get("valid"):
            ai_metrics["hierarchical_combination_analysis"] = hca["leaves"]
            leaves = list(hca["leaves"].values())
            standout = [l for l in leaves if abs(l["vs_alert_dir_baseline"]) >= 0.10]
            if standout:
                standout.sort(key=lambda l: -abs(l["vs_alert_dir_baseline"]))
                lines2 = []
                for l in standout[:5]:
                    lines2.append(
                        f"  {l['pair']}+{l['direction']}+{l['alert_key']}+{l['regime']}: "
                        f"shrunk netEV {l['shrunk_net_ev']:+.2f}% (n={l['n']}, raw {l['raw_net_ev']:+.2f}%) "
                        f"vs alert+dir baseline {l['alert_dir_baseline_net_ev']:+.2f}%"
                    )
                recommendations.append({
                    "type": "hierarchical_combination_analysis", "severity": "medium",
                    "message": (
                        f"🧬 {len(standout)} pair+direction+alert+regime combo(s) diverge "
                        f"meaningfully from their alert+direction baseline:\n" + "\n".join(lines2) +
                        "\nDiagnostic only — no per-combo threshold applied yet."
                    ),
                })

    # ── Alert-family intelligence (roadmap #9) ──
    if getattr(cfg, "ENABLE_ALERT_FAMILY_ANALYSIS", True) and real_rows:
        try:
            fam = engine.alert_family_analysis(
                real_rows,
                min_sample=getattr(cfg, "ALERT_FAMILY_MIN_SAMPLE", 20),
                shrinkage_k=getattr(cfg, "HIERARCHICAL_SHRINKAGE_K", 20.0),
            )
            ai_metrics["alert_family_analysis"] = fam
            if fam.get("valid"):
                fam_lines = []
                for name, stats in sorted(
                    (fam.get("families") or {}).items(),
                    key=lambda kv: -(kv[1].get("n") or 0),
                ):
                    if not stats.get("valid"):
                        continue
                    fam_lines.append(
                        f"  • {name}: WR {stats['wr']:.0%} n={stats['n']} "
                        f"netEV {stats.get('net_ev', 0):+.2f}% "
                        f"[{stats.get('confidence', '?')}] "
                        f"state={stats.get('evidence_state', '?')}"
                    )
                if fam_lines:
                    recommendations.append({
                        "type": "alert_family_analysis",
                        "severity": "low",
                        "message": (
                            "👨‍👩‍👧‍👦 Alert-family intelligence:\n"
                            + "\n".join(fam_lines[:8])
                            + "\nDiagnostic only — families are learning entities, not live gates."
                        ),
                    })
        except Exception as e:
            audit.record_analysis_exception("alert_family_analysis", e)

    # ── Regime transition analysis (roadmap #15) ──
    if getattr(cfg, "ENABLE_REGIME_TRANSITION_ANALYSIS", True) and real_rows:
        try:
            rta = engine.regime_transition_analysis(
                real_rows,
                min_sample=max(10, min_sample // 2),
                lookback_stable=getattr(cfg, "REGIME_TRANSITION_LOOKBACK_BARS", 4),
                post_window_hours=getattr(cfg, "REGIME_TRANSITION_POST_WINDOW_HOURS", 6),
            )
            ai_metrics["regime_transition_analysis"] = rta
            if rta.get("valid"):
                pt = rta.get("post_transition") or {}
                st = rta.get("stable") or {}
                gap = rta.get("wr_gap")
                msg_parts = [
                    f"🔄 Regime transitions detected: {rta.get('n_transitions', 0)}",
                ]
                if st.get("valid"):
                    msg_parts.append(
                        f"Stable regime WR {st['wr']:.0%} (n={st['n']})"
                    )
                if pt.get("valid"):
                    msg_parts.append(
                        f"Post-transition WR {pt['wr']:.0%} (n={pt['n']})"
                    )
                if gap is not None:
                    msg_parts.append(f"gap {gap:+.0%}")
                recommendations.append({
                    "type": "regime_transition_analysis",
                    "severity": "medium" if gap is not None and abs(gap) > 0.08 else "low",
                    "message": " | ".join(msg_parts)
                    + " — diagnostic; alerts right after a regime flip may behave differently.",
                })
        except Exception as e:
            audit.record_analysis_exception("regime_transition_analysis", e)

    # ── Strategy vs regime attribution (roadmap #17) ──
    if getattr(cfg, "ENABLE_STRATEGY_VS_REGIME_ATTRIBUTION", True) and real_rows:
        try:
            sva = engine.strategy_vs_regime_attribution(
                real_rows,
                min_sample=getattr(cfg, "STRATEGY_VS_REGIME_MIN_SAMPLE", 30),
            )
            ai_metrics["strategy_vs_regime_attribution"] = sva
            if sva.get("valid"):
                attr = sva.get("attribution", "unknown")
                sev = (
                    "high" if attr == "possible_strategy_degradation"
                    else "medium" if attr in (
                        "insufficient_current_regime_evidence", "regime_mix_shift"
                    )
                    else "low"
                )
                recommendations.append({
                    "type": "strategy_vs_regime_attribution",
                    "severity": sev,
                    "attribution": attr,
                    "message": (
                        f"🧭 Strategy vs regime: {attr.replace('_', ' ')}\n"
                        f"{sva.get('detail', '')}"
                    ),
                })
        except Exception as e:
            audit.record_analysis_exception("strategy_vs_regime_attribution", e)

    if target_floor is not None:
        attribution = engine.outcome_attribution(
            real_rows, CONFLUENCE_WEIGHTS, threshold=target_floor, min_sample=min_sample,
        )
        flagged = [
            e for e in attribution
            if e.get("rescued_valid") and e["n_rescued"] >= min_sample and e["rescued_wr"] < target_wr - 0.10
        ]
        if flagged:
            lines = [
                f"  • {e['vote']}: rescues {e['n_rescued']} trades ({e['rescued_pct']:.0%} of its True cases) "
                f"at only {e['rescued_wr']:.0%} WR [{e['rescued_wilson_lo']:.0%}-{e['rescued_wilson_hi']:.0%}]"
                for e in flagged[:5]
            ]
            recommendations.append({
                "type": "outcome_attribution", "severity": "medium",
                "message": (
                    f"Outcome attribution at threshold {target_floor:.1f}: {len(flagged)} vote(s) are "
                    f"propping up trades that clear the bar only because of that vote's weight, and "
                    f"those specific trades underperform target WR:\n" + "\n".join(lines) + "\n"
                    "Consider re-checking these votes' weights — this is diagnostic, no config "
                    "patch is auto-applied."
                ),
            })
    anomalies_check = engine.flag_anomalous_rows(real_rows, min_sample=min_sample)
    if anomalies_check["valid"] and anomalies_check["n_flagged"] > 0:
        top = anomalies_check["flagged"][:5]
        anomaly_lines = [
            f"  • {f['pair']} {f['alert_key']} pct_move={f['pct_move']:+.1f}% "
            f"(robust z={f['robust_z']:.1f}, ts={f['entry_ts']})"
            for f in top
        ]
        recommendations.append({
            "type": "data_anomaly", "severity": "medium",
            "message": (
                f"⚠️ {anomalies_check['n_flagged']} of {anomalies_check['n_total']} outcome "
                f"rows have a pct_move statistically far from the rest (median "
                f"{anomalies_check['median_pct_move']:+.2f}%):\n" + "\n".join(anomaly_lines) + "\n"
                "Worth checking these against exchange data for a bad tick before trusting "
                "the EV/WR numbers above. Not auto-excluded — could be a real outsized move."
            ),
        })

    # ── Three-Metric Outcome Analysis ────────────────────────────────
    mm_summary = engine.multi_metric_summary(real_rows, min_sample=min_sample)
    if mm_summary.get("valid"):
        close_wr = mm_summary["close_wr"]
        mfe_wr = mm_summary["mfe_wr"]
        mae_rate = mm_summary["mae_loss_rate"]
        clean_wr = mm_summary["clean_win_rate"]
        gap = mfe_wr - close_wr
        summary_msg = (
            f"📐 Three-Metric Evaluation (n={mm_summary['n']}):\n"
            f"  Close WR (point-in-time): {close_wr:.0%} "
            f"[{mm_summary['close_wilson'][0]:.0%}-{mm_summary['close_wilson'][1]:.0%}]\n"
            f"  • MFE WR (TP ever hit):    {mfe_wr:.0%} "
            f"[{mm_summary['mfe_wilson'][0]:.0%}-{mm_summary['mfe_wilson'][1]:.0%}]\n"
            f"  • MAE Loss Rate (SL hit):  {mae_rate:.0%}\n"
            f" Clean Win (TP w/o SL):   {clean_wr:.0%}"
        )
        if gap > 0.05:
            # FIX (Priority 8): MFE is opportunity/target-reach rate,
            # not "true profitability". tp_first is the defensible proxy.
            summary_msg += (
                f"\n⚠️ Gap: {gap:+.0%} of trades reached TP level but reversed "
                f"before candle {cfg.OUTCOME_LOOKAHEAD_CANDLES}. "
                f"MFE target-reach rate exceeds close-based WR by {gap:.0%} — "
                f"this represents opportunities reached, not necessarily captured. "
                f"The tp_first ordering metric is the more defensible execution proxy."
            )
        if mm_summary.get("tp_before_sl_rate") is not None:
            summary_msg += (
                f"\n• TP before SL: {mm_summary['tp_before_sl_rate']:.0%} "
                f"| SL before TP: {mm_summary.get('sl_before_tp_rate', 0):.0%} "
                f"(n={mm_summary.get('ordering_sample', '?')})"
            )
        if mm_summary.get("bonus_rate") is not None:
            summary_msg += (
                f"\n  • Bonus wins (≥{cfg.OUTCOME_BONUS_RR:.0f}R): "
                f"{mm_summary['bonus_rate']:.0%} of trades"
                f" | Avg R-multiple: {mm_summary['avg_rr_achieved']:.2f}R"
                f" | Bonus-weighted WR: {mm_summary.get('weighted_wr', 0):.1%}"
            )
        recommendations.append({
            "type": "three_metric_evaluation",
            "severity": "medium" if gap > 0.10 else "low",
            "close_wr": round(close_wr, 4),
            "mfe_wr": round(mfe_wr, 4),
            "mae_loss_rate": round(mae_rate, 4),
            "clean_win_rate": round(clean_wr, 4),
            "gap": round(gap, 4),
            "bonus_rate": round(mm_summary.get("bonus_rate", 0.0), 4),
            "avg_rr_achieved": round(mm_summary.get("avg_rr_achieved", 0.0), 2),
            "weighted_wr": round(mm_summary.get("weighted_wr", 0.0), 4),
            # ─ FDR: exact McNemar on the discordant 2×2 cells ──
            # Not a two-proportion test — see mcnemar_exact_p docstring.
            "n": mm_summary["n"],
            "mfe_only": mm_summary.get("mfe_only", 0),
            "close_only": mm_summary.get("close_only", 0),
            "message": summary_msg,
        })

    # Per-alert three-metric breakdown (worst offenders only)
    mm_per_alert = engine.multi_metric_per_alert(real_rows, min_sample=min_sample)
    big_gap_alerts = [a for a in mm_per_alert if a["gap_mfe_vs_close"] > 0.15]
    if big_gap_alerts:
        gap_lines = [
            f"  • {a['alert_key']}: close {a['close_wr']:.0%} vs MFE {a['mfe_wr']:.0%} "
            f"(gap {a['gap_mfe_vs_close']:+.0%}, n={a['n']})"
            for a in big_gap_alerts[:5]
        ]
        recommendations.append({
            "type": "close_vs_mfe_gap",
            "severity": "medium",
            "message": (
                "🔍 Alerts where MFE WR >> Close WR (take-profit would have captured "
                "these wins but the point-in-time check misses them):\n"
                + "\n".join(gap_lines)
            ),
        })

    # ── R:R and Bonus Analysis ──
    if real_rows:
        bonus_wins = sum(1 for r in real_rows if r.get("bonus_win"))
        total_wins = sum(1 for r in real_rows if r["win"])
        rr_values = [r.get("rr_achieved", 0) for r in real_rows if r.get("rr_achieved", 0) > 0]
        avg_rr = statistics.mean(rr_values) if rr_values else 0.0
        total_weight = sum(r.get("win_weight", 1.0 if r["win"] else 0.0) for r in real_rows)
        effective_n = len(real_rows)
        weighted_wr_bonus = min(total_weight / effective_n, 1.0) if effective_n else 0.0
        recommendations.append({
            "type": "rr_analysis",
            "severity": "low",
            "message": (
                f"📐 R:R Analysis (target=1:{cfg.OUTCOME_RR_TARGET:.0f}, "
                f"bonus≥{cfg.OUTCOME_BONUS_RR:.0f}R):\n"
                f"  • Wins: {total_wins}/{len(real_rows)} | "
                f"Bonus wins: {bonus_wins} ({bonus_wins/max(total_wins,1):.0%} of wins)\n"
                f"  • Avg R-multiple achieved: {avg_rr:.2f}R\n"
                f"  • Effective WR (bonus-weighted): {weighted_wr_bonus:.1%} "
                f"(raw: {total_wins/len(real_rows):.1%})"
            ),
        })
    if rec.get("overlapping_toxic"):
        worst = max(rec["overlapping_toxic"], key=lambda t: t[1])  # type: ignore[arg-type]
        recommendations.append({
            "type": "toxic_zone_note", "severity": "low",
            "message": (
                f"Note: a toxic bucket (score {worst[0]:.1f}-{worst[1]:.1f}, {worst[2]:.0%} WR) "
                f"exists at or above the recommended threshold. Cumulative stats already price "
                f"this in — worth checking the per-alert breakdown below for what's firing there."
            ),
        })

    # ── Temporal drift ──
    if rec.get("valid") and rec.get("drift_recent_wr") is not None:
        drift = rec["drift_recent_wr"] - rec["drift_older_wr"]
        if drift < -0.05:
            recommendations.append({
                "type": "temporal_drift", "severity": "high",
                "message": (
                    f"Edge may be decaying: last 14d WR {rec['drift_recent_wr']:.0%} "
                    f"({rec['drift_recent_n']} samples) vs prior WR {rec['drift_older_wr']:.0%} "
                    f"(Δ{drift:+.0%})."
                ),
            })

    # ── Per-pair breakdown ──
    pair_stats = engine.per_pair_breakdown(real_rows, min_sample=min_sample)
    weak_pairs = [p for p in pair_stats if p[1] < disable_wr]
    if weak_pairs:
        recommendations.append({
            "type": "weak_pairs", "severity": "medium",
            "message": (
                "Underperforming pairs: " +
                ", ".join(f"{p}({wr:.0%}, n={n})" for p, wr, n in weak_pairs[:5])
            ),
        })

    # ── Per-pair session breakdown ──
    if getattr(cfg, "ENABLE_SESSION_FILTER", False):
        session_stats = engine.per_pair_session_breakdown(real_rows, min_sample=min_sample)
        weak_sessions = [s for s in session_stats if s[2] < disable_wr]
        if weak_sessions:
            recommendations.append({
                "type": "weak_pair_sessions", "severity": "low",
                "message": (
                    "Underperforming pair:session combos: " +
                    ", ".join(
                        f"{pair}/{session}({wr:.0%}, n={n})"
                        for pair, session, wr, n in weak_sessions[:8]
                    )
                ),
            })

    # ── Pain-Adjusted Win Rate ──
    pawr_stats = engine.pain_adjusted_win_rate(real_rows, min_sample=min_sample)
    if pawr_stats:
        worst_pain = sorted(
            pawr_stats.items(), key=lambda kv: kv[1]["raw_wr"] - kv[1]["pawr"], reverse=True
        )[:5]
        if worst_pain and (worst_pain[0][1]["raw_wr"] - worst_pain[0][1]["pawr"]) >= 0.03:
            recommendations.append({
                "type": "pain_scoring", "severity": "low",
                "message": (
                    "Highest drawdown-masked win rates (raw WR vs Pain-Adjusted WR): " +
                    ", ".join(
                        f"{ak}({s['raw_wr']:.0%}->{s['pawr']:.0%}, mean MAE={s['mean_mae']:.2%}, n={s['n']})"
                        for ak, s in worst_pain
                    )
                ),
            })

    # ── Per-pair confluence thresholds ──
    if getattr(cfg, "ENABLE_PAIR_THRESHOLDS", False):
        pair_min_sample = getattr(cfg, "BRAIN_PAIR_THRESHOLD_MIN_SAMPLE", 30)
        pair_recs = engine.per_pair_thresholds(
            real_rows, target_winrate=target_wr, min_sample=pair_min_sample,
        )
        current_pair_thresholds = await self.sdb.get_pair_thresholds()
        pair_threshold_lines = []

        for pair, prec in pair_recs.items():
            suggested = prec["recommended"]
            current = current_pair_thresholds.get(pair, cfg.CONFLUENCE_MIN_ABS_SCORE)
            if abs(suggested - current) < 0.5:
                continue

            pair_rows = [r for r in real_rows if r["pair"] == pair]
            pair_wf = engine.validate_threshold_walk_forward(
                pair_rows, target_winrate=target_wr, min_sample=pair_min_sample,
            )

            if pair_wf.get("valid") and pair_wf.get("passed") is False:
                pair_threshold_lines.append(
                    f"  • {pair}: suggested {suggested:.1f} (was {current:.1f}) — "
                    f"NOT applied, failed walk-forward "
                    f"({pair_wf['holdout_wr']:.0%} holdout WR on n={len(pair_rows)})"
                )
                continue

            history = await self.sdb.load_threshold_history(key_suffix=pair)
            gate_ok, gate_reason = self.stability_gate.approve(suggested, history)
            if not gate_ok:
                pair_threshold_lines.append(
                    f"  • {pair}: suggested {suggested:.1f} (was {current:.1f}) — "
                    f"NOT applied, stability gate: {gate_reason}"
                )
                continue

            await self.sdb.save_threshold_value(suggested, key_suffix=pair)
            applied = await self.sdb.set_pair_threshold(pair, suggested)
            if applied:
                _wf_tag = (
                    f", WF {pair_wf['holdout_wr']:.0%} holdout WR"
                    if pair_wf.get("valid") and pair_wf.get("passed") is True
                    else ", WF provisional"
                )
                pair_threshold_lines.append(
                    f"  • {pair}: {current:.1f} -> {suggested:.1f} "
                    f"({prec['rec_wr']:.0%} WR, n={prec['rec_n']}{_wf_tag}) [applied]"
                )

        if pair_threshold_lines:
            recommendations.append({
                "type": "pair_thresholds", "severity": "medium",
                "message": (
                    "Per-pair confluence thresholds (overriding CONFLUENCE_MIN_ABS_SCORE "
                    "for these pairs only):\n" + "\n".join(pair_threshold_lines)
                ),
            })

    # ── Vote importance ──
    vote_imp = engine.vote_importance(real_rows, min_sample=min_sample)
    if vote_imp:
        best = [v for v in vote_imp if v[5] > 0.05][:3]
        worst = [v for v in vote_imp if v[5] < -0.05][-3:]
        if best or worst:
            parts = []
            if best:
                parts.append("adding edge: " + ", ".join(f"{v[0]}(+{v[5]:.0%})" for v in best))
            if worst:
                parts.append("adding noise: " + ", ".join(f"{v[0]}({v[5]:+.0%})" for v in worst))
            recommendations.append({
                "type": "vote_importance", "severity": "low",
                "message": "Vote signal quality — " + "; ".join(parts),
            })

    # ── Shadow-mode insight ──
    shadow_summary: Dict[str, Any] = {}
    if shadow_rows:
        shadow_wins = sum(1 for r in shadow_rows if r["win"])
        shadow_total = len(shadow_rows)
        hiconf = [r for r in shadow_rows if r["conf_pct"] >= cfg.BRAIN_REWARDABLE_MIN_CONFLUENCE_PCT]
        hiconf_wins = sum(1 for r in hiconf if r["win"])
        shadow_summary = {
            "total_tracked": shadow_total,
            "overall_wr": round(shadow_wins / shadow_total, 3) if shadow_total else None,
            "high_confluence_tracked": len(hiconf),
            "high_confluence_wr": round(hiconf_wins / len(hiconf), 3) if hiconf else None,
        }
        if hiconf and len(hiconf) >= cfg.BRAIN_REWARDABLE_MIN_SHADOW_SAMPLE:
            hiconf_wr = hiconf_wins / len(hiconf)
            if hiconf_wr >= cfg.BRAIN_REWARDABLE_MIN_SHADOW_WR:
                recommendations.append({
                    "type": "rewardable_pool", "severity": "medium",
                    "sample_size": len(hiconf), "win_rate": round(hiconf_wr, 3),
                    "message": (
                        f"Rejected alerts at >={cfg.BRAIN_REWARDABLE_MIN_CONFLUENCE_PCT:.0f}% confluence "
                        f"are winning {hiconf_wr:.0%} of the time ({len(hiconf)} tracked) — "
                        f"the rewardable-override gate is active and finding real edge."
                    ),
                })
    # ── Vote-Count OOD summary ──
    ood_status = "Normal"
    if real_rows:
        latest_by_alert: Dict[str, engine.Row] = {}
        for r in reversed(real_rows):
            ak = r["alert_key"]
            if ak not in latest_by_alert and r.get("votes"):
                latest_by_alert[ak] = r
            if len(latest_by_alert) >= 5:
                break
        ood_passes = 0
        ood_total = 0
        for ak, r in latest_by_alert.items():
            is_ood, detail = engine.is_vote_pattern_ood(real_rows, r["votes"], ak)
            ood_total += 1
            if not is_ood:
                ood_passes += 1
        if ood_total > 0:
            ood_status = (
                "PASS" if ood_passes == ood_total
                else f"{ood_passes}/{ood_total} PASS"
            )

    # ── Calibration LIVE gate: build + persist per-alert curves ──
    calib = engine.build_calibration_curves(
        real_rows,
        bucket_pct=getattr(cfg, "CALIBRATION_BUCKET_PCT", 5.0),
        min_sample=getattr(cfg, "CALIBRATION_MIN_SAMPLE", 15),
        shadow_rows=shadow_rows,
    )
    if calib.get("curves"):
        calibration_persisted = await self._persist_calibration_curves(calib)
        ai_metrics["calibration_ece_mean"] = calib.get("ece_mean")
        ai_metrics["calibration_persistence"] = "SUCCESS" if calibration_persisted else "FAILED"
        if getattr(cfg, "ENABLE_CALIBRATION_GATE", False):
            miscal = sorted(
                ((ak, c["ece"], c["n"]) for ak, c in calib["curves"].items()
                 if c["ece"] > 0.12 and c["n"] >= 30),
                key=lambda t: -t[1],
            )
            recommendations.append({
                "type": "calibration_gate_active",
                "severity": "medium" if miscal else ("low" if calibration_persisted else "high"),
                "message": (
                    (
                        "🎯 Calibration gate armed: dispatch filters on calibrated WR, "
                        "not raw confluence %. "
                        if calibration_persisted else
                        "⚠️ Calibration curves computed but NOT persisted to Redis — "
                        "live dispatch gate will fail-open on a stale/missing curve until "
                        "the next successful persist. "
                    )
                    + f"Mean per-alert ECE {calib.get('ece_mean', 0):.3f} "
                    "(historical/in-sample over per-alert-key curves — "
                    "not a global OOS ECE)."
                    + (
                        " Most miscalibrated: "
                        + ", ".join(f"{ak} (ECE {ece:.2f}, n={n})" for ak, ece, n in miscal[:5])
                        if miscal else ""
                    )
                ),
            })

# ── Trade-quality inputs: per-alert EV evidence + regime, persisted
    # for the dispatch-time get_trade_quality() hook (report-only) ──
    ev_by_alert: Dict[str, Any] = {}
    for alert_key, s in alert_stats.items():
        if len(s["rows"]) < min_sample:
            continue
        ev_obj = engine.ev_first_objective(s["rows"], min_sample=min_sample)
        if ev_obj.get("valid"):
            ev_obj = dict(ev_obj)
            _r_wr, _o_wr, _r_n = engine.detect_temporal_drift(s["rows"])
            if _r_wr is not None:
                ev_obj["recent_wr"] = round(_r_wr, 4)
                ev_obj["wr_drop_recent"] = round(_o_wr - _r_wr, 4)
            _train_wf, _hold_wf = engine.walk_forward_split(s["rows"])
            _oos_ok = False
            if len(_hold_wf) >= 20:
                _ho_obj = engine.ev_first_objective(_hold_wf, min_sample=20)
                _oos_ok = bool(
                    _ho_obj.get("valid")
                    and _ho_obj["p_ev_positive"] >= getattr(cfg, "BRAIN_EV_GATE_P_THRESHOLD", 0.85)
                )
            ev_obj["oos_validated"] = _oos_ok
            ev_obj["n_holdout"] = len(_hold_wf)
            ev_by_alert[alert_key] = ev_obj

    mae_mfe_profiles: Dict[str, Any] = {}
    if getattr(cfg, "ENABLE_MAE_MFE_TRADE_PLAN", True) and real_rows:
        try:
            mae_mfe_profiles = engine.mae_mfe_profiles_by_bucket(
                real_rows,
                min_sample=getattr(cfg, "MAE_MFE_MIN_SAMPLE", 15),
                sl_percentile=getattr(cfg, "MAE_MFE_SL_PERCENTILE", 70.0),
                tp1_percentile=getattr(cfg, "MAE_MFE_TP1_PERCENTILE", 60.0),
                tp2_percentile=getattr(cfg, "MAE_MFE_TP2_PERCENTILE", 85.0),
                sl_min_pct=getattr(cfg, "MAE_MFE_SL_MIN_PCT", 0.15),
                sl_max_pct=getattr(cfg, "MAE_MFE_SL_MAX_PCT", 3.0),
            )
        except Exception as e:
            audit.record_analysis_exception("mae_mfe_trade_plan", e)
            mae_mfe_profiles = {}

    # ── Adaptive dedup windows: always computed + persisted so they can be
    # inspected; the live bot only uses them if ENABLE_ADAPTIVE_DEDUP_WINDOWS ──
    try:
        _dedup_windows = engine.adaptive_dedup_windows(
            list(real_rows) + list(shadow_rows or []),
            min_gaps=int(getattr(cfg, "ADAPTIVE_DEDUP_MIN_GAPS", 30)),
            percentile=float(getattr(cfg, "ADAPTIVE_DEDUP_PERCENTILE", 10.0)),
            lo_sec=int(getattr(cfg, "ADAPTIVE_DEDUP_MIN_SEC", 120)),
            hi_sec=int(getattr(cfg, "ADAPTIVE_DEDUP_MAX_SEC", 1800)),
        )
        ai_metrics["adaptive_dedup_windows"] = _dedup_windows
        if _dedup_windows and not self.sdb.degraded:
            await self.sdb.set_metadata(
                "adaptive_dedup_windows", json_dumps(_dedup_windows), ttl=30 * 86400,
            )
    except Exception as e:
        audit.record_analysis_exception("adaptive_dedup_windows", e)

    # ── Validated TP/SL zones: sample-gated, OOS-replayed, streak-promoted.
    # Only PROMOTED zones are persisted for dispatch; candidates and the
    # reasons they failed are reported so the gap to promotion is visible. ──
    zone_blob: Dict[str, Any] = {}
    if str(getattr(cfg, "ZONE_MODE", "live")) != "off" and real_rows:
        try:
            zres = engine.zone_candidates(
                real_rows,
                min_n=int(getattr(cfg, "ZONE_MIN_N", 60)),
                min_holdout=int(getattr(cfg, "ZONE_MIN_HOLDOUT", 20)),
                min_delta_pct=float(getattr(cfg, "ZONE_MIN_DELTA_PCT", 0.05)),
                min_p_better=float(getattr(cfg, "ZONE_MIN_P_BETTER", 0.80)),
                stability_tol=float(getattr(cfg, "ZONE_STABILITY_TOL", 0.35)),
                max_deviation=float(getattr(cfg, "ZONE_MAX_DEVIATION", 2.0)),
                min_rr=float(getattr(cfg, "ZONE_MIN_RR", 1.0)),
                sl_percentile=float(getattr(cfg, "MAE_MFE_SL_PERCENTILE", 70.0)),
                tp1_percentile=float(getattr(cfg, "MAE_MFE_TP1_PERCENTILE", 60.0)),
                tp2_percentile=float(getattr(cfg, "MAE_MFE_TP2_PERCENTILE", 85.0)),
                sl_min_pct=float(getattr(cfg, "MAE_MFE_SL_MIN_PCT", 0.15)),
                sl_max_pct=float(getattr(cfg, "MAE_MFE_SL_MAX_PCT", 3.0)),
            )
            _prev_streaks: Dict[str, int] = {}
            if not self.sdb.degraded:
                _raw_streaks = await self.sdb.get_metadata("zone_pass_streaks")
                if _raw_streaks:
                    try:
                        _prev_streaks = {str(k): int(v) for k, v in json_loads(_raw_streaks).items()}
                    except Exception:
                        _prev_streaks = {}
            _promoted, _streaks = engine.zone_promote(
                zres["candidates"], _prev_streaks,
                int(getattr(cfg, "ZONE_PROMOTE_CONSECUTIVE", 2)),
            )
            if not self.sdb.degraded:
                await self.sdb.set_metadata(
                    "zone_pass_streaks", json_dumps(_streaks), ttl=30 * 86400,
                )
            zone_blob = {"median_adx": zres["median_adx"], "zones": _promoted}
            _cands = zres["candidates"]
            ai_metrics["zone_profiles"] = {
                "n_candidates": len(_cands),
                "n_passing": sum(1 for c in _cands.values() if c.get("passed")),
                "n_promoted": len(_promoted),
                "candidates": _cands,
            }
            if _cands:
                _top = sorted(
                    _cands.values(),
                    key=lambda c: (not c.get("passed"), -(c.get("delta_ev") or -9.0)),
                )[:5]
                recommendations.append({
                    "type": "zone_profiles", "severity": "low",
                    "message": (
                        f"🎯 TP/SL zones (mode {getattr(cfg, 'ZONE_MODE', 'live')}): "
                        f"{len(_cands)} candidate(s), {sum(1 for c in _cands.values() if c.get('passed'))} "
                        f"passing OOS, {len(_promoted)} promoted "
                        f"(needs {int(getattr(cfg, 'ZONE_PROMOTE_CONSECUTIVE', 2))} consecutive passes):\n"
                        + "\n".join(
                            f"  {'PROMOTED' if c['bucket'] in _promoted else ('pass ' + str(_streaks.get(c['bucket'], 0)) if c.get('passed') else 'fail')} "
                            f"{c['bucket']}: "
                            + (f"SL {c['sl_pct']:.2f}% TP1 {c['tp1_pct']:.2f}% "
                               f"ΔEV {c.get('delta_ev', 0.0):+.2f}% (n={c['n']})"
                               if c.get("sl_pct") is not None else "no profile")
                            + ("" if c.get("passed") else f" [{', '.join(c['reasons'][:2])}]")
                            for c in _top
                        )
                        + "\nZones only change the SL/TP shown in alerts; outcome labels keep the fixed bracket."
                    ),
                })
        except Exception as e:
            audit.record_analysis_exception("zone_profiles", e)
            zone_blob = {}

    # ── Regime gate: regime-conditioned, sample- and OOS-gated evidence for
    # the dispatch-time quality verdict (restrict-only) ──
    regime_gate_blob: Dict[str, Any] = {}
    if str(getattr(cfg, "REGIME_GATE_MODE", "live")) != "off" and real_rows:
        try:
            regime_gate_blob = engine.regime_gate_analysis(
                real_rows,
                min_n_downgrade=int(getattr(cfg, "REGIME_GATE_MIN_N_DOWNGRADE", 50)),
                min_n_block=int(getattr(cfg, "REGIME_GATE_MIN_N_BLOCK", 100)),
                min_holdout=int(getattr(cfg, "REGIME_GATE_MIN_HOLDOUT", 20)),
                downgrade_p=float(getattr(cfg, "REGIME_GATE_DOWNGRADE_P", 0.35)),
                block_p=float(getattr(cfg, "REGIME_GATE_BLOCK_P", 0.20)),
                min_gap_pct=float(getattr(cfg, "REGIME_GATE_MIN_GAP_PCT", 0.10)),
            )
            ai_metrics["regime_gate"] = regime_gate_blob
            _acting = [
                s for s in regime_gate_blob.get("segments", {}).values()
                if s.get("action") in ("BLOCK", "DOWNGRADE")
            ]
            if _acting:
                _acting.sort(key=lambda s: (s["action"] != "BLOCK", s["net_ev"]))
                recommendations.append({
                    "type": "regime_gate", "severity": "medium",
                    "message": (
                        f"🚦 Regime gate: {len(_acting)} alert+direction+regime segment(s) "
                        f"are weak specifically in that regime (mode "
                        f"{getattr(cfg, 'REGIME_GATE_MODE', 'live')}):\n"
                        + "\n".join(
                            f"  {s['action']} {s['alert_key']} {s['direction']} in {s['regime']}: "
                            f"netEV {s['net_ev']:+.2f}% vs {(s['baseline_net_ev'] or 0.0):+.2f}% overall "
                            f"(P={s['p_ev_positive']:.0%}, n={s['n']})"
                            for s in _acting[:5]
                        )
                        + "\nLowers the quality verdict only; hard signal gates are unchanged."
                    ),
                })
        except Exception as e:
            audit.record_analysis_exception("regime_gate", e)
            regime_gate_blob = {}

    if ev_by_alert:
        await self._persist_quality_inputs({
            "ev_by_alert": ev_by_alert,
            "regime_info": rb,
            "hierarchical_leaves": hca.get("leaves", {}) if hca.get("valid") else {},
            "hierarchical_median_adx": hca.get("median_adx"),
            "regime_gate": regime_gate_blob if regime_gate_blob.get("valid") else {},
            "zone_profiles": zone_blob,
            "mae_mfe_profiles": mae_mfe_profiles,
            "ts": int(time.time()),
        })

    # ── Market-state model: report-only refresh each cycle. Discarded
    # (previous persisted model kept as-is) unless it clears OOS EV
    if getattr(cfg, "ENABLE_MARKET_STATE_MODEL", True):
        ms_model = engine.train_market_state_model(
            real_rows,
            min_sample=getattr(cfg, "MARKET_STATE_MODEL_MIN_SAMPLE", 150),
            min_oos_p_ev_positive=getattr(cfg, "MARKET_STATE_MODEL_MIN_OOS_P", 0.70),
        )
        ai_metrics["market_state_model"] = {
            k: v for k, v in ms_model.items() if k != "beta"
        }
        if ms_model.get("valid"):
            await self._persist_market_state_model(ms_model)
            # Persist ML calibration curve for dispatch-time lookup
            ml_curve = ms_model.get("ml_calibration")
            if ml_curve and ml_curve.get("buckets"):
                await self._persist_ml_calibration_curve(ml_curve)
            recommendations.append({
                "type": "market_state_model_refreshed",
                "severity": "low",
                "message": (
                    f"🤖 Market-state model refreshed: n_train={ms_model['n_train']}, "
                    f"OOS P(EV>0)={ms_model['holdout_ev']['p_ev_positive']:.0%} "
                    f"on n_holdout={ms_model['n_holdout']}."
                ),
            })
            # Surface ECE comparison
            ml_ece = ms_model.get("ml_ece")
            conf_ece = ai_metrics.get("calibration_ece_mean")

            if ml_ece is not None:
                _ml_n = ms_model.get("n_holdout", 0)
                _conf_n = len(real_rows)
                _same_sample = (
                    _conf_n > 0
                    and _ml_n > 0
                    and abs(_ml_n - _conf_n) / max(_conf_n, 1) < 0.15
                )
                _populations_note = (
                    " Underlying populations differ: "
                    "ML uses holdout-only rows from a purge/embargo split; "
                    "conf_pct uses all real rows plus eligible shadow rows "
                    "(calibration_gate-rejected rows excluded), fit in-sample."
                )
                if not _same_sample:
                    _caveat = (
                        " ⚠️ Different evaluation samples: "
                        f"ML n={_ml_n}, conf_pct n={_conf_n}."
                        f"{_populations_note} "
                        f"ECE values are descriptive only — not a valid "
                        f"basis for model-selection between the two."
                    )
                else:
                    _caveat = (
                        " Evaluation sample sizes are similar "
                        f"(ML n={_ml_n}, conf_pct n={_conf_n})."
                        f"{_populations_note} "
                        f"Treat as descriptive, not a model-selection verdict."
                    )
                recommendations.append({
                    "type": "ml_vs_conf_ece",
                    "severity": "low",
                    "message": (
                        f"📊 ML ECE={ml_ece:.3f} vs "
                        f"conf_pct mean-per-alert ECE={conf_ece}."
                        f"{_caveat}"
                    ),
                })
        else:
            recommendations.append({
                "type": "market_state_model_rejected",
                "severity": "low",
                "message": (
                    f"🤖 Market-state model NOT updated this cycle "
                    f"({ms_model.get('error', 'unknown')}) — previous model, if any, kept live."
                ),
            })
            drift = ms_model.get("drift_check") or {}
            drifted = drift.get("drifted_features") if drift.get("valid") else None
            if drifted:
                logging.getLogger("macd_bot").warning(
                    f"Market-state model rejected this cycle AND feature drift detected "
                    f"({len(drifted)} feature(s), top PSI={drifted[0]['psi']}: {drifted[0]['feature']}) "
                    f"— the previous model may now be stale against a shifted regime, not just noisy data."
                )

    # ── Kill switch: fast-failure stop (streak / rolling drawdown) ──
    if getattr(cfg, "ENABLE_KILL_SWITCH", False):
        ks = engine.KillSwitch(
            max_consecutive_losses=getattr(cfg, "KILL_SWITCH_MAX_CONSECUTIVE_LOSSES", 6),
            max_drawdown_pct=getattr(cfg, "KILL_SWITCH_MAX_DRAWDOWN_PCT", 3.0),
            lookback_hours=getattr(cfg, "KILL_SWITCH_LOOKBACK_HOURS", 24),
            fee_pct=getattr(cfg, "BRAIN_FEE_PCT", 0.0006),
            slippage_pct=getattr(cfg, "BRAIN_SLIPPAGE_PCT", 0.0003),
        )
        ks_state = ks.evaluate(real_rows)
        ai_metrics["kill_switch"] = ks_state
        if ks_state["tripped"]:
            ttl = int(getattr(cfg, "KILL_SWITCH_COOLDOWN_HOURS", 12) * 3600)
            if self.sdb._redis and not self.sdb.degraded:
                await self.sdb._safe_redis_op(                   
                    lambda: _rc(self.sdb._redis).set(KILL_SWITCH_KEY, json_dumps(ks_state), ex=ttl),
                    2.0, "kill_switch_set",
                )
            recommendations.append({
                "type": "kill_switch",
                "severity": "critical",
                "message": (
                    f"🛑 KILL SWITCH TRIPPED: {ks_state['reason']}. "
                    f"Dispatch blocked for {ttl // 3600}h or until cleared. "
                    f"CUSUM catches slow per-key decay; this catches the fast "
                    f"cross-key bleed it can't see."
                ),
            })

    # ── Fill reconciliation: assumed vs realized execution cost ──
    if getattr(cfg, "ENABLE_FILL_RECONCILIATION", False) and real_rows:
        fr = engine.fill_reconciliation(
            real_rows,
            assumed_fee_pct=getattr(cfg, "BRAIN_FEE_PCT", 0.0006),
            assumed_slippage_pct=getattr(cfg, "BRAIN_SLIPPAGE_PCT", 0.0003),
            rr_target=cfg.OUTCOME_RR_TARGET,
            stop_pct=cfg.OUTCOME_MAE_LOSS_PCT,
            min_sample=min_sample,
        )
        if fr.get("valid"):
            tier = "measured from fills" if fr.get("measured") else "ESTIMATED from outcome moves"
            worst = ", ".join(
                f"{p['pair']} {p['gap_bps']:+.1f}bps (n={p['n']})"
                for p in fr.get("per_pair", [])[:3]
            )
            gap = fr.get("gap_bps", 0)
            recommendations.append({
                "type": "fill_reconciliation",
                "severity": "medium" if gap > 1.0 else "low",
                "message": (
                    f"🧾 Execution cost ({tier}): realized slippage "
                    f"{fr['realized_slippage_per_side'] * 10000:.1f}bps/side vs assumed "
                    f"{getattr(cfg, 'BRAIN_SLIPPAGE_PCT', 0.0003) * 10000:.1f}bps/side "
                    f"(Δ{gap:+.1f}bps). "
                    + (f"EV overstated ~{fr.get('ev_overstated_pct_per_trade', 0):.4f}%/trade. " if gap > 0 else "")
                    + (f"Worst pairs: {worst}." if worst else "")
                ),
                "delta_ev": -fr.get("ev_overstated_pct_per_trade", 0.0),
            })

    severity_order = {"high": 0, "medium": 1, "low": 2}
    recommendations.sort(key=lambda x: severity_order.get(x["severity"], 3))

    return {
        "generated_at": int(time.time()),
        "real_sample_size": len(real_rows),
        "shadow_sample_size": len(shadow_rows),
        "overall_win_rate": round(sum(1 for r in real_rows if r["win"]) / len(real_rows), 4) if real_rows else None,
        "recommendation_count": len(recommendations),
        "recommendations": recommendations,
        "shadow_summary": shadow_summary,
        "config_patch": config_patch,
        "current_config": {
            "CONFLUENCE_MIN_ABS_SCORE": cfg.CONFLUENCE_MIN_ABS_SCORE,
            "CONFLUENCE_MIN_PCT": cfg.CONFLUENCE_MIN_PCT,
        },  
        "ai_metrics": {
            **ai_metrics,
            "brier_score": round(brier, 4) if has_calibration_data else None,
            "brier_status": brier_status,
            "net_ev": round(net_ev, 4) if net_ev is not None else None,
            "half_kelly": round(half_kelly, 4) if half_kelly is not None else None,
            "cusum_drifts": len(drift_alerts),
            "threshold_history": await self.sdb.load_threshold_history(),
            "ood_status": ood_status,
            "ev_objective": (
                engine.ev_first_objective(real_rows, min_sample=min_sample)
                if real_rows else None
            ),
            # ── NEW: rolling walk-forward ─
            "rolling_wf": (
                engine.rolling_walk_forward(real_rows, n_folds=5)
                if audit.can_run("walk_forward")[0]
                and len(real_rows) >= min_sample * 6
                else None
            ),
        },
    }
