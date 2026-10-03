"""Full (prescriptive) recommendation builder for BrainEngineV2 — moved verbatim out of BrainEngineV2._generate_recommendations_full; the method now delegates here. `self` is the engine instance."""
from __future__ import annotations
import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Tuple
from brain_audit import DataCoverage, HealthStatus, get_audit
from bot_config import cfg, CONFLUENCE_WEIGHTS, CONFIG_OVERRIDE_ALLOWED_FIELDS
from brain_helpers import _extract_p_value_for_fdr
import threshold_engine as engine
from threshold_engine import (
    optimize_vote_weights,
    conditional_performance,
    interaction_miner,
    simulate_config_change,
    regime_profile_optimizer,
    hash_config_state,
    learned_actionability,
    compare_config_versions,
)
from repair_ledger import (
    record_repair_issued,
    evaluate_pending_repairs,
    repair_success_rates,
    ledger_stats,
    load_ledger_entries,
)

async def build_full_recommendations(self) -> Dict[str, Any]:
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

        # ── Auto-rollback: clear dynamic weights if a weight repair hurt ──
        if getattr(cfg, "BRAIN_AUTO_ROLLBACK_HURT", True) and fresh:
            weight_categories = {
                "weight_optimizer",
                "dynamic_weights",
                "confluence_weights",
                "vote_weights",
            }
            hurt_weight_repairs = [
                e for e in fresh
                if e.get("verdict") == "hurt"
                and (
                    e.get("category") in weight_categories
                    or e.get("type") in weight_categories
                    or "weight" in str(e.get("type", "")).lower()
                    or "weight" in str(e.get("category", "")).lower()
                )
            ]
            if hurt_weight_repairs:
                cleared = await self.sdb.clear_dynamic_weights()
                ids = [
                    e.get("id") or e.get("repair_id")
                    for e in hurt_weight_repairs
                ]
                logger.warning(
                    f"↩️ Auto-rollback: cleared dynamic_weights after "
                    f"{len(hurt_weight_repairs)} hurt repair(s): {ids} "
                    f"(cleared={cleared})"
                )
                for e in hurt_weight_repairs:
                    e["auto_rolled_back"] = True
                    e["rolled_back_at"] = int(time.time())
                await self._record_plan_event(
                    f"repair:{ids[0]}" if ids and ids[0] else "repair:unknown",
                    "rolled_back",
                    f"dynamic_weights cleared after hurt repair(s) {ids}",
                )

        # ── Objective post-apply harm monitor (roadmap #6) ──
        _mon = await self.monitor_applied_plans(real_rows, logger)
        if _mon:
            ai_metrics["apply_monitor"] = _mon
            for _ev in _mon:
                if _ev["status"] == "rolled_back":
                    _e = _ev["evidence"]
                    recommendations.append({
                        "type": "auto_rollback", "severity": "high",
                        "message": (
                            f"↩️ Auto-rolled back plan {_ev['plan_id']}: WR "
                            f"{_e['wr_pre']:.0%}→{_e['wr_post']:.0%} after apply "
                            f"(p={_e['p_value']}, n={_e['n_post']}/{_e['n_pre']})."
                            + (" Config overrides need a restart to take effect."
                               if _ev.get("needs_restart") else "")
                        ),
                    })
                elif _ev["status"] == "rollback_withheld":
                    recommendations.append({
                        "type": "auto_rollback", "severity": "low",
                        "message": (
                            f"ℹ️ Plan {_ev['plan_id']} shows a post-apply WR drop but "
                            f"auto-rollback was withheld: {_ev['reason']}."
                        ),
                    })

        # ── Champion/challenger: periodic promotion check. force=False
        # always, so this stays a no-op reporting "shadow_only_requires_force"
        # while CHALLENGER_SHADOW_ONLY is True — promotion still requires
        # an explicit force=True call elsewhere, this just makes the
        # gate's decision visible every report cycle instead of the
        # challenger sitting unchecked in Redis indefinitely. ──
        if getattr(cfg, "ENABLE_CHAMPION_CHALLENGER", False):
            try:
                promo = await self.maybe_promote_challenger()
                if promo.get("promoted"):
                    logger.warning(
                        f"🏆 Challenger promoted to champion (live "
                        f"dynamic_weights): {promo.get('meta')}"
                    )
                    ai_metrics["challenger_promotion"] = promo
                elif promo.get("reason") != "no_challenger":
                    logger.info(f"🧪 Challenger not promoted: {promo.get('reason')}")
            except Exception as e:
                logger.warning(f"Challenger promotion check failed: {e}")

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

            # ── Shadow out-of-sample veto ─────────────────────
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

                if getattr(cfg, "ENABLE_CHAMPION_CHALLENGER", False):
                    # Store as challenger only — does not enter pending live plan
                    meta = {
                        "source": "weight_optimizer",
                        "n_oos": int(wopt.get("n_oos") or wopt.get("n_samples") or 0),
                        "net_ev": wopt.get("net_ev"),
                        "champion_net_ev": wopt.get("baseline_net_ev"),
                        "shadow_only": bool(getattr(cfg, "CHALLENGER_SHADOW_ONLY", True)),
                        "confidence": conf_score,
                    }
                    ok = await self.sdb.set_challenger_weights(
                        wopt["suggested_weights"], meta=meta,
                    )
                    recommendations.append({
                        "type": "weight_optimizer",
                        "severity": "medium",
                        "message": (
                            f"🧪 Challenger weights stored (shadow only), "
                            f"not applied live. ok={ok}, n={meta['n_oos']}, "
                            f"conf={conf_score:.0%}. {oos_weight_note}"
                        ),
                    })
                else:
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

    # ─ Per-alert breakdown ───────────────────────────────���──────────
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

    # ── Config Version Regression ───────────────────────────────────
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

    # ── Strategy degradation vs regime (roadmap #17) ─────────────────
    try:
        _state = engine.classify_strategy_state(real_rows)
        ai_metrics["strategy_state"] = _state
        _st = _state["state"]
        if _st == "STRATEGY_DEGRADED":
            recommendations.append({
                "type": "strategy_state", "severity": "high",
                "message": (
                    f"🚨 Strategy degraded: WR {_state['older_wr']:.0%}→{_state['recent_wr']:.0%} "
                    f"and still {_state['mix_adjusted_drop']:.0%} below what the regime mix "
                    f"predicts. {_state['action']}."
                ),
            })
        elif _st in ("REGIME_UNDERREPRESENTED", "REGIME_SHIFT", "DEGRADED_REGIME_UNKNOWN"):
            _label = {
                "REGIME_UNDERREPRESENTED": "current regime underrepresented in history",
                "REGIME_SHIFT": "regime mix shifted toward a historically weaker regime",
                "DEGRADED_REGIME_UNKNOWN": "drop cannot be attributed (thin ADX data)",
            }[_st]
            recommendations.append({
                "type": "strategy_state", "severity": "low",
                "message": (
                    f"ℹ️ WR {_state['older_wr']:.0%}→{_state['recent_wr']:.0%}: {_label}. "
                    f"{_state['action']}. Not evidence the strategy broke."
                ),
            })
    except Exception as e:
        logging.getLogger("macd_bot").debug(f"Strategy-state classification failed (non-fatal): {e}")

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

        # ── Actionable ablation loop (roadmap #10) ─────────────────
        try:
            ablation = engine.actionable_condition_ablation(
                real_rows,
                min_sample=min_sample,
                n_permutations=_perm_n,
                noise_threshold=getattr(cfg, "ABLATION_NOISE_THRESHOLD", 0.01),
                edge_threshold=getattr(cfg, "ABLATION_EDGE_THRESHOLD", 0.03),
            )
            noise_votes = [a for a in ablation if a["action"] == "reduce_weight"]
            if noise_votes:
                parts = [
                    f"{a['vote']}(imp={a['importance']:+.3f}→×{a['suggested_weight_factor']})"
                    for a in noise_votes[:5]
                ]
                adj_payload = []
                for a in noise_votes[:5]:
                    vote = a["vote"]
                    current_w = CONFLUENCE_WEIGHTS.get(vote)
                    if current_w is None or current_w <= 0:
                        continue
                    adj_payload.append({
                        "vote": vote,
                        "current": current_w,
                        "suggested": round(current_w * a["suggested_weight_factor"], 2),
                        "category": "condition_ablation",
                        "reason": a["reason"],
                    })
                recommendations.append({
                    "type": "condition_ablation",
                    "severity": "medium",
                    "message": (
                        "Conditions adding little information (candidate weight cuts): "
                        + ", ".join(parts)
                    ),
                    "ablation": noise_votes[:5],
                    "weight_adjustments": adj_payload,  # picked up by _store_pending_plan
                })
        except Exception as e:
            logging.getLogger("macd_bot").debug(
                f"Actionable ablation failed (non-fatal): {e}"
            )

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

    # ── Bonus-aware metrics ──���────────────────────────────�����──────────
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

    # ── Champion/challenger promotion check (gated) ─────────────────
    # Challenger weights accumulate via weight_optimizer / apply_pending_plan
    # when ENABLE_CHAMPION_CHALLENGER is on. Promotion is intentionally not
    # automatic while CHALLENGER_SHADOW_ONLY is True (safe-by-default).
    if getattr(cfg, "ENABLE_CHAMPION_CHALLENGER", False):
        try:
            promo = await self.maybe_promote_challenger(force=False)
            ai_metrics["challenger_promotion"] = promo
            if promo.get("promoted"):
                meta = promo.get("meta") or {}
                recommendations.append({
                    "type": "challenger_promotion",
                    "severity": "high",
                    "message": (
                        "🏆 Challenger promoted to champion "
                        f"(n_oos={meta.get('n_oos')}, "
                        f"net_ev={meta.get('net_ev')}, "
                        f"champion_net_ev={meta.get('champion_net_ev')})."
                    ),
                })
                result["recommendations"] = recommendations
                result["recommendation_count"] = len(recommendations)
                logger.info(
                    f"🏆 Challenger promoted to champion: {promo}"
                )
            else:
                logger.info(
                    f"🧪 Challenger promotion skipped: {promo.get('reason')}"
                )
        except Exception as e:
            audit.record_analysis_exception("challenger_promotion", e)
            logger.warning(f"Challenger promotion check failed: {e}")
        _phase_mark("challenger_promotion")

    return result
