"""brain_enhanced.py — Brain orchestration (BrainEngineV2): prescriptive phases, action plans, apply/rollback lifecycle.

Public entry point: brain_engine.BrainEngine.

Split out of the original module without logic changes; the original
module remains as a facade re-exporting every name."""
from __future__ import annotations
import asyncio
import logging
import random
import math
import time
from typing import Any, Dict, List, Optional
import os
from pathlib import Path
from brain_audit import DataCoverage, HealthStatus, get_audit, reset_audit, ACTION_GATE_MIN_ROWS
from archive_reader import load_archived_outcomes
from bot_config import cfg, CONFLUENCE_WEIGHTS, CONFIG_OVERRIDE_ALLOWED_FIELDS, json_dumps, json_loads
from state import RedisKeyPrefix, RedisStateStore
from brain import BrainCore, _extract_p_value_for_fdr
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
    mark_plan_applied,
    evaluate_pending_repairs,
    repair_success_rates,
    ledger_stats,
    load_ledger_entries,
)
from alerts import escape_markdown_v2
import brain_recommend_full as _full_recs
from brain_report import (
    APPLY_SNAPSHOT_KEY,
    APPLY_SNAPSHOT_MAX,
    CHALLENGER_STREAK_KEY,
    PLAN_HISTORY_KEY,
    PLAN_HISTORY_MAX,
    _PHASE_MIN_SAMPLES,
    _Piece,
    build_brain_report_sections,
    render_report_markdown,
    render_report_messages,
)

# Backward-compatible alias for the core class (was `from brain import BrainEngine as BaseBrainEngine`).
BaseBrainEngine = BrainCore


class BrainEngineV2(BrainCore):
    """The Brain's public engine (import it as brain_engine.BrainEngine).

    Extends BrainCore with prescriptive phases 1.5-6, actionability scoring,
    plan storage and the apply / monitor / rollback lifecycle.

    Explicit overlap with BrainCore — these four hooks are overridden here and
    the core versions are reached by name, never through super():
      _get_rows / _get_layered_window_rows   archive-first, Redis fallback
                                             (core: Redis only)
      generate_recommendations               120 s cached wrapper around the
                                             full pipeline; the core method
                                             is the baseline phase
      _deliver_report                        routes to generate_report()
                                             (core: legacy _generate_and_send)
    """

    def __init__(self, sdb: RedisStateStore):
        BrainCore.__init__(self, sdb)
        self._phase_samples = _PHASE_MIN_SAMPLES
        self._recs_cache: Optional[Dict[str, Any]] = None
        self._recs_cache_ts: float = 0.0

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
        gate: Dict[str, Any] = {
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
                # Kept alongside the boolean (not just pass/fail) so the
                # Confidence Breakdown can show a HIGH/MEDIUM/NOT READY
                # tier for DEPLOYMENT instead of collapsing it to one bit.
                gate["oos_p_ev_positive"] = rwc["p_ev_positive"]

        # Profitability: net EV must be positive with high confidence
        ev_obj = engine.ev_first_objective(real_rows, min_sample=min_sample)

        if ev_obj.get("valid"):
            gate["profitability"] = (
                ev_obj["p_ev_positive"] >= getattr(cfg, "BRAIN_EV_GATE_P_THRESHOLD", 0.85)
                and ev_obj["ev_p5"] > getattr(cfg, "BRAIN_EV_GATE_P5_FLOOR", -0.10)
            )
            # Same reasoning as oos_p_ev_positive above, for CHANGE.
            gate["profit_p_ev_positive"] = ev_obj["p_ev_positive"]
            gate["ev_p5"] = ev_obj["ev_p5"]

        # ── Stability: no active CUSUM edge-decay alarm. Prefer the
        if active_drift_keys:
            gate["stability"] = False
        elif recommendations is not None:
            gate["stability"] = not any(
                r.get("type") == "cusum_drift" for r in recommendations
            )
        # ── Risk: realized max drawdown must stay inside the same budget
        dd_budget = getattr(cfg, "KILL_SWITCH_MAX_DRAWDOWN_PCT", 3.0)
        lookback_h = int(getattr(cfg, "KILL_SWITCH_LOOKBACK_HOURS", 24))
        try:
            _ks_for_gate = engine.KillSwitch(
                max_consecutive_losses=getattr(
                    cfg, "KILL_SWITCH_MAX_CONSECUTIVE_LOSSES", 6
                ),
                max_drawdown_pct=dd_budget,
                lookback_hours=lookback_h,
                fee_pct=getattr(cfg, "BRAIN_FEE_PCT", 0.0006),
                slippage_pct=getattr(cfg, "BRAIN_SLIPPAGE_PCT", 0.0003),
            ).evaluate(real_rows)
            gate["risk"] = not _ks_for_gate["tripped"]
            # Expose the window + observed values so the report can render
            # "5.2% over 24h vs 3% budget" instead of an opaque PASS/FAIL.
            gate["risk_window_hours"] = lookback_h
            gate["risk_drawdown_pct"] = _ks_for_gate["drawdown_pct"]
            gate["risk_budget_pct"] = dd_budget
            gate["risk_consecutive_losses"] = _ks_for_gate["consecutive_losses"]
            gate["risk_reason"] = _ks_for_gate.get("reason")
        except Exception as _e:
            # A gate that crashes must not take dispatch down — fail-open,
            # matching the rest of the risk checks (KillSwitch itself is
            # called fail-open in macd_unified). Expose the error so a
            # silent failure is still visible in the report.
            logging.getLogger("macd_bot").warning(
                f"Risk gate evaluation failed, failing open: {_e}"
            )
            gate["risk"] = True
            gate["risk_window_hours"] = lookback_h
            gate["risk_drawdown_pct"] = None
            gate["risk_budget_pct"] = dd_budget
            gate["risk_reason"] = f"gate error (fail-open): {_e}"

        # ── Execution: cost assumptions must be non-trivial, or every EV
        # figure above is silently optimistic. ──
        fee_pct = getattr(cfg, "BRAIN_FEE_PCT", 0.0006)
        slip_pct = getattr(cfg, "BRAIN_SLIPPAGE_PCT", 0.0003)
        gate["execution"] = fee_pct > 0 and slip_pct > 0
 
        _GATE_KEYS = ("data_quality", "oos_prediction", "profitability",
                      "stability", "risk", "execution")
        gate["actionable"] = all(gate[k] for k in _GATE_KEYS)
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

        result = await BrainCore._get_layered_window_rows(self)
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
        return await _full_recs.build_full_recommendations(self)

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
        base = await BrainCore.generate_recommendations(self)
        base["_real_rows"] = real_rows
        base["_shadow_rows"] = shadow_rows
        return base

    async def _record_plan_event(
        self, plan_id: Optional[str], status: str, note: Optional[str] = None,
    ) -> None:
        """Append one lifecycle event (blocked/pending/superseded/applied/
        rolled_back) to the capped Brain plan audit trail. Best-effort —
        never raises into the report/apply path."""
        if not plan_id:
            return
        try:
            raw = await self.sdb.get_metadata(PLAN_HISTORY_KEY)
            hist = json_loads(raw) if raw else []
            if not isinstance(hist, list):
                hist = []
            hist.append({
                "plan_id": plan_id, "status": status,
                "ts": int(time.time()), "note": note,
            })
            await self.sdb.set_metadata(
                PLAN_HISTORY_KEY, json_dumps(hist[-PLAN_HISTORY_MAX:]),
                ttl=365 * 86400,
            )
        except Exception as e:
            logging.getLogger("macd_bot").debug(f"Plan history write failed (non-fatal): {e}")

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
                    if not action_gate_passed:
                        continue
                    adj = self._root_cause_to_weight_adjustment(rec)
                    if adj:
                        weight_adjustments.append(adj)

                elif rec_type == "condition_ablation":
                    # Roadmap #10 — votes that add almost no information.
                    # Same gate rule as root-cause weight cuts.
                    if not action_gate_passed:
                        continue
                    for adj in rec.get("weight_adjustments") or []:
                        if not isinstance(adj, dict):
                            continue
                        vote = adj.get("vote")
                        if not vote:
                            continue
                        weight_adjustments.append({
                            "vote": vote,
                            "current": adj.get("current"),
                            "suggested": adj.get("suggested"),
                            "category": adj.get("category") or "condition_ablation",
                            "reason": adj.get("reason") or rec.get("message", ""),
                        })

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

            if getattr(cfg, "ENABLE_BRAIN_PLAN_IDS", True):
                try:
                    counter_raw = await self.sdb.get_metadata("brain_plan_counter")
                    counter = int(counter_raw or 0) + 1
                    await self.sdb.set_metadata(
                        "brain_plan_counter", str(counter), ttl=365 * 86400
                    )
                    plan_id = f"PLAN-{counter:03d}"
                except Exception:
                    plan_id = f"PLAN-{int(time.time())}"

                data_window_days = getattr(cfg, "BRAIN_ANALYSIS_WINDOW_DAYS", None)
                param_diffs: List[Dict[str, Any]] = []
                for p in config_patches:
                    param_diffs.append({
                        "field": p.get("field") or p.get("path"),
                        "old": p.get("old_value") or p.get("old"),
                        "new": p.get("new_value") or p.get("new") or p.get("value"),
                        "reason": p.get("reason") or p.get("source"),
                    })
                for w in weight_adjustments:
                    param_diffs.append({
                        "field": f"weight:{w.get('vote')}",
                        "old": w.get("current") or w.get("old"),
                        "new": w.get("suggested") or w.get("new") or w.get("proposed"),
                        "reason": w.get("category") or "weight_adjustment",
                    })
                for ak in disable_alerts:
                    param_diffs.append({
                        "field": f"disable:{ak}",
                        "old": "enabled",
                        "new": "disabled",
                        "reason": "brain_disable",
                    })
                for ak in reinstate_alerts:
                    param_diffs.append({
                        "field": f"reinstate:{ak}",
                        "old": "disabled",
                        "new": "enabled",
                        "reason": "brain_reinstate",
                    })

                plan_data.update({
                    "plan_id": plan_id,
                    "model_version": recs.get("ai_metrics", {}).get("config_version")
                        or recs.get("config_version"),
                    "data_window_days": data_window_days,
                    "training_timestamp": plan_data["generated_at"],
                    "parameters_changed": param_diffs,
                    "sample_size": {
                        "real": recs.get("real_sample_size"),
                        "shadow": recs.get("shadow_sample_size"),
                    },
                    "oos_result": (recs.get("ai_metrics") or {}).get("action_gate"),
                    "calibration_result": (recs.get("ai_metrics") or {}).get("calibration_ece_mean"),
                    "status": "pending",
                    "evidence": {
                        "action_gate_passed": action_gate_passed,
                        "n_patches": len(config_patches),
                        "n_disable": len(disable_alerts),
                        "n_reinstate": len(reinstate_alerts),
                        "n_weights": len(weight_adjustments),
                    },
                })
                logging.getLogger("macd_bot").info(
                    f"Brain plan stored: {plan_id} "
                    f"(patches={len(config_patches)} disable={len(disable_alerts)} "
                    f"reinstate={len(reinstate_alerts)} weights={len(weight_adjustments)})"
                )
            if plan_data.get("plan_id"):
                try:
                    prev_raw = await self.sdb.get_metadata("brain_pending_plan")
                    prev = json_loads(prev_raw) if prev_raw else {}
                    prev_id = prev.get("plan_id") if isinstance(prev, dict) else None
                    if prev_id and prev_id != plan_data["plan_id"]:
                        await self._record_plan_event(
                            prev_id, "superseded", f"replaced by {plan_data['plan_id']}"
                        )
                except Exception as e:
                    logging.getLogger("macd_bot").debug(f"Previous plan lookup failed (non-fatal): {e}")

            await self.sdb.set_metadata(
                "brain_pending_plan",
                json_dumps(plan_data),
                ttl=7 * 86400
            )
            if plan_data.get("plan_id"):
                await self._record_plan_event(
                    str(plan_data["plan_id"]),
                    "pending" if action_gate_passed else "blocked",
                    "action gate " + ("passed" if action_gate_passed else "did not pass"),
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

    async def maybe_promote_challenger(self, force: bool = False) -> Dict[str, Any]:
        """Promote challenger → live champion only if gates pass (or force=True)."""
        blob = await self.sdb.get_challenger_weights()
        if not blob or not blob.get("weights"):
            return {"promoted": False, "reason": "no_challenger"}

        meta = blob.get("meta") or {}
        n_oos = int(meta.get("n_oos") or 0)
        min_n = int(getattr(cfg, "CHALLENGER_MIN_OOS_SAMPLE", 80))
        min_lift = float(getattr(cfg, "CHALLENGER_MIN_EV_LIFT", 0.02))
        ch_ev = meta.get("net_ev")
        base_ev = meta.get("champion_net_ev")

        if not force:
            stored_at = int(blob.get("stored_at") or 0)
            min_passes = int(getattr(cfg, "CHALLENGER_MIN_CONSECUTIVE_PASSES", 3))
            min_gap = int(getattr(cfg, "CHALLENGER_STREAK_MIN_GAP_SEC", 3600))
            now_ts = int(time.time())

            streak_state: Dict[str, Any] = {}
            try:
                raw_streak = await self.sdb.get_metadata(CHALLENGER_STREAK_KEY)
                parsed = json_loads(raw_streak) if raw_streak else {}
                if isinstance(parsed, dict):
                    streak_state = parsed
            except Exception as e:
                logging.getLogger("macd_bot").debug(
                    f"Challenger streak read failed (treating as 0): {e}"
                )
            # A different challenger (new stored_at) never inherits a streak.
            if int(streak_state.get("stored_at") or 0) != stored_at:
                streak_state = {"stored_at": stored_at, "streak": 0, "last_ts": 0}
            streak = int(streak_state.get("streak") or 0)

            fail: Optional[str] = None
            if n_oos < min_n:
                fail = f"n_oos={n_oos}<{min_n}"
            elif ch_ev is None or base_ev is None:
                fail = "missing_ev"
            elif float(ch_ev) <= 0.0:
                fail = f"net_ev={float(ch_ev):.4f}<=0"
            elif float(ch_ev) < float(base_ev) + min_lift:
                fail = f"ev_lift={float(ch_ev)-float(base_ev):.4f}<{min_lift}"

            if fail is not None:
                if streak > 0:
                    await self.sdb.set_metadata(
                        CHALLENGER_STREAK_KEY,
                        json_dumps({"stored_at": stored_at, "streak": 0, "last_ts": now_ts}),
                        ttl=30 * 86400,
                    )
                return {"promoted": False, "reason": fail, "streak_reset": streak > 0}

            # All gates passed this evaluation. Count it at most once per gap.
            if now_ts - int(streak_state.get("last_ts") or 0) >= min_gap:
                streak += 1
                await self.sdb.set_metadata(
                    CHALLENGER_STREAK_KEY,
                    json_dumps({"stored_at": stored_at, "streak": streak, "last_ts": now_ts}),
                    ttl=30 * 86400,
                )
            if streak < min_passes:
                return {
                    "promoted": False,
                    "reason": f"streak={streak}<{min_passes}",
                    "streak": streak,
                }
            if getattr(cfg, "CHALLENGER_SHADOW_ONLY", True) and not force:
                return {"promoted": False, "reason": "shadow_only_requires_force", "streak": streak}

        ok = await self.sdb.promote_challenger_to_champion()
        return {"promoted": bool(ok), "reason": "ok" if ok else "redis_failed", "meta": meta}

    async def _load_apply_snapshots(self) -> List[Dict[str, Any]]:
        try:
            raw = await self.sdb.get_metadata(APPLY_SNAPSHOT_KEY)
            data = json_loads(raw) if raw else []
            return data if isinstance(data, list) else []
        except Exception:
            return []

    async def _store_apply_snapshots(self, snaps: List[Dict[str, Any]]) -> bool:
        try:
            await self.sdb.set_metadata(
                APPLY_SNAPSHOT_KEY, json_dumps(snaps[-APPLY_SNAPSHOT_MAX:]),
                ttl=365 * 86400,
            )
            return True
        except Exception as e:
            logging.getLogger("macd_bot").debug(f"Apply-snapshot write failed (non-fatal): {e}")
            return False

    async def _save_apply_snapshot(
        self, plan_id: Optional[str],
        overrides: Dict[str, Any], disabled: Dict[str, Any],
        weights: Optional[Dict[str, Any]],
    ) -> None:
        """Remember how to undo an applied plan. Best-effort; never raises."""
        if not (overrides or disabled or weights):
            return
        snaps = await self._load_apply_snapshots()
        snaps.append({
            "plan_id": plan_id or f"plan-{int(time.time())}",
            "applied_at": int(time.time()),
            "status": "active",
            "overrides": overrides, "disabled": disabled, "weights": weights,
        })
        await self._store_apply_snapshots(snaps)

    @staticmethod
    def _weights_equal(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]]) -> bool:
        if a is None or b is None:
            return a is None and b is None
        keys = set(a) | set(b)
        return all(abs(float(a.get(k, 0.0)) - float(b.get(k, 0.0))) < 1e-9 for k in keys)

    async def _revert_snapshot(self, snap: Dict[str, Any]) -> Dict[str, Any]:
        """Restore what a plan changed, but ONLY where the live value is still
        exactly what that plan wrote. If a later plan (or a human) changed it
        since, leave it alone: reverting would clobber the newer decision."""
        out: Dict[str, Any] = {"reverted": [], "skipped": [], "failed": []}
        try:
            live_ov = await self.sdb.get_config_override()
            live_dis = await self.sdb.get_disabled_alert_keys()
            live_w = await self.sdb.get_dynamic_weights()
        except Exception as e:
            out["failed"].append(f"state read failed: {e}")
            return out

        for field, d in (snap.get("overrides") or {}).items():
            if live_ov.get(field) != d.get("new"):
                out["skipped"].append(f"{field} (changed since)")
                continue
            prev = d.get("prev")
            ok = (
                await self.sdb.remove_config_override_field(field) if prev is None
                else await self.sdb.write_config_override(field, prev)
            )
            (out["reverted"] if ok else out["failed"]).append(f"{field}→{prev}")

        for ak, d in (snap.get("disabled") or {}).items():
            if (ak in live_dis) != bool(d.get("new")):
                out["skipped"].append(f"{ak} (changed since)")
                continue
            ok = await self.sdb.set_alert_key_disabled(ak, bool(d.get("prev")))
            (out["reverted"] if ok else out["failed"]).append(
                f"{ak}→{'disabled' if d.get('prev') else 'enabled'}"
            )

        w = snap.get("weights")
        if w:
            if not self._weights_equal(live_w, w.get("new")):
                out["skipped"].append("dynamic_weights (changed since)")
            else:
                prev_w = w.get("prev")
                ok = (
                    await self.sdb.clear_dynamic_weights() if prev_w is None
                    else await self.sdb.set_dynamic_weights(prev_w, source="rollback")
                )
                (out["reverted"] if ok else out["failed"]).append("dynamic_weights")
        return out

    async def monitor_applied_plans(
        self, rows: List[Dict[str, Any]], logger_run: logging.Logger,
    ) -> List[Dict[str, Any]]:
        """Objective post-apply harm check with automatic revert (roadmap #6).

        For each active snapshot older than BRAIN_ROLLBACK_MIN_HOURS, compare
        outcomes AFTER the apply with an equally long window BEFORE it. Revert
        only when ALL hold:
          * >= BRAIN_ROLLBACK_MIN_N outcomes on both sides;
          * win rate fell by >= BRAIN_ROLLBACK_MIN_WR_DROP, one-sided
            two-proportion test p <= BRAIN_ROLLBACK_ALPHA;
          * net EV did not improve;
          * the drop is not attributable to a regime shift (classify_strategy_state
            says neither REGIME_SHIFT, REGIME_UNDERREPRESENTED nor
            DEGRADED_REGIME_UNKNOWN);
          * fewer than BRAIN_ROLLBACK_MAX_PER_DAY rollbacks in the last 24h.
        Plans that stay clean for BRAIN_ROLLBACK_MONITOR_DAYS are marked cleared.
        Returns one event dict per status change. Never raises.
        """
        events: List[Dict[str, Any]] = []
        if not getattr(cfg, "BRAIN_AUTO_ROLLBACK_HURT", True):
            return events
        try:
            snaps = await self._load_apply_snapshots()
            if not any(s.get("status") == "active" for s in snaps):
                return events
            now = int(time.time())
            min_h = float(cfg.BRAIN_ROLLBACK_MIN_HOURS)
            monitor_s = float(cfg.BRAIN_ROLLBACK_MONITOR_DAYS) * 86400
            min_n = int(cfg.BRAIN_ROLLBACK_MIN_N)
            min_drop = float(cfg.BRAIN_ROLLBACK_MIN_WR_DROP)
            alpha = float(cfg.BRAIN_ROLLBACK_ALPHA)
            max_per_day = int(cfg.BRAIN_ROLLBACK_MAX_PER_DAY)
            rolled_24h = sum(
                1 for s in snaps
                if s.get("status") == "rolled_back" and now - int(s.get("rolled_back_at") or 0) < 86400
            )
            changed = False

            for snap in snaps:
                if snap.get("status") != "active":
                    continue
                applied_at = int(snap.get("applied_at") or 0)
                age_s = now - applied_at
                if age_s < min_h * 3600:
                    continue
                post = [r for r in rows if r.get("entry_ts", 0) >= applied_at]
                pre = [r for r in rows if applied_at - age_s <= r.get("entry_ts", 0) < applied_at]
                pid = snap.get("plan_id")

                harmed = False
                evidence: Dict[str, Any] = {"n_post": len(post), "n_pre": len(pre)}
                if len(post) >= min_n and len(pre) >= min_n:
                    w_post = sum(1 for r in post if r["win"])
                    w_pre = sum(1 for r in pre if r["win"])
                    wr_post, wr_pre = w_post / len(post), w_pre / len(pre)
                    pooled = (w_post + w_pre) / (len(post) + len(pre))
                    se = math.sqrt(max(pooled * (1 - pooled), 1e-12) * (1 / len(post) + 1 / len(pre)))
                    drop = wr_pre - wr_post
                    p_val = 0.5 * math.erfc((drop / se) / math.sqrt(2)) if se > 0 else 1.0
                    ev_post = engine.ev_and_kelly_for(post)[0]
                    ev_pre = engine.ev_and_kelly_for(pre)[0]
                    evidence.update({
                        "wr_pre": round(wr_pre, 4), "wr_post": round(wr_post, 4),
                        "drop": round(drop, 4), "p_value": round(p_val, 4),
                        "ev_pre": round(ev_pre, 4), "ev_post": round(ev_post, 4),
                    })
                    harmed = (
                        drop >= min_drop and p_val <= alpha and ev_post <= ev_pre
                    )

                if harmed:
                    # Regime check aligned to THIS apply point: "recent" = everything
                    # since the apply, "older" = everything before it.
                    regime = engine.classify_strategy_state(
                        rows, recent_days=age_s / 86400.0,
                        min_recent=min_n, min_older=min_n,
                    )
                    evidence["regime_state"] = regime.get("state")
                    if regime.get("state") in (
                        "REGIME_SHIFT", "REGIME_UNDERREPRESENTED", "DEGRADED_REGIME_UNKNOWN",
                    ):
                        events.append({
                            "plan_id": pid, "status": "rollback_withheld",
                            "reason": f"drop attributed to regime ({regime.get('state')})",
                            "evidence": evidence,
                        })
                        continue
                    if rolled_24h >= max_per_day:
                        events.append({
                            "plan_id": pid, "status": "rollback_withheld",
                            "reason": f"daily rollback cap {max_per_day} reached",
                            "evidence": evidence,
                        })
                        continue
                    result = await self._revert_snapshot(snap)
                    snap["status"] = "rolled_back"
                    snap["rolled_back_at"] = now
                    snap["rollback_evidence"] = evidence
                    snap["rollback_result"] = result
                    rolled_24h += 1
                    changed = True
                    note = (
                        f"WR {evidence['wr_pre']:.0%}→{evidence['wr_post']:.0%} "
                        f"(p={evidence['p_value']}), reverted={result['reverted']}, "
                        f"skipped={result['skipped']}, failed={result['failed']}"
                    )
                    await self._record_plan_event(pid, "rolled_back", note[:300])
                    logger_run.warning(f"↩️ Auto-rollback of plan {pid}: {note}")
                    events.append({
                        "plan_id": pid, "status": "rolled_back",
                        "evidence": evidence, "result": result,
                        "needs_restart": bool(snap.get("overrides")),
                    })
                elif age_s >= monitor_s:
                    snap["status"] = "cleared"
                    snap["cleared_at"] = now
                    changed = True
                    await self._record_plan_event(pid, "monitor_cleared", "no significant post-apply harm")
                    events.append({"plan_id": pid, "status": "cleared", "evidence": evidence})

            if changed:
                await self._store_apply_snapshots(snaps)
        except Exception as e:
            logger_run.warning(f"Apply-monitor failed (non-fatal): {e}")
        return events

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

            # Rollback snapshot: what each touched setting was BEFORE this plan
            # and what the plan wrote, so a later harm check can restore it.
            snap_overrides: Dict[str, Any] = {}
            snap_disabled: Dict[str, Any] = {}
            snap_weights: Optional[Dict[str, Any]] = None
            try:
                _ov_before = await self.sdb.get_config_override()
                _dis_before = await self.sdb.get_disabled_alert_keys()
                _w_before = await self.sdb.get_dynamic_weights()
            except Exception as e:
                logger_run.debug(f"Rollback snapshot pre-read failed (non-fatal): {e}")
                _ov_before, _dis_before, _w_before = {}, set(), None

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
                        snap_overrides[field] = {"prev": _ov_before.get(field), "new": value}
                        applied.append(f"✅ {field}: {value}")
                        applied_config = True
                        logger_run.info(f"Applied brain config: {field} = {value}")

            if plan_gate_passed:
                for ak in plan.get("disable_alerts", []):
                    ok = await self.sdb.set_alert_key_disabled(ak, True)
                    if ok:
                        snap_disabled[ak] = {"prev": ak in _dis_before, "new": True}
                        applied.append(f"🔴 Disabled: {ak}")
                        applied_live = True
                        logger_run.info(f"Applied brain disable: {ak}")
                for ak in plan.get("reinstate_alerts", []):
                    ok = await self.sdb.set_alert_key_disabled(ak, False)
                    if ok:
                        snap_disabled[ak] = {"prev": ak in _dis_before, "new": False}
                        applied.append(f"🟢 Reinstated: {ak}")
                        applied_live = True
                        logger_run.info(f"Applied brain reinstate: {ak}")
            else:
                logger_run.warning(
                    "Skipping disable/reinstate entries — plan did not pass action gate"
                )

            # Apply root-cause weight adjustments via dynamic weights
            # (or store as challenger when ENABLE_CHAMPION_CHALLENGER is on)
            weight_adj = plan.get("weight_adjustments", [])
            if weight_adj and plan_gate_passed:
                weights = dict(CONFLUENCE_WEIGHTS)
                for adj in weight_adj:
                    vote = adj.get("vote")
                    suggested = adj.get("suggested")
                    if vote in weights and suggested is not None:
                        weights[vote] = suggested

                if (getattr(cfg, "ENABLE_CHAMPION_CHALLENGER", False)
                        or getattr(cfg, "ENFORCE_SINGLE_WEIGHT_PATH", True)):
                    # Shadow path: store as challenger only — does not affect live gates
                    meta = {
                        "source": "root_cause_weight_adj",
                        "n_oos": 0,
                        "net_ev": None,
                        "champion_net_ev": None,
                        "shadow_only": bool(getattr(cfg, "CHALLENGER_SHADOW_ONLY", True)),
                        "votes": [a.get("vote") for a in weight_adj],
                    }
                    _existing = await self.sdb.get_challenger_weights()
                    _existing_meta = (_existing or {}).get("meta") or {}
                    if _existing_meta.get("net_ev") is not None:
                        logger_run.info(
                            "🧪 Root-cause weights not stored as challenger — an "
                            "OOS-evidenced challenger is already pending "
                            f"(source={_existing_meta.get('source')})"
                        )
                        ok = False
                    else:
                        ok = await self.sdb.set_challenger_weights(weights, meta=meta)
                    if ok:
                        for adj in weight_adj:
                            applied.append(
                                f"🧪 Challenger ⚖️ {adj.get('vote')}: "
                                f"{adj.get('current')} → {adj.get('suggested')}"
                            )
                        logger_run.info(
                            "🧪 Challenger weights stored (shadow), not live: "
                            f"{[a.get('vote') for a in weight_adj]} ok={ok}"
                        )
                    # Do NOT set applied_live — live CONFLUENCE_WEIGHTS unchanged
                else:
                    # Legacy path: write live dynamic weights (existing behavior) 
                    if await self.sdb.set_dynamic_weights(weights):
                        snap_weights = {"prev": _w_before, "new": dict(weights)}
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
                        logger_run.info(f"���� Repair ledger: marked {marked} repair(s) applied")
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
                await self._record_plan_event(plan.get("plan_id"), "applied", "; ".join(applied)[:200])
                await self._save_apply_snapshot(
                    plan.get("plan_id"), snap_overrides, snap_disabled, snap_weights,
                )
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

# Names moved to other modules, re-exported for backward compatibility.
from brain_report import (
    _report_section_failed,
    _RULE,
    _IST,
    _LADDER,
    _LADDER_NAMES,
    _MSG_LIMIT,
    _HUMAN_SECTIONS,
    _alert_family,
    _p,
    _c,
    _c_split,
    _hdr,
    _evidence_rank,
    _wrap_names,
    _collect_facts,
    _outcome_anatomy,
    _goal_wr,
    _profit_status,
    _data_status,
    _recording_status,
    _conf_status,
    _icon,
    _overall,
    _fmt_days,
    _fmt_span,
    _WIDE_EMOJI_RANGES,
    _vwidth,
    _ljust,
    _rjust,
    _table,
    _split_leading_emoji,
    _kv_table,
    _active_blockers,
    _sec_summary,
    _sec_verdict,
    _cf_verdict,
    _sec_do_now,
    _sec_profit,
    _EVIDENCE_LEGEND,
    _alert_table,
    _sec_loss,
    _sec_positive,
    _sec_scorecard,
    _sec_sessions,
    _confidence_tier,
    _confidence_breakdown,
    _sec_gate,
    _sec_reasoning_chain,
    _REPORT_SECTIONS,
    _md_prose,
    build_brain_report,
)

__all__ = [
    "DataCoverage",
    "HealthStatus",
    "get_audit",
    "reset_audit",
    "ACTION_GATE_MIN_ROWS",
    "load_archived_outcomes",
    "cfg",
    "CONFLUENCE_WEIGHTS",
    "CONFIG_OVERRIDE_ALLOWED_FIELDS",
    "json_dumps",
    "json_loads",
    "RedisKeyPrefix",
    "RedisStateStore",
    "BaseBrainEngine",
    "_extract_p_value_for_fdr",
    "engine",
    "optimize_vote_weights",
    "conditional_performance",
    "interaction_miner",
    "simulate_config_change",
    "regime_profile_optimizer",
    "hash_config_state",
    "learned_actionability",
    "compare_config_versions",
    "record_repair_issued",
    "mark_plan_applied",
    "evaluate_pending_repairs",
    "repair_success_rates",
    "ledger_stats",
    "load_ledger_entries",
    "escape_markdown_v2",
    "_PHASE_MIN_SAMPLES",
    "_report_section_failed",
    "_RULE",
    "PLAN_HISTORY_KEY",
    "PLAN_HISTORY_MAX",
    "APPLY_SNAPSHOT_KEY",
    "APPLY_SNAPSHOT_MAX",
    "CHALLENGER_STREAK_KEY",
    "_IST",
    "_LADDER",
    "_LADDER_NAMES",
    "_MSG_LIMIT",
    "_HUMAN_SECTIONS",
    "_alert_family",
    "_Piece",
    "_p",
    "_c",
    "_c_split",
    "_hdr",
    "_evidence_rank",
    "_wrap_names",
    "_collect_facts",
    "_outcome_anatomy",
    "_goal_wr",
    "_profit_status",
    "_data_status",
    "_recording_status",
    "_conf_status",
    "_icon",
    "_overall",
    "_fmt_days",
    "_fmt_span",
    "_WIDE_EMOJI_RANGES",
    "_vwidth",
    "_ljust",
    "_rjust",
    "_table",
    "_split_leading_emoji",
    "_kv_table",
    "_active_blockers",
    "_sec_summary",
    "_sec_verdict",
    "_cf_verdict",
    "_sec_do_now",
    "_sec_profit",
    "_EVIDENCE_LEGEND",
    "_alert_table",
    "_sec_loss",
    "_sec_positive",
    "_sec_scorecard",
    "_sec_sessions",
    "_confidence_tier",
    "_confidence_breakdown",
    "_sec_gate",
    "_sec_reasoning_chain",
    "_REPORT_SECTIONS",
    "build_brain_report_sections",
    "render_report_messages",
    "_md_prose",
    "render_report_markdown",
    "build_brain_report",
    "BrainEngineV2",
]
