"""brain.py — Brain core engine (persistence, calibration, trade-quality verdict, kill switch, baseline analytics).

See brain_engine.py for the single public entry point.

Split out of the original module without logic changes; the original
module remains as a facade re-exporting every name."""
from __future__ import annotations
import asyncio
import json
import logging
import time
from collections import defaultdict, Counter
from typing import Any, Dict, List, Optional, Tuple
from alerts import escape_markdown_v2

from bot_config import cfg, json_dumps, json_loads, format_ist_time, CONFLUENCE_WEIGHTS
from state import RedisKeyPrefix, RedisStateStore, _rc
from plan_replay import coerce_path_fields
from playbook import KEY_CURRENT as PLAYBOOK_KEY, playbook_lookup
import threshold_engine as engine
from threshold_engine import CUSUMDetector, StabilityGate
from brain_audit import get_audit, HealthStatus
import brain_recommend_baseline as _baseline
from brain_helpers import (
    CALIBRATION_CURVES_KEY,
    KILL_SWITCH_KEY,
    MARKET_STATE_MODEL_KEY,
    ML_CALIBRATION_KEY,
    QUALITY_INPUTS_KEY,
    _OVERRIDE_COOLDOWN_PREFIX,
    _hget_int,
    _resolve_config_path,
    _to_opt_float,
)
class BrainCore:
    """Brain core — the persistence + verdict half of the engine.

    Responsibilities (all stay here, none are duplicated elsewhere):
      * calibration / market-state / quality-input persistence and loading
      * get_trade_quality()  — the live verdict path
      * kill switch, outcome-stream reading, row parsing, CUSUM drift
      * baseline recommendations (delegates to brain_recommend_baseline)
      * the legacy report path (_generate_and_send)

    Orchestration (prescriptive plans, apply/rollback, the Brain report) lives
    in brain_enhanced.BrainEngineV2, which extends this class. Callers should
    not pick between the two: use brain_engine.BrainEngine. The names
    BrainEngine / BaseBrainEngine below are kept only as backward-compatible
    aliases of this class.
    """

    def __init__(self, sdb: RedisStateStore):
        self.sdb = sdb
        self.stability_gate = StabilityGate(
            min_history=getattr(cfg, "BRAIN_STABILITY_MIN_HISTORY", 3),
            max_jump=getattr(cfg, "BRAIN_STABILITY_MAX_JUMP", 2.0),
        )
        self._cusum_detectors: Dict[str, CUSUMDetector] = {}
        self._calib_cache: Optional[Dict[str, Any]] = None
        self._calib_cache_ts = 0.0
        self._quality_cache: Optional[Dict[str, Any]] = None
        self._quality_cache_ts = 0.0
        self._market_model_cache: Optional[Dict[str, Any]] = None
        self._market_model_cache_ts = 0.0
        self._cached_real_raw: Optional[List[Dict[str, str]]] = None
        self._ml_calib_cache: Optional[Dict[str, Any]] = None
        self._ml_calib_cache_ts = 0.0
        self._playbook_cache: Optional[Dict[str, Any]] = None
        self._playbook_cache_ts = 0.0

    async def check_rewardable_override(
        self,
        alert_key: str,
        confluence_score: Optional[float],
        confluence_total: Optional[float],
    ) -> Optional[str]:
        """Returns a short reason string if a win-rate-rejected alert should be
        let through anyway, else None. Conservative by design: requires both a
        high confluence score on THIS alert and a proven shadow win rate for
        that bucket across all pairs. Rate-limited per alert_key so a volatile
        market can't produce a flood of overrides for the same alert type."""
        if confluence_score is None or confluence_total is None or confluence_total <= 0:
            return None
        conf_pct = (confluence_score / confluence_total) * 100.0
        if conf_pct < cfg.BRAIN_REWARDABLE_MIN_CONFLUENCE_PCT:
            return None

        if self.sdb.degraded or not self.sdb._redis:
            return None

        hiconf_key = f"{RedisKeyPrefix.SHADOW_HICONF_STATS}{alert_key}"
        try:
            data = await self.sdb._safe_redis_op(
                lambda: _rc(self.sdb._redis).hgetall(hiconf_key), 2.0, f"brain_hiconf:{alert_key}",
            )
        except Exception:
            return None
        if not data:
            return None

        wins = _hget_int(data, "wins")
        losses = _hget_int(data, "losses")
        total = wins + losses
        if total < cfg.BRAIN_REWARDABLE_MIN_SHADOW_SAMPLE:
            return None

        wr = wins / total
        if wr < cfg.BRAIN_REWARDABLE_MIN_SHADOW_WR:
            return None

        cooldown_seconds = getattr(cfg, "BRAIN_OVERRIDE_COOLDOWN_SECONDS", 4 * 3600)
        cooldown_key = f"{_OVERRIDE_COOLDOWN_PREFIX}{alert_key}"
        try:
            acquired = await self.sdb._safe_redis_op(
                lambda: _rc(self.sdb._redis).set(cooldown_key, "1", nx=True, ex=cooldown_seconds),
                2.0, f"brain_override_cooldown:{alert_key}",
            )
        except Exception:
            acquired = None
        if not acquired:
            return None

        return f"{conf_pct:.0f}% confluence, shadow WR {wr:.0%} over {total} tracked rejections"

    # ── Calibration live gate ────────────────────────────────────────────

    async def _persist_calibration_curves(self, calib: Dict[str, Any]) -> bool:
        """Returns True only if the write to Redis actually succeeded, so the
        caller can report calibration_persistence accurately instead of
        assuming success just because the curves were computed."""
        if self.sdb.degraded or not self.sdb._redis:
            logging.getLogger("macd_bot").warning(
                "Calibration curve persistence skipped: Redis unavailable or state degraded."
            )
            return False

        try:
            result = await self.sdb._safe_redis_op(
                lambda: _rc(self.sdb._redis).set(
                    CALIBRATION_CURVES_KEY,
                    json_dumps(calib),
                    ex=int(
                        getattr(
                            cfg,
                            "BRAIN_ANALYSIS_WINDOW_DAYS",
                            30,
                        ) * 86400
                    ),
                ),
                2.0,
                "calibration_persist",
            )

            if result is None:
                logging.getLogger("macd_bot").warning(
                    "Calibration curve persistence returned None — "
                    "Redis write may not have completed."
                )
                return False
            return True

        except Exception as e:
            logging.getLogger("macd_bot").error(
                f"Calibration curve persistence FAILED: {e}"
            )
            return False

    @staticmethod
    def _next_stream_id(stream_id: str) -> str:
        """Smallest stream ID strictly greater than ``stream_id``."""
        ms, _, seq = str(stream_id).partition("-")
        return f"{int(ms)}-{int(seq or 0) + 1}"

    async def _stream_tips(self) -> Dict[str, str]:
        """Newest entry ID of the real and shadow outcome streams ("0-0" when
        empty). Returns {} if Redis can't be read, so callers skip stamping."""
        tips: Dict[str, str] = {}
        for kind, key in (
            ("real", RedisKeyPrefix.OUTCOME_LOG_STREAM),
            ("shadow", RedisKeyPrefix.SHADOW_LOG_STREAM),
        ):
            try:
                entries = await self.sdb._safe_redis_op(
                    lambda: _rc(self.sdb._redis).xrevrange(key, count=1),
                    3.0, f"calibration_stream_tip:{kind}",
                )
            except Exception:
                return {}
            tips[kind] = str(entries[0][0]) if entries else "0-0"
        return tips

    async def _fold_new_outcomes_into_calibration(
        self, calib: Dict[str, Any], logger_run: logging.Logger,
    ) -> None:
        """Fold outcomes appended to the streams since the last fold/rebuild
        into the existing buckets, then re-persist.

        Per-stream ID cursors live INSIDE the curve payload, so cursor and
        counts are written by one SET. If the persist fails the same rows
        are simply folded again next run (no double counting).
        """
        cursors = dict(calib.get("stream_cursors") or {})
        if "real" not in cursors or "shadow" not in cursors:
            # Payload predates cursors (e.g. written by a Brain report):
            # start folding from "now" instead of guessing what is in it.
            tips = await self._stream_tips()
            if tips:
                calib["stream_cursors"] = tips
                await self._persist_calibration_curves(calib)
                logger_run.debug("Online calibration: cursors initialised")
            return

        cap = int(getattr(cfg, "CALIBRATION_INCREMENTAL_MAX_ROWS", 80))
        min_sample = int(getattr(cfg, "CALIBRATION_MIN_SAMPLE", 15))
        folded_total = 0
        advanced = False
        for kind, key in (
            ("real", RedisKeyPrefix.OUTCOME_LOG_STREAM),
            ("shadow", RedisKeyPrefix.SHADOW_LOG_STREAM),
        ):
            start = self._next_stream_id(cursors[kind])
            entries = await self.sdb._safe_redis_op(
                lambda: _rc(self.sdb._redis).xrange(
                    key, min=start, max="+", count=cap,
                ),
                5.0, f"calibration_fold_read:{kind}",
            )
            if not entries:
                continue
            rows = self._parse_rows([fields for _id, fields in entries])
            if kind == "shadow":
                # Same selection-leakage exclusion as build_calibration_curves.
                rows = [r for r in rows if r.get("rejection_reason") != "calibration_gate"]
            folded_total += engine.fold_outcomes_into_calibration(
                calib, rows, min_sample=min_sample,
            )
            cursors[kind] = str(entries[-1][0])
            advanced = True

        if not advanced:
            return
        calib["stream_cursors"] = cursors
        calib["online_updates"] = int(calib.get("online_updates") or 0) + folded_total
        calib["online_folded_at"] = int(time.time())
        ok = await self._persist_calibration_curves(calib)
        logger_run.info(
            f"🎯 Online calibration: folded {folded_total} new outcome(s) "
            f"(persisted={ok}, ECE={calib.get('ece_mean')})"
        )

    async def maybe_refresh_calibration(self, logger_run: logging.Logger) -> None:
        """Keep the live calibration curve fresh, independent of the Brain report.

        1. Curve exists and is younger than CALIBRATION_REFRESH_MAX_AGE_HOURS:
           fold only the outcomes appended since the last run into the
           existing buckets (no stream scan, history kept).
        2. Curve missing or older than that: full rebuild from the Redis
           outcome streams and stamp fresh stream cursors.
        """
        if not getattr(cfg, "ENABLE_CALIBRATION_GATE", False):
            return
        if not getattr(cfg, "ENABLE_BRAIN", True):
            return
        if getattr(cfg, "DRY_RUN_MODE", False):
            return
        if self.sdb.degraded or not self.sdb._redis:
            logger_run.debug("Calibration refresh skipped: Redis unavailable or degraded")
            return

        max_age_hr = float(getattr(cfg, "CALIBRATION_REFRESH_MAX_AGE_HOURS", 2.0))

        # ── Fresh curve: online fold only ──
        try:
            raw = await self.sdb._safe_redis_op(
                lambda: _rc(self.sdb._redis).get(CALIBRATION_CURVES_KEY),
                2.0, "calibration_age_check",
            )
            if raw:
                existing = json_loads(raw)
                built_at = existing.get("built_at")
                if built_at is not None and (
                    existing.get("curves")
                    or existing.get("status") == "INSUFFICIENT_SAMPLES"
                ):
                    age_hr = (time.time() - float(built_at)) / 3600.0
                    if age_hr < max_age_hr:
                        if getattr(cfg, "ENABLE_ONLINE_CALIBRATION", True):
                            try:
                                await self._fold_new_outcomes_into_calibration(
                                    existing, logger_run,
                                )
                            except Exception as fold_exc:
                                logger_run.warning(
                                    f"Online calibration fold failed "
                                    f"(curve unchanged this run): {fold_exc}"
                                )
                        else:
                            logger_run.debug(
                                f"Calibration curves fresh ({age_hr:.1f}h) — skip refresh"
                            )
                        return
        except Exception as e:
            logger_run.warning(
                f"Calibration age check failed (will attempt rebuild): {e}"
            )

        # ── Missing / stale: full rebuild ──
        try:
            logger_run.info(
                f"🎯 Calibration refresh: curves missing or older than "
                f"{max_age_hr}h — rebuilding..."
            )
            # Tips are read BEFORE the rows: a row appended in between is in
            # the rebuild AND folded once next run (harmless), whereas reading
            # tips afterwards could silently skip rows.
            stream_tips = await self._stream_tips()
            real_rows, shadow_rows = await self._get_rows()
            n_real, n_shadow = len(real_rows), len(shadow_rows)
            logger_run.info(
                f"🎯 Calibration refresh input: {n_real} real, {n_shadow} shadow rows"
            )

            calib = engine.build_calibration_curves(
                real_rows,
                bucket_pct=getattr(cfg, "CALIBRATION_BUCKET_PCT", 5.0),
                min_sample=getattr(cfg, "CALIBRATION_MIN_SAMPLE", 15),
                shadow_rows=shadow_rows,
            )
            if not calib.get("curves"):
                ak_counts = Counter(r.get("alert_key") for r in real_rows)
                min_s = getattr(cfg, "CALIBRATION_MIN_SAMPLE", 15)

                logger_run.warning(
                    f"Calibration refresh: no curves built "
                    f"(need ≥{min_s} samples/alert_key). "
                    f"real={n_real} shadow={n_shadow} | "
                    f"top alert_keys: {ak_counts.most_common(5)}"
                )
                if stream_tips:
                    calib["stream_cursors"] = stream_tips

                calib["status"] = "INSUFFICIENT_SAMPLES"
                calib["reason"] = (
                    f"No alert_key has at least {min_s} samples"
                )

                ok = await self._persist_calibration_curves(calib)

                if ok:
                    logger_run.info(
                        f"🎯 Calibration gate disabled for this run: "
                        f"insufficient samples for any alert_key "
                        f"(minimum={min_s}, real={n_real}, shadow={n_shadow}). "
                        f"Existing stale calibration was replaced with "
                        f"an explicit empty calibration payload."
                    )
                else:
                    logger_run.error(
                        "❌ Could not persist empty calibration payload; "
                        "previous calibration may remain live."
                    )

                return

            if stream_tips:
                calib["stream_cursors"] = stream_tips
            ok = await self._persist_calibration_curves(calib)
            n_keys = len(calib["curves"])
            ece = calib.get("ece_mean")
            if ok:
                logger_run.info(
                    f"✅ Calibration curves rebuilt & persisted "
                    f"({n_keys} alert_key(s), mean-per-alert ECE={ece})"
                )
            else:
                logger_run.error(
                    f"❌ Calibration curves built but NOT persisted "
                    f"({n_keys} alert_key(s), ECE={ece}) — "
                    f"previous curve (if any) remains live"
                )
        except Exception as e:
            logger_run.warning(
                f"Calibration refresh failed "
                f"(gate continues on previous curve if present): {e}"
            )
    

    async def _load_calibration_curve(self, alert_key: str) -> Optional[Dict[str, Any]]:
        now = time.time()
        if self._calib_cache is None or now - self._calib_cache_ts > 300:
            if self.sdb.degraded or not self.sdb._redis:
                return None
            raw = await self.sdb._safe_redis_op(
                lambda: _rc(self.sdb._redis).get(CALIBRATION_CURVES_KEY),
                2.0, "calibration_load",
            )
            if not raw:
                return None
            try:
                self._calib_cache = json.loads(raw).get("curves", {})
            except Exception:
                return None
            self._calib_cache_ts = now
        return self._calib_cache.get(alert_key) if self._calib_cache else None

    # ── Market-state model (report-only trained fit, item #12) ────────
    async def _persist_market_state_model(self, model: Dict[str, Any]) -> None:
        if self.sdb.degraded or not self.sdb._redis:
            logging.getLogger("macd_bot").warning(
                "Market-state model persistence skipped: Redis unavailable or state degraded."
            )
            return
        try:
            result = await self.sdb._safe_redis_op(
                lambda: _rc(self.sdb._redis).set(
                    MARKET_STATE_MODEL_KEY,
                    json_dumps(model),
                    ex=int(getattr(cfg, "BRAIN_ANALYSIS_WINDOW_DAYS", 30) * 86400),
                ),
                2.0,
                "market_state_model_persist",
            )
            if result is None:
                logging.getLogger("macd_bot").warning(
                    "Market-state model persistence returned None — "
                    "Redis write may not have completed; previous model (if any) stays live."
                )
        except Exception as e:
            logging.getLogger("macd_bot").error(
                f"Market-state model persistence FAILED: {e} — previous model stays live."
            )

    async def _load_market_state_model(self) -> Optional[Dict[str, Any]]:
        now = time.time()
        if self._market_model_cache is None or now - self._market_model_cache_ts > 300:
            if self.sdb.degraded or not self.sdb._redis:
                return None
            raw = await self.sdb._safe_redis_op(
                lambda: _rc(self.sdb._redis).get(MARKET_STATE_MODEL_KEY),
                2.0, "market_state_model_load",
            )
            if not raw:
                return None
            try:
                self._market_model_cache = json.loads(raw)
            except Exception:
                return None
            self._market_model_cache_ts = now
        return self._market_model_cache

    # ── ML calibration curve (bins on model p_win, not conf_pct) ─────
    async def _persist_ml_calibration_curve(self, curve: Dict[str, Any]) -> None:
        if self.sdb.degraded or not self.sdb._redis:
            logging.getLogger("macd_bot").warning(
                "ML calibration curve persistence skipped: Redis unavailable or state degraded."
            )
            return
        try:
            result = await self.sdb._safe_redis_op(
                lambda: _rc(self.sdb._redis).set(
                    ML_CALIBRATION_KEY,
                    json_dumps(curve),
                    ex=int(getattr(cfg, "BRAIN_ANALYSIS_WINDOW_DAYS", 30) * 86400),
                ),
                2.0,
                "ml_calibration_persist",
            )
            if result is None:
                logging.getLogger("macd_bot").warning(
                    "ML calibration curve persistence returned None — "
                    "Redis write may not have completed; previous curve (if any) stays live."
                )
        except Exception as e:
            logging.getLogger("macd_bot").error(
                f"ML calibration curve persistence FAILED: {e} — previous curve stays live."
            )

    async def _load_ml_calibration_curve(self) -> Optional[Dict[str, Any]]:
        now = time.time()
        if self._ml_calib_cache is None or now - self._ml_calib_cache_ts > 300:
            if self.sdb.degraded or not self.sdb._redis:
                return None
            raw = await self.sdb._safe_redis_op(
                lambda: _rc(self.sdb._redis).get(ML_CALIBRATION_KEY),
                2.0, "ml_calibration_load",
            )
            if not raw:
                return None
            try:
                self._ml_calib_cache = json.loads(raw)
            except Exception:
                return None
            self._ml_calib_cache_ts = now
        return self._ml_calib_cache

    async def check_calibration_gate(
        self, alert_key: str, conf_pct: float,
    ) -> Tuple[bool, Optional[float]]:
        """Dispatch hook: (pass, calibrated_wr). Fail-open on any
        missing/thin data — this gate blocks on evidence of
        miscalibration, never on its own absence."""
        if not getattr(cfg, "ENABLE_CALIBRATION_GATE", False):
            return True, None
        curve = await self._load_calibration_curve(alert_key)
        if not curve:
            return True, None
        ok, cal_wr, _reason = engine.calibration_gate_decision(
            curve, conf_pct, cfg.MIN_WIN_RATE,
            min_sample=getattr(cfg, "CALIBRATION_MIN_SAMPLE", 15),
            slack=getattr(cfg, "CALIBRATION_SLACK", 0.05),
        )
        return ok, cal_wr

    # ── Trade quality score (report-only) ───────────────────────────────
    async def _persist_quality_inputs(self, quality_inputs: Dict[str, Any]) -> None:
        if self.sdb.degraded or not self.sdb._redis:
            logging.getLogger("macd_bot").warning(
                "Quality-inputs persistence skipped: Redis unavailable or state degraded."
            )
            return
        try:
            result = await self.sdb._safe_redis_op(
                lambda: _rc(self.sdb._redis).set(
                    QUALITY_INPUTS_KEY,
                    json_dumps(quality_inputs),
                    ex=int(getattr(cfg, "BRAIN_ANALYSIS_WINDOW_DAYS", 30) * 86400),
                ),
                2.0,
                "quality_inputs_persist",
            )
            if result is None:
                logging.getLogger("macd_bot").warning(
                    "Quality-inputs persistence returned None — "
                    "Redis write may not have completed; previous bundle stays live."
                )
        except Exception as e:
            logging.getLogger("macd_bot").error(
                f"Quality-inputs persistence FAILED: {e} — previous bundle stays live."
            )

    async def _load_quality_inputs(self) -> Optional[Dict[str, Any]]:
        now = time.time()
        if self._quality_cache is None or now - self._quality_cache_ts > 300:
            if self.sdb.degraded or not self.sdb._redis:
                return None
            raw = await self.sdb._safe_redis_op(
                lambda: _rc(self.sdb._redis).get(QUALITY_INPUTS_KEY),
                2.0, "quality_inputs_load",
            )
            if not raw:
                return None
            try:
                self._quality_cache = json.loads(raw)
            except Exception:
                return None
            self._quality_cache_ts = now
        return self._quality_cache

    async def get_playbook_entry(self, alert_key: str, direction: str) -> Optional[Dict[str, Any]]:
        """Learner playbook entry for an alert (None = none / stale / Redis down).
        Never raises; the playbook is cached for 5 minutes."""
        if getattr(cfg, "PLAYBOOK_MODE", "off") == "off":
            return None
        try:
            now = time.time()
            if self._playbook_cache is None or now - self._playbook_cache_ts > 300:
                if self.sdb.degraded or not self.sdb._redis:
                    return None
                raw = await self.sdb.get_metadata(PLAYBOOK_KEY)
                self._playbook_cache_ts = now
                self._playbook_cache = json.loads(raw) if raw else {}
            from alert_registry import alert_family_of
            return playbook_lookup(
                self._playbook_cache, alert_key, direction, family_fn=alert_family_of,
                now_ts=now, max_age_sec=float(getattr(cfg, "PLAYBOOK_MAX_AGE_HOURS", 36)) * 3600.0,
            )
        except Exception as e:
            logging.getLogger("macd_bot").debug(f"playbook lookup failed: {e}")
            return None

    async def get_trade_quality(
        self,
        pair: str,
        alert_key: str,
        direction: str,
        conf_pct: float,
        adx_val: Optional[float] = None,
        live_context: Optional[Dict[str, Any]] = None,
        votes: Optional[Dict[str, bool]] = None,
        session: str = "unknown",
    ) -> Optional[Dict[str, Any]]:
        """Dispatch hook: report-only trade-quality verdict for one
        prospective alert, from the per-alert EV/regime evidence the last
        brain report persisted, optionally blended with a live per-trade
        market-state prediction (item #12) once
        ENABLE_MARKET_STATE_LIVE_SCORE is on. Returns None (never blocks
        dispatch) on any missing/thin data."""
        bundle = await self._load_quality_inputs()
        if not bundle:
            return None
        ev_model_result = bundle.get("ev_by_alert", {}).get(alert_key)
        if not ev_model_result:
            return None
        calibration_curve = await self._load_calibration_curve(alert_key)
        context = dict(live_context or {})
        context.setdefault("adx_val", adx_val)

        # ── Bayesian-shrunk pair/regime edge (roadmap item: hierarchical
        leaves = bundle.get("hierarchical_leaves") or {}
        median_adx = bundle.get("hierarchical_median_adx")
        leaf_adx = context.get("adx_val")
        if median_adx is not None and leaf_adx is not None:
            regime = "trending" if leaf_adx >= median_adx else "ranging"
        else:
            regime = "unknown"

        ev_model_result = dict(ev_model_result)
        leaf_key = f"{pair}|{alert_key}|{direction}|{regime}"
        leaf = leaves.get(leaf_key)

        # Hierarchical fallback chain when exact leaf is thin/missing:
        # pair+alert+dir+regime → any same pair+alert+dir → alert baseline
        if not leaf:
            # same pair + alert + direction, any regime
            prefix = f"{pair}|{alert_key}|{direction}|"
            candidates = [
                v for k, v in leaves.items()
                if k.startswith(prefix) and v.get("n", 0) > 0
            ]
            if candidates:
                # prefer largest n
                leaf = max(candidates, key=lambda x: x.get("n", 0))
                leaf_key = f"{pair}|{alert_key}|{direction}|*"

        if leaf:
            ev_model_result["net_ev"] = leaf["shrunk_net_ev"]
            # optional: also bias p if you store shrunk_wr on leaves
            if "shrunk_wr" in leaf and leaf.get("n", 0) >= 10:
                # mild blend toward hierarchical WR without discarding model p
                p_model = float(ev_model_result.get("p_ev_positive", 0.5) or 0.5)
                p_hier = float(leaf["shrunk_wr"])
                n = float(leaf["n"])
                k = float(getattr(cfg, "HIERARCHICAL_SHRINKAGE_K", 20.0))
                w = n / (n + k)      
                ev_model_result["p_ev_positive"] = (1.0 - w) * p_model + w * p_hier
            if "shrunk_wr" in leaf:
                ev_model_result["hierarchical_wr"] = float(leaf["shrunk_wr"])
            ev_model_result["net_ev_source"] = "hierarchical_shrunk"

            ev_model_result["net_ev_leaf_n"] = leaf["n"]
            ev_model_result["net_ev_leaf_key"] = leaf_key
        else:
            ev_model_result["net_ev_source"] = "alert_baseline"
        row = {
            "pair": pair,
            "alert_key": alert_key,
            "direction": direction,
            "conf_pct": conf_pct,
            "context": context,
        }
        market_state_p_win = None
        ml_calibration_curve = None
        if getattr(cfg, "ENABLE_MARKET_STATE_MODEL", True):
            model = await self._load_market_state_model()
            market_state_p_win = engine.predict_market_state_proba(
                model, votes=votes, context=context, session=session, direction=direction,
            )
            if market_state_p_win is not None:
                ml_calibration_curve = await self._load_ml_calibration_curve()
     
        _rg_mode = str(getattr(cfg, "REGIME_GATE_MODE", "live"))
        _rg_seg = (
            engine.regime_gate_lookup(
                bundle.get("regime_gate"), alert_key, direction, context.get("adx_val"),
            ) if _rg_mode != "off" else None
        )
        try:
            result = engine.trade_quality_score(
                row, ev_model_result, calibration_curve, bundle.get("regime_info"),
                regime_gate=_rg_seg, regime_gate_mode=_rg_mode,
                target_wr=getattr(cfg, "MIN_WIN_RATE", 0.55),
                calibration_min_sample=getattr(cfg, "CALIBRATION_MIN_SAMPLE", 15),
                calibration_slack=getattr(cfg, "CALIBRATION_SLACK", 0.05),
                market_state_p_win=market_state_p_win,
                use_market_state_live=getattr(cfg, "ENABLE_MARKET_STATE_LIVE_SCORE", False),
                ml_calibration_curve=ml_calibration_curve,
            )
        except Exception:
            return None

        if getattr(cfg, "ENABLE_MAE_MFE_TRADE_PLAN", True) and result.get("verdict") != "BLOCKED":
            plan = engine.lookup_mae_mfe_plan(
                bundle.get("mae_mfe_profiles") or {}, pair, alert_key, direction,
            )
            if plan:
                result["trade_plan"] = plan
        _zone_mode = str(getattr(cfg, "ZONE_MODE", "live"))
        if _zone_mode == "live" and result.get("verdict") != "BLOCKED":
            _zone = engine.zone_lookup(
                bundle.get("zone_profiles"), pair, alert_key, direction, context.get("adx_val"),
            )
            if _zone:
                result["trade_zone"] = _zone
        return result

    # ── Kill switch ──────────────────────────────────────────────────────
    async def is_kill_switch_active(self) -> bool:
        """Dispatch hook: poll every cycle. Fail-open on Redis outage is
        consistent with the rest of the system — dispatch already cannot
        function without Redis."""
        if not getattr(cfg, "ENABLE_KILL_SWITCH", False):
            return False
        if self.sdb.degraded or not self.sdb._redis:
            return False
        try:
            return bool(await self.sdb._safe_redis_op(
                lambda: _rc(self.sdb._redis).exists(KILL_SWITCH_KEY),
                2.0, "kill_switch_poll",
            ))
        except Exception:
            return False

    async def clear_kill_switch(self) -> bool:
        """Manual reset after human review."""
        if self.sdb.degraded or not self.sdb._redis:
            return False
        try:
            await self.sdb._safe_redis_op(
                lambda: _rc(self.sdb._redis).delete(KILL_SWITCH_KEY),
                2.0, "kill_switch_clear",
            )
            return True
        except Exception:
            return False

    # ── Stream reading helpers ────────���──────────────────────────�����─────────

    async def _read_stream(self, stream_key: str, count: int) -> List[Dict[str, str]]:
        """Read the most recent `count` entries from an outcome stream."""
        if self.sdb.degraded or not self.sdb._redis:
            return []
        try:
            entries = await self.sdb._safe_redis_op(
                lambda: _rc(self.sdb._redis).xrevrange(stream_key, count=count),
                5.0, f"brain_read:{stream_key}",
            )
        except Exception as e:
            logging.getLogger("macd_bot").warning(f"Brain failed reading {stream_key}: {e}")
            return []
        if not entries:
            return []
        return [fields for _entry_id, fields in entries]

    @staticmethod
    def _parse_rows(rows: List[Dict[str, str]], window_days: Optional[int] = None) -> List[Dict[str, Any]]:
        parsed = []
        seen_keys = set()
        cutoff = int(time.time()) - window_days * 86400 if window_days else None
        for f in rows:
            try:
                score = float(f["score"])
                total = float(f["total"])
                if total <= 0:
                    continue
                entry_ts = int(f.get("entry_ts", 0))
                # FIX #1: entries without timestamps (0) must also be filtered out
                if cutoff is not None and (not entry_ts or entry_ts < cutoff):
                    continue
                pair = f.get("pair", "?")
                alert_key = f.get("alert_key", "?")
                dedup_key = (pair, alert_key, entry_ts)
                if entry_ts and dedup_key in seen_keys:
                    continue
                if entry_ts:
                    seen_keys.add(dedup_key)

                votes_raw = f.get("votes")
                try:
                    votes = json.loads(votes_raw) if votes_raw else None
                except (TypeError, ValueError):
                    votes = None

                context_raw = f.get("context")
                try:
                    row_context = json.loads(context_raw) if context_raw else None
                except (TypeError, ValueError):
                    row_context = None

                mae = _to_opt_float(f, "mae")
                mfe = _to_opt_float(f, "mfe")

                # ── Three-metric fields (backward compatible with old rows) ──
                close_win_raw = f.get("close_win")
                mfe_win_raw = f.get("mfe_win")
                mae_loss_raw = f.get("mae_loss")
                tp_first_raw = f.get("tp_first")

                base_win = f.get("win") == "1"
                close_win_val = (close_win_raw == "1") if close_win_raw else base_win
                mfe_win_val = (mfe_win_raw == "1") if mfe_win_raw else None
                mae_loss_val = (mae_loss_raw == "1") if mae_loss_raw else None
                tp_first_val = (
                    True if tp_first_raw == "1"
                    else False if tp_first_raw == "0"
                    else None
                )

                # ── R:R and Bonus fields (backward compatible) ──
                bonus_win_raw = f.get("bonus_win")
                bonus_win_val = (bonus_win_raw == "1") if bonus_win_raw else False
                rr_achieved_val = _to_opt_float(f, "rr_achieved") or 0.0

                win_weight_parsed = _to_opt_float(f, "win_weight")
                win_weight_val = win_weight_parsed if win_weight_parsed is not None else (1.0 if base_win else 0.0)

                parsed.append({
                    "pair": pair,
                    "alert_key": alert_key,
                    "direction": f.get("direction", "?"),
                    "score": score,
                    "total": total,
                    "conf_pct": score / total * 100.0,
                    "win": base_win,
                    "pct_move": float(f.get("pct_move", 0.0)),
                    "entry_ts": entry_ts,
                    "session": f.get("session", "unknown"),
                    "mae": mae,
                    "mfe": mfe,
                    "votes": votes,
                    "context": row_context,
                    # ── Three-metric fields ──
                    "close_win": close_win_val,
                    "mfe_win": mfe_win_val,
                    "mae_loss": mae_loss_val,
                    "tp_first": tp_first_val,
                    "outcome_reason": f.get("outcome_reason", "unknown"),
                    # ── R:R and Bonus fields ──
                    "bonus_win": bonus_win_val,
                    "rr_achieved": rr_achieved_val,
                    "win_weight": win_weight_val,
                    "signal_price": _to_opt_float(f, "signal_price"),
                    "fill_price": _to_opt_float(f, "fill_price"),
                    "fees_paid_pct": _to_opt_float(f, "fees_paid_pct"),
                    "net_pnl_pct": _to_opt_float(f, "net_pnl_pct"),
                    "realized_cost_pct": _to_opt_float(f, "realized_cost_pct"),
                    **coerce_path_fields(f),
                    "adx_val": _to_opt_float(f, "adx_val"), 
                    "effective_score": _to_opt_float(f, "effective_score"),
                    "effective_required": _to_opt_float(f, "effective_required"),
                    "macro_multiplier": _to_opt_float(f, "macro_multiplier"),
                    "cluster_penalty": _to_opt_float(f, "cluster_penalty"),
                    "rejection_reason": (row_context or {}).get("rejection_reason"),
                })
            except (KeyError, ValueError) as e:
                logging.getLogger("macd_bot").debug(f"Brain: dropping malformed outcome row: {e}")
                continue
        return parsed

    async def _get_rows(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Load real and shadow outcome rows. Default: Redis streams.
        Override in subclasses to read from file archives instead."""
        sample_size = getattr(cfg, "BRAIN_REPORT_STREAM_SAMPLE", 5000)
        window_days = getattr(cfg, "BRAIN_ANALYSIS_WINDOW_DAYS", 30)

        long_sample = getattr(cfg, "BRAIN_LONG_WINDOW_STREAM_SAMPLE", 15000)
        fetch_size = max(sample_size, long_sample)

        real_raw, shadow_raw = await asyncio.gather(
            self._read_stream(RedisKeyPrefix.OUTCOME_LOG_STREAM, fetch_size),
            self._read_stream(RedisKeyPrefix.SHADOW_LOG_STREAM, sample_size),
        )

        self._cached_real_raw = real_raw  # consumed by _get_layered_window_rows

        real_rows = self._parse_rows(real_raw, window_days=window_days)
        shadow_rows = self._parse_rows(shadow_raw, window_days=window_days)

        return real_rows, shadow_rows

    async def _get_layered_window_rows(
        self,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Real-outcome rows for the recent/medium/long-history comparison
        (layered_window_analysis). Medium and long are chronological
        supersets of recent, not separate populations, so this re-filters
        the raw rows already fetched by _get_rows() at
        max(sample_size, long_sample) three ways in Python, instead of
        hitting Redis again."""
        recent_days = getattr(cfg, "BRAIN_ANALYSIS_WINDOW_DAYS", 30)
        medium_days = getattr(cfg, "BRAIN_MEDIUM_WINDOW_DAYS", 90)
        long_days = getattr(cfg, "BRAIN_LONG_WINDOW_DAYS", 180)

        raw = self._cached_real_raw     
        if raw is None:

            # Fallback: _get_rows() wasn't called first this cycle.
            long_sample = getattr(cfg, "BRAIN_LONG_WINDOW_STREAM_SAMPLE", 15000)
            raw = await self._read_stream(
                RedisKeyPrefix.OUTCOME_LOG_STREAM,
                long_sample,
            )

        recent_rows = self._parse_rows(raw, window_days=recent_days)
        medium_rows = self._parse_rows(raw, window_days=medium_days)
        long_rows = self._parse_rows(raw, window_days=long_days)

        return recent_rows, medium_rows, long_rows

    # ── CUSUM drift detection ────────────────────────────────────────────
    async def _load_or_create_cusum(self, alert_key: str) -> CUSUMDetector:
        """Load persisted CUSUM state, or create a fresh detector."""
        if alert_key in self._cusum_detectors:
            return self._cusum_detectors[alert_key]
        saved = await self.sdb.load_cusum_state(alert_key)
        if saved:
            det = CUSUMDetector.from_dict(saved)
        else:
            det = CUSUMDetector(
                target_wr=cfg.MIN_WIN_RATE,
                drift_delta=getattr(cfg, "BRAIN_CUSUM_DRIFT_DELTA", 0.10),
                threshold=getattr(cfg, "BRAIN_CUSUM_THRESHOLD", 2.0),
            )
        self._cusum_detectors[alert_key] = det
        return det

    async def _check_cusum_drift(
        self, real_rows: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Run CUSUM over rows per alert_key, but only the ones not already
        fed in a previous report cycle — real_rows is a rolling window read
        fresh every run, so without a watermark the same trades would be
        replayed into the persisted detector state every cycle."""
        drift_alerts: List[Dict[str, Any]] = []
        by_alert: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in real_rows:
            by_alert[r["alert_key"]].append(r)

        if not by_alert:
            return drift_alerts

        audit = get_audit()
        # ONE pipelined read for every key's watermark + persisted state
        # (was 2-4 sequential round-trips per key). None = read failed:
        # skip the update rather than replay history from a fake 0 watermark.
        bulk = await self.sdb.load_cusum_bulk(list(by_alert))
        if bulk is None:
            audit.record_analysis(
                "cusum", HealthStatus.UNAVAILABLE,
                detail="Redis read failed — CUSUM update skipped this cycle",
            )
            return drift_alerts

        to_save: List[Tuple[str, Dict[str, Any], int]] = []
        for alert_key, rows in by_alert.items():
            watermark, saved = bulk[alert_key]
            if alert_key not in self._cusum_detectors:
                self._cusum_detectors[alert_key] = (
                    CUSUMDetector.from_dict(saved) if saved else CUSUMDetector(
                        target_wr=cfg.MIN_WIN_RATE,
                        drift_delta=getattr(cfg, "BRAIN_CUSUM_DRIFT_DELTA", 0.10),
                        threshold=getattr(cfg, "BRAIN_CUSUM_THRESHOLD", 2.0),
                    )
                )
            det = self._cusum_detectors[alert_key]
            rows_sorted = sorted(
                (r for r in rows if r.get("entry_ts", 0) > watermark),
                key=lambda r: r.get("entry_ts", 0),
            )
            # ── AUDIT: watermark sanity vs the data actually loaded ──
            wm_warning = audit.validate_cusum_watermark(
                alert_key=alert_key,
                watermark=watermark,
                newest_row_ts=max(r.get("entry_ts", 0) for r in rows),
                rows_consumed=len(rows_sorted),
                state_n=det.n,
            )
            if wm_warning:
                drift_alerts.append({
                    "type": "cusum_watermark_anomaly",
                    "severity": "medium",
                    "alert": alert_key,
                    "message": f"⚠️ {wm_warning}",
                })
            if not rows_sorted:
                continue

            last_consumed = watermark
            for r in rows_sorted:
                last_consumed = r.get("entry_ts", 0)
                # Deliberately binary: CUSUM detects edge DECAY. s_neg only
                # accumulates on losses (x < mu), so bonus-weighting wins
                # cannot change decay detection — keep raw win/loss here.
                drifted = det.update(r["win"])
                if drifted:
                    drift_alerts.append({
                        "type": "cusum_drift",
                        "severity": "high",
                        "alert": alert_key,
                        "n": det.n,
                        "s_neg": det.s_neg,
                        "h": det.h,
                        "message": (
                            f"🚨 CUSUM EDGE DECAY on {alert_key}: "
                            f"drift detected after {det.n} outcomes "
                            f"(s_neg={det.s_neg:.2f} > h={det.h:.1f}). "
                            "All config patches FROZEN for this alert. "
                            "Manual review required."
                        ),
                    })
                    break

            to_save.append((alert_key, det.to_dict(), last_consumed))
        await self.sdb.save_cusum_bulk(to_save)
        return drift_alerts

    @staticmethod
    def _reenable_evidence(
        alert_key: str,
        shadow_rows: List[Dict[str, Any]],
        disabled_at: Optional[float],
        now_ts: float,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Decide whether a brain-disabled alert key has earned re-enabling.

        Evidence must be FRESH: shadow outcomes tagged "brain_disabled",
        recorded after the disable. A disabled key produces no real outcomes,
        and its pre-disable rows only lose weight under recency decay (which
        widens the confidence interval and would otherwise let it drift back
        to "viable" with no new information).

        Rules (all must hold): cool-down elapsed; at least
        BRAIN_REENABLE_MIN_NEW_SAMPLES de-clustered outcomes; Wilson lower
        bound of the win rate >= BRAIN_REENABLE_MIN_WR_LO; and net EV > 0.
        Returns (ok, detail); detail always carries the reason and counts."""
        cooldown_h = float(getattr(cfg, "BRAIN_REENABLE_COOLDOWN_HOURS", 72))
        min_new = int(getattr(cfg, "BRAIN_REENABLE_MIN_NEW_SAMPLES", 30))
        min_lo = float(getattr(cfg, "BRAIN_REENABLE_MIN_WR_LO", 0.40))
        detail: Dict[str, Any] = {"n": 0, "needed": min_new}
        if disabled_at is not None and (now_ts - disabled_at) < cooldown_h * 3600.0:
            detail["reason"] = f"cool-down: {(now_ts - disabled_at) / 3600.0:.0f}h of {cooldown_h:.0f}h"
            return False, detail
        rows = [
            r for r in shadow_rows
            if r.get("alert_key") == alert_key
            and r.get("rejection_reason") == "brain_disabled"
            and (disabled_at is None or r["entry_ts"] > disabled_at)
        ]
        # De-cluster: outcomes overlapping within one horizon are not independent
        # evidence, so keep at most one per pair per OUTCOME_LOOKAHEAD_CANDLES.
        gap = int(getattr(cfg, "OUTCOME_LOOKAHEAD_CANDLES", 12)) * 900
        kept: List[Dict[str, Any]] = []
        last_kept: Dict[str, float] = {}
        for r in sorted(rows, key=lambda x: x["entry_ts"]):
            if r["entry_ts"] - last_kept.get(r["pair"], float("-inf")) >= gap:
                kept.append(r)
                last_kept[r["pair"]] = r["entry_ts"]
        n = len(kept)
        detail["n"] = n
        if n < min_new:
            detail["reason"] = f"waiting for evidence: {n}/{min_new} independent outcomes"
            return False, detail
        wins = sum(1 for r in kept if r["win"])
        lo, hi, _ = engine.wilson_ci(wins, n)
        ev = engine.ev_and_kelly_for(kept)[0]
        detail.update(wr=wins / n, lo=lo, hi=hi, ev=ev)
        if lo < min_lo:
            detail["reason"] = f"win-rate lower bound {lo:.0%} below {min_lo:.0%}"
            return False, detail
        if ev <= 0:
            detail["reason"] = f"net EV {ev:+.3f}%/trade is not positive"
            return False, detail
        detail["reason"] = "evidence sufficient"
        return True, detail

    def _is_alert_frozen(self, alert_key: str, drift_alerts: List[Dict]) -> bool:
        return any(
            d.get("alert") == alert_key and d["type"] == "cusum_drift"
            for d in drift_alerts
        )

    # ── Recommendations ─────────────────────────────────────────────────────
    async def generate_recommendations(self) -> Dict[str, Any]:
        """Build the full recommendation set: per-alert verdicts, a confluence
        threshold suggestion, shadow-mode insight, and a machine-readable
        config patch."""
        return await _baseline.build_baseline_recommendations(self)
    # ── Report generation / delivery ───────────────────────────────────────
    async def _next_run_count(self) -> Optional[int]:
        """Persisted run counter (Redis INCR) — safe across cron restarts."""
        if self.sdb.degraded or not self.sdb._redis:
            logging.getLogger("macd_bot").warning(
                "Brain run counter skipped: Redis is degraded or unavailable "
                "— brain report will not fire this run."
            )
            return None
        try:
            result = await self.sdb._safe_redis_op(             
                lambda: _rc(self.sdb._redis).incr(RedisKeyPrefix.BRAIN_RUN_COUNTER),
                2.0, "brain_run_counter",
            )

            if result is None:
                logging.getLogger("macd_bot").warning(
                    "Brain run counter INCR returned None (Redis op likely timed out) "
                    "— brain report will not fire this run."
                )
            return result

        except Exception as e:
            logging.getLogger("macd_bot").warning(f"Brain run counter INCR failed: {e} — brain report will not fire this run.")
            return None

    async def _rollback_run_count(self) -> None:
        """Undo the INCR from _next_run_count when report generation fails,
        so the next run retries this slot instead of waiting a full
        BRAIN_REPORT_INTERVAL_RUNS."""
        if self.sdb.degraded or not self.sdb._redis:
            return
        try:
            await self.sdb._safe_redis_op(
                lambda: _rc(self.sdb._redis).decrby(RedisKeyPrefix.BRAIN_RUN_COUNTER, 1),
                2.0, "brain_run_counter_rollback",
            )
        except Exception as e:
            logging.getLogger("macd_bot").warning(
                f"Brain run-counter rollback failed after a report error: {e} — "
                "the counter may now be off by one, which can skew when the "
                "next report is due"
            )

    @staticmethod
    def _truncate_telegram(lines: List[str], limit: int = 4000) -> str:
        """Telegram hard-caps messages at 4096 chars. Stay well under that
        and note how much was cut rather than letting the send fail outright."""
        msg = "\n".join(lines)
        if len(msg) <= limit:
            return msg
        truncated = msg[:limit]
        last_newline = truncated.rfind("\n")
        if last_newline > 0:
            truncated = truncated[:last_newline]
        # FIX #2: don't leave an unclosed markdown code block
        if truncated.count("```") % 2 == 1:
            truncated += "\n```"
        cut_chars = len(msg) - len(truncated)
        return truncated + escape_markdown_v2(
            f"\n… (truncated, {cut_chars} more characters)"
        )
        
    async def maybe_generate_report(
        self,
        pairs: List[str],
        telegram_queue: Any,
        logger_run: logging.Logger,
    ) -> None:
        """Increment the persisted run counter; if the report interval has
        elapsed, generate and send the analysis report."""
        # FIX #4: guard against zero/negative interval
        interval = getattr(cfg, "BRAIN_REPORT_INTERVAL_RUNS", 48)
        if interval <= 0:
            logger_run.warning("BRAIN_REPORT_INTERVAL_RUNS is <= 0, disabling brain reports.")
            return

        # FIX #7: self-guard
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

        # Alert-only run: only a few days of archive were checked out, so
        # analysing it would produce a misleading report. Return BEFORE
        # _next_run_count() so the counter is not consumed either.
        if getattr(cfg, "BRAIN_ARCHIVE_SHALLOW", False):
            logger_run.debug("Brain: shallow archive this run — report deferred to the full-archive run.")
            return

        run_count = await self._next_run_count()

        if run_count is None or run_count % interval != 0:
            return

        try:
            await self._deliver_report(pairs, telegram_queue, logger_run)
        except Exception:
            await self._rollback_run_count()
            raise

    async def send_report_now(self, pairs: List[str], telegram_queue: Any, logger_run: logging.Logger) -> bool:
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

        if getattr(cfg, "BRAIN_ARCHIVE_SHALLOW", False):
            logger_run.warning("Brain report requested but this run has a shallow archive checkout — skipping.")
            return True

        return await self._deliver_report(pairs, telegram_queue, logger_run)

    async def _deliver_report(self, pairs: List[str], telegram_queue: Any, logger_run: logging.Logger) -> bool:
        """Hook: which report path to invoke once the self-guards in
        maybe_generate_report()/send_report_now() have passed. Subclasses
        override this instead of re-implementing those guards."""
        return await self._generate_and_send(pairs, telegram_queue, logger_run)

    def _build_full_markdown_report(self, recs: Dict[str, Any]) -> str:
        """Full, untruncated report — no Telegram char budget, no top-N slicing.
        This is what the Telegram message's 'full detail' pointer refers to."""
        cc = recs.get("current_config", {})
        ai = recs.get("ai_metrics", {})
        out = [
            f"# Brain Report — {format_ist_time()}",
            "",
            f"Samples: {recs['real_sample_size']} real, {recs['shadow_sample_size']} shadow "
            f"| Gate: Score>={cc.get('CONFLUENCE_MIN_ABS_SCORE')} Pct>={cc.get('CONFLUENCE_MIN_PCT')}%",
        ]
        if ai.get("brier_score") is not None:
            out.append(f"Brier score: {ai['brier_score']:.3f} ({ai.get('brier_status', 'n/a')})")
        if ai.get("net_ev") is not None:
            out.append(f"Net EV: {ai['net_ev']:+.3f}%/trade | Half-Kelly: {ai.get('half_kelly', 0):.1%}")
        out.append("")

        patch = recs.get("config_patch") or []
        if patch:
            out.append("## Suggested Config Changes")
            for p in patch:
                suggested = p.get("suggested")
                if isinstance(suggested, dict):
                    cur = p.get("current") or {}
                    for k, v in suggested.items():
                        out.append(f"- `{p['path']}.{k}`: {cur.get(k, 'n/a')} -> {v}")
                else:
                    out.append(f"- `{p['path']}`: {p.get('current')} -> {suggested}")
            out.append("")

        by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in recs.get("recommendations", []):
            by_type[r["type"]].append(r)

        out.append("## All Findings")
        for rtype, items in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
            out.append(f"### {rtype} ({len(items)})")
            for r in items:
                out.append(f"- **[{r.get('severity', 'n/a')}]** {r['message']}")
            out.append("")

        return "\n".join(out)

    async def _generate_and_send(self, pairs: List[str], telegram_queue: Any, logger_run: logging.Logger) -> bool:
        logger_run.info("Brain generating analysis report...")
        recs = await self.generate_recommendations()
        cc = recs.get("current_config", {})
        ai = recs.get("ai_metrics", {})
        overall_wr = recs.get("overall_win_rate")
        target_wr = cfg.MIN_WIN_RATE

        lines: List[str] = []

        # ── HEADER ──
        lines.append("🧠 *BRAIN REPORT*")
        lines.append(escape_markdown_v2(format_ist_time()))
        lines.append(escape_markdown_v2(
            f"{recs['real_sample_size']} real | {recs['shadow_sample_size']} shadow"
        ))
        lines.append("")

        # ── 📊 HEALTH (compact) ──
        wr_icon = "✅" if (overall_wr or 0) >= target_wr else "⚠️"
        lines.append(escape_markdown_v2(
            f"📊 WR: {overall_wr:.0%} vs {target_wr:.0%} target {wr_icon} | "
            f"Gate: ≥{cc.get('CONFLUENCE_MIN_ABS_SCORE')} / ≥{cc.get('CONFLUENCE_MIN_PCT')}%"
        ))

        health_bits = []
        if ai.get("net_ev") is not None:
            health_bits.append(f"EV {ai['net_ev']:+.3f}%")
        if ai.get("half_kelly") is not None:
            health_bits.append(f"Kelly {ai['half_kelly']:.1%}")
        if ai.get("brier_score") is not None:
            health_bits.append(f"Brier {ai['brier_score']:.3f}")
        if health_bits:
            lines.append(escape_markdown_v2("   " + " | ".join(health_bits)))

        mm = next((r for r in recs["recommendations"] if r["type"] == "three_metric_evaluation"), None)
        if mm:
            lines.append(escape_markdown_v2(
                f"   Close {mm['close_wr']:.0%} | MFE {mm['mfe_wr']:.0%} | "
                f"SL {mm['mae_loss_rate']:.0%} | Clean {mm['clean_win_rate']:.0%}"
            ))
            if mm.get("bonus_rate") is not None:
                lines.append(escape_markdown_v2(
                    f"   Bonus {mm['bonus_rate']:.0%} | RR {mm['avg_rr_achieved']:.2f}R | "
                    f"Wtd WR {mm.get('weighted_wr', 0):.1%}"
                ))
        lines.append("")

        # ── 🔧 REPAIR SHOP ──
        repairs = [r for r in recs["recommendations"] if r["type"] == "repair_shop"]
        if repairs:
            lines.append(f"*🔧 REPAIR SHOP* {escape_markdown_v2(f'({len(repairs)} issues)')}")
            for i, r in enumerate(repairs[:5], 1):
                sev_icon = {"critical": "🚨", "high": "🔴", "medium": "⚠️", "low": "ℹ️"}.get(r["severity"], "•")
                # Compact: first 2 lines only
                msg_lines = r["message"].split("\n")
                compact = msg_lines[0][:120]
                if len(msg_lines) > 1:
                    compact += "\n   " + msg_lines[1].strip()[:100]
                lines.append(f"{escape_markdown_v2(f'{sev_icon} #{i} {compact}')}")
            if len(repairs) > 5:
                lines.append(escape_markdown_v2(f"   +{len(repairs)-5} more"))
            lines.append("")

        # ── 🎯 DO THIS NOW (validated changes only) ──
        patch = recs.get("config_patch") or []
        action_lines = []

        for p in patch:
            note = (p.get("note") or "").lower()
            if "informational" in note:
                continue
            path = p["path"]
            suggested = p.get("suggested")
            if isinstance(suggested, dict):
                cur = p.get("current") or {}
                changed = [(k, cur.get(k, 0), v) for k, v in suggested.items()
                           if abs(v - cur.get(k, 0)) > 0.01]
                if not changed:
                    continue
                changed.sort(key=lambda t: abs(t[2] - t[1]), reverse=True)
                top = ", ".join(f"{k} {c:g}→{s:g}" for k, c, s in changed[:4])
                extra = f" +{len(changed)-4}" if len(changed) > 4 else ""
                action_lines.append(f"• {path}: {top}{extra}")
            else:
                action_lines.append(f"• {path}: {p.get('current')} → {suggested}")

        # Counterfactual best
        counterfactuals = [r for r in recs["recommendations"] if r["type"] == "counterfactual"]
        for r in counterfactuals[:1]:
            action_lines.append(f"• {r['message'][:150]}")

        # Weight optimizer blocked?
        wf_blocked = [r for r in recs["recommendations"] if r["type"] == "weight_optimizer_blocked"]
        for r in wf_blocked[:1]:
            action_lines.append(f"• {r['message'][:150]}")

        if action_lines:
            lines.append("*🎯 DO THIS NOW*")
            for al in action_lines[:6]:
                lines.append(escape_markdown_v2(al))
            lines.append("")

        # ── 🏆 TOP ALERTS ──
        perf = next((r for r in recs["recommendations"] if r["type"] == "per_alert_breakdown"), None)
        alert_data = perf.get("data") if perf else None
        if alert_data:

            from alert_registry import BUY_ALERT_KEYS, SELL_ALERT_KEYS         

            buy_ranked = sorted((t for t in alert_data if t[0] in BUY_ALERT_KEYS), key=lambda t: -t[1])
            sell_ranked = sorted((t for t in alert_data if t[0] in SELL_ALERT_KEYS), key=lambda t: -t[1])
            if buy_ranked or sell_ranked:
                lines.append(escape_markdown_v2(f"🏆 TOP ALERTS (min {getattr(cfg, 'MIN_WIN_RATE_SAMPLE', 20)} trades)"))
                if buy_ranked:
                    lines.append(escape_markdown_v2(
                        "Buy: " + " | ".join(f"{ak} {wr:.0%}({n})" for ak, wr, n, _ in buy_ranked[:2])
                    ))
                if sell_ranked:
                    lines.append(escape_markdown_v2(
                        "Sell: " + " | ".join(f"{ak} {wr:.0%}({n})" for ak, wr, n, _ in sell_ranked[:2])
                    ))
                lines.append("")

        # ── 🤖 AI INSIGHTS ─
        ai_lines = []

        # Synergy / Poison
        synergies = [r for r in recs["recommendations"]
                     if r["type"] == "vote_interaction" and r.get("kind") == "synergy"]
        poisons = [r for r in recs["recommendations"]
                   if r["type"] == "vote_interaction" and r.get("kind") == "poison"]
        for s in synergies[:1]:
            ai_lines.append(f"🔗 {s['message'][:120]}")
        for p in poisons[:1]:
            ai_lines.append(f"☠️ {p['message'][:120]}")

        # Permutation importance
        perm = next((r for r in recs["recommendations"] if r["type"] == "permutation_importance"), None)
        if perm:
            ai_lines.append(f"🤖 {perm['message'][:130]}")

        # Regime
        regime = next((r for r in recs["recommendations"] if r["type"] == "regime_breakdown"), None)
        if regime:
            first_line = regime["message"].split("\n")[0][:130]
            ai_lines.append(f"📊 {first_line}")

        # Weight optimizer confidence
        wopt_rec = next((r for r in recs["recommendations"] if r["type"] == "weight_optimizer"), None)
        if wopt_rec:
            ai_lines.append(f"🧮 {wopt_rec['message'].split(chr(10))[0][:130]}")

        if ai_lines:
            lines.append("*🤖 AI INSIGHTS*")
            for al in ai_lines[:5]:
                lines.append(escape_markdown_v2(al))
            lines.append("")

        # ── ⚠️ CUSUM / FROZEN ──
        cusum_items = [r for r in recs["recommendations"] if r["type"] == "cusum_drift"]
        if cusum_items:
            names = ", ".join(r.get("alert", "?") for r in cusum_items[:7])
            lines.append(escape_markdown_v2(
                f"⚠️ FROZEN (CUSUM drift): {names}. Auto-tuning paused."
            ))
            lines.append("")

        # ── 📉 WEAK / AVOID ──
        disable_alerts = [r for r in recs["recommendations"] if r["type"] == "disable_alert"]
        if disable_alerts:
            lines.append("*📉 UNDERPERFORMING (candidate for disable)*")
            for r in disable_alerts[:3]:
                lines.append(escape_markdown_v2(f"• {r['message'][:130]}"))
            lines.append("")

        # ── AUTO-BLOCK ──
        auto_disabled = [r for r in recs["recommendations"] if r["type"] == "auto_disabled"]
        auto_reenabled = [r for r in recs["recommendations"] if r["type"] == "auto_reenabled"]
        if auto_disabled or auto_reenabled:
            lines.append("*🔒 AUTO-BLOCK*")
            for r in (auto_disabled + auto_reenabled)[:3]:
                icon = "🔴" if r["type"] == "auto_disabled" else "🟢"
                lines.append(f"{icon} {escape_markdown_v2(r['message'][:120])}")
            lines.append("")

        # ── REMAINING FYI (compact, one line each) ──
        shown_types = {
            "repair_shop", "weight_optimizer", "weight_optimizer_blocked",
            "per_alert_breakdown", "vote_interaction", "counterfactual",
            "cusum_drift", "disable_alert", "auto_disabled", "auto_reenabled",
            "three_metric_evaluation", "permutation_importance",
            "dynamic_weights_applied", "parameter_autopsy",
            "config_regression", "config_improvement",
        }
        others = [
            r for r in recs["recommendations"]
            if r["severity"] in ("high", "medium") and r["type"] not in shown_types
        ]
        if others:
            lines.append("*ℹ️ FYI*")
            for r in others[:6]:
                first_line = r["message"].split("\n")[0][:130]
                lines.append(f"• {escape_markdown_v2(first_line)}")
            lines.append("")

        total_recs = recs["recommendation_count"]
        if total_recs == 0:
            lines.append("No actionable signal yet — accumulating samples.")

        msg = self._truncate_telegram(lines)

        # Persist BEFORE attempting send
        report_key = f"brain_report:{int(time.time())}"
        if self.sdb._redis and not self.sdb.degraded:
            await self.sdb._safe_redis_op(
                lambda: _rc(self.sdb._redis).set(report_key, json_dumps(recs), ex=30 * 86400),
                2.0, f"brain_report_persist:{report_key}",
            )

        try:
            from outcome_storage import save_report
            report_path = save_report(self._build_full_markdown_report(recs))
            logger_run.info(f"🧠 Full brain report written to {report_path}")
        except Exception as e:
            logger_run.warning(f"Failed to write full markdown brain report: {e}")

        send_ok = False
        if telegram_queue:
            try:
                result = await telegram_queue.send(msg)
                if result:
                    send_ok = True
                else:
                    logger_run.warning(f"Brain report Telegram send returned False — persisted at {report_key}.")
            except Exception as e:
                logger_run.warning(f"Brain report Telegram send failed ({e}) — persisted at {report_key}.")

        logger_run.info(
            f"🧠 Brain report {'sent' if send_ok else 'persisted (send failed)'} | "
            f"{len(patch)} patch item(s), {total_recs} total recommendations"
        )
        return send_ok

# Backward-compatible aliases (see BrainCore docstring).
BrainEngine = BaseBrainEngine = BrainCore

# Names moved to other modules, re-exported for backward compatibility.
from brain_helpers import (
    _extract_p_value_for_fdr,
)

__all__ = [
    "escape_markdown_v2",
    "cfg",
    "json_dumps",
    "json_loads",
    "format_ist_time",
    "CONFLUENCE_WEIGHTS",
    "RedisKeyPrefix",
    "RedisStateStore",
    "_rc",
    "engine",
    "CUSUMDetector",
    "StabilityGate",
    "get_audit",
    "HealthStatus",
    "_OVERRIDE_COOLDOWN_PREFIX",
    "CALIBRATION_CURVES_KEY",
    "QUALITY_INPUTS_KEY",
    "KILL_SWITCH_KEY",
    "MARKET_STATE_MODEL_KEY",
    "ML_CALIBRATION_KEY",
    "_resolve_config_path",
    "_hget_int",
    "_to_opt_float",
    "_extract_p_value_for_fdr",
    "BrainCore",
    "BrainEngine",
    "BaseBrainEngine",
]
