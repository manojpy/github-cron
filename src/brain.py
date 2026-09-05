#!/usr/bin/env python3
"""
brain.py — analysis / reporting layer on top of the bot's existing win-rate system.

"""
from __future__ import annotations
import asyncio
import json
import logging
import statistics
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from alerts import escape_markdown_v2

from bot_config import cfg, json_dumps, format_ist_time, CONFLUENCE_WEIGHTS
from state import RedisKeyPrefix, RedisStateStore
import threshold_engine as engine

from threshold_engine import CUSUMDetector, StabilityGate

_ALERT_CONFIG_MAP = {
    "strong_reversal_buy":  "ENABLE_STRONG_REVERSAL_ALERT",
    "strong_reversal_sell": "ENABLE_STRONG_REVERSAL_ALERT",
    "choch_buy":            "ENABLE_CHOCH_ALERT",
    "choch_sell":           "ENABLE_CHOCH_ALERT",
    "dynamic_flow_cross_buy":  "ENABLE_DYNAMIC_FLOW_CROSS_ALERT",
    "dynamic_flow_cross_sell": "ENABLE_DYNAMIC_FLOW_CROSS_ALERT",
    "fib_reversal_buy":     "ENABLE_FIB_REVERSAL_ALERT",
    "fib_reversal_sell":    "ENABLE_FIB_REVERSAL_ALERT",
    "ob_reversal_buy":      "ENABLE_OB_GATE",
    "ob_reversal_sell":     "ENABLE_OB_GATE",
    "ppo_signal_up":        "ENABLE_PPO_ALERTS",
    "ppo_signal_down":      "ENABLE_PPO_ALERTS",
    "ppo_zero_up":          "ENABLE_PPO_ALERTS",
    "ppo_zero_down":        "ENABLE_PPO_ALERTS",
    "ppo_adaptive_up":      "ENABLE_PPO_ALERTS",
    "ppo_adaptive_down":    "ENABLE_PPO_ALERTS",
    "rsi_ema5_up":          "ENABLE_RSI_ALERTS",
    "rsi_ema5_down":        "ENABLE_RSI_ALERTS",
    "rsi_cross_adaptive_up":   "ENABLE_RSI_ALERTS",
    "rsi_cross_adaptive_down": "ENABLE_RSI_ALERTS",
    "ppohist_buy":          "ENABLE_PPOHIST_ALERT",
    "ppohist_sell":         "ENABLE_PPOHIST_ALERT",
    # ── previously unmapped (added this revision) ──
    "vwap_up":              "ENABLE_VWAP",
    "vwap_down":            "ENABLE_VWAP",
    "cloud_cross_up":       "ENABLE_CLOUD_CROSS_ALERT",
    "cloud_cross_down":     "ENABLE_CLOUD_CROSS_ALERT",
    "tk_conversion_up":     "ENABLE_TK_CONVERSION_CROSS",
    "tk_conversion_down":   "ENABLE_TK_CONVERSION_CROSS",
    "kijun_cross_up":       "ENABLE_KIJUN_CROSS",
    "kijun_cross_down":     "ENABLE_KIJUN_CROSS",
    "hist_rma_buy":         "ENABLE_HIST_RMA",
    "hist_rma_sell":        "ENABLE_HIST_RMA",
}
# pivot_up_r1 / pivot_down_s2 / etc. — variable-suffix family, matched by prefix
_ALERT_CONFIG_PREFIX_MAP = {
    "pivot_up_":   "ENABLE_PIVOT",
    "pivot_down_": "ENABLE_PIVOT",
}

_OVERRIDE_COOLDOWN_PREFIX = "brain_override_cooldown:"

def _resolve_config_path(alert_key: str) -> Optional[str]:
    path = _ALERT_CONFIG_MAP.get(alert_key)
    if path:
        return path
    for prefix, mapped in _ALERT_CONFIG_PREFIX_MAP.items():
        if alert_key.startswith(prefix):
            return mapped
    return None

def _hget_int(data: dict, key: str, default: int = 0) -> int:
    value = data.get(key)
    if value is None:
        value = data.get(key.encode())
    if value is None:
        return default
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    try:
        return int(value)
    except Exception:
        return default

class BrainEngine:
    """Analysis layer over the bot's existing win-rate infrastructure."""

    def __init__(self, sdb: RedisStateStore):
        self.sdb = sdb
        self.stability_gate = StabilityGate(
            min_history=getattr(cfg, "BRAIN_STABILITY_MIN_HISTORY", 3),
            max_jump=getattr(cfg, "BRAIN_STABILITY_MAX_JUMP", 2.0),
        )
        self._cusum_detectors: Dict[str, CUSUMDetector] = {}

    # ── Rewardable override ─────────────────────────────────────────────────

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
                lambda: self.sdb._redis.hgetall(hiconf_key), 2.0, f"brain_hiconf:{alert_key}",
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
                lambda: self.sdb._redis.set(cooldown_key, "1", nx=True, ex=cooldown_seconds),
                2.0, f"brain_override_cooldown:{alert_key}",
            )
        except Exception:
            acquired = None
        if not acquired:
            return None

        return f"{conf_pct:.0f}% confluence, shadow WR {wr:.0%} over {total} tracked rejections"

    # ── Stream reading helpers ──────────────────────────────────────────────

    async def _read_stream(self, stream_key: str, count: int) -> List[Dict[str, str]]:
        """Read the most recent `count` entries from an outcome stream."""
        if self.sdb.degraded or not self.sdb._redis:
            return []
        try:
            entries = await self.sdb._safe_redis_op(
                lambda: self.sdb._redis.xrevrange(stream_key, count=count),
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
                
                mae_raw = f.get("mae")
                mfe_raw = f.get("mfe")
                try:
                    mae = float(mae_raw) if mae_raw not in (None, "") else None
                except (TypeError, ValueError):
                    mae = None
                try:
                    mfe = float(mfe_raw) if mfe_raw not in (None, "") else None
                except (TypeError, ValueError):
                    mfe = None

                parsed.append({
                    "pair": pair, 
                    "alert_key": alert_key,
                    "direction": f.get("direction", "?"),
                    "score": score,
                    "total": total,
                    "conf_pct": score / total * 100.0,
                    "win": f.get("win") == "1",
                    "pct_move": float(f.get("pct_move", 0.0)),
                    "entry_ts": entry_ts,
                    "session": f.get("session", "unknown"),
                    "mae": mae,
                    "mfe": mfe,
                    "votes": votes,
                    "context": row_context,
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
        real_raw, shadow_raw = await asyncio.gather(
            self._read_stream(RedisKeyPrefix.OUTCOME_LOG_STREAM, sample_size),
            self._read_stream(RedisKeyPrefix.SHADOW_LOG_STREAM, sample_size),
        )
        real_rows = self._parse_rows(real_raw, window_days=window_days)
        shadow_rows = self._parse_rows(shadow_raw, window_days=window_days)
        return real_rows, shadow_rows

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

        for alert_key, rows in by_alert.items():
            watermark = await self.sdb.load_cusum_watermark(alert_key)
            rows_sorted = sorted(
                (r for r in rows if r.get("entry_ts", 0) > watermark),
                key=lambda r: r.get("entry_ts", 0),
            )
            if not rows_sorted:
                continue
            det = await self._load_or_create_cusum(alert_key)
            for r in rows_sorted:
                drifted = det.update(r["win"])
                if drifted:
                    drift_alerts.append({
                        "type": "cusum_drift",
                        "severity": "high",
                        "alert": alert_key,
                        "message": (
                            f"🚨 CUSUM EDGE DECAY on {alert_key}: "
                            f"drift detected after {det.n} outcomes "
                            f"(s_neg={det.s_neg:.2f} > h={det.h:.1f}). "
                            f"All config patches FROZEN for this alert. "
                            f"Manual review required."
                        ),
                    })
                    break
            await self.sdb.save_cusum_state(alert_key, det.to_dict())
            await self.sdb.save_cusum_watermark(alert_key, rows_sorted[-1]["entry_ts"])
        return drift_alerts

    def _is_alert_frozen(self, alert_key: str, drift_alerts: List[Dict]) -> bool:
        return any(
            d.get("alert") == alert_key and d["type"] == "cusum_drift"
            for d in drift_alerts
        )

    # ── Recommendations ──────────────────────────────────────────────────────

    async def generate_recommendations(self) -> Dict[str, Any]:
        """Build the full recommendation set: per-alert verdicts, a confluence
        threshold suggestion, shadow-mode insight, and a machine-readable
        config patch."""
        real_rows, shadow_rows = await self._get_rows()
        recommendations: List[Dict[str, Any]] = []
        config_patch: List[Dict[str, Any]] = []
        seen_paths = set()

        min_sample = getattr(cfg, "MIN_WIN_RATE_SAMPLE", 20)
        target_wr = cfg.MIN_WIN_RATE
        disable_wr = getattr(cfg, "BRAIN_ALERT_DISABLE_THRESHOLD_WR", 0.40)
        star_wr = getattr(cfg, "BRAIN_STAR_ALERT_WR", 0.70)

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
        for alert_key, s in alert_stats.items():
            total = len(s["rows"])
            if total < min_sample:
                continue
            wins = sum(1 for r in s["rows"] if r["win"])
            if recency_on:
                wr, n_eff, lo, hi = engine.weighted_win_rate(s["rows"], decay_days=recency_decay_days)
                sample_label = f"{total} samples (n_eff={n_eff:.0f} recency-weighted, {recency_decay_days:.0f}d decay)"
            else:
                wr = wins / total
                lo, hi, _ = engine.wilson_ci(wins, total)
                sample_label = f"{total} samples"
            auto_eligible = auto_disable_on and total >= auto_disable_min
            if hi < disable_wr:
                alert_verdicts[alert_key] = "disable"
                recommendations.append({
                    "type": "disable_alert", "severity": "high", "alert": alert_key,
                    "win_rate": round(wr, 3), "sample_size": total, "pairs_affected": len(s["pairs"]),
                    "message": (
                        f"DISABLE {alert_key}: {wr:.0%} WR over {sample_label} across "
                        f"{len(s['pairs'])} pairs (95% CI upper bound {hi:.0%}, still below {disable_wr:.0%})."
                    ),
                })
                if auto_eligible and alert_key not in current_disabled_keys:
                    if await self.sdb.set_alert_key_disabled(alert_key, True):
                        recommendations.append({
                            "type": "auto_disabled", "severity": "high", "alert": alert_key,
                            "message": (
                                f"🔒 Auto-disabled {alert_key}: {wr:.0%} WR over {sample_label} "
                                f"(≥{auto_disable_min} required)."
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
                if auto_eligible and alert_key in current_disabled_keys:
                    if await self.sdb.set_alert_key_disabled(alert_key, False):
                        recommendations.append({
                            "type": "auto_reenabled", "severity": "medium", "alert": alert_key,
                            "message": f"🔓 Re-enabled {alert_key}: recovered to {wr:.0%} WR over {sample_label}.",
                        })
            else:
                alert_verdicts[alert_key] = "monitor"
                recommendations.append({
                    "type": "monitor", "severity": "medium", "alert": alert_key,
                    "win_rate": round(wr, 3), "sample_size": total,
                    "message": f"{alert_key} viable ({wr:.0%} WR, {sample_label}).",
                })
                if auto_eligible and alert_key in current_disabled_keys:
                    if await self.sdb.set_alert_key_disabled(alert_key, False):
                        recommendations.append({
                            "type": "auto_reenabled", "severity": "medium", "alert": alert_key,
                            "message": f"🔓 Re-enabled {alert_key}: recovered to {wr:.0%} WR over {sample_label}.",
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
                        f"_ALERT_CONFIG_MAP — no config_patch was emitted. Add a mapping or disable manually."
                    ),
                })
        threshold_rec: Dict[str, Any] = {}
        net_ev = half_kelly = kelly_wr = None
        rec = engine.recommend_threshold(
            real_rows, target_winrate=target_wr, min_sample=min_sample,
        ) if real_rows else {"valid": False}

        # ── Brier Score / Calibration ────────────────────────────────────
        brier, cal_curve = engine.brier_score_and_calibration(real_rows)
        cal_alerts = engine.calibration_alert(real_rows)
        brier_status = "Healthy" if brier < 0.20 else "MISALIBRATED"
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
            if emit_patch:
                history = await self.sdb.load_threshold_history()
                gate_ok, gate_reason = self.stability_gate.approve(
                    target_floor, history,
                )
                if not gate_ok:
                    # Downgrade: don't emit the patch, warn instead
                    threshold_rec["severity"] = "medium"
                    threshold_rec["stability_blocked"] = True
                    threshold_rec["message"] += (
                        f"\n⚠️ STABILITY GATE BLOCKED: {gate_reason}. "
                        f"Patch suppressed to prevent oscillation."
                    )
                    emit_patch = False
                else:
                    await self.sdb.save_threshold_value(target_floor)
        
            # ── Net EV + Kelly sizing at recommended threshold ───────────────
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
            if emit_patch:
                config_patch.append({
                    "path": "CONFLUENCE_MIN_ABS_SCORE", "current": cfg.CONFLUENCE_MIN_ABS_SCORE,
                    "suggested": target_floor, "supporting_samples": rec_n,
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
                mc = engine.monte_carlo_walk_forward(
                    real_rows, n_simulations=cfg.BRAIN_MC_SIMULATIONS,
                    min_sample=min_sample, target_winrate=target_wr,
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
                        f"Diagnostic only — no regime-specific threshold applied yet."
                    ),
                })

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
                        "those specific trades underperform target WR:\n" + "\n".join(lines) + "\n"
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
            if rec.get("overlapping_toxic"):
                worst = max(rec["overlapping_toxic"], key=lambda t: t[1])
                recommendations.append({
                    "type": "toxic_zone_note", "severity": "low",
                    "message": (
                        f"Note: a toxic bucket (score {worst[0]:.1f}-{worst[1]:.1f}, {worst[2]:.0%} WR) "
                        f"exists at or above the recommended threshold. Cumulative stats already price "
                        f"this in — worth checking the per-alert breakdown below for what's firing there."
                    ),
                })

        # ── Config Version Regression Check ─────────────────────────────

        config_comparisons = engine.compare_config_versions(real_rows, min_sample=min_sample)
        for comp in config_comparisons:
            if comp["regression"]:
                recommendations.append({
                    "type": "config_regression", "severity": "high",
                    "message": (
                        f"🚨 Config regression: WR fell {comp['prev_wr']:.0%}→{comp['cur_wr']:.0%} "
                        f"after config {comp['prev_version']}→{comp['cur_version']} "
                        f"(n={comp['prev_n']}/{comp['cur_n']}). Consider reverting."
                    ),
                })

        # ── Temporal drift (from the same recommend_threshold() call —
        #    no separate computation, no risk of disagreeing with the CLI) ──
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

        # ── Per-pair breakdown (worst pairs only, keeps report short) ──
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

        # ── Per-pair session breakdown (informational) ──────────────────
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

        # ── Pain-Adjusted Win Rate (MAE/MFE-aware) ───────────────────────
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
       
        # ── Per-pair confluence thresholds ──────────────────────────────
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
                wf = engine.validate_threshold_walk_forward(
                    pair_rows, target_winrate=target_wr, min_sample=pair_min_sample,
                )
                if wf["valid"] and wf.get("passed") is False:
                    pair_threshold_lines.append(
                        f"  • {pair}: suggested {suggested:.1f} (was {current:.1f}) — "
                        f"NOT applied, failed walk-forward ({wf['holdout_wr']:.0%} holdout WR)"
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
                    pair_threshold_lines.append(
                        f"  • {pair}: {current:.1f} -> {suggested:.1f} "
                        f"({prec['rec_wr']:.0%} WR, n={prec['rec_n']}) [applied]"
                    )
            if pair_threshold_lines:
                recommendations.append({
                    "type": "pair_thresholds", "severity": "medium",
                    "message": (
                        "Per-pair confluence thresholds (overriding CONFLUENCE_MIN_ABS_SCORE "
                        "for these pairs only):\n" + "\n".join(pair_threshold_lines)
                    ),
                })
        # ── Vote importance (top lift / top drag only) ──
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

        # ── Shadow-mode insight: rejected alerts that would have won ──
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

        # ── Vote-Count OOD summary ─────────────────────────────────────────
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

        severity_order = {"high": 0, "medium": 1, "low": 2}
        recommendations.sort(key=lambda x: severity_order.get(x["severity"], 3))

        return {
            "generated_at": int(time.time()),
            "real_sample_size": len(real_rows),
            "shadow_sample_size": len(shadow_rows),
            "recommendation_count": len(recommendations),
            "recommendations": recommendations,
            "shadow_summary": shadow_summary,
            "config_patch": config_patch,
            "current_config": {
                "CONFLUENCE_MIN_ABS_SCORE": cfg.CONFLUENCE_MIN_ABS_SCORE,
                "CONFLUENCE_MIN_PCT": cfg.CONFLUENCE_MIN_PCT,
            },
            "ai_metrics": {
                "brier_score": round(brier, 4),
                "brier_status": brier_status,
                "net_ev": round(net_ev, 4) if net_ev is not None else None,
                "half_kelly": round(half_kelly, 4) if half_kelly is not None else None,
                "cusum_drifts": len(drift_alerts),
                "threshold_history": await self.sdb.load_threshold_history(),
                "ood_status": ood_status,
            },
        }

    # ── Report generation / delivery ────────────────────────────────────────

    async def _next_run_count(self) -> Optional[int]:
        """Persisted run counter (Redis INCR) — safe across cron restarts."""
        if self.sdb.degraded or not self.sdb._redis:
            return None
        try:
            return await self.sdb._safe_redis_op(
                lambda: self.sdb._redis.incr(RedisKeyPrefix.BRAIN_RUN_COUNTER),
                2.0, "brain_run_counter",
            )
        except Exception:
            return None

    async def _rollback_run_count(self) -> None:
        """Undo the INCR from _next_run_count when report generation fails,
        so the next run retries this slot instead of waiting a full
        BRAIN_REPORT_INTERVAL_RUNS."""
        if self.sdb.degraded or not self.sdb._redis:
            return
        try:
            await self.sdb._safe_redis_op(
                lambda: self.sdb._redis.decrby(RedisKeyPrefix.BRAIN_RUN_COUNTER, 1),
                2.0, "brain_run_counter_rollback",
            )
        except Exception:
            pass

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

        run_count = await self._next_run_count()
        if run_count is None or run_count % interval != 0:
            return

        try:
            await self._generate_and_send(pairs, telegram_queue, logger_run)
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

        return await self._generate_and_send(pairs, telegram_queue, logger_run)

    async def _generate_and_send(self, pairs: List[str], telegram_queue: Any, logger_run: logging.Logger) -> bool:
        logger_run.info("Brain generating analysis report...")
        recs = await self.generate_recommendations()

        cc = recs.get("current_config", {})
        lines = [
            "🧠 *BRAIN REPORT*",
            escape_markdown_v2(format_ist_time()),
            escape_markdown_v2(
                f"Samples: {recs['real_sample_size']} real, {recs['shadow_sample_size']} shadow | "
                f"Score≥{cc.get('CONFLUENCE_MIN_ABS_SCORE')} Pct≥{cc.get('CONFLUENCE_MIN_PCT')}%"
            ),
        ]

        # ── Calibration / drift warnings (only if something's actually wrong) ──
        ai = recs.get("ai_metrics", {})
        warn_bits = []
        if ai.get("brier_score") is not None and ai.get("brier_status") not in (None, "OK"):
            warn_bits.append(f"Brier {ai['brier_score']:.3f} ({ai['brier_status']})")
        if ai.get("cusum_drifts"):
            warn_bits.append(f"CUSUM {ai['cusum_drifts']} alert(s)")
        if warn_bits:
            lines.append(escape_markdown_v2("⚠️ " + " | ".join(warn_bits)))

        # ── Net EV / position sizing (directly actionable, worth the space) ──
        size_bits = []
        if ai.get("net_ev") is not None:
            size_bits.append(f"EV {ai['net_ev']:+.3f}%/trade")
        if ai.get("half_kelly") is not None:
            size_bits.append(f"Half-Kelly {ai['half_kelly']:.1%}")
        if size_bits:
            lines.append(escape_markdown_v2("💰 " + " | ".join(size_bits)))
        lines.append("")

        # ── Weakest / strongest alerts (extremes only — full table stays in Redis) ──
        perf = next((r for r in recs["recommendations"] if r["type"] == "per_alert_breakdown"), None)
        if perf:
            rows = [ln for ln in perf["message"].split("\n")[1:] if ln and not ln.startswith("...")]
            if rows:
                lines.append(escape_markdown_v2(f"📊 Weakest: {rows[0]}"))
                if len(rows) > 1:
                    lines.append(escape_markdown_v2(f"📊 Strongest: {rows[-1]}"))
                lines.append("")

        # ── Auto-block status (actual state changes, always worth surfacing) ──
        auto_disabled = [r for r in recs["recommendations"] if r["type"] == "auto_disabled"]
        auto_reenabled = [r for r in recs["recommendations"] if r["type"] == "auto_reenabled"]
        if auto_disabled or auto_reenabled:
            lines.append("*🔒 AUTO-BLOCK*")
            for r in (auto_disabled + auto_reenabled)[:3]:
                icon = "🔴" if r["type"] == "auto_disabled" else "🟢"
                lines.append(f"{icon} {escape_markdown_v2(r['message'][:150])}")
            lines.append("")

        # ── Suggested changes: the config patch, compacted to one line each ──
        patch = recs["config_patch"]
        patch_derived_types = {"parameter_autopsy", "weight_optimizer"}
        shown_patch = 0
        if patch:
            patch_lines = []
            for p in patch:
                note = (p.get("note") or "").lower()
                if "informational" in note:
                    continue  # derived/non-binding entry, not an action item
                path = p["path"]
                suggested = p.get("suggested")
                if isinstance(suggested, dict):
                    cur = p.get("current") or {}
                    deltas = sorted(
                        ((k, cur.get(k, 0), v) for k, v in suggested.items()),
                        key=lambda t: abs(t[2] - t[1]), reverse=True,
                    )
                    changed = [(k, c, s) for k, c, s in deltas if abs(s - c) > 0.01]
                    if not changed:
                        continue
                    top = ", ".join(f"{k} {c:g}→{s:g}" for k, c, s in changed[:6])
                    line = f"• {path}: {top}"
                    if len(changed) > 6:
                        line += f" (+{len(changed) - 6} more)"
                else:
                    line = f"• {path}: {p.get('current')} → {suggested}"
                patch_lines.append(escape_markdown_v2(line))
            if patch_lines:
                lines.append("*🎯 SUGGESTED CHANGES*")
                lines.extend(patch_lines)
                lines.append("")
                shown_patch = len(patch_lines)

        # ── Other high/medium findings not already covered by the patch above ──
        skip_types = patch_derived_types | {
            "dynamic_weights_applied", "dynamic_weights_shadow",
            "dynamic_weights_persist_failed", "auto_disabled", "auto_reenabled",
        }
        others = [
            r for r in recs["recommendations"]
            if r["severity"] in ("high", "medium") and r["type"] not in skip_types
        ]
        if others:
            lines.append("*🔎 KEY FINDINGS*")
            for r in others[:5]:
                first_line = r["message"].split("\n")[0]
                lines.append(f"• {escape_markdown_v2(first_line[:180])}")
            lines.append("")

        total_recs = recs["recommendation_count"]
        shown = shown_patch + len(auto_disabled[:3]) + len(auto_reenabled[:3]) + min(len(others), 5)
        if total_recs == 0:
            lines.append("No actionable signal yet — still accumulating samples.")
        elif total_recs > shown:
            lines.append(escape_markdown_v2(f"+{total_recs - shown} more items — full report persisted."))

        msg = self._truncate_telegram(lines)

        # Persist BEFORE attempting the Telegram send
        report_key = f"brain_report:{int(time.time())}"
        if self.sdb._redis and not self.sdb.degraded:
            result = await self.sdb._safe_redis_op(
                lambda: self.sdb._redis.set(report_key, json_dumps(recs), ex=30 * 86400),
                2.0, f"brain_report_persist:{report_key}",
            )
            if result is None:
                logger_run.warning(f"Failed to persist brain report {report_key}")

        send_ok = False
        if telegram_queue:
            try:
                result = await telegram_queue.send(msg)

                if result:
                    send_ok = True
                else:
                    logger_run.warning(
                        f"Brain report Telegram send returned False (likely API rejection) — "
                        f"report is still persisted at {report_key}. "
                        f"Check TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID and MarkdownV2 formatting."
                    )

            except Exception as e:
                logger_run.warning(
                    f"Brain report Telegram send failed ({e}) — report is still persisted at {report_key}."
                )

        logger_run.info(
            f"🧠 Brain report {'sent' if send_ok else 'persisted (send failed)'} | "
            f"{shown_patch} patch item(s), {len(others)} other finding(s), "
            f"{total_recs} total recommendations"
        )

        return send_ok