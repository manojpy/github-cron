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
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from alerts import escape_markdown_v2, BUY_ALERT_KEYS, SELL_ALERT_KEYS

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

def _extract_p_value_for_fdr(
    rec: Dict[str, Any],
    real_rows: List[Dict[str, Any]],
    min_sample: int,
) -> Optional[float]:
    """Best-effort p-value for a recommendation.

    Returns None for recs that don't make an explicit statistical claim
    (informational, repair_shop, three_metric summaries that were never
    paired-tested, etc.). Returning None excludes the rec from the BH
    pass rather than contributing a fabricated 0.5 or 1.0 that would
    pollute the correction's m count.

    Every branch guards against missing or malformed fields — the extractor
    runs inside the report loop, and a KeyError here would take down the
    whole report for one malformed rec.
    """
    rtype = rec.get("type")

    # ── Interactions: p_value already stamped by the miner ──
    # The miner is the only place that knows which arm is the correct null
    # for each of the three branches (synergy, v2-poison-v1, v1-poison-v2),
    # so we read rather than reconstruct.
    if rtype == "vote_interaction":
        p = rec.get("p_value")
        return float(p) if isinstance(p, (int, float)) else None

    # ── Calibration: one-sample against the train-split prediction ──
    # `predicted` is a fixed reference from the train split, not a second
    # sample from the same population, so this is a one-sample test. The
    # earlier two_proportion form incorrectly treated a fixed proportion
    # as if it were n i.i.d. Bernoulli draws, inflating the effective
    # sample size and shrinking the p-value.
    if rtype == "calibration_divergence":
        n = rec.get("n")
        pred = rec.get("predicted")
        obs = rec.get("observed")
        if all(isinstance(v, (int, float)) for v in (n, pred, obs)) and n > 0:
            wins_obs = int(round(obs * n))
            return engine.one_proportion_p_value(wins_obs, int(n), float(pred))
        return None

    # ── Parameter autopsy: one-sample against MIN_WIN_RATE ──
    # Claim: the worst bucket's WR is inconsistent with the target.
    if rtype == "parameter_autopsy":
        n = rec.get("n")
        wr = rec.get("wr")
        if isinstance(n, int) and isinstance(wr, (int, float)) and n > 0:
            wins = int(round(wr * n))
            return engine.one_proportion_p_value(wins, n, cfg.MIN_WIN_RATE)
        return None

    # ── Conditional gating: two-proportion, above-threshold vs below ──
    # Claim: WR differs materially depending on which side of the
    # condition the row falls on.
    if rtype == "conditional_gating":
        a_n, a_wr = rec.get("above_n"), rec.get("above_wr")
        b_n, b_wr = rec.get("below_n"), rec.get("below_wr")
        if (isinstance(a_n, int) and isinstance(b_n, int)
                and isinstance(a_wr, (int, float)) and isinstance(b_wr, (int, float))
                and a_n > 0 and b_n > 0):
            wins_a = int(round(a_wr * a_n))
            wins_b = int(round(b_wr * b_n))
            return engine.two_proportion_p_value(wins_a, a_n, wins_b, b_n)
        return None

    # ── Disable alert: one-sample against BRAIN_ALERT_DISABLE_THRESHOLD_WR ──
    # Claim: this alert's pooled WR is below the disable threshold. The
    # gate at emission time is `hi < disable_wr` (Wilson upper bound), so
    # the rec is already directional; the p-value is a second look.
    if rtype == "disable_alert":
        n = rec.get("sample_size")
        wr = rec.get("win_rate")
        if isinstance(n, int) and isinstance(wr, (int, float)) and n > 0:
            wins = int(round(wr * n))
            p0 = getattr(cfg, "BRAIN_ALERT_DISABLE_THRESHOLD_WR", 0.40)
            return engine.one_proportion_p_value(wins, n, p0)
        return None

    # ── Recovered alert: one-sample against MIN_WIN_RATE ──
    if rtype == "recovered_alert":
        n = rec.get("sample_size")
        wr = rec.get("win_rate")
        if isinstance(n, int) and isinstance(wr, (int, float)) and n > 0:
            wins = int(round(wr * n))
            return engine.one_proportion_p_value(wins, n, cfg.MIN_WIN_RATE)
        return None

    # ── Config regression: two-proportion, prev vs current ──
    if rtype in ("config_regression", "config_improvement"):
        prev_n, cur_n = rec.get("prev_n"), rec.get("cur_n")
        prev_wr, cur_wr = rec.get("prev_wr"), rec.get("cur_wr")
        if (isinstance(prev_n, int) and isinstance(cur_n, int)
                and isinstance(prev_wr, (int, float)) and isinstance(cur_wr, (int, float))
                and prev_n > 0 and cur_n > 0):
            wins_prev = int(round(prev_wr * prev_n))
            wins_cur = int(round(cur_wr * cur_n))
            return engine.two_proportion_p_value(wins_cur, cur_n, wins_prev, prev_n)
        return None

    # ── Three-metric close-vs-MFE gap: exact McNemar (paired) ──
    if rtype == "three_metric_evaluation":
        mfe_only = rec.get("mfe_only")
        close_only = rec.get("close_only")
        if isinstance(mfe_only, int) and isinstance(close_only, int):
            return engine.mcnemar_exact_p(mfe_only, close_only)
        return None

    # ── Permutation importance: top-signal t-test against 0 ──
    # Approximate: the shuffle distribution gives a standard error for the
    if rtype == "permutation_importance":
        mean = rec.get("top_importance")
        std = rec.get("top_std")
        n_perm = rec.get("n_permutations")
        if (isinstance(mean, (int, float)) and isinstance(std, (int, float))
                and isinstance(n_perm, int) and n_perm > 1 and std > 0.0):
            z = abs(mean) / (std / math.sqrt(n_perm))
            return math.erfc(z / math.sqrt(2.0))
        return None

    # ── Fallthrough: no clean single-hypothesis claim ──
    return None

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

                rr_achieved_raw = f.get("rr_achieved")
                try:
                    rr_achieved_val = float(rr_achieved_raw) if rr_achieved_raw not in (None, "") else 0.0
                except (TypeError, ValueError):
                    rr_achieved_val = 0.0

                win_weight_raw = f.get("win_weight")
                try:
                    win_weight_val = float(win_weight_raw) if win_weight_raw not in (None, "") else (1.0 if base_win else 0.0)
                except (TypeError, ValueError):
                    win_weight_val = 1.0 if base_win else 0.0

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
                wr, n_eff, lo, hi = engine.weighted_win_rate_with_bonus(
                    s["rows"], decay_days=recency_decay_days
                )
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
        target_floor: Optional[float] = None
        rec = engine.recommend_threshold(
            real_rows, target_winrate=target_wr, min_sample=min_sample,
        ) if real_rows else {"valid": False}

        # ── Brier Score / Calibration ────────────────────────────────────
        brier, cal_curve = engine.brier_score_and_calibration(real_rows)
        cal_alerts = engine.calibration_alert(real_rows)
        if cal_curve or cal_alerts:
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
            if emit_patch:
                history = await self.sdb.load_threshold_history()
                gate_ok, gate_reason = self.stability_gate.approve(
                    target_floor, history,
                )
                if not gate_ok:
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
                f"  • Close WR (point-in-time): {close_wr:.0%} "
                f"[{mm_summary['close_wilson'][0]:.0%}-{mm_summary['close_wilson'][1]:.0%}]\n"
                f"  • MFE WR (TP ever hit):    {mfe_wr:.0%} "
                f"[{mm_summary['mfe_wilson'][0]:.0%}-{mm_summary['mfe_wilson'][1]:.0%}]\n"
                f"  • MAE Loss Rate (SL hit):  {mae_rate:.0%}\n"
                f"  • Clean Win (TP w/o SL):   {clean_wr:.0%}"
            )
            if gap > 0.05:
                summary_msg += (
                    f"\n⚠️ Gap: {gap:+.0%} of trades hit TP but reversed before "
                    f"candle {cfg.OUTCOME_LOOKAHEAD_CANDLES}. "
                    f"The close-based WR underestimates true profitability by {gap:.0%}."
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
                # ── FDR: exact McNemar on the discordant 2×2 cells ──
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
                    f"🔍 Alerts where MFE WR >> Close WR (take-profit would have captured "
                    f"these wins but the point-in-time check misses them):\n"
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
            worst = max(rec["overlapping_toxic"], key=lambda t: t[1])
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
                "brier_score": round(brier, 4),
                "brier_status": brier_status,
                "net_ev": round(net_ev, 4) if net_ev is not None else None,
                "half_kelly": round(half_kelly, 4) if half_kelly is not None else None,
                "cusum_drifts": len(drift_alerts),
                "threshold_history": await self.sdb.load_threshold_history(),
                "ood_status": ood_status,
            },
        }

    # ── Report generation / delivery ─────────────────────���──────────────────

    async def _next_run_count(self) -> Optional[int]:
        """Persisted run counter (Redis INCR) — safe across cron restarts."""
        if self.sdb.degraded or not self.sdb._redis:
            logger.warning(
                "Brain run counter skipped: Redis is degraded or unavailable "
                "— brain report will not fire this run."
            )
            return None
        try:
            result = await self.sdb._safe_redis_op(
                lambda: self.sdb._redis.incr(RedisKeyPrefix.BRAIN_RUN_COUNTER),
                2.0, "brain_run_counter",
            )
            if result is None:
                logger.warning(
                    "Brain run counter INCR returned None (Redis op likely timed out) "
                    "— brain report will not fire this run."
                )
            return result
        except Exception as e:
            logger.warning(f"Brain run counter INCR failed: {e} — brain report will not fire this run.")
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
            from alerts import BUY_ALERT_KEYS, SELL_ALERT_KEYS
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

        # ── 🤖 AI INSIGHTS ──
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
            lines.append("*📉 WEAK / AVOID*")
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
            "dynamic_weights_applied", "dynamic_weights_shadow",
            "dynamic_weights_persist_failed", "parameter_autopsy",
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
                lambda: self.sdb._redis.set(report_key, json_dumps(recs), ex=30 * 86400),
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