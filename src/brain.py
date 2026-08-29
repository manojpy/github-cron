#!/usr/bin/env python3
"""
brain.py — analysis / reporting layer on top of the bot's existing win-rate system.

Deliberately does NOT duplicate outcome tracking, win-rate storage, or pending-
outcome resolution — those already live in state.py (record_pending_outcome,
resolve_pending_outcomes, get_alert_win_rate, batch_get_alert_win_rates) and are
reused as-is. This module adds three things on top:

  1. check_rewardable_override() — called from alerts.py's win-rate gate. Lets a
     win-rate-rejected alert through if its confluence score sits in the
     'rewardable' bucket AND that bucket has its own solid shadow-tracked win
     rate (pooled across pairs — per-pair shadow samples are too sparse).
     Rate-limited per alert_key (BRAIN_OVERRIDE_COOLDOWN_SECONDS) so a volatile
     market can't flood Telegram with overrides for the same alert type.
  2. generate_recommendations() — reads OUTCOME_LOG_STREAM (real trades) and
     SHADOW_LOG_STREAM (rejected-but-tracked trades) to produce per-alert win
     rates, a confluence-threshold suggestion, and a structured config patch.
  3. maybe_generate_report() — sends a Telegram report every
     BRAIN_REPORT_INTERVAL_RUNS runs. The run counter is stored in Redis
     (BRAIN_RUN_COUNTER), not in-memory, because run_once() is a fresh process
     per cron invocation and has no persistent memory across runs.

Shadow-mode recording/resolution (record_shadow_pending_outcome,
resolve_shadow_pending_outcomes) live in state.py alongside their real-trade
twins, for the same reason the rest of the outcome pipeline lives there.

── Changelog (this revision) ──────────────────────────────────────────────
Fixes a set of issues found in review, all verified against the live
codebase (state.py, gates.py, alerts.py, bot_config.py) before applying:

  • Confluence threshold search now uses the Wilson LOWER BOUND (not raw
    win rate, not "most samples") and works in RAW SCORE space, matching
    analyze_confluence_thresholds.py. The old version picked the threshold
    with the largest qualifying sample — which is mechanically always the
    lowest threshold that clears the bar, regardless of whether a higher
    threshold had a meaningfully better win rate.
  • Config patch now emits CONFLUENCE_MIN_ABS_SCORE (raw score), which is
    what the actual gate (alerts.py: max(pct_floor, CONFLUENCE_MIN_ABS_SCORE))
    often binds on — patching only CONFLUENCE_MIN_PCT could previously be a
    complete no-op.
  • Per-alert disable/star verdicts use Wilson bounds, not raw win rate, so
    a 20-sample noisy bucket can't trigger a disable recommendation.
  • Alert-config-path grouping fix: alerts that share one config key (e.g.
    tlr_buy/tlr_sell both map to ENABLE_TLR_ALERT) are only recommended for
    disable if BOTH directions are bad. If just one direction is bad, this
    now emits an "investigate" recommendation instead of silently proposing
    a patch that would also kill the good direction.
  • _ALERT_CONFIG_MAP completed (14 previously-unmapped alert types added,
    plus a prefix fallback for the pivot_up_*/pivot_down_* family). An
    unmapped disable recommendation now emits an explicit warning instead
    of silently producing no config_patch entry.
  • EV and R:R added throughout, using abs(pct_move) — pct_move is signed
    by price direction, not by trade outcome (a winning sell has a negative
    pct_move), so raw signed averaging was cancelling wins toward zero.
  • Direction split, per-pair breakdown, and vote-importance ranking added.
  • detect_temporal_drift ported, using wall-clock time (not last-trade
    timestamp, which would be wrong if the bot had downtime).
  • Alert-frequency-impact line added to the threshold recommendation.
  • BRAIN_ANALYSIS_WINDOW_DAYS filter added so old regime data (different
    weights/config) doesn't get silently mixed with current data.
  • Telegram message length capped with a "(N more items)" footer.
  • Run counter is now rolled back (not left "spent") if report generation
    raises, so the next run retries the same slot instead of waiting a
    full BRAIN_REPORT_INTERVAL_RUNS.
  • Report is persisted to Redis BEFORE the Telegram send attempt (not
    after), and the send is wrapped in try/except — a Telegram failure no
    longer loses the report.
  • maybe_generate_report() now checks ENABLE_WIN_RATE_FILTER and
    DRY_RUN_MODE itself, so whichever call site eventually invokes it gets
    the guard for free.
  • star_wr is now configurable (BRAIN_STAR_ALERT_WR, default 0.70).
  • Basic (pair, alert_key, entry_ts) de-dup on stream rows, since
    maxlen=... approximate=True xadd can occasionally double-write on retry.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from alerts import escape_markdown_v2

from bot_config import cfg, json_dumps, format_ist_time
from state import RedisKeyPrefix, RedisStateStore


_ALERT_CONFIG_MAP = {
    "strong_reversal_buy":  "ENABLE_STRONG_REVERSAL_ALERT",
    "strong_reversal_sell": "ENABLE_STRONG_REVERSAL_ALERT",
    "choch_buy":            "ENABLE_CHOCH_ALERT",
    "choch_sell":           "ENABLE_CHOCH_ALERT",
    "tlr_buy":              "ENABLE_TLR_ALERT",
    "tlr_sell":             "ENABLE_TLR_ALERT",
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


# ──────────────────────────────────────────────────────────────────────
# Statistical helpers (ported from analyze_confluence_thresholds.py so
# both tools agree on the same numbers for the same data)
# ──────────────────────────────────────────────────────────────────────

def wilson_ci(wins: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    """Wilson score interval — reliable even for small n."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, centre - margin), min(1.0, centre + margin), p


def favourable_move(row: Dict[str, Any]) -> float:
    """pct_move is signed by price direction, not by trade outcome — a
    winning sell has a negative pct_move. Always take the magnitude of the
    move that was favourable to the position."""
    return abs(row.get("pct_move", 0.0))


def expected_value(wins: int, losses: int, avg_win_pct: float, avg_loss_pct: float) -> float:
    n = wins + losses
    if n == 0:
        return 0.0
    wr = wins / n
    return wr * avg_win_pct - (1 - wr) * abs(avg_loss_pct)


def rr_ratio(avg_win_pct: float, avg_loss_pct: float) -> Optional[float]:
    if not avg_loss_pct or abs(avg_loss_pct) < 1e-6:
        return None
    return avg_win_pct / abs(avg_loss_pct)


def format_rr(rr: Optional[float]) -> str:
    return f"{rr:.2f}" if rr is not None else "n/a"


def ev_and_rr_for(rows: List[Dict[str, Any]]) -> Tuple[float, Optional[float], float, float]:
    """Returns (ev, rr, avg_win_magnitude, avg_loss_magnitude) for a row subset."""
    wins = [favourable_move(r) for r in rows if r["win"]]
    losses = [favourable_move(r) for r in rows if not r["win"]]
    avg_w = sum(wins) / len(wins) if wins else 0.0
    avg_l = sum(losses) / len(losses) if losses else 0.0
    ev = expected_value(len(wins), len(losses), avg_w, avg_l)
    rr = rr_ratio(avg_w, avg_l)
    return ev, rr, avg_w, avg_l


def detect_temporal_drift(rows: List[Dict[str, Any]], window_days: int = 14):
    """Compare win rate of recent outcomes vs older ones. Uses wall-clock
    time as "now" — NOT the last trade's timestamp, which would be wrong
    if the bot had downtime (a stale "recent" window silently shifts back
    in time)."""
    if not rows:
        return None, None, None
    now_ts = int(time.time())
    cutoff = now_ts - (window_days * 86400)
    recent = [r for r in rows if r["entry_ts"] >= cutoff]
    older = [r for r in rows if r["entry_ts"] < cutoff]
    if len(recent) < 10 or len(older) < 10:
        return None, None, None
    recent_wr = sum(r["win"] for r in recent) / len(recent)
    older_wr = sum(r["win"] for r in older) / len(older)
    return recent_wr, older_wr, len(recent)


def per_pair_breakdown(rows: List[Dict[str, Any]], min_sample: int = 10):
    stats = defaultdict(lambda: {"wins": 0, "n": 0})
    for r in rows:
        s = stats[r["pair"]]
        s["wins"] += r["win"]
        s["n"] += 1
    results = []
    for pair, s in stats.items():
        if s["n"] < min_sample:
            continue
        results.append((pair, s["wins"] / s["n"], s["n"]))
    results.sort(key=lambda x: x[1])
    return results


def vote_importance(rows: List[Dict[str, Any]], min_sample: int = 10):
    """For each vote, win rate when it's True vs when it's False.
    Sorted by lift (wr_with - wr_without), descending."""
    vote_names = set()
    for r in rows:
        if r.get("votes"):
            vote_names.update(r["votes"].keys())
    results = []
    for vn in sorted(vote_names):
        with_vote = [r for r in rows if r.get("votes") and r["votes"].get(vn) is True]
        without_vote = [r for r in rows if r.get("votes") and r["votes"].get(vn) is False]
        if len(with_vote) < min_sample or len(without_vote) < min_sample:
            continue
        wr_with = sum(r["win"] for r in with_vote) / len(with_vote)
        wr_without = sum(r["win"] for r in without_vote) / len(without_vote)
        results.append((vn, wr_with, len(with_vote), wr_without, len(without_vote), wr_with - wr_without))
    results.sort(key=lambda x: -x[5])
    return results


class BrainEngine:
    """Analysis layer over the bot's existing win-rate infrastructure."""

    def __init__(self, sdb: RedisStateStore):
        self.sdb = sdb

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

        # Cooldown: at most one override per alert_key per window, so a
        # volatile market can't flood Telegram with repeated overrides for
        # the same alert type.
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
                    "votes": votes,
                })
            except (KeyError, ValueError) as e:
                logging.getLogger("macd_bot").debug(f"Brain: dropping malformed outcome row: {e}")
                continue
        return parsed

    # ── Recommendations ──────────────────────────────────────────────────────

    async def generate_recommendations(self) -> Dict[str, Any]:
        """Build the full recommendation set: per-alert verdicts, a confluence
        threshold suggestion, shadow-mode insight, and a machine-readable
        config patch."""
        sample_size = getattr(cfg, "BRAIN_REPORT_STREAM_SAMPLE", 5000)
        window_days = getattr(cfg, "BRAIN_ANALYSIS_WINDOW_DAYS", 30)

        real_raw, shadow_raw = await asyncio.gather(
            self._read_stream(RedisKeyPrefix.OUTCOME_LOG_STREAM, sample_size),
            self._read_stream(RedisKeyPrefix.SHADOW_LOG_STREAM, sample_size),
        )
        real_rows = self._parse_rows(real_raw, window_days=window_days)
        shadow_rows = self._parse_rows(shadow_raw, window_days=window_days)

        recommendations: List[Dict[str, Any]] = []
        config_patch: List[Dict[str, Any]] = []
        seen_paths = set()

        min_sample = getattr(cfg, "MIN_WIN_RATE_SAMPLE", 20)
        target_wr = cfg.MIN_WIN_RATE
        disable_wr = getattr(cfg, "BRAIN_ALERT_DISABLE_THRESHOLD_WR", 0.40)
        star_wr = getattr(cfg, "BRAIN_STAR_ALERT_WR", 0.70)

        # ── Per-alert win rate (pooled across pairs), Wilson-bound verdicts ──
        alert_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"wins": 0, "losses": 0, "pairs": set()})
        for r in real_rows:
            s = alert_stats[r["alert_key"]]
            s["wins" if r["win"] else "losses"] += 1
            s["pairs"].add(r["pair"])

        alert_verdicts: Dict[str, str] = {}  # alert_key -> "disable" | "star" | "monitor"
        for alert_key, s in alert_stats.items():
            total = s["wins"] + s["losses"]
            if total < min_sample:
                continue
            wr = s["wins"] / total
            lo, hi, _ = wilson_ci(s["wins"], total)
            if hi < disable_wr:
                # Even the OPTIMISTIC bound is below the disable line — not noise.
                alert_verdicts[alert_key] = "disable"
                recommendations.append({
                    "type": "disable_alert", "severity": "high", "alert": alert_key,
                    "win_rate": round(wr, 3), "sample_size": total, "pairs_affected": len(s["pairs"]),
                    "message": (
                        f"DISABLE {alert_key}: {wr:.0%} WR over {total} samples across "
                        f"{len(s['pairs'])} pairs (95% CI upper bound {hi:.0%}, still below {disable_wr:.0%})."
                    ),
                })
            elif lo >= star_wr:
                # Even the PESSIMISTIC bound clears the star line.
                alert_verdicts[alert_key] = "star"
                recommendations.append({
                    "type": "star_alert", "severity": "low", "alert": alert_key,
                    "win_rate": round(wr, 3), "sample_size": total,
                    "message": (
                        f"STRONG: {alert_key} at {wr:.0%} WR over {total} samples "
                        f"(95% CI lower bound {lo:.0%}). Consider lowering its confluence bar."
                    ),
                })
            else:
                alert_verdicts[alert_key] = "monitor"
                recommendations.append({
                    "type": "monitor", "severity": "medium", "alert": alert_key,
                    "win_rate": round(wr, 3), "sample_size": total,
                    "message": f"{alert_key} viable ({wr:.0%} WR, {total} samples).",
                })

        # ── Group disable candidates by shared config path — only disable if
        #    every alert_key mapped to that path is bad (fixes the buy/sell
        #    collapse bug: ppo_signal_up bad + ppo_signal_down good must NOT
        #    disable both). ──
        path_to_keys: Dict[str, List[str]] = defaultdict(list)
        for alert_key in alert_stats:
            path = _resolve_config_path(alert_key)
            if path:
                path_to_keys[path].append(alert_key)

        for path, keys in path_to_keys.items():
            verdicts = {k: alert_verdicts.get(k) for k in keys if k in alert_verdicts}
            if not verdicts:
                continue
            if all(v == "disable" for v in verdicts.values()) and len(verdicts) == len([k for k in keys if k in alert_stats]):
                if path not in seen_paths:
                    seen_paths.add(path)
                    config_patch.append({
                        "path": path, "current": True, "suggested": False,
                        "reason": f"All alert types on this config path are underperforming: {', '.join(keys)}",
                    })
            elif "disable" in verdicts.values() and not all(v == "disable" for v in verdicts.values()):
                bad = [k for k, v in verdicts.items() if v == "disable"]
                good = [k for k, v in verdicts.items() if v != "disable"]
                recommendations.append({
                    "type": "investigate", "severity": "medium",
                    "message": (
                        f"{path} is shared by {', '.join(keys)} — {', '.join(bad)} underperforming but "
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

        # ── Confluence threshold suggestion — raw score space, Wilson lower
        #    bound, same algorithm as analyze_confluence_thresholds.py ──
        threshold_rec: Dict[str, Any] = {}
        if real_rows:
            candidate_caps = sorted(set(r["score"] for r in real_rows))
            caps_data = []
            for cap in candidate_caps:
                subset = [r for r in real_rows if r["score"] >= cap]
                if len(subset) < min_sample:
                    continue
                wins_count = sum(r["win"] for r in subset)
                wr = wins_count / len(subset)
                lo, hi, _ = wilson_ci(wins_count, len(subset))
                caps_data.append((cap, len(subset), wr, lo))

            target_floor = None
            for cap, n_pass, wr, wr_lo in caps_data:
                if wr_lo >= target_wr:
                    target_floor = cap
                    break
            if target_floor is None:
                # Fall back to raw-WR criterion on thin datasets, same as the analyzer.
                for cap, n_pass, wr, wr_lo in caps_data:
                    if wr >= target_wr:
                        target_floor = cap
                        break

            if target_floor is not None and abs(target_floor - cfg.CONFLUENCE_MIN_ABS_SCORE) >= 0.5:
                rec_subset = [r for r in real_rows if r["score"] >= target_floor]
                rec_n = len(rec_subset)
                rec_wr = sum(r["win"] for r in rec_subset) / rec_n
                ev, rr, avg_w, avg_l = ev_and_rr_for(rec_subset)

                buys = [r for r in rec_subset if r["direction"] == "buy"]
                sells = [r for r in rec_subset if r["direction"] == "sell"]
                buy_wr = sum(r["win"] for r in buys) / len(buys) if buys else None
                sell_wr = sum(r["win"] for r in sells) / len(sells) if sells else None

                total_alerts = len(real_rows)
                dropped = total_alerts - rec_n
                ts_list = [r["entry_ts"] for r in real_rows if r["entry_ts"]]
                weeks = (max(ts_list) - min(ts_list)) / (7 * 86400) if len(ts_list) > 1 else 0.1
                freq_before = total_alerts / max(weeks, 0.1)
                freq_after = rec_n / max(weeks, 0.1)

                direction_note = ""
                if buy_wr is not None and sell_wr is not None:
                    direction_note = f" | Buy WR {buy_wr:.0%} ({len(buys)}), Sell WR {sell_wr:.0%} ({len(sells)})"

                threshold_rec = {
                    "type": "confluence_threshold", "severity": "high",
                    "current_abs_score": cfg.CONFLUENCE_MIN_ABS_SCORE,
                    "suggested_abs_score": target_floor,
                    "supporting_samples": rec_n, "resulting_wr": round(rec_wr, 3),
                    "ev": round(ev, 4), "rr": rr,
                    "message": (
                        f"Set CONFLUENCE_MIN_ABS_SCORE to {target_floor:.1f} "
                        f"(currently {cfg.CONFLUENCE_MIN_ABS_SCORE:.1f}) for {rec_wr:.0%} WR, "
                        f"EV {ev:+.3f}%/trade, R:R {format_rr(rr)} across {rec_n} trades.{direction_note}\n"
                        f"Alert frequency: {freq_before:.1f}/wk -> {freq_after:.1f}/wk "
                        f"(dropping {dropped}, {dropped/total_alerts:.0%})."
                    ),
                }
                recommendations.append(threshold_rec)
                # Emit BOTH keys — the live gate takes max(pct_floor, ABS_SCORE),
                # so a percentage-only patch can be a complete no-op if the abs
                # floor is the binding constraint.
                config_patch.append({
                    "path": "CONFLUENCE_MIN_ABS_SCORE", "current": cfg.CONFLUENCE_MIN_ABS_SCORE,
                    "suggested": target_floor, "supporting_samples": rec_n,
                })
                avg_total = sum(r["total"] for r in rec_subset) / rec_n
                suggested_pct = min(100.0, (target_floor / avg_total) * 100.0) if avg_total else cfg.CONFLUENCE_MIN_PCT
                config_patch.append({
                    "path": "CONFLUENCE_MIN_PCT", "current": cfg.CONFLUENCE_MIN_PCT,
                    "suggested": round(suggested_pct, 1), "supporting_samples": rec_n,
                    "note": "Derived from suggested abs score / avg total this window — informational, "
                            "the abs score patch above is the one that reliably binds.",
                })

        # ── Temporal drift ──
        recent_wr, older_wr, recent_n = detect_temporal_drift(real_rows)
        if recent_wr is not None:
            drift = recent_wr - older_wr
            if drift < -0.05:
                recommendations.append({
                    "type": "temporal_drift", "severity": "high",
                    "message": (
                        f"Edge may be decaying: last 14d WR {recent_wr:.0%} ({recent_n} samples) "
                        f"vs prior WR {older_wr:.0%} (Δ{drift:+.0%})."
                    ),
                })

        # ── Per-pair breakdown (worst pairs only, keeps report short) ──
        pair_stats = per_pair_breakdown(real_rows, min_sample=min_sample)
        weak_pairs = [p for p in pair_stats if p[1] < disable_wr]
        if weak_pairs:
            recommendations.append({
                "type": "weak_pairs", "severity": "medium",
                "message": (
                    "Underperforming pairs: " +
                    ", ".join(f"{p}({wr:.0%}, n={n})" for p, wr, n in weak_pairs[:5])
                ),
            })

        # ── Vote importance (top lift / top drag only) ──
        vote_imp = vote_importance(real_rows, min_sample=min_sample)
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
    def _truncate_telegram(lines: List[str], limit: int = 3500) -> str:
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
        return truncated + f"\n\n… (truncated, {cut_chars} more characters)"

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

        logger_run.info("🧠 Brain generating analysis report...")
        try:
            recs = await self.generate_recommendations()
        except Exception as e:
            logger_run.error(f"Brain report generation failed: {e}")
            await self._rollback_run_count()  # no argument
            return

        cc = recs.get("current_config", {})
        lines = [
            "🧠 *BRAIN ANALYSIS REPORT*",
            f"Generated: {escape_markdown_v2(format_ist_time())}",
            f"Pairs monitored: {len(pairs)}",
            escape_markdown_v2(
                f"Real trades sampled: {recs['real_sample_size']} | Shadow-tracked: {recs['shadow_sample_size']}"
            ),
            escape_markdown_v2(
                f"Current: CONFLUENCE_MIN_ABS_SCORE={cc.get('CONFLUENCE_MIN_ABS_SCORE')} | "
                f"CONFLUENCE_MIN_PCT={cc.get('CONFLUENCE_MIN_PCT')}"
            ),
            "",
        ]
        high = [r for r in recs["recommendations"] if r["severity"] == "high"]
        med = [r for r in recs["recommendations"] if r["severity"] == "medium"]
        low = [r for r in recs["recommendations"] if r["severity"] == "low"]

        if high:
            lines.append("*🔴 HIGH PRIORITY*")
            for r in high[:6]:
                lines.append(f"• {escape_markdown_v2(r['message'])}")
            lines.append("")
        if med:
            lines.append("*🟡 WATCH*")
            for r in med[:6]:
                lines.append(f"• {escape_markdown_v2(r['message'])}")
            lines.append("")
        if low:
            lines.append("*🟢 STRONG PERFORMERS*")
            for r in low[:6]:
                lines.append(f"• {escape_markdown_v2(r['message'])}")
            lines.append("")

        ss = recs.get("shadow_summary") or {}
        if ss.get("total_tracked"):
            lines.append("*👻 SHADOW MODE*")
            if ss.get("overall_wr") is not None:
                lines.append(
                    escape_markdown_v2(
                        f"{ss['total_tracked']} rejected alert(s) tracked | overall WR {ss['overall_wr']:.0%}"
                    )
                )
            else:
                lines.append(escape_markdown_v2(f"{ss['total_tracked']} rejected alert(s) tracked"))
            if ss.get("high_confluence_wr") is not None:
                lines.append(
                    escape_markdown_v2(
                        f"High-confluence rejections ({ss['high_confluence_tracked']}): "
                        f"{ss['high_confluence_wr']:.0%} WR"
                    )
                )
            lines.append("")

        if recs["config_patch"]:
            lines.append(f"*⚙️ SUGGESTED {escape_markdown_v2('config_macd.json')} PATCH*")
            lines.append("```")
            lines.append(json_dumps(recs["config_patch"]))
            lines.append("```")

        total_recs = recs["recommendation_count"]
        shown = len(high[:6]) + len(med[:6]) + len(low[:6])
        if total_recs == 0:
            lines.append("No actionable signal yet — still accumulating samples.")
        elif total_recs > shown:
            lines.append(f"… ({total_recs - shown} more items not shown)")

        lines.append("")
        lines.append(
            escape_markdown_v2(
                "Stability: run weekly. If the recommended threshold moves by more than "
                "±2.0 points between runs, the dataset is still too thin to trust a single number."
            )
        )

        msg = self._truncate_telegram(lines)

        # Persist BEFORE attempting the Telegram send, so a send failure
        # doesn't also lose the report.
        report_key = f"brain_report:{int(time.time())}"
        if self.sdb._redis and not self.sdb.degraded:
            result = await self.sdb._safe_redis_op(
                lambda: self.sdb._redis.set(report_key, json_dumps(recs), ex=30 * 86400),
                2.0, f"brain_report_persist:{report_key}",
            )
            if result is None:
                logger_run.warning(f"Failed to persist brain report {report_key}")

        # FIX #3: only log "sent" if Telegram actually succeeded
        send_ok = False
        if telegram_queue:
            try:
                await telegram_queue.send(msg)
                send_ok = True
            except Exception as e:
                logger_run.warning(
                    f"Brain report Telegram send failed ({e}) — report is still persisted at {report_key}."
                )

        logger_run.info(
            f"🧠 Brain report {'sent' if send_ok else 'persisted (send failed)'} | "
            f"{len(high)} high, {len(med)} medium, {len(low)} low priority items"
        )