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
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

from bot_config import cfg, json_dumps, format_ist_time
from state import RedisKeyPrefix, RedisStateStore
from alerts import escape_markdown_v2

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
}

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
        that bucket across all pairs."""
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

        wins = int(data.get("wins", 0))
        losses = int(data.get("losses", 0))
        total = wins + losses
        if total < cfg.BRAIN_REWARDABLE_MIN_SHADOW_SAMPLE:
            return None

        wr = wins / total
        if wr < cfg.BRAIN_REWARDABLE_MIN_SHADOW_WR:
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
    def _parse_rows(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        parsed = []
        for f in rows:
            try:
                score = float(f["score"])
                total = float(f["total"])
                if total <= 0:
                    continue
                parsed.append({
                    "pair": f.get("pair", "?"),
                    "alert_key": f.get("alert_key", "?"),
                    "conf_pct": score / total * 100.0,
                    "win": f.get("win") == "1",
                    "pct_move": float(f.get("pct_move", 0.0)),
                })
            except (KeyError, ValueError):
                continue
        return parsed

    # ── Recommendations ──────────────────────────────────────────────────────

    async def generate_recommendations(self) -> Dict[str, Any]:
        """Build the full recommendation set: per-alert verdicts, a confluence
        threshold suggestion, shadow-mode insight, and a machine-readable
        config patch."""
        sample_size = getattr(cfg, "BRAIN_REPORT_STREAM_SAMPLE", 5000)
        real_rows = self._parse_rows(await self._read_stream(RedisKeyPrefix.OUTCOME_LOG_STREAM, sample_size))
        shadow_rows = self._parse_rows(await self._read_stream(RedisKeyPrefix.SHADOW_LOG_STREAM, sample_size))

        recommendations: List[Dict[str, Any]] = []
        config_patch: List[Dict[str, Any]] = []

        # ── Per-alert win rate (pooled across pairs) ──
        alert_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"wins": 0, "losses": 0, "pairs": set()})
        for r in real_rows:
            s = alert_stats[r["alert_key"]]
            s["wins" if r["win"] else "losses"] += 1
            s["pairs"].add(r["pair"])

        disable_wr = getattr(cfg, "BRAIN_ALERT_DISABLE_THRESHOLD_WR", 0.40)
        star_wr = 0.70
        min_sample = getattr(cfg, "MIN_WIN_RATE_SAMPLE", 20)

        for alert_key, s in alert_stats.items():
            total = s["wins"] + s["losses"]
            if total < min_sample:
                continue
            wr = s["wins"] / total
            if wr < disable_wr:
                recommendations.append({
                    "type": "disable_alert", "severity": "high", "alert": alert_key,
                    "win_rate": round(wr, 3), "sample_size": total, "pairs_affected": len(s["pairs"]),
                    "message": f"DISABLE {alert_key}: {wr:.0%} WR over {total} samples across {len(s['pairs'])} pairs.",
                })
            elif wr >= star_wr:
                recommendations.append({
                    "type": "star_alert", "severity": "low", "alert": alert_key,
                    "win_rate": round(wr, 3), "sample_size": total,
                    "message": f"STRONG: {alert_key} at {wr:.0%} WR over {total} samples. Consider lowering its confluence bar.",
                })
            else:
                recommendations.append({
                    "type": "monitor", "severity": "medium", "alert": alert_key,
                    "win_rate": round(wr, 3), "sample_size": total,
                    "message": f"{alert_key} viable ({wr:.0%} WR, {total} samples).",
                })

        # ── Confluence threshold suggestion ──
        if real_rows:
            bucket_w = getattr(cfg, "BRAIN_CONFLUENCE_BUCKET_PCT", 10.0)
            best_threshold = None
            best_n = 0
            best_wr = 0.0
            threshold = 30.0
            while threshold <= 90.0:
                bucket = [r for r in real_rows if r["conf_pct"] >= threshold]
                if len(bucket) >= min_sample:
                    wr = sum(1 for r in bucket if r["win"]) / len(bucket)
                    if wr >= cfg.MIN_WIN_RATE and len(bucket) > best_n:
                        best_n, best_threshold, best_wr = len(bucket), threshold, wr
                threshold += bucket_w

            if best_threshold is not None and abs(best_threshold - cfg.CONFLUENCE_MIN_PCT) >= bucket_w:
                recommendations.append({
                    "type": "confluence_threshold", "severity": "high",
                    "current_min_pct": cfg.CONFLUENCE_MIN_PCT, "suggested_min_pct": best_threshold,
                    "supporting_samples": best_n, "resulting_wr": round(best_wr, 3),
                    "message": (
                        f"Set CONFLUENCE_MIN_PCT to {best_threshold:.0f}% "
                        f"(currently {cfg.CONFLUENCE_MIN_PCT:.0f}%) for {best_wr:.0%} WR "
                        f"across {best_n} qualifying trades."
                    ),
                })
                config_patch.append({
                    "path": "CONFLUENCE_MIN_PCT", "current": cfg.CONFLUENCE_MIN_PCT,
                    "suggested": best_threshold, "supporting_samples": best_n,
                })

        for r in recommendations:
            if r["type"] == "disable_alert":
                cfg_path = _ALERT_CONFIG_MAP.get(r["alert"])
                if cfg_path:
                    config_patch.append({
                        "path": cfg_path, "current": True,
                        "suggested": False, "reason": r["message"],
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

    async def maybe_generate_report(
        self,
        pairs: List[str],
        telegram_queue: Any,
        logger_run: logging.Logger,
    ) -> None:
        """Increment the persisted run counter; if the report interval has
        elapsed, generate and send the analysis report."""
        run_count = await self._next_run_count()
        interval = getattr(cfg, "BRAIN_REPORT_INTERVAL_RUNS", 48)
        if run_count is None or run_count % interval != 0:
            return

        logger_run.info("🧠 Brain generating analysis report...")
        recs = await self.generate_recommendations()

        lines = [
            "🧠 *BRAIN ANALYSIS REPORT*",
            f"Generated: {format_ist_time()}",
            f"Pairs monitored: {len(pairs)}",
            f"Real trades sampled: {recs['real_sample_size']} | Shadow-tracked: {recs['shadow_sample_size']}",
            "",
        ]

        high = [r for r in recs["recommendations"] if r["severity"] == "high"]
        med = [r for r in recs["recommendations"] if r["severity"] == "medium"]
        low = [r for r in recs["recommendations"] if r["severity"] == "low"]

        if high:
            lines.append("*🔴 HIGH PRIORITY*")
            for r in high[:6]:
                lines.append(f"• {r['message']}")
            lines.append("")
        if med:
            lines.append("*🟡 WATCH*")
            for r in med[:6]:
                lines.append(f"• {r['message']}")
            lines.append("")
        if low:
            lines.append("*🟢 STRONG PERFORMERS*")
            for r in low[:6]:
                lines.append(f"• {r['message']}")
            lines.append("")

        ss = recs.get("shadow_summary") or {}
        if ss.get("total_tracked"):
            lines.append("*👻 SHADOW MODE*")
            if ss.get("overall_wr") is not None:
                lines.append(f"{ss['total_tracked']} rejected alert(s) tracked | overall WR {ss['overall_wr']:.0%}")
            else:
                lines.append(f"{ss['total_tracked']} rejected alert(s) tracked")
            if ss.get("high_confluence_wr") is not None:
                lines.append(
                    f"High-confluence rejections ({ss['high_confluence_tracked']}): "
                    f"{ss['high_confluence_wr']:.0%} WR"
                )
            lines.append("")

        if recs["config_patch"]:
            lines.append("*⚙️ SUGGESTED config_macd.json PATCH*")
            lines.append("```")
            lines.append(json_dumps(recs["config_patch"]))
            lines.append("```")

        if recs["recommendation_count"] == 0:
            lines.append("No actionable signal yet — still accumulating samples.")

        # Build lines with dynamic content escaped, formatting preserved
        lines = [
            "🧠 *BRAIN ANALYSIS REPORT*",
            f"Generated: {format_ist_time()}",
            f"Pairs monitored: {len(pairs)}",
            f"Real trades sampled: {recs['real_sample_size']} | Shadow-tracked: {recs['shadow_sample_size']}",
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
                    f"{ss['total_tracked']} rejected alert(s) tracked | overall WR {ss['overall_wr']:.0%}"
                )
            else:
                lines.append(f"{ss['total_tracked']} rejected alert(s) tracked")
            if ss.get("high_confluence_wr") is not None:
                lines.append(
                    f"High-confluence rejections ({ss['high_confluence_tracked']}): "
                    f"{ss['high_confluence_wr']:.0%} WR"
                )
            lines.append("")

        if recs["config_patch"]:
            lines.append("*⚙️ SUGGESTED config_macd.json PATCH*")
            lines.append("```")
            lines.append(json_dumps(recs["config_patch"]))
            lines.append("```")

        if recs["recommendation_count"] == 0:
            lines.append("No actionable signal yet — still accumulating samples.")

        msg = "\n".join(lines)
        if telegram_queue:
            await telegram_queue.send(msg)

        # Persist full report for external tooling / audit
        if self.sdb._redis and not self.sdb.degraded:
            try:
                report_key = f"brain_report:{int(time.time())}"
                await self.sdb._redis.set(report_key, json_dumps(recs), ex=30 * 86400)
            except Exception as e:
                logger_run.warning(f"Failed to persist brain report: {e}")

        logger_run.info(
            f"🧠 Brain report sent | {len(high)} high, {len(med)} medium, {len(low)} low priority items"
        )
