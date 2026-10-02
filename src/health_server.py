"""Lightweight health / metrics endpoint for the MACD bot (stdlib only).

The bot runs as a cron job, so there is no long-lived bot process to attach an
HTTP server to. This module reads the structured run summary that every run
already writes (RUN_SUMMARY_PATH) and exposes it:

    python src/health_server.py --serve [--port 8080]   # /health  /metrics
    python src/health_server.py --check                 # prints JSON, exit 0/1

/health  -> 200 when healthy, 503 when the last run is too old, Redis is
            degraded, or no summary exists. JSON body with the details.
/metrics -> Prometheus text format.

Environment: RUN_SUMMARY_PATH (default /tmp/data-repo/run_summary.json),
HEALTH_MAX_AGE_SEC (default 1800), OUTCOME_DATA_DIR (optional), HEALTH_PORT.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Optional, Tuple

DEFAULT_SUMMARY_PATH = "/tmp/data-repo/run_summary.json"


def load_summary(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def outcome_data_status(outcome_dir: Optional[str], now: float) -> Dict[str, Any]:
    """Newest outcome file age, so a stalled outcome pipeline is visible."""
    if not outcome_dir or not os.path.isdir(outcome_dir):
        return {"configured": bool(outcome_dir), "newest_file_age_min": None}
    newest = 0.0
    for root, _dirs, files in os.walk(outcome_dir):
        for name in files:
            try:
                newest = max(newest, os.path.getmtime(os.path.join(root, name)))
            except OSError:
                continue
    return {
        "configured": True,
        "newest_file_age_min": round((now - newest) / 60, 1) if newest else None,
    }


def build_health(
    summary: Optional[Dict[str, Any]], now: float, max_age_sec: int,
    outcome_dir: Optional[str] = None,
) -> Tuple[int, Dict[str, Any]]:
    """Return (http_status, body)."""
    if summary is None:
        return 503, {"status": "no_summary", "reason": "run summary missing or unreadable"}
    last_ts = summary.get("timestamp")
    age = (now - float(last_ts)) if isinstance(last_ts, (int, float)) else None
    problems = []
    if age is None:
        problems.append("summary has no timestamp")
    elif age > max_age_sec:
        problems.append(f"last run {int(age)}s ago (> {max_age_sec}s)")
    if summary.get("redis_status") not in (None, "OK"):
        problems.append(f"redis {summary.get('redis_status')}")
    stale = summary.get("stale_candle_pairs") or []
    completed, configured = summary.get("pairs_completed"), summary.get("pairs_configured")
    body = {
        "status": "unhealthy" if problems else "ok",
        "problems": problems,
        "last_run_age_sec": None if age is None else int(age),
        "run_duration_sec": summary.get("duration_sec"),
        "pairs": {"configured": configured, "completed": completed,
                  "deferred": summary.get("pairs_deferred")},
        "alerts_sent": summary.get("alerts_sent"),
        "redis": {"status": summary.get("redis_status"), "mem_pct": summary.get("redis_mem_pct")},
        "telegram": summary.get("telegram"),
        "telegram_dlq": summary.get("telegram_dlq"),
        "dedup": summary.get("dedup"),
        "stale_candle_pairs": stale,
        "last_successful_candle": summary.get("last_successful_candle"),
        "outcome_data": outcome_data_status(outcome_dir, now),
    }
    return (503 if problems else 200), body


def render_metrics(summary: Optional[Dict[str, Any]], now: float) -> str:
    lines = []

    def gauge(name: str, value: Any, help_text: str, labels: str = "") -> None:
        if isinstance(value, bool):
            value = int(value)
        if isinstance(value, (int, float)):
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name}{labels} {value}")

    if summary is None:
        gauge("macd_bot_summary_present", 0, "1 if a run summary could be read")
        return "\n".join(lines) + "\n"
    gauge("macd_bot_summary_present", 1, "1 if a run summary could be read")
    ts = summary.get("timestamp")
    if isinstance(ts, (int, float)):
        gauge("macd_bot_last_run_age_seconds", int(now - ts), "Seconds since the last run summary")
    gauge("macd_bot_run_duration_seconds", summary.get("duration_sec"), "Last run duration")
    gauge("macd_bot_pairs_configured", summary.get("pairs_configured"), "Pairs configured")
    gauge("macd_bot_pairs_completed", summary.get("pairs_completed"), "Pairs completed last run")
    gauge("macd_bot_alerts_sent", summary.get("alerts_sent"), "Alerts sent last run")
    gauge("macd_bot_redis_ok", summary.get("redis_status") == "OK", "1 if Redis was healthy")
    gauge("macd_bot_redis_memory_pct", summary.get("redis_mem_pct"), "Redis memory percent")
    tg = summary.get("telegram") or {}
    gauge("macd_bot_telegram_sent_ok", tg.get("sent_ok"), "Telegram sends OK last run")
    gauge("macd_bot_telegram_sent_failed", tg.get("sent_failed"), "Telegram sends failed last run")
    dlq = summary.get("telegram_dlq") or {}
    gauge("macd_bot_telegram_dlq_pending", dlq.get("pending"), "Alerts parked in the Telegram DLQ")
    gauge("macd_bot_stale_candle_pairs", len(summary.get("stale_candle_pairs") or []),
          "Pairs with no fresh successful candle")
    return "\n".join(lines) + "\n"


def _config() -> Tuple[str, int, Optional[str]]:
    return (
        os.environ.get("RUN_SUMMARY_PATH", DEFAULT_SUMMARY_PATH),
        int(os.environ.get("HEALTH_MAX_AGE_SEC", "1800")),
        os.environ.get("OUTCOME_DATA_DIR") or None,
    )


class _Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: str, ctype: str) -> None:
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
        path, max_age, outcome_dir = _config()
        summary, now = load_summary(path), time.time()
        if self.path.split("?")[0] == "/health":
            code, body = build_health(summary, now, max_age, outcome_dir)
            self._send(code, json.dumps(body), "application/json")
        elif self.path.split("?")[0] == "/metrics":
            self._send(200, render_metrics(summary, now), "text/plain; version=0.0.4")
        else:
            self._send(404, "not found", "text/plain")

    def log_message(self, fmt: str, *args: Any) -> None:  # keep stdout quiet
        return


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description="MACD bot health endpoint")
    ap.add_argument("--serve", action="store_true", help="run the HTTP server")
    ap.add_argument("--check", action="store_true", help="print health JSON, exit 0/1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("HEALTH_PORT", "8080")))
    args = ap.parse_args(argv)
    if args.serve:
        HTTPServer(("0.0.0.0", args.port), _Handler).serve_forever()
        return 0
    path, max_age, outcome_dir = _config()
    code, body = build_health(load_summary(path), time.time(), max_age, outcome_dir)
    print(json.dumps(body, indent=2))
    return 0 if code == 200 else 1


if __name__ == "__main__":
    sys.exit(main())
