#!/usr/bin/env python3
"""
export_outcomes.py — Durable archive for outcome_log_stream / shadow_log_stream

Redis is the live store, not a durable one: outcome_log_stream and
shadow_log_stream are both XADD'd with maxlen=50000 (approximate) and no
TTL, so old entries silently roll off once you cross ~50k resolved
outcomes. This script is meant to run on a schedule (see
export-outcomes.yml) and appends *only the new* entries since last run
into month-partitioned .jsonl files under a git-tracked directory, so
history survives stream trimming, TTL expiry, provider evictions, or a
free-tier cap being hit.

Resumability: XRANGE's exclusive-start syntax "(<id>" is used against a
small marker file (.state.json) that records the last stream ID exported
per stream, so reruns never duplicate or skip entries under normal
operation. If the job dies between writing a batch and updating the
marker, the next run may re-export that batch — harmless for analysis
use (duplicates are easy to dedupe on _stream_id if it ever matters),
so no distributed-lock complexity has been added for it.

Usage:
    REDIS_URL=... python3 export_outcomes.py [--data-dir src/data/outcomes]
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

try:
    import redis
except ImportError:
    sys.exit("Missing dependency: pip install redis")

STREAMS = {
    "outcome": "outcome_log_stream",
    "shadow": "shadow_log_stream",
}
BATCH_SIZE = 1000


def load_state(state_path: Path) -> dict:
    if not state_path.exists():
        return {}
    try:
        return json.loads(state_path.read_text())
    except (json.JSONDecodeError, OSError):
        # Corrupt or empty marker file — safest fallback is to re-export
        # everything from the start of each stream rather than crash.
        print(f"⚠️  Could not parse {state_path}, re-exporting from stream start", file=sys.stderr)
        return {}


def save_state(state_path: Path, state: dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True))


def month_partition(entry_ts_raw: str) -> str:
    """Partition by the outcome's own entry_ts (epoch seconds), not export
    wall-clock time, so a delayed/backfilled export still lands in the
    right month file."""
    try:
        ts = float(entry_ts_raw)
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m")
    except (TypeError, ValueError):
        return "unknown"


def decode_row(stream_id: str, fields: dict) -> dict:
    row = dict(fields)
    row["_stream_id"] = stream_id
    # votes/context are stored as JSON strings in the stream; parse them
    # back to nested objects so the archive is directly usable, not
    # double-encoded.
    for key in ("votes", "context"):
        raw = row.get(key)
        if raw:
            try:
                row[key] = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                pass  # leave as raw string if it doesn't parse
    return row


def fetch_new_entries(r: "redis.Redis", stream_key: str, last_id: str):
    """Yields (stream_id, fields) for every entry after last_id, oldest
    first, paging through in batches of BATCH_SIZE."""
    cursor = f"({last_id}" if last_id else "-"
    while True:
        batch = r.xrange(stream_key, min=cursor, count=BATCH_SIZE)
        if not batch:
            return
        for entry_id, fields in batch:
            yield entry_id, fields
        if len(batch) < BATCH_SIZE:
            return
        cursor = f"({batch[-1][0]}"


def export_stream(r: "redis.Redis", label: str, stream_key: str,
                   data_dir: Path, state: dict) -> int:
    last_id = state.get(stream_key, "0")
    by_month = defaultdict(list)
    newest_id = last_id
    count = 0

    for entry_id, fields in fetch_new_entries(r, stream_key, last_id):
        row = decode_row(entry_id, fields)
        month = month_partition(row.get("entry_ts", ""))
        by_month[month].append(row)
        newest_id = entry_id
        count += 1

    if count == 0:
        print(f"  {label}: nothing new")
        return 0

    out_dir = data_dir / label
    out_dir.mkdir(parents=True, exist_ok=True)
    for month, rows in by_month.items():
        out_file = out_dir / f"{month}.jsonl"
        with out_file.open("a") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")

    state[stream_key] = newest_id
    print(f"  {label}: exported {count} new entries across {len(by_month)} month file(s)")
    return count


def main():
    ap = argparse.ArgumentParser(description="Archive Redis outcome streams to durable JSONL")
    ap.add_argument("--data-dir", default="src/data/outcomes",
                     help="Root directory for archived JSONL files (default: src/data/outcomes)")
    args = ap.parse_args()

    redis_url = os.environ.get("REDIS_URL")
    if not redis_url:
        sys.exit("Set REDIS_URL in your environment first.")

    r = redis.from_url(redis_url, decode_responses=True)
    data_dir = Path(args.data_dir)
    state_path = data_dir / ".state.json"
    state = load_state(state_path)

    print("======================================================================")
    print("  OUTCOME ARCHIVE EXPORT")
    print("======================================================================")

    total = 0
    for label, stream_key in STREAMS.items():
        total += export_stream(r, label, stream_key, data_dir, state)

    save_state(state_path, state)

    print("======================================================================")
    if total == 0:
        print("Nothing new to commit.")
        sys.exit(0)
    print(f"Total new rows archived: {total}")
    sys.exit(0)


if __name__ == "__main__":
    main()