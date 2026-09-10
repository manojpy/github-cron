#!/usr/bin/env python3
"""File-based outcome storage for GitHub Actions persistence."""
from __future__ import annotations

import json
import os
import glob
import time
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from bot_config import cfg

# ── Outcome schema version ───────────────────────────────────────────

OUTCOME_SCHEMA_VERSION = 2

_OUTCOME_DIR = getattr(cfg, "OUTCOME_DATA_DIR", "outcome-data")
os.makedirs(os.path.join(_OUTCOME_DIR, "outcomes"), exist_ok=True)
os.makedirs(os.path.join(_OUTCOME_DIR, "shadow"), exist_ok=True)
os.makedirs(os.path.join(_OUTCOME_DIR, "reports"), exist_ok=True)

_write_locks: Dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()

def _get_lock(subdir: str) -> threading.Lock:
    with _locks_guard:
        if subdir not in _write_locks:
            _write_locks[subdir] = threading.Lock()
        return _write_locks[subdir]

def _today_file(subdir: str) -> str:
    d = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return os.path.join(_OUTCOME_DIR, subdir, f"{d}.jsonl")

def _stamp_schema(record: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallow copy of `record` with schema_version set.

    Never mutates the caller's dict — the same dict is often serialized
    elsewhere (Redis payload, log line), and stamping the version
    in-place would silently leak the JSONL schema concept into stores
    that don't use it.
    """
    stamped = dict(record)
    stamped["schema_version"] = OUTCOME_SCHEMA_VERSION
    return stamped

def append_outcome(record: Dict[str, Any], shadow: bool = False) -> None:
    """Append a single record to today's JSONL file.

    NOTE: this API is used by alerts.py to write a *pre-resolution*
    signal row that carries no `win` field — it is not an outcome, it
    only records that an alert fired. Those rows are filtered out at
    read time by archive_reader (based on presence of `win`, not on
    schema_version). The schema_version stamped here describes the
    envelope's shape; it does NOT imply the trade has resolved.
    """
    subdir = "shadow" if shadow else "outcomes"
    path = _today_file(subdir)
    lock = _get_lock(subdir)
    with lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(_stamp_schema(record), default=str) + "\n")

def append_outcome_batch(records: List[Dict[str, Any]], shadow: bool = False) -> None:
    """Append multiple resolved outcome records in one file open.
    Used by state.py at resolution time to write win/pct_move/mae/mfe."""
    if not records:
        return
    subdir = "shadow" if shadow else "outcomes"
    path = _today_file(subdir)
    lock = _get_lock(subdir)
    with lock:
        with open(path, "a", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(_stamp_schema(record), default=str) + "\n")

def load_recent_outcomes(days: int = 30, shadow: bool = False,
                          hours: Optional[int] = None) -> List[Dict[str, Any]]:
    """Read last N days (or N hours) of outcome lines (newest first)."""
    subdir = "shadow" if shadow else "outcomes"
    rows: List[Dict[str, Any]] = []
    if hours is not None:
        cutoff = time.time() - (hours * 3600)
    else:
        cutoff = time.time() - (days * 86400)
    pattern = os.path.join(_OUTCOME_DIR, subdir, "*.jsonl")
    files = sorted(glob.glob(pattern), reverse=True)
    for path in files:
        try:
            ftime = os.path.getmtime(path)
            if ftime < cutoff and len(rows) > 1000:
                break
        except OSError:
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if row.get("entry_ts", 0) >= cutoff:
                        rows.append(row)
        except Exception:
            continue
    return rows

def save_brain_state(state: Dict[str, Any]) -> None:
    path = os.path.join(_OUTCOME_DIR, "brain_state.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)

def load_brain_state() -> Optional[Dict[str, Any]]:
    path = os.path.join(_OUTCOME_DIR, "brain_state.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_report(markdown: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M")
    path = os.path.join(_OUTCOME_DIR, "reports", f"{ts}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(markdown)
    return path