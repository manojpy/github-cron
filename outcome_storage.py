#!/usr/bin/env python3
"""File-based outcome storage for GitHub Actions persistence."""
from __future__ import annotations
import json
import os
import glob
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from bot_config import cfg

_OUTCOME_DIR = os.environ.get("OUTCOME_DATA_DIR") or getattr(cfg, "OUTCOME_DATA_DIR", "outcome-data")
os.makedirs(os.path.join(_OUTCOME_DIR, "outcomes"), exist_ok=True)
os.makedirs(os.path.join(_OUTCOME_DIR, "shadow"), exist_ok=True)
os.makedirs(os.path.join(_OUTCOME_DIR, "reports"), exist_ok=True)

def _today_file(subdir: str) -> str:
    d = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return os.path.join(_OUTCOME_DIR, subdir, f"{d}.jsonl")

def append_outcome(record: Dict[str, Any], shadow: bool = False) -> None:
    """Append a single outcome record to today's JSONL file."""
    path = _today_file("shadow" if shadow else "outcomes")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")

def load_recent_outcomes(days: int = 30, shadow: bool = False) -> List[Dict[str, Any]]:
    """Read last N days of outcome lines (newest first)."""
    subdir = "shadow" if shadow else "outcomes"
    rows: List[Dict[str, Any]] = []
    cutoff = time.time() - (days * 86400)
    pattern = os.path.join(_OUTCOME_DIR, subdir, "*.jsonl")
    files = sorted(glob.glob(pattern), reverse=True)
    for path in files:
        # Stop if file is older than window
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
