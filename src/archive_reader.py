#!/usr/bin/env python3
"""archive_reader.py — Read archived JSONL outcomes for Brain reports."""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

def _parse_jsonl_row(raw: dict) -> Optional[dict]:
    """Convert archived JSONL row → Brain _parse_rows format."""
    try:
        entry_ts = int(raw.get("entry_ts", 0))
        if entry_ts <= 0:
            return None
        
        score = float(raw.get("score", 0))
        total = float(raw.get("total", 0))
        if total <= 0:
            return None

        votes = raw.get("votes")
        if isinstance(votes, str):
            try:
                votes = json.loads(votes)
            except Exception:
                votes = None
        
        context = raw.get("context")
        if isinstance(context, str):
            try:
                context = json.loads(context)
            except Exception:
                context = None

        mae = raw.get("mae")
        mfe = raw.get("mfe")
        try:
            mae = float(mae) if mae not in (None, "") else None
        except Exception:
            mae = None
        try:
            mfe = float(mfe) if mfe not in (None, "") else None
        except Exception:
            mfe = None

        has_new_fields = "mfe_win" in raw
        if has_new_fields:
            mfe_win = str(raw.get("mfe_win")) == "1" or raw.get("mfe_win") is True
            close_win = str(raw.get("close_win")) == "1" or raw.get("close_win") is True
            mae_loss = str(raw.get("mae_loss")) == "1" or raw.get("mae_loss") is True
            outcome_reason = raw.get("outcome_reason", "unknown")
            try:
                bonus_weight = float(raw.get("bonus_weight", 1.0))
            except Exception:
                bonus_weight = 1.0
        else:
            legacy_win = str(raw.get("win")) == "1" or raw.get("win") is True
            mfe_win = legacy_win
            close_win = legacy_win
            mae_loss = False
            outcome_reason = "legacy"
            bonus_weight = 1.0

        return {
            "pair": raw.get("pair", "?"),
            "alert_key": raw.get("alert_key", "?"),
            "direction": raw.get("direction", "?"),
            "score": score,
            "total": total,
            "conf_pct": score / total * 100.0,
            "win": mfe_win,
            "close_win": close_win,
            "mfe_win": mfe_win,
            "mae_loss": mae_loss,
            "outcome_reason": outcome_reason,
            "bonus_weight": bonus_weight,
            "pct_move": float(raw.get("pct_move", 0.0)),
            "entry_ts": entry_ts,
            "session": raw.get("session", "unknown"),
            "mae": mae,
            "mfe": mfe,
            "votes": votes,
            "context": context,
        }
    except Exception:
        return None

def load_archived_outcomes(
    data_dir: str,
    window_days: int = 30,
    shadow: bool = False,
) -> List[Dict[str, Any]]:
    """Load outcomes from archived JSONL files (newest first)."""
    label = "shadow" if shadow else "outcomes"
    root = Path(data_dir) / label
    if not root.exists():
        return []

    cutoff = time.time() - (window_days * 86400)
    rows: List[Dict[str, Any]] = []
    seen_ids: set = set()

    # Files are named YYYY-MM.jsonl — sort reverse to read newest months first
    files = sorted(root.glob("*.jsonl"), reverse=True)
    
    for path in files:
        # Quick mtime check: if file is entirely older than window and we have enough, stop
        try:
            if path.stat().st_mtime < cutoff and len(rows) > 5000:
                break
        except OSError:
            pass

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Deduplicate by stream ID if present
                sid = raw.get("_stream_id")
                if sid:
                    if sid in seen_ids:
                        continue
                    seen_ids.add(sid)

                # Time filter
                ts = int(raw.get("entry_ts", 0))
                if ts < cutoff:
                    continue

                parsed = _parse_jsonl_row(raw)
                if parsed:
                    rows.append(parsed)

    return rows
