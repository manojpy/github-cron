#!/usr/bin/env python3
"""archive_reader.py — Read archived JSONL outcomes for Brain reports."""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

def _coerce_bool(val, default=None):
    """Parse a possibly-tri-state field that may arrive as a native bool
    (JSONL file archive), a '1'/'0' string (Redis stream convention),
    or a 1/0 int.

    NOTE: str(True) == 'True' != '1', so the old str(x) == '1' checks
    silently read every native-bool archive field as False.
    """
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return val != 0
    s = str(val).strip().lower()
    if s in ("1", "true", "yes", "y"):
        return True
    if s in ("0", "false", "no", "n"):
        return False
    return default

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

        # ── Three-metric fields (robust to bool OR "1"/"0" strings) ──
        base_win = _coerce_bool(raw.get("win"), default=False)
        close_win_val = _coerce_bool(raw.get("close_win"), default=base_win)
        mfe_win_val = _coerce_bool(raw.get("mfe_win"), default=None)
        mae_loss_val = _coerce_bool(raw.get("mae_loss"), default=None)
        tp_first_val = _coerce_bool(raw.get("tp_first"), default=None)

        # ── R:R and Bonus fields (backward compatible with old archives) ──
        bonus_win_val = _coerce_bool(raw.get("bonus_win"), default=False)

        rr_achieved_raw = raw.get("rr_achieved")
        try:
            rr_achieved_val = (
                float(rr_achieved_raw)
                if rr_achieved_raw not in (None, "")
                else 0.0
            )
        except (TypeError, ValueError):
            rr_achieved_val = 0.0

        win_weight_raw = raw.get("win_weight")
        try:
            win_weight_val = (
                float(win_weight_raw)
                if win_weight_raw not in (None, "")
                else (1.0 if base_win else 0.0)
            )
        except (TypeError, ValueError):
            win_weight_val = 1.0 if base_win else 0.0

        return {
            "pair": raw.get("pair", "?"),
            "alert_key": raw.get("alert_key", "?"),
            "direction": raw.get("direction", "?"),
            "score": score,
            "total": total,
            "conf_pct": score / total * 100.0,
            "win": base_win,
            "pct_move": float(raw.get("pct_move", 0.0)),
            "entry_ts": entry_ts,
            "session": raw.get("session", "unknown"),
            "mae": mae,
            "mfe": mfe,
            "votes": votes,
            "context": context,
            # ── Three-metric fields ──
            "close_win": close_win_val,
            "mfe_win": mfe_win_val,
            "mae_loss": mae_loss_val,
            "tp_first": tp_first_val,
            # ── R:R and Bonus fields ──
            "bonus_win": bonus_win_val,
            "rr_achieved": rr_achieved_val,
            "win_weight": win_weight_val,
            # ── Backward-compat fields ──
            "outcome_reason": raw.get("outcome_reason") or "legacy",
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
