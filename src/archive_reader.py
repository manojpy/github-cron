#!/usr/bin/env python3
"""archive_reader.py — Read archived JSONL outcomes for Brain reports."""
from __future__ import annotations
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Single source of truth for "current schema" — importing from the writer
# guarantees reader and writer can never drift.
from outcome_storage import OUTCOME_SCHEMA_VERSION as CURRENT_SCHEMA_VERSION

_log = logging.getLogger("macd_bot")

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

def _parse_jsonl_row(raw: dict, *, drop_stale_schema: bool = True) -> Optional[dict]:
    """Convert archived JSONL row → Brain _parse_rows format.

    Filters out two classes of row that would silently corrupt
    downstream statistics:

      1. Signal-only rows — records written by alerts.py at alert-
         dispatch time that carry no `win` field. These describe that
         a signal fired, not how it resolved. If allowed through,
         `_coerce_bool(raw.get("win"), default=False)` reads them as
         losses and inflates every loss-based metric (WR, MAE rate,
         CUSUM s_neg, disable-alert verdicts).

      2. Stale-schema rows — records written before the current
         OUTCOME_SCHEMA_VERSION. On these, `mfe_win`, `mae_loss`,
         `tp_first`, `bonus_win`, `rr_achieved`, and `win_weight` are
         absent. The reader's .get(..., None) defaults turn that
         absence into a false negative for every metric that counts
         `is True`, so a mixed-vintage window reports MFE WR of 1%
         when the true rate is 40%+, and drives McNemar's discordant
         b-cell from missing-field rows rather than genuine
         close-vs-MFE disagreements.

    drop_stale_schema defaults True. Set False only for archaeology
    scripts reading a fully historical window — never for a live brain
    report.
    """
    try:
        # ── Class 1: reject signal-only rows ──
        # `win` can legitimately be False on a resolved row, so we test
        # for *presence*, not truthiness. Presence of `win` is the
        # canonical marker of "this row describes an outcome."
        if "win" not in raw:
            return None

        # ── Class 2: reject rows written under an older schema ──
        # Missing schema_version means pre-versioning (implicit v1).
        row_schema = int(raw.get("schema_version", 1))
        if drop_stale_schema and row_schema != CURRENT_SCHEMA_VERSION:
            return None

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
            # ── Provenance — lets downstream consumers audit vintage ──
            # Useful for debugging future migrations: if a metric ever
            # looks wrong again, group by this field first.
            "schema_version": row_schema,
        }
    except Exception:
        return None

def load_archived_outcomes(
    data_dir: str,
    window_days: int = 30,
    shadow: bool = False,
    *,
    drop_stale_schema: bool = True,
    return_stats: bool = False,
):
    """Load outcomes from archived JSONL files (newest first).

    Rows are filtered on three axes:
      a) entry_ts within the window,
      b) presence of the `win` field (drops pre-resolution signal rows),
      c) schema_version == CURRENT_SCHEMA_VERSION (drops mixed-vintage
         rows whose three-metric / R:R fields are absent).

    Filters (b) and (c) are essential. Without them, a mixed-vintage
    archive reports MFE WR and clean-win-rate that bear no relation to
    the underlying trade outcomes — the fields are simply missing on
    old rows, and `is True` checks count that missingness as a loss.

    return_stats: when True, returns (rows, stats) where stats carries
    per-cause drop counters. Default False preserves the legacy return
    type so existing callers keep working unchanged.
    """
    label = "shadow" if shadow else "outcomes"
    root = Path(data_dir) / label

    stats: Dict[str, int] = {
        "files_read": 0,
        "lines_total": 0,
        "lines_malformed": 0,
        "dropped_missing_win": 0,
        "dropped_stale_schema": 0,
        "dropped_before_window": 0,
        "dropped_duplicate_sid": 0,
        "kept": 0,
    }

    if not root.exists():
        return ([], stats) if return_stats else []

    cutoff = time.time() - (window_days * 86400)
    rows: List[Dict[str, Any]] = []
    seen_ids: set = set()

    # Filenames are YYYY-MM-DD.jsonl (per outcome_storage._today_file).
    # Reverse lexicographic = newest date first, which lets the mtime
    # short-circuit below actually save work.
    files = sorted(root.glob("*.jsonl"), reverse=True)

    for path in files:
        # Quick mtime check: if the whole file predates the window and
        # we already have plenty of rows, stop reading further files.
        try:
            if path.stat().st_mtime < cutoff and len(rows) > 5000:
                break
        except OSError:
            pass

        stats["files_read"] += 1
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    stats["lines_total"] += 1
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError:
                        stats["lines_malformed"] += 1
                        continue

                    # Deduplicate by stream ID if present (Redis-exported rows)
                    sid = raw.get("_stream_id")
                    if sid:
                        if sid in seen_ids:
                            stats["dropped_duplicate_sid"] += 1
                            continue
                        seen_ids.add(sid)

                    # Time filter
                    ts = int(raw.get("entry_ts", 0))
                    if ts < cutoff:
                        stats["dropped_before_window"] += 1
                        continue

                    # Pre-classify drop reasons so the counters are
                    # attributable. _parse_jsonl_row also filters these,
                    # but returns a single None with no cause, which
                    # makes debugging schema issues slow.
                    if "win" not in raw:
                        stats["dropped_missing_win"] += 1
                        continue
                    if drop_stale_schema:
                        row_schema = int(raw.get("schema_version", 1))
                        if row_schema != CURRENT_SCHEMA_VERSION:
                            stats["dropped_stale_schema"] += 1
                            continue

                    parsed = _parse_jsonl_row(raw, drop_stale_schema=drop_stale_schema)
                    if parsed:
                        rows.append(parsed)
                        stats["kept"] += 1
        except OSError:
            continue

    dropped_total = (
        stats["dropped_missing_win"]
        + stats["dropped_stale_schema"]
        + stats["dropped_before_window"]
        + stats["dropped_duplicate_sid"]
        + stats["lines_malformed"]
    )

    if dropped_total > 0:
        _log.info(
            f"📚 archive_reader ({label}): kept {stats['kept']} rows | "
            f"dropped signal-only={stats['dropped_missing_win']}, "
            f"stale-schema={stats['dropped_stale_schema']}, "
            f"out-of-window={stats['dropped_before_window']}, "
            f"dup-sid={stats['dropped_duplicate_sid']}, "
            f"malformed={stats['lines_malformed']}"
        )
    return (rows, stats) if return_stats else rows