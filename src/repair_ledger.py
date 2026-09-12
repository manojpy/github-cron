#!/usr/bin/env python3
"""repair_ledger.py — closed-loop tracking of every repair the Brain issues.

Without this, repair_shop_diagnosis() is open-loop: it fires recommendations,
the plan is applied, and nothing ever checks whether the repair worked. Every
trigger in the shop is a hand-tuned constant that has never been asked to
learn. This module gives each repair an identity, a "before" snapshot, an
"applied" event, and a post-hoc verdict — the labeled data every ML upgrade
in the shop depends on.
"""
from __future__ import annotations
import hashlib
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

from bot_config import json_dumps, json_loads

LEDGER_KEY = "brain_repair_ledger"
LEDGER_MAX = 500
LEDGER_TTL_DAYS = 90
# A plan is considered to have "applied" every repair issued within this
# window of the plan's own timestamp. The report generator and _store_
# pending_plan run back-to-back in the same brain cycle, so ±60s is safe.
APPLY_WINDOW_SEC = 60


def _repair_id(rec: Dict[str, Any], ts: int) -> str:
    """Stable ID — retries within the same 15m bucket produce the same ID,
    so a crash-and-resume does not fork the ledger."""
    payload = json_dumps({
        "type": rec.get("type"),
        "category": rec.get("category"),
        "alert": rec.get("alert"),
        "param": rec.get("param"),
        "ts_bucket": ts // 900,
    })
    return hashlib.md5(payload.encode()).hexdigest()[:12]


async def _load_ledger(sdb) -> List[dict]:
    raw = await sdb.get_metadata(LEDGER_KEY)
    if not raw:
        return []
    try:
        data = json_loads(raw)
        return data if isinstance(data, list) else []
    except Exception:
        return []


async def _save_ledger(sdb, entries: List[dict]) -> None:
    entries = entries[:LEDGER_MAX]
    await sdb.set_metadata(LEDGER_KEY, json_dumps(entries),
                            ttl=LEDGER_TTL_DAYS * 86400)


async def record_repair_issued(sdb, rec: Dict[str, Any],
                                snapshot: Dict[str, Any]) -> Optional[str]:
    """Call once per emitted repair. snapshot carries the pre-repair metrics
    the verdict will be measured against."""
    ts = int(time.time())
    rid = _repair_id(rec, ts)
    entry = {
        "id": rid,
        "issued_at": ts,
        "type": rec.get("type"),
        "category": rec.get("category"),
        "severity": rec.get("severity"),
        "alert": rec.get("alert"),
        "param": rec.get("param"),
        "message": rec.get("message"),
        "snapshot_before": dict(snapshot),
        "applied_at": None,
        "verdict": None,
        "delta_observed": None,
    }
    entries = await _load_ledger(sdb)
    # If this repair was already issued in the same 15m bucket, refresh it
    # rather than duplicating.
    for i, e in enumerate(entries):
        if e["id"] == rid:
            entries[i] = entry
            await _save_ledger(sdb, entries)
            return rid
    entries.insert(0, entry)
    await _save_ledger(sdb, entries)
    return rid


async def mark_plan_applied(sdb, plan_ts: int) -> int:
    """Called by apply_pending_plan. Marks every repair whose issued_at is
    within APPLY_WINDOW_SEC of the plan's own timestamp. Returns count."""
    entries = await _load_ledger(sdb)
    n = 0
    now = int(time.time())
    for e in entries:
        if e["applied_at"] is not None:
            continue
        if abs(e["issued_at"] - plan_ts) <= APPLY_WINDOW_SEC:
            e["applied_at"] = now
            n += 1
    if n:
        await _save_ledger(sdb, entries)
    return n


async def evaluate_pending_repairs(sdb, current_rows: List[dict],
                                    horizon_hours: int = 48,
                                    min_outcomes: int = 30) -> List[dict]:
    """Called once per brain report. For each applied repair whose horizon
    has passed, compare post-application metrics against the pre-application
    snapshot using a Wilson band so a repair is only marked helped/hurt when
    the move clears noise."""
    from threshold_engine import wilson_ci
    now = int(time.time())
    cutoff = now - horizon_hours * 3600
    entries = await _load_ledger(sdb)
    fresh_verdicts: List[dict] = []
    changed = False
    for e in entries:
        if e["applied_at"] is None or e["verdict"] is not None:
            continue
        if e["applied_at"] > cutoff:
            continue
        post = [r for r in current_rows if r["entry_ts"] >= e["applied_at"]]
        if len(post) < min_outcomes:
            continue
        post_wins = sum(1 for r in post if r["win"])
        post_wr = post_wins / len(post)
        pre_wr = e["snapshot_before"].get("overall_wr")
        if pre_wr is None:
            pre_wr = post_wr
        delta = post_wr - pre_wr
        lo, hi, _ = wilson_ci(post_wins, len(post))
        if delta > 0.03 and lo > pre_wr:
            verdict = "helped"
        elif delta < -0.03 and hi < pre_wr:
            verdict = "hurt"
        else:
            verdict = "neutral"
        e["verdict"] = verdict
        e["delta_observed"] = {
            "wr": round(delta, 4),
            "n_post": len(post),
            "wilson_lo": round(lo, 4),
            "wilson_hi": round(hi, 4),
        }
        fresh_verdicts.append(e)
        changed = True
    if changed:
        await _save_ledger(sdb, entries)
    return fresh_verdicts


async def repair_success_rates(sdb) -> Dict[str, Dict[str, float]]:
    """Aggregate verdicts → per-category help/hurt rates.
    Neutral counts half-credit: the repair didn't hurt, but it didn't help."""
    entries = await _load_ledger(sdb)
    stats: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"helped": 0, "hurt": 0, "neutral": 0, "n": 0}
    )
    for e in entries:
        if e["verdict"] is None:
            continue
        key = e.get("category") or e.get("type") or "unknown"
        stats[key][e["verdict"]] += 1
        stats[key]["n"] += 1
    return {
        k: {
            "help_rate": (v["helped"] + 0.5 * v["neutral"]) / max(1, v["n"]),
            "hurt_rate": v["hurt"] / max(1, v["n"]),
            "n": v["n"],
        }
        for k, v in stats.items()
    }

async def ledger_stats(sdb) -> Dict[str, Any]:
    """For the report header: how many repairs issued/applied/verdict'd."""
    entries = await _load_ledger(sdb)
    return {
        "issued": len(entries),
        "applied": sum(1 for e in entries if e["applied_at"] is not None),
        "verdicts": sum(1 for e in entries if e["verdict"] is not None),
        "helped": sum(1 for e in entries if e["verdict"] == "helped"),
        "hurt": sum(1 for e in entries if e["verdict"] == "hurt"),
    }

async def load_ledger_entries(sdb) -> List[dict]:
    """Public accessor for the raw ledger — used by the contextual
    repair-effectiveness model (threshold_engine.learn_repair_effectiveness).
    Kept separate from repair_success_rates() because the model needs the
    per-entry snapshot_before + verdict, not just the aggregated rates."""
    return await _load_ledger(sdb)