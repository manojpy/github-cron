"""elliott_tracker — append-only follow-up of Elliott signals and events.

A stored Elliott signal never changes. What happens afterwards is recorded as
SEPARATE events that point back at the original `event_id`:

    signal        one per tracked structure, written once, with the full feature
                  snapshot as it was at the signal candle
    first_target  the first target was reached before any adverse event
    target2       the second target was reached (trade ended)
    invalidated   a candle CLOSED beyond the structural invalidation level
    stop          the structural stop was touched
    expired       neither side was reached within ELLIOTT_WAVE_TRACK_MAX_BARS
    resolution    one per structure at the end: outcome, bars to each first touch,
                  MFE / MAE in % and ATR, later volume versus the volume at the
                  signal

What is tracked: every delivered or shadow-recorded setup, plus the Wave 5
exhaustion events (which are never trades). Structural brackets (stop at the
structure, targets from the wave) are used here; they are independent of the
bot's fixed stop/target outcome bracket, so the two can be compared.

Persistence (no new workflow changes):
  * open records:    Redis hash   metadata:ew:track      (structure_id -> JSON)
  * counters:        Redis hash   metadata:ew:stats      (family:group:field -> int)
  * event history:   <OUTCOME_DATA_DIR>/shadow/<day>.jsonl, as signal-only rows (no
    `win` field, `record_type` = "elliott_event"). The Brain's readers ignore rows
    without `win`, and the existing workflow already commits that directory.

The verdict line shown in the Telegram alert is informational only: it never gates
or ranks anything.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from bot_config import cfg, json_dumps, json_loads
from elliott_wave import evaluate_progress
from outcome_storage import append_outcome_batch
from state import RedisStateStore, _rc

__all__ = ["make_record", "register", "update_all", "verdict_for", "TRACK_KEY", "STATS_KEY"]

TRACK_KEY = "ew:track"
STATS_KEY = "ew:stats"
_REDIS_TTL_SEC = 14 * 86400
VOL_REF_BARS = 20


def _tk(sdb: RedisStateStore) -> str:
    return f"{sdb.meta_prefix}{TRACK_KEY}"


def _sk(sdb: RedisStateStore) -> str:
    return f"{sdb.meta_prefix}{STATS_KEY}"


def group_of(disposition: str, gate_state: str, status: str) -> str:
    """Which population a structure belongs to in the statistics:
    event      exhaustion event (not a trade)
    delivered  sent to Telegram
    gated      vetoed by the legacy common-relaxed / wick gate (whatever its score)
    shadow     eligible (score >= MIN_SCORE) but not delivered (delivery off, cap, pair limit...)
    watch      WATCH score band
    rejected   failed a structural filter (late entry, reward/risk, low score)
    The verdict line uses delivered + shadow only."""
    if disposition == "event":
        return "event"
    if disposition in ("delivered", "send_failed_queued"):
        return "delivered"
    if gate_state not in ("pass", "n/a"):
        return "gated"
    if status == "eligible":
        return "shadow"
    if status == "watch":
        return "watch"
    return "rejected"


def make_record(*, kind: str, pair: str, alert_key: str, family: str, direction: str,
                signal_ts: int, entry: float, atr: float, invalidation: float,
                stop: Optional[float], targets: Tuple[float, ...], vol_ref: Optional[float],
                event_id: str, structure_id: str, disposition: str, status: str,
                score: float, gate_state: str, reason: str = "") -> Dict[str, Any]:
    return {
        "kind": kind, "pair": pair, "alert_key": alert_key, "family": family,
        "direction": direction, "signal_ts": int(signal_ts), "entry": float(entry),
        "atr": float(atr), "invalidation": float(invalidation),
        "stop": None if stop is None else float(stop),
        "targets": [float(t) for t in targets], "vol_ref": vol_ref,
        "event_id": event_id, "structure_id": structure_id,
        "disposition": disposition, "status": status, "score": float(score),
        "gate": gate_state, "reason": reason, "group": group_of(disposition, gate_state, status),
        "emitted": [], "created": int(time.time()),
    }


def vol_ref_at(volume: Any, idx: int) -> Optional[float]:
    v = np.asarray(volume, dtype=np.float64)[max(0, idx - VOL_REF_BARS + 1):idx + 1]
    if len(v) == 0 or not np.all(np.isfinite(v)):
        return None
    m = float(np.mean(v))
    return m if m > 0 else None


def _event_row(rec: Dict[str, Any], event: str, extra: Optional[Dict[str, Any]] = None,
               ts: Optional[int] = None) -> Dict[str, Any]:
    """A signal-only archive row (no `win`): ignored by every outcome reader."""
    row = {
        "record_type": "elliott_event",
        "event": event,
        "ref_event_id": rec["event_id"],
        "structure_id": rec["structure_id"],
        "pair": rec["pair"],
        "alert_key": rec["alert_key"],
        "family": rec["family"],
        "direction": rec["direction"],
        "entry_ts": rec["signal_ts"],
        "group": rec["group"],
        "feature_version": rec.get("feature_version"),
        "event_ts": int(ts if ts is not None else time.time()),
    }
    if extra:
        row.update(extra)
    return row


def _write_rows(rows: List[Dict[str, Any]], log: logging.Logger) -> bool:
    if not rows:
        return True
    try:
        append_outcome_batch(rows, shadow=True)
        return True
    except Exception as exc:                       # never let archiving break a run
        log.warning(f"Elliott event archive write failed: {exc!r}")
        return False


async def register(sdb: RedisStateStore, rec: Dict[str, Any], snapshot: Dict[str, Any],
                   log: logging.Logger) -> bool:
    """Start following a structure and archive its immutable signal snapshot.
    A structure already tracked is left exactly as it was (HSETNX)."""
    rec = dict(rec)
    rec["feature_version"] = snapshot.get("elliott_feature_version")
    key = _tk(sdb)
    added = await sdb._safe_redis_op(
        lambda: _rc(sdb._redis).hsetnx(key, rec["structure_id"], json_dumps(rec)),
        2.0, "ew_track_add")
    if not added:
        return False
    await sdb._safe_redis_op(lambda: _rc(sdb._redis).expire(key, _REDIS_TTL_SEC), 2.0, "ew_track_ttl")
    _write_rows([_event_row(rec, "signal", {
        "status": rec["status"], "score": rec["score"], "disposition": rec["disposition"],
        "legacy_gate": rec["gate"], "entry": rec["entry"], "atr": rec["atr"],
        "invalidation": rec["invalidation"], "stop": rec["stop"], "targets": rec["targets"],
        "snapshot": snapshot,
    }, ts=rec["signal_ts"])], log)
    return True


async def _load_all(sdb: RedisStateStore) -> Dict[str, Dict[str, Any]]:
    raw = await sdb._safe_redis_op(lambda: _rc(sdb._redis).hgetall(_tk(sdb)), 3.0, "ew_track_load")
    out: Dict[str, Dict[str, Any]] = {}
    for k, v in (raw or {}).items():
        try:
            out[k.decode() if isinstance(k, bytes) else str(k)] = json_loads(v)
        except Exception:
            continue
    return out


_EVENT_ORDER = (("t1", "first_target"), ("t2", "target2"), ("stop", "stop"), ("invalidated", "invalidated"))


async def update_all(sdb: RedisStateStore, parsed_cache: Dict[str, Tuple[Any, Any, Any]],
                     last_idx: Dict[str, int], log: logging.Logger) -> Dict[str, int]:
    """Advance every tracked structure with the candles that closed since the last
    run. Appends each new event once, and finalises (counters + resolution row)
    structures whose outcome is decided. Never raises."""
    stats = {"tracked": 0, "events": 0, "resolved": 0}
    try:
        recs = await _load_all(sdb)
    except Exception as exc:
        log.warning(f"Elliott tracker load failed: {exc!r}")
        return stats
    stats["tracked"] = len(recs)
    max_bars = int(cfg.ELLIOTT_WAVE_TRACK_MAX_BARS)
    for sid, rec in recs.items():
        pair = str(rec.get("pair"))
        pd_15m = (parsed_cache.get(pair) or (None, None, None))[0]
        end = last_idx.get(pair)
        if pd_15m is None or end is None:
            continue
        try:
            prog = evaluate_progress(rec, pd_15m.ts, pd_15m.high, pd_15m.low, pd_15m.close,
                                     pd_15m.volume, int(end), max_bars=max_bars)
            if prog.get("outcome") == "missing_signal_candle":
                # older than the fetched window: stop following, say so, never guess
                if int(pd_15m.ts[0]) > rec["signal_ts"]:
                    prog = {"state": "final", "outcome": "unobservable", "bars": 0,
                            "last_ts": int(pd_15m.ts[int(end)]), "first_bars": {},
                            "t1_before_adverse": False}
                else:
                    continue
            rows: List[Dict[str, Any]] = []
            fb = prog.get("first_bars") or {}
            for field, ev in _EVENT_ORDER:
                if fb.get(field) and ev not in rec["emitted"]:
                    rec["emitted"].append(ev)
                    t_ev = rec["signal_ts"] + 900 * int(fb[field])
                    rows.append(_event_row(rec, ev, {"bars_after_signal": int(fb[field])}, ts=t_ev))
            final = prog.get("state") == "final"
            if final:
                rows.append(_event_row(rec, "resolution", {
                    "outcome": prog["outcome"], "bars": prog.get("bars"),
                    "first_bars": fb, "t1_before_adverse": prog.get("t1_before_adverse"),
                    "mfe_pct": prog.get("mfe_pct"), "mae_pct": prog.get("mae_pct"),
                    "mfe_atr": prog.get("mfe_atr"), "mae_atr": prog.get("mae_atr"),
                    "volume_ratio": prog.get("volume_ratio"),
                    "volume_expanded": prog.get("volume_expanded"),
                    "structural_invalidation_hit": fb.get("invalidated") is not None,
                }, ts=prog.get("last_ts")))
            _write_rows(rows, log)
            stats["events"] += len(rows)
            if not final:
                if rows:
                    await _save(sdb, sid, rec)
                continue
            # at-most-once counting: only the caller whose HDEL removed the record counts it
            removed = await _delete(sdb, sid)
            if removed:
                await _count(sdb, rec, prog)
                stats["resolved"] += 1
        except Exception as exc:
            log.warning(f"[{pair}] Elliott tracker update failed for {sid}: {exc!r}")
    return stats


async def _save(sdb: RedisStateStore, sid: str, rec: Dict[str, Any]) -> None:
    await sdb._safe_redis_op(lambda: _rc(sdb._redis).hset(_tk(sdb), sid, json_dumps(rec)),
                             2.0, "ew_track_save")


async def _delete(sdb: RedisStateStore, sid: str) -> Any:
    return await sdb._safe_redis_op(lambda: _rc(sdb._redis).hdel(_tk(sdb), sid), 2.0, "ew_track_del")


async def _incr(sdb: RedisStateStore, key: str, field: str) -> None:
    await sdb._safe_redis_op(lambda: _rc(sdb._redis).hincrby(key, field, 1), 2.0, "ew_stats_inc")


async def _count(sdb: RedisStateStore, rec: Dict[str, Any], prog: Dict[str, Any]) -> None:
    base = f"{rec['family']}:{rec['group']}"
    key = _sk(sdb)
    outcome = prog.get("outcome")
    fields = [f"{base}:n"]
    if outcome in ("t2", "stop", "invalidated", "expired"):
        fields.append(f"{base}:{outcome}")
    if prog.get("t1_before_adverse"):
        fields.append(f"{base}:t1_first")
    if outcome in ("stop", "invalidated"):
        fields.append(f"{base}:adverse")
    for f in fields:
        await _incr(sdb, key, f)
    await sdb._safe_redis_op(lambda: _rc(sdb._redis).expire(key, 90 * 86400), 2.0, "ew_stats_ttl")


async def verdict_for(sdb: RedisStateStore, family: str) -> str:
    """One informational line: how this wave family has behaved so far (delivered +
    shadow, structurally valid and not vetoed by the legacy gate). TAKE / WATCH /
    AVOID only once the sample is large enough; before that it says it is learning."""
    try:
        raw = await sdb._safe_redis_op(lambda: _rc(sdb._redis).hgetall(_sk(sdb)), 3.0, "ew_stats_load")
    except Exception:
        raw = None
    st: Dict[str, int] = {}
    for k, v in (raw or {}).items():
        k = k.decode() if isinstance(k, bytes) else str(k)
        try:
            st[k] = int(v)
        except (TypeError, ValueError):
            continue

    def g(group: str, field: str) -> int:
        return st.get(f"{family}:{group}:{field}", 0)

    succ = sum(g(gr, "t1_first") for gr in ("delivered", "shadow"))
    adv = sum(g(gr, "adverse") for gr in ("delivered", "shadow"))
    n_dec = succ + adv
    need = int(cfg.ELLIOTT_WAVE_VERDICT_MIN_SAMPLES)
    if n_dec < need:
        return f"Brain: LEARNING — {family} first target before invalidation {succ}/{n_dec} (need {need})"
    rate = succ / n_dec
    if rate >= float(cfg.ELLIOTT_WAVE_VERDICT_TAKE_RATE):
        v = "TAKE"
    elif rate < float(cfg.ELLIOTT_WAVE_VERDICT_AVOID_RATE):
        v = "AVOID"
    else:
        v = "WATCH"
    return f"Brain: {v} — {family} first target before invalidation {succ}/{n_dec} ({rate:.0%})"
