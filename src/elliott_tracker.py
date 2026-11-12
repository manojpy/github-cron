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

Persistence (no workflow changes):
  open records:    Redis hash   metadata:ew:track      (structure_id -> JSON)
  counters:        Redis hash   metadata:ew:stats      (family:group:field -> int)
  event history:   <OUTCOME_DATA_DIR>/shadow/<day>.jsonl, as signal-only rows (no
                   `win` field, `record_type` = "elliott_event"). The Brain's
                   readers ignore rows without `win`, and the existing workflow
                   already commits that directory.

Redis cost (the point of this revision): one round-trip to Aiven is ~250 ms, so
the number of SEQUENTIAL commands, not their size, is what costs runtime.
  prefetch()       1 pipeline for both hashes, launched before the CPU scan so
                   its latency is hidden behind the analysis
  register_many()  1 pipeline for N HSETNX (+1 EXPIRE), 1 archive append
  update_all()     1 pipeline for all HSET/HDEL/EXPIRE, plus a second one only
                   when something actually resolved
  verdicts_for()   1 HGETALL for every family (was one per family)
Semantics are unchanged: a structure is registered once per claim TTL and then left exactly as it was, and counters are still incremented ONLY by the caller whose HDEL
actually removed the record (at-most-once counting under concurrent runs).

The verdict line shown in the Telegram alert is informational only: it never
gates or ranks anything.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from bot_config import cfg, json_dumps, json_loads
from elliott_wave import evaluate_progress
from outcome_storage import append_outcome_batch
from state import RedisStateStore, _execute_pipeline, _rc

__all__ = [
    "make_record", "vol_ref_at", "group_of", "register", "register_many",
    "ensure_track_ttl", "prefetch", "update_all", "verdict_for", "verdicts_for",
    "TRACK_KEY", "STATS_KEY",
]

TRACK_KEY = "ew:track"
STATS_KEY = "ew:stats"
_REDIS_TTL_SEC = 14 * 86400
_STATS_TTL_SEC = 90 * 86400
VOL_REF_BARS = 20


# Registration is idempotent for the whole claim TTL, not just while the record is
# open: the guard key outlives the record, so a structure that was resolved and
# removed can never be registered (and counted) a second time when the same wave
# count is seen again on a later candle. One EVAL per structure, still one RTT.
_REGISTER_LUA = """
if redis.call("SET", KEYS[1], "1", "NX", "EX", ARGV[3]) then
    return redis.call("HSETNX", KEYS[2], ARGV[1], ARGV[2])
end
return 0
"""


def _guard_key(sdb: RedisStateStore, sid: str) -> str:
    return f"{sdb.meta_prefix}notice:ewt:{sid}"


def _guard_ttl() -> int:
    """Must outlive the record: a guard that expires before resolution lets the
    same structure be registered and counted twice. +8 bars of slack for a cron gap."""
    return max(
        int(cfg.ELLIOTT_WAVE_EVENT_TTL_HOURS) * 3600,
        (int(cfg.ELLIOTT_WAVE_TRACK_MAX_BARS) + 8) * 900,
    )


def _tk(sdb: RedisStateStore) -> str:
    return f"{sdb.meta_prefix}{TRACK_KEY}"


def _sk(sdb: RedisStateStore) -> str:
    return f"{sdb.meta_prefix}{STATS_KEY}"


def group_of(disposition: str, gate_state: str, status: str) -> str:
    """Which population a structure belongs to in the statistics:

    event      exhaustion event (not a trade)
    delivered  sent to Telegram
    gated      vetoed by the legacy common-relaxed / wick gate (whatever its score)
    shadow     eligible (score >= MIN_SCORE) but not delivered (delivery off, cap,
               pair limit...)
    watch      WATCH score band
    rejected   failed a structural filter (late entry, reward/risk, low score)

    The verdict line uses delivered + shadow only.
    """
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
                stop: Optional[float], targets: Tuple[float, ...],
                vol_ref: Optional[float], event_id: str, structure_id: str,
                disposition: str, status: str, score: float, gate_state: str,
                reason: str = "") -> Dict[str, Any]:
    return {
        "kind": kind, "pair": pair, "alert_key": alert_key, "family": family,
        "direction": direction, "signal_ts": int(signal_ts), "entry": float(entry),
        "atr": float(atr), "invalidation": float(invalidation),
        "stop": None if stop is None else float(stop),
        "targets": [float(t) for t in targets], "vol_ref": vol_ref,
        "event_id": event_id, "structure_id": structure_id,
        "disposition": disposition, "status": status, "score": float(score),
        "gate": gate_state, "reason": reason,
        "group": group_of(disposition, gate_state, status),
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


def _signal_row(rec: Dict[str, Any], snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """The immutable signal snapshot row, written once per newly tracked structure."""
    return _event_row(rec, "signal", {
        "status": rec["status"], "score": rec["score"],
        "disposition": rec["disposition"], "legacy_gate": rec["gate"],
        "entry": rec["entry"], "atr": rec["atr"],
        "invalidation": rec["invalidation"], "stop": rec["stop"],
        "targets": rec["targets"], "snapshot": snapshot,
    }, ts=rec["signal_ts"])


def _write_rows(rows: List[Dict[str, Any]], log: logging.Logger) -> bool:
    """One append for the whole run (was one per structure). True on success."""
    if not rows:
        return True
    try:
        append_outcome_batch(rows, shadow=True)
        return True
    except Exception as exc:                       # never let archiving break a run
        log.warning(f"Elliott event archive write failed ({len(rows)} row(s)): {exc!r}")
        return False


# ── decode helpers ───────────────────────────────────────────────────────
def _decode_records(raw: Any) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for k, v in (raw or {}).items():
        try:
            out[k.decode() if isinstance(k, bytes) else str(k)] = json_loads(v)
        except Exception:
            continue
    return out


def _decode_stats(raw: Any) -> Dict[str, int]:
    st: Dict[str, int] = {}
    for k, v in (raw or {}).items():
        k = k.decode() if isinstance(k, bytes) else str(k)
        try:
            st[k] = int(v)
        except (TypeError, ValueError):
            continue
    return st


async def prefetch(sdb: RedisStateStore) -> Tuple[Optional[Dict[str, Dict[str, Any]]],
                                                  Optional[Dict[str, int]]]:
    """Both hashes in ONE pipeline. Launch this BEFORE the CPU scan and await it
    afterwards: the ~250 ms round-trip then costs nothing. Returns (None, None)
    on any failure — callers fall back to a lazy load, never to fabricated data.
    Never raises."""
    if sdb.degraded or not sdb._redis:
        return None, None
    try:
        async with _rc(sdb._redis).pipeline(transaction=False) as pipe:
            pipe.hgetall(_tk(sdb))
            pipe.hgetall(_sk(sdb))
            raw = await asyncio.wait_for(_execute_pipeline(pipe), timeout=4.0)
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            return None, None
        return _decode_records(raw[0]), _decode_stats(raw[1])
    except Exception:
        return None, None


async def _load_all(sdb: RedisStateStore) -> Dict[str, Dict[str, Any]]:
    raw = await sdb._safe_redis_op(
        lambda: _rc(sdb._redis).hgetall(_tk(sdb)), 3.0, "ew_track_load")
    return _decode_records(raw)


async def _load_stats(sdb: RedisStateStore) -> Dict[str, int]:
    raw = await sdb._safe_redis_op(
        lambda: _rc(sdb._redis).hgetall(_sk(sdb)), 3.0, "ew_stats_load")
    return _decode_stats(raw)


# ── registration ─────────────────────────────────────────────────────────
async def register_many(sdb: RedisStateStore,
                        items: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]],
                        log: logging.Logger) -> Dict[str, Dict[str, Any]]:
    """Start following many structures in ONE round-trip and archive the immutable
    signal snapshots of the ones that were actually added (HSETNX), in ONE append.

    Returns {structure_id: record} for the newly added structures only, so the
    caller can merge them into a prefetched track map and get exactly what a
    post-register HGETALL would have returned. Never raises.
    """
    if not items or sdb.degraded or not sdb._redis:
        return {}

    prepared: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for rec, snapshot in items:
        r = dict(rec)
        r["feature_version"] = snapshot.get("elliott_feature_version")
        prepared.append((r, snapshot))

    key = _tk(sdb)
    try:
        async with _rc(sdb._redis).pipeline(transaction=False) as pipe:
            for r, _snap in prepared:
                pipe.eval(_REGISTER_LUA, 2, _guard_key(sdb, r["structure_id"]), key,
                          r["structure_id"], json_dumps(r), str(_guard_ttl()))
            # Same TTL refresh the old per-register EXPIRE did, at zero extra RTT.
            pipe.expire(key, _REDIS_TTL_SEC)
            results = await asyncio.wait_for(_execute_pipeline(pipe), timeout=5.0)
        if not isinstance(results, (list, tuple)) or len(results) != len(prepared) + 1:
            raise RuntimeError("register pipeline returned an unexpected result count")
    except Exception as exc:
        log.warning(f"Elliott tracker register pipeline failed: {exc!r}")
        return await _register_sequential(sdb, prepared, log)

    added: Dict[str, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []
    for (r, snap), ok in zip(prepared, results[:-1]):
        if not ok:
            continue                    # already tracked: left exactly as it was
        added[r["structure_id"]] = r
        rows.append(_signal_row(r, snap))
    _write_rows(rows, log)
    return added


async def _register_sequential(sdb: RedisStateStore,
                               prepared: List[Tuple[Dict[str, Any], Dict[str, Any]]],
                               log: logging.Logger) -> Dict[str, Dict[str, Any]]:
    """Fallback only. HSETNX is idempotent, so a partially applied pipeline cannot
    produce a duplicate record; the worst case is a lost signal row, which is
    logged."""
    added: Dict[str, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []
    for r, snap in prepared:
        try:
            _r = r
            ok = await sdb._safe_redis_op(
                lambda: _rc(sdb._redis).eval(
                    _REGISTER_LUA, 2, _guard_key(sdb, _r["structure_id"]), _tk(sdb),
                    _r["structure_id"], json_dumps(_r), str(_guard_ttl())),
                2.0, "ew_track_add_fb")
            if ok:
                added[r["structure_id"]] = r
                rows.append(_signal_row(r, snap))
        except Exception as exc:
            log.warning(f"Elliott tracker register fallback failed: {exc!r}")
    _write_rows(rows, log)
    if added:
        await ensure_track_ttl(sdb)
    return added


async def register(sdb: RedisStateStore, rec: Dict[str, Any], snapshot: Dict[str, Any],
                   log: logging.Logger) -> bool:
    """Single-structure wrapper kept for compatibility. Prefer register_many()."""
    return bool(await register_many(sdb, [(rec, snapshot)], log))


async def ensure_track_ttl(sdb: RedisStateStore) -> None:
    """One EXPIRE for the whole track hash. register_many() and update_all() already
    do this inside their own pipelines; this is for callers that register by hand."""
    if sdb.degraded or not sdb._redis:
        return
    try:
        await sdb._safe_redis_op(
            lambda: _rc(sdb._redis).expire(_tk(sdb), _REDIS_TTL_SEC), 2.0, "ew_track_ttl")
    except Exception:
        pass


# ── follow-up ────────────────────────────────────────────────────────────
_EVENT_ORDER = (("t1", "first_target"), ("t2", "target2"),
                ("stop", "stop"), ("invalidated", "invalidated"))


def _count_fields(rec: Dict[str, Any], prog: Dict[str, Any]) -> List[str]:
    """Single source of truth for the counter fields (pipeline + fallback)."""
    base = f"{rec['family']}:{rec['group']}"
    outcome = prog.get("outcome")
    fields = [f"{base}:n"]
    if outcome in ("t2", "stop", "invalidated", "expired"):
        fields.append(f"{base}:{outcome}")
    if prog.get("t1_before_adverse"):
        fields.append(f"{base}:t1_first")
    if outcome in ("stop", "invalidated"):
        fields.append(f"{base}:adverse")
    return fields


async def update_all(sdb: RedisStateStore, parsed_cache: Dict[str, Tuple[Any, Any, Any]],
                     last_idx: Dict[str, int], log: logging.Logger,
                     *, preloaded: Optional[Dict[str, Dict[str, Any]]] = None
                     ) -> Dict[str, int]:
    """Advance every tracked structure with the candles that closed since the last
    run. Appends each new event once, and finalises (counters + resolution row)
    structures whose outcome is decided. Never raises.

    `preloaded` is the track map from prefetch() merged with register_many()'s
    additions — i.e. exactly what a fresh HGETALL would return, minus one RTT.

    Ordering: ALL archive rows are appended first, then Redis is mutated. If the
    append fails nothing is persisted, so the events are re-emitted next run
    instead of being lost (same safety ordering as state.py's resolvers).
    """
    stats = {"tracked": 0, "events": 0, "resolved": 0}

    recs = preloaded
    if recs is None:
        try:
            recs = await _load_all(sdb)
        except Exception as exc:
            log.warning(f"Elliott tracker load failed: {exc!r}")
            return stats
    stats["tracked"] = len(recs)
    if not recs:
        return stats

    max_bars = int(cfg.ELLIOTT_WAVE_TRACK_MAX_BARS)
    all_rows: List[Dict[str, Any]] = []
    to_save: List[Tuple[str, Dict[str, Any]]] = []
    to_delete: List[str] = []
    to_count: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []

    for sid, rec in recs.items():
        pair = str(rec.get("pair"))
        pd_15m = (parsed_cache.get(pair) or (None, None, None))[0]
        end = last_idx.get(pair)
        if pd_15m is None or end is None:
            continue
        try:
            prog = evaluate_progress(rec, pd_15m.ts, pd_15m.high, pd_15m.low,
                                     pd_15m.close, pd_15m.volume, int(end),
                                     max_bars=max_bars)
            if prog.get("outcome") == "missing_signal_candle":
                # older than the fetched window: stop following, say so, never guess
                if int(pd_15m.ts[0]) > rec["signal_ts"]:
                    prog = {"state": "final", "outcome": "unobservable", "bars": 0,
                            "last_ts": int(pd_15m.ts[int(end)]), "first_bars": {},
                            "t1_before_adverse": False}
                else:
                    continue

            emitted = rec.setdefault("emitted", [])
            rows: List[Dict[str, Any]] = []
            fb = prog.get("first_bars") or {}
            for field, ev in _EVENT_ORDER:
                if fb.get(field) and ev not in emitted:
                    emitted.append(ev)
                    t_ev = rec["signal_ts"] + 900 * int(fb[field])
                    rows.append(_event_row(rec, ev, {"bars_after_signal": int(fb[field])},
                                           ts=t_ev))
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

            all_rows.extend(rows)
            if not final:
                if rows:
                    to_save.append((sid, rec))
                continue
            to_delete.append(sid)
            to_count.append((rec, prog))
        except Exception as exc:
            log.warning(f"[{pair}] Elliott tracker update failed for {sid}: {exc!r}")

    if not _write_rows(all_rows, log):
        # Nothing persisted: `emitted` stays as it was in Redis, so these events
        # are emitted again next run rather than silently dropped.
        return stats
    stats["events"] += len(all_rows)

    if sdb.degraded or not sdb._redis or not (to_save or to_delete):
        return stats

    client = _rc(sdb._redis)
    track_key, stats_key = _tk(sdb), _sk(sdb)
    n_save, n_del = len(to_save), len(to_delete)
    try:
        async with client.pipeline(transaction=False) as pipe:
            for sid, rec in to_save:
                pipe.hset(track_key, sid, json_dumps(rec))
            for sid in to_delete:
                pipe.hdel(track_key, sid)
            pipe.expire(track_key, _REDIS_TTL_SEC)
            results = await asyncio.wait_for(_execute_pipeline(pipe), timeout=5.0)
        if not isinstance(results, (list, tuple)) or len(results) != n_save + n_del + 1:
            raise RuntimeError("tracker pipeline returned an unexpected result count")

        # at-most-once counting: only structures whose HDEL actually removed the
        # record are counted, exactly as the old sequential path did.
        del_results = results[n_save:n_save + n_del]
        really_resolved = [(rec, prog) for removed, (rec, prog)
                           in zip(del_results, to_count) if removed]
        stats["resolved"] += len(really_resolved)
        if really_resolved:
            async with client.pipeline(transaction=False) as pipe:
                for rec, prog in really_resolved:
                    for f in _count_fields(rec, prog):
                        pipe.hincrby(stats_key, f, 1)
                pipe.expire(stats_key, _STATS_TTL_SEC)
                await asyncio.wait_for(_execute_pipeline(pipe), timeout=4.0)
    except Exception as exc:
        log.warning(f"Elliott tracker pipeline failed: {exc!r}")
        # Fallback: sequential. HDEL is idempotent, so a partially applied
        # pipeline cannot double-count here either.
        for sid, rec in to_save:
            try:
                _sid, _rec = sid, rec
                await sdb._safe_redis_op(
                    lambda: _rc(sdb._redis).hset(_tk(sdb), _sid, json_dumps(_rec)),
                    2.0, "ew_track_save_fb")
            except Exception:
                pass
        for sid, (rec, prog) in zip(to_delete, to_count):
            try:
                _sid = sid
                removed = await sdb._safe_redis_op(
                    lambda: _rc(sdb._redis).hdel(_tk(sdb), _sid), 2.0, "ew_track_del_fb")
                if removed:
                    await _count_sequential(sdb, rec, prog)
                    stats["resolved"] += 1
            except Exception:
                pass
    return stats


async def _count_sequential(sdb: RedisStateStore, rec: Dict[str, Any],
                            prog: Dict[str, Any]) -> None:
    """Fallback used only when the counter pipeline fails."""
    key = _sk(sdb)
    for f in _count_fields(rec, prog):
        _f = f
        await sdb._safe_redis_op(
            lambda: _rc(sdb._redis).hincrby(key, _f, 1), 2.0, "ew_stats_inc")
    await sdb._safe_redis_op(
        lambda: _rc(sdb._redis).expire(key, _STATS_TTL_SEC), 2.0, "ew_stats_ttl")


# ── verdict ──────────────────────────────────────────────────────────────
def _verdict_from(st: Dict[str, int], family: str) -> str:
    def g(group: str, field: str) -> int:
        return st.get(f"{family}:{group}:{field}", 0)

    succ = sum(g(gr, "t1_first") for gr in ("delivered", "shadow"))
    adv = sum(g(gr, "adverse") for gr in ("delivered", "shadow"))
    n_dec = succ + adv
    need = int(cfg.ELLIOTT_WAVE_VERDICT_MIN_SAMPLES)
    if n_dec < need:
        return (f"Brain: LEARNING — {family} first target before invalidation "
                f"{succ}/{n_dec} (need {need})")
    rate = succ / n_dec
    if rate >= float(cfg.ELLIOTT_WAVE_VERDICT_TAKE_RATE):
        v = "TAKE"
    elif rate < float(cfg.ELLIOTT_WAVE_VERDICT_AVOID_RATE):
        v = "AVOID"
    else:
        v = "WATCH"
    return (f"Brain: {v} — {family} first target before invalidation "
            f"{succ}/{n_dec} ({rate:.0%})")


async def verdicts_for(sdb: RedisStateStore, families: Sequence[str],
                       *, preloaded_stats: Optional[Dict[str, int]] = None
                       ) -> Dict[str, str]:
    """One informational line per family, from ONE HGETALL (or the prefetch).
    TAKE / WATCH / AVOID only once the sample is large enough; before that it
    says it is learning. Never gates or ranks anything."""
    fams = sorted(set(families))
    if not fams:
        return {}
    st = preloaded_stats
    if st is None:
        try:
            st = await _load_stats(sdb)
        except Exception:
            st = {}
    return {f: _verdict_from(st, f) for f in fams}


async def verdict_for(sdb: RedisStateStore, family: str) -> str:
    """Single-family wrapper kept for compatibility."""
    return (await verdicts_for(sdb, [family]))[family]