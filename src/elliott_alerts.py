"""elliott_alerts — run-level wiring for the Elliott Wave module.

Sits between the pure engine (elliott_wave.py) and the bot's infrastructure
(Redis state, Telegram, outcome tracking). Called ONCE per run, after the legacy
alerts, the pullback WATCH message and the heat-gate release have finished, so it
cannot change any legacy decision.

Isolation, by design (do not "fix" any of these):

  * Elliott rows bypass the legacy one-open-trade slot and the re-entry cooldown
    (see RedisStateStore.record_pending_outcome / record_shadow_pending_outcome /
    get_active_trade). A MACD trade never blocks a wave count and vice versa.
  * The bypass does NOT remove Elliott's own protections:
      - one Telegram alert and one Recorded row per wave structure
        (`structure_id`, claimed in Redis for ELLIOTT_WAVE_EVENT_TTL_HOURS);
      - one Shadowed row per wave structure (separate claim);
      - ELLIOTT_WAVE_MAX_ALERTS_PER_RUN, separate from MAX_ALERTS_PER_RUN.
  * Elliott never touches the legacy alert registry sets, the dedup of legacy
    alerts, the heat gate or the confluence score. Its outcome rows carry no
    confluence score, so the threshold learning of the MACD alerts is not mixed
    with Elliott scores.
  * A stored signal is immutable: the decision is taken on the last closed candle
    only, from data that existed at that candle (elliott_wave.analyze never reads
    beyond it), and nothing is rewritten later.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from alert_registry import ELLIOTT_REGISTRY
from alerts import escape_markdown_v2, queue_failed_alert
from bot_config import cfg, format_ist_time
from elliott_wave import (
    Analysis, ElliottParams, FAMILY_PRIORITY, WaveSetup, analyze,
)
from fetcher import get_last_closed_index_from_array
from state import RedisStateStore, _rc

__all__ = ["params_from_cfg", "scan_pair", "run_elliott_scan", "format_elliott_message"]

INTERVAL_MIN = 15
_FAMILY_TITLE = {"wave3": "Wave 3", "wave5": "Wave 5", "abc": "ABC reversal"}


def params_from_cfg() -> ElliottParams:
    return ElliottParams(
        pivot_len=int(cfg.ELLIOTT_WAVE_PIVOT_LENGTH),
        lookback=int(cfg.ELLIOTT_WAVE_LOOKBACK),
        min_swing_atr=float(cfg.ELLIOTT_WAVE_MIN_SWING_ATR),
        atr_len=int(cfg.ELLIOTT_WAVE_ATR_LEN),
        interval_sec=INTERVAL_MIN * 60,
        min_score=float(cfg.ELLIOTT_WAVE_MIN_SCORE),
        watch_score=float(cfg.ELLIOTT_WAVE_WATCH_SCORE),
        max_entry_distance_atr=float(cfg.ELLIOTT_WAVE_MAX_ENTRY_DISTANCE_ATR),
        max_age_bars=int(cfg.ELLIOTT_WAVE_MAX_AGE_BARS),
        max_age_bars_wave5=int(cfg.ELLIOTT_WAVE_MAX_AGE_BARS) * 2,
        min_rr=float(cfg.ELLIOTT_WAVE_MIN_RR),
        enable_wave3=bool(cfg.ELLIOTT_WAVE_ENABLE_WAVE3),
        enable_wave5=bool(cfg.ELLIOTT_WAVE_ENABLE_WAVE5),
        enable_abc=bool(cfg.ELLIOTT_WAVE_ENABLE_ABC),
    )


def alert_key_of(s: WaveSetup) -> str:
    key = f"elliott_{s.family}_{s.direction}"
    if key not in ELLIOTT_REGISTRY:
        raise KeyError(f"unregistered Elliott alert key: {key}")
    return key


def _deliver_enabled(family: str) -> bool:
    return bool(getattr(cfg, f"ELLIOTT_WAVE_DELIVER_{family.upper()}", True))


# ── per-pair scan (CPU only, no I/O) ─────────────────────────────────────

def scan_pair(pair: str, pd_15m: Any, reference_time: int,
              params: ElliottParams) -> Tuple[Optional[int], Optional[Analysis]]:
    """Analyse the last CLOSED 15m candle. Returns (index, analysis) or (None, None)
    when the candle is missing / stale / unstable (same rule as the legacy scan)."""
    if pd_15m is None or len(pd_15m) == 0:
        return None, None
    i = get_last_closed_index_from_array(pd_15m.ts, INTERVAL_MIN, reference_time, pair)
    if i is None:
        return None, None
    res = analyze(pair, pd_15m.ts, pd_15m.open, pd_15m.high, pd_15m.low,
                  pd_15m.close, pd_15m.volume, end=int(i), params=params)
    return int(i), res


# ── message ──────────────────────────────────────────────────────────────

def _px(x: float) -> str:
    ax = abs(x)
    if ax >= 1000:
        return f"{x:,.2f}"
    if ax >= 1:
        return f"{x:.4f}".rstrip("0").rstrip(".")
    return f"{x:.6g}"


def _setup_block(s: WaveSetup) -> str:
    e = escape_markdown_v2
    side = "BUY" if s.direction == "buy" else "SELL"
    icon = "🟢" if s.direction == "buy" else "🔴"
    f = s.features
    htf = {1: "aligned", -1: "against", 0: "flat", None: "n/a"}.get(s.htf_bias, "n/a")
    comp = " · ".join(
        f"{k.replace('_', ' ')} {v.score:.0f}/{v.max_score:.0f}" + ("" if v.available else "*")
        for k, v in s.detail.items()
    )
    tgt = " · ".join(f"T{i + 1} {_px(t)}" for i, t in enumerate(s.targets))
    lines = [
        f"{icon} *{e(s.pair)}* — Elliott {e(_FAMILY_TITLE[s.family])} {e(side)}",
        e(f"Score {s.score:.0f}/100 · R/R {s.rr:.1f} · HTF {htf} · age {s.age_bars} bars"),
        e(f"Count: {f.get('elliott_wave_count', '')}"),
        e(f"Entry {_px(s.entry_price)} · Stop {_px(s.stop)} · {tgt}"),
        e(f"Trigger {_px(s.trigger_level)} ({s.entry_distance_atr:.2f} ATR beyond) · "
          f"Invalidation {_px(s.invalidation)}"),
        e(f"Scores: {comp}"),
    ]
    return "\n".join(lines)


def format_elliott_message(setups: List[WaveSetup]) -> str:
    e = escape_markdown_v2
    head = e(f"🌊 {cfg.BOT_NAME} — Elliott Wave ({len(setups)})")
    body = "\n\n".join(_setup_block(s) for s in setups)
    ts = max(s.signal_ts for s in setups) + INTERVAL_MIN * 60
    foot = (f"📆  {e(format_ist_time(ts, '%d-%m-%Y'))}  "
            f"⏰ {e(format_ist_time(ts, '%H:%M IST'))}")
    note = e("* = input unavailable, scored 0")
    return f"{head}\n\n{body}\n\n{note}\n{foot}"


# ── persistence helpers ──────────────────────────────────────────────────

def _context(s: WaveSetup, disposition: str) -> Dict[str, Any]:
    """Everything the Brain may want to slice by, JSON-safe, frozen at signal time."""
    ctx: Dict[str, Any] = dict(s.features)
    ctx.update({
        "elliott_disposition": disposition,
        "elliott_trigger_level": s.trigger_level,
        "elliott_stop": s.stop,
        "elliott_targets": list(s.targets),
        "elliott_invalidation": s.invalidation,
        "elliott_atr": s.atr,
        "elliott_labels": {k: list(v) for k, v in s.labels.items()},
        "elliott_signal_ts": s.signal_ts,
    })
    return ctx


def _outcome_kwargs(pair: str, s: WaveSetup, disposition: str) -> Dict[str, Any]:
    return {
        "pair": pair,
        "alert_key": alert_key_of(s),
        "direction": s.direction,
        "entry_ts": int(s.signal_ts),
        "entry_price": float(s.entry_price),
        # The outcome store drops a row that has no score/total, so the wave score
        # (out of 100) is stored in those fields. It is NOT a confluence score:
        # archive_reader and Brain._parse_rows skip Elliott keys, so legacy
        # threshold learning never sees these rows.
        "confluence_score": float(s.score),
        "confluence_total": 100.0,
        "context": _context(s, disposition),
    }


async def _claim(sdb: RedisStateStore, kind: str, structure_id: str) -> bool:
    ttl = int(cfg.ELLIOTT_WAVE_EVENT_TTL_HOURS) * 3600
    return await sdb.claim_notice(f"ew{kind}:{structure_id}", ttl)


async def _release(sdb: RedisStateStore, kind: str, structure_id: str) -> None:
    key = f"{sdb.meta_prefix}notice:ew{kind}:{structure_id}"
    try:
        await sdb._safe_redis_op(lambda: _rc(sdb._redis).delete(key), 2.0, f"ew_release:{kind}")
    except Exception:
        pass


# ── run-level entry point ────────────────────────────────────────────────

async def run_elliott_scan(
    parsed_cache: Dict[str, Tuple[Any, Any, Any]],
    pairs: List[str],
    reference_time: int,
    state_db: RedisStateStore,
    telegram_queue: Any,
    log: logging.Logger,
    suppress_delivery: bool = False,
) -> Dict[str, int]:
    """Scan every pair on its last closed 15m candle, deliver the best eligible
    setups (cap + dedup) and record the rest as Shadowed. Never raises."""
    stats = {"pairs": 0, "candidates": 0, "eligible": 0, "delivered": 0,
             "shadowed": 0, "duplicates": 0, "errors": 0}
    if not cfg.ENABLE_ELLIOTT_WAVE:
        return stats
    try:
        return await _run(parsed_cache, pairs, reference_time, state_db,
                          telegram_queue, log, suppress_delivery, stats)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        stats["errors"] += 1
        log.error(f"Elliott scan failed (legacy alerts unaffected): {exc!r}", exc_info=True)
        return stats


async def _run(parsed_cache, pairs, reference_time, sdb, tg, log, suppress, stats) -> Dict[str, int]:
    if cfg.DRY_RUN_MODE:
        log.info("🌊 Elliott: DRY_RUN_MODE — scan skipped (nothing is claimed or recorded)")
        return stats
    if sdb.degraded:
        log.warning("🌊 Elliott: Redis unavailable — scan skipped (no way to deduplicate)")
        return stats

    params = params_from_cfg()
    sem = asyncio.Semaphore(max(1, int(cfg.EVAL_CONCURRENCY_LIMIT)))

    async def _one(pair: str):
        async with sem:
            pd_15m = (parsed_cache.get(pair) or (None, None, None))[0]
            try:
                i, res = await asyncio.to_thread(scan_pair, pair, pd_15m, reference_time, params)
            except Exception as exc:
                log.warning(f"[{pair}] Elliott analysis failed: {exc!r}")
                stats["errors"] += 1
                return pair, None
            return pair, res

    scanned = await asyncio.gather(*[_one(p) for p in pairs])

    # (pair, setup) for every triggered candidate; primary = best per pair.
    all_c: List[Tuple[str, WaveSetup, bool]] = []
    for pair, res in scanned:
        if res is None:
            continue
        stats["pairs"] += 1
        for s in res.setups:
            all_c.append((pair, s, s is res.primary))
    stats["candidates"] = len(all_c)
    stats["eligible"] = sum(1 for _, s, _p in all_c if s.status == "eligible")
    if not all_c:
        log.info(f"🌊 Elliott: {stats['pairs']} pair(s) scanned, no triggered candidate")
        return stats

    # ── delivery selection: primary + eligible + family switch, best first ──
    can_send = (bool(cfg.ELLIOTT_WAVE_TELEGRAM_ENABLED) and not suppress)
    why_not: Dict[str, str] = {}      # event_id -> disposition for non-delivered rows
    deliverable: List[Tuple[str, WaveSetup]] = []
    for pair, s, is_primary in all_c:
        if s.status != "eligible":
            why_not[s.event_id] = s.status if s.status == "watch" else f"rejected:{s.reason}"
        elif not is_primary:
            why_not[s.event_id] = "not_primary"
        elif not _deliver_enabled(s.family):
            why_not[s.event_id] = "delivery_off_family"
        elif not can_send:
            why_not[s.event_id] = "delivery_off"
        else:
            deliverable.append((pair, s))
    deliverable.sort(key=lambda t: (-t[1].score, FAMILY_PRIORITY[t[1].family], t[0]))

    cap = int(cfg.ELLIOTT_WAVE_MAX_ALERTS_PER_RUN)
    selected: List[Tuple[str, WaveSetup]] = []
    duplicates: set = set()
    for pair, s in deliverable:
        if len(selected) >= cap:
            why_not[s.event_id] = "over_cap"
            continue
        try:
            owned = await _claim(sdb, "d", s.structure_id)
        except Exception as exc:
            log.warning(f"[{pair}] Elliott claim failed: {exc!r}")
            stats["errors"] += 1
            why_not[s.event_id] = "claim_error"
            continue
        if not owned:
            duplicates.add(s.event_id)
            stats["duplicates"] += 1
            continue
        selected.append((pair, s))

    delivered_ids: set = set()
    if selected:
        delivered_ids = await _deliver(selected, reference_time, sdb, tg, log, stats, why_not)

    # ── shadow recording of everything not delivered (once per structure) ──
    if cfg.ELLIOTT_WAVE_SHADOW_MODE:
        for pair, s, _p in all_c:
            if s.event_id in delivered_ids or s.event_id in duplicates:
                continue
            disp = why_not.get(s.event_id, "not_delivered")
            if disp.startswith("rejected:") and not cfg.ELLIOTT_WAVE_SHADOW_REJECTED:
                continue
            try:
                if not await _claim(sdb, "s", s.structure_id):
                    continue
                res_s = await sdb.record_shadow_pending_outcome(
                    **_outcome_kwargs(pair, s, disp))
                if res_s == "failed":
                    await _release(sdb, "s", s.structure_id)
                    stats["errors"] += 1
                elif res_s == "created":
                    stats["shadowed"] += 1
            except Exception as exc:
                log.warning(f"[{pair}] Elliott shadow record failed: {exc!r}")
                stats["errors"] += 1
                await _release(sdb, "s", s.structure_id)

    log.info(
        f"🌊 Elliott: {stats['pairs']} pair(s), {stats['candidates']} candidate(s), "
        f"{stats['eligible']} eligible, {stats['delivered']} delivered, "
        f"{stats['shadowed']} shadowed, {stats['duplicates']} duplicate(s)"
    )
    return stats


async def _deliver(selected: List[Tuple[str, WaveSetup]], reference_time: int,
                   sdb: RedisStateStore, tg: Any, log: logging.Logger,
                   stats: Dict[str, int], why_not: Dict[str, str]) -> set:
    """Send ONE Telegram message with the selected setups, then record them as
    Recorded outcomes. A failed send is parked in the dead-letter queue together
    with the outcome rows it owes; if even that fails the claims are released so
    the next run can try again."""
    message = format_elliott_message([s for _, s in selected])
    outcomes = [_outcome_kwargs(p, s, "delivered") for p, s in selected]
    sent = False
    try:
        sent = bool(await tg.send(message))
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        log.error(f"Elliott Telegram send raised: {exc!r}")
    if sent:
        for (pair, s), kw in zip(selected, outcomes):
            try:
                ok = await sdb.record_pending_outcome(**kw)
            except Exception as exc:
                log.error(f"[{pair}] Elliott outcome record raised: {exc!r}")
                ok = False
            if ok is False:
                stats["errors"] += 1
                log.error(f"[{pair}] Elliott alert sent but its outcome row could not be recorded")
        stats["delivered"] += len(selected)
        return {s.event_id for _, s in selected}

    queued = await queue_failed_alert(
        sdb, "ELLIOTT", message, int(reference_time), [], "elliott_wave", log,
        outcomes=outcomes,
    )
    if queued:
        # Delivery is owed and will be retried by the DLQ replay; keep the claims.
        for _, s in selected:
            why_not[s.event_id] = "send_failed_queued"
        return {s.event_id for _, s in selected}
    for _, s in selected:
        await _release(sdb, "d", s.structure_id)
        why_not[s.event_id] = "send_failed"
    stats["errors"] += 1
    return set()
