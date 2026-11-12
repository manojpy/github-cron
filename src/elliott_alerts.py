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
      - ELLIOTT_WAVE_MAX_ALERTS_PER_RUN, separate from MAX_ALERTS_PER_RUN;
      - ELLIOTT_WAVE_MAX_ALERTS_PER_PAIR inside ELLIOTT_WAVE_PAIR_WINDOW_HOURS.
  * Elliott setups must pass the SAME common conditions as the MACD alerts, per
    direction (ELLIOTT_WAVE_RESPECT_LEGACY_GATE): buy_trend_common_relaxed /
    sell_trend_common_relaxed (trend, trend gate, CPR, Ichimoku cloud when that
    filter is on) AND the wick-or-reversal-candle test, i.e. exactly
    buy_common_relaxed_wick / sell_common_relaxed_wick. The relaxed values are read
    from gates.LEGACY_COMMON_SNAPSHOT, written by the legacy gate evaluation of the
    same pair and candle (nothing is recomputed, so the two can never disagree).
    If the legacy gate was not evaluated for that candle the setup is vetoed
    ("fails closed"). A vetoed setup is still recorded as Shadowed, with the
    reason, so the value of the gate stays measurable.
  * Elliott never touches the legacy alert registry sets, the dedup of legacy
    alerts, the heat gate or the confluence score. Its outcome rows carry the wave
    score in the score fields and are skipped by the Brain's legacy readers, so
    the threshold learning of the MACD alerts is not mixed with Elliott scores.
  * A stored signal is immutable: the decision is taken on the last closed candle
    only, from data that existed at that candle (elliott_wave.analyze never reads
    beyond it), and nothing is rewritten later. What happens afterwards is
    appended as separate events by elliott_tracker.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import elliott_tracker as tracker
from alert_registry import ELLIOTT_REGISTRY
from alerts import escape_markdown_v2, queue_failed_alert
from bot_config import Constants, cfg, format_ist_time, json_dumps
from elliott_wave import (
    Analysis, ElliottParams, FEATURE_VERSION, WaveEvent, WaveSetup, analyze, family_rank,
)
from fetcher import detect_reversal_candle_pattern_with_context, get_last_closed_index_from_array
from gates import LEGACY_COMMON_SNAPSHOT
from state import RedisStateStore, RedisKeyPrefix, _rc

__all__ = ["params_from_cfg", "scan_pair", "legacy_gate", "run_elliott_scan", "format_elliott_message"]

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
        enable_exhaustion=bool(cfg.ELLIOTT_WAVE_ENABLE_EXHAUSTION_EVENTS),
        primary_family=str(cfg.ELLIOTT_WAVE_PRIMARY_ENTRY),
    )


def alert_key_of(s: WaveSetup) -> str:
    key = f"elliott_{s.family}_{s.direction}"
    if key not in ELLIOTT_REGISTRY:
        raise KeyError(f"unregistered Elliott alert key: {key}")
    return key


def _deliver_enabled(family: str) -> bool:
    return bool(getattr(cfg, f"ELLIOTT_WAVE_DELIVER_{family.upper()}", True))


def _jsonable(x: Any) -> Any:
    """numpy scalars / tuples / NaN -> plain JSON-safe values."""
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


# ── the legacy common-relaxed + wick gate ────────────────────────────────

def _first_failure(snap: Dict[str, Any], side: str) -> str:
    """Name the first legacy condition that failed, Ichimoku first (the one most
    often asked about), for the reason string and the Brain."""
    if not snap.get(f"ichimoku_ok_{side}", False):
        return "ichimoku"
    if not snap.get(f"base_{side}_trend", False):
        return "trend"
    if not snap.get(f"trend_gate_ok_{side}", False):
        return "trend_gate"
    if not snap.get("cpr_ok", False):
        return "cpr"
    return "relaxed_common"


def legacy_gate(pair: str, pd_15m: Any, i: int) -> Tuple[Optional[Dict[str, Optional[str]]], Dict[str, Dict[str, Any]]]:
    """Per-direction veto reason (None = passes) plus a detail record per direction.

    pass  <=>  buy_trend_common_relaxed AND (buy_wick_ratio <= MIN_WICK_RATIO OR
               bullish reversal candle)            -- i.e. buy_common_relaxed_wick
    (sell mirrored). Returns (None, {}) when the gate is switched off."""
    if not cfg.ELLIOTT_WAVE_RESPECT_LEGACY_GATE:
        return None, {}
    snap = LEGACY_COMMON_SNAPSHOT.get(pair)
    ts_i = int(pd_15m.ts[i])
    if not snap or snap.get("i15") != i or snap.get("ts") != ts_i:
        gone = {"state": "unavailable"}
        return ({"buy": "gate_unavailable", "sell": "gate_unavailable"},
                {"buy": dict(gone), "sell": dict(gone)})
    rev_bull, rev_bear, _name, _leg = detect_reversal_candle_pattern_with_context(pd_15m, i)
    veto: Dict[str, Optional[str]] = {}
    detail: Dict[str, Dict[str, Any]] = {}
    for side, reversal in (("buy", bool(rev_bull)), ("sell", bool(rev_bear))):
        ratio = float(snap[f"{side}_wick_ratio"])
        relaxed = bool(snap[f"{side}_relaxed"])
        wick_ok = bool(ratio <= Constants.MIN_WICK_RATIO or reversal)
        reason: Optional[str] = None
        if not relaxed:
            reason = _first_failure(snap, side)
        elif not wick_ok:
            reason = "wick"
        veto[side] = reason
        detail[side] = {
            "state": "pass" if reason is None else "veto",
            "common_relaxed": relaxed,
            "ichimoku": bool(snap[f"ichimoku_ok_{side}"]),
            "trend": bool(snap[f"base_{side}_trend"]),
            "trend_gate": bool(snap[f"trend_gate_ok_{side}"]),
            "cpr": bool(snap["cpr_ok"]),
            "wick_ratio": round(ratio, 4),
            "reversal_candle": reversal,
            "wick_ok": wick_ok,
        }
    return veto, detail


# ── per-pair scan (CPU only, no I/O) ─────────────────────────────────────

def scan_pair(pair: str, pd_15m: Any, reference_time: int,
              params: ElliottParams) -> Tuple[Optional[int], Optional[Analysis]]:
    """Analyse the last CLOSED 15m candle. Returns (index, analysis) or (None, None)
    when the candle is missing / stale / unstable (same rule as the legacy scan).
    The legacy common-relaxed + wick gate is applied inside the analysis as a
    per-direction veto, and its detail is attached to every setup / event."""
    if pd_15m is None or len(pd_15m) == 0:
        return None, None
    i = get_last_closed_index_from_array(pd_15m.ts, INTERVAL_MIN, reference_time, pair)
    if i is None:
        return None, None
    veto, detail = legacy_gate(pair, pd_15m, int(i))
    res = analyze(pair, pd_15m.ts, pd_15m.open, pd_15m.high, pd_15m.low,
                  pd_15m.close, pd_15m.volume, end=int(i), params=params, external_gate=veto)
    for s in res.setups:
        if detail:
            s.features["elliott_legacy_gate_detail"] = dict(detail.get(s.direction, {}))
    for ev in res.events:
        if detail:
            ev.features["elliott_legacy_gate_detail"] = dict(detail.get(ev.direction, {}))
    return int(i), res


# ── message ──────────────────────────────────────────────────────────────

def _px(x: float) -> str:
    ax = abs(x)
    if ax >= 1000:
        return f"{x:,.2f}"
    if ax >= 1:
        return f"{x:.4f}".rstrip("0").rstrip(".")
    return f"{x:.6g}"


def _gate_line(s: WaveSetup) -> Optional[str]:
    d = s.features.get("elliott_legacy_gate_detail") or {}
    if not d or d.get("state") != "pass":
        return None
    parts = ["common relaxed ✓"]
    if d.get("ichimoku") is not None:
        parts.append("Ichimoku ✓")
    parts.append("wick/reversal ✓")
    return "Legacy gate: " + " · ".join(parts)


def _setup_block(s: WaveSetup, verdict: Optional[str] = None) -> str:
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
    gl = _gate_line(s)
    if gl:
        lines.append(e(gl))
    if verdict:
        lines.append(e(verdict))
    return "\n".join(lines)


def format_elliott_message(setups: List[WaveSetup], verdicts: Optional[Dict[str, str]] = None) -> str:
    e = escape_markdown_v2
    verdicts = verdicts or {}
    head = e(f"🌊 {cfg.BOT_NAME} — Elliott Wave ({len(setups)})")
    body = "\n\n".join(_setup_block(s, verdicts.get(s.family)) for s in setups)
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
        # own feature namespace (fv1 is the legacy one and is deliberately not bumped)
        "feature_version": FEATURE_VERSION,
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


def _snapshot(s: WaveSetup, disposition: str) -> Dict[str, Any]:
    """The immutable signal snapshot archived by the tracker: the context plus the
    full per-pillar score breakdown (score / max / reason / values / available)."""
    snap = _context(s, disposition)
    snap["elliott_score"] = float(s.score)
    snap["elliott_status"] = s.status
    snap["elliott_reason"] = s.reason
    snap["elliott_rr"] = float(s.rr)
    snap["elliott_components"] = {
        k: {"score": v.score, "max": v.max_score, "available": v.available,
            "reason": v.reason, "values": v.values}
        for k, v in s.detail.items()
    }
    return _jsonable(snap)


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
        "context": _jsonable(_context(s, disposition)),
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


async def _claim_pair_slot(sdb: RedisStateStore, pair: str) -> Optional[int]:
    """Rolling per-pair limit: ELLIOTT_WAVE_MAX_ALERTS_PER_PAIR slots, each held
    for ELLIOTT_WAVE_PAIR_WINDOW_HOURS. Returns the slot number or None if full."""
    ttl = int(cfg.ELLIOTT_WAVE_PAIR_WINDOW_HOURS) * 3600
    for k in range(int(cfg.ELLIOTT_WAVE_MAX_ALERTS_PER_PAIR)):
        if await sdb.claim_notice(f"ewp:{pair}:{k}", ttl):
            return k
    return None


async def _release_pair_slot(sdb: RedisStateStore, pair: str, k: int) -> None:
    key = f"{sdb.meta_prefix}notice:ewp:{pair}:{k}"
    try:
        await sdb._safe_redis_op(lambda: _rc(sdb._redis).delete(key), 2.0, "ew_release:pair")
    except Exception:
        pass


async def _claim_many(
    sdb: RedisStateStore,
    kind: str,
    structure_ids: List[str],
) -> Dict[str, bool]:
    """Batch SET NX EX for many structure claims in one pipeline.

    Returns {structure_id: owned}. Empty dict on Redis failure (caller should
    fall back to sequential _claim or treat everything as not-owned).
    """
    if not structure_ids or sdb.degraded or not sdb._redis:
        return {}
    ttl = int(cfg.ELLIOTT_WAVE_EVENT_TTL_HOURS) * 3600
    prefix = f"{sdb.meta_prefix}notice:ew{kind}:"
    try:
        client = _rc(sdb._redis)
        async with client.pipeline(transaction=False) as pipe:
            for sid in structure_ids:
                pipe.set(f"{prefix}{sid}", "1", nx=True, ex=ttl)
            results = await asyncio.wait_for(pipe.execute(), timeout=3.0)
        return {sid: bool(ok) for sid, ok in zip(structure_ids, results)}
    except Exception:
        return {}

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
    setups (caps + dedup) and record the rest as Shadowed. Then advance the
    follow-up of every tracked structure. Never raises."""
    stats = {"pairs": 0, "candidates": 0, "eligible": 0, "delivered": 0,
             "shadowed": 0, "duplicates": 0, "gated": 0, "pair_limited": 0,
             "events": 0, "tracked": 0, "resolved": 0, "errors": 0}
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


def _vol_ref(parsed_cache: Dict[str, Tuple[Any, Any, Any]], pair: str, idx: int) -> Optional[float]:
    pd_15m = (parsed_cache.get(pair) or (None, None, None))[0]
    return None if pd_15m is None else tracker.vol_ref_at(pd_15m.volume, idx)


def _track_record(pair: str, s: WaveSetup, disposition: str, vol_ref: Optional[float]) -> Dict[str, Any]:
    return tracker.make_record(
        kind="setup", pair=pair, alert_key=alert_key_of(s), family=s.family,
        direction=s.direction, signal_ts=s.signal_ts, entry=s.entry_price, atr=s.atr,
        invalidation=s.invalidation, stop=s.stop, targets=s.targets,
        vol_ref=vol_ref,
        event_id=s.event_id, structure_id=s.structure_id, disposition=disposition,
        status=s.status, score=s.score,
        gate_state=str(s.features.get("elliott_legacy_gate", "n/a")), reason=s.reason)


async def _run(parsed_cache, pairs, reference_time, sdb, tg, log, suppress, stats) -> Dict[str, int]:
    if cfg.DRY_RUN_MODE:
        log.info("🌊 Elliott: DRY_RUN_MODE — scan skipped (nothing is claimed or recorded)")
        return stats
    if sdb.degraded:
        log.warning("🌊 Elliott: Redis unavailable — scan skipped (no way to deduplicate)")
        return stats

    params = params_from_cfg()
    sem = asyncio.Semaphore(max(1, int(cfg.EVAL_CONCURRENCY_LIMIT)))
    last_idx: Dict[str, int] = {}

    async def _one(pair: str):
        async with sem:
            pd_15m = (parsed_cache.get(pair) or (None, None, None))[0]
            try:
                i, res = await asyncio.to_thread(scan_pair, pair, pd_15m, reference_time, params)
            except Exception as exc:
                log.warning(f"[{pair}] Elliott analysis failed: {exc!r}")
                stats["errors"] += 1
                return pair, None
            if i is not None:
                last_idx[pair] = i
            return pair, res

    scanned = await asyncio.gather(*[_one(p) for p in pairs])

    # (pair, setup, is_primary) for every triggered candidate; primary = best per pair.
    all_c: List[Tuple[str, WaveSetup, bool]] = []
    events: List[Tuple[str, WaveEvent]] = []
    for pair, res in scanned:
        if res is None:
            continue
        stats["pairs"] += 1
        for s in res.setups:
            all_c.append((pair, s, s is res.primary))
        for ev in res.events:
            events.append((pair, ev))
    stats["candidates"] = len(all_c)
    stats["eligible"] = sum(1 for _, s, _p in all_c if s.status == "eligible")
    stats["gated"] = sum(1 for _, s, _p in all_c if str(s.reason).startswith("legacy_gate:"))
    stats["events"] = len(events)

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
    primary_family = str(cfg.ELLIOTT_WAVE_PRIMARY_ENTRY)
    deliverable.sort(key=lambda t: (-t[1].score, family_rank(t[1].family, primary_family), t[0]))

    cap = int(cfg.ELLIOTT_WAVE_MAX_ALERTS_PER_RUN)
    selected: List[Tuple[str, WaveSetup]] = []
    slots: Dict[str, Tuple[str, int]] = {}      # event_id -> (pair, slot)
    duplicates: set = set()

    # Pre-claim structures for the candidates we might actually take.
    # Cap is small (default 3), so we only need the first few after sorting.
    # Batching turns N sequential SET NX into one pipeline.
    candidates = deliverable[: max(cap * 3, cap + 5)]  # small buffer for dups / pair limits
    claim_map = await _claim_many(sdb, "d", [s.structure_id for _, s in candidates])

    for pair, s in deliverable:
        if len(selected) >= cap:
            why_not[s.event_id] = "over_cap"
            continue
        try:
            # Prefer the batched result; fall back to a live claim if the batch
            # missed this structure (e.g. buffer too small or Redis hiccup).
            if s.structure_id in claim_map:
                owned = claim_map[s.structure_id]
            else:
                owned = await _claim(sdb, "d", s.structure_id)
            if not owned:
                duplicates.add(s.event_id)
                stats["duplicates"] += 1
                continue
            # Pair-slot claim stays sequential: MAX_ALERTS_PER_PAIR is 1 and
            # the total number of selected is tiny (≤ cap). Cost is negligible.
            slot = await _claim_pair_slot(sdb, pair)
        except Exception as exc:
            log.warning(f"[{pair}] Elliott claim failed: {exc!r}")
            stats["errors"] += 1
            why_not[s.event_id] = "claim_error"
            # If we already own the structure claim from the batch, release it
            # so the next run can try again.
            if s.structure_id in claim_map and claim_map[s.structure_id]:
                await _release(sdb, "d", s.structure_id)
            continue
        if slot is None:
            # the structure stays claimed: it is not delivered later, only recorded
            why_not[s.event_id] = "pair_limit"
            stats["pair_limited"] += 1
            continue
        slots[s.event_id] = (pair, slot)
        selected.append((pair, s))

    delivered_ids: set = set()
    if selected:
        verdicts: Dict[str, str] = {}
        for fam in {s.family for _, s in selected}:
            verdicts[fam] = await tracker.verdict_for(sdb, fam)
        delivered_ids = await _deliver(selected, reference_time, sdb, tg, log, stats, why_not, verdicts)
        for _, s in selected:
            if s.event_id not in delivered_ids and s.event_id in slots:
                await _release_pair_slot(sdb, *slots[s.event_id])

    # ── shadow recording of everything not delivered (once per structure) ──
    # Collect candidates first, then claim + write in two pipelines so we pay
    # ~2 RTTs instead of 2–4 × N sequential round-trips.
    recorded_for_tracking: List[Tuple[str, WaveSetup, str]] = []
    shadow_candidates: List[Tuple[str, WaveSetup, str]] = []  # (pair, setup, disp)

    for pair, s, _p in all_c:
        if s.event_id in delivered_ids:
            recorded_for_tracking.append((pair, s, why_not.get(s.event_id, "delivered")))
            continue
        if s.event_id in duplicates:
            continue
        disp = why_not.get(s.event_id, "not_delivered")
        if not cfg.ELLIOTT_WAVE_SHADOW_MODE:
            continue
        if disp.startswith("rejected:") and not cfg.ELLIOTT_WAVE_SHADOW_REJECTED:
            continue
        shadow_candidates.append((pair, s, disp))

    if shadow_candidates and not sdb.degraded and sdb._redis:
        claim_map = await _claim_many(
            sdb, "s", [s.structure_id for _, s, _ in shadow_candidates]
        )
        owned_shadows: List[Tuple[str, WaveSetup, str]] = [
            (pair, s, disp)
            for pair, s, disp in shadow_candidates
            if claim_map.get(s.structure_id)
        ]

        if owned_shadows:
            try:
                client = _rc(sdb._redis)
                async with client.pipeline(transaction=False) as pipe:
                    for pair, s, disp in owned_shadows:
                        key = (
                            f"{RedisKeyPrefix.SHADOW_PENDING}"
                            f"{pair}:{alert_key_of(s)}:{int(s.signal_ts)}"
                        )
                        payload = {
                            "direction": s.direction,
                            "entry_ts": int(s.signal_ts),
                            "entry_price": float(s.entry_price),
                            "confluence_score": float(s.score),
                            "confluence_total": 100.0,
                            "confluence_votes": None,
                            "context": {
                                **_jsonable(_context(s, disp)),
                                "cooldown_active": False,
                                "cooldown_hit_ts": None,
                            },
                        }
                        ttl_out = max(
                            (cfg.OUTCOME_LOOKAHEAD_CANDLES + 4) * 15 * 60,
                            24 * 3600,
                        )
                        pipe.set(key, json_dumps(payload), ex=ttl_out)
                    await asyncio.wait_for(pipe.execute(), timeout=3.0)

                for pair, s, disp in owned_shadows:
                    recorded_for_tracking.append((pair, s, disp))
                    stats["shadowed"] += 1
                    log.info(
                        f"[{pair}] Shadow pending created: {alert_key_of(s)} "
                        f"@ {int(s.signal_ts)} → "
                        f"shadow_pending:{pair}:{alert_key_of(s)}:{int(s.signal_ts)}"
                    )
            except Exception as exc:
                log.warning(f"Elliott batch shadow record failed: {exc!r}")
                stats["errors"] += 1
                for _, s, _ in owned_shadows:
                    await _release(sdb, "s", s.structure_id)

        elif not claim_map and shadow_candidates:
            # Batch claim failed (empty map) — fall back to sequential so we
            # do not silently drop every shadow for the run.
            for pair, s, disp in shadow_candidates:
                try:
                    if not await _claim(sdb, "s", s.structure_id):
                        continue
                    res_s = await sdb.record_shadow_pending_outcome(
                        **_outcome_kwargs(pair, s, disp)
                    )
                    if res_s == "failed":
                        await _release(sdb, "s", s.structure_id)
                        stats["errors"] += 1
                    elif res_s == "created":
                        recorded_for_tracking.append((pair, s, disp))
                        stats["shadowed"] += 1
                except Exception as e2:
                    log.warning(f"[{pair}] Elliott shadow record failed: {e2!r}")
                    stats["errors"] += 1
                    await _release(sdb, "s", s.structure_id)
    else:
        for pair, s, disp in shadow_candidates:
            recorded_for_tracking.append((pair, s, disp))

    # ── follow-up: register new structures / events, then advance all of them ──
    for pair, s, disp in recorded_for_tracking:
        try:
            rec = _track_record(pair, s, disp, _vol_ref(parsed_cache, pair, s.signal_idx))
            await tracker.register(sdb, rec, _snapshot(s, disp), log)
        except Exception as exc:
            log.warning(f"[{pair}] Elliott tracker register failed: {exc!r}")
    for pair, ev in events:
        try:
            rec = tracker.make_record(
                kind="event", pair=pair, alert_key=f"elliott_exhaustion_{ev.direction}",
                family="exhaustion", direction=ev.direction, signal_ts=ev.signal_ts,
                entry=ev.entry_price, atr=ev.atr, invalidation=ev.invalidation, stop=None,
                targets=ev.targets, vol_ref=_vol_ref(parsed_cache, pair, ev.signal_idx),
                event_id=ev.event_id, structure_id=ev.structure_id, disposition="event",
                status="event", score=float(ev.score),
                gate_state=str(ev.features.get("elliott_legacy_gate", "n/a")), reason="")
            snap = _jsonable({**ev.features, "feature_version": FEATURE_VERSION,
                              "elliott_signs": ev.signs, "elliott_labels":
                              {k: list(v) for k, v in ev.labels.items()}})
            await tracker.register(sdb, rec, snap, log)
        except Exception as exc:
            log.warning(f"[{pair}] Elliott event register failed: {exc!r}")

    # One TTL refresh for the whole track hash (replaces the old per-register EXPIRE)
    try:
        await tracker.ensure_track_ttl(sdb)
    except Exception:
        pass

    prog = await tracker.update_all(sdb, parsed_cache, last_idx, log)
    stats["tracked"], stats["resolved"] = prog["tracked"], prog["resolved"]
    log.info(
        f"🌊 Elliott: {stats['pairs']} pair(s), {stats['candidates']} candidate(s), "
        f"{stats['eligible']} eligible, {stats['gated']} vetoed by legacy gate, "
        f"{stats['delivered']} delivered, {stats['shadowed']} shadowed, "
        f"{stats['duplicates']} duplicate(s), {stats['pair_limited']} pair-limited, "
        f"{stats['events']} exhaustion event(s), {stats['tracked']} tracked, "
        f"{stats['resolved']} resolved"
    )
    return stats


async def _deliver(selected: List[Tuple[str, WaveSetup]], reference_time: int,
                   sdb: RedisStateStore, tg: Any, log: logging.Logger,
                   stats: Dict[str, int], why_not: Dict[str, str],
                   verdicts: Optional[Dict[str, str]] = None) -> set:
    """Send ONE Telegram message with the selected setups, then record them as
    Recorded outcomes. A failed send is parked in the dead-letter queue together
    with the outcome rows it owes; if even that fails the claims are released so
    the next run can try again."""
    message = format_elliott_message([s for _, s in selected], verdicts)
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
