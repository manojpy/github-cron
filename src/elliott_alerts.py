"""elliott_alerts — run-level wiring for the Elliott Wave module.

Sits between the pure engine (elliott_wave.py) and the bot's infrastructure
(Redis state, Telegram, outcome tracking). Called ONCE per run, after the legacy
alerts, the pullback WATCH message and the heat-gate release have finished, so it
cannot change any legacy decision.

Isolation, by design (do not "fix" any of these):
  Elliott rows bypass the legacy one-open-trade slot and the re-entry cooldown
  (see RedisStateStore.record_pending_outcome / record_shadow_pending_outcome /
  get_active_trade). A MACD trade never blocks a wave count and vice versa.
  The bypass does NOT remove Elliott's own protections:
    one Telegram alert and one Recorded row per wave structure
    (`structure_id`, claimed in Redis for ELLIOTT_WAVE_EVENT_TTL_HOURS);
    one Shadowed row per wave structure (separate claim);
    ELLIOTT_WAVE_MAX_ALERTS_PER_RUN, separate from MAX_ALERTS_PER_RUN;
    ELLIOTT_WAVE_MAX_ALERTS_PER_PAIR inside ELLIOTT_WAVE_PAIR_WINDOW_HOURS.

Elliott setups must pass the SAME common conditions as the MACD alerts, per
direction (ELLIOTT_WAVE_RESPECT_LEGACY_GATE): buy_trend_common_relaxed /
sell_trend_common_relaxed (trend, trend gate, CPR, Ichimoku cloud when that
filter is on) AND the wick-or-reversal-candle test, i.e. exactly
buy_common_relaxed_wick / sell_common_relaxed_wick. The relaxed values are read
from gates.LEGACY_COMMON_SNAPSHOT, written by the legacy gate evaluation of the
same pair and candle (nothing is recomputed, so the two can never disagree).
If the legacy gate was not evaluated for that candle the setup is vetoed
("fails closed"). A vetoed setup is still recorded as Shadowed, with the
reason, so the value of the gate stays measurable.

Elliott never touches the legacy alert registry sets, the dedup of legacy
alerts, the heat gate or the confluence score. Its outcome rows carry the wave
score in the score fields and are skipped by the Brain's legacy readers, so
the threshold learning of the MACD alerts is not mixed with Elliott scores.

A stored signal is immutable: the decision is taken on the last closed candle
only, from data that existed at that candle (elliott_wave.analyze never reads
beyond it), and nothing is rewritten later. What happens afterwards is
appended as separate events by elliott_tracker.

Redis cost (the point of this revision): a round-trip to Aiven is ~250 ms, so
this module spends a fixed, small number of round-trips regardless of how many
pairs / candidates there are:
  1 prefetch pipeline, launched before the CPU scan (latency hidden behind it)
  1 pipeline per delivery chunk of at most `cap` candidates (0 when nothing is
    eligible, which is the common case)
  1 pipeline for all shadow claims
  1 pipeline for all shadow rows
  1 pipeline for all tracker registrations
  1 pipeline for the tracker follow-up
  1 pipeline for all claim releases, only when something must be released
Claims are still taken ONLY for structures that can actually be delivered, so a
setup that was over cap is still deliverable on a later run.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import elliott_tracker as tracker
from alert_registry import ELLIOTT_REGISTRY, is_elliott_key
from alerts import escape_markdown_v2, queue_failed_alert
from bot_config import Constants, cfg, format_ist_time, json_dumps, json_loads, logger
from brain_helpers import EVIDENCE_CARDS_KEY, evidence_block_lines, evidence_card_lookup
from elliott_wave import (
    Analysis, ElliottParams, FEATURE_VERSION, WaveEvent, WaveSetup, analyze, family_rank,
)
from fetcher import (
    detect_reversal_candle_pattern_with_context, get_last_closed_index_from_array,
)
from gates import LEGACY_COMMON_SNAPSHOT
from state import RedisKeyPrefix, RedisStateStore, _execute_pipeline, _rc

__all__ = ["params_from_cfg", "scan_pair", "legacy_gate", "run_elliott_scan",
           "format_elliott_message", "alert_key_of", "deliver_enabled"]

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


def deliver_enabled(family: str) -> bool:
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


def legacy_gate(pair: str, pd_15m: Any, i: int
                ) -> Tuple[Optional[Dict[str, Optional[str]]], Dict[str, Dict[str, Any]]]:
    """Per-direction veto reason (None = passes) plus a detail record per direction.

    pass  <=>  buy_trend_common_relaxed AND (buy_wick_ratio <= MIN_WICK_RATIO OR
               bullish reversal candle)          -- i.e. buy_common_relaxed_wick
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
                  pd_15m.close, pd_15m.volume, end=int(i), params=params,
                  external_gate=veto)
    for s in res.setups:
        if detail:
            s.features["elliott_legacy_gate_detail"] = dict(detail.get(s.direction, {}))
    for ev in res.events:
        if detail:
            ev.features["elliott_legacy_gate_detail"] = dict(detail.get(ev.direction, {}))
    return int(i), res


# ── message ────────────────────────────────────────────────────────────
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


def _setup_block(s: WaveSetup, verdict: Optional[str] = None,
                 evidence_lines: Optional[List[str]] = None) -> str:
    e = escape_markdown_v2
    side = "BUY" if s.direction == "buy" else "SELL"
    icon = "🟢" if s.direction == "buy" else "🔴"
    f = s.features
    htf = {1: "aligned", -1: "against", 0: "flat", None: "n/a"}.get(s.htf_bias, "n/a")
    comp = " · ".join(
        f"{k.replace('_', ' ')} {v.score:.0f}/{v.max_score:.0f}"
        + ("" if v.available else "*")
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
    if evidence_lines:
        lines.extend(e(x) for x in evidence_lines)
    return "\n".join(lines)


def _evidence_lines_for(s: WaveSetup, cards: Optional[Dict[str, Optional[Dict[str, Any]]]]) -> List[str]:
    """Informational evidence block for one setup. cards=None means 'not requested'. Never raises:
    a failure here only drops the block, never the alert."""
    if cards is None:
        return []
    try:
        return evidence_block_lines(cards.get(alert_key_of(s)), alert_key_of(s), s.direction)
    except Exception as exc:
        logger.debug(f"Elliott evidence block skipped: {exc!r}")
        return []


def format_elliott_message(setups: List[WaveSetup],
                           verdicts: Optional[Dict[str, str]] = None,
                           evidence_cards: Optional[Dict[str, Optional[Dict[str, Any]]]] = None) -> str:
    e = escape_markdown_v2
    verdicts = verdicts or {}
    head = e(f"🌊 {cfg.BOT_NAME} — Elliott Wave ({len(setups)})")
    body = "\n\n".join(
        _setup_block(s, verdicts.get(s.family), _evidence_lines_for(s, evidence_cards)) for s in setups)
    ts = max(s.signal_ts for s in setups) + INTERVAL_MIN * 60
    foot = (f"📆 {e(format_ist_time(ts, '%d-%m-%Y'))} "
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


# ── claims ───────────────────────────────────────────────────────────────
# A claim's value is this run's token, not "1". Attempt 1 is a plain SET NX EX
# (one command per name, exactly like claim_notice). Only if that pipeline's
# reply is lost does attempt 2 run the Lua below, which treats a key already
# holding OUR token as owned instead of mistaking it for a duplicate. A release
# only deletes a claim carrying our token, so it can never undo another run's
# claim. Same pattern as RedisStateStore.claim_recent_alert.
_CLAIM_LUA = """
local v = redis.call("GET", KEYS[1])
if v then
    if v == ARGV[1] then return 1 end
    return 0
end
redis.call("SET", KEYS[1], ARGV[1], "EX", ARGV[2])
return 1
"""
_RELEASE_LUA = """
if redis.call("GET", KEYS[1]) == ARGV[1] then
    return redis.call("DEL", KEYS[1])
end
return 0
"""


def _claim_value(sdb: RedisStateStore) -> str:
    return f"ew:{sdb._claim_token}"


def _notice_key(sdb: RedisStateStore, name: str) -> str:
    """Same keyspace as RedisStateStore.claim_notice (metadata:notice:<name>),
    so claims written by an older version are still honoured as held."""
    return f"{sdb.meta_prefix}notice:{name}"


def _event_ttl() -> int:
    return int(cfg.ELLIOTT_WAVE_EVENT_TTL_HOURS) * 3600


async def _claim_many(sdb: RedisStateStore, names: List[str], ttl_sec: int
                      ) -> Optional[List[bool]]:
    """Claim many notices in ONE pipeline. Returns a list of owned flags aligned
    with `names`, or None when the claim state is UNKNOWN (Redis trouble) — the
    caller must then fall back to a single claim, never assume "duplicate".
    A name repeated inside one batch is owned by its first occurrence only."""
    if not names:
        return []
    if sdb.degraded or not sdb._redis:
        return None
    val = _claim_value(sdb)
    keys = [_notice_key(sdb, n) for n in names]
    for attempt in (1, 2):
        try:
            async with _rc(sdb._redis).pipeline(transaction=False) as pipe:
                for k in keys:
                    if attempt == 1:
                        pipe.set(k, val, nx=True, ex=int(ttl_sec))
                    else:
                        pipe.eval(_CLAIM_LUA, 1, k, val, str(int(ttl_sec)))
                res = await asyncio.wait_for(_execute_pipeline(pipe), timeout=4.0)
            if not isinstance(res, (list, tuple)) or len(res) != len(names):
                raise RuntimeError("claim pipeline returned an unexpected result count")
            seen: set = set()
            out: List[bool] = []
            for n, r in zip(names, res):
                owned = bool(int(r or 0)) and n not in seen
                seen.add(n)
                out.append(owned)
            return out
        except Exception as exc:
            if attempt == 2:
                logger.warning(
                    f"Elliott batch claim failed for {len(names)} name(s): {exc!r}")
    return None


async def _claim_name(sdb: RedisStateStore, name: str, ttl_sec: int) -> Optional[bool]:
    """Single claim, same semantics as _claim_many. None = unknown (fail closed:
    an alert must not be sent when its dedup claim could not be established)."""
    res = await _claim_many(sdb, [name], ttl_sec)
    return None if res is None else res[0]


async def _release_many(sdb: RedisStateStore, names: List[str]) -> None:
    """Release every claim this run decided to give back, in ONE pipeline."""
    if not names or sdb.degraded or not sdb._redis:
        return
    val = _claim_value(sdb)
    try:
        async with _rc(sdb._redis).pipeline(transaction=False) as pipe:
            for n in dict.fromkeys(names):
                pipe.eval(_RELEASE_LUA, 1, _notice_key(sdb, n), val)
            await asyncio.wait_for(_execute_pipeline(pipe), timeout=3.0)
    except Exception as exc:
        logger.warning(f"Elliott claim release failed for {len(names)} name(s): {exc!r}")


async def _claim_pair_slots(sdb: RedisStateStore, pairs: List[str]
                            ) -> Dict[str, Optional[int]]:
    """Rolling per-pair limit for several pairs at once:
    ELLIOTT_WAVE_MAX_ALERTS_PER_PAIR slots, each held for
    ELLIOTT_WAVE_PAIR_WINDOW_HOURS. Slot 0 is tried for every pair in one
    pipeline; only pairs whose slot 0 was taken go on to slot 1, and so on. With
    MAX_ALERTS_PER_PAIR=1 that is exactly one command per pair in one RTT, and
    no slot is ever claimed that we do not use. An UNKNOWN claim state leaves
    the pair unslotted (fail closed) rather than trying further slots."""
    k_max = max(0, int(cfg.ELLIOTT_WAVE_MAX_ALERTS_PER_PAIR))
    ttl = int(cfg.ELLIOTT_WAVE_PAIR_WINDOW_HOURS) * 3600
    uniq = list(dict.fromkeys(pairs))
    slots: Dict[str, Optional[int]] = {p: None for p in uniq}
    if not uniq or k_max == 0:
        return slots
    pending = list(uniq)
    for k in range(k_max):
        if not pending:
            break
        names = [f"ewp:{p}:{k}" for p in pending]
        flags = await _claim_many(sdb, names, ttl)
        still: List[str] = []
        for pos, p in enumerate(pending):
            got = flags[pos] if flags is not None else None
            if got is None:
                got = await _claim_name(sdb, names[pos], ttl)
            if got is True:
                slots[p] = k
            elif got is False:
                still.append(p)
        pending = still
    return slots


# ── pending-outcome rows, batched ────────────────────────────────────────
# The two builders below mirror RedisStateStore.record_shadow_pending_outcome /
# record_pending_outcome EXACTLY for Elliott keys, which take the documented
# bypass path: no cooldown read, no "one open trade per pair" slot, no vote-count
# history (Elliott passes no confluence votes). For those keys the store performs
# exactly ONE Redis command — a plain SET on a deterministic key with the TTL
# below — so batching it is behaviour-preserving.
# Guards that must stay in sync with state.py:
#   shadow rows   require ENABLE_BRAIN (else the store returns "disabled")
#   recorded rows require ENABLE_WIN_RATE_FILTER (else nothing is owed)
#   non-Elliott alert keys are routed to the store, never mirrored
# If the batch write fails for ANY reason we fall back to the store method row by
# row, so a future schema change in state.py degrades speed, not correctness.
def _pending_ttl() -> int:
    return max((int(cfg.OUTCOME_LOOKAHEAD_CANDLES) + 4) * 15 * 60, 24 * 3600)


def _shadow_entry(pair: str, s: WaveSetup, disp: str) -> Tuple[str, str, int, float]:
    ak = alert_key_of(s)
    ts = int(s.signal_ts)
    key = f"{RedisKeyPrefix.SHADOW_PENDING}{pair}:{ak}:{ts}"
    payload = json_dumps({
        "direction": s.direction,
        "entry_ts": ts,
        "entry_price": float(s.entry_price),
        "confluence_score": float(s.score),
        "confluence_total": 100.0,
        "confluence_votes": None,
        "context": {
            **_jsonable(_context(s, disp)),
            "cooldown_active": False,      # Elliott never reads the legacy cooldown
            "cooldown_hit_ts": None,
        },
    })
    return key, payload, _pending_ttl(), float(s.score) / 100.0


def _recorded_entry(pair: str, s: WaveSetup, disp: str) -> Tuple[str, str, int]:
    ak = alert_key_of(s)
    ts = int(s.signal_ts)
    key = f"{RedisKeyPrefix.OUTCOME_PENDING}{pair}:{ak}:{ts}"
    payload = json_dumps({
        "direction": s.direction,
        "entry_ts": ts,
        "entry_price": float(s.entry_price),
        "confluence_score": float(s.score),
        "confluence_total": 100.0,
        "confluence_votes": None,
        "adx_val": None,
        "context": _jsonable(_context(s, disp)),
        "signal_price": None,
        "fill_price": None,
        "fees_paid_pct": None,
        "effective_score": None,
        "effective_required": None,
        "macro_multiplier": None,
        "cluster_penalty": None,
        "gate_passed": None,
    })
    return key, payload, _pending_ttl()


async def _write_entries(sdb: RedisStateStore,
                         entries: List[Tuple[str, str, int]]) -> bool:
    """All pending rows in ONE pipeline. False = caller must fall back."""
    if not entries:
        return True
    if sdb.degraded or not sdb._redis:
        return False
    try:
        async with _rc(sdb._redis).pipeline(transaction=False) as pipe:
            for key, payload, ttl in entries:
                pipe.set(key, payload, ex=ttl)
            res = await asyncio.wait_for(_execute_pipeline(pipe), timeout=4.0)
        if not isinstance(res, (list, tuple)) or len(res) != len(entries):
            return False
        return not any(isinstance(r, BaseException) for r in res)
    except Exception as exc:
        logger.warning(
            f"Elliott pending-row pipeline failed ({len(entries)} row(s)): {exc!r}")
        return False


async def _record_shadow_batch(sdb: RedisStateStore,
                               owned: List[Tuple[str, WaveSetup, str]],
                               log: logging.Logger, stats: Dict[str, int]) -> List[str]:
    """Write the shadow rows for structures whose shadow claim we own.
    Returns the shadow-claim names that must be released (failed writes)."""
    if not owned:
        return []
    # Store parity: with the Brain off, record_shadow_pending_outcome returns
    # "disabled" — the claim is kept, no row is written, tracking still happens.
    if not getattr(cfg, "ENABLE_BRAIN", False):
        return []

    mirror: List[Tuple[str, WaveSetup, str]] = []
    entries: List[Tuple[str, str, int]] = []
    meta: List[Tuple[str, str, int, float]] = []
    for pair, s, disp in owned:
        if not is_elliott_key(alert_key_of(s)):
            continue                      # not ours to mirror: use the store
        key, payload, ttl, ratio = _shadow_entry(pair, s, disp)
        mirror.append((pair, s, disp))
        entries.append((key, payload, ttl))
        meta.append((pair, key, int(s.signal_ts), ratio))

    releases: List[str] = []
    if mirror and await _write_entries(sdb, entries):
        book = getattr(sdb, "_shadow_registered_this_run", None)
        for (pair, s, disp), (p, key, ts, ratio) in zip(mirror, meta):
            if book is not None:
                book[p] = (ts, ratio, key)
            stats["shadowed"] += 1
            log.info(f"[{pair}] Shadow pending created: {alert_key_of(s)} @ {ts} → {key}")
        # Compare by structure_id, never by tuple equality: WaveSetup.__eq__ can
        # raise on array-bearing fields, and identity is not what we mean here.
        mirrored_ids = {s.structure_id for _, s, _ in mirror}
        owned = [t for t in owned if t[1].structure_id not in mirrored_ids]

    # Anything the mirror did not handle (or a failed pipeline): authoritative path.
    for pair, s, disp in owned:
        try:
            res_s = await sdb.record_shadow_pending_outcome(**_outcome_kwargs(pair, s, disp))
            if res_s == "failed":
                releases.append(f"ews:{s.structure_id}")
                stats["errors"] += 1
            elif res_s == "created":
                stats["shadowed"] += 1
        except Exception as exc:
            log.warning(f"[{pair}] Elliott shadow record failed: {exc!r}")
            stats["errors"] += 1
            releases.append(f"ews:{s.structure_id}")
    return releases


async def _record_delivered_batch(sdb: RedisStateStore,
                                  selected: List[Tuple[str, WaveSetup]],
                                  outcomes: List[Dict[str, Any]],
                                  log: logging.Logger) -> List[str]:
    """Recorded rows for the delivered setups. Returns the pairs whose row could
    not be recorded (the alert already went out, so this is an error, not a retry)."""
    if not getattr(cfg, "ENABLE_WIN_RATE_FILTER", False):
        return []                          # store parity: nothing is owed
    mirror: List[Tuple[str, WaveSetup]] = []
    entries: List[Tuple[str, str, int]] = []
    for (pair, s), kw in zip(selected, outcomes):
        if not is_elliott_key(str(kw.get("alert_key", ""))):
            continue
        mirror.append((pair, s))
        entries.append(_recorded_entry(pair, s, "delivered"))
    if mirror and await _write_entries(sdb, entries):
        mirrored_ids = {s.structure_id for _, s in mirror}
        left = [(p, s) for p, s in selected if s.structure_id not in mirrored_ids]
    else:
        left = list(selected)
    failed: List[str] = []
    for pair, s in left:
        try:
            ok = await sdb.record_pending_outcome(**_outcome_kwargs(pair, s, "delivered"))
        except Exception as exc:
            log.error(f"[{pair}] Elliott outcome record raised: {exc!r}")
            ok = False
        if ok is False:
            failed.append(pair)
    return failed


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


def _vol_ref(parsed_cache: Dict[str, Tuple[Any, Any, Any]], pair: str,
             idx: int) -> Optional[float]:
    pd_15m = (parsed_cache.get(pair) or (None, None, None))[0]
    return None if pd_15m is None else tracker.vol_ref_at(pd_15m.volume, idx)


def _track_record(pair: str, s: WaveSetup, disposition: str,
                  vol_ref: Optional[float]) -> Dict[str, Any]:
    return tracker.make_record(
        kind="setup", pair=pair, alert_key=alert_key_of(s), family=s.family,
        direction=s.direction, signal_ts=s.signal_ts, entry=s.entry_price, atr=s.atr,
        invalidation=s.invalidation, stop=s.stop, targets=s.targets, vol_ref=vol_ref,
        event_id=s.event_id, structure_id=s.structure_id, disposition=disposition,
        status=s.status, score=s.score,
        gate_state=str(s.features.get("elliott_legacy_gate", "n/a")), reason=s.reason)


async def _prefetch_evidence(sdb: RedisStateStore) -> Optional[Dict[str, Any]]:
    """The Brain's evidence-cards blob for the informational Telegram block. One GET, started
    before the CPU scan so its round trip is hidden. None on any failure; never raises."""
    if not getattr(cfg, "ALERT_EVIDENCE_BLOCK", True):
        return None
    try:
        raw = await sdb._safe_redis_op(lambda: _rc(sdb._redis).get(EVIDENCE_CARDS_KEY), 2.0, "evidence_cards_load")
        return json_loads(raw) if raw else None
    except asyncio.CancelledError:
        raise
    except Exception:
        return None


async def _run(parsed_cache, pairs, reference_time, sdb, tg, log, suppress, stats
               ) -> Dict[str, int]:
    if cfg.DRY_RUN_MODE:
        log.info("🌊 Elliott: DRY_RUN_MODE — scan skipped (nothing is claimed or recorded)")
        return stats
    if sdb.degraded:
        log.warning("🌊 Elliott: Redis unavailable — scan skipped (no way to deduplicate)")
        return stats

    params = params_from_cfg()

    # One pipeline for both tracker hashes, started BEFORE the CPU scan so its
    # ~250 ms round-trip is paid by the analysis, not added to it.
    prefetch_task = asyncio.create_task(tracker.prefetch(sdb))
    evidence_task = asyncio.create_task(_prefetch_evidence(sdb))
    try:
        sem = asyncio.Semaphore(max(1, int(cfg.EVAL_CONCURRENCY_LIMIT)))
        last_idx: Dict[str, int] = {}
        async def _one(pair: str):
            async with sem:
                pd_15m = (parsed_cache.get(pair) or (None, None, None))[0]
                try:
                    i, res = await asyncio.to_thread(scan_pair, pair, pd_15m,
                                                     reference_time, params)
                except Exception as exc:
                    log.warning(f"[{pair}] Elliott analysis failed: {exc!r}")
                    stats["errors"] += 1
                    return pair, None
                if i is not None:
                    last_idx[pair] = i
                return pair, res

        scanned = await asyncio.gather(*[_one(p) for p in pairs])
        track_pre, stats_pre = await prefetch_task
    except BaseException:
        prefetch_task.cancel()
        evidence_task.cancel()
        raise

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
        elif not deliver_enabled(s.family):
            why_not[s.event_id] = "delivery_off_family"
        elif not can_send:
            why_not[s.event_id] = "delivery_off"
        else:
            deliverable.append((pair, s))

    primary_family = str(cfg.ELLIOTT_WAVE_PRIMARY_ENTRY)
    deliverable.sort(key=lambda t: (-t[1].score,
                                    family_rank(t[1].family, primary_family), t[0]))

    cap = max(0, int(cfg.ELLIOTT_WAVE_MAX_ALERTS_PER_RUN))
    ttl = _event_ttl()
    selected: List[Tuple[str, WaveSetup]] = []
    slots: Dict[str, Tuple[str, int]] = {}      # event_id -> (pair, slot)
    duplicates: set = set()
    releases: List[str] = []                    # flushed in ONE pipeline, once

    # Claim in chunks of exactly the remaining need. A structure is claimed ONLY
    # if it can still be delivered, so an over-cap setup stays deliverable on a
    # later run (this is why we do NOT pre-claim a buffer).
    idx = 0
    while idx < len(deliverable) and len(selected) < cap:
        chunk = deliverable[idx: idx + max(1, cap - len(selected))]
        idx += len(chunk)
        names = [f"ewd:{s.structure_id}" for _, s in chunk]
        flags = await _claim_many(sdb, names, ttl)
        owned: List[Tuple[str, WaveSetup]] = []
        for pos, (pair, s) in enumerate(chunk):
            got = flags[pos] if flags is not None else None
            if got is None:
                got = await _claim_name(sdb, names[pos], ttl)
            if got is None:
                # Unknown claim state: never call this a duplicate, never deliver.
                log.warning(f"[{pair}] Elliott claim state unknown — setup skipped")
                stats["errors"] += 1
                why_not[s.event_id] = "claim_error"
                continue
            if not got:
                duplicates.add(s.event_id)
                stats["duplicates"] += 1
                continue
            owned.append((pair, s))
        if not owned:
            continue
        pair_slots = await _claim_pair_slots(sdb, [p for p, _ in owned])
        for pair, s in owned:
            slot = pair_slots.get(pair)
            if slot is None:
                # the structure stays claimed: it is not delivered later, only recorded
                why_not[s.event_id] = "pair_limit"
                stats["pair_limited"] += 1
                continue
            slots[s.event_id] = (pair, slot)
            selected.append((pair, s))
    for pair, s in deliverable[idx:]:
        why_not[s.event_id] = "over_cap"

    delivered_ids: set = set()
    if selected:
        verdicts = await tracker.verdicts_for(
            sdb, [s.family for _, s in selected], preloaded_stats=stats_pre)
        evidence_cards: Optional[Dict[str, Optional[Dict[str, Any]]]] = None
        if getattr(cfg, "ALERT_EVIDENCE_BLOCK", True):
            try:
                _blob = await evidence_task
                _max_age = float(getattr(cfg, "ALERT_EVIDENCE_MAX_AGE_HOURS", 36.0)) * 3600.0
                evidence_cards = {
                    alert_key_of(s): evidence_card_lookup(_blob, alert_key_of(s), s.direction, _max_age)
                    for _, s in selected
                }
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.debug(f"Elliott evidence cards unavailable: {exc!r}")
                evidence_cards = {}
        delivered_ids = await _deliver(selected, reference_time, sdb, tg, log,
                                       stats, why_not, verdicts, releases,
                                       evidence_cards=evidence_cards)
        for _, s in selected:
            if s.event_id not in delivered_ids and s.event_id in slots:
                p, k = slots[s.event_id]
                releases.append(f"ewp:{p}:{k}")

    # ── shadow recording of everything not delivered (once per structure) ──
    # Track what we own this run (delivered setups + won ews: claims), plus any
    # claim-loser that has NO open track record — see the elif below.
    recorded_for_tracking: List[Tuple[str, WaveSetup, str]] = []
    shadow_candidates: List[Tuple[str, WaveSetup, str]] = []
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

    if shadow_candidates:
        # A structure with an open track record already holds its shadow claim (the
        # record is only ever created by the run that won it), so it needs no claim
        # command at all. This removes the one SET NX per persisting structure per
        # run, which is the whole steady-state cost of this block.
        known = track_pre if track_pre is not None else {}
        fresh = [t for t in shadow_candidates if t[1].structure_id not in known]
        snames = [f"ews:{s.structure_id}" for _, s, _ in fresh]
        sflags = await _claim_many(sdb, snames, ttl) if fresh else []
        owned_shadows: List[Tuple[str, WaveSetup, str]] = []
        for pos, (pair, s, disp) in enumerate(fresh):
            got = sflags[pos] if sflags is not None else None
            if got is None:
                got = await _claim_name(sdb, snames[pos], ttl)
            if got is None:
                stats["errors"] += 1
                got = False
            if got:
                owned_shadows.append((pair, s, disp))
                recorded_for_tracking.append((pair, s, disp))
            # A claim loser is NEVER re-registered here: if its record was resolved
            # and removed, re-adding it would count the same structure twice. The
            # registration guard in tracker.register_many enforces this as well.
        releases.extend(await _record_shadow_batch(sdb, owned_shadows, log, stats))

    if releases:
        await _release_many(sdb, releases)

    # ── follow-up: register new structures / events, then advance all of them ──
    reg_items: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for pair, s, disp in recorded_for_tracking:
        try:
            rec = _track_record(pair, s, disp, _vol_ref(parsed_cache, pair, s.signal_idx))
            reg_items.append((rec, _snapshot(s, disp)))
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
                              "elliott_signs": ev.signs,
                              "elliott_labels": {k: list(v) for k, v in ev.labels.items()}})
            reg_items.append((rec, snap))
        except Exception as exc:
            log.warning(f"[{pair}] Elliott event register failed: {exc!r}")

    added: Dict[str, Dict[str, Any]] = {}
    try:
        added = await tracker.register_many(sdb, reg_items, log)
    except Exception as exc:
        log.warning(f"Elliott tracker register batch failed: {exc!r}")

    # prefetched map + this run's additions == a post-register HGETALL, minus 1 RTT
    merged = None if track_pre is None else {**track_pre, **added}
    prog = await tracker.update_all(sdb, parsed_cache, last_idx, log, preloaded=merged)
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
                   verdicts: Optional[Dict[str, str]] = None,
                   releases: Optional[List[str]] = None,
                   evidence_cards: Optional[Dict[str, Optional[Dict[str, Any]]]] = None) -> set:
    """Send ONE Telegram message with the selected setups, then record them as
    Recorded outcomes. A failed send is parked in the dead-letter queue together
    with the outcome rows it owes; if even that fails the claims are released so
    the next run can try again."""
    releases = releases if releases is not None else []
    message = format_elliott_message([s for _, s in selected], verdicts, evidence_cards)
    outcomes = [_outcome_kwargs(p, s, "delivered") for p, s in selected]
    sent = False
    try:
        sent = bool(await tg.send(message))
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        log.error(f"Elliott Telegram send raised: {exc!r}")

    if sent:
        for pair in await _record_delivered_batch(sdb, selected, outcomes, log):
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
        releases.append(f"ewd:{s.structure_id}")
        why_not[s.event_id] = "send_failed"
    stats["errors"] += 1
    return set()