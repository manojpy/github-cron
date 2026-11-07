#!/usr/bin/env python3
"""learner.py — the SLOW half of the advisor (runs every few hours, not every 15 min).

Reads the outcome archives (stored candle-by-candle paths), builds a no-alert
control from raw 15m candles, runs the champion/challenger playbook logic from
playbook.py and writes the result to Redis metadata for the fast cron to READ.

It never places orders, never edits config and never changes alert rules.
The fast bot only uses the playbook in restrict-only mode (PLAYBOOK_MODE).

Usage (inside the bot image, OUTCOME_DATA_DIR = clone of the outcome-data repo):
    python learner.py                    # full cycle: learn, write playbook, notify
    python learner.py --dry-run          # compute + print, write nothing, send nothing
    python learner.py --no-recon         # do not rebuild missing candle paths of old alerts
    python learner.py --no-control       # skip candles (nothing can pass beyond PROMISING
                                         #   while PLAYBOOK_REQUIRE_CONTROL is true)
    python learner.py --rollback         # restore the previous playbook version
    python learner.py --digest           # print the research ledger tail
    python learner.py --scoreboard       # print the last stored scoreboard
    python learner.py --intel            # print the last stored intelligence report
    python learner.py --data-dir PATH    # override OUTCOME_DATA_DIR
Exit codes: 0 ok, 1 learner error, 2 cannot reach Redis / no data dir.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Awaitable, cast
from bot_config import cfg, json_dumps
from alert_registry import alert_family_of, pretty_alert
from alerts import TelegramQueue, escape_markdown_v2
from archive_reader import load_archived_outcomes
from autopsy import build_intelligence, format_intelligence
from bot_config import _get_session_from_ts
from brain_helpers import HYPOTHESIS_POOL_KEY
from autotune import FIELD as AT_FIELD, KEY_AUTOTUNE
from autotune import committed as at_committed, evaluate_pending as at_evaluate, observe as at_observe, propose as at_propose
from hypothesis_gen import compact_rules, update_hypothesis_pool
from intelligence import (
    KEY_FIREWALL, KEY_INTEL, KEY_ISSUES, KEY_PROMO_LEDGER, KEY_VETO_MON, KEY_WIN_POOL,
    build_report, conviction_anatomy, delivered_trend, firewall_contract, format_report, ledger_append, ledger_event,
    rule_lift, update_issue_state, veto_monitor, verdict_items, win_candidates,
)
from feedback import (
    KEY_FEEDBACK, behaviour_report, format_behaviour, format_heat_gate, format_margin,
    heat_gate_report, margin_report,
)
from pathrecon import (
    KEY_RECON, apply_recon, cache_key, cache_pack, cache_unpack, needs_recon, recon_path, windows_for,
)
from playbook import (
    ControlModel, KEY_CURRENT, KEY_LEDGER, KEY_PREV, KEY_SCORE, KEY_STATE, KEY_STATE_PREV,
    build_playbook, default_params, format_summary,
)
from scoreboard import build_scoreboard, format_scoreboard
from state import RedisStateStore
_TTL = 90 * 86400
KEY_AUTOPSY = "brain_autopsy"
_RECON_CAP = 8000   # was 5000: completed early-closed paths now share this cache (~360 B/entry, ~3 MB at the cap)
_LEDGER_CAP = 600
_CANDLE_SEC = 900

def _say(msg: str) -> None:
    print(msg, flush=True)

def params_from_cfg() -> Dict[str, Any]:
    p = default_params()
    p.update({
        "min_n": int(cfg.PLAYBOOK_MIN_N),
        "min_holdout": int(cfg.PLAYBOOK_MIN_HOLDOUT),
        "promote_consecutive": int(cfg.PLAYBOOK_PROMOTE_CONSECUTIVE),
        "min_forward": int(cfg.PLAYBOOK_MIN_FORWARD),
        "require_control": bool(cfg.PLAYBOOK_REQUIRE_CONTROL),
        "family_alpha": float(cfg.PLAYBOOK_FAMILY_ALPHA),
        "reconfirm_days": float(cfg.PLAYBOOK_RECONFIRM_DAYS),
        "expire_days": float(cfg.PLAYBOOK_EXPIRE_DAYS),
        "max_dd_r": float(cfg.PLAYBOOK_MAX_DD_R),
        "stab_floor_r": float(cfg.PLAYBOOK_STAB_FLOOR_R),
        "avoid_consecutive": int(cfg.PLAYBOOK_AVOID_CONSECUTIVE),
        "max_rules": int(cfg.PLAYBOOK_MAX_RULES),
        "size_full_lcb": float(cfg.PLAYBOOK_SIZE_FULL_LCB),
    })
    return p


def criteria_from_cfg() -> Dict[str, Any]:
    return {
        "take_min_blocks": int(cfg.PLAYBOOK_SB_TAKE_MIN_BLOCKS),
        "avoid_min_blocks": int(cfg.PLAYBOOK_SB_AVOID_MIN_BLOCKS),
        "calib_tol": float(cfg.PLAYBOOK_SB_CALIB_TOL),
        "max_dd_r": float(cfg.PLAYBOOK_MAX_DD_R),
    }


def typical_cost_by_pair(rows: List[Dict[str, Any]], min_rows: int = 5) -> Dict[str, float]:
    """Median realized round-trip cost per pair (spreads / slippage differ by pair)."""
    import statistics
    by: Dict[str, List[float]] = {}
    for r in rows:
        c = r.get("realized_cost_pct")
        if c is not None and r.get("pair"):
            by.setdefault(str(r["pair"]), []).append(float(c))
    return {p: statistics.median(v) for p, v in by.items() if len(v) >= min_rows}


def load_rows(data_dir: str, window_days: int) -> List[Dict[str, Any]]:
    """Real alerts are tagged LIVE and shadow (would-have-fired) rows SHADOW."""
    rows: List[Dict[str, Any]] = []
    for shadow in (False, True):
        try:
            got = load_archived_outcomes(data_dir, window_days=window_days, shadow=shadow)
        except Exception as e:  # a missing shadow/ folder must not kill the run
            _say(f"[learner] could not read {'shadow' if shadow else 'real'} archive: {e}")
            continue
        for r in got:
            r = dict(r)
            r["source"] = "SHADOW" if shadow else "LIVE"
            rows.append(r)
    return rows


async def fetch_control_candles(days: int) -> Tuple[Dict[str, Tuple[Any, Any, Any, Any, Any]], str]:
    """Closed 15m candles per pair for the no-alert control.  Needs >=50% of pairs."""
    from fetcher import DataFetcher, parse_candles_to_numpy
    fetcher = DataFetcher(cfg.DELTA_API_BASE)
    now = int(time.time())
    last_closed_open = (now // _CANDLE_SEC) * _CANDLE_SEC - _CANDLE_SEC
    limit = min(1900, days * 96 + 40)
    out: Dict[str, Tuple[Any, Any, Any, Any, Any]] = {}
    sem = asyncio.Semaphore(4)

    async def one(pair: str) -> None:
        async with sem:
            try:
                res = await fetcher.fetch_candles(pair, "15", limit, now, expected_open_15=last_closed_open + _CANDLE_SEC)
                pd = parse_candles_to_numpy(res) if res else None
                if pd is None:
                    return
                d = pd.as_dict()
                ts = d["timestamp"] if "timestamp" in d else d["ts"]
                keep = ts <= last_closed_open          # drop the still-forming candle
                if int(keep.sum()) < 200:
                    return
                out[pair] = (ts[keep], d["open"][keep], d["high"][keep], d["low"][keep], d["close"][keep])
            except Exception as e:
                _say(f"[learner] candles failed for {pair}: {e}")

    await asyncio.gather(*(one(p) for p in cfg.PAIRS))
    try:
        from fetcher import SessionManager
        await SessionManager.close_session()
    except Exception:
        pass
    need = max(1, len(cfg.PAIRS) // 2)
    if len(out) < need:
        return {}, f"only {len(out)}/{len(cfg.PAIRS)} pairs returned candles"
    return out, f"{len(out)}/{len(cfg.PAIRS)} pairs, {days}d"

async def recover_paths(rows: List[Dict[str, Any]], sdb: Any, max_calls: int, dry_run: bool) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Give REAL archived alerts that have no stored candle path one, rebuilt from the
    exchange's candle history (see pathrecon.py).  Results are cached in Redis so each
    row is fetched once.  Never raises: on any problem the rows are returned as they were."""
    stats = {"missing": 0, "from_cache": 0, "rebuilt": 0, "rejected": 0, "calls": 0,
             "short": 0, "extended": 0}
    try:
        full_len = int(cfg.OUTCOME_LOOKAHEAD_CANDLES) + 1
        need = [r for r in rows if needs_recon(r, full_len)]
        stats["missing"] = sum(1 for r in need if not r.get("path_fav"))
        stats["short"] = len(need) - stats["missing"]      # early-closed: path ends at the exit candle
        if not need:
            return rows, stats
        cache: Dict[str, Any] = _jl(await sdb.get_metadata(KEY_RECON, timeout=5.0), {})
        found: Dict[str, Dict[str, Any]] = {}
        todo: Dict[str, List[Dict[str, Any]]] = {}
        for r in need:
            k = cache_key(r["pair"], r["direction"], r["entry_ts"])
            p = cache.get(k)
            if p and p.get("f"):
                found[k] = cache_unpack(p)
                stats["from_cache"] += 1
                if r.get("path_fav"):
                    stats["extended"] += 1
            elif p and p.get("x"):
                stats["rejected"] += 1
            else:
                todo.setdefault(str(r["pair"]), []).append(r)
        dirty = False
        if todo and max_calls > 0:
            from fetcher import DataFetcher, parse_candles_to_numpy
            fetcher = DataFetcher(cfg.DELTA_API_BASE)
            now = int(time.time())
            last_open = (now // _CANDLE_SEC) * _CANDLE_SEC - _CANDLE_SEC
            fd, la = int(cfg.OUTCOME_FILL_DELAY_CANDLES), int(cfg.OUTCOME_LOOKAHEAD_CANDLES)
            for pair, prs in sorted(todo.items(), key=lambda kv: -len(kv[1])):
                for start, end in windows_for([int(r["entry_ts"]) for r in prs]):
                    if stats["calls"] >= max_calls:
                        break
                    end = min(end, last_open)
                    stats["calls"] += 1
                    try:
                        res = await fetcher.fetch_candles(pair, "15", 1900, end, expected_open_15=end)
                        pdata = parse_candles_to_numpy(res) if res else None
                    except Exception as e:
                        # One failed window must not throw away the paths already rebuilt.
                        _say(f"[learner] path recovery fetch failed for {pair}: {e!r}")
                        continue
                    if pdata is None:
                        continue
                    d = pdata.as_dict()
                    ts, o, h, lo, c = d["timestamp"], d["open"], d["high"], d["low"], d["close"]
                    keep = ts <= last_open
                    ts, o, h, lo, c = ts[keep], o[keep], h[keep], lo[keep], c[keep]
                    for r in prs:
                        et = int(r["entry_ts"])
                        if not (start <= et <= end):
                            continue
                        k = cache_key(r["pair"], r["direction"], et)
                        if k in found or k in cache:
                            continue
                        rec = recon_path(ts, o, h, lo, c, entry_ts=et, direction=str(r["direction"]),
                                         fill_price=r.get("fill_price"), fill_delay=fd, lookahead=la,
                                         expect_prefix=r if r.get("path_fav") else None)
                        if rec is None:
                            covered = len(ts) > 0 and et + (fd + la + 2) * _CANDLE_SEC <= int(ts[-1])
                            if covered:
                                cache[k] = {"x": 1}          # data was there but did not match: do not retry
                                stats["rejected"] += 1
                                dirty = True
                            continue
                        found[k] = rec
                        cache[k] = cache_pack(rec)
                        stats["rebuilt"] += 1
                        if r.get("path_fav"):
                            stats["extended"] += 1
                        dirty = True
            try:
                from fetcher import SessionManager
                await SessionManager.close_session()
            except Exception:
                pass
        if dirty and not dry_run:
            if len(cache) > _RECON_CAP:
                keep_keys = sorted(cache, key=lambda x: int(x.rsplit("|", 1)[-1]))[-_RECON_CAP:]
                _say(f"[learner] path cache over its cap: dropping the {len(cache) - _RECON_CAP} oldest entries "
                     f"(rows older than that are re-fetched when still inside the window)")
                cache = {k: cache[k] for k in keep_keys}
            await sdb.set_metadata(KEY_RECON, json_dumps(cache), ttl=_TTL, timeout=10.0)
        out = []
        for r in rows:
            if needs_recon(r, full_len):
                k = cache_key(r["pair"], r["direction"], r["entry_ts"])
                if k in found:
                    r = apply_recon(r, found[k])
            out.append(r)
        return out, stats
    except Exception as e:
        _say(f"[learner] path recovery skipped: {e!r}")
        return rows, stats

def _jl(raw: Optional[str], default: Any) -> Any:
    try:
        return json.loads(raw) if raw else default
    except Exception:
        return default

async def _strict_get(sdb: RedisStateStore, key: str) -> Optional[str]:
    """None only when the key is absent; any Redis failure raises (learner exits, nothing overwritten)."""
    rc = sdb._redis
    if rc is None:
        raise RuntimeError("learner: Redis client not connected")
    return await asyncio.wait_for(
        cast(Awaitable[Optional[str]], rc.get(f"{sdb.meta_prefix}{key}")),
        timeout=5.0,
    )

async def _strict_set_many(sdb: RedisStateStore, items: List[Tuple[str, str]], ttl: int) -> None:
    """All keys written in one MULTI/EXEC; any failure raises, so the learner never half-writes."""
    rc = sdb._redis
    if rc is None:
        raise RuntimeError("learner: Redis client not connected")
    async with rc.pipeline(transaction=True) as pipe:
        for key, value in items:
            pipe.set(f"{sdb.meta_prefix}{key}", value, ex=ttl)
        await asyncio.wait_for(cast(Awaitable[Any], pipe.execute()), timeout=10.0)

def _tg_chunks(text: str, limit: int = 3800) -> List[str]:
    """MarkdownV2-escaped chunks of at most `limit` chars (Telegram rejects > 4096).
    Splits on blank lines, then on single lines for an oversized paragraph."""
    chunks: List[str] = []
    cur = ""
    for para in text.split("\n\n"):
        pieces = [escape_markdown_v2(para)]
        if len(pieces[0]) > limit:
            pieces = [escape_markdown_v2(ln) for ln in para.split("\n")]
        for i, piece in enumerate(pieces):
            piece = piece[:limit]
            sep = "\n\n" if i == 0 else "\n"
            if cur and len(cur) + len(sep) + len(piece) > limit:
                chunks.append(cur)
                cur = piece
            else:
                cur = cur + sep + piece if cur else piece
    if cur:
        chunks.append(cur)
    return chunks

async def run(args: argparse.Namespace) -> int:
    data_dir = args.data_dir or os.environ.get("OUTCOME_DATA_DIR") or ""
    sdb = RedisStateStore(cfg.REDIS_URL)
    await sdb.connect()
    if sdb.degraded or not sdb._redis:
        _say("[learner] Redis unreachable — nothing done")
        return 2
    tq = None if (args.dry_run or cfg.PLAYBOOK_NOTIFY == "never") else TelegramQueue(cfg.TELEGRAM_BOT_TOKEN, cfg.TELEGRAM_CHAT_ID)

    async def notify(text: str) -> None:
        if tq is not None:
            for chunk in _tg_chunks(text):
                if not await tq.send(chunk):
                    _say("[learner] Telegram message was not delivered")
    try:
        if args.digest:
            led = _jl(await sdb.get_metadata(KEY_LEDGER, timeout=5.0), [])
            for e in led[-40:]:
                _say(json.dumps(e, ensure_ascii=False))
            return 0

        if args.scoreboard:
            sbj = _jl(await sdb.get_metadata(KEY_SCORE, timeout=5.0), {})
            _say(format_scoreboard(sbj) if sbj else "[learner] no scoreboard stored yet")
            return 0

        if args.autopsy:
            apj = _jl(await sdb.get_metadata(KEY_AUTOPSY, timeout=5.0), {})
            _say(format_intelligence(apj, name_fn=pretty_alert) if apj else "[learner] no autopsy stored yet")
            return 0

        if args.intel:
            ij = _jl(await sdb.get_metadata(KEY_INTEL, timeout=5.0), {})
            _say(format_report(ij, veto_mode=str(cfg.VETO_MONITOR_MODE), firewall_mode=str(cfg.ALERT_FIREWALL_MODE)) if ij else "[learner] no intelligence report stored yet")
            return 0

        if args.rollback:
            prev = await _strict_get(sdb, KEY_PREV)
            prev_state_raw = await _strict_get(sdb, KEY_STATE_PREV)
            if not prev:
                _say("[learner] no previous playbook stored — nothing to roll back to")
                return 1
            rb_items = [(KEY_CURRENT, prev)]
            if prev_state_raw:
                rb_items.append((KEY_STATE, prev_state_raw))
            await _strict_set_many(sdb, rb_items, _TTL)
            label = _jl(prev, {}).get("label")
            _say(f"[learner] rolled back to playbook {label}")
            await notify(f"🧠 PLAYBOOK rolled back to {label}")
            return 0

        if not data_dir or not os.path.isdir(data_dir):
            _say(f"[learner] outcome data dir not found: {data_dir!r}")
            return 2

        now_ts = time.time()
        rows = load_rows(data_dir, int(cfg.PLAYBOOK_WINDOW_DAYS))
        rstats: Dict[str, int] = {}
        if not args.no_recon and int(cfg.PLAYBOOK_RECON_MAX_CALLS) > 0:
            rows, rstats = await recover_paths(rows, sdb, int(cfg.PLAYBOOK_RECON_MAX_CALLS), bool(args.dry_run))
            _say(f"[learner] path recovery: {rstats}")
        n_path = sum(1 for r in rows if r.get("path_fav"))
        _say(f"[learner] rows={len(rows)} with_path={n_path}")

        control = None
        control_note = "skipped (--no-control)"
        if not args.no_control:
            candles, control_note = await fetch_control_candles(int(cfg.PLAYBOOK_CONTROL_DAYS))
            if candles:
                control = ControlModel(candles, max_h=12, session_fn=_get_session_from_ts)
                if control.n_pairs == 0:
                    control, control_note = None, "candle history too short"
        _say(f"[learner] control: {'on' if control else 'OFF'} ({control_note})")

        prev_state = _jl(await _strict_get(sdb, KEY_STATE), {})

        fixed = {
            "sl": float(cfg.OUTCOME_MAE_LOSS_PCT),
            "tp": float(cfg.OUTCOME_MAE_LOSS_PCT) * float(cfg.OUTCOME_RR_TARGET),
            "h": int(cfg.OUTCOME_LOOKAHEAD_CANDLES),
        }
        cost_pct = (float(cfg.BRAIN_FEE_PCT) * 2 + float(cfg.BRAIN_SLIPPAGE_PCT) * 2) * 100.0
        blob, new_state, changes, ledger = build_playbook(
            rows, now_ts=now_ts, prev_state=prev_state, cost_pct=cost_pct, fixed=fixed,
            params=params_from_cfg(), control=control, control_note=control_note,
            family_fn=alert_family_of, cost_by_pair=typical_cost_by_pair(rows),
        )
        prev_sb = _jl(await _strict_get(sdb, KEY_SCORE), {})
        sb = build_scoreboard(blob, rows, cost_pct=cost_pct, fixed=fixed, criteria=criteria_from_cfg(),
                              family_fn=alert_family_of)
        if prev_sb.get("criteria_hash") and prev_sb["criteria_hash"] != sb["criteria_hash"]:
            ledger.append({"ts": int(now_ts), "key": "*", "decision": "CRITERIA_CHANGED",
                           "reason": f"scoreboard criteria {prev_sb['criteria_hash']} -> {sb['criteria_hash']}"})
            changes.append({"key": "*", "event": "CRITERIA_CHANGED", "from": prev_sb["criteria_hash"],
                            "to": sb["criteria_hash"], "detail": "pass criteria were edited", "plan": None})
        text = format_summary(blob, changes, name_fn=pretty_alert) + "\n\n" + format_scoreboard(sb)
        fb_log = _jl(await sdb.get_metadata(KEY_FEEDBACK, timeout=5.0), {})

        if fb_log:
            text += "\n\n" + format_behaviour(behaviour_report(fb_log, rows, fixed=fixed, cost_pct=cost_pct))
        if cfg.ENABLE_PORTFOLIO_HEAT_GATE:
            text += "\n\n" + format_heat_gate(heat_gate_report(rows, fixed=fixed, cost_pct=cost_pct))
        if cfg.ENABLE_ALERT_WHY_SURVIVED:
            text += "\n\n" + format_margin(margin_report(rows, fixed=fixed, cost_pct=cost_pct))
        _rec_all = rstats.get("rebuilt", 0) + rstats.get("from_cache", 0)
        _rec_ext = rstats.get("extended", 0)
        if _rec_all - _rec_ext > 0:
            text += (f"\n\n🧩 Recovered candle paths for {_rec_all - _rec_ext} "
                     f"alert(s) archived without one (real and shadowed; early-closed trades and "
                     f"older rows) from exchange history (tagged RECON).")
        if _rec_ext > 0 or rstats.get("short"):
            text += (f"\n\n🧩 Completed the forward path of {_rec_ext} early-closed trade(s) "
                     f"(tagged RECON_EXT); {max(0, rstats.get('short', 0) - _rec_ext)} still short "
                     f"(horizon not closed yet, API budget, or history did not match).")

        _say(text)

        ap_blob: Optional[Dict[str, Any]] = None
        ap_text = ""
        ap_due = False
        if cfg.ENABLE_TRADE_AUTOPSY:
            try:
                ap_blob = build_intelligence(
                    rows, cost_pct=cost_pct, sl_pct=fixed["sl"], tp_pct=fixed["tp"],
                    group_fn=alert_family_of,
                    params={
                        "min_n": cfg.AUTOPSY_MIN_N, "edge_min_n": cfg.AUTOPSY_EDGE_MIN_N,
                        "reject_min_n": cfg.AUTOPSY_REJECT_MIN_N, "shrink_k": cfg.AUTOPSY_SHRINK_K,
                        "target": cfg.TRUST_TARGET_PRECISION, "min_take_n": cfg.TRUST_MIN_TAKE_N,
                        "windows": cfg.TRUST_WINDOWS,
                        "exh_early_bars": cfg.EXHAUSTION_EARLY_STOP_BARS,
                        "exh_min_cats": cfg.EXHAUSTION_MIN_CATEGORIES,
                        "exh_veto_score": cfg.EXHAUSTION_VETO_SCORE,
                    },
                )
                ap_blob["ts"] = int(now_ts)
                ap_text = format_intelligence(ap_blob, name_fn=pretty_alert)
                prev_ap = _jl(await sdb.get_metadata(KEY_AUTOPSY, timeout=5.0), {})
                last_sent = float(prev_ap.get("sent_ts") or 0)
                hours = float(cfg.AUTOPSY_NOTIFY_HOURS)
                ap_due = hours > 0 and (now_ts - last_sent) >= hours * 3600
                ap_blob["sent_ts"] = int(now_ts) if ap_due else int(last_sent)
                _say(ap_text)
            except Exception as e:
                _say(f"[learner] autopsy skipped: {e!r}")
                ap_blob = None

        # ── Coordinator ──
        # Writes no trading rule. It (1) judges every AVOID veto separately and can suspend one that is proven
        # harmful, (2) decides which advisor-verdict groups the delivery firewall may drop, (3) publishes the
        # forward-confirmed skip rules, (4) proposes / judges ONE bounded CONFLUENCE_MIN_PCT tightening, (5) runs the
        # weekly take-side search (report only) and (6) ranks what needs attention. Every action is restrict-only,
        # has its own mode switch and is written to the promotion ledger. Nothing is written under --dry-run.
        intel_blob: Optional[Dict[str, Any]] = None
        intel_text = ""
        intel_due = False
        intel_failed = False
        pub: Dict[str, Any] = {}                       # fields added to the playbook blob, applied only if everything succeeded
        veto_res: Optional[Dict[str, Any]] = None
        veto_state_new: Optional[Dict[str, Any]] = None
        fw_res: Optional[Dict[str, Any]] = None
        fw_state_new: Optional[Dict[str, Any]] = None
        at_state_new: Optional[Dict[str, Any]] = None
        win_pool_new: Optional[Dict[str, Any]] = None
        issues_new: Optional[Dict[str, Any]] = None
        fresh_ids: List[str] = []
        events: List[Dict[str, Any]] = []
        veto_mode = str(cfg.VETO_MONITOR_MODE)
        fw_mode = str(cfg.ALERT_FIREWALL_MODE)
        at_mode = str(cfg.AUTOTUNE_MODE)
        live_ts = [float(r["entry_ts"]) for r in rows if r.get("source") == "LIVE" and r.get("entry_ts") is not None]
        newest_ts = max(live_ts) if live_ts else None
        data_ok = bool(rows) and n_path / len(rows) >= 0.6
        if cfg.ENABLE_INTELLIGENCE:
            try:
                prev_intel = _jl(await _strict_get(sdb, KEY_INTEL), {})
                prev_pb = _jl(await _strict_get(sdb, KEY_CURRENT), {})
                hyp_pool = _jl(await _strict_get(sdb, HYPOTHESIS_POOL_KEY), {})

                # 1. veto monitor (per rule)
                if veto_mode != "off":
                    veto_res, veto_state_new = veto_monitor(
                        blob, rows, fixed=fixed, cost_pct=cost_pct,
                        prev_state=_jl(await _strict_get(sdb, KEY_VETO_MON), {}), now_ts=now_ts,
                        params={"min_blocks": int(cfg.VETO_MONITOR_MIN_BLOCKS),
                                "harm_cycles": int(cfg.VETO_MONITOR_HARM_CYCLES),
                                "resume_cycles": int(cfg.VETO_MONITOR_RESUME_CYCLES)},
                        family_fn=alert_family_of, data_ok=data_ok,
                    )
                    if veto_mode == "live" and veto_res["suspended"]:
                        pub["veto_suspended"] = veto_res["suspended"]
                    for ch in veto_res["changed"]:
                        events.append(ledger_event("VETO", ch["key"], f"{ch['event']} veto {ch['key']}: {ch['reason'] or 'proven useful again'}",
                                                   now_ts, event=ch["event"], mode=veto_mode))

                # 2. delivery firewall contract
                if fw_mode != "off":
                    fw_res, fw_state_new = firewall_contract(
                        verdict_items(rows, fixed=fixed, cost_pct=cost_pct), now_ts=now_ts, newest_ts=newest_ts,
                        n_rows=len(rows), n_path=n_path, prev_state=_jl(await _strict_get(sdb, KEY_FIREWALL), {}),
                        candidates=tuple(str(cfg.ALERT_FIREWALL_DROP).split(",")),
                    )
                    pub["firewall"] = {"armed": fw_res["armed"], "ts": int(now_ts)}
                    for ch in fw_res["changed"]:
                        events.append(ledger_event("FIREWALL", ch["verdict"], f"{ch['event']} firewall for {ch['verdict']}", now_ts,
                                                   event=ch["event"], mode=fw_mode, evidence=fw_res["per_verdict"][ch["verdict"]]["group"]))

                # 3. publish only the confirmed skip rules that beat the REST of the delivered book
                hyp_lift: Optional[Dict[str, Any]] = None
                if str(cfg.ALERT_HYPOTHESIS_VETO_MODE) != "off" and data_ok:
                    cand_rules = compact_rules(hyp_pool, 50)
                    hyp_lift = rule_lift(cand_rules, rows, fixed=fixed, cost_pct=cost_pct)
                    by_id = {str(x["id"]): x for x in cand_rules}
                    pub["hyp_rules"] = [by_id[i] for i in hyp_lift["keep"]][:int(cfg.ALERT_HYPOTHESIS_MAX_RULES)]
                    was_ids = {str(x.get("id")) for x in (prev_pb.get("hyp_rules") or [])}
                    now_ids = {str(x["id"]) for x in pub["hyp_rules"]}
                    for rid in sorted(now_ids - was_ids):
                        rtxt = next(x["text"] for x in pub["hyp_rules"] if str(x["id"]) == rid)
                        events.append(ledger_event("HYPOTHESIS_RULE", rid, f"PUBLISHED skip rule ({cfg.ALERT_HYPOTHESIS_VETO_MODE}): {rtxt}", now_ts,
                                                   evidence=hyp_lift["per_rule"].get(rid)))
                    for rid in sorted(was_ids - now_ids):
                        why = (hyp_lift["per_rule"].get(rid) or {}).get("verdict", "no longer confirmed")
                        events.append(ledger_event("HYPOTHESIS_RULE", rid, f"WITHDRAWN skip rule {rid}: {why}", now_ts))

                # 4. bounded autotune (CONFLUENCE_MIN_PCT, tighten only)
                at_report: Optional[Dict[str, Any]] = None
                if at_mode != "off":
                    at_override = await sdb.get_config_override()
                    cur_pct = float(at_override.get(AT_FIELD, cfg.CONFLUENCE_MIN_PCT))
                    at_params = {"step": float(cfg.AUTOTUNE_STEP_PCT), "ceiling": float(cfg.AUTOTUNE_CEILING_PCT),
                                 "cooldown_h": float(cfg.AUTOTUNE_COOLDOWN_HOURS), "eval_h": float(cfg.AUTOTUNE_EVAL_HOURS),
                                 "max_age_h": float(cfg.PLAYBOOK_MAX_AGE_HOURS)}
                    st_obs, ext = at_observe(_jl(await _strict_get(sdb, KEY_AUTOTUNE), {}), cur_pct, now_ts)
                    for x in ext:
                        events.append(ledger_event("AUTOTUNE", AT_FIELD, x["text"], now_ts, event=x["event"], old=x["old"], new=x["new"]))
                        at_report = {"status": "EXTERNAL_CHANGE", "text": x["text"], "evidence": {}}
                    st_fin: Dict[str, Any] = st_obs
                    done_this_cycle = False
                    if at_mode == "live" and st_obs.get("pending"):
                        ev_res, st_ev = at_evaluate(rows, st_obs, now_ts=now_ts, fixed=fixed, cost_pct=cost_pct, params=at_params)
                        if ev_res["status"] == "ROLLBACK" and not args.dry_run:
                            rs = ev_res["restore"]
                            ok = (await sdb.remove_config_override_field(AT_FIELD)) if rs["remove"] else (await sdb.write_config_override(AT_FIELD, rs["value"]))
                            if ok:
                                st_fin, done_this_cycle = st_ev, True
                                at_report = {"status": "ROLLED_BACK", "text": ev_res["text"], "evidence": ev_res["evidence"]}
                                events.append(ledger_event("AUTOTUNE", AT_FIELD, ev_res["text"], now_ts, event="ROLLED_BACK", old=ev_res["old"], new=ev_res["new"], evidence=ev_res["evidence"]))
                            else:
                                at_report = {"status": "WRITE_FAILED", "text": f"rollback of {AT_FIELD} could not be written", "evidence": ev_res["evidence"]}
                        elif ev_res["status"] == "ROLLBACK":
                            at_report = {"status": "WOULD_APPLY", "text": "--dry-run: " + ev_res["text"], "evidence": ev_res["evidence"]}
                        elif ev_res["status"] in ("KEPT", "INSUFFICIENT"):
                            st_fin, done_this_cycle = st_ev, True
                            at_report = {"status": "KEPT", "text": ev_res["text"], "evidence": ev_res.get("evidence") or {}}
                            events.append(ledger_event("AUTOTUNE", AT_FIELD, ev_res["text"], now_ts, event="KEPT", evidence=ev_res.get("evidence") or {}))
                    if not st_fin.get("pending") and not done_this_cycle and at_report is None:
                        dec, st_fin = at_propose(
                            rows, cur_pct=cur_pct, macro_live=bool(cfg.MACRO_CONTEXT_LIVE), state=st_fin, now_ts=now_ts, fixed=fixed, cost_pct=cost_pct,
                            newest_ts=newest_ts, n_rows=len(rows), n_path=n_path, data_ok=data_ok, params=at_params,
                        )
                        if dec["action"] == "APPLY":
                            if at_mode == "live" and not args.dry_run:
                                if await sdb.write_config_override(AT_FIELD, dec["new"]):
                                    st_fin = at_committed(st_fin, dec, now_ts, had_override=AT_FIELD in at_override, prev_override=at_override.get(AT_FIELD))
                                    at_report = {"status": "APPLIED", "text": dec["text"], "evidence": dec["evidence"]}
                                    events.append(ledger_event("AUTOTUNE", AT_FIELD, "APPLIED " + dec["text"], now_ts, event="APPLIED", old=dec["old"], new=dec["new"], evidence=dec["evidence"]))
                                else:
                                    at_report = {"status": "WRITE_FAILED", "text": f"{AT_FIELD} {dec['old']:g} -> {dec['new']:g} could not be written", "evidence": dec["evidence"]}
                            else:
                                at_report = {"status": "WOULD_APPLY", "text": dec["text"], "evidence": dec["evidence"]}
                    at_state_new = st_fin

                # 5. weekly take-side search (report only)
                win: List[Dict[str, Any]] = []
                win_days = int(cfg.INTEL_WIN_SEARCH_DAYS)
                if win_days > 0:
                    wp_prev = _jl(await _strict_get(sdb, KEY_WIN_POOL), {})
                    due_win = (now_ts - float(wp_prev.get("last_run_ts") or 0)) >= win_days * 86400
                    wp, wsum = update_hypothesis_pool(
                        wp_prev, [r for r in rows if r.get("source") == "LIVE"], int(now_ts), cost_pct=cost_pct,
                        budget=int(cfg.INTEL_WIN_BUDGET) if due_win else 0, shadow_max=int(cfg.HYPOTHESIS_SHADOW_MAX),
                        forward_min_n=int(cfg.HYPOTHESIS_FORWARD_MIN_N), expire_days=int(cfg.HYPOTHESIS_EXPIRE_DAYS), direction="win",
                    )
                    last_run = int(now_ts) if (due_win and str((wsum or {}).get("status")) == "ran") else int(wp_prev.get("last_run_ts") or 0)
                    wp["last_run_ts"] = last_run
                    win_pool_new = wp
                    win = win_candidates(wp, rows, cost_pct)

                # 6. one ranked, fingerprinted issue list
                intel_blob = build_report(
                    now_ts=now_ts, blob=blob, sb=sb, autopsy=ap_blob, hyp_pool=hyp_pool, veto=veto_res,
                    trend=delivered_trend(rows, fixed=fixed, cost_pct=cost_pct, now_ts=now_ts),
                    n_rows=len(rows), n_path=n_path, prev_ts=prev_pb.get("generated_ts"),
                    max_age_h=float(cfg.PLAYBOOK_MAX_AGE_HOURS), veto_mode=veto_mode,
                    firewall=fw_res, firewall_mode=fw_mode, autotune=at_report, win=win,
                    newest_ts=newest_ts, data_ok=data_ok, hyp_lift=hyp_lift,
                    anatomy=conviction_anatomy(rows, fixed=fixed, cost_pct=cost_pct),
                )
                issues_new, fresh_ids = update_issue_state(_jl(await _strict_get(sdb, KEY_ISSUES), {}), intel_blob["issues"], now_ts)
                intel_text = format_report(intel_blob, veto_mode=veto_mode, firewall_mode=fw_mode, fresh_ids=fresh_ids)
                last_sent = float(prev_intel.get("sent_ts") or 0)
                hours = float(cfg.INTEL_NOTIFY_HOURS)
                intel_due = hours > 0 and (now_ts - last_sent) >= hours * 3600
                intel_blob["sent_ts"] = int(now_ts) if intel_due else int(last_sent)
                blob.update(pub)
                _say(intel_text)
            except Exception as e:
                _say(f"[learner] intelligence skipped: {e!r}")
                intel_blob, veto_res, veto_state_new, fw_state_new, win_pool_new, issues_new = None, None, None, None, None, None
                intel_failed = True

        if args.dry_run:
            _say("[learner] --dry-run: nothing written")
            return 0

        cur = await _strict_get(sdb, KEY_CURRENT)
        cur_state = await _strict_get(sdb, KEY_STATE)
        if intel_failed:
            # The monitor could not run: keep a suspension that is already in force instead of
            # silently turning a suspended veto back on.
            kept = _jl(cur, {}).get("veto_suspended")
            if isinstance(kept, dict) and kept and veto_mode == "live":
                blob["veto_suspended"] = kept
        led = _jl(await _strict_get(sdb, KEY_LEDGER), [])
        led = (led + [e for e in ledger if e.get("decision") != "COLLECTING"])[-_LEDGER_CAP:]
        writes: List[Tuple[str, str]] = []
        if cur and changes:                      # keep the last version only when something changed
            writes.append((KEY_PREV, cur))
            if cur_state:
                writes.append((KEY_STATE_PREV, cur_state))
        writes += [
            (KEY_CURRENT, json_dumps(blob)),
            (KEY_STATE, json_dumps(new_state)),
            (KEY_SCORE, json_dumps(sb)),
            (KEY_LEDGER, json_dumps(led)),
        ]
        if ap_blob is not None:
            writes.append((KEY_AUTOPSY, json_dumps(ap_blob)))
        if intel_blob is not None:
            writes.append((KEY_INTEL, json_dumps(intel_blob)))
        if veto_state_new is not None:
            writes.append((KEY_VETO_MON, json_dumps(veto_state_new)))
        if fw_state_new is not None:
            writes.append((KEY_FIREWALL, json_dumps(fw_state_new)))
        if at_state_new is not None:
            writes.append((KEY_AUTOTUNE, json_dumps(at_state_new)))
        if win_pool_new is not None:
            writes.append((KEY_WIN_POOL, json_dumps(win_pool_new)))
        if issues_new is not None:
            writes.append((KEY_ISSUES, json_dumps(issues_new)))
        if events:
            writes.append((KEY_PROMO_LEDGER, json_dumps(ledger_append(_jl(await _strict_get(sdb, KEY_PROMO_LEDGER), []), events))))
        await _strict_set_many(sdb, writes, _TTL)
        _say(f"[learner] playbook {blob['label']} written ({len(blob['entries'])} entries, {len(changes)} change(s))")

        mode = str(cfg.PLAYBOOK_NOTIFY)
        if mode == "always" or (mode == "changes" and changes):
            await notify(text)
        if ap_blob is not None and ap_due and ap_text:
            await notify(ap_text)
        if intel_blob is not None:
            urgent = [e for e in events if e["kind"] in ("VETO", "FIREWALL", "AUTOTUNE", "HYPOTHESIS_RULE")]
            high_new = [i for i in intel_blob["issues"] if i["id"] in set(fresh_ids) and i["sev"] == "HIGH"]
            head = ""
            if urgent:
                shadow_note = {"VETO": veto_mode, "FIREWALL": fw_mode, "AUTOTUNE": at_mode, "HYPOTHESIS_RULE": str(cfg.ALERT_HYPOTHESIS_VETO_MODE)}
                head = "🧭 INTELLIGENCE — changes this cycle\n" + "\n".join(
                    f"• {e['text']}" + (" (shadow: nothing changed)" if shadow_note.get(e["kind"]) == "shadow" else "") for e in urgent)
            if intel_text and (intel_due or high_new):
                await notify((head + "\n\n" if head else "") + intel_text)
            elif head:
                await notify(head)
        return 0
    finally:
        try:
            await asyncio.wait_for(sdb.close(), timeout=3.0)
        except Exception:
            pass

        # sdb.close() only drops the reference; close the shared Redis pool too
        # (macd_unified does the same at shutdown).
        try:
            await asyncio.wait_for(RedisStateStore.shutdown_global_pool(), timeout=5.0)
        except Exception:
            pass

        # notify() opens the shared Telegram HTTP session AFTER the candle/path
        # fetches already closed theirs; without this the process exits with an
        # "Unclosed client session / Unclosed connector" warning.
        try:
            from fetcher import SessionManager
            await asyncio.wait_for(SessionManager.close_session(), timeout=5.0)
        except Exception:
            pass

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Slow learner: validated trade-plan playbook")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-control", action="store_true")
    ap.add_argument("--no-recon", action="store_true", help="do not rebuild missing candle paths from history")
    ap.add_argument("--rollback", action="store_true")
    ap.add_argument("--digest", action="store_true")
    ap.add_argument("--scoreboard", action="store_true", help="print the last stored scoreboard")
    ap.add_argument("--autopsy", action="store_true", help="print the last stored autopsy / Trust Meter")
    ap.add_argument("--intel", action="store_true", help="print the last stored intelligence report")
    ap.add_argument("--data-dir", default="")
    args = ap.parse_args(argv)
    try:
        return asyncio.run(run(args))
    except Exception as e:
        print(f"[learner] FAILED: {e!r}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
