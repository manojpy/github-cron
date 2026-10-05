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
from typing import Any, Dict, List, Optional, Tuple

from bot_config import cfg, json_dumps
from alert_registry import alert_family_of, pretty_alert
from alerts import TelegramQueue, escape_markdown_v2
from archive_reader import load_archived_outcomes
from bot_config import _get_session_from_ts

from feedback import (
    KEY_FEEDBACK, behaviour_report, format_behaviour, format_heat_gate, format_margin,
    heat_gate_report, margin_report,
)
from pathrecon import (
    apply_recon, cache_key, cache_pack, cache_unpack, needs_recon, recon_path, windows_for,
)
from playbook import (
    ControlModel, KEY_CURRENT, KEY_LEDGER, KEY_PREV, KEY_SCORE, KEY_STATE, KEY_STATE_PREV,
    build_playbook, default_params, format_summary,
)
from scoreboard import build_scoreboard, format_scoreboard
from state import RedisStateStore

_TTL = 90 * 86400
KEY_RECON = "path_recon"
_RECON_CAP = 5000
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


async def recover_paths(
    rows: List[Dict[str, Any]], sdb: Any, max_calls: int, dry_run: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Give REAL archived alerts that have no stored candle path one, rebuilt from the
    exchange's candle history (see pathrecon.py).  Results are cached in Redis so each
    row is fetched once.  Never raises: on any problem the rows are returned as they were."""
    stats = {"missing": 0, "from_cache": 0, "rebuilt": 0, "rejected": 0, "calls": 0}
    try:
        need = [r for r in rows if needs_recon(r)]
        stats["missing"] = len(need)
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
                    res = await fetcher.fetch_candles(pair, "15", 1900, end, expected_open_15=end)
                    pdata = parse_candles_to_numpy(res) if res else None
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
                                         fill_price=r.get("fill_price"), fill_delay=fd, lookahead=la)
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
                        dirty = True
            try:
                from fetcher import SessionManager
                await SessionManager.close_session()
            except Exception:
                pass
        if dirty and not dry_run:
            if len(cache) > _RECON_CAP:
                keep_keys = sorted(cache, key=lambda x: int(x.rsplit("|", 1)[-1]))[-_RECON_CAP:]
                cache = {k: cache[k] for k in keep_keys}
            await sdb.set_metadata(KEY_RECON, json_dumps(cache), ttl=_TTL, timeout=10.0)
        out = []
        for r in rows:
            if needs_recon(r):
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
            await tq.send(escape_markdown_v2(text))

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

        if args.rollback:
            prev = await sdb.get_metadata(KEY_PREV, timeout=5.0)
            prev_state_raw = await sdb.get_metadata(KEY_STATE_PREV, timeout=5.0)
            if not prev:
                _say("[learner] no previous playbook stored — nothing to roll back to")
                return 1
            await sdb.set_metadata(KEY_CURRENT, prev, ttl=_TTL, timeout=5.0)
            if prev_state_raw:
                await sdb.set_metadata(KEY_STATE, prev_state_raw, ttl=_TTL, timeout=5.0)
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

        prev_state = _jl(await sdb.get_metadata(KEY_STATE, timeout=5.0), {})
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
        prev_sb = _jl(await sdb.get_metadata(KEY_SCORE, timeout=5.0), {})
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
        if rstats.get("rebuilt") or rstats.get("from_cache"):

            text += (f"\n\n🧩 Recovered candle paths for {rstats.get('rebuilt', 0) + rstats.get('from_cache', 0)} "
                     f"older real alert(s) from exchange history (tagged RECON).")
        _say(text)

        if args.dry_run:
            _say("[learner] --dry-run: nothing written")
            return 0

        cur = await sdb.get_metadata(KEY_CURRENT, timeout=5.0)
        cur_state = await sdb.get_metadata(KEY_STATE, timeout=5.0)
        if cur and changes:                      # keep the last version only when something changed
            await sdb.set_metadata(KEY_PREV, cur, ttl=_TTL, timeout=5.0)
            if cur_state:
                await sdb.set_metadata(KEY_STATE_PREV, cur_state, ttl=_TTL, timeout=5.0)
        await sdb.set_metadata(KEY_CURRENT, json_dumps(blob), ttl=_TTL, timeout=5.0)
        await sdb.set_metadata(KEY_STATE, json_dumps(new_state), ttl=_TTL, timeout=5.0)
        await sdb.set_metadata(KEY_SCORE, json_dumps(sb), ttl=_TTL, timeout=5.0)
        led = _jl(await sdb.get_metadata(KEY_LEDGER, timeout=5.0), [])
        led = (led + [e for e in ledger if e.get("decision") != "COLLECTING"])[-_LEDGER_CAP:]
        await sdb.set_metadata(KEY_LEDGER, json_dumps(led), ttl=_TTL, timeout=5.0)
        _say(f"[learner] playbook {blob['label']} written ({len(blob['entries'])} entries, {len(changes)} change(s))")

        mode = str(cfg.PLAYBOOK_NOTIFY)
        if mode == "always" or (mode == "changes" and changes):
            await notify(text)
        return 0
    finally:
        try:
            await sdb.close()
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
    ap.add_argument("--data-dir", default="")
    args = ap.parse_args(argv)
    try:
        return asyncio.run(run(args))
    except Exception as e:
        print(f"[learner] FAILED: {e!r}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
