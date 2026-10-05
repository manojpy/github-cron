"""feedback — "Took it / Skipped it" buttons and what they teach the Brain.

Every alert message can carry one row of two buttons per pair:  ✅ Took  /  ⏭ Skip.
The bot is a cron job, not a running service, so a tap is not handled instantly:
each 15-minute run first drains Telegram's pending updates (one getUpdates call),
records the decisions in Redis and turns the tapped row into a confirmation.

Optional real fills: reply to the alert with
        BTC exit 64250            or            BTC entry 64000 exit 64250
and the numbers are attached to your latest "Took" for that pair.

Everything here is best-effort and never raises into the alert path.  Only
updates from the configured chat are accepted.
"""
from __future__ import annotations

import json
import re
import statistics
import time
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from plan_replay import Row, replay_plan

KEY_FEEDBACK = "feedback_log"
KEY_OFFSET = "tg_updates_offset"
_LOG_CAP = 3000
_TTL = 90 * 86400
_PREFIX = "ts"
_FILL_RE = re.compile(r"^\s*([A-Za-z0-9]{2,12})\s+(?:entry\s+([0-9]*\.?[0-9]+)\s+)?exit\s+([0-9]*\.?[0-9]+)\s*$", re.I)


def _short(pair: str) -> str:
    return pair[:-3] if pair.endswith("USD") and len(pair) > 3 else pair


def decision_key(pair: str, direction: str, ts: int) -> str:
    return f"{pair}|{direction.lower()}|{int(ts)}"


def build_keyboard(items: Sequence[Tuple[str, str, int]], max_rows: int = 8) -> Optional[Dict[str, Any]]:
    """items = [(pair, 'buy'|'sell', candle_ts)] -> Telegram inline keyboard, or None."""
    rows: List[List[Dict[str, str]]] = []
    seen = set()
    for pair, direction, ts in items:
        k = (pair, direction)
        if k in seen:
            continue
        seen.add(k)
        d = "b" if str(direction).lower() == "buy" else "s"
        label = f"{_short(pair)} {'BUY' if d == 'b' else 'SELL'}"
        took = f"{_PREFIX}:T:{pair}:{d}:{int(ts)}"
        skip = f"{_PREFIX}:S:{pair}:{d}:{int(ts)}"
        if len(took.encode()) > 64:
            continue
        rows.append([{"text": f"✅ Took {label}", "callback_data": took},
                     {"text": f"⏭ Skip {label}", "callback_data": skip}])
        if len(rows) >= max_rows:
            break
    return {"inline_keyboard": rows} if rows else None


def parse_callback(data: Any) -> Optional[Dict[str, Any]]:
    parts = str(data or "").split(":")
    if len(parts) != 5 or parts[0] != _PREFIX or parts[1] not in ("T", "S") or parts[3] not in ("b", "s"):
        return None
    try:
        ts = int(parts[4])
    except ValueError:
        return None
    return {"decision": parts[1], "pair": parts[2], "direction": "buy" if parts[3] == "b" else "sell", "ts": ts}


def keyboard_after_press(kb: Optional[Mapping[str, Any]], data: str, decision: str) -> Dict[str, Any]:
    """The tapped row becomes a single confirmation button; other rows stay."""
    rows_in = (kb or {}).get("inline_keyboard") or []
    out: List[List[Dict[str, str]]] = []
    for row in rows_in:
        if any(b.get("callback_data") == data for b in row):
            p = parse_callback(data)
            label = f"{_short(p['pair'])} {p['direction'].upper()}" if p else ""
            txt = f"✅ Taken: {label}" if decision == "T" else f"⏭ Skipped: {label}"
            out.append([{"text": txt, "callback_data": "noop"}])
        else:
            out.append([dict(b) for b in row])
    return {"inline_keyboard": out}


def parse_fill_reply(text: str, pairs: Sequence[str]) -> Optional[Dict[str, Any]]:
    m = _FILL_RE.match(text or "")
    if not m:
        return None
    sym = m.group(1).upper()
    pair = next((p for p in pairs if p.upper() == sym or p.upper() == sym + "USD"), None)
    if pair is None:
        return None
    return {"pair": pair, "entry": float(m.group(2)) if m.group(2) else None, "exit": float(m.group(3))}


def apply_updates(
    log: Dict[str, Any], updates: Sequence[Mapping[str, Any]], *, chat_id: str, pairs: Sequence[str], now: float,
) -> Tuple[List[Dict[str, Any]], int]:
    """Pure: fold Telegram updates into `log`.  Returns (callback actions to answer, n_changes)."""
    actions: List[Dict[str, Any]] = []
    changes = 0
    cur_kb: Dict[Any, Any] = {}          # message_id -> keyboard after earlier taps in this batch
    for u in updates:
        cq = u.get("callback_query")
        if cq:
            msg = cq.get("message") or {}
            if str((msg.get("chat") or {}).get("id")) != str(chat_id):
                continue
            data = cq.get("data")
            p = parse_callback(data)
            if p is None:
                if data == "noop":
                    actions.append({"id": cq.get("id"), "text": "Already recorded", "edit": None})
                continue
            log[decision_key(p["pair"], p["direction"], p["ts"])] = {"d": p["decision"], "at": int(now)}
            changes += 1
            mid = msg.get("message_id")
            new_kb = keyboard_after_press(cur_kb.get(mid, msg.get("reply_markup")), str(data), p["decision"])
            cur_kb[mid] = new_kb
            actions.append({
                "id": cq.get("id"), "text": "Recorded: " + ("took it" if p["decision"] == "T" else "skipped"),
                "edit": {"chat_id": msg.get("chat", {}).get("id"), "message_id": msg.get("message_id"),
                         "markup": new_kb},
            })
            continue
        m = u.get("message")
        if m and str((m.get("chat") or {}).get("id")) == str(chat_id) and m.get("text"):
            fill = parse_fill_reply(m["text"], pairs)
            if fill:
                cands = [(k, v) for k, v in log.items()
                         if k.startswith(fill["pair"] + "|") and v.get("d") == "T" and v.get("exit") is None]
                if cands:
                    k, v = max(cands, key=lambda kv: int(kv[0].rsplit("|", 1)[-1]))
                    v["exit"] = fill["exit"]
                    if fill["entry"] is not None:
                        v["entry"] = fill["entry"]
                    changes += 1
    return actions, changes


async def _tg_call(token: str, method: str, params: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    import aiohttp
    url = f"https://api.telegram.org/bot{token}/{method}"
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=8)) as s:
            async with s.post(url, json=dict(params)) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                return data if data.get("ok") else None
    except Exception:
        return None


async def poll_feedback(
    sdb: Any, token: str, chat_id: str, pairs: Sequence[str], *,
    call: Optional[Callable[[str, Mapping[str, Any]], Awaitable[Optional[Dict[str, Any]]]]] = None,
    now: Optional[float] = None,
) -> int:
    """Drain pending Telegram updates once.  Returns the number of recorded changes.
    Never raises."""
    try:
        async def _call(method: str, params: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
            return await (call(method, params) if call else _tg_call(token, method, params))

        raw_off = await sdb.get_metadata(KEY_OFFSET, timeout=3.0)
        offset = int(raw_off) if raw_off else 0
        resp = await _call("getUpdates", {"offset": offset, "timeout": 0, "limit": 100,
                                          "allowed_updates": ["callback_query", "message"]})
        updates = (resp or {}).get("result") or []
        if not updates:
            return 0
        raw_log = await sdb.get_metadata(KEY_FEEDBACK, timeout=3.0)
        try:
            log: Dict[str, Any] = json.loads(raw_log) if raw_log else {}
        except Exception:
            log = {}
        updates = sorted(updates, key=lambda u: u.get("update_id", 0))
        actions, changes = apply_updates(log, updates, chat_id=str(chat_id), pairs=pairs,
                                         now=now if now is not None else time.time())
        if changes:
            if len(log) > _LOG_CAP:
                keep = sorted(log, key=lambda k: int(k.rsplit("|", 1)[-1]))[-_LOG_CAP:]
                log = {k: log[k] for k in keep}
            await sdb.set_metadata(KEY_FEEDBACK, json.dumps(log, separators=(",", ":")), ttl=_TTL, timeout=3.0)
        await sdb.set_metadata(KEY_OFFSET, str(int(updates[-1]["update_id"]) + 1), ttl=30 * 86400, timeout=3.0)
        for a in actions:                     # best-effort UX: stale taps may fail, the record is already saved
            if a.get("id"):
                await _call("answerCallbackQuery", {"callback_query_id": a["id"], "text": a["text"]})
            e = a.get("edit")
            if e and e.get("message_id") is not None:
                await _call("editMessageReplyMarkup", {"chat_id": e["chat_id"], "message_id": e["message_id"],
                                                       "reply_markup": e["markup"]})
        return changes
    except Exception:
        return 0


# ── Learner side: what did the decisions teach? ───────────────────────────
def behaviour_report(
    log: Mapping[str, Any], rows: Sequence[Row], *, fixed: Mapping[str, float], cost_pct: float,
) -> Dict[str, Any]:
    """Taken vs skipped outcomes under the fixed plan, plus real-fill gaps.
    Rows are matched on (pair, direction, entry_ts); unmatched decisions are counted, never guessed."""
    idx: Dict[str, Row] = {}
    for row in rows:
        if row.get("path_fav") and row.get("entry_ts") is not None:
            idx.setdefault(decision_key(str(row.get("pair")), str(row.get("direction")), int(row["entry_ts"])), row)
    taken: List[float] = []
    skipped: List[float] = []
    unmatched = pending = 0
    gaps: List[float] = []
    for k, v in log.items():
        r = idx.get(k)
        if r is None:
            if int(k.rsplit("|", 1)[-1]) > time.time() - 4 * 3600:
                pending += 1
            else:
                unmatched += 1
            continue
        res = replay_plan(r, float(fixed["sl"]), float(fixed["tp"]), int(fixed["h"]), cost_pct)
        if res is None:
            pending += 1
            continue
        (taken if v.get("d") == "T" else skipped).append(res[1])
        if v.get("d") == "T" and v.get("exit") and v.get("entry"):
            sign = 1.0 if str(r.get("direction")).lower() == "buy" else -1.0
            real = sign * (float(v["exit"]) - float(v["entry"])) / float(v["entry"]) * 100.0 - cost_pct
            gaps.append(real - res[1])

    def _s(x: List[float]) -> Dict[str, Any]:
        return {"n": len(x), "ev": statistics.fmean(x) if x else None,
                "win": (sum(1 for p in x if p > 0) / len(x)) if x else None}
    return {"taken": _s(taken), "skipped": _s(skipped), "unmatched": unmatched, "pending": pending,
            "fills": {"n": len(gaps), "mean_gap": statistics.fmean(gaps) if gaps else None}}


def format_behaviour(b: Mapping[str, Any]) -> str:
    def f(s: Mapping[str, Any]) -> str:
        return "n=0" if not s["n"] else f"n={s['n']}, EV {s['ev']:+.2f}%, wins {s['win']:.0%}"
    out = ["🧭 YOUR DECISIONS (fixed plan)", f"  Took: {f(b['taken'])}", f"  Skipped: {f(b['skipped'])}"]
    t, s = b["taken"], b["skipped"]
    if t["n"] >= 10 and s["n"] >= 10 and s["ev"] is not None and t["ev"] is not None and s["ev"] > t["ev"]:
        out.append("  ⚠ the alerts you skipped did BETTER than the ones you took")
    if b["fills"]["n"]:
        out.append(f"  Real fills vs model: {b['fills']['mean_gap']:+.2f}% per trade (n={b['fills']['n']})")
    if b["unmatched"] or b["pending"]:
        out.append(f"  ({b['pending']} still resolving, {b['unmatched']} not matched to an outcome)")
    return "\n".join(out)
