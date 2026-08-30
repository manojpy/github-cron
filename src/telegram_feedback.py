#!/usr/bin/env python3
"""
telegram_feedback.py — reinforcement-loop layer: attaches "Took Trade" /
"Skipped" inline buttons to alert messages, and polls Telegram's getUpdates
(no webhook — this bot has no always-on server, so it catches button taps
made since the previous cron run) to record which ones the user acted on.

Design constraints this respects:
  - The bot is a fresh process every 15 minutes (GitHub Actions cron), not
    a long-running server, so there's no webhook endpoint to receive
    callback_query updates. Polling getUpdates once per run, with the
    offset persisted in Redis, is the only fit for this architecture.
  - A tap on a button sent runs ago must still resolve correctly even
    though this is a brand-new process with no memory of that message —
    all context (pair, alert_keys, entry_ts, direction) is looked up from
    Redis via the feedback_id embedded in callback_data, never held in
    memory.
  - Idempotent: Telegram can redeliver the same update on retry. The
    feedback_pending Redis key is deleted on first successful use, so a
    duplicate delivery finds nothing and is a safe no-op (still
    acknowledged to Telegram, just not re-logged).
  - Bounded blast radius: getUpdates is called with a short timeout, and
    only callback_query updates from the configured chat_id are processed
    — a run must not hang, and must not act on someone else's taps if the
    bot token is ever reused elsewhere.

This module owns all direct Telegram Bot API calls for the feedback loop
(getUpdates / answerCallbackQuery / editMessageReplyMarkup). Sending the
buttoned message itself is TelegramQueue.send_with_markup() in alerts.py,
kept there since it shares that class's rate limiter and retry logic.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from bot_config import cfg, json_dumps, json_loads
from state import RedisStateStore
from fetcher import SessionManager

FEEDBACK_PENDING_PREFIX = "feedback_pending:"
FEEDBACK_LOG_STREAM = "feedback_log_stream"
TELEGRAM_UPDATE_OFFSET_KEY = "telegram_update_offset"

_ACTIONS = {
    "took": "✅ Took Trade",
    "skip": "❌ Skipped",
}


def build_feedback_keyboard(feedback_id: str) -> Dict[str, Any]:
    return {
        "inline_keyboard": [[
            {"text": "✅ Took Trade", "callback_data": f"fb:{feedback_id}:took"},
            {"text": "❌ Skipped", "callback_data": f"fb:{feedback_id}:skip"},
        ]]
    }


async def record_feedback_pending(
    sdb: RedisStateStore, feedback_id: str, pair: str, alert_keys: List[str],
    entry_ts: int, direction: str, message_id: Optional[int],
) -> None:
    """Store what a feedback_id refers to, so a button tap on a fresh cron
    process (which has no memory of having sent the message) can still be
    resolved. TTL-bound — an unanswered alert eventually just expires."""
    if sdb.degraded or not sdb._redis:
        return
    key = f"{FEEDBACK_PENDING_PREFIX}{feedback_id}"
    payload = json_dumps({
        "pair": pair, "alert_keys": alert_keys, "entry_ts": entry_ts,
        "direction": direction, "message_id": message_id,
    })
    ttl = getattr(cfg, "TELEGRAM_FEEDBACK_TTL_HOURS", 24) * 3600
    try:
        await asyncio.wait_for(sdb._redis.set(key, payload, ex=ttl), timeout=2.0)
    except Exception as e:
        logging.getLogger("macd_bot").warning(f"Failed to record feedback pending {feedback_id}: {e}")


async def _telegram_api(method: str, token: str, payload: Dict[str, Any], timeout: float = 10.0) -> Optional[Dict[str, Any]]:
    session = await SessionManager.get_session()
    url = f"https://api.telegram.org/bot{token}/{method}"
    try:
        async with session.post(url, data=payload, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            return await resp.json()
    except Exception as e:
        logging.getLogger("macd_bot").warning(f"Telegram {method} failed: {e}")
        return None


async def poll_and_process_feedback(
    sdb: RedisStateStore, token: str, chat_id: str, logger_run: logging.Logger,
) -> int:
    """Call once per cron run. Fetches any callback_query updates since the
    last run, resolves each against its feedback_pending record, logs it to
    FEEDBACK_LOG_STREAM, acknowledges it on Telegram, and advances the
    stored offset. Returns the number of feedback events processed."""
    if not getattr(cfg, "ENABLE_TELEGRAM_FEEDBACK", False):
        return 0
    if sdb.degraded or not sdb._redis:
        return 0

    offset_raw = await sdb.get_metadata(TELEGRAM_UPDATE_OFFSET_KEY)
    offset = int(offset_raw) + 1 if offset_raw else None

    params: Dict[str, Any] = {"timeout": 0, "allowed_updates": json_dumps(["callback_query"])}
    if offset is not None:
        params["offset"] = offset

    resp = await _telegram_api("getUpdates", token, params, timeout=15.0)
    if not resp or not resp.get("ok"):
        return 0
    updates = resp.get("result", [])
    if not updates:
        return 0

    processed = 0
    max_update_id = offset - 1 if offset is not None else 0
    for update in updates:
        update_id = update.get("update_id", 0)
        max_update_id = max(max_update_id, update_id)
        cq = update.get("callback_query")
        if not cq:
            continue
        cq_chat_id = str(cq.get("message", {}).get("chat", {}).get("id", ""))
        if cq_chat_id != str(chat_id):
            # Not our chat — never act on it, but still ack so it isn't
            # redelivered forever by getUpdates.
            await _telegram_api("answerCallbackQuery", token, {"callback_query_id": cq["id"]})
            continue

        data = cq.get("data", "")
        parts = data.split(":")
        if len(parts) != 3 or parts[0] != "fb" or parts[2] not in _ACTIONS:
            await _telegram_api("answerCallbackQuery", token, {"callback_query_id": cq["id"]})
            continue
        _, feedback_id, action = parts

        key = f"{FEEDBACK_PENDING_PREFIX}{feedback_id}"
        try:
            raw = await asyncio.wait_for(sdb._redis.get(key), timeout=2.0)
        except Exception:
            raw = None

        if not raw:
            # Expired, or already handled by a redelivered update — ack
            # and move on rather than re-logging.
            await _telegram_api("answerCallbackQuery", token, {
                "callback_query_id": cq["id"], "text": "This alert has expired.",
            })
            continue

        try:
            meta = json_loads(raw)
        except Exception:
            meta = {}

        # Delete first — makes a redelivered update for the same tap land
        # in the "not found" branch above instead of double-logging.
        try:
            await asyncio.wait_for(sdb._redis.delete(key), timeout=2.0)
        except Exception:
            pass

        try:
            await asyncio.wait_for(
                sdb._redis.xadd(
                    FEEDBACK_LOG_STREAM,
                    {
                        "feedback_id": feedback_id,
                        "pair": meta.get("pair", "?"),
                        "alert_keys": json_dumps(meta.get("alert_keys", [])),
                        "direction": meta.get("direction", "?"),
                        "entry_ts": meta.get("entry_ts", 0),
                        "action": action,
                        "responded_ts": int(time.time()),
                    },
                    maxlen=20000, approximate=True,
                ),
                timeout=3.0,
            )
            processed += 1
        except Exception as e:
            logger_run.warning(f"Failed to log feedback {feedback_id}: {e}")

        await _telegram_api("answerCallbackQuery", token, {
            "callback_query_id": cq["id"], "text": f"Recorded: {_ACTIONS[action]}",
        })

        message_id = meta.get("message_id")
        if message_id:
            await _telegram_api("editMessageReplyMarkup", token, {
                "chat_id": chat_id, "message_id": message_id,
                "reply_markup": json_dumps({"inline_keyboard": [[
                    {"text": f"{_ACTIONS[action]} \u2713", "callback_data": "fb:noop:noop"},
                ]]}),
            })

    if max_update_id:
        try:
            await sdb.set_metadata(TELEGRAM_UPDATE_OFFSET_KEY, str(max_update_id), ttl=30 * 86400)
        except Exception as e:
            logger_run.warning(f"Failed to persist telegram update offset: {e}")

    if processed:
        logger_run.info(f"\U0001F4E9 Processed {processed} Telegram feedback response(s)")
    return processed