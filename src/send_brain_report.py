"""
Standalone entrypoint: generate a Brain analysis report and send it to Telegram.
"""

from __future__ import annotations

import asyncio
import logging
import sys

from bot_config import cfg
from state import RedisStateStore
from alerts import TelegramQueue
from brain_enhanced import BrainEngineV2
from fetcher import SessionManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("send_brain_report")


async def _close_telegram_queue(tq: TelegramQueue | None) -> None:
    if tq is None:
        return

    # Preferred: explicit close method
    try:
        close_fn = getattr(tq, "close", None)
        if callable(close_fn):
            await close_fn()
            await asyncio.sleep(0.1)
            return
    except Exception:
        logger.exception("TelegramQueue.close() failed")

    # Fallback: common aiohttp session attribute names
    for attr in ("_session", "session", "_http", "http"):
        try:
            session = getattr(tq, attr, None)
            if session is not None and hasattr(session, "closed") and not session.closed:
                await session.close()
                await asyncio.sleep(0.1)
                return
        except Exception:
            logger.exception(f"Failed closing TelegramQueue session attr={attr}")


async def main() -> int:
    sdb = RedisStateStore(cfg.REDIS_URL)
    telegram_queue = None
    exit_code = 0

    try:
        await sdb.connect()

        if sdb.degraded:
            logger.error("Redis unreachable — cannot generate brain report")
            return 1

        telegram_queue = TelegramQueue(
            cfg.TELEGRAM_BOT_TOKEN,
            cfg.TELEGRAM_CHAT_ID,
        )

        sent = await BrainEngineV2(sdb).send_report_now(
            cfg.PAIRS,
            telegram_queue,
            logger,
        )

        if sent:
            logger.info("✅ Brain report delivered")
        else:
            logger.error("❌ Brain report was not delivered to Telegram")
            exit_code = 1

    except Exception as e:
        logger.critical(f"Brain report failed: {e}", exc_info=True)
        exit_code = 1

    finally:
        # 1) Close Telegram HTTP session if TelegramQueue owns one
        try:
            await asyncio.wait_for(_close_telegram_queue(telegram_queue), timeout=5)
        except Exception:
            logger.exception("Telegram cleanup failed")

        # 2) Also close shared SessionManager, in case TelegramQueue used it
        try:
            await asyncio.wait_for(SessionManager.close_session(), timeout=5)
        except Exception:
            logger.exception("SessionManager cleanup failed")

        # 3) Close Redis properly
        try:
            await asyncio.wait_for(sdb.close(), timeout=3)
        except Exception:
            logger.exception("RedisStateStore.close() failed")

        try:
            await asyncio.wait_for(RedisStateStore.shutdown_global_pool(), timeout=5)
        except Exception:
            logger.exception("Redis global pool shutdown failed")

    return exit_code


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))