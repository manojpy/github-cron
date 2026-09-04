"""
Standalone entrypoint: generate a Brain analysis report and send it to
Telegram, independent of the main bot's run cadence / run-count throttle.

Runs inside the same Docker image as macd_unified.py (needs the same
Redis/pydantic/AOT-adjacent module graph via brain -> alerts -> state/gates),
but skips candle fetching, gate evaluation, AOT init, and numeric self-test —
it only reads existing Redis outcome data and sends the report.

Exit codes:
  0 = report generated and sent (or skipped cleanly by a config guard)
  1 = failed to generate/send
"""
from __future__ import annotations
import asyncio
import logging
import sys

from bot_config import cfg
from state import RedisStateStore
from alerts import TelegramQueue
from brain_enhanced import BrainEngineV2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("send_brain_report")


async def main() -> int:
    sdb = RedisStateStore(cfg.REDIS_URL)
    await sdb.connect()
    if sdb.degraded:
        logger.error("Redis unreachable — cannot generate brain report")
        return 1

    telegram_queue = TelegramQueue(cfg.TELEGRAM_BOT_TOKEN, cfg.TELEGRAM_CHAT_ID)

    try:
        await BrainEngineV2(sdb).send_report_now(cfg.PAIRS, telegram_queue, logger)
    except Exception as e:
        logger.critical(f"Brain report failed: {e}", exc_info=True)
        return 1
    finally:
        try:
            await sdb._redis.aclose()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
