#!/usr/bin/env python3
import asyncio
import os
import signal
import sys
import logging
import psutil
from typing import NoReturn

# Try to use uvloop for performance
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("wrapper")

try:
    from src.macd_unified import run_once, __version__
except ImportError as e:
    logger.critical(f"Failed to import core logic: {e}")
    sys.exit(1)

def log_resource_usage():
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    logger.info(f"📊 Resource Usage: Memory: {mem_mb:.2f} MB")

async def main() -> int:
    try:
        logger.info(f"🚀 Bot v{__version__} starting execution...")
        success = await run_once()
        log_resource_usage()
        return 0 if success else 2
    except KeyboardInterrupt:
        return 130
    except asyncio.CancelledError:
        logger.warning("⏱️ Execution timed out")
        return 130
    except Exception as exc:
        logger.exception(f"🔥 Critical Failure: {exc}")
        return 3

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Graceful shutdown handler
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(loop.stop()))

    try:
        sys.exit(loop.run_until_complete(main()))
    finally:
        loop.close()