#!/usr/bin/env python3
import asyncio
import os
import signal
import sys
import logging
from typing import NoReturn

# Optimization: Force uvloop usage
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, "src")
sys.path.insert(0, SRC_DIR)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
fallback_logger = logging.getLogger("wrapper")

try:
    from src.macd_unified import run_once, logger, cfg, __version__
except ImportError as e:
    fallback_logger.error(f"❌ CRITICAL: Failed to import macd_unified.py: {e}")
    sys.exit(1)

def _handle_signal(signum: int, frame) -> NoReturn:
    raise KeyboardInterrupt

async def main() -> int:
    try:
        logger.info(f"🚀 Starting Optimized Trading Bot v{__version__} (uvloop active)")
        success = await run_once()
        return 0 if success else 2
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        logger.exception(f"❌ UNHANDLED EXCEPTION: {exc}")
        return 3

if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    try:
        sys.exit(asyncio.run(main()))
    except Exception as e:
        fallback_logger.critical(f"FATAL: {e}")
        sys.exit(3)