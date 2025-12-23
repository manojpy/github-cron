#!/usr/bin/env python3
import asyncio
import os
import signal
import sys
import logging
import psutil
import time
import shutil
from pathlib import Path
from typing import NoReturn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("wrapper")

def setup_runtime_cache():
    """Populate the writable RAM cache with pre-compiled AOT files."""
    source = Path("/app/src/__pycache__")
    target = Path(os.environ.get("NUMBA_CACHE_DIR", "/tmp/numba_cache"))
    
    try:
        if source.exists():
            target.mkdir(parents=True, exist_ok=True)
            count = 0
            for f in source.glob("*.nb*"):
                shutil.copy(f, target)
                count += 1
            logger.info(f"🚀 Loaded {count} AOT files into RAM cache")
    except Exception as e:
        logger.warning(f"Failed to populate RAM cache: {e}")

# CRITICAL: Populate cache BEFORE importing logic
setup_runtime_cache()

# OPTIMIZED: Try uvloop first
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    logger.info("✅ uvloop enabled")
except ImportError:
    logger.info("ℹ️ uvloop not available")

# IMPORT LOGIC (Path is handled by Docker's PYTHONPATH="/app")
try:
    from src.macd_unified import run_once, __version__, cfg, RedisStateStore, SessionManager
except ImportError as e:
    logger.critical(f"Failed to import core logic: {e}")
    sys.exit(1)

async def main() -> int:
    try:
        logger.info(f"🚀 Bot v{__version__} starting")
        success = await run_once()
        return 0 if success else 2
    except Exception as exc:
        logger.exception(f"❌ UNHANDLED EXCEPTION: {exc}")
        return 3
    finally:
        await RedisStateStore.shutdown_global_pool()
        await SessionManager.close_session()

if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(130))
    signal.signal(signal.SIGTERM, lambda s, f: sys.exit(130))
    sys.exit(asyncio.run(main()))