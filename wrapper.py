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
            # Use rglob to ensure we catch all nested Numba artifacts
            cache_files = list(source.rglob("*.nb*"))
            for f in cache_files:
                shutil.copy(f, target / f.name)
            logger.info(f"🚀 Loaded {len(cache_files)} AOT files into RAM cache at {target}")
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
    pass

# IMPORT LOGIC (Path is handled by Docker's PYTHONPATH="/app")
try:
    from src.macd_unified import run_once, __version__, cfg, RedisStateStore, SessionManager
except ImportError as e:
    logger.critical(f"Failed to import core logic: {e}")
    sys.exit(1)

def _handle_signal(signum: int, frame) -> NoReturn:
    sig_name = signal.strsignal(signum) if hasattr(signal, 'strsignal') else str(signum)
    logger.warning(f"⚠️  Received signal {sig_name} – shutting down")
    raise KeyboardInterrupt

def log_resource_usage(stage: str = "final") -> None:
    try:
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / 1024 / 1024
        logger.info(f"📊 Resource Usage [{stage}] | Memory: {mem_mb:.1f}MB")
    except Exception:
        pass

async def main() -> int:
    start_time = time.time()
    try:
        logger.info(f"🚀 Bot v{__version__} starting")
        log_resource_usage("startup")
        
        success = await run_once()
        
        duration = time.time() - start_time
        logger.info(f"⏱️  Execution time: {duration:.2f}s")
        log_resource_usage("complete")
        return 0 if success else 2
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        logger.exception(f"❌ UNHANDLED EXCEPTION: {exc}")
        return 3
    finally:
        await RedisStateStore.shutdown_global_pool()
        await SessionManager.close_session()

if __name__ == "__main__":
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    sys.exit(asyncio.run(main()))