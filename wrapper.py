#!/usr/bin/env python3
"""
wrapper.py - Entry Point for Unified MACD Trading Bot
Handles initialization, configuration validation, execution, and resource monitoring.
Respects NUMBA_CACHE_DIR for AOT cache detection and warmup fallback.
"""
import asyncio
import os
import signal
import sys
import logging
import psutil
import time
from pathlib import Path
from typing import NoReturn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("wrapper")

# Try uvloop for faster event loop
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    UVLOOP_ENABLED = True
    logger.info("✅ uvloop enabled for faster event loop")
except ImportError:
    UVLOOP_ENABLED = False
    logger.info("ℹ️ uvloop not available, using default event loop")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "src"))

try:
    from src.macd_unified import (
        run_once,
        __version__,
        cfg,
        RedisStateStore,
        SessionManager,
    )
except ImportError as e:
    logger.critical(f"Failed to import core logic: {e}")
    sys.exit(1)


def _handle_signal(signum: int, frame) -> NoReturn:
    """Unified signal handler for graceful shutdown."""
    sig_name = signal.strsignal(signum) if hasattr(signal, "strsignal") else str(signum)
    logger.warning(f"⚠️  Received signal {sig_name} ({signum}) – shutting down")
    raise KeyboardInterrupt


def check_aot_cache() -> None:
    """
    Verify presence of Numba AOT cache and optionally perform warmup.
    Respects NUMBA_CACHE_DIR env. Defaults to /app/src/__pycache__.
    """
    cache_dir = Path(os.environ.get("NUMBA_CACHE_DIR", "/app/src/__pycache__"))
    if not cache_dir.exists():
        logger.warning(f"⚠️  NUMBA cache directory not found: {cache_dir}")
    cache_files = list(cache_dir.rglob("*.nbi"))

    # Expect at least a minimal number of compiled artifacts
    if len(cache_files) >= 15:
        logger.info(f"✅ Using AOT-compiled Numba cache ({len(cache_files)} files) in {cache_dir}")
        return

    # Respect SKIP_WARMUP to avoid JIT warmup
    if getattr(cfg, "SKIP_WARMUP", False):
        logger.warning(
            f"⚠️  AOT cache missing or incomplete in {cache_dir} "
            f"({len(cache_files)} files) and SKIP_WARMUP=true — first use will JIT compile (slower)"
        )
        return

    # Attempt warmup if cache not present and SKIP_WARMUP is false
    logger.info(
        f"⚠️  AOT cache missing or incomplete in {cache_dir} "
        f"({len(cache_files)} files) — performing JIT warmup..."
    )
    try:
        from src.macd_unified import warmup_numba
        warmup_numba()
        # Re-check after warmup
        post_files = list(cache_dir.rglob("*.nbi"))
        logger.info(f"✅ Warmup completed — cache now has {len(post_files)} files in {cache_dir}")
    except Exception as e:
        logger.warning(f"Warmup failed (non-fatal): {e}")


def log_resource_usage(stage: str = "final") -> None:
    """Log memory and CPU usage for performance monitoring."""
    try:
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / 1024 / 1024
        cpu_percent = process.cpu_percent(interval=0.1)

        logger.info(
            f"📊 Resource Usage [{stage}] | "
            f"Memory: {mem_mb:.1f}MB | "
            f"CPU: {cpu_percent:.1f}%"
        )

        # Warn if memory usage is high
        if hasattr(cfg, "MEMORY_LIMIT_BYTES"):
            limit_mb = cfg.MEMORY_LIMIT_BYTES / 1024 / 1024
            if mem_mb > (limit_mb * 0.9):
                logger.warning(
                    f"⚠️  Memory usage at {mem_mb:.1f}MB "
                    f"(90% of {limit_mb:.0f}MB limit)"
                )
    except Exception as e:
        logger.debug(f"Could not log resource usage: {e}")


async def main() -> int:
    """
    Main async entry point with enhanced monitoring.

    Returns:
        0: Success
        2: Runtime error
        3: Unhandled exception
        130: Interrupted (SIGINT/SIGTERM)
    """
    start_time = time.time()

    try:
        # Log startup info
        uvloop_status = "✅ enabled" if UVLOOP_ENABLED else "⚠️  disabled"
        numba_parallel = getattr(cfg, "NUMBA_PARALLEL", False)

        logger.info(
            f"🚀 Bot v{__version__} starting | "
            f"uvloop: {uvloop_status} | "
            f"Numba parallel: {numba_parallel} | "
            f"NUMBA_CACHE_DIR: {os.environ.get('NUMBA_CACHE_DIR', '/app/src/__pycache__')}"
        )

        # Check AOT cache before running bot
        check_aot_cache()

        # Log initial resource usage
        log_resource_usage("startup")

        # Execute main bot logic
        success = await run_once()

        # Calculate execution time
        duration = time.time() - start_time
        logger.info(f"⏱️  Execution time: {duration:.2f}s")

        # Log final resource usage
        log_resource_usage("complete")

        return 0 if success else 2

    except KeyboardInterrupt:
        duration = time.time() - start_time
        logger.warning(f"⚠️  Interrupted after {duration:.1f}s (SIGINT/SIGTERM)")
        return 130

    except asyncio.CancelledError:
        duration = time.time() - start_time
        logger.warning(f"⚠️  Cancelled after {duration:.1f}s (timeout)")
        return 130

    except Exception as exc:
        duration = time.time() - start_time
        logger.exception(f"❌ UNHANDLED EXCEPTION after {duration:.1f}s: {exc}")
        return 3

    finally:
        # Cleanup: Shutdown persistent connections
        try:
            logger.info("🧹 Shutting down persistent connections...")
            await RedisStateStore.shutdown_global_pool()
            await SessionManager.close_session()
            logger.debug("✅ Cleanup complete")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


if __name__ == "__main__":
    # Install signal handlers
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        logger.critical(f"FATAL: Unexpected error in wrapper: {e}")
        sys.exit(3)
