#!/usr/bin/env python3
"""
wrapper.py - Entry Point for Unified MACD Trading Bot
Handles initialization, configuration validation, execution, and resource monitoring.
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


# OPTIMIZED: Try uvloop first for 2-4x faster event loop
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    UVLOOP_ENABLED = True
    logger.info("✅ uvloop enabled for 2-4x faster event loop")
except ImportError:
    UVLOOP_ENABLED = False
    logger.info("ℹ️ uvloop not available, using default event loop")


# 🔥 FIX: Set PYTHONPATH and import correctly
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, "src")

# Add src to path BEFORE importing
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Also ensure the app directory is in the path
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

try:
    # 🔥 CRITICAL: Import directly from macd_unified, not src.macd_unified
    import macd_unified
    from macd_unified import run_once, __version__, cfg, RedisStateStore, SessionManager

except ImportError as e:
    logger.critical(f"Failed to import core logic: {e}")
    logger.critical(f"sys.path: {sys.path}")
    logger.critical(f"SRC_DIR: {SRC_DIR}")
    logger.critical(f"Files in SRC_DIR: {os.listdir(SRC_DIR) if os.path.exists(SRC_DIR) else 'DIR NOT FOUND'}")
    sys.exit(1)

def _handle_signal(signum: int, frame) -> NoReturn:
    """Unified signal handler for graceful shutdown."""
    sig_name = signal.strsignal(signum) if hasattr(signal, 'strsignal') else str(signum)
    logger.warning(f"⚠️  Received signal {sig_name} ({signum}) – shutting down")
    raise KeyboardInterrupt

def check_aot_cache() -> None:
    """
    Check if AOT-compiled Numba cache exists and is valid.
    
    Returns early if cache is found, otherwise triggers warmup if needed.
    """
    cache_dir = Path(os.getenv("NUMBA_CACHE_DIR", "/app/src/__pycache__"))
    
    # Check if directory exists
    if not cache_dir.exists():
        logger.warning(f"⚠️  Cache directory does not exist: {cache_dir}")
        _handle_missing_cache()
        return
    
    # Check for all Numba cache file types
    nbi_files = list(cache_dir.rglob("*.nbi"))
    nbc_files = list(cache_dir.rglob("*.nbc"))
    all_cache_files = nbi_files + nbc_files
    
    # Log detailed cache status
    logger.info(
        f"📁 AOT Cache check | "
        f"Directory: {cache_dir} | "
        f"Exists: {cache_dir.exists()} | "
        f"Index files (.nbi): {len(nbi_files)} | "
        f"Binary files (.nbc): {len(nbc_files)} | "
        f"Total: {len(all_cache_files)}"
    )
    
    # List first few files for debugging
    if all_cache_files and len(all_cache_files) < 20:
        logger.debug(f"Cache files: {[f.name for f in all_cache_files[:5]]}")

    # Threshold: expect at least 15 compiled functions (conservative estimate)
    EXPECTED_MIN_CACHE_FILES = 15
    
    if len(all_cache_files) >= EXPECTED_MIN_CACHE_FILES:
        logger.info(
            f"✅ Using AOT-compiled Numba cache ({len(all_cache_files)} files) - "
            f"no warmup needed"
        )
        return
    
    # Cache insufficient or missing
    logger.warning(
        f"⚠️  Insufficient AOT cache files: {len(all_cache_files)} < {EXPECTED_MIN_CACHE_FILES}"
    )
    _handle_missing_cache()

def _handle_missing_cache() -> None:
    """Handle missing or insufficient AOT cache."""
    skip_warmup = os.getenv("SKIP_WARMUP", "false").lower() in ("true", "1", "yes")
    
    if skip_warmup:
        logger.warning(
            "⚠️  SKIP_WARMUP=true - functions will JIT compile on first use (slower startup)\n"
            "    First run will take 30-60s longer due to JIT compilation"
        )
        return
    
    logger.info("🔧 Performing JIT warmup compilation (this will take ~10-20s)...")
    warmup_start = time.time()
    
    try:
        from macd_unified import warmup_numba
        warmup_numba()
        warmup_duration = time.time() - warmup_start
        logger.info(f"✅ JIT warmup completed in {warmup_duration:.1f}s")
    except Exception as e:
        logger.warning(f"⚠️  Warmup failed (non-fatal): {e}")
        logger.info("Functions will compile on first use instead")

# OPTIMIZED: Resource monitoring function
def log_resource_usage(stage: str = "final") -> None:
    """Log memory and CPU usage for performance monitoring."""
    try:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        mem_mb = mem_info.rss / 1024 / 1024
        
        # Get CPU percent with short interval to avoid blocking
        cpu_percent = process.cpu_percent(interval=0.1)
        
        logger.info(
            f"📊 Resource Usage [{stage}] | "
            f"Memory: {mem_mb:.1f}MB | "
            f"CPU: {cpu_percent:.1f}%"
        )
        
        # Warn if memory usage is high
        if hasattr(cfg, 'MEMORY_LIMIT_BYTES'):
            limit_mb = cfg.MEMORY_LIMIT_BYTES / 1024 / 1024
            usage_percent = (mem_mb / limit_mb) * 100
            
            if usage_percent > 90:
                logger.warning(
                    f"⚠️  High memory usage: {mem_mb:.1f}MB "
                    f"({usage_percent:.0f}% of {limit_mb:.0f}MB limit)"
                )
            elif usage_percent > 75:
                logger.info(
                    f"📊 Memory usage: {usage_percent:.0f}% of limit "
                    f"({mem_mb:.1f}MB / {limit_mb:.0f}MB)"
                )
    except Exception as e:
        logger.debug(f"Could not log resource usage: {e}")

def log_startup_info() -> None:
    """Log comprehensive startup information."""
    uvloop_status = "✅ enabled" if UVLOOP_ENABLED else "⚠️  disabled"
    numba_parallel = getattr(cfg, 'NUMBA_PARALLEL', False)
    skip_warmup = os.getenv("SKIP_WARMUP", "false").lower() in ("true", "1", "yes")
    
    logger.info("=" * 70)
    logger.info(f"🚀 MACD Trading Bot v{__version__}")
    logger.info("=" * 70)
    logger.info(f"Event Loop:      {uvloop_status}")
    logger.info(f"Numba Parallel:  {numba_parallel}")
    logger.info(f"Skip Warmup:     {skip_warmup}")
    logger.info(f"Cache Dir:       {os.getenv('NUMBA_CACHE_DIR', '/app/src/__pycache__')}")
    logger.info(f"Numba Threads:   {os.getenv('NUMBA_NUM_THREADS', '4')}")
    logger.info(f"Python Version:  {sys.version.split()[0]}")
    logger.info("=" * 70)

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
        # Log comprehensive startup info
        log_startup_info()
        
        # 🔥 CHECK AOT CACHE BEFORE RUNNING BOT
        logger.info("🔍 Checking Numba AOT cache...")
        check_aot_cache()
        
        # Log initial resource usage
        log_resource_usage("startup")
        
        logger.info("▶️  Starting bot execution...")
        
        # Execute main bot logic
        success = await run_once()
        
        # Calculate execution time
        duration = time.time() - start_time
        
        if success:
            logger.info(f"✅ Bot execution completed successfully in {duration:.2f}s")
        else:
            logger.error(f"❌ Bot execution failed after {duration:.2f}s")
        
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
        # 🔥 CLEANUP: Shutdown persistent connections
        try:
            logger.info("🧹 Shutting down persistent connections...")
            cleanup_start = time.time()
            
            await RedisStateStore.shutdown_global_pool()
            logger.debug("  ✅ Redis pool closed")
            
            await SessionManager.close_session()
            logger.debug("  ✅ HTTP session closed")
            
            cleanup_duration = time.time() - cleanup_start
            logger.info(f"✅ Cleanup completed in {cleanup_duration:.2f}s")
            
        except Exception as e:
            logger.error(f"⚠️  Error during cleanup: {e}")

if __name__ == "__main__":
    # Install signal handlers
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        logger.info("👋 Shutdown complete")
        sys.exit(130)
        
    except Exception as e:
        logger.critical(f"💥 FATAL: Unexpected error in wrapper: {e}")
        sys.exit(3)