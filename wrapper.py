#!/usr/bin/env python3
"""
wrapper.py - Entry Point for Unified MACD Trading Bot
Handles initialization, configuration validation, execution, and graceful error reporting.
"""

import asyncio
import os
import signal
import sys
import logging
from typing import NoReturn

# -----------------------------------------------------------------------------
# Path setup (idempotent, safe)
# -----------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# -----------------------------------------------------------------------------
# Fallback logger (used ONLY if macd_unified logger is unavailable)
# -----------------------------------------------------------------------------
_fallback_logging_configured = False
def _setup_fallback_logging() -> logging.Logger:
    global _fallback_logging_configured
    if not _fallback_logging_configured:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s"
        )
        _fallback_logging_configured = True
    return logging.getLogger("wrapper")

fallback_logger = _setup_fallback_logging()

# -----------------------------------------------------------------------------
# Early import validation
# -----------------------------------------------------------------------------
try:
    from src.macd_unified import run_once, logger, cfg, __version__
except ImportError as e:
    fallback_logger.error(f"❌ CRITICAL: Failed to import macd_unified.py: {e}")
    fallback_logger.error("💡 Check that src/macd_unified.py exists and the image was built correctly.")
    sys.exit(1)
except Exception as e:
    fallback_logger.error(f"❌ CONFIGURATION ERROR: {e}")
    fallback_logger.error("💡 Verify config_macd.json syntax and required fields.")
    sys.exit(1)

# -----------------------------------------------------------------------------
# Signal handling
# -----------------------------------------------------------------------------
def _handle_signal(signum: int, frame) -> NoReturn:
    sig_name = signal.strsignal(signum) if hasattr(signal, "strsignal") else str(signum)
    logger.warning(f"⚠️ Received signal {sig_name} ({signum}) – shutting down")
    raise KeyboardInterrupt

# -----------------------------------------------------------------------------
# Async entrypoint
# -----------------------------------------------------------------------------
async def main() -> int:
    """
    Main async entry point.

    Returns:
        0   Success
        2   Runtime failure
        3   Unhandled exception
        130 Interrupted (SIGINT/SIGTERM)
    """
    try:
        logger.info(f"📦 Starting Trading Bot v{__version__}")

        success = await run_once()
        if success:
            return 0

        logger.error("❌ Bot logic reported failure")
        return 2

    except KeyboardInterrupt:
        logger.warning("⚠️ Execution interrupted (SIGINT/SIGTERM)")
        return 130

    except asyncio.CancelledError:
        logger.warning("⚠️ Task cancelled (likely timeout)")
        return 130

    except Exception as exc:
        logger.exception(f"❌ UNHANDLED EXCEPTION: {exc}")
        return 3

# -----------------------------------------------------------------------------
# Process entry
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Explicit event loop policy (defensive, future-proof)
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    exit_code = 3
    try:
        exit_code = asyncio.run(main())
    except KeyboardInterrupt:
        exit_code = 130
    except Exception as e:
        fallback_logger.critical(f"FATAL: Unexpected error in wrapper: {e}")
        exit_code = 3
    finally:
        sys.exit(exit_code)
