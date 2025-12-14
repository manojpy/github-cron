#!/usr/bin/env python3
"""
wrapper.py - Entry Point for Unified MACD Trading Bot
Handles initialization, configuration validation, execution, and graceful error reporting.
"""
import asyncio
import os
import signal
import sys
from typing import NoReturn

# Add src to path (robust to __file__ issues in some environments)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, "src")
sys.path.insert(0, SRC_DIR)

# Early import validation
try:
    from src.macd_unified import run_once, logger, cfg, __version__
except ImportError as e:
    sys.stderr.write(f"❌ CRITICAL: Failed to import macd_unified.py: {e}\n")
    sys.stderr.write("💡 Check that src/macd_unified.py exists and the image was built correctly.\n")
    sys.exit(1)
except Exception as e:
    # This catches Pydantic validation errors from cfg loading
    sys.stderr.write(f"❌ CONFIGURATION ERROR: {e}\n")
    sys.stderr.write("💡 Verify config_macd.json syntax and required fields.\n")
    sys.exit(1)

def _handle_signal(signum: int, frame) -> NoReturn:
    """Unified signal handler for graceful shutdown."""
    sig_name = signal.strsignal(signum) if hasattr(signal, 'strsignal') else str(signum)
    logger.warning(f"⚠️  Received signal {sig_name} ({signum}) – shutting down")
    raise KeyboardInterrupt

async def main() -> int:
    """
    Main async entry point.
    
    Returns:
        0: Success
        2: Runtime error
        130: Interrupted (SIGINT/SIGTERM)
    """
    try:
        # Execute the main bot logic
        success = await run_once()
        
        if success:
            return 0
        else:
            logger.error("❌ Bot logic reported failure")
            return 2
            
    except KeyboardInterrupt:
        logger.warning("⚠️  Execution interrupted (SIGINT/SIGTERM)")
        return 130
        
    except asyncio.CancelledError:
        logger.warning("⚠️  Task cancelled (likely timeout)")
        return 130
        
    except Exception as exc:
        logger.exception(f"❌ UNHANDLED EXCEPTION: {exc}")
        return 2

if __name__ == "__main__":
    # Install signal handlers for container-friendly shutdown (SIGTERM from Docker/k8s)
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except Exception as e:
        # Absolute last-resort fallback
        sys.stderr.write(f"\nFATAL: Unexpected error in wrapper: {e}\n")
        sys.exit(2)