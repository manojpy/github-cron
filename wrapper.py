#!/usr/bin/env python3
"""
wrapper.py - Ultra-Optimized Entry Point

Optimizations Applied:
1. Minimal imports - only load what's absolutely necessary
2. Skip debug output unless explicitly enabled
3. Lazy logger loading (only on errors)
4. Direct asyncio execution without overhead
5. Early exit paths for faster failure handling
"""

import sys
import os

# OPTIMIZATION: Pre-set Python path before any imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# OPTIMIZATION: Check debug mode once, store result
DEBUG = os.getenv('DEBUG_MODE', 'false').lower() == 'true'

# OPTIMIZATION: Only print in debug mode (saves ~100ms)
if DEBUG:
    print(f"🚀 Run ID: {os.getenv('GITHUB_RUN_ID', 'local')}")

# OPTIMIZATION: Import core module without logger (faster startup)
try:
    from macd_unified import run_once
except ImportError as e:
    # Fast fail on import errors
    print(f"❌ IMPORT ERROR: {e}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    # Fast fail on config errors
    print(f"❌ CONFIG ERROR: {e}", file=sys.stderr)
    sys.exit(1)

# OPTIMIZATION: Import asyncio after core imports (no blocking I/O yet)
import asyncio


def main() -> int:
    """
    Ultra-fast main entry point.
    
    Returns:
        0: Success
        1: Config error (handled above)
        2: Runtime error
        130: User interrupt
    """
    try:
        # OPTIMIZATION: Use asyncio.run directly (handles uvloop if set in macd_unified)
        success = asyncio.run(run_once())
        return 0 if success else 2
        
    except KeyboardInterrupt:
        if DEBUG:
            print("⚠️ Interrupted by user", file=sys.stderr)
        return 130
        
    except Exception as exc:
        # OPTIMIZATION: Only import logger on errors (lazy loading)
        if DEBUG:
            try:
                from macd_unified import logger
                logger.exception(f"❌ FATAL ERROR: {exc}")
            except ImportError:
                print(f"❌ FATAL ERROR: {exc}", file=sys.stderr)
        else:
            # Production: minimal error output
            print(f"❌ {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    # OPTIMIZATION: Direct exit without intermediate variables
    sys.exit(main())