#!/usr/bin/env python3
"""
wrapper.py - Ultra-Fast Entry Point

Optimizations:
1. Lazy imports (only load what's needed)
2. Minimal logging/printing
3. Direct asyncio execution
4. Pre-compile imports at module level
"""

import sys
import os
import asyncio

# OPTIMIZATION: Pre-set path before any imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# OPTIMIZATION: Skip debug output entirely unless explicitly enabled
DEBUG = os.getenv('DEBUG_MODE') == 'true'
if DEBUG:
    print(f"🚀 Run: {os.getenv('GITHUB_RUN_ID', 'local')}")

# OPTIMIZATION: Import core without logger (faster)
try:
    from macd_unified import run_once
except (ImportError, Exception) as e:
    print(f"❌ {type(e).__name__}: {e}")
    sys.exit(1)

def main() -> int:
    """Ultra-fast synchronous main (asyncio.run handles event loop)"""
    try:
        return 0 if asyncio.run(run_once()) else 2
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        if DEBUG:
            from macd_unified import logger
            logger.exception(f"❌ {exc}")
        else:
            print(f"❌ {exc}")
        return 2

if __name__ == "__main__":
    sys.exit(main())