#!/usr/bin/env python3
"""
wrapper.py - Optimized Entry Point

This wrapper delegates validation and execution to the highly-optimized
src/macd_unified.py core.
"""

import sys
import os
import asyncio


# Ensure src is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def log_env_summary():
    """Print a minimal summary before logger init."""
    run_id = os.getenv('GITHUB_RUN_ID', 'local')
    if run_id != 'local':
        print(f"🚀 Run ID: {run_id}")

try:
    log_env_summary()
    # Import Core - This triggers Pydantic Validation immediately.
    # The warning filter is now active before this import.
    from src.macd_unified import run_once, logger, cfg
except ImportError as e:
    print(f"❌ CRITICAL IMPORT ERROR: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ CONFIGURATION ERROR: {e}")
    sys.exit(1)

async def main() -> int:
    """
    Main entry point.
    Returns: 0 (Success), 1 (Config Error), 2 (Runtime Error)
    """
    try:
        
        # Execute the optimized run
        success = await run_once()
        
        if success:
            return 0
        else:
            logger.error("❌ Execution failed")
            return 2
            
    except KeyboardInterrupt:
        logger.warning("⚠️ Execution interrupted by user")
        return 130
        
    except Exception as exc:
        logger.exception(f"❌ FATAL ERROR: {exc}")
        return 2

if __name__ == "__main__":
    # uvloop policy is set inside macd_unified.py at import time
    # Removed suppress_pycparser_warning() call from here as it is now called globally
    exit_code = asyncio.run(main())
    sys.exit(exit_code)