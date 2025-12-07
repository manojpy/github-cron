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
    """Print a raw summary of the environment before logger init if possible."""
    print("=" * 70)
    print("🚀 Wrapper Starting")
    print(f"   GITHUB_RUN_ID: {os.getenv('GITHUB_RUN_ID', 'N/A')}")
    print(f"   PAIRS Configured: {len(os.getenv('PAIRS', '').split(',')) if os.getenv('PAIRS') else 'Using Config File'}")
    print("=" * 70)

try:
    log_env_summary()
    # Import Core - This triggers Pydantic Validation immediately
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
        logger.info(f"📋 Loaded Bot: {cfg.BOT_NAME} | Workers: {cfg.MAX_PARALLEL_FETCH}")
        
        # Execute the optimized run
        success = await run_once()
        
        if success:
            logger.info("✅ Wrapper: Execution successful")
            return 0
        else:
            logger.error("⚠️ Wrapper: Execution reported failure")
            return 2
            
    except KeyboardInterrupt:
        logger.warning("⚠️ Execution interrupted by user")
        return 130
        
    except Exception as exc:
        logger.exception(f"❌ FATAL ERROR: {exc}")
        return 2

if __name__ == "__main__":
    # uvloop policy is set inside macd_unified.py at import time
    exit_code = asyncio.run(main())
    sys.exit(exit_code)