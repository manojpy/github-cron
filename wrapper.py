#!/usr/bin/env python3
"""
wrapper.py - Optimized Entry Point for Trading Bot

This wrapper provides a clean interface to the core macd_unified.py module,
handling initialization, error reporting, and graceful shutdown.
"""

import sys
import os
import asyncio

# Ensure src directory is in Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import and validate configuration
try:
    # Import core module - this triggers Pydantic validation immediately
    from src.macd_unified import run_once, logger, cfg, __version__
    
except ImportError as e:
    print(f"❌ CRITICAL IMPORT ERROR: {e}", file=sys.stderr)
    print(f"💡 Ensure src/macd_unified.py exists and dependencies are installed", file=sys.stderr)
    sys.exit(1)
    
except Exception as e:
    print(f"❌ CONFIGURATION ERROR: {e}", file=sys.stderr)
    print(f"💡 Check config_macd.json and environment variables", file=sys.stderr)
    sys.exit(1)

async def main() -> int:
    """
    Main execution entry point.
    
    Returns:
        0: Success
        1: Configuration error
        2: Runtime error
        130: Interrupted by user (SIGINT)
    """
    try:
        # Execute the main bot logic
        success = await run_once()
        
        if success:
            return 0
        else:
            logger.error("❌ Execution failed")
            return 2
            
    except KeyboardInterrupt:
        logger.warning("⚠️  Execution interrupted by user (SIGINT)")
        return 130
        
    except asyncio.CancelledError:
        logger.warning("⚠️  Execution cancelled (timeout or signal)")
        return 130
        
    except Exception as exc:
        logger.exception(f"❌ FATAL ERROR: {exc}")
        return 2

if __name__ == "__main__":
    # uvloop is automatically configured in macd_unified.py at import time
    
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
        
    except Exception as e:
        print(f"\n❌ FATAL: {e}\n", file=sys.stderr)
        sys.exit(2)