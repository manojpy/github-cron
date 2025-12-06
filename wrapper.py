#!/usr/bin/env python3
"""
wrapper.py - GitHub Actions / Cron-jobs.org entry point for trading bot

This wrapper:
1. Sets up environment variables from GitHub Actions
2. Validates configuration
3. Runs the bot once
4. Handles errors and exit codes properly
"""

import sys
import os
import asyncio
import logging

try:
    from src.macd_unified import run_once, logger as bot_logger, cfg
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    from macd_unified import run_once, logger as bot_logger, cfg

logger = bot_logger


def validate_environment() -> bool:
    """
    Validate that all required environment variables are set.
    Returns True if valid, False otherwise.
    """
    required_vars = {
        "TELEGRAM_BOT_TOKEN": "Telegram bot token",
        "TELEGRAM_CHAT_ID": "Telegram chat ID",
        "REDIS_URL": "Redis connection URL",
        "DELTA_API_BASE": "Delta Exchange API base URL",
    }
    
    missing = []
    for var, description in required_vars.items():
        value = os.getenv(var)
        if not value or value == "__SET_IN_GITLAB_CI__":
            missing.append(f"  - {var} ({description})")
            logger.error(f"❌ Missing or invalid: {var}")
    
    if missing:
        logger.error("=" * 70)
        logger.error("❌ CONFIGURATION ERROR: Required environment variables not set")
        logger.error("=" * 70)
        logger.error("Missing variables:")
        for item in missing:
            logger.error(item)
        logger.error("")
        logger.error("Please set these in GitHub Secrets or Cron-jobs.org headers")
        logger.error("=" * 70)
        return False
    
    logger.info("✅ Environment validation passed")
    return True


def log_environment_info() -> None:
    """Log relevant environment information for debugging."""
    logger.info("=" * 70)
    logger.info("🚀 GitHub Actions Trading Bot Wrapper Starting")
    logger.info("=" * 70)
    
    ci_info = {
        "GITHUB_SHA": os.getenv("GITHUB_SHA", "N/A")[:8],
        "GITHUB_RUN_ID": os.getenv("GITHUB_RUN_ID", "N/A"),
        "GITHUB_RUN_NUMBER": os.getenv("GITHUB_RUN_NUMBER", "N/A"),
        "GITHUB_WORKFLOW": os.getenv("GITHUB_WORKFLOW", "N/A"),
        "CONFIG_FILE": os.getenv("CONFIG_FILE", "config_macd.json"),
        "PYTHONPATH": os.getenv("PYTHONPATH", "N/A"),
    }
    
    for key, value in ci_info.items():
        logger.info(f"{key}: {value}")
    
    logger.info(f"TELEGRAM_BOT_TOKEN: {'✓ SET' if os.getenv('TELEGRAM_BOT_TOKEN') else '✗ MISSING'}")
    logger.info(f"TELEGRAM_CHAT_ID: {'✓ SET' if os.getenv('TELEGRAM_CHAT_ID') else '✗ MISSING'}")
    logger.info(f"REDIS_URL: {'✓ SET' if os.getenv('REDIS_URL') else '✗ MISSING'}")
    logger.info(f"DELTA_API_BASE: {os.getenv('DELTA_API_BASE', 'https://api.delta.exchange')}")
    logger.info("=" * 70)


async def main() -> int:
    """
    Main entry point for GitHub Actions wrapper.
    
    Returns:
        0 if successful
        1 if configuration error
        2 if bot execution failed
        130 if interrupted
    """
    
    log_environment_info()
    
    if not validate_environment():
        logger.error("❌ Environment validation failed - exiting")
        return 1
    
    os.environ.setdefault("DELTA_API_BASE", "https://api.delta.exchange")
    
    try:
        logger.info("📋 Configuration loaded from macd_unified")
        logger.info(f"✅ Bot Name: {cfg.BOT_NAME}")
        logger.info(f"📊 Monitoring {len(cfg.PAIRS)} pairs: {', '.join(cfg.PAIRS[:5])}{'...' if len(cfg.PAIRS) > 5 else ''}")
        
        logger.info("🤖 Starting bot execution...")
        success = await run_once()
        
        if success:
            logger.info("=" * 70)
            logger.info("✅ Bot execution completed successfully")
            logger.info("=" * 70)
            return 0
        else:
            logger.error("=" * 70)
            logger.error("⚠️ Bot execution completed with warnings/errors")
            logger.error("=" * 70)
            return 2
            
    except KeyboardInterrupt:
        logger.warning("⚠️ Execution interrupted by user")
        return 130
        
    except Exception as exc:
        logger.exception(f"❌ FATAL ERROR during bot execution: {exc}")
        logger.error("=" * 70)
        logger.error("💥 Bot crashed - check logs above for details")
        logger.error("=" * 70)
        return 2
    
    finally:
        logger.info("👋 Wrapper shutting down")


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)