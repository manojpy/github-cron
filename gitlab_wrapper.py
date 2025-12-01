#!/usr/bin/env python3
"""
gitlab_wrapper.py - GitLab CI/CD entry point for trading bot

This wrapper:
1. Sets up environment variables from GitLab CI
2. Validates configuration
3. Runs the bot once
4. Handles errors and exit codes properly
"""

import sys
import os
import pathlib
import asyncio
import logging
from typing import Optional

# Add src/ to Python path
sys.path.insert(0, str(pathlib.Path(__file__).parent / 'src'))

from macd_unified import run_once, setup_logging, load_config

# Configure logging to both file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("last_run.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ],
)

logger = logging.getLogger("gitlab_wrapper")


def validate_environment() -> bool:
    """
    Validate that all required environment variables are set.
    Returns True if valid, False otherwise.
    """
    required_vars = {
        "TELEGRAM_BOT_TOKEN": "Telegram bot token",
        "TELEGRAM_CHAT_ID": "Telegram chat ID",
        "REDIS_URL": "Redis connection URL",
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
        logger.error("Please set these in GitLab CI/CD Settings → Variables")
        logger.error("=" * 70)
        return False
    
    logger.info("✅ Environment validation passed")
    return True


def log_environment_info() -> None:
    """Log relevant environment information for debugging."""
    logger.info("=" * 70)
    logger.info("🚀 GitLab CI/CD Trading Bot Wrapper Starting")
    logger.info("=" * 70)
    
    # Log GitLab CI environment info
    ci_info = {
        "CI_COMMIT_SHA": os.getenv("CI_COMMIT_SHORT_SHA", "N/A"),
        "CI_PIPELINE_ID": os.getenv("CI_PIPELINE_ID", "N/A"),
        "CI_JOB_ID": os.getenv("CI_JOB_ID", "N/A"),
        "CI_JOB_STARTED_AT": os.getenv("CI_JOB_STARTED_AT", "N/A"),
        "CONFIG_FILE": os.getenv("CONFIG_FILE", "config_macd.json"),
    }
    
    for key, value in ci_info.items():
        logger.info(f"{key}: {value}")
    
    # Log masked environment variables (don't reveal secrets)
    logger.info(f"TELEGRAM_BOT_TOKEN: {'✓ SET' if os.getenv('TELEGRAM_BOT_TOKEN') else '✗ MISSING'}")
    logger.info(f"TELEGRAM_CHAT_ID: {'✓ SET' if os.getenv('TELEGRAM_CHAT_ID') else '✗ MISSING'}")
    logger.info(f"REDIS_URL: {'✓ SET' if os.getenv('REDIS_URL') else '✗ MISSING'}")
    logger.info(f"DELTA_API_BASE: {os.getenv('DELTA_API_BASE', 'https://api.delta.exchange')}")
    logger.info("=" * 70)


async def main() -> int:
    """
    Main entry point for GitLab CI wrapper.
    
    Returns:
        0 if successful
        1 if configuration error
        2 if bot execution failed
    """
    
    # Log environment information
    log_environment_info()
    
    # Validate environment variables
    if not validate_environment():
        logger.error("❌ Environment validation failed - exiting")
        return 1
    
    # Set environment variables for the bot (override config file placeholders)
    # These are already set by GitLab CI, but we ensure they're available
    os.environ.setdefault("TELEGRAM_BOT_TOKEN", os.getenv("TELEGRAM_BOT_TOKEN", ""))
    os.environ.setdefault("TELEGRAM_CHAT_ID", os.getenv("TELEGRAM_CHAT_ID", ""))
    os.environ.setdefault("REDIS_URL", os.getenv("REDIS_URL", ""))
    os.environ.setdefault("DELTA_API_BASE", os.getenv("DELTA_API_BASE", "https://api.delta.exchange"))
    
    try:
        # Load and validate configuration
        logger.info("📋 Loading configuration...")
        cfg = load_config()
        logger.info(f"✅ Configuration loaded: {cfg.BOT_NAME}")
        logger.info(f"📊 Monitoring {len(cfg.PAIRS)} pairs: {', '.join(cfg.PAIRS)}")
        
        # Run the bot once
        logger.info("🤖 Starting bot execution...")
        success = await run_once()
        
        if success:
            logger.info("=" * 70)
            logger.info("✅ Bot execution completed successfully")
            logger.info("=" * 70)
            return 0
        else:
            logger.error("=" * 70)
            logger.error("⚠️  Bot execution completed with warnings/errors")
            logger.error("=" * 70)
            return 2
            
    except KeyboardInterrupt:
        logger.warning("⚠️  Execution interrupted by user")
        return 130
        
    except Exception as exc:
        logger.exception(f"❌ FATAL ERROR during bot execution: {exc}")
        logger.error("=" * 70)
        logger.error("💥 Bot crashed - check logs above for details")
        logger.error("=" * 70)
        return 2
    
    finally:
        logger.info("👋 GitLab wrapper shutting down")


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)