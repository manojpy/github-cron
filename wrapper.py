#!/usr/bin/env python3
# wrapper.py - GitHub Actions friendly wrapper for one-off run

import os
import sys
import logging
import importlib
import traceback
from typing import Optional

# Add src to path so `import macd_unified` resolves when code is in /app/src
sys.path.insert(0, "/app/src")
sys.path.insert(0, "/app")

CONFIG_FILE = os.environ.get("CONFIG_FILE", "config_macd.json")
TZ = os.environ.get("TZ", "Asia/Kolkata")
BOT_MODULE = os.environ.get("BOT_MODULE", "macd_unified")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger("wrapper")


def set_timezone(tz: str) -> None:
    os.environ["TZ"] = tz
    try:
        import time
        if hasattr(time, "tzset"):
            time.tzset()
    except Exception:
        logger.debug("tzset not available", exc_info=True)


def validate_config(path: str) -> None:
    if not os.path.exists(path):
        logger.warning("Config file %s not found in container.", path)
    else:
        logger.info("Using config file: %s", path)


def run_module(module_name: str, config_file: Optional[str] = None) -> int:
    try:
        logger.info("Attempting to import module '%s'", module_name)
        mod = importlib.import_module(module_name)

        # Prefer run_once()
        if hasattr(mod, "run_once"):
            logger.info("Calling run_once() on %s", module_name)
            try:
                mod.run_once(config_file) if callable(mod.run_once) else mod.run_once()
            except TypeError:
                mod.run_once()
            return 0

        # Fallback to main()
        if hasattr(mod, "main"):
            logger.info("Calling main() on %s", module_name)
            try:
                mod.main(config_file)
            except TypeError:
                mod.main()
            return 0

        # Last resort: execute module as script
        logger.info("No run_once/main found. Running as script.")
        import runpy
        runpy.run_module(module_name, run_name="__main__")
        return 0

    except Exception as e:
        logger.error("Error while running module '%s': %s", module_name, e)
        logger.debug(traceback.format_exc())
        return 2


def main():
    logger.info("wrapper starting up")
    set_timezone(TZ)
    validate_config(CONFIG_FILE)

    exit_code = run_module(BOT_MODULE, CONFIG_FILE)
    if exit_code == 0:
        logger.info("Bot run completed successfully.")
    else:
        logger.error("Bot run failed with exit code %d", exit_code)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
