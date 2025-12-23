import json
import logging
import os
import sys
from pathlib import Path

from macd_unified import run_bot

CONFIG_PATH = Path(__file__).parent.parent / "config_macd.json"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

def setup_logging(config):
    log_file = config.get("LOG_FILE", "macd_bot.log")
    log_level = getattr(logging, config.get("LOG_LEVEL", "INFO").upper())
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )

def main():
    config = load_config()
    setup_logging(config)
    logging.info("Starting Unified Alert Bot...")
    run_bot(config)

if __name__ == "__main__":
    main()
