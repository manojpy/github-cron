"""Root conftest for pytest.

Two jobs:
1. `import threshold_engine` (and every other bot module) needs src/ on
   sys.path. Running `pytest tests/` puts repo root on sys.path but not
   src/, so we insert it here once for the whole test session.
2. bot_config.load_config() calls sys.exit(1) at import time when the four
   required env vars are absent. setdefault means real env values win.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "123456:CI_DUMMY_TOKEN_FOR_TESTS")
os.environ.setdefault("TELEGRAM_CHAT_ID", "12345")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("DELTA_API_BASE", "https://api.example.com")