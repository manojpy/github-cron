#!/usr/bin/env python3
# python 3.12+
"""
Improved Fibonacci Pivot Bot
- Designed for cron-jobs.org / GitHub-run environment
- Uses asyncio, aiohttp, sqlite (WAL), pydantic config validation
- Robust: circuit breaker, retries, jitter, dead-man switch, graceful shutdown
"""

from __future__ import annotations

import os
import sys
import json
import time
import asyncio
import random
import logging
import sqlite3
import traceback
import signal
import html
import fcntl
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union

import aiohttp
import pandas as pd
import numpy as np
import pytz
import psutil
from aiohttp import ClientConnectorError, ClientResponseError, TCPConnector
from logging.handlers import RotatingFileHandler

# Pydantic V2 imports
try:
    # We explicitly import specific models used in the script
    from pydantic import BaseModel, field_validator, model_validator 
except ImportError:
    print("⚠️ pydantic not found. Install with: pip install 'pydantic>=2.5.0'")
    sys.exit(1)


# -------------------------
# CONFIGURATION MODEL
# -------------------------
class Config(BaseModel):
    """
    Configuration model for the bot.
    Includes Telegram, API, indicator settings, and operational parameters.
    """
    # Essential Telegram/API Credentials
    TELEGRAM_BOT_TOKEN: str
    TELEGRAM_CHAT_ID: str
    DELTA_API_BASE: str = "https://api.india.delta.exchange"
    
    # Operational Settings
    DEBUG_MODE: bool = False
    SEND_TEST_MESSAGE: bool = True
    RESET_STATE: bool = False
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "fibonacci_pivot_bot.log"
    STATE_DB_PATH: str = "fib_state.sqlite"
    
    # System and Performance
    PID_LOCK_PATH: str = "/tmp/fibonacci_pivot_bot.lock" # NEW: Configurable Lock Path
    DEADMAN_HOURS: int = 2 # NEW: Configurable Deadman Switch time
    BOT_NAME: str = "Fibonacci Pivot Bot" # NEW: Bot Name for logging/messages
    MAX_CONCURRENCY: int = 6 # Max parallel API requests/concurrent connections (Used for ClientSession limit and semaphore)
    HTTP_TIMEOUT: int = 15
    MAX_EXEC_TIME: int = 25 # Max execution time (in seconds) for a single run
    MEMORY_LIMIT_BYTES: int = 400_000_000  # ~400MB safe default for small hosts

    # Data/Cache Settings
    PAIRS: List[str] = ["BTCUSD", "ETHUSD", "SOLUSD", "AVAXUSD", "BCHUSD", "XRPUSD", "BNBUSD", "LTCUSD", "DOTUSD", "ADAUSD", "SUIUSD", "AAVEUSD"]
    SPECIAL_PAIRS: Dict[str, Dict[str, int]] = {"SOLUSD": {"limit_15m": 210, "min_required": 180, "limit_5m": 300, "min_required_5m": 200}}
    PRODUCTS_CACHE_TTL: int = 86400
    STATE_EXPIRY_DAYS: int = 30
    DUPLICATE_SUPPRESSION_SECONDS: int = 3600
    
    # API Retry Settings
    FETCH_RETRIES: int = 3
    FETCH_BACKOFF: float = 1.5
    JITTER_MIN: float = 0.05
    JITTER_MAX: float = 0.6
    TELEGRAM_RETRIES: int = 3
    TELEGRAM_BACKOFF_BASE: float = 2.0
    
    # Indicator Settings
    PPO_FAST: int = 7
    PPO_SLOW: int = 16
    PPO_SIGNAL: int = 5
    PPO_USE_SMA: bool = False
    X1: int = 22
    X2: int = 9
    X3: int = 15
    X4: int = 5
    PIVOT_LOOKBACK_PERIOD: int = 15
    EXTREME_CANDLE_PCT: float = 8.0
    USE_RMA200: bool = True

    # Main Loop (only used if script is run in loop mode, not standard cron)
    RUN_LOOP_INTERVAL: int = 900
    
    @field_validator("TELEGRAM_BOT_TOKEN")
    @classmethod
    def token_not_default(cls, v: str) -> str:
        if not v or v == "xxxx":
            raise ValueError("TELEGRAM_BOT_TOKEN must be set and not be 'xxxx'")
        return v

    @field_validator("TELEGRAM_CHAT_ID")
    @classmethod
    def chat_id_not_default(cls, v: str) -> str:
        if not v or v == "xxxx":
            raise ValueError("TELEGRAM_CHAT_ID must be set and not be 'xxxx'")
        return v

# -------------------------
# DEFAULT CONFIG
# -------------------------
DEFAULT_CONFIG = Config(
    TELEGRAM_BOT_TOKEN="xxxx",
    TELEGRAM_CHAT_ID="xxxx",
    PID_LOCK_PATH="/tmp/fibonacci_pivot_bot.lock", # Default added here
    DEADMAN_HOURS=2, # Default added here
    BOT_NAME="Fibonacci Pivot Bot", # Default added here
).model_dump()

# -------------------------
# EXIT CODES
# -------------------------
EXIT_SUCCESS = 0
EXIT_LOCK_CONFLICT = 2
EXIT_TIMEOUT = 3
EXIT_CONFIG_ERROR = 4
EXIT_API_FAILURE = 5

# -------------------------
# CONFIG LOADER
# -------------------------
def str_to_bool(value: str) -> bool: 
    return str(value).strip().lower() in ("true", "1", "yes", "y", "t")

def load_config() -> Config:
    """Loads configuration from default, config file, and environment variables."""
    base = DEFAULT_CONFIG.copy()
    config_file = os.getenv("CONFIG_FILE", "config_fib.json")

    if Path(config_file).exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                user_cfg = json.load(f)
            
            # The config file in the snippet had MAX_PARALLEL_FETCH, 
            # we'll map it to MAX_CONCURRENCY for consistency.
            if "MAX_PARALLEL_FETCH" in user_cfg and "MAX_CONCURRENCY" not in user_cfg:
                user_cfg["MAX_CONCURRENCY"] = user_cfg["MAX_PARALLEL_FETCH"]

            base.update(user_cfg)
            print(f"✅ Loaded configuration from {config_file}")
        except Exception as e:
            print(f"⚠️ Warning: unable to parse {config_file}: {e}")
            print("Using defaults and environment overrides only.")
    else:
        print(f"⚠️ Warning: config file {config_file} not found, using defaults.")

    def override(key: str, default: Any = None, cast: Optional[callable] = None):
        """Overrides a config key with environment variable if present."""
        val = os.getenv(key)
        if val is not None:
            base[key] = cast(val) if cast else val
        else:
            base[key] = base.get(key, default)

    # Apply environment overrides (ensure they match the Config class fields)
    override("TELEGRAM_BOT_TOKEN")
    override("TELEGRAM_CHAT_ID")
    override("DEBUG_MODE", False, str_to_bool)
    override("SEND_TEST_MESSAGE", True, str_to_bool)
    override("RESET_STATE", False, str_to_bool)
    override("STATE_DB_PATH")
    override("LOG_FILE")
    override("DELTA_API_BASE")
    override("MAX_CONCURRENCY", 6, int)
    override("HTTP_TIMEOUT", 15, int)
    override("MAX_EXEC_TIME", 25, int)
    override("PID_LOCK_PATH")
    override("DEADMAN_HOURS", 2, int)
    override("BOT_NAME")

    try:
        # Use model_validate_json for V2 to parse the combined dictionary
        return Config(**base)
    except Exception as e:
        print(f"❌ Critical Configuration Error: {e}")
        raise SystemExit(EXIT_CONFIG_ERROR)

cfg = load_config()

# -------------------------
# METRICS
# -------------------------
METRICS: Dict[str, Any] = {
    "pairs_checked": 0,
    "alerts_sent": 0,
    "api_errors": 0,
    "logic_errors": 0,
    "network_errors": 0,
    "execution_time": 0.0,
    "start_time": 0.0
}

def reset_metrics():
    """Resets the run-specific metrics."""
    METRICS.update({
        "pairs_checked": 0,
        "alerts_sent": 0,
        "api_errors": 0,
        "logic_errors": 0,
        "network_errors": 0,
        "execution_time": 0.0,
        "start_time": time.time()
    })

# -------------------------
# LOGGER (Standardised Format)
# -------------------------
logger = logging.getLogger("fibonacci_pivot_bot")
log_level = getattr(logging, cfg.LOG_LEVEL if isinstance(cfg.LOG_LEVEL, str) else "INFO", logging.INFO)
logger.setLevel(logging.DEBUG if cfg.DEBUG_MODE else log_level)

class JSONFormatter(logging.Formatter):
    """Enhanced JSON formatter for standardized logging."""
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "time": self.formatTime(record), 
            "level": record.levelname, 
            "name": record.name,
            "module": record.module,
            "funcName": record.funcName,
            "lineno": record.lineno,
            "msg": record.getMessage()
        }
        return json.dumps(log_obj, ensure_ascii=False)

# Use standard format for console in debug, JSON for all else
if cfg.DEBUG_MODE:
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
else:
    formatter = JSONFormatter()

# Console Handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# File Handler
try:
    file_handler = RotatingFileHandler(cfg.LOG_FILE, maxBytes=2_000_000, backupCount=5, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
except Exception as e:
    logger.warning(f"Could not create rotating log file: {e}")

# -------------------------
# PROCESS LOCKING
# -------------------------
class ProcessLock:
    """File-based lock with stale detection using fcntl."""
    def __init__(self, lock_path: str, timeout: int = 1200):
        self.lock_path = Path(lock_path)
        self.timeout = timeout
        self.fd = None

    def acquire(self) -> bool:
        """Acquire exclusive lock. Returns True if successful."""
        try:
            self.lock_path.parent.mkdir(parents=True, exist_ok=True)
            self.fd = open(self.lock_path, 'w')
            fcntl.flock(self.fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.fd.write(str(os.getpid()))
            self.fd.flush()
            return True
        except (IOError, OSError):
            # Try to handle stale lock (lock file exists but PID is gone)
            try:
                if self.lock_path.exists():
                    with open(self.lock_path, 'r') as f:
                        raw = f.read().strip()
                    old_pid = int(raw) if raw.isdigit() else None
                    # Check if the process still exists in /proc (Linux-specific, but robust)
                    if old_pid and not os.path.exists(f"/proc/{old_pid}"):
                        logger.warning(f"Removing stale lock from PID {old_pid}")
                        try:
                            self.lock_path.unlink(missing_ok=True)
                        except Exception:
                            pass
                        # Recursive call to attempt re-acquisition
                        return self.acquire()
            except Exception:
                pass # Ignore errors during stale lock check
            
            if self.fd:
                try:
                    self.fd.close()
                except Exception:
                    pass
            return False

    def release(self):
        """Release the lock and remove the file."""
        try:
            if self.fd:
                try:
                    fcntl.flock(self.fd.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass
                try:
                    self.fd.close()
                except Exception:
                    pass
            try:
                self.lock_path.unlink(missing_ok=True)
            except Exception:
                pass
        except Exception as e:
            logger.debug(f"Error releasing lock: {e}")

# -------------------------
# CIRCUIT BREAKER
# -------------------------
class CircuitBreaker:
    """Prevent cascading failures when API is down."""
    def __init__(self, failure_threshold: int = 5, timeout: int = 300):
        self.failures = 0
        self.last_failure = 0.0
        self.threshold = failure_threshold
        self.timeout = timeout
        self.state = "CLOSED"

    async def call(self, func: callable):
        """Wrap an async function with circuit breaker logic."""
        if self.state == "OPEN":
            if time.time() - self.last_failure > self.timeout:
                self.state = "HALF_OPEN"
                logger.info("Circuit breaker: attempting HALF_OPEN")
            else:
                raise Exception(f"Circuit breaker is OPEN (wait {int(self.timeout - (time.time() - self.last_failure))}s)")

        try:
            result = await func()
            if self.state == "HALF_OPEN":
                logger.info("Circuit breaker: CLOSED (recovery successful)")
            self.failures = 0
            self.state = "CLOSED"
            return result
        except ClientConnectorError:
            self.failures += 1
            self.last_failure = time.time()
            METRICS["network_errors"] += 1
            if self.failures >= self.threshold:
                self.state = "OPEN"
                logger.error(f"Circuit breaker OPEN after {self.failures} network failures")
            raise
        except Exception:
            self.failures += 1
            self.last_failure = time.time()
            METRICS["api_errors"] += 1
            if self.failures >= self.threshold:
                self.state = "OPEN"
                logger.error(f"Circuit breaker OPEN after {self.failures} failures")
            raise

# -------------------------
# UTILITIES
# -------------------------
def now_ts() -> int:
    """Returns current UTC timestamp in seconds."""
    return int(time.time())

def human_ts() -> str:
    """Returns human-readable time in IST for alerts."""
    tz = pytz.timezone("Asia/Kolkata")
    return datetime.now(tz).strftime("%d-%m-%Y @ %H:%M IST")

def sanitize_for_telegram(text: Union[str, float, int]) -> str:
    """Prevent Telegram HTML injection."""
    return html.escape(str(text))

# -------------------------
# SQLITE STATE DB (WAL mode)
# -------------------------
class StateDB:
    """SQLite database for persistent state and metadata."""
    def __init__(self, db_path: str):
        self.db_path = db_path
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
        # WAL mode and check_same_thread=False (though not strictly needed for single-process async)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False, isolation_level=None)
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        except Exception:
            pass # Ignore if PRAGMA fails
        self._ensure_tables()
        try:
            # Set restrictive permissions
            os.chmod(self.db_path, 0o600)
        except Exception:
            pass

    def _ensure_tables(self):
        """Creates necessary tables if they don't exist."""
        cur = self._conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS states (
                pair TEXT PRIMARY KEY,
                state TEXT,
                ts INTEGER
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        self._conn.commit()

    def load_all(self) -> Dict[str, Dict[str, Any]]:
        """Loads all states."""
        cur = self._conn.cursor()
        cur.execute("SELECT pair, state, ts FROM states")
        rows = cur.fetchall()
        # State can be complex (JSON/String), here we store it as a string
        return {r[0]: {"state": r[1], "ts": int(r[2] or 0)} for r in rows}

    def get(self, pair: str) -> Optional[Dict[str, Any]]:
        """Gets state for a specific pair."""
        cur = self._conn.cursor()
        cur.execute("SELECT state, ts FROM states WHERE pair = ?", (pair,))
        r = cur.fetchone()
        if not r:
            return None
        return {"state": r[0], "ts": int(r[1] or 0)}

    def set(self, pair: str, state: Optional[str], ts: Optional[int] = None):
        """Sets or clears state for a specific pair."""
        ts = int(ts or now_ts())
        cur = self._conn.cursor()
        if state is None:
            cur.execute("DELETE FROM states WHERE pair = ?", (pair,))
        else:
            cur.execute(
                "INSERT INTO states(pair, state, ts) VALUES (?, ?, ?) "
                "ON CONFLICT(pair) DO UPDATE SET state=excluded.state, ts=excluded.ts",
                (pair, state, ts)
            )
        self._conn.commit()

    def get_metadata(self, key: str) -> Optional[str]:
        """Gets a piece of metadata."""
        cur = self._conn.cursor()
        cur.execute("SELECT value FROM metadata WHERE key = ?", (key,))
        r = cur.fetchone()
        return r[0] if r else None

    def set_metadata(self, key: str, value: str):
        """Sets a piece of metadata."""
        cur = self._conn.cursor()
        cur.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)", (key, value))
        self._conn.commit()

    def close(self):
        """Closes the database connection."""
        try:
            self._conn.close()
        except Exception:
            pass

    def __del__(self):
        """Ensures the connection is closed on object destruction."""
        try:
            self.close()
        except Exception:
            pass

# -------------------------
# PRUNE (smart daily)
# -------------------------
def prune_old_state_records(db_path: str, expiry_days: int = 30, logger_local: logging.Logger = None):
    """
    Cleans up old state records from the DB, runs only once per day.
    Uses metadata to track the last prune date.
    """
    try:
        if expiry_days <= 0:
            if logger_local:
                logger_local.debug("Pruning disabled (expiry_days <= 0)")
            return

        if not os.path.exists(db_path):
            if logger_local:
                logger_local.info(f"State DB not found at {db_path}; skipping prune.")
            return

        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        try:
            cur.execute("BEGIN EXCLUSIVE")
        except Exception:
            pass # Ignore if transaction fails

        cur.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)")
        cur.execute("SELECT value FROM metadata WHERE key='last_prune'")
        row = cur.fetchone()

        today = datetime.now(timezone.utc).date()

        if row:
            try:
                last_prune_date = datetime.fromisoformat(row[0]).date()
                if last_prune_date >= today:
                    if logger_local:
                        logger_local.debug("Prune already run today; skipping.")
                    conn.close()
                    return
            except Exception:
                pass # Ignore metadata parse errors

        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='states'")
        if not cur.fetchone():
            if logger_local:
                logger_local.debug("No states table yet; skipping prune.")
            conn.close()
            return

        # Calculate cutoff timestamp
        cutoff = int(time.time()) - (expiry_days * 86400)
        cur.execute("DELETE FROM states WHERE ts < ?", (cutoff,))
        deleted = cur.rowcount

        cur.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES ('last_prune', ?)", (datetime.utcnow().isoformat(),))
        conn.commit()

        # Vacuum the database after deleting records
        if deleted > 0:
            try:
                cur.execute("VACUUM;")
            except Exception as e:
                if logger_local:
                    logger_local.warning(f"VACUUM failed: {e}")

        conn.close()
        if logger_local:
            logger_local.info(f"Pruned {deleted} states older than {expiry_days} days from {db_path}.")
    except Exception as e:
        if logger_local:
            logger_local.warning(f"Prune failed: {e}")
            logger_local.debug(traceback.format_exc())

# -------------------------
# ASYNC HTTP HELPERS
# -------------------------
async def fetch_json_with_retries(
    session: aiohttp.ClientSession,
    url: str,
    params: dict = None,
    retries: int = 3,
    backoff: float = 1.5,
    timeout: int = 15,
    circuit_breaker: Optional[CircuitBreaker] = None
):
    """
    Fetches JSON data from a URL with retry logic, exponential backoff, and jitter.
    Integrates with the CircuitBreaker.
    """
    async def _fetch_attempt():
        """Internal function for the circuit breaker."""
        for attempt in range(1, retries + 1):
            try:
                async with session.get(url, params=params, timeout=timeout) as resp:
                    text = await resp.text()
                    
                    if resp.status >= 400:
                        logger.debug(f"HTTP {resp.status} {url} {params} - {text[:200]}")
                        # Raise for the circuit breaker to catch and increase failure count
                        resp.raise_for_status() 

                    try:
                        return await resp.json()
                    except json.JSONDecodeError:
                        logger.debug(f"Non-JSON response from {url}: {text[:200]}")
                        return {} # Return empty dict for non-JSON response

            except (asyncio.TimeoutError, ClientConnectorError) as e:
                logger.debug(f"Fetch attempt {attempt} error: {e} for {url}")
                METRICS["network_errors"] += 1
                if attempt == retries:
                    logger.warning(f"Failed to fetch {url} after {retries} attempts.")
                    raise # Re-raise for the circuit breaker's outer try block
                
                # Jittered exponential backoff sleep
                await asyncio.sleep(backoff * (attempt ** 1.2) + random.uniform(0, 0.2))
            
            except ClientResponseError as cre:
                METRICS["api_errors"] += 1
                logger.debug(f"ClientResponseError fetching {url}: {cre}")
                return None # API error (e.g., 404, 429) - stop retrying, return None
            
            except Exception as e:
                logger.exception(f"Unexpected fetch error for {url}: {e}")
                METRICS["logic_errors"] += 1
                return None # General unexpected error - stop retrying, return None
        return None # Should be unreachable if final failure raises

    if circuit_breaker:
        try:
            return await circuit_breaker.call(_fetch_attempt)
        except Exception:
            return None # Circuit breaker is open or call failed
    else:
        # If no circuit breaker, wrap in try/except to return None on failure
        try:
            return await _fetch_attempt()
        except Exception:
            return None

# -------------------------
# DATA FETCHER
# -------------------------
class DataFetcher:
    """Manages data fetching from the API, coordinating concurrency and caching."""
    def __init__(self, base_url: str, max_parallel: int = 6, circuit_breaker: Optional[CircuitBreaker] = None):
        self.base_url = base_url.rstrip("/")
        # Semaphore is mainly for self-imposed concurrency limit on the fetch_json calls
        self.semaphore = asyncio.Semaphore(max_parallel)
        self.cache: Dict[str, Tuple[float, Any]] = {}
        self.circuit_breaker = circuit_breaker

    async def fetch_products(self, session: aiohttp.ClientSession):
        """Fetches the list of products from the API."""
        url = f"{self.base_url}/v2/products"
        async with self.semaphore:
            return await fetch_json_with_retries(
                session, url,
                retries=cfg.FETCH_RETRIES,
                backoff=cfg.FETCH_BACKOFF,
                timeout=cfg.HTTP_TIMEOUT,
                circuit_breaker=self.circuit_breaker
            )

    async def fetch_candles(self, session: aiohttp.ClientSession, symbol: str, resolution: str, limit: int):
        """Fetches candlestick data for a given symbol."""
        key = f"candles:{symbol}:{resolution}:{limit}"
        
        # Simple in-memory cache for a very short period (e.g., in case of parallel calls in one run)
        if key in self.cache:
            age, data = self.cache[key]
            if time.time() - age < 5: 
                return data

        # Add jitter to API calls to prevent thundering herd
        await asyncio.sleep(random.uniform(cfg.JITTER_MIN, cfg.JITTER_MAX))
        
        url = f"{self.base_url}/v2/chart/history"
        # Calculate 'from' time based on limit
        res_minutes = int(resolution) if resolution.isdigit() else 1440
        params = {
            "resolution": resolution,
            "symbol": symbol,
            "from": int(time.time()) - (limit * res_minutes * 60),
            "to": int(time.time())
        }
        
        async with self.semaphore:
            data = await fetch_json_with_retries(
                session, url, params=params,
                retries=cfg.FETCH_RETRIES,
                backoff=cfg.FETCH_BACKOFF,
                timeout=cfg.HTTP_TIMEOUT,
                circuit_breaker=self.circuit_breaker
            )
        
        self.cache[key] = (time.time(), data)
        return data

# -------------------------
# PRODUCT CACHE
# -------------------------
def load_products_cache() -> Optional[Dict[str, dict]]:
    """Loads product data from disk cache if not expired."""
    cache_path = Path("products_cache.json")
    if cache_path.exists():
        try:
            age = time.time() - cache_path.stat().st_mtime
            if age < cfg.PRODUCTS_CACHE_TTL:
                with open(cache_path, "r", encoding="utf-8") as f:
                    logger.info(f"Using cached products ({age:.0f}s old)")
                    return json.load(f)
        except Exception as e:
            logger.debug(f"Cache load failed: {e}")
    return None

def save_products_cache(products_map: Dict[str, dict]):
    """Saves product data to disk cache."""
    try:
        with open("products_cache.json", "w", encoding="utf-8") as f:
            json.dump(products_map, f, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"Failed to save products cache: {e}")

def build_products_map(api_products: dict) -> Dict[str, dict]:
    """
    Parses the API response to create a map of configured PAIRS to product details.
    Normalizes symbols (e.g., USDT to USD).
    """
    products_map: Dict[str, dict] = {}
    if not api_products or not isinstance(api_products, dict):
        return products_map
        
    for p in api_products.get("result", []):
        try:
            symbol = p.get("symbol", "")
            # Normalization to match the user's PAIRS list (e.g., BTC_USDT -> BTCUSD)
            symbol_norm = symbol.replace("_USDT", "USD").replace("USDT", "USD")
            
            if p.get("contract_type") == "perpetual_futures":
                for pair_name in cfg.PAIRS:
                    # Check for exact match or stripped match
                    if symbol_norm == pair_name or symbol_norm.replace("_", "") == pair_name:
                        products_map[pair_name] = {
                            'id': p.get('id'),
                            'symbol': p.get('symbol'),
                            'contract_type': p.get('contract_type')
                        }
        except Exception:
            continue
            
    return products_map

# -------------------------
# CANDLE PARSING
# -------------------------
def parse_candle_response(res: dict) -> Optional[pd.DataFrame]:
    """Converts the API candle response (dict of arrays) into a pandas DataFrame."""
    if not res or not isinstance(res, dict):
        return None
    if not res.get("success", True) and "result" not in res:
        return None
        
    resr = res.get("result", {}) or {}
    arrays = [resr.get('t', []), resr.get('o', []), resr.get('h', []), resr.get('l', []), resr.get('c', []), resr.get('v', [])]
    
    if any(len(a) == 0 for a in arrays):
        return None
        
    min_len = min(map(len, arrays))
    
    # Create DataFrame with the minimum length
    df = pd.DataFrame({
        'timestamp': resr.get('t', [])[:min_len],
        'open': resr.get('o', [])[:min_len],
        'high': resr.get('h', [])[:min_len],
        'low': resr.get('l', [])[:min_len],
        'close': resr.get('c', [])[:min_len],
        'volume': resr.get('v', [])[:min_len]
    })
    
    # Clean up: sort by timestamp and remove duplicates
    df = df.sort_values('timestamp').drop_duplicates(subset='timestamp').reset_index(drop=True)
    
    # Final check for validity
    if df.empty or float(df['close'].astype(float).iloc[-1]) <= 0:
        return None
        
    return df

# -------------------------
# INDICATORS (No change required, existing implementation is fine)
# -------------------------
# ... (calculate_ema, calculate_sma, calculate_rma, calculate_ppo, smoothrng, rngfilt, calculate_cirrus_cloud, kalman_filter, calculate_smooth_rsi, calculate_worm_momentum_hist, calculate_fibonacci_pivots, get_crossover_line) ...

def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def calculate_sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=max(1, period // 3)).mean()

def calculate_rma(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(alpha=1 / period, adjust=False).mean()

def calculate_ppo(df: pd.DataFrame, fast: int, slow: int, signal: int, use_sma: bool = False):
    close = df['close'].astype(float)
    if use_sma:
        fast_ma = calculate_sma(close, fast)
        slow_ma = calculate_sma(close, slow)
    else:
        fast_ma = calculate_ema(close, fast)
        slow_ma = calculate_ema(close, slow)
        
    slow_ma = slow_ma.replace(0, np.nan).bfill().ffill() # Handle division by zero
    
    ppo = ((fast_ma - slow_ma) / slow_ma) * 100
    ppo = ppo.replace([np.inf, -np.inf], np.nan).bfill().ffill()
    
    ppo_signal = calculate_sma(ppo, signal) if use_sma else calculate_ema(ppo, signal)
    ppo_signal = ppo_signal.replace([np.inf, -np.inf], np.nan).bfill().ffill()
    
    return ppo, ppo_signal

def smoothrng(x: pd.Series, t: int, m: int) -> pd.Series:
    wper = t * 2 - 1
    avrng = calculate_ema(np.abs(x.diff().fillna(0)), t)
    smoothrng_val = calculate_ema(avrng, max(1, wper)) * m
    return smoothrng_val.clip(lower=1e-8).bfill().ffill()

def rngfilt(x: pd.Series, r: pd.Series) -> pd.Series:
    if len(x) == 0:
        return pd.Series(dtype=float)
        
    result = [x.iloc[0]]
    for i in range(1, len(x)):
        prev = result[-1]
        curr_x = x.iloc[i]
        curr_r = max(float(r.iloc[i]), 1e-8)
        
        if curr_x > prev:
            f = prev if (curr_x - curr_r) < prev else (curr_x - curr_r)
        else:
            f = prev if (curr_x + curr_r) > prev else (curr_x + curr_r)
        
        result.append(f)
        
    return pd.Series(result, index=x.index)

def calculate_cirrus_cloud(df: pd.DataFrame):
    close = df['close'].astype(float)
    smrngx1x = smoothrng(close, cfg.X1, cfg.X2)
    smrngx1x2 = smoothrng(close, cfg.X3, cfg.X4)
    filtx1 = rngfilt(close, smrngx1x)
    filtx12 = rngfilt(close, smrngx1x2)
    upw = filtx1 < filtx12
    dnw = filtx1 > filtx12
    return upw, dnw, filtx1, filtx12

def kalman_filter(src: pd.Series, length: int, R=0.01, Q=0.1) -> pd.Series:
    result = []
    estimate = np.nan
    error_est = 1.0
    error_meas = R * max(1, length)
    Q_div_length = Q / max(1, length)
    
    for i in range(len(src)):
        current = src.iloc[i]
        
        if np.isnan(estimate):
            if i > 0:
                estimate = src.iloc[i-1]
            else:
                result.append(np.nan)
                continue
        
        prediction = estimate
        kalman_gain = error_est / (error_est + error_meas)
        estimate = prediction + kalman_gain * (current - prediction)
        error_est = (1 - kalman_gain) * error_est + Q_div_length
        result.append(estimate)
        
    return pd.Series(result, index=src.index)

def calculate_smooth_rsi(df: pd.DataFrame, rsi_len: int, kalman_len: int) -> pd.Series:
    close = df['close'].astype(float)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    # Calculate RMA (Running Moving Average) for gain and loss
    avg_gain = calculate_rma(gain, rsi_len)
    avg_loss = calculate_rma(loss, rsi_len).replace(0, np.nan).bfill().ffill().clip(lower=1e-8)
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.replace([np.inf, -np.inf], np.nan).fillna(50)
    
    # Apply Kalman Filter for smoothing
    smooth_rsi = kalman_filter(rsi, kalman_len)
    
    return smooth_rsi.replace([np.inf, -np.inf], np.nan).bfill().ffill()

def calculate_vwap(df: pd.DataFrame, daily_df: pd.DataFrame) -> Optional[pd.Series]:
    """
    Calculates the Volume Weighted Average Price (VWAP) for the current 15m day.
    Uses daily open/high/low for the pivot point calculation used in VWAP start.
    """
    if daily_df is None or len(daily_df) < 2:
        return None
        
    # Get today's Daily Open (use last but one candle in daily data)
    today_open = float(daily_df['open'].iloc[-2])
    
    # Find the first 15m candle whose open is >= today's daily open
    # This aligns the 15m data with the start of the trading day (usually UTC 00:00)
    start_index = df[df['timestamp'] >= daily_df['timestamp'].iloc[-2]].index.min()
    
    if pd.isna(start_index):
        # If no candles match (e.g., ran too early in the day), assume the first candle
        start_index = 0 
    
    df_day = df.iloc[start_index:].copy()
    
    if df_day.empty:
        return None
        
    # Calculate typical price (TP) for each 15m candle
    df_day['TP'] = (df_day['high'].astype(float) + df_day['low'].astype(float) + df_day['close'].astype(float)) / 3
    
    # Calculate Cumulative Volume and Cumulative (TP * Volume)
    df_day['TPV'] = df_day['TP'] * df_day['volume'].astype(float)
    df_day['CumVol'] = df_day['volume'].astype(float).cumsum()
    df_day['CumTPV'] = df_day['TPV'].cumsum()
    
    # Calculate VWAP
    df_day['VWAP'] = (df_day['CumTPV'] / df_day['CumVol']).replace([np.inf, -np.inf], np.nan).ffill().bfill()
    
    # Merge back into the full 15m dataframe, filling NaNs
    vwap_series = df_day['VWAP'].reindex(df.index).ffill().bfill()
    
    return vwap_series

def calculate_worm_momentum_hist(df: pd.DataFrame) -> pd.Series:
    """Calculates the Worm Momentum Histogram (a custom momentum indicator)."""
    close = df['close'].astype(float)
    length = 40
    n = 20
    period = 10
    
    # Calculate worm (Welles Wilder Moving Average or RMA)
    worm = calculate_rma(close, length) 
    
    # Calculate SMA
    ma = close.rolling(window=period, min_periods=max(5, period // 3)).mean().bfill().ffill()
    
    # Calculate raw momentum
    denom = worm.replace(0, np.nan).bfill().ffill()
    raw_momentum = (worm - ma) / denom
    raw_momentum = raw_momentum.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Calculate min/max of raw momentum over 'period'
    min_med = raw_momentum.rolling(window=period, min_periods=max(5, period // 3)).min().bfill().ffill()
    max_med = raw_momentum.rolling(window=period, min_periods=max(5, period // 3)).max().bfill().ffill()
    
    # Normalize raw momentum to 0-1 range
    rng = (max_med - min_med).replace(0, np.nan)
    temp = pd.Series(0.0, index=df.index)
    valid = rng.notna() & (rng != 0)
    temp.loc[valid] = (raw_momentum.loc[valid] - min_med.loc[valid]) / rng.loc[valid]
    temp = temp.clip(0, 1).fillna(0.5) # clip and default to 0.5 if range is zero
    
    # Apply a smoothing/lagging process
    value = pd.Series(0.0, index=df.index)
    for i in range(1, n):
        prev = value.iloc[i - 1]
        val = (temp.iloc[i] - 0.5 + 0.5 * prev)
        val = max(min(val, 0.9999), -0.9999)
        value.iloc[i] = val 

    # Transform to get momentum value
    temp2 = (1 + value) / (1 - value)
    temp2 = temp2.replace([np.inf, -np.inf], np.nan).clip(lower=1e-6).fillna(1e-6)
    momentum = 0.25 * np.log(temp2)
    momentum = pd.Series(momentum, index=df.index).replace([np.inf, -np.inf], 0).fillna(0)
    
    # Calculate final histogram
    hist = pd.Series(0.0, index=df.index)
    for i in range(1, n):
        hist.iloc[i] = momentum.iloc[i] + 0.5 * hist.iloc[i - 1]
        
    return hist.replace([np.inf, -np.inf], 0).fillna(0)


def calculate_fibonacci_pivots(df_daily: pd.DataFrame) -> Optional[Dict[str, float]]:
    """Calculates Fibonacci Pivot Points from the previous day's data."""
    if df_daily is None or len(df_daily) < 2:
        return None
        
    # Use the second to last candle (yesterday's close)
    prev_day = df_daily.iloc[-2] 
    
    # H, L, C of the previous day
    high = float(prev_day['high'])
    low = float(prev_day['low'])
    close = float(prev_day['close'])
    
    # Base Pivot Point
    pivot = (high + low + close) / 3
    
    if close > pivot:
        # Bullish close
        R3 = pivot + (high - low)
        R2 = pivot + 0.618 * (high - low)
        R1 = pivot + 0.382 * (high - low)
        S1 = pivot - 0.382 * (high - low)
        S2 = pivot - 0.618 * (high - low)
        S3 = pivot - (high - low)
    elif close < pivot:
        # Bearish close
        R3 = pivot + (high - low)
        R2 = pivot + 0.618 * (high - low)
        R1 = pivot + 0.382 * (high - low)
        S1 = pivot - 0.382 * (high - low)
        S2 = pivot - 0.618 * (high - low)
        S3 = pivot - (high - low)
    else:
        # Neutral close
        R3 = pivot + (high - low)
        R2 = pivot + 0.618 * (high - low)
        R1 = pivot + 0.382 * (high - low)
        S1 = pivot - 0.382 * (high - low)
        S2 = pivot - 0.618 * (high - low)
        S3 = pivot - (high - low)
        
    return {
        'P': pivot,
        'R1': R1,
        'R2': R2,
        'R3': R3,
        'S1': S1,
        'S2': S2,
        'S3': S3,
        'high': high, # Keep H/L for the difference calculation reference
        'low': low
    }

def get_crossover_line(pivots: Dict[str, float], prev_price: float, curr_price: float, direction: str) -> Optional[Tuple[str, float]]:
    """Checks for a crossover of any pivot level between two consecutive candles."""
    if not pivots:
        return None
        
    levels = ["R3", "R2", "R1", "P", "S1", "S2", "S3"]
    
    for level_name in levels:
        line = pivots.get(level_name)
        if line is None:
            continue

        # Check for crossover
        if direction == "long":
            # Cross above the line (prev <= line < curr)
            if prev_price <= line and curr_price > line:
                return level_name, line
        elif direction == "short":
            # Cross below the line (prev >= line > curr)
            if prev_price >= line and curr_price < line:
                return level_name, line
                
    return None

# -------------------------
# STATE MANAGEMENT
# -------------------------
def should_suppress_duplicate(last_state: Optional[Dict[str, Any]], current_signal: str, suppress_secs: int) -> bool:
    """Checks if the same signal was recently sent and should be suppressed."""
    if not last_state:
        return False

    state_ts = int(last_state.get("ts", 0))
    state_signal = last_state.get("state")

    if current_signal == state_signal:
        if now_ts() - state_ts < suppress_secs:
            logger.debug(f"Suppressed duplicate signal: {current_signal} within {suppress_secs}s")
            return True
            
    return False

def cloud_state_from(upw: pd.Series, dnw: pd.Series, idx: int) -> str:
    """Determines the Cirrus Cloud state at a given index."""
    if upw.iloc[idx] and not dnw.iloc[idx]:
        return "long"
    elif dnw.iloc[idx] and not upw.iloc[idx]:
        return "short"
    else:
        return "neutral"

def telegram_log_state(pair_name: str, current_signal: str, message: str):
    """Logs the alert being sent for metrics and debugging."""
    logger.info(f"🚨 ALERT for {pair_name} - {current_signal}: {message.splitlines()[0]}")
    METRICS["alerts_sent"] += 1

# -------------------------
# TELEGRAM NOTIFICATION
# -------------------------
async def send_telegram_async(session: aiohttp.ClientSession, token: str, chat_id: str, text: str) -> bool:
    """Sends a message to Telegram using the provided shared session."""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    
    data = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": "true"
    }
    
    try:
        # Use the shared session with a default timeout
        async with session.post(url, data=data, timeout=10) as resp:
            try:
                # Need content_type=None because Telegram sometimes returns an unexpected content-type
                js = await resp.json(content_type=None)
            except Exception:
                js = {"ok": False, "status": resp.status, "text": await resp.text()}
                
            ok = js.get("ok", False)
            if ok:
                await asyncio.sleep(0.2) # Small delay to avoid rate limiting
                return True
            else:
                logger.warning(f"Telegram API error: {js}")
                return False
                
    except Exception as e:
        logger.exception(f"Telegram send failed: {e}")
        return False

async def send_telegram_with_retries(session: aiohttp.ClientSession, token: str, chat_id: str, text: str) -> bool:
    """Retries sending a Telegram message with exponential backoff."""
    last_exc = None
    
    for attempt in range(1, max(1, cfg.TELEGRAM_RETRIES) + 1):
        try:
            ok = await send_telegram_async(session, token, chat_id, text)
            if ok:
                return True
            last_exc = Exception("Telegram returned non-ok")
        except Exception as e:
            last_exc = e
        
        # Exponential backoff
        sleep_for = cfg.TELEGRAM_BACKOFF_BASE ** (attempt - 1)
        await asyncio.sleep(sleep_for + random.uniform(0, 0.3))

    logger.error(f"Telegram send failed after retries: {last_exc}")
    return False

# -------------------------
# DEAD MAN'S SWITCH
# -------------------------
async def check_dead_mans_switch(state_db: StateDB, session: aiohttp.ClientSession):
    """
    Checks if the bot has successfully completed a run within the configured DEADMAN_HOURS.
    Sends a critical alert if the bot has failed to report in.
    """
    try:
        last_success = state_db.get_metadata("last_success_run")
        
        if last_success:
            try:
                last_success_int = int(last_success)
            except Exception:
                last_success_int = now_ts()
                
            hours_since = (now_ts() - last_success_int) / 3600.0
            
            # Check if the time since last success is greater than the configured limit
            if hours_since > cfg.DEADMAN_HOURS:
                
                # Check when the last alert was sent to prevent spam
                dead_alert_ts = state_db.get_metadata("dead_alert_ts")
                if dead_alert_ts:
                    try:
                        dead_alert_int = int(dead_alert_ts)
                    except Exception:
                        dead_alert_int = 0
                else:
                    dead_alert_int = 0
                
                # Only alert once every 4 hours after the initial failure
                if now_ts() - dead_alert_int > 4 * 3600: 
                    
                    msg = (
                        f"🚨 DEAD MAN'S SWITCH: **{cfg.BOT_NAME}** hasn't succeeded in {hours_since:.1f} hours!\n"
                        f"Last success: {datetime.fromtimestamp(last_success_int).strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
                        f"Expected check-in interval: Every {cfg.RUN_LOOP_INTERVAL / 60:.0f} minutes."
                    )
                    
                    if await send_telegram_with_retries(session, cfg.TELEGRAM_BOT_TOKEN, cfg.TELEGRAM_CHAT_ID, msg):
                        logger.critical(f"Dead man's switch triggered. Sent alert.")
                        state_db.set_metadata("dead_alert_ts", str(now_ts()))
                    else:
                        logger.error("Failed to send Dead man's switch alert.")
                        
    except Exception as e:
        logger.exception(f"Dead man's switch check failed: {e}")

# -------------------------
# EVALUATION CORE
# -------------------------
async def evaluate_pair_async(
    session: aiohttp.ClientSession, 
    fetcher: DataFetcher, 
    products_map: Dict[str, dict], 
    pair_name: str, 
    last_state: Optional[Dict[str, Any]]
) -> Tuple[Optional[str], Optional[Dict[str, str]]]:
    """
    Core logic: fetches data, calculates indicators, finds signals, and sends alerts.
    
    Returns: (new_state, alert_info)
    """
    METRICS["pairs_checked"] += 1
    
    prod = products_map.get(pair_name)
    if prod is None:
        logger.warning(f"Skipping {pair_name}: Product not found in API response.")
        return None, None
        
    # Determine the required candle limits based on default or SPECIAL_PAIRS config
    limit_15m = cfg.SPECIAL_PAIRS.get(pair_name, {}).get("limit_15m", 120)
    min_required_15m = cfg.SPECIAL_PAIRS.get(pair_name, {}).get("min_required", 100)
    limit_5m = cfg.SPECIAL_PAIRS.get(pair_name, {}).get("limit_5m", 300)
    min_required_5m = cfg.SPECIAL_PAIRS.get(pair_name, {}).get("min_required_5m", 200)

    # --- 1. Data Fetching ---
    try:
        # Fetch 15m and 5m (if RMA200 is used) concurrently
        fetch_15m = fetcher.fetch_candles(session, prod['symbol'], "15", limit_15m)
        fetch_5m = fetcher.fetch_candles(session, prod['symbol'], "5", limit_5m) if cfg.USE_RMA200 else asyncio.sleep(0, result=None)
        
        results = await asyncio.gather(fetch_15m, fetch_5m, return_exceptions=True)
        
        res15 = results[0] if len(results) > 0 and not isinstance(results[0], Exception) else None
        res5 = results[1] if len(results) > 1 and not isinstance(results[1], Exception) else None
        
        if isinstance(results[0], Exception):
            logger.error(f"Exception fetching 15m data for {pair_name}: {results[0]}")
            METRICS["network_errors"] += 1
            return last_state.get("state") if last_state else None, None

    except Exception as e:
        logger.error(f"Error during concurrent fetch for {pair_name}: {e}")
        METRICS["network_errors"] += 1
        return last_state.get("state") if last_state else None, None

    df_15m = parse_candle_response(res15)
    df_5m = parse_candle_response(res5) if cfg.USE_RMA200 else None

    # Check for sufficient data
    if df_15m is None or len(df_15m) < min_required_15m:
        logger.debug(f"{pair_name}: Insufficient 15m data (got {len(df_15m) if df_15m is not None else 0}, need {min_required_15m})")
        return last_state.get("state") if last_state else None, None
    if cfg.USE_RMA200 and (df_5m is None or len(df_5m) < min_required_5m):
        logger.debug(f"{pair_name}: Insufficient 5m data (got {len(df_5m) if df_5m is not None else 0}, need {min_required_5m})")
        return last_state.get("state") if last_state else None, None
        
    # Fetch daily data for pivots
    daily_res = await fetcher.fetch_candles(session, prod['symbol'], "D", cfg.PIVOT_LOOKBACK_PERIOD + 1)
    df_daily = parse_candle_response(daily_res)

    pivots = calculate_fibonacci_pivots(df_daily)
    if pivots is None or df_daily is None or len(df_daily) < 2:
        logger.debug(f"{pair_name}: Insufficient daily data for pivots.")
        pivots_available = False
    else:
        pivots_available = True

    # --- 2. Calculate Indicators ---
    # Index for current (most recent) closed candle
    last_i_15m = -2
    last_i_5m = -2

    ppo, ppo_signal = calculate_ppo(df_15m, cfg.PPO_FAST, cfg.PPO_SLOW, cfg.PPO_SIGNAL, cfg.PPO_USE_SMA)
    upw, dnw, filtx1, filtx12 = calculate_cirrus_cloud(df_15m)
    magical_hist = calculate_worm_momentum_hist(df_15m)

    vwap_15m: Optional[pd.Series] = calculate_vwap(df_15m, df_daily)
    rma200_5m: Optional[pd.Series] = calculate_rma(df_5m['close'].astype(float), 200) if cfg.USE_RMA200 and df_5m is not None else None
    
    # --- 3. Get Current Candle Values (from last closed candle, index -2) ---
    try:
        # 15m current values
        curr_15m = df_15m.iloc[last_i_15m]
        prev_15m = df_15m.iloc[last_i_15m - 1]
        close_c = float(curr_15m['close'])
        open_c = float(curr_15m['open'])
        high_c = float(curr_15m['high'])
        low_c = float(curr_15m['low'])
        prev_close_c = float(prev_15m['close'])
        
        is_green = close_c > open_c
        is_red = close_c < open_c
        
        ppo_curr = float(ppo.iloc[last_i_15m])
        magical_curr = float(magical_hist.iloc[last_i_15m])
        cloud_state = cloud_state_from(upw, dnw, last_i_15m)
        vwap_curr = float(vwap_15m.iloc[last_i_15m]) if vwap_15m is not None and len(vwap_15m) >= abs(last_i_15m) else None
        
        # 5m RMA200 value
        rma200_5m_curr = float(rma200_5m.iloc[last_i_5m]) if rma200_5m is not None and len(rma200_5m) >= abs(last_i_5m) else None
        rma_200_available = rma200_5m_curr is not None and not np.isnan(rma200_5m_curr)

    except Exception as e:
        logger.error(f"Error extracting indicator values for {pair_name}: {e}")
        METRICS["logic_errors"] += 1
        return last_state.get("state") if last_state else None, None
        
    vwap_log_str = f"{vwap_curr:,.2f}" if vwap_curr is not None and not np.isnan(vwap_curr) else "nan"
    logger.debug(
        f"{pair_name}: close={close_c:.2f}, open={open_c:.2f}, "
        f"PPO={ppo_curr:.2f}, "
        f"MMH={magical_curr:.4f}, Cloud={cloud_state}, "
        f"VWAP={vwap_log_str}, RMA200={rma200_5m_curr}"
    )
    
    # --- 4. Validation/Condition Checks ---
    # Extreme candle check (filter out large, anomalous candles)
    candle_pct_change = abs(close_c - open_c) / open_c * 100
    if candle_pct_change > cfg.EXTREME_CANDLE_PCT:
        logger.debug(f"{pair_name}: Skipped due to extreme candle ({candle_pct_change:.2f}% > {cfg.EXTREME_CANDLE_PCT}%)")
        return last_state.get("state") if last_state else None, None
        
    # Wick checks (filter out candles with disproportionately long wicks)
    total_range = high_c - low_c
    if total_range <= 1e-6: # Avoid division by zero
        upper_wick_ok = True
        lower_wick_ok = True
    else:
        upper_wick = high_c - max(open_c, close_c)
        lower_wick = min(open_c, close_c) - low_c
        
        # Upper wick must not be more than 60% of total range for a long signal (green candle)
        upper_wick_ok = not is_green or (upper_wick / total_range) < 0.6
        # Lower wick must not be more than 60% of total range for a short signal (red candle)
        lower_wick_ok = not is_red or (lower_wick / total_range) < 0.6

    # RMA200 check: 15m close must be above 5m RMA200 for long, below for short
    rma_long_ok = not cfg.USE_RMA200 or (rma_200_available and close_c > rma200_5m_curr)
    rma_short_ok = not cfg.USE_RMA200 or (rma_200_available and close_c < rma200_5m_curr)
    
    # --- 5. Signal Logic ---
    current_signal = None
    message = None
    suppress_secs = cfg.DUPLICATE_SUPPRESSION_SECONDS
    up_sig = "🟩▲" 
    down_sig = "🟥🔻"

    # VWAP Signal (Volume Buy/Sell)
    vbuy = (
        vwap_curr is not None and 
        is_green and 
        cloud_state == "long" and 
        magical_curr > 0 and 
        upper_wick_ok and
        rma_long_ok and
        prev_close_c <= vwap_curr and close_c > vwap_curr # VWAP crossover
    )
    
    vsell = (
        vwap_curr is not None and 
        is_red and 
        cloud_state == "short" and 
        magical_curr < 0 and 
        lower_wick_ok and
        rma_short_ok and
        prev_close_c >= vwap_curr and close_c < vwap_curr # VWAP crossover
    )

    # Fibonacci Pivot Crossover Signal
    base_long_ok = (
        is_green and
        cloud_state == "long" and 
        magical_curr > 0 and
        rma_long_ok and 
        upper_wick_ok
    )
    long_crossover = get_crossover_line(pivots, prev_close_c, close_c, "long")
    fib_long = pivots_available and base_long_ok and (long_crossover is not None)
    long_crossover_name, long_crossover_line = long_crossover if long_crossover else (None, None)
    
    base_short_ok = (
        is_red and
        cloud_state == "short" and 
        magical_curr < 0 and
        rma_short_ok and
        lower_wick_ok
    )
    short_crossover = get_crossover_line(pivots, prev_close_c, close_c, "short")
    fib_short = pivots_available and base_short_ok and (short_crossover is not None)
    short_crossover_name, short_crossover_line = short_crossover if short_crossover else (None, None)

    # --- 6. Alert Generation ---
    if vbuy:
        current_signal = "vbuy"
        if not should_suppress_duplicate(last_state, current_signal, suppress_secs):
            vwap_str = f"{vwap_curr:,.2f}"
            ppo_str = f"{ppo_curr:.2f}"
            price_str = f"{close_c:,.2f}"
            message = (
                f"{up_sig} **{sanitize_for_telegram(pair_name)}** - VBuy (VWAP Cross)\n"
                f"PPO 15m: `{ppo_str}` | VWAP: `${vwap_str}`\n"
                f"Price: `${price_str}`\n"
                f"{human_ts()}"
            )
            
    elif vsell:
        current_signal = "vsell"
        if not should_suppress_duplicate(last_state, current_signal, suppress_secs):
            vwap_str = f"{vwap_curr:,.2f}"
            ppo_str = f"{ppo_curr:.2f}"
            price_str = f"{close_c:,.2f}"
            message = (
                f"{down_sig} **{sanitize_for_telegram(pair_name)}** - VSell (VWAP Cross)\n"
                f"PPO 15m: `{ppo_str}` | VWAP: `${vwap_str}`\n"
                f"Price: `${price_str}`\n"
                f"{human_ts()}"
            )
            
    elif fib_long:
        current_signal = f"fib_long_{long_crossover_name}"
        if not should_suppress_duplicate(last_state, current_signal, suppress_secs):
            line_str = f"{long_crossover_line:,.2f}"
            ppo_str = f"{ppo_curr:.2f}"
            price_str = f"{close_c:,.2f}"
            message = (
                f"{up_sig} **{sanitize_for_telegram(pair_name)}** - Fib Long (Cross {long_crossover_name})\n"
                f"PPO 15m: `{ppo_str}` | Level: `${line_str}`\n"
                f"Price: `${price_str}`\n"
                f"{human_ts()}"
            )
            
    elif fib_short:
        current_signal = f"fib_short_{short_crossover_name}"
        if not should_suppress_duplicate(last_state, current_signal, suppress_secs):
            line_str = f"{short_crossover_line:,.2f}"
            ppo_str = f"{ppo_curr:.2f}"
            price_str = f"{close_c:,.2f}"
            message = (
                f"{down_sig} **{sanitize_for_telegram(pair_name)}** - Fib Short (Cross {short_crossover_name})\n"
                f"PPO 15m: `{ppo_str}` | Level: `${line_str}`\n"
                f"Price: `${price_str}`\n"
                f"{human_ts()}"
            )

    # --- 7. Final Action ---
    alert_info = None
    if message:
        telegram_log_state(pair_name, current_signal, message)
        if await send_telegram_with_retries(session, cfg.TELEGRAM_BOT_TOKEN, cfg.TELEGRAM_CHAT_ID, message):
            alert_info = {"signal": current_signal, "message": message}
            return current_signal, alert_info
        else:
            # If sending fails, retain the last state to avoid re-alerting if retries succeed on next run
            return last_state.get("state") if last_state else None, None
            
    # No signal, keep previous state unless it was for a long-expired condition
    return last_state.get("state") if last_state else None, None


# -------------------------
# MAIN EXECUTION
# -------------------------
stop_requested = False

def sig_handler(signum, frame):
    """Graceful shutdown handler."""
    global stop_requested
    if not stop_requested:
        stop_requested = True
        logger.info(f"Signal {signum} received. Initiating graceful shutdown.")

signal.signal(signal.SIGINT, sig_handler)
signal.signal(signal.SIGTERM, sig_handler)

async def run_once(state_db: StateDB, send_test: bool = True):
    """
    Performs a single, complete run of the bot logic.
    Refactored to use a single shared aiohttp.ClientSession.
    """
    # 1. Setup Session/Fetcher/Circuit Breaker
    connector = TCPConnector(limit=cfg.MAX_CONCURRENCY, ssl=False)
    # The entire run uses this single session
    async with aiohttp.ClientSession(connector=connector, timeout=aiohttp.ClientTimeout(total=cfg.HTTP_TIMEOUT)) as session:
        
        # Check Dead Man's Switch immediately
        await check_dead_mans_switch(state_db, session)

        # 2. Initial Checks
        if cfg.RESET_STATE:
            logger.warning("RESET_STATE=True: Clearing all state data.")
            state_db.set_metadata("last_prune", str(datetime.fromtimestamp(0).isoformat())) # Force prune on next run
            for pair in cfg.PAIRS:
                state_db.set(pair, None)
            state_db.set_metadata("last_success_run", "0")
            
        if cfg.SEND_TEST_MESSAGE and send_test:
            test_msg = f"🚀 **{cfg.BOT_NAME}** is starting. Time: {human_ts()}. Execution limit: {cfg.MAX_EXEC_TIME}s."
            await send_telegram_with_retries(session, cfg.TELEGRAM_BOT_TOKEN, cfg.TELEGRAM_CHAT_ID, test_msg)
            
        products_map = load_products_cache()
        circuit = CircuitBreaker()
        fetcher_local = DataFetcher(cfg.DELTA_API_BASE, max_parallel=cfg.MAX_CONCURRENCY, circuit_breaker=circuit)

        # 3. Product Discovery (if cache miss)
        if products_map is None:
            prod_resp = await fetcher_local.fetch_products(session)
            if not prod_resp:
                logger.error("❌ Failed to fetch products from API")
                # Do not save products_map if fetch failed
                raise SystemExit(EXIT_API_FAILURE)
                
            products_map = build_products_map(prod_resp)
            save_products_cache(products_map)
            
        last_alerts = state_db.load_all()

        # 4. Run Pair Evaluations Concurrently
        tasks = [
            evaluate_pair_async(
                session, 
                fetcher_local, 
                products_map, 
                pair_name, 
                last_alerts.get(pair_name)
            )
            for pair_name in cfg.PAIRS
        ]
        
        # Use a timeout to ensure the cron job finishes within its window
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=cfg.MAX_EXEC_TIME
            )
        except asyncio.TimeoutError:
            logger.error(f"❌ Execution exceeded MAX_EXEC_TIME of {cfg.MAX_EXEC_TIME} seconds.")
            raise SystemExit(EXIT_TIMEOUT)
        
        # 5. Process Results
        alerts_sent = 0
        for pair_name, result in zip(cfg.PAIRS, results):
            if stop_requested:
                break
                
            if isinstance(result, Exception):
                logger.error(f"Error in task for {pair_name}: {result}")
                METRICS["logic_errors"] += 1
                continue
                
            new_state, alert_info = result
            
            if new_state:
                state_db.set(pair_name, new_state)
            # Only clear state if evaluate_pair_async explicitly signals it (e.g., expiry/no longer valid)
            elif new_state is None and last_alerts.get(pair_name):
                 # Clear state if the new state is None and an old state exists
                state_db.set(pair_name, None)
                
            if alert_info:
                alerts_sent += 1

        # 6. Final Cleanup / Pruning / Metrics
        prune_old_state_records(cfg.STATE_DB_PATH, cfg.STATE_EXPIRY_DAYS, logger)
        
        # Update successful run timestamp for dead man's switch
        state_db.set_metadata("last_success_run", str(now_ts()))
        
        METRICS["execution_time"] = time.time() - METRICS["start_time"]
        logger.info(f"Run completed. Pairs checked: {METRICS['pairs_checked']}, Alerts sent: {alerts_sent}. Time: {METRICS['execution_time']:.2f}s")


def main():
    """Main function to handle locking and execution mode."""
    global stop_requested
    
    # 1. Initialize DB and Lock
    prune_old_state_records(cfg.STATE_DB_PATH, cfg.STATE_EXPIRY_DAYS, logger)
    state_db = StateDB(cfg.STATE_DB_PATH)

    # Use configurable PID_LOCK_PATH
    lock = ProcessLock(lock_path=cfg.PID_LOCK_PATH, timeout=cfg.MAX_EXEC_TIME + 60) 
    
    try:
        if not lock.acquire():
            logger.warning("Exiting: Lock file in use by another process. (PID Lock Path: {cfg.PID_LOCK_PATH})")
            sys.exit(EXIT_LOCK_CONFLICT)

        reset_metrics()
        
        # Memory Check (Good safeguard for small environments)
        if psutil.virtual_memory().available < cfg.MEMORY_LIMIT_BYTES:
            logger.error(f"❌ Aborting: Insufficient memory. Available: {psutil.virtual_memory().available / 1024**2:.0f}MB, Required: {cfg.MEMORY_LIMIT_BYTES / 1024**2:.0f}MB")
            raise SystemExit(EXIT_API_FAILURE)

        # Determine if running in an infinite loop (e.g., in a Docker container) 
        # or as a single run (e.g., cron-jobs.org or GitHub Actions)
        LOOP_MODE = str_to_bool(os.getenv("LOOP_MODE", "false"))

        if LOOP_MODE:
            # --- Loop Run Logic (for long-running host) ---
            interval = cfg.RUN_LOOP_INTERVAL
            logger.info(f"🔁 Loop mode started. Running every {interval} seconds.")
            
            loop_start = METRICS.get('start_time')
            while not stop_requested:
                try:
                    # Only send the test message on the first iteration of the loop
                    asyncio.run(run_once(state_db, send_test=loop_start == METRICS.get('start_time')))
                except SystemExit as e:
                    logger.error(f"Run exited with code {e.code}, continuing loop.")
                    if e.code == EXIT_TIMEOUT:
                        logger.warning("Continuing loop after timeout exit.")
                    elif e.code != EXIT_API_FAILURE:
                        break # Stop for non-recoverable errors
                except Exception:
                    logger.exception("Unhandled in run_once")

                elapsed = time.time() - loop_start
                to_sleep = max(0, interval - elapsed)
                
                # Check for stop request while sleeping
                if to_sleep > 0:
                    time.sleep(to_sleep) # Simple sleep for a long-running process

            sys.exit(EXIT_SUCCESS)
            
        else:
            # --- Single Run Logic (for GitHub Actions / Cron) ---
            logger.info("🚀 Single run mode (exiting immediately after completion).")
            # `run_once` will use the default `send_test=True`
            asyncio.run(run_once(state_db)) 
            sys.exit(EXIT_SUCCESS) # Ensure explicit exit

    except SystemExit as e:
        logger.error(f"Exited with code {e.code}")
        sys.exit(e.code if isinstance(e.code, int) else 1)
    except Exception:
        logger.exception("Unhandled exception in main execution block")
        sys.exit(EXIT_API_FAILURE)
    finally:
        lock.release()

if __name__ == "__main__":
    main()
