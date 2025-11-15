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
    # Removed model_validator as it's not used, just BaseModel
    from pydantic import BaseModel 
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
    # NOTE: The default must be the placeholder 'xxxx' so it can be overwritten by the config file.
    TELEGRAM_BOT_TOKEN: str = "xxxx" 
    TELEGRAM_CHAT_ID: str = "xxxx"
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
    
# -------------------------
# DEFAULT CONFIG
# -------------------------
# Initialize default config without custom validators
DEFAULT_CONFIG = Config().model_dump() 

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
            # We explicitly check for 'xxxx' here for security sensitive fields
            if key in ["TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"] and val == "xxxx":
                # Do nothing, keep the file/default value (which we'll check later)
                pass 
            else:
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
        # Final Pydantic validation
        final_config = Config(**base)
        
        # --- CRITICAL SECURITY CHECK (Moved here from model validator) ---
        if final_config.TELEGRAM_BOT_TOKEN == "xxxx":
            raise ValueError("TELEGRAM_BOT_TOKEN must be set and not be 'xxxx' in config_fib.json or env.")
        if final_config.TELEGRAM_CHAT_ID == "xxxx":
            raise ValueError("TELEGRAM_CHAT_ID must be set and not be 'xxxx' in config_fib.json or env.")
        # --- END CRITICAL SECURITY CHECK ---
        
        return final_config
    except Exception as e:
        # Capture config error details for better debugging
        print(f"❌ Critical Configuration Error: {e.__class__.__name__}: {e}")
        # If the error is a ValueError from the security check, we raise a controlled exit
        if isinstance(e, ValueError):
            raise SystemExit(EXIT_CONFIG_ERROR)
        # For other Pydantic errors, re-raise the SystemExit
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
    """RMA 50 and 5m RMA 200 checks (below close) for BUY."""
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
        # Standardized logging format includes essential context
        log_obj = {
            "time": self.formatTime(record), 
            "level": record.levelname, 
            "name": record.name,
            "module": record.module,
            "funcName": record.funcName,
            "lineno": record.lineno,
            "msg": record.getMessage()
        }
        # Add traceback if an exception occurred
        if record.exc_info:
            log_obj["exc_info"] = self.formatException(record.exc_info)
            
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
                    # Use psutil.pid_exists for cross-platform robustness
                    if old_pid and not psutil.pid_exists(old_pid):
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
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='states'")
        if not cur.fetchone():
            if logger_local:
                logger_local.debug("No states table yet; skipping prune.")
            conn.close()
            return
            
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
                # Use a larger exponent (1.5) for better spread
                await asyncio.sleep(backoff * (attempt ** 1.5) + random.uniform(0, 0.2))
            
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
                            'settlement_asset': p.get('settlement_asset', '').upper(),
                            'contract_type': p.get('contract_type'),
                            'maker_fee': p.get('maker_fee', 0.0),
                            'taker_fee': p.get('taker_fee', 0.0),
                            'initial_margin': p.get('initial_margin', 0.0),
                            'maintenance_margin': p.get('maintenance_margin', 0.0),
                        }
                        break # Found a match, move to the next product
        except Exception as e:
            logger.debug(f"Error processing product {p.get('symbol', 'N/A')}: {e}")
            
    return products_map

# -------------------------
# DATA PROCESSING & INDICATORS
# -------------------------
def parse_candle_response(data: Optional[dict]) -> Optional[pd.DataFrame]:
    """
    Parses the API candle response into a clean DataFrame.
    Returns None if data is invalid or empty.
    """
    if data is None or data.get('s') != 'ok':
        return None
    
    c = data.get('c', [])
    v = data.get('v', [])
    t = data.get('t', [])
    o = data.get('o', [])
    h = data.get('h', [])
    l = data.get('l', [])
    
    if not c or len(c) < 2:
        return None

    df = pd.DataFrame({
        'timestamp': t,
        'open': o,
        'high': h,
        'low': l,
        'close': c,
        'volume': v
    })
    
    # Convert timestamps to datetime and set index (optional, but good practice)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
    df = df.set_index('datetime')
    
    # Ensure all columns are float, except timestamp
    float_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    return df

def calculate_sma(x: pd.Series, period: int) -> pd.Series:
    """Calculates Simple Moving Average (SMA)."""
    return x.rolling(window=period, min_periods=period).mean().ffill()

def calculate_ema(x: pd.Series, period: int) -> pd.Series:
    """Calculates Exponential Moving Average (EMA)."""
    # Adjust=False gives the same calculation as TradingView's EMA
    return x.ewm(span=period, adjust=False).mean().ffill()

def calculate_rma(x: pd.Series, period: int) -> pd.Series:
    """Calculates Running Moving Average (RMA). Equivalent to WMA/SMA/EMA where alpha=1/period."""
    # This is a common implementation for RSI/ATR smoothing. Equivalent to EMA with alpha=1/period.
    alpha = 1 / period
    return x.ewm(com=period - 1, adjust=False).mean()

def calculate_ppo(df: pd.DataFrame, fast: int, slow: int, signal: int, use_sma: bool = False):
    close = df['close'].astype(float)
    if use_sma:
        fast_ma = calculate_sma(close, fast)
        slow_ma = calculate_sma(close, slow)
    else:
        fast_ma = calculate_ema(close, fast)
        slow_ma = calculate_ema(close, slow)
        
    slow_ma = slow_ma.replace(0, np.nan).bfill().ffill().clip(lower=1e-8) # Handle division by zero
    
    ppo = ((fast_ma - slow_ma) / slow_ma) * 100
    ppo = ppo.replace([np.inf, -np.inf], np.nan).bfill().ffill().fillna(0) # Fill remaining NaNs with 0
    
    ppo_signal = calculate_sma(ppo, signal) if use_sma else calculate_ema(ppo, signal)
    ppo_signal = ppo_signal.replace([np.inf, -np.inf], np.nan).bfill().ffill().fillna(0)
    
    return ppo, ppo_signal

def smoothrng(x: pd.Series, t: int, m: int) -> pd.Series:
    wper = t * 2 - 1
    avrng = calculate_ema(np.abs(x.diff().fillna(0)), t)
    smoothrng_val = calculate_ema(avrng, max(1, wper)) * m
    return smoothrng_val.clip(lower=1e-8).bfill().ffill()

def rngfilt(x: pd.Series, r: pd.Series, x1: int, x2: int, x3: int, x4: int) -> Tuple[pd.Series, pd.Series]:
    """Custom Range Filter for Cirrus Cloud."""
    upw = pd.Series(False, index=x.index)
    dnw = pd.Series(False, index=x.index)
    
    for i in range(1, len(x)):
        # Calculate upper and lower range thresholds dynamically
        up_band = x.iloc[i-1] + r.iloc[i-1] * x1 / 100
        dn_band = x.iloc[i-1] - r.iloc[i-1] * x2 / 100
        
        # Upper Wick (upw) logic
        if x.iloc[i] > up_band:
            upw.iloc[i] = True
        elif x.iloc[i] < up_band * (1 - x3 / 100):
            upw.iloc[i] = False
        else:
            upw.iloc[i] = upw.iloc[i-1]
            
        # Down Wick (dnw) logic
        if x.iloc[i] < dn_band:
            dnw.iloc[i] = True
        elif x.iloc[i] > dn_band * (1 + x4 / 100):
            dnw.iloc[i] = False
        else:
            dnw.iloc[i] = dnw.iloc[i-1]
            
    return upw, dnw

def kalman_filter(data: pd.Series, length: int) -> pd.Series:
    """Applies a simple Kalman Filter to smooth data."""
    if data.empty or length <= 0:
        return pd.Series(0.0, index=data.index)
        
    kf = pd.Series(0.0, index=data.index)
    q = 0.01 
    r = 1.0
    a = 1.0
    b = 0.0
    c = 1.0
    
    # Initial values
    if not data.iloc[0]: # Handle NaN/zero start
        data.iloc[0] = 50.0 
    
    pe = pd.Series(0.0, index=data.index)
    pm = pd.Series(0.0, index=data.index)
    
    pe.iloc[0] = r
    kf.iloc[0] = data.iloc[0]
    
    for i in range(1, len(data)):
        # Prediction
        pm.iloc[i] = a * pe.iloc[i-1] + q
        kf.iloc[i] = a * kf.iloc[i-1]
        
        # Update
        gain = pm.iloc[i] * c / (c * pm.iloc[i] * c + r)
        kf.iloc[i] = kf.iloc[i] + gain * (data.iloc[i] - c * kf.iloc[i])
        pe.iloc[i] = (1 - gain * c) * pm.iloc[i]
        
    return kf

def calculate_magical_momentum(df: pd.DataFrame, length: int) -> pd.Series:
    """
    Calculates the 'Magical Momentum' histogram, a custom momentum/trend indicator.
    """
    if len(df) < length * 2:
        return pd.Series(0.0, index=df.index)

    close = df['close'].astype(float)

    # 1. Calculate smoothed RSI-like indicator
    rsi = calculate_rsi(close, length)
    
    # 2. Normalize smoothed RSI (0-100 to -1 to 1)
    normalized_rsi = (rsi - 50.0) / 50.0
    
    # 3. Apply a custom smoothing/lagging component (this is the 'magical' part)
    lag = calculate_rma(normalized_rsi, length)
    
    # 4. Momentum calculation
    # Difference between the normalized RSI and its lag
    diff = normalized_rsi - lag
    
    # Smoothing the difference
    value = calculate_rma(diff, length)
    
    # Transformation (Hyperbolic tangent style)
    value = max(min(value.iloc[-1], 0.9999), -0.9999) # Clip value to prevent log error
    
    # If the series has NaNs at the beginning, we need to handle this for the full transformation
    value_series = value.fillna(0)
    value_series = value_series.clip(lower=-0.9999, upper=0.9999)

    # Transformation
    temp2 = (1 + value_series) / (1 - value_series)
    temp2 = temp2.replace([np.inf, -np.inf], np.nan).clip(lower=1e-6).fillna(1e-6)
    momentum = 0.25 * np.log(temp2)
    momentum = pd.Series(momentum, index=df.index).replace([np.inf, -np.inf], 0).fillna(0)

    # 5. Calculate final histogram (using an IIR filter/Exponential smoothing)
    hist = pd.Series(0.0, index=df.index)
    for i in range(1, len(df)):
        # Simple IIR filter: current = input + 0.5 * previous
        hist.iloc[i] = momentum.iloc[i] + 0.5 * hist.iloc[i - 1]
        
    return hist.replace([np.inf, -np.inf], 0).fillna(0)

def calculate_vwap(df: pd.DataFrame, daily_df: pd.DataFrame) -> Optional[pd.Series]:
    """ 
    Calculates the Volume Weighted Average Price (VWAP) for the current 15m day.
    """
    if daily_df is None or len(daily_df) < 2:
        return None
    
    # Get today's Daily Open (use last but one candle in daily data)
    try:
        daily_ts = daily_df['timestamp'].iloc[-2]
    except IndexError:
        return None # Not enough daily data

    # Find the first 15m candle whose timestamp is >= today's daily open timestamp
    start_index = df[df['timestamp'] >= daily_ts].index.min()
    if pd.isna(start_index):
        # If no candles match (e.g., ran too early in the day), assume the first candle
        start_index = df.index[0] 
        
    df_day = df.loc[start_index:].copy()
    if df_day.empty:
        return None
    
    # Calculate typical price (TP) for each 15m candle
    df_day['TP'] = (df_day['high'] + df_day['low'] + df_day['close']) / 3
    
    # Calculate Cumulative Volume and Cumulative (TP * Volume)
    df_day['TPV'] = df_day['TP'] * df_day['volume']
    df_day['cum_TPV'] = df_day['TPV'].cumsum()
    df_day['cum_volume'] = df_day['volume'].cumsum()

    # Calculate VWAP: Cumulative (TP * Volume) / Cumulative Volume
    vwap = df_day['cum_TPV'] / df_day['cum_volume'].clip(lower=1e-8) # Avoid division by zero
    
    # Merge the VWAP series back into the original 15m DataFrame
    # Fill missing values (for candles before today's open) with the first calculated VWAP
    full_vwap = pd.Series(vwap, index=df.index).ffill() 
    
    return full_vwap

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
    
    # Daily range
    daily_range = high - low
    
    # Fibonacci Retracements
    fib_ratios = {
        'R3': 1.000,
        'R2': 0.618,
        'R1': 0.382,
        'P': 0.000, # Pivot point itself is 0 retracement from P
        'S1': 0.382,
        'S2': 0.618,
        'S3': 1.000,
    }

    # Calculate levels
    pivots: Dict[str, float] = {}
    pivots['P'] = pivot
    
    pivots['R1'] = pivot + (daily_range * fib_ratios['R1'])
    pivots['S1'] = pivot - (daily_range * fib_ratios['S1'])
    
    pivots['R2'] = pivot + (daily_range * fib_ratios['R2'])
    pivots['S2'] = pivot - (daily_range * fib_ratios['S2'])
    
    pivots['R3'] = pivot + (daily_range * fib_ratios['R3'])
    pivots['S3'] = pivot - (daily_range * fib_ratios['S3'])

    return pivots

# -------------------------
# SIGNAL UTILITIES
# -------------------------
def get_crossover_line(df: pd.DataFrame, lines: Dict[str, float]) -> Tuple[Optional[str], Optional[float]]:
    """
    Finds the highest Fibonacci Resistance/Pivot/Support that price crossed *above* (Long, excluding R3),
    or the lowest Fibonacci Support/Pivot/Resistance that price crossed *below* (Short, excluding S3)
    in the most recent candle (index -2).
    """
    # The relevant candle is the last closed one (index -2)
    prev_close = df['close'].iloc[-3]
    curr_close = df['close'].iloc[-2]
    
    # --- Check for Crossover (Long Signal) ---
    crossover_lines: List[Tuple[float, str]] = []
    
    # Lines for Long: P, S1, S2, S3, R1, R2. (EXCLUDE R3)
    long_lines_to_check = ['R1', 'R2', 'P', 'S1', 'S2', 'S3']
    for name in long_lines_to_check:
        line = lines[name]
        # Price must cross from below to above
        if prev_close <= line and curr_close > line:
            crossover_lines.append((line, name))
            
    # For a long signal, the line we are interested in is the HIGHEST level crossed
    if crossover_lines:
        crossover_lines.sort(key=lambda x: x[0], reverse=True) # Sort by price descending
        return crossover_lines[0][1], crossover_lines[0][0] # Return the highest level
    
    # --- Check for Crossunder (Short Signal) ---
    crossunder_lines: List[Tuple[float, str]] = []
    
    # Lines for Short: P, S1, S2, R1, R2, R3. (EXCLUDE S3)
    short_lines_to_check = ['R1', 'R2', 'R3', 'P', 'S1', 'S2']
    for name in short_lines_to_check:
        line = lines[name]
        # Price must cross from above to below
        if prev_close >= line and curr_close < line:
            crossunder_lines.append((line, name))
        
    # For a short signal, the line we are interested in is the LOWEST level crossed
    if crossunder_lines:
        crossunder_lines.sort(key=lambda x: x[0], reverse=False) # Sort by price ascending
        return crossunder_lines[0][1], crossunder_lines[0][0] # Return the lowest level
        
    return None, None

def should_suppress_duplicate(last_state: Optional[Dict[str, Any]], current_signal: str, suppress_secs: int) -> bool:
    """Checks if the same signal was recently sent and should be suppressed."""
    if not last_state:
        return False
        
    state_ts = int(last_state.get("ts", 0))
    state_signal = last_state.get("state")
    
    # Only suppress if the signal is exactly the same and within the time window
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
        async with session.post(url, data=data, timeout=cfg.HTTP_TIMEOUT) as resp:
            # Check for 429 Too Many Requests
            if resp.status == 429:
                logger.warning("Telegram rate limit hit (429)")
                return False
            
            resp.raise_for_status() # Raise for other HTTP errors (4xx or 5xx)
            
            # Check Telegram response body for success
            json_response = await resp.json()
            if not json_response.get("ok"):
                logger.error(f"Telegram API error: {json_response.get('description')}")
                return False
                
            return True
            
    except ClientConnectorError:
        logger.warning("Network error sending Telegram message.")
        return False
    except ClientResponseError as e:
        logger.error(f"HTTP error sending Telegram message: {e}")
        return False
    except Exception as e:
        logger.error(f"General error sending Telegram message: {e}")
        return False

async def send_telegram_with_retries(session: aiohttp.ClientSession, token: str, chat_id: str, text: str) -> bool:
    """Sends a Telegram message with retry logic."""
    for attempt in range(cfg.TELEGRAM_RETRIES):
        if await send_telegram_async(session, token, chat_id, text):
            return True
        
        if attempt < cfg.TELEGRAM_RETRIES - 1:
            # Exponential backoff with jitter for Telegram
            delay = cfg.TELEGRAM_BACKOFF_BASE * (2 ** attempt) + random.uniform(0, 1.0)
            logger.info(f"Retrying Telegram in {delay:.2f}s...")
            await asyncio.sleep(delay)
            
    return False

# -------------------------
# DEAD MAN'S SWITCH
# -------------------------
async def check_dead_man_switch(session: aiohttp.ClientSession, state_db: StateDB):
    """
    Checks if the bot has successfully run within the configured time.
    Sends an alert if it hasn't.
    """
    try:
        last_success = state_db.get_metadata("last_success_run")
        if last_success:
            try:
                last_success_int = int(last_success)
            except Exception:
                last_success_int = 0
                
            hours_since = (now_ts() - last_success_int) / 3600.0
            
            # Check if the time since last success is greater than the configured limit
            if hours_since > cfg.DEADMAN_HOURS:
                # Check when the last alert was sent to prevent spam
                dead_alert_ts = state_db.get_metadata("dead_alert_ts")
                dead_alert_int = 0
                if dead_alert_ts:
                    try:
                        dead_alert_int = int(dead_alert_ts)
                    except Exception:
                        dead_alert_int = 0
                
                # Only alert once every 4 hours after the initial failure
                if now_ts() - dead_alert_int > 4 * 3600:
                    msg = (
                        f"🚨 DEAD MAN'S SWITCH: **{sanitize_for_telegram(cfg.BOT_NAME)}** hasn't succeeded in {hours_since:.1f} hours!\n"
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
# MAIN LOGIC
# -------------------------
async def evaluate_pair_async(
    session: aiohttp.ClientSession,
    fetcher: DataFetcher,
    products_map: Dict[str, dict],
    pair_name: str,
    last_state: Optional[Dict[str, Any]],
) -> Optional[str]:
    """
    Fetches data, calculates indicators, applies signal logic, and sends alerts for a single pair.
    """
    METRICS["pairs_checked"] += 1
    logger.debug(f"Starting evaluation for {pair_name}")
    
    product_details = products_map.get(pair_name)
    if not product_details:
        logger.debug(f"{pair_name}: Product details not found in map.")
        return last_state.get("state") if last_state else None
        
    symbol = product_details['symbol']
    
    # Configuration for data limits (default or special pair override)
    limit_15m = cfg.SPECIAL_PAIRS.get(pair_name, {}).get("limit_15m", 150)
    min_required_15m = cfg.SPECIAL_PAIRS.get(pair_name, {}).get("min_required", 120)
    limit_5m = cfg.SPECIAL_PAIRS.get(pair_name, {}).get("limit_5m", 250)
    min_required_5m = cfg.SPECIAL_PAIRS.get(pair_name, {}).get("min_required_5m", 200)

    # --- 1. Concurrent Data Fetch ---
    try:
        # Fetch 15m data (needed for all indicators)
        task15m = fetcher.fetch_candles(session, symbol, "15", limit_15m)
        # Fetch 5m data (only if RMA200 is enabled)
        task5m = fetcher.fetch_candles(session, symbol, "5", limit_5m) if cfg.USE_RMA200 else asyncio.Future()
        # Fetch 1D data (needed for Fibonacci Pivots)
        task_daily = fetcher.fetch_candles(session, symbol, "1D", cfg.PIVOT_LOOKBACK_PERIOD)
        
        # Run all fetches concurrently
        res15, res5, res_daily = await asyncio.gather(task15m, task5m, task_daily, return_exceptions=True)
        
        # Check for exceptions
        if isinstance(res15, Exception) or isinstance(res_daily, Exception):
            logger.error(f"Error during concurrent fetch for {pair_name}. 15m: {res15}, Daily: {res_daily}")
            METRICS["network_errors"] += 1
            return last_state.get("state") if last_state else None
        if cfg.USE_RMA200 and isinstance(res5, Exception):
            logger.warning(f"Error fetching 5m data for {pair_name}. RMA200 check disabled for this run. Error: {res5}")
            res5 = None # Treat as not available
            
    except Exception as e:
        logger.error(f"Error during concurrent fetch for {pair_name}: {e}")
        METRICS["network_errors"] += 1
        return last_state.get("state") if last_state else None
        
    df_15m = parse_candle_response(res15)
    df_5m = parse_candle_response(res5) if cfg.USE_RMA200 else None
    df_daily = parse_candle_response(res_daily)

    # Check for sufficient data
    if df_15m is None or len(df_15m) < min_required_15m:
        logger.debug(f"{pair_name}: Insufficient 15m data (got {len(df_15m) if df_15m is not None else 0}, need {min_required_15m})")
        return last_state.get("state") if last_state else None
        
    if cfg.USE_RMA200 and (df_5m is None or len(df_5m) < min_required_5m):
        logger.debug(f"{pair_name}: Insufficient 5m data (got {len(df_5m) if df_5m is not None else 0}, need {min_required_5m})")
        return last_state.get("state") if last_state else None
        
    pivots = calculate_fibonacci_pivots(df_daily)
    if pivots is None or df_daily is None or len(df_daily) < 2:
        logger.debug(f"{pair_name}: Insufficient daily data for pivots.")
        pivots_available = False
    else:
        pivots_available = True

    # --- 2. Calculate Indicators ---
    # PPO (15m)
    ppo, _ = calculate_ppo(df_15m, cfg.PPO_FAST, cfg.PPO_SLOW, cfg.PPO_SIGNAL, cfg.PPO_USE_SMA)
    ppo_curr = ppo.iloc[-2]
    
    # Cirrus Cloud (15m)
    rng_15m = smoothrng(df_15m['close'], cfg.X4, 1)
    upw_15m, dnw_15m = rngfilt(df_15m['close'], rng_15m, cfg.X1, cfg.X2, cfg.X3, cfg.X4)
    
    # Magical Momentum Histogram (15m)
    magical_hist = calculate_magical_momentum(df_15m, 15)
    
    # VWAP (15m) - calculated daily
    vwap_series = calculate_vwap(df_15m, df_daily)
    
    # RMA 200 (5m) - only if enabled
    rma_200_available = False
    rma200_5m_curr = 0.0
    if cfg.USE_RMA200 and df_5m is not None:
        rma_200_5m = calculate_rma(df_5m['close'], 200)
        # Check if RMA200 has enough periods to be valid
        if len(rma_200_5m.dropna()) >= 1: 
            # We take the most recent 5m value, which is in the most recent 15m period
            rma200_5m_curr = rma_200_5m.iloc[-1] 
            rma_200_available = True
            
    # NEW: Calculate RMA 50 for 15m (needed for the new filter)
    rma_50_15m = calculate_rma(df_15m['close'], 50)
    # Check if RMA50 has enough periods to be valid
    if len(rma_50_15m.dropna()) < 1:
        logger.debug(f"{pair_name}: Insufficient data for RMA 50. Skipping.")
        return last_state.get("state") if last_state else None
    rma_50_15m_curr = rma_50_15m.iloc[-2] # Current closed 15m candle's RMA50

    # --- 3. Extract Current Candle Data (Index -2) ---
    try:
        current_candle = df_15m.iloc[-2]
        prev_close = df_15m['close'].iloc[-3]
    except IndexError:
        logger.debug(f"{pair_name}: Not enough candles for comparison.")
        return last_state.get("state") if last_state else None

    open_c = float(current_candle['open'])
    close_c = float(current_candle['close'])
    high_c = float(current_candle['high'])
    low_c = float(current_candle['low'])
    
    is_green = close_c > open_c
    is_red = close_c < open_c
    cloud_state_curr = cloud_state_from(upw_15m, dnw_15m, -2)
    magical_hist_curr = magical_hist.iloc[-2]
    vwap_curr = vwap_series.iloc[-2] if vwap_series is not None else 0.0

    # --- 4. Candle Pre-checks & Indicator Filters (Index -2) ---
    
    # Extreme Candle Check
    candle_pct_change = abs(close_c - open_c) / open_c * 100
    if candle_pct_change > cfg.EXTREME_CANDLE_PCT:
        logger.debug(f"{pair_name}: Skipped due to extreme candle ({candle_pct_change:.2f}% > {cfg.EXTREME_CANDLE_PCT}%)")
        return last_state.get("state") if last_state else None

    # Wick checks (using the user-requested 20% limit)
    total_range = high_c - low_c
    WICK_MAX_PCT = 0.20 # 20% limit from user request
    
    if total_range <= 1e-6: # Avoid division by zero
        upper_wick_ok = True
        lower_wick_ok = True
    else:
        upper_wick = high_c - max(open_c, close_c)
        lower_wick = min(open_c, close_c) - low_c

        # NEW USER LOGIC: 
        # For Long (Green Candle): Upper wick must be < 20% of total range.
        # For Short (Red Candle): Lower wick must be < 20% of total range.
        
        # If it's a green candle (long signal), check if its upper wick is small
        upper_wick_ok = not is_green or (upper_wick / total_range) < WICK_MAX_PCT
        
        # If it's a red candle (short signal), check if its lower wick is small
        lower_wick_ok = not is_red or (lower_wick / total_range) < WICK_MAX_PCT
        
    # RMA FILTER: 15m RMA50 and 5m RMA200 must confirm the trend

    # 1. RMA 50 (15m) Check
    # Buy: Close > RMA50 | Sell: Close < RMA50
    rma_50_long_ok = close_c > rma_50_15m_curr
    rma_50_short_ok = close_c < rma_50_15m_curr

    # 2. RMA 200 (5m) Check (Keep the cfg.USE_RMA200 safety check)
    # Buy: Close > RMA200 | Sell: Close < RMA200
    rma_200_long_ok = not cfg.USE_RMA200 or (rma_200_available and close_c > rma200_5m_curr)
    rma_200_short_ok = not cfg.USE_RMA200 or (rma_200_available and close_c < rma200_5m_curr)

    # Final combined RMA check
    rma_long_ok = rma_50_long_ok and rma_200_long_ok
    rma_short_ok = rma_50_short_ok and rma_200_short_ok
    
    # --- General General Conditions (LGC/SGC) ---
    # These combine all the user's non-crossover conditions (as confirmed in previous turn)
    LGC = (
        (cloud_state_curr == "long") and             # Cirrus cloud should be green
        (magical_hist_curr > 0) and                  # Magical Momentum Hist should be above 0
        (is_green) and                               # Candle should be green
        (upper_wick_ok) and                          # Upper wick check (specific wick check is applied above)
        (rma_long_ok)                                # Combined RMA check
    )

    SGC = (
        (cloud_state_curr == "short") and            # Cirrus cloud should be red
        (magical_hist_curr < 0) and                  # Magical Momentum Hist should be below 0
        (is_red) and                                 # Candle should be red
        (lower_wick_ok) and                          # Lower wick check (specific wick check is applied above)
        (rma_short_ok)                               # Combined RMA check
    )
    
    # --- 5. Signal Logic ---
    current_signal = None
    message = None
    suppress_secs = cfg.DUPLICATE_SUPPRESSION_SECONDS
    
    # VWAP Crossover/Crossunder Check
    vwap_crossover = (vwap_series is not None and prev_close <= vwap_curr and close_c > vwap_curr)
    vwap_crossunder = (vwap_series is not None and prev_close >= vwap_curr and close_c < vwap_curr)
    
    # Fibonacci Crossover/Crossunder Check
    # NOTE: The get_crossover_line is now modified to respect R3 (exclude long) and S3 (exclude short)
    long_crossover_name, long_crossover_line = get_crossover_line(df_15m, pivots) if pivots_available else (None, None)
    short_crossover_name, short_crossover_line = get_crossover_line(df_15m, pivots) if pivots_available else (None, None)

    fib_crossover = long_crossover_name is not None
    fib_crossunder = short_crossover_name is not None
    
    up_sig = "⬆️"
    down_sig = "⬇️"

    # VBuy Signal (VWAP Cross Above) - Uses LGC
    vbuy = (
        vwap_crossover and 
        LGC
    )
    
    # VSell Signal (VWAP Cross Below) - Uses SGC
    vsell = (
        vwap_crossunder and 
        SGC
    )
    
    # Fib Long Signal (Pivot/Resistance Cross Above) - Uses LGC
    fib_long = (
        fib_crossover and 
        LGC
    )

    # Fib Short Signal (Pivot/Support Cross Below) - Uses SGC
    fib_short = (
        fib_crossunder and 
        SGC
    )

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

    # --- 7. Send Alert and Update State ---
    if message and current_signal:
        telegram_log_state(pair_name, current_signal, message)
        if await send_telegram_with_retries(session, cfg.TELEGRAM_BOT_TOKEN, cfg.TELEGRAM_CHAT_ID, message):
            return current_signal
        else:
            logger.error(f"Failed to send Telegram alert for {pair_name}: {current_signal}")
            return last_state.get("state") if last_state else None
    
    # If no signal, return the previous state (if it was a signal state) or None
    return current_signal

# -------------------------
# EXECUTION CORE
# -------------------------
async def run_once(state_db: StateDB, send_test: bool = False):
    """
    Core function for a single execution run.
    """
    reset_metrics()
    logger.info("Starting bot run.")
    
    # Use a TCPConnector to respect the configured concurrency limit
    connector = TCPConnector(limit=cfg.MAX_CONCURRENCY)
    async with aiohttp.ClientSession(connector=connector) as session:
        
        # 1. Check Dead Man's Switch (runs before anything else)
        await check_dead_man_switch(session, state_db)
        
        # 2. Send Test Message (if configured)
        if cfg.SEND_TEST_MESSAGE and send_test:
            test_msg = f"✅ **{sanitize_for_telegram(cfg.BOT_NAME)}** is starting. Time: {human_ts()}. Execution limit: {cfg.MAX_EXEC_TIME}s."
            # Use the shared session for the test message
            await send_telegram_with_retries(session, cfg.TELEGRAM_BOT_TOKEN, cfg.TELEGRAM_CHAT_ID, test_msg)

        products_map = load_products_cache()
        circuit = CircuitBreaker()
        # Use MAX_CONCURRENCY for the internal semaphore to limit active API calls
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
            ) for pair_name in cfg.PAIRS
        ]
        
        # Use a timeout to ensure the cron job finishes within its window
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=cfg.MAX_EXEC_TIME
            )
        except asyncio.TimeoutError:
            logger.error(f"❌ Execution timed out after {cfg.MAX_EXEC_TIME} seconds.")
            raise SystemExit(EXIT_TIMEOUT)
        
        # 5. Process Results and Update State DB
        success_count = 0
        for pair_name, new_signal in zip(cfg.PAIRS, results):
            if isinstance(new_signal, Exception):
                logger.error(f"Exception during evaluation of {pair_name}: {new_signal}")
                METRICS["logic_errors"] += 1
                continue
            
            if new_signal is not None:
                # Update state only if a new signal was generated or the previous one was repeated (to update timestamp)
                state_db.set(pair_name, new_signal, now_ts())
                if new_signal == last_alerts.get(pair_name, {}).get("state"):
                    logger.debug(f"State updated (timestamp only) for {pair_name}: {new_signal}")
                else:
                    logger.debug(f"New state recorded for {pair_name}: {new_signal}")
            else:
                # Keep the previous state if no new signal and no reset logic is implemented here
                pass 
                
            success_count += 1
        
        # 6. Final Logging and Metrics
        METRICS["execution_time"] = time.time() - METRICS["start_time"]
        
        # Only update last_success_run if at least one pair was checked and no major failure occurred
        if success_count > 0:
            state_db.set_metadata("last_success_run", str(now_ts()))
        
        logger.info(f"Finished bot run. Pairs checked: {METRICS['pairs_checked']} | "
                    f"Alerts sent: {METRICS['alerts_sent']} | Errors: API={METRICS['api_errors']}, "
                    f"Net={METRICS['network_errors']}, Logic={METRICS['logic_errors']} | "
                    f"Time: {METRICS['execution_time']:.2f}s")

def main():
    """Main entry point for the bot execution."""
    
    # 0. Prune DB first (runs quickly)
    prune_old_state_records(cfg.STATE_DB_PATH, cfg.STATE_EXPIRY_DAYS, logger)
    
    # 1. Acquire Process Lock
    lock = ProcessLock(cfg.PID_LOCK_PATH)
    if not lock.acquire():
        logger.critical(f"Process already running or lock failed. Exiting: {cfg.PID_LOCK_PATH}")
        sys.exit(EXIT_LOCK_CONFLICT)
        
    # 2. Check Memory Limit
    try:
        if psutil.virtual_memory().available < cfg.MEMORY_LIMIT_BYTES:
            logger.critical(f"System memory too low: {psutil.virtual_memory().available / 1024**2:.0f}MB < {cfg.MEMORY_LIMIT_BYTES / 1024**2:.0f}MB")
            # Proceed anyway, but log the critical error
    except Exception:
        logger.debug("Memory check: psutil error or not supported.")
        
    state_db = StateDB(cfg.STATE_DB_PATH)
    if cfg.RESET_STATE:
        try:
            os.remove(cfg.STATE_DB_PATH)
            logger.warning(f"State DB removed due to RESET_STATE=True: {cfg.STATE_DB_PATH}")
        except Exception as e:
            logger.warning(f"Could not remove DB file: {e}")
        state_db = StateDB(cfg.STATE_DB_PATH) # Re-initialize

    try:
        # Determine if running in an infinite loop (e.g., in a Docker container)
        # or as a single run (e.g., cron-jobs.org or GitHub Actions)
        LOOP_MODE = str_to_bool(os.getenv("LOOP_MODE", "false"))
        interval = cfg.RUN_LOOP_INTERVAL
        
        if LOOP_MODE:
            # --- Loop Run Logic (for long-running host) ---
            logger.info("🚀 Loop run mode.")
            loop_start = time.time()
            stop_requested = False

            def signal_handler(signum, frame):
                nonlocal stop_requested
                logger.info(f"Signal {signum} received. Stopping loop gracefully...")
                stop_requested = True
            
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)

            while not stop_requested:
                try:
                    # Run once will send the test message only on the first run (loop_start == time.time() is imprecise)
                    # We'll rely on the simple 'send_test=True' which is only used on the first pass
                    asyncio.run(run_once(state_db, send_test=True)) 
                    cfg.SEND_TEST_MESSAGE = False # Prevent sending test message on subsequent loops
                except SystemExit as e:
                    logger.error(f"Exited with code {e.code} during loop.")
                    if e.code == EXIT_TIMEOUT:
                        logger.warning("Continuing loop after timeout exit.")
                    elif e.code != EXIT_API_FAILURE:
                        break # Stop for non-recoverable errors
                except Exception:
                    logger.exception("Unhandled in run_once")

                elapsed = time.time() - loop_start
                to_sleep = max(0, interval - elapsed)
                
                # Use simple sleep that respects stop_requested
                if to_sleep > 0 and not stop_requested:
                    logger.debug(f"Sleeping for {to_sleep:.2f} seconds.")
                    time.sleep(to_sleep)

                # Restart the timer for the next loop
                loop_start = time.time() if not stop_requested else loop_start

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
