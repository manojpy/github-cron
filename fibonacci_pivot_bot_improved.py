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
import resource
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
    from pydantic import BaseModel, field_validator
except Exception:
    print("⚠️ pydantic not found. Install with: pip install 'pydantic>=2.5.0'")
    raise

# -------------------------
# CONFIGURATION MODEL
# -------------------------
class Config(BaseModel):
    TELEGRAM_BOT_TOKEN: str
    TELEGRAM_CHAT_ID: str
    DEBUG_MODE: bool = False
    SEND_TEST_MESSAGE: bool = True
    RESET_STATE: bool = False
    DELTA_API_BASE: str = "https://api.india.delta.exchange"
    PAIRS: List[str] = ["BTCUSD", "ETHUSD", "SOLUSD", "AVAXUSD", "BCHUSD", "XRPUSD", "BNBUSD", "LTCUSD", "DOTUSD", "ADAUSD", "SUIUSD", "AAVEUSD"]
    SPECIAL_PAIRS: Dict[str, Dict[str, int]] = {"SOLUSD": {"limit_15m": 210, "min_required": 180, "limit_5m": 300, "min_required_5m": 200}}
    PPO_FAST: int = 7
    PPO_SLOW: int = 16
    PPO_SIGNAL: int = 5
    PPO_USE_SMA: bool = False
    X1: int = 22
    X2: int = 9
    X3: int = 15
    X4: int = 5
    PIVOT_LOOKBACK_PERIOD: int = 15
    STATE_DB_PATH: str = "fib_state.sqlite"
    LOG_FILE: str = "fibonacci_pivot_bot.log"
    MAX_CONCURRENCY: int = 6
    HTTP_TIMEOUT: int = 15
    FETCH_RETRIES: int = 3
    FETCH_BACKOFF: float = 1.5
    JITTER_MIN: float = 0.05
    JITTER_MAX: float = 0.6
    STATE_EXPIRY_DAYS: int = 30
    DUPLICATE_SUPPRESSION_SECONDS: int = 3600
    EXTREME_CANDLE_PCT: float = 8.0
    RUN_LOOP_INTERVAL: int = 900
    MAX_EXEC_TIME: int = 25
    USE_RMA200: bool = True
    PRODUCTS_CACHE_TTL: int = 86400
    LOG_LEVEL: str = "INFO"
    TELEGRAM_RETRIES: int = 3
    TELEGRAM_BACKOFF_BASE: float = 2.0
    MEMORY_LIMIT_BYTES: int = 400_000_000  # ~400MB safe default for small hosts
    REQUESTS_PER_MINUTE: int = 60

    @field_validator("TELEGRAM_BOT_TOKEN")
    @classmethod
    def token_not_default(cls, v):
        if not v or v == "xxxx":
            raise ValueError("TELEGRAM_BOT_TOKEN must be set and not be 'xxxx'")
        return v

    @field_validator("TELEGRAM_CHAT_ID")
    @classmethod
    def chat_id_not_default(cls, v):
        if not v or v == "xxxx":
            raise ValueError("TELEGRAM_CHAT_ID must be set and not be 'xxxx'")
        return v

# -------------------------
# DEFAULT CONFIG
# -------------------------
DEFAULT_CONFIG = {
    "TELEGRAM_BOT_TOKEN": "8462496498:AAHURmrq_syb7ab1q0R9dSPDJ-8UOCA05uU",
    "TELEGRAM_CHAT_ID": "203813932",
    "DEBUG_MODE": False,
    "SEND_TEST_MESSAGE": True,
    "RESET_STATE": False,
    "DELTA_API_BASE": "https://api.india.delta.exchange",
    "PAIRS": ["BTCUSD", "ETHUSD", "SOLUSD", "AVAXUSD", "BCHUSD", "XRPUSD", "BNBUSD", "LTCUSD", "DOTUSD", "ADAUSD", "SUIUSD", "AAVEUSD"],
    "SPECIAL_PAIRS": {"SOLUSD": {"limit_15m": 210, "min_required": 180, "limit_5m": 300, "min_required_5m": 200}},
    "PPO_FAST": 7,
    "PPO_SLOW": 16,
    "PPO_SIGNAL": 5,
    "PPO_USE_SMA": False,
    "X1": 22,
    "X2": 9,
    "X3": 15,
    "X4": 5,
    "PIVOT_LOOKBACK_PERIOD": 15,
    "STATE_DB_PATH": "fib_state.sqlite",
    "LOG_FILE": "fibonacci_pivot_bot.log",
    "MAX_CONCURRENCY": 6,
    "HTTP_TIMEOUT": 15,
    "FETCH_RETRIES": 3,
    "FETCH_BACKOFF": 1.5,
    "JITTER_MIN": 0.05,
    "JITTER_MAX": 0.6,
    "STATE_EXPIRY_DAYS": 30,
    "DUPLICATE_SUPPRESSION_SECONDS": 3600,
    "EXTREME_CANDLE_PCT": 8.0,
    "RUN_LOOP_INTERVAL": 900,
    "MAX_EXEC_TIME": 25,
    "USE_RMA200": True,
    "PRODUCTS_CACHE_TTL": 86400,
    "LOG_LEVEL": "INFO",
    "TELEGRAM_RETRIES": 3,
    "TELEGRAM_BACKOFF_BASE": 2.0,
    "MEMORY_LIMIT_BYTES": 400_000_000,
    "REQUESTS_PER_MINUTE": 60,
}

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
def str_to_bool(value): 
    return str(value).strip().lower() in ("true", "1", "yes", "y", "t")

def load_config() -> Config:
    base = DEFAULT_CONFIG.copy()
    config_file = os.getenv("CONFIG_FILE", "config_fib.json")

    if Path(config_file).exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                user_cfg = json.load(f)
            base.update(user_cfg)
            print(f"✅ Loaded configuration from {config_file}")
        except Exception as e:
            print(f"⚠️ Warning: unable to parse {config_file}: {e}")
    else:
        print(f"⚠️ Warning: config file {config_file} not found, using defaults.")

    def override(key, default=None, cast=None):
        val = os.getenv(key)
        if val is not None:
            base[key] = cast(val) if cast else val
        else:
            base[key] = base.get(key, default)

    override("DEBUG_MODE", False, str_to_bool)
    override("SEND_TEST_MESSAGE", False, str_to_bool)
    override("STATE_DB_PATH")
    override("LOG_FILE")
    override("DELTA_API_BASE")
    override("MAX_CONCURRENCY", 6, int)
    override("HTTP_TIMEOUT", 15, int)
    override("MAX_EXEC_TIME", 25, int)
    override("REQUESTS_PER_MINUTE", 60, int)

    return Config(**base)

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
# LOGGER
# -------------------------
logger = logging.getLogger("fibonacci_pivot_bot")
log_level = getattr(logging, cfg.LOG_LEVEL if isinstance(cfg.LOG_LEVEL, str) else "INFO", logging.INFO)
logger.setLevel(logging.DEBUG if cfg.DEBUG_MODE else log_level)

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {"time": self.formatTime(record), "level": record.levelname, "msg": record.getMessage()}
        return json.dumps(log_obj, ensure_ascii=False)

formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s") if cfg.DEBUG_MODE else JSONFormatter()

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

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
    """File-based lock with stale detection"""
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
            try:
                if self.lock_path.exists():
                    with open(self.lock_path, 'r') as f:
                        raw = f.read().strip()
                    old_pid = int(raw) if raw.isdigit() else None
                    if old_pid and not os.path.exists(f"/proc/{old_pid}"):
                        logger.warning(f"Removing stale lock from PID {old_pid}")
                        try:
                            self.lock_path.unlink(missing_ok=True)
                        except Exception:
                            pass
                        return self.acquire()
            except Exception:
                pass
            if self.fd:
                try:
                    self.fd.close()
                except Exception:
                    pass
            return False

    def release(self):
        """Release the lock"""
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
# ENHANCED CIRCUIT BREAKER
# -------------------------
class EnhancedCircuitBreaker:
    """Prevent cascading failures when API is down with enhanced recovery"""
    def __init__(self, failure_threshold: int = 5, timeout: int = 300, success_threshold: int = 3):
        self.failures = 0
        self.last_failure = 0.0
        self.threshold = failure_threshold
        self.timeout = timeout
        self.success_count = 0
        self.success_threshold = success_threshold
        self.state = "CLOSED"

    async def call(self, func):
        """Wrap an async function with circuit breaker logic"""
        if self.state == "OPEN":
            if time.time() - self.last_failure > self.timeout:
                self.state = "HALF_OPEN"
                logger.info("Circuit breaker: attempting HALF_OPEN")
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = await func()
            if self.state == "HALF_OPEN":
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    self.state = "CLOSED"
                    self.success_count = 0
                    logger.info("Circuit breaker: CLOSED (recovery confirmed)")
            self.failures = 0
            self.state = "CLOSED"
            return result
        except ClientConnectorError:
            self.failures += 1
            self.last_failure = time.time()
            self.success_count = 0
            METRICS["network_errors"] += 1
            if self.failures >= self.threshold:
                self.state = "OPEN"
                logger.error(f"Circuit breaker OPEN after {self.failures} network failures")
            raise
        except Exception:
            self.failures += 1
            self.last_failure = time.time()
            self.success_count = 0
            METRICS["api_errors"] += 1
            if self.failures >= self.threshold:
                self.state = "OPEN"
                logger.error(f"Circuit breaker OPEN after {self.failures} failures")
            raise

# -------------------------
# UTILITIES
# -------------------------
def now_ts() -> int:
    return int(time.time())

def human_ts() -> str:
    tz = pytz.timezone("Asia/Kolkata")
    return datetime.now(tz).strftime("%d-%m-%Y @ %H:%M IST")

def sanitize_for_telegram(text: Union[str, float, int]) -> str:
    """Prevent Telegram HTML injection"""
    return html.escape(str(text))

def is_cron_environment():
    """Detect if running in cron environment more reliably"""
    cron_indicators = [
        os.getenv("CRON_JOB"),
        os.getenv("GITHUB_ACTIONS"), 
        not os.isatty(sys.stdin.fileno()) if hasattr(sys.stdin, 'fileno') else True,  # No terminal
        "CRON" in os.getenv("USER", ""),    # Common cron usernames
        len(sys.argv) > 1 and any(arg in sys.argv for arg in ["--cron", "--once"])
    ]
    return any(cron_indicators)

# -------------------------
# ENHANCED SQLITE STATE DB (WAL mode with connection pooling)
# -------------------------
class StateDB:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn = None
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
        self._ensure_tables()
        try:
            os.chmod(self.db_path, 0o600)
        except Exception:
            pass

    def get_connection(self):
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False, 
                                       isolation_level=None, timeout=30.0)
            try:
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute("PRAGMA synchronous=NORMAL")
                self._conn.execute("PRAGMA foreign_keys=ON")
                self._conn.execute("PRAGMA busy_timeout=5000")
            except Exception:
                pass
        return self._conn

    def _ensure_tables(self):
        conn = self.get_connection()
        cur = conn.cursor()
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
        conn.commit()

    def db_operation_with_retry(self, operation, max_retries=3):
        """Execute database operation with retry on lock conflicts"""
        for attempt in range(max_retries):
            try:
                return operation()
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    time.sleep(0.1 * (2 ** attempt))
                    continue
                raise
            except Exception:
                raise

    def load_all(self) -> Dict[str, Dict[str, Any]]:
        def _load():
            conn = self.get_connection()
            cur = conn.cursor()
            cur.execute("SELECT pair, state, ts FROM states")
            rows = cur.fetchall()
            return {r[0]: {"state": r[1], "ts": int(r[2] or 0)} for r in rows}
        
        return self.db_operation_with_retry(_load)

    def get(self, pair: str) -> Optional[Dict[str, Any]]:
        def _get():
            conn = self.get_connection()
            cur = conn.cursor()
            cur.execute("SELECT state, ts FROM states WHERE pair = ?", (pair,))
            r = cur.fetchone()
            if not r:
                return None
            return {"state": r[0], "ts": int(r[1] or 0)}
        
        return self.db_operation_with_retry(_get)

    def set(self, pair: str, state: Optional[str], ts: Optional[int] = None):
        def _set():
            ts_val = int(ts or now_ts())
            conn = self.get_connection()
            cur = conn.cursor()
            if state is None:
                cur.execute("DELETE FROM states WHERE pair = ?", (pair,))
            else:
                cur.execute(
                    "INSERT INTO states(pair, state, ts) VALUES (?, ?, ?) "
                    "ON CONFLICT(pair) DO UPDATE SET state=excluded.state, ts=excluded.ts",
                    (pair, state, ts_val)
                )
            conn.commit()
        
        self.db_operation_with_retry(_set)

    def get_metadata(self, key: str) -> Optional[str]:
        def _get_meta():
            conn = self.get_connection()
            cur = conn.cursor()
            cur.execute("SELECT value FROM metadata WHERE key = ?", (key,))
            r = cur.fetchone()
            return r[0] if r else None
        
        return self.db_operation_with_retry(_get_meta)

    def set_metadata(self, key: str, value: str):
        def _set_meta():
            conn = self.get_connection()
            cur = conn.cursor()
            cur.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)", (key, value))
            conn.commit()
        
        self.db_operation_with_retry(_set_meta)

    def close(self):
        try:
            if self._conn:
                self._conn.close()
                self._conn = None
        except Exception:
            pass

    def __del__(self):
        self.close()

# -------------------------
# PRUNE (smart daily)
# -------------------------
def prune_old_state_records(db_path: str, expiry_days: int = 30, logger_local: logging.Logger = None):
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
            pass

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
                pass

        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='states'")
        if not cur.fetchone():
            if logger_local:
                logger_local.debug("No states table yet; skipping prune.")
            conn.close()
            return

        cutoff = int(time.time()) - (expiry_days * 86400)
        cur.execute("DELETE FROM states WHERE ts < ?", (cutoff,))
        deleted = cur.rowcount

        cur.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES ('last_prune', ?)", (datetime.utcnow().isoformat(),))
        conn.commit()

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
    circuit_breaker: Optional[EnhancedCircuitBreaker] = None
):
    async def _fetch():
        for attempt in range(1, retries + 1):
            try:
                async with session.get(url, params=params, timeout=timeout) as resp:
                    text = await resp.text()
                    if resp.status >= 400:
                        logger.debug(f"HTTP {resp.status} {url} {params} - {text[:200]}")
                        raise ClientResponseError(resp.request_info, resp.history, status=resp.status)

                    try:
                        return await resp.json()
                    except Exception:
                        logger.debug(f"Non-JSON response from {url}: {text[:200]}")
                    return {}
            except (asyncio.TimeoutError, ClientConnectorError) as e:
                logger.debug(f"Fetch attempt {attempt} error: {e} for {url}")
                METRICS["network_errors"] += 1
                if attempt == retries:
                    logger.warning(f"Failed to fetch {url} after {retries} attempts.")
                    return None
                await asyncio.sleep(backoff * (attempt ** 1.2) + random.uniform(0, 0.2))
            except ClientResponseError as cre:
                METRICS["api_errors"] += 1
                logger.debug(f"ClientResponseError fetching {url}: {cre}")
                return None
            except Exception as e:
                logger.exception(f"Unexpected fetch error for {url}: {e}")
                METRICS["logic_errors"] += 1
                return None
        return None

    if circuit_breaker:
        return await circuit_breaker.call(_fetch)
    else:
        return await _fetch()

# -------------------------
# RATE LIMITED DATA FETCHER
# -------------------------
class RateLimitedDataFetcher:
    def __init__(self, base_url: str, max_parallel: int = 6, timeout: int = 15, 
                 circuit_breaker: Optional[EnhancedCircuitBreaker] = None,
                 requests_per_minute: int = 60):
        self.base_url = base_url.rstrip("/")
        self.semaphore = asyncio.Semaphore(max_parallel)
        self.timeout = timeout
        self.cache: Dict[str, Tuple[float, Any]] = {}
        self.circuit_breaker = circuit_breaker
        self.requests_per_minute = requests_per_minute
        self.last_request_time = 0
        self.min_interval = 60.0 / requests_per_minute

    async def _rate_limit(self):
        """Simple rate limiting implementation"""
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.min_interval:
            await asyncio.sleep(self.min_interval - elapsed)
        self.last_request_time = time.time()

    async def fetch_products(self, session: aiohttp.ClientSession):
        await self._rate_limit()
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
        await self._rate_limit()
        key = f"candles:{symbol}:{resolution}:{limit}"
        if key in self.cache:
            age, data = self.cache[key]
            if time.time() - age < 5:
                return data

        await asyncio.sleep(random.uniform(cfg.JITTER_MIN, cfg.JITTER_MAX))
        url = f"{self.base_url}/v2/chart/history"
        params = {
            "resolution": resolution,
            "symbol": symbol,
            "from": int(time.time()) - (limit * (int(resolution) if resolution != 'D' else 1440) * 60),
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
    try:
        with open("products_cache.json", "w", encoding="utf-8") as f:
            json.dump(products_map, f, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"Failed to save products cache: {e}")

def build_products_map(api_products: dict) -> Dict[str, dict]:
    products_map: Dict[str, dict] = {}
    if not api_products or not isinstance(api_products, dict):
        return products_map
    for p in api_products.get("result", []):
        try:
            symbol = p.get("symbol", "")
            symbol_norm = symbol.replace("_USDT", "USD").replace("USDT", "USD")
            if p.get("contract_type") == "perpetual_futures":
                for pair_name in cfg.PAIRS:
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
    if not res or not isinstance(res, dict):
        return None
    if not res.get("success", True) and "result" not in res:
        return None
    resr = res.get("result", {}) or {}
    arrays = [resr.get('t', []), resr.get('o', []), resr.get('h', []), resr.get('l', []), resr.get('c', []), resr.get('v', [])]
    if any(len(a) == 0 for a in arrays):
        return None
    min_len = min(map(len, arrays))
    df = pd.DataFrame({
        'timestamp': resr.get('t', [])[:min_len],
        'open': resr.get('o', [])[:min_len],
        'high': resr.get('h', [])[:min_len],
        'low': resr.get('l', [])[:min_len],
        'close': resr.get('c', [])[:min_len],
        'volume': resr.get('v', [])[:min_len]
    })
    df = df.sort_values('timestamp').drop_duplicates(subset='timestamp').reset_index(drop=True)
    if df.empty or float(df['close'].astype(float).iloc[-1]) <= 0:
        return None
    return df

# -------------------------
# INDICATORS
# -------------------------
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
    slow_ma = slow_ma.replace(0, np.nan).bfill().ffill()
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
    avg_gain = calculate_rma(gain, rsi_len)
    avg_loss = calculate_rma(loss, rsi_len).replace(0, np.nan).bfill().ffill().clip(lower=1e-8)
    rs = avg_gain.divide(avg_loss)
    rsi_value = 100 - (100 / (1 + rs))
    rsi_value = rsi_value.replace([np.inf, -np.inf], np.nan).bfill().ffill()
    smooth_rsi = kalman_filter(rsi_value, kalman_len).bfill().ffill()
    return smooth_rsi

# Corrected Magical Momentum Histogram per Pinescript v6
def calculate_magical_momentum_hist(df: pd.DataFrame, period: int = 144, responsiveness: float = 0.9) -> pd.Series:
    n = len(df)
    if n == 0:
        return pd.Series([], dtype=float)
    if n < period + 50:
        return pd.Series(np.zeros(n), index=df.index, dtype=float)

    source = df['close'].astype(float).copy()
    sd = source.rolling(window=50, min_periods=10).std() * max(0.00001, responsiveness)
    sd = sd.bfill().ffill().clip(lower=1e-6)

    worm = source.copy()
    for i in range(1, n):
        diff = source.iloc[i] - worm.iloc[i - 1]
        delta = np.sign(diff) * sd.iloc[i] if abs(diff) > sd.iloc[i] else diff
        worm.iloc[i] = worm.iloc[i - 1] + delta

    ma = source.rolling(window=period, min_periods=max(5, period // 3)).mean().bfill().ffill()
    denom = worm.replace(0, np.nan).bfill().ffill()
    raw_momentum = (worm - ma) / denom
    raw_momentum = raw_momentum.replace([np.inf, -np.inf], np.nan).fillna(0)

    min_med = raw_momentum.rolling(window=period, min_periods=max(5, period // 3)).min().bfill().ffill()
    max_med = raw_momentum.rolling(window=period, min_periods=max(5, period // 3)).max().bfill().ffill()
    rng = (max_med - min_med).replace(0, np.nan)

    temp = pd.Series(0.0, index=df.index)
    valid = rng.notna()
    temp.loc[valid] = (raw_momentum.loc[valid] - min_med.loc[valid]) / rng.loc[valid]
    temp = temp.clip(-1, 1).fillna(0)

    value = pd.Series(0.0, index=df.index)
    for i in range(1, n):
        prev = value.iloc[i - 1]
        val = (temp.iloc[i] - 0.5 + 0.5 * prev)
        val = max(min(val, 0.9999), -0.9999)
        value.iloc[i] = val

    temp2 = (1 + value) / (1 - value)
    temp2 = temp2.replace([np.inf, -np.inf], np.nan).clip(lower=1e-6).fillna(1e-6)
    momentum = 0.25 * np.log(temp2)
    momentum = pd.Series(momentum, index=df.index).replace([np.inf, -np.inf], 0).fillna(0)
    hist = pd.Series(0.0, index=df.index)
    for i in range(1, n):
        hist.iloc[i] = momentum.iloc[i] + 0.5 * hist.iloc[i - 1]
    return hist.replace([np.inf, -np.inf], 0).fillna(0)

def calculate_vwap_daily_reset(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    df2 = df.copy()
    df2['datetime'] = pd.to_datetime(df2['timestamp'], unit='s', utc=True)
    df2['date'] = df2['datetime'].dt.date
    hlc3 = (df2['high'] + df2['low'] + df2['close']) / 3.0
    df2['hlc3_vol'] = hlc3 * df2['volume']
    df2['cum_vol'] = df2.groupby('date')['volume'].cumsum()
    df2['cum_hlc3_vol'] = df2.groupby('date')['hlc3_vol'].cumsum()
    vwap = df2['cum_hlc3_vol'] / df2['cum_vol'].replace(0, np.nan)
    return vwap.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)

# -------------------------
# PIVOT CALCULATION
# -------------------------
def calculate_fibonacci_pivots(df_daily: pd.DataFrame):
    if df_daily is None or len(df_daily) < 2:
        return None
    
    # Get the data for yesterday's candle (second to last)
    prev_day = df_daily.iloc[-2]
    high = float(prev_day['high'])
    low = float(prev_day['low'])
    close = float(prev_day['close'])

    pivot = (high + low + close) / 3
    
    # Determine which way the market moved yesterday
    if close < pivot: # Bearish close (Pivot is closer to High)
        R3 = pivot + (high - low)
        R2 = pivot + 0.618 * (high - low)
        R1 = pivot + 0.382 * (high - low)
        S1 = pivot - 0.382 * (high - low)
        S2 = pivot - 0.618 * (high - low)
        S3 = pivot - (high - low)
    elif close > pivot: # Bullish close (Pivot is closer to Low)
        R3 = pivot + (high - low)
        R2 = pivot + 0.618 * (high - low)
        R1 = pivot + 0.382 * (high - low)
        S1 = pivot - 0.382 * (high - low)
        S2 = pivot - 0.618 * (high - low)
        S3 = pivot - (high - low)
    else: # Neutral close
        R3 = pivot + (high - low)
        R2 = pivot + 0.618 * (high - low)
        R1 = pivot + 0.382 * (high - low)
        S1 = pivot - 0.382 * (high - low)
        S2 = pivot - 0.618 * (high - low)
        S3 = pivot - (high - low)

    return {
        'P': pivot, 'R1': R1, 'R2': R2, 'R3': R3,
        'S1': S1, 'S2': S2, 'S3': S3,
        'high': high, 'low': low
    }

def get_crossover_line(pivots: Dict[str, float], prev_price: float, curr_price: float, direction: str) -> Optional[Tuple[str, float]]:
    if not pivots:
        return None
    
    levels = ["R3", "R2", "R1", "P", "S1", "S2", "S3"]
    
    for level_name in levels:
        line = pivots.get(level_name)
        if line is None:
            continue
        
        # Check for crossover
        if direction == "long":
            if prev_price <= line and curr_price > line:
                return level_name, line
        elif direction == "short":
            if prev_price >= line and curr_price < line:
                return level_name, line
    
    return None

# -------------------------
# STATE MANAGEMENT
# -------------------------
def should_suppress_duplicate(last_state: Optional[Dict[str, Any]], current_signal: str, suppress_secs: int) -> bool:
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
    if upw.iloc[idx] and not dnw.iloc[idx]:
        return "bullish"
    elif dnw.iloc[idx] and not upw.iloc[idx]:
        return "bearish"
    else:
        return "neutral"

# -------------------------
# STRUCTURED LOGGING
# -------------------------
class StructuredLogger:
    @staticmethod
    def log_pair_analysis(pair_name: str, indicators: Dict[str, Any], decision: str):
        """Log structured analysis for better debugging"""
        logger.info(json.dumps({
            "type": "pair_analysis",
            "pair": pair_name,
            "timestamp": now_ts(),
            "indicators": indicators,
            "decision": decision,
            "execution_id": os.getenv("GITHUB_RUN_ID", "local")
        }))

# -------------------------
# CORE PAIR EVALUATION
# -------------------------
async def evaluate_pair_async(
    session: aiohttp.ClientSession,
    fetcher: RateLimitedDataFetcher,
    products_map: Dict[str, dict],
    pair_name: str,
    last_state: Optional[Dict[str, Any]]
) -> Optional[Tuple[str, Dict[str, Any]]]:
    METRICS["pairs_checked"] += 1
    try:
        prod = products_map.get(pair_name)
        if not prod:
            logger.debug(f"No product mapping for {pair_name}")
            return None
            
        sp = cfg.SPECIAL_PAIRS.get(pair_name, {})
        limit_15m = sp.get("limit_15m", 250)
        min_required_15m = sp.get("min_required", 150)
        limit_5m = sp.get("limit_5m", 500)
        min_required_5m = sp.get("min_required_5m", 250)
        
        # Initialize vwap_curr to None to prevent UnboundLocalError
        vwap_curr: Optional[float] = None

        tasks = [fetcher.fetch_candles(session, prod['symbol'], "15", limit_15m)]
        if cfg.USE_RMA200:
            tasks.append(fetcher.fetch_candles(session, prod['symbol'], "5", limit_5m))
            
        results = await asyncio.gather(*tasks, return_exceptions=True)

        res15 = results[0] if not isinstance(results[0], Exception) else None
        res5 = results[1] if cfg.USE_RMA200 and len(results) > 1 and not isinstance(results[1], Exception) else None

        if isinstance(results[0], Exception):
            logger.error(f"Exception fetching 15m data for {pair_name}: {results[0]}")
            METRICS["network_errors"] += 1
            
        df_15m = parse_candle_response(res15)
        df_5m = parse_candle_response(res5) if cfg.USE_RMA200 else None

        last_i_15m = -2
        last_i_5m = -2
        
        if df_15m is None or len(df_15m) < min_required_15m:
            logger.debug(f"{pair_name}: Insufficient 15m data (got {len(df_15m) if df_15m is not None else 0}, need {min_required_15m})")
            return None
            
        if cfg.USE_RMA200 and (df_5m is None or len(df_5m) < min_required_5m):
            logger.debug(f"{pair_name}: Insufficient 5m data (got {len(df_5m) if df_5m is not None else 0}, need {min_required_5m})")
            return None
            
        # Fetch daily data for pivots
        daily_res = await fetcher.fetch_candles(session, prod['symbol'], "D", cfg.PIVOT_LOOKBACK_PERIOD + 1)
        df_daily = parse_candle_response(daily_res)
        
        pivots = calculate_fibonacci_pivots(df_daily)
        
        if pivots is None or df_daily is None or len(df_daily) < 2:
            logger.debug(f"{pair_name}: Insufficient daily data for pivots.")
            pivots_available = False
        else:
            pivots_available = True
            
        # Calculate Indicators
        ppo, ppo_signal = calculate_ppo(df_15m, cfg.PPO_FAST, cfg.PPO_SLOW, cfg.PPO_SIGNAL, cfg.PPO_USE_SMA)
        upw, dnw, filtx1, filtx12 = calculate_cirrus_cloud(df_15m)
        magical_hist = calculate_magical_momentum_hist(df_15m)
        
        # VWAP Calculation (15m)
        vwap_15m: Optional[pd.Series] = None
        if df_daily is not None and len(df_daily) >= 2:
            # VWAP reset daily is calculated based on 15m candles
            vwap_15m = calculate_vwap_daily_reset(df_15m)
            try:
                # Use second-to-last candle for trigger logic
                vwap_curr = float(vwap_15m.iloc[last_i_15m])
            except Exception:
                vwap_curr = None
                
        # Extract values from 15m
        close_c = float(df_15m['close'].iloc[last_i_15m])
        open_c = float(df_15m['open'].iloc[last_i_15m])
        high_c = float(df_15m['high'].iloc[last_i_15m])
        low_c = float(df_15m['low'].iloc[last_i_15m])
        ppo_curr = float(ppo.iloc[last_i_15m])
        magical_curr = float(magical_hist.iloc[last_i_15m])
        
        # Calculate RMA50 for 15m
        rma50 = calculate_rma(df_15m['close'].astype(float), 50)
        rma50_curr = float(rma50.iloc[last_i_15m])

        # RMA 200 (5m) check
        if cfg.USE_RMA200 and df_5m is not None:
            rma200 = calculate_rma(df_5m['close'].astype(float), 200)
            try:
                rma200_5m_curr = float(rma200.iloc[last_i_5m])
                rma_200_available = not np.isnan(rma200_5m_curr)
            except Exception:
                rma_200_available = False
        else:
            rma_200_available = False

        cloud_state = cloud_state_from(upw, dnw, idx=last_i_15m)
        
        total_range = high_c - low_c
        upper_wick = high_c - max(open_c, close_c)
        lower_wick = min(open_c, close_c) - low_c
        wick_ratio = 0.35
        upper_wick_ok = upper_wick / total_range < wick_ratio if total_range > 0 else True
        lower_wick_ok = lower_wick / total_range < wick_ratio if total_range > 0 else True
        
        is_green = close_c > open_c
        is_red = close_c < open_c

        # Structured logging for analysis
        if cfg.DEBUG_MODE:
            indicators = {
                "close": round(close_c, 2),
                "open": round(open_c, 2),
                "ppo": round(ppo_curr, 2),
                "rma50": round(rma50_curr, 2),
                "magical_momentum": round(magical_curr, 4),
                "cloud_state": cloud_state,
                "vwap": round(vwap_curr, 2) if vwap_curr else None,
                "rma200": round(rma200_5m_curr, 2) if rma_200_available else None
            }
            StructuredLogger.log_pair_analysis(pair_name, indicators, "evaluating")

        # Extra debug output for indicator values
        if cfg.DEBUG_MODE:
            vwap_log_str = f"{vwap_curr:.2f}" if vwap_curr is not None and not np.isnan(vwap_curr) else "nan"
            
            logger.debug(
                f"{pair_name}: close={close_c:.2f}, open={open_c:.2f}, "
                f"PPO={ppo_curr:.2f}, RMA50={rma50_curr:.2f}, "
                f"MMH={magical_curr:.4f}, Cloud={cloud_state}, "
                f"VWAP={vwap_log_str}" 
            )

        # Reason for skip summary
        if cfg.DEBUG_MODE:
            reasons = []
            if cloud_state == "neutral": reasons.append("cloud neutral")
            if magical_curr <= 0 and is_green: reasons.append("MMH not positive for long")
            if magical_curr >= 0 and is_red: reasons.append("MMH not negative for short")
            if not upper_wick_ok and is_green: reasons.append("upper wick too long")
            if not lower_wick_ok and is_red: reasons.append("lower wick too long")
            if vwap_curr is not None and len(df_15m) >= abs(last_i_15m) + 2:
                prev_close = float(df_15m['close'].iloc[last_i_15m - 1])
                prev_vwap = float(vwap_15m.iloc[last_i_15m - 1])
                if prev_close <= prev_vwap and close_c <= vwap_curr: reasons.append("no VWAP crossover for long")
                if prev_close >= prev_vwap and close_c >= vwap_curr: reasons.append("no VWAP crossover for short")

            # RMA200 check reasons
            if cfg.USE_RMA200 and rma_200_available:
                if close_c <= rma200_5m_curr and is_green: reasons.append("15m close below 5m RMA200 for long")
                if close_c >= rma200_5m_curr and is_red: reasons.append("15m close above 5m RMA200 for short")
                
            if not reasons: reasons.append("conditions not met")
            logger.debug(f"{pair_name}: skipped because {', '.join(reasons)}")

        # RMA200 check: 5m close must be above 200 RMA for long, below for short
        rma_long_ok = not cfg.USE_RMA200 or (rma_200_available and close_c > rma200_5m_curr)
        rma_short_ok = not cfg.USE_RMA200 or (rma_200_available and close_c < rma200_5m_curr)
        
        # Base requirements for signal
        base_long_ok = (
            is_green and
            cloud_state == "bullish" and
            ppo_curr > 0 and
            magical_curr > 0 and
            upper_wick_ok and
            rma_long_ok and
            close_c > rma50_curr
        )
        
        base_short_ok = (
            is_red and
            cloud_state == "bearish" and
            ppo_curr < 0 and
            magical_curr < 0 and
            lower_wick_ok and
            rma_short_ok and
            close_c < rma50_curr
        )

        long_crossover = get_crossover_line(pivots, float(df_15m['close'].iloc[last_i_15m - 1]), close_c, "long")
        short_crossover = get_crossover_line(pivots, float(df_15m['close'].iloc[last_i_15m - 1]), close_c, "short")
        
        long_crossover_name = long_crossover[0] if long_crossover else None
        long_crossover_line = long_crossover[1] if long_crossover else None
        short_crossover_name = short_crossover[0] if short_crossover else None
        short_crossover_line = short_crossover[1] if short_crossover else None

        # VWAP Crossover Signal
        vbuy = False
        vsell = False
        if vwap_curr is not None and len(df_15m) >= abs(last_i_15m) + 2:
            prev_close = float(df_15m['close'].iloc[last_i_15m - 1])
            prev_vwap = float(vwap_15m.iloc[last_i_15m - 1])
            if base_long_ok and prev_close <= prev_vwap and close_c > vwap_curr:
                vbuy = True
            if base_short_ok and prev_close >= prev_vwap and close_c < vwap_curr:
                vsell = True

        fib_long = base_long_ok and (long_crossover_line is not None)
        fib_short = base_short_ok and (short_crossover_line is not None)
        
        up_sig = "🟩▲"
        down_sig = "🟥🔻"
        current_signal = None
        message = None
        suppress_secs = cfg.DUPLICATE_SUPPRESSION_SECONDS

        if vbuy:
            current_signal = "vbuy"
            if not should_suppress_duplicate(last_state, current_signal, suppress_secs):
                vwap_str = f"{vwap_curr:,.2f}"
                ppo_str = f"{ppo_curr:.2f}"
                price_str = f"{close_c:,.2f}"
                message = (
                    f"{up_sig} {sanitize_for_telegram(pair_name)} - VBuy\n"
                    f"Closed Above VWAP (${vwap_str})\n"
                    f"PPO 15m: {ppo_str}\n"
                    f"Price: ${price_str}\n"
                    f"{human_ts()}"
                )
        elif vsell:
            current_signal = "vsell"
            if not should_suppress_duplicate(last_state, current_signal, suppress_secs):
                vwap_str = f"{vwap_curr:,.2f}"
                ppo_str = f"{ppo_curr:.2f}"
                price_str = f"{close_c:,.2f}"
                message = (
                    f"{down_sig} {sanitize_for_telegram(pair_name)} - VSell\n"
                    f"Closed Below VWAP (${vwap_str})\n"
                    f"PPO 15m: {ppo_str}\n"
                    f"Price: ${price_str}\n"
                    f"{human_ts()}"
                )
        elif fib_long:
            current_signal = f"fib_long_{long_crossover_name}"
            if not should_suppress_duplicate(last_state, current_signal, suppress_secs):
                line_str = f"{long_crossover_line:,.2f}"
                ppo_str = f"{ppo_curr:.2f}"
                price_str = f"{close_c:,.2f}"
                message = (
                    f"{up_sig} {sanitize_for_telegram(pair_name)} - Fib Long (Cross {long_crossover_name})\n"
                    f"Price Crossed Above {long_crossover_name} (${line_str})\n"
                    f"PPO 15m: {ppo_str}\n"
                    f"Price: ${price_str}\n"
                    f"{human_ts()}"
                )
        elif fib_short:
            current_signal = f"fib_short_{short_crossover_name}"
            if not should_suppress_duplicate(last_state, current_signal, suppress_secs):
                line_str = f"{short_crossover_line:,.2f}"
                ppo_str = f"{ppo_curr:.2f}"
                price_str = f"{close_c:,.2f}"
                message = (
                    f"{down_sig} {sanitize_for_telegram(pair_name)} - Fib Short (Cross {short_crossover_name})\n"
                    f"Price Crossed Below {short_crossover_name} (${line_str})\n"
                    f"PPO 15m: {ppo_str}\n"
                    f"Price: ${price_str}\n"
                    f"{human_ts()}"
                )

        if message:
            logger.info(f"Generated alert for {pair_name}: {current_signal}")
            ok = await send_telegram_with_retries(cfg.TELEGRAM_BOT_TOKEN, cfg.TELEGRAM_CHAT_ID, message, session)
            if ok:
                METRICS["alerts_sent"] += 1
                return current_signal, {"message": message, "signal": current_signal}
            else:
                # If Telegram send fails, don't update state to allow retry
                return None, None
        
        # Update state only if a non-duplicate signal was found and sent, or if state needs to be cleared
        # If no signal, explicitly check if the last state signal is stale and clear it
        if last_state and current_signal is None and now_ts() - int(last_state.get("ts", 0)) > suppress_secs:
            logger.debug(f"{pair_name}: Last state '{last_state.get('state')}' expired. Clearing state.")
            return None, None # Clear state in DB by returning None
            
        return last_state.get("state") if last_state else None, None

    except ClientConnectorError as e:
        logger.error(f"Network error evaluating {pair_name}: {e}")
        METRICS["network_errors"] += 1
        return None, None
    except Exception as e:
        logger.error(f"Error evaluating {pair_name}: {e}")
        logger.debug(traceback.format_exc())
        METRICS["logic_errors"] += 1
        return None, None

# -------------------------
# TELEGRAM NOTIFICATIONS
# -------------------------
async def send_telegram_async(token: str, chat_id: str, text: str, session: Optional[aiohttp.ClientSession] = None) -> bool:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": "true"
    }
    
    own_session = False
    if session is None:
        connector = TCPConnector(limit=cfg.MAX_CONCURRENCY, ssl=False)
        session = aiohttp.ClientSession(connector=connector)
        own_session = True

    try:
        async with session.post(url, data=data, timeout=10) as resp:
            try:
                js = await resp.json(content_type=None)
            except Exception:
                js = {"ok": False, "status": resp.status, "text": await resp.text()}
            
            ok = js.get("ok", False)
            if ok:
                await asyncio.sleep(0.2)
                return True
            else:
                logger.warning(f"Telegram API error: {js}")
                return False
    except Exception as e:
        logger.exception(f"Telegram send failed: {e}")
        return False
    finally:
        if own_session:
            await session.close()

async def send_telegram_with_retries(token: str, chat_id: str, text: str, session: Optional[aiohttp.ClientSession] = None) -> bool:
    last_exc = None
    for attempt in range(1, max(1, cfg.TELEGRAM_RETRIES) + 1):
        try:
            ok = await send_telegram_async(token, chat_id, text, session)
            if ok:
                return True
            last_exc = Exception("Telegram returned non-ok")
        except Exception as e:
            last_exc = e
        sleep_for = cfg.TELEGRAM_BACKOFF_BASE ** (attempt - 1)
        await asyncio.sleep(sleep_for + random.uniform(0, 0.3))

    logger.error(f"Telegram send failed after retries: {last_exc}")
    return False

# -------------------------
# DEAD MAN'S SWITCH
# -------------------------
async def check_dead_mans_switch(state_db: StateDB):
    try:
        last_success = state_db.get_metadata("last_success_run")
        if last_success:
            try:
                last_success_int = int(last_success)
            except Exception:
                last_success_int = now_ts()
                
            hours_since = (now_ts() - last_success_int) / 3600.0
            
            if hours_since > 2:
                dead_alert_ts = state_db.get_metadata("dead_alert_ts")
                if dead_alert_ts:
                    try:
                        dead_alert_int = int(dead_alert_ts)
                    except Exception:
                        dead_alert_int = 0
                else:
                    dead_alert_int = 0
                
                if now_ts() - dead_alert_int > 4 * 3600:
                    msg = (
                        f"🚨 DEAD MAN'S SWITCH: Bot hasn't succeeded in {hours_since:.1f} hours!\n"
                        f"Last success: {datetime.fromtimestamp(last_success_int).strftime('%Y-%m-%d %H:%M:%S IST')}"
                    )
                    await send_telegram_with_retries(cfg.TELEGRAM_BOT_TOKEN, cfg.TELEGRAM_CHAT_ID, msg)
                    state_db.set_metadata("dead_alert_ts", str(now_ts()))
                    
    except Exception as e:
        logger.exception(f"Dead man's switch failed: {e}")

# -------------------------
# HEALTH CHECK
# -------------------------
async def health_check(state_db: Optional[StateDB] = None) -> Dict[str, Any]:
    """Simple health check that can be called externally"""
    health_status = {
        "status": "healthy",
        "timestamp": now_ts(),
        "memory_usage": psutil.Process().memory_info().rss,
        "last_success": state_db.get_metadata("last_success_run") if state_db else None,
        "metrics": METRICS.copy()
    }
    return health_status

# -------------------------
# CONFIGURATION VALIDATION
# -------------------------
def validate_configuration() -> bool:
    """Validate critical configuration before starting"""
    issues = []
    
    # Check required pairs
    if not cfg.PAIRS:
        issues.append("No trading pairs configured")
    
    # Check API endpoints are reachable
    try:
        import urllib.request
        urllib.request.urlopen(f"{cfg.DELTA_API_BASE}/v2/products", timeout=5)
    except Exception as e:
        issues.append(f"Delta API endpoint not reachable: {e}")
    
    # Check database is writable
    try:
        test_db_path = "/tmp/test_fib.db"
        test_db = StateDB(test_db_path)
        test_db.close()
        if os.path.exists(test_db_path):
            os.unlink(test_db_path)
    except Exception as e:
        issues.append(f"Database not writable: {e}")
    
    # Check Telegram credentials
    if cfg.TELEGRAM_BOT_TOKEN == "xxxx" or cfg.TELEGRAM_CHAT_ID == "xxxx":
        issues.append("Telegram credentials not configured")
    
    if issues:
        logger.error(f"Configuration issues: {issues}")
        return False
    
    logger.info("✅ Configuration validation passed")
    return True

# -------------------------
# GRACEFUL SHUTDOWN
# -------------------------
async def graceful_shutdown():
    """Wait for current operations to complete"""
    logger.info("Shutting down gracefully...")
    # Add any cleanup needed
    await asyncio.sleep(1)

# -------------------------
# CORE RUN LOGIC
# -------------------------
stop_requested = False

def request_stop(signum, frame):
    global stop_requested
    stop_requested = True
    logger.warning(f"Stop requested (signal {signum})")

async def run_once(send_test: bool = True):
    global stop_requested
    reset_metrics()
    start_time = time.time()
    
    async def check_timeout():
        while True:
            elapsed = time.time() - start_time
            if elapsed > cfg.MAX_EXEC_TIME:
                logger.error(f"Execution exceeded MAX_EXEC_TIME ({cfg.MAX_EXEC_TIME}s). Forcing exit.")
                raise SystemExit(EXIT_TIMEOUT)
            await asyncio.sleep(1)

    timeout_task = asyncio.create_task(check_timeout())
    
    try:
        # DB and State initialization
        prune_old_state_records(cfg.STATE_DB_PATH, cfg.STATE_EXPIRY_DAYS, logger)
        state_db = StateDB(cfg.STATE_DB_PATH)
        await check_dead_mans_switch(state_db)
        last_alerts = state_db.load_all()

        if cfg.RESET_STATE:
            logger.warning("🔄 RESET_STATE requested: clearing states table")
            for p in cfg.PAIRS:
                state_db.set(p, None)

        if send_test and cfg.SEND_TEST_MESSAGE:
            test_msg = (
                f"🔔 Fibonacci Pivot Bot started\n"
                f"Time: {human_ts()}\n"
                f"Debug: {'ON' if cfg.DEBUG_MODE else 'OFF'}\n"
                f"Pairs: {len(cfg.PAIRS)}"
            )
            await send_telegram_with_retries(cfg.TELEGRAM_BOT_TOKEN, cfg.TELEGRAM_CHAT_ID, test_msg)
            
        products_map = load_products_cache()
        circuit = EnhancedCircuitBreaker()
        fetcher_local = RateLimitedDataFetcher(
            cfg.DELTA_API_BASE,
            max_parallel=cfg.MAX_CONCURRENCY,
            timeout=cfg.HTTP_TIMEOUT,
            circuit_breaker=circuit,
            requests_per_minute=cfg.REQUESTS_PER_MINUTE
        )
        
        connector = TCPConnector(limit=cfg.MAX_CONCURRENCY, ssl=False)
        async with aiohttp.ClientSession(connector=connector) as session:
            
            if products_map is None:
                prod_resp = await fetcher_local.fetch_products(session)
                if not prod_resp:
                    logger.error("❌ Failed to fetch products from API")
                    state_db.close()
                    raise SystemExit(EXIT_API_FAILURE)
                products_map = build_products_map(prod_resp)
                save_products_cache(products_map)

            tasks = [
                evaluate_pair_async(
                    session,
                    fetcher_local,
                    products_map,
                    pair_name,
                    last_alerts.get(pair_name)
                ) for pair_name in cfg.PAIRS
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
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
                
                if new_state is None and last_alerts.get(pair_name):
                    # Only clear state if evaluate_pair_async explicitly signals it (e.g., expiry)
                    if result[0] is None and result[1] is None:
                         state_db.set(pair_name, None)
                
                if alert_info:
                    alerts_sent += 1

            METRICS["alerts_sent"] = alerts_sent
            
        state_db.set_metadata("last_success_run", str(now_ts()))
        state_db.set_metadata("dead_alert_ts", "0")

    except SystemExit:
        raise
    except Exception as e:
        logger.exception("Unhandled exception in run_once")
        raise SystemExit(EXIT_API_FAILURE)
    finally:
        timeout_task.cancel()
        if 'state_db' in locals():
            state_db.close()
        if 'fetcher_local' in locals():
            if hasattr(fetcher_local, 'cache'):
                fetcher_local.cache.clear()
        if 'session' in locals():
            await session.close()
            
        await asyncio.sleep(0.1)  # Allow pending tasks to complete
            
        METRICS["execution_time"] = time.time() - start_time
        
        # Performance monitoring
        execution_time = METRICS["execution_time"]
        if execution_time > cfg.MAX_EXEC_TIME * 0.8:
            logger.warning(f"Execution time ({execution_time:.2f}s) approaching limit")
        
        logger.info(
            f"✅ Run finished. Pairs: {METRICS['pairs_checked']}, "
            f"Alerts: {METRICS['alerts_sent']}, Time: {execution_time:.2f}s"
        )
        
        if cfg.DEBUG_MODE:
            logger.debug(f"Metrics: {METRICS}")


# -------------------------
# MAIN EXECUTION
# -------------------------
def main():
    # Environment detection
    cron_env = is_cron_environment()
    run_mode = "ONCE" if cron_env else "LOOP"
    
    # Enhanced startup logging
    logger.info(f"🚀 Starting Fibonacci Bot - Mode: {run_mode}")
    logger.info(f"🐍 Python: {sys.version}")
    logger.info(f"💾 Memory available: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    
    # Validate configuration first
    if not validate_configuration():
        sys.exit(EXIT_CONFIG_ERROR)
    
    # Set memory limits
    try:
        memory_limit = min(cfg.MEMORY_LIMIT_BYTES, int(psutil.virtual_memory().available * 0.8))
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
        logger.info(f"🧠 Memory limit set to: {memory_limit / (1024**3):.2f} GB")
    except (ImportError, ValueError, OSError) as e:
        logger.debug(f"Could not set memory limits: {e}")

    # Process locking to prevent concurrent runs
    lock = ProcessLock("fib_pivot.lock", timeout=cfg.RUN_LOOP_INTERVAL * 2)
    if not lock.acquire():
        logger.error("❌ Failed to acquire process lock. Another instance is running.")
        sys.exit(EXIT_LOCK_CONFLICT)

    # Signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    
    try:
        if run_mode == "LOOP":
            # --- Original Looping Logic (for local dev) ---
            interval = cfg.RUN_LOOP_INTERVAL
            logger.info(f"🔄 Loop mode: interval={interval}s")
            while not stop_requested:
                loop_start = time.time()
                try:
                    asyncio.run(run_once(send_test=False if loop_start != METRICS.get('start_time') else True))
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
                if to_sleep > 0:
                    for _ in range(int(to_sleep)):
                        if stop_requested:
                            break
                        time.sleep(1)
                    if not stop_requested:
                        remainder = to_sleep - int(to_sleep)
                        if remainder > 0:
                            time.sleep(remainder)

            sys.exit(EXIT_SUCCESS)
            
        else:
            # --- Single Run Logic (for GitHub Actions / Cron) ---
            logger.info("🚀 Single run mode (exiting immediately after completion).")
            asyncio.run(run_once())
            sys.exit(EXIT_SUCCESS) # Ensure explicit exit

    except SystemExit as e:
        logger.error(f"Exited with code {e.code}")
        sys.exit(e.code if isinstance(e.code, int) else 1)
    except Exception:
        logger.exception("Unhandled exception in main execution block")
        sys.exit(EXIT_API_FAILURE)
    finally:
        lock.release()
        try:
            asyncio.run(graceful_shutdown())
        except Exception:
            pass

if __name__ == "__main__":
    main()
