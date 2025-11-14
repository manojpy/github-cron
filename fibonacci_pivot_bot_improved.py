#!/usr/bin/env python3
# python 3.12+
"""
Improved Fibonacci Pivot Bot - Cleaned & Optimized Version
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
    from pydantic import BaseModel, field_validator
except Exception as e:
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
    SPECIAL_PAIRS: Dict[str, Dict[str, int]] = {"SOLUSD": {"limit_15m": 210,"min_required": 180,"limit_5m": 300,"min_required_5m": 200}}
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
# DEFAULT CONFIG DEFINITION
# -------------------------
DEFAULT_CONFIG = {
    "TELEGRAM_BOT_TOKEN": "8462496498:AAHYZ4xDIHvrVRjmCmZyoPhupCjRaRgiITc",
    "TELEGRAM_CHAT_ID": "203813932",
    "DEBUG_MODE": False,
    "SEND_TEST_MESSAGE": True,
    "RESET_STATE": False,
    "DELTA_API_BASE": "https://api.india.delta.exchange",
    "PAIRS": ["BTCUSD", "ETHUSD", "SOLUSD", "AVAXUSD", "BCHUSD", "XRPUSD", "BNBUSD", "LTCUSD", "DOTUSD", "ADAUSD", "SUIUSD", "AAVEUSD"],
    "SPECIAL_PAIRS": {"SOLUSD": {"limit_15m": 210,"min_required": 180,"limit_5m": 300,"min_required_5m": 200}},
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
# CONFIGURATION LOADER
# -------------------------
def str_to_bool(value: str) -> bool:
    """Safely interpret string-based booleans."""
    return str(value).strip().lower() in ("true", "1", "yes", "y", "t")

def load_config() -> Config:
    """Load JSON config file and merge with environment overrides."""
    base = DEFAULT_CONFIG.copy()
    config_file = os.getenv("CONFIG_FILE", "config_fib.json")
    
    # Load JSON config file if present
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

    # Merge environment variables with proper type casting
    def override(key: str, default: Any = None, cast: Any = None) -> None:
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
    override("MAX_PARALLEL_FETCH", 8, int)
    override("HTTP_TIMEOUT", 15, int)
    override("MAX_CONCURRENCY", 6, int)
    override("MAX_EXEC_TIME", 25, int)
    override("DEADMAN_HOURS", 2, int)

    print(f"DEBUG_MODE={base['DEBUG_MODE']}, SEND_TEST_MESSAGE={base['SEND_TEST_MESSAGE']}")
    return Config(**base)

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

def reset_metrics() -> None:
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
def setup_logger(cfg: Config) -> logging.Logger:
    """Setup and configure logger."""
    logger = logging.getLogger("fibonacci_pivot_bot")
    log_level = getattr(logging, cfg.LOG_LEVEL if isinstance(cfg.LOG_LEVEL, str) else "INFO", logging.INFO)
    logger.setLevel(logging.DEBUG if cfg.DEBUG_MODE else log_level)

    class JSONFormatter(logging.Formatter):
        def format(self, record):
            log_obj = {
                "time": self.formatTime(record),
                "level": record.levelname,
                "msg": record.getMessage()
            }
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
    
    return logger

# Load configuration and setup logger
cfg = load_config()
logger = setup_logger(cfg)

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

    def release(self) -> None:
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
# CIRCUIT BREAKER
# -------------------------
class CircuitBreaker:
    """Prevent cascading failures when API is down"""
    def __init__(self, failure_threshold: int = 5, timeout: int = 300):
        self.failures = 0
        self.last_failure = 0.0
        self.threshold = failure_threshold
        self.timeout = timeout
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
        except Exception as e:
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
    return int(time.time())

def human_ts() -> str:
    tz = pytz.timezone("Asia/Kolkata")
    return datetime.now(tz).strftime("%d-%m-%Y @ %H:%M IST")

def sanitize_for_telegram(text: Union[str, float, int]) -> str:
    """Prevent Telegram HTML injection"""
    return html.escape(str(text))

# -------------------------
# SQLITE STATE DB (WAL mode)
# -------------------------
class StateDB:
    def __init__(self, db_path: str):
        self.db_path = db_path
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False, isolation_level=None)
        # Use WAL for concurrency
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        except Exception:
            pass
        self._ensure_tables()
        try:
            os.chmod(self.db_path, 0o600)
        except Exception:
            pass

    def _ensure_tables(self) -> None:
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
        cur = self._conn.cursor()
        cur.execute("SELECT pair, state, ts FROM states")
        rows = cur.fetchall()
        return {r[0]: {"state": r[1], "ts": int(r[2] or 0)} for r in rows}

    def get(self, pair: str) -> Optional[Dict[str, Any]]:
        cur = self._conn.cursor()
        cur.execute("SELECT state, ts FROM states WHERE pair = ?", (pair,))
        r = cur.fetchone()
        if not r:
            return None
        return {"state": r[0], "ts": int(r[1] or 0)}

    def set(self, pair: str, state: Optional[str], ts: Optional[int] = None) -> None:
        ts = int(ts or now_ts())
        cur = self._conn.cursor()
        if state is None:
            cur.execute("DELETE FROM states WHERE pair = ?", (pair,))
        else:
            cur.execute("INSERT INTO states(pair, state, ts) VALUES (?, ?, ?) "
                        "ON CONFLICT(pair) DO UPDATE SET state=excluded.state, ts=excluded.ts",
                        (pair, state, ts))
        self._conn.commit()

    def get_metadata(self, key: str) -> Optional[str]:
        cur = self._conn.cursor()
        cur.execute("SELECT value FROM metadata WHERE key = ?", (key,))
        r = cur.fetchone()
        return r[0] if r else None

    def set_metadata(self, key: str, value: str) -> None:
        cur = self._conn.cursor()
        cur.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)", (key, value))
        self._conn.commit()

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

# -------------------------
# PRUNE (smart daily)
# -------------------------
def prune_old_state_records(db_path: str, expiry_days: int = 30, logger_local: logging.Logger = None) -> None:
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

        cur.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES ('last_prune', ?)",
                    (datetime.utcnow().isoformat(),))
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
    circuit_breaker: Optional[CircuitBreaker] = None
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
# DATA FETCHER
# -------------------------
class DataFetcher:
    def __init__(self, base_url: str, max_parallel: int = 6, timeout: int = 15, circuit_breaker: Optional[CircuitBreaker] = None):
        self.base_url = base_url.rstrip("/")
        self.semaphore = asyncio.Semaphore(max_parallel)
        self.timeout = timeout
        self.cache: Dict[str, Tuple[float, Any]] = {}
        self.circuit_breaker = circuit_breaker

    async def fetch_products(self, session: aiohttp.ClientSession):
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
        key = f"candles:{symbol}:{resolution}:{limit}"
        if key in self.cache:
            age, data = self.cache[key]
            # keep cached for a short time (like 5 sec) to reduce duplicate requests in same run
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

def save_products_cache(products_map: Dict[str, dict]) -> None:
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
    # support some API shapes
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

def calculate_magical_momentum_hist(df: pd.DataFrame, period: int = 144, responsiveness: float = 0.9) -> pd.Series:
    n = len(df)
    if n == 0:
        return pd.Series([], dtype=float)
    if n < period + 50:
        return pd.Series(np.zeros(n), index=df.index, dtype=float)
    
    close = df['close'].astype(float).copy()
    sd = close.rolling(window=50, min_periods=10).std() * responsiveness
    sd = sd.bfill().ffill().fillna(0.001).clip(lower=1e-6)
    
    worm = close.copy()
    for i in range(1, n):
        diff = close.iloc[i] - worm.iloc[i - 1]
        delta = np.sign(diff) * sd.iloc[i] if abs(diff) > sd.iloc[i] else diff
        worm.iloc[i] = worm.iloc[i - 1] + delta
    
    ma = close.rolling(window=period, min_periods=max(5, period // 3)).mean().bfill().ffill()
    denom = worm.replace(0, np.nan).bfill().ffill().clip(lower=1e-8)
    
    raw_momentum = ((worm - ma).fillna(0)) / denom
    raw_momentum = raw_momentum.replace([np.inf, -np.inf], 0).fillna(0)
    
    min_med = raw_momentum.rolling(window=period, min_periods=max(5, period // 3)).min().bfill().ffill()
    max_med = raw_momentum.rolling(window=period, min_periods=max(5, period // 3)).max().bfill().ffill()
    rng = (max_med - min_med).replace(0, np.nan).fillna(1e-8)
    
    temp = pd.Series(0.0, index=df.index)
    valid = rng > 1e-10
    temp.loc[valid] = (raw_momentum.loc[valid] - min_med.loc[valid]) / rng.loc[valid]
    temp = temp.clip(-1, 1).fillna(0)
    
    value = pd.Series(0.0, index=df.index)
    for i in range(1, n):
        v_prev = value.iloc[i - 1]
        v_new = (temp.iloc[i] - 0.5 + 0.5 * v_prev)
        value.iloc[i] = max(-0.9999, min(0.9999, v_new))
    
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
    return vwap.ffill().bfill()

def calculate_fibonacci_pivots(high: float, low: float, close: float) -> Dict[str, float]:
    """Classic Fibonacci Pivots using previous day's OHLC"""
    pivot = (high + low + close) / 3.0
    r1 = pivot + (high - low) * 0.382
    r2 = pivot + (high - low) * 0.618
    r3 = pivot + (high - low) * 1.000
    s1 = pivot - (high - low) * 0.382
    s2 = pivot - (high - low) * 0.618
    s3 = pivot - (high - low) * 1.000
    return {'P': pivot, 'R1': r1, 'R2': r2, 'R3': r3, 'S1': s1, 'S2': s2, 'S3': s3}

def cloud_state_from(upw: pd.Series, dnw: pd.Series, idx: int = -1) -> str:
    u = bool(upw.iloc[idx])
    d = bool(dnw.iloc[idx])
    if u and not d:
        return "green"
    elif d and not u:
        return "red"
    else:
        return "neutral"

def should_suppress_duplicate(last_state: Optional[Dict[str, Any]], current_signal: str, suppression_seconds: int) -> bool:
    if not last_state:
        return False
    last_signal = last_state.get("state")
    last_ts = int(last_state.get("ts", 0))
    if last_signal == current_signal and (now_ts() - last_ts) <= suppression_seconds:
        return True
    return False

# -------------------------
# CORE PAIR EVALUATION
# -------------------------
async def evaluate_pair_async(
    session: aiohttp.ClientSession,
    fetcher: DataFetcher,
    products_map: Dict[str, dict],
    pair_name: str,
    last_state: Optional[Dict[str, Any]]
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Evaluate a single pair with enhanced error handling"""
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
        
        tasks = [
            fetcher.fetch_candles(session, prod['symbol'], "15", limit_15m)
        ]
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
        if df_daily is None or len(df_daily) < 2:
            logger.warning(f"{pair_name}: could not fetch sufficient daily candles")
            return None
        
        df_daily['date'] = pd.to_datetime(df_daily['timestamp'], unit='s', utc=True).dt.date
        today = datetime.now(timezone.utc).date()
        yesterday = today - timedelta(days=1)
        prev_day_row = df_daily[df_daily['date'] == yesterday]
        
        if not prev_day_row.empty:
            prev = prev_day_row.iloc[-1]
        else:
            prev = df_daily.iloc[-2]
        
        prev_day_ohlc = {'high': float(prev['high']), 'low': float(prev['low']), 'close': float(prev['close'])}
        pivots = calculate_fibonacci_pivots(prev_day_ohlc['high'], prev_day_ohlc['low'], prev_day_ohlc['close'])
        vwap_15m = calculate_vwap_daily_reset(df_15m)
        ppo, _ = calculate_ppo(df_15m, cfg.PPO_FAST, cfg.PPO_SLOW, cfg.PPO_SIGNAL, cfg.PPO_USE_SMA)
        upw, dnw, _, _ = calculate_cirrus_cloud(df_15m)
        rma_50_15m = calculate_rma(df_15m['close'], 50)
        magical_hist = calculate_magical_momentum_hist(df_15m, period=144, responsiveness=0.9)
        
        idx = last_i_15m
        open_c = float(df_15m['open'].iloc[idx])
        close_c = float(df_15m['close'].iloc[idx])
        high_c = float(df_15m['high'].iloc[idx])
        low_c = float(df_15m['low'].iloc[idx])
        
        candle_pct = abs((close_c - open_c) / open_c) * 100 if open_c != 0 else 0
        if candle_pct > cfg.EXTREME_CANDLE_PCT:
            logger.info(f"{pair_name}: extreme candle {candle_pct:.2f}% > {cfg.EXTREME_CANDLE_PCT}% - skipping")
            return None
        
        try:
            ppo_curr = float(ppo.iloc[idx])
            rma50_curr = float(rma_50_15m.iloc[idx])
            magical_curr = float(magical_hist.iloc[idx]) if len(magical_hist) > abs(idx) else np.nan
        except Exception:
            logger.debug(f"{pair_name}: indicators indexing issue - skipping")
            return None
        
        rma200_5m_curr = np.nan
        if df_5m is not None:
            rma200 = calculate_rma(df_5m['close'], 200)
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
        wick_ratio = 0.35 # Max size of opposite wick relative to total range for it to be 'okay'
        upper_wick_ok = upper_wick / total_range < wick_ratio if total_range > 0 else True
        lower_wick_ok = lower_wick / total_range < wick_ratio if total_range > 0 else True
        is_green = close_c > open_c
        is_red = close_c < open_c
        
        # RMA200 check: 5m close must be above 200 RMA for long, below for short
        rma_long_ok = not cfg.USE_RMA200 or (rma_200_available and close_c > rma200_5m_curr)
        rma_short_ok = not cfg.USE_RMA200 or (rma_200_available and close_c < rma200_5m_curr)
        
        # FIBONACCI LOGIC
        long_crossover_line = None
        long_crossover_name = None
        for name in ['P', 'S1', 'S2', 'S3']:
            line = pivots.get(name)
            if df_15m['close'].iloc[idx-1] <= line and close_c > line:
                long_crossover_line = line
                long_crossover_name = name
                break
        
        short_crossover_line = None
        short_crossover_name = None
        for name in ['P', 'R1', 'R2', 'R3']:
            line = pivots.get(name)
            if df_15m['close'].iloc[idx-1] >= line and close_c < line:
                short_crossover_line = line
                short_crossover_name = name
                break
        
        # VWAP LOGIC
        vwap_curr = float(vwap_15m.iloc[idx]) if vwap_15m is not None else None
        
        # Combined Base Logic
        base_long_ok = (cloud_state == "green") and upper_wick_ok and rma_long_ok and (magical_curr > 0) and is_green
        base_short_ok = (cloud_state == "red") and lower_wick_ok and rma_short_ok and (magical_curr < 0) and is_red
        
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
        
        up_sig = "🟢⬆️"
        down_sig = "🔴⬇️"
        current_signal = None
        message = None
        last_state_pair = last_state
        suppress_secs = cfg.DUPLICATE_SUPPRESSION_SECONDS
        
        if vbuy:
            current_signal = "vbuy"
            if not should_suppress_duplicate(last_state_pair, current_signal, suppress_secs):
                vwap_str = f"{vwap_curr:,.2f}"
                ppo_str = f"{ppo_curr:.2f}"
                price_str = f"{close_c:,.2f}"
                message = (f"{up_sig} {sanitize_for_telegram(pair_name)} - VBuy\n"
                           f"Closed Above VWAP (${vwap_str})\n"
                           f"PPO 15m: {ppo_str}\n"
                           f"Price: ${price_str}\n"
                           f"{human_ts()}")
        elif vsell:
            current_signal = "vsell"
            if not should_suppress_duplicate(last_state_pair, current_signal, suppress_secs):
                vwap_str = f"{vwap_curr:,.2f}"
                ppo_str = f"{ppo_curr:.2f}"
                price_str = f"{close_c:,.2f}"
                message = (f"{down_sig} {sanitize_for_telegram(pair_name)} - VSell\n"
                           f"Closed Below VWAP (${vwap_str})\n"
                           f"PPO 15m: {ppo_str}\n"
                           f"Price: ${price_str}\n"
                           f"{human_ts()}")
        elif fib_long:
            current_signal = f"fib_long_{long_crossover_name}"
            if not should_suppress_duplicate(last_state_pair, current_signal, suppress_secs):
                line_str = f"{long_crossover_line:,.2f}"
                ppo_str = f"{ppo_curr:.2f}"
                price_str = f"{close_c:,.2f}"
                message = (f"{up_sig} {sanitize_for_telegram(pair_name)} - Fib Long (Cross {long_crossover_name})\n"
                           f"Price Crossed Above {long_crossover_name} (${line_str})\n"
                           f"PPO 15m: {ppo_str}\n"
                           f"Price: ${price_str}\n"
                           f"{human_ts()}")
        elif fib_short:
            current_signal = f"fib_short_{short_crossover_name}"
            if not should_suppress_duplicate(last_state_pair, current_signal, suppress_secs):
                line_str = f"{short_crossover_line:,.2f}"
                ppo_str = f"{ppo_curr:.2f}"
                price_str = f"{close_c:,.2f}"
                message = (f"{down_sig} {sanitize_for_telegram(pair_name)} - Fib Short (Cross {short_crossover_name})\n"
                           f"Price Crossed Below {short_crossover_name} (${line_str})\n"
                           f"PPO 15m: {ppo_str}\n"
                           f"Price: ${price_str}\n"
                           f"{human_ts()}")
        
        # Determine the new state to save
        if current_signal:
            new_state = current_signal
            new_ts = now_ts()
        elif last_state_pair and last_state_pair.get("state") in ["vbuy", "vsell", "fib_long_P", "fib_short_P"]:
            # If no new signal, but we were in a strong previous state (P/VWAP), keep it until a new signal occurs
            new_state = last_state_pair.get("state")
            new_ts = last_state_pair.get("ts")
        else:
            new_state = None
            new_ts = None
        
        if message:
            logger.info(f"SIGNAL: {message.replace('\n', ' | ')}")
            return new_state, {"message": message, "ts": now_ts(), "state": current_signal}
        elif new_state:
            return new_state, {"ts": new_ts, "state": new_state}
        else:
            return None, None
    except Exception as e:
        logger.exception(f"Error evaluating {pair_name}: {e}")
        METRICS["logic_errors"] += 1
        return None, None

# -------------------------
# TELEGRAM SENDER
# -------------------------
async def send_telegram_async(
    token: str,
    chat_id: str,
    text: str,
    session: Optional[aiohttp.ClientSession] = None
) -> bool:
    """Send Telegram message once"""
    if not token or not chat_id:
        logger.warning("Telegram not configured; skipping.")
        return False
    
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
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
    """Retries Telegram send with exponential backoff"""
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
async def check_dead_mans_switch(state_db: StateDB) -> None:
    """Alert if bot hasn't succeeded recently, with cooldown"""
    try:
        last_success = state_db.get_metadata("last_success_run")
        if last_success:
            try:
                last_success_int = int(last_success)
            except Exception:
                last_success_int = now_ts()
            hours_since = (now_ts() - last_success_int) / 3600.0
            if hours_since > 2: # Avoid repeating the dead-man alert every run: record a cooldown key
                dead_alert_ts = state_db.get_metadata("dead_alert_ts")
                if dead_alert_ts:
                    try:
                        dead_alert_int = int(dead_alert_ts)
                    except Exception:
                        dead_alert_int = 0
                else:
                    dead_alert_int = 0
                if now_ts() - dead_alert_int > 4 * 3600: # allow at most once every 4 hours
                    msg = (f"🚨 DEAD MAN'S SWITCH: Bot hasn't succeeded in {hours_since:.1f} hours!\n"
                           f"Last success: {datetime.fromtimestamp(last_success_int).strftime('%Y-%m-%d %H:%M:%S IST')}")
                    await send_telegram_with_retries(cfg.TELEGRAM_BOT_TOKEN, cfg.TELEGRAM_CHAT_ID, msg)
                    state_db.set_metadata("dead_alert_ts", str(now_ts()))
    except Exception as e:
        logger.exception(f"Dead man's switch failed: {e}")

# -------------------------
# CORE RUN LOGIC
# -------------------------
stop_requested = False

def request_stop(signum, frame) -> None:
    """Signal handler to request graceful stop"""
    global stop_requested
    stop_requested = True
    logger.warning(f"Stop requested (signal {signum})")

async def run_once(send_test: bool = True) -> None:
    global stop_requested
    reset_metrics()
    start_time = time.time()

    # Dead-man switch: ensure execution is fast enough
    async def check_timeout():
        while True:
            elapsed = time.time() - start_time
            if elapsed > cfg.MAX_EXEC_TIME:
                logger.critical(f"Execution timeout: {elapsed:.1f}s > {cfg.MAX_EXEC_TIME}s")
                raise TimeoutError("Max execution time exceeded")
            await asyncio.sleep(1)
    
    timeout_task = asyncio.create_task(check_timeout())

    # memory guard
    try:
        proc = psutil.Process(os.getpid())
    except Exception:
        proc = None

    try:
        logger.info("🚀 Starting fibonacci_pivot_bot_production run")
        # Memory check early
        if proc:
            try:
                rss = proc.memory_info().rss
                if rss > cfg.MEMORY_LIMIT_BYTES:
                    logger.critical(f"High memory usage at start: {rss} bytes > {cfg.MEMORY_LIMIT_BYTES}")
                    raise MemoryError("High memory usage")
            except Exception as e:
                logger.debug(f"Memory check failed: {e}")

        state_db = StateDB(cfg.STATE_DB_PATH)
        await check_dead_mans_switch(state_db)
        
        try:
            prune_old_state_records(cfg.STATE_DB_PATH, cfg.STATE_EXPIRY_DAYS, logger)
        except Exception as e:
            logger.warning(f"Prune error: {e}")

        last_alerts = state_db.load_all()
        if cfg.RESET_STATE:
            logger.warning("🔄 RESET_STATE requested: clearing states table")
            for p in cfg.PAIRS:
                state_db.set(p, None)

        if send_test and cfg.SEND_TEST_MESSAGE:
            test_msg = (f"🔔 Fibonacci Pivot Bot started\n"
                        f"Time: {human_ts()}\n"
                        f"Debug: {'ON' if cfg.DEBUG_MODE else 'OFF'}\n"
                        f"Pairs: {len(cfg.PAIRS)}")
            await send_telegram_with_retries(cfg.TELEGRAM_BOT_TOKEN, cfg.TELEGRAM_CHAT_ID, test_msg)

        products_map = load_products_cache()
        circuit = CircuitBreaker()
        fetcher_local = DataFetcher(
            cfg.DELTA_API_BASE,
            max_parallel=cfg.MAX_CONCURRENCY,
            timeout=cfg.HTTP_TIMEOUT,
            circuit_breaker=circuit
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
                )
                for pair_name in cfg.PAIRS
            ]

            # Run evaluation tasks concurrently with concurrency limit
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

            if alert_info and alert_info.get("message"):
                success = await send_telegram_with_retries(
                    cfg.TELEGRAM_BOT_TOKEN, cfg.TELEGRAM_CHAT_ID, alert_info["message"]
                )
                if success:
                    alerts_sent += 1
                    METRICS["alerts_sent"] += 1

        # Update metrics and dead-man switch metadata
        METRICS["execution_time"] = time.time() - start_time
        logger.info(f"✅ Run finished in {METRICS['execution_time']:.2f}s | Pairs checked: {METRICS['pairs_checked']} | Alerts sent: {alerts_sent}")
        state_db.set_metadata("last_success_run", str(now_ts()))
        state_db.set_metadata("last_run_metrics", json.dumps(METRICS))
        state_db.close()

    except TimeoutError:
        raise
    except MemoryError:
        raise
    except SystemExit:
        raise
    except asyncio.CancelledError:
        logger.warning("Run was cancelled")
        raise
    except TimeoutError as te:
        logger.critical(f"Timeout error: {te}")
        raise SystemExit(EXIT_TIMEOUT)
    except Exception as e:
        logger.exception(f"Unhandled error in run_once: {e}")
        raise
    finally:
        # ensure timeout_task is cancelled
        try:
            timeout_task.cancel()
            await timeout_task
        except asyncio.CancelledError:
            pass

# -------------------------
# CLI + SIGNAL HANDLING
# -------------------------
def main() -> None:
    """Entry point with process locking"""
    # Register signal handlers
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    # Acquire process lock
    lock = ProcessLock("/tmp/fib_bot.lock", timeout=cfg.MAX_EXEC_TIME * 2)
    if not lock.acquire():
        logger.error("❌ Another instance is running or stale lock exists. Exiting.")
        sys.exit(EXIT_LOCK_CONFLICT)

    try:
        import argparse
        parser = argparse.ArgumentParser(
            description="Improved Fibonacci Pivot Bot for Delta Exchange.",
            formatter_class=argparse.RawTextHelpFormatter
        )
        parser.add_argument("--once", action="store_true", help="Run once and exit (default).")
        parser.add_argument("--loop", type=int, default=cfg.RUN_LOOP_INTERVAL,
                            help=f"Run in a loop with specified interval in seconds (default: {cfg.RUN_LOOP_INTERVAL}).")
        parser.add_argument("--reset-state", action="store_true",
                            help="Clear the state database before running.")
        parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
        args = parser.parse_args()

        if args.reset_state:
            cfg.RESET_STATE = True
            logger.warning("Configuration override: RESET_STATE=True")
        if args.debug:
            cfg.DEBUG_MODE = True
            logger.warning("Configuration override: DEBUG_MODE=True")
            logger.setLevel(logging.DEBUG)

        if args.once:
            try:
                asyncio.run(run_once())
            except SystemExit as e:
                logger.error(f"Exited with code {e.code}")
                sys.exit(e.code if isinstance(e.code, int) else 1)
        elif args.loop:
            interval = args.loop
            logger.info(f"🔄 Loop mode: interval={interval}s")
            while not stop_requested:
                loop_start = time.time()
                try:
                    asyncio.run(run_once(send_test=False))
                except SystemExit as e:
                    logger.error(f"Run exited with code {e.code}, continuing loop.")
                except Exception:
                    logger.exception("Unhandled in run_once")

                elapsed = time.time() - loop_start
                to_sleep = max(0, interval - elapsed)
                if to_sleep > 0:
                    # break early if stop requested during sleep
                    for _ in range(int(to_sleep)):
                        if stop_requested:
                            break
                        time.sleep(1)
                    if not stop_requested:
                        remainder = to_sleep - int(to_sleep)
                        if remainder > 0:
                            time.sleep(remainder)
        else:
            try:
                asyncio.run(run_once())
            except SystemExit as e:
                logger.error(f"Exited with code {e.code}")
                sys.exit(e.code if isinstance(e.code, int) else 1)

        sys.exit(EXIT_SUCCESS)

    finally:
        lock.release()

if __name__ == "__main__":
    main()