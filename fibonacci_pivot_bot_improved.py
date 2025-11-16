#!/usr/bin/env python3
# python 3.12+
"""
Improved Fibonacci Pivot Bot - Enhanced Debug Version
- Fixed candle timing issues
- Enhanced debug logging for signal conditions
- Better error handling
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
from aiohttp import ClientConnectorError, ClientResponseError, TCPConnector, ClientTimeout
from logging.handlers import RotatingFileHandler

# Pydantic V2 imports
try:
    from pydantic import BaseModel, Field, field_validator, ConfigDict
except ImportError:
    print("❌ pydantic not found. Install with: pip install 'pydantic>=2.5.0'")
    sys.exit(1)

# -------------------------
# CONFIGURATION MODEL
# -------------------------
class Config(BaseModel):
    model_config = ConfigDict(extra='ignore')
    
    TELEGRAM_BOT_TOKEN: str = Field(..., min_length=10)
    TELEGRAM_CHAT_ID: str = Field(..., min_length=1)
    DEBUG_MODE: bool = False
    SEND_TEST_MESSAGE: bool = True
    RESET_STATE: bool = False
    DELTA_API_BASE: str = "https://api.india.delta.exchange"
    PAIRS: List[str] = Field(default=[
        "BTCUSD", "ETHUSD", "SOLUSD", "AVAXUSD", "BCHUSD", "XRPUSD", 
        "BNBUSD", "LTCUSD", "DOTUSD", "ADAUSD", "SUIUSD", "AAVEUSD"
    ])
    SPECIAL_PAIRS: Dict[str, Dict[str, int]] = Field(default={
        "SOLUSD": {"limit_15m": 210, "min_required": 180, "limit_5m": 300, "min_required_5m": 200}
    })
    X1: int = Field(default=22, ge=1)
    X2: int = Field(default=9, ge=1)
    X3: int = Field(default=15, ge=1)
    X4: int = Field(default=5, ge=1)
    PIVOT_LOOKBACK_PERIOD: int = Field(default=15, ge=1)
    STATE_DB_PATH: str = "fib_state.sqlite"
    LOG_FILE: str = "fibonacci_pivot_bot.log"
    LOCK_FILE_PATH: str = "fib_pivot.lock"
    MAX_CONCURRENCY: int = Field(default=6, ge=1, le=20)
    HTTP_TIMEOUT: int = Field(default=15, ge=5, le=60)
    FETCH_RETRIES: int = Field(default=3, ge=1, le=5)
    FETCH_BACKOFF: float = Field(default=1.5, ge=1.0, le=5.0)
    JITTER_MIN: float = Field(default=0.05, ge=0.0, le=1.0)
    JITTER_MAX: float = Field(default=0.6, ge=0.0, le=2.0)
    STATE_EXPIRY_DAYS: int = Field(default=30, ge=1)
    DUPLICATE_SUPPRESSION_SECONDS: int = Field(default=3600, ge=300)
    EXTREME_CANDLE_PCT: float = Field(default=8.0, ge=0.1, le=50.0)
    RUN_LOOP_INTERVAL: int = Field(default=900, ge=60)
    MAX_EXEC_TIME: int = Field(default=25, ge=10, le=300)
    USE_RMA200: bool = True
    PRODUCTS_CACHE_TTL: int = Field(default=86400, ge=3600)
    LOG_LEVEL: str = "INFO"
    TELEGRAM_RETRIES: int = Field(default=3, ge=1, le=5)
    TELEGRAM_BACKOFF_BASE: float = Field(default=2.0, ge=1.0, le=5.0)
    MEMORY_LIMIT_BYTES: int = Field(default=400_000_000, ge=100_000_000)
    DEADMAN_HOURS: float = Field(default=2.0, ge=0.5, le=24.0)

    @field_validator("PAIRS")
    @classmethod
    def validate_pairs(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("PAIRS list cannot be empty")
        return [p.upper().replace(" ", "") for p in v]

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of {valid_levels}")
        return v.upper()

# -------------------------
# CONFIG LOADER
# -------------------------
def str_to_bool(value: str) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes", "y", "t")

def load_config() -> Config:
    config_file = os.getenv("CONFIG_FILE", "config_fib.json")
    
    if not Path(config_file).exists():
        print(f"❌ Config file {config_file} not found")
        sys.exit(1)
    
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            user_cfg = json.load(f)
        
        # Validate required Telegram fields
        if not user_cfg.get("TELEGRAM_BOT_TOKEN") or user_cfg.get("TELEGRAM_BOT_TOKEN") == "xxxx":
            raise ValueError("TELEGRAM_BOT_TOKEN must be set in config file")
        if not user_cfg.get("TELEGRAM_CHAT_ID") or user_cfg.get("TELEGRAM_CHAT_ID") == "xxxx":
            raise ValueError("TELEGRAM_CHAT_ID must be set in config file")
        
        print(f"✅ Loaded configuration from {config_file}")
        return Config(**user_cfg)
        
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        sys.exit(1)

# Load configuration
cfg = load_config()

# -------------------------
# EXIT CODES
# -------------------------
EXIT_SUCCESS = 0
EXIT_LOCK_CONFLICT = 2
EXIT_TIMEOUT = 3
EXIT_CONFIG_ERROR = 4
EXIT_API_FAILURE = 5

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
# STANDARDIZED LOGGING
# -------------------------
class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_entry, ensure_ascii=False)

def setup_logging():
    logger = logging.getLogger("fibonacci_pivot_bot")
    
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    log_level = getattr(logging, cfg.LOG_LEVEL, logging.INFO)
    logger.setLevel(logging.DEBUG if cfg.DEBUG_MODE else log_level)
    
    # Console handler with JSON format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(JSONFormatter())
    logger.addHandler(console_handler)
    
    # File handler with rotation
    try:
        file_handler = RotatingFileHandler(
            cfg.LOG_FILE, 
            maxBytes=10_000_000,  # 10MB
            backupCount=5, 
            encoding="utf-8"
        )
        file_handler.setFormatter(JSONFormatter())
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"⚠️ Could not create log file: {e}")
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger

logger = setup_logging()

# -------------------------
# PROCESS LOCKING (Configurable Path)
# -------------------------
class ProcessLock:
    """File-based lock with stale detection and configurable path"""
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
            logger.debug(f"Acquired lock: {self.lock_path}")
            return True
        except (IOError, OSError):
            try:
                if self.lock_path.exists():
                    with open(self.lock_path, 'r') as f:
                        raw = f.read().strip()
                    old_pid = int(raw) if raw.isdigit() else None
                    if old_pid and not psutil.pid_exists(old_pid):
                        logger.warning(f"Removing stale lock from PID {old_pid}")
                        try:
                            self.lock_path.unlink(missing_ok=True)
                        except Exception:
                            pass
                        return self.acquire()
            except Exception as e:
                logger.debug(f"Error checking stale lock: {e}")
            
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
            logger.debug(f"Released lock: {self.lock_path}")
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
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        except Exception:
            pass
        self._ensure_tables()
        try:
            os.chmod(self.db_path, 0o600)
        except Exception:
            pass

    def _ensure_tables(self):
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

    def set(self, pair: str, state: Optional[str], ts: Optional[int] = None):
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
        cur = self._conn.cursor()
        cur.execute("SELECT value FROM metadata WHERE key = ?", (key,))
        r = cur.fetchone()
        return r[0] if r else None

    def set_metadata(self, key: str, value: str):
        cur = self._conn.cursor()
        cur.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)", (key, value))
        self._conn.commit()

    def close(self):
        try:
            self._conn.close()
        except Exception:
            pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

# -------------------------
# PRUNE (smart daily)
# -------------------------
def prune_old_state_records(db_path: str, expiry_days: int = 30):
    try:
        if expiry_days <= 0:
            logger.debug("Pruning disabled (expiry_days <= 0)")
            return

        if not os.path.exists(db_path):
            logger.info(f"State DB not found at {db_path}; skipping prune.")
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
                    logger.debug("Prune already run today; skipping.")
                    conn.close()
                    return
            except Exception:
                pass

        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='states'")
        if not cur.fetchone():
            logger.debug("No states table yet; skipping prune.")
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
                logger.warning(f"VACUUM failed: {e}")

        conn.close()
        logger.info(f"Pruned {deleted} states older than {expiry_days} days from {db_path}.")
    except Exception as e:
        logger.warning(f"Prune failed: {e}")
        logger.debug(traceback.format_exc())

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
                async with session.get(url, params=params, timeout=ClientTimeout(total=timeout)) as resp:
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
                logger.info(f"Using cached products ({age:.0f}s old)")
                with open(cache_path, "r", encoding="utf-8") as f:
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
    """
    Magical Momentum Histogram - matches Pine Script v6 exactly.

    Pine reference:
      sd = stdev(source, 50) * responsiveness
      var worm = source; diff = source - worm; delta = abs(diff) > sd ? sign(diff)*sd : diff; worm := worm + delta
      ma = sma(source, period)
      raw_momentum = (worm - ma) / worm
      temp = (raw_momentum - lowest(raw_momentum, period)) / (highest(raw_momentum, period) - lowest(...))
      value = 1.0
      value := value * (temp - .5 + .5 * nz(value[1]))
      value clipped to [-0.9999, 0.9999]
      temp2 = (1 + value) / (1 - value)
      momentum = .25 * log(temp2)
      momentum := momentum + .5 * nz(momentum[1])
      hist = momentum
    """
    n = len(df)
    if n == 0:
        return pd.Series([], dtype=float)
    if n < period + 50:
        return pd.Series(np.zeros(n), index=df.index, dtype=float)

    source = df['close'].astype(float).copy()

    # Match Pine: population stddev (ddof=0) and responsiveness floor
    sd = source.rolling(window=50, min_periods=10).std(ddof=0) * max(0.00001, responsiveness)
    sd = sd.bfill().ffill().clip(lower=1e-6)

    # Worm recursive clamp by sd
    worm = source.copy()
    for i in range(1, n):
        diff = source.iloc[i] - worm.iloc[i - 1]
        delta = np.sign(diff) * sd.iloc[i] if abs(diff) > sd.iloc[i] else diff
        worm.iloc[i] = worm.iloc[i - 1] + delta

    # SMA(period) of source
    ma = source.rolling(window=period, min_periods=max(5, period // 3)).mean().bfill().ffill()

    # Raw momentum normalized by worm
    denom = worm.replace(0, np.nan).bfill().ffill()
    raw_momentum = (worm - ma) / denom
    raw_momentum = raw_momentum.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Min/max over rolling window
    min_med = raw_momentum.rolling(window=period, min_periods=max(5, period // 3)).min().bfill().ffill()
    max_med = raw_momentum.rolling(window=period, min_periods=max(5, period // 3)).max().bfill().ffill()
    rng = (max_med - min_med).replace(0, np.nan)

    # temp in [0,1] from current_med normalization
    temp = pd.Series(0.0, index=df.index)
    valid = rng.notna()
    temp.loc[valid] = (raw_momentum.loc[valid] - min_med.loc[valid]) / rng.loc[valid]
    temp = temp.clip(-1, 1).fillna(0)

    # Value recursion: multiplicative as per Pine
    value = pd.Series(0.0, index=df.index)
    value.iloc[0] = 1.0
    for i in range(1, n):
        prev_val = value.iloc[i - 1]
        expr = temp.iloc[i] - 0.5 + 0.5 * prev_val
        new_val = prev_val * expr
        # Clip to [-0.9999, 0.9999]
        if new_val > 0.9999:
            new_val = 0.9999
        elif new_val < -0.9999:
            new_val = -0.9999
        value.iloc[i] = new_val

    temp2 = (1 + value) / (1 - value)
    temp2 = temp2.replace([np.inf, -np.inf], np.nan).clip(lower=1e-6).fillna(1e-6)

    # Momentum recursion on momentum itself
    momentum = pd.Series(0.0, index=df.index)
    for i in range(n):
        base = 0.25 * np.log(temp2.iloc[i])
        momentum.iloc[i] = base if i == 0 else base + 0.5 * momentum.iloc[i - 1]

    # Optional debug trace for last bars
    if cfg.DEBUG_MODE and n >= 2:
        li = -1
        pi = -2
        try:
            logger.debug(
                json.dumps({
                    "mmh_debug": {
                        "last_close": float(source.iloc[li]),
                        "last_worm": float(worm.iloc[li]),
                        "last_temp": float(temp.iloc[li]),
                        "last_value": float(value.iloc[li]),
                        "last_momentum": float(momentum.iloc[li]),
                        "prev_momentum": float(momentum.iloc[pi])
                    }
                })
            )
        except Exception:
            pass

    return momentum.replace([np.inf, -np.inf], 0).fillna(0)

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
    if close < pivot:  # Bearish close (Pivot is closer to High)
        R3 = pivot + (high - low)
        R2 = pivot + 0.618 * (high - low)
        R1 = pivot + 0.382 * (high - low)
        S1 = pivot - 0.382 * (high - low)
        S2 = pivot - 0.618 * (high - low)
        S3 = pivot - (high - low)
    elif close > pivot:  # Bullish close (Pivot is closer to Low)
        R3 = pivot + (high - low)
        R2 = pivot + 0.618 * (high - low)
        R1 = pivot + 0.382 * (high - low)
        S1 = pivot - 0.382 * (high - low)
        S2 = pivot - 0.618 * (high - low)
        S3 = pivot - (high - low)
    else:  # Neutral close
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
    
    # Buy signals: P, S1, S2, S3, R1, R2 (exclude R3)
    # Sell signals: P, S1, S2, R1, R2, R3 (exclude S3)
    if direction == "long":
        levels = ["R2", "R1", "P", "S1", "S2", "S3"]
    elif direction == "short":
        levels = ["R3", "R2", "R1", "P", "S2", "S1"]
    else:
        return None
    
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
# CORE PAIR EVALUATION
# -------------------------
async def evaluate_pair_async(
    session: aiohttp.ClientSession,
    fetcher: DataFetcher,
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

        # CORRECTED: Use -1 for last completed candle
        last_i_15m = -1
        last_i_5m = -1
        
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
        upw, dnw, filtx1, filtx12 = calculate_cirrus_cloud(df_15m)
        magical_hist = calculate_magical_momentum_hist(df_15m)
        
        # VWAP Calculation (15m)
        vwap_15m: Optional[pd.Series] = None
        if df_daily is not None and len(df_daily) >= 2:
            vwap_15m = calculate_vwap_daily_reset(df_15m)
            try:
                vwap_curr = float(vwap_15m.iloc[last_i_15m])
            except Exception:
                vwap_curr = None
                
        # Extract values from 15m - using last_i_15m for last closed candle
        close_c = float(df_15m['close'].iloc[last_i_15m])
        open_c = float(df_15m['open'].iloc[last_i_15m])
        high_c = float(df_15m['high'].iloc[last_i_15m])
        low_c = float(df_15m['low'].iloc[last_i_15m])
        magical_curr = float(magical_hist.iloc[last_i_15m])
        
        # Calculate RMA50 for 15m
        rma50 = calculate_rma(df_15m['close'].astype(float), 50)
        rma50_curr = float(rma50.iloc[last_i_15m])

        # RMA 200 (5m) check
        rma200_5m_curr = None
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
        wick_ratio = 0.20  # Changed to 20%
        upper_wick_ok = upper_wick / total_range < wick_ratio if total_range > 0 else True
        lower_wick_ok = lower_wick / total_range < wick_ratio if total_range > 0 else True
        
        is_green = close_c > open_c
        is_red = close_c < open_c

        # Enhanced debug output for signal conditions
        if cfg.DEBUG_MODE:
            vwap_log_str = f"{vwap_curr:.2f}" if vwap_curr is not None and not np.isnan(vwap_curr) else "nan"
            rma200_log_str = f"{rma200_5m_curr:.2f}" if rma200_5m_curr is not None else "N/A"
            
            logger.debug(
                f"{pair_name}: close={close_c:.2f}, open={open_c:.2f}, "
                f"RMA50={rma50_curr:.2f}, RMA200={rma200_log_str}, "
                f"MMH={magical_curr:.4f}, Cloud={cloud_state}, "
                f"VWAP={vwap_log_str}, Green={is_green}, Red={is_red}, "
                f"UpperWickOK={upper_wick_ok}, LowerWickOK={lower_wick_ok}"
            )

        # RMA200 check: close must be above 200 RMA for long, below for short
        rma_long_ok = not cfg.USE_RMA200 or (rma_200_available and close_c > rma200_5m_curr)
        rma_short_ok = not cfg.USE_RMA200 or (rma_200_available and close_c < rma200_5m_curr)
        
        # Base requirements for signal (NO PPO CONDITION)
        base_long_ok = (
            is_green and
            cloud_state == "bullish" and
            magical_curr > 0 and
            upper_wick_ok and
            rma_long_ok and
            close_c > rma50_curr
        )
        
        base_short_ok = (
            is_red and
            cloud_state == "bearish" and
            magical_curr < 0 and
            lower_wick_ok and
            rma_short_ok and
            close_c < rma50_curr
        )

        # CORRECTED: Use -2 for previous close (the candle before last completed)
        prev_close = float(df_15m['close'].iloc[-2]) if len(df_15m) >= 2 else close_c
        long_crossover = get_crossover_line(pivots, prev_close, close_c, "long")
        short_crossover = get_crossover_line(pivots, prev_close, close_c, "short")
        
        long_crossover_name = long_crossover[0] if long_crossover else None
        long_crossover_line = long_crossover[1] if long_crossover else None
        short_crossover_name = short_crossover[0] if short_crossover else None
        short_crossover_line = short_crossover[1] if short_crossover else None

        # CORRECTED VWAP Crossover Signal
        vbuy = False
        vsell = False
        if vwap_curr is not None and len(df_15m) >= 2:
            prev_vwap = float(vwap_15m.iloc[-2]) if len(vwap_15m) >= 2 else vwap_curr
            if base_long_ok and prev_close <= prev_vwap and close_c > vwap_curr:
                vbuy = True
            if base_short_ok and prev_close >= prev_vwap and close_c < vwap_curr:
                vsell = True

        fib_long = base_long_ok and (long_crossover_line is not None)
        fib_short = base_short_ok and (short_crossover_line is not None)
        
        # Enhanced signal debugging
        if cfg.DEBUG_MODE:
            logger.debug(
                f"{pair_name} SIGNAL CHECK: "
                f"base_long_ok={base_long_ok} (green={is_green}, cloud={cloud_state}, "
                f"MMH>0={magical_curr>0}, upper_wick_ok={upper_wick_ok}, "
                f"rma_long_ok={rma_long_ok}, close>RMA50={close_c>rma50_curr}), "
                f"base_short_ok={base_short_ok}, "
                f"fib_long={fib_long}, fib_short={fib_short}, "
                f"vbuy={vbuy}, vsell={vsell}"
            )
            if long_crossover:
                logger.debug(f"{pair_name} LONG CROSSOVER: {long_crossover_name} at {long_crossover_line}")
            if short_crossover:
                logger.debug(f"{pair_name} SHORT CROSSOVER: {short_crossover_name} at {short_crossover_line}")
        
        up_sig = "🔼🔵"
        down_sig = "🔽🟣"
        current_signal = None
        message = None
        suppress_secs = cfg.DUPLICATE_SUPPRESSION_SECONDS

        if vbuy:
            current_signal = "vbuy"
            if not should_suppress_duplicate(last_state, current_signal, suppress_secs):
                vwap_str = f"{vwap_curr:,.2f}"
                price_str = f"{close_c:,.2f}"
                message = (
                    f"{up_sig} {sanitize_for_telegram(pair_name)} - VBuy\n"
                    f"Closed Above VWAP (${vwap_str})\n"
                    f"Price: ${price_str}\n"
                    f"{human_ts()}"
                )
        elif vsell:
            current_signal = "vsell"
            if not should_suppress_duplicate(last_state, current_signal, suppress_secs):
                vwap_str = f"{vwap_curr:,.2f}"
                price_str = f"{close_c:,.2f}"
                message = (
                    f"{down_sig} {sanitize_for_telegram(pair_name)} - VSell\n"
                    f"Closed Below VWAP (${vwap_str})\n"
                    f"Price: ${price_str}\n"
                    f"{human_ts()}"
                )
        elif fib_long:
            current_signal = f"fib_long_{long_crossover_name}"
            if not should_suppress_duplicate(last_state, current_signal, suppress_secs):
                line_str = f"{long_crossover_line:,.2f}"
                price_str = f"{close_c:,.2f}"
                message = (
                    f"{up_sig} {sanitize_for_telegram(pair_name)} - Fib Long (Cross {long_crossover_name})\n"
                    f"Price Crossed Above {long_crossover_name} (${line_str})\n"
                    f"Price: ${price_str}\n"
                    f"{human_ts()}"
                )
        elif fib_short:
            current_signal = f"fib_short_{short_crossover_name}"
            if not should_suppress_duplicate(last_state, current_signal, suppress_secs):
                line_str = f"{short_crossover_line:,.2f}"
                price_str = f"{close_c:,.2f}"
                message = (
                    f"{down_sig} {sanitize_for_telegram(pair_name)} - Fib Short (Cross {short_crossover_name})\n"
                    f"Price Crossed Below {short_crossover_name} (${line_str})\n"
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
                return None, None
        
        if last_state and current_signal is None and now_ts() - int(last_state.get("ts", 0)) > suppress_secs:
            logger.debug(f"{pair_name}: Last state '{last_state.get('state')}' expired. Clearing state.")
            return None, None
            
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
async def send_telegram_async(token: str, chat_id: str, text: str, session: aiohttp.ClientSession) -> bool:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": "true"
    }
    
    try:
        async with session.post(url, data=data, timeout=ClientTimeout(total=10)) as resp:
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

async def send_telegram_with_retries(token: str, chat_id: str, text: str, session: aiohttp.ClientSession) -> bool:
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
# IMPROVED DEAD MAN'S SWITCH
# -------------------------
async def check_dead_mans_switch(state_db: StateDB):
    try:
        last_success = state_db.get_metadata("last_success_run")
        if not last_success:
            logger.debug("No last success run recorded yet")
            return
            
        try:
            last_success_int = int(last_success)
        except (ValueError, TypeError):
            logger.warning(f"Invalid last_success timestamp: {last_success}")
            return

        hours_since = (now_ts() - last_success_int) / 3600.0
        
        if hours_since > cfg.DEADMAN_HOURS:
            dead_alert_ts = state_db.get_metadata("dead_alert_ts")
            dead_alert_int = 0
            
            if dead_alert_ts:
                try:
                    dead_alert_int = int(dead_alert_ts)
                except (ValueError, TypeError):
                    pass
            
            # Only alert if we haven't alerted in the last 4 hours
            if now_ts() - dead_alert_int > 4 * 3600:
                last_success_dt = datetime.fromtimestamp(last_success_int).strftime('%Y-%m-%d %H:%M:%S IST')
                msg = (
                    f"🚨 DEAD MAN'S SWITCH: Bot hasn't succeeded in {hours_since:.1f} hours!\n"
                    f"Last success: {last_success_dt}\n"
                    f"Threshold: {cfg.DEADMAN_HOURS} hours\n"
                    f"Time: {human_ts()}"
                )
                
                # Use a temporary session for the dead man's switch
                connector = TCPConnector(limit=1, ssl=False)
                async with aiohttp.ClientSession(connector=connector) as temp_session:
                    if await send_telegram_with_retries(cfg.TELEGRAM_BOT_TOKEN, cfg.TELEGRAM_CHAT_ID, msg, temp_session):
                        state_db.set_metadata("dead_alert_ts", str(now_ts()))
                        logger.warning(f"Dead man's switch triggered: {hours_since:.1f} hours since last success")
                    else:
                        logger.error("Failed to send dead man's switch alert")
                        
    except Exception as e:
        logger.exception(f"Dead man's switch failed: {e}")

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
        prune_old_state_records(cfg.STATE_DB_PATH, cfg.STATE_EXPIRY_DAYS)
        state_db = StateDB(cfg.STATE_DB_PATH)
        await check_dead_mans_switch(state_db)
        last_alerts = state_db.load_all()

        if cfg.RESET_STATE:
            logger.warning("🔄 RESET_STATE requested: clearing states table")
            for p in cfg.PAIRS:
                state_db.set(p, None)

        if send_test and cfg.SEND_TEST_MESSAGE:
            test_msg = (
                f"📡 Fibonacci Pivot Bot started\n"
                f"Time: {human_ts()}\n"
                f"Debug: {'ON' if cfg.DEBUG_MODE else 'OFF'}\n"
                f"Pairs: {len(cfg.PAIRS)}"
            )
            # Create temporary session for test message
            connector = TCPConnector(limit=1, ssl=False)
            async with aiohttp.ClientSession(connector=connector) as temp_session:
                await send_telegram_with_retries(cfg.TELEGRAM_BOT_TOKEN, cfg.TELEGRAM_CHAT_ID, test_msg, temp_session)
            
        products_map = load_products_cache()
        circuit = CircuitBreaker()
        fetcher_local = DataFetcher(
            cfg.DELTA_API_BASE,
            max_parallel=cfg.MAX_CONCURRENCY,
            timeout=cfg.HTTP_TIMEOUT,
            circuit_breaker=circuit
        )
        
        # Use shared session for all HTTP requests
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
            
        METRICS["execution_time"] = time.time() - start_time
        logger.info(
            f"✅ Run finished. Pairs: {METRICS['pairs_checked']}, "
            f"Alerts: {METRICS['alerts_sent']}, Time: {METRICS['execution_time']:.2f}s"
        )
        
        if cfg.DEBUG_MODE:
            logger.debug(f"Metrics: {METRICS}")


# -------------------------
# MAIN EXECUTION
# -------------------------
def main():
    if os.getenv("GITHUB_ACTIONS"):
        logger.info("Running in GitHub Actions environment.")
        run_mode = "ONCE"
    elif os.getenv("CRON_JOB"):
        logger.info("Running in Cron Job environment.")
        run_mode = "ONCE"
    else:
        run_mode = "LOOP"

    # Memory limit check
    try:
        if psutil.virtual_memory().total < cfg.MEMORY_LIMIT_BYTES:
            logger.warning(f"Total system memory ({psutil.virtual_memory().total / (1024**3):.2f}GB) is below configured limit ({cfg.MEMORY_LIMIT_BYTES / (1024**3):.2f}GB). Proceeding but watch for OOM.")
    except Exception:
        pass

    # Process locking with configurable path
    lock = ProcessLock(cfg.LOCK_FILE_PATH, timeout=cfg.RUN_LOOP_INTERVAL * 2)
    if not lock.acquire():
        logger.error("❌ Failed to acquire process lock. Another instance is running.")
        sys.exit(EXIT_LOCK_CONFLICT)

    # Signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    
    try:
        if run_mode == "LOOP":
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
                        break
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
            logger.info("🚀 Single run mode (exiting immediately after completion).")
            asyncio.run(run_once())
            sys.exit(EXIT_SUCCESS)

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
