#!/usr/bin/env python3
# macd_bot.py
# Production-hardened MACD/PPO bot with Pydantic 2.0 config, standardized logging, and shared resources.

from __future__ import annotations

import os
import sys
import json
import time
import asyncio
import random
import logging
import sqlite3
import signal
import fcntl
import atexit
import gc
from typing import Dict, Any, Optional, Tuple, List, ClassVar
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager

# Pydantic 2.0 Imports
from pydantic import BaseModel, Field, field_validator, ValidationError

import aiohttp
import pandas as pd
import numpy as np
import pytz
import psutil
from aiohttp import ClientConnectorError, ClientResponseError, TCPConnector
from logging.handlers import RotatingFileHandler

# ==========================================================
# Pydantic Configuration Model and Loader
# ==========================================================
CONFIG_FILE = os.getenv("CONFIG_FILE", "config_macd.json")

class BotConfig(BaseModel):
    """
    Bot Configuration Model.
    All fields have sensible defaults and enforce validation rules.
    """
    # Telegram
    TELEGRAM_BOT_TOKEN: str = Field(default="xxxx", description="Your Telegram Bot Token")
    TELEGRAM_CHAT_ID: str = Field(default="xxxx", description="Your Telegram Chat ID")

    # Bot operation
    DEBUG_MODE: bool = Field(default=False)
    SEND_TEST_MESSAGE: bool = Field(default=False)
    BOT_NAME: str = Field(default="MACD Alert Bot")

    # API
    DELTA_API_BASE: str = Field(default="https://api.india.delta.exchange")
    PAIRS: List[str] = Field(default=["BTCUSD", "ETHUSD", "SOLUSD", "AVAXUSD",
                                       "BCHUSD", "XRPUSD", "BNBUSD", "LTCUSD",
                                       "DOTUSD", "ADAUSD", "SUIUSD", "AAVEUSD"])
    SPECIAL_PAIRS: Dict[str, Dict[str, int]] = Field(default={
        "SOLUSD": {"limit_15m": 210, "min_required": 180, "limit_5m": 300, "min_required_5m": 200}
    })

    # Indicator Parameters
    PPO_FAST: int = Field(default=7)
    PPO_SLOW: int = Field(default=16)
    PPO_SIGNAL: int = Field(default=5)
    PPO_USE_SMA: bool = Field(default=False)
    RMA_50_PERIOD: int = Field(default=50)
    RMA_200_PERIOD: int = Field(default=200)
    CIRRUS_CLOUD_ENABLED: bool = Field(default=True)
    X1: int = Field(default=22)
    X2: int = Field(default=9)
    X3: int = Field(default=15)
    X4: int = Field(default=5)
    SRSI_RSI_LEN: int = Field(default=21)
    SRSI_KALMAN_LEN: int = Field(default=5)

    # State/Logging/Run Parameters
    STATE_DB_PATH: str = Field(default="macd_state.sqlite")
    LOG_FILE: str = Field(default="macd_bot.log")
    PID_LOCK_PATH: str = Field(default="/tmp/macd_bot.pid", description="Path for the PID lock file")
    STATE_EXPIRY_DAYS: int = Field(default=30)
    RUN_TIMEOUT_SECONDS: int = Field(default=600, ge=60, description="Minimum 60 seconds")
    BATCH_SIZE: int = Field(default=4, ge=1)
    LOG_LEVEL: str = Field(default="INFO")

    # Concurrency/Network
    MAX_PARALLEL_FETCH: int = Field(default=4, ge=1, le=16, description="Improved validation (1-16)")
    HTTP_TIMEOUT: int = Field(default=15)
    CANDLE_FETCH_RETRIES: int = Field(default=3)
    CANDLE_FETCH_BACKOFF: float = Field(default=1.5)
    JITTER_MIN: float = Field(default=0.1)
    JITTER_MAX: float = Field(default=0.8)
    TCP_CONN_LIMIT: int = Field(default=8)
    TELEGRAM_RETRIES: int = Field(default=3)
    TELEGRAM_BACKOFF_BASE: float = Field(default=2.0)
    MEMORY_LIMIT_BYTES: int = Field(default=400_000_000)
    DEAD_MANS_COOLDOWN_SECONDS: int = Field(default=4 * 3600)

    # Pydantic 2.0 Field Validators
    @field_validator('PAIRS')
    @classmethod
    def check_pairs(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("PAIRS list cannot be empty")
        return v
    
    @field_validator('TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID', 'DELTA_API_BASE')
    @classmethod
    def check_secrets(cls, v: str) -> str:
        # Only check against the default 'xxxx' or empty string
        if v in ('xxxx', ''):
            # Allow 'xxxx' if not DEBUG_MODE
            return v
        return v

# Helper to cast env vars correctly, especially booleans
def str_to_bool(value: Any) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes", "y", "t")

def load_config(config_file: str) -> BotConfig:
    """Loads config from defaults, overrides from config file, and finally from environment variables."""
    user_config = {}
    if Path(config_file).exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                user_config = json.load(f)
                print(f"✅ Loaded configuration from {config_file}")
        except Exception as e:
            print(f"⚠️ Warning: unable to parse config file: {e}", file=sys.stderr)

    # Manually merge environment variables on top of user_config
    # This matches the old script's precedence: Defaults -> File -> Env Vars (highest)
    data = user_config.copy()
    
    # Iterate through all Pydantic fields and check environment variables
    for key, field in BotConfig.model_fields.items():
        env_val = os.getenv(key)
        if env_val is not None:
            # Handle booleans separately as Pydantic's simple string coercion might fail
            if field.annotation == bool:
                data[key] = str_to_bool(env_val)
            else:
                data[key] = env_val 

    try:
        cfg_instance = BotConfig.model_validate(data)
        
        # Final validation for Telegram fields that the user must replace 'xxxx'
        if cfg_instance.TELEGRAM_BOT_TOKEN == 'xxxx' or cfg_instance.TELEGRAM_CHAT_ID == 'xxxx':
             # Allow 'xxxx' only if in DEBUG_MODE and not sending test messages for local testing
            if not cfg_instance.DEBUG_MODE or cfg_instance.SEND_TEST_MESSAGE:
                 raise ValueError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be configured and not 'xxxx'")
        
        return cfg_instance
    except (ValidationError, ValueError) as e:
        print(f"❌ Critical Configuration Error: {e}", file=sys.stderr)
        sys.exit(1)

# Global configuration object
cfg: BotConfig = load_config(CONFIG_FILE)

# -------------------------
# Logger with Standardized Format
# -------------------------
logger = logging.getLogger("macd_bot")
# Use the config value, defaulting to INFO if invalid
log_level = getattr(logging, str(cfg.LOG_LEVEL).upper(), logging.INFO)
logger.setLevel(logging.DEBUG if cfg.DEBUG_MODE else log_level)

# Standardized Format: time.msecs | LEVEL    | module::function:line | message
fmt = logging.Formatter(
    "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s::%(funcName)s:%(lineno)d | %(message)s", 
    datefmt='%Y-%m-%d %H:%M:%S'
)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG if cfg.DEBUG_MODE else log_level)
ch.setFormatter(fmt)
logger.addHandler(ch)

try:
    log_dir = os.path.dirname(cfg.LOG_FILE) or "."
    os.makedirs(log_dir, exist_ok=True)
    fh = RotatingFileHandler(cfg.LOG_FILE, maxBytes=5_000_000, backupCount=5, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
except Exception as e:
    logger.warning(f"Could not set up rotating log file: {e}")

# -------------------------
# PID file lock (Configurable Path)
# -------------------------
class PidFileLock:
    def __init__(self, path: str):
        self.path = path
        self.fd = None

    def acquire(self) -> bool:
        try:
            # Ensure directory exists for the pid file
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            self.fd = open(self.path, "w")
            fcntl.flock(self.fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.fd.write(str(os.getpid()))
            self.fd.flush()
            atexit.register(self.release)
            logger.info(f"PID lock acquired at {self.path}")
            return True
        except (IOError, OSError):
            if self.fd:
                try:
                    self.fd.close()
                except Exception:
                    pass
            logger.error(f"Another instance is running or failed to acquire lock at {self.path}")
            return False

    def release(self):
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
            if os.path.exists(self.path):
                try:
                    os.unlink(self.path)
                    logger.debug(f"PID lock file {self.path} unlinked.")
                except Exception:
                    pass
        except Exception:
            pass

# -------------------------
# SQLite State DB (Context Manager)
# -------------------------
class StateDB:
    # ... (content remains the same as in original script, but now uses cfg object)
    def __init__(self, db_path: str):
        self.db_path = db_path
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

        self._conn = sqlite3.connect(self.db_path, check_same_thread=False, isolation_level=None)
        try:
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
        except Exception:
            pass
        self._ensure_tables()

    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

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
        ts = int(ts or time.time())
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

    def prune_old_records(self, expiry_days: int) -> int:
        if expiry_days <= 0:
            logger.debug("Pruning disabled (STATE_EXPIRY_DAYS <= 0)")
            return 0
        cur = self._conn.cursor()
        cur.execute("SELECT value FROM metadata WHERE key='last_prune'")
        row = cur.fetchone()
        today = datetime.now(timezone.utc).date()
        if row:
            try:
                last_prune_date = datetime.fromisoformat(row[0]).date()
                if last_prune_date >= today:
                    logger.debug("Daily prune already completed — skipping.")
                    return 0
            except Exception:
                pass
        cutoff = int(time.time()) - (expiry_days * 86400)
        cur.execute("DELETE FROM states WHERE ts < ?", (cutoff,))
        deleted = cur.rowcount
        cur.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES ('last_prune', ?)", (datetime.utcnow().isoformat(),))
        if deleted > 0:
            try:
                cur.execute("VACUUM;")
            except Exception as e:
                logger.warning(f"VACUUM failed: {e}")
        self._conn.commit()
        return deleted

    def close(self):
        try:
            if hasattr(self, '_conn') and self._conn:
                self._conn.close()
        except Exception:
            pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
            
# -------------------------
# Circuit Breaker (async)
# -------------------------
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, timeout: int = 300):
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.threshold = failure_threshold
        self.timeout = timeout
        self.lock = asyncio.Lock()

    async def is_open(self) -> bool:
        async with self.lock:
            if self.failure_count < self.threshold:
                return False
            if self.last_failure_time and (time.time() - self.last_failure_time > self.timeout):
                self.failure_count = 0
                self.last_failure_time = None
                return False
            return True

    async def call(self, func, *args, **kwargs):
        if await self.is_open():
            raise Exception("Circuit breaker is OPEN")
        try:
            result = await func(*args, **kwargs)
            await self.record_success()
            return result
        except Exception:
            await self.record_failure()
            raise

    async def record_failure(self):
        async with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            logger.warning(f"Circuit breaker: failure {self.failure_count}/{self.threshold}")
            return self.failure_count

    async def record_success(self):
        async with self.lock:
            if self.failure_count > 0:
                self.failure_count = 0
                self.last_failure_time = None
                logger.info("Circuit breaker: reset after success")

# -------------------------
# Rate Limited Fetcher
# -------------------------
class RateLimitedFetcher:
    def __init__(self, max_per_minute: int = 60):
        self.semaphore = asyncio.Semaphore(max_per_minute)
        self.requests = []
        
    async def call(self, func, *args, **kwargs):
        async with self.semaphore:
            # Clean old requests
            now = time.time()
            self.requests = [r for r in self.requests if now - r < 60]
            
            if len(self.requests) >= self.semaphore._value:
                await asyncio.sleep(1)
                
            self.requests.append(now)
            
            return await func(*args, **kwargs)

# -------------------------
# Async HTTP helpers with retries (uses shared session)
# -------------------------
async def async_fetch_json(session: aiohttp.ClientSession, url: str, params: dict = None,
                           retries: int = 3, backoff: float = 1.5, timeout: int = 15,
                           circuit_breaker: Optional[CircuitBreaker] = None) -> Optional[dict]:
    if circuit_breaker and await circuit_breaker.is_open():
        logger.warning(f"Circuit breaker open; skipping fetch {url}")
        return None

    for attempt in range(1, retries + 1):
        try:
            async with session.get(url, params=params, timeout=timeout) as resp:
                text = await resp.text()
                if resp.status >= 400:
                    logger.debug(f"HTTP {resp.status} {url} - {text[:200]}")
                    raise ClientResponseError(resp.request_info, resp.history, status=resp.status)
                try:
                    data = await resp.json()
                except Exception:
                    data = None
                if circuit_breaker:
                    await circuit_breaker.record_success()
                return data
        except (asyncio.TimeoutError, ClientConnectorError) as e:
            logger.debug(f"Fetch attempt {attempt} error for {url}: {e}")
            if attempt == retries:
                logger.warning(f"Failed to fetch {url} after {retries} attempts.")
                if circuit_breaker:
                    await circuit_breaker.record_failure()
                return None
            await asyncio.sleep(backoff * (attempt ** 1.2) + random.uniform(0, 0.2))
        except ClientResponseError as cre:
            logger.debug(f"ClientResponseError: {cre}")
            if circuit_breaker:
                await circuit_breaker.record_failure()
            return None
        except Exception as e:
            logger.exception(f"Unexpected fetch error for {url}: {e}")
            if circuit_breaker:
                await circuit_breaker.record_failure()
            return None
    return None

# -------------------------
# DataFetcher with Shared aiohttp.ClientSession
# -------------------------
class DataFetcher:
    def __init__(self, session: aiohttp.ClientSession, config: BotConfig):
        self.config = config
        self.session = session # Store the shared session
        self.api_base = config.DELTA_API_BASE.rstrip("/")
        self.semaphore = asyncio.Semaphore(config.MAX_PARALLEL_FETCH)
        self.timeout = config.HTTP_TIMEOUT
        self._cache: Dict[str, Tuple[float, Any]] = {}
        self._cache_max_size = 50
        self.circuit_breaker = CircuitBreaker()
        self.rate_limiter = RateLimitedFetcher(max_per_minute=60)

    def _clean_cache(self):
        """Remove oldest cache entries if cache gets too large"""
        if len(self._cache) > self._cache_max_size:
            # Remove oldest 20% of entries
            to_remove = sorted(self._cache.keys(), 
                             key=lambda k: self._cache[k][0])[:self._cache_max_size//5]
            for key in to_remove:
                del self._cache[key]

    async def fetch_products(self) -> Optional[dict]: # Removed session argument
        url = f"{self.api_base}/v2/products"
        async with self.semaphore:
            return await self.rate_limiter.call(
                async_fetch_json,
                self.session, url, # Use self.session
                retries=self.config.CANDLE_FETCH_RETRIES,
                backoff=self.config.CANDLE_FETCH_BACKOFF,
                timeout=self.timeout,
                circuit_breaker=self.circuit_breaker
            )

    async def fetch_candles(self, symbol: str, resolution: str, limit: int): # Removed session argument
        key = f"candles:{symbol}:{resolution}:{limit}"
        if key in self._cache:
            age, data = self._cache[key]
            if time.time() - age < 60:
                return data

        await asyncio.sleep(random.uniform(self.config.JITTER_MIN, self.config.JITTER_MAX))
        url = f"{self.api_base}/v2/chart/history"
        params = {
            "resolution": resolution,
            "symbol": symbol,
            "from": int(time.time()) - (limit * (int(resolution) if resolution != 'D' else 1440) * 60),
            "to": int(time.time())
        }
        async with self.semaphore:
            data = await self.rate_limiter.call(
                async_fetch_json,
                self.session, url, params=params, # Use self.session
                retries=self.config.CANDLE_FETCH_RETRIES,
                backoff=self.config.CANDLE_FETCH_BACKOFF,
                timeout=self.timeout,
                circuit_breaker=self.circuit_breaker
            )
        self._cache[key] = (time.time(), data)
        self._clean_cache()
        return data

# -------------------------
# Candle parsing & validation with Memory Optimization
# -------------------------
def parse_candles_result(result: dict) -> Optional[pd.DataFrame]:
    if not result or not isinstance(result, dict):
        return None
    if not result.get("success", True) and "result" not in result:
        return None
    res = result.get("result", {}) or {}
    required_keys = ['t', 'o', 'h', 'l', 'c', 'v']
    if not all(k in res for k in required_keys):
        return None
    for k in required_keys:
        if not isinstance(res[k], list):
            return None
    try:
        min_len = min(len(res[k]) for k in required_keys)
        df = pd.DataFrame({
            "timestamp": res['t'][:min_len],
            "open": res['o'][:min_len],
            "high": res['h'][:min_len],
            "low": res['l'][:min_len],
            "close": res['c'][:min_len],
            "volume": res['v'][:min_len]
        })
        df = df.sort_values('timestamp').drop_duplicates(subset='timestamp').reset_index(drop=True)
        
        # Optimize DataFrame memory usage
        if not df.empty:
            # Downcast numeric columns to save memory
            for col in ['open', 'high', 'low', 'close']:
                df[col] = pd.to_numeric(df[col], downcast='float', errors='coerce')
            df['volume'] = pd.to_numeric(df['volume'], downcast='float', errors='coerce')
            df['timestamp'] = pd.to_numeric(df['timestamp'], downcast='integer', errors='coerce')
        
        if df.empty:
            return None
        if float(df['close'].iloc[-1]) <= 0:
            return None
        return df
    except Exception as e:
        logger.exception(f"Failed to parse candles: {e}")
        return None

# -------------------------
# Health Check
# -------------------------
async def health_check() -> bool:
    """Verify API connectivity before full run"""
    try:
        async with aiohttp.ClientSession() as session:
            # Quick API health check
            async with session.get(f"{cfg.DELTA_API_BASE}/v2/products", 
                                 timeout=10) as resp:
                return resp.status == 200
    except Exception as e:
        logger.warning(f"Health check failed: {e}")
        return False

# -------------------------
# Resource Cleanup
# -------------------------
async def cleanup_resources():
    """Explicit cleanup for Cron-jobs.org environment"""
    # Force garbage collection
    gc.collect()
    
    # Clear pandas caches (small optimization)
    if 'pd' in globals():
        pd.DataFrame().empty

# -------------------------
# Indicators (content remains the same, uses cfg)
# -------------------------
def calculate_ema(data: pd.Series, period: int) -> pd.Series:
    return data.ewm(span=period, adjust=False).mean()

def calculate_sma(data: pd.Series, period: int) -> pd.Series:
    return data.rolling(window=period, min_periods=max(2, period//3)).mean()

def calculate_rma(data: pd.Series, period: int) -> pd.Series:
    r = data.ewm(alpha=1/period, adjust=False).mean()
    return r.bfill().ffill()

def calculate_ppo(df: pd.DataFrame, fast: int, slow: int, signal: int, use_sma: bool = False) -> Tuple[pd.Series, pd.Series]:
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
    sd = close.rolling(window=50, min_periods=50).std() * responsiveness
    sd = sd.bfill().ffill().fillna(0.001).clip(lower=1e-6)

    worm = close.copy()
    for i in range(1, n):
        diff = close.iloc[i] - worm.iloc[i - 1]
        delta = np.sign(diff) * sd.iloc[i] if abs(diff) > sd.iloc[i] else diff
        worm.iloc[i] = worm.iloc[i - 1] + delta

    ma = close.rolling(window=period, min_periods=period).mean().bfill().ffill()
    denom = worm.replace(0, np.nan).bfill().ffill().clip(lower=1e-8)

    raw_momentum = ((worm - ma).fillna(0)) / denom
    raw_momentum = raw_momentum.replace([np.inf, -np.inf], 0).fillna(0)

    min_med = raw_momentum.rolling(window=period, min_periods=period).min().bfill().ffill()
    max_med = raw_momentum.rolling(window=period, min_periods=period).max().bfill().ffill()
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

# -------------------------
# Closed-candle indexing
# -------------------------
def get_last_closed_index(df: Optional[pd.DataFrame], resolution_min: int) -> Optional[int]:
    if df is None or df.empty:
        return None
    now_ts = int(time.time())
    last_ts = int(df['timestamp'].iloc[-1])
    current_interval_start = now_ts - (now_ts % (resolution_min * 60))
    if last_ts >= current_interval_start:
        return len(df) - 2 if len(df) >= 2 else None
    return len(df) - 1

# -------------------------
# Evaluation logic
# -------------------------
def evaluate_pair_logic(pair_name: str, df_15m: pd.DataFrame, df_5m: pd.DataFrame,
                        last_state_for_pair: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    try:
        # Check for sufficient data
        last_i_15 = get_last_closed_index(df_15m, 15)
        last_i_5 = get_last_closed_index(df_5m, 5)
        if last_i_15 is None or last_i_15 < 3 or last_i_5 is None:
            logger.debug(f"Insufficient closed 15m/5m candles for {pair_name}")
            return None

        # --- Calculate Indicators ---
        magical_hist = calculate_magical_momentum_hist(df_15m)
        ppo, ppo_signal = calculate_ppo(df_15m, cfg.PPO_FAST, cfg.PPO_SLOW, cfg.PPO_SIGNAL, cfg.PPO_USE_SMA)
        rma_50 = calculate_rma(df_15m['close'], cfg.RMA_50_PERIOD)
        rma_200 = calculate_rma(df_5m['close'], cfg.RMA_200_PERIOD)
        upw, dnw, _, _ = calculate_cirrus_cloud(df_15m)
        smooth_rsi = calculate_smooth_rsi(df_15m, cfg.SRSI_RSI_LEN, cfg.SRSI_KALMAN_LEN)

        # Verify series lengths
        for series_name, s, idx in [
            ("ppo", ppo, last_i_15), 
            ("ppo_signal", ppo_signal, last_i_15), 
            ("rma_50", rma_50, last_i_15), 
            ("rma_200", rma_200, last_i_5), 
            ("smooth_rsi", smooth_rsi, last_i_15),
            ("magical_hist", magical_hist, last_i_15)
        ]:
            if s is None or len(s) <= idx:
                logger.debug(f"{series_name} not ready for {pair_name}")
                return None
        
        # --- Get current and previous values ---
        ppo_curr = float(ppo.iloc[last_i_15])
        ppo_prev = float(ppo.iloc[last_i_15 - 1])
        ppo_signal_curr = float(ppo_signal.iloc[last_i_15])
        ppo_signal_prev = float(ppo_signal.iloc[last_i_15 - 1])
        smooth_rsi_curr = float(smooth_rsi.iloc[last_i_15])
        smooth_rsi_prev = float(smooth_rsi.iloc[last_i_15 - 1])
        close_curr = float(df_15m['close'].iloc[last_i_15])
        open_curr = float(df_15m['open'].iloc[last_i_15])
        high_curr = float(df_15m['high'].iloc[last_i_15])
        low_curr = float(df_15m['low'].iloc[last_i_15])
        rma50_curr = float(rma_50.iloc[last_i_15])
        rma200_curr = float(rma_200.iloc[last_i_5])
        
        mmh_curr = float(magical_hist.iloc[last_i_15])
        mmh_prev1 = float(magical_hist.iloc[last_i_15 - 1])
        mmh_prev2 = float(magical_hist.iloc[last_i_15 - 2])
        mmh_prev3 = float(magical_hist.iloc[last_i_15 - 3])

        indicators = [ppo_curr, ppo_prev, ppo_signal_curr, ppo_signal_prev, smooth_rsi_curr, smooth_rsi_prev, rma50_curr, rma200_curr, mmh_curr, mmh_prev1, mmh_prev2, mmh_prev3]
        if any(pd.isna(x) for x in indicators):
            logger.debug(f"NaN in indicators for {pair_name}, skipping")
            return None

        # --- Conditions ---
        total_range = high_curr - low_curr
        strong_bullish_close = close_curr > high_curr - (total_range * 0.25)
        strong_bearish_close = close_curr < low_curr + (total_range * 0.25)
        price = close_curr
        formatted_time = datetime.fromtimestamp(df_15m['timestamp'].iloc[last_i_15]).strftime("%H:%M:%S UTC")

        ppo_above_signal = ppo_curr > ppo_signal_curr and ppo_prev < ppo_signal_prev
        ppo_below_signal = ppo_curr < ppo_signal_curr and ppo_prev > ppo_signal_prev
        ppo_cross_above_zero = ppo_curr > 0 and ppo_prev < 0
        ppo_cross_below_zero = ppo_curr < 0 and ppo_prev > 0
        ppo_cross_above_011 = ppo_curr > 0.11 and ppo_prev <= 0.11
        ppo_cross_below_minus011 = ppo_curr < -0.11 and ppo_prev >= -0.11
        ppo_below_030 = ppo_curr < 0.30
        ppo_above_minus030 = ppo_curr > -0.30

        close_above_rma50 = close_curr > rma50_curr
        close_below_rma50 = close_curr < rma50_curr
        close_above_rma200 = close_curr > rma200_curr
        close_below_rma200 = close_curr < rma200_curr
        
        # MMH Reversal logic
        mmh_reversal_buy = (
            mmh_prev3 < mmh_prev2 and mmh_prev2 < mmh_prev1 and mmh_curr > mmh_prev1
        )
        mmh_reversal_sell = (
            mmh_prev3 > mmh_prev2 and mmh_prev2 > mmh_prev1 and mmh_curr < mmh_prev1
        )
        srsi_cross_up_50 = (smooth_rsi_prev <= 50) and (smooth_rsi_curr > 50)
        srsi_cross_down_50 = (smooth_rsi_prev >= 50) and (smooth_rsi_curr < 50)

        cloud_state = "neutral"
        if cfg.CIRRUS_CLOUD_ENABLED:
            cloud_state = ("green" if (bool(upw.iloc[last_i_15]) and not bool(dnw.iloc[last_i_15])) else "red" if (bool(dnw.iloc[last_i_15]) and not bool(upw.iloc[last_i_15])) else "neutral")
        
        bot_name = cfg.BOT_NAME

        # Conditions dicts
        buy_mmh_reversal_conds = {
            "mmh_reversal_buy": mmh_reversal_buy,
            "close_above_rma50": close_above_rma50,
            "close_above_rma200": close_above_rma200,
            "magical_hist_curr>0": mmh_curr > 0,
            "cloud_green": (cloud_state == "green"),
            "strong_bullish_close": strong_bullish_close,
        }
        sell_mmh_reversal_conds = {
            "mmh_reversal_sell": mmh_reversal_sell,
            "close_below_rma50": close_below_rma50,
            "close_below_rma200": close_below_rma200,
            "magical_hist_curr<0": mmh_curr < 0,
            "cloud_red": (cloud_state == "red"),
            "strong_bearish_close": strong_bearish_close,
        }
        buy_srsi_conds = {
            "srsi_cross_up_50": srsi_cross_up_50,
            "ppo_above_signal": ppo_above_signal,
            "ppo_below_030": ppo_below_030,
            "close_above_rma50": close_above_rma50,
            "close_above_rma200": close_above_rma200,
            "cloud_green": (cloud_state == "green"),
            "strong_bullish_close": strong_bullish_close,
            "magical_hist_curr>0": mmh_curr > 0,
        }
        sell_srsi_conds = {
            "srsi_cross_down_50": srsi_cross_down_50,
            "ppo_below_signal": ppo_below_signal,
            "ppo_above_minus030": ppo_above_minus030,
            "close_below_rma50": close_below_rma50,
            "close_below_rma200": close_below_rma200,
            "cloud_red": (cloud_state == "red"),
            "strong_bearish_close": strong_bearish_close,
            "magical_hist_curr<0": mmh_curr < 0,
        }
        long0_conds = {
            "ppo_cross_above_zero": ppo_cross_above_zero,
            "ppo_above_signal": ppo_above_signal,
            "close_above_rma50": close_above_rma50,
            "close_above_rma200": close_above_rma200,
            "cloud_green": (cloud_state == "green"),
        }
        long011_conds = {
            "ppo_cross_above_011": ppo_cross_above_011,
            "ppo_above_signal": ppo_above_signal,
            "close_above_rma50": close_above_rma50,
            "close_above_rma200": close_above_rma200,
            "cloud_green": (cloud_state == "green"),
        }
        short0_conds = {
            "ppo_cross_below_zero": ppo_cross_below_zero,
            "ppo_below_signal": ppo_below_signal,
            "close_below_rma50": close_below_rma50,
            "close_below_rma200": close_below_rma200,
            "cloud_red": (cloud_state == "red"),
        }
        short011_conds = {
            "ppo_cross_below_minus011": ppo_cross_below_minus011,
            "ppo_below_signal": ppo_below_signal,
            "close_below_rma50": close_below_rma50,
            "close_below_rma200": close_below_rma200,
            "cloud_red": (cloud_state == "red"),
        }
        
        # --- Decision Logic (Prioritized) ---
        current_state = None
        send_message = None

        if all(buy_mmh_reversal_conds.values()):
            current_state = "long_mmh_reversal"
            send_message = f"{bot_name}\n🟢 {pair_name} - MMH REVERSAL LONG\nPrice: ${price:,.2f}\n{formatted_time}"
        elif all(sell_mmh_reversal_conds.values()):
            current_state = "short_mmh_reversal"
            send_message = f"{bot_name}\n🔴 {pair_name} - MMH REVERSAL SHORT\nPrice: ${price:,.2f}\n{formatted_time}"
        elif all(buy_srsi_conds.values()):
            current_state = "long_srsi"
            send_message = f"{bot_name}\n🟢 {pair_name} - SRSI LONG\nSRSI crossing above 50 ({smooth_rsi_curr:.2f})\nPrice: ${price:,.2f}\n{formatted_time}"
        elif all(sell_srsi_conds.values()):
            current_state = "short_srsi"
            send_message = f"{bot_name}\n🔴 {pair_name} - SRSI SHORT\nSRSI crossing below 50 ({smooth_rsi_curr:.2f})\nPrice: ${price:,.2f}\n{formatted_time}"
        elif all(long0_conds.values()):
            current_state = "long_zero"
            send_message = f"{bot_name}\n🟢 {pair_name} - LONG\nPPO crossing above 0 ({ppo_curr:.2f})\nPrice: ${price:,.2f}\n{formatted_time}"
        elif all(long011_conds.values()):
            current_state = "long_011"
            send_message = f"{bot_name}\n🟢 {pair_name} - LONG\nPPO crossing above 0.11 ({ppo_curr:.2f})\nPrice: ${price:,.2f}\n{formatted_time}"
        elif all(short0_conds.values()):
            current_state = "short_zero"
            send_message = f"{bot_name}\n🔴 {pair_name} - SHORT\nPPO crossing below 0 ({ppo_curr:.2f})\nPrice: ${price:,.2f}\n{formatted_time}"
        elif all(short011_conds.values()):
            current_state = "short_011"
            send_message = f"{bot_name}\n🔴 {pair_name} - SHORT\nPPO crossing below -0.11 ({ppo_curr:.2f})\nPrice: ${price:,.2f}\n{formatted_time}"
        
        now_ts_int = int(time.time())
        last_state_value = last_state_for_pair.get("state") if isinstance(last_state_for_pair, dict) else None

        if current_state is None:
            if last_state_value and last_state_value != 'NO_SIGNAL':
                return {"state": "NO_SIGNAL", "ts": now_ts_int}
            return last_state_for_pair

        if current_state == last_state_value:
            logger.debug(f"Idempotency: {pair_name} signal remains {current_state}.")
            return last_state_for_pair

        return {"state": current_state, "ts": now_ts_int, "message": send_message}

    except Exception as e:
        logger.exception(f"Error evaluating logic for {pair_name}: {e}")
        return None

# -------------------------
# Product mapping 
# -------------------------
def build_products_map_from_api_result(api_products: dict) -> Dict[str, dict]:
    products_map = {}
    if not api_products or not api_products.get("result"):
        logger.error("No products in API result")
        return products_map
    for p in api_products.get("result", []):
        try:
            symbol = p.get("symbol", "")
            # Normalize symbol names (e.g., BTC_USDT -> BTCUSD)
            symbol_norm = symbol.replace("_USDT", "USD").replace("USDT", "USD").replace("_", "")
            
            if p.get("contract_type") == "perpetual_futures":
                for pair_name in cfg.PAIRS:
                    if symbol_norm == pair_name:
                        products_map[pair_name] = {
                            "id": p.get("id"),
                            "symbol": p.get("symbol"),
                            "contract_type": p.get("contract_type")
                        }
                        break
        except Exception:
            continue
    logger.info(f"Mapped {len(products_map)} tradable pairs")
    return products_map

# -------------------------
# Telegram queue with Shared aiohttp.ClientSession
# -------------------------
class TelegramQueue:
    def __init__(self, session: aiohttp.ClientSession, config: BotConfig, rate_limit: float = 0.1):
        self.session = session # Store the shared session
        self.token = config.TELEGRAM_BOT_TOKEN
        self.chat_id = config.TELEGRAM_CHAT_ID
        self.config = config
        self.rate_limit = rate_limit
        self._last_send_time = 0.0
        self._send_lock = asyncio.Lock()
        
    async def _send_once(self, message: str) -> bool:
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "Markdown",
            "disable_web_page_preview": "true",
        }
        try:
            # Use self.session
            async with self.session.post(url, data=payload, timeout=5) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get("ok"):
                        return True
                    logger.error(f"Telegram API returned non-ok: {data}")
                else:
                    text = await resp.text()
                    logger.error(f"Telegram HTTP Error {resp.status}: {text[:100]}")
        except Exception as e:
            logger.error(f"Telegram send exception: {e}")
        return False

    async def _send_with_retries(self, message: str) -> bool:
        last_exc = None
        for attempt in range(1, max(1, self.config.TELEGRAM_RETRIES) + 1):
            ok = await self._send_once(message)
            if ok:
                return True
            last_exc = "Telegram non-ok or exception"
            # Backoff formula: base^(attempt-1) + jitter
            await asyncio.sleep((self.config.TELEGRAM_BACKOFF_BASE ** (attempt - 1)) + random.uniform(0, 0.3))
        logger.error(f"Telegram send failed after retries: {last_exc}")
        return False

    async def send(self, message: str) -> bool: # Removed session argument
        async with self._send_lock:
            # Enforce rate limit
            now = time.time()
            elapsed = now - self._last_send_time
            if elapsed < self.rate_limit:
                await asyncio.sleep(self.rate_limit - elapsed)
            
            result = await self._send_with_retries(message)
            self._last_send_time = time.time()
            return result

# -------------------------
# Check single pair (fetch, evaluate, alert)
# -------------------------
async def check_pair(fetcher: DataFetcher, products_map: Dict[str, dict], pair_name: str, 
                     last_state_for_pair: Optional[Dict[str, Any]], telegram_queue: TelegramQueue) -> Optional[Tuple[str, Dict[str, Any]]]:
    try:
        prod = products_map.get(pair_name)
        if not prod:
            logger.debug(f"No product mapping for {pair_name}")
            return None
            
        # Use config (now cfg) for limits
        sp = cfg.SPECIAL_PAIRS.get(pair_name, {})
        limit_15m = sp.get("limit_15m", 210)
        min_required = sp.get("min_required", 180)
        limit_5m = sp.get("limit_5m", 300)
        min_required_5m = sp.get("min_required_5m", 200)
        symbol = prod["symbol"]

        # Fetch candles concurrently. Removed session argument.
        res15, res5 = await asyncio.gather(
            fetcher.fetch_candles(symbol, "15", limit_15m),
            fetcher.fetch_candles(symbol, "5", limit_5m)
        )

        df_15m = parse_candles_result(res15)
        df_5m = parse_candles_result(res5)

        if df_15m is None or len(df_15m) < (min_required + 2):
            logger.warning(f"Insufficient 15m data for {pair_name}: {len(df_15m) if df_15m is not None else 0}/{min_required+2}")
            return None
        if df_5m is None or len(df_5m) < (min_required_5m + 2):
            logger.warning(f"Insufficient 5m data for {pair_name}: {len(df_5m) if df_5m is not None else 0}/{min_required_5m+2}")
            return None
        
        # Ensure all columns are numeric (robustness check)
        for df in (df_15m, df_5m):
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        new_state = evaluate_pair_logic(pair_name, df_15m, df_5m, last_state_for_pair)

        if not new_state:
            return None

        state_value = new_state.get("state")
        
        if state_value and "message" in new_state:
            logger.info(f"ALERT: {new_state['message']}")
            await telegram_queue.send(new_state['message'])
        
        return pair_name, new_state

    except Exception as e:
        logger.exception(f"Unhandled error in check_pair for {pair_name}: {e}")
        return None

# -------------------------
# Main runner
# -------------------------
async def run_once():
    pid_lock = PidFileLock(cfg.PID_LOCK_PATH) # Use configurable path

    if not pid_lock.acquire():
        logger.warning("Bot already running, exiting new run")
        return 

    # Memory Guard
    try:
        p = psutil.Process(os.getpid())
        rss = p.memory_info().rss
        if rss > cfg.MEMORY_LIMIT_BYTES:
            logger.critical(f"High memory usage at start: {rss} > {cfg.MEMORY_LIMIT_BYTES}")
            raise SystemExit(3)
    except Exception:
        pass

    # Prune old states
    with StateDB(cfg.STATE_DB_PATH) as sdb:
        deleted = sdb.prune_old_records(cfg.STATE_EXPIRY_DAYS)
        if deleted > 0:
            logger.info(f"Pruned {deleted} old state records")
        last_alerts = sdb.load_all()
    logger.info(f"Loaded {len(last_alerts)} previous states")
    
    # Use TCPConnector for session connection limiting
    connector = TCPConnector(limit=cfg.TCP_CONN_LIMIT, ssl=False)
    
    # Shared ClientSession Context Manager
    async with aiohttp.ClientSession(connector=connector) as session:
        # Initialize resources with the shared session
        fetcher = DataFetcher(session, cfg)
        telegram_queue = TelegramQueue(session, cfg)
        
        if cfg.SEND_TEST_MESSAGE:
            ist = pytz.timezone("Asia/Kolkata")
            current_dt = datetime.now(ist)
            test_msg = (f"🔔 {cfg.BOT_NAME} Started\n"
                        f"Time: {current_dt.strftime('%d-%m-%Y @ %H:%M IST')}\n"
                        f"Pairs: {len(cfg.PAIRS)} | Debug: {cfg.DEBUG_MODE}")
            await telegram_queue.send(test_msg) # Removed session argument
        
        logger.info("Fetching products from API...")
        prod_resp = await fetcher.fetch_products() # Removed session argument
        
        if not prod_resp:
            logger.error("Failed to fetch products; aborting run.")
            raise SystemExit(2)
        
        products_map = build_products_map_from_api_result(prod_resp)
        if not products_map:
            logger.error("No tradable pairs found; exiting.")
            raise SystemExit(2)

        pairs_to_process = [p for p in cfg.PAIRS if p in products_map]
        batch_size = max(1, cfg.BATCH_SIZE)
        logger.info(f"Processing {len(pairs_to_process)} pairs in batches of {batch_size}")
        
        all_results: List[Tuple[str, Dict[str, Any]]] = []
        
        # Process in batches for better concurrency management
        for i in range(0, len(pairs_to_process), batch_size):
            batch = pairs_to_process[i:i + batch_size]
            logger.debug(f"Processing batch: {batch}")
            
            tasks = []
            for pair in batch:
                tasks.append(
                    check_pair(
                        fetcher, products_map, pair, 
                        last_alerts.get(pair), telegram_queue
                    )
                )
            
            # Execute tasks and filter out None results (errors/no-data)
            batch_results = await asyncio.gather(*tasks)
            all_results.extend([r for r in batch_results if r is not None])
            
            # Yield control between batches
            await asyncio.sleep(random.uniform(0.1, 0.5))

        # Save all new states in a single transaction
        new_states = {pair: state for pair, state in all_results}
        with StateDB(cfg.STATE_DB_PATH) as sdb:
            for pair, state_info in new_states.items():
                sdb.set(pair, state_info["state"], state_info["ts"])
            logger.info(f"Saved/Updated {len(new_states)} pair states.")

    # Final cleanup after run
    await cleanup_resources()
    
# -------------------------
# Run wrapper with timeout (crash-free)
# -------------------------
async def run_once_with_timeout(timeout_seconds: int = 600):
    try:
        await asyncio.wait_for(run_once(), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        logger.error(f"Run timed out after {timeout_seconds}s")
        raise SystemExit(3)

# -------------------------
# SIG handling & graceful shutdown
# -------------------------
stop_requested = False
def request_stop(signum, frame):
    global stop_requested
    logger.info(f"Received signal {signum} — will stop after current run.")
    stop_requested = True
signal.signal(signal.SIGINT, request_stop)
signal.signal(signal.SIGTERM, request_stop)

# -------------------------
# CLI Main
# -------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="MACD/PPO Bot - Production Ready")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--once", action="store_true", help="Run the bot once and exit (default mode)")
    group.add_argument("--loop", type=int, help="Run the bot in a loop every N seconds (e.g., --loop 60)")
    args = parser.parse_args()
    
    # Check for critical configuration before starting
    try:
        cfg.check_secrets(cfg.TELEGRAM_BOT_TOKEN)
        cfg.check_secrets(cfg.TELEGRAM_CHAT_ID)
        cfg.check_secrets(cfg.DELTA_API_BASE)
        if cfg.TELEGRAM_BOT_TOKEN == 'xxxx' or cfg.TELEGRAM_CHAT_ID == 'xxxx':
            if not cfg.DEBUG_MODE or cfg.SEND_TEST_MESSAGE:
                 raise ValueError("Critical config missing")
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        sys.exit(2)

    pid_lock = PidFileLock(cfg.PID_LOCK_PATH)
    
    try:
        if args.once:
            try:
                # Use a simple run for 'once' mode compatible with cronjob setups
                asyncio.run(run_once_with_timeout(cfg.RUN_TIMEOUT_SECONDS))
            except SystemExit as e:
                logger.error(f"Exited with code {e.code}")
                sys.exit(e.code if isinstance(e.code, int) else 1)
        elif args.loop:
            interval = max(30, args.loop)
            logger.info(f"Starting loop mode with interval={interval}s")
            while not stop_requested:
                start = time.time()
                try:
                    asyncio.run(run_once_with_timeout(cfg.RUN_TIMEOUT_SECONDS))
                except SystemExit as e:
                    logger.error(f"Run exited with code {e.code}; continuing loop")
                except Exception:
                    logger.exception("Unhandled exception in run loop")
                elapsed = time.time() - start
                to_sleep = max(0, interval - elapsed)
                slept = 0.0
                while slept < to_sleep and not stop_requested:
                    time.sleep(min(1.0, to_sleep - slept))
                    slept += 1.0
        else:
            # Default to once if no arguments are provided (cronjob compatibility)
            try:
                asyncio.run(run_once_with_timeout(cfg.RUN_TIMEOUT_SECONDS))
            except SystemExit as e:
                logger.error(f"Exited with code {e.code}")
                sys.exit(e.code if isinstance(e.code, int) else 1)
    finally:
        pid_lock.release()

if __name__ == "__main__":
    main()
