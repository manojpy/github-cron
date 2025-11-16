#!/usr/bin/env python3
# macd_bot_improved_FULL.py
# Full corrected and improved MACD/PPO alert bot
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
from typing import Dict, Any, Optional, Tuple, List, ClassVar, get_type_hints
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import pandas as pd
import numpy as np
import pytz
import psutil
from pydantic import BaseModel, Field, field_validator
from aiohttp import ClientConnectorError, ClientResponseError, TCPConnector
from logging.handlers import RotatingFileHandler

# -------------------------
# Pydantic v2 Configuration Model
# -------------------------
class BotConfig(BaseModel):
    TELEGRAM_BOT_TOKEN: str = Field(..., min_length=1)
    TELEGRAM_CHAT_ID: str = Field(..., min_length=1)
    DEBUG_MODE: bool = False
    SEND_TEST_MESSAGE: bool = False
    BOT_NAME: str = "MACD Alert Bot"
    DELTA_API_BASE: str = "https://api.india.delta.exchange"
    PAIRS: List[str] = Field(..., min_length=1)
    SPECIAL_PAIRS: Dict[str, Dict[str, int]] = {
        "SOLUSD": {"limit_15m": 210, "min_required": 180, "limit_5m": 300, "min_required_5m": 200}
    }
    PPO_FAST: int = 7
    PPO_SLOW: int = 16
    PPO_SIGNAL: int = 5
    PPO_USE_SMA: bool = False
    RMA_50_PERIOD: int = 50
    RMA_200_PERIOD: int = 200
    CIRRUS_CLOUD_ENABLED: bool = True
    X1: int = 22
    X2: int = 9
    X3: int = 15
    X4: int = 5
    SRSI_RSI_LEN: int = 21
    SRSI_KALMAN_LEN: int = 5
    STATE_DB_PATH: str = "macd_state.sqlite"
    LOG_FILE: str = "macd_bot.log"
    PID_LOCK_PATH: str = "/tmp/macd_bot.pid"
    MAX_PARALLEL_FETCH: int = Field(8, ge=1, le=16)
    HTTP_TIMEOUT: int = 15
    CANDLE_FETCH_RETRIES: int = 3
    CANDLE_FETCH_BACKOFF: float = 1.5
    JITTER_MIN: float = 0.1
    JITTER_MAX: float = 0.8
    RUN_TIMEOUT_SECONDS: int = 600
    BATCH_SIZE: int = 4
    TCP_CONN_LIMIT: int = 8
    TELEGRAM_RETRIES: int = 3
    TELEGRAM_BACKOFF_BASE: float = 2.0
    MEMORY_LIMIT_BYTES: int = 400000000
    STATE_EXPIRY_DAYS: int = 30
    DEAD_MANS_COOLDOWN_SECONDS: int = 14400
    LOG_LEVEL: str = "INFO"

    @field_validator('LOG_LEVEL')
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'LOG_LEVEL must be one of {valid_levels}')
        return v.upper()

    @field_validator('PAIRS')
    @classmethod
    def validate_pairs(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError('PAIRS cannot be empty')
        return v

# -------------------------
# Configuration loader
# -------------------------
def load_config() -> BotConfig:
    CONFIG_FILE = os.getenv("CONFIG_FILE", "config_macd.json")
    config_data = {}
    if Path(CONFIG_FILE).exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                config_data = json.load(f)
                print(f"✅ Loaded configuration from {CONFIG_FILE}")
        except Exception as e:
            print(f"❌ Error parsing config file: {e}")
            sys.exit(1)
    else:
        print(f"❌ Config file not found: {CONFIG_FILE}")
        sys.exit(1)
    try:
        return BotConfig(**config_data)
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        sys.exit(1)

cfg = load_config()

# -------------------------
# Logging
# -------------------------
def setup_logging():
    logger = logging.getLogger("macd_bot")
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    log_level = getattr(logging, cfg.LOG_LEVEL, logging.INFO)
    if cfg.DEBUG_MODE:
        log_level = logging.DEBUG
    logger.setLevel(log_level)
    formatter = logging.Formatter(
        fmt='%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    try:
        log_dir = os.path.dirname(cfg.LOG_FILE) or "."
        os.makedirs(log_dir, exist_ok=True)
        file_handler = RotatingFileHandler(
            cfg.LOG_FILE,
            maxBytes=10_000_000,
            backupCount=5,
            encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"⚠️ Warning: Could not set up file logging: {e}")
    return logger

logger = setup_logging()

# -------------------------
# Session Manager
# -------------------------
class SessionManager:
    _session: ClassVar[Optional[aiohttp.ClientSession]] = None

    @classmethod
    def get_session(cls) -> aiohttp.ClientSession:
        if cls._session is None or cls._session.closed:
            connector = TCPConnector(
                limit=cfg.TCP_CONN_LIMIT,
                ssl=False,
                force_close=True,
                enable_cleanup_closed=True
            )
            timeout = aiohttp.ClientTimeout(total=cfg.HTTP_TIMEOUT)
            cls._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={'User-Agent': f'{cfg.BOT_NAME}/1.0'}
            )
            logger.debug("Created new shared aiohttp session")
        return cls._session

    @classmethod
    async def close_session(cls):
        if cls._session and not cls._session.closed:
            await cls._session.close()
            cls._session = None
            logger.debug("Closed shared aiohttp session")

# -------------------------
# PID Lock
# -------------------------
class PidFileLock:
    def __init__(self):
        self.path = cfg.PID_LOCK_PATH
        self.fd = None
        self.acquired = False

    def acquire(self) -> bool:
        try:
            lock_dir = os.path.dirname(self.path)
            if lock_dir and not os.path.exists(lock_dir):
                os.makedirs(lock_dir, exist_ok=True)
            self.fd = open(self.path, "w")
            fcntl.flock(self.fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.fd.write(str(os.getpid()))
            self.fd.flush()
            atexit.register(self.release)
            self.acquired = True
            logger.debug(f"Acquired PID lock: {self.path}")
            return True
        except (IOError, OSError):
            if self.fd:
                try:
                    self.fd.close()
                except Exception:
                    pass
            logger.warning(f"Could not acquire PID lock {self.path} - another instance may be running")
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
                except Exception:
                    pass
            self.acquired = False
            logger.debug("Released PID lock")
        except Exception as e:
            logger.warning(f"Error releasing PID lock: {e}")

# -------------------------
# State DB
# -------------------------
class StateDB:
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

# -------------------------
# Circuit Breaker
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
# RateLimitedFetcher
# -------------------------
class RateLimitedFetcher:
    def __init__(self, max_per_minute: int = 60):
        self.semaphore = asyncio.Semaphore(max_per_minute)
        self.requests = []

    async def call(self, func, *args, **kwargs):
        async with self.semaphore:
            now = time.time()
            self.requests = [r for r in self.requests if now - r < 60]
            if len(self.requests) >= self.semaphore._value:
                await asyncio.sleep(1)
            self.requests.append(now)
            return await func(*args, **kwargs)

# -------------------------
# Async fetch helper
# -------------------------
async def async_fetch_json(url: str, params: dict = None,
                           retries: int = 3, backoff: float = 1.5, timeout: int = 15,
                           circuit_breaker: Optional[CircuitBreaker] = None) -> Optional[dict]:
    if circuit_breaker and await circuit_breaker.is_open():
        logger.warning(f"Circuit breaker open; skipping fetch {url}")
        return None
    session = SessionManager.get_session()
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
# DataFetcher
# -------------------------
class DataFetcher:
    def __init__(self, api_base: str, max_parallel: int = None):
        self.api_base = api_base.rstrip("/")
        max_parallel = max_parallel or cfg.MAX_PARALLEL_FETCH
        if max_parallel < 1 or max_parallel > 16:
            logger.warning(f"Invalid MAX_PARALLEL_FETCH: {max_parallel}, using default: 8")
            max_parallel = 8
        self.semaphore = asyncio.Semaphore(max_parallel)
        self.timeout = cfg.HTTP_TIMEOUT
        self._cache: Dict[str, Tuple[float, Any]] = {}
        self._cache_max_size = 50
        self.circuit_breaker = CircuitBreaker()
        self.rate_limiter = RateLimitedFetcher(max_per_minute=60)

    def _clean_cache(self):
        if len(self._cache) > self._cache_max_size:
            to_remove = sorted(self._cache.keys(),
                             key=lambda k: self._cache[k][0])[:self._cache_max_size//5]
            for key in to_remove:
                del self._cache[key]

    async def fetch_products(self) -> Optional[dict]:
        url = f"{self.api_base}/v2/products"
        async with self.semaphore:
            return await self.rate_limiter.call(
                async_fetch_json,
                url,
                retries=cfg.CANDLE_FETCH_RETRIES,
                backoff=cfg.CANDLE_FETCH_BACKOFF,
                timeout=self.timeout,
                circuit_breaker=self.circuit_breaker
            )

    async def fetch_candles(self, symbol: str, resolution: str, limit: int):
        key = f"candles:{symbol}:{resolution}:{limit}"
        if key in self._cache:
            age, data = self._cache[key]
            if time.time() - age < 60:
                return data
        await asyncio.sleep(random.uniform(cfg.JITTER_MIN, cfg.JITTER_MAX))
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
                url, params=params,
                retries=cfg.CANDLE_FETCH_RETRIES,
                backoff=cfg.CANDLE_FETCH_BACKOFF,
                timeout=self.timeout,
                circuit_breaker=self.circuit_breaker
            )
        self._cache[key] = (time.time(), data)
        self._clean_cache()
        return data

# -------------------------
# Candle parsing
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
        if not df.empty:
            for col in ['open', 'high', 'low', 'close']:
                df[col] = pd.to_numeric(df[col], downcast='float', errors='coerce')
            df['volume'] = pd.to_numeric(df['volume'], downcast='float', errors='coerce')
            df['timestamp'] = pd.to_numeric(df['timestamp'], downcast='integer', errors='coerce')
            try:
                max_ts = int(df['timestamp'].max())
                if max_ts > 1_000_000_000_000:
                    df['timestamp'] = (df['timestamp'] // 1000).astype(int)
                elif max_ts > 10_000_000_000:
                    df['timestamp'] = (df['timestamp'] // 1000).astype(int)
                else:
                    df['timestamp'] = df['timestamp'].astype(int)
            except Exception:
                df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce').fillna(0).astype(int)
        if df.empty:
            return None
        if float(df['close'].iloc[-1]) <= 0:
            return None
        return df
    except Exception as e:
        logger.exception(f"Failed to parse candles: {e}")
        return None

# -------------------------
# Health check
# -------------------------
async def health_check() -> bool:
    try:
        session = SessionManager.get_session()
        async with session.get(f"{cfg.DELTA_API_BASE}/v2/products", timeout=10) as resp:
            return resp.status == 200
    except Exception as e:
        logger.warning(f"Health check failed: {e}")
        return False

# -------------------------
# Cleanup
# -------------------------
async def cleanup_resources():
    gc.collect()
    if 'pd' in globals():
        pd.DataFrame().empty

# -------------------------
# Indicators (preserve logic)
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
    """
    Improved Magical Momentum Histogram with debug logging.
    Matches existing alert semantics (sign used in conditions).
    """
    n = len(df)
    if n == 0:
        return pd.Series([], dtype=float)
    if n < period + 50:
        return pd.Series(np.zeros(n), index=df.index, dtype=float)

    close = df['close'].astype(float).copy()

    # Rolling std scaled by responsiveness
    sd = close.rolling(window=50, min_periods=10).std(ddof=0)
    sd = (sd * max(0.00001, responsiveness)).bfill().ffill().clip(lower=1e-8)

    # Adaptive "worm"
    worm = close.copy()
    for i in range(1, n):
        diff = close.iloc[i] - worm.iloc[i - 1]
        thresh = sd.iloc[i]
        delta = np.sign(diff) * thresh if abs(diff) > thresh else diff
        worm.iloc[i] = worm.iloc[i - 1] + delta

    # Baseline MA
    ma = close.rolling(window=period, min_periods=max(5, period // 3)).mean().bfill().ffill()

    # Raw momentum normalized by worm
    denom = worm.replace(0, np.nan).bfill().ffill()
    raw_momentum = (worm - ma) / denom
    raw_momentum = raw_momentum.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Range normalization
    min_med = raw_momentum.rolling(window=period, min_periods=max(5, period // 3)).min().bfill().ffill()
    max_med = raw_momentum.rolling(window=period, min_periods=max(5, period // 3)).max().bfill().ffill()
    rng = (max_med - min_med).replace(0, np.nan)

    temp = pd.Series(0.0, index=df.index)
    valid = rng.notna()
    temp.loc[valid] = (raw_momentum.loc[valid] - min_med.loc[valid]) / rng.loc[valid]
    temp = temp.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)

    # Recursive value clamp
    value = pd.Series(0.0, index=df.index)
    for i in range(1, n):
        v = temp.iloc[i] - 0.5 + 0.5 * value.iloc[i - 1]
        value.iloc[i] = max(min(v, 0.9999), -0.9999)

    # Momentum transform
    temp2 = (1.0 + value) / (1.0 - value)
    temp2 = temp2.replace([np.inf, -np.inf], np.nan).fillna(1e-6).clip(lower=1e-6)
    momentum_val = 0.25 * np.log(temp2)

    # EMA-like recursion to produce histogram
    hist = pd.Series(0.0, index=df.index)
    for i in range(1, n):
        hist.iloc[i] = float(momentum_val.iloc[i]) + 0.5 * float(hist.iloc[i - 1])

    # Debug: last 4 and raw momentum snapshot
    if cfg.DEBUG_MODE and n >= 4:
        last4 = hist.iloc[-4:].tolist()
        logger.debug(
            f"MMH Debug: last4={[round(x, 6) for x in last4]}, "
            f"last={hist.iloc[-1]:.6f}, raw_last={momentum_val.iloc[-1]:.6f}"
        )

    return hist.replace([np.inf, -np.inf], 0.0).fillna(0.0)

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
        last_i_15 = get_last_closed_index(df_15m, 15)
        last_i_5 = get_last_closed_index(df_5m, 5)
        if last_i_15 is None or last_i_15 < 3 or last_i_5 is None:
            logger.debug(f"Indexing not ready for {pair_name} (last_i_15={last_i_15}, last_i_5={last_i_5})")
            return None

        magical_hist = calculate_magical_momentum_hist(df_15m, period=144, responsiveness=0.9)
        if magical_hist is None or len(magical_hist) <= last_i_15:
            logger.debug(f"MMH missing for {pair_name}")
            return None

        mmh_curr = float(magical_hist.iloc[last_i_15])
        mmh_prev1 = float(magical_hist.iloc[last_i_15 - 1])
        mmh_prev2 = float(magical_hist.iloc[last_i_15 - 2])
        mmh_prev3 = float(magical_hist.iloc[last_i_15 - 3])

        # MMH context debug: show last4 snapshot for visibility
        if cfg.DEBUG_MODE:
            mmh_last4 = magical_hist.iloc[last_i_15 - 3:last_i_15 + 1].tolist()
            logger.debug(
                f"{pair_name} MMH Status: curr={mmh_curr:.6f}, "
                f"prev1={mmh_prev1:.6f}, prev2={mmh_prev2:.6f}, prev3={mmh_prev3:.6f}, "
                f"last4={[round(x, 6) for x in mmh_last4]}"
            )

        ppo, ppo_signal = calculate_ppo(df_15m, cfg.PPO_FAST, cfg.PPO_SLOW, cfg.PPO_SIGNAL, cfg.PPO_USE_SMA)
        rma_50 = calculate_rma(df_15m['close'], cfg.RMA_50_PERIOD)
        rma_200 = calculate_rma(df_5m['close'], cfg.RMA_200_PERIOD)
        upw, dnw, _, _ = calculate_cirrus_cloud(df_15m)
        smooth_rsi = calculate_smooth_rsi(df_15m, cfg.SRSI_RSI_LEN, cfg.SRSI_KALMAN_LEN)

        for series_name, s, idx in [
            ("ppo", ppo, last_i_15), ("ppo_signal", ppo_signal, last_i_15),
            ("rma_50", rma_50, last_i_15), ("rma_200", rma_200, last_i_5),
            ("smooth_rsi", smooth_rsi, last_i_15)
        ]:
            if s is None or len(s) <= idx:
                logger.debug(f"{series_name} not ready for {pair_name}")
                return None

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

        indicators = [ppo_curr, ppo_prev, ppo_signal_curr, ppo_signal_prev,
                      smooth_rsi_curr, smooth_rsi_prev, rma50_curr, rma200_curr,
                      mmh_curr, mmh_prev1, mmh_prev2, mmh_prev3]
        if any(pd.isna(x) for x in indicators):
            logger.debug(f"NaN in indicators for {pair_name}, skipping")
            return None

        total_range = high_curr - low_curr
        if total_range <= 0:
            strong_bullish_close = strong_bearish_close = False
        else:
            upper_wick = max(0.0, high_curr - max(open_curr, close_curr))
            lower_wick = max(0.0, min(open_curr, close_curr) - low_curr)
            bullish_candle = close_curr > open_curr
            bearish_candle = close_curr < open_curr
            strong_bullish_close = bullish_candle and (upper_wick / total_range) < 0.20
            strong_bearish_close = bearish_candle and (lower_wick / total_range) < 0.20

        ppo_cross_above_zero = (ppo_prev <= 0) and (ppo_curr > 0)
        ppo_cross_below_zero = (ppo_prev >= 0) and (ppo_curr < 0)
        ppo_cross_above_011 = (ppo_prev <= 0.11) and (ppo_curr > 0.11)
        ppo_cross_below_minus011 = (ppo_prev >= -0.11) and (ppo_curr < -0.11)
        ppo_above_signal = ppo_curr > ppo_signal_curr
        ppo_below_signal = ppo_curr < ppo_signal_curr
        ppo_below_030 = ppo_curr < 0.30
        ppo_above_minus030 = ppo_curr > -0.30
        close_above_rma50 = close_curr > rma50_curr
        close_below_rma50 = close_curr < rma50_curr
        close_above_rma200 = close_curr > rma200_curr
        close_below_rma200 = close_curr < rma200_curr

        mmh_reversal_buy = (
            mmh_curr > 0 and
            mmh_prev2 < mmh_prev3 and
            mmh_prev1 < mmh_prev2 and
            mmh_curr > mmh_prev1
        )
        mmh_reversal_sell = (
            mmh_curr < 0 and
            mmh_prev2 > mmh_prev3 and
            mmh_prev1 > mmh_prev2 and
            mmh_curr < mmh_prev1
        )

        srsi_cross_up_50 = (smooth_rsi_prev <= 50) and (smooth_rsi_curr > 50)
        srsi_cross_down_50 = (smooth_rsi_prev >= 50) and (smooth_rsi_curr < 50)

        cloud_state = "neutral"
        if cfg.CIRRUS_CLOUD_ENABLED:
            cloud_state = ("green" if (bool(upw.iloc[last_i_15]) and not bool(dnw.iloc[last_i_15]))
                           else "red" if (bool(dnw.iloc[last_i_15]) and not bool(upw.iloc[last_i_15]))
                           else "neutral")

        bot_name = cfg.BOT_NAME

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
            "strong_bullish_close": strong_bullish_close,
            "magical_hist_curr>0": mmh_curr > 0,
        }
        long011_conds = {
            "ppo_cross_above_011": ppo_cross_above_011,
            "ppo_above_signal": ppo_above_signal,
            "close_above_rma50": close_above_rma50,
            "close_above_rma200": close_above_rma200,
            "cloud_green": (cloud_state == "green"),
            "strong_bullish_close": strong_bullish_close,
            "magical_hist_curr>0": mmh_curr > 0,
        }
        short0_conds = {
            "ppo_cross_below_zero": ppo_cross_below_zero,
            "ppo_below_signal": ppo_below_signal,
            "close_below_rma50": close_below_rma50,
            "close_below_rma200": close_below_rma200,
            "cloud_red": (cloud_state == "red"),
            "strong_bearish_close": strong_bearish_close,
            "magical_hist_curr<0": mmh_curr < 0,
        }
        short011_conds = {
            "ppo_cross_below_minus011": ppo_cross_below_minus011,
            "ppo_below_signal": ppo_below_signal,
            "close_below_rma50": close_below_rma50,
            "close_below_rma200": close_below_rma200,
            "cloud_red": (cloud_state == "red"),
            "strong_bearish_close": strong_bearish_close,
            "magical_hist_curr<0": mmh_curr < 0,
        }

        current_state = None
        send_message = None
        price = close_curr
        ist = pytz.timezone("Asia/Kolkata")
        formatted_time = datetime.now(ist).strftime('%d-%m-%Y @ %H:%M IST')

        if all(buy_mmh_reversal_conds.values()):
            current_state = "buy_mmh_reversal"
            send_message = f"{bot_name}\n🟢 {pair_name} - BUY (MMH Reversal)\nMMH 15m Reversal Up ({mmh_curr:.5f})\nPrice: ${price:,.2f}\n{formatted_time}"
        elif all(sell_mmh_reversal_conds.values()):
            current_state = "sell_mmh_reversal"
            send_message = f"{bot_name}\n🔴 {pair_name} - SELL (MMH Reversal)\nMMH 15m Reversal Down ({mmh_curr:.5f})\nPrice: ${price:,.2f}\n{formatted_time}"
        elif all(buy_srsi_conds.values()):
            current_state = "buy_srsi50"
            send_message = f"{bot_name}\n▲ {pair_name} - BUY (SRSI 50)\nSRSI 15m Cross Up 50 ({smooth_rsi_curr:.2f})\nPrice: ${price:,.2f}\n{formatted_time}"
        elif all(sell_srsi_conds.values()):
            current_state = "sell_srsi50"
            send_message = f"{bot_name}\n🔻 {pair_name} - SELL (SRSI 50)\nSRSI 15m Cross Down 50 ({smooth_rsi_curr:.2f})\nPrice: ${price:,.2f}\n{formatted_time}"
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

        result = {"state": current_state, "ts": now_ts_int}
        if send_message:
            result["message"] = send_message
        return result
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
            symbol_norm = symbol.replace("_USDT", "USD").replace("USDT", "USD")
            if p.get("contract_type") == "perpetual_futures":
                for pair_name in cfg.PAIRS:
                    if symbol_norm == pair_name or symbol_norm.replace("_", "") == pair_name:
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
# TelegramQueue
# -------------------------
class TelegramQueue:
    def __init__(self, token: str, chat_id: str, rate_limit: float = 0.1):
        self.token = token
        self.chat_id = chat_id
        self.rate_limit = rate_limit
        self._last_sent = 0.0

    async def send(self, message: str) -> bool:
        session = SessionManager.get_session()
        now = time.time()
        time_since = now - self._last_sent
        if time_since < self.rate_limit:
            await asyncio.sleep(self.rate_limit - time_since)
        self._last_sent = time.time()
        return await self._send_with_retries(session, message)

    async def _send_once(self, session: aiohttp.ClientSession, message: str) -> bool:
        if not self.token or not self.chat_id or self.token == "xxxx" or self.chat_id == "xxxx":
            logger.debug("Telegram not configured; skipping alert.")
            return False
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        data = {"chat_id": self.chat_id, "text": message, "parse_mode": "HTML"}
        try:
            async with session.post(url, data=data, timeout=10) as resp:
                if resp.status == 429:
                    retry_after = int(resp.headers.get("Retry-After", 30))
                    logger.warning(f"Telegram rate limited. Retry after {retry_after}s")
                    await asyncio.sleep(retry_after)
                    return False
                try:
                    js = await resp.json(content_type=None)
                except Exception:
                    text = await resp.text()
                    js = {"ok": False, "status": resp.status, "text": text}
                ok = js.get("ok", False)
                if ok:
                    logger.info("✅ Alert sent successfully")
                    return True
                logger.warning(f"Telegram API returned non-ok: {js}")
                return False
        except Exception as e:
            logger.exception(f"Telegram send error: {e}")
            return False

    async def _send_with_retries(self, session: aiohttp.ClientSession, message: str) -> bool:
        last_exc = None
        for attempt in range(1, cfg.TELEGRAM_RETRIES + 1):
            ok = await self._send_once(session, message)
            if ok:
                return True
            last_exc = "Telegram non-ok or exception"
            await asyncio.sleep((cfg.TELEGRAM_BACKOFF_BASE ** (attempt - 1)) + random.uniform(0, 0.3))
        logger.error(f"Telegram send failed after retries: {last_exc}")
        return False

# -------------------------
# Dead Man's Switch
# -------------------------
class DeadMansSwitch:
    def __init__(self, state_db: StateDB, cooldown_seconds: int):
        self.state_db = state_db
        self.cooldown_seconds = cooldown_seconds
        self.alert_sent = False
        self.last_check_time = 0

    def _parse_last_success(self, last_success: Optional[str]) -> Optional[int]:
        if not last_success:
            return None
        try:
            return int(last_success)
        except Exception:
            pass
        try:
            dt = datetime.fromisoformat(last_success)
            return int(dt.timestamp())
        except Exception:
            return None

    def should_alert(self) -> bool:
        try:
            current_time = time.time()
            if current_time - self.last_check_time < 60:
                return False
            self.last_check_time = current_time
            last_success = self.state_db.get_metadata("last_success_run")
            if not last_success:
                logger.debug("No last_success_run found - first run?")
                return False
            last_run_ts = self._parse_last_success(last_success)
            if last_run_ts is None:
                logger.debug("Could not parse last_success_run; skipping dead man's check")
                return False
            time_since_last_run = current_time - last_run_ts
            logger.debug(f"Dead man's check: {time_since_last_run}s since last run, threshold: {self.cooldown_seconds}s")
            if time_since_last_run > self.cooldown_seconds and not self.alert_sent:
                self.alert_sent = True
                logger.warning(f"Dead man's switch triggered! Last run was {time_since_last_run/3600:.1f} hours ago")
                return True
            if time_since_last_run <= self.cooldown_seconds and self.alert_sent:
                self.alert_sent = False
                logger.info("Dead man's switch reset - bot is running normally")
            return False
        except Exception as e:
            logger.error(f"Error checking dead man's switch: {e}")
            return False

    async def send_alert(self, telegram_queue: TelegramQueue):
        session = SessionManager.get_session()
        last_success = self.state_db.get_metadata("last_success_run")
        last_run_ts = self._parse_last_success(last_success) if last_success else None
        last_run_time = datetime.fromtimestamp(last_run_ts) if last_run_ts else "Never"
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        message = (
            f"🚨 **{cfg.BOT_NAME} - DEAD MAN'S SWITCH TRIGGERED** 🚨\n"
            f"⏰ No successful run detected in **{self.cooldown_seconds // 3600} hours**\n"
            f"📅 Last successful run: `{last_run_time}`\n"
            f"🕒 Current time: `{current_time}`\n"
            f"🔍 Check bot status immediately!"
        )
        success = await telegram_queue.send(message)
        if success:
            logger.info("Dead man's switch alert sent successfully")
        return success

# -------------------------
# check_pair
# -------------------------
async def check_pair(fetcher: DataFetcher, products_map: Dict[str, dict],
                     pair_name: str, last_state_for_pair: Optional[Dict[str, Any]],
                     telegram_queue: TelegramQueue) -> Optional[Tuple[str, Dict[str, Any]]]:
    try:
        prod = products_map.get(pair_name)
        if not prod:
            logger.debug(f"No product mapping for {pair_name}")
            return None
        sp = cfg.SPECIAL_PAIRS.get(pair_name, {})
        limit_15m = sp.get("limit_15m", 210)
        min_required = sp.get("min_required", 180)
        limit_5m = sp.get("limit_5m", 300)
        min_required_5m = sp.get("min_required_5m", 200)
        symbol = prod["symbol"]
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
        for df in (df_15m, df_5m):
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        new_state = evaluate_pair_logic(pair_name, df_15m, df_5m, last_state_for_pair)
        if not new_state:
            return None
        message = new_state.get("message")
        if message:
            await telegram_queue.send(message)
        return pair_name, new_state
    except Exception as e:
        logger.exception(f"Error in check_pair for {pair_name}: {e}")
        return None

# -------------------------
# process_batch
# -------------------------
async def process_batch(fetcher: DataFetcher, products_map: Dict[str, dict],
                        batch_pairs: List[str], state_db: StateDB, telegram_queue: TelegramQueue,
                        last_alerts: Dict) -> List[Tuple[str, Dict[str, Any]]]:
    results: List[Tuple[str, Dict[str, Any]]] = []
    tasks = []
    for pair_name in batch_pairs:
        if pair_name not in products_map:
            logger.warning(f"No product mapping for {pair_name}")
            continue
        last_state = state_db.get(pair_name)
        tasks.append(asyncio.create_task(check_pair(fetcher, products_map, pair_name, last_state, telegram_queue)))
    for task in tasks:
        try:
            res = await task
            if res:
                results.append(res)
        except Exception as e:
            logger.exception(f"Batch task error: {e}")
    return results

# -------------------------
# run_once
# -------------------------
async def run_once():
    start_time = time.time()
    logger.info("="*50)
    logger.info("Starting run_once")
    try:
        if not await health_check():
            logger.error("Health check failed - API unreachable, skipping run")
            return False
        try:
            p = psutil.Process(os.getpid())
            rss = p.memory_info().rss
            if rss > cfg.MEMORY_LIMIT_BYTES:
                logger.critical(f"High memory usage at start: {rss} > {cfg.MEMORY_LIMIT_BYTES}")
                raise SystemExit(3)
        except Exception:
            pass
        with StateDB(cfg.STATE_DB_PATH) as sdb:
            deleted = sdb.prune_old_records(cfg.STATE_EXPIRY_DAYS)
            if deleted > 0:
                logger.info(f"Pruned {deleted} old state records")
        with StateDB(cfg.STATE_DB_PATH) as sdb:
            last_alerts = sdb.load_all()
            logger.info(f"Loaded {len(last_alerts)} previous states")
            dead_mans_switch = DeadMansSwitch(sdb, cfg.DEAD_MANS_COOLDOWN_SECONDS)
            telegram_queue = TelegramQueue(cfg.TELEGRAM_BOT_TOKEN, cfg.TELEGRAM_CHAT_ID)
            if dead_mans_switch.should_alert():
                logger.warning("Dead man's switch triggered - sending alert")
                await dead_mans_switch.send_alert(telegram_queue)
            fetcher = DataFetcher(cfg.DELTA_API_BASE)
            telegram_queue = TelegramQueue(cfg.TELEGRAM_BOT_TOKEN, cfg.TELEGRAM_CHAT_ID)
            if cfg.SEND_TEST_MESSAGE:
                ist = pytz.timezone("Asia/Kolkata")
                current_dt = datetime.now(ist)
                test_msg = (f"🚀 {cfg.BOT_NAME} Started\n"
                            f"Time: {current_dt.strftime('%d-%m-%Y @ %H:%M IST')}\n"
                            f"Pairs: {len(cfg.PAIRS)} | Debug: {cfg.DEBUG_MODE}")
                await telegram_queue.send(test_msg)
            logger.info("Fetching products from API...")
            prod_resp = await fetcher.fetch_products()
            if not prod_resp:
                logger.error("Failed to fetch products; aborting run.")
                return False
            products_map = build_products_map_from_api_result(prod_resp)
            if not products_map:
                logger.error("No tradable pairs found; exiting.")
                return False
            pairs_to_process = [p for p in cfg.PAIRS if p in products_map]
            batch_size = max(1, cfg.BATCH_SIZE)
            logger.info(f"Processing {len(pairs_to_process)} pairs in batches of {batch_size}")
            all_results: List[Tuple[str, Dict[str, Any]]] = []
            for i in range(0, len(pairs_to_process), batch_size):
                batch = pairs_to_process[i:i+batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}: {batch}")
                batch_results = await process_batch(fetcher, products_map, batch, sdb, telegram_queue, last_alerts)
                all_results.extend(batch_results)
                if i + batch_size < len(pairs_to_process):
                    await asyncio.sleep(random.uniform(0.5, 1.0))
            updates = 0
            alerts_sent = 0
            for pair_name, new_state in all_results:
                if not isinstance(new_state, dict):
                    continue
                prev = sdb.get(pair_name)
                if prev != new_state:
                    sdb.set(pair_name, new_state.get("state"), new_state.get("ts"))
                    updates += 1
                if new_state.get("message"):
                    alerts_sent += 1
            sdb.set_metadata("last_success_run", str(int(time.time())))
            logger.info(f"Run complete. {updates} state updates applied. Sent ~{alerts_sent} alerts.")
        await cleanup_resources()
        try:
            await SessionManager.close_session()
        except Exception:
            pass
        elapsed = time.time() - start_time
        logger.info(f"run_once finished in {elapsed:.2f}s")
        return True
    except Exception as e:
        logger.exception(f"run_once unhandled error: {e}")
        try:
            await SessionManager.close_session()
        except Exception:
            pass
        return False

# -------------------------
# Main loop / entrypoint
# -------------------------
def _install_signal_handlers(loop, stop):
    def _handler(signame):
        logger.info(f"Received signal {signame} - shutting down")
        stop.set_result(True)
    for sig in ('SIGINT', 'SIGTERM'):
        try:
            loop.add_signal_handler(getattr(signal, sig), lambda s=sig: _handler(s))
        except NotImplementedError:
            # Windows fallback
            pass

async def _main_once():
    ok = await run_once()
    return 0 if ok else 2

async def _main_loop(interval_seconds: int = 60):
    stop = asyncio.get_running_loop().create_future()
    _install_signal_handlers(asyncio.get_running_loop(), stop)
    pid = PidFileLock()
    if not pid.acquire():
        logger.error("Could not acquire PID lock; exiting.")
        return 2
    try:
        while not stop.done():
            try:
                await run_once()
            except Exception as e:
                logger.exception(f"Error in main loop iteration: {e}")
            await asyncio.sleep(interval_seconds)
    finally:
        try:
            pid.release()
        except Exception:
            pass
    return 0

def main():
    import argparse
    parser = argparse.ArgumentParser(description="MACD/PPO Alert Bot")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--interval", type=int, default=60, help="Interval seconds between runs in loop mode")
    args = parser.parse_args()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    if args.once:
        try:
            rc = loop.run_until_complete(_main_once())
            return rc
        finally:
            loop.run_until_complete(SessionManager.close_session())
            loop.close()
    else:
        try:
            rc = loop.run_until_complete(_main_loop(args.interval))
            return rc
        finally:
            try:
                loop.run_until_complete(SessionManager.close_session())
            except Exception:
                pass
            loop.close()

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted by user")

