#!/usr/bin/env python3
# macd_bot_improved.py
# Python 3.12 — Production-hardened with EXACT Pinescript MMH reversal logic
# MMH Reversal: BUY = H_Tr[3] > H_Tr[2] > H_Tr[1] and H_Tr > H_Tr[1]
#               SELL = H_Tr[3] < H_Tr[2] < H_Tr[1] and H_Tr < H_Tr[1]

from __future__ import annotations

import os
import sys
import json
import time
import asyncio
import argparse
import random
import logging
import sqlite3
import signal
import traceback
import fcntl
import atexit
from typing import Dict, Any, Optional, Tuple, List, NamedTuple
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import pandas as pd
import numpy as np
import pytz
import psutil
from aiohttp import ClientConnectorError, ClientResponseError, TCPConnector
from logging.handlers import RotatingFileHandler

# -------------------------
# Configuration Schema (with enhanced options)
# -------------------------
class Config:
    """Typed configuration holder with validation."""
    def __init__(self):
        self.TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "8462496498:AAHYZ4xDIHvrVRjmCmZyoPhupCjRaRgiITc")
        self.TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "203813932")
        self.DEBUG_MODE = self._str_to_bool(os.environ.get("DEBUG_MODE", "false"))
        self.SEND_TEST_MESSAGE = self._str_to_bool(os.environ.get("SEND_TEST_MESSAGE", "false"))
        self.DELTA_API_BASE = os.environ.get("DELTA_API_BASE", "https://api.india.delta.exchange")
        
        # Core pairs with ability to override via config file
        self.PAIRS = ["BTCUSD", "ETHUSD", "SOLUSD", "AVAXUSD", "BCHUSD", 
                     "XRPUSD", "BNBUSD", "LTCUSD", "DOTUSD", "ADAUSD", "SUIUSD", "AAVEUSD"]
        
        # Special pair configurations
        self.SPECIAL_PAIRS = {
            "SOLUSD": {
                "limit_15m": 210, 
                "min_required": 180, 
                "limit_5m": 300, 
                "min_required_5m": 200,
                "dead_mans_cooldown": 7200  # 2 hours for volatile pairs
            },
            "LTCUSD": {
                "limit_15m": 250,
                "min_required": 200,
                "limit_5m": 350,
                "min_required_5m": 250,
                "dead_mans_cooldown": 3600   # 1 hour
            }
        }
        
        # Indicator settings
        self.PPO_FAST = 7
        self.PPO_SLOW = 16
        self.PPO_SIGNAL = 5
        self.PPO_USE_SMA = False
        self.RMA_50_PERIOD = 50
        self.RMA_200_PERIOD = 200
        
        # Cirrus Cloud settings
        self.CIRRUS_CLOUD_ENABLED = True
        self.X1, self.X2, self.X3, self.X4 = 22, 9, 15, 5
        
        # Smooth RSI settings
        self.SRSI_RSI_LEN = 21
        self.SRSI_KALMAN_LEN = 5
        self.SRSI_EMA_LEN = 5
        
        # Operational settings
        self.STATE_DB_PATH = os.environ.get("STATE_DB_PATH", "macd_state.sqlite")
        self.LOG_FILE = os.environ.get("LOG_FILE", "macd_bot.log")
        self.MAX_PARALLEL_FETCH = self._int_env("MAX_PARALLEL_FETCH", 3)  # Reduced for stability
        self.HTTP_TIMEOUT = self._int_env("HTTP_TIMEOUT", 20)
        self.CANDLE_FETCH_RETRIES = self._int_env("CANDLE_FETCH_RETRIES", 5)
        self.CANDLE_FETCH_BACKOFF = self._float_env("CANDLE_FETCH_BACKOFF", 1.5)
        self.JITTER_MIN = self._float_env("JITTER_MIN", 0.2)
        self.JITTER_MAX = self._float_env("JITTER_MAX", 1.0)
        self.STATE_EXPIRY_DAYS = self._int_env("STATE_EXPIRY_DAYS", 30)
        self.RUN_TIMEOUT_SECONDS = self._int_env("RUN_TIMEOUT_SECONDS", 600)
        self.BATCH_SIZE = self._int_env("BATCH_SIZE", 2)  # Reduced for reliability
        self.LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
        self.TELEGRAM_RETRIES = self._int_env("TELEGRAM_RETRIES", 3)
        self.TELEGRAM_BACKOFF_BASE = self._float_env("TELEGRAM_BACKOFF_BASE", 2.0)
        self.MEMORY_LIMIT_BYTES = self._int_env("MEMORY_LIMIT_BYTES", 400_000_000)
        self.TCP_CONN_LIMIT = self._int_env("TCP_CONN_LIMIT", 6)
        self.DEAD_MANS_COOLDOWN_SECONDS = self._int_env("DEAD_MANS_COOLDOWN_SECONDS", 14_400)  # 4 hours
        
        # Data quality settings
        self.MAX_DATA_AGE_SECONDS = self._int_env("MAX_DATA_AGE_SECONDS", 300)  # 5 min freshness
        self.MIN_REVERSAL_CONFIRMATION = self._int_env("MIN_REVERSAL_CONFIRMATION", 1)  # 1 candle
        
        # Alert settings
        self.ALERT_COOLDOWN_SECONDS = self._int_env("ALERT_COOLDOWN_SECONDS", 900)  # 15 min between same alerts
        
    @staticmethod
    def _str_to_bool(value: str) -> bool:
        return str(value).strip().lower() in ("true", "1", "yes", "y", "t")
    
    @staticmethod
    def _int_env(key: str, default: int) -> int:
        return int(os.environ.get(key, str(default)))
    
    @staticmethod
    def _float_env(key: str, default: float) -> float:
        return float(os.environ.get(key, str(default)))

# Load configuration
CONFIG_FILE = os.getenv("CONFIG_FILE", "config_macd.json")
cfg = Config()

if Path(CONFIG_FILE).exists():
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            user_cfg = json.load(f)
            for key, value in user_cfg.items():
                if hasattr(cfg, key):
                    setattr(cfg, key, value)
        print(f"✅ Loaded configuration from {CONFIG_FILE}")
    except Exception as e:
        print(f"⚠️ Warning: unable to parse config file: {e}")

# Validate critical config
if not cfg.PAIRS:
    print("❌ PAIRS cannot be empty. Exiting.")
    sys.exit(1)

if cfg.MAX_PARALLEL_FETCH < 1 or cfg.MAX_PARALLEL_FETCH > 10:
    cfg.MAX_PARALLEL_FETCH = max(1, min(10, cfg.MAX_PARALLEL_FETCH))

print(f"DEBUG_MODE={cfg.DEBUG_MODE}, PAIRS={len(cfg.PAIRS)}, BATCH_SIZE={cfg.BATCH_SIZE}")

# -------------------------
# Enhanced Logger with performance metrics
# -------------------------
class MetricsFilter(logging.Filter):
    """Adds performance metrics to log records."""
    def filter(self, record):
        record.memory_mb = 0
        try:
            record.memory_mb = psutil.Process(os.getpid()).memory_info().rss / 1_000_000
        except Exception:
            pass
        return True

logger = logging.getLogger("macd_bot")
log_level = getattr(logging, cfg.LOG_LEVEL.upper(), logging.INFO)
logger.setLevel(logging.DEBUG if cfg.DEBUG_MODE else log_level)

# Console handler
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG if cfg.DEBUG_MODE else log_level)
fmt = logging.Formatter("%(asctime)s %(levelname)s [%(memory_mb).1fMB] %(message)s")
ch.addFilter(MetricsFilter())
ch.setFormatter(fmt)
logger.addHandler(ch)

# File handler
try:
    log_dir = os.path.dirname(cfg.LOG_FILE) or "."
    os.makedirs(log_dir, exist_ok=True)
    fh = RotatingFileHandler(cfg.LOG_FILE, maxBytes=10_000_000, backupCount=10, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.addFilter(MetricsFilter())
    fh.setFormatter(fmt)
    logger.addHandler(fh)
except Exception as e:
    logger.warning(f"Could not set up rotating log file: {e}")

# -------------------------
# PID File Lock (thread-safe)
# -------------------------
class PidFileLock:
    def __init__(self, path: str = "/tmp/macd_bot.pid"):
        self.path = Path(path)
        self.fd = None

    def acquire(self) -> bool:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.fd = open(self.path, "w")
            fcntl.flock(self.fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.fd.write(f"{os.getpid()}\n")
            self.fd.flush()
            atexit.register(self.release)
            logger.info(f"🔒 PID lock acquired: {self.path}")
            return True
        except (IOError, OSError):
            if self.fd:
                try:
                    self.fd.close()
                except Exception:
                    pass
            logger.error(f"❌ Another instance is running (PID file exists: {self.path})")
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
            if self.path.exists():
                try:
                    self.path.unlink()
                    logger.info(f"🔓 PID lock released: {self.path}")
                except Exception:
                    pass
        except Exception:
            pass

# -------------------------
# SQLite State DB with connection pooling
# -------------------------
class StateDB:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._conn = sqlite3.connect(
            self.db_path, 
            check_same_thread=False, 
            isolation_level=None,
            timeout=30.0  # Wait up to 30s for locks
        )
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute("PRAGMA temp_store=MEMORY;")
        self._conn.execute("PRAGMA cache_size=-64000;")  # 64MB cache
        self._ensure_tables()
        logger.info(f"💾 StateDB initialized: {self.db_path}")

    def _ensure_tables(self):
        cur = self._conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS states (
                pair TEXT PRIMARY KEY,
                state TEXT NOT NULL,
                ts INTEGER NOT NULL,
                last_alert_ts INTEGER DEFAULT 0
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        # Index for faster queries
        cur.execute("CREATE INDEX IF NOT EXISTS idx_ts ON states(ts)")
        self._conn.commit()

    def load_all(self) -> Dict[str, Dict[str, Any]]:
        cur = self._conn.cursor()
        cur.execute("SELECT pair, state, ts, last_alert_ts FROM states")
        return {r[0]: {"state": r[1], "ts": r[2], "last_alert_ts": r[3]} for r in cur.fetchall()}

    def get(self, pair: str) -> Optional[Dict[str, Any]]:
        cur = self._conn.cursor()
        cur.execute("SELECT state, ts, last_alert_ts FROM states WHERE pair = ?", (pair,))
        r = cur.fetchone()
        return {"state": r[0], "ts": r[1], "last_alert_ts": r[2]} if r else None

    def set(self, pair: str, state: Optional[str], ts: Optional[int] = None, alert_ts: int = 0):
        ts = int(ts or time.time())
        cur = self._conn.cursor()
        if state is None:
            cur.execute("DELETE FROM states WHERE pair = ?", (pair,))
        else:
            cur.execute(
                "INSERT INTO states(pair, state, ts, last_alert_ts) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(pair) DO UPDATE SET state=excluded.state, ts=excluded.ts, last_alert_ts=excluded.last_alert_ts",
                (pair, state, ts, alert_ts)
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
                logger.info(f"🧹 Vacuumed DB after pruning {deleted} records")
            except Exception as e:
                logger.warning(f"VACUUM failed: {e}")
        self._conn.commit()
        return deleted

    def is_alert_cooldown_active(self, pair: str, cooldown_seconds: int) -> bool:
        """Check if alert cooldown is still active for this pair."""
        cur = self._conn.cursor()
        cur.execute("SELECT last_alert_ts FROM states WHERE pair = ?", (pair,))
        r = cur.fetchone()
        if not r or not r[0]:
            return False
        last_alert = r[0]
        return (time.time() - last_alert) < cooldown_seconds

    def update_alert_timestamp(self, pair: str):
        """Update the last alert timestamp for cooldown tracking."""
        cur = self._conn.cursor()
        cur.execute("UPDATE states SET last_alert_ts = ? WHERE pair = ?", (int(time.time()), pair))
        self._conn.commit()

    def close(self):
        try:
            self._conn.close()
        except Exception:
            pass

# -------------------------
# Circuit Breaker with metrics
# -------------------------
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, timeout: int = 300):
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.threshold = failure_threshold
        self.timeout = timeout
        self.lock = asyncio.Lock()
        self.total_calls = 0
        self.total_failures = 0

    async def is_open(self) -> bool:
        async with self.lock:
            if self.failure_count < self.threshold:
                return False
            if self.last_failure_time and (time.time() - self.last_failure_time > self.timeout):
                logger.warning(f"🔵 Circuit breaker reset after timeout")
                self.failure_count = 0
                self.last_failure_time = None
                return False
            return True

    async def call(self, func, *args, **kwargs):
        self.total_calls += 1
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
            self.total_failures += 1
            self.last_failure_time = time.time()
            logger.warning(f"🔴 Circuit breaker: failure {self.failure_count}/{self.threshold}")

    async def record_success(self):
        async with self.lock:
            if self.failure_count > 0:
                logger.info(f"🟢 Circuit breaker reset after success")
                self.failure_count = 0
                self.last_failure_time = None

    def get_stats(self) -> Dict[str, Any]:
        return {
            "failure_count": self.failure_count,
            "total_calls": self.total_calls,
            "total_failures": self.total_failures,
            "failure_rate": self.total_failures / max(1, self.total_calls)
        }

# -------------------------
# Async HTTP Client with metrics
# -------------------------
class HTTPClient:
    def __init__(self, session: aiohttp.ClientSession, timeout: int = 15):
        self.session = session
        self.timeout = timeout
        self.request_count = 0
        self.error_count = 0

    async def fetch_json(self, url: str, params: dict = None, retries: int = 3, 
                        backoff: float = 1.5, circuit_breaker: Optional[CircuitBreaker] = None) -> Optional[dict]:
        self.request_count += 1
        
        if circuit_breaker and await circuit_breaker.is_open():
            logger.warning(f"⚠️ Circuit breaker open; skipping {url}")
            return None

        for attempt in range(1, retries + 1):
            try:
                async with self.session.get(url, params=params, timeout=self.timeout) as resp:
                    text = await resp.text()
                    if resp.status >= 400:
                        logger.debug(f"HTTP {resp.status} {url} - {text[:200]}")
                        raise ClientResponseError(resp.request_info, resp.history, status=resp.status)
                    data = await resp.json()
                    if circuit_breaker:
                        await circuit_breaker.record_success()
                    return data
            except (asyncio.TimeoutError, ClientConnectorError) as e:
                self.error_count += 1
                logger.debug(f"Fetch attempt {attempt}/{retries} error for {url}: {e}")
                if attempt == retries:
                    logger.warning(f"❌ Failed to fetch {url} after {retries} attempts")
                    if circuit_breaker:
                        await circuit_breaker.record_failure()
                    return None
                await asyncio.sleep(backoff * (attempt ** 1.2) + random.uniform(0, 0.2))
            except ClientResponseError as cre:
                self.error_count += 1
                logger.debug(f"ClientResponseError: {cre}")
                if circuit_breaker:
                    await circuit_breaker.record_failure()
                return None
            except Exception as e:
                self.error_count += 1
                logger.exception(f"Unexpected fetch error for {url}: {e}")
                if circuit_breaker:
                    await circuit_breaker.record_failure()
                return None
        return None

    def get_stats(self) -> Dict[str, Any]:
        return {
            "requests": self.request_count,
            "errors": self.error_count,
            "error_rate": self.error_count / max(1, self.request_count)
        }

# -------------------------
# DataFetcher with caching and validation
# -------------------------
class DataFetcher:
    def __init__(self, api_base: str, max_parallel: int = 3, timeout: int = 20):
        self.api_base = api_base.rstrip("/")
        self.semaphore = asyncio.Semaphore(max_parallel)
        self.max_parallel = max_parallel
        self.timeout = timeout
        self._cache: Dict[str, Tuple[float, Any]] = {}
        self.circuit_breaker = CircuitBreaker()
        self.fetch_count = 0

    async def fetch_products(self, session: aiohttp.ClientSession) -> Optional[dict]:
        url = f"{self.api_base}/v2/products"
        async with self.semaphore:
            client = HTTPClient(session, self.timeout)
            return await client.fetch_json(url, retries=cfg.CANDLE_FETCH_RETRIES, 
                                         backoff=cfg.CANDLE_FETCH_BACKOFF, 
                                         circuit_breaker=self.circuit_breaker)

    async def fetch_candles(self, session: aiohttp.ClientSession, symbol: str, 
                           resolution: str, limit: int) -> Optional[dict]:
        cache_key = f"candles:{symbol}:{resolution}:{limit}"
        if cache_key in self._cache:
            age, data = self._cache[cache_key]
            if time.time() - age < 60:
                logger.debug(f"📦 Cache hit for {cache_key}")
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
            client = HTTPClient(session, self.timeout)
            data = await client.fetch_json(url, params=params, retries=cfg.CANDLE_FETCH_RETRIES, 
                                         backoff=cfg.CANDLE_FETCH_BACKOFF, 
                                         circuit_breaker=self.circuit_breaker)
        
        self.fetch_count += 1
        self._cache[cache_key] = (time.time(), data)
        return data

    def get_stats(self) -> Dict[str, Any]:
        return {
            "fetch_count": self.fetch_count,
            "circuit_breaker": self.circuit_breaker.get_stats(),
            "cache_size": len(self._cache)
        }

    def clear_cache(self):
        self._cache.clear()

# -------------------------
# Candle parsing with strict validation
# -------------------------
class CandleValidator:
    @staticmethod
    def validate(df: pd.DataFrame, pair: str, timeframe: str) -> bool:
        """Comprehensive validation of candle data."""
        if df is None or df.empty:
            logger.warning(f"Empty dataframe for {pair} {timeframe}")
            return False
        
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            logger.warning(f"Missing columns {missing} for {pair} {timeframe}")
            return False
        
        # Check for recent data
        max_age = cfg.MAX_DATA_AGE_SECONDS
        data_age = time.time() - df['timestamp'].iloc[-1]
        if data_age > max_age:
            logger.warning(f"Stale data for {pair} {timeframe}: {data_age:.0f}s old")
            return False
        
        # Check for zero/negative prices
        if (df['close'] <= 0).any():
            logger.warning(f"Invalid close prices for {pair} {timeframe}")
            return False
        
        # Check for NaNs
        nan_count = df[required_cols].isnull().sum().sum()
        if nan_count > 0:
            logger.debug(f"{nan_count} NaN values in {pair} {timeframe}")
        
        return True

def parse_candles_result(result: dict) -> Optional[pd.DataFrame]:
    """Parse API result into validated DataFrame."""
    if not result or not isinstance(result, dict):
        return None
    if not result.get("success", True) and "result" not in result:
        return None
    
    res = result.get("result", {}) or {}
    required_keys = ['t', 'o', 'h', 'l', 'c', 'v']
    if not all(k in res for k in required_keys):
        return None
    
    try:
        min_len = min(len(res[k]) for k in required_keys)
        df = pd.DataFrame({
            "timestamp": res['t'][:min_len],
            "open": pd.to_numeric(res['o'][:min_len], errors='coerce'),
            "high": pd.to_numeric(res['h'][:min_len], errors='coerce'),
            "low": pd.to_numeric(res['l'][:min_len], errors='coerce'),
            "close": pd.to_numeric(res['c'][:min_len], errors='coerce'),
            "volume": pd.to_numeric(res['v'][:min_len], errors='coerce')
        })
        
        df = df.sort_values('timestamp').drop_duplicates(subset='timestamp').reset_index(drop=True)
        
        if df.empty or df['close'].iloc[-1] <= 0:
            return None
        
        return df
    except Exception as e:
        logger.exception(f"Failed to parse candles: {e}")
        return None

# -------------------------
# Technical Indicators (Optimized)
# -------------------------
class Indicators:
    @staticmethod
    def calculate_ema(data: pd.Series, period: int) -> pd.Series:
        return data.ewm(span=period, adjust=False).mean()

    @staticmethod
    def calculate_sma(data: pd.Series, period: int) -> pd.Series:
        return data.rolling(window=period, min_periods=max(2, period//3)).mean()

    @staticmethod
    def calculate_rma(data: pd.Series, period: int) -> pd.Series:
        return data.ewm(alpha=1/period, adjust=False).mean().bfill().ffill()

    @staticmethod
    def calculate_ppo(df: pd.DataFrame, fast: int, slow: int, signal: int, use_sma: bool = False) -> Tuple[pd.Series, pd.Series]:
        close = df['close'].astype(float)
        fast_ma = Indicators.calculate_sma(close, fast) if use_sma else Indicators.calculate_ema(close, fast)
        slow_ma = Indicators.calculate_sma(close, slow) if use_sma else Indicators.calculate_ema(close, slow)
        slow_ma = slow_ma.replace(0, np.nan).bfill().ffill()
        ppo = ((fast_ma - slow_ma) / slow_ma) * 100
        ppo = ppo.replace([np.inf, -np.inf], np.nan).bfill().ffill()
        ppo_signal = Indicators.calculate_sma(ppo, signal) if use_sma else Indicators.calculate_ema(ppo, signal)
        return ppo.replace([np.inf, -np.inf], np.nan).bfill().ffill(), ppo_signal.replace([np.inf, -np.inf], np.nan).bfill().ffill()

    @staticmethod
    def calculate_smooth_rsi(df: pd.DataFrame, rsi_len: int, kalman_len: int) -> pd.Series:
        close = df['close'].astype(float)
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = Indicators.calculate_rma(gain, rsi_len)
        avg_loss = Indicators.calculate_rma(loss, rsi_len).replace(0, np.nan).bfill().ffill().clip(lower=1e-8)
        rs = avg_gain.divide(avg_loss)
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.replace([np.inf, -np.inf], np.nan).bfill().ffill()
        return Indicators.kalman_filter(rsi, kalman_len).bfill().ffill()

    @staticmethod
    def kalman_filter(src: pd.Series, length: int, R: float = 0.01, Q: float = 0.1) -> pd.Series:
        result = []
        estimate = np.nan
        error_est = 1.0
        error_meas = R * max(1, length)
        Q_div_length = Q / max(1, length)
        
        for i in range(len(src)):
            current = src.iloc[i]
            if np.isnan(estimate):
                estimate = src.iloc[i-1] if i > 0 else current
                if i == 0:
                    result.append(np.nan)
                    continue
            
            prediction = estimate
            kalman_gain = error_est / (error_est + error_meas)
            estimate = prediction + kalman_gain * (current - prediction)
            error_est = (1 - kalman_gain) * error_est + Q_div_length
            result.append(estimate)
        
        return pd.Series(result, index=src.index)

    @staticmethod
    def calculate_cirrus_cloud(df: pd.DataFrame):
        close = df['close'].astype(float)
        
        def smoothrng(x: pd.Series, t: int, m: int) -> pd.Series:
            wper = t * 2 - 1
            avrng = Indicators.calculate_ema(np.abs(x.diff().fillna(0)), t)
            return Indicators.calculate_ema(avrng, max(1, wper)).clip(lower=1e-8).bfill().ffill() * m

        def rngfilt(x: pd.Series, r: pd.Series) -> pd.Series:
            result = [x.iloc[0]]
            for i in range(1, len(x)):
                prev = result[-1]
                curr_x = x.iloc[i]
                curr_r = max(float(r.iloc[i]), 1e-8)
                f = prev if (curr_x - curr_r) < prev else (curr_x - curr_r) if curr_x > prev else \
                    prev if (curr_x + curr_r) > prev else (curr_x + curr_r)
                result.append(f)
            return pd.Series(result, index=x.index)

        smrngx1x = smoothrng(close, cfg.X1, cfg.X2)
        smrngx1x2 = smoothrng(close, cfg.X3, cfg.X4)
        filtx1 = rngfilt(close, smrngx1x)
        filtx12 = rngfilt(close, smrngx1x2)
        
        upw = filtx1 < filtx12
        dnw = filtx1 > filtx12
        return upw, dnw, filtx1, filtx12

    @staticmethod
    def calculate_mmh(df: pd.DataFrame, period: int = 144, responsiveness: float = 0.9) -> pd.Series:
        """Magical Momentum Histogram with enhanced smoothing."""
        n = len(df)
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

        # ENHANCED: Double smoothing for stability
        hist_smoothed = Indicators.calculate_ema(hist, 3)
        return hist_smoothed.replace([np.inf, -np.inf], 0).fillna(0)

# -------------------------
# Signal Evaluation (EXACT Pinescript Logic)
# -------------------------
class SignalEvaluator:
    def __init__(self, pair_name: str, df_15m: pd.DataFrame, df_5m: pd.DataFrame):
        self.pair = pair_name
        self.df_15m = df_15m
        self.df_5m = df_5m
        self.last_i, self.prev_i, self.last_i_5m = self._get_indices()
        
    def _get_indices(self) -> Tuple[int, int, int]:
        def closed_index(df: pd.DataFrame, res_min: int) -> int:
            if df is None or df.empty:
                return -1
            last_ts = int(df['timestamp'].iloc[-1])
            current_start = int(time.time()) - (int(time.time()) % (res_min * 60))
            return -2 if last_ts >= current_start else -1
        return closed_index(self.df_15m, 15), closed_index(self.df_15m, 15) - 1, closed_index(self.df_5m, 5)

    def evaluate(self, last_state: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Main evaluation logic matching Pinescript exactly."""
        try:
            # Calculate all indicators
            mmh = Indicators.calculate_mmh(self.df_15m)
            
            # EXACT Pinescript logic: need 4 bars (prev3, prev2, prev1, curr)
            if len(mmh) < abs(self.last_i) + 4:
                logger.debug(f"Insufficient MMH data for {self.pair}: {len(mmh)} < {abs(self.last_i) + 4}")
                return None

            # Pinescript indices: H_Tr[3], H_Tr[2], H_Tr[1], H_Tr
            mmh_curr  = float(mmh.iloc[self.last_i])      # H_Tr
            mmh_prev1 = float(mmh.iloc[self.last_i - 1])  # H_Tr[1]
            mmh_prev2 = float(mmh.iloc[self.last_i - 2])  # H_Tr[2]
            mmh_prev3 = float(mmh.iloc[self.last_i - 3])  # H_Tr[3]

            ppo, ppo_signal = Indicators.calculate_ppo(self.df_15m, cfg.PPO_FAST, cfg.PPO_SLOW, cfg.PPO_SIGNAL, cfg.PPO_USE_SMA)
            rma_50 = Indicators.calculate_rma(self.df_15m['close'], cfg.RMA_50_PERIOD)
            rma_200 = Indicators.calculate_rma(self.df_5m['close'], cfg.RMA_200_PERIOD)  # Intentional 5m
            upw, dnw, _, _ = Indicators.calculate_cirrus_cloud(self.df_15m)
            smooth_rsi = Indicators.calculate_smooth_rsi(self.df_15m, cfg.SRSI_RSI_LEN, cfg.SRSI_KALMAN_LEN)

            # Extract current values
            ppo_curr = float(ppo.iloc[self.last_i])
            ppo_prev = float(ppo.iloc[self.prev_i])
            ppo_signal_curr = float(ppo_signal.iloc[self.last_i])
            ppo_signal_prev = float(ppo_signal.iloc[self.prev_i])
            smooth_rsi_curr = float(smooth_rsi.iloc[self.last_i])
            smooth_rsi_prev = float(smooth_rsi.iloc[self.prev_i])

            close_curr = float(self.df_15m['close'].iloc[self.last_i])
            open_curr = float(self.df_15m['open'].iloc[self.last_i])
            high_curr = float(self.df_15m['high'].iloc[self.last_i])
            low_curr = float(self.df_15m['low'].iloc[self.last_i])

            rma50_curr = float(rma_50.iloc[self.last_i])
            rma200_curr = float(rma_200.iloc[self.last_i_5m])

            # Validate no NaNs
            indicators = [ppo_curr, ppo_prev, ppo_signal_curr, ppo_signal_prev,
                         smooth_rsi_curr, smooth_rsi_prev, rma50_curr, rma200_curr,
                         mmh_curr, mmh_prev1, mmh_prev2, mmh_prev3]
            if any(pd.isna(x) for x in indicators):
                logger.debug(f"NaN in indicators for {self.pair}")
                return None

            # Candle analysis
            total_range = high_curr - low_curr
            upper_wick = lower_wick = 0.0
            strong_bullish_close = strong_bearish_close = False
            
            if total_range > 0:
                upper_wick = max(0.0, high_curr - max(open_curr, close_curr))
                lower_wick = max(0.0, min(open_curr, close_curr) - low_curr)
                bullish_candle = close_curr > open_curr
                bearish_candle = close_curr < open_curr
                strong_bullish_close = bullish_candle and (upper_wick / total_range) < 0.20
                strong_bearish_close = bearish_candle and (lower_wick / total_range) < 0.20

            # Signal conditions
            ppo_cross_above_zero = (ppo_prev <= 0) and (ppo_curr > 0)
            ppo_cross_below_zero = (ppo_prev >= 0) and (ppo_curr < 0)
            ppo_cross_above_011 = (ppo_prev <= 0.11) and (ppo_curr > 0.11)
            ppo_cross_below_minus011 = (ppo_prev >= -0.11) and (ppo_curr < -0.11)

            ppo_above_signal = ppo_curr > ppo_signal_curr
            ppo_below_signal = ppo_curr < ppo_signal_curr
            ppo_below_030 = ppo_curr < 0.30
            ppo_above_minus030 = ppo_curr > -0.30

            # EXACT Pinescript MMH Reversal Logic (NO THRESHOLD)
            # BUY: H_Tr[3] > H_Tr[2] > H_Tr[1] and H_Tr > H_Tr[1]
            # SELL: H_Tr[3] < H_Tr[2] < H_Tr[1] and H_Tr < H_Tr[1]
            mmh_reversal_buy = (mmh_prev3 > mmh_prev2 > mmh_prev1) and (mmh_curr > mmh_prev1)
            mmh_reversal_sell = (mmh_prev3 < mmh_prev2 < mmh_prev1) and (mmh_curr < mmh_prev1)

            if cfg.DEBUG_MODE and (mmh_reversal_buy or mmh_reversal_sell):
                logger.debug(f"📊 {self.pair} MMH Sequence: "
                           f"{'BUY' if mmh_reversal_buy else 'SELL'} "
                           f"prev3={mmh_prev3:.4f} "
                           f"prev2={mmh_prev2:.4f} "
                           f"prev1={mmh_prev1:.4f} "
                           f"curr={mmh_curr:.4f}")

            # RMA conditions (intentional mixed timeframe)
            close_above_rma50 = close_curr > rma50_curr
            close_below_rma50 = close_curr < rma50_curr
            close_above_rma200 = close_curr > rma200_curr
            close_below_rma200 = close_curr < rma200_curr

            srsi_cross_up_50 = (smooth_rsi_prev <= 50) and (smooth_rsi_curr > 50)
            srsi_cross_down_50 = (smooth_rsi_prev >= 50) and (smooth_rsi_curr < 50)

            # Cloud state
            cloud_state = ("green" if cfg.CIRRUS_CLOUD_ENABLED and upw.iloc[self.last_i] and not dnw.iloc[self.last_i]
                          else "red" if cfg.CIRRUS_CLOUD_ENABLED and dnw.iloc[self.last_i] and not upw.iloc[self.last_i]
                          else "neutral")

            # Condition dictionaries
            conditions = {
                "buy_mmh_reversal": {
                    "mmh_reversal_buy": mmh_reversal_buy,
                    "close_above_rma50": close_above_rma50,
                    "close_above_rma200": close_above_rma200,
                    "magical_hist_curr>0": mmh_curr > 0,
                    "cloud_green": (cloud_state == "green"),
                    "strong_bullish_close": strong_bullish_close,
                },
                "sell_mmh_reversal": {
                    "mmh_reversal_sell": mmh_reversal_sell,
                    "close_below_rma50": close_below_rma50,
                    "close_below_rma200": close_below_rma200,
                    "magical_hist_curr<0": mmh_curr < 0,
                    "cloud_red": (cloud_state == "red"),
                    "strong_bearish_close": strong_bearish_close,
                },
                "buy_srsi": {
                    "srsi_cross_up_50": srsi_cross_up_50,
                    "ppo_above_signal": ppo_above_signal,
                    "ppo_below_030": ppo_below_030,
                    "close_above_rma50": close_above_rma50,
                    "close_above_rma200": close_above_rma200,
                    "cloud_green": (cloud_state == "green"),
                    "strong_bullish_close": strong_bullish_close,
                    "magical_hist_curr>0": mmh_curr > 0,
                },
                "sell_srsi": {
                    "srsi_cross_down_50": srsi_cross_down_50,
                    "ppo_below_signal": ppo_below_signal,
                    "ppo_above_minus030": ppo_above_minus030,
                    "close_below_rma50": close_below_rma50,
                    "close_below_rma200": close_below_rma200,
                    "cloud_red": (cloud_state == "red"),
                    "strong_bearish_close": strong_bearish_close,
                    "magical_hist_curr<0": mmh_curr < 0,
                },
                "long_zero": {
                    "ppo_cross_above_zero": ppo_cross_above_zero,
                    "ppo_above_signal": ppo_above_signal,
                    "close_above_rma50": close_above_rma50,
                    "close_above_rma200": close_above_rma200,
                    "cloud_green": (cloud_state == "green"),
                    "strong_bullish_close": strong_bullish_close,
                    "magical_hist_curr>0": mmh_curr > 0,
                },
                "long_011": {
                    "ppo_cross_above_011": ppo_cross_above_011,
                    "ppo_above_signal": ppo_above_signal,
                    "close_above_rma50": close_above_rma50,
                    "close_above_rma200": close_above_rma200,
                    "cloud_green": (cloud_state == "green"),
                    "strong_bullish_close": strong_bullish_close,
                    "magical_hist_curr>0": mmh_curr > 0,
                },
                "short_zero": {
                    "ppo_cross_below_zero": ppo_cross_below_zero,
                    "ppo_below_signal": ppo_below_signal,
                    "close_below_rma50": close_below_rma50,
                    "close_below_rma200": close_below_rma200,
                    "cloud_red": (cloud_state == "red"),
                    "strong_bearish_close": strong_bearish_close,
                    "magical_hist_curr<0": mmh_curr < 0,
                },
                "short_011": {
                    "ppo_cross_below_minus011": ppo_cross_below_minus011,
                    "ppo_below_signal": ppo_below_signal,
                    "close_below_rma50": close_below_rma50,
                    "close_below_rma200": close_below_rma200,
                    "cloud_red": (cloud_state == "red"),
                    "strong_bearish_close": strong_bearish_close,
                    "magical_hist_curr<0": mmh_curr < 0,
                }
            }

            # Evaluate conditions
            for signal_name, conds in conditions.items():
                if all(conds.values()):
                    return self._create_signal(signal_name, mmh_curr, smooth_rsi_curr, 
                                             ppo_curr, close_curr, last_state)

            # No signal
            last_state_value = last_state.get("state") if last_state else None
            if last_state_value and last_state_value != 'NO_SIGNAL':
                return {"state": "NO_SIGNAL", "ts": int(time.time())}
            return last_state

        except Exception as e:
            logger.exception(f"Error evaluating {self.pair}: {e}")
            return None

    def _create_signal(self, signal_type: str, mmh_val: float, srsi_val: float, 
                      ppo_val: float, price: float, last_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Create signal dictionary with metadata."""
        ist = pytz.timezone("Asia/Kolkata")
        formatted_time = datetime.now(ist).strftime('%d-%m-%Y @ %H:%M IST')
        
        # Check if same signal was sent recently
        last_state_value = last_state.get("state") if last_state else None
        if signal_type == last_state_value:
            logger.debug(f"Idempotency: {self.pair} signal remains {signal_type}.")
            return last_state

        icons = {
            "buy_mmh_reversal": "🟢",
            "sell_mmh_reversal": "🔴",
            "buy_srsi": "⬆️",
            "sell_srsi": "⬇️",
            "long_zero": "🟢",
            "long_011": "🟢",
            "short_zero": "🔴",
            "short_011": "🔴"
        }
        
        signal_names = {
            "buy_mmh_reversal": "BUY (MMH Reversal)",
            "sell_mmh_reversal": "SELL (MMH Reversal)",
            "buy_srsi": "BUY (SRSI 50)",
            "sell_srsi": "SELL (SRSI 50)",
            "long_zero": "LONG (PPO 0)",
            "long_011": "LONG (PPO 0.11)",
            "short_zero": "SHORT (PPO 0)",
            "short_011": "SHORT (PPO -0.11)"
        }

        value_map = {
            "buy_mmh_reversal": f"MMH 15m Reversal Up ({mmh_val:.5f})",
            "sell_mmh_reversal": f"MMH 15m Reversal Down ({mmh_val:.5f})",
            "buy_srsi": f"SRSI 15m Cross Up 50 ({srsi_val:.2f})",
            "sell_srsi": f"SRSI 15m Cross Down 50 ({srsi_val:.2f})",
            "long_zero": f"PPO crossing above 0 ({ppo_val:.2f})",
            "long_011": f"PPO crossing above 0.11 ({ppo_val:.2f})",
            "short_zero": f"PPO crossing below 0 ({ppo_val:.2f})",
            "short_011": f"PPO crossing below -0.11 ({ppo_val:.2f})"
        }

        icon = icons.get(signal_type, "📊")
        name = signal_names.get(signal_type, signal_type)
        value_text = value_map.get(signal_type, "")

        message = f"{icon} {self.pair} - {name}\n{value_text}\nPrice: ${price:,.2f}\n{formatted_time}"
        
        return {
            "state": signal_type,
            "ts": int(time.time()),
            "message": message,
            "price": price,
            "mmh": mmh_val,
            "srsi": srsi_val,
            "ppo": ppo_val
        }

# -------------------------
# Product mapper
# -------------------------
class ProductMapper:
    @staticmethod
    def build(api_products: dict) -> Dict[str, dict]:
        products_map = {}
        if not api_products or not api_products.get("result"):
            logger.error("No products in API result")
            return products_map
        
        seen_symbols = set()
        for p in api_products.get("result", []):
            try:
                if p.get("contract_type") != "perpetual_futures":
                    continue
                
                symbol = p.get("symbol", "")
                symbol_norm = symbol.replace("_USDT", "USD").replace("USDT", "USD")
                
                if symbol_norm in seen_symbols:
                    continue
                    
                for pair_name in cfg.PAIRS:
                    if symbol_norm == pair_name or symbol_norm.replace("_", "") == pair_name:
                        products_map[pair_name] = {
                            "id": p.get("id"),
                            "symbol": p.get("symbol"),
                            "contract_type": p.get("contract_type")
                        }
                        seen_symbols.add(symbol_norm)
                        logger.debug(f"📌 Mapped {pair_name} -> {symbol}")
                        break
            except Exception as e:
                logger.debug(f"Error mapping product: {e}")
                continue
        
        logger.info(f"📋 Mapped {len(products_map)} tradable pairs")
        return products_map

# -------------------------
# Telegram sender with queue management
# -------------------------
class TelegramAlertQueue:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self._last_sent = 0.0
        self.messages_sent = 0
        self.errors = 0

    async def send(self, session: aiohttp.ClientSession, message: str, 
                   priority: str = "normal") -> bool:
        """Send alert with rate limiting and priority handling."""
        if not self.token or not self.chat_id or self.token == "xxxx" or self.chat_id == "xxxx":
            logger.debug("Telegram not configured; skipping alert.")
            return False

        # Rate limiting
        now = time.time()
        time_since = now - self._last_sent
        if time_since < cfg.TELEGRAM_RATE_LIMIT:
            await asyncio.sleep(cfg.TELEGRAM_RATE_LIMIT - time_since)
        
        self._last_sent = time.time()
        success = await self._send_with_retries(session, message)
        
        if success:
            self.messages_sent += 1
        else:
            self.errors += 1
        
        return success

    async def _send_once(self, session: aiohttp.ClientSession, message: str) -> bool:
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        data = {"chat_id": self.chat_id, "text": message, "parse_mode": "HTML"}
        
        try:
            async with session.post(url, data=data, timeout=10) as resp:
                if resp.status == 429:
                    retry_after = int(resp.headers.get("Retry-After", 30))
                    logger.warning(f"⚠️ Telegram rate limited. Retry after {retry_after}s")
                    await asyncio.sleep(retry_after)
                    return False
                
                try:
                    js = await resp.json(content_type=None)
                except Exception:
                    text = await resp.text()
                    js = {"ok": False, "status": resp.status, "text": text[:200]}
                
                if js.get("ok"):
                    logger.info("✅ Alert sent successfully")
                    return True
                
                logger.warning(f"Telegram API error: {js}")
                return False
        except Exception as e:
            logger.exception(f"Telegram send error: {e}")
            return False

    async def _send_with_retries(self, session: aiohttp.ClientSession, message: str) -> bool:
        for attempt in range(1, cfg.TELEGRAM_RETRIES + 1):
            if await self._send_once(session, message):
                return True
            
            backoff = (cfg.TELEGRAM_BACKOFF_BASE ** (attempt - 1)) + random.uniform(0, 0.3)
            logger.debug(f"Telegram retry {attempt}/{cfg.TELEGRAM_RETRIES} in {backoff:.2f}s")
            await asyncio.sleep(backoff)
        
        logger.error("❌ Telegram send failed after all retries")
        return False

    def get_stats(self) -> Dict[str, Any]:
        return {
            "messages_sent": self.messages_sent,
            "errors": self.errors,
            "success_rate": self.messages_sent / max(1, self.messages_sent + self.errors)
        }

# -------------------------
# Main Bot Orchestrator
# -------------------------
class MACDBot:
    def __init__(self):
        self.fetcher = DataFetcher(cfg.DELTA_API_BASE, max_parallel=cfg.MAX_PARALLEL_FETCH, 
                                  timeout=cfg.HTTP_TIMEOUT)
        self.state_db = StateDB(cfg.STATE_DB_PATH)
        self.telegram = TelegramAlertQueue(cfg.TELEGRAM_BOT_TOKEN, cfg.TELEGRAM_CHAT_ID)
        self.run_count = 0
        self.errors = 0

    async def run_once(self, session: aiohttp.ClientSession) -> bool:
        """Execute one full bot cycle."""
        start_time = time.time()
        self.run_count += 1
        
        logger.info(f"🚀 Starting run #{self.run_count} ({len(cfg.PAIRS)} pairs)")
        
        # Memory check
        try:
            memory_mb = psutil.Process(os.getpid()).memory_info().rss / 1_000_000
            if memory_mb > (cfg.MEMORY_LIMIT_BYTES / 1_000_000):
                logger.critical(f"🚨 High memory usage: {memory_mb:.1f}MB")
                raise SystemExit(3)
            logger.debug(f"💾 Memory usage: {memory_mb:.1f}MB")
        except Exception as e:
            logger.debug(f"Memory check failed: {e}")

        # Prune old states
        try:
            deleted = self.state_db.prune_old_records(cfg.STATE_EXPIRY_DAYS)
            if deleted > 0:
                logger.info(f"🧹 Pruned {deleted} old state records")
        except Exception as e:
            logger.warning(f"Prune failed: {e}")

        # Load previous states
        last_states = self.state_db.load_all()
        logger.info(f"📂 Loaded {len(last_states)} previous states")

        # Test message if enabled
        if cfg.SEND_TEST_MESSAGE and self.run_count == 1:
            await self._send_test_message(session)

        # Fetch products
        products = await self.fetcher.fetch_products(session)
        if not products:
            logger.error("❌ Failed to fetch products")
            self.errors += 1
            return False

        products_map = ProductMapper.build(products)
        if not products_map:
            logger.error("❌ No tradable pairs found")
            self.errors += 1
            return False

        # Process pairs in batches
        pairs_to_process = [p for p in cfg.PAIRS if p in products_map]
        batch_size = min(cfg.BATCH_SIZE, len(pairs_to_process))
        logger.info(f"⚙️ Processing {len(pairs_to_process)} pairs in batches of {batch_size}")
        
        results = []
        for i in range(0, len(pairs_to_process), batch_size):
            batch = pairs_to_process[i:i+batch_size]
            logger.info(f"📦 Processing batch {i//batch_size + 1}/{len(pairs_to_process)//batch_size + 1}: {batch}")
            batch_results = await self._process_batch(session, batch, products_map, last_states)
            results.extend(batch_results)
            
            # Inter-batch delay
            if i + batch_size < len(pairs_to_process):
                await asyncio.sleep(random.uniform(0.5, 1.5))

        # Persist results
        await self._persist_results(results)
        
        # Log summary
        duration = time.time() - start_time
        logger.info(f"✅ Run #{self.run_count} complete in {duration:.2f}s")
        logger.info(f"📊 Results: {len(results)} signals evaluated | Errors: {self.errors}")
        
        # Log component stats
        if cfg.DEBUG_MODE:
            logger.debug(f"Fetcher stats: {self.fetcher.get_stats()}")
            logger.debug(f"Telegram stats: {self.telegram.get_stats()}")
        
        return True

    async def _process_batch(self, session: aiohttp.ClientSession, batch: List[str], 
                            products_map: Dict[str, dict], last_states: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        """Process a batch of pairs concurrently."""
        tasks = []
        for pair in batch:
            prod = products_map.get(pair)
            if not prod:
                logger.warning(f"No product for {pair}")
                continue
            
            sp = cfg.SPECIAL_PAIRS.get(pair, {})
            limits = {
                "limit_15m": sp.get("limit_15m", 210),
                "min_required_15m": sp.get("min_required", 180),
                "limit_5m": sp.get("limit_5m", 300),
                "min_required_5m": sp.get("min_required_5m", 200)
            }
            
            last_state = last_states.get(pair)
            task = asyncio.create_task(
                self._check_pair(session, pair, prod, limits, last_state)
            )
            tasks.append(task)

        results = []
        for task in tasks:
            try:
                res = await task
                if res:
                    results.append(res)
            except Exception as e:
                logger.exception(f"Batch task error: {e}")
                self.errors += 1
        
        return results

    async def _check_pair(self, session: aiohttp.ClientSession, pair: str, prod: dict, 
                         limits: dict, last_state: Optional[Dict[str, Any]]) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Check a single pair for signals."""
        try:
            symbol = prod["symbol"]
            
            # Fetch candles
            res15, res5 = await asyncio.gather(
                self.fetcher.fetch_candles(session, symbol, "15", limits["limit_15m"]),
                self.fetcher.fetch_candles(session, symbol, "5", limits["limit_5m"])
            )
            
            df_15m = parse_candles_result(res15)
            df_5m = parse_candles_result(res5)
            
            # Validate
            if not CandleValidator.validate(df_15m, pair, "15m"):
                logger.warning(f"❌ 15m data validation failed for {pair}")
                return None
            if not CandleValidator.validate(df_5m, pair, "5m"):
                logger.warning(f"❌ 5m data validation failed for {pair}")
                return None
            
            # Check data sufficiency
            if len(df_15m) < limits["min_required_15m"] + 4:
                logger.warning(f"Insufficient 15m data for {pair}: {len(df_15m)}/{limits['min_required_15m'] + 4}")
                return None
            if len(df_5m) < limits["min_required_5m"] + 4:
                logger.warning(f"Insufficient 5m data for {pair}: {len(df_5m)}/{limits['min_required_5m'] + 4}")
                return None
            
            # Evaluate signals
            evaluator = SignalEvaluator(pair, df_15m, df_5m)
            signal = evaluator.evaluate(last_state)
            
            if not signal:
                return None
            
            # Check alert cooldown
            if signal.get("message"):
                cooldown = cfg.SPECIAL_PAIRS.get(pair, {}).get("dead_mans_cooldown", cfg.ALERT_COOLDOWN_SECONDS)
                if self.state_db.is_alert_cooldown_active(pair, cooldown):
                    logger.debug(f"⏳ Alert cooldown active for {pair} ({cooldown}s)")
                    return None
                
                # Send alert
                success = await self.telegram.send(session, signal["message"])
                if success:
                    self.state_db.update_alert_timestamp(pair)
            
            return pair, signal

        except Exception as e:
            logger.exception(f"Error checking {pair}: {e}")
            self.errors += 1
            return None

    async def _send_test_message(self, session: aiohttp.ClientSession):
        """Send startup test message."""
        ist = pytz.timezone("Asia/Kolkata")
        current_dt = datetime.now(ist)
        test_msg = (
            f"🔔 <b>MACD Bot Started</b>\n"
            f"Time: {current_dt.strftime('%d-%m-%Y @ %H:%M IST')}\n"
            f"Pairs: {len(cfg.PAIRS)} | Debug: {cfg.DEBUG_MODE}\n"
            f"Batch Size: {cfg.BATCH_SIZE} | Parallel: {cfg.MAX_PARALLEL_FETCH}"
        )
        await self.telegram.send(session, test_msg, priority="high")

    async def _persist_results(self, results: List[Tuple[str, Dict[str, Any]]]):
        """Persist signal states to database."""
        updates = 0
        alerts_sent = 0
        
        for pair, signal in results:
            if not isinstance(signal, dict):
                continue
            
            prev = self.state_db.get(pair)
            if prev != signal:
                self.state_db.set(pair, signal.get("state"), signal.get("ts"))
                updates += 1
            
            if "message" in signal:
                alerts_sent += 1
        
        self.state_db.set_metadata("last_success_run", str(int(time.time())))
        logger.info(f"💾 Persisted: {updates} state updates, {alerts_sent} alerts")

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive bot statistics."""
        return {
            "run_count": self.run_count,
            "errors": self.errors,
            "error_rate": self.errors / max(1, self.run_count),
            "fetcher": self.fetcher.get_stats(),
            "telegram": self.telegram.get_stats()
        }

# -------------------------
# Main execution with timeout
# -------------------------
async def run_with_timeout(timeout_seconds: int):
    """Run bot with timeout protection."""
    bot = MACDBot()
    
    try:
        connector = TCPConnector(limit=cfg.TCP_CONN_LIMIT, ssl=False)
        async with aiohttp.ClientSession(connector=connector) as session:
            await asyncio.wait_for(bot.run_once(session), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        logger.error(f"⏱️ Bot run timed out after {timeout_seconds}s")
        raise SystemExit(3)
    except SystemExit:
        raise
    except Exception as e:
        logger.exception(f"💥 Unhandled exception in bot run: {e}")
        raise SystemExit(4)

# -------------------------
# Signal handling
# -------------------------
stop_requested = False

def handle_signal(signum, frame):
    global stop_requested
    logger.info(f"🛑 Received signal {signum} — graceful shutdown initiated")
    stop_requested = True

signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

# -------------------------
# CLI Entry Point
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="MACD/PPO Bot - Production Ready")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--once", action="store_true", help="Run once and exit")
    group.add_argument("--loop", type=int, metavar="SECONDS", help="Run in loop every N seconds")
    parser.add_argument("--config", type=str, help="Config file path", default=CONFIG_FILE)
    args = parser.parse_args()

    # PID lock
    pid_lock = PidFileLock("/tmp/macd_bot.pid")
    if not pid_lock.acquire():
        sys.exit(2)

    try:
        if args.once:
            asyncio.run(run_with_timeout(cfg.RUN_TIMEOUT_SECONDS))
        elif args.loop:
            interval = max(30, args.loop)
            logger.info(f"🔄 Starting loop mode (interval={interval}s)")
            while not stop_requested:
                start = time.time()
                try:
                    asyncio.run(run_with_timeout(cfg.RUN_TIMEOUT_SECONDS))
                except SystemExit as e:
                    logger.error(f"Run exited with code {e.code}; continuing loop")
                except Exception:
                    logger.exception("Run failed, continuing loop")
                
                elapsed = time.time() - start
                sleep_time = max(0, interval - elapsed)
                
                # Sleep in small chunks for responsive shutdown
                slept = 0.0
                while slept < sleep_time and not stop_requested:
                    chunk = min(2.0, sleep_time - slept)
                    time.sleep(chunk)
                    slept += chunk
        else:
            asyncio.run(run_with_timeout(cfg.RUN_TIMEOUT_SECONDS))
    finally:
        pid_lock.release()

if __name__ == "__main__":
    main()
