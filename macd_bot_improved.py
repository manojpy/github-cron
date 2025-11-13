#!/usr/bin/env python3
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
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timedelta
from pathlib import Path
from contextlib import contextmanager

import aiohttp
import pandas as pd
import numpy as np
import pytz
from aiohttp import ClientConnectorError, ClientResponseError
from logging.handlers import RotatingFileHandler

# -------------------------
# Configuration with Validation
# -------------------------
DEFAULT_CONFIG = {
    "TELEGRAM_BOT_TOKEN": os.environ.get("TELEGRAM_BOT_TOKEN", "8462496498:AAHYZ4xDIHvrVRjmCmZyoPhupCjRaRgiITc"),
    "TELEGRAM_CHAT_ID": os.environ.get("TELEGRAM_CHAT_ID", "203813932"),
    "DEBUG_MODE": os.environ.get("DEBUG_MODE", "False").lower() == "true",
    "SEND_TEST_MESSAGE": os.environ.get("SEND_TEST_MESSAGE", "False").lower() == "true",
    "DELTA_API_BASE": "https://api.india.delta.exchange",
    "PAIRS": ["BTCUSD", "ETHUSD", "SOLUSD", "AVAXUSD", "BCHUSD", "XRPUSD", "BNBUSD", "LTCUSD", "DOTUSD", "ADAUSD", "SUIUSD", "AAVEUSD"],
    "SPECIAL_PAIRS": {
        "SOLUSD": {"limit_15m": 210, "min_required": 180, "limit_5m": 300, "min_required_5m": 200}
    },
    "PPO_FAST": 7,
    "PPO_SLOW": 16,
    "PPO_SIGNAL": 5,
    "PPO_USE_SMA": False,
    "RMA_50_PERIOD": 50,
    "RMA_200_PERIOD": 200,
    "CIRRUS_CLOUD_ENABLED": True,
    "X1": 22, "X2": 9, "X3": 15, "X4": 5,
    "SRSI_RSI_LEN": 21, "SRSI_KALMAN_LEN": 5, "SRSI_EMA_LEN": 5,
    "STATE_DB_PATH": os.environ.get("STATE_DB_PATH", "macd_state.sqlite"),
    "LOG_FILE": os.environ.get("LOG_FILE", "macd_bot.log"),
    "MAX_PARALLEL_FETCH": 4,  # Reduced for stability
    "HTTP_TIMEOUT": 15,
    "CANDLE_FETCH_RETRIES": 3,
    "CANDLE_FETCH_BACKOFF": 1.5,
    "JITTER_MIN": 0.1,  # Increased for better distribution
    "JITTER_MAX": 0.8,
    "STATE_EXPIRY_DAYS": int(os.environ.get("STATE_EXPIRY_DAYS", "30")),
    "RUN_TIMEOUT_SECONDS": int(os.environ.get("RUN_TIMEOUT_SECONDS", "600")),  # 10 min max
    "BATCH_SIZE": int(os.environ.get("BATCH_SIZE", "4")),  # Process pairs in batches
}

# Load and validate config
CONFIG_FILE = os.getenv("CONFIG_FILE", "config.json")
config = DEFAULT_CONFIG.copy()

if Path(CONFIG_FILE).exists():
    try:
        with open(CONFIG_FILE, "r") as f:
            user_cfg = json.load(f)
        config.update(user_cfg)
        print(f"✅ Loaded configuration from {CONFIG_FILE}")
    except Exception as e:
        print(f"❌ Error parsing {CONFIG_FILE}: {e}")
        sys.exit(1)
else:
    print(f"⚠️  Config file {CONFIG_FILE} not found, using defaults.")

# Override with env vars (highest priority)
for key in ["DEBUG_MODE", "SEND_TEST_MESSAGE", "STATE_DB_PATH", "LOG_FILE"]:
    env_val = os.getenv(key)
    if env_val is not None:
        if key in ["DEBUG_MODE", "SEND_TEST_MESSAGE"]:
            config[key] = env_val.lower() == "true"
        else:
            config[key] = env_val

# Validate critical config
if not config["PAIRS"]:
    print("❌ PAIRS list cannot be empty")
    sys.exit(1)

if config["MAX_PARALLEL_FETCH"] < 1 or config["MAX_PARALLEL_FETCH"] > 10:
    print("⚠️  MAX_PARALLEL_FETCH should be between 1-10, setting to 4")
    config["MAX_PARALLEL_FETCH"] = 4

cfg = config

# -------------------------
# PID File Lock for Cron
# -------------------------
class PidFileLock:
    def __init__(self, path: str = "/tmp/macd_bot.pid"):
        self.path = path
        self.fd = None

    def acquire(self) -> bool:
        """Acquire exclusive lock. Returns True if successful."""
        try:
            self.fd = open(self.path, 'w')
            fcntl.flock(self.fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.fd.write(str(os.getpid()))
            self.fd.flush()
            atexit.register(self.release)
            return True
        except (IOError, OSError):
            if self.fd:
                self.fd.close()
            return False

    def release(self):
        """Release lock and remove PID file."""
        try:
            if self.fd:
                fcntl.flock(self.fd.fileno(), fcntl.LOCK_UN)
                self.fd.close()
            if os.path.exists(self.path):
                os.unlink(self.path)
        except:
            pass

# -------------------------
# Logger Setup
# -------------------------
logger = logging.getLogger("macd_bot")
logger.setLevel(logging.DEBUG if cfg["DEBUG_MODE"] else logging.INFO)
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

# Console handler
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG if cfg["DEBUG_MODE"] else logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

# Rotating file handler
try:
    log_dir = os.path.dirname(cfg["LOG_FILE"]) or "."
    os.makedirs(log_dir, exist_ok=True)
    fh = RotatingFileHandler(cfg["LOG_FILE"], maxBytes=5_000_000, backupCount=5, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
except Exception as e:
    logger.warning(f"Could not set up rotating log file: {e}")

# -------------------------
# SQLite State Store (Thread-Safe)
# -------------------------
class StateDB:
    def __init__(self, db_path: str):
        self.db_path = db_path
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

    @contextmanager
    def get_connection(self):
        """Context manager for SQLite connections."""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def _ensure_tables(self, conn: sqlite3.Connection):
        """Create tables if they don't exist."""
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
        # Create cache table for indicators
        cur.execute("""
            CREATE TABLE IF NOT EXISTS indicator_cache (
                pair TEXT,
                timeframe TEXT,
                timestamp INTEGER,
                indicators TEXT,
                PRIMARY KEY (pair, timeframe, timestamp)
            )
        """)
        conn.commit()

    def load_all(self) -> Dict[str, Dict[str, Any]]:
        with self.get_connection() as conn:
            self._ensure_tables(conn)
            cur = conn.cursor()
            cur.execute("SELECT pair, state, ts FROM states")
            rows = cur.fetchall()
            return {r[0]: {"state": r[1], "ts": int(r[2] or 0)} for r in rows}

    def get(self, pair: str) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT state, ts FROM states WHERE pair = ?", (pair,))
            r = cur.fetchone()
            if not r:
                return None
            return {"state": r[0], "ts": int(r[1] or 0)}

    def set(self, pair: str, state: str, ts: Optional[int] = None):
        ts = int(ts or time.time())
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("INSERT OR REPLACE INTO states(pair, state, ts) VALUES (?, ?, ?)",
                        (pair, state, ts))
            conn.commit()

    def get_metadata(self, key: str) -> Optional[str]:
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT value FROM metadata WHERE key = ?", (key,))
            r = cur.fetchone()
            return r[0] if r else None

    def set_metadata(self, key: str, value: str):
        with self.get_connection() as conn:
            cur = conn.cursor()
            cur.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)", 
                        (key, value))
            conn.commit()

    def prune_old_records(self, expiry_days: int) -> int:
        """Prune old state records. Returns number of deleted rows."""
        if expiry_days <= 0:
            logger.debug("Pruning disabled (STATE_EXPIRY_DAYS <= 0)")
            return 0
        
        with self.get_connection() as conn:
            cur = conn.cursor()
            
            # Check if already pruned today
            cur.execute("SELECT value FROM metadata WHERE key='last_prune'")
            row = cur.fetchone()
            from datetime import timezone
            today = datetime.now(timezone.utc).date()
            
            if row:
                try:
                    last_prune_date = datetime.fromisoformat(row[0]).date()
                    if last_prune_date >= today:
                        logger.debug("Daily prune already completed — skipping.")
                        return 0
                except Exception:
                    pass
            
            # Delete old records
            cutoff = int(time.time()) - (expiry_days * 86400)
            cur.execute("DELETE FROM states WHERE ts < ?", (cutoff,))
            deleted = cur.rowcount
            
            # Update metadata
            cur.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES ('last_prune', ?)", 
                       (datetime.utcnow().isoformat(),))
            
            # VACUUM if needed
            if deleted > 0:
                try:
                    cur.execute("VACUUM")
                except Exception as e:
                    logger.warning(f"VACUUM failed: {e}")
            
            conn.commit()
            return deleted

# -------------------------
# Circuit Breaker for API Calls
# -------------------------
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, timeout: int = 300):
        self.failure_count = 0
        self.last_failure_time = None
        self.threshold = failure_threshold
        self.timeout = timeout
        self.lock = asyncio.Lock()

    async def is_open(self) -> bool:
        async with self.lock:
            if self.failure_count < self.threshold:
                return False
            if time.time() - self.last_failure_time > self.timeout:
                self.reset()
                return False
            return True

    async def call(self, func, *args, **kwargs):
        if await self.is_open():
            raise Exception(f"Circuit breaker is OPEN ({self.failure_count} failures)")
        try:
            result = await func(*args, **kwargs)
            await self.success()
            return result
        except Exception as e:
            await self.failure()
            raise e

    async def failure(self):
        async with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            logger.warning(f"Circuit breaker: failure {self.failure_count}/{self.threshold}")

    async def success(self):
        async with self.lock:
            if self.failure_count > 0:
                self.failure_count = 0
                logger.info("Circuit breaker: reset after success")

    def reset(self):
        self.failure_count = 0
        self.last_failure_time = None
        logger.info("Circuit breaker: reset")

# -------------------------
# Telegram Rate Limiter
# -------------------------
class TelegramQueue:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.queue = asyncio.Queue()
        self.last_sent = 0
        self.rate_limit = 0.1  # 10 messages per second max

    async def send(self, session: aiohttp.ClientSession, message: str) -> bool:
        """Queue message and respect rate limits."""
        await self.queue.put(message)
        
        now = time.time()
        time_since_last = now - self.last_sent
        if time_since_last < self.rate_limit:
            await asyncio.sleep(self.rate_limit - time_since_last)
        
        self.last_sent = time.time()
        return await self._send_raw(session, message)

    async def _send_raw(self, session: aiohttp.ClientSession, message: str) -> bool:
        """Send message without queuing."""
        if self.token == 'xxxx' or self.chat_id == 'xxxx':
            logger.debug("Telegram not configured; skipping alert.")
            return False
        
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        data = {"chat_id": self.chat_id, "text": message}
        
        try:
            async with session.post(url, data=data, timeout=10) as resp:
                if resp.status == 429:  # Rate limited
                    retry_after = int(resp.headers.get('Retry-After', 30))
                    logger.warning(f"Telegram rate limited. Waiting {retry_after}s")
                    await asyncio.sleep(retry_after)
                    return await self._send_raw(session, message)  # Retry once
                
                js = await resp.json()
                if js.get("ok"):
                    logger.info("✅ Alert sent successfully")
                    return True
                else:
                    logger.error(f"Telegram API error: {js}")
                    return False
        except Exception as e:
            logger.error(f"Telegram send error: {e}")
            return False

# -------------------------
# HTTP Helpers with Circuit Breaker
# -------------------------
async def async_fetch_json(session: aiohttp.ClientSession, url: str, params: dict = None,
                           retries: int = 3, backoff: float = 1.5, timeout: int = 15,
                           circuit_breaker: Optional[CircuitBreaker] = None) -> Optional[dict]:
    """Fetch JSON with retry/backoff and circuit breaker."""
    if circuit_breaker and await circuit_breaker.is_open():
        logger.error(f"Circuit breaker open, skipping {url}")
        return None
    
    for attempt in range(1, retries + 1):
        try:
            async with session.get(url, params=params, timeout=timeout) as resp:
                text = await resp.text()
                try:
                    data = await resp.json()
                except Exception:
                    data = None
                if resp.status >= 400:
                    logger.debug(f"HTTP {resp.status} for {url} (attempt {attempt}): {text[:200]}")
                    raise ClientResponseError(resp.request_info, resp.history, status=resp.status)
                
                if circuit_breaker:
                    await circuit_breaker.success()
                return data
        except (asyncio.TimeoutError, ClientConnectorError, ClientResponseError) as e:
            logger.debug(f"Fetch error {e} for {url} (attempt {attempt})")
            if attempt == retries:
                logger.error(f"Failed to fetch {url} after {retries} attempts.")
                if circuit_breaker:
                    await circuit_breaker.failure()
                return None
            await asyncio.sleep(backoff * (attempt ** 1.2) + random.uniform(0, 0.2))
        except Exception as e:
            logger.exception(f"Unexpected error fetching {url}: {e}")
            if circuit_breaker:
                await circuit_breaker.failure()
            return None
    return None

# -------------------------
# DataFetcher
# -------------------------
class DataFetcher:
    def __init__(self, api_base: str, max_parallel: int = 4, timeout: int = 15):
        self.api_base = api_base.rstrip("/")
        self.semaphore = asyncio.Semaphore(max_parallel)
        self.timeout = timeout
        self._cache = {}
        self.circuit_breaker = CircuitBreaker()

    async def fetch_products(self, session: aiohttp.ClientSession) -> Optional[dict]:
        url = f"{self.api_base}/v2/products"
        async with self.semaphore:
            return await async_fetch_json(session, url, retries=3, backoff=1.5, 
                                        timeout=self.timeout, circuit_breaker=self.circuit_breaker)

    async def fetch_candles(self, session: aiohttp.ClientSession, symbol: str, resolution: str, limit: int):
        """Fetch candles with caching."""
        key = f"candles:{symbol}:{resolution}:{limit}"
        if key in self._cache:
            cache_time, data = self._cache[key]
            if time.time() - cache_time < 60:  # Cache for 60 seconds
                return data
        
        # Jitter to reduce thundering herd
        await asyncio.sleep(random.uniform(cfg["JITTER_MIN"], cfg["JITTER_MAX"]))
        
        url = f"{self.api_base}/v2/chart/history"
        params = {
            "resolution": resolution,
            "symbol": symbol,
            "from": int(time.time()) - (limit * int(resolution) * 60),
            "to": int(time.time())
        }
        
        async with self.semaphore:
            data = await async_fetch_json(session, url, params=params, 
                                        retries=cfg["CANDLE_FETCH_RETRIES"], 
                                        backoff=cfg["CANDLE_FETCH_BACKOFF"], 
                                        timeout=cfg["HTTP_TIMEOUT"],
                                        circuit_breaker=self.circuit_breaker)
        
        if data:
            self._cache[key] = (time.time(), data)
        return data

# -------------------------
# Data Validation
# -------------------------
def validate_candle_data(df: pd.DataFrame) -> bool:
    """Validate candle DataFrame."""
    if df.empty:
        return False
    if df['close'].astype(float).iloc[-1] <= 0:
        return False
    if df.isnull().any().any():
        logger.warning("DataFrame contains NaN values")
        return False
    return True

def parse_candles_result(result: dict) -> Optional[pd.DataFrame]:
    """Convert Delta API candle response to pandas DataFrame with validation."""
    if not result or not isinstance(result, dict):
        logger.warning("Invalid result: not a dict")
        return None
    if not result.get("success"):
        logger.warning(f"API error: {result}")
        return None
    
    res = result.get("result", {})
    required_keys = ['t', 'o', 'h', 'l', 'c', 'v']
    if not all(k in res for k in required_keys):
        logger.warning(f"Missing keys in result: {list(res.keys())}")
        return None
    
    # Check types
    for k in required_keys:
        if not isinstance(res[k], list):
            logger.warning(f"Invalid type for {k}: {type(res[k])}")
            return None
    
    try:
        min_len = min(map(len, [res[k] for k in required_keys]))
        df = pd.DataFrame({
            'timestamp': res['t'][:min_len],
            'open': res['o'][:min_len],
            'high': res['h'][:min_len],
            'low': res['l'][:min_len],
            'close': res['c'][:min_len],
            'volume': res['v'][:min_len]
        })
        
        df = df.sort_values('timestamp').drop_duplicates(subset='timestamp').reset_index(drop=True)
        df = df.astype(float)
        
        if not validate_candle_data(df):
            return None
        
        return df
    except Exception as e:
        logger.error(f"Error parsing candle data: {e}")
        return None

# -------------------------
# Indicator Functions (Unchanged Logic)
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
    result_list = [x.iloc[0]]
    for i in range(1, len(x)):
        prev_f = result_list[-1]
        curr_x = x.iloc[i]
        curr_r = max(float(r.iloc[i]), 1e-8)
        if curr_x > prev_f:
            f = prev_f if (curr_x - curr_r) < prev_f else (curr_x - curr_r)
        else:
            f = prev_f if (curr_x + curr_r) > prev_f else (curr_x + curr_r)
        result_list.append(f)
    return pd.Series(result_list, index=x.index)

def calculate_cirrus_cloud(df: pd.DataFrame):
    close = df['close'].copy()
    smrngx1x = smoothrng(close, cfg["X1"], cfg["X2"])
    smrngx1x2 = smoothrng(close, cfg["X3"], cfg["X4"])
    filtx1 = rngfilt(close, smrngx1x)
    filtx12 = rngfilt(close, smrngx1x2)
    upw = filtx1 < filtx12
    dnw = filtx1 > filtx12
    return upw, dnw, filtx1, filtx12

def kalman_filter(src: pd.Series, length: int, R=0.01, Q=0.1) -> pd.Series:
    result_list = []
    estimate = np.nan
    error_est = 1.0
    error_meas = R * max(1, length)
    Q_div_length = Q / max(1, length)
    for i in range(len(src)):
        current_src = src.iloc[i]
        if np.isnan(estimate):
            if i > 0:
                estimate = src.iloc[i-1]
            else:
                result_list.append(np.nan)
                continue
        prediction = estimate
        kalman_gain = error_est / (error_est + error_meas)
        estimate = prediction + kalman_gain * (current_src - prediction)
        error_est = (1 - kalman_gain) * error_est + Q_div_length
        result_list.append(estimate)
    return pd.Series(result_list, index=src.index)

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
    responsiveness = max(1e-5, float(responsiveness))
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

# -------------------------
# Evaluation Logic
# -------------------------
def get_last_closed_indices(df_15m: pd.DataFrame, df_5m: pd.DataFrame) -> Tuple[int, int, int]:
    """Return safe indices for last closed candles across 15m and 5m timeframes."""
    now_ts = int(time.time())

    def closed_index_for(df: pd.DataFrame, resolution_min: int) -> int:
        if df is None or df.empty:
            return -1
        last_ts = int(df['timestamp'].iloc[-1])
        current_interval_start = now_ts - (now_ts % (resolution_min * 60))
        if last_ts >= current_interval_start:
            return -2  # Current candle is open, use previous
        return -1  # Most recent candle is closed

    last_i = closed_index_for(df_15m, 15)
    prev_i = last_i - 1
    last_i_5m = closed_index_for(df_5m, 5)
    return last_i, prev_i, last_i_5m

def evaluate_pair_logic(pair_name: str, df_15m: pd.DataFrame, df_5m: pd.DataFrame, 
                       last_state_for_pair: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Evaluate indicators and return new state if changed."""
    try:
        last_i, prev_i, last_i_5m = get_last_closed_indices(df_15m, df_5m)

        # Calculate indicators
        magical_hist = calculate_magical_momentum_hist(df_15m, period=144, responsiveness=0.9)
        if magical_hist.empty or len(magical_hist) < abs(last_i) + 4:
            logger.debug(f"MMH insufficient data ({len(magical_hist)}) for {pair_name}")
            return None
        
        # Get MMH values for last 4 closed candles
        mmh_curr = float(magical_hist.iloc[last_i])
        mmh_prev1 = float(magical_hist.iloc[last_i - 1])
        mmh_prev2 = float(magical_hist.iloc[last_i - 2])
        mmh_prev3 = float(magical_hist.iloc[last_i - 3])
        
        ppo, ppo_signal = calculate_ppo(df_15m, cfg["PPO_FAST"], cfg["PPO_SLOW"], 
                                       cfg["PPO_SIGNAL"], cfg["PPO_USE_SMA"])
        rma_50 = calculate_rma(df_15m['close'], cfg["RMA_50_PERIOD"])
        rma_200 = calculate_rma(df_5m['close'], cfg["RMA_200_PERIOD"])
        upw, dnw, _, _ = calculate_cirrus_cloud(df_15m)
        smooth_rsi = calculate_smooth_rsi(df_15m, cfg["SRSI_RSI_LEN"], cfg["SRSI_KALMAN_LEN"])

        # Extract values
        ppo_curr = float(ppo.iloc[last_i])
        ppo_prev = float(ppo.iloc[prev_i])
        ppo_signal_curr = float(ppo_signal.iloc[last_i])
        ppo_signal_prev = float(ppo_signal.iloc[prev_i])
        smooth_rsi_curr = float(smooth_rsi.iloc[last_i])
        smooth_rsi_prev = float(smooth_rsi.iloc[prev_i])

        close_curr = float(df_15m['close'].iloc[last_i])
        open_curr = float(df_15m['open'].iloc[last_i])
        high_curr = float(df_15m['high'].iloc[last_i])
        low_curr = float(df_15m['low'].iloc[last_i])

        rma50_curr = float(rma_50.iloc[last_i])
        rma200_curr = float(rma_200.iloc[last_i_5m])

        # Safety check
        indicators = [ppo_curr, ppo_prev, ppo_signal_curr, ppo_signal_prev, 
                     smooth_rsi_curr, smooth_rsi_prev, rma50_curr, rma200_curr,
                     mmh_curr, mmh_prev1, mmh_prev2, mmh_prev3]
        if any(pd.isna(x) for x in indicators):
            logger.debug(f"NaN in indicators for {pair_name}, skipping")
            return None

        # Candle metrics
        total_range = high_curr - low_curr
        if total_range <= 0:
            upper_wick = lower_wick = 0.0
            strong_bullish_close = strong_bearish_close = False
        else:
            upper_wick = max(0.0, high_curr - max(open_curr, close_curr))
            lower_wick = max(0.0, min(open_curr, close_curr) - low_curr)
            bullish_candle = close_curr > open_curr
            bearish_candle = close_curr < open_curr
            strong_bullish_close = bullish_candle and (upper_wick / total_range) < 0.20
            strong_bearish_close = bearish_candle and (lower_wick / total_range) < 0.20

        # Conditions
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
        
        # MMH reversal conditions
        mmh_reversal_buy = (mmh_prev3 > mmh_prev2 and mmh_prev2 > mmh_prev1 and mmh_curr > mmh_prev1)
        mmh_reversal_sell = (mmh_prev3 < mmh_prev2 and mmh_prev2 < mmh_prev1 and mmh_curr < mmh_prev1)

        srsi_cross_up_50 = (smooth_rsi_prev <= 50) and (smooth_rsi_curr > 50)
        srsi_cross_down_50 = (smooth_rsi_prev >= 50) and (smooth_rsi_curr < 50)

        cloud_state = "neutral"
        if cfg["CIRRUS_CLOUD_ENABLED"]:
            cloud_state = ("green" if (bool(upw.iloc[last_i]) and not bool(dnw.iloc[last_i]))
                           else "red" if (bool(dnw.iloc[last_i]) and not bool(upw.iloc[last_i]))
                           else "neutral")

        # Condition sets
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

        # State determination
        current_state = None
        send_message = None
        price = close_curr
        ist = pytz.timezone('Asia/Kolkata')
        current_dt = datetime.now(ist)
        formatted_time = current_dt.strftime('%d-%m-%Y @ %H:%M IST')

        # MMH Reversal signals (highest priority)
        if all(buy_mmh_reversal_conds.values()):
            current_state = "buy_mmh_reversal"
            send_message = f"🟢 {pair_name} - BUY (MMH Reversal)\nMMH 15m Reversal Up ({mmh_curr:.5f})\nPrice: ${price:,.2f}\n{formatted_time}"
            
        elif all(sell_mmh_reversal_conds.values()):
            current_state = "sell_mmh_reversal"
            send_message = f"🔴 {pair_name} - SELL (MMH Reversal)\nMMH 15m Reversal Down ({mmh_curr:.5f})\nPrice: ${price:,.2f}\n{formatted_time}"
            
        # SRSI signals
        elif all(buy_srsi_conds.values()):
            current_state = "buy_srsi50"
            send_message = f"⬆️ {pair_name} - BUY (SRSI 50)\nSRSI 15m Cross Up 50 ({smooth_rsi_curr:.2f})\nPrice: ${price:,.2f}\n{formatted_time}"
            
        elif all(sell_srsi_conds.values()):
            current_state = "sell_srsi50"
            send_message = f"⬇️ {pair_name} - SELL (SRSI 50)\nSRSI 15m Cross Down 50 ({smooth_rsi_curr:.2f})\nPrice: ${price:,.2f}\n{formatted_time}"
 
        # PPO signals
        elif all(long0_conds.values()):
            current_state = "long_zero"
            send_message = f"🟢 {pair_name} - LONG\nPPO crossing above 0 ({ppo_curr:.2f})\nPrice: ${price:,.2f}\n{formatted_time}"
            
        elif all(long011_conds.values()):
            current_state = "long_011"
            send_message = f"🟢 {pair_name} - LONG\nPPO crossing above 0.11 ({ppo_curr:.2f})\nPrice: ${price:,.2f}\n{formatted_time}"
   
        elif all(short0_conds.values()):
            current_state = "short_zero"
            send_message = f"🔴 {pair_name} - SHORT\nPPO crossing below 0 ({ppo_curr:.2f})\nPrice: ${price:,.2f}\n{formatted_time}"
            
        elif all(short011_conds.values()):
            current_state = "short_011"
            send_message = f"🔴 {pair_name} - SHORT\nPPO crossing below -0.11 ({ppo_curr:.2f})\nPrice: ${price:,.2f}\n{formatted_time}"

        # Idempotency check
        now_ts = int(time.time())
        last_state_value = last_state_for_pair.get("state") if isinstance(last_state_for_pair, dict) else None

        if current_state is None:
            # Transition to NO_SIGNAL if previously had signal
            if last_state_value and last_state_value != 'NO_SIGNAL':
                return {"state": "NO_SIGNAL", "ts": now_ts}
            return last_state_for_pair

        if current_state == last_state_value:
            logger.debug(f"Idempotency: {pair_name} signal remains {current_state}.")
            return last_state_for_pair

        # Return new signal with message
        return {"state": current_state, "ts": now_ts, "message": send_message}

    except Exception as e:
        logger.exception(f"Error evaluating logic for {pair_name}: {e}")
        return None

# -------------------------
# Product Mapping
# -------------------------
def build_products_map_from_api_result(api_products: dict) -> Dict[str, dict]:
    """Build mapping from API products to configured pairs."""
    products_map = {}
    if not api_products or not api_products.get("result"):
        logger.error("No products in API result")
        return products_map
    
    for p in api_products['result']:
        try:
            symbol = p.get('symbol', '')
            symbol_norm = symbol.replace('_USDT', 'USD').replace('USDT', 'USD')
            if p.get('contract_type') == 'perpetual_futures':
                for pair_name in cfg["PAIRS"]:
                    if symbol_norm == pair_name or symbol_norm.replace('_', '') == pair_name:
                        products_map[pair_name] = {
                            'id': p.get('id'),
                            'symbol': p.get('symbol'),
                            'contract_type': p.get('contract_type')
                        }
                        break
        except Exception as e:
            logger.debug(f"Error processing product {p}: {e}")
            continue
    
    logger.info(f"Mapped {len(products_map)} tradable pairs")
    return products_map

# -------------------------
# Batch Processor
# -------------------------
async def process_batch(session: aiohttp.ClientSession, fetcher: DataFetcher, 
                       products_map: Dict[str, dict], batch_pairs: List[str],
                       state_db: StateDB, telegram_queue: TelegramQueue) -> List[Tuple[str, Dict[str, Any]]]:
    """Process a batch of pairs."""
    results = []
    tasks = []
    
    for pair_name in batch_pairs:
        prod_info = products_map.get(pair_name)
        if not prod_info:
            logger.warning(f"No product mapping for {pair_name}")
            continue
        
        last_state = state_db.get(pair_name)
        task = asyncio.create_task(
            check_pair(session, fetcher, products_map, pair_name, last_state, telegram_queue)
        )
        tasks.append((pair_name, task))
    
    for pair_name, task in tasks:
        try:
            result = await task
            if result:
                results.append(result)
        except Exception as e:
            logger.exception(f"Error processing {pair_name}: {e}")
    
    return results

# -------------------------
# Check a Single Pair
# -------------------------
async def check_pair(session: aiohttp.ClientSession, fetcher: DataFetcher, 
                    products_map: Dict[str, dict], pair_name: str, 
                    last_state_for_pair: Optional[Dict[str, Any]],
                    telegram_queue: TelegramQueue) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Fetch candles, compute indicators, evaluate conditions, send alert if needed."""
    try:
        prod = products_map.get(pair_name)
        if not prod:
            logger.debug(f"No product mapping for {pair_name}")
            return None

        sp = cfg["SPECIAL_PAIRS"].get(pair_name, {})
        limit_15m = sp.get("limit_15m", 210)
        min_required = sp.get("min_required", 200)
        limit_5m = sp.get("limit_5m", 300)
        min_required_5m = sp.get("min_required_5m", 250)

        symbol = prod['symbol']
        
        # Fetch both timeframes in parallel
        res15, res5 = await asyncio.gather(
            fetcher.fetch_candles(session, symbol, "15", limit_15m),
            fetcher.fetch_candles(session, symbol, "5", limit_5m)
        )

        df_15m = parse_candles_result(res15)
        df_5m = parse_candles_result(res5)

        if df_15m is None or len(df_15m) < (min_required + 2):
            logger.warning(f"Insufficient 15m data for {pair_name}: {len(df_15m) if df_15m else 0}/{min_required + 2}")
            return None
        if df_5m is None or len(df_5m) < (min_required_5m + 2):
            logger.warning(f"Insufficient 5m data for {pair_name}: {len(df_5m) if df_5m else 0}/{min_required_5m + 2}")
            return None

        # Ensure numeric data
        for df in [df_15m, df_5m]:
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        new_state = evaluate_pair_logic(pair_name, df_15m, df_5m, last_state_for_pair)
        if not new_state:
            return None

        # Send alert if message present
        message = new_state.pop("message", None)
        if message:
            await telegram_queue.send(session, message)

        return pair_name, new_state

    except Exception as e:
        logger.exception(f"Error in check_pair for {pair_name}: {e}")
        return None

# -------------------------
# Main Run Function with Timeout
# -------------------------
async def run_once_with_timeout(timeout_seconds: int = 600):
    """Run once with global timeout."""
    try:
        await asyncio.wait_for(run_once(), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        logger.error(f"❌ Run timed out after {timeout_seconds}s")
        sys.exit(3)  # Cron will see this as failure

async def run_once():
    """Main execution logic."""
    start_time = time.time()
    logger.info("="*50)
    logger.info("Starting run_once")
    
    # Initialize metrics
    metrics = {
        "pairs_processed": 0,
        "pairs_failed": 0,
        "alerts_sent": 0,
        "state_changes": 0
    }

    # Prune old records
    try:
        state_db = StateDB(cfg["STATE_DB_PATH"])
        deleted = state_db.prune_old_records(cfg["STATE_EXPIRY_DAYS"])
        if deleted > 0:
            logger.info(f"Pruned {deleted} old state records")
    except Exception as e:
        logger.warning(f"Prune failed: {e}")

    # Load last alerts
    state_db = StateDB(cfg["STATE_DB_PATH"])
    last_alerts = state_db.load_all()
    logger.info(f"Loaded {len(last_alerts)} previous states")

    # Initialize components
    fetcher = DataFetcher(cfg["DELTA_API_BASE"], max_parallel=cfg["MAX_PARALLEL_FETCH"], 
                         timeout=cfg["HTTP_TIMEOUT"])
    telegram_queue = TelegramQueue(cfg["TELEGRAM_BOT_TOKEN"], cfg["TELEGRAM_CHAT_ID"])

    async with aiohttp.ClientSession() as session:
        # Test message
        if cfg["SEND_TEST_MESSAGE"]:
            ist = pytz.timezone('Asia/Kolkata')
            current_dt = datetime.now(ist)
            test_msg = (f"🔔 Bot Started\nTime: {current_dt.strftime('%d-%m-%Y @ %H:%M IST')}\n"
                       f"Pairs: {len(cfg['PAIRS'])} | Debug: {cfg['DEBUG_MODE']}")
            await telegram_queue.send(session, test_msg)

        # Fetch products
        logger.info("Fetching products from API...")
        prod_resp = await fetcher.fetch_products(session)
        if not prod_resp:
            logger.error("❌ Failed to fetch products. Aborting.")
            sys.exit(2)
        
        products_map = build_products_map_from_api_result(prod_resp)
        if not products_map:
            logger.error("❌ No tradable pairs found. Exiting.")
            sys.exit(2)

        # Process pairs in batches
        pairs_to_process = [p for p in cfg["PAIRS"] if p in products_map]
        batch_size = cfg["BATCH_SIZE"]
        logger.info(f"Processing {len(pairs_to_process)} pairs in batches of {batch_size}")
        
        all_results = []
        for i in range(0, len(pairs_to_process), batch_size):
            batch = pairs_to_process[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}: {batch}")
            
            batch_results = await process_batch(session, fetcher, products_map, 
                                              batch, state_db, telegram_queue)
            all_results.extend(batch_results)
            metrics["pairs_processed"] += len(batch_results)
            
            # Small delay between batches
            if i + batch_size < len(pairs_to_process):
                await asyncio.sleep(random.uniform(0.5, 1.0))

        # Persist changes
        for pair_name, new_state in all_results:
            if not isinstance(new_state, dict):
                metrics["pairs_failed"] += 1
                continue
            
            prev = state_db.get(pair_name)
            if prev != new_state:
                state_db.set(pair_name, new_state.get("state"), new_state.get("ts"))
                metrics["state_changes"] += 1
                if "message" in new_state:
                    metrics["alerts_sent"] += 1

    # Log metrics
    duration = time.time() - start_time
    logger.info(f"✅ Run complete in {duration:.2f}s")
    logger.info(f"Metrics: {json.dumps(metrics, indent=2)}")
    
    state_db.close()

# -------------------------
# Signal Handler
# -------------------------
stop_requested = False
def request_stop(signum, frame):
    global stop_requested
    logger.info(f"🛑 Received signal {signum}, stopping gracefully...")
    stop_requested = True

signal.signal(signal.SIGINT, request_stop)
signal.signal(signal.SIGTERM, request_stop)

# -------------------------
# CLI Entry Point
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="MACD Trading Bot - Production Ready")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--once", action="store_true", help="Run once and exit")
    group.add_argument("--loop", type=int, metavar="SECONDS", help="Run in a loop every N seconds")
    args = parser.parse_args()

    # PID file lock to prevent concurrent runs
    pid_lock = PidFileLock("/tmp/macd_bot.pid")
    if not pid_lock.acquire():
        logger.error("❌ Another instance is running. Exiting.")
        sys.exit(2)

    try:
        if args.once:
            asyncio.run(run_once_with_timeout(cfg["RUN_TIMEOUT_SECONDS"]))
        elif args.loop:
            interval = max(30, args.loop)  # Minimum 30s
            logger.info(f"🔄 Starting loop mode (interval={interval}s). Ctrl-C to stop.")
            while not stop_requested:
                start = time.time()
                try:
                    asyncio.run(run_once_with_timeout(cfg["RUN_TIMEOUT_SECONDS"]))
                except Exception:
                    logger.exception("Unhandled exception in run_once")
                
                elapsed = time.time() - start
                to_sleep = max(0, interval - elapsed)
                if to_sleep > 0:
                    logger.debug(f"Sleeping {to_sleep:.1f}s")
                    time.sleep(to_sleep)
        else:
            # Default: single run
            asyncio.run(run_once_with_timeout(cfg["RUN_TIMEOUT_SECONDS"]))
    
    except Exception as e:
        logger.exception(f"Fatal error in main: {e}")
        sys.exit(1)
    finally:
        pid_lock.release()

if __name__ == "__main__":
    main()
