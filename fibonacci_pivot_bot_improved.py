import os
import sys
import json
import time
import asyncio
import random
import logging
import sqlite3
import traceback
import tempfile
import shutil
import fcntl  # NEW: For process locking
import html   # NEW: For sanitization
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union

import aiohttp
import pandas as pd
import numpy as np
import pytz
from aiohttp import ClientConnectorError, ClientResponseError
from logging.handlers import RotatingFileHandler

# NEW: Pydantic for config validation
try:
    from pydantic import BaseModel, validator, Field
except ImportError:
    print("⚠️  pydantic not found. Install with: pip install pydantic")
    sys.exit(4)

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
    PAIRS: List[str] = Field(default_factory=lambda: ["BTCUSD","ETHUSD","SOLUSD","AVAXUSD","BCHUSD","XRPUSD","BNBUSD","LTCUSD","DOTUSD","ADAUSD","SUIUSD","AAVEUSD"])
    SPECIAL_PAIRS: Dict[str, Dict[str, int]] = Field(default_factory=lambda: {
        "SOLUSD": {"limit_15m":210,"min_required":180,"limit_5m":300,"min_required_5m":200}
    })
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
    MAX_EXEC_TIME: int = 25  # NEW: Max execution time for Cron-Jobs.org
    USE_RMA200: bool = True  # NEW: Control 5m data fetching
    PRODUCTS_CACHE_TTL: int = 86400  # NEW: Cache TTL in seconds

    @validator("TELEGRAM_BOT_TOKEN")
    def token_not_default(cls, v):
        if not v or v == "xxxx":
            raise ValueError("TELEGRAM_BOT_TOKEN must be set and not be 'xxxx'")
        return v

    @validator("TELEGRAM_CHAT_ID")
    def chat_id_not_default(cls, v):
        if not v or v == "xxxx":
            raise ValueError("TELEGRAM_CHAT_ID must be set and not be 'xxxx'")
        return v

# -------------------------
# EXIT CODES
# -------------------------
EXIT_SUCCESS = 0
EXIT_LOCK_CONFLICT = 2
EXIT_TIMEOUT = 3
EXIT_CONFIG_ERROR = 4
EXIT_API_FAILURE = 5

# -------------------------
# LOAD CONFIGURATION
# -------------------------
def load_config() -> Config:
    """Load and validate configuration with environment override"""
    # Start with defaults
    config_dict = {
        "TELEGRAM_BOT_TOKEN": os.getenv("TELEGRAM_BOT_TOKEN", "8462496498:AAHYZ4xDIHvrVRjmCmZyoPhupCjRaRgiITc"),
        "TELEGRAM_CHAT_ID": os.getenv("TELEGRAM_CHAT_ID", "203813932"),
        "DEBUG_MODE": os.getenv("DEBUG_MODE", "True").lower() == "true",
        "SEND_TEST_MESSAGE": os.getenv("SEND_TEST_MESSAGE", "True").lower() == "true",
        "RESET_STATE": os.getenv("RESET_STATE", "False").lower() == "true",
        "STATE_DB_PATH": os.getenv("STATE_DB_PATH", "fib_state.sqlite"),
        "LOG_FILE": os.getenv("LOG_FILE", "fibonacci_pivot_bot.log"),
        "MAX_EXEC_TIME": int(os.getenv("MAX_EXEC_TIME", "25")),
        "USE_RMA200": os.getenv("USE_RMA200", "True").lower() == "true",
    }

    # Load from JSON if exists
    config_file = os.getenv("CONFIG_FILE", "config.json")
    if Path(config_file).exists():
        try:
            with open(config_file, "r") as f:
                user_cfg = json.load(f)
                config_dict.update(user_cfg)
            print(f"✅ Loaded configuration from {config_file}")
        except Exception as e:
            print(f"⚠️ Warning: unable to parse {config_file}: {e}")
            sys.exit(EXIT_CONFIG_ERROR)
    else:
        print(f"⚠️ Warning: config file {config_file} not found, using environment defaults.")

    # Validate with Pydantic
    try:
        return Config(**config_dict)
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        sys.exit(EXIT_CONFIG_ERROR)

cfg = load_config()

# -------------------------
# METRICS
# -------------------------
METRICS = {
    "pairs_checked": 0,
    "alerts_sent": 0,
    "api_errors": 0,
    "execution_time": 0.0,
    "start_time": 0.0
}

def reset_metrics():
    METRICS.update({
        "pairs_checked": 0,
        "alerts_sent": 0,
        "api_errors": 0,
        "execution_time": 0.0,
        "start_time": time.time()
    })

# -------------------------
# LOGGER
# -------------------------
logger = logging.getLogger("fibonacci_pivot_bot")
logger.setLevel(logging.DEBUG if cfg.DEBUG_MODE else logging.INFO)

# NEW: JSON formatter for Cron-Jobs.org parsing
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "time": self.formatTime(record),
            "level": record.levelname,
            "msg": record.getMessage()
        }
        return json.dumps(log_obj)

# Use text formatter for debug, JSON for production
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s") if cfg.DEBUG_MODE else JSONFormatter()

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Rotating file handler
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
            
            # Try non-blocking exclusive lock
            fcntl.flock(self.fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.fd.write(str(os.getpid()))
            self.fd.flush()
            return True
        except (IOError, OSError):
            # Check for stale lock
            try:
                if self.lock_path.exists():
                    with open(self.lock_path) as f:
                        old_pid = int(f.read().strip())
                    # If process doesn't exist, lock is stale
                    if not os.path.exists(f"/proc/{old_pid}"):
                        logger.warning(f"Removing stale lock from PID {old_pid}")
                        self.lock_path.unlink(missing_ok=True)
                        return self.acquire()  # Retry
            except Exception as e:
                logger.debug(f"Stale lock check failed: {e}")
            
            if self.fd:
                self.fd.close()
            return False

    def release(self):
        """Release the lock"""
        try:
            if self.fd:
                fcntl.flock(self.fd.fileno(), fcntl.LOCK_UN)
                self.fd.close()
            self.lock_path.unlink(missing_ok=True)
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
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

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
            # Success: reset failures
            if self.state == "HALF_OPEN":
                logger.info("Circuit breaker: CLOSED (recovery successful)")
            self.failures = 0
            self.state = "CLOSED"
            return result
        except Exception as e:
            self.failures += 1
            self.last_failure = time.time()
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
        
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        # NEW: Enable WAL for concurrent access
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._ensure_tables()
        
        # Secure file permissions
        os.chmod(self.db_path, 0o600)

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
            cur.execute("INSERT INTO states(pair, state, ts) VALUES (?, ?, ?) "
                        "ON CONFLICT(pair) DO UPDATE SET state=excluded.state, ts=excluded.ts",
                        (pair, state, ts))
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
        self._conn.close()

# -------------------------
# PRUNE (smart daily)
# -------------------------
def prune_old_state_records(db_path: str, expiry_days: int = 30, logger_local: logging.Logger = None):
    """Delete old states with stale lock protection"""
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
        
        # NEW: Use exclusive transaction to prevent corruption
        cur.execute("BEGIN EXCLUSIVE")
        
        cur.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)")
        cur.execute("SELECT value FROM metadata WHERE key='last_prune'")
        row = cur.fetchone()
        
        from datetime import timezone
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
                conn.commit()
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
    """Fetch with retry logic and circuit breaker support"""
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
            except (asyncio.TimeoutError, ClientConnectorError, ClientResponseError) as e:
                logger.debug(f"Fetch attempt {attempt} error: {e} for {url}")
                METRICS["api_errors"] += 1
                if attempt == retries:
                    logger.warning(f"Failed to fetch {url} after {retries} attempts.")
                    return None
                await asyncio.sleep(backoff * (attempt ** 1.2) + random.uniform(0, 0.2))
            except Exception as e:
                logger.exception(f"Unexpected fetch error for {url}: {e}")
                METRICS["api_errors"] += 1
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
        self.cache = {}  # per-run cache
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
            return self.cache[key][1]
        
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
    """Load products from cache if fresh"""
    cache_path = Path("products_cache.json")
    if cache_path.exists():
        try:
            age = time.time() - cache_path.stat().st_mtime
            if age < cfg.PRODUCTS_CACHE_TTL:
                with open(cache_path) as f:
                    logger.info(f"Using cached products ({age:.0f}s old)")
                    return json.load(f)
        except Exception as e:
            logger.debug(f"Cache load failed: {e}")
    return None

def save_products_cache(products_map: Dict[str, dict]):
    """Save products to cache"""
    try:
        with open("products_cache.json", "w") as f:
            json.dump(products_map, f)
    except Exception as e:
        logger.warning(f"Failed to save products cache: {e}")

def build_products_map(api_products: dict) -> Dict[str, dict]:
    products_map = {}
    if not api_products:
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
    if not res.get("success"):
        return None
    resr = res.get("result", {})
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
    if df.empty or df['close'].astype(float).iloc[-1] <= 0:
        return None
    return df

# -------------------------
# INDICATORS (unchanged logic)
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

def get_previous_day_ohlc_from_api(session: aiohttp.ClientSession, symbol: str, days_back_limit: int = 15) -> Optional[dict]:
    # This function is unused in original code, keeping for compatibility
    return None

async def awaitable_get_previous_day_ohlc(session: aiohttp.ClientSession, symbol: str, days_back_limit: int = 15):
    """Get previous day's OHLC with error handling"""
    try:
        res = await fetcher.fetch_candles(session, symbol, "D", days_back_limit + 5)
        df = parse_candle_response(res)
        if df is None or len(df) < 2:
            return None
        df['date'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.date
        today = datetime.now(timezone.utc).date()
        yesterday = today - timedelta(days=1)
        prev_day_row = df[df['date'] == yesterday]
        if not prev_day_row.empty:
            prev = prev_day_row.iloc[-1]
        else:
            prev = df.iloc[-2]
        return {'high': float(prev['high']), 'low': float(prev['low']), 'close': float(prev['close'])}
    except Exception as e:
        logger.error(f"Failed to fetch daily OHLC for {symbol}: {e}")
        return None

def calculate_fibonacci_pivots(h: float, l: float, c: float) -> dict:
    pivot = (h + l + c) / 3.0
    diff = h - l
    r3 = pivot + (diff * 1.000)
    r2 = pivot + (diff * 0.618)
    r1 = pivot + (diff * 0.382)
    s1 = pivot - (diff * 0.382)
    s2 = pivot - (diff * 0.618)
    s3 = pivot - (diff * 1.000)
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

        # NEW: Conditional 5m fetch
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
        
        df_15m = parse_candle_response(res15)
        df_5m = parse_candle_response(res5) if res5 else None

        if df_15m is None or len(df_15m) < min_required_15m:
            logger.warning(f"{pair_name}: insufficient 15m data ({0 if df_15m is None else len(df_15m)})")
            return None

        nowts = time.time()
        res15s = 15 * 60
        current_15m_start = nowts - (nowts % res15s)
        last_15_incomplete = df_15m['timestamp'].iloc[-1] >= current_15m_start
        last_i_15m = -2 if last_15_incomplete else -1
        
        if len(df_15m) < abs(last_i_15m) + 1:
            logger.warning(f"{pair_name}: not enough 15m candles after index adjust")
            return None

        last_i_5m = -1
        rma_200_available = False
        if df_5m is not None and len(df_5m) >= 50:
            res5s = 5 * 60
            current_5m_start = nowts - (nowts % res5s)
            last_5_incomplete = df_5m['timestamp'].iloc[-1] >= current_5m_start
            last_i_5m = -2 if last_5_incomplete else -1
            if len(df_5m) < abs(last_i_5m) + 1:
                df_5m = None

        # NEW: Error handling for daily candles
        resd = await fetcher.fetch_candles(session, prod['symbol'], "D", cfg.PIVOT_LOOKBACK_PERIOD + 5)
        df_daily = parse_candle_response(resd)
        if df_daily is None or len(df_daily) < 2:
            logger.warning(f"{pair_name}: failed to fetch daily candles")
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

        idx = last_i_15m
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
        if total_range <= 0:
            upper_wick_ok = False
            lower_wick_ok = False
        else:
            upper_wick = high_c - max(open_c, close_c)
            lower_wick = min(open_c, close_c) - low_c
            upper_wick_ok = (upper_wick / total_range) < 0.20
            lower_wick_ok = (lower_wick / total_range) < 0.20

        vwap_curr = None
        try:
            vwap_curr = vwap_15m.iloc[last_i_15m]
            if np.isnan(vwap_curr):
                vwap_curr = None
        except Exception:
            vwap_curr = None

        is_green = close_c > open_c
        is_red = close_c < open_c

        long_pivot_lines = {'P': pivots['P'], 'R1': pivots['R1'], 'R2': pivots['R2'], 'S1': pivots['S1'], 'S2': pivots['S2']}
        long_crossover_line = None
        long_crossover_name = None
        if is_green:
            for name, line in long_pivot_lines.items():
                if open_c <= line and close_c > line:
                    long_crossover_line = line
                    long_crossover_name = name
                    break

        short_pivot_lines = {'P': pivots['P'], 'S1': pivots['S1'], 'S2': pivots['S2'], 'R1': pivots['R1'], 'R2': pivots['R2']}
        short_crossover_line = None
        short_crossover_name = None
        if is_red:
            for name, line in short_pivot_lines.items():
                if open_c >= line and close_c < line:
                    short_crossover_line = line
                    short_crossover_name = name
                    break

        if rma_200_available:
            rma_long_ok = (rma50_curr < close_c) and (rma200_5m_curr < close_c)
            rma_short_ok = (rma50_curr > close_c) and (rma200_5m_curr > close_c)
        else:
            rma_long_ok = (rma50_curr < close_c)
            rma_short_ok = (rma50_curr > close_c)

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

        # Replace the message construction section with this:

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
                message = (f"{up_sig} {sanitize_for_telegram(pair_name)} - FIB LONG\n"
                           f"Closed Above {sanitize_for_telegram(long_crossover_name)} (${line_str})\n"
                           f"PPO 15m: {ppo_str}\n"
                           f"Price: ${price_str}\n"
                           f"{human_ts()}")
        elif fib_short:
            current_signal = f"fib_short_{short_crossover_name}"
            if not should_suppress_duplicate(last_state_pair, current_signal, suppress_secs):
                line_str = f"{short_crossover_line:,.2f}"
                ppo_str = f"{ppo_curr:.2f}"
                price_str = f"{close_c:,.2f}"
                message = (f"{down_sig} {sanitize_for_telegram(pair_name)} - FIB SHORT\n"
                           f"Closed Below {sanitize_for_telegram(short_crossover_name)} (${line_str})\n"
                           f"PPO 15m: {ppo_str}\n"
                           f"Price: ${price_str}\n"
                           f"{human_ts()}")

        if current_signal is None:
            prev_state_val = last_state_pair.get("state") if last_state_pair else None
            if prev_state_val and prev_state_val != "NO_SIGNAL":
                return pair_name, {"state": "NO_SIGNAL", "ts": now_ts()}
            return None

        if message:
            success = await send_telegram_async(cfg.TELEGRAM_BOT_TOKEN, cfg.TELEGRAM_CHAT_ID, message, session)
            if success:
                METRICS["alerts_sent"] += 1
                logger.info(f"✅ Alert sent for {pair_name}: {current_signal}")
            else:
                logger.error(f"❌ Failed to send alert for {pair_name}")

        return pair_name, {"state": current_signal, "ts": now_ts()}

    except Exception as e:
        METRICS["api_errors"] += 1
        logger.exception(f"Error evaluating pair {pair_name}: {e}")
        if cfg.DEBUG_MODE:
            logger.debug(traceback.format_exc())
        return None

# -------------------------
# ASYNC TELEGRAM SENDER
# -------------------------
async def send_telegram_async(
    token: str, 
    chat_id: str, 
    text: str, 
    session: Optional[aiohttp.ClientSession] = None
) -> bool:
    """Send Telegram message with error handling"""
    if not token or not chat_id:
        logger.warning("Telegram not configured; skipping.")
        return False
    
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
    own_session = False
    
    if session is None:
        session = aiohttp.ClientSession()
        own_session = True
    
    try:
        async with session.post(url, data=data, timeout=10) as resp:
            try:
                js = await resp.json(content_type=None)
            except Exception:
                js = {"ok": False, "status": resp.status, "text": await resp.text()}
            
            ok = js.get("ok", False)
            if ok:
                await asyncio.sleep(0.2)  # Rate limiting
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

# -------------------------
# DEAD MAN'S SWITCH
# -------------------------
async def check_dead_mans_switch(state_db: StateDB):
    """Alert if bot hasn't succeeded recently"""
    try:
        last_success = state_db.get_metadata("last_success_run")
        if last_success:
            hours_since = (now_ts() - int(last_success)) / 3600
            if hours_since > 2:  # Alert if no success for 2+ hours
                await send_telegram_async(
                    cfg.TELEGRAM_BOT_TOKEN, 
                    cfg.TELEGRAM_CHAT_ID,
                    f"🚨 DEAD MAN'S SWITCH: Bot hasn't succeeded in {hours_since:.1f} hours!\n"
                    f"Last success: {datetime.fromtimestamp(int(last_success)).strftime('%Y-%m-%d %H:%M UTC')}\n"
                    f"Check logs immediately."
                )
    except Exception as e:
        logger.error(f"Dead man's switch check failed: {e}")

# -------------------------
# MAIN RUNNER
# -------------------------
async def run_once(send_test: bool = True):
    """Main execution with timeout protection"""
    start_time = time.time()
    reset_metrics()
    
    # NEW: Check execution time periodically
    async def check_timeout():
        while True:
            elapsed = time.time() - start_time
            if elapsed > cfg.MAX_EXEC_TIME:
                logger.critical(f"Execution timeout: {elapsed:.1f}s > {cfg.MAX_EXEC_TIME}s")
                sys.exit(EXIT_TIMEOUT)
            await asyncio.sleep(1)  # Check every second
    
    timeout_task = asyncio.create_task(check_timeout())
    
    try:
        logger.info("🚀 Starting fibonacci_pivot_bot_production run")

        # Check dead man's switch first
        state_db = StateDB(cfg.STATE_DB_PATH)
        await check_dead_mans_switch(state_db)

        # Prune old records
        try:
            prune_old_state_records(cfg.STATE_DB_PATH, cfg.STATE_EXPIRY_DAYS, logger)
        except Exception as e:
            logger.warning(f"Prune error: {e}")

        # Load last alerts
        last_alerts = state_db.load_all()

        # Reset state if requested
        if cfg.RESET_STATE:
            logger.warning("🔄 RESET_STATE requested: clearing states table")
            for p in cfg.PAIRS:
                state_db.set(p, None)

        # Send test message
        if send_test and cfg.SEND_TEST_MESSAGE:
            test_msg = (f"🔔 Fibonacci Pivot Bot started\n"
                       f"Time: {human_ts()}\n"
                       f"Debug: {'ON' if cfg.DEBUG_MODE else 'OFF'}\n"
                       f"Pairs: {len(cfg.PAIRS)}")
            await send_telegram_async(cfg.TELEGRAM_BOT_TOKEN, cfg.TELEGRAM_CHAT_ID, test_msg)

        # Load products (with cache)
        products_map = load_products_cache()
        fetcher_local = DataFetcher(
            cfg.DELTA_API_BASE, 
            max_parallel=cfg.MAX_CONCURRENCY, 
            timeout=cfg.HTTP_TIMEOUT,
            circuit_breaker=CircuitBreaker()  # NEW: Per-run circuit breaker
        )
        
        async with aiohttp.ClientSession() as session:
            if products_map is None:
                prod_resp = await fetcher_local.fetch_products(session)
                if not prod_resp:
                    logger.error("❌ Failed to fetch products from API")
                    state_db.close()
                    sys.exit(EXIT_API_FAILURE)
                products_map = build_products_map(prod_resp)
                save_products_cache(products_map)
            
            found = len(products_map)
            logger.info(f"✅ Found {found} tradable pairs mapped to config.")
            if found == 0:
                logger.error("❌ No products mapped; exiting")
                state_db.close()
                sys.exit(EXIT_API_FAILURE)

            # Execute pair evaluations with concurrency control
            sem = asyncio.Semaphore(cfg.MAX_CONCURRENCY)
            tasks = []
            
            for pair in cfg.PAIRS:
                if pair not in products_map:
                    logger.debug(f"Pair {pair} not in products map, skipping")
                    continue
                
                last_state = last_alerts.get(pair)
                
                async def run_one(pair_name=pair, last_s=last_state):
                    async with sem:
                        await asyncio.sleep(random.uniform(cfg.JITTER_MIN, cfg.JITTER_MAX))
                        return await evaluate_pair_async(session, fetcher_local, products_map, pair_name, last_s)
                
                tasks.append(asyncio.create_task(run_one()))

            results = await asyncio.gather(*tasks, return_exceptions=True)
            updates = 0
            
            for r in results:
                if isinstance(r, Exception):
                    logger.exception(f"Task exception: {r}")
                    continue
                if not r:
                    continue
                
                pair_name, new_state = r
                if not isinstance(new_state, dict):
                    continue
                
                prev = state_db.get(pair_name)
                if prev != new_state:
                    state_db.set(pair_name, new_state.get("state"), new_state.get("ts"))
                    updates += 1

            # Log metrics
            METRICS["execution_time"] = time.time() - start_time
            logger.info(f"📊 METRICS: {json.dumps(METRICS)}")
            
            # Update success timestamp
            state_db.set_metadata("last_success_run", str(now_ts()))
            
            logger.info(f"✅ Run complete. {updates} state updates applied. "
                       f"Checked {METRICS['pairs_checked']} pairs, sent {METRICS['alerts_sent']} alerts.")
            
            state_db.close()

    except asyncio.CancelledError:
        logger.warning("Run was cancelled")
        sys.exit(EXIT_TIMEOUT)
    except Exception as e:
        logger.exception(f"Unhandled error in run_once: {e}")
        sys.exit(1)
    finally:
        timeout_task.cancel()
        try:
            await timeout_task
        except asyncio.CancelledError:
            pass

# -------------------------
# CLI + SIGNAL HANDLING
# -------------------------
stop_requested = False

def request_stop(signum, frame):
    global stop_requested
    logger.info(f"⚠️ Signal {signum} received. Stopping after current run.")
    stop_requested = True

# Register signal handlers
signal.signal(signal.SIGINT, request_stop)
signal.signal(signal.SIGTERM, request_stop)

def main():
    """Entry point with process locking"""
    # NEW: Acquire process lock
    lock = ProcessLock("/tmp/fib_bot.lock", timeout=cfg.MAX_EXEC_TIME * 2)
    if not lock.acquire():
        logger.error("❌ Another instance is running or stale lock exists. Exiting.")
        sys.exit(EXIT_LOCK_CONFLICT)
    
    try:
        parser = argparse.ArgumentParser(description="Fibonacci Pivot Bot (Production)")
        group = parser.add_mutually_exclusive_group()
        group.add_argument("--once", action="store_true", help="Run once and exit")
        group.add_argument("--loop", type=int, metavar="SECONDS", help="Run in loop every N seconds")
        args = parser.parse_args()

        if args.once:
            asyncio.run(run_once())
        elif args.loop:
            interval = args.loop
            logger.info(f"🔄 Loop mode: interval={interval}s")
            while not stop_requested:
                loop_start = time.time()
                try:
                    asyncio.run(run_once(send_test=False))
                except Exception:
                    logger.exception("Unhandled in run_once")
                
                elapsed = time.time() - loop_start
                to_sleep = max(0, interval - elapsed)
                if to_sleep > 0:
                    time.sleep(to_sleep)
        else:
            asyncio.run(run_once())
            
        sys.exit(EXIT_SUCCESS)
        
    finally:
        lock.release()

if __name__ == "__main__":
    main()
