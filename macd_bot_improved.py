#!/usr/bin/env python3
# macd_bot_improved.py
# Production-hardened MACD/PPO bot with Pydantic config, structured logging, and enhanced reliability

from __future__ import annotations

import argparse
import asyncio
import fcntl
import gc
import json
import logging
import logging.config
import os
import signal
import sqlite3
import sys
import time
import warnings
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Literal

import aiohttp
import numpy as np
import pandas as pd
import psutil
import pytz
from aiohttp import ClientConnectorError, ClientResponseError, TCPConnector
from pydantic import (
    BaseModel,
    Field,
    HttpUrl,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    field_validator,
    ValidationError
)
from pydantic_settings import BaseSettings, SettingsConfigDict

# Suppress pandas warnings for cleaner logs
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

# ============================================================================
# Pydantic Configuration Models
# ============================================================================
class SpecialPairConfig(BaseModel):
    limit_15m: int = 210
    min_required: int = 180
    limit_5m: int = 300
    min_required_5m: int = 200

class TelegramConfig(BaseModel):
    bot_token: str = Field(..., min_length=1)
    chat_id: str = Field(..., min_length=1)

class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: Literal["json", "text"] = "json"
    file: str = "macd_bot.log"
    max_bytes: int = 5_000_000
    backup_count: int = 5

class DeadMansSwitchConfig(BaseModel):
    enabled: bool = True
    hours: PositiveInt = 2
    cooldown_seconds: NonNegativeInt = Field(
        default=14_400,  # 4 hours
        description="Cooldown between DMS alerts"
    )

class CircuitBreakerConfig(BaseModel):
    failure_threshold: PositiveInt = 5
    timeout_seconds: PositiveInt = 300

class Config(BaseSettings):
    """Main configuration with validation"""
    
    telegram: TelegramConfig = Field(...)
    debug_mode: bool = False
    send_test_message: bool = False
    delta_api_base: HttpUrl = "https://api.india.delta.exchange"
    
    pairs: List[str] = Field(default_factory=lambda: [
        "BTCUSD", "ETHUSD", "SOLUSD", "AVAXUSD", "BCHUSD",
        "XRPUSD", "BNBUSD", "LTCUSD", "DOTUSD", "ADAUSD",
        "SUIUSD", "AAVEUSD"
    ])
    
    special_pairs: Dict[str, SpecialPairConfig] = Field(default_factory=dict)
    
    # Indicator parameters
    ppo_fast: PositiveInt = 7
    ppo_slow: PositiveInt = 16
    ppo_signal: PositiveInt = 5
    ppo_use_sma: bool = False
    rma_50_period: PositiveInt = 50
    rma_200_period: PositiveInt = 200
    
    # Cloud settings
    cirrus_cloud_enabled: bool = True
    x1: PositiveInt = 22
    x2: PositiveInt = 9
    x3: PositiveInt = 15
    x4: PositiveInt = 5
    
    # RSI settings
    srsi_rsi_len: PositiveInt = 21
    srsi_kalman_len: PositiveInt = 5
    
    # State management
    state_db_path: str = "macd_state.sqlite"
    state_expiry_days: NonNegativeInt = 30
    
    # Fetching
    max_parallel_fetch: int = Field(8, ge=1, le=32)  # Increased max to 32 for I/O-bound ops
    http_timeout: PositiveInt = 15
    candle_fetch_retries: PositiveInt = 3
    candle_fetch_backoff: PositiveFloat = 1.5
    
    # Jitter
    jitter_min: PositiveFloat = 0.1
    jitter_max: PositiveFloat = 0.8
    
    # Runtime
    run_timeout_seconds: PositiveInt = 600
    batch_size: PositiveInt = 4
    
    # Rate limiting
    telegram_retries: PositiveInt = 3
    telegram_backoff_base: PositiveFloat = 2.0
    http_rate_limit_per_minute: PositiveInt = 60
    
    # Resource limits
    memory_limit_bytes: PositiveInt = 400_000_000
    tcp_conn_limit: PositiveInt = 8
    
    # Dead man's switch
    dead_mans_switch: DeadMansSwitchConfig = Field(default_factory=DeadMansSwitchConfig)
    
    # Circuit breaker
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
    
    # PID file
    pid_lock_path: str = "/tmp/macd_bot.pid"
    
    # Logging
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    # Bot identity
    bot_name: str = "MACD Alert Bot"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore"
    )
    
    @field_validator("max_parallel_fetch")
    @classmethod
    def validate_max_parallel(cls, v: int) -> int:
        """
        Validate max_parallel_fetch is reasonable for I/O-bound operations.
        For network operations, we can safely exceed CPU cores.
        """
        if v < 1:
            raise ValueError("max_parallel_fetch must be at least 1")
        if v > 32:
            warnings.warn(
                f"max_parallel_fetch={v} is very high. "
                f"Consider reducing if you experience network issues.",
                stacklevel=2
            )
        return v


# ============================================================================
# Standalone Config Loader
# ============================================================================
def load_config(path: str = "config_macd.json") -> Config:
    """Load and validate configuration from JSON file"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Convert nested dicts to proper types
        if "telegram" in data and isinstance(data["telegram"], dict):
            data["telegram"] = TelegramConfig(**data["telegram"])
        
        if "special_pairs" in data and isinstance(data["special_pairs"], dict):
            data["special_pairs"] = {
                k: SpecialPairConfig(**v) if isinstance(v, dict) else v
                for k, v in data["special_pairs"].items()
            }
        
        if "dead_mans_switch" in data and isinstance(data["dead_mans_switch"], dict):
            data["dead_mans_switch"] = DeadMansSwitchConfig(**data["dead_mans_switch"])
        
        if "circuit_breaker" in data and isinstance(data["circuit_breaker"], dict):
            data["circuit_breaker"] = CircuitBreakerConfig(**data["circuit_breaker"])
        
        if "logging" in data and isinstance(data["logging"], dict):
            data["logging"] = LoggingConfig(**data["logging"])
        
        return Config(**data)
    
    except FileNotFoundError:
        raise ValueError(f"Config file not found: {path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file: {e}")
    except ValidationError as e:
        raise ValueError(f"Configuration validation failed: {e}")


# ============================================================================
# Structured Logging Setup
# ============================================================================
class JSONLogFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
            "pid": record.process,
            "thread": record.threadName,
        }
        
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        
        # Add extra context if available
        if hasattr(record, "context"):
            log_obj["context"] = record.context
        
        return json.dumps(log_obj, ensure_ascii=False)


def setup_logging(config: LoggingConfig) -> logging.Logger:
    """Configure structured logging"""
    logger = logging.getLogger("macd_bot")
    logger.setLevel(logging.DEBUG if config.level.upper() == "DEBUG" else getattr(logging, config.level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logger.level)
    
    if config.format == "json":
        console_handler.setFormatter(JSONLogFormatter())
    else:
        console_handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
    
    logger.addHandler(console_handler)
    
    # File handler
    try:
        log_dir = os.path.dirname(config.file) or "."
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            config.file,
            maxBytes=config.max_bytes,
            backupCount=config.backup_count,
            encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(JSONLogFormatter())
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"Could not set up file logging: {e}")
    
    return logger


# ============================================================================
# PID File Lock
# ============================================================================
class PidFileLock:
    def __init__(self, path: str):
        self.path = Path(path)
        self.fd = None

    def acquire(self) -> bool:
        """Acquire PID lock with atomic operations"""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.fd = open(self.path, "w+")
            fcntl.flock(self.fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            
            # Write PID
            self.fd.seek(0)
            self.fd.truncate()
            self.fd.write(str(os.getpid()))
            self.fd.flush()
            
            # Register cleanup
            atexit.register(self.release)
            return True
        except (IOError, OSError):
            if self.fd:
                try:
                    self.fd.close()
                except Exception:
                    pass
            return False

    def release(self):
        """Release PID lock"""
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
            
            # Only remove if we own it
            if self.path.exists():
                with open(self.path, "r") as f:
                    if f.read().strip() == str(os.getpid()):
                        self.path.unlink(missing_ok=True)
        except Exception:
            pass

    def __enter__(self):
        if not self.acquire():
            raise RuntimeError(f"Could not acquire PID lock: {self.path}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


# ============================================================================
# SQLite State DB
# ============================================================================
class StateDB:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            self._conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                isolation_level=None,
                timeout=10.0
            )
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
            self._conn.execute("PRAGMA temp_store=MEMORY;")
            self._conn.execute("PRAGMA cache_size=-100000;")  # ~100MB cache
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to initialize database: {e}")
        
        self._ensure_tables()

    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _ensure_tables(self):
        """Create tables if they don't exist"""
        cur = self._conn.cursor()
        
        # States table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS states (
                pair TEXT PRIMARY KEY,
                state TEXT NOT NULL,
                ts INTEGER NOT NULL,
                updated_at INTEGER NOT NULL DEFAULT (unixepoch())
            ) STRICT
        """)
        
        # Metadata table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at INTEGER NOT NULL DEFAULT (unixepoch())
            ) STRICT
        """)
        
        # Dead man's switch tracking
        cur.execute("""
            CREATE TABLE IF NOT EXISTS dead_mans_switch (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                last_success_run INTEGER NOT NULL,
                last_alert_sent INTEGER,
                alert_state TEXT NOT NULL DEFAULT 'healthy'
            )
        """)
        
        # Initialize DMS if not exists
        cur.execute("""
            INSERT OR IGNORE INTO dead_mans_switch (id, last_success_run)
            VALUES (1, ?)
        """, (int(time.time()),))
        
        self._conn.commit()

    def load_all(self) -> Dict[str, Dict[str, Any]]:
        """Load all states"""
        cur = self._conn.cursor()
        cur.execute("SELECT pair, state, ts FROM states")
        return {row[0]: {"state": row[1], "ts": int(row[2])} for row in cur.fetchall()}

    def get(self, pair: str) -> Optional[Dict[str, Any]]:
        """Get state for a pair"""
        cur = self._conn.cursor()
        cur.execute("SELECT state, ts FROM states WHERE pair = ?", (pair,))
        row = cur.fetchone()
        return {"state": row[0], "ts": int(row[1])} if row else None

    def set(self, pair: str, state: Optional[str], ts: Optional[int] = None):
        """Set state for a pair"""
        ts = int(ts or time.time())
        cur = self._conn.cursor()
        
        if state is None:
            cur.execute("DELETE FROM states WHERE pair = ?", (pair,))
        else:
            cur.execute(
                "INSERT INTO states(pair, state, ts, updated_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(pair) DO UPDATE SET state=excluded.state, ts=excluded.ts, updated_at=excluded.updated_at",
                (pair, state, ts, ts)
            )
        
        self._conn.commit()

    def get_metadata(self, key: str) -> Optional[str]:
        """Get metadata value"""
        cur = self._conn.cursor()
        cur.execute("SELECT value FROM metadata WHERE key = ?", (key,))
        row = cur.fetchone()
        return row[0] if row else None

    def set_metadata(self, key: str, value: str):
        """Set metadata value"""
        cur = self._conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO metadata (key, value, updated_at) VALUES (?, ?, ?)",
            (key, value, int(time.time()))
        )
        self._conn.commit()

    def update_last_success_run(self):
        """Update last successful run timestamp"""
        cur = self._conn.cursor()
        cur.execute(
            "UPDATE dead_mans_switch SET last_success_run = ? WHERE id = 1",
            (int(time.time()),)
        )
        self._conn.commit()

    def get_dead_mans_switch_status(self) -> Dict[str, Any]:
        """Get DMS status"""
        cur = self._conn.cursor()
        cur.execute("""
            SELECT last_success_run, last_alert_sent, alert_state
            FROM dead_mans_switch WHERE id = 1
        """)
        row = cur.fetchone()
        if not row:
            return {}
        
        return {
            "last_success_run": int(row[0]),
            "last_alert_sent": int(row[1]) if row[1] else None,
            "alert_state": row[2]
        }

    def record_dms_alert(self, state: str):
        """Record that a DMS alert was sent"""
        cur = self._conn.cursor()
        cur.execute(
            "UPDATE dead_mans_switch SET last_alert_sent = ?, alert_state = ? WHERE id = 1",
            (int(time.time()), state)
        )
        self._conn.commit()

    def prune_old_records(self, expiry_days: int) -> int:
        """Prune old state records"""
        if expiry_days <= 0:
            logger.debug("Pruning disabled (expiry_days <= 0)")
            return 0
        
        cur = self._conn.cursor()
        
        # Check if already pruned today
        cur.execute("SELECT value FROM metadata WHERE key='last_prune_date'")
        row = cur.fetchone()
        today = datetime.now(timezone.utc).date().isoformat()
        
        if row and row[0] == today:
            logger.debug("Daily prune already completed")
            return 0
        
        # Delete old records
        cutoff = int(time.time()) - (expiry_days * 86400)
        cur.execute("DELETE FROM states WHERE ts < ?", (cutoff,))
        deleted = cur.rowcount
        
        # Update prune date
        cur.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            ("last_prune_date", today)
        )
        
        # Vacuum if we deleted anything
        if deleted > 0:
            try:
                cur.execute("VACUUM;")
            except Exception as e:
                logger.warning(f"VACUUM failed: {e}", exc_info=True)
        
        self._conn.commit()
        return deleted

    def close(self):
        """Close database connection"""
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


# ============================================================================
# Circuit Breaker
# ============================================================================
class CircuitBreaker:
    def __init__(self, failure_threshold: int, timeout: int):
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.threshold = failure_threshold
        self.timeout = timeout
        self.lock = asyncio.Lock()

    async def is_open(self) -> bool:
        """Check if circuit is open"""
        async with self.lock:
            if self.failure_count < self.threshold:
                return False
            
            if self.last_failure_time and (time.time() - self.last_failure_time > self.timeout):
                self.failure_count = 0
                self.last_failure_time = None
                logger.info("Circuit breaker: reset after timeout")
                return False
            
            return True

    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if await self.is_open():
            raise Exception(f"Circuit breaker is OPEN ({self.failure_count}/{self.threshold} failures)")
        
        try:
            result = await func(*args, **kwargs)
            await self.record_success()
            return result
        except Exception as e:
            await self.record_failure()
            raise

    async def record_failure(self):
        """Record a failure"""
        async with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            logger.warning(
                "Circuit breaker: failure recorded",
                extra={"context": {"count": self.failure_count, "threshold": self.threshold}}
            )

    async def record_success(self):
        """Record a success"""
        async with self.lock:
            if self.failure_count > 0:
                self.failure_count = 0
                self.last_failure_time = None
                logger.info("Circuit breaker: reset after success")


# ============================================================================
# Rate Limiter
# ============================================================================
class RateLimiter:
    """Token bucket rate limiter"""
    
    def __init__(self, max_per_minute: int):
        self.max_per_minute = max_per_minute
        self.tokens = max_per_minute
        self.last_refill = time.time()
        self.lock = asyncio.Lock()

    async def acquire(self):
        """Acquire a token, wait if necessary"""
        async with self.lock:
            now = time.time()
            elapsed = now - self.last_refill
            
            # Refill tokens
            self.tokens = min(
                self.max_per_minute,
                self.tokens + (elapsed * self.max_per_minute / 60)
            )
            self.last_refill = now
            
            if self.tokens < 1:
                wait_time = (1 - self.tokens) * 60 / self.max_per_minute
                logger.debug(f"Rate limiter: waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1


# ============================================================================
# HTTP Client
# ============================================================================
class HTTPClient:
    """Shared HTTP client with circuit breaker and rate limiting"""
    
    def __init__(self, config: Config):
        self.config = config
        self.connector = TCPConnector(
            limit=config.tcp_conn_limit,
            ssl=True,  # SECURITY: Enable SSL verification
            limit_per_host=config.tcp_conn_limit // 2
        )
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=config.circuit_breaker.failure_threshold,
            timeout=config.circuit_breaker.timeout_seconds
        )
        self.rate_limiter = RateLimiter(config.http_rate_limit_per_minute)

    @asynccontextmanager
    async def session(self) -> aiohttp.ClientSession:
        """Get a shared client session"""
        timeout = aiohttp.ClientTimeout(total=self.config.http_timeout)
        async with aiohttp.ClientSession(
            connector=self.connector,
            timeout=timeout,
            raise_for_status=False
        ) as session:
            yield session

    async def fetch_json(
        self,
        session: aiohttp.ClientSession,
        url: str,
        params: dict = None,
        retries: int = None,
        backoff: float = None
    ) -> Optional[dict]:
        """Fetch JSON with retries and circuit breaker"""
        if retries is None:
            retries = self.config.candle_fetch_retries
        if backoff is None:
            backoff = self.config.candle_fetch_backoff

        async def _fetch():
            await self.rate_limiter.acquire()
            return await self._fetch_once(session, url, params)

        return await self.circuit_breaker.call(_fetch)

    async def _fetch_once(
        self,
        session: aiohttp.ClientSession,
        url: str,
        params: dict = None
    ) -> Optional[dict]:
        """Single fetch attempt"""
        try:
            async with session.get(url, params=params) as resp:
                text = await resp.text()
                
                if resp.status >= 400:
                    logger.debug(
                        "HTTP error",
                        extra={"context": {"status": resp.status, "url": url, "response": text[:200]}}
                    )
                    raise ClientResponseError(resp.request_info, resp.history, status=resp.status)
                
                try:
                    return await resp.json()
                except Exception:
                    logger.debug(f"Failed to parse JSON from {url}")
                    return None
        except asyncio.TimeoutError:
            logger.debug(f"Timeout fetching {url}")
            raise
        except ClientConnectorError as e:
            logger.debug(f"Connection error for {url}: {e}")
            raise


# ============================================================================
# Data Fetcher
# ============================================================================
class DataFetcher:
    """Market data fetcher with caching"""
    
    def __init__(self, http_client: HTTPClient, config: Config):
        self.http_client = http_client
        self.config = config
        self._cache: Dict[str, Tuple[float, Any]] = {}
        self._cache_max_size = 50

    def _clean_cache(self):
        """Remove old cache entries"""
        if len(self._cache) > self._cache_max_size:
            # Remove oldest 20%
            to_remove = sorted(
                self._cache.keys(),
                key=lambda k: self._cache[k][0]
            )[: self._cache_max_size // 5]
            
            for key in to_remove:
                del self._cache[key]

    async def fetch_products(self, session: aiohttp.ClientSession) -> Optional[dict]:
        """Fetch available products"""
        url = f"{self.config.delta_api_base}/v2/products"
        return await self.http_client.fetch_json(session, url)

    async def fetch_candles(
        self,
        session: aiohttp.ClientSession,
        symbol: str,
        resolution: str,
        limit: int
    ) -> Optional[dict]:
        """Fetch candlestick data"""
        key = f"candles:{symbol}:{resolution}:{limit}"
        
        # Check cache
        if key in self._cache:
            age, data = self._cache[key]
            if time.time() - age < 60:
                return data

        await asyncio.sleep(random.uniform(self.config.jitter_min, self.config.jitter_max))
        
        url = f"{self.config.delta_api_base}/v2/chart/history"
        params = {
            "resolution": resolution,
            "symbol": symbol,
            "from": int(time.time()) - (limit * (int(resolution) if resolution != 'D' else 1440) * 60),
            "to": int(time.time())
        }
        
        data = await self.http_client.fetch_json(session, url, params=params)
        
        if data:
            self._cache[key] = (time.time(), data)
            self._clean_cache()
        
        return data


# ============================================================================
# Candle Parser
# ============================================================================
def parse_candles_result(result: dict) -> Optional[pd.DataFrame]:
    """Parse candles API response into DataFrame"""
    if not result or not isinstance(result, dict):
        return None
    
    if not result.get("success", True) and "result" not in result:
        return None
    
    res = result.get("result", {}) or {}
    required_keys = ['t', 'o', 'h', 'l', 'c', 'v']
    
    if not all(k in res for k in required_keys):
        return None
    
    # Validate all are lists
    if not all(isinstance(res[k], list) for k in required_keys):
        return None
    
    try:
        # Find minimum length to avoid mismatches
        min_len = min(len(res[k]) for k in required_keys)
        
        df = pd.DataFrame({
            "timestamp": res['t'][:min_len],
            "open": res['o'][:min_len],
            "high": res['h'][:min_len],
            "low": res['l'][:min_len],
            "close": res['c'][:min_len],
            "volume": res['v'][:min_len]
        })
        
        # Sort and deduplicate by timestamp
        df = df.sort_values('timestamp').drop_duplicates(
            subset='timestamp',
            keep='last'
        ).reset_index(drop=True)
        
        # Memory optimization
        if not df.empty:
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], downcast='float', errors='coerce')
            
            df['timestamp'] = pd.to_numeric(
                df['timestamp'],
                downcast='integer',
                errors='coerce'
            )
        
        # Validate final data
        if df.empty or df['close'].iloc[-1] <= 0:
            return None
        
        return df
    
    except Exception as e:
        logger.error(f"Failed to parse candles: {e}", exc_info=True)
        return None


# ============================================================================
# Indicators
# ============================================================================
def calculate_ema(data: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average"""
    return data.ewm(span=period, adjust=False).mean()

def calculate_sma(data: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average"""
    return data.rolling(window=period, min_periods=max(2, period//3)).mean()

def calculate_rma(data: pd.Series, period: int) -> pd.Series:
    """Calculate Running Moving Average"""
    r = data.ewm(alpha=1/period, adjust=False).mean()
    return r.bfill().ffill()

def calculate_ppo(
    df: pd.DataFrame,
    fast: int,
    slow: int,
    signal: int,
    use_sma: bool = False
) -> Tuple[pd.Series, pd.Series]:
    """Calculate Percentage Price Oscillator"""
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
    
    if use_sma:
        ppo_signal = calculate_sma(ppo, signal)
    else:
        ppo_signal = calculate_ema(ppo, signal)
    
    ppo_signal = ppo_signal.replace([np.inf, -np.inf], np.nan).bfill().ffill()
    return ppo, ppo_signal

def smoothrng(x: pd.Series, t: int, m: int) -> pd.Series:
    """Smooth range calculation"""
    wper = t * 2 - 1
    avrng = calculate_ema(np.abs(x.diff().fillna(0)), t)
    smoothrng_val = calculate_ema(avrng, max(1, wper)) * m
    return smoothrng_val.clip(lower=1e-8).bfill().ffill()

def rngfilt(x: pd.Series, r: pd.Series) -> pd.Series:
    """Range filter"""
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

def calculate_cirrus_cloud(df: pd.DataFrame, config: Config) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Calculate Cirrus Cloud indicator"""
    close = df['close'].astype(float)
    
    smrngx1x = smoothrng(close, config.x1, config.x2)
    smrngx1x2 = smoothrng(close, config.x3, config.x4)
    
    filtx1 = rngfilt(close, smrngx1x)
    filtx12 = rngfilt(close, smrngx1x2)
    
    upw = filtx1 < filtx12
    dnw = filtx1 > filtx12
    
    return upw, dnw, filtx1, filtx12

def kalman_filter(src: pd.Series, length: int, R=0.01, Q=0.1) -> pd.Series:
    """Kalman filter for smoothing"""
    result = []
    estimate = np.nan
    error_est = 1.0
    error_meas = R * max(1, length)
    Q_div_length = Q / max(1, length)
    
    for i in range(len(src)):
        current = src.iloc[i]
        
        if np.isnan(estimate):
            estimate = src.iloc[i-1] if i > 0 else current
            result.append(np.nan if i == 0 else estimate)
            continue
        
        prediction = estimate
        kalman_gain = error_est / (error_est + error_meas)
        estimate = prediction + kalman_gain * (current - prediction)
        error_est = (1 - kalman_gain) * error_est + Q_div_length
        
        result.append(estimate)
    
    return pd.Series(result, index=src.index)

def calculate_smooth_rsi(df: pd.DataFrame, rsi_len: int, kalman_len: int) -> pd.Series:
    """Calculate smoothed RSI"""
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

def calculate_magical_momentum_hist(
    df: pd.DataFrame,
    period: int = 144,
    responsiveness: float = 0.9,
    config: Config = None
) -> pd.Series:
    """Magical Momentum Histogram (faithful to Pinescript v6)"""
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


# ============================================================================
# Candle Indexing
# ============================================================================
def get_last_closed_index(df: Optional[pd.DataFrame], resolution_min: int) -> Optional[int]:
    """
    Determine the last fully closed candle index.
    
    This function uses timestamps to check if the last candle in the DataFrame
    is complete (closed) by comparing its timestamp against the current time.
    If the last candle's timestamp is >= the start of the current interval,
    it's still forming, so we return the second-to-last index.
    
    Args:
        df: DataFrame with 'timestamp' column
        resolution_min: Candle resolution in minutes
    
    Returns:
        Index of last closed candle, or None if not ready
    """
    if df is None or df.empty:
        return None
    
    now_ts = int(time.time())
    last_ts = int(df['timestamp'].iloc[-1])
    current_interval_start = now_ts - (now_ts % (resolution_min * 60))
    
    # If last timestamp is in current interval, it's not closed yet
    if last_ts >= current_interval_start:
        return len(df) - 2 if len(df) >= 2 else None
    
    return len(df) - 1


# ============================================================================
# Trading Logic Evaluation
# ============================================================================
def evaluate_pair_logic(
    pair_name: str,
    df_15m: pd.DataFrame,
    df_5m: pd.DataFrame,
    last_state_for_pair: Optional[Dict[str, Any]],
    config: Config
) -> Optional[Dict[str, Any]]:
    """Evaluate trading signals for a pair"""
    try:
        last_i_15 = get_last_closed_index(df_15m, 15)
        last_i_5 = get_last_closed_index(df_5m, 5)
        
        if last_i_15 is None or last_i_15 < 3 or last_i_5 is None:
            logger.debug(
                "Indexing not ready",
                extra={"context": {
                    "pair": pair_name,
                    "last_i_15": last_i_15,
                    "last_i_5": last_i_5
                }}
            )
            return None

        # Calculate indicators
        magical_hist = calculate_magical_momentum_hist(
            df_15m,
            period=144,
            responsiveness=0.9,
            config=config
        )
        
        if magical_hist is None or len(magical_hist) <= last_i_15:
            logger.debug("MMH not ready", extra={"context": {"pair": pair_name}})
            return None

        # Extract MMH values
        mmh_curr = float(magical_hist.iloc[last_i_15])
        mmh_prev1 = float(magical_hist.iloc[last_i_15 - 1])
        mmh_prev2 = float(magical_hist.iloc[last_i_15 - 2])
        mmh_prev3 = float(magical_hist.iloc[last_i_15 - 3])

        # Calculate other indicators
        ppo, ppo_signal = calculate_ppo(
            df_15m,
            config.ppo_fast,
            config.ppo_slow,
            config.ppo_signal,
            config.ppo_use_sma
        )
        
        rma_50 = calculate_rma(df_15m['close'], config.rma_50_period)
        rma_200 = calculate_rma(df_5m['close'], config.rma_200_period)
        
        upw, dnw, _, _ = calculate_cirrus_cloud(df_15m, config)
        smooth_rsi = calculate_smooth_rsi(
            df_15m,
            config.srsi_rsi_len,
            config.srsi_kalman_len
        )

        # Validate series lengths
        indicators = {
            "ppo": (ppo, last_i_15),
            "ppo_signal": (ppo_signal, last_i_15),
            "rma_50": (rma_50, last_i_15),
            "rma_200": (rma_200, last_i_5),
            "smooth_rsi": (smooth_rsi, last_i_15)
        }
        
        for name, (series, idx) in indicators.items():
            if series is None or len(series) <= idx:
                logger.debug(
                    "Indicator not ready",
                    extra={"context": {"pair": pair_name, "indicator": name}}
                )
                return None

        # Extract indicator values
        ppo_curr = float(ppo.iloc[last_i_15])
        ppo_prev = float(ppo.iloc[last_i_15 - 1])
        ppo_signal_curr = float(ppo_signal.iloc[last_i_15])
        ppo_signal_prev = float(ppo_signal.iloc[last_i_15 - 1])
        smooth_rsi_curr = float(smooth_rsi.iloc[last_i_15])
        smooth_rsi_prev = float(smooth_rsi.iloc[last_i_15 - 1])

        # Candle data
        close_curr = float(df_15m['close'].iloc[last_i_15])
        open_curr = float(df_15m['open'].iloc[last_i_15])
        high_curr = float(df_15m['high'].iloc[last_i_15])
        low_curr = float(df_15m['low'].iloc[last_i_15])

        rma50_curr = float(rma_50.iloc[last_i_15])
        rma200_curr = float(rma_200.iloc[last_i_5])

        # Check for NaN values
        indicator_values = [
            ppo_curr, ppo_prev, ppo_signal_curr, ppo_signal_prev,
            smooth_rsi_curr, smooth_rsi_prev, rma50_curr, rma200_curr,
            mmh_curr, mmh_prev1, mmh_prev2, mmh_prev3
        ]
        
        if any(pd.isna(x) for x in indicator_values):
            logger.debug("NaN in indicators", extra={"context": {"pair": pair_name}})
            return None

        # Candle analysis
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

        # Signal conditions
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

        # MMH reversal logic
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

        # Cloud state
        cloud_state = "neutral"
        if config.cirrus_cloud_enabled:
            cloud_up = bool(upw.iloc[last_i_15])
            cloud_down = bool(dnw.iloc[last_i_15])
            cloud_state = "green" if (cloud_up and not cloud_down) else \
                         "red" if (cloud_down and not cloud_up) else "neutral"

        # Signal conditions
        signals = {
            "buy_mmh_reversal": {
                "conditions": {
                    "mmh_reversal_buy": mmh_reversal_buy,
                    "close_above_rma50": close_above_rma50,
                    "close_above_rma200": close_above_rma200,
                    "magical_hist_curr>0": mmh_curr > 0,
                    "cloud_green": cloud_state == "green",
                    "strong_bullish_close": strong_bullish_close,
                },
                "message": f"{config.bot_name}\n🟢 {pair_name} - BUY (MMH Reversal)\nMMH 15m Reversal Up ({mmh_curr:.5f})\nPrice: ${close_curr:,.2f}"
            },
            "sell_mmh_reversal": {
                "conditions": {
                    "mmh_reversal_sell": mmh_reversal_sell,
                    "close_below_rma50": close_below_rma50,
                    "close_below_rma200": close_below_rma200,
                    "magical_hist_curr<0": mmh_curr < 0,
                    "cloud_red": cloud_state == "red",
                    "strong_bearish_close": strong_bearish_close,
                },
                "message": f"{config.bot_name}\n🔴 {pair_name} - SELL (MMH Reversal)\nMMH 15m Reversal Down ({mmh_curr:.5f})\nPrice: ${close_curr:,.2f}"
            },
            "buy_srsi50": {
                "conditions": {
                    "srsi_cross_up_50": srsi_cross_up_50,
                    "ppo_above_signal": ppo_above_signal,
                    "ppo_below_030": ppo_below_030,
                    "close_above_rma50": close_above_rma50,
                    "close_above_rma200": close_above_rma200,
                    "cloud_green": cloud_state == "green",
                    "strong_bullish_close": strong_bullish_close,
                    "magical_hist_curr>0": mmh_curr > 0,
                },
                "message": f"{config.bot_name}\n▲ {pair_name} - BUY (SRSI 50)\nSRSI 15m Cross Up 50 ({smooth_rsi_curr:.2f})\nPrice: ${close_curr:,.2f}"
            },
            "sell_srsi50": {
                "conditions": {
                    "srsi_cross_down_50": srsi_cross_down_50,
                    "ppo_below_signal": ppo_below_signal,
                    "ppo_above_minus030": ppo_above_minus030,
                    "close_below_rma50": close_below_rma50,
                    "close_below_rma200": close_below_rma200,
                    "cloud_red": cloud_state == "red",
                    "strong_bearish_close": strong_bearish_close,
                    "magical_hist_curr<0": mmh_curr < 0,
                },
                "message": f"{config.bot_name}\n🔻 {pair_name} - SELL (SRSI 50)\nSRSI 15m Cross Down 50 ({smooth_rsi_curr:.2f})\nPrice: ${close_curr:,.2f}"
            },
            "long_zero": {
                "conditions": {
                    "ppo_cross_above_zero": ppo_cross_above_zero,
                    "ppo_above_signal": ppo_above_signal,
                    "close_above_rma50": close_above_rma50,
                    "close_above_rma200": close_above_rma200,
                    "cloud_green": cloud_state == "green",
                    "strong_bullish_close": strong_bullish_close,
                    "magical_hist_curr>0": mmh_curr > 0,
                },
                "message": f"{config.bot_name}\n🟢 {pair_name} - LONG\nPPO crossing above 0 ({ppo_curr:.2f})\nPrice: ${close_curr:,.2f}"
            },
            "long_011": {
                "conditions": {
                    "ppo_cross_above_011": ppo_cross_above_011,
                    "ppo_above_signal": ppo_above_signal,
                    "close_above_rma50": close_above_rma50,
                    "close_above_rma200": close_above_rma200,
                    "cloud_green": cloud_state == "green",
                    "strong_bullish_close": strong_bullish_close,
                    "magical_hist_curr>0": mmh_curr > 0,
                },
                "message": f"{config.bot_name}\n🟢 {pair_name} - LONG\nPPO crossing above 0.11 ({ppo_curr:.2f})\nPrice: ${close_curr:,.2f}"
            },
            "short_zero": {
                "conditions": {
                    "ppo_cross_below_zero": ppo_cross_below_zero,
                    "ppo_below_signal": ppo_below_signal,
                    "close_below_rma50": close_below_rma50,
                    "close_below_rma200": close_below_rma200,
                    "cloud_red": cloud_state == "red",
                    "strong_bearish_close": strong_bearish_close,
                    "magical_hist_curr<0": mmh_curr < 0,
                },
                "message": f"{config.bot_name}\n🔴 {pair_name} - SHORT\nPPO crossing below 0 ({ppo_curr:.2f})\nPrice: ${close_curr:,.2f}"
            },
            "short_011": {
                "conditions": {
                    "ppo_cross_below_minus011": ppo_cross_below_minus011,
                    "ppo_below_signal": ppo_below_signal,
                    "close_below_rma50": close_below_rma50,
                    "close_below_rma200": close_below_rma200,
                    "cloud_red": cloud_state == "red",
                    "strong_bearish_close": strong_bearish_close,
                    "magical_hist_curr<0": mmh_curr < 0,
                },
                "message": f"{config.bot_name}\n🔴 {pair_name} - SHORT\nPPO crossing below -0.11 ({ppo_curr:.2f})\nPrice: ${close_curr:,.2f}"
            },
        }

        # Evaluate signals
        current_state = None
        send_message = None
        now_ts_int = int(time.time())
        ist = pytz.timezone("Asia/Kolkata")
        formatted_time = datetime.now(ist).strftime('%d-%m-%Y @ %H:%M IST')

        for signal_name, signal_def in signals.items():
            if all(signal_def["conditions"].values()):
                current_state = signal_name
                send_message = f"{signal_def['message']}\n{formatted_time}"
                break

        # Idempotency check
        last_state_value = last_state_for_pair.get("state") if isinstance(last_state_for_pair, dict) else None
        
        if current_state is None:
            if last_state_value and last_state_value != 'NO_SIGNAL':
                return {"state": "NO_SIGNAL", "ts": now_ts_int}
            return last_state_for_pair

        if current_state == last_state_value:
            logger.debug(
                "Signal unchanged",
                extra={"context": {"pair": pair_name, "state": current_state}}
            )
            return last_state_for_pair

        result = {
            "state": current_state,
            "ts": now_ts_int,
            "message": send_message
        }
        
        return result

    except Exception as e:
        logger.error(
            f"Error evaluating logic for {pair_name}: {e}",
            exc_info=True,
            extra={"context": {"pair": pair_name}}
        )
        return None


# ============================================================================
# Product Mapping
# ============================================================================
def build_products_map_from_api_result(
    api_products: dict,
    config: Config
) -> Dict[str, dict]:
    """Build tradable pairs map from API response"""
    if not api_products or not api_products.get("result"):
        logger.error("No products in API result")
        return {}
    
    products_map = {}
    
    for p in api_products["result"]:
        try:
            symbol = p.get("symbol", "")
            symbol_norm = symbol.replace("_USDT", "USD").replace("USDT", "USD")
            
            if p.get("contract_type") != "perpetual_futures":
                continue
            
            for pair_name in config.pairs:
                if symbol_norm == pair_name or symbol_norm.replace("_", "") == pair_name:
                    products_map[pair_name] = {
                        "id": p.get("id"),
                        "symbol": p.get("symbol"),
                        "contract_type": p.get("contract_type")
                    }
                    break
        
        except Exception as e:
            logger.debug(f"Error processing product: {e}", exc_info=True)
            continue
    
    logger.info(
        f"Mapped {len(products_map)} tradable pairs",
        extra={"context": {"pairs": list(products_map.keys())}}
    )
    return products_map


# ============================================================================
# Telegram Queue
# ============================================================================
class TelegramQueue:
    """Telegram message queue with rate limiting"""
    
    def __init__(self, config: Config):
        self.config = config
        self._last_sent = 0.0
        self._lock = asyncio.Lock()

    async def send(self, session: aiohttp.ClientSession, message: str) -> bool:
        """Send message with rate limiting"""
        async with self._lock:
            now = time.time()
            time_since = now - self._last_sent
            
            if time_since < self.config.telegram_backoff_base:
                await asyncio.sleep(self.config.telegram_backoff_base - time_since)
            
            self._last_sent = time.time()
            return await self._send_with_retries(session, message)

    async def _send_once(self, session: aiohttp.ClientSession, message: str) -> bool:
        """Single send attempt"""
        # Validate config
        if not self.config.telegram.bot_token or not self.config.telegram.chat_id:
            logger.warning("Telegram not configured")
            return False
        
        url = f"https://api.telegram.org/bot{self.config.telegram.bot_token}/sendMessage"
        data = {
            "chat_id": self.config.telegram.chat_id,
            "text": message,
            "parse_mode": "HTML"
        }
        
        try:
            async with session.post(url, data=data, timeout=10) as resp:
                if resp.status == 429:
                    retry_after = int(resp.headers.get("Retry-After", 30))
                    logger.warning(f"Telegram rate limited", extra={
                        "context": {"retry_after": retry_after}
                    })
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
                
                logger.warning(
                    "Telegram API error",
                    extra={"context": {"response": js}}
                )
                return False
        
        except asyncio.TimeoutError:
            logger.error("Telegram send timeout")
            return False
        except Exception as e:
            logger.error(
                "Telegram send error",
                exc_info=True,
                extra={"context": {"error": str(e)}}
            )
            return False

    async def _send_with_retries(self, session: aiohttp.ClientSession, message: str) -> bool:
        """Send with exponential backoff retries"""
        last_error = None
        
        for attempt in range(1, self.config.telegram_retries + 1):
            ok = await self._send_once(session, message)
            if ok:
                return True
            
            last_error = "Send failed"
            
            if attempt < self.config.telegram_retries:
                backoff = (self.config.telegram_backoff_base ** (attempt - 1)) + \
                         asyncio.get_event_loop().time() % 0.3
                await asyncio.sleep(backoff)
        
        logger.error(
            "Telegram send failed after retries",
            extra={"context": {"retries": self.config.telegram_retries}}
        )
        return False


# ============================================================================
# Pair Checker
# ============================================================================
async def check_pair(
    session: aiohttp.ClientSession,
    fetcher: DataFetcher,
    products_map: Dict[str, dict],
    pair_name: str,
    last_state_for_pair: Optional[Dict[str, Any]],
    telegram_queue: TelegramQueue,
    config: Config
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Check a single pair for signals"""
    try:
        prod = products_map.get(pair_name)
        if not prod:
            logger.debug("No product mapping", extra={"context": {"pair": pair_name}})
            return None
        
        # Get pair-specific limits
        sp = config.special_pairs.get(pair_name, SpecialPairConfig())
        symbol = prod["symbol"]
        
        # Fetch candles concurrently
        res15, res5 = await asyncio.gather(
            fetcher.fetch_candles(session, symbol, "15", sp.limit_15m),
            fetcher.fetch_candles(session, symbol, "5", sp.limit_5m)
        )
        
        # Parse candles
        df_15m = parse_candles_result(res15)
        df_5m = parse_candles_result(res5)
        
        # Validate data
        if df_15m is None or len(df_15m) < (sp.min_required + 2):
            logger.warning(
                "Insufficient 15m data",
                extra={"context": {
                    "pair": pair_name,
                    "actual": len(df_15m) if df_15m else 0,
                    "required": sp.min_required + 2
                }}
            )
            return None
        
        if df_5m is None or len(df_5m) < (sp.min_required_5m + 2):
            logger.warning(
                "Insufficient 5m data",
                extra={"context": {
                    "pair": pair_name,
                    "actual": len(df_5m) if df_5m else 0,
                    "required": sp.min_required_5m + 2
                }}
            )
            return None
        
        # Ensure numeric types
        for df in (df_15m, df_5m):
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # Evaluate signals
        new_state = evaluate_pair_logic(
            pair_name, df_15m, df_5m,
            last_state_for_pair, config
        )
        
        if not new_state:
            return None
        
        # Send alert if needed
        message = new_state.get("message")
        if message:
            await telegram_queue.send(session, message)
        
        return pair_name, new_state
        
    except Exception as e:
        logger.error(
            f"Error checking pair {pair_name}",
            exc_info=True,
            extra={"context": {"pair": pair_name, "error": str(e)}}
        )
        return None


# ============================================================================
# Batch Processing
# ============================================================================
async def process_batch(
    session: aiohttp.ClientSession,
    fetcher: DataFetcher,
    products_map: Dict[str, dict],
    batch_pairs: List[str],
    state_db: StateDB,
    telegram_queue: TelegramQueue,
    config: Config
) -> List[Tuple[str, Dict[str, Any]]]:
    """Process a batch of pairs"""
    results = []
    tasks = []
    
    for pair_name in batch_pairs:
        if pair_name not in products_map:
            logger.warning(
                "No product mapping in batch",
                extra={"context": {"pair": pair_name}}
            )
            continue
        
        last_state = state_db.get(pair_name)
        task = asyncio.create_task(
            check_pair(
                session, fetcher, products_map,
                pair_name, last_state, telegram_queue, config
            )
        )
        tasks.append(task)
    
    # Gather results with error handling
    for task in tasks:
        try:
            res = await task
            if res:
                results.append(res)
        except Exception:
            logger.exception("Batch task error")
    
    return results


# ============================================================================
# Dead Man's Switch
# ============================================================================
async def check_dead_mans_switch(
    state_db: StateDB,
    telegram_queue: TelegramQueue,
    session: aiohttp.ClientSession,
    config: Config
) -> bool:
    """
    Check if bot is running properly and alert if not.
    
    Returns True if healthy, False if alert was sent.
    """
    if not config.dead_mans_switch.enabled:
        logger.debug("DMS disabled")
        return True
    
    dms_status = state_db.get_dead_mans_switch_status()
    if not dms_status:
        logger.warning("DMS status not found")
        return True
    
    last_success = dms_status["last_success_run"]
    now = int(time.time())
    hours_since_success = (now - last_success) / 3600
    
    # Check if we should alert
    if hours_since_success < config.dead_mans_switch.hours:
        return True
    
    # Check cooldown
    last_alert = dms_status["last_alert_sent"]
    cooldown_sec = config.dead_mans_switch.cooldown_seconds
    
    if last_alert and (now - last_alert) < cooldown_sec:
        logger.debug(
            "DMS alert on cooldown",
            extra={"context": {"cooldown_remaining": cooldown_sec - (now - last_alert)}}
        )
        return False
    
    # Send alert
    ist = pytz.timezone("Asia/Kolkata")
    formatted_time = datetime.now(ist).strftime('%d-%m-%Y @ %H:%M IST')
    
    message = (
        f"🚨 {config.bot_name} - DEAD MAN'S SWITCH ALERT\n"
        f"No successful run for {hours_since_success:.1f} hours\n"
        f"Last success: {datetime.fromtimestamp(last_success, tz=ist).strftime('%d-%m-%Y @ %H:%M IST')}\n"
        f"Time: {formatted_time}"
    )
    
    try:
        await telegram_queue.send(session, message)
        state_db.record_dms_alert("alerted")
        logger.critical(
            "DMS alert sent",
            extra={"context": {
                "hours_since_success": hours_since_success,
                "last_success": last_success
            }}
        )
        return False
    except Exception as e:
        logger.error(
            "Failed to send DMS alert",
            exc_info=True,
            extra={"context": {"error": str(e)}}
        )
        return False


# ============================================================================
# Health Check
# ============================================================================
async def health_check(http_client: HTTPClient, config: Config) -> bool:
    """Verify API connectivity"""
    try:
        async with http_client.session() as session:
            url = f"{config.delta_api_base}/v2/products"
            data = await http_client.fetch_json(session, url)
            return data is not None
    except Exception as e:
        logger.warning(
            "Health check failed",
            extra={"context": {"error": str(e)}}
        )
        return False


# ============================================================================
# Main Run Logic
# ============================================================================
async def run_once(
    config: Config,
    state_db: StateDB,
    http_client: HTTPClient,
    telegram_queue: TelegramQueue
) -> Dict[str, Any]:
    """Execute one full run"""
    start_time = time.time()
    run_stats = {
        "pairs_processed": 0,
        "alerts_sent": 0,
        "state_updates": 0,
        "errors": [],
        "duration": 0
    }
    
    logger.info(
        "Starting run",
        extra={"context": {"pairs": len(config.pairs), "batch_size": config.batch_size}}
    )

    try:
        # Memory guard
        try:
            p = psutil.Process(os.getpid())
            rss = p.memory_info().rss
            if rss > config.memory_limit_bytes:
                raise MemoryError(f"Memory usage {rss} exceeds limit {config.memory_limit_bytes}")
        except MemoryError:
            raise
        except Exception:
            pass  # psutil may not be available

        # Prune old records
        deleted = state_db.prune_old_records(config.state_expiry_days)
        if deleted > 0:
            logger.info(f"Pruned {deleted} old state records")

        # Load previous states
        last_alerts = state_db.load_all()
        logger.info(f"Loaded {len(last_alerts)} previous states")

        # Health check
        if not await health_check(http_client, config):
            raise RuntimeError("API health check failed")

        # Fetch products
        async with http_client.session() as session:
            prod_resp = await fetcher.fetch_products(session)
            
            if not prod_resp:
                raise RuntimeError("Failed to fetch products")
            
            products_map = build_products_map_from_api_result(prod_resp, config)
            
            if not products_map:
                raise RuntimeError("No tradable pairs found")
            
            # Send test message if configured
            if config.send_test_message:
                ist = pytz.timezone("Asia/Kolkata")
                test_msg = (
                    f"🔔 {config.bot_name} Started\n"
                    f"Time: {datetime.now(ist).strftime('%d-%m-%Y @ %H:%M IST')}\n"
                    f"Pairs: {len(config.pairs)} | Debug: {config.debug_mode}"
                )
                await telegram_queue.send(session, test_msg)

            # Process pairs in batches
            pairs_to_process = [p for p in config.pairs if p in products_map]
            batch_size = max(1, config.batch_size)
            
            logger.info(
                f"Processing {len(pairs_to_process)} pairs",
                extra={"context": {"batch_size": batch_size}}
            )
            
            all_results = []
            
            for i in range(0, len(pairs_to_process), batch_size):
                batch = pairs_to_process[i:i+batch_size]
                logger.debug(f"Processing batch {i//batch_size + 1}: {batch}")
                
                batch_results = await process_batch(
                    session, fetcher, products_map,
                    batch, state_db, telegram_queue, config
                )
                
                all_results.extend(batch_results)
                run_stats["pairs_processed"] += len(batch)
                
                # Inter-batch delay
                if i + batch_size < len(pairs_to_process):
                    await asyncio.sleep(random.uniform(0.5, 1.0))

            # Update states
            for pair_name, new_state in all_results:
                if not isinstance(new_state, dict):
                    continue
                
                prev = state_db.get(pair_name)
                if prev != new_state:
                    state_db.set(pair_name, new_state.get("state"), new_state.get("ts"))
                    run_stats["state_updates"] += 1
                
                if new_state.get("message"):
                    run_stats["alerts_sent"] += 1

            # Update success timestamp
            state_db.update_last_success_run()

        # Check dead man's switch
        await check_dead_mans_switch(state_db, telegram_queue, session, config)
        
        # Update stats
        run_stats["duration"] = time.time() - start_time
        
        logger.info(
            "Run complete",
            extra={"context": run_stats}
        )
        
        return run_stats        
    except Exception as e:
        logger.critical(
            "Run failed",
            exc_info=True,
            extra={"context": {"error": str(e)}}
        )
        run_stats["errors"].append(str(e))
        run_stats["duration"] = time.time() - start_time
        raise
    finally:
        # Cleanup
        try:
            gc.collect()
            if 'pandas' in sys.modules:
                import pandas as pd
                pd.DataFrame().empty  # Trigger pandas cleanup
        except Exception:
            pass


# ============================================================================
# Signal Handlers
# ============================================================================
stop_requested = False

def request_stop(signum, frame):
    """Handle shutdown signals"""
    global stop_requested
    logger.info(f"Received signal {signum} - stopping gracefully")
    stop_requested = True

signal.signal(signal.SIGINT, request_stop)
signal.signal(signal.SIGTERM, request_stop)


# ============================================================================
# CLI
# ============================================================================
def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="MACD/PPO Trading Bot - Production Ready"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--once",
        action="store_true",
        help="Run once and exit"
    )
    group.add_argument(
        "--loop",
        type=int,
        metavar="SECONDS",
        help="Run in loop every N seconds"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config_macd.json",
        help="Path to config file"
    )
    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except ValueError as e:
        print(f"❌ Configuration error: {e}", file=sys.stderr)
        sys.exit(1)

    # Setup logging
    global logger
    logger = setup_logging(config.logging)
    
    logger.info(
        "Bot starting",
        extra={"context": {
            "config_file": args.config,
            "debug_mode": config.debug_mode
        }}
    )

    # PID lock
    pid_lock = PidFileLock(config.pid_lock_path)
    if not pid_lock.acquire():
        logger.critical(
            "Another instance is running",
            extra={"context": {"pid_file": config.pid_lock_path}}
        )
        sys.exit(2)

    try:
        # Initialize components
        state_db = StateDB(config.state_db_path)
        http_client = HTTPClient(config)
        
        # Create global fetcher reference for health check
        global fetcher
        fetcher = DataFetcher(http_client, config)
        
        telegram_queue = TelegramQueue(config)

        # Run mode
        if args.once:
            asyncio.run(run_once_with_timeout(config, state_db, http_client, telegram_queue))
        elif args.loop:
            interval = max(30, args.loop)
            logger.info(f"Starting loop mode (interval={interval}s)")
            
            while not stop_requested:
                start = time.time()
                
                try:
                    asyncio.run(run_once_with_timeout(config, state_db, http_client, telegram_queue))
                except Exception:
                    logger.exception("Unhandled exception in run loop")
                
                # Sleep interval with stop support
                elapsed = time.time() - start
                to_sleep = max(0, interval - elapsed)
                
                slept = 0.0
                while slept < to_sleep and not stop_requested:
                    sleep_time = min(1.0, to_sleep - slept)
                    time.sleep(sleep_time)
                    slept += sleep_time
        else:
            asyncio.run(run_once_with_timeout(config, state_db, http_client, telegram_queue))
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception:
        logger.critical("Fatal error", exc_info=True)
        sys.exit(1)
    finally:
        pid_lock.release()
        logger.info("Bot stopped")


async def run_once_with_timeout(
    config: Config,
    state_db: StateDB,
    http_client: HTTPClient,
    telegram_queue: TelegramQueue
):
    """Run with timeout wrapper"""
    try:
        await asyncio.wait_for(
            run_once(config, state_db, http_client, telegram_queue),
            timeout=config.run_timeout_seconds
        )
    except asyncio.TimeoutError:
        logger.error(f"Run timed out after {config.run_timeout_seconds}s")
        raise SystemExit(3)


# ============================================================================
# Module Globals
# ============================================================================
logger = logging.getLogger("macd_bot")
fetcher: Optional[DataFetcher] = None

if __name__ == "__main__":
    main()