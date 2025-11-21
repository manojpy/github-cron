# File: macd_unified.py
from __future__ import annotations

import os
import sys
import json
import time
import asyncio
import random
import logging
import atexit
import gc
import ssl
import signal
import hashlib
import re
import uuid
from collections import deque
from typing import Dict, Any, Optional, Tuple, List, ClassVar
from datetime import datetime, timezone, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import aiohttp
import pandas as pd
import numpy as np
import psutil
import redis.asyncio as redis
from redis.exceptions import ConnectionError as RedisConnectionError
from pydantic import BaseModel, Field, field_validator
from aiohttp import ClientConnectorError, ClientResponseError, TCPConnector
from logging.handlers import RotatingFileHandler

# -------------------------
# IST Time Converter
# -------------------------
def _to_ist_str(dt_or_ts) -> str:
    """
    Convert a datetime or timestamp (seconds) in UTC to an IST formatted string.
    Returns: YYYY-MM-DD HH:MM:SS IST
    """
    try:
        if isinstance(dt_or_ts, (int, float)):
            dt = datetime.fromtimestamp(dt_or_ts, tz=timezone.utc)
        elif isinstance(dt_or_ts, datetime):
            dt = dt_or_ts
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        else:
            # attempt parse
            dt = datetime.fromisoformat(str(dt_or_ts))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        ist = dt.astimezone(ZoneInfo("Asia/Kolkata"))
        return ist.strftime("%Y-%m-%d %H:%M:%S IST")
    except Exception:
        try:
            # fallback: treat as epoch seconds
            ts = float(dt_or_ts)
            ist = datetime.fromtimestamp(ts, tz=ZoneInfo("Asia/Kolkata"))
            return ist.strftime("%Y-%m-%d %H:%M:%S IST")
        except Exception:
            return str(dt_or_ts)

def get_current_ist_str() -> str:
    """Get current IST time as formatted string"""
    return datetime.now(timezone.utc).astimezone(ZoneInfo("Asia/Kolkata")).strftime('%d-%m-%Y @ %H:%M IST')

# -------------------------
# Version & Constants
# -------------------------
__version__ = "1.0.1"

class Constants:
    """Consolidated magic numbers"""
    MIN_WICK_RATIO = 0.2
    PPO_THRESHOLD_BUY = 0.20
    PPO_THRESHOLD_SELL = -0.20
    RSI_THRESHOLD = 50
    PPO_RSI_GUARD_BUY = 0.30
    PPO_RSI_GUARD_SELL = -0.30
    PPO_011_THRESHOLD = 0.11
    PPO_011_THRESHOLD_SELL = -0.11
    STARTUP_GRACE_PERIOD = int(os.getenv('STARTUP_GRACE_PERIOD', 300))  # 5 minutes
    HEALTH_CHECK_PORT = int(os.getenv("PORT", "10000"))

# -------------------------
# Pydantic v2 Configuration Model
# -------------------------
class BotConfig(BaseModel):
    TELEGRAM_BOT_TOKEN: str = Field(..., min_length=1)
    TELEGRAM_CHAT_ID: str = Field(..., min_length=1)
    DEBUG_MODE: bool = True  # keep debug on until sorted
    SEND_TEST_MESSAGE: bool = True
    BOT_NAME: str = "Unified Alert Bot"
    DELTA_API_BASE: str = "https://api.india.delta.exchange"
    PAIRS: List[str] = Field(default=["BTCUSD", "ETHUSD", "AVAXUSD", "BCHUSD", "XRPUSD", "BNBUSD", "LTCUSD", "DOTUSD", "ADAUSD", "SUIUSD", "AAVEUSD", "SOLUSD"], min_length=1)
    SPECIAL_PAIRS: Dict[str, Dict[str, int]] = {
        "SOLUSD": {"limit_15m": 250, "min_required": 200, "limit_5m": 350, "min_required_5m": 250}
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
    REDIS_URL: str = Field(..., min_length=1)
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
    ENABLE_DMS_PLUGIN: bool = False
    ENABLE_HEALTH_TRACKER: bool = True
    ENABLE_AUTO_RESTART: bool = False
    AUTO_RESTART_MAX_RETRIES: int = 3
    AUTO_RESTART_COOLDOWN_SEC: int = 10
    DEAD_MANS_COOLDOWN_SECONDS: int = 14400
    LOG_LEVEL: str = "DEBUG"  # explicit debug

    ENABLE_VWAP: bool = True
    ENABLE_PIVOT: bool = True
    PIVOT_LOOKBACK_PERIOD: int = 15

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

    @field_validator('TELEGRAM_BOT_TOKEN')
    @classmethod
    def validate_token(cls, v: str) -> str:
        pattern = r'^\d+:[A-Za-z0-9_-]+$'
        if not re.match(pattern, v):
            raise ValueError('Invalid Telegram bot token format')
        return v

    @field_validator('TELEGRAM_CHAT_ID')
    @classmethod
    def validate_chat_id(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('Chat ID cannot be empty')
        return v.strip()

# -------------------------------
# Configuration loader
# -------------------------------
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

    # Overwrite config values with environment variables from GitHub Secrets
    config_data["TELEGRAM_BOT_TOKEN"] = os.getenv("TELEGRAM_BOT_TOKEN")
    config_data["TELEGRAM_CHAT_ID"] = os.getenv("TELEGRAM_CHAT_ID")
    config_data["REDIS_URL"] = os.getenv("REDIS_URL")

    if not config_data["TELEGRAM_BOT_TOKEN"] or not config_data["TELEGRAM_CHAT_ID"]:
        raise ValueError("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID in environment")
    if not config_data.get("REDIS_URL"):
        raise ValueError("Missing REDIS_URL in environment or config file")

    try:
        return BotConfig(**config_data)
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        sys.exit(1)

cfg = load_config()

# -------------------------
# Structured Logging Setup
# -------------------------
class SecretFilter(logging.Filter):
    """Redact sensitive information from logs"""
    def filter(self, record):
        if hasattr(record, 'msg'):
            msg = str(record.msg)
            msg = re.sub(r'bot\d+:[A-Za-z0-9_-]+', '[REDACTED_TELEGRAM_TOKEN]', msg)
            msg = re.sub(r'chat_id=\d+', '[REDACTED_CHAT_ID]', msg)
            record.msg = msg
        return True

def setup_logging():
    logger = logging.getLogger("macd_bot")
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    log_level = getattr(logging, cfg.LOG_LEVEL, logging.DEBUG)
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
    console_handler.addFilter(SecretFilter())
    logger.addHandler(console_handler)

    # File logging is skipped in standard GitHub Actions cron environment
    if not os.getenv("RENDER"): # Check environment variable for RENDER
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
            file_handler.addFilter(SecretFilter())
            # logger.addHandler(file_handler) # Commented out for GitHub Actions
        except Exception as e:
            logger.warning(f"⚠️ Could not set up file logging: {e}")

    return logger

logger = setup_logging()

# -------------------------
# Global shutdown event for graceful termination
# -------------------------
shutdown_event = asyncio.Event()

def handle_shutdown(sig, frame):
    logger.warning(f"Received signal {sig}, shutting down gracefully...")
    shutdown_event.set()

signal.signal(signal.SIGTERM, handle_shutdown)
signal.signal(signal.SIGINT, handle_shutdown)

# -------------------------
# Session Manager with SSL
# -------------------------
class SessionManager:
    _session: ClassVar[Optional[aiohttp.ClientSession]] = None

    @classmethod
    def get_session(cls) -> aiohttp.ClientSession:
        if cls._session is None or cls._session.closed:
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = True
            ssl_context.verify_mode = ssl.CERT_REQUIRED

            connector = TCPConnector(
                limit=cfg.TCP_CONN_LIMIT,
                ssl=ssl_context,
                force_close=True,
                enable_cleanup_closed=True,
                ttl_dns_cache=300
            )
            timeout = aiohttp.ClientTimeout(total=cfg.HTTP_TIMEOUT)
            cls._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={'User-Agent': f'{cfg.BOT_NAME}/{__version__}'}
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
# Redis Distributed Lock (Replaces PID Lock)
# -------------------------
class RedisLock:
    """
    Safe Redis lock using a UUID token and an atomic Lua release.
    Ensures we only release the lock if the token matches (prevents deleting another process's lock).
    """
    RELEASE_LUA = """
    if redis.call("GET", KEYS[1]) == ARGV[1] then
        return redis.call("DEL", KEYS[1])
    else
        return 0
    end
    """

    def __init__(self, redis_client: redis.Redis, lock_key: str, expire: int = 600):
        self.redis = redis_client
        self.lock_key = f"lock:{lock_key}"
        self.expire = expire
        self.token: Optional[str] = None

    async def acquire(self) -> bool:
        try:
            token = str(uuid.uuid4())
            ok = await self.redis.set(self.lock_key, token, nx=True, ex=self.expire)
            if ok:
                self.token = token
                logger.debug(f"Acquired Redis lock: {self.lock_key} token={token}")
                return True
            logger.debug(f"Could not acquire Redis lock (held): {self.lock_key}")
            return False
        except Exception as e:
            logger.warning(f"Could not acquire Redis lock: {e}")
            return False

    async def extend(self) -> bool:
        if not self.token:
            return False
        try:
            val = await self.redis.get(self.lock_key)
            if val and (val.decode() if isinstance(val, (bytes, bytearray)) else val) == self.token:
                await self.redis.expire(self.lock_key, self.expire)
                logger.debug(f"Extended Redis lock: {self.lock_key}")
                return True
            logger.debug("Lock token mismatch on extend; not extending.")
            return False
        except Exception as e:
            logger.warning(f"Error extending Redis lock: {e}")
            return False

    async def release(self):
        if not self.token:
            return
        try:
            await self.redis.eval(self.RELEASE_LUA, 1, self.lock_key, self.token)
            logger.debug(f"Released Redis lock: {self.lock_key}")
        except Exception as e:
            logger.warning(f"Error releasing Redis lock: {e}")
        finally:
            self.token = None

class RedisStateStore:
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self._redis: Optional[redis.Redis] = None
        self.state_prefix = "pair_state:"
        self.meta_prefix = "metadata:"
        self.expiry_seconds = cfg.STATE_EXPIRY_DAYS * 86400

    async def connect(self):
        if self._redis is None:
            try:
                self._redis = redis.from_url(
                    self.redis_url,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True,
                    max_connections=10,
                    decode_responses=False
                )
                await self._redis.ping()
                logger.info("Connected to RedisStateStore")
            except RedisConnectionError as e:
                logger.critical(f"Redis connection failed: {e}")
                raise

    async def close(self):
        if self._redis:
            await self._redis.aclose()
            self._redis = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        if not self._redis:
            return None
        try:
            result = await self._redis.get(f"{self.state_prefix}{key}")
            if not result:
                return None
            data = json.loads(result.decode('utf-8'))
            return {"state": data.get("state"), "ts": int(data.get("ts", 0))}
        except Exception as e:
            logger.error(f"Failed to parse state for {key}: {e}")
            return None

    async def set(self, key: str, state: Optional[str], ts: Optional[int] = None):
        if not self._redis:
            return
        ts = int(ts or time.time())
        redis_key = f"{self.state_prefix}{key}"
        data = json.dumps({"state": state, "ts": ts})
        if self.expiry_seconds > 0:
            await self._redis.set(redis_key, data, ex=self.expiry_seconds)
        else:
            await self._redis.set(redis_key, data)

    async def get_metadata(self, key: str) -> Optional[str]:
        if not self._redis:
            return None
        try:
            result = await self._redis.get(f"{self.meta_prefix}{key}")
            return result.decode('utf-8') if result else None
        except Exception as e:
            logger.error(f"Failed to get metadata {key}: {e}")
            return None

    async def set_metadata(self, key: str, value: str):
        if not self._redis:
            return
        await self._redis.set(f"{self.meta_prefix}{key}", value)

    async def prune_old_records(self, expiry_days: int) -> int:
        if expiry_days <= 0 or self.expiry_seconds > 0:
            logger.debug("Using Redis TTL - manual pruning skipped")
            return 0

        last_prune_str = await self.get_metadata("last_prune")
        today = datetime.now(timezone.utc).date()
        if last_prune_str:
            try:
                last_prune_date = datetime.fromisoformat(last_prune_str).date()
                if last_prune_date >= today:
                    logger.debug("Daily prune already completed — skipping.")
                    return 0
            except Exception:
                pass
        await self.set_metadata("last_prune", datetime.now(timezone.utc).isoformat())
        logger.info("Daily prune check completed (relying on Redis TTLs for pair states).")
        return 0

# -------------------------
# Health Tracker
# -------------------------
class HealthTracker:
    def __init__(self, sdb: RedisStateStore):
        self.sdb = sdb

    async def record_pair_result(self, pair: str, success: bool, info: Optional[dict] = None):
        key = f"health:pair:{pair}"
        now = int(time.time())
        payload = {"last_checked": now, "last_success": now if success else None, "success": bool(success)}
        if info:
            try:
                payload.update({"info": info})
            except Exception:
                pass
        await self.sdb.set_metadata(key, json.dumps(payload))

    async def record_overall(self, summary: dict):
        key = "health:overall"
        payload = {"ts": int(time.time()), "summary": summary}
        await self.sdb.set_metadata(key, json.dumps(payload))

health_tracker: Optional[HealthTracker] = None

# -------------------------
# Circuit Breaker / Rate Limiter / Fetch Helpers
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
            if self.failure_count >= self.threshold:
                logger.critical(f"Circuit breaker OPEN for {self.timeout}s (until {self.last_failure_time + self.timeout})")
            return self.failure_count

    async def record_success(self):
        async with self.lock:
            if self.failure_count > 0:
                self.failure_count = 0
                self.last_failure_time = None
                logger.info("Circuit breaker: reset after success")

class RateLimitedFetcher:
    def __init__(self, max_per_minute: int = 60, concurrency: int = 4):
        self.max_per_minute = max_per_minute
        self.semaphore = asyncio.Semaphore(concurrency)
        self.requests = deque()
        self.lock = asyncio.Lock()

    async def call(self, func, *args, **kwargs):
        async with self.lock:
            now = time.time()
            while self.requests and now - self.requests[0] > 60:
                self.requests.popleft()

            if len(self.requests) >= self.max_per_minute:
                sleep_time = 60 - (now - self.requests[0])
                logger.debug(f"Rate limit reached; sleeping {sleep_time:.2f}s")
                await asyncio.sleep(sleep_time + random.uniform(0.05, 0.2))

            self.requests.append(time.time())

        async with self.semaphore:
            return await func(*args, **kwargs)

async def async_fetch_json(url: str, params: dict = None,
                           retries: int = 3, backoff: float = 1.5, timeout: int = 15,
                           circuit_breaker: Optional[CircuitBreaker] = None) -> Optional[dict]:
    if circuit_breaker and await circuit_breaker.is_open():
        logger.warning(f"Circuit breaker open; skipping fetch {url}")
        return None

    session = SessionManager.get_session()

    for attempt in range(1, retries + 1):
        if shutdown_event.is_set():
            logger.info("Shutdown signaled, aborting fetch")
            return None

        try:
            async with session.get(url, params=params, timeout=timeout) as resp:
                text = await resp.text()

                if resp.status == 429:
                    retry_after = resp.headers.get('Retry-After')
                    try:
                        wait_sec = int(retry_after) if retry_after else 1
                    except Exception:
                        wait_sec = 1
                    logger.warning(f"Received 429 from {url}; backing off for {wait_sec}s")
                    await asyncio.sleep(wait_sec + random.uniform(0, 0.5))
                    continue

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

        except asyncio.CancelledError:
            logger.info("Fetch cancelled")
            raise

        except Exception as e:
            logger.exception(f"Unexpected fetch error for {url}: {e}")
            if circuit_breaker:
                await circuit_breaker.record_failure()
            return None

    return None

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
        """Remove oldest 20% of cache entries when full"""
        if len(self._cache) > self._cache_max_size:
            to_remove = sorted(self._cache.keys(),
                               key=lambda k: self._cache[k][0])[:self._cache_max_size // 5]
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
        """Fetch candles with hashed cache key to avoid collisions"""
        key_str = f"{symbol}:{resolution}:{limit}"
        key_hash = f"candles:{hashlib.blake2b(key_str.encode(), digest_size=8).hexdigest()}"

        if key_hash in self._cache:
            age, data = self._cache[key_hash]
            if time.time() - age < 60:
                return data

        await asyncio.sleep(random.uniform(cfg.JITTER_MIN, cfg.JITTER_MAX))

        url = f"{self.api_base}/v2/chart/history"
        minutes = int(resolution) if resolution != 'D' else 1440
        params = {
            "resolution": resolution,
            "symbol": symbol,
            "from": int(time.time()) - (limit * minutes * 60),
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

        self._cache[key_hash] = (time.time(), data)
        self._clean_cache()
        return data

def parse_candles_result(result: Optional[dict]) -> Optional[pd.DataFrame]:
    if not result or result.get("s") != "ok" and "result" not in result:
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

        # Data Cleaning and Formatting
        df = df.sort_values('timestamp').drop_duplicates(subset='timestamp').reset_index(drop=True)

        if not df.empty:
            for col in ['open', 'high', 'low', 'close']:
                df[col] = pd.to_numeric(df[col], downcast='float', errors='coerce')
            df['volume'] = pd.to_numeric(df['volume'], downcast='float', errors='coerce')
            df['timestamp'] = pd.to_numeric(df['timestamp'], downcast='integer', errors='coerce')

            # Normalize timestamps (handle milliseconds/microseconds if present)
            try:
                max_ts = int(df['timestamp'].max())
                if max_ts > 1_000_000_000_000:
                    df['timestamp'] = (df['timestamp'] // 1000).astype(int)
                elif max_ts > 10_000_000_000:
                    df['timestamp'] = (df['timestamp'] // 1000000).astype(int)
            except Exception:
                pass

        # Drop NaN rows after conversion, especially if 'close' is affected
        df = df.dropna(subset=['close', 'timestamp']).reset_index(drop=True)

        return df

    except Exception as e:
        logger.error(f"Error parsing candles result: {e}")
        return None

def calculate_vwap(df: pd.DataFrame, length: int = 20) -> pd.Series:
    """Calculate Volume Weighted Average Price (VWAP)"""
    if len(df) < 1:
        return pd.Series([], index=df.index)
    
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    tp_volume = typical_price * df['volume']
    
    # Calculate cumulative sum for rolling period
    cum_tp_volume = tp_volume.rolling(window=length, min_periods=1).sum()
    cum_volume = df['volume'].rolling(window=length, min_periods=1).sum()
    
    vwap = (cum_tp_volume / cum_volume).fillna(df['close'])
    return vwap

def calculate_pivot_points(df: pd.DataFrame) -> Tuple[float, float, float, float, float, float, float]:
    """
    Calculate Classic Pivot Points (P, R1, R2, R3, S1, S2, S3)
    Uses the previous day's H, L, C from the daily dataframe.
    """
    if len(df) < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    
    # Get the last completed day's High, Low, Close (index -2)
    last_day = df.iloc[-2]
    h = last_day['high']
    l = last_day['low']
    c = last_day['close']
    
    P = (h + l + c) / 3
    R1 = 2 * P - l
    S1 = 2 * P - h
    R2 = P + (h - l)
    S2 = P - (h - l)
    R3 = h + 2 * (P - l)
    S3 = l - 2 * (h - P)
    
    return P, R1, R2, R3, S1, S2, S3

# -------------------------
# Indicators from TradingView PineScript Logic (Simplified)
# -------------------------

def calculate_rma(src: pd.Series, length: int) -> pd.Series:
    """
    Calculate Rolling Moving Average (RMA) as used in Pine Script.
    RMA is an EMA with a smoothing factor of 1/length.
    """
    alpha = 1.0 / length
    rma = pd.Series(np.nan, index=src.index)
    if src.empty:
        return rma

    rma.iloc[0] = src.iloc[0] # Initialize first value

    for i in range(1, len(src)):
        rma.iloc[i] = alpha * src.iloc[i] + (1 - alpha) * rma.iloc[i-1]
        
    return rma.fillna(src)

def calculate_cirrus_cloud_signals(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Calculates the 'Cloud' filters based on custom functions (smoothrng, rngfilt)
    This implementation mimics the logic for UpW/DnW signals.
    """
    close = df['close'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)

    def v1(data, len_v: int) -> pd.Series:
        if len(data) < len_v:
            return pd.Series(np.nan, index=data.index)
        data = data.rolling(window=len_v, min_periods=1).mean()
        return data.fillna(data.iloc[0])

    def smoothrng(series: pd.Series, len_v: int, len_r: int) -> pd.Series:
        rng = v1(series.diff().abs(), len_v)
        avrng = calculate_rma(rng, len_r)
        return avrng

    def rngfilt(series: pd.Series, sm_rng: pd.Series) -> pd.Series:
        filt = pd.Series(np.nan, index=series.index)
        for i in range(len(series)):
            if i == 0:
                filt.iloc[i] = series.iloc[i]
            else:
                s_rng_val = sm_rng.iloc[i]
                prev_filt = filt.iloc[i-1]
                series_val = series.iloc[i]
                
                # Pine Script: filt := series > filt[1] ? series - smrng : series < filt[1] ? series + smrng : filt[1]
                if series_val > prev_filt:
                    filt.iloc[i] = series_val - s_rng_val
                elif series_val < prev_filt:
                    filt.iloc[i] = series_val + s_rng_val
                else:
                    filt.iloc[i] = prev_filt
        return filt

    smrngx1x = smoothrng(close, cfg.X1, cfg.X2)
    smrngx1x2 = smoothrng(close, cfg.X3, cfg.X4)
    
    filtx1 = rngfilt(close, smrngx1x)
    filtx12 = rngfilt(close, smrngx1x2)
    
    # UpW/DnW are signals based on the crossover of the two filtered lines
    upw = filtx1 < filtx12
    dnw = filtx1 > filtx12
    
    return upw, dnw, filtx1, filtx12

def kalman_filter(src: pd.Series, length: int, R=0.01, Q=0.1) -> pd.Series:
    """Kalman Filter implementation for smoothing"""
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
    """
    Calculate StochRSI based logic using RMA for RSI smoothing and Kalman Filter.
    (This is a placeholder for the SRSI or similar complex indicator)
    """
    close = df['close'].astype(float)
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = calculate_rma(gain, rsi_len)
    avg_loss = calculate_rma(loss, rsi_len)
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Use Kalman filter for final smoothing, mimicking the SRSI/Kalman logic
    smooth_rsi = kalman_filter(rsi.fillna(50.0), kalman_len)
    return smooth_rsi

def calculate_ppo(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Percentage Price Oscillator (PPO)"""
    close = df['close'].astype(float)
    
    if cfg.PPO_USE_SMA:
        fast_ma = close.rolling(window=cfg.PPO_FAST, min_periods=1).mean()
        slow_ma = close.rolling(window=cfg.PPO_SLOW, min_periods=1).mean()
    else:
        fast_ma = close.ewm(span=cfg.PPO_FAST, adjust=False, min_periods=1).mean()
        slow_ma = close.ewm(span=cfg.PPO_SLOW, adjust=False, min_periods=1).mean()
        
    ppo_line = ((fast_ma - slow_ma) / slow_ma) * 100
    ppo_signal = ppo_line.ewm(span=cfg.PPO_SIGNAL, adjust=False, min_periods=1).mean()
    ppo_hist = ppo_line - ppo_signal
    
    # The original script seems to use a simpler momentum calculation for the alert condition (MMH),
    # which is often a normalized form of momentum. We'll use the calculated PPO and add a placeholder
    # for the complex MMH calculation if needed, but for now, rely on PPO.
    
    # Placeholder for the MMH style normalization logic which seems complex and unrelated to standard PPO
    # This block mimics a common Pine script normalization pattern (0 to 1 range, then scaled)
    # The original Pine script MMH is complex, involving WORM, MA, etc.
    # For this implementation, we will stick to standard PPO as the primary signal.
    
    # --- Simplified MMH/Normalized Momentum Placeholder ---
    # The actual implementation of the specific MMH (Momentum Mean Hema) is complex.
    # We will use the PPO line itself as the 'mmh_value' for the sake of completion based on
    # the snippets which rely on PPO and MMH values being present.
    # A complete, correct MMH implementation is beyond a simple code port.
    
    ppo_df = pd.DataFrame({
        'ppo': ppo_line,
        'ppo_signal': ppo_signal,
        'ppo_hist': ppo_hist,
        'mmh_value': ppo_line # Placeholder
    }, index=df.index)

    return ppo_df.replace([np.inf, -np.inf], np.nan).fillna(0)

# -------------------------
# Telegram Queue and Sender
# -------------------------
class TelegramQueue:
    def __init__(self, token: str, chat_id: str):
        self.base_url = f"https://api.telegram.org/bot{token}/"
        self.chat_id = chat_id
        self.queue: asyncio.Queue = asyncio.Queue()
        self.running_task: Optional[asyncio.Task] = None
        self.batch_size = cfg.BATCH_SIZE

    def start(self):
        if not self.running_task:
            self.running_task = asyncio.create_task(self._processor())
            logger.info("Telegram queue processor started.")

    async def stop(self):
        if self.running_task:
            self.running_task.cancel()
            try:
                await self.running_task
            except asyncio.CancelledError:
                pass
            self.running_task = None
            logger.info("Telegram queue processor stopped.")

    async def _processor(self):
        while not shutdown_event.is_set():
            try:
                # Get all available messages in a batch
                batch = []
                for _ in range(self.batch_size):
                    try:
                        message = self.queue.get_nowait()
                        batch.append(message)
                        self.queue.task_done()
                    except asyncio.QueueEmpty:
                        break

                if batch:
                    await self.send_batch(batch)

                # Wait for a short period before checking the queue again
                if not batch:
                    await asyncio.wait_for(self.queue.get(), timeout=1.0)
                
            except asyncio.TimeoutError:
                # Expected when the queue is empty
                pass
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Telegram queue processor error: {e}")
                await asyncio.sleep(1)
                
        # Process remaining messages before final stop
        while not self.queue.empty():
            batch = []
            for _ in range(self.batch_size):
                try:
                    message = self.queue.get_nowait()
                    batch.append(message)
                    self.queue.task_done()
                except asyncio.QueueEmpty:
                    break
            if batch:
                await self.send_batch(batch)
            await asyncio.sleep(0.1)

    def enqueue(self, message: str):
        self.queue.put_nowait(message)

    def escape_markdown_v2(self, text: str) -> str:
        """Escape characters in text for Telegram MarkdownV2 parsing."""
        # Escape: _ * [ ] ( ) ~ ` > # + - = | { } . !
        escape_chars = r'_*[]()~`>#+-=|{}.!'
        return re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', text)

    async def send(self, message: str) -> bool:
        """Sends a single message (or a combined batch message)."""
        url = self.base_url + "sendMessage"
        
        # Ensure message is escaped before forming the payload
        # Wrap in a try-except block to protect against malformed messages
        try:
            safe_message = self.escape_markdown_v2(message)
        except Exception as e:
            logger.error(f"Markdown escape error: {e} | Original: {message[:100]}")
            safe_message = message # Send raw if escape fails

        payload = {
            "chat_id": self.chat_id,
            "text": safe_message,
            "parse_mode": "MarkdownV2"
        }
        
        session = SessionManager.get_session()
        last_exc = None
        
        for attempt in range(1, cfg.TELEGRAM_RETRIES + 1):
            if shutdown_event.is_set():
                logger.info("Shutdown signaled, aborting Telegram send")
                return False
            
            try:
                async with session.post(url, json=payload, timeout=cfg.HTTP_TIMEOUT) as resp:
                    text = await resp.text()
                    
                    if resp.status == 200:
                        return True
                    else:
                        if resp.status in (400, 401, 403, 404):
                            # Irrecoverable errors (e.g., bad token, chat ID)
                            logger.error(f"Irrecoverable Telegram error {resp.status}: {text[:200]}")
                            return False
                        
                        logger.error(f"Telegram API FAILED (Status {resp.status}, Attempt {attempt}): {text[:200]}")
                        raise Exception(f"Telegram API error {resp.status}")
                        
            except asyncio.CancelledError:
                raise
            except Exception as e:
                last_exc = e
                logger.warning(f"Telegram send attempt {attempt} failed: {e}")
                
                if attempt < cfg.TELEGRAM_RETRIES:
                    # Exponential backoff with jitter
                    await asyncio.sleep((cfg.TELEGRAM_BACKOFF_BASE ** (attempt - 1)) + random.uniform(0, 0.3))

        logger.error(f"Telegram send failed after retries: {last_exc}")
        return False

    async def send_batch(self, messages: List[str]) -> bool:
        if not messages:
            return True
            
        max_len = 3800
        to_send = messages[:10] # Cap the number of messages in a single batch

        # Attempt to combine all messages if they fit within the limit
        combined = "\n\n".join(to_send)
        
        if len(combined) > max_len:
            # If combining fails, send them individually and await all results
            results = await asyncio.gather(*[self.send(m) for m in to_send], return_exceptions=True)
            return all(r is True for r in results)
        
        # If they fit, send as one combined message
        return await self.send(combined)

# -------------------------
# Dead Man's Switch with Startup Grace Period
# -------------------------
class DeadMansSwitch:
    def __init__(self, state_db: RedisStateStore, cooldown_seconds: int):
        self.state_db = state_db
        self.cooldown_seconds = cooldown_seconds
        self.alert_sent = False
        self.last_check_time = 0

    async def check_in(self, telegram_queue: TelegramQueue):
        now = int(time.time())
        # The first run should not trigger the DMS
        if self.last_check_time == 0:
            self.last_check_time = now
            return

        # Check if we are still within the startup grace period
        if now - self.last_check_time < cfg.STARTUP_GRACE_PERIOD:
            self.last_check_time = now
            return

        last_success_ts_str = await self.state_db.get_metadata("last_success_run")
        last_success_ts = int(last_success_ts_str) if last_success_ts_str else 0
        
        if now - last_success_ts > self.cooldown_seconds:
            # Check if an alert has already been sent recently
            if not self.alert_sent or (now - self.last_check_time) > self.cooldown_seconds:
                
                ist_now = get_current_ist_str()
                ist_last = _to_ist_str(last_success_ts)
                
                msg = telegram_queue.escape_markdown_v2(
                    f"🚨 *DEAD MAN'S SWITCH* 🚨\n\n"
                    f"The bot has *not reported a successful run* in over {self.cooldown_seconds // 3600} hours\\. "
                    f"Last successful run was at: `{ist_last}`\\. "
                    f"Current check time: `{ist_now}`\\."
                )
                
                await telegram_queue.send(msg)
                self.alert_sent = True
                self.last_check_time = now
        else:
            self.alert_sent = False
            self.last_check_time = now

# -------------------------
# Alert Keys & Messaging
# -------------------------
ALERT_KEYS = {
    # PPO
    "ppo_signal_up": "ALERT:PPO_SIG_UP",
    "ppo_signal_down": "ALERT:PPO_SIG_DOWN",
    "ppo_zero_up": "ALERT:PPO_ZERO_UP",
    "ppo_zero_down": "ALERT:PPO_ZERO_DOWN",
    "ppo_011_up": "ALERT:PPO_011_UP",
    "ppo_011_down": "ALERT:PPO_011_DOWN",
    # RMA
    "rma_50_up": "ALERT:RMA_50_UP",
    "rma_50_down": "ALERT:RMA_50_DOWN",
    "rma_200_up": "ALERT:RMA_200_UP",
    "rma_200_down": "ALERT:RMA_200_DOWN",
    # Cirrus Cloud
    "cirrus_up": "ALERT:CIRRUS_UP",
    "cirrus_down": "ALERT:CIRRUS_DOWN",
    # Srsi
    "srsi_overbought": "ALERT:SRSI_OVERBOUGHT",
    "srsi_oversold": "ALERT:SRSI_OVERSOLD",
    # VWAP
    "vwap_up": "ALERT:VWAP_UP",
    "vwap_down": "ALERT:VWAP_DOWN",
    # Pivots
    "pivot_up_P": "ALERT:PIVOT_UP_P",
    "pivot_up_S1": "ALERT:PIVOT_UP_S1",
    "pivot_up_S2": "ALERT:PIVOT_UP_S2",
    "pivot_up_S3": "ALERT:PIVOT_UP_S3",
    "pivot_up_R1": "ALERT:PIVOT_UP_R1",
    "pivot_up_R2": "ALERT:PIVOT_UP_R2",
    "pivot_up_R3": "ALERT:PIVOT_UP_R3",
    "pivot_down_P": "ALERT:PIVOT_DOWN_P",
    "pivot_down_S1": "ALERT:PIVOT_DOWN_S1",
    "pivot_down_S2": "ALERT:PIVOT_DOWN_S2",
    "pivot_down_S3": "ALERT:PIVOT_DOWN_S3",
    "pivot_down_R1": "ALERT:PIVOT_DOWN_R1",
    "pivot_down_R2": "ALERT:PIVOT_DOWN_R2",
    "pivot_down_R3": "ALERT:PIVOT_DOWN_R3",
    # MMH (Complex normalization, using PPO for now)
    "mmh_buy": "ALERT:MMH_BUY",
    "mmh_sell": "ALERT:MMH_SELL",
}

# -------------------------
# Helpers for alert state
# -------------------------
async def set_alert_state(sdb: RedisStateStore, pair: str, key: str, active: bool):
    state_key = f"{pair}:{key}"
    ts = int(time.time())
    state_val = "ACTIVE" if active else "INACTIVE"
    await sdb.set(state_key, state_val, ts)

async def was_alert_active(sdb: RedisStateStore, pair: str, key: str) -> bool:
    state_key = f"{pair}:{key}"
    st = await sdb.get(state_key)
    return (st and st.get("state") == "ACTIVE")

async def reset_alert_on_condition(sdb: RedisStateStore, pair: str, key: str, condition: bool):
    """Resets the alert state (sets to INACTIVE) if the condition is met."""
    if condition:
        await set_alert_state(sdb, pair, key, False)

def format_time_ist(ts: int) -> str:
    """Return timestamp as 'DD-MM-YYYY' + 5 spaces + 'HH:MM IST'"""
    try:
        dt = datetime.fromtimestamp(int(ts), tz=timezone.utc).astimezone(ZoneInfo("Asia/Kolkata"))
        date_part = dt.strftime('%d-%m-%Y')
        time_part = dt.strftime('%H:%M IST')
        return f"{date_part} {time_part}"
    except Exception:
        # Fallback to simple formatting
        try:
            dt = datetime.fromtimestamp(int(ts))
            return dt.strftime('%d-%m-%Y     %H:%M IST')
        except Exception:
            return "N/A"

def build_msg(title: str, pair: str, close_price: float, timestamp: int, details: str = "") -> str:
    """Formats the final Telegram message with MarkdownV2 style."""
    msg = (
        f"*{title}* {pair}\n"
        f"`Price: ${close_price:,.4f}`\n"
        f"`Details: {details}`\n"
        f"`Time: {format_time_ist(timestamp)}`"
    )
    return msg

def check_common_candle_condition(df: pd.DataFrame, is_buy: bool) -> bool:
    """
    Check the common candle condition (no opposing wick, closed higher/lower).
    This function expects the last (current) candle to be at index -1.
    """
    if len(df) < 1:
        return False
    
    try:
        current_candle = df.iloc[-1]
        o, h, l, c = current_candle['open'], current_candle['high'], current_candle['low'], current_candle['close']
        rng = max(h - l, 1e-8)
        
        if is_buy:
            # Must close higher than open (green candle)
            if c <= o:
                return False
            # Check for small upper wick (upper wick < MIN_WICK_RATIO * range)
            upper_wick = h - max(o, c)
            return upper_wick < Constants.MIN_WICK_RATIO * rng
        else: # is_sell
            # Must close lower than open (red candle)
            if c >= o:
                return False
            # Check for small lower wick (lower wick < MIN_WICK_RATIO * range)
            lower_wick = min(o, c) - l
            return lower_wick < Constants.MIN_WICK_RATIO * rng
    
    except Exception as e:
        logger.error(f"Error in common candle check: {e}")
        return False

# -------------------------
# Evaluate per pair and send alerts
# -------------------------
async def evaluate_pair_and_alert(
    pair_name: str,
    df_15m: pd.DataFrame,
    df_5m: pd.DataFrame,
    df_daily: Optional[pd.DataFrame],
    sdb: RedisStateStore,
    telegram_queue: TelegramQueue
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Evaluates indicators for a single pair and generates alerts.
    Returns: Tuple of (pair_name, {'state': 'SIGNAL'|'NO_SIGNAL', 'close': price, ...})
    """
    
    if len(df_15m) < 2 or len(df_5m) < 2:
        return pair_name, {'state': 'INSUFFICIENT_DATA', 'close': df_15m.iloc[-1]['close'] if len(df_15m)>0 else None}

    messages: List[str] = []
    ts_curr = int(df_15m.iloc[-1]['timestamp'])
    
    # Get last two 15m candles for cross checks
    close_curr = df_15m.iloc[-1]['close']
    close_prev = df_15m.iloc[-2]['close']
    
    ppo_df = calculate_ppo(df_15m)
    ppo_curr = ppo_df.iloc[-1]['ppo']
    ppo_prev = ppo_df.iloc[-2]['ppo']
    ppo_sig_curr = ppo_df.iloc[-1]['ppo_signal']
    ppo_sig_prev = ppo_df.iloc[-2]['ppo_signal']
    mmh_curr = ppo_df.iloc[-1]['mmh_value'] # Using PPO as placeholder
    
    # ------------------
    # 1. PPO/MMH Alerts
    # ------------------
    
    buy_common = check_common_candle_condition(df_15m, is_buy=True)
    sell_common = check_common_candle_condition(df_15m, is_buy=False)
    
    # PPO/Signal Cross (Momentum Shift)
    if (ppo_prev <= ppo_sig_prev) and (ppo_curr > ppo_sig_curr):
        key = ALERT_KEYS["ppo_signal_up"]
        if not await was_alert_active(sdb, pair_name, key):
            msg = build_msg("🟢 PPO Signal Cross UP", pair_name, close_curr, ts_curr, f"PPO {ppo_curr:.2f} > Sig {ppo_sig_curr:.2f}")
            messages.append(msg); await set_alert_state(sdb, pair_name, key, True)

    if (ppo_prev >= ppo_sig_prev) and (ppo_curr < ppo_sig_curr):
        key = ALERT_KEYS["ppo_signal_down"]
        if not await was_alert_active(sdb, pair_name, key):
            msg = build_msg("🔴 PPO Signal Cross DOWN", pair_name, close_curr, ts_curr, f"PPO {ppo_curr:.2f} < Sig {ppo_sig_curr:.2f}")
            messages.append(msg); await set_alert_state(sdb, pair_name, key, True)

    # PPO Zero Cross
    if (ppo_prev <= 0) and (ppo_curr > 0):
        key = ALERT_KEYS["ppo_zero_up"]
        if not await was_alert_active(sdb, pair_name, key):
            msg = build_msg("🟡 PPO Zero Cross UP", pair_name, close_curr, ts_curr, f"PPO {ppo_curr:.2f} | MMH ({mmh_curr:.2f})")
            messages.append(msg); await set_alert_state(sdb, pair_name, key, True)

    if (ppo_prev >= 0) and (ppo_curr < 0):
        key = ALERT_KEYS["ppo_zero_down"]
        if not await was_alert_active(sdb, pair_name, key):
            msg = build_msg("⚫️ PPO Zero Cross DOWN", pair_name, close_curr, ts_curr, f"PPO {ppo_curr:.2f} | MMH ({mmh_curr:.2f})")
            messages.append(msg); await set_alert_state(sdb, pair_name, key, True)

    # PPO 0.11 Thresholds
    if (ppo_prev <= Constants.PPO_011_THRESHOLD) and (ppo_curr > Constants.PPO_011_THRESHOLD):
        key = ALERT_KEYS["ppo_011_up"]
        if not await was_alert_active(sdb, pair_name, key):
            msg = build_msg("🟢 PPO cross above +0.11", pair_name, close_curr, ts_curr, f"PPO {ppo_curr:.2f} | MMH ({mmh_curr:.2f})")
            messages.append(msg); await set_alert_state(sdb, pair_name, key, True)

    if (ppo_prev >= Constants.PPO_011_THRESHOLD_SELL) and (ppo_curr < Constants.PPO_011_THRESHOLD_SELL):
        key = ALERT_KEYS["ppo_011_down"]
        if not await was_alert_active(sdb, pair_name, key):
            msg = build_msg("🔴 PPO cross below -0.11", pair_name, close_curr, ts_curr, f"PPO {ppo_curr:.2f} | MMH ({mmh_curr:.2f})")
            messages.append(msg); await set_alert_state(sdb, pair_name, key, True)

    # Reset Conditions for PPO
    await reset_alert_on_condition(sdb, pair_name, ALERT_KEYS["ppo_signal_up"], ppo_prev > ppo_sig_prev and ppo_curr <= ppo_sig_curr)
    await reset_alert_on_condition(sdb, pair_name, ALERT_KEYS["ppo_signal_down"], ppo_prev < ppo_sig_prev and ppo_curr >= ppo_sig_curr)
    await reset_alert_on_condition(sdb, pair_name, ALERT_KEYS["ppo_zero_up"], ppo_prev > 0 and ppo_curr <= 0)
    await reset_alert_on_condition(sdb, pair_name, ALERT_KEYS["ppo_zero_down"], ppo_prev < 0 and ppo_curr >= 0)
    await reset_alert_on_condition(sdb, pair_name, ALERT_KEYS["ppo_011_up"], ppo_prev > Constants.PPO_011_THRESHOLD and ppo_curr <= Constants.PPO_011_THRESHOLD)
    await reset_alert_on_condition(sdb, pair_name, ALERT_KEYS["ppo_011_down"], ppo_prev < Constants.PPO_011_THRESHOLD_SELL and ppo_curr >= Constants.PPO_011_THRESHOLD_SELL)

    # MMH Simple Alerts (Using PPO for simplicity, requires common candle check)
    if buy_common and (ppo_curr > Constants.PPO_THRESHOLD_BUY):
        key = ALERT_KEYS["mmh_buy"]
        if not await was_alert_active(sdb, pair_name, key):
            msg = build_msg("🟢 MMH/PPO Buy", pair_name, close_curr, ts_curr, f"PPO {ppo_curr:.2f} > {Constants.PPO_THRESHOLD_BUY:.2f}")
            messages.append(msg); await set_alert_state(sdb, pair_name, key, True)
    
    if sell_common and (ppo_curr < Constants.PPO_THRESHOLD_SELL):
        key = ALERT_KEYS["mmh_sell"]
        if not await was_alert_active(sdb, pair_name, key):
            msg = build_msg("🔴 MMH/PPO Sell", pair_name, close_curr, ts_curr, f"PPO {ppo_curr:.2f} < {Constants.PPO_THRESHOLD_SELL:.2f}")
            messages.append(msg); await set_alert_state(sdb, pair_name, key, True)
        
    await reset_alert_on_condition(sdb, pair_name, ALERT_KEYS["mmh_buy"], ppo_curr < Constants.PPO_THRESHOLD_BUY * 0.9)
    await reset_alert_on_condition(sdb, pair_name, ALERT_KEYS["mmh_sell"], ppo_curr > Constants.PPO_THRESHOLD_SELL * 0.9)

    # ------------------
    # 2. RMA Alerts (50 & 200 periods)
    # ------------------
    rma_50 = calculate_rma(df_15m['close'], cfg.RMA_50_PERIOD)
    rma_200 = calculate_rma(df_15m['close'], cfg.RMA_200_PERIOD)
    
    rma_50_curr = rma_50.iloc[-1]
    rma_50_prev = rma_50.iloc[-2]
    rma_200_curr = rma_200.iloc[-1]
    rma_200_prev = rma_200.iloc[-2]
    
    # 50 RMA Cross
    if (close_prev <= rma_50_prev) and (close_curr > rma_50_curr):
        key = ALERT_KEYS["rma_50_up"]
        if not await was_alert_active(sdb, pair_name, key):
            msg = build_msg("📈 50 RMA Cross UP", pair_name, close_curr, ts_curr, f"RMA: ${rma_50_curr:,.4f}")
            messages.append(msg); await set_alert_state(sdb, pair_name, key, True)

    if (close_prev >= rma_50_prev) and (close_curr < rma_50_curr):
        key = ALERT_KEYS["rma_50_down"]
        if not await was_alert_active(sdb, pair_name, key):
            msg = build_msg("📉 50 RMA Cross DOWN", pair_name, close_curr, ts_curr, f"RMA: ${rma_50_curr:,.4f}")
            messages.append(msg); await set_alert_state(sdb, pair_name, key, True)

    # 200 RMA Cross
    if (close_prev <= rma_200_prev) and (close_curr > rma_200_curr):
        key = ALERT_KEYS["rma_200_up"]
        if not await was_alert_active(sdb, pair_name, key):
            msg = build_msg("🚀 200 RMA Cross UP", pair_name, close_curr, ts_curr, f"RMA: ${rma_200_curr:,.4f}")
            messages.append(msg); await set_alert_state(sdb, pair_name, key, True)

    if (close_prev >= rma_200_prev) and (close_curr < rma_200_curr):
        key = ALERT_KEYS["rma_200_down"]
        if not await was_alert_active(sdb, pair_name, key):
            msg = build_msg(" سقوط 200 RMA Cross DOWN", pair_name, close_curr, ts_curr, f"RMA: ${rma_200_curr:,.4f}")
            messages.append(msg); await set_alert_state(sdb, pair_name, key, True)
    
    # Reset Conditions for RMA
    await reset_alert_on_condition(sdb, pair_name, ALERT_KEYS["rma_50_up"], close_prev > rma_50_prev and close_curr <= rma_50_curr)
    await reset_alert_on_condition(sdb, pair_name, ALERT_KEYS["rma_50_down"], close_prev < rma_50_prev and close_curr >= rma_50_curr)
    await reset_alert_on_condition(sdb, pair_name, ALERT_KEYS["rma_200_up"], close_prev > rma_200_prev and close_curr <= rma_200_curr)
    await reset_alert_on_condition(sdb, pair_name, ALERT_KEYS["rma_200_down"], close_prev < rma_200_prev and close_curr >= rma_200_curr)
    
    # ------------------
    # 3. Cirrus Cloud Alerts
    # ------------------
    if cfg.CIRRUS_CLOUD_ENABLED and len(df_15m) > max(cfg.X1, cfg.X3) + cfg.X4 + 2:
        upw, dnw, filtx1, filtx12 = calculate_cirrus_cloud_signals(df_15m)
        
        upw_curr = upw.iloc[-1]
        upw_prev = upw.iloc[-2]
        dnw_curr = dnw.iloc[-1]
        dnw_prev = dnw.iloc[-2]
        
        # Cloud Buy Signal: UpW (Fast line below slow, then crosses up)
        if (not upw_prev and upw_curr) and dnw_prev and (not dnw_curr): # dnw_prev: fast > slow, dnw_curr: fast <= slow
            key = ALERT_KEYS["cirrus_up"]
            if not await was_alert_active(sdb, pair_name, key):
                msg = build_msg("☁️ Cirrus Cloud Cross UP", pair_name, close_curr, ts_curr, "Cloud filter cross")
                messages.append(msg); await set_alert_state(sdb, pair_name, key, True)
        
        # Cloud Sell Signal: DnW (Fast line above slow, then crosses down)
        if (upw_prev and not upw_curr) and (not dnw_prev and dnw_curr): # upw_prev: fast < slow, upw_curr: fast >= slow
            key = ALERT_KEYS["cirrus_down"]
            if not await was_alert_active(sdb, pair_name, key):
                msg = build_msg("⛈️ Cirrus Cloud Cross DOWN", pair_name, close_curr, ts_curr, "Cloud filter cross")
                messages.append(msg); await set_alert_state(sdb, pair_name, key, True)
        
        # Reset Conditions for Cirrus (Reset when price crosses back into the cloud or logic reverses)
        # Note: The reset logic for complex indicators like Cirrus is simplified here.
        # It should ideally be reset when the trend clearly reverses.
        await reset_alert_on_condition(sdb, pair_name, ALERT_KEYS["cirrus_up"], dnw_curr and not upw_curr)
        await reset_alert_on_condition(sdb, pair_name, ALERT_KEYS["cirrus_down"], upw_curr and not dnw_curr)

    # ------------------
    # 4. Stochastic RSI Alerts (using SmoothRSI approximation)
    # ------------------
    if len(df_15m) > cfg.SRSI_RSI_LEN + cfg.SRSI_KALMAN_LEN:
        srsi = calculate_smooth_rsi(df_15m, cfg.SRSI_RSI_LEN, cfg.SRSI_KALMAN_LEN)
        srsi_curr = srsi.iloc[-1]
        srsi_prev = srsi.iloc[-2]
        
        # Oversold (Crossing below 20) - Should typically be an *exit* condition for short or *entry* for long
        if srsi_prev >= 20 and srsi_curr < 20:
            key = ALERT_KEYS["srsi_oversold"]
            if not await was_alert_active(sdb, pair_name, key):
                msg = build_msg("🧊 SRSI Oversold", pair_name, close_curr, ts_curr, f"SRSI {srsi_curr:.2f}")
                messages.append(msg); await set_alert_state(sdb, pair_name, key, True)

        # Overbought (Crossing above 80) - Should typically be an *exit* condition for long or *entry* for short
        if srsi_prev <= 80 and srsi_curr > 80:
            key = ALERT_KEYS["srsi_overbought"]
            if not await was_alert_active(sdb, pair_name, key):
                msg = build_msg("🔥 SRSI Overbought", pair_name, close_curr, ts_curr, f"SRSI {srsi_curr:.2f}")
                messages.append(msg); await set_alert_state(sdb, pair_name, key, True)

        # Reset conditions (reset when it crosses back)
        await reset_alert_on_condition(sdb, pair_name, ALERT_KEYS["srsi_oversold"], srsi_curr > 20)
        await reset_alert_on_condition(sdb, pair_name, ALERT_KEYS["srsi_overbought"], srsi_curr < 80)

    # ------------------
    # 5. VWAP Alerts
    # ------------------
    if cfg.ENABLE_VWAP:
        vwap = calculate_vwap(df_15m, length=12) # Use a typical period
        vwap_curr = vwap.iloc[-1]
        vwap_prev = vwap.iloc[-2]
        
        if (close_prev <= vwap_prev) and (close_curr > vwap_curr):
            key = ALERT_KEYS["vwap_up"]
            if not await was_alert_active(sdb, pair_name, key):
                msg = build_msg("⏫ VWAP Cross UP", pair_name, close_curr, ts_curr, f"VWAP: ${vwap_curr:,.4f}")
                messages.append(msg); await set_alert_state(sdb, pair_name, key, True)

        if (close_prev >= vwap_prev) and (close_curr < vwap_curr):
            key = ALERT_KEYS["vwap_down"]
            if not await was_alert_active(sdb, pair_name, key):
                msg = build_msg("⏬ VWAP Cross DOWN", pair_name, close_curr, ts_curr, f"VWAP: ${vwap_curr:,.4f}")
                messages.append(msg); await set_alert_state(sdb, pair_name, key, True)
            
        await reset_alert_on_condition(sdb, pair_name, ALERT_KEYS["vwap_up"], close_prev > vwap_prev and close_curr <= vwap_curr)
        await reset_alert_on_condition(sdb, pair_name, ALERT_KEYS["vwap_down"], close_prev < vwap_prev and close_curr >= vwap_curr)

    # ------------------
    # 6. Pivot Point Alerts (Daily)
    # ------------------
    if cfg.ENABLE_PIVOT and df_daily is not None and len(df_daily) >= cfg.PIVOT_LOOKBACK_PERIOD + 2:
        P, R1, R2, R3, S1, S2, S3 = calculate_pivot_points(df_daily)

        buy_levels = [("P", P), ("R1", R1), ("R2", R2), ("R3", R3)]
        for name, level in buy_levels:
            if buy_common and (close_prev <= level) and (close_curr > level):
                key = ALERT_KEYS[f"pivot_up_{name}"]
                if not await was_alert_active(sdb, pair_name, key):
                    msg = build_msg(f"🔷 Cross above {name}", pair_name, close_curr, ts_curr, f"${level:,.2f}")
                    messages.append(msg); await set_alert_state(sdb, pair_name, key, True)
            await reset_alert_on_condition(sdb, pair_name, ALERT_KEYS[f"pivot_up_{name}"], close_prev > level and close_curr <= level)

        sell_levels = [("P", P), ("S1", S1), ("S2", S2), ("S3", S3)]
        for name, level in sell_levels:
            if sell_common and (close_prev >= level) and (close_curr < level):
                key = ALERT_KEYS[f"pivot_down_{name}"]
                if not await was_alert_active(sdb, pair_name, key):
                    msg = build_msg(f"🔶 Cross below {name}", pair_name, close_curr, ts_curr, f"${level:,.2f}")
                    messages.append(msg); await set_alert_state(sdb, pair_name, key, True)
            await reset_alert_on_condition(sdb, pair_name, ALERT_KEYS[f"pivot_down_{name}"], close_prev < level and close_curr >= level)

    
    # ------------------
    # Final Action
    # ------------------
    if messages:
        # Enqueue all generated alerts
        for msg in messages:
            telegram_queue.enqueue(msg)
        
        logger.info(f"Generated {len(messages)} alerts for {pair_name}")
        return pair_name, {'state': 'SIGNAL', 'count': len(messages), 'close': close_curr}
    else:
        return pair_name, {'state': 'NO_SIGNAL', 'close': close_curr}

async def fetch_and_process_pair(
    pair_name: str,
    fetcher: DataFetcher,
    products_map: Dict[str, dict],
    state_db: RedisStateStore,
    telegram_queue: TelegramQueue
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Handles data fetching, technical analysis, and alerting for a single trading pair.
    """
    try:
        if shutdown_event.is_set():
            return None
            
        product_info = products_map.get(pair_name)
        if not product_info:
            logger.warning(f"Product info missing for {pair_name}. Skipping.")
            return None
            
        symbol = product_info["symbol"]
        special_limits = cfg.SPECIAL_PAIRS.get(pair_name, {})
        limit_15m = special_limits.get("limit_15m", 250)
        limit_5m = special_limits.get("limit_5m", 350)
        min_required = special_limits.get("min_required", 200)
        min_required_5m = special_limits.get("min_required_5m", 250)
        daily_limit = cfg.PIVOT_LOOKBACK_PERIOD + 2

        # Fetch 15m, 5m, and Daily candles concurrently
        tasks = [
            fetcher.fetch_candles(symbol, "15", limit_15m),
            fetcher.fetch_candles(symbol, "5", limit_5m),
            fetcher.fetch_candles(symbol, "D", daily_limit) if cfg.ENABLE_PIVOT else asyncio.Future()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)

        df_15m = parse_candles_result(results[0]) if not isinstance(results[0], Exception) else None
        df_5m = parse_candles_result(results[1]) if not isinstance(results[1], Exception) else None
        df_daily = parse_candles_result(results[2]) if cfg.ENABLE_PIVOT and not isinstance(results[2], Exception) else None
        
        for res in results:
            if isinstance(res, Exception):
                logger.error(f"Error fetching data for {pair_name}: {res}")
        
        if df_15m is None or len(df_15m) < (min_required + 2):
            logger.warning(f"Skipping {pair_name}: Insufficient 15m data (Count: {len(df_15m) if df_15m is not None else 0}, Required: {min_required+2})")
            return pair_name, {'state': 'DATA_TOO_SHORT', 'close': df_15m.iloc[-1]['close'] if df_15m is not None and not df_15m.empty else None}

        # Check 5m data requirement
        if df_5m is None or len(df_5m) < (min_required_5m + 2):
            logger.warning(f"5m data for {pair_name} is short (Count: {len(df_5m) if df_5m is not None else 0}, Required: {min_required_5m+2})")

        return await evaluate_pair_and_alert(pair_name, df_15m, df_5m, df_daily, state_db, telegram_queue)

    except asyncio.CancelledError:
        logger.info(f"Processing for {pair_name} cancelled")
        raise
    except Exception as e:
        logger.exception(f"Error processing {pair_name}: {e}")
        return pair_name, {'state': 'ERROR', 'error': str(e)}

async def process_batch(batch_pairs: List[str], fetcher: DataFetcher, products_map: Dict[str, dict], state_db: RedisStateStore, telegram_queue: TelegramQueue):
    """Process a batch of pairs concurrently."""
    tasks = [
        fetch_and_process_pair(pair, fetcher, products_map, state_db, telegram_queue)
        for pair in batch_pairs
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    valid_results = [r for r in results if r is not None and not isinstance(r, Exception)]
    results_map = {pair: data for pair, data in valid_results}

    # Health Tracking update
    if cfg.ENABLE_HEALTH_TRACKER and health_tracker:
        try:
            for pair_name in batch_pairs:
                success = pair_name in results_map and results_map[pair_name].get('state') != 'NO_SIGNAL'
                await health_tracker.record_pair_result(pair_name, bool(success), {'source': 'process_batch'})
            
            summary = {'checked': len(batch_pairs), 'succeeded': len(valid_results)}
            await health_tracker.record_overall(summary)
        except Exception as e:
            logger.debug(f'HealthTracker error: {e}')
            
    return valid_results

# -------------------------
# Run once (single cron execution)
# -------------------------
async def run_once() -> bool:
    try:
        start_time = time.time()
        process = psutil.Process(os.getpid())
        
        # Initial memory check
        mem_usage = process.memory_info().rss
        if mem_usage > cfg.MEMORY_LIMIT_BYTES:
            logger.critical(f"FATAL: Memory limit exceeded ({mem_usage / 1024 / 1024:.2f}MB). Aborting run.")
            return False

        fetcher = DataFetcher(cfg.DELTA_API_BASE)
        telegram_queue = TelegramQueue(cfg.TELEGRAM_BOT_TOKEN, cfg.TELEGRAM_CHAT_ID)
        
        if await fetcher.circuit_breaker.is_open():
            logger.critical("Circuit breaker OPEN. Aborting run.")
            await telegram_queue.send("🚨 Circuit breaker open - check API status")
            return False

        try:
            async with RedisStateStore(cfg.REDIS_URL) as sdb:
                global health_tracker
                health_tracker = HealthTracker(sdb)
                lock = RedisLock(sdb._redis, "run_lock", expire=cfg.RUN_TIMEOUT_SECONDS + 60)
                
                # Try to acquire a distributed lock
                if not await lock.acquire():
                    logger.warning(f"Could not acquire run lock. Another process is running (Lock Timeout: {cfg.RUN_TIMEOUT_SECONDS}s). Aborting.")
                    return False
                
                # Dead Man's Switch check
                dms = DeadMansSwitch(sdb, cfg.DEAD_MANS_COOLDOWN_SECONDS)
                await dms.check_in(telegram_queue)
                
                # Fetch product list first
                products = await fetcher.fetch_products()
                if not products:
                    logger.critical("Failed to fetch product list. Aborting run.")
                    await lock.release()
                    return False
                    
                products_map = build_products_map_from_api_result(products)

                if cfg.SEND_TEST_MESSAGE:
                    # Send test message only on first run or if explicitly enabled
                    last_test_str = await sdb.get_metadata("last_test_message_ts")
                    if not last_test_str or (time.time() - int(last_test_str) > 86400):
                        msg = build_msg(f"✅ Bot Started ({cfg.BOT_NAME} v{__version__})", "TEST-RUN", 0.0, int(time.time()), f"Pairs: {len(cfg.PAIRS)}")
                        telegram_queue.enqueue(msg)
                        await sdb.set_metadata("last_test_message_ts", str(int(time.time())))
                        logger.info("Test message sent.")

                # Start the Telegram message processor
                telegram_queue.start()

                # Process all pairs in batches
                pairs_to_process = cfg.PAIRS
                batch_size = cfg.BATCH_SIZE
                total_alerts = 0
                
                for i in range(0, len(pairs_to_process), batch_size):
                    batch = pairs_to_process[i:i + batch_size]
                    logger.info(f"Processing batch {i//batch_size + 1}/{len(pairs_to_process)//batch_size + 1}: {batch}")
                    
                    batch_results = await process_batch(batch, fetcher, products_map, sdb, telegram_queue)
                    
                    total_alerts += sum(r[1].get('count', 0) for r in batch_results if r and r[1].get('state') == 'SIGNAL')
                    
                    # Periodic checks and lock refresh
                    if time.time() - start_time > cfg.RUN_TIMEOUT_SECONDS:
                        logger.warning(f"Run timeout exceeded ({cfg.RUN_TIMEOUT_SECONDS}s). Aborting remaining batches.")
                        break

                    # Memory check
                    mem_usage = process.memory_info().rss
                    mem_limit_ratio = mem_usage / cfg.MEMORY_LIMIT_BYTES
                    if mem_limit_ratio > 0.9:
                        logger.critical(f"Memory critical: {mem_usage/1024/1024:.2f}MB")
                        await telegram_queue.send("🚨 Memory limit near - aborting run")
                        break
                    
                    # Extend the lock and wait briefly between batches
                    await lock.extend()
                    if i + batch_size < len(pairs_to_process):
                        await asyncio.sleep(1) # Small pause to allow other tasks/I/O

                # Wait for all Telegram messages to be sent
                await telegram_queue.queue.join() 
                await telegram_queue.stop()
                
                # Final lock and metadata update
                await sdb.set_metadata("last_success_run", str(int(time.time())))
                await lock.release()
                
                run_duration = time.time() - start_time
                logger.info(f"✅ Run finished successfully in {run_duration:.2f}s. Total Alerts: {total_alerts}")
                
                return True

        except RedisConnectionError:
            logger.critical("Failed to connect to Redis. Aborting run.")
            await telegram_queue.send("🚨 FATAL: Redis connection failed\\. Aborting run\\.")
            return False
            
        except asyncio.CancelledError:
            logger.info("Run cancelled")
            return False
            
        except Exception as e:
            logger.exception(f"run_once unhandled error: {e}")
            return False

    finally:
        try:
            # Ensure the HTTP session is closed regardless of success/failure
            await SessionManager.close_session()
        except Exception:
            pass

def validate_config_runtime():
    if cfg.RUN_TIMEOUT_SECONDS <= 0:
        raise ValueError("RUN_TIMEOUT_SECONDS must be positive")
    if cfg.BATCH_SIZE > len(cfg.PAIRS):
        logger.warning("BATCH_SIZE larger than PAIRS count")
    if cfg.MEMORY_LIMIT_BYTES < 100_000_000:
        logger.warning("MEMORY_LIMIT_BYTES seems very low")
    logger.info(f"✅ Runtime config validation passed for bot v{__version__}")

# -------------------------
# Main entrypoint
# -------------------------
if __name__ == "__main__":
    print(
        "🚀 Unified Trading Bot Started\n"
        f"Time: {get_current_ist_str()}\n"
        f"Pairs: {len(cfg.PAIRS)} | Debug: {cfg.DEBUG_MODE} | Version: {__version__}"
    )

    try:
        validate_config_runtime()

        success = asyncio.run(run_once())
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n👋 Bot stopped manually.")
        sys.exit(0)
    except Exception as main_e:
        logger.critical(f"Main execution failed: {main_e}")
        sys.exit(1)
