from __future__ import annotations
import os
import sys
import time
import asyncio
import random
import logging
from pathlib import Path
import ssl
import signal
import re
import uuid
import argparse
import psutil
from typing import Dict, Any, Optional, Tuple, List, ClassVar, TypedDict, Callable
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from contextvars import ContextVar
import gc
import aiohttp
import numpy as np
import redis.asyncio as redis
from redis.exceptions import ConnectionError as RedisConnectionError, RedisError
from pydantic import BaseModel, Field, field_validator, model_validator
from aiohttp import ClientConnectorError, ClientResponseError, TCPConnector, ClientError
from numba import njit

# ============================================================================
# PERFORMANCE ENHANCEMENT: Use orjson for faster JSON operations
# ============================================================================
try:
    import orjson

    def json_dumps(obj: Any) -> str:
        return orjson.dumps(
            obj,
            option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS
        ).decode('utf-8')

    def json_loads(s: str | bytes) -> Any:
        return orjson.loads(s)

    JSON_BACKEND = "orjson"
except ImportError:
    import json
    json_dumps = json.dumps
    json_loads = json.loads
    JSON_BACKEND = "stdlib"

__version__ = "1.3.0-cron-optimized-numpy-only"

class Constants:
    MIN_WICK_RATIO = 0.2
    PPO_THRESHOLD_BUY = 0.20
    PPO_THRESHOLD_SELL = -0.20
    RSI_THRESHOLD = 50
    PPO_RSI_GUARD_BUY = 0.30
    PPO_RSI_GUARD_SELL = -0.30
    PPO_011_THRESHOLD = 0.11
    PPO_011_THRESHOLD_SELL = -0.11
    STARTUP_GRACE_PERIOD = int(os.getenv('STARTUP_GRACE_PERIOD', 300))
    REDIS_LOCK_EXPIRY = max(int(os.getenv('REDIS_LOCK_EXPIRY', 900)), 900)
    CIRCUIT_BREAKER_MAX_WAIT = 300
    MAX_PRICE_CHANGE_PERCENT = 50.0
    MAX_CANDLE_GAP_MULTIPLIER = 2.0
    LOCK_EXTEND_INTERVAL = 540
    LOCK_EXTEND_JITTER_MAX = 120
    ALERT_DEDUP_WINDOW_SEC = int(os.getenv("ALERT_DEDUP_WINDOW_SEC", 840))
    CANDLE_PUBLICATION_LAG_SEC = int(os.getenv("CANDLE_PUBLICATION_LAG_SEC", 45))

TRACE_ID: ContextVar[str] = ContextVar("trace_id", default="")
PAIR_ID: ContextVar[str] = ContextVar("pair_id", default="")

def format_ist_time(dt_or_ts: Any = None, fmt: str = "%Y-%m-%d %H:%M:%S IST") -> str:
    try:
        if dt_or_ts is None:
            dt = datetime.now(timezone.utc)
        elif isinstance(dt_or_ts, (int, float)):
            dt = datetime.fromtimestamp(float(dt_or_ts), tz=timezone.utc)
        elif isinstance(dt_or_ts, datetime):
            dt = dt_or_ts
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = datetime.fromisoformat(str(dt_or_ts))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        ist = dt.astimezone(ZoneInfo("Asia/Kolkata"))
        return ist.strftime(fmt)
    except Exception:
        try:
            ts = float(dt_or_ts)
            ist = datetime.fromtimestamp(ts, tz=ZoneInfo("Asia/Kolkata"))
            return ist.strftime(fmt)
        except Exception:
            return str(dt_or_ts)

class BotConfig(BaseModel):
    TELEGRAM_BOT_TOKEN: str = Field(..., min_length=1)
    TELEGRAM_CHAT_ID: str = Field(..., min_length=1)
    REDIS_URL: str = Field(..., min_length=1)
    DELTA_API_BASE: str = Field(..., min_length=1)
    DEBUG_MODE: bool = Field(default=False, env='DEBUG_MODE')
    SEND_TEST_MESSAGE: bool = True
    BOT_NAME: str = "Unified Alert Bot"
    PAIRS: List[str] = Field(default=["BTCUSD", "ETHUSD", "AVAXUSD", "BCHUSD", "XRPUSD", "BNBUSD", "LTCUSD", "DOTUSD", "ADAUSD", "SUIUSD", "AAVEUSD", "SOLUSD"], min_length=1)
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
    LOG_FILE: str = "macd_bot.log"
    MAX_PARALLEL_FETCH: int = Field(8, ge=1, le=16)
    HTTP_TIMEOUT: int = 15
    CANDLE_FETCH_RETRIES: int = 3
    CANDLE_FETCH_BACKOFF: float = 1.5
    JITTER_MIN: float = 0.1
    JITTER_MAX: float = 0.8
    RUN_TIMEOUT_SECONDS: int = 600
    BATCH_SIZE: int = 4
    TCP_CONN_LIMIT: int = 8
    TCP_CONN_LIMIT_PER_HOST: int = 10
    TELEGRAM_RETRIES: int = 3
    TELEGRAM_BACKOFF_BASE: float = 2.0
    MEMORY_LIMIT_BYTES: int = 400_000_000
    STATE_EXPIRY_DAYS: int = 30
    LOG_LEVEL: str = "INFO"
    ENABLE_VWAP: bool = True
    ENABLE_PIVOT: bool = True
    PIVOT_LOOKBACK_PERIOD: int = 15
    FAIL_ON_REDIS_DOWN: bool = False
    FAIL_ON_TELEGRAM_DOWN: bool = False
    TELEGRAM_RATE_LIMIT_PER_MINUTE: int = 20
    TELEGRAM_BURST_SIZE: int = 5
    REDIS_CONNECTION_RETRIES: int = 3
    REDIS_RETRY_DELAY: float = 2.0
    INDICATOR_THREAD_LIMIT: int = 3
    DRY_RUN_MODE: bool = Field(default=False, description="Dry-run: log alerts without sending")
    MIN_RUN_TIMEOUT: int = Field(default=300, ge=300, le=1800, description="Min/max run timeout bounds")
    MAX_ALERTS_PER_PAIR: int = Field(default=8, ge=5, le=15, description="Max alerts per pair per run")

    @field_validator('TELEGRAM_BOT_TOKEN')
    def validate_token(cls, v: str) -> str:
        if not re.match(r'^\d+:[A-Za-z0-9_-]+$', v):
            raise ValueError('Invalid Telegram bot token format')
        return v

    @field_validator('TELEGRAM_CHAT_ID')
    def validate_chat_id(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('Chat ID cannot be empty')
        return v.strip()

    @field_validator('DELTA_API_BASE')
    def validate_api_base(cls, v: str) -> str:
        if not re.match(r'^(https?://)[A-Za-z0-9\.\-:_/]+$', v.strip()):
            raise ValueError('DELTA_API_BASE must be a valid http(s) URL')
        return v.strip().rstrip('/')

    @model_validator(mode='after')
    def validate_logic(self) -> 'BotConfig':
        if self.PPO_FAST >= self.PPO_SLOW:
            raise ValueError('PPO_FAST must be less than PPO_SLOW')
        if self.RUN_TIMEOUT_SECONDS < self.MIN_RUN_TIMEOUT:
            raise ValueError(
                f'RUN_TIMEOUT_SECONDS ({self.RUN_TIMEOUT_SECONDS}s) must be >= MIN_RUN_TIMEOUT ({self.MIN_RUN_TIMEOUT}s)'
            )
        if self.RUN_TIMEOUT_SECONDS >= Constants.REDIS_LOCK_EXPIRY:
            raise ValueError(
                f'REDIS_LOCK_EXPIRY ({Constants.REDIS_LOCK_EXPIRY}s) must be > RUN_TIMEOUT_SECONDS ({self.RUN_TIMEOUT_SECONDS}s)'
            )
        if self.TELEGRAM_RATE_LIMIT_PER_MINUTE < 10 or self.TELEGRAM_RATE_LIMIT_PER_MINUTE > 30:
            raise ValueError('TELEGRAM_RATE_LIMIT_PER_MINUTE must be 10-30')
        return self

def load_config() -> BotConfig:
    config_file = os.getenv("CONFIG_FILE", "config_macd.json")
    data: Dict[str, Any] = {}
    if Path(config_file).exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                data = json_loads(f.read())
        except Exception as exc:
            print(f"❌ ERROR: Config file {config_file} is not valid JSON", file=sys.stderr)
            print(f"❌ Details: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"⚠️ WARNING: Config file {config_file} not found, using environment variables only", file=sys.stderr)

    for key in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "REDIS_URL", "DELTA_API_BASE"):
        env_value = os.getenv(key)
        if env_value:
            data[key] = env_value
        if not data.get(key) or data.get(key) == "__SET_IN_GITLAB_CI__":
            print(f"❌ ERROR: Missing required config: {key}", file=sys.stderr)
            print(f"❌ Set this in GitLab CI/CD Settings → Variables", file=sys.stderr)
            sys.exit(1)
    try:
        return BotConfig(**data)
    except Exception as exc:
        print(f"❌ ERROR: Pydantic validation failed", file=sys.stderr)
        print(f"❌ Details: {exc}", file=sys.stderr)
        sys.exit(1)

cfg = load_config()

class SecretFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = str(record.getMessage())
            if ("bot" in msg and ":" in msg) or ("redis://" in msg) or ("chat_id=" in msg):
                msg = re.sub(r'\b\d{6,}:[A-Za-z0-9_-]{20,}\b', '[REDACTED_TELEGRAM_TOKEN]', msg)
                msg = re.sub(r'chat_id=\d+', '[REDACTED_CHAT_ID]', msg)
                msg = re.sub(r'(redis://[^@]+@)', 'redis://[REDACTED]@', msg)
                record.msg = msg
        except Exception:
            pass
        return True

class TraceContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = TRACE_ID.get()
        record.pair_id = PAIR_ID.get()
        return True

class SafeFormatter(logging.Formatter):
    _SECRET_RE = re.compile(r"\b\d{6,}:[A-Za-z0-9_-]{20,}\b")
    _CHAT_RE = re.compile(r"chat_id=\d+")
    _REDIS_RE = re.compile(r"(redis://[^@]+@)")

    def format(self, record: logging.LogRecord) -> str:
        record.args = None
        formatted = super().format(record)
        if ("bot" in formatted and ":" in formatted) or ("redis://" in formatted) or ("chat_id=" in formatted):
            formatted = self._SECRET_RE.sub("[REDACTED_TOKEN]", formatted)
            formatted = self._CHAT_RE.sub("chat_id=[REDACTED]", formatted)
            formatted = self._REDIS_RE.sub("redis://[REDACTED]@", formatted)
        return formatted

def setup_logging() -> logging.Logger:
    logger = logging.getLogger("macd_bot")
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    level = logging.DEBUG if cfg.DEBUG_MODE else getattr(logging, cfg.LOG_LEVEL, logging.INFO)
    logger.setLevel(level)
    logger.propagate = False
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(SafeFormatter(
        fmt='%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    console.addFilter(SecretFilter())
    console.addFilter(TraceContextFilter())
    logger.addHandler(console)
    return logger

logger = setup_logging()
shutdown_event = asyncio.Event()

def _sync_signal_handler(sig: int, frame: Any) -> None:
    logger.warning(f"Received signal {sig}, initiating async shutdown...")
    try:
        loop = asyncio.get_running_loop()
        loop.call_soon_threadsafe(shutdown_event.set)
    except RuntimeError:
        pass

signal.signal(signal.SIGTERM, _sync_signal_handler)
signal.signal(signal.SIGINT, _sync_signal_handler)

async def cancel_all_tasks(grace_seconds: int = 5) -> None:
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    if not tasks:
        return
    logger.info(f"Cancelling {len(tasks)} tasks...")
    for t in tasks:
        t.cancel()
    try:
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=grace_seconds)
    except asyncio.TimeoutError:
        logger.warning("Timeout while cancelling tasks")

_STARTUP_BANNER_PRINTED = False

def print_startup_banner_once() -> None:
    global _STARTUP_BANNER_PRINTED
    if _STARTUP_BANNER_PRINTED:
        return
    _STARTUP_BANNER_PRINTED = True
    logger.info(
        f"🚀 Bot v{__version__} | Pairs: {len(cfg.PAIRS)} | Workers: {cfg.MAX_PARALLEL_FETCH} | "
        f"Timeout: {cfg.RUN_TIMEOUT_SECONDS}s | Redis Lock: {Constants.REDIS_LOCK_EXPIRY}s"
    )

print_startup_banner_once()

class SessionManager:
    _session: ClassVar[Optional[aiohttp.ClientSession]] = None
    _ssl_context: ClassVar[Optional[ssl.SSLContext]] = None
    _lock: ClassVar[asyncio.Lock] = asyncio.Lock()

    @classmethod
    def _get_ssl_context(cls) -> ssl.SSLContext:
        if cls._ssl_context is None:
            ctx = ssl.create_default_context()
            ctx.check_hostname = True
            ctx.verify_mode = ssl.CERT_REQUIRED
            cls._ssl_context = ctx
        return cls._ssl_context

    @classmethod
    async def get_session(cls) -> aiohttp.ClientSession:
        async with cls._lock:
            if cls._session is None or cls._session.closed:
                connector = TCPConnector(
                    limit=cfg.TCP_CONN_LIMIT,
                    limit_per_host=cfg.TCP_CONN_LIMIT_PER_HOST,
                    ssl=cls._get_ssl_context(),
                    force_close=False,
                    enable_cleanup_closed=True,
                    ttl_dns_cache=3600,
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
    async def close_session(cls) -> None:
        async with cls._lock:
            if cls._session and not cls._session.closed:
                try:
                    await cls._session.close()
                    await asyncio.sleep(0.25)
                except Exception as e:
                    logger.warning(f"Error closing session: {e}")
                finally:
                    cls._session = None
                    logger.debug("Closed shared aiohttp session")

class RetryCategory:
    NETWORK = "network"
    RATE_LIMIT = "rate_limit"
    API_ERROR = "api_error"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"

def categorize_exception(exc: Exception) -> str:
    if isinstance(exc, asyncio.TimeoutError):
        return RetryCategory.TIMEOUT
    elif isinstance(exc, (ClientConnectorError, aiohttp.ClientConnectorError)):
        return RetryCategory.NETWORK
    elif isinstance(exc, ClientResponseError):
        if hasattr(exc, 'status') and exc.status == 429:
            return RetryCategory.RATE_LIMIT
        return RetryCategory.API_ERROR
    elif isinstance(exc, (ClientError, aiohttp.ClientError)):
        return RetryCategory.NETWORK
    return RetryCategory.UNKNOWN

async def retry_async(
    fn: Callable,
    *args,
    retries: int = 3,
    base_backoff: float = 0.8,
    cap: float = 30.0,
    jitter_min: float = 0.05,
    jitter_max: float = 0.5,
    on_error: Optional[Callable[[Exception, int, str], None]] = None,
    **kwargs
):
    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        if shutdown_event.is_set():
            raise asyncio.CancelledError()
        try:
            return await fn(*args, **kwargs)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            last_exc = e
            category = categorize_exception(e)
            if on_error:
                try:
                    on_error(e, attempt, category)
                except Exception:
                    pass
            if attempt >= retries:
                break
            base_delay = min(cap, base_backoff * (2 ** (attempt - 1)))
            jitter = base_delay * random.uniform(jitter_min, jitter_max)
            sleep_time = base_delay + jitter
            logger.debug(
                f"Retry attempt {attempt}/{retries} after {sleep_time:.2f}s | "
                f"Category: {category} | Error: {str(e)[:100]}"
            )
            await asyncio.sleep(sleep_time)
    raise last_exc or RuntimeError("retry_async: unknown failure")

def get_trigger_timestamp() -> int:
    trigger_ts_str = os.getenv("TRIGGER_TIMESTAMP")
    if trigger_ts_str:
        try:
            trigger_ts = int(trigger_ts_str)
            now = int(time.time())
            if abs(now - trigger_ts) > 600:
                logger.warning(f"TRIGGER_TIMESTAMP ({trigger_ts}) is >10 min from now ({now}), using current time")
                return now
            logger.debug(f"Using TRIGGER_TIMESTAMP from env: {trigger_ts}")
            return trigger_ts
        except (ValueError, TypeError):
            logger.warning(f"Invalid TRIGGER_TIMESTAMP: {trigger_ts_str}, using current time")
    return int(time.time())

def calculate_expected_candle_timestamp(reference_time: int, interval_minutes: int) -> int:
    interval_seconds = interval_minutes * 60
    current_window = reference_time // interval_seconds
    last_closed_candle = (current_window * interval_seconds) - interval_seconds
    return last_closed_candle

def validate_candle_timestamp(candle_ts: int, expected_ts: int, tolerance_seconds: int = 120) -> bool:
    diff = abs(candle_ts - expected_ts)
    if diff > tolerance_seconds:
        logger.error(f"Candle timestamp mismatch! Expected ~{expected_ts}, got {candle_ts} (diff: {diff}s)")
        return False
    return True

def parse_candles_to_numpy(result: Optional[Dict[str, Any]]) -> Optional[Dict[str, np.ndarray]]:
    if not result or not isinstance(result, dict):
        return None
    res = result.get("result", {}) or {}
    required = ("t", "o", "h", "l", "c", "v")
    if not all(k in res for k in required):
        return None
    try:
        data = {
            "timestamp": np.array(res["t"], dtype=np.int64),
            "open":   np.array(res["o"], dtype=np.float64),
            "high":   np.array(res["h"], dtype=np.float64),
            "low":    np.array(res["l"], dtype=np.float64),
            "close":  np.array(res["c"], dtype=np.float64),
            "volume": np.array(res["v"], dtype=np.float64),
        }
        if data["timestamp"][-1] > 1_000_000_000_000:
            data["timestamp"] = (data["timestamp"] // 1000).astype(np.int64)
        if data["close"][-1] <= 0 or np.isnan(data["close"][-1]):
            return None
        return data
    except Exception:
        return None

def validate_candle_arrays(arr: Optional[Dict[str, np.ndarray]], required_len: int = 0, interval_seconds: Optional[int] = None) -> Tuple[bool, Optional[str]]:
    try:
        if arr is None:
            return False, "No data"
        ts = arr.get("timestamp")
        close = arr.get("close")
        if ts is None or close is None or ts.size == 0 or close.size == 0:
            return False, "Missing timestamp/close"
        if required_len and ts.size < required_len:
            return False, f"Insufficient data: {ts.size} < {required_len}"
        if np.isnan(close).any() or np.any(close <= 0):
            return False, "Invalid close prices (NaN or <= 0)"
        if not np.all(ts[1:] >= ts[:-1]):
            return False, "Timestamps not monotonic increasing"

        if ts.size >= 2:
            diffs = np.diff(ts)
            if diffs.size > 0:
                median_diff = np.median(diffs)
                max_expected_gap = median_diff * Constants.MAX_CANDLE_GAP_MULTIPLIER
                gaps = diffs[diffs > max_expected_gap]
                if gaps.size > 0:
                    logger.warning(f"Detected {gaps.size} candle gaps (median: {median_diff}s, max gap: {gaps.max()}s)")

        if close.size >= 2:
            pct = np.abs(np.diff(close) / close[:-1]) * 100.0
            extreme = pct[pct > Constants.MAX_PRICE_CHANGE_PERCENT]
            if extreme.size > 0:
                logger.warning(f"Detected {extreme.size} extreme price changes (max: {extreme.max():.2f}%)")
                return False, f"Extreme price spike detected: {extreme.max():.2f}%"

        return True, None
    except Exception as e:
        logger.error(f"Array validation failed: {e}")
        return False, f"Validation error: {str(e)}"

def get_last_closed_index_from_array(ts: np.ndarray, interval_minutes: int, reference_time: Optional[int] = None) -> Optional[int]:
    if ts is None or ts.size < 2:
        return None
    if reference_time is None:
        reference_time = get_trigger_timestamp()
    last_ts = int(ts[-1])
    interval_seconds = interval_minutes * 60
    publication_buffer = Constants.CANDLE_PUBLICATION_LAG_SEC
    expected_last_closed = calculate_expected_candle_timestamp(reference_time, interval_minutes)
    validate_candle_timestamp(last_ts, expected_last_closed, tolerance_seconds=interval_seconds)
    if reference_time >= (last_ts + interval_seconds + publication_buffer):
        return ts.size - 1
    else:
        return ts.size - 2 if ts.size >= 2 else None

def build_products_map_from_api_result(api_products: Optional[Dict[str, Any]]) -> Dict[str, dict]:
    products_map: Dict[str, dict] = {}
    if not api_products or not api_products.get("result"):
        return products_map
    valid_pattern = re.compile(r'^[A-Z0-9_]+$')
    for p in api_products["result"]:
        try:
            symbol = p.get("symbol", "")
            if not valid_pattern.match(symbol):
                continue
            symbol_norm = symbol.replace("_USDT", "USD").replace("USDT", "USD")
            if p.get("contract_type") == "perpetual_futures":
                for pair_name in cfg.PAIRS:
                    if symbol_norm == pair_name or symbol_norm.replace("_", "") == pair_name:
                        products_map[pair_name] = {"id": p.get("id"), "symbol": p.get("symbol"), "contract_type": p.get("contract_type")}
                        break
        except Exception:
            pass
    return products_map

class RateLimitedFetcher:
    def __init__(self, max_per_minute: int = 60, concurrency: int = 4):
        self.max_per_minute = max_per_minute
        self.semaphore = asyncio.Semaphore(concurrency)
        self.requests: deque[float] = deque()
        self.lock = asyncio.Lock()
        self.total_waits = 0
        self.total_wait_time = 0.0

    async def call(self, func: Callable, *args, **kwargs):
        async with self.lock:
            now = time.time()
            while self.requests and now - self.requests[0] > 60:
                self.requests.popleft()
            if len(self.requests) >= self.max_per_minute:
                sleep_time = 60 - (now - self.requests[0])
                jitter = random.uniform(0.05, 0.2)
                total_sleep = sleep_time + jitter
                self.total_waits += 1
                self.total_wait_time += total_sleep
                logger.debug(
                    f"Rate limit reached ({len(self.requests)}/{self.max_per_minute}), "
                    f"sleeping {total_sleep:.2f}s | Total waits: {self.total_waits}"
                )
                await asyncio.sleep(total_sleep)
                now = time.time()
                while self.requests and now - self.requests[0] > 60:
                    self.requests.popleft()
            self.requests.append(time.time())
        async with self.semaphore:
            return await func(*args, **kwargs)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_waits": self.total_waits,
            "total_wait_time_seconds": round(self.total_wait_time, 2),
            "current_queue_size": len(self.requests),
            "max_per_minute": self.max_per_minute
        }

async def async_fetch_json(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    retries: int = 3,
    backoff: float = 1.5,
    timeout: int = 15
) -> Optional[Dict[str, Any]]:
    session = await SessionManager.get_session()
    last_error = None
    retry_stats = {
        RetryCategory.NETWORK: 0,
        RetryCategory.RATE_LIMIT: 0,
        RetryCategory.API_ERROR: 0,
        RetryCategory.TIMEOUT: 0,
        RetryCategory.UNKNOWN: 0
    }
    for attempt in range(1, retries + 1):
        if shutdown_event.is_set():
            return None
        try:
            async with session.get(url, params=params, timeout=timeout) as resp:
                if resp.status == 429:
                    retry_after = resp.headers.get('Retry-After')
                    wait_sec = min(int(retry_after) if retry_after else 1, Constants.CIRCUIT_BREAKER_MAX_WAIT)
                    jitter = random.uniform(0.1, 0.5)
                    total_wait = wait_sec + jitter
                    logger.warning(
                        f"Rate limited (429) | URL: {url[:80]} | "
                        f"Retry-After: {retry_after} | Waiting: {total_wait:.2f}s"
                    )
                    retry_stats[RetryCategory.RATE_LIMIT] += 1
                    await asyncio.sleep(total_wait)
                    continue

                if resp.status >= 500:
                    retry_stats[RetryCategory.API_ERROR] += 1
                    logger.warning(
                        f"Server error {resp.status} on attempt {attempt}/{retries} | URL: {url[:80]}"
                    )
                    if attempt < retries:
                        base_delay = min(Constants.CIRCUIT_BREAKER_MAX_WAIT / 10, backoff * (2 ** (attempt - 1)))
                        jitter = base_delay * random.uniform(0.1, 0.5)
                        await asyncio.sleep(base_delay + jitter)
                    continue

                if resp.status >= 400:
                    logger.error(
                        f"Client error {resp.status} for {url[:80]} | This usually indicates invalid request"
                    )
                    return None

                data = await resp.json(loads=json_loads)
                if any(retry_stats.values()):
                    logger.info(
                        f"Fetch succeeded after retries | URL: {url[:80]} | Stats: {retry_stats}"
                    )
                return data

        except (asyncio.TimeoutError, ClientConnectorError, ClientError, ClientResponseError) as e:
            last_error = e
            category = categorize_exception(e)
            retry_stats[category] = retry_stats.get(category, 0) + 1
            logger.warning(
                f"Fetch error (attempt {attempt}/{retries}) | Category: {category} | URL: {url[:80]} | Error: {str(e)[:100]}"
            )
            if attempt < retries:
                base_delay = min(Constants.CIRCUIT_BREAKER_MAX_WAIT / 10, backoff * (2 ** (attempt - 1)))
                jitter = base_delay * random.uniform(0.1, 0.5)
                await asyncio.sleep(base_delay + jitter)
        except Exception as e:
            last_error = e
            retry_stats[RetryCategory.UNKNOWN] += 1
            logger.exception(f"Unexpected fetch error for {url[:80]}: {e}")
            break

    logger.error(
        f"Failed to fetch after {retries} attempts | URL: {url[:80]} | "
        f"Stats: {retry_stats} | Last error: {last_error}"
    )
    return None

class DataFetcher:
    def __init__(self, api_base: str, max_parallel: Optional[int] = None):
        self.api_base = api_base.rstrip("/")
        max_parallel = max_parallel or cfg.MAX_PARALLEL_FETCH
        self.semaphore = asyncio.Semaphore(max_parallel)
        self.timeout = cfg.HTTP_TIMEOUT
        self.rate_limiter = RateLimitedFetcher(max_per_minute=60)
        self.fetch_stats = {
            "products_success": 0,
            "products_failed": 0,
            "candles_success": 0,
            "candles_failed": 0
        }

    async def fetch_products(self) -> Optional[Dict[str, Any]]:
        url = f"{self.api_base}/v2/products"
        async with self.semaphore:
            result = await self.rate_limiter.call(
                async_fetch_json,
                url,
                retries=5,
                backoff=2.0,
                timeout=self.timeout
            )
            if result:
                self.fetch_stats["products_success"] += 1
                logger.debug(f"Products fetch successful | URL: {url}")
            else:
                self.fetch_stats["products_failed"] += 1
                logger.warning(f"Products fetch failed | URL: {url}")
            return result

    async def fetch_candles(
        self,
        symbol: str,
        resolution: str,
        limit: int,
        reference_time: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        if reference_time is None:
            reference_time = get_trigger_timestamp()

        minutes = int(resolution) if resolution != "D" else 1440
        window = minutes * 60
        current_window = reference_time // window
        last_close = (current_window * window)
        cushion = max(3, min(Constants.CANDLE_PUBLICATION_LAG_SEC, window - 3))
        last_close -= cushion

        params = {
            "resolution": resolution,
            "symbol": symbol,
            "from": last_close - (limit * window),
            "to": last_close
        }
        url = f"{self.api_base}/v2/chart/history"
        async with self.semaphore:
            data = await self.rate_limiter.call(
                async_fetch_json,
                url,
                params=params,
                retries=cfg.CANDLE_FETCH_RETRIES,
                backoff=cfg.CANDLE_FETCH_BACKOFF,
                timeout=self.timeout
            )
            if data:
                self.fetch_stats["candles_success"] += 1
            else:
                self.fetch_stats["candles_failed"] += 1
                logger.warning(
                    f"Candles fetch failed | Symbol: {symbol} | Resolution: {resolution} | Params: {params}"
                )
            return data

# -----------------------------
# Numba kernels
# -----------------------------

@njit(fastmath=True, cache=False)
def _sma_loop(data: np.ndarray, period: int) -> np.ndarray:
    n = len(data)
    out = np.empty(n, dtype=np.float64)
    out[:] = np.nan
    window_sum = 0.0
    count = 0
    for i in range(n):
        val = data[i]
        if not np.isnan(val):
            window_sum += val
            count += 1
        if i >= period:
            old_val = data[i - period]
            if not np.isnan(old_val):
                window_sum -= old_val
                count -= 1
        min_periods = max(2, period // 3)
        if count >= min_periods:
            out[i] = window_sum / count
        else:
            out[i] = np.nan
    return out

@njit(fastmath=True, cache=False)
def _kalman_loop(src: np.ndarray, length: int, R: float, Q: float) -> np.ndarray:
    n = len(src)
    result = np.empty(n, dtype=np.float64)
    estimate = src[0] if not np.isnan(src[0]) else 0.0
    error_est = 1.0
    error_meas = R * max(1.0, float(length))
    Q_div_length = Q / max(1.0, float(length))
    for i in range(n):
        current = src[i]
        if np.isnan(current):
            result[i] = estimate
            continue
        if np.isnan(estimate):
            estimate = current
        prediction = estimate
        kalman_gain = error_est / (error_est + error_meas)
        estimate = prediction + kalman_gain * (current - prediction)
        error_est = (1.0 - kalman_gain) * error_est + Q_div_length
        result[i] = estimate
    return result

@njit(fastmath=True, cache=False)
def _vwap_daily_loop(high, low, close, volume, timestamps) -> np.ndarray:
    n = len(close)
    vwap = np.empty(n, dtype=np.float64)
    cum_vol = 0.0
    cum_pv = 0.0
    prev_day = timestamps[0] // 86400
    for i in range(n):
        curr_ts = timestamps[i]
        curr_day = curr_ts // 86400
        if curr_day != prev_day:
            cum_vol = 0.0
            cum_pv = 0.0
            prev_day = curr_day
        h = high[i]
        l = low[i]
        c = close[i]
        v = volume[i]
        if np.isnan(h) or np.isnan(l) or np.isnan(c) or np.isnan(v):
            vwap[i] = vwap[i-1] if i > 0 else c
            continue
        avg_price = (h + l + c) / 3.0
        cum_vol += v
        cum_pv += (avg_price * v)
        if cum_vol == 0:
            vwap[i] = c
        else:
            vwap[i] = cum_pv / cum_vol
    return vwap

@njit(fastmath=True, cache=False)
def _ema_loop(data: np.ndarray, alpha: float) -> np.ndarray:
    n = len(data)
    out = np.empty(n, dtype=np.float64)
    out[0] = data[0] if not np.isnan(data[0]) else 0.0
    for i in range(1, n):
        curr = data[i]
        if np.isnan(curr):
            out[i] = out[i-1]
        else:
            out[i] = alpha * curr + (1 - alpha) * out[i-1]
    return out

@njit(fastmath=True, cache=False)
def _rng_filter_loop(x: np.ndarray, r: np.ndarray) -> np.ndarray:
    n = len(x)
    filt = np.zeros(n, dtype=np.float64)
    filt[0] = x[0] if not np.isnan(x[0]) else 0.0
    for i in range(1, n):
        prev_filt = filt[i - 1]
        curr_x = x[i]
        curr_r = r[i]
        if np.isnan(curr_r) or np.isnan(curr_x):
            filt[i] = prev_filt
            continue
        if curr_x > prev_filt:
            target = curr_x - curr_r
            if target < prev_filt:
                filt[i] = prev_filt
            else:
                filt[i] = target
        else:
            target = curr_x + curr_r
            if target > prev_filt:
                filt[i] = prev_filt
            else:
                filt[i] = target
    return filt

@njit(fastmath=True, cache=False)
def _calc_mmh_worm_loop(close_arr, sd_arr, rows):
    worm_arr = np.empty(rows, dtype=np.float64)
    first_val = close_arr[0] if not np.isnan(close_arr[0]) else 0.0
    worm_arr[0] = first_val
    for i in range(1, rows):
        src = close_arr[i] if not np.isnan(close_arr[i]) else worm_arr[i - 1]
        prev_worm = worm_arr[i - 1]
        diff = src - prev_worm
        sd_i = sd_arr[i]
        if np.isnan(sd_i):
            delta = diff
        else:
            delta = (np.sign(diff) * sd_i) if (np.abs(diff) > sd_i) else diff
        worm_arr[i] = prev_worm + delta
    return worm_arr

@njit(fastmath=True, cache=False)
def _calc_mmh_value_loop(temp_arr, rows):
    value_arr = np.zeros(rows, dtype=np.float64)
    value_arr[0] = 1.0
    for i in range(1, rows):
        prev_v = value_arr[i - 1] if not np.isnan(value_arr[i - 1]) else 1.0
        t = temp_arr[i] if not np.isnan(temp_arr[i]) else 0.5
        v = t - 0.5 + 0.5 * prev_v
        value_arr[i] = max(-0.9999, min(0.9999, v))
    return value_arr

@njit(fastmath=True, cache=False)
def _calc_mmh_momentum_loop(momentum_arr, rows):
    for i in range(1, rows):
        prev = momentum_arr[i - 1] if not np.isnan(momentum_arr[i - 1]) else 0.0
        momentum_arr[i] = momentum_arr[i] + 0.5 * prev
    return momentum_arr

# -----------------------------
# Indicator sanitization
# -----------------------------

def sanitize_indicator_array(name: str, arr: np.ndarray) -> np.ndarray:
    try:
        arr = arr.copy()
        arr[np.isinf(arr)] = np.nan
        n = arr.size
        if n == 0:
            return arr
        last = np.nan
        for i in range(n):
            if np.isnan(arr[i]):
                arr[i] = last if not np.isnan(last) else arr[i]
            else:
                last = arr[i]
        nextv = np.nan
        for i in range(n - 1, -1, -1):
            if np.isnan(arr[i]):
                arr[i] = nextv if not np.isnan(nextv) else 0.0
            else:
                nextv = arr[i]
        return arr
    except Exception as e:
        logger.error(f"Failed to sanitize indicator {name}: {e}")
        return np.zeros_like(arr, dtype=np.float64)

# -----------------------------
# Pure-NumPy indicators
# -----------------------------

def calculate_vwap_numpy_arrays(data_15m: Dict[str, np.ndarray]) -> np.ndarray:
    if data_15m is None or data_15m["close"].size == 0:
        return np.empty(0, dtype=np.float64)
    h = data_15m["high"].astype(np.float64)
    l = data_15m["low"].astype(np.float64)
    c = data_15m["close"].astype(np.float64)
    v = data_15m["volume"].astype(np.float64)
    t = data_15m["timestamp"].astype(np.int64)
    res = _vwap_daily_loop(h, l, c, v, t)
    return sanitize_indicator_array("VWAP", res)

def calculate_smooth_rsi_numpy_array(close: np.ndarray, rsi_len: int, kalman_len: int) -> np.ndarray:
    close = close.astype(np.float64)
    delta = np.zeros_like(close)
    if close.size >= 2:
        delta[1:] = close[1:] - close[:-1]
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    alpha = 1.0 / rsi_len
    avg_gain = _ema_loop(gain, alpha)
    avg_loss = _ema_loop(loss, alpha)
    avg_loss = np.where(avg_loss == 0, 1e-10, avg_loss)
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    smooth_rsi = _kalman_loop(rsi, kalman_len, 0.01, 0.1)
    return sanitize_indicator_array("SmoothRSI", smooth_rsi)

def calculate_rma_numpy_array(values: np.ndarray, period: int) -> np.ndarray:
    if values.size == 0:
        return values
    alpha = 1.0 / period
    result = _ema_loop(values.astype(np.float64), alpha)
    return result

def calculate_cirrus_cloud_numba_arrays(close: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    close = close.astype(np.float64)
    diff = np.zeros_like(close)
    if close.size >= 2:
        diff[1:] = np.abs(close[1:] - close[:-1])
    alpha_t = 2.0 / (cfg.X1 + 1)
    avrng = _ema_loop(diff, alpha_t)
    wper = cfg.X1 * 2 - 1
    alpha_w = 2.0 / (wper + 1)
    smooth_rng_x1 = _ema_loop(avrng, alpha_w) * cfg.X2
    filt_x1 = _rng_filter_loop(close, smooth_rng_x1)
    alpha_t2 = 2.0 / (cfg.X3 + 1)
    avrng2 = _ema_loop(diff, alpha_t2)
    wper2 = cfg.X3 * 2 - 1
    alpha_w2 = 2.0 / (wper2 + 1)
    smooth_rng_x2 = _ema_loop(avrng2, alpha_w2) * cfg.X4
    filt_x12 = _rng_filter_loop(close, smooth_rng_x2)
    upw = filt_x1 < filt_x12
    dnw = filt_x1 > filt_x12
    return upw, dnw, filt_x1, filt_x12

def calculate_ppo_numpy_arrays(close: np.ndarray, fast: int, slow: int, signal: int) -> Tuple[np.ndarray, np.ndarray]:
    close = close.astype(np.float64)
    alpha_fast = 2.0 / (fast + 1)
    fast_ma = _ema_loop(close, alpha_fast)
    alpha_slow = 2.0 / (slow + 1)
    slow_ma = _ema_loop(close, alpha_slow)
    with np.errstate(divide='ignore', invalid='ignore'):
        ppo = ((fast_ma - slow_ma) / slow_ma) * 100.0
    ppo = np.nan_to_num(ppo, nan=0.0, posinf=0.0, neginf=0.0)
    alpha_signal = 2.0 / (signal + 1)
    ppo_sig = _ema_loop(ppo, alpha_signal)
    return ppo, ppo_sig

def calculate_magical_momentum_hist_array(close: np.ndarray, period: int = 144, responsiveness: float = 0.9) -> np.ndarray:
    try:
        if close is None or close.size == 0:
            return np.empty(0, dtype=np.float64)
        close = close.astype(np.float64)
        rows = close.size
        resp_clamped = max(0.00001, min(1.0, float(responsiveness)))
        delta = np.zeros_like(close)
        if rows >= 2:
            delta[1:] = np.abs(close[1:] - close[:-1])
        alpha_sd = 2.0 / (50 + 1)
        sd = _ema_loop(delta, alpha_sd) * resp_clamped
        sd = sanitize_indicator_array("MMH_SD", sd)
        worm_arr = _calc_mmh_worm_loop(close, sd, rows)
        ma = _sma_loop(close, period)
        ma = sanitize_indicator_array("MMH_MA", ma)
        raw = (worm_arr - ma) / np.where(worm_arr == 0.0, np.nan, worm_arr)
        raw[np.isinf(raw)] = np.nan
        raw = np.nan_to_num(raw, nan=0.0)
        raw = sanitize_indicator_array("MMH_RAW", raw)
        min_med = np.empty(rows, dtype=np.float64)
        max_med = np.empty(rows, dtype=np.float64)
        for i in range(rows):
            start = 0 if i - period + 1 < 0 else i - period + 1
            window = raw[start:i+1]
            min_med[i] = np.min(window) if window.size > 0 else 0.0
            max_med[i] = np.max(window) if window.size > 0 else 1.0
        denom = max_med - min_med
        denom[denom == 0] = 1e-12
        temp = (raw - min_med) / denom
        temp = np.clip(temp, 0.0, 1.0)
        temp = np.where(np.isnan(temp), 0.5, temp)
        temp = sanitize_indicator_array("MMH_TEMP", temp)
        value_arr = _calc_mmh_value_loop(temp, rows)
        temp2 = (1.0 + value_arr) / (1.0 - value_arr)
        temp2[np.isinf(temp2)] = np.nan
        temp2 = np.nan_to_num(temp2, nan=1.0)
        temp2 = np.clip(temp2, 1e-8, 1e8)
        momentum = 0.25 * np.log(temp2)
        momentum = sanitize_indicator_array("MMH_MOMENTUM", momentum)
        momentum_arr = momentum.astype(np.float64)
        momentum_arr = _calc_mmh_momentum_loop(momentum_arr, rows)
        hist = momentum_arr
        hist = sanitize_indicator_array("MMH_HIST", hist)
        return hist
    except Exception as e:
        logger.error(f"MMH calculation failed: {e}")
        return np.zeros(close.size, dtype=np.float64)

indicator_semaphore = asyncio.Semaphore(cfg.INDICATOR_THREAD_LIMIT)

async def calculate_indicator_threaded(func: Callable, *args, **kwargs):
    async with indicator_semaphore:
        return await asyncio.to_thread(func, *args, **kwargs)

def calculate_all_indicators_sync(data_15m: Dict[str, np.ndarray], data_5m: Dict[str, np.ndarray], data_daily: Optional[Dict[str, np.ndarray]]) -> dict:
    results = {}
    close_15 = data_15m["close"]
    close_5 = data_5m["close"]
    ppo, ppo_signal = calculate_ppo_numpy_arrays(close_15, cfg.PPO_FAST, cfg.PPO_SLOW, cfg.PPO_SIGNAL)
    results['ppo'] = ppo
    results['ppo_signal'] = ppo_signal
    results['smooth_rsi'] = calculate_smooth_rsi_numpy_array(close_15, cfg.SRSI_RSI_LEN, cfg.SRSI_KALMAN_LEN)
    if cfg.ENABLE_VWAP:
        results['vwap'] = calculate_vwap_numpy_arrays(data_15m)
    else:
        results['vwap'] = np.zeros_like(close_15, dtype=np.float64)
    mmh = calculate_magical_momentum_hist_array(close_15)
    results['mmh'] = np.nan_to_num(mmh, nan=0.0, posinf=0.0, neginf=0.0)
    if cfg.CIRRUS_CLOUD_ENABLED:
        upw, dnw, filtx1, filtx12 = calculate_cirrus_cloud_numba_arrays(close_15)
        results['upw'] = upw
        results['dnw'] = dnw
    else:
        results['upw'] = np.zeros_like(close_15, dtype=bool)
        results['dnw'] = np.zeros_like(close_15, dtype=bool)
    results['rma50_15'] = calculate_rma_numpy_array(close_15, cfg.RMA_50_PERIOD)
    results['rma200_5'] = calculate_rma_numpy_array(close_5, cfg.RMA_200_PERIOD)
    return results

def check_candle_quality_with_reason_arrays(open_arr: np.ndarray, high_arr: np.ndarray, low_arr: np.ndarray, close_arr: np.ndarray, idx: int, is_buy: bool) -> Tuple[bool, str]:
    try:
        if idx < 0 or idx >= close_arr.size:
            return False, f"Invalid index {idx}"
        o = float(open_arr[idx]); h = float(high_arr[idx]); l = float(low_arr[idx]); c = float(close_arr[idx])
        candle_range = h - l
        if candle_range < 1e-8:
            return False, "Candle range too small"
        if is_buy:
            if c <= o:
                return False, f"Not green candle (C={c:.4f} <= O={o:.4f})"
            upper_wick = h - c
            wick_ratio = upper_wick / candle_range
            if wick_ratio >= Constants.MIN_WICK_RATIO:
                return False, f"Upper wick {wick_ratio*100:.2f}% > {Constants.MIN_WICK_RATIO*100:.0f}%"
            return True, "Passed"
        else:
            if c >= o:
                return False, f"Not red candle (C={c:.4f} >= O={o:.4f})"
            lower_wick = c - l
            wick_ratio = lower_wick / candle_range
            if wick_ratio >= Constants.MIN_WICK_RATIO:
                return False, f"Lower wick {wick_ratio*100:.2f}% > {Constants.MIN_WICK_RATIO*100:.0f}%"
            return True, "Passed"
    except Exception as e:
        return False, f"Error: {str(e)}"

# -----------------------------
# Redis state store with Lua dedup
# -----------------------------

class RedisStateStore:
    DEDUP_LUA = """
    -- KEYS[1] = recent_key
    -- ARGV[1] = value
    -- ARGV[2] = expiry seconds
    if redis.call("EXISTS", KEYS[1]) == 1 then
        return 0
    else
        redis.call("SET", KEYS[1], ARGV[1])
        redis.call("EXPIRE", KEYS[1], tonumber(ARGV[2]))
        return 1
    end
    """

    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self._redis: Optional[redis.Redis] = None
        self.state_prefix = "pair_state:"
        self.meta_prefix = "metadata:"
        self.alert_prefix = "alert:"
        self.expiry_seconds = cfg.STATE_EXPIRY_DAYS * 86400
        self.alert_expiry_seconds = cfg.STATE_EXPIRY_DAYS * 86400
        self.metadata_expiry_seconds = 7 * 86400
        self.degraded = False
        self.degraded_alerted = False
        self._connection_attempts = 0
        self._dedup_sha: Optional[str] = None

    async def _attempt_connect(self, timeout: float = 5.0) -> bool:
        try:
            self._redis = redis.from_url(
                self.redis_url,
                socket_connect_timeout=timeout,
                socket_timeout=timeout,
                retry_on_timeout=True,
                max_connections=10,
                decode_responses=True,
            )
            ok = await self._ping_with_retry(timeout)
            if ok:
                if cfg.DEBUG_MODE:
                    logger.debug("Connected to RedisStateStore (decode_responses=True, max_connections=10)")
                self.degraded = False
                self.degraded_alerted = False
                self._connection_attempts = 0
                try:
                    self._dedup_sha = await self._redis.script_load(self.DEDUP_LUA)
                    logger.debug("Loaded Lua script for alert deduplication")
                except Exception as e:
                    logger.warning(f"Failed to load Lua script for dedup: {e}")
                    self._dedup_sha = None
                return True
            else:
                raise RedisConnectionError("ping failed after retries")
        except Exception as exc:
            logger.error(f"Redis connection attempt failed: {exc}")
            if self._redis:
                try:
                    await self._redis.aclose()
                except Exception:
                    pass
                self._redis = None
            return False

    async def connect(self, timeout: float = 5.0) -> None:
        if self._redis is not None and not self.degraded:
            try:
                if await self._ping_with_retry(1.0):
                    logger.debug("Redis connection healthy")
                    return
            except Exception:
                logger.debug("Redis ping failed, attempting reconnect")

        for attempt in range(1, cfg.REDIS_CONNECTION_RETRIES + 1):
            self._connection_attempts = attempt
            if cfg.DEBUG_MODE:
                logger.debug(f"Redis connection attempt {attempt}/{cfg.REDIS_CONNECTION_RETRIES}")

            if await self._attempt_connect(timeout):
                test_key = f"smoke_test:{uuid.uuid4().hex[:8]}"
                test_val = "ok"
                if (
                    await self._safe_redis_op(
                        self._redis.set(test_key, test_val, ex=10), 2.0, "smoke_set"
                    )
                    and await self._safe_redis_op(
                        self._redis.get(test_key), 2.0, "smoke_get", lambda r: r == test_val
                    )
                ):
                    await self._safe_redis_op(
                        self._redis.delete(test_key), 1.0, "smoke_cleanup"
                    )
                    expiry_mode = "TTL-based" if self.expiry_seconds > 0 else "manual"
                    logger.info(f"✅ Redis connected ({self._redis.connection_pool.max_connections} connections, {expiry_mode} expiry)")
                    info = await self._safe_redis_op(
                        self._redis.info("memory"), 3.0, "info_memory", lambda r: r
                    )
                    if info:
                        policy = info.get("maxmemory_policy", "unknown")
                        if policy in ("volatile-lru", "allkeys-lru"):
                            logger.warning(f"⚠️ Redis using {policy} - keys may be evicted under memory pressure")
                    self.degraded = False
                    self.degraded_alerted = False
                    return
                else:
                    logger.warning("Redis smoke test failed, marking degraded")

            if attempt < cfg.REDIS_CONNECTION_RETRIES:
                delay = cfg.REDIS_RETRY_DELAY * attempt
                logger.warning(f"Retrying Redis connection in {delay}s...")
                await asyncio.sleep(delay)

        logger.critical("❌ Redis connection failed after all retries")
        self.degraded = True
        self._redis = None
        logger.warning("""
🚨 REDIS DEGRADED MODE ACTIVE:
- Alert deduplication:  DISABLED (may get duplicates)
- State persistence:    DISABLED (alerts reset each run)
- Trading alerts:       STILL ACTIVE (core functionality preserved)
""")
        if cfg.FAIL_ON_REDIS_DOWN:
            raise RedisConnectionError("Redis unavailable after all retries – FAIL_ON_REDIS_DOWN=true")

    async def close(self) -> None:
        if self._redis:
            try:
                await self._redis.aclose()
            except Exception:
                pass
            self._redis = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def _ping_with_retry(self, timeout: float) -> bool:
        return (
            await self._safe_redis_op(
                self._redis.ping(), timeout, "ping", lambda r: bool(r)
            )
        ) is True

    async def _safe_redis_op(self, coro, timeout: float, op_name: str, parser: Optional[Callable] = None):
        if not self._redis:
            return None
        async def _do():
            return await asyncio.wait_for(coro, timeout=timeout)
        try:
            result = await retry_async(
                _do,
                retries=3,
                base_backoff=0.6,
                cap=3.0,
                on_error=lambda e, a, c: logger.debug(f"Redis {op_name} error (attempt {a}): {e}"),
            )
            return parser(result) if parser else result
        except (asyncio.TimeoutError, RedisConnectionError, RedisError) as e:
            logger.error(f"Redis {op_name} failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to {op_name}: {e}")
            return None

    async def get(self, key: str, timeout: float = 2.0) -> Optional[Dict[str, Any]]:
        return await self._safe_redis_op(
            self._redis.get(f"{self.state_prefix}{key}"),
            timeout,
            f"get {key}",
            lambda r: json_loads(r) if r else None,
        )

    async def set(self, key: str, state: Optional[Any], ts: Optional[int] = None, timeout: float = 2.0) -> None:
        ts = int(ts or time.time())
        redis_key = f"{self.state_prefix}{key}"
        data = json_dumps({"state": state, "ts": ts})
        await self._safe_redis_op(
            self._redis.set(redis_key, data, ex=self.expiry_seconds if self.expiry_seconds > 0 else None),
            timeout,
            f"set {key}",
        )

    async def mget_states(self, keys: List[str], timeout: float = 2.0) -> Dict[str, Optional[Dict[str, Any]]]:
        if not self._redis or not keys:
            return {}
        redis_keys = [f"{self.state_prefix}{k}" for k in keys]
        results = await self._safe_redis_op(
            self._redis.mget(redis_keys),
            timeout,
            f"mget {len(keys)} keys",
            lambda r: r if r else [],
        )
        if not results:
            return {}
        output: Dict[str, Optional[Dict[str, Any]]] = {}
        for idx, key in enumerate(keys):
            if idx < len(results) and results[idx]:
                try:
                    output[key] = json_loads(results[idx])
                except Exception:
                    output[key] = None
            else:
                output[key] = None
        return output

    async def batch_set_states(self, updates: List[Tuple[str, Any, Optional[int]]], timeout: float = 4.0) -> None:
        if self.degraded or not updates or not self._redis:
            return
        try:
            pipe = self._redis.pipeline()
            now = int(time.time())
            for key, state, custom_ts in updates:
                ts = custom_ts if custom_ts is not None else now
                data = json_dumps({"state": state, "ts": ts})
                full_key = f"{self.state_prefix}{key}"
                if self.expiry_seconds > 0:
                    pipe.set(full_key, data, ex=self.expiry_seconds)
                else:
                    pipe.set(full_key, data)
            await asyncio.wait_for(pipe.execute(), timeout=timeout)
        except Exception as e:
            logger.error(f"Batch state update failed (falling back to individual): {e}")
            for key, state, custom_ts in updates:
                await self.set(key, state, custom_ts)

    async def set_metadata(self, key: str, value: str, timeout: float = 2.0) -> None:
        await self._safe_redis_op(
            self._redis.set(f"{self.meta_prefix}{key}", value, ex=self.metadata_expiry_seconds),
            timeout,
            f"set_metadata {key}",
        )

    async def check_recent_alert(self, pair: str, alert_key: str, ts: int) -> bool:
        if self.degraded:
            return True
        window = ts // 900
        recent_key = f"recent_alert:{pair}:{alert_key}:{window}"
        try:
            if self._dedup_sha:
                res = await self._redis.evalsha(self._dedup_sha, 1, recent_key, "1", str(Constants.ALERT_DEDUP_WINDOW_SEC))
            else:
                res = await self._redis.set(recent_key, "1", nx=True, ex=Constants.ALERT_DEDUP_WINDOW_SEC)
            return bool(res)
        except Exception as e:
            logger.error(f"Dedup check failed: {e}")
            return True

    async def batch_check_recent_alerts(self, checks: List[Tuple[str, str, int]]) -> Dict[str, bool]:
        if self.degraded or not checks:
            return {f"{pair}:{alert_key}": True for pair, alert_key, _ in checks}
        results: Dict[str, bool] = {}
        for pair, alert_key, ts in checks:
            ok = await self.check_recent_alert(pair, alert_key, ts)
            results[f"{pair}:{alert_key}"] = ok
        return results

class RedisLock:
    RELEASE_LUA = """
    if redis.call("GET", KEYS[1]) == ARGV[1] then
        return redis.call("DEL", KEYS[1])
    else
        return 0
    end
    """

    def __init__(self, redis_client: Optional[redis.Redis], lock_key: str, expire: int | None = None):
        self.redis = redis_client
        self.lock_key = f"lock:{lock_key}"
        self.expire = expire or Constants.REDIS_LOCK_EXPIRY
        self.token: Optional[str] = None
        self.lost = False
        self.acquired_by_me = False
        self.last_extend_time = 0.0

    async def acquire(self, timeout: float = 5.0) -> bool:
        if not self.redis:
            logger.warning("Redis not available; cannot acquire lock")
            return False
        try:
            token = str(uuid.uuid4())
            ok = await asyncio.wait_for(self.redis.set(self.lock_key, token, nx=True, ex=self.expire), timeout=timeout)
            if ok:
                self.token = token
                self.acquired_by_me = True
                self.last_extend_time = time.time()
                logger.info(f"🔒 Lock acquired: {self.lock_key.replace('lock:', '')} ({self.expire}s)")
                return True
            logger.warning(f"Could not acquire Redis lock (held): {self.lock_key}")
            return False
        except Exception as e:
            logger.error(f"Redis lock acquisition failed: {e}")
            return False

    async def extend(self, timeout: float = 3.0) -> bool:
        if not self.token or not self.redis or not self.acquired_by_me:
            self.lost = True
            return False
        try:
            raw_val = await asyncio.wait_for(self.redis.get(self.lock_key), timeout=timeout)
            if raw_val is None:
                logger.warning("Lock lost during extend (key missing)")
                self.lost = True
                self.acquired_by_me = False
                return False
            current_token = str(raw_val)
            if current_token != self.token:
                logger.warning("Lock token mismatch on extend")
                self.lost = True
                self.acquired_by_me = False
                return False
            await asyncio.wait_for(self.redis.expire(self.lock_key, self.expire), timeout=timeout)
            self.last_extend_time = time.time()
            logger.debug(f"Extended Redis lock: {self.lock_key}")
            return True
        except Exception as e:
            logger.error(f"Error extending Redis lock: {e}")
            self.lost = True
            self.acquired_by_me = False
            return False

    def should_extend(self) -> bool:
        if not self.acquired_by_me or self.lost:
            return False
        base_interval = Constants.LOCK_EXTEND_INTERVAL
        jitter = random.uniform(0, Constants.LOCK_EXTEND_JITTER_MAX)
        extend_threshold = base_interval + jitter
        elapsed = time.time() - self.last_extend_time
        return elapsed >= extend_threshold

    async def release(self, timeout: float = 3.0) -> None:
        if not self.token or not self.redis or not self.acquired_by_me:
            return
        try:
            await asyncio.wait_for(self.redis.eval(self.RELEASE_LUA, 1, self.lock_key, self.token), timeout=timeout)
            logger.info(f"🔓 Lock released")
        except Exception as e:
            logger.error(f"Error releasing Redis lock: {e}")
        finally:
            self.token = None
            self.acquired_by_me = False

_ESCAPE_RE = re.compile(r'[_*

\[\]

()~`>#+-=|{}.!]')

def escape_markdown_v2(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    return _ESCAPE_RE.sub(r'\\\g<0>', text)

class TokenBucket:
    def __init__(self, rate: int, burst: int):
        self.rate = rate
        self.burst = burst
        self.tokens = float(burst)
        self.last_update = time.monotonic()
        self.lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self.lock:
                now = time.monotonic()
                elapsed = now - self.last_update
                self.tokens = min(self.burst, self.tokens + elapsed * (self.rate / 60))
                self.last_update = now
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                wait_time = (1 - self.tokens) / (self.rate / 60)
            await asyncio.sleep(wait_time)

class TelegramQueue:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.token_bucket = TokenBucket(cfg.TELEGRAM_RATE_LIMIT_PER_MINUTE, cfg.TELEGRAM_BURST_SIZE)

    async def send(self, message: str, priority: str = "normal") -> bool:
        try:
            await asyncio.wait_for(self._send_impl(message), timeout=30.0)
            return True
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            if not cfg.FAIL_ON_TELEGRAM_DOWN:
                return False
            raise

    async def _send_impl(self, message: str) -> bool:
        await self.token_bucket.acquire()
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        params = {"chat_id": self.chat_id, "text": message, "parse_mode": "MarkdownV2"}
        session = await SessionManager.get_session()
        for attempt in range(1, cfg.TELEGRAM_RETRIES + 1):
            if shutdown_event.is_set():
                return False
            try:
                async with session.post(url, data=params, timeout=10) as resp:
                    if resp.status == 429:
                        wait_sec = min(int(resp.headers.get("Retry-After", 1)), Constants.CIRCUIT_BREAKER_MAX_WAIT)
                        await asyncio.sleep(wait_sec + random.uniform(0.1, 0.5))
                        continue
                    if resp.status == 200:
                        return True
                    if resp.status in (400, 401, 403, 404):
                        logger.error(f"Telegram API error {resp.status} - check token/chat_id")
                        return False
                    raise Exception(f"Telegram API error {resp.status}")
            except Exception as e:
                logger.warning(f"Telegram send attempt {attempt} failed: {e}")
                if attempt < cfg.TELEGRAM_RETRIES:
                    await asyncio.sleep(min((cfg.TELEGRAM_BACKOFF_BASE ** (attempt - 1)), 30))
        return False

    async def send_batch(self, messages: List[str]) -> bool:
        if not messages:
            return True
        if len(messages) > 5:
            summary = f"🚨 MARKET ALERT: {len(messages)} signals detected\n\n"
            preview = "\n".join([f"• {msg[:50]}..." for msg in messages[:10]])
            return await self.send(escape_markdown_v2(summary + preview))
        max_len = 3800
        to_send = messages[:10]
        combined = "\n\n".join(to_send)
        if len(combined) > max_len:
            results = await asyncio.gather(*[self.send(m) for m in to_send], return_exceptions=True)
            return all(r is True for r in results if isinstance(r, bool))
        return await self.send(combined)

def build_single_msg(title: str, pair: str, price: float, ts: int, extra: Optional[str] = None) -> str:
    parts = title.split(" ", 1)
    symbols = parts[0] if len(parts) == 2 else ""
    description = parts[1] if len(parts) == 2 else title
    price_str = f"${price:,.2f}"
    line1 = f"{symbols} {pair} - {price_str}".strip()
    line2 = f"{description} : {extra}" if extra else f"{description}"
    line3 = format_ist_time(ts, "%d-%m-%Y     %H:%M IST")
    return escape_markdown_v2(f"{line1}\n{line2}\n{line3}")

def build_batched_msg(pair: str, price: float, ts: int, items: List[Tuple[str, str]]) -> str:
    headline_emoji = items[0][0].split(" ", 1)[0]
    headline = f"{headline_emoji} **{pair}** • ${price:,.2f}  {format_ist_time(ts, '%d-%m-%Y %H:%M IST')}"
    bullets = []
    for idx, (title, extra) in enumerate(items):
        prefix = "└─" if idx == len(items) - 1 else "├─"
        bullets.append(f"{prefix} {title} | {extra}")
    body = "\n".join(bullets)
    return escape_markdown_v2(f"{headline}\n{body}")

class AlertDefinition(TypedDict):
    key: str
    title: str
    check_fn: Callable[[Any, Any, Any, Any], bool]
    extra_fn: Callable[[Any, Any, Any, Any, Dict[str, Any]], str]
    requires: List[str]

ALERT_DEFINITIONS: List[AlertDefinition] = [
    {"key": "ppo_signal_up", "title": "🟢 PPO cross above signal", "check_fn": lambda ctx, ppo, ppo_sig, rsi: (ctx["buy_common"] and (ppo["prev"] <= ppo_sig["prev"]) and (ppo["curr"] > ppo_sig["curr"]) and (ppo["curr"] < Constants.PPO_THRESHOLD_BUY)), "extra_fn": lambda ctx, ppo, ppo_sig, rsi, _: f"PPO {ppo['curr']:.2f} vs Sig {ppo_sig['curr']:.2f} | MMH ({ctx['mmh_curr']:.2f})", "requires": ["ppo", "ppo_signal"]},
    {"key": "ppo_signal_down", "title": "🔴 PPO cross below signal", "check_fn": lambda ctx, ppo, ppo_sig, rsi: (ctx["sell_common"] and (ppo["prev"] >= ppo_sig["prev"]) and (ppo["curr"] < ppo_sig["curr"]) and (ppo["curr"] > Constants.PPO_THRESHOLD_SELL)), "extra_fn": lambda ctx, ppo, ppo_sig, rsi, _: f"PPO {ppo['curr']:.2f} vs Sig {ppo_sig['curr']:.2f} | MMH ({ctx['mmh_curr']:.2f})", "requires": ["ppo", "ppo_signal"]},
    {"key": "ppo_zero_up", "title": "🟢 PPO cross above 0", "check_fn": lambda ctx, ppo, ppo_sig, rsi: ctx["buy_common"] and (ppo["prev"] <= 0.0) and (ppo["curr"] > 0.0), "extra_fn": lambda ctx, ppo, ppo_sig, rsi, _: f"PPO {ppo['curr']:.2f} | MMH ({ctx['mmh_curr']:.2f})", "requires": ["ppo"]},
    {"key": "ppo_zero_down", "title": "🔴 PPO cross below 0", "check_fn": lambda ctx, ppo, ppo_sig, rsi: ctx["sell_common"] and (ppo["prev"] >= 0.0) and (ppo["curr"] < 0.0), "extra_fn": lambda ctx, ppo, ppo_sig, rsi, _: f"PPO {ppo['curr']:.2f} | MMH ({ctx['mmh_curr']:.2f})", "requires": ["ppo"]},
    {"key": "ppo_011_up", "title": "🟢 PPO cross above 0.11", "check_fn": lambda ctx, ppo, ppo_sig, rsi: (ctx["buy_common"] and (ppo["prev"] <= Constants.PPO_011_THRESHOLD) and (ppo["curr"] > Constants.PPO_011_THRESHOLD)), "extra_fn": lambda ctx, ppo, ppo_sig, rsi, _: f"PPO {ppo['curr']:.2f} | MMH ({ctx['mmh_curr']:.2f})", "requires": ["ppo"]},
    {"key": "ppo_011_down", "title": "🔴 PPO cross below -0.11", "check_fn": lambda ctx, ppo, ppo_sig, rsi: (ctx["sell_common"] and (ppo["prev"] >= Constants.PPO_011_THRESHOLD_SELL) and (ppo["curr"] < Constants.PPO_011_THRESHOLD_SELL)), "extra_fn": lambda ctx, ppo, ppo_sig, rsi, _: f"PPO {ppo['curr']:.2f} | MMH ({ctx['mmh_curr']:.2f})", "requires": ["ppo"]},
    {"key": "rsi_50_up", "title": "🟢 RSI cross above 50 (PPO < 0.30)", "check_fn": lambda ctx, ppo, ppo_sig, rsi: (ctx["buy_common"] and (rsi["prev"] <= Constants.RSI_THRESHOLD) and (rsi["curr"] > Constants.RSI_THRESHOLD) and (ppo["curr"] < Constants.PPO_RSI_GUARD_BUY)), "extra_fn": lambda ctx, ppo, ppo_sig, rsi, _: f"RSI {rsi['curr']:.2f} | PPO {ppo['curr']:.2f} | MMH ({ctx['mmh_curr']:.2f})", "requires": ["ppo", "rsi"]},
    {"key": "rsi_50_down", "title": "🔴 RSI cross below 50 (PPO > -0.30)", "check_fn": lambda ctx, ppo, ppo_sig, rsi: (ctx["sell_common"] and (rsi["prev"] >= Constants.RSI_THRESHOLD) and (rsi["curr"] < Constants.RSI_THRESHOLD) and (ppo["curr"] > Constants.PPO_RSI_GUARD_SELL)), "extra_fn": lambda ctx, ppo, ppo_sig, rsi, _: f"RSI {rsi['curr']:.2f} | PPO {ppo['curr']:.2f} | MMH ({ctx['mmh_curr']:.2f})", "requires": ["ppo", "rsi"]},
    {"key": "vwap_up", "title": "🔵▲ Price cross above VWAP", "check_fn": lambda ctx, ppo, ppo_sig, rsi: (ctx["buy_common"] and (ctx["close_prev"] <= ctx["vwap_prev"]) and (ctx["close_curr"] > ctx["vwap_curr"])), "extra_fn": lambda ctx, ppo, ppo_sig, rsi, _: f"MMH ({ctx['mmh_curr']:.2f})", "requires": []},
    {"key": "vwap_down", "title": "🟣▼ Price cross below VWAP", "check_fn": lambda ctx, ppo, ppo_sig, rsi: (ctx["sell_common"] and (ctx["close_prev"] >= ctx["vwap_prev"]) and (ctx["close_curr"] < ctx["vwap_curr"])), "extra_fn": lambda ctx, ppo, ppo_sig, rsi, _: f"MMH ({ctx['mmh_curr']:.2f})", "requires": []},
    {"key": "mmh_buy", "title": "🔵⬆️ MMH Reversal BUY", "check_fn": lambda ctx, ppo, ppo_sig, rsi: ctx["mmh_reversal_buy"], "extra_fn": lambda ctx, ppo, ppo_sig, rsi, _: f"MMH ({ctx['mmh_curr']:.2f})", "requires": []},
    {"key": "mmh_sell", "title": "🟣⬇️ MMH Reversal SELL", "check_fn": lambda ctx, ppo, ppo_sig, rsi: ctx["mmh_reversal_sell"], "extra_fn": lambda ctx, ppo, ppo_sig, rsi, _: f"MMH ({ctx['mmh_curr']:.2f})", "requires": []},
]

PIVOT_LEVELS = ["P", "S1", "S2", "S3", "R1", "R2", "R3"]
BUY_PIVOT_DEFS = [{"key": f"pivot_up_{level}", "title": f"🟢🔷 Cross above {level}", "check_fn": lambda ctx, ppo, ppo_sig, rsi, level=level: (ctx["buy_common"] and (ctx["close_prev"] <= ctx["pivots"][level]) and (ctx["close_curr"] > ctx["pivots"][level])), "extra_fn": lambda ctx, ppo, ppo_sig, rsi, _, level=level: f"${ctx['pivots'][level]:,.2f} | MMH ({ctx['mmh_curr']:.2f})", "requires": ["pivots"]} for level in ["P", "S1", "S2", "S3", "R1", "R2"]]
SELL_PIVOT_DEFS = [{"key": f"pivot_down_{level}", "title": f"🔴🔶 Cross below {level}", "check_fn": lambda ctx, ppo, ppo_sig, rsi, level=level: (ctx["sell_common"] and (ctx["close_prev"] >= ctx["pivots"][level]) and (ctx["close_curr"] < ctx["pivots"][level])), "extra_fn": lambda ctx, ppo, ppo_sig, rsi, _, level=level: f"${ctx['pivots'][level]:,.2f} | MMH ({ctx['mmh_curr']:.2f})", "requires": ["pivots"]} for level in ["P", "S1", "S2", "R1", "R2", "R3"]]
ALERT_DEFINITIONS.extend(BUY_PIVOT_DEFS)
ALERT_DEFINITIONS.extend(SELL_PIVOT_DEFS)

ALERT_KEYS: Dict[str, str] = {d["key"]: f"ALERT:{d['key'].upper()}" for d in ALERT_DEFINITIONS}
for level in PIVOT_LEVELS:
    ALERT_KEYS[f"pivot_up_{level}"] = f"ALERT:PIVOT_UP_{level}"
    ALERT_KEYS[f"pivot_down_{level}"] = f"ALERT:PIVOT_DOWN_{level}"

INDICATOR_CACHE: Dict[str, Dict[str, Any]] = {}

async def set_alert_state(sdb: RedisStateStore, pair: str, key: str, active: bool) -> None:
    if sdb.degraded:
        return
    state_key = f"{pair}:{key}"
    ts = int(time.time())
    state_val = "ACTIVE" if active else "INACTIVE"
    await sdb.set(state_key, state_val, ts)

async def was_alert_active(sdb: RedisStateStore, pair: str, key: str) -> bool:
    if sdb.degraded:
        return False
    state_key = f"{pair}:{key}"
    st = await sdb.get(state_key)
    return st is not None and st.get("state") == "ACTIVE"

async def check_multiple_alert_states(sdb: RedisStateStore, pair: str, keys: List[str]) -> Dict[str, bool]:
    if sdb.degraded or not keys:
        return {k: False for k in keys}
    state_keys = [f"{pair}:{k}" for k in keys]
    results = await sdb.mget_states(state_keys)
    output = {}
    for key in keys:
        state_key = f"{pair}:{key}"
        st = results.get(state_key)
        output[key] = st is not None and st.get("state") == "ACTIVE"
    return output

def compute_pivots_numpy(data_daily: Optional[Dict[str, np.ndarray]]) -> Dict[str, float]:
    piv: Dict[str, float] = {}
    if not cfg.ENABLE_PIVOT or data_daily is None:
        return piv
    ts = data_daily.get("timestamp"); high = data_daily.get("high"); low = data_daily.get("low"); close = data_daily.get("close")
    if ts is None or high is None or low is None or close is None or ts.size < 1:
        return piv
    today_utc = int(time.time()) // 86400
    yesterday_day = today_utc - 1
    days = ts // 86400
    idxs = np.where(days == yesterday_day)[0]
    if idxs.size == 0:
        return piv
    i = idxs[-1]
    H_prev = float(high[i]); L_prev = float(low[i]); C_prev = float(close[i])
    rng_prev = H_prev - L_prev
    if rng_prev <= 1e-8:
        return piv
    P = (H_prev + L_prev + C_prev) / 3.0
    piv = {
        "P": P,
        "R1": P + rng_prev * 0.382,
        "R2": P + rng_prev * 0.618,
        "R3": P + rng_prev,
        "S1": P - rng_prev * 0.382,
        "S2": P - rng_prev * 0.618,
        "S3": P - rng_prev,
    }
    return piv

async def evaluate_pair_and_alert(
    pair_name: str,
    data_15m: Dict[str, np.ndarray],
    data_5m: Dict[str, np.ndarray],
    data_daily: Optional[Dict[str, np.ndarray]],
    sdb: RedisStateStore,
    telegram_queue: TelegramQueue,
    correlation_id: str,
    reference_time: int
) -> Optional[Tuple[str, Dict[str, Any]]]:

    logger_pair = logging.getLogger(f"macd_bot.{pair_name}.{correlation_id}")
    PAIR_ID.set(pair_name)
    pair_start_time = time.time()

    try:
        ts15 = data_15m["timestamp"]; ts5 = data_5m["timestamp"]
        i15 = get_last_closed_index_from_array(ts15, 15, reference_time)
        i5 = get_last_closed_index_from_array(ts5, 5, reference_time)
        if i15 is None or i15 < 3 or i5 is None:
            logger_pair.warning(f"Insufficient closed candles for {pair_name}")
            return None

        cache_key = f"{pair_name}:{reference_time//900}"
        if cache_key in INDICATOR_CACHE:
            indicators = INDICATOR_CACHE[cache_key]
        else:
            gc.disable()
            indicators = calculate_all_indicators_sync(data_15m, data_5m, data_daily)
            gc.enable()
            INDICATOR_CACHE[cache_key] = indicators

        ppo = indicators['ppo']
        ppo_signal = indicators['ppo_signal']
        smooth_rsi = indicators['smooth_rsi']
        vwap = indicators['vwap']
        mmh = indicators['mmh']
        upw = indicators['upw']
        dnw = indicators['dnw']
        rma50_15_series = indicators['rma50_15']
        rma200_5_series = indicators['rma200_5']

        piv = compute_pivots_numpy(data_daily)

        close_curr = float(data_15m["close"][i15])
        close_prev = float(data_15m["close"][i15 - 1])
        ts_curr = int(data_15m["timestamp"][i15])

        ppo_curr = float(ppo[i15]); ppo_prev = float(ppo[i15 - 1])
        ppo_sig_curr = float(ppo_signal[i15]); ppo_sig_prev = float(ppo_signal[i15 - 1])

        rsi_curr = float(smooth_rsi[i15]); rsi_prev = float(smooth_rsi[i15 - 1])

        vwap_curr = float(vwap[i15]) if vwap.size else 0.0
        vwap_prev = float(vwap[i15 - 1]) if vwap.size else 0.0

        mmh_curr = float(mmh[i15])
        mmh_m1   = float(mmh[i15 - 1])

        cloud_up   = bool(upw[i15]) and not bool(dnw[i15])
        cloud_down = bool(dnw[i15]) and not bool(upw[i15])

        rma50_15 = float(rma50_15_series[i15])
        rma200_5 = float(rma200_5_series[i5])

        base_buy_common  = rma50_15 < close_curr and rma200_5 < close_curr
        base_sell_common = rma50_15 > close_curr and rma200_5 > close_curr

        if base_buy_common:
            base_buy_common = base_buy_common and (mmh_curr > 0 and cloud_up)
        if base_sell_common:
            base_sell_common = base_sell_common and (mmh_curr < 0 and cloud_down)

        buy_candle_passed,  buy_candle_reason  = check_candle_quality_with_reason_arrays(
            data_15m["open"], data_15m["high"], data_15m["low"], data_15m["close"], i15, is_buy=True
        )
        sell_candle_passed, sell_candle_reason = check_candle_quality_with_reason_arrays(
            data_15m["open"], data_15m["high"], data_15m["low"], data_15m["close"], i15, is_buy=False
        )

        buy_common  = base_buy_common  and buy_candle_passed
        sell_common = base_sell_common and sell_candle_passed

        mmh_reversal_buy  = False
        mmh_reversal_sell = False

        if i15 >= 3:
            mmh_m3 = float(mmh[i15 - 3])
            mmh_m2 = float(mmh[i15 - 2])
            mmh_reversal_buy = (
                buy_common and mmh_curr > 0 and mmh_m3 > mmh_m2 > mmh_m1 and mmh_curr > mmh_m1
            )
            mmh_reversal_sell = (
                sell_common and mmh_curr < 0 and mmh_m3 < mmh_m2 < mmh_m1 and mmh_curr < mmh_m1
            )

        context = {
            "buy_common": buy_common,
            "sell_common": sell_common,
            "close_curr": close_curr,
            "close_prev": close_prev,
            "ts_curr": ts_curr,
            "ppo_curr": ppo_curr,
            "ppo_prev": ppo_prev,
            "ppo_sig_curr": ppo_sig_curr,
            "ppo_sig_prev": ppo_sig_prev,
            "rsi_curr": rsi_curr,
            "rsi_prev": rsi_prev,
            "vwap_curr": vwap_curr,
            "vwap_prev": vwap_prev,
            "mmh_curr": mmh_curr,
            "mmh_m1": mmh_m1,
            "mmh_reversal_buy": mmh_reversal_buy,
            "mmh_reversal_sell": mmh_reversal_sell,
            "pivots": piv,
            "vwap": vwap.size > 0,
            "candle_quality_failed_buy": base_buy_common and not buy_candle_passed,
            "candle_quality_failed_sell": base_sell_common and not sell_candle_passed,
            "candle_rejection_reason_buy": buy_candle_reason if (base_buy_common and not buy_candle_passed) else None,
            "candle_rejection_reason_sell": sell_candle_reason if (base_sell_common and not sell_candle_passed) else None,
        }

        ppo_ctx = {"curr": context["ppo_curr"], "prev": context["ppo_prev"]}
        ppo_sig_ctx = {"curr": context["ppo_sig_curr"], "prev": context["ppo_sig_prev"]}
        rsi_ctx = {"curr": context["rsi_curr"], "prev": context["rsi_prev"]}

        raw_alerts: List[Tuple[str, str, str]] = []

        alert_keys_to_check = []
        for def_ in ALERT_DEFINITIONS:
            if "pivots" in def_["requires"] and not context.get("pivots"):
                continue
            if "vwap" in def_["requires"] and not context.get("vwap"):
                continue
            alert_keys_to_check.append(def_["key"])

        previous_states = await check_multiple_alert_states(
            sdb, pair_name, [ALERT_KEYS[k] for k in alert_keys_to_check]
        )

        states_to_update = []
        for def_ in ALERT_DEFINITIONS:
            if "pivots" in def_["requires"] and not context.get("pivots"):
                continue
            if "vwap" in def_["requires"] and not context.get("vwap"):
                continue
            try:
                key = ALERT_KEYS[def_["key"]]
                if def_["check_fn"](context, ppo_ctx, ppo_sig_ctx, rsi_ctx):
                    if not previous_states.get(key, False):
                        extra = def_["extra_fn"](context, ppo_ctx, ppo_sig_ctx, rsi_ctx, None)
                        raw_alerts.append((def_["title"], extra, def_["key"]))
                        states_to_update.append((f"{pair_name}:{key}", "ACTIVE", None))
            except Exception as e:
                logger_pair.warning(f"Alert check failed for {pair_name}, key={def_['key']}: {e}")

        if states_to_update:
            await sdb.batch_set_states(states_to_update)

        resets_to_apply = []

        if ppo_prev > ppo_sig_prev and ppo_curr <= ppo_sig_curr:
            resets_to_apply.append((f"{pair_name}:{ALERT_KEYS['ppo_signal_up']}", "INACTIVE", None))
        if ppo_prev < ppo_sig_prev and ppo_curr >= ppo_sig_curr:
            resets_to_apply.append((f"{pair_name}:{ALERT_KEYS['ppo_signal_down']}", "INACTIVE", None))

        if ppo_prev > 0 and ppo_curr <= 0:
            resets_to_apply.append((f"{pair_name}:{ALERT_KEYS['ppo_zero_up']}", "INACTIVE", None))
        if ppo_prev < 0 and ppo_curr >= 0:
            resets_to_apply.append((f"{pair_name}:{ALERT_KEYS['ppo_zero_down']}", "INACTIVE", None))

        if ppo_prev > Constants.PPO_011_THRESHOLD and ppo_curr <= Constants.PPO_011_THRESHOLD:
            resets_to_apply.append((f"{pair_name}:{ALERT_KEYS['ppo_011_up']}", "INACTIVE", None))
        if ppo_prev < Constants.PPO_011_THRESHOLD_SELL and ppo_curr >= Constants.PPO_011_THRESHOLD_SELL:
            resets_to_apply.append((f"{pair_name}:{ALERT_KEYS['ppo_011_down']}", "INACTIVE", None))

        if rsi_prev > Constants.RSI_THRESHOLD and rsi_curr <= Constants.RSI_THRESHOLD:
            resets_to_apply.append((f"{pair_name}:{ALERT_KEYS['rsi_50_up']}", "INACTIVE", None))
        if rsi_prev < Constants.RSI_THRESHOLD and rsi_curr >= Constants.RSI_THRESHOLD:
            resets_to_apply.append((f"{pair_name}:{ALERT_KEYS['rsi_50_down']}", "INACTIVE", None))

        if context["vwap"]:
            if close_prev > vwap_prev and close_curr <= vwap_curr:
                resets_to_apply.append((f"{pair_name}:{ALERT_KEYS['vwap_up']}", "INACTIVE", None))
            if close_prev < vwap_prev and close_curr >= vwap_curr:
                resets_to_apply.append((f"{pair_name}:{ALERT_KEYS['vwap_down']}", "INACTIVE", None))

        if piv:
            for level_name, level_value in piv.items():
                if close_prev > level_value and close_curr <= level_value:
                    resets_to_apply.append((f"{pair_name}:{ALERT_KEYS[f'pivot_up_{level_name}']}", "INACTIVE", None))
                if close_prev < level_value and close_curr >= level_value:
                    resets_to_apply.append((f"{pair_name}:{ALERT_KEYS[f'pivot_down_{level_name}']}", "INACTIVE", None))

        if (mmh_curr > 0) and (mmh_curr <= mmh_m1):
            if await was_alert_active(sdb, pair_name, ALERT_KEYS["mmh_buy"]):
                resets_to_apply.append((f"{pair_name}:{ALERT_KEYS['mmh_buy']}", "INACTIVE", None))
        if (mmh_curr < 0) and (mmh_curr >= mmh_m1):
            if await was_alert_active(sdb, pair_name, ALERT_KEYS["mmh_sell"]):
                resets_to_apply.append((f"{pair_name}:{ALERT_KEYS['mmh_sell']}", "INACTIVE", None))

        all_state_changes = states_to_update + resets_to_apply
        if all_state_changes:
            await sdb.batch_set_states(all_state_changes)

        if raw_alerts:
            dedup_checks = [(pair_name, alert_key, ts_curr) for _, _, alert_key in raw_alerts]
            dedup_results = await sdb.batch_check_recent_alerts(dedup_checks)
            alerts_to_send = []
            for title, extra, alert_key in raw_alerts:
                composite_key = f"{pair_name}:{alert_key}"
                if dedup_results.get(composite_key, True):
                    alerts_to_send.append((title, extra, alert_key))
                else:
                    logger_pair.debug(f"Skipping duplicate alert: {composite_key}")
            alerts_to_send = alerts_to_send[:cfg.MAX_ALERTS_PER_PAIR]
        else:
            alerts_to_send = []

        if alerts_to_send:
            if len(alerts_to_send) == 1:
                title, extra, _ = alerts_to_send[0]
                msg = build_single_msg(title, pair_name, close_curr, ts_curr, extra)
            else:
                items = [(title, extra) for title, extra, _ in alerts_to_send[:25]]
                msg = build_batched_msg(pair_name, close_curr, ts_curr, items)
            await telegram_queue.send(msg)
            new_state = {
                "state": "ALERT_SENT",
                "ts": int(time.time()),
                "summary": {"alerts": len(alerts_to_send)}
            }
            logger_pair.info(f"✅ Sent {len(alerts_to_send)} alerts for {pair_name}: "
                             f"{[ak for _, _, ak in alerts_to_send]}")
        else:
            new_state = {"state": "NO_SIGNAL", "ts": int(time.time())}

        cloud = "green" if cloud_up else ("red" if cloud_down else "neutral")
        reasons = []
        if not context["buy_common"] and not context["sell_common"]:
            reasons.append("Trend filter blocked")
        if context.get("candle_quality_failed_buy"):
            buy_reason = context.get("candle_rejection_reason_buy", "Unknown")
            reasons.append(f"BUY candle quality: {buy_reason}")
        if context.get("candle_quality_failed_sell"):
            sell_reason = context.get("candle_rejection_reason_sell", "Unknown")
            reasons.append(f"SELL candle quality: {sell_reason}")
        suppression_reason = "; ".join(reasons) if reasons else "No conditions met"

        new_state["summary"] = {
            "cloud": cloud,
            "mmh_hist": round(mmh_curr, 4),
            "suppression": suppression_reason,
            "candle_quality": {
                "buy_passed": not context.get("candle_quality_failed_buy", False),
                "sell_passed": not context.get("candle_quality_failed_sell", False),
                "buy_reason": context.get("candle_rejection_reason_buy"),
                "sell_reason": context.get("candle_rejection_reason_sell"),
            }
        }

        status_msg = f"✓ {pair_name} | cloud={cloud} mmh={mmh_curr:.2f}"
        if alerts_to_send:
            status_msg += f" | 🔔 {len(alerts_to_send)} alerts sent"
        elif base_buy_common and not buy_candle_passed:
            status_msg += f" | BUY blocked: {buy_candle_reason}"
        elif base_sell_common and not sell_candle_passed:
            status_msg += f" | SELL blocked: {sell_candle_reason}"
        else:
            status_msg += " | No signals"
        logger_pair.info(status_msg)

        if cfg.DEBUG_MODE:
            logger_pair.debug(f"Pair total: {time.time() - pair_start_time:.2f}s")

        return pair_name, new_state

    except Exception as e:
        logger_pair.exception(f"❌ Error in evaluate_pair_and_alert for {pair_name}: {e}")
        return None
    finally:
        try:
            if 'ppo' in locals():
                del ppo, ppo_signal, smooth_rsi, vwap, mmh
            if cfg.CIRRUS_CLOUD_ENABLED and 'upw' in locals():
                del upw, dnw
        except Exception as e:
            logger.warning(f"Cleanup error (non-critical): {e}")
        finally:
            PAIR_ID.set("")

async def check_pair(
    pair_name: str,
    fetcher: DataFetcher,
    products_map: Dict[str, dict],
    state_db: RedisStateStore,
    telegram_queue: TelegramQueue,
    correlation_id: str,
    reference_time: int
) -> Optional[Tuple[str, Dict[str, Any]]]:
    try:
        if shutdown_event.is_set():
            return None

        product_info = products_map.get(pair_name)
        if not product_info:
            logger.warning(f"Product info not found for {pair_name}")
            return None
        symbol = product_info["symbol"]

        daily_limit = cfg.PIVOT_LOOKBACK_PERIOD + 10

        tasks = [
            fetcher.fetch_candles(symbol, "15", 300, reference_time),
            fetcher.fetch_candles(symbol, "5", 400, reference_time),
            fetcher.fetch_candles(symbol, "D", daily_limit, reference_time) if cfg.ENABLE_PIVOT else asyncio.sleep(0),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        data_15m = parse_candles_to_numpy(results[0] if not isinstance(results[0], Exception) else None)
        data_5m  = parse_candles_to_numpy(results[1] if not isinstance(results[1], Exception) else None)
        data_daily = parse_candles_to_numpy(results[2]) if (cfg.ENABLE_PIVOT and not isinstance(results[2], Exception)) else None

        valid_15m, reason_15m = validate_candle_arrays(data_15m, required_len=220)
        valid_5m,  reason_5m  = validate_candle_arrays(data_5m, required_len=280)

        if not valid_15m or not valid_5m:
            logger.warning(f"Insufficient data for {pair_name}: 15m={reason_15m}, 5m={reason_5m}")
            return None

        return await evaluate_pair_and_alert(
            pair_name, data_15m, data_5m, data_daily, state_db,
            telegram_queue, correlation_id, reference_time
        )
    except Exception as e:
        logger.exception(f"Error in check_pair for {pair_name}: {e}")
        return None

async def worker_process_pair(
    worker_id: int,
    pair_queue: asyncio.Queue,
    fetcher: DataFetcher,
    products_map: Dict[str, dict],
    state_db: RedisStateStore,
    telegram_queue: TelegramQueue,
    correlation_id: str,
    reference_time: int,
    lock: RedisLock,
    results: List[Tuple[str, Dict[str, Any]]],
    results_lock: asyncio.Lock
) -> None:
    logger_worker = logging.getLogger(f"macd_bot.worker_{worker_id}")
    while True:
        try:
            try:
                pair_name = await asyncio.wait_for(pair_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                if pair_queue.empty():
                    break
                continue

            if pair_name is None:
                pair_queue.task_done()
                break
            if shutdown_event.is_set():
                pair_queue.task_done()
                break

            logger_worker.debug(f"Worker {worker_id} processing {pair_name}")
            try:
                result = await check_pair(
                    pair_name, fetcher, products_map, state_db,
                    telegram_queue, correlation_id, reference_time
                )
                if result:
                    async with results_lock:
                        results.append(result)
                    if cfg.DEBUG_MODE:
                        pair, state = result
                        summary = state.get("summary", {})
                        logger_worker.debug(
                            f"Worker {worker_id} completed {pair} | "
                            f"cloud={summary.get('cloud','n/a')} | "
                            f"mmh_hist={summary.get('mmh_hist','n/a')}"
                        )
                else:
                    if cfg.DEBUG_MODE:
                        logger_worker.debug(f"Worker {worker_id}: {pair_name} returned None")
            except Exception as e:
                logger_worker.error(f"Worker {worker_id} error processing {pair_name}: {e}")

            if lock.should_extend():
                if not await lock.extend(timeout=3.0):
                    logger_worker.error(f"Worker {worker_id}: Failed to extend Redis lock")
                    pair_queue.task_done()
                    break

            pair_queue.task_done()

        except asyncio.CancelledError:
            logger_worker.debug(f"Worker {worker_id} cancelled")
            break
        except Exception as e:
            logger_worker.error(f"Worker {worker_id} unexpected error: {e}")
            break

    logger_worker.debug(f"Worker {worker_id} exiting")

async def process_pairs_with_workers(
    fetcher: DataFetcher,
    products_map: Dict[str, dict],
    pairs_to_process: List[str],
    state_db: RedisStateStore,
    telegram_queue: TelegramQueue,
    correlation_id: str,
    lock: RedisLock,
    reference_time: int
) -> List[Tuple[str, Dict[str, Any]]]:
    logger_main = logging.getLogger("macd_bot.worker_pool")

    pair_queue: asyncio.Queue = asyncio.Queue()
    for pair in pairs_to_process:
        await pair_queue.put(pair)

    results: List[Tuple[str, Dict[str, Any]]] = []
    results_lock = asyncio.Lock()

    if len(pairs_to_process) <= cfg.MAX_PARALLEL_FETCH:
        num_workers = max(2, len(pairs_to_process) // 2) if len(pairs_to_process) > 1 else 1
    else:
        num_workers = cfg.MAX_PARALLEL_FETCH
    num_workers = min(num_workers, len(pairs_to_process)) if pairs_to_process else 1

    logger_main.info(f"📊 Processing {len(pairs_to_process)} pairs with {num_workers}-worker pool")

    workers = []
    for worker_id in range(num_workers):
        worker = asyncio.create_task(
            worker_process_pair(
                worker_id, pair_queue, fetcher, products_map,
                state_db, telegram_queue, correlation_id,
                reference_time, lock, results, results_lock
            )
        )
        workers.append(worker)

    await pair_queue.join()

    for _ in range(num_workers):
        await pair_queue.put(None)

    await asyncio.gather(*workers, return_exceptions=True)
    return results

def warmup_numba() -> None:
    logger.info("🔥 Warming up Numba JIT compiler...")
    try:
        length = 500
        close = np.random.random(length).astype(np.float64) * 1000
        sd    = np.random.random(length).astype(np.float64) * 0.01
        ts    = np.arange(length, dtype=np.int64)
        vol   = np.random.random(length).astype(np.float64) * 100
        _calc_mmh_worm_loop(close, sd, length)
        _calc_mmh_value_loop(np.random.random(length).astype(np.float64), length)
        _calc_mmh_momentum_loop(np.random.random(length).astype(np.float64), length)
        _sma_loop(close, 50)
        _ema_loop(close, 0.1)
        _kalman_loop(close, 21, 0.01, 0.1)
        _rng_filter_loop(close, sd)
        _vwap_daily_loop(close, close, close, vol, ts)
        logger.info("✅ Numba warm-up complete.")
    except Exception as e:
        logger.warning(f"Numba warm-up failed (non-fatal): {e}")

async def run_once() -> bool:
    correlation_id = uuid.uuid4().hex[:8]
    TRACE_ID.set(correlation_id)
    logger_run = logging.getLogger(f"macd_bot.run.{correlation_id}")
    start_time = time.time()

    reference_time = get_trigger_timestamp()
    logger_run.info(f"Using reference timestamp: {reference_time} ({format_ist_time(reference_time)})")

    lock_acquired = False
    lock: Optional[RedisLock] = None

    try:
        process = psutil.Process()
        container_memory_mb = process.memory_info().rss / 1024 / 1024
        limit_mb = cfg.MEMORY_LIMIT_BYTES / 1024 / 1024

        if container_memory_mb >= limit_mb:
            logger_run.critical(f"🚨 Memory limit exceeded at startup ({container_memory_mb:.1f}MB / {limit_mb:.1f}MB)")
            return False

        sdb = RedisStateStore(cfg.REDIS_URL)
        await sdb.connect()
        logger_run.debug("Redis connection established")

        try:
            if sdb.degraded and not sdb.degraded_alerted:
                logger_run.critical("⚠️ Redis is in degraded mode – alert deduplication disabled!")
                telegram_queue = TelegramQueue(cfg.TELEGRAM_BOT_TOKEN, cfg.TELEGRAM_CHAT_ID)
                await telegram_queue.send(escape_markdown_v2(
                    f"⚠️ {cfg.BOT_NAME} - REDIS DEGRADED MODE\n"
                    f"Alert deduplication is disabled. You may receive duplicate alerts.\n"
                    f"Time: {format_ist_time()}"
                ))
                sdb.degraded_alerted = True

            fetcher = DataFetcher(cfg.DELTA_API_BASE)
            telegram_queue = TelegramQueue(cfg.TELEGRAM_BOT_TOKEN, cfg.TELEGRAM_CHAT_ID)

            lock = RedisLock(sdb._redis, "macd_bot_run")
            lock_acquired = await lock.acquire(timeout=5.0)
            if not lock_acquired:
                logger_run.warning("❌ Another instance is running (Redis lock held)")
                return False

            try:
                if cfg.SEND_TEST_MESSAGE:
                    await telegram_queue.send(escape_markdown_v2(
                        f"🚀 Unified Bot - Run Started\n"
                        f"Date : {format_ist_time(datetime.now(timezone.utc))}\n"
                        f"Corr. ID: {correlation_id}"
                    ))

                PRODUCTS_CACHE = getattr(run_once, '_products_cache', {"data": None, "until": 0.0})
                now = time.time()
                if PRODUCTS_CACHE["data"] is None or now > PRODUCTS_CACHE["until"]:
                    logger_run.info("Fetching fresh products list from Delta API...")
                    prod_resp = await fetcher.fetch_products()
                    if not prod_resp:
                        logger_run.error("Failed to fetch products map")
                        return False
                    PRODUCTS_CACHE["data"] = prod_resp
                    PRODUCTS_CACHE["until"] = now + 28_800  # 8 hours
                    run_once._products_cache = PRODUCTS_CACHE
                else:
                    logger_run.debug("Using cached products list")
                    prod_resp = PRODUCTS_CACHE["data"]

                products_map = build_products_map_from_api_result(prod_resp)
                pairs_to_process = [p for p in cfg.PAIRS if p in products_map]

                if len(pairs_to_process) < len(cfg.PAIRS):
                    missing = set(cfg.PAIRS) - set(pairs_to_process)
                    logger_run.warning(f"Missing products for pairs: {missing}")

                logger_run.info(f"📊 Processing {len(pairs_to_process)} pairs using WORKER POOL architecture")

                all_results = await process_pairs_with_workers(
                    fetcher, products_map, pairs_to_process,
                    sdb, telegram_queue, correlation_id,
                    lock, reference_time
                )

                fetcher_stats = fetcher.get_stats()
                logger_run.info(
                    f"📡 Fetch stats | "
                    f"Products: {fetcher_stats['products_success']}✅/{fetcher_stats['products_failed']}❌ | "
                    f"Candles: {fetcher_stats['candles_success']}✅/{fetcher_stats['candles_failed']}❌"
                )

                alerts_sent = 0
                for _, state in all_results:
                    summary = state.get("summary", {})
                    alerts_sent += int(summary.get("alerts", 0))

                run_duration = time.time() - start_time
                container_memory_mb = process.memory_info().rss / 1024 / 1024
                redis_status = "OK" if not sdb.degraded else "DEGRADED"
                summary = (
                    f"✅ RUN COMPLETE | {run_duration:.1f}s | {len(all_results)} pairs | "
                    f"{alerts_sent} alerts | Mem: {int(container_memory_mb)}MB | Redis: {redis_status}"
                )
                logger_run.info(summary)

                MAX_ALERTS_PER_RUN = 50
                if alerts_sent > MAX_ALERTS_PER_RUN:
                    await telegram_queue.send(escape_markdown_v2(
                        f"⚠️ HIGH VOLUME: {alerts_sent} alerts | Pairs: {len(all_results)}"
                    ))

                return True

            except asyncio.TimeoutError:
                logger_run.error("Run timed out")
                return False
            except Exception as e:
                logger_run.exception(f"Error within Redis context: {e}")
                raise
            finally:
                if lock_acquired and lock and lock.acquired_by_me:
                    await lock.release(timeout=3.0)
                else:
                    logger_run.debug("Lock not acquired by this instance, skipping release")

        finally:
            await sdb.close()

    except asyncio.CancelledError:
        logger_run.info("Run cancelled (Task Cancellation)")
        return False
    except Exception as e:
        logger_run.exception(f"❌ Fatal error in run_once: {e}")
        try:
            await cancel_all_tasks(grace_seconds=5)
        except Exception:
            pass
        return False
    finally:
        try:
            if 'all_results' in locals():
                del all_results
            if 'products_map' in locals():
                del products_map
            if 'fetcher' in locals():
                del fetcher
            gc.collect()
            logger_run.debug("Final cleanup completed")
        except Exception as e:
            logger_run.warning(f"Final cleanup error (non-critical): {e}")
        await SessionManager.close_session()
        TRACE_ID.set("")

# -----------------------------
# Part 5 helpers (advanced performance & consistency)
# -----------------------------

GLOBAL_BUF_SIZE = 600
GLOBAL_CLOSE_BUF = np.empty(GLOBAL_BUF_SIZE, dtype=np.float64)
GLOBAL_DIFF_BUF  = np.empty(GLOBAL_BUF_SIZE, dtype=np.float64)
GLOBAL_SD_BUF    = np.empty(GLOBAL_BUF_SIZE, dtype=np.float64)
GLOBAL_VWAP_BUF  = np.empty(GLOBAL_BUF_SIZE, dtype=np.float64)
GLOBAL_TEMP_BUF  = np.empty(GLOBAL_BUF_SIZE, dtype=np.float64)
GLOBAL_MIN_BUF   = np.empty(GLOBAL_BUF_SIZE, dtype=np.float64)
GLOBAL_MAX_BUF   = np.empty(GLOBAL_BUF_SIZE, dtype=np.float64)

def _copy_into_buf(src: np.ndarray, buf: np.ndarray) -> np.ndarray:
    n = src.size
    if n <= buf.size:
        buf[:n] = src
        return buf[:n]
    return src

def arr_get(a: Dict[str, np.ndarray], key: str, default_len: int = 0, dtype=float) -> np.ndarray:
    v = a.get(key)
    if v is None:
        return np.zeros(default_len, dtype=dtype)
    return v

def safe_float(x: Any, fallback: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return fallback

def ensure_bool_array(arr: np.ndarray, length: int) -> np.ndarray:
    if arr is None or arr.size == 0:
        return np.zeros(length, dtype=bool)
    return arr.astype(bool)

def _validate_alert_keys_on_import():
    missing = []
    for d in ALERT_DEFINITIONS:
        k = d["key"]
        if k not in ALERT_KEYS:
            missing.append(k)
    if missing:
        logger.warning(f"Alert keys missing for definitions: {missing}")

_validate_alert_keys_on_import()

def coerce_data_types(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out = {}
    for k, v in data.items():
        if k == "timestamp":
            out[k] = v.astype(np.int64, copy=False)
        else:
            out[k] = v.astype(np.float64, copy=False)
    return out

# -----------------------------
# uvloop and CLI entrypoint
# -----------------------------

try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    logger.info(f"✅ uvloop enabled | {JSON_BACKEND} enabled")
except ImportError:
    logger.info(f"ℹ️ uvloop not available (using default) | {JSON_BACKEND} enabled")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="macd_unified", description="Unified MACD/alerts runner")
    parser.add_argument("--debug", action="store_true", help="Set logger to DEBUG")
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
        for h in logger.handlers:
            h.setLevel(logging.DEBUG)
        logger.info("Debug mode enabled via CLI flag")

    warmup_numba()

    try:
        success = asyncio.run(run_once())
        if success:
            logger.info("✅ Bot run completed successfully")
            sys.exit(0)
        else:
            logger.error("❌ Bot run failed")
            sys.exit(1)
    except (asyncio.CancelledError, KeyboardInterrupt):
        logger.info("Bot stopped by timeout or user")
        sys.exit(130)
    except Exception as exc:
        logger.critical(f"Fatal error: {exc}", exc_info=True)
        sys.exit(1)
