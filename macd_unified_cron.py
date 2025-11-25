# macd_unified_cron.py
from __future__ import annotations
import os
import sys
import json
import time
import asyncio
import random
import logging
import logging.handlers
import gc
import ssl
import signal
import hashlib
import re
import uuid
import argparse
import atexit
import psutil
from collections import deque
from typing import Dict, Any, Optional, Tuple, List, ClassVar, TypedDict, Callable
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
from contextvars import ContextVar

import aiohttp
from aiohttp import web
import pandas as pd
import numpy as np
import redis.asyncio as redis
from redis.exceptions import ConnectionError as RedisConnectionError, RedisError
from pydantic import BaseModel, Field, field_validator, model_validator
from aiohttp import ClientConnectorError, ClientResponseError, TCPConnector, ClientError

# ---------- Prometheus optional ----------
PROMETHEUS_ENABLED = os.getenv("PROMETHEUS_ENABLED", "false").lower() in ("1", "true", "yes")
try:
    if PROMETHEUS_ENABLED:
        from prometheus_client import Counter, Gauge, Histogram, start_http_server
    else:
        Counter = Gauge = Histogram = None
except Exception:
    PROMETHEUS_ENABLED = False
    Counter = Gauge = Histogram = None

# ---------- Version & Constants ----------
__version__ = "1.0.5-production"

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
    HEALTH_CHECK_PORT = int(os.getenv("PORT", "10000"))  # Prometheus
    REDIS_LOCK_EXPIRY = max(int(os.getenv('REDIS_LOCK_EXPIRY', 1200)), 1200)
    CIRCUIT_BREAKER_MAX_WAIT = 300
    MEMORY_CHECK_INTERVAL = 30
    MEMORY_LIMIT_PERCENT = float(os.getenv("MEMORY_LIMIT_PERCENT", 85.0))

# Optional health HTTP server (separate from Prometheus)
HEALTH_SERVER_ENABLED = os.getenv("HEALTH_SERVER_ENABLED", "false").lower() in ("1", "true", "yes")
HEALTH_HTTP_PORT = int(os.getenv("HEALTH_HTTP_PORT", "10001"))

# ---------- Context vars ----------
TRACE_ID: ContextVar[str] = ContextVar("trace_id", default="")
PAIR_ID: ContextVar[str] = ContextVar("pair_id", default="")

# ---------- IST helper ----------
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

# ---------- Pydantic config ----------
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
    MEMORY_LIMIT_BYTES: int = 400_000_000
    STATE_EXPIRY_DAYS: int = 30
    ENABLE_DMS_PLUGIN: bool = False
    ENABLE_HEALTH_TRACKER: bool = True
    DEAD_MANS_COOLDOWN_SECONDS: int = 14400
    LOG_LEVEL: str = "INFO"
    ENABLE_VWAP: bool = True
    ENABLE_PIVOT: bool = True
    PIVOT_LOOKBACK_PERIOD: int = 15
    FAIL_ON_REDIS_DOWN: bool = False
    FAIL_ON_TELEGRAM_DOWN: bool = False
    TELEGRAM_RATE_LIMIT_PER_MINUTE: int = 20
    TELEGRAM_BURST_SIZE: int = 5
    ENABLE_AUTO_RESTART: bool = False
    AUTO_RESTART_MAX_RETRIES: int = 3
    AUTO_RESTART_COOLDOWN_SEC: int = 10

    @field_validator('LOG_LEVEL')
    def validate_log_level(cls, v: str) -> str:
        valid = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid:
            raise ValueError(f'LOG_LEVEL must be one of {valid}')
        return v.upper()

    @field_validator('PAIRS')
    def validate_pairs(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError('PAIRS cannot be empty')
        return v

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
        if self.RUN_TIMEOUT_SECONDS >= Constants.REDIS_LOCK_EXPIRY:
            raise ValueError(
                f'REDIS_LOCK_EXPIRY ({Constants.REDIS_LOCK_EXPIRY}s) must be greater than '
                f'RUN_TIMEOUT_SECONDS ({self.RUN_TIMEOUT_SECONDS}s). '
                f'Increase REDIS_LOCK_EXPIRY env var to at least {self.RUN_TIMEOUT_SECONDS + 300}.'
            )
        return self

# ---------- Config loader ----------
def load_config() -> BotConfig:
    config_file = os.getenv("CONFIG_FILE", "config_macd.json")
    data: Dict[str, Any] = {}
    if Path(config_file).exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error parsing config file: {e}")
            sys.exit(1)
    else:
        # In CI (GitHub Actions), we expect env vars to be provided;
        # minimal placeholders to pass validation if file not present.
        for key in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "REDIS_URL", "DELTA_API_BASE"):
            data[key] = os.getenv(key)
            if not data[key]:
                print(f"Missing required config: {key}")
                sys.exit(1)

    # Overlay env onto file config
    for key in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "REDIS_URL", "DELTA_API_BASE"):
        data[key] = os.getenv(key, data.get(key))
        if not data[key]:
            raise ValueError(f"Missing required config: {key}")

    try:
        return BotConfig(**data)
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        sys.exit(1)

cfg = load_config()

# ---------- Structured logging ----------
class SecretFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = str(record.getMessage())
            msg = re.sub(r'\b\d{6,}:[A-Za-z0-9_-]{20,}\b', '[REDACTED_TELEGRAM_TOKEN]', msg)
            msg = re.sub(r'chat_id=\d+', '[REDACTED_CHAT_ID]', msg)
            record.msg = msg
        except Exception:
            pass
        return True

class TraceContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = TRACE_ID.get()
        record.pair_id = PAIR_ID.get()
        return True

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = {
            "ts": datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] + 'Z',
            "level": record.levelname,
            "logger": record.name,
            "fn": f"{record.funcName}:{record.lineno}",
            "msg": record.getMessage(),
        }
        if getattr(record, "trace_id", ""):
            base["trace_id"] = record.trace_id
        if getattr(record, "pair_id", ""):
            base["pair_id"] = record.pair_id
        if record.exc_info:
            base["exc"] = self.formatException(record.exc_info)
        return json.dumps(base, ensure_ascii=False)

def setup_logging() -> logging.Logger:
    logger = logging.getLogger("macd_bot")
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    level = logging.DEBUG if cfg.DEBUG_MODE else getattr(logging, cfg.LOG_LEVEL, logging.INFO)
    logger.setLevel(level)
    logger.propagate = False
    use_json = os.getenv("LOG_JSON", "false").lower() in ("1", "true", "yes")
    formatter = JsonFormatter() if use_json else logging.Formatter(
        fmt='%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    console.addFilter(SecretFilter())
    console.addFilter(TraceContextFilter())
    logger.addHandler(console)

    # Optional file logging (off by default in CI to avoid artifacts)
    if os.getenv("FILE_LOGGING", "false").lower() in ("1", "true", "yes"):
        try:
            file_handler = logging.handlers.TimedRotatingFileHandler(
                filename=cfg.LOG_FILE, when="midnight", interval=1, backupCount=7, encoding="utf-8", utc=True
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            file_handler.addFilter(SecretFilter())
            file_handler.addFilter(TraceContextFilter())
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Failed to setup file logging: {e}")
    return logger

logger = setup_logging()
logger.info(f"Using DELTA_API_BASE: {cfg.DELTA_API_BASE}")

# ---------- Global shutdown ----------
shutdown_event = asyncio.Event()
def _sync_signal_handler(sig: int, frame: Any) -> None:
    try:
        logger.warning(f"Received signal {sig}, initiating async shutdown...")
    except Exception:
        pass
    try:
        loop = asyncio.get_running_loop()
        loop.call_soon_threadsafe(shutdown_event.set)
    except RuntimeError:
        try:
            shutdown_event.set()
        except Exception:
            pass

signal.signal(signal.SIGTERM, _sync_signal_handler)
signal.signal(signal.SIGINT, _sync_signal_handler)

def _atexit_cleanup() -> None:
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(SessionManager.close_session())
        loop.close()
    except Exception:
        pass
atexit.register(_atexit_cleanup)

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

# ---------- Memory guard ----------
class MemoryMonitor:
    def __init__(self, limit_percent: float = Constants.MEMORY_LIMIT_PERCENT):
        self.limit_percent = limit_percent
        self.last_check = 0.0
        self.interval = Constants.MEMORY_CHECK_INTERVAL

    def should_check(self) -> bool:
        now = time.time()
        if now - self.last_check >= self.interval:
            self.last_check = now
            return True
        return False

    def is_critical(self) -> bool:
        return psutil.virtual_memory().percent >= self.limit_percent

    def check_memory(self) -> Tuple[int, float]:
        vm = psutil.virtual_memory()
        return int(vm.used), float(vm.percent)

# ---------- Prometheus ----------
metrics_started = False
if PROMETHEUS_ENABLED and Counter and Gauge and Histogram:
    try:
        METRIC_ALERTS_SENT = Counter("bot_alerts_sent_total", "Total alerts sent")
        METRIC_FETCH_ERRORS = Counter("bot_fetch_errors_total", "Total fetch errors")
        METRIC_FAILED_PAIRS = Counter("bot_failed_pairs_total", "Total number of failed pairs")
        METRIC_RUN_DURATION = Histogram("bot_run_duration_seconds", "Run duration in seconds")
        METRIC_MEMORY_USAGE = Gauge("bot_memory_usage_mb", "Memory usage in MB")
        METRIC_REDIS_LOCK_FAILS = Counter("bot_redis_lock_extend_failures_total", "Redis lock extend failures")
        METRIC_REDIS_MEMORY_USED = Gauge("redis_used_memory_bytes", "Redis used_memory from INFO")
        METRIC_REDIS_KEYS = Gauge("redis_keys_total", "Total keys in Redis (db0)")
        start_http_server(Constants.HEALTH_CHECK_PORT)
        metrics_started = True
        logger.info(f"Prometheus metrics server started on port {Constants.HEALTH_CHECK_PORT}")
    except Exception as e:
        logger.warning(f"Failed to start Prometheus metrics: {e}")

# ---------- Startup banner ----------
_STARTUP_BANNER_PRINTED = False
def print_startup_banner_once() -> None:
    global _STARTUP_BANNER_PRINTED
    if _STARTUP_BANNER_PRINTED:
        return
    _STARTUP_BANNER_PRINTED = True
    logger.info(
        f"🚀 UNIFIED TRADING BOT - PRODUCTION\n"
        f"Version: {__version__} | Time: {format_ist_time(datetime.now(timezone.utc), '%d-%m-%Y @%H:%M IST')}\n"
        f"Pairs: {len(cfg.PAIRS)} | Debug: {cfg.DEBUG_MODE} | Lock Expiry: {Constants.REDIS_LOCK_EXPIRY}s | Run Timeout: {cfg.RUN_TIMEOUT_SECONDS}s"
    )
print_startup_banner_once()

# ---------- HTTP session manager ----------
class SessionManager:
    _session: ClassVar[Optional[aiohttp.ClientSession]] = None
    @classmethod
    def get_session(cls) -> aiohttp.ClientSession:
        if cls._session is None or cls._session.closed:
            ssl_ctx = ssl.create_default_context()
            ssl_ctx.check_hostname = True
            ssl_ctx.verify_mode = ssl.CERT_REQUIRED
            connector = TCPConnector(
                limit=cfg.TCP_CONN_LIMIT,
                ssl=ssl_ctx,
                force_close=False,
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
    async def close_session(cls) -> None:
        if cls._session and not cls._session.closed:
            try:
                await cls._session.close()
            except Exception:
                pass
            cls._session = None
            logger.debug("Closed shared aiohttp session")

# ---------- Retry / backoff helpers ----------
def _exp_backoff_sleep(attempt: int, base: float, cap: float) -> float:
    sleep = min(cap, base * (2 ** (attempt - 1)))
    return max(0.05, sleep * random.uniform(0.5, 1.5))

async def retry_async(
    fn: Callable,
    *args,
    retries: int = 3,
    base_backoff: float = 0.8,
    cap: float = 30.0,
    on_error: Optional[Callable[[Exception, int], None]] = None,
    **kwargs,
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
            if on_error:
                try:
                    on_error(e, attempt)
                except Exception:
                    pass
            if attempt >= retries:
                break
            await asyncio.sleep(_exp_backoff_sleep(attempt, base_backoff, cap))
    raise last_exc or RuntimeError("retry_async: unknown failure")

# ---------- Redis: defensive wrapper ----------
class RedisStateStore:
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self._redis: Optional[redis.Redis] = None
        self.state_prefix = "pair_state:"
        self.meta_prefix = "metadata:"
        self.expiry_seconds = cfg.STATE_EXPIRY_DAYS * 86400

    async def connect(self, timeout: float = 5.0) -> None:
        if self._redis is not None:
            return
        try:
            self._redis = redis.from_url(
                self.redis_url,
                socket_connect_timeout=timeout,
                socket_timeout=timeout,
                retry_on_timeout=True,
                max_connections=10,
                decode_responses=False,
            )
            await asyncio.wait_for(self._redis.ping(), timeout=timeout)
            logger.info("✅ Connected to RedisStateStore")
        except (asyncio.TimeoutError, RedisConnectionError, RedisError):
            logger.critical("Redis connection failed")
            if cfg.FAIL_ON_REDIS_DOWN:
                raise
            logger.warning("Continuing without Redis (degraded mode)")
            self._redis = None
        except Exception as e:
            logger.critical(f"Unexpected Redis connect error: {e}")
            if cfg.FAIL_ON_REDIS_DOWN:
                raise
            logger.warning("Continuing without Redis (degraded mode)")
            self._redis = None

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
                on_error=lambda e, a: logger.debug(f"Redis {op_name} error (attempt {a}): {e}"),
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
            lambda r: json.loads(r.decode("utf-8")) if r else None,
        )

    async def set(self, key: str, state: Optional[str], ts: Optional[int] = None, timeout: float = 2.0) -> None:
        ts = int(ts or time.time())
        redis_key = f"{self.state_prefix}{key}"
        data = json.dumps({"state": state, "ts": ts})
        await self._safe_redis_op(
            self._redis.set(redis_key, data, ex=self.expiry_seconds if self.expiry_seconds > 0 else None),
            timeout,
            f"set {key}",
        )

    async def get_metadata(self, key: str, timeout: float = 2.0) -> Optional[str]:
        return await self._safe_redis_op(
            self._redis.get(f"{self.meta_prefix}{key}"),
            timeout,
            f"get_metadata {key}",
            lambda r: r.decode("utf-8") if r else None,
        )

    async def set_metadata(self, key: str, value: str, timeout: float = 2.0) -> None:
        await self._safe_redis_op(
            self._redis.set(f"{self.meta_prefix}{key}", value),
            timeout,
            f"set_metadata {key}",
        )

    async def _prune_old_records(self, expiry_days: int) -> int:
        if expiry_days <= 0 or self.expiry_seconds > 0:
            logger.debug("Using Redis TTL – manual pruning skipped")
            return 0
        last_prune_str = await self.get_metadata("last_prune")
        today = datetime.now(timezone.utc).date()
        if last_prune_str:
            try:
                last_prune_date = datetime.fromisoformat(last_prune_str).date()
                if last_prune_date >= today:
                    logger.debug("Daily prune already completed – skipping.")
                    return 0
            except Exception:
                pass
        await self.set_metadata("last_prune", datetime.now(timezone.utc).isoformat())
        logger.info("Daily prune check completed (relying on Redis TTLs).")
        return 0

    async def prune_old_records(self, expiry_days: int) -> int:
        return await self._prune_old_records(expiry_days)

    async def monitor_memory(self) -> Optional[Dict[str, Any]]:
        if not self._redis:
            return None
        try:
            info_raw = await self._safe_redis_op(
                self._redis.info("all"),
                timeout=3.0,
                op_name="info all",
                parser=lambda r: r,
            )
            if not info_raw:
                return None
            mem_used = info_raw.get("memory", {}).get("used_memory")
            keyspace = info_raw.get("keyspace", {})
            db0 = keyspace.get("db0", {})
            keys = db0.get("keys", 0)
            logger.info(f"Redis INFO: used_memory={mem_used} bytes, db0_keys={keys}")
            if PROMETHEUS_ENABLED and METRIC_REDIS_MEMORY_USED and METRIC_REDIS_KEYS:
                try:
                    if mem_used is not None:
                        METRIC_REDIS_MEMORY_USED.set(float(mem_used))
                    METRIC_REDIS_KEYS.set(float(keys))
                except Exception:
                    pass
            return {"used_memory": mem_used, "db0_keys": keys}
        except Exception as e:
            logger.debug(f"Redis monitor_memory error: {e}")
            return None

# ---------- Redis distributed lock ----------
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

    async def acquire(self, timeout: float = 5.0) -> bool:
        if not self.redis:
            logger.warning("Redis not connected – cannot acquire lock")
            return False
        try:
            token = str(uuid.uuid4())
            ok = await asyncio.wait_for(
                self.redis.set(self.lock_key, token, nx=True, ex=self.expire),
                timeout=timeout
            )
            if ok:
                self.token = token
                logger.info(f"🔒 Acquired Redis lock: {self.lock_key} (expires in {self.expire}s)")
                return True
            logger.warning(f"Could not acquire Redis lock (held): {self.lock_key}")
            return False
        except asyncio.TimeoutError:
            logger.error(f"Redis lock acquisition timed out after {timeout}s")
            return False
        except RedisError as e:
            logger.error(f"Redis lock acquisition failed: {e}")
            return False

    async def extend(self, timeout: float = 3.0) -> bool:
        if not self.token or not self.redis:
            return False
        try:
            val = await asyncio.wait_for(self.redis.get(self.lock_key), timeout=timeout)
            if val and (val.decode() if isinstance(val, (bytes, bytearray)) else val) == self.token:
                await asyncio.wait_for(self.redis.expire(self.lock_key, self.expire), timeout=timeout)
                logger.debug(f"Extended Redis lock: {self.lock_key}")
                return True
            logger.warning("Lock token mismatch on extend; not extending")
            if PROMETHEUS_ENABLED and METRIC_REDIS_LOCK_FAILS:
                try:
                    METRIC_REDIS_LOCK_FAILS.inc()
                except Exception:
                    pass
            return False
        except asyncio.TimeoutError:
            logger.error("Lock extend timed out - may have lost lock")
            if PROMETHEUS_ENABLED and METRIC_REDIS_LOCK_FAILS:
                try:
                    METRIC_REDIS_LOCK_FAILS.inc()
                except Exception:
                    pass
            return False
        except RedisError as e:
            logger.error(f"Error extending Redis lock: {e}")
            if PROMETHEUS_ENABLED and METRIC_REDIS_LOCK_FAILS:
                try:
                    METRIC_REDIS_LOCK_FAILS.inc()
                except Exception:
                    pass
            return False

    async def release(self, timeout: float = 3.0) -> None:
        if not self.token or not self.redis:
            return
        try:
            await asyncio.wait_for(
                self.redis.eval(self.RELEASE_LUA, 1, self.lock_key, self.token),
                timeout=timeout
            )
            logger.info(f"🔓 Released Redis lock: {self.lock_key}")
        except asyncio.TimeoutError:
            logger.error("Lock release timed out - lock will expire naturally")
        except RedisError as e:
            logger.error(f"Error releasing Redis lock: {e}")
        finally:
            self.token = None

# ---------- HTTP fetch ----------
async def async_fetch_json(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    retries: int = 3,
    backoff: float = 1.5,
    timeout: int = 15,
    circuit_breaker: Optional['CircuitBreaker'] = None,
) -> Optional[Dict[str, Any]]:
    if circuit_breaker and await circuit_breaker.is_open():
        logger.warning(f"Circuit breaker open; skipping fetch {url}")
        return None
    session = SessionManager.get_session()
    for attempt in range(1, retries + 1):
        if shutdown_event.is_set():
            logger.info("Shutdown signaled, aborting fetch")
            return None
        try:
            async def _do_get():
                async with session.get(url, params=params, timeout=timeout) as resp:
                    text = await resp.text()
                    if resp.status == 429:
                        retry_after = resp.headers.get('Retry-After')
                        try:
                            wait_sec = min(int(retry_after) if retry_after else 1, Constants.CIRCUIT_BREAKER_MAX_WAIT)
                        except Exception:
                            wait_sec = 1
                        logger.warning(f"Received 429 from {url}; backing off for {wait_sec}s")
                        await asyncio.sleep(wait_sec + random.uniform(0, 0.5))
                        raise ClientResponseError(resp.request_info, resp.history, status=429)
                    if resp.status >= 500:
                        logger.debug(f"HTTP {resp.status} {url} - {text[:200]}")
                        raise ClientResponseError(resp.request_info, resp.history, status=resp.status)
                    if resp.status >= 400:
                        logger.debug(f"HTTP {resp.status} {url} - {text[:200]}")
                        return None
                    try:
                        return await resp.json()
                    except Exception:
                        return None

            data = await retry_async(
                _do_get,
                retries=1,
                base_backoff=backoff,
                cap=Constants.CIRCUIT_BREAKER_MAX_WAIT / 4,
                on_error=lambda e, a: logger.debug(f"HTTP get error (attempt {attempt}.{a}) for {url}: {e}")
            )
            if circuit_breaker:
                await circuit_breaker.record_success()
            return data
        except (asyncio.TimeoutError, ClientConnectorError, ClientError) as e:
            logger.debug(f"Fetch attempt {attempt} error for {url}: {e}")
            if attempt == retries:
                logger.warning(f"Failed to fetch {url} after {retries} attempts")
                if circuit_breaker:
                    await circuit_breaker.record_failure()
                if PROMETHEUS_ENABLED and METRIC_FETCH_ERRORS:
                    try:
                        METRIC_FETCH_ERRORS.inc()
                    except Exception:
                        pass
                return None
            await asyncio.sleep(_exp_backoff_sleep(attempt, backoff, Constants.CIRCUIT_BREAKER_MAX_WAIT / 10))
        except ClientResponseError as cre:
            if getattr(cre, "status", None) == 429 or getattr(cre, "status", 0) >= 500:
                logger.debug(f"Retriable ClientResponseError: {cre}")
                if attempt == retries:
                    if circuit_breaker:
                        await circuit_breaker.record_failure()
                    if PROMETHEUS_ENABLED and METRIC_FETCH_ERRORS:
                        try:
                            METRIC_FETCH_ERRORS.inc()
                        except Exception:
                            pass
                    return None
                await asyncio.sleep(_exp_backoff_sleep(attempt, backoff, Constants.CIRCUIT_BREAKER_MAX_WAIT / 10))
                continue
            logger.debug(f"Non-retryable ClientResponseError: {cre}")
            if circuit_breaker:
                await circuit_breaker.record_failure()
            if PROMETHEUS_ENABLED and METRIC_FETCH_ERRORS:
                try:
                    METRIC_FETCH_ERRORS.inc()
                except Exception:
                    pass
            return None
        except asyncio.CancelledError:
            logger.info("Fetch cancelled")
            raise
        except Exception as e:
            logger.exception(f"Unexpected fetch error for {url}: {e}")
            if circuit_breaker:
                await circuit_breaker.record_failure()
            if PROMETHEUS_ENABLED and METRIC_FETCH_ERRORS:
                try:
                    METRIC_FETCH_ERRORS.inc()
                except Exception:
                    pass
            return None
    return None

# ---------- Circuit breaker ----------
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
                if PROMETHEUS_ENABLED and METRIC_CB_OPEN:
                    try:
                        METRIC_CB_OPEN.set(0.0)
                    except Exception:
                        pass
                return False
            if self.last_failure_time and (time.time() - self.last_failure_time > self.timeout):
                self.failure_count = 0
                self.last_failure_time = None
                logger.info("✅ Circuit breaker reset after timeout")
                if PROMETHEUS_ENABLED and METRIC_CB_OPEN:
                    try:
                        METRIC_CB_OPEN.set(0.0)
                    except Exception:
                        pass
                return False
            if PROMETHEUS_ENABLED and METRIC_CB_OPEN:
                try:
                    METRIC_CB_OPEN.set(1.0)
                except Exception:
                    pass
            return True

    async def call(self, func: Callable, *args, **kwargs):  # type: ignore
        if await self.is_open():
            wait_remaining = self.timeout - (time.time() - (self.last_failure_time or 0))
            raise Exception(f"Circuit breaker is OPEN (wait {wait_remaining:.0f}s)")
        try:
            result = await func(*args, **kwargs)
            await self.record_success()
            return result
        except Exception:
            await self.record_failure()
            raise

    async def record_failure(self) -> int:
        async with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            logger.warning(f"⚠️ Circuit breaker: failure {self.failure_count}/{self.threshold}")
            if PROMETHEUS_ENABLED and METRIC_CB_FAILURES:
                try:
                    METRIC_CB_FAILURES.inc()
                except Exception:
                    pass
            if self.failure_count >= self.threshold:
                logger.critical(f"🟡 Circuit breaker OPEN for {self.timeout}s")
                if PROMETHEUS_ENABLED and METRIC_CB_OPEN:
                    try:
                        METRIC_CB_OPEN.set(1.0)
                    except Exception:
                        pass
            return self.failure_count

    async def record_success(self) -> None:
        async with self.lock:
            if self.failure_count > 0:
                self.failure_count = 0
                self.last_failure_time = None
                logger.info("✅ Circuit breaker: reset after success")
                if PROMETHEUS_ENABLED and METRIC_CB_OPEN:
                    try:
                        METRIC_CB_OPEN.set(0.0)
                    except Exception:
                        pass

# ---------- Rate limited fetcher ----------
class RateLimitedFetcher:
    def __init__(self, max_per_minute: int = 60, concurrency: int = 4):
        self.max_per_minute = max_per_minute
        self.semaphore = asyncio.Semaphore(concurrency)
        self.requests: deque[float] = deque()
        self.lock = asyncio.Lock()

    async def call(self, func: Callable, *args, **kwargs):  # type: ignore
        async with self.lock:
            now = time.time()
            while self.requests and now - self.requests[0] > 60:
                self.requests.popleft()
            if len(self.requests) >= self.max_per_minute:
                sleep_time = min(60 - (now - self.requests[0]), 60)
                logger.debug(f"Rate limit reached; sleeping {sleep_time:.2f}s")
                await asyncio.sleep(sleep_time + random.uniform(0.05, 0.2))
            self.requests.append(now)
        async with self.semaphore:
            return await func(*args, **kwargs)

# ---------- Data fetcher ----------
class DataFetcher:
    def __init__(self, api_base: str, max_parallel: Optional[int] = None):
        self.api_base = api_base.rstrip("/")
        max_parallel = max_parallel or cfg.MAX_PARALLEL_FETCH
        if not (1 <= max_parallel <= 16):
            logger.warning(f"Invalid MAX_PARALLEL_FETCH: {max_parallel}, using 8")
            max_parallel = 8
        self.semaphore = asyncio.Semaphore(max_parallel)
        self.timeout = cfg.HTTP_TIMEOUT
        self._cache: Dict[str, Tuple[float, Any]] = {}
        self._cache_max_size = 50
        self.circuit_breaker = CircuitBreaker()
        self.rate_limiter = RateLimitedFetcher(max_per_minute=60)

    def _clean_cache(self) -> None:
        if len(self._cache) > self._cache_max_size:
            to_remove = sorted(self._cache.keys(), key=lambda k: self._cache[k][0])[: self._cache_max_size // 5]
            for k in to_remove:
                del self._cache[k]

    async def fetch_products(self) -> Optional[Dict[str, Any]]:
        url = f"{self.api_base}/v2/products"
        async with self.semaphore:
            return await self.rate_limiter.call(
                async_fetch_json,
                url,
                retries=cfg.CANDLE_FETCH_RETRIES,
                backoff=cfg.CANDLE_FETCH_BACKOFF,
                timeout=self.timeout,
                circuit_breaker=self.circuit_breaker,
            )

    async def fetch_candles(self, symbol: str, resolution: str, limit: int) -> Optional[Dict[str, Any]]:
        key_str = f"{symbol}:{resolution}:{limit}"
        key_hash = f"candles:{hashlib.blake2b(key_str.encode(), digest_size=8).hexdigest()}"
        if key_hash in self._cache:
            age, data = self._cache[key_hash]
            if time.time() - age < 60:
                return data
        await asyncio.sleep(random.uniform(cfg.JITTER_MIN, cfg.JITTER_MAX))
        url = f"{self.api_base}/v2/chart/history"
        minutes = int(resolution) if resolution != "D" else 1440
        params = {
            "resolution": resolution,
            "symbol": symbol,
            "from": int(time.time()) - (limit * minutes * 60),
            "to": int(time.time()),
        }
        async with self.semaphore:
            data = await self.rate_limiter.call(
                async_fetch_json,
                url,
                params=params,
                retries=cfg.CANDLE_FETCH_RETRIES,
                backoff=cfg.CANDLE_FETCH_BACKOFF,
                timeout=self.timeout,
                circuit_breaker=self.circuit_breaker,
            )
        if data:
            self._cache[key_hash] = (time.time(), data)
            self._clean_cache()
        return data

# ---------- Candle parsing & validation ----------
def validate_candle_df(df: pd.DataFrame, required_len: int = 0) -> bool:
    try:
        if df is None or df.empty:
            logger.warning("DataFrame is None or empty")
            return False
        if df["close"].isna().any():
            logger.warning("Found NaN in close prices")
            return False
        if (df["close"] <= 0).any():
            logger.warning("Found non-positive close prices")
            return False
        if not df["timestamp"].is_monotonic_increasing:
            logger.warning("Timestamps are not monotonically increasing")
            return False
        if len(df) < required_len:
            logger.warning(f"DataFrame length {len(df)} < required {required_len}")
            return False
        return True
    except Exception as e:
        logger.error(f"DataFrame validation failed: {e}")
        return False

def parse_candles_result(result: Optional[Dict[str, Any]]) -> Optional[pd.DataFrame]:
    if not result or not isinstance(result, dict):
        return None
    if not result.get("success", True) and "result" not in result:
        return None
    res = result.get("result", {}) or {}
    required_keys = ["t", "o", "h", "l", "c", "v"]
    if not all(k in res for k in required_keys):
        return None
    for k in required_keys:
        if not isinstance(res[k], list):
            return None
    try:
        min_len = min(len(res[k]) for k in required_keys)
        df = pd.DataFrame({
            "timestamp": res["t"][:min_len],
            "open": res["o"][:min_len],
            "high": res["h"][:min_len],
            "low": res["l"][:min_len],
            "close": res["c"][:min_len],
            "volume": res["v"][:min_len],
        })
        df = df.sort_values("timestamp").drop_duplicates(subset="timestamp").reset_index(drop=True)
        if df.empty:
            return None
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], downcast="float", errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], downcast="float", errors="coerce")
        df["timestamp"] = pd.to_numeric(df["timestamp"], downcast="integer", errors="coerce")
        max_ts = int(df["timestamp"].max())
        if max_ts > 1_000_000_000_000:
            df["timestamp"] = (df["timestamp"] // 1000).astype(int)
        elif max_ts > 10_000_000_000:
            df["timestamp"] = (df["timestamp"] // 1000).astype(int)
        else:
            df["timestamp"] = df["timestamp"].astype(int)
        if float(df["close"].iloc[-1]) <= 0:
            return None
        return df
    except Exception as e:
        logger.exception(f"Failed to parse candles: {e}")
        return None

# ---------- Indicators (unchanged) ----------
def validate_indicator_series(series: pd.Series, name: str) -> pd.Series:
    try:
        series = series.replace([np.inf, -np.inf], np.nan).bfill().ffill()
        if series.isna().all():
            logger.warning(f"Indicator {name} is all NaN, returning zeros")
            return pd.Series([0.0] * len(series), index=series.index)
        return series
    except Exception as e:
        logger.error(f"Indicator validation failed for {name}: {e}")
        return pd.Series([0.0] * len(series), index=series.index)

def calculate_ema(data: pd.Series, period: int) -> pd.Series:
    return validate_indicator_series(data.ewm(span=period, adjust=False).mean(), "EMA")

def calculate_sma(data: pd.Series, period: int) -> pd.Series:
    return validate_indicator_series(
        data.rolling(window=period, min_periods=max(2, period // 3)).mean(),
        "SMA"
    )

def calculate_rma(data: pd.Series, period: int) -> pd.Series:
    r = data.ewm(alpha=1 / period, adjust=False).mean()
    return validate_indicator_series(r.bfill().ffill(), "RMA")

def calculate_ppo(df: pd.DataFrame, fast: int, slow: int, signal: int, use_sma: bool = False) -> Tuple[pd.Series, pd.Series]:
    close = df["close"].astype(float)
    fast_ma = calculate_sma(close, fast) if use_sma else calculate_ema(close, fast)
    slow_ma = calculate_sma(close, slow) if use_sma else calculate_ema(close, slow)
    slow_ma = slow_ma.replace(0, np.nan).bfill().ffill()
    ppo = ((fast_ma - slow_ma) / slow_ma) * 100
    ppo = validate_indicator_series(ppo.replace([np.inf, -np.inf], np.nan).bfill().ffill(), "PPO")
    ppo_signal = calculate_sma(ppo, signal) if use_sma else calculate_ema(ppo, signal)
    ppo_signal = validate_indicator_series(ppo_signal.replace([np.inf, -np.inf], np.nan).bfill().ffill(), "PPO_SIGNAL")
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

def calculate_cirrus_cloud(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    close = df["close"].astype(float)
    smrngx1x = smoothrng(close, cfg.X1, cfg.X2)
    smrngx1x2 = smoothrng(close, cfg.X3, cfg.X4)
    filtx1 = rngfilt(close, smrngx1x)
    filtx12 = rngfilt(close, smrngx1x2)
    upw = filtx1 < filtx12
    dnw = filtx1 > filtx12
    return upw, dnw, filtx1, filtx12

def kalman_filter(src: pd.Series, length: int, R: float = 0.01, Q: float = 0.1) -> pd.Series:
    result = []
    estimate = np.nan
    error_est = 1.0
    error_meas = R * max(1, length)
    Q_div_length = Q / max(1, length)
    for i in range(len(src)):
        current = src.iloc[i]
        if np.isnan(estimate):
            estimate = src.iloc[i - 1] if i > 0 else current
        prediction = estimate
        kalman_gain = error_est / (error_est + error_meas)
        estimate = prediction + kalman_gain * (current - prediction)
        error_est = (1 - kalman_gain) * error_est + Q_div_length
        result.append(estimate)
    return pd.Series(result, index=src.index)

def calculate_smooth_rsi(df: pd.DataFrame, rsi_len: int, kalman_len: int) -> pd.Series:
    close = df["close"].astype(float)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = calculate_rma(gain, rsi_len)
    avg_loss = calculate_rma(loss, rsi_len).replace(0, np.nan).bfill().ffill().clip(lower=1e-8)
    rs = avg_gain.divide(avg_loss)
    rsi_value = 100 - (100 / (1 + rs))
    rsi_value = rsi_value.replace([np.inf, -np.inf], np.nan).bfill().ffill()
    smooth_rsi = kalman_filter(rsi_value, kalman_len).bfill().ffill()
    return validate_indicator_series(smooth_rsi, "SmoothRSI")

def calculate_magical_momentum_hist(df: pd.DataFrame, period: int = 144, responsiveness: float = 0.9) -> pd.Series:
    try:
        source = pd.to_numeric(df["close"], errors="coerce").astype(float)
        sd = source.rolling(window=50, min_periods=1).std(ddof=1) * max(0.00001, responsiveness)
        sd = validate_indicator_series(sd, "MMH_SD")
        worm = [source.iloc[0]]
        for i in range(1, len(source)):
            prev_worm = worm[-1]
            s_i = source.iloc[i]
            sd_i = float(sd.iloc[i]) if pd.notna(sd.iloc[i]) else 0.0
            if pd.isna(s_i):
                worm.append(prev_worm)
                continue
            diff = s_i - prev_worm
            delta = np.sign(diff) * sd_i if abs(diff) > sd_i else diff
            worm.append(prev_worm + delta)
        worm = pd.Series(worm, index=source.index)
        ma = source.rolling(window=period, min_periods=1).mean()
        ma = validate_indicator_series(ma, "MMH_MA")
        raw_momentum = (worm - ma) / worm
        raw_momentum = validate_indicator_series(
            raw_momentum.replace([np.inf, -np.inf], np.nan).fillna(0.0), "MMH_RAW"
        )
        min_med = raw_momentum.rolling(window=period, min_periods=1).min()
        max_med = raw_momentum.rolling(window=period, min_periods=1).max()
        denom = (max_med - min_med).replace(0, 1e-12)
        temp = (raw_momentum - min_med) / denom
        temp = temp.clip(lower=0.0, upper=1.0).fillna(0.5)
        value = pd.Series(index=source.index, dtype=float)
        value.iloc[0] = 1.0
        for i in range(1, len(temp)):
            prev_val = value.iloc[i - 1]
            if pd.isna(prev_val):
                prev_val = 1.0
            temp_val = temp.iloc[i]
            if pd.isna(temp_val):
                temp_val = 0.5
            multiplier = temp_val - 0.5 + 0.5 * prev_val
            val = 1.0 * multiplier
            value.iloc[i] = np.clip(val, -0.9999, 0.9999)
        temp2 = (1.0 + value) / (1.0 - value)
        temp2 = temp2.replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(lower=1e-8, upper=1e8)
        momentum = 0.25 * np.log(temp2)
        momentum = validate_indicator_series(momentum, "MMH_MOMENTUM")

        for i in range(1, len(momentum)):
            prev_m = momentum.iloc[i - 1] if pd.notna(momentum.iloc[i - 1]) else 0.0
            m_i = momentum.iloc[i] if pd.notna(momentum.iloc[i]) else 0.0
            momentum.iloc[i] = m_i + 0.5 * prev_m
        return momentum
    except Exception as e:
        logger.error(f"Error in calculate_magical_momentum_hist: {e}")
        return pd.Series([0.0] * len(df), index=df.index)

def calculate_vwap_daily_reset(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    df2 = df.copy()
    df2["datetime"] = pd.to_datetime(df2["timestamp"], unit="s", utc=True)
    df2["date"] = df2["datetime"].dt.date
    hlc3 = (df2["high"] + df2["low"] + df2["close"]) / 3.0
    df2["hlc3_vol"] = hlc3 * df2["volume"]
    df2["cum_vol"] = df2.groupby("date")["volume"].cumsum()
    df2["cum_hlc3_vol"] = df2.groupby("date")["hlc3_vol"].cumsum()
    vwap = df2["cum_hlc3_vol"] / df2["cum_vol"].replace(0, np.nan)
    return validate_indicator_series(vwap.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0), "VWAP")

def get_last_closed_index(df: pd.DataFrame, interval_minutes: int) -> Optional[int]:
    if df is None or df.empty or len(df) < 2:
        return None
    last_ts = int(df["timestamp"].iloc[-1])
    current_interval_start = (last_ts // (interval_minutes * 60)) * (interval_minutes * 60)
    return len(df) - 2 if last_ts >= current_interval_start else len(df) - 1

# ---------- Health tracker ----------
class HealthTracker:
    def __init__(self, sdb: RedisStateStore):
        self.sdb = sdb

    async def record_pair_result(self, pair: str, success: bool, info: Optional[Dict[str, Any]] = None) -> None:
        key = f"health:pair:{pair}"
        now = int(time.time())
        payload = {"last_checked": now, "last_success": now if success else None, "success": bool(success)}
        if info:
            payload.update({"info": info})
        await self.sdb.set_metadata(key, json.dumps(payload))

    async def record_overall(self, summary: Dict[str, Any]) -> None:
        key = "health:overall"
        payload = {"ts": int(time.time()), "summary": summary}
        await self.sdb.set_metadata(key, json.dumps(payload))

health_tracker: Optional[HealthTracker] = None

# ---------- Dead Man’s Switch ----------
class DeadMansSwitch:
    def __init__(self, sdb: RedisStateStore, cooldown_seconds: int):
        self.sdb = sdb
        self.cooldown_seconds = cooldown_seconds
        self.alert_sent = False
        self.last_check_time = 0.0
        self.startup_time = time.time()
        self.alert_recovered = False

    def _parse_last_success(self, last_success: Optional[str]) -> Optional[int]:
        if not last_success:
            return None
        try:
            return int(last_success)
        except Exception:
            try:
                dt = datetime.fromisoformat(last_success)
                return int(dt.replace(tzinfo=timezone.utc).timestamp())
            except Exception:
                return None

    async def should_alert(self) -> bool | str:
        try:
            now = time.time()
            if now - self.startup_time < Constants.STARTUP_GRACE_PERIOD:
                return False
            if now - self.last_check_time < 60:
                return False
            self.last_check_time = now
            last_success = await self.sdb.get_metadata("last_success_run")
            if not last_success:
                logger.debug("No last_success_run found – first run?")
                return False
            last_run_ts = self._parse_last_success(last_success)
            if last_run_ts is None:
                logger.debug("Could not parse last_success_run")
                return False
            time_since_last_run = now - last_run_ts
            logger.debug(f"Dead man's check: {time_since_last_run}s since last run")
            if time_since_last_run <= self.cooldown_seconds and self.alert_sent and not self.alert_recovered:
                self.alert_recovered = True
                logger.info("✅ Dead man's switch recovered")
                return "RECOVERED"
            if time_since_last_run > self.cooldown_seconds and not self.alert_sent:
                self.alert_sent = True
                self.alert_recovered = False
                logger.warning(f"⚠️ Dead man's switch triggered! Last run: {time_since_last_run/3600:.1f}h ago")
                return True
            if time_since_last_run <= self.cooldown_seconds and self.alert_sent:
                self.alert_sent = False
            return False
        except Exception as e:
            logger.error(f"Error checking dead man's switch: {e}")
            return False

# ---------- Telegram ----------
def escape_markdown_v2(text: str) -> str:
    try:
        if not isinstance(text, str):
            text = str(text)
        text = text.replace("\\", "\\\\")
        pattern = r'([_\*\[\]\(\)~`>#+\-=|{}\.\!])'
        return re.sub(pattern, r'\\\1', text)
    except Exception:
        return str(text).replace("\\", "\\\\")

class TokenBucket:
    def __init__(self, rate: int, burst: int):
        self.rate = rate
        self.burst = burst
        self.tokens = float(burst)
        self.last_update = time.time()
        self.lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self.lock:
                now = time.time()
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
        self._send_semaphore = asyncio.Semaphore(1)
        self._last_sent = 0.0
        self.rate_limit = 0.1
        self.token_bucket = TokenBucket(cfg.TELEGRAM_RATE_LIMIT_PER_MINUTE, cfg.TELEGRAM_BURST_SIZE)
        self.circuit_breaker = CircuitBreaker(failure_threshold=3, timeout=300)

    async def send(self, message: str) -> bool:
        try:
            await asyncio.wait_for(self.circuit_breaker.call(self._send_impl, message), timeout=30.0)
            if PROMETHEUS_ENABLED and METRIC_ALERTS_SENT:
                try:
                    METRIC_ALERTS_SENT.inc()
                except Exception:
                    pass
            return True
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            if not cfg.FAIL_ON_TELEGRAM_DOWN:
                logger.warning("Continuing despite Telegram failure (degraded mode)")
                return False
            raise

    async def _send_impl(self, message: str) -> bool:
        await self.token_bucket.acquire()
        await asyncio.sleep(max(0, self.rate_limit - (time.time() - self._last_sent)))
        self._last_sent = time.time()
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        params = {"chat_id": self.chat_id, "text": message, "parse_mode": "MarkdownV2"}
        session = SessionManager.get_session()
        for attempt in range(1, cfg.TELEGRAM_RETRIES + 1):
            if shutdown_event.is_set():
                logger.info("Shutdown signaled, aborting Telegram send")
                return False
            try:
                async with session.post(url, data=params, timeout=10) as resp:
                    text = await resp.text()
                    if resp.status == 429:
                        retry_after = resp.headers.get("Retry-After")
                        try:
                            wait_sec = min(int(retry_after) if retry_after else 1, Constants.CIRCUIT_BREAKER_MAX_WAIT)
                        except Exception:
                            wait_sec = 1
                        logger.warning(f"Telegram rate limited (429). Retry-After: {wait_sec}s")
                        await asyncio.sleep(wait_sec + random.uniform(0.1, 0.5))
                        continue
                    if resp.status == 200:
                        logger.debug("Telegram message sent successfully")
                        return True
                    if resp.status in (400, 401, 403, 404):
                        logger.error(f"Irrecoverable Telegram error {resp.status}: {text[:200]}")
                        return False
                    logger.error(f"Telegram API failed (Status {resp.status}, Attempt {attempt}): {text[:200]}")
                    raise Exception(f"Telegram API error {resp.status}")
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"Telegram send attempt {attempt} failed: {e}")
                if attempt < cfg.TELEGRAM_RETRIES:
                    sleep_time = min((cfg.TELEGRAM_BACKOFF_BASE ** (attempt - 1)) + random.uniform(0, 0.3), 30)
                    await asyncio.sleep(sleep_time)
        logger.error("Telegram send failed after retries")
        return False

    async def send_batch(self, messages: List[str]) -> bool:
        if not messages:
            return True
        max_len = 3800
        to_send = messages[:10]
        combined = "\n\n".join(to_send)
        if len(combined) > max_len:
            results = await asyncio.gather(*[self.send(m) for m in to_send], return_exceptions=True)
            return all(r is True for r in results)
        return await self.send(combined)

# ---------- Alerts ----------
class AlertDefinition(TypedDict):
    key: str
    title: str
    check_fn: Callable[[Any, Any, Any, Any], bool]
    extra_fn: Callable[[Any, Any, Any, Any, Dict[str, Any]], str]
    requires: List[str]

ALERT_DEFINITIONS: List[AlertDefinition] = [
    {
        "key": "ppo_signal_up",
        "title": "🟢 PPO cross above signal",
        "check_fn": lambda ctx, ppo, ppo_sig, rsi: (
            ctx["buy_common"] and (ppo["prev"] <= ppo_sig["prev"]) and (ppo["curr"] > ppo_sig["curr"]) and (ppo["curr"] < Constants.PPO_THRESHOLD_BUY)
        ),
        "extra_fn": lambda ctx, ppo, ppo_sig, rsi: f"PPO {ppo['curr']:.2f} vs Sig {ppo_sig['curr']:.2f} | MMH ({ctx['mmh_curr']:.2f})",
        "requires": ["ppo", "ppo_signal"],
    },
    {
        "key": "ppo_signal_down",
        "title": "🔴 PPO cross below signal",
        "check_fn": lambda ctx, ppo, ppo_sig, rsi: (
            ctx["sell_common"] and (ppo["prev"] >= ppo_sig["prev"]) and (ppo["curr"] < ppo_sig["curr"]) and (ppo["curr"] > Constants.PPO_THRESHOLD_SELL)
        ),
        "extra_fn": lambda ctx, ppo, ppo_sig, rsi: f"PPO {ppo['curr']:.2f} vs Sig {ppo_sig['curr']:.2f} | MMH ({ctx['mmh_curr']:.2f})",
        "requires": ["ppo", "ppo_signal"],
    },
    {
        "key": "ppo_zero_up",
        "title": "🟢 PPO cross above 0",
        "check_fn": lambda ctx, ppo, ppo_sig, rsi: ctx["buy_common"] and (ppo["prev"] <= 0.0) and (ppo["curr"] > 0.0),
        "extra_fn": lambda ctx, ppo, ppo_sig, rsi: f"PPO {ppo['curr']:.2f} | MMH ({ctx['mmh_curr']:.2f})",
        "requires": ["ppo"],
    },
    {
        "key": "ppo_zero_down",
        "title": "🔴 PPO cross below 0",
        "check_fn": lambda ctx, ppo, ppo_sig, rsi: ctx["sell_common"] and (ppo["prev"] >= 0.0) and (ppo["curr"] < 0.0),
        "extra_fn": lambda ctx, ppo, ppo_sig, rsi: f"PPO {ppo['curr']:.2f} | MMH ({ctx['mmh_curr']:.2f})",
        "requires": ["ppo"],
    },
    {
        "key": "ppo_011_up",
        "title": "🟢 PPO cross above 0.11",
        "check_fn": lambda ctx, ppo, ppo_sig, rsi: (
            ctx["buy_common"] and (ppo["prev"] <= Constants.PPO_011_THRESHOLD) and (ppo["curr"] > Constants.PPO_011_THRESHOLD)
        ),
        "extra_fn": lambda ctx, ppo, ppo_sig, rsi: f"PPO {ppo['curr']:.2f} | MMH ({ctx['mmh_curr']:.2f})",
        "requires": ["ppo"],
    },
    {
        "key": "ppo_011_down",
        "title": "🔴 PPO cross below -0.11",
        "check_fn": lambda ctx, ppo, ppo_sig, rsi: (
            ctx["sell_common"] and (ppo["prev"] >= Constants.PPO_011_THRESHOLD_SELL) and (ppo["curr"] < Constants.PPO_011_THRESHOLD_SELL)
        ),
        "extra_fn": lambda ctx, ppo, ppo_sig, rsi: f"PPO {ppo['curr']:.2f} | MMH ({ctx['mmh_curr']:.2f})",
        "requires": ["ppo"],
    },
    {
        "key": "rsi_50_up",
        "title": "🟢 RSI cross above 50 (PPO < 0.30)",
        "check_fn": lambda ctx, ppo, ppo_sig, rsi: (
            ctx["buy_common"] and (rsi["prev"] <= Constants.RSI_THRESHOLD) and (rsi["curr"] > Constants.RSI_THRESHOLD) and (ppo["curr"] < Constants.PPO_RSI_GUARD_BUY)
        ),
        "extra_fn": lambda ctx, ppo, ppo_sig, rsi: f"RSI {rsi['curr']:.2f} | PPO {ppo['curr']:.2f} | MMH ({ctx['mmh_curr']:.2f})",
        "requires": ["ppo", "rsi"],
    },
    {
        "key": "rsi_50_down",
        "title": "🔴 RSI cross below 50 (PPO > -0.30)",
        "check_fn": lambda ctx, ppo, ppo_sig, rsi: (
            ctx["sell_common"] and (rsi["prev"] >= Constants.RSI_THRESHOLD) and (rsi["curr"] < Constants.RSI_THRESHOLD) and (ppo["curr"] > Constants.PPO_RSI_GUARD_SELL)
        ),
        "extra_fn": lambda ctx, ppo, ppo_sig, rsi: f"RSI {rsi['curr']:.2f} | PPO {ppo['curr']:.2f} | MMH ({ctx['mmh_curr']:.2f})",
        "requires": ["ppo", "rsi"],
    },
    {
        "key": "vwap_up",
        "title": "🔵 Price cross above VWAP",
        "check_fn": lambda ctx, ppo, ppo_sig, rsi: (
            ctx["buy_common"] and (ctx["close_prev"] <= ctx["vwap_prev"]) and (ctx["close_curr"] > ctx["vwap_curr"])
        ),
        "extra_fn": lambda ctx, ppo, ppo_sig, rsi: f"MMH ({ctx['mmh_curr']:.2f})",
        "requires": [],
    },
    {
        "key": "vwap_down",
        "title": "🟣 Price cross below VWAP",
        "check_fn": lambda ctx, ppo, ppo_sig, rsi: (
            ctx["sell_common"] and (ctx["close_prev"] >= ctx["vwap_prev"]) and (ctx["close_curr"] < ctx["vwap_curr"])
        ),
        "extra_fn": lambda ctx, ppo, ppo_sig, rsi: f"MMH ({ctx['mmh_curr']:.2f})",
        "requires": [],
    },
    {
        "key": "mmh_buy",
        "title": "⚡️🔵 MMH Reversal BUY",
        "check_fn": lambda ctx, ppo, ppo_sig, rsi: ctx["mmh_reversal_buy"],
        "extra_fn": lambda ctx, ppo, ppo_sig, rsi: f"MMH ({ctx['mmh_curr']:.2f})",
        "requires": [],
    },
    {
        "key": "mmh_sell",
        "title": "⚡️🟣 MMH Reversal SELL",
        "check_fn": lambda ctx, ppo, ppo_sig, rsi: ctx["mmh_reversal_sell"],
        "extra_fn": lambda ctx, ppo, ppo_sig, rsi: f"MMH ({ctx['mmh_curr']:.2f})",
        "requires": [],
    },
]

PIVOT_LEVELS = ["P", "S1", "S2", "S3", "R1", "R2", "R3"]

BUY_PIVOT_DEFS = [
    {
        "key": f"pivot_up_{level}",
        "title": f"🔷 Cross above {level}",
        "check_fn": lambda ctx, ppo, ppo_sig, rsi, level=level: (
            ctx["buy_common"] and (ctx["close_prev"] <= ctx["pivots"][level]) and (ctx["close_curr"] > ctx["pivots"][level])
        ),
        "extra_fn": lambda ctx, ppo, ppo_sig, rsi, level=level: f"${ctx['pivots'][level]:,.2f} | MMH ({ctx['mmh_curr']:.2f})",
        "requires": ["pivots"],
    }
    for level in ["P", "S1", "S2", "S3", "R1", "R2"]
]

SELL_PIVOT_DEFS = [
    {
        "key": f"pivot_down_{level}",
        "title": f"🔶 Cross below {level}",
        "check_fn": lambda ctx, ppo, ppo_sig, rsi, level=level: (
            ctx["sell_common"] and (ctx["close_prev"] >= ctx["pivots"][level]) and (ctx["close_curr"] < ctx["pivots"][level])
        ),
        "extra_fn": lambda ctx, ppo, ppo_sig, rsi, level=level: f"${ctx['pivots'][level]:,.2f} | MMH ({ctx['mmh_curr']:.2f})",
        "requires": ["pivots"],
    }
    for level in ["P", "S1", "S2", "R1", "R2", "R3"]
]

ALERT_DEFINITIONS.extend(BUY_PIVOT_DEFS)
ALERT_DEFINITIONS.extend(SELL_PIVOT_DEFS)

ALERT_KEYS: Dict[str, str] = {d["key"]: f"ALERT:{d['key'].upper()}" for d in ALERT_DEFINITIONS}
for level in PIVOT_LEVELS:
    ALERT_KEYS[f"pivot_up_{level}"] = f"ALERT:PIVOT_UP_{level}"
    ALERT_KEYS[f"pivot_down_{level}"] = f"ALERT:PIVOT_DOWN_{level}"

# ---------- Alert state helpers ----------
async def set_alert_state(sdb: RedisStateStore, pair: str, key: str, active: bool) -> None:
    state_key = f"{pair}:{key}"
    ts = int(time.time())
    state_val = "ACTIVE" if active else "INACTIVE"
    await sdb.set(state_key, state_val, ts)

async def was_alert_active(sdb: RedisStateStore, pair: str, key: str) -> bool:
    state_key = f"{pair}:{key}"
    st = await sdb.get(state_key)
    return st is not None and st.get("state") == "ACTIVE"

async def reset_alert_on_condition(sdb: RedisStateStore, pair: str, key: str, condition: bool) -> None:
    if condition:
        await set_alert_state(sdb, pair, key, False)

def build_msg(title: str, pair: str, price: float, ts: int, extra: Optional[str] = None) -> str:
    parts = title.split(" ", 1)
    symbols = parts[0] if len(parts) == 2 else ""
    description = parts[1] if len(parts) == 2 else title
    price_str = f"${price:,.2f}"
    line1 = f"{symbols} {pair} - {price_str}".strip()
    line2 = f"{description} : {extra}" if extra else f"{description}"
    line3 = format_ist_time(ts, "%d-%m-%Y     %H:%M IST")
    visual = f"{line1}\n{line2}\n{line3}"
    return escape_markdown_v2(visual)

def check_common_conditions(df_15m: pd.DataFrame, idx: int, is_buy: bool) -> bool:
    try:
        row = df_15m.iloc[idx]
        o, c, h, l = float(row["open"]), float(row["close"]), float(row["high"]), float(row["low"])
        rng = max(h - l, 1e-8)
        if is_buy:
            if c <= o:
                return False
            upper_wick = h - max(o, c)
            return upper_wick < Constants.MIN_WICK_RATIO * rng
        else:
            if c >= o:
                return False
            lower_wick = min(o, c) - l
            return lower_wick < Constants.MIN_WICK_RATIO * rng
    except Exception as e:
        logger.error(f"Error in common candle check: {e}")
        return False

# ---------- Alert processing ----------
async def process_alert_definitions(
    ctx: Dict[str, Any],
    sdb: RedisStateStore,
    pair_name: str,
    messages: List[str],
    alerts_to_activate: List[str],
) -> None:
    ppo_ctx = {"curr": ctx["ppo_curr"], "prev": ctx["ppo_prev"]}
    ppo_sig_ctx = {"curr": ctx["ppo_sig_curr"], "prev": ctx["ppo_sig_prev"]}
    rsi_ctx = {"curr": ctx["rsi_curr"], "prev": ctx["rsi_prev"]}
    for def_ in ALERT_DEFINITIONS:
        if "pivots" in def_["requires"] and not ctx.get("pivots"):
            continue
        if "vwap" in def_["requires"] and not ctx.get("vwap"):
            continue
        try:
            if def_["check_fn"](ctx, ppo_ctx, ppo_sig_ctx, rsi_ctx):
                key = ALERT_KEYS[def_["key"]]
                if not await was_alert_active(sdb, pair_name, key):
                    extra = def_["extra_fn"](ctx, ppo_ctx, ppo_sig_ctx, rsi_ctx)
                    msg = build_msg(def_["title"], pair_name, ctx["close_curr"], ctx["ts_curr"], extra)
                    messages.append(msg)
                    alerts_to_activate.append(key)
        except Exception as e:
            logger.error(f"Error processing alert {def_['key']}: {e}")

def derive_suppression_reason(ctx: Dict[str, Any]) -> str:
    reasons = []
    if not ctx["buy_common"] and not ctx["sell_common"]:
        reasons.append("Trend filter blocked (RMA + cloud/MMH)")
    if ctx.get("candle_quality_failed_buy"):
        reasons.append("Candle quality check failed (buy)")
    if ctx.get("candle_quality_failed_sell"):
        reasons.append("Candle quality check failed (sell)")
    if ctx["buy_common"] and ctx["ppo_curr"] >= Constants.PPO_THRESHOLD_BUY:
        reasons.append("PPO below buy threshold not met")
    if ctx["sell_common"] and ctx["ppo_curr"] <= Constants.PPO_THRESHOLD_SELL:
        reasons.append("PPO above sell threshold not met")
    if ctx["buy_common"] and ctx["rsi_prev"] <= Constants.RSI_THRESHOLD and not (
        ctx["rsi_curr"] > Constants.RSI_THRESHOLD and ctx["ppo_curr"] < Constants.PPO_RSI_GUARD_BUY
    ):
        reasons.append("RSI guard not satisfied (buy)")
    if ctx["sell_common"] and ctx["rsi_prev"] >= Constants.RSI_THRESHOLD and not (
        ctx["rsi_curr"] < Constants.RSI_THRESHOLD and ctx["ppo_curr"] > Constants.PPO_RSI_GUARD_SELL
    ):
        reasons.append("RSI guard not satisfied (sell)")
    return "; ".join(reasons[:2]) if reasons else "No alert conditions met"

# ---------- Pair evaluation ----------
async def evaluate_pair_and_alert(
    pair_name: str,
    df_15m: pd.DataFrame,
    df_5m: pd.DataFrame,
    df_daily: Optional[pd.DataFrame],
    sdb: RedisStateStore,
    telegram_queue: TelegramQueue,
    correlation_id: str,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    logger_pair = logging.getLogger(f"macd_bot.{pair_name}.{correlation_id}")
    PAIR_ID.set(pair_name)
    try:
        i15 = get_last_closed_index(df_15m, 15)
        i5 = get_last_closed_index(df_5m, 5)
        if i15 is None or i15 < 3 or i5 is None:
            logger_pair.debug(f"Indexing not ready (i15={i15}, i5={i5})")
            return None

        ppo, ppo_signal = calculate_ppo(df_15m, cfg.PPO_FAST, cfg.PPO_SLOW, cfg.PPO_SIGNAL, cfg.PPO_USE_SMA)
        smooth_rsi = calculate_smooth_rsi(df_15m, cfg.SRSI_RSI_LEN, cfg.SRSI_KALMAN_LEN)
        vwap = calculate_vwap_daily_reset(df_15m) if cfg.ENABLE_VWAP else pd.Series(index=df_15m.index, dtype=float)
        mmh = calculate_magical_momentum_hist(df_15m)

        if cfg.CIRRUS_CLOUD_ENABLED:
            upw, dnw, _, _ = calculate_cirrus_cloud(df_15m)
        else:
            upw = dnw = pd.Series(False, index=df_15m.index)

        piv: Dict[str, float] = {}
        if cfg.ENABLE_PIVOT and df_daily is not None and len(df_daily) >= 2:
            try:
                df_daily["date"] = pd.to_datetime(df_daily["timestamp"], unit="s", utc=True).dt.date
                completed_days = df_daily["date"] < datetime.now(timezone.utc).date()
                if completed_days.sum() >= 1:
                    prev_daily = df_daily[completed_days].iloc[-1]
                    H_prev = float(prev_daily["high"])
                    L_prev = float(prev_daily["low"])
                    C_prev = float(prev_daily["close"])
                    P = (H_prev + L_prev + C_prev) / 3.0
                    R3 = P + ((H_prev - L_prev) * 1.000)
                    R2 = P + ((H_prev - L_prev) * 0.618)
                    R1 = P + ((H_prev - L_prev) * 0.382)
                    S1 = P - ((H_prev - L_prev) * 0.382)
                    S2 = P - ((H_prev - L_prev) * 0.618)
                    S3 = P - ((H_prev - L_prev) * 1.000)
                    piv = {"P": P, "R1": R1, "R2": R2, "R3": R3, "S1": S1, "S2": S2, "S3": S3}
            except Exception as e:
                logger_pair.warning(f"Error calculating pivots: {e}")

        close_curr = float(df_15m["close"].iloc[i15])
        close_prev = float(df_15m["close"].iloc[i15 - 1])
        ts_curr = int(df_15m["timestamp"].iloc[i15])

        ppo_curr = float(ppo.iloc[i15])
        ppo_prev = float(ppo.iloc[i15 - 1])
        ppo_sig_curr = float(ppo_signal.iloc[i15])
        ppo_sig_prev = float(ppo_signal.iloc[i15 - 1])
        rsi_curr = float(smooth_rsi.iloc[i15])
        rsi_prev = float(smooth_rsi.iloc[i15 - 1])
        vwap_curr = float(vwap.iloc[i15]) if not vwap.empty else 0.0
        vwap_prev = float(vwap.iloc[i15 - 1]) if not vwap.empty else 0.0
        mmh_curr = float(mmh.iloc[i15])
        mmh_m1 = float(mmh.iloc[i15 - 1])

        cloud_up = bool(upw.iloc[i15]) and not bool(dnw.iloc[i15])
        cloud_down = bool(dnw.iloc[i15]) and not bool(upw.iloc[i15])

        rma50_15 = float(calculate_rma(df_15m["close"], cfg.RMA_50_PERIOD).iloc[i15])
        rma200_5 = float(calculate_rma(df_5m["close"], cfg.RMA_200_PERIOD).iloc[i5])

        base_buy_common = rma50_15 < close_curr and rma200_5 < close_curr
        base_sell_common = rma50_15 > close_curr and rma200_5 > close_curr

        if base_buy_common:
            base_buy_common = base_buy_common and (mmh_curr > 0 and cloud_up)
        if base_sell_common:
            base_sell_common = base_sell_common and (mmh_curr < 0 and cloud_down)

        buy_common_candle_ok = check_common_conditions(df_15m, i15, is_buy=True)
        sell_common_candle_ok = check_common_conditions(df_15m, i15, is_buy=False)

        buy_common = base_buy_common and buy_common_candle_ok
        sell_common = base_sell_common and sell_common_candle_ok

        mmh_reversal_buy = False
        mmh_reversal_sell = False
        if i15 >= 3:
            mmh_m3 = float(mmh.iloc[i15 - 3])
            mmh_m2 = float(mmh.iloc[i15 - 2])
            mmh_reversal_buy = buy_common and (mmh_curr > 0) and (mmh_m2 < mmh_m3) and (mmh_m1 < mmh_m2) and (mmh_curr > mmh_m1)
            mmh_reversal_sell = sell_common and (mmh_curr < 0) and (mmh_m2 > mmh_m3) and (mmh_m1 > mmh_m2) and (mmh_curr < mmh_m1)

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
            "vwap": not pd.Series(dtype=float).equals(vwap),
            "candle_quality_failed_buy": base_buy_common and not buy_common_candle_ok,
            "candle_quality_failed_sell": base_sell_common and not sell_common_candle_ok,
        }

        messages: List[str] = []
        alerts_to_activate: List[str] = []

        await process_alert_definitions(context, sdb, pair_name, messages, alerts_to_activate)

        await reset_alert_on_condition(sdb, pair_name, ALERT_KEYS["ppo_signal_up"], ppo_prev > ppo_sig_prev and ppo_curr <= ppo_sig_curr)
        await reset_alert_on_condition(sdb, pair_name, ALERT_KEYS["ppo_signal_down"], ppo_prev < ppo_sig_prev and ppo_curr >= ppo_sig_curr)
        await reset_alert_on_condition(sdb, pair_name, ALERT_KEYS["ppo_zero_up"], ppo_prev > 0 and ppo_curr <= 0)
        await reset_alert_on_condition(sdb, pair_name, ALERT_KEYS["ppo_zero_down"], ppo_prev < 0 and ppo_curr >= 0)
        await reset_alert_on_condition(sdb, pair_name, ALERT_KEYS["ppo_011_up"], ppo_prev > Constants.PPO_011_THRESHOLD and ppo_curr <= Constants.PPO_011_THRESHOLD)
        await reset_alert_on_condition(sdb, pair_name, ALERT_KEYS["ppo_011_down"], ppo_prev < Constants.PPO_011_THRESHOLD_SELL and ppo_curr >= Constants.PPO_011_THRESHOLD_SELL)
        await reset_alert_on_condition(sdb, pair_name, ALERT_KEYS["rsi_50_up"], rsi_prev > Constants.RSI_THRESHOLD and rsi_curr <= Constants.RSI_THRESHOLD)
        await reset_alert_on_condition(sdb, pair_name, ALERT_KEYS["rsi_50_down"], rsi_prev < Constants.RSI_THRESHOLD and rsi_curr >= Constants.RSI_THRESHOLD)

        if context["vwap"]:
            await reset_alert_on_condition(sdb, pair_name, ALERT_KEYS["vwap_up"], close_prev > vwap_prev and close_curr <= vwap_curr)
            await reset_alert_on_condition(sdb, pair_name, ALERT_KEYS["vwap_down"], close_prev < vwap_prev and close_curr >= vwap_curr)

        if piv:
            for level_name, level_value in piv.items():
                await reset_alert_on_condition(sdb, pair_name, ALERT_KEYS[f"pivot_up_{level_name}"], close_prev > level_value and close_curr <= level_value)
                await reset_alert_on_condition(sdb, pair_name, ALERT_KEYS[f"pivot_down_{level_name}"], close_prev < level_value and close_curr >= level_value)

        if (mmh_curr > 0) and (mmh_curr <= mmh_m1) and (await was_alert_active(sdb, pair_name, ALERT_KEYS["mmh_buy"])):
            await set_alert_state(sdb, pair_name, ALERT_KEYS["mmh_buy"], False)
        if (mmh_curr < 0) and (mmh_curr >= mmh_m1) and (await was_alert_active(sdb, pair_name, ALERT_KEYS["mmh_sell"])):
            await set_alert_state(sdb, pair_name, ALERT_KEYS["mmh_sell"], False)

        new_state: Dict[str, Any] = {"state": "NO_SIGNAL", "ts": int(time.time())}
        if messages:
            success = await telegram_queue.send_batch(messages)
            if success:
                for key in alerts_to_activate:
                    await set_alert_state(sdb, pair_name, key, True)
                new_state["state"] = "ALERT_SENT"
                logger_pair.info(f"✅ Sent {len(messages)} alerts for {pair_name}")
            else:
                logger_pair.error(f"❌ Failed to send alerts for {pair_name}")

        cloud = "green" if cloud_up else ("red" if cloud_down else "neutral")
        suppression_reason = derive_suppression_reason(context) if new_state["state"] != "ALERT_SENT" else "None"
        new_state["summary"] = {
            "cloud": cloud,
            "mmh_hist": round(mmh_curr, 4),
            "suppression": suppression_reason,
        }
        return pair_name, new_state
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger_pair.exception(f"❌ Error in evaluate_pair_and_alert for {pair_name}: {e}")
        return None
    finally:
        try:
            PAIR_ID.set("")
        except Exception:
            pass

# ---------- Single pair check ----------
async def check_pair(
    pair_name: str,
    fetcher: DataFetcher,
    products_map: Dict[str, dict],
    state_db: RedisStateStore,
    telegram_queue: TelegramQueue,
    correlation_id: str,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    try:
        if shutdown_event.is_set():
            return None
        product_info = products_map.get(pair_name)
        if not product_info:
            logger.warning(f"Product info missing for {pair_name}")
            return None
        symbol = product_info["symbol"]

        limit_15m = 250
        limit_5m = 350
        min_required = 200
        min_required_5m = 250
        daily_limit = cfg.PIVOT_LOOKBACK_PERIOD + 2

        tasks = [
            fetcher.fetch_candles(symbol, "15", limit_15m),
            fetcher.fetch_candles(symbol, "5", limit_5m),
            fetcher.fetch_candles(symbol, "D", daily_limit) if cfg.ENABLE_PIVOT else asyncio.Future(),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        df_15m = parse_candles_result(results[0]) if not isinstance(results[0], Exception) else None
        df_5m = parse_candles_result(results[1]) if not isinstance(results[1], Exception) else None
        df_daily = parse_candles_result(results[2]) if cfg.ENABLE_PIVOT and not isinstance(results[2], Exception) else None

        for res in results:
            if isinstance(res, Exception):
                logger.error(f"Error fetching data for {pair_name}: {res}")
                if PROMETHEUS_ENABLED and METRIC_FETCH_ERRORS:
                    try:
                        METRIC_FETCH_ERRORS.inc()
                    except Exception:
                        pass

        if not validate_candle_df(df_15m, min_required + 2):
            logger.warning(f"Insufficient 15m data for {pair_name}: {len(df_15m) if df_15m is not None else 0}/{min_required+2}")
            return None
        if not validate_candle_df(df_5m, min_required_5m + 2):
            logger.warning(f"Insufficient 5m data for {pair_name}: {len(df_5m) if df_5m is not None else 0}/{min_required_5m+2}")
            return None

        for df in (df_15m, df_5m):
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return await evaluate_pair_and_alert(pair_name, df_15m, df_5m, df_daily, state_db, telegram_queue, correlation_id)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.exception(f"Error in check_pair for {pair_name}: {e}")
        return None

# ---------- Batch processing ----------
async def process_batch(
    fetcher: DataFetcher,
    products_map: Dict[str, dict],
    batch_pairs: List[str],
    state_db: RedisStateStore,
    telegram_queue: TelegramQueue,
    correlation_id: str,
    memory_monitor: MemoryMonitor,
) -> List[Tuple[str, Dict[str, Any]]]:
    tasks = []
    per_pair_timeout = max(15, int(cfg.RUN_TIMEOUT_SECONDS / max(1, len(batch_pairs))))
    for pair_name in batch_pairs:
        if shutdown_event.is_set():
            break
        if pair_name not in products_map:
            logger.warning(f"Skipping {pair_name} – not in products map")
            continue
        tasks.append(
            asyncio.wait_for(
                check_pair(pair_name, fetcher, products_map, state_db, telegram_queue, correlation_id),
                timeout=per_pair_timeout,
            )
        )
    results = await asyncio.gather(*tasks, return_exceptions=True)

    valid_results = []
    failed_pairs = []
    for idx, r in enumerate(results):
        pair_name = batch_pairs[idx] if idx < len(batch_pairs) else "unknown"
        if isinstance(r, Exception):
            logger.error(f"Pair {pair_name} failed: {r}")
            failed_pairs.append(pair_name)
            if PROMETHEUS_ENABLED and METRIC_FAILED_PAIRS:
                try:
                    METRIC_FAILED_PAIRS.inc()
                except Exception:
                    pass
            if health_tracker:
                await health_tracker.record_pair_result(pair_name, False, {"error": str(r)})
        elif r is not None:
            valid_results.append(r)

    if failed_pairs:
        await telegram_queue.send(escape_markdown_v2(f"⚠️ Failed pairs: {', '.join(failed_pairs[:5])}"))

    if health_tracker is not None:
        results_map = {r[0]: r[1] for r in valid_results}
        for pair_name in batch_pairs:
            if pair_name in products_map:
                success = pair_name in results_map and results_map[pair_name].get("state") != "NO_SIGNAL"
                await health_tracker.record_pair_result(pair_name, bool(success), {"source": "process_batch"})
        summary = {"checked": len(batch_pairs), "succeeded": len(valid_results)}
        await health_tracker.record_overall(summary)

    try:
        write_heartbeat()
        await state_db.set_metadata("last_progress", str(int(time.time())))
    except Exception as e:
        logger.debug(f"Heartbeat/meta error: {e}")

    if memory_monitor.should_check():
        used_bytes, mem_percent = memory_monitor.check_memory()
        if PROMETHEUS_ENABLED and METRIC_MEMORY_USAGE:
            try:
                METRIC_MEMORY_USAGE.set(float(used_bytes) / (1024.0 * 1024.0))
            except Exception:
                pass
        if memory_monitor.is_critical():
            logger.critical(f"🚨 Memory critical: {mem_percent:.1f}%")
            await telegram_queue.send(escape_markdown_v2(f"🚨 Memory limit reached ({mem_percent:.1f}%) – aborting batch"))

    return valid_results

# ---------- Run summary ----------
def log_run_summary(all_results: List[Tuple[str, Dict[str, Any]]], pairs_order: List[str]) -> None:
    data_by_pair = {pair: state for pair, state in all_results}
    lines = ["📊 End-of-run summary (cloud, MMH last, suppression):"]
    for pair in pairs_order:
        state = data_by_pair.get(pair)
        if not state or "summary" not in state:
            lines.append(f"- {pair}: N/A")
            continue
        s = state["summary"]
        lines.append(f"- {pair}: ☁️ {s.get('cloud','unknown')}, MMH={s.get('mmh_hist','n/a')}, Suppression={s.get('suppression','None')}")
    for line in lines:
        logger.info(line)

# ---------- Telegram helpers ----------
def build_startup_telegram_message(correlation_id: str) -> str:
    date_str = format_ist_time(datetime.now(timezone.utc), "%d-%m-%Y @%H:%M IST")
    raw = f"🚀 Unified Bot - Run Started\nDate : {date_str}\nCorr. ID: {correlation_id}"
    return escape_markdown_v2(raw)

def build_dms_plugin_message(last_success_meta: Optional[str]) -> Optional[str]:
    if not last_success_meta:
        return None
    ts_int: Optional[int] = None
    try:
        ts_int = int(last_success_meta)
    except Exception:
        try:
            dt = datetime.fromisoformat(last_success_meta)
            ts_int = int(dt.replace(tzinfo=timezone.utc).timestamp())
        except Exception:
            return None
    date_str = format_ist_time(ts_int, "%d-%m-%Y @ %H:%M IST")
    raw = f"📡 [DMS ] Unified Bot\nSuccess: {ts_int} on {date_str}"
    return escape_markdown_v2(raw)

def write_heartbeat(path: str = "/tmp/macd_bot.health") -> None:
    try:
        payload = {
            "ts": int(time.time()),
            "ist": format_ist_time(time.time(), "%d-%m-%Y @%H:%M IST"),
            "version": __version__,
        }
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(payload))
    except Exception as e:
        logger.debug(f"Heartbeat write failed: {e}")

# ---------- Product mapping ----------
def build_products_map_from_api_result(api_products: Optional[Dict[str, Any]]) -> Dict[str, dict]:
    products_map: Dict[str, dict] = {}
    stats = {"total": 0, "skipped": 0, "mapped": 0}

    if not api_products or not api_products.get("result"):
        logger.error(f"No products in API result (keys: {list(api_products.keys()) if isinstance(api_products, dict) else 'n/a'})")
        return products_map

    valid_pattern = re.compile(r'^[A-Z0-9_]+$')

    for p in api_products["result"]:
        try:
            symbol = p.get("symbol", "")
            stats["total"] += 1

            if not valid_pattern.match(symbol):
                stats["skipped"] += 1
                continue

            symbol_norm = symbol.replace("_USDT", "USD").replace("USDT", "USD")
            if p.get("contract_type") == "perpetual_futures":
                for pair_name in cfg.PAIRS:
                    if symbol_norm == pair_name or symbol_norm.replace("_", "") == pair_name:
                        products_map[pair_name] = {
                            "id": p.get("id"),
                            "symbol": p.get("symbol"),
                            "contract_type": p.get("contract_type"),
                        }
                        stats["mapped"] += 1
                        break
        except Exception as e:
            logger.debug(f"Error processing product: {e}")
            continue

    logger.info(f"Product mapping: {stats['mapped']} mapped, {stats['skipped']} skipped, {stats['total']} total")
    return products_map

# ---------- Health HTTP server (optional) ----------
class HealthHttpServer:
    def __init__(self, port: int):
        self.port = port
        self.app: Optional[web.Application] = None
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None

    async def start(self):
        try:
            self.app = web.Application()
            self.app.add_routes([web.get('/health', self.handle_health)])
            self.runner = web.AppRunner(self.app)
            await self.runner.setup()
            self.site = web.TCPSite(self.runner, '0.0.0.0', self.port)
            await self.site.start()
            logger.info(f"Health HTTP server started on port {self.port}")
        except Exception as e:
            logger.warning(f"Failed to start health HTTP server: {e}")

    async def stop(self):
        try:
            if self.site:
                await self.site.stop()
            if self.runner:
                await self.runner.cleanup()
        except Exception:
            pass

    async def handle_health(self, request: web.Request) -> web.Response:
        try:
            vm = psutil.virtual_memory()
            mem_percent = vm.percent
            status = {
                "status": "ok" if mem_percent < Constants.MEMORY_LIMIT_PERCENT and not shutdown_event.is_set() else "degraded",
                "version": __version__,
                "memory_percent": mem_percent,
                "time": format_ist_time(time.time(), "%d-%m-%Y @%H:%M IST"),
            }
            return web.json_response(status)
        except Exception as e:
            return web.json_response({"status": "error", "error": str(e)}, status=500)

health_server: Optional[HealthHttpServer] = None

# ---------- Main run_once ----------
async def run_once() -> bool:
    correlation_id = uuid.uuid4().hex[:8]
    TRACE_ID.set(correlation_id)
    logger_run = logging.getLogger(f"macd_bot.run.{correlation_id}")
    start_time = time.time()
    if PROMETHEUS_ENABLED and METRIC_RUN_DURATION:
        run_timer = METRIC_RUN_DURATION.time()
    else:
        run_timer = None
    memory_monitor = MemoryMonitor(Constants.MEMORY_LIMIT_PERCENT)

    try:
        if HEALTH_SERVER_ENABLED:
            try:
                global health_server
                health_server = HealthHttpServer(HEALTH_HTTP_PORT)
                asyncio.create_task(health_server.start())
            except Exception as e:
                logger_run.warning(f"Health server start failed: {e}")

        if memory_monitor.is_critical():
            mem_percent = psutil.virtual_memory().percent
            logger_run.critical(f"🚨 Memory limit exceeded at startup: {mem_percent:.1f}%")
            return False

        fetcher = DataFetcher(cfg.DELTA_API_BASE)
        telegram_queue = TelegramQueue(cfg.TELEGRAM_BOT_TOKEN, cfg.TELEGRAM_CHAT_ID)

        logger_run.info(f"🔌 Circuit breaker config: timeout={fetcher.circuit_breaker.timeout}s, threshold={fetcher.circuit_breaker.threshold}")

        if await fetcher.circuit_breaker.is_open():
            logger_run.critical("🔴 Circuit breaker OPEN at startup")
            await telegram_queue.send(escape_markdown_v2("🚨 Circuit breaker open – check API status"))
            return False

        async with RedisStateStore(cfg.REDIS_URL) as sdb:
            global health_tracker
            health_tracker = HealthTracker(sdb)

            lock = RedisLock(sdb._redis, "macd_bot_run")
            if not await lock.acquire(timeout=5.0):
                logger_run.warning("❌ Another instance is running (Redis lock held)")
                return False

            try:
                dms = DeadMansSwitch(sdb, cfg.DEAD_MANS_COOLDOWN_SECONDS)
                dms_result = await dms.should_alert()
                if dms_result == "RECOVERED":
                    recovery_msg = escape_markdown_v2(f"✅ {cfg.BOT_NAME} - BOT RECOVERED\nBot is running normally again.")
                    await telegram_queue.send(recovery_msg)
                elif dms_result is True:
                    dms_message = escape_markdown_v2(
                        f"⚠️ {cfg.BOT_NAME} - DEAD MAN'S SWITCH ALERT\n"
                        f"Bot has not successfully run in over {cfg.DEAD_MANS_COOLDOWN_SECONDS / 3600:.1f} hours."
                    )
                    await telegram_queue.send(dms_message)

                try:
                    await sdb.set_metadata("dms:last_alert_ts", str(int(time.time())))
                    if cfg.ENABLE_DMS_PLUGIN:
                        last_success_meta = await sdb.get_metadata("last_success_run")
                        dms_plugin_msg = build_dms_plugin_message(last_success_meta)
                        if dms_plugin_msg:
                            await telegram_queue.send(dms_plugin_msg)
                except Exception as e:
                    logger_run.debug(f"DMS plugin error: {e}")

                await sdb._prune_old_records(cfg.STATE_EXPIRY_DAYS)

                if cfg.SEND_TEST_MESSAGE:
                    logger_run.info("Sending Telegram startup message...")
                    start_msg = build_startup_telegram_message(correlation_id)
                    success = await telegram_queue.send(start_msg)
                    if success:
                        logger_run.info("✅ Startup message sent successfully")
                    else:
                        logger_run.error("❌ Startup message failed")

                logger_run.info("Fetching products from API...")
                prod_resp = await fetcher.fetch_products()
                if not prod_resp:
                    logger_run.error("❌ Failed to fetch products")
                    return False

                products_map = build_products_map_from_api_result(prod_resp)
                if not products_map:
                    logger_run.error("❌ No tradable pairs found")
                    return False

                pairs_to_process = [p for p in cfg.PAIRS if p in products_map]
                batch_size = max(1, cfg.BATCH_SIZE)
                logger_run.info(f"📊 Processing {len(pairs_to_process)} pairs in batches of {batch_size}")

                all_results: List[Tuple[str, Dict[str, Any]]] = []
                for i in range(0, len(pairs_to_process), batch_size):
                    if shutdown_event.is_set():
                        logger_run.info("🛑 Shutdown signaled, aborting")
                        break
                    elapsed = time.time() - start_time
                    if elapsed > cfg.RUN_TIMEOUT_SECONDS:
                        logger_run.warning(f"⏱️ Run timeout ({cfg.RUN_TIMEOUT_SECONDS}s) exceeded at {elapsed:.1f}s")
                        break
                    batch_pairs = pairs_to_process[i : i + batch_size]
                    batch_num = i // batch_size + 1
                    total_batches = (len(pairs_to_process) + batch_size - 1) // batch_size
                    logger_run.info(f"🔄 Processing batch {batch_num}/{total_batches}: {batch_pairs}")
                    batch_results = await process_batch(
                        fetcher, products_map, batch_pairs, sdb,
                        telegram_queue, correlation_id, memory_monitor
                    )
                    all_results.extend(batch_results)

                    if memory_monitor.is_critical():
                        _, mem_after = memory_monitor.check_memory()
                        logger_run.critical(f"🚨 Memory critical: {mem_after:.1f}% – aborting")
                        await telegram_queue.send(
                            escape_markdown_v2(f"🚨 Memory limit reached ({mem_after:.1f}%) – aborting run")
                        )
                        break

                    if not await lock.extend(timeout=3.0):
                        logger_run.warning("⚠️ Failed to extend Redis lock")
                        break

                    if i + batch_size < len(pairs_to_process):
                        await asyncio.sleep(1)

                await sdb.set_metadata("last_success_run", str(int(time.time())))

                alerts_sent = sum(1 for r in all_results if r[1].get("state") == "ALERT_SENT")
                elapsed = time.time() - start_time
                logger_run.info(f"✅ Run complete: {alerts_sent} alerts sent in {elapsed:.2f}s")
                log_run_summary(all_results, pairs_to_process)

            finally:
                await lock.release(timeout=3.0)

        await SessionManager.close_session()

        if run_timer:
            try:
                run_timer.observe(time.time() - start_time)
            except Exception:
                pass

        if random.random() < 0.1:
            gc.collect()
            _, mem_after = memory_monitor.check_memory()
            logger_run.debug(f"🗑️ GC completed, memory: {mem_after:.1f}%")

        return True

    except asyncio.CancelledError:
        logger_run.info("🛑 Run cancelled")
        try:
            await SessionManager.close_session()
        except Exception:
            pass
        return False
    except Exception as e:
        logger_run.exception(f"❌ Fatal error in run_once: {e}")
        try:
            await SessionManager.close_session()
        except Exception:
            pass
        try:
            await cancel_all_tasks(grace_seconds=5)
        except Exception:
            pass
        return False
    finally:
        try:
            TRACE_ID.set("")
        except Exception:
            pass

# ---------- Optional uvloop integration ----------
try:
    import uvloop, asyncio
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    logger.info("✅ uvloop enabled as asyncio event loop")
except ImportError:
    logger.info("ℹ️ uvloop not installed, using default asyncio loop")
except Exception as e:
    logger.warning(f"⚠️ uvloop could not be enabled: {e}")

# ---------- Runtime config validation ----------
def validate_config_runtime() -> None:
    if cfg.RUN_TIMEOUT_SECONDS <= 0:
        raise ValueError("RUN_TIMEOUT_SECONDS must be positive")
    if cfg.BATCH_SIZE > len(cfg.PAIRS):
        logger.warning(f"⚠️ BATCH_SIZE ({cfg.BATCH_SIZE}) > PAIRS count ({len(cfg.PAIRS)})")
    if cfg.MEMORY_LIMIT_BYTES < 100_000_000:
        logger.warning(f"⚠️ MEMORY_LIMIT_BYTES seems very low: {cfg.MEMORY_LIMIT_BYTES/1024/1024:.0f}MB")
    if Constants.REDIS_LOCK_EXPIRY <= cfg.RUN_TIMEOUT_SECONDS:
        logger.critical("❌ REDIS_LOCK_EXPIRY must be greater than RUN_TIMEOUT_SECONDS")
        raise ValueError(
            f"Invalid lock expiry: {Constants.REDIS_LOCK_EXPIRY}s <= {cfg.RUN_TIMEOUT_SECONDS}s. "
            f"Set REDIS_LOCK_EXPIRY env var to at least {cfg.RUN_TIMEOUT_SECONDS + 300}."
        )
    if cfg.TELEGRAM_RATE_LIMIT_PER_MINUTE <= 0:
        raise ValueError("TELEGRAM_RATE_LIMIT_PER_MINUTE must be positive")
    if cfg.TELEGRAM_BURST_SIZE > cfg.TELEGRAM_RATE_LIMIT_PER_MINUTE:
        logger.warning("⚠️ TELEGRAM_BURST_SIZE > rate limit")
    logger.info(f"✅ Runtime config validation passed for bot v{__version__}")
    logger.info(f"📊 Config: {len(cfg.PAIRS)} pairs, {cfg.BATCH_SIZE} batch size, {cfg.RUN_TIMEOUT_SECONDS}s timeout")
    logger.info(f"🔒 Redis lock expiry: {Constants.REDIS_LOCK_EXPIRY}s (safe margin: {Constants.REDIS_LOCK_EXPIRY - cfg.RUN_TIMEOUT_SECONDS}s)")
    logger.info(f"💬 Telegram rate limit: {cfg.TELEGRAM_RATE_LIMIT_PER_MINUTE}/min, burst: {cfg.TELEGRAM_BURST_SIZE}")
    logger.info(f"🔗 API base: {cfg.DELTA_API_BASE}")

# ---------- Entrypoint ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="macd_unified_cron", description="Unified MACD/alerts runner (cron-ready)")
    parser.add_argument("--once", action="store_true", help="Run one pass and exit (useful for testing)")
    parser.add_argument("--debug", action="store_true", help="Set logger to DEBUG for this run")
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
        for h in logger.handlers:
            h.setLevel(logging.DEBUG)

    validate_config_runtime()

    try:
        # +60s global guard helps CI avoid stuck runs, while inner RUN_TIMEOUT_SECONDS controls job’s length
        success = asyncio.run(asyncio.wait_for(run_once(), timeout=cfg.RUN_TIMEOUT_SECONDS + 60))
        sys.exit(0 if success else 2)
    except asyncio.TimeoutError:
        logger.error(f"⏱️ Global run timeout exceeded ({cfg.RUN_TIMEOUT_SECONDS + 60}s)")
        try:
            asyncio.run(SessionManager.close_session())
        except Exception:
            pass
        sys.exit(2)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        try:
            asyncio.run(SessionManager.close_session())
        except Exception:
            pass
        sys.exit(130)
