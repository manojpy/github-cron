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
import math
import gc
from collections import deque, defaultdict
from typing import Dict, Any, Optional, Tuple, List, ClassVar, TypedDict, Callable
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from contextvars import ContextVar
from urllib.parse import urlparse, parse_qs

import aiohttp
from aiohttp import web
import numpy as np
import redis.asyncio as redis
from redis.exceptions import ConnectionError as RedisConnectionError, RedisError
from pydantic import BaseModel, Field, field_validator, model_validator
from aiohttp import ClientConnectorError, ClientResponseError, TCPConnector, ClientError
from numba import njit, prange
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning, module='pycparser')
warnings.filterwarnings('ignore', message='.*parsing methods must have __doc__.*')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MACD_BOT")

try:
    import numba_compiled
    USE_AOT = True
    logger.info("✅ AOT modules loaded")
except ImportError:
    USE_AOT = False
    logger.info("ℹ️ AOT modules not found, using JIT fallback")

try:
    import orjson
    
    def json_dumps(obj: Any) -> str:
        return orjson.dumps(obj, option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS).decode('utf-8')

    def json_loads(s: str | bytes) -> Any:
        return orjson.loads(s)
    
    JSON_BACKEND = "orjson"
except ImportError:
    import json
    json_dumps = json.dumps
    json_loads = json.loads
    JSON_BACKEND = "stdlib"

__version__ = "1.4.0-performance-optimized"

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
    MMH_VALUE_CLIP = 0.9999999
    ZERO_DIVISION_GUARD = 1e-12
    INFINITY_CLAMP = 1e8
    TELEGRAM_MAX_MESSAGE_LENGTH = 3800
    TELEGRAM_MESSAGE_PREVIEW_LENGTH = 50
    MAX_PIVOT_DISTANCE_PCT = 100.0

# ============================================================================
# STATIC PRODUCTS MAP (Eliminates API fetch overhead)
# ============================================================================
STATIC_PRODUCTS_MAP = {
    "BTCUSD": {"id": 139, "symbol": "BTCUSDT", "contract_type": "perpetual_futures"},
    "ETHUSD": {"id": 140, "symbol": "ETHUSDT", "contract_type": "perpetual_futures"},
    "AVAXUSD": {"id": 262, "symbol": "AVAXUSDT", "contract_type": "perpetual_futures"},
    "BCHUSD": {"id": 186, "symbol": "BCHUSDT", "contract_type": "perpetual_futures"},
    "XRPUSD": {"id": 141, "symbol": "XRPUSDT", "contract_type": "perpetual_futures"},
    "BNBUSD": {"id": 275, "symbol": "BNBUSDT", "contract_type": "perpetual_futures"},
    "LTCUSD": {"id": 163, "symbol": "LTCUSDT", "contract_type": "perpetual_futures"},
    "DOTUSD": {"id": 228, "symbol": "DOTUSDT", "contract_type": "perpetual_futures"},
    "ADAUSD": {"id": 142, "symbol": "ADAUSDT", "contract_type": "perpetual_futures"},
    "SUIUSD": {"id": 666, "symbol": "SUIUSDT", "contract_type": "perpetual_futures"},
    "AAVEUSD": {"id": 227, "symbol": "AAVEUSDT", "contract_type": "perpetual_futures"},
    "SOLUSD": {"id": 143, "symbol": "SOLUSDT", "contract_type": "perpetual_futures"},
}

# Validation will happen after cfg is loaded

PIVOT_LEVELS = ["P", "S1", "S2", "S3", "R1", "R2", "R3"]

class CompiledPatterns:
    VALID_SYMBOL = re.compile(r'^[A-Z0-9_]+$')
    ESCAPE_MARKDOWN = re.compile(r'[_*\[\]()~`>#+-=|{}.!]')
    SECRET_TOKEN = re.compile(r'\b\d{6,}:[A-Za-z0-9_-]{20,}\b')
    CHAT_ID = re.compile(r'chat_id=\d+')
    REDIS_CREDS = re.compile(r'(redis://[^@]+@)')

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

# ============================================================================
# OPTIMIZATION 1: Enhanced BotConfig with performance tuning
# ============================================================================
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
    
    # OPTIMIZED: Increased parallel fetch for better throughput
    MAX_PARALLEL_FETCH: int = Field(12, ge=1, le=20)
    HTTP_TIMEOUT: int = 15
    CANDLE_FETCH_RETRIES: int = 3
    CANDLE_FETCH_BACKOFF: float = 1.5
    JITTER_MIN: float = 0.1
    JITTER_MAX: float = 0.8
    RUN_TIMEOUT_SECONDS: int = 600
    BATCH_SIZE: int = 4
    
    # OPTIMIZED: Increased connection limits for parallel operations
    TCP_CONN_LIMIT: int = 16
    TCP_CONN_LIMIT_PER_HOST: int = 12
    
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
    
    # NEW: Performance tuning options
    NUMBA_PARALLEL: bool = Field(default=True, description="Enable Numba parallel execution")
    SKIP_WARMUP: bool = Field(default=False, description="Skip Numba warmup (faster startup)")
    PRODUCTS_CACHE_TTL: int = Field(default=28800, description="Products cache TTL in seconds (8 hours)")

    PIVOT_MAX_DISTANCE_PCT: float = Field(
        default=100.0,
        ge=10.0,
        le=500.0,
        description="Maximum allowed distance (%) between current price and pivot level for alert validity"
    )

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

# Validate STATIC_PRODUCTS_MAP after cfg is loaded
_missing_products = set(cfg.PAIRS) - set(STATIC_PRODUCTS_MAP.keys())
if _missing_products:
    logger.warning(
        f"⚠️ Missing product mappings for: {_missing_products}. "
        f"Add them to STATIC_PRODUCTS_MAP or remove from PAIRS config."
    )

# Logging setup (same as original)
class SecretFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = str(record.getMessage())
            if any(x in msg for x in ("TOKEN", "redis://", "chat_id")):
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
    def format(self, record: logging.LogRecord) -> str:
        record.args = None
        formatted = super().format(record)
        formatted = CompiledPatterns.SECRET_TOKEN.sub("[REDACTED_TOKEN]", formatted)
        formatted = CompiledPatterns.CHAT_ID.sub("chat_id=[REDACTED]", formatted)
        formatted = CompiledPatterns.REDIS_CREDS.sub("redis://[REDACTED]@", formatted)
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
        fmt='%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s | [%(trace_id)s] | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))    
    console.addFilter(SecretFilter())
    console.addFilter(TraceContextFilter())  
    logger.addHandler(console)    
    logger.debug(
        f"Logging configured | Level: {logging.getLevelName(level)} | "
        f"Format: structured with trace_id | Output: stdout"
    )

    return logger

logger = setup_logging()
shutdown_event = asyncio.Event()

def debug_if(condition: bool, logger_obj: logging.Logger, msg_fn: Callable[[], str]) -> None:
    if condition and logger_obj.isEnabledFor(logging.DEBUG):
        logger_obj.debug(msg_fn())
        

def info_if_important(logger_obj: logging.Logger, is_important: bool, msg: str) -> None:
    if is_important:
        logger_obj.info(msg)
    elif cfg.DEBUG_MODE:
        logger_obj.debug(msg)


_VALIDATION_DONE = False

def validate_runtime_config() -> None:
    global _VALIDATION_DONE
    if _VALIDATION_DONE:
        logger.debug("Configuration validation skipped (cached)")
        return
    
    errors = []
    warnings = []
    
    try:
        from urllib.parse import urlparse
        parsed = urlparse(cfg.REDIS_URL)
        if parsed.scheme not in ('redis', 'rediss'):
            errors.append(f"Invalid REDIS_URL scheme: {parsed.scheme} (must be redis:// or rediss://)")
        if not parsed.hostname:
            errors.append("REDIS_URL missing hostname")
    except Exception as e:
        errors.append(f"Failed to parse REDIS_URL: {e}")
    
    if not CompiledPatterns.SECRET_TOKEN.match(cfg.TELEGRAM_BOT_TOKEN):
        errors.append("TELEGRAM_BOT_TOKEN format invalid (should be: 123456:ABC-DEF...)")
    
    if not cfg.DELTA_API_BASE.startswith(('http://', 'https://')):
        errors.append("DELTA_API_BASE must start with http:// or https://")
    
    if not cfg.PAIRS or len(cfg.PAIRS) == 0:
        errors.append("PAIRS list is empty - no trading pairs configured")
    
    if cfg.MAX_PARALLEL_FETCH < 1 or cfg.MAX_PARALLEL_FETCH > 20:
        warnings.append(f"MAX_PARALLEL_FETCH={cfg.MAX_PARALLEL_FETCH} is outside recommended range (1-20)")
    
    if cfg.HTTP_TIMEOUT < 5 or cfg.HTTP_TIMEOUT > 60:
        warnings.append(f"HTTP_TIMEOUT={cfg.HTTP_TIMEOUT}s is outside recommended range (5-60s)")
    
    if len(cfg.PAIRS) > 20:
        warnings.append(f"Large number of pairs ({len(cfg.PAIRS)}) may exceed timeout limits")
    
    if cfg.MEMORY_LIMIT_BYTES < 200_000_000:
        warnings.append(f"MEMORY_LIMIT_BYTES={cfg.MEMORY_LIMIT_BYTES} is very low (minimum recommended: 200MB)")
    
    if errors:
        logger.critical("Configuration validation FAILED:")
        for error in errors:
            logger.critical(f"  ERROR: {error}")
        raise ValueError(f"Configuration validation failed with {len(errors)} error(s)")
    
    if warnings:
        logger.warning("Configuration warnings:")
        for warning in warnings:
            logger.warning(f"  WARNING: {warning}")
    
    logger.info(
        f"Configuration validated successfully | "
        f"Pairs: {len(cfg.PAIRS)} | Workers: {cfg.MAX_PARALLEL_FETCH} | "
        f"Timeout: {cfg.RUN_TIMEOUT_SECONDS}s"
    )
    
    _VALIDATION_DONE = True

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
    
    effective_time = reference_time - 45 
    
    current_period_start = (effective_time // interval_seconds) * interval_seconds
    last_closed_candle_open = current_period_start - interval_seconds
    
    return last_closed_candle_open

_ESCAPE_RE = re.compile(r'[_*\[\]()~`>#+-=|{}.!]')

def escape_markdown_v2(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    return CompiledPatterns.ESCAPE_MARKDOWN.sub(r'\\\g<0>', text)

def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
            return default
        return float(value)
    except (ValueError, TypeError, OverflowError):
        return default

# ============================================================================
# OPTIMIZATION 2: Parallel Numba Functions with prange
# ============================================================================

@njit(nogil=True, fastmath=True, cache=True, parallel=True)
def _sanitize_array_numba_parallel(arr: np.ndarray, default: float) -> np.ndarray:
    """OPTIMIZED: Parallel sanitization with prange"""
    out = np.empty_like(arr)
    for i in prange(len(arr)):  # PARALLEL LOOP
        val = arr[i]
        if np.isnan(val) or np.isinf(val):
            out[i] = default
        else:
            out[i] = val
    return out

@njit(nogil=True, fastmath=True, cache=True)
def _sanitize_array_numba(arr: np.ndarray, default: float) -> np.ndarray:
    """Original serial version for small arrays"""
    out = np.empty_like(arr)
    for i in range(len(arr)):
        val = arr[i]
        if np.isnan(val) or np.isinf(val):
            out[i] = default
        else:
            out[i] = val
    return out

def sanitize_indicator_array(arr: np.ndarray, name: str, default: float = 0.0) -> np.ndarray:
    """OPTIMIZED: Uses smart wrapper with AOT support and adaptive thresholds"""
    try:
        if arr is None or len(arr) == 0:
            logger.warning(f"Indicator {name} is None or empty")
            return np.array([default], dtype=np.float64)
        
        # ⚡ Use smart wrapper (handles AOT + parallel threshold logic)
        sanitized = sanitize_array_wrapper(arr, default)
        
        if np.all(sanitized == default):
            logger.warning(f"Indicator {name} is all {default} after sanitization")
        
        return sanitized
        
    except Exception as e:
        logger.error(f"Failed to sanitize indicator {name}: {e}")
        return np.full(len(arr) if arr is not None else 1, default, dtype=np.float64)
        
@njit(nogil=True, fastmath=True, cache=True, parallel=True)
def _sma_loop_parallel(data: np.ndarray, period: int) -> np.ndarray:
    """OPTIMIZED: Parallel SMA calculation"""
    n = len(data)
    out = np.empty(n, dtype=np.float64)
    out[:] = np.nan
    min_periods = max(2, period // 3)
    
    for i in prange(n):  # PARALLEL LOOP
        window_sum = 0.0
        count = 0
        
        start = max(0, i - period + 1)
        for j in range(start, i + 1):
            val = data[j]
            if not np.isnan(val):
                window_sum += val
                count += 1
        
        if count >= min_periods:
            out[i] = window_sum / count
        else:
            out[i] = np.nan
            
    return out

@njit(nogil=True, fastmath=True, cache=True)
def _sma_loop(data: np.ndarray, period: int) -> np.ndarray:
    """Original serial SMA (fallback)"""
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

@njit(nogil=True, fastmath=True, cache=True)
def _ema_loop(data: np.ndarray, alpha: float) -> np.ndarray:
    """EMA is inherently serial (each value depends on previous)"""
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

@njit(nogil=True, fastmath=True, cache=True)
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

@njit(nogil=True, fastmath=True, cache=True)
def _vwap_daily_loop(
    high: np.ndarray, 
    low: np.ndarray, 
    close: np.ndarray, 
    volume: np.ndarray, 
    timestamps: np.ndarray
) -> np.ndarray:
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

@njit(nogil=True, fastmath=True, cache=True)
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


# ============================================================================
# PART 3: Optimized MMH and Indicator Calculations
# ============================================================================


@njit(nogil=True, fastmath=True, cache=True)
def _calc_mmh_worm_loop(close_arr: np.ndarray, sd_arr: np.ndarray, rows: int) -> np.ndarray:
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

@njit(nogil=True, fastmath=True, cache=True)
def _calc_mmh_value_loop(temp_arr: np.ndarray, rows: int) -> np.ndarray:
    value_arr = np.zeros(rows, dtype=np.float64)
    value_arr[0] = 1.0
    
    for i in range(1, rows):
        prev_v = value_arr[i - 1] if not np.isnan(value_arr[i - 1]) else 1.0
        t = temp_arr[i] if not np.isnan(temp_arr[i]) else 0.5
        v = t - 0.5 + 0.5 * prev_v
        value_arr[i] = max(-0.9999, min(0.9999, v))
    
    return value_arr

@njit(nogil=True, fastmath=True, cache=True)
def _calc_mmh_momentum_loop(momentum_arr: np.ndarray, rows: int) -> np.ndarray:
    for i in range(1, rows):
        prev = momentum_arr[i - 1] if not np.isnan(momentum_arr[i - 1]) else 0.0
        momentum_arr[i] = momentum_arr[i] + 0.5 * prev
    return momentum_arr

# ============================================================================
# OPTIMIZATION 3: Welford's Algorithm with Parallel Preprocessing
# ============================================================================

@njit(nogil=True, fastmath=True, cache=True, parallel=True)
def _rolling_std_welford_parallel(close: np.ndarray, period: int, responsiveness: float) -> np.ndarray:
    """
    OPTIMIZED: Parallel rolling standard deviation using Welford's algorithm.
    Uses prange for independent window calculations.
    """
    n = len(close)
    sd = np.empty(n, dtype=np.float64)
    resp = max(0.00001, min(1.0, responsiveness))
    
    # Parallel loop - each window is independent
    for i in prange(n):
        mean = 0.0
        m2 = 0.0
        count = 0
        
        start = max(0, i - period + 1)
        for j in range(start, i + 1):
            val = close[j]
            if not np.isnan(val):
                count += 1
                delta = val - mean
                mean += delta / count
                delta2 = val - mean
                m2 += delta * delta2
        
        if count > 1:
            variance = m2 / count
            sd[i] = np.sqrt(max(0.0, variance)) * resp
        else:
            sd[i] = 0.0
    
    return sd

@njit(nogil=True, fastmath=True, cache=True)
def _rolling_std_welford(close: np.ndarray, period: int, responsiveness: float) -> np.ndarray:
    """Original serial version (fallback)"""
    n = len(close)
    sd = np.empty(n, dtype=np.float64)
    resp = max(0.00001, min(1.0, responsiveness))
    
    for i in range(n):
        mean = 0.0
        m2 = 0.0
        count = 0
        
        start = max(0, i - period + 1)
        for j in range(start, i + 1):
            val = close[j]
            if not np.isnan(val):
                count += 1
                delta = val - mean
                mean += delta / count
                delta2 = val - mean
                m2 += delta * delta2
        
        if count > 1:
            variance = m2 / count
            sd[i] = np.sqrt(max(0.0, variance)) * resp
        else:
            sd[i] = 0.0
    
    return sd


@njit(nogil=True, fastmath=True, cache=True)
def _rolling_mean_numba(close: np.ndarray, period: int) -> np.ndarray:
    """Original serial version"""
    rows = len(close)
    ma = np.empty(rows, dtype=np.float64)
    for i in range(rows):
        start = max(0, i - period + 1)
        sum_val = 0.0
        count = 0
        for j in range(start, i + 1):
            val = close[j]
            if not np.isnan(val):
                sum_val += val
                count += 1
        ma[i] = sum_val / count if count > 0 else 0.0
    return ma

@njit(nogil=True, fastmath=True, cache=True, parallel=True)
def _rolling_min_max_numba_parallel(arr: np.ndarray, period: int) -> Tuple[np.ndarray, np.ndarray]:
    """OPTIMIZED: Parallel rolling min/max"""
    rows = len(arr)
    min_arr = np.empty(rows, dtype=np.float64)
    max_arr = np.empty(rows, dtype=np.float64)
    
    for i in prange(rows):
        start = max(0, i - period + 1)
        min_arr[i] = np.min(arr[start:i+1])
        max_arr[i] = np.max(arr[start:i+1])
    
    return min_arr, max_arr

@njit(nogil=True, fastmath=True, cache=True)
def _rolling_min_max_numba(arr: np.ndarray, period: int) -> Tuple[np.ndarray, np.ndarray]:
    """Original serial version"""
    rows = len(arr)
    min_arr = np.empty(rows, dtype=np.float64)
    max_arr = np.empty(rows, dtype=np.float64)
    for i in range(rows):
        start = max(0, i - period + 1)
        min_arr[i] = np.min(arr[start:i+1])
        max_arr[i] = np.max(arr[start:i+1])
    return min_arr, max_arr

# ============================================================================
# OPTIMIZATION 4: Streamlined PPO/RSI with Reduced Allocations
# ============================================================================

@njit(nogil=True, fastmath=True, cache=True)
def _calculate_ppo_core(close: np.ndarray, fast: int, slow: int, signal: int) -> Tuple[np.ndarray, np.ndarray]:
    """Core PPO calculation - fully compiled with GIL released."""
    n = len(close)
    
    alpha_fast = 2.0 / (fast + 1)
    fast_ma = _ema_loop(close, alpha_fast)
    
    alpha_slow = 2.0 / (slow + 1)
    slow_ma = _ema_loop(close, alpha_slow)
    
    # OPTIMIZED: Pre-allocate and fill in one pass
    ppo = np.empty(n, dtype=np.float64)
    for i in range(n):
        if abs(slow_ma[i]) < 1e-12:
            ppo[i] = 0.0
        else:
            ppo[i] = ((fast_ma[i] - slow_ma[i]) / slow_ma[i]) * 100.0
    
    alpha_signal = 2.0 / (signal + 1)
    ppo_sig = _ema_loop(ppo, alpha_signal)
    
    return ppo, ppo_sig

@njit(nogil=True, fastmath=True, cache=True)
def _calculate_rsi_core(close: np.ndarray, rsi_len: int) -> np.ndarray:
    """Core RSI calculation with GIL released."""
    n = len(close)
    
    # OPTIMIZED: Combined delta calculation and gain/loss separation
    delta = np.zeros(n, dtype=np.float64)
    gain = np.zeros(n, dtype=np.float64)
    loss = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        delta[i] = close[i] - close[i-1]
        if delta[i] > 0:
            gain[i] = delta[i]
        elif delta[i] < 0:
            loss[i] = -delta[i]
    
    alpha = 1.0 / rsi_len
    avg_gain = _ema_loop(gain, alpha)
    avg_loss = _ema_loop(loss, alpha)
    
    rsi = np.empty(n, dtype=np.float64)
    for i in range(n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_smooth_rsi_numpy(close: np.ndarray, rsi_len: int, kalman_len: int) -> np.ndarray:
    try:
        if close is None or len(close) < rsi_len:
            logger.warning(f"Smooth RSI: Insufficient data")
            return np.full(len(close) if close is not None else 1, 50.0, dtype=np.float64)
        
        rsi = _calculate_rsi_core(close, rsi_len)
        smooth_rsi = _kalman_loop(rsi, kalman_len, 0.01, 0.1)
        
        # ⚡ Use smart wrapper (handles AOT + threshold 300)
        smooth_rsi = sanitize_array_wrapper(smooth_rsi, 50.0)
        
        return smooth_rsi
        
    except Exception as e:
        logger.error(f"Smooth RSI calculation failed: {e}")
        return np.full(len(close) if close is not None else 1, 50.0, dtype=np.float64)
 
def calculate_ppo_numpy(close: np.ndarray, fast: int, slow: int, signal: int) -> Tuple[np.ndarray, np.ndarray]:
    try:
        if close is None or len(close) < max(fast, slow):
            logger.warning(f"PPO: Insufficient data")
            default_len = len(close) if close is not None else 1
            return np.zeros(default_len, dtype=np.float64), np.zeros(default_len, dtype=np.float64)
        
        ppo, ppo_sig = _calculate_ppo_core(close, fast, slow, signal)
        
        # ⚡ Use smart wrapper (handles AOT + threshold 300)
        ppo = sanitize_array_wrapper(ppo, 0.0)
        ppo_sig = sanitize_array_wrapper(ppo_sig, 0.0)
        
        return ppo, ppo_sig
    except Exception as e:
        logger.error(f"PPO calculation failed: {e}")
        default_len = len(close) if close is not None else 1
        return np.zeros(default_len, dtype=np.float64), np.zeros(default_len, dtype=np.float64)

def calculate_vwap_numpy(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                         volume: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    try:
        if any(x is None or len(x) == 0 for x in [high, low, close, volume, timestamps]):
            logger.warning("VWAP: Missing or empty input arrays")
            return np.zeros_like(close) if close is not None else np.array([0.0])
        
        vwap = _vwap_daily_loop(high, low, close, volume, timestamps)
        vwap = sanitize_indicator_array(vwap, "VWAP", default=close[-1] if len(close) > 0 else 0.0)
        return vwap
        
    except Exception as e:
        logger.error(f"VWAP calculation failed: {e}")
        return np.zeros_like(close) if close is not None else np.array([0.0])

def calculate_rma_numpy(data: np.ndarray, period: int) -> np.ndarray:
    try:
        if data is None or len(data) < period:
            logger.warning(f"RMA: Insufficient data (len={len(data) if data is not None else 0})")
            return np.zeros_like(data) if data is not None else np.array([0.0])
        
        alpha = 1.0 / period
        rma = _ema_loop(data, alpha)
        rma = sanitize_indicator_array(rma, f"RMA_{period}", default=0.0)
        return rma
        
    except Exception as e:
        logger.error(f"RMA calculation failed: {e}")
        return np.zeros_like(data) if data is not None else np.array([0.0])

def calculate_cirrus_cloud_numba(close: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        if close is None or len(close) < max(cfg.X1, cfg.X3):
            logger.warning(f"Cirrus Cloud: Insufficient data (len={len(close) if close is not None else 0})")
            default_len = len(close) if close is not None else 1
            return (np.zeros(default_len, dtype=bool), 
                    np.zeros(default_len, dtype=bool),
                    np.zeros(default_len, dtype=np.float64),
                    np.zeros(default_len, dtype=np.float64))
        
        diff = np.zeros_like(close)
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
        
    except Exception as e:
        logger.error(f"Cirrus Cloud calculation failed: {e}")
        default_len = len(close) if close is not None else 1
        return (np.zeros(default_len, dtype=bool), 
                np.zeros(default_len, dtype=bool),
                np.zeros(default_len, dtype=np.float64),
                np.zeros(default_len, dtype=np.float64))

# ============================================================================
# OPTIMIZATION 5: Streamlined MMH with Smart Parallel Execution
# ============================================================================

def calculate_magical_momentum_hist(close: np.ndarray, period: int = 144, responsiveness: float = 0.9) -> np.ndarray:
    """OPTIMIZED: MMH with conditional parallel execution"""
    try:
        if close is None or len(close) < period:
            logger.warning(f"MMH: Insufficient data (len={len(close) if close is not None else 0})")
            return np.zeros(len(close) if close is not None else 1, dtype=np.float64)
        
        rows = len(close)
        resp_clamped = max(0.00001, min(1.0, float(responsiveness)))
        
        close_c = np.ascontiguousarray(close) if not close.flags['C_CONTIGUOUS'] else close
        
        sd = rolling_std_wrapper(close_c, 50, resp_clamped)
    
        worm_arr = _calc_mmh_worm_loop(close_c, sd, rows)
    
        
        ma = rolling_mean_wrapper(close_c, period)

        with np.errstate(divide='ignore', invalid='ignore'):
            raw = (worm_arr - ma) / worm_arr
        
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    
        min_med, max_med = rolling_minmax_wrapper(raw, period)
        
        denom = max_med - min_med
        denom = np.where(denom == 0, Constants.ZERO_DIVISION_GUARD, denom)
        temp = (raw - min_med) / denom
        temp = np.clip(temp, 0.0, 1.0)
        temp = np.nan_to_num(temp, nan=0.5)
        
        value_arr = _calc_mmh_value_loop(temp, rows)
        value_arr = np.clip(value_arr, -Constants.MMH_VALUE_CLIP, Constants.MMH_VALUE_CLIP)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            temp2 = (1.0 + value_arr) / (1.0 - value_arr)
            temp2 = np.nan_to_num(temp2, nan=1e8, posinf=1e8, neginf=-1e8)
        
        momentum = 0.25 * np.log(temp2)
        momentum = np.nan_to_num(momentum, nan=0.0)
        
        momentum_arr = momentum.copy()
        momentum_arr = _calc_mmh_momentum_loop(momentum_arr, rows)
        
        momentum_arr = sanitize_indicator_array(momentum_arr, "MMH_Hist", default=0.0)
        
        return momentum_arr
        
    except Exception as e:
        logger.error(f"MMH calculation failed: {e}")
        return np.zeros(len(close) if close is not None else 1, dtype=np.float64)

@njit(nogil=True, fastmath=True, cache=True)
def _vectorized_wick_check_buy(
    open_arr: np.ndarray, 
    high_arr: np.ndarray, 
    low_arr: np.ndarray, 
    close_arr: np.ndarray,
    min_wick_ratio: float
) -> np.ndarray:
    
    n = len(close_arr)
    result = np.zeros(n, dtype=np.bool_)
    
    for i in range(n):
        o, h, l, c = open_arr[i], high_arr[i], low_arr[i], close_arr[i]
        
        if c <= o:
            result[i] = False
            continue
        
        candle_range = h - l
        if candle_range < 1e-8:
            result[i] = False
            continue
        
        upper_wick = h - c
        wick_ratio = upper_wick / candle_range
        
        result[i] = wick_ratio < min_wick_ratio
    
    return result

@njit(nogil=True, fastmath=True, cache=True)
def _vectorized_wick_check_sell(
    open_arr: np.ndarray, 
    high_arr: np.ndarray, 
    low_arr: np.ndarray, 
    close_arr: np.ndarray,
    min_wick_ratio: float
) -> np.ndarray:
   
    n = len(close_arr)
    result = np.zeros(n, dtype=np.bool_)
    
    for i in range(n):
        o, h, l, c = open_arr[i], high_arr[i], low_arr[i], close_arr[i]
        
        if c >= o:
            result[i] = False
            continue
        
        candle_range = h - l
        if candle_range < 1e-8:
            result[i] = False
            continue
        
        lower_wick = c - l
        wick_ratio = lower_wick / candle_range
        
        result[i] = wick_ratio < min_wick_ratio
    
    return result

# ============================================================================
# AOT/JIT SMART WRAPPERS with Adaptive Thresholds
# ============================================================================

def ema_loop(data: np.ndarray, alpha: float) -> np.ndarray:
    """
    Wrapper for EMA calculation.
    EMA is inherently serial, so no parallel version exists.
    """
    if USE_AOT:
        return numba_compiled.ema_loop(data, alpha)
    else:
        return _ema_loop(data, alpha)


def sma_loop(data: np.ndarray, period: int) -> np.ndarray:
    """
    Wrapper for SMA calculation with smart threshold.
    Threshold: 600 (moderate cost operation)
    """
    if USE_AOT:
        return numba_compiled.sma_loop(data, period)
    else:
        # ⚡ OPTIMIZED: SMA is moderate cost - threshold 600
        if cfg.NUMBA_PARALLEL and len(data) >= 600:
            return _sma_loop_parallel(data, period)
        else:
            return _sma_loop(data, period)


def sanitize_array_wrapper(arr: np.ndarray, default: float) -> np.ndarray:
    """
    Wrapper for array sanitization with smart threshold.
    Threshold: 300 (cheap operation - just checks)
    """
    if USE_AOT:
        return numba_compiled.sanitize_array(arr, default)
    else:
        # ⚡ OPTIMIZED: Sanitization is cheap - threshold 300
        if cfg.NUMBA_PARALLEL and len(arr) >= 300:
            return _sanitize_array_numba_parallel(arr, default)
        else:
            return _sanitize_array_numba(arr, default)


def rolling_std_wrapper(close: np.ndarray, period: int, responsiveness: float) -> np.ndarray:
    """
    Wrapper for rolling standard deviation with smart threshold.
    Threshold: 400 (expensive operation - sqrt, variance)
    """
    if USE_AOT:
        # AOT doesn't have this function yet, fall back to JIT
        if cfg.NUMBA_PARALLEL and len(close) >= 400:
            return _rolling_std_welford_parallel(close, period, responsiveness)
        else:
            return _rolling_std_welford(close, period, responsiveness)
    else:
        # ⚡ OPTIMIZED: Rolling STD is expensive - threshold 400
        if cfg.NUMBA_PARALLEL and len(close) >= 400:
            return _rolling_std_welford_parallel(close, period, responsiveness)
        else:
            return _rolling_std_welford(close, period, responsiveness)


def rolling_mean_wrapper(close: np.ndarray, period: int) -> np.ndarray:
    """
    Wrapper for rolling mean with smart threshold.
    Threshold: 600 (moderate cost operation)
    """
    if USE_AOT:
        # Use AOT SMA if available
        return numba_compiled.sma_loop(close, period)
    else:
        # ⚡ OPTIMIZED: Rolling mean is moderate - threshold 600
        if cfg.NUMBA_PARALLEL and len(close) >= 600:
            return _rolling_mean_numba_parallel(close, period)
        else:
            return _rolling_mean_numba(close, period)


def rolling_minmax_wrapper(arr: np.ndarray, period: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Wrapper for rolling min/max with smart threshold.
    Threshold: 500 (moderate cost operation)
    """
    if USE_AOT:
        # AOT doesn't have this function yet, fall back to JIT
        if cfg.NUMBA_PARALLEL and len(arr) >= 500:
            return _rolling_min_max_numba_parallel(arr, period)
        else:
            return _rolling_min_max_numba(arr, period)
    else:
        # ⚡ OPTIMIZED: Rolling min/max is moderate - threshold 500
        if cfg.NUMBA_PARALLEL and len(arr) >= 500:
            return _rolling_min_max_numba_parallel(arr, period)
        else:
            return _rolling_min_max_numba(arr, period)


def vectorized_wick_check_buy_wrapper(
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    min_wick_ratio: float
) -> np.ndarray:
    """Wrapper for buy candle quality check - uses AOT if available"""
    if USE_AOT:
        return numba_compiled.vectorized_wick_buy(
            open_arr, high_arr, low_arr, close_arr, min_wick_ratio
        )
    else:
        return _vectorized_wick_check_buy(
            open_arr, high_arr, low_arr, close_arr, min_wick_ratio
        )


def vectorized_wick_check_sell_wrapper(
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    min_wick_ratio: float
) -> np.ndarray:
    """Wrapper for sell candle quality check - uses AOT if available"""
    if USE_AOT:
        return numba_compiled.vectorized_wick_sell(
            open_arr, high_arr, low_arr, close_arr, min_wick_ratio
        )
    else:
        return _vectorized_wick_check_sell(
            open_arr, high_arr, low_arr, close_arr, min_wick_ratio
        )
# ============================================================================
# OPTIMIZATION 6: Faster Numba Warmup with Targeted Functions
# ============================================================================

def warmup_numba() -> None:
    """OPTIMIZED: Proper warmup with realistic array sizes for parallel compilation.
    Corrected to include wick checks and rolling stats to prevent first-run lag spikes.
    """
    logger.info("Warming up Numba JIT compiler (optimized parallel)...")        
    try:
        # CRITICAL FIX: Use realistic array sizes (300-400 elements)
        # This ensures parallel functions actually compile with prange
        length_small = 100   # For serial functions
        length_large = 400   # For parallel functions (matches real 15m/5m data)
        
        # Data preparation
        close_small = np.random.random(length_small).astype(np.float64) * 1000
        
        close_large = np.random.random(length_large).astype(np.float64) * 1000
        open_large = close_large + np.random.random(length_large).astype(np.float64) * 2
        high_large = np.maximum(open_large, close_large) + 1.0
        low_large = np.minimum(open_large, close_large) - 1.0
        volume_large = np.random.random(length_large).astype(np.float64) * 1000
        timestamps = np.arange(length_large, dtype=np.int64) * 900

        # 1. CORE SERIAL FUNCTIONS
        critical_funcs = [
            lambda: _ema_loop(close_small, 0.1),
            lambda: _calculate_ppo_core(close_small, 7, 16, 5),
            lambda: _calculate_rsi_core(close_small, 21),
        ]
        
        # 2. VWAP & VOLUME LOGIC
        critical_funcs.append(
            lambda: _vwap_daily_loop(high_large, low_large, close_large, volume_large, timestamps)
        )
        
        # 3. CRITICAL PARALLEL FUNCTIONS (Forces prange compilation)
        if cfg.NUMBA_PARALLEL:
            critical_funcs.extend([
                # Array handling
                lambda: _sanitize_array_numba_parallel(close_large, 0.0),
                
                # Rolling Statistics (Essential for indicators)
                lambda: _rolling_std_welford_parallel(close_large, 50, 0.9),
                lambda: _rolling_mean_numba_parallel(close_large, 144),
                lambda: _rolling_min_max_numba_parallel(close_large, 144),
                lambda: _sma_loop_parallel(close_large, 20),
                
                # FIX: Vectorized Candle Quality Checks
                # These must be warmed up or the first 15m evaluation will be delayed
                lambda: _vectorized_wick_check_buy(open_large, high_large, low_large, close_large, 0.1, 0.4),
                lambda: _vectorized_wick_check_sell(open_large, high_large, low_large, close_large, 0.1, 0.4)
            ])
        
        # 4. EXECUTE COMPILATION IN PARALLEL
        import concurrent.futures
        # Using a ThreadPool to trigger Numba compilation across multiple functions simultaneously
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(func) for func in critical_funcs]
            # 10 second timeout is usually sufficient for full JIT compilation
            completed, _ = concurrent.futures.wait(futures, timeout=10.0)
            
        warmup_mode = "parallel" if cfg.NUMBA_PARALLEL else "serial"
        logger.info(f"Numba warm-up complete ({warmup_mode} mode, {len(critical_funcs)} functions compiled)")
            
    except Exception as e:
        logger.warning(f"Numba warm-up failed (non-fatal): {e}")

async def calculate_indicator_threaded(func: Callable, *args, **kwargs):
    return await asyncio.to_thread(func, *args, **kwargs)


def calculate_pivot_levels_numpy(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    timestamps: np.ndarray
) -> Dict[str, float]:
    piv: Dict[str, float] = {}
    
    try:
        if len(timestamps) < 2:
            logger.warning("Pivot calc: insufficient data")
            return piv
        
        days = timestamps // 86400
        now_utc = datetime.now(timezone.utc)
        yesterday = (now_utc - timedelta(days=1)).date()
        yesterday_ts = datetime(
            yesterday.year, 
            yesterday.month, 
            yesterday.day, 
            tzinfo=timezone.utc
        ).timestamp()
        yesterday_day_number = int(yesterday_ts) // 86400        
        yesterday_mask = days == yesterday_day_number
        
        if not np.any(yesterday_mask):
            logger.warning(
                f"No data for yesterday (day #{yesterday_day_number}). "
                f"Available days: {np.unique(days)}"
            )
            return piv
        yesterday_high = high[yesterday_mask]
        yesterday_low = low[yesterday_mask]
        yesterday_close = close[yesterday_mask]
        
        num_candles = len(yesterday_high)
        if num_candles == 0:
            logger.warning("No candles found for yesterday")
            return piv
        
        H_prev = float(np.max(yesterday_high))
        L_prev = float(np.min(yesterday_low))
        C_prev = float(yesterday_close[-1])
        
        rng_prev = H_prev - L_prev
        
        if rng_prev < 1e-8:
            logger.warning(f"Invalid pivot range: {rng_prev}")
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
                
    except Exception as e:
        logger.error(f"Pivot calculation failed: {e}", exc_info=True)
    
    return piv

def calculate_all_indicators_numpy(
    data_15m: Dict[str, np.ndarray],
    data_5m: Dict[str, np.ndarray],
    data_daily: Optional[Dict[str, np.ndarray]]
) -> Dict[str, np.ndarray]:
    """OPTIMIZED: Disable GC during all indicator calculations"""
    
    # CRITICAL FIX: Disable GC for entire indicator calculation
    # This prevents GC checks during intermediate array allocations
    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        gc.disable()
    
    try:
        results = {}    
        close_15m = data_15m["close"]
        close_5m = data_5m["close"]
        
        ppo, ppo_signal = calculate_ppo_numpy(
            close_15m, cfg.PPO_FAST, cfg.PPO_SLOW, cfg.PPO_SIGNAL
        )
        results['ppo'] = ppo
        results['ppo_signal'] = ppo_signal
        
        results['smooth_rsi'] = calculate_smooth_rsi_numpy(
            close_15m, cfg.SRSI_RSI_LEN, cfg.SRSI_KALMAN_LEN
        )
        
        if cfg.ENABLE_VWAP:
            results['vwap'] = calculate_vwap_numpy(
                data_15m["high"], data_15m["low"], data_15m["close"],
                data_15m["volume"], data_15m["timestamp"]
            )
        else:
            results['vwap'] = np.zeros_like(close_15m)
        
        mmh = calculate_magical_momentum_hist(close_15m)
        results['mmh'] = np.nan_to_num(mmh, nan=0.0, posinf=0.0, neginf=0.0)
        
        if cfg.CIRRUS_CLOUD_ENABLED:
            upw, dnw, filtx1, filtx12 = calculate_cirrus_cloud_numba(close_15m)
            results['upw'] = upw
            results['dnw'] = dnw
        else:
            results['upw'] = np.zeros(len(close_15m), dtype=bool)
            results['dnw'] = np.zeros(len(close_15m), dtype=bool)
        
        results['rma50_15'] = calculate_rma_numpy(close_15m, cfg.RMA_50_PERIOD)
        results['rma200_5'] = calculate_rma_numpy(close_5m, cfg.RMA_200_PERIOD)
        
        if cfg.ENABLE_PIVOT and data_daily is not None:
            # Quick check: is price within 5% of yesterday's high/low range?
            last_close = float(close_15m[-1])
            daily_high = float(data_daily["high"][-1])
            daily_low = float(data_daily["low"][-1])
            daily_range = daily_high - daily_low
            
            # If price is far from yesterday's range, skip pivot calculation
            if abs(last_close - daily_high) < daily_range * 0.5 or \
               abs(last_close - daily_low) < daily_range * 0.5:
                results['pivots'] = calculate_pivot_levels_numpy(
                    data_daily["high"], data_daily["low"],
                    data_daily["close"], data_daily["timestamp"]
                )
            else:
                results['pivots'] = {}
                if cfg.DEBUG_MODE:
                    logger.debug(f"Skipped pivot calc (price {last_close:.2f} far from range {daily_low:.2f}-{daily_high:.2f})")
        else:
            results['pivots'] = {}
        
        return results
        
    finally:
        if gc_was_enabled:
            gc.enable()

def precompute_candle_quality(
    data_15m: Dict[str, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    # ⚡ Use wrappers that select AOT/JIT automatically
    buy_quality = vectorized_wick_check_buy_wrapper(
        data_15m["open"],
        data_15m["high"],
        data_15m["low"],
        data_15m["close"],
        Constants.MIN_WICK_RATIO
    )
    
    sell_quality = vectorized_wick_check_sell_wrapper(
        data_15m["open"],
        data_15m["high"],
        data_15m["low"],
        data_15m["close"],
        Constants.MIN_WICK_RATIO
    )
    
    return buy_quality, sell_quality

# ============================================================================
# OPTIMIZATION 7: Enhanced HTTP Session with Connection Pooling
# ============================================================================

class SessionManager:
    _session: ClassVar[Optional[aiohttp.ClientSession]] = None
    _ssl_context: ClassVar[Optional[ssl.SSLContext]] = None
    _lock: ClassVar[asyncio.Lock] = asyncio.Lock()
    _creation_time: ClassVar[float] = 0.0
    _request_count: ClassVar[int] = 0
    _session_reuse_limit: ClassVar[int] = 2000  # OPTIMIZED: Increased from 1000

    @classmethod
    def _get_ssl_context(cls) -> ssl.SSLContext:
        if cls._ssl_context is None:
            ctx = ssl.create_default_context()
            ctx.check_hostname = True
            ctx.verify_mode = ssl.CERT_REQUIRED
            ctx.minimum_version = ssl.TLSVersion.TLSv1_2
            ctx.options |= ssl.OP_NO_SSLv2 | ssl.OP_NO_SSLv3 | ssl.OP_NO_TLSv1 | ssl.OP_NO_TLSv1_1
            cls._ssl_context = ctx
            logger.debug("SSL context created with TLSv1.2+ minimum")
        return cls._ssl_context

    @classmethod
    async def get_session(cls) -> aiohttp.ClientSession:
        async with cls._lock:
            should_recreate = False

            if cls._session is None or cls._session.closed:
                should_recreate = True
                reason = "no session" if cls._session is None else "session closed"
            elif cls._request_count >= cls._session_reuse_limit:
                should_recreate = True
                reason = f"request limit reached ({cls._request_count})"
                logger.info(f"Session recreation triggered: {reason}")

            if should_recreate:
                if cls._session and not cls._session.closed:
                    try:
                        await cls._session.close()
                        await asyncio.sleep(0.1)  # OPTIMIZED: Reduced from 0.25s
                    except Exception as e:
                        logger.warning(f"Error closing old session: {e}")

                connector = TCPConnector(
                    limit=cfg.TCP_CONN_LIMIT,
                    limit_per_host=cfg.TCP_CONN_LIMIT_PER_HOST,
                    ssl=cls._get_ssl_context(),
                    force_close=False,
                    enable_cleanup_closed=True,
                    ttl_dns_cache=3600,
                    keepalive_timeout=90,  # OPTIMIZED: Increased from 60s
                    family=0,
                )

                timeout = aiohttp.ClientTimeout(
                    total=cfg.HTTP_TIMEOUT,
                    connect=8,  # OPTIMIZED: Reduced from 10s
                    sock_read=cfg.HTTP_TIMEOUT,
                )

                cls._session = aiohttp.ClientSession(
                    connector=connector,
                    timeout=timeout,
                    headers={
                        'User-Agent': f'{cfg.BOT_NAME}/{__version__}',
                        'Accept': 'application/json',
                        'Accept-Encoding': 'gzip, deflate',
                        'Connection': 'keep-alive',  # OPTIMIZED: Added explicit keep-alive
                    },
                    raise_for_status=False,
                )

                cls._creation_time = time.time()
                cls._request_count = 0

                logger.info(
                    f"HTTP session created | "
                    f"Pool: {cfg.TCP_CONN_LIMIT} total, {cfg.TCP_CONN_LIMIT_PER_HOST} per host | "
                    f"Timeout: {cfg.HTTP_TIMEOUT}s | Keepalive: 90s"
                )
            return cls._session

    @classmethod
    def track_request(cls) -> None:
        cls._request_count += 1

    @classmethod
    async def close_session(cls) -> None:
        async with cls._lock:
            if cls._session and not cls._session.closed:
                try:
                    session_age = time.time() - cls._creation_time
                    logger.debug(
                        f"Closing HTTP session | "
                        f"Age: {session_age:.1f}s | Requests served: {cls._request_count}"
                    )
                    await cls._session.close()
                    await asyncio.sleep(0.1)  # OPTIMIZED: Reduced from 0.25s
                    logger.info("HTTP session closed successfully")
                except Exception as e:
                    logger.warning(f"Error closing session: {e}")
                finally:
                    cls._session = None
                    cls._request_count = 0
                    cls._creation_time = 0.0
            else:
                logger.debug("Session already closed or not created")

    @classmethod
    def get_stats(cls) -> Dict[str, Any]:
        if cls._session is None:
            return {
                "active": False,
                "request_count": 0,
                "age_seconds": 0.0,
            }
        age = time.time() - cls._creation_time if cls._creation_time > 0 else 0.0
        return {
            "active": not cls._session.closed,
            "request_count": cls._request_count,
            "age_seconds": round(age, 1),
            "requests_until_recreation": cls._session_reuse_limit - cls._request_count,
        }

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
            logger.debug(f"Shutdown requested, aborting fetch: {url[:80]}")
            return None
        
        try:
            async with session.get(url, params=params, timeout=timeout) as resp:
                if resp.status == 429:
                    retry_after = resp.headers.get('Retry-After')
                    wait_sec = min(int(retry_after) if retry_after else 2, Constants.CIRCUIT_BREAKER_MAX_WAIT)
                    jitter = random.uniform(0.1, 0.5)
                    total_wait = wait_sec + jitter
                    
                    retry_stats[RetryCategory.RATE_LIMIT] += 1
                    logger.warning(
                        f"Rate limited (429) | URL: {url[:80]} | "
                        f"Retry-After: {retry_after}s | Waiting: {total_wait:.2f}s | "
                        f"Attempt: {attempt}/{retries}"
                    )
                    
                    await asyncio.sleep(total_wait)
                    continue
                
                if resp.status >= 500:
                    retry_stats[RetryCategory.API_ERROR] += 1
                    logger.warning(
                        f"Server error {resp.status} | URL: {url[:80]} | "
                        f"Attempt: {attempt}/{retries}"
                    )
                    
                    if attempt < retries:
                        base_delay = min(
                            Constants.CIRCUIT_BREAKER_MAX_WAIT / 10,
                            backoff * (2 ** (attempt - 1))
                        )
                        jitter = base_delay * random.uniform(0.1, 0.5)
                        total_delay = base_delay + jitter
                        
                        logger.debug(f"Retrying after {total_delay:.2f}s...")
                        await asyncio.sleep(total_delay)
                    continue
                
                if resp.status >= 400:
                    logger.error(
                        f"Client error {resp.status} for {url[:80]} | "
                        f"This usually indicates invalid request - not retrying"
                    )
                    return None
                data = await resp.json(loads=json_loads)
                
                SessionManager.track_request()
                
                if any(retry_stats.values()):
                    logger.info(
                        f"Fetch succeeded after retries | URL: {url[:80]} | "
                        f"Attempts: {attempt} | Stats: {retry_stats}"
                    )
                
                return data
                
        except asyncio.TimeoutError as e:
            last_error = e
            retry_stats[RetryCategory.TIMEOUT] += 1
            
            logger.warning(
                f"Timeout (attempt {attempt}/{retries}) | "
                f"URL: {url[:80]} | Timeout: {timeout}s"
            )
            
            if attempt < retries:
                base_delay = min(
                    Constants.CIRCUIT_BREAKER_MAX_WAIT / 10,
                    backoff * (2 ** (attempt - 1))
                )
                jitter = base_delay * random.uniform(0.1, 0.5)
                total_delay = base_delay + jitter
                
                logger.debug(f"Retrying after {total_delay:.2f}s...")
                await asyncio.sleep(total_delay)
        
        except (ClientConnectorError, ClientError, ClientResponseError) as e:
            last_error = e
            category = categorize_exception(e)
            retry_stats[category] = retry_stats.get(category, 0) + 1
            
            logger.warning(
                f"Network error (attempt {attempt}/{retries}) | "
                f"Category: {category} | URL: {url[:80]} | Error: {str(e)[:100]}"
            )
            
            if attempt < retries:
                base_delay = min(
                    Constants.CIRCUIT_BREAKER_MAX_WAIT / 10,
                    backoff * (2 ** (attempt - 1))
                )
                jitter = base_delay * random.uniform(0.1, 0.5)
                total_delay = base_delay + jitter
                
                logger.debug(f"Retrying after {total_delay:.2f}s...")
                await asyncio.sleep(total_delay)
        
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

class APICircuitBreaker:
    """Prevents cascading failures when API is degraded"""
    
    def __init__(self, failure_threshold: int = 3, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.last_failure_time = 0.0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.success_count = 0
        
    def record_success(self) -> None:
        """Record successful API call"""
        if self.state == "HALF_OPEN":
            self.success_count += 1
            if self.success_count >= 2:
                logger.info("🟢 Circuit breaker: Recovered, transitioning to CLOSED")
                self.state = "CLOSED"
                self.failures = 0
                self.success_count = 0
        elif self.state == "CLOSED":
            # Reset failure count on success
            if self.failures > 0:
                self.failures = max(0, self.failures - 1)
    
    def record_failure(self) -> None:
        """Record failed API call"""
        self.failures += 1
        self.last_failure_time = time.time()
        
        if self.failures >= self.failure_threshold and self.state == "CLOSED":
            logger.warning(
                f"🔴 Circuit breaker: OPENED after {self.failures} failures. "
                f"Blocking requests for {self.recovery_timeout}s"
            )
            self.state = "OPEN"
    
    def can_attempt(self) -> Tuple[bool, Optional[str]]:
        """Check if request should be allowed"""
        if self.state == "CLOSED":
            return True, None
        
        if self.state == "OPEN":
            # Check if recovery timeout has passed
            elapsed = time.time() - self.last_failure_time
            if elapsed >= self.recovery_timeout:
                logger.info("🟡 Circuit breaker: Transitioning to HALF_OPEN (testing recovery)")
                self.state = "HALF_OPEN"
                self.success_count = 0
                return True, None
            
            return False, f"Circuit breaker OPEN (retry in {self.recovery_timeout - elapsed:.0f}s)"
        
class DataFetcher:
    """
    Asynchronous data fetcher with
    - configurable concurrency
    - rate-limiting
    - circuit-breaker
    - detailed statistics
    """

    def __init__(self, api_base: str, max_parallel: Optional[int] = None):
        self.api_base = api_base.rstrip("/")

        # Fall back to global config if caller did not supply a limit
        max_parallel = max_parallel or cfg.MAX_PARALLEL_FETCH

        self.semaphore   = asyncio.Semaphore(max_parallel)
        self.timeout     = cfg.HTTP_TIMEOUT
        self.rate_limiter = RateLimitedFetcher(
            max_per_minute=60,
            concurrency=max_parallel
        )

        # Circuit-breaker hardening
        self.circuit_breaker = APICircuitBreaker(
            failure_threshold=3,
            recovery_timeout=60
        )

        # Simple counters for observability
        self.fetch_stats = {
            "products_success":        0,
            "products_failed":         0,
            "candles_success":         0,
            "candles_failed":          0,
            "circuit_breaker_blocks":  0,
        }

        logger.debug(
            f"DataFetcher initialised | max_parallel={max_parallel} | "
            f"rate_limit=60/min | timeout={self.timeout}s"
        )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------

    async def fetch_candles(
        self,
        symbol: str,
        resolution: str,
        limit: int,
        reference_time: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch OHLCV candles for a single symbol/resolution pair.
        Returns the raw JSON dict from the API or None on failure.
        """

        # --- Circuit-breaker guard ------------------------------------
        can_proceed, reason = self.circuit_breaker.can_attempt()
        if not can_proceed:
            logger.warning(
                f"Circuit breaker blocked request for {symbol}: {reason}"
            )
            self.fetch_stats["circuit_breaker_blocks"] += 1
            return None

        # --- Time handling --------------------------------------------
        if reference_time is None:
            reference_time = get_trigger_timestamp()

        minutes = int(resolution) if resolution != "D" else 1440
        interval_seconds = minutes * 60

        # Expected open of the last *closed* candle
        expected_open_ts = calculate_expected_candle_timestamp(
            reference_time, minutes
        )

        # Ask for a few extra future periods to work around API lag
        buffer_periods = 3
        to_time   = reference_time + (interval_seconds * buffer_periods)
        from_time = expected_open_ts - (limit * interval_seconds)

        params = {
            "resolution": resolution,
            "symbol":     symbol,
            "from":       int(from_time),
            "to":         int(to_time),
        }

        url = f"{self.api_base}/v2/chart/history"

        # --- Actual HTTP call -----------------------------------------
        async with self.semaphore:
            data = await self.rate_limiter.call(
                async_fetch_json,
                url,
                params=params,
                retries=cfg.CANDLE_FETCH_RETRIES,
                backoff=cfg.CANDLE_FETCH_BACKOFF,
                timeout=self.timeout
            )

        # --- Logging / sanity checks ----------------------------------
        if data:
            num_candles = len(data.get("t", []))
            if num_candles:
                last_candle_open_ts = data["t"][-1]
                diff = abs(expected_open_ts - last_candle_open_ts)

                if diff > 300:          # > 5 min drift
                    if last_candle_open_ts < expected_open_ts:
                        logger.warning(
                            f"⚠️  API DELAY | {symbol} {resolution} | "
                            f"Expected: {format_ist_time(expected_open_ts)} | "
                            f"Got: {format_ist_time(last_candle_open_ts)} "
                            f"(Diff: {diff}s)"
                        )
                    else:
                        logger.debug(
                            f"API Ahead | {symbol} {resolution} | Diff: {diff}s"
                        )
                else:
                    logger.debug(
                        f"✅ Scanned {symbol} {resolution} | "
                        f"Latest: {format_ist_time(last_candle_open_ts)}"
                    )

        return data

    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """
        Return a snapshot of internal counters including success rates.
        """
        stats = self.fetch_stats.copy()
        stats["rate_limiter"] = self.rate_limiter.get_stats()

        total_products = stats["products_success"] + stats["products_failed"]
        total_candles  = stats["candles_success"]  + stats["candles_failed"]

        if total_products:
            stats["products_success_rate"] = round(
                stats["products_success"] / total_products * 100, 1
            )
        if total_candles:
            stats["candles_success_rate"] = round(
                stats["candles_success"] / total_candles * 100, 1
            )
        return stats

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    async def fetch_candles_batch(
        self,
        requests: List[Tuple[str, str, int]],
        reference_time: Optional[int] = None
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Fire off many `fetch_candles` calls concurrently.
        Returns dict keyed by  "symbol_resolution".
        """
        if reference_time is None:
            reference_time = get_trigger_timestamp()

        tasks       = []
        request_keys = []

        for symbol, resolution, limit in requests:
            task = self.fetch_candles(symbol, resolution, limit, reference_time)
            tasks.append(task)
            request_keys.append(f"{symbol}_{resolution}")

        results = await asyncio.gather(*tasks, return_exceptions=True)

        output = {}
        for key, result in zip(request_keys, results):
            output[key] = None if isinstance(result, Exception) else result
        return output

    # ------------------------------------------------------------------

    async def fetch_all_candles_truly_parallel(
        self,
        pair_requests: List[Tuple[str, List[Tuple[str, int]]]],
        reference_time: Optional[int] = None,
    ) -> Dict[str, Dict[str, Optional[Dict[str, Any]]]]:
        """
        Even higher-level batch helper:
        [
          ("BTC-USD", [("5", 200), ("15", 200)]),
          ("ETH-USD", [("5", 200), ("60", 100)]),
        ]
        Returns nested dict:  symbol -> resolution -> data|None
        """
        if reference_time is None:
            reference_time = get_trigger_timestamp()

        all_tasks     = []
        task_metadata = []

        for symbol, resolutions in pair_requests:
            for resolution, limit in resolutions:
                task = self.fetch_candles(symbol, resolution, limit, reference_time)
                all_tasks.append(task)
                task_metadata.append((symbol, resolution))

        results = await asyncio.gather(*all_tasks, return_exceptions=True)

        output        = {}
        success_count = 0

        for (symbol, resolution), result in zip(task_metadata, results):
            output.setdefault(symbol, {})[resolution] = (
                None if isinstance(result, Exception) else result
            )
            if result and not isinstance(result, Exception):
                success_count += 1

        logger.info(
            f"✅ Parallel fetch complete | Success: {success_count}/{len(all_tasks)}"
        )
        return output


def parse_candles_to_numpy(
    result: Optional[Dict[str, Any]],
) -> Optional[Dict[str, np.ndarray]]:
    if not result or not isinstance(result, dict):
        return None

    res = result.get("result", {}) or {}
    required = ("t", "o", "h", "l", "c", "v")
    if not all(k in res for k in required):
        return None

    try:
        n = len(res["t"])
        if n == 0:
            return None

        data = {
            "timestamp": np.empty(n, dtype=np.int64),
            "open": np.empty(n, dtype=np.float64),
            "high": np.empty(n, dtype=np.float64),
            "low": np.empty(n, dtype=np.float64),
            "close": np.empty(n, dtype=np.float64),
            "volume": np.empty(n, dtype=np.float64),
        }

        data["timestamp"][:] = res["t"]
        data["open"][:] = res["o"]
        data["high"][:] = res["h"]
        data["low"][:] = res["l"]
        data["close"][:] = res["c"]
        data["volume"][:] = res["v"]

        if data["timestamp"][-1] > 1_000_000_000_000:
            data["timestamp"] //= 1000

        if np.isnan(data["close"][-1]) or data["close"][-1] <= 0:
            return None

        return data

    except Exception as e:
        logger.error(f"Failed to parse candles: {e}")
        return None

def validate_candle_data(
    data: Optional[Dict[str, np.ndarray]],
    required_len: int = 0,
) -> Tuple[bool, Optional[str]]:
    
    try:
        if data is None or not data:
            return False, "Data is None or empty"
        
        # ⚡ NEW: Data Freshness Check
        close = data.get("close")
        timestamp = data.get("timestamp")
        
        if timestamp is None or len(timestamp) == 0:
            return False, "Timestamp array is empty"
        
        # Check if data is stale (API frozen/cached)
        current_time = get_trigger_timestamp()
        last_candle_time = int(timestamp[-1])
        staleness = current_time - last_candle_time
        
        # Allow 15min candle + 2min buffer for API lag
        MAX_STALENESS = (15 * 60) + 120  # 17 minutes
        
        if staleness > MAX_STALENESS:
            return False, (
                f"Data is stale: {staleness}s old (max: {MAX_STALENESS}s). "
                f"Last candle: {format_ist_time(last_candle_time)} | "
                f"Current: {format_ist_time(current_time)}"
            )
        
        if close is None or len(close) == 0:
            return False, "Close array is empty"
        close = data.get("close")
        timestamp = data.get("timestamp")

        if close is None or len(close) == 0:
            return False, "Close array is empty"

        if np.any(np.isnan(close)) or np.any(close <= 0):
            return False, "Invalid close prices (NaN or <= 0)"

        if timestamp is None or len(timestamp) == 0:
            return False, "Timestamp array is empty"
            
        if not np.all(timestamp[1:] >= timestamp[:-1]):
            return False, "Timestamps not monotonic increasing"

        if len(close) < required_len:
            return False, f"Insufficient data: {len(close)} < {required_len}"

        if len(close) >= 2:
            time_diffs = np.diff(timestamp)
            if len(time_diffs) > 0:
                median_diff = np.median(time_diffs)
                max_expected_gap = median_diff * Constants.MAX_CANDLE_GAP_MULTIPLIER
                gaps = time_diffs[time_diffs > max_expected_gap]
                if len(gaps) > 0:
                    logger.warning(
                        f"Detected {len(gaps)} candle gaps "
                        f"(median: {median_diff}s, max gap: {gaps.max()}s)"
                    )

        if len(close) >= 2:
            price_changes = np.abs(np.diff(close) / close[:-1]) * 100
            extreme_changes = price_changes[price_changes > Constants.MAX_PRICE_CHANGE_PERCENT]
            if len(extreme_changes) > 0:
                logger.warning(
                    f"Detected {len(extreme_changes)} extreme price changes "
                    f"(max: {extreme_changes.max():.2f}%)"
                )
                return False, f"Extreme price spike detected: {extreme_changes.max():.2f}%"

        return True, None
    except Exception as e:
        logger.error(f"Data validation failed: {e}")
        return False, f"Validation error: {str(e)}"

def get_last_closed_index_from_array(
    timestamps: np.ndarray,
    interval_minutes: int,
    reference_time: Optional[int] = None,
) -> Optional[int]:
    if timestamps is None or timestamps.size < 2:
        logger.warning("No timestamps or insufficient data")
        return None

    if reference_time is None:
        reference_time = get_trigger_timestamp()

    interval_seconds = interval_minutes * 60
    current_period_start = (reference_time // interval_seconds) * interval_seconds

    valid_mask = timestamps < current_period_start
    valid_indices = np.nonzero(valid_mask)[0]

    if valid_indices.size == 0:
        logger.info(
            f"No fully closed {interval_minutes}m candle available yet | "
            f"Latest ts: {format_ist_time(timestamps[-1])} | "
            f"Current period start: {format_ist_time(current_period_start)}"
        )
        return None

    last_closed_idx = int(valid_indices[-1])

    debug_if(
        cfg.DEBUG_MODE,
        logger,
        lambda: (
            f"Selected fully closed candle | Index: {last_closed_idx}\n"
            f"TS: {format_ist_time(timestamps[last_closed_idx])}"
        ),
    )
    return last_closed_idx


def validate_candle_timestamp(
    candle_ts: int,
    reference_time: int,
    interval_minutes: int,
    tolerance_seconds: int = 120,
) -> bool:
    interval_seconds = interval_minutes * 60
    current_window = reference_time // interval_seconds
    expected_open_ts = (current_window * interval_seconds) - interval_seconds

    diff = abs(candle_ts - expected_open_ts)
    if diff > tolerance_seconds:
        logger.error(
            f"Candle timestamp mismatch! Expected open ~{expected_open_ts}, "
            f"got {candle_ts} (diff: {diff}s)"
        )
        return False

    logger.debug(
        f"Candle timestamp validated | Expected open: {expected_open_ts} | "
        f"Got: {candle_ts} | Diff: {diff}s"
    )
    return True


def build_products_map_from_api_result(
    api_products: Optional[Dict[str, Any]]
) -> Dict[str, dict]:
    products_map: Dict[str, dict] = {}
    if not api_products or not api_products.get("result"):
        return products_map

    valid_pattern = CompiledPatterns.VALID_SYMBOL
    for p in api_products["result"]:
        try:
            symbol = p.get("symbol", "")
            if not valid_pattern.match(symbol):
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
                        break
        except Exception:
            # Silently skip malformed product entries
            pass

    return products_map

class RedisStateStore:
    DEDUP_LUA = """
    local key = KEYS[1]
    local ttl = ARGV[1]
    if redis.call("EXISTS", key) == 1 then
        return 0
    else
        redis.call("SET", key, "1", "EX", ttl)
        return 1
    end
    """

    _global_pool: ClassVar[Optional[redis.Redis]] = None
    _pool_healthy: ClassVar[bool] = False
    _pool_lock: ClassVar[asyncio.Lock] = asyncio.Lock()
    _pool_created_at: ClassVar[float] = 0.0
    _pool_reuse_count: ClassVar[int] = 0

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
        self._dedup_script_sha = None

        if cfg.DEBUG_MODE:
            logger.debug(
                f"RedisStateStore initialized | "
                f"State TTL: {cfg.STATE_EXPIRY_DAYS}d | "
                f"Alert TTL: {cfg.STATE_EXPIRY_DAYS}d | "
                f"Metadata TTL: 7d"
            )

    async def _attempt_connect(self, timeout: float = 5.0) -> bool:
        try:
            self._redis = redis.from_url(
                self.redis_url,
                socket_connect_timeout=timeout,
                socket_timeout=timeout,
                retry_on_timeout=True,
                max_connections=32,
                decode_responses=True,
            )
            ok = await self._ping_with_retry(timeout)
            if ok:
                if cfg.DEBUG_MODE:
                    logger.debug("Connected to RedisStateStore (decode_responses=True, max_connections=32)")
                self.degraded = False
                self.degraded_alerted = False
                self._connection_attempts = 0

                async with RedisStateStore._pool_lock:
                    RedisStateStore._global_pool = self._redis
                    RedisStateStore._pool_healthy = True
                    RedisStateStore._pool_created_at = time.time()
                    RedisStateStore._pool_reuse_count = 0
                    if cfg.DEBUG_MODE:
                        logger.debug("💾 Redis connection saved to global pool")

                try:
                    self._dedup_script_sha = await self._redis.script_load(self.DEDUP_LUA)
                    if cfg.DEBUG_MODE:
                        logger.debug("Loaded Redis Lua script for alert deduplication")
                except Exception as e:
                    logger.warning(f"Failed to load Lua script (will fallback): {e}")
                    self._dedup_script_sha = None

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
        async with RedisStateStore._pool_lock:
            if RedisStateStore._global_pool and RedisStateStore._pool_healthy:
                try:
                    await asyncio.wait_for(RedisStateStore._global_pool.ping(), timeout=1.0)
                    self._redis = RedisStateStore._global_pool
                    RedisStateStore._pool_reuse_count += 1

                    if not self._dedup_script_sha:
                        try:
                            self._dedup_script_sha = await self._redis.script_load(self.DEDUP_LUA)
                        except Exception as e:
                            logger.warning(f"Lua script load failed: {e}")

                    self.degraded = False
                    return
                except Exception as e:
                    if cfg.DEBUG_MODE:
                        logger.debug(f"Pool health check failed: {e}, creating new pool")
                    RedisStateStore._pool_healthy = False

        for attempt in range(1, cfg.REDIS_CONNECTION_RETRIES + 1):
            self._connection_attempts = attempt
            if cfg.DEBUG_MODE:
                logger.debug(f"Redis connection attempt {attempt}/{cfg.REDIS_CONNECTION_RETRIES}")

            if await self._attempt_connect(timeout):
                test_key = f"smoke_test:{uuid.uuid4().hex[:8]}"
                test_val = "ok"
                set_ok = await self._safe_redis_op(
                    lambda: self._redis.set(test_key, test_val, ex=10),
                    2.0, "smoke_set"
                )
                get_ok = await self._safe_redis_op(
                    lambda: self._redis.get(test_key),
                    2.0, "smoke_get", parser=lambda r: r == test_val
                )
                if set_ok and get_ok:
                    await self._safe_redis_op(lambda: self._redis.delete(test_key), 1.0, "smoke_cleanup")
                    expiry_mode = "TTL-based" if self.expiry_seconds > 0 else "manual"
                    logger.info(
                        f"✅ Redis connected ({self._redis.connection_pool.max_connections} connections, {expiry_mode} expiry)"
                    )
                    info = await self._safe_redis_op(lambda: self._redis.info("memory"), 3.0, "info_memory")
                    if info and cfg.DEBUG_MODE:
                        policy = info.get("maxmemory_policy", "unknown")
                        logger.debug(f"Redis memory policy: {policy}")
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
        if self._redis and self._redis != RedisStateStore._global_pool:
            try:
                await self._redis.aclose()
                if cfg.DEBUG_MODE:
                    logger.debug("Closed non-pool Redis connection")
            except Exception:
                pass
        self._redis = None

    @classmethod
    async def shutdown_global_pool(cls) -> None:
        async with cls._pool_lock:
            if cls._global_pool:
                try:
                    if hasattr(cls._global_pool, 'connection_pool') and cls._global_pool.connection_pool:
                        pool_age = time.time() - cls._pool_created_at
                        reuse_count = cls._pool_reuse_count
                        
                        logger.info(
                            f"Shutting down global Redis pool | "
                            f"Age: {pool_age:.1f}s | "
                            f"Reuses: {reuse_count}"
                        )
                        
                        await cls._global_pool.aclose()
                        await asyncio.sleep(0.25)  # Allow cleanup
                        
                        cls._global_pool = None
                        cls._pool_healthy = False
                        cls._pool_created_at = 0.0
                        cls._pool_reuse_count = 0
                        
                        logger.debug("✅ Global Redis pool shutdown complete")
                    else:
                        logger.debug("Redis pool already closed")
                        cls._global_pool = None
                        cls._pool_healthy = False
                        
                except Exception as e:
                    logger.error(f"Error shutting down global Redis pool: {e}")
                    cls._global_pool = None
                    cls._pool_healthy = False
            else:
                logger.debug("Global Redis pool already closed or not created")

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def _ping_with_retry(self, timeout: float) -> bool:
        result = await self._safe_redis_op(lambda: self._redis.ping(), timeout, "ping")
        return bool(result)

    async def _safe_redis_op(
        self,
        fn: Callable[[], Any],
        timeout: float,
        op_name: str,
        parser: Optional[Callable[[Any], Any]] = None,
    ):
        if not self._redis:
            return None
        try:
            coro = fn()
            result = await asyncio.wait_for(coro, timeout=timeout)
            return parser(result) if parser else result
        except (asyncio.TimeoutError, RedisConnectionError, RedisError) as e:
            logger.error(f"Redis {op_name} failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to {op_name}: {e}")
            return None

    async def get(self, key: str, timeout: float = 2.0) -> Optional[Dict[str, Any]]:
        return await self._safe_redis_op(
            lambda: self._redis.get(f"{self.state_prefix}{key}"),
            timeout,
            f"get {key}",
            parser=lambda r: json_loads(r) if r else None,
        )

    async def set(
        self,
        key: str,
        state: Optional[Any],
        ts: Optional[int] = None,
        timeout: float = 2.0,
    ) -> None:
        ts = int(ts or time.time())
        redis_key = f"{self.state_prefix}{key}"
        data = json_dumps({"state": state, "ts": ts})
        await self._safe_redis_op(
            lambda: self._redis.set(
                redis_key,
                data,
                ex=self.expiry_seconds if self.expiry_seconds > 0 else None,
            ),
            timeout,
            f"set {key}",
        )

    async def get_metadata(self, key: str, timeout: float = 2.0) -> Optional[str]:
        return await self._safe_redis_op(
            lambda: self._redis.get(f"{self.meta_prefix}{key}"),
            timeout,
            f"get_metadata {key}",
            parser=lambda r: r if r else None,
        )

    async def set_metadata(
        self, key: str, value: str, timeout: float = 2.0
    ) -> None:
        await self._safe_redis_op(
            lambda: self._redis.set(
                f"{self.meta_prefix}{key}", value, ex=self.metadata_expiry_seconds
            ),
            timeout,
            f"set_metadata {key}",
        )

    async def check_recent_alert(
        self, pair: str, alert_key: str, ts: int
    ) -> bool:
        if self.degraded:
            return True

        window = (ts // Constants.ALERT_DEDUP_WINDOW_SEC) * Constants.ALERT_DEDUP_WINDOW_SEC
        recent_key = f"recent_alert:{pair}:{alert_key}:{window}"

        if self._dedup_script_sha:
            try:
                result = await self._safe_redis_op(
                    lambda: self._redis.evalsha(
                        self._dedup_script_sha, 1, recent_key, str(Constants.ALERT_DEDUP_WINDOW_SEC)
                    ),
                    timeout=2.0,
                    op_name=f"evalsha_dedup_{pair}:{alert_key}",
                )
                should_send = bool(result)
                if cfg.DEBUG_MODE and not should_send:
                    logger.debug(f"Dedup: Skipping duplicate {pair}:{alert_key}")
                return should_send
            except Exception as e:
                if cfg.DEBUG_MODE:
                    logger.debug(f"Lua script failed, fallback to SET NX: {e}")

        result = await self._safe_redis_op(
            lambda: self._redis.set(recent_key, "1", nx=True, ex=Constants.ALERT_DEDUP_WINDOW_SEC),
            timeout=2.0,
            op_name=f"check_recent_alert {pair}:{alert_key}",
            parser=lambda r: bool(r),
        )
        should_send = bool(result)
        if cfg.DEBUG_MODE and not should_send:
            logger.debug(f"Dedup: Skipping duplicate {pair}:{alert_key}")
        return should_send

    async def batch_check_recent_alerts(
        self, checks: List[Tuple[str, str, int]]
    ) -> Dict[str, bool]:
        if self.degraded or not checks:
            return {f"{pair}:{alert_key}": True for pair, alert_key, _ in checks}

        try:
            async with self._redis.pipeline() as pipe:
                keys_map: Dict[str, str] = {}
                for pair, alert_key, ts in checks:
                    window = (ts // Constants.ALERT_DEDUP_WINDOW_SEC) * Constants.ALERT_DEDUP_WINDOW_SEC
                    recent_key = f"recent_alert:{pair}:{alert_key}:{window}"
                    composite = f"{pair}:{alert_key}"
                    keys_map[recent_key] = composite
                    pipe.set(recent_key, "1", nx=True, ex=Constants.ALERT_DEDUP_WINDOW_SEC)

                results = await asyncio.wait_for(pipe.execute(), timeout=3.0)

            output: Dict[str, bool] = {}
            for idx, (recent_key, composite_key) in enumerate(keys_map.items()):
                # True if SET succeeded (new key = send alert)
                should_send = bool(results[idx]) if idx < len(results) else True
                output[composite_key] = should_send

            if cfg.DEBUG_MODE:
                duplicates = sum(1 for v in output.values() if not v)
                if duplicates > 0:
                    logger.debug(f"Batch dedup: {duplicates}/{len(checks)} duplicates filtered")

            return output
        except Exception as e:
            logger.error(f"Batch check_recent_alerts failed: {e}")
            return {f"{pair}:{alert_key}": True for pair, alert_key, _ in checks}

    async def mget_states(
        self,
        keys: List[str],
        timeout: float = 2.0
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        if not self._redis or not keys:
            return {}

        redis_keys = [f"{self.state_prefix}{k}" for k in keys]

        try:
            results = await asyncio.wait_for(self._redis.mget(redis_keys), timeout=timeout)
            if not results:
                return {}
            output: Dict[str, Optional[Dict[str, Any]]] = {}
            for idx, key in enumerate(keys):
                val = results[idx] if idx < len(results) else None
                if val:
                    try:
                        output[key] = json_loads(val)
                    except Exception as e:
                        if cfg.DEBUG_MODE:
                            logger.debug(f"Failed to parse state for {key}: {e}")
                        output[key] = None
                else:
                    output[key] = None
            return output
        except Exception as e:
            logger.error(f"mget_states failed for {len(keys)} keys: {e} | Keys sample: {keys[:5]}")
            return {}

    async def batch_set_states(
        self,
        updates: List[Tuple[str, Any, Optional[int]]],
        timeout: float = 4.0,
    ) -> None:
        if self.degraded or not updates or not self._redis:
            return

        try:
            async with self._redis.pipeline() as pipe:
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

            if cfg.DEBUG_MODE:
                logger.debug(f"Batch updated {len(updates)} states atomically")
        except Exception as e:
            logger.error(f"Batch state update failed (falling back to individual): {e}")
            for key, state, custom_ts in updates:
                await self.set(key, state, custom_ts)

    async def atomic_eval_batch(
        self,
        pair: str,
        alert_keys: List[str],
        state_updates: List[Tuple[str, Any, Optional[int]]],
        dedup_checks: List[Tuple[str, str, int]]
    ) -> Tuple[Dict[str, bool], Dict[str, bool]]:
        if self.degraded:
            empty_prev = {k: False for k in alert_keys}
            empty_dedup = {f"{p}:{ak}": True for p, ak, _ in dedup_checks}
            return empty_prev, empty_dedup

        try:
            async with self._redis.pipeline() as pipe:
                # Phase 1: MGET for previous states
                state_keys = [f"{pair}:{k}" for k in alert_keys]
                pipe.mget(state_keys)

                # Phase 2: SET for state updates
                now = int(time.time())
                for key, state, custom_ts in state_updates:
                    ts = custom_ts if custom_ts is not None else now
                    data = json_dumps({"state": state, "ts": ts})
                    full_key = f"{self.state_prefix}{key}"
                    if self.expiry_seconds > 0:
                        pipe.set(full_key, data, ex=self.expiry_seconds)
                    else:
                        pipe.set(full_key, data)

                # Phase 3: SET NX for dedup
                for pair_name, alert_key, ts in dedup_checks:
                    window = (ts // Constants.ALERT_DEDUP_WINDOW_SEC) * Constants.ALERT_DEDUP_WINDOW_SEC
                    recent_key = f"recent_alert:{pair_name}:{alert_key}:{window}"
                    pipe.set(recent_key, "1", nx=True, ex=Constants.ALERT_DEDUP_WINDOW_SEC)

            # Execute all at once
            results = await asyncio.wait_for(pipe.execute(), timeout=4.0)

            # Parse results
            num_state_keys = len(state_keys)
            num_updates = len(state_updates)

            # Previous states (first result from MGET)
            prev_states = {}
            mget_results = results[0] if results else []
            for idx, key in enumerate(alert_keys):
                val = mget_results[idx] if idx < len(mget_results) else None
                if val:
                    try:
                        parsed = json_loads(val)
                        prev_states[key] = parsed.get("state") == "ACTIVE"
                    except Exception:
                        prev_states[key] = False
                else:
                    prev_states[key] = False

            # Dedup results (results after MGET and SETs)
            dedup_results = {}
            dedup_start_idx = 1 + num_updates  # Skip MGET result + SET results
            for idx, (pair_name, alert_key, _) in enumerate(dedup_checks):
                result_idx = dedup_start_idx + idx
                should_send = bool(results[result_idx]) if result_idx < len(results) else True
                dedup_results[f"{pair_name}:{alert_key}"] = should_send

            return prev_states, dedup_results

        except Exception as e:
            logger.error(f"atomic_eval_batch failed: {e}")
            # Fallback to individual operations
            state_keys = [f"{pair}:{k}" for k in alert_keys]
            prev_states_raw = await self.mget_states(state_keys)
            prev_states = {k: (v.get("state") == "ACTIVE" if v else False) for k, v in prev_states_raw.items()}
            await self.batch_set_states(state_updates)
            dedup_results = await self.batch_check_recent_alerts(dedup_checks)
            return prev_states, dedup_results

    async def atomic_batch_update(
        self,
        updates: List[Tuple[str, Any, Optional[int]]],
        deletes: Optional[List[str]] = None,
        timeout: float = 4.0,
    ) -> bool:
        if self.degraded or not self._redis:
            return False

        try:
            async with self._redis.pipeline() as pipe:
                now = int(time.time())

                for key, state, custom_ts in updates:
                    ts = custom_ts if custom_ts is not None else now
                    data = json_dumps({"state": state, "ts": ts})
                    full_key = f"{self.state_prefix}{key}"
                    if self.expiry_seconds > 0:
                        pipe.set(full_key, data, ex=self.expiry_seconds)
                    else:
                        pipe.set(full_key, data)

                if deletes:
                    for key in deletes:
                        pipe.delete(f"{self.state_prefix}{key}")

                await asyncio.wait_for(pipe.execute(), timeout=timeout)
            return True
        except Exception as e:
            logger.error(f"Atomic batch update failed: {e}")
            return False

class RedisLock:
    RELEASE_LUA = """
    if redis.call("GET", KEYS[1]) == ARGV[1] then
        return redis.call("DEL", KEYS[1])
    else
        return 0
    end
    """

    def __init__(
        self,
        redis_client: Optional[redis.Redis],
        lock_key: str,
        expire: int | None = None,
    ):
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
            ok = await asyncio.wait_for(
                self.redis.set(self.lock_key, token, nx=True, ex=self.expire),
                timeout=timeout,
            )
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

            await asyncio.wait_for(
                self.redis.expire(self.lock_key, self.expire), timeout=timeout
            )
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
            await asyncio.wait_for(
                self.redis.eval(self.RELEASE_LUA, 1, self.lock_key, self.token),
                timeout=timeout,
            )
            logger.info(f"🔓 Lock released")
        except Exception as e:
            logger.error(f"Error releasing Redis lock: {e}")
        finally:
            self.token = None
            self.acquired_by_me = False

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
        
        max_len = Constants.TELEGRAM_MAX_MESSAGE_LENGTH
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
    {"key": "vwap_up","title": "🔵▲ Price cross above VWAP","check_fn": lambda ctx, ppo, ppo_sig, rsi: (ctx["buy_common"] and ctx["close_prev"] <= ctx["vwap_prev"] and ctx["close_curr"] > ctx["vwap_curr"] + 0.0002), "extra_fn": lambda ctx, ppo, ppo_sig, rsi, _: f"MMH ({ctx['mmh_curr']:.2f})", "requires": []},
    {"key": "vwap_down", "title": "🟣▼ Price cross below VWAP", "check_fn": lambda ctx, ppo, ppo_sig, rsi: (ctx["sell_common"] and ctx["close_prev"] >= ctx["vwap_prev"] and ctx["close_curr"] < ctx["vwap_curr"] - 0.0002), "extra_fn": lambda ctx, ppo, ppo_sig, rsi, _: f"MMH ({ctx['mmh_curr']:.2f})", "requires": []},
    {"key": "mmh_buy", "title": "🔵⬆️ MMH Reversal BUY", "check_fn": lambda ctx, ppo, ppo_sig, rsi: ctx["mmh_reversal_buy"], "extra_fn": lambda ctx, ppo, ppo_sig, rsi, _: f"MMH ({ctx['mmh_curr']:.2f})", "requires": []},
    {"key": "mmh_sell", "title": "🟣⬇ MMH Reversal SELL", "check_fn": lambda ctx, ppo, ppo_sig, rsi: ctx["mmh_reversal_sell"], "extra_fn": lambda ctx, ppo, ppo_sig, rsi, _: f"MMH ({ctx['mmh_curr']:.2f})", "requires": []},
]

def _validate_pivot_cross(
    ctx: Dict[str, Any], 
    level: str, 
    is_buy: bool
) -> Tuple[bool, Optional[str]]:
    """
    Returns (is_valid_cross, suppression_reason_or_None)
    """
    if not ctx.get("pivots") or level not in ctx["pivots"]:
        return False, "Pivot level missing"

    level_value = ctx["pivots"][level]
    close_curr = ctx["close_curr"]
    
    price_diff_pct = abs(level_value - close_curr) / close_curr * 100
    
    if price_diff_pct > cfg.PIVOT_MAX_DISTANCE_PCT:
        reason = (
            f"Pivot {level} too far: ${level_value:.2f} "
            f"(diff {price_diff_pct:.1f}% > {cfg.PIVOT_MAX_DISTANCE_PCT}%)"
        )
        return False, reason
    
    if is_buy:
        cross = (ctx["close_prev"] <= level_value) and (close_curr > level_value)
    else:
        cross = (ctx["close_prev"] >= level_value) and (close_curr < level_value)
    
    return cross, None

BUY_PIVOT_DEFS = [
    {
        "key": f"pivot_up_{level}", 
        "title": f"🟢⬆️ Cross above {level}",
        "check_fn": lambda ctx, ppo, ppo_sig, rsi, level=level: (
            ctx["buy_common"] and _validate_pivot_cross(ctx, level, is_buy=True)[0]
        ), 
        "extra_fn": lambda ctx, ppo, ppo_sig, rsi, _, level=level: (
            f"${ctx['pivots'][level]:,.2f} | MMH ({ctx['mmh_curr']:.2f})"
            + (f" [Suppressed: {_validate_pivot_cross(ctx, level, True)[1]}]" 
               if not _validate_pivot_cross(ctx, level, True)[0] and ctx.get("pivots") else "")
        ),
        "requires": ["pivots"]
    }
    for level in ["P", "S1", "S2", "S3", "R1", "R2"]
]

SELL_PIVOT_DEFS = [
    {
        "key": f"pivot_down_{level}",
        "title": f"🔴⬇️ Cross below {level}",
        "check_fn": lambda ctx, ppo, ppo_sig, rsi, level=level: (
            ctx["sell_common"] and _validate_pivot_cross(ctx, level, is_buy=False)[0]
        ),
        "extra_fn": lambda ctx, ppo, ppo_sig, rsi, _, level=level: (
            f"${ctx['pivots'][level]:,.2f} | MMH ({ctx['mmh_curr']:.2f})"
            + (f" [Suppressed: {_validate_pivot_cross(ctx, level, False)[1]}]" 
               if not _validate_pivot_cross(ctx, level, False)[0] and ctx.get("pivots") else "")
        ),
        "requires": ["pivots"]
    }
    for level in ["P", "S1", "S2", "R1", "R2", "R3"]
]

ALERT_DEFINITIONS.extend(BUY_PIVOT_DEFS)
ALERT_DEFINITIONS.extend(SELL_PIVOT_DEFS)

ALERT_DEFINITIONS_MAP = {d["key"]: d for d in ALERT_DEFINITIONS}

ALERT_KEYS: Dict[str, str] = {
    d["key"]: f"ALERT:{d['key'].upper()}" 
    for d in ALERT_DEFINITIONS
}

for level in PIVOT_LEVELS:
    ALERT_KEYS[f"pivot_up_{level}"] = f"ALERT:PIVOT_UP_{level}"
    ALERT_KEYS[f"pivot_down_{level}"] = f"ALERT:PIVOT_DOWN_{level}"

logger.debug(f"Alert keys initialized: {len(ALERT_KEYS)} mappings")

def validate_alert_definitions() -> None:
    errors = []
    
    keys_seen = set()
    for def_ in ALERT_DEFINITIONS:
        key = def_["key"]
        if key in keys_seen:
            errors.append(f"Duplicate alert key: {key}")
        keys_seen.add(key)
    
    required_fields = ["key", "title", "check_fn", "extra_fn", "requires"]
    for idx, def_ in enumerate(ALERT_DEFINITIONS):
        for field in required_fields:
            if field not in def_:
                errors.append(f"Alert definition {idx} missing field: {field}")
        
        if not callable(def_.get("check_fn")):
            errors.append(f"Alert {def_.get('key', idx)}: check_fn is not callable")
        if not callable(def_.get("extra_fn")):
            errors.append(f"Alert {def_.get('key', idx)}: extra_fn is not callable")
        
        if not isinstance(def_.get("requires", []), list):
            errors.append(f"Alert {def_.get('key', idx)}: requires must be a list")
    
    for def_ in ALERT_DEFINITIONS:
        if def_["key"] not in ALERT_KEYS:
            errors.append(f"Alert key {def_['key']} missing from ALERT_KEYS mapping")
    
    if errors:
        error_msg = "❌ ALERT DEFINITION VALIDATION FAILED:\n" + "\n".join(f"  - {e}" for e in errors)
        logger.critical(error_msg)
        raise ValueError(error_msg)
    
    logger.debug(f"✅ Validated {len(ALERT_DEFINITIONS)} alert definitions ({len(ALERT_KEYS)} keys)")

validate_alert_definitions()

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
    
    try:
        results = await sdb.mget_states(state_keys)
        
        output = {}
        for key in keys:
            state_key = f"{pair}:{key}"
            st = results.get(state_key)
            output[key] = st is not None and st.get("state") == "ACTIVE"
        
        return output
        
    except Exception as e:
        logger.error(f"check_multiple_alert_states failed for {pair}: {e}")
        return {k: False for k in keys}

def check_common_conditions(
    open_val: float,
    high_val: float,
    low_val: float,
    close_val: float,
    is_buy: bool
) -> bool:
    
    try:
        candle_range = high_val - low_val
        if candle_range < 1e-8:
            return False

        if is_buy:
            if close_val <= open_val:
                return False
            upper_wick = high_val - close_val
            wick_ratio = upper_wick / candle_range
            return wick_ratio < Constants.MIN_WICK_RATIO
        else:
            if close_val >= open_val:
                return False
            lower_wick = close_val - low_val
            wick_ratio = lower_wick / candle_range
            return wick_ratio < Constants.MIN_WICK_RATIO

    except Exception as e:
        logger.error(f"check_common_conditions failed: {e}")
        return False

def check_candle_quality_with_reason(
    open_val: float,
    high_val: float,
    low_val: float,
    close_val: float,
    is_buy: bool
) -> Tuple[bool, str]:
    
    try:
        candle_range = high_val - low_val
        if candle_range < 1e-8:
            return False, "Candle range too small"

        if is_buy:
            if close_val <= open_val:
                return False, f"Not green candle (C={close_val:.2f} <= O={open_val:.2f})"

            upper_wick = high_val - close_val
            wick_ratio = upper_wick / candle_range

            if wick_ratio >= Constants.MIN_WICK_RATIO:
                return False, f"Upper wick {wick_ratio*100:.0f}% > {Constants.MIN_WICK_RATIO*100:.0f}%"

            return True, "Passed"

        else:
            if close_val >= open_val:
                return False, f"Not red candle (C={close_val:.2f} >= O={open_val:.2f})"

            lower_wick = close_val - low_val
            wick_ratio = lower_wick / candle_range

            if wick_ratio >= Constants.MIN_WICK_RATIO:
                return False, f"Lower wick {wick_ratio*100:.0f}% > {Constants.MIN_WICK_RATIO*100:.0f}%"

            return True, "Passed"

    except Exception as e:
        return False, f"Error: {str(e)}"

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
        # ===================================================================
        # STEP 1: IDENTIFY CLOSED CANDLE INDICES
        # ===================================================================
        i15 = get_last_closed_index_from_array(data_15m["timestamp"], 15, reference_time)
        i5 = get_last_closed_index_from_array(data_5m["timestamp"], 5, reference_time)

        if i15 is None or i15 < 3 or i5 is None:
            logger_pair.warning(f"Insufficient data for {pair_name}: i15={i15}, i5={i5}")
            return None

        # ===================================================================
        # STEP 2: ⚡ QUICK PRE-CHECK (Skip expensive calculations if possible)
        # ===================================================================
        close_15m = data_15m["close"]
        close_5m = data_5m["close"]
        timestamps_15m = data_15m["timestamp"]

        # Quick trend check without full indicator calculation
        i15_quick = get_last_closed_index_from_array(timestamps_15m, 15, reference_time)
        if i15_quick is None or i15_quick < 3:
            logger_pair.debug(f"Insufficient data for {pair_name}")
            return None

        close_curr_quick = close_15m[i15_quick]
        open_curr_quick = data_15m["open"][i15_quick]

        # Quick candle color check
        is_green = close_curr_quick > open_curr_quick
        is_red = close_curr_quick < open_curr_quick

        # If neither green nor red candle, skip expensive indicators
        if not is_green and not is_red:
            logger_pair.debug(f"Doji/neutral candle for {pair_name}, skipping indicators")
            return None

        # ===================================================================
        # STEP 3: NOW calculate indicators (only if candle has direction)
        # ===================================================================
        indicators = await asyncio.to_thread(
            calculate_all_indicators_numpy, data_15m, data_5m, data_daily
        )

        # Extract indicator arrays from dictionary
        ppo = indicators['ppo']
        ppo_signal = indicators['ppo_signal']
        smooth_rsi = indicators['smooth_rsi']
        vwap = indicators['vwap']
        mmh = indicators['mmh']
        upw = indicators['upw']
        dnw = indicators['dnw']
        rma50_15 = indicators['rma50_15']
        rma200_5 = indicators['rma200_5']
        piv = indicators['pivots']

        # Extract raw data arrays from input dictionaries
        close_15m = data_15m["close"]
        open_15m = data_15m["open"]
        high_15m = data_15m["high"]
        low_15m = data_15m["low"]
        timestamps_15m = data_15m["timestamp"]
        close_5m = data_5m["close"]

        # ===================================================================
        # STEP 3: ✅ NOW extract values at target indices (after arrays exist)
        # ===================================================================
        close_curr = close_15m[i15]       # already np.float64
        close_prev = close_15m[i15 - 1]   # already np.float64
        close_5m_val = close_5m[i5]       # already np.float64
        ts_curr = int(timestamps_15m[i15])  # keep int() for timestamps

        # Extract indicator values at target indices
        rma50_15_val = rma50_15[i15]      # already np.float64
        rma200_5_val = rma200_5[i5]       # already np.float64

        # ===================================================================
        # STEP 4: TIMESTAMP VALIDATION (STRICT ONLY FOR 15m)
        # ===================================================================
        latest_ts = int(timestamps_15m[i15])
        expected_open_ts = calculate_expected_candle_timestamp(reference_time, 15)

        if not validate_candle_timestamp(
            candle_ts=latest_ts,
            reference_time=reference_time,
            interval_minutes=15,
            tolerance_seconds=300
        ):
            logger_pair.info(
                f"Skipping {pair_name} - latest 15m candle is not confirmed closed "
                f"(got open {format_ist_time(latest_ts)}, expected ~{format_ist_time(expected_open_ts)})"
            )
            return None

        # ===================================================================
        # STEP 5: DAILY PIVOT RESET (after validation passes)
        # ===================================================================
        if piv and cfg.ENABLE_PIVOT:
            current_day = int(ts_curr // 86400)
            pivot_day_key = f"{pair_name}:pivot_day"
            last_pivot_day_str = await sdb.get_metadata(pivot_day_key)
            last_pivot_day = int(last_pivot_day_str) if last_pivot_day_str else None

            if last_pivot_day != current_day:
                delete_keys = []
                for level in ["P", "S1", "S2", "S3", "R1", "R2", "R3"]:
                    up_key = f"{pair_name}:{ALERT_KEYS[f'pivot_up_{level}']}"
                    down_key = f"{pair_name}:{ALERT_KEYS[f'pivot_down_{level}']}"
                    delete_keys.extend([up_key, down_key])

                if delete_keys:
                    success = await sdb.atomic_batch_update([], deletes=delete_keys)
                    if not success and cfg.DEBUG_MODE:
                        logger_pair.warning(
                            f"Atomic delete failed for pivot reset, falling back"
                        )
                        await asyncio.gather(
                            *[sdb._redis.delete(k) for k in delete_keys],
                            return_exceptions=True
                        )
                await sdb.set_metadata(pivot_day_key, str(current_day))

        # ===================================================================
        # STEP 6: EXTRACT REMAINING CANDLE & INDICATOR DATA
        # ===================================================================
        open_curr = open_15m[i15]
        high_curr = high_15m[i15]
        low_curr = low_15m[i15]

        # Extract indicator values for current and previous candles
        ppo_curr, ppo_prev = ppo[i15], ppo[i15 - 1]
        ppo_sig_curr, ppo_sig_prev = ppo_signal[i15], ppo_signal[i15 - 1]
        rsi_curr, rsi_prev = smooth_rsi[i15], smooth_rsi[i15 - 1]
        vwap_curr = vwap[i15] if len(vwap) > i15 else 0.0
        vwap_prev = vwap[i15 - 1] if len(vwap) > (i15 - 1) else 0.0
        mmh_curr, mmh_m1 = mmh[i15], mmh[i15 - 1]

        # Cloud indicators
        cloud_up = bool(upw[i15]) and not bool(dnw[i15])
        cloud_down = bool(dnw[i15]) and not bool(upw[i15])

        # Candle color
        is_green_candle = close_curr > open_curr
        is_red_candle = close_curr < open_curr

        debug_if(True, logger_pair,
                 lambda: f"Candle analysis | O={open_curr:.2f} C={close_curr:.2f}")

        # ===================================================================
        # STEP 7: TREND & QUALITY FILTERS
        # ===================================================================
        base_buy_trend = (rma50_15_val < close_curr and rma200_5_val < close_5m_val)
        base_sell_trend = (rma50_15_val > close_curr and rma200_5_val > close_5m_val)

        # Add MMH and cloud confirmation
        if base_buy_trend:
            base_buy_trend = base_buy_trend and (mmh_curr > 0 and cloud_up)
        if base_sell_trend:
            base_sell_trend = base_sell_trend and (mmh_curr < 0 and cloud_down)

        # Candle quality checks
        buy_quality_arr, sell_quality_arr = precompute_candle_quality(data_15m)
        buy_candle_passed = bool(buy_quality_arr[i15])
        sell_candle_passed = bool(sell_quality_arr[i15])
        buy_candle_reason, sell_candle_reason = None, None

        if base_buy_trend:
            if not is_green_candle:
                buy_candle_passed = False
                buy_candle_reason = (
                    f"NOT GREEN CANDLE (O={open_curr:.4f} C={close_curr:.4f})"
                )
                logger_pair.debug(
                    f"❌ BUY alert blocked for {pair_name} | {buy_candle_reason}"
                )
            elif not buy_candle_passed:
                _, buy_candle_reason = check_candle_quality_with_reason(
                    open_curr, high_curr, low_curr, close_curr, is_buy=True
                )

        if base_sell_trend:
            if not is_red_candle:
                sell_candle_passed = False
                sell_candle_reason = (
                    f"NOT RED CANDLE (O={open_curr:.4f} C={close_curr:.4f})"
                )
                logger_pair.debug(
                    f"❌ SELL alert blocked for {pair_name} | {sell_candle_reason}"
                )
            elif not sell_candle_passed:
                _, sell_candle_reason = check_candle_quality_with_reason(
                    open_curr, high_curr, low_curr, close_curr, is_buy=False
                )

        buy_common = base_buy_trend and buy_candle_passed and is_green_candle
        sell_common = base_sell_trend and sell_candle_passed and is_red_candle

        # Log rejection reasons
        if base_buy_trend and not buy_common:
            logger_pair.debug(
                f"🚫 BUY rejected for {pair_name} | "
                f"Reason: {buy_candle_reason or 'Unknown'} | "
                f"Green={is_green_candle} Quality={buy_candle_passed}"
            )
        if base_sell_trend and not sell_common:
            logger_pair.debug(
                f"🚫 SELL rejected for {pair_name} | "
                f"Reason: {sell_candle_reason or 'Unknown'} | "
                f"Red={is_red_candle} Quality={sell_candle_passed}"
            )

        # ===================================================================
        # STEP 8: MMH REVERSAL LOGIC
        # ===================================================================
        mmh_reversal_buy = False
        mmh_reversal_sell = False
        if i15 >= 3:
            mmh_m3, mmh_m2 = float(mmh[i15 - 3]), float(mmh[i15 - 2])
            mmh_reversal_buy = (
                buy_common
                and mmh_curr > 0
                and mmh_m3 > mmh_m2 > mmh_m1
                and mmh_curr > mmh_m1
            )
            mmh_reversal_sell = (
                sell_common
                and mmh_curr < 0
                and mmh_m3 < mmh_m2 < mmh_m1
                and mmh_curr < mmh_m1
            )

        # ===================================================================
        # STEP 9: BUILD CONTEXT FOR ALERT EVALUATION
        # ===================================================================
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
            "vwap": cfg.ENABLE_VWAP,
            "candle_quality_failed_buy": base_buy_trend and not buy_candle_passed,
            "candle_quality_failed_sell": base_sell_trend and not sell_candle_passed,
            "candle_rejection_reason_buy": buy_candle_reason,
            "candle_rejection_reason_sell": sell_candle_reason,
            "is_green_candle": is_green_candle,
            "is_red_candle": is_red_candle,
            "pivot_suppressions": []
        }

        ppo_ctx = {"curr": ppo_curr, "prev": ppo_prev}
        ppo_sig_ctx = {"curr": ppo_sig_curr, "prev": ppo_sig_prev}
        rsi_ctx = {"curr": rsi_curr, "prev": rsi_prev}

        # ===================================================================
        # STEP 10: PREPARE FOR BATCHED REDIS OPERATION
        # ===================================================================
        raw_alerts = []
        alert_keys_to_check = [
            def_["key"] for def_ in ALERT_DEFINITIONS
            if not (
                ("pivots" in def_["requires"] and not context.get("pivots"))
                or ("vwap" in def_["requires"] and not context.get("vwap"))
            )
        ]
        redis_alert_keys = [ALERT_KEYS[k] for k in alert_keys_to_check]
        all_state_changes = []

        # Get previous states FIRST (needed for trigger logic)
        previous_states = await check_multiple_alert_states(
            sdb, pair_name, redis_alert_keys
        )

        # ===================================================================
        # STEP 11: TRIGGER LOGIC - CHECK ALL ALERTS
        # ===================================================================
        for alert_key in alert_keys_to_check:
            def_ = ALERT_DEFINITIONS_MAP.get(alert_key)
            if not def_:
                continue

            try:
                key = ALERT_KEYS[alert_key]
                trigger = False

                # Special handling for pivot alerts
                if alert_key.startswith("pivot_up_") or alert_key.startswith("pivot_down_"):
                    level = alert_key.split("_")[-1]
                    is_buy = alert_key.startswith("pivot_up_")
                    valid_cross, reason = _validate_pivot_cross(context, level, is_buy)

                    if not valid_cross and reason and context.get("pivots"):
                        context["pivot_suppressions"].append(reason)

                    trigger = (
                        (is_buy and context["buy_common"])
                        or (not is_buy and context["sell_common"])
                    ) and valid_cross
                else:
                    # Standard alert check
                    trigger = def_["check_fn"](context, ppo_ctx, ppo_sig_ctx, rsi_ctx)

                # Only add alert if triggered AND not previously active
                if trigger and not previous_states.get(key, False):
                    extra = def_["extra_fn"](context, ppo_ctx, ppo_sig_ctx, rsi_ctx, None)
                    raw_alerts.append((def_["title"], extra, def_["key"]))
                    all_state_changes.append((f"{pair_name}:{key}", "ACTIVE", None))

                    if cfg.DEBUG_MODE:
                        logger_pair.debug(f"✅ NEW alert: {pair_name}:{alert_key}")

            except Exception as e:
                logger_pair.warning(
                    f"Alert check failed for {pair_name}, key={alert_key}: {e}"
                )

        # ===================================================================
        # STEP 12: STATE RESETS (for crossover alerts that can re-trigger)
        # ===================================================================
        if ppo_prev > ppo_sig_prev and ppo_curr <= ppo_sig_curr:
            all_state_changes.append(
                (f"{pair_name}:{ALERT_KEYS['ppo_signal_up']}", "INACTIVE", None)
            )
        if ppo_prev < ppo_sig_prev and ppo_curr >= ppo_sig_curr:
            all_state_changes.append(
                (f"{pair_name}:{ALERT_KEYS['ppo_signal_down']}", "INACTIVE", None)
            )

        if context["vwap"]:
            if close_prev >= vwap_prev and close_curr < vwap_curr:
                all_state_changes.append(
                    (f"{pair_name}:{ALERT_KEYS['vwap_up']}", "INACTIVE", None)
                )
            if close_prev <= vwap_prev and close_curr > vwap_curr:
                all_state_changes.append(
                    (f"{pair_name}:{ALERT_KEYS['vwap_down']}", "INACTIVE", None)
                )

        if piv:
            for lvl_n, lvl_v in piv.items():
                if close_prev > lvl_v and close_curr <= lvl_v:
                    all_state_changes.append(
                        (f"{pair_name}:{ALERT_KEYS[f'pivot_up_{lvl_n}']}", "INACTIVE", None)
                    )
                if close_prev < lvl_v and close_curr >= lvl_v:
                    all_state_changes.append(
                        (f"{pair_name}:{ALERT_KEYS[f'pivot_down_{lvl_n}']}", "INACTIVE", None)
                    )

        # MMH reversal resets
        if (mmh_curr > 0 and mmh_curr <= mmh_m1) and \
           await was_alert_active(sdb, pair_name, ALERT_KEYS["mmh_buy"]):
            all_state_changes.append(
                (f"{pair_name}:{ALERT_KEYS['mmh_buy']}", "INACTIVE", None)
            )
        if (mmh_curr < 0 and mmh_curr >= mmh_m1) and \
           await was_alert_active(sdb, pair_name, ALERT_KEYS["mmh_sell"]):
            all_state_changes.append(
                (f"{pair_name}:{ALERT_KEYS['mmh_sell']}", "INACTIVE", None)
            )

        # ===================================================================
        # STEP 13: ATOMIC BATCH OPERATION (states + dedup in single Redis call)
        # ===================================================================
        # ⚡ OPTIMIZED: Single atomic Redis operation
        dedup_checks = [(pair_name, ak, ts_curr) for _, _, ak in raw_alerts]

        if all_state_changes or dedup_checks:
            # Single Redis pipeline: get states + update + dedup
            previous_states, dedup_results = await sdb.atomic_eval_batch(
                pair_name,
                redis_alert_keys,
                all_state_changes,
                dedup_checks
            )
        else:
            dedup_results = {}

        # ===================================================================
        # STEP 14: DEDUP & SEND ALERTS
        # ===================================================================
        alerts_to_send = []
        if raw_alerts:
            for title, extra, ak in raw_alerts:
                # Check if we should send (not a duplicate)
                if dedup_results.get(f"{pair_name}:{ak}", True):
                    alerts_to_send.append((title, extra, ak))

        if alerts_to_send:
            # Limit alerts per pair
            alerts_to_send = alerts_to_send[:cfg.MAX_ALERTS_PER_PAIR]

            # Format message
            if len(alerts_to_send) == 1:
                title, extra, _ = alerts_to_send[0]
                msg = build_single_msg(title, pair_name, close_curr, ts_curr, extra)
            else:
                items = [(t, e) for t, e, _ in alerts_to_send[:25]]
                msg = build_batched_msg(pair_name, close_curr, ts_curr, items)

            # Send to Telegram
            if not cfg.DRY_RUN_MODE:
                await telegram_queue.send(msg)

            logger_pair.info(
                f"🔵🎯🟠 Sent {len(alerts_to_send)} alerts for {pair_name} | "
                f"{[ak for _, _, ak in alerts_to_send]}"
            )

        # ===================================================================
        # STEP 15: SUMMARY OUTPUT
        # ===================================================================
        reasons = []
        if not buy_common and not sell_common:
            reasons.append("Trend filter blocked")
        if context.get("candle_quality_failed_buy"):
            reasons.append(f"BUY quality: {buy_candle_reason}")
        if context.get("candle_quality_failed_sell"):
            reasons.append(f"SELL quality: {sell_candle_reason}")
        if context.get("pivot_suppressions"):
            reasons.extend(context["pivot_suppressions"])

        status_msg = (
            f"✔ {pair_name} | "
            f"cloud={'green' if cloud_up else 'red' if cloud_down else 'neutral'} "
            f"mmh={mmh_curr:.2f}"
        )

        if not alerts_to_send:
            logger_pair.debug(
                status_msg +
                (" | No signals" if not (base_buy_trend or base_sell_trend) else "")
            )

        return pair_name, {
            "state": "ALERT_SENT" if alerts_to_send else "NO_SIGNAL",
            "ts": int(time.time()),
            "summary": {
                "alerts": len(alerts_to_send),
                "cloud": "green" if cloud_up else "red" if cloud_down else "neutral",
                "mmh_hist": round(mmh_curr, 4),
                "suppression": "; ".join(reasons) or "No conditions met"
            }
        }

    except Exception as e:
        logger_pair.exception(f"❌ Error in evaluate_pair_and_alert for {pair_name}: {e}")
        return None

    finally:
        PAIR_ID.set("")

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
    
    # 1. PHASE 1: PARALLEL DATA FETCH
    logger_main.info(f"📡 Phase 1: Fetching candles for {len(pairs_to_process)} pairs...")
    fetch_start = time.time()
    
    limit_15m = max(200, cfg.RMA_200_PERIOD + 25)
    limit_5m = max(300, cfg.RMA_200_PERIOD + 50)
    daily_limit = cfg.PIVOT_LOOKBACK_PERIOD + 5 if cfg.ENABLE_PIVOT else 0
    
    pair_requests = []
    for pair_name in pairs_to_process:
        product_info = products_map.get(pair_name)
        if not product_info:
            continue
        
        symbol = product_info["symbol"]
        resolutions = [("15", limit_15m), ("5", limit_5m)]
        if cfg.ENABLE_PIVOT:
            resolutions.append(("D", daily_limit))
        
        pair_requests.append((symbol, resolutions))
    
    # Fetch all candles in parallel
    all_candles = await fetcher.fetch_all_candles_truly_parallel(
        pair_requests, reference_time
    )
    
    # 2. PHASE 2: PREPARE TASKS (Parsing & evaluation will happen in Phase 3)
    logger_main.info(f"🧠 Phase 2: Preparing evaluation tasks for fetched pairs...")
    
    tasks = []
    for pair_name, candles_by_res in all_candles.items():
        if not candles_by_res:  # No data fetched at all
            continue
        # Only include if we at least have some candles
        if any(candles_by_res.get(res) for res in ["15", "5", "D"] if res in candles_by_res):
            tasks.append((pair_name, candles_by_res))
    
    if not tasks:
        logger_main.info("No valid candle data received for any pair.")
        return []
    
    # 3. PHASE 3: EVALUATE WITH CONCURRENCY CONTROL
    logger_main.info(f"🧠 Phase 3: Evaluating {len(tasks)} pairs...")
    eval_start = time.time()
    
    # Limit concurrent heavy work (CPU + Redis I/O)
    semaphore = asyncio.Semaphore(cfg.MAX_PARALLEL_FETCH)
    
    async def guarded_eval(task_data):
        p_name, candles = task_data
        async with semaphore:
            try:
                if cfg.DEBUG_MODE:
                    logger_main.debug(
                        f"Processing {p_name} | Candle keys: {list(candles.keys())}"
                    )
                
                # Parse candles
                data_15m = parse_candles_to_numpy(candles.get("15"))
                data_5m = parse_candles_to_numpy(candles.get("5"))
                data_daily = (
                    parse_candles_to_numpy(candles.get("D"))
                    if cfg.ENABLE_PIVOT else None
                )
                
                # Quick validation on 15m (critical for strategy)
                v15, r15 = validate_candle_data(
                    data_15m, required_len=cfg.RMA_200_PERIOD
                )
                if not v15:
                    logger_main.warning(f"Skipping {p_name}: 15m invalid ({r15})")
                    return None
                
                # Full evaluation + alerting
                return await evaluate_pair_and_alert(
                    p_name,
                    data_15m,
                    data_5m,
                    data_daily,
                    state_db,
                    telegram_queue,
                    correlation_id,
                    reference_time
                )
            except Exception as e:
                logger_main.error(f"Error evaluating {p_name}: {e}", exc_info=True)
                return None
    
    # Run all evaluations concurrently
    results = await asyncio.gather(*[guarded_eval(task) for task in tasks])
    
    # Filter out failed/skipped pairs
    valid_results = [r for r in results if r is not None]
    
    total_time = time.time() - fetch_start
    logger_main.info(
        f"✅ Run complete | Successful: {len(valid_results)}/{len(pairs_to_process)} "
        f"| Total time: {total_time:.2f}s "
        f"(Fetch+Prep: {eval_start - fetch_start:.2f}s, Eval: {time.time() - eval_start:.2f}s)"
    )
    
    return valid_results

# ============================================================================
# OPTIMIZATION 9: Enhanced run_once with Smart Product Caching
# ============================================================================

async def run_once() -> bool:
    """OPTIMIZED: Smarter caching and parallel execution"""
    
    gc.disable()
    
    correlation_id = uuid.uuid4().hex[:8]
    TRACE_ID.set(correlation_id)
    logger_run = logging.getLogger(f"macd_bot.run.{correlation_id}")
    start_time = time.time()

    reference_time = get_trigger_timestamp()
    logger_run.info(
        f"🚀 Run started | Correlation ID: {correlation_id} | "
        f"Reference time: {reference_time} ({format_ist_time(reference_time)})"
    )

    sdb: Optional[RedisStateStore] = None
    lock: Optional[RedisLock] = None
    lock_acquired = False
    fetcher: Optional[DataFetcher] = None
    telegram_queue: Optional[TelegramQueue] = None
    
    alerts_sent = 0
    MAX_ALERTS_PER_RUN = 50

    try:
        process = psutil.Process()
        container_memory_mb = process.memory_info().rss / 1024 / 1024
        limit_mb = cfg.MEMORY_LIMIT_BYTES / 1024 / 1024
        
        if container_memory_mb >= limit_mb:
            logger_run.critical(
                f"🚨 Memory limit exceeded at startup "
                f"({container_memory_mb:.1f}MB / {limit_mb:.1f}MB)"
            )
            return False

        # ===================================================================
        # STEP 1: PRODUCTS MAP (Static-first with API fallback)
        # ===================================================================
        products_map = None
        pairs_to_process = []
        now = time.time()
        
        # Try static map first
        USE_STATIC_MAP = True
        STATIC_MAP_REFRESH_DAYS = 7
        
        if USE_STATIC_MAP:
            # Check if we need API refresh (connect to Redis temporarily)
            last_api_check_key = "last_products_api_check"
            last_check = 0.0
            
            try:
                temp_sdb = RedisStateStore(cfg.REDIS_URL)
                await temp_sdb.connect()
                last_check_str = await temp_sdb.get_metadata(last_api_check_key)
                last_check = float(last_check_str) if last_check_str else 0.0
                await temp_sdb.close()
            except Exception:
                last_check = 0.0
            
            days_since_check = (now - last_check) / 86400
            
            if days_since_check < STATIC_MAP_REFRESH_DAYS:
                logger_run.info(f"⚡ Using static products map (last API check: {days_since_check:.1f} days ago)")
                products_map = STATIC_PRODUCTS_MAP.copy()
            else:
                logger_run.info(f"🔄 Refreshing products map from API (last check: {days_since_check:.1f} days ago)")
                USE_STATIC_MAP = False
        
        if not USE_STATIC_MAP or products_map is None:
            # Fallback to API fetch
            logger_run.info("📡 Fetching products list from Delta API...")
            temp_fetcher = DataFetcher(cfg.DELTA_API_BASE)
            prod_resp = await temp_fetcher.fetch_products()
            
            if not prod_resp:
                logger_run.warning("⚠️ API fetch failed, falling back to static map")
                products_map = STATIC_PRODUCTS_MAP.copy()
            else:
                products_map = build_products_map_from_api_result(prod_resp)
                
                # Update last check timestamp (will do this after Redis connects properly)
                logger_run.info(f"✅ Products list fetched from API ({len(products_map)} pairs)")
        
        pairs_to_process = [p for p in cfg.PAIRS if p in products_map]

        if len(pairs_to_process) < len(cfg.PAIRS):
            missing = set(cfg.PAIRS) - set(pairs_to_process)
            logger_run.warning(f"⚠️ Missing products for pairs: {missing}")
        
        if not pairs_to_process:
            logger_run.error("❌ No valid pairs to process - aborting run")
            return False

        # ===================================================================
        # STEP 2: CONNECT TO REDIS
        # ===================================================================
        logger_run.debug("Connecting to Redis...")
        sdb = RedisStateStore(cfg.REDIS_URL)
        await sdb.connect()
        
        # Update API check timestamp if we fetched from API
        if not USE_STATIC_MAP and products_map and not sdb.degraded:
            try:
                await sdb.set_metadata(last_api_check_key, str(now))
            except Exception as e:
                logger_run.debug(f"Failed to update API check timestamp: {e}")
        
        if sdb.degraded and not sdb.degraded_alerted:
            logger_run.critical(
                "⚠️ Redis is in degraded mode – alert deduplication disabled!"
            )
            telegram_queue = TelegramQueue(cfg.TELEGRAM_BOT_TOKEN, cfg.TELEGRAM_CHAT_ID)
            await telegram_queue.send(escape_markdown_v2(
                f"⚠️ {cfg.BOT_NAME} - REDIS DEGRADED MODE\n"
                f"Alert deduplication is disabled. You may receive duplicate alerts.\n"
                f"Time: {format_ist_time()}"
            ))
            sdb.degraded_alerted = True

        # ===================================================================
        # STEP 3: INITIALIZE FETCHER & TELEGRAM
        # ===================================================================
        fetcher = DataFetcher(cfg.DELTA_API_BASE)
        if telegram_queue is None:
            telegram_queue = TelegramQueue(cfg.TELEGRAM_BOT_TOKEN, cfg.TELEGRAM_CHAT_ID)

        # ===================================================================
        # STEP 4: ACQUIRE DISTRIBUTED LOCK
        # ===================================================================
        lock = RedisLock(sdb._redis, "macd_bot_run")
        lock_acquired = await lock.acquire(timeout=5.0)
        
        if not lock_acquired:
            logger_run.warning(
                "⏸️ Another instance is running (Redis lock held) - exiting gracefully"
            )
            return False

        logger_run.debug("🔒 Distributed lock acquired successfully")

        # ===================================================================
        # STEP 5: SEND TEST MESSAGE (if enabled)
        # ===================================================================
        if cfg.SEND_TEST_MESSAGE:
            await telegram_queue.send(escape_markdown_v2(
                f"🔥 {cfg.BOT_NAME} - Run Started\n"
                f"Date: {format_ist_time(datetime.now(timezone.utc))}\n"
                f"Correlation ID: {correlation_id}\n"
                f"Pairs: {len(pairs_to_process)}"
            ))

        # ===================================================================
        # STEP 6: PROCESS ALL PAIRS
        # ===================================================================
        logger_run.debug(
            f"📊 Processing {len(pairs_to_process)} pairs using optimized parallel architecture"
        )

        all_results = await process_pairs_with_workers(
            fetcher, products_map, pairs_to_process,
            sdb, telegram_queue, correlation_id,
            lock, reference_time
        )

        # ===================================================================
        # STEP 7: COLLECT STATISTICS
        # ===================================================================
        for _, state in all_results:
            if state.get("state") == "ALERT_SENT":
                alerts_sent += state.get("summary", {}).get("alerts", 0)

        fetcher_stats = fetcher.get_stats()
        logger_run.info(
            f"📡 Fetch statistics | "
            f"Products: {fetcher_stats['products_success']}✅/{fetcher_stats['products_failed']}❌ | "
            f"Candles: {fetcher_stats['candles_success']}✅/{fetcher_stats['candles_failed']}❌"
        )
        
        if "rate_limiter" in fetcher_stats:
            rate_stats = fetcher_stats["rate_limiter"]
            if rate_stats.get("total_waits", 0) > 0:
                logger_run.info(
                    f"🚦 Rate limiting stats | "
                    f"Waits: {rate_stats['total_waits']} | "
                    f"Total wait time: {rate_stats['total_wait_time_seconds']:.1f}s"
                )

        # ===================================================================
        # STEP 8: FINAL SUMMARY
        # ===================================================================
        final_memory_mb = process.memory_info().rss / 1024 / 1024
        memory_delta = final_memory_mb - container_memory_mb
        
        run_duration = time.time() - start_time
        redis_status = "OK" if not sdb.degraded else "DEGRADED"
        
        summary = (
            f"✅ RUN COMPLETE | "
            f"Duration: {run_duration:.1f}s | "
            f"Pairs: {len(all_results)}/{len(pairs_to_process)} | "
            f"Alerts: {alerts_sent} | "
            f"Memory: {int(final_memory_mb)}MB (Δ{memory_delta:+.0f}MB) | "
            f"Redis: {redis_status}"
        )
        logger_run.info(summary)

        if alerts_sent > MAX_ALERTS_PER_RUN:
            await telegram_queue.send(escape_markdown_v2(
                f"⚠️ HIGH ALERT VOLUME\n"
                f"Alerts sent: {alerts_sent}\n"
                f"Pairs processed: {len(all_results)}\n"
                f"Time: {format_ist_time()}"
            ))

        return True

    except asyncio.TimeoutError:
        logger_run.error("⏱️ Run timed out - exceeded RUN_TIMEOUT_SECONDS")
        return False
    
    except asyncio.CancelledError:
        logger_run.warning("🛑 Run cancelled (shutdown signal received)")
        return False
    
    except Exception as e:
        logger_run.exception(f"❌ Fatal error in run_once: {e}")
        
        if telegram_queue:
            try:
                await telegram_queue.send(escape_markdown_v2(
                    f"❌ {cfg.BOT_NAME} - FATAL ERROR\n"
                    f"Error: {str(e)[:200]}\n"
                    f"Correlation ID: {correlation_id}\n"
                    f"Time: {format_ist_time()}"
                ))
            except Exception:
                logger_run.error("Failed to send error notification")
        
        return False
    
    finally:
        logger_run.debug("🧹 Starting resource cleanup...")
        
        # Release distributed lock
        if lock_acquired and lock and lock.acquired_by_me:
            try:
                await lock.release(timeout=3.0)
                logger_run.debug("🔓 Redis lock released")
            except Exception as e:
                logger_run.error(f"Error releasing lock: {e}")
        
        # Close Redis connection (but NOT the global pool)
        if sdb:
            try:
                await sdb.close()
                logger_run.debug("✅ Redis connection closed")
            except Exception as e:
                logger_run.error(f"Error closing Redis: {e}")
        
        # Memory cleanup
        try:
            if 'all_results' in locals():
                del all_results
            if 'products_map' in locals():
                del products_map
            if 'fetcher' in locals():
                del fetcher
            if 'telegram_queue' in locals():
                del telegram_queue
            
            gc.collect()
            
            logger_run.debug("✅ Memory cleanup completed")
        except Exception as e:
            logger_run.warning(f"Memory cleanup warning (non-critical): {e}")
        
        TRACE_ID.set("")
        PAIR_ID.set("")
        
        logger_run.debug("🏁 Resource cleanup finished")
        
        gc.enable()


# ============================================================================
# UVLOOP SETUP & MAIN ENTRY POINT
# ============================================================================

try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    logger.info(f"✅ uvloop enabled | {JSON_BACKEND} enabled")
except ImportError:
    logger.info(f"ℹ️ uvloop not available (using default) | {JSON_BACKEND} enabled")


if if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="macd_unified",
        description="Unified MACD/alerts runner with NumPy optimization"
    )
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging")
    parser.add_argument("--validate-only", action="store_true", help="Validate config and exit")
    parser.add_argument("--skip-warmup", action="store_true", help="Skip Numba JIT warmup")
    args = parser.parse_args()

    # Enable debug logging if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        for h in logger.handlers:
            h.setLevel(logging.DEBUG)
        logger.info("Debug mode enabled via CLI flag")

    # Validate configuration early
    try:
        validate_runtime_config()
    except ValueError as e:
        logger.critical(f"Configuration validation failed: {e}")
        sys.exit(1)
    
    if args.validate_only:
        logger.info("Configuration validation passed - exiting (--validate-only mode)")
        sys.exit(0)

    # Warmup Numba JIT unless explicitly skipped
    if not args.skip_warmup and not cfg.SKIP_WARMUP:
        warmup_numba()
    else:
        logger.info("Skipping Numba warmup (faster startup)")

    async def main_with_cleanup():
        """Run the bot with proper resource cleanup on exit"""
        try:
            return await run_once()
        finally:
            # Cleanup persistent connections on shutdown
            logger.info("🧹 Shutting down persistent connections...")
            try:
                await RedisStateStore.shutdown_global_pool()
                logger.debug("✅ Redis pool closed")
            except Exception as e:
                logger.error(f"Error closing Redis pool: {e}")
            
            try:
                await SessionManager.close_session()
                logger.debug("✅ HTTP session closed")
            except Exception as e:
                logger.error(f"Error closing HTTP session: {e}")

    # Run the async main function
    try:
        success = asyncio.run(main_with_cleanup())
        if success:
            logger.info("✅ Bot run completed successfully")
            sys.exit(0)
        else:
            logger.error("❌ Bot run failed")
            sys.exit(1)
    except (asyncio.CancelledError, KeyboardInterrupt):
        logger.info("Bot stopped by timeout or user interrupt")
        sys.exit(130)
    except Exception as exc:
        logger.critical(f"Fatal error: {exc}", exc_info=True)
        sys.exit(1)