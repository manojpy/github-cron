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
from numba import njit
import warnings


warnings.filterwarnings('ignore', category=RuntimeWarning, module='pycparser')
warnings.filterwarnings('ignore', message='.*parsing methods must have __doc__.*')


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

__version__ = "1.3.0-numpy-optimized"

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
            # Quick check before expensive regex
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
    """Optimized debug logger - only calls msg_fn if debug enabled"""
    if condition and logger_obj.isEnabledFor(logging.DEBUG):
        logger_obj.debug(msg_fn())


# ADD this helper for conditional info logging:
def info_if_important(logger_obj: logging.Logger, is_important: bool, msg: str) -> None:
    """Only log INFO for important events, use DEBUG otherwise"""
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
    
    if cfg.MAX_PARALLEL_FETCH < 1 or cfg.MAX_PARALLEL_FETCH > 16:
        warnings.append(f"MAX_PARALLEL_FETCH={cfg.MAX_PARALLEL_FETCH} is outside recommended range (1-16)")
    
    if cfg.HTTP_TIMEOUT < 5 or cfg.HTTP_TIMEOUT > 60:
        warnings.append(f"HTTP_TIMEOUT={cfg.HTTP_TIMEOUT}s is outside recommended range (5-60s)")
    
    if cfg.RUN_TIMEOUT_SECONDS < 300:
        errors.append(f"RUN_TIMEOUT_SECONDS={cfg.RUN_TIMEOUT_SECONDS}s is too low (minimum: 300s)")
    
    if cfg.RUN_TIMEOUT_SECONDS >= Constants.REDIS_LOCK_EXPIRY:
        errors.append(
            f"RUN_TIMEOUT_SECONDS ({cfg.RUN_TIMEOUT_SECONDS}s) must be less than "
            f"REDIS_LOCK_EXPIRY ({Constants.REDIS_LOCK_EXPIRY}s)"
        )
    
    if not cfg.PAIRS or len(cfg.PAIRS) == 0:
        errors.append("PAIRS list is empty - no trading pairs configured")
    
    if len(cfg.PAIRS) > 20:
        warnings.append(f"Large number of pairs ({len(cfg.PAIRS)}) may exceed timeout limits")
    
    if cfg.MEMORY_LIMIT_BYTES < 200_000_000:  # 200MB minimum
        warnings.append(f"MEMORY_LIMIT_BYTES={cfg.MEMORY_LIMIT_BYTES} is very low (minimum recommended: 200MB)")
    
    if cfg.PPO_FAST >= cfg.PPO_SLOW:
        errors.append(f"PPO_FAST ({cfg.PPO_FAST}) must be less than PPO_SLOW ({cfg.PPO_SLOW})")
    
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
    current_window = reference_time // interval_seconds
    last_closed_candle = (current_window * interval_seconds) - interval_seconds
    return last_closed_candle

def validate_candle_timestamp(candle_ts: int, expected_ts: int, tolerance_seconds: int = 120) -> bool:
    diff = abs(candle_ts - expected_ts)
    if diff > tolerance_seconds:
        logger.error(f"Candle timestamp mismatch! Expected ~{expected_ts}, got {candle_ts} (diff: {diff}s)")
        return False

_ESCAPE_RE = re.compile(r'[_*\[\]()~`>#+-=|{}.!]')

def escape_markdown_v2(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    return CompiledPatterns.ESCAPE_MARKDOWN.sub(r'\\\g<0>', text)

# ============================================================================
# PART 5: NUMBA-OPTIMIZED INDICATOR CALCULATIONS (ENHANCED SAFETY)
# ============================================================================

def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
            return default
        return float(value)
    except (ValueError, TypeError, OverflowError):
        return default

def sanitize_indicator_array(arr: np.ndarray, name: str, default: float = 0.0) -> np.ndarray:
    try:
        if arr is None or len(arr) == 0:
            logger.warning(f"Indicator {name} is None or empty")
            return np.array([default], dtype=np.float64)
        
        arr = np.where(np.isinf(arr), np.nan, arr)
        
        nan_count = np.sum(np.isnan(arr))
        if nan_count > 0:
            logger.debug(f"Indicator {name} has {nan_count}/{len(arr)} NaN values, filling with {default}")
        
        arr = np.where(np.isnan(arr), default, arr)
        
        if np.all(arr == default):
            logger.warning(f"Indicator {name} is all {default} after sanitization")
        
        return arr.astype(np.float64)
        
    except Exception as e:
        logger.error(f"Failed to sanitize indicator {name}: {e}")
        return np.full(len(arr) if arr is not None else 1, default, dtype=np.float64)

@njit(fastmath=True, cache=True)
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

@njit(fastmath=True, cache=True)
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

@njit(fastmath=True, cache=True)
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

@njit(fastmath=True, cache=True)
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

@njit(fastmath=True, cache=True)
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

@njit(fastmath=True, cache=True)
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

@njit(fastmath=True, cache=True)
def _calc_mmh_value_loop(temp_arr, rows):
    value_arr = np.zeros(rows, dtype=np.float64)
    value_arr[0] = 1.0
    
    for i in range(1, rows):
        prev_v = value_arr[i - 1] if not np.isnan(value_arr[i - 1]) else 1.0
        t = temp_arr[i] if not np.isnan(temp_arr[i]) else 0.5
        v = t - 0.5 + 0.5 * prev_v
        value_arr[i] = max(-0.9999, min(0.9999, v))
    
    return value_arr

@njit(fastmath=True, cache=True)
def _calc_mmh_momentum_loop(momentum_arr, rows):
    for i in range(1, rows):
        prev = momentum_arr[i - 1] if not np.isnan(momentum_arr[i - 1]) else 0.0
        momentum_arr[i] = momentum_arr[i] + 0.5 * prev
    return momentum_arr

@njit(fastmath=True, cache=True)
def _rolling_std_numba(close: np.ndarray, period: int, responsiveness: float) -> np.ndarray:
    rows = len(close)
    sd = np.empty(rows, dtype=np.float64)
    resp = max(0.00001, min(1.0, responsiveness))
    for i in range(rows):
        start = max(0, i - period + 1)
        sum_val = 0.0
        sum_sq = 0.0
        count = 0
        for j in range(start, i + 1):
            val = close[j]
            if not np.isnan(val):
                sum_val += val
                sum_sq += val * val
                count += 1
        if count > 1:
            mean = sum_val / count
            var = (sum_sq / count) - mean * mean
            sd[i] = np.sqrt(max(0.0, var)) * resp
        else:
            sd[i] = 0.0
    return sd

@njit(fastmath=True, cache=True)
def _rolling_mean_numba(close: np.ndarray, period: int) -> np.ndarray:
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

@njit(fastmath=True, cache=True)
def _rolling_min_max_numba(arr: np.ndarray, period: int) -> Tuple[np.ndarray, np.ndarray]:
    rows = len(arr)
    min_arr = np.empty(rows, dtype=np.float64)
    max_arr = np.empty(rows, dtype=np.float64)
    for i in range(rows):
        start = max(0, i - period + 1)
        min_arr[i] = np.min(arr[start:i+1])
        max_arr[i] = np.max(arr[start:i+1])
    return min_arr, max_arr

def calculate_ppo_numpy(close: np.ndarray, fast: int, slow: int, signal: int) -> Tuple[np.ndarray, np.ndarray]:
    try:
        if close is None or len(close) < max(fast, slow):
            logger.warning(f"PPO: Insufficient data (len={len(close) if close is not None else 0})")
            default_len = len(close) if close is not None else 1
            return np.zeros(default_len, dtype=np.float64), np.zeros(default_len, dtype=np.float64)
        
        alpha_fast = 2.0 / (fast + 1)
        fast_ma = _ema_loop(close, alpha_fast)
        
        alpha_slow = 2.0 / (slow + 1)
        slow_ma = _ema_loop(close, alpha_slow)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            ppo = ((fast_ma - slow_ma) / slow_ma) * 100.0
        
        ppo = np.nan_to_num(ppo, nan=0.0, posinf=0.0, neginf=0.0)
        
        alpha_signal = 2.0 / (signal + 1)
        ppo_sig = _ema_loop(ppo, alpha_signal)
        
        ppo = sanitize_indicator_array(ppo, "PPO", default=0.0)
        ppo_sig = sanitize_indicator_array(ppo_sig, "PPO_Signal", default=0.0)
        
        return ppo, ppo_sig
        
    except Exception as e:
        logger.error(f"PPO calculation failed: {e}")
        default_len = len(close) if close is not None else 1
        return np.zeros(default_len, dtype=np.float64), np.zeros(default_len, dtype=np.float64)

def calculate_smooth_rsi_numpy(close: np.ndarray, rsi_len: int, kalman_len: int) -> np.ndarray:
    try:
        if close is None or len(close) < rsi_len:
            logger.warning(f"Smooth RSI: Insufficient data (len={len(close) if close is not None else 0})")
            return np.full(len(close) if close is not None else 1, 50.0, dtype=np.float64)
        
        delta = np.zeros_like(close)
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
        
        smooth_rsi = sanitize_indicator_array(smooth_rsi, "Smooth_RSI", default=50.0)
        
        return smooth_rsi
        
    except Exception as e:
        logger.error(f"Smooth RSI calculation failed: {e}")
        return np.full(len(close) if close is not None else 1, 50.0, dtype=np.float64)

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

def calculate_magical_momentum_hist(close: np.ndarray, period: int = 144, responsiveness: float = 0.9) -> np.ndarray:
    try:
        if close is None or len(close) < period:
            logger.warning(f"MMH: Insufficient data (len={len(close) if close is not None else 0})")
            return np.zeros(len(close) if close is not None else 1, dtype=np.float32)
        
        rows = len(close)
        resp_clamped = max(0.00001, min(1.0, float(responsiveness)))
        
        sd = _rolling_std_numba(close.astype(np.float32), 50, resp_clamped)
        
        worm_arr = _calc_mmh_worm_loop(close.astype(np.float32), sd, rows)
        
        ma = _rolling_mean_numba(close.astype(np.float32), period)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            raw = (worm_arr - ma) / worm_arr
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        
        min_med, max_med = _rolling_min_max_numba(raw, period)
        
        denom = max_med - min_med
        denom = np.where(denom == 0, 1e-12, denom)
        temp = (raw - min_med) / denom
        temp = np.clip(temp, 0.0, 1.0)
        temp = np.nan_to_num(temp, nan=0.5)
        
        value_arr = _calc_mmh_value_loop(temp, rows)
        value_arr = np.clip(value_arr, -0.9999999, 0.9999999)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            temp2 = (1.0 + value_arr) / (1.0 - value_arr)
            temp2 = np.nan_to_num(temp2, nan=1e8, posinf=1e8, neginf=-1e8)
        
        momentum = 0.25 * np.log(temp2)
        momentum = np.nan_to_num(momentum, nan=0.0)
        
        momentum_arr = momentum.copy()
        momentum_arr = _calc_mmh_momentum_loop(momentum_arr, rows)
        
        momentum_arr = sanitize_indicator_array(momentum_arr.astype(np.float32), "MMH_Hist", default=0.0)
        
        return momentum_arr
        
    except Exception as e:
        logger.error(f"MMH calculation failed: {e}")
        return np.zeros(len(close) if close is not None else 1, dtype=np.float32)

def warmup_numba() -> None:
        logger.info("Warming up Numba JIT compiler (parallel)...")
        
        try:
            length = 100
            close = np.random.random(length).astype(np.float64) * 1000
            high = close + np.random.random(length).astype(np.float64) * 10
            low = close - np.random.random(length).astype(np.float64) * 10
            volume = np.random.random(length).astype(np.float64) * 1000
            timestamps = np.arange(length, dtype=np.int64) * 900
            sd = np.random.random(length).astype(np.float64) * 0.01
            temp = np.random.random(length).astype(np.float64)
            
            critical_funcs = [
                lambda: _ema_loop(close, 0.1),
                lambda: _sma_loop(close, 20),
                lambda: _kalman_loop(close, 21, 0.01, 0.1),
                lambda: _vwap_daily_loop(high, low, close, volume, timestamps),
                lambda: _rng_filter_loop(close, sd),
                lambda: _calc_mmh_worm_loop(close, sd, length),
                lambda: _calc_mmh_value_loop(temp, length),
                lambda: _rolling_std_numba(close, 50, 0.9),
                lambda: _fast_array_copy(close, sd, length),
            ]
            
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(func) for func in critical_funcs]
                concurrent.futures.wait(futures, timeout=10.0)
            
            logger.info("Numba warm-up complete (parallel compilation)")
            
        except Exception as e:
            logger.warning(f"Numba warm-up failed (non-fatal): {e}")

indicator_semaphore = asyncio.Semaphore(cfg.INDICATOR_THREAD_LIMIT)

async def calculate_indicator_threaded(func: Callable, *args, **kwargs):
    return await asyncio.to_thread(func, *args, **kwargs)

# ============================================================================
# PRE-ALLOCATED NUMPY BUFFERS FOR ADVANCED OPTIMIZATION
# ============================================================================

_BUFFER_SIZE = 500
_BUFFER_LOCK = asyncio.Lock()

GLOBAL_CLOSE_BUF = np.empty(_BUFFER_SIZE, dtype=np.float64)
GLOBAL_HIGH_BUF = np.empty(_BUFFER_SIZE, dtype=np.float64)
GLOBAL_LOW_BUF = np.empty(_BUFFER_SIZE, dtype=np.float64)
GLOBAL_VOLUME_BUF = np.empty(_BUFFER_SIZE, dtype=np.float64)
GLOBAL_TIMESTAMP_BUF = np.empty(_BUFFER_SIZE, dtype=np.int64)
GLOBAL_TEMP_BUF_1 = np.empty(_BUFFER_SIZE, dtype=np.float64)
GLOBAL_TEMP_BUF_2 = np.empty(_BUFFER_SIZE, dtype=np.float64)

@njit(fastmath=True, cache=False, nogil=True)
def _fast_array_copy(src: np.ndarray, dst: np.ndarray, n: int) -> None:
    for i in range(n):
        dst[i] = src[i]

def get_scratch_buffers(size: int) -> Dict[str, np.ndarray]:
    if size > _BUFFER_SIZE:
        logger.debug(f"Buffer size {size} exceeds {_BUFFER_SIZE}, allocating new arrays")
        return {
            'close': np.empty(size, dtype=np.float64),
            'high': np.empty(size, dtype=np.float64),
            'low': np.empty(size, dtype=np.float64),
            'volume': np.empty(size, dtype=np.float64),
            'timestamp': np.empty(size, dtype=np.int64),
            'temp1': np.empty(size, dtype=np.float64),
            'temp2': np.empty(size, dtype=np.float64),
        }
    
    return {
        'close': GLOBAL_CLOSE_BUF[:size],
        'high': GLOBAL_HIGH_BUF[:size],
        'low': GLOBAL_LOW_BUF[:size],
        'volume': GLOBAL_VOLUME_BUF[:size],
        'timestamp': GLOBAL_TIMESTAMP_BUF[:size],
        'temp1': GLOBAL_TEMP_BUF_1[:size],
        'temp2': GLOBAL_TEMP_BUF_2[:size],
    }

def copy_to_scratch(data: Dict[str, np.ndarray], buffers: Dict[str, np.ndarray]) -> None:
    n = len(data['close'])
    if 'close' in buffers:
        _fast_array_copy(data['close'], buffers['close'], n)
    if 'high' in buffers and 'high' in data:
        _fast_array_copy(data['high'], buffers['high'], n)
    if 'low' in buffers and 'low' in data:
        _fast_array_copy(data['low'], buffers['low'], n)
    if 'volume' in buffers and 'volume' in data:
        _fast_array_copy(data['volume'], buffers['volume'], n)
    if 'timestamp' in buffers and 'timestamp' in data:
        # Special handling for int64
        for i in range(n):
            buffers['timestamp'][i] = data['timestamp'][i]

# ============================================================================
# PART 6: PIVOT CALCULATIONS & ALERT LOGIC
# ============================================================================

def calculate_pivot_levels_numpy(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    timestamps: np.ndarray
) -> Dict[str, float]:
    piv: Dict[str, float] = {}
    
    try:
        if len(timestamps) < 2:
            return piv
        
        days = timestamps // 86400  # 86400 seconds in a day
        
        now_utc = datetime.now(timezone.utc)
        yesterday = (now_utc - timedelta(days=1)).date()
        yesterday_day_number = int(yesterday.strftime('%s')) // 86400
        
        yesterday_mask = days == yesterday_day_number
        
        if not np.any(yesterday_mask):
            return piv
        
        yesterday_high = high[yesterday_mask]
        yesterday_low = low[yesterday_mask]
        yesterday_close = close[yesterday_mask]
        
        if len(yesterday_high) == 0:
            return piv
        
        H_prev = float(np.max(yesterday_high))
        L_prev = float(np.min(yesterday_low))
        C_prev = float(yesterday_close[-1])
        
        rng_prev = H_prev - L_prev
        
        if rng_prev > 1e-8:
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
        logger.warning(f"Pivot calculation failed: {e}")
    
    return piv

def calculate_all_indicators_numpy(
    data_15m: Dict[str, np.ndarray],
    data_5m: Dict[str, np.ndarray],
    data_daily: Optional[Dict[str, np.ndarray]]
) -> Dict[str, np.ndarray]:
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
        results['pivots'] = calculate_pivot_levels_numpy(
            data_daily["high"], data_daily["low"],
            data_daily["close"], data_daily["timestamp"]
        )
    else:
        results['pivots'] = {}
    
    return results

def precompute_candle_quality(
    data_15m: Dict[str, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pre-compute wick validation for all candles at once.
    Returns: (buy_quality, sell_quality) boolean arrays
    """
    buy_quality = _vectorized_wick_check_buy(
        data_15m["open"],
        data_15m["high"],
        data_15m["low"],
        data_15m["close"],
        Constants.MIN_WICK_RATIO
    )
    
    sell_quality = _vectorized_wick_check_sell(
        data_15m["open"],
        data_15m["high"],
        data_15m["low"],
        data_15m["close"],
        Constants.MIN_WICK_RATIO
    )
    
    return buy_quality, sell_quality

# ============================================================================
# SESSION MANAGEMENT & NETWORK
# ============================================================================

class SessionManager:
    _session: ClassVar[Optional[aiohttp.ClientSession]] = None
    _ssl_context: ClassVar[Optional[ssl.SSLContext]] = None
    _lock: ClassVar[asyncio.Lock] = asyncio.Lock()
    _creation_time: ClassVar[float] = 0.0
    _request_count: ClassVar[int] = 0
    _session_reuse_limit: ClassVar[int] = 1000  # Recreate session after N requests

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
                        await asyncio.sleep(0.25)  # Allow cleanup
                    except Exception as e:
                        logger.warning(f"Error closing old session: {e}")
                
                connector = TCPConnector(
                    limit=cfg.TCP_CONN_LIMIT,
                    limit_per_host=cfg.TCP_CONN_LIMIT_PER_HOST,
                    ssl=cls._get_ssl_context(),
                    force_close=False,  # Keep connections alive
                    enable_cleanup_closed=True,
                    ttl_dns_cache=3600,  # 1 hour DNS cache
                    keepalive_timeout=60,  # Increased from 30s for better reuse
                    family=0,  # Allow both IPv4 and IPv6
                )
                
                timeout = aiohttp.ClientTimeout(
                    total=cfg.HTTP_TIMEOUT,
                    connect=10,  # Connection timeout
                    sock_read=cfg.HTTP_TIMEOUT,  # Socket read timeout
                )
                
                cls._session = aiohttp.ClientSession(
                    connector=connector,
                    timeout=timeout,
                    headers={
                        'User-Agent': f'{cfg.BOT_NAME}/{__version__}',
                        'Accept': 'application/json',
                        'Accept-Encoding': 'gzip, deflate',
                    },
                    raise_for_status=False,
                )
                
                cls._creation_time = time.time()
                cls._request_count = 0
                
                logger.info(
                    f"HTTP session created | "
                    f"Pool: {cfg.TCP_CONN_LIMIT} total, {cfg.TCP_CONN_LIMIT_PER_HOST} per host | "
                    f"Timeout: {cfg.HTTP_TIMEOUT}s | "
                    f"DNS cache: 3600s | "
                    f"Keepalive: 30s"
                )
            
            cls._request_count += 1
            
            return cls._session

    @classmethod
    async def close_session(cls) -> None:
        async with cls._lock:
            if cls._session and not cls._session.closed:
                try:
                    session_age = time.time() - cls._creation_time
                    
                    logger.debug(
                        f"Closing HTTP session | "
                        f"Age: {session_age:.1f}s | "
                        f"Requests served: {cls._request_count}"
                    )
                    
                    await cls._session.close()
                    
                    await asyncio.sleep(0.25)
                    
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

# ============================================================================
# PART 4: DATA FETCHING & VALIDATION
# ============================================================================

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
        """Execute function with rate limiting."""
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

class DataFetcher:
    def __init__(self, api_base: str, max_parallel: Optional[int] = None):
        self.api_base = api_base.rstrip("/")
        max_parallel = max_parallel or cfg.MAX_PARALLEL_FETCH
        
        self.semaphore = asyncio.Semaphore(max_parallel)
        
        self.timeout = cfg.HTTP_TIMEOUT
        
        self.rate_limiter = RateLimitedFetcher(
            max_per_minute=60,
            concurrency=max_parallel
        )
        
        self.fetch_stats = {
            "products_success": 0,
            "products_failed": 0,
            "candles_success": 0,
            "candles_failed": 0,
            "total_wait_time": 0.0,
            "rate_limit_hits": 0
        }
        
        logger.debug(
            f"DataFetcher initialized | max_parallel={max_parallel} | "
            f"rate_limit=60/min | timeout={self.timeout}s"
        )

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
                
                result = data.get("result", {})
                if result and all(k in result for k in ("t", "o", "h", "l", "c", "v")):
                    num_candles = len(result.get("t", []))
                    logger.debug(
                        f"Candles fetch successful | Symbol: {symbol} | "
                        f"Resolution: {resolution} | Count: {num_candles}"
                    )
                else:
                    logger.warning(
                        f"Candles response missing required fields | Symbol: {symbol} | "
                        f"Resolution: {resolution}"
                    )
                    self.fetch_stats["candles_failed"] += 1
                    return None
            else:
                self.fetch_stats["candles_failed"] += 1
                logger.warning(
                    f"Candles fetch failed | Symbol: {symbol} | "
                    f"Resolution: {resolution} | Params: {params}"
                )
            
            return data
    
    def get_stats(self) -> Dict[str, Any]:
        stats = self.fetch_stats.copy()
        stats["rate_limiter"] = self.rate_limiter.get_stats()
        
        total_products = stats["products_success"] + stats["products_failed"]
        total_candles = stats["candles_success"] + stats["candles_failed"]
        
        if total_products > 0:
            stats["products_success_rate"] = round(
                stats["products_success"] / total_products * 100, 1
            )
        
        if total_candles > 0:
            stats["candles_success_rate"] = round(
                stats["candles_success"] / total_candles * 100, 1
            )
        
        return stats

    async def fetch_candles_batch(
        self,
        requests: List[Tuple[str, str, int]],
        reference_time: Optional[int] = None
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        if reference_time is None:
            reference_time = get_trigger_timestamp()
        
        logger.debug(
            f"Batch fetching {len(requests)} candle requests "
            f"(using concurrent individual fetches)"
        )
        
        tasks = []
        request_keys = []
        
        for symbol, resolution, limit in requests:
            task = self.fetch_candles(symbol, resolution, limit, reference_time)
            tasks.append(task)
            request_keys.append(f"{symbol}_{resolution}")
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        output = {}
        for key, result in zip(request_keys, results):
            if isinstance(result, Exception):
                logger.error(f"Batch fetch failed for {key}: {result}")
                output[key] = None
            else:
                output[key] = result
        
        success_count = sum(1 for v in output.values() if v is not None)
        logger.debug(
            f"Batch fetch complete | "
            f"Success: {success_count}/{len(requests)} | "
            f"Failed: {len(requests) - success_count}"
        )
        
        return output

    async def fetch_all_candles_truly_parallel(
        self,
        pair_requests: List[Tuple[str, List[Tuple[str, int]]]],
        reference_time: Optional[int] = None
    ) -> Dict[str, Dict[str, Optional[Dict[str, Any]]]]:
        """
        Fetch ALL candles for ALL pairs in ONE parallel batch.
        
        Args:
            pair_requests: [(symbol, [(resolution, limit), ...]), ...]
            
        Returns:
            {symbol: {resolution: candle_data}}
        """
        if reference_time is None:
            reference_time = get_trigger_timestamp()
        
        all_tasks = []
        task_metadata = []  # (symbol, resolution)
        
        # Build all tasks upfront
        for symbol, resolutions in pair_requests:
            for resolution, limit in resolutions:
                task = self.fetch_candles(symbol, resolution, limit, reference_time)
                all_tasks.append(task)
                task_metadata.append((symbol, resolution))
        
        logger.info(
            f"🚀 Parallel fetch: {len(all_tasks)} candle requests "
            f"for {len(pair_requests)} pairs"
        )
        
        # Fire ALL requests simultaneously
        results = await asyncio.gather(*all_tasks, return_exceptions=True)
        
        # Organize results by symbol and resolution
        output = {}
        for (symbol, resolution), result in zip(task_metadata, results):
            if symbol not in output:
                output[symbol] = {}
            
            if isinstance(result, Exception):
                logger.error(f"Fetch failed for {symbol} {resolution}: {result}")
                output[symbol][resolution] = None
            else:
                output[symbol][resolution] = result
        
        success_count = sum(
            1 for symbol_data in output.values() 
            for res_data in symbol_data.values() 
            if res_data is not None
        )
        
        logger.info(
            f"✅ Parallel fetch complete: {success_count}/{len(all_tasks)} successful"
        )
        
        return output     

def parse_candles_to_numpy(result: Optional[Dict[str, Any]]) -> Optional[Dict[str, np.ndarray]]:
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
            "open": np.empty(n, dtype=np.float32),
            "high": np.empty(n, dtype=np.float32),
            "low": np.empty(n, dtype=np.float32),
            "close": np.empty(n, dtype=np.float32),
            "volume": np.empty(n, dtype=np.float32),
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
        
    except Exception:
        return None

def validate_candle_data(data: Optional[Dict[str, np.ndarray]], required_len: int = 0) -> Tuple[bool, Optional[str]]:
    try:
        if data is None or not data:
            return False, "Data is None or empty"
        
        close = data.get("close")
        timestamp = data.get("timestamp")
        
        if close is None or len(close) == 0:
            return False, "Close array is empty"
        
        if np.any(np.isnan(close)) or np.any(close <= 0):
            return False, "Invalid close prices (NaN or <= 0)"
        
        if timestamp is None or len(timestamp) == 0:
            return False, "Timestamp array is empty"
        
        if not np.all(np.diff(timestamp) >= 0):  # allows equal timestamps (rare but safe)
            return False, "Timestamps not non-decreasing"
        
        if len(close) < required_len:
            return False, f"Insufficient data: {len(close)} < {required_len}"
        
        if len(close) >= 2:
            time_diffs = np.diff(timestamp)
            if len(time_diffs) > 0:
                median_diff = np.median(time_diffs)
                max_expected_gap = median_diff * Constants.MAX_CANDLE_GAP_MULTIPLIER
                if np.any(time_diffs > max_expected_gap):
                    gaps = time_diffs[time_diffs > max_expected_gap]
                    logger.warning(f"Detected {len(gaps)} candle gaps (median: {median_diff}s, max gap: {gaps.max()}s)")
        
        if len(close) >= 2:
            price_changes = np.abs(np.diff(close) / close[:-1]) * 100
            extreme_changes = price_changes[price_changes > Constants.MAX_PRICE_CHANGE_PERCENT]
            if len(extreme_changes) > 0:
                logger.warning(f"Detected {len(extreme_changes)} extreme price changes (max: {extreme_changes.max():.2f}%)")
                return False, f"Extreme price spike detected: {extreme_changes.max():.2f}%"
        
        return True, None
    except Exception as e:
        logger.error(f"Data validation failed: {e}")
        return False, f"Validation error: {str(e)}"

def get_last_closed_index_from_array(
    timestamps: np.ndarray,
    interval_minutes: int,
    reference_time: Optional[int] = None
) -> Optional[int]:
    if timestamps is None or timestamps.size < 2:
        return None

    if reference_time is None:
        reference_time = get_trigger_timestamp()

    last_ts = int(timestamps[-1])
    interval_seconds = interval_minutes * 60

    expected_last_closed = calculate_expected_candle_timestamp(reference_time, interval_minutes)

    diff_seconds = abs(last_ts - expected_last_closed)

    if diff_seconds > 300:
        logger.warning(
            f"Last candle timestamp significantly off! "
            f"Got {last_ts} ({format_ist_time(last_ts)}), "
            f"expected ~{expected_last_closed} ({format_ist_time(expected_last_closed)}), "
            f"diff={diff_seconds}s"
        )
    elif diff_seconds > 0:
        # Optional: very quiet debug for tiny drifts (1–300 seconds) — usually normal
        logger.debug(
            f"Minor acceptable timestamp drift: {last_ts} vs expected {expected_last_closed} "
            f"({diff_seconds}s)"
        )

    publication_buffer = Constants.CANDLE_PUBLICATION_LAG_SEC

    if reference_time >= (last_ts + interval_seconds + publication_buffer):
        return timestamps.size - 1
    else:
        return timestamps.size - 2 if timestamps.size >= 2 else None

def build_products_map_from_api_result(api_products: Optional[Dict[str, Any]]) -> Dict[str, dict]:
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
                        products_map[pair_name] = {"id": p.get("id"), "symbol": p.get("symbol"), "contract_type": p.get("contract_type")}
                        break
        except Exception:
            pass
    return products_map

# ============================================================================
# COMPLETE OPTIMIZED RedisStateStore CLASS
# ============================================================================

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
    
    # Class-level connection pool for reuse across runs
    _global_pool: ClassVar[Optional[redis.Redis]] = None
    _pool_healthy: ClassVar[bool] = False
    _pool_lock: ClassVar[asyncio.Lock] = asyncio.Lock()
    _pool_created_at: ClassVar[float] = 0.0
    _pool_reuse_count: ClassVar[int] = 0
    
    def __init__(self, redis_url: str):
        from urllib.parse import urlparse, parse_qs
        
        self.redis_url = redis_url
        self._redis: Optional[redis.Redis] = None
        
        try:
            parsed = urlparse(redis_url)
            
            if parsed.scheme not in ('redis', 'rediss'):
                raise ValueError(f"Invalid Redis URL scheme: {parsed.scheme}")
            
            self.redis_host = parsed.hostname or 'localhost'
            self.redis_port = parsed.port or 6379
            self.redis_db = parsed.path.lstrip('/') or '0'
            
            if cfg.DEBUG_MODE:
                logger.debug(
                    f"Redis URL parsed | Host: {self.redis_host} | "
                    f"Port: {self.redis_port} | DB: {self.redis_db} | "
                    f"Secure: {parsed.scheme == 'rediss'}"
                )
            
        except Exception as e:
            logger.error(f"Failed to parse Redis URL: {e}")
            self.redis_host = 'localhost'
            self.redis_port = 6379
            self.redis_db = '0'
        
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
                
                # Save to global pool for reuse
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
        # Try reusing global pool first (major optimization)
        async with RedisStateStore._pool_lock:
            if RedisStateStore._global_pool and RedisStateStore._pool_healthy:
                try:
                    # Quick health check (non-blocking)
                    await asyncio.wait_for(
                        RedisStateStore._global_pool.ping(), timeout=1.0
                    )
                    self._redis = RedisStateStore._global_pool
                    RedisStateStore._pool_reuse_count += 1
                    
                    pool_age = time.time() - RedisStateStore._pool_created_at
                    logger.info(
                        f"♻️  Reusing Redis pool (age: {pool_age:.1f}s, "
                        f"reuse count: {RedisStateStore._pool_reuse_count})"
                    )
                    
                    # Load Lua script if needed
                    if not self._dedup_script_sha:
                        try:
                            self._dedup_script_sha = await self._redis.script_load(
                                self.DEDUP_LUA
                            )
                        except Exception as e:
                            logger.warning(f"Lua script load failed: {e}")
                    
                    self.degraded = False
                    return
                    
                except Exception as e:
                    logger.debug(f"Pool health check failed: {e}, creating new pool")
                    RedisStateStore._pool_healthy = False

        # Fallback: check existing connection
        if self._redis is not None and not self.degraded:
            try:
                if await self._ping_with_retry(1.0):
                    logger.debug("Redis connection healthy")
                    return
            except Exception:
                logger.debug("Redis ping failed, attempting reconnect")

        # Create new connection
        for attempt in range(1, cfg.REDIS_CONNECTION_RETRIES + 1):
            self._connection_attempts = attempt
            if cfg.DEBUG_MODE:
                logger.debug(f"Redis connection attempt {attempt}/{cfg.REDIS_CONNECTION_RETRIES}")

            if await self._attempt_connect(timeout):
                # Smoke test
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
                    
                    # Check memory policy
                    info = await self._safe_redis_op(
                        self._redis.info("memory"), 3.0, "info_memory", lambda r: r
                    )
                    if info:
                        policy = info.get("maxmemory_policy", "unknown")
                        if policy in ("volatile-lru", "allkeys-lru") and cfg.DEBUG_MODE:
                            logger.debug(f"Redis using {policy} - keys may be evicted under memory pressure")
                    
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
        # Don't close global pool - let it persist
        if self._redis and self._redis != RedisStateStore._global_pool:
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

    async def _safe_redis_op(
        self,
        coro,
        timeout: float,
        op_name: str,
        parser: Optional[Callable] = None,
    ):
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
                on_error=lambda e, a, c: logger.debug(
                    f"Redis {op_name} error (attempt {a}): {e}"
                ) if cfg.DEBUG_MODE else None,
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
            self._redis.set(
                redis_key,
                data,
                ex=self.expiry_seconds if self.expiry_seconds > 0 else None,
            ),
            timeout,
            f"set {key}",
        )

    async def get_metadata(self, key: str, timeout: float = 2.0) -> Optional[str]:
        return await self._safe_redis_op(
            self._redis.get(f"{self.meta_prefix}{key}"),
            timeout,
            f"get_metadata {key}",
            lambda r: r if r else None,
        )

    async def set_metadata(
        self, key: str, value: str, timeout: float = 2.0
    ) -> None:
        await self._safe_redis_op(
            self._redis.set(
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

        window = ts // 900
        recent_key = f"recent_alert:{pair}:{alert_key}:{window}"

        if self._dedup_script_sha:
            try:
                result = await asyncio.wait_for(
                    self._redis.evalsha(
                        self._dedup_script_sha,
                        1,
                        recent_key,
                        str(Constants.ALERT_DEDUP_WINDOW_SEC)
                    ),
                    timeout=2.0
                )
                return bool(result)
            except Exception as e:
                if cfg.DEBUG_MODE:
                    logger.debug(f"Lua script failed, fallback to SET NX: {e}")

        result = await self._safe_redis_op(
            self._redis.set(
                recent_key, "1", nx=True, ex=Constants.ALERT_DEDUP_WINDOW_SEC
            ),
            timeout=2.0,
            op_name=f"check_recent_alert {pair}:{alert_key}",
            parser=lambda r: bool(r),
        )

        return result is True

    async def batch_check_recent_alerts(
        self, checks: List[Tuple[str, str, int]]
    ) -> Dict[str, bool]:
        if self.degraded or not checks:
            return {f"{pair}:{alert_key}": True for pair, alert_key, _ in checks}

        try:
            pipeline = self._redis.pipeline()
            keys_map = {}

            for pair, alert_key, ts in checks:
                window = ts // 900
                recent_key = f"recent_alert:{pair}:{alert_key}:{window}"
                keys_map[recent_key] = f"{pair}:{alert_key}"
                pipeline.set(
                    recent_key, "1", nx=True, ex=Constants.ALERT_DEDUP_WINDOW_SEC
                )

            results = await asyncio.wait_for(pipeline.execute(), timeout=3.0)

            output = {}
            for idx, (recent_key, composite_key) in enumerate(keys_map.items()):
                should_send = bool(results[idx]) if idx < len(results) else True
                output[composite_key] = should_send

            return output

        except Exception as e:
            logger.error(f"Batch check_recent_alerts failed: {e}")
            return {f"{pair}:{alert_key}": True for pair, alert_key, _ in checks}

    async def mget_states(
        self, keys: List[str], timeout: float = 2.0
    ) -> Dict[str, Optional[Dict[str, Any]]]:
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

    async def batch_set_states(
        self,
        updates: List[Tuple[str, Any, Optional[int]]],
        timeout: float = 4.0,
    ) -> None:
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

    async def atomic_batch_update(
        self,
        updates: List[Tuple[str, Any, Optional[int]]],
        deletes: List[str] = None
    ) -> bool:
        if self.degraded or not self._redis:
            return False
        
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
            
            if deletes:
                for key in deletes:
                    pipe.delete(f"{self.state_prefix}{key}")
            
            await asyncio.wait_for(pipe.execute(), timeout=4.0)
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


# ============================================================================
# PART 7: TELEGRAM QUEUE & MESSAGE FORMATTING
# ============================================================================

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
BUY_PIVOT_DEFS = [{"key": f"pivot_up_{level}", "title": f"🟢📷 Cross above {level}", "check_fn": lambda ctx, ppo, ppo_sig, rsi, level=level: (ctx["buy_common"] and (ctx["close_prev"] <= ctx["pivots"][level]) and (ctx["close_curr"] > ctx["pivots"][level])), "extra_fn": lambda ctx, ppo, ppo_sig, rsi, _, level=level: f"${ctx['pivots'][level]:,.2f} | MMH ({ctx['mmh_curr']:.2f})", "requires": ["pivots"]} for level in ["P", "S1", "S2", "S3", "R1", "R2"]]
SELL_PIVOT_DEFS = [{"key": f"pivot_down_{level}", "title": f"🔴📶 Cross below {level}", "check_fn": lambda ctx, ppo, ppo_sig, rsi, level=level: (ctx["sell_common"] and (ctx["close_prev"] >= ctx["pivots"][level]) and (ctx["close_curr"] < ctx["pivots"][level])), "extra_fn": lambda ctx, ppo, ppo_sig, rsi, _, level=level: f"${ctx['pivots'][level]:,.2f} | MMH ({ctx['mmh_curr']:.2f})", "requires": ["pivots"]} for level in ["P", "S1", "S2", "R1", "R2", "R3"]]
ALERT_DEFINITIONS.extend(BUY_PIVOT_DEFS)
ALERT_DEFINITIONS.extend(SELL_PIVOT_DEFS)
ALERT_DEFINITIONS_MAP = {d["key"]: d for d in ALERT_DEFINITIONS}

ALERT_KEYS: Dict[str, str] = {d["key"]: f"ALERT:{d['key'].upper()}" for d in ALERT_DEFINITIONS}
for level in PIVOT_LEVELS:
    ALERT_KEYS[f"pivot_up_{level}"] = f"ALERT:PIVOT_UP_{level}"
    ALERT_KEYS[f"pivot_down_{level}"] = f"ALERT:PIVOT_DOWN_{level}"

# ============================================================================
# ALERT KEY VALIDATION (Consistency Safety)
# ============================================================================

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
    
    logger.info(f"✅ Validated {len(ALERT_DEFINITIONS)} alert definitions ({len(ALERT_KEYS)} keys)")

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
    results = await sdb.mget_states(state_keys)
    
    output = {}
    for key in keys:
        state_key = f"{pair}:{key}"
        st = results.get(state_key)
        output[key] = st is not None and st.get("state") == "ACTIVE"
    
    return output

@njit(fastmath=True, cache=True)
def _vectorized_wick_check_buy(
    open_arr: np.ndarray, 
    high_arr: np.ndarray, 
    low_arr: np.ndarray, 
    close_arr: np.ndarray,
    min_wick_ratio: float
) -> np.ndarray:
    """
    Vectorized BUY wick validation.
    Returns boolean array: True = passed, False = rejected
    """
    n = len(close_arr)
    result = np.zeros(n, dtype=np.bool_)
    
    for i in range(n):
        o, h, l, c = open_arr[i], high_arr[i], low_arr[i], close_arr[i]
        
        # Must be green candle
        if c <= o:
            result[i] = False
            continue
        
        candle_range = h - l
        if candle_range < 1e-8:
            result[i] = False
            continue
        
        # Check upper wick
        upper_wick = h - c
        wick_ratio = upper_wick / candle_range
        
        result[i] = wick_ratio < min_wick_ratio
    
    return result

@njit(fastmath=True, cache=True)
def _vectorized_wick_check_sell(
    open_arr: np.ndarray, 
    high_arr: np.ndarray, 
    low_arr: np.ndarray, 
    close_arr: np.ndarray,
    min_wick_ratio: float
) -> np.ndarray:
    """
    Vectorized SELL wick validation.
    Returns boolean array: True = passed, False = rejected
    """
    n = len(close_arr)
    result = np.zeros(n, dtype=np.bool_)
    
    for i in range(n):
        o, h, l, c = open_arr[i], high_arr[i], low_arr[i], close_arr[i]
        
        # Must be red candle
        if c >= o:
            result[i] = False
            continue
        
        candle_range = h - l
        if candle_range < 1e-8:
            result[i] = False
            continue
        
        # Check lower wick
        lower_wick = c - l
        wick_ratio = lower_wick / candle_range
        
        result[i] = wick_ratio < min_wick_ratio
    
    return result


# Keep the old function for single-candle checks, but add fast path:
def check_common_conditions(
    open_val: float,
    high_val: float,
    low_val: float,
    close_val: float,
    is_buy: bool
) -> bool:
    """Fast path for single candle validation"""
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

# ============================================================================
# COMPLETE OPTIMIZED evaluate_pair_and_alert FUNCTION
# ============================================================================

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
    """
    Optimized evaluation function with:
    - Vectorized wick validation
    - Reduced logging overhead
    - Conditional dedup checks
    - Better memory management
    """
    
    logger_pair = logging.getLogger(f"macd_bot.{pair_name}.{correlation_id}")
    PAIR_ID.set(pair_name)
    pair_start_time = time.time()

    try:
        # ===== PHASE 1: Validate Closed Candles =====
        i15 = get_last_closed_index_from_array(data_15m["timestamp"], 15, reference_time)
        i5 = get_last_closed_index_from_array(data_5m["timestamp"], 5, reference_time)
        
        if i15 is None or i15 < 3 or i5 is None:
            logger_pair.warning(f"Insufficient closed candles for {pair_name}")
            return None
        
        # ===== PHASE 2: Calculate Indicators =====
        gc.disable()
        try:
            indicators = await asyncio.to_thread(
                calculate_all_indicators_numpy, data_15m, data_5m, data_daily
            )
        finally:
            gc.enable()
        
        # Extract indicators
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

        # ===== PHASE 3: Extract Current Candle Data =====
        close_15m = data_15m["close"]
        open_15m = data_15m["open"]
        high_15m = data_15m["high"]
        low_15m = data_15m["low"]
        timestamps_15m = data_15m["timestamp"]
        
        close_curr = float(close_15m[i15])
        close_prev = float(close_15m[i15 - 1])
        ts_curr = int(timestamps_15m[i15])
        
        open_curr = float(open_15m[i15])
        high_curr = float(high_15m[i15])
        low_curr = float(low_15m[i15])

        ppo_curr = float(ppo[i15])
        ppo_prev = float(ppo[i15 - 1])
        ppo_sig_curr = float(ppo_signal[i15])
        ppo_sig_prev = float(ppo_signal[i15 - 1])

        rsi_curr = float(smooth_rsi[i15])
        rsi_prev = float(smooth_rsi[i15 - 1])

        vwap_curr = float(vwap[i15]) if len(vwap) > 0 else 0.0
        vwap_prev = float(vwap[i15 - 1]) if len(vwap) > 0 else 0.0

        mmh_curr = float(mmh[i15])
        mmh_m1 = float(mmh[i15 - 1])

        cloud_up = bool(upw[i15]) and not bool(dnw[i15])
        cloud_down = bool(dnw[i15]) and not bool(upw[i15])

        rma50_15_val = float(rma50_15[i15])
        rma200_5_val = float(rma200_5[i5])

        # ===== PHASE 4: Base Trend Filters =====
        base_buy_common = rma50_15_val < close_curr and rma200_5_val < close_curr
        base_sell_common = rma50_15_val > close_curr and rma200_5_val > close_curr

        if base_buy_common:
            base_buy_common = base_buy_common and (mmh_curr > 0 and cloud_up)

        if base_sell_common:
            base_sell_common = base_sell_common and (mmh_curr < 0 and cloud_down)

        # ===== PHASE 5: Vectorized Candle Quality Check =====
        # Pre-compute candle quality for all candles (vectorized - MUCH faster)
        buy_quality_arr, sell_quality_arr = precompute_candle_quality(data_15m)
        
        buy_candle_passed = bool(buy_quality_arr[i15])
        sell_candle_passed = bool(sell_quality_arr[i15])
        
        # Get rejection reasons only if needed (lazy evaluation)
        buy_candle_reason = None
        sell_candle_reason = None
        
        if base_buy_common and not buy_candle_passed:
            _, buy_candle_reason = check_candle_quality_with_reason(
                open_curr, high_curr, low_curr, close_curr, is_buy=True
            )
        
        if base_sell_common and not sell_candle_passed:
            _, sell_candle_reason = check_candle_quality_with_reason(
                open_curr, high_curr, low_curr, close_curr, is_buy=False
            )

        # Final buy/sell conditions
        buy_common = base_buy_common and buy_candle_passed
        sell_common = base_sell_common and sell_candle_passed

        # ===== PHASE 6: MMH Reversal Detection =====
        mmh_reversal_buy = False
        mmh_reversal_sell = False

        if i15 >= 3:
            mmh_m3 = float(mmh[i15 - 3])
            mmh_m2 = float(mmh[i15 - 2])

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

        # ===== PHASE 7: Build Alert Context =====
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
            "candle_quality_failed_buy": base_buy_common and not buy_candle_passed,
            "candle_quality_failed_sell": base_sell_common and not sell_candle_passed,
            "candle_rejection_reason_buy": buy_candle_reason,
            "candle_rejection_reason_sell": sell_candle_reason,
        }

        ppo_ctx = {"curr": context["ppo_curr"], "prev": context["ppo_prev"]}
        ppo_sig_ctx = {"curr": context["ppo_sig_curr"], "prev": context["ppo_sig_prev"]}
        rsi_ctx = {"curr": context["rsi_curr"], "prev": context["rsi_prev"]}

        # ===== PHASE 8: Check Alert Conditions =====
        raw_alerts: List[Tuple[str, str, str]] = []

        # Filter alert definitions based on enabled features
        alert_keys_to_check = []
        for def_ in ALERT_DEFINITIONS:
            if "pivots" in def_["requires"] and not context.get("pivots"):
                continue
            if "vwap" in def_["requires"] and not context.get("vwap"):
                continue
            alert_keys_to_check.append(def_["key"])

        # Batch fetch previous states
        previous_states = await check_multiple_alert_states(
            sdb, pair_name, [ALERT_KEYS[k] for k in alert_keys_to_check]
        )

        states_to_update = []
        
        # Check each alert condition
        for alert_key in alert_keys_to_check:
            def_ = ALERT_DEFINITIONS_MAP.get(alert_key)
            if not def_:
                continue
    
            try:
                key = ALERT_KEYS[alert_key]
                if def_["check_fn"](context, ppo_ctx, ppo_sig_ctx, rsi_ctx):
                    if not previous_states.get(key, False):
                        extra = def_["extra_fn"](context, ppo_ctx, ppo_sig_ctx, rsi_ctx, None)
                        raw_alerts.append((def_["title"], extra, def_["key"]))
                        states_to_update.append((f"{pair_name}:{key}", "ACTIVE", None))
            except Exception as e:
                logger_pair.warning(f"Alert check failed for {pair_name}, key={def_['key']}: {e}")

        # Batch update active alerts
        if states_to_update:
            await sdb.batch_set_states(states_to_update)

        # ===== PHASE 9: Reset Alert States =====
        resets_to_apply = []

        # PPO signal crossovers
        if ppo_prev > ppo_sig_prev and ppo_curr <= ppo_sig_curr:
            resets_to_apply.append((f"{pair_name}:{ALERT_KEYS['ppo_signal_up']}", "INACTIVE", None))
        if ppo_prev < ppo_sig_prev and ppo_curr >= ppo_sig_curr:
            resets_to_apply.append((f"{pair_name}:{ALERT_KEYS['ppo_signal_down']}", "INACTIVE", None))

        # PPO zero crossovers
        if ppo_prev > 0 and ppo_curr <= 0:
            resets_to_apply.append((f"{pair_name}:{ALERT_KEYS['ppo_zero_up']}", "INACTIVE", None))
        if ppo_prev < 0 and ppo_curr >= 0:
            resets_to_apply.append((f"{pair_name}:{ALERT_KEYS['ppo_zero_down']}", "INACTIVE", None))

        # PPO 0.11 thresholds
        if ppo_prev > Constants.PPO_011_THRESHOLD and ppo_curr <= Constants.PPO_011_THRESHOLD:
            resets_to_apply.append((f"{pair_name}:{ALERT_KEYS['ppo_011_up']}", "INACTIVE", None))
        if ppo_prev < Constants.PPO_011_THRESHOLD_SELL and ppo_curr >= Constants.PPO_011_THRESHOLD_SELL:
            resets_to_apply.append((f"{pair_name}:{ALERT_KEYS['ppo_011_down']}", "INACTIVE", None))

        # RSI crossovers
        if rsi_prev > Constants.RSI_THRESHOLD and rsi_curr <= Constants.RSI_THRESHOLD:
            resets_to_apply.append((f"{pair_name}:{ALERT_KEYS['rsi_50_up']}", "INACTIVE", None))
        if rsi_prev < Constants.RSI_THRESHOLD and rsi_curr >= Constants.RSI_THRESHOLD:
            resets_to_apply.append((f"{pair_name}:{ALERT_KEYS['rsi_50_down']}", "INACTIVE", None))

        # VWAP crossovers
        if context["vwap"]:
            if close_prev > vwap_prev and close_curr <= vwap_curr:
                resets_to_apply.append((f"{pair_name}:{ALERT_KEYS['vwap_up']}", "INACTIVE", None))
            if close_prev < vwap_prev and close_curr >= vwap_curr:
                resets_to_apply.append((f"{pair_name}:{ALERT_KEYS['vwap_down']}", "INACTIVE", None))

        # Pivot level crossovers
        if piv:
            for level_name, level_value in piv.items():
                if close_prev > level_value and close_curr <= level_value:
                    resets_to_apply.append((f"{pair_name}:{ALERT_KEYS[f'pivot_up_{level_name}']}", "INACTIVE", None))
                if close_prev < level_value and close_curr >= level_value:
                    resets_to_apply.append((f"{pair_name}:{ALERT_KEYS[f'pivot_down_{level_name}']}", "INACTIVE", None))

        # MMH reversals
        if (mmh_curr > 0) and (mmh_curr <= mmh_m1):
            if await was_alert_active(sdb, pair_name, ALERT_KEYS["mmh_buy"]):
                resets_to_apply.append((f"{pair_name}:{ALERT_KEYS['mmh_buy']}", "INACTIVE", None))
        if (mmh_curr < 0) and (mmh_curr >= mmh_m1):
            if await was_alert_active(sdb, pair_name, ALERT_KEYS["mmh_sell"]):
                resets_to_apply.append((f"{pair_name}:{ALERT_KEYS['mmh_sell']}", "INACTIVE", None))

        # Batch update all resets
        if resets_to_apply:
            await sdb.batch_set_states(resets_to_apply)

        # ===== PHASE 10: Deduplication (Optimized) =====
        alerts_to_send = []
        
        if not raw_alerts:
            # No alerts - skip all dedup logic
            pass
        elif sdb.degraded:
            # Degraded mode: send all alerts without dedup
            if cfg.DEBUG_MODE:
                logger_pair.debug(f"Redis degraded, skipping dedup for {len(raw_alerts)} alerts")
            alerts_to_send = raw_alerts[:cfg.MAX_ALERTS_PER_PAIR]
        else:
            # Normal mode: check dedup in batch
            dedup_checks = [(pair_name, alert_key, ts_curr) for _, _, alert_key in raw_alerts]
            dedup_results = await sdb.batch_check_recent_alerts(dedup_checks)

            for title, extra, alert_key in raw_alerts:
                composite_key = f"{pair_name}:{alert_key}"
                if dedup_results.get(composite_key, True):
                    alerts_to_send.append((title, extra, alert_key))
                else:
                    if cfg.DEBUG_MODE:
                        logger_pair.debug(f"Skipping duplicate alert: {composite_key}")
            
            alerts_to_send = alerts_to_send[:cfg.MAX_ALERTS_PER_PAIR]

        # ===== PHASE 11: Send Alerts =====
        if alerts_to_send:
            if len(alerts_to_send) == 1:
                title, extra, _ = alerts_to_send[0]
                msg = build_single_msg(title, pair_name, close_curr, ts_curr, extra)
            else:
                items = [(title, extra) for title, extra, _ in alerts_to_send[:25]]
                msg = build_batched_msg(pair_name, close_curr, ts_curr, items)

            if not cfg.DRY_RUN_MODE:
                await telegram_queue.send(msg)

            new_state = {
                "state": "ALERT_SENT",
                "ts": int(time.time()),
                "summary": {"alerts": len(alerts_to_send)}
            }
            logger_pair.info(
                f"✅ Sent {len(alerts_to_send)} alerts for {pair_name}: "
                f"{[ak for _, _, ak in alerts_to_send]}"
            )
        else:
            new_state = {"state": "NO_SIGNAL", "ts": int(time.time())}

        # ===== PHASE 12: Build Final State Summary =====
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

        alerts_count = new_state.get("summary", {}).get("alerts", 0)

        new_state["summary"] = {
            "alerts": alerts_count,
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

        # ===== PHASE 13: Conditional Logging (Reduced Overhead) =====
        is_important = (
            alerts_to_send or 
            context.get("candle_quality_failed_buy") or 
            context.get("candle_quality_failed_sell")
        )
        
        status_msg = f"✔ {pair_name} | cloud={cloud} mmh={mmh_curr:.2f}"
        
        if alerts_to_send:
            status_msg += f" | 🔔 {len(alerts_to_send)} alerts sent"
            logger_pair.info(status_msg)
        elif base_buy_common and not buy_candle_passed:
            status_msg += f" | BUY blocked: {buy_candle_reason}"
            logger_pair.info(status_msg)
        elif base_sell_common and not sell_candle_passed:
            status_msg += f" | SELL blocked: {sell_candle_reason}"
            logger_pair.info(status_msg)
        else:
            # No signals - only log in debug mode
            if cfg.DEBUG_MODE:
                logger_pair.debug(status_msg + " | No signals")

        if cfg.DEBUG_MODE:
            logger_pair.debug(f"Pair total: {time.time() - pair_start_time:.2f}s")

        return pair_name, new_state

    except Exception as e:
        logger_pair.exception(f"❌ Error in evaluate_pair_and_alert for {pair_name}: {e}")
        return None

    finally:
        # Cleanup - no DataFrame references, just arrays
        try:
            del data_15m, data_5m, data_daily
            if 'ppo' in locals():
                del ppo, ppo_signal, smooth_rsi, vwap, mmh
            if cfg.CIRRUS_CLOUD_ENABLED and 'upw' in locals():
                del upw, dnw
        except Exception as e:
            if cfg.DEBUG_MODE:
                logger_pair.warning(f"Cleanup error (non-critical): {e}")
        finally:
            PAIR_ID.set("")     
       
# ============================================================================
# PART 9: WORKER POOL & PAIR PROCESSING
# ============================================================================

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

        requests = [
            (symbol, "15", 300),
            (symbol, "5", 400),
        ]
        
        if cfg.ENABLE_PIVOT:
            requests.append((symbol, "D", daily_limit))
        
        batch_results = await fetcher.fetch_candles_batch(requests, reference_time)
        
        data_15m = parse_candles_to_numpy(batch_results.get(f"{symbol}_15"))
        data_5m = parse_candles_to_numpy(batch_results.get(f"{symbol}_5"))
        data_daily = parse_candles_to_numpy(batch_results.get(f"{symbol}_D")) if cfg.ENABLE_PIVOT else None

        valid_15m, reason_15m = validate_candle_data(data_15m, 220)
        valid_5m, reason_5m = validate_candle_data(data_5m, 280)
    
        if not valid_15m or not valid_5m:
            logger.warning(f"Insufficient data for {pair_name}: 15m={reason_15m}, 5m={reason_5m}")
            return None
        
        return await evaluate_pair_and_alert(
            pair_name, data_15m, data_5m, data_daily,
            state_db, telegram_queue, correlation_id, reference_time
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

            # Efficient debug log: no lambda, no string formatting unless needed
            if cfg.DEBUG_MODE and logger_worker.isEnabledFor(logging.DEBUG):
                logger_worker.debug(f"Worker processing {pair_name}")

            try:
                result = await check_pair(
                    pair_name, fetcher, products_map, state_db,
                    telegram_queue, correlation_id, reference_time
                )

                if result:
                    async with results_lock:
                        results.append(result)

                    # Efficient debug log for completion
                    if cfg.DEBUG_MODE and logger_worker.isEnabledFor(logging.DEBUG):
                        summary = result[1].get('summary', {})
                        logger_worker.debug(
                            f"Worker completed {result[0]} | "
                            f"cloud={summary.get('cloud', 'n/a')} | "
                            f"mmh_hist={summary.get('mmh_hist', 'n/a')}"
                        )
                else:
                    if cfg.DEBUG_MODE and logger_worker.isEnabledFor(logging.DEBUG):
                        logger_worker.debug(f"Worker: {pair_name} returned None")

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
    """
    Optimized: Fetch ALL candles first, then evaluate in parallel.
    """
    logger_main = logging.getLogger("macd_bot.worker_pool")
    
    # PHASE 1: Fetch all candles in one parallel batch
    logger_main.info(f"📡 Phase 1: Fetching candles for {len(pairs_to_process)} pairs...")
    fetch_start = time.time()
    
    daily_limit = cfg.PIVOT_LOOKBACK_PERIOD + 10 if cfg.ENABLE_PIVOT else 0
    
    pair_requests = []
    for pair_name in pairs_to_process:
        product_info = products_map.get(pair_name)
        if not product_info:
            continue
        
        symbol = product_info["symbol"]
        resolutions = [("15", 300), ("5", 400)]
        if cfg.ENABLE_PIVOT:
            resolutions.append(("D", daily_limit))
        
        pair_requests.append((symbol, resolutions))
    
    all_candles = await fetcher.fetch_all_candles_truly_parallel(
        pair_requests, reference_time
    )
    
    fetch_duration = time.time() - fetch_start
    logger_main.info(f"✅ Phase 1 complete in {fetch_duration:.2f}s")
    
    # PHASE 2: Parse and validate candles
    logger_main.info("🔍 Phase 2: Parsing candle data...")
    parse_start = time.time()
    
    valid_pairs_data = {}
    for pair_name in pairs_to_process:
        product_info = products_map.get(pair_name)
        if not product_info:
            continue
        
        symbol = product_info["symbol"]
        candles = all_candles.get(symbol, {})
        
        data_15m = parse_candles_to_numpy(candles.get("15"))
        data_5m = parse_candles_to_numpy(candles.get("5"))
        data_daily = parse_candles_to_numpy(candles.get("D")) if cfg.ENABLE_PIVOT else None
        
        valid_15m, reason_15m = validate_candle_data(data_15m, 220)
        valid_5m, reason_5m = validate_candle_data(data_5m, 280)
        
        if not valid_15m or not valid_5m:
            logger_main.warning(
                f"Invalid data for {pair_name}: 15m={reason_15m}, 5m={reason_5m}"
            )
            continue
        
        valid_pairs_data[pair_name] = {
            "data_15m": data_15m,
            "data_5m": data_5m,
            "data_daily": data_daily
        }
    
    parse_duration = time.time() - parse_start
    logger_main.info(
        f"✅ Phase 2 complete in {parse_duration:.2f}s | "
        f"Valid pairs: {len(valid_pairs_data)}/{len(pairs_to_process)}"
    )
    
    # PHASE 3: Evaluate pairs in parallel
    logger_main.info(f"⚙️  Phase 3: Evaluating {len(valid_pairs_data)} pairs...")
    eval_start = time.time()
    
    eval_tasks = []
    for pair_name, pair_data in valid_pairs_data.items():
        task = evaluate_pair_and_alert(
            pair_name,
            pair_data["data_15m"],
            pair_data["data_5m"],
            pair_data["data_daily"],
            state_db,
            telegram_queue,
            correlation_id,
            reference_time
        )
        eval_tasks.append(task)
    
    results = await asyncio.gather(*eval_tasks, return_exceptions=True)
    
    # Filter out exceptions
    valid_results = []
    for result in results:
        if isinstance(result, Exception):
            logger_main.error(f"Evaluation error: {result}")
        elif result is not None:
            valid_results.append(result)
    
    eval_duration = time.time() - eval_start
    logger_main.info(f"✅ Phase 3 complete in {eval_duration:.2f}s")
    
    total_duration = fetch_duration + parse_duration + eval_duration
    logger_main.info(
        f"🎯 Total processing: {total_duration:.2f}s | "
        f"Fetch: {fetch_duration:.1f}s | Parse: {parse_duration:.1f}s | "
        f"Eval: {eval_duration:.1f}s"
    )
    
    return valid_results

# ============================================================================
# PART 10: MAIN RUN LOOP & ENTRY POINT
# ============================================================================

async def run_once() -> bool:
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

    PRODUCTS_CACHE = getattr(run_once, '_products_cache', {"data": None, "until": 0.0})
    now = time.time()

    if PRODUCTS_CACHE["data"] is None or now > PRODUCTS_CACHE["until"]:
        logger_run.info("📡 Fetching fresh products list from Delta API...")

        # Create temporary fetcher just for products
        temp_fetcher = DataFetcher(cfg.DELTA_API_BASE)
        prod_resp = await temp_fetcher.fetch_products()

        if not prod_resp:
            logger_run.error("❌ Failed to fetch products map - aborting run")
            return False

        PRODUCTS_CACHE["data"] = prod_resp
        PRODUCTS_CACHE["until"] = now + 28_800  # 8 hours
        run_once._products_cache = PRODUCTS_CACHE
        logger_run.info("✅ Products list cached for 8 hours")
    else:
        logger_run.debug(f"♻️  Using cached products (TTL: {PRODUCTS_CACHE['until'] - now:.0f}s)")
        prod_resp = PRODUCTS_CACHE["data"]

    products_map = build_products_map_from_api_result(prod_resp)
    pairs_to_process = [p for p in cfg.PAIRS if p in products_map]

    if len(pairs_to_process) < len(cfg.PAIRS):
        missing = set(cfg.PAIRS) - set(pairs_to_process)
        logger_run.warning(f"⚠️ Missing products for pairs: {missing}")     

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

        logger_run.debug("Connecting to Redis...")
        sdb = RedisStateStore(cfg.REDIS_URL)
        await sdb.connect()
        logger_run.debug("✅ Redis connection established")

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

        fetcher = DataFetcher(cfg.DELTA_API_BASE)
        if telegram_queue is None:
            telegram_queue = TelegramQueue(cfg.TELEGRAM_BOT_TOKEN, cfg.TELEGRAM_CHAT_ID)

        lock = RedisLock(sdb._redis, "macd_bot_run")
        lock_acquired = await lock.acquire(timeout=5.0)
        
        if not lock_acquired:
            logger_run.warning(
                "⏸️ Another instance is running (Redis lock held) - exiting gracefully"
            )
            return False

        logger_run.info("🔒 Distributed lock acquired successfully")

        if cfg.SEND_TEST_MESSAGE:
            await telegram_queue.send(escape_markdown_v2(
                f"🚀 {cfg.BOT_NAME} - Run Started\n"
                f"Date: {format_ist_time(datetime.now(timezone.utc))}\n"
                f"Correlation ID: {correlation_id}\n"
                f"Pairs: {len(cfg.PAIRS)}"
            ))

        all_results = await process_pairs_with_workers(
            fetcher, products_map, pairs_to_process,
            sdb, telegram_queue, correlation_id,
            lock, reference_time
        )

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
        
        if lock_acquired and lock and lock.acquired_by_me:
            try:
                await lock.release(timeout=3.0)
                logger_run.debug("🔓 Redis lock released")
            except Exception as e:
                logger_run.error(f"Error releasing lock: {e}")
        
        if sdb:
            try:
                await sdb.close()
                logger_run.debug("✅ Redis connection closed")
            except Exception as e:
                logger_run.error(f"Error closing Redis: {e}")
        
        try:
            await SessionManager.close_session()
            logger_run.debug("✅ HTTP session closed")
        except Exception as e:
            logger_run.error(f"Error closing HTTP session: {e}")
        
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

# ============================================================================
# ENTRY POINT
# ============================================================================

try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    logger.info(f"✅ uvloop enabled | {JSON_BACKEND} enabled")
except ImportError:
    logger.info(f"ℹ️ uvloop not available (using default) | {JSON_BACKEND} enabled")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="macd_unified",
        description="Unified MACD/alerts runner with NumPy optimization"
    )
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging")
    parser.add_argument("--validate-only", action="store_true", help="Validate config and exit")
    parser.add_argument("--skip-warmup", action="store_true", help="Skip Numba JIT warmup")
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
        for h in logger.handlers:
            h.setLevel(logging.DEBUG)
        logger.info("Debug mode enabled via CLI flag")

    try:
        validate_runtime_config()
    except ValueError as e:
        logger.critical(f"Configuration validation failed: {e}")
        sys.exit(1)
    
    if args.validate_only:
        logger.info("Configuration validation passed - exiting (--validate-only mode)")
        sys.exit(0)

    if not args.skip_warmup:
        warmup_numba()
    else:
        logger.info("Skipping Numba warmup (--skip-warmup flag)")

    try:
        success = asyncio.run(run_once())
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