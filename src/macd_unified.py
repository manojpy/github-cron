from __future__ import annotations   
import logging
import aot_bridge
import os
import sys
import time
import asyncio
import random
from pathlib import Path
import ssl
import signal
import re
import uuid
import argparse
import psutil
import math
import gc
import json
from collections import deque, defaultdict
from typing import Dict, Any, Optional, Tuple, List, ClassVar, TypedDict, Callable, Set, Deque
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from contextvars import ContextVar
from urllib.parse import urlparse, parse_qs
import aiohttp
from aiohttp import web
import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator
from aiohttp import ClientConnectorError, ClientResponseError, TCPConnector, ClientError
from numba import njit, prange
import contextlib 
import traceback
from redis_state_store import RedisStateStore 

from aot_bridge import (
    sanitize_array_numba,
    sanitize_array_numba_parallel,
    ema_loop,
    ema_loop_alpha,
    kalman_loop,
    vwap_daily_loop,
    rng_filter_loop,
    smooth_range,
    calculate_trends_with_state, 
    calc_mmh_worm_loop,
    calc_mmh_value_loop,
    calc_mmh_momentum_loop,
    rolling_std,
    calc_mmh_momentum_smoothing,
    rolling_mean_numba,
    rolling_min_max_numba,
    calculate_ppo_core,
    calculate_rsi_core,
    calculate_atr_rma, 
    vectorized_wick_check_buy,
    vectorized_wick_check_sell
)

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

__version__ = "1.8.0-stable"

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
    CIRCUIT_BREAKER_MAX_WAIT = 300
    MAX_PRICE_CHANGE_PERCENT = 50.0
    MAX_CANDLE_GAP_MULTIPLIER = 2.0
    INFINITY_CLAMP = 1e8
    LOCK_EXTEND_INTERVAL = 540
    LOCK_EXTEND_JITTER_MAX = 120
    ALERT_DEDUP_WINDOW_SEC = int(os.getenv("ALERT_DEDUP_WINDOW_SEC", 600))
    CANDLE_PUBLICATION_LAG_SEC = int(os.getenv("CANDLE_PUBLICATION_LAG_SEC", 45))
    TELEGRAM_MAX_MESSAGE_LENGTH = 4096
    TELEGRAM_MESSAGE_PREVIEW_LENGTH = 50
    VWAP_MAX_DISTANCE_PCT = 1.0
    PIVOT_MAX_DISTANCE_PCT = 0.5
    INTER_BATCH_DELAY: float = 0.5
    MIN_CANDLES_FOR_INDICATORS = 250
    CANDLE_SAFETY_BUFFER = 100
    CANDLE_PUBLICATION_LAG_SEC = 20

PIVOT_LEVELS = ["P", "S1", "S2", "S3", "R1", "R2", "R3"]

class CompiledPatterns:
    VALID_SYMBOL = re.compile(r'^[A-Z0-9_]+$')
    ESCAPE_MARKDOWN = re.compile(r'[_*\[\]()~`>#+-=|{}.!]')
    SECRET_TOKEN = re.compile(r'\b\d{6,}:[A-Za-z0-9_-]{20,}\b')   # still useful for Telegram bot tokens
    CHAT_ID = re.compile(r'chat_id=\d+')
    GITHUB_TOKEN = re.compile(r'(ghp_[A-Za-z0-9]{36,})')          # mask GitHub PATs like ghp_xxx...

TRACE_ID: ContextVar[str] = ContextVar("trace_id", default="")
PAIR_ID: ContextVar[str] = ContextVar("pair_id", default="")


class BotConfig(BaseModel):
    TELEGRAM_BOT_TOKEN: str = Field(..., min_length=1)
    TELEGRAM_CHAT_ID: str = Field(..., min_length=1)
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
    SRSI_EMA_LEN: int = 5
    ATR_SHORT: int = 5
    ATR_LONG: int = 14
    LOG_FILE: str = "macd_bot.log"
    MAX_PARALLEL_FETCH: int = Field(12, ge=1, le=20)
    HTTP_TIMEOUT: int = 6
    CANDLE_FETCH_RETRIES: int = 1
    CANDLE_FETCH_BACKOFF: float = 1
    JITTER_MIN: float = 0.1
    JITTER_MAX: float = 0.8
    RUN_TIMEOUT_SECONDS: int = 600
    BATCH_SIZE: int = 4
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
    FAIL_ON_TELEGRAM_DOWN: bool = False
    TELEGRAM_RATE_LIMIT_PER_MINUTE: int = 20
    TELEGRAM_BURST_SIZE: int = 5
    INDICATOR_THREAD_LIMIT: int = 3
    MACD_BOT_TOKEN: Optional[str] = Field(default=None, description="GitHub PAT for state storage")
    MACD_REPO: Optional[str] = Field(default=None, description="GitHub repo (owner/repo)")
    MACD_BRANCH: str = Field(default="main", description="Branch for state files")
    DRY_RUN_MODE: bool = Field(default=False, description="Dry-run: log alerts without sending")
    MIN_RUN_TIMEOUT: int = Field(default=300, ge=300, le=1800, description="Min/max run timeout bounds")
    MAX_ALERTS_PER_PAIR: int = Field(default=8, ge=5, le=15, description="Max alerts per pair per run")
    NUMBA_PARALLEL: bool = Field(default=True, description="Enable Numba parallel execution")
    SKIP_WARMUP: bool = Field(default=False, description="Skip Numba warmup (faster startup)")
    PRODUCTS_CACHE_TTL: int = Field(default=28800, description="Products cache TTL in seconds (8 hours)")
    PIVOT_MAX_DISTANCE_PCT: float = Field(default=1.5, description="Max distance from pivot to trigger alert")

    RVOL_THRESHOLD: float = Field(
        default=1.0, 
        ge=0.5, 
        le=2.0,
        description="Volatility expansion threshold (Pine: atr1 > atr3 * rvolThreshold). "
                    "1.0 = baseline, 1.1 = require 10% expansion, 1.5 = require 50% expansion"
    )
    ENABLE_RVOL_ALERT: bool = Field(
        default=True,
        description="Enable volatility expansion check (RVOL alert). "
                    "When False, wick patterns are evaluated without volatility requirement"
    )
    MAX_CANDLE_STALENESS_SEC: int = Field(
        default=1200,
        ge=600,
        le=3600,
        description="Maximum allowed candle staleness in seconds (10-60 minutes)"
    )
    RATE_LIMIT_PER_MINUTE: int = Field(
        default=60,
        ge=10,
        le=120,
        description="Max HTTP requests per minute to Delta API"
    )
    CB_FAILURE_THRESHOLD: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Consecutive failures before circuit breaker opens"
    )
    CB_RECOVERY_TIMEOUT: int = Field(
        default=60,
        ge=10,
        le=600,
        description="Seconds to wait before half-open probe"
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

    @field_validator('PIVOT_LOOKBACK_PERIOD')
    def validate_pivot_lookback(cls, v: int) -> int:
        if v < 5:
            raise ValueError(
                f'PIVOT_LOOKBACK_PERIOD must be >= 5 (need minimum historical data), got {v}'
            )
        if v > 365:
            raise ValueError(
                f'PIVOT_LOOKBACK_PERIOD > 365 days is excessive, got {v}'
            )
        return v

    @field_validator('DELTA_API_BASE')
    def validate_api_base(cls, v: str) -> str:
        if not re.match(r'^(https?://)[A-Za-z0-9\.\-:_/]+$', v.strip()):
            raise ValueError('DELTA_API_BASE must be a valid http(s) URL')
        return v.strip().rstrip('/')

    @model_validator(mode='after')
    def validate_logic(self) -> 'BotConfig':
        
        errors = []
        warnings = []

        if self.PPO_FAST >= self.PPO_SLOW:
            errors.append('PPO_FAST must be less than PPO_SLOW')

        if self.RUN_TIMEOUT_SECONDS < self.MIN_RUN_TIMEOUT:
            errors.append(
                f'RUN_TIMEOUT_SECONDS ({self.RUN_TIMEOUT_SECONDS}s) must be >= '
                f'MIN_RUN_TIMEOUT ({self.MIN_RUN_TIMEOUT}s)'
            )

        if self.ENABLE_PIVOT and self.PIVOT_MAX_DISTANCE_PCT < 1.0:
            errors.append('PIVOT_MAX_DISTANCE_PCT should be >= 1.0 for meaningful alerts')

        ranges = {
            'RMA_50_PERIOD': (self.RMA_50_PERIOD, 20, 100),
            'RMA_200_PERIOD': (self.RMA_200_PERIOD, 100, 300),
            'SRSI_RSI_LEN': (self.SRSI_RSI_LEN, 5, 50),
            'SRSI_KALMAN_LEN': (self.SRSI_KALMAN_LEN, 2, 20),
        }
        
        for name, (val, min_v, max_v) in ranges.items():
            if not (min_v <= val <= max_v):
                errors.append(f'{name} must be {min_v}-{max_v}, got {val}')

        if self.MAX_ALERTS_PER_PAIR > 15:
            warnings.append(
                f'MAX_ALERTS_PER_PAIR={self.MAX_ALERTS_PER_PAIR} is very high, may cause spam'
            )

        if self.MAX_PARALLEL_FETCH < 1 or self.MAX_PARALLEL_FETCH > 20:
            warnings.append(
                f'MAX_PARALLEL_FETCH={self.MAX_PARALLEL_FETCH} is outside recommended range (1-20)'
            )

        if self.HTTP_TIMEOUT < 5 or self.HTTP_TIMEOUT > 60:
            warnings.append(
                f'HTTP_TIMEOUT={self.HTTP_TIMEOUT}s is outside recommended range (5-60s)'
            )

        if len(self.PAIRS) > 20:
            warnings.append(
                f'Large number of pairs ({len(self.PAIRS)}) may exceed timeout limits'
            )

        if self.MEMORY_LIMIT_BYTES < 200_000_000:
            warnings.append(
                f'MEMORY_LIMIT_BYTES={self.MEMORY_LIMIT_BYTES} is very low '
                f'(minimum recommended: 200MB)'
            )

        if self.RVOL_THRESHOLD < 0.5 or self.RVOL_THRESHOLD > 2.0:
            errors.append(f'RVOL_THRESHOLD {self.RVOL_THRESHOLD} outside range [0.5, 2.0]')

        if self.MAX_CANDLE_STALENESS_SEC < 300:
            warnings.append(f'MAX_CANDLE_STALENESS_SEC very low ({self.MAX_CANDLE_STALENESS_SEC}s)')

        if errors:
            error_msg = 'Configuration validation failed:\n  ' + '\n  '.join(errors)
            raise ValueError(error_msg)

        self._validation_warnings = warnings

        return self

def load_config() -> BotConfig:
    config_file = os.getenv("CONFIG_FILE", "config_macd.json")
    data: Dict[str, Any] = {}
    if Path(config_file).exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                data = json_loads(f.read())

        except Exception as exc:
            error_msg = f"❌ ERROR: Config file {config_file} is not valid JSON: {exc}"
            print(error_msg, file=sys.stderr)
            sys.exit(1)
        
    else:
        print(f"⚠️ WARNING: Config file {config_file} not found, using environment variables only", file=sys.stderr)

    for key in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "MACD_BOT_TOKEN", "MACD_REPO", "MACD_BRANCH", "DELTA_API_BASE"):
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
            if any(x in msg for x in ("TOKEN", "redis://", "chat_id")):
                msg = re.sub(r'\b\d{6,}:[A-Za-z0-9_-]{20,}\b', '[REDACTED_TELEGRAM_TOKEN]', msg)
                msg = re.sub(r'chat_id=\d+', '[REDACTED_CHAT_ID]', msg)
                msg = CompiledPatterns.GITHUB_TOKEN.sub("[REDACTED_GITHUB_PAT]", msg)
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
        if record.msg:
            msg_str = str(record.msg)
            msg_str = CompiledPatterns.SECRET_TOKEN.sub("[REDACTED_TOKEN]", msg_str)
            msg_str = CompiledPatterns.CHAT_ID.sub("chat_id=[REDACTED]", msg_str)
            msg_str = CompiledPatterns.GITHUB_TOKEN.sub("[REDACTED_GITHUB_PAT]", msg_str)
            record.msg = msg_str  # optional, could skip reassigning

        if record.args:
            if isinstance(record.args, dict):
                record.args = {k: self._mask_secret(v) for k, v in record.args.items()}
            elif isinstance(record.args, tuple):
                record.args = tuple(self._mask_secret(v) for v in record.args)

        formatted = super().format(record)

        formatted = CompiledPatterns.SECRET_TOKEN.sub("[REDACTED_TOKEN]", formatted)
        formatted = CompiledPatterns.CHAT_ID.sub("chat_id=[REDACTED]", formatted)
        formatted = CompiledPatterns.GITHUB_TOKEN.sub("[REDACTED_GITHUB_PAT]", formatted)

        return formatted

    @staticmethod
    def _mask_secret(value: Any) -> Any:
        if value is None:
            return value
        value_str = str(value)
        masked = CompiledPatterns.SECRET_TOKEN.sub("[REDACTED_TELEGRAM_TOKEN]", value_str)
        masked = CompiledPatterns.GITHUB_TOKEN.sub("[REDACTED_GITHUB_PAT]", masked)
        masked = CompiledPatterns.CHAT_ID.sub("chat_id=[REDACTED]", masked)
        return masked

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

_IST_TZ = ZoneInfo("Asia/Kolkata")

def format_ist_time(dt_or_ts: Any = None, fmt: str = "%Y-%m-%d %H:%M:%S IST") -> str:
    try:
        if dt_or_ts is None:
            dt = datetime.now(timezone.utc)

        elif isinstance(dt_or_ts, datetime):
            dt = dt_or_ts
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        else:
            try:
                ts = float(dt_or_ts)
                if ts > 1_000_000_000_000:
                    ts /= 1000
                dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            except (ValueError, TypeError):
                dt = datetime.fromisoformat(str(dt_or_ts))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(_IST_TZ).strftime(fmt)
    except Exception as e:
        if cfg.DEBUG_MODE:
            logger.debug(f"format_ist_time parsing failed for '{dt_or_ts}'")
        return str(dt_or_ts)

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
        return
    
    errors = []
    warnings = []
    if hasattr(cfg, '_validation_warnings'):
        warnings.extend(cfg._validation_warnings)
    
    if not cfg.MACD_BOT_TOKEN or cfg.MACD_BOT_TOKEN.startswith("__SET_IN_"):
        errors.append("Missing or invalid MACD_BOT_TOKEN")

    if not cfg.MACD_REPO or "/" not in cfg.MACD_REPO:
        errors.append("MACD_REPO must be in format 'owner/repo'")

    if not cfg.MACD_BRANCH:
        errors.append("MACD_BRANCH cannot be empty")

    if not CompiledPatterns.SECRET_TOKEN.match(cfg.TELEGRAM_BOT_TOKEN):
        errors.append("TELEGRAM_BOT_TOKEN format invalid (should be: 123456:ABC-DEF...)")
    
    if not cfg.DELTA_API_BASE.startswith(('http://', 'https://')):
        errors.append("DELTA_API_BASE must start with http:// or https://")
    
    if not cfg.PAIRS or len(cfg.PAIRS) == 0:
        errors.append("PAIRS list is empty - no trading pairs configured")
    
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
        f"📡 Bot v{__version__} | Pairs: {len(cfg.PAIRS)} | Workers: {cfg.MAX_PARALLEL_FETCH} | "
        f"Timeout: {cfg.RUN_TIMEOUT_SECONDS}s | State Store: GitHub JSON"
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

    current_interval_open = (reference_time // interval_seconds) * interval_seconds

    last_closed_candle_open = current_interval_open - interval_seconds

    return last_closed_candle_open

_ESCAPE_RE = re.compile(r'[_*\[\]()~`>#+-=|{}.!]')

def escape_markdown_v2(text: str) -> str:
    return _ESCAPE_RE.sub(r'\\\g<0>', str(text))

def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
            return default
        return float(value)
    except (ValueError, TypeError, OverflowError):
        return default

def calculate_smooth_rsi_numpy(close: np.ndarray, rsi_len: int, kalman_len: int) -> np.ndarray:
    try:
        if close is None:
            logger.warning("Smooth RSI: Input close array is None")
            return np.full(1, 50.0, dtype=np.float64)
        
        if len(close) < rsi_len + kalman_len:
            logger.warning(
                f"Smooth RSI: Insufficient data (len={len(close)}, "
                f"required={rsi_len + kalman_len})"
            )
            return np.full(len(close), 50.0, dtype=np.float64)

        rsi = calculate_rsi_core(close, rsi_len)
        
        smooth_rsi = kalman_loop(rsi, kalman_len, 0.01, 0.1)
        
        smooth_rsi = ema_loop(smooth_rsi, float(cfg.SRSI_EMA_LEN))
        
        if cfg.NUMBA_PARALLEL and len(smooth_rsi) >= 200:
            smooth_rsi = sanitize_array_numba_parallel(smooth_rsi, 50.0)
        else:
            smooth_rsi = sanitize_array_numba(smooth_rsi, 50.0)
        
        return smooth_rsi
        
    except Exception as e:
        logger.error(f"Smooth RSI calculation failed: {e}")
        default_len = len(close) if close is not None else 1
        return np.full(default_len, 50.0, dtype=np.float64)

def calculate_ppo_numpy(close: np.ndarray, fast: int, slow: int, signal: int) -> Tuple[np.ndarray, np.ndarray]:
    try:
        if close is None or len(close) < max(fast, slow):
            logger.warning(f"PPO: Insufficient data")
            default_len = len(close) if close is not None else 1
            return np.zeros(default_len, dtype=np.float64), np.zeros(default_len, dtype=np.float64)
        
        ppo, ppo_sig = calculate_ppo_core(close, fast, slow, signal)
        
        if cfg.NUMBA_PARALLEL and len(ppo) >= 200:
            ppo = sanitize_array_numba_parallel(ppo, 0.0)
            ppo_sig = sanitize_array_numba_parallel(ppo_sig, 0.0)
        else:
            ppo = sanitize_array_numba(ppo, 0.0)
            ppo_sig = sanitize_array_numba(ppo_sig, 0.0)
        
        return ppo, ppo_sig
    except Exception as e:
        logger.error(f"PPO calculation failed: {e}")
        default_len = len(close) if close is not None else 1
        return np.zeros(default_len, dtype=np.float64), np.zeros(default_len, dtype=np.float64)

def calculate_vwap_numpy(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                         volume: np.ndarray, timestamps: np.ndarray) -> np.ndarray:   
    try:
        if close is None or len(close) == 0:
            return np.array([], dtype=np.float64)

        ts_secs = timestamps.astype(np.int64).copy()
        
        if ts_secs[0] > 1_000_000_000_000:
            ts_secs //= 1000

        hlc3 = (high + low + close) / 3.0

        vwap = vwap_daily_loop(hlc3, volume, ts_secs)

        if np.any(np.isnan(vwap)):
            vwap = np.where(np.isnan(vwap), close, vwap)

        return vwap

    except Exception as e:
        logger.error(f"VWAP calculation failed: {e}")
        return np.full_like(close, np.nan)

def calculate_rma_numpy(data: np.ndarray, period: int) -> np.ndarray:
    try:
        if data is None or len(data) < period:
            return np.zeros_like(data) if data is not None else np.array([0.0]) 
   
        alpha = 1.0 / period
        rma = ema_loop_alpha(data, alpha)
        rma = sanitize_array_numba(rma, 0.0)
        return rma       
    except Exception as e:
        logger.error(f"RMA calculation failed: {e}")
        return np.zeros_like(data) if data is not None else np.array([0.0])

def calculate_cirrus_cloud_numba(close: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        if close is None or len(close) < max(cfg.X1, cfg.X3):
            default_len = len(close) if close is not None else 1
            return (
                np.zeros(default_len, dtype=bool),
                np.zeros(default_len, dtype=bool),
                np.zeros(default_len, dtype=np.float64),
                np.zeros(default_len, dtype=np.float64)
            )
        close = np.asarray(close, dtype=np.float64)
        
        smrng_x1 = smooth_range(close, cfg.X1, cfg.X2)
        smrng_x2 = smooth_range(close, cfg.X3, cfg.X4)
        filt_x1 = rng_filter_loop(close, smrng_x1)
        filt_x12 = rng_filter_loop(close, smrng_x2)

        upw, dnw = calculate_trends_with_state(filt_x1, filt_x12)   
        return upw, dnw, filt_x1, filt_x12
        
    except Exception as e:
        default_len = len(close) if close is not None else 1
        return (
            np.zeros(default_len, dtype=bool),
            np.zeros(default_len, dtype=bool),
            np.zeros(default_len, dtype=np.float64),
            np.zeros(default_len, dtype=np.float64)
        )
        
def calculate_magical_momentum_hist(close: np.ndarray, period: int = 144, responsiveness: float = 0.9) -> np.ndarray:  
    try:
        if close is None or len(close) < period:
            return np.full(len(close) if close is not None else 1, np.nan, dtype=np.float64)

        rows = len(close)
        resp_clamped = max(0.00001, min(1.0, float(responsiveness)))
        close_c = np.ascontiguousarray(close, dtype=np.float64)

        sd = rolling_std(close_c, 50, resp_clamped)

        worm_arr = calc_mmh_worm_loop(close_c, sd, rows)

        ma = rolling_mean_numba(close_c, period)

        raw = np.empty(rows, dtype=np.float64)

        nonzero_mask = np.abs(worm_arr) >= 1e-10

        raw[nonzero_mask] = (
            (worm_arr[nonzero_mask] - ma[nonzero_mask]) 
            / worm_arr[nonzero_mask]
        )

        raw[~nonzero_mask] = np.nan

        min_med, max_med = rolling_min_max_numba(raw, period)

        value_arr = calc_mmh_value_loop(raw, min_med, max_med, rows)

        momentum = calc_mmh_momentum_loop(value_arr, rows)

        momentum = calc_mmh_momentum_smoothing(momentum, rows)

        return momentum

    except Exception as e:
        traceback.print_exc()
        return np.full(len(close) if close is not None else 1, np.nan, dtype=np.float64)

def warmup_if_needed() -> None:
    
    if aot_bridge.is_using_aot() or cfg.SKIP_WARMUP:
        logger.info("🚨 Skipping JIT warmup (AOT active or explicitly disabled)")
        return

    logger.info("🔥 Warming up JIT (core indicators only)...")
    warmup_start = time.time()
    
    try:
        test_data = np.random.random(200).astype(np.float64) * 1000.0
        test_data_ts = np.arange(200).astype(np.int64) * 900
        
        _ = aot_bridge.ema_loop(test_data, 7.0)
        _ = aot_bridge.ema_loop_alpha(test_data, 0.2)
        
        _ = aot_bridge.calculate_ppo_core(test_data, 7, 16, 5)
        
        _ = aot_bridge.calculate_rsi_core(test_data, 21)
        
        _ = aot_bridge.rolling_std(test_data, 14, 0.5)
        _ = aot_bridge.rolling_mean_numba(test_data, 14)
        
        _ = aot_bridge.kalman_loop(test_data, 10, 0.1, 0.01)
        
        _ = aot_bridge.smooth_range(test_data, 22, 2)
        _ = aot_bridge.rng_filter_loop(test_data, test_data * 0.5)
        _ = aot_bridge.calculate_trends_with_state(test_data, test_data * 0.9)
        
        _ = aot_bridge.calculate_atr_rma(test_data, test_data * 0.8, test_data, 5)
        
        warmup_elapsed = time.time() - warmup_start
        logger.info(f"✅ JIT warmup complete ({warmup_elapsed:.2f}s) | "
                   f"Core functions: EMA, PPO, RSI, RMA, Kalman, Filters")

    except Exception as e:
        warmup_elapsed = time.time() - warmup_start
        logger.warning(f"🚫 Warmup partially failed (non-fatal, {warmup_elapsed:.2f}s elapsed): {e}")
       
async def calculate_indicator_threaded(func: Callable, *args, **kwargs) -> Any:
    return await asyncio.to_thread(func, *args, **kwargs)

def calculate_pivot_levels_numpy(high: np.ndarray, low: np.ndarray, close: np.ndarray, timestamps: np.ndarray) -> Dict[str, float]:
    piv = {k: 0.0 for k in ["P", "R1", "R2", "R3", "S1", "S2", "S3"]}
    try:
        if len(timestamps) < 2:
            logger.warning("Pivot calc: insufficient data")
            return piv

        if np.any(np.isnan(high)) or np.any(np.isnan(low)) or np.any(np.isnan(close)):
            logger.warning("Pivot calc: NaN values in OHLC data")
            return piv

        days = timestamps // 86400
        now_utc = datetime.now(timezone.utc)
        yesterday = (now_utc - timedelta(days=1)).date()
        yesterday_ts = datetime(yesterday.year, yesterday.month, yesterday.day, tzinfo=timezone.utc).timestamp()
        yesterday_day_number = int(yesterday_ts) // 86400

        yesterday_mask = (days == yesterday_day_number)
        if not np.any(yesterday_mask):
            logger.warning(f"Pivot calc: Yesterday ({yesterday.isoformat()}) not found in data")
            return piv

        yesterday_high = high[yesterday_mask]
        yesterday_low = low[yesterday_mask]
        yesterday_close = close[yesterday_mask]
        if len(yesterday_high) == 0:
            logger.warning("No candles found for pivot day")
            return piv

        H_prev = float(np.max(yesterday_high))
        L_prev = float(np.min(yesterday_low))
        C_prev = float(yesterday_close[-1])
        rng_prev = H_prev - L_prev
        if rng_prev < 1e-8:
            logger.warning(f"Invalid pivot range: {rng_prev}")
            return piv

        P = (H_prev + L_prev + C_prev) / 3.0
        piv.update({
            "P": P,
            "R1": P + rng_prev * 0.382,
            "R2": P + rng_prev * 0.618,
            "R3": P + rng_prev,
            "S1": P - rng_prev * 0.382,
            "S2": P - rng_prev * 0.618,
            "S3": P - rng_prev,
        })

    except Exception as e:
        logger.error(f"Pivot calculation failed: {e}", exc_info=True)

    # Defensive cleanup
    for k, val in piv.items():
        if np.isnan(val) or np.isinf(val) or val <= 0:
            logger.warning(f"Invalid pivot {k}: {val}, resetting to 0.0")
            piv[k] = 0.0

    return piv
          
def calculate_all_indicators_numpy(data_15m: Dict[str, np.ndarray], data_5m: Dict[str, np.ndarray], data_daily: Optional[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    try:
        close_15m = data_15m["close"]
        close_5m = data_5m["close"]
        n_15m = len(close_15m)
        n_5m = len(close_5m)
        
        results = { 
            'ppo': np.empty(n_15m, dtype=np.float64),
            'ppo_signal': np.empty(n_15m, dtype=np.float64),
            'smooth_rsi': np.empty(n_15m, dtype=np.float64),
            'vwap': np.empty(n_15m, dtype=np.float64),
            'mmh': np.empty(n_15m, dtype=np.float64),
            'upw': np.zeros(n_15m, dtype=bool),
            'dnw': np.zeros(n_15m, dtype=bool),
            'rma50_15': np.empty(n_15m, dtype=np.float64),
            'rma200_5': np.empty(n_5m, dtype=np.float64),
            'pivots': {},
            'atr_short': np.empty(n_15m, dtype=np.float64),
            'atr_long': np.empty(n_15m, dtype=np.float64),
        }

        ppo, ppo_signal = calculate_ppo_numpy(
            close_15m, cfg.PPO_FAST, cfg.PPO_SLOW, cfg.PPO_SIGNAL
        )
        results['ppo'] = ppo
        results['ppo_signal'] = ppo_signal
        
        results['smooth_rsi'] = calculate_smooth_rsi_numpy(
            close_15m, cfg.SRSI_RSI_LEN, cfg.SRSI_KALMAN_LEN
        )

        if cfg.ENABLE_VWAP:
            high_15m = data_15m["high"]
            low_15m = data_15m["low"]
            volume_15m = data_15m["volume"]
            ts_15m = data_15m["timestamp"]
            
            if cfg.DEBUG_MODE and len(ts_15m) > 0:
                logger.debug(
                    f"VWAP input | "
                    f"First ts: {ts_15m[0]} ({format_ist_time(ts_15m[0])}) | "
                    f"Last ts: {ts_15m[-1]} ({format_ist_time(ts_15m[-1])}) | "
                    f"Bars: {len(ts_15m)}"
                )
            
            results["vwap"] = calculate_vwap_numpy(
                high_15m, low_15m, close_15m, volume_15m, ts_15m
            )
        else:
            results["vwap"] = np.full(n_15m, np.nan)

        mmh = calculate_magical_momentum_hist(close_15m)       
        results['mmh'] = mmh
        
        if cfg.CIRRUS_CLOUD_ENABLED:
            upw, dnw, _, _ = calculate_cirrus_cloud_numba(close_15m)
            results['upw'] = upw
            results['dnw'] = dnw
        else:
            results['upw'].fill(False)
            results['dnw'].fill(False)
        
        results['rma50_15'] = calculate_rma_numpy(close_15m, cfg.RMA_50_PERIOD)
        results['rma200_5'] = calculate_rma_numpy(close_5m, cfg.RMA_200_PERIOD)

        results['atr_short'] = calculate_atr_rma(
            data_15m["high"], data_15m["low"], data_15m["close"], cfg.ATR_SHORT
        )

        results['atr_long'] = calculate_atr_rma(
           data_15m["high"], data_15m["low"], data_15m["close"], cfg.ATR_LONG
        )


        if cfg.ENABLE_PIVOT and data_daily is not None:
            last_close = float(close_15m[-1])
            daily_high = float(data_daily["high"][-1])
            daily_low = float(data_daily["low"][-1])
            daily_range = daily_high - daily_low
            
            should_calculate = False
            if daily_range > 0:
                distance_from_high = abs(last_close - daily_high)
                distance_from_low = abs(last_close - daily_low)
                should_calculate = (
                    distance_from_high < daily_range * 0.5 or 
                    distance_from_low < daily_range * 0.5
                )
            
            if should_calculate:
                results['pivots'] = calculate_pivot_levels_numpy(
                    data_daily["high"], 
                    data_daily["low"],
                    data_daily["close"], 
                    data_daily["timestamp"]
                )
            else:
                results['pivots'] = {}
                if cfg.DEBUG_MODE:
                    logger.debug(
                        f"Skipped pivot calc (price {last_close:.2f} far from range "
                        f"{daily_low:.2f}-{daily_high:.2f})"
                    )            
        else:
            results['pivots'] = {}
            pass
        
        for key in ['ppo', 'ppo_signal', 'smooth_rsi', 'mmh', 'rma50_15', 'rma200_5']:
            arr = results[key]
            if np.any(np.isinf(arr)):
                logger.warning(f"Infinity detected in {key}, clamping values")
                results[key] = np.clip(arr, -Constants.INFINITY_CLAMP, Constants.INFINITY_CLAMP)        
        return results    
   
    except Exception as e:
        logger.error(f"calculate_all_indicators_numpy failed: {e}", exc_info=True)
        
        try:
            n = len(data_15m.get("close", []))
            if n == 0:
                logger.critical(
                    "calculate_all_indicators_numpy: No close data available - cannot proceed"
                )
                return None
        except Exception:
            logger.critical(
                "calculate_all_indicators_numpy: Cannot determine data length - aborting"
            )
            return None
        
        logger.warning(
            f"calculate_all_indicators_numpy: Returning NaN arrays ({n} elements) "
            f"due to calculation failure"
        )
        return {
            'ppo': np.full(n, np.nan, dtype=np.float64),
            'ppo_signal': np.full(n, np.nan, dtype=np.float64),
            'smooth_rsi': np.full(n, np.nan, dtype=np.float64),
            'vwap': np.full(n, np.nan, dtype=np.float64),
            'mmh': np.full(n, np.nan, dtype=np.float64),
            'upw': np.zeros(n, dtype=bool),
            'dnw': np.zeros(n, dtype=bool),
            'rma50_15': np.full(n, np.nan, dtype=np.float64),
            'rma200_5': np.full(len(data_5m.get("close", [])), np.nan, dtype=np.float64),
            'pivots': {},
            'atr_short': np.full(n, np.nan, dtype=np.float64),
            'atr_long': np.full(n, np.nan, dtype=np.float64),
        }

def precompute_candle_quality(data_15m: Dict[str, np.ndarray], atr_short: np.ndarray, 
                              atr_long: np.ndarray, rvol_threshold: float = 1.0,
                              enable_rvol_alert: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    try:
        n_ohlc = len(data_15m["close"])
        
        if n_ohlc == 0:
            logger.error("precompute_candle_quality: OHLC data is empty")
            return np.zeros(0, dtype=bool), np.zeros(0, dtype=bool)
        
        required_ohlc = ["open", "high", "low", "close"]
        for key in required_ohlc:
            if key not in data_15m:
                logger.error(f"precompute_candle_quality: Missing OHLC key '{key}'")
                return np.zeros(n_ohlc, dtype=bool), np.zeros(n_ohlc, dtype=bool)
            
            arr = data_15m[key]
            if arr is None or len(arr) == 0:
                logger.error(f"precompute_candle_quality: OHLC '{key}' is None or empty")
                return np.zeros(n_ohlc, dtype=bool), np.zeros(n_ohlc, dtype=bool)
            
            if len(arr) != n_ohlc:
                logger.error(
                    f"precompute_candle_quality: Length mismatch in '{key}' "
                    f"(expected {n_ohlc}, got {len(arr)})"
                )
                return np.zeros(n_ohlc, dtype=bool), np.zeros(n_ohlc, dtype=bool)
        
        if atr_short is None or len(atr_short) == 0:
            logger.error("precompute_candle_quality: atr_short is None or empty")
            return np.zeros(n_ohlc, dtype=bool), np.zeros(n_ohlc, dtype=bool)
        
        if atr_long is None or len(atr_long) == 0:
            logger.error("precompute_candle_quality: atr_long is None or empty")
            return np.zeros(n_ohlc, dtype=bool), np.zeros(n_ohlc, dtype=bool)
        
        if len(atr_short) != n_ohlc:
            logger.error(
                f"precompute_candle_quality: atr_short length mismatch "
                f"(expected {n_ohlc}, got {len(atr_short)})"
            )
            return np.zeros(n_ohlc, dtype=bool), np.zeros(n_ohlc, dtype=bool)
        
        if len(atr_long) != n_ohlc:
            logger.error(
                f"precompute_candle_quality: atr_long length mismatch "
                f"(expected {n_ohlc}, got {len(atr_long)})"
            )
            return np.zeros(n_ohlc, dtype=bool), np.zeros(n_ohlc, dtype=bool)
        
        if not isinstance(rvol_threshold, (int, float)) or rvol_threshold <= 0:
            logger.error(
                f"precompute_candle_quality: Invalid rvol_threshold={rvol_threshold} "
                f"(must be positive number)"
            )
            return np.zeros(n_ohlc, dtype=bool), np.zeros(n_ohlc, dtype=bool)
        
        if rvol_threshold > 10.0:
            logger.warning(
                f"precompute_candle_quality: rvol_threshold={rvol_threshold} seems very high "
                f"(typical range 0.5-2.0)"
            )
        effective_rvol_threshold = rvol_threshold if enable_rvol_alert else 0.0
        
        if cfg.DEBUG_MODE:
            rvol_status = "ENABLED" if enable_rvol_alert else "DISABLED"
            logger.debug(f"precompute_candle_quality: RVOL Alert {rvol_status}")        

        open_arr = data_15m["open"]
        high_arr = data_15m["high"]
        low_arr = data_15m["low"]
        close_arr = data_15m["close"]
        
        buy_quality = vectorized_wick_check_buy(
            open_arr, high_arr, low_arr, close_arr,
            Constants.MIN_WICK_RATIO, atr_short, atr_long, effective_rvol_threshold
        )
        
        sell_quality = vectorized_wick_check_sell(
            open_arr, high_arr, low_arr, close_arr,
            Constants.MIN_WICK_RATIO, atr_short, atr_long, effective_rvol_threshold
        )
        
        if len(buy_quality) != n_ohlc or len(sell_quality) != n_ohlc:
            logger.error(
                f"precompute_candle_quality: Output array size mismatch "
                f"(buy_quality={len(buy_quality)}, sell_quality={len(sell_quality)}, "
                f"expected={n_ohlc})"
            )
            return np.zeros(n_ohlc, dtype=bool), np.zeros(n_ohlc, dtype=bool)
        
        if cfg.DEBUG_MODE:
            buy_count = np.sum(buy_quality)
            sell_count = np.sum(sell_quality)
            logger.debug(
                f"precompute_candle_quality complete | "
                f"Buy signals: {buy_count}/{n_ohlc} | "
                f"Sell signals: {sell_count}/{n_ohlc}"
            )
        
        return buy_quality, sell_quality
    
    except Exception as e:
        logger.error(
            f"precompute_candle_quality: Unexpected error: {e}",
            exc_info=True
        )
        try:
            n = len(data_15m.get("close", []))
        except:
            n = 0
        
        if n == 0:
            logger.error("precompute_candle_quality: Cannot determine array size - returning empty")
            return np.zeros(0, dtype=bool), np.zeros(0, dtype=bool)
        
        logger.warning(f"precompute_candle_quality: Returning conservative (all False) arrays of size {n}")
        return np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)

class SessionManager:
    _session: ClassVar[Optional[aiohttp.ClientSession]] = None
    _ssl_context: ClassVar[Optional[ssl.SSLContext]] = None
    _lock: ClassVar[asyncio.Lock] = asyncio.Lock()
    _creation_time: ClassVar[float] = 0.0
    _request_count: ClassVar[int] = 0
    _session_reuse_limit: ClassVar[int] = 1000

    @classmethod
    def _get_ssl_context(cls) -> ssl.SSLContext:
        if cls._ssl_context is None:
            ctx = ssl.create_default_context()
            ctx.check_hostname = True
            ctx.verify_mode = ssl.CERT_REQUIRED
            ctx.minimum_version = ssl.TLSVersion.TLSv1_2
            cls._ssl_context = ctx
            logger.debug("SSL context created with TLSv1.2+ minimum")
        return cls._ssl_context

    @classmethod
    async def get_session(cls) -> aiohttp.ClientSession:
        async with cls._lock:
            # Check if we need to recreate
            should_recreate = False
            reason = None

            if cls._session is None or cls._session.closed:
                should_recreate = True
                reason = "no session"
            elif cls._request_count >= cls._session_reuse_limit:
                should_recreate = True
                reason = f"request limit reached ({cls._request_count})"
                logger.info(f"Session recreation triggered: {reason}")

            if should_recreate:
                if cls._session and not cls._session.closed:
                    try:
                        await cls._session.close()
                        await asyncio.sleep(0.1)
                    except Exception as e:
                        logger.warning(f"Error closing old session: {e}")

                connector = TCPConnector(
                    limit=cfg.TCP_CONN_LIMIT,
                    limit_per_host=cfg.TCP_CONN_LIMIT_PER_HOST,
                    ssl=cls._get_ssl_context(),
                    force_close=False,
                    enable_cleanup_closed=True,
                    ttl_dns_cache=3600,
                    keepalive_timeout=90,
                    family=0,
                )

                timeout = aiohttp.ClientTimeout(
                    total=cfg.HTTP_TIMEOUT,
                    connect=8,
                    sock_read=cfg.HTTP_TIMEOUT,
                )

                cls._session = aiohttp.ClientSession(
                    connector=connector,
                    timeout=timeout,
                    headers={
                        "User-Agent": f"{cfg.BOT_NAME}/{__version__}",
                        "Accept": "application/json",
                        "Accept-Encoding": "gzip, deflate",
                        "Connection": "keep-alive",
                    },
                    raise_for_status=False,
                )
                cls._creation_time = time.time()
                cls._request_count = 0

                if cfg.DEBUG_MODE:
                    logger.debug("HTTP session created")

            return cls._session

    @classmethod
    def track_request(cls) -> None:
        cls._request_count += 1

        threshold_warning = cls._session_reuse_limit * 0.8
        if cls._request_count == int(threshold_warning):
            logger.debug(
                f"Session approaching recreation threshold: "
                f"{cls._request_count}/{cls._session_reuse_limit} requests"
            )

    @classmethod
    async def close_session(cls) -> None:
        async with cls._lock:
            if cls._session and not cls._session.closed:
                try:
                    session_age = time.time() - cls._creation_time

                    if logger.isEnabledFor(logging.DEBUG):
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
        if hasattr(exc, "status") and exc.status == 429:
            return RetryCategory.RATE_LIMIT
        return RetryCategory.API_ERROR
    elif isinstance(exc, (ClientError, aiohttp.ClientError)):
        return RetryCategory.NETWORK
    return RetryCategory.UNKNOWN

async def retry_async(fn: Callable, *args, retries: int = 3, base_backoff: float = 0.8, cap: float = 30.0, jitter_min: float = 0.05, jitter_max: float = 0.5, on_error: Optional[Callable[[Exception, int, str], None]] = None, **kwargs):
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

async def async_fetch_json(url: str, params: Optional[Dict[str, Any]] = None, retries: int = 3, backoff: float = 1.5, timeout: int = 15) -> Optional[Dict[str, Any]]:   
    session = await SessionManager.get_session()    
    retry_stats = {
        RetryCategory.NETWORK: 0,
        RetryCategory.RATE_LIMIT: 0,
        RetryCategory.API_ERROR: 0,
        RetryCategory.TIMEOUT: 0,
        RetryCategory.UNKNOWN: 0
    }
    last_error: Optional[Exception] = None
    
    for attempt in range(1, retries + 1):
        if shutdown_event.is_set():
            logger.debug(f"Shutdown requested, aborting fetch: {url[:80]}")
            return None
        
        try:
            async with session.get(url, params=params, timeout=timeout) as resp:
                if resp.status == 429:
                    retry_after = resp.headers.get('Retry-After')
                    wait_sec = min(
                        int(retry_after) if retry_after else 2,
                        Constants.CIRCUIT_BREAKER_MAX_WAIT
                    )
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
                f"URL: {url[:80]} | Timeout configured: {timeout}s"
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
        self.concurrency = concurrency
        self.semaphore = asyncio.Semaphore(concurrency)
        self.requests: deque[float] = deque()
        self.lock = asyncio.Lock()
        self.total_waits = 0
        self.total_wait_time = 0.0
        self.last_request_time = 0.0

    async def call(self, func: Callable, *args, **kwargs):
        async with self.lock:
            now = time.time()
            while self.requests and now - self.requests[0] > 60.0:
                self.requests.popleft()
            if len(self.requests) >= self.max_per_minute:
                oldest_request_age = now - self.requests[0]
                wait_needed = max(0.0, 60.0 - oldest_request_age)
                
                jitter = random.uniform(0.05, 0.2)
                total_sleep = wait_needed + jitter
                
                self.total_waits += 1
                self.total_wait_time += total_sleep
                
                logger.debug(
                    f"Rate limit reached ({len(self.requests)}/{self.max_per_minute}), "
                    f"sleeping {total_sleep:.2f}s | Total waits: {self.total_waits}"
                )
                
                await asyncio.sleep(total_sleep)
                
                now = time.time()
                while self.requests and now - self.requests[0] > 60.0:
                    self.requests.popleft()
        
        async with self.lock:
            self.requests.append(time.time())
            self.last_request_time = time.time()
        
        async with self.semaphore:
            return await func(*args, **kwargs)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_waits": self.total_waits,
            "total_wait_time_seconds": round(self.total_wait_time, 2),
            "current_queue_size": len(self.requests),
            "max_per_minute": self.max_per_minute,
            "concurrency_limit": self.concurrency,
            "requests_in_window": len(self.requests),
        }

class APICircuitBreaker:  
    def __init__(self, failure_threshold: int = 3, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.last_failure_time = 0.0
        self.state = "CLOSED"
        self.success_count = 0
        
    def record_success(self) -> None:
        if self.state == "HALF_OPEN":
            self.success_count += 1
            if self.success_count >= 2:
                logger.info("💫 Circuit breaker: Recovered, transitioning to CLOSED")
                self.state = "CLOSED"
                self.failures = 0
                self.success_count = 0
        elif self.state == "CLOSED":
            if self.failures > 0:
                self.failures = max(0, self.failures - 1)
   
    def record_failure(self) -> None:
        self.failures += 1
        self.last_failure_time = time.time()
        
        if self.failures >= self.failure_threshold and self.state == "CLOSED":
            logger.warning(
                f"⚠️ Circuit breaker: OPENED after {self.failures} failures. "
                f"Blocking requests for {self.recovery_timeout}s"
            )
            self.state = "OPEN"
    
    def can_attempt(self) -> Tuple[bool, Optional[str]]:
        if self.state == "CLOSED":
            return True, None
        
        if self.state == "OPEN":
            elapsed = time.time() - self.last_failure_time
            if elapsed >= self.recovery_timeout:
                logger.info("🟡 Circuit breaker: Transitioning to HALF_OPEN (testing recovery)")
                self.state = "HALF_OPEN"
                self.success_count = 0
                return True, None            
            return False, f"Circuit breaker OPEN (retry in {self.recovery_timeout - elapsed:.0f}s)"        
        return True, None

class DataFetcher:
    def __init__(self, api_base: str, *, session: Optional[aiohttp.ClientSession] = None, max_parallel: Optional[int] = None):
        self.api_base = api_base.rstrip("/")
        self._external_session = session
        max_parallel = max_parallel or cfg.MAX_PARALLEL_FETCH
        self.semaphore = asyncio.Semaphore(max_parallel)
        self.timeout = cfg.HTTP_TIMEOUT
        self.rate_limiter = RateLimitedFetcher(
            max_per_minute=cfg.RATE_LIMIT_PER_MINUTE,
            concurrency=max_parallel,
        )
        self.circuit_breaker = APICircuitBreaker(
            failure_threshold=cfg.CB_FAILURE_THRESHOLD,
            recovery_timeout=cfg.CB_RECOVERY_TIMEOUT,
        )
        self.fetch_stats = {
            "products": {"success": 0, "failed": 0},
            "candles": {"success": 0, "failed": 0},
            "circuit_breaker_blocks": 0,
            "rate_limiter_waits": 0,
            "total_wait_time": 0.0,
        }

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._external_session is not None:
            return self._external_session
        return await SessionManager.get_session()
  
    async def fetch_candles(self, symbol: str, resolution: str, limit: int, reference_time: int, expected_open_15: Optional[int] = None) -> Optional[Dict[str, Any]]:
        can_proceed, reason = self.circuit_breaker.can_attempt()
        if not can_proceed:
            logger.warning(f"Circuit breaker blocked candles {symbol}: {reason}")
            self.fetch_stats["circuit_breaker_blocks"] += 1
            self.fetch_stats["candles"]["failed"] += 1
            return None

        minutes = int(resolution) if resolution != "D" else 1440
        interval_seconds = minutes * 60

        if minutes == 15 and expected_open_15 is not None:
            expected_open_ts = expected_open_15
        else:
            expected_open_ts = calculate_expected_candle_timestamp(reference_time, minutes)

        buffer_periods = 3
        to_time   = reference_time + (interval_seconds * buffer_periods)
        from_time = expected_open_ts - (limit * interval_seconds)

        params = {
            "resolution": resolution,
            "symbol": symbol,
            "from": int(from_time),
            "to": int(to_time),
        }
        url = f"{self.api_base}/v2/chart/history"

        async with self.semaphore:
            data = await self.rate_limiter.call(
                async_fetch_json,
                url,
                params=params,
                retries=cfg.CANDLE_FETCH_RETRIES,
                backoff=cfg.CANDLE_FETCH_BACKOFF,
                timeout=self.timeout,
            )

            if data:
                result = data.get("result", {})
                if result and all(k in result for k in ("t", "o", "h", "l", "c", "v")):
                    self.circuit_breaker.record_success()
                    self.fetch_stats["candles"]["success"] += 1

                    num_candles = len(result.get("t", []))
                    if num_candles > 0:
                        last_open = result["t"][-1]
                        diff = abs(expected_open_ts - last_open)
                        if diff > 300:
                            if last_open < expected_open_ts:
                                logger.warning(
                                    f"⚠️ API DELAY | {symbol} {resolution} | "
                                    f"Expected: {format_ist_time(expected_open_ts)} | "
                                    f"Got: {format_ist_time(last_open)} (Diff: {diff}s)"
                                )
                            else:
                                logger.debug(f"API Ahead | {symbol} {resolution} | Diff: {diff}s")
                        else:
                            logger.debug(
                                f"✅ Scanned {symbol} {resolution} | "
                                f"Latest: {format_ist_time(last_open)} | Candles: {num_candles}"
                            )
                    return data
                else:
                    logger.warning(f"Candles response missing fields | Symbol: {symbol}")
                    self.fetch_stats["candles"]["failed"] += 1
                    self.circuit_breaker.record_failure()
            else:
                logger.warning(f"Candles fetch failed | Symbol: {symbol}")
                self.fetch_stats["candles"]["failed"] += 1
                self.circuit_breaker.record_failure()

            return None

    def get_stats(self) -> Dict[str, Any]:
        stats = {
            "products": self.fetch_stats["products"].copy(),
            "candles": self.fetch_stats["candles"].copy(),
            "circuit_breaker_blocks": self.fetch_stats["circuit_breaker_blocks"],
            "rate_limiter": self.rate_limiter.get_stats(),
        }
        
        total_products = stats["products"]["success"] + stats["products"]["failed"]
        total_candles = stats["candles"]["success"] + stats["candles"]["failed"]
        
        if total_products > 0:
            stats["products"]["success_rate"] = round(
                stats["products"]["success"] / total_products * 100, 1
            )
        
        if total_candles > 0:
            stats["candles"]["success_rate"] = round(
                stats["candles"]["success"] / total_candles * 100, 1
            )        
        return stats

    async def fetch_candles_batch(self, requests: List[Tuple[str, str, int]], reference_time: Optional[int] = None) -> Dict[str, Optional[Dict[str, Any]]]:
        if reference_time is None:
            reference_time = get_trigger_timestamp()
        
        tasks = []
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

    async def fetch_all_candles_truly_parallel(self, pair_requests: List[Tuple[str, List[Tuple[str, int]]]], reference_time: Optional[int] = None) -> Dict[str, Dict[str, Optional[Dict[str, Any]]]]:
        if reference_time is None:
            reference_time = get_trigger_timestamp()

        expected_open_15 = calculate_expected_candle_timestamp(reference_time, 15)
        all_tasks = []
        task_metadata = []
        for symbol, resolutions in pair_requests:
            for resolution, limit in resolutions:
                task = self.fetch_candles(
                    symbol, resolution, limit, reference_time, expected_open_15
                )
                all_tasks.append(task)
                task_metadata.append((symbol, resolution))

        results = await asyncio.gather(*all_tasks, return_exceptions=True)
        output = {}
        success_count = 0
        
        for (symbol, resolution), result in zip(task_metadata, results):
            if symbol not in output: 
                output[symbol] = {}
            if isinstance(result, Exception):
                output[symbol][resolution] = None
            else:
                output[symbol][resolution] = result
                if result: 
                    success_count += 1
        logger.info(f"📏 Parallel fetch complete | Success: {success_count}/{len(all_tasks)}")
        return output

def parse_candles_to_numpy(result: Optional[Dict[str, Any]]) -> Optional[Dict[str, np.ndarray]]:
    if not result or not isinstance(result, dict):
        logger.warning("parse_candles_to_numpy: result is None or not dict")
        return None
    
    res = result.get("result", {}) or {}
    required = ("t", "o", "h", "l", "c", "v")
    if not all(k in res for k in required):
        logger.warning(f"parse_candles_to_numpy: missing required fields. Have: {list(res.keys())}")
        return None

    try:
        data = {
            "timestamp": np.asarray(res["t"], dtype=np.int64),
            "open":      np.asarray(res["o"], dtype=np.float64),
            "high":      np.asarray(res["h"], dtype=np.float64),
            "low":       np.asarray(res["l"], dtype=np.float64),
            "close":     np.asarray(res["c"], dtype=np.float64),
            "volume":    np.asarray(res["v"], dtype=np.float64),
        }

        n = len(data["timestamp"])
        if n == 0:
            logger.warning("parse_candles_to_numpy: empty candle array")
            return None

        if data["timestamp"][-1] > 1_000_000_000_000:
            data["timestamp"] //= 1000

        o, h, l, c = data["open"], data["high"], data["low"], data["close"]

        any_nan = np.isnan(o) | np.isnan(h) | np.isnan(l) | np.isnan(c)
        any_inf = np.isinf(o) | np.isinf(h) | np.isinf(l) | np.isinf(c)
        
        invalid_rel = ~((l <= o) & (o <= h) & (l <= c) & (c <= h))
        
        non_positive = (o <= 0) | (h <= 0) | (l <= 0) | (c <= 0)

        error_mask = any_nan | any_inf | invalid_rel | non_positive
        error_count = np.sum(error_mask)

        if error_count > 0:
            logger.error(f"parse_candles_to_numpy: {error_count} invalid candles found.")
            first_errors = np.where(error_mask)[0][:5]
            for idx in first_errors:
                logger.error(f"  Index {idx}: Data Anomaly (O={o[idx]:.2f} H={h[idx]:.2f} L={l[idx]:.2f} C={c[idx]:.2f})")
            return None

        hl_mid = (h + l) / 2.0
        close_deviation = np.abs(c - hl_mid) / (hl_mid + 1e-9)
        deviation_mask = close_deviation > 0.5
        deviation_count = np.sum(deviation_mask)

        if deviation_count > 0:
            logger.warning(
                f"parse_candles_to_numpy: Found {deviation_count} candles with "
                f"Close deviating >50% from High-Low midpoint (potential data anomaly)"
            )

        return data

    except Exception as e:
        logger.error(f"parse_candles_to_numpy: Exception during parsing: {e}")
        return None

def validate_candle_data(data: Optional[Dict[str, np.ndarray]], required_len: int = 0) -> Tuple[bool, Optional[str]]:
    try:
        if data is None or not data:
            return False, "Data is None or empty"
        
        close = data.get("close")
        timestamp = data.get("timestamp")
        open_arr = data.get("open")
        high_arr = data.get("high")
        low_arr = data.get("low")
        
        if timestamp is None or len(timestamp) == 0:
            return False, "Timestamp array is empty"
        
        if close is None or len(close) == 0:
            return False, "Close array is empty"
        
        if open_arr is None or high_arr is None or low_arr is None:
            return False, "Missing OHLC data (open, high, or low)"
        
        if len(close) < required_len:
            return False, f"Insufficient data: {len(close)} < {required_len}"
        
        if not np.all(timestamp[1:] >= timestamp[:-1]):
            return False, "Timestamps not monotonic increasing"
        
        if len(open_arr) != len(close) or len(high_arr) != len(close) or len(low_arr) != len(close):
            return False, "OHLC arrays have mismatched lengths"
        
        return True, None
    
    except Exception as e:
        logger.error(f"Data validation exception: {e}", exc_info=True)
        return False, f"Validation error: {str(e)}"

def get_last_closed_index_from_array(timestamps: np.ndarray, interval_minutes: int, reference_time: Optional[int] = None) -> Optional[int]:
    if timestamps is None or timestamps.size < 2:
        logger.warning("No timestamps or insufficient data")
        return None

    if reference_time is None:
        reference_time = get_trigger_timestamp()

    interval_seconds = interval_minutes * 60
    current_period_start = (reference_time // interval_seconds) * interval_seconds

    last_closed_period_start = current_period_start - interval_seconds
    valid_mask = (timestamps >= last_closed_period_start) & (timestamps < current_period_start)

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
            f"Selected fully closed candle | Index: {last_closed_idx} | "
            f"TS: {format_ist_time(timestamps[last_closed_idx])}"
        ),
    )
    return last_closed_idx

def validate_candle_data_at_index(data: Optional[Dict[str, np.ndarray]], selected_index: int, reference_time: int, interval_minutes: int = 15) -> Tuple[bool, Optional[str]]:
    try:
        if data is None or not data:
            return False, "Data is None or empty"
        
        close = data.get("close")
        timestamp = data.get("timestamp")
        open_arr = data.get("open")
        high_arr = data.get("high")
        low_arr = data.get("low")
        
        if any(arr is None or len(arr) == 0 for arr in [close, timestamp, open_arr, high_arr, low_arr]):
            return False, "Missing or empty OHLC/timestamp data"
        
        if selected_index < 0 or selected_index >= len(close):
            return False, f"Selected index {selected_index} out of range [0, {len(close)})"
        
        selected_candle_time = int(timestamp[selected_index])
        current_time = reference_time
        interval_seconds = interval_minutes * 60
        current_period_start = (current_time // interval_seconds) * interval_seconds
        
        if selected_candle_time >= current_period_start:
            return False, (
                f"Selected candle is still forming! "
                f"ts={format_ist_time(selected_candle_time)} "
                f"current_period_start={format_ist_time(current_period_start)}"
            )
        
        staleness = current_time - selected_candle_time
        MAX_STALENESS = cfg.MAX_CANDLE_STALENESS_SEC
        
        if staleness > MAX_STALENESS:
            return False, (
                f"Selected candle is stale: {staleness}s old (max: {MAX_STALENESS}s). "    
                f"Candle: {format_ist_time(selected_candle_time)} | "
                f"Current: {format_ist_time(current_time)}"
            )
        
        o = open_arr[selected_index]
        h = high_arr[selected_index]
        l = low_arr[selected_index]
        c = close[selected_index]
        
        if np.isnan(o) or np.isnan(h) or np.isnan(l) or np.isnan(c):
            return False, f"Selected candle has NaN values: O={o} H={h} L={l} C={c}"
        
        if np.isinf(o) or np.isinf(h) or np.isinf(l) or np.isinf(c):
            return False, f"Selected candle has Inf values: O={o} H={h} L={l} C={c}"
        
        if o <= 0 or h <= 0 or l <= 0 or c <= 0:
            return False, f"Selected candle has non-positive prices: O={o} H={h} L={l} C={c}"
        
        if not (l <= o and o <= h and l <= c and c <= h and l <= h):
            return False, (
                f"Selected candle has invalid OHLC relationships! "
                f"O={o:.2f} H={h:.2f} L={l:.2f} C={c:.2f} "
                f"[L≤O={l<=o} O≤H={o<=h} L≤C={l<=c} C≤H={c<=h} L≤H={l<=h}]"
            )
        
        if selected_index > 0:
            prev_close = close[selected_index - 1]
            if not np.isnan(prev_close):
                price_change_pct = abs(c - prev_close) / prev_close * 100
                if price_change_pct > Constants.MAX_PRICE_CHANGE_PERCENT:
                    return False, f"Extreme price spike in selected candle: {price_change_pct:.2f}%"
        
        return True, None
    
    except Exception as e:
        logger.error(f"Candle validation exception at index {selected_index}: {e}", exc_info=True)
        return False, f"Validation error: {str(e)}"

def validate_candle_timestamp(candle_ts: int, reference_time: int, interval_minutes: int, tolerance_seconds: int = 120) -> bool:
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

def build_products_map_from_cfg() -> Dict[str, dict]:
    products_map: Dict[str, dict] = {}
    for pair in cfg.PAIRS:
        products_map[pair] = {
            "id": pair,                 
            "symbol": pair,
            "contract_type": "perpetual_futures"
        }
    logger.info(
        f"📦 Product map built from cfg: {len(products_map)}/{len(cfg.PAIRS)} matched | "
        f"Coverage: {(len(products_map)/len(cfg.PAIRS))*100:.0f}%"
    )
    return products_map

async def fetch_all_pairs_candles(fetcher: DataFetcher, reference_time: int) -> Dict[str, Dict[str, Optional[Dict[str, Any]]]]:
    requests = []
    for pair in cfg.PAIRS:
        requests.append((pair, "15", cfg.MIN_CANDLES_FOR_INDICATORS))
        requests.append((pair, "5", cfg.MIN_CANDLES_FOR_INDICATORS))
        requests.append((pair, "D", cfg.MIN_CANDLES_FOR_INDICATORS))
    return await fetcher.fetch_all_candles_truly_parallel(
        [(pair, [("15", cfg.MIN_CANDLES_FOR_INDICATORS),
                 ("5", cfg.MIN_CANDLES_FOR_INDICATORS),
                 ("D", cfg.MIN_CANDLES_FOR_INDICATORS)]) for pair in cfg.PAIRS],
        reference_time
    )

class SimpleLock:
    """Simple async lock for single-process synchronization (GitHub Actions)"""
    def __init__(self):
        self._lock = asyncio.Lock()
        self.acquired_by_me = False

    async def acquire(self, timeout: float = 5.0) -> bool:
        """Acquire the lock with timeout"""
        try:
            await asyncio.wait_for(self._lock.acquire(), timeout)
            self.acquired_by_me = True
            logger.debug("🔐 SimpleLock acquired successfully")
            return True
        except asyncio.TimeoutError:
            logger.warning("⏱️ SimpleLock acquisition timed out")
            return False
        except Exception as e:
            logger.error(f"SimpleLock acquisition error: {e}")
            return False

    async def release(self, timeout: float = 5.0) -> bool:
        """Release the lock"""
        try:
            if self.acquired_by_me:
                self._lock.release()
                self.acquired_by_me = False
                logger.debug("🔓 SimpleLock released successfully")
            return True
        except Exception as e:
            logger.warning(f"SimpleLock release error: {e}")
            return False

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

        def _safe_truncate_utf8(text: str, max_bytes: int) -> str:
            encoded = text.encode('utf-8')
            if len(encoded) <= max_bytes:
                return text
            return encoded[:max_bytes].decode('utf-8', errors='ignore')

        MAX_LEN = Constants.TELEGRAM_MAX_MESSAGE_LENGTH
        SAFETY_MARGIN = 100  # Account for URL encoding overhead
        EFFECTIVE_MAX = MAX_LEN - SAFETY_MARGIN
        SEPARATOR = "\n\n"
        SEP_BYTES = len(SEPARATOR.encode('utf-8'))

        batches: List[List[str]] = []
        current: List[str] = []
        current_bytes: int = 0

        for msg in messages:
            try:
                msg_bytes = len(msg.encode('utf-8'))
            except Exception as e:
                logger.warning(f"Failed to encode message: {e}, skipping")
                continue

            estimated_encoded = int(msg_bytes * 1.15)
    
            needed = estimated_encoded
            if current:
                needed += SEP_BYTES

            if estimated_encoded > EFFECTIVE_MAX:
                if current:
                    batches.append(current)
                    current = []
                    current_bytes = 0
                truncated = _safe_truncate_utf8(msg, EFFECTIVE_MAX)
                batches.append([truncated])
                continue

            if current_bytes + needed > EFFECTIVE_MAX:
                batches.append(current)
                current = []
                current_bytes = 0

            current.append(msg)
            current_bytes += needed

        if current:
            batches.append(current)

        if len(batches) > 1:
            logger.info(f"Split alerts into {len(batches)} Telegram messages")

        results = []
        for idx, batch in enumerate(batches):
            text = SEPARATOR.join(batch)
            results.append(await self.send(text))

            if idx < len(batches) - 1:
                await asyncio.sleep(Constants.INTER_BATCH_DELAY)

        return all(results)

def build_single_msg(title, pair, price, ts, extra=None):
    if not title: title = "ALERT"
    parts = title.split(" ", 1)
    symbols = parts[0]
    description = parts[1] if len(parts) == 2 else title
    price_str = f"${price:,.2f}" if isinstance(price, (int, float)) else "N/A"
    line2 = f"{description} : {extra}" if extra else description
    return escape_markdown_v2(f"{symbols} {pair} - {price_str}\n{line2}\n{format_ist_time(ts, '%d-%m-%Y %H:%M IST')}")
        
def build_batched_msg(pair: str, price: float, ts: int, items: List[Tuple[str, str]]) -> str:
    if not items:
        return escape_markdown_v2(f"{pair} - ${price:,.2f} {format_ist_time(ts, '%d-%m-%Y %H:%M IST')}")
    
    headline_emoji = items[0][0].split(" ", 1)[0] if items[0][0] else "📊"
    headline = f"{headline_emoji} **{pair}** • ${price:,.2f}  {format_ist_time(ts, '%d-%m-%Y %H:%M IST')}"
    
    bullets = []
    for idx, (title, extra) in enumerate(items):
        prefix = "└➤" if idx == len(items) - 1 else "├➤"
        bullets.append(f"{prefix} {title} | {extra}")
    
    body = "\n".join(bullets)
    return escape_markdown_v2(f"{headline}\n{body}")

def create_pivot_alert(level: str, is_buy: bool) -> AlertDefinition:
    """Factory function to create pivot alert definition without lambdas"""
    if is_buy:
        return {
            "key": f"pivot_up_{level}",
            "title": f"🟢⬆️ Cross above {level}",
            "check_fn": lambda ctx, ppo, ppo_sig, rsi, _: (
                ctx.get("buy_common", False) and 
                get_pivot_alert_info(ctx, level, is_buy=True)[0]
            ),
            "extra_fn": lambda ctx, ppo, ppo_sig, rsi, _: (
                f"${ctx['pivots'][level]:,.2f} | MMH ({ctx['mmh_curr']:.2f}) "
                f"[Dist: {abs(ctx['pivots'][level] - ctx['close_curr'])/ctx['close_curr']*100:.2f}%]"
            ),
            "requires": ["pivots"]
        }
    else:
        return {
            "key": f"pivot_down_{level}",
            "title": f"🔴⬇️ Cross below {level}",
            "check_fn": lambda ctx, ppo, ppo_sig, rsi, _: (
                ctx.get("sell_common", False) and 
                get_pivot_alert_info(ctx, level, is_buy=False)[0]
            ),
            "extra_fn": lambda ctx, ppo, ppo_sig, rsi, _: (
                f"${ctx['pivots'][level]:,.2f} | MMH ({ctx['mmh_curr']:.2f}) "
                f"[Dist: {abs(ctx['pivots'][level] - ctx['close_curr'])/ctx['close_curr']*100:.2f}%]"
            ),
            "requires": ["pivots"]
        }

class AlertDefinition(TypedDict):
    key: str
    title: str
    check_fn: Callable[[Any, Any, Any, Any], bool]
    extra_fn: Callable[[Any, Any, Any, Any, Dict[str, Any]], str]
    requires: List[str]
    
ALERT_DEFINITIONS: List[AlertDefinition] = [
    {"key":"ppo_signal_up","title":"🟢 PPO cross above signal","check_fn":lambda ctx,ppo,ppo_sig,rsi:(ctx.get("buy_common",False) and (ppo.get("prev",np.nan)<=ppo_sig.get("prev",np.nan)) and (ppo.get("curr",np.nan)>ppo_sig.get("curr",np.nan)) and (ppo.get("curr",np.nan)<Constants.PPO_THRESHOLD_BUY)),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"PPO {ppo.get('curr',0):.2f} vs Sig {ppo_sig.get('curr',0):.2f} | Wick {ctx.get('buy_wick_ratio',0)*100:.1f}% | MMH ({ctx.get('mmh_curr',0):.2f})","requires":["ppo","ppo_signal"]},
    {"key":"ppo_signal_down","title":"🔴 PPO cross below signal","check_fn":lambda ctx,ppo,ppo_sig,rsi:(ctx.get("sell_common",False) and (ppo.get("prev",np.nan)>=ppo_sig.get("prev",np.nan)) and (ppo.get("curr",np.nan)<ppo_sig.get("curr",np.nan)) and (ppo.get("curr",np.nan)>Constants.PPO_THRESHOLD_SELL)),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"PPO {ppo.get('curr',0):.2f} vs Sig {ppo_sig.get('curr',0):.2f} | Wick {ctx.get('sell_wick_ratio',0)*100:.1f}% | MMH ({ctx.get('mmh_curr',0):.2f})","requires":["ppo","ppo_signal"]},
    {"key":"ppo_zero_up","title":"🟢 PPO cross above 0","check_fn":lambda ctx,ppo,ppo_sig,rsi:(ctx.get("buy_common",False) and (ppo.get("prev",np.nan)<=0.0) and (ppo.get("curr",np.nan)>0.0)),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"PPO {ppo.get('curr',0):.2f} | Wick {ctx.get('buy_wick_ratio',0)*100:.1f}% | MMH ({ctx.get('mmh_curr',0):.2f})","requires":["ppo"]},
    {"key":"ppo_zero_down","title":"🔴 PPO cross below 0","check_fn":lambda ctx,ppo,ppo_sig,rsi:(ctx.get("sell_common",False) and (ppo.get("prev",np.nan)>=0.0) and (ppo.get("curr",np.nan)<0.0)),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"PPO {ppo.get('curr',0):.2f} | Wick {ctx.get('sell_wick_ratio',0)*100:.1f}% | MMH ({ctx.get('mmh_curr',0):.2f})","requires":["ppo"]},
    {"key":"ppo_011_up","title":"🟢 PPO cross above 0.11","check_fn":lambda ctx,ppo,ppo_sig,rsi:(ctx.get("buy_common",False) and (ppo.get("prev",np.nan)<=Constants.PPO_011_THRESHOLD) and (ppo.get("curr",np.nan)>Constants.PPO_011_THRESHOLD)),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"PPO {ppo.get('curr',0):.2f} | Wick {ctx.get('buy_wick_ratio',0)*100:.1f}% | MMH ({ctx.get('mmh_curr',0):.2f})","requires":["ppo"]},
    {"key":"ppo_011_down","title":"🔴 PPO cross below -0.11","check_fn":lambda ctx,ppo,ppo_sig,rsi:(ctx.get("sell_common",False) and (ppo.get("prev",np.nan)>=Constants.PPO_011_THRESHOLD_SELL) and (ppo.get("curr",np.nan)<Constants.PPO_011_THRESHOLD_SELL)),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"PPO {ppo.get('curr',0):.2f} | Wick {ctx.get('sell_wick_ratio',0)*100:.1f}% | MMH ({ctx.get('mmh_curr',0):.2f})","requires":["ppo"]},
    {"key":"rsi_50_up","title":"🟢 RSI cross above 50 (PPO < 0.30)","check_fn":lambda ctx,ppo,ppo_sig,rsi:(ctx.get("buy_common",False) and (rsi.get("prev",50)<=Constants.RSI_THRESHOLD) and (rsi.get("curr",50)>Constants.RSI_THRESHOLD) and (ppo.get("curr",np.nan)<Constants.PPO_RSI_GUARD_BUY)),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"RSI {rsi.get('curr',50):.2f} | PPO {ppo.get('curr',0):.2f} | Wick {ctx.get('buy_wick_ratio',0)*100:.1f}% | MMH ({ctx.get('mmh_curr',0):.2f})","requires":["ppo","rsi"]},
    {"key":"rsi_50_down","title":"🔴 RSI cross below 50 (PPO > -0.30)","check_fn":lambda ctx,ppo,ppo_sig,rsi:(ctx.get("sell_common",False) and (rsi.get("prev",50)>=Constants.RSI_THRESHOLD) and (rsi.get("curr",50)<Constants.RSI_THRESHOLD) and (ppo.get("curr",np.nan)>Constants.PPO_RSI_GUARD_SELL)),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"RSI {rsi.get('curr',50):.2f} | PPO {ppo.get('curr',0):.2f} | Wick {ctx.get('sell_wick_ratio',0)*100:.1f}% | MMH ({ctx.get('mmh_curr',0):.2f})","requires":["ppo","rsi"]},
    {"key":"vwap_up","title":"🔵▲ Price cross above VWAP","check_fn":lambda ctx,ppo,ppo_sig,rsi:(ctx.get("buy_common",False) and (ctx.get("close_prev",0)<=ctx.get("vwap_prev",0)) and (ctx.get("close_curr",0)>ctx.get("vwap_curr",0))),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"VWAP {ctx.get('vwap_curr',0):.2f} | Wick {ctx.get('buy_wick_ratio',0)*100:.1f}% | MMH ({ctx.get('mmh_curr',0):.2f})","requires":["vwap"]},
    {"key":"vwap_down","title":"🟣▼ Price cross below VWAP","check_fn":lambda ctx,ppo,ppo_sig,rsi:(ctx.get("sell_common",False) and (ctx.get("close_prev",0)>=ctx.get("vwap_prev",0)) and (ctx.get("close_curr",0)<ctx.get("vwap_curr",0))),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"VWAP {ctx.get('vwap_curr',0):.2f} | Wick {ctx.get('sell_wick_ratio',0)*100:.1f}% | MMH ({ctx.get('mmh_curr',0):.2f})","requires":["vwap"]},
    {"key":"mmh_buy","title":"🔵⬆️ MMH Reversal BUY","check_fn":lambda ctx,ppo,ppo_sig,rsi:(ctx.get("buy_common",False) and ctx.get("mmh_reversal_buy",False)),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"MMH ({ctx.get('mmh_curr',0):.2f}) | Wick {ctx.get('buy_wick_ratio',0)*100:.1f}%","requires":[]},
    {"key":"mmh_sell","title":"🟣⬇️ MMH Reversal SELL","check_fn":lambda ctx,ppo,ppo_sig,rsi:(ctx.get("sell_common",False) and ctx.get("mmh_reversal_sell",False)),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"MMH ({ctx.get('mmh_curr',0):.2f}) | Wick {ctx.get('sell_wick_ratio',0)*100:.1f}%","requires":[]}
]
def _validate_pivot_cross(ctx: Dict[str, Any], level: str, is_buy: bool) -> Tuple[bool, Optional[str]]: 
    pivots = ctx.get("pivots")
    if not pivots or level not in pivots:
        return False, "No pivot data"

    level_value = pivots[level]
    if level_value <= 0:
        return False, "Invalid pivot value"

    close_curr = ctx.get("close_curr")
    close_prev = ctx.get("close_prev")

    if close_curr is None or close_prev is None:
        return False, "Missing close data"

    if is_buy:
        crossed = close_prev <= level_value < close_curr
    else:
        crossed = close_prev >= level_value > close_curr

    if not crossed:
        return False, "No pivot cross"

    try:
        price_diff_pct = (abs(level_value - close_curr) / level_value) * 100
    except ZeroDivisionError:
        return False, "Pivot invalid (zero)"

    if price_diff_pct > Constants.PIVOT_MAX_DISTANCE_PCT:
        return False, (
            f"Pivot too far: price {close_curr:.2f} is {price_diff_pct:.2f}% "
            f"away from {level} pivot {level_value:.2f}"
        )

    return True, None

def get_pivot_alert_info(ctx: Dict[str, Any], level: str, is_buy: bool) -> Tuple[bool, Optional[str]]:
    cache_key = f"_pivot_cache_{level}_{'buy' if is_buy else 'sell'}"
    
    if cache_key not in ctx:
        ctx[cache_key] = _validate_pivot_cross(ctx, level, is_buy)
    
    return ctx[cache_key]

BUY_PIVOT_DEFS = [create_pivot_alert(level, is_buy=True) 
                  for level in ("P", "S1", "S2", "S3", "R1", "R2")]

SELL_PIVOT_DEFS = [create_pivot_alert(level, is_buy=False) 
                   for level in ("P", "S1", "S2", "R1", "R2", "R3")]

ALERT_DEFINITIONS.extend(BUY_PIVOT_DEFS)
ALERT_DEFINITIONS.extend(SELL_PIVOT_DEFS)

ALERT_DEFINITIONS_MAP = {d["key"]: d for d in ALERT_DEFINITIONS}

ALERT_KEYS: Dict[str, str] = {
    d["key"]: f"ALERT:{d['key'].upper()}" for d in ALERT_DEFINITIONS
}

logger.debug("Alert keys initialized: %s mappings", len(ALERT_KEYS))
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

def _validate_vwap_cross(ctx: Dict[str, Any], is_buy: bool, previous_states: Dict[str, bool]) -> Tuple[bool, Optional[str]]:
    vwap_curr = ctx.get("vwap_curr")
    vwap_prev = ctx.get("vwap_prev")
    close_curr = ctx["close_curr"]
    close_prev = ctx["close_prev"]
    open_curr = ctx["open_curr"]
    high_curr = ctx["high_curr"]
    low_curr = ctx["low_curr"]
    
    if vwap_curr is None or np.isnan(vwap_curr) or vwap_prev is None or np.isnan(vwap_prev):
        return False, "VWAP not available"
    
    if is_buy:
        if close_curr <= open_curr:
            return False, f"Red candle: O={open_curr:.2f}, C={close_curr:.2f}"
    
        crossed_above = (
            (close_prev <= vwap_prev and close_curr > vwap_curr) or  # Crossed in current candle
            (close_prev < vwap_prev and close_curr >= vwap_curr)  # Allow equality for slow crosses
        )
    
        if not crossed_above:
            return False, (
                f"No cross up: "
                f"prev={close_prev:.2f} vs vwap_prev={vwap_prev:.2f}, "
                f"curr={close_curr:.2f} vs vwap_curr={vwap_curr:.2f}"
            )

        dist_pct = abs(close_curr - vwap_curr) / vwap_curr * 100
        if dist_pct > Constants.VWAP_MAX_DISTANCE_PCT:
            return False, f"Price too far from VWAP: {dist_pct:.2f}% > {Constants.VWAP_MAX_DISTANCE_PCT}%"
        
        candle_range = high_curr - low_curr
        if candle_range > 1e-9:
            body_top = max(open_curr, close_curr)
            upper_wick = high_curr - body_top
            wick_ratio = upper_wick / candle_range
            if wick_ratio >= Constants.MIN_WICK_RATIO:
                return False, f"Upper wick too large: {wick_ratio*100:.1f}% >= {Constants.MIN_WICK_RATIO*100:.1f}%"
        
        return True, None
    
    else:
        if close_curr >= open_curr:
            return False, f"Green candle: O={open_curr:.2f}, C={close_curr:.2f}"
        
        if not (close_prev > vwap_prev and close_curr < vwap_curr):
            return False, (
                f"No cross down: "
                f"prev={close_prev:.2f} vs vwap_prev={vwap_prev:.2f}, "
                f"curr={close_curr:.2f} vs vwap_curr={vwap_curr:.2f}"
            )
        
        dist_pct = abs(close_curr - vwap_curr) / vwap_curr * 100
        if dist_pct > Constants.VWAP_MAX_DISTANCE_PCT:
            return False, f"Price too far from VWAP: {dist_pct:.2f}% > {Constants.VWAP_MAX_DISTANCE_PCT}%"
        
        candle_range = high_curr - low_curr
        if candle_range > 1e-9:
            body_bottom = min(open_curr, close_curr)
            lower_wick = body_bottom - low_curr
            wick_ratio = lower_wick / candle_range
            if wick_ratio >= Constants.MIN_WICK_RATIO:
                return False, f"Lower wick too large: {wick_ratio*100:.1f}% >= {Constants.MIN_WICK_RATIO*100:.1f}%"
        
        return True, None

async def check_multiple_alert_states(sdb: RedisStateStore, pair: str, keys: List[str]) -> Dict[str, bool]:
    if sdb.degraded or not keys:
        return {k: False for k in keys}
    try:
        results = await sdb.batch_get_and_set_alerts(pair, keys, [])
        output: Dict[str, bool] = {}       
        for key in keys:
            st = results.get(key)
            output[key] = isinstance(st, dict) and st.get("state") == "ACTIVE"

        return output

    except Exception as e:
        logger.error(f"check_multiple_alert_states failed for {pair} | keys={len(keys)} | error={e}")
        return {k: False for k in keys}

def check_candle_quality_with_reason(open_val, high_val, low_val, close_val, is_buy, precomputed_ratio: Optional[float] = None) -> Tuple[bool, str]:
    try:
        candle_range = high_val - low_val
        if candle_range < 1e-8:
            return False, "Range too small"

        # Use precomputed wick ratio if provided
        if precomputed_ratio is not None:
            wick_ratio = precomputed_ratio
        else:
            # Fallback: calculate wick ratio based on candle type
            if is_buy:
                upper_wick = high_val - close_val
                if upper_wick < 0:
                    return False, f"Corrupted data (H={high_val:.5f} < C={close_val:.5f})"
                wick_ratio = upper_wick / candle_range
            else:
                lower_wick = close_val - low_val
                if lower_wick < 0:
                    return False, f"Corrupted data (L={low_val:.5f} > C={close_val:.5f})"
                wick_ratio = lower_wick / candle_range

        if is_buy:
            if close_val <= open_val:
                return False, f"Not green (C={close_val:.5f} ≤ O={open_val:.5f})"

            if wick_ratio >= Constants.MIN_WICK_RATIO:
                return False, f"Upper wick {wick_ratio*100:.1f}% ≥ {Constants.MIN_WICK_RATIO*100:.1f}%"
            return True, f"✅ Green wick:{wick_ratio*100:.1f}%"

        else:
            if close_val >= open_val:
                return False, f"Not red (C={close_val:.5f} ≥ O={open_val:.5f})"

            if wick_ratio >= Constants.MIN_WICK_RATIO:
                return False, f"Lower wick {wick_ratio*100:.1f}% ≥ {Constants.MIN_WICK_RATIO*100:.1f}%"
            return True, f"✅ Red wick:{wick_ratio*100:.1f}%"

    except Exception as e:
        return False, f"Error: {str(e)}"

async def evaluate_pair_and_alert(pair_name: str, data_15m: Dict[str, np.ndarray], data_5m: Dict[str, np.ndarray],
    data_daily: Optional[Dict[str, np.ndarray]], sdb: RedisStateStore, telegram_queue: TelegramQueue, correlation_id: str,
    reference_time: int, alignment_cache: Dict[str, int]) -> Optional[Tuple[str, Dict[str, Any]]]:

    logger_pair = logging.getLogger(f"macd_bot.{pair_name}.{correlation_id}")
    PAIR_ID.set(pair_name)
    ALERT_COOLDOWN_SECONDS = 300
    close_15m = None
    open_15m = None
    timestamps_15m = None
    indicators = None
    ppo = None
    ppo_signal = None
    smooth_rsi = None
    vwap = None
    mmh = None
    upw = None
    dnw = None
    rma50_15 = None
    rma200_5 = None
    piv = None
    context = None

    try:
        i15 = get_last_closed_index_from_array(data_15m["timestamp"], 15, reference_time)

        if i15 is None or i15 < 4:
            if cfg.DEBUG_MODE:
                logger_pair.debug(f"Insufficient 15m data: {i15} closed (need >=4)")
            return None

        ts_15m_val = data_15m["timestamp"][i15]
        ts_5m_arr = data_5m["timestamp"]
        ts_curr = int(ts_15m_val)
        
        if ts_curr > 1e10:
            ts_curr = ts_curr // 1000
        
        if not validate_candle_timestamp(ts_curr, reference_time, 15, 120):
            if cfg.DEBUG_MODE:
                logger_pair.debug(f"Skipping {pair_name} - 15m candle not confirmed closed")
            return None
            
        interval_seconds = 15 * 60
        candle_close_ts = ts_curr + interval_seconds
        time_since_close = reference_time - candle_close_ts


        if reference_time - ts_curr > 1200:
            if cfg.DEBUG_MODE:
                logger_pair.debug(f"Skipping {pair_name} - candle too stale ({reference_time - ts_curr}s > 600s)")
            return None
        
        if time_since_close < Constants.CANDLE_PUBLICATION_LAG_SEC:
            if cfg.DEBUG_MODE:
                logger_pair.debug(
                    f"Skipping {pair_name} - candle not finalized ({time_since_close}s < {Constants.CANDLE_PUBLICATION_LAG_SEC}s)"
                )
            return None

        cache_key = f"{pair_name}_{ts_15m_val}"
        if cache_key in alignment_cache:
            i5 = alignment_cache[cache_key]
        else:
            idx = np.searchsorted(ts_5m_arr, ts_15m_val, side='right') - 1
            i5 = int(max(0, min(idx, len(ts_5m_arr) - 1)))
            alignment_cache[cache_key] = i5

        if i5 < 2:
            if cfg.DEBUG_MODE:
                logger_pair.debug(f"Insufficient aligned 5m data: {i5} (need >=2)")
            return None

        close_15m = data_15m["close"]
        open_15m = data_15m["open"]
        timestamps_15m = data_15m["timestamp"]

        close_curr_quick = close_15m[i15]
        open_curr_quick = open_15m[i15]
        is_green = close_curr_quick > open_curr_quick
        is_red = close_curr_quick < open_curr_quick

        if not is_green and not is_red:
            if logger_pair.isEnabledFor(logging.DEBUG):
                logger_pair.debug(
                    f"Doji/neutral candle for {pair_name} "
                    f"(O={open_curr_quick:.2f}, C={close_curr_quick:.2f}), skipping indicators"
                )
            return None
       
        indicators = await asyncio.to_thread(
            calculate_all_indicators_numpy, data_15m, data_5m, data_daily
        )
        
        if indicators is None:
            logger_pair.error(f"Skipping {pair_name}: all indicators failed to calculate")
            return None

        ppo = indicators["ppo"]
        ppo_signal = indicators["ppo_signal"]
        smooth_rsi = indicators["smooth_rsi"]
        vwap = indicators["vwap"]
        mmh = indicators["mmh"]
        upw = indicators["upw"]
        dnw = indicators["dnw"]
        rma50_15 = indicators["rma50_15"]
        rma200_5 = indicators["rma200_5"]
        piv = indicators["pivots"]
        close_curr = close_15m[i15]
        close_prev = close_15m[i15 - 1]
        open_curr = open_15m[i15]
        high_curr = data_15m["high"][i15]
        low_curr = data_15m["low"][i15]
        open_prev = open_15m[i15 - 1] if i15 >= 1 else open_curr
        high_prev = data_15m["high"][i15 - 1] if i15 >= 1 else high_curr
        low_prev = data_15m["low"][i15 - 1] if i15 >= 1 else low_curr
        close_5m_val = data_5m["close"][i5]
        ppo_curr = ppo[i15]
        ppo_prev = ppo[i15 - 1] if i15 >= 1 else ppo[i15]
        ppo_sig_curr = ppo_signal[i15]
        ppo_sig_prev = ppo_signal[i15 - 1] if i15 >= 1 else ppo_signal[i15]
        rsi_curr = smooth_rsi[i15]
        rsi_prev = smooth_rsi[i15 - 1] if i15 >= 1 else smooth_rsi[i15]
        vwap_enabled = cfg.ENABLE_VWAP
        vwap_available = False
        vwap_curr = None
        vwap_prev = None

        if vwap_enabled and vwap is not None and len(vwap) > i15:
            try:
                vwap_curr = vwap[i15]
                vwap_prev = vwap[i15 - 1] if i15 >= 1 else vwap[i15]

                if not (np.isnan(vwap_curr) or np.isnan(vwap_prev) or vwap_curr <= 0 or vwap_prev <= 0):
                    vwap_available = True
                else:
                    if cfg.DEBUG_MODE:
                        logger_pair.debug(f"VWAP data invalid: curr={vwap_curr}, prev={vwap_prev}")
                    vwap_curr = None
                    vwap_prev = None
            except (IndexError, TypeError) as e:
                logger_pair.warning(f"Error accessing VWAP data: {e}")
                vwap_curr = None
                vwap_prev = None
        else:
            if vwap_enabled and cfg.DEBUG_MODE:
                logger_pair.debug(
                    f"VWAP unavailable: enabled={vwap_enabled}, "
                    f"vwap_is_none={vwap is None}, len={len(vwap) if vwap is not None else 0}, "
                    f"i15={i15}"
                )

        mmh_curr = mmh[i15]
        mmh_m1 = mmh[i15 - 1] if i15 >= 1 else 0.0
        mmh_m2 = mmh[i15 - 2] if i15 >= 2 else 0.0
        mmh_m3 = mmh[i15 - 3] if i15 >= 3 else 0.0

        MIN_MMH_BARS_VALID = 160
        has_valid_mmh = (
            i15 >= MIN_MMH_BARS_VALID and
            not np.isnan(mmh_curr) and 
            not np.isnan(mmh_m1) and 
            not np.isnan(mmh_m2) and 
            not np.isnan(mmh_m3)
        )

        if not has_valid_mmh:
            if cfg.DEBUG_MODE:
                skip_reason = (
                    f"MMH warmup" if i15 < MIN_MMH_BARS_VALID 
                    else f"MMH NaN (idx={i15})"
                )
                logger_pair.debug(f"Skipping MMH alerts: {skip_reason}")

        rma50_15_val = rma50_15[i15]
        rma200_5_val = rma200_5[i5]

        
        cloud_up = bool(upw[i15]) and not bool(dnw[i15])
        cloud_down = bool(dnw[i15]) and not bool(upw[i15])

        candle_range = high_curr - low_curr
        if candle_range <= 1e-9:
            actual_buy_wick_ratio = 1.0
            actual_sell_wick_ratio = 1.0
        else:        
            upper_wick = high_curr - close_curr
            actual_buy_wick_ratio = upper_wick / candle_range

            lower_wick = close_curr - low_curr
            actual_sell_wick_ratio = lower_wick / candle_range

        if cfg.DEBUG_MODE:
            logger_pair.debug(
                f"PHASE 7 [15M]: O={open_curr:.5f} H={high_curr:.5f} L={low_curr:.5f} C={close_curr:.5f}"
            )
            logger_pair.debug(
                f"Wick (15M): Buy={actual_buy_wick_ratio*100:.2f}% Sell={actual_sell_wick_ratio*100:.2f}% "
                f"(threshold={Constants.MIN_WICK_RATIO*100:.1f}%)"
            )

        if cfg.ENABLE_PIVOT or cfg.ENABLE_VWAP:
            current_utc_dt = datetime.fromtimestamp(reference_time, tz=timezone.utc)
            current_date = current_utc_dt.date()
            day_tracker_key = f"{pair_name}:last_reset_date"

            last_reset_date = None
            last_reset_date_str = None
            try:
                last_reset_date_str = await sdb.get_metadata(day_tracker_key)
                if last_reset_date_str:
                    last_reset_date = datetime.fromisoformat(last_reset_date_str).date()
            except Exception as e:
                logger_pair.warning(f"Failed to parse last_reset_date '{last_reset_date_str}': {e}")
                last_reset_date = None

            logger_pair.debug(
                f"Daily reset check | stored='{last_reset_date_str}' | "
                f"parsed={last_reset_date} | current={current_date} | "
                f"needs_reset={last_reset_date != current_date}"
            )

            if last_reset_date != current_date:
                try:
                    await sdb.set_metadata(day_tracker_key, current_date.isoformat())
                    logger_pair.debug(f"✅ Saved reset date: {current_date}")
                except Exception as e:
                    logger_pair.error(f"❌ Failed to save reset date – aborting daily reset: {e}")
                    raise

                delete_keys = []
                if cfg.ENABLE_PIVOT:
                    pivot_alerts = [
                        "pivot_up_P", "pivot_up_S1", "pivot_up_S2", "pivot_up_S3",
                        "pivot_up_R1", "pivot_up_R2",
                        "pivot_down_P", "pivot_down_S1", "pivot_down_S2",
                        "pivot_down_S3", "pivot_down_R1", "pivot_down_R2", "pivot_down_R3"
                    ]
                    for alert_key in pivot_alerts:
                        redis_key = ALERT_KEYS.get(alert_key)
                        if redis_key:
                            delete_keys.append(f"{pair_name}:{redis_key}")

                if cfg.ENABLE_VWAP:
                    vwap_alerts = ["vwap_up", "vwap_down"]
                    for alert_key in vwap_alerts:
                        redis_key = ALERT_KEYS.get(alert_key)
                        if redis_key:
                            delete_keys.append(f"{pair_name}:{redis_key}")

                if delete_keys:
                    await sdb.atomic_batch_update([], deletes=delete_keys)
                    logger_pair.info(f"🔄 Daily reset on {current_date}. Cleared {len(delete_keys)} alerts")
                else:
                    logger_pair.debug(f"🔄 Daily reset on {current_date} (no alerts to clear)")
        else:
            logger_pair.debug("Daily reset disabled (ENABLE_PIVOT and ENABLE_VWAP both false)")

        base_buy_trend = (rma50_15_val < close_curr) and (rma200_5_val < close_5m_val)
        base_sell_trend = (rma50_15_val > close_curr) and (rma200_5_val > close_5m_val)

        if base_buy_trend:
            base_buy_trend = base_buy_trend and (mmh_curr > 0) and cloud_up
        if base_sell_trend:
            base_sell_trend = base_sell_trend and (mmh_curr < 0) and cloud_down

        buy_quality_arr, sell_quality_arr = precompute_candle_quality(
            data_15m, indicators["atr_short"], indicators["atr_long"], cfg.RVOL_THRESHOLD,
            cfg.ENABLE_RVOL_ALERT)

        buy_candle_passed = bool(buy_quality_arr[i15])
        sell_candle_passed = bool(sell_quality_arr[i15])

        buy_candle_reason = None
        sell_candle_reason = None

        if base_buy_trend and not buy_candle_passed:
            _, buy_candle_reason = check_candle_quality_with_reason(
                open_curr, high_curr, low_curr, close_curr, is_buy=True
            )

        if base_sell_trend and not sell_candle_passed:
            _, sell_candle_reason = check_candle_quality_with_reason(
                open_curr, high_curr, low_curr, close_curr, is_buy=False
            )

        buy_common = base_buy_trend and buy_candle_passed
        sell_common = base_sell_trend and sell_candle_passed

        if not has_valid_mmh:
            if cfg.DEBUG_MODE:
                logger_pair.debug(
                    f"Skipping MMH alerts for {pair_name}: "
                    f"warmup period (NaN values detected)"
                )
            mmh_reversal_buy = False
            mmh_reversal_sell = False
        else:
            mmh_reversal_buy = (buy_common and mmh_curr > 0 and mmh_m3 > mmh_m2 > mmh_m1 and mmh_curr > mmh_m1)
            
            mmh_reversal_sell = (sell_common and mmh_curr < 0 and mmh_m3 < mmh_m2 < mmh_m1 and mmh_curr < mmh_m1)

        context = {
            "close_curr": close_curr,
            "close_prev": close_prev,
            "open_curr": open_curr,
            "high_curr": high_curr,
            "low_curr": low_curr,
            "ts_curr": ts_curr,
            "close_5m_val": close_5m_val,
            "ppo_curr": ppo_curr,
            "ppo_prev": ppo_prev,
            "ppo_sig_curr": ppo_sig_curr,
            "ppo_sig_prev": ppo_sig_prev,
            "rsi_curr": rsi_curr,
            "rsi_prev": rsi_prev,
            "vwap_curr": vwap_curr,
            "vwap_prev": vwap_prev,
            "vwap_available": vwap_available,
            "vwap_enabled": cfg.ENABLE_VWAP and vwap_available,
            "mmh_curr": mmh_curr,
            "mmh_m1": mmh_m1,
            "mmh_m2": mmh_m2,
            "mmh_m3": mmh_m3,
            "mmh_reversal_buy": mmh_reversal_buy,
            "mmh_reversal_sell": mmh_reversal_sell,
            "rma50_15_val": rma50_15_val,
            "rma200_5_val": rma200_5_val,
            "cloud_up": cloud_up,
            "cloud_down": cloud_down,
            "candle_quality_failed_buy": base_buy_trend and not buy_candle_passed,
            "candle_quality_failed_sell": base_sell_trend and not sell_candle_passed,
            "buy_wick_ratio": actual_buy_wick_ratio,
            "sell_wick_ratio": actual_sell_wick_ratio,
            "is_green": is_green,
            "is_red": is_red,
            "buy_common": buy_common,
            "sell_common": sell_common,
            "wick_ratio_timeframe": "15m",
            "pivots": piv if piv else {},
        }

        context["pivot_suppressions"] = []

        ppo_ctx = {"curr": ppo_curr, "prev": ppo_prev}
        ppo_sig_ctx = {"curr": ppo_sig_curr, "prev": ppo_sig_prev}
        rsi_ctx = {"curr": rsi_curr, "prev": rsi_prev}

        raw_alerts = []
        
        alert_keys_to_check = [
            d["key"] for d in ALERT_DEFINITIONS
            if not (
                ("pivots" in d["requires"] and (not cfg.ENABLE_PIVOT or not piv or not any(piv.values()))) or
                ("vwap" in d["requires"] and (not cfg.ENABLE_VWAP or not vwap_available)) or
                ("ppo" in d["requires"] and (ppo_ctx is None)) or
                ("ppo_signal" in d["requires"] and (ppo_sig_ctx is None)) or
                ("rsi" in d["requires"] and (rsi_ctx is None))
            )
        ]

        if not piv or not any(piv.values()):
            alert_keys_to_check = [
                k for k in alert_keys_to_check
                if not k.startswith("pivot_")
            ]

        state_store_alert_keys = [ALERT_KEYS[k] for k in alert_keys_to_check]

        previous_states = await check_multiple_alert_states(
            sdb, pair_name, state_store_alert_keys
        )

        all_state_changes = []

        for alert_key in alert_keys_to_check:
            def_ = ALERT_DEFINITIONS_MAP.get(alert_key)
            if not def_:
                continue

            key = ALERT_KEYS[alert_key]
            trigger = False

            is_buy_signal = any(x in alert_key for x in ["up", "buy"])
            is_sell_signal = any(x in alert_key for x in ["down", "sell"])

            if alert_key.startswith("pivot_up_") or alert_key.startswith("pivot_down_"):
                level = alert_key.split("_")[-1]
                is_buy = alert_key.startswith("pivot_up_")

                try:
                    valid_cross, reason = _validate_pivot_cross(context, level, is_buy)

                    if not valid_cross and reason and piv:
                        context["pivot_suppressions"].append(f"{alert_key}: {reason}")

                    trigger = (
                        (is_buy and buy_common) or (not is_buy and sell_common)
                    ) and valid_cross
                except Exception as e:
                    logger_pair.error(
                        f"Pivot alert check failed for {alert_key}: {e}",
                        exc_info=True
                    )
                    trigger = False

            elif alert_key in ("vwap_up", "vwap_down"):
                if not vwap_available:
                    if cfg.DEBUG_MODE:
                        logger_pair.debug(f"Skipping {alert_key}: VWAP data unavailable")
                    continue

                trigger = False
                try:
                    trigger = def_["check_fn"](context, ppo_ctx, ppo_sig_ctx, rsi_ctx)
        
                    if cfg.DEBUG_MODE:
                        is_buy = (alert_key == "vwap_up")
                        valid_cross, reason = _validate_vwap_cross(context, is_buy, previous_states)
                        if trigger:
                            logger_pair.debug(
                                f"✅ {alert_key}: Close={context['close_curr']:.2f}, "
                                f"VWAP={context['vwap_curr']:.2f}, "
                                f"buy_common={context.get('buy_common', False)}, "
                                f"sell_common={context.get('sell_common', False)}"
                            )
                        else:
                            logger_pair.debug(f"❌ {alert_key}: buy_common or sell_common not met")
    
                except Exception as e:
                    logger_pair.error(f"VWAP check failed for {alert_key}: {e}", exc_info=True)
                    trigger = False
            else:
                trigger = False
                trigger_error = None
                
                try:
                    trigger = def_["check_fn"](context, ppo_ctx, ppo_sig_ctx, rsi_ctx)
                except Exception as e:
                    if isinstance(e, KeyError):
                        logger_pair.error(f"Missing context key for {alert_key}: {e}")
                    elif isinstance(e, TypeError):
                        logger_pair.error(f"Type error in {alert_key}: {e}")
                    else:
                        logger_pair.error(f"Alert check failed for {alert_key}: {e}", exc_info=True)
                    trigger = False
                    trigger_error = str(e)

            if trigger and not previous_states.get(key, False):
                extra = ""
                try:
                    extra = def_["extra_fn"](context, ppo_ctx, ppo_sig_ctx, rsi_ctx, None) or ""
                except Exception as e:
                    logger_pair.error(
                        f"Alert extra_fn failed for {alert_key}, firing alert without extra details: {e}",
                        exc_info=False
                    )
                    extra = f"(Error: {str(e)[:50]})"

                raw_alerts.append((def_["title"], extra, def_["key"]))
                all_state_changes.append((f"{pair_name}:{key}", "ACTIVE", None))

                if cfg.DEBUG_MODE:
                    logger_pair.debug(
                        f"✅ Alert FIRED: {alert_key} | "
                        f"buy_common={buy_common} sell_common={sell_common} | "
                        f"buy_passed={buy_candle_passed} sell_passed={sell_candle_passed} | "
                        f"Wick ratios: buy={actual_buy_wick_ratio*100:.1f}% sell={actual_sell_wick_ratio*100:.1f}% | "
                        f"Candle: O={open_curr:.2f} H={high_curr:.2f} L={low_curr:.2f} C={close_curr:.2f}"
                    )

        resets_to_apply = []

        if ppo_prev > ppo_sig_prev and ppo_curr <= ppo_sig_curr:
            resets_to_apply.append((f"{pair_name}:{ALERT_KEYS['ppo_signal_up']}", "INACTIVE", None))

        if ppo_prev < ppo_sig_prev and ppo_curr >= ppo_sig_curr:
            resets_to_apply.append((f"{pair_name}:{ALERT_KEYS['ppo_signal_down']}", "INACTIVE", None))

        if ppo_prev > 0 and ppo_curr <= 0:
            resets_to_apply.append((f"{pair_name}:{ALERT_KEYS['ppo_zero_up']}", "INACTIVE", None))

        elif not buy_common:
            if await was_alert_active(sdb, pair_name, ALERT_KEYS['ppo_zero_up']):
                resets_to_apply.append((f"{pair_name}:{ALERT_KEYS['ppo_zero_up']}", "INACTIVE", None))
           
        if ppo_prev < 0 and ppo_curr >= 0:
            resets_to_apply.append((f"{pair_name}:{ALERT_KEYS['ppo_zero_down']}", "INACTIVE", None))

        elif not sell_common:
            if await was_alert_active(sdb, pair_name, ALERT_KEYS['ppo_zero_down']):
                resets_to_apply.append((f"{pair_name}:{ALERT_KEYS['ppo_zero_down']}", "INACTIVE", None))

        if ppo_prev > Constants.PPO_011_THRESHOLD and ppo_curr <= Constants.PPO_011_THRESHOLD:
            resets_to_apply.append((f"{pair_name}:{ALERT_KEYS['ppo_011_up']}", "INACTIVE", None))

        elif not buy_common:
            if await was_alert_active(sdb, pair_name, ALERT_KEYS['ppo_011_up']):
                resets_to_apply.append((f"{pair_name}:{ALERT_KEYS['ppo_011_up']}", "INACTIVE", None)) 
         
        if ppo_prev < Constants.PPO_011_THRESHOLD_SELL and ppo_curr >= Constants.PPO_011_THRESHOLD_SELL:
            resets_to_apply.append((f"{pair_name}:{ALERT_KEYS['ppo_011_down']}", "INACTIVE", None))

        elif not sell_common:
            if await was_alert_active(sdb, pair_name, ALERT_KEYS['ppo_011_down']):
                resets_to_apply.append((f"{pair_name}:{ALERT_KEYS['ppo_011_down']}", "INACTIVE", None))

        if rsi_prev > Constants.RSI_THRESHOLD and rsi_curr <= Constants.RSI_THRESHOLD:
            resets_to_apply.append((f"{pair_name}:{ALERT_KEYS['rsi_50_up']}", "INACTIVE", None))

        elif not buy_common or ppo_curr >= Constants.PPO_RSI_GUARD_BUY:
            if await was_alert_active(sdb, pair_name, ALERT_KEYS['rsi_50_up']):
                resets_to_apply.append((f"{pair_name}:{ALERT_KEYS['rsi_50_up']}", "INACTIVE", None))
                
        if rsi_prev < Constants.RSI_THRESHOLD and rsi_curr >= Constants.RSI_THRESHOLD:
            resets_to_apply.append((f"{pair_name}:{ALERT_KEYS['rsi_50_down']}", "INACTIVE", None))

        elif not sell_common or ppo_curr <= Constants.PPO_RSI_GUARD_SELL:
            if await was_alert_active(sdb, pair_name, ALERT_KEYS['rsi_50_down']):
                resets_to_apply.append((f"{pair_name}:{ALERT_KEYS['rsi_50_down']}", "INACTIVE", None))
                
        if cfg.ENABLE_VWAP and vwap_available:
            if close_prev > vwap_prev and close_curr <= vwap_curr:
                resets_to_apply.append((f"{pair_name}:{ALERT_KEYS['vwap_up']}", "INACTIVE", None))

            elif not buy_common:
                if await was_alert_active(sdb, pair_name, ALERT_KEYS['vwap_up']):
                    resets_to_apply.append((f"{pair_name}:{ALERT_KEYS['vwap_up']}", "INACTIVE", None))
                    
            if close_prev < vwap_prev and close_curr >= vwap_curr:
                resets_to_apply.append((f"{pair_name}:{ALERT_KEYS['vwap_down']}", "INACTIVE", None))
            elif not sell_common:
                if await was_alert_active(sdb, pair_name, ALERT_KEYS['vwap_down']):
                    resets_to_apply.append((f"{pair_name}:{ALERT_KEYS['vwap_down']}", "INACTIVE", None))
                   
        if piv:
            for level_name, level_value in piv.items():
                up_key = f"pivot_up_{level_name}"
                if up_key in ALERT_KEYS:
                    if close_prev > level_value and close_curr <= level_value:
                        resets_to_apply.append((f"{pair_name}:{ALERT_KEYS[up_key]}", "INACTIVE", None))

                down_key = f"pivot_down_{level_name}"
                if down_key in ALERT_KEYS:
                    if close_prev < level_value and close_curr >= level_value:
                        resets_to_apply.append((f"{pair_name}:{ALERT_KEYS[down_key]}", "INACTIVE", None))

        if (mmh_curr > 0) and (mmh_curr <= mmh_m1):
            if await was_alert_active(sdb, pair_name, ALERT_KEYS["mmh_buy"]):
                resets_to_apply.append((f"{pair_name}:{ALERT_KEYS['mmh_buy']}", "INACTIVE", None))
        if (mmh_curr < 0) and (mmh_curr >= mmh_m1):
            if await was_alert_active(sdb, pair_name, ALERT_KEYS["mmh_sell"]):
                resets_to_apply.append((f"{pair_name}:{ALERT_KEYS['mmh_sell']}", "INACTIVE", None))

        all_state_changes.extend(resets_to_apply)

        if all_state_changes:
            await sdb.atomic_batch_update(all_state_changes)

        alerts_to_send = raw_alerts[:cfg.MAX_ALERTS_PER_PAIR]

        pivot_count = sum(1 for _, _, k in alerts_to_send if k.startswith("pivot_"))
        if pivot_count > 3:
            logger_pair.warning(
                f"Limiting pivot alerts for {pair_name}: {pivot_count} triggered, keeping 3"
            )
            pivot_alerts = [(t, e, k) for t, e, k in alerts_to_send if k.startswith("pivot_")][:3]
            other_alerts = [(t, e, k) for t, e, k in alerts_to_send if not k.startswith("pivot_")]
            alerts_to_send = other_alerts + pivot_alerts

        filtered_alerts = []
        for alert_title, alert_extra, alert_key in alerts_to_send:
            cooldown_key = f"{pair_name}:{alert_key}:last_sent"
      
            if not sdb.degraded:
                try:
                    last_sent_str = await sdb.get_metadata(cooldown_key)
                    if last_sent_str:
                        last_ts = int(float(last_sent_str))
                        elapsed = ts_curr - last_ts
                        if elapsed < ALERT_COOLDOWN_SECONDS:
                            logger_pair.debug(
                                f"Alert {alert_key} cooldown: {ALERT_COOLDOWN_SECONDS - elapsed:.0f}s remaining"
                            )
                            continue
                except (ValueError, TypeError):
                    pass
    
            filtered_alerts.append((alert_title, alert_extra, alert_key))
    
            if not sdb.degraded:
                try:
                    await sdb.set_metadata(cooldown_key, str(ts_curr))
                except Exception as e:
                    logger_pair.debug(f"Failed to set cooldown: {e}")

        alerts_to_send = filtered_alerts

        if alerts_to_send:
            try:
                if len(alerts_to_send) == 1:
                    title, extra, _ = alerts_to_send[0]
                    msg = build_single_msg(title, pair_name, close_curr, ts_curr, extra)
                else:
                    items = [(t, e) for t, e, _ in alerts_to_send[:25]]
                    msg = build_batched_msg(pair_name, close_curr, ts_curr, items)

                if not cfg.DRY_RUN_MODE:
                    send_success = await telegram_queue.send(msg)
                    if not send_success:
                        logger_pair.error(f"Alert dispatch failed | {pair_name}")
                else:
                    logger_pair.info(f"[DRY RUN] Would send: {msg[:100]}...")

                logger_pair.info(
                    f"🔔🎯🟢 Sent {len(alerts_to_send)} alerts for {pair_name} | "
                    f"Keys: {[ak for _, _, ak in alerts_to_send]}"
                )
            except Exception as e:
                logger_pair.error(f"Error sending alerts: {e}", exc_info=False)

        reasons = []
        if not buy_common and not sell_common:
            reasons.append("Trend filter blocked")

        if context.get("candle_quality_failed_buy") and buy_candle_reason:
            reasons.append(f"BUY quality: {buy_candle_reason}")

        if context.get("candle_quality_failed_sell") and sell_candle_reason:
            reasons.append(f"SELL quality: {sell_candle_reason}")

        if context.get("pivot_suppressions"):
            reasons.extend(context["pivot_suppressions"])

        if ppo_prev <= 0 and ppo_curr > 0 and not buy_common:
            if not base_buy_trend:
                reasons.append("PPO>0 blocked: base_buy_trend=False")
            elif not buy_candle_passed:
                reasons.append("PPO>0 blocked: candle quality failed")
            elif not is_green:
                reasons.append(f"PPO>0 blocked: candle not green (C={close_curr:.5f}, O={open_curr:.5f})")

        if ppo_prev >= 0 and ppo_curr < 0 and not sell_common:
            if not base_sell_trend:
                reasons.append("PPO<0 blocked: base_sell_trend=False")
            elif not sell_candle_passed:
                reasons.append("PPO<0 blocked: candle quality failed")
            elif not is_red:
                reasons.append(f"PPO<0 blocked: candle not red (C={close_curr:.5f}, O={open_curr:.5f})")

        if ppo_prev <= Constants.PPO_011_THRESHOLD and ppo_curr > Constants.PPO_011_THRESHOLD and not buy_common:
            reasons.append("PPO>+0.11 blocked: buy_common=False")

        if ppo_prev >= Constants.PPO_011_THRESHOLD_SELL and ppo_curr < Constants.PPO_011_THRESHOLD_SELL and not sell_common:
            reasons.append("PPO<-0.11 blocked: sell_common=False")

        if rsi_prev <= Constants.RSI_THRESHOLD and rsi_curr > Constants.RSI_THRESHOLD:
            if ppo_curr >= Constants.PPO_RSI_GUARD_BUY:
                reasons.append(f"RSI>50 blocked: PPO={ppo_curr:.2f} ≥ guard {Constants.PPO_RSI_GUARD_BUY}")
            elif not buy_common:
                reasons.append("RSI>50 blocked: buy_common=False")

        if rsi_prev >= Constants.RSI_THRESHOLD and rsi_curr < Constants.RSI_THRESHOLD:
            if ppo_curr <= Constants.PPO_RSI_GUARD_SELL:
                reasons.append(f"RSI<50 blocked: PPO={ppo_curr:.2f} ≤ guard {Constants.PPO_RSI_GUARD_SELL}")
            elif not sell_common:
                reasons.append("RSI<50 blocked: sell_common=False")

        if cfg.ENABLE_VWAP and vwap_available:
            if close_prev <= vwap_prev and close_curr > vwap_curr and not buy_common:
                reasons.append("VWAP up-cross blocked: buy_common=False")
            if close_prev >= vwap_prev and close_curr < vwap_curr and not sell_common:
                reasons.append("VWAP down-cross blocked: sell_common=False")

        failed_conditions = [
            name for name, val in [
                ("buy_common", buy_common),
                ("sell_common", sell_common),
                ("is_green", is_green),
                ("is_red", is_red)
            ] if not val
        ]

        if not alerts_to_send:
            cloud_state = "green" if cloud_up else "red" if cloud_down else "neutral"

            logger_pair.debug(
                f"😒 {pair_name} | "
                f"cloud={cloud_state} mmh={mmh_curr:.2f} | "
                f"Suppression: {', '.join(failed_conditions + reasons) if (failed_conditions or reasons) else 'No conditions met'}"
            )

        return pair_name, {
            "state": "ALERT_SENT" if alerts_to_send else "NO_SIGNAL",
            "ts": int(time.time()),
            "summary": {
                "alerts": len(alerts_to_send),
                "cloud": "green" if cloud_up else "red" if cloud_down else "neutral",
                "mmh_hist": round(mmh_curr, 4),
                "suppression": ", ".join(failed_conditions + reasons) if (failed_conditions or reasons) else "No conditions met"
            }
        }

    except asyncio.CancelledError:
        logger_pair.warning(f"Evaluation cancelled for {pair_name}")
        raise

    except Exception as e:
        logger_pair.exception(
            f"❌ Error in evaluate_pair_and_alert for {pair_name}: {e} | "
            f"Correlation: {correlation_id}"
        )
        return None

    finally:
        PAIR_ID.set("")
        if data_15m is not None:
            data_15m = None
        if data_5m is not None:
            data_5m = None
        if data_daily is not None:
            data_daily = None
        if indicators is not None:
            indicators = None
        if context is not None:
            context = None
    try:
        process = psutil.Process() 
        current_memory_mb = process.memory_info().rss / 1024 / 1024
        memory_limit_mb = cfg.MEMORY_LIMIT_BYTES / 1024 / 1024        
     
        if current_memory_mb > (memory_limit_mb * 0.8):
            logger_pair.warning(
                f"Memory spike: {current_memory_mb:.0f}MB / {memory_limit_mb:.0f}MB"
            )
    except Exception:
        pass

logger_main = logging.getLogger("macd_bot.worker_pool")
    
async def guarded_eval(task_data, state_db, telegram_queue, correlation_id, reference_time, alignment_cache: Dict[str, int]):  
    p_name, candles = task_data
    data_15m = None
    data_5m = None
    data_daily = None
    
    try:
        data_15m = parse_candles_to_numpy(candles.get("15"))
        data_5m = parse_candles_to_numpy(candles.get("5"))
        data_daily = parse_candles_to_numpy(candles.get("D")) if cfg.ENABLE_PIVOT else None
        
        if data_15m is None:
            logger_main.warning(f"Skipping {p_name}: 15m parse failed")
            return None
        
        if data_5m is None:
            logger_main.warning(f"Skipping {p_name}: 5m parse failed")
            return None
        
        reference_time = get_trigger_timestamp()
        
        i15 = get_last_closed_index_from_array(data_15m["timestamp"], 15, reference_time)
        if i15 is None or i15 < 4:
            if cfg.DEBUG_MODE:
                logger_main.debug(f"Skipping {p_name}: insufficient closed candles (i15={i15})")
            return None
        
        v15_selected, r15_selected = validate_candle_data_at_index(
            data_15m, i15, reference_time, interval_minutes=15
        )
        if not v15_selected:
            logger_main.warning(f"Skipping {p_name}: selected candle invalid ({r15_selected})")
            return None
        
        result = await evaluate_pair_and_alert(
            p_name, data_15m, data_5m, data_daily,
            state_db, telegram_queue, correlation_id, reference_time, alignment_cache
        )
        
        return result
    
    except asyncio.CancelledError:
        logger_main.warning(f"Evaluation cancelled for {p_name}")
        raise
    
    except Exception as e:
        logger_main.error(f"Error in {p_name} evaluation: {e}", exc_info=False)
        return None
    
    finally:
        data_15m = None
        data_5m = None
        data_daily = None
    pass
        
async def process_pairs_with_workers(fetcher: DataFetcher, products_map: Dict[str, dict],
    pairs_to_process: List[str], state_db: RedisStateStore, telegram_queue: TelegramQueue, 
    correlation_id: str, lock: SimpleLock, reference_time: int) -> List[Tuple[str, Dict[str, Any]]]:

    alignment_cache: Dict[str, int] = {}

    logger_main.info(f"🔡 Phase 1: Fetching candles for {len(pairs_to_process)} pairs...")
    fetch_start = time.time()

    limit_15m = Constants.MIN_CANDLES_FOR_INDICATORS + Constants.CANDLE_SAFETY_BUFFER
    limit_5m = Constants.MIN_CANDLES_FOR_INDICATORS + Constants.CANDLE_SAFETY_BUFFER
    daily_limit = cfg.PIVOT_LOOKBACK_PERIOD if cfg.ENABLE_PIVOT else 0

    pair_requests = []
    valid_tasks = []
    for pair_name in pairs_to_process:
        product_info = products_map.get(pair_name)
        if not product_info:
            continue

        symbol = product_info["symbol"]
        resolutions = [("15", limit_15m), ("5", limit_5m)]
        if cfg.ENABLE_PIVOT:
            resolutions.append(("D", daily_limit))

        pair_requests.append((symbol, resolutions))
        # valid_tasks will be populated after fetch
        valid_tasks.append((pair_name, symbol))

    all_candles = await fetcher.fetch_all_candles_truly_parallel(
        pair_requests, reference_time
    )

    fetch_elapsed = time.time() - fetch_start
    logger_main.info(f"✅ Phase 1 complete: {fetch_elapsed:.1f}s")

    logger_main.debug("⚙️ Phase 2: Preparing evaluation tasks...")

    prepared_tasks = []
    for pair_name, symbol in valid_tasks:
        candles = all_candles.get(symbol, {})
        prepared_tasks.append((pair_name, candles))

    logger_main.debug(f"Ready to evaluate {len(prepared_tasks)} pairs")

    logger_main.debug(f"🧠 Phase 3: Evaluating {len(prepared_tasks)} pairs...")
    eval_start = time.time()

    results = await asyncio.gather( 
        *[ 
            guarded_eval(
                t, state_db, telegram_queue, correlation_id, 
                reference_time, alignment_cache, 
            ) for t in prepared_tasks], 
        return_exceptions=True, 
    )

    eval_elapsed = time.time() - eval_start
    logger_main.debug(f"Evaluation complete: {eval_elapsed:.1f}s")

    valid_results = []
    for r in results:
        if isinstance(r, Exception):
            logger_main.warning(f"Evaluation raised exception: {r}")
            continue
        if r is not None:
            valid_results.append(r)

    logger_main.debug(
        f"Results: {len(valid_results)} successful, {len(results) - len(valid_results)} failed"
    )

    results = None
    all_candles = None
    prepared_tasks = None
    pair_requests = None
    valid_tasks = None

    process = psutil.Process()

    def log_memory_usage(stage: str):
        try:
            mem_mb = process.memory_info().rss / 1024 / 1024
            limit_mb = cfg.MEMORY_LIMIT_BYTES / 1024 / 1024
            usage_pct = (mem_mb / limit_mb) * 100
            if cfg.DEBUG_MODE:
                logger_main.debug(
                    f"{stage}: {mem_mb:.0f}MB / {limit_mb:.0f}MB ({usage_pct:.0f}%)"
                )
            return mem_mb, limit_mb, usage_pct
        except Exception as e:
            logger_main.debug(f"Memory reporting failed at {stage}: {e}")
            return None, None, None

    peak_memory_mb, limit_mb, usage_pct = log_memory_usage("⚠️ Peak memory after batch")
    if peak_memory_mb and peak_memory_mb > limit_mb * 0.7:
        logger_main.warning(
            f"⚠️ High memory after batch: {peak_memory_mb:.0f}MB / {limit_mb:.0f}MB "
            f"({usage_pct:.0f}%)"
        )

    logger_main.debug("🧹 Released all fetch-phase data (all_candles, results, etc)")
    gc.collect()

    current_memory_mb, limit_mb, usage_pct = log_memory_usage("💾 Memory after batch cleanup")
    if current_memory_mb and current_memory_mb > limit_mb * 0.8:
        logger_main.warning(
            f"⚠️ Memory still high after cleanup: {current_memory_mb:.0f}MB ({usage_pct:.0f}%). "
            f"Possible memory leak?"
        )

    logger_main.info(
        f"🧠 Worker pool complete: {len(valid_results)}/{len(pairs_to_process)} pairs evaluated"
    )

    return valid_results

async def run_once() -> bool:
    MAX_ALERTS_PER_RUN = 50
    all_results: List[Tuple[str, Dict[str, Any]]] = []
    correlation_id = uuid.uuid4().hex[:8]
    TRACE_ID.set(correlation_id)
    logger_run = logging.getLogger(f"macd_bot.run.{correlation_id}")
    start_time = time.time()
    sdb: Optional[RedisStateStore] = None
    lock: Optional[SimpleLock] = None
    fetcher: Optional[DataFetcher] = None
    telegram_queue: Optional[TelegramQueue] = None
    lock_acquired = False
    lock_extension_task: Optional[asyncio.Task] = None
    alerts_sent = 0
    
    products_map: Optional[Dict[str, dict]] = None
    pairs_to_process: List[str] = []
    
    reference_time = get_trigger_timestamp()
    logger_run.info(
        f"🎯 Run started | Correlation ID: {correlation_id} | "
        f"Reference time: {reference_time} ({format_ist_time(reference_time)})"
    )

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

        logger_run.debug("📦 Initializing HTTP fetcher...")
        fetcher = DataFetcher(cfg.DELTA_API_BASE)
        pairs_to_process = list(cfg.PAIRS)
        products_map = build_products_map_from_cfg()

        if not pairs_to_process:
            logger_run.error("❌ No pairs configured - aborting")
            return False

        logger_run.info(f"🔄 Processing {len(pairs_to_process)} pairs from config")

        logger_run.debug("Connecting to GitHub...")
        sdb = RedisStateStore(cfg.MACD_BOT_TOKEN, cfg.MACD_REPO, cfg.MACD_BRANCH)
        await sdb.connect()

        if sdb.degraded and not sdb.degraded_alerted:
            logger_run.critical(
                "🚨 GitHub is in degraded mode – alert deduplication disabled!"
            )
            telegram_queue = TelegramQueue(cfg.TELEGRAM_BOT_TOKEN, cfg.TELEGRAM_CHAT_ID)
            await telegram_queue.send(escape_markdown_v2(
                f"⚠️ {cfg.BOT_NAME} - GITHUB DEGRADED MODE\n"
                f"Alert deduplication is disabled. You may receive duplicate alerts.\n"
                f"Time: {format_ist_time()}"
            ))
            sdb.degraded_alerted = True

        if telegram_queue is None:
            telegram_queue = TelegramQueue(cfg.TELEGRAM_BOT_TOKEN, cfg.TELEGRAM_CHAT_ID)

        lock = SimpleLock()
        lock_acquired = await lock.acquire(timeout=5.0)

        if not lock_acquired:
            logger_run.warning(
                "⏸️ Another instance is running (GitHub lock held) - exiting gracefully"
            )
            return False

        async def extend_lock_periodically(lock_obj: SimpleLock, interval: int = 300):
            """Monitor lock status (GitHub backend doesn't need refresh)"""
            while not shutdown_event.is_set():
                try:
                    await asyncio.sleep(interval)

                    if not lock_obj.acquired_by_me:
                        logger_run.critical(
                            "🚨 Lock lost unexpectedly - initiating graceful shutdown"
                        )
                        shutdown_event.set()
                        return

                    logger_run.debug("✅ Lock monitor: Still holding lock")

                except asyncio.CancelledError:
                    logger_run.debug("Lock monitor: Cancelled")
                    break

                except Exception as e:
                    logger_run.error(f"Lock monitor error: {e}")

        if cfg.SEND_TEST_MESSAGE:
            await telegram_queue.send(escape_markdown_v2(
                f"🔥 {cfg.BOT_NAME} - Run Started\n"
                f"Date: {format_ist_time(datetime.now(timezone.utc))}\n"
                f"Correlation ID: {correlation_id}\n"
                f"Pairs: {len(pairs_to_process)}"
            ))

        logger_run.debug(
            f"🔔 Processing {len(pairs_to_process)} pairs using optimized parallel architecture"
        )

        logger_run.info("Starting evaluation phase...")
        logger_run.debug("Garbage collection disabled during evaluation loop...")
        
        all_results = await process_pairs_with_workers(fetcher, products_map, pairs_to_process, sdb, telegram_queue, correlation_id, lock, reference_time)
        gc.collect()

        logger_run.debug("Cleanup phase with normal garbage collection...")

        for _, state in all_results:
            if state.get("state") == "ALERT_SENT":
                extra_alerts = state.get("summary", {}).get("alerts", 0)

                if alerts_sent > MAX_ALERTS_PER_RUN:
                    logger_run.warning(
                        f"Alert limit exceeded ({alerts_sent}/{MAX_ALERTS_PER_RUN}), "
                        f"skipping remaining alerts"
                    )
                    break

                if alerts_sent + extra_alerts > MAX_ALERTS_PER_RUN:
                    logger_run.warning(
                        f"Alert limit would be exceeded, skipping {extra_alerts} alerts"
                    )
                    break

                alerts_sent += extra_alerts

        if alerts_sent >= MAX_ALERTS_PER_RUN:
            logger_run.critical(
                "ALERT VOLUME EXCEEDED: %d/%d", alerts_sent, MAX_ALERTS_PER_RUN
            )
            await telegram_queue.send(
                escape_markdown_v2(
                    f"⚠️ HIGH ALERT VOLUME\n"
                    f"Alerts sent: {alerts_sent} (limit: {MAX_ALERTS_PER_RUN})\n"
                    f"Please review configuration for excessive signals.\n"
                    f"Time: {format_ist_time()}"
                )
            )

        if fetcher is None:
            logger_run.error("❌ Fetcher is None - cannot get stats")
            return False

        fetcher_stats = fetcher.get_stats()

        total_required = fetcher_stats['candles']['success'] + fetcher_stats['candles']['failed']
        candles_str = f"{fetcher_stats['candles']['success']}/{total_required}"

        logger_run.info(
            f"📊 Fetch Stats | "
            f"Products: config only | "
            f"Candles: {candles_str}"
        )

        if "rate_limiter" in fetcher_stats and fetcher_stats["rate_limiter"].get("total_waits", 0) > 0:
            rate_stats = fetcher_stats["rate_limiter"]
            logger_run.info(
                f"🚦 Rate limiting | "
                f"Waits: {rate_stats['total_waits']} | "
                f"Total wait: {rate_stats['total_wait_time_seconds']:.1f}s"
            )

        final_memory_mb = process.memory_info().rss / 1024 / 1024
        memory_delta = final_memory_mb - container_memory_mb
        run_duration = time.time() - start_time
        github_status = "OK" if (sdb and not sdb.degraded) else "DEGRADED"

        summary = (
            f"🎯🌏 RUN COMPLETE | "
            f"Duration: {run_duration:.1f}s | "
            f"Pairs: {len(all_results)}/{len(pairs_to_process)} | "
            f"Alerts: {alerts_sent} | "
            f"Memory: {int(final_memory_mb)}MB (Δ{memory_delta:+.0f}MB) | "
            f"GitHub: {github_status}"
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
        logger_run.warning("❌ Run cancelled (shutdown signal received)")
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
        if lock_extension_task:
            try:
                lock_extension_task.cancel()
                await asyncio.wait_for(lock_extension_task, timeout=1.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            except Exception as e:
                logger_run.error(f"Error cancelling lock extension task: {e}")

        if lock_acquired and lock and lock.acquired_by_me:
            try:
                await asyncio.wait_for(lock.release(timeout=3.0), timeout=4.0)
                logger_run.debug("🔏 GitHub-backed lock released")
            except asyncio.TimeoutError:
                logger_run.error("Timeout releasing lock")
            except Exception as e:
                logger_run.error(f"Error releasing lock: {e}", exc_info=False)

        if sdb:
            try:
                await asyncio.wait_for(sdb.close(), timeout=3.0)
                logger_run.debug("✅ GitHub state store connection closed")
            except asyncio.TimeoutError:
                logger_run.error("Timeout closing GitHub state store")
            except Exception as e:
                logger_run.error(f"Error closing Github state store: {e}", exc_info=False)

        try:
            await asyncio.wait_for(
                RedisStateStore.shutdown_global_pool(),
                timeout=5.0
            )
        except asyncio.TimeoutError:
            logger_run.error("Timeout shutting down GitHub state store pool")
        except Exception as e:
            logger_run.error(f"Error shutting down GitHub state store pool: {e}")

        try:
            await asyncio.wait_for(
                SessionManager.close_session(),
                timeout=5.0
            )
            logger_run.debug("✅ HTTP session closed")
        except asyncio.TimeoutError:
            logger_run.error("Timeout closing HTTP session")
        except Exception as e:
            logger_run.error(f"Error closing HTTP session: {e}", exc_info=False)

        try:
            TRACE_ID.set("")
            PAIR_ID.set("")
        except Exception:
            pass

        try:
            gc.collect()
            if cfg.DEBUG_MODE:
                logger_run.debug("🥃 Final garbage collection completed")
        except Exception as e:
            logger_run.debug(f"GC error: {e}")

        logger_run.debug("🧹 Resource cleanup finished")

try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    logger.info(f"🌎 uvloop enabled | {JSON_BACKEND} enabled")
except ImportError:
    logger.info(f"❌ uvloop not available (using default) | {JSON_BACKEND} enabled")

if __name__ == "__main__":
    aot_bridge.ensure_initialized()
    
    if not aot_bridge.is_using_aot():
        reason = aot_bridge.get_fallback_reason() or "Unknown"
        logger.warning("⚠️ AOT not available, using JIT fallback. Reason: %s", reason)
        logger.warning("⚠ Performance will be degraded. First run may be slow.")
        
        if os.getenv("REQUIRE_AOT", "false").lower() == "true":
            logger.critical("❌ REQUIRE_AOT=true but AOT unavailable - exiting")
            sys.exit(1)
    else:
        logger.info("✅ Verified: AOT artifacts loaded successfully")

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
        warmup_if_needed()
    else:
        logger.info("Skipping Numba warmup (faster startup)")

    async def main_with_cleanup():
        sdb = None # We need to ensure we can access the instance created in run_once
        try:
            return await run_once()
        finally:
            logger.info("🧹 Shutting down persistent connections...")
            try:
                await RedisStateStore.shutdown_global_pool() 
            except Exception as e:
                logger.error(f"Error calling shutdown_global_pool: {e}")
            try:
                await SessionManager.close_session()
                logger.debug("⏰ HTTP session closed")
            except Exception as e:
                logger.error(f"Error closing HTTP session: {e}")

    try:
        success = asyncio.run(main_with_cleanup())
        if success:
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
