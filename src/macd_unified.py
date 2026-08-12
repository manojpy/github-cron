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
import gc
import json
from collections import deque
from typing import Dict, Any, Optional, Tuple, List, ClassVar, TypedDict, Callable, Set, Deque, Union
from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from contextvars import ContextVar
from urllib.parse import urlparse
import aiohttp
import numpy as np
import redis.asyncio as redis
from redis.exceptions import ConnectionError as RedisConnectionError, RedisError
from pydantic import BaseModel, Field, field_validator, model_validator
from aiohttp import ClientConnectorError, ClientResponseError, TCPConnector, ClientError
import traceback

from aot_bridge import (
    sanitize_array_numba,
    ema_loop,
    ema_loop_alpha,
    ema_loop_pine,
    kalman_loop,
    vwap_daily_loop_safe,
    rolling_mean_numba,
    rolling_min_max_numba,
    calculate_ppo_core,
    calculate_rsi_core,
    true_range_numba, 
    calculate_atr_rma, 
    calculate_adx_core
)

try:
    import orjson

    def json_dumps(obj: Any) -> str:
        """Fast path: orjson natively handles NumPy types and string keys."""
        return orjson.dumps(obj, option=orjson.OPT_SERIALIZE_NUMPY).decode("utf-8")

    def json_loads(s: str | bytes) -> Any:
        return orjson.loads(s)

    JSONDecodeError = orjson.JSONDecodeError
    JSON_BACKEND = "orjson"

except ImportError:
    import json

    class _NumpyFallbackEncoder(json.JSONEncoder):
        """Ensures stdlib json does not crash on NumPy types if orjson is missing."""
        def default(self, obj: Any) -> Any:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.generic):
                return obj.item()
            return super().default(obj)

    def json_dumps(obj: Any) -> str:
        return json.dumps(obj, cls=_NumpyFallbackEncoder)

    def json_loads(s: str | bytes) -> Any:
        return json.loads(s)

    JSONDecodeError = json.JSONDecodeError
    JSON_BACKEND = "stdlib"

def normalize_timestamp(ts: Union[int, float]) -> int:   
    ts_int = int(ts)
   
    if ts_int > 1_000_000_000_000:  # > year 33658 in seconds, definitely milliseconds
        ts_int = ts_int // 1000
    
    if ts_int < 0 or ts_int > 4102444800:
        raise ValueError(f"Normalized timestamp {ts_int} out of valid range [0, 4102444800]")
    
    return ts_int

def normalize_timestamp_array(ts: np.ndarray) -> np.ndarray:
    """Vectorized version of normalize_timestamp() — no Python-level loop."""
    ts_int = np.asarray(ts, dtype=np.int64)
    ts_int = np.where(ts_int > 1_000_000_000_000, ts_int // 1000, ts_int)
    bad = (ts_int < 0) | (ts_int > 4102444800)
    if np.any(bad):
        raise ValueError(f"Normalized timestamp(s) out of range: {ts_int[bad][:5].tolist()}")
    return ts_int

class CprNotReadyError(Exception):
    """
    Raised by _find_closed_daily_candle() when yesterday's daily candle
    is not yet present in the fetched data array.

    This is a normal, expected condition in the minutes immediately after
    00:00 UTC before the exchange/API emits the new daily bar.

    The caller should:
      - Set nr_cpr = nan  (cpr_ok = False -> alerts silently blocked)
      - NOT log a warning (this is not an error)
      - Let the 15-minute scheduler retry on the next run automatically
    """
    pass

__version__ = "1.8.0-stable"

class Constants:
    MIN_WICK_RATIO = 0.2
    PPO_RSI_GUARD_BUY = 0.50
    PPO_RSI_GUARD_SELL = -0.50
    PPO_SIGNAL_CROSS_MAX_BUY = 0.30
    PPO_SIGNAL_CROSS_MIN_SELL = -0.30
    CIRCUIT_BREAKER_MAX_WAIT = 300
    INFINITY_CLAMP = 1e8
    TELEGRAM_MAX_MESSAGE_LENGTH = 4096
    VWAP_MAX_DISTANCE_PCT = 2.0
    INTER_BATCH_DELAY: float = 0.5
    MIN_CANDLES_FOR_INDICATORS = 250
    CANDLE_SAFETY_BUFFER = 100
    MIN_CLOSED_CANDLES_15M = 4          
    MIN_ALIGNED_5M_CANDLES = 200               
    CANDLE_FETCH_BUFFER_PERIODS = 3 
    API_TIMESTAMP_TOLERANCE_SEC = 300
    MIN_CANDLE_AGE_FROM_OPEN = 850
    MIN_BODY_RATIO = 0.30
    HIGH_DEVIATION_THRESHOLD = 0.5
    
PIVOT_LEVELS_BUY = ["P", "S1", "S2", "S3", "R1", "R2"]
PIVOT_LEVELS_SELL = ["P", "S1", "S2", "R1", "R2", "R3"]

PIVOT_LEVELS = ["P", "S1", "S2", "S3", "R1", "R2", "R3"]

class CompiledPatterns:
    VALID_SYMBOL = re.compile(r'^[A-Z0-9_]+$')
    ESCAPE_MARKDOWN = re.compile(r'[_*\[\]()~`>#+\-=|{}.!]') 
    SECRET_TOKEN = re.compile(r'\b\d{6,}:[A-Za-z0-9_-]{20,}\b')
    CHAT_ID = re.compile(r'chat_id=\d+')
    REDIS_CREDS = re.compile(r'(redis://[^@]+@)')

TRACE_ID: ContextVar[str] = ContextVar("trace_id", default="")
PAIR_ID: ContextVar[str] = ContextVar("pair_id", default="")

class BotConfig(BaseModel):
    TELEGRAM_BOT_TOKEN: str = Field(..., min_length=1)
    TELEGRAM_CHAT_ID: str = Field(..., min_length=1)
    REDIS_URL: str = Field(..., min_length=1)
    DELTA_API_BASE: str = Field(..., min_length=1)
    DEBUG_MODE: bool = Field(default=False)
    SEND_TEST_MESSAGE: bool = Field(default=True, description="Send test message on startup")
    BOT_NAME: str = "Unified Alert Bot"
    PAIRS: List[str] = Field(default=["ETHUSD", "AVAXUSD", "XRPUSD", "BNBUSD", "LTCUSD", "DOTUSD", "ADAUSD", "SUIUSD", "AAVEUSD", "SOLUSD", "PAXGUSD", "PIPPINUSD", "RIVERUSD", "BLESSUSD", "BASEDUSD","SKYAIUSD","HUSD","EDENUSD","XAUTUSD", "ZECUSD", "LABUSD", "BTCUSD", "LINKUSD", "ARBUSD", "KITEUSD", "VVVUSD", "BEATUSD", "BILLUSD", "BCHUSD", "WLDUSD" ], min_length=1) 
    PPO_FAST: int = Field(default=7, ge=1, le=50, description="PPO fast period")
    PPO_SLOW: int = Field(default=16, ge=2, le=100, description="PPO slow period")
    PPO_SIGNAL: int = Field(default=5, ge=1, le=25, description="PPO signal period")
    RMA_50_PERIOD: int = Field(default=50, ge=10, le=200, description="RMA 50 period")
    RMA_200_PERIOD: int = Field(default=200, ge=50, le=500, description="RMA 200 period")
    VOLUME_EMA_LENGTH: int = Field(default=20, ge=2, le=100, description="EMA period for 15m volume, used as wide-CPR confirmation (candle volume > EMA)")
    CPR_ADAPTIVE_CALM: float = Field(default=1.0, ge=0.1, le=20.0, description="Min % move from prev close for wide-CPR bypass, calm regime")
    CPR_ADAPTIVE_VOLATILE: float = Field(default=3.5, ge=0.1, le=20.0, description="Min % move from prev close for wide-CPR bypass, volatile regime") 
    ENABLE_HIST_RMA: bool = Field(default=True, description="Enable RMA 10/30 histogram reversal alerts")
    HIST_RMA_FAST: int = Field(default=10, ge=2, le=100, description="Histogram RMA fast period")
    HIST_RMA_SLOW: int = Field(default=30, ge=5, le=200, description="Histogram RMA slow period")
    ENABLE_PPO_GATE: bool = Field(default=True, description="Enable PPO(32,84,20) as trend gate")
    PPO_GATE_FAST: int = Field(default=32, ge=1, le=100, description="Gate PPO fast period")
    PPO_GATE_SLOW: int = Field(default=84, ge=2, le=200, description="Gate PPO slow period")
    PPO_GATE_SIGNAL: int = Field(default=20, ge=1, le=50, description="Gate PPO signal period")
    RSI_GUARD_ENABLED: bool = Field(default=False, description="Enable RSI(89) Kalman-smoothed vs EMA(50) as an alternate trend gate, OR'd with PPO gate")
    RSI_GUARD_RSI_LEN: int = Field(default=89, ge=2, le=200, description="RSI Guard RSI length")
    RSI_GUARD_KALMAN_LEN: int = Field(default=9, ge=1, le=50, description="RSI Guard Kalman smoothing length")
    RSI_GUARD_EMA_LEN: int = Field(default=50, ge=1, le=200, description="RSI Guard EMA length applied to Kalman-smoothed RSI")
    SRSI_RSI_LEN: int = 14
    SRSI_KALMAN_LEN: int = 9
    SRSI_EMA_LEN: int = 5
    ATR_SHORT: int = Field(default=5, ge=1, le=50)
    ATR_LONG: int = Field(default=14, ge=2, le=200)
    MAX_PARALLEL_FETCH: int = Field(15, ge=1, le=20)
    HTTP_TIMEOUT: int = 15
    CANDLE_FETCH_RETRIES: int = 3
    CANDLE_FETCH_BACKOFF: float = 1.5
    RUN_TIMEOUT_SECONDS: int = 600
    FETCH_PHASE_TIMEOUT_SEC: int = 90
    TCP_CONN_LIMIT: int = 16
    TCP_CONN_LIMIT_PER_HOST: int = 16
    TELEGRAM_RETRIES: int = 3
    TELEGRAM_BACKOFF_BASE: float = 2.0
    MEMORY_LIMIT_BYTES: int = 400_000_000
    STATE_EXPIRY_DAYS: int = 11
    LOG_LEVEL: str = "INFO"
    ENABLE_ADX_FILTER: bool = Field(default=True)
    ENABLE_RVOL_ALERT: bool = Field(default=True)
    ENABLE_VWAP: bool = Field(default=True)
    ENABLE_PIVOT: bool = Field(default=True)
    ENABLE_CPR: bool = Field(default=False)
    ENABLE_CPR_ADX_RVOL_CONFIRM: bool = Field(default=False, description="DEPRECATED: no longer used in CPR momentum gate (see CPR_MOMENTUM_BODY_RATIO_MIN)")
    CPR_THRESHOLD_PCT: float = Field(default=0.010, ge=0.001, le=0.10)
    CPR_MOMENTUM_BODY_RATIO_MIN: float = Field(default=0.50, ge=0.0, le=1.0, description="Min |close-open|/range for candle-body-conviction momentum vote")
    PIVOT_LOOKBACK_PERIOD: int = 15
    FAIL_ON_REDIS_DOWN: bool = False
    FAIL_ON_TELEGRAM_DOWN: bool = False
    TELEGRAM_RATE_LIMIT_PER_MINUTE: int = 20
    TELEGRAM_BURST_SIZE: int = 5
    REDIS_CONNECTION_RETRIES: int = 3
    REDIS_RETRY_DELAY: float = 2.0
    REDIS_LOCK_EXPIRY: int = Field(default=900, ge=900, description="Redis lock TTL in seconds")
    ALERT_DEDUP_WINDOW_SEC: int = Field(default=120, ge=0, description="Dedup window for repeat alerts")
    ENABLE_ALERT_COALESCING: bool = Field(default=True) 
    COALESCE_DEDUP_WINDOW_SEC: int = Field(default=1800, ge=0)
    ENABLE_CONFLUENCE_GATE: bool = Field(default=False) 
    CONFLUENCE_MIN_VOTES: int = Field(default=3, ge=2, le=10) 
    DRY_RUN_MODE: bool = Field(default=False)
    SKIP_WARMUP: bool = Field(default=False)
    REJECT_HIGH_DEVIATION: bool = Field( default=False)
    SANITIZE_BAD_CANDLES: bool = Field(default=False, description="If True, drop individual invalid candles instead of rejecting the whole fetch")
    ICHIMOKU_CLOUD_ENABLED: bool = Field(default=True, description="Enable Ichimoku Cloud as trend gate")
    ICHIMOKU_CONVERSION_PERIODS: int = Field(default=9, ge=1, le=300, description="Ichimoku conversion line length")
    ICHIMOKU_BASE_PERIODS: int = Field(default=26, ge=1, le=400, description="Ichimoku base line length")
    ICHIMOKU_SPANB_PERIODS: int = Field(default=52, ge=1, le=500, description="Ichimoku leading span B length")
    ICHIMOKU_DISPLACEMENT: int = Field(default=26, ge=1, le=400, description="Ichimoku cloud forward displacement")
    ICHIMOKU_TK_GUARD_ENABLED: bool = Field(default=True, description="Require 15m Tenkan(conversion) vs Kijun(base) alignment: buy needs conversion>=base, sell needs conversion<=base")
    RMA_CLOUD_ENABLED: bool = Field(default=True, description="Enable RMA(fast)/RMA(50) 15m cloud as trend gate; green (buy) when RMA_fast>RMA50, red (sell) when RMA_fast<RMA50. Reuses the existing RMA50(15m)/RMA_50_PERIOD used for base trend.")
    RMA_CLOUD_FAST_PERIOD: int = Field(default=20, ge=2, le=200, description="RMA Cloud fast period (15m). Slow leg reuses RMA_50_PERIOD.")
    ENABLE_TK_CONVERSION_CROSS: bool = Field(default=True, description="Enable 15m alert when close crosses above/below the Ichimoku conversion (Tenkan) line, subject to all other buy/sell common conditions")
    ENABLE_CLOUD_CROSS_ALERT: bool = Field(default=True, description="Enable 15m alert when close crosses above/below the Ichimoku cloud (23,65,130,65), subject to all other buy/sell common conditions") 
    ENABLE_KIJUN_CROSS: bool = Field(default=True, description="Enable 15m alert when close crosses above/below the Ichimoku base (Kijun) line (23,65,130,65), subject to all other buy/sell common conditions")

    FAST_ICHIMOKU_CONVERSION_PERIODS: int = Field(default=9, ge=1, le=300, description="Fast Ichimoku (alert-only) conversion line length")
    FAST_ICHIMOKU_BASE_PERIODS: int = Field(default=26, ge=1, le=400, description="Fast Ichimoku (alert-only) base line length")
    FAST_ICHIMOKU_SPANB_PERIODS: int = Field(default=52, ge=1, le=500, description="Fast Ichimoku (alert-only) leading span B length")
    FAST_ICHIMOKU_DISPLACEMENT: int = Field(default=26, ge=1, le=400, description="Fast Ichimoku (alert-only) cloud forward displacement")
    ENABLE_FAST_ICHIMOKU_CLOUD_CROSS: bool = Field(default=True, description="15m alert: close crosses fast Ichimoku cloud (9,26,52,26), gated by future fast-cloud color and Fast Tenkan/Kijun alignment, plus all other buy/sell common conditions")
    ENABLE_FAST_ICHIMOKU_TENKAN_CROSS: bool = Field(default=True, description="15m alert: close crosses fast Ichimoku Tenkan line (9,26,52,26), gated by close beyond current fast cloud, future fast-cloud color, and Fast Tenkan/Kijun alignment, plus all other buy/sell common conditions")
    
    EVAL_CONCURRENCY_LIMIT: int = Field(default=5, ge=1, le=30, description="Max pairs evaluated concurrently")
    MIN_RUN_TIMEOUT: int = Field(default=480, ge=300, le=1800)  # Min/max run timeout in seconds (5-30 min)
    MAX_ALERTS_PER_PAIR: int = Field(default=8, ge=5, le=15)  # Max alerts per pair per run    
    MAX_ALERTS_PER_RUN: int = Field(default=50, ge=10, le=200)  
    PIVOT_MAX_DISTANCE_PCT: float = Field(default=1.0)  # Max distance from pivot to trigger alert (1.5%)
    RVOL_THRESHOLD: float = Field(default=1.0, ge=0.5, le=2.0)  # Volatility expansion threshold (1.0=baseline, 1.5=50% expansion required) 
    ATR_ADAPTIVE_ENABLED: bool = Field(default=True)
    ATR_PCTL_LOOKBACK: int = Field(default=96, ge=20, le=500)
    ATR_PCTL_MIN_HISTORY: int = Field(default=50, ge=10, le=400)
    ADAPTIVE_MULT_CALM: float = Field(default=0.85, ge=0.1, le=2.0)
    ADAPTIVE_MULT_VOLATILE: float = Field(default=1.4, ge=0.5, le=3.0)
    ADX_DI_LENGTH: int = Field(default=14, ge=5, le=30)
    ADX_SMOOTHING_LENGTH: int = Field(default=14, ge=5, le=30)
    ADX_ADAPTIVE_TARGET_PCTL: float = Field(default=60.0, ge=1.0, le=99.0, description="ADX threshold = this percentile of the pair's own trailing ADX history")
    ADX_ADAPTIVE_BAND_WIDTH: float = Field(default=0.0, ge=0.0, le=40.0)
    ADX_ADAPTIVE_FALLBACK: float = Field(default=18.0, ge=5.0, le=50.0, description="ADX threshold used during warm-up or when ATR_ADAPTIVE_ENABLED=False")
    PPO_ADAPTIVE_CALM: float = Field(default=0.08, ge=0.01, le=1.0, description="PPO cross threshold in calm regime")
    PPO_ADAPTIVE_VOLATILE: float = Field(default=0.20, ge=0.01, le=1.0, description="PPO cross threshold in volatile regime")
    RSI_ADAPTIVE_BUY_CALM: float = Field(default=55.0, ge=50.0, le=90.0, description="RSI buy level in calm regime")
    RSI_ADAPTIVE_BUY_VOLATILE: float = Field(default=70.0, ge=50.0, le=90.0, description="RSI buy level in volatile regime")
    RSI_ADAPTIVE_SELL_CALM: float = Field(default=45.0, ge=10.0, le=50.0, description="RSI sell level in calm regime")
    RSI_ADAPTIVE_SELL_VOLATILE: float = Field(default=30.0, ge=10.0, le=50.0, description="RSI sell level in volatile regime")
    MAX_CANDLE_STALENESS_SEC: int = Field(default=1200, ge=600, le=3600)  # Max candle age in seconds (10-60 min)
    RATE_LIMIT_PER_MINUTE: int = Field(default=400, ge=90, le=600)
    CONFIRM_RATE_LIMIT_PER_MINUTE: int = Field(default=20, ge=5, le=60)
    CB_FAILURE_THRESHOLD: int = Field(default=3, ge=1, le=10)  # Failures before circuit breaker opens
    CB_RECOVERY_TIMEOUT: int = Field(default=60, ge=10, le=600)  # Circuit breaker recovery wait time (seconds)
    DAILY_RESET_BUFFER_SEC: int = Field(default=300, ge=0, le=3600)  # Buffer after midnight before allowing daily resets (VWAP/pivots)
    MIN_CANDLES_PER_DAY: int = Field(default=94, ge=50, le=100)  # Minimum candles for complete day (94=23h for 15m candles)
    CANDLE_MIN_AGE_BUFFER: int = Field(default=60, ge=0, le=600)  # Seconds to wait after candle interval before using (ensures finalized data)
        
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

    @field_validator('PPO_FAST', 'PPO_SLOW', 'PPO_SIGNAL')
    @classmethod
    def validate_ppo_params(cls, v):
        if not (1 <= v <= 100):
            raise ValueError(f'PPO parameter must be 1-100, got {v}')
        return v

    @model_validator(mode='after')
    def validate_adaptive_rvol(self) -> 'BotConfig':
        if self.ATR_SHORT >= self.ATR_LONG:
            raise ValueError(
                f'ATR_SHORT ({self.ATR_SHORT}) must be < ATR_LONG ({self.ATR_LONG}) '
                f'— the RVOL ratio assumes short-period ATR is compared against a longer baseline'
            )
        if self.ATR_ADAPTIVE_ENABLED:
            if self.ATR_PCTL_MIN_HISTORY >= self.ATR_PCTL_LOOKBACK:
                raise ValueError(
                    f'ATR_PCTL_MIN_HISTORY ({self.ATR_PCTL_MIN_HISTORY}) must be < '
                    f'ATR_PCTL_LOOKBACK ({self.ATR_PCTL_LOOKBACK})'
                )

            if self.ADAPTIVE_MULT_CALM >= self.ADAPTIVE_MULT_VOLATILE:
                raise ValueError(
                    f'ADAPTIVE_MULT_CALM ({self.ADAPTIVE_MULT_CALM}) must be < '
                    f'ADAPTIVE_MULT_VOLATILE ({self.ADAPTIVE_MULT_VOLATILE})'
                )
            if self.PPO_ADAPTIVE_CALM >= self.PPO_ADAPTIVE_VOLATILE:
                raise ValueError(
                    f'PPO_ADAPTIVE_CALM ({self.PPO_ADAPTIVE_CALM}) must be < '
                    f'PPO_ADAPTIVE_VOLATILE ({self.PPO_ADAPTIVE_VOLATILE})'
                )
            if self.RSI_ADAPTIVE_BUY_CALM >= self.RSI_ADAPTIVE_BUY_VOLATILE:
                raise ValueError(
                    f'RSI_ADAPTIVE_BUY_CALM ({self.RSI_ADAPTIVE_BUY_CALM}) must be < '
                    f'RSI_ADAPTIVE_BUY_VOLATILE ({self.RSI_ADAPTIVE_BUY_VOLATILE})'
                )
            if self.RSI_ADAPTIVE_SELL_CALM <= self.RSI_ADAPTIVE_SELL_VOLATILE:
                raise ValueError(
                    f'RSI_ADAPTIVE_SELL_CALM ({self.RSI_ADAPTIVE_SELL_CALM}) must be > '
                    f'RSI_ADAPTIVE_SELL_VOLATILE ({self.RSI_ADAPTIVE_SELL_VOLATILE}) '
                    f'— sell threshold drops as volatility rises'
                )
            if self.CPR_ADAPTIVE_CALM >= self.CPR_ADAPTIVE_VOLATILE:
                raise ValueError(
                    f'CPR_ADAPTIVE_CALM ({self.CPR_ADAPTIVE_CALM}) must be < '
                    f'CPR_ADAPTIVE_VOLATILE ({self.CPR_ADAPTIVE_VOLATILE})'
                )

            if self.ADX_ADAPTIVE_BAND_WIDTH > 0:
                lo = self.ADX_ADAPTIVE_TARGET_PCTL - self.ADX_ADAPTIVE_BAND_WIDTH / 2.0
                hi = self.ADX_ADAPTIVE_TARGET_PCTL + self.ADX_ADAPTIVE_BAND_WIDTH / 2.0
                if lo < 1.0 or hi > 99.0:
                    raise ValueError(
                        f'ADX_ADAPTIVE_TARGET_PCTL ({self.ADX_ADAPTIVE_TARGET_PCTL}) ± '
                        f'band/2 ({self.ADX_ADAPTIVE_BAND_WIDTH / 2.0}) produces range '
                        f'[{lo:.1f}, {hi:.1f}] which exceeds [1, 99]'
                    )
        return self

    @model_validator(mode='after')
    def validate_ppo_ordering(self) -> 'BotConfig':
        if self.PPO_FAST >= self.PPO_SLOW:
            raise ValueError(
                f'PPO_FAST ({self.PPO_FAST}) must be strictly less than '
                f'PPO_SLOW ({self.PPO_SLOW})'
            )

        if self.PPO_GATE_FAST >= self.PPO_GATE_SLOW:
            raise ValueError(
                f'PPO_GATE_FAST ({self.PPO_GATE_FAST}) must be strictly less than '
                f'PPO_GATE_SLOW ({self.PPO_GATE_SLOW})'
            )
        if self.ENABLE_HIST_RMA and self.HIST_RMA_FAST >= self.HIST_RMA_SLOW:
            raise ValueError(
                f'HIST_RMA_FAST ({self.HIST_RMA_FAST}) must be strictly less than '
                f'HIST_RMA_SLOW ({self.HIST_RMA_SLOW})'
            )

        if self.RMA_CLOUD_ENABLED and self.RMA_CLOUD_FAST_PERIOD >= self.RMA_50_PERIOD:
            raise ValueError(
                f'RMA_CLOUD_FAST_PERIOD ({self.RMA_CLOUD_FAST_PERIOD}) must be strictly less than '
                f'RMA_50_PERIOD ({self.RMA_50_PERIOD}), since the cloud slow leg reuses RMA_50_PERIOD'
            )

        return self

    @model_validator(mode='after')
    def validate_logic(self) -> 'BotConfig':
        
        errors = []
        warnings = []

        if self.RUN_TIMEOUT_SECONDS < self.MIN_RUN_TIMEOUT:
            errors.append(
                f'RUN_TIMEOUT_SECONDS ({self.RUN_TIMEOUT_SECONDS}s) must be >= '
                f'MIN_RUN_TIMEOUT ({self.MIN_RUN_TIMEOUT}s)'
            )

        if self.RUN_TIMEOUT_SECONDS >= self.REDIS_LOCK_EXPIRY:
            errors.append(
                f'REDIS_LOCK_EXPIRY ({self.REDIS_LOCK_EXPIRY}s) must be > '
                f'RUN_TIMEOUT_SECONDS ({self.RUN_TIMEOUT_SECONDS}s)'
            )

        if self.TELEGRAM_RATE_LIMIT_PER_MINUTE < 10 or self.TELEGRAM_RATE_LIMIT_PER_MINUTE > 30:
            errors.append('TELEGRAM_RATE_LIMIT_PER_MINUTE must be 10-30')

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

        min_batches = -(-len(self.PAIRS) // self.MAX_PARALLEL_FETCH)  # ceil division
        estimated_runtime = min_batches * Constants.INTER_BATCH_DELAY * 100  # heuristic, recalibrate with observed data
        safe_fraction = 0.8

        if min_batches > 3 or estimated_runtime > self.RUN_TIMEOUT_SECONDS * safe_fraction:
            warnings.append(
                f"PAIRS={len(self.PAIRS)} with MAX_PARALLEL_FETCH={self.MAX_PARALLEL_FETCH} "
                f"requires {min_batches} sequential fetch batches. "
                f"Estimated runtime ~{int(estimated_runtime)}s may exceed safe window "
                f"({int(self.RUN_TIMEOUT_SECONDS * safe_fraction)}s of RUN_TIMEOUT_SECONDS={self.RUN_TIMEOUT_SECONDS}s). "
                f"Verify actual runtime before adding more pairs."  
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

    for field_name, field_info in BotConfig.model_fields.items():
        env_value = os.getenv(field_name)
        if env_value is None:
            continue
        if field_info.annotation is str:
            data[field_name] = env_value  # never JSON-decode str fields (e.g. all-digit chat IDs)
        else:
            try:
                data[field_name] = json_loads(env_value)
            except Exception:
                data[field_name] = env_value

    for key in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "REDIS_URL", "DELTA_API_BASE"):
        val = data.get(key, "")
        if not val or val.startswith("__SET_IN_"):
            print(f"❌ ERROR: Missing required config: {key}", file=sys.stderr)
            print(f"❌ Set this in your CI/CD secrets (GitHub Actions → Secrets, GitLab → Variables)", file=sys.stderr)
            sys.exit(1)
    try:
        return BotConfig(**data)
    except Exception as exc:
        print(f"❌ ERROR: Pydantic validation failed", file=sys.stderr)
        print(f"❌ Details: {exc}", file=sys.stderr)
        sys.exit(1)

cfg = load_config()

class TraceContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = TRACE_ID.get()
        record.pair_id = PAIR_ID.get()
        return True

class SafeFormatter(logging.Formatter):
    @staticmethod
    def _apply_all_redactions(text: str) -> str:
        if not any(s in text for s in (':', 'redis://', 'chat_id')):
            return text
    
        text = CompiledPatterns.SECRET_TOKEN.sub("[REDACTED_TOKEN]", text)
        text = CompiledPatterns.CHAT_ID.sub("chat_id=[REDACTED]", text)
        text = CompiledPatterns.REDIS_CREDS.sub("redis://[REDACTED]", text)
        return text
    
    def format(self, record: logging.LogRecord) -> str:
        if record.msg:
            record.msg = self._apply_all_redactions(str(record.msg))
        
        if record.args:
            if isinstance(record.args, dict):
                record.args = {k: self._mask_secret(v) for k, v in record.args.items()}
            elif isinstance(record.args, tuple):
                record.args = tuple(self._mask_secret(v) for v in record.args)
        
        formatted = super().format(record)
        return self._apply_all_redactions(formatted)
  
    @staticmethod
    def _mask_secret(value: Any) -> Any:
        """Mask sensitive values while preserving numeric types for %d/%f format specifiers."""
        if value is None:
            return value
        if isinstance(value, (int, float, bool)):
            return value
        return SafeFormatter._apply_all_redactions(str(value))

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
    # REMOVED: console.addFilter(SecretFilter())  -- SafeFormatter already redacts
    console.addFilter(TraceContextFilter())  
    logger.addHandler(console)
    logger.debug(
        f"Logging configured | Level: {logging.getLevelName(level)} | "
        f"Format: structured with trace_id | Output: stdout"
    )
    return logger

logger = setup_logging()
logger_main = logger

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

_pair_eval_counter = 0
MEMORY_CHECK_INTERVAL_PAIRS = 5  # only sample RSS every N pair evaluations

_VALIDATION_DONE = False

def validate_runtime_config() -> None:
    global _VALIDATION_DONE
    if _VALIDATION_DONE:       
        return   
    errors = []
    warnings = []
    if hasattr(cfg, '_validation_warnings'):
        warnings.extend(cfg._validation_warnings)
    
    try:
        from urllib.parse import urlparse
        parsed = urlparse(cfg.REDIS_URL)
        if parsed.scheme not in ('redis', 'rediss'):
            errors.append(f"Invalid REDIS_URL scheme: {parsed.scheme} (must be redis:// or rediss://)")
        if not parsed.hostname:
            errors.append("REDIS_URL missing hostname")
    except Exception as e:
        errors.append(f"Failed to parse REDIS_URL: {e}")
    
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

def validate_indicator_array(arr: Optional[np.ndarray], name: str, 
                            min_valid_values: int = 1) -> Tuple[bool, Optional[str]]:    
    if arr is None:
        return False, f"{name} is None"
    
    if len(arr) == 0:
        return False, f"{name} is empty array"
    
    if np.all(np.isnan(arr)):
        return False, f"{name} is all NaN values"
    
    valid_count = np.sum(~np.isnan(arr))
    if valid_count < min_valid_values:
        return False, f"{name} has only {valid_count} valid values (need {min_valid_values})"
    
    return True, None

def validate_indicators_dict(indicators: Optional[dict], required_keys: List[str]) -> Tuple[bool, Optional[str]]:   
    if indicators is None:
        return False, "Indicators dict is None"
    
    if not isinstance(indicators, dict):
        return False, f"Indicators is {type(indicators)}, not dict"
    
    missing_keys = set(required_keys) - set(indicators.keys())
    if missing_keys:
        return False, f"Missing indicator keys: {missing_keys}"
    
    for key in required_keys:
        is_valid, msg = validate_indicator_array(indicators[key], f"indicators[{key}]")
        if not is_valid:
            return False, msg
    
    return True, None

def is_previous_day_complete(timestamps: np.ndarray, current_time: int, min_candles: int = 90, buffer_seconds: int = 300) -> Tuple[bool, str]:
    if len(timestamps) == 0:
        return False, "No timestamp data available"
    
    days = timestamps // 86400
    current_day_number = current_time // 86400
    unique_days = np.unique(days)
    past_days = unique_days[unique_days < current_day_number]
    
    if len(past_days) == 0:
        return False, f"No previous days found before {current_day_number}"
        
    previous_day_number = past_days[-1]
    
    previous_day_mask = (days == previous_day_number)
    previous_day_candles = timestamps[previous_day_mask]
    
    if len(previous_day_candles) == 0:
        return False, f"No candles for day #{previous_day_number}"
    
    if len(previous_day_candles) < min_candles:
        return False, (
            f"Insufficient candles: {len(previous_day_candles)} "
            f"(need >={min_candles})"
        )
    
    seconds_into_day = current_time % 86400
    if seconds_into_day < buffer_seconds:
        return False, (
            f"Within buffer: {seconds_into_day}s < {buffer_seconds}s"
        )
    
    last_candle_day = previous_day_candles[-1] // 86400
    if last_candle_day != previous_day_number:
        return False, "Last candle has wrong day number"
    
    return True, "Complete"

def _find_closed_daily_candle(data_daily: Dict[str, np.ndarray], reference_time: int):
    if data_daily is None:
        raise CprNotReadyError("data_daily is None")

    ts_arr = data_daily.get("timestamp")
    hi_arr = data_daily.get("high")
    lo_arr = data_daily.get("low")
    cl_arr = data_daily.get("close")

    for name, arr in (("timestamp", ts_arr), ("high", hi_arr),
                      ("low", lo_arr), ("close", cl_arr)):
        if arr is None or len(arr) == 0:
            raise CprNotReadyError(f"daily {name} array empty or missing")

    for name, arr in (("high", hi_arr), ("low", lo_arr), ("close", cl_arr)):
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            raise ValueError(f"daily {name} contains NaN/Inf — data corrupt")

    day_numbers   = ts_arr // 86400
    yesterday_num = (reference_time // 86400) - 1

    mask = (day_numbers == yesterday_num)
    if not np.any(mask):
        yesterday_date = datetime.fromtimestamp(
            yesterday_num * 86400, tz=timezone.utc
        ).date()
        raise CprNotReadyError(
            f"Yesterday's candle ({yesterday_date}) not yet in daily array "
            f"({len(ts_arr)} candles, newest day={int(day_numbers[-1])}). "
            f"Will retry next run."
        )

    idx       = int(np.where(mask)[0][-1])  # last bar of that day
    candle_ts = int(ts_arr[idx])
    d_high    = float(hi_arr[idx])
    d_low     = float(lo_arr[idx])
    d_close   = float(cl_arr[idx])

    if d_high < d_low:
        raise ValueError(f"Corrupt candle: high({d_high}) < low({d_low})")
    if d_low <= 0 or d_high <= 0 or d_close <= 0:
        raise ValueError(f"Non-positive OHLC: H={d_high} L={d_low} C={d_close}")
    if (d_high - d_low) < 1e-8:
        raise ValueError(f"Degenerate candle: range={d_high - d_low:.2e}")
    if not (d_low <= d_close <= d_high):
        raise ValueError(f"Close outside H/L: H={d_high} L={d_low} C={d_close}")

    return d_high, d_low, d_close, candle_ts

def _find_today_daily_open(data_daily: Dict[str, np.ndarray], reference_time: int) -> Optional[float]:
    ts_arr = data_daily.get("timestamp")
    op_arr = data_daily.get("open")
    if ts_arr is None or op_arr is None or len(ts_arr) == 0:
        return None

    day_numbers = ts_arr // 86400
    today_num = reference_time // 86400

    mask = (day_numbers == today_num)
    if not np.any(mask):
        return None

    idx = int(np.where(mask)[0][-1])
    open_val = float(op_arr[idx])

    if np.isnan(open_val) or np.isinf(open_val) or open_val <= 0:
        return None

    return open_val

def validate_conversion_cross(close_prev: float, close_curr: float,
    conv_prev: float, conv_curr: float, is_buy: bool) -> Tuple[bool, Optional[str]]:
    vals = [close_prev, close_curr, conv_prev, conv_curr]
    if any(np.isnan(v) for v in vals):
        return False, "NaN in inputs"
    if close_prev <= 0 or close_curr <= 0:
        return False, "Non-positive close"

    if is_buy:
        crossed = (close_prev <= conv_prev) and (close_curr > conv_curr)
        if not crossed:
            return False, "No bullish conversion-line cross"
        return True, None
    else:
        crossed = (close_prev >= conv_prev) and (close_curr < conv_curr)
        if not crossed:
            return False, "No bearish conversion-line cross"
        return True, None

def validate_cloud_cross(close_prev: float, close_curr: float,
    cloud_upper_prev: float, cloud_upper_curr: float,
    cloud_lower_prev: float, cloud_lower_curr: float, is_buy: bool) -> Tuple[bool, Optional[str]]:

    vals = [close_prev, close_curr, cloud_upper_prev, cloud_upper_curr, cloud_lower_prev, cloud_lower_curr]
    if any(np.isnan(v) for v in vals):
        return False, "NaN in inputs"
    if close_prev <= 0 or close_curr <= 0:
        return False, "Non-positive close"

    if is_buy:
        crossed = (close_prev <= cloud_upper_prev) and (close_curr > cloud_upper_curr)
        if not crossed:
            return False, "No bullish cloud cross"
        return True, None
    else:
        crossed = (close_prev >= cloud_lower_prev) and (close_curr < cloud_lower_curr)
        if not crossed:
            return False, "No bearish cloud cross"
        return True, None

def validate_vwap_cross(close_prev: float, close_curr: float, vwap_prev: float, vwap_curr: float, is_buy: bool,
    min_deviation: float = 0.001, max_deviation_pct: float = Constants.VWAP_MAX_DISTANCE_PCT) -> Tuple[bool, Optional[str]]:
 
    vals = [close_prev, close_curr, vwap_prev, vwap_curr]
    
    if any(np.isnan(v) for v in vals):
        return False, "NaN in inputs"
    
    if any(v <= 0 for v in vals):
        return False, "Non-positive values"

    if is_buy:
        crossed = (close_prev <= vwap_prev) and (close_curr > vwap_curr)
        if not crossed:
            return False, "No bullish cross"
        
        sep = (close_curr - vwap_curr) / vwap_curr
        if sep < min_deviation:
            return False, f"Separation {sep*100:.3f}% < {min_deviation*100:.1f}%"
        if sep * 100 > max_deviation_pct:
            return False, f"Separation {sep*100:.3f}% > max {max_deviation_pct:.1f}% — likely bad candle"
        return True, None
    
    else:
        crossed = (close_prev >= vwap_prev) and (close_curr < vwap_curr)
        if not crossed:
            return False, "No bearish cross"
        
        sep = (vwap_curr - close_curr) / vwap_curr
        if sep < min_deviation:
            return False, f"Separation {sep*100:.3f}% < {min_deviation*100:.1f}%"
        if sep * 100 > max_deviation_pct:
            return False, f"Separation {sep*100:.3f}% > max {max_deviation_pct:.1f}% — likely bad candle"
        return True, None

def get_utc_date_key(timestamp: int) -> str:
    utc_dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    return utc_dt.date().isoformat()

def should_reset_daily_state(current_timestamp: int, 
                             last_reset_timestamp_str: Optional[str]) -> bool:
    current_date_str = get_utc_date_key(current_timestamp)
    
    if not last_reset_timestamp_str:
        return True  # Never reset before
    
    return last_reset_timestamp_str != current_date_str

def _sync_signal_handler(sig: int, frame: Any) -> None:
    logger.warning(f"Received signal {sig}, initiating async shutdown...")
    try:
        loop = asyncio.get_running_loop()
        loop.call_soon_threadsafe(shutdown_event.set)
    except RuntimeError:
        pass

signal.signal(signal.SIGTERM, _sync_signal_handler)
signal.signal(signal.SIGINT, _sync_signal_handler)

_STARTUP_BANNER_PRINTED = False
def print_startup_banner_once() -> None:
    global _STARTUP_BANNER_PRINTED
    if _STARTUP_BANNER_PRINTED:
        return
    _STARTUP_BANNER_PRINTED = True
    logger.info(
        f"📡 Bot v{__version__} | Pairs: {len(cfg.PAIRS)} | Workers: {cfg.MAX_PARALLEL_FETCH} | "
        f"Timeout: {cfg.RUN_TIMEOUT_SECONDS}s | Redis Lock: {cfg.REDIS_LOCK_EXPIRY}s"
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
    
    return int(datetime.now(timezone.utc).timestamp())

def calculate_expected_candle_timestamp(reference_time: int, interval_minutes: int) -> int: 
    interval_seconds = interval_minutes * 60
    current_interval_open = (reference_time // interval_seconds) * interval_seconds
    last_closed_candle_open = current_interval_open - interval_seconds
    return last_closed_candle_open

def escape_markdown_v2(text: str) -> str:
    return CompiledPatterns.ESCAPE_MARKDOWN.sub(r'\\\g<0>', str(text))

def calculate_smooth_rsi_numpy(close: np.ndarray, rsi_len: int, kalman_len: int, ema_len: int) -> Tuple[np.ndarray, np.ndarray]:
    try:
        if close is None:
            logger.warning("Smooth RSI: Input close array is None")
            flat = np.full(1, 50.0, dtype=np.float64)
            return flat, flat.copy()

        if len(close) < rsi_len + kalman_len + ema_len:
            logger.warning(f"Smooth RSI: Insufficient data (len={len(close)}, required={rsi_len + kalman_len + ema_len})")
            flat = np.full(len(close), 50.0, dtype=np.float64)
            return flat, flat.copy()

        rsi = calculate_rsi_core(close, rsi_len)
        smooth_rsi = kalman_loop(rsi, kalman_len, 0.01, 0.1)
        rsi_ema = ema_loop(smooth_rsi, float(ema_len))
        smooth_rsi = sanitize_array_numba(smooth_rsi, 50.0)
        rsi_ema    = sanitize_array_numba(rsi_ema, 50.0)

        return smooth_rsi, rsi_ema
    except Exception as e:
        logger.error(f"Smooth RSI calculation failed: {e}")
        default_len = len(close) if close is not None else 1
        flat = np.full(default_len, 50.0, dtype=np.float64)
        return flat, flat.copy()

def calculate_volume_ema_numpy(volume: np.ndarray, length: int) -> np.ndarray:
    try:
        if volume is None or len(volume) < length:
            logger.warning(f"Volume EMA: Insufficient data (len={len(volume) if volume is not None else 0}, required={length})")
            default_len = len(volume) if volume is not None else 1
            return np.full(default_len, np.nan, dtype=np.float64)

        vol_ema = ema_loop(volume.astype(np.float64), float(length))
        vol_ema = sanitize_array_numba(vol_ema, np.nan)

        return vol_ema
    except Exception as e:
        logger.error(f"Volume EMA calculation failed: {e}")
        default_len = len(volume) if volume is not None else 1
        return np.full(default_len, np.nan, dtype=np.float64)

def calculate_ppo_numpy(close: np.ndarray, fast: int, slow: int, signal: int) -> Tuple[np.ndarray, np.ndarray]:
    try:
        if close is None or len(close) < max(fast, slow):
            logger.warning(f"PPO: Insufficient data")
            default_len = len(close) if close is not None else 1
            return np.full(default_len, np.nan, dtype=np.float64), np.full(default_len, np.nan, dtype=np.float64)

        ppo, ppo_sig = calculate_ppo_core(close, fast, slow, signal)
        ppo     = sanitize_array_numba(ppo,     np.nan)
        ppo_sig = sanitize_array_numba(ppo_sig, np.nan)

        return ppo, ppo_sig

    except Exception as e:
        logger.error(f"PPO calculation failed: {e}")
        default_len = len(close) if close is not None else 1
        return np.full(default_len, np.nan, dtype=np.float64), np.full(default_len, np.nan, dtype=np.float64)
    
def calculate_vwap_numpy(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, timestamps: np.ndarray,
    reference_time: Optional[int] = None) -> np.ndarray:
    try:
        hlc3 = (high + low + close) / 3.0
        
        if len(timestamps) > 1 and np.any(np.diff(timestamps) < 0):
            logger.warning(
                "[%s] Timestamps not sorted — VWAP may be incorrect",
                PAIR_ID.get() or "?"
            )  
        return vwap_daily_loop_safe(hlc3, volume, timestamps)
    except Exception as e:
        logger.error(f"VWAP calculation failed: {e}", exc_info=True)
        return np.full(len(close), np.nan, dtype=np.float64)

def calculate_rma_numpy(data: np.ndarray, period: int) -> np.ndarray:
    try:
        if data is None or len(data) < period:
            return np.full_like(data, np.nan) if data is not None else np.array([np.nan])

        alpha = 1.0 / period
        rma = ema_loop_alpha(data, alpha)
        rma = sanitize_array_numba(rma, np.nan)
        return rma
    except Exception as e:
        logger.error(f"RMA calculation failed: {e}")
        return np.full_like(data, np.nan) if data is not None else np.array([np.nan]) 

def calculate_ichimoku_numpy(high: np.ndarray, low: np.ndarray, close: np.ndarray, conversion_periods: int = 9, base_periods: int = 26, span_b_periods: int = 52, displacement: int = 26) -> Dict[str, np.ndarray]:
    try:
        n = len(high)
        if n == 0:
            raise ValueError("Empty input arrays")

        # Conversion Line (Tenkan-sen): (highest high + lowest low) / 2
        _, hh_conv = rolling_min_max_numba(high, conversion_periods)
        ll_conv, _ = rolling_min_max_numba(low, conversion_periods)
        conversion_line = (hh_conv + ll_conv) / 2.0

        # Base Line (Kijun-sen): (highest high + lowest low) / 2
        _, hh_base = rolling_min_max_numba(high, base_periods)
        ll_base, _ = rolling_min_max_numba(low, base_periods)
        base_line = (hh_base + ll_base) / 2.0

        # Leading Span A (Senkou Span A): (Conversion + Base) / 2
        lead_line1 = (conversion_line + base_line) / 2.0

        # Leading Span B (Senkou Span B): (highest high + lowest low) / 2
        _, hh_spanb = rolling_min_max_numba(high, span_b_periods)
        ll_spanb, _ = rolling_min_max_numba(low, span_b_periods)
        lead_line2 = (hh_spanb + ll_spanb) / 2.0

        # Displace cloud forward (Pine: offset = displacement - 1)
        lag = displacement - 1
        cloud_upper = np.full(n, np.nan, dtype=np.float64)
        cloud_lower = np.full(n, np.nan, dtype=np.float64)

        if lag > 0 and n > lag:
            cloud_upper[lag:] = np.maximum(lead_line1[:-lag], lead_line2[:-lag])
            cloud_lower[lag:] = np.minimum(lead_line1[:-lag], lead_line2[:-lag])
        elif lag == 0:
            cloud_upper[:] = np.maximum(lead_line1, lead_line2)
            cloud_lower[:] = np.minimum(lead_line1, lead_line2)

        return {
            'cloud_upper': cloud_upper,
            'cloud_lower': cloud_lower,
            'future_green': lead_line1 >= lead_line2,
            'future_red': lead_line1 <= lead_line2,
            'conversion_line': conversion_line,
            'base_line': base_line,
            'lead_line1': lead_line1,
            'lead_line2': lead_line2,
        }

    except Exception as e:
        logger.error(f"Ichimoku calculation failed: {e}", exc_info=True)
        n = len(high) if high is not None else 1
        return {
            'cloud_upper': np.full(n, np.nan, dtype=np.float64),
            'cloud_lower': np.full(n, np.nan, dtype=np.float64),
            'future_green': np.zeros(n, dtype=bool),
            'future_red': np.zeros(n, dtype=bool),
            'conversion_line': np.full(n, np.nan, dtype=np.float64),
            'base_line': np.full(n, np.nan, dtype=np.float64),
            'lead_line1': np.full(n, np.nan, dtype=np.float64),
            'lead_line2': np.full(n, np.nan, dtype=np.float64),
        }

def calculate_rsi_guard_numpy(close: np.ndarray, rsi_len: int, kalman_len: int, ema_len: int) -> Tuple[np.ndarray, np.ndarray]:
    try:
        if close is None or len(close) < rsi_len + kalman_len + ema_len:
            default_len = len(close) if close is not None else 1
            logger.warning(
                f"RSI Guard: Insufficient data (len={default_len}, "
                f"required={rsi_len + kalman_len + ema_len})"
            )
            return (np.full(default_len, np.nan, dtype=np.float64),
                    np.full(default_len, np.nan, dtype=np.float64))

        rsi = calculate_rsi_core(close, rsi_len)
        smooth_rsi = kalman_loop(rsi, kalman_len, 0.01, 0.1)
        rsi_ema = ema_loop(smooth_rsi, float(ema_len))
        smooth_rsi = sanitize_array_numba(smooth_rsi, np.nan)
        rsi_ema    = sanitize_array_numba(rsi_ema, np.nan)

        return smooth_rsi, rsi_ema

    except Exception as e:
        logger.error(f"RSI Guard calculation failed: {e}")
        default_len = len(close) if close is not None else 1
        return (np.full(default_len, np.nan, dtype=np.float64),
                np.full(default_len, np.nan, dtype=np.float64))

def warmup_if_needed() -> None:
    """Hybrid warmup with comprehensive coverage & better logging"""
   
    is_prod = os.path.isfile("/.dockerenv")
    
    if aot_bridge.is_using_aot() or getattr(cfg, 'SKIP_WARMUP', False) or is_prod:
        reason = ("AOT active" if aot_bridge.is_using_aot() else 
                 "Explicitly disabled" if getattr(cfg, 'SKIP_WARMUP', False) else "Production mode")
        logger.info(f"🚨 Skipping JIT warmup ({reason})")
        if aot_bridge.is_using_aot():
            logger.debug("Native library status: Operational (Zero-warmup mode)")
        return

    logger.info("🔥 AOT not found. Warming up JIT (core indicators)...")
    warmup_start = time.time()
    
    try:
        test_data = np.random.random(150).astype(np.float64) * 1000.0
        
        now_ts = int(time.time())
        test_ts = np.arange(now_ts - (150 * 900), now_ts, 900, dtype=np.int64)
        
        _ = aot_bridge.ema_loop(test_data, 7.0)
        _ = aot_bridge.ema_loop_alpha(test_data, 0.2)
        _ = aot_bridge.ema_loop_pine(test_data, 7.0) 
        _ = aot_bridge.calculate_ppo_core(test_data, 7, 16, 5)
        _ = aot_bridge.calculate_rsi_core(test_data, 21)
        _ = aot_bridge.rolling_mean_numba(test_data, 14)
        _ = aot_bridge.kalman_loop(test_data, 10, 0.1, 0.01)
        _ = aot_bridge.rolling_min_max_numba(test_data, 23)
        _ = aot_bridge.calculate_atr_rma(test_data, test_data * 0.8, test_data, 5)
        _ = aot_bridge.calculate_adx_core(test_data, test_data * 0.8, test_data * 0.9, 14, 14)
        _ = aot_bridge.vwap_daily_loop_safe(test_data, test_data, test_ts)
        
        warmup_elapsed = time.time() - warmup_start
        logger.info(f"✅ JIT warmup complete ({warmup_elapsed:.2f}s)")

    except Exception as e:
        warmup_elapsed = time.time() - warmup_start
        logger.warning(f"🚫 Warmup failed (non-fatal, {warmup_elapsed:.2f}s): {e}")

def calculate_pivot_levels_numpy(high: np.ndarray, low: np.ndarray, close: np.ndarray, timestamps_daily: np.ndarray, timestamps_15m: np.ndarray, reference_time: int) -> Dict[str, float]:   
    piv = {k: 0.0 for k in ["P", "R1", "R2", "R3", "S1", "S2", "S3"]}
    
    try:
        if len(timestamps_daily) < 2:
            logger.warning("Pivot calc: insufficient data (< 2 daily candles)")
            return piv

        if np.any(np.isnan(high)) or np.any(np.isnan(low)) or np.any(np.isnan(close)):
            logger.warning("Pivot calc: NaN values in OHLC")
            return piv

        is_complete, reason = is_previous_day_complete(
            timestamps_15m,  # Use 15m for validation
            reference_time,
            
            min_candles=cfg.MIN_CANDLES_PER_DAY,
            buffer_seconds=cfg.DAILY_RESET_BUFFER_SEC
        )
        
        if not is_complete:
            if cfg.DEBUG_MODE:
                logger.debug(f"Pivot calc skipped: {reason}")
            return piv

        days = timestamps_daily // 86400
        current_day_number = reference_time // 86400
        unique_days = np.unique(days)
        past_days = unique_days[unique_days < current_day_number]
        
        if len(past_days) == 0:
            logger.warning(f"No daily candles found before day #{current_day_number}")
            return piv
            
        yesterday_day_number = past_days[-1]

        yesterday_mask = (days == yesterday_day_number)
        
        if not np.any(yesterday_mask):
            logger.warning(f"No daily candle for day #{yesterday_day_number}")
            return piv

        yesterday_high = high[yesterday_mask]
        yesterday_low = low[yesterday_mask]
        yesterday_close = close[yesterday_mask]

        if len(yesterday_high) == 0:
            logger.warning("No candles for pivot day")
            return piv

        H_prev = np.max(yesterday_high)
        L_prev = np.min(yesterday_low)
        C_prev = yesterday_close[-1]
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

        if cfg.DEBUG_MODE:
            logger.debug(
                f"✅ Pivots calculated | H={H_prev:.2f} L={L_prev:.2f} "
                f"C={C_prev:.2f} | P={P:.2f}"
            )

    except Exception as e:
        logger.error(f"Pivot calculation failed: {e}", exc_info=True)

    for k, val in piv.items():
        if np.isnan(val) or np.isinf(val) or val <= 0:
            logger.warning(f"Invalid pivot {k}: {val}, reset to 0.0")
            piv[k] = 0.0

    return piv

def calculate_gate_indicators_numpy(data_15m: Dict[str, np.ndarray], data_5m: Dict[str, np.ndarray], data_daily: Optional[Dict[str, np.ndarray]], reference_time: int) -> Optional[Dict[str, np.ndarray]]:
    """Cheap indicators needed ONLY for the main buy/sell gate."""
    try:
        close_15m = data_15m["close"]
        close_5m = data_5m["close"]
        n_15m = len(close_15m)
        n_5m = len(close_5m)

        ok, msg = _validate_ohlc_arrays(data_15m, n_15m)
        if not ok:
            logger.error(f"Gate indicators: 15m OHLC validation failed — {msg}")
            return None

        ok, msg = _validate_ohlc_arrays(data_5m, n_5m)
        if not ok:
            logger.error(f"Gate indicators: 5m OHLC validation failed — {msg}")
            return None

        results: Dict[str, Any] = {}

        # ── Trend: RMA 50/200 ──
        results['rma50_15'] = calculate_rma_numpy(close_15m, cfg.RMA_50_PERIOD)
        results['rma200_5'] = calculate_rma_numpy(close_5m, cfg.RMA_200_PERIOD)

        # ── Ichimoku (cloud + TK guard) ──
        if cfg.ICHIMOKU_CLOUD_ENABLED or cfg.ICHIMOKU_TK_GUARD_ENABLED:
            ichimoku = calculate_ichimoku_numpy(
                data_15m["high"], data_15m["low"], close_15m,
                cfg.ICHIMOKU_CONVERSION_PERIODS,
                cfg.ICHIMOKU_BASE_PERIODS,
                cfg.ICHIMOKU_SPANB_PERIODS,
                cfg.ICHIMOKU_DISPLACEMENT,
            )
            results['ichimoku_cloud_upper'] = ichimoku['cloud_upper']
            results['ichimoku_cloud_lower'] = ichimoku['cloud_lower']
            results['ichimoku_future_green'] = ichimoku['future_green']
            results['ichimoku_future_red'] = ichimoku['future_red']
            results['ichimoku_conversion_line'] = ichimoku['conversion_line']
            results['ichimoku_base_line'] = ichimoku['base_line']
        else:
            nan_arr = np.full(n_15m, np.nan, dtype=np.float64)
            bool_arr = np.zeros(n_15m, dtype=bool)
            results['ichimoku_cloud_upper'] = nan_arr.copy()
            results['ichimoku_cloud_lower'] = nan_arr.copy()
            results['ichimoku_future_green'] = bool_arr.copy()
            results['ichimoku_future_red'] = bool_arr.copy()
            results['ichimoku_conversion_line'] = nan_arr.copy()
            results['ichimoku_base_line'] = nan_arr.copy()

        # ── Fast Ichimoku (9,26,52,26) — alert-only, independent of the slow Ichimoku gate above ──
        if cfg.ENABLE_FAST_ICHIMOKU_CLOUD_CROSS or cfg.ENABLE_FAST_ICHIMOKU_TENKAN_CROSS:
            fast_ichimoku = calculate_ichimoku_numpy(
                data_15m["high"], data_15m["low"], close_15m,
                cfg.FAST_ICHIMOKU_CONVERSION_PERIODS,
                cfg.FAST_ICHIMOKU_BASE_PERIODS,
                cfg.FAST_ICHIMOKU_SPANB_PERIODS,
                cfg.FAST_ICHIMOKU_DISPLACEMENT,
            )
            results['fast_ichimoku_cloud_upper'] = fast_ichimoku['cloud_upper']
            results['fast_ichimoku_cloud_lower'] = fast_ichimoku['cloud_lower']
            results['fast_ichimoku_future_green'] = fast_ichimoku['future_green']
            results['fast_ichimoku_future_red'] = fast_ichimoku['future_red']
            results['fast_ichimoku_conversion_line'] = fast_ichimoku['conversion_line']
            results['fast_ichimoku_base_line'] = fast_ichimoku['base_line']
        else:
            nan_arr_fast = np.full(n_15m, np.nan, dtype=np.float64)
            bool_arr_fast = np.zeros(n_15m, dtype=bool)
            results['fast_ichimoku_cloud_upper'] = nan_arr_fast.copy()
            results['fast_ichimoku_cloud_lower'] = nan_arr_fast.copy()
            results['fast_ichimoku_future_green'] = bool_arr_fast.copy()
            results['fast_ichimoku_future_red'] = bool_arr_fast.copy()
            results['fast_ichimoku_conversion_line'] = nan_arr_fast.copy()
            results['fast_ichimoku_base_line'] = nan_arr_fast.copy()

        # ── Volatility: ATR + ADX ──
        results['atr_short'] = calculate_atr_rma(
            data_15m["high"], data_15m["low"], data_15m["close"], cfg.ATR_SHORT
        )
        results['atr_long'] = calculate_atr_rma(
            data_15m["high"], data_15m["low"], data_15m["close"], cfg.ATR_LONG
        )
        results['adx'] = calculate_adx_core(
            data_15m["high"], data_15m["low"], data_15m["close"],
            cfg.ADX_DI_LENGTH, cfg.ADX_SMOOTHING_LENGTH
        )

        ok, msg = _validate_atr_arrays(results['atr_short'], results['atr_long'], n_15m)
        if not ok:
            logger.warning(f"Gate indicators: ATR validation — {msg}")

        # ── Volume EMA (used by CPR bypass) ──
        results['volume_ema'] = calculate_volume_ema_numpy(
            data_15m["volume"], cfg.VOLUME_EMA_LENGTH
        )

        # ── Trend gates ──
        if cfg.ENABLE_PPO_GATE:
            pg, pgs = calculate_ppo_numpy(
                close_15m, cfg.PPO_GATE_FAST, cfg.PPO_GATE_SLOW, cfg.PPO_GATE_SIGNAL
            )
            results['ppo_gate'] = pg
            results['ppo_gate_signal'] = pgs
        else:
            results['ppo_gate'] = np.full(n_15m, np.nan, dtype=np.float64)
            results['ppo_gate_signal'] = np.full(n_15m, np.nan, dtype=np.float64)

        if cfg.RSI_GUARD_ENABLED:
            rgs, rge = calculate_rsi_guard_numpy(
                close_15m, cfg.RSI_GUARD_RSI_LEN,
                cfg.RSI_GUARD_KALMAN_LEN, cfg.RSI_GUARD_EMA_LEN
            )
            results['rsi_guard_smooth'] = rgs
            results['rsi_guard_ema'] = rge
        else:
            results['rsi_guard_smooth'] = np.full(n_15m, np.nan, dtype=np.float64)
            results['rsi_guard_ema'] = np.full(n_15m, np.nan, dtype=np.float64)

        if cfg.RMA_CLOUD_ENABLED:
            results['rma_cloud_fast_15'] = calculate_rma_numpy(close_15m, cfg.RMA_CLOUD_FAST_PERIOD)
        else:
            results['rma_cloud_fast_15'] = np.full(n_15m, np.nan, dtype=np.float64)

        # ── CPR (daily) ──
        if cfg.ENABLE_CPR and data_daily is not None:
            try:
                d_high, d_low, d_close, d_ts = _find_closed_daily_candle(
                    data_daily, reference_time
                )
                _pivot = (d_high + d_low + d_close) / 3.0
                _bc = (d_high + d_low) / 2.0
                _tc = (_pivot - _bc) + _pivot
                results['nr_cpr'] = abs(_tc - _bc)
                results['cpr_ok'] = results['nr_cpr'] < (d_close * cfg.CPR_THRESHOLD_PCT)
                results['prev_day_close'] = d_close
            except CprNotReadyError as e:
                logger.debug(f"CPR not ready: {e}")
                results['nr_cpr'] = float('nan')
                results['cpr_ok'] = False
            except ValueError as e:
                logger.warning(f"CPR skipped — bad candle: {e}")
                results['nr_cpr'] = float('nan')
                results['cpr_ok'] = False
            except Exception as e:
                logger.error(f"CPR unexpected error: {e}", exc_info=True)
                results['nr_cpr'] = float('nan')
                results['cpr_ok'] = False
        elif not cfg.ENABLE_CPR:
            results['nr_cpr'] = float('nan')
            results['cpr_ok'] = True
        else:
            logger.warning("CPR gate: ENABLE_CPR=True but data_daily is None")
            results['nr_cpr'] = float('nan')
            results['cpr_ok'] = False

        # Sanitize
        for key in ('rma50_15', 'rma200_5', 'adx', 'atr_short', 'atr_long',
                    'ppo_gate', 'ppo_gate_signal', 'rsi_guard_smooth',
                    'rsi_guard_ema', 'volume_ema', 'rma_cloud_fast_15'):
            arr = results[key]
            if np.any(np.isinf(arr)):
                results[key] = np.clip(arr, -Constants.INFINITY_CLAMP, Constants.INFINITY_CLAMP)

        return results

    except Exception as e:
        logger.error(f"calculate_gate_indicators_numpy failed: {e}", exc_info=True)
        return None

def calculate_alert_indicators_numpy(data_15m: Dict[str, np.ndarray], data_5m: Dict[str, np.ndarray], data_daily: Optional[Dict[str, np.ndarray]], reference_time: int) -> Optional[Dict[str, np.ndarray]]:
    """Expensive indicators needed ONLY when a pair passes the main gate."""
    try:
        close_15m = data_15m["close"]
        n_15m = len(close_15m)

        results: Dict[str, Any] = {}

        # PPO (fast) — used by alert triggers
        ppo, ppo_signal = calculate_ppo_numpy(
            close_15m, cfg.PPO_FAST, cfg.PPO_SLOW, cfg.PPO_SIGNAL
        )
        results['ppo'] = ppo
        results['ppo_signal'] = ppo_signal

        # Smooth RSI — used by alert triggers
        results['smooth_rsi'], results['smooth_rsi_ema'] = calculate_smooth_rsi_numpy(
            close_15m, cfg.SRSI_RSI_LEN, cfg.SRSI_KALMAN_LEN, cfg.SRSI_EMA_LEN
        )

        # VWAP — used by VWAP alerts
        if cfg.ENABLE_VWAP:
            results['vwap'] = calculate_vwap_numpy(
                data_15m["high"], data_15m["low"], close_15m,
                data_15m["volume"], data_15m["timestamp"], reference_time
            )
        else:
            results['vwap'] = np.full(n_15m, np.nan, dtype=np.float64)

        if cfg.ENABLE_HIST_RMA:
            rma_fast = calculate_rma_numpy(close_15m, cfg.HIST_RMA_FAST)
            rma_slow = calculate_rma_numpy(close_15m, cfg.HIST_RMA_SLOW)
            results['hist_rma'] = rma_fast - rma_slow
        else:
            results['hist_rma'] = np.full(n_15m, np.nan, dtype=np.float64)

        if cfg.ENABLE_PIVOT and data_daily is not None:
            results['pivots'] = calculate_pivot_levels_numpy(
                data_daily["high"], data_daily["low"], data_daily["close"],
                data_daily["timestamp"], data_15m["timestamp"], reference_time
            )
        else:
            results['pivots'] = {}

        # Sanitize
        for key in ('ppo', 'ppo_signal', 'smooth_rsi', 'smooth_rsi_ema', 'hist_rma'):
            arr = results[key]
            if np.any(np.isinf(arr)):
                results[key] = np.clip(arr, -Constants.INFINITY_CLAMP, Constants.INFINITY_CLAMP)

        return results

    except Exception as e:
        logger.error(f"calculate_alert_indicators_numpy failed: {e}", exc_info=True)
        return None

async def _blanket_reset_pair(
    sdb: RedisStateStore, pair_name: str, logger_pair: logging.Logger
) -> int:
    """Reset every active alert state for a pair without computing per-indicator logic."""
    all_keys = list(ALERT_KEYS.values())
    previous_states = await sdb.batch_get_all_alert_states(pair_name, all_keys)
    resets = [
        (f"{pair_name}:{rk}", "INACTIVE", None)
        for rk in all_keys
        if previous_states.get(rk, False)
    ]
    if resets:
        await sdb.atomic_batch_update(resets)
        logger_pair.debug(
            f"[{pair_name}] Blanket reset: {len(resets)} active state(s) cleared"
        )
    return len(resets)

async def _clear_all_redis_states(sdb: RedisStateStore, pairs: List[str], logger: logging.Logger) -> Tuple[int, int]:
    if sdb.degraded or not sdb._redis:
        logger.warning("Redis degraded — skipping mass state purge")
        return 0, 0

    state_hash_keys: List[str] = [f"{sdb.state_prefix}{pair}" for pair in pairs]
    dedup_keys: List[str] = [
        f"{RedisKeyPrefix.RECENT_ALERT}{pair}:{alert_key}"
        for pair in pairs
        for alert_key in ALERT_KEYS.values()
    ]

    deleted_states = 0
    deleted_dedups = 0

    try:
        if state_hash_keys:
            deleted_states = await sdb._redis.delete(*state_hash_keys)
        if dedup_keys:
            deleted_dedups = await sdb._redis.delete(*dedup_keys)

        logger.info(
            f"🧹 MASS RESET complete | "
            f"State hash keys deleted: {deleted_states}/{len(state_hash_keys)} | "
            f"Dedup keys deleted: {deleted_dedups}/{len(dedup_keys)}"
        )
        return deleted_states, deleted_dedups
    except Exception as e:
        logger.error(f"Mass reset failed: {e}")
        return 0, 0

def _validate_ohlc_arrays(data_15m: Dict[str, np.ndarray], 
                         expected_len: int) -> Tuple[bool, Optional[str]]:  
    required_keys = ["open", "high", "low", "close"]    
    for key in required_keys:
        if key not in data_15m:
            return False, f"Missing OHLC key '{key}'"
        
        arr = data_15m[key]
        if arr is None or len(arr) == 0:
            return False, f"OHLC '{key}' is None or empty"
        
        if len(arr) != expected_len:
            return False, f"Length mismatch in '{key}': {len(arr)} != {expected_len}"
    
    return True, None

def get_atr_percentile(atr_long_arr: np.ndarray, i15: int, cfg: BotConfig) -> Optional[float]:
    """Percentile rank (0.0=calmest .. 1.0=most volatile) of current long-ATR
    against its trailing ATR_PCTL_LOOKBACK window. None if insufficient history."""
    lookback = cfg.ATR_PCTL_LOOKBACK
    start = i15 - lookback
    if start < 0:
        return None

    window = atr_long_arr[start:i15]
    valid = window[~np.isnan(window)]
    if len(valid) < cfg.ATR_PCTL_MIN_HISTORY:
        return None

    current = atr_long_arr[i15]
    if np.isnan(current) or current <= 0:
        return None

    n = len(valid)
    count_lt = np.sum(valid < current)
    count_eq = np.sum(valid == current)
    return (count_lt + 0.5 * count_eq) / n

def get_atr_percentile_smoothed(atr_long_arr: np.ndarray, i15: int, cfg: BotConfig, ema_period: int = 5) -> Optional[float]:
    required_depth = cfg.ATR_PCTL_LOOKBACK + ema_period
    if i15 < required_depth:
        return get_atr_percentile(atr_long_arr, i15, cfg)  # unsmoothed during warmup

    pctl_values = []
    for j in range(i15 - ema_period + 1, i15 + 1):
        p = get_atr_percentile(atr_long_arr, j, cfg)
        if p is not None:
            pctl_values.append(p)

    if not pctl_values:
        return None

    alpha = 2.0 / (ema_period + 1)
    ema = pctl_values[0]
    for val in pctl_values[1:]:
        ema = alpha * val + (1.0 - alpha) * ema
    return ema

def _scale_by_pctl(pctl: Optional[float], calm: float, volatile: float, fallback_pctl: float = 0.5) -> float:
    """Linearly scales a [calm, volatile] range by pctl. Falls back to fallback_pctl
    (midpoint by default) when pctl is unavailable, so callers always get a usable value."""
    p = pctl if pctl is not None else fallback_pctl
    val = calm + p * (volatile - calm)
    lo, hi = min(calm, volatile), max(calm, volatile)
    return max(lo, min(hi, val))

def get_adaptive_rvol_threshold(atr_long_arr: np.ndarray, i15: int, cfg: BotConfig) -> Optional[float]:
    if not cfg.ATR_ADAPTIVE_ENABLED:
        return None
    pctl = get_atr_percentile(atr_long_arr, i15, cfg)
    if pctl is None:
        return None
    return _scale_by_pctl(pctl, cfg.ADAPTIVE_MULT_CALM, cfg.ADAPTIVE_MULT_VOLATILE)

def get_adaptive_adx_threshold(adx_arr: np.ndarray, i15: int, cfg: BotConfig) -> float:
    """Self-referential: ADX threshold = the value at cfg.ADX_ADAPTIVE_TARGET_PCTL of
    this pair's own trailing ADX history — so each pair gets its own baseline instead
    of one fixed number for all ~30 pairs. Falls back to ADX_ADAPTIVE_FALLBACK when
    ATR_ADAPTIVE_ENABLED=False or there isn't enough history yet."""
    if not cfg.ATR_ADAPTIVE_ENABLED:
        return cfg.ADX_ADAPTIVE_FALLBACK

    lookback = cfg.ATR_PCTL_LOOKBACK
    start = i15 - lookback
    if start < 0:
        return cfg.ADX_ADAPTIVE_FALLBACK

    window = adx_arr[start:i15]
    valid = window[~np.isnan(window)]
    if len(valid) < cfg.ATR_PCTL_MIN_HISTORY:
        return cfg.ADX_ADAPTIVE_FALLBACK

    sorted_valid = np.sort(valid)
    n = len(sorted_valid)

    band_half = cfg.ADX_ADAPTIVE_BAND_WIDTH / 2.0
    if band_half <= 0.0:
        idx = int(cfg.ADX_ADAPTIVE_TARGET_PCTL / 100.0 * (n - 1))
        idx = max(0, min(n - 1, idx))
        return float(sorted_valid[idx])

    lo_pctl = max(0.0, cfg.ADX_ADAPTIVE_TARGET_PCTL - band_half)
    hi_pctl = min(100.0, cfg.ADX_ADAPTIVE_TARGET_PCTL + band_half)
    lo_idx = max(0, int(lo_pctl / 100.0 * (n - 1)))
    hi_idx = min(n - 1, int(hi_pctl / 100.0 * (n - 1)))
    band = sorted_valid[lo_idx:hi_idx + 1]
    return float(np.median(band))

def get_adaptive_adx_threshold_smoothed(adx_arr: np.ndarray, i15: int, cfg: BotConfig, ema_period: int = 5) -> float:
    required_depth = cfg.ATR_PCTL_LOOKBACK + ema_period
    if i15 < required_depth:
        return get_adaptive_adx_threshold(adx_arr, i15, cfg)  # unsmoothed fallback

    thresholds = [get_adaptive_adx_threshold(adx_arr, j, cfg) for j in range(i15 - ema_period + 1, i15 + 1)]
    alpha = 2.0 / (ema_period + 1)
    ema = thresholds[0]
    for val in thresholds[1:]:
        ema = alpha * val + (1.0 - alpha) * ema
    return ema

def _get_smoothed_pctl(atr_long_arr, i15, cfg) -> Optional[float]:
    return get_atr_percentile_smoothed(atr_long_arr, i15, cfg) if cfg.ATR_ADAPTIVE_ENABLED else None

def get_adaptive_threshold(atr_long_arr, i15, cfg, calm_attr: str, volatile_attr: str) -> float:
    pctl = _get_smoothed_pctl(atr_long_arr, i15, cfg)
    return _scale_by_pctl(pctl, getattr(cfg, calm_attr), getattr(cfg, volatile_attr))

def get_adaptive_ppo_threshold(atr_long_arr, i15, cfg) -> float:
    return get_adaptive_threshold(atr_long_arr, i15, cfg, "PPO_ADAPTIVE_CALM", "PPO_ADAPTIVE_VOLATILE")

def get_adaptive_rsi_thresholds(atr_long_arr: np.ndarray, i15: int, cfg: BotConfig) -> Tuple[float, float]:
    pctl = _get_smoothed_pctl(atr_long_arr, i15, cfg)
    buy = _scale_by_pctl(pctl, cfg.RSI_ADAPTIVE_BUY_CALM, cfg.RSI_ADAPTIVE_BUY_VOLATILE)
    sell = _scale_by_pctl(pctl, cfg.RSI_ADAPTIVE_SELL_CALM, cfg.RSI_ADAPTIVE_SELL_VOLATILE)
    return buy, sell

def get_adaptive_cpr_threshold(atr_long_arr, i15, cfg) -> float:
    return get_adaptive_threshold(atr_long_arr, i15, cfg, "CPR_ADAPTIVE_CALM", "CPR_ADAPTIVE_VOLATILE")

def _validate_atr_arrays(atr_short: np.ndarray, atr_long: np.ndarray, 
                        expected_len: int) -> Tuple[bool, Optional[str]]:   
    if atr_short is None or atr_long is None:
        return False, "ATR arrays are None"
    
    if len(atr_short) == 0 or len(atr_long) == 0:
        return False, "ATR arrays are empty"
    
    if len(atr_short) != expected_len:
        return False, f"atr_short length mismatch: {len(atr_short)} != {expected_len}"
    
    if len(atr_long) != expected_len:
        return False, f"atr_long length mismatch: {len(atr_long)} != {expected_len}"
    
    return True, None

class SessionManager:
    _session: ClassVar[Optional[aiohttp.ClientSession]] = None
    _ssl_context: ClassVar[Optional[ssl.SSLContext]] = None
    _lock: ClassVar[Optional[asyncio.Lock]] = None
    _creation_time: ClassVar[float] = 0.0

    @classmethod
    def _get_lock(cls) -> asyncio.Lock:
        if cls._lock is None:
            cls._lock = asyncio.Lock()
        return cls._lock

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
        old_session_to_close: Optional[aiohttp.ClientSession] = None
        async with cls._get_lock():  
            should_recreate = cls._session is None or cls._session.closed
            if should_recreate:
                if cls._session and not cls._session.closed:
                    old_session_to_close = cls._session

                connector = TCPConnector(
                    limit=max(cfg.TCP_CONN_LIMIT, cfg.MAX_PARALLEL_FETCH),
                    limit_per_host=max(cfg.TCP_CONN_LIMIT_PER_HOST, cfg.MAX_PARALLEL_FETCH),
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
                cls._creation_time = time.monotonic()

                if cfg.DEBUG_MODE:
                    logger.debug("HTTP session created")

            new_session = cls._session

        if old_session_to_close is not None:
            try:
                await old_session_to_close.close()
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.warning(f"Error closing old session: {e}")

        return new_session

    @classmethod
    async def close_session(cls) -> None:
        session_to_close: Optional[aiohttp.ClientSession] = None
        session_age = 0.0
        async with cls._get_lock():
            if cls._session and not cls._session.closed:
                session_to_close = cls._session
                session_age = time.monotonic() - cls._creation_time
                cls._session = None
                cls._creation_time = 0.0
            else:
                logger.debug("Session already closed or not created")

        if session_to_close is not None:
            try:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"Closing HTTP session | Age: {session_age:.1f}s"
                    )
                await session_to_close.close()
                await asyncio.sleep(0.1)  # OPTIMIZED: Reduced from 0.25s
                logger.info("HTTP session closed successfully")
            except Exception as e:
                logger.warning(f"Error closing session: {e}")

    @classmethod
    def get_stats(cls) -> Dict[str, Any]:
        if cls._session is None:
            return {
                "active": False,
                "age_seconds": 0.0,
            }
        age = time.monotonic() - cls._creation_time if cls._creation_time > 0 else 0.0
        return {
            "active": not cls._session.closed,
            "age_seconds": round(age, 1),
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

def compute_backoff(base: float, attempt: int, cap: float = 30.0, jitter_range: Tuple[float, float] = (0.1, 0.5)) -> float:
    """Exponential backoff with jitter. base=starting delay in seconds, attempt=1-indexed retry count."""
    base_delay = min(base * (2 ** (attempt - 1)), cap)
    jitter = base_delay * random.uniform(*jitter_range)
    return base_delay + jitter

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
                    try:
                        retry_val = int(retry_after) if retry_after else 2
                    except (ValueError, TypeError):
                        retry_val = 5
                    wait_sec = min(retry_val, Constants.CIRCUIT_BREAKER_MAX_WAIT)
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
                        total_delay = compute_backoff(backoff, attempt, cap=Constants.CIRCUIT_BREAKER_MAX_WAIT / 10)
                        await asyncio.sleep(total_delay)
                    continue
                
                if resp.status >= 400:
                    logger.error(
                        f"Client error {resp.status} for {url[:80]} | "
                        f"This usually indicates invalid request - not retrying"
                    )
                    return None
                
                try:
                    data = await resp.json(loads=json_loads)
                except (JSONDecodeError, TypeError, ValueError) as e:
                    retry_stats[RetryCategory.API_ERROR] += 1
                    logger.warning(
                        f"Malformed JSON on 200 OK (attempt {attempt}/{retries}) | "
                        f"URL: {url[:80]} | Error: {str(e)[:100]}"
                    )
                    if attempt < retries:
                        total_delay = compute_backoff(backoff, attempt, cap=Constants.CIRCUIT_BREAKER_MAX_WAIT / 10)
                        await asyncio.sleep(total_delay)
                    continue         
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
                total_delay = compute_backoff(backoff, attempt, cap=Constants.CIRCUIT_BREAKER_MAX_WAIT / 10)
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
                total_delay = compute_backoff(backoff, attempt, cap=Constants.CIRCUIT_BREAKER_MAX_WAIT / 10)
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
        while True:
            sleep_needed = 0.0
            async with self.lock:
                now = time.monotonic()
                while self.requests and now - self.requests[0] > 60.0:
                    self.requests.popleft()
                if len(self.requests) < self.max_per_minute:
                    self.requests.append(now)
                    self.last_request_time = now
                    break
                else:
                    oldest_request_age = now - self.requests[0]
                    wait_needed = max(0.0, 60.0 - oldest_request_age)
                    sleep_needed = wait_needed + random.uniform(0.05, 0.2)
                    self.total_waits += 1
                    logger.debug(
                        f"Rate limit reached ({len(self.requests)}/{self.max_per_minute}), "
                        f"sleeping {sleep_needed:.2f}s | Total waits: {self.total_waits}"
                    )
            
            t0 = time.monotonic()
            try:
                await asyncio.sleep(sleep_needed)
            except asyncio.CancelledError:
                self.total_wait_time += max(0.0, time.monotonic() - t0)
                raise
            self.total_wait_time += time.monotonic() - t0

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
        self._lock = asyncio.Lock()          # NEW

    async def record_success(self) -> None:  # NEW: async
        async with self._lock:
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

    async def record_failure(self) -> None:   # NEW: async
        async with self._lock:
            self.failures += 1
            self.last_failure_time = time.time()

            if self.failures >= self.failure_threshold and self.state == "CLOSED":
                logger.warning(
                    f"⚠️ Circuit breaker: OPENED after {self.failures} failures. "
                    f"Blocking requests for {self.recovery_timeout}s"
                )
                self.state = "OPEN"

    async def can_attempt(self) -> Tuple[bool, Optional[str]]:  # NEW: async
        async with self._lock:
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
        self.timeout = cfg.HTTP_TIMEOUT
        self.rate_limiter = RateLimitedFetcher(
            max_per_minute=cfg.RATE_LIMIT_PER_MINUTE,
            concurrency=max_parallel,
        )
        self.confirm_rate_limiter = RateLimitedFetcher(
            max_per_minute=cfg.CONFIRM_RATE_LIMIT_PER_MINUTE,
            concurrency=2,
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
  
    async def fetch_candles(self, symbol: str, resolution: str, limit: int, reference_time: int, expected_open_15: Optional[int] = None, for_confirmation: bool = False) -> Optional[Dict[str, Any]]:
        can_proceed, reason = await self.circuit_breaker.can_attempt()
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

        buffer_periods = Constants.CANDLE_FETCH_BUFFER_PERIODS
        to_time = reference_time + (interval_seconds * buffer_periods)
        from_time = expected_open_ts - (limit * interval_seconds)

        params = {
            "resolution": resolution,
            "symbol": symbol,
            "from": int(from_time),
            "to": int(to_time),
        }
        url = f"{self.api_base}/v2/chart/history"
        limiter = self.confirm_rate_limiter if for_confirmation else self.rate_limiter

        data = await limiter.call(
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
                await self.circuit_breaker.record_success()
                self.fetch_stats["candles"]["success"] += 1

                num_candles = len(result.get("t", []))
                if num_candles > 0:
                    last_open = result["t"][-1]
                    diff = abs(expected_open_ts - last_open)

                    if diff > Constants.API_TIMESTAMP_TOLERANCE_SEC:
                        if last_open < expected_open_ts:
                            if logger.isEnabledFor(logging.DEBUG):
                                logger.debug(
                                    f"⚠️ API DELAY | {symbol} {resolution} | "
                                    f"Expected: {format_ist_time(expected_open_ts)} | "
                                    f"Got: {format_ist_time(last_open)} "
                                    f"(Diff: {diff}s > tolerance {Constants.API_TIMESTAMP_TOLERANCE_SEC}s)"
                                )
                        else:
                            logger.debug(f"API Ahead | {symbol} {resolution} | Diff: {diff}s")
                    else:
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(
                                f"✅ Scanned {symbol} {resolution} | "
                                f"Latest: {format_ist_time(last_open)} | Candles: {num_candles}"
                            )
                return data
            else:
                logger.warning(f"Candles response missing fields | Symbol: {symbol}")
                self.fetch_stats["candles"]["failed"] += 1
                await self.circuit_breaker.record_failure()
        else:
            logger.warning(f"Candles fetch failed | Symbol: {symbol}")
            self.fetch_stats["candles"]["failed"] += 1
            await self.circuit_breaker.record_failure()

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
        results = await asyncio.wait_for(
            asyncio.gather(*all_tasks, return_exceptions=True),
            timeout=cfg.FETCH_PHASE_TIMEOUT_SEC
        )
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

def validate_indicator_values(indicators_dict: Dict[str, float], names: List[str]) -> Tuple[bool, str]:
    for name in names:
        val = indicators_dict.get(name)
        if val is None or np.isnan(val):
            return False, f"{name} is NaN"
    return True, "OK"

def validate_candle_for_alerts(data_15m: Dict[str, np.ndarray], candle_index: int, reference_time: int, pair_name: str, min_wick_ratio: float = 0.20) -> Tuple[bool, bool, Optional[Dict[str, Any]], Optional[str]]:
    try:
        o = float(data_15m["open"][candle_index])
        h = float(data_15m["high"][candle_index])
        l = float(data_15m["low"][candle_index])
        c = float(data_15m["close"][candle_index])
        ts = int(data_15m["timestamp"][candle_index])
        vol = float(data_15m["volume"][candle_index])
    except (IndexError, KeyError, ValueError, TypeError) as e:
        return False, False, None, f"Data access error: {e}"
    
    if any(np.isnan([o, h, l, c])) or any(np.isinf([o, h, l, c])):
        return False, False, None, f"Invalid OHLC: contains NaN or Inf"
    
    if any(x <= 0 for x in [o, h, l, c]):
        return False, False, None, f"Invalid OHLC: non-positive values"
    
    if not (l <= o <= h and l <= c <= h):
        return False, False, None, f"Invalid OHLC: relationships broken (O={o:.4f} H={h:.4f} L={l:.4f} C={c:.4f})"
    
    if vol <= 0:
        return False, False, None, "Zero volume candle — likely exchange placeholder or maintenance window"
    
    interval_seconds = 15 * 60
    candle_age = reference_time - ts
    
    candle_close_time = ts + interval_seconds
    time_since_candle_closed = reference_time - candle_close_time
     
    if not candle_is_stable(ts, reference_time, interval_minutes=15):
        return False, False, None, (
            f"Candle at {format_ist_time(ts)} not stable yet "
            f"(buffer {cfg.CANDLE_MIN_AGE_BUFFER}s, min age {Constants.MIN_CANDLE_AGE_FROM_OPEN}s)"
        )
   
    if candle_age > cfg.MAX_CANDLE_STALENESS_SEC:
        return False, False, None, (
            f"Candle age {candle_age}s from open is > {cfg.MAX_CANDLE_STALENESS_SEC}s. "
            f"This is a stale candle from a previous period! "
            f"(Opened: {format_ist_time(ts)}, Current: {format_ist_time(reference_time)})"
        )
    if cfg.DEBUG_MODE:
        logger.debug(
            f"[{pair_name}] Validating candle at index {candle_index}: "
            f"Open={format_ist_time(ts)}, Age={candle_age}s, "
            f"O={o:.2f} H={h:.2f} L={l:.2f} C={c:.2f}"
        )
    if candle_index > 0:
        prev_candle_ts = int(data_15m["timestamp"][candle_index - 1])
        expected_prev_ts = ts - interval_seconds
        if abs(prev_candle_ts - expected_prev_ts) > 60:
            return False, False, None, (
                f"Gap before signal candle: expected prev at {format_ist_time(expected_prev_ts)} "
                f"but found {format_ist_time(prev_candle_ts)} "
                f"(diff={abs(prev_candle_ts - expected_prev_ts)}s). Crossover data unreliable."
            )

    if candle_index + 1 < len(data_15m["timestamp"]):
        next_candle_ts = int(data_15m["timestamp"][candle_index + 1])
        expected_next_ts = ts + interval_seconds
        next_candle_is_still_forming = (next_candle_ts + interval_seconds) > reference_time

        if not next_candle_is_still_forming and abs(next_candle_ts - expected_next_ts) > 60:
            return False, False, None, ( 
                f"Gap detected: Expected next candle at {format_ist_time(expected_next_ts)} " 
                f"but found at {format_ist_time(next_candle_ts)} " f"(diff={abs(next_candle_ts - expected_next_ts)}s). Data may be incomplete." 
            ) 
    candle_range = h - l
    
    if candle_range < 1e-9:
        return False, False, None, f"Zero-range candle (H={h:.4f} L={l:.4f})"
    
    if c > o:
        is_green = True
        is_red = False
        candle_color = "GREEN"
        upper_wick = h - c
        lower_wick = o - l
        body = c - o
    elif c < o:
        is_green = False
        is_red = True
        candle_color = "RED"
        upper_wick = h - o
        lower_wick = c - l
        body = o - c
    else:
        is_green = False
        is_red = False
        candle_color = "DOJI"
        upper_wick = h - o
        lower_wick = c - l
        body = 0.0
    
    calculated_range = upper_wick + body + lower_wick

    if abs(calculated_range - candle_range) > 1e-6 * max(candle_range, 1.0):
        return False, False, None, (
            f"Candle structure error: wicks+body={calculated_range:.6f} "
            f"!= range={candle_range:.6f}"
        )
   
    body_ratio      = body / candle_range
    upper_wick_ratio = upper_wick / candle_range
    lower_wick_ratio = lower_wick / candle_range

    is_valid_for_buy  = (is_green and upper_wick_ratio < min_wick_ratio and body_ratio >= Constants.MIN_BODY_RATIO)
    is_valid_for_sell = (is_red   and lower_wick_ratio < min_wick_ratio and body_ratio >= Constants.MIN_BODY_RATIO)

    candle_info = {
        "timestamp": ts,
        "open": o,
        "high": h,
        "low": l,
        "close": c,
        "volume": vol,
        "range": candle_range,
        "color": candle_color,
        "is_green": is_green,
        "is_red": is_red,
        "body": body,
        "body_ratio": body_ratio,
        "upper_wick": upper_wick,
        "lower_wick": lower_wick,
        "upper_wick_ratio": upper_wick_ratio,
        "lower_wick_ratio": lower_wick_ratio,
        "candle_age_seconds": candle_age,
        "time_since_closed": time_since_candle_closed,
        "is_valid_for_buy": is_valid_for_buy,
        "is_valid_for_sell": is_valid_for_sell,
    }
    if not is_valid_for_buy and not is_valid_for_sell:
        if is_green:
            reason = (
                f"GREEN candle rejected: upper wick {upper_wick_ratio*100:.1f}% "
                f"≥ {min_wick_ratio*100:.0f}% or body {body_ratio*100:.1f}% < {Constants.MIN_BODY_RATIO*100:.0f}%"
            )
        elif is_red:
            reason = (
                f"RED candle rejected: lower wick {lower_wick_ratio*100:.1f}% "
                f"≥ {min_wick_ratio*100:.0f}% or body {body_ratio*100:.1f}% < {Constants.MIN_BODY_RATIO*100:.0f}%"
            )
        else:
            reason = f"DOJI candle rejected: body {body_ratio*100:.1f}% < {Constants.MIN_BODY_RATIO*100:.0f}%"
        return False, False, candle_info, reason

    return is_valid_for_buy, is_valid_for_sell, candle_info, None

def parse_candles_to_numpy(result: Optional[Dict[str, Any]]) -> Optional[PriceData]:
    try:   
        if not result or not isinstance(result, dict):
            logger.warning("parse_candles_to_numpy: result is None or not dict")
            return None
    
        res = result.get("result", {}) or {}
        required_fields = ("t", "o", "h", "l", "c", "v")
    
        if not all(k in res for k in required_fields):
            missing = [k for k in required_fields if k not in res]
            logger.warning(
                f"parse_candles_to_numpy: Missing required fields: {missing} | "
                f"Available: {list(res.keys())}"
            )
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
    
        except (ValueError, TypeError) as e:
            logger.error(f"parse_candles_to_numpy: Failed to convert data to arrays: {e}")
            return None
    
        n = len(data["timestamp"])
    
        if n == 0:
            logger.warning("parse_candles_to_numpy: empty candle array (n=0)")
            return None
    
        lengths = {k: len(data[k]) for k in ["open", "high", "low", "close", "volume"]}
        if len(set(lengths.values())) != 1:
            bad = {k: v for k, v in lengths.items() if v != n}
            logger.error(f"Length mismatch: {bad}")
            return None
    
        data["timestamp"] = np.where(data["timestamp"] > 1_000_000_000_000, data["timestamp"] // 1000, data["timestamp"])

        o, h, l, c = data["open"], data["high"], data["low"], data["close"]
    
        error_mask = (
            np.isnan(o) | np.isnan(h) | np.isnan(l) | np.isnan(c) |  # NaN check
            np.isinf(o) | np.isinf(h) | np.isinf(l) | np.isinf(c) |  # Inf check
            ~((l <= o) & (o <= h) & (l <= c) & (c <= h)) |            # Relationship check
             (o <= 0) | (h <= 0) | (l <= 0) | (c <= 0)                 # Non-positive check
        )
        error_count = np.sum(error_mask)
    
        if error_count > 0:
            error_indices = np.where(error_mask)[0]
            first_errors = error_indices[:min(5, len(error_indices))]
            logger.error(f"parse_candles_to_numpy: {error_count} invalid candle(s) detected")
            for idx in first_errors:
                logger.error(f"  Index {idx}: O={o[idx]:.2f} H={h[idx]:.2f} L={l[idx]:.2f} C={c[idx]:.2f}")
  
            if cfg.SANITIZE_BAD_CANDLES and error_count < n:
                keep_mask = ~error_mask
                logger.warning(
                    f"parse_candles_to_numpy: SANITIZE_BAD_CANDLES=True — dropping {error_count} "
                    f"bad candle(s), keeping {n - error_count}/{n}"
                )
                for k in data:
                    data[k] = data[k][keep_mask]
                o, h, l, c = data["open"], data["high"], data["low"], data["close"]
                n = len(data["timestamp"])
            else:
                logger.error("parse_candles_to_numpy: Rejecting data due to invalid candles")
                return None

        v = data["volume"]
        volume_error_mask = ~np.isfinite(v) | (v < 0)
        volume_error_count = np.sum(volume_error_mask)

        if volume_error_count > 0:
            logger.error(
                f"parse_candles_to_numpy: Found {volume_error_count} invalid volume value(s) out of {n} "
                f"({volume_error_count / n * 100:.1f}%)"
            )
            vol_error_indices = np.where(volume_error_mask)[0]
            for idx in vol_error_indices[:min(5, len(vol_error_indices))]:
                logger.error(f"  Index {idx}: Volume={v[idx]}")
            logger.error("parse_candles_to_numpy: Rejecting data due to invalid volume")
            return None

        hl_mid = (h + l) / 2.0
        candle_range = h - l
        close_deviation = np.abs(c - hl_mid) / (hl_mid + 1e-9)
        deviation_mask = close_deviation > Constants.HIGH_DEVIATION_THRESHOLD
        deviation_count = np.sum(deviation_mask)
 
        if deviation_count > 0:
            dev_indices = np.where(deviation_mask)[0].tolist()
            logger.warning(
                f"parse_candles_to_numpy: {deviation_count} candle(s) with "
                f"close/price deviation > {Constants.HIGH_DEVIATION_THRESHOLD} "
                f"| Indices: {dev_indices[:5]}"
            )
            if cfg.DEBUG_MODE and deviation_count <= 5:
                for idx in dev_indices:
                    dev_pct = close_deviation[idx] * 100
                    logger.debug(
                        f" Index {idx}: Deviation {dev_pct:.2f}% | "
                        f"Mid={hl_mid[idx]:.2f} Close={c[idx]:.2f}"
                    )
            if cfg.REJECT_HIGH_DEVIATION:
                logger.warning("Rejecting candle data due to high deviation (REJECT_HIGH_DEVIATION=True)")
                return None

        if n > 1:
            ts_diffs = np.diff(data["timestamp"])
            min_diff = np.min(ts_diffs)
            max_diff = np.max(ts_diffs)
        
            if min_diff <= 0:
                bad_idx = np.where(ts_diffs <= 0)[0]
                logger.warning(
                    f"parse_candles_to_numpy: {len(bad_idx)} non-monotonic/duplicate timestamp(s) found "
                    f"(indices {bad_idx[:5].tolist()}) | Min diff: {min_diff}s, Max diff: {max_diff}s | "
                    f"Continuing — get_last_closed_index_from_array will reject if this is near the target candle."
                )
                return None
        
            if cfg.DEBUG_MODE:
                logger.debug(
                    f"parse_candles_to_numpy: Timestamp range | "
                    f"First: {format_ist_time(data['timestamp'][0])} | "
                    f"Last: {format_ist_time(data['timestamp'][-1])} | "
                    f"Count: {n} candles"
                )
    
        if cfg.DEBUG_MODE:
            logger.debug(
                f"parse_candles_to_numpy: SUCCESSFUL | "
                f"Candles: {n} | "
                f"Range: {format_ist_time(data['timestamp'][0])} to {format_ist_time(data['timestamp'][-1])}"
            )
    
        return PriceData.from_dict(data)
    except Exception as e:
        logger.error(
            f"parse_candles_to_numpy: Unexpected exception: {e}",
            exc_info=True
        )
        return None

def candle_is_stable(ts_open: int, reference_time: int, interval_minutes: int = 15) -> bool:
    """Check if a candle is fully closed and past the safety buffer."""
    interval_seconds = interval_minutes * 60
    time_since_closed = reference_time - (ts_open + interval_seconds)
    age_from_open = reference_time - ts_open
    return (
        time_since_closed >= cfg.CANDLE_MIN_AGE_BUFFER
        and age_from_open >= Constants.MIN_CANDLE_AGE_FROM_OPEN
    )

def get_last_closed_index_from_array(timestamps: np.ndarray, interval_minutes: int, 
                                     reference_time: Optional[int] = None, 
                                     pair_name: Optional[str] = None) -> Optional[int]:
    if timestamps is None or timestamps.size < 1:
        return None

    if reference_time is None:
        reference_time = get_trigger_timestamp()
    reference_time = normalize_timestamp(reference_time)

    interval_seconds = interval_minutes * 60
    
    current_period_start = (reference_time // interval_seconds) * interval_seconds
    expected_ts_open_time = current_period_start - interval_seconds

    candle_close_time = expected_ts_open_time + interval_seconds
    time_since_candle_closed = reference_time - candle_close_time

    try:
        ts_normalized = normalize_timestamp_array(timestamps)
    except Exception as e:
        logger.error("[%s] Timestamp normalization failed: %s", pair_name or "?", e)
        return None

    if ts_normalized.size >= 2 and np.any(np.diff(ts_normalized) <= 0):
        target_area_mask = np.abs(ts_normalized - expected_ts_open_time) <= interval_seconds
        if np.any(np.diff(ts_normalized[target_area_mask]) <= 0):
            logger.warning("[%s] Timestamps corrupted near target; rejecting.", pair_name or "?")
            return None
        else:
            logger.info("[%s] Duplicates exist but not near target.", pair_name or "?")

    matches = np.flatnonzero(np.abs(ts_normalized - expected_ts_open_time) <= 1)
    if matches.size == 0:
        if logger.isEnabledFor(logging.DEBUG):
            last_ts = format_ist_time(ts_normalized[-1]) if ts_normalized.size else 'N/A'
            count = int(ts_normalized.size)
            last5_list = [format_ist_time(t) for t in ts_normalized[-5:]]
            last5_str = str(last5_list)

            logger.debug(
                "[%s] Target %dm open %s not found. last_ts=%s count=%s last5=%s",
                pair_name or "?", int(interval_minutes), format_ist_time(expected_ts_open_time),
                last_ts, count, last5_str
            )
        return None

    last_closed_idx = int(matches[-1])
    actual_candle_open = int(ts_normalized[last_closed_idx])

    if not candle_is_stable(actual_candle_open, reference_time, interval_minutes):
        logger.warning(
            "[%s] Candle %dm actual open %s not stable. Skipping.",
            pair_name or "?",
            int(interval_minutes),
            format_ist_time(actual_candle_open),
        )
        return None

    actual_close = actual_candle_open + interval_seconds
    if reference_time < actual_close:
        logger.error(
            "[%s] LOGIC ERROR: Candle not closed! Closes %s, ref %s",
            pair_name or "?",
            format_ist_time(actual_close),
            format_ist_time(reference_time)
        )
        return None

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "[%s] Selected CLOSED %dm candle idx=%d %s-%s (closed %ds ago)",
            pair_name or "?", int(interval_minutes), last_closed_idx,
            format_ist_time(actual_candle_open), format_ist_time(actual_close),
            int(time_since_candle_closed)
        )
    return last_closed_idx

@dataclass(frozen=True)
class CandleSnapshot:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_green: bool
    is_red: bool
    is_valid_for_buy: bool
    is_valid_for_sell: bool

@dataclass(slots=True)
class PriceData:
    """Typed replacement for the loose {"timestamp": arr, "open": arr, ...} dict
    returned by parse_candles_to_numpy()."""
    ts: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray

    @classmethod
    def from_dict(cls, d: Dict[str, np.ndarray]) -> "PriceData":
        return cls(
            ts=d["timestamp"], open=d["open"], high=d["high"],
            low=d["low"], close=d["close"], volume=d["volume"],
        )

    def as_dict(self) -> Dict[str, np.ndarray]:
        """Back-compat shim for call sites not yet migrated off dict-style access."""
        return {"timestamp": self.ts, "open": self.open, "high": self.high,
                "low": self.low, "close": self.close, "volume": self.volume}

    def __len__(self) -> int:
        return len(self.ts)

@dataclass(slots=True)
class IndicatorCache:
    """Typed replacement for the merged gate_indicators/alert_indicators dict."""
    # -- gate indicators (Phase 1, cheap) --
    rma50_15: np.ndarray
    rma200_5: np.ndarray
    ichimoku_cloud_upper: np.ndarray
    ichimoku_cloud_lower: np.ndarray
    ichimoku_future_green: np.ndarray
    ichimoku_future_red: np.ndarray
    ichimoku_conversion_line: np.ndarray
    ichimoku_base_line: np.ndarray
    fast_ichimoku_cloud_upper: np.ndarray
    fast_ichimoku_cloud_lower: np.ndarray
    fast_ichimoku_future_green: np.ndarray
    fast_ichimoku_future_red: np.ndarray
    fast_ichimoku_conversion_line: np.ndarray
    fast_ichimoku_base_line: np.ndarray
    adx: np.ndarray
    atr_short: np.ndarray
    atr_long: np.ndarray
    volume_ema: np.ndarray
    ppo_gate: np.ndarray
    ppo_gate_signal: np.ndarray
    rsi_guard_smooth: np.ndarray
    rsi_guard_ema: np.ndarray
    rma_cloud_fast_15: np.ndarray
    cpr_ok: bool = True
    nr_cpr: float = float("nan")
    prev_day_close: float = float("nan")
    # -- alert indicators (Phase 2, expensive) --
    ppo: Optional[np.ndarray] = None
    ppo_signal: Optional[np.ndarray] = None
    smooth_rsi: Optional[np.ndarray] = None
    smooth_rsi_ema: Optional[np.ndarray] = None
    vwap: Optional[np.ndarray] = None
    hist_rma: Optional[np.ndarray] = None
    pivots: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dicts(cls, gate: Dict[str, Any], alert: Optional[Dict[str, Any]] = None) -> "IndicatorCache":
        merged = {**gate, **(alert or {})}
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in merged.items() if k in known})

    def as_dict(self) -> Dict[str, Any]:
        """Back-compat shim — mirrors the old `{**gate_indicators, **alert_indicators}` merge."""
        return {f: getattr(self, f) for f in self.__dataclass_fields__}

@dataclass(slots=True)
class GateResult:
    # -- identity / indices --
    pair_name: str
    i15: int
    i5: int
    ts_curr: int
    reference_time: int

    # -- candle info --
    candle_info: Dict[str, Any]
    o: float; h: float; l: float; c: float
    open_curr: float; high_curr: float; low_curr: float; close_curr: float
    close_prev: float
    close_5m_val: float
    is_green: bool; is_red: bool
    is_valid_for_buy: bool; is_valid_for_sell: bool
    candle_index: int
    min_wick_ratio: float
    buy_wick_ratio: float; sell_wick_ratio: float

    # -- gate/alert indicator dicts (still raw dicts — untouched by this pass) --
    gate_indicators: Dict[str, Any]
    
    # -- trend --
    base_buy_trend: bool; base_sell_trend: bool
    rma50_15_val: float; rma200_5_val: float

    # -- ichimoku cloud --
    cloud_up: Optional[bool]; cloud_down: Optional[bool]
    cloud_upper_val: float; cloud_lower_val: float
    cloud_upper_prev: float; cloud_lower_prev: float
    ichimoku_gate_ok_buy: Optional[bool]; ichimoku_gate_ok_sell: Optional[bool]
    confirmation_buy: bool; confirmation_sell: bool
    cloud_group_ok_buy: bool; cloud_group_ok_sell: bool

    # -- TK guard --
    tk_conversion_curr: float; tk_conversion_prev: float
    tk_base_curr: float; tk_base_prev: float
    tk_guard_ok_buy: Optional[bool]; tk_guard_ok_sell: Optional[bool]

    # -- fast ichimoku (alert-only) --
    fast_future_green: bool; fast_future_red: bool
    fast_cloud_upper_curr: float; fast_cloud_lower_curr: float
    fast_cloud_upper_prev: float; fast_cloud_lower_prev: float
    fast_tk_conversion_curr: float; fast_tk_conversion_prev: float
    fast_tk_base_curr: float; fast_tk_base_prev: float
    fast_tenkan_ge_kijun: bool; fast_tenkan_le_kijun: bool

    # -- oscillator group votes --
    oscillator_group_ok_buy: bool; oscillator_group_ok_sell: bool
    ppo_gate_arr: np.ndarray; ppo_gate_signal_arr: np.ndarray
    ppo_gate_curr: float; ppo_gate_prev: float
    ppo_gate_sig_curr: float; ppo_gate_sig_prev: float
    ppo_gate_ok_buy: bool; ppo_gate_ok_sell: bool
    rsi_guard_smooth_curr: float; rsi_guard_ema_curr: float
    rsi_guard_ok_buy: bool; rsi_guard_ok_sell: bool
    rma_cloud_fast_curr: float
    rma_cloud_ok_buy: bool; rma_cloud_ok_sell: bool

    # -- trend gate combination --
    trend_gate_ok_buy: bool; trend_gate_ok_sell: bool

    # -- volatility / ADX / RVOL --
    adx_val: float; adx_adaptive_threshold: float; adx_ok: bool
    rvol_bypass_ok: bool; rvol_ok: bool
    adaptive_rvol_check: bool
    momentum_count: int
    volatility_filter_ok: bool

    # -- CPR --
    cpr_ok: bool; nr_cpr: float
    effective_cpr_ok: bool
    cpr_adaptive_min_pct_move: float
    move_from_prev_close_ok: bool

    # -- adaptive thresholds carried into Phase 2 --
    ppo_adaptive_threshold: float
    rsi_adaptive_buy: float; rsi_adaptive_sell: float

    # -- final gate decision --
    buy_common: bool
    sell_common: bool

    # -- misc data passed through --
    data_15m: PriceData
    close_prev_invalid: bool = False

def compute_confluence_score(gr: "GateResult", is_buy: bool) -> Tuple[int, int]:
    """Count independent gate votes that agree for the given direction.
    Read-only over already-computed GateResult fields — does not alter any
    existing trigger/gate logic. Returns (score, total_available_votes)."""
    votes: List[bool] = []

    if is_buy:
        votes.append(gr.base_buy_trend)
        if cfg.ICHIMOKU_CLOUD_ENABLED and gr.ichimoku_gate_ok_buy is not None:
            votes.append(gr.ichimoku_gate_ok_buy)
        if cfg.RMA_CLOUD_ENABLED and gr.rma_cloud_ok_buy is not None:
            votes.append(gr.rma_cloud_ok_buy)
        if cfg.ENABLE_PPO_GATE:
            votes.append(gr.ppo_gate_ok_buy)
        if cfg.RSI_GUARD_ENABLED:
            votes.append(gr.rsi_guard_ok_buy)
        if cfg.ICHIMOKU_TK_GUARD_ENABLED and gr.tk_guard_ok_buy is not None:
            votes.append(gr.tk_guard_ok_buy)
        if cfg.ENABLE_ADX_FILTER:
            votes.append(gr.adx_ok)
        if cfg.ENABLE_RVOL_ALERT or cfg.ATR_ADAPTIVE_ENABLED:
            votes.append(gr.rvol_ok)
        if cfg.ENABLE_CPR:
            votes.append(gr.effective_cpr_ok)
    else:
        votes.append(gr.base_sell_trend)
        if cfg.ICHIMOKU_CLOUD_ENABLED and gr.ichimoku_gate_ok_sell is not None:
            votes.append(gr.ichimoku_gate_ok_sell)
        if cfg.RMA_CLOUD_ENABLED and gr.rma_cloud_ok_sell is not None:
            votes.append(gr.rma_cloud_ok_sell)
        if cfg.ENABLE_PPO_GATE:
            votes.append(gr.ppo_gate_ok_sell)
        if cfg.RSI_GUARD_ENABLED:
            votes.append(gr.rsi_guard_ok_sell)
        if cfg.ICHIMOKU_TK_GUARD_ENABLED and gr.tk_guard_ok_sell is not None:
            votes.append(gr.tk_guard_ok_sell)
        if cfg.ENABLE_ADX_FILTER:
            votes.append(gr.adx_ok)
        if cfg.ENABLE_RVOL_ALERT or cfg.ATR_ADAPTIVE_ENABLED:
            votes.append(gr.rvol_ok)
        if cfg.ENABLE_CPR:
            votes.append(gr.effective_cpr_ok)

    score = sum(1 for v in votes if v)
    return score, len(votes)

async def confirm_candle_unchanged(fetcher: DataFetcher, symbol: str, pair_name: str,
    ts_curr: int, cached: CandleSnapshot, reference_time: int, logger_pair: logging.Logger) -> Optional[bool]:
    """Returns True=unchanged, False=confirmed repaint/mismatch, None=inconclusive (fetch/network failure)."""
    try:
        raw = await fetcher.fetch_candles(symbol, "15", 5, reference_time, for_confirmation=True) 
        fresh = parse_candles_to_numpy(raw)
        if fresh is None:
            logger_pair.warning(f"[{pair_name}] Confirmation fetch failed — inconclusive, releasing dedup claim")
            return None
        matches = np.flatnonzero(np.abs(fresh.ts - ts_curr) <= 5)
        if matches.size == 0:
            logger_pair.warning(f"[{pair_name}] Confirmation candle {format_ist_time(ts_curr)} not found — inconclusive, releasing dedup claim")
            return None

        idx = int(matches[-1])
        fo = float(fresh.open[idx])
        fh = float(fresh.high[idx])
        fl = float(fresh.low[idx])
        fc = float(fresh.close[idx])
        fvol = float(fresh.volume[idx])

        # Volume check (matches validate_candle_for_alerts)
        if fvol <= 0:
            logger_pair.warning(f"[{pair_name}] Confirmation candle has zero volume — suppressing")
            return False

        # Color consistency check
        was_green = cached.is_green
        was_red = cached.is_red
        is_now_green = fc > fo
        is_now_red = fc < fo
        if (was_green and not is_now_green) or (was_red and not is_now_red):
            logger_pair.warning(
                f"[{pair_name}] Confirmation candle COLOR changed: "
                f"was {'green' if was_green else 'red'}, now "
                f"{'green' if is_now_green else 'red' if is_now_red else 'doji'}"
            )
            return False

        def _price_match(a: float, b: float) -> bool:
            abs_diff = abs(a - b)
            if abs_diff <= 1e-6:
                return True
            rel_diff = abs_diff / max(abs(a), abs(b), 1e-12)
            return rel_diff <= 1e-6

        if (not _price_match(fo, cached.open) or not _price_match(fh, cached.high) or
            not _price_match(fl, cached.low) or not _price_match(fc, cached.close)):

            logger_pair.warning(
                f"[{pair_name}] 🔁 Candle CHANGED since first fetch — repaint detected, suppressing alert | "
                f"First: O={cached.open:.4f} H={cached.high:.4f} L={cached.low:.4f} C={cached.close:.4f} | "
                f"Now:   O={fo:.4f} H={fh:.4f} L={fl:.4f} C={fc:.4f}"
            )
            return False

        return True
    except Exception as e:
        logger_pair.warning(f"[{pair_name}] Confirmation check errored: {e} — inconclusive, releasing dedup claim")
        return None

def independent_candle_reverify(data_15m: Dict[str, np.ndarray], candle_index: int, cached: CandleSnapshot, min_wick_ratio: float, pair_name: str, logger_pair: logging.Logger) -> bool:
    try:
        raw_o = float(data_15m["open"][candle_index])
        raw_h = float(data_15m["high"][candle_index])
        raw_l = float(data_15m["low"][candle_index])
        raw_c = float(data_15m["close"][candle_index])
        raw_ts = int(data_15m["timestamp"][candle_index])
        raw_vol = float(data_15m["volume"][candle_index])
    except (IndexError, KeyError, TypeError, ValueError) as e:
        logger_pair.error(
            f"[{pair_name}] Independent re-verify: cannot read raw OHLCV at index {candle_index}: {e} — suppressing alert"
        )
        return False

    if any(np.isnan([raw_o, raw_h, raw_l, raw_c])) or any(np.isinf([raw_o, raw_h, raw_l, raw_c])):
        logger_pair.error(
            f"[{pair_name}] Independent re-verify: raw OHLC contains NaN/Inf at index {candle_index} — suppressing alert"
        )
        return False

    if raw_ts != cached.timestamp:
        logger_pair.error(
            f"[{pair_name}] Independent re-verify TIMESTAMP MISMATCH: raw={raw_ts} cached={cached.timestamp} "
            f"— suppressing alert"
        )
        return False

    if raw_vol <= 0:
        logger_pair.error(
            f"[{pair_name}] Independent re-verify: zero/negative volume ({raw_vol}) at dispatch — suppressing alert"
        )
        return False

    def _close_enough(a: float, b: float) -> bool:
        abs_diff = abs(a - b)
        rel_tolerance = 1e-6 * max(abs(a), abs(b), 1.0)
        abs_floor = 1e-8  # noise floor for sub-cent priced coins, e.g. $0.00001
        return abs_diff <= max(rel_tolerance, abs_floor)

    mismatches = [
        tag for tag, a, b in (
            ("open", raw_o, cached.open), ("high", raw_h, cached.high),
            ("low", raw_l, cached.low), ("close", raw_c, cached.close),
        ) if not _close_enough(a, b)
    ]
    if mismatches:
        logger_pair.error(
            f"[{pair_name}] Independent re-verify OHLC MISMATCH on {mismatches} at index {candle_index} "
            f"| raw O={raw_o:.6f} H={raw_h:.6f} L={raw_l:.6f} C={raw_c:.6f} "
            f"| cached O={cached.open:.6f} H={cached.high:.6f} L={cached.low:.6f} C={cached.close:.6f} "
            f"— suppressing alert"
        )
        return False

    raw_range = raw_h - raw_l
    if raw_range < 1e-9:
        logger_pair.error(f"[{pair_name}] Independent re-verify: zero-range candle at dispatch — suppressing alert")
        return False

    raw_is_green = raw_c > raw_o
    raw_is_red = raw_c < raw_o

    if raw_is_green != cached.is_green or raw_is_red != cached.is_red:
        logger_pair.error(
            f"[{pair_name}] Independent re-verify COLOR MISMATCH: "
            f"raw(green={raw_is_green}, red={raw_is_red}) vs cached(green={cached.is_green}, red={cached.is_red}) "
            f"| O={raw_o:.4f} C={raw_c:.4f} — suppressing alert"
        )
        return False

    hi_body = max(raw_o, raw_c)
    lo_body = min(raw_o, raw_c)
    raw_upper_wick = raw_h - hi_body
    raw_lower_wick = lo_body - raw_l
    raw_body = hi_body - lo_body

    raw_body_ratio = raw_body / raw_range
    raw_upper_ratio = raw_upper_wick / raw_range
    raw_lower_ratio = raw_lower_wick / raw_range

    raw_valid_buy = raw_is_green and raw_upper_ratio < min_wick_ratio and raw_body_ratio >= Constants.MIN_BODY_RATIO
    raw_valid_sell = raw_is_red and raw_lower_ratio < min_wick_ratio and raw_body_ratio >= Constants.MIN_BODY_RATIO

    if raw_valid_buy != cached.is_valid_for_buy or raw_valid_sell != cached.is_valid_for_sell:
        logger_pair.error(
            f"[{pair_name}] Independent re-verify VALIDITY MISMATCH: "
            f"raw(buy={raw_valid_buy}, sell={raw_valid_sell}) vs cached(buy={cached.is_valid_for_buy}, sell={cached.is_valid_for_sell}) "
            f"| upper_ratio={raw_upper_ratio:.4f} lower_ratio={raw_lower_ratio:.4f} body_ratio={raw_body_ratio:.4f} "
            f"— suppressing alert"
        )
        return False
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

class RedisKeyPrefix:
    """Centralized Redis key prefixes"""
    PAIR_STATE = "pair_state:"
    METADATA = "metadata:"
    ALERT = "alert:"
    RECENT_ALERT = "recent_alert:"
    LOCK = "lock:"

class RedisStateStore:
    POOL_MAX_AGE_SECONDS = 3600
    SCRIPT_RELOAD_LOCK_TIMEOUT = 2.0

    _global_pools: ClassVar[Dict[str, Optional[redis.Redis]]] = {}
    _pool_healthy: ClassVar[Dict[str, bool]] = {}
    _pool_created_at: ClassVar[Dict[str, float]] = {}
    _pool_reuse_count: ClassVar[Dict[str, int]] = {}
    _pool_lock: ClassVar[Optional[asyncio.Lock]] = None
    _script_reload_lock: ClassVar[Optional[asyncio.Lock]] = None

    @classmethod
    def _get_pool_lock(cls) -> asyncio.Lock:
        if cls._pool_lock is None:
            cls._pool_lock = asyncio.Lock()
        return cls._pool_lock

    @classmethod
    def _get_script_reload_lock(cls) -> asyncio.Lock:
        if cls._script_reload_lock is None:
            cls._script_reload_lock = asyncio.Lock()
        return cls._script_reload_lock

    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self._redis: Optional[redis.Redis] = None

        self.state_prefix = RedisKeyPrefix.PAIR_STATE
        self.meta_prefix = RedisKeyPrefix.METADATA
        self.alert_prefix = RedisKeyPrefix.ALERT

        self.expiry_seconds = max(cfg.STATE_EXPIRY_DAYS * 86400 if cfg.STATE_EXPIRY_DAYS > 0 else 0, 7 * 86400)
        self.alert_expiry_seconds = cfg.STATE_EXPIRY_DAYS * 86400
        self.metadata_expiry_seconds = 7 * 86400

        self.degraded = False
        self.degraded_alerted = False
        self._connection_attempts = 0

        if cfg.DEBUG_MODE and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"RedisStateStore initialized | "
                f"State TTL: {cfg.STATE_EXPIRY_DAYS}d | "
                f"Alert TTL: {cfg.STATE_EXPIRY_DAYS}d | "
                f"Metadata TTL: 7d"
            )

    async def _record_redis_failure(self, operation: str, exc: Exception) -> None:
        logger.error(f"Redis operation '{operation}' failed: {exc}")
        if self.degraded:
            return
        self.degraded = True
        logger.warning(f"Redis marked degraded after failure in '{operation}' — attempting one reconnect")
        try:
            reconnected = await self._attempt_connect(timeout=5.0)
            if reconnected:
                logger.info(f"Redis reconnected after failure in '{operation}'")
                self.degraded = False
            else:
                logger.critical(f"Redis reconnect failed after '{operation}' — staying degraded for remainder of run")
        except Exception as reconnect_exc:
            logger.critical(f"Redis reconnect attempt itself failed: {reconnect_exc} — staying degraded")

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
            if not ok:
                raise RedisConnectionError("ping failed after retries")

            logger.info("Redis connected")
            self.degraded = False
            self.degraded_alerted = False
            self._connection_attempts = 0

            async with RedisStateStore._get_pool_lock():
                existing_pool = RedisStateStore._global_pools.get(self.redis_url)
                pool_is_healthy = False
                if existing_pool:
                    try:
                        pool_is_healthy = await asyncio.wait_for(existing_pool.ping(), timeout=1.0)
                    except Exception:
                        pool_is_healthy = False

                if existing_pool and pool_is_healthy:
                    if self._redis is not existing_pool:
                        await self._redis.aclose()
                    self._redis = existing_pool
                    logger.debug("Using pool created by another coroutine")
                else:
                    if existing_pool and existing_pool is not self._redis:
                        try:
                            await existing_pool.aclose()
                        except Exception:
                            pass
                    RedisStateStore._global_pools[self.redis_url] = self._redis
                    RedisStateStore._pool_healthy[self.redis_url] = True
                    RedisStateStore._pool_created_at[self.redis_url] = time.time()
                    RedisStateStore._pool_reuse_count[self.redis_url] = 0
                    if cfg.DEBUG_MODE:
                        logger.debug("Redis connection saved to per-URL pool")

                return True

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
        pool_reused = False

        async with RedisStateStore._get_pool_lock():
            pool = RedisStateStore._global_pools.get(self.redis_url)
            healthy = RedisStateStore._pool_healthy.get(self.redis_url, False)

            if pool and healthy:
                pool_age = time.time() - RedisStateStore._pool_created_at.get(self.redis_url, 0.0)
                if pool_age > self.POOL_MAX_AGE_SECONDS:
                    logger.info(f"Redis pool aged {pool_age:.0f}s, refreshing")
                    RedisStateStore._pool_healthy[self.redis_url] = False
                    try:
                        await pool.aclose()
                    except Exception:
                        pass
                    RedisStateStore._global_pools[self.redis_url] = None
                else:
                    try:
                        ok = await self._ping_with_retry(timeout)
                        if ok:
                            self._redis = pool
                            RedisStateStore._pool_reuse_count[self.redis_url] = \
                                RedisStateStore._pool_reuse_count.get(self.redis_url, 0) + 1
                            self.degraded = False
                            pool_reused = True
                            return
                    except Exception as e:
                        if cfg.DEBUG_MODE:
                            logger.debug(f"Pool health check failed: {e}, creating new pool")
                        RedisStateStore._pool_healthy[self.redis_url] = False
                        pool_reused = False

        if pool_reused:
            return

        for attempt in range(1, cfg.REDIS_CONNECTION_RETRIES + 1):
            if await self._attempt_connect(timeout):
                max_conn = getattr(self._redis.connection_pool, "max_connections", "?")
                logger.info(f"✅ Redis connected ({max_conn} max)")
                self.degraded = False
                self.degraded_alerted = False
                return

            if attempt < cfg.REDIS_CONNECTION_RETRIES:
                delay = compute_backoff(cfg.REDIS_RETRY_DELAY, attempt)
                logger.warning(f"Retrying Redis connection in {delay:.1f}s...")
                await asyncio.sleep(delay)

        logger.critical("❌ Redis connection failed after all retries")
        self.degraded = True
        if self._redis:
            try:
                await self._redis.aclose()
            except Exception:
                pass
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
        self._redis = None

    @classmethod
    async def shutdown_global_pool(cls, redis_url: Optional[str] = None) -> None:
        async with cls._get_pool_lock():
            urls = [redis_url] if redis_url else list(cls._global_pools.keys())
            for url in urls:
                pool = cls._global_pools.get(url)
                if pool:
                    try:
                        pool_age = time.time() - cls._pool_created_at.get(url, 0.0)
                        reuse_count = cls._pool_reuse_count.get(url, 0)
                        logger.debug(f"Shutting down Redis pool | url={url} | Age: {pool_age:.1f}s | Reuses: {reuse_count}")

                        await pool.aclose()
                        await asyncio.sleep(0.25)

                    except Exception as e:
                        logger.error(f"Error shutting down Redis pool {url}: {e}")

                cls._global_pools.pop(url, None)
                cls._pool_healthy.pop(url, None)
                cls._pool_created_at.pop(url, None)
                cls._pool_reuse_count.pop(url, None)
            
    async def _ping_with_retry(self, timeout: float) -> bool:
        result = await self._safe_redis_op(lambda: self._redis.ping(), timeout, "ping")
        return bool(result)

    async def _safe_redis_op(self, fn: Callable[[], Any], timeout: float, op_name: str, parser: Optional[Callable[[Any], Any]] = None):
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

    async def set(self, key: str, state: Optional[Any], ts: Optional[int] = None, timeout: float = 2.0) -> None:
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

    async def set_metadata(self, key: str, value: str, timeout: float = 2.0) -> None:
        await self._safe_redis_op(
            lambda: self._redis.set(
                f"{self.meta_prefix}{key}",
                value,
                ex=self.metadata_expiry_seconds
            ),
            timeout,
            f"set_metadata {key}",
        )

    async def check_recent_alert(self, pair: str, alert_key: str, ts: int, window_sec: Optional[int] = None) -> bool:
        if self.degraded:
            return True
        recent_key = f"{RedisKeyPrefix.RECENT_ALERT}{pair}:{alert_key}"
        effective_window = window_sec if window_sec is not None else cfg.ALERT_DEDUP_WINDOW_SEC
        try:
            result = await asyncio.wait_for(
                self._redis.set(recent_key, str(ts), nx=True, ex=effective_window),
                timeout=3.0
            )
            should_send = bool(result)
            if cfg.DEBUG_MODE and not should_send:
                logger.debug(f"Dedup: Skipping duplicate {pair}:{alert_key}")
            return should_send
        except Exception as e:
            logger.error(f"Dedup check FAILED for {pair}:{alert_key}: {e}")
            return False   # fail-closed, not fail-open

    async def release_recent_alert(self, pair: str, alert_key: str) -> None:
        """Undo a dedup claim if the message didn't actually get delivered."""
        if self.degraded:
            return
        recent_key = f"{RedisKeyPrefix.RECENT_ALERT}{pair}:{alert_key}"
        try:
            await asyncio.wait_for(self._redis.delete(recent_key), timeout=1.0)
        except Exception as e:
            logger.warning(f"Failed to release dedup claim for {pair}:{alert_key}: {e}")

    async def batch_get_all_alert_states(self, pair: str, alert_keys: List[str], timeout: float = 3.0) -> Dict[str, bool]:
        if not self._redis or self.degraded or not alert_keys:
            return {k: False for k in alert_keys}

        try:
            hash_key = f"{self.state_prefix}{pair}"
            hash_data = await asyncio.wait_for(
                self._redis.hgetall(hash_key),
                timeout=timeout,
            )

            states: Dict[str, bool] = {}
            for key in alert_keys:
                val = hash_data.get(key)

                if val is None:
                    states[key] = False
                    continue

                try:
                    parsed_state = json_loads(val)
                    states[key] = parsed_state.get("state") == "ACTIVE"
                except (JSONDecodeError, TypeError) as e:
                    if cfg.DEBUG_MODE:
                        logger.debug(f"Failed to parse state for {pair}:{key}: {e}")
                    states[key] = False
                except Exception as e:
                    logger.error(f"Unexpected error parsing state for {pair}:{key}: {e}")
                    states[key] = False

            return states
        except asyncio.TimeoutError as e:
            await self._record_redis_failure(f"batch_get_all_alert_states({pair})", e)
            return {k: False for k in alert_keys}
        except Exception as e:
            await self._record_redis_failure(f"batch_get_all_alert_states({pair})", e)
            return {k: False for k in alert_keys}

    async def atomic_batch_update(self, updates: List[Tuple[str, Any, Optional[int]]], deletes: Optional[List[str]] = None, timeout: float = 4.0) -> bool:
        if self.degraded or not self._redis:
            return False

        if not updates and not deletes:
            return True

        try:
            async with self._redis.pipeline() as pipe:
                now = int(time.time())
                touched_hashes: Set[str] = set()

                hash_writes: Dict[str, Dict[str, str]] = {}
                for key, state, custom_ts in (updates or []):
                    pair, sep, field = key.partition(":")
                    if not sep:
                        logger.error(
                            f"Skipping malformed state key (expected 'pair:field'): {key}"
                        )
                        continue
                    ts = custom_ts if custom_ts is not None else now
                    try:
                        data = json_dumps({"state": state, "ts": ts})
                    except Exception as e:
                        logger.error(f"Failed to serialize state for {key}: {e}")
                        continue
                    hash_key = f"{self.state_prefix}{pair}"
                    hash_writes.setdefault(hash_key, {})[field] = data

                for hash_key, mapping in hash_writes.items():
                    pipe.hset(hash_key, mapping=mapping)
                    touched_hashes.add(hash_key)

                hash_deletes: Dict[str, List[str]] = {}
                for key in (deletes or []):
                    if not key:
                        continue
                    raw_key = (
                        key[len(self.state_prefix) :]
                        if key.startswith(self.state_prefix)
                        else key
                    )
                    pair, sep, field = raw_key.partition(":")
                    if not sep:
                        logger.error(
                            f"Skipping malformed delete key (expected 'pair:field'): {key}"
                        )
                        continue
                    hash_key = f"{self.state_prefix}{pair}"
                    hash_deletes.setdefault(hash_key, []).append(field)

                for hash_key, fields in hash_deletes.items():
                    pipe.hdel(hash_key, *fields)
                    touched_hashes.add(hash_key)

                if self.expiry_seconds > 0:
                    for hash_key in touched_hashes:
                        pipe.expire(hash_key, self.expiry_seconds)

                await asyncio.wait_for(pipe.execute(), timeout=timeout)
            return True
        except asyncio.TimeoutError as e:
            await self._record_redis_failure("atomic_batch_update", e)
            return False
        except Exception as e:
            await self._record_redis_failure("atomic_batch_update", e)
            return False

class RedisLock:    
    RELEASE_LUA = """
    if redis.call("GET", KEYS[1]) == ARGV[1] then
        return redis.call("DEL", KEYS[1])
    else
        return 0
    end
    """
    EXTEND_LUA = """
    if redis.call("GET", KEYS[1]) == ARGV[1] then
        return redis.call("EXPIRE", KEYS[1], ARGV[2])
    else
        return 0
    end
    """
    def __init__(self, redis_client: Optional[redis.Redis], lock_key: str, expire: int | None = None):
        self.redis = redis_client
        self.lock_key = f"{RedisKeyPrefix.LOCK}{lock_key}"
        self.expire = expire or cfg.REDIS_LOCK_EXPIRY
        self.token: Optional[str] = None
        self.lost = False
        self.acquired_by_me = False
        self.last_extend_time = time.monotonic() 

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
                self.lost = False
                self.last_extend_time = time.monotonic()
                
                logger.info(
                    f"🔐 Lock acquired: {self.lock_key.replace('lock:', '')} ({self.expire}s)"
                )
                return True

            logger.warning(f"Could not acquire Redis lock (held): {self.lock_key}")
            return False
            
        except asyncio.TimeoutError:
            logger.error(f"Timeout acquiring lock {self.lock_key} after {timeout}s")
            return False
        except Exception as e:
            logger.error(f"Redis lock acquisition failed: {e}")
            return False

    async def extend(self, timeout: float = 3.0) -> bool:     
        if not self.token or not self.redis or not self.acquired_by_me:
            self.lost = True
            return False    
        try:
            result = await asyncio.wait_for(
                self.redis.eval(
                    self.EXTEND_LUA,
                    1,
                    self.lock_key,
                    self.token,
                    self.expire,
                ),
                timeout=timeout,
            )

            if result:
                self.last_extend_time = time.monotonic()
                if cfg.DEBUG_MODE:
                    logger.debug(f"Extended Redis lock: {self.lock_key} (now {self.expire}s)")
                return True
            else:
                logger.warning("Lock lost during extend (token mismatch or key missing)")
                self.lost = True
                self.acquired_by_me = False
                return False
                
        except asyncio.TimeoutError:
            logger.error(f"Timeout extending lock {self.lock_key} after {timeout}s")
            self.lost = True
            self.acquired_by_me = False
            return False
        except Exception as e:
            logger.error(f"Error extending Redis lock: {e}")
            self.lost = True
            self.acquired_by_me = False
            return False

    @classmethod
    def get_lock_extend_interval(cls) -> int:    
        extend_at = int(cfg.REDIS_LOCK_EXPIRY * 0.7)
        return max(60, min(extend_at, 540)) 

    def should_extend(self) -> bool:     
        if not self.acquired_by_me or self.lost:
            return False

        extend_threshold = self.__class__.get_lock_extend_interval()       
        elapsed = time.monotonic() - self.last_extend_time 
        should_extend = elapsed >= extend_threshold
        
        if cfg.DEBUG_MODE and should_extend:
            logger.debug(
                f"Lock extension eligible | "
                f"Elapsed: {elapsed:.0f}s | "
                f"Threshold: {extend_threshold}s"
            )
        
        return should_extend

    async def release(self, timeout: float = 3.0) -> None:     
        if not self.token or not self.redis or not self.acquired_by_me:
            return
        try:
            result = await asyncio.wait_for(
                self.redis.eval(self.RELEASE_LUA, 1, self.lock_key, self.token),
                timeout=timeout,
            )
        
            if result:
                logger.info(f"🔏 Lock released: {self.lock_key.replace('lock:', '')}")
                self.acquired_by_me = False
                self.token = None
            else:
                logger.warning(
                    f"Lock release failed (token mismatch): {self.lock_key} | "
                    f"Lock was stolen or lost"
                )
                self.lost = True
                self.acquired_by_me = False
    
        except asyncio.TimeoutError:
            logger.error(f"Timeout releasing lock {self.lock_key} after {timeout}s")
            self.lost = True
            self.acquired_by_me = False
        except Exception as e:
            logger.error(f"Error releasing Redis lock: {e}")
            self.lost = True
            self.acquired_by_me = False
    
        finally:
            self.token = None

    def __repr__(self) -> str:
        status = "HELD" if self.acquired_by_me else ("LOST" if self.lost else "RELEASED")
        token_display = self.token[:8] + "..." if self.token else "None"
        return f"RedisLock({self.lock_key}:{status}:{token_display})"

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
            return bool(
                await asyncio.wait_for(
                    self._send_impl(message),
                    timeout=45.0
                )
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            if cfg.FAIL_ON_TELEGRAM_DOWN:
                raise
            return False

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
                    delay = compute_backoff(1.0, attempt)
                    logger.debug(f"Retrying Telegram request in {delay:.1f}s (attempt {attempt})...")
                    await asyncio.sleep(delay)
        return False

    async def send_batch(self, messages: List[str]) -> bool:   
        if not messages:
            return True

        def _safe_truncate_utf8(text: str, max_bytes: int) -> str:
            encoded = text.encode('utf-8')
            if len(encoded) <= max_bytes:
                return text
            truncated = encoded[:max_bytes]
            # Strip continuation bytes (0x80-0xBF) from the tail
            while truncated and truncated[-1] & 0xC0 == 0x80:
                truncated = truncated[:-1]
            return truncated.decode('utf-8', errors='ignore')
   
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

def _clean_extra_text(extra: Optional[str]) -> str:
    """Helper to strip emojis, OHLC data, and technical metadata."""
    if not extra:
        return ""
    extra_clean = re.sub(r'[🟢🔴🔵🟣]', '', extra)  
    extra_clean = re.sub(r'\(O:[\d.]+ H:[\d.]+ L:[\d.]+ C:[\d.]+\)', '', extra_clean)  
    extra_clean = re.sub(r'\[i15=\d+,\s*[\d-]+\s+[\d:]+\s+IST\]', '', extra_clean)  
    return extra_clean.strip()

def _format_price(price: Any) -> str:
    """Safely format price to 2 decimal places."""
    return f"${price:,.2f}" if isinstance(price, (int, float)) else "N/A"

def build_single_msg(title: str, pair: str, price: Any, ts: int, extra: Optional[str] = None) -> str:
    """Build a beautifully formatted Telegram single alert using MarkdownV2."""
    if not title: 
        title = "ALERT"
    
    parts = title.split(" ", 1)
    symbols = parts[0]
    description = parts[1] if len(parts) == 2 else title
    
    # 1. Format the raw strings
    price_str = _format_price(price)
    extra_clean = _clean_extra_text(extra)
    date_str = format_ist_time(ts, '%d-%m-%Y')
    time_str = format_ist_time(ts, '%H:%M IST')
    
    # 2. ESCAPE INDIVIDUAL DATA (Crucial for MarkdownV2 stability)
    e_symbols = escape_markdown_v2(symbols)
    e_pair = escape_markdown_v2(pair)
    e_price = escape_markdown_v2(price_str)
    e_desc = escape_markdown_v2(description)
    e_extra = escape_markdown_v2(extra_clean)
    e_date = escape_markdown_v2(date_str)
    e_time = escape_markdown_v2(time_str)
    
    # 3. APPLY TELEGRAM LAYOUT TAGS
    # Bold pair and bold price. Note: Literal hyphens '\-' must be escaped in MarkdownV2.
    line1 = f"{e_symbols} *{e_pair}* \\- *{e_price}*"
    
    # Bold the alert type, italicize the extra context details
    if e_extra:
        line2 = f"*{e_desc}* : _{e_extra}_"
    else:
        line2 = f"*{e_desc}*"
    
    spacing = " " * 12
    line3 = f"📅 {e_date}{spacing}⏰ {e_time}"
    
    # Return the raw composite string (do NOT wrap this in escape_markdown_v2)
    return f"{line1}\n{line2}\n{line3}"
        
def build_batched_msg(pair: str, price: Any, ts: int, items: List[Tuple[str, str]]) -> str:
    """Build a beautifully formatted Telegram batched alert using MarkdownV2."""
    price_str = _format_price(price)
    date_str = format_ist_time(ts, '%d-%m-%Y')
    time_str = format_ist_time(ts, '%H:%M IST')
    
    e_pair = escape_markdown_v2(pair)
    e_price = escape_markdown_v2(price_str)
    e_date = escape_markdown_v2(date_str)
    e_time = escape_markdown_v2(time_str)
    spacing = " " * 12
    
    if not items:
        return f"*{e_pair}* \\- *{e_price}*\n🗓️ {e_date}{spacing}🕙 {e_time}"
    
    headline_emoji = items[0][0].split(" ", 1)[0] if items[0][0] else "📊"
    e_headline_emoji = escape_markdown_v2(headline_emoji)
    
    line1 = f"{e_headline_emoji} *{e_pair}* \\- *{e_price}*"
    
    alert_lines = []
    for idx, (title, extra) in enumerate(items):
        parts = title.split(" ", 1)
        description = parts[1] if len(parts) == 2 else title
        extra_clean = _clean_extra_text(extra)
        
        e_desc = escape_markdown_v2(description)
        e_extra = escape_markdown_v2(extra_clean)
        
        prefix = "└➤" if idx == len(items) - 1 else "├➤"
        
        if e_extra:
            alert_lines.append(f"{prefix} *{e_desc}* : _{e_extra}_")
        else:
            alert_lines.append(f"{prefix} *{e_desc}*")
    
    body = "\n".join(alert_lines)
    datetime_line = f"📆  {e_date}{spacing}⏰ {e_time}"
    
    return f"{line1}\n{body}\n{datetime_line}"

def create_pivot_alert(level: str, is_buy: bool) -> AlertDefinition:
    """Factory function to create pivot alert definition without lambdas"""
    if is_buy:
        return {
            "key": f"pivot_up_{level}",
            "title": f"🟢⬆️ Cross above {level}",
        "check_fn": lambda ctx, ppo, ppo_sig, rsi: (
            ctx.get("buy_common", False) and
            get_pivot_alert_info(ctx, level, is_buy=True)[0]
        ),
            "extra_fn": lambda ctx, ppo, ppo_sig, rsi, _: (
                f"${ctx['pivots'][level]:,.2f}"
                f"[Dist: {abs(ctx['pivots'][level] - ctx['close_curr'])/ctx['pivots'][level]*100:.2f}%]"
            ),
            "requires": ["pivots"]
        }
    else:
        return {
            "key": f"pivot_down_{level}",
            "title": f"🔴⬇️ Cross below {level}",
        "check_fn": lambda ctx, ppo, ppo_sig, rsi: (
            ctx.get("sell_common", False) and
            get_pivot_alert_info(ctx, level, is_buy=False)[0]
        ),
            "extra_fn": lambda ctx, ppo, ppo_sig, rsi, _: (
                f"${ctx['pivots'][level]:,.2f}"
                f"[Dist: {abs(ctx['pivots'][level] - ctx['close_curr'])/ctx['pivots'][level]*100:.2f}%]"
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

    {"key":"ppo_signal_up","title":"🟢 PPO cross▲signal","check_fn":lambda ctx,ppo,ppo_sig,rsi:(ctx.get("buy_common",False) and (ppo.get("prev",np.nan)<=ppo_sig.get("prev",np.nan)) and (ppo.get("curr",np.nan)>ppo_sig.get("curr",np.nan)) and (ppo.get("curr",np.nan)<Constants.PPO_SIGNAL_CROSS_MAX_BUY) and (ctx.get("ppo_gate_curr",np.nan)<Constants.PPO_RSI_GUARD_BUY)),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"PPO {ppo.get('curr',0):.2f} vs Sig {ppo_sig.get('curr',0):.2f} | PPOgate {ctx.get('ppo_gate_curr',0):.2f} | Wick {ctx.get('buy_wick_ratio',0)*100:.1f}%","requires":["ppo","ppo_signal"]},
    {"key":"ppo_signal_down","title":"🔴 PPO cross▼signal","check_fn":lambda ctx,ppo,ppo_sig,rsi:(ctx.get("sell_common",False) and (ppo.get("prev",np.nan)>=ppo_sig.get("prev",np.nan)) and (ppo.get("curr",np.nan)<ppo_sig.get("curr",np.nan)) and (ppo.get("curr",np.nan)>Constants.PPO_SIGNAL_CROSS_MIN_SELL) and (ctx.get("ppo_gate_curr",np.nan)>Constants.PPO_RSI_GUARD_SELL)),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"PPO {ppo.get('curr',0):.2f} vs Sig {ppo_sig.get('curr',0):.2f} | PPOgate {ctx.get('ppo_gate_curr',0):.2f} | Wick {ctx.get('sell_wick_ratio',0)*100:.1f}%","requires":["ppo","ppo_signal"]},
    {"key":"ppo_zero_up","title":"🟢 PPO cross▲0","check_fn":lambda ctx,ppo,ppo_sig,rsi:(ctx.get("buy_common",False) and (ppo.get("prev",np.nan)<=0.0) and (ppo.get("curr",np.nan)>0.0) and (ctx.get("ppo_gate_curr",np.nan)<Constants.PPO_RSI_GUARD_BUY)),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"PPO {ppo.get('curr',0):.2f} | PPOgate {ctx.get('ppo_gate_curr',0):.2f} | Wick {ctx.get('buy_wick_ratio',0)*100:.1f}%","requires":["ppo"]},
    {"key":"ppo_zero_down","title":"🔴 PPO cross▼0","check_fn":lambda ctx,ppo,ppo_sig,rsi:(ctx.get("sell_common",False) and (ppo.get("prev",np.nan)>=0.0) and (ppo.get("curr",np.nan)<0.0) and (ctx.get("ppo_gate_curr",np.nan)>Constants.PPO_RSI_GUARD_SELL)),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"PPO {ppo.get('curr',0):.2f} | PPOgate {ctx.get('ppo_gate_curr',0):.2f} | Wick {ctx.get('sell_wick_ratio',0)*100:.1f}%","requires":["ppo"]},
    {"key":"ppo_adaptive_up","title":"🟢 PPO cross▲adapt","check_fn":lambda ctx,ppo,ppo_sig,rsi:(ctx.get("buy_common",False) and (ppo.get("prev",np.nan)<=ctx.get("ppo_adaptive_threshold",0.11)) and (ppo.get("curr",np.nan)>ctx.get("ppo_adaptive_threshold",0.11)) and (ctx.get("ppo_gate_curr",np.nan)<Constants.PPO_RSI_GUARD_BUY)),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"PPO {ppo.get('curr',0):.2f} vs adapt {ctx.get('ppo_adaptive_threshold',0):.3f} | PPOgate {ctx.get('ppo_gate_curr',0):.2f} | Wick {ctx.get('buy_wick_ratio',0)*100:.1f}%","requires":["ppo"]},
    {"key":"ppo_adaptive_down","title":"🔴 PPO cross▼adapt","check_fn":lambda ctx,ppo,ppo_sig,rsi:(ctx.get("sell_common",False) and (ppo.get("prev",np.nan)>=-ctx.get("ppo_adaptive_threshold",0.11)) and (ppo.get("curr",np.nan)<-ctx.get("ppo_adaptive_threshold",0.11)) and (ctx.get("ppo_gate_curr",np.nan)>Constants.PPO_RSI_GUARD_SELL)),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"PPO {ppo.get('curr',0):.2f} vs adapt {-ctx.get('ppo_adaptive_threshold',0):.3f} | PPOgate {ctx.get('ppo_gate_curr',0):.2f} | Wick {ctx.get('sell_wick_ratio',0)*100:.1f}%","requires":["ppo"]},
    {"key":"rsi_ema5_up","title":"🟢 RSI▲EMA5","check_fn":lambda ctx,ppo,ppo_sig,rsi:(ctx.get("buy_common",False) and (rsi.get("prev",50)<=rsi.get("ema_prev",50)) and (rsi.get("curr",50)>rsi.get("ema_curr",50)) and (rsi.get("curr",50)<ctx.get("rsi_adaptive_buy",60)) and (ctx.get("ppo_gate_curr",np.nan)<Constants.PPO_RSI_GUARD_BUY)),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"RSI {rsi.get('curr',50):.2f} ▲EMA5 {rsi.get('ema_curr',50):.2f} | cap {ctx.get('rsi_adaptive_buy',0):.1f} | PPOgate {ctx.get('ppo_gate_curr',0):.2f} | Wick {ctx.get('buy_wick_ratio',0)*100:.1f}%","requires":["rsi"]},
    {"key":"rsi_ema5_down","title":"🔴 RSI▼EMA5","check_fn":lambda ctx,ppo,ppo_sig,rsi:(ctx.get("sell_common",False) and (rsi.get("prev",50)>=rsi.get("ema_prev",50)) and (rsi.get("curr",50)<rsi.get("ema_curr",50)) and (rsi.get("curr",50)>ctx.get("rsi_adaptive_sell",40)) and (ctx.get("ppo_gate_curr",np.nan)>Constants.PPO_RSI_GUARD_SELL)),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"RSI {rsi.get('curr',50):.2f} ▼EMA5 {rsi.get('ema_curr',50):.2f} | cap {ctx.get('rsi_adaptive_sell',0):.1f} | PPOgate {ctx.get('ppo_gate_curr',0):.2f} | Wick {ctx.get('sell_wick_ratio',0)*100:.1f}%","requires":["rsi"]},
    {"key":"rsi_cross_adaptive_up","title":"🟢 RSI▲adapt","check_fn":lambda ctx,ppo,ppo_sig,rsi:(ctx.get("buy_common",False) and (rsi.get("curr",50)>rsi.get("ema_curr",50)) and (rsi.get("prev",50)<=ctx.get("rsi_adaptive_buy",60)) and (rsi.get("curr",50)>ctx.get("rsi_adaptive_buy",60)) and (ctx.get("ppo_gate_curr",np.nan)<Constants.PPO_RSI_GUARD_BUY)),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"RSI {rsi.get('curr',50):.2f} ▲{ctx.get('rsi_adaptive_buy',0):.1f} | EMA5 {rsi.get('ema_curr',50):.2f} | PPOgate {ctx.get('ppo_gate_curr',0):.2f} | Wick {ctx.get('buy_wick_ratio',0)*100:.1f}%","requires":["rsi"]},
    {"key":"rsi_cross_adaptive_down","title":"🔴 RSI▼adapt","check_fn":lambda ctx,ppo,ppo_sig,rsi:(ctx.get("sell_common",False) and (rsi.get("curr",50)<rsi.get("ema_curr",50)) and (rsi.get("prev",50)>=ctx.get("rsi_adaptive_sell",40)) and (rsi.get("curr",50)<ctx.get("rsi_adaptive_sell",40)) and (ctx.get("ppo_gate_curr",np.nan)>Constants.PPO_RSI_GUARD_SELL)),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"RSI {rsi.get('curr',50):.2f} ▼{ctx.get('rsi_adaptive_sell',0):.1f} | EMA5 {rsi.get('ema_curr',50):.2f} | PPOgate {ctx.get('ppo_gate_curr',0):.2f} | Wick {ctx.get('sell_wick_ratio',0)*100:.1f}%","requires":["rsi"]},
    {"key":"vwap_up","title":"🔵▲ VWAP Cross","check_fn":lambda ctx,ppo,ppo_sig,rsi:ctx.get("buy_common",False),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"VWAP {ctx.get('vwap_curr',0):.2f} | Wick {ctx.get('buy_wick_ratio',0)*100:.1f}%","requires":["vwap"]},
    {"key":"vwap_down","title":"🟣▼ VWAP Cross","check_fn":lambda ctx,ppo,ppo_sig,rsi:ctx.get("sell_common",False),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"VWAP {ctx.get('vwap_curr',0):.2f} | Wick {ctx.get('sell_wick_ratio',0)*100:.1f}%","requires":["vwap"]},
    {"key":"hist_rma_buy","title":"🔵⬆️ RMA Rev BUY","check_fn":lambda ctx,ppo,ppo_sig,rsi:(ctx.get("buy_common",False) and ctx.get("hist_reversal_buy",False)),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"Hist ({ctx.get('hist_curr',0):.4f}) | Wick {ctx.get('buy_wick_ratio',0)*100:.1f}%","requires":[]},
    {"key":"hist_rma_sell","title":"🟣⬇️ RMA Rev SELL","check_fn":lambda ctx,ppo,ppo_sig,rsi:(ctx.get("sell_common",False) and ctx.get("hist_reversal_sell",False)),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"Hist ({ctx.get('hist_curr',0):.4f}) | Wick {ctx.get('sell_wick_ratio',0)*100:.1f}%","requires":[]},
    {"key":"ppohist_buy","title":"🟢🔥 PPO Rev BUY","check_fn":lambda ctx,ppo,ppo_sig,rsi:(ctx.get("buy_common",False) and ctx.get("ppohist_reversal_buy",False)),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"PPOHist ({ctx.get('ppohist_curr',0):.4f}) | Wick {ctx.get('buy_wick_ratio',0)*100:.1f}%","requires":[]},
    {"key":"ppohist_sell","title":"🔴🔥 PPO Rev SELL","check_fn":lambda ctx,ppo,ppo_sig,rsi:(ctx.get("sell_common",False) and ctx.get("ppohist_reversal_sell",False)),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"PPOHist ({ctx.get('ppohist_curr',0):.4f}) | Wick {ctx.get('sell_wick_ratio',0)*100:.1f}%","requires":[]}, 
    {"key":"cloud_cross_up","title":"☁️🟢 Cloud Up Cross","check_fn":lambda ctx,ppo,ppo_sig,rsi:ctx.get("buy_common",False),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"Cloud Upper {ctx.get('cloud_upper_curr',0):.2f} | Wick {ctx.get('buy_wick_ratio',0)*100:.1f}%","requires":[]},
    {"key":"cloud_cross_down","title":"☁️🔴 Cloud Down Cross","check_fn":lambda ctx,ppo,ppo_sig,rsi:ctx.get("sell_common",False),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"Cloud Lower {ctx.get('cloud_lower_curr',0):.2f} | Wick {ctx.get('sell_wick_ratio',0)*100:.1f}%","requires":[]}, 
    {"key":"tk_conversion_up","title":"🌐🟢 Tenkan Cross","check_fn":lambda ctx,ppo,ppo_sig,rsi:ctx.get("buy_common",False),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"Conv {ctx.get('tk_conversion_curr',0):.2f} | Wick {ctx.get('buy_wick_ratio',0)*100:.1f}%","requires":[]},
    {"key":"tk_conversion_down","title":"🌐🔴 Tenkan Cross","check_fn":lambda ctx,ppo,ppo_sig,rsi:ctx.get("sell_common",False),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"Conv {ctx.get('tk_conversion_curr',0):.2f} | Wick {ctx.get('sell_wick_ratio',0)*100:.1f}%","requires":[]}, 
    {"key":"kijun_cross_up","title":"⚓🟢 Kijun Cross","check_fn":lambda ctx,ppo,ppo_sig,rsi:ctx.get("buy_common",False),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"Base {ctx.get('tk_base_curr',0):.2f} | Wick {ctx.get('buy_wick_ratio',0)*100:.1f}%","requires":[]},
    {"key":"kijun_cross_down","title":"⚓🔴 Kijun Cross","check_fn":lambda ctx,ppo,ppo_sig,rsi:ctx.get("sell_common",False),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"Base {ctx.get('tk_base_curr',0):.2f} | Wick {ctx.get('sell_wick_ratio',0)*100:.1f}%","requires":[]}, 
    {"key":"fast_cloud_cross_up","title":"⚡☁️🟢 Fast Cloud Up Cross","check_fn":lambda ctx,ppo,ppo_sig,rsi:(ctx.get("buy_common",False) and ctx.get("fast_future_green",False) and ctx.get("fast_tenkan_ge_kijun",False)),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"FastCloud {ctx.get('fast_cloud_upper_curr',0):.2f} | Wick {ctx.get('buy_wick_ratio',0)*100:.1f}%","requires":[]},
    {"key":"fast_cloud_cross_down","title":"⚡☁️🔴 Fast Cloud Down Cross","check_fn":lambda ctx,ppo,ppo_sig,rsi:(ctx.get("sell_common",False) and ctx.get("fast_future_red",False) and ctx.get("fast_tenkan_le_kijun",False)),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"FastCloud {ctx.get('fast_cloud_lower_curr',0):.2f} | Wick {ctx.get('sell_wick_ratio',0)*100:.1f}%","requires":[]},
    {"key":"fast_tenkan_cross_up","title":"⚡🌐🟢 Fast Tenkan Cross","check_fn":lambda ctx,ppo,ppo_sig,rsi:(ctx.get("buy_common",False) and ctx.get("close_curr",float('-inf'))>ctx.get("fast_cloud_upper_curr",float('inf')) and ctx.get("fast_future_green",False) and ctx.get("fast_tenkan_ge_kijun",False)),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"FastConv {ctx.get('fast_tk_conversion_curr',0):.2f} | Wick {ctx.get('buy_wick_ratio',0)*100:.1f}%","requires":[]},
    {"key":"fast_tenkan_cross_down","title":"⚡🌐🔴 Fast Tenkan Cross","check_fn":lambda ctx,ppo,ppo_sig,rsi:(ctx.get("sell_common",False) and ctx.get("close_curr",float('inf'))<ctx.get("fast_cloud_lower_curr",float('-inf')) and ctx.get("fast_future_red",False) and ctx.get("fast_tenkan_le_kijun",False)),"extra_fn":lambda ctx,ppo,ppo_sig,rsi,_:f"FastConv {ctx.get('fast_tk_conversion_curr',0):.2f} | Wick {ctx.get('sell_wick_ratio',0)*100:.1f}%","requires":[]}
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

    if close_curr is None or close_prev is None or np.isnan(close_curr) or np.isnan(close_prev):
        return False, "Missing or invalid close data"

    # Precise cross verification
    if is_buy:
        crossed = close_prev <= level_value < close_curr
    else:
        crossed = close_prev >= level_value > close_curr

    if not crossed:
        return False, "No pivot cross"

    price_diff_pct = (abs(level_value - close_curr) / level_value) * 100
    max_distance = cfg.PIVOT_MAX_DISTANCE_PCT

    if price_diff_pct > max_distance:
        return False, (
            f"Pivot too far: price {close_curr:.2f} is {price_diff_pct:.2f}% "
            f"away from {level} pivot {level_value:.2f} (max {max_distance}%)"
        )

    return True, None

def _build_resets(pair_name: str, context: dict, conditional_states: dict) -> List[Tuple[str, str, None]]:
    """Generic cross-reset engine. Emits INACTIVE updates when a cross that was
    previously ACTIVE has now reversed."""
    resets: List[Tuple[str, str, None]] = []

    def _add(up_key: str, down_key: str,
             curr: float, prev: float,
             up_thr_curr: float, up_thr_prev: float,
             down_thr_curr: float, down_thr_prev: float) -> None:
        if prev > up_thr_prev and curr <= up_thr_curr:
            rk = ALERT_KEYS.get(up_key)
            if rk and conditional_states.get(rk, False):
                resets.append((f"{pair_name}:{rk}", "INACTIVE", None))
        if prev < down_thr_prev and curr >= down_thr_curr:
            rk = ALERT_KEYS.get(down_key)
            if rk and conditional_states.get(rk, False):
                resets.append((f"{pair_name}:{rk}", "INACTIVE", None))

    # ── PPO ──
    ppo_c, ppo_p = context["ppo_curr"], context["ppo_prev"]
    ps_c,  ps_p  = context["ppo_sig_curr"], context["ppo_sig_prev"]
    thr = context["ppo_adaptive_threshold"]
    _add("ppo_signal_up", "ppo_signal_down", ppo_c, ppo_p, ps_c, ps_p, ps_c, ps_p)
    _add("ppo_zero_up",   "ppo_zero_down",   ppo_c, ppo_p, 0.0, 0.0, 0.0, 0.0)
    _add("ppo_adaptive_up", "ppo_adaptive_down", ppo_c, ppo_p, thr, thr, -thr, -thr)

    # ── RSI ──
    rsi_c, rsi_p = context["rsi_curr"], context["rsi_prev"]
    ema_c, ema_p = context["rsi_ema_curr"], context["rsi_ema_prev"]
    _add("rsi_ema5_up", "rsi_ema5_down", rsi_c, rsi_p, ema_c, ema_p, ema_c, ema_p)
    buy_thr, sell_thr = context["rsi_adaptive_buy"], context["rsi_adaptive_sell"]
    _add("rsi_cross_adaptive_up", "rsi_cross_adaptive_down",
         rsi_c, rsi_p, buy_thr, buy_thr, sell_thr, sell_thr)

    # ── VWAP ──
    if context.get("vwap_available"):
        _add("vwap_up", "vwap_down", context["close_curr"], context["close_prev"],
             context["vwap_curr"], context["vwap_prev"], context["vwap_curr"], context["vwap_prev"])
    else:
        for k in ("vwap_up", "vwap_down"):
            rk = ALERT_KEYS.get(k)
            if rk and conditional_states.get(rk, False):
                resets.append((f"{pair_name}:{rk}", "INACTIVE", None))

    # ── Cloud crosses (slow + fast) ──
    for up_k, down_k, cu, cu_p, cl, cl_p in (
        ("cloud_cross_up", "cloud_cross_down",
         "cloud_upper_curr", "cloud_upper_prev", "cloud_lower_curr", "cloud_lower_prev"),
        ("fast_cloud_cross_up", "fast_cloud_cross_down",
         "fast_cloud_upper_curr", "fast_cloud_upper_prev", "fast_cloud_lower_curr", "fast_cloud_lower_prev"),
    ):
        cu_c, cu_pr = context.get(cu), context.get(cu_p)
        cl_c, cl_pr = context.get(cl), context.get(cl_p)
        if all(v is not None and not np.isnan(v) for v in (cu_c, cu_pr, cl_c, cl_pr)):
            _add(up_k, down_k, context["close_curr"], context["close_prev"], cu_c, cu_pr, cl_c, cl_pr)
        else:
            for k in (up_k, down_k):
                rk = ALERT_KEYS.get(k)
                if rk and conditional_states.get(rk, False):
                    resets.append((f"{pair_name}:{rk}", "INACTIVE", None))

    # ── Conversion / Kijun / Fast Tenkan ──
    for up_k, down_k, conv, conv_p in (
        ("tk_conversion_up", "tk_conversion_down", "tk_conversion_curr", "tk_conversion_prev"),
        ("kijun_cross_up",   "kijun_cross_down",   "tk_base_curr",       "tk_base_prev"),
        ("fast_tenkan_cross_up", "fast_tenkan_cross_down", "fast_tk_conversion_curr", "fast_tk_conversion_prev"),
    ):
        c_c, c_p = context.get(conv), context.get(conv_p)
        if c_c is not None and c_p is not None and not np.isnan(c_c) and not np.isnan(c_p):
            _add(up_k, down_k, context["close_curr"], context["close_prev"], c_c, c_p, c_c, c_p)
        else:
            for k in (up_k, down_k):
                rk = ALERT_KEYS.get(k)
                if rk and conditional_states.get(rk, False):
                    resets.append((f"{pair_name}:{rk}", "INACTIVE", None))

    # ── Hist RMA ──
    hist_c, hist_m1 = context["hist_curr"], context["hist_m1"]
    for k, cond in (("hist_rma_buy",  np.isnan(hist_c) or hist_c <= 1e-8 or hist_c <= hist_m1),
                    ("hist_rma_sell", np.isnan(hist_c) or hist_c >= -1e-8 or hist_c >= hist_m1)):
        rk = ALERT_KEYS.get(k)
        if rk and conditional_states.get(rk, False) and cond:
            resets.append((f"{pair_name}:{rk}", "INACTIVE", None))

    # ── PPO Hist ──
    ph_c, ph_m1 = context["ppohist_curr"], context["ppohist_m1"]
    for k, cond in (("ppohist_buy",  np.isnan(ph_c) or ph_c <= 1e-8 or ph_c <= ph_m1),
                    ("ppohist_sell", np.isnan(ph_c) or ph_c >= -1e-8 or ph_c >= ph_m1)):
        rk = ALERT_KEYS.get(k)
        if rk and conditional_states.get(rk, False) and cond:
            resets.append((f"{pair_name}:{rk}", "INACTIVE", None))

    # ── Pivots ──
    piv = context.get("pivots", {})
    close_c, close_p = context["close_curr"], context["close_prev"]
    if not piv:
        for lvl in set(PIVOT_LEVELS_BUY + PIVOT_LEVELS_SELL):
            for prefix in ("pivot_up_", "pivot_down_"):
                k = f"{prefix}{lvl}"
                rk = ALERT_KEYS.get(k)
                if rk and conditional_states.get(rk, False):
                    resets.append((f"{pair_name}:{rk}", "INACTIVE", None))
    else:
        for lvl, val in piv.items():
            up_k = f"pivot_up_{lvl}"
            rk = ALERT_KEYS.get(up_k)
            if rk and conditional_states.get(rk, False) and close_p > val and close_c <= val:
                resets.append((f"{pair_name}:{rk}", "INACTIVE", None))
            down_k = f"pivot_down_{lvl}"
            rk = ALERT_KEYS.get(down_k)
            if rk and conditional_states.get(rk, False) and close_p < val and close_c >= val:
                resets.append((f"{pair_name}:{rk}", "INACTIVE", None))

    return resets

def get_pivot_alert_info(ctx: Dict[str, Any], level: str, is_buy: bool) -> Tuple[bool, Optional[str]]:
    cache_key = f"_pivot_cache_{level}_{'buy' if is_buy else 'sell'}"
    
    if cache_key not in ctx:
        ctx[cache_key] = _validate_pivot_cross(ctx, level, is_buy)
    
    return ctx[cache_key]

BUY_PIVOT_DEFS = [create_pivot_alert(level, is_buy=True) 
                  for level in PIVOT_LEVELS_BUY]

SELL_PIVOT_DEFS = [create_pivot_alert(level, is_buy=False) 
                   for level in PIVOT_LEVELS_SELL]

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

BUY_ALERT_KEYS: Set[str] = {
    "ppo_signal_up", "ppo_zero_up", "ppo_adaptive_up",
    "rsi_ema5_up", "rsi_cross_adaptive_up", "vwap_up", "hist_rma_buy", "ppohist_buy",
    "cloud_cross_up", "tk_conversion_up", "kijun_cross_up",
    "fast_cloud_cross_up", "fast_tenkan_cross_up",
}
BUY_ALERT_KEYS.update(f"pivot_up_{level}" for level in PIVOT_LEVELS_BUY)

SELL_ALERT_KEYS: Set[str] = {
    "ppo_signal_down", "ppo_zero_down", "ppo_adaptive_down",
    "rsi_ema5_down", "rsi_cross_adaptive_down", "vwap_down", "hist_rma_sell", "ppohist_sell",
    "cloud_cross_down", "tk_conversion_down", "kijun_cross_down",
    "fast_cloud_cross_down", "fast_tenkan_cross_down",
}
SELL_ALERT_KEYS.update(f"pivot_down_{level}" for level in PIVOT_LEVELS_SELL)

async def _eval_gate(pair_name: str, data_15m: PriceData, data_5m: PriceData,
    data_daily: Optional[Dict[str, np.ndarray]], sdb: RedisStateStore, correlation_id: str,
    reference_time: int) -> Union[GateResult, Tuple[str, Dict[str, Any]], None]:
    if reference_time is None:
        reference_time = get_trigger_timestamp()

    logger_pair = logging.getLogger(f"macd_bot.{pair_name}.{correlation_id}")
    PAIR_ID.set(pair_name)
    close_15m = None
    timestamps_15m = None
    rma50_15 = None
    rma200_5 = None

    try:
        i15 = get_last_closed_index_from_array(data_15m.ts, 15, reference_time, pair_name)
        if i15 is None or i15 < Constants.MIN_CLOSED_CANDLES_15M:
            return None

        is_valid_for_buy, is_valid_for_sell, candle_info, error_msg = validate_candle_for_alerts(
            data_15m=data_15m.as_dict(),
            candle_index=i15,
            reference_time=reference_time,
            pair_name=pair_name,
            min_wick_ratio=Constants.MIN_WICK_RATIO
        )
        if not is_valid_for_buy and not is_valid_for_sell:
            if candle_info is None:
                logger_pair.debug(
                    f"[{pair_name}] Hard-rejecting candle: {error_msg}"
                )
                await _blanket_reset_pair(sdb, pair_name, logger_pair)
                return pair_name, {
                    "state": "HARD_REJECT",
                    "ts": int(time.time()),
                    "summary": {
                        "alerts": 0,
                        "future_cloud": "neutral",
                        "hist_rma": 0.0,
                        "suppression": f"Hard reject: {error_msg}"
                    }
                }
            logger_pair.debug(
                f"[{pair_name}] Wick-rejected candle  blanket reset only. Reason: {error_msg}"
            )
            await _blanket_reset_pair(sdb, pair_name, logger_pair)
            return pair_name, {
                "state": "NO_SIGNAL",
                "ts": int(time.time()),
                "summary": {
                    "alerts": 0,
                    "future_cloud": "neutral",
                    "hist_rma": 0.0,
                    "suppression": f"Wick rejected: {error_msg}"
                }
            }
        o = candle_info["open"]
        h = candle_info["high"]
        l = candle_info["low"]
        c = candle_info["close"]
        ts_curr = candle_info["timestamp"]
        is_green = candle_info["is_green"]
        is_red = candle_info["is_red"]
        buy_wick_ratio = candle_info["upper_wick_ratio"]
        sell_wick_ratio = candle_info["lower_wick_ratio"]
  
        if is_valid_for_buy and not is_green:
            raise RuntimeError(
                f"[{pair_name}] INVARIANT VIOLATED: is_valid_for_buy=True on non-green candle | "
                f"O={o:.2f} C={c:.2f}"
            )
        if is_valid_for_sell and not is_red:
            raise RuntimeError(
                f"[{pair_name}] INVARIANT VIOLATED: is_valid_for_sell=True on non-red candle | "
                f"O={o:.2f} C={c:.2f}"
            )

        logger_pair.debug(
            f"[{pair_name}] 🕯️ Candle | O={o:.2f} H={h:.2f} L={l:.2f} C={c:.2f} | "
            f"{'🟢 GREEN' if is_green else '🔴 RED'} | "
            f"ValidBuy={is_valid_for_buy} ValidSell={is_valid_for_sell}"
        )
        open_curr = o
        high_curr = h
        low_curr = l
        close_curr = c
        candle_range = h - l

        close_15m = data_15m.close
        timestamps_15m = data_15m.ts

        interval_5m_sec = 5 * 60
        expected_5m_open = (reference_time // interval_5m_sec) * interval_5m_sec - interval_5m_sec

        ts_5m_arr = normalize_timestamp_array(data_5m.ts)

        matches_5m = np.flatnonzero(np.abs(ts_5m_arr - expected_5m_open) <= 30)

        if matches_5m.size > 0:
            i5 = int(matches_5m[-1])
            actual_5m_ts = int(ts_5m_arr[i5])
        else:
            ts_15m_val = int(normalize_timestamp(int(data_15m.ts[i15])))
            window_mask = (ts_5m_arr >= ts_15m_val) & (ts_5m_arr < ts_15m_val + 900)
            if np.any(window_mask):
                fallback_idx = int(np.flatnonzero(window_mask)[-1])
                i5 = fallback_idx
                actual_5m_ts = int(ts_5m_arr[fallback_idx])
                if logger_pair.isEnabledFor(logging.DEBUG):
                    logger_pair.debug(
                        f"[{pair_name}] 5m fallback: using {format_ist_time(actual_5m_ts)} "
                        f"(expected {format_ist_time(expected_5m_open)} not available)"
                    )    
            else:
                logger_pair.warning(
                    f"[{pair_name}] 5m candle not found at {format_ist_time(expected_5m_open)} "
                    f"and no fallback in 15m window. Range: {format_ist_time(int(ts_5m_arr[0]))} "
                    f"to {format_ist_time(int(ts_5m_arr[-1]))}"
                )
                return None

        time_since_5m_closed = reference_time - (actual_5m_ts + interval_5m_sec)
        if time_since_5m_closed < cfg.CANDLE_MIN_AGE_BUFFER:
            logger_pair.warning(
                f"[{pair_name}] 5m candle at {format_ist_time(actual_5m_ts)} not stable yet "
                f"(closed {time_since_5m_closed}s ago, need {cfg.CANDLE_MIN_AGE_BUFFER}s). Skipping."
            )
            return None

        ts_15m_val = int(normalize_timestamp(int(data_15m.ts[i15])))
        if actual_5m_ts < ts_15m_val or actual_5m_ts >= ts_15m_val + 900:
            logger_pair.error(
                f"[{pair_name}] 5m/15m misalignment: 5m={format_ist_time(actual_5m_ts)} "
                f"outside 15m window {format_ist_time(ts_15m_val)}-{format_ist_time(ts_15m_val + 900)}"
            )
            return None

        expected_last_5m = ts_15m_val + 600
        if actual_5m_ts != expected_last_5m:
            if logger_pair.isEnabledFor(logging.DEBUG):
                logger_pair.debug(
                    f"[{pair_name}] Using non-last 5m candle: got {format_ist_time(actual_5m_ts)}, "
                    f"expected {format_ist_time(expected_last_5m)}"
                )

        if i5 < Constants.MIN_ALIGNED_5M_CANDLES:
            return None

        if logger_pair.isEnabledFor(logging.DEBUG):
            logger_pair.debug(
                f"[{pair_name}] 5m candle selected | "
                f"Open={format_ist_time(actual_5m_ts)} | i5={i5} | "
                f"Close={data_5m.close[i5]:.2f}"
            )

        # ═══════════════════════════════════════════════════════
        # PHASE 1 — Gate indicators only (cheap)
        # ═══════════════════════════════════════════════════════
        gate_indicators = await asyncio.to_thread(
            calculate_gate_indicators_numpy, data_15m.as_dict(), data_5m.as_dict(), data_daily, reference_time
        )
        if gate_indicators is None:
            logger_pair.error(f"Skipping {pair_name}: gate indicators failed")
            return None

        # ── Extract gate values ──
        rma50_15 = gate_indicators["rma50_15"]
        rma200_5 = gate_indicators["rma200_5"]
        ichimoku_cloud_upper = gate_indicators["ichimoku_cloud_upper"]
        ichimoku_cloud_lower = gate_indicators["ichimoku_cloud_lower"]
        ichimoku_future_green = gate_indicators["ichimoku_future_green"]
        ichimoku_future_red = gate_indicators["ichimoku_future_red"]
        ichimoku_conversion_line = gate_indicators["ichimoku_conversion_line"]
        ichimoku_base_line = gate_indicators["ichimoku_base_line"]
        fast_ichimoku_cloud_upper = gate_indicators["fast_ichimoku_cloud_upper"]
        fast_ichimoku_cloud_lower = gate_indicators["fast_ichimoku_cloud_lower"]
        fast_ichimoku_future_green = gate_indicators["fast_ichimoku_future_green"]
        fast_ichimoku_future_red = gate_indicators["fast_ichimoku_future_red"]
        fast_ichimoku_conversion_line = gate_indicators["fast_ichimoku_conversion_line"]
        fast_ichimoku_base_line = gate_indicators["fast_ichimoku_base_line"]
        adx_arr = gate_indicators["adx"]
        atr_short_arr = gate_indicators["atr_short"]
        atr_long_arr = gate_indicators["atr_long"]
        volume_ema_arr = gate_indicators["volume_ema"]
        ppo_gate_arr = gate_indicators["ppo_gate"]
        ppo_gate_signal_arr = gate_indicators["ppo_gate_signal"]
        rsi_guard_smooth_arr = gate_indicators["rsi_guard_smooth"]
        rsi_guard_ema_arr = gate_indicators["rsi_guard_ema"]
        rma_cloud_fast_arr = gate_indicators["rma_cloud_fast_15"]
        cpr_ok = gate_indicators.get('cpr_ok', not cfg.ENABLE_CPR)
        nr_cpr = gate_indicators.get('nr_cpr', float('nan'))
        prev_day_close = gate_indicators.get('prev_day_close', float('nan'))

        future_green = ichimoku_future_green[i15]
        future_red = ichimoku_future_red[i15]

        cloud_upper_val = ichimoku_cloud_upper[i15]
        cloud_lower_val = ichimoku_cloud_lower[i15]
        cloud_upper_prev = ichimoku_cloud_upper[i15 - 1]
        cloud_lower_prev = ichimoku_cloud_lower[i15 - 1]

        ichimoku_cloud_ready = not (
            np.isnan(cloud_upper_val) or np.isnan(cloud_lower_val)
            or np.isnan(cloud_upper_prev) or np.isnan(cloud_lower_prev)
        )
        if ichimoku_cloud_ready:
            above_cloud = close_curr > cloud_upper_val
            below_cloud = close_curr < cloud_lower_val
            cloud_up = future_green and above_cloud
            cloud_down = future_red and below_cloud
        else:
            logger_pair.debug(
                f"[{pair_name}] Ichimoku cloud NaN at i15={i15} (warmup/gap). "
                f"Ichimoku cloud gate abstains (None) — not counted in cloud-group vote."
            )
            above_cloud = None
            below_cloud = None
            cloud_up = None
            cloud_down = None

        tk_conversion_curr = ichimoku_conversion_line[i15]
        tk_conversion_prev = ichimoku_conversion_line[i15 - 1]
        tk_base_curr = ichimoku_base_line[i15]
        tk_base_prev = ichimoku_base_line[i15 - 1]
        tk_guard_valid = not (np.isnan(tk_conversion_curr) or np.isnan(tk_base_curr))

        if cfg.ICHIMOKU_TK_GUARD_ENABLED:
            if tk_guard_valid:
                tk_guard_ok_buy = (tk_conversion_curr >= tk_base_curr) and (close_curr > tk_base_curr)
                tk_guard_ok_sell = (tk_conversion_curr <= tk_base_curr) and (close_curr < tk_base_curr)
            else:
                logger_pair.debug(
                    f"[{pair_name}] TK lines not ready at i15={i15}. "
                    f"TK guard abstains (None) this run — not counted in majority vote."
                )
                tk_guard_ok_buy = None
                tk_guard_ok_sell = None
        else:
            tk_guard_ok_buy = None
            tk_guard_ok_sell = None

        fast_future_green = fast_ichimoku_future_green[i15]
        fast_future_red = fast_ichimoku_future_red[i15]

        fast_cloud_upper_curr = fast_ichimoku_cloud_upper[i15]
        fast_cloud_lower_curr = fast_ichimoku_cloud_lower[i15]
        fast_cloud_upper_prev = fast_ichimoku_cloud_upper[i15 - 1]
        fast_cloud_lower_prev = fast_ichimoku_cloud_lower[i15 - 1]

        fast_tk_conversion_curr = fast_ichimoku_conversion_line[i15]
        fast_tk_conversion_prev = fast_ichimoku_conversion_line[i15 - 1]
        fast_tk_base_curr = fast_ichimoku_base_line[i15]
        fast_tk_base_prev = fast_ichimoku_base_line[i15 - 1]

        fast_tk_valid = not (np.isnan(fast_tk_conversion_curr) or np.isnan(fast_tk_base_curr))
        fast_tenkan_ge_kijun = fast_tk_valid and (fast_tk_conversion_curr >= fast_tk_base_curr)
        fast_tenkan_le_kijun = fast_tk_valid and (fast_tk_conversion_curr <= fast_tk_base_curr)

        close_prev = close_15m[i15 - 1]

        close_prev_invalid = False
        if np.isnan(close_prev) or np.isinf(close_prev) or close_prev <= 0:
            logger_pair.warning(
                f"[{pair_name}] Previous candle close invalid ({close_prev}). "
                f"Skipping all cross-based alerts this run."
            )
            close_prev_invalid = True

        if close_prev_invalid:
            logger_pair.warning(
                f"[{pair_name}] close_prev invalid — skipping all cross alerts"
            )
            await _blanket_reset_pair(sdb, pair_name, logger_pair)
            return pair_name, {
                "state": "INVALID_PREV_CLOSE",
                "ts": int(time.time()),
                "summary": {
                    "alerts": 0,
                    "future_cloud": "neutral",
                    "hist_rma": 0.0,
                    "suppression": "close_prev was NaN/Inf/≤0"
                }
            }  
        close_5m_val = data_5m.close[i5]
        rma50_15_val = rma50_15[i15]
        rma200_5_val = rma200_5[i5]

        base_buy_trend = (rma50_15_val < close_curr) and (rma200_5_val < close_5m_val)
        base_sell_trend = (rma50_15_val > close_curr) and (rma200_5_val > close_5m_val)

        if cfg.ICHIMOKU_CLOUD_ENABLED:
            ichimoku_gate_ok_buy = cloud_up
            ichimoku_gate_ok_sell = cloud_down
        else:
            ichimoku_gate_ok_buy = None
            ichimoku_gate_ok_sell = None


        adx_val = adx_arr[i15] if not np.isnan(adx_arr[i15]) else 0.0
        adx_adaptive_threshold = get_adaptive_adx_threshold_smoothed(adx_arr, i15, cfg)
        adx_raw_check = adx_val >= adx_adaptive_threshold
        adx_ok = adx_raw_check if cfg.ENABLE_ADX_FILTER else True
        adx_bypass_ok = adx_raw_check

        atr_short_val = atr_short_arr[i15]
        atr_long_val = atr_long_arr[i15]

        atr_ratio_valid = (
            not np.isnan(atr_short_val) and not np.isnan(atr_long_val) and atr_long_val > 1e-9
        )
        atr_ratio = (atr_short_val / atr_long_val) if atr_ratio_valid else float('nan')

        adaptive_threshold = None
        if cfg.ATR_ADAPTIVE_ENABLED:
            adaptive_threshold = get_adaptive_rvol_threshold(atr_long_arr, i15, cfg)

        ppo_adaptive_threshold = get_adaptive_ppo_threshold(atr_long_arr, i15, cfg)
        rsi_adaptive_buy, rsi_adaptive_sell = get_adaptive_rsi_thresholds(atr_long_arr, i15, cfg)
        cpr_adaptive_min_pct_move = get_adaptive_cpr_threshold(atr_long_arr, i15, cfg)

        volume_curr = data_15m.volume[i15]
        volume_ema_curr = volume_ema_arr[i15]
        if not np.isnan(volume_curr) and not np.isnan(volume_ema_curr) and volume_ema_curr > 1e-9:
            volume_above_ema_ok = volume_curr > volume_ema_curr
        else:
            volume_above_ema_ok = False

        rvol_bypass_ok = atr_ratio_valid and (atr_ratio >= cfg.RVOL_THRESHOLD)

        adaptive_rvol_check = (
            atr_ratio_valid
            and adaptive_threshold is not None
            and atr_ratio >= adaptive_threshold
        )
        adx_pass = adx_raw_check if cfg.ENABLE_ADX_FILTER else False
        rvol_static_pass = rvol_bypass_ok if cfg.ENABLE_RVOL_ALERT else False
        rvol_adaptive_pass = adaptive_rvol_check  # False if ATR_ADAPTIVE_ENABLED=False
        
        adx_prev = adx_arr[i15 - 1] if i15 >= 1 else adx_val
        adx_rising = (
            not np.isnan(adx_val) and not np.isnan(adx_prev)
            and adx_prev > 0 and adx_val > adx_prev
        )

        rvol_vote_ok = rvol_static_pass or rvol_adaptive_pass

        body_conviction_ok = (
            candle_range > 1e-9
            and (abs(close_curr - open_curr) / candle_range) >= cfg.CPR_MOMENTUM_BODY_RATIO_MIN
        )
        momentum_conditions = [
            adx_bypass_ok,         # 1. ADX level >= threshold
            adx_rising,            # 2. ADX rising vs prior bar
            rvol_vote_ok,          # 3. RVOL (static or adaptive, single vote — not both)
            volume_above_ema_ok,   # 4. Volume > EMA(volume)
            body_conviction_ok,    # 5. Candle body conviction (|close-open|/range)
        ]
        momentum_count = sum(momentum_conditions)

        any_vol_feature_enabled = cfg.ENABLE_ADX_FILTER or cfg.ENABLE_RVOL_ALERT or cfg.ATR_ADAPTIVE_ENABLED
        volatility_filter_ok = (not any_vol_feature_enabled) or (momentum_count >= 3)
        rvol_ok = volatility_filter_ok

        if not np.isnan(prev_day_close) and prev_day_close > 0:
            pct_move_from_prev_close = abs(close_curr - prev_day_close) / prev_day_close * 100.0
            move_from_prev_close_ok = pct_move_from_prev_close >= cpr_adaptive_min_pct_move
        else:
            pct_move_from_prev_close = float('nan')
            move_from_prev_close_ok = False

        if cfg.ENABLE_CPR:
            if cpr_ok:  # Narrow CPR: momentum now enforced globally via volatility_filter_ok
                effective_cpr_ok = True
            else:       # Wide CPR: same, plus mandatory min % move from prior close
                effective_cpr_ok = move_from_prev_close_ok
        else:
            effective_cpr_ok = True

        if cfg.DEBUG_MODE and cfg.ENABLE_CPR:
            logger_pair.debug(
                f"[{pair_name}] CPR {'narrow' if cpr_ok else 'WIDE'} | "
                f"effective={effective_cpr_ok} | momentum={momentum_count}/5 "
                f"(adx={adx_val:.1f}[{adx_bypass_ok},{adx_rising}], "
                f"rvol={rvol_vote_ok}[static={rvol_static_pass},adaptive={rvol_adaptive_pass}]"
                f"[thr={adaptive_threshold if adaptive_threshold is not None else float('nan'):.3f}], "
                f"vol_ema={volume_above_ema_ok}, body={body_conviction_ok}) | "
                f"move_from_prev_close={pct_move_from_prev_close:.2f}%[{move_from_prev_close_ok}] | "
                f"NR_CPR={nr_cpr:.4f}"
            )
        if cfg.DEBUG_MODE:
            ratio_str = f"{atr_ratio:.3f}" if atr_ratio_valid else "n/a"
            adaptive_str = f"{adaptive_threshold:.3f}" if adaptive_threshold is not None else "n/a"
            logger_pair.debug(
                f"[{pair_name}] Volatility filter | "
                f"ratio={ratio_str} | "
                f"static={cfg.RVOL_THRESHOLD:.3f}[{rvol_bypass_ok}] | "
                f"adaptive={adaptive_str}[{adaptive_rvol_check}] | "
                f"adx={adx_val:.1f}[{adx_pass}] | "
                f"market_filter={volatility_filter_ok}"
            )
        ppo_gate_curr = ppo_gate_arr[i15]
        ppo_gate_prev = ppo_gate_arr[i15 - 1] if i15 >= 1 else ppo_gate_arr[i15]
        ppo_gate_sig_curr = ppo_gate_signal_arr[i15]
        ppo_gate_sig_prev = ppo_gate_signal_arr[i15 - 1] if i15 >= 1 else ppo_gate_signal_arr[i15]
        rsi_guard_smooth_curr = rsi_guard_smooth_arr[i15]
        rsi_guard_ema_curr = rsi_guard_ema_arr[i15]
        rma_cloud_fast_curr = rma_cloud_fast_arr[i15]

        if cfg.ENABLE_PPO_GATE:
            if not np.isnan(ppo_gate_curr) and not np.isnan(ppo_gate_sig_curr):
                ppo_gate_ok_buy = ppo_gate_curr > ppo_gate_sig_curr
                ppo_gate_ok_sell = ppo_gate_curr < ppo_gate_sig_curr
            else:
                ppo_gate_ok_buy = None
                ppo_gate_ok_sell = None
        else:
            ppo_gate_ok_buy = None
            ppo_gate_ok_sell = None

        if cfg.RSI_GUARD_ENABLED:
            if not np.isnan(rsi_guard_smooth_curr) and not np.isnan(rsi_guard_ema_curr):
                rsi_guard_ok_buy = rsi_guard_smooth_curr > rsi_guard_ema_curr
                rsi_guard_ok_sell = rsi_guard_smooth_curr < rsi_guard_ema_curr
            else:
                rsi_guard_ok_buy = None
                rsi_guard_ok_sell = None
        else:
            rsi_guard_ok_buy = None
            rsi_guard_ok_sell = None

        if cfg.RMA_CLOUD_ENABLED:
            if not np.isnan(rma_cloud_fast_curr) and not np.isnan(rma50_15_val):
                rma_cloud_ok_buy = rma_cloud_fast_curr > rma50_15_val
                rma_cloud_ok_sell = rma_cloud_fast_curr < rma50_15_val
            else:
                rma_cloud_ok_buy = None
                rma_cloud_ok_sell = None
        else:
            rma_cloud_ok_buy = None
            rma_cloud_ok_sell = None

        cloud_group_enabled = cfg.RMA_CLOUD_ENABLED or cfg.ICHIMOKU_CLOUD_ENABLED
        oscillator_group_enabled = cfg.ENABLE_PPO_GATE or cfg.RSI_GUARD_ENABLED or cfg.ICHIMOKU_TK_GUARD_ENABLED

        active_cloud_buy = [g for g in (ichimoku_gate_ok_buy, rma_cloud_ok_buy) if g is not None]
        if active_cloud_buy:
            cloud_group_ok_buy = sum(active_cloud_buy) >= 1
        elif cloud_group_enabled:
            logger_pair.debug(
                f"[{pair_name}] Cloud group: both gates abstained (warmup/gap) — buy denied."
            )
            cloud_group_ok_buy = False
        else:
            cloud_group_ok_buy = True

        active_cloud_sell = [g for g in (ichimoku_gate_ok_sell, rma_cloud_ok_sell) if g is not None]
        if active_cloud_sell:
            cloud_group_ok_sell = sum(active_cloud_sell) >= 1
        elif cloud_group_enabled:
            logger_pair.debug(
                f"[{pair_name}] Cloud group: both gates abstained (warmup/gap) — sell denied."
            )
            cloud_group_ok_sell = False
        else:
            cloud_group_ok_sell = True

        confirmation_buy = cloud_group_ok_buy
        confirmation_sell = cloud_group_ok_sell

        active_osc_buy = [g for g in (ppo_gate_ok_buy, rsi_guard_ok_buy, tk_guard_ok_buy) if g is not None]
        if active_osc_buy:
            oscillator_group_ok_buy = sum(active_osc_buy) >= 1
        elif oscillator_group_enabled:
            logger_pair.debug(
                f"[{pair_name}] Oscillator group: all gates abstained (warmup/gap) — buy denied."
            )
            oscillator_group_ok_buy = False
        else:
            oscillator_group_ok_buy = True

        active_osc_sell = [g for g in (ppo_gate_ok_sell, rsi_guard_ok_sell, tk_guard_ok_sell) if g is not None]
        if active_osc_sell:
            oscillator_group_ok_sell = sum(active_osc_sell) >= 1
        elif oscillator_group_enabled:
            logger_pair.debug(
                f"[{pair_name}] Oscillator group: all gates abstained (warmup/gap) — sell denied."
            )
            oscillator_group_ok_sell = False
        else:
            oscillator_group_ok_sell = True

        trend_gate_ok_buy = cloud_group_ok_buy and oscillator_group_ok_buy
        trend_gate_ok_sell = cloud_group_ok_sell and oscillator_group_ok_sell

        buy_common = (
            base_buy_trend and is_valid_for_buy
            and volatility_filter_ok and effective_cpr_ok
            and trend_gate_ok_buy
        )
        sell_common = (
            base_sell_trend and is_valid_for_sell
            and volatility_filter_ok and effective_cpr_ok
            and trend_gate_ok_sell
        )
        # ═══════════════════════════════════════════════════════
        # EARLY EXIT — Skip expensive indicators if gate is closed
        # ═══════════════════════════════════════════════════════
        if not buy_common and not sell_common:
            await _blanket_reset_pair(sdb, pair_name, logger_pair)
            reasons = []
            if not base_buy_trend and not base_sell_trend:
                reasons.append("base_trend=False")
            if not confirmation_buy and not confirmation_sell:
                reasons.append("cloud_align=False")
            if not volatility_filter_ok:
                reasons.append(
                    f"market_filter=False (adx={adx_val:.1f}, "
                    f"rvol_static={rvol_bypass_ok}, rvol_adaptive={adaptive_rvol_check})"
                )
            if not effective_cpr_ok:
                reasons.append("cpr=False")
            if not trend_gate_ok_buy and not trend_gate_ok_sell:
                reasons.append("trend_gate=False")
            logger_pair.debug(
                f"😒 {pair_name} | Gate blocked | "
                f"Suppression: {', '.join(reasons)}"
            )
            return pair_name, {
                "state": "NO_SIGNAL",
                "ts": int(time.time()),
                "summary": {
                    "alerts": 0,
                    "future_cloud": "green" if cloud_up else "red" if cloud_down else "neutral",
                    "hist_rma": 0.0,
                    "suppression": f"Gate blocked: {', '.join(reasons)}"
                }
            }
        return GateResult(
            pair_name=pair_name, i15=i15, i5=i5, ts_curr=ts_curr, reference_time=reference_time,
            candle_info=candle_info, o=o, h=h, l=l, c=c,
            open_curr=open_curr, high_curr=high_curr, low_curr=low_curr, close_curr=close_curr,
            close_prev=close_prev, close_5m_val=close_5m_val,
            is_green=is_green, is_red=is_red,
            is_valid_for_buy=is_valid_for_buy, is_valid_for_sell=is_valid_for_sell,
            candle_index=i15, min_wick_ratio=Constants.MIN_WICK_RATIO,
            buy_wick_ratio=buy_wick_ratio, sell_wick_ratio=sell_wick_ratio,
            gate_indicators=gate_indicators,
            base_buy_trend=base_buy_trend, base_sell_trend=base_sell_trend,
            rma50_15_val=rma50_15_val, rma200_5_val=rma200_5_val,
            cloud_up=cloud_up, cloud_down=cloud_down,
            cloud_upper_val=cloud_upper_val, cloud_lower_val=cloud_lower_val,
            cloud_upper_prev=cloud_upper_prev, cloud_lower_prev=cloud_lower_prev,
            ichimoku_gate_ok_buy=ichimoku_gate_ok_buy, ichimoku_gate_ok_sell=ichimoku_gate_ok_sell,
            confirmation_buy=confirmation_buy, confirmation_sell=confirmation_sell,
            cloud_group_ok_buy=cloud_group_ok_buy, cloud_group_ok_sell=cloud_group_ok_sell,
            tk_conversion_curr=tk_conversion_curr, tk_conversion_prev=tk_conversion_prev,
            tk_base_curr=tk_base_curr, tk_base_prev=tk_base_prev,
            tk_guard_ok_buy=tk_guard_ok_buy, tk_guard_ok_sell=tk_guard_ok_sell,
            fast_future_green=fast_future_green, fast_future_red=fast_future_red,
            fast_cloud_upper_curr=fast_cloud_upper_curr, fast_cloud_lower_curr=fast_cloud_lower_curr,
            fast_cloud_upper_prev=fast_cloud_upper_prev, fast_cloud_lower_prev=fast_cloud_lower_prev,
            fast_tk_conversion_curr=fast_tk_conversion_curr, fast_tk_conversion_prev=fast_tk_conversion_prev,
            fast_tk_base_curr=fast_tk_base_curr, fast_tk_base_prev=fast_tk_base_prev,
            fast_tenkan_ge_kijun=fast_tenkan_ge_kijun, fast_tenkan_le_kijun=fast_tenkan_le_kijun,
            oscillator_group_ok_buy=oscillator_group_ok_buy, oscillator_group_ok_sell=oscillator_group_ok_sell,
            ppo_gate_arr=ppo_gate_arr, ppo_gate_signal_arr=ppo_gate_signal_arr,
            ppo_gate_curr=ppo_gate_curr, ppo_gate_prev=ppo_gate_prev,
            ppo_gate_sig_curr=ppo_gate_sig_curr, ppo_gate_sig_prev=ppo_gate_sig_prev,
            ppo_gate_ok_buy=ppo_gate_ok_buy, ppo_gate_ok_sell=ppo_gate_ok_sell,
            rsi_guard_smooth_curr=rsi_guard_smooth_curr, rsi_guard_ema_curr=rsi_guard_ema_curr,
            rsi_guard_ok_buy=rsi_guard_ok_buy, rsi_guard_ok_sell=rsi_guard_ok_sell,
            rma_cloud_fast_curr=rma_cloud_fast_curr,
            rma_cloud_ok_buy=rma_cloud_ok_buy, rma_cloud_ok_sell=rma_cloud_ok_sell,
            trend_gate_ok_buy=trend_gate_ok_buy, trend_gate_ok_sell=trend_gate_ok_sell,
            adx_val=adx_val, adx_adaptive_threshold=adx_adaptive_threshold, adx_ok=adx_ok,
            rvol_bypass_ok=rvol_bypass_ok, rvol_ok=rvol_ok, adaptive_rvol_check=adaptive_rvol_check,
            momentum_count=momentum_count, volatility_filter_ok=volatility_filter_ok,
            cpr_ok=cpr_ok, nr_cpr=nr_cpr, effective_cpr_ok=effective_cpr_ok,
            cpr_adaptive_min_pct_move=cpr_adaptive_min_pct_move, move_from_prev_close_ok=move_from_prev_close_ok,
            ppo_adaptive_threshold=ppo_adaptive_threshold,
            rsi_adaptive_buy=rsi_adaptive_buy, rsi_adaptive_sell=rsi_adaptive_sell,
            buy_common=buy_common, sell_common=sell_common,
            data_15m=data_15m, close_prev_invalid=close_prev_invalid,
        )
    except asyncio.CancelledError:
        logger_pair.warning(f"Evaluation cancelled for {pair_name}")
        raise
    except RuntimeError as e:
        logger_pair.critical(f"🚨 INVARIANT VIOLATION in {pair_name}: {e}")
        return pair_name, {
            "state": "INVARIANT_VIOLATION",
            "ts": int(time.time()),
            "summary": {
                "alerts": 0,
                "future_cloud": "neutral",
                "hist_rma": 0.0,
                "error": str(e)
            }
        }
    except Exception as e:
        logger_pair.exception(
            f"❌ Error in _eval_gate for {pair_name}: {e} | Correlation: {correlation_id}"
        )
        return None

async def _eval_alerts(gr: GateResult, data_5m: PriceData, data_daily: Optional[Dict[str, np.ndarray]],
    reference_time: int, sdb: RedisStateStore, correlation_id: str, logger_pair: logging.Logger
) -> Union[Tuple[Dict[str, Any], Dict[str, bool], List[Tuple[str, str, str]]], Tuple[str, Dict[str, Any]], None]:
    pair_name = gr.pair_name
    i15 = gr.i15
    data_15m = gr.data_15m
    close_curr = gr.close_curr
    close_prev = gr.close_prev
    is_green, is_red = gr.is_green, gr.is_red
    is_valid_for_buy, is_valid_for_sell = gr.is_valid_for_buy, gr.is_valid_for_sell
    buy_wick_ratio, sell_wick_ratio = gr.buy_wick_ratio, gr.sell_wick_ratio
    rma50_15_val, rma200_5_val = gr.rma50_15_val, gr.rma200_5_val
    cloud_up, cloud_down = gr.cloud_up, gr.cloud_down
    cloud_upper_val, cloud_lower_val = gr.cloud_upper_val, gr.cloud_lower_val
    cloud_upper_prev, cloud_lower_prev = gr.cloud_upper_prev, gr.cloud_lower_prev
    ichimoku_gate_ok_buy, ichimoku_gate_ok_sell = gr.ichimoku_gate_ok_buy, gr.ichimoku_gate_ok_sell
    cloud_group_ok_buy, cloud_group_ok_sell = gr.cloud_group_ok_buy, gr.cloud_group_ok_sell
    tk_conversion_curr, tk_conversion_prev = gr.tk_conversion_curr, gr.tk_conversion_prev
    tk_base_curr, tk_base_prev = gr.tk_base_curr, gr.tk_base_prev
    tk_guard_ok_buy, tk_guard_ok_sell = gr.tk_guard_ok_buy, gr.tk_guard_ok_sell
    fast_future_green, fast_future_red = gr.fast_future_green, gr.fast_future_red
    fast_cloud_upper_curr, fast_cloud_lower_curr = gr.fast_cloud_upper_curr, gr.fast_cloud_lower_curr
    fast_cloud_upper_prev, fast_cloud_lower_prev = gr.fast_cloud_upper_prev, gr.fast_cloud_lower_prev
    fast_tk_conversion_curr, fast_tk_conversion_prev = gr.fast_tk_conversion_curr, gr.fast_tk_conversion_prev
    fast_tk_base_curr, fast_tk_base_prev = gr.fast_tk_base_curr, gr.fast_tk_base_prev
    fast_tenkan_ge_kijun, fast_tenkan_le_kijun = gr.fast_tenkan_ge_kijun, gr.fast_tenkan_le_kijun
    oscillator_group_ok_buy, oscillator_group_ok_sell = gr.oscillator_group_ok_buy, gr.oscillator_group_ok_sell
    ppo_gate_arr, ppo_gate_signal_arr = gr.ppo_gate_arr, gr.ppo_gate_signal_arr
    ppo_gate_curr, ppo_gate_prev = gr.ppo_gate_curr, gr.ppo_gate_prev
    ppo_gate_sig_curr, ppo_gate_sig_prev = gr.ppo_gate_sig_curr, gr.ppo_gate_sig_prev
    rsi_guard_smooth_curr, rsi_guard_ema_curr = gr.rsi_guard_smooth_curr, gr.rsi_guard_ema_curr
    rma_cloud_fast_curr = gr.rma_cloud_fast_curr
    rma_cloud_ok_buy, rma_cloud_ok_sell = gr.rma_cloud_ok_buy, gr.rma_cloud_ok_sell
    trend_gate_ok_buy, trend_gate_ok_sell = gr.trend_gate_ok_buy, gr.trend_gate_ok_sell
    adx_adaptive_threshold = gr.adx_adaptive_threshold
    momentum_count = gr.momentum_count
    effective_cpr_ok = gr.effective_cpr_ok
    cpr_adaptive_min_pct_move = gr.cpr_adaptive_min_pct_move
    move_from_prev_close_ok = gr.move_from_prev_close_ok
    ppo_adaptive_threshold = gr.ppo_adaptive_threshold
    rsi_adaptive_buy, rsi_adaptive_sell = gr.rsi_adaptive_buy, gr.rsi_adaptive_sell
    buy_common, sell_common = gr.buy_common, gr.sell_common
    close_prev_invalid = gr.close_prev_invalid

    try:
        alert_indicators = await asyncio.to_thread(
            calculate_alert_indicators_numpy, data_15m.as_dict(), data_5m.as_dict(), data_daily, reference_time
        )
        if alert_indicators is None:
            logger_pair.error(f"Skipping {pair_name}: alert indicators failed")
            return None

        indicators = IndicatorCache.from_dicts(gr.gate_indicators, alert_indicators)

        critical_indicators = ["ppo", "ppo_signal", "smooth_rsi", "smooth_rsi_ema"]
        is_valid, msg = validate_indicators_dict(indicators.as_dict(), critical_indicators)
        if not is_valid:
            logger_pair.warning(f"Skipping {pair_name}: {msg}")
            return None

        ppo = indicators.ppo
        ppo_signal = indicators.ppo_signal
        smooth_rsi = indicators.smooth_rsi
        smooth_rsi_ema = indicators.smooth_rsi_ema
        vwap = indicators.vwap
        hist_rma = indicators.hist_rma
        piv = indicators.pivots or {}

        ppo_sig_curr = ppo_signal[i15]
        ppo_sig_prev = ppo_signal[i15 - 1] if i15 >= 1 else ppo_signal[i15]
        ppo_curr = ppo[i15]
        ppo_prev = ppo[i15 - 1] if i15 >= 1 else ppo[i15]
        ppohist_curr = ppo_gate_curr - ppo_gate_sig_curr
        rsi_curr = smooth_rsi[i15]
        rsi_prev = smooth_rsi[i15 - 1] if i15 >= 1 else smooth_rsi[i15]
        rsi_ema_curr = smooth_rsi_ema[i15]
        rsi_ema_prev = smooth_rsi_ema[i15 - 1] if i15 >= 1 else smooth_rsi_ema[i15]

        vwap_enabled = cfg.ENABLE_VWAP
        vwap_available = False
        vwap_curr = None
        vwap_prev = None
        if vwap_enabled and not close_prev_invalid and vwap is not None and len(vwap) > i15:
            try:
                vwap_curr = vwap[i15]
                vwap_prev = vwap[i15 - 1] if i15 >= 1 else vwap[i15]
                if (not np.isnan(vwap_curr) and not np.isnan(vwap_prev)
                        and vwap_curr > 0 and vwap_prev > 0):
                    vwap_available = True
                    if cfg.DEBUG_MODE:
                        logger_pair.debug(
                            f"[{pair_name}] VWAP OK: curr={vwap_curr:.4f}, prev={vwap_prev:.4f}"
                        )
                else:
                    if cfg.DEBUG_MODE:
                        logger_pair.debug(
                            f"[{pair_name}] VWAP invalid: curr={vwap_curr}, prev={vwap_prev}"
                        )
                    vwap_curr = None
                    vwap_prev = None
            except (IndexError, TypeError) as e:
                logger_pair.warning(f"[{pair_name}] VWAP access error: {e}")
                vwap_curr = None
                vwap_prev = None
        else:
            if vwap_enabled and cfg.DEBUG_MODE:
                logger_pair.debug(
                    f"[{pair_name}] VWAP unavailable: enabled={vwap_enabled}, "
                    f"vwap_is_none={vwap is None}, "
                    f"len={len(vwap) if vwap is not None else 0}, i15={i15}"
                )

        hist_curr = hist_rma[i15]
        hist_m1 = hist_rma[i15 - 1] if i15 >= 1 else 0.0
        hist_m2 = hist_rma[i15 - 2] if i15 >= 2 else 0.0
        hist_m3 = hist_rma[i15 - 3] if i15 >= 3 else 0.0

        ppohist_m1 = (ppo_gate_arr[i15-1] - ppo_gate_signal_arr[i15-1]) if i15 >= 1 else 0.0
        ppohist_m2 = (ppo_gate_arr[i15-2] - ppo_gate_signal_arr[i15-2]) if i15 >= 2 else 0.0
        ppohist_m3 = (ppo_gate_arr[i15-3] - ppo_gate_signal_arr[i15-3]) if i15 >= 3 else 0.0

        MIN_HIST_RMA_BARS_VALID = cfg.HIST_RMA_SLOW * 3
        has_valid_hist_rma = (
            cfg.ENABLE_HIST_RMA and
            i15 >= MIN_HIST_RMA_BARS_VALID and
            not np.isnan(hist_curr) and not np.isnan(hist_m1) and
            not np.isnan(hist_m2) and not np.isnan(hist_m3)
        )

        if not has_valid_hist_rma and cfg.DEBUG_MODE and cfg.ENABLE_HIST_RMA:
            skip_reason = (
                f"Hist RMA warmup" if i15 < MIN_HIST_RMA_BARS_VALID
                else f"Hist RMA NaN (idx={i15})"
            )
            logger_pair.debug(f"Skipping Hist RMA alerts: {skip_reason}")

        if not has_valid_hist_rma:
            hist_reversal_buy = False
            hist_reversal_sell = False
        else:
            hist_reversal_buy = (
                buy_common and hist_curr > 0
                and hist_m3 > hist_m2 > hist_m1 and hist_curr > hist_m1
            )
            hist_reversal_sell = (
                sell_common and hist_curr < 0
                and hist_m3 < hist_m2 < hist_m1 and hist_curr < hist_m1
            )

        MIN_PPOHIST_BARS_VALID = 160
        has_valid_ppohist = (
            cfg.ENABLE_PPO_GATE and
            i15 >= MIN_PPOHIST_BARS_VALID and
            not np.isnan(ppohist_curr) and not np.isnan(ppohist_m1) and
            not np.isnan(ppohist_m2) and not np.isnan(ppohist_m3)
        )

        if not has_valid_ppohist:
            ppohist_reversal_buy = False
            ppohist_reversal_sell = False
        else:
            ppohist_reversal_buy = (
                buy_common and ppohist_curr > 0
                and ppohist_m3 > ppohist_m2 > ppohist_m1 and ppohist_curr > ppohist_m1
            )
            ppohist_reversal_sell = (
                sell_common and ppohist_curr < 0
                and ppohist_m3 < ppohist_m2 < ppohist_m1 and ppohist_curr < ppohist_m1
            )

        values_to_check = {
            'ppo_curr': ppo_curr, 'ppo_prev': ppo_prev,
            'rsi_curr': rsi_curr, 'rsi_prev': rsi_prev,
            'rsi_ema_curr': rsi_ema_curr, 'rsi_ema_prev': rsi_ema_prev,
            'ppo_sig_curr': ppo_sig_curr, 'ppo_sig_prev': ppo_sig_prev,
        }
        is_valid, msg = validate_indicator_values(values_to_check, list(values_to_check.keys()))
        if not is_valid:
            logger_pair.debug(msg)
            return None

        context = {
            "close_curr": close_curr, "close_prev": close_prev,
            "ppo_curr": ppo_curr, "ppo_prev": ppo_prev,
            "ppo_sig_curr": ppo_sig_curr, "ppo_sig_prev": ppo_sig_prev,
            "rsi_curr": rsi_curr, "rsi_prev": rsi_prev,
            "rsi_ema_curr": rsi_ema_curr, "rsi_ema_prev": rsi_ema_prev,
            "vwap_curr": vwap_curr, "vwap_prev": vwap_prev,
            "hist_curr": hist_curr, "hist_m1": hist_m1, "hist_m2": hist_m2, "hist_m3": hist_m3,
            "hist_reversal_buy": hist_reversal_buy, "hist_reversal_sell": hist_reversal_sell,
            "rma50_15_val": rma50_15_val, "rma200_5_val": rma200_5_val,
            "ppo_gate_curr": ppo_gate_curr, "ppo_gate_prev": ppo_gate_prev,
            "ppo_gate_sig_curr": ppo_gate_sig_curr, "ppo_gate_sig_prev": ppo_gate_sig_prev,
            "rsi_guard_smooth_curr": rsi_guard_smooth_curr, "rsi_guard_ema_curr": rsi_guard_ema_curr,
            "trend_gate_ok_buy": trend_gate_ok_buy, "trend_gate_ok_sell": trend_gate_ok_sell,
            "cloud_up": cloud_up, "cloud_down": cloud_down,
            "cloud_upper_curr": cloud_upper_val, "cloud_upper_prev": cloud_upper_prev,
            "cloud_lower_curr": cloud_lower_val, "cloud_lower_prev": cloud_lower_prev,
            "tk_guard_ok_buy": tk_guard_ok_buy, "tk_guard_ok_sell": tk_guard_ok_sell,
            "tk_conversion_curr": tk_conversion_curr, "tk_conversion_prev": tk_conversion_prev, "tk_base_curr": tk_base_curr, "tk_base_prev": tk_base_prev,
            "fast_cloud_upper_curr": fast_cloud_upper_curr, "fast_cloud_upper_prev": fast_cloud_upper_prev,
            "fast_cloud_lower_curr": fast_cloud_lower_curr, "fast_cloud_lower_prev": fast_cloud_lower_prev,
            "fast_tk_conversion_curr": fast_tk_conversion_curr, "fast_tk_conversion_prev": fast_tk_conversion_prev,
            "fast_tk_base_curr": fast_tk_base_curr, "fast_tk_base_prev": fast_tk_base_prev,
            "fast_future_green": bool(fast_future_green), "fast_future_red": bool(fast_future_red),
            "fast_tenkan_ge_kijun": fast_tenkan_ge_kijun, "fast_tenkan_le_kijun": fast_tenkan_le_kijun,
            "rma_cloud_ok_buy": rma_cloud_ok_buy, "rma_cloud_ok_sell": rma_cloud_ok_sell,
            "rma_cloud_fast_curr": rma_cloud_fast_curr, "rma_cloud_slow_curr": rma50_15_val,
            "ichimoku_gate_ok_buy": ichimoku_gate_ok_buy, "ichimoku_gate_ok_sell": ichimoku_gate_ok_sell, 
            "cloud_group_ok_buy": cloud_group_ok_buy, "cloud_group_ok_sell": cloud_group_ok_sell,
            "oscillator_group_ok_buy": oscillator_group_ok_buy, "oscillator_group_ok_sell": oscillator_group_ok_sell,
            "buy_common": buy_common, "sell_common": sell_common,
            "vwap_available": vwap_available,
            "vwap_enabled": cfg.ENABLE_VWAP and vwap_available,
            "ppohist_curr": ppohist_curr, "ppohist_m1": ppohist_m1,
            "ppohist_m2": ppohist_m2, "ppohist_m3": ppohist_m3,
            "ppohist_reversal_buy": ppohist_reversal_buy, "ppohist_reversal_sell": ppohist_reversal_sell,
            "adx_adaptive_threshold": adx_adaptive_threshold,
            "ppo_adaptive_threshold": ppo_adaptive_threshold,
            "rsi_adaptive_buy": rsi_adaptive_buy,
            "rsi_adaptive_sell": rsi_adaptive_sell,
            "buy_wick_ratio": buy_wick_ratio,
            "sell_wick_ratio": sell_wick_ratio,
            "is_green": is_green, "is_red": is_red,
            "pivots": piv if piv else {},
            "pivot_suppressions": [],
            "nr_cpr": indicators.nr_cpr,
            "cpr_ok": effective_cpr_ok,
            "momentum_count": momentum_count,
            "move_from_prev_close_ok": move_from_prev_close_ok, 
            "cpr_adaptive_min_pct_move": cpr_adaptive_min_pct_move,
        }

        ppo_ctx = {"curr": ppo_curr, "prev": ppo_prev}
        ppo_sig_ctx = {"curr": ppo_sig_curr, "prev": ppo_sig_prev}
        rsi_ctx = {"curr": rsi_curr, "prev": rsi_prev, "ema_curr": rsi_ema_curr, "ema_prev": rsi_ema_prev}

        alert_keys_to_check = []
        for d in ALERT_DEFINITIONS:
            key = d["key"]
            requires = d.get("requires", [])
            
            skip = False
            if "pivots" in requires and (not cfg.ENABLE_PIVOT or not piv or not any(piv.values())):
                skip = True
            elif "vwap" in requires and not vwap_available:
                skip = True
            elif "ppo" in requires and ppo_ctx is None:
                skip = True
            elif "ppo_signal" in requires and ppo_sig_ctx is None:
                skip = True
            elif "rsi" in requires and rsi_ctx is None:
                skip = True
            
            if not skip:
                alert_keys_to_check.append(key)

        all_redis_alert_keys = list(ALERT_KEYS.values())
        previous_states = await sdb.batch_get_all_alert_states(
            pair_name, all_redis_alert_keys
        )

        raw_alerts: List[Tuple[str, str, str]] = []

        # ── Registry for cross-based alerts (same pattern as _build_resets) ──
        _CROSS_HANDLERS = {
            "vwap": {
                "keys": {"vwap_up", "vwap_down"},
                "enabled": vwap_available,
                "validator": validate_vwap_cross,
                "ctx_args": ("close_prev", "close_curr", "vwap_prev", "vwap_curr"),
            },
            "cloud_cross": {
                "keys": {"cloud_cross_up", "cloud_cross_down"},
                "enabled": cfg.ENABLE_CLOUD_CROSS_ALERT,
                "validator": validate_cloud_cross,
                "ctx_args": ("close_prev", "close_curr", "cloud_upper_prev", "cloud_upper_curr",
                             "cloud_lower_prev", "cloud_lower_curr"),
            },
            "tk_conversion": {
                "keys": {"tk_conversion_up", "tk_conversion_down"},
                "enabled": cfg.ENABLE_TK_CONVERSION_CROSS,
                "validator": validate_conversion_cross,
                "ctx_args": ("close_prev", "close_curr", "tk_conversion_prev", "tk_conversion_curr"),
            },
            "kijun_cross": {
                "keys": {"kijun_cross_up", "kijun_cross_down"},
                "enabled": cfg.ENABLE_KIJUN_CROSS,
                "validator": validate_conversion_cross,
                "ctx_args": ("close_prev", "close_curr", "tk_base_prev", "tk_base_curr"),
            },
            "fast_cloud_cross": {
                "keys": {"fast_cloud_cross_up", "fast_cloud_cross_down"},
                "enabled": cfg.ENABLE_FAST_ICHIMOKU_CLOUD_CROSS,
                "validator": validate_cloud_cross,
                "ctx_args": ("close_prev", "close_curr", "fast_cloud_upper_prev", "fast_cloud_upper_curr",
                             "fast_cloud_lower_prev", "fast_cloud_lower_curr"),
            },
            "fast_tenkan_cross": {
                "keys": {"fast_tenkan_cross_up", "fast_tenkan_cross_down"},
                "enabled": cfg.ENABLE_FAST_ICHIMOKU_TENKAN_CROSS,
                "validator": validate_conversion_cross,
                "ctx_args": ("close_prev", "close_curr", "fast_tk_conversion_prev", "fast_tk_conversion_curr"),
            },
        }

        for alert_key in alert_keys_to_check:
            def_ = ALERT_DEFINITIONS_MAP.get(alert_key)
            if not def_:
                continue

            if alert_key in BUY_ALERT_KEYS:
                if not is_green:
                    logger_pair.debug(
                        f"[{pair_name}] 🚫 BLOCKED BUY: {alert_key} on RED candle! "
                        f"O={gr.open_curr:.2f} C={close_curr:.2f}"
                    )
                    continue
                if not is_valid_for_buy:
                    if cfg.DEBUG_MODE:
                        logger_pair.debug(f"Skipping {alert_key}: not valid for buy")
                    continue
        
            if alert_key in SELL_ALERT_KEYS:
                if not is_red:
                    logger_pair.debug(
                        f"[{pair_name}] 🚫 BLOCKED SELL: {alert_key} on GREEN candle! "
                        f"O={gr.open_curr:.2f} C={close_curr:.2f}"
                    )
                    continue
                if not is_valid_for_sell:
                    if cfg.DEBUG_MODE:
                        logger_pair.debug(f"Skipping {alert_key}: not valid for sell")
                    continue

            if is_green and alert_key.startswith("pivot_down"):
                logger_pair.debug(
                    f"[{pair_name}] LOGIC ERROR: GREEN candle firing pivot_down '{alert_key}'. "
                    f"Skipping to prevent false alert."
                )
                continue

            if is_red and alert_key.startswith("pivot_up"):
                logger_pair.debug(
                    f"[{pair_name}] LOGIC ERROR: RED candle firing pivot_up '{alert_key}'. "
                    f"Skipping to prevent false alert."
                )
                continue

            key = ALERT_KEYS[alert_key]
            trigger = False

            # ── Cross-alert dispatch ──
            handled = False
            for handler in _CROSS_HANDLERS.values():
                if alert_key not in handler["keys"]:
                    continue
                if not handler["enabled"]:
                    if cfg.DEBUG_MODE:
                        logger_pair.debug(f"Skipping {alert_key}: cross prerequisite disabled")
                    handled = True
                    break
                try:
                    is_buy_side = alert_key.endswith("_up")
                    args = [context[k] for k in handler["ctx_args"]] + [is_buy_side]
                    valid_cross, cross_reason = handler["validator"](*args)
                    if valid_cross:
                        trigger = def_["check_fn"](context, ppo_ctx, ppo_sig_ctx, rsi_ctx)
                    elif cfg.DEBUG_MODE:
                        logger_pair.debug(f"{alert_key} cross check: {cross_reason}")
                except Exception as e:
                    logger_pair.debug(f"{alert_key} cross check failed: {e}", exc_info=True)
                handled = True
                break

            if not handled:
                if alert_key.startswith("pivot_up_") or alert_key.startswith("pivot_down_"):
                    level = alert_key.split("_")[-1]
                    is_buy = alert_key.startswith("pivot_up_")
                    try:
                        valid_cross, reason = get_pivot_alert_info(context, level, is_buy)
                        if not valid_cross and reason and piv:
                             context["pivot_suppressions"].append(f"{alert_key}: {reason}")
                        trigger = def_["check_fn"](context, ppo_ctx, ppo_sig_ctx, rsi_ctx)
                    except Exception as e:
                        logger_pair.debug(f"Pivot alert check failed for {alert_key}: {e}", exc_info=True)
                        trigger = False
                else:
                    try:
                        trigger = def_["check_fn"](context, ppo_ctx, ppo_sig_ctx, rsi_ctx)
                    except Exception as e:
                        logger_pair.debug(f"Alert check failed for {alert_key}: {e}", exc_info=True)
                        trigger = False

            if trigger and not previous_states.get(key, False):
                extra = ""
                try:
                    base_extra = def_["extra_fn"](context, ppo_ctx, ppo_sig_ctx, rsi_ctx, None) or ""
                    extra = base_extra
                except Exception as e:
                    logger_pair.debug(f"Alert extra_fn failed for {alert_key}: {e}", exc_info=cfg.DEBUG_MODE)
                    extra = f"(Error: {str(e)[:100]})"

                raw_alerts.append((def_["title"], extra, def_["key"]))
            
                if cfg.DEBUG_MODE:
                    logger_pair.debug(
                        f"✅ Alert FIRED: {alert_key} | "
                        f"buy_common={buy_common} sell_common={sell_common} | "
                        f"Candle: O={gr.open_curr:.2f} C={close_curr:.2f}"
                    )

        conditional_states = previous_states

        return context, conditional_states, raw_alerts

    except asyncio.CancelledError:
        logger_pair.warning(f"Evaluation cancelled for {pair_name}")
        raise
    except RuntimeError as e:
        logger_pair.critical(f"🚨 INVARIANT VIOLATION in {pair_name}: {e}")
        return pair_name, {
            "state": "INVARIANT_VIOLATION",
            "ts": int(time.time()),
            "summary": {
                "alerts": 0,
                "future_cloud": "neutral",
                "hist_rma": 0.0,
                "error": str(e)
            }
        }
    except Exception as e:
        logger_pair.exception(
            f"❌ Error in _eval_alerts for {pair_name}: {e} | Correlation: {correlation_id}"
        )
        return None

async def _apply_and_dispatch_alerts(gr: GateResult, context: Dict[str, Any], conditional_states: Dict[str, bool],
    raw_alerts: List[Tuple[str, str, str]], sdb: RedisStateStore, telegram_queue: TelegramQueue,
    fetcher: DataFetcher, symbol: str, correlation_id: str, logger_pair: logging.Logger,
    alerts_sent_ref: List[int], alerts_sent_lock: asyncio.Lock, max_alerts_per_run: int
) -> Tuple[str, Dict[str, Any]]:
   
    pair_name = gr.pair_name
    i15, ts_curr, reference_time = gr.i15, gr.ts_curr, gr.reference_time
    data_15m = gr.data_15m
    candle_info = gr.candle_info
    o, h, l, c = gr.o, gr.h, gr.l, gr.c
    close_curr, close_prev = gr.close_curr, gr.close_prev
    is_green, is_red = gr.is_green, gr.is_red
    is_valid_for_buy, is_valid_for_sell = gr.is_valid_for_buy, gr.is_valid_for_sell
    base_buy_trend, base_sell_trend = gr.base_buy_trend, gr.base_sell_trend
    rma50_15_val = gr.rma50_15_val
    cloud_up, cloud_down = gr.cloud_up, gr.cloud_down
    ichimoku_gate_ok_buy, ichimoku_gate_ok_sell = gr.ichimoku_gate_ok_buy, gr.ichimoku_gate_ok_sell
    confirmation_buy, confirmation_sell = gr.confirmation_buy, gr.confirmation_sell
    cloud_group_ok_buy, cloud_group_ok_sell = gr.cloud_group_ok_buy, gr.cloud_group_ok_sell
    tk_conversion_curr, tk_conversion_prev = gr.tk_conversion_curr, gr.tk_conversion_prev
    tk_base_curr, tk_base_prev = gr.tk_base_curr, gr.tk_base_prev
    oscillator_group_ok_buy, oscillator_group_ok_sell = gr.oscillator_group_ok_buy, gr.oscillator_group_ok_sell
    ppo_gate_curr, ppo_gate_sig_curr = gr.ppo_gate_curr, gr.ppo_gate_sig_curr
    ppo_gate_ok_buy, ppo_gate_ok_sell = gr.ppo_gate_ok_buy, gr.ppo_gate_ok_sell
    rsi_guard_smooth_curr, rsi_guard_ema_curr = gr.rsi_guard_smooth_curr, gr.rsi_guard_ema_curr
    rsi_guard_ok_buy, rsi_guard_ok_sell = gr.rsi_guard_ok_buy, gr.rsi_guard_ok_sell
    rma_cloud_fast_curr = gr.rma_cloud_fast_curr
    rma_cloud_ok_buy, rma_cloud_ok_sell = gr.rma_cloud_ok_buy, gr.rma_cloud_ok_sell
    adx_val, adx_adaptive_threshold, adx_ok = gr.adx_val, gr.adx_adaptive_threshold, gr.adx_ok
    rvol_ok = gr.rvol_ok
    ppo_adaptive_threshold = gr.ppo_adaptive_threshold
    rsi_adaptive_buy, rsi_adaptive_sell = gr.rsi_adaptive_buy, gr.rsi_adaptive_sell
    buy_common, sell_common = gr.buy_common, gr.sell_common

    hist_curr, hist_m1, hist_m2, hist_m3 = context["hist_curr"], context["hist_m1"], context["hist_m2"], context["hist_m3"]
    hist_reversal_buy, hist_reversal_sell = context["hist_reversal_buy"], context["hist_reversal_sell"]
    vwap_curr, vwap_prev, vwap_available = context["vwap_curr"], context["vwap_prev"], context["vwap_available"]
    ppo_curr, ppo_prev = context["ppo_curr"], context["ppo_prev"]
    rsi_curr, rsi_prev = context["rsi_curr"], context["rsi_prev"]
    rsi_ema_curr, rsi_ema_prev = context["rsi_ema_curr"], context["rsi_ema_prev"]

    all_state_changes = []

    try:
        resets_to_apply = _build_resets(pair_name, context, conditional_states)

        all_state_changes.extend(resets_to_apply)

        pivot_count = sum(1 for _, _, k in raw_alerts if k.startswith("pivot_"))
        if pivot_count > 3:
            logger_pair.warning(
                f"Limiting pivot alerts for {pair_name}: {pivot_count} triggered, keeping 3"
            )
            pivot_alerts = [(t, e, k) for t, e, k in raw_alerts if k.startswith("pivot_")][:3]
            other_alerts = [(t, e, k) for t, e, k in raw_alerts if not k.startswith("pivot_")]
            capped_alerts = other_alerts + pivot_alerts
        else:
            capped_alerts = raw_alerts

        alerts_to_send = capped_alerts[:cfg.MAX_ALERTS_PER_PAIR]

        cached_snapshot: Optional[CandleSnapshot] = None
        if alerts_to_send:
            cached_snapshot = CandleSnapshot(
                timestamp=ts_curr, open=o, high=h, low=l, close=c,
                volume=candle_info["volume"],
                is_green=is_green, is_red=is_red,
                is_valid_for_buy=is_valid_for_buy, is_valid_for_sell=is_valid_for_sell,
            )
            reverified = independent_candle_reverify(
                data_15m=data_15m.as_dict(), candle_index=i15,
                cached=cached_snapshot,
                min_wick_ratio=Constants.MIN_WICK_RATIO,
                pair_name=pair_name, logger_pair=logger_pair,
            )
            if not reverified:
                logger_pair.warning(
                    f"[{pair_name}] Independent re-verify failed — alert suppressed. No dedup/coalesce "
                    f"claim was taken yet, so this will be re-attempted next run if the trigger persists."
                )
                alerts_to_send = []

        if alerts_to_send and cfg.ENABLE_CONFLUENCE_GATE:
            is_buy_dir = bool(gr.is_green and gr.is_valid_for_buy)
            score, total = compute_confluence_score(gr, is_buy=is_buy_dir)
            required = min(cfg.CONFLUENCE_MIN_VOTES, total)
            if score < required:
                logger_pair.info(
                    f"[{pair_name}] Confluence gate blocked dispatch: {score}/{total} votes (need {required})"
                )
                alerts_to_send = []

        coalesced_dedup_key: Optional[str] = None
        if alerts_to_send and cfg.ENABLE_ALERT_COALESCING:
            direction = "BUY" if (gr.is_green and gr.is_valid_for_buy) else "SELL"
            coalesced_dedup_key = f"coalesced_{direction}"
            should_send = await sdb.check_recent_alert(
                pair_name, coalesced_dedup_key, ts_curr, window_sec=cfg.COALESCE_DEDUP_WINDOW_SEC
            )
            if not should_send:
                logger_pair.debug(
                    f"[{pair_name}] Coalesced {direction} alert deduped (within "
                    f"{cfg.COALESCE_DEDUP_WINDOW_SEC}s) — skipping dispatch"
                )
                alerts_to_send = []
        elif alerts_to_send:
            deduped_alerts = []
            for alert_title, alert_extra, alert_key in alerts_to_send:
                should_send = await sdb.check_recent_alert(pair_name, alert_key, ts_curr)
                if not should_send:
                    logger_pair.debug(f"Alert {alert_key} skipped (dedup window)")
                    continue
                deduped_alerts.append((alert_title, alert_extra, alert_key))
            alerts_to_send = deduped_alerts

        async def _release_dedup_claims() -> None:
            """Releases whichever kind of claim was taken in step 4 above."""
            if coalesced_dedup_key:
                await sdb.release_recent_alert(pair_name, coalesced_dedup_key)
            else:
                for _, _, alert_key in alerts_to_send:
                    await sdb.release_recent_alert(pair_name, alert_key)

        new_alert_activations = []
        for _, _, alert_key in alerts_to_send:
            new_alert_activations.append(
                (f"{pair_name}:{ALERT_KEYS[alert_key]}", "ACTIVE", None)
            )
        async def _refund_alert_budget(n: int) -> None:
            """Undo the optimistic budget reservation when a send does not go out."""
            if n > 0 and alerts_sent_ref is not None and alerts_sent_lock is not None:
                async with alerts_sent_lock:
                    alerts_sent_ref[0] = max(0, alerts_sent_ref[0] - n)

        limit_reached = False

        if alerts_to_send and alerts_sent_ref is not None and alerts_sent_lock is not None:
            async with alerts_sent_lock:
                current_total = alerts_sent_ref[0]
                if current_total >= max_alerts_per_run:
                    limit_reached = True
                else:
                    alerts_sent_ref[0] += len(alerts_to_send)

            if limit_reached:
                logger_pair.warning(
                    f"Global alert limit reached ({current_total}/{max_alerts_per_run}), "
                    f"skipping {len(alerts_to_send)} alerts for {pair_name}"
                )
                if all_state_changes:
                    persist_ok = await sdb.atomic_batch_update(all_state_changes)
                    if not persist_ok:
                        logger_pair.error(
                            f"[{pair_name}] State persistence failed — alert state may be inconsistent this run"
                        )
                await _release_dedup_claims()
                return pair_name, {
                    "state": "LIMIT_REACHED",
                    "ts": int(time.time()),
                    "summary": {
                        "alerts": 0,
                        "future_cloud": "green" if cloud_up else "red" if cloud_down else "neutral",
                        "hist_rma": round(hist_curr, 4),
                        "suppression": f"Global limit {max_alerts_per_run} reached"
                    }
                }
        if alerts_to_send:
            try:
                if len(alerts_to_send) == 1:
                    title, extra, _ = alerts_to_send[0]
                    msg = build_single_msg(title, pair_name, close_curr, ts_curr, extra)
                else:
                    items = [(t, e) for t, e, _ in alerts_to_send[:25]]
                    msg = build_batched_msg(pair_name, close_curr, ts_curr, items)

                if not cfg.DRY_RUN_MODE:
                    reconfirmed = await confirm_candle_unchanged(
                        fetcher, symbol, pair_name, ts_curr, cached_snapshot, reference_time, logger_pair
                    )
                    if reconfirmed is None:
                        logger_pair.warning(
                            f"[{pair_name}] Confirmation inconclusive — alert suppressed this run, "
                            f"dedup key RELEASED so it can retry next run"
                        )
                    await _release_dedup_claims()
                    await _refund_alert_budget(len(alerts_to_send))
                        send_success = False
                    elif reconfirmed is False:           
                        logger_pair.warning(
                            f"[{pair_name}] 🔁 Confirmed repaint in send-queue window — "
                            f"alert suppressed, dedup key KEPT to prevent duplicates"
                        )
                        await _refund_alert_budget(len(alerts_to_send))
                        send_success = False
                    else:
                        send_success = await telegram_queue.send(msg)

                    if send_success:
                        all_state_changes.extend(new_alert_activations)
                        logger_pair.info(
                            f"🔔🎯🟢 Sent {len(alerts_to_send)} alerts for {pair_name} | "
                            f"Keys: {[ak for _, _, ak in alerts_to_send]}"
                        )
                    else:
                        await _refund_alert_budget(len(alerts_to_send))
                        logger_pair.error(
                            f"Alert dispatch failed | {pair_name} | "
                            f"State NOT marked ACTIVE, dedup claim retained for retry next run | "
                            f"Budget refunded"
                        )               
                else:
                    # DRY RUN: mark ACTIVE anyway so this run mirrors production dedup/reset behavior
                    all_state_changes.extend(new_alert_activations)
                    logger_pair.info(f"[DRY RUN] Would send: {msg[:100]}...")

            except Exception as e:
                await _refund_alert_budget(len(alerts_to_send))
                logger_pair.error(
                    f"Alert dispatch exception for {pair_name}: {e} | "
                    f"State NOT marked ACTIVE, dedup key retained, budget refunded — "
                    f"will not retry until window expires"
                )

        if all_state_changes:
            await sdb.atomic_batch_update(all_state_changes)

        failed_conditions = [
            name for name, val in [
                ("buy_common", buy_common),
                ("sell_common", sell_common),
            ] if not val
        ]

        reasons = []
        if not alerts_to_send:
            if not buy_common and not sell_common:
                reasons.append("Trend filter blocked")
            
            if context.get("pivot_suppressions"):
                reasons.extend(context["pivot_suppressions"])

            if ppo_prev <= 0 and ppo_curr > 0 and not buy_common:
                if not base_buy_trend:
                    reasons.append("PPO>0 blocked: base_buy_trend=False")
                elif not confirmation_buy:
                    reasons.append("PPO>0 blocked: confirmation_buy=False (future cloud)")
                elif not is_valid_for_buy:
                    reasons.append("PPO>0 blocked: Knox rejected candle (wick/color/timing)")
                else:
                    reasons.append(
                        f"PPO>0 blocked: market filter "
                        f"(adx_ok={adx_ok} [{adx_val:.1f} vs {adx_adaptive_threshold:.1f}], "
                        f"rvol_ok={rvol_ok})"
                    )           
        
            if ppo_prev >= 0 and ppo_curr < 0 and not sell_common:
                if not base_sell_trend:
                    reasons.append("PPO<0 blocked: base_sell_trend=False")
                elif not confirmation_sell:
                    reasons.append("PPO<0 blocked: confirmation_sell=False (future cloud)")
                elif not is_valid_for_sell:
                    reasons.append("PPO<0 blocked: Knox rejected candle (wick/color/timing)")
                else:
                    reasons.append(
                        f"PPO<0 blocked: market filter "
                        f"(adx_ok={adx_ok} [{adx_val:.1f} vs {adx_adaptive_threshold:.1f}], "
                        f"rvol_ok={rvol_ok})"
                    )
       
            if ppo_prev <= ppo_adaptive_threshold and ppo_curr > ppo_adaptive_threshold and not buy_common:
                if not base_buy_trend:
                    reasons.append("PPO>+adapt blocked: base_buy_trend=False")
                elif not confirmation_buy:
                    reasons.append("PPO>+adapt blocked: confirmation_buy=False (future cloud)")
                elif not is_valid_for_buy:
                    reasons.append("PPO>+adapt blocked: Knox rejected candle")
                else:
                    reasons.append(
                        f"PPO>+{ppo_adaptive_threshold:.3f} blocked: market filter "
                        f"(adx_ok={adx_ok} [{adx_val:.1f} vs {adx_adaptive_threshold:.1f}], "
                        f"rvol_ok={rvol_ok})"             
                    )
        
            if ppo_prev >= -ppo_adaptive_threshold and ppo_curr < -ppo_adaptive_threshold and not sell_common:
                if not base_sell_trend:
                    reasons.append("PPO<-adapt blocked: base_sell_trend=False")
                elif not confirmation_sell:
                    reasons.append("PPO<-adapt blocked: confirmation_sell=False (future cloud)")
                elif not is_valid_for_sell:
                    reasons.append("PPO<-adapt blocked: Knox rejected candle")
                else:
                    reasons.append(
                        f"PPO<-{ppo_adaptive_threshold:.3f} blocked: market filter "
                        f"(adx_ok={adx_ok} [{adx_val:.1f} vs {adx_adaptive_threshold:.1f}], "
                        f"rvol_ok={rvol_ok})"
                    )      

            if rsi_prev <= rsi_ema_prev and rsi_curr > rsi_ema_curr:
                if rsi_curr >= rsi_adaptive_buy:
                    reasons.append(f"RSI>EMA5 blocked: RSI={rsi_curr:.2f} ≥ cap {rsi_adaptive_buy:.1f}")
                elif ppo_gate_curr >= Constants.PPO_RSI_GUARD_BUY:
                    reasons.append(f"RSI>EMA5 blocked: PPO={ppo_gate_curr:.2f} ≥ guard {Constants.PPO_RSI_GUARD_BUY}")
                elif not buy_common:
                    if not base_buy_trend:
                        reasons.append("RSI>EMA5 blocked: base_buy_trend=False")
                    elif not confirmation_buy:
                        reasons.append("RSI>EMA5 blocked: confirmation_buy=False (future cloud)")
                    elif not is_valid_for_buy:
                        reasons.append("RSI>EMA5 blocked: Knox rejected candle")
                    else:
                        reasons.append(
                            f"RSI>EMA5 blocked: market filter "
                            f"(adx_ok={adx_ok} [{adx_val:.1f} vs {adx_adaptive_threshold:.1f}], "
                            f"rvol_ok={rvol_ok})"
                        )

            if rsi_prev >= rsi_ema_prev and rsi_curr < rsi_ema_curr:
                if rsi_curr <= rsi_adaptive_sell:
                    reasons.append(f"RSI<EMA5 blocked: RSI={rsi_curr:.2f} ≤ cap {rsi_adaptive_sell:.1f}")
                elif ppo_gate_curr <= Constants.PPO_RSI_GUARD_SELL:
                    reasons.append(f"RSI<EMA5 blocked: PPO={ppo_gate_curr:.2f} ≤ guard {Constants.PPO_RSI_GUARD_SELL}")
                elif not sell_common:
                    if not base_sell_trend:
                        reasons.append("RSI<EMA5 blocked: base_sell_trend=False")
                    elif not confirmation_sell:
                        reasons.append("RSI<EMA5 blocked: confirmation_sell=False (future cloud)")
                    elif not is_valid_for_sell:
                        reasons.append("RSI<EMA5 blocked: Knox rejected candle")
                    else:
                        reasons.append(
                            f"RSI<EMA5 blocked: market filter "
                            f"(adx_ok={adx_ok} [{adx_val:.1f} vs {adx_adaptive_threshold:.1f}], "
                            f"rvol_ok={rvol_ok})"
                        )
       
            if cfg.ENABLE_VWAP and vwap_available:
                if close_prev <= vwap_prev and close_curr > vwap_curr and not buy_common:
                    if not base_buy_trend:
                        reasons.append("VWAP up-cross blocked: base_buy_trend=False")
                    elif not confirmation_buy:
                        reasons.append("VWAP up-cross blocked: confirmation_buy=False")
                    elif not is_valid_for_buy:
                        reasons.append("VWAP up-cross blocked: Knox rejected candle")
                    else:
                        reasons.append(
                            f"VWAP up-cross blocked: market filter "
                            f"(adx_ok={adx_ok} [{adx_val:.1f} vs {adx_adaptive_threshold:.1f}], "
                            f"rvol_ok={rvol_ok})"
                        )
            
                if close_prev >= vwap_prev and close_curr < vwap_curr and not sell_common:
                    if not base_sell_trend:
                        reasons.append("VWAP down-cross blocked: base_sell_trend=False")
                    elif not confirmation_sell:
                        reasons.append("VWAP down-cross blocked: confirmation_sell=False")
                    elif not is_valid_for_sell:
                        reasons.append("VWAP down-cross blocked: Knox rejected candle")
                    else:
                        reasons.append(
                            f"VWAP down-cross blocked: market filter "
                            f"(adx_ok={adx_ok} [{adx_val:.1f} vs {adx_adaptive_threshold:.1f}], "
                            f"rvol_ok={rvol_ok})"
                        )

            if cfg.ENABLE_PPO_GATE:
                if not ppo_gate_ok_buy:
                    reasons.append(f"PPO Gate buy: Gate({ppo_gate_curr:.2f}) <= Signal({ppo_gate_sig_curr:.2f})")
                if not ppo_gate_ok_sell:
                    reasons.append(f"PPO Gate sell: Gate({ppo_gate_curr:.2f}) >= Signal({ppo_gate_sig_curr:.2f})")

            if cfg.ENABLE_TK_CONVERSION_CROSS:
                if close_prev <= tk_conversion_prev and close_curr > tk_conversion_curr and not buy_common:
                    if not base_buy_trend:
                        reasons.append("Conversion up-cross blocked: base_buy_trend=False")
                    elif not confirmation_buy:
                        reasons.append("Conversion up-cross blocked: confirmation_buy=False")
                    elif not is_valid_for_buy:
                        reasons.append("Conversion up-cross blocked: Knox rejected candle")
                    else:
                        reasons.append(
                            f"Conversion up-cross blocked: market filter "
                            f"(adx_ok={adx_ok} [{adx_val:.1f} vs {adx_adaptive_threshold:.1f}], "
                            f"rvol_ok={rvol_ok})"
                        )

                if close_prev >= tk_conversion_prev and close_curr < tk_conversion_curr and not sell_common:
                    if not base_sell_trend:
                        reasons.append("Conversion down-cross blocked: base_sell_trend=False")
                    elif not confirmation_sell:
                        reasons.append("Conversion down-cross blocked: confirmation_sell=False")
                    elif not is_valid_for_sell:
                        reasons.append("Conversion down-cross blocked: Knox rejected candle")
                    else:
                        reasons.append(
                            f"Conversion down-cross blocked: market filter "
                            f"(adx_ok={adx_ok} [{adx_val:.1f} vs {adx_adaptive_threshold:.1f}], "
                            f"rvol_ok={rvol_ok})"
                        )

            if cfg.ENABLE_KIJUN_CROSS:
                if close_prev <= tk_base_prev and close_curr > tk_base_curr and not buy_common:
                    if not base_buy_trend:
                        reasons.append("Kijun up-cross blocked: base_buy_trend=False")
                    elif not confirmation_buy:
                        reasons.append("Kijun up-cross blocked: confirmation_buy=False")
                    elif not is_valid_for_buy:
                        reasons.append("Kijun up-cross blocked: Knox rejected candle")
                    else:
                        reasons.append(
                            f"Kijun up-cross blocked: market filter "
                            f"(adx_ok={adx_ok} [{adx_val:.1f} vs {adx_adaptive_threshold:.1f}], "
                            f"rvol_ok={rvol_ok})"
                        )

                if close_prev >= tk_base_prev and close_curr < tk_base_curr and not sell_common:
                    if not base_sell_trend:
                        reasons.append("Kijun down-cross blocked: base_sell_trend=False")
                    elif not confirmation_sell:
                        reasons.append("Kijun down-cross blocked: confirmation_sell=False")
                    elif not is_valid_for_sell:
                        reasons.append("Kijun down-cross blocked: Knox rejected candle")
                    else:
                        reasons.append(
                            f"Kijun down-cross blocked: market filter "
                            f"(adx_ok={adx_ok} [{adx_val:.1f} vs {adx_adaptive_threshold:.1f}], "
                            f"rvol_ok={rvol_ok})"
                        )

            if cfg.ENABLE_HIST_RMA:
                if buy_common and not hist_reversal_buy:
                    if np.isnan(hist_curr):
                        reasons.append("Hist RMA buy: NaN")
                    elif hist_curr <= 0:
                        reasons.append(f"Hist RMA buy: hist_curr={hist_curr:.2f} <= 0")
                    elif not (hist_m3 > hist_m2 > hist_m1):
                        reasons.append(f"Hist RMA buy: sequence not rising ({hist_m3:.2f} > {hist_m2:.2f} > {hist_m1:.2f})")
                    elif not (hist_curr > hist_m1):
                        reasons.append(f"Hist RMA buy: no acceleration ({hist_curr:.2f} <= {hist_m1:.2f})")
                if sell_common and not hist_reversal_sell:
                    if np.isnan(hist_curr):
                        reasons.append("Hist RMA sell: NaN")
                    elif hist_curr >= 0:
                        reasons.append(f"Hist RMA sell: hist_curr={hist_curr:.2f} >= 0")
                    elif not (hist_m3 < hist_m2 < hist_m1):
                        reasons.append(f"Hist RMA sell: sequence not falling ({hist_m3:.2f} < {hist_m2:.2f} < {hist_m1:.2f})")
                    elif not (hist_curr < hist_m1):
                        reasons.append(f"Hist RMA sell: no acceleration ({hist_curr:.2f} >= {hist_m1:.2f})")

            if cfg.RSI_GUARD_ENABLED:
                if not rsi_guard_ok_buy:
                    reasons.append(f"RSI Guard buy: RSI({rsi_guard_smooth_curr:.2f}) <= EMA({rsi_guard_ema_curr:.2f})")
                if not rsi_guard_ok_sell:
                    reasons.append(f"RSI Guard sell: RSI({rsi_guard_smooth_curr:.2f}) >= EMA({rsi_guard_ema_curr:.2f})")

            if cfg.RMA_CLOUD_ENABLED:
                if not rma_cloud_ok_buy:
                    reasons.append(f"RMA Cloud buy: RMA{cfg.RMA_CLOUD_FAST_PERIOD}({rma_cloud_fast_curr:.2f}) <= RMA{cfg.RMA_50_PERIOD}({rma50_15_val:.2f})")
                if not rma_cloud_ok_sell:
                    reasons.append(f"RMA Cloud sell: RMA{cfg.RMA_CLOUD_FAST_PERIOD}({rma_cloud_fast_curr:.2f}) >= RMA{cfg.RMA_50_PERIOD}({rma50_15_val:.2f})")

            if cfg.ICHIMOKU_CLOUD_ENABLED:
                if not ichimoku_gate_ok_buy:
                    reasons.append(f"Ichimoku Cloud buy: price not above cloud / future not green (vote)")
                if not ichimoku_gate_ok_sell:
                    reasons.append(f"Ichimoku Cloud sell: price not below cloud / future not red (vote)")

            if not cloud_group_ok_buy:
                reasons.append("Cloud group buy: need 1-of-2 (Ichimoku/RMA cloud) — 0 agreed")
            if not cloud_group_ok_sell:
                reasons.append("Cloud group sell: need 1-of-2 (Ichimoku/RMA cloud) — 0 agreed")
            if not oscillator_group_ok_buy:
                reasons.append("Oscillator group buy: need 2-of-3 (PPO/RSI/TK) — not met")
            if not oscillator_group_ok_sell:
                reasons.append("Oscillator group sell: need 2-of-3 (PPO/RSI/TK) — not met")

            logger_pair.debug(f"😒 {pair_name} | Suppression: {', '.join(reasons)}") 

        return pair_name, {
            "state": "ALERT_SENT" if alerts_to_send else "NO_SIGNAL",
            "ts": int(time.time()),
            "summary": {
                "alerts": len(alerts_to_send),
                "future_cloud": "green" if cloud_up else "red" if cloud_down else "neutral",
                "hist_rma": round(hist_curr, 4), 
                "suppression": ", ".join(failed_conditions + reasons) if (failed_conditions or reasons) else "No conditions met"
            }
        }

    except asyncio.CancelledError:
        logger_pair.warning(f"Evaluation cancelled for {pair_name}")
        raise
    except RuntimeError as e:
        logger_pair.critical(f"🚨 INVARIANT VIOLATION in {pair_name}: {e}")
        return pair_name, {
            "state": "INVARIANT_VIOLATION",
            "ts": int(time.time()),
            "summary": {
                "alerts": 0,
                "future_cloud": "neutral",
                "hist_rma": 0.0,
                "error": str(e)
            }
        }
    except Exception as e:
        logger_pair.exception(
            f"❌ Error in _apply_and_dispatch_alerts for {pair_name}: {e} | Correlation: {correlation_id}"
        )
        return None

async def evaluate_pair_and_alert(pair_name: str, data_15m: PriceData, data_5m: PriceData,
    data_daily: Optional[Dict[str, np.ndarray]], sdb: RedisStateStore, telegram_queue: TelegramQueue, correlation_id: str,
    reference_time: int, fetcher: DataFetcher, symbol: str, alerts_sent_ref: List[int] = None, alerts_sent_lock: asyncio.Lock = None,
    max_alerts_per_run: int = cfg.MAX_ALERTS_PER_RUN) -> Optional[Tuple[str, Dict[str, Any]]]:

    logger_pair = logging.getLogger(f"macd_bot.{pair_name}.{correlation_id}")

        gr = await _eval_gate(pair_name, data_15m, data_5m, data_daily, sdb, correlation_id, reference_time)
        if gr is None:
            return None
        if isinstance(gr, tuple):
            return gr  # hard reject / wick reject / gate blocked -- already final

        if cfg.ENABLE_CONFLUENCE_GATE and (gr.buy_common or gr.sell_common):
            score, total = compute_confluence_score(gr, is_buy=gr.buy_common)
            required = min(cfg.CONFLUENCE_MIN_VOTES, total)
            if score < required:
                logger_pair.info(
                    f"[{pair_name}] Confluence gate blocked: {score}/{total} votes "
                    f"(need {required}) — skipping Phase-2 indicators"
                )
                await _blanket_reset_pair(sdb, pair_name, logger_pair)
                return pair_name, {
                    "state": "NO_SIGNAL",
                    "ts": int(time.time()),
                    "summary": {
                        "alerts": 0,
                        "future_cloud": "green" if gr.cloud_up else "red" if gr.cloud_down else "neutral",
                        "hist_rma": 0.0,
                        "suppression": f"Confluence gate: {score}/{total} votes, need {required}"
                    }
                }

        alert_result = await _eval_alerts(gr, data_5m, data_daily, reference_time, sdb, correlation_id, logger_pair)
        if alert_result is None:
            return None
        if isinstance(alert_result, tuple) and len(alert_result) == 2:
            return alert_result  # reserved: RuntimeError path inside _eval_alerts
        context, conditional_states, raw_alerts = alert_result

        return await _apply_and_dispatch_alerts(
            gr, context, conditional_states, raw_alerts, sdb, telegram_queue, fetcher, symbol,
            correlation_id, logger_pair, alerts_sent_ref, alerts_sent_lock, max_alerts_per_run
        )
    finally:
        PAIR_ID.set("")
        global _pair_eval_counter
        _pair_eval_counter += 1
        if _pair_eval_counter % MEMORY_CHECK_INTERVAL_PAIRS == 0:
            try:
                process = psutil.Process()
                current_memory_mb = process.memory_info().rss / 1024 / 1024
                memory_limit_mb = cfg.MEMORY_LIMIT_BYTES / 1024 / 1024
                if current_memory_mb > (memory_limit_mb * 0.8):
                    logger_pair.warning(f"Memory spike: {current_memory_mb:.0f}MB / {memory_limit_mb:.0f}MB")
            except Exception:
                pass

async def guarded_eval(task_data, state_db, telegram_queue, correlation_id, reference_time, fetcher,
                       alerts_sent_ref=None, alerts_sent_lock=None, max_alerts_per_run=cfg.MAX_ALERTS_PER_RUN ):

    p_name, symbol, candles = task_data

    try:
        pd_15m = parse_candles_to_numpy(candles.get("15"))
        pd_5m = parse_candles_to_numpy(candles.get("5"))
        pd_daily = parse_candles_to_numpy(candles.get("D")) if (cfg.ENABLE_PIVOT or cfg.ENABLE_CPR) else None

        if pd_15m is None:
            logger_main.warning(f"Skipping {p_name}: 15m parse failed")
            return None
        
        if pd_5m is None:
            logger_main.warning(f"Skipping {p_name}: 5m parse failed")
            return None

        data_15m = pd_15m
        data_5m = pd_5m
        data_daily = pd_daily.as_dict() if pd_daily is not None else None

        result = await evaluate_pair_and_alert(
            p_name, data_15m, data_5m, data_daily,
            state_db, telegram_queue, correlation_id, reference_time, fetcher, symbol,
            alerts_sent_ref, alerts_sent_lock, max_alerts_per_run
        )
        return result
    
    except asyncio.CancelledError:
        logger_main.warning(f"Evaluation cancelled for {p_name}")
        raise
    
    except Exception as e:
        logger_main.error(f"Error in {p_name} evaluation: {e}", exc_info=False)
        return None
    
    finally:
        pass

async def process_pairs_with_workers(fetcher: DataFetcher, products_map: Dict[str, dict],
    pairs_to_process: List[str], state_db: RedisStateStore, telegram_queue: TelegramQueue,
    correlation_id: str, lock: RedisLock, reference_time: int,
    alerts_sent_ref: List[int] = None, alerts_sent_lock: asyncio.Lock = None,
    max_alerts_per_run: int = cfg.MAX_ALERTS_PER_RUN) -> List[Tuple[str, Dict[str, Any]]]:

    logger_main.info(f"🔡 Phase 1: Fetching candles for {len(pairs_to_process)} pairs...")
    fetch_start = time.time()

    limit_15m = 300
    limit_5m = max(
        Constants.MIN_CANDLES_FOR_INDICATORS + Constants.CANDLE_SAFETY_BUFFER,
        cfg.RMA_200_PERIOD * 3 
    )
    daily_limit = cfg.PIVOT_LOOKBACK_PERIOD if (cfg.ENABLE_PIVOT or cfg.ENABLE_CPR) else 0

    pair_requests = []
    valid_tasks = []
    for pair_name in pairs_to_process:
        product_info = products_map.get(pair_name)
        if not product_info:
            continue

        symbol = product_info["symbol"]
        resolutions = [("15", limit_15m), ("5", limit_5m)]

        if cfg.ENABLE_PIVOT or cfg.ENABLE_CPR:
            resolutions.append(("D", daily_limit))

        pair_requests.append((symbol, resolutions))
        valid_tasks.append((pair_name, symbol))

    all_candles = await fetcher.fetch_all_candles_truly_parallel(
        pair_requests, reference_time
    )

    fetch_elapsed = time.time() - fetch_start
    logger_main.info(f"🌀 Phase 1 complete: {fetch_elapsed:.1f}s")

    logger_main.debug("⚙️ Phase 2: Preparing evaluation tasks...")

    prepared_tasks = []
    for pair_name, symbol in valid_tasks:
        candles = all_candles.get(symbol, {})
        prepared_tasks.append((pair_name, symbol, candles))

    logger_main.debug(f"Ready to evaluate {len(prepared_tasks)} pairs")

    logger_main.debug(f"🧠 Phase 3: Evaluating {len(prepared_tasks)} pairs...")
    eval_start = time.time()
    eval_semaphore = asyncio.Semaphore(cfg.EVAL_CONCURRENCY_LIMIT)  # NEW, e.g. 5

    async def _bounded_eval(t):
        async with eval_semaphore:
            return await guarded_eval(
                t, state_db, telegram_queue, correlation_id,
                reference_time, fetcher, alerts_sent_ref, alerts_sent_lock, max_alerts_per_run
            )

    results = await asyncio.gather(
        *[_bounded_eval(t) for t in prepared_tasks],
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
    del results, prepared_tasks, pair_requests, valid_tasks
    
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
    logger_main.debug("🧹 Fetch-phase data deleted, GC forced")
    current_memory_mb, limit_mb, usage_pct = log_memory_usage("💾 Memory after batch cleanup")
    if current_memory_mb and current_memory_mb > limit_mb * 0.8:
        logger_main.warning(
            f"⚠️ Memory still high after cleanup: {current_memory_mb:.0f}MB ({usage_pct:.0f}%). "
            f"Possible memory leak?"
        )

    knox_approved = len(valid_results)
    knox_rejected = len(pairs_to_process) - knox_approved
    
    logger_main.info(
        f"🎯🧠 Knox: {knox_approved} approved, {knox_rejected} rejected "
        f"({len(pairs_to_process)} total evaluated)"
    )
    return valid_results

async def run_once() -> bool:
    MAX_ALERTS_PER_RUN = cfg.MAX_ALERTS_PER_RUN
    all_results: List[Tuple[str, Dict[str, Any]]] = []
    correlation_id = uuid.uuid4().hex[:8]
    TRACE_ID.set(correlation_id)
    logger_run = logging.getLogger(f"macd_bot.run.{correlation_id}")
    start_time = time.time()
    sdb: Optional[RedisStateStore] = None
    lock: Optional[RedisLock] = None
    fetcher: Optional[DataFetcher] = None
    telegram_queue: Optional[TelegramQueue] = None
    lock_acquired = False
    lock_extension_task: Optional[asyncio.Task] = None
    alerts_sent_lock = asyncio.Lock()

    
    products_map: Optional[Dict[str, dict]] = None
    pairs_to_process: List[str] = []
    
    reference_time = get_trigger_timestamp()
    logger_run.info(
        f"🎯 Run started | Correlation ID: {correlation_id} | "
        f"Reference time: {reference_time} ({format_ist_time(reference_time)})"
    )
    logger_run.debug(
        f"Momentum gate active (all alerts) | 3-of-5 vote | "
        f"body_ratio_min={cfg.CPR_MOMENTUM_BODY_RATIO_MIN}"
    )
    if cfg.ENABLE_CPR:
        logger_run.info(
            f"CPR gate active | threshold={cfg.CPR_THRESHOLD_PCT} | "
            f"wide CPR requires move_from_prev_close (see CPR_THRESHOLD_PCT/adaptive)"
        )
    else:
        logger_run.debug("CPR gate disabled")
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

        logger_run.debug("Connecting to Redis...")
        sdb = RedisStateStore(cfg.REDIS_URL)
        await sdb.connect()

        if os.getenv("CLEAR_ALL_STATES", "false").lower() == "true":
            if sdb and not sdb.degraded:
                logger_run.warning("🚨 CLEAR_ALL_STATES requested — purging all Redis alert states...")
                st, dd = await _clear_all_redis_states(sdb, pairs_to_process, logger_run)
                if telegram_queue is None:
                    telegram_queue = TelegramQueue(cfg.TELEGRAM_BOT_TOKEN, cfg.TELEGRAM_CHAT_ID)
                await telegram_queue.send(escape_markdown_v2(
                    f"🧹 {cfg.BOT_NAME} All stored alert states cleared\n"
                    f"State keys: {st} | Dedup keys: {dd}\n"
                    f"Time: {format_ist_time()}"
                ))
            else:
                logger_run.error("CLEAR_ALL_STATES=true but Redis is unavailable/degraded")

        if sdb.degraded and not sdb.degraded_alerted:
            logger_run.critical(
                "🚨 Redis is in degraded mode – alert deduplication disabled!"
            )

        if sdb and not sdb.degraded and (cfg.ENABLE_PIVOT or cfg.ENABLE_VWAP):
            logger_run.debug("Checking daily reset conditions...")
            day_tracker_key = "global:last_reset_date"
            current_date_str = get_utc_date_key(reference_time)
            
            last_reset_date_str = None
            try:
                last_reset_date_str = await sdb.get_metadata(day_tracker_key)
            except Exception as e:
                logger_run.warning(f"Failed to get last reset date: {e}")
     

            if should_reset_daily_state(reference_time, last_reset_date_str):
                logger_run.info(f"🔄 New day detected ({current_date_str}). Resetting daily states...")
    
                all_delete_keys = []
    
                if cfg.ENABLE_PIVOT:
                    pivot_alerts = (
                        [f"pivot_up_{level}" for level in PIVOT_LEVELS_BUY] +
                        [f"pivot_down_{level}" for level in PIVOT_LEVELS_SELL]
                    )
        
                    for pair in pairs_to_process:
                        for alert_key in pivot_alerts:
                            redis_key = ALERT_KEYS.get(alert_key)
                            if redis_key:
                                all_delete_keys.append(f"{pair}:{redis_key}")
    
                if cfg.ENABLE_VWAP:
                    vwap_alerts = ["vwap_up", "vwap_down"]
                    for pair in pairs_to_process:
                        for alert_key in vwap_alerts:
                            redis_key = ALERT_KEYS.get(alert_key)
                            if redis_key:
                                all_delete_keys.append(f"{pair}:{redis_key}")
   
                if all_delete_keys:
                    try:
                        await sdb.atomic_batch_update([], deletes=all_delete_keys)
                        logger_run.info(
                            f"✅ Cleared {len(all_delete_keys)} daily alert keys "
                            f"from {len(pairs_to_process)} pairs"
                        )
                    except Exception as e:
                        logger_run.error(f"❌ Failed to delete daily reset keys: {e}")
                        raise
    
                try:
                    await sdb.set_metadata(day_tracker_key, current_date_str)
                    logger_run.info(f"✅ Daily reset complete ({current_date_str})")
                except Exception as e:
                    logger_run.error(f"❌Failed to save reset date: {e}")
            else:
                logger_run.debug(f"No daily reset needed (last reset: {last_reset_date_str})")

        if sdb.degraded and not sdb.degraded_alerted:
            telegram_queue = TelegramQueue(cfg.TELEGRAM_BOT_TOKEN, cfg.TELEGRAM_CHAT_ID)
            await telegram_queue.send(escape_markdown_v2(
                f"⚠️ {cfg.BOT_NAME} - REDIS DEGRADED MODE\n"
                f"Alert deduplication is disabled. You may receive duplicate alerts.\n"
                f"Time: {format_ist_time()}"
            ))
            sdb.degraded_alerted = True

        if telegram_queue is None:
            telegram_queue = TelegramQueue(cfg.TELEGRAM_BOT_TOKEN, cfg.TELEGRAM_CHAT_ID)

        if sdb.degraded:
            logger_run.warning(
                "⚠️ Redis degraded — skipping distributed lock, proceeding without "
                "duplicate-run protection (core alerting still runs)"
            )
            lock = None
            lock_acquired = False
        else:
            lock = RedisLock(sdb._redis, "macd_bot_run")
            lock_acquired = await lock.acquire(timeout=5.0)
            if not lock_acquired:
                logger_run.warning(
                    "⚠️ Another instance is running (Redis lock held) - exiting gracefully"
                )
                return False

        async def extend_lock_periodically(lock_obj: RedisLock, telegram_queue: TelegramQueue):
            while not shutdown_event.is_set():
                try:
                    if lock_obj.should_extend():
                        success = await lock_obj.extend(timeout=3.0)
                        if success:
                            logger_run.debug("🔒 Lock extended successfully")
                        else:
                            logger_run.critical("✘ Lock extension failed...")
                            try:
                                await telegram_queue.send(escape_markdown_v2(
                                    f"⚠️ Lock extension failed for {lock_obj.lock_key}"
                                ))
                            except Exception as e:
                                logger_run.error(f"Failed to send lock failure alert: {e}")
                            shutdown_event.set()
                            return

                    time_since_extend = time.monotonic() - lock_obj.last_extend_time
                    time_until_threshold = max(0, lock_obj.get_lock_extend_interval() - time_since_extend)
                    sleep_time = max(30, min(180, int(time_until_threshold * 0.75)))

                    try:
                        await asyncio.wait_for(shutdown_event.wait(), timeout=sleep_time)
                    except asyncio.TimeoutError:
                        pass
            
                except asyncio.CancelledError:
                    break

                except Exception as e:
                    logger_run.error(f"Lock extension task error: {e}")
                    await asyncio.sleep(60)

        lock_extension_task = (
            asyncio.create_task(extend_lock_periodically(lock, telegram_queue))
            if lock is not None else None
        )
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
        alerts_sent_ref = [0] 
        all_results = await process_pairs_with_workers(
            fetcher, products_map, pairs_to_process, sdb, telegram_queue, 
            correlation_id, lock, reference_time,
            alerts_sent_ref, alerts_sent_lock, MAX_ALERTS_PER_RUN
        ) 

        logger_run.debug("Cleanup phase with normal garbage collection...")

        fetcher_stats = fetcher.get_stats()

        total_required = fetcher_stats['candles']['success'] + fetcher_stats['candles']['failed']
        candles_str = f"{fetcher_stats['candles']['success']}/{total_required}"

        logger_run.info(
            f"Fetch Stats | "
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
        redis_status = "OK" if (sdb and not sdb.degraded) else "DEGRADED"

        summary = (
            f"🎯🌏 RUN COMPLETE | "
            f"Duration: {run_duration:.1f}s | "
            f"Pairs: {len(all_results)}/{len(pairs_to_process)} | "
            f"Alerts: {alerts_sent_ref[0]} | "
            f"Memory: {int(final_memory_mb)}MB (Δ{memory_delta:+.0f}MB) | "
            f"Redis: {redis_status}"
        )
        logger_run.info(summary)

        if alerts_sent_ref[0] > MAX_ALERTS_PER_RUN:
            await telegram_queue.send(escape_markdown_v2(
                f"⚠️ HIGH ALERT VOLUME\n"
                f"Alerts sent: {alerts_sent_ref[0]}\n"
                f"Pairs processed: {len(all_results)}\n"
                f"Time: {format_ist_time()}"
            ))

        return True

    except asyncio.TimeoutError:
        logger_run.error("⚠️ Run timed out - exceeded RUN_TIMEOUT_SECONDS")
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
                logger_run.debug("🔏 Redis lock released")
            except asyncio.TimeoutError:
                logger_run.error("Timeout releasing lock")
            except Exception as e:
                logger_run.error(f"Error releasing lock: {e}", exc_info=False)

        if sdb:
            try:
                await asyncio.wait_for(sdb.close(), timeout=3.0)
                logger_run.debug("✅ Redis connection closed")
            except asyncio.TimeoutError:
                logger_run.error("Timeout closing Redis")
            except Exception as e:
                logger_run.error(f"Error closing Redis: {e}", exc_info=False)

        try:
            await asyncio.wait_for(
                RedisStateStore.shutdown_global_pool(),
                timeout=5.0
            )
        except asyncio.TimeoutError:
            logger_run.error("Timeout shutting down Redis pool")
        except Exception as e:
            logger_run.error(f"Error shutting down Redis pool: {e}")

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
        logger.warning("❌ AOT not available, using JIT fallback. Reason: %s", reason)
        logger.warning("⚠️ Performance may be degraded. First run may be slow.")

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
        try:
            async with asyncio.timeout(cfg.RUN_TIMEOUT_SECONDS):
                return await run_once()
        except TimeoutError:
            logger.critical(
                "Run exceeded hard deadline: %ss",
                cfg.RUN_TIMEOUT_SECONDS
            )
            return False
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
