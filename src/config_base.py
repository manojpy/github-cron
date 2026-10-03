"""config_base — static configuration building blocks: JSON backend, timestamp normalisation, scoring-weight and override constants, pivot levels, context dataclasses, log redaction patterns/filters and the trace ContextVars. Imports nothing from the bot (no cfg), so any module can use it.

Split out of the original module without logic changes; the original
module remains as a facade re-exporting every name."""
from __future__ import annotations
import re
import logging
import json
from typing import Dict, Any, List, Union, Set
from dataclasses import dataclass
from zoneinfo import ZoneInfo
from contextvars import ContextVar
import numpy as np
import alert_registry as _alert_registry

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

CONFLUENCE_WEIGHTS: Dict[str, float] = {
    "base_trend": 3.0,
    "ichimoku_cloud": 2.0,
    "rma_cloud": 2.0,
    "dynamic_flow_ribbon": 2.0,
    "ppo_cross": 2.0,
    "rsi_guard": 2.0,
    "tk_guard": 2.0,
    "adx": 1.0,
    "rvol": 1.5,
    "cpr": 1.0,
    "oi_funding": 2.0,
    "order_block": 2.0,
    "adx_strength": 1.0,
    "atr_percentile": 1.5,
    "volume_percentile": 1.0,
    "ppo_gate_momentum":  1.0,
    "rsi_guard_momentum": 1.0,
    "rma_cloud_momentum": 1.0,
    "vwap_momentum": 1.0,
}

CONFIG_OVERRIDE_ALLOWED_FIELDS: Set[str] = {
    "CONFLUENCE_MIN_ABS_SCORE",
    "CONFLUENCE_MIN_PCT",
    "RSI_ADAPTIVE_BUY_VOLATILE",
    "RSI_ADAPTIVE_SELL_VOLATILE",
    "PPO_ADAPTIVE_VOLATILE",
}

BRAIN_DISABLED_KEYS_METADATA_KEY = "brain_disabled_alert_keys"

CONFIG_OVERRIDE_METADATA_KEY = "config_override"

PAIR_THRESHOLDS_METADATA_KEY = "pair_confluence_thresholds"

class Constants:
    MIN_WICK_RATIO = 0.2
    PPO_RSI_GUARD_BUY = 0.50
    PPO_RSI_GUARD_SELL = -0.50
    PPO_SIGNAL_CROSS_MAX_BUY = 0.30
    PPO_SIGNAL_CROSS_MIN_SELL = -0.30
    RSI_SIGNAL_CROSS_MAX_BUY = 65
    RSI_SIGNAL_CROSS_MIN_SELL = 35
    CIRCUIT_BREAKER_MAX_WAIT = 300
    INFINITY_CLAMP = 1e8
    VWAP_MAX_DISTANCE_PCT = 2.0
    INTER_BATCH_DELAY: float = 0.5
    MIN_CANDLES_FOR_INDICATORS = 250
    CANDLE_SAFETY_BUFFER = 100
    MIN_CLOSED_CANDLES_15M = 4          
    MIN_ALIGNED_5M_CANDLES = 200               
    CANDLE_FETCH_BUFFER_PERIODS = 3 
    API_TIMESTAMP_TOLERANCE_SEC = 300
    MIN_CANDLE_AGE_FROM_OPEN = 850
    DAILY_CACHE_SETTLE_RUNS = 3          # skip daily-candle caching for this many 15m runs after UTC midnight
    DAILY_CACHE_SETTLE_SEC = DAILY_CACHE_SETTLE_RUNS * 900   # 2700s = 45min — let the exchange's daily bar settle before trusting it for the whole day
    MIN_BODY_RATIO = 0.50
    HIGH_DEVIATION_THRESHOLD = 0.5
    REVERSAL_MARUBOZU_BODY_RATIO = 0.90 
    REVERSAL_PINBAR_WICK_RATIO = 0.66        
    REVERSAL_PINBAR_BODY_MAX_RATIO = 0.30 
    REVERSAL_STAR_BIG_BODY_MIN_RATIO = 0.50 
    REVERSAL_STAR_SMALL_BODY_MAX_RATIO = 0.30
    REVERSAL_SOLDIERS_MIN_BODY_RATIO = 0.55
    REVERSAL_PIERCING_MIN_PENETRATION = 0.50 
    REVERSAL_HARAMI_MAX_BODY_RATIO = 0.50 
    REVERSAL_TWEEZER_TOLERANCE_PCT = 0.05
    REVERSAL_PRIOR_LEG_LOOKBACK = 4 
    REVERSAL_PRIOR_LEG_MIN_RANGE_MULT = 0.5 
    OSCILLATOR_GROUP_MIN_VOTES = 1
    REVERSAL_MIN_PRIOR_BODY_RATIO: float = 0.40
    MACRO_CORR_WINDOW = 20
    MACRO_CORR_LOW = 0.3
    MACRO_CORR_HIGH = 0.6
    MACRO_MULT_MODERATE = 1.15
    MACRO_MULT_FULL = 1.30
    MACRO_RS_EASE_FACTOR = 0.75

PIVOT_LEVELS_BUY: List[str] = _alert_registry.PIVOT_LEVELS_BUY

PIVOT_LEVELS_SELL: List[str] = _alert_registry.PIVOT_LEVELS_SELL

@dataclass
class BtcMacroContext:
    """Snapshot of the macro reference pair's (default BTCUSD) directional
    state for one run, computed once and passed to every other pair's
    _apply_and_dispatch_alerts() call. Shadow-mode only for now — see
    cfg.ENABLE_MACRO_CONTEXT_GATE."""
    confirmation_buy: bool
    confirmation_sell: bool
    adx_ok: bool
    close: float          # current/trigger 15m candle close
    open: float            # current/trigger 15m candle open
    closes: np.ndarray     # recent 15m close series, for rolling correlation

@dataclass
class ClusterContext:
    """Same-run directional-cluster snapshot across the whole pair universe,
    computed via a gate-only pre-pass before Phase-3 dispatch (see
    _compute_directional_cluster in macd_unified.py). Used to detect and
    penalize 'beta trap' situations — e.g. BTC pumps and 8 correlated alts
    all fire BUY at once, which is really one trade wearing 8 costumes.
    LIVE — see cfg.ENABLE_CLUSTER_GATE / CLUSTER_PCT_THRESHOLD / CLUSTER_PENALTY_PCT."""
    buy_count: int
    sell_count: int
    total_pairs: int
    buy_pct: float
    sell_pct: float

@dataclass
class BiasContext:
    """Same-run Ichimoku directional-bias snapshot across the whole pair
    universe (23/65/130/65 on 15m by default — see BIAS_ICHIMOKU_* config),
    computed via a pre-pass before Phase-3 dispatch (see
    _compute_bias_context in macd_unified.py). Purely a cosmetic header
    prepended to every outgoing Telegram message this run — see
    cfg.ENABLE_BIAS_HEADER. Never gates or filters an alert.
    A pair counts as:
      up      - last closed 15m close above the cloud AND future cloud green
      down    - last closed 15m close below the cloud AND future cloud red
      neutral - everything else (inside the cloud, NaN cloud, or a
                mismatched price/future-cloud combination)"""
    up_count: int
    down_count: int
    neutral_count: int
    total_pairs: int
    up_pct: float
    down_pct: float
    neutral_pct: float

class CompiledPatterns:
    VALID_SYMBOL = re.compile(r'^[A-Z0-9_]+$')
    ESCAPE_MARKDOWN = re.compile(r'[_*\[\]()~`>#+\-=|{}.!]') 
    SECRET_TOKEN = re.compile(r'\b\d{6,}:[A-Za-z0-9_-]{20,}\b')
    CHAT_ID = re.compile(r'chat_id=\d+')
    REDIS_CREDS = re.compile(r'(redis://[^@]+@)')

TRACE_ID: ContextVar[str] = ContextVar("trace_id", default="")

PAIR_ID: ContextVar[str] = ContextVar("pair_id", default="")

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

_IST_TZ = ZoneInfo("Asia/Kolkata")

MEMORY_CHECK_INTERVAL_PAIRS = 5  # only sample RSS every N pair evaluations
