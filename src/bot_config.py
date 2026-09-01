from __future__ import annotations
import os
import sys
import re
import logging
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Union, Set
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from contextvars import ContextVar
import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict, PrivateAttr

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
}
CONFIG_OVERRIDE_METADATA_KEY = "config_override"

class Constants:
    MIN_WICK_RATIO = 0.2
    PPO_RSI_GUARD_BUY = 0.50
    PPO_RSI_GUARD_SELL = -0.50
    PPO_SIGNAL_CROSS_MAX_BUY = 0.30
    PPO_SIGNAL_CROSS_MIN_SELL = -0.30
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
    CLOUD_GROUP_MIN_VOTES_OF_3 = 2




PIVOT_LEVELS_BUY = ["P", "S1", "S2", "S3", "R1", "R2"]
PIVOT_LEVELS_SELL = ["P", "S1", "S2", "R1", "R2", "R3"]

class CompiledPatterns:
    VALID_SYMBOL = re.compile(r'^[A-Z0-9_]+$')
    ESCAPE_MARKDOWN = re.compile(r'[_*\[\]()~`>#+\-=|{}.!]') 
    SECRET_TOKEN = re.compile(r'\b\d{6,}:[A-Za-z0-9_-]{20,}\b')
    CHAT_ID = re.compile(r'chat_id=\d+')
    REDIS_CREDS = re.compile(r'(redis://[^@]+@)')

TRACE_ID: ContextVar[str] = ContextVar("trace_id", default="")
PAIR_ID: ContextVar[str] = ContextVar("pair_id", default="")

class BotConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    _validation_warnings: List[str] = PrivateAttr(default_factory=list)
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
    ENABLE_PPO_ALERTS: bool = Field(default=True, description="Master switch for PPO cross alerts: signal-line cross, zero-line cross, adaptive-threshold cross (6 alert types)")
    ENABLE_PPOHIST_ALERT: bool = Field(default=True, description="Enable PPO Histogram Reversal alerts (ppohist_buy/sell). Independent of ENABLE_PPO_GATE, which drives the unrelated PPO trend-gate/confluence logic")   
    ENABLE_RSI_ALERTS: bool = Field(default=True, description="Master switch for RSI cross alerts: EMA5 cross, adaptive-threshold cross (4 alert types). Independent of RSI_GUARD_ENABLED, which is an unrelated trend gate")
    ENABLE_HIST_RMA: bool = Field(default=True, description="Enable RMA 10/30 histogram reversal alerts")
    HIST_RMA_FAST: int = Field(default=10, ge=2, le=100, description="Histogram RMA fast period")
    HIST_RMA_SLOW: int = Field(default=30, ge=5, le=200, description="Histogram RMA slow period")
    ENABLE_PPO_GATE: bool = Field(default=True, description="Enable PPO(32,84,20) as trend gate")
    PPO_GATE_FAST: int = Field(default=32, ge=1, le=100, description="Gate PPO fast period")
    PPO_GATE_SLOW: int = Field(default=84, ge=2, le=200, description="Gate PPO slow period")
    PPO_GATE_SIGNAL: int = Field(default=20, ge=1, le=50, description="Gate PPO signal period")
    PPOHIST_WARMUP_BUFFER_BARS: int = Field(default=56, ge=0, le=200)
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
    CONFLUENCE_MIN_PCT: float = Field(default=60.0, ge=1.0, le=100.0, description="Min percentage of the achievable confluence total required to pass. Denominator = sum of weights of enabled, non-abstaining votes this cycle, so the threshold auto-scales when votes are enabled/disabled — no manual retuning needed")
    CONFLUENCE_MIN_ABS_SCORE: float = Field(default=18.0, ge=0.0, le=50.0, description="Absolute weighted-score floor required to pass the confluence gate, applied alongside CONFLUENCE_MIN_PCT. The stricter of the two (percentage-of-total vs this fixed floor) wins, so a low-vote-count cycle can't clear the gate on percentage alone")
    OB_MIN_OTHER_SCORE: float = Field(default=3.0, ge=0.0, le=50.0, description="Min weighted score from votes OTHER than base_trend and order_block required before the OB vote is allowed to count toward the confluence total. base_trend is excluded because it's a precondition for evaluation, not independent confluence. Default 3.0 is set above oi_funding's 2.5 weight so oi_funding alone can't pair with OB to clear the gate")
    OB_MIN_PENETRATION_ATR_MULT: float = Field(default=0.05, ge=0.0, le=2.0, description="Minimum close penetration beyond the zone edge (top for demand, bottom for supply), scaled by ATR_SHORT, required to count as a confirmed reversal. 0 disables the check. Prevents a close a fraction of a tick beyond the zone from counting as 'reversed'")
    OB_CONFIRM_LOOKAHEAD_CANDLES: int = Field(default=5, ge=0, le=10, description="Candles of grace after a zone is first touched during which a close beyond the opposite edge (+ OB_MIN_PENETRATION_ATR_MULT) still counts as a confirmed reversal. 0 restores the old same-candle-only behavior. A close that fully breaks the zone in the invalidating direction during the grace window kills it immediately")
    OB_PERSISTENCE_CANDLES: int = Field(default=2, ge=0, le=10, description="How many additional closed 15m candles after an OB confirmation to keep the gate valid. 0 = exact-candle-only (legacy).")
    ENABLE_WIN_RATE_FILTER: bool = Field(default=False)
    ENABLE_OI_FUNDING_FILTER: bool = Field(default=False, description="Block BUY/SELL when OI isn't rising (vs pair's own history) AND funding is crowded (vs pair's own history) in the alert direction")
    OI_FUNDING_HISTORY_LEN: int = Field(default=30, ge=5, le=200, description="Rolling window of past OI/funding samples kept per pair (in run cycles, e.g. 30 runs @ 15m cadence ≈ 7.5h)")
    MIN_OI_FUNDING_SAMPLES: int = Field(default=8, ge=3, description="Min warm-up samples before the adaptive gate activates for a pair; fail-open until then")
    OI_RISING_PERCENTILE: float = Field(default=0.50, ge=0.0, le=1.0, description="OI delta must exceed this percentile of the pair's own recent |delta| history to count as 'rising with conviction'")
    OI_DELTA_REF_SAMPLES: int = Field(default=3, ge=1, le=20, description="Number of most-recent OI history samples averaged to form the reference point for oi_delta, instead of comparing oi_now against only the single last sample. Smooths out a single anomalous tick (exchange glitch, brief liquidation cascade spike) from distorting the delta. 1 reproduces the old single-sample behavior")
    FUNDING_CROWDED_PERCENTILE: float = Field(default=0.80, ge=0.5, le=1.0, description="Current funding must be at/above this percentile (BUY) or at/below its complement (SELL) of the pair's own recent funding history to count as 'crowded'")
    FUNDING_ABS_FLOOR: float = Field(default=0.0005, ge=0.0, description="Min |funding| required before percentile-crowding applies at all, so a flat near-zero history can't self-trigger 'crowded'")
    MIN_OI_USD: float = Field(default=75000, ge=0.0, description="Ignore the OI/funding gate entirely for pairs whose current OI is below this floor (quote currency). 0 disables the floor")
    OI_FUNDING_MAX_SAMPLE_AGE_SEC: int = Field(default=10800, ge=300, description="Prune OI/funding samples older than this (default 180min ≈ 12 cycles @15m, matching OI_DIVERGENCE_LOOKBACK_SAMPLES). Prevents comparing a stale pre-outage sample as if only one cycle passed")
    ENABLE_OI_PRICE_DIVERGENCE: bool = Field(default=False, description="Block BUY when price is rising but OI is falling (short-covering, not new demand); block SELL when price is falling but OI is falling (long liquidation, not new supply). Requires ticker mark price to be available.")
    OI_DIVERGENCE_LOOKBACK_SAMPLES: int = Field(default=12, ge=2, le=200, description="How many OI/price history samples back to compare against for divergence (default 12 runs @15m ≈ 3h)")
    OI_DIVERGENCE_MIN_PRICE_ROC_PCT: float = Field(default=0.3, ge=0.0, le=50.0, description="Min absolute price move (%) over the lookback window before divergence logic applies at all")
    OI_DIVERGENCE_MIN_OI_FALL_PCT: float = Field(default=2.0, ge=0.0, le=100.0, description="Min OI decline (%) over the lookback window to count as 'falling with conviction' (closing/covering, not new positioning)")
    ENABLE_OB_GATE: bool = Field(default=False, description="Add institutional order-block (supply/demand) reversal on 15m as a confluence vote. Abstains (None) unless a fresh, first-touch reversal off an unmitigated zone confirms this cycle")
    OB_FILTER_CONFLUENCE: bool = Field(default=False) 
    OB_LOOKBACK_CANDLES: int = Field(default=50, ge=20, le=500, description="How many closed 15m candles back to scan for order-block zones (default 96 ≈ 24h)")
    OB_IMPULSE_LOOKAHEAD: int = Field(default=3, ge=1, le=10, description="Candles after a candidate base candle checked for the impulsive displacement that confirms it as an order block")
    ENABLE_OB_PREMIUM_DISCOUNT_FILTER: bool = Field(default=False, description="Only accept demand-zone OB reversals below the 50% equilibrium of the OB_LOOKBACK_CANDLES dealing range (discount), and supply-zone reversals above it (premium). Zones on the wrong side are skipped entirely.")
    OUTCOME_LOOKAHEAD_CANDLES: int = Field(default=8, ge=1, le=96) 
    OUTCOME_FAVORABLE_MOVE_PCT: float = Field(default=0.3, ge=0.01, le=10.0) 
    MIN_WIN_RATE_SAMPLE: int = Field(default=20, ge=1)    
    MIN_WIN_RATE: float = Field(default=0.55, ge=0.0, le=1.0)    
    ENABLE_BRAIN: bool = Field(default=False, description="Master switch for the Brain analysis/shadow-mode/reporting layer. Requires ENABLE_WIN_RATE_FILTER to be meaningful")
    BRAIN_SHADOW_MODE: bool = Field(default=True, description="When an alert is rejected by the win-rate filter, keep tracking what would have happened instead of discarding it")
    BRAIN_REPORT_INTERVAL_RUNS: int = Field(default=48, ge=1, le=2000, description="Send a Telegram analysis report every N cron runs (default 48 runs ≈ 12h at 15m cadence)")
    BRAIN_REWARDABLE_MIN_CONFLUENCE_PCT: float = Field(default=80.0, ge=50.0, le=100.0, description="Min confluence % required for a win-rate-rejected alert to be eligible for a rewardable override")
    BRAIN_REWARDABLE_MIN_SHADOW_SAMPLE: int = Field(default=10, ge=3, description="Min resolved shadow samples in the high-confluence bucket for this alert_key before an override is trusted")
    BRAIN_REWARDABLE_MIN_SHADOW_WR: float = Field(default=0.60, ge=0.5, le=1.0, description="Shadow win rate required in the high-confluence bucket to allow rewardable overrides through")
    BRAIN_CONFLUENCE_BUCKET_PCT: float = Field(default=10.0, ge=1.0, le=50.0, description="Bucket width (in confluence %) used when the brain scans for a better CONFLUENCE_MIN_PCT in its report")
    BRAIN_REPORT_STREAM_SAMPLE: int = Field(default=5000, ge=100, le=50000, description="Max recent OUTCOME_LOG_STREAM/SHADOW_LOG_STREAM entries the brain reads per report")
    BRAIN_ALERT_DISABLE_THRESHOLD_WR: float = Field(default=0.40, ge=0.0, le=1.0, description="Pooled win rate (across all pairs) below which the brain recommends disabling an alert_key entirely in its report")
    BRAIN_OVERRIDE_COOLDOWN_SECONDS: int = Field(default=14400, ge=600, le=86400, description="Min seconds between rewardable overrides for the same alert_key")
    BRAIN_STAR_ALERT_WR: float = Field(default=0.70, ge=0.5, le=1.0, description="Win rate above which an alert is flagged as a star performer")
    BRAIN_ANALYSIS_WINDOW_DAYS: int = Field(default=30, ge=7, le=365, description="Only analyze outcomes from the last N days")
    ENABLE_TELEGRAM_FEEDBACK: bool = Field(default=False, description="Attach 'Took Trade'/'Skipped' inline buttons to alerts and poll Telegram getUpdates each run to record taps into feedback_log_stream")
    TELEGRAM_FEEDBACK_TTL_HOURS: int = Field(default=24, ge=1, le=168, description="How long a feedback_pending record (and its buttons) stays live before the alert is considered expired/unanswered")
    BRAIN_MC_SIMULATIONS: int = Field(default=50, ge=0, le=500, description="Block-bootstrap Monte Carlo simulations for the robustness check in the periodic brain report. 0 disables it (offline-only, never affects live gating).")
    DRY_RUN_MODE: bool = Field(default=False)
    SKIP_WARMUP: bool = Field(default=False)
    REJECT_HIGH_DEVIATION: bool = Field( default=False)
    SANITIZE_BAD_CANDLES: bool = Field(default=False, description="If True, drop individual invalid candles instead of rejecting the whole fetch")
    ICHIMOKU_CLOUD_ENABLED: bool = Field(default=True, description="Enable Ichimoku Cloud as trend gate")
    ICHIMOKU_CONVERSION_PERIODS: int = Field(default=9, ge=1, le=300, description="Ichimoku conversion line length")
    ICHIMOKU_BASE_PERIODS: int = Field(default=26, ge=1, le=400, description="Ichimoku base line length")
    ICHIMOKU_SPANB_PERIODS: int = Field(default=52, ge=1, le=500, description="Ichimoku leading span B length")
    ICHIMOKU_DISPLACEMENT: int = Field(default=26, ge=1, le=400, description="Ichimoku cloud forward displacement")
    ICHIMOKU_TK_CONVERSION_PERIODS: int = Field(default=23, ge=1, le=300, description="Tenkan (conversion) length used for TK guard + cross alerts, independent of cloud conversion length")
    ICHIMOKU_TK_BASE_PERIODS: int = Field(default=65, ge=1, le=400, description="Kijun (base) length used for TK guard + cross alerts, independent of cloud base length")
    ICHIMOKU_TK_GUARD_ENABLED: bool = Field(default=True, description="Require 15m Tenkan(conversion) vs Kijun(base) alignment: buy needs conversion>=base, sell needs conversion<=base")
    RMA_CLOUD_ENABLED: bool = Field(default=True, description="Enable RMA(fast)/RMA(50) 15m cloud as trend gate; green (buy) when RMA_fast>RMA50, red (sell) when RMA_fast<RMA50. Reuses the existing RMA50(15m)/RMA_50_PERIOD used for base trend.")
    RMA_CLOUD_FAST_PERIOD: int = Field(default=20, ge=2, le=200, description="RMA Cloud fast period (15m). Slow leg reuses RMA_50_PERIOD.")
    DYNAMIC_FLOW_RIBBON_ENABLED: bool = Field(default=True, description="Enable the 15m Dynamic Flow Ribbon (BigBeluga) as a third cloud-group trend gate alongside Ichimoku Cloud and RMA Cloud; green (buy) when the band-flip direction is bullish, red (sell) when bearish")
    DYNAMIC_FLOW_FACTOR: float = Field(default=3.0, ge=0.1, le=20.0, description="Dynamic Flow Ribbon band-width multiplier (Pine 'Length' input) — bands sit at basis \u00b1 factor*dist")
    DYNAMIC_FLOW_BASIS_LENGTH: int = Field(default=15, ge=2, le=200, description="Dynamic Flow Ribbon basis EMA period (15m, applied to hlc3)")
    DYNAMIC_FLOW_DIST_LENGTH: int = Field(default=200, ge=10, le=500, description="Dynamic Flow Ribbon distance SMA period (15m, applied to high-low) used to size the bands")
    ENABLE_DYNAMIC_FLOW_CROSS_ALERT: bool = Field(default=False, description="Add 15m Dynamic Flow Ribbon crossover/cross-under alert: fires on the candle where the ribbon's band-flip direction actually flips (not just 'currently bullish/bearish'), gated by buy_trend_common_relaxed/sell_trend_common_relaxed plus a wick-ratio-or-reversal-pattern condition — same gating style as CHoCH. Requires DYNAMIC_FLOW_RIBBON_ENABLED, since it detects a flip in that indicator's own array")
    ENABLE_TK_CONVERSION_CROSS: bool = Field(default=True, description="Enable 15m alert when close crosses above/below the Ichimoku conversion (Tenkan) line, subject to all other buy/sell common conditions")
    ENABLE_CLOUD_CROSS_ALERT: bool = Field(default=True, description="Enable 15m alert when close crosses above/below the Ichimoku cloud (9,26,52,26), subject to all other buy/sell common conditions") 
    ENABLE_KIJUN_CROSS: bool = Field(default=True, description="Enable 15m alert when close crosses above/below the Ichimoku base (Kijun) line (23,65), subject to all other buy/sell common conditions")
    ENABLE_STRONG_REVERSAL_ALERT: bool = Field(default=True, description="Enable candlestick reversal-pattern alert (Engulfing/Piercing/Star/Soldiers-Crows/Tweezer/Harami/Marubozu/Pinbar) on top of full buy_common/sell_common confluence")
    ENABLE_CHOCH_ALERT: bool = Field(default=False, description="Add 15m Change-of-Character (CHoCH) alert: fires on the displacement candle that recovers back through a swept short-term low/high, before any structural pivot is broken, gated by buy_trend_common/sell_trend_common plus a wick or reversal-pattern condition")
    CHOCH_SWING_LEN: int = Field(default=3, ge=2, le=20, description="Bars on each side used to confirm a short-term (minor) swing pivot for CHoCH structure — the lower highs / higher lows the break is measured against")
    CHOCH_LOOKBACK_CANDLES: int = Field(default=40, ge=10, le=200, description="How many closed 15m candles back from the current candle to scan for a qualifying CHoCH structure (swing pivots + sweep)")
    CHOCH_CONFIRM_WINDOW_CANDLES: int = Field(default=6, ge=1, le=20, description="Max candles allowed between the liquidity-sweep candle and the displacement candle. A displacement found further back than this is rejected as stale")
    CHOCH_ALLOW_SAME_CANDLE_SWEEP: bool = Field(default=False, description="If True, the sweep and the displacement candle may be the same 15m candle. If False (default), the displacement must be strictly after the sweep candle, removing same-candle sweep/displacement ambiguity")
    CHOCH_MIN_SWEEP_DISTANCE_ATR: float = Field(default=0.05, ge=0.0, le=2.0, description="Minimum distance (in ATR_SHORT multiples) the sweep wick must pierce beyond the prior short-term low/high to count as a real liquidity sweep rather than noise")
    CHOCH_MIN_DISPLACEMENT_BODY_RATIO: float = Field(default=0.45, ge=0.0, le=1.0, description="Minimum body-to-range ratio required on the displacement/entry candle so the CHoCH is backed by a real move rather than a thin/indecisive close")
    CHOCH_REQUIRE_FVG: bool = Field(default=False, description="If True, an unfilled direction-specific Fair Value Gap must also exist within the sweep-to-displacement window for the CHoCH to qualify. If False, FVG presence is still detected and reported in the alert reason as a bonus, not a requirement")
    CHOCH_CHECK_POI_TAP: bool = Field(default=False, description="Bonus confluence only, not a hard requirement: also check whether the sweep-to-displacement window touched an existing demand/supply order-block zone (POI). That window is now typically 1 candle (often the same candle), so POI taps will register less often than under the old break-based logic. Reuses the OB gate's zone detection (same OB_LOOKBACK_CANDLES/OB_FILTER_CONFLUENCE settings) and appends 'POI tap' to the CHoCH alert reason when true")
    CHOCH_PERSISTENCE_CANDLES: int = Field(default=1, ge=0, le=9, description="How many additional closed 15m candles after a displacement candle to keep the gate valid. Invalidation now checks price against the swept level, not a structural pivot — see below. 0 = exact-candle-only (fresh displacement required every cycle)")
    ENABLE_FIB_REVERSAL_ALERT: bool = Field(default=False, description="Enable Fibonacci Pivot Reversal alerts: price retraces into the 50-78.6% zone of the last major swing leg, with a confluence vote across the zone touch, oscillator divergence, wick/pattern rejection, and volume exhaustion")
    FIB_REVERSAL_CONFLUENCE_REQUIRED: int = Field(default=3, ge=1, le=4, description="Minimum number of the 4 confluence checks (wick/pattern rejection, Fibonacci zone, oscillator divergence, volume exhaustion) that must pass for a Fib Reversal alert to fire")
    FIB_REVERSAL_SWING_LENGTH: int = Field(default=5, ge=2, le=200, description="Bars on each side used to confirm a major swing pivot for the Fibonacci leg — matches the OB detection swing_len so the zone is anchored to the same structural swings")
    FIB_REVERSAL_SWING_LOOKBACK_CANDLES: int = Field(default=150, ge=20, le=2000, description="How many candles back to search for the swing pivots that anchor the Fibonacci leg and the divergence comparison")
    FIB_REVERSAL_ZONE_LOW: float = Field(default=0.5, ge=0.0, le=1.0, description="Lower bound of the Fibonacci retracement zone (as a fraction of the leg from the anchor swing to the extreme reached since) that counts as a zone touch")
    FIB_REVERSAL_ZONE_HIGH: float = Field(default=0.786, ge=0.0, le=1.0, description="Upper bound of the Fibonacci retracement zone — default 0.5-0.786 is the conventional 'golden zone'")
    FIB_REVERSAL_VOL_DRYUP_LOOKBACK: int = Field(default=6, ge=2, le=50, description="Number of candles compared for the volume dry-up check: mean volume over the N candles before the touch candle vs mean volume over the N candles before that")
    FIB_REVERSAL_VOL_SPIKE_MULT: float = Field(default=1.3, ge=1.0, le=5.0, description="Touch candle's volume must exceed its volume EMA by this multiple to count as an exhaustion/reversal spike") 
    FIB_REVERSAL_MAX_DIVERGENCE_AGE_BARS: int = Field(default=50, ge=5, le=500, description="Max bars between the anchor swing and the prior swing for divergence comparison")
    FIB_REVERSAL_MAJOR_SWING_LENGTH: int = Field(default=50, ge=2, le=200, description="Fallback only when no minor pivot exists...")
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
    ENABLE_ADX_STRENGTH_VOTE: bool = Field(default=False, description="Confluence vote: ADX in top ADX_STRENGTH_PCTL of its own history — a stricter secondary bar on top of the existing adx_ok gate, not a duplicate of it")
    ADX_STRENGTH_PCTL: float = Field(default=80.0, ge=1.0, le=99.0, description="Percentile threshold for the adx_strength confluence vote. Should be set meaningfully above ADX_ADAPTIVE_TARGET_PCTL so this vote and the base 'adx' vote aren't answering the same question")
    ENABLE_ATR_PCTL_VOTE: bool = Field(default=False, description="Confluence vote: current volatility (ATR) in top ATR_PCTL_VOTE_MIN of its own history — a volatility-regime check, distinct from the existing rvol vote which checks short/long ATR expansion trend")
    ATR_PCTL_VOTE_MIN: float = Field(default=0.60, ge=0.0, le=1.0, description="Min ATR percentile rank (0-1) required for the atr_percentile confluence vote to pass")
    ENABLE_VOLUME_PCTL_VOTE: bool = Field(default=False, description="Confluence vote: current volume in top VOLUME_PCTL_VOTE_MIN of its own trailing history — more robust than the existing EMA-based volume_above_ema_ok check, which a single spike can drag upward for several bars")
    VOLUME_PCTL_VOTE_MIN: float = Field(default=0.70, ge=0.0, le=1.0, description="Min volume percentile rank (0-1) required for the volume_percentile confluence vote to pass")
    VOLUME_PCTL_LOOKBACK: int = Field(default=96, ge=20, le=500, description="Rolling window (in 15m candles) for volume percentile ranking")
    VOLUME_PCTL_MIN_HISTORY: int = Field(default=50, ge=10, le=400, description="Min warm-up samples before volume percentile activates; fails open (vote excluded) until then")
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
    ENABLE_PPO_GATE_MOMENTUM_VOTE: bool = Field(default=False) 
    ENABLE_RSI_GUARD_MOMENTUM_VOTE: bool = Field(default=False) 
    ENABLE_RMA_CLOUD_MOMENTUM_VOTE: bool = Field(default=False) 
    ENABLE_VWAP_MOMENTUM_VOTE: bool = Field(default=False)
    BRAIN_STABILITY_MIN_HISTORY: int = Field(default=3, ge=1, le=20, description="StabilityGate: min threshold history entries before gating kicks in")
    BRAIN_STABILITY_MAX_JUMP: float = Field(default=2.0, ge=0.1, le=10.0, description="StabilityGate: max allowed deviation (in score points) from median history")
    BRAIN_CUSUM_DRIFT_DELTA: float = Field(default=0.10, ge=0.01, le=0.50, description="CUSUM: sensitivity to WR shift (delta parameter)")
    BRAIN_CUSUM_THRESHOLD: float = Field(default=2.0, ge=0.5, le=10.0, description="CUSUM: alarm threshold (h parameter)")
    BRAIN_FEE_PCT: float = Field(default=0.0006, ge=0.0, le=0.01, description="Taker fee per side (0.06%) used in EV/Kelly calculations")
    BRAIN_SLIPPAGE_PCT: float = Field(default=0.0003, ge=0.0, le=0.01, description="Estimated slippage per side used in EV/Kelly calculations")
    BRAIN_OOD_ENABLED: bool = Field(default=True, description="Vote-count OOD gate on/off")

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
        if self.ENABLE_VOLUME_PCTL_VOTE and self.VOLUME_PCTL_MIN_HISTORY >= self.VOLUME_PCTL_LOOKBACK:
            raise ValueError(
                f'VOLUME_PCTL_MIN_HISTORY ({self.VOLUME_PCTL_MIN_HISTORY}) must be < '
                f'VOLUME_PCTL_LOOKBACK ({self.VOLUME_PCTL_LOOKBACK})'
            )
        if self.ENABLE_ADX_STRENGTH_VOTE and self.ADX_STRENGTH_PCTL <= self.ADX_ADAPTIVE_TARGET_PCTL:
            raise ValueError(
                f'ADX_STRENGTH_PCTL ({self.ADX_STRENGTH_PCTL}) should be > '
                f'ADX_ADAPTIVE_TARGET_PCTL ({self.ADX_ADAPTIVE_TARGET_PCTL}), otherwise the '
                f'adx_strength vote duplicates the existing adx gate instead of adding a stricter bar'
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
    def validate_oi_divergence_window(self) -> 'BotConfig':
        if self.ENABLE_OI_PRICE_DIVERGENCE:
            required_age_sec = self.OI_DIVERGENCE_LOOKBACK_SAMPLES * 900  # 900s = 1 cycle @15m
            if self.OI_FUNDING_MAX_SAMPLE_AGE_SEC < required_age_sec:
                raise ValueError(
                    f'OI_FUNDING_MAX_SAMPLE_AGE_SEC ({self.OI_FUNDING_MAX_SAMPLE_AGE_SEC}s) is less than '
                    f'OI_DIVERGENCE_LOOKBACK_SAMPLES * 900 ({required_age_sec}s) — history will be pruned '
                    f'before the divergence lookback can be satisfied, so ENABLE_OI_PRICE_DIVERGENCE will '
                    f'silently never fire. Raise OI_FUNDING_MAX_SAMPLE_AGE_SEC or lower OI_DIVERGENCE_LOOKBACK_SAMPLES.'
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
    def validate_confluence_floor(self) -> 'BotConfig':
        if self.ENABLE_CONFLUENCE_GATE:
            max_achievable = sum(CONFLUENCE_WEIGHTS.values())
            if self.CONFLUENCE_MIN_ABS_SCORE > max_achievable:
                raise ValueError(
                    f'CONFLUENCE_MIN_ABS_SCORE ({self.CONFLUENCE_MIN_ABS_SCORE}) exceeds the max '
                    f'achievable weighted total ({max_achievable}) — every alert would be blocked forever'
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
            logger.debug(f"format_ist_time parsing failed for '{dt_or_ts}': {e}")
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