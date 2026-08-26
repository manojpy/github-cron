from __future__ import annotations
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from collections import OrderedDict
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import aot_bridge
from bot_config import cfg, logger, Constants, CprNotReadyError, BotConfig, PAIR_ID

from aot_bridge import (
    sanitize_array_numba, ema_loop, ema_loop_alpha, ema_loop_pine, kalman_loop,
    vwap_daily_loop_safe, rolling_mean_numba, rolling_min_max_numba,
    calculate_ppo_core, calculate_rsi_core, true_range_numba, calculate_atr_rma,
    calculate_adx_core, percentile_rank_numba,
)

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
    
def calculate_vwap_numpy(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
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

        # ── Ichimoku cloud (9/26/52/26 — cloud + future cloud only) ──
        if cfg.ICHIMOKU_CLOUD_ENABLED:
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
        else:
            nan_arr = np.full(n_15m, np.nan, dtype=np.float64)
            bool_arr = np.zeros(n_15m, dtype=bool)
            results['ichimoku_cloud_upper'] = nan_arr.copy()
            results['ichimoku_cloud_lower'] = nan_arr.copy()
            results['ichimoku_future_green'] = bool_arr.copy()
            results['ichimoku_future_red'] = bool_arr.copy()

        # ── Tenkan/Kijun (23/65 — TK guard + Tenkan/Kijun cross alerts only) ──
        # Independent of the cloud's own conversion/base periods above.
        if cfg.ICHIMOKU_TK_GUARD_ENABLED or cfg.ENABLE_TK_CONVERSION_CROSS or cfg.ENABLE_KIJUN_CROSS:
            _, hh_tk_conv = rolling_min_max_numba(data_15m["high"], cfg.ICHIMOKU_TK_CONVERSION_PERIODS)
            ll_tk_conv, _ = rolling_min_max_numba(data_15m["low"], cfg.ICHIMOKU_TK_CONVERSION_PERIODS)
            results['ichimoku_conversion_line'] = (hh_tk_conv + ll_tk_conv) / 2.0

            _, hh_tk_base = rolling_min_max_numba(data_15m["high"], cfg.ICHIMOKU_TK_BASE_PERIODS)
            ll_tk_base, _ = rolling_min_max_numba(data_15m["low"], cfg.ICHIMOKU_TK_BASE_PERIODS)
            results['ichimoku_base_line'] = (hh_tk_base + ll_tk_base) / 2.0
        else:
            nan_arr_tk = np.full(n_15m, np.nan, dtype=np.float64)
            results['ichimoku_conversion_line'] = nan_arr_tk.copy()
            results['ichimoku_base_line'] = nan_arr_tk.copy()

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

        if cfg.ENABLE_VWAP:
            results['vwap_gate'] = calculate_vwap_numpy(
                data_15m["high"], data_15m["low"], close_15m,
                data_15m["volume"], data_15m["timestamp"]
            )
        else:
            results['vwap_gate'] = np.full(n_15m, np.nan, dtype=np.float64)

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
                    'rsi_guard_ema', 'volume_ema', 'rma_cloud_fast_15', 'vwap_gate'):
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
                data_15m["volume"], data_15m["timestamp"]
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

def _normalize_samples(raw: List[Any]) -> List[List[float]]:
    """Accepts either the new [ts, value] format or the old bare-float format
    (pre-migration Redis payloads). Old entries get ts=0 so the staleness
    prune below drops them on the next run instead of crashing on them."""
    out: List[List[float]] = []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            out.append([item[0], item[1]])
        elif isinstance(item, (int, float)):
            out.append([0, item])
    return out

def _prune_stale_samples(samples: List[List[float]], max_age_sec: int, now_ts: int) -> List[List[float]]:
    if max_age_sec <= 0:
        return samples
    return [s for s in samples if (now_ts - s[0]) <= max_age_sec]

def _percentile_rank(value: float, history: List[float]) -> float:
    if not history:
        return 0.5
    less = sum(1 for h in history if h < value)
    equal = sum(1 for h in history if h == value)
    return (less + 0.5 * equal) / len(history)

@dataclass(slots=True)
class OrderBlock:
    index: int
    top: float
    bottom: float
    is_demand: bool
    is_internal: bool = False 

def find_pine_ob(h, l, c, threshold_arr, break_idx, pivot_idx, is_bullish_break):
    best_val = -np.inf if not is_bullish_break else np.inf
    best_idx = -1

    # range(1, break_idx - pivot_idx)  →  1 .. (break_idx-pivot_idx-1) inclusive
    for i in range(1, break_idx - pivot_idx):
        bar_idx = break_idx - i
        candle_range = h[bar_idx] - l[bar_idx]

        threshold = threshold_arr[bar_idx] * 2.0
        if np.isnan(threshold) or candle_range >= threshold:
            continue

        if not is_bullish_break:          # Bearish OB  →  find highest high
            if h[bar_idx] > best_val:
                best_val = h[bar_idx]
                best_idx = bar_idx
        else:                             # Bullish OB  →  find lowest low
            if l[bar_idx] < best_val:
                best_val = l[bar_idx]
                best_idx = bar_idx

    if best_idx != -1:
        return OrderBlock(
            index=best_idx,
            top=h[best_idx],
            bottom=l[best_idx],
            is_demand=is_bullish_break,
            is_internal=False,            # caller overrides
        )
    return None

def calculate_pine_order_blocks(o, h, l, c, atr200, ob_filter: str = 'Atr', swing_len: int = 50, internal_len: int = 5, filter_confluence: bool = False, show_iob: bool = True, iob_showlast: int = 5, show_ob: bool = True, ob_showlast: int = 5):
    """
    Faithful Python port of the Pine Order Block v5 logic.
    """
    n = len(h)
    if n < max(swing_len, internal_len) + 2:
        return [], []
    bar_idx = np.arange(n, dtype=np.float64)
    ranges = h - l
    with np.errstate(divide='ignore', invalid='ignore'):
        cmr_arr = np.cumsum(ranges) / np.maximum(1.0, bar_idx)
    if n > 0:
        cmr_arr[0] = ranges[0]

    threshold_arr = atr200 if ob_filter == 'Atr' else cmr_arr

    def get_swings(length):
        tops = []
        btms = []
        os = 0
        _, upper_arr = rolling_min_max_numba(h, length)
        lower_arr, _ = rolling_min_max_numba(l, length)

        for i in range(length, n):
            upper = upper_arr[i]
            lower = lower_arr[i]

            h_len = h[i - length]
            l_len = l[i - length]

            prev_os = os
            if h_len > upper:
                os = 0
            elif l_len < lower:
                os = 1

            if os == 0 and prev_os != 0:
                tops.append((i, h_len, i - length))
            elif os == 1 and prev_os != 1:
                btms.append((i, l_len, i - length))
        return tops, btms

    swing_tops, swing_btms = get_swings(swing_len)
    int_tops, int_btms = get_swings(internal_len)

    swing_top_map = {idx: (price, piv_idx) for idx, price, piv_idx in swing_tops}
    swing_btm_map = {idx: (price, piv_idx) for idx, price, piv_idx in swing_btms}
    int_top_map     = {idx: (price, piv_idx) for idx, price, piv_idx in int_tops}
    int_btm_map     = {idx: (price, piv_idx) for idx, price, piv_idx in int_btms}

    active_iobs = []
    active_obs  = []

    top_y = np.nan;  top_x = -1;   top_cross = True
    btm_y = np.nan;  btm_x = -1;   btm_cross = True
    itop_y = np.nan; itop_x = -1;  itop_cross = True
    ibtm_y = np.nan; ibtm_x = -1;  ibtm_cross = True

    # --- Concordant filters (Pine: bull_concordant / bear_concordant) --------
    def _bull_concordant(idx: int) -> bool:
        if not filter_confluence:
            return True
        # Pine: high - math.max(close, open) > math.min(close, open) - low
        upper_wick = h[idx] - max(c[idx], o[idx])
        lower_wick = min(c[idx], o[idx]) - l[idx]
        return upper_wick > lower_wick

    def _bear_concordant(idx: int) -> bool:
        if not filter_confluence:
            return True
        # Pine: high - math.max(close, open) < math.min(close, open) - low
        upper_wick = h[idx] - max(c[idx], o[idx])
        lower_wick = min(c[idx], o[idx]) - l[idx]
        return upper_wick < lower_wick

    # --- Main bar loop -------------------------------------------------------
    for i in range(1, n):
        # Update confirmed swings
        if i in swing_top_map:
            top_y, top_x = swing_top_map[i]
            top_cross = True
        if i in swing_btm_map:
            btm_y, btm_x = swing_btm_map[i]
            btm_cross = True
        if i in int_top_map:
            itop_y, itop_x = int_top_map[i]
            itop_cross = True
        if i in int_btm_map:
            ibtm_y, ibtm_x = int_btm_map[i]
            ibtm_cross = True

        c_curr = c[i]
        c_prev = c[i - 1]

        # ---- Internal Bullish Structure Break (crossover close, itop_y) ----
        if (show_iob and not np.isnan(itop_y) and itop_cross
                and c_curr > itop_y and c_prev <= itop_y):
            if not np.isnan(top_y) and top_y == itop_y:
                pass                                   # Pine: top_y != itop_y
            elif _bull_concordant(i):
                itop_cross = False
                ob = find_pine_ob(h, l, c, threshold_arr, i, itop_x,
                                  is_bullish_break=True)
                if ob:
                    ob.is_internal = True
                    active_iobs.append(ob)

        # ---- Internal Bearish Structure Break (crossunder close, ibtm_y) ---
        if (show_iob and not np.isnan(ibtm_y) and ibtm_cross
                and c_curr < ibtm_y and c_prev >= ibtm_y):
            if not np.isnan(btm_y) and btm_y == ibtm_y:
                pass
            elif _bear_concordant(i):
                ibtm_cross = False
                ob = find_pine_ob(h, l, c, threshold_arr, i, ibtm_x,
                                  is_bullish_break=False)
                if ob:
                    ob.is_internal = True
                    active_iobs.append(ob)

        # ---- Swing Bullish Structure Break ---------------------------------
        if (show_ob and not np.isnan(top_y) and top_cross
                and c_curr > top_y and c_prev <= top_y):
            top_cross = False
            ob = find_pine_ob(h, l, c, threshold_arr, i, top_x,
                              is_bullish_break=True)
            if ob:
                ob.is_internal = False
                active_obs.append(ob)

        # ---- Swing Bearish Structure Break ---------------------------------
        if (show_ob and not np.isnan(btm_y) and btm_cross
                and c_curr < btm_y and c_prev >= btm_y):
            btm_cross = False
            ob = find_pine_ob(h, l, c, threshold_arr, i, btm_x,
                              is_bullish_break=False)
            if ob:
                ob.is_internal = False
                active_obs.append(ob)

        # ---- Invalidate broken Internal OBs --------------------------------
        surviving_iobs = []
        for ob in active_iobs:
            if ob.is_demand and c_curr < ob.bottom:
                continue          # Bullish IOB broken
            if not ob.is_demand and c_curr > ob.top:
                continue          # Bearish IOB broken
            surviving_iobs.append(ob)
        active_iobs = surviving_iobs

        # ---- Invalidate broken Swing OBs -----------------------------------
        surviving_obs = []
        for ob in active_obs:
            if ob.is_demand and c_curr < ob.bottom:
                continue
            if not ob.is_demand and c_curr > ob.top:
                continue
            surviving_obs.append(ob)
        active_obs = surviving_obs

    # Return the most recent N active zones (Pine: show_last)
    final_iobs = active_iobs[-iob_showlast:] if iob_showlast > 0 else []
    final_obs  = active_obs[-ob_showlast:]   if ob_showlast  > 0 else []
    return final_iobs, final_obs

def _order_block_gate_reason(o, h, l, c, atr_short_arr, i15, cfg_obj):
    atr200 = calculate_atr_rma(h, l, c, 200)
    
    active_iobs, active_obs = calculate_pine_order_blocks(
        o[:i15 + 1], h[:i15 + 1], l[:i15 + 1], c[:i15 + 1], atr200[:i15 + 1],
        swing_len=cfg_obj.OB_LOOKBACK_CANDLES,
        internal_len=5,
        filter_confluence=cfg_obj.OB_FILTER_CONFLUENCE,
        iob_showlast=10, 
        ob_showlast=10,
    )
    zones = active_obs + active_iobs
    equilibrium = None
    lookback = cfg_obj.OB_LOOKBACK_CANDLES
    if cfg_obj.ENABLE_OB_PREMIUM_DISCOUNT_FILTER and i15 > lookback:
        range_high = np.max(h[i15 - lookback:i15])
        range_low  = np.min(l[i15 - lookback:i15])
        equilibrium = (range_high + range_low) / 2.0
        
    ob_ok_buy = ob_ok_sell = None
    reason_buy = reason_sell = None
    grace = cfg_obj.OB_CONFIRM_LOOKAHEAD_CANDLES
    persistence = getattr(cfg_obj, "OB_PERSISTENCE_CANDLES", 2)

    # Track recent confirmations so the gate stays valid for a couple of candles
    last_buy_confirm_idx = -1
    last_sell_confirm_idx = -1
    last_buy_zone = None
    last_sell_zone = None
    
    for z in zones:
        if equilibrium is not None:
            if z.is_demand and z.top >= equilibrium:
                continue
            if not z.is_demand and z.bottom <= equilibrium:
                continue

        confirm_end = min(z.index + cfg_obj.OB_IMPULSE_LOOKAHEAD, i15 - 1)
        test_start = confirm_end + 1
        if test_start > i15:
            continue

        touch_idx = -1
        zone_dead = False
        confirmed_idx = -1
        for idx in range(test_start, i15 + 1):
            touched = (l[idx] <= z.top) and (h[idx] >= z.bottom)
            if touch_idx == -1:
                if not touched:
                    continue
                touch_idx = idx
            elif idx - touch_idx > grace:
                zone_dead = True
                break
            atr_idx = atr_short_arr[idx]
            min_penetration = (
                cfg_obj.OB_MIN_PENETRATION_ATR_MULT * atr_idx
                if cfg_obj.OB_MIN_PENETRATION_ATR_MULT > 0 and not np.isnan(atr_idx) and atr_idx > 0
                else 0.0
            )
            if z.is_demand:
                if c[idx] > z.top + min_penetration:
                    confirmed_idx = idx
                    break
                if c[idx] < z.bottom:
                    zone_dead = True
                    break
            else:
                if c[idx] < z.bottom - min_penetration:
                    confirmed_idx = idx
                    break
                if c[idx] > z.top:
                    zone_dead = True
                    break

        if touch_idx == -1 or zone_dead:
            continue

        # Compute prior leg for any confirmation (current or recent)
        leg_ref = touch_idx - 1
        zone_prior_leg = _prior_leg_direction(
            c, h, l, leg_ref, Constants.REVERSAL_PRIOR_LEG_LOOKBACK
        )

        if confirmed_idx == i15:
            if z.is_demand and zone_prior_leg == -1:
                ob_ok_buy = True
                zone_type = "Internal" if z.is_internal else "Swing"
                reason_buy = (
                    f"Pine {zone_type} Demand OB {z.bottom:.4g}-{z.top:.4g} (idx {z.index}) "
                    f"reversed after down-leg, touched idx {touch_idx}"
                )
            elif (not z.is_demand) and zone_prior_leg == 1:
                ob_ok_sell = True
                zone_type = "Internal" if z.is_internal else "Swing"
                reason_sell = (
                    f"Pine {zone_type} Supply OB {z.bottom:.4g}-{z.top:.4g} (idx {z.index}) "
                    f"reversed after up-leg, touched idx {touch_idx}"
                )
            else:
                zone_type = "Internal" if z.is_internal else "Swing"
                logger.debug(
                    f"[{PAIR_ID.get() or '?'}] Pine {zone_type} OB "
                    f"{'demand' if z.is_demand else 'supply'} "
                    f"idx {z.index} closed beyond edge but prior leg is {zone_prior_leg} "
                    f"(need {-1 if z.is_demand else 1}) — vote withheld"
                )

        elif confirmed_idx != -1:
            # Remember recent confirmation for persistence across subsequent candles
            if z.is_demand and zone_prior_leg == -1:
                if confirmed_idx > last_buy_confirm_idx:
                    last_buy_confirm_idx = confirmed_idx
                    last_buy_zone = z
            elif not z.is_demand and zone_prior_leg == 1:
                if confirmed_idx > last_sell_confirm_idx:
                    last_sell_confirm_idx = confirmed_idx
                    last_sell_zone = z
            continue     
        else:
            zone_type = "Internal" if z.is_internal else "Swing"
            if z.is_demand and ob_ok_buy is not True:
                ob_ok_buy = False
                reason_buy = (
                    f"Pine {zone_type} Demand OB {z.bottom:.4g}-{z.top:.4g} (idx {z.index}) "
                    f"touched idx {touch_idx}, awaiting reversal ({i15 - touch_idx}/{grace})"
                )
            elif not z.is_demand and ob_ok_sell is not True:
                ob_ok_sell = False
                reason_sell = (
                    f"Pine {zone_type} Supply OB {z.bottom:.4g}-{z.top:.4g} (idx {z.index}) "
                    f"touched idx {touch_idx}, awaiting reversal ({i15 - touch_idx}/{grace})"
                )

    # Persistence: if no current confirmation, allow recent ones to stay valid
    if ob_ok_buy is not True and last_buy_confirm_idx >= 0 and (i15 - last_buy_confirm_idx) <= persistence:
        ob_ok_buy = True
        zone_type = "Internal" if last_buy_zone.is_internal else "Swing"
        reason_buy = (
            f"Pine {zone_type} Demand OB {last_buy_zone.bottom:.4g}-{last_buy_zone.top:.4g} (idx {last_buy_zone.index}) "
            f"confirmed idx {last_buy_confirm_idx}, still valid ({i15 - last_buy_confirm_idx}/{persistence})"
        )

    if ob_ok_sell is not True and last_sell_confirm_idx >= 0 and (i15 - last_sell_confirm_idx) <= persistence:
        ob_ok_sell = True
        zone_type = "Internal" if last_sell_zone.is_internal else "Swing"
        reason_sell = (
            f"Pine {zone_type} Supply OB {last_sell_zone.bottom:.4g}-{last_sell_zone.top:.4g} (idx {last_sell_zone.index}) "
            f"confirmed idx {last_sell_confirm_idx}, still valid ({i15 - last_sell_confirm_idx}/{persistence})"
        )

    reason = reason_buy if ob_ok_buy else (reason_sell if ob_ok_sell else (reason_buy or reason_sell))
    logger.debug(
        f"[{PAIR_ID.get() or '?'}] Pine OB diag | zones found: {len(zones)} | "
        f"equilibrium={'%.4f' % equilibrium if equilibrium is not None else 'N/A'} | "
        f"ob_ok_buy={ob_ok_buy} ob_ok_sell={ob_ok_sell} | "
        f"reason={reason or 'no zone touched/confirmed'}"
    )
    return ob_ok_buy, ob_ok_sell, reason

def _get_minor_swings(h: np.ndarray, l: np.ndarray, length: int, start: int, end: int):
    tops: List[Tuple[int, float, int]] = []
    btms: List[Tuple[int, float, int]] = []
    n = len(h)
    if n == 0 or length < 1:
        return tops, btms
    end = min(end, n - 1)
    if end < length:
        return tops, btms

    _, upper_arr = rolling_min_max_numba(h, length)
    lower_arr, _ = rolling_min_max_numba(l, length)

    os = None
    for i in range(length, end + 1):
        k = i - length
        right_upper = upper_arr[i]
        right_lower = lower_arr[i]
        h_len = h[k]
        l_len = l[k]
        if np.isnan(right_upper) or np.isnan(right_lower) or np.isnan(h_len) or np.isnan(l_len):
            continue
        # Symmetric check: candidate pivot must also beat the `length` bars to
        # its LEFT, matching Pine's ta.pivothigh(length, length) / pivotlow.
        if k - 1 < 0 or np.isnan(upper_arr[k - 1]) or np.isnan(lower_arr[k - 1]):
            continue
        left_upper = upper_arr[k - 1]
        left_lower = lower_arr[k - 1]

        prev_os = os
        if h_len > right_upper and h_len > left_upper:
            os = 0
        elif l_len < right_lower and l_len < left_lower:
            os = 1
        else:
            os = prev_os

        if i < start:
            continue
        if os == 0 and prev_os != 0:
            tops.append((i, h_len, k))
        elif os == 1 and prev_os != 1:
            btms.append((i, l_len, k))
    return tops, btms

def _bullish_fvg_at(h: np.ndarray, l: np.ndarray, k: int) -> bool:
    """Direction-specific 3-candle Fair Value Gap ending at k: low[k] > high[k-2]."""
    if k < 2:
        return False
    if np.isnan(h[k - 2]) or np.isnan(l[k]):
        return False
    return l[k] > h[k - 2]

def _bearish_fvg_at(h: np.ndarray, l: np.ndarray, k: int) -> bool:
    """Direction-specific 3-candle Fair Value Gap ending at k: high[k] < low[k-2]."""
    if k < 2:
        return False
    if np.isnan(l[k - 2]) or np.isnan(h[k]):
        return False
    return h[k] < l[k - 2]

def _choch_poi_tap(o, h, l, c, br, sweep_idx, is_buy, cfg_obj):
    try:
        atr200 = calculate_atr_rma(h[:br + 1], l[:br + 1], c[:br + 1], 200)
        active_iobs, active_obs = calculate_pine_order_blocks(
            o[:br + 1], h[:br + 1], l[:br + 1], c[:br + 1], atr200,
            swing_len=cfg_obj.OB_LOOKBACK_CANDLES,
            internal_len=5,
            filter_confluence=cfg_obj.OB_FILTER_CONFLUENCE,
            iob_showlast=10, ob_showlast=10,
        )
    except Exception:
        return False
    for z in (active_obs + active_iobs):
        if z.is_demand != is_buy:
            continue
        for idx in range(sweep_idx, br + 1):
            if l[idx] <= z.top and h[idx] >= z.bottom:
                return True
    return False

def _choch_gate_reason(o, h, l, c, ts, atr_short_arr, i15, cfg_obj):
    pair = PAIR_ID.get() or "?"
    n = len(c)
    length = cfg_obj.CHOCH_SWING_LEN
    lookback = cfg_obj.CHOCH_LOOKBACK_CANDLES
    window = cfg_obj.CHOCH_CONFIRM_WINDOW_CANDLES
    same_candle_ok = cfg_obj.CHOCH_ALLOW_SAME_CANDLE_SWEEP
    persistence = cfg_obj.CHOCH_PERSISTENCE_CANDLES
    min_body_ratio = cfg_obj.CHOCH_MIN_DISPLACEMENT_BODY_RATIO
    check_poi = getattr(cfg_obj, "CHOCH_CHECK_POI_TAP", False)

    if i15 is None or i15 >= n or i15 < length + 2:
        return None, None, None, False, False, False, False

    scan_start = max(length, i15 - lookback - persistence)

    tops, btms = _get_minor_swings(h, l, length, scan_start, i15)

    def _nearest_before(pivots, before_idx):
        for (idx, price, piv_idx) in reversed(pivots):
            if idx < before_idx:
                return idx, price, piv_idx
        return None

    def _continuity_ok(start_idx, end_idx) -> bool:
        """Verify no timestamp gaps in [start_idx, end_idx]."""
        for idx in range(start_idx + 1, end_idx + 1):
            if (ts[idx] - ts[idx - 1]) != 900:
                logger.debug(
                    f"[{pair}] CHoCH: candle gap at idx {idx} "
                    f"({ts[idx]} - {ts[idx-1]} != 900)"
                )
                return False
        return True

    def _scan(is_buy: bool):
        sweep_pivots = btms if is_buy else tops
        found_entry = None
        found_sweep_idx = None
        found_sweep_price = None
        found_reason = None
        found_fvg = False

        for r in range(i15, scan_start - 1, -1):
            # Localized NaN/Inf guard — a NaN candle simply fails to match
            if not (np.isfinite(o[r]) and np.isfinite(h[r]) and
                    np.isfinite(l[r]) and np.isfinite(c[r]) and
                    np.isfinite(atr_short_arr[r])):
                continue

            rng = h[r] - l[r]
            if rng <= 1e-12:
                continue

            body = (c[r] - o[r]) if is_buy else (o[r] - c[r])
            if body <= 0 or (body / rng) < min_body_ratio:
                continue

            sweep_hi = r if same_candle_ok else r - 1
            sweep_lo = max(scan_start, r - window)
            sweep_idx = None
            sweep_level_price = None
            for k in range(sweep_hi, sweep_lo - 1, -1):
                if k < 0:
                    break
                if not (np.isfinite(l[k]) and np.isfinite(h[k]) and
                        np.isfinite(c[k]) and np.isfinite(atr_short_arr[k])):
                    continue

                pivot = _nearest_before(sweep_pivots, k)
                if pivot is None:
                    continue
                _, level_price, _ = pivot

                min_sweep_dist = cfg_obj.CHOCH_MIN_SWEEP_DISTANCE_ATR * atr_short_arr[k]
                if is_buy:
                    wick_swept = l[k] < level_price - min_sweep_dist
                else:
                    wick_swept = h[k] > level_price + min_sweep_dist
                if not wick_swept:
                    continue

                recovered = (c[r] > level_price) if is_buy else (c[r] < level_price)
                if not recovered:
                    continue

                sweep_idx = k
                sweep_level_price = level_price
                break

            if sweep_idx is None:
                continue
            if not same_candle_ok and r == sweep_idx:
                continue

            # Continuity: sweep must flow into the displacement candle without gaps
            if not _continuity_ok(sweep_idx, r):
                continue

            fvg_fn = _bullish_fvg_at if is_buy else _bearish_fvg_at
            has_fvg = any(
                fvg_fn(h, l, k)
                for k in range(max(sweep_idx, 2), r + 1)
            )
            if cfg_obj.CHOCH_REQUIRE_FVG and not has_fvg:
                continue

            found_entry = r
            found_sweep_idx = sweep_idx
            found_sweep_price = sweep_level_price
            found_fvg = has_fvg
            found_reason = (
                f"{'Bullish' if is_buy else 'Bearish'} CHoCH (early/displacement): swept "
                f"{'low' if is_buy else 'high'} {sweep_level_price:.4g} @idx{sweep_idx}, "
                f"displacement close {c[r]:.4g} @idx{r}" + (" | FVG" if has_fvg else "")
            )
            break

        if found_entry is None:
            return None, None, False, False

        # POI tap (bonus only)
        poi_tap = False
        if check_poi:
            poi_tap = _choch_poi_tap(o, h, l, c, found_entry, found_sweep_idx, is_buy, cfg_obj)
            if poi_tap:
                found_reason = f"{found_reason} | POI tap"

        age = i15 - found_entry
        if age == 0:
            return True, found_reason, found_fvg, poi_tap

        if age > persistence:
            return False, (
                f"{found_reason} (stale, {age} candles old > persistence {persistence})"
            ), found_fvg, poi_tap

        if not _continuity_ok(found_entry, i15):
            return False, (
                f"{found_reason} (data gap between entry @{found_entry} and now @{i15})"
            ), found_fvg, poi_tap

        if is_buy and c[i15] <= found_sweep_price:
            return False, (
                f"Bullish CHoCH stale: close {c[i15]:.4g} back below swept level {found_sweep_price:.4g}"
            ), found_fvg, poi_tap

        if not is_buy and c[i15] >= found_sweep_price:
            return False, (
                f"Bearish CHoCH stale: close {c[i15]:.4g} back above swept level {found_sweep_price:.4g}"
            ), found_fvg, poi_tap

        return True, found_reason, found_fvg, poi_tap

    choch_ok_buy, reason_buy, fvg_buy, poi_tap_buy = _scan(is_buy=True)
    choch_ok_sell, reason_sell, fvg_sell, poi_tap_sell = _scan(is_buy=False)

    reason = (
        reason_buy if choch_ok_buy
        else (reason_sell if choch_ok_sell else (reason_buy or reason_sell))
    )
    logger.debug(
        f"[{pair}] CHoCH diag | ok_buy={choch_ok_buy} ok_sell={choch_ok_sell} | "
        f"reason={reason or 'no qualifying sweep+break structure found'}"
    )
    return (
        choch_ok_buy, choch_ok_sell, reason,
        fvg_buy, fvg_sell, poi_tap_buy, poi_tap_sell
    )

def _oi_price_divergence_reason(oi_now: float, oi_history: List[List[float]], price_now: Optional[float], price_history: List[List[float]], is_buy: bool) -> Optional[str]:
    if not cfg.ENABLE_OI_PRICE_DIVERGENCE or price_now is None:
        return None
    lookback = cfg.OI_DIVERGENCE_LOOKBACK_SAMPLES
    if len(oi_history) < lookback or len(price_history) < lookback:
        return None

    oi_ref = oi_history[-lookback][1]
    price_ref = price_history[-lookback][1]
    if oi_ref is None or price_ref is None or oi_ref <= 0 or price_ref <= 0:
        return None

    oi_roc_pct = (oi_now - oi_ref) / oi_ref * 100.0
    price_roc_pct = (price_now - price_ref) / price_ref * 100.0

    if oi_roc_pct > -cfg.OI_DIVERGENCE_MIN_OI_FALL_PCT:
        return None  # OI isn't falling with conviction — no divergence signal

    if is_buy and price_roc_pct >= cfg.OI_DIVERGENCE_MIN_PRICE_ROC_PCT:
        return (
            f"OI/price divergence: BUY blocked | price {price_ref:.4g}→{price_now:.4g} "
            f"({price_roc_pct:+.2f}%) rising on falling OI ({oi_roc_pct:+.2f}%) — "
            f"looks like short-covering, not new demand"
        )
    if not is_buy and price_roc_pct <= -cfg.OI_DIVERGENCE_MIN_PRICE_ROC_PCT:
        return (
            f"OI/price divergence: SELL blocked | price {price_ref:.4g}→{price_now:.4g} "
            f"({price_roc_pct:+.2f}%) falling on falling OI ({oi_roc_pct:+.2f}%) — "
            f"looks like long-liquidation, not new supply"
        )
    return None

def _oi_funding_gate_reason(oi_now: Optional[float], oi_history: List[List[float]], funding: Optional[float], funding_history: List[List[float]], is_buy: bool,
    oi_usd_now: Optional[float] = None,
    price_now: Optional[float] = None, price_history: Optional[List[List[float]]] = None) -> Optional[str]:
    if oi_now is None:
        return None
    if cfg.MIN_OI_USD > 0 and (oi_usd_now is None or oi_usd_now < cfg.MIN_OI_USD):
        return None 
    if len(oi_history) < cfg.MIN_OI_FUNDING_SAMPLES:
        return None

    divergence_reason = _oi_price_divergence_reason(oi_now, oi_history, price_now, price_history or [], is_buy)
    if divergence_reason is not None:
        return divergence_reason

    oi_values = [v for _, v in oi_history if v is not None]
    if len(oi_values) < cfg.MIN_OI_FUNDING_SAMPLES:
        return None

    ref_n = min(cfg.OI_DELTA_REF_SAMPLES, len(oi_values))
    recent_ref = oi_values[-ref_n:]
    prev_oi = sum(recent_ref) / len(recent_ref)
    oi_delta = oi_now - prev_oi

    delta_history = [abs(b - a) for a, b in zip(oi_values[:-1], oi_values[1:])]
    adaptive_min_rise = 0.0
    if delta_history:
        sorted_deltas = sorted(delta_history)
        idx = min(int(cfg.OI_RISING_PERCENTILE * len(sorted_deltas)), len(sorted_deltas) - 1)
        adaptive_min_rise = sorted_deltas[idx]
    oi_rising = oi_delta > adaptive_min_rise

    not_crowded = True
    funding_pctile = None
    funding_values = [v for _, v in funding_history if v is not None]
    if funding is not None and len(funding_values) >= cfg.MIN_OI_FUNDING_SAMPLES:
        funding_pctile = _percentile_rank(funding, funding_values)
        above_floor = abs(funding) >= cfg.FUNDING_ABS_FLOOR
        if is_buy:
            crowded = above_floor and funding > 0 and funding_pctile >= cfg.FUNDING_CROWDED_PERCENTILE
        else:
            crowded = above_floor and funding < 0 and funding_pctile <= (1.0 - cfg.FUNDING_CROWDED_PERCENTILE)
        not_crowded = not crowded

    if oi_rising or not_crowded:
        return None

    side = "BUY" if is_buy else "SELL"
    fp_str = f"{funding_pctile:.2f}" if funding_pctile is not None else "n/a"
    return (
        f"OI/funding gate: {side} blocked | OI {ref_n}-smpl avg {prev_oi:.0f}→{oi_now:.0f} "
        f"(delta {oi_delta:+.0f} vs pair's own adaptive min {adaptive_min_rise:.0f}) "
        f"| funding {funding:.4f} self-pctile={fp_str} (crowded vs own history)"
    )

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

_PCTL_RANK_CACHE_MAXSIZE = 256
_pctl_rank_cache: "OrderedDict[tuple, Optional[float]]" = OrderedDict()
_PCTL_CACHE_MISS = object()

def _array_percentile_rank(arr: np.ndarray, i: int, lookback: int, min_history: int,
                             allow_zero: bool = False) -> Optional[float]:
    start = i - lookback
    if start < 0:
        return None

    current = arr[i]
    cache_key = (id(arr), i, lookback, min_history, allow_zero, current)
    cached = _pctl_rank_cache.get(cache_key, _PCTL_CACHE_MISS)
    if cached is not _PCTL_CACHE_MISS:
        _pctl_rank_cache.move_to_end(cache_key)
        return cached
    raw = percentile_rank_numba(arr, int(i), int(lookback), int(min_history), bool(allow_zero))
    result = None if np.isnan(raw) else float(raw)

    _pctl_rank_cache[cache_key] = result
    if len(_pctl_rank_cache) > _PCTL_RANK_CACHE_MAXSIZE:
        _pctl_rank_cache.popitem(last=False)
    return result

def get_atr_percentile(atr_long_arr: np.ndarray, i15: int, cfg: BotConfig) -> Optional[float]:
    """Percentile rank (0.0=calmest .. 1.0=most volatile) of current long-ATR
    against its trailing ATR_PCTL_LOOKBACK window. None if insufficient history."""
    return _array_percentile_rank(atr_long_arr, i15, cfg.ATR_PCTL_LOOKBACK, cfg.ATR_PCTL_MIN_HISTORY)

def get_adx_percentile(adx_arr: np.ndarray, i15: int, cfg: BotConfig) -> Optional[float]:
    """Percentile rank (0.0=weakest .. 1.0=strongest) of current ADX against its
    trailing ATR_PCTL_LOOKBACK window (reuses the same lookback as ATR for consistency)."""
    return _array_percentile_rank(adx_arr, i15, cfg.ATR_PCTL_LOOKBACK, cfg.ATR_PCTL_MIN_HISTORY, allow_zero=True)

def get_volume_percentile(volume_arr: np.ndarray, i15: int, cfg: BotConfig) -> Optional[float]:
    """Percentile rank (0.0=quietest .. 1.0=busiest) of current volume against its
    trailing VOLUME_PCTL_LOOKBACK window."""
    return _array_percentile_rank(volume_arr, i15, cfg.VOLUME_PCTL_LOOKBACK, cfg.VOLUME_PCTL_MIN_HISTORY)

def _scale_by_pctl(pctl: Optional[float], calm: float, volatile: float, fallback_pctl: float = 0.5) -> float:
    """Linearly scales a [calm, volatile] range by pctl. Falls back to fallback_pctl
    (midpoint by default) when pctl is unavailable, so callers always get a usable value."""
    p = pctl if pctl is not None else fallback_pctl
    val = calm + p * (volatile - calm)
    lo, hi = min(calm, volatile), max(calm, volatile)
    return max(lo, min(hi, val))

def get_adaptive_rvol_threshold(atr_long_arr: np.ndarray, i15: int, cfg: BotConfig,
                                 pctl: Optional[float] = None) -> Optional[float]:
    if not cfg.ATR_ADAPTIVE_ENABLED:
        return None
    if pctl is None:
        pctl = _get_smoothed_pctl(atr_long_arr, i15, cfg)
    if pctl is None:
        return None
    return _scale_by_pctl(pctl, cfg.ADAPTIVE_MULT_CALM, cfg.ADAPTIVE_MULT_VOLATILE)

_ADX_THRESH_CACHE_MAXSIZE = 256
_adx_thresh_cache: "OrderedDict[tuple, float]" = OrderedDict()
_ADX_CACHE_MISS = object()

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

    current = adx_arr[i15] if i15 < len(adx_arr) else np.nan
    cache_key = (id(adx_arr), i15, lookback, cfg.ADX_ADAPTIVE_TARGET_PCTL,
                 cfg.ADX_ADAPTIVE_BAND_WIDTH, current)
    cached = _adx_thresh_cache.get(cache_key, _ADX_CACHE_MISS)
    if cached is not _ADX_CACHE_MISS:
        _adx_thresh_cache.move_to_end(cache_key)
        return cached

    window = adx_arr[start:i15]
    valid = window[~np.isnan(window)]
    if len(valid) < cfg.ATR_PCTL_MIN_HISTORY:
        result = cfg.ADX_ADAPTIVE_FALLBACK
    else:
        sorted_valid = np.sort(valid)
        n = len(sorted_valid)

        band_half = cfg.ADX_ADAPTIVE_BAND_WIDTH / 2.0
        if band_half <= 0.0:
            idx = int(cfg.ADX_ADAPTIVE_TARGET_PCTL / 100.0 * (n - 1))
            idx = max(0, min(n - 1, idx))
            result = float(sorted_valid[idx])
        else:
            lo_pctl = max(0.0, cfg.ADX_ADAPTIVE_TARGET_PCTL - band_half)
            hi_pctl = min(100.0, cfg.ADX_ADAPTIVE_TARGET_PCTL + band_half)
            lo_idx = max(0, int(lo_pctl / 100.0 * (n - 1)))
            hi_idx = min(n - 1, int(hi_pctl / 100.0 * (n - 1)))
            band = sorted_valid[lo_idx:hi_idx + 1]
            result = float(np.median(band))

    _adx_thresh_cache[cache_key] = result
    if len(_adx_thresh_cache) > _ADX_THRESH_CACHE_MAXSIZE:
        _adx_thresh_cache.popitem(last=False)
    return result

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

def get_adaptive_threshold(atr_long_arr, i15, cfg, calm_attr: str, volatile_attr: str,
                            pctl: Optional[float] = None) -> float:
    if pctl is None:
        pctl = _get_smoothed_pctl(atr_long_arr, i15, cfg)
    return _scale_by_pctl(pctl, getattr(cfg, calm_attr), getattr(cfg, volatile_attr))

def get_adaptive_ppo_threshold(atr_long_arr, i15, cfg, pctl: Optional[float] = None) -> float:
    return get_adaptive_threshold(atr_long_arr, i15, cfg, "PPO_ADAPTIVE_CALM", "PPO_ADAPTIVE_VOLATILE", pctl=pctl)

def get_adaptive_rsi_thresholds(atr_long_arr: np.ndarray, i15: int, cfg: BotConfig,
                                 pctl: Optional[float] = None) -> Tuple[float, float]:
    if pctl is None:
        pctl = _get_smoothed_pctl(atr_long_arr, i15, cfg)
    buy = _scale_by_pctl(pctl, cfg.RSI_ADAPTIVE_BUY_CALM, cfg.RSI_ADAPTIVE_BUY_VOLATILE)
    sell = _scale_by_pctl(pctl, cfg.RSI_ADAPTIVE_SELL_CALM, cfg.RSI_ADAPTIVE_SELL_VOLATILE)
    return buy, sell

def get_adaptive_cpr_threshold(atr_long_arr, i15, cfg, pctl: Optional[float] = None) -> float:
    return get_adaptive_threshold(atr_long_arr, i15, cfg, "CPR_ADAPTIVE_CALM", "CPR_ADAPTIVE_VOLATILE", pctl=pctl)

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

def _prior_leg_direction(closes: np.ndarray, highs: np.ndarray, lows: np.ndarray,
                         end_idx: int, lookback: int) -> int:
    first = end_idx - lookback
    if end_idx < 0 or first < 0 or end_idx >= len(closes):
        return 0
    win_c = closes[first:end_idx + 1]
    win_h = highs[first:end_idx + 1]
    win_l = lows[first:end_idx + 1]
    if np.any(np.isnan(win_c)) or np.any(np.isnan(win_h)) or np.any(np.isnan(win_l)):
        return 0
    last_close = float(win_c[-1])
    if last_close <= 0:
        return 0
    avg_range = float(np.mean(win_h - win_l))
    if avg_range <= 1e-12:
        return 0
    if last_close > 0 and (avg_range / last_close) < 1e-5:  # Less than 0.001% average range
        return 0
    margin = Constants.REVERSAL_PRIOR_LEG_MIN_RANGE_MULT * avg_range
    hi = float(np.max(win_c))
    lo = float(np.min(win_c))
    drawdown = hi - last_close
    runup    = last_close - lo
    if drawdown >= margin and drawdown >= runup:
        return -1
    if runup >= margin and runup >= drawdown:
        return 1
    return 0

@dataclass
class TrendlineState:
    """A line through two confirmed same-type swing fractals.
    anchor2 is the older pivot, anchor1 the more recent one — matches the
    Pine script's recent_1/recent_2 naming in Auto_Trendline.txt."""
    slope: float
    intercept: float
    anchor1_idx: int
    anchor1_price: float
    anchor2_idx: int
    anchor2_price: float
    confirmed_at: int  # confirmation index of anchor1 (when the line became knowable, no lookahead)

def _get_trendline_state(h: np.ndarray, l: np.ndarray, length: int, i15: int,
                          is_buy: bool) -> Optional[TrendlineState]:
    tops, btms = _get_minor_swings(h, l, length, start=0, end=i15)
    pivots = btms if is_buy else tops
    if len(pivots) < 2:
        return None

    conf_idx1, price1, bar_idx1 = pivots[-1]
    _, price2, bar_idx2 = pivots[-2]

    if bar_idx1 == bar_idx2:
        return None
    if is_buy and not (price1 > price2):
        return None
    if not is_buy and not (price1 < price2):
        return None

    m = (price1 - price2) / (bar_idx1 - bar_idx2)
    b = price1 - m * bar_idx1

    return TrendlineState(
        slope=m, intercept=b,
        anchor1_idx=bar_idx1, anchor1_price=price1,
        anchor2_idx=bar_idx2, anchor2_price=price2,
        confirmed_at=conf_idx1,
    )

def _trendline_value_at(tl: TrendlineState, idx: int) -> float:
    """Project the trendline's y-value forward/backward to any bar index."""
    return tl.slope * idx + tl.intercept


def classify_trendline_interaction(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray,
                                    atr_short_arr: np.ndarray, tl: TrendlineState, idx: int,
                                    is_buy: bool, tolerance_atr: float,
                                    break_distance_atr: float, break_body_ratio: float) -> Optional[str]:
    if idx < 0 or idx >= len(c):
        return None
    line_val = _trendline_value_at(tl, idx)
    atr = atr_short_arr[idx] if idx < len(atr_short_arr) else np.nan
    if not np.isfinite(line_val) or not np.isfinite(atr) or atr <= 0:
        return None
    if np.isnan(o[idx]) or np.isnan(h[idx]) or np.isnan(l[idx]) or np.isnan(c[idx]):
        return None

    tol = tolerance_atr * atr
    break_dist = break_distance_atr * atr
    rng = h[idx] - l[idx]
    body = abs(c[idx] - o[idx])
    body_ratio = (body / rng) if rng > 1e-12 else 0.0

    if is_buy:
        # Ascending support line: price should stay above it.
        closed_through = c[idx] < (line_val - break_dist)
        if closed_through and body_ratio >= break_body_ratio:
            return "break"
        near = l[idx] <= (line_val + tol) and c[idx] >= (line_val - tol)
        if near:
            return "touch"
    else:
        # Descending resistance line: price should stay below it.
        closed_through = c[idx] > (line_val + break_dist)
        if closed_through and body_ratio >= break_body_ratio:
            return "break"
        near = h[idx] >= (line_val - tol) and c[idx] <= (line_val + tol)
        if near:
            return "touch"
    return None

def _tlr_evaluate_touch(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray,
                         atr_short_arr: np.ndarray, i15: int, cfg_obj, is_buy: bool,
                         prior_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    length = cfg_obj.TLR_FRACTAL_PERIODS
    tl = _get_trendline_state(h, l, length, i15, is_buy)
    if tl is None:
        return {"gate_ok": False, "reason": "no valid trendline", "tl": None, "state": prior_state}

    state = dict(prior_state) if prior_state else {}
    line_changed = (
        state.get("anchor1_idx") != tl.anchor1_idx or
        state.get("anchor2_idx") != tl.anchor2_idx
    )
    if line_changed:
        # A new fractal redefined the line -- start the touch count over,
        # regardless of whether the old line was previously marked broken.
        state = {
            "anchor1_idx": tl.anchor1_idx, "anchor2_idx": tl.anchor2_idx,
            "touch_count": 0, "last_touch_idx": -1, "broken": False,
        }

    if state.get("broken"):
        return {"gate_ok": False, "reason": "trendline previously broken, awaiting new line",
                "tl": tl, "state": state}

    result = classify_trendline_interaction(
        o, h, l, c, atr_short_arr, tl, i15, is_buy,
        cfg_obj.TLR_TOUCH_TOLERANCE_ATR, cfg_obj.TLR_BREAK_DISTANCE_ATR, cfg_obj.TLR_BREAK_BODY_RATIO,
    )

    if result == "break":
        state["broken"] = True
        return {"gate_ok": False, "reason": "trendline broken this bar", "tl": tl, "state": state}

    if result == "touch" and state.get("last_touch_idx") != i15:
        state["touch_count"] = state.get("touch_count", 0) + 1
        state["last_touch_idx"] = i15

    required = cfg_obj.TLR_REQUIRED_TOUCH_NUMBER
    gate_ok = (result == "touch" and state.get("touch_count") == required)
    reason = (
        f"touch #{state.get('touch_count')} of {required} required"
        if result == "touch" else "no touch this bar"
    )
    return {"gate_ok": gate_ok, "reason": reason, "tl": tl, "state": state}

def _tlr_fib_confluence(h: np.ndarray, l: np.ndarray, tl: TrendlineState, i15: int,
                         is_buy: bool, cfg_obj) -> bool:
    if tl is None:
        return False
    anchor_idx = tl.anchor2_idx
    if anchor_idx < 0 or anchor_idx >= i15:
        return False
    zone_lo_pct, zone_hi_pct = cfg_obj.TLR_FIB_ZONE_LOW, cfg_obj.TLR_FIB_ZONE_HIGH

    if is_buy:
        leg_low = tl.anchor2_price
        window = h[anchor_idx:i15 + 1]
        if window.size == 0 or np.all(np.isnan(window)):
            return False
        leg_high = float(np.nanmax(window))
        if leg_high <= leg_low:
            return False
        touch_price = l[i15]
        fib_hi = leg_high - zone_lo_pct * (leg_high - leg_low)
        fib_lo = leg_high - zone_hi_pct * (leg_high - leg_low)
    else:
        leg_high = tl.anchor2_price
        window = l[anchor_idx:i15 + 1]
        if window.size == 0 or np.all(np.isnan(window)):
            return False
        leg_low = float(np.nanmin(window))
        if leg_high <= leg_low:
            return False
        touch_price = h[i15]
        fib_lo = leg_low + zone_lo_pct * (leg_high - leg_low)
        fib_hi = leg_low + zone_hi_pct * (leg_high - leg_low)

    if not np.isfinite(touch_price):
        return False
    return bool(fib_lo <= touch_price <= fib_hi)

def _tlr_sr_confluence(h: np.ndarray, l: np.ndarray, atr_short_arr: np.ndarray, i15: int,
                        is_buy: bool, cfg_obj) -> bool:
    length = cfg_obj.TLR_FRACTAL_PERIODS
    lookback_start = max(0, i15 - cfg_obj.TLR_SR_LOOKBACK_CANDLES)
    if i15 - 1 < lookback_start:
        return False
    tops, btms = _get_minor_swings(h, l, length, start=lookback_start, end=i15 - 1)
    levels = [p for _, p, _ in (tops + btms)]
    if not levels:
        return False

    atr = atr_short_arr[i15] if i15 < len(atr_short_arr) else np.nan
    if not np.isfinite(atr) or atr <= 0:
        return False
    cluster_tol = cfg_obj.TLR_SR_CLUSTER_ATR * atr

    levels_sorted = sorted(levels)
    clusters: List[List[float]] = [[levels_sorted[0]]]
    for lvl in levels_sorted[1:]:
        if lvl - clusters[-1][-1] <= cluster_tol:
            clusters[-1].append(lvl)
        else:
            clusters.append([lvl])
    qualifying = [cl for cl in clusters if len(cl) >= cfg_obj.TLR_SR_MIN_TOUCHES]
    if not qualifying:
        return False

    touch_price = l[i15] if is_buy else h[i15]
    if not np.isfinite(touch_price):
        return False
    for cl in qualifying:
        if (min(cl) - cluster_tol) <= touch_price <= (max(cl) + cluster_tol):
            return True
    return False

def _tlr_rsi_momentum_confluence(rsi_arr: np.ndarray, i15: int,
                                  prior_touch_idx: Optional[int], is_buy: bool) -> bool:
    if prior_touch_idx is None or prior_touch_idx < 0:
        return False
    if prior_touch_idx >= len(rsi_arr) or i15 >= len(rsi_arr):
        return False
    rsi_now, rsi_prev = rsi_arr[i15], rsi_arr[prior_touch_idx]
    if not (np.isfinite(rsi_now) and np.isfinite(rsi_prev)):
        return False
    return rsi_now > rsi_prev if is_buy else rsi_now < rsi_prev

def _tlr_confluence_vote(h: np.ndarray, l: np.ndarray, atr_short_arr: np.ndarray,
                          rsi_arr: np.ndarray, tl: TrendlineState, i15: int, is_buy: bool,
                          cfg_obj, prior_touch_idx: Optional[int],
                          ob_ok: bool, pattern_ok: bool) -> Tuple[bool, int, Dict[str, bool]]:
    fib_ok = bool(_tlr_fib_confluence(h, l, tl, i15, is_buy, cfg_obj))
    sr_ok = bool(_tlr_sr_confluence(h, l, atr_short_arr, i15, is_buy, cfg_obj))
    rsi_ok = bool(_tlr_rsi_momentum_confluence(rsi_arr, i15, prior_touch_idx, is_buy))
    votes = {"ob": bool(ob_ok), "fib": fib_ok, "sr": sr_ok, "pattern": bool(pattern_ok), "rsi_momentum": rsi_ok}
    passed = sum(votes.values())
    vote_ok = bool(passed >= cfg_obj.TLR_CONFLUENCE_REQUIRED)
    return vote_ok, passed, votes
