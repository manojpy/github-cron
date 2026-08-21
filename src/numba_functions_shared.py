# ============================================================================
# Shared Numba Function Definitions - Single Source of Truth
# ============================================================================

import os
import sys
import hashlib
from pathlib import Path
from importlib.metadata import version as _pkg_version

_numba_ver = _pkg_version("numba")
_cache_version = hashlib.md5(f"{_numba_ver}-{sys.version}".encode()).hexdigest()[:8]
os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    str(Path(__file__).parent / ".numba_cache" / _cache_version),
)

import numpy as np
from numba import njit, prange, types
from typing import Dict, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# SIGNATURES — single source of truth, shared by the @njit decorators below
# and EXPORT_CONFIG at the bottom of this file. Never write a raw string
# signature in more than one place; add/change a signature here only.
# ============================================================================
f8, i4, i8 = types.float64, types.int32, types.int64

SIG_SANITIZE_ARRAY   = f8[:](f8[:], f8)
SIG_ROLLING_MEAN     = f8[:](f8[:], i4)
SIG_ROLLING_MIN_MAX  = types.Tuple((f8[:], f8[:]))(f8[:], i4)
SIG_EMA_LOOP         = f8[:](f8[:], f8)
SIG_EMA_LOOP_PINE    = f8[:](f8[:], f8)
SIG_EMA_LOOP_ALPHA   = f8[:](f8[:], f8)
SIG_KALMAN_LOOP      = f8[:](f8[:], i4, f8, f8)
SIG_VWAP_DAILY       = f8[:](f8[:], f8[:], i8[:])
SIG_PPO_CORE         = types.Tuple((f8[:], f8[:]))(f8[:], i4, i4, i4)
SIG_RSI_CORE         = f8[:](f8[:], i4)
SIG_TRUE_RANGE       = f8[:](f8[:], f8[:], f8[:])
SIG_ATR_RMA          = f8[:](f8[:], f8[:], f8[:], i4)
SIG_ADX_CORE         = f8[:](f8[:], f8[:], f8[:], i4, i4)


# ============================================================================
# 1. SANITIZATION
# ============================================================================

@njit(SIG_SANITIZE_ARRAY, nogil=True, cache=True)
def sanitize_array_numba(arr, default):
    """Replace NaN and Inf with default value - O(n)"""
    out = np.empty_like(arr)
    for i in range(len(arr)):
        val = arr[i]
        out[i] = default if (np.isnan(val) or np.isinf(val)) else val
    return out

@njit(SIG_ROLLING_MEAN, nogil=True, cache=True)
def rolling_mean_numba(data, period):
    n = len(data)
    out = np.full(n, np.nan, dtype=np.float64)

    if period <= 0:
        return out
    has_nan = False
    for i in range(n):
        if np.isnan(data[i]):
            has_nan = True
            break

    if not has_nan:
        window_sum = 0.0
        for i in range(n):
            window_sum += data[i]
            if i >= period:
                window_sum -= data[i - period]
            if i >= period - 1:
                out[i] = window_sum / period
        return out

    window_sum = 0.0
    nan_count = 0
    queue = np.zeros(period, dtype=np.float64)
    is_nan_queue = np.zeros(period, dtype=np.bool_)
    queue_idx = 0

    for i in range(n):
        curr = data[i]
        curr_is_nan = np.isnan(curr)

        if i >= period:
            old_val = queue[queue_idx]
            old_is_nan = is_nan_queue[queue_idx]
            if old_is_nan:
                nan_count -= 1
            else:
                window_sum -= old_val

        if curr_is_nan:
            nan_count += 1
            queue[queue_idx] = 0.0
            is_nan_queue[queue_idx] = True
        else:
            window_sum += curr
            queue[queue_idx] = curr
            is_nan_queue[queue_idx] = False

        queue_idx = (queue_idx + 1) % period

        if i >= period - 1:
            out[i] = np.nan if nan_count > 0 else (window_sum / period)

    return out

@njit(SIG_ROLLING_MIN_MAX, nogil=True, cache=True)
def rolling_min_max_numba(arr, period):
    """Match Pine's ta.lowest/ta.highest: output na unless full window of non-nan values."""
    n = len(arr)
    if period <= 0:
        return np.full(n, np.nan, dtype=np.float64), np.full(n, np.nan, dtype=np.float64)
    min_arr = np.full(n, np.nan, dtype=np.float64)
    max_arr = np.full(n, np.nan, dtype=np.float64)

    min_deque = np.zeros(period, dtype=np.int32)
    max_deque = np.zeros(period, dtype=np.int32)
    min_h = min_t = 0
    max_h = max_t = 0

    valid_count = 0
    valid_buffer = np.zeros(period, dtype=np.bool_)
    buf_idx = 0

    for i in range(n):
        val = arr[i]
        is_valid = not np.isnan(val)

        if i >= period:
            old_valid = valid_buffer[buf_idx]
            if old_valid:
                valid_count -= 1
                if min_h < min_t and min_deque[min_h % period] == i - period:
                    min_h += 1
                if max_h < max_t and max_deque[max_h % period] == i - period:
                    max_h += 1

        valid_buffer[buf_idx] = is_valid
        if is_valid:
            valid_count += 1
            while min_t > min_h and arr[min_deque[(min_t - 1) % period]] >= val:
                min_t -= 1
            min_deque[min_t % period] = i
            min_t += 1
            while max_t > max_h and arr[max_deque[(max_t - 1) % period]] <= val:
                max_t -= 1
            max_deque[max_t % period] = i
            max_t += 1

        buf_idx = (buf_idx + 1) % period

        if i >= period - 1 and valid_count == period:
            min_arr[i] = arr[min_deque[min_h % period]]
            max_arr[i] = arr[max_deque[max_h % period]]

    return min_arr, max_arr

# ============================================================================
# 2. EMA FUNCTIONS
# ============================================================================

@njit(SIG_EMA_LOOP, nogil=True, cache=True)
def ema_loop(data, length_float):
    n = len(data)
    length = int(length_float)
    alpha = 2.0 / (length + 1)
    out = np.full(n, np.nan, dtype=np.float64)

    start_idx = -1
    for i in range(n):
        if not np.isnan(data[i]):
            start_idx = i
            break

    if start_idx == -1 or n < (start_idx + length):
        return out

    sum_val = 0.0
    for i in range(start_idx, start_idx + length):
        sum_val += data[i]

    seed_idx = start_idx + length - 1
    out[seed_idx] = sum_val / length

    for i in range(seed_idx + 1, n):
        curr = data[i]
        if np.isnan(curr):
            out[i] = out[i-1]
        else:
            out[i] = alpha * curr + (1.0 - alpha) * out[i-1]

    return out


@njit(SIG_EMA_LOOP_PINE, nogil=True, cache=True)
def ema_loop_pine(data, length_float):
    n = len(data)
    length = int(length_float)
    alpha = 2.0 / (length + 1)
    out = np.full(n, np.nan, dtype=np.float64)

    # Find first non-NaN input
    start_idx = -1
    for i in range(n):
        if not np.isnan(data[i]):
            start_idx = i
            break

    if start_idx == -1:
        return out

    out[start_idx] = data[start_idx]

    for i in range(start_idx + 1, n):
        curr = data[i]
        if np.isnan(curr):
            out[i] = out[i-1]
        else:
            out[i] = alpha * curr + (1.0 - alpha) * out[i-1]

    return out


@njit(SIG_EMA_LOOP_ALPHA, nogil=True, cache=True)
def ema_loop_alpha(data, alpha):
    n = len(data)
    out = np.full(n, np.nan, dtype=np.float64)

    first_valid_idx = -1
    for i in range(n):
        if not np.isnan(data[i]):
            first_valid_idx = i
            break

    if first_valid_idx == -1:
        return out

    period = int(1.0 / alpha + 0.5)

    if first_valid_idx + period <= n:
        sma_sum = 0.0
        valid_count = 0
        for i in range(first_valid_idx, first_valid_idx + period):
            if not np.isnan(data[i]):
                sma_sum += data[i]
                valid_count += 1
        sma_init = sma_sum / valid_count 
        for i in range(first_valid_idx, first_valid_idx + period):
            if not np.isnan(data[i]):
                out[i] = sma_init

        start_idx = first_valid_idx + period
        prev = sma_init  # internal recursion anchor, independent of what's exposed in out[]
    else:
        out[first_valid_idx] = data[first_valid_idx]
        start_idx = first_valid_idx + 1
        prev = data[first_valid_idx]

    for i in range(start_idx, n):
        curr = data[i]
        if np.isnan(curr):
            out[i] = prev
        else:
            out[i] = alpha * curr + (1.0 - alpha) * prev
        prev = out[i]

    return out

# ============================================================================
# 4. KALMAN / VWAP
# ============================================================================

@njit(SIG_KALMAN_LOOP, nogil=True, cache=True)
def kalman_loop(src, length, R, Q):
    """Kalman filter in O(n) - FIXED: applies formula on first valid bar"""
    n = len(src)
    result = np.full(n, np.nan, dtype=np.float64)

    first_valid_idx = -1
    for i in range(n):
        if not np.isnan(src[i]):
            first_valid_idx = i
            break

    if first_valid_idx == -1:
        return result

    estimate = src[first_valid_idx]
    error_est = 1.0
    error_meas = R * (float(length) if float(length) > 1.0 else 1.0)
    Q_div_length = Q / (float(length) if float(length) > 1.0 else 1.0)

    prediction = estimate
    kalman_gain = error_est / (error_est + error_meas)
    estimate = prediction + kalman_gain * (src[first_valid_idx] - prediction)
    error_est = (1.0 - kalman_gain) * error_est + Q_div_length
    result[first_valid_idx] = estimate

    for i in range(first_valid_idx + 1, n):
        current = src[i]
        if np.isnan(current):
            error_est = error_est + Q_div_length
            result[i] = estimate
            continue
        prediction = estimate
        kalman_gain = error_est / (error_est + error_meas)
        estimate = prediction + kalman_gain * (current - estimate)
        error_est = (1.0 - kalman_gain) * error_est + Q_div_length
        result[i] = estimate

    return result

@njit(SIG_VWAP_DAILY, nogil=True, cache=True)
def vwap_daily_loop_safe(hlc3, volumes, timestamps):
    n = len(hlc3)
    vwap = np.empty(n, dtype=np.float64)
    cum_pv = 0.0
    cum_vol = 0.0
    last_day = -1

    for i in range(n):
        day = timestamps[i] // 86400
        if day != last_day:
            cum_pv = 0.0
            cum_vol = 0.0
            last_day = day
        cum_pv += hlc3[i] * volumes[i]
        cum_vol += volumes[i]
        vwap[i] = cum_pv / cum_vol if cum_vol > 0.0 else hlc3[i]

    return vwap

# ============================================================================
# 5. OSCILLATORS AND TECHNICAL INDICATORS
# ============================================================================

@njit(SIG_PPO_CORE, nogil=True, cache=True)
def calculate_ppo_core(close, fast, slow, signal):
    n = len(close)
    fast_ma = ema_loop_pine(close, float(fast))
    slow_ma = ema_loop_pine(close, float(slow))

    ppo = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        f = fast_ma[i]
        s = slow_ma[i]
        if not np.isnan(f) and not np.isnan(s) and s != 0.0:
            ppo_val = ((f - s) / s) * 100.0
            if ppo_val > 1000.0:
                ppo_val = 1000.0
            elif ppo_val < -1000.0:
                ppo_val = -1000.0
            ppo[i] = ppo_val

    ppo_sig = ema_loop_pine(ppo, float(signal))
    return ppo, ppo_sig

@njit(SIG_RSI_CORE, nogil=True, cache=True)
def calculate_rsi_core(close, period):
    n = len(close)
    rsi = np.full(n, np.nan, dtype=np.float64)

    if n <= period:
        return rsi

    first_valid_idx = -1
    last_valid_close = 0.0
    for i in range(n):
        if not np.isnan(close[i]):
            first_valid_idx = i
            last_valid_close = close[i]
            break

    if first_valid_idx == -1:
        return rsi

    avg_gain = 0.0
    avg_loss = 0.0
    prev_valid = last_valid_close
    warmup_end = min(first_valid_idx + period + 1, n)

    for i in range(first_valid_idx + 1, warmup_end):
        curr = close[i]
        if not np.isnan(curr):
            diff = curr - prev_valid
            if diff > 0.0:
                avg_gain += diff
            else:
                avg_loss += -diff
            if i < first_valid_idx + period:
                prev_valid = curr

    avg_gain /= period
    avg_loss /= period

    alpha = 1.0 / period

    for i in range(first_valid_idx + period, n):
        curr = close[i]
        if not np.isnan(curr):
            diff = curr - prev_valid
            if diff > 0.0:
                avg_gain = (diff * alpha) + (avg_gain * (1.0 - alpha))
                avg_loss = (avg_loss * (1.0 - alpha))
            else:
                avg_gain = (avg_gain * (1.0 - alpha))
                avg_loss = (-diff * alpha) + (avg_loss * (1.0 - alpha))
            prev_valid = curr

        if avg_loss == 0.0:
            rsi[i] = 100.0 if avg_gain > 0.0 else 50.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))

    return rsi

@njit(SIG_TRUE_RANGE, nogil=True, cache=True)
def true_range_numba(high, low, close):
    """Shared True Range calc — previously duplicated in calculate_atr_rma and calculate_adx_core."""
    n = len(close)
    tr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]

    for i in range(1, n):
        h = high[i]
        l = low[i]
        c = close[i - 1]
        tr1 = h - l
        tr2 = abs(h - c)
        tr3 = abs(l - c)
        tr[i] = max(tr1, tr2, tr3)

    return tr

@njit(SIG_ATR_RMA, nogil=True, cache=True)
def calculate_atr_rma(high, low, close, period):
    n = len(close)
    if n < period:
        return np.full(n, np.nan, dtype=np.float64)

    tr = true_range_numba(high, low, close)

    alpha = 1.0 / float(period)
    atr = ema_loop_alpha(tr, alpha)
    return atr

@njit(SIG_ADX_CORE, nogil=True, cache=True)
def calculate_adx_core(high, low, close, di_length, adx_length):
    n = len(high)
    adx = np.full(n, np.nan, dtype=np.float64)

    if n < di_length + adx_length:
        return adx

    tr = true_range_numba(high, low, close)
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)

    for i in range(1, n):
        h = high[i]
        l = low[i]
        prev_h = high[i - 1]
        prev_l = low[i - 1]

        up = h - prev_h
        down = prev_l - l
        plus_dm[i] = up if (up > down and up > 0) else 0.0
        minus_dm[i] = down if (down > up and down > 0) else 0.0

    alpha_di = 1.0 / float(di_length)
    plus_dm_smooth = ema_loop_alpha(plus_dm, alpha_di)
    minus_dm_smooth = ema_loop_alpha(minus_dm, alpha_di)
    tr_smooth = ema_loop_alpha(tr, alpha_di)

    for i in range(n):
        if tr_smooth[i] > 0.0 and not np.isnan(tr_smooth[i]):
            plus_dm_smooth[i] = 100.0 * plus_dm_smooth[i] / tr_smooth[i]
            minus_dm_smooth[i] = 100.0 * minus_dm_smooth[i] / tr_smooth[i]
        else:
            plus_dm_smooth[i] = 0.0
            minus_dm_smooth[i] = 0.0

    for i in range(n):
        di_diff = abs(plus_dm_smooth[i] - minus_dm_smooth[i])
        di_sum = plus_dm_smooth[i] + minus_dm_smooth[i]
        tr_smooth[i] = 0.0 if di_sum == 0.0 else 100.0 * di_diff / di_sum

    alpha_adx = 1.0 / float(adx_length)
    adx = ema_loop_alpha(tr_smooth, alpha_adx)
    return adx

from aot_version import SOURCE_VERSION  # noqa: E402

# ============================================================================
# AOT EXPORT CONFIGURATION
# ============================================================================

EXPORT_CONFIG = {
    'sanitize_array_numba':  SIG_SANITIZE_ARRAY,
    'rolling_mean_numba':    SIG_ROLLING_MEAN,
    'rolling_min_max_numba': SIG_ROLLING_MIN_MAX,
    'ema_loop':              SIG_EMA_LOOP,
    'ema_loop_pine':         SIG_EMA_LOOP_PINE,          # NEW
    'ema_loop_alpha':        SIG_EMA_LOOP_ALPHA,
    'kalman_loop':           SIG_KALMAN_LOOP,
    'vwap_daily_loop_safe':  SIG_VWAP_DAILY,
    'calculate_ppo_core':    SIG_PPO_CORE,
    'calculate_rsi_core':    SIG_RSI_CORE,
    'true_range_numba':      SIG_TRUE_RANGE,
    'calculate_atr_rma':     SIG_ATR_RMA,
    'calculate_adx_core':    SIG_ADX_CORE,
}

__all__ = list(EXPORT_CONFIG.keys())

# Guard: raise immediately at import if count drops unexpectedly
expected_min_functions = 13
if len(__all__) < expected_min_functions:
    raise AssertionError(
        f"Expected at least {expected_min_functions} exported functions, "
        f"but only {len(__all__)} found: {__all__}"
    )

logger.info(f"✅ Exported {len(__all__)} Numba-compiled functions for AOT")
