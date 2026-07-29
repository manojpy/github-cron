# ============================================================================
# Shared Numba Function Definitions - Single Source of Truth
# ============================================================================

import numpy as np
from numba import njit, prange, types
from typing import Dict, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# 1. SANITIZATION
# ============================================================================

@njit("f8[:](f8[:], f8)", nogil=True, cache=True)
def sanitize_array_numba(arr, default):
    """Replace NaN and Inf with default value - O(n)"""
    out = np.empty_like(arr)
    for i in range(len(arr)):
        val = arr[i]
        out[i] = default if (np.isnan(val) or np.isinf(val)) else val
    return out


@njit("f8[:](f8[:], f8)", nogil=True, parallel=True, cache=True)
def sanitize_array_numba_parallel(arr, default):
    """Replace NaN and Inf with default value (parallel) - O(n)"""
    out = np.empty_like(arr)
    for i in prange(len(arr)):
        val = arr[i]
        out[i] = default if (np.isnan(val) or np.isinf(val)) else val
    return out

@njit("f8[:](f8[:], i4)", nogil=True, cache=True)
def rolling_mean_numba(data, period):
    """Calculate rolling mean matching Pine's ta.sma: returns NaN for first (period - 1)
    bars AND for any window that contains a NaN (full valid window required).

    FIX: the previous version accumulated a plain running sum. A single NaN entering
    the window poisoned window_sum permanently -- even after that NaN aged out of the
    window, `window_sum -= old_val` (NaN - NaN) kept the sum (and every bar after it)
    as NaN forever. This version tracks NaNs in the circular buffer explicitly so the
    poisoning clears correctly once the NaN leaves the window, and correctly emits NaN
    only while a NaN is actually inside the window.
    
    OPTIMIZED: Fast path for data with no NaNs uses a simple running sum, avoiding
    circular buffer overhead entirely.
    """
    n = len(data)
    out = np.full(n, np.nan, dtype=np.float64)

    if period <= 0:
        return out

    # Fast path: check for any NaNs
    has_nan = False
    for i in range(n):
        if np.isnan(data[i]):
            has_nan = True
            break

    if not has_nan:
        # Simple running sum — O(n), no circular buffer overhead
        window_sum = 0.0
        for i in range(n):
            window_sum += data[i]
            if i >= period:
                window_sum -= data[i - period]
            if i >= period - 1:
                out[i] = window_sum / period
        return out

    # Slow path: original circular buffer with explicit NaN tracking
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

@njit("Tuple((f8[:], f8[:]))(f8[:], i4)", nogil=True, cache=True)
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

@njit("f8[:](f8[:], f8)", nogil=True, cache=True)
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


@njit("f8[:](f8[:], f8)", nogil=True, cache=True)
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


@njit("f8[:](f8[:], f8)", nogil=True, cache=True)
def ema_loop_alpha(data, alpha):
    """EMA with explicit alpha parameter - with proper SMA initialisation for RMA.

    FIX: the previous version wrote the same fabricated `sma_init` constant into
    EVERY position of the seed window, including positions where the source data
    itself was NaN (a real gap). That produced a false flatline during warm-up on
    sparse data. Now, seed-window bars with real data still get `sma_init` (this is
    unchanged from before, and is the behavior already validated against Pine's
    ta.rma/ta.ema seeding), but seed-window bars that were genuinely NaN in the
    source stay NaN in the output instead of being papered over. Recursion after
    the seed window uses a separate `prev` anchor (not out[i-1]) so this doesn't
    break the EMA chain when the last seed bar happens to be one of those NaNs.
    When there are no NaNs in the seed window, output is identical to before.
    """
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

@njit("f8[:](f8[:], i4, f8, f8)", nogil=True, cache=True)
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
            # FIX: predict-only step during a data gap. Previously error_est was
            # frozen while a gap was skipped, so the filter stayed artificially
            # confident in a stale estimate and was slow to react once real data
            # resumed. Growing error_est here (same process-noise term used on a
            # normal step) lets confidence decay across the gap, so kalman_gain is
            # naturally higher on the next real observation. No effect when there
            # are no gaps, since this branch is only reached on NaN input.
            error_est = error_est + Q_div_length
            result[i] = estimate
            continue
        prediction = estimate
        kalman_gain = error_est / (error_est + error_meas)
        estimate = prediction + kalman_gain * (current - estimate)
        error_est = (1.0 - kalman_gain) * error_est + Q_div_length
        result[i] = estimate

    return result


@njit("f8[:](f8[:], f8[:], i8[:])", nogil=True, cache=True)
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

@njit("Tuple((f8[:], f8[:]))(f8[:], i4, i4, i4)", nogil=True, cache=True)
def calculate_ppo_core(close, fast, slow, signal):
    """
    Percentage Price Oscillator (PPO) matching Pine Script behavior exactly.
    Returns only (ppo, ppo_sig) to avoid unpacking errors. No histogram included.
    """
    n = len(close)
    fast_ma = ema_loop_pine(close, float(fast))
    slow_ma = ema_loop_pine(close, float(slow))

    ppo = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        f = fast_ma[i]
        s = slow_ma[i]
        if not np.isnan(f) and not np.isnan(s) and s != 0.0:
            ppo_val = ((f - s) / s) * 100.0
            ppo[i] = np.clip(ppo_val, -1000.0, 1000.0)  # Prevent absurd outliers

    ppo_sig = ema_loop_pine(ppo, float(signal))
    return ppo, ppo_sig

@njit("f8[:](f8[:], i4)", nogil=True, cache=True)
def calculate_rsi_core(close, period):
    """Calculate RSI in O(n) - single pass gains/losses, then EMA.

    FIX: default fill changed from 50.0 to NaN. Bars before RSI has enough history
    now correctly report "no value yet" instead of a fabricated midline reading.
    
    OPTIMIZED: Eliminated full-length gain/loss arrays. Gains/losses are computed
    on the fly during both the warm-up and smoothing phases, saving 2 * n * 8 bytes.
    """
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

    # Warm-up: accumulate over the first 'period' bars after first_valid_idx
    avg_gain = 0.0
    avg_loss = 0.0
    last_valid = last_valid_close
    warmup_end = min(first_valid_idx + period + 1, n)
    for i in range(first_valid_idx + 1, warmup_end):
        curr = close[i]
        if not np.isnan(curr):
            diff = curr - last_valid
            if diff > 0.0:
                avg_gain += diff
            else:
                avg_loss += -diff
            last_valid = curr
        # NaN bars contribute 0, matching original behavior

    avg_gain /= period
    avg_loss /= period

    alpha = 1.0 / period

    # Smoothing phase
    for i in range(first_valid_idx + period, n):
        curr = close[i]
        if not np.isnan(curr):
            diff = curr - last_valid
            if diff > 0.0:
                avg_gain = (diff * alpha) + (avg_gain * (1.0 - alpha))
                avg_loss = (avg_loss * (1.0 - alpha))
            else:
                avg_gain = (avg_gain * (1.0 - alpha))
                avg_loss = (-diff * alpha) + (avg_loss * (1.0 - alpha))
            last_valid = curr
        # NaN bars leave averages unchanged, matching original

        if avg_loss == 0.0:
            rsi[i] = 100.0 if avg_gain > 0.0 else 50.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))

    return rsi

@njit("f8[:](f8[:], f8[:], f8[:], i4)", nogil=True, cache=True)
def calculate_atr_rma(high, low, close, period):
    n = len(close)
    if n < period:
        return np.full(n, np.nan, dtype=np.float64)

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

    alpha = 1.0 / float(period)
    atr = ema_loop_alpha(tr, alpha)
    return atr

@njit("f8[:](f8[:], f8[:], f8[:], i4, i4)", nogil=True, cache=True)
def calculate_adx_core(high, low, close, di_length, adx_length):
    """Calculate ADX in O(n) using Pine Script equivalent logic.
    
    OPTIMIZED: Reuses smoothed DM/TR buffers for DI/raw_ADX, cutting
    temporary allocations from 11 arrays down to 6.
    """
    n = len(high)
    adx = np.full(n, np.nan, dtype=np.float64)

    if n < di_length + adx_length:
        return adx

    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)

    tr[0] = high[0] - low[0]

    for i in range(1, n):
        h = high[i]
        l = low[i]
        prev_h = high[i - 1]
        prev_l = low[i - 1]
        prev_c = close[i - 1]

        tr1 = h - l
        tr2 = abs(h - prev_c)
        tr3 = abs(l - prev_c)
        tr[i] = max(tr1, tr2, tr3)

        up = h - prev_h
        down = prev_l - l
        plus_dm[i] = up if (up > down and up > 0) else 0.0
        minus_dm[i] = down if (down > up and down > 0) else 0.0

    alpha_di = 1.0 / float(di_length)
    plus_dm_smooth = ema_loop_alpha(plus_dm, alpha_di)
    minus_dm_smooth = ema_loop_alpha(minus_dm, alpha_di)
    tr_smooth = ema_loop_alpha(tr, alpha_di)

    # Reuse plus_dm_smooth / minus_dm_smooth as plus_di / minus_di
    for i in range(n):
        if tr_smooth[i] > 0.0 and not np.isnan(tr_smooth[i]):
            plus_dm_smooth[i] = 100.0 * plus_dm_smooth[i] / tr_smooth[i]
            minus_dm_smooth[i] = 100.0 * minus_dm_smooth[i] / tr_smooth[i]
        else:
            plus_dm_smooth[i] = 0.0
            minus_dm_smooth[i] = 0.0

    # Reuse tr_smooth as raw_adx buffer
    for i in range(n):
        di_diff = abs(plus_dm_smooth[i] - minus_dm_smooth[i])
        di_sum = plus_dm_smooth[i] + minus_dm_smooth[i]
        tr_smooth[i] = 0.0 if di_sum == 0.0 else 100.0 * di_diff / di_sum

    alpha_adx = 1.0 / float(adx_length)
    adx = ema_loop_alpha(tr_smooth, alpha_adx)
    return adx

# ============================================================================
# SOURCE VERSION -- imported from aot_version.py (see that file for why it's
# kept separate). aot_build.py stamps this into a sidecar file next to the
# compiled .so, and aot_bridge.py refuses to trust an AOT library whose stamp
# doesn't match this live value -- preventing a stale compiled binary from
# silently running old logic forever.
# ============================================================================
from aot_version import SOURCE_VERSION  # noqa: E402

# ============================================================================
# AOT EXPORT CONFIGURATION
# ============================================================================

EXPORT_CONFIG = {
    'sanitize_array_numba':          'f8[:](f8[:], f8)',
    'sanitize_array_numba_parallel': 'f8[:](f8[:], f8)',
    'rolling_mean_numba':            'f8[:](f8[:], i4)',
    'rolling_min_max_numba':         'Tuple((f8[:], f8[:]))(f8[:], i4)',
    'ema_loop':                      'f8[:](f8[:], f8)',
    'ema_loop_pine':                 'f8[:](f8[:], f8)',          # NEW
    'ema_loop_alpha':                'f8[:](f8[:], f8)', 
    'kalman_loop':                   'f8[:](f8[:], i4, f8, f8)',
    'vwap_daily_loop_safe':          'f8[:](f8[:], f8[:], i8[:])',
    'calculate_ppo_core':            'Tuple((f8[:], f8[:]))(f8[:], i4, i4, i4)',
    'calculate_rsi_core':            'f8[:](f8[:], i4)',
    'calculate_atr_rma':             'f8[:](f8[:], f8[:], f8[:], i4)',
    'calculate_adx_core':            'f8[:](f8[:], f8[:], f8[:], i4, i4)',
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
