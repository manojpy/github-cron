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


@njit("f8[:](f8[:], i4, f8)", nogil=True, cache=True)
def rolling_std(close, period, responsiveness):
    
    n = len(close)
    sd = np.full(n, np.nan, dtype=np.float64)
    resp = max(0.00001, min(1.0, responsiveness))

    if n < 2 or period < 2:
        return np.zeros(n, dtype=np.float64)

    for i in range(period - 1, n):
        window = close[i - period + 1 : i + 1]
        valid_window = window[~np.isnan(window)]
        if len(valid_window) >= 2:
            sd[i] = np.std(valid_window) * resp
        else:
            sd[i] = 0.0

    mask = np.isnan(sd)
    sd[mask] = 0.0
    return sd


@njit("f8[:](f8[:], i4)", nogil=True, cache=True)
def rolling_mean_numba(data, period):
    """
    Calculate rolling mean matching Pine's ta.sma: returns NaN for first (period - 1) bars.
    FIXED: Uses a robust tracking queue to prevent NaNs from permanently poisoning 
    the running window sum.
    """
    n = len(data)
    out = np.full(n, np.nan, dtype=np.float64)

    if period <= 0 or n < period:
        return out

    window_sum = 0.0
    nan_count = 0
    queue = np.zeros(period, dtype=np.float64)
    queue_nan = np.zeros(period, dtype=np.bool_)
    queue_idx = 0

    for i in range(n):
        curr = data[i]
        curr_is_nan = np.isnan(curr)

        # Remove oldest element sliding out of the window
        if i >= period:
            old_idx = queue_idx
            old_val = queue[old_idx]
            old_is_nan = queue_nan[old_idx]
            if old_is_nan:
                nan_count -= 1
            else:
                window_sum -= old_val

        # Add newest element sliding into the window
        queue[queue_idx] = 0.0 if curr_is_nan else curr
        queue_nan[queue_idx] = curr_is_nan
        if curr_is_nan:
            nan_count += 1
        else:
            window_sum += curr

        queue_idx = (queue_idx + 1) % period

        # Fill output only if the entire sliding window is valid
        if i >= period - 1:
            if nan_count == 0:
                out[i] = window_sum / period
            else:
                out[i] = np.nan

    return out

@njit("Tuple((f8[:], f8[:]))(f8[:], i4)", nogil=True, cache=True)
def rolling_min_max_numba(arr, period):
    """Match Pine's ta.lowest/ta.highest: output na unless full window of non-nan values."""
    n = len(arr)
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

@njit("f8[:](f8[:], f8[:], i8)", nogil=True, cache=True)
def calc_mmh_worm_loop(close_arr, sd_arr, rows):
    """Calculate worm array - Pine's exact logic"""
    worm_arr = np.empty(rows, dtype=np.float64)
    worm_arr[0] = close_arr[0]
    for i in range(1, rows):
        src = close_arr[i]
        prev_worm = worm_arr[i - 1]
        diff = src - prev_worm
        sd_i = sd_arr[i]
        if np.abs(diff) > sd_i:
            delta = np.sign(diff) * sd_i
        else:
            delta = diff
        worm_arr[i] = prev_worm + delta
    return worm_arr


@njit("f8[:](f8[:], f8[:], f8[:], i4)", nogil=True, cache=True)
def calc_mmh_value_loop(raw_arr, min_arr, max_arr, rows):
    """Corrected value loop with NaN propagation to match Pine Script recursion"""
    value_arr = np.full(rows, np.nan, dtype=np.float64)
    for i in range(rows):
        raw = raw_arr[i]
        mn = min_arr[i]
        mx = max_arr[i]
        denom = mx - mn
        if np.isnan(raw) or np.isnan(mn) or np.isnan(mx) or np.abs(denom) < 1e-10:
            temp = np.nan
        else:
            temp = (raw - mn) / denom

        if np.isnan(temp):
            value_arr[i] = np.nan
        else:
            prev_v = value_arr[i-1] if i > 0 else np.nan
            prev_v_safe = 0.0 if np.isnan(prev_v) else prev_v
            v = 1.0 * (temp - 0.5 + 0.5 * prev_v_safe)
            if v > 0.9999: v = 0.9999
            if v < -0.9999: v = -0.9999
            value_arr[i] = v

    return value_arr


@njit("f8[:](f8[:], i4)", nogil=True, cache=True)
def calc_mmh_momentum_loop(value_arr, rows):
    """Corrected momentum transform (log-odds)"""
    momentum = np.full(rows, np.nan, dtype=np.float64)
    for i in range(rows):
        v = value_arr[i]
        if np.isnan(v):
            momentum[i] = np.nan
        else:
            val_clamped = max(-0.99999, min(0.99999, v))
            temp2 = (1.0 + val_clamped) / (1.0 - val_clamped)
            momentum[i] = 0.25 * np.log(temp2)
    return momentum


@njit("f8[:](f8[:], i4)", nogil=True, cache=True)
def calc_mmh_momentum_smoothing(momentum_arr, rows):
    """Corrected final smoothing with NaN propagation"""
    result = np.full(rows, np.nan, dtype=np.float64)
    for i in range(rows):
        curr = momentum_arr[i]
        if np.isnan(curr):
            result[i] = np.nan
        else:
            prev = result[i-1] if i > 0 else np.nan
            prev_safe = 0.0 if np.isnan(prev) else prev
            result[i] = curr + 0.5 * prev_safe
    return result


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
    """
    EMA with explicit alpha parameter.
    FIXED: Implements true Wilder's RMA/EMA startup logic. Avoids flatline backfilling 
    by waiting until 'period' valid bars have occurred before emitting the SMA seed.
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

    # Track window counts precisely to align with Pine Script's RMA startup
    valid_count = 0
    sma_sum = 0.0
    init_idx = -1
    
    for i in range(first_valid_idx, n):
        if not np.isnan(data[i]):
            valid_count += 1
            sma_sum += data[i]
            if valid_count == period:
                init_idx = i
                break

    if init_idx == -1:
        return out

    # Start RMA with SMA calculation on the 'period'-th valid index
    out[init_idx] = sma_sum / period

    for i in range(init_idx + 1, n):
        curr = data[i]
        if np.isnan(curr):
            out[i] = out[i-1]
        else:
            out[i] = alpha * curr + (1.0 - alpha) * out[i-1]

    return out

# ============================================================================
# 4. KALMAN / VWAP
# ============================================================================

@njit("f8[:](f8[:], i4, f8, f8)", nogil=True, cache=True)
def kalman_loop(src, length, R, Q):
    """
    Kalman filter in O(n)
    FIXED: Updates state uncertainty variables even when encountering NaNs to 
    prevent the tracker from treating stale confidence as ground truth.
    """
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
            # FIXED: Propagate prediction but inflate uncertainty/error covariance!
            error_est = error_est + Q_div_length
            result[i] = estimate
            continue
        
        prediction = estimate
        kalman_gain = error_est / (error_est + error_meas)
        estimate = prediction + kalman_gain * (current - prediction)
        error_est = (1.0 - kalman_gain) * error_est + Q_div_length
        result[i] = estimate

    return result


@njit("f8[:](f8[:], f8[:], i8[:])", nogil=True, cache=True)
def vwap_daily_loop_safe(hlc3, volumes, timestamps):
    """
    FIXED: Automatically detects and normalizes Millisecond vs Second timestamps,
    preventing intra-day VWAP resets every 86 seconds on millisecond scale formats.
    """
    n = len(hlc3)
    vwap = np.empty(n, dtype=np.float64)
    cum_pv = 0.0
    cum_vol = 0.0
    last_day = -1

    divisor = 86400
    # Dynamically detect if timestamps are in milliseconds (e.g. > 10^10)
    if n > 0 and timestamps[0] > 5 * 10**10:
        divisor = 86400000

    for i in range(n):
        day = timestamps[i] // divisor
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
    # Using the Pine-exact EMA helper to prevent warm-up mismatches
    fast_ma = ema_loop_pine(close, float(fast))
    slow_ma = ema_loop_pine(close, float(slow))

    ppo = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        f = fast_ma[i]
        s = slow_ma[i]
        if not np.isnan(f) and not np.isnan(s) and s != 0.0:
            ppo[i] = ((f - s) / s) * 100.0

    ppo_sig = ema_loop_pine(ppo, float(signal))
    return ppo, ppo_sig

@njit("f8[:](f8[:], i4)", nogil=True, cache=True)
def calculate_rsi_core(close, period):
    """
    Calculate RSI in O(n) with robust NaN gap handling.
    FIXED: Baseline defaults to standard NaN to avoid false warm-up baselines.
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

    gain = np.zeros(n, dtype=np.float64)
    loss = np.zeros(n, dtype=np.float64)

    for i in range(first_valid_idx + 1, n):
        curr = close[i]
        if not np.isnan(curr):
            diff = curr - last_valid_close
            if diff > 0.0:
                gain[i] = diff
            else:
                loss[i] = -diff
            last_valid_close = curr

    avg_gain = 0.0
    avg_loss = 0.0
    valid_count = 0
    seed_idx = -1

    for i in range(first_valid_idx + 1, n):
        if not np.isnan(close[i]):
            valid_count += 1
            avg_gain += gain[i]
            avg_loss += loss[i]
            if valid_count == period:
                seed_idx = i
                break

    if seed_idx == -1:
        return rsi

    avg_gain /= period
    avg_loss /= period

    if avg_loss == 0.0:
        rsi[seed_idx] = 100.0 if avg_gain > 0.0 else 50.0
    else:
        rs = avg_gain / avg_loss
        rsi[seed_idx] = 100.0 - (100.0 / (1.0 + rs))

    alpha = 1.0 / period

    for i in range(seed_idx + 1, n):
        if not np.isnan(close[i]):
            avg_gain = (gain[i] * alpha) + (avg_gain * (1.0 - alpha))
            avg_loss = (loss[i] * alpha) + (avg_loss * (1.0 - alpha))

            if avg_loss == 0.0:
                rsi[i] = 100.0 if avg_gain > 0.0 else 50.0
            else:
                rs = avg_gain / avg_loss
                rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = rsi[i-1]

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
    """
    Calculate ADX in O(n) matching Pine Script logic.
    FIXED: Bypasses non-vectorized np.where allocations with an optimized manual loop.
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

    plus_di = np.full(n, np.nan, dtype=np.float64)
    minus_di = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        if tr_smooth[i] > 0.0 and not np.isnan(tr_smooth[i]):
            plus_di[i] = 100.0 * plus_dm_smooth[i] / tr_smooth[i]
            minus_di[i] = 100.0 * minus_dm_smooth[i] / tr_smooth[i]
        else:
            plus_di[i] = 0.0
            minus_di[i] = 0.0

    raw_adx = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        p_di = plus_di[i]
        m_di = minus_di[i]
        if np.isnan(p_di) or np.isnan(m_di):
            raw_adx[i] = np.nan
        else:
            di_sum = p_di + m_di
            if di_sum == 0.0:
                raw_adx[i] = 0.0
            else:
                raw_adx[i] = 100.0 * abs(p_di - m_di) / di_sum

    alpha_adx = 1.0 / float(adx_length)
    adx = ema_loop_alpha(raw_adx, alpha_adx)
    return adx




# ============================================================================
# AOT EXPORT CONFIGURATION
# ============================================================================

EXPORT_CONFIG = {
    'sanitize_array_numba':          'f8[:](f8[:], f8)',
    'sanitize_array_numba_parallel': 'f8[:](f8[:], f8)',
    'rolling_std':                   'f8[:](f8[:], i4, f8)',
    'rolling_mean_numba':            'f8[:](f8[:], i4)',
    'rolling_min_max_numba':         'Tuple((f8[:], f8[:]))(f8[:], i4)',
    'calc_mmh_worm_loop':            'f8[:](f8[:], f8[:], i8)',
    'calc_mmh_value_loop':           'f8[:](f8[:], f8[:], f8[:], i4)',
    'calc_mmh_momentum_loop':        'f8[:](f8[:], i4)',
    'calc_mmh_momentum_smoothing':   'f8[:](f8[:], i4)',
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
expected_min_functions = 18
if len(__all__) < expected_min_functions:
    raise AssertionError(
        f"Expected at least {expected_min_functions} exported functions, "
        f"but only {len(__all__)} found: {__all__}"
    )

logger.info(f"✅ Exported {len(__all__)} Numba-compiled functions for AOT")
