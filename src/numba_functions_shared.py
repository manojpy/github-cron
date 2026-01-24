# ============================================================================
# Shared Numba Function Definitions - Single Source of Truth 
# ============================================================================

import numpy as np
from numba import njit, prange, types
from numba.types import Tuple, bool_

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

# ============================================================================
# 2. STATISTICAL FUNCTIONS
# ============================================================================

@njit("f8[:](f8[:], i4, f8)", nogil=True, cache=True)
def rolling_std(close, period, responsiveness):
    """
    STABLE Population Standard Deviation matching Pine Script's ta.stdev.
    Uses slicing to prevent floating-point catastrophic cancellation on high-price assets.
    """
    n = len(close)
    sd = np.full(n, np.nan, dtype=np.float64)
    # Clamp responsiveness
    resp = max(0.00001, min(1.0, responsiveness))

    # We need at least 2 values for a standard deviation
    if n < 2 or period < 2:
        return np.zeros(n, dtype=np.float64)

    for i in range(period - 1, n):
        # Slicing in Numba is efficient (view-based)
        window = close[i - period + 1 : i + 1]
        
        # Count valid numbers in window
        valid_window = window[~np.isnan(window)]
        
        if len(valid_window) >= 2:
            # np.std is numerically stable and matches population std (ddof=0)
            sd[i] = np.std(valid_window) * resp
        else:
            sd[i] = 0.0
            
    # Handle the initial warmup period by filling with 0.0 or propagating first valid
    mask = np.isnan(sd)
    sd[mask] = 0.0
    
    return sd

@njit("f8[:](f8[:], i4)", nogil=True, cache=True)
def rolling_mean_numba(data, period):
    """Calculate rolling mean matching Pine's ta.sma"""
    n = len(data)
    out = np.full(n, np.nan, dtype=np.float64)

    if period <= 0:
        return out

    window_sum = 0.0
    queue = np.zeros(period, dtype=np.float64)
    queue_idx = 0

    for i in range(n):
        curr = data[i]
        if i >= period:
            window_sum -= queue[queue_idx]

        window_sum += curr
        queue[queue_idx] = curr
        queue_idx = (queue_idx + 1) % period

        if i >= period - 1:
            out[i] = window_sum / period
    return out

@njit("Tuple((f8[:], f8[:]))(f8[:], i4)", nogil=True, cache=True)
def rolling_min_max_numba(arr, period):
    """Match Pine's ta.lowest/ta.highest: output na unless full window of non-nan values."""
    n = len(arr)
    min_arr = np.full(n, np.nan, dtype=np.float64)
    max_arr = np.full(n, np.nan, dtype=np.float64)

    min_deque = np.zeros(period, dtype=np.int32)
    max_deque = np.zeros(period, dtype=np.int32)
    min_h = min_t = max_h = max_t = 0

    valid_count = 0
    valid_buffer = np.zeros(period, dtype=bool_)
    buf_idx = 0

    for i in range(n):
        val = arr[i]
        is_valid = not np.isnan(val)

        if i >= period:
            if valid_buffer[buf_idx]:
                valid_count -= 1
                if min_h < min_t and min_deque[min_h] == i - period: min_h += 1
                if max_h < max_t and max_deque[max_h] == i - period: max_h += 1

        valid_buffer[buf_idx] = is_valid
        if is_valid:
            valid_count += 1
            while min_t > min_h and arr[min_deque[min_t - 1]] >= val: min_t -= 1
            min_deque[min_t] = i
            min_t += 1
            while max_t > max_h and arr[max_deque[max_t - 1]] <= val: max_t -= 1
            max_deque[max_t] = i
            max_t += 1

        buf_idx = (buf_idx + 1) % period
        if i >= period - 1 and valid_count == period:
            min_arr[i] = arr[min_deque[min_h]]
            max_arr[i] = arr[max_deque[max_h]]

    return min_arr, max_arr

# ============================================================================
# 3. MAGICAL MOMENTUM (MMH) COMPONENTS
# ============================================================================

@njit("f8[:](f8[:], f8[:], i8)", nogil=True, cache=True)
def calc_mmh_worm_loop(close_arr, sd_arr, rows):
    """Calculate worm array - Pine's exact logic"""
    worm_arr = np.empty(rows, dtype=np.float64)
    worm_arr[0] = close_arr[0]

    for i in range(1, rows):
        diff = close_arr[i] - worm_arr[i - 1]
        sd_i = sd_arr[i]
        delta = (np.sign(diff) * sd_i) if np.abs(diff) > sd_i else diff
        worm_arr[i] = worm_arr[i - 1] + delta
    return worm_arr


@njit("f8[:](f8[:], f8[:], f8[:], i4)", nogil=True, cache=True)
def calc_mmh_value_loop(raw_arr, min_arr, max_arr, rows):
    """Recursive value loop with NaN propagation"""
    value_arr = np.full(rows, np.nan, dtype=np.float64)
    
    for i in range(rows):
        raw, mn, mx = raw_arr[i], min_arr[i], max_arr[i]
        denom = mx - mn
        
        if np.isnan(raw) or np.isnan(mn) or np.isnan(mx) or np.abs(denom) < 1e-10:
            temp = np.nan
        else:
            temp = (raw - mn) / denom

        if np.isnan(temp):
            value_arr[i] = np.nan
        else:
            prev_v = value_arr[i-1] if (i > 0 and not np.isnan(value_arr[i-1])) else 0.0
            v = 1.0 * (temp - 0.5 + 0.5 * prev_v)
            value_arr[i] = max(-0.9999, min(0.9999, v))
            
    return value_arr

@njit("f8[:](f8[:], i4)", nogil=True, cache=True)
def calc_mmh_momentum_loop(value_arr, rows):
    """Log-odds momentum transform"""
    momentum = np.full(rows, np.nan, dtype=np.float64)
    for i in range(rows):
        v = value_arr[i]
        if not np.isnan(v):
            val_clamped = max(-0.99999, min(0.99999, v))
            momentum[i] = 0.25 * np.log((1.0 + val_clamped) / (1.0 - val_clamped))
    return momentum

@njit("f8[:](f8[:], i4)", nogil=True, cache=True)
def calc_mmh_momentum_smoothing(momentum_arr, rows):
    """Final smoothing with nz() logic"""
    result = np.full(rows, np.nan, dtype=np.float64)
    for i in range(rows):
        curr = momentum_arr[i]
        if not np.isnan(curr):
            prev_safe = result[i-1] if (i > 0 and not np.isnan(result[i-1])) else 0.0
            result[i] = curr + 0.5 * prev_safe
    return result

# ============================================================================
# 4. MOVING AVERAGES & FILTERS
# ============================================================================

@njit("f8[:](f8[:], f8)", nogil=True, cache=True)
def ema_loop(data, alpha_or_period):
    """EMA in O(n)"""
    n = len(data)
    alpha = 2.0 / (alpha_or_period + 1.0) if alpha_or_period > 1.0 else alpha_or_period
    out = np.full(n, np.nan, dtype=np.float64)
    
    idx = -1
    for i in range(n):
        if not np.isnan(data[i]):
            idx = i
            break
    if idx == -1: return out
    
    out[idx] = data[idx]
    for i in range(idx + 1, n):
        out[i] = out[i-1] if np.isnan(data[i]) else (alpha * data[i] + (1.0 - alpha) * out[i-1])
    return out


@njit("f8[:](f8[:], f8)", nogil=True, cache=True)
def ema_loop_alpha(data, alpha):
    """EMA with explicit alpha and SMA warmup for RMA"""
    n = len(data)
    out = np.full(n, np.nan, dtype=np.float64)
    
    idx = -1
    for i in range(n):
        if not np.isnan(data[i]):
            idx = i
            break
    if idx == -1: return out
    
    period = int(1.0 / alpha + 0.5)
    if idx + period <= n:
        sma = np.nanmean(data[idx : idx + period])
        for i in range(idx, idx + period): out[i] = sma
        start_idx = idx + period
    else:
        out[idx] = data[idx]
        start_idx = idx + 1
    
    for i in range(start_idx, n):
        out[i] = out[i-1] if np.isnan(data[i]) else (alpha * data[i] + (1.0 - alpha) * out[i-1])
    return out

@njit("f8[:](f8[:], f8[:])", nogil=True, cache=True)
def rng_filter_loop(x, r):
    """Range filter logic"""
    n = len(x)
    filt = np.zeros(n, dtype=np.float64)
    filt[0] = x[0] if not np.isnan(x[0]) else 0.0

    for i in range(1, n):
        curr_x, curr_r, prev = x[i], r[i], filt[i - 1]
        if np.isnan(curr_x) or np.isnan(curr_r):
            filt[i] = prev
        elif curr_x > prev:
            filt[i] = max(prev, curr_x - curr_r)
        else:
            filt[i] = min(prev, curr_x + curr_r)
    return filt

@njit("f8[:](f8[:], i4, i4)", nogil=True, cache=True)
def smooth_range(close, t, m):
    """Smoothed range calculation using double EMA"""
    n = len(close)
    diff = np.zeros(n, dtype=np.float64)
    for i in range(1, n): diff[i] = abs(close[i] - close[i - 1])

    alpha_t = 2.0 / (float(t) + 1.0)
    avrng = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        avrng[i] = (alpha_t * diff[i] + (1.0 - alpha_t) * avrng[i-1]) if not np.isnan(diff[i]) else avrng[i-1]

    alpha_w = 2.0 / (float(t * 2 - 1) + 1.0)
    smooth = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        smooth[i] = (alpha_w * avrng[i] + (1.0 - alpha_w) * smooth[i-1]) if not np.isnan(avrng[i]) else smooth[i-1]

    return smooth * float(m)

@njit("Tuple((b1[:], b1[:]))(f8[:], f8[:])", nogil=True, cache=True)
def calculate_trends_with_state(filt_x1, filt_x12):
    """Determine trend cloud states"""
    n = len(filt_x1)
    upw, dnw = np.empty(n, dtype=bool_), np.empty(n, dtype=bool_)
    upw[0] = filt_x1[0] <= filt_x12[0]
    dnw[0] = not upw[0]

    for i in range(1, n):
        if filt_x1[i] < filt_x12[i]:
            upw[i], dnw[i] = True, False
        elif filt_x1[i] > filt_x12[i]:
            upw[i], dnw[i] = False, True
        else:
            upw[i], dnw[i] = upw[i-1], dnw[i-1]
    return upw, dnw


@njit("f8[:](f8[:], i4, f8, f8)", nogil=True, cache=True)
def kalman_loop(src, length, R, Q):
    """Kalman filter implementation"""
    n = len(src)
    result = np.full(n, np.nan, dtype=np.float64)
    idx = -1
    for i in range(n):
        if not np.isnan(src[i]):
            idx = i
            break
    if idx == -1: return result

    estimate, error_est = src[idx], 1.0
    error_meas = R * max(1.0, float(length))
    q_step = Q / max(1.0, float(length))

    for i in range(idx, n):
        if not np.isnan(src[i]):
            gain = error_est / (error_est + error_meas)
            estimate = estimate + gain * (src[i] - estimate)
            error_est = (1.0 - gain) * error_est + q_step
        result[i] = estimate
    return result

# ============================================================================
# 5. MARKET & OSCILLATORS
# ============================================================================

@njit("f8[:](f8[:], f8[:], f8[:], f8[:], i8[:])", nogil=True, cache=True)
def vwap_daily_loop(high, low, close, volume, timestamps):
    n = len(close)
    vwap = np.full(n, np.nan, dtype=np.float64)
    cum_pv = cum_vol = 0.0
    last_day = -1
    last_v = np.nan

    ts_s = timestamps // 1000 if np.any(timestamps > 1e12) else timestamps

    for i in range(n):
        day = ts_s[i] // 86400
        if day != last_day:
            cum_pv = cum_vol = 0.0
            last_day = day
        
        if not (np.isnan(high[i]) or volume[i] <= 0):
            cum_pv += ((high[i] + low[i] + close[i]) / 3.0) * volume[i]
            cum_vol += volume[i]
            last_v = cum_pv / cum_vol
        vwap[i] = last_v
    return vwap

@njit("Tuple((f8[:], f8[:]))(f8[:], i4, i4, i4)", nogil=True, cache=True)
def calculate_ppo_core(close, fast, slow, signal):
    f_ma = ema_loop(close, float(fast))
    s_ma = ema_loop(close, float(slow))
    ppo = np.zeros_like(close)
    for i in range(len(close)):
        if s_ma[i] != 0 and not np.isnan(s_ma[i]):
            ppo[i] = ((f_ma[i] - s_ma[i]) / s_ma[i]) * 100.0
    return ppo, ema_loop(ppo, float(signal))


@njit("f8[:](f8[:], i4)", nogil=True, cache=True)
def calculate_rsi_core(close, period):
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    diff = np.empty(n - 1, dtype=np.float64)
    for i in range(n - 1):
        diff[i] = close[i+1] - close[i]
        
    gains = np.zeros(n - 1, dtype=np.float64)
    losses = np.zeros(n - 1, dtype=np.float64)
    
    for i in range(n - 1):
        if diff[i] > 0:
            gains[i] = diff[i]
        else:
            losses[i] = -diff[i]

    alpha = 1.0 / period
    avg_g = np.nanmean(gains[1:period+1]); avg_l = np.nanmean(losses[1:period+1])
    
    for i in range(period, n):
        avg_g = (gains[i] * alpha) + (avg_g * (1.0 - alpha))
        avg_l = (losses[i] * alpha) + (avg_l * (1.0 - alpha))
        if avg_l == 0: rsi[i] = 100.0 if avg_g > 0 else 50.0
        else: rsi[i] = 100.0 - (100.0 / (1.0 + avg_g / avg_l))
    return rsi

@njit("f8[:](f8[:], f8[:], f8[:], i4)", nogil=True, cache=True)
def calculate_atr_rma(high, low, close, period):
    tr = np.zeros_like(close)
    tr[0] = high[0] - low[0]
    for i in range(1, len(close)):
        tr[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
    return ema_loop_alpha(tr, 1.0/period)

# ============================================================================
# 6. PATTERN RECOGNITION
# ============================================================================

@njit("b1[:](f8[:], f8[:], f8[:], f8[:], f8, f8[:], f8[:], f8)", nogil=True, cache=True)
def vectorized_wick_check_buy(open_p, high_p, low_p, close_p, min_wick_ratio, atr_s, atr_l, rvol):
    n = len(close_p)
    res = np.zeros(n, dtype=bool_)
    for i in range(n):
        c_range = high_p[i] - low_p[i]
        if not np.isnan(atr_s[i]) and atr_s[i] > (atr_l[i] * rvol) and c_range > 1e-9 and close_p[i] > open_p[i]:
            res[i] = ((high_p[i] - close_p[i]) / c_range) < min_wick_ratio
    return res

@njit("b1[:](f8[:], f8[:], f8[:], f8[:], f8, f8[:], f8[:], f8)", nogil=True, cache=True)
def vectorized_wick_check_sell(open_p, high_p, low_p, close_p, min_wick_ratio, atr_s, atr_l, rvol):
    n = len(close_p)
    res = np.zeros(n, dtype=bool_)
    for i in range(n):
        c_range = high_p[i] - low_p[i]
        if not np.isnan(atr_s[i]) and atr_s[i] > (atr_l[i] * rvol) and c_range > 1e-9 and close_p[i] < open_p[i]:
            res[i] = ((close_p[i] - low_p[i]) / c_range) < min_wick_ratio
    return res

# ============================================================================
# AOT EXPORT CONFIGURATION (Import this in aot_build.py)
# ============================================================================

EXPORT_CONFIG = {
    'sanitize_array_numba': 'f8[:](f8[:], f8)',
    'sanitize_array_numba_parallel': 'f8[:](f8[:], f8)',
    'rolling_std': 'f8[:](f8[:], i4, f8)',
    'rolling_mean_numba': 'f8[:](f8[:], i4)',
    'rolling_min_max_numba': 'Tuple((f8[:], f8[:]))(f8[:], i4)',
    'calc_mmh_worm_loop': 'f8[:](f8[:], f8[:], i8)',
    'calc_mmh_value_loop': 'f8[:](f8[:], f8[:], f8[:], i4)',
    'calc_mmh_momentum_loop': 'f8[:](f8[:], i4)',
    'calc_mmh_momentum_smoothing': 'f8[:](f8[:], i4)',
    'ema_loop': 'f8[:](f8[:], f8)',
    'ema_loop_alpha': 'f8[:](f8[:], f8)',
    'rng_filter_loop': 'f8[:](f8[:], f8[:])',
    'smooth_range': 'f8[:](f8[:], i4, i4)',
    'calculate_trends_with_state': 'Tuple((b1[:], b1[:]))(f8[:], f8[:])',
    'kalman_loop': 'f8[:](f8[:], i4, f8, f8)',
    'vwap_daily_loop': 'f8[:](f8[:], f8[:], f8[:], f8[:], i8[:])',
    'calculate_ppo_core': 'Tuple((f8[:], f8[:]))(f8[:], i4, i4, i4)',
    'calculate_rsi_core': 'f8[:](f8[:], i4)',
    'calculate_atr_rma': 'f8[:](f8[:], f8[:], f8[:], i4)',
    'vectorized_wick_check_buy': 'b1[:](f8[:], f8[:], f8[:], f8[:], f8, f8[:], f8[:], f8)',
    'vectorized_wick_check_sell': 'b1[:](f8[:], f8[:], f8[:], f8[:], f8, f8[:], f8[:], f8)',
}

__all__ = list(EXPORT_CONFIG.keys())
assert len(__all__) == 21