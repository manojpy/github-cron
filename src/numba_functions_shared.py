"""
Shared Numba Function Definitions - Single Source of Truth
============================================================
All 22 Numba functions defined with explicit @njit signatures.
Fixed: Syntax error in kalman_loop.
Optimized: Parallel rolling functions changed from O(n*p) to O(n).
"""

import numpy as np
from numba import njit, prange, types
from numba.types import Tuple

# ============================================================================
# SANITIZATION FUNCTIONS
# ============================================================================

@njit("f8[:](f8[:], f8)", nogil=True, cache=True)
def sanitize_array_numba(arr, default):
    """Replace NaN and Inf with default value"""
    out = np.empty_like(arr)
    for i in range(len(arr)):
        val = arr[i]
        out[i] = default if (np.isnan(val) or np.isinf(val)) else val
    return out

@njit("f8[:](f8[:], f8)", nogil=True, parallel=True, cache=True)
def sanitize_array_numba_parallel(arr, default):
    """Replace NaN and Inf with default value (parallel)"""
    out = np.empty_like(arr)
    for i in prange(len(arr)):
        val = arr[i]
        out[i] = default if (np.isnan(val) or np.isinf(val)) else val
    return out

# ============================================================================
# STATISTICAL FUNCTIONS (O(n) Efficiency)
# ============================================================================

@njit("f8[:](f8[:], i4, f8)", nogil=True, cache=True)
def rolling_std(close, period, responsiveness):
    n = len(close)
    sd = np.zeros(n, dtype=np.float64)
    if n < period: return sd

    sum_val = 0.0
    sq_sum = 0.0
    for i in range(n):
        val = close[i]
        sum_val += val
        sq_sum += val * val
        if i >= period:
            old_val = close[i - period]
            sum_val -= old_val
            sq_sum -= old_val * old_val
        if i >= period - 1:
            mean = sum_val / period
            var = (sq_sum / period) - (mean * mean)
            sd[i] = np.sqrt(max(0.0, var)) * responsiveness
    return sd

@njit("f8[:](f8[:], i4, f8)", nogil=True, parallel=True, cache=True)
def rolling_std_parallel(close, period, responsiveness):
    """Parallel O(n) version using block-based calculation"""
    n = len(close)
    sd = np.zeros(n, dtype=np.float64)
    if n < period: return sd
    
    # Divide work into chunks for parallel threads
    num_threads = 8 # Adjusted by Numba at runtime
    chunk_size = (n + num_threads - 1) // num_threads
    
    for thread_idx in prange(num_threads):
        start_idx = thread_idx * chunk_size
        end_idx = min(start_idx + chunk_size, n)
        
        # Each thread must look back 'period' bars to initialize its window
        lookback_start = max(0, start_idx - period + 1)
        lookback_start = max(0, start_idx - period + 1)
        current_sum = 0.0
        current_sq_sum = 0.0
        for j in range(lookback_start, start_idx):
            val = close[j]
            current_sum += val
            current_sq_sum += val * val
            
        for i in range(start_idx, end_idx):
            val = close[i]
            current_sum += val
            current_sq_sum += val * val
            if i >= period:
                old = close[i - period]
                current_sum -= old
                current_sq_sum -= old * old
            if i >= period - 1:
                mean = current_sum / period
                var = (current_sq_sum / period) - (mean * mean)
                sd[i] = np.sqrt(max(0.0, var)) * responsiveness
    return sd

@njit("f8[:](f8[:], i4)", nogil=True, cache=True)
def rolling_mean_numba(data, period):
    n = len(data)
    out = np.zeros(n, dtype=np.float64)
    if n < period: return out
    current_sum = 0.0
    for i in range(n):
        current_sum += data[i]
        if i >= period:
            current_sum -= data[i - period]
        if i >= period - 1:
            out[i] = current_sum / period
    return out

@njit("f8[:](f8[:], i4)", nogil=True, parallel=True, cache=True)
def rolling_mean_numba_parallel(data, period):
    n = len(data)
    out = np.zeros(n, dtype=np.float64)
    if n < period: return out

    num_threads = 8
    chunk_size = (n + num_threads - 1) // num_threads
    for thread_idx in prange(num_threads):
        start_idx = thread_idx * chunk_size
        end_idx = min(start_idx + chunk_size, n)
        lookback_start = max(0, start_idx - period + 1)
        current_sum = 0.0
        for j in range(lookback_start, min(start_idx, n)):
            current_sum += data[j]
        for i in range(start_idx, end_idx):
            current_sum += data[i]
            if i >= period:
                current_sum -= data[i - period]
            if i >= period - 1:
                out[i] = current_sum / period
    return out

@njit("Tuple((f8[:], f8[:]))(f8[:], i4)", nogil=True, cache=True)
def rolling_min_max_numba(arr, period):
    rows = len(arr)
    min_arr = np.full(rows, np.nan, dtype=np.float64)
    max_arr = np.full(rows, np.nan, dtype=np.float64)
    for i in range(rows):
        start = max(0, i - period + 1)
        min_v, max_v = np.inf, -np.inf
        valid = False
        for j in range(start, i + 1):
            v = arr[j]
            if not np.isnan(v):
                valid = True
                if v < min_v: min_v = v
                if v > max_v: max_v = v
        if valid:
            min_arr[i], max_arr[i] = min_v, max_v
    return min_arr, max_arr

@njit("Tuple((f8[:], f8[:]))(f8[:], i4)", nogil=True, parallel=True, cache=True)
def rolling_min_max_numba_parallel(arr, period):
    rows = len(arr)
    min_arr = np.full(rows, np.nan, dtype=np.float64)
    max_arr = np.full(rows, np.nan, dtype=np.float64)
    for i in prange(rows):
        start = max(0, i - period + 1)
        min_v, max_v = np.inf, -np.inf
        valid = False
        for j in range(start, i + 1):
            v = arr[j]
            if not np.isnan(v):
                valid = True
                if v < min_v: min_v = v
                if v > max_v: max_v = v
        if valid:
            min_arr[i], max_arr[i] = min_v, max_v
    return min_arr, max_arr

# ============================================================================
# MOVING AVERAGES & FILTERS
# ============================================================================

@njit("f8[:](f8[:], f8)", nogil=True, cache=True)
def ema_loop(data, alpha_or_period):
    n = len(data)
    alpha = 2.0 / (alpha_or_period + 1.0) if alpha_or_period > 1.0 else alpha_or_period
    out = np.full(n, np.nan, dtype=np.float64)
    first_idx = -1
    for i in range(n):
        if not np.isnan(data[i]):
            first_idx = i
            break
    if first_idx == -1: return out
    out[first_idx] = data[first_idx]
    for i in range(first_idx + 1, n):
        curr = data[i]
        out[i] = out[i-1] if np.isnan(curr) else (alpha * curr + (1.0 - alpha) * out[i-1])
    return out

@njit("f8[:](f8[:], f8)", nogil=True, cache=True)
def ema_loop_alpha(data, alpha):
    return ema_loop(data, alpha)

@njit("f8[:](f8[:], f8[:])", nogil=True, cache=True)
def rng_filter_loop(x, r):
    n = len(x)
    filt = np.zeros(n, dtype=np.float64)
    first = -1
    for i in range(n):
        if not np.isnan(x[i]):
            first = i
            break
    if first == -1: return filt
    filt[first] = x[first]
    for i in range(first + 1, n):
        curr_x, curr_r, prev = x[i], r[i], filt[i-1]
        if np.isnan(curr_x) or np.isnan(curr_r):
            filt[i] = prev
            continue
        if curr_x > prev:
            filt[i] = max(prev, curr_x - curr_r)
        else:
            filt[i] = min(prev, curr_x + curr_r)
    return filt

@njit("f8[:](f8[:], i4, i4)", nogil=True, cache=True)
def smooth_range(close, t, m):
    n = len(close)
    diff = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        diff[i] = abs(close[i] - close[i-1])
    avrng = ema_loop(diff, float(t))
    smoothrng = ema_loop(avrng, float(t * 2 - 1))
    return smoothrng * float(m)

@njit("Tuple((b1[:], b1[:]))(f8[:], f8[:])", nogil=True, cache=True)
def calculate_trends_with_state(filt_x1, filt_x12):
    n = len(filt_x1)
    upw = np.zeros(n, dtype=np.bool_)
    dnw = np.zeros(n, dtype=np.bool_)
    upw[0] = filt_x1[0] < filt_x12[0]
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
    """Kalman filter implementation - FIXED SYNTAX"""
    n = len(src)
    result = np.full(n, np.nan, dtype=np.float64)
    first_idx = -1
    for i in range(n):
        if not np.isnan(src[i]):
            first_idx = i
            break
    if first_idx == -1: return result
    
    estimate = src[first_idx]
    error_est = 1.0
    val_l = float(length) if length > 1 else 1.0
    error_meas = R * val_l
    q_div = Q / val_l

    for i in range(first_idx + 1, n):
        curr = src[i]
        if np.isnan(curr):
            result[i] = estimate
            continue
        kalman_gain = error_est / (error_est + error_meas)
        estimate = estimate + kalman_gain * (curr - estimate)
        error_est = (1.0 - kalman_gain) * error_est + q_div
        result[i] = estimate
    return result

# ============================================================================
# MARKET INDICATORS & OSCILLATORS
# ============================================================================

@njit("f8[:](f8[:], f8[:], f8[:], f8[:], i8[:])", nogil=True, cache=True)
def vwap_daily_loop(high, low, close, volume, day_id):
    n = len(close)
    vwap = np.empty(n, dtype=np.float64)
    cum_vol, cum_pv, prev_day = 0.0, 0.0, -1
    last_v = np.nan
    for i in range(n):
        if day_id[i] != prev_day:
            prev_day, cum_vol, cum_pv = day_id[i], 0.0, 0.0
        h, l, c, v = high[i], low[i], close[i], volume[i]
        if np.isnan(h) or v <= 0:
            vwap[i] = vwap[i-1] if i > 0 else c
            continue
        cum_vol += v
        cum_pv += ((h + l + c) / 3.0) * v
        last_v = cum_pv / cum_vol
        vwap[i] = last_v
    return vwap

@njit("Tuple((f8[:], f8[:]))(f8[:], i4, i4, i4)", nogil=True, cache=True)
def calculate_ppo_core(close, fast, slow, signal):
    fast_ma = ema_loop(close, float(fast))
    slow_ma = ema_loop(close, float(slow))
    ppo = np.zeros_like(close)
    for i in range(len(close)):
        if slow_ma[i] != 0 and not np.isnan(slow_ma[i]):
            ppo[i] = ((fast_ma[i] - slow_ma[i]) / slow_ma[i]) * 100.0
    ppo_sig = ema_loop(ppo, float(signal))
    return ppo, ppo_sig

@njit("f8[:](f8[:], i4)", nogil=True, cache=True)
def calculate_rsi_core(close, period):
    n = len(close)
    rsi = np.zeros(n, dtype=np.float64)
    if n <= period: return rsi
    gains, losses = np.zeros(n), np.zeros(n)
    for i in range(1, n):
        diff = close[i] - close[i-1]
        if diff > 0: gains[i] = diff
        else: losses[i] = -diff
    alpha = 1.0 / period
    avg_g, avg_l = np.mean(gains[1:period+1]), np.mean(losses[1:period+1])
    for i in range(period + 1, n):
        avg_g = (gains[i] * alpha) + (avg_g * (1.0 - alpha))
        avg_l = (losses[i] * alpha) + (avg_l * (1.0 - alpha))
        if avg_l == 0: rsi[i] = 100.0 if avg_g > 0 else 50.0
        else: rsi[i] = 100.0 - (100.0 / (1.0 + avg_g/avg_l))
    return rsi

# ============================================================================
# MMH COMPONENTS & WICK CHECKS
# ============================================================================

@njit("f8[:](f8[:], f8[:], i8)", nogil=True, cache=True)
def calc_mmh_worm_loop(close_arr, sd_arr, rows):
    worm = np.zeros(rows)
    worm[0] = close_arr[0] if not np.isnan(close_arr[0]) else 0.0
    for i in range(1, rows):
        diff = close_arr[i] - worm[i-1]
        sd = sd_arr[i]
        delta = (np.sign(diff) * sd) if abs(diff) > sd else diff
        worm[i] = worm[i-1] + delta
    return worm

@njit("f8[:](f8[:], i8)", nogil=True, cache=True)
def calc_mmh_value_loop(temp, rows):
    val = np.zeros(rows)
    for i in range(rows):
        prev = val[i-1] if i > 0 else 0.0
        v = (temp[i] - 0.5 + 0.5 * prev)
        val[i] = max(-0.9999, min(0.9999, v))
    return val

@njit("f8[:](f8[:], i8)", nogil=True, cache=True)
def calc_mmh_momentum_loop(mom, rows):
    for i in range(1, rows):
        mom[i] += 0.5 * mom[i-1]
    return mom

@njit("b1[:](f8[:], f8[:], f8[:], f8[:], f8)", nogil=True, cache=True)
def vectorized_wick_check_buy(o, h, l, c, ratio):
    res = np.zeros(len(c), dtype=np.bool_)
    for i in range(len(c)):
        rng = h[i] - l[i]
        if c[i] > o[i] and rng > 1e-8:
            res[i] = ((h[i] - c[i]) / rng) < ratio
    return res

@njit("b1[:](f8[:], f8[:], f8[:], f8[:], f8)", nogil=True, cache=True)
def vectorized_wick_check_sell(o, h, l, c, ratio):
    res = np.zeros(len(c), dtype=np.bool_)
    for i in range(len(c)):
        rng = h[i] - l[i]
        if c[i] < o[i] and rng > 1e-8:
            res[i] = ((c[i] - l[i]) / rng) < ratio
    return res

__all__ = [
    'sanitize_array_numba', 'sanitize_array_numba_parallel',
    'ema_loop', 'ema_loop_alpha', 'rng_filter_loop', 'smooth_range',
    'calculate_trends_with_state', 'kalman_loop', 'vwap_daily_loop',
    'rolling_std', 'rolling_std_parallel', 'rolling_mean_numba',
    'rolling_mean_numba_parallel', 'rolling_min_max_numba',
    'rolling_min_max_numba_parallel', 'calculate_ppo_core',
    'calculate_rsi_core', 'calc_mmh_worm_loop', 'calc_mmh_value_loop',
    'calc_mmh_momentum_loop', 'vectorized_wick_check_buy', 'vectorized_wick_check_sell'
]


assert len(__all__) == 22, "Numba export list mismatch – did you forget to add/delete a function?"