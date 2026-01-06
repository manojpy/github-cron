"""
Shared Numba Function Definitions - Single Source of Truth
============================================================

All 22 Numba functions defined ONCE with explicit @njit signatures.
Used by both:
  - aot_build.py (AOT via CC.export)
  - aot_bridge.py (JIT fallback)

Explicit signatures guarantee stable symbol names for AOT exports and
mathematical consistency between AOT and JIT execution paths.
"""

import numpy as np
from numba import njit, prange, types
from numba.types import Tuple


# ============================================================================
# SANITIZATION FUNCTIONS (1-2)
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
# MOVING AVERAGES (3-4)
# ============================================================================

@njit("f8[:](f8[:], i32)", nogil=True, cache=True)
def ema_loop(arr, length):
    """Standard Exponential Moving Average"""
    n = len(arr)
    out = np.zeros(n)
    if n == 0: return out
    alpha = 2.0 / (length + 1.0)
    out[0] = arr[0]
    for i in range(1, n):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i-1]
    return out

@njit("f8[:](f8[:], f8)", nogil=True, cache=True)
def ema_loop_alpha(arr, alpha):
    """EMA with custom alpha"""
    n = len(arr)
    out = np.zeros(n)
    if n == 0: return out
    out[0] = arr[0]
    for i in range(1, n):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i-1]
    return out

# ============================================================================
# FILTERS (5-8)
# ============================================================================

@njit("f8[:](f8[:], f8[:])", nogil=True, cache=True)
def rng_filter_loop(x, r):
    """Range Filter recursive logic"""
    n = len(x)
    rng_filt = np.zeros(n)
    rng_filt[0] = x[0]
    for i in range(1, n):
        if x[i] > rng_filt[i-1]:
            if x[i] - r[i] < rng_filt[i-1]:
                rng_filt[i] = rng_filt[i-1]
            else:
                rng_filt[i] = x[i] - r[i]
        else:
            if x[i] + r[i] > rng_filt[i-1]:
                rng_filt[i] = rng_filt[i-1]
            else:
                rng_filt[i] = x[i] + r[i]
    return rng_filt

@njit("f8[:](f8[:], i32, f8)", nogil=True, cache=True)
def smooth_range(arr, period, multiplier):
    """Average Range calculation for Range Filter"""
    n = len(arr)
    diff = np.zeros(n)
    for i in range(1, n):
        diff[i] = abs(arr[i] - arr[i-1])
    
    # Simple Moving Average of diff
    alpha = 1.0 / period
    sma_diff = np.zeros(n)
    sma_diff[0] = diff[0]
    for i in range(1, n):
        sma_diff[i] = alpha * diff[i] + (1.0 - alpha) * sma_diff[i-1]
    
    return sma_diff * multiplier

@njit("Tuple((i64[:], i64[:], b1[:]))(f8[:], f8[:], f8[:], f8[:], i64[:], i64[:], b1[:], f8, i32)", nogil=True, cache=True)
def calculate_trends_with_state(filt, h, l, c, up, dn, is_up, min_wick, n):
    """Trend and Wick state calculation"""
    for i in range(1, n):
        up[i] = up[i-1]
        dn[i] = dn[i-1]
        if filt[i] > filt[i-1]: up[i] += 1
        elif filt[i] < filt[i-1]: dn[i] += 1
        
        if filt[i] > filt[i-1]: is_up[i] = True
        elif filt[i] < filt[i-1]: is_up[i] = False
        else: is_up[i] = is_up[i-1]
    return up, dn, is_up

@njit("f8[:](f8[:], f8, f8)", nogil=True, cache=True)
def kalman_loop(arr, gain, k):
    """Kalman Filter recursive logic"""
    n = len(arr)
    out = np.zeros(n)
    if n == 0: return out
    out[0] = arr[0]
    velocity = 0.0
    for i in range(1, n):
        prev_filt = out[i-1]
        error = arr[i] - prev_filt
        velocity = velocity + gain * error * k
        out[i] = prev_filt + velocity + error * gain
    return out

# ============================================================================
# MARKET INDICATORS (9)
# ============================================================================

@njit("f8[:](f8[:], f8[:], i64[:])", nogil=True, cache=True)
def vwap_daily_loop(price, volume, session_id):
    """Daily Anchored VWAP"""
    n = len(price)
    out = np.zeros(n)
    pv_sum = 0.0
    vol_sum = 0.0
    for i in range(n):
        if i > 0 and session_id[i] != session_id[i-1]:
            pv_sum = 0.0
            vol_sum = 0.0
        pv_sum += price[i] * volume[i]
        vol_sum += volume[i]
        out[i] = pv_sum / vol_sum if vol_sum > 0 else price[i]
    return out

# ============================================================================
# STATISTICAL FUNCTIONS (10-15) - MATCHES PINESCRIPT ta.stdev
# ============================================================================

@njit("f8[:](f8[:], i32)", nogil=True, cache=True)
def rolling_std_welford(arr, window):
    """Matches Pine ta.stdev (Bessel's correction ddof=1)"""
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < window: return out
    for i in range(window - 1, n):
        s_sum = 0.0
        for j in range(i - window + 1, i + 1):
            s_sum += arr[j]
        mean = s_sum / window
        sq_diff = 0.0
        for j in range(i - window + 1, i + 1):
            sq_diff += (arr[j] - mean) ** 2
        out[i] = np.sqrt(sq_diff / (window - 1))
    return out

@njit("f8[:](f8[:], i32)", nogil=True, parallel=True, cache=True)
def rolling_std_welford_parallel(arr, window):
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < window: return out
    for i in prange(window - 1, n):
        s_sum = 0.0
        for j in range(i - window + 1, i + 1):
            s_sum += arr[j]
        mean = s_sum / window
        sq_diff = 0.0
        for j in range(i - window + 1, i + 1):
            sq_diff += (arr[j] - mean) ** 2
        out[i] = np.sqrt(sq_diff / (window - 1))
    return out

@njit("f8[:](f8[:], i32)", nogil=True, cache=True)
def rolling_mean_numba(arr, window):
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < window: return out
    curr_sum = 0.0
    for i in range(window): curr_sum += arr[i]
    out[window-1] = curr_sum / window
    for i in range(window, n):
        curr_sum += arr[i] - arr[i-window]
        out[i] = curr_sum / window
    return out

@njit("f8[:](f8[:], i32)", nogil=True, parallel=True, cache=True)
def rolling_mean_numba_parallel(arr, window):
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < window: return out
    for i in prange(window - 1, n):
        s = 0.0
        for j in range(i - window + 1, i + 1): s += arr[j]
        out[i] = s / window
    return out

@njit("Tuple((f8[:], f8[:]))(f8[:], i32)", nogil=True, cache=True)
def rolling_min_max_numba(arr, window):
    n = len(arr)
    mins = np.full(n, np.nan, dtype=np.float64)
    maxs = np.full(n, np.nan, dtype=np.float64)
    if n < window: return mins, maxs
    for i in range(window - 1, n):
        curr_min = arr[i]
        curr_max = arr[i]
        for j in range(i - window + 1, i):
            if arr[j] < curr_min: curr_min = arr[j]
            if arr[j] > curr_max: curr_max = arr[j]
        mins[i] = curr_min
        maxs[i] = curr_max
    return mins, maxs

@njit("Tuple((f8[:], f8[:]))(f8[:], i32)", nogil=True, parallel=True, cache=True)
def rolling_min_max_numba_parallel(arr, window):
    n = len(arr)
    mins = np.full(n, np.nan, dtype=np.float64)
    maxs = np.full(n, np.nan, dtype=np.float64)
    if n < window: return mins, maxs
    for i in prange(window - 1, n):
        c_min = arr[i]
        c_max = arr[i]
        for j in range(i - window + 1, i):
            if arr[j] < c_min: c_min = arr[j]
            if arr[j] > c_max: c_max = arr[j]
        mins[i] = c_min
        maxs[i] = c_max
    return mins, maxs

# ============================================================================
# OSCILLATORS (16-17)
# ============================================================================

@njit("f8[:](f8[:], f8[:])", nogil=True, cache=True)
def calculate_ppo_core(fast_ema, slow_ema):
    n = len(fast_ema)
    ppo = np.zeros(n)
    for i in range(n):
        if abs(slow_ema[i]) > 1e-10:
            ppo[i] = ((fast_ema[i] - slow_ema[i]) / slow_ema[i]) * 100.0
    return ppo

@njit("f8[:](f8[:], i32)", nogil=True, cache=True)
def calculate_rsi_core(arr, period):
    n = len(arr)
    rsi = np.full(n, 50.0, dtype=np.float64)
    if n <= period: return rsi
    gains = np.zeros(n)
    losses = np.zeros(n)
    for i in range(1, n):
        diff = arr[i] - arr[i-1]
        if diff > 0: gains[i] = diff
        else: losses[i] = -diff
    
    alpha = 1.0 / period
    avg_gain = 0.0
    avg_loss = 0.0
    for i in range(1, period + 1):
        avg_gain += gains[i]
        avg_loss += losses[i]
    avg_gain /= period
    avg_loss /= period
    
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss > 0:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    return rsi

# ============================================================================
# MMH COMPONENTS (18-20) - PINESCRIPT v6 ACCURATE
# ============================================================================

@njit("f8[:](f8[:], f8[:], i32)", nogil=True, cache=True)
def calc_mmh_worm_loop(source, stdev_50, n):
    out = np.zeros(n)
    if n == 0: return out
    curr = source[0]
    out[0] = curr
    for i in range(1, n):
        sd = stdev_50[i]
        diff = source[i] - curr
        if np.isnan(sd):
            delta = diff
        else:
            delta = np.sign(diff) * sd if abs(diff) > sd else diff
        curr += delta
        out[i] = curr
    return out

@njit("f8[:](f8[:], i32)", nogil=True, cache=True)
def calc_mmh_value_loop(temp, n):
    out = np.zeros(n)
    for i in range(1, n):
        # Pine: value := (temp - .5 + .5 * nz(value[1]))
        val = (temp[i] - 0.5) + (0.5 * out[i-1])
        if val > 0.9999: val = 0.9999
        elif val < -0.9999: val = -0.9999
        out[i] = val
    return out

@njit("f8[:](f8[:], i32)", nogil=True, cache=True)
def calc_mmh_momentum_loop(value_arr, n):
    out = np.zeros(n)
    for i in range(1, n):
        v = value_arr[i]
        t2 = (1.0 + v) / (1.0 - v)
        raw_mom = 0.25 * np.log(t2) if t2 > 0 else 0.0
        # Pine: momentum := momentum + .5 * nz(momentum[1])
        out[i] = raw_mom + (0.5 * out[i-1])
    return out

# ============================================================================
# PATTERN RECOGNITION (21-22)
# ============================================================================

@njit("b1[:](f8[:], f8[:], f8[:], f8[:], f8)", nogil=True, cache=True)
def vectorized_wick_check_buy(o, h, l, c, ratio):
    n = len(c)
    res = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        if c[i] <= o[i]: continue
        rng = h[i] - l[i]
        if rng < 1e-8: continue
        res[i] = ((h[i] - c[i]) / rng) < ratio
    return res

@njit("b1[:](f8[:], f8[:], f8[:], f8[:], f8)", nogil=True, cache=True)
def vectorized_wick_check_sell(o, h, l, c, ratio):
    n = len(c)
    res = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        if c[i] >= o[i]: continue
        rng = h[i] - l[i]
        if rng < 1e-8: continue
        res[i] = ((c[i] - l[i]) / rng) < ratio
    return res


# ============================================================================
# METADATA
# ============================================================================

__all__ = [
    # Sanitization
    'sanitize_array_numba',
    'sanitize_array_numba_parallel',

    # Moving Averages
    'ema_loop',
    'ema_loop_alpha',

    # Filters
    'rng_filter_loop',
    'smooth_range',
    'calculate_trends_with_state',
    'kalman_loop',

    # Market Indicators
    'vwap_daily_loop',

    # Statistical
    'rolling_std_welford',
    'rolling_std_welford_parallel',
    'rolling_mean_numba',
    'rolling_mean_numba_parallel',
    'rolling_min_max_numba',
    'rolling_min_max_numba_parallel',

    # Oscillators
    'calculate_ppo_core',
    'calculate_rsi_core',

    # MMH Components
    'calc_mmh_worm_loop',
    'calc_mmh_value_loop',
    'calc_mmh_momentum_loop',

    # Pattern Recognition
    'vectorized_wick_check_buy',
    'vectorized_wick_check_sell',
]

# Total: 22 functions
assert len(__all__) == 22, f"Expected 22 functions, found {len(__all__)}"
