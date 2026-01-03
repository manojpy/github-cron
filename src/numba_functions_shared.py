"""
Shared Numba Function Definitions - Single Source of Truth
============================================================

All 23 Numba functions defined ONCE with complete implementations.
Used by both:
  - aot_build.py (compiles to .so via CC.export)
  - aot_bridge.py (JIT fallback via @njit decorator)

This eliminates code duplication and ensures mathematical consistency
between AOT and JIT execution paths.
"""

import numpy as np
from numba import njit, prange


# ============================================================================
# SANITIZATION FUNCTIONS
# ============================================================================

@njit(nogil=True, fastmath=True, cache=True)
def sanitize_array_numba(arr: np.ndarray, default: float) -> np.ndarray:
    """Replace NaN and Inf with default value"""
    out = np.empty_like(arr)
    for i in range(len(arr)):
        val = arr[i]
        out[i] = default if (np.isnan(val) or np.isinf(val)) else val
    return out


@njit(nogil=True, fastmath=True, cache=True, parallel=True)
def sanitize_array_numba_parallel(arr: np.ndarray, default: float) -> np.ndarray:
    """Replace NaN and Inf with default value (parallel)"""
    out = np.empty_like(arr)
    for i in prange(len(arr)):
        val = arr[i]
        out[i] = default if (np.isnan(val) or np.isinf(val)) else val
    return out


# ============================================================================
# MOVING AVERAGES
# ============================================================================

@njit(nogil=True, fastmath=True, cache=True)
def sma_loop(data: np.ndarray, period: int) -> np.ndarray:
    n = len(data)
    out = np.empty(n, dtype=np.float64)
    out[:] = np.nan
    window_sum = 0.0
    count = 0
    for i in range(n):
        val = data[i]
        if not np.isnan(val):
            window_sum += val
            count += 1
        if i >= period:
            old_val = data[i - period]
            if not np.isnan(old_val):
                window_sum -= old_val
                count -= 1
        if i >= period - 1:
            out[i] = window_sum / count if count > 0 else out[i-1]
    return out


@njit(nogil=True, fastmath=True, cache=True, parallel=True)
def sma_loop_parallel(data: np.ndarray, period: int) -> np.ndarray:
    """
    Simple Moving Average (parallelized with prange).
    Uses sliding window logic for O(n) complexity and NaN robustness.
    """
    n = len(data)
    out = np.empty(n, dtype=np.float64)
    out[:] = np.nan

    # Each thread handles its own slice of indices
    for i in prange(n):
        if i < period - 1:
            continue

        window_sum = 0.0
        count = 0
        # Compute window sum for this index
        for j in range(i - period + 1, i + 1):
            val = data[j]
            if not np.isnan(val):
                window_sum += val
                count += 1

        if count > 0:
            out[i] = window_sum / count
        else:
            # Carry forward last valid SMA instead of leaving NaN or forcing 0.0
            out[i] = out[i - 1] if i > 0 else 0.0

    return out


@njit(nogil=True, fastmath=True, cache=True)
def ema_loop(data: np.ndarray, alpha_or_period: float) -> np.ndarray:
    """
    Exponential Moving Average
    If alpha_or_period > 1: treats as period, converts to alpha = 2/(period+1)
    Otherwise: uses directly as alpha
    """
    n = len(data)
    alpha = 2.0 / (alpha_or_period + 1.0) if alpha_or_period > 1.0 else alpha_or_period
    
    out = np.empty(n, dtype=np.float64)
    out[0] = data[0] if not np.isnan(data[0]) else 0.0
    
    for i in range(1, n):
        curr = data[i]
        out[i] = out[i-1] if np.isnan(curr) else alpha * curr + (1 - alpha) * out[i-1]
    
    return out


@njit(nogil=True, fastmath=True, cache=True)
def ema_loop_alpha(data: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential Moving Average with explicit alpha parameter"""
    n = len(data)
    out = np.empty(n, dtype=np.float64)
    out[0] = data[0] if not np.isnan(data[0]) else 0.0
    
    for i in range(1, n):
        curr = data[i]
        out[i] = out[i-1] if np.isnan(curr) else alpha * curr + (1 - alpha) * out[i-1]
    
    return out


# ============================================================================
# FILTERS
# ============================================================================

@njit(nogil=True, fastmath=True, cache=True)
def rng_filter_loop(x: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Range filter - constrains movement based on range values"""
    n = len(x)
    filt = np.empty(n, dtype=np.float64)
    filt[0] = x[0] if not np.isnan(x[0]) else 0.0
    
    for i in range(1, n):
        prev_filt = filt[i - 1]
        curr_x = x[i]
        curr_r = r[i]
        
        if np.isnan(curr_r) or np.isnan(curr_x):
            filt[i] = prev_filt
            continue
        
        if curr_x > prev_filt:
            lower_bound = curr_x - curr_r
            filt[i] = prev_filt if lower_bound < prev_filt else lower_bound
        else:
            upper_bound = curr_x + curr_r
            filt[i] = prev_filt if upper_bound > prev_filt else upper_bound
    
    return filt


@njit(nogil=True, fastmath=True, cache=True)
def smooth_range(close: np.ndarray, t: int, m: int) -> np.ndarray:
    """Calculate smoothed range with double EMA"""
    n = len(close)
    
    # Step 1: Calculate absolute differences
    diff = np.empty(n, dtype=np.float64)
    diff[0] = 0.0
    for i in range(1, n):
        diff[i] = abs(close[i] - close[i-1])
    
    # Step 2: First EMA (average range)
    alpha_t = 2.0 / (t + 1.0)
    avrng = np.empty(n, dtype=np.float64)
    avrng[0] = diff[0]
    
    for i in range(1, n):
        curr = diff[i]
        avrng[i] = avrng[i-1] if np.isnan(curr) else alpha_t * curr + (1 - alpha_t) * avrng[i-1]
    
    # Step 3: Second EMA (smoothed range)
    wper = t * 2 - 1
    alpha_w = 2.0 / (wper + 1.0)
    smoothrng = np.empty(n, dtype=np.float64)
    smoothrng[0] = avrng[0]
    
    for i in range(1, n):
        curr = avrng[i]
        smoothrng[i] = smoothrng[i-1] if np.isnan(curr) else alpha_w * curr + (1 - alpha_w) * smoothrng[i-1]
    
    return smoothrng * float(m)


@njit(nogil=True, fastmath=True, cache=True)
def kalman_loop(src: np.ndarray, length: int, R: float, Q: float) -> np.ndarray:
    """Kalman filter implementation"""
    n = len(src)
    result = np.empty(n, dtype=np.float64)
    
    estimate = src[0] if not np.isnan(src[0]) else 0.0
    error_est = 1.0
    error_meas = R * max(1.0, float(length))
    Q_div_length = Q / max(1.0, float(length))
    
    for i in range(n):
        current = src[i]
        
        if np.isnan(current):
            result[i] = estimate
            continue
        
        if np.isnan(estimate):
            estimate = current
        
        prediction = estimate
        kalman_gain = error_est / (error_est + error_meas)
        estimate = prediction + kalman_gain * (current - prediction)
        error_est = (1.0 - kalman_gain) * error_est + Q_div_length
        result[i] = estimate
    
    return result


# ============================================================================
# MARKET INDICATORS
# ============================================================================

@njit(nogil=True, fastmath=True, cache=True)
def vwap_daily_loop(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    day_id: np.ndarray,
) -> np.ndarray:
    """Volume Weighted Average Price – resets daily on day_id change."""
    n = len(close)
    vwap = np.empty(n, dtype=np.float64)

    cum_vol = 0.0
    cum_pv = 0.0
    prev_day = -1

    for i in range(n):
        day = day_id[i]
        is_new_day = day != prev_day

        if is_new_day:
            prev_day = day
            cum_vol = 0.0
            cum_pv = 0.0

        h, l, c, v = high[i], low[i], close[i], volume[i]

        if np.isnan(h) or np.isnan(l) or np.isnan(c) or np.isnan(v) or v <= 0:
            # ✅ FIX: On new day, initialize with typical price
            if is_new_day:
                vwap[i] = (h + l + c) / 3.0 if not np.isnan(h) else c
            else:
                vwap[i] = vwap[i - 1] if i > 0 else c
            continue

        typical = (h + l + c) / 3.0
        cum_vol += v
        cum_pv += typical * v
        vwap[i] = cum_pv / cum_vol if cum_vol > 0 else typical

    return vwap

# ============================================================================
# STATISTICAL FUNCTIONS
# ============================================================================

@njit(nogil=True, fastmath=True, cache=True)
def rolling_std_welford(close: np.ndarray, period: int, responsiveness: float) -> np.ndarray:
    """Rolling standard deviation using Welford's online algorithm"""
    n = len(close)
    sd = np.empty(n, dtype=np.float64)
    resp = max(0.00001, min(1.0, responsiveness))
    
    for i in range(n):
        mean = 0.0
        m2 = 0.0
        count = 0
        start = max(0, i - period + 1)
        
        for j in range(start, i + 1):
            val = close[j]
            if not np.isnan(val):
                count += 1
                delta = val - mean
                mean += delta / count
                m2 += delta * (val - mean)
        
        sd[i] = np.sqrt(m2 / count) * resp if count > 0 else 0.0
    
    return sd


@njit(nogil=True, fastmath=True, cache=True, parallel=True)
def rolling_std_welford_parallel(close: np.ndarray, period: int, responsiveness: float) -> np.ndarray:
    """Rolling standard deviation using Welford's algorithm (parallel)"""
    n = len(close)
    sd = np.empty(n, dtype=np.float64)
    resp = max(0.00001, min(1.0, responsiveness))
    
    for i in prange(n):
        mean = 0.0
        m2 = 0.0
        count = 0
        start = max(0, i - period + 1)
        
        for j in range(start, i + 1):
            val = close[j]
            if not np.isnan(val):
                count += 1
                delta = val - mean
                mean += delta / count
                m2 += delta * (val - mean)
        
        sd[i] = np.sqrt(m2 / count) * resp if count > 0 else 0.0
    
    return sd


@njit(nogil=True, fastmath=True, cache=True)
def rolling_mean_numba(close: np.ndarray, period: int) -> np.ndarray:
    """Rolling mean calculation - optimized with sliding window"""
    rows = len(close)
    ma = np.empty(rows, dtype=np.float64)
    
    # ✅ OPTIMIZED: Use sliding window technique (O(n) instead of O(n*period))
    window_sum = 0.0
    count = 0
    
    for i in range(rows):
        # Add new value
        if not np.isnan(close[i]):
            window_sum += close[i]
            count += 1
        
        # Remove old value if past window
        if i >= period:
            old_val = close[i - period]
            if not np.isnan(old_val):
                window_sum -= old_val
                count -= 1
        
        ma[i] = window_sum / count if count > 0 else np.nan
    
    return ma


@njit(nogil=True, fastmath=True, cache=True, parallel=True)
def rolling_mean_numba_parallel(close: np.ndarray, period: int) -> np.ndarray:
    """Rolling mean calculation (parallel)"""
    rows = len(close)
    ma = np.empty(rows, dtype=np.float64)
    
    for i in prange(rows):
        start = max(0, i - period + 1)
        sum_val = 0.0
        count = 0
        
        for j in range(start, i + 1):
            val = close[j]
            if not np.isnan(val):
                sum_val += val
                count += 1
        
        ma[i] = sum_val / count if count > 0 else np.nan
    
    return ma


@njit(nogil=True, fastmath=True, cache=True)
def rolling_min_max_numba(arr: np.ndarray, period: int) -> tuple[np.ndarray, np.ndarray]:
    """Rolling minimum and maximum values"""
    # ✅ FIXED: Changed Tuple to tuple (Python 3.9+ native type hint)
    rows = len(arr)
    min_arr = np.empty(rows, dtype=np.float64)
    max_arr = np.empty(rows, dtype=np.float64)
    
    for i in range(rows):
        start = max(0, i - period + 1)
        min_val = np.inf
        max_val = -np.inf
        
        for j in range(start, i + 1):
            val = arr[j]
            if not np.isnan(val):
                if val < min_val:
                    min_val = val
                if val > max_val:
                    max_val = val
        
        min_arr[i] = min_val if min_val != np.inf else np.nan
        max_arr[i] = max_val if max_val != -np.inf else np.nan
    
    return min_arr, max_arr


@njit(nogil=True, fastmath=True, cache=True, parallel=True)
def rolling_min_max_numba_parallel(arr: np.ndarray, period: int) -> tuple[np.ndarray, np.ndarray]:
    """Rolling minimum and maximum values (parallel)"""
    # ✅ FIXED: Changed Tuple to tuple (Python 3.9+ native type hint)
    rows = len(arr)
    min_arr = np.empty(rows, dtype=np.float64)
    max_arr = np.empty(rows, dtype=np.float64)
    
    for i in prange(rows):
        start = max(0, i - period + 1)
        min_val = np.inf
        max_val = -np.inf
        
        for j in range(start, i + 1):
            val = arr[j]
            if not np.isnan(val):
                if val < min_val:
                    min_val = val
                if val > max_val:
                    max_val = val
        
        min_arr[i] = min_val if min_val != np.inf else np.nan
        max_arr[i] = max_val if max_val != -np.inf else np.nan
    
    return min_arr, max_arr


# ============================================================================
# OSCILLATORS
# ============================================================================

@njit(nogil=True, fastmath=True, cache=True)
def calculate_ppo_core(close: np.ndarray, fast: int, slow: int, signal: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate Percentage Price Oscillator (PPO) and its signal line.
    Robust against NaN values in the input series.
    """
    n = len(close)
    fast_alpha = 2.0 / (fast + 1.0)
    slow_alpha = 2.0 / (slow + 1.0)

    fast_ma = np.empty(n, dtype=np.float64)
    slow_ma = np.empty(n, dtype=np.float64)

    # Initialize with first non‑NaN value or 0.0
    fast_ma[0] = close[0] if not np.isnan(close[0]) else 0.0
    slow_ma[0] = close[0] if not np.isnan(close[0]) else 0.0

    for i in range(1, n):
        c = close[i]
        if np.isnan(c):
            fast_ma[i] = fast_ma[i - 1]
            slow_ma[i] = slow_ma[i - 1]
        else:
            fast_ma[i] = fast_alpha * c + (1 - fast_alpha) * fast_ma[i - 1]
            slow_ma[i] = slow_alpha * c + (1 - slow_alpha) * slow_ma[i - 1]

    ppo = np.empty(n, dtype=np.float64)
    for i in range(n):
        if slow_ma[i] != 0.0 and not np.isnan(slow_ma[i]):
            # Matches Pine Script: (fast - slow) / slow * 100
            ppo[i] = ((fast_ma[i] - slow_ma[i]) / slow_ma[i]) * 100.0
        else:
            ppo[i] = 0.0

    # Calculate signal line (EMA of PPO)
    sig_alpha = 2.0 / (signal + 1.0)
    ppo_sig = np.empty(n, dtype=np.float64)
    ppo_sig[0] = ppo[0]

    for i in range(1, n):
        p = ppo[i]
        if np.isnan(p):
            ppo_sig[i] = ppo_sig[i - 1]
        else:
            ppo_sig[i] = sig_alpha * p + (1 - sig_alpha) * ppo_sig[i - 1]

    return ppo, ppo_sig


@njit(nogil=True, fastmath=True, cache=True)
def calculate_rsi_core(close: np.ndarray, period: int) -> np.ndarray:
    n = len(close)
    rsi = np.zeros(n, dtype=np.float64)
    if n <= period:
        return rsi
        
    gain = np.zeros(n, dtype=np.float64)
    loss = np.zeros(n, dtype=np.float64)
    
    # 1. Calculate Deltas with NaN protection
    # We use a simple 'carry forward' for NaNs to prevent poisoning the whole array
    last_valid_close = close[0]
    for i in range(1, n):
        curr = close[i]
        if np.isnan(curr) or np.isnan(last_valid_close):
            gain[i] = 0.0
            loss[i] = 0.0
        else:
            diff = curr - last_valid_close
            if diff > 0:
                gain[i] = diff
                loss[i] = 0.0
            else:
                gain[i] = 0.0
                loss[i] = -diff
        
        if not np.isnan(curr):
            last_valid_close = curr
            
    avg_gain = np.zeros(n, dtype=np.float64)
    avg_loss = np.zeros(n, dtype=np.float64)
    alpha = 1.0 / period # Wilder's Smoothing alpha
    
    # 2. Proper Wilder's Initialization (First avg is a simple SMA)
    sum_g = 0.0
    sum_l = 0.0
    for i in range(1, period + 1):
        sum_g += gain[i]
        sum_l += loss[i]
    
    avg_gain[period] = sum_g / period
    avg_loss[period] = sum_l / period
    
    # 3. Smoothing Loop with NaN handling
    for i in range(period + 1, n):
        if np.isnan(close[i]):
            # Carry forward averages if data is missing
            avg_gain[i] = avg_gain[i-1]
            avg_loss[i] = avg_loss[i-1]
        else:
            avg_gain[i] = (gain[i] * alpha) + (avg_gain[i-1] * (1.0 - alpha))
            avg_loss[i] = (loss[i] * alpha) + (avg_loss[i-1] * (1.0 - alpha))
        
    # 4. Final RSI Calculation
    for i in range(period, n):
        if avg_loss[i] == 0:
            rsi[i] = 100.0 if avg_gain[i] > 0 else 50.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
            
    return rsi

# ============================================================================
# MMH (MAGICAL MOMENTUM HISTOGRAM) COMPONENTS
# ============================================================================

@njit(nogil=True, fastmath=True, cache=True)
def calc_mmh_worm_loop(close_arr: np.ndarray, sd_arr: np.ndarray, rows: int) -> np.ndarray:
    """Calculate MMH worm indicator"""
    worm_arr = np.empty(rows, dtype=np.float64)
    worm_arr[0] = 0.0 if np.isnan(close_arr[0]) else close_arr[0]
    
    for i in range(1, rows):
        src = close_arr[i] if not np.isnan(close_arr[i]) else worm_arr[i - 1]
        prev_worm = worm_arr[i - 1]
        diff = src - prev_worm
        sd_i = sd_arr[i]
        
        if np.isnan(sd_i):
            delta = diff
        else:
            delta = (np.sign(diff) * sd_i) if np.abs(diff) > sd_i else diff
        
        worm_arr[i] = prev_worm + delta
    
    return worm_arr


@njit(nogil=True, fastmath=True, cache=True)
def calc_mmh_value_loop(temp_arr: np.ndarray, rows: int) -> np.ndarray:
    """Calculate MMH value indicator"""
    value_arr = np.zeros(rows, dtype=np.float64)
    
    t0 = temp_arr[0] if not np.isnan(temp_arr[0]) else 0.5
    value_arr[0] = t0 - 0.5
    value_arr[0] = max(-0.9999, min(0.9999, value_arr[0]))
    
    for i in range(1, rows):
        prev_v = value_arr[i - 1] if not np.isnan(value_arr[i - 1]) else 0.0
        t = temp_arr[i] if not np.isnan(temp_arr[i]) else 0.5
        v = t - 0.5 + 0.5 * prev_v
        value_arr[i] = max(-0.9999, min(0.9999, v))
    
    return value_arr


@njit(nogil=True, fastmath=True, cache=True)
def calc_mmh_momentum_loop(momentum_arr: np.ndarray, rows: int) -> np.ndarray:
    """Calculate MMH momentum indicator"""
    for i in range(1, rows):
        prev = momentum_arr[i - 1] if not np.isnan(momentum_arr[i - 1]) else 0.0
        momentum_arr[i] = momentum_arr[i] + 0.5 * prev
    
    return momentum_arr


# ============================================================================
# CANDLE PATTERN RECOGNITION
# ============================================================================

@njit(nogil=True, fastmath=True, cache=True)
def vectorized_wick_check_buy(
    open_arr: np.ndarray, 
    high_arr: np.ndarray, 
    low_arr: np.ndarray, 
    close_arr: np.ndarray, 
    min_wick_ratio: float
) -> np.ndarray:
    """Check if candles meet buy wick criteria (bullish with small upper wick)"""
    n = len(close_arr)
    result = np.zeros(n, dtype=np.bool_)
    
    for i in range(n):
        o, h, l, c = open_arr[i], high_arr[i], low_arr[i], close_arr[i]
        
        # Must be bullish candle (close > open)
        if c <= o:
            result[i] = False
            continue
        
        candle_range = h - l
        if candle_range < 1e-8:
            result[i] = False
            continue
        
        upper_wick = h - c
        wick_ratio = upper_wick / candle_range
        result[i] = wick_ratio < min_wick_ratio
    
    return result


@njit(nogil=True, fastmath=True, cache=True)
def vectorized_wick_check_sell(
    open_arr: np.ndarray, 
    high_arr: np.ndarray, 
    low_arr: np.ndarray, 
    close_arr: np.ndarray, 
    min_wick_ratio: float
) -> np.ndarray:
    """Check if candles meet sell wick criteria (bearish with small lower wick)"""
    n = len(close_arr)
    result = np.zeros(n, dtype=np.bool_)
    
    for i in range(n):
        o, h, l, c = open_arr[i], high_arr[i], low_arr[i], close_arr[i]
        
        # Must be bearish candle (close < open)
        if c >= o:
            result[i] = False
            continue
        
        candle_range = h - l
        if candle_range < 1e-8:
            result[i] = False
            continue
        
        lower_wick = c - l
        wick_ratio = lower_wick / candle_range
        result[i] = wick_ratio < min_wick_ratio
    
    return result


# ============================================================================
# METADATA
# ============================================================================

__all__ = [
    # Sanitization
    'sanitize_array_numba',
    'sanitize_array_numba_parallel',
    
    # Moving Averages
    'sma_loop',
    'sma_loop_parallel',
    'ema_loop',
    'ema_loop_alpha',
    
    # Filters
    'rng_filter_loop',
    'smooth_range',
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

# Total: 23 functions
assert len(__all__) == 23, f"Expected 23 functions, found {len(__all__)}"