"""
Shared Numba Function Definitions - Single Source of Truth
============================================================

All 24 Numba functions defined ONCE with complete implementations.
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


@njit(nogil=True, fastmath=True, cache=True)
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

@njit(nogil=True, cache=True)
def sma_loop(data: np.ndarray, period: int) -> np.ndarray:
    n = len(data)
    out = np.full(n, np.nan, dtype=np.float64)
    window_sum = 0.0

    for i in range(n):
        val = data[i]
        window_sum += val

        if i >= period:
            window_sum -= data[i - period]

        if i >= period - 1:
            out[i] = window_sum / period

    return out

@njit(nogil=True, cache=True)
def sma_loop_parallel(data: np.ndarray, period: int) -> np.ndarray:
    n = len(data)
    out = np.full(n, np.nan, dtype=np.float64)

    for i in prange(period - 1, n):
        s = 0.0
        for j in range(i - period + 1, i + 1):
            s += data[j]
        out[i] = s / period

    return out

# ============================================================================
# PUBLIC API — PARALLEL DROP-IN
# ============================================================================

@njit(nogil=True, cache=True)
def rolling_std_welford(close: np.ndarray, period: int, responsiveness: float) -> np.ndarray:
    """
    Pine-accurate ta.stdev():
    - Uses SMA mean
    - Uses population variance
    - Fixed window
    """
    n = len(close)
    sd = np.full(n, np.nan, dtype=np.float64)

    mean = sma_loop(close, period)

    for i in range(n):
        if i < period - 1:
            continue

        var_sum = 0.0
        for j in range(i - period + 1, i + 1):
            diff = close[j] - mean[i]
            var_sum += diff * diff

        sd[i] = np.sqrt(var_sum / period) * responsiveness

    return sd

@njit(nogil=True, cache=True)
def rolling_std_welford_parallel(close: np.ndarray, period: int, responsiveness: float) -> np.ndarray:
    """
    Pine-accurate ta.stdev():
    - SMA mean
    - population variance
    - fixed window
    """
    n = len(close)
    sd = np.full(n, np.nan, dtype=np.float64)
    mean = sma_loop_parallel(close, period)

    for i in prange(period - 1, n):
        var_sum = 0.0
        m = mean[i]
        for j in range(i - period + 1, i + 1):
            d = close[j] - m
            var_sum += d * d
        sd[i] = np.sqrt(var_sum / period) * responsiveness

    return sd

@njit(nogil=True, cache=True)
def rolling_mean_numba(close: np.ndarray, period: int) -> np.ndarray:
    return sma_loop(close, period)

@njit(nogil=True, cache=True)
def rolling_mean_numba_parallel(close: np.ndarray, period: int) -> np.ndarray:
    return sma_loop_parallel(close, period)


@njit(nogil=True, cache=True)
def rolling_min_max_numba(
    arr: np.ndarray,
    period: int
) -> tuple[np.ndarray, np.ndarray]:
    n = len(arr)
    min_arr = np.full(n, np.nan, dtype=np.float64)
    max_arr = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        if i < period - 1:
            continue

        lo = arr[i]
        hi = arr[i]

        for j in range(i - period + 1, i + 1):
            v = arr[j]
            if not np.isnan(v):
                if v < lo:
                    lo = v
                if v > hi:
                    hi = v

        min_arr[i] = lo
        max_arr[i] = hi

    return min_arr, max_arr


@njit(nogil=True, cache=True)
def rolling_min_max_numba_parallel(
    arr: np.ndarray,
    period: int
) -> tuple[np.ndarray, np.ndarray]:
    n = len(arr)
    min_arr = np.full(n, np.nan, dtype=np.float64)
    max_arr = np.full(n, np.nan, dtype=np.float64)

    for i in prange(n):
        if i < period - 1:
            continue

        lo = arr[i]
        hi = arr[i]

        for j in range(i - period + 1, i + 1):
            v = arr[j]
            if not np.isnan(v):
                if v < lo:
                    lo = v
                if v > hi:
                    hi = v

        min_arr[i] = lo
        max_arr[i] = hi

    return min_arr, max_arr


# ============================================================================
# STATEFUL MMH (SERIAL — PINE STYLE)
# ============================================================================

@njit(nogil=True, cache=True)
def calc_mmh_worm_loop(close_arr: np.ndarray, sd_arr: np.ndarray, rows: int) -> np.ndarray:
    worm = np.empty(rows, dtype=np.float64)
    worm[0] = close_arr[0]

    for i in range(1, rows):
        prev = worm[i - 1]
        diff = close_arr[i] - prev
        sd = sd_arr[i]

        if not np.isnan(sd) and abs(diff) > sd:
            diff = np.sign(diff) * sd

        worm[i] = prev + diff

    return worm


@njit(nogil=True, cache=True)
def calc_mmh_value_loop(temp_arr: np.ndarray, rows: int) -> np.ndarray:
    value = np.zeros(rows, dtype=np.float64)

    for i in range(rows):
        t = temp_arr[i] if not np.isnan(temp_arr[i]) else 0.5
        prev = value[i - 1] if i > 0 else 0.0

        v = (t - 0.5) + 0.5 * prev  # weight = 1

        if v > 0.9999:
            v = 0.9999
        elif v < -0.9999:
            v = -0.9999

        value[i] = v

    return value


@njit(nogil=True, cache=True)
def calc_mmh_momentum_loop(momentum_arr: np.ndarray, rows: int) -> np.ndarray:
    for i in range(1, rows):
        momentum_arr[i] += 0.5 * momentum_arr[i - 1]
    return momentum_arr

@njit(nogil=True, cache=True)
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


@njit(nogil=True, cache=True)
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

@njit(nogil=True, cache=True)
def rng_filter_loop(x: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    Exact PineScript-equivalent range filter:
    - nz(prev) behaviour
    - equality bias
    - no floating drift
    """
    n = len(x)
    filt = np.empty(n, dtype=np.float64)

    # Pine: rngfiltx1x1 = x (initial assignment)
    filt[0] = 0.0 if np.isnan(x[0]) else x[0]

    for i in range(1, n):
        curr_x = x[i]
        curr_r = r[i]

        # Pine nz()
        prev = filt[i - 1]
        if np.isnan(prev):
            prev = 0.0

        if np.isnan(curr_x) or np.isnan(curr_r):
            filt[i] = prev
            continue

        # IMPORTANT: strict Pine logic
        if curr_x > prev:
            candidate = curr_x - curr_r
            filt[i] = prev if candidate < prev else candidate
        else:
            candidate = curr_x + curr_r
            filt[i] = prev if candidate > prev else candidate

    return filt



@njit(nogil=True, cache=True)
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

@njit(nogil=True, cache=True)
def calculate_trends_with_state(filt_x1: np.ndarray, filt_x12: np.ndarray) -> tuple:
    """
    Calculate trends with state persistence to match Pine Script visual behavior.
    When filtx1 == filtx12, maintains the previous trend state.
    
    Returns:
        tuple: (upw, dnw) - Two boolean arrays
    """
    n = len(filt_x1)
    upw = np.empty(n, dtype=np.bool_)
    dnw = np.empty(n, dtype=np.bool_)
    
    # Initialize first bar - default to uptrend if equal
    if filt_x1[0] < filt_x12[0]:
        upw[0] = True
        dnw[0] = False
    elif filt_x1[0] > filt_x12[0]:
        upw[0] = False
        dnw[0] = True
    else:
        # Equal on first bar - default to uptrend
        upw[0] = True
        dnw[0] = False
    
    # Process remaining bars with state persistence
    for i in range(1, n):
        if filt_x1[i] < filt_x12[i]:
            # Clear uptrend
            upw[i] = True
            dnw[i] = False
        elif filt_x1[i] > filt_x12[i]:
            # Clear downtrend
            upw[i] = False
            dnw[i] = True
        else:
            # Equal - maintain previous state
            upw[i] = upw[i-1]
            dnw[i] = dnw[i-1]
    
    return upw, dnw

@njit(nogil=True, cache=True)
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

@njit(nogil=True, cache=True)
def vwap_daily_loop(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    day_id: np.ndarray,
) -> np.ndarray:
    """
    Volume Weighted Average Price (VWAP)
    - Uses HLC3
    - Resets on day_id change
    - TradingView-consistent behavior:
      * VWAP only updates when volume > 0
      * No price fallbacks
    """
    n = len(close)
    vwap = np.empty(n, dtype=np.float64)

    cum_vol = 0.0
    cum_pv = 0.0
    prev_day = -1
    last_valid_vwap = np.nan

    for i in range(n):
        day = day_id[i]
        if day != prev_day:
            prev_day = day
            cum_vol = 0.0
            cum_pv = 0.0
            last_valid_vwap = np.nan

        h = high[i]
        l = low[i]
        c = close[i]
        v = volume[i]

        # Skip invalid bars (TradingView behavior)
        if (
            np.isnan(h)
            or np.isnan(l)
            or np.isnan(c)
            or np.isnan(v)
            or v <= 0.0
        ):
            vwap[i] = last_valid_vwap if not np.isnan(last_valid_vwap) else c
            continue

        typical = (h + l + c) / 3.0
        cum_vol += v
        cum_pv += typical * v

        if cum_vol > 0.0:
            last_valid_vwap = cum_pv / cum_vol
            vwap[i] = last_valid_vwap
        else:
            vwap[i] = last_valid_vwap if not np.isnan(last_valid_vwap) else c

    return vwap

# ============================================================================
# OSCILLATORS
# ============================================================================

@njit(nogil=True, cache=True)
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


@njit(nogil=True, cache=True)
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

# Total: 24 functions
assert len(__all__) == 24, f"Expected 24 functions, found {len(__all__)}"