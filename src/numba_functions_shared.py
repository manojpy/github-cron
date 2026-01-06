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
# STATISTICAL
# ============================================================================

@njit("f8[:](f8[:], i4, f8)", nogil=True, cache=True)
def rolling_std_welford(close, period, responsiveness):
    """
    Pine-accurate standard deviation using SMA-based variance.
    Matches ta.stdev() behavior:
    - Uses fixed-window SMA for mean
    - Uses population SD (divides by period, not period-1)
    - NaN in window → result is NaN (here: we return 0.0 for warmup/NaN)
    """
    n = len(close)
    sd = np.empty(n, dtype=np.float64)
    resp = 0.00001 if responsiveness < 0.00001 else (1.0 if responsiveness > 1.0 else responsiveness)

    for i in range(n):
        if i < period - 1:
            sd[i] = 0.0
            continue

        window_sum = 0.0
        window_count = 0
        has_nan = False

        start = i - period + 1
        for j in range(start, i + 1):
            val = close[j]
            if np.isnan(val):
                has_nan = True
                break
            window_sum += val
            window_count += 1

        if has_nan or window_count == 0:
            sd[i] = 0.0
            continue

        mean = window_sum / window_count

        variance_sum = 0.0
        for j in range(start, i + 1):
            val = close[j]
            diff = val - mean
            variance_sum += diff * diff

        variance = variance_sum / period
        sd[i] = np.sqrt(variance) * resp

    return sd


@njit("f8[:](f8[:], i4, f8)", nogil=True, cache=True)
def rolling_std_welford_parallel(close, period, responsiveness):
    """Pine-accurate standard deviation (parallel version)"""
    n = len(close)
    sd = np.empty(n, dtype=np.float64)
    resp = 0.00001 if responsiveness < 0.00001 else (1.0 if responsiveness > 1.0 else responsiveness)

    for i in prange(n):
        if i < period - 1:
            sd[i] = 0.0
            continue

        window_sum = 0.0
        window_count = 0
        has_nan = False

        start = i - period + 1
        for j in range(start, i + 1):
            val = close[j]
            if np.isnan(val):
                has_nan = True
                break
            window_sum += val
            window_count += 1

        if has_nan or window_count == 0:
            sd[i] = 0.0
            continue

        mean = window_sum / window_count

        variance_sum = 0.0
        for j in range(start, i + 1):
            val = close[j]
            diff = val - mean
            variance_sum += diff * diff

        variance = variance_sum / period
        sd[i] = np.sqrt(variance) * resp

    return sd


@njit("f8[:](f8[:], i4)", nogil=True, cache=True)
def rolling_mean_numba(data, period):
    """
    Pine-accurate SMA:
    - First (period-1) bars → NaN
    - Does NOT skip NaN values
    - If any NaN in window → result is NaN
    """
    n = len(data)
    out = np.empty(n, dtype=np.float64)
    out[:] = np.nan

    for i in range(n):
        if i < period - 1:
            out[i] = np.nan
            continue

        window_sum = 0.0
        has_nan = False

        start = i - period + 1
        for j in range(start, i + 1):
            val = data[j]
            if np.isnan(val):
                has_nan = True
                break
            window_sum += val

        out[i] = np.nan if has_nan else (window_sum / period)

    return out


@njit("f8[:](f8[:], i4)", nogil=True, parallel=True, cache=True)
def rolling_mean_numba_parallel(data, period):
    """Pine-accurate SMA (parallel version)"""
    n = len(data)
    out = np.empty(n, dtype=np.float64)
    out[:] = np.nan

    for i in prange(n):
        if i < period - 1:
            out[i] = np.nan
            continue

        window_sum = 0.0
        has_nan = False

        start = i - period + 1
        for j in range(start, i + 1):
            val = data[j]
            if np.isnan(val):
                has_nan = True
                break
            window_sum += val

        out[i] = np.nan if has_nan else (window_sum / period)

    return out


@njit("Tuple((f8[:], f8[:]))(f8[:], i4)", nogil=True, cache=True)
def rolling_min_max_numba(arr, period):
    """
    Rolling min/max that ignores NaN but preserves them in output.
    Pine behavior: ta.lowest() / ta.highest() ignore na values.
    """
    rows = len(arr)
    min_arr = np.empty(rows, dtype=np.float64)
    max_arr = np.empty(rows, dtype=np.float64)

    for i in range(rows):
        if i < period - 1:
            min_arr[i] = np.nan
            max_arr[i] = np.nan
            continue

        start = i - period + 1
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


@njit("Tuple((f8[:], f8[:]))(f8[:], i4)", nogil=True, parallel=True, cache=True)
def rolling_min_max_numba_parallel(arr, period):
    """Rolling min/max (parallel version)"""
    rows = len(arr)
    min_arr = np.empty(rows, dtype=np.float64)
    max_arr = np.empty(rows, dtype=np.float64)

    for i in prange(rows):
        if i < period - 1:
            min_arr[i] = np.nan
            max_arr[i] = np.nan
            continue

        start = i - period + 1
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
# MOVING AVERAGES
# ============================================================================

@njit("f8[:](f8[:], f8)", nogil=True, cache=True)
def ema_loop(data, alpha_or_period):
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
        out[i] = out[i - 1] if np.isnan(curr) else (alpha * curr + (1.0 - alpha) * out[i - 1])

    return out


@njit("f8[:](f8[:], f8)", nogil=True, cache=True)
def ema_loop_alpha(data, alpha):
    """Exponential Moving Average with explicit alpha parameter"""
    n = len(data)
    out = np.empty(n, dtype=np.float64)
    out[0] = data[0] if not np.isnan(data[0]) else 0.0

    for i in range(1, n):
        curr = data[i]
        out[i] = out[i - 1] if np.isnan(curr) else (alpha * curr + (1.0 - alpha) * out[i - 1])

    return out


# ============================================================================
# FILTERS
# ============================================================================

@njit("f8[:](f8[:], f8[:])", nogil=True, cache=True)
def rng_filter_loop(x, r):
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

        if curr_x > prev:
            candidate = curr_x - curr_r
            filt[i] = prev if candidate < prev else candidate
        else:
            candidate = curr_x + curr_r
            filt[i] = prev if candidate > prev else candidate

    return filt


@njit("f8[:](f8[:], i4, i4)", nogil=True, cache=True)
def smooth_range(close, t, m):
    """Calculate smoothed range with double EMA"""
    n = len(close)

    # Step 1: Calculate absolute differences
    diff = np.empty(n, dtype=np.float64)
    diff[0] = 0.0
    for i in range(1, n):
        diff[i] = abs(close[i] - close[i - 1])

    # Step 2: First EMA (average range)
    alpha_t = 2.0 / (t + 1.0)
    avrng = np.empty(n, dtype=np.float64)
    avrng[0] = diff[0]

    for i in range(1, n):
        curr = diff[i]
        avrng[i] = avrng[i - 1] if np.isnan(curr) else (alpha_t * curr + (1.0 - alpha_t) * avrng[i - 1])

    # Step 3: Second EMA (smoothed range)
    wper = t * 2 - 1
    alpha_w = 2.0 / (wper + 1.0)
    smoothrng = np.empty(n, dtype=np.float64)
    smoothrng[0] = avrng[0]

    for i in range(1, n):
        curr = avrng[i]
        smoothrng[i] = smoothrng[i - 1] if np.isnan(curr) else (alpha_w * curr + (1.0 - alpha_w) * smoothrng[i - 1])

    return smoothrng * float(m)


@njit("Tuple((b1[:], b1[:]))(f8[:], f8[:])", nogil=True, cache=True)
def calculate_trends_with_state(filt_x1, filt_x12):
    """
    Calculate trends with state persistence to match Pine Script visual behavior.
    When filtx1 == filtx12, maintains the previous trend state.
    Returns: (upw, dnw) boolean arrays
    """
    n = len(filt_x1)
    upw = np.empty(n, dtype=np.bool_)
    dnw = np.empty(n, dtype=np.bool_)

    if filt_x1[0] < filt_x12[0]:
        upw[0] = True
        dnw[0] = False
    elif filt_x1[0] > filt_x12[0]:
        upw[0] = False
        dnw[0] = True
    else:
        upw[0] = True
        dnw[0] = False

    for i in range(1, n):
        if filt_x1[i] < filt_x12[i]:
            upw[i] = True
            dnw[i] = False
        elif filt_x1[i] > filt_x12[i]:
            upw[i] = False
            dnw[i] = True
        else:
            upw[i] = upw[i - 1]
            dnw[i] = dnw[i - 1]

    return upw, dnw


@njit("f8[:](f8[:], i4, f8, f8)", nogil=True, cache=True)
def kalman_loop(src, length, R, Q):
    """Kalman filter implementation"""
    n = len(src)
    result = np.empty(n, dtype=np.float64)

    estimate = src[0] if not np.isnan(src[0]) else 0.0
    error_est = 1.0
    error_meas = R * (float(length) if float(length) > 1.0 else 1.0)
    Q_div_length = Q / (float(length) if float(length) > 1.0 else 1.0)

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

@njit("f8[:](f8[:], f8[:], f8[:], f8[:], i8[:])", nogil=True, cache=True)
def vwap_daily_loop(high, low, close, volume, day_id):
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

        if np.isnan(h) or np.isnan(l) or np.isnan(c) or np.isnan(v) or v <= 0.0:
            # Advance with previous vwap; first bar falls back to close
            vwap[i] = vwap[i-1] if i > 0 and not np.isnan(vwap[i-1]) else c
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

@njit("Tuple((f8[:], f8[:]))(f8[:], i4, i4, i4)", nogil=True, cache=True)
def calculate_ppo_core(close, fast, slow, signal):
    """
    Calculate Percentage Price Oscillator (PPO) and its signal line.
    Robust against NaN values in the input series.
    """
    n = len(close)
    fast_alpha = 2.0 / (fast + 1.0)
    slow_alpha = 2.0 / (slow + 1.0)

    fast_ma = np.empty(n, dtype=np.float64)
    slow_ma = np.empty(n, dtype=np.float64)

    fast_ma[0] = close[0] if not np.isnan(close[0]) else 0.0
    slow_ma[0] = close[0] if not np.isnan(close[0]) else 0.0

    for i in range(1, n):
        c = close[i]
        if np.isnan(c):
            fast_ma[i] = fast_ma[i - 1]
            slow_ma[i] = slow_ma[i - 1]
        else:
            fast_ma[i] = fast_alpha * c + (1.0 - fast_alpha) * fast_ma[i - 1]
            slow_ma[i] = slow_alpha * c + (1.0 - slow_alpha) * slow_ma[i - 1]

    ppo = np.empty(n, dtype=np.float64)
    for i in range(n):
        if slow_ma[i] != 0.0 and not np.isnan(slow_ma[i]):
            ppo[i] = ((fast_ma[i] - slow_ma[i]) / slow_ma[i]) * 100.0
        else:
            ppo[i] = 0.0

    sig_alpha = 2.0 / (signal + 1.0)
    ppo_sig = np.empty(n, dtype=np.float64)
    ppo_sig[0] = ppo[0]

    for i in range(1, n):
        p = ppo[i]
        ppo_sig[i] = ppo_sig[i - 1] if np.isnan(p) else (sig_alpha * p + (1.0 - sig_alpha) * ppo_sig[i - 1])

    return ppo, ppo_sig


@njit("f8[:](f8[:], i4)", nogil=True, cache=True)
def calculate_rsi_core(close, period):
    n = len(close)
    rsi = np.zeros(n, dtype=np.float64)
    if n <= period:
        return rsi

    gain = np.zeros(n, dtype=np.float64)
    loss = np.zeros(n, dtype=np.float64)

    last_valid_close = close[0]
    for i in range(1, n):
        curr = close[i]
        if np.isnan(curr) or np.isnan(last_valid_close):
            gain[i] = 0.0
            loss[i] = 0.0
        else:
            diff = curr - last_valid_close
            if diff > 0.0:
                gain[i] = diff
                loss[i] = 0.0
            else:
                gain[i] = 0.0
                loss[i] = -diff

        if not np.isnan(curr):
            last_valid_close = curr

    avg_gain = np.zeros(n, dtype=np.float64)
    avg_loss = np.zeros(n, dtype=np.float64)
    alpha = 1.0 / period

    sum_g = 0.0
    sum_l = 0.0
    for i in range(1, period + 1):
        sum_g += gain[i]
        sum_l += loss[i]

    avg_gain[period] = sum_g / period
    avg_loss[period] = sum_l / period

    for i in range(period + 1, n):
        if np.isnan(close[i]):
            avg_gain[i] = avg_gain[i - 1]
            avg_loss[i] = avg_loss[i - 1]
        else:
            avg_gain[i] = (gain[i] * alpha) + (avg_gain[i - 1] * (1.0 - alpha))
            avg_loss[i] = (loss[i] * alpha) + (avg_loss[i - 1] * (1.0 - alpha))

    for i in range(period, n):
        if avg_loss[i] == 0.0:
            rsi[i] = 100.0 if avg_gain[i] > 0.0 else 50.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))

    return rsi


# ============================================================================
# MMH COMPONENTS
# ============================================================================

@njit("f8[:](f8[:], f8[:], i8)", nogil=True, cache=True)
def calc_mmh_worm_loop(close_arr, sd_arr, rows):
    """
    Calculate MMH worm indicator.
    Ensures worm[0] = close[0] (fallback to 0.0 if NaN).
    """
    worm_arr = np.empty(rows, dtype=np.float64)

    worm_arr[0] = close_arr[0] if not np.isnan(close_arr[0]) else 0.0

    for i in range(1, rows):
        src = close_arr[i] if not np.isnan(close_arr[i]) else worm_arr[i - 1]
        prev_worm = worm_arr[i - 1]
        diff = src - prev_worm
        sd_i = sd_arr[i]

        if np.isnan(sd_i) or sd_i == 0.0:
            delta = diff
        else:
            delta = (np.sign(diff) * sd_i) if np.abs(diff) > sd_i else diff

        worm_arr[i] = prev_worm + delta

    return worm_arr


@njit("f8[:](f8[:], i8)", nogil=True, cache=True)
def calc_mmh_value_loop(temp_arr, rows):
    value_arr = np.zeros(rows, dtype=np.float64)
    
    t0 = temp_arr[0] if not np.isnan(temp_arr[0]) else 0.5
    value_arr[0] = np.clip(1.0 * (t0 - 0.5 + 0.5 * 0.0), -0.9999, 0.9999)  # 0.5*2=1.0 scaling

    for i in range(1, rows):
        prev_v = value_arr[i - 1]
        t = temp_arr[i] if not np.isnan(temp_arr[i]) else 0.5
        v = 1.0 * (t - 0.5 + 0.5 * prev_v)
        value_arr[i] = np.clip(v, -0.9999, 0.9999)

    return value_arr

@njit("f8[:](f8[:], i8)", nogil=True, cache=True)
def calc_mmh_momentum_loop(momentum_arr, rows):
    momentum_out = np.empty_like(momentum_arr)
    momentum_out[0] = momentum_arr[0]
    for i in range(1, rows):
        prev = momentum_out[i - 1] if not np.isnan(momentum_out[i - 1]) else 0.0
        momentum_out[i] = momentum_arr[i] + 0.5 * prev  # Pine: momentum + 0.5*nz(momentum[1])
    return momentum_out


# ============================================================================
# CANDLE PATTERN RECOGNITION
# ============================================================================

@njit("b1[:](f8[:], f8[:], f8[:], f8[:], f8)", nogil=True, cache=True)
def vectorized_wick_check_buy(open_arr, high_arr, low_arr, close_arr, min_wick_ratio):
    """Check if candles meet buy wick criteria (bullish with small upper wick)"""
    n = len(close_arr)
    result = np.zeros(n, dtype=np.bool_)

    for i in range(n):
        o = open_arr[i]
        h = high_arr[i]
        l = low_arr[i]
        c = close_arr[i]

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


@njit("b1[:](f8[:], f8[:], f8[:], f8[:], f8)", nogil=True, cache=True)
def vectorized_wick_check_sell(open_arr, high_arr, low_arr, close_arr, min_wick_ratio):
    """Check if candles meet sell wick criteria (bearish with small lower wick)"""
    n = len(close_arr)
    result = np.zeros(n, dtype=np.bool_)

    for i in range(n):
        o = open_arr[i]
        h = high_arr[i]
        l = low_arr[i]
        c = close_arr[i]

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
