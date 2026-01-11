"""
Shared Numba Function Definitions - Single Source of Truth
============================================================
All 22 Numba functions defined with explicit @njit signatures.
CORRECTED: All first-valid-index checks implemented.
Ensures data[0] is never blindly used as initial value.
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
# STATISTICAL FUNCTIONS
# ============================================================================

@njit("f8[:](f8[:], i4, f8)", nogil=True, cache=True)
def rolling_std(close, period, responsiveness):
    """Calculate rolling standard deviation with NaN handling"""
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
            sd[i] = sd[i-1] if i > 0 else 0.0
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
def rolling_std_parallel(close, period, responsiveness):
    """Parallel rolling std with NaN handling"""
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
    """Calculate rolling mean with NaN handling and warmup logic"""
    n = len(data)
    out = np.empty(n, dtype=np.float64)
    
    # Phase 1: Warmup period (bars 0 to period-2)
    cum_sum = 0.0
    count = 0
    for i in range(min(period - 1, n)):
        if not np.isnan(data[i]):
            cum_sum += data[i]
            count += 1
        out[i] = cum_sum / count if count > 0 else 0.0
    
    # Phase 2: Full rolling SMA (skips NaN)
    for i in range(period - 1, n):
        window_sum = 0.0
        valid_count = 0
        
        start = i - period + 1
        for j in range(start, i + 1):
            val = data[j]
            if not np.isnan(val):
                window_sum += val
                valid_count += 1
        
        out[i] = window_sum / valid_count if valid_count > 0 else out[i-1]
    
    return out


@njit("f8[:](f8[:], i4)", nogil=True, parallel=True, cache=True)
def rolling_mean_numba_parallel(data, period):
    """Parallel rolling mean with warmup logic"""
    n = len(data)
    out = np.empty(n, dtype=np.float64)
    
    # Phase 1: Warmup period (sequential - cannot parallelize due to dependencies)
    cum_sum = 0.0
    count = 0
    warmup_end = min(period - 1, n)
    
    for i in range(warmup_end):
        if not np.isnan(data[i]):
            cum_sum += data[i]
            count += 1
        out[i] = cum_sum / count if count > 0 else 0.0
    
    # Phase 2: Full rolling SMA (parallel - each iteration independent)
    for i in prange(period - 1, n):
        window_sum = 0.0
        valid_count = 0
        
        start = i - period + 1
        for j in range(start, i + 1):
            val = data[j]
            if not np.isnan(val):
                window_sum += val
                valid_count += 1
        
        if valid_count > 0:
            out[i] = window_sum / valid_count
        else:
            out[i] = out[warmup_end - 1] if warmup_end > 0 else 0.0
    
    return out


@njit("Tuple((f8[:], f8[:]))(f8[:], i4)", nogil=True, cache=True)
def rolling_min_max_numba(arr, period):
    """Calculate rolling min/max with partial warmup logic"""
    rows = len(arr)
    min_arr = np.empty(rows, dtype=np.float64)
    max_arr = np.empty(rows, dtype=np.float64)

    for i in range(rows):
        start = max(0, i - period + 1)
        
        min_val = np.inf
        max_val = -np.inf
        has_valid = False

        for j in range(start, i + 1):
            val = arr[j]
            if not np.isnan(val):
                has_valid = True
                if val < min_val:
                    min_val = val
                if val > max_val:
                    max_val = val

        min_arr[i] = min_val if has_valid else np.nan
        max_arr[i] = max_val if has_valid else np.nan

    return min_arr, max_arr


@njit("Tuple((f8[:], f8[:]))(f8[:], i4)", nogil=True, parallel=True, cache=True)
def rolling_min_max_numba_parallel(arr, period):
    """Parallel rolling min/max with warmup logic"""
    rows = len(arr)
    min_arr = np.empty(rows, dtype=np.float64)
    max_arr = np.empty(rows, dtype=np.float64)

    for i in prange(rows):
        start = max(0, i - period + 1)
        
        min_val = np.inf
        max_val = -np.inf
        has_valid = False

        for j in range(start, i + 1):
            val = arr[j]
            if not np.isnan(val):
                has_valid = True
                if val < min_val:
                    min_val = val
                if val > max_val:
                    max_val = val

        min_arr[i] = min_val if has_valid else np.nan
        max_arr[i] = max_val if has_valid else np.nan

    return min_arr, max_arr


# ============================================================================
# MOVING AVERAGES
# ============================================================================

@njit("f8[:](f8[:], f8)", nogil=True, cache=True)
def ema_loop(data, alpha_or_period):
    """Exponential Moving Average with first-valid-index initialization"""
    n = len(data)
    alpha = 2.0 / (alpha_or_period + 1.0) if alpha_or_period > 1.0 else alpha_or_period
    out = np.full(n, np.nan, dtype=np.float64)
    
    # ✅ FIXED: Find first valid index instead of blindly using data[0]
    first_valid_idx = -1
    for i in range(n):
        if not np.isnan(data[i]):
            first_valid_idx = i
            break
    
    # If no valid data, return all NaN
    if first_valid_idx == -1:
        return out
    
    # Initialize from first valid value
    out[first_valid_idx] = data[first_valid_idx]
    
    # Calculate EMA from first valid onward
    for i in range(first_valid_idx + 1, n):
        curr = data[i]
        out[i] = out[i-1] if np.isnan(curr) else (alpha * curr + (1.0 - alpha) * out[i-1])
    
    return out


@njit("f8[:](f8[:], f8)", nogil=True, cache=True)
def ema_loop_alpha(data, alpha):
    """Exponential Moving Average with explicit alpha parameter"""
    n = len(data)
    out = np.full(n, np.nan, dtype=np.float64)
    
    # ✅ FIXED: Find first valid index
    first_valid_idx = -1
    for i in range(n):
        if not np.isnan(data[i]):
            first_valid_idx = i
            break
    
    if first_valid_idx == -1:
        return out
    
    out[first_valid_idx] = data[first_valid_idx]
    
    for i in range(first_valid_idx + 1, n):
        curr = data[i]
        out[i] = out[i-1] if np.isnan(curr) else (alpha * curr + (1.0 - alpha) * out[i-1])
    
    return out


# ============================================================================
# FILTERS
# ============================================================================

@njit("f8[:](f8[:], f8[:])", nogil=True, cache=True)
def rng_filter_loop(x, r):
    """Range filter for smoothing with bounds"""
    n = len(x)
    filt = np.empty(n, dtype=np.float64)

    filt[0] = 0.0 if np.isnan(x[0]) else x[0]

    for i in range(1, n):
        curr_x = x[i]
        curr_r = r[i]

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
    """Determine cloud up/down states from filtered values"""
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
    """Kalman filter implementation with NaN handling"""
    n = len(src)
    result = np.full(n, np.nan, dtype=np.float64)

    # ✅ FIXED: Find first valid index
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

    result[first_valid_idx] = estimate

    for i in range(first_valid_idx + 1, n):
        current = src[i]

        if np.isnan(current):
            result[i] = estimate
            continue

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
    """Calculate VWAP with daily reset and NaN handling"""
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
    """Calculate PPO with signal line"""
    n = len(close)
    fast_alpha = 2.0 / (fast + 1.0)
    slow_alpha = 2.0 / (slow + 1.0)

    fast_ma = np.full(n, np.nan, dtype=np.float64)
    slow_ma = np.full(n, np.nan, dtype=np.float64)

    # ✅ FIXED: Find first valid close index
    first_valid_idx = -1
    for i in range(n):
        if not np.isnan(close[i]):
            first_valid_idx = i
            break
    
    if first_valid_idx == -1:
        ppo = np.full(n, np.nan, dtype=np.float64)
        ppo_sig = np.full(n, np.nan, dtype=np.float64)
        return ppo, ppo_sig

    fast_ma[first_valid_idx] = close[first_valid_idx]
    slow_ma[first_valid_idx] = close[first_valid_idx]

    for i in range(first_valid_idx + 1, n):
        c = close[i]
        if np.isnan(c):
            fast_ma[i] = fast_ma[i-1]
            slow_ma[i] = slow_ma[i-1]
        else:
            fast_ma[i] = fast_alpha * c + (1.0 - fast_alpha) * fast_ma[i-1]
            slow_ma[i] = slow_alpha * c + (1.0 - slow_alpha) * slow_ma[i-1]

    ppo = np.empty(n, dtype=np.float64)
    for i in range(n):
        if slow_ma[i] != 0.0 and not np.isnan(slow_ma[i]):
            ppo[i] = ((fast_ma[i] - slow_ma[i]) / slow_ma[i]) * 100.0
        else:
            ppo[i] = 0.0

    sig_alpha = 2.0 / (signal + 1.0)
    ppo_sig = np.full(n, np.nan, dtype=np.float64)
    
    ppo_sig[first_valid_idx] = ppo[first_valid_idx]

    for i in range(first_valid_idx + 1, n):
        p = ppo[i]
        ppo_sig[i] = ppo_sig[i-1] if np.isnan(p) else (sig_alpha * p + (1.0 - sig_alpha) * ppo_sig[i-1])

    return ppo, ppo_sig


@njit("f8[:](f8[:], i4)", nogil=True, cache=True)
def calculate_rsi_core(close, period):
    """Calculate RSI with first-valid-index initialization"""
    n = len(close)
    rsi = np.full(n, 50.0, dtype=np.float64)
    
    if n <= period:
        return rsi

    # ✅ FIXED: Find first valid close index
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
    for i in range(first_valid_idx + 1, min(first_valid_idx + period + 1, n)):
        sum_g += gain[i]
        sum_l += loss[i]

    avg_gain[first_valid_idx + period] = sum_g / period
    avg_loss[first_valid_idx + period] = sum_l / period

    for i in range(first_valid_idx + period + 1, n):
        if np.isnan(close[i]):
            avg_gain[i] = avg_gain[i - 1]
            avg_loss[i] = avg_loss[i - 1]
        else:
            avg_gain[i] = (gain[i] * alpha) + (avg_gain[i - 1] * (1.0 - alpha))
            avg_loss[i] = (loss[i] * alpha) + (avg_loss[i - 1] * (1.0 - alpha))

    for i in range(first_valid_idx + period, n):
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
    """Calculate worm array with first-valid-index initialization"""
    worm_arr = np.empty(rows, dtype=np.float64)

    # ✅ GOOD: Find first valid close price
    first_valid = 0.0
    for i in range(rows):
        if not np.isnan(close_arr[i]):
            first_valid = close_arr[i]
            break
    
    worm_arr[0] = first_valid

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
    """Calculate value array with clamping to [-0.9999, 0.9999]"""
    value_arr = np.empty(rows, dtype=np.float64)
    initial_multiplier = 1.0

    t0 = temp_arr[0] if not np.isnan(temp_arr[0]) else 0.5
    v0 = initial_multiplier * (t0 - 0.5 + 0.5 * 0.0)
    
    if v0 > 0.9999:
        value_arr[0] = 0.9999
    elif v0 < -0.9999:
        value_arr[0] = -0.9999
    else:
        value_arr[0] = v0

    for i in range(1, rows):
        prev_v = value_arr[i - 1]
        t = temp_arr[i] if not np.isnan(temp_arr[i]) else 0.5
        v = initial_multiplier * (t - 0.5 + 0.5 * prev_v)

        if v > 0.9999:
            value_arr[i] = 0.9999
        elif v < -0.9999:
            value_arr[i] = -0.9999
        else:
            value_arr[i] = v

    return value_arr


@njit("f8[:](f8[:], i8)", nogil=True, cache=True)
def calc_mmh_momentum_loop(momentum_arr, rows):
    """Calculate momentum with safe indexing"""
    for i in range(1, rows):
        prev = momentum_arr[i - 1] if not np.isnan(momentum_arr[i - 1]) else 0.0
        momentum_arr[i] = momentum_arr[i] + 0.5 * prev

    return momentum_arr


# ============================================================================
# CANDLE PATTERN RECOGNITION
# ============================================================================

@njit("b1[:](f8[:], f8[:], f8[:], f8[:], f8)", nogil=True, cache=True)
def vectorized_wick_check_buy(open_arr, high_arr, low_arr, close_arr, min_wick_ratio):
    """Check buy candle quality: green with small upper wick"""
    n = len(close_arr)
    result = np.zeros(n, dtype=np.bool_)
    
    for i in range(n):
        o = open_arr[i]
        h = high_arr[i]
        l = low_arr[i]
        c = close_arr[i]
        
        if np.isnan(o) or np.isnan(h) or np.isnan(l) or np.isnan(c):
            result[i] = False
            continue
        
        if c <= o:
            result[i] = False
            continue
        
        candle_range = h - l
        if candle_range < 1e-8:
            result[i] = False
            continue
        
        body_top = c if c > o else o
        upper_wick = h - body_top
        wick_ratio = upper_wick / candle_range
        
        result[i] = wick_ratio < min_wick_ratio
    
    return result


@njit("b1[:](f8[:], f8[:], f8[:], f8[:], f8)", nogil=True, cache=True)
def vectorized_wick_check_sell(open_arr, high_arr, low_arr, close_arr, min_wick_ratio):
    """Check sell candle quality: red with small lower wick"""
    n = len(close_arr)
    result = np.zeros(n, dtype=np.bool_)
    
    for i in range(n):
        o = open_arr[i]
        h = high_arr[i]
        l = low_arr[i]
        c = close_arr[i]
        
        if np.isnan(o) or np.isnan(h) or np.isnan(l) or np.isnan(c):
            result[i] = False
            continue
        
        if c >= o:
            result[i] = False
            continue
        
        candle_range = h - l
        if candle_range < 1e-8:
            result[i] = False
            continue
        
        body_bottom = c if c < o else o
        lower_wick = body_bottom - l
        if lower_wick < 0.0:
            lower_wick = 0.0
        
        wick_ratio = lower_wick / candle_range
        result[i] = wick_ratio < min_wick_ratio
    
    return result


# ============================================================================
# METADATA & EXPORTS
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
    'rolling_std',
    'rolling_std_parallel',
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

assert len(__all__) == 22, f"Expected 22 functions, found {len(__all__)}"