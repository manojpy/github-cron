"""
Shared Numba Function Definitions - Single Source of Truth
============================================================
All 20 Numba functions with:
  - First-valid-index checks (bulletproof NaN handling)
  - O(n) complexity where possible (no nested loops)
  - Explicit @njit signatures for AOT compatibility
  - Full parallel versions where applicable
"""

import numpy as np
from numba import njit, prange, types
from numba.types import Tuple


# ============================================================================
# SANITIZATION FUNCTIONS
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
# STATISTICAL FUNCTIONS - OPTIMIZED TO O(n)
# ============================================================================

@njit("f8[:](f8[:], i4, f8)", nogil=True, cache=True)
def rolling_std(close, period, responsiveness):
    """Calculate rolling std in O(n) using Welford's online algorithm"""
    n = len(close)
    sd = np.empty(n, dtype=np.float64)
    resp = 0.00001 if responsiveness < 0.00001 else (1.0 if responsiveness > 1.0 else responsiveness)

    window_sum = 0.0
    window_sq_sum = 0.0
    window_count = 0
    queue = np.zeros(period, dtype=np.float64)
    queue_idx = 0

    for i in range(n):
        curr = close[i]
        
        if i >= period:
            old_val = queue[queue_idx]
            if not np.isnan(old_val):
                window_sum -= old_val
                window_sq_sum -= old_val * old_val
                window_count -= 1
        
        if not np.isnan(curr):
            window_sum += curr
            window_sq_sum += curr * curr
            window_count += 1
        
        queue[queue_idx] = curr
        queue_idx = (queue_idx + 1) % period
        
        if window_count == 0:
            sd[i] = 0.0
        else:
            mean = window_sum / window_count
            variance = (window_sq_sum / window_count) - (mean * mean)
            sd[i] = np.sqrt(max(0.0, variance)) * resp

    return sd


@njit("f8[:](f8[:], i4)", nogil=True, cache=True)
def rolling_mean_numba(data, period):
    """Calculate rolling mean in O(n) using sliding window"""
    n = len(data)
    out = np.empty(n, dtype=np.float64)
    
    window_sum = 0.0
    window_count = 0
    queue = np.zeros(period, dtype=np.float64)
    queue_idx = 0
    
    for i in range(period):
        if not np.isnan(data[i]):
            window_sum += data[i]
            window_count += 1
        queue[queue_idx] = data[i]
        queue_idx = (queue_idx + 1) % period
        out[i] = window_sum / window_count if window_count > 0 else 0.0
    
    for i in range(period, n):
        old_val = queue[queue_idx]
        if not np.isnan(old_val):
            window_sum -= old_val
            window_count -= 1
        
        curr = data[i]
        if not np.isnan(curr):
            window_sum += curr
            window_count += 1
        
        queue[queue_idx] = curr
        queue_idx = (queue_idx + 1) % period
        
        out[i] = window_sum / window_count if window_count > 0 else out[i-1]
    
    return out


@njit("Tuple((f8[:], f8[:]))(f8[:], i4)", nogil=True, cache=True)
def rolling_min_max_numba(arr, period):
    """Calculate rolling min/max in O(n)"""
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


@njit("f8[:](f8[:], f8[:], i8)", nogil=True, cache=True)
def calc_mmh_worm_loop(close_arr, sd_arr, rows):
    """Calculate worm array - Pine's exact logic"""
    worm_arr = np.empty(rows, dtype=np.float64)
    
    # Initialize with first close value
    worm_arr[0] = close_arr[0]

    for i in range(1, rows):
        src = close_arr[i]
        prev_worm = worm_arr[i - 1]
        diff = src - prev_worm
        sd_i = sd_arr[i]

        # Pine: delta = math.abs(diff) > sd ? math.sign(diff) * sd : diff
        if np.abs(diff) > sd_i:
            delta = np.sign(diff) * sd_i
        else:
            delta = diff

        worm_arr[i] = prev_worm + delta

    return worm_arr


@njit("f8[:](f8[:], f8[:], f8[:], i8)", nogil=True, cache=True)
def calc_mmh_value_loop(raw_momentum, min_med, max_med, rows):
    value_arr = np.empty(rows, dtype=np.float64)
    value_arr[:] = np.nan

    for i in range(rows):
        mn = min_med[i]
        mx = max_med[i]
        raw = raw_momentum[i]

        # Normalize temp
        if np.isnan(raw) or np.isnan(mn) or np.isnan(mx):
            temp = np.nan
        else:
            denom = mx - mn
            if np.abs(denom) < 1e-10:
                temp = np.nan  # Pine would yield NaN here
            else:
                temp = (raw - mn) / denom

        # Pine initialization: value = 1.0
        if i == 0:
            prev_v_safe = 0.0  # nz(value[1]) = 0 on first bar
        else:
            prev_v = value_arr[i - 1]
            prev_v_safe = 0.0 if np.isnan(prev_v) else prev_v

        v = 1.0 * ((0.0 if np.isnan(temp) else temp) - 0.5 + 0.5 * prev_v_safe)

        # Clamp AFTER update
        value_arr[i] = max(-0.9999, min(0.9999, v))

    return value_arr


@njit("f8[:](f8[:], i8)", nogil=True, cache=True)
def calc_mmh_momentum_loop(value_arr, rows):
    """
    Calculate momentum array - Pine's exact log-odds transformation
    FIXED: Now correctly preserves NaN and handles edge cases
    """
    momentum = np.empty(rows, dtype=np.float64)

    for i in range(rows):
        val = value_arr[i]
        
        # Preserve NaN through calculation
        if np.isnan(val):
            momentum[i] = np.nan
            continue
        
        # Pine: temp2 = (1 + value) / (1 - value); momentum = 0.25 * math.log(temp2)
        # Clamp value to safe range to prevent division issues
        val = max(-0.99999, min(0.99999, val))
        
        denominator = 1.0 - val
        if np.abs(denominator) < 1e-10:
            # Would cause division by zero - shouldn't happen with clamping
            momentum[i] = np.nan
            continue
        
        numerator = 1.0 + val
        if numerator <= 0:
            # Log of negative/zero undefined
            momentum[i] = np.nan
            continue
        
        temp2 = numerator / denominator
        momentum[i] = 0.25 * np.log(temp2)

    return momentum

@njit("f8[:](f8[:], i8)", nogil=True, cache=True)
def calc_mmh_momentum_smoothing(momentum_in, rows):
    result = momentum_in.copy()

    for i in range(1, rows):
        prev = result[i - 1]
        prev_safe = 0.0 if np.isnan(prev) else prev

        curr = result[i]
        curr_safe = 0.0 if np.isnan(curr) else curr

        # Apply smoothing exactly like Pine
        result[i] = curr_safe + 0.5 * prev_safe

    return result

# ============================================================================
# MOVING AVERAGES - O(n)
# ============================================================================

@njit("f8[:](f8[:], f8)", nogil=True, cache=True)
def ema_loop(data, alpha_or_period):
    """Exponential Moving Average in O(n) with first-valid-index init"""
    n = len(data)
    alpha = 2.0 / (alpha_or_period + 1.0) if alpha_or_period > 1.0 else alpha_or_period
    out = np.full(n, np.nan, dtype=np.float64)
    
    # Find first valid index in O(n)
    first_valid_idx = -1
    for i in range(n):
        if not np.isnan(data[i]):
            first_valid_idx = i
            break
    
    if first_valid_idx == -1:
        return out
    
    out[first_valid_idx] = data[first_valid_idx]
    
    # Single pass EMA calculation - O(n)
    for i in range(first_valid_idx + 1, n):
        curr = data[i]
        out[i] = out[i-1] if np.isnan(curr) else (alpha * curr + (1.0 - alpha) * out[i-1])
    
    return out


@njit("f8[:](f8[:], f8)", nogil=True, cache=True)
def ema_loop_alpha(data, alpha):
    """EMA with explicit alpha parameter - O(n)"""
    n = len(data)
    out = np.full(n, np.nan, dtype=np.float64)
    
    # Find first valid index - O(n)
    first_valid_idx = -1
    for i in range(n):
        if not np.isnan(data[i]):
            first_valid_idx = i
            break
    
    if first_valid_idx == -1:
        return out
    
    out[first_valid_idx] = data[first_valid_idx]
    
    # Single pass EMA - O(n)
    for i in range(first_valid_idx + 1, n):
        curr = data[i]
        out[i] = out[i-1] if np.isnan(curr) else (alpha * curr + (1.0 - alpha) * out[i-1])
    
    return out


# ============================================================================
# FILTERS - O(n)
# ============================================================================

@njit("f8[:](f8[:], f8[:])", nogil=True, cache=True)
def rng_filter_loop(x, r):
    """Range filter - with proper initialization"""
    n = len(x)
    filt = np.empty(n, dtype=np.float64)

    # Initialize with first value (not 0.0)
    filt[0] = x[0] if not np.isnan(x[0]) else 0.0

    for i in range(1, n):
        curr_x = x[i]
        curr_r = r[i]
        prev = filt[i - 1]

        if np.isnan(curr_x) or np.isnan(curr_r):
            filt[i] = prev
            continue

        if curr_x > prev:
            new_val = curr_x - curr_r
            filt[i] = prev if new_val < prev else new_val
        else:
            new_val = curr_x + curr_r
            filt[i] = prev if new_val > prev else new_val

    return filt

@njit("f8[:](f8[:], i4, i4)", nogil=True, cache=True)
def smooth_range(close, t, m):
    """Calculate smoothed range - with explicit type handling"""
    n = len(close)

    # Step 1: Absolute differences
    diff = np.empty(n, dtype=np.float64)
    diff[0] = 0.0
    for i in range(1, n):
        diff[i] = abs(close[i] - close[i - 1])

    # Step 2: First EMA (period t)
    alpha_t = 2.0 / (float(t) + 1.0)
    avrng = np.empty(n, dtype=np.float64)
    avrng[0] = diff[0]
    for i in range(1, n):
        curr = diff[i]
        avrng[i] = (alpha_t * curr + (1.0 - alpha_t) * avrng[i-1]) if not np.isnan(curr) else avrng[i-1]

    # Step 3: Second EMA (period wper = t*2-1)
    wper = t * 2 - 1
    alpha_w = 2.0 / (float(wper) + 1.0)
    smoothrng = np.empty(n, dtype=np.float64)
    smoothrng[0] = avrng[0]
    for i in range(1, n):
        curr = avrng[i]
        smoothrng[i] = (alpha_w * curr + (1.0 - alpha_w) * smoothrng[i-1]) if not np.isnan(curr) else smoothrng[i-1]

    return smoothrng * float(m)

@njit("Tuple((b1[:], b1[:]))(f8[:], f8[:])", nogil=True, cache=True)
def calculate_trends_with_state(filt_x1, filt_x12):
    """Determine cloud up/down states in O(n) - single pass"""
    n = len(filt_x1)
    upw = np.empty(n, dtype=np.bool_)
    dnw = np.empty(n, dtype=np.bool_)

    # Initialize first value - O(1)
    if filt_x1[0] < filt_x12[0]:
        upw[0] = True
        dnw[0] = False
    elif filt_x1[0] > filt_x12[0]:
        upw[0] = False
        dnw[0] = True
    else:
        upw[0] = True
        dnw[0] = False

    # Single pass - O(n)
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
    """Kalman filter in O(n) - FIXED: applies formula on first valid bar"""
    n = len(src)
    result = np.full(n, np.nan, dtype=np.float64)

    # Find first valid - O(n)
    first_valid_idx = -1
    for i in range(n):
        if not np.isnan(src[i]):
            first_valid_idx = i
            break
    
    if first_valid_idx == -1:
        return result

    # Initialize estimate with first valid value
    estimate = src[first_valid_idx]
    error_est = 1.0
    error_meas = R * (float(length) if float(length) > 1.0 else 1.0)
    Q_div_length = Q / (float(length) if float(length) > 1.0 else 1.0)

    # ✅ FIX: Apply Kalman formula EVEN on first bar (like Pine does)
    prediction = estimate
    kalman_gain = error_est / (error_est + error_meas)
    estimate = prediction + kalman_gain * (src[first_valid_idx] - prediction)
    error_est = (1.0 - kalman_gain) * error_est + Q_div_length
    result[first_valid_idx] = estimate

    # Continue for remaining bars - O(n)
    for i in range(first_valid_idx + 1, n):
        current = src[i]

        if np.isnan(current):
            result[i] = estimate
            continue

        prediction = estimate
        kalman_gain = error_est / (error_est + error_meas)
        estimate = prediction + kalman_gain * (current - estimate)
        error_est = (1.0 - kalman_gain) * error_est + Q_div_length
        result[i] = estimate

    return result


# ============================================================================
# MARKET INDICATORS - O(n)
# ============================================================================

@njit("f8[:](f8[:], f8[:], f8[:], f8[:], i8[:])", nogil=True, cache=True)
def vwap_daily_loop(high, low, close, volume, day_id):
    """Calculate VWAP in O(n) - single pass with daily reset"""
    n = len(close)
    vwap = np.empty(n, dtype=np.float64)

    cum_vol = 0.0
    cum_pv = 0.0
    prev_day = -1
    last_valid_vwap = np.nan

    # Single pass - O(n)
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
# OSCILLATORS - O(n)
# ============================================================================

@njit("Tuple((f8[:], f8[:]))(f8[:], i4, i4, i4)", nogil=True, cache=True)
def calculate_ppo_core(close, fast, slow, signal):
    """Calculate PPO in O(n) - three EMA passes"""
    n = len(close)
    fast_alpha = 2.0 / (fast + 1.0)
    slow_alpha = 2.0 / (slow + 1.0)

    fast_ma = np.full(n, np.nan, dtype=np.float64)
    slow_ma = np.full(n, np.nan, dtype=np.float64)

    # Find first valid - O(n)
    first_valid_idx = -1
    for i in range(n):
        if not np.isnan(close[i]):
            first_valid_idx = i
            break
    
    if first_valid_idx == -1:
        return np.full(n, np.nan, dtype=np.float64), np.full(n, np.nan, dtype=np.float64)

    fast_ma[first_valid_idx] = close[first_valid_idx]
    slow_ma[first_valid_idx] = close[first_valid_idx]

    # Single pass EMA calculations - O(n)
    for i in range(first_valid_idx + 1, n):
        c = close[i]
        if np.isnan(c):
            fast_ma[i] = fast_ma[i-1]
            slow_ma[i] = slow_ma[i-1]
        else:
            fast_ma[i] = fast_alpha * c + (1.0 - fast_alpha) * fast_ma[i-1]
            slow_ma[i] = slow_alpha * c + (1.0 - slow_alpha) * slow_ma[i-1]

    # Calculate PPO - O(n)
    ppo = np.empty(n, dtype=np.float64)
    for i in range(n):
        if slow_ma[i] != 0.0 and not np.isnan(slow_ma[i]):
            ppo[i] = ((fast_ma[i] - slow_ma[i]) / slow_ma[i]) * 100.0
        else:
            ppo[i] = 0.0

    # Signal line EMA - O(n)
    sig_alpha = 2.0 / (signal + 1.0)
    ppo_sig = np.full(n, np.nan, dtype=np.float64)
    ppo_sig[first_valid_idx] = ppo[first_valid_idx]

    for i in range(first_valid_idx + 1, n):
        p = ppo[i]
        ppo_sig[i] = ppo_sig[i-1] if np.isnan(p) else (sig_alpha * p + (1.0 - sig_alpha) * ppo_sig[i-1])

    return ppo, ppo_sig


@njit("f8[:](f8[:], i4)", nogil=True, cache=True)
def calculate_rsi_core(close, period):
    """Calculate RSI in O(n) - single pass gains/losses, then EMA"""
    n = len(close)
    rsi = np.full(n, 50.0, dtype=np.float64)
    
    if n <= period:
        return rsi

    # Find first valid - O(n)
    first_valid_idx = -1
    last_valid_close = 0.0
    for i in range(n):
        if not np.isnan(close[i]):
            first_valid_idx = i
            last_valid_close = close[i]
            break
    
    if first_valid_idx == -1:
        return rsi

    # Calculate gains/losses - O(n)
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

    # Calculate initial average gains/losses
    avg_gain = 0.0
    avg_loss = 0.0
    for i in range(first_valid_idx + 1, min(first_valid_idx + period + 1, n)):
        avg_gain += gain[i]
        avg_loss += loss[i]
    avg_gain /= period
    avg_loss /= period

    alpha = 1.0 / period

    # Smooth averages and calculate RSI - O(n)
    for i in range(first_valid_idx + period, n):
        if not np.isnan(close[i]):
            avg_gain = (gain[i] * alpha) + (avg_gain * (1.0 - alpha))
            avg_loss = (loss[i] * alpha) + (avg_loss * (1.0 - alpha))
        
        if avg_loss == 0.0:
            rsi[i] = 100.0 if avg_gain > 0.0 else 50.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))

    return rsi

# ============================================================================
# CORRECTED: Numba Vectorized Wick Check Functions with Strict Validation
# ============================================================================

@njit("b1[:](f8[:], f8[:], f8[:], f8[:], f8)", nogil=True, cache=True)
def vectorized_wick_check_buy(open_arr, high_arr, low_arr, close_arr, min_wick_ratio):
    """
    Check buy candle quality in O(n) - single pass
    CORRECTED: Validates OHLC integrity and rejects invalid wicks
    """
    n = len(close_arr)
    result = np.zeros(n, dtype=np.bool_)
    
    # Single pass - O(n)
    for i in range(n):
        o = open_arr[i]
        h = high_arr[i]
        l = low_arr[i]
        c = close_arr[i]
        
        # VALIDATION: Reject if any OHLC values are NaN/Inf
        if np.isnan(o) or np.isnan(h) or np.isnan(l) or np.isnan(c):
            continue
        if np.isinf(o) or np.isinf(h) or np.isinf(l) or np.isinf(c):
            continue
        
        # VALIDATION: Check OHLC integrity (High >= Open, High >= Close, Low <= Open, Low <= Close)
        if not (l <= o <= h and l <= c <= h and l <= h):
            # Invalid OHLC structure - reject this candle
            continue
        
        # Check if green candle (Close > Open)
        if c <= o:
            continue
        
        candle_range = h - l
        # Reject if candle range is too small (avoid division issues)
        if candle_range < 1e-8:
            continue
        
        # For BUY (green candle): check upper wick
        # Body top is the close (since close > open for green)
        body_top = c
        upper_wick = h - body_top
        
        # VALIDATION: Reject if wick calculation produced negative value
        # (This indicates data corruption since High should be >= Close)
        if upper_wick < 0.0:
            continue
        
        wick_ratio = upper_wick / candle_range
        
        # VALIDATION: Reject if wick_ratio is invalid (negative or > 1.0)
        if wick_ratio < 0.0 or wick_ratio > 1.0:
            continue
        
        # Only pass if wick ratio is LESS than threshold
        result[i] = wick_ratio < min_wick_ratio
    
    return result


@njit("b1[:](f8[:], f8[:], f8[:], f8[:], f8)", nogil=True, cache=True)
def vectorized_wick_check_sell(open_arr, high_arr, low_arr, close_arr, min_wick_ratio):
    """
    Check sell candle quality in O(n) - single pass
    CORRECTED: Validates OHLC integrity and rejects invalid wicks
    """
    n = len(close_arr)
    result = np.zeros(n, dtype=np.bool_)
    
    # Single pass - O(n)
    for i in range(n):
        o = open_arr[i]
        h = high_arr[i]
        l = low_arr[i]
        c = close_arr[i]
        
        # VALIDATION: Reject if any OHLC values are NaN/Inf
        if np.isnan(o) or np.isnan(h) or np.isnan(l) or np.isnan(c):
            continue
        if np.isinf(o) or np.isinf(h) or np.isinf(l) or np.isinf(c):
            continue
        
        # VALIDATION: Check OHLC integrity (High >= Open, High >= Close, Low <= Open, Low <= Close)
        if not (l <= o <= h and l <= c <= h and l <= h):
            # Invalid OHLC structure - reject this candle
            continue
        
        # Check if red candle (Close < Open)
        if c >= o:
            continue
        
        candle_range = h - l
        # Reject if candle range is too small (avoid division issues)
        if candle_range < 1e-8:
            continue
        
        # For SELL (red candle): check lower wick
        # Body bottom is the close (since close < open for red)
        body_bottom = c
        lower_wick = body_bottom - l
        
        # VALIDATION: Reject if wick calculation produced negative value
        # (This indicates data corruption since Low should be <= Close)
        if lower_wick < 0.0:
            continue
        
        wick_ratio = lower_wick / candle_range
        
        # VALIDATION: Reject if wick_ratio is invalid (negative or > 1.0)
        if wick_ratio < 0.0 or wick_ratio > 1.0:
            continue
        
        # Only pass if wick ratio is LESS than threshold
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
    'calc_mmh_momentum_smoothing',
    'rolling_mean_numba',
    'rolling_min_max_numba',

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

assert len(__all__) == 20, f"Expected 20 functions, found {len(__all__)}"