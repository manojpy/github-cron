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
    """Calculate rolling mean matching Pine's ta.sma: returns NaN for first (period - 1) bars."""
    n = len(data)
    out = np.full(n, np.nan, dtype=np.float64)

    if period <= 0:
        return out

    window_sum = 0.0
    queue = np.zeros(period, dtype=np.float64)
    queue_idx = 0

    for i in range(n):
        curr = data[i]

        # Remove old value once window is full (starting at i == period)
        if i >= period:
            old_val = queue[queue_idx]
            window_sum -= old_val

        # Add current value
        window_sum += curr
        queue[queue_idx] = curr
        queue_idx = (queue_idx + 1) % period

        # Only assign output once we have a full window (i >= period - 1)
        if i >= period - 1:
            out[i] = window_sum / period

    return out

@njit("Tuple((f8[:], f8[:]))(f8[:], i4)", nogil=True, cache=True)
def rolling_min_max_numba(arr, period):
    """Match Pine's ta.lowest/ta.highest: output na unless full window of non-nan values."""
    n = len(arr)
    min_arr = np.full(n, np.nan, dtype=np.float64)
    max_arr = np.full(n, np.nan, dtype=np.float64)

    # Use double-ended queue for indices (monotonic queues)
    min_deque = np.zeros(period, dtype=np.int32)
    max_deque = np.zeros(period, dtype=np.int32)
    min_h = min_t = 0
    max_h = max_t = 0

    # Track count of non-nan in current window
    valid_count = 0
    # Circular buffer to store validity (True/False) for each position
    valid_buffer = np.zeros(period, dtype=np.bool_)
    buf_idx = 0

    for i in range(n):
        val = arr[i]
        is_valid = not np.isnan(val)

        # Remove old element if window is full
        if i >= period:
            old_valid = valid_buffer[buf_idx]
            if old_valid:
                valid_count -= 1
                # Also remove from deques if head matches old index
                if min_h < min_t and min_deque[min_h] == i - period:
                    min_h += 1
                if max_h < max_t and max_deque[max_h] == i - period:
                    max_h += 1

        # Add new element
        valid_buffer[buf_idx] = is_valid
        if is_valid:
            valid_count += 1
            # Maintain min deque (increasing)
            while min_t > min_h and arr[min_deque[min_t - 1]] >= val:
                min_t -= 1
            min_deque[min_t] = i
            min_t += 1
            # Maintain max deque (decreasing)
            while max_t > max_h and arr[max_deque[max_t - 1]] <= val:
                max_t -= 1
            max_deque[max_t] = i
            max_t += 1

        buf_idx = (buf_idx + 1) % period

        # Only output if we have a full window of valid data
        if i >= period - 1 and valid_count == period:
            min_arr[i] = arr[min_deque[min_h]]
            max_arr[i] = arr[max_deque[max_h]]

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


@njit("f8[:](f8[:], f8[:], f8[:], i4)", nogil=True, cache=True)
def calc_mmh_value_loop(raw_arr, min_arr, max_arr, rows):
    """Corrected value loop with NaN propagation to match Pine Script recursion"""
    value_arr = np.full(rows, np.nan, dtype=np.float64)
    
    for i in range(rows):
        raw = raw_arr[i]
        mn = min_arr[i]
        mx = max_arr[i]
        
        # 1. Calculate temp (Must be NaN if inputs are NaN or range is zero)
        denom = mx - mn
        if np.isnan(raw) or np.isnan(mn) or np.isnan(mx) or np.abs(denom) < 1e-10:
            temp = np.nan
        else:
            temp = (raw - mn) / denom

        # 2. Calculate recursive value
        # In Pine, if temp is na, value becomes na.
        if np.isnan(temp):
            value_arr[i] = np.nan
        else:
            # Get previous value; use nz() logic (0.0 if previous is na)
            prev_v = value_arr[i-1] if i > 0 else np.nan
            prev_v_safe = 0.0 if np.isnan(prev_v) else prev_v
            
            # Formula: value := 1.0 * (temp - 0.5 + 0.5 * nz(value[1]))
            v = 1.0 * (temp - 0.5 + 0.5 * prev_v_safe)
            
            # Clamp to Pine limits
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
            # Formula: .25 * math.log((1 + value) / (1 - value))
            # Safe clamping for log
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
        
        # In Pine: momentum := momentum + .5 * nz(momentum[1])
        # If current momentum is na, the result of the addition is na.
        if np.isnan(curr):
            result[i] = np.nan
        else:
            prev = result[i-1] if i > 0 else np.nan
            prev_safe = 0.0 if np.isnan(prev) else prev
            result[i] = curr + 0.5 * prev_safe
            
    return result

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
    """EMA with explicit alpha parameter - with proper SMA initialization for RMA"""
    n = len(data)
    out = np.full(n, np.nan, dtype=np.float64)
    
    # Find first valid index
    first_valid_idx = -1
    for i in range(n):
        if not np.isnan(data[i]):
            first_valid_idx = i
            break
    
    if first_valid_idx == -1:
        return out
    
    # Derive period from alpha (for RMA: alpha = 1/period)
    period = int(1.0 / alpha + 0.5)
    
    # Initialize with SMA of first 'period' values if possible
    if first_valid_idx + period <= n:
        sma_sum = 0.0
        valid_count = 0
        for i in range(first_valid_idx, first_valid_idx + period):
            if not np.isnan(data[i]):
                sma_sum += data[i]
                valid_count += 1
        sma_init = sma_sum / valid_count if valid_count > 0 else data[first_valid_idx]
        
        # Fill warmup period with SMA value
        for i in range(first_valid_idx, first_valid_idx + period):
            out[i] = sma_init
        
        start_idx = first_valid_idx + period
    else:
        # Not enough data for full period, fallback
        out[first_valid_idx] = data[first_valid_idx]
        start_idx = first_valid_idx + 1
    
    # Apply EMA/RMA formula
    for i in range(start_idx, n):
        curr = data[i]
        out[i] = out[i-1] if np.isnan(curr) else (alpha * curr + (1.0 - alpha) * out[i-1])
    
    return out

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


@njit("f8[:](f8[:], f8[:], f8[:], f8[:], i8[:])", nogil=True, cache=True)
def vwap_daily_loop(high, low, close, volume, timestamps):
    
    n = len(close)
    vwap = np.full(n, np.nan, dtype=np.float64)
    cum_pv = 0.0
    cum_vol = 0.0
    last_day = -1
    last_valid_vwap = np.nan

    # Handle both second and millisecond timestamps
    ts_adj = timestamps.copy()
    if np.any(ts_adj > 1_000_000_000_000):  # ms → s
        ts_adj = ts_adj // 1000

    for i in range(n):
        current_day = ts_adj[i] // 86400

        # DAILY RESET
        if current_day != last_day:
            cum_pv = 0.0
            cum_vol = 0.0
            last_day = current_day
            last_valid_vwap = np.nan

        h, l, c, v = high[i], low[i], close[i], volume[i]

        # SKIP INVALID BARS
        if np.isnan(h) or np.isnan(l) or np.isnan(c) or np.isnan(v) or v <= 0:
            vwap[i] = last_valid_vwap
            continue

        # ACCUMULATE HLC3
        typical_price = (h + l + c) / 3.0
        cum_pv += typical_price * v
        cum_vol += v

        # CALCULATE VWAP
        if cum_vol > 0:
            vwap[i] = cum_pv / cum_vol
            last_valid_vwap = vwap[i]
        else:
            vwap[i] = last_valid_vwap

    return vwap

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

@njit("f8[:](f8[:], f8[:], f8[:], i4)", nogil=True, cache=True)
def calculate_atr_rma(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    
    n = len(close)
    if n < period:
        return np.full(n, np.nan, dtype=np.float64)
    
    # Step 1: Calculate True Range
    tr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        h = high[i]
        l = low[i]
        c = close[i - 1]  # Previous close
        
        tr1 = h - l                # High - Low
        tr2 = abs(h - c)           # Abs(High - Previous Close)
        tr3 = abs(l - c)           # Abs(Low - Previous Close)
        tr[i] = max(tr1, tr2, tr3)
    
    # Step 2: Apply RMA (Wilder's) with alpha = 1/period
    alpha = 1.0 / float(period)
    atr = ema_loop_alpha(tr, alpha)
    
    return atr

@njit("b1[:](f8[:], f8[:], f8[:], f8[:], f8, f8[:], f8[:], f8)", nogil=True, cache=True)
def vectorized_wick_check_buy(open_p, high_p, low_p, close_p, min_wick_ratio, 
                               atr_short, atr_long, rvol_threshold):
    
    n = len(close_p)
    result = np.zeros(n, dtype=np.bool_)
    
    for i in range(n):
        # GATE 1: Volatility expansion with threshold (Pine: atr1 > atr3 * rvolThreshold)
        if np.isnan(atr_short[i]) or np.isnan(atr_long[i]):
            continue
        
        if atr_short[i] <= (atr_long[i] * rvol_threshold):
            continue  # Volatility not expanding enough, skip this candle
        
        # GATE 2: Candle range
        candle_range = high_p[i] - low_p[i]
        if candle_range <= 1e-9:
            continue  # Range too small
        
        # GATE 3: Must be green candle for buy
        if close_p[i] <= open_p[i]:
            continue
        
        # GATE 4: Upper wick ratio (rejection wicks)
        upper_wick = high_p[i] - close_p[i]
        if upper_wick < 0:
            continue  # Corrupted data
        
        wick_ratio = upper_wick / candle_range
        result[i] = wick_ratio < min_wick_ratio
    
    return result


@njit("b1[:](f8[:], f8[:], f8[:], f8[:], f8, f8[:], f8[:], f8)", nogil=True, cache=True)
def vectorized_wick_check_sell(open_p, high_p, low_p, close_p, min_wick_ratio, 
                                atr_short, atr_long, rvol_threshold):
    
    n = len(close_p)
    result = np.zeros(n, dtype=np.bool_)
    
    for i in range(n):
        # GATE 1: Volatility expansion with threshold (Pine: atr1 > atr3 * rvolThreshold)
        if np.isnan(atr_short[i]) or np.isnan(atr_long[i]):
            continue
        
        if atr_short[i] <= (atr_long[i] * rvol_threshold):
            continue  # Volatility not expanding enough, skip this candle
        
        # GATE 2: Candle range
        candle_range = high_p[i] - low_p[i]
        if candle_range <= 1e-9:
            continue  # Range too small
        
        # GATE 3: Must be red candle for sell
        if close_p[i] >= open_p[i]:
            continue
        
        # GATE 4: Lower wick ratio (rejection wicks)
        lower_wick = close_p[i] - low_p[i]
        if lower_wick < 0:
            continue  # Corrupted data
        
        wick_ratio = lower_wick / candle_range
        result[i] = wick_ratio < min_wick_ratio
    
    return result

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