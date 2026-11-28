#======================================================
# Shared Numba Function Definitions - Single Source of Truth
#======================================================

import os
import sys
import hashlib
from pathlib import Path
from importlib.metadata import version as _pkg_version

_numba_ver = _pkg_version("numba")
_cache_version = hashlib.md5(f"{_numba_ver}-{sys.version}".encode()).hexdigest()[:8]
os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    str(Path(__file__).parent / ".numba_cache" / _cache_version),
)

import numpy as np
from numba import njit, types
import logging

logger = logging.getLogger(__name__)

#=====================================================
# SIGNATURES — single source of truth, shared by the @njit decorators below
#=====================================================
f8, i4, i8 = types.float64, types.int32, types.int64

SIG_SANITIZE_ARRAY   = f8[:](f8[:], f8)
SIG_ROLLING_MEAN     = f8[:](f8[:], i4)
SIG_ROLLING_MIN_MAX  = types.Tuple((f8[:], f8[:]))(f8[:], i4)
SIG_EMA_LOOP         = f8[:](f8[:], f8)
SIG_EMA_LOOP_PINE    = f8[:](f8[:], f8)
SIG_EMA_LOOP_ALPHA   = f8[:](f8[:], f8)
SIG_KALMAN_LOOP      = f8[:](f8[:], i4, f8, f8)
SIG_VWAP_DAILY       = f8[:](f8[:], f8[:], i8[:])
SIG_PPO_CORE         = types.Tuple((f8[:], f8[:]))(f8[:], i4, i4, i4)
SIG_RSI_CORE         = f8[:](f8[:], i4)
SIG_TRUE_RANGE       = f8[:](f8[:], f8[:], f8[:])
SIG_ATR_RMA          = f8[:](f8[:], f8[:], f8[:], i4)
SIG_ADX_CORE         = f8[:](f8[:], f8[:], f8[:], i4, i4)
SIG_PERCENTILE_RANK  = f8(f8[:], i4, i4, i4, types.boolean)
SIG_DFR_DIRECTION    = types.Tuple((f8[:], f8[:]))(f8[:], f8[:], f8[:], f8)
SIG_SMOOTH_RNG       = f8[:](f8[:], i4, f8)
SIG_RANGE_FILTER     = f8[:](f8[:], f8[:])
SIG_GAUSSIAN_FILTER     = f8[:](f8[:], i4, i4)
SIG_EFFICIENCY_RATIO    = f8[:](f8[:], i4)
SIG_GAUSSIAN_TREND_LINE = types.Tuple((f8[:], i4[:]))(f8[:], f8[:], f8[:])
SIG_WMA_LOOP                 = f8[:](f8[:], i4)
SIG_HMA_LOOP                 = f8[:](f8[:], i4)
SIG_MOMENTUM_OSCILLATOR_CORE = f8[:](f8[:], f8[:], f8[:], i4)
SIG_AUTO_TRENDLINE           = types.Tuple((f8[:], f8[:], f8[:], f8[:], f8[:], f8[:]))(f8[:], f8[:], f8[:], i4)

@njit(SIG_SANITIZE_ARRAY, nogil=True, cache=True)
def sanitize_array_numba(arr, default):
    """Replace NaN and Inf with default value - O(n)"""
    out = np.empty_like(arr)
    for i in range(len(arr)):
        val = arr[i]
        out[i] = default if (np.isnan(val) or np.isinf(val)) else val
    return out

@njit(SIG_ROLLING_MEAN, nogil=True, cache=True)
def rolling_mean_numba(data, period):
    n = len(data)
    out = np.full(n, np.nan, dtype=np.float64)

    if period <= 0:
        return out
    has_nan = False
    for i in range(n):
        if np.isnan(data[i]):
            has_nan = True
            break

    if not has_nan:
        window_sum = 0.0
        for i in range(n):
            window_sum += data[i]
            if i >= period:
                window_sum -= data[i - period]
            if i >= period - 1:
                out[i] = window_sum / period
        return out

    window_sum = 0.0
    nan_count = 0
    queue = np.zeros(period, dtype=np.float64)
    is_nan_queue = np.zeros(period, dtype=np.bool_)
    queue_idx = 0

    for i in range(n):
        curr = data[i]
        curr_is_nan = np.isnan(curr)

        if i >= period:
            old_val = queue[queue_idx]
            old_is_nan = is_nan_queue[queue_idx]
            if old_is_nan:
                nan_count -= 1
            else:
                window_sum -= old_val

        if curr_is_nan:
            nan_count += 1
            queue[queue_idx] = 0.0
            is_nan_queue[queue_idx] = True
        else:
            window_sum += curr
            queue[queue_idx] = curr
            is_nan_queue[queue_idx] = False

        queue_idx = (queue_idx + 1) % period

        if i >= period - 1:
            out[i] = np.nan if nan_count > 0 else (window_sum / period)

    return out

@njit(SIG_ROLLING_MIN_MAX, nogil=True, cache=True)
def rolling_min_max_numba(arr, period):
    """Match Pine's ta.lowest/ta.highest: output na unless full window of non-nan values."""
    n = len(arr)
    if period <= 0:
        return np.full(n, np.nan, dtype=np.float64), np.full(n, np.nan, dtype=np.float64)
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

@njit(SIG_EMA_LOOP, nogil=True, cache=True)
def ema_loop(data, length_float):
    n = len(data)
    length = int(length_float)
    out = np.full(n, np.nan, dtype=np.float64)

    if length <= 0:
        return out

    alpha = 2.0 / (length + 1)

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
            out[i] = out[i - 1]
        else:
            out[i] = alpha * curr + (1.0 - alpha) * out[i - 1]

    return out

@njit(SIG_EMA_LOOP_PINE, nogil=True, cache=True)
def ema_loop_pine(data, length_float):
    n = len(data)
    length = int(length_float)
    out = np.full(n, np.nan, dtype=np.float64)

    if length <= 0:
        return out

    alpha = 2.0 / (length + 1)

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

@njit(SIG_EMA_LOOP_ALPHA, nogil=True, cache=True)
def ema_loop_alpha(data, alpha):
    n = len(data)
    out = np.full(n, np.nan, dtype=np.float64)

    if alpha <= 0.0 or alpha > 1.0:
        return out

    period = int(1.0 / alpha + 0.5)
    if period < 1:
        period = 1

    sma_sum = 0.0
    valid_count = 0
    seed_idx = -1
    for i in range(n):
        if not np.isnan(data[i]):
            sma_sum += data[i]
            valid_count += 1
            if valid_count == period:
                seed_idx = i
                break

    if seed_idx == -1:
        return out

    prev = sma_sum / period
    out[seed_idx] = prev

    for i in range(seed_idx + 1, n):
        curr = data[i]
        if not np.isnan(curr):
            prev = alpha * curr + (1.0 - alpha) * prev
        out[i] = prev

    return out


@njit(SIG_KALMAN_LOOP, nogil=True, cache=True)
def kalman_loop(src, length, R, Q):
    """Pine-exact kalman_filter(src, length, R, Q) from 'Smoothed RSI v9.0b'.

    Pine keeps `var estimate = na` and runs the SAME update on every bar, even
    while src is still na:
        if na(estimate): estimate := src[1]
        prediction := estimate
        kalman_gain := error_est / (error_est + error_meas)
        estimate := prediction + kalman_gain * (src - prediction)
        error_est := (1 - kalman_gain) * error_est + Q / length
    Consequences reproduced here: (a) error_est keeps evolving through the RSI
    warm-up bars; (b) the first non-na output appears ONE bar after src's first
    valid bar, seeded with src[1]; (c) a na src makes estimate na again, and it
    re-seeds from src[1] on the next bar.
    """
    n = len(src)
    result = np.full(n, np.nan, dtype=np.float64)

    safe_len = float(length) if float(length) > 1.0 else 1.0
    error_est = 1.0
    error_meas = R * safe_len
    q_div_length = Q / safe_len
    estimate = np.nan

    for i in range(n):
        s_prev = src[i - 1] if i > 0 else np.nan
        if np.isnan(estimate):
            estimate = s_prev
        prediction = estimate
        kalman_gain = error_est / (error_est + error_meas)
        estimate = prediction + kalman_gain * (src[i] - prediction)
        error_est = (1.0 - kalman_gain) * error_est + q_div_length
        result[i] = estimate

    return result

@njit(SIG_VWAP_DAILY, nogil=True, cache=True)
def vwap_daily_loop_safe(hlc3, volumes, timestamps):
    n = len(hlc3)
    vwap = np.empty(n, dtype=np.float64)
    cum_pv = 0.0
    cum_vol = 0.0
    last_day = -1

    for i in range(n):
        day = timestamps[i] // 86400
        if day != last_day:
            cum_pv = 0.0
            cum_vol = 0.0
            last_day = day
        cum_pv += hlc3[i] * volumes[i]
        cum_vol += volumes[i]
        # Pine: volSum > 0 ? pvSum / volSum : na  (no hlc3 fallback)
        vwap[i] = cum_pv / cum_vol if cum_vol > 0.0 else np.nan

    return vwap

@njit(SIG_PPO_CORE, nogil=True, cache=True)
def calculate_ppo_core(close, fast, slow, signal):
    n = len(close)
    fast_ma = ema_loop_pine(close, float(fast))
    slow_ma = ema_loop_pine(close, float(slow))

    ppo = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        f = fast_ma[i]
        s = slow_ma[i]
        if not np.isnan(f) and not np.isnan(s) and s != 0.0:
            ppo_val = ((f - s) / s) * 100.0
            if ppo_val > 1000.0:
                ppo_val = 1000.0
            elif ppo_val < -1000.0:
                ppo_val = -1000.0
            ppo[i] = ppo_val

    ppo_sig = ema_loop_pine(ppo, float(signal))
    return ppo, ppo_sig

# Numba
@njit(SIG_RSI_CORE, nogil=True, cache=True)
def calculate_rsi_core(close, period):
    n = len(close)
    rsi = np.full(n, np.nan, dtype=np.float64)

    if period <= 0 or n <= period:
        return rsi
    first_valid_idx = -1
    for i in range(n):
        if not np.isnan(close[i]):
            first_valid_idx = i
            break

    if first_valid_idx == -1:
        return rsi

    seed_idx = first_valid_idx + period
    if seed_idx >= n:
        return rsi

    avg_gain = 0.0
    avg_loss = 0.0
    prev_valid = close[first_valid_idx]
    valid_count = 0

    # NaN-tolerant warmup: skip NaNs, use available valid bars
    for i in range(first_valid_idx + 1, seed_idx + 1):
        curr = close[i]
        if np.isnan(curr):
            continue
        
        diff = curr - prev_valid
        if diff > 0.0:
            avg_gain += diff
        else:
            avg_loss += -diff
        
        prev_valid = curr
        valid_count += 1

    if valid_count == 0:
        return rsi

    # Use actual valid count for averaging if fewer than period
    avg_period = max(1, min(period, valid_count))
    avg_gain /= avg_period
    avg_loss /= avg_period

    # Seed RSI value
    if avg_loss == 0.0:
        rsi[seed_idx] = 100.0 if avg_gain > 0.0 else 50.0
    else:
        rs = avg_gain / avg_loss
        rsi[seed_idx] = 100.0 - (100.0 / (1.0 + rs))

    alpha = 1.0 / period       # period == 0 → inf

    # Wilder smoothing after the seed bar
    for i in range(seed_idx + 1, n):
        curr = close[i]

        if np.isnan(curr):
            rsi[i] = rsi[i - 1]
            continue

        diff = curr - prev_valid

        if diff > 0.0:
            avg_gain = (diff * alpha) + (avg_gain * (1.0 - alpha))
            avg_loss = avg_loss * (1.0 - alpha)
        else:
            avg_gain = avg_gain * (1.0 - alpha)
            avg_loss = (-diff * alpha) + (avg_loss * (1.0 - alpha))

        prev_valid = curr

        if avg_loss == 0.0:
            rsi[i] = 100.0 if avg_gain > 0.0 else 50.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))

    return rsi

@njit(SIG_TRUE_RANGE, nogil=True, cache=True)
def true_range_numba(high, low, close):
    n = len(close)
    if n == 0:
        return np.empty(0, dtype=np.float64)

    tr = np.empty(n, dtype=np.float64)

    # Bar 0 has no close[1] -> Pine ta.tr(true) falls back to high - low
    h0, l0 = high[0], low[0]
    if np.isnan(h0) or np.isnan(l0):
        tr[0] = np.nan
    else:
        tr[0] = h0 - l0

    for i in range(1, n):
        h = high[i]
        l = low[i]
        c = close[i - 1]

        if np.isnan(h) or np.isnan(l):
            # math.max(na, x) == na in Pine
            tr[i] = np.nan
        elif np.isnan(c):
            # Pine ta.tr(true): na(close[1]) -> high - low on ANY bar, not just bar 0
            tr[i] = h - l
        else:
            # All three operands are finite: max() ordering cannot matter here
            tr[i] = max(h - l, abs(h - c), abs(l - c))

    return tr

@njit(SIG_ATR_RMA, nogil=True, cache=True)
def calculate_atr_rma(high, low, close, period):
    n = len(close)
    # Guard: ensure period > 0 before dividing 1.0 / period
    if period <= 0 or n < period:
        return np.full(n, np.nan, dtype=np.float64)

    tr = true_range_numba(high, low, close)
    alpha = 1.0 / float(period)
    atr = ema_loop_alpha(tr, alpha)
    return atr

@njit(SIG_ADX_CORE, nogil=True, cache=True)
def calculate_adx_core(high, low, close, di_length, adx_length):
    n = len(high)

    # Guard: prevent zero-division in alpha_di and alpha_adx
    if di_length <= 0 or adx_length <= 0 or n < (di_length + adx_length):
        return np.full(n, np.nan, dtype=np.float64)

    # Pine ta.dmi/ADX: ta.tr (handle_na=false) is na on bar 0, and
    # up/down = ta.change(high)/-ta.change(low) are na on bar 0, so the Wilder
    # (RMA) seeds use bars 1..di_length, not 0..di_length-1.
    tr = true_range_numba(high, low, close)
    tr[0] = np.nan
    plus_dm = np.full(n, np.nan, dtype=np.float64)
    minus_dm = np.full(n, np.nan, dtype=np.float64)

    for i in range(1, n):
        h = high[i]
        l = low[i]
        prev_h = high[i - 1]
        prev_l = low[i - 1]

        up = h - prev_h
        down = prev_l - l
        plus_dm[i] = up if (up > down and up > 0) else 0.0
        minus_dm[i] = down if (down > up and down > 0) else 0.0

    alpha_di = 1.0 / float(di_length)
    plus_dm_smooth = ema_loop_alpha(plus_dm, alpha_di)
    minus_dm_smooth = ema_loop_alpha(minus_dm, alpha_di)
    tr_smooth = ema_loop_alpha(tr, alpha_di)

    # plus = fixnan(100 * rma(plusDM) / rma(tr)): x/0 is na and fixnan carries
    # the last valid value forward; before the first valid value it stays na.
    last_plus = np.nan
    last_minus = np.nan
    for i in range(n):
        if not np.isnan(tr_smooth[i]) and tr_smooth[i] > 0.0:
            last_plus = 100.0 * plus_dm_smooth[i] / tr_smooth[i]
            last_minus = 100.0 * minus_dm_smooth[i] / tr_smooth[i]
        plus_dm_smooth[i] = last_plus
        minus_dm_smooth[i] = last_minus

    # dx = abs(plus - minus) / (sum == 0 ? 1 : sum); na while plus/minus are na.
    for i in range(n):
        if np.isnan(plus_dm_smooth[i]) or np.isnan(minus_dm_smooth[i]):
            tr_smooth[i] = np.nan
        else:
            di_diff = abs(plus_dm_smooth[i] - minus_dm_smooth[i])
            di_sum = plus_dm_smooth[i] + minus_dm_smooth[i]
            tr_smooth[i] = 0.0 if di_sum == 0.0 else 100.0 * di_diff / di_sum
    alpha_adx = 1.0 / float(adx_length)
    adx = ema_loop_alpha(tr_smooth, alpha_adx)
    return adx

@njit(SIG_PERCENTILE_RANK, nogil=True, cache=True)
def percentile_rank_numba(arr, i, lookback, min_history, allow_zero):
    n = len(arr)
    if i < 0 or i >= n:
        return np.nan
    start = i - lookback
    if start < 0:
        return np.nan
    current = arr[i]
    if np.isnan(current):
        return np.nan
    if not allow_zero and current <= 0.0:
        return np.nan

    count_valid = 0
    count_lt = 0
    count_eq = 0
    for j in range(start, i):
        v = arr[j]
        if np.isnan(v):
            continue
        count_valid += 1
        if v < current:
            count_lt += 1
        elif v == current:
            count_eq += 1

    if count_valid == 0 or count_valid < min_history:
        return np.nan

    return (count_lt + 0.5 * count_eq) / count_valid

@njit(SIG_DFR_DIRECTION, nogil=True, cache=True)
def dynamic_flow_direction_loop(src, basis, dist, factor):
    n = len(src)
    direction_out = np.full(n, np.nan, dtype=np.float64)
    line_out = np.full(n, np.nan, dtype=np.float64)

    lower_band_prev = 0.0  # nz(lowerBand[1]) default
    upper_band_prev = 0.0  # nz(upperBand[1]) default
    trend_is_upper_prev = True

    for i in range(n):
        d = dist[i]
        b = basis[i]

        if np.isnan(d) or np.isnan(b):
            lower_band = lower_band_prev
            upper_band = upper_band_prev
        else:
            raw_upper = b + factor * d
            raw_lower = b - factor * d
            # FIX: Pine src[1] on bar 0 is na. na < x is na → ternary false.
            src_prev = src[i - 1] if i >= 1 else np.nan

            # Pine: lowerBand := lowerBand > prevLowerBand or src[1] < prevLowerBand ? lowerBand : prevLowerBand
            lower_band = raw_lower if (raw_lower > lower_band_prev or src_prev < lower_band_prev) else lower_band_prev
            # Pine: upperBand := upperBand < prevUpperBand or src[1] > prevUpperBand ? upperBand : prevUpperBand
            upper_band = raw_upper if (raw_upper < upper_band_prev or src_prev > upper_band_prev) else upper_band_prev

        dist_prev = dist[i - 1] if i >= 1 else np.nan

        # --- Pine direction logic -----------------------------------------
        if np.isnan(dist_prev):
            # Pine: if na(dist[1]) → _direction := 1
            direction = 1.0
        elif trend_is_upper_prev:
            # prevTrendLine was at the upper band
            direction = -1.0 if src[i] > upper_band else 1.0
        else:
            # prevTrendLine was at the lower band
            direction = 1.0 if src[i] < lower_band else -1.0
        # -----------------------------------------------------------------

        trend_is_upper_prev = (direction == 1.0)

        # Warmup suppression: only report direction once dist itself is valid.
        if not np.isnan(d):
            direction_out[i] = direction
            line_out[i] = (lower_band + upper_band) / 2.0

        # Carry forward for next bar's nz() reference.
        lower_band_prev = lower_band
        upper_band_prev = upper_band

    return direction_out, line_out

#=====================================================
# Cirrus Cloud kernels (Pine: smoothrngX1 / rngfiltx1x1)
#=====================================================
@njit(SIG_SMOOTH_RNG, nogil=True, cache=True)
def smoothrng_loop(x, t, m):
    """Pine: ta.ema(ta.ema(math.abs(x - x[1]), t), 2*t - 1) * m

    ta.ema in Pine is seeded with the FIRST source value (not an SMA), so this
    uses ema_loop_pine (never ema_loop). abs(x - x[1]) is na on bar 0.
    """
    n = len(x)
    if n == 0:
        return np.empty(0, dtype=np.float64)

    diff_arr = np.empty(n, dtype=np.float64)
    diff_arr[0] = np.nan
    for i in range(1, n):
        if np.isnan(x[i]) or np.isnan(x[i - 1]):
            diff_arr[i] = np.nan
        else:
            diff_arr[i] = abs(x[i] - x[i - 1])

    wper = t * 2 - 1
    avrng = ema_loop_pine(diff_arr, float(t))
    smoothrng = ema_loop_pine(avrng, float(wper))

    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        val = smoothrng[i]
        out[i] = np.nan if np.isnan(val) else val * m
    return out

@njit(SIG_RANGE_FILTER, nogil=True, cache=True)
def range_filter_loop(x, r):
    """Pine rngfiltx1x1:
        x > nz(prev) ? (x - r < nz(prev) ? nz(prev) : x - r)
                     : (x + r > nz(prev) ? nz(prev) : x + r)
    A na input gives a na output, and the next bar sees nz(na) == 0.
    """
    n = len(x)
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out

    prev_filt = 0.0
    for i in range(n):
        curr_x = x[i]
        curr_r = r[i]

        if np.isnan(curr_x) or np.isnan(curr_r):
            out[i] = np.nan
            prev_filt = 0.0
            continue

        if curr_x > prev_filt:
            if curr_x - curr_r < prev_filt:
                pass
            else:
                prev_filt = curr_x - curr_r
        else:
            if curr_x + curr_r > prev_filt:
                pass
            else:
                prev_filt = curr_x + curr_r

        out[i] = prev_filt
    return out

#=====================================================
# Gaussian Filter Trend kernels (Pine: QuantAlgo Gaussian Filter Trend)
#=====================================================

@njit(SIG_GAUSSIAN_FILTER, nogil=True, cache=True)
def gaussian_filter_loop(src, length, poles):
    n = len(src)
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0 or length < 1 or poles < 1:
        return out
    PI = 3.14159265358979323846
    beta  = (1.0 - np.cos(2.0 * PI / float(length))) / \
            (1.414 ** (2.0 / float(poles)) - 1.0)
    alpha = -beta + np.sqrt(beta * beta + 2.0 * beta)
    prev_s1 = np.nan
    prev_s2 = np.nan
    prev_s3 = np.nan
    prev_s4 = np.nan
    for i in range(n):
        curr_src = src[i]
        if np.isnan(curr_src):
            out[i] = np.nan
            prev_s1 = prev_s2 = prev_s3 = prev_s4 = np.nan
            continue
        s1 = alpha * curr_src + (1.0 - alpha) * (curr_src if np.isnan(prev_s1) else prev_s1)
        s2 = alpha * s1       + (1.0 - alpha) * (curr_src if np.isnan(prev_s2) else prev_s2)
        s3 = alpha * s2       + (1.0 - alpha) * (curr_src if np.isnan(prev_s3) else prev_s3)
        s4 = alpha * s3       + (1.0 - alpha) * (curr_src if np.isnan(prev_s4) else prev_s4)
        if poles <= 1:
            out[i] = s1
        elif poles == 2:
            out[i] = s2
        elif poles == 3:
            out[i] = s3
        else:
            out[i] = s4
        prev_s1, prev_s2, prev_s3, prev_s4 = s1, s2, s3, s4
    return out

@njit(SIG_EFFICIENCY_RATIO, nogil=True, cache=True)
def efficiency_ratio_loop(src, length):
    n = len(src)
    er = np.full(n, np.nan, dtype=np.float64)
    if n == 0 or length < 1:
        return er

    buf = np.zeros(length, dtype=np.float64)
    filled = 0
    pos = 0
    path_sum = 0.0

    for i in range(1, n):
        if not np.isnan(src[i]) and not np.isnan(src[i - 1]):
            d = abs(src[i] - src[i - 1])
            if filled == length:
                path_sum -= buf[pos]
                buf[pos] = d
                path_sum += d
                pos = (pos + 1) % length
            else:
                buf[filled] = d
                path_sum += d
                filled += 1
                pos = filled % length

        if filled == length:
            if path_sum == 0.0:
                er[i] = 0.0
            elif i >= length and not np.isnan(src[i]) and not np.isnan(src[i - length]):
                er[i] = abs(src[i] - src[i - length]) / path_sum

    return er

@njit(SIG_GAUSSIAN_TREND_LINE, nogil=True, cache=True)
def gaussian_trend_line_loop(filtered, upper, lower):
    n = len(filtered)
    tl = np.full(n, np.nan, dtype=np.float64)
    dir_arr = np.zeros(n, dtype=np.int32)
    if n == 0:
        return tl, dir_arr
    curr_tl = filtered[0]
    if not np.isnan(upper[0]) and not np.isnan(curr_tl) and upper[0] < curr_tl:
        curr_tl = upper[0]
    if not np.isnan(lower[0]) and not np.isnan(curr_tl) and lower[0] > curr_tl:
        curr_tl = lower[0]
    tl[0] = curr_tl
    dir_arr[0] = 0
    for i in range(1, n):
        prev_tl = tl[i - 1]
        curr_tl = filtered[i] if np.isnan(prev_tl) else prev_tl
        if not np.isnan(upper[i]) and not np.isnan(curr_tl) and upper[i] < curr_tl:
            curr_tl = upper[i]
        if not np.isnan(lower[i]) and not np.isnan(curr_tl) and lower[i] > curr_tl:
            curr_tl = lower[i]
        tl[i] = curr_tl
        if not np.isnan(curr_tl) and not np.isnan(prev_tl):
            if curr_tl > prev_tl:
                dir_arr[i] = 1
            elif curr_tl < prev_tl:
                dir_arr[i] = -1
            else:
                dir_arr[i] = dir_arr[i - 1]
        else:
            dir_arr[i] = dir_arr[i - 1]
    return tl, dir_arr

#=====================================================
# Momentum Oscillator kernels (Pine: AlgoAlpha Momentum Oscillator)
#=====================================================

@njit(SIG_WMA_LOOP, nogil=True, cache=True)
def wma_loop(src, length):
    """Weighted Moving Average - strict NaN propagation."""
    n = len(src)
    out = np.full(n, np.nan, dtype=np.float64)
    if length <= 0:
        return out
    norm = length * (length + 1) / 2.0
    for i in range(length - 1, n):
        s = 0.0
        has_nan = False
        for j in range(length):
            val = src[i - j]
            if np.isnan(val):
                has_nan = True
                break
            s += val * (length - j)
        if not has_nan:
            out[i] = s / norm
    return out

@njit(SIG_HMA_LOOP, nogil=True, cache=True)
def hma_loop(src, length):
    """Hull Moving Average - exact Pine math.round() parity."""
    n = len(src)
    out = np.full(n, np.nan, dtype=np.float64)
    if length <= 0:
        return out
    
    # Pine's math.round(x) for positive numbers maps to int(x + 0.5)
    half_len = int(length / 2.0 + 0.5)
    if half_len < 1: half_len = 1
    sqrt_len = int(np.sqrt(length) + 0.5)
    if sqrt_len < 1: sqrt_len = 1
        
    wma1 = wma_loop(src, half_len)
    wma2 = wma_loop(src, length)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        if not np.isnan(wma1[i]) and not np.isnan(wma2[i]):
            diff[i] = 2.0 * wma1[i] - wma2[i]
            
    out = wma_loop(diff, sqrt_len)
    return out

@njit(SIG_MOMENTUM_OSCILLATOR_CORE, nogil=True, cache=True)
def momentum_oscillator_core(high, low, close, length):
    """AlgoAlpha Momentum Oscillator exact port."""
    n = len(close)
    out_vf = np.full(n, np.nan, dtype=np.float64)
    if n == 0 or length <= 0:
        return out_vf
        
    min_low, _ = rolling_min_max_numba(low, length)
    _, max_high = rolling_min_max_numba(high, length)
    
    l = ema_loop_pine(min_low, float(length))
    h = ema_loop_pine(max_high, float(length))
    
    val1 = np.full(n, np.nan, dtype=np.float64)
    d_state = 0
    
    for i in range(n):
        c = close[i]
        h_val = h[i]
        l_val = l[i]
        
        # Pine parity: if any input to crossover/crossunder or val1 is NaN, 
        # the state does not update and val1 remains NaN.
        if np.isnan(c) or np.isnan(h_val) or np.isnan(l_val):
            continue
            
        if i > 0:
            c_prev = close[i-1]
            h_prev = h[i-1]
            l_prev = l[i-1]
            
            crossover = False
            crossunder = False
            
            if not np.isnan(c_prev) and not np.isnan(h_prev):
                if c > h_val and c_prev <= h_prev:
                    crossover = True
                    
            if not np.isnan(c_prev) and not np.isnan(l_prev):
                if c < l_val and c_prev >= l_prev:
                    crossunder = True
                    
            if crossover: d_state = 1
            if crossunder: d_state = -1
                
        val = l_val if d_state == 1 else h_val
        val1[i] = c - val
        
    val2 = hma_loop(val1, length)
    
    hl_diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        if not np.isnan(high[i]) and not np.isnan(low[i]):
            hl_diff[i] = high[i] - low[i]
            
    ema_hl = ema_loop_pine(hl_diff, float(length * 2))
    
    for i in range(n):
        v2 = val2[i]
        e_hl = ema_hl[i]
        if not np.isnan(v2) and not np.isnan(e_hl):
            if e_hl != 0.0:
                out_vf[i] = (v2 / e_hl * 100.0) / 8.0
            else:
                # Pine evaluates division by zero as ±Infinity. 
                # We deliberately leave it as NaN to prevent downstream Inf-clamping 
                # cascades in the bot's sanitize logic.
                pass
                
    return out_vf 

#=====================================================
# Auto Trendline [DojiEmoji] kernel - Pine-exact (full history, no filters)
#=====================================================
@njit(SIG_AUTO_TRENDLINE, nogil=True, cache=True)
def auto_trendline_loop(high, low, close, fractal_period):
    """Pine: ph = ta.pivothigh(n, n)[1] / pl = ta.pivotlow(n, n)[1] (strict > / <),
    recent_up1/2 + recent_dn1/2 fractal pointers, y = m*bar_index + recent1 - m*i_recent1,
    cross = ta.cross(close, y) evaluated bar by bar (each bar uses its OWN line).

    Returns (y_up, y_dn, cross_upper, cross_lower, up_class, dn_class):
      y_up / y_dn            line value per bar (NaN until two fractals exist on that side)
      cross_upper/_lower     +1.0 crossover, -1.0 crossunder, 0.0 none
      up_class / dn_class    1.0 HH / HL, -1.0 LH / LL, 2.0 fractal with no comparison, 0.0 none
    """
    T = len(close)
    n = fractal_period
    y_up = np.full(T, np.nan, dtype=np.float64)
    y_dn = np.full(T, np.nan, dtype=np.float64)
    cross_upper = np.zeros(T, dtype=np.float64)
    cross_lower = np.zeros(T, dtype=np.float64)
    up_class = np.zeros(T, dtype=np.float64)
    dn_class = np.zeros(T, dtype=np.float64)
    if n < 2 or T < 2 * n + 2:
        return y_up, y_dn, cross_upper, cross_lower, up_class, dn_class

    up1 = np.nan
    up2 = np.nan
    dn1 = np.nan
    dn2 = np.nan
    iu1 = -1
    iu2 = -1
    id1 = -1
    id2 = -1

    for t in range(T):
        # [1] shift: the fractal confirmed on bar t sits on bar t - n - 1
        k = t - n - 1
        if k >= n:
            hk = high[k]
            lk = low[k]
            is_hi = not np.isnan(hk)
            is_lo = not np.isnan(lk)
            for j in range(k - n, k + n + 1):
                if j == k:
                    continue
                hj = high[j]
                lj = low[j]
                if is_hi and (np.isnan(hj) or not (hk > hj)):
                    is_hi = False
                if is_lo and (np.isnan(lj) or not (lk < lj)):
                    is_lo = False
                if (not is_hi) and (not is_lo):
                    break
            if is_lo:
                dn2 = dn1
                id2 = id1
                dn1 = lk
                id1 = k
                if id2 >= 0:
                    if dn1 > dn2:
                        dn_class[t] = 1.0
                    elif dn1 < dn2:
                        dn_class[t] = -1.0
                    else:
                        dn_class[t] = 2.0
                else:
                    dn_class[t] = 2.0
            if is_hi:
                up2 = up1
                iu2 = iu1
                up1 = hk
                iu1 = k
                if iu2 >= 0:
                    if up1 > up2:
                        up_class[t] = 1.0
                    elif up1 < up2:
                        up_class[t] = -1.0
                    else:
                        up_class[t] = 2.0
                else:
                    up_class[t] = 2.0

        if iu2 >= 0 and iu1 != iu2:
            m = (up2 - up1) / (iu2 - iu1)
            y_up[t] = (m * t) + up1 - (m * iu1)
        if id2 >= 0 and id1 != id2:
            m = (dn2 - dn1) / (id2 - id1)
            y_dn[t] = (m * t) + dn1 - (m * id1)

        if t >= 1:
            c0 = close[t]
            c1 = close[t - 1]
            yu0 = y_up[t]
            yu1 = y_up[t - 1]
            if not (np.isnan(yu0) or np.isnan(yu1) or np.isnan(c0) or np.isnan(c1)):
                if c0 > yu0 and c1 <= yu1:
                    cross_upper[t] = 1.0
                elif c0 < yu0 and c1 >= yu1:
                    cross_upper[t] = -1.0
            yd0 = y_dn[t]
            yd1 = y_dn[t - 1]
            if not (np.isnan(yd0) or np.isnan(yd1) or np.isnan(c0) or np.isnan(c1)):
                if c0 > yd0 and c1 <= yd1:
                    cross_lower[t] = 1.0
                elif c0 < yd0 and c1 >= yd1:
                    cross_lower[t] = -1.0

    return y_up, y_dn, cross_upper, cross_lower, up_class, dn_class

#=====================================================
# AOT EXPORT CONFIGURATION
#=====================================================

EXPORT_CONFIG = {
    'sanitize_array_numba':  SIG_SANITIZE_ARRAY,
    'rolling_mean_numba':    SIG_ROLLING_MEAN,
    'rolling_min_max_numba': SIG_ROLLING_MIN_MAX,
    'ema_loop':              SIG_EMA_LOOP,
    'ema_loop_pine':         SIG_EMA_LOOP_PINE,          # NEW
    'ema_loop_alpha':        SIG_EMA_LOOP_ALPHA,
    'kalman_loop':           SIG_KALMAN_LOOP,
    'vwap_daily_loop_safe':  SIG_VWAP_DAILY,
    'calculate_ppo_core':    SIG_PPO_CORE,
    'calculate_rsi_core':    SIG_RSI_CORE,
    'true_range_numba':      SIG_TRUE_RANGE,
    'calculate_atr_rma':     SIG_ATR_RMA,
    'calculate_adx_core':    SIG_ADX_CORE,
    'percentile_rank_numba': SIG_PERCENTILE_RANK,
    'dynamic_flow_direction_loop': SIG_DFR_DIRECTION,
    'smoothrng_loop':        SIG_SMOOTH_RNG,
    'range_filter_loop':     SIG_RANGE_FILTER,
    'gaussian_filter_loop':  SIG_GAUSSIAN_FILTER,
    'efficiency_ratio_loop': SIG_EFFICIENCY_RATIO,
    'gaussian_trend_line_loop': SIG_GAUSSIAN_TREND_LINE,
    'wma_loop':                 SIG_WMA_LOOP,
    'hma_loop':                 SIG_HMA_LOOP,
    'momentum_oscillator_core': SIG_MOMENTUM_OSCILLATOR_CORE,
    'auto_trendline_loop':      SIG_AUTO_TRENDLINE,
}

from aot_meta import (  # noqa: E402
    SOURCE_VERSION as SOURCE_VERSION,
    AOT_FUNCTION_NAMES,
)

_exported_names = set(EXPORT_CONFIG.keys())
_registry_names = set(AOT_FUNCTION_NAMES)

if _exported_names != _registry_names:
    _missing_from_registry = sorted(_exported_names - _registry_names)
    _missing_from_export   = sorted(_registry_names - _exported_names)
    raise AssertionError(
        f"EXPORT_CONFIG (numba_functions_shared.py) and AOT_FUNCTION_NAMES "
        f"(aot_function_registry.py) have drifted apart. "
        f"In EXPORT_CONFIG but not the registry -- add to aot_function_registry.py: "
        f"{_missing_from_registry}. "
        f"In the registry but not EXPORT_CONFIG -- add to EXPORT_CONFIG: "
        f"{_missing_from_export}."
    )

__all__ = list(EXPORT_CONFIG.keys()) + ["SOURCE_VERSION"]

logger.info(f"✅ Exported {len(__all__)}/{len(AOT_FUNCTION_NAMES)} Numba-compiled functions for AOT")