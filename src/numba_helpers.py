"""
Pure Numba helpers – no imports of aot_bridge or anything else.
During AOT compilation the functions are **plain**; at runtime they
are auto-decorated with @njit so the rest of the code never notices.
"""
from __future__ import annotations
import os
import numpy as np
from numba import njit, prange

# ------------------------------------------------------------------
# Are we inside the AOT build?
# ------------------------------------------------------------------
_AOT_BUILD = os.getenv("AOT_BUILD") == "1"

def _maybe_njit(*dec_args, **dec_kwargs):
    """Return identity (no-op) when AOT_BUILD=1, else @njit."""
    if _AOT_BUILD:
        return lambda f: f          # plain function
    return njit(*dec_args, **dec_kwargs)    # real decorator

# ------------------------------------------------------------------
# 1.  sanitisation
# ------------------------------------------------------------------
@_maybe_njit(nogil=True, fastmath=True, cache=True, parallel=True)
def _sanitize_array_numba_parallel(arr: np.ndarray, default: float) -> np.ndarray:
    out = np.empty_like(arr)
    for i in prange(len(arr)):
        val = arr[i]
        out[i] = default if np.isnan(val) or np.isinf(val) else val
    return out

@_maybe_njit(nogil=True, fastmath=True, cache=True)
def _sanitize_array_numba(arr: np.ndarray, default: float) -> np.ndarray:
    out = np.empty_like(arr)
    for i in range(len(arr)):
        val = arr[i]
        out[i] = default if np.isnan(val) or np.isinf(val) else val
    return out

# ------------------------------------------------------------------
# 2.  moving averages
# ------------------------------------------------------------------
@_maybe_njit(nogil=True, fastmath=True, cache=True)
def _sma_loop(data: np.ndarray, period: int) -> np.ndarray:
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
        out[i] = window_sum / count if count > 0 else np.nan
    return out

@_maybe_njit(nogil=True, fastmath=True, cache=True, parallel=True)
def _sma_loop_parallel(data: np.ndarray, period: int) -> np.ndarray:
    n = len(data)
    out = np.empty(n, dtype=np.float64)
    out[:] = np.nan
    for i in prange(n):
        window_sum = 0.0
        count = 0
        start = max(0, i - period + 1)
        for j in range(start, i + 1):
            val = data[j]
            if not np.isnan(val):
                window_sum += val
                count += 1
        out[i] = window_sum / count if count > 0 else np.nan
    return out

@_maybe_njit(nogil=True, fastmath=True, cache=True)
def _ema_loop(data: np.ndarray, alpha_or_period) -> np.ndarray:
    n = len(data)
    if alpha_or_period > 1.0:
        alpha = 2.0 / (alpha_or_period + 1.0)
    else:
        alpha = alpha_or_period
    out = np.empty(n, dtype=np.float64)
    out[0] = data[0] if not np.isnan(data[0]) else 0.0
    for i in range(1, n):
        curr = data[i]
        out[i] = out[i-1] if np.isnan(curr) else alpha * curr + (1 - alpha) * out[i-1]
    return out

@_maybe_njit(nogil=True, fastmath=True, cache=True)
def _ema_loop_alpha(data: np.ndarray, alpha: float) -> np.ndarray:
    n = len(data)
    out = np.empty(n, dtype=np.float64)
    out[0] = data[0] if not np.isnan(data[0]) else 0.0
    for i in range(1, n):
        curr = data[i]
        out[i] = out[i-1] if np.isnan(curr) else alpha * curr + (1 - alpha) * out[i-1]
    return out

# ------------------------------------------------------------------
# 3.  Kalman
# ------------------------------------------------------------------
@_maybe_njit(nogil=True, fastmath=True, cache=True)
def _kalman_loop(src: np.ndarray, length: int, R: float, Q: float) -> np.ndarray:
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

# ------------------------------------------------------------------
# 4.  VWAP
# ------------------------------------------------------------------
@_maybe_njit(nogil=True, fastmath=True, cache=True)
def _vwap_daily_loop(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    timestamps: np.ndarray
) -> np.ndarray:
    n = len(close)
    vwap = np.empty(n, dtype=np.float64)
    cum_vol = 0.0
    cum_pv = 0.0
    current_session_day = -1
    for i in range(n):
        ts = timestamps[i]
        day = ts // 86400
        h, l, c, v = high[i], low[i], close[i], volume[i]
        if day != current_session_day:
            current_session_day = day
            cum_vol = 0.0
            cum_pv = 0.0
        if np.isnan(h) or np.isnan(l) or np.isnan(c) or np.isnan(v) or v <= 0:
            vwap[i] = vwap[i-1] if i > 0 else c
            continue
        typical_price = (h + l + c) / 3.0
        cum_vol += v
        cum_pv += typical_price * v
        vwap[i] = cum_pv / cum_vol if cum_vol > 0 else typical_price
    return vwap

# ------------------------------------------------------------------
# 5.  RNG filter & smooth range
# ------------------------------------------------------------------
@_maybe_njit(nogil=True, fastmath=True, cache=True)
def _rng_filter_loop(x: np.ndarray, r: np.ndarray) -> np.ndarray:
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
            target = curr_x - curr_r
            filt[i] = max(prev_filt, target)
        else:
            target = curr_x + curr_r
            filt[i] = min(prev_filt, target)
    return filt

@_maybe_njit(nogil=True, fastmath=True, cache=True)
def _smooth_range(close: np.ndarray, t: int, m: int) -> np.ndarray:
    n = len(close)
    diff = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        diff[i] = abs(close[i] - close[i-1])
    avrng = _ema_loop(diff, t)
    wper = t * 2 - 1
    smoothrng = _ema_loop(avrng, wper)
    return smoothrng * m

# ------------------------------------------------------------------
# 6.  MMH internals
# ------------------------------------------------------------------
@_maybe_njit(nogil=True, fastmath=True, cache=True)
def _calc_mmh_worm_loop(close_arr: np.ndarray, sd_arr: np.ndarray, rows: int) -> np.ndarray:
    worm_arr = np.empty(rows, dtype=np.float64)
    first_val = close_arr[0]
    if np.isnan(first_val):
        worm_arr[0] = 0.0
    else:
        worm_arr[0] = first_val
    for i in range(1, rows):
        src = close_arr[i] if not np.isnan(close_arr[i]) else worm_arr[i - 1]
        prev_worm = worm_arr[i - 1]
        diff = src - prev_worm
        sd_i = sd_arr[i]
        if np.isnan(sd_i):
            delta = diff
        else:
            delta = (np.sign(diff) * sd_i) if (np.abs(diff) > sd_i) else diff
        worm_arr[i] = prev_worm + delta
    return worm_arr

@_maybe_njit(nogil=True, fastmath=True, cache=True)
def _calc_mmh_value_loop(temp_arr: np.ndarray, rows: int) -> np.ndarray:
    value_arr = np.zeros(rows, dtype=np.float64)
    value_arr[0] = 0.0
    for i in range(1, rows):
        prev_v = value_arr[i - 1] if not np.isnan(value_arr[i - 1]) else 0.0
        t = temp_arr[i] if not np.isnan(temp_arr[i]) else 0.5
        v = t - 0.5 + 0.5 * prev_v
        value_arr[i] = max(-0.9999, min(0.9999, v))
    return value_arr

@_maybe_njit(nogil=True, fastmath=True, cache=True)
def _calc_mmh_momentum_loop(momentum_arr: np.ndarray, rows: int) -> np.ndarray:
    for i in range(1, rows):
        prev = momentum_arr[i - 1] if not np.isnan(momentum_arr[i - 1]) else 0.0
        momentum_arr[i] = momentum_arr[i] + 0.5 * prev
    return momentum_arr

# ------------------------------------------------------------------
# 7.  rolling statistics
# ------------------------------------------------------------------
@_maybe_njit(nogil=True, fastmath=True, cache=True)
def _rolling_std_welford(close: np.ndarray, period: int, responsiveness: float) -> np.ndarray:
    n = len(close)
    sd = np.empty(n, dtype=np.float64)
    resp = max(0.0001, min(1.0, responsiveness))
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
                delta2 = val - mean
                m2 += delta * delta2
        if count > 1:
            variance = m2 / count
            sd[i] = np.sqrt(max(0.0, variance)) * resp
        else:
            sd[i] = 0.0
    return sd

@_maybe_njit(nogil=True, fastmath=True, cache=True, parallel=True)
def _rolling_std_welford_parallel(close: np.ndarray, period: int, responsiveness: float):
    n = len(close)
    sd = np.empty(n, dtype=np.float64)
    resp = max(0.0001, min(1.0, responsiveness))
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
                delta2 = val - mean
                m2 += delta * delta2
        if count > 1:
            variance = m2 / count
            sd[i] = np.sqrt(max(0.0, variance)) * resp
        else:
            sd[i] = 0.0
    return sd

@_maybe_njit(nogil=True, fastmath=True, cache=True)
def _rolling_mean_numba(close: np.ndarray, period: int) -> np.ndarray:
    rows = len(close)
    ma = np.empty(rows, dtype=np.float64)
    for i in range(rows):
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

@_maybe_njit(nogil=True, fastmath=True, cache=True, parallel=True)
def _rolling_mean_numba_parallel(close: np.ndarray, period: int) -> np.ndarray:
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

@_maybe_njit(nogil=True, fastmath=True, cache=True)
def _rolling_min_max_numba(arr: np.ndarray, period: int):
    rows = len(arr)
    min_arr = np.empty(rows, dtype=np.float64)
    max_arr = np.empty(rows, dtype=np.float64)
    for i in range(rows):
        start = max(0, i - period + 1)
        min_arr[i] = np.nanmin(arr[start:i + 1])
        max_arr[i] = np.nanmax(arr[start:i + 1])
    return min_arr, max_arr

@_maybe_njit(nogil=True, fastmath=True, cache=True, parallel=True)
def _rolling_min_max_numba_parallel(arr: np.ndarray, period: int):
    rows = len(arr)
    min_arr = np.empty(rows, dtype=np.float64)
    max_arr = np.empty(rows, dtype=np.float64)
    for i in prange(rows):
        start = max(0, i - period + 1)
        min_arr[i] = np.nanmin(arr[start:i + 1])
        max_arr[i] = np.nanmax(arr[start:i + 1])
    return min_arr, max_arr

# ------------------------------------------------------------------
# 8.  indicators
# ------------------------------------------------------------------
@_maybe_njit(nogil=True, fastmath=True, cache=True)
def _calculate_ppo_core(close: np.ndarray, fast: int, slow: int, signal: int):
    fast_ma = _ema_loop(close, fast)
    slow_ma = _ema_loop(close, slow)
    n = len(close)
    ppo = np.empty(n, dtype=np.float64)
    for i in range(n):
        if np.isnan(slow_ma[i]) or abs(slow_ma[i]) < 1e-12:
            ppo[i] = 0.0
        else:
            ppo[i] = ((fast_ma[i] - slow_ma[i]) / slow_ma[i]) * 100.0
    ppo_sig = _ema_loop(ppo, signal)
    return ppo, ppo_sig

@_maybe_njit(nogil=True, fastmath=True, cache=True)
def _calculate_rsi_core(close: np.ndarray, rsi_len: int):
    n = len(close)
    delta = np.zeros(n, dtype=np.float64)
    gain = np.zeros(n, dtype=np.float64)
    loss = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        delta[i] = close[i] - close[i-1]
        if delta[i] > 0:
            gain[i] = delta[i]
        elif delta[i] < 0:
            loss[i] = -delta[i]
    alpha = 1.0 / rsi_len
    avg_gain = np.empty(n, dtype=np.float64)
    avg_loss = np.empty(n, dtype=np.float64)
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]
    for i in range(1, n):
        avg_gain[i] = alpha * gain[i] + (1 - alpha) * avg_gain[i-1]
        avg_loss[i] = alpha * loss[i] + (1 - alpha) * avg_loss[i-1]
    rsi = np.empty(n, dtype=np.float64)
    for i in range(n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    return rsi

# ------------------------------------------------------------------
# 9.  candle quality
# ------------------------------------------------------------------
@_maybe_njit(nogil=True, fastmath=True, cache=True)
def _vectorized_wick_check_buy(
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    min_wick_ratio: float
) -> np.ndarray:
    n = len(close_arr)
    result = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        o, h, l, c = open_arr[i], high_arr[i], low_arr[i], close_arr[i]
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

@_maybe_njit(nogil=True, fastmath=True, cache=True)
def _vectorized_wick_check_sell(
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    min_wick_ratio: float
) -> np.ndarray:
    n = len(close_arr)
    result = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        o, h, l, c = open_arr[i], high_arr[i], low_arr[i], close_arr[i]
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

# ------------------------------------------------------------------
# 10.  AOT-build: re-decorate callees so Numba can type them
# ------------------------------------------------------------------
if _AOT_BUILD:
    from numba import njit          # local import to avoid runtime cost
    # list every function that is **called** by another function
    _callees = (


        _sanitize_array_numba, _sanitize_array_numba_parallel,
        _sma_loop, _sma_loop_parallel, _calculate_rsi_core, _calculate_ppo_core,
        _ema_loop, _ema_loop_alpha, _kalman_loop, _vwap_daily_loop,
        _rng_filter_loop, _smooth_range, _calc_mmh_worm_loop,
        _calc_mmh_value_loop, _calc_mmh_momentum_loop,
        _rolling_std_welford, _rolling_std_welford_parallel,
        _rolling_mean_numba, _rolling_mean_numba_parallel,
        _rolling_min_max_numba, _rolling_min_max_numba_parallel,
        _vectorized_wick_check_buy, _vectorized_wick_check_sell,
    )
    # decorate with empty signature – Numba will infer it
    for f in _callees:
        globals()[f.__name__] = njit(f)
