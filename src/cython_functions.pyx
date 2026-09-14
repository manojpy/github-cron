# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
"""
Cython replacement for numba_functions_shared.py

Drop-in equivalent of every @njit function previously compiled via
numba.pycc (AOT) or executed under Numba JIT.  Function names, argument
types, and return values are identical so callers (indicators.py,
aot_bridge.py) only need to change the import path.
"""

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport isnan, isinf, fabs, fmax, fmin

np.import_array()

# ══════════════════════════════════════════════════════════════════════
# 1. sanitize_array_numba
# ══════════════════════════════════════════════════════════════════════
def sanitize_array_numba(double[:] arr, double default):
    """Replace NaN and Inf with default value – O(n)."""
    cdef int n = arr.shape[0]
    cdef np.ndarray out = np.empty(n, dtype=np.float64)
    cdef double[:] ov = out
    cdef int i
    cdef double val
    for i in range(n):
        val = arr[i]
        ov[i] = default if (isnan(val) or isinf(val)) else val
    return out

# ══════════════════════════════════════════════════════════════════════
# 2. rolling_mean_numba
# ══════════════════════════════════════════════════════════════════════
def rolling_mean_numba(double[:] data, int period):
    cdef int n = data.shape[0]
    cdef np.ndarray out = np.full(n, np.nan, dtype=np.float64)
    cdef double[:] ov = out
    cdef int i
    cdef int has_nan = 0

    if period <= 0:
        return out

    for i in range(n):
        if isnan(data[i]):
            has_nan = 1
            break

    if not has_nan:
        # Fast path – no NaN in input
        window_sum = 0.0
        for i in range(n):
            window_sum += data[i]
            if i >= period:
                window_sum -= data[i - period]
            if i >= period - 1:
                ov[i] = window_sum / period
        return out

    # Slow path – NaN-tolerant sliding window
    cdef double w_sum = 0.0
    cdef int nan_count = 0
    cdef np.ndarray queue_np = np.zeros(period, dtype=np.float64)
    cdef double[:] queue = queue_np
    cdef np.ndarray is_nan_q_np = np.zeros(period, dtype=np.bool_)
    cdef unsigned char[:] is_nan_q = is_nan_q_np
    cdef int queue_idx = 0
    cdef double curr
    cdef int curr_is_nan

    for i in range(n):
        curr = data[i]
        curr_is_nan = 1 if isnan(curr) else 0

        if i >= period:
            if is_nan_q[queue_idx]:
                nan_count -= 1
            else:
                w_sum -= queue[queue_idx]

        if curr_is_nan:
            nan_count += 1
            queue[queue_idx] = 0.0
            is_nan_q[queue_idx] = 1
        else:
            w_sum += curr
            queue[queue_idx] = curr
            is_nan_q[queue_idx] = 0

        queue_idx = (queue_idx + 1) % period

        if i >= period - 1:
            ov[i] = np.nan if nan_count > 0 else (w_sum / period)

    return out

# ══════════════════════════════════════════════════════════════════════
# 3. rolling_min_max_numba
# ══════════════════════════════════════════════════════════════════════
def rolling_min_max_numba(double[:] arr, int period):
    """Match Pine's ta.lowest/ta.highest: output NaN unless full window of non-NaN values."""
    cdef int n = arr.shape[0]

    if period <= 0:
        return (np.full(n, np.nan, dtype=np.float64),
                np.full(n, np.nan, dtype=np.float64))

    cdef np.ndarray min_np = np.full(n, np.nan, dtype=np.float64)
    cdef np.ndarray max_np = np.full(n, np.nan, dtype=np.float64)
    cdef double[:] min_arr = min_np
    cdef double[:] max_arr = max_np

    cdef np.ndarray min_dq_np = np.zeros(period, dtype=np.int32)
    cdef np.ndarray max_dq_np = np.zeros(period, dtype=np.int32)
    cdef int[:] min_deque = min_dq_np
    cdef int[:] max_deque = max_dq_np

    cdef int min_h = 0, min_t = 0
    cdef int max_h = 0, max_t = 0
    cdef int valid_count = 0

    cdef np.ndarray vb_np = np.zeros(period, dtype=np.bool_)
    cdef unsigned char[:] valid_buffer = vb_np
    cdef int buf_idx = 0

    cdef int i
    cdef double val
    cdef int is_valid

    for i in range(n):
        val = arr[i]
        is_valid = 0 if isnan(val) else 1

        if i >= period:
            if valid_buffer[buf_idx]:
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

    return min_np, max_np

# ══════════════════════════════════════════════════════════════════════
# 4. ema_loop
# ══════════════════════════════════════════════════════════════════════
def ema_loop(double[:] data, double length_float):
    cdef int n = data.shape[0]
    cdef int length = <int>length_float
    cdef double alpha = 2.0 / (length + 1)
    cdef np.ndarray out_np = np.full(n, np.nan, dtype=np.float64)
    cdef double[:] out = out_np
    cdef int start_idx = -1
    cdef int i
    cdef double sum_val, curr

    for i in range(n):
        if not isnan(data[i]):
            start_idx = i
            break

    if start_idx == -1 or n < (start_idx + length):
        return out_np

    sum_val = 0.0
    for i in range(start_idx, start_idx + length):
        sum_val += data[i]

    cdef int seed_idx = start_idx + length - 1
    out[seed_idx] = sum_val / length

    for i in range(seed_idx + 1, n):
        curr = data[i]
        if isnan(curr):
            out[i] = out[i - 1]
        else:
            out[i] = alpha * curr + (1.0 - alpha) * out[i - 1]

    return out_np

# ══════════════════════════════════════════════════════════════════════
# 5. ema_loop_pine
# ══════════════════════════════════════════════════════════════════════
def ema_loop_pine(double[:] data, double length_float):
    cdef int n = data.shape[0]
    cdef int length = <int>length_float
    cdef double alpha = 2.0 / (length + 1)
    cdef np.ndarray out_np = np.full(n, np.nan, dtype=np.float64)
    cdef double[:] out = out_np
    cdef int start_idx = -1
    cdef int i
    cdef double curr

    for i in range(n):
        if not isnan(data[i]):
            start_idx = i
            break

    if start_idx == -1:
        return out_np

    out[start_idx] = data[start_idx]

    for i in range(start_idx + 1, n):
        curr = data[i]
        if isnan(curr):
            out[i] = out[i - 1]
        else:
            out[i] = alpha * curr + (1.0 - alpha) * out[i - 1]

    return out_np

# ══════════════════════════════════════════════════════════════════════
# 6. ema_loop_alpha
# ══════════════════════════════════════════════════════════════════════
def ema_loop_alpha(double[:] data, double alpha):
    cdef int n = data.shape[0]
    cdef np.ndarray out_np = np.full(n, np.nan, dtype=np.float64)
    cdef double[:] out = out_np
    cdef int first_valid_idx = -1
    cdef int i
    cdef int period, start_idx
    cdef double sma_sum, sma_init, prev, curr
    cdef int valid_count

    for i in range(n):
        if not isnan(data[i]):
            first_valid_idx = i
            break

    if first_valid_idx == -1:
        return out_np

    period = <int>(1.0 / alpha + 0.5)

    if first_valid_idx + period <= n:
        sma_sum = 0.0
        valid_count = 0
        for i in range(first_valid_idx, first_valid_idx + period):
            if not isnan(data[i]):
                sma_sum += data[i]
                valid_count += 1
        sma_init = sma_sum / valid_count if valid_count > 0 else 0.0
        for i in range(first_valid_idx, first_valid_idx + period):
            if not isnan(data[i]):
                out[i] = sma_init
        start_idx = first_valid_idx + period
        prev = sma_init
    else:
        out[first_valid_idx] = data[first_valid_idx]
        start_idx = first_valid_idx + 1
        prev = data[first_valid_idx]

    for i in range(start_idx, n):
        curr = data[i]
        if isnan(curr):
            out[i] = prev
        else:
            out[i] = alpha * curr + (1.0 - alpha) * prev
        prev = out[i]

    return out_np

# ══════════════════════════════════════════════════════════════════════
# 7. kalman_loop
# ══════════════════════════════════════════════════════════════════════
def kalman_loop(double[:] src, int length, double R, double Q):
    """Kalman filter in O(n) – applies formula on first valid bar."""
    cdef int n = src.shape[0]
    cdef np.ndarray result_np = np.full(n, np.nan, dtype=np.float64)
    cdef double[:] result = result_np
    cdef int first_valid_idx = -1
    cdef int i
    cdef double estimate, error_est, error_meas, Q_div_length
    cdef double prediction, kalman_gain, current
    cdef double length_f = <double>length

    for i in range(n):
        if not isnan(src[i]):
            first_valid_idx = i
            break

    if first_valid_idx == -1:
        return result_np

    cdef double safe_len = length_f if length_f > 1.0 else 1.0
    estimate = src[first_valid_idx]
    error_est = 1.0
    error_meas = R * safe_len
    Q_div_length = Q / safe_len

    # First bar: apply Kalman update
    prediction = estimate
    kalman_gain = error_est / (error_est + error_meas)
    estimate = prediction + kalman_gain * (src[first_valid_idx] - prediction)
    error_est = (1.0 - kalman_gain) * error_est + Q_div_length
    result[first_valid_idx] = estimate

    for i in range(first_valid_idx + 1, n):
        current = src[i]
        if isnan(current):
            error_est = error_est + Q_div_length
            result[i] = estimate
            continue
        prediction = estimate
        kalman_gain = error_est / (error_est + error_meas)
        estimate = prediction + kalman_gain * (current - estimate)
        error_est = (1.0 - kalman_gain) * error_est + Q_div_length
        result[i] = estimate

    return result_np

# ══════════════════════════════════════════════════════════════════════
# 8. vwap_daily_loop_safe
# ══════════════════════════════════════════════════════════════════════
def vwap_daily_loop_safe(double[:] hlc3, double[:] volumes, long long[:] timestamps):
    cdef int n = hlc3.shape[0]
    cdef np.ndarray vwap_np = np.empty(n, dtype=np.float64)
    cdef double[:] vwap = vwap_np
    cdef double cum_pv = 0.0, cum_vol = 0.0
    cdef long long last_day = -1
    cdef int i
    cdef long long day

    for i in range(n):
        day = timestamps[i] // 86400
        if day != last_day:
            cum_pv = 0.0
            cum_vol = 0.0
            last_day = day
        cum_pv += hlc3[i] * volumes[i]
        cum_vol += volumes[i]
        vwap[i] = cum_pv / cum_vol if cum_vol > 0.0 else hlc3[i]

    return vwap_np

# ══════════════════════════════════════════════════════════════════════
# 9. calculate_ppo_core
# ══════════════════════════════════════════════════════════════════════
def calculate_ppo_core(double[:] close, int fast, int slow, int signal):
    cdef int n = close.shape[0]
    cdef np.ndarray fast_ma = ema_loop_pine(close, <double>fast)
    cdef np.ndarray slow_ma = ema_loop_pine(close, <double>slow)
    cdef double[:] fm = fast_ma
    cdef double[:] sm = slow_ma

    cdef np.ndarray ppo_np = np.full(n, np.nan, dtype=np.float64)
    cdef double[:] ppo = ppo_np
    cdef int i
    cdef double f, s, ppo_val

    for i in range(n):
        f = fm[i]
        s = sm[i]
        if not isnan(f) and not isnan(s) and s != 0.0:
            ppo_val = ((f - s) / s) * 100.0
            if ppo_val > 1000.0:
                ppo_val = 1000.0
            elif ppo_val < -1000.0:
                ppo_val = -1000.0
            ppo[i] = ppo_val

    cdef np.ndarray ppo_sig = ema_loop_pine(ppo_np, <double>signal)
    return ppo_np, ppo_sig

# ══════════════════════════════════════════════════════════════════════
# 10. calculate_rsi_core
# ══════════════════════════════════════════════════════════════════════
def calculate_rsi_core(double[:] close, int period):
    cdef int n = close.shape[0]
    cdef np.ndarray rsi_np = np.full(n, np.nan, dtype=np.float64)
    cdef double[:] rsi = rsi_np

    if n <= period:
        return rsi_np

    cdef int first_valid_idx = -1
    cdef int i
    cdef double avg_gain = 0.0, avg_loss = 0.0, diff, rs, alpha, curr
    cdef double prev_valid
    cdef int valid_count = 0
    cdef int seed_idx, avg_period

    for i in range(n):
        if not isnan(close[i]):
            first_valid_idx = i
            break

    if first_valid_idx == -1:
        return rsi_np

    seed_idx = first_valid_idx + period
    if seed_idx >= n:
        return rsi_np

    prev_valid = close[first_valid_idx]

    # NaN-tolerant warmup
    for i in range(first_valid_idx + 1, seed_idx + 1):
        curr = close[i]
        if isnan(curr):
            continue
        diff = curr - prev_valid
        if diff > 0.0:
            avg_gain += diff
        else:
            avg_loss += -diff
        prev_valid = curr
        valid_count += 1

    if valid_count == 0:
        return rsi_np

    avg_period = period if period <= valid_count else valid_count
    if avg_period < 1:
        avg_period = 1
    avg_gain /= avg_period
    avg_loss /= avg_period

    if avg_loss == 0.0:
        rsi[seed_idx] = 100.0 if avg_gain > 0.0 else 50.0
    else:
        rs = avg_gain / avg_loss
        rsi[seed_idx] = 100.0 - (100.0 / (1.0 + rs))

    alpha = 1.0 / period

    for i in range(seed_idx + 1, n):
        curr = close[i]
        if isnan(curr):
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

    return rsi_np

# ══════════════════════════════════════════════════════════════════════
# 11. true_range_numba
# ══════════════════════════════════════════════════════════════════════
def true_range_numba(double[:] high, double[:] low, double[:] close):
    """Shared True Range calc."""
    cdef int n = close.shape[0]
    cdef np.ndarray tr_np = np.empty(n, dtype=np.float64)
    cdef double[:] tr = tr_np
    cdef int i
    cdef double h, l, c, tr1, tr2, tr3

    tr[0] = high[0] - low[0]

    for i in range(1, n):
        h = high[i]
        l = low[i]
        c = close[i - 1]
        tr1 = h - l
        tr2 = fabs(h - c)
        tr3 = fabs(l - c)
        tr[i] = fmax(tr1, fmax(tr2, tr3))

    return tr_np

# ══════════════════════════════════════════════════════════════════════
# 12. calculate_atr_rma
# ══════════════════════════════════════════════════════════════════════
def calculate_atr_rma(double[:] high, double[:] low, double[:] close, int period):
    cdef int n = close.shape[0]
    if n < period:
        return np.full(n, np.nan, dtype=np.float64)

    cdef np.ndarray tr = true_range_numba(high, low, close)
    cdef double alpha = 1.0 / (<double>period)
    cdef np.ndarray atr = ema_loop_alpha(tr, alpha)
    return atr

# ══════════════════════════════════════════════════════════════════════
# 13. calculate_adx_core
# ══════════════════════════════════════════════════════════════════════
def calculate_adx_core(double[:] high, double[:] low, double[:] close,
                       int di_length, int adx_length):
    cdef int n = high.shape[0]
    cdef np.ndarray adx_np = np.full(n, np.nan, dtype=np.float64)
    cdef double[:] adx

    if n < di_length + adx_length:
        return adx_np

    cdef np.ndarray tr_np = true_range_numba(high, low, close)
    cdef double[:] tr = tr_np

    cdef np.ndarray pdm_np = np.zeros(n, dtype=np.float64)
    cdef np.ndarray mdm_np = np.zeros(n, dtype=np.float64)
    cdef double[:] plus_dm = pdm_np
    cdef double[:] minus_dm = mdm_np
    cdef int i
    cdef double h, l, prev_h, prev_l, up, down

    for i in range(1, n):
        h = high[i]
        l = low[i]
        prev_h = high[i - 1]
        prev_l = low[i - 1]
        up = h - prev_h
        down = prev_l - l
        plus_dm[i] = up if (up > down and up > 0) else 0.0
        minus_dm[i] = down if (down > up and down > 0) else 0.0

    cdef double alpha_di = 1.0 / (<double>di_length)
    cdef np.ndarray pds_np = ema_loop_alpha(pdm_np, alpha_di)
    cdef np.ndarray mds_np = ema_loop_alpha(mdm_np, alpha_di)
    cdef np.ndarray trs_np = ema_loop_alpha(tr_np, alpha_di)
    cdef double[:] pds = pds_np
    cdef double[:] mds = mds_np
    cdef double[:] trs = trs_np

    for i in range(n):
        if trs[i] > 0.0 and not isnan(trs[i]):
            pds[i] = 100.0 * pds[i] / trs[i]
            mds[i] = 100.0 * mds[i] / trs[i]
        else:
            pds[i] = 0.0
            mds[i] = 0.0
    cdef double di_diff, di_sum
    for i in range(n):
        di_diff = fabs(pds[i] - mds[i])
        di_sum = pds[i] + mds[i]
        
        trs[i] = 0.0 if di_sum == 0.0 else 100.0 * di_diff / di_sum

    cdef double alpha_adx = 1.0 / (<double>adx_length)
    adx_np = ema_loop_alpha(trs_np, alpha_adx)
    return adx_np

# ══════════════════════════════════════════════════════════════════════
# 14. percentile_rank_numba
# ══════════════════════════════════════════════════════════════════════
def percentile_rank_numba(double[:] arr, int i, int lookback,
                          int min_history, bint allow_zero):
    """Single-pass O(lookback) percentile rank. Returns NaN where the
    Python version returned None."""
    cdef int start = i - lookback
    if start < 0:
        return np.nan

    cdef double current = arr[i]
    if isnan(current):
        return np.nan
    if not allow_zero and current <= 0.0:
        return np.nan

    cdef int count_valid = 0
    cdef int count_lt = 0
    cdef int count_eq = 0
    cdef int j
    cdef double v

    for j in range(start, i):
        v = arr[j]
        if isnan(v):
            continue
        count_valid += 1
        if v < current:
            count_lt += 1
        elif v == current:
            count_eq += 1

    if count_valid < min_history:
        return np.nan

    return (count_lt + 0.5 * count_eq) / count_valid

# ══════════════════════════════════════════════════════════════════════
# 15. dynamic_flow_direction_loop
# ══════════════════════════════════════════════════════════════════════
def dynamic_flow_direction_loop(double[:] src, double[:] basis,
                                double[:] dist, double factor):
    cdef int n = src.shape[0]
    cdef np.ndarray dir_np = np.full(n, np.nan, dtype=np.float64)
    cdef np.ndarray line_np = np.full(n, np.nan, dtype=np.float64)
    cdef double[:] direction_out = dir_np
    cdef double[:] line_out = line_np

    cdef double lower_band_prev = 0.0
    cdef double upper_band_prev = 0.0
    cdef bint trend_is_upper_prev = True

    cdef int i
    cdef double d, b, raw_upper, raw_lower, src_prev
    cdef double lower_band, upper_band
    cdef double dist_prev, direction

    for i in range(n):
        d = dist[i]
        b = basis[i]

        if isnan(d) or isnan(b):
            lower_band = lower_band_prev
            upper_band = upper_band_prev
        else:
            raw_upper = b + factor * d
            raw_lower = b - factor * d
            src_prev = src[i - 1] if i >= 1 else np.nan
            lower_band = raw_lower if (raw_lower > lower_band_prev or src_prev < lower_band_prev) else lower_band_prev
            upper_band = raw_upper if (raw_upper < upper_band_prev or src_prev > upper_band_prev) else upper_band_prev

        dist_prev = dist[i - 1] if i >= 1 else np.nan

        if isnan(dist_prev):
            direction = 1.0
        elif trend_is_upper_prev:
            direction = -1.0 if src[i] > upper_band else 1.0
        else:
            direction = 1.0 if src[i] < lower_band else -1.0

        trend_is_upper_prev = (direction == 1.0)

        if not isnan(d):
            direction_out[i] = direction
            line_out[i] = (lower_band + upper_band) / 2.0

        lower_band_prev = lower_band
        upper_band_prev = upper_band

    return dir_np, line_np

# ══════════════════════════════════════════════════════════════════════
# EXPORT REGISTRY (replaces Numba EXPORT_CONFIG)
# ══════════════════════════════════════════════════════════════════════
EXPORTED_FUNCTION_NAMES = [
    "sanitize_array_numba",
    "rolling_mean_numba",
    "rolling_min_max_numba",
    "ema_loop",
    "ema_loop_pine",
    "ema_loop_alpha",
    "kalman_loop",
    "vwap_daily_loop_safe",
    "calculate_ppo_core",
    "calculate_rsi_core",
    "true_range_numba",
    "calculate_atr_rma",
    "calculate_adx_core",
    "percentile_rank_numba",
    "dynamic_flow_direction_loop",
]

__all__ = EXPORTED_FUNCTION_NAMES + ["EXPORTED_FUNCTION_NAMES"]