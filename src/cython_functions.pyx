# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True
#
# WARNING: with cdivision=True, `%` on cdef ints follows C truncation semantics
# (sign of dividend), NOT Python floor semantics. Any use of `%` on a value that
# can be negative will silently differ from the Numba implementation.
# All current `%` sites in rolling_min_max_numba operate on monotonically
# increasing indices (min_h, min_t, max_h, max_t, buf_idx) and are provably
# non-negative. If you refactor those to allow negative indices, use a helper
# `mod_floor(x, m)` or drop cdivision for that function.
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
from libc.math cimport isnan, isinf, fabs, fmax, fmin, cos, sqrt, pow

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
    # Hoisted here so both the fast and slow paths use C doubles (the previous
    # revision declared them mid-function, which Cython rejects).
    cdef double window_sum
    cdef double w_sum
    cdef int nan_count
    cdef np.ndarray queue_np
    cdef double[:] queue
    cdef np.ndarray is_nan_q_np
    cdef unsigned char[:] is_nan_q
    cdef int queue_idx
    cdef double curr
    cdef int curr_is_nan

    if period <= 0:
        return out

    for i in range(n):
        if isnan(data[i]):
            has_nan = 1
            break

    if not has_nan:
        window_sum = 0.0
        for i in range(n):
            window_sum += data[i]
            if i >= period:
                window_sum -= data[i - period]
            if i >= period - 1:
                ov[i] = window_sum / period
        return out

    # Slow path – NaN-tolerant sliding window
    w_sum = 0.0
    nan_count = 0
    queue_np = np.zeros(period, dtype=np.float64)
    queue = queue_np
    is_nan_q_np = np.zeros(period, dtype=np.bool_)
    is_nan_q = is_nan_q_np
    queue_idx = 0

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
    cdef np.ndarray out_np = np.full(n, np.nan, dtype=np.float64)
    cdef double[:] out = out_np
    cdef int start_idx = -1
    cdef int i
    cdef int seed_idx
    cdef double sum_val, curr
    cdef double alpha

    if length <= 0:
        return out_np

    alpha = 2.0 / (length + 1)

    for i in range(n):
        if not isnan(data[i]):
            start_idx = i
            break

    if start_idx == -1 or n < (start_idx + length):
        return out_np

    sum_val = 0.0
    for i in range(start_idx, start_idx + length):
        sum_val += data[i]

    seed_idx = start_idx + length - 1
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
    cdef np.ndarray out_np = np.full(n, np.nan, dtype=np.float64)
    cdef double[:] out = out_np
    cdef int start_idx = -1
    cdef int i
    cdef double curr
    cdef double alpha

    # Guard first: length_float in (0, 1) truncates to 0, which would make
    # alpha = 2.0 and collapse the EMA into "current bar only". Also avoids
    # a potential division-by-zero if `length` is ever used as a divisor.
    if length <= 0:
        return out_np

    alpha = 2.0 / (length + 1)

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

# ════════════════════════════════════════════════════════════════���═
# 6. ema_loop_alpha
# ═══════════════════════════════════════════════════════════════════
def ema_loop_alpha(double[:] data, double alpha):
    cdef int n = data.shape[0]
    cdef np.ndarray out_np = np.full(n, np.nan, dtype=np.float64)
    cdef double[:] out = out_np
    cdef int first_valid_idx = -1
    cdef int i
    cdef int period, start_idx
    cdef double sma_sum, sma_init, prev, curr
    cdef int valid_count

    if alpha <= 0.0 or alpha > 1.0:
        return out_np

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
        # period >= 1 (guaranteed by alpha guard) and data[first_valid_idx]
        # is non-NaN, so valid_count >= 1 here. No zero-division possible.
        sma_init = sma_sum / valid_count
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
    cdef double safe_len = length_f if length_f > 1.0 else 1.0
    for i in range(n):
        if not isnan(src[i]):
            first_valid_idx = i
            break

    if first_valid_idx == -1:
        return result_np

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
    cdef int first_valid_idx = -1
    cdef int i
    cdef double avg_gain = 0.0, avg_loss = 0.0, diff, rs, alpha, curr
    cdef double prev_valid
    cdef int valid_count = 0
    cdef int seed_idx, avg_period

    if period <= 0 or n <= period:
        return rsi_np

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
    cdef int n = close.shape[0]
    cdef np.ndarray tr_np = np.empty(n, dtype=np.float64)
    cdef double[:] tr = tr_np
    cdef int i
    cdef double h, l, c, tr1, tr2, tr3

    if n == 0:
        return np.empty(0, dtype=np.float64)
        
    h = high[0]
    l = low[0]
    if isnan(h) or isnan(l):
        tr[0] = np.nan
    else:
        tr[0] = h - l

    for i in range(1, n):
        h = high[i]
        l = low[i]
        c = close[i - 1]
        
        # Pine parity: math.max(na, x) == na. Explicitly propagate NaN.
        if isnan(h) or isnan(l) or isnan(c):
            tr[i] = np.nan
            continue
            
        tr1 = h - l
        tr2 = fabs(h - c)
        tr3 = fabs(l - c)
        
        # Safe to use fmax now because NaNs are explicitly filtered out
        tr[i] = fmax(tr1, fmax(tr2, tr3))
        
    return tr_np

# ═══════════════════════════════════════════════════════════════
# 12. calculate_atr_rma
# ══════════════════════════════════════════════════════════════════════

def calculate_atr_rma(double[:] high, double[:] low, double[:] close, int period):
    cdef int n = close.shape[0]
    cdef np.ndarray tr
    cdef np.ndarray atr
    cdef double alpha

    # Guard MUST run before 1.0 / period.
    if period <= 0 or n < period:
        return np.full(n, np.nan, dtype=np.float64)

    tr = true_range_numba(high, low, close)
    alpha = 1.0 / (<double>period)
    atr = ema_loop_alpha(tr, alpha)
    return atr
# ═════════════════════════════════════════════════════════════════════
# 13. calculate_adx_core
# ══════════════════════════════════════════════════════════════════════

def calculate_adx_core(double[:] high, double[:] low, double[:] close,
                       int di_length, int adx_length):
    cdef int n = high.shape[0]
    cdef np.ndarray tr_np
    cdef np.ndarray pdm_np
    cdef np.ndarray mdm_np
    cdef np.ndarray pds_np
    cdef np.ndarray mds_np
    cdef np.ndarray trs_np
    cdef np.ndarray adx_np
    cdef double[:] tr
    cdef double[:] plus_dm
    cdef double[:] minus_dm
    cdef double[:] pds
    cdef double[:] mds
    cdef double[:] trs
    cdef double alpha_di, alpha_adx, di_diff, di_sum
    cdef double h, l, prev_h, prev_l, up, down
    cdef int i

    # 1. Guard first
    if di_length <= 0 or adx_length <= 0 or n < (di_length + adx_length):
        return np.full(n, np.nan, dtype=np.float64)

    # 2. True Range
    tr_np = true_range_numba(high, low, close)
    tr = tr_np

    # 3. Directional Movement (populate BEFORE smoothing)
    pdm_np = np.zeros(n, dtype=np.float64)
    mdm_np = np.zeros(n, dtype=np.float64)
    plus_dm = pdm_np
    minus_dm = mdm_np

    for i in range(1, n):
        h = high[i]
        l = low[i]
        prev_h = high[i - 1]
        prev_l = low[i - 1]
        up = h - prev_h
        down = prev_l - l
        plus_dm[i] = up if (up > down and up > 0) else 0.0
        minus_dm[i] = down if (down > up and down > 0) else 0.0

    # 4. Wilder smoothing of +DM, -DM, TR
    alpha_di = 1.0 / (<double>di_length)
    pds_np = ema_loop_alpha(pdm_np, alpha_di)
    mds_np = ema_loop_alpha(mdm_np, alpha_di)
    trs_np = ema_loop_alpha(tr_np, alpha_di)
    pds = pds_np
    mds = mds_np
    trs = trs_np

    # 5. +DI / -DI
    for i in range(n):
        if trs[i] > 0.0 and not isnan(trs[i]):
            pds[i] = 100.0 * pds[i] / trs[i]
            mds[i] = 100.0 * mds[i] / trs[i]
        else:
            pds[i] = 0.0
            mds[i] = 0.0

    # 6. DX (reuses trs_np as scratch)
    for i in range(n):
        di_diff = fabs(pds[i] - mds[i])
        di_sum = pds[i] + mds[i]
        trs[i] = 0.0 if di_sum == 0.0 else 100.0 * di_diff / di_sum

    # 7. ADX = Wilder smoothing of DX
    alpha_adx = 1.0 / (<double>adx_length)
    adx_np = ema_loop_alpha(trs_np, alpha_adx)
    return adx_np
# ══════════════════════════════════════════════════════════════════════
# 14. percentile_rank_numba
# ══════════════════════════════════════════════════════════════════════
def percentile_rank_numba(double[:] arr, int i, int lookback,
                          int min_history, bint allow_zero):
    cdef int n = arr.shape[0]
    cdef int start
    cdef double current
    cdef int count_valid = 0
    cdef int count_lt = 0
    cdef int count_eq = 0
    cdef int j
    cdef double v

    # Boundary check to prevent segfaults with boundscheck=False
    if i < 0 or i >= n:
        return np.nan

    start = i - lookback
    if start < 0:
        return np.nan

    current = arr[i]
    if isnan(current):
        return np.nan
    if not allow_zero and current <= 0.0:
        return np.nan

    for j in range(start, i):
        v = arr[j]
        if isnan(v):
            continue
        count_valid += 1
        if v < current:
            count_lt += 1
        elif v == current:
            count_eq += 1

    if count_valid == 0 or count_valid < min_history:
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
# 16. smoothrng_loop — Pine ta.ema(ta.ema(|close - close[1]|, t), 2t-1) * m
#     Pine ta.ema seeds on the FIRST value -> ema_loop_pine (never ema_loop)
# ══════════════════════════════════════════════════════════════════════
def smoothrng_loop(double[:] x, int t, double m):
    cdef int n = x.shape[0]
    cdef int i
    cdef int wper
    cdef np.ndarray diff_np
    cdef double[:] diff_arr
    cdef np.ndarray avrng
    cdef np.ndarray smoothrng
    cdef np.ndarray out_np
    cdef double[:] out
    cdef double[:] sm

    if n == 0:
        return np.empty(0, dtype=np.float64)

    diff_np = np.empty(n, dtype=np.float64)
    diff_arr = diff_np
    diff_arr[0] = np.nan
    for i in range(1, n):
        if isnan(x[i]) or isnan(x[i - 1]):
            diff_arr[i] = np.nan
        else:
            diff_arr[i] = fabs(x[i] - x[i - 1])

    wper = t * 2 - 1
    avrng = ema_loop_pine(diff_np, <double>t)
    smoothrng = ema_loop_pine(avrng, <double>wper)
    sm = smoothrng

    out_np = np.empty(n, dtype=np.float64)
    out = out_np
    for i in range(n):
        if isnan(sm[i]):
            out[i] = np.nan
        else:
            out[i] = sm[i] * m
    return out_np

# ══════════════════════════════════════════════════════════════════════
# 17. range_filter_loop — Pine rngfiltx1x1 (na in -> na out, next nz == 0)
# ══════════════════════════════════════════════════════════════════════
def range_filter_loop(double[:] x, double[:] r):
    cdef int n = x.shape[0]
    cdef np.ndarray out_np = np.full(n, np.nan, dtype=np.float64)
    cdef double[:] out = out_np
    cdef int i
    cdef double curr_x, curr_r, prev_filt

    if n == 0:
        return out_np

    prev_filt = 0.0

    for i in range(n):
        curr_x = x[i]
        curr_r = r[i]

        if isnan(curr_x) or isnan(curr_r):
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
    return out_np


# ══════════════════════════════════════════════════════════════════════
# 18. gaussian_filter_loop — Pine: gaussianFilter(src, length, poleCount)
# ══════════════════════════════════════════════════════════════════════
def gaussian_filter_loop(double[:] src, int length, int poles):
    """4-pole cascade Gaussian filter."""
    cdef int n = src.shape[0]
    cdef np.ndarray out_np = np.full(n, np.nan, dtype=np.float64)
    cdef double[:] out = out_np
    cdef int i
    cdef double PI = 3.14159265358979323846
    cdef double beta, alpha, curr_src
    cdef double s1, s2, s3, s4
    cdef double prev_s1, prev_s2, prev_s3, prev_s4

    if n == 0 or length < 1 or poles < 1:
        return out_np

    beta  = (1.0 - cos(2.0 * PI / <double>length)) / \
            (pow(1.414, 2.0 / <double>poles) - 1.0)
    alpha = -beta + sqrt(beta * beta + 2.0 * beta)

    prev_s1 = np.nan
    prev_s2 = np.nan
    prev_s3 = np.nan
    prev_s4 = np.nan

    for i in range(n):
        curr_src = src[i]
        if isnan(curr_src):
            out[i] = np.nan
            prev_s1 = prev_s2 = prev_s3 = prev_s4 = np.nan
            continue

        s1 = alpha * curr_src + (1.0 - alpha) * (curr_src if isnan(prev_s1) else prev_s1)
        s2 = alpha * s1       + (1.0 - alpha) * (curr_src if isnan(prev_s2) else prev_s2)
        s3 = alpha * s2       + (1.0 - alpha) * (curr_src if isnan(prev_s3) else prev_s3)
        s4 = alpha * s3       + (1.0 - alpha) * (curr_src if isnan(prev_s4) else prev_s4)

        if poles <= 1:
            out[i] = s1
        elif poles == 2:
            out[i] = s2
        elif poles == 3:
            out[i] = s3
        else:
            out[i] = s4

        prev_s1, prev_s2, prev_s3, prev_s4 = s1, s2, s3, s4

    return out_np

# ══════════════════════════════════════════════════════════════════════
# 19. efficiency_ratio_loop — Pine ER for the Gaussian trend
# ══════════════════════════════════════════════════════════════════════
def efficiency_ratio_loop(double[:] src, int length):
    """Pine ER for the Gaussian trend."""
    cdef int n = src.shape[0]
    cdef np.ndarray er_np = np.full(n, np.nan, dtype=np.float64)
    cdef double[:] er = er_np
    cdef np.ndarray diff_np = np.full(n, np.nan, dtype=np.float64)
    cdef double[:] diffs = diff_np
    cdef int i
    cdef double path_sum, net_move
    cdef int nan_count = 0

    if n == 0 or length < 1:
        return er_np

    for i in range(1, n):
        if not isnan(src[i]) and not isnan(src[i - 1]):
            diffs[i] = fabs(src[i] - src[i - 1])

    path_sum = 0.0
    for i in range(1, n):
        if isnan(diffs[i]):
            nan_count += 1
        else:
            path_sum += diffs[i]

        if i - length >= 1:
            if isnan(diffs[i - length]):
                nan_count -= 1
            else:
                path_sum -= diffs[i - length]

        if i >= length and nan_count == 0:
            if not isnan(src[i]) and not isnan(src[i - length]):
                net_move = fabs(src[i] - src[i - length])
                er[i] = 0.0 if path_sum == 0.0 else net_move / path_sum

    return er_np

# ══════════════════════════════════════════════════════════════════════
# 20. gaussian_trend_line_loop — Pine stepped deadband line + direction
# ══════════════════════════════════════════════════════════════════════
def gaussian_trend_line_loop(double[:] filtered, double[:] upper, double[:] lower):
    """Pine stepped deadband line + direction."""
    cdef int n = filtered.shape[0]
    cdef np.ndarray tl_np = np.full(n, np.nan, dtype=np.float64)
    cdef np.ndarray dir_np = np.zeros(n, dtype=np.int32)
    cdef double[:] tl = tl_np
    cdef int[:] dir_arr = dir_np
    cdef int i
    cdef double prev_tl, curr_tl

    if n == 0:
        return tl_np, dir_np

    curr_tl = filtered[0]
    if not isnan(upper[0]) and not isnan(curr_tl) and upper[0] < curr_tl:
        curr_tl = upper[0]
    if not isnan(lower[0]) and not isnan(curr_tl) and lower[0] > curr_tl:
        curr_tl = lower[0]
    tl[0] = curr_tl
    dir_arr[0] = 0

    for i in range(1, n):
        prev_tl = tl[i - 1]
        curr_tl = filtered[i] if isnan(prev_tl) else prev_tl

        if not isnan(upper[i]) and not isnan(curr_tl) and upper[i] < curr_tl:
            curr_tl = upper[i]
        if not isnan(lower[i]) and not isnan(curr_tl) and lower[i] > curr_tl:
            curr_tl = lower[i]

        tl[i] = curr_tl

        if not isnan(curr_tl) and not isnan(prev_tl):
            if curr_tl > prev_tl:
                dir_arr[i] = 1
            elif curr_tl < prev_tl:
                dir_arr[i] = -1
            else:
                dir_arr[i] = dir_arr[i - 1]
        else:
            dir_arr[i] = dir_arr[i - 1]

    return tl_np, dir_np

# ══════════════════════════════════════════════════════════════════════
# 21-23. wma_loop / hma_loop / momentum_oscillator_core — Pine: AlgoAlpha Momentum Oscillator
# ══════════════════════════════════════════════════════════════════════
def wma_loop(double[:] src, int length):
    cdef int n = src.shape[0]
    cdef np.ndarray out_np = np.full(n, np.nan, dtype=np.float64)
    cdef double[:] out = out_np
    cdef int i, j
    cdef double s, norm, val
    cdef int has_nan
    if length <= 0:
        return out_np
    norm = length * (length + 1) / 2.0
    for i in range(length - 1, n):
        s = 0.0
        has_nan = 0
        for j in range(length):
            val = src[i - j]
            if isnan(val):
                has_nan = 1
                break
            s += val * (length - j)
        if not has_nan:
            out[i] = s / norm
    return out_np

def hma_loop(double[:] src, int length):
    cdef int n = src.shape[0]
    cdef np.ndarray out_np = np.full(n, np.nan, dtype=np.float64)
    cdef int half_len, sqrt_len
    cdef np.ndarray wma1_np, wma2_np, diff_np
    cdef double[:] wma1, wma2, diff
    cdef int i
    if length <= 0:
        return out_np
        
    half_len = <int>(length / 2.0 + 0.5)
    if half_len < 1: half_len = 1
    sqrt_len = <int>(sqrt(<double>length) + 0.5)
    if sqrt_len < 1: sqrt_len = 1
        
    wma1_np = wma_loop(src, half_len)
    wma2_np = wma_loop(src, length)
    wma1 = wma1_np
    wma2 = wma2_np
    
    diff_np = np.full(n, np.nan, dtype=np.float64)
    diff = diff_np
    for i in range(n):
        if not isnan(wma1[i]) and not isnan(wma2[i]):
            diff[i] = 2.0 * wma1[i] - wma2[i]
            
    out_np = wma_loop(diff_np, sqrt_len)
    return out_np


def momentum_oscillator_core(double[:] high, double[:] low, double[:] close, int length):
    cdef int n = close.shape[0]
    cdef np.ndarray out_vf_np = np.full(n, np.nan, dtype=np.float64)
    cdef double[:] out_vf = out_vf_np
    cdef tuple mm_low, mm_high
    cdef np.ndarray min_low_np, max_high_np
    cdef double[:] min_low, max_high
    cdef np.ndarray l_np, h_np, val1_np, hl_diff_np, val2_np, ema_hl_np
    cdef double[:] l, h, val1, hl_diff, val2, ema_hl
    cdef int d_state = 0
    cdef int i
    cdef double c, h_val, l_val, val, c_prev, h_prev, l_prev, v2, e_hl
    cdef bint crossover, crossunder
    
    if n == 0 or length <= 0:
        return out_vf_np
        
    mm_low = rolling_min_max_numba(low, length)
    mm_high = rolling_min_max_numba(high, length)
    min_low_np = mm_low[0]
    max_high_np = mm_high[1]
    
    min_low = min_low_np
    max_high = max_high_np
    
    l_np = ema_loop_pine(min_low_np, <double>length)
    h_np = ema_loop_pine(max_high_np, <double>length)
    l = l_np
    h = h_np
    
    val1_np = np.full(n, np.nan, dtype=np.float64)
    val1 = val1_np
    
    for i in range(n):
        c = close[i]
        h_val = h[i]
        l_val = l[i]
        
        # Pine parity: if any input to crossover/crossunder or val1 is NaN, 
        # the state does not update and val1 remains NaN.
        if isnan(c) or isnan(h_val) or isnan(l_val):
            continue
            
        if i > 0:
            c_prev = close[i-1]
            h_prev = h[i-1]
            l_prev = l[i-1]
            
            crossover = False
            crossunder = False
            
            if not isnan(c_prev) and not isnan(h_prev):
                if c > h_val and c_prev <= h_prev:
                    crossover = True
                    
            if not isnan(c_prev) and not isnan(l_prev):
                if c < l_val and c_prev >= l_prev:
                    crossunder = True
                    
            if crossover: d_state = 1
            if crossunder: d_state = -1
                
        val = l_val if d_state == 1 else h_val
        val1[i] = c - val
        
    val2_np = hma_loop(val1_np, length)
    val2 = val2_np
    
    hl_diff_np = np.full(n, np.nan, dtype=np.float64)
    hl_diff = hl_diff_np
    for i in range(n):
        if not isnan(high[i]) and not isnan(low[i]):
            hl_diff[i] = high[i] - low[i]
            
    ema_hl_np = ema_loop_pine(hl_diff_np, <double>(length * 2))
    ema_hl = ema_hl_np
    
    for i in range(n):
        v2 = val2[i]
        e_hl = ema_hl[i]
        if not isnan(v2) and not isnan(e_hl):
            if e_hl != 0.0:
                out_vf[i] = (v2 / e_hl * 100.0) / 8.0
            else:
                # Pine evaluates division by zero as ±Infinity. 
                # We deliberately leave it as NaN to prevent downstream Inf-clamping 
                # cascades in the bot's sanitize logic.
                pass
                
    return out_vf_np

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
    "smoothrng_loop",
    "range_filter_loop",
    "gaussian_filter_loop",
    "efficiency_ratio_loop",
    "gaussian_trend_line_loop",
    "wma_loop",
    "hma_loop",
    "momentum_oscillator_core",
]

from aot_meta import (  # noqa: E402
    SOURCE_VERSION as SOURCE_VERSION,
    AOT_FUNCTION_NAMES,
)

# Callers that iterated EXPORT_CONFIG.items() on the Numba side keep working.
# Cython has no signature object, so values are None — only the keys matter.
EXPORT_CONFIG = {name: None for name in EXPORTED_FUNCTION_NAMES}

_exported_names = set(EXPORTED_FUNCTION_NAMES)
_registry_names = set(AOT_FUNCTION_NAMES)

if _exported_names != _registry_names:
    _missing_from_registry = sorted(_exported_names - _registry_names)
    _missing_from_export   = sorted(_registry_names - _exported_names)
    raise AssertionError(
        f"EXPORTED_FUNCTION_NAMES (cython_functions) and AOT_FUNCTION_NAMES "
        f"(aot_meta.py) have drifted apart. "
        f"In cython_functions but not the registry -- add to aot_meta.py: "
        f"{_missing_from_registry}. "
        f"In the registry but not cython_functions -- add to "
        f"EXPORTED_FUNCTION_NAMES: {_missing_from_export}."
    )

__all__ = (
    list(EXPORT_CONFIG.keys())
    + ["EXPORT_CONFIG", "EXPORTED_FUNCTION_NAMES", "SOURCE_VERSION"]
)
