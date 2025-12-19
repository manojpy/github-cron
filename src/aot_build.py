# src/aot_build.py
from numba.pycc import CC
import numpy as np

cc = CC('indicators_aot')  # produces indicators_aot.*.so

# Note: Use serial kernels for AOT reliability.
# Parallel prange kernels will remain JITed at runtime.

# float64[:](float64[:], float64)
@cc.export('_sanitize_array_numba', 'float64[:](float64[:], float64)')
def _sanitize_array_numba(arr, default):
    n = arr.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        val = arr[i]
        if np.isnan(val) or np.isinf(val):
            out[i] = default
        else:
            out[i] = val
    return out

# float64[:](float64[:], float64)
@cc.export('_ema_loop', 'float64[:](float64[:], float64)')
def _ema_loop(data, alpha):
    n = data.shape[0]
    out = np.empty(n, dtype=np.float64)
    out[0] = data[0] if not np.isnan(data[0]) else 0.0
    for i in range(1, n):
        curr = data[i]
        if np.isnan(curr):
            out[i] = out[i-1]
        else:
            out[i] = alpha * curr + (1.0 - alpha) * out[i-1]
    return out

# (ppo, ppo_sig) -> two arrays via struct return is not supported; build as two exports.
# float64[:](float64[:], int64, int64)
@cc.export('_calculate_ppo_core_part1', 'float64[:](float64[:], int64, int64)')
def _calculate_ppo_core_part1(close, fast, slow):
    n = close.shape[0]
    alpha_fast = 2.0 / (fast + 1.0)
    alpha_slow = 2.0 / (slow + 1.0)
    # EMA loops inline for AOT determinism
    fast_ma = np.empty(n, dtype=np.float64)
    slow_ma = np.empty(n, dtype=np.float64)
    fast_ma[0] = close[0] if not np.isnan(close[0]) else 0.0
    slow_ma[0] = close[0] if not np.isnan(close[0]) else 0.0
    for i in range(1, n):
        c = close[i]
        prev_f = fast_ma[i-1]
        prev_s = slow_ma[i-1]
        fast_ma[i] = prev_f if np.isnan(c) else alpha_fast * c + (1.0 - alpha_fast) * prev_f
        slow_ma[i] = prev_s if np.isnan(c) else alpha_slow * c + (1.0 - alpha_slow) * prev_s
    ppo = np.empty(n, dtype=np.float64)
    for i in range(n):
        s = slow_ma[i]
        if abs(s) < 1e-12:
            ppo[i] = 0.0
        else:
            ppo[i] = ((fast_ma[i] - s) / s) * 100.0
    return ppo

# float64[:](float64[:], int64)
@cc.export('_ppo_signal', 'float64[:](float64[:], int64)')
def _ppo_signal(ppo, signal):
    n = ppo.shape[0]
    alpha_signal = 2.0 / (signal + 1.0)
    out = np.empty(n, dtype=np.float64)
    out[0] = ppo[0] if not np.isnan(ppo[0]) else 0.0
    for i in range(1, n):
        v = ppo[i]
        prev = out[i-1]
        out[i] = prev if np.isnan(v) else alpha_signal * v + (1.0 - alpha_signal) * prev
    return out

# float64[:](float64[:], int64)
@cc.export('_calculate_rsi_core', 'float64[:](float64[:], int64)')
def _calculate_rsi_core(close, rsi_len):
    n = close.shape[0]
    delta = np.zeros(n, dtype=np.float64)
    gain = np.zeros(n, dtype=np.float64)
    loss = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        d = close[i] - close[i-1]
        delta[i] = d
        if d > 0.0:
            gain[i] = d
        elif d < 0.0:
            loss[i] = -d
    alpha = 1.0 / rsi_len
    avg_gain = np.empty(n, dtype=np.float64)
    avg_loss = np.empty(n, dtype=np.float64)
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]
    for i in range(1, n):
        g = gain[i]
        l = loss[i]
        avg_gain[i] = alpha * g + (1.0 - alpha) * avg_gain[i-1]
        avg_loss[i] = alpha * l + (1.0 - alpha) * avg_loss[i-1]
    rsi = np.empty(n, dtype=np.float64)
    for i in range(n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    return rsi

# float64[:](float64[:], float64[:], float64[:], float64[:], int64[:])
@cc.export('_vwap_daily_loop', 'float64[:](float64[:], float64[:], float64[:], float64[:], int64[:])')
def _vwap_daily_loop(high, low, close, volume, timestamps):
    n = close.shape[0]
    vwap = np.empty(n, dtype=np.float64)
    cum_vol = 0.0
    cum_pv = 0.0
    prev_day = timestamps[0] // 86400
    for i in range(n):
        curr_ts = timestamps[i]
        curr_day = curr_ts // 86400
        if curr_day != prev_day:
            cum_vol = 0.0
            cum_pv = 0.0
            prev_day = curr_day
        h = high[i]; l = low[i]; c = close[i]; v = volume[i]
        if np.isnan(h) or np.isnan(l) or np.isnan(c) or np.isnan(v):
            vwap[i] = vwap[i-1] if i > 0 else c
            continue
        avg_price = (h + l + c) / 3.0
        cum_vol += v
        cum_pv += avg_price * v
        vwap[i] = c if cum_vol == 0.0 else (cum_pv / cum_vol)
    return vwap

# float64[:](float64[:], float64[:])
@cc.export('_rolling_mean_numba', 'float64[:](float64[:], int64)')
def _rolling_mean_numba(close, period):
    rows = close.shape[0]
    ma = np.empty(rows, dtype=np.float64)
    for i in range(rows):
        start = 0 if i - period + 1 < 0 else (i - period + 1)
        sum_val = 0.0
        count = 0
        for j in range(start, i + 1):
            val = close[j]
            if not np.isnan(val):
                sum_val += val
                count += 1
        ma[i] = (sum_val / count) if count > 0 else 0.0
    return ma

# (min, max) as two exports for AOT
@cc.export('_rolling_min_numba', 'float64[:](float64[:], int64)')
def _rolling_min_numba(arr, period):
    rows = arr.shape[0]
    out = np.empty(rows, dtype=np.float64)
    for i in range(rows):
        start = 0 if i - period + 1 < 0 else (i - period + 1)
        window = arr[start:i+1]
        out[i] = np.min(window)
    return out

@cc.export('_rolling_max_numba', 'float64[:](float64[:], int64)')
def _rolling_max_numba(arr, period):
    rows = arr.shape[0]
    out = np.empty(rows, dtype=np.float64)
    for i in range(rows):
        start = 0 if i - period + 1 < 0 else (i - period + 1)
        window = arr[start:i+1]
        out[i] = np.max(window)
    return out

# BUY wick check: bool[:](float64[:], float64[:], float64[:], float64[:], float64)
@cc.export('_vectorized_wick_check_buy', 'bool_[:](float64[:], float64[:], float64[:], float64[:], float64)')
def _vectorized_wick_check_buy(open_arr, high_arr, low_arr, close_arr, min_wick_ratio):
    n = close_arr.shape[0]
    result = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        o = open_arr[i]; h = high_arr[i]; l = low_arr[i]; c = close_arr[i]
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

# SELL wick check: bool[:](float64[:], float64[:], float64[:], float64[:], float64)
@cc.export('_vectorized_wick_check_sell', 'bool_[:](float64[:], float64[:], float64[:], float64[:], float64)')
def _vectorized_wick_check_sell(open_arr, high_arr, low_arr, close_arr, min_wick_ratio):
    n = close_arr.shape[0]
    result = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        o = open_arr[i]; h = high_arr[i]; l = low_arr[i]; c = close_arr[i]
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

if __name__ == '__main__':
    cc.compile()
