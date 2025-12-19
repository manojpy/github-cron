# src/aot_build.py
from numba.pycc import CC
import numpy as np

cc = CC("indicators_aot")  # produces indicators_aot.*.so

# f8[:](f8[:], f8)
@cc.export("_sanitize_array_numba", "f8[:](f8[:], f8)")
def _sanitize_array_numba(arr, default):
    n = arr.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        val = arr[i]
        out[i] = default if (np.isnan(val) or np.isinf(val)) else val
    return out

# f8[:](f8[:], f8)
@cc.export("_ema_loop", "f8[:](f8[:], f8)")
def _ema_loop(data, alpha):
    n = data.shape[0]
    out = np.empty(n, dtype=np.float64)
    out[0] = data[0] if not np.isnan(data[0]) else 0.0
    for i in range(1, n):
        curr = data[i]
        out[i] = out[i-1] if np.isnan(curr) else (alpha * curr + (1.0 - alpha) * out[i-1])
    return out

# f8[:](f8[:], i8, i8)
@cc.export("_calculate_ppo_core_part1", "f8[:](f8[:], i8, i8)")
def _calculate_ppo_core_part1(close, fast, slow):
    n = close.shape[0]
    alpha_fast = 2.0 / (fast + 1.0)
    alpha_slow = 2.0 / (slow + 1.0)
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
        ppo[i] = 0.0 if abs(s) < 1e-12 else ((fast_ma[i] - s) / s) * 100.0
    return ppo

# f8[:](f8[:], i8)
@cc.export("_ppo_signal", "f8[:](f8[:], i8)")
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

# f8[:](f8[:], i8)
@cc.export("_calculate_rsi_core", "f8[:](f8[:], i8)")
def _calculate_rsi_core(close, rsi_len):
    n = close.shape[0]
    gain = np.zeros(n, dtype=np.float64)
    loss = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        d = close[i] - close[i-1]
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
        avg_gain[i] = alpha * gain[i] + (1.0 - alpha) * avg_gain[i-1]
        avg_loss[i] = alpha * loss[i] + (1.0 - alpha) * avg_loss[i-1]
    rsi = np.empty(n, dtype=np.float64)
    for i in range(n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    return rsi

# f8[:](f8[:], f8[:], f8[:], f8[:], i8[:])
@cc.export("_vwap_daily_loop", "f8[:](f8[:], f8[:], f8[:], f8[:], i8[:])")
def _vwap_daily_loop(high, low, close, volume, timestamps):
    n = close.shape[0]
    vwap = np.empty(n, dtype=np.float64)
    cum_vol = 0.0
    cum_pv = 0.0
    prev_day = timestamps[0] // 86400
    for i in range(n):
        curr_day = timestamps[i] // 86400
        if curr_day != prev_day:
            cum_vol = 0.0
            cum_pv = 0.0
            prev_day = curr_day
        h, l, c, v = high[i], low[i], close[i], volume[i]
        if np.isnan(h) or np.isnan(l) or np.isnan(c) or np.isnan(v):
            vwap[i] = vwap[i-1] if i > 0 else c
            continue
        avg_price = (h + l + c) / 3.0
        cum_vol += v
        cum_pv += avg_price * v
        vwap[i] = c if cum_vol == 0.0 else (cum_pv / cum_vol)
    return vwap

# f8[:](f8[:], i8)
@cc.export("_rolling_mean_numba", "f8[:](f8[:], i8)")
def _rolling_mean_numba(close, period):
    rows = close.shape[0]
    ma = np.empty(rows, dtype=np.float64)
    for i in range(rows):
        start = max(0, i - period + 1)
        vals = close[start:i+1]
        vals = vals[~np.isnan(vals)]
        ma[i] = np.mean(vals) if vals.size > 0 else 0.0
    return ma

# f8[:](f8[:], i8)
@cc.export("_rolling_min_numba", "f8[:](f8[:], i8)")
def _rolling_min_numba(arr, period):
    rows = arr.shape[0]
    out = np.empty(rows, dtype=np.float64)
    for i in range(rows):
        start = max(0, i - period + 1)
        out[i] = np.nanmin(arr[start:i+1])
    return out

# f8[:](f8[:], i8)
@cc.export("_rolling_max_numba", "f8[:](f8[:], i8)")
def _rolling_max_numba(arr, period):
    rows = arr.shape[0]
    out = np.empty(rows, dtype=np.float64)
    for i in range(rows):
        start = max(0, i - period + 1)
        out[i] = np.nanmax(arr[start:i+1])
    return out

# b1[:](f8[:], f8[:], f8[:], f8[:], f8)
@cc.export("_vectorized_wick_check_buy", "b1[:](f8[:], f8[:], f8[:], f8[:], f8)")
def _vectorized_wick_check_buy(open_arr, high_arr, low_arr, close_arr, min_wick_ratio):
    n = close_arr.shape[0]
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

# b1[:](f8[:], f8[:], f8[:], f8[:], f8)
@cc.export("_vectorized_wick_check_sell", "b1[:](f8[:], f8[:], f8[:], f8[:], f8)")
def _vectorized_wick_check_sell(open_arr, high_arr, low_arr, close_arr, min_wick_ratio):
    n = close_arr.shape[0]
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

if __name__ == "__main__":
    cc.compile()
