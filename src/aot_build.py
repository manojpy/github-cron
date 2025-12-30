#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import numpy as np
from numba.pycc import CC
from numba import njit, prange

# =========================================================================
# INTERNAL JIT CORES (Essential for AOT Linkage)
# =========================================================================

@njit(nopython=True, cache=True)
def _sma_core(data, period):
    n = len(data)
    out = np.empty(n, dtype=np.float64)
    out[:] = np.nan
    window_sum, count = 0.0, 0
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

@njit(nopython=True, cache=True)
def _ema_core(data, alpha_or_period):
    n = len(data)
    alpha = 2.0 / (alpha_or_period + 1.0) if alpha_or_period > 1.0 else alpha_or_period
    out = np.empty(n, dtype=np.float64)
    out[0] = data[0] if not np.isnan(data[0]) else 0.0
    for i in range(1, n):
        curr = data[i]
        out[i] = out[i-1] if np.isnan(curr) else alpha * curr + (1 - alpha) * out[i-1]
    return out

@njit(nopython=True, cache=True)
def _std_welford_core(close, period, responsiveness):
    n = len(close)
    sd = np.empty(n, dtype=np.float64)
    resp = max(0.00001, min(1.0, responsiveness))
    mean, m2, count = 0.0, 0.0, 0
    for i in range(n):
        val = close[i]
        if i >= period:
            old_val = close[i - period]
            if not np.isnan(old_val):
                delta_old = old_val - mean
                mean -= delta_old / (count - 1) if count > 1 else mean
                m2 -= delta_old * (old_val - mean)
                count -= 1
        if not np.isnan(val):
            count += 1
            delta = val - mean
            mean += delta / count
            m2 += delta * (val - mean)
        variance = m2 / count if count > 1 else 0.0
        sd[i] = np.sqrt(max(0.0, variance)) * resp
    return sd

@njit(nopython=True, cache=True)
def _min_max_core(arr, period):
    rows = len(arr)
    min_arr, max_arr = np.empty(rows), np.empty(rows)
    for i in range(rows):
        start = max(0, i - period + 1)
        min_v, max_v = np.inf, -np.inf
        for j in range(start, i + 1):
            v = arr[j]
            if not np.isnan(v):
                if v < min_v: min_v = v
                if v > max_v: max_v = v
        min_arr[i] = min_v if min_v != np.inf else np.nan
        max_arr[i] = max_v if max_v != -np.inf else np.nan
    return min_arr, max_arr

# =========================================================================
# AOT COMPILATION EXPORTS (23 Functions)
# =========================================================================

def compile_module():
    output_dir = Path(__file__).parent
    cc = CC('macd_aot_compiled')
    cc.output_dir = str(output_dir)
    cc.verbose = False

    # 1. UTILITIES
    @cc.export('sanitize_array_numba', 'f8[:](f8[:], f8)')
    def sanitize_array_numba(arr, default):
        out = np.empty_like(arr)
        for i in range(len(arr)):
            val = arr[i]
            out[i] = default if (np.isnan(val) or np.isinf(val)) else val
        return out

    @cc.export('sanitize_array_numba_parallel', 'f8[:](f8[:], f8)')
    def sanitize_array_numba_parallel(arr, default):
        n = len(arr)
        out = np.empty_like(arr)
        for i in prange(n):
            val = arr[i]
            out[i] = default if (np.isnan(val) or np.isinf(val)) else val
        return out

    # 2. ROLLING
    @cc.export('sma_loop', 'f8[:](f8[:], i4)')
    def sma_loop(data, period):
        return _sma_core(data, period)

    @cc.export('sma_loop_parallel', 'f8[:](f8[:], i4)')
    def sma_loop_parallel(data, period):
        return _sma_core(data, period)

    @cc.export('rolling_mean_numba', 'f8[:](f8[:], i4)')
    def rolling_mean_numba(close, period):
        return _sma_core(close, period)

    @cc.export('rolling_mean_numba_parallel', 'f8[:](f8[:], i4)')
    def rolling_mean_numba_parallel(close, period):
        return _sma_core(close, period)

    @cc.export('rolling_std_welford', 'f8[:](f8[:], i4, f8)')
    def rolling_std_welford(close, period, responsiveness):
        return _std_welford_core(close, period, responsiveness)

    @cc.export('rolling_std_welford_parallel', 'f8[:](f8[:], i4, f8)')
    def rolling_std_welford_parallel(close, period, responsiveness):
        return _std_welford_core(close, period, responsiveness)

    @cc.export('rolling_min_max_numba', 'Tuple((f8[:], f8[:]))(f8[:], i4)')
    def rolling_min_max_numba(arr, period):
        return _min_max_core(arr, period)

    @cc.export('rolling_min_max_numba_parallel', 'Tuple((f8[:], f8[:]))(f8[:], i4)')
    def rolling_min_max_numba_parallel(arr, period):
        return _min_max_core(arr, period)

    # 3. EMA & SIGNALS
    @cc.export('ema_loop', 'f8[:](f8[:], f8)')
    def ema_loop(data, alpha_or_period):
        return _ema_core(data, alpha_or_period)

    @cc.export('ema_loop_alpha', 'f8[:](f8[:], f8)')
    def ema_loop_alpha(data, alpha):
        return _ema_core(data, alpha)

    @cc.export('calculate_ppo_core', 'Tuple((f8[:], f8[:]))(f8[:], i4, i4, i4)')
    def calculate_ppo_core(close, fast, slow, signal):
        f_ma = _ema_core(close, float(fast))
        s_ma = _ema_core(close, float(slow))
        n = len(close)
        ppo = np.empty(n)
        for i in range(n):
            if np.isnan(s_ma[i]) or abs(s_ma[i]) < 1e-12:
                ppo[i] = 0.0
            else:
                ppo[i] = ((f_ma[i] - s_ma[i]) / s_ma[i]) * 100.0
        ppo_sig = _ema_core(ppo, float(signal))
        return ppo, ppo_sig

    @cc.export('calculate_rsi_core', 'f8[:](f8[:], i4)')
    def calculate_rsi_core(close, rsi_len):
        n = len(close)
        alpha = 1.0 / rsi_len
        gain, loss = np.zeros(n), np.zeros(n)
        for i in range(1, n):
            delta = close[i] - close[i-1]
            if delta > 0: gain[i] = delta
            elif delta < 0: loss[i] = -delta
        avg_g = _ema_core(gain, alpha)
        avg_l = _ema_core(loss, alpha)
        rsi = np.empty(n)
        for i in range(n):
            if avg_l[i] < 1e-10: rsi[i] = 100.0 if avg_g[i] > 1e-10 else 50.0
            else: rsi[i] = 100.0 - (100.0 / (1.0 + (avg_g[i] / avg_l[i])))
        return rsi

    # 4. FILTERS
    @cc.export('rng_filter_loop', 'f8[:](f8[:], f8[:])')
    def rng_filter_loop(x, r):
        n = len(x)
        filt = np.empty(n)
        filt[0] = x[0] if not np.isnan(x[0]) else 0.0
        for i in range(1, n):
            if np.isnan(r[i]) or np.isnan(x[i]): filt[i] = filt[i-1]
            elif x[i] > filt[i-1]: filt[i] = max(filt[i-1], x[i] - r[i])
            else: filt[i] = min(filt[i-1], x[i] + r[i])
        return filt

    @cc.export('smooth_range', 'f8[:](f8[:], i4, i4)')
    def smooth_range(close, t, m):
        n = len(close)
        diff = np.zeros(n)
        for i in range(1, n): diff[i] = abs(close[i] - close[i-1])
        avrng = _ema_core(diff, float(t))
        smoothrng = _ema_core(avrng, float(t * 2 - 1))
        return smoothrng * float(m)

    @cc.export('kalman_loop', 'f8[:](f8[:], i4, f8, f8)')
    def kalman_loop(src, length, R, Q):
        n = len(src)
        result = np.empty(n)
        estimate = src[0] if not np.isnan(src[0]) else 0.0
        err_est, err_meas = 1.0, R * max(1.0, float(length))
        q_div = Q / max(1.0, float(length))
        for i in range(n):
            if np.isnan(src[i]): result[i] = estimate
            else:
                gain = err_est / (err_est + err_meas)
                estimate = estimate + gain * (src[i] - estimate)
                err_est = (1.0 - gain) * err_est + q_div
                result[i] = estimate
        return result

    @cc.export('vwap_daily_loop', 'f8[:](f8[:], f8[:], f8[:], f8[:], i8[:])')
    def vwap_daily_loop(high, low, close, volume, timestamps):
        n = len(close)
        vwap = np.empty(n)
        c_vol, c_pv, cur_day = 0.0, 0.0, -1
        for i in range(n):
            day = timestamps[i] // 86400
            if day != cur_day: 
                cur_day = day
                c_vol = 0.0
                c_pv = 0.0
            if np.isnan(high[i]) or volume[i] <= 0:
                vwap[i] = vwap[i-1] if i > 0 else close[i]
                continue
            typical = (high[i] + low[i] + close[i]) / 3.0
            c_vol += volume[i]
            c_pv += typical * volume[i]
            vwap[i] = c_pv / c_vol if c_vol > 0 else typical
        return vwap

    # 5. MMH & WICKS
    @cc.export('calc_mmh_worm_loop', 'f8[:](f8[:], f8[:], i4)')
    def calc_mmh_worm_loop(close_arr, sd_arr, rows):
        worm = np.empty(rows)
        worm[0] = close_arr[0] if not np.isnan(close_arr[0]) else 0.0
        for i in range(1, rows):
            src = close_arr[i] if not np.isnan(close_arr[i]) else worm[i-1]
            diff = src - worm[i-1]
            sd_i = sd_arr[i]
            delta = (np.sign(diff) * sd_i) if (not np.isnan(sd_i) and abs(diff) > sd_i) else diff
            worm[i] = worm[i-1] + delta
        return worm

    @cc.export('calc_mmh_value_loop', 'f8[:](f8[:], i4)')
    def calc_mmh_value_loop(temp_arr, rows):
        val = np.zeros(rows)
        val[0] = max(-0.9999, min(0.9999, (temp_arr[0] if not np.isnan(temp_arr[0]) else 0.5) - 0.5))
        for i in range(1, rows):
            t = temp_arr[i] if not np.isnan(temp_arr[i]) else 0.5
            v = t - 0.5 + 0.5 * val[i-1]
            val[i] = max(-0.9999, min(0.9999, v))
        return val

    @cc.export('calc_mmh_momentum_loop', 'f8[:](f8[:], i4)')
    def calc_mmh_momentum_loop(momentum_arr, rows):
        for i in range(1, rows): 
            momentum_arr[i] += 0.5 * momentum_arr[i-1]
        return momentum_arr

    @cc.export('vectorized_wick_check_buy', 'b1[:](f8[:], f8[:], f8[:], f8[:], f8)')
    def vectorized_wick_check_buy(o, h, l, c, ratio):
        n = len(c)
        res = np.zeros(n, dtype=np.bool_)
        for i in range(n):
            if c[i] > o[i]:
                rng = h[i] - l[i]
                if rng > 1e-8: res[i] = ((h[i] - c[i]) / rng) < ratio
        return res

    @cc.export('vectorized_wick_check_sell', 'b1[:](f8[:], f8[:], f8[:], f8[:], f8)')
    def vectorized_wick_check_sell(o, h, l, c, ratio):
        n = len(c)
        res = np.zeros(n, dtype=np.bool_)
        for i in range(n):
            if c[i] < o[i]:
                rng = h[i] - l[i]
                if rng > 1e-8: res[i] = ((c[i] - l[i]) / rng) < ratio
        return res

    try:
        print("🚀 Compiling Performance Module...")
        cc.compile()
        return True
    except Exception as e:
        print(f"❌ ERROR: Compilation failed: {e}")
        return False

if __name__ == '__main__':
    sys.exit(0 if compile_module() else 1)