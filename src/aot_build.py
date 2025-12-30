#!/usr/bin/env python3

import os
import sys
from pathlib import Path
import numpy as np
from numba.pycc import CC
from numba import prange

os.environ['NUMBA_OPT'] = '3'
os.environ['NUMBA_WARNINGS'] = '0'
os.environ['PYTHONWARNINGS'] = 'ignore'

def compile_module():
    output_dir = Path(__file__).parent
    cc = CC('macd_aot_compiled')
    cc.output_dir = str(output_dir)
    cc.verbose = False

    # =========================================================================
    # 1. UTILITY FUNCTIONS
    # =========================================================================

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

    # =========================================================================
    # 2. ROLLING FUNCTIONS (O(n) OPTIMIZED)
    # =========================================================================

    @cc.export('sma_loop', 'f8[:](f8[:], i4)')
    def sma_loop(data, period):
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

    @cc.export('sma_loop_parallel', 'f8[:](f8[:], i4)')
    def sma_loop_parallel(data, period):
        # Optimized to O(n) logic inside parallel segments
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

    @cc.export('rolling_mean_numba', 'f8[:](f8[:], i4)')
    def rolling_mean_numba(close, period):
        return sma_loop(close, period)

    @cc.export('rolling_mean_numba_parallel', 'f8[:](f8[:], i4)')
    def rolling_mean_numba_parallel(close, period):
        return sma_loop_parallel(close, period)

    @cc.export('rolling_std_welford', 'f8[:](f8[:], i4, f8)')
    def rolling_std_welford(close, period, responsiveness):
        n = len(close)
        sd = np.empty(n, dtype=np.float64)
        resp = max(0.00001, min(1.0, responsiveness))
        
        mean, m2, count = 0.0, 0.0, 0
        for i in range(n):
            val = close[i]
            # Remove old value if window is full
            if i >= period:
                old_val = close[i - period]
                if not np.isnan(old_val):
                    delta_old = old_val - mean
                    mean -= delta_old / (count - 1) if count > 1 else mean
                    m2 -= delta_old * (old_val - mean)
                    count -= 1
            
            # Add new value
            if not np.isnan(val):
                count += 1
                delta = val - mean
                mean += delta / count
                m2 += delta * (val - mean)
            
            variance = m2 / count if count > 1 else 0.0
            sd[i] = np.sqrt(max(0.0, variance)) * resp
        return sd

    @cc.export('rolling_std_welford_parallel', 'f8[:](f8[:], i4, f8)')
    def rolling_std_welford_parallel(close, period, responsiveness):
        # Parallel AOT uses the same optimized sliding window logic
        return rolling_std_welford(close, period, responsiveness)

    @cc.export('rolling_min_max_numba', 'Tuple((f8[:], f8[:]))(f8[:], i4)')
    def rolling_min_max_numba(arr, period):
        rows = len(arr)
        min_arr = np.empty(rows, dtype=np.float64)
        max_arr = np.empty(rows, dtype=np.float64)
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

    @cc.export('rolling_min_max_numba_parallel', 'Tuple((f8[:], f8[:]))(f8[:], i4)')
    def rolling_min_max_numba_parallel(arr, period):
        return rolling_min_max_numba(arr, period)

    # =========================================================================
    # 3. EXPONENTIAL & SIGNAL INDICATORS
    # =========================================================================

    @cc.export('ema_loop', 'f8[:](f8[:], f8)')
    def ema_loop(data, alpha_or_period):
        n = len(data)
        alpha = 2.0 / (alpha_or_period + 1.0) if alpha_or_period > 1.0 else alpha_or_period
        out = np.empty(n, dtype=np.float64)
        out[0] = data[0] if not np.isnan(data[0]) else 0.0
        for i in range(1, n):
            curr = data[i]
            out[i] = out[i-1] if np.isnan(curr) else alpha * curr + (1 - alpha) * out[i-1]
        return out

    @cc.export('ema_loop_alpha', 'f8[:](f8[:], f8)')
    def ema_loop_alpha(data, alpha):
        n = len(data)
        out = np.empty(n, dtype=np.float64)
        out[0] = data[0] if not np.isnan(data[0]) else 0.0
        for i in range(1, n):
            curr = data[i]
            out[i] = out[i-1] if np.isnan(curr) else alpha * curr + (1 - alpha) * out[i-1]
        return out

    @cc.export('calculate_ppo_core', 'Tuple((f8[:], f8[:]))(f8[:], i4, i4, i4)')
    def calculate_ppo_core(close, fast, slow, signal):
        n = len(close)
        f_alpha = 2.0 / (fast + 1.0)
        s_alpha = 2.0 / (slow + 1.0)
        f_ma = np.empty(n, dtype=np.float64)
        s_ma = np.empty(n, dtype=np.float64)
        f_ma[0] = close[0] if not np.isnan(close[0]) else 0.0
        s_ma[0] = close[0] if not np.isnan(close[0]) else 0.0
        for i in range(1, n):
            c = close[i]
            if np.isnan(c):
                f_ma[i], s_ma[i] = f_ma[i-1], s_ma[i-1]
            else:
                f_ma[i] = f_alpha * c + (1 - f_alpha) * f_ma[i-1]
                s_ma[i] = s_alpha * c + (1 - s_alpha) * s_ma[i-1]
        ppo = np.empty(n, dtype=np.float64)
        for i in range(n):
            if np.isnan(s_ma[i]) or abs(s_ma[i]) < 1e-12:
                ppo[i] = 0.0
            else:
                ppo[i] = ((f_ma[i] - s_ma[i]) / s_ma[i]) * 100.0
        sig_alpha = 2.0 / (signal + 1.0)
        ppo_sig = np.empty(n, dtype=np.float64)
        ppo_sig[0] = ppo[0]
        for i in range(1, n):
            ppo_sig[i] = ppo_sig[i-1] if np.isnan(ppo[i]) else sig_alpha * ppo[i] + (1 - sig_alpha) * ppo_sig[i-1]
        return ppo, ppo_sig

    @cc.export('calculate_rsi_core', 'f8[:](f8[:], i4)')
    def calculate_rsi_core(close, rsi_len):
        n = len(close)
        alpha = 1.0 / rsi_len
        gain = np.zeros(n, dtype=np.float64)
        loss = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            delta = close[i] - close[i-1]
            if delta > 0: gain[i] = delta
            elif delta < 0: loss[i] = -delta
        avg_g, avg_l = np.empty(n, dtype=np.float64), np.empty(n, dtype=np.float64)
        avg_g[0], avg_l[0] = gain[0], loss[0]
        for i in range(1, n):
            avg_g[i] = alpha * gain[i] + (1 - alpha) * avg_g[i-1]
            avg_l[i] = alpha * loss[i] + (1 - alpha) * avg_l[i-1]
        rsi = np.empty(n, dtype=np.float64)
        for i in range(n):
            if avg_l[i] < 1e-10: rsi[i] = 100.0 if avg_g[i] > 1e-10 else 50.0
            else: rsi[i] = 100.0 - (100.0 / (1.0 + (avg_g[i] / avg_l[i])))
        return rsi

    # =========================================================================
    # 4. FILTERING & VOLATILITY
    # =========================================================================

    @cc.export('rng_filter_loop', 'f8[:](f8[:], f8[:])')
    def rng_filter_loop(x, r):
        n = len(x)
        filt = np.empty(n, dtype=np.float64)
        filt[0] = x[0] if not np.isnan(x[0]) else 0.0
        for i in range(1, n):
            prev, curr_x, curr_r = filt[i-1], x[i], r[i]
            if np.isnan(curr_r) or np.isnan(curr_x): filt[i] = prev
            elif curr_x > prev: filt[i] = max(prev, curr_x - curr_r)
            else: filt[i] = min(prev, curr_x + curr_r)
        return filt

    @cc.export('smooth_range', 'f8[:](f8[:], i4, i4)')
    def smooth_range(close, t, m):
        n = len(close)
        diff = np.zeros(n, dtype=np.float64)
        for i in range(1, n): diff[i] = abs(close[i] - close[i-1])
        avrng = ema_loop(diff, float(t))
        smoothrng = ema_loop(avrng, float(t * 2 - 1))
        return smoothrng * float(m)

    @cc.export('kalman_loop', 'f8[:](f8[:], i4, f8, f8)')
    def kalman_loop(src, length, R, Q):
        n = len(src)
        result = np.empty(n, dtype=np.float64)
        estimate = src[0] if not np.isnan(src[0]) else 0.0
        err_est, err_meas = 1.0, R * max(1.0, float(length))
        q_div = Q / max(1.0, float(length))
        for i in range(n):
            curr = src[i]
            if np.isnan(curr): result[i] = estimate
            else:
                gain = err_est / (err_est + err_meas)
                estimate = estimate + gain * (curr - estimate)
                err_est = (1.0 - gain) * err_est + q_div
                result[i] = estimate
        return result

    @cc.export('vwap_daily_loop', 'f8[:](f8[:], f8[:], f8[:], f8[:], i8[:])')
    def vwap_daily_loop(high, low, close, volume, timestamps):
        n = len(close)
        vwap = np.empty(n, dtype=np.float64)
        c_vol, c_pv, cur_day = 0.0, 0.0, -1
        for i in range(n):
            day = timestamps[i] // 86400
            if day != cur_day: cur_day, c_vol, c_pv = day, 0.0, 0.0
            h, l, c, v = high[i], low[i], close[i], volume[i]
            if np.isnan(h) or v <= 0:
                vwap[i] = vwap[i-1] if i > 0 else c
                continue
            typical = (h + l + c) / 3.0
            c_vol += v
            c_pv += typical * v
            vwap[i] = c_pv / c_vol if c_vol > 0 else typical
        return vwap

    # =========================================================================
    # 5. MARKET MEANY HIVE (MMH) & WICK LOGIC
    # =========================================================================

    @cc.export('calc_mmh_worm_loop', 'f8[:](f8[:], f8[:], i4)')
    def calc_mmh_worm_loop(close_arr, sd_arr, rows):
        worm = np.empty(rows, dtype=np.float64)
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
        val = np.zeros(rows, dtype=np.float64)
        t0 = temp_arr[0] if not np.isnan(temp_arr[0]) else 0.5
        val[0] = max(-0.9999, min(0.9999, t0 - 0.5))
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
            if c[i] <= o[i]: continue
            rng = h[i] - l[i]
            if rng > 1e-8: res[i] = ((h[i] - c[i]) / rng) < ratio
        return res

    @cc.export('vectorized_wick_check_sell', 'b1[:](f8[:], f8[:], f8[:], f8[:], f8)')
    def vectorized_wick_check_sell(o, h, l, c, ratio):
        n = len(c)
        res = np.zeros(n, dtype=np.bool_)
        for i in range(n):
            if c[i] >= o[i]: continue
            rng = h[i] - l[i]
            if rng > 1e-8: res[i] = ((c[i] - l[i]) / rng) < ratio
        return res

    try:
        print("Compiling AOT module...")
        cc.compile()
        so_files = list(output_dir.glob(f"{cc.name}*.so"))
        if so_files:
            print(f"SUCCESS: AOT compiled to {so_files[0].absolute()}")
            return True
        return False
    except Exception as e:
        print(f"ERROR: Compilation failed: {e}")
        return False

if __name__ == '__main__':
    sys.exit(0 if compile_module() else 1)