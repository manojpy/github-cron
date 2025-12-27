#!/usr/bin/env python3
import os, sys
from pathlib import Path
from numba.pycc import CC
import numpy as np

# Suppress warnings
os.environ['NUMBA_OPT'] = '3'
os.environ['NUMBA_WARNINGS'] = '0'
os.environ['PYTHONWARNINGS'] = 'ignore'

def compile_module():
    output_dir = Path(__file__).parent
    cc = CC('macd_aot_compiled')
    cc.output_dir = str(output_dir)
    cc.verbose = False  # Quieter compilation

    # 1-2: Sanitization
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
        for i in range(n):
            val = arr[i]
            out[i] = default if (np.isnan(val) or np.isinf(val)) else val
        return out

    # 3-4: SMA
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
        n = len(data)
        out = np.empty(n, dtype=np.float64)
        out[:] = np.nan
        for i in range(n):
            window_sum, count = 0.0, 0
            start = max(0, i - period + 1)
            for j in range(start, i + 1):
                val = data[j]
                if not np.isnan(val):
                    window_sum += val
                    count += 1
            out[i] = window_sum / count if count > 0 else np.nan
        return out

    # 5-6: EMA
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

    # 7: Kalman
    @cc.export('kalman_loop', 'f8[:](f8[:], i4, f8, f8)')
    def kalman_loop(src, length, R, Q):
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
            prediction = estimate
            kalman_gain = error_est / (error_est + error_meas)
            estimate = prediction + kalman_gain * (current - prediction)
            error_est = (1.0 - kalman_gain) * error_est + Q_div_length
            result[i] = estimate
        return result

    # 8: VWAP
    @cc.export('vwap_daily_loop', 'f8[:](f8[:], f8[:], f8[:], f8[:], i8[:])')
    def vwap_daily_loop(high, low, close, volume, timestamps):
        n = len(close)
        vwap = np.empty(n, dtype=np.float64)
        cum_vol, cum_pv, current_day = 0.0, 0.0, -1
        for i in range(n):
            day = timestamps[i] // 86400
            if day != current_day:
                current_day, cum_vol, cum_pv = day, 0.0, 0.0
            h, l, c, v = high[i], low[i], close[i], volume[i]
            if np.isnan(h) or np.isnan(l) or np.isnan(c) or np.isnan(v) or v <= 0:
                vwap[i] = vwap[i-1] if i > 0 else c
                continue
            typical = (h + l + c) / 3.0
            cum_vol += v
            cum_pv += typical * v
            vwap[i] = cum_pv / cum_vol if cum_vol > 0 else typical
        return vwap

    # 9-10: Filters
    @cc.export('rng_filter_loop', 'f8[:](f8[:], f8[:])')
    def rng_filter_loop(x, r):
        n = len(x)
        filt = np.empty(n, dtype=np.float64)
        filt[0] = x[0] if not np.isnan(x[0]) else 0.0
        for i in range(1, n):
            prev_filt = filt[i - 1]
            curr_x, curr_r = x[i], r[i]
            if np.isnan(curr_r) or np.isnan(curr_x):
                filt[i] = prev_filt
                continue
            if curr_x > prev_filt:
                filt[i] = max(prev_filt, curr_x - curr_r)
            else:
                filt[i] = min(prev_filt, curr_x + curr_r)
        return filt

    @cc.export('smooth_range', 'f8[:](f8[:], i4, i4)')
    def smooth_range(close, t, m):
        n = len(close)
        diff = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            diff[i] = abs(close[i] - close[i-1])

        alpha_t = 2.0 / (t + 1.0)
        avrng = np.empty(n, dtype=np.float64)
        avrng[0] = diff[0] if not np.isnan(diff[0]) else 0.0
        for i in range(1, n):
            curr = diff[i]
            avrng[i] = avrng[i-1] if np.isnan(curr) else alpha_t * curr + (1 - alpha_t) * avrng[i-1]

        wper = t * 2 - 1
        alpha_w = 2.0 / (wper + 1.0)
        smoothrng = np.empty(n, dtype=np.float64)
        smoothrng[0] = avrng[0]
        for i in range(1, n):
            curr = avrng[i]
            smoothrng[i] = smoothrng[i-1] if np.isnan(curr) else alpha_w * curr + (1 - alpha_w) * smoothrng[i-1]

        return smoothrng * m

    # 11-13: MMH (FIXED)
    @cc.export('calc_mmh_worm_loop', 'f8[:](f8[:], f8[:], i4)')
    def calc_mmh_worm_loop(close_arr, sd_arr, rows):
        worm_arr = np.empty(rows, dtype=np.float64)
        worm_arr[0] = 0.0 if np.isnan(close_arr[0]) else close_arr[0]
        for i in range(1, rows):
            src = close_arr[i] if not np.isnan(close_arr[i]) else worm_arr[i - 1]
            prev_worm = worm_arr[i - 1]
            diff = src - prev_worm
            sd_i = sd_arr[i]
            if np.isnan(sd_i):
                delta = diff
            else:
                delta = (np.sign(diff) * sd_i) if abs(diff) > sd_i else diff
            worm_arr[i] = prev_worm + delta
        return worm_arr

    @cc.export('calc_mmh_value_loop', 'f8[:](f8[:], i4)')
    def calc_mmh_value_loop(temp_arr, rows):
        """
        OPTIMIZED: Combines value recursion + momentum smoothing in single pass.
        
        Implements full Pine Script logic:
        1. value := (temp - 0.5 + 0.5 * nz(value[1]))
        2. temp2 = (1 + value) / (1 - value)
        3. momentum = 0.25 * log(temp2)
        4. momentum := momentum + 0.5 * nz(momentum[1])
        
        Returns final smoothed histogram values directly.
        """
        value_arr = np.zeros(rows, dtype=np.float64)
        mom_arr = np.zeros(rows, dtype=np.float64)
        
        for i in range(rows):
            # STEP 1: Fisher Value Recursion (linear, not multiplicative)
            t = temp_arr[i] if not np.isnan(temp_arr[i]) else 0.5
            prev_v = value_arr[i - 1] if i > 0 else 0.0
            
            v = t - 0.5 + 0.5 * prev_v
            v = max(-0.9999, min(0.9999, v))
            value_arr[i] = v
            
            # STEP 2: Log Transform (Fisher)
            temp2 = (1.0 + v) / (1.0 - v)
            raw_mom = 0.25 * np.log(temp2)
            
            # STEP 3: Momentum Recursion (Histogram Smoothing)
            prev_mom = mom_arr[i - 1] if i > 0 else 0.0
            mom_arr[i] = raw_mom + 0.5 * prev_mom
        
        return mom_arr


    # 14-15: Rolling Std (FIXED - Population variance, 0.00001 floor)
    @cc.export('rolling_std_welford', 'f8[:](f8[:], i4, f8)')
    def rolling_std_welford(close, period, responsiveness):
        """FIXED: Uses Population SD (divide by N) to match Pine's ta.stdev"""
        n = len(close)
        sd = np.empty(n, dtype=np.float64)
        # Match Pine's math.max(0.00001, ...) exactly
        resp = max(0.00001, min(1.0, responsiveness))
        
        for i in range(n):
            mean, m2, count = 0.0, 0.0, 0
            start = max(0, i - period + 1)
            for j in range(start, i + 1):
                val = close[j]
                if not np.isnan(val):
                    count += 1
                    delta = val - mean
                    mean += delta / count
                    # Welford's algorithm second pass
                    m2 += delta * (val - mean)
            
            if count > 0:
                # Population variance: m2 / count (NOT m2 / (count - 1))
                sd[i] = np.sqrt(m2 / count) * resp
            else:
                sd[i] = 0.0
        return sd

    @cc.export('rolling_std_welford_parallel', 'f8[:](f8[:], i4, f8)')
    def rolling_std_welford_parallel(close, period, responsiveness):
        """FIXED: Parallel version with population SD"""
        n = len(close)
        sd = np.empty(n, dtype=np.float64)
        resp = max(0.00001, min(1.0, responsiveness))
        
        for i in range(n):
            mean, m2, count = 0.0, 0.0, 0
            start = max(0, i - period + 1)
            for j in range(start, i + 1):
                val = close[j]
                if not np.isnan(val):
                    count += 1
                    delta = val - mean
                    mean += delta / count
                    m2 += delta * (val - mean)
            
            if count > 0:
                sd[i] = np.sqrt(m2 / count) * resp
            else:
                sd[i] = 0.0
        return sd

    # 16-17: Rolling Mean
    @cc.export('rolling_mean_numba', 'f8[:](f8[:], i4)')
    def rolling_mean_numba(close, period):
        rows = len(close)
        ma = np.empty(rows, dtype=np.float64)
        for i in range(rows):
            start = max(0, i - period + 1)
            sum_val, count = 0.0, 0
            for j in range(start, i + 1):
                val = close[j]
                if not np.isnan(val):
                    sum_val += val
                    count += 1
            ma[i] = sum_val / count if count > 0 else np.nan
        return ma

    @cc.export('rolling_mean_numba_parallel', 'f8[:](f8[:], i4)')
    def rolling_mean_numba_parallel(close, period):
        rows = len(close)
        ma = np.empty(rows, dtype=np.float64)
        for i in range(rows):
            start = max(0, i - period + 1)
            sum_val, count = 0.0, 0
            for j in range(start, i + 1):
                val = close[j]
                if not np.isnan(val):
                    sum_val += val
                    count += 1
            ma[i] = sum_val / count if count > 0 else np.nan
        return ma

    # 18-19: Rolling Min/Max
    @cc.export('rolling_min_max_numba', 'Tuple((f8[:], f8[:]))(f8[:], i4)')
    def rolling_min_max_numba(arr, period):
        rows = len(arr)
        min_arr = np.empty(rows, dtype=np.float64)
        max_arr = np.empty(rows, dtype=np.float64)
        for i in range(rows):
            start = max(0, i - period + 1)
            min_val = np.inf
            max_val = -np.inf
            for j in range(start, i + 1):
                val = arr[j]
                if not np.isnan(val):
                    if val < min_val:
                        min_val = val
                    if val > max_val:
                        max_val = val
            min_arr[i] = min_val if min_val != np.inf else np.nan
            max_arr[i] = max_val if max_val != -np.inf else np.nan
        return min_arr, max_arr

    @cc.export('rolling_min_max_numba_parallel', 'Tuple((f8[:], f8[:]))(f8[:], i4)')
    def rolling_min_max_numba_parallel(arr, period):
        rows = len(arr)
        min_arr = np.empty(rows, dtype=np.float64)
        max_arr = np.empty(rows, dtype=np.float64)
        for i in range(rows):
            start = max(0, i - period + 1)
            min_val = np.inf
            max_val = -np.inf
            for j in range(start, i + 1):
                val = arr[j]
                if not np.isnan(val):
                    if val < min_val:
                        min_val = val
                    if val > max_val:
                        max_val = val
            min_arr[i] = min_val if min_val != np.inf else np.nan
            max_arr[i] = max_val if max_val != -np.inf else np.nan
        return min_arr, max_arr

    # 20: PPO
    @cc.export('calculate_ppo_core', 'Tuple((f8[:], f8[:]))(f8[:], i4, i4, i4)')
    def calculate_ppo_core(close, fast, slow, signal):
        n = len(close)

        fast_alpha = 2.0 / (fast + 1.0)
        slow_alpha = 2.0 / (slow + 1.0)

        fast_ma = np.empty(n, dtype=np.float64)
        slow_ma = np.empty(n, dtype=np.float64)
        fast_ma[0] = close[0] if not np.isnan(close[0]) else 0.0
        slow_ma[0] = close[0] if not np.isnan(close[0]) else 0.0

        for i in range(1, n):
            curr = close[i]
            if np.isnan(curr):
                fast_ma[i] = fast_ma[i - 1]
                slow_ma[i] = slow_ma[i - 1]
            else:
                fast_ma[i] = fast_alpha * curr + (1 - fast_alpha) * fast_ma[i - 1]
                slow_ma[i] = slow_alpha * curr + (1 - slow_alpha) * slow_ma[i - 1]

        ppo = np.empty(n, dtype=np.float64)
        for i in range(n):
            if np.isnan(slow_ma[i]) or abs(slow_ma[i]) < 1e-12:
                ppo[i] = 0.0
            else:
                ppo[i] = ((fast_ma[i] - slow_ma[i]) / slow_ma[i]) * 100.0

        sig_alpha = 2.0 / (signal + 1.0)
        ppo_sig = np.empty(n, dtype=np.float64)
        ppo_sig[0] = ppo[0]
        for i in range(1, n):
            if np.isnan(ppo[i]):
                ppo_sig[i] = ppo_sig[i - 1]
            else:
                ppo_sig[i] = sig_alpha * ppo[i] + (1 - sig_alpha) * ppo_sig[i - 1]

        return ppo, ppo_sig

    # 21: RSI
    @cc.export('calculate_rsi_core', 'f8[:](f8[:], i4)')
    def calculate_rsi_core(close, rsi_len):
        n = len(close)
        gain = np.zeros(n, dtype=np.float64)
        loss = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            delta = close[i] - close[i - 1]
            if delta > 0:
                gain[i] = delta
            elif delta < 0:
                loss[i] = -delta

        alpha = 1.0 / rsi_len
        avg_gain = np.empty(n, dtype=np.float64)
        avg_loss = np.empty(n, dtype=np.float64)
        avg_gain[0] = gain[0]
        avg_loss[0] = loss[0]
        for i in range(1, n):
            avg_gain[i] = alpha * gain[i] + (1 - alpha) * avg_gain[i - 1]
            avg_loss[i] = alpha * loss[i] + (1 - alpha) * avg_loss[i - 1]

        rsi = np.empty(n, dtype=np.float64)
        for i in range(n):
            if avg_loss[i] < 1e-10:
                rsi[i] = 100.0
            else:
                rsi[i] = 100.0 - (100.0 / (1.0 + (avg_gain[i] / avg_loss[i])))
        return rsi

    # 22-23: Wick Checks
    @cc.export('vectorized_wick_check_buy', 'b1[:](f8[:], f8[:], f8[:], f8[:], f8)')
    def vectorized_wick_check_buy(o, h, l, c, ratio):
        n = len(c)
        result = np.zeros(n, dtype=np.bool_)
        for i in range(n):
            if c[i] <= o[i]:
                continue
            rng = h[i] - l[i]
            if rng < 1e-8:
                continue
            result[i] = ((h[i] - c[i]) / rng) < ratio
        return result

    @cc.export('vectorized_wick_check_sell', 'b1[:](f8[:], f8[:], f8[:], f8[:], f8)')
    def vectorized_wick_check_sell(o, h, l, c, ratio):
        n = len(c)
        result = np.zeros(n, dtype=np.bool_)
        for i in range(n):
            if c[i] >= o[i]:
                continue
            rng = h[i] - l[i]
            if rng < 1e-8:
                continue
            result[i] = ((c[i] - l[i]) / rng) < ratio
        return result

    try:
        print("Compiling AOT module...")
        cc.compile()
        
        so_files = list(output_dir.glob(f"{cc.name}*.so"))
        
        if so_files:
            output = so_files[0]
            size_kb = output.stat().st_size / 1024
            print(f"SUCCESS: AOT compiled {output.name} ({size_kb:.1f} KB)")
            return True
        else:
            print("ERROR: No .so file generated")
            return False
            
    except Exception as e:
        print(f"ERROR: Compilation failed: {e}")
        return False

if __name__ == '__main__':
    sys.exit(0 if compile_module() else 1)