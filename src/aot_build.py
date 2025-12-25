#!/usr/bin/env python3
import os, sys
from pathlib import Path
from numba.pycc import CC
import numpy as np

os.environ['NUMBA_OPT'] = '3'

def compile_module():
    output_dir = Path(__file__).parent / '__pycache__'
    output_dir.mkdir(exist_ok=True)
    cc = CC('macd_aot_compiled')
    cc.output_dir = str(output_dir)
    
    print("Compiling 23 functions...")
    
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
        return sanitize_array_numba(arr, default)
    
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
        for i in range(n):
            window_sum, count = 0.0, 0
            start = max(0, i - period + 1)
            for j in range(start, i + 1):
                if not np.isnan(data[j]):
                    window_sum += data[j]
                    count += 1
            out[i] = window_sum / count if count > 0 else np.nan
        return out
    
    # 5-6: EMA
    @cc.export('ema_loop', 'f8[:](f8[:], f8)')
    def ema_loop(data, alpha_or_period):
        n = len(data)
        alpha = 2.0/(alpha_or_period+1.0) if alpha_or_period>1.0 else alpha_or_period
        out = np.empty(n, dtype=np.float64)
        out[0] = data[0] if not np.isnan(data[0]) else 0.0
        for i in range(1, n):
            curr = data[i]
            out[i] = out[i-1] if np.isnan(curr) else alpha*curr + (1-alpha)*out[i-1]
        return out
    
    @cc.export('ema_loop_alpha', 'f8[:](f8[:], f8)')
    def ema_loop_alpha(data, alpha):
        return ema_loop(data, alpha)
    
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
            if np.isnan(h) or np.isnan(l) or np.isnan(c) or np.isnan(v) or v<=0:
                vwap[i] = vwap[i-1] if i>0 else c
                continue
            typical = (h+l+c)/3.0
            cum_vol += v
            cum_pv += typical * v
            vwap[i] = cum_pv/cum_vol if cum_vol>0 else typical
        return vwap
    
    # 9-10: Filters
    @cc.export('rng_filter_loop', 'f8[:](f8[:], f8[:])')
    def rng_filter_loop(x, r):
        n = len(x)
        filt = np.empty(n, dtype=np.float64)
        filt[0] = x[0] if not np.isnan(x[0]) else 0.0
        for i in range(1, n):
            prev_filt = filt[i-1]
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
        avrng = ema_loop(diff, float(t))
        wper = t * 2 - 1
        smoothrng = ema_loop(avrng, float(wper))
        return smoothrng * m
    
    # 11-13: MMH
    @cc.export('calc_mmh_worm_loop', 'f8[:](f8[:], f8[:], i4)')
    def calc_mmh_worm_loop(close_arr, sd_arr, rows):
        worm_arr = np.empty(rows, dtype=np.float64)
        worm_arr[0] = 0.0 if np.isnan(close_arr[0]) else close_arr[0]
        for i in range(1, rows):
            src = close_arr[i] if not np.isnan(close_arr[i]) else worm_arr[i-1]
            prev_worm = worm_arr[i-1]
            diff = src - prev_worm
            sd_i = sd_arr[i]
            delta = diff if np.isnan(sd_i) else ((np.sign(diff)*sd_i) if abs(diff)>sd_i else diff)
            worm_arr[i] = prev_worm + delta
        return worm_arr
    
    @cc.export('calc_mmh_value_loop', 'f8[:](f8[:], i4)')
    def calc_mmh_value_loop(temp_arr, rows):
        value_arr = np.zeros(rows, dtype=np.float64)
        for i in range(1, rows):
            prev_v = value_arr[i-1] if not np.isnan(value_arr[i-1]) else 0.0
            t = temp_arr[i] if not np.isnan(temp_arr[i]) else 0.5
            v = t - 0.5 + 0.5 * prev_v
            value_arr[i] = max(-0.9999, min(0.9999, v))
        return value_arr
    
    @cc.export('calc_mmh_momentum_loop', 'f8[:](f8[:], i4)')
    def calc_mmh_momentum_loop(momentum_arr, rows):
        for i in range(1, rows):
            prev = momentum_arr[i-1] if not np.isnan(momentum_arr[i-1]) else 0.0
            momentum_arr[i] = momentum_arr[i] + 0.5 * prev
        return momentum_arr
    
    # 14-15: Rolling Std
    @cc.export('rolling_std_welford', 'f8[:](f8[:], i4, f8)')
    def rolling_std_welford(close, period, responsiveness):
        n = len(close)
        sd = np.empty(n, dtype=np.float64)
        resp = max(0.0001, min(1.0, responsiveness))
        for i in range(n):
            mean, m2, count = 0.0, 0.0, 0
            start = max(0, i-period+1)
            for j in range(start, i+1):
                val = close[j]
                if not np.isnan(val):
                    count += 1
                    delta = val - mean
                    mean += delta / count
                    m2 += delta * (val - mean)
            sd[i] = 0.0 if count<=1 else np.sqrt(max(0.0, m2/count)) * resp
        return sd
    
    @cc.export('rolling_std_welford_parallel', 'f8[:](f8[:], i4, f8)')
    def rolling_std_welford_parallel(close, period, responsiveness):
        return rolling_std_welford(close, period, responsiveness)
    
    # 16-17: Rolling Mean
    @cc.export('rolling_mean_numba', 'f8[:](f8[:], i4)')
    def rolling_mean_numba(close, period):
        rows = len(close)
        ma = np.empty(rows, dtype=np.float64)
        for i in range(rows):
            start = max(0, i-period+1)
            sum_val, count = 0.0, 0
            for j in range(start, i+1):
                if not np.isnan(close[j]):
                    sum_val += close[j]
                    count += 1
            ma[i] = sum_val/count if count>0 else np.nan
        return ma
    
    @cc.export('rolling_mean_numba_parallel', 'f8[:](f8[:], i4)')
    def rolling_mean_numba_parallel(close, period):
        return rolling_mean_numba(close, period)
    
    # 18-19: Rolling Min/Max
    @cc.export('rolling_min_max_numba', 'Tuple((f8[:], f8[:]))(f8[:], i4)')
    def rolling_min_max_numba(arr, period):
        rows = len(arr)
        min_arr, max_arr = np.empty(rows, dtype=np.float64), np.empty(rows, dtype=np.float64)
        for i in range(rows):
            start = max(0, i-period+1)
            min_arr[i] = np.nanmin(arr[start:i+1])
            max_arr[i] = np.nanmax(arr[start:i+1])
        return min_arr, max_arr
    
    @cc.export('rolling_min_max_numba_parallel', 'Tuple((f8[:], f8[:]))(f8[:], i4)')
    def rolling_min_max_numba_parallel(arr, period):
        return rolling_min_max_numba(arr, period)
    
    # 20: PPO
    @cc.export('calculate_ppo_core', 'Tuple((f8[:], f8[:]))(f8[:], i4, i4, i4)')
    def calculate_ppo_core(close, fast, slow, signal):
        n = len(close)
        fast_ma = ema_loop(close, float(fast))
        slow_ma = ema_loop(close, float(slow))
        ppo = np.empty(n, dtype=np.float64)
        for i in range(n):
            ppo[i] = 0.0 if abs(slow_ma[i])<1e-12 else ((fast_ma[i]-slow_ma[i])/slow_ma[i])*100.0
        ppo_sig = ema_loop(ppo, float(signal))
        return ppo, ppo_sig
    
    # 21: RSI
    @cc.export('calculate_rsi_core', 'f8[:](f8[:], i4)')
    def calculate_rsi_core(close, rsi_len):
        n = len(close)
        gain, loss = np.zeros(n, dtype=np.float64), np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            delta = close[i] - close[i-1]
            if delta > 0: gain[i] = delta
            elif delta < 0: loss[i] = -delta
        alpha = 1.0/rsi_len
        avg_gain = ema_loop_alpha(gain, alpha)
        avg_loss = ema_loop_alpha(loss, alpha)
        rsi = np.empty(n, dtype=np.float64)
        for i in range(n):
            rsi[i] = 100.0 if avg_loss[i]<1e-10 else 100.0 - (100.0/(1.0+avg_gain[i]/avg_loss[i]))
        return rsi
    
    # 22-23: Wick Checks
    @cc.export('vectorized_wick_check_buy', 'b1[:](f8[:], f8[:], f8[:], f8[:], f8)')
    def vectorized_wick_check_buy(o, h, l, c, ratio):
        n = len(c)
        result = np.zeros(n, dtype=np.bool_)
        for i in range(n):
            if c[i]<=o[i]: continue
            rng = h[i]-l[i]
            if rng<1e-8: continue
            result[i] = (h[i]-c[i])/rng < ratio
        return result
    
    @cc.export('vectorized_wick_check_sell', 'b1[:](f8[:], f8[:], f8[:], f8[:], f8)')
    def vectorized_wick_check_sell(o, h, l, c, ratio):
        n = len(c)
        result = np.zeros(n, dtype=np.bool_)
        for i in range(n):
            if c[i]>=o[i]: continue
            rng = h[i]-l[i]
            if rng<1e-8: continue
            result[i] = (c[i]-l[i])/rng < ratio
        return result
    
    try:
        cc.compile()
        output = Path(cc.output_dir) / f"{cc.name}.so"
        if output.exists():
            print(f"✅ Compiled: {output.stat().st_size/1024:.1f} KB")
            return True
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == '__main__':
    sys.exit(0 if compile_module() else 1)