#!/usr/bin/env python3
"""
AOT Compilation Script - Builds optimized .so module for all 23 Numba functions

Synchronized with aot_bridge.py for complete compatibility
Produces platform-specific shared libraries with maximum optimization
"""

import os
import sys
import sysconfig
import multiprocessing
from pathlib import Path
import numpy as np
from numba.pycc import CC
from numba import prange, types

# Aggressive optimization flags
os.environ.update({
    'NUMBA_OPT': '3',
    'NUMBA_LOOP_VECTORIZE': '1',
    'NUMBA_CPU_NAME': 'native',  
    'NUMBA_CPU_FEATURES': '+avx2,+fma',
    'NUMBA_WARNINGS': '0',
    'NUMBA_DISABLE_JIT': '0',
    'NUMBA_THREADING_LAYER': 'omp',  # OpenMP for parallelism
})


def get_output_filename(base_name: str) -> str:
    """
    Generate platform-specific output filename matching what aot_bridge expects
    
    Args:
        base_name: Base module name (e.g., 'macd_aot_compiled')
        
    Returns:
        Full filename with platform-specific extension
    """

    ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')
    
    # Check for None, empty string, or other falsy values
    if not ext_suffix:
        # Fallback for different platforms
        import platform
        system = platform.system()
        if system == 'Windows':
            ext_suffix = '.pyd'
        elif system == 'Darwin':
            ext_suffix = '.dylib'
        else:
            # Linux fallback
            py_version = f"{sys.version_info.major}{sys.version_info.minor}"
            ext_suffix = f".cpython-{py_version}-x86_64-linux-gnu.so"
    
    return f"{base_name}{ext_suffix}"


def compile_module():
    """
    Compile all 23 functions to AOT shared library
    
    Returns:
        bool: True if compilation successful, False otherwise
    """
    output_dir = Path(__file__).parent
    module_name = 'macd_aot_compiled'
    
    cc = CC(module_name)
    cc.output_dir = str(output_dir)
    cc.verbose = True  # Enable verbose for debugging
    
    # Enable parallel AOT compilation
    n_jobs = max(1, multiprocessing.cpu_count() - 1)
    os.environ['NUMBA_NUM_THREADS'] = str(n_jobs)
    os.environ['OMP_NUM_THREADS'] = str(min(4, n_jobs))  # Limit OpenMP threads
    
    expected_output = output_dir / get_output_filename(module_name)
    
    print("=" * 70)
    print("AOT COMPILATION STARTING")
    print("=" * 70)
    print(f"📦 Module name: {module_name}")
    print(f"📂 Output directory: {output_dir.absolute()}")
    print(f"🎯 Expected output: {expected_output}")
    print(f"🔧 Compilation threads: {n_jobs}")
    print(f"🧵 OpenMP threads: {os.environ['OMP_NUM_THREADS']}")
    print(f"🐍 Python version: {sys.version_info.major}.{sys.version_info.minor}")
    print("=" * 70)
    
    # ========================================================================
    # FUNCTION EXPORTS - Must match aot_bridge.py exactly
    # ========================================================================
    
    # 1. Sanitize functions
    @cc.export('sanitize_array_numba', 'f8[:](f8[:], f8)')
    def sanitize_array_numba(arr, default):
        """Remove NaN and Inf values from array"""
        out = np.empty_like(arr)
        for i in range(len(arr)):
            val = arr[i]
            out[i] = default if (np.isnan(val) or np.isinf(val)) else val
        return out

    @cc.export('sanitize_array_numba_parallel', 'f8[:](f8[:], f8)')
    def sanitize_array_numba_parallel(arr, default):
        """Remove NaN and Inf values from array (parallel version)"""
        n = len(arr)
        out = np.empty_like(arr)
        for i in prange(n):
            val = arr[i]
            out[i] = default if (np.isnan(val) or np.isinf(val)) else val
        return out

    # 2. SMA functions
    @cc.export('sma_loop', 'f8[:](f8[:], i4)')
    def sma_loop(data, period):
        """Simple Moving Average - rolling window implementation"""
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
        """Simple Moving Average (parallel version)"""
        n = len(data)
        out = np.empty(n, dtype=np.float64)
        out[:] = np.nan
        for i in prange(n):
            window_sum, count = 0.0, 0
            start = max(0, i - period + 1)
            for j in range(start, i + 1):
                val = data[j]
                if not np.isnan(val):
                    window_sum += val
                    count += 1
            out[i] = window_sum / count if count > 0 else np.nan
        return out

    # 3. EMA functions
    @cc.export('ema_loop', 'f8[:](f8[:], f8)')
    def ema_loop(data, alpha_or_period):
        """
        Exponential Moving Average
        If alpha_or_period > 1, treats it as period and converts to alpha
        Otherwise uses it as alpha directly
        """
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
        """Exponential Moving Average with explicit alpha parameter"""
        n = len(data)
        out = np.empty(n, dtype=np.float64)
        out[0] = data[0] if not np.isnan(data[0]) else 0.0
        for i in range(1, n):
            curr = data[i]
            out[i] = out[i-1] if np.isnan(curr) else alpha * curr + (1 - alpha) * out[i-1]
        return out

    # 4. Range filter
    @cc.export('rng_filter_loop', 'f8[:](f8[:], f8[:])')
    def rng_filter_loop(x, r):
        """Range filter - constrains movement based on range values"""
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
                lower_bound = curr_x - curr_r
                filt[i] = prev_filt if lower_bound < prev_filt else lower_bound
            else:
                upper_bound = curr_x + curr_r
                filt[i] = prev_filt if upper_bound > prev_filt else upper_bound
        return filt

    # 5. Smooth range
    @cc.export('smooth_range', 'f8[:](f8[:], i4, i4)')
    def smooth_range(close, t, m):
        """Calculate smoothed range with double EMA"""
        n = len(close)
        diff = np.empty(n, dtype=np.float64)
        diff[0] = 0.0
        for i in range(1, n):
            diff[i] = abs(close[i] - close[i-1])
        
        alpha_t = 2.0 / (t + 1.0)
        avrng = np.empty(n, dtype=np.float64)
        avrng[0] = diff[0]
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
        
        return smoothrng * float(m)

    # 6. Kalman filter
    @cc.export('kalman_loop', 'f8[:](f8[:], i4, f8, f8)')
    def kalman_loop(src, length, R, Q):
        """Kalman filter implementation"""
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

    # 7. VWAP daily
    @cc.export('vwap_daily_loop', 'f8[:](f8[:], f8[:], f8[:], f8[:], i8[:])')
    def vwap_daily_loop(high, low, close, volume, timestamps):
        """Volume Weighted Average Price - resets daily"""
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

    # 8. Rolling standard deviation (Welford's algorithm)
    @cc.export('rolling_std_welford', 'f8[:](f8[:], i4, f8)')
    def rolling_std_welford(close, period, responsiveness):
        """Rolling standard deviation using Welford's online algorithm"""
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
            sd[i] = np.sqrt(m2 / count) * resp if count > 0 else 0.0
        
        return sd

    # 9. Rolling standard deviation (parallel)
    @cc.export('rolling_std_welford_parallel', 'f8[:](f8[:], i4, f8)')
    def rolling_std_welford_parallel(close, period, responsiveness):
        """Rolling standard deviation using Welford's algorithm (parallel)"""
        n = len(close)
        sd = np.empty(n, dtype=np.float64)
        resp = max(0.00001, min(1.0, responsiveness))
        
        for i in prange(n):
            mean, m2, count = 0.0, 0.0, 0
            start = max(0, i - period + 1)
            for j in range(start, i + 1):
                val = close[j]
                if not np.isnan(val):
                    count += 1
                    delta = val - mean
                    mean += delta / count
                    m2 += delta * (val - mean)
            sd[i] = np.sqrt(m2 / count) * resp if count > 0 else 0.0
        
        return sd

    # 10. MMH Worm calculation
    @cc.export('calc_mmh_worm_loop', 'f8[:](f8[:], f8[:], i4)')
    def calc_mmh_worm_loop(close_arr, sd_arr, rows):
        """Calculate MMH worm indicator"""
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
                delta = (np.sign(diff) * sd_i) if np.abs(diff) > sd_i else diff
            
            worm_arr[i] = prev_worm + delta
        
        return worm_arr

    # 11. MMH Value calculation
    @cc.export('calc_mmh_value_loop', 'f8[:](f8[:], i4)')
    def calc_mmh_value_loop(temp_arr, rows):
        """Calculate MMH value indicator"""
        value_arr = np.zeros(rows, dtype=np.float64)
        t0 = temp_arr[0] if not np.isnan(temp_arr[0]) else 0.5
        value_arr[0] = t0 - 0.5
        value_arr[0] = max(-0.9999, min(0.9999, value_arr[0]))
        
        for i in range(1, rows):
            prev_v = value_arr[i - 1] if not np.isnan(value_arr[i - 1]) else 0.0
            t = temp_arr[i] if not np.isnan(temp_arr[i]) else 0.5
            v = t - 0.5 + 0.5 * prev_v
            value_arr[i] = max(-0.9999, min(0.9999, v))
        
        return value_arr

    # 12. MMH Momentum calculation
    @cc.export('calc_mmh_momentum_loop', 'f8[:](f8[:], i4)')
    def calc_mmh_momentum_loop(momentum_arr, rows):
        """Calculate MMH momentum indicator"""
        for i in range(1, rows):
            prev = momentum_arr[i - 1] if not np.isnan(momentum_arr[i - 1]) else 0.0
            momentum_arr[i] = momentum_arr[i] + 0.5 * prev
        return momentum_arr

    # 13. Rolling mean
    @cc.export('rolling_mean_numba', 'f8[:](f8[:], i4)')
    def rolling_mean_numba(close, period):
        """Rolling mean calculation"""
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

    # 14. Rolling mean (parallel)
    @cc.export('rolling_mean_numba_parallel', 'f8[:](f8[:], i4)')
    def rolling_mean_numba_parallel(close, period):
        """Rolling mean calculation (parallel)"""
        rows = len(close)
        ma = np.empty(rows, dtype=np.float64)
        
        for i in prange(rows):
            start = max(0, i - period + 1)
            sum_val, count = 0.0, 0
            for j in range(start, i + 1):
                val = close[j]
                if not np.isnan(val):
                    sum_val += val
                    count += 1
            ma[i] = sum_val / count if count > 0 else np.nan
        
        return ma

    # 15. Rolling min/max
    @cc.export(
        'rolling_min_max_numba',
        types.Tuple((types.float64[:], types.float64[:]))(types.float64[:], types.int32)
    )
    def rolling_min_max_numba(arr, period):
        """Rolling minimum and maximum values"""
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

    # 16. Rolling min/max (parallel)
    @cc.export(
        'rolling_min_max_numba_parallel',
        types.Tuple((types.float64[:], types.float64[:]))(types.float64[:], types.int32)
    )
    def rolling_min_max_numba_parallel(arr, period):
        """Rolling minimum and maximum values (parallel)"""
        rows = len(arr)
        min_arr = np.empty(rows, dtype=np.float64)
        max_arr = np.empty(rows, dtype=np.float64)
        
        for i in prange(rows):
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

    # 17. PPO (Percentage Price Oscillator)
    @cc.export('calculate_ppo_core', 'Tuple((f8[:], f8[:]))(f8[:], i4, i4, i4)')
    def calculate_ppo_core(close, fast, slow, signal):
        """Calculate Percentage Price Oscillator and signal line"""
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

    # 18. RSI (Relative Strength Index)
    @cc.export('calculate_rsi_core', 'f8[:](f8[:], i4)')
    def calculate_rsi_core(close, rsi_len):
        """Calculate Relative Strength Index"""
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
                rs = avg_gain[i] / avg_loss[i]
                rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        
        return rsi

    # 19. Vectorized wick check (buy)
    @cc.export('vectorized_wick_check_buy', 'b1[:](f8[:], f8[:], f8[:], f8[:], f8)')
    def vectorized_wick_check_buy(open_arr, high_arr, low_arr, close_arr, min_wick_ratio):
        """Check if candles meet buy wick criteria"""
        n = len(close_arr)
        result = np.zeros(n, dtype=np.bool_)
        
        for i in range(n):
            o, h, l, c = open_arr[i], high_arr[i], low_arr[i], close_arr[i]
            
            # Must be bullish candle
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

    # 20. Vectorized wick check (sell)
    @cc.export('vectorized_wick_check_sell', 'b1[:](f8[:], f8[:], f8[:], f8[:], f8)')
    def vectorized_wick_check_sell(open_arr, high_arr, low_arr, close_arr, min_wick_ratio):
        """Check if candles meet sell wick criteria"""
        n = len(close_arr)
        result = np.zeros(n, dtype=np.bool_)
        
        for i in range(n):
            o, h, l, c = open_arr[i], high_arr[i], low_arr[i], close_arr[i]
            
            # Must be bearish candle
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

    # ========================================================================
    # COMPILATION
    # ========================================================================
    
    print(f"\n🔨 Compiling {module_name} with all 23 functions...")
    print("📋 Function list:")
    functions = [
        "1. sanitize_array_numba", "2. sanitize_array_numba_parallel",
        "3. sma_loop", "4. sma_loop_parallel",
        "5. ema_loop", "6. ema_loop_alpha",
        "7. rng_filter_loop", "8. smooth_range",
        "9. kalman_loop", "10. vwap_daily_loop",
        "11. rolling_std_welford", "12. rolling_std_welford_parallel",
        "13. calc_mmh_worm_loop", "14. calc_mmh_value_loop", 
        "15. calc_mmh_momentum_loop",
        "16. rolling_mean_numba", "17. rolling_mean_numba_parallel",
        "18. rolling_min_max_numba", "19. rolling_min_max_numba_parallel",
        "20. calculate_ppo_core", "21. calculate_rsi_core",
        "22. vectorized_wick_check_buy", "23. vectorized_wick_check_sell",
    ]
    for func in functions:
        print(f"   ✓ {func}")
    
    print("\n⏳ Starting compilation (this may take 1-3 minutes)...")
    
    try:
        cc.compile()
        
        # Find generated .so file
        so_files = list(output_dir.glob(f"{module_name}*.so")) + \
                   list(output_dir.glob(f"{module_name}*.pyd")) + \
                   list(output_dir.glob(f"{module_name}*.dylib"))
        
        if so_files:
            output = so_files[0]
            size_kb = output.stat().st_size / 1024
            
            print("\n" + "=" * 70)
            print("✅ COMPILATION SUCCESSFUL")
            print("=" * 70)
            print(f"📦 Output file: {output.name}")
            print(f"📂 Absolute path: {output.absolute()}")
            print(f"💾 File size: {size_kb:.1f} KB")
            print(f"🎯 Functions compiled: 23/23")
            print("=" * 70)
            
            # Verify file is readable
            if output.exists() and output.stat().st_size > 0:
                print("✅ File verification: PASSED")
                return True
            else:
                print("❌ File verification: FAILED (file empty or unreadable)")
                return False
        else:
            print("\n" + "=" * 70)
            print("❌ COMPILATION FAILED")
            print("=" * 70)
            print("❌ No output file generated")
            print(f"📂 Searched in: {output_dir.absolute()}")
            print(f"🔍 Expected pattern: {module_name}*.so/pyd/dylib")
            
            # List what files are present
            all_files = list(output_dir.glob("*"))
            print(f"\n📁 Files in output directory:")
            for f in all_files[:10]:  # Show first 10 files
                print(f"   - {f.name}")
            
            print("=" * 70)
            return False
            
    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ COMPILATION ERROR")
        print("=" * 70)
        print(f"❌ Exception: {type(e).__name__}")
        print(f"❌ Message: {e}")
        print("=" * 70)
        
        import traceback
        print("\n🔍 Full traceback:")
        traceback.print_exc()
        
        return False


if __name__ == '__main__':
    print("🚀 AOT Build Script")
    print(f"🐍 Python: {sys.version}")
    print(f"📦 NumPy: {np.__version__}")
    
    success = compile_module()
    
    if success:
        print("\n✅ Build completed successfully!")
        print("💡 You can now import the AOT module in your code")
        sys.exit(0)
    else:
        print("\n❌ Build failed!")
        print("💡 Falling back to JIT compilation will be used automatically")
        sys.exit(1)