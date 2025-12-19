"""
Ahead-of-Time compiled Numba functions for faster startup
Build with: python -m numba.pycc --python numba_aot.py
"""

from numba.pycc import CC
import numpy as np

cc = CC('numba_compiled')
cc.verbose = True

# ============================================================================
# Core calculation functions
# ============================================================================

@cc.export('ema_loop', 'f8[:](f8[:], f8)')
def _ema_loop_aot(data, alpha):
    """Exponential Moving Average - AOT compiled"""
    n = len(data)
    out = np.empty(n, dtype=np.float64)
    out[0] = data[0] if not np.isnan(data[0]) else 0.0
    
    for i in range(1, n):
        curr = data[i]
        if np.isnan(curr):
            out[i] = out[i-1]
        else:
            out[i] = alpha * curr + (1 - alpha) * out[i-1]
    return out

@cc.export('sma_loop', 'f8[:](f8[:], i8)')
def _sma_loop_aot(data, period):
    """Simple Moving Average - AOT compiled"""
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

        min_periods = max(2, period // 3)
        if count >= min_periods:
            out[i] = window_sum / count
        else:
            out[i] = np.nan
            
    return out

@cc.export('sanitize_array', 'f8[:](f8[:], f8)')
def _sanitize_array_aot(arr, default):
    """Sanitize array - AOT compiled"""
    out = np.empty_like(arr)
    for i in range(len(arr)):
        val = arr[i]
        if np.isnan(val) or np.isinf(val):
            out[i] = default
        else:
            out[i] = val
    return out

@cc.export('vectorized_wick_buy', 'b1[:](f8[:], f8[:], f8[:], f8[:], f8)')
def _vectorized_wick_check_buy_aot(open_arr, high_arr, low_arr, close_arr, min_wick_ratio):
    """Wick check for buy signals - AOT compiled"""
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

@cc.export('vectorized_wick_sell', 'b1[:](f8[:], f8[:], f8[:], f8[:], f8)')
def _vectorized_wick_check_sell_aot(open_arr, high_arr, low_arr, close_arr, min_wick_ratio):
    """Wick check for sell signals - AOT compiled"""
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

if __name__ == '__main__':
    cc.compile()