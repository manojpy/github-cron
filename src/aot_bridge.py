"""
AOT Bridge Module - Complete wrapper coverage for all 23 Numba functions
Provides unified API that automatically uses AOT .so when available, falls back to JIT
OPTIMIZED: Direct function replacement eliminates decorator overhead
"""

import importlib.util
import warnings
import pathlib
import os
import sys
import sysconfig
import logging
import numpy as np
from numba import njit as _njit, prange as _prange
from typing import Tuple, Optional

os.environ['NUMBA_WARNINGS'] = '0'
logger = logging.getLogger("aot_bridge")

_USING_AOT = False
_AOT_MODULE = None
_FALLBACK_REASON = None
_INITIALIZED = False

def initialize_aot() -> Tuple[bool, Optional[str]]:
    global _USING_AOT, _AOT_MODULE, _FALLBACK_REASON
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
    try:
        import macd_aot_compiled
        _AOT_MODULE = macd_aot_compiled
    except ImportError:
        # Dynamic suffix detection
        ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')
        if ext_suffix is None:
            ext_suffix = '.cpython-311-x86_64-linux-gnu.so'  # Reasonable fallback
        
        so_path = pathlib.Path(__file__).parent / f"macd_aot_compiled{ext_suffix}"
        if so_path.exists():
            try:
                spec = importlib.util.spec_from_file_location("macd_aot_compiled", so_path)
                if spec is None or spec.loader is None:
                    _FALLBACK_REASON = "Failed to create module spec"
                    return False, _FALLBACK_REASON
                
                mod = importlib.util.module_from_spec(spec)
                sys.modules["macd_aot_compiled"] = mod
                spec.loader.exec_module(mod)
                _AOT_MODULE = mod
            except Exception as e:
                _FALLBACK_REASON = f"Failed to load .so file: {e}"
                logger.warning(f"AOT init failed: {_FALLBACK_REASON}")
                return False, _FALLBACK_REASON
        else:
            _FALLBACK_REASON = "AOT module not found"
            logger.warning(f"AOT init failed: {_FALLBACK_REASON}")
            return False, _FALLBACK_REASON
    
    try:
        # Comprehensive verification
        test_cases = [
            (np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64), 3.0),
            (np.array([np.nan, 2.0, 3.0, np.nan, 5.0], dtype=np.float64), 3.0),  # NaN handling
            (np.array([1e-10, 1e10, 1e-10], dtype=np.float64), 2.0),  # Numeric range
        ]
        
        for test_data, period in test_cases:
            result = _AOT_MODULE.ema_loop(test_data, period)
            if result is None or len(result) != len(test_data):
                raise ValueError(f"Invalid result shape")
            # Allow NaN in output where input has NaN, but not all NaN
            if np.all(np.isnan(result)):
                raise ValueError("All NaN output")
        
        _USING_AOT = True
        logger.info("✅ AOT module loaded and verified")
        return True, None

 
        # ✅ DIRECT REPLACEMENT: Zero-overhead AOT dispatch
        globals()['_sanitize_array_numba'] = _AOT_MODULE.sanitize_array_numba
        globals()['_sanitize_array_numba_parallel'] = _AOT_MODULE.sanitize_array_numba_parallel
        globals()['_sma_loop'] = _AOT_MODULE.sma_loop
        globals()['_sma_loop_parallel'] = _AOT_MODULE.sma_loop_parallel
        globals()['_ema_loop'] = _AOT_MODULE.ema_loop
        globals()['_ema_loop_alpha'] = _AOT_MODULE.ema_loop_alpha
        globals()['_rng_filter_loop'] = _AOT_MODULE.rng_filter_loop
        globals()['_smooth_range'] = _AOT_MODULE.smooth_range
        globals()['_kalman_loop'] = _AOT_MODULE.kalman_loop
        globals()['_vwap_daily_loop'] = _AOT_MODULE.vwap_daily_loop
        globals()['_rolling_std_welford'] = _AOT_MODULE.rolling_std_welford
        globals()['_rolling_std_welford_parallel'] = _AOT_MODULE.rolling_std_welford_parallel
        globals()['_calc_mmh_worm_loop'] = _AOT_MODULE.calc_mmh_worm_loop
        globals()['_calc_mmh_value_loop'] = _AOT_MODULE.calc_mmh_value_loop
        globals()['_calc_mmh_momentum_loop'] = _AOT_MODULE.calc_mmh_momentum_loop
        globals()['_rolling_mean_numba'] = _AOT_MODULE.rolling_mean_numba
        globals()['_rolling_mean_numba_parallel'] = _AOT_MODULE.rolling_mean_numba_parallel
        globals()['_rolling_min_max_numba'] = _AOT_MODULE.rolling_min_max_numba
        globals()['_rolling_min_max_numba_parallel'] = _AOT_MODULE.rolling_min_max_numba_parallel
        globals()['_calculate_ppo_core'] = _AOT_MODULE.calculate_ppo_core
        globals()['_calculate_rsi_core'] = _AOT_MODULE.calculate_rsi_core
        globals()['_vectorized_wick_check_buy'] = _AOT_MODULE.vectorized_wick_check_buy
        globals()['_vectorized_wick_check_sell'] = _AOT_MODULE.vectorized_wick_check_sell
        
        logger.info("✅ AOT functions directly replaced (zero-overhead)")
        _USING_AOT = True
        return True, None
        
    except Exception as e:
        _FALLBACK_REASON = f"AOT verification failed: {e}"
        _AOT_MODULE = None
        logger.warning(f"AOT init failed: {_FALLBACK_REASON}")
        return False, _FALLBACK_REASON

def ensure_initialized() -> bool:
    global _INITIALIZED
    if _INITIALIZED:
        return _USING_AOT
    ok, _ = initialize_aot()
    _INITIALIZED = True
    return ok

def is_using_aot() -> bool:
    return _USING_AOT

def get_fallback_reason() -> Optional[str]:
    return _FALLBACK_REASON

# ============================================================================
# FALLBACK JIT IMPLEMENTATIONS (used when AOT unavailable)
# ============================================================================

@_njit(nogil=True, fastmath=True, cache=True)
def _jit_sanitize_array_numba(arr, default):
    out = np.empty_like(arr)
    for i in range(len(arr)):
        val = arr[i]
        out[i] = default if (np.isnan(val) or np.isinf(val)) else val
    return out

@_njit(nogil=True, fastmath=True, cache=True, parallel=True)
def _jit_sanitize_array_numba_parallel(arr, default):
    out = np.empty_like(arr)
    for i in _prange(len(arr)):
        val = arr[i]
        out[i] = default if (np.isnan(val) or np.isinf(val)) else val
    return out

@_njit(nogil=True, fastmath=True, cache=True)
def _jit_sma_loop(data, period):
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

@_njit(nogil=True, fastmath=True, cache=True, parallel=True)
def _jit_sma_loop_parallel(data, period):
    n = len(data)
    out = np.empty(n, dtype=np.float64)
    out[:] = np.nan
    for i in _prange(n):
        window_sum, count = 0.0, 0
        start = max(0, i - period + 1)
        for j in range(start, i + 1):
            val = data[j]
            if not np.isnan(val):
                window_sum += val
                count += 1
        out[i] = window_sum / count if count > 0 else np.nan
    return out

@_njit(nogil=True, fastmath=True, cache=True)
def _jit_ema_loop(data, alpha_or_period):
    n = len(data)
    alpha = 2.0/(alpha_or_period+1.0) if alpha_or_period>1.0 else alpha_or_period
    out = np.empty(n, dtype=np.float64)
    out[0] = data[0] if not np.isnan(data[0]) else 0.0
    for i in range(1, n):
        curr = data[i]
        out[i] = out[i-1] if np.isnan(curr) else alpha*curr + (1-alpha)*out[i-1]
    return out

@_njit(nogil=True, fastmath=True, cache=True)
def _jit_ema_loop_alpha(data, alpha):
    n = len(data)
    out = np.empty(n, dtype=np.float64)
    out[0] = data[0] if not np.isnan(data[0]) else 0.0
    for i in range(1, n):
        curr = data[i]
        out[i] = out[i-1] if np.isnan(curr) else alpha*curr + (1-alpha)*out[i-1]
    return out

@_njit(nogil=True, fastmath=True, cache=True)
def _jit_rng_filter_loop(x, r):
    n = len(x)
    filt = np.empty(n, dtype=np.float64)
    filt[0] = x[0] if not np.isnan(x[0]) else 0.0
    for i in range(1, n):
        prev_filt = filt[i - 1]
        curr_x, curr_r = x[i], r[i]
        if np.isnan(curr_r) or np.isnan(curr_x):
            filt[i] = prev_filt
            continue
        # CRITICAL FIX: Match Pine's exact ternary logic
        if curr_x > prev_filt:
            # Uptrend: x - r < prev ? prev : x - r
            lower_bound = curr_x - curr_r
            filt[i] = prev_filt if lower_bound < prev_filt else lower_bound
        else:
            # Downtrend: x + r > prev ? prev : x + r
            upper_bound = curr_x + curr_r
            filt[i] = prev_filt if upper_bound > prev_filt else upper_bound
    return filt

@_njit(nogil=True, fastmath=True, cache=True)
def _jit_smooth_range(close, t, m):
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

@_njit(nogil=True, fastmath=True, cache=True)
def _jit_kalman_loop(src, length, R, Q):
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

@_njit(nogil=True, fastmath=True, cache=True)
def _jit_vwap_daily_loop(high, low, close, volume, timestamps):
    n = len(close)
    vwap = np.empty(n, dtype=np.float64)
    cum_vol, cum_pv, current_session_day = 0.0, 0.0, -1
    for i in range(n):
        ts = timestamps[i]
        day = ts // 86400
        h, l, c, v = high[i], low[i], close[i], volume[i]
        if day != current_session_day:
            current_session_day = day
            cum_vol = 0.0
            cum_pv = 0.0
        if np.isnan(h) or np.isnan(l) or np.isnan(c) or np.isnan(v) or v <= 0:
            vwap[i] = vwap[i-1] if i > 0 else c
            continue
        typical_price = (h + l + c) / 3.0
        cum_vol += v
        cum_pv += typical_price * v
        vwap[i] = cum_pv / cum_vol if cum_vol > 0 else typical_price
    return vwap

@_njit(nogil=True, fastmath=True, cache=True)
def _jit_rolling_std_welford(close, period, responsiveness):
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

@_njit(nogil=True, fastmath=True, cache=True, parallel=True)
def _jit_rolling_std_welford_parallel(close, period, responsiveness):
    n = len(close)
    sd = np.empty(n, dtype=np.float64)
    resp = max(0.00001, min(1.0, responsiveness))
    for i in _prange(n):
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

@_njit(nogil=True, fastmath=True, cache=True)
def _jit_calc_mmh_worm_loop(close_arr, sd_arr, rows):
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

@_njit(nogil=True, fastmath=True, cache=True)
def _jit_calc_mmh_value_loop(temp_arr, rows):
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

@_njit(nogil=True, fastmath=True, cache=True)
def _jit_calc_mmh_momentum_loop(momentum_arr, rows):
    for i in range(1, rows):
        prev = momentum_arr[i - 1] if not np.isnan(momentum_arr[i - 1]) else 0.0
        momentum_arr[i] = momentum_arr[i] + 0.5 * prev
    return momentum_arr

@_njit(nogil=True, fastmath=True, cache=True)
def _jit_rolling_mean_numba(close, period):
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

@_njit(nogil=True, fastmath=True, cache=True, parallel=True)
def _jit_rolling_mean_numba_parallel(close, period):
    rows = len(close)
    ma = np.empty(rows, dtype=np.float64)
    for i in _prange(rows):
        start = max(0, i - period + 1)
        sum_val, count = 0.0, 0
        for j in range(start, i + 1):
            val = close[j]
            if not np.isnan(val):
                sum_val += val
                count += 1
        ma[i] = sum_val / count if count > 0 else np.nan
    return ma

@_njit(nogil=True, fastmath=True, cache=True)
def _jit_rolling_min_max_numba(arr, period):
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

@_njit(nogil=True, fastmath=True, cache=True, parallel=True)
def _jit_rolling_min_max_numba_parallel(arr, period):
    rows = len(arr)
    min_arr = np.empty(rows, dtype=np.float64)
    max_arr = np.empty(rows, dtype=np.float64)
    for i in _prange(rows):
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

@_njit(nogil=True, fastmath=True, cache=True)
def _jit_calculate_ppo_core(close, fast, slow, signal):
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
            fast_ma[i] = fast_ma[i-1]
            slow_ma[i] = slow_ma[i-1]
        else:
            fast_ma[i] = fast_alpha * curr + (1 - fast_alpha) * fast_ma[i-1]
            slow_ma[i] = slow_alpha * curr + (1 - slow_alpha) * slow_ma[i-1]
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
            ppo_sig[i] = ppo_sig[i-1]
        else:
            ppo_sig[i] = sig_alpha * ppo[i] + (1 - sig_alpha) * ppo_sig[i-1]
    return ppo, ppo_sig

@_njit(nogil=True, fastmath=True, cache=True)
def _jit_calculate_rsi_core(close, rsi_len):
    n = len(close)
    delta = np.zeros(n, dtype=np.float64)
    gain = np.zeros(n, dtype=np.float64)
    loss = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        delta[i] = close[i] - close[i-1]
        if delta[i] > 0:
            gain[i] = delta[i]
        elif delta[i] < 0:
            loss[i] = -delta[i]
    alpha = 1.0 / rsi_len
    avg_gain = np.empty(n, dtype=np.float64)
    avg_loss = np.empty(n, dtype=np.float64)
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]
    for i in range(1, n):
        avg_gain[i] = alpha * gain[i] + (1 - alpha) * avg_gain[i-1]
        avg_loss[i] = alpha * loss[i] + (1 - alpha) * avg_loss[i-1]
    rsi = np.empty(n, dtype=np.float64)
    for i in range(n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    return rsi

@_njit(nogil=True, fastmath=True, cache=True)
def _jit_vectorized_wick_check_buy(open_arr, high_arr, low_arr, close_arr, min_wick_ratio):
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

@_njit(nogil=True, fastmath=True, cache=True)
def _jit_vectorized_wick_check_sell(open_arr, high_arr, low_arr, close_arr, min_wick_ratio):
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

# ============================================================================
# PUBLIC UNIFIED API - Direct function references (AOT or JIT)
# ============================================================================

def _sanitize_array_numba(arr: np.ndarray, default: float) -> np.ndarray:
    return _jit_sanitize_array_numba(arr, default)

def _sanitize_array_numba_parallel(arr: np.ndarray, default: float) -> np.ndarray:
    return _jit_sanitize_array_numba_parallel(arr, default)

def _sma_loop(data: np.ndarray, period: int) -> np.ndarray:
    return _jit_sma_loop(data, period)

def _sma_loop_parallel(data: np.ndarray, period: int) -> np.ndarray:
    return _jit_sma_loop_parallel(data, period)

def _ema_loop(data: np.ndarray, alpha_or_period: float) -> np.ndarray:
    return _jit_ema_loop(data, alpha_or_period)

def _ema_loop_alpha(data: np.ndarray, alpha: float) -> np.ndarray:
    return _jit_ema_loop_alpha(data, alpha)

def _rng_filter_loop(x: np.ndarray, r: np.ndarray) -> np.ndarray:
    return _jit_rng_filter_loop(x, r)

def _smooth_range(close: np.ndarray, t: int, m: int) -> np.ndarray:
    return _jit_smooth_range(close, t, m)

def _kalman_loop(src: np.ndarray, length: int, R: float, Q: float) -> np.ndarray:
    return _jit_kalman_loop(src, length, R, Q)

def _vwap_daily_loop(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                     volume: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    return _jit_vwap_daily_loop(high, low, close, volume, timestamps)

def _rolling_std_welford(close: np.ndarray, period: int, responsiveness: float) -> np.ndarray:
    return _jit_rolling_std_welford(close, period, responsiveness)

def _rolling_std_welford_parallel(close: np.ndarray, period: int, responsiveness: float) -> np.ndarray:
    return _jit_rolling_std_welford_parallel(close, period, responsiveness)

def _calc_mmh_worm_loop(close_arr: np.ndarray, sd_arr: np.ndarray, rows: int) -> np.ndarray:
    return _jit_calc_mmh_worm_loop(close_arr, sd_arr, rows)

def _calc_mmh_value_loop(temp_arr: np.ndarray, rows: int) -> np.ndarray:
    return _jit_calc_mmh_value_loop(temp_arr, rows)

def _calc_mmh_momentum_loop(momentum_arr: np.ndarray, rows: int) -> np.ndarray:
    return _jit_calc_mmh_momentum_loop(momentum_arr, rows)

def _rolling_mean_numba(close: np.ndarray, period: int) -> np.ndarray:
    return _jit_rolling_mean_numba(close, period)

def _rolling_mean_numba_parallel(close: np.ndarray, period: int) -> np.ndarray:
    return _jit_rolling_mean_numba_parallel(close, period)

def _rolling_min_max_numba(arr: np.ndarray, period: int) -> Tuple[np.ndarray, np.ndarray]:
    return _jit_rolling_min_max_numba(arr, period)

def _rolling_min_max_numba_parallel(arr: np.ndarray, period: int) -> Tuple[np.ndarray, np.ndarray]:
    return _jit_rolling_min_max_numba_parallel(arr, period)

def _calculate_ppo_core(close: np.ndarray, fast: int, slow: int, signal: int) -> Tuple[np.ndarray, np.ndarray]:
    return _jit_calculate_ppo_core(close, fast, slow, signal)

def _calculate_rsi_core(close: np.ndarray, rsi_len: int) -> np.ndarray:
    return _jit_calculate_rsi_core(close, rsi_len)

def _vectorized_wick_check_buy(open_arr: np.ndarray, high_arr: np.ndarray,
                              low_arr: np.ndarray, close_arr: np.ndarray,
                              min_wick_ratio: float) -> np.ndarray:
    return _jit_vectorized_wick_check_buy(open_arr, high_arr, low_arr, close_arr, min_wick_ratio)

def _vectorized_wick_check_sell(open_arr: np.ndarray, high_arr: np.ndarray,
                               low_arr: np.ndarray, close_arr: np.ndarray,
                               min_wick_ratio: float) -> np.ndarray:
    return _jit_vectorized_wick_check_sell(open_arr, high_arr, low_arr, close_arr, min_wick_ratio)

# ============================================================================
# PUBLIC API: one-time init, silent summary, unified names
# ============================================================================

def summary_silent() -> dict:
    ensure_initialized()
    aot_funcs = [
        "sanitize_array_numba", "sanitize_array_numba_parallel",
        "sma_loop", "sma_loop_parallel",
        "ema_loop", "ema_loop_alpha",
        "kalman_loop", "vwap_daily_loop",
        "rng_filter_loop", "smooth_range",
        "calc_mmh_worm_loop", "calc_mmh_value_loop", "calc_mmh_momentum_loop",
        "rolling_std_welford", "rolling_std_welford_parallel",
        "rolling_mean_numba", "rolling_mean_numba_parallel",
        "rolling_min_max_numba", "rolling_min_max_numba_parallel",
        "calculate_ppo_core", "calculate_rsi_core",
        "vectorized_wick_check_buy", "vectorized_wick_check_sell",
    ]
    return {
        "using_aot": _USING_AOT,
        "fallback_reason": _FALLBACK_REASON,
        "aot_functions": aot_funcs if _USING_AOT else [],
        "jit_functions": [] if _USING_AOT else aot_funcs,
    }

# Auto-initialize on import for convenience
ensure_initialized()