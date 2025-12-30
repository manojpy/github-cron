"""
AOT Bridge Module - Zero-Overhead Performance Dispatcher
Uses leading-underscore naming convention to match existing macd_unified.py imports.
"""
import importlib.util
import warnings
import pathlib
import os
import sys
import logging
import numpy as np
from numba import njit, prange
from typing import Tuple, Optional

os.environ['NUMBA_WARNINGS'] = '0'
logger = logging.getLogger("aot_bridge")

_USING_AOT = False
_AOT_MODULE = None
_FALLBACK_REASON = None
_INITIALIZED = False

# =========================================================================
# 1. JIT FALLBACKS (Optimized O(n) Logic)
# =========================================================================

@njit(cache=True)
def _sanitize_array_numba(arr, default):
    out = np.empty_like(arr)
    for i in range(len(arr)):
        val = arr[i]
        out[i] = default if (np.isnan(val) or np.isinf(val)) else val
    return out

@njit(parallel=True, cache=True)
def _sanitize_array_numba_parallel(arr, default):
    n = len(arr)
    out = np.empty_like(arr)
    for i in prange(n):
        val = arr[i]
        out[i] = default if (np.isnan(val) or np.isinf(val)) else val
    return out

@njit(cache=True)
def _sma_loop(data, period):
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

@njit(parallel=True, cache=True)
def _sma_loop_parallel(data, period):
    return _sma_loop(data, period)

@njit(cache=True)
def _rolling_mean_numba(close, period):
    return _sma_loop(close, period)

@njit(parallel=True, cache=True)
def _rolling_mean_numba_parallel(close, period):
    return _sma_loop(close, period)

@njit(cache=True)
def _rolling_std_welford(close, period, responsiveness):
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

@njit(parallel=True, cache=True)
def _rolling_std_welford_parallel(close, period, responsiveness):
    return _rolling_std_welford(close, period, responsiveness)

@njit(cache=True)
def _rolling_min_max_numba(arr, period):
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

@njit(parallel=True, cache=True)
def _rolling_min_max_numba_parallel(arr, period):
    return _rolling_min_max_numba(arr, period)

@njit(cache=True)
def _ema_loop(data, alpha_or_period):
    n = len(data)
    alpha = 2.0 / (alpha_or_period + 1.0) if alpha_or_period > 1.0 else alpha_or_period
    out = np.empty(n, dtype=np.float64)
    out[0] = data[0] if not np.isnan(data[0]) else 0.0
    for i in range(1, n):
        curr = data[i]
        out[i] = out[i-1] if np.isnan(curr) else alpha * curr + (1 - alpha) * out[i-1]
    return out

@njit(cache=True)
def _ema_loop_alpha(data, alpha):
    return _ema_loop(data, alpha)

@njit(cache=True)
def _calculate_ppo_core(close, fast, slow, signal):
    n = len(close)
    f_ma = _ema_loop(close, float(fast))
    s_ma = _ema_loop(close, float(slow))
    ppo = np.empty(n, dtype=np.float64)
    for i in range(n):
        if np.isnan(s_ma[i]) or abs(s_ma[i]) < 1e-12:
            ppo[i] = 0.0
        else:
            ppo[i] = ((f_ma[i] - s_ma[i]) / s_ma[i]) * 100.0
    ppo_sig = _ema_loop(ppo, float(signal))
    return ppo, ppo_sig

@njit(cache=True)
def _calculate_rsi_core(close, rsi_len):
    n = len(close)
    alpha = 1.0 / rsi_len
    gain, loss = np.zeros(n), np.zeros(n)
    for i in range(1, n):
        delta = close[i] - close[i-1]
        if delta > 0: gain[i] = delta
        elif delta < 0: loss[i] = -delta
    avg_g = _ema_loop(gain, alpha)
    avg_l = _ema_loop(loss, alpha)
    rsi = np.empty(n)
    for i in range(n):
        if avg_l[i] < 1e-10: rsi[i] = 100.0 if avg_g[i] > 1e-10 else 50.0
        else: rsi[i] = 100.0 - (100.0 / (1.0 + (avg_g[i] / avg_l[i])))
    return rsi

@njit(cache=True)
def _rng_filter_loop(x, r):
    n = len(x)
    filt = np.empty(n)
    filt[0] = x[0] if not np.isnan(x[0]) else 0.0
    for i in range(1, n):
        if np.isnan(r[i]) or np.isnan(x[i]): filt[i] = filt[i-1]
        elif x[i] > filt[i-1]: filt[i] = max(filt[i-1], x[i] - r[i])
        else: filt[i] = min(filt[i-1], x[i] + r[i])
    return filt

@njit(cache=True)
def _smooth_range(close, t, m):
    n = len(close)
    diff = np.zeros(n)
    for i in range(1, n): diff[i] = abs(close[i] - close[i-1])
    avrng = _ema_loop(diff, float(t))
    smoothrng = _ema_loop(avrng, float(t * 2 - 1))
    return smoothrng * float(m)

@njit(cache=True)
def _kalman_loop(src, length, R, Q):
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

@njit(cache=True)
def _vwap_daily_loop(high, low, close, volume, timestamps):
    n = len(close)
    vwap = np.empty(n)
    c_vol, c_pv, cur_day = 0.0, 0.0, -1
    for i in range(n):
        day = timestamps[i] // 86400
        if day != cur_day: cur_day, c_vol, c_pv = day, 0.0, 0.0
        if np.isnan(high[i]) or volume[i] <= 0:
            vwap[i] = vwap[i-1] if i > 0 else close[i]
            continue
        typical = (high[i] + low[i] + close[i]) / 3.0
        c_vol += volume[i]; c_pv += typical * volume[i]
        vwap[i] = c_pv / c_vol if c_vol > 0 else typical
    return vwap

@njit(cache=True)
def _calc_mmh_worm_loop(close_arr, sd_arr, rows):
    worm = np.empty(rows)
    worm[0] = close_arr[0] if not np.isnan(close_arr[0]) else 0.0
    for i in range(1, rows):
        src = close_arr[i] if not np.isnan(close_arr[i]) else worm[i-1]
        diff = src - worm[i-1]
        sd_i = sd_arr[i]
        delta = (np.sign(diff) * sd_i) if (not np.isnan(sd_i) and abs(diff) > sd_i) else diff
        worm[i] = worm[i-1] + delta
    return worm

@njit(cache=True)
def _calc_mmh_value_loop(temp_arr, rows):
    val = np.zeros(rows)
    val[0] = max(-0.9999, min(0.9999, (temp_arr[0] if not np.isnan(temp_arr[0]) else 0.5) - 0.5))
    for i in range(1, rows):
        t = temp_arr[i] if not np.isnan(temp_arr[i]) else 0.5
        v = t - 0.5 + 0.5 * val[i-1]
        val[i] = max(-0.9999, min(0.9999, v))
    return val

@njit(cache=True)
def _calc_mmh_momentum_loop(momentum_arr, rows):
    for i in range(1, rows): momentum_arr[i] += 0.5 * momentum_arr[i-1]
    return momentum_arr

@njit(cache=True)
def _vectorized_wick_check_buy(o, h, l, c, ratio):
    n = len(c)
    res = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        if c[i] > o[i]:
            rng = h[i] - l[i]
            if rng > 1e-8: res[i] = ((h[i] - c[i]) / rng) < ratio
    return res

@njit(cache=True)
def _vectorized_wick_check_sell(o, h, l, c, ratio):
    n = len(c)
    res = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        if c[i] < o[i]:
            rng = h[i] - l[i]
            if rng > 1e-8: res[i] = ((c[i] - l[i]) / rng) < ratio
    return res

# =========================================================================
# 2. INITIALIZATION LOGIC (Hot-Swapping to Underscored Names)
# =========================================================================

# List of function names as they appear in the AOT compiled module (macd_aot_compiled)
_AOT_EXPORT_NAMES = [
    "sanitize_array_numba", "sanitize_array_numba_parallel", "sma_loop", "sma_loop_parallel",
    "rolling_mean_numba", "rolling_mean_numba_parallel", "rolling_std_welford", "rolling_std_welford_parallel",
    "rolling_min_max_numba", "rolling_min_max_numba_parallel", "ema_loop", "ema_loop_alpha",
    "calculate_ppo_core", "calculate_rsi_core", "rng_filter_loop", "smooth_range",
    "kalman_loop", "vwap_daily_loop", "calc_mmh_worm_loop", "calc_mmh_value_loop",
    "calc_mmh_momentum_loop", "vectorized_wick_check_buy", "vectorized_wick_check_sell"
]

def initialize_aot() -> Tuple[bool, Optional[str]]:
    global _USING_AOT, _AOT_MODULE, _FALLBACK_REASON, _INITIALIZED
    if _INITIALIZED: return _USING_AOT, _FALLBACK_REASON
    
    _INITIALIZED = True
    try:
        import macd_aot_compiled
        _AOT_MODULE = macd_aot_compiled
        
        this_module = sys.modules[__name__]
        for name in _AOT_EXPORT_NAMES:
            if hasattr(_AOT_MODULE, name):
                # Map AOT function (e.g., sma_loop) to local underscored name (e.g., _sma_loop)
                setattr(this_module, f"_{name}", getattr(_AOT_MODULE, name))
        
        _USING_AOT = True
        return True, None
    except ImportError:
        _FALLBACK_REASON = "AOT binary (macd_aot_compiled) not found. Using JIT."
    except Exception as e:
        _FALLBACK_REASON = f"AOT Load Error: {str(e)}"
    
    return False, _FALLBACK_REASON

def ensure_initialized():
    if not _INITIALIZED: initialize_aot()

def summary_silent() -> dict:
    ensure_initialized()
    return {
        "using_aot": _USING_AOT,
        "fallback_reason": _FALLBACK_REASON,
        "function_count": len(_AOT_EXPORT_NAMES)
    }

# Initialize immediately upon module load
initialize_aot()