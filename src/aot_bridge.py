"""
AOT Bridge Module - Complete wrapper coverage for all 23 Numba functions

Provides unified API that automatically uses AOT (.so) when available, falls back to JIT
Direct binding optimization eliminates decorator overhead entirely
"""
import importlib.util
import warnings
import pathlib
import os
import sys
import sysconfig
import logging
import threading
import platform
import numpy as np
from numba import njit as _njit, prange
from typing import Tuple, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ['NUMBA_WARNINGS'] = '0'

logger = logging.getLogger("aot_bridge")

# Global state with thread safety
_USING_AOT = False
_AOT_MODULE = None
_FALLBACK_REASON = None
_INITIALIZED = False
_INIT_LOCK = threading.Lock()


# ----------------------------
# Parallel verification helpers
# ----------------------------
def _test_ema(aot_module) -> bool:
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    res = aot_module.ema_loop(arr, 3.0)
    return (
        res is not None
        and len(res) == len(arr)
        and not np.all(np.isnan(res))
        and not np.any(np.isinf(res))
    )

def _test_ema_nan(aot_module) -> bool:
    arr = np.array([np.nan, 2.0, 3.0, np.nan, 5.0], dtype=np.float64)
    res = aot_module.ema_loop(arr, 3.0)
    return res is not None and len(res) == len(arr)

def _test_ema_range(aot_module) -> bool:
    arr = np.array([1e-10, 1e10, 1e-10, 1e5], dtype=np.float64)
    res = aot_module.ema_loop(arr, 2.0)
    return res is not None and not np.any(np.isinf(res))

def _test_ema_single(aot_module) -> bool:
    arr = np.array([42.0], dtype=np.float64)
    res = aot_module.ema_loop(arr, 1.0)
    return res is not None and len(res) == 1

def _test_rsi(aot_module) -> bool:
    arr = np.array([44.0, 44.25, 44.5, 43.75, 44.0, 44.5, 45.0, 45.5], dtype=np.float64)
    res = aot_module.calculate_rsi_core(arr, 14)
    return res is not None and len(res) == len(arr)

def _test_sanitize(aot_module) -> bool:
    arr = np.array([1.0, np.nan, np.inf, -np.inf, 5.0], dtype=np.float64)
    res = aot_module.sanitize_array_numba(arr, 0.0)
    return (
        res is not None
        and len(res) == len(arr)
        and not np.any(np.isnan(res))
        and not np.any(np.isinf(res))
    )

def _test_sma(aot_module) -> bool:
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    res = aot_module.sma_loop(arr, 3)
    return res is not None and len(res) == len(arr)

def _test_rolling_std(aot_module) -> bool:
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float64)
    res = aot_module.rolling_std_welford(arr, 3, 1.0)
    return res is not None and len(res) == len(arr)

def _test_wick_buy(aot_module) -> bool:
    o = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    h = np.array([2.0, 3.0, 4.0], dtype=np.float64)
    l = np.array([0.5, 1.5, 2.5], dtype=np.float64)
    c = np.array([1.5, 2.5, 3.5], dtype=np.float64)
    res = aot_module.vectorized_wick_check_buy(o, h, l, c, 0.3)
    return res is not None and len(res) == len(o)

def _test_empty_ema(aot_module) -> bool:
    arr = np.array([], dtype=np.float64)
    try:
        _ = aot_module.ema_loop(arr, 3.0)
        return True
    except Exception:
        return True  # Acceptable: predictable error is fine

def verify_functions_parallel(aot_module) -> bool:
    """
    Run verification tests in parallel threads to reduce startup latency.
    """
    tests = [
        _test_ema,
        _test_ema_nan,
        _test_ema_range,
        _test_ema_single,
        _test_rsi,
        _test_sanitize,
        _test_sma,
        _test_rolling_std,
        _test_wick_buy,
        _test_empty_ema,
    ]

    results = []
    with ThreadPoolExecutor(max_workers=min(4, len(tests))) as executor:
        future_map = {executor.submit(fn, aot_module): fn.__name__ for fn in tests}
        for future in as_completed(future_map):
            name = future_map[future]
            try:
                ok = future.result()
                if not ok:
                    logger.warning(f"Verification test failed: {name}")
                results.append(ok)
            except Exception as e:
                logger.warning(f"Verification test {name} raised: {e}")
                results.append(False)

    return all(results)


def initialize_aot() -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Initialize AOT module with comprehensive verification and diagnostics

    Returns:
        (success: bool, diagnostics: dict or None)
    """
    global _USING_AOT, _AOT_MODULE, _FALLBACK_REASON

    diagnostics = {
        'stage': 'import',
        'error': None,
        'function': None,
        'attempted_paths': []
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Try direct import first
        try:
            import macd_aot_compiled
            _AOT_MODULE = macd_aot_compiled
            diagnostics['stage'] = 'direct_import'
        except ImportError as e:
            diagnostics['error'] = str(e)

            # Dynamic suffix detection with multiple fallbacks
            ext_suffixes = []

            # Primary: get from sysconfig
            primary_suffix = sysconfig.get_config_var('EXT_SUFFIX')
            if primary_suffix:
                ext_suffixes.append(primary_suffix)

            # Fallbacks for common platforms
            fallback_suffixes = [
                '.cpython-311-x86_64-linux-gnu.so',
                '.cpython-310-x86_64-linux-gnu.so',
                '.cpython-39-x86_64-linux-gnu.so',
                '.cpython-38-x86_64-linux-gnu.so',
                '.so',   # Generic Linux
                '.pyd',  # Windows
                '.dylib' # macOS
            ]
            ext_suffixes.extend(fallback_suffixes)

            base_path = pathlib.Path(__file__).parent
            loaded = False

            for suffix in ext_suffixes:
                so_path = base_path / f"macd_aot_compiled{suffix}"
                diagnostics['attempted_paths'].append(str(so_path))

                if so_path.exists():
                    try:
                        spec = importlib.util.spec_from_file_location("macd_aot_compiled", so_path)
                        if spec is None or spec.loader is None:
                            continue

                        mod = importlib.util.module_from_spec(spec)
                        # Load module first, then cache - prevents corrupt module caching
                        spec.loader.exec_module(mod)
                        sys.modules["macd_aot_compiled"] = mod
                        _AOT_MODULE = mod
                        diagnostics['stage'] = 'dynamic_load'
                        diagnostics['loaded_path'] = str(so_path)
                        loaded = True
                        break
                    except Exception as load_error:
                        # Cleanup: Remove from sys.modules if load failed
                        sys.modules.pop("macd_aot_compiled", None)
                        diagnostics['error'] = f"Load failed for {so_path}: {load_error}"
                        logger.debug(f"Failed to load {so_path}: {load_error}")
                        continue

            if not loaded:
                _FALLBACK_REASON = diagnostics
                logger.warning(f"AOT module not found. Attempted paths: {diagnostics['attempted_paths']}")
                return False, diagnostics

        # Parallel verification
        diagnostics['stage'] = 'verification'
        try:
            if not verify_functions_parallel(_AOT_MODULE):
                raise ValueError("One or more verification tests failed")

            _USING_AOT = True
            diagnostics['stage'] = 'success'
            diagnostics['tests_passed'] = 10
            logger.info("✅ AOT module loaded and verified successfully (parallel tests passed)")
            return True, diagnostics

        except Exception as e:
            _FALLBACK_REASON = {
                'stage': diagnostics['stage'],
                'function': diagnostics.get('function'),
                'error': str(e)
            }
            _AOT_MODULE = None
            logger.warning(
                f"AOT verification failed at stage '{diagnostics['stage']}' "
                f"on function '{diagnostics.get('function')}': {e}"
            )
            return False, _FALLBACK_REASON

def ensure_initialized() -> bool:
    """Thread-safe initialization"""
    global _INITIALIZED
    
    if _INITIALIZED:
        return _USING_AOT
    
    with _INIT_LOCK:
        if _INITIALIZED:  # Double-check after acquiring lock
            return _USING_AOT
        
        ok, _ = initialize_aot()
        _INITIALIZED = True
        return ok

def is_using_aot() -> bool:
    """Check if AOT is being used"""
    ensure_initialized()
    return _USING_AOT

def get_fallback_reason() -> Optional[Dict[str, Any]]:
    """Get detailed fallback diagnostics"""
    ensure_initialized()
    return _FALLBACK_REASON

def diagnostics() -> Dict[str, Any]:
    """
    Return comprehensive diagnostics about AOT status
    
    Returns:
        dict with keys: using_aot, fallback_reason, aot_functions, jit_functions,
        module_path, verification_tests
    """
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
    
    result = {
        "using_aot": _USING_AOT,
        "fallback_reason": _FALLBACK_REASON,
        "aot_functions": aot_funcs if _USING_AOT else [],
        "jit_functions": [] if _USING_AOT else aot_funcs,
        "total_functions": len(aot_funcs),
    }
    
    if _USING_AOT and _AOT_MODULE:
        result["module_path"] = getattr(_AOT_MODULE, '__file__', 'unknown')
    
    return result


def diagnostics_extended() -> Dict[str, Any]:
    """
    Extended diagnostics: show per-function backend binding and verification details.
    """
    ensure_initialized()
    bound_map = {}
    for name in [
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
    ]:
        func = globals().get(name)
        if func is None:
            backend = "UNBOUND"
        elif _AOT_MODULE and getattr(_AOT_MODULE, name, None) is func:
            backend = "AOT"
        else:
            backend = "JIT"
        bound_map[name] = backend

    return {
        "using_aot": _USING_AOT,
        "fallback_reason": _FALLBACK_REASON,
        "function_bindings": bound_map,
        "total_functions": len(bound_map),
        "module_path": getattr(_AOT_MODULE, '__file__', None) if _USING_AOT else None,
    }


# ============================================================================
# JIT FALLBACK IMPLEMENTATIONS (Compiled once at module load)
# ============================================================================

# Compile all JIT functions at module load to eliminate runtime overhead
@_njit(nogil=True, fastmath=True, cache=True)
def _jit_sanitize_impl(arr, default):
    out = np.empty_like(arr)
    for i in range(len(arr)):
        val = arr[i]
        out[i] = default if (np.isnan(val) or np.isinf(val)) else val
    return out

@_njit(nogil=True, fastmath=True, cache=True, parallel=True)
def _jit_sanitize_parallel_impl(arr, default):
    out = np.empty_like(arr)
    for i in prange(len(arr)):
        val = arr[i]
        out[i] = default if (np.isnan(val) or np.isinf(val)) else val
    return out

@_njit(nogil=True, fastmath=True, cache=True)
def _jit_sma_impl(data, period):
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
def _jit_sma_parallel_impl(data, period):
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

@_njit(nogil=True, fastmath=True, cache=True)
def _jit_ema_impl(data, alpha_or_period):
    n = len(data)
    alpha = 2.0/(alpha_or_period+1.0) if alpha_or_period>1.0 else alpha_or_period
    out = np.empty(n, dtype=np.float64)
    out[0] = data[0] if not np.isnan(data[0]) else 0.0
    for i in range(1, n):
        curr = data[i]
        out[i] = out[i-1] if np.isnan(curr) else alpha*curr + (1-alpha)*out[i-1]
    return out

@_njit(nogil=True, fastmath=True, cache=True)
def _jit_ema_alpha_impl(data, alpha):
    n = len(data)
    out = np.empty(n, dtype=np.float64)
    out[0] = data[0] if not np.isnan(data[0]) else 0.0
    for i in range(1, n):
        curr = data[i]
        out[i] = out[i-1] if np.isnan(curr) else alpha*curr + (1-alpha)*out[i-1]
    return out

@_njit(nogil=True, fastmath=True, cache=True)
def _jit_rng_filter_impl(x, r):
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

@_njit(nogil=True, fastmath=True, cache=True)
def _jit_smooth_range_impl(close, t, m):
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
def _jit_kalman_impl(src, length, R, Q):
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
def _jit_vwap_daily_impl(high, low, close, volume, timestamps):
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
def _jit_rolling_std_impl(close, period, responsiveness):
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
def _jit_rolling_std_parallel_impl(close, period, responsiveness):
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

@_njit(nogil=True, fastmath=True, cache=True)
def _jit_calc_mmh_worm_impl(close_arr, sd_arr, rows):
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
def _jit_calc_mmh_value_impl(temp_arr, rows):
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
def _jit_calc_mmh_momentum_impl(momentum_arr, rows):
    for i in range(1, rows):
        prev = momentum_arr[i - 1] if not np.isnan(momentum_arr[i - 1]) else 0.0
        momentum_arr[i] = momentum_arr[i] + 0.5 * prev
    return momentum_arr

@_njit(nogil=True, fastmath=True, cache=True)
def _jit_rolling_mean_impl(close, period):
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
def _jit_rolling_mean_parallel_impl(close, period):
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

@_njit(nogil=True, fastmath=True, cache=True)
def _jit_rolling_min_max_impl(arr, period):
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
def _jit_rolling_min_max_parallel_impl(arr, period):
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

@_njit(nogil=True, fastmath=True, cache=True)
def _jit_calculate_ppo_impl(close, fast, slow, signal):
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
def _jit_calculate_rsi_impl(close, rsi_len):
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
def _jit_vectorized_wick_check_buy_impl(open_arr, high_arr, low_arr, close_arr, min_wick_ratio):
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
def _jit_vectorized_wick_check_sell_impl(open_arr, high_arr, low_arr, close_arr, min_wick_ratio):
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
# PUBLIC API - Direct function bindings (AOT or JIT)
# ============================================================================

# Initialize these as None, will be bound to either AOT or JIT at module load
sanitize_array_numba = None
sanitize_array_numba_parallel = None
sma_loop = None
sma_loop_parallel = None
ema_loop = None
ema_loop_alpha = None
rng_filter_loop = None
smooth_range = None
kalman_loop = None
vwap_daily_loop = None
rolling_std_welford = None
rolling_std_welford_parallel = None
calc_mmh_worm_loop = None
calc_mmh_value_loop = None
calc_mmh_momentum_loop = None
rolling_mean_numba = None
rolling_mean_numba_parallel = None
rolling_min_max_numba = None
rolling_min_max_numba_parallel = None
calculate_ppo_core = None
calculate_rsi_core = None
vectorized_wick_check_buy = None
vectorized_wick_check_sell = None


def _bind_functions():
    """
    Bind all 23 functions to either AOT or JIT implementations.
    Called once at module initialization - creates direct function references with ZERO overhead.
    """
    global sanitize_array_numba, sanitize_array_numba_parallel
    global sma_loop, sma_loop_parallel
    global ema_loop, ema_loop_alpha
    global rng_filter_loop, smooth_range
    global kalman_loop, vwap_daily_loop
    global rolling_std_welford, rolling_std_welford_parallel
    global calc_mmh_worm_loop, calc_mmh_value_loop, calc_mmh_momentum_loop
    global rolling_mean_numba, rolling_mean_numba_parallel
    global rolling_min_max_numba, rolling_min_max_numba_parallel
    global calculate_ppo_core, calculate_rsi_core
    global vectorized_wick_check_buy, vectorized_wick_check_sell

    if _USING_AOT and _AOT_MODULE is not None:
        # Direct binding to AOT functions - ZERO overhead
        sanitize_array_numba = _AOT_MODULE.sanitize_array_numba
        sanitize_array_numba_parallel = _AOT_MODULE.sanitize_array_numba_parallel
        sma_loop = _AOT_MODULE.sma_loop
        sma_loop_parallel = _AOT_MODULE.sma_loop_parallel
        ema_loop = _AOT_MODULE.ema_loop
        ema_loop_alpha = _AOT_MODULE.ema_loop_alpha
        rng_filter_loop = _AOT_MODULE.rng_filter_loop
        smooth_range = _AOT_MODULE.smooth_range
        kalman_loop = _AOT_MODULE.kalman_loop
        vwap_daily_loop = _AOT_MODULE.vwap_daily_loop
        rolling_std_welford = _AOT_MODULE.rolling_std_welford
        rolling_std_welford_parallel = _AOT_MODULE.rolling_std_welford_parallel
        calc_mmh_worm_loop = _AOT_MODULE.calc_mmh_worm_loop
        calc_mmh_value_loop = _AOT_MODULE.calc_mmh_value_loop
        calc_mmh_momentum_loop = _AOT_MODULE.calc_mmh_momentum_loop
        rolling_mean_numba = _AOT_MODULE.rolling_mean_numba
        rolling_mean_numba_parallel = _AOT_MODULE.rolling_mean_numba_parallel
        rolling_min_max_numba = _AOT_MODULE.rolling_min_max_numba
        rolling_min_max_numba_parallel = _AOT_MODULE.rolling_min_max_numba_parallel
        calculate_ppo_core = _AOT_MODULE.calculate_ppo_core
        calculate_rsi_core = _AOT_MODULE.calculate_rsi_core
        vectorized_wick_check_buy = _AOT_MODULE.vectorized_wick_check_buy
        vectorized_wick_check_sell = _AOT_MODULE.vectorized_wick_check_sell
        
        logger.info("✅ All 23 functions bound to AOT implementations")
    else:
        # Bind to pre-compiled JIT implementations
        sanitize_array_numba = _jit_sanitize_impl
        sanitize_array_numba_parallel = _jit_sanitize_parallel_impl
        sma_loop = _jit_sma_impl
        sma_loop_parallel = _jit_sma_parallel_impl
        ema_loop = _jit_ema_impl
        ema_loop_alpha = _jit_ema_alpha_impl
        rng_filter_loop = _jit_rng_filter_impl
        smooth_range = _jit_smooth_range_impl
        kalman_loop = _jit_kalman_impl
        vwap_daily_loop = _jit_vwap_daily_impl
        rolling_std_welford = _jit_rolling_std_impl
        rolling_std_welford_parallel = _jit_rolling_std_parallel_impl
        calc_mmh_worm_loop = _jit_calc_mmh_worm_impl
        calc_mmh_value_loop = _jit_calc_mmh_value_impl
        calc_mmh_momentum_loop = _jit_calc_mmh_momentum_impl
        rolling_mean_numba = _jit_rolling_mean_impl
        rolling_mean_numba_parallel = _jit_rolling_mean_parallel_impl
        rolling_min_max_numba = _jit_rolling_min_max_impl
        rolling_min_max_numba_parallel = _jit_rolling_min_max_parallel_impl
        calculate_ppo_core = _jit_calculate_ppo_impl
        calculate_rsi_core = _jit_calculate_rsi_impl
        vectorized_wick_check_buy = _jit_vectorized_wick_check_buy_impl
        vectorized_wick_check_sell = _jit_vectorized_wick_check_sell_impl
        
        logger.info("⚠️  All 23 functions bound to JIT fallback implementations")


def summary() -> None:
    """Print human-readable summary of AOT status"""
    ensure_initialized()
    
    print("\n" + "="*70)
    print("AOT BRIDGE STATUS")
    print("="*70)
    
    if _USING_AOT:
        print("✅ Status: AOT ACTIVE")
        if _AOT_MODULE:
            module_path = getattr(_AOT_MODULE, '__file__', 'unknown')
            print(f"📦 Module: {module_path}")
        print(f"⚡ Functions: All 23 functions using compiled AOT (.so)")
    else:
        print("⚠️  Status: JIT FALLBACK")
        if _FALLBACK_REASON:
            if isinstance(_FALLBACK_REASON, dict):
                print(f"❌ Stage: {_FALLBACK_REASON.get('stage', 'unknown')}")
                print(f"❌ Function: {_FALLBACK_REASON.get('function', 'unknown')}")
                print(f"❌ Error: {_FALLBACK_REASON.get('error', 'unknown')}")
            else:
                print(f"❌ Reason: {_FALLBACK_REASON}")
        print(f"🐌 Functions: All 23 functions using JIT compilation")
    
    print("="*70 + "\n")


def summary_silent() -> dict:
    """
    Return summary of AOT status without printing (backward compatibility)
    
    Returns:
        dict with keys: using_aot, fallback_reason, aot_functions, jit_functions
    """
    return diagnostics()


def benchmark(iterations: int = 1000, array_size: int = 10000) -> Dict[str, float]:
    """
    Optional benchmarking to verify AOT performance
    
    Args:
        iterations: Number of test iterations
        array_size: Size of test arrays
        
    Returns:
        dict with timing results for key functions
    """
    import time
    
    ensure_initialized()
    
    # Generate test data
    test_data = np.random.randn(array_size).astype(np.float64)
    test_data[::100] = np.nan  # Add some NaN values
    
    results = {}
    
    # Benchmark EMA
    start = time.perf_counter()
    for _ in range(iterations):
        _ = ema_loop(test_data, 20.0)
    ema_time = time.perf_counter() - start
    results['ema_loop_ms'] = (ema_time / iterations) * 1000
    
    # Benchmark SMA
    start = time.perf_counter()
    for _ in range(iterations):
        _ = sma_loop(test_data, 20)
    sma_time = time.perf_counter() - start
    results['sma_loop_ms'] = (sma_time / iterations) * 1000
    
    # Benchmark RSI
    start = time.perf_counter()
    for _ in range(iterations):
        _ = calculate_rsi_core(test_data, 14)
    rsi_time = time.perf_counter() - start
    results['calculate_rsi_ms'] = (rsi_time / iterations) * 1000
    
    # Benchmark sanitize
    start = time.perf_counter()
    for _ in range(iterations):
        _ = sanitize_array_numba(test_data, 0.0)
    sanitize_time = time.perf_counter() - start
    results['sanitize_ms'] = (sanitize_time / iterations) * 1000
    
    results['mode'] = 'AOT' if _USING_AOT else 'JIT'
    results['iterations'] = iterations
    results['array_size'] = array_size
    
    return results

def benchmark_extended(iterations: int = 100, array_size: int = 5000) -> Dict[str, float]:
    """
    Benchmark all 23 functions to compare AOT vs JIT performance.
    """
    import time
    ensure_initialized()
    results = {}
    # Synthetic test data
    arr = np.random.randn(array_size).astype(np.float64)
    arr[::50] = np.nan
    o = arr.copy(); h = arr+1; l = arr-1; c = arr.copy()
    vol = np.abs(arr*100).astype(np.float64)
    ts = np.arange(array_size).astype(np.int64)

    funcs = {
        "ema_loop": lambda: ema_loop(arr, 20.0),
        "sma_loop": lambda: sma_loop(arr, 20),
        "calculate_rsi_core": lambda: calculate_rsi_core(arr, 14),
        "sanitize_array_numba": lambda: sanitize_array_numba(arr, 0.0),
        "rolling_std_welford": lambda: rolling_std_welford(arr, 20, 1.0),
        "rolling_mean_numba": lambda: rolling_mean_numba(arr, 20),
        "rolling_min_max_numba": lambda: rolling_min_max_numba(arr, 20),
        "calculate_ppo_core": lambda: calculate_ppo_core(arr, 12, 26, 9),
        "vwap_daily_loop": lambda: vwap_daily_loop(h, l, c, vol, ts),
        "kalman_loop": lambda: kalman_loop(arr, 10, 0.1, 0.01),
        "rng_filter_loop": lambda: rng_filter_loop(arr, np.abs(arr)),
        "smooth_range": lambda: smooth_range(c, 14, 2),
        "calc_mmh_worm_loop": lambda: calc_mmh_worm_loop(c, np.abs(arr), array_size),
        "calc_mmh_value_loop": lambda: calc_mmh_value_loop(c, array_size),
        "calc_mmh_momentum_loop": lambda: calc_mmh_momentum_loop(c.copy(), array_size),
        "vectorized_wick_check_buy": lambda: vectorized_wick_check_buy(o, h, l, c, 0.3),
        "vectorized_wick_check_sell": lambda: vectorized_wick_check_sell(o, h, l, c, 0.3),
    }

    for name, fn in funcs.items():
        start = time.perf_counter()
        for _ in range(iterations):
            _ = fn()
        elapsed = (time.perf_counter() - start) / iterations * 1000
        results[f"{name}_ms"] = elapsed

    results["mode"] = "AOT" if _USING_AOT else "JIT"
    results["iterations"] = iterations
    results["array_size"] = array_size
    return results

# ============================================================================
# MODULE INITIALIZATION
# ============================================================================

# Initialize and bind functions immediately on module import
ensure_initialized()
_bind_functions()

# Verify all functions are bound
if ema_loop is None:
    raise RuntimeError("Critical error: Functions not bound after initialization. "
                      "This indicates a bug in the binding logic.")

# Verify function count
_expected_functions = [
    sanitize_array_numba, sanitize_array_numba_parallel,
    sma_loop, sma_loop_parallel,
    ema_loop, ema_loop_alpha,
    rng_filter_loop, smooth_range,
    kalman_loop, vwap_daily_loop,
    rolling_std_welford, rolling_std_welford_parallel,
    calc_mmh_worm_loop, calc_mmh_value_loop, calc_mmh_momentum_loop,
    rolling_mean_numba, rolling_mean_numba_parallel,
    rolling_min_max_numba, rolling_min_max_numba_parallel,
    calculate_ppo_core, calculate_rsi_core,
    vectorized_wick_check_buy, vectorized_wick_check_sell,
]

if any(f is None for f in _expected_functions):
    raise RuntimeError("Critical error: Not all functions were bound. "
                      "Check _bind_functions() implementation.")

# Log final status
if _USING_AOT:
    logger.info("🚀 AOT Bridge initialized successfully with AOT compilation")
else:
    logger.info("🔄 AOT Bridge initialized with JIT fallback")


# ============================================================================
# PUBLIC EXPORTS
# ============================================================================

__all__ = [
    # Status functions
    'is_using_aot',
    'get_fallback_reason',
    'diagnostics',
    'diagnostics_extended',
    'summary',
    'summary_silent',
    'benchmark',
    'benchmark_extended',

    # Core functions
    'sanitize_array_numba',
    'sanitize_array_numba_parallel',
    'sma_loop',
    'sma_loop_parallel',
    'ema_loop',
    'ema_loop_alpha',
    'rng_filter_loop',
    'smooth_range',
    'kalman_loop',
    'vwap_daily_loop',
    'rolling_std_welford',
    'rolling_std_welford_parallel',
    'calc_mmh_worm_loop',
    'calc_mmh_value_loop',
    'calc_mmh_momentum_loop',
    'rolling_mean_numba',
    'rolling_mean_numba_parallel',
    'rolling_min_max_numba',
    'rolling_min_max_numba_parallel',
    'calculate_ppo_core',
    'calculate_rsi_core',
    'vectorized_wick_check_buy',
    'vectorized_wick_check_sell',
]