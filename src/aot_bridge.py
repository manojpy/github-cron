"""
AOT Bridge Module - Complete wrapper coverage for all 23 Numba functions
Provides unified API that automatically uses AOT (.so) when available, falls back to JIT
"""
import importlib.util
import warnings
import pathlib
import os
import sys
import logging
import functools
import numpy as np
from numba import njit, prange
from typing import Tuple, Optional

# ✅ Suppress Numba warnings at import time
os.environ['NUMBA_WARNINGS'] = '0'

logger = logging.getLogger("aot_bridge")

_USING_AOT = False
_AOT_MODULE = None
_FALLBACK_REASON = None
_INITIALIZED = False


def aot_guard(func_name: str):
    def decorator(jit_func):
        @functools.wraps(jit_func)
        def wrapper(*args, **kwargs):
            if _USING_AOT and _AOT_MODULE is not None:
                try:
                    return getattr(_AOT_MODULE, func_name)(*args, **kwargs)
                except AttributeError:
                    logger.warning(f"⚠️ AOT module missing function {func_name}, falling back to JIT")
            # Fallback path
            logger.warning(f"⚠️ Fallback to JIT: {func_name} not using AOT")
            return jit_func(*args, **kwargs)
        return wrapper
    return decorator


def initialize_aot() -> Tuple[bool, Optional[str]]:
    global _USING_AOT, _AOT_MODULE, _FALLBACK_REASON
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            import macd_aot_compiled
            _AOT_MODULE = macd_aot_compiled
        except ImportError:
            so_path = pathlib.Path(__file__).parent / "macd_aot_compiled.cpython-311-x86_64-linux-gnu.so"
            if so_path.exists():
                try:
                    spec = importlib.util.spec_from_file_location("macd_aot_compiled", so_path)
                    if spec is None or spec.loader is None:
                        _FALLBACK_REASON = "Failed to create module spec"
                        logger.warning(f"AOT init failed: {_FALLBACK_REASON}")   # <-- add here
                        return False, _FALLBACK_REASON
                    mod = importlib.util.module_from_spec(spec)
                    sys.modules["macd_aot_compiled"] = mod
                    spec.loader.exec_module(mod)
                    _AOT_MODULE = mod
                except Exception as e:
                    _FALLBACK_REASON = f"Failed to load .so file: {e}"
                    logger.warning(f"AOT init failed: {_FALLBACK_REASON}")       # <-- add here
                    return False, _FALLBACK_REASON
            else:
                _FALLBACK_REASON = "AOT module not found"
                logger.warning(f"AOT init failed: {_FALLBACK_REASON}")           # <-- add here
                return False, _FALLBACK_REASON

        # Verify with a simple test
        try:
            test_data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
            _ = _AOT_MODULE.ema_loop(test_data, 3.0)
            _USING_AOT = True
            return True, None
        except Exception as e:
            _FALLBACK_REASON = f"AOT verification failed: {e}"
            _AOT_MODULE = None
            logger.warning(f"AOT init failed: {_FALLBACK_REASON}")               # <-- add here
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
# UNIFIED API - All 23 Functions with AOT/JIT Auto-Selection (decorator applied)
# ============================================================================

# 1-2: SANITIZATION
@aot_guard("sanitize_array_numba")
def _sanitize_array_numba(arr: np.ndarray, default: float) -> np.ndarray:
    from numba import njit
    @njit(nogil=True, fastmath=True, cache=True)
    def _jit(arr, default):
        out = np.empty_like(arr)
        for i in range(len(arr)):
            val = arr[i]
            out[i] = default if (np.isnan(val) or np.isinf(val)) else val
        return out
    return _jit(arr, default)

@aot_guard("sanitize_array_numba_parallel")
def _sanitize_array_numba_parallel(arr: np.ndarray, default: float) -> np.ndarray:
    from numba import njit, prange
    @njit(nogil=True, fastmath=True, cache=True, parallel=True)
    def _jit(arr, default):
        out = np.empty_like(arr)
        for i in prange(len(arr)):
            val = arr[i]
            out[i] = default if (np.isnan(val) or np.isinf(val)) else val
        return out
    return _jit(arr, default)

# 3-4: SMA
@aot_guard("sma_loop")
def _sma_loop(data: np.ndarray, period: int) -> np.ndarray:
    from numba import njit
    @njit(nogil=True, fastmath=True, cache=True)
    def _jit(data, period):
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
    return _jit(data, period)

@aot_guard("sma_loop_parallel")
def _sma_loop_parallel(data: np.ndarray, period: int) -> np.ndarray:
    from numba import njit, prange
    @njit(nogil=True, fastmath=True, cache=True, parallel=True)
    def _jit(data, period):
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
    return _jit(data, period)

# 5-6: EMA
@aot_guard("ema_loop")
def _ema_loop(data: np.ndarray, alpha_or_period: float) -> np.ndarray:
    from numba import njit
    @njit(nogil=True, fastmath=True, cache=True)
    def _jit(data, alpha_or_period):
        n = len(data)
        alpha = 2.0/(alpha_or_period+1.0) if alpha_or_period>1.0 else alpha_or_period
        out = np.empty(n, dtype=np.float64)
        out[0] = data[0] if not np.isnan(data[0]) else 0.0
        for i in range(1, n):
            curr = data[i]
            out[i] = out[i-1] if np.isnan(curr) else alpha*curr + (1-alpha)*out[i-1]
        return out
    return _jit(data, alpha_or_period)

@aot_guard("ema_loop_alpha")
def _ema_loop_alpha(data: np.ndarray, alpha: float) -> np.ndarray:
    from numba import njit
    @njit(nogil=True, fastmath=True, cache=True)
    def _jit(data, alpha):
        n = len(data)
        out = np.empty(n, dtype=np.float64)
        out[0] = data[0] if not np.isnan(data[0]) else 0.0
        for i in range(1, n):
            curr = data[i]
            out[i] = out[i-1] if np.isnan(curr) else alpha*curr + (1-alpha)*out[i-1]
        return out
    return _jit(data, alpha)

# 7: KALMAN FILTER
@aot_guard("kalman_loop")
def _kalman_loop(src: np.ndarray, length: int, R: float, Q: float) -> np.ndarray:
    from numba import njit
    @njit(nogil=True, fastmath=True, cache=True)
    def _jit(src, length, R, Q):
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
    return _jit(src, length, R, Q)

# 8: VWAP
@aot_guard("vwap_daily_loop")
def _vwap_daily_loop(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                     volume: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    from numba import njit
    @njit(nogil=True, fastmath=True, cache=True)
    def _jit(high, low, close, volume, timestamps):
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
    return _jit(high, low, close, volume, timestamps)

# 9-10: RANGE FILTERS
@aot_guard("rng_filter_loop")
def _rng_filter_loop(x: np.ndarray, r: np.ndarray) -> np.ndarray:
    from numba import njit
    @njit(nogil=True, fastmath=True, cache=True)
    def _jit(x, r):
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
                target = curr_x - curr_r
                filt[i] = max(prev_filt, target)
            else:
                target = curr_x + curr_r
                filt[i] = min(prev_filt, target)
        return filt
    return _jit(x, r)

@aot_guard("smooth_range")
def _smooth_range(close: np.ndarray, t: int, m: int) -> np.ndarray:
    from numba import njit
    @njit(nogil=True, fastmath=True, cache=True)
    def _jit(close, t, m):
        n = len(close)
        diff = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            diff[i] = abs(close[i] - close[i-1])
        # Inline EMA for avrng
        alpha_t = 2.0 / (t + 1.0)
        avrng = np.empty(n, dtype=np.float64)
        avrng[0] = diff[0] if not np.isnan(diff[0]) else 0.0
        for i in range(1, n):
            curr = diff[i]
            avrng[i] = avrng[i-1] if np.isnan(curr) else alpha_t * curr + (1 - alpha_t) * avrng[i-1]
        # Inline EMA for smoothrng
        wper = t * 2 - 1
        alpha_w = 2.0 / (wper + 1.0)
        smoothrng = np.empty(n, dtype=np.float64)
        smoothrng[0] = avrng[0]
        for i in range(1, n):
            curr = avrng[i]
            smoothrng[i] = smoothrng[i-1] if np.isnan(curr) else alpha_w * curr + (1 - alpha_w) * smoothrng[i-1]
        return smoothrng * m
    return _jit(close, t, m)

# 11-13: MMH HELPERS
@aot_guard("calc_mmh_worm_loop")
def _calc_mmh_worm_loop(close_arr: np.ndarray, sd_arr: np.ndarray, rows: int) -> np.ndarray:
    from numba import njit
    @njit(nogil=True, fastmath=True, cache=True)
    def _jit(close_arr, sd_arr, rows):
        worm_arr = np.empty(rows, dtype=np.float64)
        first_val = close_arr[0]
        worm_arr[0] = 0.0 if np.isnan(first_val) else first_val
        for i in range(1, rows):
            src = close_arr[i] if not np.isnan(close_arr[i]) else worm_arr[i - 1]
            prev_worm = worm_arr[i - 1]
            diff = src - prev_worm
            sd_i = sd_arr[i]
            if np.isnan(sd_i):
                delta = diff
            else:
                delta = (np.sign(diff) * sd_i) if (np.abs(diff) > sd_i) else diff
            worm_arr[i] = prev_worm + delta
        return worm_arr
    return _jit(close_arr, sd_arr, rows)

@aot_guard("calc_mmh_value_loop")
def _calc_mmh_value_loop(temp_arr: np.ndarray, rows: int) -> np.ndarray:
    """
    CRITICAL FIX: Pine Script logic is:
    value = 0.5 * 2  // Initial value = 1.0
    value := value * (temp - .5 + .5 * nz(value[1]))
    
    This means: value[i] = value[i-1] * (temp[i] - 0.5 + 0.5 * value[i-1])
    NOT: value[i] = 1.0 * (temp[i] - 0.5 + 0.5 * value[i-1])
    """
    from numba import njit
    @njit(nogil=True, fastmath=True, cache=True)
    def _jit(temp_arr, rows):
        value_arr = np.zeros(rows, dtype=np.float64)
        value_arr[0] = 1.0  # Initial: value = 0.5 * 2 = 1.0
        
        for i in range(1, rows):
            prev_v = value_arr[i - 1] if not np.isnan(value_arr[i - 1]) else 1.0
            t = temp_arr[i] if not np.isnan(temp_arr[i]) else 0.5
            
            # Pine: value := value * (temp - .5 + .5 * nz(value[1]))
            # This is a MULTIPLICATIVE update, not additive
            inner = t - 0.5 + 0.5 * prev_v
            v = prev_v * inner
            
            # Clip to prevent overflow
            value_arr[i] = max(-0.9999, min(0.9999, v))
        
        return value_arr
    return _jit(temp_arr, rows)


@aot_guard("calc_mmh_momentum_loop")
def _calc_mmh_momentum_loop(momentum_arr: np.ndarray, rows: int) -> np.ndarray:
    from numba import njit
    @njit(nogil=True, fastmath=True, cache=True)
    def _jit(momentum_arr, rows):
        for i in range(1, rows):
            prev = momentum_arr[i - 1] if not np.isnan(momentum_arr[i - 1]) else 0.0
            momentum_arr[i] = momentum_arr[i] + 0.5 * prev
        return momentum_arr
    return _jit(momentum_arr, rows)

@aot_guard("rolling_std_welford")
def _rolling_std_welford(close: np.ndarray, period: int, responsiveness: float) -> np.ndarray:
    """Pine's ta.stdev uses SAMPLE std dev (n-1)"""
    from numba import njit
    @njit(nogil=True, fastmath=True, cache=True)
    def _jit(close, period, responsiveness):
        n = len(close)
        sd = np.empty(n, dtype=np.float64)
        resp = max(0.0001, min(1.0, responsiveness))
        for i in range(n):
            mean, m2, count = 0.0, 0.0, 0
            start = max(0, i - period + 1)
            for j in range(start, i + 1):
                val = close[j]
                if not np.isnan(val):
                    count += 1
                    delta = val - mean
                    mean += delta / count
                    delta2 = val - mean
                    m2 += delta * delta2
            if count > 1:
                variance = m2 / (count - 1)  # Sample std dev
                sd[i] = np.sqrt(max(0.0, variance)) * resp
            else:
                sd[i] = 0.0
        return sd
    return _jit(close, period, responsiveness)

@aot_guard("rolling_std_welford_parallel")
def _rolling_std_welford_parallel(close: np.ndarray, period: int, responsiveness: float) -> np.ndarray:
    from numba import njit, prange
    @njit(nogil=True, fastmath=True, cache=True, parallel=True)
    def _jit(close, period, responsiveness):
        n = len(close)
        sd = np.empty(n, dtype=np.float64)
        resp = max(0.0001, min(1.0, responsiveness))
        for i in prange(n):
            mean, m2, count = 0.0, 0.0, 0
            start = max(0, i - period + 1)
            for j in range(start, i + 1):
                val = close[j]
                if not np.isnan(val):
                    count += 1
                    delta = val - mean
                    mean += delta / count
                    delta2 = val - mean
                    m2 += delta * delta2
            if count > 1:
                variance = m2 / (count - 1)
                sd[i] = np.sqrt(max(0.0, variance)) * resp
            else:
                sd[i] = 0.0
        return sd
    return _jit(close, period, responsiveness)

# 16-17: ROLLING MEAN
@aot_guard("rolling_mean_numba")
def _rolling_mean_numba(close: np.ndarray, period: int) -> np.ndarray:
    from numba import njit
    @njit(nogil=True, fastmath=True, cache=True)
    def _jit(close, period):
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
    return _jit(close, period)

@aot_guard("rolling_mean_numba_parallel")
def _rolling_mean_numba_parallel(close: np.ndarray, period: int) -> np.ndarray:
    from numba import njit, prange
    @njit(nogil=True, fastmath=True, cache=True, parallel=True)
    def _jit(close, period):
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
    return _jit(close, period)

# 18-19: ROLLING MIN/MAX (Tuple return)
@aot_guard("rolling_min_max_numba")
def _rolling_min_max_numba(arr: np.ndarray, period: int) -> Tuple[np.ndarray, np.ndarray]:
    from numba import njit
    @njit(nogil=True, fastmath=True, cache=True)
    def _jit(arr, period):
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
    return _jit(arr, period)

@aot_guard("rolling_min_max_numba_parallel")
def _rolling_min_max_numba_parallel(arr: np.ndarray, period: int) -> Tuple[np.ndarray, np.ndarray]:
    from numba import njit, prange
    @njit(nogil=True, fastmath=True, cache=True, parallel=True)
    def _jit(arr, period):
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
    return _jit(arr, period)

# 20: PPO (Tuple return)
@aot_guard("calculate_ppo_core")
def _calculate_ppo_core(close: np.ndarray, fast: int, slow: int, signal: int) -> Tuple[np.ndarray, np.ndarray]:
    from numba import njit
    @njit(nogil=True, fastmath=True, cache=True)
    def _jit(close, fast, slow, signal):
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
    return _jit(close, fast, slow, signal)

# 21: RSI
@aot_guard("calculate_rsi_core")
def _calculate_rsi_core(close: np.ndarray, rsi_len: int) -> np.ndarray:
    from numba import njit
    @njit(nogil=True, fastmath=True, cache=True)
    def _jit(close, rsi_len):
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
    return _jit(close, rsi_len)

# 22-23: WICK CHECKS
@aot_guard("vectorized_wick_check_buy")
def _vectorized_wick_check_buy(open_arr: np.ndarray, high_arr: np.ndarray,
                               low_arr: np.ndarray, close_arr: np.ndarray,
                               min_wick_ratio: float) -> np.ndarray:
    from numba import njit
    @njit(nogil=True, fastmath=True, cache=True)
    def _jit(open_arr, high_arr, low_arr, close_arr, min_wick_ratio):
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
    return _jit(open_arr, high_arr, low_arr, close_arr, min_wick_ratio)

@aot_guard("vectorized_wick_check_sell")
def _vectorized_wick_check_sell(open_arr: np.ndarray, high_arr: np.ndarray,
                                low_arr: np.ndarray, close_arr: np.ndarray,
                                min_wick_ratio: float) -> np.ndarray:
    from numba import njit
    @njit(nogil=True, fastmath=True, cache=True)
    def _jit(open_arr, high_arr, low_arr, close_arr, min_wick_ratio):
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
    return _jit(open_arr, high_arr, low_arr, close_arr, min_wick_ratio)


# ============================================================================
# PUBLIC API: one-time init, silent summary, unified names
# ============================================================================

def summary_silent() -> dict:
    """Return AOT/JIT coverage summary without printing."""
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