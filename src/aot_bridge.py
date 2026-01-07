"""
AOT Bridge Module - Runtime AOT/JIT Function Dispatcher
========================================================

Provides transparent fallback between AOT-compiled (.so) and JIT-compiled
functions. Automatically detects AOT availability and falls back to JIT
if necessary.
"""

import os
import sys
import platform
import warnings
from pathlib import Path
from typing import Optional, Any, Tuple

import importlib.util
import numpy as np

# Global state
_aot_module: Optional[Any] = None
_using_aot: bool = False
_fallback_reason: Optional[str] = None
_initialized: bool = False


def get_library_extension() -> str:
    """Get platform-specific shared library extension"""
    system = platform.system()
    if system == "Linux":
        return ".so"
    elif system == "Darwin":
        return ".dylib"
    elif system == "Windows":
        return ".dll"
    else:
        return ".so"  # Fallback


def find_aot_library(module_name: str = "macd_aot_compiled") -> Optional[Path]:
   
    extension = get_library_extension()

    # Accept both normalized and ABI-suffixed filenames
    candidates = [
        f"{module_name}{extension}",
        f"{module_name}.cpython-311-x86_64-linux-gnu{extension}",
        f"{module_name}.cpython-311{extension}",
    ]

    search_paths = []
    env_path = os.getenv("AOT_LIB_PATH")
    if env_path:
        search_paths.append(Path(env_path))
    search_paths += [Path.cwd(), Path(__file__).parent]

    for search_dir in search_paths:
        for name in candidates:
            p = search_dir / name
            if p.exists():
                return p

        # Wildcard fallback in case of different ABI tag
        found = list(search_dir.glob(f"{module_name}*{extension}"))
        if found:
            return found[0]

    return None


def load_aot_module(library_path: Path, module_name: str = "macd_aot_compiled") -> Optional[Any]:
    
    try:
        spec = importlib.util.spec_from_file_location(module_name, str(library_path))
        if spec is None or spec.loader is None:
            warnings.warn(f"Cannot create import spec for {library_path}")
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        
        sys.modules[module_name] = mod
        return mod
    except Exception as e:
        warnings.warn(f"Failed to import AOT module {library_path}: {e}")
        return None


def initialize_aot(module_name: str = "macd_aot_compiled") -> Tuple[bool, Optional[str]]:
   
    global _aot_module, _using_aot, _fallback_reason

    library_path = find_aot_library(module_name)
    if library_path is None:
        return False, f"AOT library {module_name}{get_library_extension()} not found"

    _aot_module = load_aot_module(library_path, module_name)
    if _aot_module is None:
        return False, f"Failed to import AOT module at {library_path}"

    # Verify critical functions exist as Python-callable attributes
    critical_functions = [
        'sanitize_array_numba',
        'ema_loop',
        'calculate_ppo_core',
    ]

    missing = [fn for fn in critical_functions if not hasattr(_aot_module, fn)]
    if missing:
        return False, f"AOT library missing critical function: {missing[0]}"

    _using_aot = True
    return True, None


def initialize_jit_fallback() -> None:
   
    global _fallback_reason

    try:
        from numba_functions_shared import (
            sanitize_array_numba,
            sanitize_array_numba_parallel,
            ema_loop,
            ema_loop_alpha,
            kalman_loop,
            vwap_daily_loop,
            rng_filter_loop,
            smooth_range,
            calculate_trends_with_state,
            calc_mmh_worm_loop,
            calc_mmh_value_loop,
            calc_mmh_momentum_loop,
            rolling_std,
            rolling_std_parallel,
            rolling_mean_numba,
            rolling_mean_numba_parallel,
            rolling_min_max_numba,
            rolling_min_max_numba_parallel,
            calculate_ppo_core,
            calculate_rsi_core,
            vectorized_wick_check_buy,
            vectorized_wick_check_sell,
        )

        globals()['_jit_sanitize_array_numba'] = sanitize_array_numba
        globals()['_jit_sanitize_array_numba_parallel'] = sanitize_array_numba_parallel
        globals()['_jit_ema_loop'] = ema_loop
        globals()['_jit_ema_loop_alpha'] = ema_loop_alpha
        globals()['_jit_kalman_loop'] = kalman_loop
        globals()['_jit_vwap_daily_loop'] = vwap_daily_loop
        globals()['_jit_rng_filter_loop'] = rng_filter_loop
        globals()['_jit_smooth_range'] = smooth_range
        globals()['_jit_calculate_trends_with_state'] = calculate_trends_with_state
        globals()['_jit_calc_mmh_worm_loop'] = calc_mmh_worm_loop
        globals()['_jit_calc_mmh_value_loop'] = calc_mmh_value_loop
        globals()['_jit_calc_mmh_momentum_loop'] = calc_mmh_momentum_loop
        globals()['_jit_rolling_std'] = rolling_std
        globals()['_jit_rolling_std_parallel'] = rolling_std_parallel
        globals()['_jit_rolling_mean_numba'] = rolling_mean_numba
        globals()['_jit_rolling_mean_numba_parallel'] = rolling_mean_numba_parallel
        globals()['_jit_rolling_min_max_numba'] = rolling_min_max_numba
        globals()['_jit_rolling_min_max_numba_parallel'] = rolling_min_max_numba_parallel
        globals()['_jit_calculate_ppo_core'] = calculate_ppo_core
        globals()['_jit_calculate_rsi_core'] = calculate_rsi_core
        globals()['_jit_vectorized_wick_check_buy'] = vectorized_wick_check_buy
        globals()['_jit_vectorized_wick_check_sell'] = vectorized_wick_check_sell

    except ImportError as e:
        _fallback_reason = f"JIT fallback failed: {e}"
        raise RuntimeError(f"Cannot initialize JIT fallback: {e}")


def ensure_initialized() -> None:
    global _initialized, _fallback_reason, _using_aot

    if _initialized:
        return

    success, reason = initialize_aot()

    if success:
        _using_aot = True
        _fallback_reason = None
    else:
        _fallback_reason = reason
        _using_aot = False
        initialize_jit_fallback()

    _initialized = True


def is_using_aot() -> bool:
    return _using_aot


def get_fallback_reason() -> Optional[str]:
    return _fallback_reason


def requires_warmup() -> bool:
   
    return not _using_aot


# ============================================================================
# FUNCTION WRAPPERS - Automatic AOT/JIT dispatch
# ============================================================================

def sanitize_array_numba(arr: np.ndarray, default: float) -> np.ndarray:
    if _using_aot:
        return _aot_module.sanitize_array_numba(arr, default)
    return _jit_sanitize_array_numba(arr, default)


def sanitize_array_numba_parallel(arr: np.ndarray, default: float) -> np.ndarray:
    if _using_aot:
        return _aot_module.sanitize_array_numba_parallel(arr, default)
    return _jit_sanitize_array_numba_parallel(arr, default)


def ema_loop(data: np.ndarray, alpha_or_period: float) -> np.ndarray:
    if _using_aot:
        return _aot_module.ema_loop(data, alpha_or_period)
    return _jit_ema_loop(data, alpha_or_period)


def ema_loop_alpha(data: np.ndarray, alpha: float) -> np.ndarray:
    if _using_aot:
        return _aot_module.ema_loop_alpha(data, alpha)
    return _jit_ema_loop_alpha(data, alpha)


def kalman_loop(src: np.ndarray, length: int, R: float, Q: float) -> np.ndarray:
    if _using_aot:
        return _aot_module.kalman_loop(src, length, R, Q)
    return _jit_kalman_loop(src, length, R, Q)


def vwap_daily_loop(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    day_id: np.ndarray,
) -> np.ndarray:
    if _using_aot:
        return _aot_module.vwap_daily_loop(high, low, close, volume, day_id)
    return _jit_vwap_daily_loop(high, low, close, volume, day_id)


def rng_filter_loop(x: np.ndarray, r: np.ndarray) -> np.ndarray:
    if _using_aot:
        return _aot_module.rng_filter_loop(x, r)
    return _jit_rng_filter_loop(x, r)


def smooth_range(close: np.ndarray, t: int, m: int) -> np.ndarray:
    if _using_aot:
        return _aot_module.smooth_range(close, t, m)
    return _jit_smooth_range(close, t, m)


def calculate_trends_with_state(
    filt_x1: np.ndarray,
    filt_x12: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    if _using_aot:
        return _aot_module.calculate_trends_with_state(filt_x1, filt_x12)
    return _jit_calculate_trends_with_state(filt_x1, filt_x12)


def calc_mmh_worm_loop(close_arr: np.ndarray, sd_arr: np.ndarray, rows: int) -> np.ndarray:
    if _using_aot:
        return _aot_module.calc_mmh_worm_loop(close_arr, sd_arr, rows)
    return _jit_calc_mmh_worm_loop(close_arr, sd_arr, rows)


def calc_mmh_value_loop(temp_arr: np.ndarray, rows: int) -> np.ndarray:
    if _using_aot:
        return _aot_module.calc_mmh_value_loop(temp_arr, rows)
    return _jit_calc_mmh_value_loop(temp_arr, rows)


def calc_mmh_momentum_loop(momentum_arr: np.ndarray, rows: int) -> np.ndarray:
    if _using_aot:
        return _aot_module.calc_mmh_momentum_loop(momentum_arr, rows)
    return _jit_calc_mmh_momentum_loop(momentum_arr, rows)


def rolling_std(close: np.ndarray, period: int, responsiveness: float) -> np.ndarray:
    if _using_aot:
        return _aot_module.rolling_std(close, period, responsiveness)
    return _jit_rolling_std(close, period, responsiveness)


def rolling_std_parallel(close: np.ndarray, period: int, responsiveness: float) -> np.ndarray:
    if _using_aot:
        return _aot_module.rolling_std_parallel(close, period, responsiveness)
    return _jit_rolling_std_parallel(close, period, responsiveness)


def rolling_mean_numba(data: np.ndarray, period: int) -> np.ndarray:
    if _using_aot:
        return _aot_module.rolling_mean_numba(data, period)
    return _jit_rolling_mean_numba(data, period)


def rolling_mean_numba_parallel(data: np.ndarray, period: int) -> np.ndarray:
    if _using_aot:
        return _aot_module.rolling_mean_numba_parallel(data, period)
    return _jit_rolling_mean_numba_parallel(data, period)


def rolling_min_max_numba(arr: np.ndarray, period: int) -> Tuple[np.ndarray, np.ndarray]:
    if _using_aot:
        return _aot_module.rolling_min_max_numba(arr, period)
    return _jit_rolling_min_max_numba(arr, period)


def rolling_min_max_numba_parallel(arr: np.ndarray, period: int) -> Tuple[np.ndarray, np.ndarray]:
    if _using_aot:
        return _aot_module.rolling_min_max_numba_parallel(arr, period)
    return _jit_rolling_min_max_numba_parallel(arr, period)


def calculate_ppo_core(
    close: np.ndarray,
    fast: int,
    slow: int,
    signal: int
) -> Tuple[np.ndarray, np.ndarray]:
    if _using_aot:
        return _aot_module.calculate_ppo_core(close, fast, slow, signal)
    return _jit_calculate_ppo_core(close, fast, slow, signal)


def calculate_rsi_core(close: np.ndarray, period: int) -> np.ndarray:
    if _using_aot:
        return _aot_module.calculate_rsi_core(close, period)
    return _jit_calculate_rsi_core(close, period)


def vectorized_wick_check_buy(
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    min_wick_ratio: float
) -> np.ndarray:
    if _using_aot:
        return _aot_module.vectorized_wick_check_buy(
            open_arr, high_arr, low_arr, close_arr, min_wick_ratio
        )
    return _jit_vectorized_wick_check_buy(
        open_arr, high_arr, low_arr, close_arr, min_wick_ratio
    )


def vectorized_wick_check_sell(
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    min_wick_ratio: float
) -> np.ndarray:
    if _using_aot:
        return _aot_module.vectorized_wick_check_sell(
            open_arr, high_arr, low_arr, close_arr, min_wick_ratio
        )
    return _jit_vectorized_wick_check_sell(
        open_arr, high_arr, low_arr, close_arr, min_wick_ratio
    )


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Initialization
    'ensure_initialized',
    'is_using_aot',
    'get_fallback_reason',
    'requires_warmup',

    # Sanitization
    'sanitize_array_numba',
    'sanitize_array_numba_parallel',

    # Moving Averages
    'ema_loop',
    'ema_loop_alpha',

    # Filters
    'kalman_loop',
    'rng_filter_loop',
    'smooth_range',
    'calculate_trends_with_state',

    # Market Indicators
    'vwap_daily_loop',

    # Statistical
    'rolling_std',
    'rolling_std_parallel',
    'rolling_mean_numba',
    'rolling_mean_numba_parallel',
    'rolling_min_max_numba',
    'rolling_min_max_numba_parallel',

    # Oscillators
    'calculate_ppo_core',
    'calculate_rsi_core',

    # MMH Components
    'calc_mmh_worm_loop',
    'calc_mmh_value_loop',
    'calc_mmh_momentum_loop',

    # Pattern Recognition
    'vectorized_wick_check_buy',
    'vectorized_wick_check_sell',
]


# Auto-initialize on import (optional - can also call ensure_initialized() manually)
try:
    ensure_initialized()
except Exception as e:
    warnings.warn(f"Auto-initialization failed: {e}. Call ensure_initialized() manually.")
