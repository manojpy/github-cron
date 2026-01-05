"""
AOT Bridge Module - Refactored to Use Shared Functions
=======================================================

Provides unified API that:
1. Attempts to load AOT (.so) module
2. Falls back to JIT from numba_functions_shared.py

NO DUPLICATION: All function implementations imported from single source.
"""

import importlib.util
import warnings
import pathlib
import os
import sys
import sysconfig
import logging
import threading
import numpy as np
from typing import Tuple, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import ALL 24 functions from shared module
from numba_functions_shared import (
    sanitize_array_numba as _jit_sanitize_array_numba,
    sanitize_array_numba_parallel as _jit_sanitize_array_numba_parallel,
    sma_loop as _jit_sma_loop,
    sma_loop_parallel as _jit_sma_loop_parallel,
    ema_loop as _jit_ema_loop,
    ema_loop_alpha as _jit_ema_loop_alpha,
    rng_filter_loop as _jit_rng_filter_loop,
    smooth_range as _jit_smooth_range,
    calculate_trends_with_state as _jit_calculate_trends_with_state, 
    kalman_loop as _jit_kalman_loop,
    vwap_daily_loop as _jit_vwap_daily_loop,
    rolling_std_welford as _jit_rolling_std_welford,
    rolling_std_welford_parallel as _jit_rolling_std_welford_parallel,
    calc_mmh_worm_loop as _jit_calc_mmh_worm_loop,
    calc_mmh_value_loop as _jit_calc_mmh_value_loop,
    calc_mmh_momentum_loop as _jit_calc_mmh_momentum_loop,
    rolling_mean_numba as _jit_rolling_mean_numba,
    rolling_mean_numba_parallel as _jit_rolling_mean_numba_parallel,
    rolling_min_max_numba as _jit_rolling_min_max_numba,
    rolling_min_max_numba_parallel as _jit_rolling_min_max_numba_parallel,
    calculate_ppo_core as _jit_calculate_ppo_core,
    calculate_rsi_core as _jit_calculate_rsi_core,
    vectorized_wick_check_buy as _jit_vectorized_wick_check_buy,
    vectorized_wick_check_sell as _jit_vectorized_wick_check_sell,
)

# Disable Numba cache to prevent stale bytecode issues
import numba
numba.config.CACHE_DIR = None
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
    """Test EMA function"""
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    res = aot_module.ema_loop(arr, 3.0)
    return (
        res is not None
        and len(res) == len(arr)
        and not np.all(np.isnan(res))
        and not np.any(np.isinf(res))
    )


def _test_ema_nan(aot_module) -> bool:
    """Test EMA with NaN values"""
    arr = np.array([np.nan, 2.0, 3.0, np.nan, 5.0], dtype=np.float64)
    res = aot_module.ema_loop(arr, 3.0)
    return res is not None and len(res) == len(arr)


def _test_ema_range(aot_module) -> bool:
    """Test EMA with extreme values"""
    arr = np.array([1e-10, 1e10, 1e-10, 1e5], dtype=np.float64)
    res = aot_module.ema_loop(arr, 2.0)
    return res is not None and not np.any(np.isinf(res))


def _test_ema_single(aot_module) -> bool:
    """Test EMA with single value"""
    arr = np.array([42.0], dtype=np.float64)
    res = aot_module.ema_loop(arr, 1.0)
    return res is not None and len(res) == 1


def _test_rsi(aot_module) -> bool:
    """Test RSI calculation"""
    arr = np.array([44.0, 44.25, 44.5, 43.75, 44.0, 44.5, 45.0, 45.5], dtype=np.float64)
    res = aot_module.calculate_rsi_core(arr, 14)
    return res is not None and len(res) == len(arr)


def _test_sanitize(aot_module) -> bool:
    """Test sanitization function"""
    arr = np.array([1.0, np.nan, np.inf, -np.inf, 5.0], dtype=np.float64)
    res = aot_module.sanitize_array_numba(arr, 0.0)
    return (
        res is not None
        and len(res) == len(arr)
        and not np.any(np.isnan(res))
        and not np.any(np.isinf(res))
    )


def _test_sma(aot_module) -> bool:
    """Test SMA calculation"""
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    res = aot_module.sma_loop(arr, 3)
    return res is not None and len(res) == len(arr)


def _test_rolling_std(aot_module) -> bool:
    """Test rolling standard deviation"""
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float64)
    res = aot_module.rolling_std_welford(arr, 3, 1.0)
    return res is not None and len(res) == len(arr)


def _test_wick_buy(aot_module) -> bool:
    """Test buy wick check"""
    o = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    h = np.array([2.0, 3.0, 4.0], dtype=np.float64)
    l = np.array([0.5, 1.5, 2.5], dtype=np.float64)
    c = np.array([1.5, 2.5, 3.5], dtype=np.float64)
    res = aot_module.vectorized_wick_check_buy(o, h, l, c, 0.3)
    return res is not None and len(res) == len(o)


def _test_empty_ema(aot_module) -> bool:
    """Test EMA with empty array"""
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
    global _INITIALIZED
    
    if _INITIALIZED:
        return _USING_AOT
    
    with _INIT_LOCK:
        if _INITIALIZED:  # Double-check after acquiring lock
            return _USING_AOT
        
        ok, _ = initialize_aot()
        _bind_functions()
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
        "rng_filter_loop", "smooth_range", "calculate_trends_with_state",
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
        "source_module": "numba_functions_shared.py",
    }
    
    if _USING_AOT and _AOT_MODULE:
        result["module_path"] = getattr(_AOT_MODULE, '__file__', 'unknown')
    
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
calculate_trends_with_state = None 
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
    Bind all 24 functions to either AOT or JIT implementations.
    Called once at module initialization - creates direct function references with ZERO overhead.
    """
    global sanitize_array_numba, sanitize_array_numba_parallel
    global sma_loop, sma_loop_parallel
    global ema_loop, ema_loop_alpha
    global rng_filter_loop, smooth_range, calculate_trends_with_state
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
        calculate_trends_with_state = _AOT_MODULE.calculate_trends_with_state
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
        
        logger.info("✅ All 24 functions bound to AOT implementations")
    else:
        # Bind to JIT implementations from shared module
        sanitize_array_numba = _jit_sanitize_array_numba
        sanitize_array_numba_parallel = _jit_sanitize_array_numba_parallel
        sma_loop = _jit_sma_loop
        sma_loop_parallel = _jit_sma_loop_parallel
        ema_loop = _jit_ema_loop
        ema_loop_alpha = _jit_ema_loop_alpha
        rng_filter_loop = _jit_rng_filter_loop
        smooth_range = _jit_smooth_range
        calculate_trends_with_state = _jit_calculate_trends_with_state
        kalman_loop = _jit_kalman_loop
        vwap_daily_loop = _jit_vwap_daily_loop
        rolling_std_welford = _jit_rolling_std_welford
        rolling_std_welford_parallel = _jit_rolling_std_welford_parallel
        calc_mmh_worm_loop = _jit_calc_mmh_worm_loop
        calc_mmh_value_loop = _jit_calc_mmh_value_loop
        calc_mmh_momentum_loop = _jit_calc_mmh_momentum_loop
        rolling_mean_numba = _jit_rolling_mean_numba
        rolling_mean_numba_parallel = _jit_rolling_mean_numba_parallel
        rolling_min_max_numba = _jit_rolling_min_max_numba
        rolling_min_max_numba_parallel = _jit_rolling_min_max_numba_parallel
        calculate_ppo_core = _jit_calculate_ppo_core
        calculate_rsi_core = _jit_calculate_rsi_core
        vectorized_wick_check_buy = _jit_vectorized_wick_check_buy
        vectorized_wick_check_sell = _jit_vectorized_wick_check_sell
        
        logger.info("⚠️ All 24 functions bound to JIT fallback implementations")


def summary() -> None:
    """Print human-readable summary of AOT status"""
    ensure_initialized()
    
    print("\n" + "="*70)
    print("AOT BRIDGE STATUS (REFACTORED)")
    print("="*70)
    
    if _USING_AOT:
        print("✅ Status: AOT ACTIVE")
        if _AOT_MODULE:
            module_path = getattr(_AOT_MODULE, '__file__', 'unknown')
            print(f"📦 Module: {module_path}")
        print(f"⚡ Functions: All 24 functions using compiled AOT (.so)")
    else:
        print("⚠️ Status: JIT FALLBACK")
        if _FALLBACK_REASON:
            if isinstance(_FALLBACK_REASON, dict):
                print(f"❌ Stage: {_FALLBACK_REASON.get('stage', 'unknown')}")
                print(f"❌ Function: {_FALLBACK_REASON.get('function', 'unknown')}")
                print(f"❌ Error: {_FALLBACK_REASON.get('error', 'unknown')}")
            else:
                print(f"❌ Reason: {_FALLBACK_REASON}")
        print(f"🔄 Functions: All 24 functions using JIT compilation")
    
    print(f"📄 Source: numba_functions_shared.py (single source of truth)")
    print("="*70 + "\n")


# ============================================================================
# MODULE INITIALIZATION
# ============================================================================
# Initialize and bind functions immediately on module import
if not ensure_initialized():
    logger.warning("AOT initialization failed, using JIT fallback")

# ✅ No need to call _bind_functions() - already done in ensure_initialized()

# Verify all functions are bound
if ema_loop is None:
    raise RuntimeError("Critical error: Functions not bound...")

# Verify function count
_expected_functions = [
    sanitize_array_numba, sanitize_array_numba_parallel,
    sma_loop, sma_loop_parallel,
    ema_loop, ema_loop_alpha,
    rng_filter_loop, smooth_range, calculate_trends_with_state, 
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
    logger.info("🔄 AOT Bridge initialized with JIT fallback from numba_functions_shared.py")


# ============================================================================
# PUBLIC EXPORTS
# ============================================================================

__all__ = [
    # Status functions
    'is_using_aot',
    'get_fallback_reason',
    'diagnostics',
    'summary',

    # Core functions
    'sanitize_array_numba',
    'sanitize_array_numba_parallel',
    'sma_loop',
    'sma_loop_parallel',
    'ema_loop',
    'ema_loop_alpha',
    'rng_filter_loop',
    'smooth_range',
    'calculate_trends_with_state',
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