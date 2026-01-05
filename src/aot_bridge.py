import importlib.util
import pathlib
import sys
import sysconfig
import logging
import numpy as np
import numba_functions_shared as shared

logger = logging.getLogger("aot_bridge")
_USING_AOT = False
_AOT_MODULE = None

# Function placeholders
sanitize_array_numba = None
sanitize_array_numba_parallel = None
rolling_std_pine_accurate = None
rolling_std_pine_accurate_parallel = None
sma_pine_accurate = None
sma_pine_accurate_parallel = None
ema_loop = None
ema_loop_alpha = None
rng_filter_loop = None
smooth_range = None
calculate_trends_with_state = None
kalman_loop = None
vwap_daily_loop = None
rolling_std_welford = None
rolling_std_welford_parallel = None
rolling_mean_numba = None
rolling_mean_numba_parallel = None
rolling_min_max_numba = None
rolling_min_max_numba_parallel = None
calculate_ppo_core = None
calculate_rsi_core = None
calc_mmh_worm_loop = None
calc_mmh_value_loop = None
calc_mmh_momentum_loop = None
vectorized_wick_check_buy = None
vectorized_wick_check_sell = None

def _bind_functions():
    global _USING_AOT, _AOT_MODULE
    funcs = [
        'sanitize_array_numba', 'sanitize_array_numba_parallel',
        'rolling_std_pine_accurate', 'rolling_std_pine_accurate_parallel',
        'sma_pine_accurate', 'sma_pine_accurate_parallel',
        'ema_loop', 'ema_loop_alpha', 'rng_filter_loop', 'smooth_range',
        'calculate_trends_with_state', 'kalman_loop', 'vwap_daily_loop',
        'rolling_std_welford', 'rolling_std_welford_parallel',
        'rolling_mean_numba', 'rolling_mean_numba_parallel',
        'rolling_min_max_numba', 'rolling_min_max_numba_parallel',
        'calculate_ppo_core', 'calculate_rsi_core',
        'calc_mmh_worm_loop', 'calc_mmh_value_loop', 'calc_mmh_momentum_loop',
        'vectorized_wick_check_buy', 'vectorized_wick_check_sell'
    ]

    # Try Loading AOT
    try:
        import macd_aot_compiled
        _AOT_MODULE = macd_aot_compiled
        _USING_AOT = True
    except ImportError:
        logger.warning("AOT module not found, falling back to JIT")

    # Map functions
    for name in funcs:
        if _USING_AOT and hasattr(_AOT_MODULE, name):
            globals()[name] = getattr(_AOT_MODULE, name)
        else:
            globals()[name] = getattr(shared, name)

_bind_functions()

def is_using_aot(): return _USING_AOT