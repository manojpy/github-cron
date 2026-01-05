"""
AOT Bridge Module - Refactored for 26 Shared Functions
"""

import importlib.util
import warnings
import pathlib
import sys
import sysconfig
import logging
import threading
import numpy as np
from typing import Tuple, Optional, Dict, Any

# Map JIT implementations from shared source (Matches updated names)
from numba_functions_shared import (
    sanitize_array_numba as _jit_sanitize_array_numba,
    sanitize_array_numba_parallel as _jit_sanitize_array_numba_parallel,
    rolling_std_pine_accurate as _jit_rolling_std_pine_accurate,
    rolling_std_pine_accurate_parallel as _jit_rolling_std_pine_accurate_parallel,
    sma_pine_accurate as _jit_sma_pine_accurate,
    sma_pine_accurate_parallel as _jit_sma_pine_accurate_parallel,
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

logger = logging.getLogger("aot_bridge")
_USING_AOT = False
_AOT_MODULE = None

# Initialize global function placeholders
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
    global _USING_AOT, _AOT_MODULE
    # Bindings for all 26 functions
    funcs = [
        ('sanitize_array_numba', _jit_sanitize_array_numba),
        ('sanitize_array_numba_parallel', _jit_sanitize_array_numba_parallel),
        ('ema_loop', _jit_ema_loop),
        ('ema_loop_alpha', _jit_ema_loop_alpha),
        ('rng_filter_loop', _jit_rng_filter_loop),
        ('smooth_range', _jit_smooth_range),
        ('calculate_trends_with_state', _jit_calculate_trends_with_state),
        ('kalman_loop', _jit_kalman_loop),
        ('vwap_daily_loop', _jit_vwap_daily_loop),
        ('rolling_std_welford', _jit_rolling_std_welford),
        ('rolling_std_welford_parallel', _jit_rolling_std_welford_parallel),
        ('calc_mmh_worm_loop', _jit_calc_mmh_worm_loop),
        ('calc_mmh_value_loop', _jit_calc_mmh_value_loop),
        ('calc_mmh_momentum_loop', _jit_calc_mmh_momentum_loop),
        ('rolling_mean_numba', _jit_rolling_mean_numba),
        ('rolling_mean_numba_parallel', _jit_rolling_mean_numba_parallel),
        ('rolling_min_max_numba', _jit_rolling_min_max_numba),
        ('rolling_min_max_numba_parallel', _jit_rolling_min_max_numba_parallel),
        ('calculate_ppo_core', _jit_calculate_ppo_core),
        ('calculate_rsi_core', _jit_calculate_rsi_core),
        ('vectorized_wick_check_buy', _jit_vectorized_wick_check_buy),
        ('vectorized_wick_check_sell', _jit_vectorized_wick_check_sell),
    ]
    
    # Special mappings for the renamed SMA functions
    globals()['sma_loop'] = getattr(_AOT_MODULE, 'sma_pine_accurate') if _USING_AOT else _jit_sma_pine_accurate
    globals()['sma_loop_parallel'] = getattr(_AOT_MODULE, 'sma_pine_accurate_parallel') if _USING_AOT else _jit_sma_pine_accurate_parallel

    for name, jit_func in funcs:
        globals()[name] = getattr(_AOT_MODULE, name) if (_USING_AOT and hasattr(_AOT_MODULE, name)) else jit_func

# Attempt AOT load
try:
    import macd_aot_compiled
    _AOT_MODULE = macd_aot_compiled
    _USING_AOT = True
    logger.info("🚀 AOT Bridge: Using compiled binaries")
except ImportError:
    logger.warning("🔄 AOT Bridge: Falling back to JIT")

_bind_functions()