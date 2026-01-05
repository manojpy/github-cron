#!/usr/bin/env python3
import os
import sys
import sysconfig
import multiprocessing
import platform 
from pathlib import Path
import numpy as np
from numba.pycc import CC
from numba import types

# Import ALL 26 functions from the shared module
from numba_functions_shared import (
    sanitize_array_numba, sanitize_array_numba_parallel,
    rolling_std_pine_accurate, rolling_std_pine_accurate_parallel,
    sma_pine_accurate, sma_pine_accurate_parallel,
    ema_loop, ema_loop_alpha,
    rng_filter_loop, smooth_range, calculate_trends_with_state, 
    kalman_loop, vwap_daily_loop,
    rolling_std_welford, rolling_std_welford_parallel,
    rolling_mean_numba, rolling_mean_numba_parallel,
    rolling_min_max_numba, rolling_min_max_numba_parallel,
    calculate_ppo_core, calculate_rsi_core,
    calc_mmh_worm_loop, calc_mmh_value_loop, calc_mmh_momentum_loop,
    vectorized_wick_check_buy, vectorized_wick_check_sell
)

def get_output_filename(base_name: str) -> str:
    ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')
    if not ext_suffix:
        system = platform.system()
        if system == 'Windows': ext_suffix = '.pyd'
        elif system == 'Darwin': ext_suffix = '.dylib'
        else: ext_suffix = '.so'
    return f"{base_name}{ext_suffix}"

def compile_module():
    output_dir = Path(__file__).parent
    module_name = 'macd_aot_compiled'
    cc = CC(module_name)
    cc.output_dir = str(output_dir)
    
    # Exporting all 26 Functions from numba_functions_shared.py
    # Sanitization
    cc.export('sanitize_array_numba', 'f8[:](f8[:], f8)')(sanitize_array_numba)
    cc.export('sanitize_array_numba_parallel', 'f8[:](f8[:], f8)')(sanitize_array_numba_parallel)
    
    # Pine-Accurate Functions
    cc.export('rolling_std_pine_accurate', 'f8[:](f8[:], i4, f8)')(rolling_std_pine_accurate)
    cc.export('rolling_std_pine_accurate_parallel', 'f8[:](f8[:], i4, f8)')(rolling_std_pine_accurate_parallel)
    cc.export('sma_pine_accurate', 'f8[:](f8[:], i4)')(sma_pine_accurate)
    cc.export('sma_pine_accurate_parallel', 'f8[:](f8[:], i4)')(sma_pine_accurate_parallel)
    
    # EMA & Filters
    cc.export('ema_loop', 'f8[:](f8[:], f8)')(ema_loop)
    cc.export('ema_loop_alpha', 'f8[:](f8[:], f8)')(ema_loop_alpha)
    cc.export('rng_filter_loop', 'f8[:](f8[:], f8[:])')(rng_filter_loop)
    cc.export('smooth_range', 'f8[:](f8[:], i4, i4)')(smooth_range)
    # FIX: Use Tuple((...)) for multiple returns and b1[:] for boolean arrays
    cc.export('calculate_trends_with_state', 'Tuple((b1[:], b1[:]))(f8[:], f8[:])')(calculate_trends_with_state)
    cc.export('kalman_loop', 'f8[:](f8[:], i4, f8, f8)')(kalman_loop)
    # FIX: vwap_daily_loop requires 5 arguments (high, low, close, volume, day_id)
    cc.export('vwap_daily_loop', 'f8[:](f8[:], f8[:], f8[:], f8[:], i8[:])')(vwap_daily_loop)
    
    # Statistical & Legacy Names
    cc.export('rolling_std_welford', 'f8[:](f8[:], i4, f8)')(rolling_std_welford)
    cc.export('rolling_std_welford_parallel', 'f8[:](f8[:], i4, f8)')(rolling_std_welford_parallel)
    cc.export('rolling_mean_numba', 'f8[:](f8[:], i4)')(rolling_mean_numba)
    cc.export('rolling_mean_numba_parallel', 'f8[:](f8[:], i4)')(rolling_mean_numba_parallel)
    
    # Min/Max (FIX: Tuple return)
    cc.export('rolling_min_max_numba', 'Tuple((f8[:], f8[:]))(f8[:], i4)')(rolling_min_max_numba)
    cc.export('rolling_min_max_numba_parallel', 'Tuple((f8[:], f8[:]))(f8[:], i4)')(rolling_min_max_numba_parallel)
    
    # Oscillators (FIX: Tuple return for PPO)
    cc.export('calculate_ppo_core', 'Tuple((f8[:], f8[:]))(f8[:], i4, i4, i4)')(calculate_ppo_core)
    cc.export('calculate_rsi_core', 'f8[:](f8[:], i4)')(calculate_rsi_core)
    
    # MMH Components
    cc.export('calc_mmh_worm_loop', 'f8[:](f8[:], f8[:], i4)')(calc_mmh_worm_loop)
    cc.export('calc_mmh_value_loop', 'f8[:](f8[:], i4)')(calc_mmh_value_loop)
    cc.export('calc_mmh_momentum_loop', 'f8[:](f8[:], i4)')(calc_mmh_momentum_loop)
    
    # Wick Checks (b1[:] = boolean array)
    cc.export('vectorized_wick_check_buy', 'b1[:](f8[:], f8[:], f8[:], f8[:], f8)')(vectorized_wick_check_buy)
    cc.export('vectorized_wick_check_sell', 'b1[:](f8[:], f8[:], f8[:], f8[:], f8)')(vectorized_wick_check_sell)

    print(f"🚀 Compiling {module_name} with {n_jobs} threads...")
    cc.compile()
    return True

if __name__ == '__main__':
    if compile_module():
        print("✅ Build completed successfully!")
        sys.exit(0)