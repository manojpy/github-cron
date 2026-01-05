#!/usr/bin/env python3
"""
AOT Compilation Script - Refactored to Use Shared Functions
============================================================
All 26 functions are exported using explicit Numba types for stability.
"""

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

def compile_module():
    output_dir = Path(__file__).parent
    module_name = 'macd_aot_compiled'
    cc = CC(module_name)
    cc.output_dir = str(output_dir)
    
    # --- Sanitization (1-2) ---
    cc.export('sanitize_array_numba', 'f8[:](f8[:], f8)')(sanitize_array_numba)
    cc.export('sanitize_array_numba_parallel', 'f8[:](f8[:], f8)')(sanitize_array_numba_parallel)
    
    # --- Pine-Accurate Functions (3-6) ---
    cc.export('rolling_std_pine_accurate', 'f8[:](f8[:], i4, f8)')(rolling_std_pine_accurate)
    cc.export('rolling_std_pine_accurate_parallel', 'f8[:](f8[:], i4, f8)')(rolling_std_pine_accurate_parallel)
    cc.export('sma_pine_accurate', 'f8[:](f8[:], i4)')(sma_pine_accurate)
    cc.export('sma_pine_accurate_parallel', 'f8[:](f8[:], i4)')(sma_pine_accurate_parallel)
    
    # --- EMA & Filters (7-12) ---
    cc.export('ema_loop', 'f8[:](f8[:], f8)')(ema_loop)
    cc.export('ema_loop_alpha', 'f8[:](f8[:], f8)')(ema_loop_alpha)
    cc.export('rng_filter_loop', 'f8[:](f8[:], f8[:])')(rng_filter_loop)
    cc.export('smooth_range', 'f8[:](f8[:], i4, i4)')(smooth_range)
    cc.export('kalman_loop', 'f8[:](f8[:], i4, f8, f8)')(kalman_loop)
    # Fix: Tuple of boolean arrays for trends
    cc.export('calculate_trends_with_state', 
              types.Tuple((types.b1[:], types.b1[:]))(types.f8[:], types.f8[:]))(calculate_trends_with_state)

    # --- Market Indicators (13) ---
    cc.export('vwap_daily_loop', 'f8[:](f8[:], f8[:], f8[:], f8[:], i8[:])')(vwap_daily_loop)

    # --- Statistical Aliases (14-19) ---
    cc.export('rolling_std_welford', 'f8[:](f8[:], i4, f8)')(rolling_std_welford)
    cc.export('rolling_std_welford_parallel', 'f8[:](f8[:], i4, f8)')(rolling_std_welford_parallel)
    cc.export('rolling_mean_numba', 'f8[:](f8[:], i4)')(rolling_mean_numba)
    cc.export('rolling_mean_numba_parallel', 'f8[:](f8[:], i4)')(rolling_mean_numba_parallel)
    # Fix: Tuple of float arrays for Min/Max
    cc.export('rolling_min_max_numba', 
              types.Tuple((types.f8[:], types.f8[:]))(types.f8[:], types.i4))(rolling_min_max_numba)
    cc.export('rolling_min_max_numba_parallel', 
              types.Tuple((types.f8[:], types.f8[:]))(types.f8[:], types.i4))(rolling_min_max_numba_parallel)
    
    # --- Oscillators (20-21) ---
    # Fix: Tuple of float arrays for PPO
    cc.export('calculate_ppo_core', 
              types.Tuple((types.f8[:], types.f8[:]))(types.f8[:], types.i4, types.i4, types.i4))(calculate_ppo_core)
    cc.export('calculate_rsi_core', 'f8[:](f8[:], i4)')(calculate_rsi_core)
    
    # --- MMH Components (22-24) ---
    cc.export('calc_mmh_worm_loop', 'f8[:](f8[:], f8[:], i4)')(calc_mmh_worm_loop)
    cc.export('calc_mmh_value_loop', 'f8[:](f8[:], i4)')(calc_mmh_value_loop)
    cc.export('calc_mmh_momentum_loop', 'f8[:](f8[:], i4)')(calc_mmh_momentum_loop)
    
    # --- Patterns (25-26) ---
    cc.export('vectorized_wick_check_buy', 'b1[:](f8[:], f8[:], f8[:], f8[:], f8)')(vectorized_wick_check_buy)
    cc.export('vectorized_wick_check_sell', 'b1[:](f8[:], f8[:], f8[:], f8[:], f8)')(vectorized_wick_check_sell)

    cc.compile()
    return True

if __name__ == '__main__':
    if compile_module():
        print("✅ Build successful!")
        sys.exit(0)
    sys.exit(1)