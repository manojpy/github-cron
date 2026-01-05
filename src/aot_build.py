#!/usr/bin/env python3
"""
AOT Compilation Script - Refactored for 26 Shared Functions
============================================================
All implementations are sourced from numba_functions_shared.py.
Uses explicit Numba types for stability in GitHub Actions/Docker environments.
"""

import os
import sys
import sysconfig
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
    
    # --- Type Shortcuts ---
    f8 = types.float64
    f8_1d = types.float64[:]
    i4 = types.int32
    i8_1d = types.int64[:]
    b1_1d = types.boolean[:]

    # ========================================================================
    # 1. SANITIZATION
    # ========================================================================
    cc.export('sanitize_array_numba', f8_1d(f8_1d, f8))(sanitize_array_numba)
    cc.export('sanitize_array_numba_parallel', f8_1d(f8_1d, f8))(sanitize_array_numba_parallel)
    
    # ========================================================================
    # 2. PINE-ACCURATE MOVING AVERAGES
    # ========================================================================
    cc.export('rolling_std_pine_accurate', f8_1d(f8_1d, i4, f8))(rolling_std_pine_accurate)
    cc.export('rolling_std_pine_accurate_parallel', f8_1d(f8_1d, i4, f8))(rolling_std_pine_accurate_parallel)
    cc.export('sma_pine_accurate', f8_1d(f8_1d, i4))(sma_pine_accurate)
    cc.export('sma_pine_accurate_parallel', f8_1d(f8_1d, i4))(sma_pine_accurate_parallel)
    
    # ========================================================================
    # 3. EMA & FILTERS
    # ========================================================================
    cc.export('ema_loop', f8_1d(f8_1d, f8))(ema_loop)
    cc.export('ema_loop_alpha', f8_1d(f8_1d, f8))(ema_loop_alpha)
    cc.export('rng_filter_loop', f8_1d(f8_1d, f8_1d))(rng_filter_loop)
    cc.export('smooth_range', f8_1d(f8_1d, i4, i4))(smooth_range)
    cc.export('kalman_loop', f8_1d(f8_1d, i4, f8, f8))(kalman_loop)
    
    # FIX: Trends returns a Tuple of two boolean arrays
    cc.export('calculate_trends_with_state', 
              types.Tuple((b1_1d, b1_1d))(f8_1d, f8_1d))(calculate_trends_with_state)

    # ========================================================================
    # 4. MARKET INDICATORS (VWAP requires 5 arguments)
    # ========================================================================
    cc.export('vwap_daily_loop', f8_1d(f8_1d, f8_1d, f8_1d, f8_1d, i8_1d))(vwap_daily_loop)

    # ========================================================================
    # 5. STATISTICAL ALIASES & EXTREMES
    # ========================================================================
    cc.export('rolling_std_welford', f8_1d(f8_1d, i4, f8))(rolling_std_welford)
    cc.export('rolling_std_welford_parallel', f8_1d(f8_1d, i4, f8))(rolling_std_welford_parallel)
    cc.export('rolling_mean_numba', f8_1d(f8_1d, i4))(rolling_mean_numba)
    cc.export('rolling_mean_numba_parallel', f8_1d(f8_1d, i4))(rolling_mean_numba_parallel)
    
    # FIX: Min/Max returns a Tuple of two float arrays
    cc.export('rolling_min_max_numba', 
              types.Tuple((f8_1d, f8_1d))(f8_1d, i4))(rolling_min_max_numba)
    cc.export('rolling_min_max_numba_parallel', 
              types.Tuple((f8_1d, f8_1d))(f8_1d, i4))(rolling_min_max_numba_parallel)
    
    # ========================================================================
    # 6. OSCILLATORS
    # ========================================================================
    # FIX: PPO returns a Tuple (ppo_line, signal_line)
    cc.export('calculate_ppo_core', 
              types.Tuple((f8_1d, f8_1d))(f8_1d, i4, i4, i4))(calculate_ppo_core)
    cc.export('calculate_rsi_core', f8_1d(f8_1d, i4))(calculate_rsi_core)
    
    # ========================================================================
    # 7. MMH COMPONENTS
    # ========================================================================
    cc.export('calc_mmh_worm_loop', f8_1d(f8_1d, f8_1d, i4))(calc_mmh_worm_loop)
    cc.export('calc_mmh_value_loop', f8_1d(f8_1d, i4))(calc_mmh_value_loop)
    cc.export('calc_mmh_momentum_loop', f8_1d(f8_1d, i4))(calc_mmh_momentum_loop)
    
    # ========================================================================
    # 8. PATTERN RECOGNITION (Returns Boolean Array)
    # ========================================================================
    cc.export('vectorized_wick_check_buy', b1_1d(f8_1d, f8_1d, f8_1d, f8_1d, f8))(vectorized_wick_check_buy)
    cc.export('vectorized_wick_check_sell', b1_1d(f8_1d, f8_1d, f8_1d, f8_1d, f8))(vectorized_wick_check_sell)

    print("🔨 Compiling 26 functions into macd_aot_compiled...")
    try:
        cc.compile()
        return True
    except Exception as e:
        print(f"❌ AOT Compilation failed: {e}")
        return False

if __name__ == '__main__':
    if compile_module():
        print("✅ AOT Build Successful!")
        sys.exit(0)
    else:
        sys.exit(1)