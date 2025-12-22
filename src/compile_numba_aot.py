#!/usr/bin/env python3
"""
compile_numba_aot.py
Corrected version to fix NameErrors and Signature mismatches.
"""
import os
import sys
import time
import numpy as np
from pathlib import Path

# 1. Environment & Path Configuration
os.environ['NUMBA_CACHE_DIR'] = '/app/src/__pycache__'
os.environ['NUMBA_NUM_THREADS'] = '4'
os.environ['NUMBA_THREADING_LAYER'] = 'omp'

# Mock minimal config to bypass macd_unified's validation logic
os.environ['TELEGRAM_BOT_TOKEN'] = '000000:MOCK_TOKEN'
os.environ['TELEGRAM_CHAT_ID'] = '000000000'
os.environ['REDIS_URL'] = 'redis://localhost:6379'
os.environ['DELTA_API_BASE'] = 'https://api.delta.exchange'

# Ensure the app directory is in the path
sys.path.insert(0, '/app')

print("🔧 Starting Full Numba AOT compilation...")
start_time = time.time()

# 2. Import ALL Functions
print("📦 Importing Numba functions from macd_unified...")
try:
    from src.macd_unified import (
        _sanitize_array_numba, _sanitize_array_numba_parallel,
        _sma_loop, _sma_loop_parallel, _ema_loop, _kalman_loop,
        _vwap_daily_loop, _rng_filter_loop, _smooth_range,
        _calc_mmh_worm_loop, _calc_mmh_value_loop, _calc_mmh_momentum_loop,
        _rolling_std_welford, _rolling_std_welford_parallel,
        _rolling_mean_numba, _rolling_mean_numba_parallel,
        _rolling_min_max_numba, _rolling_min_max_numba_parallel,
        _calculate_ppo_core, _calculate_rsi_core,
        _vectorized_wick_check_buy, _vectorized_wick_check_sell
    )
except ImportError as e:
    print(f"❌ Critical Error: Could not import functions: {e}")
    sys.exit(1)

# 3. Setup Dummy Data for Compilation
size = 500
close_data = np.random.random(size).astype(np.float64)
high_data = (close_data + 0.05).astype(np.float64)
low_data = (close_data - 0.05).astype(np.float64)
open_data = (close_data - 0.01).astype(np.float64)
vol_data = np.random.random(size).astype(np.float64)
ts_data = np.linspace(1600000000, 1600086400, size).astype(np.int64)

# Define variables that were missing in your previous version
close_small = close_data[:size] # Ensure we have data for 'close_small'
close_large = close_data
open_arr = open_data
high_arr = high_data
low_arr = low_data

# 4. Compilation Registry
functions_to_compile = [
    ("_sanitize_array_numba", lambda: _sanitize_array_numba(close_data, 0.0)),
    ("_sanitize_array_numba_parallel", lambda: _sanitize_array_numba_parallel(close_data, 0.0)),
    ("_sma_loop", lambda: _sma_loop(close_data, 50)),
    ("_sma_loop_parallel", lambda: _sma_loop_parallel(close_data, 50)),
    ("_ema_loop", lambda: _ema_loop(close_data, 14)),
    ("_kalman_loop", lambda: _kalman_loop(close_data, 10)),
    ("_vwap_daily_loop", lambda: _vwap_daily_loop(high_data, low_data, close_data, vol_data, ts_data)),
    # FIXED: Only passing 2 arguments to _rng_filter_loop
    ("_rng_filter_loop", lambda: _rng_filter_loop(1.0, 1.0)), 
    ("_smooth_range", lambda: _smooth_range(close_data, 14, 2)),
    ("_calc_mmh_worm_loop", lambda: _calc_mmh_worm_loop(close_data, close_data, size)),
    ("_calc_mmh_value_loop", lambda: _calc_mmh_value_loop(close_data, size)),
    ("_calc_mmh_momentum_loop", lambda: _calc_mmh_momentum_loop(close_data, size)),
    ("_rolling_std_welford", lambda: _rolling_std_welford(close_data, 50, 0.5)),
    ("_rolling_std_welford_parallel", lambda: _rolling_std_welford_parallel(close_data, 50, 0.5)),
    ("_rolling_mean_numba", lambda: _rolling_mean_numba(close_data, 20)),
    ("_rolling_mean_numba_parallel", lambda: _rolling_mean_numba_parallel(close_data, 20)),
    ("_rolling_min_max_numba", lambda: _rolling_min_max_numba(close_data, 20)),
    ("_rolling_min_max_numba_parallel", lambda: _rolling_min_max_numba_parallel(close_data, 20)),
    ("_calculate_ppo_core", lambda: _calculate_ppo_core(close_data, 7, 16, 5)),
    ("_calculate_rsi_core", lambda: _calculate_rsi_core(close_small, 14)),
    ("_vectorized_wick_check_buy", lambda: _vectorized_wick_check_buy(open_arr, high_arr, low_arr, close_small, 0.2)),
    ("_vectorized_wick_check_sell", lambda: _vectorized_wick_check_sell(open_arr, high_arr, low_arr, close_large, 0.2))
]

# 5. Execution Loop
compiled_count = 0
failed_count = 0

for name, task in functions_to_compile:
    try:
        f_start = time.time()
        task()
        f_duration = time.time() - f_start
        print(f"  ✅ {name:<35} | {f_duration:.3f}s")
        compiled_count += 1
    except Exception as e:
        print(f"  ❌ {name:<35} | FAILED: {str(e)[:60]}")
        failed_count += 1

# 6. Summary Report
total_time = time.time() - start_time
print(f"\n{'='*70}\n🏁 Full AOT Compilation Summary\n{'='*70}")
print(f"✅ Successfully compiled: {compiled_count}/{len(functions_to_compile)}")
print(f"❌ Failed:                {failed_count}")
print(f"⏱️  Total Duration:        {total_time:.2f}s")
print(f"💾 Cache Location:        {os.environ['NUMBA_CACHE_DIR']}\n{'='*70}\n")

# Verify cache files
cache_path = Path(os.environ['NUMBA_CACHE_DIR'])
if cache_path.exists():
    nbi_count = len(list(cache_path.glob("*.nbi")))
    nbc_count = len(list(cache_path.glob("*.nbc")))
    print(f"📁 Cache verification: Found {nbi_count} index and {nbc_count} binary files.")
    if nbi_count == 0:
        print("⚠️  WARNING: No files saved. Ensure @njit(cache=True) is set in macd_unified.py")

sys.exit(0 if failed_count == 0 else 1)