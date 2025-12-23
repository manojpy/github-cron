#!/usr/bin/env python3
"""
AOT (Ahead-of-Time) Compilation Script for Numba Functions
Standalone script that compiles Numba functions without loading full config
"""
import os
import sys
import time
import numpy as np
from pathlib import Path

# Set cache directory and Numba settings
os.environ['NUMBA_CACHE_DIR'] = '/app/src/__pycache__'
os.environ['NUMBA_NUM_THREADS'] = '4'
os.environ['NUMBA_THREADING_LAYER'] = 'omp'

# Mock minimal config to bypass validation
os.environ['TELEGRAM_BOT_TOKEN'] = '000000:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'
os.environ['TELEGRAM_CHAT_ID'] = '000000000'
os.environ['REDIS_URL'] = 'redis://localhost:6379'
os.environ['DELTA_API_BASE'] = 'https://api.delta.exchange'

print("🔧 Starting Numba AOT compilation...")
print(f"📂 Cache directory: {os.environ['NUMBA_CACHE_DIR']}")
print(f"🧵 Threads: {os.environ['NUMBA_NUM_THREADS']}")
start_time = time.time()

# Import only the Numba functions we need to compile
sys.path.insert(0, '/app')

print("📦 Importing Numba functions...")
try:
    from src.macd_unified import (
        _sanitize_array_numba,
        _sanitize_array_numba_parallel,
        _sma_loop,
        _sma_loop_parallel,
        _ema_loop,
        _ema_loop_alpha,
        _kalman_loop,
        _vwap_daily_loop,
        _rng_filter_loop,
        _smooth_range,
        _calc_mmh_worm_loop,
        _calc_mmh_value_loop,
        _calc_mmh_momentum_loop,
        _rolling_std_welford,
        _rolling_std_welford_parallel,
        _rolling_mean_numba,
        _rolling_mean_numba_parallel,
        _rolling_min_max_numba,
        _rolling_min_max_numba_parallel,
        _calculate_ppo_core,
        _calculate_rsi_core,
        _vectorized_wick_check_buy,
        _vectorized_wick_check_sell
    )

    print("✅ Module imported successfully")
except Exception as e:
    print(f"❌ Failed to import functions: {e}")
    sys.exit(1)

# Realistic data sizes for 15m/5m candles
SIZE_SMALL = 200   # For 15m data
SIZE_LARGE = 400   # For 5m data

# Generate test data
print("📊 Generating test data...")
close_small = np.random.random(SIZE_SMALL).astype(np.float64) * 50000 + 30000
close_large = np.random.random(SIZE_LARGE).astype(np.float64) * 50000 + 30000
open_arr = close_large + np.random.random(SIZE_LARGE).astype(np.float64) * 100
high_arr = np.maximum(open_arr, close_large) + np.random.random(SIZE_LARGE).astype(np.float64) * 50
low_arr = np.minimum(open_arr, close_large) - np.random.random(SIZE_LARGE).astype(np.float64) * 50
volume = np.random.random(SIZE_LARGE).astype(np.float64) * 1000000
timestamps = np.arange(SIZE_LARGE, dtype=np.int64) * 900

# Define functions to compile
functions_to_compile = [
    # Array sanitization (both serial and parallel)
    ("_sanitize_array_numba", lambda: _sanitize_array_numba(close_small, 0.0)),
    ("_sanitize_array_numba_parallel", lambda: _sanitize_array_numba_parallel(close_large, 0.0)),
    
    # Moving averages (both serial and parallel)
    ("_sma_loop", lambda: _sma_loop(close_small, 50)),
    ("_sma_loop_parallel", lambda: _sma_loop_parallel(close_large, 50)),
    ("_ema_loop (period)", lambda: _ema_loop(close_small, 12)),  # Test with period
    ("_ema_loop_alpha", lambda: _ema_loop_alpha(close_small, 0.1)),  # Test with alpha
    
    # Advanced indicators
    ("_kalman_loop", lambda: _kalman_loop(close_small, 5, 0.01, 0.1)),
    ("_vwap_daily_loop", lambda: _vwap_daily_loop(high_arr, low_arr, close_large, volume, timestamps)),
    ("_rng_filter_loop", lambda: _rng_filter_loop(close_small, np.random.random(SIZE_SMALL) * 10)),
    ("_smooth_range", lambda: _smooth_range(close_small, 22, 9)),
    
    # MMH components
    ("_calc_mmh_worm_loop", lambda: _calc_mmh_worm_loop(close_small, np.random.random(SIZE_SMALL) * 5, SIZE_SMALL)),
    ("_calc_mmh_value_loop", lambda: _calc_mmh_value_loop(np.random.random(SIZE_SMALL), SIZE_SMALL)),
    ("_calc_mmh_momentum_loop", lambda: _calc_mmh_momentum_loop(np.random.random(SIZE_SMALL), SIZE_SMALL)),
    
    # Rolling statistics (both serial and parallel)
    ("_rolling_std_welford", lambda: _rolling_std_welford(close_small, 50, 0.9)),
    ("_rolling_std_welford_parallel", lambda: _rolling_std_welford_parallel(close_large, 50, 0.9)),
    ("_rolling_mean_numba", lambda: _rolling_mean_numba(close_small, 144)),
    ("_rolling_mean_numba_parallel", lambda: _rolling_mean_numba_parallel(close_large, 144)),
    ("_rolling_min_max_numba", lambda: _rolling_min_max_numba(close_small, 144)),
    ("_rolling_min_max_numba_parallel", lambda: _rolling_min_max_numba_parallel(close_large, 144)),
    
    # Core indicators (these use integer periods internally)
    ("_calculate_ppo_core", lambda: _calculate_ppo_core(close_small, 7, 16, 5)),
    ("_calculate_rsi_core", lambda: _calculate_rsi_core(close_small, 21)),
    
    # Candle quality checks (vectorized)
    ("_vectorized_wick_check_buy", lambda: _vectorized_wick_check_buy(open_arr, high_arr, low_arr, close_large, 0.2)),
    ("_vectorized_wick_check_sell", lambda: _vectorized_wick_check_sell(open_arr, high_arr, low_arr, close_large, 0.2))
]

compiled_count = 0
failed_count = 0
failed_functions = []

print(f"\n🔨 Compiling {len(functions_to_compile)} Numba functions...\n")

for name, func in functions_to_compile:
    try:
        func_start = time.time()
        result = func()
        func_time = time.time() - func_start
        print(f"  ✅ {name:<40} compiled in {func_time:.3f}s")
        compiled_count += 1
    except Exception as e:
        error_msg = str(e)[:100]
        print(f"  ❌ {name:<40} FAILED: {error_msg}")
        failed_count += 1
        failed_functions.append((name, error_msg))

total_time = time.time() - start_time

print(f"\n{'='*70}")
print(f"📦 AOT Compilation Summary")
print(f"{'='*70}")
print(f"✅ Successfully compiled: {compiled_count}/{len(functions_to_compile)}")
if failed_count > 0:
    print(f"❌ Failed: {failed_count}")
    print(f"\n⚠️ Failed functions:")
    for fname, error in failed_functions:
        print(f"   - {fname}: {error}")
print(f"⏱️ Total time: {total_time:.2f}s")
print(f"💾 Cache location: {os.environ.get('NUMBA_CACHE_DIR', 'default')}")
print(f"{'='*70}\n")

# Verify cache directory with detailed statistics
cache_dir = Path(os.environ.get('NUMBA_CACHE_DIR', '/app/src/__pycache__'))
if cache_dir.exists():
    cache_files_nbi = list(cache_dir.rglob('*.nbi'))
    cache_files_nbc = list(cache_dir.rglob('*.nbc'))
    cache_files_npz = list(cache_dir.rglob('*.npz'))
    
    total_files = len(cache_files_nbi) + len(cache_files_nbc) + len(cache_files_npz)
    
    print(f"🔍 Cache contains:")
    print(f"   - {len(cache_files_nbi)} .nbi files (compiled functions)")
    print(f"   - {len(cache_files_nbc)} .nbc files (bytecode)")
    print(f"   - {len(cache_files_npz)} .npz files (data)")
    print(f"   - {total_files} files total\n")
    
    if total_files > 0:
        # Calculate total size
        total_size = sum(f.stat().st_size for f in (cache_files_nbi + cache_files_nbc + cache_files_npz))
        total_size_kb = total_size / 1024
        print(f"💾 Total cache size: {total_size_kb:.1f} KB\n")
        
        print(f"📂 Sample cache files (.nbi):")
        for f in sorted(cache_files_nbi)[:5]:
            rel_path = f.relative_to(cache_dir)
            file_size = f.stat().st_size / 1024
            print(f"   {rel_path} ({file_size:.1f} KB)")
        if len(cache_files_nbi) > 5:
            print(f"   ... and {len(cache_files_nbi) - 5} more .nbi files")
        
        if cache_files_nbc:
            print(f"\n📂 Sample cache files (.nbc):")
            for f in sorted(cache_files_nbc)[:5]:
                rel_path = f.relative_to(cache_dir)
                file_size = f.stat().st_size / 1024
                print(f"   {rel_path} ({file_size:.1f} KB)")
            if len(cache_files_nbc) > 5:
                print(f"   ... and {len(cache_files_nbc) - 5} more .nbc files")
    
    # Validation check
    MIN_EXPECTED_FILES = 15
    if total_files < MIN_EXPECTED_FILES:
        print(f"\n⚠️ WARNING: Expected at least {MIN_EXPECTED_FILES} cache files, found only {total_files}")
        print(f"   This may indicate incomplete compilation. Check for errors above.")
    else:
        print(f"\n✅ Cache validation passed ({total_files} >= {MIN_EXPECTED_FILES} files)")
else:
    print("⚠️ Warning: Cache directory not found!")
    print(f"   Expected location: {cache_dir}")

if failed_count > 0:
    print(f"\n⚠️ Some functions failed to compile - container will still work but may be slower")
    print(f"   Consider investigating the {failed_count} failed function(s) above")
    # Don't fail the build - partial AOT is better than no AOT
    sys.exit(0)
else:
    print("\n🎉 All functions compiled successfully!")
    sys.exit(0)