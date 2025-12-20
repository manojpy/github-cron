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
        _kalman_loop,
        _vwap_daily_loop,
        _rng_filter_loop,
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

functions_to_compile = [
    # Array sanitization
    ("_sanitize_array_numba", lambda: _sanitize_array_numba(close_small, 0.0)),
    ("_sanitize_array_numba_parallel", lambda: _sanitize_array_numba_parallel(close_large, 0.0)),
    
    # Moving averages
    ("_sma_loop", lambda: _sma_loop(close_small, 50)),
    ("_sma_loop_parallel", lambda: _sma_loop_parallel(close_large, 50)),
    ("_ema_loop", lambda: _ema_loop(close_small, 0.1)),
    
    # Advanced indicators
    ("_kalman_loop", lambda: _kalman_loop(close_small, 5, 0.01, 0.1)),
    ("_vwap_daily_loop", lambda: _vwap_daily_loop(high_arr, low_arr, close_large, volume, timestamps)),
    ("_rng_filter_loop", lambda: _rng_filter_loop(close_small, np.random.random(SIZE_SMALL) * 10)),
    
    # MMH components
    ("_calc_mmh_worm_loop", lambda: _calc_mmh_worm_loop(close_small, np.random.random(SIZE_SMALL) * 5, SIZE_SMALL)),
    ("_calc_mmh_value_loop", lambda: _calc_mmh_value_loop(np.random.random(SIZE_SMALL), SIZE_SMALL)),
    ("_calc_mmh_momentum_loop", lambda: _calc_mmh_momentum_loop(np.random.random(SIZE_SMALL), SIZE_SMALL)),
    
    # Rolling statistics
    ("_rolling_std_welford", lambda: _rolling_std_welford(close_small, 50, 0.9)),
    ("_rolling_std_welford_parallel", lambda: _rolling_std_welford_parallel(close_large, 50, 0.9)),
    ("_rolling_mean_numba", lambda: _rolling_mean_numba(close_small, 144)),
    ("_rolling_mean_numba_parallel", lambda: _rolling_mean_numba_parallel(close_large, 144)),
    ("_rolling_min_max_numba", lambda: _rolling_min_max_numba(close_small, 144)),
    ("_rolling_min_max_numba_parallel", lambda: _rolling_min_max_numba_parallel(close_large, 144)),
    
    # Core indicators
    ("_calculate_ppo_core", lambda: _calculate_ppo_core(close_small, 7, 16, 5)),
    ("_calculate_rsi_core", lambda: _calculate_rsi_core(close_small, 21)),
    
    # Candle quality checks
    ("_vectorized_wick_check_buy", lambda: _vectorized_wick_check_buy(open_arr, high_arr, low_arr, close_large, 0.2)),
    ("_vectorized_wick_check_sell", lambda: _vectorized_wick_check_sell(open_arr, high_arr, low_arr, close_large, 0.2))
]

compiled_count = 0
failed_count = 0

print(f"\n🔨 Compiling {len(functions_to_compile)} Numba functions...\n")

for name, func in functions_to_compile:
    try:
        func_start = time.time()
        result = func()
        func_time = time.time() - func_start
        print(f"  ✅ {name:<35} compiled in {func_time:.3f}s")
        compiled_count += 1
    except Exception as e:
        print(f"  ❌ {name:<35} FAILED: {str(e)[:50]}")
        failed_count += 1

total_time = time.time() - start_time

print(f"\n{'='*70}")
print(f"📦 AOT Compilation Summary")
print(f"{'='*70}")
print(f"✅ Successfully compiled: {compiled_count}/{len(functions_to_compile)}")
if failed_count > 0:
    print(f"❌ Failed: {failed_count}")
print(f"⏱️  Total time: {total_time:.2f}s")
print(f"💾 Cache location: {os.environ.get('NUMBA_CACHE_DIR', 'default')}")
print(f"{'='*70}\n")

# Verify cache directory
cache_dir = Path(os.environ.get('NUMBA_CACHE_DIR', '/app/src/__pycache__'))
if cache_dir.exists():
    cache_files = list(cache_dir.rglob('*.nbi'))
    print(f"📁 Cache contains {len(cache_files)} compiled files")
    if len(cache_files) > 0:
        print(f"📂 Sample cache files:")
        for f in sorted(cache_files)[:5]:
            rel_path = f.relative_to(cache_dir)
            file_size = f.stat().st_size / 1024
            print(f"   {rel_path} ({file_size:.1f} KB)")
        if len(cache_files) > 5:
            print(f"   ... and {len(cache_files) - 5} more files")
else:
    print("⚠️  Warning: Cache directory not found!")

if failed_count > 0:
    print("\n⚠️  Some functions failed to compile - container will still work but may be slower")
    sys.exit(0)  # Don't fail the build
else:
    print("\n🎉 All functions compiled successfully!")
    sys.exit(0)