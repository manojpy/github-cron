#!/usr/bin/env python3
import os
import sys
import time
import numpy as np
from pathlib import Path

# 1. ALIGN WITH RUNTIME: Match the Docker build and runtime cache paths
os.environ['NUMBA_CACHE_DIR'] = '/app/src/__pycache__'
os.environ['NUMBA_NUM_THREADS'] = '4'
os.environ['NUMBA_THREADING_LAYER'] = 'omp'
os.environ['PYTHONDONTWRITEBYTECODE'] = '0'  # CRITICAL for Numba's locator

# Mock minimal config to bypass validation during import
os.environ['TELEGRAM_BOT_TOKEN'] = '000000:AA'
os.environ['TELEGRAM_CHAT_ID'] = '0000'
os.environ['REDIS_URL'] = 'redis://localhost:6379'
os.environ['DELTA_API_BASE'] = 'https://api.delta.exchange'

# 2. MATCH THE BOT'S IMPORT STYLE
# We rely on PYTHONPATH="/app" set in the Dockerfile
try:
    from src.macd_unified import (
        _sanitize_array_numba, _sanitize_array_numba_parallel,
        _sma_loop, _sma_loop_parallel, _ema_loop, _ema_loop_alpha,
        _kalman_loop, _vwap_daily_loop, _rng_filter_loop, _smooth_range,
        _calc_mmh_worm_loop, _calc_mmh_value_loop, _calc_mmh_momentum_loop,
        _rolling_std_welford, _rolling_std_welford_parallel,
        _rolling_mean_numba, _rolling_mean_numba_parallel,
        _rolling_min_max_numba, _rolling_min_max_numba_parallel,
        _calculate_ppo_core, _calculate_rsi_core,
        _vectorized_wick_check_buy, _vectorized_wick_check_sell
    )
    print("✅ Module imported successfully using standard package paths")
except Exception as e:
    print(f"❌ Failed to import functions: {e}")
    sys.exit(1)

def compile_all():
    print("🔧 Starting AOT compilation for all indicators...")
    start_time = time.time()

    # Constants for data generation
    SIZE_SMALL = 200
    SIZE_LARGE = 400

    # Generate varied test data to "bake in" signatures
    close_small = np.random.random(SIZE_SMALL).astype(np.float64) * 50000 + 30000
    close_large = np.random.random(SIZE_LARGE).astype(np.float64) * 50000 + 30000
    open_arr = close_large + np.random.random(SIZE_LARGE).astype(np.float64) * 100
    high_arr = np.maximum(open_arr, close_large) + np.random.random(SIZE_LARGE).astype(np.float64) * 50
    low_arr = np.minimum(open_arr, close_large) - np.random.random(SIZE_LARGE).astype(np.float64) * 50
    volume = np.random.random(SIZE_LARGE).astype(np.float64) * 1000000
    timestamps = np.arange(SIZE_LARGE, dtype=np.int64) * 900

    # Full compilation list
    functions_to_compile = [
        ("_sanitize_array_numba", lambda: _sanitize_array_numba(close_small, 0.0)),
        ("_sanitize_array_numba_parallel", lambda: _sanitize_array_numba_parallel(close_large, 0.0)),
        
        # Moving averages
        ("_sma_loop", lambda: _sma_loop(close_small, 50)),
        ("_sma_loop_parallel", lambda: _sma_loop_parallel(close_large, 50)),
        ("_ema_loop", lambda: _ema_loop(close_small, 0.1)),
        ("_ema_loop_alpha", lambda: _ema_loop_alpha(close_small, 0.1)),
        
        # Advanced indicators
        ("_kalman_loop", lambda: _kalman_loop(close_small, 5, 0.01, 0.1)),
        ("_vwap_daily_loop", lambda: _vwap_daily_loop(high_arr, low_arr, close_large, volume, timestamps)),
        ("_rng_filter_loop", lambda: _rng_filter_loop(close_small, np.random.random(SIZE_SMALL) * 10)),
        ("_smooth_range", lambda: _smooth_range(close_small, 22, 9)),
        
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
    for name, func in functions_to_compile:
        try:
            t0 = time.time()
            func()
            print(f"  ✅ {name:<35} ({time.time()-t0:.3f}s)")
            compiled_count += 1
        except Exception as e:
            print(f"  ❌ {name:<35} FAILED: {str(e)[:50]}")

    print(f"\n📦 AOT Compilation Summary")
    print(f"Successfully compiled: {compiled_count}/{len(functions_to_compile)}")
    print(f"Total time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    # Ensure the cache directory exists before starting
    Path(os.environ['NUMBA_CACHE_DIR']).mkdir(parents=True, exist_ok=True)
    compile_all()
    sys.exit(0)