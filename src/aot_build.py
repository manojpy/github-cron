#!/usr/bin/env python3
"""
Compile the heavy Numba helpers into a true AOT shared object.
Executed once per architecture (or in CI) – produces _macd_aot.so
"""
from __future__ import annotations
import sys
from pathlib import Path

# ------------------------------------------------------------------
# Add the folder that contains numba_helpers.py to Python path
# ------------------------------------------------------------------
_SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(_SRC_DIR))

# ------------------------------------------------------------------
# Import the **raw** Numba functions – **no** bridge, **no** macd_unified
# ------------------------------------------------------------------
from numba_helpers import (          # noqa  (we need them in globals)
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
    _vectorized_wick_check_sell,
)

# ------------------------------------------------------------------
# Numba C compiler instance
# ------------------------------------------------------------------
from numba.pycc import CC

cc = CC("_macd_aot")
cc.verbose = True

# signatures that return tuples must be declared explicitly
cc.export(
    "_calculate_ppo_core",
    "Tuple((float64[:], float64[:]))(float64[:], int32, int32, int32)"
)(lambda *a: None)

cc.export(
    "_rolling_min_max_numba",
    "Tuple((float64[:], float64[:]))(float64[:], int32)"
)(lambda *a: None)

cc.export(
    "_rolling_min_max_numba_parallel",
    "Tuple((float64[:], float64[:]))(float64[:], int32)"
)(lambda *a: None)

# ------------------------------------------------------------------
# Register every function with auto-signature
# ------------------------------------------------------------------
AOT_FUNCTIONS = [
    "_sanitize_array_numba",
    "_sanitize_array_numba_parallel",
    "_sma_loop",
    "_sma_loop_parallel",
    "_ema_loop",
    "_ema_loop_alpha",
    "_kalman_loop",
    "_vwap_daily_loop",
    "_rng_filter_loop",
    "_smooth_range",
    _calc_mmh_worm_loop,
    "_calc_mmh_value_loop",
    "_calc_mmh_momentum_loop",
    "_rolling_std_welford",
    "_rolling_std_welford_parallel",
    "_rolling_mean_numba",
    "_rolling_mean_numba_parallel",
    "_vectorized_wick_check_buy",
    "_vectorized_wick_check_sell",
    "_calculate_rsi_core",
]

for name in AOT_FUNCTIONS:
    cc.export(name, globals()[name])

# ------------------------------------------------------------------
# Compile
# ------------------------------------------------------------------
if __name__ == "__main__":
    out_dir = _SRC_DIR
    cc.output_dir = str(out_dir)
    cc.compile()
    print("✅ AOT compilation finished ->", out_dir / "_macd_aot.so")
