#!/usr/bin/env python3
"""
Compile the heavy Numba helpers into a true AOT shared object.
Executed once per architecture (or in CI) – produces _macd_aot.so
"""
from __future__ import annotations
import os
import sys
import numpy as np
from numba.pycc import CC
from pathlib import Path

# ------------------------------------------------------------------
# List every function we want in the .so (exactly the ones you asked)
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
    "_calc_mmh_worm_loop",
    "_calc_mmh_value_loop",
    "_calc_mmh_momentum_loop",
    "_rolling_std_welford",
    "_rolling_std_welford_parallel",
    "_rolling_mean_numba",
    "_rolling_mean_numba_parallel",
    "_rolling_min_max_numba",
    "_rolling_min_max_numba_parallel",
    "_calculate_ppo_core",
    "_calculate_rsi_core",
    "_vectorized_wick_check_buy",
    "_vectorized_wick_check_sell",
]

# ------------------------------------------------------------------
# Numba C compiler instance
# ------------------------------------------------------------------
cc = CC("_macd_aot")          # output name  ->  _macd_aot.so
cc.verbose = True

# signatures that return **tuples** must be declared explicitly
cc.export("_calculate_ppo_core", "Tuple((float64[:], float64[:]))(float64[:], int32, int32, int32)")(
    lambda *a: None
)
cc.export("_rolling_min_max_numba", "Tuple((float64[:], float64[:]))(float64[:], int32)")(
    lambda *a: None
)
cc.export("_rolling_min_max_numba_parallel", "Tuple((float64[:], float64[:]))(float64[:], int32)")(
    lambda *a: None
)

# ------------------------------------------------------------------
# Import the real implementations from the original file
# ------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from macd_unified import (                               # noqa
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
# Register every function with auto-signature (except the tuple ones above)
# ------------------------------------------------------------------
for name in AOT_FUNCTIONS:
    func = globals()[name]
    if name in {"_calculate_ppo_core", "_rolling_min_max_numba", "_rolling_min_max_numba_parallel"}:
        continue   # already done manually
    cc.export(name, func)

# ------------------------------------------------------------------
# Compile
# ------------------------------------------------------------------
if __name__ == "__main__":
    out_dir = Path(__file__).parent
    cc.output_dir = str(out_dir)
    cc.compile()
    print("✅ AOT compilation finished ->", out_dir / "_macd_aot.so")
