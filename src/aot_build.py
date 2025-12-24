#!/usr/bin/env python3
"""
Compile the heavy Numba helpers into a true AOT shared object.
Executed once per architecture (or in CI) – produces _macd_aot.so
"""
from __future__ import annotations
import os
import sys
import shutil
from pathlib import Path

# ------------------------------------------------------------------
# Signal that helpers must stay **plain** (no @njit)
# ------------------------------------------------------------------
os.environ["AOT_BUILD"] = "1"

# ------------------------------------------------------------------
# Add the folder that contains numba_helpers.py to Python path
# ------------------------------------------------------------------
_SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(_SRC_DIR))

# ------------------------------------------------------------------
# Import the **raw** Numba functions – **no** bridge, **no** macd_unified
# ------------------------------------------------------------------
from numba_helpers import (          # noqa
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
    _vectorized_wick_check_buy,
    _vectorized_wick_check_sell,
    _calculate_rsi_core,
    _calculate_ppo_core,
)

# ------------------------------------------------------------------
# Numba C compiler instance
# ------------------------------------------------------------------
from numba.pycc import CC

cc = CC("_macd_aot")
cc.verbose = True

# ------------------------------------------------------------------
# 1.  Helper: plain signature
# ------------------------------------------------------------------
def _sig(ret, *args):
    return f"{ret}({','.join(args)})"

# ------------------------------------------------------------------
# 2.  Register every function with an **explicit** signature
#    Match actual definitions in numba_helpers.py
# ------------------------------------------------------------------
SIGS = {
    # sanitise
    "_sanitize_array_numba":           _sig("float64[:]", "float64[:]", "float64"),
    "_sanitize_array_numba_parallel":  _sig("float64[:]", "float64[:]", "float64"),

    # moving averages
    "_sma_loop":                       _sig("float64[:]", "float64[:]", "int32"),
    "_sma_loop_parallel":              _sig("float64[:]", "float64[:]", "int32"),
    "_ema_loop":                       _sig("float64[:]", "float64[:]", "float64"),
    "_ema_loop_alpha":                 _sig("float64[:]", "float64[:]", "float64"),

    # Kalman
    "_kalman_loop":                    _sig("float64[:]", "float64[:]", "int32", "float64", "float64"),

    # VWAP
    "_vwap_daily_loop":                _sig("float64[:]", "float64[:]", "float64[:]", "float64[:]", "int64[:]"),

    # RNG
    "_rng_filter_loop":                _sig("float64[:]", "float64[:]", "float64[:]"),
    "_smooth_range":                   _sig("float64[:]", "float64[:]", "int32", "int32"),

    # MMH
    "_calc_mmh_worm_loop":             _sig("float64[:]", "float64[:]", "float64[:]", "int32"),
    "_calc_mmh_value_loop":            _sig("float64[:]", "float64[:]", "int32"),
    "_calc_mmh_momentum_loop":         _sig("float64[:]", "float64[:]", "int32"),

    # rolling stats
    "_rolling_std_welford":            _sig("float64[:]", "float64[:]", "int32", "float64"),
    "_rolling_std_welford_parallel":   _sig("float64[:]", "float64[:]", "int32", "float64"),
    "_rolling_mean_numba":             _sig("float64[:]", "float64[:]", "int32"),
    "_rolling_mean_numba_parallel":    _sig("float64[:]", "float64[:]", "int32"),
    "_rolling_min_max_numba":          "Tuple((float64[:], float64[:]))(float64[:], int32)",
    "_rolling_min_max_numba_parallel": "Tuple((float64[:], float64[:]))(float64[:], int32)",

    # indicators
    "_calculate_ppo_core":             "Tuple((float64[:], float64[:]))(float64[:], int32, int32, int32)",
    "_calculate_rsi_core":             _sig("float64[:]", "float64[:]", "int32"),

    # candle quality
    "_vectorized_wick_check_buy":      _sig("bool[:]", "float64[:]", "float64[:]", "float64[:]", "float64[:]", "float64"),
    "_vectorized_wick_check_sell":     _sig("bool[:]", "float64[:]", "float64[:]", "float64[:]", "float64[:]", "float64"),
}

for name, sig in SIGS.items():
    cc.export(name, sig)(globals()[name])

# ------------------------------------------------------------------
# Compile and normalize filename
# ------------------------------------------------------------------
if __name__ == "__main__":
    out_dir = _SRC_DIR
    cc.output_dir = str(out_dir)
    cc.compile()
    # Normalize ABI-suffixed filename to _macd_aot.so
    built = list(out_dir.glob("_macd_aot*.so"))
    if not built:
        raise RuntimeError("❌ No AOT .so produced")
    target = out_dir / "_macd_aot.so"
    if built[0].name != target.name:
        shutil.move(str(built[0]), target)
    print("✅ AOT compilation finished ->", target)
