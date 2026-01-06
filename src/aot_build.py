import sys
import os
import argparse
import platform
import importlib.util
from pathlib import Path
import numpy as np
from numba.pycc import CC
from numba import types

# Import the shared functions
try:
    import numba_functions_shared as nfs
except ImportError:
    print("ERROR: numba_functions_shared.py not found.")
    sys.exit(1)

def verify_compilation(lib_path: Path):
    """Verify that the compiled library can be loaded and symbols exist."""
    print(f"🔍 Verifying binary: {lib_path}")
    try:
        spec = importlib.util.spec_from_file_location("test_aot", str(lib_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # Check one key function
        if hasattr(mod, "ema_loop"):
            print("✅ AOT binary verified successfully.")
        else:
            raise AttributeError("ema_loop not found in binary")
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        sys.exit(1)

def build():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--module-name", type=str, default="macd_aot_compiled")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    cc = CC(args.module_name)
    cc.verbose = True
    
    # We use Numba's internal type mapping to ensure strings like 'i32' work
    # f8[:] = float64 array, i32 = int32, b1 = boolean
    
    # 1-2. Sanitization
    cc.export("sanitize_array_numba", "f8[:](f8[:], f8)")(nfs.sanitize_array_numba)
    cc.export("sanitize_array_numba_parallel", "f8[:](f8[:], f8)")(nfs.sanitize_array_numba_parallel)

    # 3-4. Moving Averages
    cc.export("ema_loop", "f8[:](f8[:], i32)")(nfs.ema_loop)
    cc.export("ema_loop_alpha", "f8[:](f8[:], f8)")(nfs.ema_loop_alpha)

    # 5-8. Filters
    cc.export("rng_filter_loop", "f8[:](f8[:], f8[:])")(nfs.rng_filter_loop)
    cc.export("smooth_range", "f8[:](f8[:], i32, f8)")(nfs.smooth_range)
    cc.export("calculate_trends_with_state", "Tuple((i64[:], i64[:], b1[:]))(f8[:], f8[:], f8[:], f8[:], i64[:], i64[:], b1[:], f8, i32)")(nfs.calculate_trends_with_state)
    cc.export("kalman_loop", "f8[:](f8[:], f8, f8)")(nfs.kalman_loop)

    # 9. Market Indicators
    cc.export("vwap_daily_loop", "f8[:](f8[:], f8[:], i64[:])")(nfs.vwap_daily_loop)

    # 10-15. Statistical
    cc.export("rolling_std_welford", "f8[:](f8[:], i32)")(nfs.rolling_std_welford)
    cc.export("rolling_std_welford_parallel", "f8[:](f8[:], i32)")(nfs.rolling_std_welford_parallel)
    cc.export("rolling_mean_numba", "f8[:](f8[:], i32)")(nfs.rolling_mean_numba)
    cc.export("rolling_mean_numba_parallel", "f8[:](f8[:], i32)")(nfs.rolling_mean_numba_parallel)
    cc.export("rolling_min_max_numba", "Tuple((f8[:], f8[:]))(f8[:], i32)")(nfs.rolling_min_max_numba)
    cc.export("rolling_min_max_numba_parallel", "Tuple((f8[:], f8[:]))(f8[:], i32)")(nfs.rolling_min_max_numba_parallel)

    # 16-17. Oscillators
    cc.export("calculate_ppo_core", "f8[:](f8[:], f8[:])")(nfs.calculate_ppo_core)
    cc.export("calculate_rsi_core", "f8[:](f8[:], i32)")(nfs.calculate_rsi_core)

    # 18-20. MMH Components
    cc.export("calc_mmh_worm_loop", "f8[:](f8[:], f8[:], i32)")(nfs.calc_mmh_worm_loop)
    cc.export("calc_mmh_value_loop", "f8[:](f8[:], i32)")(nfs.calc_mmh_value_loop)
    cc.export("calc_mmh_momentum_loop", "f8[:](f8[:], i32)")(nfs.calc_mmh_momentum_loop)

    # 21-22. Patterns
    cc.export("vectorized_wick_check_buy", "b1[:](f8[:], f8[:], f8[:], f8[:], f8)")(nfs.vectorized_wick_check_buy)
    cc.export("vectorized_wick_check_sell", "b1[:](f8[:], f8[:], f8[:], f8[:], f8)")(nfs.vectorized_wick_check_sell)

    # Compile
    cc.output_dir = str(args.output_dir)
    cc.compile()
    
    # Locate the compiled file
    ext = ".so" if platform.system() != "Windows" else ".pyd"
    files = list(args.output_dir.glob(f"{args.module_name}*{ext}"))
    if files and args.verify:
        verify_compilation(files[0])

if __name__ == "__main__":
    build()