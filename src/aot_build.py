"""
AOT Build Script - Compiles Numba functions to native .so libraries
====================================================================
Uses explicit Numba Type objects to avoid string evaluation NameErrors.
"""

import sys
import os
import argparse
import platform
import importlib.util
from pathlib import Path

try:
    import numpy as np
    from numba.pycc import CC
    from numba import types
except ImportError as e:
    print(f"ERROR: Required dependencies missing: {e}", file=sys.stderr)
    sys.exit(1)

# Import shared function definitions
try:
    import numba_functions_shared as nfs
except ImportError:
    # Try adding current dir to path for Docker builds
    sys.path.append(os.getcwd())
    try:
        import numba_functions_shared as nfs
    except ImportError:
        print("ERROR: numba_functions_shared.py not found in path.", file=sys.stderr)
        sys.exit(1)

def verify_compilation(lib_path: Path):
    """Attempt to load the compiled library to verify it's valid."""
    print(f"🔍 Verifying compiled library: {lib_path}")
    try:
        spec = importlib.util.spec_from_file_location("verify_mod", str(lib_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if hasattr(mod, "ema_loop"):
            print("✅ AOT binary verified and loadable.")
        else:
            raise AttributeError("Binary loaded but symbols are missing.")
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        sys.exit(1)

def build():
    parser = argparse.ArgumentParser(description="AOT Build for MACD Bot")
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--module-name", type=str, default="macd_aot_compiled")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    cc = CC(args.module_name)
    cc.verbose = True
    
    # Define Type Aliases for Readability
    f8_1d = types.float64[:]
    i64_1d = types.int64[:]
    b1_1d = types.bool_[:]
    f8 = types.float64
    i32 = types.int32

    # ============================================================================
    # EXPORT SIGNATURES
    # ============================================================================
    
    # 1-2. Sanitization
    cc.export("sanitize_array_numba", f8_1d(f8_1d, f8))(nfs.sanitize_array_numba)
    cc.export("sanitize_array_numba_parallel", f8_1d(f8_1d, f8))(nfs.sanitize_array_numba_parallel)

    # 3-4. Moving Averages
    cc.export("ema_loop", f8_1d(f8_1d, i32))(nfs.ema_loop)
    cc.export("ema_loop_alpha", f8_1d(f8_1d, f8))(nfs.ema_loop_alpha)

    # 5-8. Filters
    cc.export("rng_filter_loop", f8_1d(f8_1d, f8_1d))(nfs.rng_filter_loop)
    cc.export("smooth_range", f8_1d(f8_1d, i32, f8))(nfs.smooth_range)
    cc.export("calculate_trends_with_state", types.Tuple((i64_1d, i64_1d, b1_1d))(f8_1d, f8_1d, f8_1d, f8_1d, i64_1d, i64_1d, b1_1d, f8, i32))(nfs.calculate_trends_with_state)
    cc.export("kalman_loop", f8_1d(f8_1d, f8, f8))(nfs.kalman_loop)

    # 9. Market Indicators
    cc.export("vwap_daily_loop", f8_1d(f8_1d, f8_1d, i64_1d))(nfs.vwap_daily_loop)

    # 10-15. Statistical
    cc.export("rolling_std_welford", f8_1d(f8_1d, i32))(nfs.rolling_std_welford)
    cc.export("rolling_std_welford_parallel", f8_1d(f8_1d, i32))(nfs.rolling_std_welford_parallel)
    cc.export("rolling_mean_numba", f8_1d(f8_1d, i32))(nfs.rolling_mean_numba)
    cc.export("rolling_mean_numba_parallel", f8_1d(f8_1d, i32))(nfs.rolling_mean_numba_parallel)
    cc.export("rolling_min_max_numba", types.Tuple((f8_1d, f8_1d))(f8_1d, i32))(nfs.rolling_min_max_numba)
    cc.export("rolling_min_max_numba_parallel", types.Tuple((f8_1d, f8_1d))(f8_1d, i32))(nfs.rolling_min_max_numba_parallel)

    # 16-17. Oscillators
    cc.export("calculate_ppo_core", f8_1d(f8_1d, f8_1d))(nfs.calculate_ppo_core)
    cc.export("calculate_rsi_core", f8_1d(f8_1d, i32))(nfs.calculate_rsi_core)

    # 18-20. MMH Components
    cc.export("calc_mmh_worm_loop", f8_1d(f8_1d, f8_1d, i32))(nfs.calc_mmh_worm_loop)
    cc.export("calc_mmh_value_loop", f8_1d(f8_1d, i32))(nfs.calc_mmh_value_loop)
    cc.export("calc_mmh_momentum_loop", f8_1d(f8_1d, i32))(nfs.calc_mmh_momentum_loop)

    # 21-22. Pattern Recognition
    cc.export("vectorized_wick_check_buy", b1_1d(f8_1d, f8_1d, f8_1d, f8_1d, f8))(nfs.vectorized_wick_check_buy)
    cc.export("vectorized_wick_check_sell", b1_1d(f8_1d, f8_1d, f8_1d, f8_1d, f8))(nfs.vectorized_wick_check_sell)

    # Compile
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cc.output_dir = str(args.output_dir)
    cc.compile()
    
    # Locate Output
    ext = ".so" if platform.system() != "Windows" else ".pyd"
    targets = list(args.output_dir.glob(f"{args.module_name}*{ext}"))
    
    if targets:
        print(f"✅ Compilation target created: {targets[0]}")
        if args.verify:
            verify_compilation(targets[0])
    else:
        print("❌ Error: No output file generated.")
        sys.exit(1)

if __name__ == "__main__":
    build()