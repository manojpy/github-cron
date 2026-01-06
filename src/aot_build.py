"""
AOT Build Script - Compiles Numba functions to native .so libraries
====================================================================

Compiles all 22 shared Numba functions to platform-specific shared libraries
using Numba's CC (compile cache) export mechanism.
"""

import sys
import os
import argparse
import platform
import shutil
import importlib.util
from pathlib import Path
from typing import List, Tuple as TypingTuple

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
    print("ERROR: numba_functions_shared.py not found in path.", file=sys.stderr)
    sys.exit(1)

def verify_compilation(lib_path: Path):
    """Attempt to load the compiled library to verify it's valid."""
    print(f"🔍 Verifying compiled library: {lib_path}")
    try:
        spec = importlib.util.spec_from_file_location("verify_mod", str(lib_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        
        # Check if a few key functions exist
        required = ["sanitize_array_numba", "calc_mmh_momentum_loop", "rolling_std_welford"]
        for func in required:
            if not hasattr(mod, func):
                raise AttributeError(f"Missing exported function: {func}")
        
        print("✅ Verification successful: Library is loadable and contains required symbols.")
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        sys.exit(1)

def compile_module(cc: CC, output_dir: Path, module_name: str) -> Path:
    """Execute the compilation process and return the path to the resulting file."""
    print(f"🚀 Starting AOT compilation for module: {module_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run compilation
    cc.compile()
    
    # Identify the extension based on OS
    ext = ".so"
    if platform.system() == "Windows": ext = ".pyd"
    
    # Numba CC creates files like name.cpython-310-x86_64-linux-gnu.so
    compiled_files = list(output_dir.glob(f"{module_name}*{ext}"))
    if not compiled_files:
        raise FileNotFoundError(f"Could not find compiled library in {output_dir}")
        
    return compiled_files[0]

def build():
    parser = argparse.ArgumentParser(description="AOT Build for MACD Bot")
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--module-name", type=str, default="macd_aot_compiled")
    parser.add_argument("--verify", action="store_true", help="Verify library after build")
    args = parser.parse_args()

    cc = CC(args.module_name)
    cc.verbose = True
    cc.output_dir = str(args.output_dir)

    # ============================================================================
    # EXPORT SIGNATURES (ALL 22 FUNCTIONS)
    # ============================================================================
    
    # 1-2. Sanitization
    cc.export("sanitize_array_numba", "f8[:](f8[:], f8)")(nfs.sanitize_array_numba)
    cc.export("sanitize_array_numba_parallel", "f8[:](f8[:], f8)")(nfs.sanitize_array_numba_parallel)

    # 3-4. Moving Averages
    cc.export("ema_loop", "f8[:](f8[:], i32)")(nfs.ema_loop)
    cc.export("ema_loop_alpha", "f8[:](f8[:], f8)")(nfs.ema_loop_alpha)

    # 5-8. Filters
    cc.export("rng_filter_loop", "f8[:](f8[:], f8[:])")(nfs.rng_filter_loop)
    cc.export("smooth_range", "f8[:](f8[:], i32, f8)")(nfs.smooth_range)
    # Complex Tuple: (i64[:], i64[:], b1[:])
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

    # 21-22. Pattern Recognition
    cc.export("vectorized_wick_check_buy", "b1[:](f8[:], f8[:], f8[:], f8[:], f8)")(nfs.vectorized_wick_check_buy)
    cc.export("vectorized_wick_check_sell", "b1[:](f8[:], f8[:], f8[:], f8[:], f8)")(nfs.vectorized_wick_check_sell)

    try:
        lib_path = compile_module(cc, args.output_dir, args.module_name)
        
        print("\n" + "=" * 70)
        print("✅ AOT BUILD SUCCESSFUL")
        print(f"Output: {lib_path.name}")
        print("=" * 70)
        
        if args.verify:
            verify_compilation(lib_path)
            
        return 0
    except Exception as e:
        print(f"\n❌ AOT BUILD FAILED: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(build())