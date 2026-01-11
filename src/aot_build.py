"""
AOT Build Script - Compiles Numba functions to native .so libraries
====================================================================

Compiles all 20 shared Numba functions to platform-specific shared libraries
using Numba's CC (compile cache) export mechanism.

Usage:
    python aot_build.py [--output-dir OUTPUT_DIR] [--parallel]

Output:
    - numba_aot.so (Linux)
    - numba_aot.dylib (macOS)
    - numba_aot.dll (Windows)
"""

import sys
import os
import argparse
import platform
import shutil
from pathlib import Path
from typing import List, Dict, Any

try:
    import numpy as np
    from numba.pycc import CC
    from numba import types
    from numba.types import Tuple
except ImportError as e:
    print(f"ERROR: Required dependencies missing: {e}", file=sys.stderr)
    print("Install with: pip install numba numpy", file=sys.stderr)
    sys.exit(1)

# Import shared function definitions
try:
    from numba_functions_shared import (
        sanitize_array_numba,
        sanitize_array_numba_parallel,
        ema_loop,
        ema_loop_alpha,
        kalman_loop,
        vwap_daily_loop,
        rng_filter_loop,
        smooth_range,
        calculate_trends_with_state,
        calc_mmh_worm_loop,
        calc_mmh_value_loop,
        calc_mmh_momentum_loop,
        calc_mmh_momentum_smoothing,
        rolling_std,
        rolling_mean_numba,
        rolling_min_max_numba,
        calculate_ppo_core,
        calculate_rsi_core,
        vectorized_wick_check_buy,
        vectorized_wick_check_sell,
    )
except ImportError as e:
    print(f"ERROR: Cannot import shared functions: {e}", file=sys.stderr)
    print("Ensure numba_functions_shared.py is in the same directory", file=sys.stderr)
    sys.exit(1)


def get_platform_extension() -> str:
    system = platform.system()
    if system == "Linux":
        return ".so"
    elif system == "Darwin":
        return ".dylib"
    elif system == "Windows":
        return ".dll"
    else:
        raise RuntimeError(f"Unsupported platform: {system}")


def create_aot_module(output_dir: Path, module_name: str = "numba_aot") -> CC:
  
    print(f"🔧 Configuring AOT compilation context...")
    print(f"   Module name: {module_name}")
    print(f"   Output directory: {output_dir}")
    
    # Create CC context
    cc = CC(module_name)
    cc.output_dir = str(output_dir)
    
    # Numba type definitions for clarity
    f64_1d = types.float64[:]
    i64_1d = types.int64[:]
    b_1d = types.boolean[:]
    f64 = types.float64
    i32 = types.int32
    
    print(f"✅ CC context created")
    return cc, f64_1d, i64_1d, b_1d, f64, i32


def export_all_functions(cc: CC, type_defs: tuple) -> Dict[str, Any]:
    
    f64_1d, i64_1d, b_1d, f64, i32 = type_defs
    
    signatures = {}
    
    print(f"\n📦 Exporting functions to AOT module...")
    
    # ========================================================================
    # 1-2. SANITIZATION FUNCTIONS
    # ========================================================================
    
    print("  [1/22] sanitize_array_numba")
    cc.export('sanitize_array_numba', 'f8[:](f8[:], f8)')(sanitize_array_numba)
    signatures['sanitize_array_numba'] = (f64_1d, f64)
    
    print("  [2/22] sanitize_array_numba_parallel")
    cc.export('sanitize_array_numba_parallel', 'f8[:](f8[:], f8)')(sanitize_array_numba_parallel)
    signatures['sanitize_array_numba_parallel'] = (f64_1d, f64)
    
    # ========================================================================
    # 3-4. EMA FUNCTIONS
    # ========================================================================
    
    print("  [3/22] ema_loop")
    cc.export('ema_loop', 'f8[:](f8[:], f8)')(ema_loop)
    signatures['ema_loop'] = (f64_1d, f64)
    
    print("  [4/22] ema_loop_alpha")
    cc.export('ema_loop_alpha', 'f8[:](f8[:], f8)')(ema_loop_alpha)
    signatures['ema_loop_alpha'] = (f64_1d, f64)
    
    # ========================================================================
    # 5. KALMAN FILTER
    # ========================================================================
    
    print("  [5/22] kalman_loop")
    cc.export('kalman_loop', 'f8[:](f8[:], i4, f8, f8)')(kalman_loop)
    signatures['kalman_loop'] = (f64_1d, i32, f64, f64)
    
    # ========================================================================
    # 6. VWAP
    # ========================================================================
    
    print("  [6/22] vwap_daily_loop")
    cc.export('vwap_daily_loop', 'f8[:](f8[:], f8[:], f8[:], f8[:], i8[:])')(vwap_daily_loop)
    signatures['vwap_daily_loop'] = (f64_1d, f64_1d, f64_1d, f64_1d, i64_1d)
    
    # ========================================================================
    # 7-9. FILTER FUNCTIONS
    # ========================================================================
    
    print("  [7/22] rng_filter_loop")
    cc.export('rng_filter_loop', 'f8[:](f8[:], f8[:])')(rng_filter_loop)
    signatures['rng_filter_loop'] = (f64_1d, f64_1d)
    
    print("  [8/22] smooth_range")
    cc.export('smooth_range', 'f8[:](f8[:], i4, i4)')(smooth_range)
    signatures['smooth_range'] = (f64_1d, i32, i32)
    
    print("  [9/22] calculate_trends_with_state")
    # Returns tuple: (bool[:], bool[:])
    cc.export('calculate_trends_with_state', 'Tuple((b1[:], b1[:]))(f8[:], f8[:])')(calculate_trends_with_state)
    signatures['calculate_trends_with_state'] = (f64_1d, f64_1d)
    
    # ========================================================================
    # 10-12. MMH COMPONENTS
    # ========================================================================
    
    print("  [10/22] calc_mmh_worm_loop")
    cc.export('calc_mmh_worm_loop', 'f8[:](f8[:], f8[:], i8)')(calc_mmh_worm_loop)
    signatures['calc_mmh_worm_loop'] = (f64_1d, f64_1d, types.int64)
    
    print("  [11/22] calc_mmh_value_loop")
    cc.export('calc_mmh_value_loop', 'f8[:](f8[:], i8)')(calc_mmh_value_loop)
    signatures['calc_mmh_value_loop'] = (f64_1d, types.int64)
    
    print("  [12/22] calc_mmh_momentum_loop")
    cc.export('calc_mmh_momentum_loop', 'f8[:](f8[:], i8)')(calc_mmh_momentum_loop)
    signatures['calc_mmh_momentum_loop'] = (f64_1d, types.int64)

    print("  [12.5/23] calc_mmh_momentum_smoothing")
    cc.export('calc_mmh_momentum_smoothing', 'f8[:](f8[:], i8)')(calc_mmh_momentum_smoothing)
    signatures['calc_mmh_momentum_smoothing'] = (f64_1d, types.int64)

    
    # ========================================================================
    # 13-14. ROLLING STANDARD DEVIATION
    # ========================================================================
    
    print("  [13/22] rolling_std_welford")
    cc.export('rolling_std', 'f8[:](f8[:], i4, f8)')(rolling_std)
    signatures['rolling_std'] = (f64_1d, i32, f64)
    
    
    # ========================================================================
    # 15-16. ROLLING MEAN
    # ========================================================================
    
    print("  [15/22] rolling_mean_numba")
    cc.export('rolling_mean_numba', 'f8[:](f8[:], i4)')(rolling_mean_numba)
    signatures['rolling_mean_numba'] = (f64_1d, i32)
    
    
    # ========================================================================
    # 17-18. ROLLING MIN/MAX (TUPLE RETURNS)
    # ========================================================================
    
    print("  [17/22] rolling_min_max_numba")
    # Returns tuple: (float64[:], float64[:])
    cc.export('rolling_min_max_numba', 'Tuple((f8[:], f8[:]))(f8[:], i4)')(rolling_min_max_numba)
    signatures['rolling_min_max_numba'] = (f64_1d, i32)
    
    # ========================================================================
    # 19-20. OSCILLATORS (TUPLE RETURNS)
    # ========================================================================
    
    print("  [19/22] calculate_ppo_core")
    # Returns tuple: (float64[:], float64[:])
    cc.export('calculate_ppo_core', 'Tuple((f8[:], f8[:]))(f8[:], i4, i4, i4)')(calculate_ppo_core)
    signatures['calculate_ppo_core'] = (f64_1d, i32, i32, i32)
    
    print("  [20/22] calculate_rsi_core")
    cc.export('calculate_rsi_core', 'f8[:](f8[:], i4)')(calculate_rsi_core)
    signatures['calculate_rsi_core'] = (f64_1d, i32)
    
    # ========================================================================
    # 21-22. CANDLE PATTERN RECOGNITION
    # ========================================================================
    
    print("  [21/22] vectorized_wick_check_buy")
    cc.export('vectorized_wick_check_buy', 'b1[:](f8[:], f8[:], f8[:], f8[:], f8)')(vectorized_wick_check_buy)
    signatures['vectorized_wick_check_buy'] = (f64_1d, f64_1d, f64_1d, f64_1d, f64)
    
    print("  [22/22] vectorized_wick_check_sell")
    cc.export('vectorized_wick_check_sell', 'b1[:](f8[:], f8[:], f8[:], f8[:], f8)')(vectorized_wick_check_sell)
    signatures['vectorized_wick_check_sell'] = (f64_1d, f64_1d, f64_1d, f64_1d, f64)

    print(f"✅ All 20 functions exported successfully\n")
    
    return signatures


def compile_module(cc: CC, output_dir: Path, module_name: str) -> Path:
    
    extension = get_platform_extension()
    expected_output = output_dir / f"{module_name}{extension}"
    
    print(f"🔨 Starting compilation...")
    print(f"   Expected output: {expected_output}")
    print(f"   Output directory: {output_dir}")
    print(f"   This may take 2-5 minutes...\n")
    
    try:
        # Compile using Numba CC
        cc.compile()
        print(f"✅ CC.compile() completed")
        
        # List all files in output directory for debugging
        print(f"\n📂 Contents of {output_dir}:")
        if output_dir.exists():
            for item in sorted(output_dir.iterdir()):
                if item.is_file():
                    size_kb = item.stat().st_size / 1024
                    print(f"   - {item.name} ({size_kb:.1f} KB)")
        else:
            print(f"   ⚠️ Directory does not exist!")
        
        # Check for expected output
        if not expected_output.exists():
            # Try to find any .so/.dylib/.dll files
            pattern = f"*{extension}"
            found_libs = list(output_dir.glob(pattern))
            
            if found_libs:
                print(f"\n⚠️ Expected {expected_output.name} not found")
                print(f"   But found these libraries: {[f.name for f in found_libs]}")
                # Use the first one found
                library_path = found_libs[0]
                print(f"   Using: {library_path}")
            else:
                raise FileNotFoundError(
                    f"Expected output {expected_output} not found. "
                    f"No {extension} files in {output_dir}"
                )
        else:
            library_path = expected_output
        
        file_size = library_path.stat().st_size / 1024 / 1024  # MB
        print(f"\n✅ Compilation successful!")
        print(f"   Output: {library_path}")
        print(f"   Size: {file_size:.2f} MB")
        
        return library_path
        
    except Exception as e:
        print(f"\n❌ Compilation failed: {e}", file=sys.stderr)
        
        # Additional debugging info
        print(f"\nDebug info:")
        print(f"   CC output_dir: {cc.output_dir}")
        print(f"   CC name: {cc.name}")
        print(f"   Expected extension: {extension}")
        
        raise


def verify_compilation(library_path: Path) -> bool:
    
    print(f"\n🔍 Verifying compiled library...")
    
    try:
        import ctypes
        lib = ctypes.CDLL(str(library_path))
        print(f"✅ Library loads successfully")
        
        # Check for a sample function
        if hasattr(lib, 'sanitize_array_numba'):
            print(f"✅ Sample function 'sanitize_array_numba' found")
        else:
            print(f"⚠️  Warning: Sample function not found (may be mangled)")
        
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}", file=sys.stderr)
        return False


def main():
    """Main AOT build orchestration"""
    parser = argparse.ArgumentParser(
        description="Compile Numba functions to native shared library (AOT)"
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path.cwd(),
        help='Output directory for compiled artifacts (default: current directory)'
    )
    parser.add_argument(
        '--module-name',
        type=str,
        default='macd_aot_compiled',  # Changed from 'numba_aot'
        help='Name of output module (default: macd_aot_compiled)'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify compiled library after build'
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("AOT Compilation Script - Numba Functions to Native Library")
    print("=" * 70)
    print(f"Python: {sys.version}")
    print(f"NumPy: {np.__version__}")
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"Target extension: {get_platform_extension()}")
    print(f"Module name: {args.module_name}")
    print("=" * 70)
    print()
    
    try:
        # Step 1: Create compilation context
        cc, f64_1d, i64_1d, b_1d, f64, i32 = create_aot_module(
            args.output_dir, 
            args.module_name
        )
        
        # Step 2: Export all functions
        signatures = export_all_functions(cc, (f64_1d, i64_1d, b_1d, f64, i32))
        
        # Step 3: Compile
        library_path = compile_module(cc, args.output_dir, args.module_name)
        
        # Step 4: Verify compilation output exists
        if not library_path.exists():
            raise FileNotFoundError(f"Compilation claimed success but {library_path} not found")
        
        # Step 5: List all output files for debugging
        print(f"\n📂 Build artifacts in {args.output_dir}:")
        for item in sorted(args.output_dir.iterdir()):
            if item.is_file():
                size_mb = item.stat().st_size / 1024 / 1024
                print(f"   {item.name} ({size_mb:.2f} MB)")
        
        # Step 6: Optional verification
        if args.verify:
            verify_compilation(library_path)
        
        print("\n" + "=" * 70)
        print("✅ AOT BUILD COMPLETE")
        print("=" * 70)
        print(f"Compiled library: {library_path}")
        print(f"Functions exported: {len(signatures)}")
        print(f"\nTo use: import aot_bridge; aot_bridge.ensure_initialized()")
        print("=" * 70)
        
        return 0
        
    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ AOT BUILD FAILED")
        print("=" * 70)
        print(f"Error: {e}", file=sys.stderr)
        
        # Print traceback for debugging
        import traceback
        traceback.print_exc()
        
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())