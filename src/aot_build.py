"""
AOT Build Script - Compiles Numba functions to native .so libraries
====================================================================

Fixed version with better error reporting and diagnostics.
"""

import sys
import os
import argparse
import platform
import shutil
from pathlib import Path
from typing import List, Dict, Any
import traceback

try:
    import numpy as np
    from numba.pycc import CC
    from numba import types
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
        calculate_atr_rma, 
        vectorized_wick_check_buy,
        vectorized_wick_check_sell,
    )
except ImportError as e:
    print(f"ERROR: Cannot import shared functions: {e}", file=sys.stderr)
    print("Ensure numba_functions_shared.py is in the same directory", file=sys.stderr)
    traceback.print_exc()
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


def create_aot_module(output_dir: Path, module_name: str = "macd_aot_compiled") -> tuple:
    print(f"🔧 Configuring AOT compilation context...")
    print(f"   Module name: {module_name}")
    print(f"   Output directory: {output_dir}")
    print(f"   Python version: {sys.version}")
    print(f"   NumPy version: {np.__version__}")
    
    try:
        cc = CC(module_name)
        cc.output_dir = str(output_dir)
        
        # Numba type definitions
        f64_1d = types.float64[:]
        i64_1d = types.int64[:]
        b_1d = types.boolean[:]
        f64 = types.float64
        i32 = types.int32
        
        print(f"✅ CC context created")
        return cc, f64_1d, i64_1d, b_1d, f64, i32
    
    except Exception as e:
        print(f"❌ Failed to create CC context: {e}")
        traceback.print_exc()
        raise


def export_all_functions(cc: CC, type_defs: tuple) -> Dict[str, Any]:
    f64_1d, i64_1d, b_1d, f64, i32 = type_defs
    signatures = {}
    
    print(f"\n🔦 Exporting functions to AOT module...")
    
    functions_to_export = [
        ('sanitize_array_numba', 'f8[:](f8[:], f8)', sanitize_array_numba),
        ('sanitize_array_numba_parallel', 'f8[:](f8[:], f8)', sanitize_array_numba_parallel),
        ('ema_loop', 'f8[:](f8[:], f8)', ema_loop),
        ('ema_loop_alpha', 'f8[:](f8[:], f8)', ema_loop_alpha),
        ('kalman_loop', 'f8[:](f8[:], i4, f8, f8)', kalman_loop),
        ('vwap_daily_loop', 'f8[:](f8[:], f8[:], f8[:], f8[:], i8[:])', vwap_daily_loop),
        ('calculate_atr_rma', 'f8[:](f8[:], f8[:], f8[:], i4)', calculate_atr_rma),
        ('rng_filter_loop', 'f8[:](f8[:], f8[:])', rng_filter_loop),
        ('smooth_range', 'f8[:](f8[:], i4, i4)', smooth_range),
        ('calculate_trends_with_state', 'Tuple((b1[:], b1[:]))(f8[:], f8[:])', calculate_trends_with_state),
        ('calc_mmh_worm_loop', 'f8[:](f8[:], f8[:], i8)', calc_mmh_worm_loop), 
        ('calc_mmh_value_loop', 'f8[:](f8[:], f8[:], f8[:], i8)', calc_mmh_value_loop),
        ('calc_mmh_momentum_loop', 'f8[:](f8[:], i8)', calc_mmh_momentum_loop),
        ('calc_mmh_momentum_smoothing', 'f8[:](f8[:], i8)', calc_mmh_momentum_smoothing),
        ('rolling_std', 'f8[:](f8[:], i4, f8)', rolling_std),
        ('rolling_mean_numba', 'f8[:](f8[:], i4)', rolling_mean_numba),
        ('rolling_min_max_numba', 'Tuple((f8[:], f8[:]))(f8[:], i4)', rolling_min_max_numba),
        ('calculate_ppo_core', 'Tuple((f8[:], f8[:]))(f8[:], i4, i4, i4)', calculate_ppo_core),
        ('calculate_rsi_core', 'f8[:](f8[:], i4)', calculate_rsi_core),
        ('vectorized_wick_check_buy', 'b1[:](f8[:], f8[:], f8[:], f8[:], f8, f8[:], f8[:], f8)', vectorized_wick_check_buy),
        ('vectorized_wick_check_sell', 'b1[:](f8[:], f8[:], f8[:], f8[:], f8, f8[:], f8[:], f8)', vectorized_wick_check_sell),
    ]

    for idx, (name, sig, func) in enumerate(functions_to_export, 1):
        try:
            print(f"  [{idx:2d}/21] {name}...", end=" ", flush=True)
            cc.export(name, sig)(func)
            signatures[name] = sig
            print("✅")
        except Exception as e:
            print(f"❌\n        Error: {e}")
            traceback.print_exc()
            raise
    
    print(f"\n✅ All 21 functions exported successfully\n")
    return signatures


def compile_module(cc: CC, output_dir: Path, module_name: str) -> Path:
    extension = get_platform_extension()
    expected_output = output_dir / f"{module_name}{extension}"
    
    print(f"🔨 Starting compilation...")
    print(f"   Expected output: {expected_output}")
    print(f"   This may take 2-5 minutes...\n")
    
    try:
        # Pre-compilation checks
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created output directory: {output_dir}")
        
        # Compile
        print("⏳ Running CC.compile()...")
        cc.compile()
        print(f"✅ CC.compile() completed\n")
        
        # List output directory contents
        print(f"📂 Contents of {output_dir}:")
        if output_dir.exists():
            files = list(output_dir.iterdir())
            if files:
                for item in sorted(files):
                    if item.is_file():
                        size_kb = item.stat().st_size / 1024
                        print(f"   - {item.name} ({size_kb:.1f} KB)")
            else:
                print(f"   (empty)")
        else:
            print(f"   ❌ Directory does not exist!")
            raise FileNotFoundError(f"Output directory was not created: {output_dir}")
        
        # Locate compiled library
        if expected_output.exists():
            library_path = expected_output
            print(f"\n✅ Found expected output: {library_path.name}")
        else:
            # Search for any .so/.dylib/.dll
            pattern = f"*{extension}"
            found_libs = list(output_dir.glob(pattern))
            
            if found_libs:
                library_path = found_libs[0]
                print(f"\n⚠️  Expected {expected_output.name} not found")
                print(f"   But found: {library_path.name}")
            else:
                raise FileNotFoundError(
                    f"No compiled library found. Expected {expected_output} or any {extension} files in {output_dir}"
                )
        
        file_size = library_path.stat().st_size / 1024 / 1024
        print(f"\n✅ Compilation successful!")
        print(f"   Output: {library_path}")
        print(f"   Size: {file_size:.2f} MB")
        
        return library_path
        
    except Exception as e:
        print(f"\n❌ Compilation failed: {e}")
        traceback.print_exc()
        
        print(f"\n🔍 Debug info:")
        print(f"   CC.output_dir: {cc.output_dir}")
        print(f"   CC.name: {cc.name}")
        print(f"   Expected extension: {extension}")
        print(f"   Output dir exists: {output_dir.exists()}")
        
        raise


def verify_compilation(library_path: Path) -> bool:
    print(f"\n🔍 Verifying compiled library...")
    
    try:
        import ctypes
        print(f"   Loading: {library_path}")
        lib = ctypes.CDLL(str(library_path))
        print(f"✅ Library loads successfully")
        
        # Check for sample function
        if hasattr(lib, 'sanitize_array_numba'):
            print(f"✅ Sample function 'sanitize_array_numba' found")
        else:
            print(f"⚠️  Warning: Sample function not in ctypes interface (may be normal)")
        
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        traceback.print_exc()
        return False


def main():
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
        default='macd_aot_compiled',
        help='Name of output module (default: macd_aot_compiled)'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify compiled library after build'
    )
    
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("AOT Compilation Script - Numba Functions to Native Library")
    print("=" * 70)
    print(f"Python: {sys.version}")
    print(f"NumPy: {np.__version__}")
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"Target extension: {get_platform_extension()}")
    print(f"Module name: {args.module_name}")
    print(f"Output directory: {args.output_dir}")
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
        
        # Step 5: List all output files
        print(f"\n📂 Final build artifacts in {args.output_dir}:")
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
        print(f"\nTo use in Docker: COPY {library_path.name} /build/")
        print(f"To use locally: import aot_bridge; aot_bridge.ensure_initialized()")
        print("=" * 70)
        
        return 0
        
    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ AOT BUILD FAILED")
        print("=" * 70)
        print(f"Error: {e}", file=sys.stderr)
        traceback.print_exc()
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())