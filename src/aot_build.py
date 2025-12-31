#!/usr/bin/env python3
"""
AOT Compilation Script - Refactored to Use Shared Functions
============================================================

Imports ALL function implementations from numba_functions_shared.py
and exports them to AOT-compiled .so module.

NO DUPLICATION: Mathematical logic lives in ONE place.
"""

import os
import sys
import sysconfig
import multiprocessing
from pathlib import Path
import numpy as np
from numba.pycc import CC
from numba import types

# Import ALL 23 functions from the shared module
from numba_functions_shared import (
    sanitize_array_numba,
    sanitize_array_numba_parallel,
    sma_loop,
    sma_loop_parallel,
    ema_loop,
    ema_loop_alpha,
    rng_filter_loop,
    smooth_range,
    kalman_loop,
    vwap_daily_loop,
    rolling_std_welford,
    rolling_std_welford_parallel,
    calc_mmh_worm_loop,
    calc_mmh_value_loop,
    calc_mmh_momentum_loop,
    rolling_mean_numba,
    rolling_mean_numba_parallel,
    rolling_min_max_numba,
    rolling_min_max_numba_parallel,
    calculate_ppo_core,
    calculate_rsi_core,
    vectorized_wick_check_buy,
    vectorized_wick_check_sell,
)

# Aggressive optimization flags
os.environ.update({
    'NUMBA_OPT': '3',
    'NUMBA_LOOP_VECTORIZE': '1',
    'NUMBA_CPU_NAME': 'native',
    'NUMBA_CPU_FEATURES': '+avx2,+fma',
    'NUMBA_WARNINGS': '0',
    'NUMBA_DISABLE_JIT': '0',
    'NUMBA_THREADING_LAYER': 'omp',
})


def get_output_filename(base_name: str) -> str:
    """Generate platform-specific output filename"""
    ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')
    
    if not ext_suffix:
        import platform
        system = platform.system()
        if system == 'Windows':
            ext_suffix = '.pyd'
        elif system == 'Darwin':
            ext_suffix = '.dylib'
        else:
            py_version = f"{sys.version_info.major}{sys.version_info.minor}"
            ext_suffix = f".cpython-{py_version}-x86_64-linux-gnu.so"
    
    return f"{base_name}{ext_suffix}"


def compile_module():
    """
    Compile all 23 functions to AOT shared library
    
    Uses function implementations from numba_functions_shared.py
    """
    output_dir = Path(__file__).parent
    module_name = 'macd_aot_compiled'
    
    cc = CC(module_name)
    cc.output_dir = str(output_dir)
    cc.verbose = True
    
    # Enable parallel compilation
    n_jobs = max(1, multiprocessing.cpu_count() - 1)
    os.environ['NUMBA_NUM_THREADS'] = str(n_jobs)
    os.environ['OMP_NUM_THREADS'] = str(min(4, n_jobs))
    
    expected_output = output_dir / get_output_filename(module_name)
    
    print("=" * 70)
    print("AOT COMPILATION STARTING (REFACTORED)")
    print("=" * 70)
    print(f"📦 Module name: {module_name}")
    print(f"📂 Output directory: {output_dir.absolute()}")
    print(f"🎯 Expected output: {expected_output}")
    print(f"🔧 Compilation threads: {n_jobs}")
    print(f"🧵 OpenMP threads: {os.environ['OMP_NUM_THREADS']}")
    print(f"🐍 Python version: {sys.version_info.major}.{sys.version_info.minor}")
    print(f"✅ Using shared functions from: numba_functions_shared.py")
    print("=" * 70)
    
    # ========================================================================
    # FUNCTION EXPORTS - Import from shared module, export with CC
    # ========================================================================
    
    # 1-2. Sanitization
    cc.export('sanitize_array_numba', 'f8[:](f8[:], f8)')(sanitize_array_numba)
    cc.export('sanitize_array_numba_parallel', 'f8[:](f8[:], f8)')(sanitize_array_numba_parallel)
    
    # 3-4. SMA
    cc.export('sma_loop', 'f8[:](f8[:], i4)')(sma_loop)
    cc.export('sma_loop_parallel', 'f8[:](f8[:], i4)')(sma_loop_parallel)
    
    # 5-6. EMA
    cc.export('ema_loop', 'f8[:](f8[:], f8)')(ema_loop)
    cc.export('ema_loop_alpha', 'f8[:](f8[:], f8)')(ema_loop_alpha)
    
    # 7-9. Filters
    cc.export('rng_filter_loop', 'f8[:](f8[:], f8[:])')(rng_filter_loop)
    cc.export('smooth_range', 'f8[:](f8[:], i4, i4)')(smooth_range)
    cc.export('kalman_loop', 'f8[:](f8[:], i4, f8, f8)')(kalman_loop)
    
    # 10. VWAP
    cc.export('vwap_daily_loop', 'f8[:](f8[:], f8[:], f8[:], f8[:], i8[:])')(vwap_daily_loop)
    
    # 11-12. Rolling Standard Deviation
    cc.export('rolling_std_welford', 'f8[:](f8[:], i4, f8)')(rolling_std_welford)
    cc.export('rolling_std_welford_parallel', 'f8[:](f8[:], i4, f8)')(rolling_std_welford_parallel)
    
    # 13-15. MMH Components
    cc.export('calc_mmh_worm_loop', 'f8[:](f8[:], f8[:], i4)')(calc_mmh_worm_loop)
    cc.export('calc_mmh_value_loop', 'f8[:](f8[:], i4)')(calc_mmh_value_loop)
    cc.export('calc_mmh_momentum_loop', 'f8[:](f8[:], i4)')(calc_mmh_momentum_loop)
    
    # 16-17. Rolling Mean
    cc.export('rolling_mean_numba', 'f8[:](f8[:], i4)')(rolling_mean_numba)
    cc.export('rolling_mean_numba_parallel', 'f8[:](f8[:], i4)')(rolling_mean_numba_parallel)
    
    # 18-19. Rolling Min/Max
    cc.export(
        'rolling_min_max_numba',
        types.Tuple((types.float64[:], types.float64[:]))(types.float64[:], types.int32)
    )(rolling_min_max_numba)
    
    cc.export(
        'rolling_min_max_numba_parallel',
        types.Tuple((types.float64[:], types.float64[:]))(types.float64[:], types.int32)
    )(rolling_min_max_numba_parallel)
    
    # 20-21. Oscillators
    cc.export(
        'calculate_ppo_core',
        types.Tuple((types.float64[:], types.float64[:]))(
            types.float64[:], types.int32, types.int32, types.int32
        )
    )(calculate_ppo_core)
    
    cc.export('calculate_rsi_core', 'f8[:](f8[:], i4)')(calculate_rsi_core)
    
    # 22-23. Pattern Recognition
    cc.export('vectorized_wick_check_buy', 'b1[:](f8[:], f8[:], f8[:], f8[:], f8)')(vectorized_wick_check_buy)
    cc.export('vectorized_wick_check_sell', 'b1[:](f8[:], f8[:], f8[:], f8[:], f8)')(vectorized_wick_check_sell)
    
    # ========================================================================
    # COMPILATION
    # ========================================================================
    
    print(f"\n🔨 Compiling {module_name} with all 23 functions...")
    print("📋 Function list:")
    functions = [
        "1. sanitize_array_numba", "2. sanitize_array_numba_parallel",
        "3. sma_loop", "4. sma_loop_parallel",
        "5. ema_loop", "6. ema_loop_alpha",
        "7. rng_filter_loop", "8. smooth_range",
        "9. kalman_loop", "10. vwap_daily_loop",
        "11. rolling_std_welford", "12. rolling_std_welford_parallel",
        "13. calc_mmh_worm_loop", "14. calc_mmh_value_loop",
        "15. calc_mmh_momentum_loop",
        "16. rolling_mean_numba", "17. rolling_mean_numba_parallel",
        "18. rolling_min_max_numba", "19. rolling_min_max_numba_parallel",
        "20. calculate_ppo_core", "21. calculate_rsi_core",
        "22. vectorized_wick_check_buy", "23. vectorized_wick_check_sell",
    ]
    for func in functions:
        print(f"   ✓ {func}")
    
    print("\n⏳ Starting compilation (this may take 1-3 minutes)...")
    
    try:
        cc.compile()
        
        # Find generated .so file
        so_files = list(output_dir.glob(f"{module_name}*.so")) + \
                   list(output_dir.glob(f"{module_name}*.pyd")) + \
                   list(output_dir.glob(f"{module_name}*.dylib"))
        
        if so_files:
            output = so_files[0]
            size_kb = output.stat().st_size / 1024
            
            print("\n" + "=" * 70)
            print("✅ COMPILATION SUCCESSFUL")
            print("=" * 70)
            print(f"📦 Output file: {output.name}")
            print(f"📂 Absolute path: {output.absolute()}")
            print(f"💾 File size: {size_kb:.1f} KB")
            print(f"🎯 Functions compiled: 23/23")
            print(f"🔗 Source: numba_functions_shared.py (single source of truth)")
            print("=" * 70)
            
            if output.exists() and output.stat().st_size > 0:
                print("✅ File verification: PASSED")
                return True
            else:
                print("❌ File verification: FAILED (file empty or unreadable)")
                return False
        else:
            print("\n" + "=" * 70)
            print("❌ COMPILATION FAILED")
            print("=" * 70)
            print("❌ No output file generated")
            print(f"📂 Searched in: {output_dir.absolute()}")
            print(f"🔍 Expected pattern: {module_name}*.so/pyd/dylib")
            
            all_files = list(output_dir.glob("*"))
            print(f"\n🔍 Files in output directory:")
            for f in all_files[:10]:
                print(f"   - {f.name}")
            
            print("=" * 70)
            return False
            
    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ COMPILATION ERROR")
        print("=" * 70)
        print(f"❌ Exception: {type(e).__name__}")
        print(f"❌ Message: {e}")
        print("=" * 70)
        
        import traceback
        print("\n🔍 Full traceback:")
        traceback.print_exc()
        
        return False


if __name__ == '__main__':
    print("🚀 AOT Build Script (Refactored)")
    print(f"🐍 Python: {sys.version}")
    print(f"📦 NumPy: {np.__version__}")
    
    success = compile_module()
    
    if success:
        print("\n✅ Build completed successfully!")
        print("💡 All functions sourced from numba_functions_shared.py")
        print("💡 No code duplication - single source of truth maintained")
        sys.exit(0)
    else:
        print("\nâŒ Build failed!")
        print("💡 Falling back to JIT compilation will be used automatically")
        sys.exit(1)