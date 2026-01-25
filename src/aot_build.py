"""
AOT Build Script - Compiles Numba functions to native .so libraries
====================================================================
Automatically syncs with numba_functions_shared.py via EXPORT_CONFIG.
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

# Import shared function definitions and the centralized Export Config
try:
    import numba_functions_shared as shared
    from numba_functions_shared import EXPORT_CONFIG
except ImportError as e:
    print(f"ERROR: Cannot import numba_functions_shared: {e}", file=sys.stderr)
    sys.exit(1)


def compile_module(cc: CC, output_dir: Path, module_name: str) -> Path:
    """Execute the compilation process."""
    print(f"🛠️ Starting compilation for module: {module_name}...")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set the target filename (Numba handles extensions based on platform)
    output_base = output_dir / module_name
    
    # Trigger compilation
    # Note: cc.compile() generates the file in the current directory or specified path
    os.chdir(output_dir)
    cc.compile()
    
    # Find the produced library (handles platform differences .so, .pyd, .dylib)
    # and potential Python version tags (e.g. .cpython-311-x86_64-linux-gnu.so)
    ext = ".pyd" if platform.system() == "Windows" else ".so"
    libraries = list(output_dir.glob(f"{module_name}*{ext}"))
    
    if not libraries:
        raise FileNotFoundError(f"Compilation failed: No library files found matching {module_name}*{ext}")
    
    return libraries[0]


def verify_compilation(lib_path: Path):
    """Basic verification of the compiled library."""
    print(f"🧪 Verifying {lib_path.name}...")
    try:
        # Get the module name without extensions or ABI tags
        module_pure_name = lib_path.name.split('.')[0]
        sys.path.append(str(lib_path.parent))
        
        import importlib
        compiled_mod = importlib.import_module(module_pure_name)
        
        exported_count = 0
        for func_name in EXPORT_CONFIG.keys():
            if hasattr(compiled_mod, func_name):
                exported_count += 1
            else:
                print(f"⚠️ Warning: {func_name} not found in compiled library!")
        
        print(f"✅ Verification successful: {exported_count}/{len(EXPORT_CONFIG)} functions verified.")
    except Exception as e:
        print(f"❌ Verification failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Compile Numba functions to AOT library.")
    parser.add_argument("--module-name", default="macd_aot_compiled", help="Name of the output module")
    parser.add_argument("--output-dir", default=".", help="Directory for output files")
    parser.add_argument("--verify", action="store_true", help="Attempt to load and verify the library after build")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    cc = CC(args.module_name)
    cc.verbose = True

    print("=" * 70)
    print(f"🚀 AOT BUILDER - {platform.system()} {platform.machine()}")
    print("=" * 70)

    try:
        # Step 1: Register functions from the Shared Config
        print(f"📋 Registering {len(EXPORT_CONFIG)} functions from shared definitions...")
        
        for func_name, signature in EXPORT_CONFIG.items():
            # Get the function object from the shared module
            func_obj = getattr(shared, func_name, None)
            
            if func_obj is None:
                print(f"❌ Error: Function '{func_name}' defined in EXPORT_CONFIG but not found in numba_functions_shared.py")
                sys.exit(1)
            
            # Export function with its signature
            cc.export(func_name, signature)(func_obj)
            print(f"   + {func_name} ({signature})")

        # Step 2: Run Compilation
        library_path = compile_module(cc, output_dir, args.module_name)
        
        # Step 3: Show artifacts
        print(f"\n📂 Build artifacts in {output_dir}:")
        size_mb = library_path.stat().st_size / 1024 / 1024
        print(f"   {library_path.name} ({size_mb:.2f} MB)")
        
        # Step 4: Verification
        if args.verify:
            verify_compilation(library_path)
        
        print("\n" + "=" * 70)
        print("✅ AOT BUILD COMPLETE")
        print("=" * 70)
        print(f"Library: {library_path}")
        print("=" * 70)
        
        return 0
        
    except Exception as e:
        print(f"\n💥 FATAL ERROR during compilation:")
        print("-" * 40)
        traceback.print_exc()
        print("-" * 40)
        return 1


if __name__ == "__main__":
    sys.exit(main())