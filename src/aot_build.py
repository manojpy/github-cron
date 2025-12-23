#!/usr/bin/env python3
"""
Compile *all* Numba-decorated helpers to a single shared object ahead-of-time.
Run this **inside the container that will run the bot** (same glibc / arch).
"""
from __future__ import annotations

import os
import sys
import shutil
import subprocess
import inspect
import types
from pathlib import Path
from typing import List, Tuple

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------
SRC_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
BUILD_DIR    = PROJECT_ROOT / "build" / "aot"
BLACKLIST    = {"warmup_numba"}   # functions you do *not* want AOT

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def ensure_clean_build_dir() -> None:
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

def discover_numba_functions() -> List[Tuple[str, str]]:
    """
    Import macd_unified and return [(module_name, func_name), ...] for every
    function decorated with @njit / @jit / @guvectorize etc.
    """
    sys.path.insert(0, str(SRC_DIR))
    import macd_unified as mu

    exported: List[Tuple[str, str]] = []
    for name, obj in inspect.getmembers(mu, inspect.isfunction):
        if name in BLACKLIST:
            continue
        # Numba stores its metadata in a private attribute
        if hasattr(obj, "py_func") or hasattr(obj, "_sig"):
            exported.append(("macd_unified", name))
    sys.path.remove(str(SRC_DIR))
    return exported

def compile_functions(func_list: List[Tuple[str, str]]) -> None:
    """
    Use numba.pycc.CC to build one shared object that contains every function.
    """
    try:
        from numba.pycc import CC
    except ImportError as exc:  # pragma: no cover
        print("numba.pycc not available – AOT impossible on this arch", exc)
        sys.exit(1)

    cc = CC("aot_compiled", str(BUILD_DIR))
    cc.verbose = True

    # Bring the module into scope so Numba can resolve signatures
    sys.path.insert(0, str(SRC_DIR))
    import macd_unified as mu

    for mod, fname in func_list:
        if mod != "macd_unified":
            raise RuntimeError("Only macd_unified supported")
        func = getattr(mu, fname)
        # Export under the *original* name so aot_bridge can find it
        cc.export(fname, func._sig)(func)  # type: ignore[attr-defined]

    cc.compile()
    print("✅ AOT compilation finished – .so should be in", BUILD_DIR)

def verify_artifact() -> None:
    so_files = list(BUILD_DIR.glob("aot_compiled*.so"))
    if not so_files:
        print("❌ No shared object produced")
        sys.exit(1)
    print("✅ Detected", so_files[0])

# ------------------------------------------------------------------
# Entry-point
# ------------------------------------------------------------------
if __name__ == "__main__":
    ensure_clean_build_dir()
    numba_funcs = discover_numba_functions()
    if not numba_funcs:
        print("⚠️  No Numba functions discovered – nothing to compile")
        sys.exit(0)
    print(f"🔍 Discovered {len(numba_funcs)} Numba functions")
    compile_functions(numba_funcs)
    verify_artifact()
