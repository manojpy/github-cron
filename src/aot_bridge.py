"""
AOT Bridge Module - Runtime AOT/JIT Function Dispatcher
========================================================
Dynamically routes function calls to AOT-compiled binaries (.so/.pyd)
or falls back to JIT-compiled Numba functions from numba_functions_shared.py.
"""

import os
import sys
import platform
import logging
import importlib.util
from pathlib import Path
from typing import Optional, Any

# Configure logging for the bridge
logger = logging.getLogger("aot_bridge")

# Global state
_aot_module: Optional[Any] = None
_using_aot: bool = False
_fallback_reason: Optional[str] = None
_initialized: bool = False

# Import shared functions and config for JIT fallback
try:
    import numba_functions_shared as shared
    from numba_functions_shared import EXPORT_CONFIG
except ImportError as e:
    logger.critical(f"CRITICAL: Could not import numba_functions_shared.py: {e}")
    raise


def get_library_extension() -> str:
    """Get platform-specific shared library extension."""
    system = platform.system()
    if system == "Windows":
        return ".pyd"
    return ".so"  # Linux, Darwin (macOS) use .so for Python extensions


def find_aot_library(module_name: str = "macd_aot_compiled") -> Optional[Path]:
    """Search for the compiled AOT library in common locations."""
    ext = get_library_extension()
    
    # Candidates including ABI-suffixed names (e.g. .cpython-311-x86_64-linux-gnu.so)
    search_dirs = [
        Path("."),
        Path(__file__).parent,
        Path(os.getcwd()),
    ]
    
    if os.getenv("AOT_LIB_PATH"):
        search_dirs.insert(0, Path(os.getenv("AOT_LIB_PATH")))

    for s_dir in search_dirs:
        if not s_dir.exists():
            continue
        # Search for any file starting with module_name and ending with ext
        for file in s_dir.glob(f"{module_name}*{ext}"):
            return file
    return None


def ensure_initialized():
    """
    Initializes the bridge by loading the AOT library and mapping functions.
    This is called automatically on import, but can be re-run if needed.
    """
    global _aot_module, _using_aot, _fallback_reason, _initialized
    
    if _initialized:
        return
    
    lib_path = find_aot_library()
    
    if lib_path:
        try:
            # Load the compiled module
            module_name = "macd_aot_compiled"
            spec = importlib.util.spec_from_file_location(module_name, str(lib_path))
            if spec and spec.loader:
                _aot_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(_aot_module)
                
                # Verify that all required functions exist in the AOT binary
                missing = [f for f in EXPORT_CONFIG.keys() if not hasattr(_aot_module, f)]
                
                if not missing:
                    _using_aot = True
                    logger.info(f"🚀 AOT logic initialized successfully from {lib_path.name}")
                else:
                    _fallback_reason = f"AOT binary missing functions: {', '.join(missing)}"
                    logger.warning(f"⚠️ {_fallback_reason}. Falling back to JIT.")
            else:
                _fallback_reason = "Failed to create spec/loader for AOT library."
        except Exception as e:
            _fallback_reason = f"Error loading AOT library: {str(e)}"
            logger.error(f"❌ {_fallback_reason}")
    else:
        _fallback_reason = "AOT library file not found (look for macd_aot_compiled.so/pyd)"
        logger.info("ℹ️ AOT library not found. Using JIT (numba_functions_shared).")

    # Map functions to the global namespace of this module
    _inject_functions()
    _initialized = True


def _inject_functions():
    """
    Dynamically injects functions into this module's globals.
    If AOT is available and has the function, it uses it.
    Otherwise, it uses the JIT version from numba_functions_shared.
    """
    for func_name in EXPORT_CONFIG.keys():
        aot_func = getattr(_aot_module, func_name, None) if _using_aot else None
        jit_func = getattr(shared, func_name, None)
        
        if aot_func:
            globals()[func_name] = aot_func
        elif jit_func:
            globals()[func_name] = jit_func
        else:
            logger.error(f"Function {func_name} not found in AOT or JIT sources!")


# ============================================================================
# PUBLIC API & METADATA
# ============================================================================

def is_using_aot() -> bool:
    return _using_aot

def get_fallback_reason() -> Optional[str]:
    return _fallback_reason

def requires_warmup() -> bool:
    """If using JIT, a warmup is recommended to avoid lag on the first run."""
    return not _using_aot

# Initialize on import
ensure_initialized()

# Define __all__ based on the shared config to ensure visibility to 'from aot_bridge import *'
__all__ = [
    'ensure_initialized', 
    'is_using_aot', 
    'get_fallback_reason', 
    'requires_warmup'
] + list(EXPORT_CONFIG.keys())