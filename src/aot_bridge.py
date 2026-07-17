"""
AOT Bridge Module - Runtime AOT/JIT Function Dispatcher (OPTIMIZED)
====================================================================

Provides transparent fallback between AOT-compiled (.so) and JIT-compiled
functions with zero-overhead dispatch via lookup dictionary.

Performance: ~5-6 seconds faster than wrapper-based approach.
"""

import os
import sys
import platform
import warnings
from pathlib import Path
from typing import Optional, Any, Callable, Dict, Tuple

import importlib.util
import numpy as np

# Needed on every run (including the AOT-success path) to detect a stale .so.
# Deliberately imported from the tiny, numba-free aot_version.py -- NOT from
# numba_functions_shared.py -- so this check never forces a `numba` import
# when AOT is active. See aot_version.py for the full rationale.
try:
    from aot_version import SOURCE_VERSION as _SHARED_SOURCE_VERSION
except ImportError:
    _SHARED_SOURCE_VERSION = None

# Suppress Numba/pcparser warnings at import time
#warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*parsing methods must have __doc__.*")
#warnings.filterwarnings("ignore", category=DeprecationWarning)
#warnings.filterwarnings("ignore", message=".*inspect.getargspec.*")

# Global state
_aot_module: Optional[Any] = None
_using_aot: bool = False
_fallback_reason: Optional[str] = None
_initialized: bool = False

# High-performance dispatch dictionary (set at init)
_dispatch: Dict[str, Callable] = {}

# JIT function storage (for fallback)
_jit_functions: Dict[str, Callable] = {}

# ============================================================================
# REQUIRED FUNCTIONS
# ============================================================================

REQUIRED_AOT_FUNCTIONS = [
    'sanitize_array_numba',
    'sanitize_array_numba_parallel',
    'ema_loop',
    'ema_loop_alpha',
    'ema_loop_pine',
    'kalman_loop',
    'vwap_daily_loop_safe',
    'calc_mmh_worm_loop',
    'calc_mmh_value_loop',
    'calc_mmh_momentum_loop',
    'rolling_std',
    'rolling_mean_numba',
    'calc_mmh_momentum_smoothing',
    'rolling_min_max_numba',
    'calculate_ppo_core',
    'calculate_rsi_core',
    'calculate_atr_rma',
    'calculate_adx_core',
]


def get_library_extension() -> str:
    """Extension for compiled Python extension modules.

    Mirrors exactly what aot_build.py's compile_module() produces: .pyd on
    Windows, .so everywhere else. macOS Python C-extensions use .so (not
    .dylib) even though the underlying binary is a Mach-O dylib -- using
    .dylib here would mean find_aot_library() never finds a real AOT build
    on macOS and silently falls back to JIT on every run.
    """
    return ".pyd" if platform.system() == "Windows" else ".so"


def find_aot_library(module_name: str = "macd_aot_compiled") -> Optional[Path]:
    """Search for compiled AOT library in standard locations"""
    extension = get_library_extension()

    search_paths = []
    env_path = os.getenv("AOT_LIB_PATH")
    if env_path:
        search_paths.append(Path(env_path))
    search_paths += [Path.cwd(), Path(__file__).parent]

    for search_dir in search_paths:
        if not search_dir.exists():
            continue

        # Try exact matches first
        for name in [f"{module_name}{extension}", f"{module_name}.cpython-311{extension}"]:
            p = search_dir / name
            if p.exists():
                return p

        # Wildcard fallback for ABI-tagged names. FIX: glob() order is not
        # guaranteed, so if more than one matching build artifact is ever left in
        # the same directory (e.g. a stale file from a prior build), the old
        # `found[0]` could silently load the wrong (stale) library. Sort by mtime
        # descending so the most recently built .so always wins.
        found = list(search_dir.glob(f"{module_name}*{extension}"))
        if found:
            found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return found[0]

    return None


def load_aot_module(library_path: Path, module_name: str = "macd_aot_compiled") -> Optional[Any]:
    """Load AOT compiled module from shared library"""
    try:
        spec = importlib.util.spec_from_file_location(module_name, str(library_path))
        if spec is None or spec.loader is None:
            warnings.warn(f"Cannot create import spec for {library_path}")
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sys.modules[module_name] = mod
        return mod
    except Exception as e:
        warnings.warn(f"Failed to import AOT module {library_path}: {e}")
        return None


def check_aot_version_stamp(library_path: Path, module_name: str) -> Tuple[bool, Optional[str]]:
    """Compare the `.version` sidecar written by aot_build.py against the live
    SOURCE_VERSION in numba_functions_shared.py.

    Returns (ok, reason). A missing sidecar is treated as unverifiable-but-OK
    (not fatal), so upgrading aot_bridge.py doesn't break a build made before
    version stamping existed -- but any mismatch is treated as a hard failure,
    because that's exactly the "stale .so silently running old logic" scenario
    this check exists to catch.
    """
    version_path = library_path.parent / f"{module_name}.version"
    if not version_path.exists():
        return True, None
    if _SHARED_SOURCE_VERSION is None:
        return True, None

    stamped = version_path.read_text().strip()
    if stamped != _SHARED_SOURCE_VERSION:
        return False, (
            f"AOT library is STALE: compiled from SOURCE_VERSION '{stamped}' but the "
            f"current source is '{_SHARED_SOURCE_VERSION}'. Rebuild with aot_build.py "
            f"and redeploy the .so -- falling back to JIT in the meantime so results "
            f"stay correct."
        )
    return True, None


def initialize_aot(module_name: str = "macd_aot_compiled") -> Tuple[bool, Optional[str]]:
    """Attempt to initialize AOT module and verify ALL required functions exist.

    Verifying every function in REQUIRED_AOT_FUNCTIONS (rather than a small
    sample) guarantees that once this returns success=True, the dispatch
    dictionary in ensure_initialized() can safely reference every AOT
    attribute directly without risking an AttributeError from a stale build.
    """
    global _aot_module, _using_aot, _fallback_reason

    library_path = find_aot_library(module_name)
    if library_path is None:
        return False, f"AOT library {module_name}{get_library_extension()} not found"

    version_ok, version_reason = check_aot_version_stamp(library_path, module_name)
    if not version_ok:
        warnings.warn(version_reason)
        return False, version_reason

    _aot_module = load_aot_module(library_path, module_name)
    if _aot_module is None:
        return False, f"Failed to import AOT module at {library_path}"

    # Verify every function the dispatch table needs, not just a handful.
    missing = [fn for fn in REQUIRED_AOT_FUNCTIONS if not hasattr(_aot_module, fn)]
    if missing:
        return False, f"AOT library missing {len(missing)} function(s): {missing}"

    _using_aot = True
    return True, None


def initialize_jit_fallback() -> None:
    """Initialize JIT fallback functions from numba_functions_shared"""
    global _jit_functions, _fallback_reason

    try:
        # Import all 18 functions (already cached by Python)
        from numba_functions_shared import (
            sanitize_array_numba,
            sanitize_array_numba_parallel,
            ema_loop,
            ema_loop_alpha,
            ema_loop_pine,
            kalman_loop,
            vwap_daily_loop_safe,
            calc_mmh_worm_loop,
            calc_mmh_value_loop,
            calc_mmh_momentum_loop,
            rolling_std,
            rolling_mean_numba,
            calc_mmh_momentum_smoothing,
            rolling_min_max_numba,
            calculate_ppo_core,
            calculate_rsi_core,
            calculate_atr_rma,
            calculate_adx_core,
        )

        # Store in dictionary for dispatch
        _jit_functions = {
            'sanitize_array_numba': sanitize_array_numba,
            'sanitize_array_numba_parallel': sanitize_array_numba_parallel,
            'ema_loop': ema_loop,
            'ema_loop_alpha': ema_loop_alpha,
            'ema_loop_pine': ema_loop_pine,
            'kalman_loop': kalman_loop,
            'vwap_daily_loop_safe': vwap_daily_loop_safe,
            'calc_mmh_worm_loop': calc_mmh_worm_loop,
            'calc_mmh_value_loop': calc_mmh_value_loop,
            'calc_mmh_momentum_loop': calc_mmh_momentum_loop,
            'rolling_std': rolling_std,
            'rolling_mean_numba': rolling_mean_numba,
            'calc_mmh_momentum_smoothing': calc_mmh_momentum_smoothing,
            'rolling_min_max_numba': rolling_min_max_numba,
            'calculate_ppo_core': calculate_ppo_core,
            'calculate_rsi_core': calculate_rsi_core,
            'calculate_atr_rma': calculate_atr_rma,
            'calculate_adx_core': calculate_adx_core,
        }

    except ImportError as e:
        _fallback_reason = f"JIT fallback failed: {e}"
        raise RuntimeError(f"Cannot initialize JIT fallback: {e}")


def _build_aot_dispatch() -> Dict[str, Callable]:
    """Build the AOT dispatch dict. Isolated in its own function so
    ensure_initialized() can safely try/except around it as a defense-in-depth
    safety net (belt-and-suspenders on top of the REQUIRED_AOT_FUNCTIONS
    check in initialize_aot())."""
    return {
        'sanitize_array_numba': _aot_module.sanitize_array_numba,
        'sanitize_array_numba_parallel': _aot_module.sanitize_array_numba_parallel,
        'ema_loop': _aot_module.ema_loop,
        'ema_loop_alpha': _aot_module.ema_loop_alpha,
        'ema_loop_pine': _aot_module.ema_loop_pine,
        'kalman_loop': _aot_module.kalman_loop,
        'vwap_daily_loop_safe': _aot_module.vwap_daily_loop_safe,
        'calc_mmh_worm_loop': _aot_module.calc_mmh_worm_loop,
        'calc_mmh_value_loop': _aot_module.calc_mmh_value_loop,
        'calc_mmh_momentum_loop': _aot_module.calc_mmh_momentum_loop,
        'rolling_std': _aot_module.rolling_std,
        'rolling_mean_numba': _aot_module.rolling_mean_numba,
        'calc_mmh_momentum_smoothing': _aot_module.calc_mmh_momentum_smoothing,
        'rolling_min_max_numba': _aot_module.rolling_min_max_numba,
        'calculate_ppo_core': _aot_module.calculate_ppo_core,
        'calculate_rsi_core': _aot_module.calculate_rsi_core,
        'calculate_atr_rma': _aot_module.calculate_atr_rma,
        'calculate_adx_core': _aot_module.calculate_adx_core,
    }


def ensure_initialized() -> None:
    """Initialize dispatch table with either AOT or JIT functions"""
    global _initialized, _fallback_reason, _using_aot, _dispatch

    if _initialized:
        return

    success, reason = initialize_aot()

    if success:

        try:
            _dispatch = _build_aot_dispatch()
            _using_aot = True
            _fallback_reason = None
        except AttributeError as e:
            warnings.warn(
                f"AOT module unexpectedly missing an attribute ({e}) despite "
                f"passing verification -- falling back to JIT. Check that "
                f"REQUIRED_AOT_FUNCTIONS in aot_bridge.py matches EXPORT_CONFIG "
                f"in numba_functions_shared.py."
            )
            _using_aot = False
            _fallback_reason = f"AOT dispatch build failed: {e}"
            initialize_jit_fallback()
            _dispatch = _jit_functions
    else:
        _fallback_reason = reason
        _using_aot = False
        initialize_jit_fallback()

        # Dispatch to JIT functions
        _dispatch = _jit_functions

    _initialized = True


def is_using_aot() -> bool:
    """Check if AOT compilation is active"""
    return _using_aot


def get_fallback_reason() -> Optional[str]:
    """Get reason for JIT fallback (if any)"""
    return _fallback_reason


def requires_warmup() -> bool:
    """Check if JIT warmup is needed (AOT doesn't need warmup)"""
    return not _using_aot


# ============================================================================
# HIGH-PERFORMANCE DISPATCH INTERFACE
# ============================================================================

def sanitize_array_numba(arr: np.ndarray, default: float) -> np.ndarray:
    return _dispatch['sanitize_array_numba'](arr, default)

def sanitize_array_numba_parallel(arr: np.ndarray, default: float) -> np.ndarray:
    return _dispatch['sanitize_array_numba_parallel'](arr, default)

def ema_loop(data: np.ndarray, alpha_or_period: float) -> np.ndarray:
    return _dispatch['ema_loop'](data, alpha_or_period)

def ema_loop_alpha(data: np.ndarray, alpha: float) -> np.ndarray:
    return _dispatch['ema_loop_alpha'](data, alpha)

def ema_loop_pine(data: np.ndarray, length: float) -> np.ndarray:
    """Pine-style EMA: seeds on first bar (nz(ema[1], src)). Used for Cirrus Cloud."""
    return _dispatch['ema_loop_pine'](data, length)

def kalman_loop(src: np.ndarray, length: int, R: float, Q: float) -> np.ndarray:
    return _dispatch['kalman_loop'](src, length, R, Q)

def vwap_daily_loop_safe(hlc3: np.ndarray, volumes: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    return _dispatch['vwap_daily_loop_safe'](hlc3, volumes, timestamps)

def calc_mmh_worm_loop(close_arr: np.ndarray, sd_arr: np.ndarray, rows: int) -> np.ndarray:
    return _dispatch['calc_mmh_worm_loop'](close_arr, sd_arr, rows)

def calc_mmh_value_loop(temp_arr: np.ndarray, min_med: np.ndarray, max_med: np.ndarray, rows: int) -> np.ndarray:
    return _dispatch['calc_mmh_value_loop'](temp_arr, min_med, max_med, rows)

def calc_mmh_momentum_loop(momentum_arr: np.ndarray, rows: int) -> np.ndarray:
    return _dispatch['calc_mmh_momentum_loop'](momentum_arr, rows)

def calc_mmh_momentum_smoothing(momentum: np.ndarray, rows: int) -> np.ndarray:
    return _dispatch['calc_mmh_momentum_smoothing'](momentum, rows)

def rolling_std(close: np.ndarray, period: int, responsiveness: float) -> np.ndarray:
    return _dispatch['rolling_std'](close, period, responsiveness)

def rolling_mean_numba(data: np.ndarray, period: int) -> np.ndarray:
    return _dispatch['rolling_mean_numba'](data, period)

def rolling_min_max_numba(arr: np.ndarray, period: int) -> Tuple[np.ndarray, np.ndarray]:
    return _dispatch['rolling_min_max_numba'](arr, period)

def calculate_ppo_core(close: np.ndarray, fast: int, slow: int, signal: int) -> Tuple[np.ndarray, np.ndarray]:
    return _dispatch['calculate_ppo_core'](close, fast, slow, signal)

def calculate_rsi_core(close: np.ndarray, period: int) -> np.ndarray:
    return _dispatch['calculate_rsi_core'](close, period)

def calculate_atr_rma(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    return _dispatch['calculate_atr_rma'](high, low, close, period)

def calculate_adx_core(high: np.ndarray, low: np.ndarray, close: np.ndarray, di_length: int, adx_length: int) -> np.ndarray:
    return _dispatch['calculate_adx_core'](high, low, close, di_length, adx_length)


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Initialization
    'ensure_initialized',
    'is_using_aot',
    'get_fallback_reason',
    'requires_warmup',

    # Sanitization
    'sanitize_array_numba',
    'sanitize_array_numba_parallel',

    # Moving Averages
    'ema_loop',
    'ema_loop_alpha',
    'ema_loop_pine',

    # Filters
    'kalman_loop',

    # Market Indicators
    'vwap_daily_loop_safe',

    # Statistical
    'rolling_std',
    'rolling_mean_numba',
    'calc_mmh_momentum_smoothing',
    'rolling_min_max_numba',

    # Oscillators
    'calculate_ppo_core',
    'calculate_rsi_core',

    # MMH Components
    'calc_mmh_worm_loop',
    'calc_mmh_value_loop',
    'calc_mmh_momentum_loop',

    # Pattern Recognition
    'calculate_atr_rma',
    'calculate_adx_core',
]

# Auto-initialize on import
try:
    ensure_initialized()
except Exception as e:
    warnings.warn(f"Auto-initialization failed: {e}. Call ensure_initialized() manually.")
