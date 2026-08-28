"""
AOT Bridge Module - Runtime AOT/JIT Function Dispatcher (OPTIMIZED)
====================================================================

Provides transparent fallback between AOT-compiled (.so) and JIT-compiled
functions with zero-overhead dispatch via lookup dictionary.

Performance: ~5-6 seconds faster than wrapper-based approach.

FUNCTION LIST: driven entirely by AOT_FUNCTION_NAMES in
aot_function_registry.py -- the JIT-fallback dict and the AOT dispatch dict
are both built from that single list, so adding a new function only means:
(1) add its name to aot_function_registry.py, (2) add its @njit function to
numba_functions_shared.py, (3) add its thin wrapper function below. The
wrapper functions themselves stay hand-written on purpose (they're the
typed public API other modules import from), but the completeness check at
the bottom of this file raises immediately at import time if a wrapper is
ever forgotten -- instead of failing later, silently or otherwise.
"""

import os
import sys
import platform
import warnings
from pathlib import Path
from typing import Optional, Any, Callable, Dict, Tuple

import importlib.util
import numpy as np

from aot_function_registry import AOT_FUNCTION_NAMES as REQUIRED_AOT_FUNCTIONS

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


def get_library_extension() -> str:
    
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
    global _aot_module, _using_aot

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
    """Initialize JIT fallback functions from numba_functions_shared.
    Driven by REQUIRED_AOT_FUNCTIONS (from aot_function_registry.py) --
    adding a function there is enough; nothing here needs to change."""
    global _jit_functions, _fallback_reason

    try:
        import numba_functions_shared as _shared

        missing = [name for name in REQUIRED_AOT_FUNCTIONS if not hasattr(_shared, name)]
        if missing:
            raise ImportError(
                f"numba_functions_shared is missing {len(missing)} function(s) listed "
                f"in aot_function_registry.py: {missing}"
            )

        _jit_functions = {name: getattr(_shared, name) for name in REQUIRED_AOT_FUNCTIONS}

    except ImportError as e:
        _fallback_reason = f"JIT fallback failed: {e}"
        raise RuntimeError(f"Cannot initialize JIT fallback: {e}")


def _build_aot_dispatch() -> Dict[str, Callable]:
    """Build the AOT dispatch dict. Isolated in its own function so
    ensure_initialized() can safely try/except around it as a defense-in-depth
    safety net (belt-and-suspenders on top of the REQUIRED_AOT_FUNCTIONS
    check in initialize_aot())."""
    return {name: getattr(_aot_module, name) for name in REQUIRED_AOT_FUNCTIONS}


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
                f"aot_function_registry.py matches EXPORT_CONFIG "
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
# ----------------------------------------------------------------------------
# Hand-written on purpose: this is the typed public API other modules import
# from (`from aot_bridge import rolling_min_max_numba, ...`). Auto-generating
# these via globals()/lambda would erase type hints, argument names, and
# docstrings like ema_loop_pine's -- not worth it to save a few lines.
# If you add a name to aot_function_registry.py, add its wrapper here; the
# completeness check at the bottom of this file will raise at import time
# if you forget.
# ============================================================================

def sanitize_array_numba(arr: np.ndarray, default: float) -> np.ndarray:
    return _dispatch['sanitize_array_numba'](arr, default)

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

def rolling_mean_numba(data: np.ndarray, period: int) -> np.ndarray:
    return _dispatch['rolling_mean_numba'](data, period)

def rolling_min_max_numba(arr: np.ndarray, period: int) -> Tuple[np.ndarray, np.ndarray]:
    return _dispatch['rolling_min_max_numba'](arr, period)

def calculate_ppo_core(close: np.ndarray, fast: int, slow: int, signal: int) -> Tuple[np.ndarray, np.ndarray]:
    return _dispatch['calculate_ppo_core'](close, fast, slow, signal)

def calculate_rsi_core(close: np.ndarray, period: int) -> np.ndarray:
    return _dispatch['calculate_rsi_core'](close, period)

def true_range_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    return _dispatch['true_range_numba'](high, low, close)

def calculate_atr_rma(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    return _dispatch['calculate_atr_rma'](high, low, close, period)

def calculate_adx_core(high: np.ndarray, low: np.ndarray, close: np.ndarray, di_length: int, adx_length: int) -> np.ndarray:
    return _dispatch['calculate_adx_core'](high, low, close, di_length, adx_length)

def percentile_rank_numba(arr: np.ndarray, i: int, lookback: int, min_history: int, allow_zero: bool) -> float:
    """Returns NaN (not None) where the caller's window is invalid -- the
    Optional[float]/None translation happens in the caller, not here, since
    Numba functions can't return None."""
    return _dispatch['percentile_rank_numba'](arr, i, lookback, min_history, allow_zero)


# ============================================================================
# COMPLETENESS CHECK -- catches a forgotten wrapper at import time instead of
# an ImportError at some later call site.
# ============================================================================

_missing_wrappers = [name for name in REQUIRED_AOT_FUNCTIONS if name not in globals()]
if _missing_wrappers:
    raise RuntimeError(
        f"aot_bridge.py: {_missing_wrappers} are listed in aot_function_registry.py "
        f"but have no wrapper function defined above. Add:\n"
        f"    def {_missing_wrappers[0]}(...):\n"
        f"        return _dispatch['{_missing_wrappers[0]}'](...)"
    )


# ============================================================================
# MODULE EXPORTS -- derived from REQUIRED_AOT_FUNCTIONS, so a name added to
# the registry is automatically exported once its wrapper exists above.
# ============================================================================

__all__ = [
    'ensure_initialized',
    'is_using_aot',
    'get_fallback_reason',
    'requires_warmup',
] + REQUIRED_AOT_FUNCTIONS

# Auto-initialize on import
try:
    ensure_initialized()
except Exception as e:
    warnings.warn(f"Auto-initialization failed: {e}. Call ensure_initialized() manually.")
