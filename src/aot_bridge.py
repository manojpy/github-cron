"""
AOT Bridge Module — Runtime Compiled / JIT Function Dispatcher
Why SOURCE_VERSION still lives here (despite aot_bridge.py no longer
reading it on every run):
  • numba_functions_shared.py and cython_functions.pyx each assert
    their own export list against AOT_FUNCTION_NAMES at import time;
    editing a function body without bumping SOURCE_VERSION won't be
    caught by those assertions (they compare names, not bodies), so
    the bump is still the manual signal that a rebuild is required.
  • Callers that imported SOURCE_VERSION from aot_bridge in the old
    pycc days can still read it from here directly.
"""

import warnings
import numpy as np
from types import ModuleType
from typing import Optional, Callable, Dict, Tuple



# Central function registry — single source of truth lives in aot_meta.py
# (zero-import module, safe to import on every run).
from aot_meta import AOT_FUNCTION_NAMES as REQUIRED_AOT_FUNCTIONS

# ──────────────────────────────────────────────────────────────────────
# Global state
# ──────────────────────────────────────────────────────────────────────
_compiled_module: Optional[ModuleType] = None   # the live Cython module (or None)
_using_compiled: bool = False
_fallback_reason: Optional[str] = None
_initialized: bool = False

# High-performance dispatch table (populated once by ensure_initialized)
_dispatch: Dict[str, Callable] = {}

# JIT fallback storage
_jit_functions: Dict[str, Callable] = {}

# ──────────────────────────────────────────────────────────────────────
# Initialisation helpers
# ──────────────────────────────────────────────────────────────────────

def initialize_compiled() -> Tuple[bool, Optional[str]]:
    """Attempt to import the Cython-compiled extension."""
    global _compiled_module
    try:
        import cython_functions as _mod          # ← the compiled .so / .pyd

        missing = [fn for fn in REQUIRED_AOT_FUNCTIONS
                   if not hasattr(_mod, fn)]
        if missing:
            return False, (
                f"Cython module loaded but is missing "
                f"{len(missing)} function(s): {missing}"
            )
   
        # ── FIX (Priority 10): fail closed. A compiled .so that doesn't
        # expose SOURCE_VERSION, or an environment where aot_meta is
        # missing, means we CANNOT prove freshness — reject the Cython
        # backend and let JIT fallback take over.
        try:
            from aot_meta import SOURCE_VERSION as EXPECTED_VERSION
        except ImportError:
            return False, (
                "aot_meta.SOURCE_VERSION unavailable — cannot verify compiled "
                "artifact freshness. Refusing to accept Cython backend."
            )

        compiled_version = getattr(_mod, "SOURCE_VERSION", None)
        if compiled_version is None:
            return False, (
                "Compiled .so does not expose SOURCE_VERSION — cannot verify "
                "freshness. Rebuild the Cython extension."
            )
        if compiled_version != EXPECTED_VERSION:
            return False, (
                f"SOURCE_VERSION mismatch: compiled .so reports "
                f"'{compiled_version}' but aot_meta expects "
                f"'{EXPECTED_VERSION}'. Stale artifact rejected."
            )
        
        _compiled_module = _mod
        return True, None

    except ImportError as exc:
        return False, f"Cython module not importable: {exc}"
    except Exception as exc:
        return False, f"Cython module raised on import: {exc}"

def initialize_jit_fallback() -> None:
    """Import every required function from the Numba JIT module."""
    global _jit_functions, _fallback_reason
    try:
        import numba_functions_shared as _shared

        missing = [name for name in REQUIRED_AOT_FUNCTIONS
                   if not hasattr(_shared, name)]
        if missing:
            raise ImportError(
                f"numba_functions_shared is missing {len(missing)} "
                f"function(s) listed in the registry: {missing}"
            )
        _jit_functions = {
            name: getattr(_shared, name) for name in REQUIRED_AOT_FUNCTIONS
        }
    except Exception as exc:
        # Catch broadly: a drift assertion (AssertionError) or any other
        # import-time failure should surface with its original message,
        # not as an opaque AttributeError three frames later.
        _fallback_reason = f"JIT fallback failed: {exc}"
        raise RuntimeError(f"Cannot initialise JIT fallback: {exc}") from exc

def ensure_initialized() -> None:
    """Idempotent: pick Cython if available, else Numba JIT."""
    global _initialized, _fallback_reason, _using_compiled, _dispatch

    if _initialized:
        return

    success, reason = initialize_compiled()

    if success:
        try:
            _dispatch = {
                name: getattr(_compiled_module, name)
                for name in REQUIRED_AOT_FUNCTIONS
            }
            _using_compiled = True
            _fallback_reason = None
        except AttributeError as exc:
            warnings.warn(
                f"Cython module unexpectedly missing an attribute ({exc}) "
                f"despite passing verification — falling back to JIT."
            )
            _using_compiled = False
            _fallback_reason = f"Cython dispatch build failed: {exc}"
            initialize_jit_fallback()
            _dispatch = _jit_functions
    else:
        _fallback_reason = reason
        _using_compiled = False
        initialize_jit_fallback()
        _dispatch = _jit_functions

    _initialized = True


# ──────────────────────────────────────────────────────────────────────
# Introspection (called by macd_unified.py and indicators.py)
# ──────────────────────────────────────────────────────────────────────
def is_using_aot() -> bool:
    """True when the Cython backend is active."""
    return _using_compiled

def active_backend_module() -> str:
    if _using_compiled and _compiled_module is not None:
        return _compiled_module.__name__
    return "numba_functions_shared"
 
def get_fallback_reason() -> Optional[str]:
    """Why we fell back to JIT (None when Cython is active)."""
    return _fallback_reason


def requires_warmup() -> bool:
    """JIT needs a warm-up pass; Cython does not."""
    return not _using_compiled

# ──────────────────────────────────────────────────────────────────────
# HIGH-PERFORMANCE DISPATCH INTERFACE  (identical signatures to before)
# ──────────────────────────────────────────────────────────────────────
def sanitize_array_numba(arr: np.ndarray, default: float) -> np.ndarray:
    return _dispatch["sanitize_array_numba"](arr, default)

def ema_loop(data: np.ndarray, alpha_or_period: float) -> np.ndarray:
    return _dispatch["ema_loop"](data, alpha_or_period)

def ema_loop_alpha(data: np.ndarray, alpha: float) -> np.ndarray:
    return _dispatch["ema_loop_alpha"](data, alpha)

def ema_loop_pine(data: np.ndarray, length: float) -> np.ndarray:
    """Pine-style EMA: seeds on first bar (nz(ema[1], src))."""
    return _dispatch["ema_loop_pine"](data, length)

def kalman_loop(src: np.ndarray, length: int, R: float, Q: float) -> np.ndarray:
    return _dispatch["kalman_loop"](src, length, R, Q)

def vwap_daily_loop_safe(hlc3: np.ndarray, volumes: np.ndarray,
                         timestamps: np.ndarray) -> np.ndarray:
    return _dispatch["vwap_daily_loop_safe"](hlc3, volumes, timestamps)

def rolling_mean_numba(data: np.ndarray, period: int) -> np.ndarray:
    return _dispatch["rolling_mean_numba"](data, period)

def rolling_min_max_numba(arr: np.ndarray, period: int) -> Tuple[np.ndarray, np.ndarray]:
    return _dispatch["rolling_min_max_numba"](arr, period)

def calculate_ppo_core(close: np.ndarray, fast: int, slow: int,
                       signal: int) -> Tuple[np.ndarray, np.ndarray]:
    return _dispatch["calculate_ppo_core"](close, fast, slow, signal)

def calculate_rsi_core(close: np.ndarray, period: int) -> np.ndarray:
    return _dispatch["calculate_rsi_core"](close, period)

def true_range_numba(high: np.ndarray, low: np.ndarray,
                     close: np.ndarray) -> np.ndarray:
    return _dispatch["true_range_numba"](high, low, close)

def calculate_atr_rma(high: np.ndarray, low: np.ndarray,
                      close: np.ndarray, period: int) -> np.ndarray:
    return _dispatch["calculate_atr_rma"](high, low, close, period)

def calculate_adx_core(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                       di_length: int, adx_length: int) -> np.ndarray:
    return _dispatch["calculate_adx_core"](high, low, close, di_length, adx_length)

def percentile_rank_numba(arr: np.ndarray, i: int, lookback: int,
                          min_history: int, allow_zero: bool) -> float:
    return _dispatch["percentile_rank_numba"](arr, i, lookback, min_history, allow_zero)

def dynamic_flow_direction_loop(src: np.ndarray, basis: np.ndarray,
                                dist: np.ndarray,
                                factor: float) -> Tuple[np.ndarray, np.ndarray]:
    return _dispatch["dynamic_flow_direction_loop"](src, basis, dist, factor)


# ────────────────────────���─────────────────────────────────────────────
# COMPLETENESS CHECK — catches a forgotten wrapper at import time
# ──────────────────────────────────────────────────────────────────────
_missing_wrappers = [
    name for name in REQUIRED_AOT_FUNCTIONS if name not in globals()
]
if _missing_wrappers:
    raise RuntimeError(
        f"aot_bridge.py: {_missing_wrappers} are listed in the registry "
        f"but have no wrapper function defined above. Add:\n"
        f"    def {_missing_wrappers[0]}(...):\n"
        f"        return _dispatch['{_missing_wrappers[0]}'](...)"
    )


# ──────────────────────────────────────────────────────────────────────
# MODULE EXPORTS
# ──────────────────────────────────────────────────────────────────────
__all__ = [
    "ensure_initialized",
    "is_using_aot",
    "get_fallback_reason",
    "requires_warmup",
] + list(REQUIRED_AOT_FUNCTIONS)

# ──────────────────────────────────────────────────────────────────────
# Auto-initialise on import
# ──────────────────────────────────────────────────────────────────────
try:
    ensure_initialized()
except Exception as exc:
    # Fail at import time, not at the first indicator call.  A silent warn
    # here leaves _dispatch = {} and turns every downstream call into an
    # opaque `KeyError`, which is far harder to diagnose than a hard crash
    # at container start.
    raise RuntimeError(
        f"aot_bridge: neither Cython nor JIT backend could be initialised. "
        f"Last error: {exc}"
    ) from exc