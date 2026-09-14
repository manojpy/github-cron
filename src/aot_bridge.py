"""
AOT Bridge Module — Runtime Compiled / JIT Function Dispatcher

Tries to import the Cython-compiled extension (``cython_functions``).
If it isn't available (not built, not installed, wrong platform), falls
back transparently to the Numba JIT functions in
``numba_functions_shared``.  Every downstream caller (indicators.py,
gates.py, alerts.py, macd_unified.py) keeps importing from *this*
module exactly as before — the dispatch table hides the backend.

What changed vs. the old pycc version
--------------------------------------
• ``find_aot_library``, ``load_aot_module``, ``check_aot_version_stamp``,
  ``get_library_extension`` — all removed.  Cython modules are ordinary
  importable packages; no path-searching or ``importlib`` tricks needed.
• Version-stamp checking removed (Cython wheels embed their own
  metadata; staleness is caught by ``pip install`` / Docker layer
  caching instead).
• ``initialize_aot`` → ``initialize_compiled`` (same contract).
• ``is_using_aot`` / ``requires_warmup`` kept as-is so
  ``macd_unified.py`` and ``indicators.py::warmup_if_needed`` need
  zero edits.
"""

import warnings
from typing import Optional, Callable, Dict, Tuple

import numpy as np

# Central function registry — single source of truth for every name
# that must exist in whichever backend is active.
try:
    from aot_meta import AOT_FUNCTION_NAMES as REQUIRED_AOT_FUNCTIONS
except ImportError:
    # Hardcoded fallback so the bridge still works if aot_meta is absent
    REQUIRED_AOT_FUNCTIONS = [
        "sanitize_array_numba",
        "rolling_mean_numba",
        "rolling_min_max_numba",
        "ema_loop",
        "ema_loop_pine",
        "ema_loop_alpha",
        "kalman_loop",
        "vwap_daily_loop_safe",
        "calculate_ppo_core",
        "calculate_rsi_core",
        "true_range_numba",
        "calculate_atr_rma",
        "calculate_adx_core",
        "percentile_rank_numba",
        "dynamic_flow_direction_loop",
    ]

# ──────────────────────────────────────────────────────────────────────
# Global state
# ──────────────────────────────────────────────────────────────────────
_compiled_module: Optional[object] = None   # the live Cython module (or None)
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
    except ImportError as exc:
        _fallback_reason = f"JIT fallback failed: {exc}"
        raise RuntimeError(f"Cannot initialise JIT fallback: {exc}")


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


# ──────────────────────────────────────────────────────────────────────
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
# Auto-initialise on import (same behaviour as before)
# ──────────────────────────────────────────────────────────────────────
try:
    ensure_initialized()
except Exception as exc:
    warnings.warn(
        f"Auto-initialisation failed: {exc}. "
        f"Call ensure_initialized() manually."
    )