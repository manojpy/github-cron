"""
Loader bridge:
1. Try to import the AOT compiled .so
2. On failure -> re-export the JIT versions from macd_unified
"""
from __future__ import annotations
import warnings
from pathlib import Path

_SO_PATH = Path(__file__).parent / "_macd_aot.so"

# ------------------------------------------------------------------
# Default fallback (JIT) – import **only** when we are not building
# ------------------------------------------------------------------
_FALLBACK = {}
try:
    # This import will succeed only when macd_unified is **fully** loaded
    from macd_unified import (                       # noqa
        _sanitize_array_numba,
        _sanitize_array_numba_parallel,
        _sma_loop,
        _sma_loop_parallel,
        _ema_loop,
        _ema_loop_alpha,
        _kalman_loop,
        _vwap_daily_loop,
        _rng_filter_loop,
        _smooth_range,
        _calc_mmh_worm_loop,
        _calc_mmh_value_loop,
        _calc_mmh_momentum_loop,
        _rolling_std_welford,
        _rolling_std_welford_parallel,
        _rolling_mean_numba,
        _rolling_mean_numba_parallel,
        _rolling_min_max_numba,
        _rolling_min_max_numba_parallel,
        _calculate_ppo_core,
        _calculate_rsi_core,
        _vectorized_wick_check_buy,
        _vectorized_wick_check_sell,
    )
    _FALLBACK.update(locals())
except Exception as exc:
    # If we are inside the AOT build this will fail – that is **expected**
    # The AOT build does **not** use the bridge anyway
    _FALLBACK = {}

# ------------------------------------------------------------------
# Attempt AOT load
# ------------------------------------------------------------------
_AOT = {}
if _SO_PATH.exists():
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("_macd_aot", _SO_PATH)
        _aot_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_aot_mod)
        _AOT.update({k: getattr(_aot_mod, k) for k in _FALLBACK if hasattr(_aot_mod, k)})
        warnings.warn("✅ Using AOT compiled _macd_aot.so", stacklevel=1)
    except Exception as exc:
        warnings.warn(f"⚠️  AOT .so load failed ({exc}) – falling back to JIT", stacklevel=1)

# ------------------------------------------------------------------
# Public API – pick AOT if available else JIT
# ----------------------------------------------------------------__
for _name in list(_FALLBACK):
    globals()[_name] = _AOT.get(_name, _FALLBACK[_name])
    __all__ = list(_FALLBACK)   # type: ignore
