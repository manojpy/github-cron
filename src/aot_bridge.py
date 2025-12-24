"""
Loader bridge:
1. Try to import the AOT compiled _macd_aot.so
2. On failure -> re-export the JIT versions from numba_helpers
Provides diagnostics via summary().
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path

_SO_PATH = Path(__file__).parent / "_macd_aot.so"

# ------------------------------------------------------------------
# Names exposed by this bridge (runtime API)
# ------------------------------------------------------------------
__all__ = [
    "_sanitize_array_numba",
    "_sanitize_array_numba_parallel",
    "_sma_loop",
    "_sma_loop_parallel",
    "_ema_loop",
    "_ema_loop_alpha",
    "_kalman_loop",
    "_vwap_daily_loop",
    "_rng_filter_loop",
    "_smooth_range",
    "_calc_mmh_worm_loop",
    "_calc_mmh_value_loop",
    "_calc_mmh_momentum_loop",
    "_rolling_std_welford",
    "_rolling_std_welford_parallel",
    "_rolling_mean_numba",
    "_rolling_mean_numba_parallel",
    "_rolling_min_max_numba",
    "_rolling_min_max_numba_parallel",
    "_calculate_ppo_core",
    "_calculate_rsi_core",
    "_vectorized_wick_check_buy",
    "_vectorized_wick_check_sell",
]

class _LazyModule:
    """Load AOT or JIT implementations on first attribute access, with diagnostics."""
    def __init__(self):
        self._loaded = False
        self._impl = {}
        self._aot_funcs: list[str] = []
        self._jit_funcs: list[str] = []

    def __getattr__(self, name: str):
        if not self._loaded:
            self._load()
        if name in self._impl:
            return self._impl[name]
        raise AttributeError(name)

    def _load(self) -> None:
        # 1. try AOT
        if _SO_PATH.exists():
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location("_macd_aot", _SO_PATH)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                for name in __all__:
                    if hasattr(mod, name):
                        self._impl[name] = getattr(mod, name)
                        self._aot_funcs.append(name)
                if self._impl:
                    warnings.warn("✅ Using AOT compiled _macd_aot.so", stacklevel=2)
                    self._loaded = True
                    return
            except Exception as exc:
                warnings.warn(f"⚠️  AOT .so load failed ({exc}) – falling back to JIT", stacklevel=2)

        # 2. fallback to JIT
        try:
            from numba_helpers import (  # noqa
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
            local_map = {
                "_sanitize_array_numba": _sanitize_array_numba,
                "_sanitize_array_numba_parallel": _sanitize_array_numba_parallel,
                "_sma_loop": _sma_loop,
                "_sma_loop_parallel": _sma_loop_parallel,
                "_ema_loop": _ema_loop,
                "_ema_loop_alpha": _ema_loop_alpha,
                "_kalman_loop": _kalman_loop,
                "_vwap_daily_loop": _vwap_daily_loop,
                "_rng_filter_loop": _rng_filter_loop,
                "_smooth_range": _smooth_range,
                "_calc_mmh_worm_loop": _calc_mmh_worm_loop,
                "_calc_mmh_value_loop": _calc_mmh_value_loop,
                "_calc_mmh_momentum_loop": _calc_mmh_momentum_loop,
                "_rolling_std_welford": _rolling_std_welford,
                "_rolling_std_welford_parallel": _rolling_std_welford_parallel,
                "_rolling_mean_numba": _rolling_mean_numba,
                "_rolling_mean_numba_parallel": _rolling_mean_numba_parallel,
                "_rolling_min_max_numba": _rolling_min_max_numba,
                "_rolling_min_max_numba_parallel": _rolling_min_max_numba_parallel,
                "_calculate_ppo_core": _calculate_ppo_core,
                "_calculate_rsi_core": _calculate_rsi_core,
                "_vectorized_wick_check_buy": _vectorized_wick_check_buy,
                "_vectorized_wick_check_sell": _vectorized_wick_check_sell,
            }
            for name in __all__:
                if name in local_map:
                    self._impl[name] = local_map[name]
                    self._jit_funcs.append(name)
            warnings.warn("✅ Using JIT fallbacks from numba_helpers", stacklevel=2)
        except Exception as exc:
            raise RuntimeError("Unable to load either AOT or JIT implementations") from exc
        self._loaded = True

    def summary(self) -> None:
        print("📊 AOT vs JIT summary")
        print(f"AOT functions: {self._aot_funcs}")
        print(f"JIT fallbacks: {self._jit_funcs}")

# install the lazy loader into sys.modules so `from aot_bridge import X` works
sys.modules[__name__] = _LazyModule()  # type: ignore
