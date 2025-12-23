"""
AOT-aware entry point.

1.  Re-export everything from macd_unified  (so all classes / constants exist)
2.  Overlay the *hot* Numba helpers with their AOT versions when possible.
"""
from __future__ import annotations

# ------------------------------------------------------------------
# 1.  Bring the whole original module into our namespace
# ------------------------------------------------------------------
from .macd_unified import *  # noqa: F403,F401  (keep lint happy)

# ------------------------------------------------------------------
# 2.  Replace selected helpers with fastest implementation available
# ------------------------------------------------------------------
from .aot_bridge import get_impl

# list every helper we want to accelerate
_AOT_FUNCTIONS = (
    "_calculate_ppo_core",
    "_calculate_rsi_core",
    "_vwap_daily_loop",
    "_sanitize_array_numba",
    "_sanitize_array_numba_parallel",
    "_sma_loop",
    "_sma_loop_parallel",
    "_ema_loop",
    "_ema_loop_alpha",
    "_kalman_loop",
    "_rolling_std_welford",
    "_rolling_std_welford_parallel",
    "_rolling_mean_numba",
    "_rolling_mean_numba_parallel",
    "_rolling_min_max_numba",
    "_rolling_min_max_numba_parallel",
    "_calc_mmh_worm_loop",
    "_calc_mmh_value_loop",
    "_calc_mmh_momentum_loop",
    "_rng_filter_loop",
    "_smooth_range",
    "_vectorized_wick_check_buy",
    "_vectorized_wick_check_sell",
)


for _fname in __AOT_FUNCTIONS:
    globals()[_fname] = get_impl(_fname)  # overlay AOT || JIT

# ------------------------------------------------------------------
# 3.  (Optional) advertise that we are AOT-ready
# ------------------------------------------------------------------
try:
    from .aot_bridge import _AOT_AVAILABLE  # type: ignore[attr-defined]
    if _AOT_AVAILABLE:  # type: ignore[name-defined]
        import logging
        logging.getLogger("wrapper").info("AOT compiled helpers loaded")
except Exception:  # noqa: BLE001
    pass
