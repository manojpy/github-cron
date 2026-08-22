"""
Canonical list of AOT/JIT-dispatched function names.

Both numba_functions_shared.py (defines + compiles the functions) and
aot_bridge.py (dispatches to either the AOT .so or the JIT fallback) import
this list. This module has zero dependencies -- no numba, no numpy -- on
purpose:

  * aot_bridge.py imports it at module level, before it knows whether an
    AOT .so is even available. If this list lived inside
    numba_functions_shared.py instead, importing it here would trigger
    numba_functions_shared's `from numba import njit` and eager-compile
    every JIT fallback function even when AOT succeeds -- defeating the
    whole point of AOT (fast startup, no numba import needed).
  * numba_functions_shared.py also imports it, purely to assert at import
    time that EXPORT_CONFIG hasn't drifted from this list.

To add a new AOT-dispatched function: add its name here once, add its
signature + @njit function + EXPORT_CONFIG entry in
numba_functions_shared.py, and add its thin wrapper function in
aot_bridge.py. Nothing else needs to change -- REQUIRED_AOT_FUNCTIONS,
the JIT-fallback dict, and the AOT dispatch dict are all derived from
this list automatically.
"""

AOT_FUNCTION_NAMES = [
    'sanitize_array_numba',
    'ema_loop',
    'ema_loop_alpha',
    'ema_loop_pine',
    'kalman_loop',
    'vwap_daily_loop_safe',
    'rolling_mean_numba',
    'rolling_min_max_numba',
    'calculate_ppo_core',
    'calculate_rsi_core',
    'true_range_numba',
    'calculate_atr_rma',
    'calculate_adx_core',
    'percentile_rank_numba',
]
