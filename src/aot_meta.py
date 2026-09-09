"""
Merged: single source of truth for the AOT source version stamp AND the
canonical list of AOT/JIT-dispatched function names.

(Formerly two separate files — aot_version.py and aot_function_registry.py —
merged since they were always imported together by aot_bridge.py,
numba_functions_shared.py, and aot_build.py, and are already COPY'd into the
same Docker layer.)

DELIBERATELY has zero imports beyond the standard library (in fact, zero
imports at all). Bump SOURCE_VERSION here any time you change the LOGIC of a
function in numba_functions_shared.py's EXPORT_CONFIG (not needed for
signature-only or comment changes).

Why SOURCE_VERSION lives here instead of directly in numba_functions_shared.py:
aot_bridge.py needs to read this value on every single run -- including the
normal, successful AOT path -- to detect a stale compiled .so. If this constant
lived in numba_functions_shared.py, reading it would require importing that
whole module, which imports `numba` itself. Importing numba costs real,
measurable time (multiple seconds depending on environment/cache state) and was
previously *never paid* on the AOT-success path by design (see the
"Skipping JIT warmup (AOT active)" log line in macd_unified.py). Keeping the
version stamp in this tiny standalone file lets aot_bridge.py verify freshness
without reintroducing that cost on every run.

numba_functions_shared.py and aot_build.py both import SOURCE_VERSION from
here too, so there is exactly one place to update it.
"""

SOURCE_VERSION = "2026-09-01.1"  # dynamic_flow_direction_loop now also returns the plotted midline (line_) for price-cross detection

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
    'dynamic_flow_direction_loop',
]
