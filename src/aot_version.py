"""
Single source of truth for the AOT source version stamp.

DELIBERATELY has zero imports beyond the standard library. Bump SOURCE_VERSION
here any time you change the LOGIC of a function in numba_functions_shared.py's
EXPORT_CONFIG (not needed for signature-only or comment changes).

Why this lives in its own file instead of directly in numba_functions_shared.py:
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
SOURCE_VERSION = "2026-08-30.1"  # add dynamic_flow_direction_loop (Dynamic Flow Ribbon)