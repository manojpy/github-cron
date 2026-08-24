"""
numeric_selftest.py
====================
Independent numerical verification of the AOT/JIT-dispatched indicator
functions, run once at startup before any pair is evaluated.

"""

from typing import List, Tuple
import os
import numpy as np
import aot_bridge


def _ref_ema_alpha(data: List[float], alpha: float) -> List[float]:
    """Mirrors ema_loop_alpha's SMA-seeded Wilder/RMA convention: the first
    `period = round(1/alpha)` bars are seeded with a plain average, then the
    recursion `alpha*curr + (1-alpha)*prev` takes over. A naive first-value
    seed (plain EMA) does NOT match this function -- confirmed against
    numba_functions_shared.calculate_atr_rma's actual seeding logic, not
    assumed."""
    n = len(data)
    period = int(round(1.0 / alpha))
    out = [None] * n
    if n < period:
        period = n
    sma_seed = sum(data[:period]) / period
    for i in range(period):
        out[i] = sma_seed
    prev = sma_seed
    for i in range(period, n):
        prev = alpha * data[i] + (1.0 - alpha) * prev
        out[i] = prev
    return out


def _ref_true_range(high: List[float], low: List[float], close: List[float]) -> List[float]:
    tr = [high[0] - low[0]]
    for i in range(1, len(close)):
        tr.append(max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        ))
    return tr


def _ref_atr_rma(high: List[float], low: List[float], close: List[float], period: int) -> List[float]:
    tr = _ref_true_range(high, low, close)
    return _ref_ema_alpha(tr, 1.0 / period)


def _ref_rolling_min_max(arr: List[float], period: int) -> Tuple[List[float], List[float]]:
    mins, maxs = [], []
    for i in range(len(arr)):
        window = arr[max(0, i - period + 1): i + 1]
        mins.append(min(window))
        maxs.append(max(window))
    return mins, maxs


# ---------------------------------------------------------------------------
# Deterministic synthetic fixtures -- no RNG, must be byte-identical every run
# ---------------------------------------------------------------------------

def _fixtures():
    n = 60
    idx = np.arange(n, dtype=np.float64)
    # Gentle uptrend with a sinusoidal wobble: deterministic, non-degenerate,
    # exercises both rising and falling legs so RSI/ADX see real movement.
    close = 100.0 + 0.15 * idx + 2.0 * np.sin(idx / 3.0)
    high = close + 0.8
    low = close - 0.8
    open_ = close - 0.2
    volume = np.full(n, 1000.0, dtype=np.float64)

    flat = np.full(n, 50.0, dtype=np.float64)  # degenerate: zero volatility

    return {
        "trend": (open_, high, low, close, volume),
        "flat": (flat, flat.copy(), flat.copy(), flat, volume),
    }


TOLERANCE = 1e-6


def _check(name: str, actual, expected, failures: List[str], tol: float = TOLERANCE) -> None:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    both_nan = np.isnan(actual) & np.isnan(expected)
    mask = ~both_nan
    if not np.any(mask):
        return
    diff = np.abs(actual[mask] - expected[mask])
    max_diff = np.nanmax(diff)
    if max_diff > tol:
        bad = np.nanargmax(diff)
        failures.append(
            f"{name}: max abs diff {max_diff:.3e} > tol {tol:.1e} "
            f"(worst: actual={actual[mask][bad]:.6f}, expected={expected[mask][bad]:.6f})"
        )


def run_self_test() -> Tuple[bool, List[str]]:
    """Runs golden-value checks plus, when AOT is active, an AOT-vs-JIT
    cross-check. Returns (all_passed, list_of_failure_messages)."""
    failures: List[str] = []
    fixtures = _fixtures()
    open_, high, low, close, volume = fixtures["trend"]
    flat_o, flat_h, flat_l, flat_c, flat_v = fixtures["flat"]
    period = 14

    # --- Golden-value checks against the independent reference ---
    expected_atr = _ref_atr_rma(high.tolist(), low.tolist(), close.tolist(), period)
    actual_atr = aot_bridge.calculate_atr_rma(high, low, close, period)
    _check("calculate_atr_rma (trend)", actual_atr, expected_atr, failures)

    expected_min, expected_max = _ref_rolling_min_max(close.tolist(), 10)
    actual_min, actual_max = aot_bridge.rolling_min_max_numba(close, 10)
    _check("rolling_min_max_numba.min (trend)", actual_min, expected_min, failures)
    _check("rolling_min_max_numba.max (trend)", actual_max, expected_max, failures)

    expected_ema = _ref_ema_alpha(close.tolist(), 0.2)
    actual_ema = aot_bridge.ema_loop_alpha(close, 0.2)
    _check("ema_loop_alpha (trend)", actual_ema, expected_ema, failures)

    # --- Analytic invariants on the degenerate flat fixture ---
    flat_atr = aot_bridge.calculate_atr_rma(flat_h, flat_l, flat_c, period)
    flat_atr_tail = flat_atr[period:]
    if flat_atr_tail.size and np.nanmax(np.abs(flat_atr_tail)) > TOLERANCE:
        failures.append(
            f"calculate_atr_rma (flat): expected ~0 on zero-volatility series, "
            f"got max {np.nanmax(np.abs(flat_atr_tail)):.3e}"
        )

    flat_rsi = aot_bridge.calculate_rsi_core(flat_c, period)
    flat_rsi_tail = flat_rsi[period:]
    if flat_rsi_tail.size and np.nanmax(np.abs(flat_rsi_tail - 50.0)) > TOLERANCE:
        failures.append(
            f"calculate_rsi_core (flat): expected 50.0 on zero-movement series, "
            f"got max deviation {np.nanmax(np.abs(flat_rsi_tail - 50.0)):.3e}"
        )

    if aot_bridge.is_using_aot() and os.getenv("NUMERIC_SELFTEST_JIT_CROSSCHECK", "false").lower() == "true":
        import numba_functions_shared as jit_ref  # lazy: see module docstring

        jit_atr = jit_ref.calculate_atr_rma(high, low, close, period)
        _check("AOT vs JIT: calculate_atr_rma", actual_atr, jit_atr, failures)

        aot_rsi = aot_bridge.calculate_rsi_core(close, period)
        jit_rsi = jit_ref.calculate_rsi_core(close, period)
        _check("AOT vs JIT: calculate_rsi_core", aot_rsi, jit_rsi, failures)

        aot_adx = aot_bridge.calculate_adx_core(high, low, close, 14, 14)
        jit_adx = jit_ref.calculate_adx_core(high, low, close, 14, 14)
        _check("AOT vs JIT: calculate_adx_core", aot_adx, jit_adx, failures)

        aot_ppo, aot_ppo_sig = aot_bridge.calculate_ppo_core(close, 12, 26, 9)
        jit_ppo, jit_ppo_sig = jit_ref.calculate_ppo_core(close, 12, 26, 9)
        _check("AOT vs JIT: calculate_ppo_core.ppo", aot_ppo, jit_ppo, failures)
        _check("AOT vs JIT: calculate_ppo_core.signal", aot_ppo_sig, jit_ppo_sig, failures)

    return (len(failures) == 0, failures)
