"""
numeric_selftest.py
====================
Independent numerical verification of the AOT/JIT-dispatched indicator
functions, run once at startup before any pair is evaluated.

"""
from typing import List, Optional, Tuple
import math
import os
import numpy as np
import aot_bridge

def _ref_ema_alpha(data: List[float], alpha: float) -> List[float]:
    n = len(data)
    period = int(1.0 / alpha + 0.5)
    out: List[float] = [0.0] * n
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
    """Pine ta.lowest/ta.highest semantics: NaN until a full window exists."""
    mins, maxs = [], []
    for i in range(len(arr)):
        if i < period - 1:
            mins.append(float("nan"))
            maxs.append(float("nan"))
            continue
        window = arr[i - period + 1: i + 1]
        mins.append(min(window))
        maxs.append(max(window))
    return mins, maxs

def _ref_dynamic_flow(src: List[float], basis: List[float], dist: List[float], factor: float) -> Tuple[List[float], List[float]]:
    """Independent pure-Python re-derivation of the Dynamic Flow Ribbon
    (direction + plotted midline), mirroring the Pine algorithm bar-by-bar."""
    n = len(src)
    direction_out = [float("nan")] * n
    line_out = [float("nan")] * n

    lower_band_prev = 0.0
    upper_band_prev = 0.0
    trend_is_upper_prev = True

    def isnan(x):
        return x != x

    for i in range(n):
        d, b = dist[i], basis[i]
        if isnan(d) or isnan(b):
            lower_band, upper_band = lower_band_prev, upper_band_prev
        else:
            raw_upper = b + factor * d
            raw_lower = b - factor * d
            src_prev = src[i - 1] if i >= 1 else float("nan")
            lower_band = raw_lower if (raw_lower > lower_band_prev or src_prev < lower_band_prev) else lower_band_prev
            upper_band = raw_upper if (raw_upper < upper_band_prev or src_prev > upper_band_prev) else upper_band_prev

        dist_prev = dist[i - 1] if i >= 1 else float("nan")
        if isnan(dist_prev):
            direction = 1.0
        elif trend_is_upper_prev:
            direction = -1.0 if src[i] > upper_band else 1.0
        else:
            direction = 1.0 if src[i] < lower_band else -1.0
        trend_is_upper_prev = (direction == 1.0)

        if not isnan(d):
            direction_out[i] = direction
            line_out[i] = (lower_band + upper_band) / 2.0

        lower_band_prev, upper_band_prev = lower_band, upper_band

    return direction_out, line_out

def _ref_pine_ema(data: List[float], length: int) -> List[float]:
    """Pine ta.ema: seeded with the first finite value (not an SMA)."""
    alpha = 2.0 / (length + 1)
    out = [float("nan")] * len(data)
    prev: Optional[float] = None
    for i, v in enumerate(data):
        if v != v:                      # NaN keeps the last value, like ema_loop_pine
            if prev is not None:
                out[i] = prev
            continue
        prev = v if prev is None else alpha * v + (1.0 - alpha) * prev
        out[i] = prev
    return out

def _ref_cirrus(close: List[float], t: int, m: float) -> Tuple[List[float], List[float]]:
    """Independent Pine reference for Cirrus Cloud: smoothrngX1 then rngfiltx1x1.
    Returns (smooth_range, range_filter)."""
    diff = [float("nan")] + [abs(close[i] - close[i - 1]) for i in range(1, len(close))]
    sm = _ref_pine_ema(_ref_pine_ema(diff, t), 2 * t - 1)
    rng = [v * m if v == v else float("nan") for v in sm]
    prev = 0.0
    filt: List[float] = []
    for x, r in zip(close, rng):
        if x != x or r != r:
            filt.append(float("nan"))
            prev = 0.0
            continue
        if x > prev:
            prev = prev if (x - r) < prev else (x - r)
        else:
            prev = prev if (x + r) > prev else (x + r)
        filt.append(prev)
    return rng, filt

def _ref_gaussian_filter(src: List[float], length: int, poles: int) -> List[float]:
    """Independent Pine reference for gaussianFilter (nz(stageN, src) seeding)."""
    beta = (1.0 - math.cos(2.0 * math.pi / length)) / (math.pow(1.414, 2.0 / poles) - 1.0)
    alpha = -beta + math.sqrt(beta * beta + 2.0 * beta)
    st: List[Optional[float]] = [None, None, None, None]
    out: List[float] = []
    for x in src:
        s1 = alpha * x + (1.0 - alpha) * (x if st[0] is None else st[0])
        s2 = alpha * s1 + (1.0 - alpha) * (x if st[1] is None else st[1])
        s3 = alpha * s2 + (1.0 - alpha) * (x if st[2] is None else st[2])
        s4 = alpha * s3 + (1.0 - alpha) * (x if st[3] is None else st[3])
        st = [s1, s2, s3, s4]
        out.append(s1 if poles <= 1 else s2 if poles == 2 else s3 if poles == 3 else s4)
    return out

def _ref_efficiency_ratio(src: List[float], length: int) -> List[float]:
    """Pine: abs(src - src[len]) / math.sum(abs(ta.change(src)), len); 0 when path == 0."""
    out: List[float] = []
    for i in range(len(src)):
        if i < length:
            out.append(float("nan"))
            continue
        path = sum(abs(src[j] - src[j - 1]) for j in range(i - length + 1, i + 1))
        out.append(0.0 if path == 0.0 else abs(src[i] - src[i - length]) / path)
    return out

def _ref_gaussian_trend_line(filt: List[float], up: List[float], lo: List[float]) -> Tuple[List[float], List[int]]:
    """Pine stepped dead-band line + sticky direction (na comparisons are false)."""
    tl: List[float] = []
    dr: List[int] = []
    for i in range(len(filt)):
        cur = filt[i] if i == 0 or tl[i - 1] != tl[i - 1] else tl[i - 1]
        if up[i] == up[i] and cur == cur and up[i] < cur:
            cur = up[i]
        if lo[i] == lo[i] and cur == cur and lo[i] > cur:
            cur = lo[i]
        d = 0 if i == 0 else dr[i - 1]
        if i > 0 and cur == cur and tl[i - 1] == tl[i - 1]:
            d = 1 if cur > tl[i - 1] else -1 if cur < tl[i - 1] else d
        tl.append(cur)
        dr.append(d)
    return tl, dr

def _ref_momentum_oscillator(high: List[float], low: List[float], close: List[float], length: int) -> List[float]:
    """Independent bar-by-bar Pine reference for the AlgoAlpha Momentum Oscillator
    (ta.lowest/highest -> ta.ema -> crossover state -> ta.hma -> ta.ema(high-low, 2*len))."""
    n = len(close)
    nan = float("nan")

    def _ema(src: List[float], ln: int) -> List[float]:
        a = 2.0 / (ln + 1)
        out = [nan] * len(src)
        for i, x in enumerate(src):
            prev = out[i - 1] if i > 0 else nan
            out[i] = x if prev != prev else a * x + (1.0 - a) * prev
        return out

    def _wma(src: List[float], ln: int) -> List[float]:
        out = [nan] * len(src)
        for i in range(ln - 1, len(src)):
            w = [src[i - j] for j in range(ln)]
            if any(v != v for v in w):
                continue
            out[i] = sum(v * (ln - j) for j, v in enumerate(w)) / (ln * (ln + 1) / 2.0)
        return out

    lo_min, _ = _ref_rolling_min_max(low, length)
    _, hi_max = _ref_rolling_min_max(high, length)
    l_ = _ema(lo_min, length)
    h_ = _ema(hi_max, length)

    d = 0
    val1 = [nan] * n
    for i in range(n):
        if i > 0 and close[i] > h_[i] and close[i - 1] <= h_[i - 1]:
            d = 1
        if i > 0 and close[i] < l_[i] and close[i - 1] >= l_[i - 1]:
            d = -1
        val1[i] = close[i] - (l_[i] if d == 1 else h_[i])

    half = int(length / 2.0 + 0.5)
    sq = int(math.sqrt(length) + 0.5)
    w1, w2 = _wma(val1, half), _wma(val1, length)
    diff = [nan if (a != a or b != b) else 2.0 * a - b for a, b in zip(w1, w2)]
    val2 = _wma(diff, sq)

    e_hl = _ema([high[i] - low[i] for i in range(n)], length * 2)
    return [nan if (v != v or e != e or e == 0.0) else (v / e * 100.0) / 8.0 for v, e in zip(val2, e_hl)]

# ---------------------------------------------------------------------------
# Deterministic synthetic fixtures -- no RNG, must be byte-identical every run
# ---------------------------------------------------------------------------

def _fixtures():
    n = 60
    idx = np.arange(n, dtype=np.float64)
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

def _check(name, actual, expected, failures, tol=TOLERANCE):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)

    if actual.shape != expected.shape:
        failures.append(f"{name}: shape mismatch {actual.shape} vs {expected.shape}")
        return

    # NaN-pattern mismatch: one side NaN, the other not.
    nan_mismatch = np.isnan(actual) != np.isnan(expected)
    if np.any(nan_mismatch):
        bad = np.where(nan_mismatch)[0][:5].tolist()
        failures.append(f"{name}: NaN pattern mismatch at indices {bad}")
        return

    mask = ~np.isnan(actual)
    if not np.any(mask):
        return
    diff = np.abs(actual[mask] - expected[mask])
    max_diff = np.max(diff)
    if max_diff > tol:
        bad = np.argmax(diff)
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

    hlc3 = (high + low + close) / 3.0
    dfr_basis = aot_bridge.ema_loop_pine(hlc3, 5.0)
    dfr_dist = aot_bridge.rolling_mean_numba(high - low, 10)
    expected_dir, expected_line = _ref_dynamic_flow(
        hlc3.tolist(), dfr_basis.tolist(), dfr_dist.tolist(), 3.0
    )
    actual_dir, actual_line = aot_bridge.dynamic_flow_direction_loop(hlc3, dfr_basis, dfr_dist, 3.0)
    _check("dynamic_flow_direction_loop.direction (trend)", actual_dir, expected_dir, failures)
    _check("dynamic_flow_direction_loop.line (trend)", actual_line, expected_line, failures)

    # Cirrus Cloud kernels (smoothrng_loop / range_filter_loop) vs the Pine reference.
    for _t, _m in ((22, 9.0), (15, 5.0)):
        _exp_rng, _exp_filt = _ref_cirrus(close.tolist(), _t, _m)
        _act_rng = aot_bridge.smoothrng_loop(close, _t, _m)
        _check(f"smoothrng_loop (trend, t={_t})", _act_rng, _exp_rng, failures)
        _act_filt = aot_bridge.range_filter_loop(close, _act_rng)
        _check(f"range_filter_loop (trend, t={_t})", _act_filt, _exp_filt, failures)

    # Gaussian Filter Trend kernels vs the Pine reference.
    for _len, _poles in ((14, 4), (8, 2), (21, 3), (30, 1)):
        _exp_g = _ref_gaussian_filter(close.tolist(), _len, _poles)
        _check(f"gaussian_filter_loop (trend, len={_len}, poles={_poles})",
               aot_bridge.gaussian_filter_loop(close, _len, _poles), _exp_g, failures)
    for _len in (7, 10, 18):
        _check(f"efficiency_ratio_loop (trend, len={_len})",
               aot_bridge.efficiency_ratio_loop(close, _len), _ref_efficiency_ratio(close.tolist(), _len), failures)
    _g_filt = aot_bridge.gaussian_filter_loop(close, 14, 4)
    _g_band = np.full(len(close), 0.9, dtype=np.float64)
    _g_band[:12] = np.nan
    _exp_tl, _exp_dir = _ref_gaussian_trend_line(_g_filt.tolist(), (_g_filt + _g_band).tolist(), (_g_filt - _g_band).tolist())
    _act_tl, _act_dir = aot_bridge.gaussian_trend_line_loop(_g_filt, _g_filt + _g_band, _g_filt - _g_band)
    _check("gaussian_trend_line_loop.line (trend)", _act_tl, _exp_tl, failures)
    _check("gaussian_trend_line_loop.direction (trend)", _act_dir.astype(np.float64),
           [float(v) for v in _exp_dir], failures)

    # Momentum Oscillator (AlgoAlpha) vs the independent Pine reference.
    for _len in (10, 14):
        _check(f"momentum_oscillator_core (trend, len={_len})",
               aot_bridge.momentum_oscillator_core(high, low, close, _len),
               _ref_momentum_oscillator(high.tolist(), low.tolist(), close.tolist(), _len), failures)

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

        for _t, _m in ((22, 9.0), (15, 5.0)):
            _aot_rng = aot_bridge.smoothrng_loop(close, _t, _m)
            _jit_rng = jit_ref.smoothrng_loop(close, _t, _m)
            _check(f"AOT vs JIT: smoothrng_loop (t={_t})", _aot_rng, _jit_rng, failures)
            _check(f"AOT vs JIT: range_filter_loop (t={_t})",
                   aot_bridge.range_filter_loop(close, _aot_rng),
                   jit_ref.range_filter_loop(close, _jit_rng), failures)

        for _len, _poles in ((14, 4), (8, 2)):
            _check(f"AOT vs JIT: gaussian_filter_loop (len={_len}, poles={_poles})",
                   aot_bridge.gaussian_filter_loop(close, _len, _poles),
                   jit_ref.gaussian_filter_loop(close, _len, _poles), failures)
        _check("AOT vs JIT: efficiency_ratio_loop",
               aot_bridge.efficiency_ratio_loop(close, 10), jit_ref.efficiency_ratio_loop(close, 10), failures)
        _jit_tl, _jit_dir = jit_ref.gaussian_trend_line_loop(_g_filt, _g_filt + _g_band, _g_filt - _g_band)
        _check("AOT vs JIT: gaussian_trend_line_loop.line", _act_tl, _jit_tl, failures)
        _check("AOT vs JIT: gaussian_trend_line_loop.direction",
               _act_dir.astype(np.float64), _jit_dir.astype(np.float64), failures)

        for _len in (10, 14):
            _check(f"AOT vs JIT: momentum_oscillator_core (len={_len})",
                   aot_bridge.momentum_oscillator_core(high, low, close, _len),
                   jit_ref.momentum_oscillator_core(high, low, close, _len), failures)

        jit_dfr_dir, jit_dfr_line = jit_ref.dynamic_flow_direction_loop(hlc3, dfr_basis, dfr_dist, 3.0)
        _check("AOT vs JIT: dynamic_flow_direction_loop.direction", actual_dir, jit_dfr_dir, failures)
        _check("AOT vs JIT: dynamic_flow_direction_loop.line", actual_line, jit_dfr_line, failures)

    return (len(failures) == 0, failures)