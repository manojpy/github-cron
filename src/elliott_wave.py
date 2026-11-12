"""elliott_wave — pure, deterministic Elliott Wave candidate engine.

No Telegram, Redis, config or bot imports: numpy arrays in, plain dataclasses
out. Everything the bot needs from it goes through `analyze()`.

What it finds (one candidate per family and direction, never a combinatorial
search):

    wave3  Wave 1 and Wave 2 are complete and price has just broken the last
           internal high of the Wave 2 decline  -> Wave 3 is starting.
    wave5  Waves 1-4 are complete and price has just closed beyond the Wave 3
           extreme                                -> Wave 5 is starting.
    abc    A zigzag correction (A, B, C) of the prior trend is complete and
           price has just broken the last internal high of the C decline.

SELL is the exact mirror of BUY: prices are multiplied by -1 and the BUY logic
is reused unchanged, so the two sides cannot drift apart.

No repainting
    * `analyze(..., end=i)` reads ONLY candles 0..i. Candle i is the last
      CLOSED candle; the caller must never pass a forming candle.
    * A pivot with `length` candles on each side is usable only once its
      right-hand candles have closed (`confirm_idx = idx + length <= end`).
    * Re-running `analyze` with more candles appended and the same `end`
      returns the identical result (tested), so a stored alert never changes.

Mandatory rules reject a count outright, whatever its score:
    1. Wave 2 may not retrace beyond the start of Wave 1.
    2. Wave 3 may not be the shortest of Waves 1, 3 and 5 (checked on a
       complete impulse). Nothing else is imposed on its length: a Wave 3 only
       marginally longer than Wave 1 or 5 is valid and is judged by the score.
    3. Wave 4 may not overlap Wave 1 price territory (standard impulse).
The 0-100 score only ranks VALID counts; it never rescues an invalid one.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

__all__ = [
    "ElliottParams", "Pivot", "WaveSetup", "Analysis",
    "ComponentScore", "ENGINE_VERSION", "FEATURE_VERSION", "WaveEvent", "evaluate_progress",
    "family_rank", "replay_events",
    "analyze", "confirmed_pivots", "normalize_pivots", "htf_bias",
    "check_impulse_rules", "check_abc_rules",
    "FAMILY_PRIORITY", "SCORE_WEIGHTS",
]

BUY = "buy"
SELL = "sell"

# Part of every event id: a change to the counting rules must never reuse old ids.
ENGINE_VERSION = "ew1"

# Version of the Elliott feature set written into an outcome row's `context`
# (stamped as `feature_version`). Deliberately a separate namespace from the legacy
# feature store ("fv1"): the legacy version is untouched, so no legacy dataset or
# cache is invalidated, and the two feature sets can never be mixed by accident.
# Bump when the meaning or set of `elliott_*` fields changes.
FEATURE_VERSION = "ewfv1"

# Tie-break when several families qualify on one pair/candle: Wave 3 first.
FAMILY_PRIORITY: Dict[str, int] = {"wave3": 0, "wave5": 1, "abc": 2}

# Documented starting weights (research parameters, not optimised values).
SCORE_WEIGHTS: Dict[str, float] = {
    "structure": 25.0, "fibonacci": 20.0, "market_structure": 20.0,
    "momentum": 15.0, "volume": 10.0, "htf": 10.0,
}


@dataclass(frozen=True)
class ElliottParams:
    pivot_len: int = 3                  # candles each side of a swing pivot
    internal_pivot_len: int = 2         # finer pivots used for the internal trigger level
    lookback: int = 240                 # structural lookback (candles)
    warmup: int = 60                    # extra candles so indicators are warm
    min_bars: int = 80                  # clean candles required, else no signal
    min_swing_atr: float = 1.0          # minimum swing between pivots, in ATR
    atr_len: int = 14
    interval_sec: int = 900
    min_score: float = 80.0             # >= : eligible for delivery
    watch_score: float = 70.0           # >= : WATCH (recorded, not delivered)
    max_entry_distance_atr: float = 1.5  # anti-chase: max distance beyond the trigger level
    max_age_bars: int = 16              # wave3 / abc: max candles since the last pivot was confirmed
    max_age_bars_wave5: int = 32
    min_rr: float = 1.5                 # minimum reward/risk to the first target
    stop_buffer_atr: float = 0.25
    enable_wave3: bool = True
    enable_wave5: bool = True
    enable_abc: bool = True
    enable_exhaustion: bool = True      # Wave 5 exhaustion EVENTS (never a trade on their own)
    primary_family: str = "wave3"       # wins ties when several families qualify on one pair/candle
    htf_min_hours: int = 24
    htf_ema_len: int = 21


@dataclass(frozen=True)
class Pivot:
    idx: int            # candle index of the extreme
    price: float
    kind: int           # +1 swing high, -1 swing low
    confirm_idx: int    # first candle at which the pivot was knowable


@dataclass(frozen=True)
class ComponentScore:
    """One scoring pillar. `available=False` means the input was missing and
    the pillar scored 0 (missing data never earns points)."""
    name: str
    score: float
    max_score: float
    reason: str
    values: Dict[str, Any]
    available: bool = True


@dataclass
class WaveSetup:
    family: str                         # "wave3" | "wave5" | "abc"
    direction: str                      # "buy" | "sell"
    status: str                         # "eligible" | "watch" | "rejected"
    reason: str                         # why rejected ("" otherwise)
    score: float
    components: Dict[str, float]
    labels: Dict[str, Tuple[int, int, float]]   # name -> (absolute idx, ts, price)
    event_id: str                       # unique decision id (includes the signal candle)
    structure_id: str                   # identity of the wave structure (dedup key; excludes the signal candle)
    signal_idx: int
    signal_ts: int
    entry_price: float
    trigger_level: float
    entry_distance_atr: float
    invalidation: float
    stop: float
    targets: Tuple[float, ...]
    rr: float
    atr: float
    age_bars: int
    htf_bias: Optional[int]             # +1 aligned with trade, -1 against, 0 flat, None unavailable
    alt_counts: List[str] = field(default_factory=list)
    detail: Dict[str, ComponentScore] = field(default_factory=dict)
    features: Dict[str, Any] = field(default_factory=dict)
    pair: str = ""


@dataclass
class WaveEvent:
    """An informational structural event that is NOT a trade setup. Today: the
    completion of a five-wave impulse showing exhaustion ("exhaustion"). It is
    recorded and followed up so its reversal behaviour can be measured, but it
    never produces an alert on its own."""
    kind: str                           # "exhaustion"
    direction: str                      # the direction the event points to (reversal side)
    signs: Dict[str, bool]              # which exhaustion signs are present
    score: int                          # number of signs present
    labels: Dict[str, Tuple[int, int, float]]
    event_id: str
    structure_id: str
    signal_idx: int
    signal_ts: int
    entry_price: float                  # close of the candle that confirmed the Wave 5 pivot
    invalidation: float                 # a close beyond this ends the reversal idea
    targets: Tuple[float, ...]          # 0.382 / 0.618 retracement of Wave 5
    atr: float
    features: Dict[str, Any] = field(default_factory=dict)
    pair: str = ""


@dataclass
class Analysis:
    setups: List[WaveSetup]
    primary: Optional[WaveSetup]
    n_pivots: int
    window_start: int
    window_end: int
    diagnostics: Dict[str, Any]
    events: List[WaveEvent] = field(default_factory=list)


def family_rank(family: str, primary_family: str = "wave3") -> int:
    """Selection order when several families qualify: the configured primary
    family first, then the default order wave3 > wave5 > abc."""
    return 0 if family == primary_family else 1 + FAMILY_PRIORITY.get(family, 9)


# ── indicators (causal: value at i uses candles 0..i only) ───────────────

def _wilder_atr(h: np.ndarray, l: np.ndarray, c: np.ndarray, n: int) -> np.ndarray:
    size = len(c)
    out = np.full(size, np.nan)
    if size < n or n < 1:
        return out
    tr = np.empty(size)
    tr[0] = h[0] - l[0]
    if size > 1:
        pc = c[:-1]
        tr[1:] = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - pc), np.abs(l[1:] - pc)))
    out[n - 1] = float(np.mean(tr[:n]))
    for i in range(n, size):
        out[i] = (out[i - 1] * (n - 1) + tr[i]) / n
    return out


def _ema(x: np.ndarray, n: int) -> np.ndarray:
    out = np.empty(len(x))
    if len(x) == 0:
        return out
    a = 2.0 / (n + 1.0)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1.0 - a) * out[i - 1]
    return out


def _macd_hist(c: np.ndarray, fast: int = 12, slow: int = 26, sig: int = 9) -> np.ndarray:
    line = _ema(c, fast) - _ema(c, slow)
    return line - _ema(line, sig)


def _rsi(c: np.ndarray, n: int = 14) -> np.ndarray:
    size = len(c)
    out = np.full(size, np.nan)
    if size <= n:
        return out
    d = np.diff(c)
    up = np.where(d > 0, d, 0.0)
    dn = np.where(d < 0, -d, 0.0)
    au = float(np.mean(up[:n]))
    ad = float(np.mean(dn[:n]))
    out[n] = 100.0 if ad == 0 else 100.0 - 100.0 / (1.0 + au / ad)
    for i in range(n + 1, size):
        au = (au * (n - 1) + up[i - 1]) / n
        ad = (ad * (n - 1) + dn[i - 1]) / n
        out[i] = 100.0 if ad == 0 else 100.0 - 100.0 / (1.0 + au / ad)
    return out


# ── swing pivots ─────────────────────────────────────────────────────────

def confirmed_pivots(h: np.ndarray, l: np.ndarray, length: int, first: int = 0) -> List[Pivot]:
    """Swing highs/lows with `length` candles on each side, chronological.

    A high needs a strictly lower left side and a not-higher right side (the
    first of two equal highs wins); lows mirror it. A candle that is both a
    high and a low pivot (outside bar) is skipped, and flat stretches never
    produce a pivot. `confirm_idx = idx + length`, so nothing is returned that
    needed a candle after the last one in the arrays."""
    n = len(h)
    w = 2 * length + 1
    if length < 1 or n < w:
        return []
    hw = sliding_window_view(h, w)
    lw = sliding_window_view(l, w)
    hc, lc = hw[:, length], lw[:, length]
    is_hi = (hc > hw[:, :length].max(axis=1)) & (hc >= hw[:, length + 1:].max(axis=1))
    is_lo = (lc < lw[:, :length].min(axis=1)) & (lc <= lw[:, length + 1:].min(axis=1))
    both = is_hi & is_lo
    is_hi &= ~both
    is_lo &= ~both
    out: List[Pivot] = []
    for pos in np.flatnonzero(is_hi | is_lo):
        idx = int(pos) + length
        if idx < first:
            continue
        if is_hi[pos]:
            out.append(Pivot(idx, float(h[idx]), 1, idx + length))
        else:
            out.append(Pivot(idx, float(l[idx]), -1, idx + length))
    return out


def normalize_pivots(pivots: Sequence[Pivot], atr: np.ndarray, min_swing_atr: float) -> List[Pivot]:
    """Alternating high/low list.

    * consecutive same-direction pivots collapse into the more extreme one;
    * a pivot closer than `min_swing_atr` ATR to the previous opposite pivot
      is dropped as noise;
    * a pivot with unknown/zero ATR is dropped (never guess a threshold).
    Original `idx` and `confirm_idx` are preserved."""
    out: List[Pivot] = []
    for p in pivots:
        a = float(atr[p.idx]) if 0 <= p.idx < len(atr) else float("nan")
        if not (math.isfinite(a) and a > 0):
            continue
        if not out:
            out.append(p)
            continue
        last = out[-1]
        if p.kind == last.kind:
            if (p.kind == 1 and p.price > last.price) or (p.kind == -1 and p.price < last.price):
                out[-1] = p
            continue
        if abs(p.price - last.price) < min_swing_atr * a:
            continue
        out.append(p)
    return out


# ── higher timeframe (built from closed 15m candles, no extra fetch) ─────

def htf_bias(ts: np.ndarray, c: np.ndarray, end: int, interval_sec: int = 900,
             min_hours: int = 24, ema_len: int = 21) -> Optional[int]:
    """+1 / -1 / 0 hourly trend from complete, contiguous 1h candles, or None.

    An hour counts only when its four 15m candles are all present, on the
    expected timestamps, and closed (<= end). The newest forming hour is
    ignored. Any missing candle ends the contiguous run, and a run shorter
    than `min_hours` returns None instead of inventing a direction."""
    per = 3600 // interval_sec
    if per < 1 or end < per or end >= len(ts):
        return None
    ts_i = ts[: end + 1].astype(np.int64)
    hours: Dict[int, Dict[int, float]] = {}
    for k in range(max(0, end - 24 * per * 8), end + 1):      # bounded scan
        hour = int(ts_i[k] // 3600)
        slot = int((ts_i[k] % 3600) // interval_sec)
        if (ts_i[k] % interval_sec) != 0:
            return None
        hours.setdefault(hour, {})[slot] = float(c[k])
    complete = sorted(hr for hr, slots in hours.items() if len(slots) == per)
    if not complete:
        return None
    run = [complete[-1]]
    for hr in reversed(complete[:-1]):
        if hr == run[-1] - 1:
            run.append(hr)
        else:
            break
    run.reverse()
    if len(run) < min_hours:
        return None
    closes = np.array([hours[hr][per - 1] for hr in run])     # close of the last 15m slot
    if not np.all(np.isfinite(closes)):
        return None
    e = _ema(closes, ema_len)
    rising = e[-1] > e[-3]
    falling = e[-1] < e[-3]
    if closes[-1] > e[-1] and rising:
        return 1
    if closes[-1] < e[-1] and falling:
        return -1
    return 0


# ── rule checks (public: used by tests and by the evaluators) ────────────

def check_impulse_rules(p: Sequence[float], is_up: bool = True) -> Optional[str]:
    """None when the 3..6 impulse points satisfy the mandatory rules, else the
    name of the first violated rule. `p` = [P0 start, P1 end W1, P2 end W2,
    P3 end W3, P4 end W4, P5 end W5]; for a down-impulse pass is_up=False.
    Wave 3 only has to avoid being the SHORTEST of Waves 1, 3, 5: a Wave 3 that
    is marginally longer than Wave 1 or 5 is valid. Proportions are scored,
    never validated. With Wave 5 still unknown (<6 points) Wave 3 cannot yet
    be judged against it."""
    s = 1.0 if is_up else -1.0
    q = [s * float(x) for x in p]
    n = len(q)
    if n < 3:
        return "too_few_points"
    if not q[1] > q[0]:
        return "wave1_not_impulsive"
    if not q[2] > q[0]:
        return "wave2_beyond_origin"                    # rule 1
    if not q[2] < q[1]:
        return "wave2_not_a_correction"
    if n >= 4 and not q[3] > q[2]:
        return "wave3_not_impulsive"
    if n >= 5:
        if not q[4] > q[1]:
            return "wave4_overlaps_wave1"               # rule 3
        if not q[4] < q[3]:
            return "wave4_not_a_correction"
    if n >= 6:
        w1, w3, w5 = q[1] - q[0], q[3] - q[2], q[5] - q[4]
        if not q[5] > q[3]:
            return "wave5_truncated"
        if w3 < w1 and w3 < w5:
            return "wave3_shortest"                     # rule 2
    return None


def check_abc_rules(z: Optional[float], s_: float, a: float, b: float, c: float,
                    is_up_trend: bool = True) -> Optional[str]:
    """Zigzag correction of a prior trend: S (start of the correction), A, B, C.
    `z` is the low before S (prior trend start); None when unknown.
    is_up_trend=True describes a down-correction (A down, B up, C down)."""
    sg = 1.0 if is_up_trend else -1.0
    S, A, B, C = sg * s_, sg * a, sg * b, sg * c
    if not (S > A):
        return "a_not_a_leg"
    if not (A < B < S):
        return "b_not_a_partial_retrace"                # B must stay below S (no flat/expanded)
    if not C < A:
        return "c_not_beyond_a"                         # C must make a new extreme
    if z is None:
        return "no_prior_trend"
    Z = sg * z
    if not S > Z:
        return "no_prior_leg"
    if not C > Z:
        return "correction_erased_prior_trend"
    return None


# ── scoring helpers ──────────────────────────────────────────────────────

def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _zone_pts(value: float, zones: Sequence[Tuple[float, float]], max_pts: float,
              tol: float, width: float, relative: bool = False) -> float:
    """Best score over Fibonacci zones. Full weight inside `tol` of a zone,
    fading to zero at `width`. Zones carry a quality weight (0..1)."""
    best = 0.0
    for z, wgt in zones:
        d = abs(value - z) / (z if relative and z else 1.0)
        best = max(best, wgt * _clamp(1.0 - (d - tol) / (width - tol)))
    return max_pts * best


def score_momentum(s: int, hist: np.ndarray, rsi: np.ndarray, o: float, h: float,
                   l: float, c: float, end: int) -> ComponentScore:
    """15 points: MACD/PPO histogram 5, RSI 5, candle thrust 5. `s` = +1 buy / -1 sell."""
    hs = s * hist
    if hs[end] > 0 and hs[end] >= hs[end - 1]:
        p_hist = 5.0
    elif hs[end] > 0:
        p_hist = 3.5
    elif hs[end] > hs[end - 1] > hs[end - 2]:
        p_hist = 2.0
    else:
        p_hist = 0.0
    r_now, r_prev = s * (rsi[end] - 50.0), rsi[end] - rsi[end - 3]
    if not math.isfinite(r_now) or not math.isfinite(r_prev):
        p_rsi = 0.0
    elif r_now > 0 and s * r_prev > 0:
        p_rsi = 5.0
    elif r_now > 0:
        p_rsi = 3.5
    elif s * r_prev > 0:
        p_rsi = 2.0
    else:
        p_rsi = 0.0
    rng = h - l
    body = s * (c - o)
    if rng > 0:
        body_ratio = body / rng
        loc = ((c - l) / rng) if s > 0 else ((h - c) / rng)
        if body_ratio >= 0.5 and loc >= 0.7:
            p_thr = 5.0
        elif body_ratio >= 0.35 and loc >= 0.6:
            p_thr = 3.0
        elif body > 0:
            p_thr = 1.0
        else:
            p_thr = 0.0
    else:
        p_thr = 0.0
    return ComponentScore(
        "momentum", p_hist + p_rsi + p_thr, 15.0,
        f"hist {p_hist:g}/5, rsi {p_rsi:g}/5, thrust {p_thr:g}/5",
        {"hist": p_hist, "rsi": p_rsi, "thrust": p_thr})


def score_volume(vol: np.ndarray, end: int, prev_leg: Tuple[int, int],
                 now_leg: Tuple[int, int]) -> ComponentScore:
    """10 points: breakout candle vs its 20-candle average (5) and volume of the
    new leg vs the correction leg (5). Unusable volume scores 0 and is flagged."""
    if end < 21 or not np.all(np.isfinite(vol[end - 20:end + 1])):
        return ComponentScore("volume", 0.0, 10.0, "volume unavailable", {}, False)
    base = float(np.mean(vol[end - 20:end]))
    if base <= 0:
        return ComponentScore("volume", 0.0, 10.0, "volume unavailable", {}, False)
    ratio = float(vol[end]) / base
    pts = 5.0 if ratio >= 1.5 else 3.0 if ratio >= 1.0 else 1.0 if ratio >= 0.8 else 0.0
    a0, a1 = prev_leg
    b0, b1 = now_leg
    prev_mean = float(np.mean(vol[a0:a1 + 1])) if a1 >= a0 else 0.0
    now_mean = float(np.mean(vol[b0:b1 + 1])) if b1 >= b0 else 0.0
    r2 = (now_mean / prev_mean) if prev_mean > 0 else float("nan")
    if math.isfinite(r2):
        pts += 5.0 if r2 >= 1.2 else 3.0 if r2 >= 1.0 else 0.0
    return ComponentScore("volume", pts, 10.0, f"breakout vol x{ratio:.2f}, leg vol x{r2:.2f}",
                          {"breakout_ratio": ratio, "leg_ratio": r2})


def score_htf(bias: Optional[int], s: int) -> ComponentScore:
    """10 points from the completed 1h trend: aligned 10, flat 4, against 0,
    unavailable 0 (flagged)."""
    if bias is None:
        return ComponentScore("htf", 0.0, 10.0, "1h trend unavailable", {"aligned": None}, False)
    aligned = bias * s
    pts = 10.0 if aligned > 0 else 4.0 if aligned == 0 else 0.0
    word = "aligned" if aligned > 0 else "flat" if aligned == 0 else "against"
    return ComponentScore("htf", pts, 10.0, f"1h {word}", {"aligned": aligned})


def score_structure(family: str, m: Dict[str, float]) -> ComponentScore:
    """25 points for the CLARITY of an already-valid count (leg size, time
    proportion, alternation, trigger quality). Validity itself is a hard
    pass/reject in the evaluators and earns nothing here."""
    pts = 0.0
    parts: Dict[str, float] = {}
    if family == "wave3":
        w = m["w1_atr"]
        parts["w1_size"] = 9.0 if w >= 3.0 else 6.0 if w >= 2.0 else 3.0 if w >= 1.5 else 0.0
        parts["proportion"] = 6.0 if m["dur1"] > 0 and 0.3 <= m["dur2"] / m["dur1"] <= 3.0 else 0.0
        parts["pivot_trigger"] = 5.0 if m["real_trigger"] else 0.0
        parts["depth"] = 5.0 if m["retr2"] >= 0.382 else 2.5 if m["retr2"] >= 0.236 else 0.0
    elif family == "wave5":
        e3 = m["ext3"]
        parts["w3_extension"] = 9.0 if e3 >= 1.618 else 6.0 if e3 >= 1.0 else 3.0
        parts["proportion"] = 6.0 if m["dur2"] > 0 and 0.3 <= m["dur4"] / m["dur2"] <= 3.0 else 0.0
        parts["alternation"] = 5.0 if abs(m["retr2"] - m["retr4"]) >= 0.2 else 0.0
        parts["w1_size"] = 5.0 if m["w1_atr"] >= 2.0 else 2.5 if m["w1_atr"] >= 1.5 else 0.0
    else:
        pr = m["prior_ratio"]
        parts["prior_leg"] = 9.0 if pr >= 1.0 else 6.0 if pr >= 0.6 else 3.0
        parts["c_extension"] = 6.0 if 0.618 <= m["ext_c"] <= 2.618 else 0.0
        parts["b_retracement"] = 5.0 if 0.382 <= m["retr_b"] <= 0.886 else 0.0
        parts["a_size"] = 5.0 if m["a_atr"] >= 2.0 else 2.5 if m["a_atr"] >= 1.5 else 0.0
    pts = sum(parts.values())
    return ComponentScore("structure", pts, 25.0, ", ".join(f"{k} {v:g}" for k, v in parts.items()),
                          dict(parts))


def score_fibonacci(family: str, m: Dict[str, float]) -> ComponentScore:
    """20 points: how close the measured ratios sit to Fibonacci zones. A soft
    quality feature, never a rejection."""
    if family == "wave3":
        pts = _zone_pts(m["retr2"], [(0.618, 1.0), (0.5, 0.9), (0.786, 0.85), (0.382, 0.6)],
                        20.0, 0.03, 0.12)
        vals = {"wave2_retracement": m["retr2"]}
    elif family == "wave5":
        pts = (_zone_pts(m["ext3"], [(1.618, 1.0), (2.0, 0.9), (2.618, 0.85)], 10.0, 0.05, 0.25, True)
               + _zone_pts(m["retr4"], [(0.382, 1.0), (0.5, 0.9), (0.236, 0.85)], 10.0, 0.03, 0.12))
        vals = {"wave3_extension": m["ext3"], "wave4_retracement": m["retr4"]}
    else:
        pts = (_zone_pts(m["retr_b"], [(0.618, 1.0), (0.5, 0.95), (0.382, 0.85), (0.786, 0.85)],
                         10.0, 0.03, 0.12)
               + _zone_pts(m["ext_c"], [(1.0, 1.0), (0.618, 0.9), (1.618, 0.85)], 10.0, 0.05, 0.25, True))
        vals = {"b_retracement": m["retr_b"], "c_extension": m["ext_c"]}
    return ComponentScore("fibonacci", pts, 20.0,
                          ", ".join(f"{k} {v:.3f}" for k, v in vals.items()), vals)


def score_bos(family: str, beyond_swing: bool, decisive: bool) -> ComponentScore:
    """20 points for the break of structure. The trigger itself (close beyond
    the internal high / Wave 3 extreme) is required to get here and earns the
    base; breaking the larger swing earns the rest."""
    if family == "wave5":
        pts = 14.0 + (6.0 if decisive else 2.0)
        why = "close beyond Wave 3 high" + (", decisive" if decisive else ", marginal")
    else:
        pts = 12.0 + (8.0 if beyond_swing else 3.0)
        why = "internal high broken" + (", swing high broken" if beyond_swing else ", swing high not yet")
    return ComponentScore("market_structure", pts, 20.0, why,
                          {"beyond_swing": beyond_swing, "decisive": decisive})


def _internal_trigger(Ht: np.ndarray, ip: List[Pivot], lo_idx: int, hi_idx: int,
                      length: int) -> Tuple[float, bool]:
    """Last internal swing high strictly inside (lo_idx, hi_idx) from the finer
    pivot set. Without one (a straight decline) fall back to the highest high of
    the candles leading into the final low. Returns (level, is_real_pivot)."""
    cand = [p for p in ip if p.kind == 1 and lo_idx < p.idx < hi_idx]
    if cand:
        return float(cand[-1].price), True
    start = max(lo_idx + 1, hi_idx - length)
    return float(np.max(Ht[start:hi_idx + 1])), False


# ── the analysis ─────────────────────────────────────────────────────────

def _clean_start(ts: np.ndarray, o: np.ndarray, h: np.ndarray, l: np.ndarray,
                 c: np.ndarray, s0: int, end: int, interval_sec: int) -> int:
    """First index after the last gap / duplicate / non-finite / impossible
    candle inside [s0, end]; structures are never built across a break."""
    start = s0
    seg = slice(s0, end + 1)
    ts_w = ts[seg].astype(np.int64)
    if len(ts_w) > 1:
        gaps = np.flatnonzero(np.diff(ts_w) != interval_sec)
        if len(gaps):
            start = max(start, s0 + int(gaps[-1]) + 1)
    bad = ~(np.isfinite(o[seg]) & np.isfinite(h[seg]) & np.isfinite(l[seg]) & np.isfinite(c[seg]))
    with np.errstate(invalid="ignore"):
        bad |= (h[seg] < l[seg]) | (l[seg] <= 0)
    idx = np.flatnonzero(bad)
    if len(idx):
        start = max(start, s0 + int(idx[-1]) + 1)
    return start


def analyze(pair: str, ts: np.ndarray, open_: np.ndarray, high: np.ndarray, low: np.ndarray,
            close: np.ndarray, volume: np.ndarray, end: Optional[int] = None,
            params: Optional[ElliottParams] = None,
            external_gate: Optional[Mapping[str, Optional[str]]] = None) -> Analysis:
    """Find wave candidates as they were knowable at candle `end` (last CLOSED
    candle). Only candles 0..end are read.

    `external_gate` lets the caller veto a whole direction with a reason, e.g.
    {"buy": None, "sell": "ichimoku"}: a vetoed direction still produces its
    setups, but as status "rejected" / reason "legacy_gate:<why>", so the decision
    and its reason are recorded. None = no external gate (feature shows "n/a")."""
    P = params or ElliottParams()
    n_all = len(ts)
    diag: Dict[str, Any] = {"rejects": {}}
    if end is None:
        end = n_all - 1
    if n_all == 0 or end < 0 or end >= n_all:
        diag["skipped"] = "bad_end_index"
        return Analysis([], None, 0, 0, max(end, 0), diag)

    o_all = np.asarray(open_, dtype=np.float64)
    h_all = np.asarray(high, dtype=np.float64)
    l_all = np.asarray(low, dtype=np.float64)
    c_all = np.asarray(close, dtype=np.float64)
    v_all = np.asarray(volume, dtype=np.float64)
    ts_all = np.asarray(ts, dtype=np.int64)

    s0 = max(0, end - (P.lookback + P.warmup) + 1)
    s0 = _clean_start(ts_all, o_all, h_all, l_all, c_all, s0, end, P.interval_sec)
    if end - s0 + 1 < P.min_bars:
        diag["skipped"] = "insufficient_clean_history"
        return Analysis([], None, 0, s0, end, diag)

    seg = slice(s0, end + 1)
    t, o, h, l, c, v = ts_all[seg], o_all[seg], h_all[seg], l_all[seg], c_all[seg], v_all[seg]
    n = len(t)
    e = n - 1                                   # local index of the last closed candle

    atr = _wilder_atr(h, l, c, P.atr_len)
    atr_e = float(atr[e])
    if not (math.isfinite(atr_e) and atr_e > 0):
        diag["skipped"] = "atr_unavailable"
        return Analysis([], None, 0, s0, end, diag)
    hist = _macd_hist(c)
    rsi = _rsi(c)

    first = max(P.atr_len, P.pivot_len)
    raw = confirmed_pivots(h, l, P.pivot_len, first)
    raw_in = confirmed_pivots(h, l, P.internal_pivot_len, first)
    # structural lookback applies to the pivots, indicators keep their warm-up
    cut = max(0, e - P.lookback + 1)
    raw = [p for p in raw if p.idx >= cut]
    pv = normalize_pivots(raw, atr, P.min_swing_atr)
    diag["n_pivots"] = len(pv)

    bias = htf_bias(ts_all, c_all, end, P.interval_sec, P.htf_min_hours, P.htf_ema_len)

    setups: List[WaveSetup] = []
    events: List[WaveEvent] = []

    def reject(key: str, why: str) -> None:
        diag["rejects"][key] = why

    for s, direction in ((1, BUY), (-1, SELL)):
        # transformed space: higher is always "with the trade"
        Ht = h if s > 0 else -l
        Lt = l if s > 0 else -h
        Ot, Ct = s * o, s * c
        tp = [Pivot(p.idx, s * p.price, s * p.kind, p.confirm_idx) for p in pv]
        ip = [Pivot(p.idx, s * p.price, s * p.kind, p.confirm_idx) for p in raw_in]
        evaluators = []
        if P.enable_wave3:
            evaluators.append(("wave3", _eval_wave3))
        if P.enable_wave5:
            evaluators.append(("wave5", _eval_wave5))
        if P.enable_abc:
            evaluators.append(("abc", _eval_abc))
        gate_reason = external_gate.get(direction) if external_gate is not None else None
        ctx = _Ctx(pair=pair, P=P, s=s, direction=direction, off=s0, e=e, ts=t, o=o, h=h, l=l,
                   c=c, v=v, Ht=Ht, Lt=Lt, Ot=Ot, Ct=Ct, atr=atr, atr_e=atr_e, hist=hist,
                   rsi=rsi, tp=tp, ip=ip, bias=bias, block=gate_reason,
                   gate_state=("n/a" if external_gate is None else (gate_reason or "pass")))
        if tp and tp[-1].kind == 1:
            # the last confirmed pivot is a high: a finished impulse, not a setup start
            if P.enable_exhaustion:
                try:
                    ev = _eval_exhaustion(ctx)
                except (IndexError, ValueError, ZeroDivisionError, FloatingPointError):
                    ev = None
                if ev is not None:
                    events.append(ev)
        if not tp or tp[-1].kind != -1:
            reject(f"{direction}", "last_pivot_not_a_low")
            continue
        for fam, fn in evaluators:
            try:
                res = fn(ctx)
            except (IndexError, ValueError, ZeroDivisionError, FloatingPointError):
                reject(f"{fam}_{direction}", "evaluation_error")
                continue
            if isinstance(res, str):
                reject(f"{fam}_{direction}", res)
            elif res is not None:
                setups.append(res)

    rank = {"eligible": 0, "watch": 1, "rejected": 2}
    setups.sort(key=lambda x: (rank[x.status], family_rank(x.family, P.primary_family), -x.score))
    primary = setups[0] if setups else None
    if primary is not None:
        others = setups[1:]
        # structurally valid counts on the same pivots that have not triggered
        pending = [f"{k}:{why}" for k, why in diag["rejects"].items()
                   if not (why.startswith("rule:") or why in ("no_pattern", "last_pivot_not_a_low",
                                                              "evaluation_error"))]
        primary.alt_counts = [f"{x.family}_{x.direction}:{x.status}:{x.score:.0f}" for x in others] + pending
        gap = (primary.score - max(x.score for x in others)) if others else None
        why_primary = (f"status {primary.status}; family order {P.primary_family} first, then wave3>wave5>abc; "
                       f"then score" + (f" (gap {gap:+.1f} to best alternative)" if gap is not None
                                        else " (no competing triggered count)"))
        primary.features["elliott_alt_counts"] = list(primary.alt_counts)
        primary.features["elliott_primary_reason"] = why_primary
        primary.features["elliott_score_gap"] = round(gap, 2) if gap is not None else None
    return Analysis(setups, primary, len(pv), s0, end, diag, events)


# ── candidate evaluators ─────────────────────────────────────────────────

@dataclass
class _Ctx:
    pair: str
    P: ElliottParams
    s: int
    direction: str
    off: int
    e: int
    ts: np.ndarray
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    v: np.ndarray
    Ht: np.ndarray
    Lt: np.ndarray
    Ot: np.ndarray
    Ct: np.ndarray
    atr: np.ndarray
    atr_e: float
    hist: np.ndarray
    rsi: np.ndarray
    tp: List[Pivot]
    ip: List[Pivot]
    bias: Optional[int]
    block: Optional[str] = None         # external (legacy) gate veto for this direction
    gate_state: str = "n/a"             # "n/a" | "pass" | <veto reason>


def _label(ctx: _Ctx, p: Pivot) -> Tuple[int, int, float]:
    return (ctx.off + p.idx, int(ctx.ts[p.idx]), float(ctx.s * p.price))


def _ids(pair: str, direction: str, family: str, pivots: Sequence[Tuple[int, float]],
         signal_ts: int) -> Tuple[str, str]:
    """(structure_id, event_id). Canonical timestamps and UNROUNDED prices, plus
    the engine version. structure_id identifies the wave structure and is the
    deduplication key (a persisting count must not re-alert); event_id adds the
    signal candle and identifies this one decision."""
    canon = "|".join([ENGINE_VERSION, pair, direction, family]
                     + [f"{int(t)}:{float(px)!r}" for t, px in pivots])
    sid = "ews-" + hashlib.sha1(canon.encode()).hexdigest()[:16]
    eid = "ewe-" + hashlib.sha1(f"{canon}|{int(signal_ts)}".encode()).hexdigest()[:16]
    return sid, eid


def _finish(ctx: _Ctx, family: str, pts: Sequence[Pivot], names: Sequence[str], trigger: float,
            stop_t: float, invalidation_t: float, targets_t: Sequence[float],
            structure: ComponentScore, fib: ComponentScore, bos: ComponentScore,
            prev_leg: Tuple[int, int], now_leg: Tuple[int, int], extra: Dict[str, Any],
            age: int, rr_target: int = 0) -> WaveSetup:
    P, s, e = ctx.P, ctx.s, ctx.e
    entry_t = float(ctx.Ct[e])
    entry_dist = (entry_t - trigger) / ctx.atr_e
    risk = entry_t - stop_t
    reward = (targets_t[rr_target] - entry_t) if len(targets_t) > rr_target else float("nan")
    rr = (reward / risk) if risk > 0 and math.isfinite(reward) else float("nan")

    mom = score_momentum(s, ctx.hist, ctx.rsi, float(ctx.o[e]), float(ctx.h[e]),
                         float(ctx.l[e]), float(ctx.c[e]), e)
    vol = score_volume(ctx.v, e, prev_leg, now_leg)
    htf = score_htf(ctx.bias, s)
    detail = {c_.name: c_ for c_ in (structure, fib, bos, mom, vol, htf)}
    comps = {k: round(v.score, 2) for k, v in detail.items()}
    score = round(sum(comps.values()), 2)

    reason = ""
    if ctx.block:
        reason = f"legacy_gate:{ctx.block}"
    elif entry_dist > P.max_entry_distance_atr:
        reason = "late_entry"
    elif not (math.isfinite(rr) and rr >= P.min_rr):
        reason = "reward_risk"
    if reason:
        status = "rejected"
    elif score >= P.min_score:
        status = "eligible"
    elif score >= P.watch_score:
        status, reason = "watch", "score_watch"
    else:
        status, reason = "rejected", "score"

    inv_dist_pct = abs(entry_t - invalidation_t) / abs(entry_t) * 100.0 if entry_t else float("nan")
    key_pivots = [(int(ctx.ts[p.idx]), ctx.s * p.price) for p in pts]
    sid, eid = _ids(ctx.pair, ctx.direction, family, key_pivots, int(ctx.ts[e]))
    labels = {nm: _label(ctx, p) for nm, p in zip(names, pts)}
    feats: Dict[str, Any] = {
        "elliott_engine_version": ENGINE_VERSION,
        "elliott_feature_version": FEATURE_VERSION,
        "elliott_legacy_gate": ctx.gate_state,
        "elliott_wave_family": family,
        "elliott_wave_direction": ctx.direction,
        "elliott_wave_count": extra.pop("count_text", ""),
        "elliott_wave_score": score,
        "elliott_wave_status": status,
        "elliott_wave_reason": reason,
        "elliott_wave_duration": int(e - pts[0].idx),
        "elliott_wave_fib_quality": round(fib.score / fib.max_score, 3),
        "elliott_wave_bos_confirmed": True,
        "elliott_wave_ppo_alignment": mom.values["hist"] > 0,
        "elliott_wave_rsi_alignment": mom.values["rsi"] > 0,
        "elliott_wave_volume_expansion": (vol.score >= 6.0) if vol.available else None,
        "elliott_wave_volume_available": vol.available,
        "elliott_wave_htf_alignment": htf.values.get("aligned"),
        "elliott_wave_entry_distance_atr": round(entry_dist, 3),
        "elliott_wave_invalidation_distance_pct": round(inv_dist_pct, 3),
        "elliott_wave_id": eid,
        "elliott_structure_id": sid,
        "elliott_wave_age_bars": int(age),
        "elliott_wave_rr": round(rr, 2) if math.isfinite(rr) else None,
        "elliott_wave_components": dict(comps),
        "elliott_wave_component_detail": {
            k: {"score": round(v.score, 2), "max": v.max_score, "reason": v.reason,
                "available": v.available} for k, v in detail.items()},
    }
    feats.update(extra)
    return WaveSetup(
        family=family, direction=ctx.direction, status=status, reason=reason, score=score,
        components=comps, labels=labels, event_id=eid, structure_id=sid,
        signal_idx=ctx.off + e, signal_ts=int(ctx.ts[e]), entry_price=float(ctx.c[e]),
        trigger_level=float(s * trigger), entry_distance_atr=round(entry_dist, 3),
        invalidation=float(s * invalidation_t), stop=float(s * stop_t),
        targets=tuple(float(s * x) for x in targets_t), rr=(round(rr, 2) if math.isfinite(rr) else 0.0),
        atr=ctx.atr_e, age_bars=int(age), htf_bias=htf.values.get("aligned"), detail=detail,
        features=feats, pair=ctx.pair,
    )


def _eval_exhaustion(ctx: _Ctx) -> Optional[WaveEvent]:
    """A complete, rule-valid five-wave impulse whose Wave 5 high has JUST been
    confirmed, showing at least one exhaustion sign that matters (momentum
    divergence against Wave 3, or a truncated fifth). Emitted once, on the candle
    that confirms the Wave 5 pivot. It is an event to record and follow up, never a
    reversal trade by itself."""
    tp, e = ctx.tp, ctx.e
    if len(tp) < 6 or [p.kind for p in tp[-6:]] != [-1, 1, -1, 1, -1, 1]:
        return None
    p0, p1, p2, p3, p4, p5 = tp[-6:]
    if e != p5.confirm_idx:
        return None
    if check_impulse_rules([p.price for p in (p0, p1, p2, p3, p4, p5)], True) is not None:
        return None
    w1, w3, w5 = p1.price - p0.price, p3.price - p2.price, p5.price - p4.price
    truncated = p5.price <= p3.price
    # indicators live in original price space: multiply by the side so that "lower
    # in the direction of the impulse" means the same thing for up and down impulses
    sg = ctx.s
    hist_div = (not truncated) and bool(sg * ctx.hist[p5.idx] < sg * ctx.hist[p3.idx])
    rsi_div = (not truncated) and bool(
        math.isfinite(ctx.rsi[p5.idx]) and math.isfinite(ctx.rsi[p3.idx])
        and sg * ctx.rsi[p5.idx] < sg * ctx.rsi[p3.idx])
    weak_w5 = bool(w5 <= w1)
    v3 = ctx.v[p2.idx:p3.idx + 1]
    v5 = ctx.v[p4.idx:p5.idx + 1]
    vol_known = bool(len(v3) and len(v5) and np.all(np.isfinite(v3)) and np.all(np.isfinite(v5))
                     and float(np.mean(v3)) > 0)
    vol_fade = bool(vol_known and float(np.mean(v5)) < float(np.mean(v3)))
    if not (hist_div or rsi_div or truncated):
        return None
    signs = {"hist_divergence": hist_div, "rsi_divergence": rsi_div, "truncated_fifth": truncated,
             "weak_wave5": weak_w5, "volume_fade": vol_fade}
    score = int(sum(signs.values()))
    direction = SELL if ctx.s > 0 else BUY       # the impulse's opposite side
    key_pivots = [(int(ctx.ts[p.idx]), ctx.s * p.price) for p in (p0, p1, p2, p3, p4, p5)]
    sid, eid = _ids(ctx.pair, direction, "exhaustion", key_pivots, int(ctx.ts[e]))
    atr_e = ctx.atr_e
    entry = float(ctx.c[e])
    feats: Dict[str, Any] = {
        "elliott_engine_version": ENGINE_VERSION,
        "elliott_feature_version": FEATURE_VERSION,
        "elliott_event_kind": "exhaustion",
        "elliott_wave_family": "exhaustion",
        "elliott_wave_direction": direction,
        "elliott_wave_id": eid,
        "elliott_structure_id": sid,
        "elliott_exhaustion_signs": dict(signs),
        "elliott_exhaustion_score": score,
        "elliott_wave1_length_atr": round(w1 / atr_e, 3),
        "elliott_wave3_length_atr": round(w3 / atr_e, 3),
        "elliott_wave5_length_atr": round(w5 / atr_e, 3),
        "elliott_wave5_vs_wave3": round(w5 / w3, 3) if w3 > 0 else None,
        "elliott_volume_available": vol_known,
    }
    return WaveEvent(
        kind="exhaustion", direction=direction, signs=signs, score=score,
        labels={nm: _label(ctx, p) for nm, p in zip(("P0", "P1", "P2", "P3", "P4", "P5"),
                                                    (p0, p1, p2, p3, p4, p5))},
        event_id=eid, structure_id=sid, signal_idx=ctx.off + e, signal_ts=int(ctx.ts[e]),
        entry_price=entry, invalidation=float(ctx.s * p5.price),
        targets=(float(ctx.s * (p5.price - 0.382 * w5)), float(ctx.s * (p5.price - 0.618 * w5))),
        atr=atr_e, features=feats, pair=ctx.pair,
    )


def _eval_wave3(ctx: _Ctx) -> Any:
    tp, e, P = ctx.tp, ctx.e, ctx.P
    if len(tp) < 3 or [p.kind for p in tp[-3:]] != [-1, 1, -1]:
        return "no_pattern"
    p0, p1, p2 = tp[-3:]
    rule = check_impulse_rules([p0.price, p1.price, p2.price], True)
    if rule:
        return f"rule:{rule}"
    if e > p2.idx:
        after_low = float(np.min(ctx.Lt[p2.idx + 1:e + 1]))
        if after_low <= p0.price:
            return "invalidated"
        if after_low < p2.price:
            return "wave2_still_extending"
    age = e - p2.confirm_idx
    if age < 0:
        return "pivot_not_confirmed"
    if age > P.max_age_bars:
        return "stale"
    trig, real = _internal_trigger(ctx.Ht, ctx.ip, p1.idx, p2.idx, P.pivot_len)
    if not (ctx.Ct[e] > trig and ctx.Ct[e] > ctx.Ot[e]):
        return "no_trigger"
    w1 = p1.price - p0.price
    m = {"w1_atr": w1 / ctx.atr_e, "retr2": (p1.price - p2.price) / w1,
         "dur1": p1.idx - p0.idx, "dur2": p2.idx - p1.idx, "real_trigger": real}
    bos = score_bos("wave3", bool(ctx.Ct[e] > p1.price), False)
    extra = {
        "count_text": "1-2 complete, 3 starting",
        "elliott_wave1_length_atr": round(m["w1_atr"], 2),
        "elliott_wave2_retracement": round(m["retr2"], 3),
        "elliott_wave3_extension": None, "elliott_wave4_retracement": None,
        "elliott_wave5_extension": None,
        "elliott_trigger_is_pivot": bool(real),
    }
    return _finish(ctx, "wave3", [p0, p1, p2], ["W0", "W1", "W2"], trig,
                   p2.price - P.stop_buffer_atr * ctx.atr_e, p0.price,
                   [p2.price + 1.618 * w1, p2.price + 2.618 * w1],
                   score_structure("wave3", m), score_fibonacci("wave3", m), bos,
                   (p1.idx, p2.idx), (p2.idx + 1, e), extra, age)


def _eval_wave5(ctx: _Ctx) -> Any:
    tp, e, P = ctx.tp, ctx.e, ctx.P
    if len(tp) < 5 or [p.kind for p in tp[-5:]] != [-1, 1, -1, 1, -1]:
        return "no_pattern"
    p0, p1, p2, p3, p4 = tp[-5:]
    rule = check_impulse_rules([p0.price, p1.price, p2.price, p3.price, p4.price], True)
    if rule:
        return f"rule:{rule}"
    if e > p4.idx:
        after_low = float(np.min(ctx.Lt[p4.idx + 1:e + 1]))
        if after_low <= p1.price:
            return "invalidated"
        if after_low < p4.price:
            return "wave4_still_extending"
    age = e - p4.confirm_idx
    if age < 0:
        return "pivot_not_confirmed"
    if age > P.max_age_bars_wave5:
        return "stale"
    trig = p3.price                                         # break of the Wave 3 extreme
    if not (ctx.Ct[e] > trig and ctx.Ct[e] > ctx.Ot[e]):
        return "no_trigger"
    w1, w3 = p1.price - p0.price, p3.price - p2.price
    m = {"w1_atr": w1 / ctx.atr_e, "retr2": (p1.price - p2.price) / w1,
         "retr4": (p3.price - p4.price) / w3, "ext3": w3 / w1,
         "dur2": p2.idx - p1.idx, "dur4": p4.idx - p3.idx}
    bos = score_bos("wave5", True, bool((ctx.Ct[e] - trig) >= 0.25 * ctx.atr_e))
    extra = {
        "count_text": "1-2-3-4 complete, 5 starting",
        "elliott_wave1_length_atr": round(m["w1_atr"], 2),
        "elliott_wave2_retracement": round(m["retr2"], 3),
        "elliott_wave3_extension": round(m["ext3"], 3),
        "elliott_wave4_retracement": round(m["retr4"], 3),
        "elliott_wave5_extension": None,
        # Wave 3 shorter than Wave 1 is still a valid count while Wave 5 is
        # unknown (it only fails if Wave 5 ends up longer than Wave 3).
        "elliott_wave3_shorter_than_wave1": bool(w3 < w1),
    }
    return _finish(ctx, "wave5", [p0, p1, p2, p3, p4], ["W0", "W1", "W2", "W3", "W4"], trig,
                   p4.price - P.stop_buffer_atr * ctx.atr_e, p1.price,
                   [p4.price + 1.0 * w1, p4.price + 1.618 * w1],
                   score_structure("wave5", m), score_fibonacci("wave5", m), bos,
                   (p3.idx, p4.idx), (p4.idx + 1, e), extra, age)


def _eval_abc(ctx: _Ctx) -> Any:
    tp, e, P = ctx.tp, ctx.e, ctx.P
    if len(tp) < 4 or [p.kind for p in tp[-4:]] != [1, -1, 1, -1]:
        return "no_pattern"
    s_, a, b, c = tp[-4:]
    z = tp[-5] if len(tp) >= 5 and tp[-5].kind == -1 else None
    rule = check_abc_rules(z.price if z else None, s_.price, a.price, b.price, c.price, True)
    if rule:
        return f"rule:{rule}"
    if e > c.idx and float(np.min(ctx.Lt[c.idx + 1:e + 1])) < c.price:
        return "wave_c_still_extending"
    age = e - c.confirm_idx
    if age < 0:
        return "pivot_not_confirmed"
    if age > P.max_age_bars:
        return "stale"
    trig, real = _internal_trigger(ctx.Ht, ctx.ip, b.idx, c.idx, P.pivot_len)
    if not (ctx.Ct[e] > trig and ctx.Ct[e] > ctx.Ot[e]):
        return "no_trigger"
    leg_a = s_.price - a.price
    m = {"prior_ratio": ((s_.price - z.price) / leg_a) if z else 0.0,
         "ext_c": (b.price - c.price) / leg_a, "retr_b": (b.price - a.price) / leg_a,
         "a_atr": leg_a / ctx.atr_e}
    bos = score_bos("abc", bool(ctx.Ct[e] > b.price), False)
    extra = {
        "count_text": "A-B-C complete",
        "elliott_wave1_length_atr": round(m["a_atr"], 2),
        "elliott_wave2_retracement": None, "elliott_wave3_extension": None,
        "elliott_wave4_retracement": None, "elliott_wave5_extension": None,
        "elliott_abc_b_retracement": round(m["retr_b"], 3),
        "elliott_abc_c_extension": round(m["ext_c"], 3),
        "elliott_abc_leg_duration": int(a.idx - s_.idx),
    }
    # B first; R/R is judged against S (trend resumption)
    return _finish(ctx, "abc", [s_, a, b, c], ["S", "A", "B", "C"], trig,
                   c.price - P.stop_buffer_atr * ctx.atr_e, c.price, [b.price, s_.price],
                   score_structure("abc", m), score_fibonacci("abc", m), bos,
                   (b.idx, c.idx), (c.idx + 1, e), extra, age, rr_target=1)


def evaluate_progress(rec: Mapping[str, Any], ts: np.ndarray, high: np.ndarray, low: np.ndarray,
                      close: np.ndarray, volume: np.ndarray, end: int,
                      max_bars: int = 48) -> Dict[str, Any]:
    """What happened AFTER a stored signal, from the candles that followed it.

    Pure and stateless: the same record and candles always give the same answer,
    and nothing here can change the original signal (it only describes what came
    later; callers append the result as a separate event).

    `rec` needs: signal_ts, direction ("buy"/"sell"), entry, atr, invalidation,
    targets (sequence), and optionally stop and vol_ref (mean volume up to the
    signal). Within one candle the adverse outcome is assumed to come first
    (pessimistic), exactly like the bot's stop/target replay.

    Returns state "open" or "final", outcome one of
      open | t2 | stop | invalidated | expired | missing_signal_candle
    first-touch bar counts (candles after the signal), excursions in % and ATR
    until the trade ended, and the later volume relative to `vol_ref`."""
    ts_a = np.asarray(ts, dtype=np.int64)
    h_a = np.asarray(high, dtype=np.float64)
    l_a = np.asarray(low, dtype=np.float64)
    c_a = np.asarray(close, dtype=np.float64)
    v_a = np.asarray(volume, dtype=np.float64)
    end = int(min(end, len(ts_a) - 1))
    sig = int(rec["signal_ts"])
    i0 = int(np.searchsorted(ts_a[:end + 1], sig))
    if end < 0 or i0 > end or int(ts_a[i0]) != sig:
        return {"state": "open", "outcome": "missing_signal_candle"}
    buy = rec["direction"] == BUY
    entry, atr = float(rec["entry"]), float(rec["atr"])
    inv = float(rec["invalidation"])
    stop = rec.get("stop")
    stop = float(stop) if stop is not None else None
    targets = [float(x) for x in (rec.get("targets") or ())]
    first: Dict[str, Optional[int]] = {"t1": None, "t2": None, "stop": None, "invalidated": None}
    mfe = mae = 0.0
    outcome = "open"
    j_end = i0
    last = min(end, i0 + int(max_bars))
    for j in range(i0 + 1, last + 1):
        k = j - i0
        hi, lo, cl = float(h_a[j]), float(l_a[j]), float(c_a[j])
        fav = (hi - entry) if buy else (entry - lo)
        adv = (entry - lo) if buy else (hi - entry)
        inv_hit = (cl < inv) if buy else (cl > inv)
        stop_hit = stop is not None and ((lo <= stop) if buy else (hi >= stop))
        mae = max(mae, adv)
        j_end = j
        if inv_hit or stop_hit:
            if stop_hit:
                first["stop"] = first["stop"] or k
            if inv_hit:
                first["invalidated"] = first["invalidated"] or k
            outcome = "stop" if stop_hit else "invalidated"
            break
        mfe = max(mfe, fav)
        for ti, key in ((0, "t1"), (1, "t2")):
            if len(targets) > ti and first[key] is None and ((hi >= targets[ti]) if buy else (lo <= targets[ti])):
                first[key] = k
        if first["t2"] is not None:
            outcome = "t2"
            break
    else:
        if last - i0 >= int(max_bars):
            outcome = "expired"
    post = v_a[i0 + 1:j_end + 1]
    vol_ref = rec.get("vol_ref")
    vol_ratio = None
    if vol_ref and float(vol_ref) > 0 and len(post) and np.all(np.isfinite(post)):
        vol_ratio = round(float(np.mean(post)) / float(vol_ref), 3)
    return {
        "state": "open" if outcome == "open" else "final",
        "outcome": outcome,
        "bars": int(j_end - i0),
        "last_ts": int(ts_a[j_end]),
        "first_bars": first,
        "t1_before_adverse": first["t1"] is not None,
        "mfe_pct": round(mfe / entry * 100.0, 4) if entry else None,
        "mae_pct": round(mae / entry * 100.0, 4) if entry else None,
        "mfe_atr": round(mfe / atr, 3) if atr > 0 else None,
        "mae_atr": round(mae / atr, 3) if atr > 0 else None,
        "volume_ratio": vol_ratio,
        "volume_expanded": (vol_ratio >= 1.2) if vol_ratio is not None else None,
    }


def replay_events(pair: str, ts: np.ndarray, open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                  close: np.ndarray, volume: np.ndarray, start: int, stop: Optional[int] = None,
                  params: Optional[ElliottParams] = None) -> List[WaveSetup]:
    """Point-in-time replay: for every candle `end` in [start, stop] run analyze()
    with ONLY candles 0..end visible and keep the first decision per
    structure_id (what a live run would have alerted on first). Use it on
    historical candles to measure the module before trusting it."""
    stop = (len(ts) - 1) if stop is None else stop
    seen: Dict[str, WaveSetup] = {}
    for end in range(start, stop + 1):
        a = analyze(pair, ts[:end + 1], open_[:end + 1], high[:end + 1], low[:end + 1],
                    close[:end + 1], volume[:end + 1], end=end, params=params)
        if a.primary is not None and a.primary.structure_id not in seen:
            seen[a.primary.structure_id] = a.primary
    return list(seen.values())
