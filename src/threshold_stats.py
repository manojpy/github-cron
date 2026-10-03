"""Statistical primitives, bracket/EV maths and row-level helpers (lowest layer: depends on nothing else in the threshold stack).

Split out of the original module without logic changes; the original
module remains as a facade re-exporting every name."""
from __future__ import annotations
import math
import time
import random
import statistics
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Tuple
from bot_config import cfg

Row = Dict[str, Any]

CapRow = Tuple[float, int, float, float]  # (cap, n, wr, wilson_lower_bound)

_EV_FIRST_CACHE: Dict[Tuple[int, Any, Any, int, int, float, float], Dict[str, Any]] = {}

def _ev_first_cache_key(
    rows: List[Row], min_sample: int, fee_pct: float, slippage_pct: float,
) -> Tuple[int, Any, Any, int, int, float, float]:
    n = len(rows)
    first_ts = rows[0].get("entry_ts", 0) if n else 0
    last_ts = rows[-1].get("entry_ts", 0) if n else 0
    # n + first/last ts alone collide for different row sets that share a size
    # and time span (e.g. two alert keys firing on the same candles, or regime
    # segments of one alert), returning another set's cached EV. The content
    # checksum makes the key specific to the actual rows and their outcomes.
    content = hash(tuple(
        (r.get("entry_ts", 0), r.get("pair"), r.get("alert_key"),
         r.get("direction"), bool(r.get("win")), r.get("outcome_reason"))
        for r in rows
    ))
    return (n, first_ts, last_ts, content, min_sample, fee_pct, slippage_pct)

def clear_ev_first_cache() -> None:
    """Drop all memoised ev_first_objective() results. Call at the top of
    each Brain report cycle to bound memory growth in a long-running
    process; safe to call anytime since the cache is purely an optimization."""
    _EV_FIRST_CACHE.clear()

def wilson_ci(wins: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    """Wilson score interval — reliable even for small n. Returns
    (lower_bound, upper_bound, raw_p)."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, centre - margin), min(1.0, centre + margin), p

def two_proportion_p_value(wins_a: int, n_a: int, wins_b: int, n_b: int) -> float:
    """Two-sided two-proportion z-test p-value (no SciPy).

    Uses the pooled-variance test statistic:
        z = (p_a - p_b) / sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))
    then two-sided normal tail via the error function.

    Returns 1.0 when either sample is empty or the pooled rate is
    degenerate (0 or 1) — "no evidence" is the correct conservative
    answer there and keeps BH from rejecting on noise.
    """
    if n_a <= 0 or n_b <= 0:
        return 1.0
    p_a = wins_a / n_a
    p_b = wins_b / n_b
    p_pool = (wins_a + wins_b) / (n_a + n_b)
    if p_pool <= 0.0 or p_pool >= 1.0:
        return 1.0
    se = math.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n_a + 1.0 / n_b))
    if se <= 0.0:
        return 1.0
    z = abs(p_a - p_b) / se
    # Two-sided normal tail: p = 2 * (1 - Phi(z)) = erfc(z / sqrt(2))
    return math.erfc(z / math.sqrt(2.0))

def one_proportion_p_value(wins: int, n: int, p0: float) -> float:
    """Two-sided one-sample proportion z-test against a fixed reference rate p0.

    Used for claims of the form "the observed win rate at X is (or is not)
    consistent with a target/policy rate" — disable_alert, recovered_alert,
    parameter_autopsy. Two-sided (not one-sided) is deliberate: the BH
    correction downstream assumes all p-values are valid under their
    respective nulls, and using a mix of one- and two-sided tests skews
    the ordering that BH depends on. The claim is directional, but the
    Wilson-upper/lower gate at emission time already enforced the
    direction; the p-value is a secondary confirmation.

    Returns 1.0 when the test is degenerate (empty sample, p0 outside the
    open unit interval, or zero standard error) — conservative, and it
    keeps BH from rejecting on a computation artifact.
    """
    if n <= 0 or not (0.0 < p0 < 1.0):
        return 1.0
    p_hat = wins / n
    se = math.sqrt(p0 * (1.0 - p0) / n)
    if se <= 0.0:
        return 1.0
    z = abs(p_hat - p0) / se
    return math.erfc(z / math.sqrt(2.0))

def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar test on the discordant pair counts (b, c).

    Applies when two classifiers are run on the same rows and we want to
    know whether they disagree systematically — e.g. close_win vs mfe_win
    in three_metric_evaluation. This is NOT a two-proportion test: the
    concordant pairs (both-win and neither-win) carry no information about
    which classifier is better, only the discordant b/c cells do. Running
    a two-proportion test on the marginal rates would overstate the
    sample size by treating concordant pairs as evidence, which is the
    error the earlier wiring made.

    Exact for n_discordant <= 200 via math.comb (0.5**n_d underflows
    beyond ~1000, but by then p is already effectively 0). Normal
    approximation with continuity correction above that threshold.

    Returns 1.0 when there are no discordant pairs.
    """
    if b < 0 or c < 0:
        return 1.0
    n_d = b + c
    if n_d == 0:
        return 1.0
    k_min = min(b, c)

    if n_d > 200:
        # Continuity-corrected normal approximation:
        # z = (|b - n_d/2| - 0.5) / sqrt(n_d)/2
        z = (abs(b - n_d / 2) - 0.5) / (math.sqrt(n_d) / 2)
        z = max(0.0, z)
        return math.erfc(z / math.sqrt(2.0))

    tail = sum(math.comb(n_d, i) for i in range(k_min + 1)) * (0.5 ** n_d)
    return min(1.0, 2.0 * tail)

def benjamini_hochberg(p_values: List[float], alpha: float = 0.10) -> List[bool]:
    """Benjamini-Hochberg FDR control.

    Given a list of p-values (one per hypothesis tested), returns a parallel
    boolean list where True = "reject the null" at the target FDR level.

    The BH procedure: sort p-values ascending, find the largest rank k such
    that p_(k) <= (k/m) * alpha, then reject every hypothesis with
    p <= p_(k). Under independence (or PRDS), the expected proportion of
    false discoveries among rejections is <= alpha.

    Why this matters here: the brain evaluates hundreds of hypotheses per
    report. Without correction, the "disable alert", "poison pair", and
    "calibration divergence" recommendations are dominated by chance hits
    as the search space grows.

    Interpretation note: alpha=0.10 (not 0.05) is deliberate. FDR at 10%
    says "at most 1 in 10 of my flags is noise", which is a reasonable
    tolerance for a human-reviewed report where acting on a false positive
    is cheap compared to missing a real signal.
    """
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda t: t[1])
    reject = [False] * m
    cutoff_rank = 0
    for rank, (_, p) in enumerate(indexed, start=1):
        if p <= alpha * rank / m:
            cutoff_rank = rank
    if cutoff_rank > 0:
        cutoff_p = indexed[cutoff_rank - 1][1]
        for orig_idx, p in indexed:
            if p <= cutoff_p:
                reject[orig_idx] = True
    return reject

def recency_weight(entry_ts: Optional[float], now_ts: float, decay_days: float = 7.0) -> float:
    """Exponential recency weight: exp(-age_days / decay_days). ... NOTE:
    decay_days is an exponential time constant, not a strict half-life —
    the true half-life is decay_days * ln(2) (~4.85 days at the default 7).
    Under this formula an outcome from right now is weighted ~e (2.72x)
    more than one exactly decay_days old, and one 3x that age is weighted
    ~e^-3 (~5%)."""
    if not entry_ts or decay_days <= 0:
        return 1.0
    age_days = max(0.0, (now_ts - float(entry_ts)) / 86400.0)
    return math.exp(-age_days / decay_days)

def weighted_win_rate(
    rows: List[Row], now_ts: Optional[float] = None, decay_days: float = 7.0,
) -> Tuple[Optional[float], float, float, float]:
    """Recency-weighted win rate + a weighted Wilson-CI band. Plain recency
    weighting only — every win counts as exactly 1.0 regardless of size.
    For bonus-adjusted weighting (overshoot wins count for more), use
    weighted_win_rate_with_bonus instead.

    Returns (weighted_wr, n_eff, wilson_lo, wilson_hi).
    """
    if now_ts is None:
        now_ts = time.time()
    if not rows:
        return None, 0.0, 0.0, 0.0
    sum_w = sum_w2 = sum_ww = 0.0
    for r in rows:
        w = recency_weight(r.get("entry_ts"), now_ts, decay_days)
        sum_w += w
        sum_w2 += w * w
        if r["win"]:
            sum_ww += w
    if sum_w <= 0:
        return None, 0.0, 0.0, 0.0
    weighted_wr = min(sum_ww / sum_w, 1.0)
    n_eff = (sum_w ** 2) / sum_w2 if sum_w2 > 0 else 0.0
    lo, hi, _ = wilson_ci(round(weighted_wr * n_eff), max(1, round(n_eff)))
    return weighted_wr, n_eff, lo, hi

def weighted_win_rate_with_bonus(
    rows: List[Row],
    now_ts: Optional[float] = None,
    decay_days: float = 7.0,
    n_boot: int = 400,
    seed: int = 42,
) -> Tuple[Optional[float], float, float, float]:
    """Win rate where bonus wins (exceeded 1:2 target) count for more,
    with a proper bootstrap CI instead of a re-parameterized Wilson.

    The old approach plugged weighted_wr * n_eff into wilson_ci() as if it
    were an integer count of Bernoulli trials. That formula's derivation
    assumes i.i.d. Bernoulli observations; real-valued weights violate it,
    so the resulting "CI" had no coverage guarantee and was typically too
    narrow — every consumer that gates on it (auto-disable, auto-reinstate,
    rewardable pool) was systematically overconfident.

    Bootstrap: resample rows with replacement n_boot times, recompute the
    weighted statistic on each resample, take the 2.5/97.5 percentiles.
    This is honest about effective sample size because the variance of
    n_eff is baked into the resampled statistic, not injected post-hoc.

    n_boot=400 is the default: enough to stabilize the 2.5th percentile
    estimate for realistic n (200-5000 rows) without adding meaningful
    runtime to a report that already runs a 50-sim Monte Carlo.
    """
    if now_ts is None:
        now_ts = time.time()
    if not rows:
        return None, 0.0, 0.0, 0.0

    def _weighted_once(sample: List[Row]) -> Tuple[float, float]:
        sum_w = sum_w2 = sum_ww = 0.0
        for r in sample:
            rw = recency_weight(r.get("entry_ts"), now_ts, decay_days)
            ww = r.get("win_weight", 1.0 if r["win"] else 0.0)
            sum_w += rw
            sum_w2 += rw * rw
            if r["win"]:
                sum_ww += rw * ww
        if sum_w <= 0:
            return 0.0, 0.0
        wr = min(sum_ww / sum_w, 1.0)
        n_eff_local = (sum_w ** 2) / sum_w2 if sum_w2 > 0 else 0.0
        return wr, n_eff_local

    point_wr, point_n_eff = _weighted_once(rows)

    if len(rows) < 10 or n_boot <= 0:
        # Not enough rows for a meaningful bootstrap — return a conservatively
        # wide [0, 1] interval so downstream CI-gated callers stay closed
        # rather than firing on a spuriously tight band.
        return point_wr, point_n_eff, 0.0, 1.0

    rng = random.Random(seed)
    n = len(rows)
    boots: List[float] = []
    for _ in range(n_boot):
        sample = [rows[rng.randrange(n)] for _ in range(n)]
        wr, _ = _weighted_once(sample)
        boots.append(wr)
    boots.sort()
    lo_idx = max(0, int(0.025 * (len(boots) - 1)))
    hi_idx = min(len(boots) - 1, int(0.975 * (len(boots) - 1)))
    return point_wr, point_n_eff, boots[lo_idx], boots[hi_idx]

# ── Bracket trade model ─────────────────────────────────────────────
# Outcomes are labelled by which level is touched first: the stop
# (OUTCOME_MAE_LOSS_PCT) or the target (stop * OUTCOME_RR_TARGET). P&L must
# describe that SAME trade. pct_move / the stored net_pnl_pct are measured
# at the 12-candle close, which is a different (hold-to-horizon) trade.
_TARGET_REASONS = frozenset({"target_hit", "target_hit_ever"})

_STOP_REASONS = frozenset({"stop_hit", "stop_hit_ever", "ambiguous_same_candle"})

_UNTAGGED_REASONS = frozenset({None, "", "legacy", "unknown"})

def bracket_exit_pct(row: Row) -> Optional[float]:
    """Gross signed exit, in percent, if the trade was closed by its stop or
    target (+target / -stop); None when neither decided it (timeout at the
    horizon, or no usable tag). Same-candle TP+SL ties count as the stop,
    matching how state.py labels them. Rows written before outcome_reason
    existed fall back to their tp_first flag."""
    reason = row.get("outcome_reason")
    tp_first = row.get("tp_first")
    stop_pct = float(cfg.OUTCOME_MAE_LOSS_PCT)
    if reason in _TARGET_REASONS or (reason in _UNTAGGED_REASONS and tp_first is True):
        return stop_pct * float(cfg.OUTCOME_RR_TARGET)
    if reason in _STOP_REASONS or (reason in _UNTAGGED_REASONS and tp_first is False):
        return -stop_pct
    return None

def row_net_pnl_pct(row: Row, total_cost_pct: float) -> float:
    """Signed, cost-adjusted P&L of one resolved trade in percent, under the
    bracket model. Stop/target exits use the bracket distance minus the row's
    own realized_cost_pct (flat total_cost_pct if absent). Trades that hit
    neither level exit at the horizon close, so they keep the stored
    close-based net_pnl_pct, else the legacy magnitude-by-label estimate.
    Single source of truth for EV, Kelly, profit factor, drawdown, training
    weights and the kill switch."""
    exit_pct = bracket_exit_pct(row)
    if exit_pct is not None:
        cost = row.get("realized_cost_pct")
        cost = float(cost) if cost is not None else total_cost_pct
        return exit_pct - cost
    net = row.get("net_pnl_pct")
    if net is not None:
        return float(net)
    mag = abs(float(row.get("pct_move", 0.0)))
    return (mag - total_cost_pct) if row["win"] else -(mag + total_cost_pct)

def favourable_move(row: Row) -> float:
    """Gross move magnitude in percent: the bracket distance when the stop or
    target decided the trade, else the horizon-close move."""
    exit_pct = bracket_exit_pct(row)
    if exit_pct is not None:
        return abs(exit_pct)
    return abs(float(row.get("pct_move", 0.0)))

def expected_value(wins: int, losses: int, avg_win_pct: float, avg_loss_pct: float) -> float:
    """EV per trade in % terms. Positive = profitable long-run."""
    n = wins + losses
    if n == 0:
        return 0.0
    wr = wins / n
    return wr * avg_win_pct - (1 - wr) * abs(avg_loss_pct)

def rr_ratio(avg_win_pct: float, avg_loss_pct: float) -> Optional[float]:
    """Reward:risk ratio. None if there's no meaningful loss magnitude to
    divide by (guards against a near-zero avg_loss producing a nonsense
    ratio)."""
    if not avg_loss_pct or abs(avg_loss_pct) < 1e-6:
        return None
    return avg_win_pct / abs(avg_loss_pct)

def format_rr(rr: Optional[float]) -> str:
    return f"{rr:.2f}" if rr is not None else "n/a"

def ev_and_rr_for(rows: List[Row]) -> Tuple[float, Optional[float], float, float]:
    """EV, R:R, avg win magnitude, avg loss magnitude for a row subset.
    Uses favourable_move() throughout so buy/sell signs never cancel."""
    wins = [favourable_move(r) for r in rows if r["win"]]
    losses = [favourable_move(r) for r in rows if not r["win"]]
    avg_w = sum(wins) / len(wins) if wins else 0.0
    avg_l = sum(losses) / len(losses) if losses else 0.0
    ev = expected_value(len(wins), len(losses), avg_w, avg_l)
    rr = rr_ratio(avg_w, avg_l)
    return ev, rr, avg_w, avg_l

def smooth(values: List[float], window: int = 3) -> List[float]:
    """Centered moving average — used only to de-noise knee-point
    detection, never to alter numbers actually shown to a user."""
    n = len(values)
    if n < window:
        return values[:]
    half = window // 2
    out = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out

def walk_forward_split(
    rows: List[Row],
    train_frac: float = 0.67,
    lookahead_sec: Optional[int] = None,   # None → (fill delay + lookahead + 1) candles * 900
    embargo_sec: int = 900,          # one candle of buffer
) -> Tuple[List[Row], List[Row]]:
    """Chronological split by entry_ts with purge + embargo.

    A row at entry_ts=T has its forward outcome window extend to T+lookahead_sec.
    Without purging, the last lookahead_sec/second worth of train rows leak
    their outcome windows into the first holdout rows — a well-known CV
    leakage bug (López de Prado, "Advances in Financial ML", ch. 7).

    Purge: drop train rows whose outcome window overlaps the holdout.
    Embargo: drop holdout rows within embargo_sec of the cut so residual
    autocorrelation decays before scoring starts.

    Rows missing entry_ts (0) sort first, into the train side (unchanged).
    """
    if lookahead_sec is None:
        lookahead_sec = (
            max(0, int(getattr(cfg, "OUTCOME_FILL_DELAY_CANDLES", 1)))
            + int(cfg.OUTCOME_LOOKAHEAD_CANDLES)
            + 1
        ) * 900

    ordered = sorted(rows, key=lambda r: r.get("entry_ts", 0))
    if not ordered:
        return [], []

    split_idx = int(len(ordered) * train_frac)
    if split_idx <= 0 or split_idx >= len(ordered):
        return ordered[:split_idx], ordered[split_idx:]

    cut_ts = ordered[split_idx].get("entry_ts", 0)

    # Purge train rows whose forward outcome window spills past the cut.
    # The train row's label would otherwise be partially determined by
    # prices the holdout set is about to see.
    train = [
        r for r in ordered[:split_idx]
        if r.get("entry_ts", 0) + lookahead_sec < cut_ts - embargo_sec
    ]
    # Embargo: skip the first embargo_sec of the holdout so the very first
    # scored rows aren't still statistically coupled to the train tail.
    holdout = [
        r for r in ordered[split_idx:]
        if r.get("entry_ts", 0) >= cut_ts + embargo_sec
    ]
    return train, holdout

def confidence_label(n: int, wilson_lo: float, wilson_hi: float) -> str:
    """Translate sample size + Wilson interval width into a plain-language
    confidence label. n alone is misleading — 100 trades with a wide CI is
    weaker evidence than 40 trades with a tight one — so this uses both."""
    width = wilson_hi - wilson_lo
    if n < 20 or width > 0.35:
        return "LOW"
    if n < 50 or width > 0.20:
        return "MEDIUM"
    if n < 150 or width > 0.10:
        return "HIGH"
    return "VERY HIGH"

def sample_evidence_state(
    n: int,
    oos_validated: bool = False,
    insufficient_n: int = 15,
    shadow_n: int = 50,
    eligible_n: int = 100,
) -> str:
    """Explicit sample-aware learning states (roadmap item #24).

    n < insufficient_n          → INSUFFICIENT  (no adjustment)
    insufficient_n ≤ n < shadow → SHADOW        (monitor / simulate only)
    shadow ≤ n < eligible       → ELIGIBLE      (candidate for change)
    n ≥ eligible + OOS pass     → ACTIONABLE
    n ≥ eligible without OOS    → ELIGIBLE
    """
    if n < insufficient_n:
        return "INSUFFICIENT"
    if n < shadow_n:
        return "SHADOW"
    if n < eligible_n:
        return "ELIGIBLE"
    if oos_validated:
        return "ACTIONABLE"
    return "ELIGIBLE"

def find_knee_point(caps_data: List[CapRow], min_sample: int = 30, smooth_window: int = 3) -> Optional[float]:
    """Where marginal WR gain per +1 score flattens. WR values are smoothed
    first to resist single-bucket noise. Returns the (unsmoothed) score at
    the knee, or None if there isn't enough data to detect one reliably."""
    if len(caps_data) < max(6, smooth_window * 2):
        return None
    wrs = [c[2] for c in caps_data]
    smoothed_wrs = smooth(wrs, window=smooth_window)

    best_knee = None
    best_ratio = 0.0
    for i in range(1, len(caps_data) - 1):
        prev_cap = caps_data[i - 1][0]
        curr_cap, curr_n = caps_data[i][0], caps_data[i][1]
        next_cap = caps_data[i + 1][0]
        prev_wr, curr_wr, next_wr = smoothed_wrs[i - 1], smoothed_wrs[i], smoothed_wrs[i + 1]
        marginal_before = (curr_wr - prev_wr) / max(curr_cap - prev_cap, 0.01)
        marginal_after = (next_wr - curr_wr) / max(next_cap - curr_cap, 0.01)
        drop = marginal_before - marginal_after
        if drop > best_ratio and curr_n >= min_sample:
            best_ratio = drop
            best_knee = curr_cap
    return best_knee

def compute_ev_by_cap(rows: List[Row], caps: List[float], min_sample: int = 10):
    """EV (and win/loss magnitudes for R:R) per trade for each cap level.
    Returns list of (cap, n, wr, ev, avg_win_magnitude, avg_loss_magnitude)."""
    ev_data = []
    for cap in caps:
        subset = [r for r in rows if r["score"] >= cap]
        if len(subset) < min_sample:
            continue
        ev, _rr, avg_w, avg_l = ev_and_rr_for(subset)
        wr = sum(r["win"] for r in subset) / len(subset)
        ev_data.append((cap, len(subset), wr, ev, avg_w, avg_l))
    return ev_data

def pain_adjusted_win_rate(rows: List[Row], min_sample: int = 10) -> Dict[str, Dict[str, Any]]:
    stats: DefaultDict[str, Dict[str, Any]] = defaultdict(lambda: {"wins": 0, "n": 0, "maes": []})
    for r in rows:
        s = stats[r["alert_key"]]
        s["wins"] += r["win"]
        s["n"] += 1
        mae = r.get("mae")
        if mae is not None:
            s["maes"].append(mae)
    results: Dict[str, Dict[str, Any]] = {}
    for ak, s in stats.items():
        if s["n"] < min_sample or not s["maes"]:
            continue
        raw_wr = s["wins"] / s["n"]
        mean_mae = statistics.mean(s["maes"])
        results[ak] = {
            "raw_wr": raw_wr, "mean_mae": mean_mae,
            "pawr": raw_wr * (1 - mean_mae),
            "n": s["n"], "mae_sample": len(s["maes"]),
        }
    return results

def detect_temporal_drift(rows: List[Row], window_days: int = 14):
    """Compare win rate of recent outcomes vs older ones. Uses wall-clock
    time (time.time()) as "now" — NOT the last trade's timestamp, which
    would silently shift the "recent" window backward if the bot had any
    downtime. Returns (recent_wr, older_wr, recent_n), or (None, None,
    None) if either side is too thin to compare."""
    if not rows:
        return None, None, None
    now_ts = int(time.time())
    cutoff = now_ts - (window_days * 86400)
    recent = [r for r in rows if r["entry_ts"] >= cutoff]
    older = [r for r in rows if r["entry_ts"] < cutoff]
    if len(recent) < 10 or len(older) < 10:
        return None, None, None
    recent_wr = sum(r["win"] for r in recent) / len(recent)
    older_wr = sum(r["win"] for r in older) / len(older)
    return recent_wr, older_wr, len(recent)

# ══════════════════════════════════════════════════════════════════════
#  NEW: Cost-Aware EV + Kelly Sizing  (Recommended.txt §5)
# ══════════════════════════════════════════════════════════════════════
def ev_and_kelly_for(
    rows: List[Row],
    fee_pct: float = 0.0006,
    slippage_pct: float = 0.0003,
) -> Tuple[float, float, float]:
    """Net EV after round-trip fees + slippage, plus Half-Kelly fraction.
    Returns (net_ev_pct, half_kelly_fraction, win_rate).

    P&L per row comes from row_net_pnl_pct(): the bracket exit (+target /
    -stop) net of the row's own realized cost (actual fees plus measured
    entry slippage when available), so it describes the same trade the
    win/loss label does. Timeouts keep their close-based net_pnl_pct."""
    if not rows:
        return 0.0, 0.0, 0.0
    # entry + exit for both fee and slippage; fee_pct/slippage_pct are
    # fractions (e.g. 0.0006 = 0.06%) so this must be *100 to land in the
    # same percentage-point units as pct_move (price_diff/price * 100).
    total_cost = ((fee_pct * 2) + (slippage_pct * 2)) * 100

    net_moves: List[float] = [row_net_pnl_pct(r, total_cost) for r in rows]

    wins = [m for m in net_moves if m > 0]
    losses = [abs(m) for m in net_moves if m <= 0]
    wr = len(wins) / len(net_moves) if net_moves else 0.0
    avg_win = statistics.mean(wins) if wins else 0.0
    avg_loss = statistics.mean(losses) if losses else 0.0
    ev = statistics.mean(net_moves) if net_moves else 0.0
    b = (avg_win / avg_loss) if avg_loss > 0 else 1.0
    full_kelly = (wr * b - (1 - wr)) / b if b > 0 else 0.0
    half_kelly = max(0.0, min(full_kelly * 0.5, 0.25))  # cap 25 %
    return ev, half_kelly, wr

# ════════════════════════════════════════════════════════════════════���══
#  FIXED: Brier Score & Calibration Curve  (Recommended.txt §2)
# ═══════════════════════════════════════════════════════════════════════
def brier_score_and_calibration(
    rows: List[Row],
    bucket_size: float = 1.0,
    train_frac: float = 0.67,
) -> Tuple[float, List[Dict[str, Any]]]:
    """Brier score (0 = perfect, 0.25 = coin-flip) + calibration curve.
    Uses a train/holdout split to provide a true out-of-sample calibration check."""
    if not rows:
        return 0.5, []

    train_rows, holdout_rows = walk_forward_split(rows, train_frac)
    use_oos = len(train_rows) >= 20 and len(holdout_rows) >= 20

    if use_oos:
        # Build train buckets for predicted probabilities
        train_buckets: Dict[float, Dict[str, int]] = {}
        for r in train_rows:
            b = int(r["score"] // bucket_size) * bucket_size
            train_buckets.setdefault(b, {"wins": 0, "n": 0})
            train_buckets[b]["wins"] += int(r["win"])
            train_buckets[b]["n"] += 1

        # Build holdout buckets for observed probabilities
        holdout_buckets: Dict[float, Dict[str, int]] = {}
        for r in holdout_rows:
            b = int(r["score"] // bucket_size) * bucket_size
            holdout_buckets.setdefault(b, {"wins": 0, "n": 0})
            holdout_buckets[b]["wins"] += int(r["win"])
            holdout_buckets[b]["n"] += 1

        curve: List[Dict[str, Any]] = []
        total_brier = 0.0
        count = 0

        # Only report buckets that exist in BOTH train and holdout
        all_buckets = set(train_buckets.keys()).intersection(set(holdout_buckets.keys()))

        for b in sorted(all_buckets):
            t_d = train_buckets[b]
            h_d = holdout_buckets[b]

            if t_d["n"] < 5 or h_d["n"] < 5:
                continue

            predicted_p = t_d["wins"] / t_d["n"]
            observed_p = h_d["wins"] / h_d["n"]

            curve.append({
                "score_floor": b,
                "predicted_p": predicted_p,
                "observed_p": observed_p,
                "n": h_d["n"],  # n represents the holdout samples evaluated
                "is_oos": True,
            })

            total_brier += h_d["wins"] * ((predicted_p - 1.0) ** 2)
            total_brier += (h_d["n"] - h_d["wins"]) * (predicted_p ** 2)
            count += h_d["n"]

        brier = (total_brier / count) if count > 0 else 0.5
        return brier, curve

    else:
        # Fallback: in-sample if data too thin
        buckets: Dict[float, Dict[str, int]] = {}
        for r in rows:
            b = int(r["score"] // bucket_size) * bucket_size
            buckets.setdefault(b, {"wins": 0, "n": 0})
            buckets[b]["wins"] += int(r["win"])
            buckets[b]["n"] += 1
        curve = []
        total_brier = 0.0
        count = 0
        for b in sorted(buckets.keys()):
            d = buckets[b]
            if d["n"] < 5:
                continue
            p = d["wins"] / d["n"]
            curve.append({
                "score_floor": b,
                "predicted_p": p,
                "observed_p": p,
                "n": d["n"],
                "is_oos": False,  # Flagged so calibration_alert ignores it
            })
            total_brier += d["wins"] * ((p - 1.0) ** 2)
            total_brier += (d["n"] - d["wins"]) * (p ** 2)
            count += d["n"]

        brier = (total_brier / count) if count > 0 else 0.5
        return brier, curve

def calibration_alert(
    rows: List[Row],
    bucket_size: float = 1.0,
    max_divergence: float = 0.10,
) -> List[Dict[str, Any]]:
    """Return buckets where predicted vs observed WR diverges > max_divergence."""
    _brier, curve = brier_score_and_calibration(rows, bucket_size)
    alerts: List[Dict[str, Any]] = []
    for c in curve:
        if c["n"] < 10:
            continue
        # Skip in-sample buckets — only flag true OOS miscalibration
        if not c.get("is_oos", True):
            continue
        lo, hi, _ = wilson_ci(
            int(round(c["observed_p"] * c["n"])), c["n"]
        )
        if abs(c["predicted_p"] - c["observed_p"]) > max_divergence:
            alerts.append({
                "score_floor": c["score_floor"],
                "predicted": c["predicted_p"],
                "observed": c["observed_p"],
                "n": c["n"],
                "wilson_lo": lo,
                "wilson_hi": hi,
            })
    return alerts

def _percentile(data: List[float], p: float) -> float:
    """Linear-interpolation percentile (numpy-compatible)."""
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * p / 100.0
    f = int(math.floor(k))
    c = int(math.ceil(k))
    if f == c:
        return s[f]
    return s[f] * (c - k) + s[c] * (k - f)

def bootstrap_ev_ci(
    rows: List[Row],
    n_sims: int = 1000,
    block_size: int = 20,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Block-bootstrap EV distribution. Returns mean / p5 / p95 EV."""
    if len(rows) < block_size * 3:
        return {"valid": False, "error": "insufficient_data"}

    rng = random.Random(seed)
    ordered = sorted(rows, key=lambda r: r.get("entry_ts", 0))
    blocks = [
        ordered[i:i + block_size]
        for i in range(0, len(ordered), block_size)
    ]
    blocks = [b for b in blocks if b]
    if len(blocks) < 5:
        return {"valid": False, "error": "insufficient_blocks"}

    ev_samples: List[float] = []
    for _ in range(n_sims):
        sampled = rng.choices(blocks, k=len(blocks))
        flat = [r for blk in sampled for r in blk]
        ev, _hk, _wr = ev_and_kelly_for(flat)
        ev_samples.append(ev)

    ev_samples.sort()
    n = len(ev_samples)
    p5_idx = max(0, min(n - 1, round(0.05 * (n - 1))))
    p95_idx = max(0, min(n - 1, round(0.95 * (n - 1))))
    return {
        "valid": True,
        "n_simulations": n,
        "ev_mean": statistics.fmean(ev_samples),
        "ev_p5": ev_samples[p5_idx],
        "ev_p95": ev_samples[p95_idx],
        "ev_std": statistics.pstdev(ev_samples) if n > 1 else 0.0,
    }

def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)

def _prob_edge_broken(wins: int, n: int, target_wr: float,
                       prior_strength: float = 10.0) -> float:
    """Bayesian P(true_wr < target_wr) under a Beta posterior.

    Replaces point-estimate triggers like `wr < disable_wr` with a
    continuous posterior probability. A trigger of P > 0.90 is roughly
    equivalent to the old fixed threshold on large samples, but
    automatically downweights small ones (they shrink toward 0.5 instead
    of firing on 3 noisy rows)."""
    if n <= 0:
        return 0.5
    a0 = prior_strength * target_wr
    b0 = prior_strength * (1.0 - target_wr)
    a = a0 + wins
    b = b0 + (n - wins)
    mean = a / (a + b)
    var = (a * b) / ((a + b) ** 2 * (a + b + 1))
    if var <= 0:
        return 0.5
    z = (target_wr - mean) / math.sqrt(var)
    return 0.5 * math.erfc(-z / math.sqrt(2.0)) 

def _prob_ev_negative(rows: List[Row], n_sims: int = 400) -> float:
    """P(true EV <= 0) via block-bootstrap.
    
    WARNING: Returns 0.5 (neutral) when len(rows) < 60 because
    bootstrap_ev_ci requires block_size * 3 = 60 rows minimum.
    Callers with fewer rows should fall back to the point estimate
    (net_ev <= 0) instead of trusting this return value.
    """
    bs = bootstrap_ev_ci(rows, n_sims=n_sims)
    if not bs.get("valid"):
        return 0.5
    ev_mean = bs["ev_mean"]
    ev_std = bs["ev_std"]
    if ev_std <= 0:
        return 0.5
    z = (0.0 - ev_mean) / ev_std
    return 0.5 * math.erfc(-z / math.sqrt(2.0))

def _prob_ev_positive(ev_mean: float, ev_std: float) -> float:
    if ev_std <= 0:
        return 0.5
    z = ev_mean / ev_std
    return 0.5 * math.erfc(-z / math.sqrt(2.0))

def ev_first_objective(
    rows: List[Row],
    min_sample: int = 20,
    fee_pct: float = 0.0006,
    slippage_pct: float = 0.0003,
) -> Dict[str, Any]:
    """Unified profitability assessment. Replaces WR-first gating.

    Returns the full evidence stack:
    - net_ev, p_ev_positive, ev_p5 (5th percentile)
    - profit_factor, max_drawdown_pct
    - wr (demoted to informational)
    - n, confidence label

    Memoised per report cycle — see _EV_FIRST_CACHE. Returns a fresh
    dict on every call so callers cannot mutate the cached value.
    """
    if len(rows) < min_sample:
        return {"valid": False, "error": "insufficient_data", "n": len(rows)}

    _cache_key = _ev_first_cache_key(rows, min_sample, fee_pct, slippage_pct)
    _cached = _EV_FIRST_CACHE.get(_cache_key)
    if _cached is not None:
        return dict(_cached)

    net_ev, half_kelly, wr = ev_and_kelly_for(rows, fee_pct, slippage_pct)

    # Bootstrap EV distribution for uncertainty
    bs = bootstrap_ev_ci(rows, n_sims=500, seed=42)
    ev_p5 = bs.get("ev_p5", net_ev) if bs.get("valid") else net_ev
    ev_std = bs.get("ev_std", 0.0) if bs.get("valid") else 0.0

    p_ev_positive = _prob_ev_positive(net_ev, ev_std)

    # ── Net per-row P&L (bracket model, via row_net_pnl_pct — identical to
    # ev_and_kelly_for, so profit_factor/max_drawdown are net of costs and
    # consistent with net_ev above) ──
    total_cost = (fee_pct * 2 + slippage_pct * 2) * 100
    net_pnls: List[float] = [row_net_pnl_pct(r, total_cost) for r in rows]

    # Profit factor (net of costs)
    wins = [p for p in net_pnls if p > 0]
    losses = [abs(p) for p in net_pnls if p <= 0]
    gross_profit = sum(wins) if wins else 0.0
    gross_loss = sum(losses) if losses else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Max drawdown (cumulative net PnL trough)

    ordered_idx = sorted(range(len(rows)), key=lambda i: rows[i].get("entry_ts", 0))
    equity = 1.0
    peak_equity = 1.0
    max_dd = 0.0
    for i in ordered_idx:
        equity *= (1.0 + net_pnls[i] / 100.0)
        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / peak_equity
        if dd > max_dd:
            max_dd = dd
    max_dd *= 100.0

    n = len(rows)
    win_count = sum(1 for r in rows if r["win"])
    lo, hi, _ = wilson_ci(win_count, n)

    _result = {
        "valid": True,
        "n": n,
        "wr": wr,
        "wilson_lo": lo,
        "wilson_hi": hi,
        "net_ev": round(net_ev, 4),
        "p_ev_positive": round(p_ev_positive, 4),
        "ev_p5": round(ev_p5, 4),
        "ev_std": round(ev_std, 4),
        "half_kelly": round(half_kelly, 4),
        "profit_factor": round(profit_factor, 3),
        "max_drawdown_pct": round(max_dd, 3),
        "confidence": confidence_label(n, lo, hi),
    }
    _EV_FIRST_CACHE[_cache_key] = _result
    return dict(_result)

def per_trade_ev(
    calibrated_p: float,
    reward_r: float,          # e.g. R:R = 2.0 → reward_r = 2.0
    risk_r: float = 1.0,      # stop-loss in R units (usually 1.0)
    fee_pct: float = 0.0006,
    slippage_pct: float = 0.0003,
    sl_pct: Optional[float] = None,  # if known, convert costs into R
) -> Dict[str, Any]:
    """Per-trade EV from a calibrated probability (future pipeline).

    EV = p * reward_r − (1−p) * risk_r − costs_in_R
    This is NOT the historical alert-key bucket EV used by
    trade_quality_score today.
    """
    if calibrated_p is None or not (0.0 <= calibrated_p <= 1.0):
        return {"valid": False, "error": "bad_p"}
    # Round-trip cost, matching ev_and_kelly_for / ev_first_objective.
    # fee_pct/slippage_pct are fractions (0.0006 = 0.06%), so *100 puts
    # the value in percentage points.
    costs_pct = ((fee_pct * 2) + (slippage_pct * 2)) * 100.0
    if sl_pct and sl_pct > 0:
        # sl_pct arrives as a fraction (0.005 = 0.5%). Convert to percent
        # so the units match costs_pct, then divide to express cost as a
        # fraction of the stop distance (R units).
        costs_r = costs_pct / (sl_pct * 100.0)
    else:
        # Fallback: default stop from live config. Using OUTCOME_MAE_LOSS_PCT
        # keeps the fallback consistent with how outcome resolution grades
        # the trade.
        _default_stop_pct = float(getattr(cfg, "OUTCOME_MAE_LOSS_PCT", 0.5))
        costs_r = costs_pct / _default_stop_pct
    net_ev = (
        calibrated_p * reward_r
        - (1.0 - calibrated_p) * risk_r
        - costs_r
    )
    return {
        "valid": True,
        "net_ev": round(net_ev, 4),
        "calibrated_p": round(calibrated_p, 4),
        "reward_r": reward_r,
        "risk_r": risk_r,
        "costs_r": round(costs_r, 4),
    }

def _flatten_row_features(row: Row) -> Dict[str, float]:
    """Flatten a row's votes + numeric context into a flat feature dict.
    Vote booleans become 0/1 under 'vote:<name>'; numeric context values
    under 'ctx:<key>'. Used by the decision-stump root-cause scanner and
    the PSI drift detector."""
    feats: Dict[str, float] = {}
    votes = row.get("votes") or {}
    if isinstance(votes, dict):
        for vn, v in votes.items():
            feats[f"vote:{vn}"] = 1.0 if v else 0.0
    ctx = row.get("context") or {}
    if isinstance(ctx, dict):
        for k, v in ctx.items():
            if isinstance(v, bool):
                feats[f"ctx:{k}"] = 1.0 if v else 0.0
            elif isinstance(v, (int, float)):
                feats[f"ctx:{k}"] = float(v)
    return feats

def population_stability_index(
    baseline: List[float],
    recent: List[float],
    bins: int = 10,
    eps: float = 1e-4,
) -> Optional[float]:
    """PSI between two samples. <0.1 stable, 0.1-0.25 moderate shift,
    >0.25 large shift. The classic 'why did my model stop working' metric."""
    if len(baseline) < 30 or len(recent) < 30:
        return None
    lo = min(min(baseline), min(recent))
    hi = max(max(baseline), max(recent))
    if hi - lo < 1e-12:
        return 0.0

    def _bucket_pcts(vals: List[float]) -> List[float]:
        counts = [0] * bins
        for v in vals:
            idx = min(bins - 1, max(0, int((v - lo) / (hi - lo) * bins)))
            counts[idx] += 1
        n = len(vals)
        return [max(c / n, eps) for c in counts]

    b = _bucket_pcts(baseline)
    r = _bucket_pcts(recent)
    return sum((rb - bb) * math.log(rb / bb) for bb, rb in zip(b, r))
