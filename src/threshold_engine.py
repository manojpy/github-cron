#!/usr/bin/env python3
"""
threshold_engine.py — shared, pure-function analysis library for confluence
threshold recommendations.

"""
from __future__ import annotations
import hashlib
import json 
import math
import time
import random
import statistics
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple
from bot_config import cfg, format_ist_time

Row = Dict[str, Any]
CapRow = Tuple[float, int, float, float]  # (cap, n, wr, wilson_lower_bound)

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

def favourable_move(row: Row) -> float:
    """pct_move is signed by price direction, not by trade outcome — a
    winning sell has a negative pct_move. Always take the magnitude of the
    move that was favourable to the position, or wins/losses from opposite
    directions cancel toward zero when averaged."""
    return abs(row.get("pct_move", 0.0))

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

def build_buckets(rows: List[Row], bucket_size: float = 1.0) -> Dict[float, Dict[str, int]]:
    buckets: Dict[float, Dict[str, int]] = defaultdict(lambda: {"wins": 0, "n": 0})
    for row in rows:
        b = int(row["score"] // bucket_size) * bucket_size
        buckets[b]["n"] += 1
        buckets[b]["wins"] += row["win"]
    return buckets

def detect_toxic_zones(
    buckets: Dict[float, Dict[str, int]], bucket_size: float = 1.0, min_sample: int = 10,
) -> List[Tuple[float, float, float, int]]:
    """Score buckets where even the Wilson UPPER bound is below 50% —
    informational. Callers should not use this as a hard floor on a
    recommendation: a toxic bucket sandwiched between two good ones just
    means that slice is worth investigating, not that everything below it
    is unsafe (the cumulative cap stats already price it in)."""
    toxic = []
    for b in sorted(buckets):
        d = buckets[b]
        if d["n"] < min_sample:
            continue
        wr = d["wins"] / d["n"]
        lo, hi, _ = wilson_ci(d["wins"], d["n"])
        if hi < 0.50:
            toxic.append((b, b + bucket_size, wr, d["n"]))
    return toxic

def detect_anomalous_buckets(
    buckets: Dict[float, Dict[str, int]], bucket_size: float = 1.0,
    min_sample: int = 10, drop_pct: float = 0.15,
) -> List[Tuple[float, float, float, float, float, int]]:
    """A bucket whose WR sits well below BOTH immediate neighbors — a dip
    surrounded by strength, worth investigating rather than silently
    trusted (e.g. one bad alert-type polluting a single score band)."""
    sorted_b = sorted(buckets.keys())
    anomalies = []
    for idx, b in enumerate(sorted_b):
        if idx == 0 or idx == len(sorted_b) - 1:
            continue
        prev_b, next_b = sorted_b[idx - 1], sorted_b[idx + 1]
        if abs(prev_b + bucket_size - b) > 1e-9 or abs(b + bucket_size - next_b) > 1e-9:
            continue  # neighbors aren't actually adjacent (empty bins between)
        d, dp, dn = buckets[b], buckets[prev_b], buckets[next_b]
        if d["n"] < min_sample or dp["n"] < min_sample or dn["n"] < min_sample:
            continue
        wr, wr_prev, wr_next = d["wins"] / d["n"], dp["wins"] / dp["n"], dn["wins"] / dn["n"]
        if wr_prev - wr >= drop_pct and wr_next - wr >= drop_pct:
            anomalies.append((b, b + bucket_size, wr, wr_prev, wr_next, d["n"]))
    return anomalies

def build_caps_data(rows: List[Row], min_sample: int = 20) -> Tuple[List[float], List[CapRow]]:
    # Sort descending by score so we can accumulate suffix stats in one pass
    sorted_rows = sorted(rows, key=lambda r: r["score"], reverse=True)
    candidate_caps = sorted(set(r["score"] for r in rows))  # ascending for output
    caps_data: List[CapRow] = []
    
    cumulative_wins = 0
    cumulative_n = 0
    row_idx = 0
    total_rows = len(sorted_rows)
    
    # Walk caps from highest to lowest, accumulating rows that qualify
    for cap in reversed(candidate_caps):
        while row_idx < total_rows and sorted_rows[row_idx]["score"] >= cap:
            cumulative_wins += int(sorted_rows[row_idx]["win"])
            cumulative_n += 1
            row_idx += 1
        
        if cumulative_n >= min_sample:
            wr = cumulative_wins / cumulative_n
            lo, _hi, _p = wilson_ci(cumulative_wins, cumulative_n)
            caps_data.append((cap, cumulative_n, wr, lo))
    
    caps_data.reverse()  # back to ascending order
    return candidate_caps, caps_data
 
def walk_forward_split(
    rows: List[Row],
    train_frac: float = 0.67,
    lookahead_sec: int = 8 * 900,   # OUTCOME_LOOKAHEAD_CANDLES * 900
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

def validate_threshold_walk_forward(
    rows: List[Row],
    target_winrate: float = 0.55,
    min_sample: int = 20,
    bucket_size: float = 1.0,
    train_frac: float = 0.67,
    slack: float = 0.05,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"valid": False}
    train_rows, holdout_rows = walk_forward_split(rows, train_frac)
    result["train_n"] = len(train_rows)
    result["holdout_n"] = len(holdout_rows)

    if len(train_rows) < min_sample * 2:
        result["error"] = "insufficient_train"
        return result
    if len(holdout_rows) < min_sample:
        result["error"] = "insufficient_holdout"
        return result

    train_result = recommend_threshold(train_rows, target_winrate, min_sample, bucket_size)
    result["train_result"] = train_result
    if not train_result["valid"]:
        result["error"] = train_result.get("error", "train_invalid")
        return result

    recommended = train_result["recommended"]
    result["recommended"] = recommended
    result["valid"] = True

    holdout_subset = [r for r in holdout_rows if r["score"] >= recommended]
    n_ho = len(holdout_subset)
    result["holdout_n_at_threshold"] = n_ho
    if n_ho < 5:
        result["passed"] = None
        result["error"] = "holdout_too_thin_at_threshold"
        return result

    wins_ho = sum(r["win"] for r in holdout_subset)
    wr_ho = wins_ho / n_ho
    lo, hi, _ = wilson_ci(wins_ho, n_ho)
    result.update({
        "holdout_wr": wr_ho, "holdout_wilson_lo": lo, "holdout_wilson_hi": hi,
        "degraded_pct": train_result["rec_wr"] - wr_ho,
        "passed": lo >= (target_winrate - slack),
    })
    return result

def monte_carlo_walk_forward(
    rows: List[Row],
    n_simulations: int = 100,
    train_frac: float = 0.7,
    min_sample: int = 20,
    target_winrate: float = 0.55,
    bucket_size: float = 1.0,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Offline-only robustness check — never touches live gating. Runs many
    walk-forward validations against block-bootstrap resamples of the same
    history, instead of trusting one chronological split. A single split can
    look good or bad by luck of exactly where the cut falls; resampling
    contiguous blocks (not individual rows, which would destroy the
    within-block time-correlation daily/session patterns actually have) and
    re-running the split many times shows whether that result was typical
    or a fluke of one particular window.

    Returns a distribution of out-of-sample win rates across simulations,
    not a single number — report the mean AND the spread (oos_wr_p5 is the
    one that matters most: "how bad could this plausibly get").
    """
    rng = random.Random(seed)
    if len(rows) < min_sample * 2:
        return {"valid": False, "error": "insufficient_data", "n_rows": len(rows)}

    ordered = sorted(rows, key=lambda r: r.get("entry_ts", 0))
    block_size = max(10, len(ordered) // 20)
    blocks = [ordered[i:i + block_size] for i in range(0, len(ordered), block_size)]
    blocks = [b for b in blocks if b]
    if len(blocks) < 5:
        return {"valid": False, "error": "insufficient_blocks", "n_blocks": len(blocks)}

    oos_wr_list: List[float] = []
    threshold_list: List[float] = []

    for _ in range(n_simulations):
        sampled_blocks = rng.choices(blocks, k=len(blocks))
        sampled_rows = [row for block in sampled_blocks for row in block]
        sampled_rows.sort(key=lambda r: r.get("entry_ts", 0))

        train_rows, holdout_rows = walk_forward_split(sampled_rows, train_frac)
        if len(train_rows) < min_sample * 2 or len(holdout_rows) < min_sample:
            continue

        train_result = recommend_threshold(train_rows, target_winrate, min_sample, bucket_size)
        if not train_result["valid"]:
            continue

        threshold = train_result["recommended"]
        holdout_subset = [r for r in holdout_rows if r["score"] >= threshold]
        if len(holdout_subset) < 5:
            continue

        wins = sum(r["win"] for r in holdout_subset)
        oos_wr_list.append(wins / len(holdout_subset))
        threshold_list.append(threshold)

    if len(oos_wr_list) < 10:
        return {
            "valid": False, "error": "too_few_valid_simulations",
            "n_valid": len(oos_wr_list), "n_requested": n_simulations,
        }

    oos_wr_list.sort()
    n = len(oos_wr_list)
    mean_wr = statistics.fmean(oos_wr_list)
    std_wr = statistics.pstdev(oos_wr_list) if n > 1 else 0.0
    p5_idx = max(0, min(n - 1, round(0.05 * (n - 1))))
    p95_idx = max(0, min(n - 1, round(0.95 * (n - 1))))

    return {
        "valid": True,
        "n_simulations": n,
        "n_requested": n_simulations,
        "oos_wr_mean": mean_wr,
        "oos_wr_std": std_wr,
        "oos_wr_p5": oos_wr_list[p5_idx],
        "oos_wr_p95": oos_wr_list[p95_idx],
        "threshold_mean": statistics.fmean(threshold_list),
        "threshold_std": statistics.pstdev(threshold_list) if len(threshold_list) > 1 else 0.0,
        # Mean/spread ratio — a rough "is this edge stable or just lucky
        # sometimes" signal. Not a statistical test, just a sort key for
        # the report; read oos_wr_p5 for the actual worst-case number.
        "robustness_score": mean_wr / max(std_wr, 0.01),
    }

def flag_anomalous_rows(
    rows: List[Row],
    mad_threshold: float = 6.0,
    min_sample: int = 30,
) -> Dict[str, Any]:
    valid_moves = [r for r in rows if r.get("pct_move") is not None]
    if len(valid_moves) < min_sample:
        return {"valid": False, "error": "insufficient_data", "n": len(valid_moves)}

    moves = sorted(r["pct_move"] for r in valid_moves)
    n = len(moves)
    median = moves[n // 2] if n % 2 else (moves[n // 2 - 1] + moves[n // 2]) / 2.0
    abs_devs = sorted(abs(m - median) for m in moves)
    mad = abs_devs[n // 2] if n % 2 else (abs_devs[n // 2 - 1] + abs_devs[n // 2]) / 2.0
    # 1.4826 is the standard consistency constant that scales MAD to
    # approximate a stdev under a normal distribution, so mad_threshold
    # reads on roughly the same scale as an ordinary z-score.
    scaled_mad = mad * 1.4826

    flagged = []
    if scaled_mad > 0:
        for r in valid_moves:
            z = abs(r["pct_move"] - median) / scaled_mad
            if z > mad_threshold:
                flagged.append({
                    "pair": r.get("pair"), "alert_key": r.get("alert_key"),
                    "entry_ts": r.get("entry_ts"), "pct_move": r["pct_move"],
                    "robust_z": z,
                })
    flagged.sort(key=lambda f: -f["robust_z"])

    return {
        "valid": True, "n_total": len(valid_moves), "n_flagged": len(flagged),
        "median_pct_move": median, "scaled_mad": scaled_mad,
        "flagged": flagged,
    }

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

def regime_breakdown(rows: List[Row], min_sample: int = 20) -> Dict[str, Any]:
    """Rule-based regime split — no clustering, no ML. Splits rows into
    'trending' vs 'ranging' at the MEDIAN adx_val actually present in this
    window (a self-relative quantile split, not a hardcoded ADX threshold
    like 25 — 'trending' should mean relatively trending for THIS data,
    since typical ADX levels vary by pair and period).

    Purely diagnostic — this never changes live gating on its own. It only
    tells you whether your edge holds up the same way in both regimes, or
    is concentrated in one of them, which is a prerequisite for ever
    trusting a regime-specific threshold multiplier.

    Returns {"valid": False, "error": ...} if fewer than 2*min_sample rows
    carry an adx_val (older outcome-log rows won't — the field was added
    later, and this degrades gracefully rather than erroring on mixed old/
    new data)."""
    with_regime = [r for r in rows if r.get("adx_val") is not None]
    result: Dict[str, Any] = {"valid": False, "n_with_adx": len(with_regime), "n_total": len(rows)}
    if len(with_regime) < min_sample * 2:
        result["error"] = "insufficient_adx_tagged_rows"
        return result

    adx_values = sorted(r["adx_val"] for r in with_regime)
    mid = len(adx_values) // 2
    median_adx = (
        adx_values[mid] if len(adx_values) % 2
        else (adx_values[mid - 1] + adx_values[mid]) / 2.0
    )
    result["median_adx"] = median_adx

    buckets: Dict[str, List[Row]] = {"trending": [], "ranging": []}
    for r in with_regime:
        buckets["trending" if r["adx_val"] >= median_adx else "ranging"].append(r)

    regimes: Dict[str, Any] = {}
    for label, bucket_rows in buckets.items():
        bn = len(bucket_rows)
        if bn < min_sample:
            regimes[label] = {"valid": False, "n": bn, "error": "insufficient_sample"}
            continue
        wins = sum(r["win"] for r in bucket_rows)
        wr = wins / bn
        lo, hi, _ = wilson_ci(wins, bn)
        regimes[label] = {
            "valid": True, "n": bn, "wr": wr,
            "wilson_lo": lo, "wilson_hi": hi,
            "confidence": confidence_label(bn, lo, hi),
        }
    result["regimes"] = regimes
    result["valid"] = True

    if regimes.get("trending", {}).get("valid") and regimes.get("ranging", {}).get("valid"):
        result["wr_gap"] = regimes["trending"]["wr"] - regimes["ranging"]["wr"]
    return result

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

def per_pair_thresholds(
    rows: List[Row],
    target_winrate: float = 0.55,
    min_sample: int = 30,
) -> Dict[str, Dict[str, Any]]:
    """Per-pair analogue of recommend_threshold(): groups rows by pair and
    independently runs the same knee/EV/target-floor logic for each pair
    that clears min_sample. Returns {pair: recommend_threshold()-result}
    — only for pairs where a valid (result["valid"] is True) recommendation
    was found. Callers should still walk-forward-validate and stability-gate
    each pair's result before applying it, same as the global recommendation."""
    by_pair: Dict[str, List[Row]] = defaultdict(list)
    for r in rows:
        by_pair[r["pair"]].append(r)
    results: Dict[str, Dict[str, Any]] = {}
    for pair, pair_rows in by_pair.items():
        if len(pair_rows) < min_sample:
            continue
        rec = recommend_threshold(pair_rows, target_winrate=target_winrate, min_sample=min_sample)
        if rec.get("valid"):
            results[pair] = rec
    return results

def per_pair_breakdown(rows: List[Row], min_sample: int = 10):
    stats = defaultdict(lambda: {"wins": 0, "n": 0})
    for r in rows:
        s = stats[r["pair"]]
        s["wins"] += r["win"]
        s["n"] += 1
    results = []
    for pair, s in stats.items():
        if s["n"] < min_sample:
            continue
        results.append((pair, s["wins"] / s["n"], s["n"]))
    results.sort(key=lambda x: x[1])
    return results

def per_pair_session_breakdown(rows: List[Row], min_sample: int = 10):
    """Groups by (pair, session) — e.g. reveals a pair performing well in
    Asian hours but randomly in the Dead Zone. Returns a list of
    (pair, session, win_rate, n) tuples, sorted worst win-rate first."""
    stats = defaultdict(lambda: {"wins": 0, "n": 0})
    for r in rows:
        s = stats[(r["pair"], r.get("session", "unknown"))]
        s["wins"] += r["win"]
        s["n"] += 1
    results = []
    for (pair, session), s in stats.items():
        if s["n"] < min_sample:
            continue
        results.append((pair, session, s["wins"] / s["n"], s["n"]))
    results.sort(key=lambda x: x[2])
    return results

def session_breakdown(rows: List[Row], min_sample: int = 10):
    """Groups by session ONLY (asian/london/ny/dead) — for comparing overall
    session performance, as opposed to per_pair_session_breakdown()'s
    (pair, session) granularity. Returns (session, win_rate, n) tuples,
    sorted worst win-rate first."""
    stats = defaultdict(lambda: {"wins": 0, "n": 0})
    for r in rows:
        s = stats[r.get("session", "unknown")]
        s["wins"] += r["win"]
        s["n"] += 1
    results = []
    for session, s in stats.items():
        if s["n"] < min_sample:
            continue
        results.append((session, s["wins"] / s["n"], s["n"]))
    results.sort(key=lambda x: x[1])
    return results

def per_alert_breakdown(rows: List[Row], min_sample: int = 10):
    stats = defaultdict(lambda: {"wins": 0, "n": 0, "scores": []})
    for r in rows:
        s = stats[r["alert_key"]]
        s["wins"] += r["win"]
        s["n"] += 1
        s["scores"].append(r["score"])
    results = []
    for ak, s in stats.items():
        if s["n"] < min_sample:
            continue
        wr = s["wins"] / s["n"]
        avg_score = sum(s["scores"]) / len(s["scores"])
        results.append((ak, wr, s["n"], avg_score))
    results.sort(key=lambda x: x[1])
    return results

def pain_adjusted_win_rate(rows: List[Row], min_sample: int = 10) -> Dict[str, Dict[str, Any]]:
    stats = defaultdict(lambda: {"wins": 0, "n": 0, "maes": []})
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

def outcome_attribution(
    rows: List[Row],
    weights: Dict[str, float],
    threshold: float,
    min_sample: int = 10,
) -> List[Dict[str, Any]]:
 
    vote_names = set()
    for r in rows:
        if r.get("votes"):
            vote_names.update(r["votes"].keys())

    results = []
    for vn in sorted(vote_names):
        weight = weights.get(vn)
        if weight is None or weight <= 0:
            continue
        with_vote = [r for r in rows if r.get("votes") and r["votes"].get(vn) is True]
        if len(with_vote) < min_sample:
            continue

        rescued = [r for r in with_vote if threshold <= r["score"] < threshold + weight]
        comfortable = [r for r in with_vote if r["score"] >= threshold + weight]

        entry: Dict[str, Any] = {
            "vote": vn, "weight": weight,
            "n_with_vote": len(with_vote),
            "n_rescued": len(rescued),
            "rescued_pct": len(rescued) / len(with_vote) if with_vote else 0.0,
        }
        if len(rescued) >= min_sample:
            wins = sum(r["win"] for r in rescued)
            wr = wins / len(rescued)
            lo, hi, _ = wilson_ci(wins, len(rescued))
            entry.update({
                "rescued_valid": True, "rescued_wr": wr,
                "rescued_wilson_lo": lo, "rescued_wilson_hi": hi,
                "rescued_confidence": confidence_label(len(rescued), lo, hi),
            })
        else:
            entry["rescued_valid"] = False
        if len(comfortable) >= min_sample:
            entry["comfortable_wr"] = sum(r["win"] for r in comfortable) / len(comfortable)
            entry["comfortable_n"] = len(comfortable)
        results.append(entry)

    results.sort(key=lambda e: -e["n_rescued"])
    return results

def vote_importance(rows: List[Row], min_sample: int = 10):
    """For each vote, win rate when True vs False. Sorted by lift
    (wr_with - wr_without), descending — tells you which votes add edge
    vs which are just noise."""
    vote_names = set()
    for r in rows:
        if r.get("votes"):
            vote_names.update(r["votes"].keys())
    results = []
    for vn in sorted(vote_names):
        with_vote = [r for r in rows if r.get("votes") and r["votes"].get(vn) is True]
        without_vote = [r for r in rows if r.get("votes") and r["votes"].get(vn) is False]
        if len(with_vote) < min_sample or len(without_vote) < min_sample:
            continue
        wr_with = sum(r["win"] for r in with_vote) / len(with_vote)
        wr_without = sum(r["win"] for r in without_vote) / len(without_vote)
        results.append((vn, wr_with, len(with_vote), wr_without, len(without_vote), wr_with - wr_without))
    results.sort(key=lambda x: -x[5])
    return results

def vote_combo_breakdown(rows: List[Row], lo: float, hi: float, min_sample: int = 20):
    """Within a score band, win rate by which votes actually fired
    together. Returns (band_rows, combo_stats) or None if no vote data in
    that band."""
    band_rows = [row for row in rows if lo <= row["score"] < hi and row.get("votes") is not None]
    if not band_rows:
        return None
    combo_stats: Dict[Tuple[str, ...], Dict[str, int]] = defaultdict(lambda: {"wins": 0, "n": 0})
    for row in band_rows:
        combo = tuple(sorted(k for k, v in row["votes"].items() if v))
        combo_stats[combo]["n"] += 1
        combo_stats[combo]["wins"] += row["win"]
    return band_rows, combo_stats

def direction_split(rows: List[Row]) -> Tuple[Optional[float], int, Optional[float], int]:
    """Returns (buy_wr, buy_n, sell_wr, sell_n)."""
    buys = [r for r in rows if r["direction"] == "buy"]
    sells = [r for r in rows if r["direction"] == "sell"]
    buy_wr = sum(r["win"] for r in buys) / len(buys) if buys else None
    sell_wr = sum(r["win"] for r in sells) / len(sells) if sells else None
    return buy_wr, len(buys), sell_wr, len(sells)

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

def recommend_threshold(
    rows: List[Row],
    target_winrate: float = 0.55,
    min_sample: int = 20,
    bucket_size: float = 1.0,
) -> Dict[str, Any]:
    """
    Returns a dict. ALWAYS check result["valid"] before trusting
    result["recommended"] — an invalid result still returns a populated
    dict (so callers can report *why* it's invalid: see result["error"]),
    but result["recommended"] is not present at all unless valid is True.

    On success (valid=True), also includes:
      recommended, rec_n, rec_wr, rec_ev, rec_rr, rec_avg_win, rec_avg_loss,
      rec_wilson_lo, rec_wilson_hi, confidence,
      buy_wr, buy_n, sell_wr, sell_n,
      drift_recent_wr, drift_older_wr, drift_recent_n,
      alerts_per_week_before, alerts_per_week_after, dropped, dropped_pct,
      toxic_ceiling, overlapping_toxic,
      knee, best_ev, target_floor, caps_data, candidate_caps,
      toxic_zones, anomalies, buckets, overall_wr, n
    """
    result: Dict[str, Any] = {"valid": False, "n": len(rows)}
    if not rows:
        result["error"] = "no_rows"
        return result

    n = len(rows)
    result["overall_wr"] = sum(r["win"] for r in rows) / n

    buckets = build_buckets(rows, bucket_size)
    result["buckets"] = buckets
    toxic_zones = detect_toxic_zones(buckets, bucket_size, min_sample)
    result["toxic_zones"] = toxic_zones
    result["anomalies"] = detect_anomalous_buckets(buckets, bucket_size, min_sample)

    candidate_caps, caps_data = build_caps_data(rows, min_sample=min_sample)
    result["candidate_caps"] = candidate_caps
    result["caps_data"] = caps_data

    if not caps_data:
        result["error"] = "no_caps_data"
        return result

    knee = find_knee_point(caps_data, min_sample=min_sample)
    result["knee"] = knee

    ev_data = compute_ev_by_cap(rows, candidate_caps, min_sample=min_sample)
    result["ev_data"] = ev_data
    best_ev = max(ev_data, key=lambda x: x[3]) if ev_data else None
    result["best_ev"] = best_ev
    target_floor = None
    for cap, _n_pass, wr, wr_lo in caps_data:
        if wr_lo >= target_winrate:
            target_floor = cap
            break
    if target_floor is None:
        for cap, _n_pass, wr, _wr_lo in caps_data:
            if wr >= target_winrate:
                target_floor = cap
                break
    result["target_floor"] = target_floor

    if knee is None and best_ev is None and target_floor is None:
        result["error"] = "no_valid_floor"
        return result

    knee_floor = knee if knee is not None else 0.0
    ev_floor = best_ev[0] if best_ev else 0.0
    recommended = max(knee_floor, ev_floor, target_floor or 0.0)
    if recommended <= 0.0:
        result["error"] = "zero_recommendation"
        return result

    rec_subset = [r for r in rows if r["score"] >= recommended]
    rec_n = len(rec_subset)
    rec_wr = sum(r["win"] for r in rec_subset) / rec_n if rec_n else 0.0
    ev, rr, avg_w, avg_l = ev_and_rr_for(rec_subset)
    rec_wilson_lo, rec_wilson_hi = (0.0, 0.0)
    if rec_n:
        rec_wilson_lo, rec_wilson_hi, _ = wilson_ci(int(round(rec_wr * rec_n)), rec_n)
    result.update({
        "recommended": recommended,
        "rec_n": rec_n, "rec_wr": rec_wr, "rec_ev": ev, "rec_rr": rr,
        "rec_avg_win": avg_w, "rec_avg_loss": avg_l,
        "rec_wilson_lo": rec_wilson_lo, "rec_wilson_hi": rec_wilson_hi,
        "confidence": confidence_label(rec_n, rec_wilson_lo, rec_wilson_hi) if rec_n else "LOW",
    })
    buy_wr, buy_n, sell_wr, sell_n = direction_split(rec_subset)
    result.update({"buy_wr": buy_wr, "buy_n": buy_n, "sell_wr": sell_wr, "sell_n": sell_n})

    recent_wr, older_wr, recent_n = detect_temporal_drift(rows)
    result.update({"drift_recent_wr": recent_wr, "drift_older_wr": older_wr, "drift_recent_n": recent_n})

    total_alerts = n
    dropped = total_alerts - rec_n
    ts_list = [r["entry_ts"] for r in rows if r.get("entry_ts")]
    weeks = (max(ts_list) - min(ts_list)) / (7 * 86400) if len(ts_list) > 1 else 0.1
    result.update({
        "alerts_per_week_before": total_alerts / max(weeks, 0.1),
        "alerts_per_week_after": rec_n / max(weeks, 0.1),
        "dropped": dropped,
        "dropped_pct": dropped / total_alerts if total_alerts else 0.0,
    })

    toxic_ceiling = max((t[1] for t in toxic_zones), default=0.0)
    result["toxic_ceiling"] = toxic_ceiling
    result["overlapping_toxic"] = [
        t for t in toxic_zones if t[0] < recommended < t[1] or recommended <= t[0]
    ]

    result["valid"] = True
    return result

# ══════════════════════════════════════════════════════════════════════
#  NEW: Cost-Aware EV + Kelly Sizing  (Recommended.txt §5)
# ══════════════════════════════════════════════════════════════════════

def ev_and_kelly_for(
    rows: List[Row],
    fee_pct: float = 0.0006,
    slippage_pct: float = 0.0003,
) -> Tuple[float, float, float]:
    """Net EV after round-trip fees + slippage, plus Half-Kelly fraction.
    Returns (net_ev_pct, half_kelly_fraction, win_rate)."""
    if not rows:
        return 0.0, 0.0, 0.0
    total_cost = (fee_pct * 2) + (slippage_pct * 2)  # entry + exit for both
    net_moves = []
    for r in rows:
        mag = abs(r.get("pct_move", 0.0))
        if r["win"]:
            net_moves.append(mag - total_cost)
        else:
            net_moves.append(-(mag + total_cost))
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

        curve: List[Dict[str, Any]] = []
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

# ═══════════════════════════════════════════════════════════════════════
#  NEW: Sequential CUSUM Drift Detector  (Recommended.txt §4)
# ══════════════════════════════════════════════════════════════════��════

class CUSUMDetector:
    """Page-Hinkley / CUSUM for binary outcomes. Online, O(1) memory."""

    def __init__(
        self,
        target_wr: float = 0.55,
        drift_delta: float = 0.10,
        threshold: float = 2.0,
    ):
        self.mu = target_wr
        self.delta = drift_delta
        self.h = threshold
        self.s_pos = 0.0
        self.s_neg = 0.0
        self.n = 0

    def update(self, win: bool) -> bool:
        """Feed one outcome. Returns True when drift is detected."""
        self.n += 1
        x = 1.0 if win else 0.0
        self.s_pos = max(0.0, self.s_pos + (x - self.mu) - self.delta / 2)
        self.s_neg = max(0.0, self.s_neg + (self.mu - x) - self.delta / 2)
        return self.s_neg > self.h  # edge-decay direction

    def status(self) -> Dict[str, Any]:
        return {
            "drift_detected": self.s_neg > self.h,
            "s_pos": self.s_pos,
            "s_neg": self.s_neg,
            "n": self.n,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mu": self.mu, "delta": self.delta, "h": self.h,
            "s_pos": self.s_pos, "s_neg": self.s_neg, "n": self.n,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CUSUMDetector":
        det = cls(
            target_wr=d.get("mu", 0.55),
            drift_delta=d.get("delta", 0.10),
            threshold=d.get("h", 2.0),
        )
        det.s_pos = d.get("s_pos", 0.0)
        det.s_neg = d.get("s_neg", 0.0)
        det.n = d.get("n", 0)
        return det


# ══════════════════════════════════════════════════════════════════════
#  NEW: Config Stability Gate  (Recommended.txt §3)
# ═══════════════════════════════════════════════════════════════════════

class StabilityGate:
    """Prevents threshold oscillation across consecutive brain runs."""

    def __init__(self, min_history: int = 3, max_jump: float = 2.0):
        self.min_history = min_history
        self.max_jump = max_jump

    def approve(
        self, proposed: float, history: List[float],
    ) -> Tuple[bool, str]:
        if len(history) < self.min_history:
            return True, "insufficient_history"
        median = statistics.median(history)
        deviation = abs(proposed - median)
        if deviation > self.max_jump:
            return False, (
                f"proposed {proposed:.1f} deviates {deviation:.1f} "
                f"from median {median:.1f} (max {self.max_jump})"
            )
        return True, "ok"


# ═══════════════════════════════════════════════════════════════════════
#  NEW: Vote-Count OOD Gate  (Recommended.txt §8)
# ══════════════════════════════════════════════════════════════════════

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

def is_vote_pattern_ood(
    rows: List[Row],
    current_votes: Dict[str, bool],
    alert_key: str,
) -> Tuple[bool, Dict[str, Any]]:
    """Reject if vote count is outside historical 5th-95th percentile.
    Returns (is_ood, detail_dict)."""
    historical_counts = []
    for r in rows:
        if r.get("alert_key") != alert_key or not r.get("votes"):
            continue
        historical_counts.append(
            sum(1 for v in r["votes"].values() if v)
        )
    if len(historical_counts) < 10:
        return False, {"reason": "insufficient_history", "n": len(historical_counts)}

    current_count = sum(1 for v in current_votes.values() if v)
    lo = _percentile(historical_counts, 5)
    hi = _percentile(historical_counts, 95)
    ood = current_count < lo or current_count > hi
    return ood, {
        "current_count": current_count,
        "hist_p5": lo,
        "hist_p95": hi,
        "n_history": len(historical_counts),
    }

def is_vote_count_ood(
    current_count: int,
    historical_counts: List[int],
    min_history: int = 10,
    margin: int = 2,
    p5: int = 5,
    p95: int = 95,
    relaxed_mode: bool = True,
) -> Tuple[bool, Dict[str, Any]]:
    """Lightweight variant of is_vote_pattern_ood() for callers that only
    have a running list of past vote-counts (e.g. a capped Redis list per
    alert_key) rather than full Row objects. This is what the live
    dispatch path uses; the offline analyzer still uses
    is_vote_pattern_ood() directly against full rows.
    
    relaxed_mode adds a margin to the percentile bounds so that small
    deviations from the historical range don't trigger false positives.
    """
    if len(historical_counts) < min_history:
        return False, {
            "reason": "insufficient_history", 
            "n": len(historical_counts),
            "min_history": min_history,
        }
    
    counts_f = [float(c) for c in historical_counts]
    lo = _percentile(counts_f, p5)
    hi = _percentile(counts_f, p95)
    
    # Apply margin if relaxed mode is enabled
    if relaxed_mode:
        ood = current_count < (lo - margin) or current_count > (hi + margin)
    else:
        ood = current_count < lo or current_count > hi
    
    return ood, {
        "current_count": current_count,
        "hist_p5": lo,
        "hist_p95": hi,
        "n_history": len(historical_counts),
        "margin_applied": margin if relaxed_mode else 0,
        "relaxed_mode": relaxed_mode,
    }

# ═══════════════════════════════════════════════════════════════════════
#  NEW: Block-Bootstrap EV Confidence Intervals  (Recommended.txt §6)
# ═══════════════════════════════════════════════════════════════════════

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

# ═══════════════════════════════════════════════════════════════════════
#  PHASE 1.5 — VOTE WEIGHT OPTIMIZER (Logistic Regression)
# ═══════════════════════════════════════════════════════════════════════

def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)

# ═══════════════════════════════════════════════════════════════════════
#  PHASE 2 — PARAMETER AUTOPSY ENGINE
# ═══════════════════════════════════════════════════════════════════════

def parameter_autopsy(
    rows: List[Row],
    param_field: str,
    target_winrate: float = 0.55,
    min_sample: int = 30,
    n_quantiles: int = 5,
    higher_is_worse: bool = False,  # NEW: direction parameter
) -> Dict[str, Any]:
    valid = [r for r in rows if r.get("context") and r["context"].get(param_field) is not None]
    if len(valid) < min_sample:
        return {"valid": False, "error": f"insufficient_data: {len(valid)} < {min_sample}"}

    valid.sort(key=lambda r: r["context"][param_field])
    bucket_size = max(1, len(valid) // n_quantiles)
    buckets: List[Dict[str, Any]] = []

    for i in range(n_quantiles):
        lo = i * bucket_size
        hi = (i + 1) * bucket_size if i < n_quantiles - 1 else len(valid)
        chunk = valid[lo:hi]
        vals = [r["context"][param_field] for r in chunk]
        wins = sum(r["win"] for r in chunk)
        n = len(chunk)
        wr = wins / n
        wlo, whi, _ = wilson_ci(wins, n)
        buckets.append({
            "range": (round(min(vals), 4), round(max(vals), 4)),
            "n": n,
            "wr": round(wr, 4),
            "wilson_lo": round(wlo, 4),
            "wilson_hi": round(whi, 4),
        })

    optimal_cutoff = None
    
    # FIX: Iterate in the correct direction based on parameter semantics
    if higher_is_worse:
        # For parameters where higher values are bad (e.g., RSI buy cap):
        # Find the LOWEST value where performance drops below target
        for b in buckets:
            if b["wilson_hi"] < target_winrate:
                optimal_cutoff = b["range"][0]
                break
    else:
        # For parameters where higher values are good (e.g., ADX strength):
        # Find the HIGHEST value where performance drops below target
        for b in reversed(buckets):
            if b["wilson_hi"] < target_winrate:
                optimal_cutoff = b["range"][1]
                break

    if optimal_cutoff is None:
        optimal_cutoff = buckets[-1]["range"][1] if higher_is_worse else buckets[0]["range"][0]

    return {
        "valid": True,
        "param": param_field,
        "buckets": buckets,
        "optimal_cutoff": round(optimal_cutoff, 4),
        "higher_is_worse": higher_is_worse,
    }

# ═══════════════════════════════════════════════════════════════════════
#  PHASE 3 — CONDITIONAL ALERT GATING
# ═══════════════════════════════════════════════════════════════════════

def conditional_performance(
    rows: List[Row],
    alert_key: str,
    condition_field: str,
    condition_threshold: float,
    min_sample: int = 15,
) -> Dict[str, Any]:
    subset = [
        r for r in rows
        if r.get("alert_key") == alert_key
        and r.get("context")
        and r["context"].get(condition_field) is not None
    ]
    if len(subset) < min_sample * 2:
        return {"valid": False, "error": "insufficient_data"}

    above = [r for r in subset if r["context"][condition_field] > condition_threshold]
    below = [r for r in subset if r["context"][condition_field] <= condition_threshold]
    if len(above) < min_sample or len(below) < min_sample:
        return {"valid": False, "error": "insufficient_split"}

    def _stats(chunk: List[Row]) -> Dict[str, Any]:
        wins = sum(r["win"] for r in chunk)
        n = len(chunk)
        wr = wins / n
        lo, hi, _ = wilson_ci(wins, n)
        return {"n": n, "wr": wr, "wilson_lo": lo, "wilson_hi": hi}

    a_stats = _stats(above)
    b_stats = _stats(below)
    gap = a_stats["wr"] - b_stats["wr"]

    recommendation = "neutral"
    if gap < -0.10 and a_stats["wilson_hi"] < 0.50:
        recommendation = "disable_when_above"
    elif gap > 0.10 and b_stats["wilson_hi"] < 0.50:
        recommendation = "disable_when_below"

    return {
        "valid": True,
        "alert_key": alert_key,
        "condition": f"{condition_field} > {condition_threshold}",
        "above": a_stats,
        "below": b_stats,
        "gap": round(gap, 4),
        "recommendation": recommendation,
    }
# ═══════════════════════════════════════════════════════════════════════
#  PHASE 4 — VOTE INTERACTION MINER
# ═══════════════════════════════════════════════════════════════════════

def interaction_miner(
    rows: List[Row],
    min_sample: int = 20,
) -> List[Dict[str, Any]]:
    """Mine pairwise vote interactions (synergy + poison).

    Each emitted dict carries the raw counts (n_both, n_only_v1, n_only_v2,
    n_neither) and a precomputed two-proportion p_value alongside the win-
    rate point estimates. Downstream consumers — notably the Benjamini-
    Hochberg FDR pass in brain.py — read p_value directly rather than
    reconstructing it from partial rates; the previous version emitted
    only rates, so every interaction was treated as equally trustworthy
    regardless of the sample size behind it.

    p_value is None when the miner cannot form a valid comparison (e.g.
    the reference arm is too thin to pass min_sample); the FDR pass
    treats None as "not tested" and leaves the recommendation alone.
    """
    vote_names: Set[str] = set()
    for r in rows:
        if r.get("votes"):
            vote_names.update(r["votes"].keys())
    vote_names = sorted(vote_names)
    interactions: List[Dict[str, Any]] = []

    for i, v1 in enumerate(vote_names):
        for v2 in vote_names[i + 1 :]:
            both = [r for r in rows if r.get("votes") and r["votes"].get(v1) and r["votes"].get(v2)]
            only_v1 = [r for r in rows if r.get("votes") and r["votes"].get(v1) and not r["votes"].get(v2)]
            only_v2 = [r for r in rows if r.get("votes") and r["votes"].get(v2) and not r["votes"].get(v1)]
            neither = [r for r in rows if r.get("votes") and not r["votes"].get(v1) and not r["votes"].get(v2)]

            n_both = len(both)
            n_only_v1 = len(only_v1)
            n_only_v2 = len(only_v2)
            n_neither = len(neither)

            # Outer gate unchanged: we need a valid "together" arm AND a
            # valid "v1 alone" arm. Both are load-bearing; without either,
            # any signal we detect would be a comparison against noise.
            if n_both < min_sample or n_only_v1 < min_sample:
                continue

            has_v2_sample = n_only_v2 >= min_sample
            has_neither_sample = n_neither >= min_sample

            wins_both = sum(r["win"] for r in both)
            wins_only_v1 = sum(r["win"] for r in only_v1)

            wr_both = wins_both / n_both
            wr_only_v1 = wins_only_v1 / n_only_v1

            # Only compute the v2-alone and neither rates when their sample
            # actually clears min_sample. Falling back to a fabricated 0.0
            # (the old behavior) silently produced a hypothetical "0% WR"
            # comparison arm that no data supported — fine when some other
            # arm was higher in max(), wrong when it wasn't.
            wr_only_v2 = (
                sum(r["win"] for r in only_v2) / n_only_v2
                if has_v2_sample else None
            )
            wr_neither = (
                sum(r["win"] for r in neither) / n_neither
                if has_neither_sample else None
            )

            # ── Synergy ─────────────────────────────────────────────────
            # Baseline = the strongest of whichever comparison arms have
            # sufficient data. v1-alone always qualifies (outer gate), so
            # valid_baselines is never empty.
            valid_baselines = [wr_only_v1]
            if wr_only_v2 is not None:
                valid_baselines.append(wr_only_v2)
            if wr_neither is not None:
                valid_baselines.append(wr_neither)
            synergy_baseline = max(valid_baselines)

            synergy = wr_both - synergy_baseline
            if synergy > 0.10:
                # Significance test: together vs v1-alone. v1-alone is the
                # one arm guaranteed valid by the outer gate, so the test
                # is always computable.
                p = two_proportion_p_value(
                    wins_both, n_both, wins_only_v1, n_only_v1
                )
                entry: Dict[str, Any] = {
                    "pair": (v1, v2),
                    "type": "synergy",
                    "delta": round(synergy, 4),
                    "wr_both": round(wr_both, 4),
                    "wr_only_v1": round(wr_only_v1, 4),
                    "n_both": n_both,
                    "n_only_v1": n_only_v1,
                    "p_value": round(p, 6),
                }
                if wr_only_v2 is not None:
                    entry["wr_only_v2"] = round(wr_only_v2, 4)
                    entry["n_only_v2"] = n_only_v2
                if wr_neither is not None:
                    entry["wr_neither"] = round(wr_neither, 4)
                    entry["n_neither"] = n_neither
                interactions.append(entry)

            # ── v2 poisons v1 ───────────────────────────────────────────
            # Reference arm is v1-alone (guaranteed valid). Test: does
            # adding v2 drag v1's win rate down?
            poison_v1 = wr_only_v1 - wr_both
            if poison_v1 > 0.15 and n_both >= min_sample:
                p = two_proportion_p_value(
                    wins_both, n_both, wins_only_v1, n_only_v1
                )
                entry = {
                    "pair": (v1, v2),
                    "type": "poison",
                    "delta": round(-poison_v1, 4),
                    "wr_both": round(wr_both, 4),
                    "wr_only_v1": round(wr_only_v1, 4),
                    "n_both": n_both,
                    "n_only_v1": n_only_v1,
                    "poisoner": v2,
                    "victim": v1,
                    "note": f"{v2} poisons {v1}",
                    "p_value": round(p, 6),
                }
                if wr_only_v2 is not None:
                    entry["wr_only_v2"] = round(wr_only_v2, 4)
                    entry["n_only_v2"] = n_only_v2
                if wr_neither is not None:
                    entry["wr_neither"] = round(wr_neither, 4)
                    entry["n_neither"] = n_neither
                interactions.append(entry)

            # ── v1 poisons v2 ───────────────────────����───────────────────
            if has_v2_sample:
                poison_v2 = wr_only_v2 - wr_both
                if poison_v2 > 0.15 and n_both >= min_sample:
                    wins_only_v2 = sum(r["win"] for r in only_v2)
                    p = two_proportion_p_value(
                        wins_both, n_both, wins_only_v2, n_only_v2
                    )
                    entry = {
                        "pair": (v1, v2),
                        "type": "poison",
                        "delta": round(-poison_v2, 4),
                        "wr_both": round(wr_both, 4),
                        "wr_only_v1": round(wr_only_v1, 4),
                        "wr_only_v2": round(wr_only_v2, 4),
                        "n_both": n_both,
                        "n_only_v1": n_only_v1,
                        "n_only_v2": n_only_v2,
                        "poisoner": v1,
                        "victim": v2,
                        "note": f"{v1} poisons {v2}",
                        "p_value": round(p, 6),
                    }
                    if wr_neither is not None:
                        entry["wr_neither"] = round(wr_neither, 4)
                        entry["n_neither"] = n_neither
                    interactions.append(entry)

    interactions.sort(key=lambda x: -abs(x["delta"]))
    return interactions

# ═══════════════════════════════════════════════════════════════════════
#  PHASE 5 — COUNTERFACTUAL SIMULATOR
# ═══════════════════════════════════════════════════════════════════════

def simulate_config_change(
    rows: List[Row],
    baseline_ev: float,
    new_threshold: Optional[float] = None,
    new_params: Optional[Dict[str, float]] = None,
) -> Optional[Dict[str, Any]]:
    simulated: List[Row] = []
    for r in rows:
        if new_threshold is not None and r["score"] < new_threshold:
            continue
        if new_params and r.get("context"):
            blocked = False
            for param, max_val in new_params.items():
                if r["context"].get(param) is not None and r["context"][param] > max_val:
                    blocked = True
                    break
            if blocked:
                continue
        simulated.append(r)

    if not simulated:
        return None

    wins = sum(r["win"] for r in simulated)
    n = len(simulated)
    wr = wins / n
    
    # FIX: Use ev_and_kelly_for() for NET EV (deducts fees/slippage)
    # This ensures consistent comparison with baseline_ev
    net_ev, _half_kelly, _wr = ev_and_kelly_for(simulated)
    
    # Also compute gross EV for R:R display
    gross_ev, rr, _, _ = ev_and_rr_for(simulated)
    
    return {
        "n": n,
        "wr": round(wr, 4),
        "ev": round(net_ev, 4),  # NET EV - consistent with baseline
        "gross_ev": round(gross_ev, 4),  # Additional info for reference
        "delta_n": n - len(rows),
        "delta_ev": round(net_ev - baseline_ev, 4),  # Compare NET vs NET
        "filtered_out": len(rows) - n,
        "rr": rr,
    } 

# ═══════════════════════════════════════════════════════════════════════
#  PHASE 6 — DYNAMIC REGIME PROFILES
# ═══════════════════════════════════════════════════════════════════════

def regime_profile_optimizer(
    rows: List[Row],
    regime_field: str = "adx_val",
    n_regimes: int = 3,
    min_sample: int = 25,
    target_winrate: float = 0.55,
) -> Dict[str, Any]:
    valid = [r for r in rows if r.get("context") and r["context"].get(regime_field) is not None]
    if len(valid) < min_sample * n_regimes:
        return {"valid": False, "error": "insufficient_data"}

    values = sorted(r["context"][regime_field] for r in valid)
    cuts = [values[int(len(values) * i / n_regimes)] for i in range(1, n_regimes)]

    regimes: List[Dict[str, Any]] = []
    prev = float("-inf")
    for i, cut in enumerate(cuts + [float("inf")]):
        chunk = [r for r in valid if prev <= r["context"][regime_field] < cut]
        prev = cut
        if len(chunk) < min_sample:
            continue
        rec = recommend_threshold(chunk, target_winrate=target_winrate, min_sample=min_sample)
        if rec["valid"]:
            regimes.append({
                "regime_id": i,
                "range": (
                    round(min(r["context"][regime_field] for r in chunk), 2),
                    round(max(r["context"][regime_field] for r in chunk), 2),
                ),
                "n": len(chunk),
                "recommended_threshold": rec["recommended"],
                "wr": round(rec["rec_wr"], 4),
                "ev": round(rec["rec_ev"], 4),
            })
    return {"valid": True, "regime_field": regime_field, "regimes": regimes}

# ══════════════════════════════════════════════════════════════════════
#  RISK FLAGS — Config Version Hash & Actionability
# ═══════════════════════════════════════════════════════════════════════
def compare_config_versions(
    rows: List[Row],
    min_sample: int = 20,
) -> List[Dict[str, Any]]:
    """Group real outcome rows by their tagged config_version and compare
    WR across consecutive versions (ordered by first-seen entry_ts), so a
    config_patch that tanks WR gets flagged instead of going unnoticed."""
    by_version: Dict[str, List[Row]] = defaultdict(list)
    for r in rows:
        ctx = r.get("context")
        if not ctx:
            continue
        cv = ctx.get("config_version")
        if not cv:
            continue
        by_version[cv].append(r)

    if len(by_version) < 2:
        return []

    version_order = sorted(
        by_version.keys(),
        key=lambda v: min(r.get("entry_ts", 0) for r in by_version[v]),
    )

    comparisons: List[Dict[str, Any]] = []
    for prev_v, cur_v in zip(version_order, version_order[1:]):
        prev_rows = by_version[prev_v]
        cur_rows = by_version[cur_v]
        if len(prev_rows) < min_sample or len(cur_rows) < min_sample:
            continue

        prev_wins = sum(r["win"] for r in prev_rows)
        cur_wins = sum(r["win"] for r in cur_rows)
        prev_wr = prev_wins / len(prev_rows)
        cur_wr = cur_wins / len(cur_rows)
        prev_lo, prev_hi, _ = wilson_ci(prev_wins, len(prev_rows))
        cur_lo, cur_hi, _ = wilson_ci(cur_wins, len(cur_rows))
        delta = cur_wr - prev_wr

        comparisons.append({
            "prev_version": prev_v,
            "cur_version": cur_v,
            "prev_n": len(prev_rows),
            "cur_n": len(cur_rows),
            "prev_wr": round(prev_wr, 4),
            "cur_wr": round(cur_wr, 4),
            "delta_wr": round(delta, 4),
            # non-overlapping Wilson intervals = statistically meaningful move
            "regression": delta < -0.05 and cur_hi < prev_lo,
            "improvement": delta > 0.05 and cur_lo > prev_hi,
        })

    return comparisons

_STRUCTURAL_CONFIG_FIELDS: Tuple[str, ...] = (
    # ── gate periods ──
    "PPO_FAST", "PPO_SLOW", "PPO_SIGNAL",
    "PPO_GATE_FAST", "PPO_GATE_SLOW", "PPO_GATE_SIGNAL",
    "RMA_50_PERIOD", "RMA_200_PERIOD", "RMA_CLOUD_FAST_PERIOD",
    "RSI_GUARD_RSI_LEN", "RSI_GUARD_KALMAN_LEN", "RSI_GUARD_EMA_LEN",
    "SRSI_RSI_LEN", "SRSI_KALMAN_LEN", "SRSI_EMA_LEN",
    "HIST_RMA_FAST", "HIST_RMA_SLOW",
    "ATR_SHORT", "ATR_LONG",
    "ADX_DI_LENGTH", "ADX_SMOOTHING_LENGTH",
    "ICHIMOKU_CONVERSION_PERIODS", "ICHIMOKU_BASE_PERIODS",
    "ICHIMOKU_SPANB_PERIODS", "ICHIMOKU_DISPLACEMENT",
    "ICHIMOKU_TK_CONVERSION_PERIODS", "ICHIMOKU_TK_BASE_PERIODS",
    "DYNAMIC_FLOW_FACTOR", "DYNAMIC_FLOW_BASIS_LENGTH", "DYNAMIC_FLOW_DIST_LENGTH",
    # ── lookback windows that reshape the feature set ──
    "OB_LOOKBACK_CANDLES", "OB_IMPULSE_LOOKAHEAD", "OB_CONFIRM_LOOKAHEAD_CANDLES",
    "OB_PERSISTENCE_CANDLES", "OB_MIN_PENETRATION_ATR_MULT",
    "CHOCH_SWING_LEN", "CHOCH_LOOKBACK_CANDLES", "CHOCH_CONFIRM_WINDOW_CANDLES",
    "FIB_REVERSAL_SWING_LENGTH", "FIB_REVERSAL_SWING_LOOKBACK_CANDLES",
    "ATR_PCTL_LOOKBACK", "VOLUME_PCTL_LOOKBACK",
    "PIVOT_LOOKBACK_PERIOD",
    # ── adaptive thresholds carried on each alert ──
    "PPO_ADAPTIVE_CALM", "PPO_ADAPTIVE_VOLATILE",
    "RSI_ADAPTIVE_BUY_CALM", "RSI_ADAPTIVE_BUY_VOLATILE",
    "RSI_ADAPTIVE_SELL_CALM", "RSI_ADAPTIVE_SELL_VOLATILE",
    "ADX_ADAPTIVE_TARGET_PCTL", "ADX_STRENGTH_PCTL",
    "ATR_PCTL_VOTE_MIN", "VOLUME_PCTL_VOTE_MIN",
    "RVOL_THRESHOLD", "ADAPTIVE_MULT_CALM", "ADAPTIVE_MULT_VOLATILE",
    # ── flags that toggle which votes exist at all ──
    "ENABLE_PPO_GATE", "RSI_GUARD_ENABLED",
    "RMA_CLOUD_ENABLED", "ICHIMOKU_CLOUD_ENABLED", "DYNAMIC_FLOW_RIBBON_ENABLED",
    "ICHIMOKU_TK_GUARD_ENABLED",
    "ENABLE_ADX_STRENGTH_VOTE", "ENABLE_ATR_PCTL_VOTE", "ENABLE_VOLUME_PCTL_VOTE",
    "ENABLE_PPO_GATE_MOMENTUM_VOTE", "ENABLE_RSI_GUARD_MOMENTUM_VOTE",
    "ENABLE_RMA_CLOUD_MOMENTUM_VOTE", "ENABLE_VWAP_MOMENTUM_VOTE",
    "ENABLE_OB_GATE", "ENABLE_OI_FUNDING_FILTER", "ENABLE_CPR",
    "OB_MIN_OTHER_SCORE",
)

def hash_config_state(
    weights: Dict[str, float],
    threshold: float,
    min_pct: float,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> str:
    """Stable fingerprint of the config knobs that actually change the
    feature/outcome distribution.

    The original only hashed weights + threshold + min_pct. That misses
    every tunable that reshapes the input features — e.g. flipping
    ICHIMOKU_CLOUD_ENABLED, changing OB_LOOKBACK_CANDLES, or swapping
    PPO_GATE_FAST changes the vote set/values without changing any of the
    three hashed fields. Rows tagged with the same hash but generated
    under different structural settings then get compared as if they were
    the same version, quietly corrupting compare_config_versions().

    extra_fields: callers that already have a subset of cfg values handy
    can pass them in. When None, we pull from the live cfg via the
    imported reference. `default=str` on json.dumps is a defensive
    fallback so any unexpected non-serializable straggler doesn't crash
    the entire report.
    """
    if extra_fields is None:
        extra_fields = {}
        for field in _STRUCTURAL_CONFIG_FIELDS:
            try:
                extra_fields[field] = getattr(cfg, field, None)
            except Exception:
                pass
    payload = json.dumps(
        {
            "w": weights,
            "t": threshold,
            "p": min_pct,
            "x": extra_fields,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.md5(payload.encode()).hexdigest()[:12] 

def score_actionability(rec: Dict[str, Any]) -> float:
    impact = abs(rec.get("delta_ev", 0)) * 100.0
    confidence = 1.0
    if rec.get("wilson_hi") is not None and rec.get("wilson_lo") is not None:
        confidence = max(0.1, 1.0 - (rec["wilson_hi"] - rec["wilson_lo"]))
    effort = 1.0
    if rec.get("type") == "dynamic_regime_profile":
        effort = 3.0
    elif rec.get("type") == "conditional_gating":
        effort = 2.0
    return impact * confidence / effort


def learned_actionability(
    rec: Dict[str, Any],
    success_rates: Optional[Dict[str, Dict[str, float]]] = None,
) -> float:
    """Hand-formula blended with two learned signals:

    1. Category-level help-rate from the repair ledger (per-category
       empirical track record).
    2. Per-repair P(helps | current state) from the contextual logistic
       model (`learn_repair_effectiveness`), when available on the rec
       as `p_helps_learned`.

    Falls back to the hand-formula verbatim when neither signal has
    enough data — no repair is penalized for being new."""
    base = score_actionability(rec)

    # ── Signal 1: contextual ML prediction for THIS repair ──
    # Multiplicative modulation in [0.5, 1.5] so a strongly predicted-help
    # repair gets boosted and a predicted-hurt one gets suppressed, but
    # neither can zero the base score. Without this, the learned model's
    # output is written into the rec and read by nothing.
    p_help = rec.get("p_helps_learned")
    if isinstance(p_help, (int, float)):
        base *= (0.5 + float(p_help))

    # ── Signal 2: category-level empirical blend ──
    if not success_rates:
        return base
    cat = rec.get("category") or rec.get("type") or "unknown"
    stats = success_rates.get(cat)
    if not stats or stats.get("n", 0) < 8:
        return base
    help_rate = stats.get("help_rate", 0.5)
    hurt_rate = stats.get("hurt_rate", 0.0)
    # Net empirical value in [0, 1]; 0.5 = neutral
    learned = max(0.0, min(1.0, help_rate * (1.0 - hurt_rate)))
    blend = 0.5 * base + 0.5 * (base * learned * 2.0)
    return blend


# ═══════════════════════════════════════════════════════════════════════
#  THREE-METRIC OUTCOME ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def multi_metric_summary(rows: List[Row], min_sample: int = 10) -> Dict[str, Any]:
    """Compute all three win/loss metrics across the full row set.
    Returns close_wr, mfe_wr, mae_loss_rate, clean_win_rate,
    and tp_before_sl ordering stats."""
    n = len(rows)
    if n < min_sample:
        return {"valid": False, "error": "insufficient_data", "n": n}

    close_wins = sum(1 for r in rows if r.get("close_win", r["win"]))
    mfe_wins = sum(1 for r in rows if r.get("mfe_win") is True)
    mae_losses = sum(1 for r in rows if r.get("mae_loss") is True)

    # ── McNemar 2×2 table for close_win vs mfe_win ────────────────────
    paired_rows = [r for r in rows if r.get("mfe_win") is not None]
    n_paired = len(paired_rows)
    if n_paired >= 1:
        both_wins = sum(
            1 for r in paired_rows
            if r.get("close_win", r["win"]) and r["mfe_win"] is True
        )
        mfe_only = sum(
            1 for r in paired_rows
            if r.get("close_win", r["win"]) is False and r["mfe_win"] is True
        )
        close_only = sum(
            1 for r in paired_rows
            if r.get("close_win", r["win"]) and r["mfe_win"] is False
        )
        neither_wins = n_paired - both_wins - mfe_only - close_only
    else:
        both_wins = mfe_only = close_only = neither_wins = 0

    # tp_first: True means TP was hit before SL (the "realistic" win)
    tp_first_rows = [r for r in rows if r.get("tp_first") is not None]
    tp_before_sl = sum(1 for r in tp_first_rows if r["tp_first"] is True)
    sl_before_tp = sum(1 for r in tp_first_rows if r["tp_first"] is False)

    # mfe_win but also mae_loss — hit both levels
    both_hit = sum(
        1 for r in rows
        if r.get("mfe_win") is True and r.get("mae_loss") is True
    )

    # "Clean" wins: reached TP without ever hitting SL level
    clean_wins = sum(
        1 for r in rows
        if r.get("mfe_win") is True and r.get("mae_loss") is not True
    )
    result: Dict[str, Any] = {
        "valid": True,
        "n": n,
        "close_wr": close_wins / n,
        "mfe_wr": mfe_wins / n,
        "mae_loss_rate": mae_losses / n,
        "clean_win_rate": clean_wins / n,
        "both_hit_rate": both_hit / n,
        # ── McNemar inputs (see mcnemar_exact_p) ──
        "mfe_only": mfe_only,
        "close_only": close_only,
        "both_wins": both_wins,
        "neither_wins": neither_wins,
        "n_paired": n_paired,
    }
    if tp_first_rows:
        result["tp_before_sl_rate"] = tp_before_sl / len(tp_first_rows)
        result["sl_before_tp_rate"] = sl_before_tp / len(tp_first_rows)
        result["ordering_sample"] = len(tp_first_rows)

    # ── Bonus stats ──
    bonus_wins = sum(1 for r in rows if r.get("bonus_win"))
    rr_values = [r.get("rr_achieved", 0) for r in rows if r.get("rr_achieved", 0) > 0]
    result["bonus_wins"] = bonus_wins
    result["bonus_rate"] = bonus_wins / n if n else 0.0
    result["avg_rr_achieved"] = statistics.fmean(rr_values) if rr_values else 0.0

    # ── Bonus-weighted win rate (win_weight-aware) ──
    total_win_weight = sum(
        r.get("win_weight", 1.0 if r["win"] else 0.0) for r in rows
    )
    result["weighted_wr"] = min(total_win_weight / n, 1.0) if n else 0.0
    result["weighted_wr_raw"] = sum(1 for r in rows if r["win"]) / n

    # Wilson CIs for the key metrics
    close_lo, close_hi, _ = wilson_ci(close_wins, n)
    mfe_lo, mfe_hi, _ = wilson_ci(mfe_wins, n)
    result["close_wilson"] = (close_lo, close_hi)
    result["mfe_wilson"] = (mfe_lo, mfe_hi)
    return result

def multi_metric_per_alert(rows: List[Row], min_sample: int = 10) -> List[Dict[str, Any]]:
    """Per-alert breakdown showing all three metrics. Sorted by mfe_wr
    descending (most actionable metric first)."""
    by_alert: Dict[str, List[Row]] = defaultdict(list)
    for r in rows:
        by_alert[r["alert_key"]].append(r)

    results = []
    for ak, alert_rows in by_alert.items():
        if len(alert_rows) < min_sample:
            continue
        n = len(alert_rows)
        close_wins = sum(1 for r in alert_rows if r.get("close_win", r["win"]))
        mfe_wins = sum(1 for r in alert_rows if r.get("mfe_win") is True)
        mae_losses = sum(1 for r in alert_rows if r.get("mae_loss") is True)
        clean_wins = sum(
            1 for r in alert_rows
            if r.get("mfe_win") is True and r.get("mae_loss") is not True
        )

        results.append({
            "alert_key": ak,
            "n": n,
            "close_wr": close_wins / n,
            "mfe_wr": mfe_wins / n,
            "mae_loss_rate": mae_losses / n,
            "clean_win_rate": clean_wins / n,
            "gap_mfe_vs_close": (mfe_wins - close_wins) / n,
        })

    results.sort(key=lambda x: -x["mfe_wr"])
    return results

def multi_metric_per_pair(rows: List[Row], min_sample: int = 15) -> List[Dict[str, Any]]:
    """Per-pair three-metric breakdown."""
    by_pair: Dict[str, List[Row]] = defaultdict(list)
    for r in rows:
        by_pair[r["pair"]].append(r)

    results = []
    for pair, pair_rows in by_pair.items():
        if len(pair_rows) < min_sample:
            continue
        n = len(pair_rows)
        close_wins = sum(1 for r in pair_rows if r.get("close_win", r["win"]))
        mfe_wins = sum(1 for r in pair_rows if r.get("mfe_win") is True)
        mae_losses = sum(1 for r in pair_rows if r.get("mae_loss") is True)

        results.append({
            "pair": pair,
            "n": n,
            "close_wr": close_wins / n,
            "mfe_wr": mfe_wins / n,
            "mae_loss_rate": mae_losses / n,
        })

    results.sort(key=lambda x: -x["mfe_wr"])
    return results

# ═══════════════════════════════════════════════════════════════════════
#  ENHANCED WEIGHT OPTIMIZER — Walk-Forward + Confidence + Delta Limit
# ═══════════════════════════════════════════════════════════════════════

def _build_vote_dataset(
    rows: List[Row],
    vote_names: List[str],
) -> Tuple[List[List[float]], List[float], List[float]]:
    """Extract vote feature matrix, labels, and sample weights."""
    X: List[List[float]] = []
    y: List[float] = []
    sw: List[float] = []
    for r in rows:
        votes = r.get("votes")
        if not votes or not isinstance(votes, dict):
            continue
        vec = [1.0] + [1.0 if votes.get(vn) else 0.0 for vn in vote_names]
        X.append(vec)
        y.append(1.0 if r["win"] else 0.0)
        sw.append(r.get("win_weight", 1.0) if r["win"] else 1.0)
    return X, y, sw

def _train_logistic(
    X: List[List[float]], y: List[float], sample_weights: List[float],
    max_iter: int = 2000, lr: float = 0.05, l2: float = 0.01,
) -> List[float]:
    """Pure gradient-descent logistic regression. Returns beta vector."""
    n = len(X)
    if n == 0:
        return []
    n_features = len(X[0])
    win_rate = sum(y) / n
    beta = [0.0] * n_features
    beta[0] = math.log(win_rate / (1 - win_rate)) if 0 < win_rate < 1 else 0.0
    total_sw = sum(sample_weights) or 1.0

    for iteration in range(max_iter):
        grad = [0.0] * n_features
        for i in range(n):
            z = sum(beta[j] * X[i][j] for j in range(n_features))
            p = _sigmoid(z)
            error = p - y[i]
            w = sample_weights[i]
            for j in range(n_features):
                grad[j] += error * X[i][j] * w
        step = lr * (0.5 * (1 + math.cos(math.pi * iteration / max_iter)))
        for j in range(n_features):
            grad[j] = grad[j] / total_sw + l2 * beta[j]
            beta[j] -= step * grad[j]
    return beta

def _score_with_beta(
    rows: List[Row], beta: List[float], vote_names: List[str],
    threshold: float = 0.5,
) -> Tuple[float, int]:
    """Apply logistic beta to rows, return (win_rate, n) for predicted-positive."""
    hits = 0
    total = 0
    for r in rows:
        votes = r.get("votes")
        if not votes or not isinstance(votes, dict):
            continue
        z = beta[0] + sum(
            beta[j + 1] for j, vn in enumerate(vote_names) if votes.get(vn)
        )
        if _sigmoid(z) >= threshold:
            total += 1
            if r["win"]:
                hits += 1
    wr = hits / total if total else 0.0
    return wr, total

def _map_coefficients_to_weights(
    beta: List[float],
    vote_names: List[str],
    current_weights: Dict[str, float],
    max_delta: float = 2.0,
) -> Dict[str, float]:
    """Map logistic coefficients → confluence weights with delta limiting.
    
    KEY FIX: Instead of the old aggressive `3.0 * (coeff / avg)` formula,
    this uses a soft tanh-based mapping anchored to CURRENT weights, then
    clamps the per-cycle delta to ±max_delta. This prevents 1→5 / 3→0 jumps.
    """
    coeffs = beta[1:] if len(beta) > 1 else []
    if not coeffs:
        return dict(current_weights)

    # Soft mapping: tanh squashes extreme coefficients
    positive_coeffs = [max(0.0, c) for c in coeffs]
    total_pos = sum(positive_coeffs)
    if total_pos <= 0:
        return dict(current_weights)

    suggested: Dict[str, float] = {}
    for idx, vn in enumerate(vote_names):
        c = coeffs[idx]
        current = current_weights.get(vn, 1.0)

        if c < -0.05:
            # Negative coefficient → reduce weight, but don't zero it in one step
            raw = current * 0.5
        elif positive_coeffs[idx] > 0:
            # Proportional share, scaled to [0.5, 5.0] range via tanh
            share = positive_coeffs[idx] / total_pos
            raw = 0.5 + 4.5 * math.tanh(share * len(vote_names) * 0.5)
        else:
            raw = current * 0.75  # Near-zero coefficient → gentle decay

        # ── DELTA LIMIT: prevent extreme per-cycle jumps ──
        delta = raw - current
        delta = max(-max_delta, min(max_delta, delta))
        final = max(0.0, min(5.0, current + delta))
        suggested[vn] = round(final, 2)

    return suggested

def optimize_vote_weights(
    rows: List[Row],
    current_weights: Dict[str, float],
    min_sample: int = 100,
    max_iter: int = 2000,
    lr: float = 0.05,
    l2: float = 0.01,
    walk_forward: bool = True,
    max_weight_delta: float = 2.0,
    wf_train_frac: float = 0.67,
) -> Dict[str, Any]:
    """Data-driven CONFLUENCE_WEIGHTS via logistic regression.
    
    v2 ENHANCEMENTS:
    • Walk-forward validation: trains on older 67%, validates on newer 33%.
      If holdout WR degrades, the result is flagged invalid.
    • Delta-limited weight mapping: max ±max_weight_delta per vote per cycle.
    • Confidence score: combines sample size, convergence, and WF margin.
    • Bootstrap stability: runs 5 bootstrap resamples, reports coefficient
      variance as a stability metric.
    """
    vote_names = sorted(current_weights.keys())
    X, y, sample_weights = _build_vote_dataset(rows, vote_names)
    n = len(X)

    if n < min_sample:
        return {"valid": False, "error": f"insufficient_data: {n} < {min_sample}"}

    wf_passed: Optional[bool] = None
    wf_holdout_wr: Optional[float] = None
    wf_baseline_wr: Optional[float] = None
    confidence_score: float = 0.0

    if walk_forward and n >= min_sample * 2:
        # ── Walk-forward split ──
        train_rows, holdout_rows = walk_forward_split(rows, wf_train_frac)
        X_train, y_train, sw_train = _build_vote_dataset(train_rows, vote_names)

        if len(X_train) < min_sample // 2 or len(holdout_rows) < min_sample // 3:
            # Fall back to full-data training with low confidence
            beta = _train_logistic(X, y, sample_weights, max_iter, lr, l2)
            confidence_score = min(0.3, n / 1000.0)
        else:
            beta = _train_logistic(X_train, y_train, sw_train, max_iter, lr, l2)

            # Validate on holdout
            wf_holdout_wr, wf_n = _score_with_beta(holdout_rows, beta, vote_names)
            wf_baseline_wr = sum(r["win"] for r in holdout_rows) / len(holdout_rows) if holdout_rows else 0.0

            if wf_n >= 10:
                wf_passed = wf_holdout_wr >= wf_baseline_wr
                if not wf_passed:
                    return {
                        "valid": False,
                        "error": "walk_forward_degraded",
                        "n_samples": n,
                        "holdout_wr": round(wf_holdout_wr, 4),
                        "baseline_holdout_wr": round(wf_baseline_wr, 4),
                        "message": (
                            f"Optimized weights DEGRADE holdout WR: "
                            f"{wf_holdout_wr:.0%} vs baseline {wf_baseline_wr:.0%}. "
                            f"Keeping current weights."
                        ),
                    }
                confidence_score = min(1.0, (n / 500.0) * (1.0 + (wf_holdout_wr - wf_baseline_wr) * 5.0))
            else:
                wf_passed = None
                confidence_score = min(0.4, n / 1000.0)
    else:
        beta = _train_logistic(X, y, sample_weights, max_iter, lr, l2)
        confidence_score = min(0.3, n / 1000.0)  # No WF = low confidence

    # ── Bootstrap stability check (5 resamples) ──
    bootstrap_betas: List[List[float]] = []
    rng = random.Random(42)
    for _ in range(5):
        indices = [rng.randint(0, n - 1) for _ in range(n)]
        X_boot = [X[i] for i in indices]
        y_boot = [y[i] for i in indices]
        sw_boot = [sample_weights[i] for i in indices]
        b = _train_logistic(X_boot, y_boot, sw_boot, max_iter // 2, lr, l2)
        if b:
            bootstrap_betas.append(b)

    coeff_stability: Dict[str, float] = {}
    if len(bootstrap_betas) >= 3:
        for j, vn in enumerate(vote_names):
            vals = [b[j + 1] for b in bootstrap_betas if len(b) > j + 1]
            if vals:
                coeff_stability[vn] = round(statistics.pstdev(vals), 4)
        avg_instability = statistics.mean(coeff_stability.values()) if coeff_stability else 1.0
        confidence_score *= max(0.3, 1.0 - avg_instability)

    # ── Map to weights with delta limiting ──
    suggested = _map_coefficients_to_weights(beta, vote_names, current_weights, max_weight_delta)

    # ── Identify negative votes ──
    coeffs = beta[1:] if len(beta) > 1 else []
    negative_votes = [
        (vn, round(coeffs[i], 4))
        for i, vn in enumerate(vote_names)
        if i < len(coeffs) and coeffs[i] < -0.05
    ]

    # ── Compute actual changed votes ──
    changed = []
    for k, new_v in suggested.items():
        old_v = current_weights.get(k, 0.0)
        if abs(new_v - old_v) > 0.1:
            changed.append((k, old_v, new_v))

    intercept = beta[0] if beta else 0.0

    return {
        "valid": True,
        "n_samples": n,
        "intercept": round(intercept, 4),
        "current_weights": dict(current_weights),
        "suggested_weights": suggested,
        "negative_votes": negative_votes,
        "changed_votes": changed,
        "walk_forward_passed": wf_passed,
        "holdout_wr": round(wf_holdout_wr, 4) if wf_holdout_wr is not None else None,
        "baseline_holdout_wr": round(wf_baseline_wr, 4) if wf_baseline_wr is not None else None,
        "confidence": round(confidence_score, 3),
        "confidence_label": confidence_label(
            n,
            max(0.0, 0.5 - confidence_score * 0.3),
            min(1.0, 0.5 + confidence_score * 0.3),
        ),
        "coeff_stability": coeff_stability,
    }

# ═══════════════════════════════════════════════════════════════════════
#  PERMUTATION VOTE IMPORTANCE (AI/ML)
# ═══════════════════════════════════════════════════════════════════════

def permutation_vote_importance(
    rows: List[Row],
    min_sample: int = 30,
    n_permutations: int = 20,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """ML-style permutation importance: shuffle each vote's values and
    measure the WR drop. Votes whose permutation causes the biggest WR
    drop are the most important. More robust than simple with/without
    comparison because it preserves the marginal distribution."""
    if len(rows) < min_sample:
        return []

    vote_names: Set[str] = set()
    for r in rows:
        if r.get("votes"):
            vote_names.update(r["votes"].keys())
    vote_names = sorted(vote_names)
    if not vote_names:
        return []

    rng = random.Random(seed)
    baseline_wr = sum(r["win"] for r in rows) / len(rows)
    results = []

    for vn in vote_names:
        drops = []
        for _ in range(n_permutations):
            shuffled_rows = []
            vote_vals = [r.get("votes", {}).get(vn) for r in rows]
            rng.shuffle(vote_vals)
            for i, r in enumerate(rows):
                sr = dict(r)
                if sr.get("votes"):
                    sr["votes"] = dict(sr["votes"])
                    sr["votes"][vn] = vote_vals[i]
                shuffled_rows.append(sr)
            perm_wr = sum(r["win"] for r in shuffled_rows) / len(shuffled_rows)
            drops.append(baseline_wr - perm_wr)
        mean_drop = statistics.fmean(drops)
        results.append({
            "vote": vn,
            "importance": round(mean_drop, 4),
            "std": round(statistics.pstdev(drops), 4) if len(drops) > 1 else 0.0,
            "direction": "positive" if mean_drop > 0 else "negative",
        })

    results.sort(key=lambda x: -abs(x["importance"]))
    return results

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
    """Posterior P(true EV <= 0) under a flat prior.

    Block-bootstraps the EV statistic via bootstrap_ev_ci() and applies a
    normal approximation on the (mean, std) of the bootstrap distribution.
    Under a flat prior the bootstrap distribution's shape equals the
    posterior's shape around the sample mean, so Phi(-mean/std) is the
    posterior probability that the true EV is <= 0.

    Returns 0.5 (maximum uncertainty) when the sample is too thin for a
    meaningful bootstrap — same convention as _prob_edge_broken()."""
    bs = bootstrap_ev_ci(rows, n_sims=n_sims)
    if not bs.get("valid"):
        return 0.5
    ev_mean = bs["ev_mean"]
    ev_std = bs["ev_std"]
    if ev_std <= 0:
        return 1.0 if ev_mean <= 0 else 0.0
    z = (0.0 - ev_mean) / ev_std
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


# ═══════════════════════════════════════════════════════════════════════
#  ML DIAGNOSTICS — Root Cause, Drift, Change-Point, Repair Learning
# ═══════════════════════════════════════════════════════════════════════

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


def diagnose_root_cause(
    rows: List[Row],
    target_wr: float = 0.55,
    min_segment: int = 15,
    n_quantiles: int = 5,
    top_k: int = 3,
) -> Dict[str, Any]:
    """Decision-stump root-cause attribution.

    Scans every numeric context feature and boolean vote, finds the single
    threshold split that isolates the worst-performing segment of trades,
    and returns ranked 'toxic segment' rules. Converts the repair shop's
    'review trades manually' guidance into an automated, ML-driven answer
    to *why* the system is losing.

    Pure Python, no external deps. Segments are Wilson-gated so tiny noisy
    slices are never reported as confident causes, and each segment carries
    a two-proportion p-value for the downstream BH/FDR pass.
    """
    n = len(rows)
    if n < min_segment * 2:
        return {"valid": False, "error": "insufficient_data", "n": n}

    overall_wins = sum(r["win"] for r in rows)
    overall_wr = overall_wins / n

    feats_per_row = [_flatten_row_features(r) for r in rows]
    feature_names = sorted({f for d in feats_per_row for f in d})
    if not feature_names:
        return {"valid": False, "error": "no_features", "n": n}

    candidates: List[Dict[str, Any]] = []
    for feat in feature_names:
        pairs = [
            (feats_per_row[i][feat], rows[i]["win"])
            for i in range(n) if feat in feats_per_row[i]
        ]
        if len(pairs) < min_segment * 2:
            continue
        values = sorted(p[0] for p in pairs)
        q_thr = sorted({
            values[min(len(values) - 1, int(len(values) * q / n_quantiles))]
            for q in range(1, n_quantiles)
        })
        for thr in q_thr:
            left = [p for p in pairs if p[0] <= thr]
            right = [p for p in pairs if p[0] > thr]
            for side_vals, side_rule in ((left, "<="), (right, ">")):

                seg_n = len(side_vals)
                if seg_n < min_segment or seg_n > len(pairs) - min_segment:
                    continue
                seg_wins = sum(w for _, w in side_vals)
                seg_wr = seg_wins / seg_n
                lo, hi, _ = wilson_ci(seg_wins, seg_n)
                lift = overall_wr - seg_wr
                # Only interested in segments confidently WORSE than average.
                if lift <= 0 or hi >= overall_wr:
                    continue
                p = two_proportion_p_value(
                    seg_wins, seg_n, overall_wins - seg_wins, n - seg_n,
                )
                candidates.append({
                    "feature": feat,
                    "rule": f"{feat} {side_rule} {thr:.3f}",
                    "op": side_rule,
                    "threshold": thr,
                    "segment_wr": seg_wr,
                    "segment_n": seg_n,
                    "lift_vs_overall": lift,
                    "coverage": seg_n / n,
                    "wilson_lo": lo,
                    "wilson_hi": hi,
                    "isolation_score": lift * (seg_n / n),
                    "confident": hi < target_wr,
                    "p_value": p,
                })

    candidates.sort(key=lambda c: -c["isolation_score"])
    seen: Set[str] = set()
    deduped: List[Dict[str, Any]] = []
    for c in candidates:
        if c["feature"] in seen:
            continue
        seen.add(c["feature"])
        deduped.append(c)

    return {
        "valid": bool(deduped),
        "overall_wr": overall_wr,
        "n": n,
        "segments": deduped[:top_k],
    }

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

def detect_feature_drift(
    rows: List[Row],
    recent_n: int = 100,
    psi_threshold: float = 0.25,
) -> Dict[str, Any]:
    """Compare context-feature distributions between the most recent
    `recent_n` rows and the older baseline. Large PSI = regime/feature
    drift — the market changed under the strategy's feet."""
    if len(rows) < recent_n + 60:
        return {"valid": False, "error": "insufficient_data"}
    ordered = sorted(rows, key=lambda r: r.get("entry_ts", 0))
    feats = [_flatten_row_features(r) for r in ordered]
    recent_feats, base_feats = feats[-recent_n:], feats[:-recent_n]
    feature_names = sorted({f for d in feats for f in d})

    drifted: List[Dict[str, Any]] = []
    for feat in feature_names:
        base_vals = [d[feat] for d in base_feats if feat in d]
        rec_vals = [d[feat] for d in recent_feats if feat in d]
        psi = population_stability_index(base_vals, rec_vals)
        if psi is not None and psi >= psi_threshold:
            drifted.append({"feature": feat, "psi": round(psi, 3)})
    drifted.sort(key=lambda d: -d["psi"])
    return {"valid": True, "drifted_features": drifted, "n_recent": recent_n}

def find_wr_change_point(
    rows: List[Row],
    min_side: int = 25,
    min_delta: float = 0.08,
) -> Dict[str, Any]:
    """Locate the timestamp where win rate shifted most, via an O(n)
    two-sample scan using prefix sums. Returns change point + before/after
    WR + the dominant config_version on each side, so a regression can be
    attributed to a specific config patch."""
    ordered = sorted(
        (r for r in rows if r.get("entry_ts", 0) > 0),
        key=lambda r: r["entry_ts"],
    )
    n = len(ordered)
    if n < min_side * 2:
        return {"valid": False, "error": "insufficient_data"}

    pref = [0] * (n + 1)
    for i, r in enumerate(ordered):
        pref[i + 1] = pref[i] + (1 if r["win"] else 0)

    best: Optional[Dict[str, Any]] = None
    for i in range(min_side, n - min_side):
        wl = pref[i]
        wr_ = pref[n] - pref[i]
        p = two_proportion_p_value(wl, i, wr_, n - i)
        delta = wr_ / (n - i) - wl / i
        score = abs(delta) * (1.0 - p)
        if best is None or score > best["score"]:
            best = {
                "score": score,
                "index": i,
                "change_ts": ordered[i]["entry_ts"],
                "wr_before": wl / i,
                "wr_after": wr_ / (n - i),
                "delta": delta,
                "p_value": p,
            }

    if best is None or abs(best["delta"]) < min_delta:
        return {"valid": False, "error": "no_significant_change_point"}

    def _dominant_version(chunk: List[Row]) -> Optional[str]:
        counts: Dict[str, int] = defaultdict(int)
        for r in chunk:
            cv = (r.get("context") or {}).get("config_version")
            if cv:
                counts[cv] += 1
        return max(counts, key=counts.get) if counts else None

    best["version_before"] = _dominant_version(ordered[:best["index"]])
    best["version_after"] = _dominant_version(ordered[best["index"]:])
    best["valid"] = True
    return best


def learn_repair_effectiveness(
    ledger_records: List[Dict[str, Any]],
    current_state: Dict[str, Any],
    min_records: int = 30,
) -> Dict[str, Any]:
    """Contextual logistic model: P(repair helps | system state).

    Trains on resolved repair-ledger entries (snapshot_before + verdict)
    and predicts, for the CURRENT system state, how likely each repair
    category is to actually help. Reuses the same pure-Python logistic
    trainer as the vote-weight optimizer — no external deps.
    """
    resolved = [
        r for r in ledger_records
        if r.get("verdict") in ("helped", "hurt")
    ]
    if len(resolved) < min_records:
        return {"valid": False, "error": "insufficient_ledger_data"}

    categories = sorted({
        r.get("category") or r.get("type") or "unknown" for r in resolved
    })
    state_keys = ["overall_wr", "n", "net_ev", "brier"]

    def _feats(state: Dict[str, Any], cat: str) -> List[float]:
        base = []
        for k in state_keys:
            v = state.get(k)
            v = float(v) if isinstance(v, (int, float)) else 0.0
            if k == "n":
                v = math.log1p(v)  # compress sample-size scale
            base.append(v)
        onehot = [1.0 if cat == c else 0.0 for c in categories]
        return base + onehot

    X: List[List[float]] = []
    y: List[float] = []
    for rec in resolved:
        cat = rec.get("category") or rec.get("type") or "unknown"
        X.append([1.0] + _feats(rec.get("snapshot_before") or {}, cat))
        y.append(1.0 if rec["verdict"] == "helped" else 0.0)

    beta = _train_logistic(X, y, [1.0] * len(X),
                           max_iter=1500, lr=0.05, l2=0.02)
    if not beta:
        return {"valid": False, "error": "logistic_train_failed"}

    preds: Dict[str, float] = {}
    for cat in categories:
        vec = [1.0] + _feats(current_state, cat)
        z = sum(b * x for b, x in zip(beta, vec))
        preds[cat] = round(_sigmoid(z), 4)

    return {
        "valid": True,
        "n_records": len(resolved),
        "p_help_by_category": preds,
    }

# ═══════════════════════════════════════════════════════════════════════
#  REPAIR SHOP DIAGNOSIS ENGINE
# ═════════════════════════════════════════════════════════════════════

def repair_shop_diagnosis(
    rows: List[Row],
    drift_alerts: List[Dict[str, Any]],
    config: Dict[str, Any],
    target_wr: float = 0.55,
    disable_wr: float = 0.40,
    min_sample: int = 20,
) -> List[Dict[str, Any]]:
    """One-stop repair shop: diagnoses the system's health and produces
    prioritized, actionable fix recommendations. Each item has:
    • severity: critical / high / medium / low
    • category: what's broken
    • diagnosis: what the data shows
    • action: what to do about it
    • expected_impact: rough estimate of improvement
    """
    repairs: List[Dict[str, Any]] = []
    if not rows:
        return repairs

    n = len(rows)
    if n < min_sample:
        return [{
            "severity": "medium",
            "category": "insufficient_data",
            "diagnosis": (
                f"Only {n} outcome rows in the analysis window — below the "
                f"minimum {min_sample} needed for any other diagnostic to be "
                f"statistically meaningful."
            ),
            "action": (
                f"Wait for the archive to accumulate at least {min_sample} "
                f"resolved outcomes before acting on any Brain advisory. At "
                f"current resolution rates this typically takes a few days."
            ),
            "expected_impact": (
                "Prevents acting on sample noise as if it were a real signal."
            ),
        }]

    overall_wr = sum(r["win"] for r in rows) / n
    buy_rows = [r for r in rows if r["direction"] == "buy"]
    sell_rows = [r for r in rows if r["direction"] == "sell"]
    buy_wr = sum(r["win"] for r in buy_rows) / len(buy_rows) if buy_rows else None
    sell_wr = sum(r["win"] for r in sell_rows) / len(sell_rows) if sell_rows else None

    # ── 1. CRITICAL: Overall WR collapse ──
    
    collapse_floor = target_wr * 0.5
    overall_wins = sum(1 for r in rows if r["win"])
    p_overall_broken = _prob_edge_broken(overall_wins, n, collapse_floor)

    if p_overall_broken > 0.90:
        severity = "critical" if p_overall_broken > 0.97 else "high"
        repairs.append({
            "severity": severity,
            "category": "win_rate_collapse",
            "diagnosis": (
                f"Overall WR {overall_wr:.0%} (n={n}) — posterior "
                f"P(true WR < {collapse_floor:.0%}) = {p_overall_broken:.1%}. "
                f"The strategy has negative edge."
            ),
            "action": (
                "1) STOP all live trading immediately. "
                "2) Raise CONFLUENCE_MIN_ABS_SCORE by +2 to filter weak signals. "
                f"3) Review the last {min(30, n)} trades manually for a systematic error "
                "(bad data, wrong timeframe, API issues)."
            ),
            "expected_impact": "Prevents further losses while diagnosing root cause.",
            "posterior": round(p_overall_broken, 4),
            "scope": {"kind": "global"},
        })
    # ── 2. Directional collapse (sell or buy side broken) ──
    drifted_sell = [d for d in drift_alerts if "sell" in d.get("alert", "") or "down" in d.get("alert", "")]
    drifted_buy = [d for d in drift_alerts if "buy" in d.get("alert", "") or "up" in d.get("alert", "")]

    if sell_wr is not None:
        sell_wins = sum(1 for r in sell_rows if r["win"])
        p_sell_broken = _prob_edge_broken(sell_wins, len(sell_rows), disable_wr)
        if p_sell_broken > 0.90 and len(drifted_sell) >= 2:
            severity = "critical" if p_sell_broken > 0.97 else "high"
            repairs.append({
                "severity": severity,
                "category": "sell_side_collapse",
                "diagnosis": (
                    f"Sell WR {sell_wr:.0%} (n={len(sell_rows)}) with "
                    f"{len(drifted_sell)} sell alerts CUSUM-drifted. "
                    f"Posterior P(true_sell_wr < {disable_wr:.0%}) = {p_sell_broken:.2%}."
                ),
                "action": (
                    f"Disable ALL sell alerts until manual review: "
                    f"{', '.join(d.get('alert', '?') for d in drifted_sell[:5])}. "
                    f"Check if market structure changed (trending up = sells fail)."
                ),
                "expected_impact": f"Removing {len(sell_rows)} losing sell trades lifts overall WR to ~{buy_wr:.0%}." if buy_wr else "Removes systematic losses.",
                "posterior": round(p_sell_broken, 4),
                "scope": {"kind": "direction", "value": "sell"},
            })
    if buy_wr is not None:
        buy_wins = sum(1 for r in buy_rows if r["win"])
        p_buy_broken = _prob_edge_broken(buy_wins, len(buy_rows), disable_wr)
        if p_buy_broken > 0.90 and len(drifted_buy) >= 2:
            severity = "critical" if p_buy_broken > 0.97 else "high"
            repairs.append({
                "severity": severity,
                "category": "buy_side_collapse",
                "diagnosis": (
                    f"Buy WR {buy_wr:.0%} (n={len(buy_rows)}) with {len(drifted_buy)} buy alerts drifted. "
                    f"Posterior P(true_buy_wr < {disable_wr:.0%}) = {p_buy_broken:.2%}."
                ),
                "action": f"Disable drifted buy alerts: {', '.join(d.get('alert', '?') for d in drifted_buy[:5])}.",
                "expected_impact": "Stops bleeding on the buy side.",
                "posterior": round(p_buy_broken, 4),
                "scope": {"kind": "direction", "value": "buy"},
            })
    # ── 3. CUSUM drift freeze ──
    if drift_alerts:
        drifted_names = [d.get("alert", "?") for d in drift_alerts]
        drifted_keys = sorted({d.get("alert") for d in drift_alerts if d.get("alert")})
        repairs.append({
            "severity": "high",
            "category": "cusum_drift",
            "diagnosis": (
                f"{len(drift_alerts)} alert(s) show CUSUM edge decay: "
                f"{', '.join(drifted_names[:6])}."
            ),
            "action": (
                "Config patches FROZEN for these alerts. "
                "Manual review required before re-enabling auto-tuning. "
                "Check if a recent config change or market regime shift caused the decay."
            ),
            "expected_impact": "Prevents auto-tuning from optimizing a broken signal.",
            "scope": {"kind": "alert_keys", "value": drifted_keys},
        })
    # ── 4. Gate threshold too low ──
    current_threshold = config.get("CONFLUENCE_MIN_ABS_SCORE", 18.0)
    rec = recommend_threshold(rows, target_winrate=target_wr, min_sample=min_sample)

    if rec.get("valid") and rec["recommended"] > current_threshold + 0.5:
        repairs.append({
            "severity": "high",
            "category": "threshold_too_low",
            "diagnosis": (
                f"Current gate Score≥{current_threshold:.1f} lets through trades with "
                f"{overall_wr:.0%} WR. Raising to {rec['recommended']:.1f} would achieve "
                f"{rec['rec_wr']:.0%} WR on {rec['rec_n']} samples."
            ),
            "action": (
                f"Raise CONFLUENCE_MIN_ABS_SCORE from {current_threshold:.1f} to "
                f"{rec['recommended']:.1f}. This drops {rec['dropped']} weak trades "
                f"({rec['dropped_pct']:.0%})."
            ),
            "expected_impact": f"WR improvement: {overall_wr:.0%} → {rec['rec_wr']:.0%} (+{rec['rec_wr']-overall_wr:.0%}).",
            # The repair's effect is confined to trades in the newly-
            # blocked band — those are exactly the ones it says are bad.
            "scope": {"kind": "score_band",
                      "value": [float(current_threshold), float(rec["recommended"])]},
        })

    # ── 5. Brier / calibration check ──
    brier, _ = brier_score_and_calibration(rows)
    if brier >= 0.20:
        repairs.append({
            "severity": "medium",
            "category": "miscalibration",
            "diagnosis": f"Brier score {brier:.3f} ≥ 0.20 — predicted probabilities are miscalibrated.",
            "action": "Review confluence weight distribution. Consider running the weight optimizer with walk-forward validation.",
            "expected_impact": "Better calibrated scores → more reliable threshold gating.",
            "scope": {"kind": "global"},
        })

    # ── 6. EV / Kelly check ──
    net_ev, half_kelly, _ = ev_and_kelly_for(rows)
    p_ev_negative = _prob_ev_negative(rows)
    # The bootstrap behind _prob_ev_negative needs n>=60; below that it always
    # returns 0.5 (neutral) and this check would never fire. Fall back to the
    # point estimate for that window so an early bad EV isn't silently missed.
    bootstrap_ready = n >= 60
    ev_triggered = (p_ev_negative > 0.80) if bootstrap_ready else (net_ev <= 0)
    if ev_triggered:
        severity = "critical" if (bootstrap_ready and p_ev_negative > 0.95) else "high"
        diag_stat = (
            f"posterior P(true EV <= 0) = {p_ev_negative:.1%}."
            if bootstrap_ready else
            "(n<60 — point estimate; too small for the posterior test)."
        )
        repairs.append({
            "severity": severity,
            "category": "negative_ev",
            "diagnosis": (
                f"Net EV {net_ev:+.3f}%/trade after fees/slippage (n={n}) — "
                f"{diag_stat} Strategy is unprofitable."
            ),
            "action": (
                "1) Increase CONFLUENCE_MIN_ABS_SCORE to filter weak signals. "
                "2) Check if fee/slippage assumptions (0.06% + 0.03% per side) match your exchange. "
                "3) Consider widening OUTCOME_FAVORABLE_MOVE_PCT if TP is too tight."
            ),
            "expected_impact": "Positive EV is the minimum requirement for a viable strategy.",
            "posterior": round(p_ev_negative, 4),
            "scope": {"kind": "global"},
        })

    # ── 7. Sample size warning ──
    if n < 200:
        repairs.append({
            "severity": "medium",
            "category": "insufficient_data",
            "diagnosis": f"Only {n} samples in the analysis window. Statistical power is limited.",
            "action": (
                "Widen BRAIN_ANALYSIS_WINDOW_DAYS or lower BRAIN_REPORT_STREAM_SAMPLE "
                "to accumulate more data before trusting optimizer outputs. "
                "Treat all suggestions as provisional until n≥300."
            ),
            "expected_impact": "Prevents overfitting to small samples.",
            "scope": {"kind": "global"},
        })

    # ── 8. ML: Automated root-cause attribution ──
    rc = diagnose_root_cause(
        rows, target_wr=target_wr,
        min_segment=max(15, min_sample // 2),
    )
    if rc.get("valid") and rc["segments"]:
        seg = rc["segments"][0]
        repairs.append({
            "severity": "high",
            "category": "root_cause",
            "diagnosis": (
                f"Losses concentrate where `{seg['rule']}`: that segment wins "
                f"only {seg['segment_wr']:.0%} (n={seg['segment_n']}) vs overall "
                f"{rc['overall_wr']:.0%}, covering {seg['coverage']:.0%} of trades."
            ),
            "action": (
                f"Add a guard blocking trades when {seg['rule']}, or reduce the "
                f"weight of the offending vote/feature."
            ),
            "expected_impact": (
                f"Removing this segment lifts overall WR by "
                f"~{seg['lift_vs_overall']:.0%}."
            ),
            "p_value": seg["p_value"],
            # Scope: the exact subset the segment rule selects — the only
            # rows this repair can possibly affect.
            "scope": {
                "kind": "segment",
                "value": {
                    "feature": seg["feature"],
                    "op": seg["op"],
                    "threshold": seg["threshold"],
                },
            },
        })
    # ── 9. ML: Feature / regime drift (PSI) ──
    drift = detect_feature_drift(rows)
    if drift.get("valid") and drift["drifted_features"]:
        top = ", ".join(
            f"{d['feature']} (PSI {d['psi']:.2f})"
            for d in drift["drifted_features"][:4]
        )
        repairs.append({
            "severity": "medium",
            "category": "regime_shift",
            "diagnosis": (
                f"Feature distributions shifted recently: {top}. The market "
                f"regime this config was tuned on has changed."
            ),
            "action": (
                "Re-run the weight optimizer on recent-only data, and treat "
                "older-window recommendations as stale until the regime settles."
            ),
            "expected_impact": (
                "Re-tunes gates to the current regime instead of a stale one."
            ),
            "scope": {"kind": "global"},
        })

    # ── 10. ML: Change-point + config attribution ──
    cp = find_wr_change_point(rows)
    if cp.get("valid"):
        direction = "dropped" if cp["delta"] < 0 else "improved"
        version_note = ""
        if (cp.get("version_before") and cp.get("version_after")
                and cp["version_before"] != cp["version_after"]):
            version_note = (
                f" Config version changed at the break: "
                f"{cp['version_before']} → {cp['version_after']}."
            )
        repairs.append({
            "severity": "high" if cp["delta"] < 0 else "low",
            "category": "config_regression_pinpoint",
            "diagnosis": (
                f"Win rate {direction} from {cp['wr_before']:.0%} to "
                f"{cp['wr_after']:.0%} (Δ{cp['delta']:+.0%}) at "
                f"{format_ist_time(cp['change_ts'])}.{version_note}"
            ),
            "action": (
                "Review what changed at that timestamp. If a config patch "
                "landed then, consider reverting it."
                if cp["delta"] < 0 else
                "Note the improvement and the change that caused it."
            ),
            "expected_impact": (
                "Pinpoints exactly when the edge broke, instead of a vague "
                "'recently' window."
            ),
            "p_value": cp["p_value"],
            "scope": (
                {"kind": "config_version", "value": cp["version_after"]}
                if cp.get("version_after") else {"kind": "global"}
            ),
        })

    # Sort by severity
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    repairs.sort(key=lambda x: severity_order.get(x["severity"], 4))
    shop_max = getattr(cfg, "BRAIN_REPAIR_SHOP_MAX", 3)
    if len(repairs) > shop_max:
        repairs = repairs[:shop_max]
    return repairs

# ══════════════════════════════════════════════════════════════════════
#  CALIBRATION GATE (live): per-alert conf_pct → observed WR curve
# ══════════════════════════════════════════════════════════════════════

def build_calibration_curves(
    rows: List[Row],
    bucket_pct: float = 5.0,
    min_sample: int = 15,
) -> Dict[str, Any]:
    """Per-alert-key calibration: bucketed confluence % → observed win rate.

    A raw confluence score is a weighted vote total, not a probability.
    This maps what the score DISPLAYS (conf_pct, treated as the implied
    probability claim /100) to what actually HAPPENED, per alert key, so
    the live gate can filter on calibrated probability instead of face
    value. ECE = standard expected calibration error over buckets.
    """
    by_ak: Dict[str, List[Row]] = defaultdict(list)
    for r in rows:
        by_ak[r["alert_key"]].append(r)

    curves: Dict[str, Any] = {}
    for ak, ak_rows in by_ak.items():
        if len(ak_rows) < min_sample:
            continue
        buckets: Dict[int, List[Row]] = defaultdict(list)
        for r in ak_rows:
            buckets[int(r["conf_pct"] // bucket_pct)].append(r)
        out = []
        for b in sorted(buckets):
            chunk = buckets[b]
            n = len(chunk)
            wins = sum(r["win"] for r in chunk)
            wr = wins / n
            lo, hi, _ = wilson_ci(wins, n)
            pred = statistics.mean(r["conf_pct"] for r in chunk) / 100.0
            out.append({
                "lo": b * bucket_pct, "hi": (b + 1) * bucket_pct,
                "predicted": round(pred, 4), "observed": round(wr, 4),
                "n": n, "trusted": n >= min_sample,
                "wilson_lo": round(lo, 4), "wilson_hi": round(hi, 4),
            })
        total = len(ak_rows)
        ece = sum((bk["n"] / total) * abs(bk["observed"] - bk["predicted"]) for bk in out)
        curves[ak] = {"buckets": out, "ece": round(ece, 4), "n": total}

    ece_values = [c["ece"] for c in curves.values()]
    return {
        "curves": curves,
        "ece_mean": round(statistics.fmean(ece_values), 4) if ece_values else None,
        "built_at": int(time.time()),
    }

def calibration_gate_decision(
    curve: Dict[str, Any],
    conf_pct: float,
    target_wr: float,
    min_sample: int = 15,
    slack: float = 0.05,
) -> Tuple[bool, Optional[float], str]:
    """(pass, calibrated_wr, reason) for one alert at one conf_pct.

    Blocks only when the trusted bucket is CONFIDENTLY below target
    (observed < target−slack AND wilson_hi < target) — a calibration
    gate must never block on its own uncertainty, only on evidence of
    miscalibration. Fail-open on thin/missing buckets.
    """
    buckets = curve.get("buckets", [])
    if not buckets:
        return True, None, "no_curve"
    chosen = None
    for bk in buckets:
        if bk["lo"] <= conf_pct < bk["hi"]:
            chosen = bk
            break
    if chosen is None:  # conf_pct outside covered range → nearest bucket
        chosen = min(buckets, key=lambda bk: min(abs(conf_pct - bk["lo"]), abs(conf_pct - bk["hi"])))
    if not chosen.get("trusted") or chosen["n"] < min_sample:
        return True, chosen["observed"], "thin_bucket_fail_open"
    cal_wr = chosen["observed"]
    if cal_wr < target_wr - slack and chosen["wilson_hi"] < target_wr:
        return False, cal_wr, (
            f"calibrated WR {cal_wr:.0%} "
            f"[{chosen['wilson_lo']:.0%}-{chosen['wilson_hi']:.0%}] below "
            f"{target_wr:.0%} target at conf {conf_pct:.0f}%"
        )
    return True, cal_wr, "ok"


# ══════════════════════════════════════════════════════════════════════
#  PORTFOLIO HEAT — hard exposure caps, independent of confluence math
# ══════════════════════════════════════════════════════════════════════

def portfolio_heat_check(
    open_positions: List[Dict[str, Any]],
    new_pair: str,
    new_direction: str,
    max_concurrent: int = 6,
    max_net_directional: int = 4,
    max_same_direction_pct: float = 1.0,
) -> Dict[str, Any]:
    """Hard veto on total book exposure.

    ClusterContext penalizes a same-run correlated herd at the SCORE
    level — soft, and it still lets one alert per pair through. Five
    pairs firing one at a time over three hours, all effectively
    long-BTC-beta, pass every pair-level gate. This looks at the book
    as a whole: how many positions are open, how lopsided they are,
    and whether the new one makes it worse.

    open_positions: [{"pair": ..., "direction": "buy"|"sell"}, ...]
    Never raises — a risk gate that crashes must not take dispatch down.
    """
    stats: Dict[str, Any] = {"open_count": len(open_positions)}
    try:
        open_pairs = {p.get("pair") for p in open_positions}
        longs = sum(1 for p in open_positions if p.get("direction") == "buy")
        sells = sum(1 for p in open_positions if p.get("direction") == "sell")
        stats.update(longs=longs, sells=sells, net=longs - sells)

        if new_pair in open_pairs:
            return {"blocked": True, "reason": f"{new_pair} already has an open position", **stats}
        if len(open_positions) >= max_concurrent:
            return {"blocked": True,
                    "reason": f"max concurrent positions ({max_concurrent}) reached", **stats}

        new_longs = longs + (1 if new_direction == "buy" else 0)
        new_sells = sells + (1 if new_direction == "sell" else 0)
        net = new_longs - new_sells
        if abs(net) > max_net_directional:
            return {"blocked": True,
                    "reason": f"net directional exposure would be {net:+d} (limit ±{max_net_directional})",
                    **stats}

        same = new_longs if new_direction == "buy" else new_sells
        total = len(open_positions) + 1
        if max_same_direction_pct < 1.0 and same / total > max_same_direction_pct:
            return {"blocked": True,
                    "reason": f"{same}/{total} positions same direction (> {max_same_direction_pct:.0%})",
                    **stats}
        return {"blocked": False, "reason": "ok", **stats}
    except Exception as e:
        return {"blocked": False, "reason": f"gate error (fail-open): {e}", **stats}


# ══════════════════════════════════════════════════════════════════════
#  KILL SWITCH — fast-failure halt (streak / rolling drawdown)
# ══════════════════════════════════════════════════════════════════════

class KillSwitch:
    """Hard halt on live bleed: N consecutive losses or X% cost-adjusted
    drawdown inside a rolling window.

    CUSUM watches per-alert-key edge decay and needs samples to
    accumulate; a correlated wipeout across ten keys in two hours trips
    nothing per-key but kills the book. This is the circuit breaker for
    STRATEGY risk (APICircuitBreaker covers API risk).

    Pure function of outcome rows — no hidden state. The brain arms a
    Redis flag on trip; dispatch polls it. For sub-report latency wire
    evaluate() into the outcome-resolution path too (see integration
    notes). PnL convention mirrors ev_and_kelly_for exactly so the
    drawdown number agrees with the EV numbers in the report.
    """

    def __init__(
        self,
        max_consecutive_losses: int = 6,
        max_drawdown_pct: float = 3.0,
        lookback_hours: int = 24,
        fee_pct: float = 0.0006,
        slippage_pct: float = 0.0003,
    ):
        self.max_consecutive_losses = max_consecutive_losses
        self.max_drawdown_pct = max_drawdown_pct
        self.lookback_hours = lookback_hours
        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct

    def evaluate(self, rows: List[Row], now_ts: Optional[float] = None) -> Dict[str, Any]:
        if now_ts is None:
            now_ts = time.time()
        result: Dict[str, Any] = {
            "tripped": False, "reason": None,
            "consecutive_losses": 0, "drawdown_pct": 0.0,
            "pnl_pct": 0.0, "n_window": 0,
        }
        if not rows:
            return result

        ordered = sorted(
            (r for r in rows if r.get("entry_ts", 0) > 0),
            key=lambda r: r["entry_ts"],
        )
        cutoff = now_ts - self.lookback_hours * 3600

        # Losing streak, counted from the tail, freshness-gated to the
        # lookback window so an old pre-downtime streak can't trip it.
        streak = 0
        for r in reversed(ordered):
            if r["entry_ts"] < cutoff or r["win"]:
                break
            streak += 1
        result["consecutive_losses"] = streak

        # Rolling cost-adjusted PnL (same convention as ev_and_kelly_for).
        total_cost = (self.fee_pct * 2) + (self.slippage_pct * 2)
        window = [r for r in ordered if r["entry_ts"] >= cutoff]
        pnl = 0.0
        for r in window:
            mag = abs(r.get("pct_move", 0.0))
            pnl += (mag - total_cost) if r["win"] else -(mag + total_cost)
        result.update(
            pnl_pct=round(pnl, 4),
            drawdown_pct=round(-pnl, 4) if pnl < 0 else 0.0,
            n_window=len(window),
        )

        reasons = []
        if streak >= self.max_consecutive_losses:
            reasons.append(f"{streak} consecutive losses (limit {self.max_consecutive_losses})")
        if pnl <= -self.max_drawdown_pct:
            reasons.append(
                f"{pnl:+.2f}% PnL over last {self.lookback_hours}h "
                f"(limit -{self.max_drawdown_pct}%)"
            )
        if reasons:
            result["tripped"] = True
            result["reason"] = " and ".join(reasons)
        return result


# ══════════════════════════════════════════════════════════════════════
#  FILL RECONCILIATION — assumed vs realized execution cost
# ═══════════════════════════════════════════════════════════════════���══

def fill_reconciliation(
    rows: List[Row],
    assumed_fee_pct: float = 0.0006,
    assumed_slippage_pct: float = 0.0003,
    rr_target: float = 2.0,
    stop_pct: float = 0.5,
    min_sample: int = 10,
) -> Dict[str, Any]:
    """Compare ASSUMED execution cost (the fixed fee+slippage baked into
    every EV/Kelly number) against REALIZED cost.

    Two evidence tiers:
    1. MEASURED — rows carrying signal_price/fill_price. Direction-aware
       slippage: a buy filling above signal, or a sell below, is paying
       up. This is the ground truth once the outcome writer records fills.
    2. ESTIMATED — no fill data yet: realized |pct_move| on wins vs the
       theoretical TP distance (rr_target × stop_pct). A systematic
       shortfall is execution leakage — labeled an estimate because
       outcome resolution is not a fill feed.

    Liquidity varies a lot across a ~30-pair universe, so the per-pair
    breakdown is first-class output, not an afterthought.
    """
    measured = [r for r in rows if r.get("signal_price") and r.get("fill_price")]

    # ── Tier 1: measured fills ─────────────────────────────────────
    if len(measured) >= min_sample:
        slips: List[float] = []
        per_pair: Dict[str, List[float]] = defaultdict(list)
        fees: List[float] = []
        for r in measured:
            sig, fill = r["signal_price"], r["fill_price"]
            if sig <= 0:
                continue
            slip = (fill - sig) / sig if r["direction"] == "buy" else (sig - fill) / sig
            slips.append(slip)
            per_pair[r["pair"]].append(slip)
            if r.get("fees_paid_pct") is not None:
                fees.append(r["fees_paid_pct"])
        if len(slips) >= min_sample:
            mean_slip = statistics.fmean(slips)
            realized = max(0.0, mean_slip)
            pair_rows = []
            for pair, ps in per_pair.items():
                if len(ps) < max(3, min_sample // 2):
                    continue
                pm = statistics.fmean(ps)
                pair_rows.append({
                    "pair": pair, "n": len(ps),
                    "realized_slippage_per_side": round(max(0.0, pm), 6),
                    "gap_bps": round((pm - assumed_slippage_pct) * 10000, 1),
                })
            pair_rows.sort(key=lambda x: -x["gap_bps"])
            result: Dict[str, Any] = {
                "valid": True, "measured": True, "n": len(slips),
                "realized_slippage_per_side": round(realized, 6),
                "assumed_slippage_per_side": assumed_slippage_pct,
                "gap_bps": round((mean_slip - assumed_slippage_pct) * 10000, 2),
                # two sides per round trip
                "ev_overstated_pct_per_trade": round(2 * max(0.0, mean_slip - assumed_slippage_pct), 6),
                "per_pair": pair_rows,
            }
            if fees:
                result["realized_fee_pct"] = round(statistics.fmean(fees), 6)
                result["fee_gap_bps"] = round((result["realized_fee_pct"] - assumed_fee_pct) * 10000, 2)
            return result

    # ── Tier 2: estimated from win-move shortfall ──────────────────
    wins = [abs(r.get("pct_move", 0.0)) for r in rows if r["win"]]
    if len(wins) < min_sample:
        return {"valid": False, "measured": False, "error": "insufficient_data", "n": len(wins)}
    realized_pp = statistics.fmean(wins)
    expected_pp = rr_target * stop_pct          # theoretical TP distance, same units
    shortfall_pp = max(0.0, expected_pp - realized_pp)
    implied_frac = (shortfall_pp / 2.0) / 100.0  # split across entry+exit, pp → fraction

    per_pair = defaultdict(list)
    for r in rows:
        if r["win"]:
            per_pair[r["pair"]].append(abs(r.get("pct_move", 0.0)))
    pair_rows = []
    for pair, moves in per_pair.items():
        if len(moves) < max(3, min_sample // 2):
            continue
        implied_p = max(0.0, (expected_pp - statistics.fmean(moves)) / 2.0) / 100.0
        pair_rows.append({
            "pair": pair, "n": len(moves),
            "realized_slippage_per_side": round(implied_p, 6),
            "gap_bps": round((implied_p - assumed_slippage_pct) * 10000, 1),
        })
    pair_rows.sort(key=lambda x: -x["gap_bps"])

    return {
        "valid": True, "measured": False, "n": len(wins),
        "realized_move_pct_win": round(realized_pp, 4),
        "expected_move_pct_win": round(expected_pp, 4),
        "realized_slippage_per_side": round(implied_frac, 6),
        "assumed_slippage_per_side": assumed_slippage_pct,
        "gap_bps": round((implied_frac - assumed_slippage_pct) * 10000, 2),
        "ev_overstated_pct_per_trade": round(2 * max(0.0, implied_frac - assumed_slippage_pct), 6),
        "per_pair": pair_rows,
        "note": "estimated from win-move shortfall; wire fill prices into the "
                "outcome writer for the measured tier",
    }