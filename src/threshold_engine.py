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
    """Recency-weighted win rate + a weighted Wilson-CI band. ... Returns
    (weighted_wr, n_eff, wilson_lo, wilson_hi), where n_eff is Kish's
    effective sample size (sum(w)^2 / sum(w^2), always <= len(rows))..."""
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
    weighted_wr = sum_ww / sum_w
    n_eff = (sum_w ** 2) / sum_w2 if sum_w2 > 0 else 0.0
    lo, hi, _ = wilson_ci(round(weighted_wr * n_eff), max(1, round(n_eff)))
    return weighted_wr, n_eff, lo, hi

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
 
def walk_forward_split(rows: List[Row], train_frac: float = 0.67) -> Tuple[List[Row], List[Row]]:
    """Chronological split by entry_ts (not random) — a threshold has to
    survive time moving forward, not just a random resample of the same
    period. Rows missing entry_ts (0) sort first, into the train side."""
    ordered = sorted(rows, key=lambda r: r.get("entry_ts", 0))
    split_idx = int(len(ordered) * train_frac)
    return ordered[:split_idx], ordered[split_idx:]

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

# ══════════════════════���════════════════════════════════════════════════
#  NEW: Cost-Aware EV + Kelly Sizing  (Recommended.txt §5)
# ═══════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════
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


# ══════════════════════════���════════════════════════════════════════════
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
# ═══════════════════════��═══════════════════════════════════════════════

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

def optimize_vote_weights(
    rows: List[Row],
    current_weights: Dict[str, float],
    min_sample: int = 100,
    max_iter: int = 2000,
    lr: float = 0.05,
    l2: float = 0.01,
) -> Dict[str, Any]:
    """Data-driven CONFLUENCE_WEIGHTS via gradient-descent logistic regression."""
    vote_names = sorted(current_weights.keys())
    X: List[List[float]] = []
    y: List[float] = []

    for r in rows:
        votes = r.get("votes")
        if not votes or not isinstance(votes, dict):
            continue
        vec = [1.0] + [1.0 if votes.get(vn) else 0.0 for vn in vote_names]
        X.append(vec)
        y.append(1.0 if r["win"] else 0.0)

    n = len(X)
    if n < min_sample:
        return {"valid": False, "error": f"insufficient_data: {n} < {min_sample}"}

    win_rate = sum(y) / n
    beta = [0.0] * (len(vote_names) + 1)
    beta[0] = math.log(win_rate / (1 - win_rate)) if 0 < win_rate < 1 else 0.0

    for iteration in range(max_iter):
        grad = [0.0] * len(beta)
        for i in range(n):
            z = sum(beta[j] * X[i][j] for j in range(len(beta)))
            p = _sigmoid(z)
            error = p - y[i]
            for j in range(len(beta)):
                grad[j] += error * X[i][j]
        # Normalize by n, add L2 regularization
        for j in range(len(beta)):
            grad[j] = grad[j] / n + l2 * beta[j]
        # Use cosine-annealed learning rate (0.05 → 0.001)
        step = lr * (0.5 * (1 + math.cos(math.pi * iteration / max_iter)))
        for j in range(len(beta)):
            beta[j] -= step * grad[j]

    intercept = beta[0]
    coeffs = beta[1:]
    positive_coeffs = [max(0.0, c) for c in coeffs]
    total_pos = sum(positive_coeffs)

    suggested: Dict[str, float] = {}
    negative_votes: List[Tuple[str, float]] = []
    for idx, vn in enumerate(vote_names):
        c = coeffs[idx]
        if c < -0.05:
            negative_votes.append((vn, round(c, 4)))
        if total_pos > 0 and positive_coeffs[idx] > 0:
            raw = 3.0 * (positive_coeffs[idx] / (total_pos / len(vote_names)))
            suggested[vn] = round(min(5.0, max(0.5, raw)), 2)
        else:
            suggested[vn] = 0.0

    return {
        "valid": True,
        "n_samples": n,
        "intercept": round(intercept, 4),
        "current_weights": dict(current_weights),
        "suggested_weights": suggested,
        "negative_votes": negative_votes,
    }


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

            if len(both) < min_sample or len(only_v1) < min_sample:
                continue

            wr_both = sum(r["win"] for r in both) / len(both)
            wr_only_v1 = sum(r["win"] for r in only_v1) / len(only_v1)
            wr_only_v2 = sum(r["win"] for r in only_v2) / len(only_v2) if only_v2 else 0.0
            wr_neither = sum(r["win"] for r in neither) / len(neither) if neither else 0.0

            synergy = wr_both - max(wr_only_v1, wr_only_v2, wr_neither)
            if synergy > 0.10:
                interactions.append({
                    "pair": (v1, v2),
                    "type": "synergy",
                    "delta": round(synergy, 4),
                    "wr_both": round(wr_both, 4),
                    "wr_only_v1": round(wr_only_v1, 4),
                    "wr_only_v2": round(wr_only_v2, 4),
                    "n_both": len(both),
                })

            poison = wr_only_v1 - wr_both
            if poison > 0.15 and len(both) >= min_sample:
                interactions.append({
                    "pair": (v1, v2),
                    "type": "poison",
                    "delta": round(-poison, 4),
                    "wr_both": round(wr_both, 4),
                    "wr_only_v1": round(wr_only_v1, 4),
                    "n_both": len(both),
                    "note": f"{v2} poisons {v1}",
                })

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

# ════════════��══════════════════════════════════════════════════════════
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

def hash_config_state(weights: Dict[str, float], threshold: float, min_pct: float) -> str:
    payload = json.dumps({"w": weights, "t": threshold, "p": min_pct}, sort_keys=True)
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
