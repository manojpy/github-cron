#!/usr/bin/env python3
"""
threshold_engine.py — shared, pure-function analysis library for confluence
threshold recommendations.

No I/O in this file: no Redis, no argparse, no print(), no sys.exit(). Both
analyze_confluence_thresholds.py (the CLI report) and brain.py (the Telegram
bot report) import this module and call recommend_threshold() (and the
individual pieces below it) so they always compute the SAME answer from the
SAME data. Previously these two lived as separately-maintained copies and
had already drifted (raw-WR vs Wilson bounds, percentage space vs raw score
space, signed vs unsigned pct_move) — this module exists to make that class
of bug structurally impossible going forward, not just to fix the specific
instances of it found so far.

Row shape expected by every function here (one dict per resolved outcome,
matching what load_rows() / BrainEngine._parse_rows() produce from
outcome_log_stream / shadow_log_stream):
    {
        "pair": str,
        "alert_key": str,
        "direction": "buy" | "sell",
        "score": float,
        "total": float,
        "win": bool,
        "pct_move": float,   # SIGNED by price direction, not by trade
                              # outcome — a winning sell has pct_move < 0.
                              # Use favourable_move(row) for magnitude.
        "entry_ts": int,
        "votes": dict[str, bool] | None,
    }
"""
from __future__ import annotations

import math
import time
import random
import statistics
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

Row = Dict[str, Any]
CapRow = Tuple[float, int, float, float]  # (cap, n, wr, wilson_lower_bound)


# ────────────────────────────────────────────────────────────────────────
# Statistical primitives
# ────────────────────────────────────────────────────────────────────────

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

# ────────────────────────────────────────────────────────────────────────
# Bucket-level analysis
# ────────────────────────────────────────────────────────────────────────
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

# ────────────────────────────────────────────────────────────────────────
# Cumulative cap analysis
# ────────────────────────────────────────────────────────────────────────
def build_caps_data(rows: List[Row], min_sample: int = 20) -> Tuple[List[float], List[CapRow]]:
    """candidate_caps: every distinct observed score (ascending).
    caps_data: (cap, n, wr, wilson_lower_bound) for caps whose cumulative
    subset (score >= cap) has at least min_sample rows — this is the list
    knee-point / target-floor / EV searches all operate on."""
    candidate_caps = sorted(set(r["score"] for r in rows))
    caps_data: List[CapRow] = []
    for cap in candidate_caps:
        subset = [r for r in rows if r["score"] >= cap]
        if len(subset) < min_sample:
            continue
        wins_count = sum(r["win"] for r in subset)
        wr = wins_count / len(subset)
        lo, _hi, _p = wilson_ci(wins_count, len(subset))
        caps_data.append((cap, len(subset), wr, lo))
    return candidate_caps, caps_data


# ────────────────────────────────────────────────────────────────────────
# Walk-forward validation — guards against a threshold that only looks
# good because it was chosen FROM the data being used to judge it.
# ────────────────────────────────────────────────────────────────────────

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


# ────────────────────────────────────────────────────────────────────────
# Per-slice breakdowns
# ────────────────────────────────────────────────────────────────────────

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


# ────────────────────────────────────────────────────────────────────────
# The unified recommendation — the single source of truth both consumers
# call. Pure function: takes rows + params, returns a structured dict.
# Callers format this however they need (CLI report vs Telegram message);
# nothing here prints or does I/O.
# ────────────────────────────────────────────────────────────────────────

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
