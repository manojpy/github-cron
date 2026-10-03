"""Walk-forward / Monte-Carlo validation, drift detectors (CUSUM, stability gate) and OOD checks.

Split out of the original module without logic changes; the original
module remains as a facade re-exporting every name."""
from __future__ import annotations
import math
import random
import statistics
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
from bot_config import cfg
from threshold_analysis import (
    recommend_threshold,
)
from threshold_stats import (
    Row,
    _flatten_row_features,
    _percentile,
    ev_and_kelly_for,
    population_stability_index,
    row_net_pnl_pct,
    two_proportion_p_value,
    walk_forward_split,
    wilson_ci,
)

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
    oos_ev_list: List[float] = []

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
        # ── NEW: track EV alongside WR ──
        ev_val, _hk, _wr = ev_and_kelly_for(holdout_subset)
        oos_ev_list.append(ev_val)

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
        "robustness_score": mean_wr / max(std_wr, 0.01),
        # ── NEW: EV distribution ──
        "oos_ev_mean": statistics.fmean(oos_ev_list) if oos_ev_list else 0.0,
        "oos_ev_p5": (
            sorted(oos_ev_list)[max(0, min(len(oos_ev_list) - 1, round(0.05 * (len(oos_ev_list) - 1))))]
            if oos_ev_list else 0.0
        ),
        "p_ev_positive": (
            sum(1 for e in oos_ev_list if e > 0) / len(oos_ev_list)
            if oos_ev_list else 0.0
        ),
    }

def rolling_walk_forward(
    rows: List[Row],
    n_folds: int = 5,
    train_frac: float = 0.60,
    min_sample: int = 20,
    target_winrate: float = 0.55,
    lookahead_sec: Optional[int] = None,
    embargo_sec: int = 900,
) -> Dict[str, Any]:
    """Multi-fold chronological walk-forward.

    Splits the timeline into n_folds sequential windows.
    For each fold, trains on the preceding data, tests on the fold.
    Aggregates OOS performance across all folds.

    ── Purge + embargo (roadmap item: purged/embargoed OOS) ──
    Every fold's cut gets the same discipline walk_forward_split() applies
    to its single split: purge drops train rows whose outcome window
    (entry_ts + lookahead_sec) spills past the cut, so a train label isn't
    partly determined by prices the test fold is about to see; embargo
    drops the first embargo_sec of the test fold so it isn't still
    statistically coupled to the train tail. Previously only the
    single-split helper had this — rolling_walk_forward trained on
    ordered[:train_end] and tested on ordered[test_start:] with zero gap
    at every one of its n_folds cuts, which is the same leakage bug in
    n_folds places instead of one. This function's p_ev_positive feeds a
    hard ML-eligibility gate in brain_enhanced.py's _action_gate_check(),
    so a leaky estimate here was a real (not just diagnostic) risk.
    """
    if lookahead_sec is None:
        lookahead_sec = (
            max(0, int(getattr(cfg, "OUTCOME_FILL_DELAY_CANDLES", 1)))
            + int(cfg.OUTCOME_LOOKAHEAD_CANDLES)
            + 1
        ) * 900

    ordered = sorted(rows, key=lambda r: r.get("entry_ts", 0))
    n = len(ordered)
    if n < min_sample * (n_folds + 1):
        return {"valid": False, "error": "insufficient_data", "n": n}

    fold_size = n // (n_folds + 1)

    oos_wr_list: List[float] = []
    oos_ev_list: List[float] = []
    oos_n_list: List[int] = []
    thresholds: List[float] = []

    for fold_idx in range(n_folds):
        train_end = fold_size * (fold_idx + 1)
        test_start = train_end
        test_end = min(test_start + fold_size, n)
        if test_start >= n:
            continue

        cut_ts = ordered[test_start].get("entry_ts", 0)

        # Purge: drop train rows whose outcome window spills past this
        # fold's cut (same rule as walk_forward_split()).
        train_rows = [
            r for r in ordered[:train_end]
            if r.get("entry_ts", 0) + lookahead_sec < cut_ts - embargo_sec
        ]
        # Embargo: skip the first embargo_sec of this fold's test window.
        test_rows = [
            r for r in ordered[test_start:test_end]
            if r.get("entry_ts", 0) >= cut_ts + embargo_sec
        ]

        if len(train_rows) < min_sample * 2 or len(test_rows) < min_sample:
            continue
        train_result = recommend_threshold(
            train_rows, target_winrate=target_winrate, min_sample=min_sample
        )
        if not train_result.get("valid"):
            continue

        threshold = train_result["recommended"]
        thresholds.append(threshold)

        test_subset = [r for r in test_rows if r["score"] >= threshold]
        if len(test_subset) < 5:
            continue

        wins = sum(r["win"] for r in test_subset)
        oos_wr_list.append(wins / len(test_subset))
        oos_n_list.append(len(test_subset))

        ev, _, _ = ev_and_kelly_for(test_subset)
        oos_ev_list.append(ev)

    if len(oos_wr_list) < 3:
        return {
            "valid": False,
            "error": "too_few_valid_folds",
            "n_folds": len(oos_wr_list),
        }

    return {
        "valid": True,
        "n_folds": len(oos_wr_list),
        "oos_wr_mean": statistics.fmean(oos_wr_list),
        "oos_wr_std": statistics.pstdev(oos_wr_list),
        "oos_ev_mean": statistics.fmean(oos_ev_list),
        "oos_ev_std": statistics.pstdev(oos_ev_list),
        "oos_ev_p5": min(oos_ev_list), 
        "positive_fold_rate": sum(1 for e in oos_ev_list if e > 0) / len(oos_ev_list),
        "p_ev_positive": sum(1 for e in oos_ev_list if e > 0) / len(oos_ev_list),  # alias for back-compat
        "oos_total_n": sum(oos_n_list),
        "threshold_mean": statistics.fmean(thresholds),
        "threshold_std": statistics.pstdev(thresholds) if len(thresholds) > 1 else 0.0,
        "stability_score": (
            statistics.fmean(oos_ev_list) / max(statistics.pstdev(oos_ev_list), 0.01)
        ),
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

def is_vote_pattern_ood(
    rows: List[Row],
    current_votes: Dict[str, bool],
    alert_key: str,
) -> Tuple[bool, Dict[str, Any]]:
    """Reject if vote count is outside historical 5th-95th percentile.
    Returns (is_ood, detail_dict)."""
    historical_counts: List[float] = []
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
        return max(counts, key=lambda k: counts[k]) if counts else None

    best["version_before"] = _dominant_version(ordered[:best["index"]])
    best["version_after"] = _dominant_version(ordered[best["index"]:])
    best["valid"] = True
    return best

def _holdout_ev_check(rows: List[Row], fee_pct: float = 0.0006, slippage_pct: float = 0.0003) -> Optional[Dict[str, float]]:
    """Net EV and P(true mean P&L > 0) for a SMALL holdout, without the block
    bootstrap (which needs >= 60 rows and so can never confirm a 20-59 row
    holdout). Uses the per-trade net P&L (same bracket model as everywhere
    else) and a normal approximation on the mean: p = Phi(mean / (sd/sqrt(n))).
    Deliberately simple and iid; returns None when it cannot be computed."""
    n = len(rows)
    if n < 2:
        return None
    total_cost = (fee_pct * 2 + slippage_pct * 2) * 100
    pnls = [row_net_pnl_pct(r, total_cost) for r in rows]
    mean = statistics.fmean(pnls)
    sd = statistics.stdev(pnls)
    if sd <= 0:
        return {"net_ev": mean, "p_ev_positive": 1.0 if mean > 0 else (0.0 if mean < 0 else 0.5)}
    z = mean / (sd / math.sqrt(n))
    return {"net_ev": mean, "p_ev_positive": 0.5 * math.erfc(-z / math.sqrt(2.0))}
