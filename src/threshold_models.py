"""Learned models: vote-weight optimisation, market-state model, calibration curves.

Split out of the original module without logic changes; the original
module remains as a facade re-exporting every name."""
from __future__ import annotations
import math
import time
import random
import statistics
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple
from bot_config import cfg
from threshold_stats import (
    Row,
    _sigmoid,
    confidence_label,
    ev_and_kelly_for,
    ev_first_objective,
    row_net_pnl_pct,
    walk_forward_split,
    wilson_ci,
)
from threshold_validation import (
    detect_feature_drift,
)

def _build_vote_dataset(
    rows: List[Row],
    vote_names: List[str],
    use_pnl_weighting: bool = False,
) -> Tuple[List[List[float]], List[float], List[float]]:
    
    raw_weights = _pnl_sample_weights(rows) if use_pnl_weighting else None

    X: List[List[float]] = []
    y: List[float] = []
    sw: List[float] = []
    for i, r in enumerate(rows):
        votes = r.get("votes")
        if not votes or not isinstance(votes, dict):
            continue
        vec = [1.0] + [1.0 if votes.get(vn) else 0.0 for vn in vote_names]
        X.append(vec)
        y.append(1.0 if r["win"] else 0.0)
        if raw_weights is not None:
            sw.append(raw_weights[i])
        else:
            sw.append(r.get("win_weight", 1.0) if r["win"] else 1.0)
    return X, y, sw

def build_market_state_features(
    rows: List[Row],
    feature_names: Optional[List[str]] = None,
) -> Tuple[List[List[float]], List[float], List[str]]:
    """Extract full market-state feature matrix from rows.

    Features: votes + numeric context + session encoding + direction.
    Returns (X, y, feature_names).
    """
    if feature_names is None:
        all_features: Set[str] = set()
        for r in rows:
            if r.get("votes"):
                all_features.update(f"vote:{k}" for k in r["votes"])
            if r.get("context"):
                for k, v in r["context"].items():
                    if isinstance(v, (int, float)):
                        all_features.add(f"ctx:{k}")
        feature_names = sorted(all_features)

    X: List[List[float]] = []
    y: List[float] = []

    for r in rows:
        vec = [1.0]  # intercept
        votes = r.get("votes") or {}
        ctx = r.get("context") or {}

        for fname in feature_names:
            if fname.startswith("vote:"):
                vname = fname[5:]
                vec.append(1.0 if votes.get(vname) else 0.0)
            elif fname.startswith("ctx:"):
                cname = fname[4:]
                val = ctx.get(cname)
                vec.append(float(val) if isinstance(val, (int, float)) else 0.0)

        # Session encoding
        session = r.get("session", "unknown")
        for s in ["asian", "london", "ny", "dead"]:
            vec.append(1.0 if session == s else 0.0)

        # Direction
        vec.append(1.0 if r.get("direction") == "buy" else 0.0)

        X.append(vec)
        y.append(1.0 if r["win"] else 0.0)

    return X, y, feature_names

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
    target_wr: float = 0.55,
) -> Tuple[float, int, float]:
    """Apply logistic beta to rows. Returns (net_ev, n_kept, wr) for
    predicted-positive rows. Net EV uses the same cost model as
    ev_first_objective so the WF veto is comparable to the optimizer's
    own objective."""
    kept: List[Row] = []
    for r in rows:
        votes = r.get("votes")
        if not votes or not isinstance(votes, dict):
            continue
        z = beta[0] + sum(
            beta[j + 1] for j, vn in enumerate(vote_names) if votes.get(vn)
        )
        if _sigmoid(z) >= threshold:
            kept.append(r)
    if not kept:
        return 0.0, 0, 0.0
    net_ev, _hk, wr = ev_and_kelly_for(kept)
    return net_ev, len(kept), wr

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
    use_pnl_weighting = getattr(cfg, "ENABLE_PNL_WEIGHTED_TRAINING", True)
    X, y, sample_weights = _build_vote_dataset(
        rows, vote_names, use_pnl_weighting=use_pnl_weighting,
    )
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
        X_train, y_train, sw_train = _build_vote_dataset(
            train_rows, vote_names, use_pnl_weighting=use_pnl_weighting,
        )
        if len(X_train) < min_sample // 2 or len(holdout_rows) < min_sample // 3:
            # Fall back to full-data training with low confidence
            beta = _train_logistic(X, y, sample_weights, max_iter, lr, l2)
            confidence_score = min(0.3, n / 1000.0)
        else:
            beta = _train_logistic(X_train, y_train, sw_train, max_iter, lr, l2)

            wf_ev, wf_n, wf_holdout_wr = _score_with_beta(holdout_rows, beta, vote_names)
            _, _, wf_baseline_wr = _score_with_beta(holdout_rows, [0.0]*len(beta), vote_names, threshold=0.0)
            # Simpler: baseline net EV is just the holdout's own net EV.
            baseline_ev, _, _ = ev_and_kelly_for(holdout_rows)

            if wf_n >= 10:
                wf_passed = (wf_ev >= baseline_ev) and (wf_holdout_wr >= wf_baseline_wr - 0.05)
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
    voted_rows = sorted(
        (r for r in rows if isinstance(r.get("votes"), dict)),
        key=lambda r: r.get("entry_ts", 0),
    )
    block_size = max(5, min(20, len(voted_rows) // 10 or 1))
    blocks = [
        voted_rows[i:i + block_size]
        for i in range(0, len(voted_rows), block_size)
    ]
    blocks = [b for b in blocks if b]

    bootstrap_betas: List[List[float]] = []
    rng = random.Random(42)
    if len(blocks) >= 5:
        for _ in range(5):
            sampled_blocks = rng.choices(blocks, k=len(blocks))
            boot_rows = [r for blk in sampled_blocks for r in blk]
            X_boot, y_boot, sw_boot = _build_vote_dataset(
                boot_rows, vote_names, use_pnl_weighting=use_pnl_weighting,
            )
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

def permutation_vote_importance(
    rows: List[Row],
    min_sample: int = 30,
    n_permutations: int = 20,
    seed: int = 42,
    min_side: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Per-vote information test: how much of the WR gap between rows where
    the vote fired and rows where it did not survives shuffling the vote?

    importance = |observed WR gap| - mean(|WR gap| after shuffling the vote
    column). A vote with no information scores ~0; an informative one scores
    roughly its true gap. A vote needs at least ``min_side`` rows on BOTH
    sides to be scored; otherwise it is omitted (unmeasurable, not "noise").
    """
    if len(rows) < min_sample:
        return []
    side_floor = int(min_side) if min_side is not None else max(10, min_sample // 3)

    vote_names: Set[str] = set()
    for r in rows:
        if r.get("votes"):
            vote_names.update(r["votes"].keys())
    if not vote_names:
        return []

    wins = [1 if r["win"] else 0 for r in rows]
    rng = random.Random(seed)
    results: List[Dict[str, Any]] = []

    def _gap(mask: List[bool]) -> Optional[float]:
        n_on = sum(mask)
        n_off = len(mask) - n_on
        if n_on < side_floor or n_off < side_floor:
            return None
        w_on = sum(w for m, w in zip(mask, wins) if m)
        w_off = sum(wins) - w_on
        return w_on / n_on - w_off / n_off

    for vn in sorted(vote_names):
        mask = [bool((r.get("votes") or {}).get(vn)) for r in rows]
        observed = _gap(mask)
        if observed is None:
            continue
        null_gaps: List[float] = []
        shuffled = list(mask)
        for _ in range(max(1, n_permutations)):
            rng.shuffle(shuffled)
            g = _gap(shuffled)
            if g is not None:
                null_gaps.append(abs(g))
        if not null_gaps:
            continue
        excess = abs(observed) - statistics.fmean(null_gaps)
        p_value = (1 + sum(1 for g in null_gaps if g >= abs(observed))) / (1 + len(null_gaps))
        results.append({
            "vote": vn,
            "importance": round(excess, 4),
            "p_value": round(p_value, 3),
            "observed_gap": round(observed, 4),
            "std": round(statistics.pstdev(null_gaps), 4) if len(null_gaps) > 1 else 0.0,
            "direction": "positive" if observed > 0 else "negative",
        })

    results.sort(key=lambda x: -abs(x["importance"]))
    return results

def actionable_condition_ablation(
    rows: List[Row],
    min_sample: int = 40,
    n_permutations: int = 15,
    noise_threshold: float = 0.01,
    edge_threshold: float = 0.03,
) -> List[Dict[str, Any]]:
    """Turn permutation importance into concrete actions.

    Returns a list of dicts:
      {
        "vote": str,
        "importance": float,          # mean WR drop when shuffled
        "direction": "noise"|"edge"|"weak",
        "action": "reduce_weight"|"keep"|"investigate",
        "suggested_weight_factor": float,  # 0.5 = halve, 1.0 = leave
        "reason": str,
      }

    - importance ≈ 0  → condition adds almost no information (noise)
    - importance >> 0 → condition carries real edge
    Never auto-applies; caller feeds this into the plan / challenger path.
    """
    raw = permutation_vote_importance(
        rows, min_sample=min_sample, n_permutations=n_permutations,
    )
    out: List[Dict[str, Any]] = []
    for item in raw:
        imp = float(item.get("importance") or 0.0)
        vote = item["vote"]
        p_val = float(item.get("p_value", 0.0))
        if imp < noise_threshold:
            out.append({
                "vote": vote,
                "importance": imp,
                "direction": "noise",
                "action": "reduce_weight",
                "suggested_weight_factor": 0.5,
                "reason": (
                    f"Permutation importance {imp:+.3f} ≈ 0 — "
                    f"condition adds almost no information once others are present"
                ),
            })
        elif imp >= edge_threshold and p_val <= 0.10:
            out.append({
                "vote": vote,
                "importance": imp,
                "direction": "edge",
                "action": "keep",
                "suggested_weight_factor": 1.0,
                "reason": (
                    f"Permutation importance {imp:+.3f} — "
                    f"shuffling this vote materially hurts WR"
                ),
            })
        else:
            out.append({
                "vote": vote,
                "importance": imp,
                "direction": "weak",
                "action": "investigate",
                "suggested_weight_factor": 1.0,
                "reason": f"Permutation importance {imp:+.3f} — weak / unstable signal",
            })
    return out

def _pnl_sample_weights(
    rows: List[Row],
    fee_pct: float = 0.0006,
    slippage_pct: float = 0.0003,
    min_weight: float = 0.15,
) -> List[float]:
    """Per-row training weights from actual net P&L magnitude (item #13):
    a trade that made/lost 2% should pull the fit harder than one that
    made/lost 0.05%, instead of every row counting as one classification
    example regardless of size. Uses the same net_pnl_pct-with-fallback
    definition as ev_first_objective() so 'important' means the same
    thing during training as it does during OOS acceptance.

    Weights are normalized to mean 1.0 so the effective sample size (and
    therefore existing min_sample/l2 tuning) stays comparable to plain
    uniform weighting, and floored at min_weight so a near-zero-P&L row
    still contributes rather than vanishing from the fit entirely."""
    total_cost = (fee_pct * 2 + slippage_pct * 2) * 100
    raw: List[float] = []

    for r in rows:
        net_pnl = row_net_pnl_pct(r, total_cost)
        raw.append(max(min_weight, abs(float(net_pnl))))

    mean_w = statistics.fmean(raw) if raw else 1.0
    if mean_w <= 0:
        return [1.0] * len(rows)
    return [w / mean_w for w in raw]

def oos_permutation_importance(
    rows: List[Row],
    min_sample: int = 30,
    n_permutations: int = 15,
    train_frac: float = 0.67,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """OOS permutation importance using EV deterioration.

    Trains a logistic model on the train split, then measures
    how much OOS EV drops when each feature is shuffled.
    """
    train_rows, holdout_rows = walk_forward_split(rows, train_frac)
    if len(train_rows) < min_sample or len(holdout_rows) < min_sample:
        return []

    X_train, y_train, feat_names = build_market_state_features(train_rows)
    X_hold, y_hold, _ = build_market_state_features(holdout_rows, feat_names)
    sw_train = (
        _pnl_sample_weights(train_rows)
        if getattr(cfg, "ENABLE_PNL_WEIGHTED_TRAINING", True)
        else [1.0] * len(X_train)
    )
    beta = _train_logistic(X_train, y_train, sw_train, max_iter=1500)
    if not beta:
        return []

    def _eval_ev(X: List[List[float]], y_rows: List[Row]) -> float:
        kept = []
        for i, row in enumerate(y_rows):
            z = sum(b * x for b, x in zip(beta, X[i]))
            if _sigmoid(z) >= 0.5:
                kept.append(row)
        if len(kept) < 5:
            return 0.0
        ev, _, _ = ev_and_kelly_for(kept)
        return ev

    baseline_ev = _eval_ev(X_hold, holdout_rows)
    rng = random.Random(seed)
    results: List[Dict[str, Any]] = []

    for feat_idx, fname in enumerate(feat_names):
        drops = []
        for _ in range(n_permutations):
            X_shuffled = [row[:] for row in X_hold]
            col_vals = [X_shuffled[i][feat_idx + 1] for i in range(len(X_shuffled))]
            rng.shuffle(col_vals)
            for i in range(len(X_shuffled)):
                X_shuffled[i][feat_idx + 1] = col_vals[i]

            shuffled_ev = _eval_ev(X_shuffled, holdout_rows)
            drops.append(baseline_ev - shuffled_ev)

        mean_drop = statistics.fmean(drops)
        results.append({
            "feature": fname,
            "importance_ev": round(mean_drop, 5),
            "std": round(statistics.pstdev(drops), 5) if len(drops) > 1 else 0.0,
            "direction": "positive" if mean_drop > 0 else "negative",
        })
    results.sort(key=lambda x: -abs(float(x["importance_ev"])))
    return results

def train_market_state_model(
    rows: List[Row],
    min_sample: int = 150,
    train_frac: float = 0.67,
    min_oos_p_ev_positive: float = 0.70,
) -> Dict[str, Any]:
    """Train the market-state logistic model (votes + numeric context +
    session + direction — Recommended.txt items #4/#9) on a purge/embargo
    -safe split, then decide whether to accept it purely on OOS
    profitability of the holdout, not training-set classification
    accuracy (item #6). Returns a dict with valid=False and a reason if
    the fit doesn't clear the bar — callers should keep serving whatever
    model they already had rather than overwrite it with this one.

    Also builds an ML calibration curve on the full holdout (p_win, label)
    and returns its ECE so callers can compare against conf_pct ECE
    before wiring the future EV pipeline.
    """
    if len(rows) < min_sample:
        return {"valid": False, "error": "insufficient_data", "n": len(rows)}

    train_rows, holdout_rows = walk_forward_split(rows, train_frac)
    if len(train_rows) < min_sample * 0.5 or len(holdout_rows) < min_sample * 0.3:
        return {
            "valid": False, "error": "insufficient_split",
            "n_train": len(train_rows), "n_holdout": len(holdout_rows),
        }
    try:
        drift_check = detect_feature_drift(rows, recent_n=len(holdout_rows))
    except Exception:
        drift_check = {"valid": False, "error": "drift_check_failed"}

    X_train, y_train, feat_names = build_market_state_features(train_rows)
    X_hold, y_hold, _ = build_market_state_features(holdout_rows, feat_names)
    sw_train = (
        _pnl_sample_weights(train_rows)
        if getattr(cfg, "ENABLE_PNL_WEIGHTED_TRAINING", True)
        else [1.0] * len(X_train)
    )
    beta = _train_logistic(X_train, y_train, sw_train, max_iter=1500)
    if not beta:
        return {"valid": False, "error": "training_failed", "drift_check": drift_check}

    # OOS acceptance gate: only the trades the model would actually have
    # taken (P(win) >= 0.5) on data it never trained on, evaluated on net
    # EV — mirrors ev_first_objective's acceptance bar, not just accuracy.
    kept = [
        row for i, row in enumerate(holdout_rows)
        if _sigmoid(sum(b * x for b, x in zip(beta, X_hold[i]))) >= 0.5
    ]
    min_kept = max(10, int(min_sample * 0.2))
    if len(kept) < min_kept:
        return {
            "valid": False, "error": "holdout_too_thin_at_decision_boundary",
            "n_kept": len(kept), "min_kept": min_kept, "drift_check": drift_check,
        }
    holdout_ev_obj = ev_first_objective(kept, min_sample=min_kept)
    if not holdout_ev_obj.get("valid") or holdout_ev_obj["p_ev_positive"] < min_oos_p_ev_positive:
        return {
            "valid": False, "error": "oos_ev_not_convincing",
            "holdout_ev": holdout_ev_obj, "drift_check": drift_check,
        }

    # ── ML calibration curve on full holdout (p_win vs realized label) ──
    # Same math as build_calibration_curves, different input column.
    # Used only for ECE comparison and (later) calibrated P lookup;
    # does not affect the OOS EV acceptance gate above.
    holdout_preds = [
        _sigmoid(sum(b * x for b, x in zip(beta, X_hold[i])))
        for i in range(len(holdout_rows))
    ]
    holdout_labels = [bool(r["win"]) for r in holdout_rows]
    ml_calib = build_ml_calibration_curve(
        holdout_preds,
        holdout_labels,
        n_bins=10,
        min_sample=max(10, min_sample // 10),
    )

    return {
        "valid": True,
        "beta": beta,
        "feature_names": feat_names,
        "n_train": len(train_rows),
        "n_holdout": len(holdout_rows),
        "n_kept_at_decision": len(kept),
        "holdout_ev": holdout_ev_obj,
        "ml_calibration": ml_calib,
        "ml_ece": ml_calib.get("ece"),
        "trained_at": int(time.time()),
        "drift_check": drift_check,
    }

def predict_market_state_proba(
    model: Optional[Dict[str, Any]],
    votes: Optional[Dict[str, bool]],
    context: Optional[Dict[str, Any]],
    session: str = "unknown",
    direction: str = "buy",
) -> Optional[float]:
    """Score one live, prospective trade against a persisted
    train_market_state_model() fit — the piece that was missing for a
    true per-trade contextual prediction (item #12), as opposed to the
    per-alert-key aggregate EV bucket trade_quality_score() used before.
    Returns None (never raises) if there's no valid model yet or the
    feature vector can't be built."""
    if not model or not model.get("valid"):
        return None
    beta = model.get("beta")
    feat_names = model.get("feature_names")
    if not beta or not feat_names:
        return None
    row: Row = {
        "votes": votes or {}, "context": context or {},
        "session": session, "direction": direction, "win": False,
    }
    X, _, _ = build_market_state_features([row], feat_names)
    if not X:
        return None
    z = sum(b * x for b, x in zip(beta, X[0]))
    return _sigmoid(z)

def build_calibration_curves(
    rows: List[Row],
    bucket_pct: float = 5.0,
    min_sample: int = 15,
    shadow_rows: Optional[List[Row]] = None,
) -> Dict[str, Any]:
    """Per-alert-key calibration: bucketed confluence % → observed win rate.
    A raw confluence score is a weighted vote total, not a probability.
    This maps what the score DISPLAYS (conf_pct, treated as the implied
    probability claim /100) to what actually HAPPENED, per alert key, so
    the live gate can filter on calibrated probability instead of face
    value. ECE = standard expected calibration error over buckets.

    Buckets are equal-FREQUENCY (quantile), not equal-width: alert
    dispatch already gates on a confluence floor, so conf_pct is
    right-truncated/skewed rather than uniform across 0-100%, and fixed
    bucket_pct-wide bins leave most of them sparse or empty right where
    the gate threshold lives. bucket_pct still sets the target bin count
    (100/bucket_pct, capped by how many min_sample-sized groups the data
    actually supports), so existing CALIBRATION_BUCKET_PCT config values
    keep behaving the same way in spirit.
    """
    by_ak: Dict[str, List[Row]] = defaultdict(list)
    for r in rows:
        by_ak[r["alert_key"]].append(r)

    # ── FIX (Priority 6): Merge shadow rows into the calibration
    if shadow_rows:
        for r in shadow_rows:
            rejection = (r.get("context") or {}).get("rejection_reason", "")
            if rejection == "calibration_gate":
                continue  # Avoid selection leakage
            by_ak[r["alert_key"]].append(r)

    target_bins = max(1, round(100.0 / bucket_pct)) if bucket_pct > 0 else 20

    curves: Dict[str, Any] = {}
    for ak, ak_rows in by_ak.items():
        if len(ak_rows) < min_sample:
            continue
        ordered = sorted(ak_rows, key=lambda r: r["conf_pct"])
        n_bins = max(1, min(target_bins, len(ordered) // min_sample))
        chunk_size = math.ceil(len(ordered) / n_bins)
        chunks = [ordered[i:i + chunk_size] for i in range(0, len(ordered), chunk_size)]
        chunks = [c for c in chunks if c]

        # Contiguous 0-100 coverage: each boundary sits halfway between one
        # chunk's max conf_pct and the next chunk's min, so a live conf_pct
        # anywhere in 0-100 lands in exactly one bucket (no gaps for
        # calibration_gate_decision's range lookup to fall through).
        boundaries = [0.0]
        for i in range(len(chunks) - 1):
            boundaries.append(
                (chunks[i][-1]["conf_pct"] + chunks[i + 1][0]["conf_pct"]) / 2.0
            )
        boundaries.append(100.0)

        out = []
        for idx, chunk in enumerate(chunks):
            n = len(chunk)
            wins = sum(r["win"] for r in chunk)
            wr = wins / n
            lo, hi, _ = wilson_ci(wins, n)
            pred = statistics.mean(r["conf_pct"] for r in chunk) / 100.0
            out.append({
                "lo": round(boundaries[idx], 2), "hi": round(boundaries[idx + 1], 2),
                "predicted": round(pred, 4), "observed": round(wr, 4),
                "n": n, "trusted": n >= min_sample,
                "wilson_lo": round(lo, 4), "wilson_hi": round(hi, 4),
            })
        total = len(ak_rows)
        ece = sum(
            (bk["n"] / total) * abs(bk["observed"] - bk["predicted"])
            for bk in out
        )
        curves[ak] = {
            "buckets": out,
            "ece": round(ece, 4),
            "n": total,
        }
    ece_values = [c["ece"] for c in curves.values()]

    return {
        "curves": curves,
        "ece_mean": round(statistics.fmean(ece_values), 4) if ece_values else None,
        "ece_mean_label": "mean_per_alert_ece",
        "built_at": int(time.time()),
    }

def fold_outcomes_into_calibration(
    calib: Dict[str, Any],
    rows: List[Row],
    min_sample: int = 15,
) -> int:
    """Online update: fold newly resolved rows into EXISTING curve buckets.

    Mutates ``calib`` in place and returns how many rows were folded. Bucket
    boundaries are not moved (re-quantiling needs the full rebuild); only
    n / wins / observed / predicted / Wilson CI / ECE change, so all history
    already in the curve is kept. ``built_at`` is left untouched so the
    age-based full rebuild still fires on schedule.
    """
    curves = calib.get("curves") or {}
    folded = 0
    touched: Set[str] = set()
    for r in rows:
        curve = curves.get(r.get("alert_key"))
        if not curve:
            continue
        buckets = curve.get("buckets") or []
        if not buckets or r.get("conf_pct") is None:
            continue
        conf = float(r["conf_pct"])
        last_idx = len(buckets) - 1
        chosen = None
        for idx, bk in enumerate(buckets):
            if bk["lo"] <= conf < bk["hi"] or (idx == last_idx and conf == bk["hi"]):
                chosen = bk
                break
        if chosen is None:
            continue
        n = int(chosen["n"])
        wins = int(chosen["wins"]) if "wins" in chosen else int(round(chosen["observed"] * n))
        new_n = n + 1
        new_wins = wins + (1 if r.get("win") else 0)
        chosen["predicted"] = round((chosen["predicted"] * n + conf / 100.0) / new_n, 4)
        chosen["n"] = new_n
        chosen["wins"] = new_wins
        chosen["observed"] = round(new_wins / new_n, 4)
        lo, hi, _ = wilson_ci(new_wins, new_n)
        chosen["wilson_lo"] = round(lo, 4)
        chosen["wilson_hi"] = round(hi, 4)
        chosen["trusted"] = new_n >= min_sample
        curve["n"] = int(curve.get("n") or 0) + 1
        touched.add(r["alert_key"])
        folded += 1

    for ak in touched:
        curve = curves[ak]
        total = sum(bk["n"] for bk in curve["buckets"])
        if total > 0:
            curve["ece"] = round(
                sum((bk["n"] / total) * abs(bk["observed"] - bk["predicted"])
                    for bk in curve["buckets"]),
                4,
            )
    if touched:
        ece_values = [c["ece"] for c in curves.values() if c.get("ece") is not None]
        calib["ece_mean"] = round(statistics.fmean(ece_values), 4) if ece_values else None
    return folded

def build_ml_calibration_curve(
    predictions: List[float],
    labels: List[bool],
    n_bins: int = 10,
    min_sample: int = 15,
) -> Dict[str, Any]:
    """Calibration curve on model output: bins on raw P(profit), not conf_pct.

    predictions: OOS holdout p_win values in [0, 1]
    labels: corresponding resolved win/loss (True/False)
    Returns same shape as one curve from build_calibration_curves so
    calibration_gate_decision / ECE math can be reused.
    """
    if len(predictions) != len(labels) or len(predictions) < min_sample:
        return {"buckets": [], "ece": None, "n": len(predictions)}

    pairs = sorted(zip(predictions, labels), key=lambda t: t[0])
    n = len(pairs)
    n_bins = max(1, min(n_bins, n // min_sample))
    chunk_size = math.ceil(n / n_bins)
    chunks = [pairs[i:i + chunk_size] for i in range(0, n, chunk_size)]
    chunks = [c for c in chunks if c]
    boundaries = [0.0]
    for i in range(len(chunks) - 1):
        prev_pred: float = chunks[i][-1][0]
        next_pred: float = chunks[i + 1][0][0]
        boundaries.append((prev_pred + next_pred) / 2.0)
    boundaries.append(1.0)
    out = []
    for idx, chunk in enumerate(chunks):
        n_c = len(chunk)
        wins = sum(1 for _, y in chunk if y)
        wr = wins / n_c
        lo, hi, _ = wilson_ci(wins, n_c)
        pred = statistics.mean(p for p, _ in chunk)
        out.append({
            "lo": round(boundaries[idx], 4),
            "hi": round(boundaries[idx + 1], 4),
            "predicted": round(pred, 4),
            "observed": round(wr, 4),
            "n": n_c,
            "trusted": n_c >= min_sample,
            "wilson_lo": round(lo, 4),
            "wilson_hi": round(hi, 4),
        })
    ece = sum((bk["n"] / n) * abs(bk["observed"] - bk["predicted"]) for bk in out)
    return {
        "buckets": out,
        "ece": round(ece, 4),
        "n": n,
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
    _last_idx = len(buckets) - 1
    for idx, bk in enumerate(buckets):
        if bk["lo"] <= conf_pct < bk["hi"] or (idx == _last_idx and conf_pct == bk["hi"]):
            chosen = bk
            break
    if chosen is None:
        return True, None, "out_of_range_fail_open"
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

def ml_calibration_lookup(
    curve: Dict[str, Any],
    p_win: float,
    min_sample: int = 15,
) -> Tuple[Optional[float], str]:
    """Return (calibrated_P, reason) for a raw model p_win.
    Returns (None, reason) for thin/untrusted buckets so the caller falls
    back to the RAW model prediction. A calibration layer must never
    replace a raw probability with a bucket's noisy empirical WR on a
    sample too small to trust — same fail-open discipline as
    calibration_gate_decision."""
    buckets = curve.get("buckets", [])
    if not buckets:
        return None, "no_curve"
    chosen = None
    _last_idx = len(buckets) - 1
    for idx, bk in enumerate(buckets):
        if bk["lo"] <= p_win < bk["hi"] or (idx == _last_idx and p_win == bk["hi"]):
            chosen = bk
            break
    if chosen is None:    
        return None, "out_of_range_fail_open"
    if not chosen.get("trusted") or chosen["n"] < min_sample:
        return None, "thin_bucket_fail_open"
    return chosen["observed"], "ok"
