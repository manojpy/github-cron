"""Bucket, breakdown, family, regime and threshold-recommendation analytics.

Split out of the original module without logic changes; the original
module remains as a facade re-exporting every name."""
from __future__ import annotations
import math
import time
import statistics
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple
import alert_registry as _alert_registry
from alert_registry import alert_family_of
from threshold_stats import (
    CapRow,
    Row,
    _percentile,
    compute_ev_by_cap,
    confidence_label,
    detect_temporal_drift,
    ev_and_kelly_for,
    ev_and_rr_for,
    ev_first_objective,
    find_knee_point,
    sample_evidence_state,
    two_proportion_p_value,
    wilson_ci,
)

def adaptive_dedup_windows(
    rows: List[Row], *, min_gaps: int = 30, percentile: float = 10.0,
    lo_sec: int = 120, hi_sec: int = 1800,
) -> Dict[str, Dict[str, Any]]:
    """Per-alert-key dedup window from historical same-pair inter-arrival times.

    For every (alert_key, pair) series the gaps between consecutive distinct
    fires are collected; per alert key the chosen percentile of all gaps is the
    window, clamped to [lo_sec, hi_sec]. Keys with fewer than `min_gaps` gaps
    are omitted (caller keeps the fixed window). Pure and deterministic."""
    series: DefaultDict[Tuple[str, str], List[int]] = defaultdict(list)
    for r in rows:
        ak, pair, ts = r.get("alert_key"), r.get("pair"), r.get("entry_ts")
        if not ak or not pair or not ts:
            continue
        series[(str(ak), str(pair))].append(int(ts))
    gaps_by_key: DefaultDict[str, List[int]] = defaultdict(list)
    for (ak, _pair), stamps in series.items():
        ordered = sorted(set(stamps))
        gaps_by_key[ak].extend(b - a for a, b in zip(ordered, ordered[1:]))
    out: Dict[str, Dict[str, Any]] = {}
    for ak, gaps in gaps_by_key.items():
        if len(gaps) < min_gaps:
            continue
        raw = _percentile([float(g) for g in gaps], percentile)
        out[ak] = {
            "window_sec": int(min(max(raw, lo_sec), hi_sec)),
            "raw_sec": int(raw),
            "n_gaps": len(gaps),
        }
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

_ALERT_FAMILY_PREFIXES: List[Tuple[str, str]] = _alert_registry.FAMILY_PREFIXES

def alert_family_analysis(
    rows: List[Row],
    min_sample: int = 20,
    shrinkage_k: float = 20.0,
) -> Dict[str, Any]:
    """Alert-family intelligence (roadmap item #9).

    Treats families as separate learning entities:
      Family → Pair → Regime → Historical outcome
    Uses empirical-Bayes shrinkage toward the global family mean so thin
    pair/regime leaves cannot dominate.
    """
    result: Dict[str, Any] = {"valid": False, "families": {}, "n_total": len(rows)}
    if not rows:
        result["error"] = "no_rows"
        return result

    by_family: DefaultDict[str, List[Row]] = defaultdict(list)
    for r in rows:
        fam = alert_family_of(str(r.get("alert_key") or ""))
        by_family[fam].append(r)

    families: Dict[str, Any] = {}
    for fam, fam_rows in by_family.items():
        n = len(fam_rows)
        if n < min_sample:
            families[fam] = {
                "valid": False, "n": n, "error": "insufficient_sample",
                "evidence_state": sample_evidence_state(n),
            }
            continue
        wins = sum(1 for r in fam_rows if r.get("win"))
        wr = wins / n
        lo, hi, _ = wilson_ci(wins, n)
        net_evs = [
            float(r["net_pnl_pct"]) for r in fam_rows
            if r.get("net_pnl_pct") is not None
        ]
        net_ev = (sum(net_evs) / len(net_evs)) if net_evs else 0.0

        by_pair: DefaultDict[str, List[Row]] = defaultdict(list)
        for r in fam_rows:
            by_pair[str(r.get("pair") or "unknown")].append(r)
        pairs: Dict[str, Any] = {}
        for pair, pair_rows in by_pair.items():
            pn = len(pair_rows)
            if pn < max(5, min_sample // 4):
                continue
            pw = sum(1 for r in pair_rows if r.get("win"))
            raw_wr = pw / pn
            shrunk_wr = (pn * raw_wr + shrinkage_k * wr) / (pn + shrinkage_k)
            pairs[pair] = {
                "n": pn, "raw_wr": round(raw_wr, 4),
                "shrunk_wr": round(shrunk_wr, 4),
                "evidence_state": sample_evidence_state(pn),
            }

        families[fam] = {
            "valid": True, "n": n, "wr": round(wr, 4),
            "wilson_lo": lo, "wilson_hi": hi,
            "confidence": confidence_label(n, lo, hi),
            "net_ev": round(net_ev, 4),
            "evidence_state": sample_evidence_state(n),
            "pairs": pairs,
        }

    result["families"] = families
    result["valid"] = any(f.get("valid") for f in families.values())
    return result

def regime_transition_analysis(
    rows: List[Row],
    min_sample: int = 15,
    lookback_stable: int = 4,
    post_window_hours: int = 6,
) -> Dict[str, Any]:
    """Detect regime transitions and compare post-transition vs stable outcomes
    (roadmap item #15).
    """
    result: Dict[str, Any] = {
        "valid": False, "n_total": len(rows),
        "transitions": [], "post_transition": {}, "stable": {},
    }
    with_adx = [
        r for r in rows
        if r.get("adx_val") is not None and r.get("ts") is not None
    ]
    if len(with_adx) < min_sample * 2:
        result["error"] = "insufficient_adx_tagged_rows"
        return result

    with_adx = sorted(with_adx, key=lambda r: int(r["ts"]))
    adx_vals = sorted(r["adx_val"] for r in with_adx)
    mid = len(adx_vals) // 2
    median_adx = (
        adx_vals[mid] if len(adx_vals) % 2
        else (adx_vals[mid - 1] + adx_vals[mid]) / 2.0
    )
    result["median_adx"] = median_adx

    def _reg(r: Row) -> str:
        return "trending" if r["adx_val"] >= median_adx else "ranging"

    post_window_sec = post_window_hours * 3600
    transition_events: List[Dict[str, Any]] = []
    stable_streak = 1
    prev_reg = _reg(with_adx[0])

    for i in range(1, len(with_adx)):
        cur_reg = _reg(with_adx[i])
        if cur_reg == prev_reg:
            stable_streak += 1
        else:
            if stable_streak >= lookback_stable:
                ts = int(with_adx[i]["ts"])
                transition_events.append({
                    "ts": ts,
                    "from": prev_reg,
                    "to": cur_reg,
                    "stable_before": stable_streak,
                })
            stable_streak = 1
            prev_reg = cur_reg

    result["transitions"] = transition_events[-20:]
    result["n_transitions"] = len(transition_events)

    post_rows: List[Row] = []
    stable_rows: List[Row] = []
    stable_streak = 1
    prev_reg = _reg(with_adx[0])
    active_post_until: Optional[int] = None

    for i, r in enumerate(with_adx):
        cur_reg = _reg(r)
        ts = int(r["ts"])
        if i > 0 and cur_reg != prev_reg:
            if stable_streak >= lookback_stable:
                active_post_until = ts + post_window_sec
            stable_streak = 1
            prev_reg = cur_reg
        else:
            if i > 0:
                stable_streak += 1

        if active_post_until is not None and ts <= active_post_until:
            post_rows.append(r)
        else:
            stable_rows.append(r)
            if active_post_until is not None and ts > active_post_until:
                active_post_until = None

    def _bucket_stats(bucket: List[Row], label: str) -> Dict[str, Any]:
        n = len(bucket)
        if n < min_sample:
            return {"valid": False, "n": n, "error": "insufficient_sample", "label": label}
        wins = sum(1 for r in bucket if r.get("win"))
        wr = wins / n
        lo, hi, _ = wilson_ci(wins, n)
        return {
            "valid": True, "n": n, "wr": round(wr, 4),
            "wilson_lo": lo, "wilson_hi": hi,
            "confidence": confidence_label(n, lo, hi),
            "evidence_state": sample_evidence_state(n),
            "label": label,
        }

    result["post_transition"] = _bucket_stats(post_rows, "post_transition")
    result["stable"] = _bucket_stats(stable_rows, "stable")
    pt = result["post_transition"]
    st = result["stable"]
    if pt.get("valid") and st.get("valid"):
        result["wr_gap"] = round(pt["wr"] - st["wr"], 4)
        result["valid"] = True
    elif st.get("valid") or pt.get("valid"):
        result["valid"] = True
    return result

def strategy_vs_regime_attribution(
    rows: List[Row],
    min_sample: int = 30,
    recent_fraction: float = 0.25,
) -> Dict[str, Any]:
    """Separate strategy degradation from market regime change (roadmap #17)."""
    result: Dict[str, Any] = {
        "valid": False, "n_total": len(rows),
        "attribution": "unknown",
    }
    if len(rows) < min_sample:
        result["error"] = "insufficient_sample"
        result["evidence_state"] = sample_evidence_state(len(rows))
        return result

    sorted_rows = sorted(
        [r for r in rows if r.get("ts") is not None],
        key=lambda r: int(r["ts"]),
    )
    if len(sorted_rows) < min_sample:
        result["error"] = "insufficient_timestamped"
        return result

    cut = max(min_sample // 2, int(len(sorted_rows) * (1.0 - recent_fraction)))
    older = sorted_rows[:cut]
    recent = sorted_rows[cut:]
    if len(recent) < max(10, min_sample // 3):
        result["error"] = "insufficient_recent"
        return result

    def _wr(bucket: List[Row]) -> Tuple[float, int]:
        n = len(bucket)
        if n == 0:
            return 0.5, 0
        return sum(1 for r in bucket if r.get("win")) / n, n

    older_wr, older_n = _wr(older)
    recent_wr, recent_n = _wr(recent)
    result["older_wr"] = round(older_wr, 4)
    result["older_n"] = older_n
    result["recent_wr"] = round(recent_wr, 4)
    result["recent_n"] = recent_n
    result["wr_drop"] = round(older_wr - recent_wr, 4)

    rb = regime_breakdown(rows, min_sample=max(10, min_sample // 2))
    result["regime_breakdown"] = {
        "valid": rb.get("valid"),
        "median_adx": rb.get("median_adx"),
        "regimes": rb.get("regimes"),
    }

    if not rb.get("valid"):
        if result["wr_drop"] > 0.10 and recent_n >= min_sample // 2:
            result["attribution"] = "possible_strategy_degradation"
            result["detail"] = (
                f"Recent WR {recent_wr:.0%} vs older {older_wr:.0%} "
                f"(Δ{result['wr_drop']:+.0%}); regime data insufficient to separate causes."
            )
        else:
            result["attribution"] = "no_significant_drop"
            result["detail"] = "No material performance drop or insufficient regime tags."
        result["valid"] = True
        result["evidence_state"] = sample_evidence_state(len(rows))
        return result

    regimes = rb.get("regimes") or {}
    recent_with_adx = [r for r in recent if r.get("adx_val") is not None]
    median_adx = rb.get("median_adx")
    if recent_with_adx and median_adx is not None:
        n_trend = sum(1 for r in recent_with_adx if r["adx_val"] >= median_adx)
        n_range = len(recent_with_adx) - n_trend
        dominant = "trending" if n_trend >= n_range else "ranging"
        dom_share = max(n_trend, n_range) / len(recent_with_adx)
        result["recent_dominant_regime"] = dominant
        result["recent_dominant_share"] = round(dom_share, 3)

        hist_reg = regimes.get(dominant) or {}
        if hist_reg.get("valid"):
            hist_wr = hist_reg["wr"]
            result["historical_regime_wr"] = round(hist_wr, 4)
            result["historical_regime_n"] = hist_reg["n"]
            if hist_reg["n"] < min_sample:
                result["attribution"] = "insufficient_current_regime_evidence"
                result["detail"] = (
                    f"Recent market is {dominant} ({dom_share:.0%}) but historical "
                    f"{dominant} sample is only n={hist_reg['n']} — do not change thresholds."
                )
            elif result["wr_drop"] > 0.08 and abs(hist_wr - older_wr) < 0.05:
                result["attribution"] = "regime_mix_shift"
                result["detail"] = (
                    f"Overall WR dropped {result['wr_drop']:+.0%}, but historical "
                    f"{dominant} edge remains {hist_wr:.0%} (n={hist_reg['n']}). "
                    f"Likely regime mix shift, not strategy break."
                )
            elif result["wr_drop"] > 0.10 and hist_wr < older_wr - 0.05:
                result["attribution"] = "possible_strategy_degradation"
                result["detail"] = (
                    f"Recent WR {recent_wr:.0%} and historical {dominant} WR {hist_wr:.0%} "
                    f"both below older overall {older_wr:.0%} — edge may be decaying."
                )
            else:
                result["attribution"] = "no_significant_drop"
                result["detail"] = "Performance within normal variation for current regime mix."
        else:
            result["attribution"] = "insufficient_current_regime_evidence"
            result["detail"] = (
                f"Recent market is {dominant} but that regime lacks valid historical stats."
            )
    else:
        result["attribution"] = "insufficient_current_regime_evidence"
        result["detail"] = "Recent outcomes lack ADX tags for regime attribution."

    result["valid"] = True
    result["evidence_state"] = sample_evidence_state(len(rows))
    return result

def ensemble_decision(
    bayesian_p: Optional[float] = None,
    ml_p: Optional[float] = None,
    ev_p: Optional[float] = None,
    recent_wr: Optional[float] = None,
    weights: Optional[Dict[str, float]] = None,
    n_evidence: int = 0,
    oos_validated: bool = False,
) -> Dict[str, Any]:
    """Blend hierarchical Bayesian + ML + EV model + recent performance
    into one calibrated ensemble probability (roadmap item #22).
    """
    w = weights or {
        "bayesian": 0.30, "ml": 0.30, "ev": 0.25, "recent": 0.15,
    }
    components: Dict[str, Optional[float]] = {
        "bayesian": bayesian_p,
        "ml": ml_p,
        "ev": ev_p,
        "recent": recent_wr,
    }
    active = {k: v for k, v in components.items() if v is not None}
    if not active:
        return {
            "valid": False, "error": "no_components",
            "ensemble_p": None,
            "evidence_state": sample_evidence_state(n_evidence, oos_validated),
        }

    total_w = sum(w.get(k, 0.0) for k in active)
    if total_w <= 0:
        total_w = float(len(active))
        norm_w = {k: 1.0 / total_w for k in active}
    else:
        norm_w = {k: w.get(k, 0.0) / total_w for k in active}

    ensemble_p = sum(norm_w[k] * float(active[k]) for k in active)
    contributions = {
        k: {"value": round(float(active[k]), 4), "weight": round(norm_w[k], 3)}
        for k in active
    }
    return {
        "valid": True,
        "ensemble_p": round(ensemble_p, 4),
        "components": contributions,
        "n_components": len(active),
        "evidence_state": sample_evidence_state(n_evidence, oos_validated),
    }

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

def layered_window_analysis(
    recent_rows: List[Row],
    medium_rows: List[Row],
    long_rows: List[Row],
    min_sample: int = 20,
) -> Dict[str, Any]:
    """Per-alert-key comparison of recent vs medium vs long-history net EV,
    so the brain can tell 'this alert is historically good but currently
    in a weak patch' apart from 'this alert has always been weak' — the
    same recent-only window can't distinguish the two on its own.

    `recent_rows`/`medium_rows`/`long_rows` are the same outcome-log rows
    parsed with progressively wider window_days cutoffs (e.g. 30/90/180) —
    medium and long are chronological supersets of recent, not disjoint
    slices. Purely diagnostic: never changes live gating on its own.

    An alert_key is only scored if it has >= min_sample rows in BOTH the
    recent and long windows (the two endpoints being compared) — a key
    with only medium-window evidence is skipped rather than guessed at.
    """
    def _group(rows: List[Row]) -> Dict[str, List[Row]]:
        g: Dict[str, List[Row]] = defaultdict(list)
        for r in rows:
            g[r["alert_key"]].append(r)
        return g

    def _summarize(rows: List[Row]) -> Optional[Dict[str, Any]]:
        if len(rows) < min_sample:
            return None
        ev, _, wr = ev_and_kelly_for(rows)
        return {"n": len(rows), "wr": round(wr, 3), "net_ev": round(ev, 4)}

    recent_g, medium_g, long_g = _group(recent_rows), _group(medium_rows), _group(long_rows)
    all_keys = set(recent_g) | set(medium_g) | set(long_g)

    per_alert: Dict[str, Any] = {}
    for ak in sorted(all_keys):
        recent = _summarize(recent_g.get(ak, []))
        medium = _summarize(medium_g.get(ak, []))
        long_ = _summarize(long_g.get(ak, []))
        if recent is None or long_ is None:
            continue

        recent_ev, long_ev = recent["net_ev"], long_["net_ev"]
        if recent_ev <= 0.0 and long_ev > 0.05:
            verdict = "historically_good_currently_weak"
        elif recent_ev > 0.05 and long_ev <= 0.0:
            verdict = "recently_emerging_edge"
        elif recent_ev <= 0.0 and long_ev <= 0.0:
            verdict = "always_weak"
        elif recent_ev > 0.0 and long_ev > 0.0:
            verdict = "consistently_good"
        else:
            verdict = "mixed"

        per_alert[ak] = {
            "recent": recent, "medium": medium, "long": long_,
            "verdict": verdict,
            "recent_vs_long_ev_gap": round(recent_ev - long_ev, 4),
        }

    return {"valid": bool(per_alert), "per_alert": per_alert}

def hierarchical_combination_analysis(
    rows: List[Row],
    min_leaf_sample: int = 10,
    shrinkage_k: float = 20.0,
) -> Dict[str, Any]:
   
    if not rows:
        return {"valid": False}

    adx_vals = sorted(
        r["adx_val"] for r in rows
        if (r.get("adx_val") if isinstance(r, dict) else None) is not None
    )
    median_adx: Optional[float] = None
    if adx_vals:
        mid = len(adx_vals) // 2
        median_adx = (
            adx_vals[mid] if len(adx_vals) % 2
            else (adx_vals[mid - 1] + adx_vals[mid]) / 2.0
        )

    def _regime(r: Row) -> str:
        adx_val = r.get("adx_val")
        if median_adx is None or adx_val is None:
            return "unknown"
        return "trending" if adx_val >= median_adx else "ranging"

    def _raw(bucket_rows: List[Row]) -> Dict[str, Any]:
        n = len(bucket_rows)
        if n == 0:
            return {"n": 0, "wr": 0.0, "net_ev": 0.0}
        wins = sum(1 for r in bucket_rows if r["win"])
        ev, _, _ = ev_and_kelly_for(bucket_rows)
        return {"n": n, "wr": wins / n, "net_ev": ev}

    def _shrink(raw: Dict[str, Any], parent: Dict[str, Any]) -> Dict[str, float]:
        n = raw["n"]
        wr = (n * raw["wr"] + shrinkage_k * parent["wr"]) / (n + shrinkage_k)
        ev = (n * raw["net_ev"] + shrinkage_k * parent["net_ev"]) / (n + shrinkage_k)
        return {"wr": wr, "net_ev": ev}

    by_alert: DefaultDict[str, List[Row]] = defaultdict(list)
    by_alert_dir: DefaultDict[Tuple[str, str], List[Row]] = defaultdict(list)
    by_alert_dir_regime: DefaultDict[Tuple[str, str, str], List[Row]] = defaultdict(list)
    by_leaf: DefaultDict[Tuple[str, str, str, str], List[Row]] = defaultdict(list)

    for r in rows:
        ak = r["alert_key"]
        d = r.get("direction", "?")
        p = r.get("pair", "?")
        reg = _regime(r)
        by_alert[ak].append(r)
        by_alert_dir[(ak, d)].append(r)
        by_alert_dir_regime[(ak, d, reg)].append(r)
        by_leaf[(ak, d, reg, p)].append(r)

    global_shrunk = _raw(rows)  # root has no parent to shrink toward

    alert_shrunk: Dict[str, Dict[str, float]] = {
        ak: _shrink(_raw(bucket), global_shrunk) for ak, bucket in by_alert.items()
    }
    dir_shrunk: Dict[Tuple[str, str], Dict[str, float]] = {
        key: _shrink(_raw(bucket), alert_shrunk[key[0]])
        for key, bucket in by_alert_dir.items()
    }
    regime_shrunk: Dict[Tuple[str, str, str], Dict[str, float]] = {
        key: _shrink(_raw(bucket), dir_shrunk[(key[0], key[1])])
        for key, bucket in by_alert_dir_regime.items()
    }

    leaves: Dict[str, Any] = {}
    for (ak, d, reg, p), bucket in by_leaf.items():
        raw = _raw(bucket)
        if raw["n"] < min_leaf_sample:
            continue
        shrunk = _shrink(raw, regime_shrunk[(ak, d, reg)])
        parent_dir = dir_shrunk[(ak, d)]
        key = f"{p}|{ak}|{d}|{reg}"
        leaves[key] = {
            "pair": p, "alert_key": ak, "direction": d, "regime": reg,
            "n": raw["n"],
            "raw_wr": round(raw["wr"], 3), "shrunk_wr": round(shrunk["wr"], 3),
            "raw_net_ev": round(raw["net_ev"], 4), "shrunk_net_ev": round(shrunk["net_ev"], 4),
            "alert_dir_baseline_net_ev": round(parent_dir["net_ev"], 4),
            "vs_alert_dir_baseline": round(shrunk["net_ev"] - parent_dir["net_ev"], 4),
        }

    return {"valid": bool(leaves), "median_adx": median_adx, "leaves": leaves}

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
    stats: DefaultDict[str, Dict[str, int]] = defaultdict(lambda: {"wins": 0, "n": 0})
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
    stats: DefaultDict[Tuple[str, str], Dict[str, int]] = defaultdict(lambda: {"wins": 0, "n": 0})
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
    stats: DefaultDict[str, Dict[str, int]] = defaultdict(lambda: {"wins": 0, "n": 0})
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
    stats: DefaultDict[str, Dict[str, Any]] = defaultdict(lambda: {"wins": 0, "n": 0, "scores": []})
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

def classify_strategy_state(
    rows: List[Row],
    recent_days: float = 14,
    min_recent: int = 20,
    min_older: int = 40,
    min_regime_n: int = 15,
    drop_threshold: float = 0.10,
    alpha: float = 0.10,
    trend_adx: float = 25.0,
    degraded_alpha: float = 0.05,
) -> Dict[str, Any]:
    """Is a recent win-rate drop 'the strategy broke' or 'we are in a regime
    the history barely covers'? (roadmap #17)

    Regime = trending if adx_val >= ``trend_adx`` (fixed), else ranging. A
    median split would put half of the history in each regime by construction,
    so an underrepresented regime could never be detected.

    States: INSUFFICIENT_DATA, STABLE, DEGRADED_REGIME_UNKNOWN,
    REGIME_UNDERREPRESENTED, REGIME_SHIFT, STRATEGY_DEGRADED (the hardest to
    earn, since it is the costly label). Advisory only; never changes a gate.
    """
    now_ts = int(time.time())
    cutoff = now_ts - int(recent_days * 86400)
    recent = [r for r in rows if r.get("entry_ts", 0) >= cutoff]
    older = [r for r in rows if 0 < r.get("entry_ts", 0) < cutoff]
    out: Dict[str, Any] = {
        "state": "INSUFFICIENT_DATA", "recent_n": len(recent), "older_n": len(older),
        "recent_wr": None, "older_wr": None, "drop": None,
        "expected_recent_wr": None, "mix_adjusted_drop": None,
        "regimes": {}, "action": "collect more outcomes before judging",
    }
    if len(recent) < min_recent or len(older) < min_older:
        return out

    r_wr = sum(1 for r in recent if r["win"]) / len(recent)
    o_wr = sum(1 for r in older if r["win"]) / len(older)
    drop = o_wr - r_wr
    out.update({"recent_wr": round(r_wr, 4), "older_wr": round(o_wr, 4), "drop": round(drop, 4)})

    # One-sided two-proportion z-test: is recent WR really below older WR?
    pooled = (sum(1 for r in recent if r["win"]) + sum(1 for r in older if r["win"])) / (
        len(recent) + len(older))
    se = math.sqrt(max(pooled * (1 - pooled), 1e-12) * (1 / len(recent) + 1 / len(older)))
    z = drop / se if se > 0 else 0.0
    p_value = 0.5 * math.erfc(z / math.sqrt(2))
    out["p_value"] = round(p_value, 4)

    if drop < drop_threshold or p_value > alpha:
        out.update({"state": "STABLE", "action": "no action"})
        return out

    older_adx = sorted(float(r["adx_val"]) for r in older if r.get("adx_val") is not None)
    has_adx = lambda r: r.get("adx_val") is not None  # noqa: E731
    if (len(older_adx) < 0.6 * len(older)
            or sum(1 for r in recent if has_adx(r)) < 0.6 * len(recent)):
        out.update({
            "state": "DEGRADED_REGIME_UNKNOWN",
            "action": "win rate fell but ADX coverage is too thin to separate decay from regime",
        })
        return out

    def regime_of(r: Row) -> Optional[str]:
        if r.get("adx_val") is None:
            return None
        return "trending" if float(r["adx_val"]) >= trend_adx else "ranging"

    recent_r = [r for r in recent if regime_of(r)]
    older_r = [r for r in older if regime_of(r)]
    regimes: Dict[str, Any] = {}
    for name in ("trending", "ranging"):
        o = [r for r in older_r if regime_of(r) == name]
        c = [r for r in recent_r if regime_of(r) == name]
        regimes[name] = {
            "older_n": len(o), "recent_n": len(c),
            "older_wr": round(sum(1 for r in o if r["win"]) / len(o), 4) if o else None,
            "recent_wr": round(sum(1 for r in c if r["win"]) / len(c), 4) if c else None,
            "recent_share": round(len(c) / len(recent_r), 4) if recent_r else 0.0,
        }
    out["regimes"] = regimes

    thin = [n for n, g in regimes.items()
            if g["recent_share"] >= 0.25 and g["older_n"] < min_regime_n]
    if thin:
        out.update({
            "state": "REGIME_UNDERREPRESENTED", "underrepresented": thin,
            "action": (
                f"recent trades are in {', '.join(thin)} where history has "
                f"<{min_regime_n} trades: do not retune; keep shadow-logging"
            ),
        })
        return out

    expected = sum(
        g["recent_share"] * g["older_wr"] for g in regimes.values() if g["older_wr"] is not None
    )
    mix_adj = expected - r_wr
    out["expected_recent_wr"] = round(expected, 4)
    out["mix_adjusted_drop"] = round(mix_adj, 4)
    # Variance of (expected - recent): recent sampling noise PLUS the noise in
    # the per-regime historical WRs the expectation is built from.
    var_recent = max(r_wr * (1 - r_wr), 1e-12) / len(recent)
    var_expected = sum(
        (g["recent_share"] ** 2) * (g["older_wr"] * (1 - g["older_wr"])) / g["older_n"]
        for g in regimes.values() if g["older_wr"] is not None and g["older_n"] > 0
    )
    se_mix = math.sqrt(var_recent + var_expected)
    mix_p = 0.5 * math.erfc((mix_adj / se_mix) / math.sqrt(2)) if se_mix > 0 else 1.0
    out["mix_adjusted_p_value"] = round(mix_p, 4)
    if mix_adj >= 0.6 * drop_threshold and mix_p <= degraded_alpha:
        out.update({
            "state": "STRATEGY_DEGRADED",
            "action": "win rate is down even within the same regimes: review recent changes, freeze retuning",
        })
    else:
        out.update({
            "state": "REGIME_SHIFT",
            "action": "drop is explained by more trades in a historically weaker regime: expected, do not retune",
        })
    return out

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
    # ── EV-first target floor selection ──
    target_floor = None
    best_ev_at_cap = None

    _cheap_positive_cap: Optional[float] = None
    for cap, _n_pass, _wr, _wr_lo in caps_data:
        subset = [r for r in rows if r["score"] >= cap]
        if len(subset) < min_sample:
            continue
        _ev_point, _hk, _wr_point = ev_and_kelly_for(subset)
        if _ev_point > 0:
            if _cheap_positive_cap is None or cap < _cheap_positive_cap:
                _cheap_positive_cap = cap

    if _cheap_positive_cap is not None:
        _final_subset = [r for r in rows if r["score"] >= _cheap_positive_cap]
        _full_ev_obj = ev_first_objective(_final_subset, min_sample=min_sample)
        if (
            _full_ev_obj.get("valid")
            and _full_ev_obj["p_ev_positive"] >= 0.85
            and _full_ev_obj["ev_p5"] > -0.10
        ):
            target_floor = _cheap_positive_cap
            best_ev_at_cap = _full_ev_obj

    ev_gate_passed = target_floor is not None

    # Fallback: if no cap clears the EV gate, use the WR floor
    # but flag it as provisional
    if target_floor is None:
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
    result["ev_gate_passed"] = ev_gate_passed
    if best_ev_at_cap:
        result["target_ev_objective"] = best_ev_at_cap
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
    sorted_vote_names: List[str] = sorted(vote_names)
    interactions: List[Dict[str, Any]] = []

    for i, v1 in enumerate(sorted_vote_names):
        for v2 in sorted_vote_names[i + 1 :]:
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

            # ── v1 poisons v2 ─────────────────────────────────────────
            
            if has_v2_sample:
                assert wr_only_v2 is not None  # guaranteed by has_v2_sample
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

    results: List[Dict[str, Any]] = []
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
    results: List[Dict[str, Any]] = []
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
