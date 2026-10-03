"""Root-cause diagnosis, repair effectiveness, config simulation/versioning.

Split out of the original module without logic changes; the original
module remains as a facade re-exporting every name."""
from __future__ import annotations
import hashlib
import json
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple
from bot_config import cfg, format_ist_time
from threshold_analysis import (
    recommend_threshold,
)
from threshold_models import (
    _train_logistic,
)
from threshold_stats import (
    Row,
    _flatten_row_features,
    _prob_edge_broken,
    _prob_ev_negative,
    _sigmoid,
    brier_score_and_calibration,
    ev_and_kelly_for,
    ev_and_rr_for,
    two_proportion_p_value,
    wilson_ci,
)
from threshold_validation import (
    detect_feature_drift,
    find_wr_change_point,
)

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
# ══════════════════════════════════════════════════════════════════════
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
    "ENABLE_VWAP", "ENABLE_PIVOT",
    "ENABLE_PPO_ALERTS", "ENABLE_PPOHIST_ALERT", "ENABLE_RSI_ALERTS",
    "ENABLE_HIST_RMA", "ENABLE_RVOL_ALERT", "ENABLE_ADX_FILTER",
    "ENABLE_CLOUD_CROSS_ALERT", "ENABLE_DYNAMIC_FLOW_CROSS_ALERT",
    "ENABLE_TK_CONVERSION_CROSS", "ENABLE_KIJUN_CROSS", "ENABLE_EQUILIBRIUM_CROSS",
    "ENABLE_STRONG_REVERSAL_ALERT", "ENABLE_CHOCH_ALERT", "ENABLE_FIB_REVERSAL_ALERT",
    "ENABLE_OB_PREMIUM_DISCOUNT_FILTER", "OB_FILTER_CONFLUENCE",
    "ENABLE_OI_PRICE_DIVERGENCE",
    "ENABLE_MACRO_CONTEXT_GATE", "MACRO_CONTEXT_LIVE", "ENABLE_CLUSTER_GATE",
    "ENABLE_CONFLUENCE_GATE", "ENABLE_WIN_RATE_FILTER",
    "ENABLE_SESSION_FILTER", "ENABLE_ALERT_COALESCING",
    "CHOCH_REQUIRE_FVG", "CHOCH_CHECK_POI_TAP", "CHOCH_ALLOW_SAME_CANDLE_SWEEP",
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

    # ── FIX (Priority 7): Programmatically include ALL BotConfig fields
    _NON_BEHAVIORAL_FIELDS = frozenset({
        # ── infra / credentials / rate limits (unchanged) ──
        "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "REDIS_URL",
        "DELTA_API_BASE", "LOG_LEVEL", "DEBUG_MODE", "SEND_TEST_MESSAGE",
        "BOT_NAME", "DRY_RUN_MODE", "SKIP_WARMUP", "FAIL_ON_REDIS_DOWN",
        "FAIL_ON_TELEGRAM_DOWN", "TELEGRAM_RATE_LIMIT_PER_MINUTE",
        "TELEGRAM_BURST_SIZE", "REDIS_CONNECTION_RETRIES", "REDIS_RETRY_DELAY",
        "HTTP_TIMEOUT", "CANDLE_FETCH_RETRIES", "CANDLE_FETCH_BACKOFF",
        "MAX_PARALLEL_FETCH", "TCP_CONN_LIMIT", "TCP_CONN_LIMIT_PER_HOST",
        "MEMORY_LIMIT_BYTES", "TELEGRAM_RETRIES", "TELEGRAM_BACKOFF_BASE",
        "RUN_TIMEOUT_SECONDS", "FETCH_PHASE_TIMEOUT_SEC",
        "EVAL_CONCURRENCY_LIMIT", "MIN_RUN_TIMEOUT",
        "BRAIN_USE_FILE_STORAGE", "OUTCOME_DATA_DIR",
        "BRAIN_REPORT_ON_DEMAND", "BRAIN_REPORT_INTERVAL_RUNS", "BRAIN_ARCHIVE_SHALLOW",
        "BRAIN_REPORT_STREAM_SAMPLE", "BRAIN_LONG_WINDOW_STREAM_SAMPLE",
        # ── FIX (Priority 7): report-only Brain flags. Toggling any of
        # these changes nothing about which trades fire or what's stored
        # on the outcome row — only the analysis/report output. Without
        # this, flipping a report flag emits a spurious 'regression' in
        # compare_config_versions() / config_regression_pinpoint. ──
        "BRAIN_MC_SIMULATIONS",
        "BRAIN_PERMUTATION_IMPORTANCE",
        "BRAIN_MAX_PLAN_ENTRIES",
        "BRAIN_REPAIR_SHOP_MAX",
        "ENABLE_LAYERED_WINDOW_ANALYSIS",
        "BRAIN_CONFLUENCE_BUCKET_PCT",
        "ENABLE_HIERARCHICAL_COMBINATION_ANALYSIS",
        "HIERARCHICAL_MIN_LEAF_SAMPLE",
        "HIERARCHICAL_SHRINKAGE_K",
        "BRAIN_WEIGHT_OPTIMIZER_MAX_DELTA",
        "BRAIN_WEIGHT_OPTIMIZER_WALK_FORWARD",
        "BRAIN_WEIGHT_OPTIMIZER_MIN_CONFIDENCE",
        "BRAIN_STABILITY_MIN_HISTORY",
        "BRAIN_STABILITY_MAX_JUMP",
        "BRAIN_CUSUM_DRIFT_DELTA",
        "BRAIN_CUSUM_THRESHOLD",
        "BRAIN_EV_GATE_P_THRESHOLD",
        "BRAIN_EV_GATE_P5_FLOOR",
        "ENABLE_MARKET_STATE_MODEL",
        "MARKET_STATE_MODEL_MIN_SAMPLE",
        "MARKET_STATE_MODEL_MIN_OOS_P",
        "ENABLE_MARKET_STATE_LIVE_SCORE",
        "ENABLE_PNL_WEIGHTED_TRAINING",
        "ENABLE_FILL_RECONCILIATION",
        "BRAIN_AUDIT_ENABLED",
        "BRAIN_AUDIT_MIN_ROWS_FOR_RECOMMENDATION",
        "BRAIN_AUDIT_MIN_HISTORY_RATIO",
    })
    try:
        for field_name in type(cfg).model_fields:
            if field_name in _NON_BEHAVIORAL_FIELDS:
                continue
            if field_name in extra_fields:
                continue  # Caller already provided it
            try:
                extra_fields[field_name] = getattr(cfg, field_name, None)
            except Exception:
                pass
    except Exception:
        # Fallback to the manual tuple if model_fields isn't available
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

def learn_repair_effectiveness(
    ledger_records: List[Dict[str, Any]],
    current_state: Dict[str, Any],
    min_records: int = 100,
) -> Dict[str, Any]:
    """Trains on resolved repair-ledger entries (snapshot_before + verdict)
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
        alert_values = [d.get("alert") for d in drift_alerts]
        drifted_keys = sorted({a for a in alert_values if a})
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
            # Wiring #4: mechanical config patch — this repair is just
            # "set the field to this value," so hand it straight to the
            # patch pipeline instead of leaving it prose-only.
            "config_field": "CONFLUENCE_MIN_ABS_SCORE",
            "config_current": float(current_threshold),
            "config_suggested": float(rec["recommended"]),
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
            # Structured so the brain can auto-propose a revert patch,
            # not just describe the regression in prose.
            "version_before": cp.get("version_before"),
            "version_after": cp.get("version_after"),
            "delta_wr": cp["delta"],
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
