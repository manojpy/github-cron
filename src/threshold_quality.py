"""Trade-quality verdict path: MAE/MFE plans, TP/SL zones, regime gate, kill switch, risk checks, trade_quality_score.

Split out of the original module without logic changes; the original
module remains as a facade re-exporting every name."""
from __future__ import annotations
import math
import time
import statistics
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Tuple
from bot_config import cfg
from alert_registry import alert_family_of
from threshold_analysis import (
    ensemble_decision,
)
from threshold_models import (
    calibration_gate_decision,
    ml_calibration_lookup,
)
from threshold_stats import (
    Row,
    _percentile,
    ev_first_objective,
    per_trade_ev,
    row_net_pnl_pct,
    sample_evidence_state,
    two_proportion_p_value,
    walk_forward_split,
)
from threshold_validation import (
    _holdout_ev_check,
)

def mae_mfe_trade_plan(
    rows: List[Row],
    sl_percentile: float = 70.0,
    tp1_percentile: float = 60.0,
    tp2_percentile: float = 85.0,
    sl_min_pct: float = 0.15,
    sl_max_pct: float = 3.0,
) -> Optional[Dict[str, Any]]:
    """... (median MAE would be hit ~half the time; sl_percentile covers
    most of the historical adverse excursion, clamped to a safety range;
    TP1/TP2 from MFE percentiles; percentages, not fractions)"""
    maes = [abs(r["mae"]) * 100.0 for r in rows if r.get("mae") is not None]
    mfes = [abs(r["mfe"]) * 100.0 for r in rows if r.get("mfe") is not None]
    if not maes or not mfes:
        return None
    tp_first_rows = [r for r in rows if r.get("tp_first") is not None]
    tp_first_rate = (
        sum(1 for r in tp_first_rows if r["tp_first"] is True) / len(tp_first_rows)
        if tp_first_rows else None
    )
    sl = max(sl_min_pct, min(sl_max_pct, _percentile(maes, sl_percentile)))
    tp1 = _percentile(mfes, tp1_percentile)
    tp2 = max(tp1, _percentile(mfes, tp2_percentile))
    return {
        "n": len(rows), "n_mae": len(maes), "n_mfe": len(mfes),
        "median_mae_pct": round(_percentile(maes, 50.0), 3),
        "median_mfe_pct": round(_percentile(mfes, 50.0), 3),
        "sl_suggested_pct": round(sl, 3),
        "tp1_suggested_pct": round(tp1, 3),
        "tp2_suggested_pct": round(tp2, 3),
        "tp_first_rate": round(tp_first_rate, 3) if tp_first_rate is not None else None,
        "n_tp_first": len(tp_first_rows),
        "sl_percentile": sl_percentile, "tp1_percentile": tp1_percentile, "tp2_percentile": tp2_percentile,
    }

def mae_mfe_profiles_by_bucket(rows, min_sample=15, **plan_kwargs):
    """Builds mae_mfe_trade_plan() at pair+alert+direction ("leaf:"),
    alert+direction ("alert_dir:"), alert ("alert:"), and "global"."""
    by_leaf, by_alert_dir, by_alert = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in rows:
        ak, d, p = r.get("alert_key", "?"), r.get("direction", "?"), r.get("pair", "?")
        by_leaf[f"{p}|{ak}|{d}"].append(r)
        by_alert_dir[f"{ak}|{d}"].append(r)
        by_alert[ak].append(r)
    profiles = {}
    for prefix, buckets in (("leaf", by_leaf), ("alert_dir", by_alert_dir), ("alert", by_alert)):
        for key, bucket in buckets.items():
            if len(bucket) < min_sample:
                continue
            plan = mae_mfe_trade_plan(bucket, **plan_kwargs)
            if plan:
                profiles[f"{prefix}:{key}"] = plan
    if len(rows) >= min_sample:
        plan = mae_mfe_trade_plan(rows, **plan_kwargs)
        if plan:
            profiles["global"] = plan
    return profiles

def lookup_mae_mfe_plan(profiles, pair, alert_key, direction):
    """Most specific bucket with enough history wins, else falls back
    down to alert, then global. Returns a copy tagged with the bucket used."""
    for key in (f"leaf:{pair}|{alert_key}|{direction}", f"alert_dir:{alert_key}|{direction}",
                f"alert:{alert_key}", "global"):
        plan = profiles.get(key)
        if plan:
            plan = dict(plan)
            plan["bucket"] = key
            return plan
    return None

def zone_replay_pnl(
    row: Row, sl_pct: float, tp_pct: float, total_cost_pct: float,
) -> Optional[float]:
    """Net P&L (percent) the trade WOULD have produced with a different
    stop/target, from its recorded excursions. MAE/MFE are whole-window
    extremes with no ordering, so when BOTH levels were touched the trade is
    counted as a STOP (pessimistic: the zone's replayed EV is a lower bound).
    Neither touched -> exits at the horizon close. None if the row lacks the
    needed fields."""
    mae, mfe = row.get("mae"), row.get("mfe")
    if mae is None or mfe is None:
        return None
    cost = row.get("realized_cost_pct")
    cost = float(cost) if cost is not None else total_cost_pct
    if abs(float(mae)) * 100.0 >= sl_pct:
        return -sl_pct - cost
    if abs(float(mfe)) * 100.0 >= tp_pct:
        return tp_pct - cost
    pm = row.get("pct_move")
    if pm is None:
        return None
    gross = float(pm) if row.get("direction") == "buy" else -float(pm)
    return gross - cost

def _zone_bucket_keys(row: Row, median_adx: Optional[float]) -> List[str]:
    ak, d, pair = row.get("alert_key"), row.get("direction"), row.get("pair")
    if not ak or not d or not pair:
        return []
    reg = _regime_label(row.get("adx_val"), median_adx)
    fam = alert_family_of(str(ak))
    keys = [f"alertdir:{ak}|{d}", f"famdir:{fam}|{d}"]
    if reg != "unknown":
        keys += [f"leafreg:{pair}|{ak}|{d}|{reg}", f"alertreg:{ak}|{d}|{reg}", f"famreg:{fam}|{d}|{reg}"]
    return keys

def _zone_median_adx(rows: List[Row]) -> Optional[float]:
    vals = sorted(float(r["adx_val"]) for r in rows if r.get("adx_val") is not None)
    if not vals:
        return None
    mid = len(vals) // 2
    return vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2.0

def zone_candidates(
    rows: List[Row],
    *,
    min_n: int = 60,
    min_holdout: int = 20,
    min_delta_pct: float = 0.05,
    min_p_better: float = 0.80,
    stability_tol: float = 0.35,
    max_deviation: float = 2.0,
    min_rr: float = 1.0,
    sl_percentile: float = 70.0,
    tp1_percentile: float = 60.0,
    tp2_percentile: float = 85.0,
    sl_min_pct: float = 0.15,
    sl_max_pct: float = 3.0,
    fee_pct: float = 0.0006,
    slippage_pct: float = 0.0003,
) -> Dict[str, Any]:
    """Build and VALIDATE a TP/SL zone per bucket (pair+alert+dir+regime,
    alert+dir+regime, family+dir+regime, alert+dir, family+dir).

    A candidate PASSES only if all hold:
      * sample        - n >= min_n rows carrying MAE/MFE;
      * OOS replay    - zone learned on the chronological TRAIN split, replayed
                        on the purged/embargoed HOLDOUT (>= min_holdout rows),
                        beats the fixed bracket on the same trades by
                        >= min_delta_pct, with paired-difference probability
                        >= min_p_better;
      * stability     - SL and TP1 re-learned on the holdout differ from the
                        train values by <= stability_tol (relative);
      * safety rails  - SL and TP1 within [1/max_deviation, max_deviation] x the
                        fixed bracket, and TP1/SL >= min_rr.
    Pure and deterministic. Returns {"median_adx", "candidates": {key: {...}}}."""
    usable = [r for r in rows if r.get("mae") is not None and r.get("mfe") is not None]
    median_adx = _zone_median_adx(usable)
    total_cost = (fee_pct * 2 + slippage_pct * 2) * 100
    fixed_sl = float(cfg.OUTCOME_MAE_LOSS_PCT)
    fixed_tp = fixed_sl * float(cfg.OUTCOME_RR_TARGET)

    buckets: DefaultDict[str, List[Row]] = defaultdict(list)
    for r in usable:
        for k in _zone_bucket_keys(r, median_adx):
            buckets[k].append(r)

    plan_kw = dict(sl_percentile=sl_percentile, tp1_percentile=tp1_percentile,
                   tp2_percentile=tp2_percentile, sl_min_pct=sl_min_pct, sl_max_pct=sl_max_pct)
    out: Dict[str, Dict[str, Any]] = {}
    for key, bucket in sorted(buckets.items()):
        n = len(bucket)
        if n < min_n:
            continue
        train, hold = walk_forward_split(bucket)
        cand: Dict[str, Any] = {"bucket": key, "n": n, "n_holdout": len(hold), "passed": False, "reasons": []}
        out[key] = cand
        plan = mae_mfe_trade_plan(train, **plan_kw)
        if not plan or len(hold) < min_holdout:
            cand["reasons"].append("insufficient_train_or_holdout")
            continue
        sl, tp1, tp2 = plan["sl_suggested_pct"], plan["tp1_suggested_pct"], plan["tp2_suggested_pct"]
        cand.update({"sl_pct": sl, "tp1_pct": tp1, "tp2_pct": tp2, "tp_first_rate": plan.get("tp_first_rate"), "n_tp_first": plan.get("n_tp_first")})
        # safety rails
        if sl <= 0 or tp1 <= 0 or tp1 / sl < min_rr:
            cand["reasons"].append("rr_below_floor")
        for name, val, ref in (("sl", sl, fixed_sl), ("tp1", tp1, fixed_tp)):
            if ref > 0 and not (ref / max_deviation <= val <= ref * max_deviation):
                cand["reasons"].append(f"{name}_outside_deviation_rail")
        # stability: re-learn on the holdout split
        hold_plan = mae_mfe_trade_plan(hold, **plan_kw)
        if not hold_plan:
            cand["reasons"].append("holdout_plan_unavailable")
        else:
            for name, a, b in (("sl", sl, hold_plan["sl_suggested_pct"]),
                               ("tp1", tp1, hold_plan["tp1_suggested_pct"])):
                if a > 0 and abs(b - a) / a > stability_tol:
                    cand["reasons"].append(f"{name}_unstable")
        # OOS replay vs the fixed bracket, paired per trade
        diffs: List[float] = []
        zone_pnls: List[float] = []
        base_pnls: List[float] = []
        for r in hold:
            z = zone_replay_pnl(r, sl, tp1, total_cost)
            if z is None:
                continue
            b = row_net_pnl_pct(r, total_cost)
            zone_pnls.append(z)
            base_pnls.append(b)
            diffs.append(z - b)
        if len(diffs) < min_holdout:
            cand["reasons"].append("holdout_replay_too_thin")
        else:
            mean_d = statistics.fmean(diffs)
            sd = statistics.stdev(diffs)
            if sd <= 0:
                p_better = 1.0 if mean_d > 0 else 0.5
            else:
                p_better = 0.5 * math.erfc(-(mean_d / (sd / math.sqrt(len(diffs)))) / math.sqrt(2.0))
            cand.update({
                "zone_ev": round(statistics.fmean(zone_pnls), 4),
                "fixed_ev": round(statistics.fmean(base_pnls), 4),
                "delta_ev": round(mean_d, 4),
                "p_better": round(p_better, 4),
                "n_replayed": len(diffs),
            })
            if mean_d < min_delta_pct:
                cand["reasons"].append("delta_below_margin")
            if p_better < min_p_better:
                cand["reasons"].append("not_significantly_better")
        cand["passed"] = not cand["reasons"]
    return {"median_adx": median_adx, "candidates": out}

def zone_promote(
    candidates: Dict[str, Dict[str, Any]],
    prev_streaks: Dict[str, int],
    required_passes: int = 2,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    """Streak-based promotion. A passing candidate's streak increments, a
    failing or vanished one drops out (reset = demotion). A zone is promoted
    only once its streak reaches `required_passes`. Returns
    (promoted_zones, new_streaks)."""
    streaks: Dict[str, int] = {}
    promoted: Dict[str, Dict[str, Any]] = {}
    for key, cand in candidates.items():
        if not cand.get("passed"):
            continue
        s = int(prev_streaks.get(key, 0)) + 1
        streaks[key] = s
        if s >= required_passes:
            promoted[key] = dict(cand, streak=s)
    return promoted, streaks

def zone_lookup(
    blob: Optional[Dict[str, Any]], pair: str, alert_key: str, direction: str,
    adx_val: Optional[float],
) -> Optional[Dict[str, Any]]:
    """Most specific PROMOTED zone for a prospective alert, else None.
    Order: pair+alert+dir+regime, alert+dir+regime, family+dir+regime,
    alert+dir, family+dir."""
    if not blob or not isinstance(blob, dict):
        return None
    zones = blob.get("zones") or {}
    if not zones:
        return None
    reg = _regime_label(adx_val, blob.get("median_adx"))
    fam = alert_family_of(str(alert_key))
    keys = []
    if reg != "unknown":
        keys += [f"leafreg:{pair}|{alert_key}|{direction}|{reg}",
                 f"alertreg:{alert_key}|{direction}|{reg}",
                 f"famreg:{fam}|{direction}|{reg}"]
    keys += [f"alertdir:{alert_key}|{direction}", f"famdir:{fam}|{direction}"]
    for k in keys:
        z = zones.get(k)
        if z:
            return dict(z, bucket=k)
    return None

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
        total_cost = ((self.fee_pct * 2) + (self.slippage_pct * 2)) * 100
        window = [r for r in ordered if r["entry_ts"] >= cutoff]
        equity = 1.0
        for r in window:
            equity *= (1.0 + row_net_pnl_pct(r, total_cost) / 100.0)
        pnl = (equity - 1.0) * 100.0
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
            pair_rows: List[Dict[str, Any]] = []
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
    wins = [abs(float(r.get("pct_move", 0.0))) for r in rows if r["win"]]
    if len(wins) < min_sample:
        return {"valid": False, "measured": False, "error": "insufficient_data", "n": len(wins)}
    realized_pp = statistics.fmean(wins)
    expected_pp = rr_target * stop_pct          # theoretical TP distance, same units
    shortfall_pp = max(0.0, expected_pp - realized_pp)
    implied_frac = (shortfall_pp / 2.0) / 100.0  # split across entry+exit, pp → fraction

    per_pair_est: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        if r["win"]:
            per_pair_est[r["pair"]].append(abs(float(r.get("pct_move", 0.0))))

    pair_rows_est: List[Dict[str, Any]] = []
    for pair, moves in per_pair_est.items():
        if len(moves) < max(3, min_sample // 2):
            continue
        implied_p = max(0.0, (expected_pp - statistics.fmean(moves)) / 2.0) / 100.0
        pair_rows_est.append({
            "pair": pair, "n": len(moves),
            "realized_slippage_per_side": round(implied_p, 6),
            "gap_bps": round((implied_p - assumed_slippage_pct) * 10000, 1),
        })

    pair_rows_est.sort(key=lambda x: -float(x["gap_bps"]))

    return {
        "valid": True, "measured": False, "n": len(wins),
        "realized_move_pct_win": round(realized_pp, 4),
        "expected_move_pct_win": round(expected_pp, 4),
        "realized_slippage_per_side": round(implied_frac, 6),
        "assumed_slippage_per_side": assumed_slippage_pct,
        "gap_bps": round((implied_frac - assumed_slippage_pct) * 10000, 2),
        "ev_overstated_pct_per_trade": round(2 * max(0.0, implied_frac - assumed_slippage_pct), 6),
        "per_pair": pair_rows_est,
        "note": "estimated from win-move shortfall; wire fill prices into the "
                "outcome writer for the measured tier",
    }

def _regime_label(adx_val: Optional[float], median_adx: Optional[float]) -> str:
    """Same trending/ranging split as regime_breakdown and the hierarchical
    leaves: ADX at or above the window median is 'trending'."""
    if adx_val is None or median_adx is None:
        return "unknown"
    return "trending" if float(adx_val) >= float(median_adx) else "ranging"

def regime_gate_analysis(
    rows: List[Row],
    *,
    min_n_downgrade: int = 50,
    min_n_block: int = 100,
    min_holdout: int = 20,
    downgrade_p: float = 0.35,
    block_p: float = 0.20,
    min_gap_pct: float = 0.10,
    min_sample: int = 15,
) -> Dict[str, Any]:
    """Regime-conditioned evidence for the live quality gate.

    For every (alert_key, direction, regime) segment this asks "is this alert
    bad specifically in THIS regime?":
      * sample gate   - DOWNGRADE needs n >= min_n_downgrade, BLOCK needs
                        n >= min_n_block;
      * regime gap    - segment net EV must be >= min_gap_pct worse than the
                        same alert+direction over all regimes;
      * probability   - segment P(net EV > 0) <= downgrade_p (or block_p);
      * OOS confirm   - BLOCK additionally needs a purged/embargoed
                        chronological holdout of >= min_holdout rows that is
                        itself negative with P(net EV > 0) <= block_p.
    The output can only ever restrict a verdict. Pure and deterministic."""
    adx_vals = sorted(float(r["adx_val"]) for r in rows if r.get("adx_val") is not None)
    out: Dict[str, Any] = {"valid": False, "median_adx": None, "segments": {}, "n_evaluated": 0}
    if len(adx_vals) < 2 * min_sample:
        out["error"] = "insufficient_adx_tagged_rows"
        return out
    mid = len(adx_vals) // 2
    median_adx = adx_vals[mid] if len(adx_vals) % 2 else (adx_vals[mid - 1] + adx_vals[mid]) / 2.0
    out["median_adx"] = median_adx

    by_ad: DefaultDict[Tuple[str, str], List[Row]] = defaultdict(list)
    by_adr: DefaultDict[Tuple[str, str, str], List[Row]] = defaultdict(list)
    for r in rows:
        ak, d = r.get("alert_key"), r.get("direction")
        if not ak or not d:
            continue
        reg = _regime_label(r.get("adx_val"), median_adx)
        if reg == "unknown":
            continue
        by_ad[(str(ak), str(d))].append(r)
        by_adr[(str(ak), str(d), reg)].append(r)

    base_cache: Dict[Tuple[str, str], Optional[float]] = {}
    for (ak, d, reg), seg in sorted(by_adr.items()):
        n = len(seg)
        if n < min_n_downgrade:
            continue
        ev = ev_first_objective(seg, min_sample=min_sample)
        if not ev.get("valid"):
            continue
        out["n_evaluated"] += 1
        if (ak, d) not in base_cache:
            b = ev_first_objective(by_ad[(ak, d)], min_sample=min_sample)
            base_cache[(ak, d)] = float(b["net_ev"]) if b.get("valid") else None
        base_ev = base_cache[(ak, d)]
        seg_ev = float(ev["net_ev"])
        p_pos = float(ev["p_ev_positive"])
        gap = (seg_ev - base_ev) if base_ev is not None else None
        regime_specific = gap is not None and gap <= -float(min_gap_pct)

        train, hold = walk_forward_split(seg)
        hold_ev = _holdout_ev_check(hold) if len(hold) >= min_holdout else None
        hold_negative = bool(
            hold_ev
            and float(hold_ev["net_ev"]) < 0.0
            and float(hold_ev["p_ev_positive"]) <= block_p
        )
        action = "NONE"
        if regime_specific and seg_ev < 0.0:
            if n >= min_n_block and p_pos <= block_p and hold_negative:
                action = "BLOCK"
            elif p_pos <= downgrade_p:
                action = "DOWNGRADE"
        out["segments"][f"{ak}|{d}|{reg}"] = {
            "alert_key": ak, "direction": d, "regime": reg, "n": n,
            "net_ev": round(seg_ev, 4), "p_ev_positive": round(p_pos, 4),
            "baseline_net_ev": None if base_ev is None else round(base_ev, 4),
            "gap_vs_baseline": None if gap is None else round(gap, 4),
            "n_holdout": len(hold),
            "holdout_net_ev": None if not hold_ev else round(float(hold_ev["net_ev"]), 4),
            "holdout_p_ev_positive": None if not hold_ev else round(float(hold_ev["p_ev_positive"]), 4),
            "oos_confirmed_negative": hold_negative,
            "action": action,
        }
    out["valid"] = True
    return out

def regime_gate_lookup(
    blob: Optional[Dict[str, Any]], alert_key: str, direction: str, adx_val: Optional[float],
) -> Optional[Dict[str, Any]]:
    """Persisted segment for the CURRENT regime of a prospective alert, or None
    (unknown ADX, no blob, or no actionable segment)."""
    if not blob or not isinstance(blob, dict):
        return None
    reg = _regime_label(adx_val, blob.get("median_adx"))
    if reg == "unknown":
        return None
    seg = (blob.get("segments") or {}).get(f"{alert_key}|{direction}|{reg}")
    if not seg or seg.get("action") not in ("BLOCK", "DOWNGRADE"):
        return None
    return seg

def apply_regime_gate(result: Dict[str, Any], seg: Optional[Dict[str, Any]], mode: str) -> None:
    """Apply a regime-gate segment to a trade-quality result IN PLACE.

    Restrict-only: BLOCK -> BLOCKED, DOWNGRADE -> at most LOW. A verdict that is
    already BLOCKED or LOW is never changed, and nothing is ever raised.
    mode 'shadow' only annotates; 'off' does nothing."""
    if not seg or mode == "off":
        return
    action = str(seg.get("action"))
    note = (
        f"{seg.get('regime')} regime: net EV {float(seg.get('net_ev', 0.0)):+.2f}% "
        f"vs {float(seg.get('baseline_net_ev') or 0.0):+.2f}% overall "
        f"(P={float(seg.get('p_ev_positive', 0.0)):.0%}, n={seg.get('n')})"
    )
    entry = {"action": action, "note": note, "n": seg.get("n"), "regime": seg.get("regime")}
    if mode != "live":
        result["regime_gate_shadow"] = entry
        return
    before = result.get("verdict")
    if before == "BLOCKED":
        return
    if action == "BLOCK":
        result["verdict_before_regime_gate"] = before
        result["verdict"] = "BLOCKED"
        result["reason"] = f"regime_gate: {note}"
        entry["applied"] = True
    elif action == "DOWNGRADE" and before in ("HIGH", "MEDIUM"):
        result["verdict_before_regime_gate"] = before
        result["verdict"] = "LOW"
        entry["applied"] = True
    else:
        entry["applied"] = False
    result["regime_gate"] = entry

def apply_ensemble_gate(
    result: Dict[str, Any], mode: str, min_n: int, max_p: float,
    n_oos: int, oos_validated: bool,
) -> None:
    """Apply the ensemble probability to a trade-quality result IN PLACE.

    Restrict-only: when the gate is eligible (out-of-sample validated and at
    least min_n trades) and ensemble_p <= max_p, a HIGH verdict drops to MEDIUM
    and a MEDIUM verdict drops to LOW. LOW/BLOCKED verdicts are never changed
    and nothing is ever raised. mode 'shadow' only annotates; 'off' does nothing."""
    if mode == "off":
        return
    ens_p = result.get("ensemble_p")
    before = result.get("verdict")
    if ens_p is None or before not in ("HIGH", "MEDIUM"):
        return
    if not (bool(oos_validated) and int(n_oos) >= int(min_n)):
        return
    if float(ens_p) > float(max_p):
        return
    target = "MEDIUM" if before == "HIGH" else "LOW"
    entry = {
        "ensemble_p": round(float(ens_p), 4), "max_p": float(max_p),
        "n_oos": int(n_oos), "from": before, "to": target,
    }
    if mode != "live":
        result["ensemble_gate_shadow"] = entry
        return
    result["verdict_before_ensemble_gate"] = before
    result["verdict"] = target
    entry["applied"] = True
    result["ensemble_gate"] = entry

def gate_shadow_comparison(rows: List[Row], min_flagged: int = 20) -> Dict[str, Any]:
    """Did the alerts a gate flagged really do worse than the ones it left alone?

    A row counts as flagged by a gate when its stored context["gate_shadow"]
    holds that gate's shadow ('would act') entry or a live entry with
    applied=True. Compared against every other resolved row (win rate, net EV,
    two-sided p-value). 'ready' needs min_flagged rows on both sides;
    'looks_justified' means ready, flagged win rate lower and p <= 0.20.
    Purely diagnostic: never changes a verdict."""
    out: Dict[str, Any] = {"valid": True, "gates": {}}
    for gate in ("ensemble", "regime"):
        flagged: List[Row] = []
        clean: List[Row] = []
        for r in rows:
            ctx = r.get("context")
            gs = ctx.get("gate_shadow") if isinstance(ctx, dict) else None
            gs = gs if isinstance(gs, dict) else {}
            live = gs.get(f"{gate}_gate")
            if gs.get(f"{gate}_gate_shadow") or (isinstance(live, dict) and live.get("applied")):
                flagged.append(r)
            else:
                clean.append(r)
        nf, nc = len(flagged), len(clean)
        wf = sum(1 for r in flagged if r.get("win"))
        wc = sum(1 for r in clean if r.get("win"))
        wr_f = (wf / nf) if nf else None
        wr_c = (wc / nc) if nc else None
        p = two_proportion_p_value(wf, nf, wc, nc)
        ev_f = ev_first_objective(flagged, min_sample=min_flagged) if nf else {}
        ev_c = ev_first_objective(clean, min_sample=min_flagged) if nc else {}
        ready = nf >= min_flagged and nc >= min_flagged
        worse = wr_f is not None and wr_c is not None and wr_f < wr_c
        entry = {
            "n_flagged": nf, "n_unflagged": nc,
            "wr_flagged": None if wr_f is None else round(wr_f, 4),
            "wr_unflagged": None if wr_c is None else round(wr_c, 4),
            "net_ev_flagged": round(ev_f["net_ev"], 4) if ev_f.get("valid") else None,
            "net_ev_unflagged": round(ev_c["net_ev"], 4) if ev_c.get("valid") else None,
            "p_value": round(p, 4),
            "ready": ready,
            "looks_justified": bool(ready and worse and p <= 0.20),
        }
        if nf:
            entry["text"] = (
                f"{gate}: flagged n={nf} WR={wr_f:.0%} vs unflagged n={nc} "
                f"WR={(wr_c if wr_c is not None else 0.0):.0%} (p={p:.2f}) -> "
                + ("looks justified" if entry["looks_justified"]
                   else "not enough evidence yet" if not ready
                   else "no evidence the gate helps")
            )
        out["gates"][gate] = entry
    return out

def trade_quality_score(
    row: Row,
    ev_model_result: Dict[str, Any],
    calibration_curve: Optional[Dict[str, Any]],
    regime_info: Optional[Dict[str, Any]],
    kill_switch_active: bool = False,
    portfolio_blocked: bool = False,
    target_wr: float = 0.55,
    calibration_min_sample: int = 15,
    calibration_slack: float = 0.05,
    market_state_p_win: Optional[float] = None,
    use_market_state_live: bool = False,
    ml_calibration_curve: Optional[Dict[str, Any]] = None,
    regime_gate: Optional[Dict[str, Any]] = None,
    regime_gate_mode: str = "off",
) -> Dict[str, Any]:
    """Unified quality assessment for a single prospective trade."""
    result: Dict[str, Any] = {
        "pair": row.get("pair"),
        "alert_key": row.get("alert_key"),
        "direction": row.get("direction"),
    }
    # ── Layer 1: Hard vetoes ──
    if kill_switch_active:
        result["verdict"] = "BLOCKED"
        result["reason"] = "kill_switch_active"
        return result
    if portfolio_blocked:
        result["verdict"] = "BLOCKED"
        result["reason"] = "portfolio_heat_limit"
        return result

    # ── Layer 2: Probabilistic assessment ──
    p_profit = ev_model_result.get("p_ev_positive", 0.5)
    net_ev = ev_model_result.get("net_ev", 0.0)
    ev_p5 = ev_model_result.get("ev_p5", net_ev)
    n_oos = ev_model_result.get("n", 0)

    # ── Layer 2b: live per-trade market-state prediction (item #12).
    market_state_p_win_cal = market_state_p_win
    calibration_reason = None
    if market_state_p_win is not None and ml_calibration_curve:
        p_cal, calibration_reason = ml_calibration_lookup(
            ml_calibration_curve, market_state_p_win,
        )
        if p_cal is not None:
            market_state_p_win_cal = p_cal

    p_profit_effective = p_profit
    if market_state_p_win_cal is not None:
        if use_market_state_live:
            p_profit_effective = 0.5 * p_profit + 0.5 * market_state_p_win_cal
    per_trade = None
    if (market_state_p_win is not None
            and market_state_p_win_cal is not None
            and use_market_state_live):
        rr = (row.get("context") or {}).get("rr", 2.0)  # or from MAE/MFE profile
        per_trade = per_trade_ev(market_state_p_win_cal, reward_r=float(rr))
        per_trade["p_raw"] = round(market_state_p_win, 4)
        per_trade["calibration_reason"] = calibration_reason or "no_curve"
        result["per_trade_ev"] = per_trade

    # ── Layer 3: Calibration ──
    cal_wr = None
    if calibration_curve:
        conf_pct = row.get("conf_pct", 50.0)
        ok, cal_wr, reason = calibration_gate_decision(
            calibration_curve, conf_pct,
            target_wr=target_wr, min_sample=calibration_min_sample, slack=calibration_slack,
        )

        result["calibration"] = {
            "pass": ok, "calibrated_wr": cal_wr, "reason": reason,
        }
        if not ok:
            result["verdict"] = "BLOCKED"
            result["reason"] = f"calibration_gate: {reason}"
            return result

    # ── Layer 4: Regime compatibility ─
    regime_ok = True
    if regime_info and regime_info.get("valid"):
        adx = (row.get("context") or {}).get("adx_val")
        if adx is not None:
            median_adx = regime_info.get("median_adx", adx)
            regime = "trending" if adx >= median_adx else "ranging"
            reg_data = regime_info.get("regimes", {}).get(regime, {})
            if reg_data.get("valid"):
                # Always expose the regime WR (not only when it is a warning)
                # so Telegram can show "trending regime WR 64%".
                result["regime_name"] = regime
                result["regime_wr"] = round(float(reg_data.get("wr", 0.5)), 4)
            if reg_data.get("valid") and reg_data.get("wr", 0.5) < 0.40:
                regime_ok = False
                result["regime_warning"] = (
                    f"{regime} regime WR={reg_data['wr']:.0%}"
                )
    # ── Layer 5: Composite score ──
    evidence_factor = min(1.0, n_oos / 200.0)
    quality = (
        0.40 * p_profit_effective
        + 0.25 * max(0.0, min(1.0, (net_ev + 0.5) / 1.5))
        + 0.15 * evidence_factor
        + 0.10 * (0.5 if regime_ok else 0.0)
        + 0.10 * (cal_wr if cal_wr is not None else 0.5)
    )

    # ─Verdict ──
    if quality >= 0.70 and p_profit_effective >= 0.85 and ev_p5 > -0.10:
        verdict = "HIGH"
    elif quality >= 0.50 and p_profit_effective >= 0.65:
        verdict = "MEDIUM"
    else:
        verdict = "LOW"

    evidence_state = sample_evidence_state(
        n_oos,
        oos_validated=bool(ev_model_result.get("oos_validated")),
        insufficient_n=int(getattr(cfg, "SAMPLE_INSUFFICIENT_N", 15)),
        shadow_n=int(getattr(cfg, "SAMPLE_SHADOW_N", 50)),
        eligible_n=int(getattr(cfg, "SAMPLE_ELIGIBLE_N", 100)),
    )
    verdict_uncapped = verdict
    if getattr(cfg, "ENABLE_EVIDENCE_VERDICT_CAP", True):
        if evidence_state == "INSUFFICIENT":
            verdict = "LOW"
        elif evidence_state == "SHADOW" and verdict == "HIGH":
            verdict = "MEDIUM"

    result.update({
        "verdict": verdict,
        "quality_score": round(quality, 3),
        "p_ev_positive": round(p_profit, 3),
        "market_state_p_win": (
            round(market_state_p_win_cal, 3) if market_state_p_win_cal is not None else None
        ),
        "market_state_p_win_raw": (
            round(market_state_p_win, 3) if market_state_p_win is not None else None
        ),
        "market_state_live": bool(use_market_state_live and market_state_p_win_cal is not None),
        "net_ev": round(net_ev, 4),
        "ev_p5": round(ev_p5, 4),
        "n_oos": n_oos,
        "evidence_strength": (
            "strong" if n_oos >= 200
            else "moderate" if n_oos >= 50
            else "weak"
        ),
        "evidence_state": evidence_state,
        "verdict_uncapped": verdict_uncapped,
        "oos_validated": bool(ev_model_result.get("oos_validated")),
        "drift_warning": bool((ev_model_result.get("wr_drop_recent") or 0.0) >= 0.10),
        "regime_compatible": regime_ok,

        "alert_family": alert_family_of(str(row.get("alert_key") or "")),
    })

    # ── Regime gate (restrict-only; sample + OOS gated upstream) ──
    apply_regime_gate(result, regime_gate, regime_gate_mode)

    # ── Ensemble layer (roadmap #22): blend Bayesian + ML + EV + recent ──
    if getattr(cfg, "ENABLE_ENSEMBLE_DECISION", True):
        bayesian_p = ev_model_result.get("hierarchical_wr") or ev_model_result.get("bayesian_wr")
        recent_wr = ev_model_result.get("recent_wr")
        ens = ensemble_decision(
            bayesian_p=float(bayesian_p) if bayesian_p is not None else None,
            ml_p=float(market_state_p_win_cal) if market_state_p_win_cal is not None else None,
            ev_p=float(p_profit) if p_profit is not None else None,
            recent_wr=float(recent_wr) if recent_wr is not None else None,
            weights={
                "bayesian": float(getattr(cfg, "ENSEMBLE_WEIGHT_BAYESIAN", 0.30)),
                "ml": float(getattr(cfg, "ENSEMBLE_WEIGHT_ML", 0.30)),
                "ev": float(getattr(cfg, "ENSEMBLE_WEIGHT_EV", 0.25)),
                "recent": float(getattr(cfg, "ENSEMBLE_WEIGHT_RECENT", 0.15)),
            },
            n_evidence=n_oos,
            oos_validated=bool(ev_model_result.get("oos_validated")),
        )
        if ens.get("valid"):
            result["ensemble_p"] = ens["ensemble_p"]
            result["ensemble_components"] = ens.get("components")

            if ens["ensemble_p"] is not None:
                result["p_ev_positive_ensemble"] = ens["ensemble_p"]
        apply_ensemble_gate(
            result,
            str(getattr(cfg, "ENSEMBLE_GATE_MODE", "shadow")),
            int(getattr(cfg, "ENSEMBLE_GATE_MIN_N", 100)),
            float(getattr(cfg, "ENSEMBLE_GATE_MAX_P", 0.40)),
            int(n_oos),
            bool(ev_model_result.get("oos_validated")),
        )

    # Advisory size hint only — never used to place orders in this bot
    if getattr(cfg, "ENABLE_BRAIN_SIZE_HINT", False):
        if result.get("verdict") == "BLOCKED":
            hint = float(getattr(cfg, "BRAIN_SIZE_HINT_BLOCKED", 0.0))
        elif result.get("verdict") == "HIGH":
            hint = float(getattr(cfg, "BRAIN_SIZE_HINT_HIGH", 1.0))
        elif result.get("verdict") == "MEDIUM":
            hint = float(getattr(cfg, "BRAIN_SIZE_HINT_MEDIUM", 0.5))
        else:
            hint = float(getattr(cfg, "BRAIN_SIZE_HINT_LOW", 0.25))
        # Shrink further if evidence is weak
        if result.get("evidence_strength") == "weak":
            hint *= 0.5
        result["size_hint"] = round(max(0.0, min(1.0, hint)), 3)
        result["size_hint_note"] = "advisory_only"

    return result
