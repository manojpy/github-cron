"""plan_replay — path-aware trade-plan counterfactuals.

Pure functions, no cfg / Redis / file access, safe to import anywhere.

Every resolved outcome now carries the candle-by-candle price path measured
from the simulated fill price, in percent, in the TRADE's direction:
    path_fav[i]   best favourable excursion inside candle i   (high for a buy)
    path_adv[i]   worst adverse excursion inside candle i     (low for a buy)
    path_close[i] signed close of candle i
Index 0 is the fill candle; index h is the close h candles after the fill.

With that path any (stop, target, horizon) plan can be replayed on the SAME
historical trades, with true ordering (which level was touched first) instead
of the MAE/MFE-only approximation.

Conventions (deliberately pessimistic):
  * a candle that touches both stop and target counts as a STOP;
  * stop/target fill exactly at their level (no gap-through);
  * a plan that touches neither exits at the close of candle `horizon`.
"""
from __future__ import annotations

import json
import math
import statistics
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

Row = Dict[str, Any]

DEFAULT_SL_GRID: Tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.75, 1.0, 1.25, 1.5)
DEFAULT_TP_GRID: Tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.75, 1.0, 1.25, 1.5, 2.0)
DEFAULT_HORIZONS: Tuple[int, ...] = (4, 6, 8, 12)
BE_GRID: Tuple[float, ...] = (0.0, 0.3, 0.5, 0.75)      # move stop to breakeven after +x%
TRAIL_GRID: Tuple[float, ...] = (0.0, 0.3, 0.5)         # trail x% behind the best move
EXIT_VARIANTS = len(BE_GRID) * len(TRAIL_GRID) - 1      # every combination except plain
REACH_LEVELS: Tuple[float, ...] = (0.3, 0.5, 0.75, 1.0, 1.5, 2.0)

_CANDLE_SEC = 15 * 60


# ── Serialisation (resolver -> stream/archive -> readers) ─────────────────
def outcome_path_fields(result: Mapping[str, Any], *, stream: bool) -> Dict[str, Any]:
    """Extra fields the resolver adds to every stored outcome row.
    stream=True -> Redis-stream strings; stream=False -> native JSON values."""
    path = None
    if result.get("path_fav") is not None:
        path = {"f": result["path_fav"], "a": result["path_adv"], "c": result["path_close"]}
    scalars = {
        "outcome_class": result.get("outcome_class"),
        "tp_candle": result.get("tp_candle"),
        "sl_candle": result.get("sl_candle"),
        "mfe_candle": result.get("mfe_candle"),
        "mae_candle": result.get("mae_candle"),
        "net_pnl_hold_pct": result.get("net_pnl_hold_pct"),
        "plan_sl_pct": result.get("plan_sl_pct"),
        "plan_tp_pct": result.get("plan_tp_pct"),
        "plan_horizon": result.get("plan_horizon"),
    }
    if not stream:
        out: Dict[str, Any] = dict(scalars)
        out["path"] = path
        return out
    out = {k: ("" if v is None else str(v)) for k, v in scalars.items()}
    out["path"] = json.dumps(path, separators=(",", ":")) if path else ""
    return out


StreamField = Union[bytes, memoryview, str, int, float]


def outcome_path_stream_fields(result: Mapping[str, Any]) -> Dict[StreamField, StreamField]:
    """outcome_path_fields(stream=True) typed for Redis stream writes (every value is a string)."""
    out: Dict[StreamField, StreamField] = {}
    for k, v in outcome_path_fields(result, stream=True).items():
        out[k] = str(v)
    return out


def _opt_float(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _opt_int(v: Any) -> Optional[int]:
    f = _opt_float(v)
    return int(f) if f is not None else None


def coerce_path_fields(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """Inverse of outcome_path_fields for BOTH stream strings and archive JSON.
    Rows written before this feature simply yield Nones (never fabricated)."""
    path = raw.get("path")
    if isinstance(path, (str, bytes)):
        try:
            path = json.loads(path) if path else None
        except (TypeError, ValueError):
            path = None
    fav = adv = cls = None
    if isinstance(path, dict):
        fav, adv, cls = path.get("f"), path.get("a"), path.get("c")
        if not (isinstance(fav, list) and isinstance(adv, list) and isinstance(cls, list)
                and len(fav) == len(adv) == len(cls) and fav):
            fav = adv = cls = None
    oc = raw.get("outcome_class")
    return {
        "outcome_class": oc if oc in ("tp", "sl", "timeout") else None,
        "tp_candle": _opt_int(raw.get("tp_candle")),
        "sl_candle": _opt_int(raw.get("sl_candle")),
        "mfe_candle": _opt_int(raw.get("mfe_candle")),
        "mae_candle": _opt_int(raw.get("mae_candle")),
        "net_pnl_hold_pct": _opt_float(raw.get("net_pnl_hold_pct")),
        "plan_sl_pct": _opt_float(raw.get("plan_sl_pct")),
        "plan_tp_pct": _opt_float(raw.get("plan_tp_pct")),
        "plan_horizon": _opt_int(raw.get("plan_horizon")),
        "path_fav": fav, "path_adv": adv, "path_close": cls,
    }


def recost_rows(rows: Sequence[Dict[str, Any]], flat_cost_pct: float) -> int:
    """Re-state each row's round-trip cost as `flat_cost_pct`, in place.

    net = gross - cost, so a row costed at c_old becomes net + (c_old - flat).
    Gross P&L is untouched; only the cost line moves.  Applies to net_pnl_pct and
    net_pnl_hold_pct, and sets realized_cost_pct so every replay that prefers the
    row's own cost sees the same figure.  Rows without a stored cost are left
    alone (their consumers already fall back to the flat default).  Idempotent.
    Returns how many rows changed."""
    flat = float(flat_cost_pct)
    changed = 0
    for r in rows:
        old = r.get("realized_cost_pct")
        if old is None:
            continue
        delta = float(old) - flat
        if abs(delta) < 1e-9:
            continue
        for k in ("net_pnl_pct", "net_pnl_hold_pct"):
            v = r.get(k)
            if v is not None:
                r[k] = float(v) + delta
        r["realized_cost_pct"] = flat
        changed += 1
    return changed

# ── Exact replay ──────────────────────────────────────────────────────────

def has_path(row: Mapping[str, Any]) -> bool:
    return bool(row.get("path_fav")) and bool(row.get("path_adv")) and bool(row.get("path_close"))

def replay_exit(fav: Sequence[float], adv: Sequence[float], cls: Sequence[float], sl: float, tp: float, horizon: int, be: float = 0.0, trail: float = 0.0, *, allow_prefix: bool = False) -> Optional[Tuple[str, float]]:
    """Exact replay of a bracket that may also move its stop.

    be     once a candle's best move reaches +be% the stop moves to breakeven (0 = off)
    trail  trailing stop kept `trail`% behind the best move so far (0 = off)

    Stop moves take effect from the NEXT candle: the order of events inside one
    candle is unknown, so the replay never credits a protective stop that may not
    have existed yet.  A candle touching both stop and target counts as the stop.
    Returns (class, gross P&L %) -- cost is subtracted by the caller.  A stop that
    was moved into profit exits with that profit; class is then still "sl" (a stop
    exit).  Levels are in favourable % from the fill; adv is unclamped, so a candle
    whose low stayed above the fill has a negative adv."""
    n = len(fav)
    if n <= horizon and not allow_prefix:
        return None
    stop = -sl
    be_on = False
    best = -1e9
    for i in range(min(horizon + 1, n)):
        if -adv[i] <= stop:             # stop touched (a tie with the target goes to the stop)
            return "sl", stop
        if fav[i] >= tp:
            return "tp", tp
        if fav[i] > best:
            best = fav[i]
        if be > 0 and not be_on and fav[i] >= be:
            be_on = True
        lvl = -sl
        if be_on and lvl < 0.0:
            lvl = 0.0
        if trail > 0 and best - trail > lvl:
            lvl = best - trail
        if lvl > stop:
            stop = lvl
    if n <= horizon:
        return None                     # censored (path ends at the exit candle, no touch seen)
    return "timeout", float(cls[horizon])

def replay_plan(row: Mapping[str, Any], sl_pct: float, tp_pct: float, horizon: int, default_cost_pct: float, be: float = 0.0, trail: float = 0.0, *, allow_prefix: bool = False) -> Optional[Tuple[str, float]]:
    """(outcome_class, net_pnl_pct) this trade WOULD have produced under the
    given plan, or None if the row has no path / too few candles.

    allow_prefix=False (default, used by every plan COMPARISON: autotune, intelligence,
    playbook, scoreboard, feedback): a path shorter than the horizon returns None, so all
    plans in a comparison are evaluated on the same set of trades.
    allow_prefix=True: an early-closed trade (stop/target touched before the horizon) stores
    its path only up to the exit candle; a touch INSIDE that prefix is exact, an untouched
    prefix stays None (censored). Only safe for a single-plan, per-row label (feature_store),
    never for ranking plans against each other, because each plan would then be scored on a
    different subset of trades."""
    if not has_path(row) or sl_pct <= 0 or tp_pct <= 0 or horizon < 0:
        return None
    fav, adv, cls = row["path_fav"], row["path_adv"], row["path_close"]
    n = len(fav)
    if n <= horizon and not allow_prefix:
        return None
    cost = row.get("realized_cost_pct")
    cost = float(cost) if cost is not None else float(default_cost_pct)
    if be > 0 or trail > 0:
        res = replay_exit(fav, adv, cls, sl_pct, tp_pct, horizon, be, trail, allow_prefix=allow_prefix)
        return None if res is None else (res[0], res[1] - cost)
    for i in range(min(horizon + 1, n)):
        hit_sl = adv[i] >= sl_pct
        hit_tp = fav[i] >= tp_pct
        if hit_sl:                      # same-candle tie -> stop (pessimistic)
            return "sl", -sl_pct - cost
        if hit_tp:
            return "tp", tp_pct - cost
    if n <= horizon:
        return None                     # censored: exited before the horizon, no touch seen
    return "timeout", float(cls[horizon]) - cost


BLOCK_SEC = 3 * 3600   # alerts inside one 3-hour block share the same market move


def cluster_se(values: Sequence[float], stamps: Sequence[Optional[float]], block_sec: int = BLOCK_SEC) -> Tuple[float, int]:
    """Standard error of the mean that respects clustering in time.

    30 pairs move together, so 10 alerts from one BTC swing are closer to ONE
    observation than ten.  Rows are grouped into `block_sec` time blocks and the
    cluster-robust variance  sum_b(sum_{i in b} e_i)^2 / n^2  is used (with the
    usual nb/(nb-1) small-sample factor).  Rows without a timestamp each count
    as their own block (i.e. the plain iid formula).  Returns (se, n_blocks)."""
    n = len(values)
    if n < 2:
        return float("inf"), n
    mean = sum(values) / n
    blocks: Dict[Any, float] = {}
    for i, (v, t) in enumerate(zip(values, stamps)):
        k = int(t // block_sec) if t is not None else ("row", i)
        blocks[k] = blocks.get(k, 0.0) + (v - mean)
    nb = len(blocks)
    if nb < 2:
        return float("inf"), nb
    var = sum(b * b for b in blocks.values()) / (n * n) * (nb / (nb - 1))
    return math.sqrt(var), nb


def max_drawdown(pnls: Sequence[float]) -> float:
    """Largest peak-to-trough fall of the cumulative P&L curve (percentage points)."""
    peak = cum = dd = 0.0
    for p in pnls:
        cum += p
        peak = max(peak, cum)
        dd = max(dd, peak - cum)
    return dd

def evaluate_plan(rows: Sequence[Row], sl_pct: float, tp_pct: float, horizon: int, default_cost_pct: float, be: float = 0.0, trail: float = 0.0) -> Optional[Dict[str, Any]]:
    pnls: List[float] = []
    stamps: List[Optional[float]] = []
    counts = {"tp": 0, "sl": 0, "timeout": 0}
    ordered = sorted(rows, key=lambda r: (r.get("entry_ts") is None, r.get("entry_ts") or 0))
    for r in ordered:
        res = replay_plan(r, sl_pct, tp_pct, horizon, default_cost_pct, be, trail)
        if res is None:
            continue
        counts[res[0]] += 1
        pnls.append(res[1])
        stamps.append(r.get("entry_ts"))
    n = len(pnls)
    if n == 0:
        return None
    ev = statistics.fmean(pnls)
    se, n_blocks = cluster_se(pnls, stamps)
    half = n // 2
    half_min = min(statistics.fmean(pnls[:half]), statistics.fmean(pnls[half:])) if half >= 3 else ev
    return {
        "sl": sl_pct, "tp": tp_pct, "h": horizon, "be": be, "trail": trail, "n": n, "n_blocks": n_blocks,
        "tp_rate": counts["tp"] / n, "sl_rate": counts["sl"] / n, "timeout_rate": counts["timeout"] / n,
        "p_profit": sum(1 for p in pnls if p > 0) / n,
        "ev": ev, "se": se,
        "ev_lcb": ev - 1.645 * se if se != float("inf") else float("-inf"),
        "worst10": sorted(pnls)[max(0, int(0.1 * (n - 1)))],
        "max_dd": max_drawdown(pnls), "half_min": half_min,
    }

def grid_search(rows: Sequence[Row], default_cost_pct: float, sl_grid: Sequence[float] = DEFAULT_SL_GRID, tp_grid: Sequence[float] = DEFAULT_TP_GRID, horizons: Sequence[int] = DEFAULT_HORIZONS) -> List[Dict[str, Any]]:
    out = []
    for h in horizons:
        for sl in sl_grid:
            for tp in tp_grid:
                e = evaluate_plan(rows, sl, tp, h, default_cost_pct)
                if e:
                    out.append(e)
    return out

def exit_variants(plan_sl: float, plan_tp: float) -> List[Tuple[float, float]]:
    """(be, trail) pairs worth testing around a chosen bracket (breakeven must be
    below the target; a trail wider than the stop adds nothing)."""
    out = []
    for be in BE_GRID:
        for tr in TRAIL_GRID:
            if be == 0.0 and tr == 0.0:
                continue
            if be >= plan_tp or tr >= plan_tp:
                continue
            out.append((be, tr))
    return out

def select_plan(rows: Sequence[Row], default_cost_pct: float, fixed_sl: float, fixed_tp: float, fixed_h: int, *, min_n: int = 40, min_holdout: int = 15, train_frac: float = 0.6, z: float = 1.645, test_exits: bool = True, min_exit_lift: float = 0.03) -> Dict[str, Any]:
    """Pick the best plan on the chronological TRAIN split, then judge it on the
    untouched HOLDOUT (train rows too close to the split are embargoed).

    After the bracket is chosen, breakeven / trailing-stop variants of THAT bracket
    are tried on train only; a variant replaces the plain bracket only if it beats
    it on train by `min_exit_lift` percentage points.
    status: COLLECTING | NO_EDGE | FAILED_OOS | PROMISING | VALIDATED."""
    usable = sorted((r for r in rows if has_path(r) and r.get("entry_ts") is not None),
                    key=lambda r: r["entry_ts"])
    res: Dict[str, Any] = {"n_path": len(usable), "status": "COLLECTING"}
    if len(usable) < min_n:
        return res
    cut = int(len(usable) * train_frac)
    hold = usable[cut:]
    split_ts = hold[0]["entry_ts"] if hold else usable[-1]["entry_ts"]
    embargo = (max(DEFAULT_HORIZONS) + 2) * _CANDLE_SEC
    train = [r for r in usable[:cut] if r["entry_ts"] <= split_ts - embargo]
    if len(train) < max(10, min_n // 2) or len(hold) < min_holdout:
        return res
    cands = [c for c in grid_search(train, default_cost_pct) if c["n"] >= max(10, min_n // 2)]
    if not cands:
        return res
    best = max(cands, key=lambda c: c["ev"])
    if test_exits:
        plain, chosen = best, best
        for be, tr in exit_variants(plain["sl"], plain["tp"]):
            v = evaluate_plan(train, plain["sl"], plain["tp"], plain["h"], default_cost_pct, be, tr)
            if v and v["n"] >= plain["n"] and v["ev"] >= plain["ev"] + min_exit_lift and v["ev"] > chosen["ev"]:
                chosen = v
        best = chosen
    hold_eval = evaluate_plan(hold, best["sl"], best["tp"], best["h"], default_cost_pct,
                              best.get("be", 0.0), best.get("trail", 0.0))
    cur_all = evaluate_plan(usable, fixed_sl, fixed_tp, fixed_h, default_cost_pct)
    res.update({"best": best, "holdout": hold_eval, "current": cur_all, "n_train": len(train), "n_hold": len(hold),
                "_hold_rows": hold})
    if best["ev"] <= 0:
        res["status"] = "NO_EDGE"            # no plan in the grid makes money -> the SIGNAL is the problem
    elif hold_eval is None or hold_eval["ev"] <= 0:
        res["status"] = "FAILED_OOS"
    elif hold_eval["se"] != float("inf") and hold_eval["ev"] - z * hold_eval["se"] > 0:
        res["status"] = "VALIDATED"
    else:
        res["status"] = "PROMISING"
    return res

VOL_BUCKETS: Tuple[str, ...] = ("low", "mid", "high")

def _vol_bucket(v: float, cuts: Tuple[float, float]) -> str:
    return VOL_BUCKETS[0] if v <= cuts[0] else VOL_BUCKETS[1] if v <= cuts[1] else VOL_BUCKETS[2]

def volatility_plan_study(rows: Sequence[Row], default_cost_pct: float, fixed_sl: float, fixed_tp: float, fixed_h: int, *, min_n: int = 150, min_bucket_train: int = 25, train_frac: float = 0.6, z: float = 1.645) -> Dict[str, Any]:
    """Would a different stop / target / horizon per volatility regime beat ONE plan tuned on the same data?

    Regime = the ATR percentile stored when the alert fired (low / mid / high
    terciles, cut points taken from the OLDER trades). For each regime the best
    (stop, target, horizon) is chosen on older trades only, then both that plan
    and the fixed plan are replayed on the newer, untouched trades of the same
    regime; the comparison is per trade, paired, with a time-blocked standard
    error. The yardstick is the best SINGLE plan chosen on the same older trades,
    not the current fixed plan: any tuned plan beats an arbitrary one, so only a
    single tuned plan isolates what volatility conditioning adds. 'supported' =
    regime plans beat that single plan with a one-sided lower bound above zero.
    The gain over the current fixed plan is reported separately. Descriptive only:
    nothing is applied, and the fixed 12-candle label used everywhere else is
    untouched."""
    usable = sorted(
        (r for r in rows if has_path(r) and r.get("entry_ts") is not None
         and isinstance((r.get("context") or {}).get("atr_pctl"), (int, float))),
        key=lambda r: r["entry_ts"])
    out: Dict[str, Any] = {"status": "insufficient", "n": len(usable), "causal": False, "buckets": {}, "n_candidates": 0}
    if len(usable) < min_n:
        return out
    cut = int(len(usable) * train_frac)
    hold = usable[cut:]
    split_ts = hold[0]["entry_ts"]
    embargo = (max(DEFAULT_HORIZONS) + 2) * _CANDLE_SEC
    train = [r for r in usable[:cut] if r["entry_ts"] <= split_ts - embargo]
    if len(train) < min_n // 2 or len(hold) < min_n // 4:
        return out
    vals = sorted(float(r["context"]["atr_pctl"]) for r in train)
    cuts = (vals[len(vals) // 3], vals[(2 * len(vals)) // 3])
    out["cuts"] = [round(cuts[0], 2), round(cuts[1], 2)]

    def bucket_of(r: Row) -> str:
        return _vol_bucket(float(r["context"]["atr_pctl"]), cuts)

    g_cands = [c for c in grid_search(train, default_cost_pct) if c["n"] >= max(10, min_bucket_train // 2)]
    if not g_cands:
        return out
    g_best = max(g_cands, key=lambda c: c["ev"])
    out["single_plan"] = {"sl": g_best["sl"], "tp": g_best["tp"], "h": g_best["h"], "train_ev": round(g_best["ev"], 4)}
    paired: List[Tuple[float, float, float]] = []   # (regime plan, single tuned plan, current fixed plan) per holdout trade
    stamps: List[float] = []
    grid_size = len(DEFAULT_SL_GRID) * len(DEFAULT_TP_GRID) * len(DEFAULT_HORIZONS)
    for name in VOL_BUCKETS:
        tr = [r for r in train if bucket_of(r) == name]
        ho = [r for r in hold if bucket_of(r) == name]
        allb = [r for r in usable if bucket_of(r) == name]
        mfes = [float(max(r["path_fav"])) for r in allb]
        maes = [float(max(r["path_adv"])) for r in allb]
        tmf = [r["mfe_candle"] for r in allb if r.get("mfe_candle") is not None]
        info: Dict[str, Any] = {
            "n": len(allb), "n_train": len(tr), "n_hold": len(ho),
            "median_best_move": round(statistics.median(mfes), 3) if mfes else None,
            "median_worst_move": round(statistics.median(maes), 3) if maes else None,
            "median_candle_of_best": statistics.median(tmf) if tmf else None,
        }
        out["buckets"][name] = info
        if len(tr) < min_bucket_train or not ho:
            info["plan"] = None
            continue
        cands = [c for c in grid_search(tr, default_cost_pct) if c["n"] >= max(10, min_bucket_train // 2)]
        out["n_candidates"] += grid_size
        if not cands:
            info["plan"] = None
            continue
        best = max(cands, key=lambda c: c["ev"])
        info["plan"] = {"sl": best["sl"], "tp": best["tp"], "h": best["h"], "train_ev": round(best["ev"], 4)}
        reg: List[float] = []
        sgl: List[float] = []
        for r in ho:
            a = replay_plan(r, best["sl"], best["tp"], best["h"], default_cost_pct)
            b = replay_plan(r, g_best["sl"], g_best["tp"], g_best["h"], default_cost_pct)
            f = replay_plan(r, fixed_sl, fixed_tp, fixed_h, default_cost_pct)
            if a is None or b is None or f is None:
                continue
            reg.append(a[1])
            sgl.append(b[1])
            paired.append((a[1], b[1], f[1]))
            stamps.append(float(r["entry_ts"]))
        if reg:
            info["hold_ev_regime"] = round(sum(reg) / len(reg), 4)
            info["hold_ev_single"] = round(sum(sgl) / len(sgl), 4)
    if len(paired) < max(20, min_n // 5):
        return out
    delta = [a - b for a, b, _f in paired]
    d_mean = sum(delta) / len(delta)
    se, nb = cluster_se(delta, stamps)
    lo = None if se == float("inf") else d_mean - z * se
    hi = None if se == float("inf") else d_mean + z * se
    out.update({
        "n_hold_pairs": len(paired), "overall_delta": round(d_mean, 4),
        "gain_over_fixed": round(sum(a - f for a, _b, f in paired) / len(paired), 4),
        "single_over_fixed": round(sum(b - f for _a, b, f in paired) / len(paired), 4),
        "overall_lo": None if lo is None else round(lo, 4), "n_blocks": nb,
        "fixed_plan": {"sl": fixed_sl, "tp": fixed_tp, "h": fixed_h},
        "status": "insufficient" if lo is None else "supported" if lo > 0 else "worse" if (hi is not None and hi < 0) else "not_distinguishable",

    })
    return out

# ── Partial exits and limit entries (descriptive replays; nothing here is applied) ─────────
SCALE_T1_GRID: Tuple[float, ...] = (0.5, 0.75, 1.0)    # first slice is taken at +x%
SCALE_FRAC_GRID: Tuple[float, ...] = (0.33, 0.5, 0.67) # share of the position taken at that first level
LIMIT_OFFSET_GRID: Tuple[float, ...] = (0.1, 0.2, 0.3, 0.5)  # limit order sits x% better than the market fill
LIMIT_WAIT_GRID: Tuple[int, ...] = (1, 2, 4)           # candles the limit order is left working
LIMIT_THROUGH_PCT = 0.02                               # price must trade this far THROUGH the limit to count as filled


def replay_scaleout(fav: Sequence[float], adv: Sequence[float], cls: Sequence[float], sl: float, tp: float, horizon: int, t1: float, frac: float, rest_be: bool = True) -> Optional[Tuple[str, float]]:
    """Exact replay of a bracket that sells `frac` of the position at +t1% and lets the rest run.

    The rest keeps the target `tp`; its stop is the original stop, or breakeven when
    rest_be.  The stop move takes effect from the NEXT candle (order inside a candle is
    unknown).  A candle touching the stop counts as the stop for everything still open,
    even when it also touched a target.  A candle that reaches `tp` without touching the
    stop has passed through t1 as well, so both slices fill.  Costs are one round trip on
    the whole position, so the caller subtracts the usual single cost.
    Returns (class, gross P&L %): tp | sl | timeout | partial (first slice taken, rest
    stopped, breakeven or timed out), or None when the path is shorter than the horizon."""
    if len(fav) <= horizon or not (0.0 < frac < 1.0) or not (0.0 < t1 < tp):
        return None
    got = False
    stop = -sl
    for i in range(horizon + 1):
        if -adv[i] <= stop:
            return ("partial", frac * t1 + (1.0 - frac) * stop) if got else ("sl", stop)
        if fav[i] >= tp:
            return "tp", frac * t1 + (1.0 - frac) * tp
        if not got and fav[i] >= t1:
            got = True
            if rest_be:
                stop = 0.0
    last = float(cls[horizon])
    return ("partial", frac * t1 + (1.0 - frac) * last) if got else ("timeout", last)


def replay_limit(fav: Sequence[float], adv: Sequence[float], cls: Sequence[float], sl: float, tp: float, horizon: int, offset: float, wait: int, through: float = LIMIT_THROUGH_PCT) -> Optional[Tuple[str, float]]:
    """Exact replay of entering with a limit order `offset`% better than the market fill.

    The order works for `wait` candles and fills when the adverse move reaches
    offset + through (price must trade through the limit, not just touch it).  Stop and
    target are then measured from the limit price.  Pessimistic by construction: in the
    fill candle only the stop can fire (the target cannot be credited, the order of events
    inside the candle is unknown); the stop is checked there because price had to cross the
    limit on its way down to it.  Fills at the limit price, never better.  The path after
    the fill is the real one, so the adverse selection of limit orders (they fill most
    often just before price keeps moving against you) is already in the result.
    Returns ("nofill", 0.0) when the order never filled, else (class, gross P&L % on the
    limit price); the exit candle stays the market plan's, so a late fill has less time."""
    if len(fav) <= horizon or offset <= 0.0 or offset >= tp or wait < 1:
        return None
    need = offset + through
    fill = -1
    for i in range(min(wait, horizon + 1)):
        if adv[i] >= need:
            fill = i
            break
    if fill < 0:
        return "nofill", 0.0
    stop_lvl = offset + sl
    tp_lvl = tp - offset
    if adv[fill] >= stop_lvl:
        return "sl", -sl
    for j in range(fill + 1, horizon + 1):
        if adv[j] >= stop_lvl:
            return "sl", -sl
        if fav[j] >= tp_lvl:
            return "tp", tp
    return "timeout", float(cls[horizon]) + offset


def _row_cost(row: Mapping[str, Any], default_cost_pct: float) -> float:
    c = row.get("realized_cost_pct")
    return float(c) if c is not None else float(default_cost_pct)


def _full_path_rows(rows: Sequence[Row], horizon: int) -> List[Row]:
    return sorted((r for r in rows if has_path(r) and r.get("entry_ts") is not None and len(r["path_fav"]) > horizon),
                  key=lambda r: r["entry_ts"])


def _paired_summary(base: Sequence[float], new: Sequence[float], stamps: Sequence[float], z: float) -> Dict[str, Any]:
    """Per-trade paired comparison of two P&L series (same trades, same order)."""
    n = len(base)
    delta = [b - a for a, b in zip(base, new)]
    d_mean = sum(delta) / n
    se, nb = cluster_se(delta, stamps)
    ok = se != float("inf")
    lo = d_mean - z * se if ok else None
    hi = d_mean + z * se if ok else None
    return {
        "n": n, "n_blocks": nb, "ev_base": sum(base) / n, "ev_new": sum(new) / n,
        "delta": d_mean, "lo": lo, "hi": hi,
        "green_base": sum(1 for p in base if p > 0) / n, "green_new": sum(1 for p in new if p > 0) / n,
        "dd_base": max_drawdown(base), "dd_new": max_drawdown(new),
        "status": "insufficient" if lo is None else "supported" if lo > 0 else "worse" if (hi is not None and hi < 0) else "not_distinguishable",
    }


def _split_for_study(usable: Sequence[Row], train_frac: float) -> Tuple[List[Row], List[Row]]:
    cut = int(len(usable) * train_frac)
    hold = list(usable[cut:])
    if not hold:
        return [], []
    split_ts = hold[0]["entry_ts"]
    embargo = (max(DEFAULT_HORIZONS) + 2) * _CANDLE_SEC
    return [r for r in usable[:cut] if r["entry_ts"] <= split_ts - embargo], hold


def _scale_net(r: Row, sl: float, tp: float, h: int, t1: float, frac: float, be: bool, cost: float) -> Optional[Tuple[str, float]]:
    res = replay_scaleout(r["path_fav"], r["path_adv"], r["path_close"], sl, tp, h, t1, frac, be)
    return None if res is None else (res[0], res[1] - cost)


def scaleout_study(rows: Sequence[Row], default_cost_pct: float, fixed_sl: float, fixed_tp: float, fixed_h: int, *, min_n: int = 80, min_hold: int = 25, train_frac: float = 0.6, z: float = 1.645) -> Dict[str, Any]:
    """Would taking part of the profit early (and protecting the rest) beat the plain bracket?

    Two views, both paired per trade against the plain fixed plan on the SAME trades:
      reference  one fixed, pre-stated variant (half off at +1%, breakeven for the rest)
                 on every trade; nothing was fitted, so every trade is out of sample.
      tuned      the best (first level, share, breakeven yes/no) on older trades only, then
                 judged on the newer, untouched ones with a time-blocked lower bound.
    Descriptive only: nothing is applied, and the recorded outcome label is unchanged."""
    usable = _full_path_rows(rows, fixed_h)
    out: Dict[str, Any] = {"status": "insufficient", "n": len(usable), "fixed_plan": {"sl": fixed_sl, "tp": fixed_tp, "h": fixed_h}}
    if len(usable) < min_n:
        return out
    stamps = [float(r["entry_ts"]) for r in usable]
    base = [replay_plan(r, fixed_sl, fixed_tp, fixed_h, default_cost_pct) for r in usable]
    ref_t1 = min(SCALE_T1_GRID[-1], fixed_tp * 0.5)
    ref = [_scale_net(r, fixed_sl, fixed_tp, fixed_h, ref_t1, 0.5, True, _row_cost(r, default_cost_pct)) for r in usable]
    pairs = [(b[1], x[1], t) for b, x, t in zip(base, ref, stamps) if b is not None and x is not None]
    if pairs:
        out["reference"] = _paired_summary([p[0] for p in pairs], [p[1] for p in pairs], [p[2] for p in pairs], z)
        out["reference"].update({"t1": ref_t1, "frac": 0.5, "be": True})
    train, hold = _split_for_study(usable, train_frac)
    if len(train) < max(20, min_n // 2) or len(hold) < min_hold:
        return out
    cands = [(t1, f, be) for t1 in SCALE_T1_GRID if t1 < fixed_tp for f in SCALE_FRAC_GRID for be in (True, False)]
    out["n_candidates"] = len(cands)
    best: Optional[Tuple[float, Tuple[float, float, bool]]] = None
    for c in cands:
        d = []
        for r in train:
            b = replay_plan(r, fixed_sl, fixed_tp, fixed_h, default_cost_pct)
            x = _scale_net(r, fixed_sl, fixed_tp, fixed_h, c[0], c[1], c[2], _row_cost(r, default_cost_pct))
            if b is not None and x is not None:
                d.append(x[1] - b[1])
        if len(d) >= max(20, min_n // 2):
            m = sum(d) / len(d)
            if best is None or m > best[0]:
                best = (m, c)
    if best is None:
        return out
    t1, f, be = best[1]
    hp: List[Tuple[float, float, float]] = []
    for r in hold:
        b = replay_plan(r, fixed_sl, fixed_tp, fixed_h, default_cost_pct)
        x = _scale_net(r, fixed_sl, fixed_tp, fixed_h, t1, f, be, _row_cost(r, default_cost_pct))
        if b is not None and x is not None:
            hp.append((b[1], x[1], float(r["entry_ts"])))
    if len(hp) < min_hold:
        return out
    hs = _paired_summary([p[0] for p in hp], [p[1] for p in hp], [p[2] for p in hp], z)
    out.update({"best": {"t1": t1, "frac": f, "be": be, "train_delta": round(best[0], 4)}, "hold": hs,
                "n_train": len(train), "status": hs["status"]})
    return out


def limit_entry_study(rows: Sequence[Row], default_cost_pct: float, fixed_sl: float, fixed_tp: float, fixed_h: int, *, entry_saving_pct: float = 0.0, min_n: int = 80, min_hold: int = 25, min_train_fills: int = 15, train_frac: float = 0.6, z: float = 1.645) -> Dict[str, Any]:
    """Would a limit-order entry beat entering at market, signal for signal?

    Every comparison is PER SIGNAL: a limit order that never fills earns 0 on that
    signal, so missed winners are charged.  entry_saving_pct is the cost a limit entry
    saves (entry slippage, plus any maker-fee discount); the exit is still costed in full.
    Skipping a signal is NOT counted as an improvement: in a market where the plain plan
    loses after costs, trading less looks better by itself, which says nothing about the
    entry.  So a signal that never fills is charged the round-trip cost the market trade
    would have paid (its gross P&L is lost, its cost is not saved); ev_literal reports the
    other accounting (unfilled earns 0) and skip_saving the difference.
    Same two views as scaleout_study: a fixed reference (limit 0.2% better, left working
    2 candles) on every trade, and a (offset, wait) tuned on older trades then judged on
    newer ones.  Descriptive only."""
    usable = _full_path_rows(rows, fixed_h)
    out: Dict[str, Any] = {"status": "insufficient", "n": len(usable), "entry_saving_pct": entry_saving_pct,
                           "fixed_plan": {"sl": fixed_sl, "tp": fixed_tp, "h": fixed_h}}
    if len(usable) < min_n:
        return out

    def one(r: Row, off: float, wait: int) -> Optional[Tuple[str, float]]:
        res = replay_limit(r["path_fav"], r["path_adv"], r["path_close"], fixed_sl, fixed_tp, fixed_h, off, wait)
        if res is None:
            return None
        if res[0] == "nofill":
            return "nofill", -_row_cost(r, default_cost_pct)
        return res[0], res[1] - max(0.0, _row_cost(r, default_cost_pct) - entry_saving_pct)

    def summarise(rs: Sequence[Row], off: float, wait: int) -> Optional[Dict[str, Any]]:
        trip: List[Tuple[float, float, float, bool, bool]] = []
        for r in rs:
            b = replay_plan(r, fixed_sl, fixed_tp, fixed_h, default_cost_pct)
            x = one(r, off, wait)
            if b is not None and x is not None:
                trip.append((b[1], x[1], float(r["entry_ts"]), x[0] != "nofill", b[0] == "tp"))
        if len(trip) < 2:
            return None
        s = _paired_summary([t[0] for t in trip], [t[1] for t in trip], [t[2] for t in trip], z)
        filled = [t for t in trip if t[3]]
        missed = [t for t in trip if not t[3]]
        s.update({
            "offset": off, "wait": wait, "fill_rate": len(filled) / len(trip), "n_filled": len(filled),
            "skip_saving": sum(-t[1] for t in missed) / len(trip) if missed else 0.0,
            "ev_filled": (sum(t[1] for t in filled) / len(filled)) if filled else None,
            "ev_base_same": (sum(t[0] for t in filled) / len(filled)) if filled else None,
            "missed_target_share": (sum(1 for t in missed if t[4]) / len(missed)) if missed else None,
        })
        return s

    ref = summarise(usable, 0.2, 2)
    if ref:
        out["reference"] = ref
    train, hold = _split_for_study(usable, train_frac)
    if len(train) < max(20, min_n // 2) or len(hold) < min_hold:
        return out
    cands = [(o, w) for o in LIMIT_OFFSET_GRID if o < fixed_tp for w in LIMIT_WAIT_GRID]
    out["n_candidates"] = len(cands)
    best: Optional[Tuple[float, Tuple[float, int]]] = None
    for c in cands:
        s = summarise(train, c[0], c[1])
        if s and s["n_filled"] >= min_train_fills and (best is None or s["delta"] > best[0]):
            best = (s["delta"], c)
    if best is None:
        return out
    hs = summarise(hold, best[1][0], best[1][1])
    if hs is None or hs["n"] < min_hold:
        return out
    out.update({"best": {"offset": best[1][0], "wait": best[1][1], "train_delta": round(best[0], 4)}, "hold": hs,
                "n_train": len(train), "status": hs["status"]})
    return out


def _study_verdict(st: str, what: str) -> str:
    return {
        "supported": f"Verdict: {what} beat the plain plan on unseen trades, with the lower bound above zero.",
        "worse": f"Verdict: {what} did worse than the plain plan on unseen trades.",
        "not_distinguishable": f"Verdict: no proven difference in average result versus the plain plan ({what}).",
    }.get(st, "Verdict: not enough trades yet.")


def scaleout_lines(st: Mapping[str, Any]) -> List[str]:
    if not st or not st.get("n"):
        return []
    fp = st["fixed_plan"]
    out = [f"PARTIAL EXIT: take part of the profit early (n={st['n']} trades with full price paths; plain plan = stop {fp['sl']:g}% / target {fp['tp']:g}% / {fp['h']} candles)"]
    r = st.get("reference")
    if r:
        out.append(f"  Fixed test, no fitting: sell half at +{r['t1']:g}%, then breakeven stop for the rest, target stays {fp['tp']:g}%.")
        out.append(f"  Trades ending in profit after costs {r['green_base']:.0%} → {r['green_new']:.0%} | average per trade {r['ev_base']:+.2f}% → {r['ev_new']:+.2f}%"
                   f" | worst cumulative fall {r['dd_base']:.1f}% → {r['dd_new']:.1f}%")
    b, h = st.get("best"), st.get("hold")
    if b and h:
        be_txt = "breakeven stop for the rest" if b["be"] else "original stop for the rest"
        out.append(f"  Best variant on older trades: sell {b['frac']:.0%} at +{b['t1']:g}%, {be_txt}. On newer unseen trades (n={h['n']}):")
        lo = "n/a" if h["lo"] is None else f"{h['lo']:+.2f}%"
        out.append(f"  profit {h['green_base']:.0%} → {h['green_new']:.0%} | average {h['ev_base']:+.2f}% → {h['ev_new']:+.2f}% (gain {h['delta']:+.2f}%, lower bound {lo})")
        out.append("  " + _study_verdict(st["status"], "the tuned partial exit"))
    elif r:
        out.append("  Tuned test: not enough trades yet.")
    return out


def limit_lines(st: Mapping[str, Any]) -> List[str]:
    if not st or not st.get("n"):
        return []
    out = [f"LIMIT-ORDER ENTRY instead of market (n={st['n']} trades with full price paths; per signal; limit entry credited {st['entry_saving_pct']:.2f}% lower cost)"]
    r = st.get("reference")
    if r:
        miss = "" if r["missed_target_share"] is None else f", {r['missed_target_share']:.0%} of the unfilled signals would have reached the target"
        out.append(f"  Fixed test, no fitting: limit {r['offset']:g}% better, left working {r['wait']} candles. Filled {r['fill_rate']:.0%}{miss}.")
        out.append(f"  Average per signal {r['ev_base']:+.2f}% → {r['ev_new']:+.2f}% | trades ending in profit after costs (per signal) {r['green_base']:.0%} → {r['green_new']:.0%}")
        out.append(f"  Signals that never fill are charged the cost they would have paid, so skipping a signal is not mistaken for a better entry "
                   f"(counting them as simply not taken would add {r['skip_saving']:+.2f}%).")
        if r["ev_filled"] is not None:
            out.append(f"  On the filled trades alone: market {r['ev_base_same']:+.2f}% → limit {r['ev_filled']:+.2f}% "
                       f"(flatters limit orders: the trades that dip first are the ones a market entry does worst on).")
    b, h = st.get("best"), st.get("hold")
    if b and h:
        out.append(f"  Best on older trades: limit {b['offset']:g}% better, {b['wait']} candles. On newer unseen signals (n={h['n']}): filled {h['fill_rate']:.0%}")
        lo = "n/a" if h["lo"] is None else f"{h['lo']:+.2f}%"
        out.append(f"  average per signal {h['ev_base']:+.2f}% → {h['ev_new']:+.2f}% (gain {h['delta']:+.2f}%, lower bound {lo})")
        out.append("  " + _study_verdict(st["status"], "the tuned limit entry"))
    elif r:
        out.append("  Tuned test: not enough trades yet.")
    return out

# ── Works on legacy rows too (MAE/MFE only) ───────────────────────────────
def row_excursions(row: Mapping[str, Any], min_len: int = 0) -> Optional[Tuple[float, float]]:
    """(best move, worst move against) in %, over the FULL outcome window, or None.
    A trade that hit its stop (or target) early is graded on the path up to that
    candle only, so its stored mfe / mae stop at the exit: the worst move can never
    show much beyond the stop and the best move misses any later rally. Those rows
    are only used when a full-length path is stored (min_len candles, rebuilt from
    exchange history by pathrecon); otherwise they are skipped, not guessed."""
    if has_path(row) and len(row["path_fav"]) >= max(1, min_len):
        return (max(0.0, max(float(x) for x in row["path_fav"])),
                max(0.0, max(float(x) for x in row["path_adv"])))
    if row.get("early_close") or row.get("mfe") is None or row.get("mae") is None:
        return None
    return abs(float(row["mfe"])) * 100.0, abs(float(row["mae"])) * 100.0


def reach_table(rows: Sequence[Row], levels: Sequence[float] = REACH_LEVELS, min_len: int = 0) -> Dict[str, Any]:
    """Share of trades whose best / worst excursion reached each level.
    Excursions come from row_excursions (full window only; min_len = full path length)."""
    ex = [e for e in (row_excursions(r, min_len) for r in rows) if e is not None]
    mfes = [e[0] for e in ex]
    maes = [e[1] for e in ex]
    if not mfes or not maes:
        return {}
    return {
        "n": len(mfes),
        "mfe": {lv: sum(1 for m in mfes if m >= lv) / len(mfes) for lv in levels},
        "mae": {lv: sum(1 for m in maes if m >= lv) / len(maes) for lv in levels},
        "median_mfe": statistics.median(mfes), "median_mae": statistics.median(maes),
    }


def diagnose_row(row: Mapping[str, Any], sl_pct: float, tp_pct: float) -> str:
    """Why did a non-winning trade fail?  Returns one of:
    WIN, stop_too_tight, reversed_after_progress, signal_failed,
    target_too_ambitious, slow_drift.   mfe/mae are fractions."""
    cls = row.get("outcome_class")
    if cls is None:
        r = row.get("outcome_reason")
        cls = "tp" if r in ("target_hit", "target_hit_ever") else \
              "sl" if r in ("stop_hit", "stop_hit_ever", "ambiguous_same_candle") else "timeout"
    if cls == "tp":
        return "WIN"
    mfe = abs(float(row.get("mfe") or 0.0)) * 100.0
    if cls == "sl":
        fav = row.get("path_fav")
        sl_c = row.get("sl_candle")
        if fav and sl_c is not None:
            reached_later = max(fav[sl_c:] or [0.0]) >= tp_pct
        else:
            reached_later = mfe >= tp_pct
        if reached_later:
            return "stop_too_tight"
        if mfe >= 0.5 * tp_pct:
            return "reversed_after_progress"
        return "signal_failed"
    if mfe >= 0.6 * tp_pct:
        return "target_too_ambitious"
    if mfe < 0.25 * tp_pct:
        return "signal_failed"
    return "slow_drift"


_DIAG_LABELS = {
    "signal_failed": "signal never moved your way",
    "reversed_after_progress": "moved your way, then reversed into the stop",
    "stop_too_tight": "stopped out, then price reached the target",
    "target_too_ambitious": "got most of the way to target, timed out",
    "slow_drift": "drifted, no decisive move",
}


def plan_exit_suffix(p: Mapping[str, Any]) -> str:
    """' | breakeven at +0.5% | trail 0.3%' for plans that move their stop, else ''."""
    bits = []
    if float(p.get("be") or 0) > 0:
        bits.append(f"stop to breakeven at +{float(p['be']):g}%")
    if float(p.get("trail") or 0) > 0:
        bits.append(f"trail {float(p['trail']):g}%")
    return (" | " + " | ".join(bits)) if bits else ""

# ── Report text ───────────────────────────────────────────────────────────
def build_plan_lab(rows: Sequence[Row], *, sl_pct: float, tp_pct: float, horizon: int, cost_pct: float, name_fn: Callable[[str], str] = str, min_n: int = 40, max_groups: int = 6, entry_saving_pct: float = 0.0) -> Tuple[Optional[str], Dict[str, Any]]:
    """Plain-text 'Trade-plan lab' block for the Brain report + machine facts."""
    if not rows:
        return None, {}
    facts: Dict[str, Any] = {}
    lines: List[str] = []

    rt = reach_table(rows, min_len=horizon + 1)
    if rt:
        facts["reach"] = rt
        lv = list(rt["mfe"].keys())
        lines.append(f"HOW FAR TRADES ACTUALLY MOVE (n={rt['n']}, {horizon} candles)")
        lines.append("Best move reached:  " + "  ".join(f"≥{x:g}% {rt['mfe'][x]:.0%}" for x in lv))
        lines.append("Worst move against: " + "  ".join(f"≥{x:g}% {rt['mae'][x]:.0%}" for x in lv))
        lines.append(f"Median best {rt['median_mfe']:.2f}% | median worst {rt['median_mae']:.2f}% "
                     f"| your plan: stop {sl_pct:g}% / target {tp_pct:g}%")

    diag: Dict[str, int] = {}
    losers = 0
    for r in rows:
        d = diagnose_row(r, sl_pct, tp_pct)
        if d == "WIN":
            continue
        losers += 1
        diag[d] = diag.get(d, 0) + 1
    if losers:
        facts["diagnosis"] = diag
        lines.append("")
        lines.append(f"WHY NON-WINNERS FAILED (n={losers})")
        for k, v in sorted(diag.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {v / losers:>4.0%}  {_DIAG_LABELS.get(k, k)}")

    groups: Dict[str, List[Row]] = {}
    for r in rows:
        if has_path(r):
            groups.setdefault(f"{r.get('alert_key', '?')}|{r.get('direction', '?')}", []).append(r)
    n_path = sum(len(v) for v in groups.values())
    lines.append("")
    lines.append(f"BEST TRADE PLAN PER ALERT (exact path replay; {n_path}/{len(rows)} rows have a stored path)")
    shown = 0
    plans: Dict[str, Any] = {}
    for key, grp in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        sel = select_plan(grp, cost_pct, sl_pct, tp_pct, horizon, min_n=min_n)
        plans[key] = sel
        if sel["status"] == "COLLECTING" or shown >= max_groups:
            continue
        shown += 1
        ak, d = key.split("|", 1)
        b, h_, cur = sel["best"], sel["holdout"], sel.get("current")
        icon = {"VALIDATED": "🟢", "PROMISING": "🟡", "FAILED_OOS": "🔴", "NO_EDGE": "🔴"}[sel["status"]]
        if sel["status"] == "NO_EDGE":
            lines.append(f"{icon} {name_fn(ak)}: NO plan in the grid is profitable → the SIGNAL has no edge here (n={sel['n_path']})")
            continue
        lines.append(
            f"{icon} {name_fn(ak)}: stop {b['sl']:g}% / target {b['tp']:g}% / {b['h']} candles"
            f"{plan_exit_suffix(b)} | "
            f"train EV {b['ev']:+.2f}% → holdout {(h_['ev'] if h_ else float('nan')):+.2f}% "
            f"(n={sel['n_hold']}) | now {(cur['ev'] if cur else float('nan')):+.2f}% | {sel['status']}"
        )
    facts["plans"] = plans
    if shown == 0:
        need = min_n
        best_have = max((len(v) for v in groups.values()), default=0)
        lines.append(f"  Collecting: need ≥{need} path rows per alert; best alert has {best_have}. "
                     f"Path capture started with the new resolver, so this fills in as alerts resolve.")
    for key, study, fmt in (
        ("scaleout", lambda: scaleout_study(rows, cost_pct, sl_pct, tp_pct, horizon), scaleout_lines),
        ("limit", lambda: limit_entry_study(rows, cost_pct, sl_pct, tp_pct, horizon, entry_saving_pct=entry_saving_pct), limit_lines),
    ):
        try:                                   # a failing study must not take the rest of the lab down
            st = study()
            facts[key] = st
            extra = fmt(st)
        except Exception:
            extra = []
        if extra:
            lines.append("")
            lines.extend(extra)
    lines.append("")
    lines.append("Read: VALIDATED = profitable on unseen later trades with a positive lower bound. "
                 "Nothing here is auto-applied.")
    return "\n".join(lines), facts
