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


# ── Exact replay ──────────────────────────────────────────────────────────
def has_path(row: Mapping[str, Any]) -> bool:
    return bool(row.get("path_fav")) and bool(row.get("path_adv")) and bool(row.get("path_close"))


def replay_exit(
    fav: Sequence[float], adv: Sequence[float], cls: Sequence[float],
    sl: float, tp: float, horizon: int, be: float = 0.0, trail: float = 0.0,
) -> Optional[Tuple[str, float]]:
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
    if len(fav) <= horizon:
        return None
    stop = -sl
    be_on = False
    best = -1e9
    for i in range(horizon + 1):
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
    return "timeout", float(cls[horizon])


def replay_plan(
    row: Mapping[str, Any], sl_pct: float, tp_pct: float, horizon: int, default_cost_pct: float,
    be: float = 0.0, trail: float = 0.0,
) -> Optional[Tuple[str, float]]:
    """(outcome_class, net_pnl_pct) this trade WOULD have produced under the
    given plan, or None if the row has no path / too few candles."""
    if not has_path(row) or sl_pct <= 0 or tp_pct <= 0 or horizon < 0:
        return None
    fav, adv, cls = row["path_fav"], row["path_adv"], row["path_close"]
    if len(fav) <= horizon:
        return None
    cost = row.get("realized_cost_pct")
    cost = float(cost) if cost is not None else float(default_cost_pct)
    if be > 0 or trail > 0:
        res = replay_exit(fav, adv, cls, sl_pct, tp_pct, horizon, be, trail)
        return None if res is None else (res[0], res[1] - cost)
    for i in range(horizon + 1):
        hit_sl = adv[i] >= sl_pct
        hit_tp = fav[i] >= tp_pct
        if hit_sl:                      # same-candle tie -> stop (pessimistic)
            return "sl", -sl_pct - cost
        if hit_tp:
            return "tp", tp_pct - cost
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


def evaluate_plan(
    rows: Sequence[Row], sl_pct: float, tp_pct: float, horizon: int, default_cost_pct: float,
    be: float = 0.0, trail: float = 0.0,
) -> Optional[Dict[str, Any]]:
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


def grid_search(
    rows: Sequence[Row], default_cost_pct: float,
    sl_grid: Sequence[float] = DEFAULT_SL_GRID,
    tp_grid: Sequence[float] = DEFAULT_TP_GRID,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> List[Dict[str, Any]]:
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


def select_plan(
    rows: Sequence[Row], default_cost_pct: float, fixed_sl: float, fixed_tp: float, fixed_h: int,
    *, min_n: int = 40, min_holdout: int = 15, train_frac: float = 0.6, z: float = 1.645,
    test_exits: bool = True, min_exit_lift: float = 0.03,
) -> Dict[str, Any]:
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


# ── Works on legacy rows too (MAE/MFE only) ───────────────────────────────
def reach_table(rows: Sequence[Row], levels: Sequence[float] = REACH_LEVELS) -> Dict[str, Any]:
    """Share of trades whose best / worst excursion reached each level.
    mfe / mae are stored as fractions (0.01 = 1%)."""
    mfes = [abs(float(r["mfe"])) * 100.0 for r in rows if r.get("mfe") is not None]
    maes = [abs(float(r["mae"])) * 100.0 for r in rows if r.get("mae") is not None]
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
def build_plan_lab(
    rows: Sequence[Row], *, sl_pct: float, tp_pct: float, horizon: int, cost_pct: float,
    name_fn: Callable[[str], str] = str, min_n: int = 40, max_groups: int = 6,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """Plain-text 'Trade-plan lab' block for the Brain report + machine facts."""
    if not rows:
        return None, {}
    facts: Dict[str, Any] = {}
    lines: List[str] = []

    rt = reach_table(rows)
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
    lines.append("")
    lines.append("Read: VALIDATED = profitable on unseen later trades with a positive lower bound. "
                 "Nothing here is auto-applied.")
    return "\n".join(lines), facts
