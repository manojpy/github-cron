"""scoreboard — pre-committed pass criteria for the playbook.

Pure functions.  The criteria are fixed BEFORE the data is looked at (they come
from config, and their hash is stored with every scoreboard so a later change is
visible in the ledger).  Only FORWARD rows count: trades that happened after a
plan became champion (TAKE side) or after a group was proven a loser (AVOID
side), so a plan can never be graded on the data that selected it.

  TAKE   (served champion plans, pooled, each trade counted once)
    T1  independent 3-hour blocks            >= take_min_blocks
    T2  forward EV lower bound               >  0
    T3  realized win share vs predicted      within calib_tol
    T4  forward max drawdown                 <= max_dd_r stops
  AVOID  (groups restricted to AVOID, fixed plan, pooled)
    A1  independent 3-hour blocks            >= avoid_min_blocks
    A2  forward EV upper bound               <  0

READY only when every criterion passes.  Until then the playbook is a research
tool, not a promise.
"""
from __future__ import annotations

import hashlib
import json
import statistics
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from plan_replay import Row, cluster_se, max_drawdown, replay_plan
from playbook import group_rows

Z = 1.645


def default_criteria() -> Dict[str, Any]:
    return {"take_min_blocks": 60, "calib_tol": 0.10, "max_dd_r": 15.0, "avoid_min_blocks": 30}


def criteria_hash(c: Mapping[str, Any]) -> str:
    return hashlib.sha1(json.dumps(dict(c), sort_keys=True).encode()).hexdigest()[:10]


def _pool(items: List[Tuple[Row, float, float]]) -> Dict[str, Any]:
    """items = [(row, net_pnl, plan_sl)] -> pooled forward statistics."""
    items = sorted(items, key=lambda t: t[0]["entry_ts"])
    pn = [p for _, p, _ in items]
    n = len(pn)
    if n < 2:
        return {"n": n, "n_blocks": 0, "ev": None, "se": None, "lcb": None, "ucb": None, "dd_r": None, "p_profit": None}
    ev = statistics.fmean(pn)
    se, nb = cluster_se(pn, [r["entry_ts"] for r, _, _ in items])
    ok = se != float("inf")
    avg_sl = statistics.fmean([s for _, _, s in items])
    return {
        "n": n, "n_blocks": nb, "ev": ev, "se": se if ok else None,
        "lcb": ev - Z * se if ok else None, "ucb": ev + Z * se if ok else None,
        "dd_r": max_drawdown(pn) / avg_sl if avg_sl > 0 else None,
        "p_profit": sum(1 for p in pn if p > 0) / n,
    }


def _crit(cid: str, label: str, value: Any, target: str, passed: bool) -> Dict[str, Any]:
    return {"id": cid, "label": label, "value": value, "target": target, "pass": bool(passed)}


def build_scoreboard(
    blob: Mapping[str, Any], rows: Sequence[Row], *, cost_pct: float, fixed: Mapping[str, float],
    criteria: Optional[Mapping[str, Any]] = None, family_fn: Optional[Callable[[str], str]] = None,
    family_pool: bool = True,
) -> Dict[str, Any]:
    C = dict(default_criteria())
    C.update(criteria or {})
    groups = group_rows(rows, family_fn, family_pool)
    entries = blob.get("entries") or {}

    # TAKE side: each trade counted once, under the MOST SPECIFIC served entry.
    seen: set = set()
    take_items: List[Tuple[Row, float, float]] = []
    predicted: List[Tuple[float, int]] = []
    served = sorted(blob.get("served") or [], key=lambda k: (k.startswith("fam:"), k))
    for k in served:
        e = entries.get(k) or {}
        plan, since = e.get("plan"), e.get("since_ts")
        if not plan or since is None:
            continue
        hp = (e.get("holdout") or {}).get("p_profit")
        used = 0
        for r in groups.get(k, []):
            rid = (r.get("pair"), r.get("alert_key"), r.get("entry_ts"))
            if r["entry_ts"] <= since or rid in seen:
                continue
            res = replay_plan(r, float(plan["sl"]), float(plan["tp"]), int(plan["h"]),
                              cost_pct, float(plan.get("be") or 0), float(plan.get("trail") or 0))
            if res is None:
                continue
            seen.add(rid)
            used += 1
            take_items.append((r, res[1], float(plan["sl"])))
        if hp is not None and used:
            predicted.append((float(hp), used))
    t = _pool(take_items)
    pred = (sum(p * n for p, n in predicted) / sum(n for _, n in predicted)) if predicted else None
    take_crit = [
        _crit("T1", "independent 3h blocks of forward trades", t["n_blocks"], f">= {C['take_min_blocks']}",
              t["n_blocks"] >= C["take_min_blocks"]),
        _crit("T2", "forward EV lower bound (%)", None if t["lcb"] is None else round(t["lcb"], 3), "> 0",
              t["lcb"] is not None and t["lcb"] > 0),
        _crit("T3", "realized win share vs predicted",
              None if (t["p_profit"] is None or pred is None) else f"{t['p_profit']:.0%} vs {pred:.0%}",
              f"within {C['calib_tol']:.0%}",
              t["p_profit"] is not None and pred is not None and abs(t["p_profit"] - pred) <= C["calib_tol"]),
        _crit("T4", "forward max drawdown (stops)", None if t["dd_r"] is None else round(t["dd_r"], 1),
              f"<= {C['max_dd_r']:g}", t["dd_r"] is not None and t["dd_r"] <= C["max_dd_r"]),
    ]

    # AVOID side: fixed plan, forward of the moment the AVOID was issued.
    fx = (float(fixed["sl"]), float(fixed["tp"]), int(fixed["h"]))
    avoid_items: List[Tuple[Row, float, float]] = []
    seen_a: set = set()
    for k, e in sorted(entries.items(), key=lambda kv: (kv[0].startswith("fam:"), kv[0])):
        if e.get("restrict") != "AVOID" or e.get("avoid_since_ts") is None:
            continue
        for r in groups.get(k, []):
            rid = (r.get("pair"), r.get("alert_key"), r.get("entry_ts"))
            if r["entry_ts"] <= e["avoid_since_ts"] or rid in seen_a:
                continue
            res = replay_plan(r, fx[0], fx[1], fx[2], cost_pct)
            if res is None:
                continue
            seen_a.add(rid)
            avoid_items.append((r, res[1], fx[0]))
    a = _pool(avoid_items)
    avoid_crit = [
        _crit("A1", "independent 3h blocks of forward trades", a["n_blocks"], f">= {C['avoid_min_blocks']}",
              a["n_blocks"] >= C["avoid_min_blocks"]),
        _crit("A2", "forward EV upper bound (%)", None if a["ucb"] is None else round(a["ucb"], 3), "< 0",
              a["ucb"] is not None and a["ucb"] < 0),
    ]
    take_has = bool(take_items)
    avoid_has = bool(avoid_items)
    ready = take_has and all(c["pass"] for c in take_crit) and (not avoid_has or all(c["pass"] for c in avoid_crit))
    return {
        "criteria": C, "criteria_hash": criteria_hash(C), "playbook": blob.get("label"),
        "take": {"n": t["n"], "ev": t["ev"], "criteria": take_crit, "evaluated": take_has},
        "avoid": {"n": a["n"], "ev": a["ev"], "criteria": avoid_crit, "evaluated": avoid_has},
        "status": "READY" if ready else "NOT_READY",
        "note": ("Forward trades only. Brain's own calibration error (ECE) is judged in the Brain report; "
                 "T3 is an independent forward check of the plans' predicted win share."),
    }


def format_scoreboard(sb: Mapping[str, Any]) -> str:
    def line(c: Mapping[str, Any]) -> str:
        v = "n/a" if c["value"] is None else c["value"]
        return f"  {'✅' if c['pass'] else '❌'} {c['id']} {c['label']}: {v} (need {c['target']})"
    out = [f"📋 SCOREBOARD [{sb.get('criteria_hash')}] — {sb.get('status')}"]
    t, a = sb["take"], sb["avoid"]
    out.append(f"TAKE side: {t['n']} forward trade(s)" + ("" if t["evaluated"] else " — no served plan has forward data yet"))
    out += [line(c) for c in t["criteria"]]
    out.append(f"AVOID side: {a['n']} forward trade(s)" + ("" if a["evaluated"] else " — no AVOID issued yet"))
    out += [line(c) for c in a["criteria"]]
    return "\n".join(out)
