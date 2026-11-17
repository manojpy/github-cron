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

KEY_SUPPRESSION = "learner_suppression_scorecard"


def default_suppression_criteria() -> Dict[str, Any]:
    return {"min_blocks": 20, "min_n": 30, "precision_target": 0.55,
            "max_opportunity_loss": 0.35, "recent_days": 14}


def _precision_lo(losers: int, n: int) -> Optional[float]:
    """One-sided 95% Wilson lower bound on the share of held-back trades that
    would have lost money. Local import keeps scoreboard's import graph flat."""
    from threshold_stats import wilson_ci
    if n <= 0:
        return None
    lo, _hi, _ = wilson_ci(losers, n)
    return lo


def build_suppression_scorecard(
    rows: Sequence[Row], *, cost_pct: float, fixed: Mapping[str, float],
    now_ts: float, criteria: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Did suppression actually help? Two questions the playbook scoreboard
    does not answer:

      suppression precision  of the alerts a gate held back, what share would
                             have LOST money on the fixed bracket after costs?
      opportunity loss       of ALL profitable opportunities (delivered plus
                             held back), what share did suppression remove?

    Population: delivered (LIVE) rows and genuinely suppressed SHADOW rows.
    Candidate-only rows (delivered but unrecorded) are NOT suppression and are
    excluded. Every reason is replayed on the SAME fixed bracket as the
    scoreboard, with the same cluster-robust SE over 3h blocks. A reason is
    EFFECTIVE only when its precision lower bound clears the target AND its
    opportunity share is bounded; COSTLY when the held-back trades provably
    earned money or it removes too much of the good book; RECOVERING when the
    recent window's held-back trades now earn money after losing before.
    Criteria are pre-committed and hashed, exactly like the playbook scoreboard.
    """
    C = dict(default_suppression_criteria())
    C.update(criteria or {})
    sl, tp, h = float(fixed["sl"]), float(fixed["tp"]), int(fixed["h"])
    cut = float(now_ts) - float(C["recent_days"]) * 86400.0


    delivered: List[Tuple[Row, float, float]] = []
    by_reason: Dict[str, List[Tuple[Row, float, float]]] = {}
    for r in rows:
        if r.get("candidate_only") or r.get("entry_ts") is None:
            continue
        src = r.get("source")
        _ctx_raw = r.get("context")
        ctx: Dict[str, Any] = _ctx_raw if isinstance(_ctx_raw, dict) else {}
        reason = r.get("rejection_reason") or ctx.get("rejection_reason")
        if src == "LIVE":
            bucket = delivered
        elif src == "SHADOW" and reason:
            bucket = by_reason.setdefault(str(reason), [])
        else:
            continue
        res = replay_plan(r, sl, tp, h, cost_pct)
        if res is not None:
            bucket.append((r, res[1], sl))

    def _wins(items: List[Tuple[Row, float, float]]) -> int:
        return sum(1 for _, p, _ in items if p > 0)

    del_wins = _wins(delivered)
    sup_items = [it for items in by_reason.values() for it in items]
    sup_wins = _wins(sup_items)
    total_wins = del_wins + sup_wins

    per: Dict[str, Any] = {}
    for reason, items in sorted(by_reason.items()):
        p = _pool(items)
        n = int(p["n"])
        wins = _wins(items)
        losers = n - wins
        precision = (losers / n) if n else None
        prec_lo = _precision_lo(losers, n)
        opp = (wins / total_wins) if total_wins else None
        recent = [it for it in items if float(it[0]["entry_ts"]) > cut]
        pr = _pool(recent)
        enough = n >= int(C["min_n"]) and p["n_blocks"] >= int(C["min_blocks"])
        if not enough:
            verdict = "COLLECTING"
        elif (pr["lcb"] is not None and pr["lcb"] > 0
              and p["ev"] is not None and p["ev"] <= 0):
            verdict = "RECOVERING"
        elif p["lcb"] is not None and p["lcb"] > 0:
            verdict = "COSTLY"
        elif opp is not None and opp > float(C["max_opportunity_loss"]):
            verdict = "COSTLY"
        elif prec_lo is not None and prec_lo >= float(C["precision_target"]):
            verdict = "EFFECTIVE"
        else:
            verdict = "NO_BENEFIT_YET"
        per[reason] = {
            "verdict": verdict, "n": n, "n_blocks": p["n_blocks"],
            "ev": p["ev"], "lcb": p["lcb"], "ucb": p["ucb"],
            "precision": precision, "precision_lo": prec_lo,
            "opportunity_loss": opp, "profitable_held_back": wins,
            "recent": {"n": pr["n"], "ev": pr["ev"], "lcb": pr["lcb"]},
        }

    pd, ps = _pool(delivered), _pool(sup_items)
    allp = _pool(delivered + sup_items)
    lift = (None if pd["ev"] is None or allp["ev"] is None
            else float(pd["ev"]) - float(allp["ev"]))
    overall_precision = ((len(sup_items) - sup_wins) / len(sup_items)) if sup_items else None
    overall_opp = (sup_wins / total_wins) if total_wins else None
    crit = [
        _crit("S1", "delivered EV beats take-everything EV (%)",
              None if lift is None else round(lift, 3), "> 0",
              lift is not None and lift > 0),
        _crit("S2", "overall suppression precision",
              None if overall_precision is None else f"{overall_precision:.0%}",
              f">= {C['precision_target']:.0%}",
              overall_precision is not None and overall_precision >= float(C["precision_target"])),
        _crit("S3", "overall opportunity loss",
              None if overall_opp is None else f"{overall_opp:.0%}",
              f"<= {C['max_opportunity_loss']:.0%}",
              overall_opp is not None and overall_opp <= float(C["max_opportunity_loss"])),
    ]
    return {
        "criteria": C, "criteria_hash": criteria_hash(C),
        "delivered": {"n": pd["n"], "ev": pd["ev"], "lcb": pd["lcb"]},
        "suppressed": {"n": ps["n"], "ev": ps["ev"], "ucb": ps["ucb"]},
        "take_everything": {"n": allp["n"], "ev": allp["ev"]},
        "suppression_lift": lift,
        "precision": overall_precision, "opportunity_loss": overall_opp,
        "reasons": per, "criteria_checks": crit,
        "status": ("NO_SUPPRESSION" if not sup_items else
                   "HELPING" if all(c["pass"] for c in crit) else "MIXED"),
        "note": ("Fixed bracket, after costs. Suppressed = shadow rows carrying a "
                 "rejection reason; candidate-only rows are excluded. A gate that "
                 "holds back winners is COSTLY however high its precision looks."),
    }


def format_suppression_scorecard(sc: Mapping[str, Any]) -> str:
    def _pct(v: Any) -> str:
        return "n/a" if v is None else f"{float(v):.0%}"

    def _ev(v: Any) -> str:
        return "n/a" if v is None else f"{float(v):+.2f}%"

    d, s, a = sc["delivered"], sc["suppressed"], sc["take_everything"]
    out = [f"🧮 SUPPRESSION [{sc.get('criteria_hash')}] — {sc.get('status')}",
           f"  delivered {d['n']} ({_ev(d['ev'])}/trade) · held back {s['n']} ({_ev(s['ev'])}/trade)",
           f"  take-everything baseline: {a['n']} trades, {_ev(a['ev'])}/trade "
           f"→ suppression lift {_ev(sc.get('suppression_lift'))}",
           f"  precision {_pct(sc.get('precision'))} · opportunity loss {_pct(sc.get('opportunity_loss'))}"]
    for c in sc.get("criteria_checks") or []:
        v = "n/a" if c["value"] is None else c["value"]
        out.append(f"  {'✅' if c['pass'] else '❌'} {c['id']} {c['label']}: {v} (need {c['target']})")
    for reason, r in (sc.get("reasons") or {}).items():
        out.append(
            f"  · {reason}: {r['verdict']} — n={r['n']}/{r['n_blocks']} blocks, "
            f"held-back EV {_ev(r['ev'])}, precision {_pct(r['precision'])} "
            f"(lo {_pct(r['precision_lo'])}), opportunity loss {_pct(r['opportunity_loss'])}"
        )
    return "\n".join(out)

def _crit(cid: str, label: str, value: Any, target: str, passed: bool) -> Dict[str, Any]:
    return {"id": cid, "label": label, "value": value, "target": target, "pass": bool(passed)}

def build_scoreboard(blob: Mapping[str, Any], rows: Sequence[Row], *, cost_pct: float, fixed: Mapping[str, float], criteria: Optional[Mapping[str, Any]] = None, family_fn: Optional[Callable[[str], str]] = None, family_pool: bool = True) -> Dict[str, Any]:
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
