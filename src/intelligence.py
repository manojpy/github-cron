"""intelligence — the learner's coordinator (v2): diagnosis, evidence, and the
closed-loop checks on the playbook veto and the AVOID delivery firewall.

Pure functions (no Redis, no cfg, no network, no files): everything is passed in,
so the logic is deterministic and replayable.  It adds NO new optimiser; it reads
what the existing modules already produced (playbook, scoreboard, autopsy,
hypothesis pools) and answers, every learner cycle:

  1. Is each veto helping?   `veto_monitor` judges every AVOID group SEPARATELY
     (rule-specific rollback).  For a group it compares the trades the playbook
     restricts (forward of the moment that AVOID was issued) with the delivered
     trades, on the fixed SL/TP bracket after costs, using the scoreboard's
     cluster-robust standard error.  The comparison is stratified by session and
     direction (a veto that mostly rejects Asian-session trades is compared with
     delivered Asian-session trades, not with the whole book) and the confidence
     level is Bonferroni-adjusted for the number of groups judged.
        H1  the restricted trades would have EARNED money (lower bound > 0)
        H2  the restricted trades did significantly BETTER than matched delivered ones
     "No demonstrated benefit" is NOT harm: it keeps the veto and keeps collecting.
     Harm must be seen on `harm_cycles` consecutive cycles to suspend ONE group's
     veto; it resumes only after `resume_cycles` cycles of proven benefit.
  2. May AVOID alerts be suppressed?   `firewall_contract` is the delivery-readiness
     contract: fresh data, outcome coverage, enough independent blocks, a proven
     loser group (upper bound < 0, stable across both halves of time), a
     measurable improvement of the delivered book and a bounded opportunity cost.
     Only verdicts named in `candidates` (default AVOID) can arm; arming needs
     consecutive passes, disarming is immediate.  TAKE is never equated with
     quality here, and WATCH keeps being delivered unless explicitly configured.
  3. Is the delivered population improving?   `delivered_trend`.
  4. What needs attention?   `build_report` ranks issues with an evidence payload
     and an automation-safety flag; `update_issue_state` fingerprints them so a
     recurring issue is not re-sent on every run; `ledger_append` keeps the
     promotion ledger of every automatic change.

The only things this layer can do to production are (a) suspend the veto of one
AVOID group (it then logs but still sends) and (b) arm the firewall for a proven
verdict group; both are restrict-only reversals/restrictions the alert path
applies, and both are published through the playbook blob.  Neither edits config.
"""
from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from hypothesis_gen import row_features, rules_hit
from plan_replay import Row, has_path, replay_plan
from playbook import group_rows
from scoreboard import _pool as _fwd_pool
from threshold_stats import row_net_pnl_pct

KEY_VETO_MON = "intel_veto_monitor"
KEY_INTEL = "intel_report"
KEY_ISSUES = "intel_issues"
KEY_PROMO_LEDGER = "intel_ledger"
KEY_FIREWALL = "intel_firewall"
KEY_WIN_POOL = "intel_win_pool"

Z = 1.645
_DAY = 86400
Items = List[Tuple[Row, float, float]]
_SEV_ORDER = {"HIGH": 0, "MED": 1, "LOW": 2, "INFO": 3}
_SEV_ICON = {"HIGH": "🔴", "MED": "🟠", "LOW": "🟡", "INFO": "🔵"}


def default_monitor_params() -> Dict[str, Any]:
    return {"min_blocks": 20, "harm_cycles": 2, "resume_cycles": 4, "alpha": 0.05, "min_cell": 5}


def default_firewall_params() -> Dict[str, Any]:
    return {"min_blocks": 30, "arm_cycles": 2, "z_arm": 2.33, "max_share": 0.4, "max_age_h": 36.0, "min_path_share": 0.6, "min_half": 8}


def _z(alpha: float, m: int) -> float:
    return NormalDist().inv_cdf(1.0 - alpha / max(1, int(m)))


def _rid(r: Mapping[str, Any]) -> Tuple[Any, Any, Any]:
    return (r.get("pair"), r.get("alert_key"), r.get("entry_ts"))


def _rnd(v: Any, nd: int = 3) -> Optional[float]:
    return None if v is None else round(float(v), nd)


def _brief(p: Mapping[str, Any]) -> Dict[str, Any]:
    return {"n": int(p["n"]), "n_blocks": int(p["n_blocks"]), "ev": _rnd(p["ev"]), "lcb": _rnd(p["lcb"]),
            "ucb": _rnd(p["ucb"]), "p_profit": _rnd(p["p_profit"])}


def _gap(a: Mapping[str, Any], b: Mapping[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """(a.ev - b.ev, its standard error, one-sided lower bound at Z) for two independent samples."""
    if a["ev"] is None or b["ev"] is None or a["se"] is None or b["se"] is None:
        return None, None, None
    g = float(a["ev"]) - float(b["ev"])
    se = math.hypot(float(a["se"]), float(b["se"]))
    return g, se, g - Z * se


def _fixed_tuple(fixed: Mapping[str, float]) -> Tuple[float, float, int]:
    return float(fixed["sl"]), float(fixed["tp"]), int(fixed["h"])


def _stratified_gap(a: Items, b: Items, min_cell: int = 5) -> Tuple[Optional[float], Optional[float], int]:
    """Inverse-variance weighted mean of (ev_a - ev_b) over (session, direction) cells where
    BOTH samples have at least `min_cell` trades.  Returns (gap, standard error, cells used)."""
    ca: Dict[Tuple[str, str], Items] = {}
    cb: Dict[Tuple[str, str], Items] = {}
    for src, dst in ((a, ca), (b, cb)):
        for it in src:
            dst.setdefault((str(it[0].get("session") or "?"), str(it[0].get("direction") or "?")), []).append(it)
    num = den = 0.0
    used = 0
    for key, ia in ca.items():
        ib = cb.get(key)
        if not ib or len(ia) < min_cell or len(ib) < min_cell:
            continue
        pa, pb = _fwd_pool(ia), _fwd_pool(ib)
        if pa["se"] is None or pb["se"] is None or pa["ev"] is None or pb["ev"] is None:
            continue
        var = float(pa["se"]) ** 2 + float(pb["se"]) ** 2
        if var <= 0:
            continue
        w = 1.0 / var
        num += w * (float(pa["ev"]) - float(pb["ev"]))
        den += w
        used += 1
    if used == 0 or den <= 0:
        return None, None, 0
    return num / den, math.sqrt(1.0 / den), used


def restricted_by_group(
    blob: Mapping[str, Any], rows: Sequence[Row], *, fixed: Mapping[str, float], cost_pct: float,
    family_fn: Optional[Callable[[str], str]] = None,
) -> Tuple[Dict[str, Items], Items, Dict[str, float]]:
    """(per-AVOID-group forward trades, delivered trades, {group: avoid_since_ts}); items are
    (row, net_pnl_pct, sl_pct).  A trade belongs to one group only (specific key before family).
    Forward = after that group's AVOID was issued.  While a veto is live these are the
    'playbook_veto' shadow rows; while it is suspended they are real delivered alerts, so the
    test never goes blind.  Delivered = real (LIVE) trades since the first AVOID that are in no group."""
    entries = blob.get("entries") or {}
    since = {k: float(e["avoid_since_ts"]) for k, e in entries.items()
             if e.get("restrict") == "AVOID" and e.get("avoid_since_ts") is not None}
    if not since:
        return {}, [], {}
    sl, tp, h = _fixed_tuple(fixed)
    groups = group_rows(rows, family_fn)
    seen: Set[Tuple[Any, Any, Any]] = set()
    by_group: Dict[str, Items] = {}
    for k in sorted(since, key=lambda kk: (kk.startswith("fam:"), kk)):
        items: Items = []
        for r in groups.get(k, []):
            rid = _rid(r)
            if r["entry_ts"] <= since[k] or rid in seen:
                continue
            res = replay_plan(r, sl, tp, h, cost_pct)
            if res is None:
                continue
            seen.add(rid)
            items.append((r, res[1], sl))
        by_group[k] = items
    t0 = min(since.values())
    delivered: Items = []
    for r in rows:
        if r.get("source") != "LIVE" or r.get("entry_ts") is None or not has_path(r):
            continue
        if r["entry_ts"] <= t0 or _rid(r) in seen:
            continue
        res = replay_plan(r, sl, tp, h, cost_pct)
        if res is not None:
            delivered.append((r, res[1], sl))
    return by_group, delivered, since


def veto_monitor(
    blob: Mapping[str, Any], rows: Sequence[Row], *, fixed: Mapping[str, float], cost_pct: float,
    prev_state: Optional[Mapping[str, Any]], now_ts: float, params: Optional[Mapping[str, Any]] = None,
    family_fn: Optional[Callable[[str], str]] = None, data_ok: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Returns (result, new_state).  `result["suspended"]` maps every AVOID group whose veto SHOULD be
    suspended now to {since, reason}; the caller decides (by mode) whether to act on it.
    With data_ok=False nothing changes (verdict DATA_QUALITY): no confident call on bad data."""
    P = default_monitor_params()
    P.update(params or {})
    prev_groups: Dict[str, Any] = dict((prev_state or {}).get("groups") or {})
    by_group, delivered, since = restricted_by_group(blob, rows, fixed=fixed, cost_pct=cost_pct, family_fn=family_fn)
    pools = {k: _fwd_pool(items) for k, items in by_group.items()}
    min_b = int(P["min_blocks"])
    evaluable = [k for k, p in pools.items() if p["se"] is not None and p["n_blocks"] >= min_b]
    z_b = _z(float(P["alpha"]), len(evaluable))

    out_groups: Dict[str, Any] = {}
    new_groups: Dict[str, Any] = {}
    changed: List[Dict[str, str]] = []
    for k in sorted(since):
        p = pools[k]
        ps = prev_groups.get(k) or {}
        susp = bool(ps.get("suspended"))
        was = susp
        s_since: Optional[int] = int(ps["since"]) if ps.get("since") else None
        reason = str(ps.get("reason") or "")
        hs, cs = int(ps.get("harm_streak") or 0), int(ps.get("clear_streak") or 0)
        comp = [it for it in delivered if it[0]["entry_ts"] > since[k]]
        gap, gap_se, cells = _stratified_gap(by_group[k], comp, int(P["min_cell"]))
        gap_lcb = None if gap is None or gap_se is None else gap - z_b * gap_se
        enough = k in evaluable
        harm_profit = bool(enough and p["ev"] is not None and p["se"] is not None and p["ev"] - z_b * p["se"] > 0)
        harm_anti = bool(enough and gap_lcb is not None and gap_lcb > 0)
        harm = harm_profit or harm_anti
        helping = bool(enough and p["ucb"] is not None and p["ucb"] < 0)
        harm_text = ""
        if harm_profit:
            harm_text = (f"restricted trades would have EARNED {p['ev']:+.2f}% per trade "
                         f"({p['n_blocks']} independent blocks, confidence adjusted for {max(1, len(evaluable))} rule(s))")
        elif harm_anti and gap is not None:
            harm_text = (f"restricted trades did {gap:+.2f}% per trade BETTER than matched delivered ones "
                         f"({cells} session/direction cell(s))")
        event: Optional[str] = None
        if not data_ok:
            verdict = "DATA_QUALITY"
        else:
            if harm:
                hs, cs = hs + 1, 0
            else:
                hs = 0
                cs = cs + 1 if helping else 0
            if not susp and hs >= int(P["harm_cycles"]):
                susp, s_since, reason, event = True, int(now_ts), harm_text, "SUSPENDED"
            elif susp and cs >= int(P["resume_cycles"]):
                susp, s_since, reason, event = False, None, "", "RESUMED"
            elif susp and harm_text:
                reason = harm_text
            verdict = "HARMFUL" if harm else "HELPING" if helping else "NO_BENEFIT_YET" if enough else "COLLECTING"
        if event:
            changed.append({"key": k, "event": event, "reason": reason})
        new_groups[k] = {"suspended": susp, "since": s_since, "reason": reason, "harm_streak": hs, "clear_streak": cs}
        out_groups[k] = {"verdict": verdict, "suspended": susp, "since": s_since, "reason": reason, "changed": event,
                         "harm_streak": hs, "clear_streak": cs, "restricted": _brief(p), "gap": _rnd(gap), "gap_lcb": _rnd(gap_lcb),
                         "cells": cells, "was_suspended": was,
                         "tests": {"restricted_earns_money": harm_profit, "restricted_beats_matched_delivered": harm_anti,
                                   "restricted_proven_loser": helping}}
    all_items = [it for items in by_group.values() for it in items]
    verdicts = [g["verdict"] for g in out_groups.values()]
    pooled = ("NO_AVOID" if not out_groups else "DATA_QUALITY" if not data_ok else "HARMFUL" if "HARMFUL" in verdicts
              else "HELPING" if verdicts and all(v == "HELPING" for v in verdicts)
              else "NO_BENEFIT_YET" if "NO_BENEFIT_YET" in verdicts else "COLLECTING")
    result: Dict[str, Any] = {
        "verdict": pooled, "n_avoid_groups": len(out_groups), "groups": out_groups, "changed": changed,
        "suspended": {k: {"since": g["since"], "reason": g["reason"]} for k, g in new_groups.items() if g["suspended"]},
        "harm_cycles": int(P["harm_cycles"]), "resume_cycles": int(P["resume_cycles"]), "min_blocks": min_b,
        "z_adj": round(z_b, 2), "restricted": _brief(_fwd_pool(all_items)), "delivered": _brief(_fwd_pool(delivered)),
    }
    return result, {"groups": new_groups, "last_ts": int(now_ts)}


def delivered_trend(
    rows: Sequence[Row], *, fixed: Mapping[str, float], cost_pct: float, now_ts: float, days: int = 7, min_blocks: int = 8,
) -> Dict[str, Any]:
    """Real (LIVE) trades of the last `days` days against the `days` before, on the fixed bracket."""
    sl, tp, h = _fixed_tuple(fixed)
    cut, start = now_ts - days * _DAY, now_ts - 2 * days * _DAY
    recent: Items = []
    prior: Items = []
    for r in rows:
        ts = r.get("entry_ts")
        if r.get("source") != "LIVE" or ts is None or not has_path(r) or ts <= start:
            continue
        res = replay_plan(r, sl, tp, h, cost_pct)
        if res is None:
            continue
        (recent if ts > cut else prior).append((r, res[1], sl))
    a, b = _fwd_pool(recent), _fwd_pool(prior)
    gap, se, gap_lcb = _gap(a, b)
    if gap is None or se is None or gap_lcb is None or a["n_blocks"] < min_blocks or b["n_blocks"] < min_blocks:
        verdict = "INSUFFICIENT"
    else:
        verdict = "IMPROVING" if gap_lcb > 0 else "DETERIORATING" if gap + Z * se < 0 else "NO_CLEAR_CHANGE"
    return {"days": days, "recent": _brief(a), "prior": _brief(b), "gap": _rnd(gap), "verdict": verdict}


# ── Delivery firewall: readiness contract ─────────────────────────────────────

def verdict_items(rows: Sequence[Row], *, fixed: Mapping[str, float], cost_pct: float) -> Dict[str, Items]:
    """Trades grouped by the advisor verdict that was stored BEFORE the outcome was known.
    Population = delivered (LIVE) plus the alerts the firewall itself dropped ('verdict_firewall'),
    so the evidence keeps flowing after the firewall goes live."""
    sl, tp, h = _fixed_tuple(fixed)
    out: Dict[str, Items] = {}
    for r in rows:
        src = r.get("source")
        if not (src == "LIVE" or (src == "SHADOW" and r.get("rejection_reason") == "verdict_firewall")):
            continue
        ctx = r.get("context")
        seen = ctx.get("advisor_seen") if isinstance(ctx, dict) else None
        v = seen.get("verdict") if isinstance(seen, dict) else None
        if not v or r.get("entry_ts") is None or not has_path(r):
            continue
        res = replay_plan(r, sl, tp, h, cost_pct)
        if res is not None:
            out.setdefault(str(v), []).append((r, res[1], sl))
    return out


def _half_means(items: Items, min_half: int) -> Optional[Tuple[float, float]]:
    s = sorted(items, key=lambda it: it[0]["entry_ts"])
    mid = len(s) // 2
    a, b = [x[1] for x in s[:mid]], [x[1] for x in s[mid:]]
    if len(a) < min_half or len(b) < min_half:
        return None
    return sum(a) / len(a), sum(b) / len(b)


def firewall_contract(
    groups: Mapping[str, Items], *, now_ts: float, newest_ts: Optional[float], n_rows: int, n_path: int,
    prev_state: Optional[Mapping[str, Any]], candidates: Sequence[str] = ("AVOID",), params: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Which verdict groups may be suppressed.  Returns (result, new_state); result["armed"] lists them."""
    P = default_firewall_params()
    P.update(params or {})
    ps = dict(prev_state or {})
    prev_armed: Dict[str, Any] = dict(ps.get("armed") or {})
    prev_streak: Dict[str, int] = {k: int(v) for k, v in (ps.get("streak") or {}).items()}
    all_items = [it for items in groups.values() for it in items]
    pa = _fwd_pool(all_items)
    fresh = newest_ts is not None and (now_ts - float(newest_ts)) <= float(P["max_age_h"]) * 3600.0
    coverage = n_rows > 0 and n_path / n_rows >= float(P["min_path_share"])
    per: Dict[str, Any] = {}
    armed: Dict[str, Any] = {}
    streak: Dict[str, int] = {}
    changed: List[Dict[str, str]] = []
    for v in candidates:
        items = list(groups.get(v, []))
        others = [it for it in all_items if it not in items] if items else all_items
        p, po = _fwd_pool(items), _fwd_pool(others)
        ucb_arm = None if p["ev"] is None or p["se"] is None else float(p["ev"]) + float(P["z_arm"]) * float(p["se"])
        halves = _half_means(items, int(P["min_half"]))
        agg_before, agg_after = sum(x[1] for x in all_items), sum(x[1] for x in others)
        share = len(items) / len(all_items) if all_items else 1.0
        checks = {
            "fresh_data": bool(fresh),
            "outcome_coverage": bool(coverage),
            "enough_blocks": p["n_blocks"] >= int(P["min_blocks"]),
            "proven_loser": bool(ucb_arm is not None and ucb_arm < 0),
            "stable_over_time": bool(halves is not None and halves[0] < 0 and halves[1] < 0),
            "improves_book": bool(items and po["ev"] is not None and pa["ev"] is not None and po["ev"] > pa["ev"] and agg_after > agg_before),
            "bounded_opportunity_cost": share <= float(P["max_share"]),
        }
        eligible = all(checks.values())
        st = prev_streak.get(v, 0) + 1 if eligible else 0
        hold = bool(fresh and coverage and p["n_blocks"] >= int(P["min_blocks"]) and p["ucb"] is not None and p["ucb"] < 0)
        was = v in prev_armed
        now_armed = hold if was else (eligible and st >= int(P["arm_cycles"]))
        event: Optional[str] = None
        if now_armed and not was:
            event = "ARMED"
        elif was and not now_armed:
            event = "DISARMED"
        if now_armed:
            armed[v] = {"since": (prev_armed.get(v) or {}).get("since") or int(now_ts)}
        streak[v] = st
        sess: Dict[str, Any] = {}
        for sname in sorted({str(it[0].get("session") or "?") for it in items}):
            si = [it for it in items if str(it[0].get("session") or "?") == sname]
            if len(si) >= 5:
                sess[sname] = {"n": len(si), "ev": _rnd(sum(x[1] for x in si) / len(si))}
        per[v] = {"checks": checks, "eligible": eligible, "armed": now_armed, "streak": st, "event": event,
                  "group": _brief(p), "others": _brief(po), "book_ev": _rnd(pa["ev"]), "ev_after": _rnd(po["ev"]),
                  "agg_before": _rnd(agg_before, 2), "agg_after": _rnd(agg_after, 2), "removed_share": _rnd(share, 3),
                  "win_share": _rnd(p["p_profit"]), "sessions": sess}
        if event:
            changed.append({"verdict": v, "event": event})
    result = {"armed": sorted(armed), "per_verdict": per, "changed": changed, "fresh": bool(fresh), "coverage": bool(coverage)}
    return result, {"armed": armed, "streak": streak, "last_ts": int(now_ts)}


# ── Skip-rule lift test ───────────────────────────────────────────────────────

def default_lift_params() -> Dict[str, Any]:
    return {"alpha": 0.05, "min_matched": 30, "min_blocks": 15, "max_share": 0.4, "min_cell": 5}


def rule_lift(
    rules: Sequence[Mapping[str, Any]], rows: Sequence[Row], *, fixed: Mapping[str, float], cost_pct: float,
    params: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Which confirmed skip rules actually make the delivered book better?
    The hypothesis search confirms a rule when its later trades lose money (mean < 0).  When the whole book
    loses money after costs, almost any large slice 'loses money', so that test cannot tell a useful rule from a
    slice that is merely typical.  Here each rule is judged against the REST of the delivered book over the same
    period, stratified by session and direction, at a Bonferroni-adjusted confidence:
      keep  <=>  enough independent blocks AND matched trades lose money AND the matched-minus-rest gap has an
                 upper bound below zero (matched trades are significantly WORSE than the rest).
    Kept rules are then added best-first until together they would have removed `max_share` of the book.
    Population: delivered (LIVE) trades plus the trades a live hypothesis veto dropped (shadow-tracked), so a rule
    keeps being judged after it starts blocking."""
    P = default_lift_params()
    P.update(params or {})
    sl, tp, h = _fixed_tuple(fixed)
    pop: List[Tuple[Row, Dict[str, Any], float, bool]] = []          # (row, features, net pnl, is_live)
    for r in rows:
        src = r.get("source")
        live = src == "LIVE"
        if not (live or (src == "SHADOW" and r.get("rejection_reason") == "hypothesis_veto")):
            continue
        if r.get("entry_ts") is None or not has_path(r) or not isinstance(r.get("context"), dict):
            continue
        res = replay_plan(r, sl, tp, h, cost_pct)
        if res is not None:
            pop.append((r, row_features(r), res[1], live))
    per: Dict[str, Any] = {}
    matched_ids: Dict[str, Set[Tuple[Any, Any, Any]]] = {}
    stage: Dict[str, Tuple[Items, Items]] = {}
    for rule in rules:
        rid = str(rule["id"])
        since = float(rule.get("since_ts") or 0)
        rdict = dict(rule)
        m_items: Items = []
        rest: Items = []
        ids: Set[Tuple[Any, Any, Any]] = set()
        for r, feats, pnl, live in pop:
            if r["entry_ts"] <= since:
                continue
            if rules_hit([rdict], feats) is not None:
                m_items.append((r, pnl, sl))
                ids.add(_rid(r))
            elif live:
                rest.append((r, pnl, sl))
        stage[rid], matched_ids[rid] = (m_items, rest), ids
    evaluable = [rid for rid, (m, _) in stage.items()
                 if len(m) >= int(P["min_matched"]) and _fwd_pool(m)["n_blocks"] >= int(P["min_blocks"])]
    z_b = _z(float(P["alpha"]), len(evaluable))
    scored: List[Tuple[float, str]] = []
    for rid, (m_items, rest) in stage.items():
        pm = _fwd_pool(m_items)
        gap, gse, cells = _stratified_gap(m_items, rest, int(P["min_cell"]))
        gap_ucb = None if gap is None or gse is None else gap + z_b * gse
        if rid not in evaluable:
            verdict = "COLLECTING"
        elif gap is None:
            verdict = "UNTESTABLE"
        elif pm["ev"] is not None and float(pm["ev"]) < 0 and gap_ucb is not None and gap_ucb < 0:
            verdict = "LIFT"
            scored.append((gap, rid))
        else:
            verdict = "NO_LIFT"
        per[rid] = {"verdict": verdict, "matched": _brief(pm), "gap": _rnd(gap), "gap_ucb": _rnd(gap_ucb), "cells": cells,
                    "rest_ev": _rnd(_fwd_pool(rest)["ev"])}
    scored.sort()
    keep: List[str] = []
    union: Set[Tuple[Any, Any, Any]] = set()
    base = {_rid(r) for r, _f, _p, _l in pop}
    for _gap_v, rid in scored:
        cand = union | matched_ids[rid]
        if base and len(cand) / len(base) > float(P["max_share"]):
            per[rid]["verdict"] = "OVER_COVERAGE"
            continue
        union = cand
        keep.append(rid)
    return {"keep": keep, "per_rule": per, "union_share": _rnd(len(union) / len(base), 3) if base else None, "z_adj": round(z_b, 2),
            "n_rules": len(per)}


# ── Take-side candidates (weekly search) ──────────────────────────────────────

def win_candidates(pool: Optional[Mapping[str, Any]], rows: Sequence[Row], cost_pct: float, limit: int = 5) -> List[Dict[str, Any]]:
    """Confirmed take-side rules with their forward lift over the whole delivered book (the baseline
    policy) over the same period.  The pool stores P&L negated (a rule that 'loses' on the negated
    series wins), so the rule mean is -fwd.mean.  Report only; nothing is applied."""
    out: List[Dict[str, Any]] = []
    live = [r for r in rows if r.get("source") == "LIVE" and r.get("entry_ts") is not None]
    for rec in ((pool or {}).get("confirmed") or {}).values():
        since = float(rec.get("since_ts") or 0)
        base: List[float] = []
        for r in live:
            if r["entry_ts"] <= since:
                continue
            try:
                base.append(row_net_pnl_pct(r, cost_pct))
            except (KeyError, TypeError, ValueError):
                continue
        fwd = rec.get("fwd") or {}
        mean = fwd.get("mean")
        if mean is None:
            continue
        b = sum(base) / len(base) if base else None
        out.append({"id": rec.get("id"), "text": rec.get("text"), "n": fwd.get("n"), "rule_ev": _rnd(-float(mean)),
                    "baseline_ev": _rnd(b), "lift": None if b is None else _rnd(-float(mean) - b),
                    "parent": rec.get("parent"), "registered_ts": rec.get("since_ts")})
    out.sort(key=lambda x: -(x["lift"] if x["lift"] is not None else -9))
    return out[:limit]


# ── Issues, persistence, ledger ───────────────────────────────────────────────

def _issue(sev: str, code: str, text: str, action: str, *, subject: str = "", evidence: Optional[Mapping[str, Any]] = None,
           auto_safe: bool = False) -> Dict[str, Any]:
    return {"id": f"{code}:{subject}" if subject else code, "sev": sev, "code": code, "text": text, "action": action,
            "evidence": dict(evidence or {}), "auto_safe": bool(auto_safe)}


def _ev_line(e: Mapping[str, Any]) -> str:
    bits = []
    if e.get("n") is not None:
        bits.append(f"n={e['n']}")
    if e.get("n_blocks") is not None:
        bits.append(f"{e['n_blocks']} blocks")
    if e.get("ev") is not None:
        bits.append(f"EV {e['ev']:+.2f}%")
    if e.get("lcb") is not None:
        bits.append(f"LCB {e['lcb']:+.2f}%")
    if e.get("ucb") is not None:
        bits.append(f"UCB {e['ucb']:+.2f}%")
    return ", ".join(bits)


def build_report(
    *, now_ts: float, blob: Mapping[str, Any], sb: Optional[Mapping[str, Any]], autopsy: Optional[Mapping[str, Any]],
    hyp_pool: Optional[Mapping[str, Any]], veto: Optional[Mapping[str, Any]], trend: Optional[Mapping[str, Any]],
    n_rows: int, n_path: int, prev_ts: Optional[float] = None, max_age_h: float = 36.0, min_path_share: float = 0.6,
    veto_mode: str = "off", firewall: Optional[Mapping[str, Any]] = None, firewall_mode: str = "off",
    autotune: Optional[Mapping[str, Any]] = None, win: Optional[Sequence[Mapping[str, Any]]] = None,
    newest_ts: Optional[float] = None, data_ok: bool = True, hyp_lift: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """One ranked issue list (with evidence and an automation-safety flag) from what the learner computed."""
    issues: List[Dict[str, Any]] = []

    if not data_ok:
        issues.append(_issue("HIGH", "DATA_QUALITY", f"Only {n_path}/{n_rows} archived rows have a candle path "
                             f"({(n_path / n_rows if n_rows else 0):.0%}); veto/firewall judgements are PAUSED (no confident call on thin data).",
                             "Check outcome archiving and path recovery; nothing is changed meanwhile.",
                             evidence={"n": n_rows, "with_path": n_path}))
    if newest_ts is not None and (now_ts - float(newest_ts)) / 3600.0 > max_age_h:
        issues.append(_issue("HIGH", "DATA_STALE", f"The newest delivered trade is {(now_ts - float(newest_ts)) / 3600.0:.0f}h old (limit {max_age_h:g}h).",
                             "Check the bot cron and the outcome archive; firewall evidence is stale.", evidence={"age_h": round((now_ts - float(newest_ts)) / 3600.0, 1)}))

    for k, g in ((veto or {}).get("groups") or {}).items():
        if g["verdict"] == "HARMFUL":
            stance = ("SUSPENDED (veto logs but still sends)" if g["suspended"] and veto_mode == "live"
                      else "would suspend" if g["suspended"] else f"harm seen {g['harm_streak']}/{veto.get('harm_cycles') if veto else '?'} cycle(s)")
            issues.append(_issue("HIGH", "VETO_HARMFUL", f"Veto '{k}' looks harmful: {g['reason'] or 'see monitor'} — {stance}.",
                                 "The monitor already acted on this rule only; set VETO_MONITOR_MODE=shadow to stop it acting." if veto_mode == "live"
                                 else "Set VETO_MONITOR_MODE=live to let the monitor suspend this one rule, or ALERT_VETO_MODE=shadow to stop all vetoes.",
                                 subject=k, evidence=g["restricted"], auto_safe=True))
        elif g["suspended"]:
            issues.append(_issue("MED", "VETO_SUSPENDED", f"Veto '{k}' is suspended and waits for proof it helps again ({g['clear_streak']}/{veto.get('resume_cycles') if veto else '?'} clear cycles).",
                                 "No action needed; it resumes by itself.", subject=k, evidence=g["restricted"], auto_safe=True))
        elif g["verdict"] == "NO_BENEFIT_YET":
            issues.append(_issue("LOW", "VETO_NO_BENEFIT_YET", f"Veto '{k}' has no demonstrated benefit yet (enough data, but the restricted trades are not proven losers). It keeps shadow-tracking.",
                                 "Nothing to do; it is not harmful.", subject=k, evidence=g["restricted"]))

    if prev_ts and (now_ts - float(prev_ts)) / 3600.0 > max_age_h:
        gap_h = (now_ts - float(prev_ts)) / 3600.0
        issues.append(_issue("HIGH", "LEARNER_GAP", f"The learner did not run for {gap_h:.0f}h (limit {max_age_h:g}h): the veto failed open (sent everything) in that gap.",
                             "Check the Cronjobs.org job for learner.yml and its last HTTP response.", evidence={"gap_h": round(gap_h, 1)}))

    if trend and trend.get("verdict") == "DETERIORATING":
        r, p = trend["recent"], trend["prior"]
        issues.append(_issue("HIGH", "DELIVERED_WORSE", f"Delivered alerts got worse: EV {r['ev']:+.2f}% per trade in the last {trend['days']}d vs {p['ev']:+.2f}% before.",
                             "Open the Brain report's per-alert verdicts; look for the alert family that changed.", evidence=r))

    if firewall:
        for v, d in (firewall.get("per_verdict") or {}).items():
            if d["event"]:
                issues.append(_issue("MED", f"FIREWALL_{d['event']}", f"Delivery firewall {d['event']} for verdict {v} ({firewall_mode} mode).",
                                     "Restrict-only; disarms automatically when the group stops being a proven loser.", subject=v, evidence=d["group"], auto_safe=True))
            elif not d["armed"]:
                failed = [c for c, ok in d["checks"].items() if not ok]
                issues.append(_issue("INFO", "FIREWALL_NOT_READY", f"Firewall for {v} not armed; unmet: {', '.join(failed) or 'consecutive-pass requirement'}.",
                                     "Wait for more forward trades; nothing to change.", subject=v, evidence=d["group"]))

    if autotune:
        st = autotune.get("status")
        if st in ("ROLLED_BACK", "APPLIED", "KEPT", "EXTERNAL_CHANGE"):
            issues.append(_issue("MED" if st in ("ROLLED_BACK", "EXTERNAL_CHANGE") else "INFO", f"AUTOTUNE_{st}", str(autotune.get("text") or st),
                                 "Recorded in the promotion ledger.", evidence=autotune.get("evidence") or {}, auto_safe=True))
        elif st == "WRITE_FAILED":
            issues.append(_issue("HIGH", "AUTOTUNE_WRITE_FAILED", str(autotune.get("text") or st),
                                 "The config-override write failed; nothing changed and it will be retried next cycle.", evidence=autotune.get("evidence") or {}))
        elif st in ("WOULD_APPLY",):
            issues.append(_issue("INFO", "AUTOTUNE_WOULD_APPLY", str(autotune.get("text") or st),
                                 "Shadow mode: set AUTOTUNE_MODE=live to let it apply this single bounded tightening.", evidence=autotune.get("evidence") or {}))

    ctl = blob.get("control") or {}
    if not ctl.get("available"):
        issues.append(_issue("MED", "NO_CONTROL", f"No matched control baseline ({ctl.get('note') or 'unavailable'}): no plan can pass beyond PROMISING and no AVOID can be issued.",
                             "Check the candle fetch in the learner run (DELTA_API_BASE, rate limits)."))

    summ = blob.get("summary") or {}
    decaying = int(summ.get("RECONFIRM_DUE", 0)) + int(summ.get("EXPIRED", 0)) + int(summ.get("DEMOTED", 0))
    if decaying:
        issues.append(_issue("MED", "CHAMPION_DECAY", f"{decaying} playbook entr{'y is' if decaying == 1 else 'ies are'} demoted, expired or overdue for re-confirmation.",
                             "Nothing to do: they stop being served automatically. They are re-proven on new data.", evidence={"n": decaying}, auto_safe=True))

    for x in [x for x in ((autopsy or {}).get("rejections") or {}).get("reasons", []) if x.get("status") == "COSTLY"][:3]:
        issues.append(_issue("MED", "COSTLY_GATE", f"Gate '{x['reason']}' rejected trades that did BETTER than the delivered ones (n={x['n']}, mean {x['mean_net']:+.2f}%).",
                             "Investigate this gate; it is report-only, nothing was changed.", subject=str(x["reason"]), evidence={"n": x["n"], "ev": x["mean_net"]}))

    confirmed = (hyp_pool or {}).get("confirmed") or {}
    if confirmed:
        per = (hyp_lift or {}).get("per_rule") or {}
        n_keep = len((hyp_lift or {}).get("keep") or [])
        counts: Dict[str, int] = {}
        for d in per.values():
            counts[d["verdict"]] = counts.get(d["verdict"], 0) + 1
        top = [str(v.get("text", "?")) for k, v in confirmed.items() if (per.get(k) or {}).get("verdict") == "LIFT"][:3]
        issues.append(_issue("INFO", "HYPOTHESES_CONFIRMED",
                             f"{len(confirmed)} self-generated 'skip when' rule(s) passed forward testing; {n_keep} also beat the REST of the delivered book "
                             f"({', '.join(f'{v} {k}' for k, v in sorted(counts.items())) or 'not judged'})" + (": " + "; ".join(top) if top else "") + ".",
                             "Only rules that beat the rest of the book are published to the alert path, and only block if ALERT_HYPOTHESIS_VETO_MODE=live.",
                             evidence={"n": len(confirmed)}))

    for c in (win or [])[:3]:
        issues.append(_issue("INFO", "TAKE_SIDE_CANDIDATE", f"Take-side candidate: {c.get('text')} — forward EV {c.get('rule_ev'):+.2f}% vs delivered book {c.get('baseline_ev') if c.get('baseline_ev') is not None else 'n/a'}% "
                             f"(lift {c.get('lift') if c.get('lift') is not None else 'n/a'}, n={c.get('n')}).",
                             "Report only. A candidate needs your approval before it can change any alert.", subject=str(c.get("id")),
                             evidence={"n": c.get("n"), "ev": c.get("rule_ev")}))

    if sb and sb.get("status") != "READY":
        failed = [c["id"] for side in ("take", "avoid") for c in (sb.get(side) or {}).get("criteria", []) if not c.get("pass")]
        issues.append(_issue("INFO", "SCOREBOARD_NOT_READY", f"Scoreboard NOT_READY; unmet: {', '.join(failed) or 'no served plan yet'}.",
                             "Wait for more forward trades; nothing to change."))

    tm = (autopsy or {}).get("trust") or {}
    if tm:
        issues.append(_issue("INFO", "TRUST_LEVEL", f"TAKE trust level L{tm.get('level', 0)} ({tm.get('status', '?')}), "
                             f"{tm.get('n_with_verdict', 0)} resolved trade(s) carry a stored verdict.",
                             "TAKE is not yet a promise until the level reaches L3."))

    issues.sort(key=lambda i: _SEV_ORDER.get(i["sev"], 9))
    return {"ts": int(now_ts), "playbook": blob.get("label"), "status": issues[0]["sev"] if issues else "OK", "issues": issues,
            "trend": trend, "veto": veto, "firewall": firewall, "autotune": autotune, "win": list(win or [])}


def update_issue_state(prev: Optional[Mapping[str, Any]], issues: Sequence[Mapping[str, Any]], now_ts: float,
                       keep_days: int = 7) -> Tuple[Dict[str, Any], List[str]]:
    """Fingerprint persistence: returns (new_state, ids that are NEW or escalated in severity)."""
    prev = dict(prev or {})
    new: Dict[str, Any] = {}
    fresh: List[str] = []
    cur: Set[str] = set()
    for i in issues:
        iid = str(i["id"])
        cur.add(iid)
        p = prev.get(iid)
        if p and not p.get("resolved_ts"):
            new[iid] = {**p, "last_seen": int(now_ts), "sev": i["sev"], "count": int(p.get("count", 1)) + 1}
            if _SEV_ORDER.get(str(i["sev"]), 9) < _SEV_ORDER.get(str(p.get("sev")), 9):
                fresh.append(iid)
        else:
            new[iid] = {"first_seen": int(now_ts), "last_seen": int(now_ts), "sev": i["sev"], "count": 1, "code": i["code"]}
            fresh.append(iid)
    for iid, p in prev.items():
        if iid in cur:
            continue
        if p.get("resolved_ts"):
            if now_ts - float(p["resolved_ts"]) <= keep_days * _DAY:
                new[iid] = p
        else:
            new[iid] = {**p, "resolved_ts": int(now_ts)}
    return new, fresh


def ledger_event(kind: str, subject: str, text: str, now_ts: float, **extra: Any) -> Dict[str, Any]:
    """One promotion-ledger record: what changed, why, and the evidence at the time."""
    return {"ts": int(now_ts), "kind": kind, "subject": subject, "text": text, **extra}


def ledger_append(prev: Optional[Sequence[Mapping[str, Any]]], events: Sequence[Mapping[str, Any]], cap: int = 300) -> List[Dict[str, Any]]:
    out = [dict(e) for e in events] + [dict(e) for e in (prev or [])]
    return out[:cap]


def format_report(rep: Mapping[str, Any], *, veto_mode: str = "off", firewall_mode: str = "off", max_issues: int = 10,
                  fresh_ids: Optional[Sequence[str]] = None) -> str:
    def stat(p: Mapping[str, Any]) -> str:
        if not p or not p.get("n"):
            return "no data"
        ev = "n/a" if p["ev"] is None else f"{p['ev']:+.2f}%"
        lcb = "" if p["lcb"] is None else f" (lower bound {p['lcb']:+.2f}%)"
        wr = "" if p["p_profit"] is None else f", win share {p['p_profit']:.0%}"
        return f"n={p['n']}, {p['n_blocks']} blocks, EV {ev}{lcb}{wr}"

    lines = [f"🧭 INTELLIGENCE · playbook {rep.get('playbook')} · status {rep.get('status')}"]
    tr = rep.get("trend") or {}
    if tr:
        lines += ["", f"DELIVERED ALERTS (fixed bracket, after costs) — {tr.get('verdict')}",
                  f"  last {tr['days']}d: {stat(tr['recent'])}", f"  before  : {stat(tr['prior'])}"]
    v = rep.get("veto") or {}
    if v:
        lines += ["", f"VETO [{veto_mode}] — {v.get('verdict')}"]
        if v.get("n_avoid_groups"):
            lines += [f"  all restricted ({v['n_avoid_groups']} rule(s)): {stat(v['restricted'])}", f"  delivered: {stat(v['delivered'])}"]
            for k, g in (v.get("groups") or {}).items():
                lines.append(f"  · {k}: {g['verdict']}{' ⛔ suspended' if g['suspended'] else ''} — {stat(g['restricted'])}")
    fw = rep.get("firewall") or {}
    if fw:
        lines += ["", f"FIREWALL [{firewall_mode}] — armed: {', '.join(fw.get('armed') or []) or 'none'}"]
        for name, d in (fw.get("per_verdict") or {}).items():
            lines.append(f"  · {name}: {stat(d['group'])}; checks failed: {', '.join(c for c, ok in d['checks'].items() if not ok) or 'none'}")
    iss = rep.get("issues") or []
    fresh = set(fresh_ids or [])
    lines += ["", "ISSUES" if iss else "No issues found."]
    for i in iss[:max_issues]:
        lines.append(f"{_SEV_ICON.get(i['sev'], '•')} {i['sev']}{' · NEW' if i.get('id') in fresh else ''} · {i['text']}")
        ev = _ev_line(i.get("evidence") or {})
        lines.append(f"   → {i['action']}" + (f"  [{ev}]" if ev else "") + ("  [safe to automate]" if i.get("auto_safe") else ""))
    if len(iss) > max_issues:
        lines.append(f"… and {len(iss) - max_issues} more")
    lines += ["", "Only automatic actions: per-rule veto suspension, AVOID firewall, one bounded tightening (all switchable, all in the ledger). No rule is edited."]
    return "\n".join(lines)
