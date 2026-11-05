"""autotune — the narrowest possible self-adjustment: tighten ONE field, one small step, with proof.

Pure functions (no Redis, no cfg, no network): the learner passes in rows, the current
config value, and the persisted state, and executes whatever comes back.

Scope (deliberately tiny):
  * field      CONFLUENCE_MIN_PCT only (already on CONFIG_OVERRIDE_ALLOWED_FIELDS)
  * direction  UP only (stricter gate = fewer alerts); never relaxes
  * size       at most `step` points per change, never above `ceiling`
  * cadence    one change per `cooldown_h`; no new change while the previous one is being judged

Why this one change is evaluable offline: every archived LIVE row stores `effective_score`,
`effective_required` and `total`.  The gate is  required = max(total*P/100, abs_floor) * m
(m = macro multiplier when MACRO_CONTEXT_LIVE, else 1; abs_floor may be a learned per-pair value
when ENABLE_PAIR_THRESHOLDS is on).  Raising P to P' can only raise the first term, so the new
requirement is  max(effective_required, total*P'/100*m)  whatever the abs floor was, and a row
would have been removed iff  effective_score < that.  No config value other than P' is needed.
The removed trades are replayed on the fixed SL/TP bracket after costs.

A change is proposed only when ALL checks pass (the same discipline as the firewall):
  fresh data, outcome coverage, enough independent 3h blocks, the removed band is a PROVEN
  loser (EV + z_arm*SE < 0) in BOTH halves of time, removing it improves the delivered
  book's total after costs, and the opportunity cost (share of delivered trades removed) is
  bounded.  It must pass on `arm_cycles` consecutive learner cycles.

After applying, the change is judged: delivered trades since the change vs the same number of
days before it.  Clear deterioration (upper bound of the gap < 0) rolls it back to the exact
previous state (including "no override existed").  Otherwise it is KEPT.  If anything else
changes the field (the Brain also writes it), autotune notices, records EXTERNAL_CHANGE and
stands down for a full cooldown.  Everything is returned as ledger events.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from plan_replay import Row, has_path, replay_plan
from scoreboard import _pool as _fwd_pool

KEY_AUTOTUNE = "intel_autotune"
FIELD = "CONFLUENCE_MIN_PCT"
_DAY = 86400
Items = List[Tuple[Row, float, float]]


def default_params() -> Dict[str, Any]:
    return {"step": 1.0, "ceiling": 70.0, "cooldown_h": 72.0, "eval_h": 72.0, "lookback_days": 21, "min_blocks": 30,
            "z_arm": 2.33, "arm_cycles": 2, "max_share": 0.15, "min_half": 8, "max_age_h": 36.0, "min_path_share": 0.6,
            "z_harm": 1.645, "min_eval_blocks": 12}


def _rnd(v: Any, nd: int = 3) -> Optional[float]:
    return None if v is None else round(float(v), nd)


def removed_band(rows: Sequence[Row], *, new_pct: float, macro_live: bool, fixed: Mapping[str, float],
                 cost_pct: float, since_ts: float) -> Tuple[Items, Items]:
    """(removed, all_live) trades since `since_ts`.  Rows without a stored gate state, a total or a
    candle path are skipped (they cannot be judged, so they are neither removed nor counted)."""
    sl, tp, h = float(fixed["sl"]), float(fixed["tp"]), int(fixed["h"])
    removed: Items = []
    live: Items = []
    for r in rows:
        if r.get("source") != "LIVE" or r.get("entry_ts") is None or r["entry_ts"] <= since_ts or not has_path(r):
            continue
        es, rq, total = r.get("effective_score"), r.get("effective_required"), r.get("total")
        if es is None or not rq or float(rq) <= 0 or not total or float(total) <= 0:
            continue
        res = replay_plan(r, sl, tp, h, cost_pct)
        if res is None:
            continue
        item = (r, res[1], sl)
        live.append(item)
        m = float(r.get("macro_multiplier") or 1.0) if macro_live else 1.0
        if float(es) < max(float(rq), float(total) * new_pct / 100.0 * m):
            removed.append(item)
    return removed, live


def _halves(items: Items, min_half: int) -> Optional[Tuple[float, float]]:
    s = sorted(items, key=lambda it: it[0]["entry_ts"])
    mid = len(s) // 2
    a, b = [x[1] for x in s[:mid]], [x[1] for x in s[mid:]]
    if len(a) < min_half or len(b) < min_half:
        return None
    return sum(a) / len(a), sum(b) / len(b)


def observe(state: Optional[Mapping[str, Any]], cur_value: float, now_ts: float) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Detect a change of the field that autotune did not make.  `cur_value` is the value the bot
    will actually use.  Returns (state, events).  The first observation just records the value."""
    st: Dict[str, Any] = dict(state or {})
    events: List[Dict[str, Any]] = []
    expected = st.get("expected")
    if expected is not None and abs(float(expected) - float(cur_value)) > 1e-9:
        events.append({"kind": "AUTOTUNE", "subject": FIELD, "event": "EXTERNAL_CHANGE",
                       "text": f"{FIELD} changed {float(expected):g} -> {float(cur_value):g} outside autotune; standing down for a full cooldown",
                       "old": float(expected), "new": float(cur_value)})
        st["pending"] = None
        st["last_change_ts"] = int(now_ts)
        st["streak"] = 0
    st["expected"] = float(cur_value)
    return st, events


def propose(
    rows: Sequence[Row], *, cur_pct: float, macro_live: bool = False, state: Mapping[str, Any], now_ts: float, fixed: Mapping[str, float],
    cost_pct: float, newest_ts: Optional[float], n_rows: int, n_path: int, data_ok: bool = True, params: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Returns (decision, new_state).  decision["action"] is "APPLY" or "NONE".  The streak is the only state
    this updates; the caller records the change itself only after the write succeeds (see `committed`)."""
    P = default_params()
    P.update(params or {})
    st: Dict[str, Any] = dict(state or {})
    new_pct = round(float(cur_pct) + float(P["step"]), 4)
    last = float(st.get("last_change_ts") or 0)
    since = max(last, now_ts - int(P["lookback_days"]) * _DAY)
    removed, live = removed_band(rows, new_pct=new_pct, macro_live=macro_live, fixed=fixed, cost_pct=cost_pct, since_ts=since)
    pr, pa = _fwd_pool(removed), _fwd_pool(live)
    kept = [it for it in live if it not in removed]
    agg_before, agg_after = sum(x[1] for x in live), sum(x[1] for x in kept)
    share = len(removed) / len(live) if live else 1.0
    ucb = None if pr["ev"] is None or pr["se"] is None else float(pr["ev"]) + float(P["z_arm"]) * float(pr["se"])
    halves = _halves(removed, int(P["min_half"]))
    fresh = newest_ts is not None and (now_ts - float(newest_ts)) <= float(P["max_age_h"]) * 3600.0
    checks = {
        "data_ok": bool(data_ok),
        "fresh_data": bool(fresh),
        "outcome_coverage": bool(n_rows > 0 and n_path / n_rows >= float(P["min_path_share"])),
        "no_pending_evaluation": not st.get("pending"),
        "cooldown_elapsed": (now_ts - last) >= float(P["cooldown_h"]) * 3600.0,
        "below_ceiling": new_pct <= float(P["ceiling"]),
        "enough_blocks": pr["n_blocks"] >= int(P["min_blocks"]),
        "removed_proven_loser": bool(ucb is not None and ucb < 0),
        "stable_over_time": bool(halves is not None and halves[0] < 0 and halves[1] < 0),
        "improves_book": bool(removed and agg_after > agg_before),
        "bounded_opportunity_cost": share <= float(P["max_share"]),
    }
    eligible = all(checks.values())
    streak = int(st.get("streak") or 0) + 1 if eligible else 0
    st["streak"] = streak
    action = "APPLY" if eligible and streak >= int(P["arm_cycles"]) else "NONE"
    ev = {"field": FIELD, "old": float(cur_pct), "new": new_pct, "removed": {"n": pr["n"], "n_blocks": pr["n_blocks"], "ev": _rnd(pr["ev"]),
          "ucb": _rnd(ucb), "win_share": _rnd(pr["p_profit"])}, "book_ev": _rnd(pa["ev"]), "removed_share": _rnd(share),
          "agg_before": _rnd(agg_before, 2), "agg_after": _rnd(agg_after, 2), "since_ts": int(since)}
    decision = {"action": action, "field": FIELD, "old": float(cur_pct), "new": new_pct, "checks": checks, "eligible": eligible,
                "streak": streak, "evidence": ev,
                "text": f"{FIELD} {float(cur_pct):g} -> {new_pct:g}: removes {len(removed)}/{len(live)} delivered trades "
                        f"(EV {_rnd(pr['ev'])}%, removed-band UCB {_rnd(ucb)}%), book total {_rnd(agg_before, 1)} -> {_rnd(agg_after, 1)}"}
    return decision, st


def committed(state: Mapping[str, Any], decision: Mapping[str, Any], now_ts: float, *, had_override: bool,
              prev_override: Optional[float]) -> Dict[str, Any]:
    """New state AFTER the write succeeded (never before)."""
    st = dict(state)
    st["pending"] = {"field": FIELD, "old": float(decision["old"]), "new": float(decision["new"]), "applied_ts": int(now_ts),
                     "had_override": bool(had_override), "prev_override": prev_override, "evidence": decision.get("evidence")}
    st["last_change_ts"] = int(now_ts)
    st["expected"] = float(decision["new"])
    st["streak"] = 0
    return st


def evaluate_pending(
    rows: Sequence[Row], state: Mapping[str, Any], *, now_ts: float, fixed: Mapping[str, float], cost_pct: float,
    params: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Judge a pending change once its evaluation window has elapsed.
    Returns (result, new_state); result["status"] in {"NONE", "PENDING", "KEPT", "ROLLBACK", "INSUFFICIENT"}.
    ROLLBACK carries the exact previous state to restore (`restore`: {"remove": bool, "value": float|None})."""
    P = default_params()
    P.update(params or {})
    st: Dict[str, Any] = dict(state)
    pend = st.get("pending")
    if not pend:
        return {"status": "NONE"}, st
    applied = float(pend["applied_ts"])
    window = float(P["eval_h"]) * 3600.0
    if now_ts - applied < window:
        return {"status": "PENDING", "hours_left": round((applied + window - now_ts) / 3600.0, 1)}, st
    sl, tp, h = float(fixed["sl"]), float(fixed["tp"]), int(fixed["h"])
    post: Items = []
    pre: Items = []
    for r in rows:
        ts = r.get("entry_ts")
        if r.get("source") != "LIVE" or ts is None or not has_path(r):
            continue
        if applied < ts <= now_ts:
            bucket = post
        elif applied - (now_ts - applied) < ts <= applied:
            bucket = pre
        else:
            continue
        res = replay_plan(r, sl, tp, h, cost_pct)
        if res is not None:
            bucket.append((r, res[1], sl))
    a, b = _fwd_pool(post), _fwd_pool(pre)
    if a["se"] is None or b["se"] is None or a["n_blocks"] < int(P["min_eval_blocks"]) or b["n_blocks"] < int(P["min_eval_blocks"]):
        if now_ts - applied > 3 * window:           # never judged: stop waiting, keep the (bounded) change, say so
            st["pending"] = None
            return {"status": "INSUFFICIENT", "text": f"{FIELD} change {pend['old']:g} -> {pend['new']:g} could not be judged (too few trades); kept",
                    "evidence": {"post_blocks": a["n_blocks"], "pre_blocks": b["n_blocks"]}}, st
        return {"status": "PENDING", "hours_left": 0.0, "note": "waiting for enough trades"}, st
    gap = float(a["ev"]) - float(b["ev"])
    se = math.hypot(float(a["se"]), float(b["se"]))
    ucb = gap + float(P["z_harm"]) * se
    ev = {"post": {"n": a["n"], "n_blocks": a["n_blocks"], "ev": _rnd(a["ev"])}, "pre": {"n": b["n"], "n_blocks": b["n_blocks"], "ev": _rnd(b["ev"])},
          "gap": _rnd(gap), "gap_ucb": _rnd(ucb)}
    st["pending"] = None
    if ucb < 0:
        st["last_change_ts"] = int(now_ts)
        st["expected"] = float(pend["old"])
        return {"status": "ROLLBACK", "field": FIELD, "old": float(pend["new"]), "new": float(pend["old"]),
                "restore": {"remove": not pend.get("had_override"), "value": pend.get("prev_override")},
                "text": f"{FIELD} {pend['old']:g} -> {pend['new']:g} made delivered trades clearly worse (gap {gap:+.2f}%, upper bound {ucb:+.2f}%); rolling back",
                "evidence": ev}, st
    return {"status": "KEPT", "text": f"{FIELD} {pend['old']:g} -> {pend['new']:g} judged: delivered EV {_rnd(a['ev'])}% vs {_rnd(b['ev'])}% before (not clearly worse); kept",
            "evidence": ev}, st
