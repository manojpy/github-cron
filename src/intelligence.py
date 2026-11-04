"""intelligence — the learner's coordinator: one read-only diagnosis of the whole
system, plus a closed-loop check on the live playbook veto.

Pure functions (no Redis, no cfg, no network, no files): everything is passed in,
so the logic is deterministic and replayable.  It adds NO new optimiser; it reads
what the existing modules already produced (playbook, scoreboard, autopsy,
hypothesis pool) and answers three questions every learner cycle:

  1. Is the veto helping?   `veto_monitor` compares what the playbook restricts
     (AVOID groups, forward of the moment each AVOID was issued) with what was
     delivered, on the fixed SL/TP bracket after costs, with the same
     cluster-robust standard error the scoreboard uses.  Two harm tests:
        H1  the restricted alerts would have EARNED money (lower bound > 0)
        H2  the restricted alerts did significantly BETTER than the delivered ones
     Harm must be seen on `harm_cycles` consecutive cycles before the veto is
     suspended.  It resumes only after `resume_cycles` consecutive cycles in which
     the restriction is proven useful again (upper bound < 0).  Suspending only
     means "the veto logs but still sends" (the same as ALERT_VETO_MODE=shadow);
     it never blocks anything, never edits config and never touches an alert rule.
  2. Is the delivered population improving?   `delivered_trend` (last N days vs
     the N days before).
  3. What needs attention?   `build_report` turns the stored results into a short,
     ranked issue list with a suggested human action for each.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from plan_replay import Row, has_path, replay_plan
from playbook import group_rows
from scoreboard import _pool as _fwd_pool

KEY_VETO_MON = "intel_veto_monitor"
KEY_INTEL = "intel_report"

Z = 1.645
_DAY = 86400
Items = List[Tuple[Row, float, float]]
_SEV_ORDER = {"HIGH": 0, "MED": 1, "LOW": 2, "INFO": 3}
_SEV_ICON = {"HIGH": "🔴", "MED": "🟠", "LOW": "🟡", "INFO": "🔵"}


def default_monitor_params() -> Dict[str, Any]:
    return {"min_blocks": 20, "harm_cycles": 2, "resume_cycles": 4}


def _rid(r: Mapping[str, Any]) -> Tuple[Any, Any, Any]:
    return (r.get("pair"), r.get("alert_key"), r.get("entry_ts"))


def _rnd(v: Any, nd: int = 3) -> Optional[float]:
    return None if v is None else round(float(v), nd)


def _brief(p: Mapping[str, Any]) -> Dict[str, Any]:
    return {"n": int(p["n"]), "n_blocks": int(p["n_blocks"]), "ev": _rnd(p["ev"]), "lcb": _rnd(p["lcb"]),
            "ucb": _rnd(p["ucb"]), "p_profit": _rnd(p["p_profit"])}


def _gap(a: Mapping[str, Any], b: Mapping[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """(a.ev - b.ev, one-sided lower bound of that gap) for two independent samples."""
    if a["ev"] is None or b["ev"] is None or a["se"] is None or b["se"] is None:
        return None, None
    g = float(a["ev"]) - float(b["ev"])
    return g, g - Z * math.hypot(float(a["se"]), float(b["se"]))


def _fixed_tuple(fixed: Mapping[str, float]) -> Tuple[float, float, int]:
    return float(fixed["sl"]), float(fixed["tp"]), int(fixed["h"])


def restricted_vs_delivered(
    blob: Mapping[str, Any], rows: Sequence[Row], *, fixed: Mapping[str, float], cost_pct: float,
    family_fn: Optional[Callable[[str], str]] = None,
) -> Tuple[Items, Items, int]:
    """(restricted, delivered, n_avoid_groups), each item (row, net_pnl_pct, sl_pct).

    restricted = trades of every AVOID group that happened AFTER that group's AVOID was
                 issued (the same pool the scoreboard's AVOID side uses; each trade once).
                 While the veto is live these are the 'playbook_veto' shadow rows; while it
                 is suspended they are real delivered alerts, so the test never goes blind.
    delivered  = real (LIVE) trades since the first AVOID that are not in `restricted`."""
    entries = blob.get("entries") or {}
    avoid = {k: e for k, e in entries.items() if e.get("restrict") == "AVOID" and e.get("avoid_since_ts") is not None}
    if not avoid:
        return [], [], 0
    sl, tp, h = _fixed_tuple(fixed)
    groups = group_rows(rows, family_fn)
    seen: Set[Tuple[Any, Any, Any]] = set()
    restricted: Items = []
    for k, e in sorted(avoid.items(), key=lambda kv: (kv[0].startswith("fam:"), kv[0])):
        since = float(e["avoid_since_ts"])
        for r in groups.get(k, []):
            rid = _rid(r)
            if r["entry_ts"] <= since or rid in seen:
                continue
            res = replay_plan(r, sl, tp, h, cost_pct)
            if res is None:
                continue
            seen.add(rid)
            restricted.append((r, res[1], sl))
    t0 = min(float(e["avoid_since_ts"]) for e in avoid.values())
    delivered: Items = []
    for r in rows:
        if r.get("source") != "LIVE" or r.get("entry_ts") is None or not has_path(r):
            continue
        if r["entry_ts"] <= t0 or _rid(r) in seen:
            continue
        res = replay_plan(r, sl, tp, h, cost_pct)
        if res is not None:
            delivered.append((r, res[1], sl))
    return restricted, delivered, len(avoid)


def veto_monitor(
    blob: Mapping[str, Any], rows: Sequence[Row], *, fixed: Mapping[str, float], cost_pct: float,
    prev_state: Optional[Mapping[str, Any]], now_ts: float, params: Optional[Mapping[str, Any]] = None,
    family_fn: Optional[Callable[[str], str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Returns (result, new_state).  `result["suspended"]` says whether the veto SHOULD be
    suspended now; the caller decides (by mode) whether to act on it."""
    P = default_monitor_params()
    P.update(params or {})
    st = dict(prev_state or {})
    was_suspended = bool(st.get("suspended"))
    suspended = was_suspended
    since: Optional[int] = int(st["since"]) if st.get("since") else None
    reason = str(st.get("reason") or "")
    harm_streak = int(st.get("harm_streak") or 0)
    clear_streak = int(st.get("clear_streak") or 0)

    restricted, delivered, n_avoid = restricted_vs_delivered(blob, rows, fixed=fixed, cost_pct=cost_pct, family_fn=family_fn)
    rs, dl = _fwd_pool(restricted), _fwd_pool(delivered)
    gap, gap_lcb = _gap(rs, dl)
    min_b = int(P["min_blocks"])
    enough = rs["se"] is not None and rs["n_blocks"] >= min_b
    harm_profit = bool(enough and rs["lcb"] is not None and rs["lcb"] > 0)
    harm_anti = bool(enough and dl["n_blocks"] >= min_b and gap_lcb is not None and gap_lcb > 0)
    harm = harm_profit or harm_anti
    helping = bool(enough and rs["ucb"] is not None and rs["ucb"] < 0)

    harm_text = ""
    if harm_profit:
        harm_text = (f"restricted alerts would have EARNED {rs['ev']:+.2f}% per trade "
                     f"(lower bound {rs['lcb']:+.2f}%, {rs['n_blocks']} independent blocks)")
    elif harm_anti and gap is not None and gap_lcb is not None:
        harm_text = (f"restricted alerts did {gap:+.2f}% per trade BETTER than delivered ones "
                     f"(lower bound {gap_lcb:+.2f}%)")

    changed: Optional[str] = None
    if not n_avoid:
        verdict = "NO_AVOID"
        harm_streak = clear_streak = 0
        if was_suspended:
            suspended, since, reason, changed = False, None, "", "RESUMED"
    else:
        if harm:
            harm_streak, clear_streak = harm_streak + 1, 0
        else:
            harm_streak = 0
            clear_streak = clear_streak + 1 if helping else 0
        if not suspended and harm_streak >= int(P["harm_cycles"]):
            suspended, since, reason, changed = True, int(now_ts), harm_text, "SUSPENDED"
        elif suspended and clear_streak >= int(P["resume_cycles"]):
            suspended, since, reason, changed = False, None, "", "RESUMED"
        elif suspended and harm_text:
            reason = harm_text
        verdict = "HARMFUL" if harm else "HELPING" if helping else "COLLECTING" if not enough else "NEUTRAL"

    result: Dict[str, Any] = {
        "verdict": verdict, "n_avoid_groups": n_avoid, "suspended": suspended, "since": since, "reason": reason,
        "changed": changed, "harm_streak": harm_streak, "clear_streak": clear_streak,
        "harm_cycles": int(P["harm_cycles"]), "resume_cycles": int(P["resume_cycles"]), "min_blocks": min_b,
        "restricted": _brief(rs), "delivered": _brief(dl),
        "gap": _rnd(gap), "gap_lcb": _rnd(gap_lcb),
        "tests": {"restricted_earns_money": harm_profit, "restricted_beats_delivered": harm_anti, "restricted_proven_loser": helping},
    }
    new_state = {"suspended": suspended, "since": since, "reason": reason, "harm_streak": harm_streak,
                 "clear_streak": clear_streak, "last_ts": int(now_ts)}
    return result, new_state


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
    gap, gap_lcb = _gap(a, b)
    if gap is None or gap_lcb is None or a["n_blocks"] < min_blocks or b["n_blocks"] < min_blocks:
        verdict = "INSUFFICIENT"
    else:
        se = math.hypot(float(a["se"]), float(b["se"]))
        verdict = "IMPROVING" if gap_lcb > 0 else "DETERIORATING" if gap + Z * se < 0 else "NO_CLEAR_CHANGE"
    return {"days": days, "recent": _brief(a), "prior": _brief(b), "gap": _rnd(gap), "verdict": verdict}


def _issue(sev: str, code: str, text: str, action: str) -> Dict[str, str]:
    return {"sev": sev, "code": code, "text": text, "action": action}


def build_report(
    *, now_ts: float, blob: Mapping[str, Any], sb: Optional[Mapping[str, Any]], autopsy: Optional[Mapping[str, Any]],
    hyp_pool: Optional[Mapping[str, Any]], veto: Optional[Mapping[str, Any]], trend: Optional[Mapping[str, Any]],
    n_rows: int, n_path: int, prev_ts: Optional[float] = None, max_age_h: float = 36.0, min_path_share: float = 0.6,
    veto_mode: str = "off",
) -> Dict[str, Any]:
    """One ranked issue list from everything the learner already computed."""
    issues: List[Dict[str, str]] = []

    if veto and veto.get("verdict") == "HARMFUL":
        stance = ("SUSPENDED (veto logs but still sends)" if veto.get("suspended") and veto_mode == "live"
                  else "would suspend" if veto.get("suspended") else
                  f"harm seen {veto.get('harm_streak')}/{veto.get('harm_cycles')} cycle(s)")
        issues.append(_issue("HIGH", "VETO_HARMFUL", f"The playbook veto looks harmful: {veto.get('reason') or 'see monitor'} — {stance}.",
                             "The monitor already acted; review the AVOID groups, or set VETO_MONITOR_MODE=shadow to stop it acting." if veto_mode == "live"
                             else "Review the AVOID groups; set VETO_MONITOR_MODE=live to let the monitor suspend the veto, or ALERT_VETO_MODE=shadow to stop the veto yourself."))
    elif veto and veto.get("suspended"):
        issues.append(_issue("MED", "VETO_SUSPENDED", "The veto is suspended and is waiting for proof that the restriction helps again "
                             f"({veto.get('clear_streak')}/{veto.get('resume_cycles')} clear cycles).",
                             "No action needed; it resumes by itself once the restricted alerts are proven losers again."))

    if prev_ts and (now_ts - float(prev_ts)) / 3600.0 > max_age_h:
        gap_h = (now_ts - float(prev_ts)) / 3600.0
        issues.append(_issue("HIGH", "LEARNER_GAP", f"The learner did not run for {gap_h:.0f}h (limit {max_age_h:g}h): the veto failed open (sent everything) in that gap.",
                             "Check the Cronjobs.org job for learner.yml and its last HTTP response."))

    if trend and trend.get("verdict") == "DETERIORATING":
        r, p = trend["recent"], trend["prior"]
        issues.append(_issue("HIGH", "DELIVERED_WORSE", f"Delivered alerts got worse: EV {r['ev']:+.2f}% per trade in the last {trend['days']}d vs {p['ev']:+.2f}% before.",
                             "Open the Brain report's per-alert verdicts; look for the alert family that changed."))

    ctl = blob.get("control") or {}
    if not ctl.get("available"):
        issues.append(_issue("MED", "NO_CONTROL", f"No matched control baseline ({ctl.get('note') or 'unavailable'}): no plan can pass beyond PROMISING and no AVOID can be issued.",
                             "Check the candle fetch in the learner run (DELTA_API_BASE, rate limits)."))

    if n_rows and n_path / n_rows < min_path_share:
        issues.append(_issue("MED", "LOW_PATH_COVERAGE", f"Only {n_path}/{n_rows} archived rows have a candle path ({n_path / n_rows:.0%}); the rest cannot be replayed.",
                             "Leave path recovery on (PLAYBOOK_RECON_MAX_CALLS > 0) and check outcome archiving."))

    summ = blob.get("summary") or {}
    decaying = int(summ.get("RECONFIRM_DUE", 0)) + int(summ.get("EXPIRED", 0)) + int(summ.get("DEMOTED", 0))
    if decaying:
        issues.append(_issue("MED", "CHAMPION_DECAY", f"{decaying} playbook entr{'y is' if decaying == 1 else 'ies are'} demoted, expired or overdue for re-confirmation.",
                             "Nothing to do: they stop being served automatically. They are re-proven on new data."))

    costly = [x for x in ((autopsy or {}).get("rejections") or {}).get("reasons", []) if x.get("status") == "COSTLY"]
    for x in costly[:3]:
        issues.append(_issue("MED", "COSTLY_GATE", f"Gate '{x['reason']}' rejected trades that did BETTER than the delivered ones (n={x['n']}, mean {x['mean_net']:+.2f}%).",
                             "Investigate this gate; it is report-only, nothing was changed."))

    confirmed = (hyp_pool or {}).get("confirmed") or {}
    if confirmed:
        top = sorted(confirmed.values(), key=lambda v: -(v.get("confirmed_at") or 0))[:3]
        txt = "; ".join(str(v.get("text", "?")) for v in top)
        issues.append(_issue("INFO", "HYPOTHESES_CONFIRMED", f"{len(confirmed)} self-generated 'skip when' rule(s) passed forward testing and are NOT applied: {txt}.",
                             "Review them; they only become vetoes if you wire them in on purpose."))

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
    top_sev = issues[0]["sev"] if issues else "OK"
    return {"ts": int(now_ts), "playbook": blob.get("label"), "status": top_sev, "issues": issues, "trend": trend, "veto": veto}


def format_report(rep: Mapping[str, Any], *, veto_mode: str = "off", max_issues: int = 10) -> str:
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
        lines += ["", f"VETO [{veto_mode}] — {v.get('verdict')}" + ("  ⛔ suspended" if v.get("suspended") else "")]
        if v.get("n_avoid_groups"):
            lines += [f"  restricted ({v['n_avoid_groups']} AVOID group(s)): {stat(v['restricted'])}",
                      f"  delivered: {stat(v['delivered'])}"]
    iss = rep.get("issues") or []
    lines += ["", "ISSUES" if iss else "No issues found."]
    for i in iss[:max_issues]:
        lines.append(f"{_SEV_ICON.get(i['sev'], '•')} {i['sev']} · {i['text']}")
        lines.append(f"   → {i['action']}")
    if len(iss) > max_issues:
        lines.append(f"… and {len(iss) - max_issues} more")
    lines += ["", "Read-only. The only automatic action is suspending the playbook veto (VETO_MONITOR_MODE=live); no rule or config is edited."]
    return "\n".join(lines)
