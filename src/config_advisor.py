"""config_advisor — the read-only configuration adviser (runs inside the learner cycle).

Pure functions (no Redis, no cfg, no network, no files): the learner passes in what it has
already computed (rows, playbook blob, scoreboard, veto monitor, autotune report, ...), the
CURRENT config values and the persisted advisor state, and executes whatever comes back.

It NEVER edits config.  It answers one question every learner cycle:
    "Is there a config change that is genuinely worth making?"
and usually the answer is NO.  When the answer is yes it produces ONE exact change
(field, current -> proposed, rollback value, expiry) backed by evidence that this module
re-computed itself; it does not trust another module's verdict.

What it can recommend (a strict whitelist, see GATES / the value sweeps below):
  * shadow -> live      a restrict-only gate whose blocked trades are a PROVEN loser
  * live -> shadow      a live gate whose blocked trades are a PROVEN earner (the gate is hurting)
  * bool flip           same, for the true/false switches that have a measurable gate
  * value change        CONFLUENCE_MIN_PCT (tighten, bounded) and EXHAUSTION_VETO_SCORE
  * agreement flips     PLAYBOOK_MODE, AUTOTUNE_MODE, VETO_MONITOR_MODE (several modules agree)

Evidence standard for a "proven loser" (promotion) - ALL of:
  fresh data, outcome coverage, enough independent 3h blocks, upper bound < 0 at a
  Bonferroni-adjusted z, worse than the REST of the book (session/direction stratified),
  both halves of time negative, the latest 40% negative on its own (a recency check: a gate is
  the user's own fixed rule and nothing is fitted, so this tests stability, it is not a holdout),
  not one pair's doing (top-pair share + leave-top-pair-out) and at least `min_pairs` pairs,
  bounded share of the book, and the same verdict on `promote_cycles` consecutive evidence
  refreshes (persistence on refreshed, overlapping data: NOT independent replication).
The two value sweeps (CONFLUENCE_MIN_PCT, EXHAUSTION_VETO_SCORE) choose a candidate using only the
older 60% of the book and then test the ONE chosen candidate once on the newest 40% (a real holdout).
A "proven earner" (demotion) mirrors it, with a lower bar (`demote_cycles`) because
removing a restriction is the safer direction.  A tier is attached:
  E0 data not trustworthy   E1 too little data   E2 data enough, a check fails (watch)
  E3 every check passes, not yet repeated        E4 passed on consecutive refreshes (send)

Every recommendation is fingerprinted, rate-limited, expires, is revoked (with hysteresis)
when its evidence stops holding, is detected as APPLIED when the config reaches the proposed
value, and is then verified on real delivered trades (CONFIRMED / NEUTRAL / HARMFUL; a
HARMFUL verdict produces a rollback recommendation).
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from autopsy import independent_rows
from autotune import removed_band
from hypothesis_gen import row_features, rules_hit
from intelligence import _gap, _stratified_gap, _z, restricted_by_group, verdict_items
from plan_replay import Row, has_path, replay_plan
from scoreboard import _pool as _fwd_pool

KEY_ADVISOR = "config_advisor"
Items = List[Tuple[Row, float, float]]
_DAY = 86400
_TIER_ORDER = {"E0": 0, "E1": 1, "E2": 2, "E3": 3, "E4": 4}
_LIVE_STATES = ("NEW", "SHADOW", "SENT")            # open: NEW = not delivered yet, SHADOW = tracked in shadow mode only, SENT = delivered
POLICY_VERSION = 2                                  # bump when the evidence rules change: streaks from an older policy start again
_POLICY_KEYS = ("min_blocks", "min_n", "min_pairs", "alpha", "z_floor", "z_demote", "z_oos", "min_half", "oos_frac",
                "oos_min_blocks", "max_share", "max_pair_share", "min_cell", "lookback_days", "step_pct", "max_steps",
                "ceiling_pct", "veto_scores", "min_cats")
_LOG_CAP = 200
_IST = timezone(timedelta(hours=5, minutes=30))

# Fields the adviser may NEVER recommend changing, whatever a module says (hard safety gates,
# data-integrity checks, risk controls, outcome definitions, anything that moves the yardstick).
NEVER_TOUCH = frozenset({
    "ENABLE_KILL_SWITCH", "KILL_SWITCH_MODE", "BRAIN_OOD_ENABLED", "ENABLE_PORTFOLIO_HEAT_GATE",
    "ENABLE_CALIBRATION_GATE", "ENABLE_CONFLUENCE_GATE", "DRY_RUN_MODE", "OUTCOME_MAE_LOSS_PCT",
    "OUTCOME_RR_TARGET", "OUTCOME_LOOKAHEAD_CANDLES", "OUTCOME_COST_MODEL", "BRAIN_FEE_PCT",
    "BRAIN_SLIPPAGE_PCT", "MIN_WIN_RATE", "LAYER_BOUNDARY_MODE", "RUN_TIMEOUT_SECONDS",
    # The Brain's shadow-rejection evidence comes from this filter. Switching it off
    # removes the main quality gate AND starves every gate test in this module.
    "ENABLE_WIN_RATE_FILTER",
})


def default_params() -> Dict[str, Any]:
    return {
        "min_blocks": 30, "min_n": 30, "min_pairs": 5, "alpha": 0.05, "z_floor": 2.33, "z_demote": 1.96, "z_oos": 1.645,
        "min_half": 8, "oos_frac": 0.4, "oos_min_blocks": 8, "max_share": 0.35, "max_pair_share": 0.5,
        "promote_cycles": 3, "demote_cycles": 2, "send_tier": "E4", "expiry_days": 7, "cooldown_hours": 72.0,
        "max_per_cycle": 2, "max_per_day": 3, "verify_days": 7, "verify_min_blocks": 8, "max_age_h": 36.0,
        "min_path_share": 0.6, "digest_hours": 24.0, "mute": (), "revoke_cycles": 2, "snooze_days": 14,
        "step_pct": 1.0, "max_steps": 3, "ceiling_pct": 70.0, "lookback_days": 30, "min_cats": 2,
        "veto_scores": (3, 4, 5), "veto_score": 4, "macro_live": False, "min_cell": 5, "overridden": (),
    }


# ── small helpers ─────────────────────────────────────────────────────────────

def _rnd(v: Any, nd: int = 3) -> Optional[float]:
    return None if v is None else round(float(v), nd)


def _f(v: Any) -> Optional[float]:
    """Full precision for every number a decision reads; numbers are rounded only when text is built."""
    return None if v is None else float(v)


def _pct(v: Any) -> str:
    return "n/a" if v is None else f"{v:+.2f}%"


def _ts_str(ts: float) -> str:
    return datetime.fromtimestamp(float(ts), tz=_IST).strftime("%d %b %H:%M IST")


def _fmt_val(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _fp(field: str, cur: Any, new: Any) -> str:
    return hashlib.sha1(f"{field}|{_fmt_val(cur)}|{_fmt_val(new)}".encode()).hexdigest()[:10]


def _policy_params(P: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: (list(P[k]) if isinstance(P[k], (tuple, list)) else P[k]) for k in _POLICY_KEYS if k in P}


def _policy_hash(P: Mapping[str, Any]) -> str:
    """Fingerprint of every setting that changes what counts as evidence, plus the code's POLICY_VERSION."""
    return hashlib.sha1(f"{POLICY_VERSION}|{json.dumps(_policy_params(P), sort_keys=True, default=str)}".encode()).hexdigest()[:10]


def _provenance(rows: Sequence[Row], fixed: Mapping[str, float], cost_pct: float, P: Mapping[str, Any], n_rows: int,
                n_path: int, newest_ts: Optional[float], policy: str) -> Dict[str, Any]:
    """What a recommendation was computed from, so it can be reconstructed months later."""
    live = sorted((str(r.get("pair")), int(r["entry_ts"])) for r in rows if r.get("source") == "LIVE" and r.get("entry_ts") is not None)
    return {"policy": policy, "policy_version": POLICY_VERSION, "params": _policy_params(P),
            "fixed": {k: fixed[k] for k in ("sl", "tp", "h") if k in fixed}, "cost_pct": _rnd(cost_pct, 6),
            "n_rows": int(n_rows), "n_with_path": int(n_path), "rows_without_path": int(n_rows) - int(n_path),
            "newest_ts": None if newest_ts is None else int(newest_ts), "live_rows": len(live),
            "live_oldest_ts": live[0][1] if live else None, "live_fingerprint": hashlib.sha1(repr(live).encode()).hexdigest()[:12]}


def _ctx(r: Mapping[str, Any]) -> Dict[str, Any]:
    c = r.get("context")
    return c if isinstance(c, dict) else {}


def _items(rows: Sequence[Row], fixed: Mapping[str, float], cost_pct: float,
           pred: Optional[Callable[[Row], bool]] = None) -> Items:
    """(row, net pnl %, plan sl) on the fixed bracket for every row that has a stored candle path."""
    sl, tp, h = float(fixed["sl"]), float(fixed["tp"]), int(fixed["h"])
    out: Items = []
    for r in rows:
        if r.get("entry_ts") is None or not has_path(r) or (pred is not None and not pred(r)):
            continue
        res = replay_plan(r, sl, tp, h, cost_pct)
        if res is not None:
            out.append((r, res[1], sl))
    return out


def _live(rows: Sequence[Row]) -> List[Row]:
    return [r for r in rows if r.get("source") == "LIVE"]


def _shadow_tag(rows: Sequence[Row], tag: str) -> List[Row]:
    """Independent (one open trade per pair) shadow rows that carry a rejection tag."""
    sel = [r for r in rows if r.get("source") == "SHADOW" and str(r.get("rejection_reason") or _ctx(r).get("rejection_reason") or "").split(":")[0].strip() == tag]
    return independent_rows(sel)


# ── the statistics every gate test shares ─────────────────────────────────────

def restriction_stats(F: Items, R: Items, P: Mapping[str, Any], z: float) -> Dict[str, Any]:
    """Numbers for 'what does restricting set F do to the book that contains F and R'.
    F = trades the gate blocks / would block, R = the rest of the delivered book (same period)."""
    pf, pr = _fwd_pool(F), _fwd_pool(R)
    ev, se = pf["ev"], pf["se"]
    ok = ev is not None and se is not None
    s = sorted(F, key=lambda it: it[0]["entry_ts"])
    mid = len(s) // 2
    ha = [x[1] for x in s[:mid]]
    hb = [x[1] for x in s[mid:]]
    half_ok = len(ha) >= int(P["min_half"]) and len(hb) >= int(P["min_half"])
    split = int(len(s) * (1.0 - float(P["oos_frac"])))
    oos = _fwd_pool(s[split:]) if len(s) - split >= 2 else _fwd_pool([])
    gap, gse, cells = _stratified_gap(F, R, int(P["min_cell"]))
    if gap is None:                                   # no session/direction cell with both samples: plain gap
        g2, se2, _lb = _gap(pf, pr)
        gap, gse, cells = g2, se2, 0
    pairs: Dict[str, int] = {}
    for it in F:
        pairs[str(it[0].get("pair") or "?")] = pairs.get(str(it[0].get("pair") or "?"), 0) + 1
    top_pair, top_n = (max(pairs.items(), key=lambda kv: kv[1]) if pairs else ("", 0))
    rest_nets = [it[1] for it in F if str(it[0].get("pair") or "?") != top_pair]
    total = len(F) + len(R)
    return {
        "n": pf["n"], "n_blocks": pf["n_blocks"], "ev": _f(ev), "se": _f(se),
        "lcb": _f(ev - z * se) if ok else None, "ucb": _f(ev + z * se) if ok else None,
        "n_rest": pr["n"], "rest_ev": _f(pr["ev"]),
        "gap": _f(gap), "gap_ucb": _f(gap + z * gse) if gap is not None and gse is not None else None,
        "gap_lcb": _f(gap - z * gse) if gap is not None and gse is not None else None, "cells": cells,
        "half_a": _f(sum(ha) / len(ha)) if ha else None, "half_b": _f(sum(hb) / len(hb)) if hb else None, "halves_ok": half_ok,
        "oos_n": oos["n"], "oos_blocks": oos["n_blocks"], "oos_ev": _f(oos["ev"]),
        "oos_ucb": _f(oos["ev"] + float(P["z_oos"]) * oos["se"]) if oos["ev"] is not None and oos["se"] is not None else None,
        "oos_lcb": _f(oos["ev"] - float(P["z_oos"]) * oos["se"]) if oos["ev"] is not None and oos["se"] is not None else None,
        "n_pairs": len(pairs), "top_pair": top_pair, "top_pair_share": _f(top_n / len(F)) if F else None,
        "ev_without_top_pair": _f(sum(rest_nets) / len(rest_nets)) if rest_nets else None,
        "share": _f(len(F) / total) if total else None, "agg_f": _f(sum(it[1] for it in F)),
        "newest_ts": int(max(it[0]["entry_ts"] for it in F)) if F else None, "z": round(z, 2), "oos_frac": float(P["oos_frac"]),
    }

def loser_checks(st: Mapping[str, Any], P: Mapping[str, Any]) -> Dict[str, bool]:
    """Promotion standard: the restricted set is a proven loser and the evidence is not a fluke."""
    return {
        "enough_blocks": bool(st["n_blocks"] >= int(P["min_blocks"]) and st["n"] >= int(P["min_n"])),
        "enough_pairs": bool(st["n_pairs"] >= int(P["min_pairs"])),
        "proven_loser": bool(st["ucb"] is not None and st["ucb"] < 0),
        "worse_than_rest": bool(st["gap_ucb"] is not None and st["gap_ucb"] < 0),
        "stable_both_halves": bool(st["halves_ok"] and st["half_a"] is not None and st["half_b"] is not None and st["half_a"] < 0 and st["half_b"] < 0),
        "recent_loser": bool(st["oos_blocks"] >= int(P["oos_min_blocks"]) and st["oos_ucb"] is not None and st["oos_ucb"] < 0),
        "not_one_pair": bool(st["top_pair_share"] is not None and st["top_pair_share"] <= float(P["max_pair_share"])
                             and st["ev_without_top_pair"] is not None and st["ev_without_top_pair"] < 0),
        "bounded_opportunity_cost": bool(st["share"] is not None and st["share"] <= float(P["max_share"])),
    }


def earner_checks(st: Mapping[str, Any], P: Mapping[str, Any]) -> Dict[str, bool]:
    """Demotion standard: what the gate blocks is a proven earner (the gate is costing money)."""
    return {
        "enough_blocks": bool(st["n_blocks"] >= int(P["min_blocks"]) and st["n"] >= int(P["min_n"])),
        "enough_pairs": bool(st["n_pairs"] >= int(P["min_pairs"])),
        "proven_earner": bool(st["lcb"] is not None and st["lcb"] > 0),
        "stable_both_halves": bool(st["halves_ok"] and st["half_a"] is not None and st["half_b"] is not None and st["half_a"] > 0 and st["half_b"] > 0),
        "recent_earner": bool(st["oos_blocks"] >= int(P["oos_min_blocks"]) and st["oos_lcb"] is not None and st["oos_lcb"] > 0),
        "not_one_pair": bool(st["top_pair_share"] is not None and st["top_pair_share"] <= float(P["max_pair_share"])
                             and st["ev_without_top_pair"] is not None and st["ev_without_top_pair"] > 0),
    }


def tier_of(checks: Mapping[str, bool], streak: int, need: int, data_ok: bool) -> str:
    if not data_ok:
        return "E0"
    if not checks.get("enough_blocks", True):
        return "E1"
    if all(checks.values()):
        return "E4" if streak >= need else "E3"
    return "E2"


def _failed(checks: Mapping[str, bool]) -> List[str]:
    return [k for k, v in checks.items() if not v]


# ── gate populations: (blocked-or-would-be-blocked trades F, rest of the delivered book R) ──

class Ctx:
    """Everything a population needs, built once per cycle."""

    def __init__(self, rows: Sequence[Row], blob: Mapping[str, Any], fixed: Mapping[str, float], cost_pct: float,
                 family_fn: Optional[Callable[[str], str]], hyp_rules: Sequence[Mapping[str, Any]],
                 drop_verdicts: Sequence[str], veto_score: int, min_cats: int) -> None:
        self.rows, self.blob, self.fixed, self.cost = rows, blob, fixed, cost_pct
        self.family_fn, self.hyp_rules = family_fn, list(hyp_rules)
        self.drop, self.veto_score, self.min_cats = tuple(drop_verdicts), int(veto_score), int(min_cats)
        self.live_items = _items(_live(rows), fixed, cost_pct)


def pop_veto(c: Ctx) -> Tuple[Items, Items]:
    rows_v = _live(c.rows) + _shadow_tag(c.rows, "playbook_veto")     # not rows other gates blocked for their own reasons
    by_group, delivered, _since = restricted_by_group(c.blob, rows_v, fixed=c.fixed, cost_pct=c.cost, family_fn=c.family_fn)
    return [it for items in by_group.values() for it in items], delivered


def pop_firewall(c: Ctx) -> Tuple[Items, Items]:
    groups = verdict_items(c.rows, fixed=c.fixed, cost_pct=c.cost)
    F = [it for v in c.drop for it in groups.get(v, [])]
    fid = {id(it) for it in F}
    return F, [it for items in groups.values() for it in items if id(it) not in fid]


def _exh_pairs(c: Ctx) -> List[Tuple[int, int, Tuple[Row, float, float]]]:
    veto_rows = _shadow_tag(c.rows, "exhaustion_veto")
    out = []
    for it in _items(_live(c.rows) + veto_rows, c.fixed, c.cost):
        sc, ct = _ctx(it[0]).get("exh_score"), _ctx(it[0]).get("exh_cats")
        if isinstance(sc, bool) or not isinstance(sc, (int, float)):
            continue
        out.append((int(sc), int(ct) if isinstance(ct, (int, float)) and not isinstance(ct, bool) else 0, it))
    return out


def pop_exhaustion(c: Ctx, score: Optional[int] = None) -> Tuple[Items, Items]:
    t = c.veto_score if score is None else int(score)
    pts = _exh_pairs(c)
    return ([p[2] for p in pts if p[0] >= t and p[1] >= c.min_cats], [p[2] for p in pts if not (p[0] >= t and p[1] >= c.min_cats)])


def pop_gate(c: Ctx, name: str) -> Tuple[Items, Items]:
    a, b = f"{name}_gate_shadow", f"{name}_gate"
    def hit(r: Row) -> bool:
        gs = _ctx(r).get("gate_shadow")
        return isinstance(gs, dict) and bool(gs.get(a) or gs.get(b))
    return [it for it in c.live_items if hit(it[0])], [it for it in c.live_items if not hit(it[0])]


def pop_mlev(c: Ctx) -> Tuple[Items, Items]:
    def mv(r: Row) -> Optional[bool]:
        m = _ctx(r).get("ml_ev_shadow")
        return bool(m.get("would_block")) if isinstance(m, dict) and "would_block" in m else None
    F = [it for it in c.live_items if mv(it[0]) is True] + _items(_shadow_tag(c.rows, "ml_ev_gate"), c.fixed, c.cost)
    R = [it for it in c.live_items if mv(it[0]) is False]
    return F, R


def pop_hypothesis(c: Ctx) -> Tuple[Items, Items]:
    if not c.hyp_rules:
        return [], []
    rules = [dict(r) for r in c.hyp_rules]
    since = min(float(r.get("since_ts") or 0) for r in rules)
    F: Items = []
    R: Items = []
    for it in _items(_live(c.rows) + _shadow_tag(c.rows, "hypothesis_veto"), c.fixed, c.cost):
        r = it[0]
        if r["entry_ts"] <= since or not isinstance(r.get("context"), dict):
            continue
        feats = row_features(r, setup=True)
        if any(rules_hit([rule], feats) is not None and r["entry_ts"] > float(rule.get("since_ts") or 0) for rule in rules):
            F.append(it)
        elif r.get("source") == "LIVE":
            R.append(it)
    return F, R


def pop_winrate(c: Ctx) -> Tuple[Items, Items]:
    return _items(_shadow_tag(c.rows, "win_rate_filter"), c.fixed, c.cost), list(c.live_items)


# field -> (kind, on value, off value, population fn, label, can_promote)
GATES: Dict[str, Dict[str, Any]] = {
    "ALERT_VETO_MODE": {"kind": "mode", "pop": pop_veto, "label": "playbook veto (AVOID groups)", "promote": True},
    "ALERT_FIREWALL_MODE": {"kind": "mode", "pop": pop_firewall, "label": "delivery firewall (AVOID verdicts)", "promote": True},
    "EXHAUSTION_MODE": {"kind": "mode", "pop": pop_exhaustion, "label": "late-trend exhaustion veto", "promote": True},
    "ALERT_HYPOTHESIS_VETO_MODE": {"kind": "mode", "pop": pop_hypothesis, "label": "self-generated skip rules", "promote": True},
    "REGIME_GATE_MODE": {"kind": "mode", "pop": lambda c: pop_gate(c, "regime"), "label": "regime gate", "promote": True, "acts": "downgrade"},
    "ENSEMBLE_GATE_MODE": {"kind": "mode", "pop": lambda c: pop_gate(c, "ensemble"), "label": "ensemble gate", "promote": True, "acts": "downgrade"},
    "EPISTEMIC_GATE_MODE": {"kind": "mode", "pop": lambda c: pop_gate(c, "epistemic"), "label": "novelty (epistemic) gate", "promote": True, "acts": "downgrade"},
    "UNCERTAINTY_GATE_MODE": {"kind": "mode", "pop": lambda c: pop_gate(c, "uncertainty"), "label": "uncertainty gate", "promote": True, "acts": "downgrade"},
    "ENABLE_ML_EV_GATE": {"kind": "bool", "pop": pop_mlev, "label": "ML expected-value hard gate", "promote": True},
    "ENABLE_WIN_RATE_FILTER": {"kind": "bool", "pop": pop_winrate, "label": "pair+key win-rate filter", "promote": False},
}


WATCHED_FIELDS: Tuple[str, ...] = tuple(GATES) + (
    "EXHAUSTION_VETO_SCORE", "CONFLUENCE_MIN_PCT", "PLAYBOOK_MODE", "AUTOTUNE_MODE", "VETO_MONITOR_MODE")


def current_values(get: Callable[[str], Any], override: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The value each watched field has RIGHT NOW: `get(name)` reads the loaded config, and a Redis override (the only
    thing that can differ from the file, and only for the two allow-listed fields) wins."""
    out: Dict[str, Any] = {}
    for f in WATCHED_FIELDS:
        try:
            out[f] = get(f)
        except AttributeError:
            continue
    for k, v in (override or {}).items():
        if k in out:
            out[k] = v
    return out


def _is_on(spec: Mapping[str, Any], cur: Any) -> Optional[bool]:
    """True = restricting, False = measuring only, None = not evaluated (mode 'off' / unknown)."""
    if spec["kind"] == "bool":
        return bool(cur) if isinstance(cur, bool) else None
    return {"live": True, "shadow": False}.get(str(cur))


def _target(spec: Mapping[str, Any], turn_on: bool) -> Any:
    if spec["kind"] == "bool":
        return bool(turn_on)
    return "live" if turn_on else "shadow"


def _kind_of(spec: Mapping[str, Any], turn_on: bool) -> str:
    if spec["kind"] == "bool":
        return "bool_flip"
    return "shadow_to_live" if turn_on else "live_to_shadow"


# ── suggestion construction ───────────────────────────────────────────────────

def _stat_lines(st: Mapping[str, Any], what: str) -> List[str]:
    def p(v: Any) -> str:
        return "n/a" if v is None else f"{v:+.2f}%"
    return [
        f"{what}: n={st['n']} in {st['n_blocks']} independent 3h blocks, {st['n_pairs']} pairs, average {p(st['ev'])}/trade "
        f"(upper {p(st['ucb'])}, lower {p(st['lcb'])} at z={st['z']})",
        f"rest of the book: n={st['n_rest']}, average {p(st['rest_ev'])}/trade; gap {p(st['gap'])} (upper {p(st['gap_ucb'])})",
        f"older half {p(st['half_a'])} | newer half {p(st['half_b'])} | latest {int(round(100 * float(st['oos_frac'])))}% alone {p(st['oos_ev'])} "
        f"(n={st['oos_n']}, {st['oos_blocks']} blocks)",
        f"concentration: top pair {st['top_pair']} = {0 if st['top_pair_share'] is None else round(100 * st['top_pair_share'])}% of the set; "
        f"without it {p(st['ev_without_top_pair'])}/trade; set is {0 if st['share'] is None else round(100 * st['share'])}% of the book",
    ]


def _sugg(*, field: str, cur: Any, new: Any, kind: str, cid: str, group: str, direction: str, title: str,
          tier: str, passed: bool, checks: Mapping[str, bool], stats: Optional[Mapping[str, Any]], lines: Sequence[str],
          risks: Sequence[str], alternatives: Sequence[str], gain: float, need: int, streak: int,
          rollback: Any = None, prio: int = 5) -> Dict[str, Any]:
    return {
        "id": _fp(field, cur, new), "cid": cid, "group": group, "field": field, "from": cur, "to": new, "kind": kind,
        "direction": direction, "title": title, "tier": tier, "passed": bool(passed), "checks": dict(checks),
        "failed": _failed(checks), "stats": dict(stats or {}), "lines": list(lines), "risks": list(risks),
        "alternatives": list(alternatives), "gain": float(gain), "need": int(need), "streak": int(streak),
        "rollback": cur if rollback is None else rollback, "prio": int(prio),
    }


# ── candidate generators ──────────────────────────────────────────────────────

def _gate_candidates(c: Ctx, current: Mapping[str, Any], P: Mapping[str, Any], cand: Mapping[str, Any],
                     data_ok: bool, z_prom: float) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for field, spec in GATES.items():
        if field in NEVER_TOUCH or field not in current:
            continue
        on = _is_on(spec, current[field])
        if on is None:
            continue
        F, R = spec["pop"](c)
        if not F:
            continue
        turn_on = not on
        if turn_on and not spec["promote"]:
            continue
        z = z_prom if turn_on else float(P["z_demote"])
        st = restriction_stats(F, R, P, z)
        checks = loser_checks(st, P) if turn_on else earner_checks(st, P)
        cid = f"{field}:{'on' if turn_on else 'off'}"
        need = int(P["promote_cycles"] if turn_on else P["demote_cycles"])
        streak = int((cand.get(cid) or {}).get("streak") or 0)
        tier = tier_of(checks, streak, need, data_ok)          # provisional; advise() recomputes it after this cycle's streak update
        new = _target(spec, turn_on)
        dg = spec.get("acts") == "downgrade"          # verdict gates lower a verdict one step; the others block the alert
        if turn_on:
            title = f"Turn ON the {spec['label']}: {field} {_fmt_val(current[field])} -> {_fmt_val(new)}"
            why = (f"The trades it would {'downgrade' if dg else 'block'} lose money, clearly worse than the rest of the book.")
            risks = [("Some alerts would read one verdict step lower (and may then be dropped by a verdict-based firewall) although they would have won."
                      if dg else "May suppress some profitable marginal trades (see 'share of the book')."),
                     "The edge was measured on history and can disappear as the market changes; it is re-checked every cycle and withdrawn if it stops holding."]
            what = f"trades the gate would {'downgrade' if dg else 'block'}"
        else:
            title = f"Turn OFF the {spec['label']}: {field} {_fmt_val(current[field])} -> {_fmt_val(new)}"
            why = f"The trades it {'downgrades' if dg else 'blocks'} would have MADE money; the gate is costing profit."
            risks = [("Those alerts keep their higher verdict again, including the occasional large loss."
                      if dg else "Removing the gate lets those trades through again, including the occasional large loss."),
                     "Other gates and the kill switch are unaffected."]
            what = f"trades the gate {'downgraded' if dg else 'blocked'}"
        lines = [why] + _stat_lines(st, what)
        gain = abs(float(st["agg_f"] or 0.0))
        out.append(_sugg(field=field, cur=current[field], new=new, kind=_kind_of(spec, turn_on), cid=cid, group="EXHAUSTION" if field == "EXHAUSTION_MODE" else field,
                         direction="PROMOTE" if turn_on else "DEMOTE", title=title, tier=tier, passed=all(checks.values()),
                         checks=checks, stats=st, lines=lines, risks=risks, alternatives=[], gain=gain, need=need, streak=streak))
    return out


def _sweep_eval(cands: Sequence[Tuple[Any, Items, Items]], P: Mapping[str, Any], z: float) -> Dict[Any, Dict[str, Any]]:
    """Select on the older part of the book, confirm on the newest part.
    cands = [(label, restricted set F, rest R)].  Every candidate is drawn from the SAME book (F + R is identical for
    all labels), so one time cut serves them all.  Selection looks ONLY at trades before the cut; the newest
    `oos_frac` of the book (the holdout) is looked at afterwards, once, for the candidate that was chosen.
    The size minimums are scaled to the older part, so the total data needed stays about the same."""
    out: Dict[Any, Dict[str, Any]] = {}
    if not cands:
        return out
    book = sorted(float(it[0]["entry_ts"]) for it in list(cands[0][1]) + list(cands[0][2]))
    if len(book) < 4:
        return out
    frac = 1.0 - float(P["oos_frac"])
    cut = book[min(len(book) - 1, int(len(book) * frac))]
    Pt = dict(P)
    Pt["min_blocks"] = max(1, int(math.ceil(int(P["min_blocks"]) * frac)))
    Pt["min_n"] = max(1, int(math.ceil(int(P["min_n"]) * frac)))
    z_oos = float(P["z_oos"])
    for label, F, R in cands:
        Ft = [it for it in F if float(it[0]["entry_ts"]) < cut]
        Rt = [it for it in R if float(it[0]["entry_ts"]) < cut]
        Fh = [it for it in F if float(it[0]["entry_ts"]) >= cut]
        if not Ft:
            continue
        train = restriction_stats(Ft, Rt, Pt, z)
        checks = {k: v for k, v in loser_checks(train, Pt).items() if k != "recent_loser"}   # the holdout replaces the recency check
        hp = _fwd_pool(Fh)
        hev, hse = hp["ev"], hp["se"]
        hold = {"n": hp["n"], "blocks": hp["n_blocks"], "ev": _f(hev), "z": z_oos,
                "ucb": _f(hev + z_oos * hse) if hev is not None and hse is not None else None}
        hold_ok = bool(hold["blocks"] >= int(P["oos_min_blocks"]) and hold["ucb"] is not None and hold["ucb"] < 0)
        out[label] = {"train": train, "checks": checks, "hold": hold, "hold_ok": hold_ok, "cut": cut,
                      "full": restriction_stats(F, R, P, z)}
    return out


def _holdout_line(e: Mapping[str, Any], P: Mapping[str, Any]) -> str:
    h = e["hold"]
    return (f"untouched holdout (newest {int(round(100 * float(P['oos_frac'])))}% of the book by time, tested once after the choice): "
            f"n={h['n']} in {h['blocks']} blocks, average {_pct(h['ev'])} (upper {_pct(h['ucb'])} at z={h['z']})")


def _exhaustion_value(c: Ctx, current: Mapping[str, Any], P: Mapping[str, Any], cand: Mapping[str, Any],
                      data_ok: bool, z_prom: float) -> Tuple[List[Dict[str, Any]], List[str]]:
    field = "EXHAUSTION_VETO_SCORE"
    if field in NEVER_TOUCH or field not in current or str(current.get("EXHAUSTION_MODE")) == "off":
        return [], []
    cur = int(current[field])
    cands: List[Tuple[Any, Items, Items]] = []
    for t in sorted({int(x) for x in P["veto_scores"]} | {cur}):
        F, R = pop_exhaustion(c, t)
        if F:
            cands.append((t, F, R))
    per = _sweep_eval(cands, P, z_prom)
    notes = [f"score {t}: {'passes selection' if all(e['checks'].values()) else 'fails ' + ', '.join(_failed(e['checks']))} "
             f"(blocks {e['full']['n']}, avg {_pct(e['full']['ev'])})" for t, e in sorted(per.items())]
    if cur not in per or all(per[cur]["checks"].values()):
        return [], notes                        # the configured score is already a proven-loser cut (or has no data): nothing to change
    passing = [t for t, e in per.items() if all(e["checks"].values()) and t != cur]
    if not passing:
        return [], notes
    new = min(passing, key=lambda t: (abs(t - cur), -t))
    e = per[new]
    if not e["hold_ok"]:
        return [], notes + [f"score {new} was chosen on the older data but failed the untouched holdout ({_holdout_line(e, P)}): not recommended"]
    st = dict(e["full"])
    st["holdout"], st["train_end_ts"] = e["hold"], int(e["cut"])
    checks = {**e["checks"], "holdout_loser": True}
    cid = f"{field}:{new}"
    streak = int((cand.get(cid) or {}).get("streak") or 0)
    need = int(P["promote_cycles"])
    tier = tier_of(checks, streak, need, data_ok)
    alts = [f"Not {t}: {'fails ' + ', '.join(_failed(x['checks']))}" for t, x in sorted(per.items()) if t not in (new, cur)]
    alts.append(f"Keep {cur}: fails {', '.join(_failed(per[cur]['checks'])) or 'nothing'}")
    lines = ([f"A veto at score {new} would block a set that was a proven loser on the older data AND on the untouched newest data; "
              f"the configured score {cur} would not."]
             + _stat_lines(st, f"trades with exhaustion score >= {new}") + [_holdout_line(e, P)])
    return [_sugg(field=field, cur=cur, new=new, kind="value_delta", cid=cid, group="EXHAUSTION", direction="VALUE",
                  title=f"Change the exhaustion veto score: {field} {cur} -> {new}", tier=tier, passed=all(checks.values()),
                  checks=checks, stats=st, lines=lines,
                  risks=["A different score changes how many alerts the exhaustion veto would stop once EXHAUSTION_MODE is live."],
                  alternatives=alts, gain=abs(float(st["agg_f"] or 0.0)), need=need, streak=streak)], notes


def _confluence_value(rows: Sequence[Row], current: Mapping[str, Any], P: Mapping[str, Any], cand: Mapping[str, Any],
                      fixed: Mapping[str, float], cost_pct: float, now_ts: float, data_ok: bool,
                      z_prom: float) -> Tuple[List[Dict[str, Any]], List[str]]:
    field = "CONFLUENCE_MIN_PCT"
    if field in NEVER_TOUCH or field not in current or str(current.get("AUTOTUNE_MODE")) == "live":
        return [], []                           # autotune owns this field while it is live: observe, never fight it
    cur = float(current[field])
    since = now_ts - int(P["lookback_days"]) * _DAY
    cands: List[Tuple[Any, Items, Items]] = []
    for k in range(1, int(P["max_steps"]) + 1):
        cand_pct = round(cur + k * float(P["step_pct"]), 4)
        if cand_pct > float(P["ceiling_pct"]):
            break
        removed, live = removed_band(rows, new_pct=cand_pct, macro_live=bool(P["macro_live"]), fixed=fixed, cost_pct=cost_pct, since_ts=since)
        rid = {id(it) for it in removed}
        kept = [it for it in live if id(it) not in rid]
        if removed:
            cands.append((cand_pct, removed, kept))
    per = _sweep_eval(cands, P, z_prom)
    notes = [f"{cp:g}: {'passes selection' if all(e['checks'].values()) else 'fails ' + ', '.join(_failed(e['checks']))} "
             f"(removes {e['full']['n']}, avg {_pct(e['full']['ev'])})" for cp, e in sorted(per.items())]
    passing = [cp for cp in sorted(per) if all(per[cp]["checks"].values())]
    if not passing:
        return [], notes
    new = passing[0]                            # smallest passing step: a larger one must not be "better" by assumption
    e = per[new]
    if not e["hold_ok"]:
        return [], notes + [f"{new:g} was chosen on the older data but failed the untouched holdout ({_holdout_line(e, P)}): not recommended"]
    st = dict(e["full"])
    st["holdout"], st["train_end_ts"] = e["hold"], int(e["cut"])
    checks = {**e["checks"], "holdout_loser": True}
    cid = f"{field}:{new:g}"
    streak = int((cand.get(cid) or {}).get("streak") or 0)
    need = int(P["promote_cycles"])
    tier = tier_of(checks, streak, need, data_ok)
    alts = [f"Not {cp:g}: {'a larger step is not required - the smallest step that passes is preferred' if all(x['checks'].values()) else 'fails ' + ', '.join(_failed(x['checks']))}"
            for cp, x in sorted(per.items()) if cp != new]
    lines = ([f"Raising the confluence floor by {new - cur:g} point(s) would have removed a band of delivered alerts that was a proven loser "
              f"on the older data AND on the untouched newest data."] + _stat_lines(st, f"alerts below {new:g}%") + [_holdout_line(e, P)])
    return [_sugg(field=field, cur=cur, new=new, kind="value_delta", cid=cid, group="CONFLUENCE", direction="VALUE",
                  title=f"Raise the confluence floor: {field} {cur:g} -> {new:g}", tier=tier, passed=True, checks=checks, stats=st,
                  lines=lines, risks=["Fewer alerts: the removed band is a share of what you receive today (see 'share of the book').",
                                      "Tighten only: the adviser never recommends loosening this field."]
                  + ([f"A Redis override of {field} is active (written by the Brain or autotune) and wins over config_macd.json: remove that override or the edit has no effect."]
                     if field in set(P["overridden"]) else []),
                  alternatives=alts, gain=abs(float(st["agg_f"] or 0.0)), need=need, streak=streak)], notes


def _agreement_candidates(current: Mapping[str, Any], P: Mapping[str, Any], cand: Mapping[str, Any], data_ok: bool,
                          sb: Optional[Mapping[str, Any]], trend: Optional[Mapping[str, Any]], blob: Mapping[str, Any],
                          veto_res: Optional[Mapping[str, Any]], at_report: Optional[Mapping[str, Any]],
                          at_streak: int, newest_ts: Optional[float] = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    need = int(P["promote_cycles"])
    # A streak must mean "passed again on NEWER trades", exactly as for the gate tests.
    # Without newest_ts in stats, _update_candidates treats every cycle as a refresh and
    # three idle learner runs would reach E4 on unchanged evidence.
    _stamp: Dict[str, Any] = {"newest_ts": int(newest_ts) if newest_ts is not None else None}

    # PLAYBOOK_MODE shadow -> live: the pre-committed scoreboard is READY and nothing contradicts it.
    f = "PLAYBOOK_MODE"
    if f in current and f not in NEVER_TOUCH and str(current[f]) == "shadow" and sb:
        checks = {
            "scoreboard_ready": sb.get("status") == "READY",
            "control_available": bool((blob.get("control") or {}).get("available")),
            "delivered_not_deteriorating": trend is not None and trend.get("verdict") not in ("DETERIORATING", None),
            "data_ok": bool(data_ok),
        }
        cid = f"{f}:live"
        ps = all(checks.values())
        failed_crit = [c["id"] for side in ("take", "avoid") for c in (sb.get(side) or {}).get("criteria", []) if not c.get("pass")]
        lines = ["The learner's pre-committed scoreboard passes every criterion on forward trades only (restrict-only overlay).",
                 f"scoreboard {sb.get('status')} (criteria hash {sb.get('criteria_hash')}); unmet: {', '.join(failed_crit) or 'none'}; "
                 f"delivered trend: {(trend or {}).get('verdict')}"]
        out.append(_sugg(field=f, cur="shadow", new="live", kind="shadow_to_live", cid=cid, group=f, direction="PROMOTE",
                         title=f"Turn the playbook overlay on: {f} shadow -> live", tier="E2",
                         passed=ps, checks=checks, stats=_stamp, lines=lines,
                         risks=["Restrict-only: the playbook can only mark an alert AVOID/WATCH or add a skip rule, never add or upgrade one."],
                         alternatives=[], gain=0.0, need=need, streak=int((cand.get(cid) or {}).get("streak") or 0)))

    # AUTOTUNE_MODE shadow -> live: its own bounded proposal passed in shadow on consecutive cycles.
    f = "AUTOTUNE_MODE"
    if f in current and f not in NEVER_TOUCH and str(current[f]) == "shadow" and at_report is not None:
        checks = {"would_apply_now": at_report.get("status") == "WOULD_APPLY", "data_ok": bool(data_ok),
                  "autotune_streak": int(at_streak) >= 2}
        cid = f"{f}:live"
        ps = all(checks.values())
        ev = at_report.get("evidence") or {}
        out.append(_sugg(field=f, cur="shadow", new="live", kind="shadow_to_live", cid=cid, group="CONFLUENCE", direction="PROMOTE",
                         title=f"Let autotune apply its bounded tightening: {f} shadow -> live",
                         tier="E2", passed=ps, checks=checks, stats=_stamp,
                         lines=["Autotune's own proposal (CONFLUENCE_MIN_PCT, tighten-only, one small step) cleared all its readiness checks.",
                                str(at_report.get("text") or ""), f"evidence: removed band {ev.get('removed')}"],
                         risks=["Autotune judges every change after AUTOTUNE_EVAL_HOURS and rolls it back on clear harm; it cannot relax the field."],
                         alternatives=[], gain=0.0, need=need, streak=int((cand.get(cid) or {}).get("streak") or 0)))

    # VETO_MONITOR_MODE -> live: a veto that is still blocking was judged harmful but the monitor may not act.
    f = "VETO_MONITOR_MODE"
    if f in current and f not in NEVER_TOUCH and str(current[f]) != "live" and str(current.get("ALERT_VETO_MODE")) == "live" and veto_res:
        harmful = {k: g for k, g in (veto_res.get("groups") or {}).items() if g.get("verdict") == "HARMFUL" and int(g.get("harm_streak") or 0) >= int(veto_res.get("harm_cycles") or 2)}
        checks = {"harmful_rule_seen": bool(harmful), "data_ok": bool(data_ok)}
        cid = f"{f}:live"
        ps = all(checks.values())
        need_d = int(P["demote_cycles"])
        out.append(_sugg(field=f, cur=current[f], new="live", kind="shadow_to_live", cid=cid, group="ALERT_VETO_MODE", direction="DEMOTE",
                         title=f"Let the monitor suspend a harmful veto rule: {f} {current[f]} -> live",
                         tier="E2", passed=ps, checks=checks, stats=_stamp, prio=1,
                         lines=["The veto monitor judged these live veto rule(s) HARMFUL on consecutive cycles but is not allowed to suspend them:"]
                         + [f"{k}: {g.get('reason') or 'see monitor'}" for k, g in harmful.items()],
                         risks=["The monitor suspends ONE rule's veto (it logs but still sends) and resumes it only after proven benefit again."],
                         alternatives=[], gain=0.0, need=need_d, streak=int((cand.get(cid) or {}).get("streak") or 0)))
    return out


# ── life-cycle: fingerprints, cool-downs, expiry, revocation, verification ───

def before_after(rows: Sequence[Row], applied_ts: float, now_ts: float, fixed: Mapping[str, float], cost_pct: float,
                 min_blocks: int) -> Dict[str, Any]:
    """Delivered (LIVE) trades since a change vs the same span before it, on the fixed bracket after costs."""
    span = now_ts - applied_ts
    post: Items = []
    pre: Items = []
    for it in _items(_live(rows), fixed, cost_pct):
        ts = float(it[0]["entry_ts"])
        if applied_ts < ts <= now_ts:
            post.append(it)
        elif applied_ts - span < ts <= applied_ts:
            pre.append(it)
    a, b = _fwd_pool(post), _fwd_pool(pre)
    gap, se, _lb = _gap(a, b)
    out: Dict[str, Any] = {"post": {"n": a["n"], "blocks": a["n_blocks"], "ev": _rnd(a["ev"])}, "pre": {"n": b["n"], "blocks": b["n_blocks"], "ev": _rnd(b["ev"])},
                           "gap": _rnd(gap), "ready": bool(gap is not None and a["n_blocks"] >= min_blocks and b["n_blocks"] >= min_blocks)}
    if out["ready"] and gap is not None and se is not None:
        z = 1.645
        out["verdict"] = "CONFIRMED" if gap - z * se > 0 else "HARMFUL" if gap + z * se < 0 else "NEUTRAL"
    return out


def _match_current(rec: Mapping[str, Any], cur_value: Any) -> str:
    """'proposed' | 'original' | 'other' - where the config is now relative to a recommendation."""
    def same(a: Any, b: Any) -> bool:
        if isinstance(a, bool) or isinstance(b, bool):
            return isinstance(a, bool) and isinstance(b, bool) and a == b
        try:
            return abs(float(a) - float(b)) < 1e-9
        except (TypeError, ValueError):
            return str(a) == str(b)
    if same(cur_value, rec["to"]):
        return "proposed"
    if same(cur_value, rec["from"]):
        return "original"
    return "other"


def _new_state() -> Dict[str, Any]:
    return {"version": 1, "recs": {}, "cand": {}, "log": [], "last_digest_ts": 0, "last_digest_hash": "", "cycles": 0, "policy": ""}


def _log(state: Dict[str, Any], now_ts: float, event: str, rec: Mapping[str, Any], extra: str = "") -> None:
    state["log"].insert(0, {"ts": int(now_ts), "event": event, "id": rec.get("id"), "field": rec.get("field"),
                            "from": rec.get("from"), "to": rec.get("to"), "tier": rec.get("tier"), "note": extra})
    del state["log"][_LOG_CAP:]


def _update_candidates(state: Dict[str, Any], all_sugg: Sequence[Mapping[str, Any]], data_ok: bool) -> None:
    """Streaks count consecutive cycles that pass AND brought newer trades; a failing cycle resets, a stale pass holds."""
    cand: Dict[str, Any] = dict(state.get("cand") or {})
    seen = set()
    for s in all_sugg:
        cid = s["cid"]
        seen.add(cid)
        prev = cand.get(cid) or {}
        streak = int(prev.get("streak") or 0)
        newest = (s.get("stats") or {}).get("newest_ts")
        if not data_ok or not s["passed"]:
            streak = 0
        else:
            refreshed = newest is None or float(newest) > float(prev.get("newest_ts") or 0)
            if refreshed:
                streak += 1
        cand[cid] = {"streak": streak, "newest_ts": newest if newest is not None else prev.get("newest_ts"), "tier": s["tier"]}
    for cid in list(cand):
        if cid not in seen:
            cand[cid] = {**cand[cid], "streak": 0}
    state["cand"] = cand


def advise(*, now_ts: float, rows: Sequence[Row], blob: Mapping[str, Any], sb: Optional[Mapping[str, Any]],
           trend: Optional[Mapping[str, Any]], veto_res: Optional[Mapping[str, Any]], at_report: Optional[Mapping[str, Any]],
           at_streak: int, hyp_rules: Sequence[Mapping[str, Any]], current: Mapping[str, Any], fixed: Mapping[str, float],
           cost_pct: float, data_ok: bool, newest_ts: Optional[float], n_rows: int, n_path: int,
           prev_state: Optional[Mapping[str, Any]], params: Optional[Mapping[str, Any]] = None,
           family_fn: Optional[Callable[[str], str]] = None, drop_verdicts: Sequence[str] = ("AVOID",),
           deliver: bool = True) -> Dict[str, Any]:
    """One evaluation.  Returns {"state", "to_send": [alert dicts], "digest": text|None, "suggestions": [...], "notes": [...], "flags": {...}}.
    Nothing is written anywhere: the caller sends the alerts and then calls `mark_sent`.
    deliver=False is shadow mode: a recommendation is evaluated and tracked but never re-offered (state SHADOW).
    deliver=True re-arms the SHADOW ones that still pass this cycle, so switching shadow -> on really delivers them."""
    P = default_params()
    P.update(params or {})
    state = _new_state()
    state.update({k: v for k, v in (prev_state or {}).items() if k in state})
    state["recs"] = {k: dict(v) for k, v in (state.get("recs") or {}).items()}
    state["cand"] = dict(state.get("cand") or {})
    state["log"] = list(state.get("log") or [])
    state["cycles"] = int(state.get("cycles") or 0) + 1

    # An earlier version stored shadow-mode recommendations as SENT although nothing was delivered. Repair them once:
    # they must not read as "sent" (no WITHDRAWN notice for something you never saw, and they are re-offered when you switch on).
    _real = {e.get("id") for e in state["log"] if e.get("event") == "SENT"}
    _shadowed = {e.get("id") for e in state["log"] if e.get("event") == "SENT_SHADOW"}
    for _rec in state["recs"].values():
        if _rec.get("state") == "SENT" and _rec.get("id") in _shadowed and _rec.get("id") not in _real:
            _rec.update({"state": "SHADOW", "sent_ts": None})
    # Streaks belong to the evidence rules that produced them: when the policy changes they start again.
    policy = _policy_hash(P)
    if state.get("policy") != policy:
        state["cand"] = {k: {**v, "streak": 0} for k, v in state["cand"].items()}
        if state.get("policy"):
            _log(state, now_ts, "POLICY_CHANGED", {"id": None, "field": None}, f"{state.get('policy')} -> {policy}: streaks restart")
        state["policy"] = policy

    fresh = newest_ts is not None and (now_ts - float(newest_ts)) <= float(P["max_age_h"]) * 3600.0
    coverage = n_rows > 0 and n_path / n_rows >= float(P["min_path_share"])
    healthy = bool(data_ok and fresh and coverage)
    flags = {"fresh": bool(fresh), "coverage": bool(coverage), "healthy": healthy}

    # evaluated gates this cycle -> Bonferroni z for the promotion tests
    # Every promotion-type test of this cycle counts: each gate that is only measuring, plus every candidate of the two sweeps.
    m = (sum(1 for f, sp in GATES.items() if f in current and _is_on(sp, current[f]) is False and sp["promote"])
         + len({int(x) for x in P["veto_scores"]} | {int(current.get("EXHAUSTION_VETO_SCORE", P["veto_score"]))}) + int(P["max_steps"]))
    z_prom = max(float(P["z_floor"]), _z(float(P["alpha"]), m))
    ctx = Ctx(rows, blob, fixed, cost_pct, family_fn, hyp_rules, drop_verdicts, int(current.get("EXHAUSTION_VETO_SCORE", P["veto_score"])), int(P["min_cats"]))
    cand = state["cand"]
    suggs: List[Dict[str, Any]] = []
    notes: List[str] = []
    suggs += _gate_candidates(ctx, current, P, cand, healthy, z_prom)
    ev_s, ev_notes = _exhaustion_value(ctx, current, P, cand, healthy, z_prom)
    cf_s, cf_notes = _confluence_value(rows, current, P, cand, fixed, cost_pct, now_ts, healthy, z_prom)
    suggs += ev_s + cf_s
    suggs += _agreement_candidates(current, P, cand, healthy, sb, trend, blob, veto_res, at_report, at_streak,
                                   newest_ts=newest_ts)
    suggs = [s for s in suggs if s["field"] not in NEVER_TOUCH and s["field"] not in set(P["mute"])]
    if ev_notes and not ev_s:
        notes.append("EXHAUSTION_VETO_SCORE: no change - " + "; ".join(ev_notes))
    if cf_notes and not cf_s:
        notes.append("CONFLUENCE_MIN_PCT: no change - " + "; ".join(cf_notes))
    _update_candidates(state, suggs, healthy)
    for s in suggs:                                   # the tier that counts includes this cycle's refresh
        s["streak"] = int(state["cand"][s["cid"]]["streak"])
        s["tier"] = tier_of(s["checks"], s["streak"], s["need"], healthy)

    recs = state["recs"]
    by_id = {s["id"]: s for s in suggs}
    to_send: List[Dict[str, Any]] = []
    horizon = float(P["expiry_days"]) * _DAY

    # 1. existing recommendations: applied / stale / expired / revoked / verified
    for rid, rec in list(recs.items()):
        field = rec["field"]
        where = _match_current(rec, current[field]) if field in current else "other"
        st = rec["state"]
        if st in _LIVE_STATES:
            if where == "proposed":
                rec.update({"state": "VERIFYING", "applied_ts": int(now_ts)})
                _log(state, now_ts, "APPLIED", rec, "config reached the proposed value")
            elif where == "other":
                rec["state"] = "STALE"
                _log(state, now_ts, "STALE", rec, "config changed to a different value")
            elif now_ts >= float(rec["expires_ts"]):
                rec["state"] = "EXPIRED"
                rec["expired_n"] = int(rec.get("expired_n") or 0) + 1
                if rec["expired_n"] >= 2:
                    rec["snooze_until"] = int(now_ts + float(P["snooze_days"]) * _DAY)
                _log(state, now_ts, "EXPIRED", rec)
            else:
                cur = by_id.get(rid)
                if cur is not None and _TIER_ORDER[cur["tier"]] >= 3:
                    rec["miss"] = 0
                    if _TIER_ORDER[cur["tier"]] > _TIER_ORDER[rec["tier"]]:
                        _log(state, now_ts, "ESCALATED", rec, f"{rec['tier']} -> {cur['tier']}")
                    rec.update({"tier": cur["tier"], "lines": cur["lines"], "stats": cur["stats"], "checks": cur["checks"], "last_eval_ts": int(now_ts)})
                elif not healthy:
                    # E0 means the DATA is not trustworthy this cycle, not that the
                    # evidence went away. Hold the recommendation and judge it again
                    # on the next healthy cycle; never withdraw on a data outage.
                    rec["held_ts"] = int(now_ts)
                else:
                    rec["miss"] = int(rec.get("miss") or 0) + 1
                    if rec["miss"] >= int(P["revoke_cycles"]):
                        why = ", ".join((cur or {}).get("failed") or []) or "no longer proposed"
                        rec["state"] = "REVOKED"
                        rec["revoked_reason"] = why
                        _log(state, now_ts, "REVOKED", rec, why)
                        if rec.get("sent_ts"):
                            to_send.append({"type": "WITHDRAWN", "id": rid, "text": format_withdrawn(rec, why)})
        elif st == "VERIFYING":
            if where == "original":
                rec["state"] = "REVERTED"
                _log(state, now_ts, "REVERTED", rec, "config returned to the original value")
            elif where == "other":
                rec["state"] = "STALE"
                _log(state, now_ts, "STALE", rec, "config changed again during verification")

            elif healthy and now_ts - float(rec["applied_ts"]) >= float(P["verify_days"]) * _DAY:
                v = before_after(rows, float(rec["applied_ts"]), now_ts, fixed, cost_pct, int(P["verify_min_blocks"]))
                rec["verify"] = v
                if v.get("verdict"):
                    rec["state"] = v["verdict"]
                    _log(state, now_ts, v["verdict"], rec, f"delivered EV {v['post']['ev']}% after vs {v['pre']['ev']}% before")
                    if v["verdict"] == "HARMFUL":
                        to_send.append({"type": "ROLLBACK", "id": rid, "text": format_rollback(rec, v)})
                elif now_ts - float(rec["applied_ts"]) >= 3 * float(P["verify_days"]) * _DAY:
                    rec["state"] = "NEUTRAL"
                    _log(state, now_ts, "NEUTRAL", rec, "too few trades to judge; left as is")

    # 2. new / refreshed recommendations
    day_sent = sum(1 for e in state["log"] if e["event"] == "SENT" and now_ts - float(e["ts"]) < _DAY)
    budget = min(int(P["max_per_cycle"]), int(P["max_per_day"]) - day_sent)
    send_rank = _TIER_ORDER[str(P["send_tier"])]
    ranked = sorted([s for s in suggs if s["passed"] and _TIER_ORDER[s["tier"]] >= 3],
                    key=lambda s: (s["direction"] != "DEMOTE", -_TIER_ORDER[s["tier"]], s["prio"], -s["gain"]))
    used_groups = set()
    for s in ranked:
        rid = s["id"]
        rec = recs.get(rid)
        if rec is None or rec["state"] not in _LIVE_STATES:
            if rec is not None and rec.get("snooze_until") and now_ts < float(rec["snooze_until"]) and not (_TIER_ORDER[s["tier"]] > _TIER_ORDER.get(rec.get("tier", "E0"), 0)):
                continue
            if rec is not None and rec.get("sent_ts") and now_ts - float(rec["sent_ts"]) < float(P["cooldown_hours"]) * 3600.0:
                continue
            if any(r.get("group") == s["group"] and r["state"] in _LIVE_STATES for r in recs.values()):
                continue                                  # one open recommendation per group: no silent cascades
            rec = {"id": rid, "field": s["field"], "from": s["from"], "to": s["to"], "kind": s["kind"], "direction": s["direction"],
                   "state": "NEW", "created_ts": int(now_ts), "expires_ts": int(now_ts + horizon), "sent_ts": None, "miss": 0,
                   "rollback": s["rollback"], "title": s["title"], "alternatives": s["alternatives"], "risks": s["risks"],
                   "tier": s["tier"], "lines": s["lines"], "stats": s["stats"], "checks": s["checks"], "group": s["group"],
                   "expired_n": int((rec or {}).get("expired_n") or 0), "snooze_until": (rec or {}).get("snooze_until")}
            rec["prov"] = _provenance(rows, fixed, cost_pct, P, n_rows, n_path, newest_ts, policy)
            recs[rid] = rec
            _log(state, now_ts, "NEW", rec)
        if rec["state"] == "SHADOW" and deliver:          # shadow -> on: offer it again, but only because it passes THIS cycle
            rec["state"] = "NEW"
            _log(state, now_ts, "REARMED", rec, "tracked in shadow mode, still valid now that delivery is on")
        if rec["state"] != "NEW":
            continue
        if _TIER_ORDER[s["tier"]] >= send_rank and budget > 0 and s["group"] not in used_groups:
            to_send.append({"type": "RECOMMEND", "id": rid, "text": format_alert(rec, now_ts)})
            used_groups.add(s["group"])
            budget -= 1
    for rec in recs.values():                              # a trimmed history: keep terminal records a while, then drop
        if rec["state"] not in _LIVE_STATES + ("VERIFYING",) and now_ts - float(rec.get("created_ts") or 0) > 120 * _DAY:
            rec["_drop"] = True
    state["recs"] = {k: v for k, v in recs.items() if not v.get("_drop")}

    state["suggestion_view"] = [{"id": s["id"], "field": s["field"], "from": s["from"], "to": s["to"], "tier": s["tier"], "streak": s["streak"],
                                 "failed": s["failed"]} for s in suggs]
    digest = _digest(state, notes, P, now_ts, healthy, suggs)
    return {"state": state, "to_send": to_send, "digest": digest, "suggestions": suggs, "notes": notes, "flags": flags}


def mark_sent(state: Mapping[str, Any], ids: Sequence[str], now_ts: float, shadow: bool = False) -> Dict[str, Any]:
    """Call ONLY after Telegram accepted the message: until then a recommendation stays NEW and is retried next cycle.
    shadow=True (CONFIG_ADVISOR_MODE=shadow) records "would have been sent": the state becomes SHADOW, not SENT, because
    nothing was delivered. A SHADOW recommendation is offered for real when delivery is switched on and it still passes."""
    st = dict(state)
    recs = {k: dict(v) for k, v in (st.get("recs") or {}).items()}
    st["log"] = list(st.get("log") or [])
    for rid in ids:
        rec = recs.get(rid)
        if rec is not None and rec["state"] == "NEW":
            if shadow:
                rec.update({"state": "SHADOW", "shadow_ts": int(now_ts)})
                _log(st, now_ts, "SENT_SHADOW", rec)
            else:
                rec.update({"state": "SENT", "sent_ts": int(now_ts)})
                _log(st, now_ts, "SENT", rec)
    st["recs"] = recs
    return st


def mark_digest(state: Mapping[str, Any], digest_hash: str, now_ts: float) -> Dict[str, Any]:
    st = dict(state)
    st["last_digest_ts"] = int(now_ts)
    st["last_digest_hash"] = digest_hash
    return st


# ── text ──────────────────────────────────────────────────────────────────────

def _apply_how(rec: Mapping[str, Any]) -> str:
    return (f"How: edit config_macd.json, set \"{rec['field']}\" to {json_val(rec['to'])}, commit. The bot and the learner read it on their next run. "
            f"Nothing was changed automatically.")


def json_val(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return f"{v:g}"
    return f"\"{v}\""


def format_alert(rec: Mapping[str, Any], now_ts: float) -> str:
    tier_name = {"E3": "CANDIDATE", "E4": "STRONG"}.get(str(rec["tier"]), str(rec["tier"]))
    icon = "🔻" if rec["direction"] == "DEMOTE" else "🔧"
    head = f"{icon} CONFIG ADVISOR · {rec['tier']} {tier_name} · {rec['kind'].replace('_', ' ')}"
    lines = [head, rec["title"], ""]
    lines += [f"• {x}" for x in rec["lines"]]
    if rec.get("alternatives"):
        lines += ["", "Alternatives rejected:"] + [f"• {x}" for x in rec["alternatives"]]
    lines += ["", "Risk:"] + [f"• {x}" for x in rec["risks"]]
    lines += ["", f"Rollback: set {rec['field']} back to {json_val(rec['rollback'])}.",
              f"Review by: {_ts_str(float(rec['expires_ts']))} (the proposal expires then and is re-checked from scratch).",
              _apply_how(rec),
              "This is an evidence summary on past trades, not a forecast. If the evidence stops holding the advisor withdraws this."]
    return "\n".join(lines)


def format_withdrawn(rec: Mapping[str, Any], why: str) -> str:
    return (f"⚪ CONFIG ADVISOR · WITHDRAWN\n{rec['title']}\nThe evidence no longer passes ({why}). Do not apply it; if you already did, "
            f"the advisor will verify the result on real trades.")


def format_rollback(rec: Mapping[str, Any], v: Mapping[str, Any]) -> str:
    return (f"🔴 CONFIG ADVISOR · ROLLBACK RECOMMENDED\n{rec['field']} was changed {json_val(rec['from'])} -> {json_val(rec['to'])} "
            f"and delivered trades got clearly worse: {v['post']['ev']:+.2f}%/trade after (n={v['post']['n']}) vs {v['pre']['ev']:+.2f}% before (n={v['pre']['n']}).\n"
            f"Rollback: set \"{rec['field']}\" back to {json_val(rec['from'])} in config_macd.json and commit.\n"
            f"Caution: this compares delivered trades before vs after the change, and the market changed too. "
            f"Treat it as a warning to review, not proof the change caused it.")


def _digest(state: Mapping[str, Any], notes: Sequence[str], P: Mapping[str, Any], now_ts: float, healthy: bool,
            suggs: Sequence[Mapping[str, Any]]) -> Optional[str]:
    recs = state.get("recs") or {}
    open_r = [r for r in recs.values() if r["state"] == "SENT"]          # only what was actually recommended to you
    ver = [r for r in recs.values() if r["state"] == "VERIFYING"]
    open_ids = {r["id"] for r in open_r}
    watch = [s for s in suggs if _TIER_ORDER[s["tier"]] in (2, 3) or (s["tier"] == "E4" and s["id"] not in open_ids)]
    done: Dict[str, int] = {}
    for r in recs.values():
        if now_ts - float(r.get("created_ts") or 0) <= 90 * _DAY:
            done[r["state"]] = done.get(r["state"], 0) + 1
    lines = ["🧰 CONFIG ADVISOR · daily check"]
    if not healthy:
        lines.append("⚠️ Data is not trustworthy this cycle (stale, thin path coverage or a data issue): no new recommendations are made.")
    if open_r:
        lines.append("Open recommendations:")
        lines += [f"• {r['field']} {_fmt_val(r['from'])} -> {_fmt_val(r['to'])} ({r['tier']}, {r['state'].lower()}, review by {_ts_str(float(r['expires_ts']))})" for r in open_r]
    if ver:
        lines.append("Applied, being verified on real trades:")
        lines += [f"• {r['field']} -> {_fmt_val(r['to'])} (since {_ts_str(float(r['applied_ts']))})" for r in ver]
    if watch:
        lines.append("Watchlist (not strong enough to recommend):")
        for s in watch[:6]:
            lines.append(f"• {s['field']} {_fmt_val(s['from'])} -> {_fmt_val(s['to'])}: {s['tier']}, passed {s['streak']}/{s['need']} refreshes"
                         + (f"; failing: {', '.join(s['failed'])}" if s["failed"] else ""))
    if not (open_r or ver or watch) and notes:
        lines.append("No change recommended.")
    lines += [f"• {n}" for n in notes[:4]]
    if done:
        lines.append("Track record (90 days): " + ", ".join(f"{k.lower()} {v}" for k, v in sorted(done.items())))
    if len(lines) <= 1:
        return None
    return "\n".join(lines)


def digest_due(state: Mapping[str, Any], text: Optional[str], now_ts: float, hours: float) -> Tuple[bool, str]:
    """(due, hash): at most once per `hours`, and only when the content changed since the last digest."""
    if not text or hours <= 0:
        return False, ""
    h = hashlib.sha1(text.encode()).hexdigest()[:12]
    elapsed = now_ts - float(state.get("last_digest_ts") or 0)
    return (elapsed >= hours * 3600.0 and h != state.get("last_digest_hash")), h


def format_state(state: Mapping[str, Any]) -> str:
    """Console view for `learner.py --advisor`."""
    recs = sorted((state.get("recs") or {}).values(), key=lambda r: -float(r.get("created_ts") or 0))
    lines = [f"CONFIG ADVISOR state: {state.get('cycles', 0)} cycle(s), {len(recs)} recommendation(s)"]
    for r in recs[:20]:
        lines.append(f"  [{r['state']}] {r['field']} {_fmt_val(r['from'])} -> {_fmt_val(r['to'])} {r['tier']} id={r['id']}")
    for s in (state.get("suggestion_view") or [])[:20]:
        lines.append(f"  eval {s['field']} {_fmt_val(s['from'])} -> {_fmt_val(s['to'])}: {s['tier']} streak {s['streak']} failing={s['failed']}")
    for e in (state.get("log") or [])[:15]:
        lines.append(f"  log {_ts_str(float(e['ts']))} {e['event']} {e.get('field')} {e.get('note') or ''}")
    return "\n".join(lines)
