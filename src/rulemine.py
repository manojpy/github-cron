"""rulemine — shallow, restrict-only "when NOT to take it" rules.

Pure functions.  For one alert group it looks for small conditions (at most 3
atoms, one per feature) under which the alert LOST money, and keeps a rule only
if it also loses on later, unseen trades with an upper confidence bound below
zero.  The z used is already inflated for every rule tried (see playbook.py).

Features are the three that exist both in the archived outcome row and at alert
time, so a rule can be applied live without any new plumbing:
    session   asian | london | ny | dead
    conf      confluence score as a percent  (score / total * 100)
    adx       ADX value of the signal candle

A matching rule can only ever lower a TAKE to WATCH.  It never loosens anything.
"""
from __future__ import annotations

import itertools
import statistics
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from plan_replay import cluster_se

Atom = Tuple[str, str, Any]            # (feature, op, value)  op: eq | lt | ge
Rule = Tuple[Atom, ...]

_NUM_FEATURES = ("conf", "adx")
_CANDLE_SEC = 900


def row_ctx(r: Mapping[str, Any]) -> Dict[str, Any]:
    """Feature view of an outcome row (or of a live alert, same keys)."""
    conf = r.get("conf_pct")
    if conf is None and r.get("score") is not None and r.get("total"):
        conf = float(r["score"]) / float(r["total"]) * 100.0
    return {"session": r.get("session"), "conf": conf, "adx": r.get("adx_val")}


def atom_matches(a: Atom, ctx: Mapping[str, Any]) -> bool:
    v = ctx.get(a[0])
    if v is None or v == "" or v == "unknown":
        return False
    try:
        if a[1] == "eq":
            return str(v) == str(a[2])
        x = float(v)
        return x < float(a[2]) if a[1] == "lt" else x >= float(a[2])
    except (TypeError, ValueError):
        return False


def rule_matches(rule: Sequence[Sequence[Any]], ctx: Mapping[str, Any]) -> bool:
    return bool(rule) and all(atom_matches((a[0], a[1], a[2]), ctx) for a in rule)


def describe_rule(rule: Sequence[Sequence[Any]]) -> str:
    names = {"conf": "confluence", "adx": "ADX", "session": "session"}
    out = []
    for f, op, v in rule:
        if op == "eq":
            out.append(f"{names[f]} = {v}")
        else:
            val = f"{float(v):.0f}" if f == "conf" else f"{float(v):.1f}"
            out.append(f"{names[f]} {'<' if op == 'lt' else '≥'} {val}" + ("%" if f == "conf" else ""))
    return " and ".join(out)


def _quantiles(vals: List[float], qs: Sequence[float]) -> List[float]:
    s = sorted(vals)
    n = len(s)
    return sorted({round(s[min(n - 1, int(q * (n - 1)))], 4) for q in qs}) if n else []


def build_atoms(rows: Sequence[Mapping[str, Any]], min_support: int) -> List[Atom]:
    ctxs = [row_ctx(r) for r in rows]
    atoms: List[Atom] = []
    sess: Dict[str, int] = {}
    for c in ctxs:
        if c["session"] not in (None, "", "unknown"):
            sess[str(c["session"])] = sess.get(str(c["session"]), 0) + 1
    atoms += [("session", "eq", s) for s, k in sorted(sess.items()) if k >= min_support]
    for f in _NUM_FEATURES:
        vals = [float(c[f]) for c in ctxs if c[f] is not None and c[f] != ""]
        for t in _quantiles(vals, (0.25, 0.5, 0.75)):
            atoms.append((f, "lt", t))
            atoms.append((f, "ge", t))
    return atoms


def enumerate_rules(atoms: Sequence[Atom], max_depth: int = 3) -> List[Rule]:
    by_feat: Dict[str, List[Atom]] = {}
    for a in atoms:
        by_feat.setdefault(a[0], []).append(a)
    feats = sorted(by_feat)
    rules: List[Rule] = []
    for d in range(1, max_depth + 1):
        for fs in itertools.combinations(feats, d):
            for combo in itertools.product(*(by_feat[f] for f in fs)):
                rules.append(tuple(combo))
    return rules


def mine_rules(
    items: Sequence[Tuple[Mapping[str, Any], float]], *, z: float, max_rules: int = 120,
    min_train_n: int = 15, min_hold_n: int = 8, train_frac: float = 0.6, max_keep: int = 3,
    max_hold_cover: float = 0.7, embargo_sec: int = 14 * _CANDLE_SEC,
) -> Tuple[List[Dict[str, Any]], int]:
    """items = [(row, net_pnl_pct)] for ONE group under the plan being judged.
    Returns (rules, n_tested).  At most `max_rules` candidates are ever scored on
    the holdout, so the search space (and the multiple-testing price) is capped."""
    its = sorted((it for it in items if it[0].get("entry_ts") is not None), key=lambda it: it[0]["entry_ts"])
    n = len(its)
    if n < min_train_n + min_hold_n:
        return [], 0
    cut = int(n * train_frac)
    hold = its[cut:]
    if not hold:
        return [], 0
    split_ts = hold[0][0]["entry_ts"]
    train = [it for it in its[:cut] if it[0]["entry_ts"] <= split_ts - embargo_sec]
    if len(train) < min_train_n or len(hold) < min_hold_n:
        return [], 0
    atoms = build_atoms([r for r, _ in train], min_train_n)
    cands: List[Tuple[float, Rule]] = []
    tctx = [(row_ctx(r), p) for r, p in train]
    for rule in enumerate_rules(atoms):
        pn = [p for c, p in tctx if rule_matches(rule, c)]
        if len(pn) >= min_train_n:
            m = statistics.fmean(pn)
            if m < 0:
                cands.append((m, rule))
    cands.sort(key=lambda t: t[0])
    cands = cands[:max_rules]
    hctx = [(row_ctx(r), r.get("entry_ts"), p) for r, p in hold]
    found: List[Dict[str, Any]] = []
    for m_tr, rule in cands:
        sub = [(ts, p) for c, ts, p in hctx if rule_matches(rule, c)]
        if len(sub) < min_hold_n or len(sub) > max_hold_cover * len(hold):
            continue
        vals = [p for _, p in sub]
        m = statistics.fmean(vals)
        se, nb = cluster_se(vals, [ts for ts, _ in sub])
        if m < 0 and nb >= 3 and se != float("inf") and m + z * se < 0:
            found.append({
                "rule": [list(a) for a in rule], "text": describe_rule(rule),
                "n_train": sum(1 for c, _ in tctx if rule_matches(rule, c)), "ev_train": round(m_tr, 4),
                "n_hold": len(sub), "ev_hold": round(m, 4), "ucb": round(m + z * se, 4),
            })
    found.sort(key=lambda d: d["ucb"])
    kept: List[Dict[str, Any]] = []
    for f in found:                      # drop rules that merely narrow an already-kept rule
        atoms = {tuple(a) for a in f["rule"]}
        if any({tuple(a) for a in k["rule"]} <= atoms for k in kept):
            continue
        kept.append(f)
    return kept[:max_keep], len(cands)
