"""hypothesis_gen — gated, self-generated "skip when" hypotheses.

Pure functions: no Redis, no cfg, nothing is ever applied. Each cycle it
proposes a bounded number of new candidate rules, pushes them through a fixed
ladder, and remembers every one it ever tried:

  1. PROPOSE   up to `budget` rules of one or two conditions drawn from every
               numeric context field, vote, session and direction stored with
               the alerts. A condition is a fixed threshold (a quantile of the
               OLDER trades) or an on/off value, so a rule is fully defined the
               moment it is created. Rules already tried are never retried.
  2. SELECT    a rule must lose money on the older trades, then ALSO lose on the
               newer, untouched trades with an upper confidence bound below
               zero, where the confidence level is tightened for every rule
               ever tried (Bonferroni over the cumulative count). Rules that
               match too large a share of all trades are dropped.
  3. SHADOW    survivors are registered with a start time. From then on only
               trades that happen AFTER that time count (true forward test);
               at most `shadow_max` run at once.
  4. CONFIRM   a shadow rule is confirmed when enough forward trades exist and
               its upper bound is below zero (tightened by the number of rules
               running); REJECTED if forward trades did not lose money; EXPIRED
               if it never gathered enough. Confirmed rules keep being checked
               and are REVOKED if later trades stop losing.

This module never applies anything. The learner publishes confirmed rules
(`compact_rules`); the alert path may use them only when ALERT_HYPOTHESIS_VETO_MODE
is live. `direction="win"` runs the SAME ladder on negated P&L, so a confirmed rule
there is a 'take when' candidate: it is reported, never applied."""

from __future__ import annotations

import hashlib
import math
import random
from statistics import NormalDist
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from alert_registry import alert_family_of
from plan_replay import cluster_se
from threshold_stats import row_net_pnl_pct
from threshold_models import _PRICE_LEVEL_CTX

Row = Dict[str, Any]
Atom = Tuple[str, str, Any]          # (feature, op, value); op: ge | lt | eq
_CANDLE_SEC = 900
_MAX_TRIED = 5000

# Setup cohort keys (added by row_features(setup=True)): the alert family + side, the exact alert key, the pair.
COHORT_FEATURES = ("fam", "key", "pair")
_COHORT_LABEL = {"fam": "alert family", "key": "alert", "pair": "pair"}
# Shadow rows that may join the rule search. Only alerts that cleared every signal and confluence gate and were
# dropped by the per-run cap: an unbiased sample of what would have been delivered. Rows dropped by an
# outcome-aware control (playbook_veto, verdict_firewall) or by a feature-selecting gate (confluence_gate,
# ood_gate, calibration_gate, portfolio_heat) are left out on purpose. "hypothesis_veto" rows are added by the
# callers themselves (they keep their context).
SHADOW_POOL_REASONS = ("run_cap",)


def shadow_pool_rows(rows: Sequence[Row]) -> List[Row]:
    """The shadow rows allowed into the rule search (see SHADOW_POOL_REASONS)."""
    return [r for r in rows if r.get("rejection_reason") in SHADOW_POOL_REASONS]


def row_features(r: Row, setup: bool = False) -> Dict[str, Any]:
    """Everything knowable when the alert fired, flattened to one dict.
    setup=True (rule search, rule lift, live veto) also adds the cohort keys fam / key / pair and falls back to
    the row's top-level adx_val, so shadow rows that stored no context can still be mined and matched live."""
    f: Dict[str, Any] = {}
    for k, v in (r.get("context") or {}).items():
        if isinstance(v, bool) or k in _PRICE_LEVEL_CTX:
            continue
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            f["ctx:" + k] = float(v)
    for k, v in (r.get("votes") or {}).items():
        f["vote:" + k] = bool(v)
    if r.get("session"):
        f["session"] = r["session"]
    if r.get("direction"):
        f["dir"] = r["direction"]
    if setup:
        adx = r.get("adx_val")
        if ("ctx:adx_val" not in f and isinstance(adx, (int, float)) and not isinstance(adx, bool)
                and math.isfinite(float(adx))):
            f["ctx:adx_val"] = float(adx)
        atr, adxp = f.get("ctx:atr_pctl"), f.get("ctx:adx_pctl")
        if atr is not None and adxp is not None:
            # Market regime from two percentile ranks (0-1) that the live and the stored context both carry.
            f["regime"] = "volatile" if atr >= 0.8 else ("trending" if adxp >= 0.6 else "ranging")
        ak = r.get("alert_key")
        if ak:
            f["key"] = str(ak)
            fam = alert_family_of(str(ak))
            f["fam"] = f"{fam}|{r['direction']}" if r.get("direction") else fam
        if r.get("pair"):
            f["pair"] = str(r["pair"])
    return f


def _match(atom: Sequence[Any], f: Dict[str, Any]) -> bool:
    v = f.get(atom[0])
    if v is None:
        return False
    if atom[1] == "ge":
        return not isinstance(v, (str, bool)) and v >= atom[2]
    if atom[1] == "lt":
        return not isinstance(v, (str, bool)) and v < atom[2]
    return v == atom[2]

def rules_hit(rules: Sequence[Dict[str, Any]], feats: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """First published rule whose every condition matches `feats`, else None."""
    for rule in rules:
        atoms = rule.get("atoms") or []
        if atoms and all(_match(a, feats) for a in atoms):
            return rule
    return None


def setup_advice(feats: Dict[str, Any], win_rules: Optional[Sequence[Dict[str, Any]]],
                 loss_rules: Optional[Sequence[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """Advisory annotation for ONE alert from the published setup library, or None when nothing matches.
    kind 'win'  = a published take-side setup matches (ev = its later average net result, %);
    kind 'loss' = a published skip setup matches but did not drop the alert (veto mode is not live);
    kind 'mixed' = both match, so the evidence conflicts. Never raises, never changes the alert."""
    try:
        w = rules_hit(list(win_rules or []), feats)
        lo = rules_hit(list(loss_rules or []), feats)
    except Exception:
        return None
    hit = w if w is not None else lo
    if hit is None:
        return None
    if w is not None and lo is not None:
        return {"kind": "mixed", "id": str(w.get("id")), "text": str(w.get("text", "")), "ev": w.get("ev"),
                "n": w.get("fwd_n"), "loss_id": str(lo.get("id")), "loss_text": str(lo.get("text", "")),
                "loss_ev": lo.get("fwd_mean")}
    kind = "win" if w is not None else "loss"
    ev = hit.get("ev") if kind == "win" else hit.get("fwd_mean")
    return {"kind": kind, "id": str(hit.get("id")), "text": str(hit.get("text", "")), "ev": ev, "n": hit.get("fwd_n")}


def compact_rules(pool: Optional[Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
    """Confirmed rules in the small form the alert path needs, best forward evidence first."""
    out: List[Dict[str, Any]] = []
    for rec in ((pool or {}).get("confirmed") or {}).values():
        fwd = rec.get("fwd") or {}
        if not rec.get("atoms") or fwd.get("mean") is None:
            continue
        out.append({"id": rec["id"], "atoms": rec["atoms"], "text": rec.get("text", ""), "since_ts": rec.get("since_ts"),
                    "fwd_n": fwd.get("n"), "fwd_mean": fwd.get("mean")})
    out.sort(key=lambda x: (x["fwd_mean"], -(x["fwd_n"] or 0)))
    return out[:max(0, int(limit))]

def describe_atom(a: Sequence[Any]) -> str:
    name = str(a[0]).replace("ctx:", "").replace("vote:", "")
    if a[0] in _COHORT_LABEL:
        return f"{_COHORT_LABEL[a[0]]} = {a[2]}"
    if a[0].startswith("vote:"):
        return f"{name} vote {'on' if a[2] else 'off'}"
    if a[1] == "ge":
        return f"{name} >= {a[2]:.4g}"
    if a[1] == "lt":
        return f"{name} < {a[2]:.4g}"
    return f"{name} = {a[2]}"


def describe_rule(atoms: Sequence[Sequence[Any]]) -> str:
    return " and ".join(describe_atom(a) for a in atoms)


def rule_id(atoms: Sequence[Sequence[Any]]) -> str:
    key = "|".join(sorted(f"{a[0]}:{a[1]}:{a[2]}" for a in atoms))
    return hashlib.md5(key.encode()).hexdigest()[:12]


def build_atom_pool(train_feats: Sequence[Dict[str, Any]], min_cohort: int = 0, n_ctx: int = 0) -> List[Atom]:
    """Candidate conditions, thresholds taken from the older trades only.
    min_cohort: a cohort value (fam / key / pair) needs at least this many older trades to become an atom.
    n_ctx: older trades that carry the full stored context. Numeric fields need 70% coverage of THOSE, so
    context-free shadow rows do not push every numeric field below the coverage floor (0 = all trades)."""
    n = len(train_feats)
    denom = n_ctx if n_ctx > 0 else n
    numeric: Dict[str, List[float]] = {}
    cats: Dict[str, Set[Any]] = {}
    cohort_n: Dict[Tuple[str, Any], int] = {}
    for f in train_feats:
        for k, v in f.items():
            if k.startswith("ctx:"):
                numeric.setdefault(k, []).append(v)
            elif k.startswith("vote:"):
                cats.setdefault(k, set()).update((True, False))
            else:
                cats.setdefault(k, set()).add(v)
                if k in COHORT_FEATURES:
                    cohort_n[(k, v)] = cohort_n.get((k, v), 0) + 1
    atoms: List[Atom] = []
    for name, vals in sorted(numeric.items()):
        if len(vals) < 0.7 * denom or len(set(vals)) < 20:
            continue
        s = sorted(vals)
        for q in (0.2, 0.4, 0.6, 0.8):
            t = float(f"{s[int(q * (len(s) - 1))]:.4g}")
            atoms.append((name, "ge", t))
            atoms.append((name, "lt", t))
    for name, vs in sorted(cats.items()):
        for v in sorted(vs, key=str):
            if name in COHORT_FEATURES and cohort_n.get((name, v), 0) < min_cohort:
                continue
            atoms.append((name, "eq", v))
    return list(dict.fromkeys(atoms))


def _stats(items: List[Tuple[float, float]]) -> Dict[str, Any]:
    n = len(items)
    if n == 0:
        return {"n": 0, "mean": None, "se": None}
    vals = [p for _t, p in items]
    m = sum(vals) / n
    se, _nb = cluster_se(vals, [t for t, _p in items]) if n >= 2 else (float("inf"), 0)
    return {"n": n, "mean": m, "se": None if se == float("inf") else se}


def _ub(st: Dict[str, Any], z: float) -> Optional[float]:
    return None if st["mean"] is None or st["se"] is None else st["mean"] + z * st["se"]


def _z(alpha: float, m: int) -> float:
    return NormalDist().inv_cdf(1.0 - alpha / max(1, m))


def _forward(rec: Dict[str, Any], feats: List[Dict[str, Any]], ts: List[float], pnls: List[float]) -> Dict[str, Any]:
    items = [(ts[i], pnls[i]) for i in range(len(ts))
             if ts[i] >= rec["since_ts"] and all(_match(a, feats[i]) for a in rec["atoms"])]
    st = _stats(items)
    return {"n": st["n"], "mean": None if st["mean"] is None else round(st["mean"], 4),
            "se": None if st["se"] is None else round(st["se"], 4)}


def _deepen(cohort_atoms: Sequence[Atom], free_atoms: Sequence[Atom], hits: Any, tr_set: Set[int],
            pnls: Sequence[float], skip: Set[str], *, min_train: int, max_atoms: int,
            budget: int) -> Dict[str, Tuple[Atom, ...]]:
    """Beam search from each cohort (one alert family / alert / pair) to rules of up to max_atoms conditions,
    one condition per level. Children are chosen ONLY from the older trades: a child must keep at least
    min_train of them, drop at least 10% of its parent's trades, and have a lower upper confidence bound (mean + 1.645 SE) than its parent. The beam keeps
    the best budget // (max_atoms - 1) improving children per level, even while their bound is still above
    zero (a 3-condition edge often needs a weaker 2-condition step on the way); only children whose bound is
    below zero are RETURNED. The newer (holdout) trades are never looked at here, so they stay an untouched
    test. Every child returned will be tested on the holdout and the caller counts each one in the ledger."""
    def key(combo: Sequence[Atom]) -> Tuple[Atom, ...]:
        return tuple(sorted(combo, key=lambda a: (a[0], a[1], str(a[2]))))

    def ub(m: Set[int]) -> float:
        n = len(m)
        vals = [pnls[i] for i in m]
        mu = sum(vals) / n
        var = sum((v - mu) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
        return mu + 1.645 * math.sqrt(var / n)

    out: Dict[str, Tuple[Atom, ...]] = {}
    quota = max(1, budget // max(1, max_atoms - 1))
    level: List[Tuple[float, Tuple[Atom, ...], Set[int]]] = []
    for a in cohort_atoms:
        m = hits(a) & tr_set
        if len(m) >= 2 * min_train:
            level.append((ub(m), (a,), m))
    for _depth in range(2, max_atoms + 1):
        if not level or len(out) >= budget:
            break
        grown: Dict[str, Tuple[float, Tuple[Atom, ...], Set[int]]] = {}
        for base_ub, combo, m in level:
            used = {a[0] for a in combo}
            for a in free_atoms:
                if a[0] in used:
                    continue
                mc = m & hits(a)
                if len(mc) < min_train or len(mc) > 0.9 * len(m):      # a real refinement removes >= 10% of its parent
                    continue
                u = ub(mc)
                if u >= base_ub:
                    continue
                child = key(combo + (a,))
                cid = rule_id(child)
                if cid in skip or cid in out:
                    continue
                if cid not in grown or u < grown[cid][0]:
                    grown[cid] = (u, child, mc)
        take = sorted(grown.items(), key=lambda kv: (kv[1][0], kv[0]))[:quota]
        for cid, (u, child, _mc) in take:
            if u < 0 and len(out) < budget:
                out[cid] = child
        level = [v for _cid, v in take if len(v[2]) >= 2 * min_train]
    return out


def update_hypothesis_pool(prev: Optional[Dict[str, Any]], rows: Sequence[Row], now: int, *, cost_pct: float,
                           budget: int = 300, shadow_max: int = 10, forward_min_n: int = 30,
                           expire_days: int = 30, alpha: float = 0.05, min_train: int = 30,
                           min_hold: int = 15, max_cover: float = 0.5, min_rows: int = 200,
                           direction: str = "lose", deep_budget: int = 100,
                           max_atoms: int = 4) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    prev = prev if isinstance(prev, dict) else {}
    pool: Dict[str, Any] = {
        "tried_count": int(prev.get("tried_count") or 0),
        "tried": dict(prev.get("tried") or {}),
        "shadow": {k: dict(v) for k, v in (prev.get("shadow") or {}).items()},
        "confirmed": {k: dict(v) for k, v in (prev.get("confirmed") or {}).items()},
        "history": list(prev.get("history") or []),
    }
    events: List[Dict[str, Any]] = []
    usable = sorted((r for r in rows if r.get("entry_ts") is not None
                     and (isinstance(r.get("context"), dict) or isinstance(r.get("votes"), dict))),
                    key=lambda r: r["entry_ts"])
    summary: Dict[str, Any] = {"status": "insufficient_data", "n_rows": len(usable), "n_tested": 0,
                               "n_context_rows": sum(1 for r in usable if r.get("context")),
                               "tried_total": pool["tried_count"], "z_required": None, "survivors": 0,
                               "events": events}
    feats = [row_features(r, setup=True) for r in usable]
    ts = [float(r["entry_ts"]) for r in usable]
    sign = -1.0 if direction == "win" else 1.0          # "win" searches the negated series: a rule that loses on it wins
    pnls = [sign * row_net_pnl_pct(r, cost_pct) for r in usable]

    def log(rec: Dict[str, Any], status: str) -> None:
        ev = {"ts": int(now), "id": rec["id"], "text": rec["text"], "status": status, "forward": rec.get("fwd")}
        events.append(ev)
        pool["history"].insert(0, ev)

    # ── forward stage: shadow rules and confirmed rules ──
    active = len(pool["shadow"])
    z_f = _z(alpha, max(1, active))
    for rid in list(pool["shadow"]):
        rec = pool["shadow"][rid]
        rec["fwd"] = _forward(rec, feats, ts, pnls)
        st = {"n": rec["fwd"]["n"], "mean": rec["fwd"]["mean"], "se": rec["fwd"]["se"]}
        if st["n"] >= forward_min_n and st["mean"] is not None:
            ub = _ub(st, z_f)
            if ub is not None and ub < 0:
                rec["status"] = "confirmed"
                rec["confirmed_at"] = int(now)
                pool["confirmed"][rid] = pool["shadow"].pop(rid)
                log(rec, "confirmed")
                continue
            if st["mean"] >= 0:
                pool["shadow"].pop(rid)
                log(rec, "rejected")
                continue
        if now - rec["since_ts"] > expire_days * 86400:
            pool["shadow"].pop(rid)
            log(rec, "expired")
    for rid in list(pool["confirmed"]):
        rec = pool["confirmed"][rid]
        rec["fwd"] = _forward(rec, feats, ts, pnls)
        if rec["fwd"]["n"] >= forward_min_n and rec["fwd"]["mean"] is not None and rec["fwd"]["mean"] >= 0:
            pool["confirmed"].pop(rid)
            log(rec, "revoked")

    # ── proposal + selection ──
    free = shadow_max - len(pool["shadow"])
    if len(usable) >= min_rows and budget > 0 and free > 0:
        cut = int(len(usable) * 0.6)
        split_ts = ts[cut]
        embargo = 14 * _CANDLE_SEC
        tr_idx = [i for i in range(cut) if ts[i] <= split_ts - embargo]
        ho_idx = list(range(cut, len(usable)))
        atoms = build_atom_pool([feats[i] for i in tr_idx], min_cohort=min_train,
                                n_ctx=sum(1 for i in tr_idx if usable[i].get("context")))
        # Cohort atoms (alert family / alert / pair) anchor most candidates, so a rule describes ONE kind of alert
        # instead of pooling unrelated alerts; the remaining candidates keep the old unanchored search.
        free_atoms = [a for a in atoms if a[0] not in COHORT_FEATURES]
        by_cohort: Dict[str, List[Atom]] = {}
        for a in atoms:
            if a[0] in COHORT_FEATURES:
                by_cohort.setdefault(a[0], []).append(a)
        cohort_names = sorted(by_cohort)
        if len(atoms) >= 4 and len(tr_idx) >= min_train and len(ho_idx) >= min_hold:
            rng = random.Random(int(now // 86400))
            cands: Dict[str, Tuple[Atom, ...]] = {}
            tries = 0
            while len(cands) < budget and tries < budget * 20:
                tries += 1
                if cohort_names and free_atoms and rng.random() < 0.7:
                    anchor = rng.choice(by_cohort[rng.choice(cohort_names)])
                    pick = [anchor, rng.choice(free_atoms)]
                    k = 2
                else:
                    base = free_atoms if len(free_atoms) >= 2 else atoms
                    k = 1 if rng.random() < 0.3 else 2
                    pick = rng.sample(base, k)
                if k == 2 and pick[0][0] == pick[1][0]:
                    continue
                combo = tuple(sorted(pick, key=lambda a: (a[0], a[1], str(a[2]))))
                cid = rule_id(combo)
                if cid in pool["tried"] or cid in cands:
                    continue
                cands[cid] = combo
            cache: Dict[Atom, Set[int]] = {}

            def hits(a: Atom) -> Set[int]:
                if a not in cache:
                    cache[a] = {i for i in range(len(usable)) if _match(a, feats[i])}
                return cache[a]

            tr_set, ho_set = set(tr_idx), set(ho_idx)
            n_deep = 0
            if deep_budget > 0 and max_atoms > 2:
                deep = _deepen([a for a in atoms if a[0] in COHORT_FEATURES], free_atoms, hits, tr_set, pnls,
                               set(pool["tried"]) | set(cands),
                               min_train=min_train, max_atoms=max_atoms, budget=deep_budget)
                n_deep = len(deep)
                cands.update(deep)
            pool["tried_count"] += len(cands)
            for cid in cands:
                pool["tried"][cid] = int(now)
            if len(pool["tried"]) > _MAX_TRIED:
                keep = sorted(pool["tried"].items(), key=lambda kv: -kv[1])[:_MAX_TRIED]
                pool["tried"] = dict(keep)
            z_adj = _z(alpha, pool["tried_count"])
            summary.update({"status": "ran", "n_tested": len(cands), "n_deep": n_deep,
                            "tried_total": pool["tried_count"], "z_required": round(z_adj, 2)})
            passed: List[Tuple[float, str, Tuple[Atom, ...], Dict[str, Any], Dict[str, Any], Set[int]]] = []
            for cid, combo in cands.items():
                m = set.intersection(*(hits(a) for a in combo))
                mt, mh = m & tr_set, m & ho_set
                if len(mt) < min_train or len(mh) < min_hold or len(m) > max_cover * len(usable):
                    continue
                st_t = _stats([(ts[i], pnls[i]) for i in sorted(mt)])
                ub_t = _ub(st_t, 1.645)
                if st_t["mean"] is None or st_t["mean"] >= 0 or ub_t is None or ub_t >= 0:
                    continue
                st_h = _stats([(ts[i], pnls[i]) for i in sorted(mh)])
                ub_h = _ub(st_h, z_adj)
                if ub_h is None or ub_h >= 0:
                    continue
                passed.append((ub_h, cid, combo, st_t, st_h, m))
            passed.sort(key=lambda x: x[0])
            summary["survivors"] = len(passed)
            chosen: List[Set[int]] = []
            for ub_h, cid, combo, st_t, st_h, m in passed:
                if len(chosen) >= free:
                    break
                if any(len(m & c) / max(1, len(m | c)) > 0.6 for c in chosen):
                    continue            # nearly the same trades as a rule already taken
                chosen.append(m)
                rec = {
                    "id": cid, "atoms": [list(a) for a in combo], "text": describe_rule(combo),
                    "since_ts": int(now), "status": "shadow",
                    "train": {"n": st_t["n"], "mean": round(st_t["mean"], 4)},
                    "hold": {"n": st_h["n"], "mean": round(st_h["mean"], 4), "ub": round(ub_h, 4)},
                    "fwd": {"n": 0, "mean": None, "se": None},
                }
                pool["shadow"][cid] = rec
                log(rec, "shadow")
    pool["history"] = pool["history"][:50]
    pool["updated_at"] = int(now)
    summary["shadow"] = [{k: v[k] for k in ("id", "text", "since_ts", "hold", "fwd")} for v in pool["shadow"].values()]
    summary["confirmed"] = [{k: v[k] for k in ("id", "text", "fwd", "hold")} for v in pool["confirmed"].values()]
    summary["tried_total"] = pool["tried_count"]
    return pool, summary