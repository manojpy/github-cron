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

A confirmed rule is only a candidate for human review. Nothing is blocked."""
from __future__ import annotations

import hashlib
import math
import random
from statistics import NormalDist
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from plan_replay import cluster_se
from threshold_stats import row_net_pnl_pct
from threshold_models import _PRICE_LEVEL_CTX

Row = Dict[str, Any]
Atom = Tuple[str, str, Any]          # (feature, op, value); op: ge | lt | eq
_CANDLE_SEC = 900
_MAX_TRIED = 5000


def row_features(r: Row) -> Dict[str, Any]:
    """Everything knowable when the alert fired, flattened to one dict."""
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


def describe_atom(a: Sequence[Any]) -> str:
    name = str(a[0]).replace("ctx:", "").replace("vote:", "")
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


def build_atom_pool(train_feats: Sequence[Dict[str, Any]]) -> List[Atom]:
    """Candidate conditions, thresholds taken from the older trades only."""
    n = len(train_feats)
    numeric: Dict[str, List[float]] = {}
    cats: Dict[str, Set[Any]] = {}
    for f in train_feats:
        for k, v in f.items():
            if k.startswith("ctx:"):
                numeric.setdefault(k, []).append(v)
            elif k.startswith("vote:"):
                cats.setdefault(k, set()).update((True, False))
            else:
                cats.setdefault(k, set()).add(v)
    atoms: List[Atom] = []
    for name, vals in sorted(numeric.items()):
        if len(vals) < 0.7 * n or len(set(vals)) < 20:
            continue
        s = sorted(vals)
        for q in (0.2, 0.4, 0.6, 0.8):
            t = float(f"{s[int(q * (len(s) - 1))]:.4g}")
            atoms.append((name, "ge", t))
            atoms.append((name, "lt", t))
    for name, vs in sorted(cats.items()):
        for v in sorted(vs, key=str):
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


def update_hypothesis_pool(prev: Optional[Dict[str, Any]], rows: Sequence[Row], now: int, *, cost_pct: float,
                           budget: int = 300, shadow_max: int = 10, forward_min_n: int = 30,
                           expire_days: int = 30, alpha: float = 0.05, min_train: int = 30,
                           min_hold: int = 15, max_cover: float = 0.5, min_rows: int = 200) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    prev = prev if isinstance(prev, dict) else {}
    pool: Dict[str, Any] = {
        "tried_count": int(prev.get("tried_count") or 0),
        "tried": dict(prev.get("tried") or {}),
        "shadow": {k: dict(v) for k, v in (prev.get("shadow") or {}).items()},
        "confirmed": {k: dict(v) for k, v in (prev.get("confirmed") or {}).items()},
        "history": list(prev.get("history") or []),
    }
    events: List[Dict[str, Any]] = []
    usable = sorted((r for r in rows if r.get("entry_ts") is not None and isinstance(r.get("context"), dict)),
                    key=lambda r: r["entry_ts"])
    summary: Dict[str, Any] = {"status": "insufficient_data", "n_rows": len(usable), "n_tested": 0,
                               "tried_total": pool["tried_count"], "z_required": None, "survivors": 0,
                               "events": events}
    feats = [row_features(r) for r in usable]
    ts = [float(r["entry_ts"]) for r in usable]
    pnls = [row_net_pnl_pct(r, cost_pct) for r in usable]

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
        atoms = build_atom_pool([feats[i] for i in tr_idx])
        if len(atoms) >= 4 and len(tr_idx) >= min_train and len(ho_idx) >= min_hold:
            rng = random.Random(int(now // 86400))
            cands: Dict[str, Tuple[Atom, ...]] = {}
            tries = 0
            while len(cands) < budget and tries < budget * 20:
                tries += 1
                k = 1 if rng.random() < 0.3 else 2
                pick = rng.sample(atoms, k)
                if k == 2 and pick[0][0] == pick[1][0]:
                    continue
                combo = tuple(sorted(pick, key=lambda a: (a[0], a[1], str(a[2]))))
                cid = rule_id(combo)
                if cid in pool["tried"] or cid in cands:
                    continue
                cands[cid] = combo
            pool["tried_count"] += len(cands)
            for cid in cands:
                pool["tried"][cid] = int(now)
            if len(pool["tried"]) > _MAX_TRIED:
                keep = sorted(pool["tried"].items(), key=lambda kv: -kv[1])[:_MAX_TRIED]
                pool["tried"] = dict(keep)
            z_adj = _z(alpha, pool["tried_count"])
            summary.update({"status": "ran", "n_tested": len(cands), "tried_total": pool["tried_count"],
                            "z_required": round(z_adj, 2)})
            cache: Dict[Atom, Set[int]] = {}

            def hits(a: Atom) -> Set[int]:
                if a not in cache:
                    cache[a] = {i for i in range(len(usable)) if _match(a, feats[i])}
                return cache[a]

            tr_set, ho_set = set(tr_idx), set(ho_idx)
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