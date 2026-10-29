"""autopsy — per-trade strengths/weaknesses, shadow rejection audit and the Trust Meter.

Pure functions (no Redis, no files, no network).  Report-only: nothing here
changes a verdict, a gate or an alert rule.

What it answers, for the Recorded (LIVE) and Shadowed (SHADOW) populations
separately:

  1. TRADE AUTOPSY   For every resolved trade: what did the Brain know BEFORE
                     it (point-in-time, with an embargo so unresolved
                     overlapping trades are never used), which conditions
                     historically helped or hurt this kind of trade, and why
                     this one won or lost.
  2. EDGE MAP        Which conditions carry a real lift, tagged with an
                     evidence level (L1 exploratory, L2 = survives a
                     false-discovery correction AND keeps its sign in both
                     chronological halves).  Never promoted above L2 here:
                     L3+ only comes from the playbook's holdout and control.
  3. REJECTION AUDIT For every shadow rejection reason: was rejecting right?
  4. TRUST METER     How far can the TAKE verdict be trusted: precision with a
                     Wilson interval, coverage, conviction calibration,
                     ranking quality (AUC), net EV and stability over time.

Success means the trade made money under the bracket model after costs
(row_net_pnl_pct > 0), not the MFE-based `win` flag.

Shadow rows written before ENABLE_SINGLE_ACTIVE_SHADOW often repeat one market
move several times; independent_rows() collapses those before any statistic.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Callable, DefaultDict, Dict, List, Mapping, Optional, Sequence, Tuple

from hypothesis_gen import row_features
from plan_replay import _DIAG_LABELS, cluster_se, diagnose_row
from threshold_stats import row_net_pnl_pct, two_proportion_p_value, wilson_ci

Row = Dict[str, Any]

_CANDLE_SEC = 900
_AMBIGUOUS = "ambiguous_same_candle"
TAKE, WATCH, AVOID = "TAKE", "WATCH", "AVOID"
CONV_BANDS: Tuple[Tuple[float, float], ...] = ((0.0, 50.0), (50.0, 60.0), (60.0, 70.0), (70.0, 80.0), (80.0, 101.0))


# ── small helpers ─────────────────────────────────────────────────────────
def _pop_of(r: Mapping[str, Any]) -> str:
    return "SHADOW" if (r.get("source") == "SHADOW" or r.get("shadow")) else "LIVE"


def _row_id(r: Mapping[str, Any]) -> str:
    return str(r.get("_stream_id") or f"{r.get('pair')}:{r.get('alert_key')}:{r.get('entry_ts')}")


def _shrunk(wins: float, n: float, prior: float, k: float) -> float:
    return (wins + k * prior) / (n + k)


def _auc(scores: Sequence[float], labels: Sequence[bool]) -> Optional[float]:
    """Rank-based AUC (Mann-Whitney, ties averaged); None if one class is missing."""
    n_pos = sum(1 for y in labels if y)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for t in range(i, j + 1):
            ranks[order[t]] = avg
        i = j + 1
    rank_sum = sum(ranks[i] for i, y in enumerate(labels) if y)
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _bh_pass(pvals: Sequence[float], q: float = 0.10) -> List[bool]:
    """Benjamini-Hochberg false-discovery control across every condition tested."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    kmax = 0
    for rank, i in enumerate(order, 1):
        if pvals[i] <= q * rank / m:
            kmax = rank
    out = [False] * m
    for rank, i in enumerate(order, 1):
        if rank <= kmax:
            out[i] = True
    return out


def _outcome(r: Mapping[str, Any], cost_pct: float) -> Optional[Tuple[float, bool]]:
    """(net P&L %, success) or None when the row cannot be graded."""
    try:
        pnl = float(row_net_pnl_pct(dict(r), cost_pct))
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(pnl):
        return None
    return pnl, pnl > 0.0


def _hold_candles(r: Mapping[str, Any], horizon: int, fill_delay: int) -> int:
    """Candles a trade stayed open: up to its stop/target candle when known, else the horizon."""
    hits = [c for c in (r.get("tp_candle"), r.get("sl_candle")) if isinstance(c, (int, float))]
    if hits:
        return int(min(min(hits) + 1, horizon)) + fill_delay
    return horizon + fill_delay


def independent_rows(rows: Sequence[Row], *, horizon: int = 12, fill_delay: int = 1) -> List[Row]:
    """One row per open trade: for each (population, pair) keep a row only when the
    previously kept one has already closed.  This is the rule Recorded trades follow
    live; applying it to old shadow rows removes repeated counts of one move."""
    by: DefaultDict[Tuple[str, str], List[Row]] = defaultdict(list)
    for r in rows:
        if r.get("entry_ts") is None or not r.get("pair"):
            continue
        by[(_pop_of(r), str(r["pair"]))].append(r)
    kept: List[Row] = []
    for grp in by.values():
        grp.sort(key=lambda x: int(x["entry_ts"]))
        free_at = -1
        for r in grp:
            ts = int(r["entry_ts"])
            if ts < free_at:
                continue
            kept.append(r)
            free_at = ts + _hold_candles(r, horizon, fill_delay) * _CANDLE_SEC
    kept.sort(key=lambda x: int(x["entry_ts"]))
    return kept


# ── conditions (atoms) ────────────────────────────────────────────────────
def _edges(feats: Sequence[Mapping[str, Any]], min_cover: float = 0.7, min_distinct: int = 8) -> Dict[str, Tuple[float, float]]:
    """Tercile edges per numeric feature.  Built from feature values only (no outcomes),
    so it cannot leak a result into the buckets."""
    vals: DefaultDict[str, List[float]] = defaultdict(list)
    for f in feats:
        for k, v in f.items():
            if k.startswith("ctx:"):
                vals[k[4:]].append(float(v))
    n = len(feats)
    out: Dict[str, Tuple[float, float]] = {}
    for name, v in vals.items():
        if len(v) < min_cover * n or len(set(v)) < min_distinct:
            continue
        s = sorted(v)
        lo = s[int((len(s) - 1) / 3)]
        hi = s[int(2 * (len(s) - 1) / 3)]
        if lo < hi:
            out[name] = (lo, hi)
    return out


def _atoms(f: Mapping[str, Any], edges: Mapping[str, Tuple[float, float]]) -> List[str]:
    out: List[str] = []
    for k, v in f.items():
        if k.startswith("ctx:"):
            e = edges.get(k[4:])
            if e is None:
                continue
            out.append(f"{k[4:]} {'low' if v < e[0] else 'high' if v >= e[1] else 'mid'}")
        elif k.startswith("vote:"):
            out.append(f"{k[5:]} vote {'on' if v else 'off'}")
        elif k == "session":
            out.append(f"session {v}")
    return out


def _atom_text(atom: str, edges: Mapping[str, Tuple[float, float]]) -> str:
    name, _, bucket = atom.rpartition(" ")
    e = edges.get(name)
    if e is None or bucket not in ("low", "mid", "high"):
        return atom
    if bucket == "low":
        return f"{name} low (<{e[0]:.3g})"
    if bucket == "high":
        return f"{name} high (>={e[1]:.3g})"
    return f"{name} mid ({e[0]:.3g}-{e[1]:.3g})"


def _prepare(rows: Sequence[Row], cost_pct: float) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for r in rows:
        if r.get("entry_ts") is None or r.get("win") is None:
            continue
        out = _outcome(r, cost_pct)
        if out is None:
            continue
        items.append({"r": r, "ts": int(r["entry_ts"]), "f": row_features(r), "pnl": out[0], "y": out[1]})
    items.sort(key=lambda it: it["ts"])
    return items


# ── 1. per-trade autopsy (point in time) ──────────────────────────────────
def build_autopsies(
    rows: Sequence[Row], *, cost_pct: float, sl_pct: float, tp_pct: float,
    group_fn: Optional[Callable[[str], str]] = None, min_n: int = 12, shrink_k: float = 10.0,
    min_lift: float = 0.03, embargo_candles: int = 14, top: int = 3, keep_last: int = 40,
) -> Dict[str, Any]:
    """Autopsy of every resolved trade using ONLY trades that had already finished
    (entry + embargo <= this entry).  Conditions are judged inside the trade's own
    alert group when that group has enough history, else across all alerts (partial
    pooling); each lift is shrunk toward its baseline."""
    gf = group_fn or (lambda ak: ak)
    items = _prepare(rows, cost_pct)
    n_items = len(items)
    if n_items == 0:
        return {"n": 0, "autopsies": [], "edges": {}}
    edges = _edges([it["f"] for it in items])
    emb = embargo_candles * _CANDLE_SEC
    for it in items:
        it["atoms"] = _atoms(it["f"], edges)
        it["g"] = f"{gf(str(it['r'].get('alert_key', '?')))}|{it['r'].get('direction', '?')}"

    grp: Dict[str, List[int]] = {}
    atom_stats: Dict[Tuple[str, str], List[int]] = {}
    g_all = [0, 0]
    j = 0
    autopsies: List[Dict[str, Any]] = []

    for i, it in enumerate(items):
        while j < i and items[j]["ts"] + emb <= it["ts"]:
            o = items[j]
            y = 1 if o["y"] else 0
            g_all[0] += 1
            g_all[1] += y
            s = grp.setdefault(o["g"], [0, 0])
            s[0] += 1
            s[1] += y
            for a in o["atoms"]:
                for scope in (o["g"], "*"):
                    t = atom_stats.setdefault((scope, a), [0, 0])
                    t[0] += 1
                    t[1] += y
            j += 1
        p_glob = (g_all[1] + 1.0) / (g_all[0] + 2.0)
        gn, gw = grp.get(it["g"], [0, 0])
        p0 = _shrunk(gw, gn, p_glob, shrink_k)
        findings: List[Tuple[float, str, int, str]] = []
        for a in it["atoms"]:
            st = atom_stats.get((it["g"], a))
            if st is not None and st[0] >= min_n:
                findings.append((_shrunk(st[1], st[0], p0, shrink_k) - p0, a, st[0], "alert"))
                continue
            st = atom_stats.get(("*", a))
            if st is not None and st[0] >= min_n:
                findings.append((_shrunk(st[1], st[0], p_glob, shrink_k) - p_glob, a, st[0], "all"))
        pos = sorted((x for x in findings if x[0] >= min_lift), key=lambda x: -x[0])[:top]
        neg = sorted((x for x in findings if x[0] <= -min_lift), key=lambda x: x[0])[:top]
        r = it["r"]
        won = bool(it["y"])
        mode = "WIN" if won else diagnose_row(r, sl_pct, tp_pct)
        if mode == "WIN":
            mode_txt = "win"
        else:
            mode_txt = _DIAG_LABELS.get(mode, mode)
        seen = (r.get("context") or {}).get("advisor_seen") if isinstance(r.get("context"), dict) else None
        seen_out = None
        if isinstance(seen, dict) and seen.get("verdict"):
            seen_out = {"verdict": seen.get("verdict"), "conviction": seen.get("conviction")}
        ambiguous = r.get("outcome_reason") == _AMBIGUOUS
        autopsies.append({
            "id": _row_id(r), "pair": r.get("pair"), "alert_key": r.get("alert_key"),
            "dir": r.get("direction"), "pop": _pop_of(r), "ts": it["ts"],
            "result": "WIN" if won else "LOSS", "pnl": round(it["pnl"], 3), "mode": mode,
            "ambiguous": ambiguous, "p_prior": round(p0, 3), "n_prior": gn,
            "helped": [[_atom_text(a, edges), round(l, 3), n, sc] for l, a, n, sc in pos],
            "hurt": [[_atom_text(a, edges), round(l, 3), n, sc] for l, a, n, sc in neg],
            "seen": seen_out, "mode_text": mode_txt,
        })
    return {"n": n_items, "autopsies": autopsies[-keep_last:], "edges": {k: [round(v[0], 4), round(v[1], 4)] for k, v in edges.items()}}


# ── 2. edge map (what carries a real lift) ────────────────────────────────
def edge_map(
    rows: Sequence[Row], *, cost_pct: float, min_n: int = 20, shrink_k: float = 10.0,
    top: int = 4, fdr_q: float = 0.10,
) -> Dict[str, Any]:
    items = _prepare(rows, cost_pct)
    n = len(items)
    if n < 2 * min_n:
        return {"n": n, "status": "COLLECTING", "helps": [], "hurts": []}
    edges = _edges([it["f"] for it in items])
    half = n // 2
    tot: DefaultDict[str, List[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    h1: DefaultDict[str, List[float]] = defaultdict(lambda: [0.0, 0.0])
    h2: DefaultDict[str, List[float]] = defaultdict(lambda: [0.0, 0.0])
    W = sum(1 for it in items if it["y"])
    W1 = sum(1 for it in items[:half] if it["y"])
    W2 = W - W1
    for idx, it in enumerate(items):
        y = 1.0 if it["y"] else 0.0
        for a in _atoms(it["f"], edges):
            t = tot[a]
            t[0] += 1
            t[1] += y
            t[2] += it["pnl"]
            h = h1 if idx < half else h2
            h[a][0] += 1
            h[a][1] += y
    base = W / n
    cands: List[Dict[str, Any]] = []
    pvals: List[float] = []
    for a, (na, wa, sp) in tot.items():
        nc, wc = n - na, W - wa
        if na < min_n or nc < min_n:
            continue
        p = two_proportion_p_value(int(wa), int(na), int(wc), int(nc))
        lift = _shrunk(wa, na, base, shrink_k) - base
        sgn = 1 if lift > 0 else -1
        ok_halves = True
        for hh, nh, wh in ((h1[a], half, W1), (h2[a], n - half, W2)):
            ncomp = nh - hh[0]
            if hh[0] < max(5, min_n // 2) or ncomp < 5:
                ok_halves = False
                break
            d = hh[1] / hh[0] - (wh - hh[1]) / ncomp
            if d * sgn <= 0:
                ok_halves = False
                break
        cands.append({"atom": a, "n": int(na), "rate": round(wa / na, 3), "base": round(base, 3),
                      "lift": round(lift, 3), "mean_net": round(sp / na, 3), "p": round(p, 4),
                      "consistent": ok_halves})
        pvals.append(p)
    if not cands:
        return {"n": n, "status": "COLLECTING", "helps": [], "hurts": []}
    for c, ok in zip(cands, _bh_pass(pvals, fdr_q)):
        c["level"] = 2 if (ok and c["consistent"]) else 1
        c["text"] = _atom_text(c["atom"], edges)
    helps = sorted((c for c in cands if c["lift"] > 0), key=lambda c: (-c["level"], -c["lift"]))[:top]
    hurts = sorted((c for c in cands if c["lift"] < 0), key=lambda c: (-c["level"], c["lift"]))[:top]
    return {"n": n, "status": "OK", "base_rate": round(base, 3), "tested": len(cands), "helps": helps, "hurts": hurts}


# ── 3. shadow rejection audit ─────────────────────────────────────────────
def _reason_of(r: Mapping[str, Any]) -> str:
    reason = r.get("rejection_reason")
    if not reason and isinstance(r.get("context"), dict):
        reason = r["context"].get("rejection_reason")
    return str(reason or "unknown").split(":")[0].strip()[:48] or "unknown"


def rejection_audit(shadow_rows: Sequence[Row], live_rows: Sequence[Row], *, cost_pct: float, min_n: int = 20, z: float = 1.645) -> Dict[str, Any]:
    """Was each rejection right?  Compared with the Recorded trades.  Rejections are not a
    random sample of all signals, so COSTLY means 'look at this gate', never 'remove it'.
    calibration_gate rejections are excluded: they would grade the gate by its own choice."""
    live = [o for o in (_outcome(r, cost_pct) for r in live_rows) if o is not None]
    live_rate = (sum(1 for o in live if o[1]) / len(live)) if live else None
    live_net = (sum(o[0] for o in live) / len(live)) if live else None
    groups: DefaultDict[str, List[Tuple[float, float, bool]]] = defaultdict(list)
    excluded = 0
    for r in shadow_rows:
        o = _outcome(r, cost_pct)
        if o is None or r.get("entry_ts") is None:
            continue
        reason = _reason_of(r)
        if reason == "calibration_gate":
            excluded += 1
            continue
        groups[reason].append((float(r["entry_ts"]), o[0], o[1]))
    out_rows: List[Dict[str, Any]] = []
    for reason, g in groups.items():
        n = len(g)
        wins = sum(1 for _, _, y in g if y)
        lo, hi, p = wilson_ci(wins, n)
        vals = [x[1] for x in g]
        mean = sum(vals) / n
        se, _nb = cluster_se(vals, [x[0] for x in g]) if n >= 2 else (float("inf"), 0)
        if n < min_n or se == float("inf"):
            status = "COLLECTING"
        elif mean + z * se < 0:
            status = "JUSTIFIED"
        elif mean - z * se > 0:
            status = "COSTLY"
        else:
            status = "UNCLEAR"
        out_rows.append({"reason": reason, "n": n, "success": round(p, 3), "wilson": [round(lo, 3), round(hi, 3)],
                         "mean_net": round(mean, 3), "se": None if se == float("inf") else round(se, 3), "status": status})
    out_rows.sort(key=lambda x: -x["n"])
    return {"live_success": None if live_rate is None else round(live_rate, 3),
            "live_mean_net": None if live_net is None else round(live_net, 3),
            "excluded_calibration": excluded, "reasons": out_rows}


# ── 4. trust meter ────────────────────────────────────────────────────────
def _seen(r: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    ctx = r.get("context")
    s = ctx.get("advisor_seen") if isinstance(ctx, dict) else None
    return s if isinstance(s, dict) and s.get("verdict") in (TAKE, WATCH, AVOID) else None


def _pnl_lb(pairs: Sequence[Tuple[float, float]], z: float = 1.645) -> Tuple[Optional[float], Optional[float]]:
    """(mean net, one-sided lower bound) with a time-clustered standard error."""
    if len(pairs) < 2:
        return None, None
    vals = [p for _, p in pairs]
    mean = sum(vals) / len(vals)
    se, _nb = cluster_se(vals, [t for t, _ in pairs])
    if se == float("inf"):
        return mean, None
    return mean, mean - z * se


def trust_meter(
    live_rows: Sequence[Row], *, cost_pct: float, sl_pct: float, tp_pct: float,
    target: float = 0.80, min_take_n: int = 100, windows: int = 4, min_window_n: int = 10,
) -> Dict[str, Any]:
    """How far the TAKE verdict can be trusted, measured only on verdicts that were stored
    BEFORE the outcome was known (context['advisor_seen'])."""
    be = (sl_pct + cost_pct) / (sl_pct + tp_pct) if (sl_pct + tp_pct) > 0 else 0.5
    rows = []
    for r in live_rows:
        s = _seen(r)
        o = _outcome(r, cost_pct)
        if s is None or o is None or r.get("entry_ts") is None:
            continue
        rows.append((int(r["entry_ts"]), s, o[0], o[1]))
    rows.sort(key=lambda x: x[0])
    out: Dict[str, Any] = {"n_live": len(live_rows), "n_with_verdict": len(rows), "breakeven_precision": round(be, 3),
                           "target": target, "verdicts": {}, "bands": [], "auc": None}
    if not rows:
        out.update({"level": 0, "status": "COLLECTING", "note": "no resolved trade carries a stored verdict yet"})
        return out
    for v in (TAKE, WATCH, AVOID):
        g = [x for x in rows if x[1]["verdict"] == v]
        if not g:
            continue
        n = len(g)
        w = sum(1 for x in g if x[3])
        lo, hi, p = wilson_ci(w, n)
        mean, nlb = _pnl_lb([(x[0], x[2]) for x in g])
        out["verdicts"][v] = {"n": n, "coverage": round(n / len(rows), 3), "precision": round(p, 3),
                              "wilson": [round(lo, 3), round(hi, 3)],
                              "mean_net": None if mean is None else round(mean, 3),
                              "net_lb": None if nlb is None else round(nlb, 3)}
    conv = [(x, float(x[1]["conviction"])) for x in rows if isinstance(x[1].get("conviction"), (int, float))]
    for lo_b, hi_b in CONV_BANDS:
        g2 = [x for x, c in conv if lo_b <= c < hi_b]
        if not g2:
            continue
        w2 = sum(1 for x in g2 if x[3])
        l2, h2, p2 = wilson_ci(w2, len(g2))
        mc = sum(float(x[1]["conviction"]) for x in g2) / len(g2)
        out["bands"].append({"band": f"{int(lo_b)}-{int(min(hi_b, 100))}", "n": len(g2), "mean_conviction": round(mc, 1),
                             "precision": round(p2, 3), "wilson": [round(l2, 3), round(h2, 3)]})
    a = _auc([c for _, c in conv], [x[3] for x, _ in conv]) if conv else None
    out["auc"] = None if a is None else round(a, 3)
    take = [x for x in rows if x[1]["verdict"] == TAKE]
    nt = len(take)
    tv = out["verdicts"].get(TAKE)
    wins_ok: List[Dict[str, Any]] = []
    if nt >= windows * min_window_n:
        size = nt // windows
        for wi in range(windows):
            seg = take[wi * size:(wi + 1) * size if wi < windows - 1 else nt]
            wp = sum(1 for x in seg if x[3]) / len(seg)
            wm = sum(x[2] for x in seg) / len(seg)
            wins_ok.append({"n": len(seg), "precision": round(wp, 3), "mean_net": round(wm, 3),
                            "ok": bool(len(seg) >= min_window_n and wp > be and wm > 0)})
    out["windows"] = wins_ok
    if tv is None or nt < 30:
        level, status = 0, "COLLECTING"
    else:
        lb = tv["wilson"][0]
        edge_ok = lb > be and tv["net_lb"] is not None and tv["net_lb"] > 0
        if nt < min_take_n or not edge_ok:
            level, status = 1, "EXPLORATORY"
        elif wins_ok and all(w["ok"] for w in wins_ok):
            level, status = 3, "FORWARD_VALIDATED"
        else:
            level, status = 2, "SUPPORTED"
        out["meets_target"] = bool(lb >= target)
    out.update({"level": level, "status": status})
    return out


# ── data quality ──────────────────────────────────────────────────────────
def data_quality(live_rows: Sequence[Row], shadow_rows: Sequence[Row], shadow_independent: int) -> Dict[str, Any]:
    def share(rs: Sequence[Row], pred: Callable[[Row], bool]) -> Optional[float]:
        return round(sum(1 for r in rs if pred(r)) / len(rs), 3) if rs else None
    return {
        "live_rows": len(live_rows), "shadow_rows": len(shadow_rows), "shadow_independent": shadow_independent,
        "ambiguous_live": share(live_rows, lambda r: r.get("outcome_reason") == _AMBIGUOUS),
        "ambiguous_shadow": share(shadow_rows, lambda r: r.get("outcome_reason") == _AMBIGUOUS),
        "live_with_verdict": sum(1 for r in live_rows if _seen(r) is not None),
        "live_with_path": sum(1 for r in live_rows if r.get("path_fav")),
    }


# ── orchestration + text ──────────────────────────────────────────────────
def build_intelligence(
    rows: Sequence[Row], *, cost_pct: float, sl_pct: float, tp_pct: float,
    group_fn: Optional[Callable[[str], str]] = None, params: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    p = dict(params or {})
    min_n = int(p.get("min_n", 12))
    live_all = [r for r in rows if _pop_of(r) == "LIVE"]
    shadow_all = [r for r in rows if _pop_of(r) == "SHADOW"]
    live = live_all                         # Recorded trades already follow one-open-trade-per-pair
    shadow = independent_rows(shadow_all)   # old shadow rows may repeat one move several times
    blob: Dict[str, Any] = {"quality": data_quality(live_all, shadow_all, len(shadow)), "pops": {}}
    for name, pop in (("LIVE", live), ("SHADOW", shadow)):
        blob["pops"][name] = {
            "autopsy": build_autopsies(pop, cost_pct=cost_pct, sl_pct=sl_pct, tp_pct=tp_pct, group_fn=group_fn,
                                       min_n=min_n, shrink_k=float(p.get("shrink_k", 10.0)),
                                       keep_last=int(p.get("keep_last", 40))),
            "edges": edge_map(pop, cost_pct=cost_pct, min_n=int(p.get("edge_min_n", 20)), shrink_k=float(p.get("shrink_k", 10.0))),
        }
    blob["exhaustion"] = exhaustion_study(
        list(live) + list(shadow), cost_pct=cost_pct,
        early_bars=int(p.get("exh_early_bars", 2)), min_n=int(p.get("exh_min_n", 30)))
    blob["rejections"] = rejection_audit(shadow, live, cost_pct=cost_pct, min_n=int(p.get("reject_min_n", 20)))
    blob["trust"] = trust_meter(live, cost_pct=cost_pct, sl_pct=sl_pct, tp_pct=tp_pct,
                                target=float(p.get("target", 0.80)), min_take_n=int(p.get("min_take_n", 100)),
                                windows=int(p.get("windows", 4)))
    return blob

# ── 5. late-trend exhaustion study ────────────────────────────────────────
def exhaustion_study(rows: Sequence[Row], *, cost_pct: float, early_bars: int = 2, min_n: int = 30,
                     veto_scores: Sequence[int] = (3, 4, 5), min_cats: int = 2) -> Dict[str, Any]:
    """Does the exhaustion score predict stops within `early_bars` candles?
    For each possible veto threshold: how many early stops it would have caught,
    how many winners it would have blocked, and the net P&L per trade before and after.
    Report only; nothing here changes an alert."""
    pts: List[Tuple[int, int, bool, float, bool]] = []
    for r in rows:
        ctx = r.get("context") if isinstance(r.get("context"), dict) else {}
        sc, ct = ctx.get("exh_score"), ctx.get("exh_cats")
        if isinstance(sc, bool) or not isinstance(sc, (int, float)):
            continue
        o = _outcome(r, cost_pct)
        if o is None:
            continue
        sl_c = r.get("sl_candle")
        early = (r.get("outcome_class") == "sl" and isinstance(sl_c, (int, float))
                 and not isinstance(sl_c, bool) and int(sl_c) < early_bars)
        pts.append((int(sc), int(ct) if isinstance(ct, (int, float)) else 0, bool(early), o[0], o[1]))
    n = len(pts)
    out: Dict[str, Any] = {"n": n, "early_bars": early_bars, "status": "COLLECTING" if n < min_n else "OK",
                           "bands": [], "cuts": []}
    if n == 0:
        return out
    n_early = sum(1 for p in pts if p[2])
    n_win = sum(1 for p in pts if p[4])
    out["early_rate"] = round(n_early / n, 3)
    mean_all = sum(p[3] for p in pts) / n
    for label, lo, hi in (("0-1", 0, 1), ("2", 2, 2), ("3", 3, 3), ("4+", 4, 6)):
        g = [p for p in pts if lo <= p[0] <= hi]
        if g:
            out["bands"].append({"band": label, "n": len(g),
                                 "early": round(sum(1 for p in g if p[2]) / len(g), 3),
                                 "success": round(sum(1 for p in g if p[4]) / len(g), 3),
                                 "mean_net": round(sum(p[3] for p in g) / len(g), 3)})
    for t in veto_scores:
        blocked = [p for p in pts if p[0] >= t and p[1] >= min_cats]
        kept = [p for p in pts if not (p[0] >= t and p[1] >= min_cats)]
        out["cuts"].append({
            "score": t, "blocked": len(blocked), "kept_share": round(len(kept) / n, 3),
            "early_caught": None if not n_early else round(sum(1 for p in blocked if p[2]) / n_early, 3),
            "winners_lost": None if not n_win else round(sum(1 for p in blocked if p[4]) / n_win, 3),
            "net_before": round(mean_all, 3),
            "net_after": None if not kept else round(sum(p[3] for p in kept) / len(kept), 3),
        })
    return out

def _pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.0%}"

def format_intelligence(blob: Mapping[str, Any], name_fn: Callable[[str], str] = str, max_chars: int = 3400) -> str:
    q = blob.get("quality") or {}
    lines: List[str] = []
    lines.append(f"🔬 BRAIN AUTOPSY | Recorded {q.get('live_rows', 0)} | Shadowed {q.get('shadow_rows', 0)} rows "
                 f"-> {q.get('shadow_independent', 0)} independent")
    if q.get("ambiguous_live") is not None or q.get("ambiguous_shadow") is not None:
        lines.append(f"Same-candle stop/target ties (graded as stop): Recorded {_pct(q.get('ambiguous_live'))}, "
                     f"Shadowed {_pct(q.get('ambiguous_shadow'))}")
    tm = blob.get("trust") or {}
    lines.append("")
    lines.append(f"🎯 TRUST METER | level L{tm.get('level', 0)} {tm.get('status', 'COLLECTING')} "
                 f"| verdicts stored on {tm.get('n_with_verdict', 0)} resolved trades")
    for v in (TAKE, WATCH, AVOID):
        d = (tm.get("verdicts") or {}).get(v)
        if d:
            nl = "n/a" if d.get("net_lb") is None else f"{d['net_lb']:+.2f}"
            lines.append(f"{v}: n={d['n']} ({d['coverage']:.0%} of verdicts) | precision {d['precision']:.0%} "
                         f"[{d['wilson'][0]:.0%}-{d['wilson'][1]:.0%}] | net {d['mean_net']:+.2f}%/trade (lower bound {nl})")
    if tm.get("verdicts"):
        lines.append(f"Break-even precision {tm['breakeven_precision']:.0%} | target {tm['target']:.0%}"
                     + (f" | meets target: {'yes' if tm.get('meets_target') else 'not proven'}" if "meets_target" in tm else "")
                     + (f" | conviction ranking AUC {tm['auc']:.2f}" if tm.get("auc") is not None else ""))
    if tm.get("bands"):
        lines.append("Conviction vs reality: " + "; ".join(
            f"{b['band']}% -> {b['precision']:.0%} (n={b['n']})" for b in tm["bands"]))
    if tm.get("windows"):
        lines.append("TAKE stability by period: " + " | ".join(
            f"{w['precision']:.0%}{'✓' if w['ok'] else '✗'}" for w in tm["windows"]))
    if tm.get("note"):
        lines.append(tm["note"])
    for name in ("LIVE", "SHADOW"):
        pop = (blob.get("pops") or {}).get(name) or {}
        em = pop.get("edges") or {}
        lines.append("")
        lines.append(f"{'Recorded' if name == 'LIVE' else 'Shadowed'} conditions (n={em.get('n', 0)}, "
                     f"base success {_pct(em.get('base_rate'))}):")
        if em.get("status") != "OK":
            lines.append("  collecting evidence")
            continue
        for tag, key in (("helps", "helps"), ("hurts", "hurts")):
            for c in em.get(key, [])[:3]:
                lines.append(f"  {tag}: {c['text']} n={c['n']} {c['rate']:.0%} vs {c['base']:.0%} "
                             f"{c['mean_net']:+.2f}%/trade [L{c['level']}]")
    rj = blob.get("rejections") or {}
    shown = [x for x in rj.get("reasons", []) if x["status"] != "COLLECTING"][:4]
    lines.append("")
    lines.append(f"🛡 SHADOW REJECTIONS (Recorded success {_pct(rj.get('live_success'))}):")
    if not shown:
        lines.append("  collecting evidence")
    for x in shown:
        lines.append(f"  {x['reason']}: n={x['n']} success {x['success']:.0%} net {x['mean_net']:+.2f}% -> {x['status']}")

    ex = blob.get("exhaustion") or {}
    if ex.get("n"):
        lines.append("")
        lines.append(f"⏳ LATE-TREND EXHAUSTION (n={ex['n']}, stopped within {ex['early_bars']} candles: "
                     f"{_pct(ex.get('early_rate'))}) {ex.get('status', '')}")
        for b in ex.get("bands", []):
            lines.append(f"  score {b['band']}: n={b['n']} early-stop {b['early']:.0%} success {b['success']:.0%} "
                         f"net {b['mean_net']:+.2f}%")
        if ex.get("status") == "OK":
            for c in ex.get("cuts", []):
                if c["blocked"]:
                    ec = "n/a" if c["early_caught"] is None else f"{c['early_caught']:.0%}"
                    wl = "n/a" if c["winners_lost"] is None else f"{c['winners_lost']:.0%}"
                    na = "n/a" if c["net_after"] is None else f"{c['net_after']:+.2f}"
                    lines.append(f"  veto at >={c['score']}: blocks {c['blocked']} | catches {ec} of early stops | "
                                 f"loses {wl} of winners | keeps {c['kept_share']:.0%} | net {c['net_before']:+.2f} -> {na}%")
    last = ((((blob.get("pops") or {}).get("LIVE") or {}).get("autopsy") or {}).get("autopsies") or [])[-3:]
    if last:
        lines.append("")
        lines.append("Latest Recorded autopsies:")
        for a in last:
            hurt = "; ".join(h[0] for h in a["hurt"][:2]) or "no clear weakness"
            helped = "; ".join(h[0] for h in a["helped"][:2]) or "no clear strength"
            lines.append(f"  {a['pair']} {name_fn(str(a['alert_key']))} {a['dir']} {a['result']} {a['pnl']:+.2f}% ({a['mode_text']}) | "
                         f"prior {a['p_prior']:.0%} | helped: {helped} | hurt: {hurt}")
    lines.append("")
    lines.append("Report only. L1 = exploratory, L2 = survives false-discovery control and both time halves. Nothing here changes a verdict.")
    text = "\n".join(lines)
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."
