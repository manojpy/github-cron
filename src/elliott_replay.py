#!/usr/bin/env python3
"""elliott_replay.py — validation tooling for the Elliott Wave module (not a test).

Two modes, one statistics layer:

  replay   Point-in-time replay on historical 15m candles of several pairs. For every
           candle `end` the engine sees ONLY candles 0..end (exactly what a live run
           would have seen), every triggered setup and every exhaustion event is
           followed forward with the same structural stop / invalidation / targets the
           live tracker uses, and the result is summarised.
  stats    Aggregates what the live bot has already archived: the append-only
           `elliott_event` rows in <data-dir>/shadow/*.jsonl[.gz] (signal rows joined to
           their resolution rows).

What is measured per (family, group, score bucket): number of structures, share whose
FIRST target came before the structural stop / invalidation, share reaching the second
target, share that ended adverse, share that expired, median bars to first target,
median MFE / MAE in ATR and %, how often invalidation was hit and how often volume
expanded after the signal. In replay mode `--baseline` also follows random entries with
the SAME percentage stop and target distances in the same direction, so the table shows
whether the wave structure adds anything over just trading the same bracket at random
(the "edge" column is the structure's first-target-first rate minus the random one).

Limits you should know before trusting a number:
  * The replay does NOT apply the legacy common-relaxed / wick gate: that gate needs the
    bot's full indicator pipeline. Replay therefore measures the wave structure alone;
    use `stats` mode (groups delivered / shadow / gated) for the effect of the gate.
  * Candles are used as given. Wide candles that touch both a target and the stop are
    resolved adverse-first (pessimistic), like the bot's own outcome replay.
  * Small samples are noise. The report prints n next to every rate; treat n < 30 as
    anecdote.

Usage
  python3 scripts/elliott_replay.py replay --csv data/BTCUSD.csv data/ETHUSD.csv
  python3 scripts/elliott_replay.py replay --fetch MARK:BTCUSD MARK:ETHUSD --days 30
  python3 scripts/elliott_replay.py replay --csv ... --baseline --lookahead-check 40
  python3 scripts/elliott_replay.py stats --data-dir outcome-data --days 30

CSV columns (header required, any order, case-insensitive): time|timestamp|ts|t (seconds,
or milliseconds), open, high, low, close, volume. The pair name is the file name.
Needs only numpy (and the repo's src/elliott_wave.py); --fetch also needs network.
"""
from __future__ import annotations

import argparse
import csv
import glob
import gzip
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import elliott_wave as ew  # noqa: E402

INTERVAL = 900
VOL_REF_BARS = 20
BUCKETS = ((0, 70, "<70"), (70, 80, "70-79"), (80, 90, "80-89"), (90, 101, "90+"))


# ── data loading ─────────────────────────────────────────────────────────

def _norm_ts(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.int64)
    return a // 1000 if len(a) and a[0] > 10**11 else a


def load_csv(path: str) -> Tuple[np.ndarray, ...]:
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        cols = {(k or "").strip().lower(): k for k in (rd.fieldnames or [])}
        tcol = next((cols[k] for k in ("time", "timestamp", "ts", "t") if k in cols), None)
        need = [cols.get(k) for k in ("open", "high", "low", "close", "volume")]
        if tcol is None or any(x is None for x in need):
            raise SystemExit(f"{path}: needs columns time|timestamp|ts|t, open, high, low, close, volume")
        rows = [[float(r[tcol])] + [float(r[c]) for c in need] for r in rd]
    arr = np.asarray(rows, dtype=np.float64)
    arr = arr[np.argsort(arr[:, 0], kind="stable")]
    ts = _norm_ts(arr[:, 0].astype(np.int64))
    return ts, arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4], arr[:, 5]


def fetch_pair(symbol: str, days: int, api_base: str, resolution: str = "15m") -> Tuple[np.ndarray, ...]:
    """Delta chart-history endpoint, same one the bot uses; paged in 1500-candle blocks."""
    end = int(time.time())
    start = end - days * 86400
    out: Dict[int, Tuple[float, ...]] = {}
    block = 1500 * INTERVAL
    t0 = start
    while t0 < end:
        t1 = min(end, t0 + block)
        q = urllib.parse.urlencode({"resolution": resolution, "symbol": symbol, "from": t0, "to": t1})
        with urllib.request.urlopen(f"{api_base}/v2/chart/history?{q}", timeout=30) as r:
            res = json.loads(r.read().decode()).get("result", {}) or {}
        for t, o, h, l, c, v in zip(res.get("t", []), res.get("o", []), res.get("h", []),
                                    res.get("l", []), res.get("c", []), res.get("v", [])):
            out[int(t)] = (float(o), float(h), float(l), float(c), float(v))
        t0 = t1
        time.sleep(0.25)
    keys = sorted(out)
    a = np.asarray([out[k] for k in keys], dtype=np.float64).reshape(-1, 5)
    return (np.asarray(keys, dtype=np.int64), a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4])


# ── replay ───────────────────────────────────────────────────────────────

def _vol_ref(v: np.ndarray, idx: int) -> Optional[float]:
    w = v[max(0, idx - VOL_REF_BARS + 1):idx + 1]
    m = float(np.mean(w)) if len(w) else 0.0
    return m if m > 0 and np.all(np.isfinite(w)) else None


def replay_pair(pair: str, arr: Tuple[np.ndarray, ...], params: "ew.ElliottParams",
                start: int) -> List[Dict[str, Any]]:
    """First decision per structure, as a live run would have seen it. For a structure
    that later becomes eligible the eligible decision is kept (that is when the bot
    would deliver). Exhaustion events are included as family "exhaustion"."""
    ts, o, h, l, c, v = arr
    first: Dict[str, Tuple[Any, bool]] = {}
    ev_seen: Dict[str, Any] = {}
    for end in range(start, len(ts)):
        a = ew.analyze(pair, ts, o, h, l, c, v, end=end, params=params)    # reads 0..end only
        for s in a.setups:
            cur = first.get(s.structure_id)
            if cur is None or (cur[0].status != "eligible" and s.status == "eligible"):
                first[s.structure_id] = (s, s is a.primary)
        for e in a.events:
            ev_seen.setdefault(e.structure_id, e)
    recs: List[Dict[str, Any]] = []
    for s, prim in first.values():
        recs.append({"pair": pair, "family": s.family, "direction": s.direction, "group": s.status,
                     "status": s.status, "score": float(s.score), "signal_ts": int(s.signal_ts),
                     "signal_idx": int(s.signal_idx), "entry": s.entry_price, "atr": s.atr,
                     "invalidation": s.invalidation, "stop": s.stop, "targets": list(s.targets),
                     "primary": prim, "reason": s.reason,
                     "vol_ref": _vol_ref(v, int(s.signal_idx))})
    for e in ev_seen.values():
        recs.append({"pair": pair, "family": "exhaustion", "direction": e.direction, "group": "event",
                     "status": "event", "score": float(e.score), "signal_ts": int(e.signal_ts),
                     "signal_idx": int(e.signal_idx), "entry": e.entry_price, "atr": e.atr,
                     "invalidation": e.invalidation, "stop": None, "targets": list(e.targets),
                     "primary": False, "reason": "", "vol_ref": _vol_ref(v, int(e.signal_idx))})
    return recs


def follow(rec: Dict[str, Any], arr: Tuple[np.ndarray, ...], max_bars: int) -> Optional[Dict[str, Any]]:
    """Forward result in the common row shape, or None when the future is too short
    to decide (an unfinished structure must not count as a result)."""
    ts, o, h, l, c, v = arr
    p = ew.evaluate_progress(rec, ts, h, l, c, v, end=len(ts) - 1, max_bars=max_bars)
    if p.get("state") != "final":
        return None
    fb = p.get("first_bars") or {}
    return {**{k: rec[k] for k in ("pair", "family", "direction", "group", "status", "score")},
            "outcome": p["outcome"], "t1_first": bool(p.get("t1_before_adverse")),
            "bars_t1": fb.get("t1"), "invalidated": fb.get("invalidated") is not None,
            "mfe_atr": p.get("mfe_atr"), "mae_atr": p.get("mae_atr"),
            "mfe_pct": p.get("mfe_pct"), "mae_pct": p.get("mae_pct"),
            "vol_expanded": p.get("volume_expanded")}


def baseline_rows(rec: Dict[str, Any], arr: Tuple[np.ndarray, ...], max_bars: int,
                  rng: np.random.Generator, n: int, lo: int) -> List[Dict[str, Any]]:
    """Random entries on the same pair, same direction, same PERCENT stop / target
    distances as the setup. The yardstick for 'does the structure add anything'."""
    ts, o, h, l, c, v = arr
    entry = float(rec["entry"])
    if entry <= 0 or rec["family"] == "exhaustion" and rec["stop"] is None and not rec["targets"]:
        return []
    buy = rec["direction"] == "buy"
    inv_pct = abs(entry - float(rec["invalidation"])) / entry
    stop_pct = abs(entry - float(rec["stop"])) / entry if rec.get("stop") is not None else None
    t_pct = [abs(float(t) - entry) / entry for t in rec["targets"]]
    out: List[Dict[str, Any]] = []
    hi = len(ts) - max_bars - 2
    if hi <= lo:
        return out
    for j in rng.integers(lo, hi, size=n):
        e0 = float(c[j])
        sgn = 1.0 if buy else -1.0
        b = {"signal_ts": int(ts[j]), "direction": rec["direction"], "entry": e0,
             "atr": float(rec["atr"]) * e0 / entry,
             "invalidation": e0 - sgn * inv_pct * e0,
             "stop": None if stop_pct is None else e0 - sgn * stop_pct * e0,
             "targets": [e0 + sgn * x * e0 for x in t_pct], "vol_ref": _vol_ref(v, int(j))}
        p = ew.evaluate_progress(b, ts, h, l, c, v, end=len(ts) - 1, max_bars=max_bars)
        if p.get("state") != "final":
            continue
        out.append({"family": rec["family"], "group": rec["group"], "t1_first": bool(p.get("t1_before_adverse")),
                    "outcome": p["outcome"]})
    return out


def lookahead_check(pair: str, arr: Tuple[np.ndarray, ...], params: "ew.ElliottParams",
                    start: int, samples: int) -> Tuple[int, int]:
    """analyze(full arrays, end=e) must equal analyze(arrays cut at e): same ids and scores."""
    ts, o, h, l, c, v = arr
    rng = np.random.default_rng(7)
    ends = rng.integers(start, len(ts), size=min(samples, len(ts) - start))
    bad = 0
    for e in ends:
        e = int(e)
        a = ew.analyze(pair, ts, o, h, l, c, v, end=e, params=params)
        b = ew.analyze(pair, ts[:e + 1], o[:e + 1], h[:e + 1], l[:e + 1], c[:e + 1], v[:e + 1],
                       end=e, params=params)
        ka = [(s.event_id, round(s.score, 6), s.status) for s in a.setups]
        kb = [(s.event_id, round(s.score, 6), s.status) for s in b.setups]
        bad += int(ka != kb or [x.event_id for x in a.events] != [x.event_id for x in b.events])
    return len(ends), bad


# ── statistics ───────────────────────────────────────────────────────────

def bucket_of(score: Optional[float]) -> str:
    for lo, hi, name in BUCKETS:
        if score is not None and lo <= score < hi:
            return name
    return "n/a"


def _med(xs: Iterable[Optional[float]]) -> Optional[float]:
    a = [float(x) for x in xs if x is not None]
    return float(np.median(a)) if a else None


def _f(x: Optional[float], nd: int = 2) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def summarize(rows: List[Dict[str, Any]], key=lambda r: (r["family"], r["group"], bucket_of(r.get("score"))),
              base: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    groups: Dict[Tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    bgroups: Dict[Tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in (base or []):
        bgroups[(r["family"], r["group"])].append(r)
    out = []
    for k in sorted(groups):
        g = groups[k]
        n = len(g)
        t1 = sum(1 for r in g if r["t1_first"])
        row = {
            "key": k, "n": n,
            "t1_first": t1 / n,
            "t2": sum(1 for r in g if r["outcome"] == "t2") / n,
            "adverse": sum(1 for r in g if r["outcome"] in ("stop", "invalidated")) / n,
            "expired": sum(1 for r in g if r["outcome"] == "expired") / n,
            "invalidated": sum(1 for r in g if r.get("invalidated")) / n,
            "bars_t1": _med(r.get("bars_t1") for r in g),
            "mfe_atr": _med(r.get("mfe_atr") for r in g), "mae_atr": _med(r.get("mae_atr") for r in g),
            "mfe_pct": _med(r.get("mfe_pct") for r in g), "mae_pct": _med(r.get("mae_pct") for r in g),
        }
        vx = [r["vol_expanded"] for r in g if r.get("vol_expanded") is not None]
        row["vol_expanded"] = (sum(1 for x in vx if x) / len(vx)) if vx else None
        bl = bgroups.get((k[0], k[1]))
        if bl:
            row["base_n"] = len(bl)
            row["base_t1_first"] = sum(1 for r in bl if r["t1_first"]) / len(bl)
            row["edge"] = row["t1_first"] - row["base_t1_first"]
        out.append(row)
    return out


def render(title: str, rows: List[Dict[str, Any]], with_edge: bool) -> str:
    hdr = ["family", "group", "score", "n", "T1 first", "T2", "adverse", "expired", "inval.",
           "bars→T1", "MFE ATR", "MAE ATR", "MFE %", "MAE %", "vol↑"]
    if with_edge:
        hdr += ["random T1 first", "edge"]
    lines = [f"### {title}", "", "| " + " | ".join(hdr) + " |", "|" + "|".join("---" for _ in hdr) + "|"]
    for r in rows:
        f, g, b = r["key"]
        cells = [f, g, b, str(r["n"]), f"{r['t1_first']:.0%}", f"{r['t2']:.0%}", f"{r['adverse']:.0%}",
                 f"{r['expired']:.0%}", f"{r['invalidated']:.0%}", _f(r["bars_t1"], 0), _f(r["mfe_atr"]),
                 _f(r["mae_atr"]), _f(r["mfe_pct"]), _f(r["mae_pct"]),
                 "-" if r["vol_expanded"] is None else f"{r['vol_expanded']:.0%}"]
        if with_edge:
            cells += [("-" if "edge" not in r else f"{r['base_t1_first']:.0%} (n={r['base_n']})"),
                      ("-" if "edge" not in r else f"{r['edge']:+.0%}")]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ── stats mode: archived tracker rows ────────────────────────────────────

def _read_rows(path: str) -> Iterable[Dict[str, Any]]:
    op = gzip.open if path.endswith(".gz") else open
    try:
        with op(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(d, dict) and d.get("record_type") == "elliott_event":
                        yield d
    except OSError:
        return


def load_archived(data_dir: str, days: int) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    cutoff = time.time() - days * 86400
    sig: Dict[str, Dict[str, Any]] = {}
    res: Dict[str, Dict[str, Any]] = {}
    for p in sorted(glob.glob(os.path.join(data_dir, "shadow", "*.jsonl*"))):
        for d in _read_rows(p):
            if int(d.get("entry_ts") or 0) < cutoff:
                continue
            if d.get("event") == "signal":
                sig.setdefault(d["ref_event_id"], d)
            elif d.get("event") == "resolution":
                res.setdefault(d["ref_event_id"], d)
    rows = []
    for eid, r in res.items():
        s = sig.get(eid)
        if s is None:
            continue
        fb = r.get("first_bars") or {}
        rows.append({"pair": r["pair"], "family": r["family"], "direction": r["direction"],
                     "group": s.get("group") or r.get("group"), "status": s.get("status"),
                     "score": s.get("score"), "outcome": r.get("outcome"),
                     "t1_first": bool(r.get("t1_before_adverse")), "bars_t1": fb.get("t1"),
                     "invalidated": bool(r.get("structural_invalidation_hit")),
                     "mfe_atr": r.get("mfe_atr"), "mae_atr": r.get("mae_atr"),
                     "mfe_pct": r.get("mfe_pct"), "mae_pct": r.get("mae_pct"),
                     "vol_expanded": r.get("volume_expanded")})
    return rows, {"signals": len(sig), "resolved": len(rows), "open": len(sig) - len(rows)}


# ── main ─────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    r = sub.add_parser("replay", help="point-in-time replay on historical candles")
    r.add_argument("--csv", nargs="*", default=[], help="candle CSV files, one per pair")
    r.add_argument("--fetch", nargs="*", default=[], help="symbols to download (needs network)")
    r.add_argument("--days", type=int, default=30)
    r.add_argument("--api-base", default=os.environ.get("DELTA_API_BASE", "https://api.india.delta.exchange"))
    r.add_argument("--start", type=int, default=400, help="first candle index replayed (warm-up before it)")
    r.add_argument("--max-bars", type=int, default=48)
    r.add_argument("--baseline", action="store_true", help="compare with random entries of equal bracket")
    r.add_argument("--baseline-n", type=int, default=5)
    r.add_argument("--lookahead-check", type=int, default=0, metavar="N", help="verify N random candles")
    r.add_argument("--min-score", type=float, default=80.0)
    r.add_argument("--watch-score", type=float, default=70.0)
    r.add_argument("--json", default=None)
    s = sub.add_parser("stats", help="aggregate the live bot's archived tracker rows")
    s.add_argument("--data-dir", default=os.environ.get("OUTCOME_DATA_DIR", "outcome-data"))
    s.add_argument("--days", type=int, default=30)
    s.add_argument("--json", default=None)
    a = ap.parse_args(argv)

    if a.mode == "stats":
        rows, meta = load_archived(a.data_dir, a.days)
        print(f"## Elliott archived statistics (last {a.days} days)\n")
        print(f"signals {meta['signals']} · resolved {meta['resolved']} · still open/unresolved {meta['open']}\n")
        if not rows:
            print("No resolved Elliott structures yet.")
            return 0
        table = summarize(rows)
        print(render("By family / group / score", table, False))
        print("\n" + render("By family / group", summarize(rows, key=lambda x: (x["family"], x["group"], "all")), False))
        if a.json:
            json.dump(table, open(a.json, "w"), indent=1, default=str)
        return 0

    params = ew.ElliottParams(min_score=a.min_score, watch_score=a.watch_score)
    data: Dict[str, Tuple[np.ndarray, ...]] = {}
    for p in a.csv:
        data[os.path.splitext(os.path.basename(p))[0]] = load_csv(p)
    for sym in a.fetch:
        print(f"fetching {sym} ...", file=sys.stderr)
        data[sym] = fetch_pair(sym, a.days, a.api_base)
    if not data:
        raise SystemExit("give --csv files and/or --fetch symbols")

    results: List[Dict[str, Any]] = []
    base: List[Dict[str, Any]] = []
    rng = np.random.default_rng(11)
    for pair, arr in data.items():
        n = len(arr[0])
        t0 = time.time()
        recs = replay_pair(pair, arr, params, a.start)
        done = 0
        for rec in recs:
            res = follow(rec, arr, a.max_bars)
            if res is None:
                continue
            done += 1
            results.append(res)
            if a.baseline:
                base += baseline_rows(rec, arr, a.max_bars, rng, a.baseline_n, a.start)
        print(f"{pair}: {n} candles, {len(recs)} structures, {done} with a decided future "
              f"({time.time() - t0:.0f}s)", file=sys.stderr)
        if a.lookahead_check:
            k, bad = lookahead_check(pair, arr, params, a.start, a.lookahead_check)
            print(f"{pair}: look-ahead check {k} candles, mismatches {bad}", file=sys.stderr)
            if bad:
                print("LOOK-AHEAD DETECTED — results are invalid", file=sys.stderr)
                return 2

    print(f"## Elliott point-in-time replay · {len(data)} pair(s) · wave structure only (no legacy gate)\n")
    if not results:
        print("No structure with a decided future in this data.")
        return 0
    table = summarize(results, base=base if a.baseline else None)
    print(render("By family / status / score", table, a.baseline))
    print("\n" + render("By family / status", summarize(
        results, key=lambda x: (x["family"], x["group"], "all"), base=base if a.baseline else None), a.baseline))
    elig = [x for x in results if x["group"] == "eligible"]
    print(f"\neligible structures: {len(elig)} over {len(data)} pair(s)")
    if a.json:
        json.dump(table, open(a.json, "w"), indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
